//! Resumable queue engine.
//!
//! The engine is the only component that writes run state: it turns an immutable
//! [`QueuePlan`] into persisted jobs, picks work through the resource-aware scheduler,
//! runs each attempt through an injected [`WorkerLauncher`], independently verifies the
//! staged output through an injected [`AttemptVerifier`], publishes it through the
//! content store, and finally commits the `Verified` state together with the artifact —
//! in that order, so no downstream job can ever observe an unverified artifact.

use std::io::Write;
use std::path::{Path, PathBuf};

use thiserror::Error;
use zkie_types::{Digest32, ExecutionBackendId, ProofFlavorId, ResourceCapacity, ResourceRequest};

use crate::{
    worker_spec, AggregationArity, ArtifactMetadata, ArtifactRole, ArtifactStore, AttemptId,
    ContentStore, DbError, FailureCode, FailureKind, FailureRecord, FailureStage, JobEvent, JobId,
    JobKind, JobState, KeyMetadata, KeyRole, KeyStore, MemoryAction, ObjectMetadata, ProtocolError,
    RunDb, RunningAttempt, SchedulableJob, Scheduler, SchedulerClock, SchedulerConfig,
    SchedulerError, StoreError, TrustedKeyIdentity, TrustedProofIdentity, VerifiedArtifact,
    VerifiedKey, WorkerJob, WorkerLauncher, WorkerOutcome, WorkerSpec,
};

/// Staged proof payload produced by a proving attempt.
pub const PROOF_FILE: &str = "proof.bin";
/// Staged key payload produced by a prepare attempt.
pub const KEY_FILE: &str = "key.bin";

/// Identity binding committed for a proof-like artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofBinding {
    pub role: ArtifactRole,
    pub k: u32,
    pub arity: AggregationArity,
    pub artifact_manifest_digest: Digest32,
    pub run_identity_digest: Digest32,
    pub public_statement_digest: Digest32,
    pub circuit_digest: Digest32,
    pub verifying_key_digest: Digest32,
    pub srs_source_digest: Digest32,
    pub shard_identity_digest: Digest32,
    pub witness_artifact_digest: Digest32,
    pub proof_flavor: ProofFlavorId,
    pub execution_backend: ExecutionBackendId,
}

/// Identity binding committed for a prepared key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyBinding {
    pub srs_source_digest: Digest32,
    pub proof_flavor: ProofFlavorId,
    pub circuit_digest: Digest32,
    pub k: u32,
    pub aggregation_arity: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
// The plan is built once and then only read; a boxed variant would add indirection to
// every field access for no measurable gain.
#[allow(clippy::large_enum_variant)]
pub enum JobBinding {
    Proof(ProofBinding),
    Key(KeyBinding),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedJob {
    pub logical_id: String,
    pub kind: JobKind,
    pub depends_on: Vec<String>,
    pub request: ResourceRequest,
    pub job: WorkerJob,
    pub binding: JobBinding,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueuePlan {
    pub run_digest: Digest32,
    pub jobs: Vec<PlannedJob>,
}

/// Independently checks a staged attempt before it may be published as verified.
pub trait AttemptVerifier {
    fn verify(
        &mut self,
        planned: &PlannedJob,
        spec: &WorkerSpec,
        staged: &Path,
    ) -> Result<(), String>;
}

/// Supplies the admission action for a round from the current run attempts.
pub trait QueueMemoryMonitor {
    fn action(&mut self, running: &[RunningAttempt]) -> MemoryAction;
}

/// Admits unconditionally; used when no monitor is wired.
pub struct AlwaysAdmit;

impl QueueMemoryMonitor for AlwaysAdmit {
    fn action(&mut self, _running: &[RunningAttempt]) -> MemoryAction {
        MemoryAction::Admit
    }
}

pub struct QueueRuntime<'a> {
    pub launcher: &'a mut dyn WorkerLauncher,
    pub verifier: &'a mut dyn AttemptVerifier,
    pub monitor: &'a mut dyn QueueMemoryMonitor,
    pub staging_root: PathBuf,
    pub artifact_root: PathBuf,
    pub key_root: PathBuf,
    pub capacity: ResourceCapacity,
    pub max_rounds: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QueueOutcome {
    pub verified: Vec<JobId>,
    pub terminal: Vec<JobId>,
    pub blocked: Vec<JobId>,
    pub rounds: u32,
}

impl QueueOutcome {
    pub fn is_complete(&self) -> bool {
        self.terminal.is_empty() && self.blocked.is_empty()
    }
}

#[derive(Debug, Error)]
pub enum QueueError {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error("queue made no progress after {0} rounds")]
    Stalled(u32),
    #[error("plan job {job} depends on unknown job {predecessor}")]
    UnknownPredecessor { job: String, predecessor: String },
    #[error("attempt {attempt} produced no staged output for {logical_id}")]
    MissingStaged { attempt: String, logical_id: String },
    #[error("attempt {attempt} failed independent verification: {message}")]
    VerificationRejected { attempt: String, message: String },
    #[error("queue I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Drives the plan until nothing is runnable, then reports the settled state.
pub fn run_queue<C: SchedulerClock>(
    db: &mut RunDb,
    plan: &QueuePlan,
    runtime: &mut QueueRuntime<'_>,
    clock: C,
) -> Result<QueueOutcome, QueueError> {
    // An interrupted run resumes: attempts left active by a crash go back to `Requeued`
    // before anything new is picked, and that does not count as a cryptographic failure.
    db.recover_interrupted()?;
    let ids = ensure_planned(db, plan)?;
    let capacity = runtime.capacity.clone();
    let scheduler = Scheduler::new(SchedulerConfig::new(capacity.clone()), clock);
    let mut rounds = 0_u32;
    loop {
        rounds += 1;
        if rounds > runtime.max_rounds {
            return Err(QueueError::Stalled(runtime.max_rounds));
        }
        db.refresh_readiness()?;
        let candidates = ready_candidates(db, plan, &ids)?;
        if candidates.is_empty() {
            break;
        }
        // Attempts run one at a time here, so the running set is empty at decision time.
        let action = runtime.monitor.action(&[]);
        let decision = scheduler.decide_with_memory(db, &candidates, capacity.clone(), action)?;
        if decision.selected.is_empty() {
            break;
        }
        for job_id in decision.selected {
            // `ids` is built in plan order, so the index selects the matching plan entry.
            let index = ids
                .iter()
                .position(|(_, id)| id == &job_id)
                .ok_or_else(|| QueueError::MissingStaged {
                    attempt: String::new(),
                    logical_id: job_id.as_str().to_owned(),
                })?;
            let planned = plan.jobs[index].clone();
            run_one(db, plan, &planned, &job_id, runtime)?;
        }
    }
    let mut outcome = summarize(db)?;
    outcome.rounds = rounds - 1;
    Ok(outcome)
}

/// Creates the persisted jobs and edges once; re-running is a no-op, which is what makes
/// an interrupted queue resumable.
fn ensure_planned(db: &mut RunDb, plan: &QueuePlan) -> Result<Vec<(String, JobId)>, QueueError> {
    let mut ids = Vec::new();
    let mut created = Vec::new();
    for planned in &plan.jobs {
        match existing(db, &planned.logical_id) {
            Some(id) => {
                ids.push((planned.logical_id.clone(), id));
                created.push(false);
            }
            None => {
                let id = db.insert_job(&planned.logical_id, planned.kind)?;
                ids.push((planned.logical_id.clone(), id));
                created.push(true);
            }
        }
    }
    for (index, planned) in plan.jobs.iter().enumerate() {
        if !created[index] {
            continue;
        }
        let successor = ids[index].1.clone();
        for predecessor_name in &planned.depends_on {
            let predecessor = ids
                .iter()
                .find(|(name, _)| name == predecessor_name)
                .map(|(_, id)| id.clone())
                .ok_or_else(|| QueueError::UnknownPredecessor {
                    job: planned.logical_id.clone(),
                    predecessor: predecessor_name.clone(),
                })?;
            db.add_dependency(&predecessor, &successor)?;
        }
    }
    Ok(ids)
}

fn existing(db: &RunDb, logical_id: &str) -> Option<JobId> {
    db.jobs()
        .ok()?
        .into_iter()
        .find(|record| record.logical_job_id == logical_id)
        .map(|record| record.id)
}

fn lookup(ids: &[(String, JobId)], logical_id: &str) -> Result<JobId, QueueError> {
    ids.iter()
        .find(|(name, _)| name == logical_id)
        .map(|(_, id)| id.clone())
        .ok_or_else(|| QueueError::UnknownPredecessor {
            job: logical_id.to_owned(),
            predecessor: logical_id.to_owned(),
        })
}

fn ready_candidates(
    db: &RunDb,
    plan: &QueuePlan,
    ids: &[(String, JobId)],
) -> Result<Vec<SchedulableJob>, QueueError> {
    let mut candidates = Vec::new();
    for planned in &plan.jobs {
        let id = lookup(ids, &planned.logical_id)?;
        let record = db.job(&id)?;
        if record.state == JobState::Ready {
            candidates.push(SchedulableJob::new(id, planned.request.clone(), 0));
        }
    }
    Ok(candidates)
}

fn summarize(db: &RunDb) -> Result<QueueOutcome, QueueError> {
    let mut outcome = QueueOutcome::default();
    for record in db.jobs()? {
        match record.state {
            JobState::Verified => outcome.verified.push(record.id),
            JobState::Blocked => outcome.blocked.push(record.id),
            JobState::VerificationFailed => outcome.terminal.push(record.id),
            _ if record.terminal_failure => outcome.terminal.push(record.id),
            _ => {}
        }
    }
    Ok(outcome)
}

fn run_one(
    db: &mut RunDb,
    plan: &QueuePlan,
    planned: &PlannedJob,
    job_id: &JobId,
    runtime: &mut QueueRuntime<'_>,
) -> Result<(), QueueError> {
    let attempt = db.begin_attempt(job_id, start_event(planned.kind))?;
    let staging = runtime.staging_root.join(attempt.as_str());
    std::fs::create_dir_all(&staging)?;
    let spec = worker_spec(
        plan.run_digest,
        job_id.clone(),
        attempt.clone(),
        planned.job.clone(),
        staging.clone(),
        &planned.request,
    )?;
    let result_path = staging.join("worker-result.txt");
    let result = runtime.launcher.launch(&spec, &result_path)?;
    match result.outcome {
        WorkerOutcome::Succeeded => {
            finish_success(db, planned, job_id, &attempt, &spec, &staging, runtime)
        }
        WorkerOutcome::Failed { .. } => {
            let failure = FailureRecord::new(
                FailureKind::Execution,
                stage_for(planned.kind),
                FailureCode::WorkerExit,
            );
            db.handle_attempt_failure(job_id, &attempt, failure, None)?;
            Ok(())
        }
    }
}

fn start_event(kind: JobKind) -> JobEvent {
    match kind {
        JobKind::Witness => JobEvent::StartWitnessing,
        JobKind::Prepare => JobEvent::StartPreparing,
        JobKind::LeafProof => JobEvent::StartProving,
        JobKind::LeafVerification | JobKind::NativeAggregate => JobEvent::StartVerification,
    }
}

fn stage_for(kind: JobKind) -> FailureStage {
    match kind {
        JobKind::Witness => FailureStage::Witness,
        JobKind::Prepare => FailureStage::Prepare,
        JobKind::LeafProof => FailureStage::Prove,
        JobKind::LeafVerification | JobKind::NativeAggregate => FailureStage::Verify,
    }
}

/// Independent verification happens first, publication second, and only then does the
/// run state move to `Verified` — so a downstream job can never see an unverified object.
fn finish_success(
    db: &mut RunDb,
    planned: &PlannedJob,
    job_id: &JobId,
    attempt: &AttemptId,
    spec: &WorkerSpec,
    staging: &Path,
    runtime: &mut QueueRuntime<'_>,
) -> Result<(), QueueError> {
    runtime
        .verifier
        .verify(planned, spec, staging)
        .map_err(|message| QueueError::VerificationRejected {
            attempt: attempt.as_str().to_owned(),
            message,
        })?;
    match planned.kind {
        JobKind::LeafVerification => {
            db.complete_verification(job_id, attempt, "independent-verifier")?;
            return Ok(());
        }
        JobKind::Witness => {
            db.apply_attempt_event(job_id, attempt, JobEvent::WitnessCompleted)?;
        }
        JobKind::Prepare => {
            db.apply_attempt_event(job_id, attempt, JobEvent::PreparationCompleted)?;
        }
        JobKind::LeafProof => {
            db.apply_attempt_event(job_id, attempt, JobEvent::ProofProduced)?;
        }
        // The aggregate is already `Verifying`; publication performs the transition.
        JobKind::NativeAggregate => {}
    }
    match &planned.binding {
        JobBinding::Proof(binding) => {
            commit_proof(db, planned, job_id, attempt, staging, binding, runtime)
        }
        JobBinding::Key(binding) => {
            commit_key(db, planned, job_id, attempt, staging, binding, runtime)
        }
    }
}

fn read_staged(
    staging: &Path,
    file: &str,
    attempt: &AttemptId,
    logical_id: &str,
) -> Result<Vec<u8>, QueueError> {
    std::fs::read(staging.join(file)).map_err(|_| QueueError::MissingStaged {
        attempt: attempt.as_str().to_owned(),
        logical_id: logical_id.to_owned(),
    })
}

fn content_digest(bytes: &[u8]) -> Digest32 {
    Digest32::new(*blake3::hash(bytes).as_bytes())
}

fn commit_proof(
    db: &mut RunDb,
    planned: &PlannedJob,
    job_id: &JobId,
    attempt: &AttemptId,
    staging: &Path,
    binding: &ProofBinding,
    runtime: &mut QueueRuntime<'_>,
) -> Result<(), QueueError> {
    let bytes = read_staged(staging, PROOF_FILE, attempt, &planned.logical_id)?;
    let digest = content_digest(&bytes);
    let identity = TrustedProofIdentity::new(
        binding.role,
        planned.kind,
        binding.k,
        binding.arity,
        db.run_id(),
        job_id,
        attempt,
        digest,
        binding.artifact_manifest_digest,
        binding.run_identity_digest,
        binding.public_statement_digest,
        binding.circuit_digest,
        binding.verifying_key_digest,
        binding.srs_source_digest,
        binding.shard_identity_digest,
        binding.witness_artifact_digest,
        binding.proof_flavor.clone(),
        binding.execution_backend.clone(),
    )?;
    let metadata = ObjectMetadata::Artifact(ArtifactMetadata::new_proof(
        planned.logical_id.clone(),
        digest,
        bytes.len() as u64,
        identity.digest(),
    )?);
    let store = ArtifactStore::open(&runtime.artifact_root)?;
    let mut staged = store.stage(&metadata)?;
    staged.write_all(&bytes)?;
    let published = store.publish(store.validate(staged)?)?;
    let verified = VerifiedArtifact::attest(identity, published)?;
    db.commit_verified_artifact(&verified)?;
    Ok(())
}

fn commit_key(
    db: &mut RunDb,
    planned: &PlannedJob,
    job_id: &JobId,
    attempt: &AttemptId,
    staging: &Path,
    binding: &KeyBinding,
    runtime: &mut QueueRuntime<'_>,
) -> Result<(), QueueError> {
    let bytes = read_staged(staging, KEY_FILE, attempt, &planned.logical_id)?;
    let digest = content_digest(&bytes);
    let base = KeyMetadata::new(
        digest,
        bytes.len() as u64,
        binding.srs_source_digest,
        binding.proof_flavor.clone(),
        binding.circuit_digest,
        binding.k,
        binding.aggregation_arity,
    )?;
    let identity = TrustedKeyIdentity::new(
        KeyRole::ProvingAndVerifyingKey,
        planned.kind,
        db.run_id(),
        job_id,
        attempt,
        base.clone(),
    )?;
    let metadata = ObjectMetadata::Key(base.bind_identity(identity.digest()));
    let store = KeyStore::open(&runtime.key_root)?;
    let mut staged = store.stage(&metadata)?;
    staged.write_all(&bytes)?;
    let published = store.publish(store.validate(staged)?)?;
    let verified = VerifiedKey::attest(identity, published)?;
    db.commit_verified_key(&verified)?;
    Ok(())
}
