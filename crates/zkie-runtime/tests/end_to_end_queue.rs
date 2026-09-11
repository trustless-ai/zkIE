use std::fs;
use std::path::{Path, PathBuf};

use zkie_runtime::{
    run_queue, AggregationArity, AlwaysAdmit, ArtifactRole, AttemptVerifier, JobBinding, JobKind,
    JobState, KeyBinding, PlannedJob, ProofBinding, QueueError, QueuePlan, QueueRuntime, RunDb,
    SchedulerClock, WorkerJob, WorkerJobKind, WorkerLauncher, WorkerMeasurements, WorkerResult,
    WorkerSpec,
};
use zkie_types::{Digest32, ExecutionBackendId, ProofFlavorId, ResourceCapacity, ResourceRequest};

const GIB: u64 = 1_073_741_824;

fn digest(seed: u8) -> Digest32 {
    Digest32::new([seed; 32])
}

fn script_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-queue-{label}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn request(cpu: u32, gib: u64) -> ResourceRequest {
    ResourceRequest::new(cpu, gib * GIB, 0, 0).unwrap()
}

fn capacity(cpu: u32, gib: u64) -> ResourceCapacity {
    ResourceCapacity::new(cpu, gib * GIB, 0, 0).unwrap()
}

fn worker_job(kind: WorkerJobKind, logical_id: &str) -> WorkerJob {
    WorkerJob::new(kind, [("logical_id".to_owned(), logical_id.to_owned())]).unwrap()
}

fn proof_binding(kind: WorkerJobKind) -> JobBinding {
    let role = match kind {
        WorkerJobKind::Witness => ArtifactRole::Witness,
        WorkerJobKind::LeafProof => ArtifactRole::LeafProof,
        WorkerJobKind::NativeAggregate => ArtifactRole::NativeVerifiedManifest,
        WorkerJobKind::Prepare | WorkerJobKind::LeafVerification => ArtifactRole::LeafProof,
    };
    JobBinding::Proof(ProofBinding {
        role,
        k: 19,
        arity: if kind == WorkerJobKind::NativeAggregate {
            AggregationArity::Actual(3)
        } else {
            AggregationArity::NotApplicable
        },
        artifact_manifest_digest: digest(11),
        run_identity_digest: digest(12),
        public_statement_digest: digest(13),
        circuit_digest: digest(14),
        verifying_key_digest: digest(15),
        srs_source_digest: digest(16),
        shard_identity_digest: digest(17),
        witness_artifact_digest: digest(18),
        proof_flavor: ProofFlavorId::parse("halo2-kzg-v1").unwrap(),
        execution_backend: ExecutionBackendId::parse("halo2-cpu").unwrap(),
    })
}

fn planned(
    logical_id: &str,
    kind: JobKind,
    worker: WorkerJobKind,
    depends_on: &[&str],
) -> PlannedJob {
    PlannedJob {
        logical_id: logical_id.to_owned(),
        kind,
        depends_on: depends_on.iter().map(|name| (*name).to_owned()).collect(),
        request: request(2, 8),
        job: worker_job(worker, logical_id),
        binding: proof_binding(worker),
    }
}

/// Three proof shards feeding one aggregate through a broadcast edge.
fn broadcast_plan() -> QueuePlan {
    QueuePlan {
        run_digest: digest(1),
        jobs: vec![
            planned("shard-0", JobKind::LeafProof, WorkerJobKind::LeafProof, &[]),
            planned("shard-1", JobKind::LeafProof, WorkerJobKind::LeafProof, &[]),
            planned("shard-2", JobKind::LeafProof, WorkerJobKind::LeafProof, &[]),
            planned(
                "aggregate",
                JobKind::NativeAggregate,
                WorkerJobKind::NativeAggregate,
                &["shard-0", "shard-1", "shard-2"],
            ),
        ],
    }
}

/// A prepare job exists only to exercise the key-identity publication path.
fn prepare_plan() -> QueuePlan {
    QueuePlan {
        run_digest: digest(2),
        jobs: vec![PlannedJob {
            logical_id: "prepare".to_owned(),
            kind: JobKind::Prepare,
            depends_on: Vec::new(),
            request: request(1, 2),
            job: worker_job(WorkerJobKind::Prepare, "prepare"),
            binding: JobBinding::Key(KeyBinding {
                srs_source_digest: digest(21),
                proof_flavor: ProofFlavorId::parse("halo2-kzg-v1").unwrap(),
                circuit_digest: digest(22),
                k: 19,
                aggregation_arity: 3,
            }),
        }],
    }
}

#[derive(Default)]
struct RecordingLauncher {
    order: Vec<String>,
    failures_remaining: usize,
    always_fail_logical: Option<String>,
    writes: Vec<(String, Vec<u8>)>,
}

impl WorkerLauncher for RecordingLauncher {
    fn launch(
        &mut self,
        spec: &WorkerSpec,
        _result_path: &Path,
    ) -> Result<WorkerResult, zkie_runtime::ProtocolError> {
        let name = spec
            .job_id
            .as_str()
            .rsplit(':')
            .next()
            .unwrap_or_default()
            .to_owned();
        self.order.push(name.clone());
        let budgeted = self.failures_remaining > 0;
        if budgeted {
            self.failures_remaining -= 1;
        }
        let targeted = self.always_fail_logical.as_deref() == Some(name.as_str());
        if budgeted || targeted {
            return Ok(WorkerResult::failed(
                spec,
                "worker-exit",
                WorkerMeasurements::default(),
            ));
        }
        let file = if spec.job.kind == WorkerJobKind::Prepare {
            "key.bin"
        } else {
            "proof.bin"
        };
        let payload = format!("{}:{}", spec.job_id, spec.attempt_id).into_bytes();
        fs::write(spec.staged_output_dir.join(file), &payload).unwrap();
        self.writes.push((file.to_owned(), payload));
        Ok(WorkerResult::succeeded(
            spec,
            WorkerMeasurements {
                peak_resident_bytes: 4 * GIB,
                wall_millis: 5,
            },
        ))
    }
}

#[derive(Default)]
struct AcceptingVerifier {
    reject: Option<String>,
}

impl AttemptVerifier for AcceptingVerifier {
    fn verify(
        &mut self,
        _planned: &PlannedJob,
        _spec: &WorkerSpec,
        _staged: &Path,
    ) -> Result<(), String> {
        match &self.reject {
            Some(message) => Err(message.clone()),
            None => Ok(()),
        }
    }
}

struct FixedClock;

impl SchedulerClock for FixedClock {
    fn now_unix_seconds(&self) -> i64 {
        1_000
    }
}

struct Harness {
    root: PathBuf,
    staging: PathBuf,
    artifacts: PathBuf,
    keys: PathBuf,
    db_path: PathBuf,
}

impl Harness {
    fn new(label: &str) -> Self {
        let root = script_dir(label);
        let staging = root.join("staging");
        let artifacts = root.join("artifacts");
        let keys = root.join("keys");
        for directory in [&staging, &artifacts, &keys] {
            fs::create_dir_all(directory).unwrap();
        }
        Self {
            db_path: root.join("run.sqlite"),
            root,
            staging,
            artifacts,
            keys,
        }
    }

    fn open(&self, label: &str) -> RunDb {
        RunDb::open(&self.db_path, label).unwrap()
    }

    /// Runs the queue once with the supplied launcher and verifier.
    fn run(
        &self,
        db: &mut RunDb,
        plan: &QueuePlan,
        launcher: &mut RecordingLauncher,
        verifier: &mut AcceptingVerifier,
        capacity: ResourceCapacity,
    ) -> Result<zkie_runtime::QueueOutcome, QueueError> {
        let mut monitor = AlwaysAdmit;
        let mut runtime = QueueRuntime {
            launcher,
            verifier,
            monitor: &mut monitor,
            staging_root: self.staging.clone(),
            artifact_root: self.artifacts.clone(),
            key_root: self.keys.clone(),
            capacity,
            max_rounds: 64,
        };
        run_queue(db, plan, &mut runtime, FixedClock)
    }

    fn cleanup(&self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn logical(job: &zkie_runtime::JobId) -> String {
    job.as_str()
        .split_once(':')
        .map(|(_, rest)| rest.to_owned())
        .unwrap_or_else(|| job.as_str().to_owned())
}

fn record(db: &RunDb, name: &str) -> zkie_runtime::JobRecord {
    db.jobs()
        .unwrap()
        .into_iter()
        .find(|record| record.logical_job_id == name)
        .unwrap_or_else(|| panic!("no job named {name}"))
}

#[test]
fn three_shard_broadcast_dag_runs_in_two_waves_and_settles_verified() {
    let harness = Harness::new("two-waves");
    let plan = broadcast_plan();
    let mut db = harness.open("queue-two-waves");
    let mut launcher = RecordingLauncher::default();
    let mut verifier = AcceptingVerifier::default();

    let outcome = harness
        .run(
            &mut db,
            &plan,
            &mut launcher,
            &mut verifier,
            capacity(4, 16),
        )
        .unwrap();

    // Two 2-core / 8 GiB proof jobs fit at once, so the third lands in wave two and the
    // aggregate can only start once every shard is Verified.
    assert_eq!(
        launcher.order,
        vec!["shard-0", "shard-1", "shard-2", "aggregate"]
    );
    assert_eq!(outcome.rounds, 3);
    assert_eq!(outcome.verified.len(), 4);
    assert!(outcome.is_complete());
    for name in ["shard-0", "shard-1", "shard-2", "aggregate"] {
        assert_eq!(record(&db, name).state, JobState::Verified);
    }
    assert_eq!(logical(&record(&db, "aggregate").id), "aggregate");
    harness.cleanup();
}

#[test]
fn a_tight_cpu_budget_serialises_the_shards_without_exceeding_it() {
    let harness = Harness::new("tight");
    let plan = broadcast_plan();
    let mut db = harness.open("queue-tight");
    let mut launcher = RecordingLauncher::default();
    let mut verifier = AcceptingVerifier::default();

    // 3 cores cannot hold two 2-core jobs, so every job gets its own wave.
    let outcome = harness
        .run(
            &mut db,
            &plan,
            &mut launcher,
            &mut verifier,
            capacity(3, 64),
        )
        .unwrap();

    assert_eq!(
        launcher.order,
        vec!["shard-0", "shard-1", "shard-2", "aggregate"]
    );
    assert_eq!(outcome.rounds, 4);
    assert!(outcome.is_complete());
    harness.cleanup();
}

#[test]
fn a_prepare_job_publishes_a_key_and_reaches_verified() {
    let harness = Harness::new("prepare");
    let plan = prepare_plan();
    let mut db = harness.open("queue-prepare");
    let mut launcher = RecordingLauncher::default();
    let mut verifier = AcceptingVerifier::default();

    let outcome = harness
        .run(
            &mut db,
            &plan,
            &mut launcher,
            &mut verifier,
            capacity(4, 16),
        )
        .unwrap();

    assert!(outcome.is_complete());
    assert_eq!(outcome.verified.len(), 1);
    assert_eq!(record(&db, "prepare").state, JobState::Verified);
    assert_eq!(launcher.writes.len(), 1);
    assert_eq!(launcher.writes[0].0, "key.bin");
    harness.cleanup();
}

#[test]
fn a_rejected_verification_never_reaches_verified_and_the_run_resumes() {
    let harness = Harness::new("rejected");
    let plan = prepare_plan();
    let mut db = harness.open("queue-rejected");
    let mut launcher = RecordingLauncher::default();
    let mut rejecting = AcceptingVerifier {
        reject: Some("proof does not verify".to_owned()),
    };

    let error = harness
        .run(
            &mut db,
            &plan,
            &mut launcher,
            &mut rejecting,
            capacity(4, 16),
        )
        .unwrap_err();
    assert!(matches!(error, QueueError::VerificationRejected { .. }));
    assert_ne!(record(&db, "prepare").state, JobState::Verified);

    // The rejected attempt is still active; resuming recovers it and finishes the run.
    let mut accepting = AcceptingVerifier::default();
    let outcome = harness
        .run(
            &mut db,
            &plan,
            &mut launcher,
            &mut accepting,
            capacity(4, 16),
        )
        .unwrap();
    assert!(outcome.is_complete());
    assert_eq!(record(&db, "prepare").state, JobState::Verified);
    harness.cleanup();
}

#[test]
fn a_terminal_failure_blocks_every_dependent() {
    let harness = Harness::new("blocked");
    let plan = broadcast_plan();
    let mut db = harness.open("queue-blocked");
    let mut launcher = RecordingLauncher {
        always_fail_logical: Some("shard-0".to_owned()),
        ..RecordingLauncher::default()
    };
    let mut verifier = AcceptingVerifier::default();

    let outcome = harness
        .run(
            &mut db,
            &plan,
            &mut launcher,
            &mut verifier,
            capacity(4, 16),
        )
        .unwrap();

    assert_eq!(record(&db, "shard-1").state, JobState::Verified);
    assert_eq!(record(&db, "shard-2").state, JobState::Verified);

    let failed = record(&db, "shard-0");
    assert!(failed.terminal_failure);
    assert_eq!(failed.execution_retries, 2);
    assert_eq!(failed.state, JobState::ExecutionFailed);

    let blocked = record(&db, "aggregate");
    assert_eq!(blocked.state, JobState::Blocked);
    assert_eq!(blocked.blocking_predecessor_id, Some(failed.id.clone()));
    assert!(!outcome.is_complete());
    assert_eq!(outcome.blocked.len(), 1);
    harness.cleanup();
}
