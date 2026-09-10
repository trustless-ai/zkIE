use std::fs::File;
use std::path::Path;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use thiserror::Error;

use crate::{
    AggregationArity, ArtifactRole, ArtifactStore, AttemptId, FailureCode, FailureKind,
    FailureRecord, FailureStage, JobEvent, JobId, JobKind, JobState, KeyMetadata, KeyRole,
    KeyStore, ReservationKey, StateError, StoreError, TransitionError, TrustedKeyIdentity,
    TrustedProofIdentity, VerificationRecordId, VerifiedArtifact, VerifiedKey,
};
use zkie_types::Digest32;

pub const SCHEMA_VERSION: u32 = 3;

const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at_unix_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at_unix_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    logical_job_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    state TEXT NOT NULL,
    failure_kind TEXT,
    failure_stage TEXT,
    failure_code TEXT,
    failure_summary TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count BETWEEN 0 AND 4294967295),
    blocking_predecessor_id TEXT,
    created_at_unix_ms INTEGER NOT NULL,
    updated_at_unix_ms INTEGER NOT NULL,
    UNIQUE(run_id, logical_job_id),
    UNIQUE(run_id, job_id),
    FOREIGN KEY(run_id, blocking_predecessor_id) REFERENCES jobs(run_id, job_id)
);
CREATE TABLE IF NOT EXISTS dependencies (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    predecessor_job_id TEXT NOT NULL,
    successor_job_id TEXT NOT NULL,
    PRIMARY KEY(run_id, predecessor_job_id, successor_job_id),
    CHECK(predecessor_job_id <> successor_job_id),
    FOREIGN KEY(run_id, predecessor_job_id) REFERENCES jobs(run_id, job_id) ON DELETE CASCADE,
    FOREIGN KEY(run_id, successor_job_id) REFERENCES jobs(run_id, job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS dependencies_by_successor
    ON dependencies(run_id, successor_job_id);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL,
    state TEXT NOT NULL,
    started_at_unix_ms INTEGER NOT NULL,
    finished_at_unix_ms INTEGER,
    UNIQUE(run_id, attempt_id),
    UNIQUE(run_id, job_id, attempt_id),
    FOREIGN KEY(run_id, job_id) REFERENCES jobs(run_id, job_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_attempt_per_job
    ON attempts(run_id, job_id) WHERE finished_at_unix_ms IS NULL;
CREATE TABLE IF NOT EXISTS verification_records (
    verification_record_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    verifier TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL,
    UNIQUE(run_id, verification_record_id),
    UNIQUE(run_id, job_id, verification_record_id),
    FOREIGN KEY(run_id, job_id) REFERENCES jobs(run_id, job_id) ON DELETE CASCADE,
    FOREIGN KEY(run_id, job_id, attempt_id) REFERENCES attempts(run_id, job_id, attempt_id)
);
CREATE TABLE IF NOT EXISTS state_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL,
    attempt_id TEXT,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    event_type TEXT NOT NULL,
    failure_kind TEXT,
    failure_stage TEXT,
    failure_code TEXT,
    failure_summary TEXT,
    blocking_predecessor_id TEXT,
    verification_record_id TEXT,
    created_at_unix_ms INTEGER NOT NULL,
    FOREIGN KEY(run_id, job_id) REFERENCES jobs(run_id, job_id) ON DELETE CASCADE,
    FOREIGN KEY(run_id, job_id, attempt_id) REFERENCES attempts(run_id, job_id, attempt_id),
    FOREIGN KEY(run_id, blocking_predecessor_id) REFERENCES jobs(run_id, job_id),
    FOREIGN KEY(run_id, job_id, verification_record_id)
        REFERENCES verification_records(run_id, job_id, verification_record_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    logical_job_id TEXT NOT NULL,
    digest TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL,
    FOREIGN KEY(run_id, logical_job_id) REFERENCES jobs(run_id, logical_job_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS resource_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL,
    observed_at_unix_ms INTEGER NOT NULL,
    resident_bytes INTEGER NOT NULL,
    cpu_millis INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(run_id, attempt_id) REFERENCES attempts(run_id, attempt_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS resource_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit_digest TEXT NOT NULL,
    circuit_k INTEGER NOT NULL,
    proof_flavor TEXT NOT NULL,
    execution_backend TEXT NOT NULL,
    hardware_profile TEXT NOT NULL,
    max_observed_peak_bytes INTEGER NOT NULL,
    updated_at_unix_ms INTEGER NOT NULL,
    UNIQUE(circuit_digest, circuit_k, proof_flavor, execution_backend, hardware_profile)
);
"#;

const MIGRATE_V1_TO_V2: &str = r#"
ALTER TABLE artifacts RENAME TO artifacts_v1_legacy;
CREATE TABLE artifacts (
    artifact_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL,
    logical_job_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    object_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    store_identity_digest TEXT NOT NULL,
    identity_digest TEXT NOT NULL,
    attestation_digest TEXT NOT NULL,
    artifact_role TEXT NOT NULL,
    job_kind TEXT NOT NULL,
    circuit_k INTEGER NOT NULL,
    aggregation_arity_tag INTEGER NOT NULL,
    aggregation_arity INTEGER,
    artifact_manifest_digest TEXT NOT NULL,
    run_identity_digest TEXT NOT NULL,
    public_statement_digest TEXT NOT NULL,
    circuit_digest TEXT NOT NULL,
    verifying_key_digest TEXT NOT NULL,
    srs_source_digest TEXT NOT NULL,
    shard_identity_digest TEXT NOT NULL,
    witness_artifact_digest TEXT NOT NULL,
    proof_flavor TEXT NOT NULL,
    execution_backend TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL,
    PRIMARY KEY(run_id, artifact_id),
    UNIQUE(run_id, logical_job_id),
    UNIQUE(run_id, object_digest),
    FOREIGN KEY(run_id, job_id) REFERENCES jobs(run_id, job_id) ON DELETE CASCADE,
    FOREIGN KEY(run_id, job_id, attempt_id) REFERENCES attempts(run_id, job_id, attempt_id)
);
CREATE INDEX artifacts_by_object_digest ON artifacts(object_digest);
CREATE TABLE key_artifacts (
    key_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL,
    logical_job_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    object_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    content_size INTEGER NOT NULL,
    store_identity_digest TEXT NOT NULL,
    identity_digest TEXT NOT NULL,
    attestation_digest TEXT NOT NULL,
    key_role TEXT NOT NULL,
    job_kind TEXT NOT NULL,
    srs_source_digest TEXT NOT NULL,
    proof_flavor TEXT NOT NULL,
    circuit_digest TEXT NOT NULL,
    circuit_k INTEGER NOT NULL,
    aggregation_arity INTEGER NOT NULL,
    created_at_unix_ms INTEGER NOT NULL,
    PRIMARY KEY(run_id, key_id),
    UNIQUE(run_id, logical_job_id),
    UNIQUE(run_id, object_digest),
    FOREIGN KEY(run_id, job_id) REFERENCES jobs(run_id, job_id) ON DELETE CASCADE,
    FOREIGN KEY(run_id, job_id, attempt_id) REFERENCES attempts(run_id, job_id, attempt_id)
);
CREATE INDEX key_artifacts_by_job ON key_artifacts(run_id, job_id);
"#;

const MIGRATE_V2_TO_V3: &str = r#"
ALTER TABLE jobs ADD COLUMN execution_retries INTEGER NOT NULL DEFAULT 0
    CHECK(execution_retries BETWEEN 0 AND 2);
ALTER TABLE jobs ADD COLUMN resource_requeues INTEGER NOT NULL DEFAULT 0
    CHECK(resource_requeues BETWEEN 0 AND 3);
ALTER TABLE jobs ADD COLUMN terminal_failure INTEGER NOT NULL DEFAULT 0
    CHECK(terminal_failure IN (0, 1));
UPDATE jobs SET terminal_failure=1 WHERE state IN ('verification-failed', 'blocked');
"#;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("sqlite error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("invalid state value: {0}")]
    State(#[from] StateError),
    #[error(transparent)]
    Transition(#[from] TransitionError),
    #[error("unknown {column} value {value}")]
    UnknownEnum { column: &'static str, value: String },
    #[error("job not found: {0}")]
    JobNotFound(JobId),
    #[error("attempt not found: {0}")]
    AttemptNotFound(AttemptId),
    #[error("begin_attempt requires a start event")]
    NotAttemptStart,
    #[error("attempt start events must use begin_attempt")]
    AttemptStartRequiresBegin,
    #[error("attempt events must include an attempt id")]
    AttemptEventRequiresAttempt,
    #[error("event is restricted to a controlled scheduler or verification API")]
    ControlledEventRequired,
    #[error("attempt {attempt} is not the unique open attempt for job {job}")]
    AttemptNotOpenForJob { attempt: AttemptId, job: JobId },
    #[error("job {job} already has an open attempt")]
    OpenAttemptConflict { job: JobId },
    #[error("job {job} exhausted its u32 attempt counter")]
    AttemptCountExhausted { job: JobId },
    #[error("attempt {attempt} is not an allowed verification source for job {job}")]
    InvalidVerificationSource { attempt: AttemptId, job: JobId },
    #[error("{event} cannot start a {kind:?} job")]
    WrongStartForJobKind { kind: JobKind, event: &'static str },
    #[error("system clock precedes the Unix epoch")]
    InvalidSystemTime,
    #[error("only an artifact-store publication can be committed as a verified job artifact")]
    ArtifactPublicationRequired,
    #[error("published artifact failed durable-store revalidation: {0}")]
    Store(#[from] StoreError),
    #[error("verified artifact attestation is bound to another run")]
    VerifiedArtifactRunMismatch,
    #[error("verified object identity job kind does not match the persisted job")]
    VerifiedIdentityJobKindMismatch,
    #[error("schema v1 contains legacy artifacts that require explicit reverification")]
    LegacyArtifactsRequireReverification,
    #[error("resource peak does not fit SQLite's exact non-negative integer representation")]
    ResourcePeakOutOfRange,
    #[error("resource-exceeded failures require a reservation key and observed peak; other failures must not provide them")]
    InvalidFailureObservation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobRecord {
    pub id: JobId,
    pub logical_job_id: String,
    pub kind: JobKind,
    pub state: JobState,
    pub failure_kind: Option<FailureKind>,
    pub failure_stage: Option<FailureStage>,
    pub failure_code: Option<FailureCode>,
    pub failure_summary: Option<String>,
    pub attempt_count: u32,
    pub execution_retries: u32,
    pub resource_requeues: u32,
    pub terminal_failure: bool,
    pub blocking_predecessor_id: Option<JobId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryDecision {
    Requeued,
    Terminal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptRecord {
    pub id: AttemptId,
    pub job_id: JobId,
    pub state: JobState,
    pub started_at_unix_ms: i64,
    pub finished_at_unix_ms: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateEventRecord {
    pub from_state: JobState,
    pub to_state: JobState,
    pub event_type: String,
    pub created_at_unix_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqliteSettings {
    pub journal_mode: String,
    pub foreign_keys: bool,
    pub synchronous: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ArtifactRecord {
    pub artifact_id: String,
    pub object_digest: Digest32,
    pub content_digest: Digest32,
    pub store_identity_digest: Digest32,
    pub identity_digest: Digest32,
    pub attestation_digest: Digest32,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct KeyArtifactRecord {
    pub key_id: String,
    pub object_digest: Digest32,
    pub content_digest: Digest32,
    pub store_identity_digest: Digest32,
    pub identity_digest: Digest32,
    pub attestation_digest: Digest32,
}

pub struct RunDb {
    connection: Connection,
    run_id: String,
}

mod artifacts;
mod core;
mod engine;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PersistedEventType {
    DependenciesSatisfied,
    DependencyBlocked,
    StartWitnessing,
    WitnessCompleted,
    StartPreparing,
    StartProving,
    ProofProduced,
    StartVerification,
    VerificationSucceeded,
    ExecutionFailed,
    VerificationFailed,
    PreparationCompleted,
    ResourceExceeded,
    Interrupted,
    RetryExecution,
    RetryResourceExceeded,
    RecoveryRequeue,
}

impl TryFrom<&str> for PersistedEventType {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "dependencies-satisfied" => Ok(Self::DependenciesSatisfied),
            "dependency-blocked" => Ok(Self::DependencyBlocked),
            "start-witnessing" => Ok(Self::StartWitnessing),
            "witness-completed" => Ok(Self::WitnessCompleted),
            "start-preparing" => Ok(Self::StartPreparing),
            "start-proving" => Ok(Self::StartProving),
            "proof-produced" => Ok(Self::ProofProduced),
            "start-verification" => Ok(Self::StartVerification),
            "verification-succeeded" => Ok(Self::VerificationSucceeded),
            "execution-failed" => Ok(Self::ExecutionFailed),
            "verification-failed" => Ok(Self::VerificationFailed),
            "preparation-completed" => Ok(Self::PreparationCompleted),
            "resource-exceeded" => Ok(Self::ResourceExceeded),
            "interrupted" => Ok(Self::Interrupted),
            "retry-execution" => Ok(Self::RetryExecution),
            "retry-resource-exceeded" => Ok(Self::RetryResourceExceeded),
            "recovery-requeue" => Ok(Self::RecoveryRequeue),
            _ => Err(()),
        }
    }
}

fn start_matches_kind(kind: JobKind, event: &JobEvent) -> bool {
    matches!(
        (kind, event),
        (JobKind::Witness, JobEvent::StartWitnessing)
            | (JobKind::Prepare, JobEvent::StartPreparing)
            | (JobKind::LeafProof, JobEvent::StartProving)
            | (
                JobKind::LeafVerification | JobKind::NativeAggregate,
                JobEvent::StartVerification
            )
    )
}

fn require_unique_open_attempt(
    transaction: &Transaction<'_>,
    run_id: &str,
    job: &JobId,
    attempt: &AttemptId,
) -> Result<(), DbError> {
    let mut statement = transaction.prepare(
        "SELECT attempt_id FROM attempts WHERE run_id=?1 AND job_id=?2 AND finished_at_unix_ms IS NULL ORDER BY attempt_id",
    )?;
    let open = statement
        .query_map(params![run_id, job.as_str()], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    if open.len() == 1 && open[0] == attempt.as_str() {
        Ok(())
    } else {
        Err(DbError::AttemptNotOpenForJob {
            attempt: attempt.clone(),
            job: job.clone(),
        })
    }
}

// Kept with the crate-internal verification entry point until its worker is introduced.
#[allow(dead_code)]
fn require_verification_source(
    transaction: &Transaction<'_>,
    run_id: &str,
    job: &JobId,
    attempt: &AttemptId,
) -> Result<(), DbError> {
    let (kind_raw, state_raw): (String, String) = transaction
        .query_row(
            "SELECT kind,state FROM jobs WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?
        .ok_or_else(|| DbError::JobNotFound(job.clone()))?;
    let kind = parse_kind(&kind_raw)?;
    let state = parse_state(&state_raw)?;
    if matches!(
        (kind, state),
        (
            JobKind::LeafVerification | JobKind::NativeAggregate,
            JobState::Verifying
        )
    ) {
        return require_unique_open_attempt(transaction, run_id, job, attempt);
    }
    if matches!(
        (kind, state),
        (JobKind::Witness, JobState::WitnessReady)
            | (JobKind::Prepare, JobState::Prepared)
            | (JobKind::LeafProof, JobState::Proved)
    ) {
        let latest: Option<(String, String, Option<i64>)> = transaction
            .query_row(
                "SELECT attempt_id,state,finished_at_unix_ms FROM attempts WHERE run_id=?1 AND job_id=?2 ORDER BY rowid DESC LIMIT 1",
                params![run_id, job.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if let Some((latest_id, attempt_state, Some(_))) = latest {
            if latest_id == attempt.as_str() && parse_state(&attempt_state)? == state {
                return Ok(());
            }
        }
    }
    Err(DbError::InvalidVerificationSource {
        attempt: attempt.clone(),
        job: job.clone(),
    })
}

fn apply_event_tx(
    transaction: &Transaction<'_>,
    run_id: &str,
    job: &JobId,
    attempt: Option<&AttemptId>,
    event: &JobEvent,
    attempt_count: Option<u32>,
) -> Result<JobState, DbError> {
    let current = current_state(transaction, run_id, job)?;
    let next = current.transition(event.clone())?;
    let now = unix_ms()?;
    let failure = event.failure();
    let blocker = event.blocking_predecessor();
    transaction.execute(
        "INSERT INTO state_events(run_id,job_id,attempt_id,from_state,to_state,event_type,failure_kind,failure_stage,failure_code,failure_summary,blocking_predecessor_id,verification_record_id,created_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
        params![
            run_id,
            job.as_str(),
            attempt.map(AttemptId::as_str),
            current.as_str(),
            next.as_str(),
            event.name(),
            failure.map(|value| value.kind().as_str()),
            failure.map(|value| value.stage().as_str()),
            failure.map(|value| value.code().as_str()),
            failure.map(FailureRecord::summary),
            blocker.map(JobId::as_str),
            event.verification_record().map(VerificationRecordId::as_str),
            now,
        ],
    )?;
    if failure.is_some() || blocker.is_some() {
        let dependency_failure = blocker.is_some().then_some(FailureKind::Dependency);
        let dependency_stage = blocker.is_some().then_some(FailureStage::Scheduler);
        transaction.execute(
            "UPDATE jobs SET state=?1,failure_kind=?2,failure_stage=?3,failure_code=?4,failure_summary=?5,blocking_predecessor_id=?6,attempt_count=COALESCE(?7,attempt_count),updated_at_unix_ms=?8 WHERE run_id=?9 AND job_id=?10",
            params![
                next.as_str(),
                failure.map(|value| value.kind().as_str()).or_else(|| dependency_failure.map(FailureKind::as_str)),
                failure.map(|value| value.stage().as_str()).or_else(|| dependency_stage.map(FailureStage::as_str)),
                failure.map(|value| value.code().as_str()).or(blocker.map(|_| FailureCode::PredecessorFailed.as_str())),
                failure.map(FailureRecord::summary).or(blocker.map(|_| FailureKind::Dependency.summary())),
                blocker.map(JobId::as_str),
                attempt_count.map(i64::from),
                now,
                run_id,
                job.as_str(),
            ],
        )?;
    } else {
        let clear = matches!(event, JobEvent::DependenciesSatisfied);
        transaction.execute(
            "UPDATE jobs SET state=?1,failure_kind=CASE WHEN ?2 THEN NULL ELSE failure_kind END,failure_stage=CASE WHEN ?2 THEN NULL ELSE failure_stage END,failure_code=CASE WHEN ?2 THEN NULL ELSE failure_code END,failure_summary=CASE WHEN ?2 THEN NULL ELSE failure_summary END,blocking_predecessor_id=CASE WHEN ?2 THEN NULL ELSE blocking_predecessor_id END,attempt_count=COALESCE(?3,attempt_count),updated_at_unix_ms=?4 WHERE run_id=?5 AND job_id=?6",
            params![next.as_str(), clear, attempt_count.map(i64::from), now, run_id, job.as_str()],
        )?;
    }
    if !next.is_active() {
        transaction.execute(
            "UPDATE attempts SET state=?1,finished_at_unix_ms=COALESCE(finished_at_unix_ms,?2) WHERE run_id=?3 AND job_id=?4 AND finished_at_unix_ms IS NULL",
            params![next.as_str(), now, run_id, job.as_str()],
        )?;
    } else if let Some(attempt) = attempt {
        transaction.execute(
            "UPDATE attempts SET state=?1 WHERE attempt_id=?2",
            params![next.as_str(), attempt.as_str()],
        )?;
    }
    Ok(next)
}

fn current_state(
    transaction: &Transaction<'_>,
    run_id: &str,
    job: &JobId,
) -> Result<JobState, DbError> {
    let value = transaction
        .query_row(
            "SELECT state FROM jobs WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .ok_or_else(|| DbError::JobNotFound(job.clone()))?;
    parse_state(&value)
}

fn predecessor_states(
    transaction: &Transaction<'_>,
    run_id: &str,
    successor: &JobId,
) -> Result<Vec<(JobId, JobState, bool)>, DbError> {
    let mut statement = transaction.prepare(
        "SELECT j.job_id,j.state,j.terminal_failure FROM dependencies d JOIN jobs j ON j.job_id=d.predecessor_job_id WHERE d.run_id=?1 AND d.successor_job_id=?2 ORDER BY j.job_id",
    )?;
    let rows = statement.query_map(params![run_id, successor.as_str()], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;
    rows.map(|row| {
        let (id, state, terminal) = row?;
        Ok((JobId::new(id)?, parse_state(&state)?, terminal == 1))
    })
    .collect()
}

fn exact_peak_to_sql(peak_bytes: u64) -> Result<i64, DbError> {
    i64::try_from(peak_bytes).map_err(|_| DbError::ResourcePeakOutOfRange)
}

fn exact_peak_from_sql(peak_bytes: i64) -> Result<u64, DbError> {
    u64::try_from(peak_bytes).map_err(|_| DbError::ResourcePeakOutOfRange)
}

fn record_resource_peak_tx(
    transaction: &Transaction<'_>,
    key: &ReservationKey,
    peak_bytes: u64,
) -> Result<(), DbError> {
    transaction.execute(
        "INSERT INTO resource_history(
            circuit_digest,circuit_k,proof_flavor,execution_backend,hardware_profile,
            max_observed_peak_bytes,updated_at_unix_ms
         ) VALUES(?1,?2,?3,?4,?5,?6,?7)
         ON CONFLICT(circuit_digest,circuit_k,proof_flavor,execution_backend,hardware_profile)
         DO UPDATE SET
            max_observed_peak_bytes=MAX(max_observed_peak_bytes,excluded.max_observed_peak_bytes),
            updated_at_unix_ms=excluded.updated_at_unix_ms",
        params![
            key.circuit_digest.to_string(),
            i64::from(key.k),
            key.proof_flavor.as_str(),
            key.execution_backend.as_str(),
            key.hardware_profile.to_string(),
            exact_peak_to_sql(peak_bytes)?,
            unix_ms()?,
        ],
    )?;
    Ok(())
}

fn block_descendants_tx(
    transaction: &Transaction<'_>,
    run_id: &str,
    failed: &JobId,
) -> Result<(), DbError> {
    let mut frontier = vec![failed.clone()];
    while let Some(predecessor) = frontier.pop() {
        let successors = {
            let mut statement = transaction.prepare(
                "SELECT successor_job_id FROM dependencies
                 WHERE run_id=?1 AND predecessor_job_id=?2 ORDER BY successor_job_id",
            )?;
            let rows = statement
                .query_map(params![run_id, predecessor.as_str()], |row| {
                    row.get::<_, String>(0)
                })?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        for raw_successor in successors {
            let successor = JobId::new(raw_successor)?;
            let state = current_state(transaction, run_id, &successor)?;
            if state == JobState::Blocked {
                continue;
            }
            if !matches!(state, JobState::Pending | JobState::Requeued) {
                continue;
            }
            apply_event_tx(
                transaction,
                run_id,
                &successor,
                None,
                &JobEvent::DependencyBlocked {
                    predecessor: predecessor.clone(),
                },
                None,
            )?;
            transaction.execute(
                "UPDATE jobs SET terminal_failure=1 WHERE run_id=?1 AND job_id=?2",
                params![run_id, successor.as_str()],
            )?;
            frontier.push(successor);
        }
    }
    Ok(())
}

fn parse_state(value: &str) -> Result<JobState, DbError> {
    JobState::try_from(value).map_err(|_| DbError::UnknownEnum {
        column: "job_state",
        value: value.to_owned(),
    })
}

fn parse_kind(value: &str) -> Result<JobKind, DbError> {
    JobKind::try_from(value).map_err(|_| DbError::UnknownEnum {
        column: "job_kind",
        value: value.to_owned(),
    })
}

fn parse_failure_kind(value: &str) -> Result<FailureKind, DbError> {
    FailureKind::try_from(value).map_err(|_| DbError::UnknownEnum {
        column: "failure_kind",
        value: value.to_owned(),
    })
}

fn parse_digest(column: &'static str, value: String) -> Result<Digest32, DbError> {
    Digest32::from_str(&value).map_err(|_| DbError::UnknownEnum { column, value })
}

fn parse_u32(column: &'static str, value: i64) -> Result<u32, DbError> {
    u32::try_from(value).map_err(|_| DbError::UnknownEnum {
        column,
        value: value.to_string(),
    })
}

fn parse_failure_stage(value: &str) -> Result<FailureStage, DbError> {
    FailureStage::try_from(value).map_err(|_| DbError::UnknownEnum {
        column: "failure_stage",
        value: value.to_owned(),
    })
}

fn parse_failure_code(value: &str) -> Result<FailureCode, DbError> {
    FailureCode::try_from(value).map_err(|_| DbError::UnknownEnum {
        column: "failure_code",
        value: value.to_owned(),
    })
}

fn validate_text_id(value: &str) -> Result<(), StateError> {
    JobId::new(value.to_owned()).map(|_| ())
}

fn unix_ms() -> Result<i64, DbError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| DbError::InvalidSystemTime)?;
    i64::try_from(duration.as_millis()).map_err(|_| DbError::InvalidSystemTime)
}
