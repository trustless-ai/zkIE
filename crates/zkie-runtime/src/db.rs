use std::fs::File;
use std::path::Path;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use thiserror::Error;

use crate::{
    AggregationArity, ArtifactRole, ArtifactStore, AttemptId, FailureCode, FailureKind,
    FailureRecord, FailureStage, JobEvent, JobId, JobKind, JobState, KeyMetadata, KeyRole,
    KeyStore, StateError, StoreError, TransitionError, TrustedKeyIdentity, TrustedProofIdentity,
    VerificationRecordId, VerifiedArtifact, VerifiedKey,
};
use zkie_types::Digest32;

pub const SCHEMA_VERSION: u32 = 2;

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
    pub blocking_predecessor_id: Option<JobId>,
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

impl RunDb {
    pub fn open(path: impl AsRef<Path>, run_id: &str) -> Result<Self, DbError> {
        validate_text_id(run_id)?;
        let mut connection = Connection::open(path)?;
        let has_schema_version: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version')",
            [],
            |row| row.get(0),
        )?;
        let persisted_version = if has_schema_version {
            connection.query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get::<_, Option<i64>>(0)
            })?
        } else {
            None
        };
        if let Some(version) = persisted_version {
            if version < 1 || version > i64::from(SCHEMA_VERSION) {
                return Err(DbError::UnknownEnum {
                    column: "schema_version",
                    value: version.to_string(),
                });
            }
        }
        if persisted_version == Some(1) {
            let legacy_artifacts: i64 =
                connection.query_row("SELECT COUNT(*) FROM artifacts", [], |row| row.get(0))?;
            let legacy_verified: i64 = connection.query_row(
                "SELECT COUNT(*) FROM jobs WHERE state='verified'",
                [],
                |row| row.get(0),
            )?;
            let legacy_success_records: i64 =
                connection.query_row("SELECT COUNT(*) FROM verification_records", [], |row| {
                    row.get(0)
                })?;
            if legacy_artifacts != 0 || legacy_verified != 0 || legacy_success_records != 0 {
                return Err(DbError::LegacyArtifactsRequireReverification);
            }
        }

        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", true)?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute_batch(SCHEMA_V1)?;
        let now = unix_ms()?;
        if persisted_version.is_none() {
            transaction.execute(
                "INSERT INTO schema_version(version, applied_at_unix_ms) VALUES(1, ?1)",
                params![now],
            )?;
        }
        if persisted_version.is_none() || persisted_version == Some(1) {
            transaction.execute_batch(MIGRATE_V1_TO_V2)?;
            transaction.execute(
                "INSERT INTO schema_version(version, applied_at_unix_ms) VALUES(2, ?1)",
                params![now],
            )?;
        }
        transaction.execute(
            "INSERT OR IGNORE INTO runs(run_id, created_at_unix_ms) VALUES(?1, ?2)",
            params![run_id, now],
        )?;
        transaction.commit()?;
        Ok(Self {
            connection,
            run_id: run_id.to_owned(),
        })
    }

    pub fn schema_version(&self) -> Result<u32, DbError> {
        let value: i64 =
            self.connection
                .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                    row.get(0)
                })?;
        u32::try_from(value).map_err(|_| DbError::UnknownEnum {
            column: "schema_version",
            value: value.to_string(),
        })
    }

    pub fn sqlite_settings(&self) -> Result<SqliteSettings, DbError> {
        let journal_mode: String =
            self.connection
                .pragma_query_value(None, "journal_mode", |row| row.get(0))?;
        let foreign_keys: i64 =
            self.connection
                .pragma_query_value(None, "foreign_keys", |row| row.get(0))?;
        let synchronous: i64 = self
            .connection
            .pragma_query_value(None, "synchronous", |row| row.get(0))?;
        Ok(SqliteSettings {
            journal_mode: journal_mode.to_ascii_lowercase(),
            foreign_keys: foreign_keys == 1,
            synchronous: match synchronous {
                2 => "FULL".into(),
                other => other.to_string(),
            },
        })
    }

    pub fn insert_job(&mut self, logical_job_id: &str, kind: JobKind) -> Result<JobId, DbError> {
        validate_text_id(logical_job_id)?;
        let id = JobId::new(format!("{}:{logical_job_id}", self.run_id))?;
        let now = unix_ms()?;
        self.connection.execute(
            "INSERT INTO jobs(job_id, run_id, logical_job_id, kind, state, created_at_unix_ms, updated_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6,?6)",
            params![id.as_str(), self.run_id, logical_job_id, kind.as_str(), JobState::Pending.as_str(), now],
        )?;
        Ok(id)
    }

    pub fn add_dependency(
        &mut self,
        predecessor: &JobId,
        successor: &JobId,
    ) -> Result<(), DbError> {
        self.require_job_in_run(predecessor)?;
        self.require_job_in_run(successor)?;
        self.connection.execute(
            "INSERT INTO dependencies(run_id, predecessor_job_id, successor_job_id) VALUES(?1,?2,?3)",
            params![self.run_id, predecessor.as_str(), successor.as_str()],
        )?;
        Ok(())
    }

    pub fn job(&self, id: &JobId) -> Result<JobRecord, DbError> {
        let raw = self
            .connection
            .query_row(
                "SELECT logical_job_id,kind,state,failure_kind,failure_stage,failure_code,failure_summary,attempt_count,blocking_predecessor_id FROM jobs WHERE run_id=?1 AND job_id=?2",
                params![self.run_id, id.as_str()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, Option<String>>(6)?,
                        row.get::<_, i64>(7)?,
                        row.get::<_, Option<String>>(8)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| DbError::JobNotFound(id.clone()))?;
        Ok(JobRecord {
            id: id.clone(),
            logical_job_id: raw.0,
            kind: parse_kind(&raw.1)?,
            state: parse_state(&raw.2)?,
            failure_kind: raw.3.as_deref().map(parse_failure_kind).transpose()?,
            failure_stage: raw.4.as_deref().map(parse_failure_stage).transpose()?,
            failure_code: raw.5.as_deref().map(parse_failure_code).transpose()?,
            failure_summary: raw.6,
            attempt_count: u32::try_from(raw.7).map_err(|_| DbError::UnknownEnum {
                column: "attempt_count",
                value: raw.7.to_string(),
            })?,
            blocking_predecessor_id: raw.8.map(JobId::new).transpose()?,
        })
    }

    pub fn attempt(&self, id: &AttemptId) -> Result<AttemptRecord, DbError> {
        let raw = self
            .connection
            .query_row(
                "SELECT job_id,state,started_at_unix_ms,finished_at_unix_ms FROM attempts WHERE run_id=?1 AND attempt_id=?2",
                params![self.run_id, id.as_str()],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?
            .ok_or_else(|| DbError::AttemptNotFound(id.clone()))?;
        Ok(AttemptRecord {
            id: id.clone(),
            job_id: JobId::new(raw.0)?,
            state: parse_state(&raw.1)?,
            started_at_unix_ms: raw.2,
            finished_at_unix_ms: raw.3,
        })
    }

    pub fn events(&self, id: &JobId) -> Result<Vec<StateEventRecord>, DbError> {
        self.require_job_in_run(id)?;
        let mut statement = self.connection.prepare(
            "SELECT from_state,to_state,event_type,created_at_unix_ms FROM state_events WHERE run_id=?1 AND job_id=?2 ORDER BY event_id",
        )?;
        let rows = statement.query_map(params![self.run_id, id.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;
        rows.map(|row| {
            let (from, to, event_type, created_at_unix_ms) = row?;
            PersistedEventType::try_from(event_type.as_str()).map_err(|_| {
                DbError::UnknownEnum {
                    column: "event_type",
                    value: event_type.clone(),
                }
            })?;
            Ok(StateEventRecord {
                from_state: parse_state(&from)?,
                to_state: parse_state(&to)?,
                event_type,
                created_at_unix_ms,
            })
        })
        .collect()
    }

    #[allow(dead_code)]
    pub(crate) fn artifact_for_job(&self, job: &JobId) -> Result<Option<ArtifactRecord>, DbError> {
        self.require_job_in_run(job)?;
        let raw = self
            .connection
            .query_row(
                "SELECT artifact_id,object_digest,content_digest,store_identity_digest,identity_digest,attestation_digest FROM artifacts WHERE run_id=?1 AND job_id=?2",
                params![self.run_id, job.as_str()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .optional()?;
        raw.map(
            |(artifact_id, object, content, store, identity, attestation)| {
                Ok(ArtifactRecord {
                    artifact_id,
                    object_digest: parse_digest("artifact_object_digest", object)?,
                    content_digest: parse_digest("artifact_content_digest", content)?,
                    store_identity_digest: parse_digest("artifact_store_identity_digest", store)?,
                    identity_digest: parse_digest("artifact_identity_digest", identity)?,
                    attestation_digest: parse_digest("artifact_attestation_digest", attestation)?,
                })
            },
        )
        .transpose()
    }

    #[allow(dead_code)]
    pub(crate) fn reopen_artifact_for_job(
        &self,
        store: &ArtifactStore,
        job: &JobId,
    ) -> Result<Option<File>, DbError> {
        let Some(record) = self.artifact_for_job(job)? else {
            return Ok(None);
        };
        let identity = self.persisted_proof_identity(job)?;
        if identity.digest() != record.identity_digest {
            return Err(StoreError::ProofIdentityMismatch.into());
        }
        let (file, metadata) =
            store.reopen_persisted(record.store_identity_digest, record.object_digest)?;
        let artifact = match metadata {
            crate::ObjectMetadata::Artifact(value) => value,
            crate::ObjectMetadata::Key(_) => return Err(StoreError::WrongStoreKind.into()),
        };
        if artifact.artifact_id() != record.artifact_id
            || artifact.proof_identity_digest() != Some(record.identity_digest)
            || artifact.content_digest() != record.content_digest
        {
            return Err(StoreError::ProofIdentityMismatch.into());
        }
        let (run_id, job_id, attempt_id): (String, String, String) = self.connection.query_row(
            "SELECT run_id,job_id,attempt_id FROM artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        let attempt = AttemptId::new(attempt_id)?;
        if crate::proof_attestation_digest(
            record.identity_digest,
            record.object_digest,
            &run_id,
            &JobId::new(job_id)?,
            &attempt,
        ) != record.attestation_digest
        {
            return Err(StoreError::ProofIdentityMismatch.into());
        }
        Ok(Some(file))
    }

    fn persisted_proof_identity(&self, job: &JobId) -> Result<TrustedProofIdentity, DbError> {
        type RawProofIdentity = (
            String,
            String,
            i64,
            i64,
            Option<i64>,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
        );
        let raw: RawProofIdentity = self.connection.query_row(
            "SELECT artifact_role,job_kind,circuit_k,aggregation_arity_tag,aggregation_arity,
                    run_id,job_id,attempt_id,content_digest,artifact_manifest_digest,
                    run_identity_digest,public_statement_digest,circuit_digest,verifying_key_digest,
                    srs_source_digest,shard_identity_digest,witness_artifact_digest,
                    proof_flavor,execution_backend,identity_digest
             FROM artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                    row.get(10)?,
                    row.get(11)?,
                    row.get(12)?,
                    row.get(13)?,
                    row.get(14)?,
                    row.get(15)?,
                    row.get(16)?,
                    row.get(17)?,
                    row.get(18)?,
                    row.get(19)?,
                ))
            },
        )?;
        let role = match raw.0.as_str() {
            "witness" => ArtifactRole::Witness,
            "prepared-material" => ArtifactRole::PreparedMaterial,
            "leaf-proof" => ArtifactRole::LeafProof,
            "native-verified-manifest" => ArtifactRole::NativeVerifiedManifest,
            value => {
                return Err(DbError::UnknownEnum {
                    column: "artifact_role",
                    value: value.to_owned(),
                })
            }
        };
        let kind = JobKind::try_from(raw.1.as_str()).map_err(|_| DbError::UnknownEnum {
            column: "artifact_job_kind",
            value: raw.1.clone(),
        })?;
        let k = parse_u32("artifact_circuit_k", raw.2)?;
        let arity = match (raw.3, raw.4) {
            (0, None) => AggregationArity::NotApplicable,
            (1, Some(value)) => {
                AggregationArity::Actual(parse_u32("artifact_aggregation_arity", value)?)
            }
            _ => {
                return Err(DbError::UnknownEnum {
                    column: "artifact_aggregation_arity_tag",
                    value: raw.3.to_string(),
                })
            }
        };
        let run_id = raw.5;
        let persisted_job = JobId::new(raw.6)?;
        let attempt = AttemptId::new(raw.7)?;
        let identity = TrustedProofIdentity::new(
            role,
            kind,
            k,
            arity,
            &run_id,
            &persisted_job,
            &attempt,
            parse_digest("artifact_content_digest", raw.8)?,
            parse_digest("artifact_manifest_digest", raw.9)?,
            parse_digest("artifact_run_identity_digest", raw.10)?,
            parse_digest("artifact_public_statement_digest", raw.11)?,
            parse_digest("artifact_circuit_digest", raw.12)?,
            parse_digest("artifact_verifying_key_digest", raw.13)?,
            parse_digest("artifact_srs_source_digest", raw.14)?,
            parse_digest("artifact_shard_identity_digest", raw.15)?,
            parse_digest("artifact_witness_artifact_digest", raw.16)?,
            zkie_types::ProofFlavorId::parse(raw.17).map_err(|_| StoreError::InvalidMetadata)?,
            zkie_types::ExecutionBackendId::parse(raw.18)
                .map_err(|_| StoreError::InvalidMetadata)?,
        )?;
        if identity.digest() != parse_digest("artifact_identity_digest", raw.19)? {
            return Err(StoreError::ProofIdentityMismatch.into());
        }
        Ok(identity)
    }

    #[allow(dead_code)]
    pub(crate) fn key_for_job(&self, job: &JobId) -> Result<Option<KeyArtifactRecord>, DbError> {
        self.require_job_in_run(job)?;
        let raw = self.connection.query_row(
            "SELECT key_id,object_digest,content_digest,store_identity_digest,identity_digest,attestation_digest FROM key_artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?, row.get::<_, String>(4)?, row.get::<_, String>(5)?)),
        ).optional()?;
        raw.map(|(key_id, object, content, store, identity, attestation)| {
            Ok(KeyArtifactRecord {
                key_id,
                object_digest: parse_digest("key_object_digest", object)?,
                content_digest: parse_digest("key_content_digest", content)?,
                store_identity_digest: parse_digest("key_store_identity_digest", store)?,
                identity_digest: parse_digest("key_identity_digest", identity)?,
                attestation_digest: parse_digest("key_attestation_digest", attestation)?,
            })
        })
        .transpose()
    }

    #[allow(dead_code)]
    pub(crate) fn reopen_key_for_job(
        &self,
        store: &KeyStore,
        job: &JobId,
    ) -> Result<Option<File>, DbError> {
        let Some(record) = self.key_for_job(job)? else {
            return Ok(None);
        };
        let identity = self.persisted_key_identity(job)?;
        if identity.digest() != record.identity_digest {
            return Err(StoreError::KeyIdentityMismatch.into());
        }
        let (file, metadata) =
            store.reopen_persisted(record.store_identity_digest, record.object_digest)?;
        let key = match metadata {
            crate::ObjectMetadata::Key(value) => value,
            crate::ObjectMetadata::Artifact(_) => return Err(StoreError::WrongStoreKind.into()),
        };
        if key.key_identity_digest() != Some(record.identity_digest)
            || key.content_digest() != record.content_digest
        {
            return Err(StoreError::KeyIdentityMismatch.into());
        }
        let (run_id, job_id, attempt_id): (String, String, String) = self.connection.query_row(
            "SELECT run_id,job_id,attempt_id FROM key_artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        if crate::key_attestation_digest(
            record.identity_digest,
            record.object_digest,
            &run_id,
            &JobId::new(job_id)?,
            &AttemptId::new(attempt_id)?,
        ) != record.attestation_digest
        {
            return Err(StoreError::KeyIdentityMismatch.into());
        }
        Ok(Some(file))
    }

    fn persisted_key_identity(&self, job: &JobId) -> Result<TrustedKeyIdentity, DbError> {
        type RawKeyIdentity = (
            String,
            String,
            String,
            String,
            String,
            String,
            i64,
            String,
            String,
            String,
            i64,
            i64,
            String,
        );
        let raw: RawKeyIdentity = self.connection.query_row(
            "SELECT key_role,job_kind,run_id,job_id,attempt_id,content_digest,content_size,
                    srs_source_digest,proof_flavor,circuit_digest,circuit_k,aggregation_arity,
                    identity_digest
             FROM key_artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                    row.get(10)?,
                    row.get(11)?,
                    row.get(12)?,
                ))
            },
        )?;
        if raw.0 != "proving-and-verifying-key" {
            return Err(DbError::UnknownEnum {
                column: "key_role",
                value: raw.0,
            });
        }
        let kind = JobKind::try_from(raw.1.as_str()).map_err(|_| DbError::UnknownEnum {
            column: "key_job_kind",
            value: raw.1.clone(),
        })?;
        let run_id = raw.2;
        let persisted_job = JobId::new(raw.3)?;
        let attempt = AttemptId::new(raw.4)?;
        let metadata = KeyMetadata::new(
            parse_digest("key_content_digest", raw.5)?,
            u64::try_from(raw.6).map_err(|_| DbError::UnknownEnum {
                column: "key_content_size",
                value: raw.6.to_string(),
            })?,
            parse_digest("key_srs_source_digest", raw.7)?,
            zkie_types::ProofFlavorId::parse(raw.8).map_err(|_| StoreError::InvalidMetadata)?,
            parse_digest("key_circuit_digest", raw.9)?,
            parse_u32("key_circuit_k", raw.10)?,
            parse_u32("key_aggregation_arity", raw.11)?,
        )?;
        let identity = TrustedKeyIdentity::new(
            KeyRole::ProvingAndVerifyingKey,
            kind,
            &run_id,
            &persisted_job,
            &attempt,
            metadata,
        )?;
        if identity.digest() != parse_digest("key_identity_digest", raw.12)? {
            return Err(StoreError::KeyIdentityMismatch.into());
        }
        Ok(identity)
    }

    /// Atomically records a durable artifact and the successful independent verification.
    ///
    /// The crate-internal attestation binds the independently verified proof identity and source
    /// attempt to the exact durable object; callers outside the trusted verifier cannot construct it.
    #[allow(dead_code)]
    pub(crate) fn commit_verified_artifact(
        &mut self,
        verified: &VerifiedArtifact,
    ) -> Result<JobState, DbError> {
        let (attested_run, job, source_attempt, identity, published) = verified.binding();
        if attested_run != self.run_id {
            return Err(DbError::VerifiedArtifactRunMismatch);
        }
        let verifier = identity.verifier_id();
        let artifact_id = published
            .artifact_id()
            .ok_or(DbError::ArtifactPublicationRequired)?;
        let metadata = match published.metadata() {
            crate::ObjectMetadata::Artifact(metadata) => metadata,
            crate::ObjectMetadata::Key(_) => return Err(DbError::ArtifactPublicationRequired),
        };
        let audit = identity.audit();
        let (arity_tag, arity) = audit.aggregation_arity();
        let identity_digest = identity.digest();
        let attestation_digest = verified.attestation_digest();
        published.ensure_verified()?;
        published.ensure_current_locator()?;
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Full proof hashing happens before taking SQLite's write lock. Within the transaction only
        // inode/locator identity is checked again before the attested row and state transition land.
        published.ensure_current_locator()?;
        published.ensure_pinned_file_identity()?;
        require_verification_source(&transaction, &run_id, job, source_attempt)?;
        let (logical_job_id, persisted_kind): (String, String) = transaction.query_row(
            "SELECT logical_job_id,kind FROM jobs WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if persisted_kind != audit.job_kind() {
            return Err(DbError::VerifiedIdentityJobKindMismatch);
        }
        let ordinal: i64 = transaction.query_row(
            "SELECT COUNT(*) + 1 FROM verification_records WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| row.get(0),
        )?;
        let verification =
            VerificationRecordId::new(format!("{}:verification:{ordinal}", job.as_str()))?;
        let now = unix_ms()?;
        transaction.execute(
            "INSERT INTO verification_records(verification_record_id,run_id,job_id,attempt_id,verifier,created_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6)",
            params![verification.as_str(), run_id, job.as_str(), source_attempt.as_str(), verifier, now],
        )?;
        transaction.execute(
            "INSERT INTO artifacts(
                artifact_id,run_id,job_id,logical_job_id,attempt_id,object_digest,content_digest,
                store_identity_digest,identity_digest,attestation_digest,artifact_role,job_kind,
                circuit_k,aggregation_arity_tag,aggregation_arity,artifact_manifest_digest,
                run_identity_digest,public_statement_digest,circuit_digest,verifying_key_digest,
                srs_source_digest,shard_identity_digest,witness_artifact_digest,proof_flavor,
                execution_backend,created_at_unix_ms
             ) VALUES(
                ?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,
                ?19,?20,?21,?22,?23,?24,?25,?26
             )",
            params![
                artifact_id,
                run_id,
                job.as_str(),
                logical_job_id,
                source_attempt.as_str(),
                published.digest().to_string(),
                metadata.content_digest().to_string(),
                published.store_identity_digest().to_string(),
                identity_digest.to_string(),
                attestation_digest.to_string(),
                audit.role(),
                audit.job_kind(),
                i64::from(audit.k()),
                arity_tag,
                arity.map(i64::from),
                audit.artifact_manifest_digest().to_string(),
                audit.run_identity_digest().to_string(),
                audit.public_statement_digest().to_string(),
                audit.circuit_digest().to_string(),
                audit.verifying_key_digest().to_string(),
                audit.srs_source_digest().to_string(),
                audit.shard_identity_digest().to_string(),
                audit.witness_artifact_digest().to_string(),
                audit.proof_flavor(),
                audit.execution_backend(),
                now,
            ],
        )?;
        let event = JobEvent::VerificationSucceeded {
            verification_record: Some(verification),
        };
        let state = apply_event_tx(
            &transaction,
            &run_id,
            job,
            Some(source_attempt),
            &event,
            None,
        )?;
        transaction.commit()?;
        Ok(state)
    }

    #[allow(dead_code)]
    pub(crate) fn commit_verified_key(
        &mut self,
        verified: &VerifiedKey,
    ) -> Result<JobState, DbError> {
        let (attested_run, job, source_attempt, identity, published) = verified.binding();
        if attested_run != self.run_id {
            return Err(DbError::VerifiedArtifactRunMismatch);
        }
        let metadata = identity.metadata();
        let identity_digest = identity.digest();
        let attestation_digest = verified.attestation_digest();
        published.ensure_verified()?;
        published.ensure_current_locator()?;
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        published.ensure_current_locator()?;
        published.ensure_pinned_file_identity()?;
        require_verification_source(&transaction, &run_id, job, source_attempt)?;
        let (logical_job_id, persisted_kind): (String, String) = transaction.query_row(
            "SELECT logical_job_id,kind FROM jobs WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if persisted_kind != identity.job_kind() {
            return Err(DbError::VerifiedIdentityJobKindMismatch);
        }
        let ordinal: i64 = transaction.query_row(
            "SELECT COUNT(*) + 1 FROM verification_records WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| row.get(0),
        )?;
        let verification =
            VerificationRecordId::new(format!("{}:verification:{ordinal}", job.as_str()))?;
        let now = unix_ms()?;
        transaction.execute(
            "INSERT INTO verification_records(verification_record_id,run_id,job_id,attempt_id,verifier,created_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6)",
            params![verification.as_str(), run_id, job.as_str(), source_attempt.as_str(), metadata.proof_flavor().as_str(), now],
        )?;
        transaction.execute(
            "INSERT INTO key_artifacts(key_id,run_id,job_id,logical_job_id,attempt_id,object_digest,content_digest,content_size,store_identity_digest,identity_digest,attestation_digest,key_role,job_kind,srs_source_digest,proof_flavor,circuit_digest,circuit_k,aggregation_arity,created_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)",
            params![
                identity_digest.to_string(), run_id, job.as_str(), logical_job_id,
                source_attempt.as_str(), published.digest().to_string(),
                metadata.content_digest().to_string(), i64::try_from(metadata.size()).map_err(|_| StoreError::InvalidMetadata)?, published.store_identity_digest().to_string(),
                identity_digest.to_string(), attestation_digest.to_string(), identity.role(),
                identity.job_kind(), metadata.srs_source_digest().to_string(),
                metadata.proof_flavor().as_str(), metadata.circuit_digest().to_string(),
                i64::from(metadata.k()), i64::from(metadata.aggregation_arity()), now,
            ],
        )?;
        let state = apply_event_tx(
            &transaction,
            &run_id,
            job,
            Some(source_attempt),
            &JobEvent::VerificationSucceeded {
                verification_record: Some(verification),
            },
            None,
        )?;
        transaction.commit()?;
        Ok(state)
    }

    // The crate-internal verifier worker introduced by the scheduler consumes this entry point.
    #[allow(dead_code)]
    pub(crate) fn complete_verification(
        &mut self,
        job: &JobId,
        attempt: &AttemptId,
        verifier: &str,
    ) -> Result<JobState, DbError> {
        validate_text_id(verifier)?;
        if self.job(job)?.kind != JobKind::LeafVerification {
            return Err(DbError::ControlledEventRequired);
        }
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        require_verification_source(&transaction, &run_id, job, attempt)?;
        let ordinal: i64 = transaction.query_row(
            "SELECT COUNT(*) + 1 FROM verification_records WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| row.get(0),
        )?;
        let id = VerificationRecordId::new(format!("{}:verification:{ordinal}", job.as_str()))?;
        transaction.execute(
            "INSERT INTO verification_records(verification_record_id,run_id,job_id,attempt_id,verifier,created_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6)",
            params![id.as_str(), run_id, job.as_str(), attempt.as_str(), verifier, unix_ms()?],
        )?;
        let event = JobEvent::VerificationSucceeded {
            verification_record: Some(id),
        };
        let state = apply_event_tx(&transaction, &run_id, job, Some(attempt), &event, None)?;
        transaction.commit()?;
        Ok(state)
    }

    // The crate-internal verifier worker introduced by the scheduler consumes this entry point.
    #[allow(dead_code)]
    pub(crate) fn fail_verification(
        &mut self,
        job: &JobId,
        source_attempt: &AttemptId,
        failure: FailureRecord,
    ) -> Result<JobState, DbError> {
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        require_verification_source(&transaction, &run_id, job, source_attempt)?;
        let event = JobEvent::VerificationFailed(failure);
        let state = apply_event_tx(
            &transaction,
            &run_id,
            job,
            Some(source_attempt),
            &event,
            None,
        )?;
        transaction.commit()?;
        Ok(state)
    }

    pub fn apply_event(&mut self, job: &JobId, event: JobEvent) -> Result<JobState, DbError> {
        match event {
            JobEvent::StartWitnessing
            | JobEvent::StartPreparing
            | JobEvent::StartProving
            | JobEvent::StartVerification => return Err(DbError::AttemptStartRequiresBegin),
            JobEvent::WitnessCompleted
            | JobEvent::PreparationCompleted
            | JobEvent::ProofProduced
            | JobEvent::ExecutionFailed(_)
            | JobEvent::ResourceExceeded(_)
            | JobEvent::Interrupted(_) => return Err(DbError::AttemptEventRequiresAttempt),
            JobEvent::DependenciesSatisfied
            | JobEvent::DependencyBlocked { .. }
            | JobEvent::VerificationSucceeded { .. }
            | JobEvent::VerificationFailed(_)
            | JobEvent::RecoveryRequeue => return Err(DbError::ControlledEventRequired),
            JobEvent::RetryExecution | JobEvent::RetryResourceExceeded => {}
        }
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let state = apply_event_tx(&transaction, &run_id, job, None, &event, None)?;
        transaction.commit()?;
        Ok(state)
    }

    pub fn apply_attempt_event(
        &mut self,
        job: &JobId,
        attempt: &AttemptId,
        event: JobEvent,
    ) -> Result<JobState, DbError> {
        if !matches!(
            event,
            JobEvent::WitnessCompleted
                | JobEvent::PreparationCompleted
                | JobEvent::ProofProduced
                | JobEvent::ExecutionFailed(_)
                | JobEvent::ResourceExceeded(_)
                | JobEvent::Interrupted(_)
        ) {
            return Err(DbError::ControlledEventRequired);
        }
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        require_unique_open_attempt(&transaction, &run_id, job, attempt)?;
        let state = apply_event_tx(&transaction, &run_id, job, Some(attempt), &event, None)?;
        transaction.commit()?;
        Ok(state)
    }

    pub fn begin_attempt(&mut self, job: &JobId, event: JobEvent) -> Result<AttemptId, DbError> {
        if !event.starts_attempt() {
            return Err(DbError::NotAttemptStart);
        }
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (kind_raw, count): (String, i64) = transaction
            .query_row(
                "SELECT kind,attempt_count FROM jobs WHERE run_id=?1 AND job_id=?2",
                params![run_id, job.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?
            .ok_or_else(|| DbError::JobNotFound(job.clone()))?;
        let kind = parse_kind(&kind_raw)?;
        if !start_matches_kind(kind, &event) {
            return Err(DbError::WrongStartForJobKind {
                kind,
                event: event.name(),
            });
        }
        let open_attempts: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM attempts WHERE run_id=?1 AND job_id=?2 AND finished_at_unix_ms IS NULL",
            params![run_id, job.as_str()],
            |row| row.get(0),
        )?;
        if open_attempts != 0 {
            return Err(DbError::OpenAttemptConflict { job: job.clone() });
        }
        let count = u32::try_from(count).map_err(|_| DbError::UnknownEnum {
            column: "attempt_count",
            value: count.to_string(),
        })?;
        let next_count = count
            .checked_add(1)
            .ok_or_else(|| DbError::AttemptCountExhausted { job: job.clone() })?;
        let attempt = AttemptId::new(format!("{}:attempt:{next_count}", job.as_str()))?;
        let next = current_state(&transaction, &run_id, job)?.transition(event.clone())?;
        let now = unix_ms()?;
        transaction.execute(
            "INSERT INTO attempts(attempt_id,run_id,job_id,state,started_at_unix_ms) VALUES(?1,?2,?3,?4,?5)",
            params![attempt.as_str(), run_id, job.as_str(), next.as_str(), now],
        )?;
        apply_event_tx(
            &transaction,
            &run_id,
            job,
            Some(&attempt),
            &event,
            Some(next_count),
        )?;
        transaction.commit()?;
        Ok(attempt)
    }

    pub fn refresh_readiness(&mut self) -> Result<usize, DbError> {
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let candidates = {
            let mut statement = transaction.prepare(
                "SELECT job_id,state FROM jobs WHERE run_id=?1 AND state IN ('pending','requeued') ORDER BY job_id",
            )?;
            let rows = statement.query_map([&run_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        let mut ready = 0;
        for (raw_id, raw_state) in candidates {
            let job = JobId::new(raw_id)?;
            parse_state(&raw_state)?;
            let predecessors = predecessor_states(&transaction, &run_id, &job)?;
            if predecessors
                .iter()
                .all(|(_, state)| *state == JobState::Verified)
            {
                apply_event_tx(
                    &transaction,
                    &run_id,
                    &job,
                    None,
                    &JobEvent::DependenciesSatisfied,
                    None,
                )?;
                ready += 1;
            } else if let Some((predecessor, _)) = predecessors
                .iter()
                .find(|(_, predecessor_state)| predecessor_state.is_permanent_failure())
            {
                apply_event_tx(
                    &transaction,
                    &run_id,
                    &job,
                    None,
                    &JobEvent::DependencyBlocked {
                        predecessor: predecessor.clone(),
                    },
                    None,
                )?;
            }
        }
        transaction.commit()?;
        Ok(ready)
    }

    pub fn recover_interrupted(&mut self) -> Result<usize, DbError> {
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let active = {
            let mut statement = transaction.prepare(
                "SELECT a.attempt_id,a.job_id,j.state FROM attempts a JOIN jobs j ON j.job_id=a.job_id WHERE a.run_id=?1 AND a.finished_at_unix_ms IS NULL ORDER BY a.attempt_id",
            )?;
            let rows = statement.query_map([&run_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        let mut recovered = 0;
        for (attempt_raw, job_raw, state_raw) in active {
            let state = parse_state(&state_raw)?;
            if !state.is_active() {
                continue;
            }
            let attempt = AttemptId::new(attempt_raw)?;
            let job = JobId::new(job_raw)?;
            let interruption = FailureRecord::new(
                FailureKind::Interrupted,
                FailureStage::Scheduler,
                FailureCode::SchedulerRestart,
            );
            apply_event_tx(
                &transaction,
                &run_id,
                &job,
                Some(&attempt),
                &JobEvent::Interrupted(interruption),
                None,
            )?;
            apply_event_tx(
                &transaction,
                &run_id,
                &job,
                Some(&attempt),
                &JobEvent::RecoveryRequeue,
                None,
            )?;
            transaction.execute(
                "UPDATE attempts SET state='interrupted',finished_at_unix_ms=?1 WHERE attempt_id=?2 AND finished_at_unix_ms IS NULL",
                params![unix_ms()?, attempt.as_str()],
            )?;
            recovered += 1;
        }
        transaction.commit()?;
        Ok(recovered)
    }

    fn require_job_in_run(&self, job: &JobId) -> Result<(), DbError> {
        let present: bool = self.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM jobs WHERE run_id=?1 AND job_id=?2)",
            params![self.run_id, job.as_str()],
            |row| row.get(0),
        )?;
        if present {
            Ok(())
        } else {
            Err(DbError::JobNotFound(job.clone()))
        }
    }
}

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
) -> Result<Vec<(JobId, JobState)>, DbError> {
    let mut statement = transaction.prepare(
        "SELECT j.job_id,j.state FROM dependencies d JOIN jobs j ON j.job_id=d.predecessor_job_id WHERE d.run_id=?1 AND d.successor_job_id=?2 ORDER BY j.job_id",
    )?;
    let rows = statement.query_map(params![run_id, successor.as_str()], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    rows.map(|row| {
        let (id, state) = row?;
        Ok((JobId::new(id)?, parse_state(&state)?))
    })
    .collect()
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Write};

    use super::*;
    use crate::{
        AggregationArity, ArtifactMetadata, ArtifactRole, ArtifactStore, ContentStore, KeyMetadata,
        KeyRole, KeyStore, ObjectMetadata, TrustedKeyIdentity, TrustedProofIdentity,
        VerifiedArtifact, VerifiedKey,
    };
    use zkie_types::{ExecutionBackendId, ProofFlavorId};

    fn test_db(label: &str) -> (RunDb, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "zkie-runtime-unit-{label}-{}-{:?}.sqlite",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_file(&path);
        (RunDb::open(&path, label).unwrap(), path)
    }

    fn ready_job(db: &mut RunDb, logical_job_id: &str, kind: JobKind) -> JobId {
        let job = db.insert_job(logical_job_id, kind).unwrap();
        db.refresh_readiness().unwrap();
        job
    }

    fn digest(bytes: &[u8]) -> Digest32 {
        Digest32::new(*blake3::hash(bytes).as_bytes())
    }

    fn verified_artifact(
        root: &Path,
        run_id: &str,
        job: &JobId,
        attempt: &AttemptId,
    ) -> VerifiedArtifact {
        let bytes = b"verified proof";
        let identity = TrustedProofIdentity::new(
            ArtifactRole::LeafProof,
            JobKind::LeafProof,
            19,
            AggregationArity::NotApplicable,
            run_id,
            job,
            attempt,
            digest(bytes),
            digest(b"artifact-manifest"),
            digest(b"run-identity"),
            digest(b"public-statement"),
            digest(b"circuit"),
            digest(b"vk"),
            digest(b"srs"),
            digest(b"shard-identity"),
            digest(b"witness-artifact"),
            ProofFlavorId::parse("halo2-kzg-v1").unwrap(),
            ExecutionBackendId::parse("halo2-cpu").unwrap(),
        )
        .unwrap();
        let metadata = ObjectMetadata::Artifact(
            ArtifactMetadata::new_proof(
                "proof",
                digest(bytes),
                bytes.len() as u64,
                identity.digest(),
            )
            .unwrap(),
        );
        let store = ArtifactStore::open(root).unwrap();
        let mut staged = store.stage(&metadata).unwrap();
        staged.write_all(bytes).unwrap();
        let published = store.publish(store.validate(staged).unwrap()).unwrap();
        VerifiedArtifact::attest(identity, published).unwrap()
    }

    fn verified_key(root: &Path, run_id: &str, job: &JobId, attempt: &AttemptId) -> VerifiedKey {
        let bytes = b"verified key material";
        let base_metadata = KeyMetadata::new(
            digest(bytes),
            bytes.len() as u64,
            digest(b"srs"),
            ProofFlavorId::parse("halo2-kzg-v1").unwrap(),
            digest(b"circuit"),
            19,
            4,
        )
        .unwrap();
        let identity = TrustedKeyIdentity::new(
            KeyRole::ProvingAndVerifyingKey,
            JobKind::Prepare,
            run_id,
            job,
            attempt,
            base_metadata.clone(),
        )
        .unwrap();
        let metadata = ObjectMetadata::Key(base_metadata.bind_identity(identity.digest()));
        let store = KeyStore::open(root).unwrap();
        let mut staged = store.stage(&metadata).unwrap();
        staged.write_all(bytes).unwrap();
        let published = store.publish(store.validate(staged).unwrap()).unwrap();
        VerifiedKey::attest(identity, published).unwrap()
    }

    #[test]
    fn verified_artifact_capability_is_bound_and_committed_atomically() {
        let (mut db, path) = test_db("verified-artifact");
        let job = ready_job(&mut db, "leaf", JobKind::LeafProof);
        let attempt = db.begin_attempt(&job, JobEvent::StartProving).unwrap();
        db.apply_attempt_event(&job, &attempt, JobEvent::ProofProduced)
            .unwrap();
        let store_root = path.with_extension("objects");
        let wrong = verified_artifact(&store_root, "wrong-run", &job, &attempt);

        assert!(db.commit_verified_artifact(&wrong).is_err());
        assert_eq!(db.job(&job).unwrap().state, JobState::Proved);
        assert_eq!(db.artifact_for_job(&job).unwrap(), None);

        let verified = verified_artifact(&store_root, "verified-artifact", &job, &attempt);
        db.commit_verified_artifact(&verified).unwrap();
        assert_eq!(db.job(&job).unwrap().state, JobState::Verified);
        assert!(db.artifact_for_job(&job).unwrap().is_some());

        drop(db);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(store_root);
    }

    #[test]
    fn proof_identity_and_locator_survive_restart_and_reopen() {
        let (mut db, path) = test_db("artifact-restart");
        let job = ready_job(&mut db, "leaf", JobKind::LeafProof);
        let attempt = db.begin_attempt(&job, JobEvent::StartProving).unwrap();
        db.apply_attempt_event(&job, &attempt, JobEvent::ProofProduced)
            .unwrap();
        let store_root = path.with_extension("objects");
        let verified = verified_artifact(&store_root, "artifact-restart", &job, &attempt);
        db.commit_verified_artifact(&verified).unwrap();
        let before = db.artifact_for_job(&job).unwrap().unwrap();
        drop(db);
        let moved_store_root = path.with_extension("objects-moved");
        fs::rename(&store_root, &moved_store_root).unwrap();

        let db = RunDb::open(&path, "artifact-restart").unwrap();
        let store = ArtifactStore::open(&moved_store_root).unwrap();
        let after = db.artifact_for_job(&job).unwrap().unwrap();
        assert_eq!(after, before);
        let mut file = db.reopen_artifact_for_job(&store, &job).unwrap().unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"verified proof");

        drop(db);
        let _ = fs::remove_file(path);
        let _ = fs::remove_dir_all(moved_store_root);
    }

    #[test]
    fn replacing_named_objects_directory_makes_database_commit_fail_closed() {
        let (mut db, path) = test_db("objects-replaced");
        let job = ready_job(&mut db, "leaf", JobKind::LeafProof);
        let attempt = db.begin_attempt(&job, JobEvent::StartProving).unwrap();
        db.apply_attempt_event(&job, &attempt, JobEvent::ProofProduced)
            .unwrap();
        let store_root = path.with_extension("objects");
        let verified = verified_artifact(&store_root, "objects-replaced", &job, &attempt);
        fs::rename(store_root.join("objects"), store_root.join("objects-old")).unwrap();
        fs::create_dir(store_root.join("objects")).unwrap();

        assert!(matches!(
            db.commit_verified_artifact(&verified),
            Err(DbError::Store(StoreError::StoreLocatorChanged))
        ));
        assert_eq!(db.job(&job).unwrap().state, JobState::Proved);
        assert_eq!(db.artifact_for_job(&job).unwrap(), None);

        drop(db);
        let _ = fs::remove_file(path);
        let _ = fs::remove_dir_all(store_root);
    }

    #[test]
    fn key_publication_persists_full_identity_and_requires_matching_store_on_reopen() {
        let (mut db, path) = test_db("key-restart");
        let job = ready_job(&mut db, "prepare", JobKind::Prepare);
        let attempt = db.begin_attempt(&job, JobEvent::StartPreparing).unwrap();
        db.apply_attempt_event(&job, &attempt, JobEvent::PreparationCompleted)
            .unwrap();
        let store_root = path.with_extension("keys");
        let verified = verified_key(&store_root, "key-restart", &job, &attempt);
        db.commit_verified_key(&verified).unwrap();
        let before = db.key_for_job(&job).unwrap().unwrap();
        drop(db);

        let db = RunDb::open(&path, "key-restart").unwrap();
        assert_eq!(db.key_for_job(&job).unwrap().unwrap(), before);
        let store = KeyStore::open(&store_root).unwrap();
        let mut file = db.reopen_key_for_job(&store, &job).unwrap().unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"verified key material");

        let other_store = KeyStore::open(path.with_extension("wrong-keys")).unwrap();
        assert!(db.reopen_key_for_job(&other_store, &job).is_err());
        db.connection
            .execute(
                "UPDATE key_artifacts SET identity_digest=?1 WHERE run_id=?2 AND job_id=?3",
                params![
                    digest(b"wrong-key-identity").to_string(),
                    "key-restart",
                    job.as_str()
                ],
            )
            .unwrap();
        assert!(db.reopen_key_for_job(&store, &job).is_err());

        drop(db);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(store_root);
        let _ = fs::remove_dir_all(path.with_extension("wrong-keys"));
    }

    #[test]
    fn tampered_persisted_identity_is_rejected_on_reopen() {
        let (mut db, path) = test_db("identity-tamper");
        let job = ready_job(&mut db, "leaf", JobKind::LeafProof);
        let attempt = db.begin_attempt(&job, JobEvent::StartProving).unwrap();
        db.apply_attempt_event(&job, &attempt, JobEvent::ProofProduced)
            .unwrap();
        let store_root = path.with_extension("objects");
        let verified = verified_artifact(&store_root, "identity-tamper", &job, &attempt);
        db.commit_verified_artifact(&verified).unwrap();
        db.connection
            .execute(
                "UPDATE artifacts SET identity_digest=?1 WHERE run_id=?2 AND job_id=?3",
                params![
                    digest(b"wrong-identity").to_string(),
                    "identity-tamper",
                    job.as_str()
                ],
            )
            .unwrap();
        let store = ArtifactStore::open(&store_root).unwrap();
        assert!(db.reopen_artifact_for_job(&store, &job).is_err());

        drop(db);
        let _ = fs::remove_file(path);
        let _ = fs::remove_dir_all(store_root);
    }

    #[test]
    fn empty_v1_database_migrates_atomically_but_legacy_artifacts_require_reverification() {
        let empty_path =
            std::env::temp_dir().join(format!("zkie-v1-empty-{}.sqlite", std::process::id()));
        let legacy_path =
            std::env::temp_dir().join(format!("zkie-v1-legacy-{}.sqlite", std::process::id()));
        for path in [&empty_path, &legacy_path] {
            let _ = fs::remove_file(path);
            let connection = Connection::open(path).unwrap();
            connection.execute_batch(SCHEMA_V1).unwrap();
            connection
                .execute(
                    "INSERT INTO schema_version(version,applied_at_unix_ms) VALUES(1,0)",
                    [],
                )
                .unwrap();
        }
        let migrated = RunDb::open(&empty_path, "migrated").unwrap();
        assert_eq!(migrated.schema_version().unwrap(), 2);
        drop(migrated);

        let legacy = Connection::open(&legacy_path).unwrap();
        legacy
            .execute(
                "INSERT INTO runs(run_id,created_at_unix_ms) VALUES('legacy',0)",
                [],
            )
            .unwrap();
        legacy.execute("INSERT INTO jobs(job_id,run_id,logical_job_id,kind,state,created_at_unix_ms,updated_at_unix_ms) VALUES('legacy:leaf','legacy','leaf','leaf-proof','verified',0,0)", []).unwrap();
        legacy.execute("INSERT INTO artifacts(artifact_id,run_id,logical_job_id,digest,path,created_at_unix_ms) VALUES('proof','legacy','leaf',?1,'/untrusted/legacy/path',0)", params![digest(b"legacy").to_string()]).unwrap();
        drop(legacy);
        assert!(matches!(
            RunDb::open(&legacy_path, "legacy"),
            Err(DbError::LegacyArtifactsRequireReverification)
        ));
        let unchanged = Connection::open(&legacy_path).unwrap();
        let version: i64 = unchanged
            .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(version, 1);

        let _ = fs::remove_file(empty_path);
        let _ = fs::remove_file(legacy_path);
    }

    #[test]
    fn controlled_verification_records_and_transitions_atomically() {
        let (mut db, path) = test_db("atomic-verification");
        let job = ready_job(&mut db, "leaf", JobKind::LeafVerification);
        let attempt = db.begin_attempt(&job, JobEvent::StartVerification).unwrap();

        assert!(matches!(
            db.complete_verification(&job, &attempt, "bad\0verifier"),
            Err(DbError::State(_))
        ));
        assert_eq!(db.job(&job).unwrap().state, JobState::Verifying);
        assert_eq!(db.events(&job).unwrap().len(), 2);

        db.complete_verification(&job, &attempt, "independent-verifier")
            .unwrap();
        assert_eq!(db.job(&job).unwrap().state, JobState::Verified);
        assert!(db.attempt(&attempt).unwrap().finished_at_unix_ms.is_some());
        assert_eq!(db.events(&job).unwrap().len(), 3);

        drop(db);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn proof_verification_record_binds_the_finished_source_attempt() {
        let (mut db, path) = test_db("proof-verification");
        let job = ready_job(&mut db, "leaf", JobKind::LeafProof);
        let attempt = db.begin_attempt(&job, JobEvent::StartProving).unwrap();
        db.apply_attempt_event(&job, &attempt, JobEvent::ProofProduced)
            .unwrap();

        assert!(matches!(
            db.complete_verification(&job, &attempt, "independent-proof-verifier"),
            Err(DbError::ControlledEventRequired)
        ));
        assert_eq!(db.job(&job).unwrap().state, JobState::Proved);

        drop(db);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn every_job_kind_has_a_complete_success_path_to_verified() {
        let (mut db, path) = test_db("all-success-paths");
        let cases = [
            (
                JobKind::Witness,
                JobEvent::StartWitnessing,
                Some(JobEvent::WitnessCompleted),
                JobState::WitnessReady,
            ),
            (
                JobKind::Prepare,
                JobEvent::StartPreparing,
                Some(JobEvent::PreparationCompleted),
                JobState::Prepared,
            ),
            (
                JobKind::LeafProof,
                JobEvent::StartProving,
                Some(JobEvent::ProofProduced),
                JobState::Proved,
            ),
            (
                JobKind::LeafVerification,
                JobEvent::StartVerification,
                None,
                JobState::Verifying,
            ),
            (
                JobKind::NativeAggregate,
                JobEvent::StartVerification,
                None,
                JobState::Verifying,
            ),
        ];

        for (index, (kind, start, completion, intermediate)) in cases.into_iter().enumerate() {
            let job = ready_job(&mut db, &format!("job-{index}"), kind);
            let attempt = db.begin_attempt(&job, start).unwrap();
            if let Some(completion) = completion {
                db.apply_attempt_event(&job, &attempt, completion).unwrap();
            }
            assert_eq!(db.job(&job).unwrap().state, intermediate);
            if kind == JobKind::LeafVerification {
                db.complete_verification(&job, &attempt, "independent-verifier")
                    .unwrap();
                assert_eq!(db.job(&job).unwrap().state, JobState::Verified);
            } else {
                assert!(matches!(
                    db.complete_verification(&job, &attempt, "independent-verifier"),
                    Err(DbError::ControlledEventRequired)
                ));
                assert_eq!(db.job(&job).unwrap().state, intermediate);
            }
        }

        drop(db);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn every_job_kind_can_persist_independent_verification_failure() {
        let (mut db, path) = test_db("all-verification-failures");
        let cases = [
            (
                JobKind::Witness,
                JobEvent::StartWitnessing,
                Some(JobEvent::WitnessCompleted),
            ),
            (
                JobKind::Prepare,
                JobEvent::StartPreparing,
                Some(JobEvent::PreparationCompleted),
            ),
            (
                JobKind::LeafProof,
                JobEvent::StartProving,
                Some(JobEvent::ProofProduced),
            ),
            (JobKind::LeafVerification, JobEvent::StartVerification, None),
            (JobKind::NativeAggregate, JobEvent::StartVerification, None),
        ];

        for (index, (kind, start, completion)) in cases.into_iter().enumerate() {
            let job = ready_job(&mut db, &format!("job-{index}"), kind);
            let attempt = db.begin_attempt(&job, start).unwrap();
            if let Some(completion) = completion {
                db.apply_attempt_event(&job, &attempt, completion).unwrap();
            }
            let failure = FailureRecord::new(
                FailureKind::Verification,
                FailureStage::Verify,
                FailureCode::InvalidResult,
            );
            db.fail_verification(&job, &attempt, failure).unwrap();
            assert_eq!(db.job(&job).unwrap().state, JobState::VerificationFailed);
        }

        drop(db);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn verification_rejects_a_non_latest_finished_source_attempt() {
        let (mut db, path) = test_db("latest-verification-source");
        let job = ready_job(&mut db, "leaf", JobKind::LeafProof);
        let source = db.begin_attempt(&job, JobEvent::StartProving).unwrap();
        db.apply_attempt_event(&job, &source, JobEvent::ProofProduced)
            .unwrap();
        let store_root = path.with_extension("latest-artifact");
        let verified = verified_artifact(&store_root, "latest-verification-source", &job, &source);
        let later = AttemptId::new("latest-verification-source:leaf:attempt:2").unwrap();
        db.connection
            .execute(
                "INSERT INTO attempts(attempt_id,run_id,job_id,state,started_at_unix_ms,finished_at_unix_ms) VALUES(?1,?2,?3,'proved',2,2)",
                params![later.as_str(), db.run_id, job.as_str()],
            )
            .unwrap();

        assert!(matches!(
            db.commit_verified_artifact(&verified),
            Err(DbError::InvalidVerificationSource { .. })
        ));
        assert_eq!(db.job(&job).unwrap().state, JobState::Proved);

        drop(db);
        let _ = fs::remove_file(path);
        let _ = fs::remove_dir_all(store_root);
    }
}
