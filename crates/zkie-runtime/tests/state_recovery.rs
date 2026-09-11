use std::fs;
use std::path::PathBuf;

use rusqlite::{params, Connection};
use zkie_runtime::{
    DbError, FailureCode, FailureKind, FailureRecord, FailureStage, JobEvent, JobId, JobKind,
    JobState, RunDb, VerificationRecordId, SCHEMA_VERSION,
};

fn temp_db(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-runtime-{label}-{}-{}.sqlite",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = fs::remove_file(&path);
    path
}

fn failure(kind: FailureKind, stage: FailureStage) -> FailureRecord {
    FailureRecord::new(kind, stage, FailureCode::WorkerExit)
}

fn insert_ready(db: &mut RunDb, logical_job_id: &str, kind: JobKind) -> JobId {
    let job = db.insert_job(logical_job_id, kind).unwrap();
    db.refresh_readiness().unwrap();
    assert_eq!(db.job(&job).unwrap().state, JobState::Ready);
    job
}

fn seed_verified_predecessor(db: &mut RunDb, path: &PathBuf, logical_job_id: &str) -> JobId {
    let job = db
        .insert_job(logical_job_id, JobKind::LeafVerification)
        .unwrap();
    let raw = Connection::open(path).unwrap();
    raw.execute(
        "UPDATE jobs SET state='verified' WHERE job_id=?1",
        [job.as_str()],
    )
    .unwrap();
    job
}

fn seed_verification_failed(db: &mut RunDb, path: &PathBuf, logical_job_id: &str) -> JobId {
    let job = db
        .insert_job(logical_job_id, JobKind::LeafVerification)
        .unwrap();
    let raw = Connection::open(path).unwrap();
    raw.execute(
        "UPDATE jobs SET state='verification-failed' WHERE job_id=?1",
        [job.as_str()],
    )
    .unwrap();
    job
}

#[test]
fn state_machine_defines_every_legal_pipeline_and_retry_transition() {
    let verification = VerificationRecordId::new("verification-1").unwrap();
    let cases = [
        (
            JobState::Pending,
            JobEvent::DependenciesSatisfied,
            JobState::Ready,
        ),
        (
            JobState::Requeued,
            JobEvent::DependenciesSatisfied,
            JobState::Ready,
        ),
        (
            JobState::Ready,
            JobEvent::StartWitnessing,
            JobState::Witnessing,
        ),
        (
            JobState::Witnessing,
            JobEvent::WitnessCompleted,
            JobState::WitnessReady,
        ),
        (
            JobState::WitnessReady,
            JobEvent::StartPreparing,
            JobState::Preparing,
        ),
        (
            JobState::Ready,
            JobEvent::StartPreparing,
            JobState::Preparing,
        ),
        (
            JobState::Preparing,
            JobEvent::PreparationCompleted,
            JobState::Prepared,
        ),
        (
            JobState::Preparing,
            JobEvent::StartProving,
            JobState::Proving,
        ),
        (JobState::Ready, JobEvent::StartProving, JobState::Proving),
        (JobState::Proving, JobEvent::ProofProduced, JobState::Proved),
        (
            JobState::WitnessReady,
            JobEvent::StartVerification,
            JobState::Verifying,
        ),
        (
            JobState::Preparing,
            JobEvent::StartVerification,
            JobState::Verifying,
        ),
        (
            JobState::Proved,
            JobEvent::StartVerification,
            JobState::Verifying,
        ),
        (
            JobState::Ready,
            JobEvent::StartVerification,
            JobState::Verifying,
        ),
        (
            JobState::Verifying,
            JobEvent::VerificationSucceeded {
                verification_record: Some(verification.clone()),
            },
            JobState::Verified,
        ),
        (
            JobState::ExecutionFailed,
            JobEvent::RetryExecution,
            JobState::Requeued,
        ),
        (
            JobState::ResourceExceeded,
            JobEvent::RetryResourceExceeded,
            JobState::Requeued,
        ),
        (
            JobState::Interrupted,
            JobEvent::RecoveryRequeue,
            JobState::Requeued,
        ),
    ];
    for (from, event, expected) in cases {
        assert_eq!(from.transition(event).unwrap(), expected);
    }

    for active in [
        JobState::Witnessing,
        JobState::Preparing,
        JobState::Proving,
        JobState::Verifying,
    ] {
        assert_eq!(
            active
                .transition(JobEvent::Interrupted(failure(
                    FailureKind::Interrupted,
                    FailureStage::Scheduler,
                )))
                .unwrap(),
            JobState::Interrupted
        );
        assert_eq!(
            active
                .transition(JobEvent::ResourceExceeded(failure(
                    FailureKind::ResourceExceeded,
                    FailureStage::Prove,
                )))
                .unwrap(),
            JobState::ResourceExceeded
        );
    }
}

#[test]
fn terminal_failures_and_unrecorded_verification_are_rejected() {
    assert!(JobState::VerificationFailed
        .transition(JobEvent::RetryExecution)
        .is_err());
    assert!(JobState::Blocked
        .transition(JobEvent::RetryExecution)
        .is_err());
    assert!(JobState::Blocked
        .transition(JobEvent::DependenciesSatisfied)
        .is_err());
    assert!(JobState::Blocked
        .transition(JobEvent::DependencyBlocked {
            predecessor: zkie_runtime::JobId::new("failed-parent").unwrap(),
        })
        .is_err());
    assert!(JobState::Verified
        .transition(JobEvent::RetryExecution)
        .is_err());
    assert!(JobState::Proved
        .transition(JobEvent::VerificationSucceeded {
            verification_record: None,
        })
        .is_err());
    assert!(JobState::Verifying
        .transition(JobEvent::VerificationSucceeded {
            verification_record: None,
        })
        .is_err());
}

#[test]
fn readiness_waits_for_live_predecessors_blocks_on_permanent_failure_and_requires_all_verified() {
    let path = temp_db("readiness");
    let mut db = RunDb::open(&path, "run-ready").unwrap();
    let first = insert_ready(&mut db, "first", JobKind::LeafVerification);
    let first_attempt = db
        .begin_attempt(&first, JobEvent::StartVerification)
        .unwrap();
    db.handle_attempt_failure(
        &first,
        &first_attempt,
        failure(FailureKind::Execution, FailureStage::Verify),
        None,
    )
    .unwrap();
    let second = seed_verified_predecessor(&mut db, &path, "second");
    let downstream = db
        .insert_job("downstream", JobKind::NativeAggregate)
        .unwrap();
    db.add_dependency(&first, &downstream).unwrap();
    db.add_dependency(&second, &downstream).unwrap();

    // The retryable root is admitted again, while its dependent stays pending.
    assert_eq!(db.refresh_readiness().unwrap(), 1);
    let waiting = db.job(&downstream).unwrap();
    assert_eq!(waiting.state, JobState::Pending);
    assert_eq!(waiting.blocking_predecessor_id, None);

    let raw = Connection::open(&path).unwrap();
    raw.execute(
        "UPDATE jobs SET state='verification-failed' WHERE job_id=?1",
        [first.as_str()],
    )
    .unwrap();
    drop(raw);
    assert_eq!(db.refresh_readiness().unwrap(), 0);
    let blocked = db.job(&downstream).unwrap();
    assert_eq!(blocked.state, JobState::Blocked);
    assert_eq!(blocked.blocking_predecessor_id, Some(first.clone()));
    assert_eq!(db.refresh_readiness().unwrap(), 0);
    assert!(matches!(
        db.apply_event(&downstream, JobEvent::DependenciesSatisfied),
        Err(DbError::ControlledEventRequired)
    ));
    assert_eq!(db.job(&downstream).unwrap().state, JobState::Blocked);

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn all_verified_predecessors_unlock_exactly_once() {
    let path = temp_db("unlock");
    let mut db = RunDb::open(&path, "run-unlock").unwrap();
    let first = seed_verified_predecessor(&mut db, &path, "first");
    let second = seed_verified_predecessor(&mut db, &path, "second");
    let downstream = db
        .insert_job("downstream", JobKind::NativeAggregate)
        .unwrap();
    db.add_dependency(&first, &downstream).unwrap();
    db.add_dependency(&second, &downstream).unwrap();

    assert_eq!(db.refresh_readiness().unwrap(), 1);
    assert_eq!(db.refresh_readiness().unwrap(), 0);
    assert_eq!(db.job(&downstream).unwrap().state, JobState::Ready);

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn failure_columns_and_attempt_count_are_independent_and_sanitized() {
    let path = temp_db("failure");
    let mut db = RunDb::open(&path, "run-failure").unwrap();
    let job = insert_ready(&mut db, "leaf", JobKind::LeafProof);
    let attempt = db.begin_attempt(&job, JobEvent::StartProving).unwrap();
    db.handle_attempt_failure(
        &job,
        &attempt,
        failure(FailureKind::Execution, FailureStage::Prove),
        None,
    )
    .unwrap();
    let row = db.job(&job).unwrap();
    assert_eq!(row.state, JobState::Requeued);
    assert_eq!(row.attempt_count, 1);
    assert_eq!(row.failure_kind, Some(FailureKind::Execution));
    assert_eq!(row.failure_stage, Some(FailureStage::Prove));
    assert_eq!(row.failure_code, Some(FailureCode::WorkerExit));
    assert_eq!(row.failure_summary.as_deref(), Some("execution failed"));
    assert_eq!(
        db.attempt(&attempt).unwrap().state,
        JobState::ExecutionFailed
    );

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn restart_recovery_is_idempotent_and_preserves_terminal_jobs() {
    let path = temp_db("restart");
    let active;
    let verified;
    let failed;
    {
        let mut db = RunDb::open(&path, "run-restart").unwrap();
        active = insert_ready(&mut db, "active", JobKind::LeafProof);
        verified = seed_verified_predecessor(&mut db, &path, "verified");
        failed = seed_verification_failed(&mut db, &path, "failed");
        db.begin_attempt(&active, JobEvent::StartProving).unwrap();
    }

    let mut reopened = RunDb::open(&path, "run-restart").unwrap();
    assert_eq!(reopened.recover_interrupted().unwrap(), 1);
    assert_eq!(reopened.job(&active).unwrap().state, JobState::Requeued);
    assert_eq!(
        reopened
            .events(&active)
            .unwrap()
            .iter()
            .map(|event| event.to_state)
            .collect::<Vec<_>>(),
        vec![
            JobState::Ready,
            JobState::Proving,
            JobState::Interrupted,
            JobState::Requeued
        ]
    );
    assert_eq!(reopened.job(&verified).unwrap().state, JobState::Verified);
    assert_eq!(
        reopened.job(&failed).unwrap().state,
        JobState::VerificationFailed
    );
    assert_eq!(reopened.recover_interrupted().unwrap(), 0);

    drop(reopened);
    let _ = fs::remove_file(path);
}

#[test]
fn schema_enables_durability_guards_rejects_duplicates_and_unknown_enums() {
    let path = temp_db("schema");
    let mut db = RunDb::open(&path, "run-schema").unwrap();
    let job = db.insert_job("same", JobKind::Witness).unwrap();
    assert!(db.insert_job("same", JobKind::Witness).is_err());
    let settings = db.sqlite_settings().unwrap();
    assert_eq!(settings.journal_mode, "wal");
    assert!(settings.foreign_keys);
    assert_eq!(settings.synchronous, "FULL");
    assert_eq!(db.schema_version().unwrap(), SCHEMA_VERSION);
    db.refresh_readiness().unwrap();
    let attempt = db.begin_attempt(&job, JobEvent::StartWitnessing).unwrap();
    drop(db);

    let raw = Connection::open(&path).unwrap();
    let tables = [
        "runs",
        "jobs",
        "dependencies",
        "attempts",
        "state_events",
        "artifacts",
        "resource_samples",
        "resource_history",
        "schema_version",
        "verification_records",
    ];
    for table in tables {
        let count: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [table],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "missing table {table}");
    }
    let successor_index: i64 = raw
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='dependencies_by_successor'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(successor_index, 1);
    let insert_artifact = "INSERT INTO artifacts(
        artifact_id,run_id,job_id,logical_job_id,attempt_id,object_digest,content_digest,
        store_identity_digest,identity_digest,attestation_digest,artifact_role,job_kind,circuit_k,
        aggregation_arity_tag,aggregation_arity,artifact_manifest_digest,run_identity_digest,
        public_statement_digest,circuit_digest,verifying_key_digest,srs_source_digest,
        shard_identity_digest,witness_artifact_digest,proof_flavor,execution_backend,created_at_unix_ms
    ) VALUES(?1,'run-schema',?2,'same',?3,'digest-1','content','store','identity','attestation',
        'witness','witness',1,0,NULL,'manifest','run','statement','circuit','vk','srs','shard',
        'witness','flavor','backend',1)";
    raw.execute(
        insert_artifact,
        params!["a1", job.as_str(), attempt.as_str()],
    )
    .unwrap();
    assert!(raw
        .execute(
            insert_artifact,
            params!["a2", job.as_str(), attempt.as_str()]
        )
        .is_err());
    raw.execute(
        "INSERT INTO runs(run_id,created_at_unix_ms) VALUES('run-schema-two',1)",
        [],
    )
    .unwrap();
    raw.execute(
        "INSERT INTO jobs(job_id,run_id,logical_job_id,kind,state,created_at_unix_ms,updated_at_unix_ms) VALUES('run-schema-two:same','run-schema-two','same','witness','witnessing',1,1)",
        [],
    )
    .unwrap();
    raw.execute(
        "INSERT INTO attempts(attempt_id,run_id,job_id,state,started_at_unix_ms) VALUES('run-schema-two:same:attempt:1','run-schema-two','run-schema-two:same','witnessing',1)",
        [],
    )
    .unwrap();
    let cross_run_insert = insert_artifact.replace("'run-schema'", "'run-schema-two'");
    raw.execute(
        &cross_run_insert,
        params!["a1", "run-schema-two:same", "run-schema-two:same:attempt:1"],
    )
    .unwrap();
    raw.execute(
        "UPDATE jobs SET state='future-state' WHERE job_id=?1",
        [job.as_str()],
    )
    .unwrap();
    drop(raw);

    let reopened = RunDb::open(&path, "run-schema").unwrap();
    assert!(matches!(
        reopened.job(&job),
        Err(DbError::UnknownEnum {
            column: "job_state",
            ..
        })
    ));
    drop(reopened);
    let _ = fs::remove_file(path);
}

#[test]
fn attempt_count_rejects_values_above_u32_and_exhaustion_without_wrapping() {
    let path = temp_db("attempt-limit");
    let mut db = RunDb::open(&path, "run-attempt-limit").unwrap();
    let job = insert_ready(&mut db, "leaf", JobKind::LeafProof);

    let raw = Connection::open(&path).unwrap();
    assert!(raw
        .execute(
            "UPDATE jobs SET attempt_count=?1 WHERE job_id=?2",
            rusqlite::params![i64::from(u32::MAX) + 1, job.as_str()],
        )
        .is_err());
    raw.execute(
        "UPDATE jobs SET attempt_count=?1 WHERE job_id=?2",
        rusqlite::params![i64::from(u32::MAX), job.as_str()],
    )
    .unwrap();
    drop(raw);

    assert!(matches!(
        db.begin_attempt(&job, JobEvent::StartProving),
        Err(DbError::AttemptCountExhausted { .. })
    ));
    assert_eq!(db.job(&job).unwrap().attempt_count, u32::MAX);

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn indexed_readiness_handles_a_wide_verified_predecessor_set() {
    let path = temp_db("wide-dag");
    let mut db = RunDb::open(&path, "run-wide-dag").unwrap();
    let predecessors = (0..128)
        .map(|index| {
            db.insert_job(&format!("predecessor-{index}"), JobKind::LeafVerification)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let raw = Connection::open(&path).unwrap();
    for predecessor in &predecessors {
        raw.execute(
            "UPDATE jobs SET state='verified' WHERE job_id=?1",
            [predecessor.as_str()],
        )
        .unwrap();
    }
    drop(raw);
    let downstream = db
        .insert_job("downstream", JobKind::NativeAggregate)
        .unwrap();
    for predecessor in &predecessors {
        db.add_dependency(predecessor, &downstream).unwrap();
    }

    assert_eq!(db.refresh_readiness().unwrap(), 1);
    assert_eq!(db.job(&downstream).unwrap().state, JobState::Ready);

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn opening_a_database_with_an_unknown_schema_version_is_rejected() {
    let path = temp_db("future-schema");
    let raw = Connection::open(&path).unwrap();
    raw.execute(
        "CREATE TABLE schema_version(version INTEGER PRIMARY KEY, applied_at_unix_ms INTEGER NOT NULL)",
        [],
    )
    .unwrap();
    raw.execute(
        "INSERT INTO schema_version(version, applied_at_unix_ms) VALUES(?1, 1)",
        [i64::from(SCHEMA_VERSION) + 1],
    )
    .unwrap();
    raw.execute("CREATE TABLE sentinel(value TEXT NOT NULL)", [])
        .unwrap();
    raw.execute("INSERT INTO sentinel(value) VALUES('unchanged')", [])
        .unwrap();
    let journal_before: String = raw
        .pragma_query_value(None, "journal_mode", |row| row.get(0))
        .unwrap();
    let schema_before: Vec<String> = {
        let mut statement = raw
            .prepare("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name")
            .unwrap();
        statement
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    drop(raw);

    assert!(matches!(
        RunDb::open(&path, "run-future-schema"),
        Err(DbError::UnknownEnum {
            column: "schema_version",
            ..
        })
    ));
    let raw = Connection::open(&path).unwrap();
    let journal_after: String = raw
        .pragma_query_value(None, "journal_mode", |row| row.get(0))
        .unwrap();
    let schema_after: Vec<String> = {
        let mut statement = raw
            .prepare("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name")
            .unwrap();
        statement
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    assert_eq!(journal_after, journal_before);
    assert_eq!(schema_after, schema_before);
    assert_eq!(
        raw.query_row("SELECT value FROM sentinel", [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        "unchanged"
    );
    drop(raw);
    let _ = fs::remove_file(path);
}

#[test]
fn failure_summaries_are_fixed_by_typed_failure_kind() {
    let cases = [
        (
            FailureKind::Execution,
            FailureCode::WorkerExit,
            "execution failed",
        ),
        (
            FailureKind::Verification,
            FailureCode::InvalidResult,
            "verification failed",
        ),
        (
            FailureKind::ResourceExceeded,
            FailureCode::ResourceLimit,
            "resource limit exceeded",
        ),
        (
            FailureKind::Dependency,
            FailureCode::PredecessorFailed,
            "dependency failed",
        ),
        (
            FailureKind::Interrupted,
            FailureCode::SchedulerRestart,
            "interrupted",
        ),
    ];
    for (kind, code, expected) in cases {
        let record = FailureRecord::new(kind, FailureStage::Scheduler, code);
        assert_eq!(record.summary(), expected);
    }
}

#[test]
fn untrusted_worker_text_is_never_written_to_persisted_failure_fields() {
    let path = temp_db("untrusted-failure-text");
    let mut db = RunDb::open(&path, "run-untrusted-failure-text").unwrap();
    let messages = [
        "client_secret=one",
        "github_token=two",
        "aws_secret_access_key=three",
        "x-api-key: four",
        "credential=five",
        "cookie=six",
        "session=seven",
        "key=eight",
    ];
    for (index, message) in messages.into_iter().enumerate() {
        let job = insert_ready(&mut db, &format!("job-{index}"), JobKind::LeafProof);
        let attempt = db.begin_attempt(&job, JobEvent::StartProving).unwrap();
        let record = FailureRecord::from_untrusted_message(
            FailureKind::Execution,
            FailureStage::Prove,
            FailureCode::WorkerExit,
            message,
        );
        db.handle_attempt_failure(&job, &attempt, record, None)
            .unwrap();
        let persisted = db.job(&job).unwrap();
        assert_eq!(persisted.failure_summary.as_deref(), Some("[REDACTED]"));
        assert_eq!(persisted.failure_code, Some(FailureCode::WorkerExit));
        assert!(!persisted.failure_summary.unwrap().contains(message));
    }
    let raw = Connection::open(&path).unwrap();
    let persisted_text: String = raw
        .query_row(
            "SELECT COALESCE(GROUP_CONCAT(value, ' '), '') FROM (SELECT failure_summary AS value FROM jobs UNION ALL SELECT failure_summary FROM state_events)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    for secret_value in [
        "one", "two", "three", "four", "five", "six", "seven", "eight",
    ] {
        assert!(!persisted_text.contains(secret_value));
    }
    drop(raw);

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn failed_schema_bootstrap_leaves_no_partial_runtime_schema() {
    let path = temp_db("failed-bootstrap");
    let raw = Connection::open(&path).unwrap();
    raw.execute("CREATE VIEW runs AS SELECT 1 AS incompatible", [])
        .unwrap();
    drop(raw);

    assert!(RunDb::open(&path, "run-failed-bootstrap").is_err());

    let raw = Connection::open(&path).unwrap();
    for name in ["schema_version", "jobs", "attempts", "state_events"] {
        let count: i64 = raw
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [name],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0, "partial table left behind: {name}");
    }
    drop(raw);
    let _ = fs::remove_file(path);
}

#[test]
fn relational_references_cannot_cross_run_boundaries() {
    let path = temp_db("cross-run");
    let (first, first_attempt) = {
        let mut db = RunDb::open(&path, "run-a").unwrap();
        let job = insert_ready(&mut db, "a", JobKind::Witness);
        let attempt = db.begin_attempt(&job, JobEvent::StartWitnessing).unwrap();
        (job, attempt)
    };
    let (second, second_attempt) = {
        let mut db = RunDb::open(&path, "run-b").unwrap();
        let job = insert_ready(&mut db, "b", JobKind::Witness);
        let attempt = db.begin_attempt(&job, JobEvent::StartWitnessing).unwrap();
        (job, attempt)
    };

    let raw = Connection::open(&path).unwrap();
    raw.pragma_update(None, "foreign_keys", true).unwrap();
    assert!(raw
        .execute(
            "INSERT INTO dependencies(run_id, predecessor_job_id, successor_job_id) VALUES('run-a', ?1, ?2)",
            [second.as_str(), first.as_str()],
        )
        .is_err());
    assert!(raw
        .execute(
            "INSERT INTO verification_records(verification_record_id,run_id,job_id,attempt_id,verifier,created_at_unix_ms) VALUES('cross-record','run-a',?1,?2,'verifier',1)",
            [first.as_str(), second_attempt.as_str()],
        )
        .is_err());
    assert!(raw
        .execute(
            "INSERT INTO state_events(run_id,job_id,attempt_id,from_state,to_state,event_type,created_at_unix_ms) VALUES('run-a',?1,?2,'witnessing','interrupted','interrupted',1)",
            [first.as_str(), second_attempt.as_str()],
        )
        .is_err());
    assert!(raw
        .execute(
            "INSERT INTO state_events(run_id,job_id,from_state,to_state,event_type,blocking_predecessor_id,created_at_unix_ms) VALUES('run-a',?1,'pending','blocked','dependency-blocked',?2,1)",
            [first.as_str(), second.as_str()],
        )
        .is_err());
    assert!(raw
        .execute(
            "UPDATE jobs SET blocking_predecessor_id=?1 WHERE run_id='run-a' AND job_id=?2",
            [second.as_str(), first.as_str()],
        )
        .is_err());
    assert!(raw
        .execute(
            "INSERT INTO resource_samples(run_id,attempt_id,observed_at_unix_ms,resident_bytes) VALUES('run-a',?1,1,1)",
            [second_attempt.as_str()],
        )
        .is_err());
    raw.execute(
        "INSERT INTO verification_records(verification_record_id,run_id,job_id,attempt_id,verifier,created_at_unix_ms) VALUES('run-b-record','run-b',?1,?2,'verifier',1)",
        [second.as_str(), second_attempt.as_str()],
    )
    .unwrap();
    assert!(raw
        .execute(
            "INSERT INTO state_events(run_id,job_id,from_state,to_state,event_type,verification_record_id,created_at_unix_ms) VALUES('run-a',?1,'verifying','verified','verification-succeeded','run-b-record',1)",
            [first.as_str()],
        )
        .is_err());
    assert_ne!(first_attempt.as_str(), second_attempt.as_str());

    drop(raw);
    let _ = fs::remove_file(path);
}

#[test]
fn newly_inserted_jobs_are_always_pending() {
    let path = temp_db("new-job-state");
    let mut db = RunDb::open(&path, "run-new-job-state").unwrap();
    let job = db.insert_job("new", JobKind::Witness).unwrap();
    assert_eq!(db.job(&job).unwrap().state, JobState::Pending);
    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn unknown_persisted_event_types_are_rejected() {
    let path = temp_db("unknown-event");
    let mut db = RunDb::open(&path, "run-unknown-event").unwrap();
    let job = db.insert_job("root", JobKind::Witness).unwrap();
    db.refresh_readiness().unwrap();
    drop(db);

    let raw = Connection::open(&path).unwrap();
    raw.execute("UPDATE state_events SET event_type='future-event'", [])
        .unwrap();
    drop(raw);

    let reopened = RunDb::open(&path, "run-unknown-event").unwrap();
    assert!(matches!(
        reopened.events(&job),
        Err(DbError::UnknownEnum {
            column: "event_type",
            ..
        })
    ));
    drop(reopened);
    let _ = fs::remove_file(path);
}

#[test]
fn attempt_events_cannot_bypass_attempt_identity_checks() {
    let path = temp_db("attempt-api");
    let mut db = RunDb::open(&path, "run-attempt-api").unwrap();
    let job = insert_ready(&mut db, "leaf", JobKind::LeafProof);

    assert!(matches!(
        db.apply_event(&job, JobEvent::StartProving),
        Err(DbError::AttemptStartRequiresBegin)
    ));
    let attempt = db.begin_attempt(&job, JobEvent::StartProving).unwrap();
    assert!(matches!(
        db.apply_event(&job, JobEvent::ProofProduced),
        Err(DbError::AttemptEventRequiresAttempt)
    ));
    assert_eq!(db.job(&job).unwrap().state, JobState::Proving);
    db.apply_attempt_event(&job, &attempt, JobEvent::ProofProduced)
        .unwrap();
    assert_eq!(db.job(&job).unwrap().state, JobState::Proved);

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn stale_and_wrong_job_attempts_cannot_commit_worker_results() {
    let path = temp_db("stale-attempt");
    let mut db = RunDb::open(&path, "run-stale-attempt").unwrap();
    let first = insert_ready(&mut db, "first", JobKind::LeafProof);
    let second = insert_ready(&mut db, "second", JobKind::LeafProof);

    let stale = db.begin_attempt(&first, JobEvent::StartProving).unwrap();
    db.handle_attempt_failure(
        &first,
        &stale,
        failure(FailureKind::Execution, FailureStage::Prove),
        None,
    )
    .unwrap();
    db.refresh_readiness().unwrap();
    let current = db.begin_attempt(&first, JobEvent::StartProving).unwrap();
    let other = db.begin_attempt(&second, JobEvent::StartProving).unwrap();

    assert!(matches!(
        db.apply_attempt_event(&first, &stale, JobEvent::ProofProduced),
        Err(DbError::AttemptNotOpenForJob { .. })
    ));
    assert!(matches!(
        db.apply_attempt_event(&first, &other, JobEvent::ProofProduced),
        Err(DbError::AttemptNotOpenForJob { .. })
    ));
    assert_eq!(db.job(&first).unwrap().state, JobState::Proving);
    db.apply_attempt_event(&first, &current, JobEvent::ProofProduced)
        .unwrap();

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn job_kind_selects_the_only_valid_attempt_start_and_prevents_overlap() {
    let path = temp_db("kind-start");
    let mut db = RunDb::open(&path, "run-kind-start").unwrap();
    let cases = [
        (JobKind::Witness, JobEvent::StartWitnessing),
        (JobKind::Prepare, JobEvent::StartPreparing),
        (JobKind::LeafProof, JobEvent::StartProving),
        (JobKind::LeafVerification, JobEvent::StartVerification),
        (JobKind::NativeAggregate, JobEvent::StartVerification),
    ];

    for (index, (kind, start)) in cases.into_iter().enumerate() {
        let job = insert_ready(&mut db, &format!("job-{index}"), kind);
        let wrong_start = if kind == JobKind::Witness {
            JobEvent::StartProving
        } else {
            JobEvent::StartWitnessing
        };
        assert!(matches!(
            db.begin_attempt(&job, wrong_start),
            Err(DbError::WrongStartForJobKind { .. })
        ));
        let attempt = db.begin_attempt(&job, start).unwrap();
        assert!(matches!(
            db.begin_attempt(&job, JobEvent::StartVerification),
            Err(DbError::OpenAttemptConflict { .. }) | Err(DbError::WrongStartForJobKind { .. })
        ));
        assert_eq!(db.attempt(&attempt).unwrap().job_id, job);
    }

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn scheduler_and_verification_events_are_not_accepted_by_generic_apply_event() {
    let path = temp_db("controlled-events");
    let mut db = RunDb::open(&path, "run-controlled-events").unwrap();
    let job = db.insert_job("leaf", JobKind::LeafVerification).unwrap();
    let predecessor = zkie_runtime::JobId::new("run-controlled-events:parent").unwrap();
    let verification = VerificationRecordId::new("untrusted-record").unwrap();

    for event in [
        JobEvent::DependenciesSatisfied,
        JobEvent::DependencyBlocked { predecessor },
        JobEvent::RecoveryRequeue,
        JobEvent::VerificationSucceeded {
            verification_record: Some(verification),
        },
        JobEvent::VerificationFailed(failure(FailureKind::Verification, FailureStage::Verify)),
    ] {
        assert!(matches!(
            db.apply_event(&job, event),
            Err(DbError::ControlledEventRequired)
        ));
    }
    assert_eq!(db.job(&job).unwrap().state, JobState::Pending);
    assert!(db.events(&job).unwrap().is_empty());

    drop(db);
    let _ = fs::remove_file(path);
}
