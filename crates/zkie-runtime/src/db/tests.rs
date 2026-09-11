use std::fs;
use std::io::{Read, Write};

use super::*;
use crate::{
    AggregationArity, ArtifactMetadata, ArtifactRole, ArtifactStore, ContentStore, KeyMetadata,
    KeyRole, KeyStore, ObjectMetadata, TrustedKeyIdentity, TrustedProofIdentity, VerifiedArtifact,
    VerifiedKey,
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
    assert_eq!(migrated.schema_version().unwrap(), SCHEMA_VERSION);
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
fn v2_to_v3_migration_preserves_history_and_marks_existing_permanent_failures() {
    let path = std::env::temp_dir().join(format!(
        "zkie-v2-history-{}-{:?}.sqlite",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = fs::remove_file(&path);
    let connection = Connection::open(&path).unwrap();
    connection.execute_batch(SCHEMA_V1).unwrap();
    connection
        .execute(
            "INSERT INTO schema_version(version,applied_at_unix_ms) VALUES(1,0)",
            [],
        )
        .unwrap();
    connection.execute_batch(MIGRATE_V1_TO_V2).unwrap();
    connection
        .execute(
            "INSERT INTO schema_version(version,applied_at_unix_ms) VALUES(2,0)",
            [],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO runs(run_id,created_at_unix_ms) VALUES('v2-history',0)",
            [],
        )
        .unwrap();
    connection.execute(
            "INSERT INTO jobs(job_id,run_id,logical_job_id,kind,state,created_at_unix_ms,updated_at_unix_ms)
             VALUES('v2-history:failed','v2-history','failed','leaf-proof','verification-failed',0,0)",
            [],
        ).unwrap();
    let reservation_key = ReservationKey {
        circuit_digest: digest(b"v2-circuit"),
        k: 19,
        proof_flavor: ProofFlavorId::parse("halo2-kzg-v1").unwrap(),
        execution_backend: ExecutionBackendId::parse("halo2-cpu").unwrap(),
        hardware_profile: digest(b"v2-hardware"),
    };
    connection.execute(
            "INSERT INTO resource_history(circuit_digest,circuit_k,proof_flavor,execution_backend,hardware_profile,max_observed_peak_bytes,updated_at_unix_ms)
             VALUES(?1,?2,?3,?4,?5,?6,0)",
            params![
                reservation_key.circuit_digest.to_string(),
                i64::from(reservation_key.k),
                reservation_key.proof_flavor.as_str(),
                reservation_key.execution_backend.as_str(),
                reservation_key.hardware_profile.to_string(),
                123_i64,
            ],
        ).unwrap();
    drop(connection);

    let migrated = RunDb::open(&path, "v2-history").unwrap();
    assert_eq!(migrated.schema_version().unwrap(), SCHEMA_VERSION);
    assert!(
        migrated
            .job(&JobId::new("v2-history:failed").unwrap())
            .unwrap()
            .terminal_failure
    );
    assert_eq!(
        migrated.max_observed_peak_bytes(&reservation_key).unwrap(),
        Some(123)
    );

    drop(migrated);
    let _ = fs::remove_file(path);
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
