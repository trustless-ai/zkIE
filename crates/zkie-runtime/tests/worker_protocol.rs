use std::path::{Path, PathBuf};

use zkie_runtime::{
    sanitize_code, worker_spec, AttemptId, JobId, ProtocolError, SchedulableJob, Scheduler,
    SchedulerClock, SchedulerConfig, WorkerJob, WorkerJobKind, WorkerLauncher, WorkerMeasurements,
    WorkerOutcome, WorkerResult, WorkerSpec, WORKER_SCHEMA_VERSION,
};
use zkie_types::{Digest32, ResourceCapacity, ResourceRequest};

const GIB: u64 = 1_073_741_824;

fn digest(seed: u8) -> Digest32 {
    Digest32::new([seed; 32])
}

fn job_id(name: &str) -> JobId {
    JobId::new(name).unwrap()
}

fn attempt_id(name: &str) -> AttemptId {
    AttemptId::new(name).unwrap()
}

fn request(cpu_cores: u32, gpu_count: u32) -> ResourceRequest {
    ResourceRequest::new(cpu_cores, 8 * GIB, gpu_count, u64::from(gpu_count) * GIB).unwrap()
}

fn job(kind: WorkerJobKind) -> WorkerJob {
    WorkerJob::new(
        kind,
        [
            ("run".to_owned(), "run-1".to_owned()),
            ("shard".to_owned(), "0".to_owned()),
        ],
    )
    .unwrap()
}

fn spec(kind: WorkerJobKind) -> WorkerSpec {
    worker_spec(
        digest(1),
        job_id("run-1:leaf"),
        attempt_id("run-1:leaf:attempt:1"),
        job(kind),
        PathBuf::from("/var/lib/zkie/staged"),
        &request(4, 2),
    )
    .unwrap()
}

#[test]
fn every_job_kind_round_trips_through_the_wire_format() {
    for kind in WorkerJobKind::ALL {
        let original = spec(kind);
        let encoded = original.encode().unwrap();
        assert!(encoded.starts_with("zkie-worker-spec 1\n"));
        assert_eq!(WorkerSpec::decode(&encoded).unwrap(), original);
        assert_eq!(original.job.kind, kind);
        assert_eq!(WorkerJobKind::parse(kind.as_str()), Some(kind));
    }
}

#[test]
fn worker_spec_derives_threads_and_devices_from_the_reserved_resources() {
    let derived = spec(WorkerJobKind::LeafProof);
    assert_eq!(derived.rayon_threads, 4);
    assert_eq!(derived.cuda_devices, vec![0, 1]);
    assert_eq!(derived.schema_version, WORKER_SCHEMA_VERSION);
    assert_eq!(derived.job.field("shard"), Some("0"));

    let cpu_only = worker_spec(
        digest(2),
        job_id("run-1:verify"),
        attempt_id("run-1:verify:attempt:1"),
        job(WorkerJobKind::LeafVerification),
        PathBuf::from("/var/lib/zkie/staged"),
        &request(1, 0),
    )
    .unwrap();
    assert_eq!(cpu_only.rayon_threads, 1);
    assert!(cpu_only.cuda_devices.is_empty());
}

#[test]
fn unknown_schema_versions_and_unknown_keys_are_rejected() {
    let mut future = spec(WorkerJobKind::Witness);
    future.schema_version = WORKER_SCHEMA_VERSION + 1;
    assert!(matches!(
        future.encode(),
        Err(ProtocolError::UnsupportedSchemaVersion(version)) if version == WORKER_SCHEMA_VERSION + 1
    ));

    let future_wire = format!("zkie-worker-spec {}\n", WORKER_SCHEMA_VERSION + 7);
    assert!(matches!(
        WorkerSpec::decode(&future_wire),
        Err(ProtocolError::UnsupportedSchemaVersion(_))
    ));

    assert!(matches!(
        WorkerSpec::decode("zkie-worker-spec 1\nextra_key 1\n"),
        Err(ProtocolError::Malformed(_))
    ));
    assert!(matches!(
        WorkerSpec::decode("zkie-worker-spec 1\n"),
        Err(ProtocolError::Malformed(_))
    ));
    assert!(matches!(
        WorkerSpec::decode("zkie-worker-result 1\n"),
        Err(ProtocolError::Malformed(_))
    ));
}

#[test]
fn relative_output_directories_and_duplicate_devices_are_rejected() {
    let mut relative = spec(WorkerJobKind::Prepare);
    relative.staged_output_dir = PathBuf::from("staged");
    assert!(matches!(
        relative.validate(),
        Err(ProtocolError::RelativeOutputDir(_))
    ));
    assert!(matches!(
        relative.encode(),
        Err(ProtocolError::RelativeOutputDir(_))
    ));

    let mut duplicate = spec(WorkerJobKind::LeafProof);
    duplicate.cuda_devices = vec![0, 0];
    assert!(matches!(
        duplicate.validate(),
        Err(ProtocolError::DuplicateCudaDevice(0))
    ));

    let mut no_threads = spec(WorkerJobKind::LeafProof);
    no_threads.rayon_threads = 0;
    assert!(matches!(
        no_threads.validate(),
        Err(ProtocolError::ZeroThreads)
    ));
}

#[test]
fn a_result_from_another_run_job_or_attempt_is_rejected() {
    let expected = spec(WorkerJobKind::LeafProof);
    let measurements = WorkerMeasurements {
        peak_resident_bytes: 3 * GIB,
        wall_millis: 42,
    };

    let good = WorkerResult::succeeded(&expected, measurements);
    assert!(good.verify_against(&expected).is_ok());
    assert_eq!(WorkerResult::decode(&good.encode().unwrap()).unwrap(), good);

    let mut foreign_run = good.clone();
    foreign_run.run_digest = digest(9);
    assert!(matches!(
        foreign_run.verify_against(&expected),
        Err(ProtocolError::RunDigestMismatch)
    ));

    let mut foreign_job = good.clone();
    foreign_job.job_id = job_id("run-1:other");
    assert!(matches!(
        foreign_job.verify_against(&expected),
        Err(ProtocolError::JobMismatch { .. })
    ));

    let mut stale_attempt = good;
    stale_attempt.attempt_id = attempt_id("run-1:leaf:attempt:2");
    assert!(matches!(
        stale_attempt.verify_against(&expected),
        Err(ProtocolError::AttemptMismatch { .. })
    ));
}

#[test]
fn worker_failure_codes_are_sanitized_before_they_are_persisted() {
    assert_eq!(sanitize_code("cuda-out-of-memory"), "cuda-out-of-memory");
    assert_eq!(sanitize_code("bad code/with spaces!"), "badcodewithspaces");
    assert_eq!(sanitize_code("\u{1}\u{2}"), "unknown");
    assert_eq!(sanitize_code(&"x".repeat(200)).len(), 64);

    let expected = spec(WorkerJobKind::LeafProof);
    let result = WorkerResult::failed(
        &expected,
        "secret token=abcd\ninjected",
        WorkerMeasurements::default(),
    );
    match &result.outcome {
        WorkerOutcome::Failed { code } => {
            assert_eq!(code, "secrettokenabcdinjected");
            assert!(!code.contains(' ') && !code.contains('\n'));
        }
        WorkerOutcome::Succeeded => panic!("expected a failure result"),
    }
    let decoded = WorkerResult::decode(&result.encode().unwrap()).unwrap();
    assert_eq!(decoded, result);
}

/// In-process launcher used to exercise scheduler wiring without spawning binaries.
struct FakeLauncher {
    attempt_override: Option<AttemptId>,
    launches: usize,
}

impl WorkerLauncher for FakeLauncher {
    fn launch(
        &mut self,
        spec: &WorkerSpec,
        _result_path: &Path,
    ) -> Result<WorkerResult, ProtocolError> {
        self.launches += 1;
        let mut result = WorkerResult::succeeded(
            spec,
            WorkerMeasurements {
                peak_resident_bytes: 5 * GIB,
                wall_millis: 7,
            },
        );
        if let Some(attempt) = &self.attempt_override {
            result.attempt_id = attempt.clone();
        }
        result.verify_against(spec)?;
        Ok(result)
    }
}

struct FixedClock(i64);

impl SchedulerClock for FixedClock {
    fn now_unix_seconds(&self) -> i64 {
        self.0
    }
}

#[test]
fn scheduler_builds_a_spec_the_launcher_can_only_answer_for_that_attempt() {
    let capacity = ResourceCapacity::new(8, 64 * GIB, 2, 24 * GIB).unwrap();
    let scheduler = Scheduler::new(SchedulerConfig::new(capacity), FixedClock(1_000));
    let candidate = SchedulableJob::new(job_id("run-1:leaf"), request(6, 1), 1_000);

    let built = scheduler
        .spec_for(
            &candidate,
            digest(3),
            attempt_id("run-1:leaf:attempt:1"),
            job(WorkerJobKind::LeafProof),
            PathBuf::from("/var/lib/zkie/staged"),
        )
        .unwrap();
    assert_eq!(built.rayon_threads, 6);
    assert_eq!(built.cuda_devices, vec![0]);

    let mut honest = FakeLauncher {
        attempt_override: None,
        launches: 0,
    };
    let result = honest
        .launch(&built, Path::new("/var/lib/zkie/staged/result.txt"))
        .unwrap();
    assert_eq!(result.attempt_id, built.attempt_id);
    assert_eq!(honest.launches, 1);

    let mut impostor = FakeLauncher {
        attempt_override: Some(attempt_id("run-1:leaf:attempt:9")),
        launches: 0,
    };
    assert!(matches!(
        impostor.launch(&built, Path::new("/var/lib/zkie/staged/result.txt")),
        Err(ProtocolError::AttemptMismatch { .. })
    ));
}
