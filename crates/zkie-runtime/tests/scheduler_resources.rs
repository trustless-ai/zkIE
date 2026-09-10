use std::fs;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicI64, Ordering},
    Arc,
};

use zkie_runtime::{
    calibrated_reservation, EstimateError, FailureCode, FailureKind, FailureRecord, FailureStage,
    JobEvent, JobKind, JobState, ReservationKey, RetryDecision, RunDb, SchedulableJob, Scheduler,
    SchedulerClock, SchedulerConfig, SchedulerError,
};
use zkie_types::{Digest32, ExecutionBackendId, ProofFlavorId, ResourceCapacity, ResourceRequest};

const GIB: u64 = 1_073_741_824;

fn request(cpu: u32, gib: u64, gpus: u32, vram_gib: u64) -> ResourceRequest {
    ResourceRequest::new(cpu, gib * GIB, gpus, vram_gib * GIB).unwrap()
}

fn capacity(cpu: u32, gib: u64, gpus: u32, vram_gib: u64) -> ResourceCapacity {
    ResourceCapacity::new(cpu, gib * GIB, gpus, vram_gib * GIB).unwrap()
}

fn key(seed: u8) -> ReservationKey {
    ReservationKey {
        circuit_digest: Digest32::new([seed; 32]),
        k: 19,
        proof_flavor: ProofFlavorId::parse("halo2-kzg-v1").unwrap(),
        execution_backend: ExecutionBackendId::parse("halo2-cpu").unwrap(),
        hardware_profile: Digest32::new([9; 32]),
    }
}

fn temp_db(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-scheduler-{label}-{}-{}.sqlite",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = fs::remove_file(&path);
    path
}

#[derive(Clone)]
struct FakeClock(Arc<AtomicI64>);

impl FakeClock {
    fn new(now: i64) -> Self {
        Self(Arc::new(AtomicI64::new(now)))
    }

    fn advance(&self, seconds: i64) {
        self.0.fetch_add(seconds, Ordering::SeqCst);
    }
}

impl SchedulerClock for FakeClock {
    fn now_unix_seconds(&self) -> i64 {
        self.0.load(Ordering::SeqCst)
    }
}

#[test]
fn calibration_uses_integer_ceiling_and_static_max_without_changing_other_dimensions() {
    let static_request = request(7, 100, 2, 24);
    let calibrated = calibrated_reservation(static_request.clone(), Some(101 * GIB)).unwrap();

    assert_eq!(calibrated.ram_bytes, (101 * GIB * 115).div_ceil(100));
    assert_eq!(calibrated.cpu_cores, static_request.cpu_cores);
    assert_eq!(calibrated.gpu_count, static_request.gpu_count);
    assert_eq!(
        calibrated.gpu_vram_bytes_per_device,
        static_request.gpu_vram_bytes_per_device
    );
    assert_eq!(
        calibrated_reservation(request(4, 200, 0, 0), Some(100 * GIB))
            .unwrap()
            .ram_bytes,
        200 * GIB
    );
}

#[test]
fn calibration_reports_overflow_instead_of_wrapping() {
    let large_but_representable = 1_000_000_000_000_000_000_u64;
    assert_eq!(
        calibrated_reservation(request(1, 1, 0, 0), Some(large_but_representable))
            .unwrap()
            .ram_bytes,
        1_150_000_000_000_000_000
    );
    assert_eq!(
        calibrated_reservation(request(1, 1, 0, 0), Some(u64::MAX)),
        Err(EstimateError::Overflow)
    );
}

#[test]
fn resource_history_requires_the_complete_reservation_key() {
    let path = temp_db("history-key");
    let mut db = RunDb::open(&path, "history-key").unwrap();
    let base = key(1);
    db.record_resource_peak(&base, 10 * GIB).unwrap();
    db.record_resource_peak(&base, 12 * GIB).unwrap();
    db.record_resource_peak(&base, 11 * GIB).unwrap();

    assert_eq!(db.max_observed_peak_bytes(&base).unwrap(), Some(12 * GIB));
    for different in [
        ReservationKey {
            circuit_digest: Digest32::new([2; 32]),
            ..base.clone()
        },
        ReservationKey {
            k: 20,
            ..base.clone()
        },
        ReservationKey {
            proof_flavor: ProofFlavorId::parse("stark-v1").unwrap(),
            ..base.clone()
        },
        ReservationKey {
            execution_backend: ExecutionBackendId::parse("halo2-cuda").unwrap(),
            ..base.clone()
        },
        ReservationKey {
            hardware_profile: Digest32::new([8; 32]),
            ..base.clone()
        },
    ] {
        assert_eq!(db.max_observed_peak_bytes(&different).unwrap(), None);
    }

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn deterministic_best_fit_fills_ram_without_exceeding_cpu_and_uses_exact_gpu_dimensions() {
    let path = temp_db("best-fit");
    let mut db = RunDb::open(&path, "best-fit").unwrap();
    let j250 = db.insert_job("j250", JobKind::LeafProof).unwrap();
    let j150 = db.insert_job("j150", JobKind::LeafProof).unwrap();
    let j100 = db.insert_job("j100", JobKind::LeafProof).unwrap();
    let gpu_too_large = db.insert_job("gpu-too-large", JobKind::LeafProof).unwrap();
    let vram_too_large = db.insert_job("vram-too-large", JobKind::LeafProof).unwrap();
    let clock = FakeClock::new(1_000);
    let scheduler = Scheduler::new(SchedulerConfig::new(capacity(56, 400, 2, 24)), clock);
    let candidates = vec![
        SchedulableJob::new(j100.clone(), request(20, 100, 0, 0), 900),
        SchedulableJob::new(j150.clone(), request(26, 150, 0, 0), 900),
        SchedulableJob::new(j250.clone(), request(30, 250, 0, 0), 900),
        SchedulableJob::new(gpu_too_large, request(1, 1, 3, 24), 900),
        SchedulableJob::new(vram_too_large, request(1, 1, 1, 25), 900),
    ];

    let decision = scheduler
        .decide(&mut db, &candidates, capacity(56, 400, 2, 24))
        .unwrap();
    assert_eq!(decision.selected, vec![j250, j150]);
    assert_eq!(decision.remaining, capacity(0, 0, 2, 24));

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn caller_reported_availability_cannot_exceed_configured_budget() {
    let path = temp_db("configured-budget");
    let mut db = RunDb::open(&path, "configured-budget").unwrap();
    let oversized = db.insert_job("oversized", JobKind::LeafProof).unwrap();
    let scheduler = Scheduler::new(
        SchedulerConfig::new(capacity(56, 400, 0, 0)),
        FakeClock::new(1_000),
    );
    let candidates = [SchedulableJob::new(oversized, request(40, 500, 0, 0), 900)];

    let decision = scheduler
        .decide(&mut db, &candidates, capacity(64, 800, 0, 0))
        .unwrap();
    assert!(decision.selected.is_empty());
    assert_eq!(decision.remaining, capacity(56, 400, 0, 0));

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn duplicate_candidate_ids_are_rejected_instead_of_scheduling_one_job_twice() {
    let path = temp_db("duplicate-candidate");
    let mut db = RunDb::open(&path, "duplicate-candidate").unwrap();
    let job = db.insert_job("job", JobKind::LeafProof).unwrap();
    let scheduler = Scheduler::new(
        SchedulerConfig::new(capacity(8, 16, 0, 0)),
        FakeClock::new(1_000),
    );
    let candidates = [
        SchedulableJob::new(job.clone(), request(1, 1, 0, 0), 900),
        SchedulableJob::new(job.clone(), request(1, 1, 0, 0), 900),
    ];

    assert!(matches!(
        scheduler.decide(&mut db, &candidates, capacity(8, 16, 0, 0)),
        Err(SchedulerError::DuplicateCandidate(id)) if id == job
    ));

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn scheduler_never_selects_a_job_until_every_predecessor_is_verified() {
    let path = temp_db("verified-only");
    let mut db = RunDb::open(&path, "verified-only").unwrap();
    let predecessor = db.insert_job("predecessor", JobKind::LeafProof).unwrap();
    let downstream = db
        .insert_job("downstream", JobKind::NativeAggregate)
        .unwrap();
    db.add_dependency(&predecessor, &downstream).unwrap();
    let scheduler = Scheduler::new(
        SchedulerConfig::new(capacity(8, 16, 0, 0)),
        FakeClock::new(1_000),
    );
    let candidates = [SchedulableJob::new(
        downstream.clone(),
        request(1, 1, 0, 0),
        0,
    )];

    let decision = scheduler
        .decide(&mut db, &candidates, capacity(8, 16, 0, 0))
        .unwrap();
    assert!(decision.selected.is_empty());
    assert_eq!(db.job(&downstream).unwrap().state, JobState::Pending);

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn scheduler_rechecks_predecessors_even_when_a_job_was_already_ready() {
    let path = temp_db("late-dependency");
    let mut db = RunDb::open(&path, "late-dependency").unwrap();
    let downstream = db
        .insert_job("downstream", JobKind::NativeAggregate)
        .unwrap();
    db.refresh_readiness().unwrap();
    assert_eq!(db.job(&downstream).unwrap().state, JobState::Ready);
    let predecessor = db.insert_job("predecessor", JobKind::LeafProof).unwrap();
    db.add_dependency(&predecessor, &downstream).unwrap();
    let scheduler = Scheduler::new(
        SchedulerConfig::new(capacity(8, 16, 0, 0)),
        FakeClock::new(1_000),
    );

    let decision = scheduler
        .decide(
            &mut db,
            &[SchedulableJob::new(downstream, request(1, 1, 0, 0), 0)],
            capacity(8, 16, 0, 0),
        )
        .unwrap();
    assert!(decision.selected.is_empty());

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn exactly_600_seconds_activates_conservative_reservation_then_runs_oldest_aged_job_first() {
    let path = temp_db("aging");
    let mut db = RunDb::open(&path, "aging").unwrap();
    let aged = db.insert_job("aged-300", JobKind::LeafProof).unwrap();
    let young = db.insert_job("young-100", JobKind::LeafProof).unwrap();
    let clock = FakeClock::new(1_599);
    let scheduler = Scheduler::new(SchedulerConfig::new(capacity(56, 400, 0, 0)), clock.clone());
    let candidates = [
        SchedulableJob::new(aged.clone(), request(30, 300, 0, 0), 1_000),
        SchedulableJob::new(young.clone(), request(10, 100, 0, 0), 1_599),
    ];

    let before = scheduler
        .decide(&mut db, &candidates, capacity(26, 200, 0, 0))
        .unwrap();
    assert_eq!(before.selected, vec![young]);
    assert_eq!(before.reserved_for, None);

    clock.advance(1);
    let reserved = scheduler
        .decide(&mut db, &candidates, capacity(26, 200, 0, 0))
        .unwrap();
    assert!(reserved.selected.is_empty());
    assert_eq!(reserved.reserved_for, Some(aged.clone()));

    let runnable = scheduler
        .decide(&mut db, &candidates, capacity(56, 400, 0, 0))
        .unwrap();
    assert_eq!(runnable.selected.first(), Some(&aged));

    drop(db);
    let _ = fs::remove_file(path);
}

fn failure(kind: FailureKind) -> FailureRecord {
    FailureRecord::new(
        kind,
        FailureStage::Prove,
        match kind {
            FailureKind::ResourceExceeded => FailureCode::ResourceLimit,
            _ => FailureCode::WorkerExit,
        },
    )
}

fn ready_leaf(db: &mut RunDb, name: &str) -> zkie_runtime::JobId {
    let job = db.insert_job(name, JobKind::LeafProof).unwrap();
    db.refresh_readiness().unwrap();
    job
}

#[test]
fn execution_gets_two_retries_and_resource_exceeded_gets_three_requeues_with_peak_first() {
    let path = temp_db("retries");
    let mut db = RunDb::open(&path, "retries").unwrap();
    let execution = ready_leaf(&mut db, "execution");
    for expected_attempt in 1..=3 {
        let attempt = db
            .begin_attempt(&execution, JobEvent::StartProving)
            .unwrap();
        let decision = db
            .handle_attempt_failure(&execution, &attempt, failure(FailureKind::Execution), None)
            .unwrap();
        assert_eq!(
            decision,
            if expected_attempt <= 2 {
                RetryDecision::Requeued
            } else {
                RetryDecision::Terminal
            }
        );
        if expected_attempt <= 2 {
            db.refresh_readiness().unwrap();
        }
    }
    let execution_row = db.job(&execution).unwrap();
    assert_eq!(execution_row.execution_retries, 2);
    assert_eq!(execution_row.resource_requeues, 0);
    assert!(execution_row.terminal_failure);

    let resource = ready_leaf(&mut db, "resource");
    let history_key = key(4);
    for expected_attempt in 1..=4 {
        let attempt = db.begin_attempt(&resource, JobEvent::StartProving).unwrap();
        let peak = expected_attempt as u64 * 10 * GIB;
        let decision = db
            .handle_attempt_failure(
                &resource,
                &attempt,
                failure(FailureKind::ResourceExceeded),
                Some((&history_key, peak)),
            )
            .unwrap();
        assert_eq!(
            db.max_observed_peak_bytes(&history_key).unwrap(),
            Some(peak)
        );
        assert_eq!(
            decision,
            if expected_attempt <= 3 {
                RetryDecision::Requeued
            } else {
                RetryDecision::Terminal
            }
        );
        if expected_attempt <= 3 {
            db.refresh_readiness().unwrap();
        }
    }
    let resource_row = db.job(&resource).unwrap();
    assert_eq!(resource_row.execution_retries, 0);
    assert_eq!(resource_row.resource_requeues, 3);
    assert!(resource_row.terminal_failure);

    drop(db);
    let _ = fs::remove_file(path);
}

#[test]
fn retry_exhaustion_recursively_blocks_descendants_with_their_direct_predecessor() {
    let path = temp_db("recursive-block");
    let mut db = RunDb::open(&path, "recursive-block").unwrap();
    let root = ready_leaf(&mut db, "root");
    let child = db.insert_job("child", JobKind::NativeAggregate).unwrap();
    let grandchild = db
        .insert_job("grandchild", JobKind::NativeAggregate)
        .unwrap();
    db.add_dependency(&root, &child).unwrap();
    db.add_dependency(&child, &grandchild).unwrap();
    for attempt_number in 1..=3 {
        let attempt = db.begin_attempt(&root, JobEvent::StartProving).unwrap();
        db.handle_attempt_failure(&root, &attempt, failure(FailureKind::Execution), None)
            .unwrap();
        if attempt_number <= 2 {
            db.refresh_readiness().unwrap();
        }
    }

    assert_eq!(db.job(&child).unwrap().state, JobState::Blocked);
    assert_eq!(db.job(&child).unwrap().blocking_predecessor_id, Some(root));
    assert_eq!(db.job(&grandchild).unwrap().state, JobState::Blocked);
    assert_eq!(
        db.job(&grandchild).unwrap().blocking_predecessor_id,
        Some(child)
    );

    drop(db);
    let _ = fs::remove_file(path);
}
