use zkie_runtime::{
    descendant_resident_bytes, newest_attempt, terminate_attempt, AdmissionPolicy, MemoryAction,
    MemoryEnforcement, MemoryLimits, MonitorError, ProcessControl, ProcessEntry, ProcessTable,
    ProcessTreeMonitor, RunningAttempt, TerminationOutcome, TerminationSignal,
    HARD_BREACH_SAMPLES,
};

const GIB: u64 = MemoryLimits::GIB;

fn limits() -> MemoryLimits {
    MemoryLimits::gib(400, 440).unwrap()
}

fn attempt(name: &str, pid: u32, started_at_unix_ms: i64) -> RunningAttempt {
    RunningAttempt::new(
        zkie_runtime::AttemptId::new(name).unwrap(),
        pid,
        started_at_unix_ms,
    )
}

#[test]
fn limits_reject_zero_and_an_inverted_budget() {
    assert!(matches!(
        MemoryLimits::new(0, 1),
        Err(MonitorError::InvalidLimits)
    ));
    assert!(matches!(
        MemoryLimits::new(100, 100),
        Err(MonitorError::InvalidLimits)
    ));
    assert!(matches!(
        MemoryLimits::new(100, 99),
        Err(MonitorError::InvalidLimits)
    ));
    assert_eq!(limits().admission_bytes, 400 * GIB);
    assert_eq!(limits().hard_limit_bytes, 440 * GIB);
}

#[test]
fn admission_freezes_above_the_budget_and_terminates_only_after_consecutive_hard_breaches() {
    let mut policy = AdmissionPolicy::new(limits());

    // Below the admission budget: normal admission.
    assert_eq!(policy.observe(399 * GIB), MemoryAction::Admit);
    assert_eq!(policy.observe(400 * GIB), MemoryAction::Admit);

    // Between the budget and the hard limit: freeze admission, keep running workers.
    assert_eq!(policy.observe(401 * GIB), MemoryAction::FreezeAdmission);
    assert_eq!(policy.observe(440 * GIB), MemoryAction::FreezeAdmission);
    assert_eq!(policy.consecutive_hard_breaches(), 0);

    // Above the hard limit: freeze until the streak reaches the termination threshold.
    assert_eq!(policy.observe(441 * GIB), MemoryAction::FreezeAdmission);
    assert_eq!(policy.observe(441 * GIB), MemoryAction::FreezeAdmission);
    assert_eq!(policy.observe(441 * GIB), MemoryAction::TerminateNewest);
    assert_eq!(policy.consecutive_hard_breaches(), 0);
    assert_eq!(HARD_BREACH_SAMPLES, 3);
}

#[test]
fn one_sample_below_the_hard_limit_resets_the_consecutive_breach_counter() {
    let mut policy = AdmissionPolicy::new(limits());

    assert_eq!(policy.observe(441 * GIB), MemoryAction::FreezeAdmission);
    assert_eq!(policy.observe(441 * GIB), MemoryAction::FreezeAdmission);
    assert_eq!(policy.observe(439 * GIB), MemoryAction::FreezeAdmission);
    assert_eq!(policy.consecutive_hard_breaches(), 0);

    // The streak restarts: two more hard breaches must not terminate anything.
    assert_eq!(policy.observe(441 * GIB), MemoryAction::FreezeAdmission);
    assert_eq!(policy.observe(441 * GIB), MemoryAction::FreezeAdmission);
    assert_eq!(policy.observe(10 * GIB), MemoryAction::Admit);
}

#[test]
fn newest_attempt_is_selected_by_start_time_with_a_deterministic_tie_break() {
    let older = attempt("run:older", 10, 1_000);
    let newer = attempt("run:newer", 11, 2_000);
    assert_eq!(
        newest_attempt(&[older.clone(), newer.clone()]).unwrap(),
        newer
    );
    assert_eq!(newest_attempt(&[]), None);

    let first = attempt("run:a", 20, 5_000);
    let second = attempt("run:b", 21, 5_000);
    assert_eq!(
        newest_attempt(&[first, second]).unwrap().attempt.as_str(),
        "run:b"
    );
}

#[derive(Default)]
struct FakeControl {
    signals: Vec<TerminationSignal>,
    checks: u32,
    /// `None` never exits; `Some(n)` stays alive for the first `n` liveness checks.
    alive_checks: Option<u32>,
}

impl ProcessControl for FakeControl {
    fn signal(&mut self, _pid: u32, signal: TerminationSignal) -> Result<(), MonitorError> {
        self.signals.push(signal);
        Ok(())
    }

    fn is_alive(&mut self, _pid: u32) -> bool {
        self.checks += 1;
        match self.alive_checks {
            None => true,
            Some(limit) => self.checks <= limit,
        }
    }
}

#[test]
fn termination_escalates_from_sigterm_to_sigkill_only_when_the_attempt_survives() {
    let target = attempt("run:leaf", 4_242, 7);
    let peak = 12 * GIB;

    let mut graceful = FakeControl {
        alive_checks: Some(2),
        ..FakeControl::default()
    };
    let report = terminate_attempt(&mut graceful, &target, peak, 2).unwrap();
    assert_eq!(report.outcome, TerminationOutcome::ExitedAfterTerminate);
    assert_eq!(report.observed_peak_bytes, peak);
    assert_eq!(report.attempt, target.attempt);
    assert_eq!(graceful.signals, vec![TerminationSignal::Terminate]);

    let mut forced = FakeControl {
        alive_checks: Some(5),
        ..FakeControl::default()
    };
    let report = terminate_attempt(&mut forced, &target, peak, 2).unwrap();
    assert_eq!(report.outcome, TerminationOutcome::ForcedKill);
    assert_eq!(
        forced.signals,
        vec![TerminationSignal::Terminate, TerminationSignal::Kill]
    );

    let mut stuck = FakeControl {
        alive_checks: None,
        ..FakeControl::default()
    };
    let report = terminate_attempt(&mut stuck, &target, peak, 2).unwrap();
    assert_eq!(report.outcome, TerminationOutcome::DidNotExit);
    assert_eq!(
        stuck.signals,
        vec![TerminationSignal::Terminate, TerminationSignal::Kill]
    );
}

#[test]
fn descendant_sampling_is_transitive_deduplicated_and_empty_without_roots() {
    let entries = [
        ProcessEntry {
            pid: 100,
            parent_pid: 1,
            resident_bytes: 1_000,
        },
        ProcessEntry {
            pid: 101,
            parent_pid: 100,
            resident_bytes: 200,
        },
        ProcessEntry {
            pid: 102,
            parent_pid: 101,
            resident_bytes: 30,
        },
        ProcessEntry {
            pid: 200,
            parent_pid: 1,
            resident_bytes: 9_999,
        },
    ];

    assert_eq!(descendant_resident_bytes(&entries, &[100]), 1_230);
    // Overlapping roots must not double count the shared subtree.
    assert_eq!(descendant_resident_bytes(&entries, &[100, 101]), 1_230);
    assert_eq!(descendant_resident_bytes(&entries, &[100, 200]), 11_229);
    assert_eq!(descendant_resident_bytes(&entries, &[]), 0);
    assert_eq!(descendant_resident_bytes(&entries, &[777]), 0);
}

struct FakeTable {
    entries: Vec<ProcessEntry>,
    calls: u32,
}

impl ProcessTable for FakeTable {
    fn snapshot(&mut self) -> Result<Vec<ProcessEntry>, MonitorError> {
        self.calls += 1;
        Ok(self.entries.clone())
    }
}

#[test]
fn portable_monitor_records_best_effort_warns_once_and_applies_the_policy() {
    let mut monitor = ProcessTreeMonitor::new(
        FakeTable {
            entries: vec![
                ProcessEntry {
                    pid: 10,
                    parent_pid: 1,
                    resident_bytes: 300 * GIB,
                },
                ProcessEntry {
                    pid: 11,
                    parent_pid: 10,
                    resident_bytes: 120 * GIB,
                },
            ],
            calls: 0,
        },
        limits(),
    );

    assert_eq!(monitor.enforcement(), MemoryEnforcement::BestEffort);
    assert_eq!(MemoryEnforcement::BestEffort.as_str(), "best-effort");
    assert_eq!(MemoryEnforcement::CgroupV2.as_str(), "cgroup-v2");

    let warning = monitor.take_startup_warning().expect("first warning");
    assert!(warning.contains("\"mode\":\"best-effort\""));
    assert!(monitor.take_startup_warning().is_none());

    let running = [attempt("run:leaf", 10, 1)];
    let sample = monitor.sample(&running).unwrap();
    assert_eq!(sample.resident_bytes, 420 * GIB);
    assert_eq!(sample.action, MemoryAction::FreezeAdmission);

    // A quiet machine with no running attempts admits new work again.
    let sample = monitor.sample(&[]).unwrap();
    assert_eq!(sample.resident_bytes, 0);
    assert_eq!(sample.action, MemoryAction::Admit);
}
