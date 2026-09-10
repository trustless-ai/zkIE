//! Memory admission control, portable sampling and hard-limit enforcement.
//!
//! The scheduler admits work against an explicit admission budget. The monitor freezes
//! admission above that budget and terminates a runaway attempt above the hard limit.
//! Kernel enforcement (cgroup v2 on Linux) is preferred; portable process-tree sampling
//! is only used when the operator opts in explicitly, and the run records
//! [`MemoryEnforcement::BestEffort`] so the weaker guarantee is never implicit.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::AttemptId;

/// Consecutive samples above the hard limit before the newest attempt is terminated.
pub const HARD_BREACH_SAMPLES: u32 = 3;
/// Polls a terminated attempt is given to exit before its process group is killed.
pub const TERMINATION_GRACE_POLLS: u32 = 10;

#[derive(Debug, Error)]
pub enum MonitorError {
    #[error("cgroup v2 memory delegation is unavailable")]
    NoDelegatedCgroup,
    #[error("memory hard limit must be greater than the admission budget")]
    InvalidLimits,
    #[error("monitor I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("process sampling failed: {0}")]
    Sampling(String),
    #[error("attempt {0} has no observable process")]
    MissingProcess(AttemptId),
}

/// Which mechanism is actually enforcing the hard limit for a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEnforcement {
    /// `memory.high`/`memory.max` in a delegated cgroup v2 subtree.
    CgroupV2,
    /// Periodic sampling plus controlled termination, recorded explicitly.
    BestEffort,
}

impl MemoryEnforcement {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CgroupV2 => "cgroup-v2",
            Self::BestEffort => "best-effort",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryLimits {
    pub admission_bytes: u64,
    pub hard_limit_bytes: u64,
}

impl MemoryLimits {
    pub const GIB: u64 = 1_073_741_824;

    pub fn new(admission_bytes: u64, hard_limit_bytes: u64) -> Result<Self, MonitorError> {
        if admission_bytes == 0 || hard_limit_bytes <= admission_bytes {
            return Err(MonitorError::InvalidLimits);
        }
        Ok(Self {
            admission_bytes,
            hard_limit_bytes,
        })
    }

    pub fn gib(admission_gib: u64, hard_limit_gib: u64) -> Result<Self, MonitorError> {
        Self::new(
            admission_gib
                .checked_mul(Self::GIB)
                .ok_or(MonitorError::InvalidLimits)?,
            hard_limit_gib
                .checked_mul(Self::GIB)
                .ok_or(MonitorError::InvalidLimits)?,
        )
    }
}

/// What the scheduler must do with the newest observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAction {
    /// Below the admission budget: keep admitting work.
    Admit,
    /// Above the admission budget: stop admitting new work, keep running workers.
    FreezeAdmission,
    /// [`HARD_BREACH_SAMPLES`] consecutive samples above the hard limit.
    TerminateNewest,
}

/// Pure admission policy: no I/O, fully deterministic, injectable in tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionPolicy {
    limits: MemoryLimits,
    consecutive_hard_breaches: u32,
}

impl AdmissionPolicy {
    pub fn new(limits: MemoryLimits) -> Self {
        Self {
            limits,
            consecutive_hard_breaches: 0,
        }
    }

    pub fn limits(&self) -> MemoryLimits {
        self.limits
    }

    pub fn consecutive_hard_breaches(&self) -> u32 {
        self.consecutive_hard_breaches
    }

    pub fn observe(&mut self, resident_bytes: u64) -> MemoryAction {
        if resident_bytes > self.limits.hard_limit_bytes {
            self.consecutive_hard_breaches += 1;
            if self.consecutive_hard_breaches >= HARD_BREACH_SAMPLES {
                self.consecutive_hard_breaches = 0;
                return MemoryAction::TerminateNewest;
            }
            return MemoryAction::FreezeAdmission;
        }
        self.consecutive_hard_breaches = 0;
        if resident_bytes > self.limits.admission_bytes {
            MemoryAction::FreezeAdmission
        } else {
            MemoryAction::Admit
        }
    }
}

/// A worker the scheduler believes is currently running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunningAttempt {
    pub attempt: AttemptId,
    pub pid: u32,
    pub started_at_unix_ms: i64,
}

impl RunningAttempt {
    pub fn new(attempt: AttemptId, pid: u32, started_at_unix_ms: i64) -> Self {
        Self {
            attempt,
            pid,
            started_at_unix_ms,
        }
    }
}

/// Newest running attempt by start time, tie-broken by attempt id for determinism.
pub fn newest_attempt(attempts: &[RunningAttempt]) -> Option<RunningAttempt> {
    attempts
        .iter()
        .max_by(|left, right| {
            left.started_at_unix_ms
                .cmp(&right.started_at_unix_ms)
                .then_with(|| left.attempt.cmp(&right.attempt))
        })
        .cloned()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationSignal {
    Terminate,
    Kill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationOutcome {
    ExitedAfterTerminate,
    ForcedKill,
    DidNotExit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminationReport {
    pub attempt: AttemptId,
    pub pid: u32,
    pub observed_peak_bytes: u64,
    pub outcome: TerminationOutcome,
}

/// Process signalling, separated so termination can be exercised without real processes.
pub trait ProcessControl {
    fn signal(&mut self, pid: u32, signal: TerminationSignal) -> Result<(), MonitorError>;
    fn is_alive(&mut self, pid: u32) -> bool;
}

/// SIGTERM, bounded grace period, then SIGKILL on the same process group.
pub fn terminate_attempt(
    control: &mut dyn ProcessControl,
    target: &RunningAttempt,
    observed_peak_bytes: u64,
    grace_polls: u32,
) -> Result<TerminationReport, MonitorError> {
    control.signal(target.pid, TerminationSignal::Terminate)?;
    if wait_for_exit(control, target.pid, grace_polls) {
        return Ok(report(
            target,
            observed_peak_bytes,
            TerminationOutcome::ExitedAfterTerminate,
        ));
    }
    control.signal(target.pid, TerminationSignal::Kill)?;
    let outcome = if wait_for_exit(control, target.pid, grace_polls) {
        TerminationOutcome::ForcedKill
    } else {
        TerminationOutcome::DidNotExit
    };
    Ok(report(target, observed_peak_bytes, outcome))
}

fn wait_for_exit(control: &mut dyn ProcessControl, pid: u32, polls: u32) -> bool {
    if !control.is_alive(pid) {
        return true;
    }
    for _ in 0..polls {
        if !control.is_alive(pid) {
            return true;
        }
    }
    false
}

fn report(
    target: &RunningAttempt,
    observed_peak_bytes: u64,
    outcome: TerminationOutcome,
) -> TerminationReport {
    TerminationReport {
        attempt: target.attempt.clone(),
        pid: target.pid,
        observed_peak_bytes,
        outcome,
    }
}

/// One row of a process table snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessEntry {
    pub pid: u32,
    pub parent_pid: u32,
    pub resident_bytes: u64,
}

/// Process-table source, separated so the portable monitor is testable without processes.
pub trait ProcessTable {
    fn snapshot(&mut self) -> Result<Vec<ProcessEntry>, MonitorError>;
}

/// Resident bytes of `roots` plus every transitive descendant, counting each pid once.
pub fn descendant_resident_bytes(entries: &[ProcessEntry], roots: &[u32]) -> u64 {
    if roots.is_empty() {
        return 0;
    }
    let mut by_parent: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (index, entry) in entries.iter().enumerate() {
        by_parent.entry(entry.parent_pid).or_default().push(index);
    }
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut stack: Vec<u32> = roots.to_vec();
    let mut total = 0_u64;
    while let Some(pid) = stack.pop() {
        if !seen.insert(pid) {
            continue;
        }
        if let Some(index) = entries.iter().position(|entry| entry.pid == pid) {
            total = total.saturating_add(entries[index].resident_bytes);
        }
        if let Some(children) = by_parent.get(&pid) {
            for index in children {
                stack.push(entries[*index].pid);
            }
        }
    }
    total
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySample {
    pub resident_bytes: u64,
    pub action: MemoryAction,
}

/// Portable fallback monitor: samples the worker process trees and applies the policy.
pub struct ProcessTreeMonitor<T> {
    table: T,
    policy: AdmissionPolicy,
    warning_taken: bool,
}

impl<T: ProcessTable> ProcessTreeMonitor<T> {
    pub fn new(table: T, limits: MemoryLimits) -> Self {
        Self {
            table,
            policy: AdmissionPolicy::new(limits),
            warning_taken: false,
        }
    }

    pub fn enforcement(&self) -> MemoryEnforcement {
        MemoryEnforcement::BestEffort
    }

    /// Structured warning text, returned exactly once per monitor.
    pub fn take_startup_warning(&mut self) -> Option<String> {
        if self.warning_taken {
            return None;
        }
        self.warning_taken = true;
        Some(format!(
            "{{\"event\":\"memory_enforcement\",\"mode\":\"{}\",\"admission_bytes\":{},\"hard_limit_bytes\":{}}}",
            MemoryEnforcement::BestEffort.as_str(),
            self.policy.limits().admission_bytes,
            self.policy.limits().hard_limit_bytes,
        ))
    }

    pub fn sample(&mut self, running: &[RunningAttempt]) -> Result<MemorySample, MonitorError> {
        let entries = self.table.snapshot()?;
        let roots = running
            .iter()
            .map(|attempt| attempt.pid)
            .collect::<Vec<_>>();
        let resident_bytes = descendant_resident_bytes(&entries, &roots);
        Ok(MemorySample {
            resident_bytes,
            action: self.policy.observe(resident_bytes),
        })
    }
}

/// Linux `cgroup v2` monitor over a delegated subtree.
#[cfg(target_os = "linux")]
pub struct CgroupV2Monitor {
    run_dir: std::path::PathBuf,
    policy: AdmissionPolicy,
}

#[cfg(target_os = "linux")]
impl CgroupV2Monitor {
    const ROOT: &'static str = "/sys/fs/cgroup";

    /// Verifies delegated `memory` control and creates the run-specific child cgroup.
    pub fn open(limits: MemoryLimits) -> Result<Self, MonitorError> {
        let controllers = std::fs::read_to_string(format!("{}/cgroup.controllers", Self::ROOT))?;
        if !controllers.split_whitespace().any(|name| name == "memory") {
            return Err(MonitorError::NoDelegatedCgroup);
        }
        let run_dir =
            std::path::PathBuf::from(format!("{}/zkie-run-{}", Self::ROOT, std::process::id()));
        std::fs::create_dir_all(&run_dir)?;
        std::fs::write(
            run_dir.join("memory.high"),
            limits.admission_bytes.to_string(),
        )?;
        std::fs::write(
            run_dir.join("memory.max"),
            limits.hard_limit_bytes.to_string(),
        )?;
        Ok(Self {
            run_dir,
            policy: AdmissionPolicy::new(limits),
        })
    }

    pub fn enforcement(&self) -> MemoryEnforcement {
        MemoryEnforcement::CgroupV2
    }

    /// Moves a worker into the run cgroup; the scheduler itself stays outside it.
    pub fn attach(&self, pid: u32) -> Result<(), MonitorError> {
        std::fs::write(self.run_dir.join("cgroup.procs"), pid.to_string())?;
        Ok(())
    }

    pub fn sample(&mut self) -> Result<MemorySample, MonitorError> {
        let raw = std::fs::read_to_string(self.run_dir.join("memory.current"))?;
        let resident_bytes = raw
            .trim()
            .parse::<u64>()
            .map_err(|error| MonitorError::Sampling(error.to_string()))?;
        Ok(MemorySample {
            resident_bytes,
            action: self.policy.observe(resident_bytes),
        })
    }
}

/// `/proc`-based process table for Linux hosts without a delegated cgroup.
#[cfg(target_os = "linux")]
#[derive(Default)]
pub struct ProcProcessTable;

#[cfg(target_os = "linux")]
impl ProcessTable for ProcProcessTable {
    fn snapshot(&mut self) -> Result<Vec<ProcessEntry>, MonitorError> {
        let mut entries = Vec::new();
        for directory in std::fs::read_dir("/proc")? {
            let directory = directory?;
            let name = directory.file_name();
            let Ok(pid) = name.to_string_lossy().parse::<u32>() else {
                continue;
            };
            let Some((parent_pid, resident_bytes)) = read_proc_stat(&directory.path()) else {
                continue;
            };
            entries.push(ProcessEntry {
                pid,
                parent_pid,
                resident_bytes,
            });
        }
        Ok(entries)
    }
}

#[cfg(target_os = "linux")]
fn read_proc_stat(directory: &std::path::Path) -> Option<(u32, u64)> {
    let stat = std::fs::read_to_string(directory.join("stat")).ok()?;
    let end = stat.rfind(") ")?;
    let mut fields = stat[end + 2..].split_whitespace();
    let _state = fields.next()?;
    let parent_pid = fields.next()?.parse::<u32>().ok()?;
    let statm = std::fs::read_to_string(directory.join("statm")).ok()?;
    let resident_pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some((parent_pid, resident_pages.saturating_mul(4096)))
}

/// Process table for non-Linux hosts, read from the portable `ps` snapshot.
#[cfg(not(target_os = "linux"))]
#[derive(Default)]
pub struct PsProcessTable;

#[cfg(not(target_os = "linux"))]
impl ProcessTable for PsProcessTable {
    fn snapshot(&mut self) -> Result<Vec<ProcessEntry>, MonitorError> {
        let output = std::process::Command::new("ps")
            .args(["-o", "pid=,ppid=,rss=", "-ax"])
            .output()?;
        if !output.status.success() {
            return Err(MonitorError::Sampling(
                "ps exited without a process snapshot".to_owned(),
            ));
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let mut entries = Vec::new();
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            let (Some(pid), Some(parent_pid), Some(rss_kib)) =
                (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            let (Ok(pid), Ok(parent_pid), Ok(rss_kib)) = (
                pid.parse::<u32>(),
                parent_pid.parse::<u32>(),
                rss_kib.parse::<u64>(),
            ) else {
                continue;
            };
            entries.push(ProcessEntry {
                pid,
                parent_pid,
                resident_bytes: rss_kib.saturating_mul(1024),
            });
        }
        Ok(entries)
    }
}
