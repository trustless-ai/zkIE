//! Command-line surface for the resource-aware proving queue.
//!
//! Parsing and validation live here, away from process startup, so every rule the
//! operator depends on — exact byte conversion, fan-in bounds, the explicit best-effort
//! acknowledgement — is exercised by ordinary unit tests.

use thiserror::Error;

use std::path::PathBuf;

use crate::{JobKind, JobRecord, JobState, MemoryLimits, MonitorError, DEFAULT_AGING_SECONDS};

/// One GiB in bytes, exactly as the scheduler and monitor count it.
pub const GIB: u64 = 1_073_741_824;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliCommand {
    Plan,
    Prepare,
    ProveQueue,
    Status,
    Verify,
    Worker,
}

impl CliCommand {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::Prepare => "prepare",
            Self::ProveQueue => "prove-queue",
            Self::Status => "status",
            Self::Verify => "verify",
            Self::Worker => "worker",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        [
            Self::Plan,
            Self::Prepare,
            Self::ProveQueue,
            Self::Status,
            Self::Verify,
            Self::Worker,
        ]
        .into_iter()
        .find(|command| command.as_str() == value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunConfig {
    pub memory_budget_bytes: u64,
    pub memory_hard_limit_bytes: u64,
    pub cpu_budget_cores: u32,
    pub cuda_devices: Vec<u32>,
    pub aggregation_fan_in: u32,
    pub allow_best_effort_hard_limit: bool,
    /// Run directory for `status` / `verify`.
    pub run_dir: Option<PathBuf>,
    /// Worker-only: the immutable spec to execute.
    pub spec_path: Option<PathBuf>,
    /// Worker-only: where the result must be written.
    pub result_path: Option<PathBuf>,
}

impl RunConfig {
    /// 400 GiB admission / 440 GiB hard limit / 56 cores / fan-in 4, matching the
    /// calibration machine, with the portable fallback left unacknowledged.
    pub fn defaults() -> Self {
        Self {
            memory_budget_bytes: 400 * GIB,
            memory_hard_limit_bytes: 440 * GIB,
            cpu_budget_cores: 56,
            cuda_devices: Vec::new(),
            aggregation_fan_in: 4,
            allow_best_effort_hard_limit: false,
            run_dir: None,
            spec_path: None,
            result_path: None,
        }
    }

    pub fn memory_limits(&self) -> Result<MemoryLimits, MonitorError> {
        MemoryLimits::new(self.memory_budget_bytes, self.memory_hard_limit_bytes)
    }

    pub fn validate(&self) -> Result<(), CliError> {
        self.memory_limits()
            .map_err(|_| CliError::HardLimitNotAboveAdmission)?;
        if self.cpu_budget_cores == 0 {
            return Err(CliError::ZeroCpuBudget);
        }
        if !(2..=16).contains(&self.aggregation_fan_in) {
            return Err(CliError::InvalidFanIn(self.aggregation_fan_in));
        }
        let mut seen = Vec::new();
        for device in &self.cuda_devices {
            if seen.contains(device) {
                return Err(CliError::DuplicateCudaDevice(*device));
            }
            seen.push(*device);
        }
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CliError {
    #[error("missing command")]
    MissingCommand,
    #[error("unknown command {0}")]
    UnknownCommand(String),
    #[error("unknown argument {0}")]
    UnknownArgument(String),
    #[error("missing value for {0}")]
    MissingValue(&'static str),
    #[error("invalid integer for {0}: {1}")]
    InvalidInteger(&'static str, String),
    #[error("memory hard limit must be greater than the admission budget")]
    HardLimitNotAboveAdmission,
    #[error("CPU budget must be non-zero")]
    ZeroCpuBudget,
    #[error("aggregation fan-in {0} is outside 2..=16")]
    InvalidFanIn(u32),
    #[error("duplicate CUDA device {0}")]
    DuplicateCudaDevice(u32),
}

/// Parses `argv` (without the program name) into a command plus a validated configuration.
pub fn parse_args<I, S>(args: I) -> Result<(CliCommand, RunConfig), CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let command_text = args.next().ok_or(CliError::MissingCommand)?;
    let command = CliCommand::parse(&command_text).ok_or(CliError::UnknownCommand(command_text))?;
    let mut config = RunConfig::defaults();
    let mut remaining = args.peekable();
    while let Some(argument) = remaining.next() {
        match argument.as_str() {
            "--memory-budget-gib" => {
                let raw = remaining
                    .next()
                    .ok_or(CliError::MissingValue("--memory-budget-gib"))?;
                config.memory_budget_bytes = parse_gib("--memory-budget-gib", &raw)?;
            }
            "--memory-hard-limit-gib" => {
                let raw = remaining
                    .next()
                    .ok_or(CliError::MissingValue("--memory-hard-limit-gib"))?;
                config.memory_hard_limit_bytes = parse_gib("--memory-hard-limit-gib", &raw)?;
            }
            "--cpu-budget-cores" => {
                let raw = remaining
                    .next()
                    .ok_or(CliError::MissingValue("--cpu-budget-cores"))?;
                config.cpu_budget_cores = parse_u32("--cpu-budget-cores", &raw)?;
            }
            "--aggregation-fan-in" => {
                let raw = remaining
                    .next()
                    .ok_or(CliError::MissingValue("--aggregation-fan-in"))?;
                config.aggregation_fan_in = parse_u32("--aggregation-fan-in", &raw)?;
            }
            "--cuda-device" => {
                let raw = remaining
                    .next()
                    .ok_or(CliError::MissingValue("--cuda-device"))?;
                config.cuda_devices.push(parse_u32("--cuda-device", &raw)?);
            }
            "--allow-best-effort-hard-limit" => config.allow_best_effort_hard_limit = true,
            "--run-dir" => {
                let raw = remaining
                    .next()
                    .ok_or(CliError::MissingValue("--run-dir"))?;
                config.run_dir = Some(PathBuf::from(raw));
            }
            "--spec" => {
                let raw = remaining.next().ok_or(CliError::MissingValue("--spec"))?;
                config.spec_path = Some(PathBuf::from(raw));
            }
            "--result" => {
                let raw = remaining.next().ok_or(CliError::MissingValue("--result"))?;
                config.result_path = Some(PathBuf::from(raw));
            }
            other => return Err(CliError::UnknownArgument(other.to_owned())),
        }
    }
    config.validate()?;
    Ok((command, config))
}

fn parse_gib(flag: &'static str, raw: &str) -> Result<u64, CliError> {
    let gib = raw
        .parse::<u64>()
        .map_err(|_| CliError::InvalidInteger(flag, raw.to_owned()))?;
    gib.checked_mul(GIB)
        .ok_or_else(|| CliError::InvalidInteger(flag, raw.to_owned()))
}

fn parse_u32(flag: &'static str, raw: &str) -> Result<u32, CliError> {
    raw.parse::<u32>()
        .map_err(|_| CliError::InvalidInteger(flag, raw.to_owned()))
}

/// One line per job, ordered by logical id, so `status` output is stable across runs.
pub fn render_status(records: &[JobRecord], aging_seconds: i64) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "policy: admission_freeze_budget_bytes={} aging_seconds={}",
        RunConfig::defaults().memory_budget_bytes,
        aging_seconds
    ));
    for record in records {
        lines.push(format!(
            "job {} kind={} state={} attempts={} execution_retries={} resource_requeues={} terminal={} failure={} stage={} code={} blocked_by={}",
            record.logical_job_id,
            record.kind.as_str(),
            record.state.as_str(),
            record.attempt_count,
            record.execution_retries,
            record.resource_requeues,
            record.terminal_failure,
            record
                .failure_kind
                .map(|kind| kind.as_str())
                .unwrap_or("none"),
            record
                .failure_stage
                .map(|stage| stage.as_str())
                .unwrap_or("none"),
            record
                .failure_code
                .map(|code| code.as_str())
                .unwrap_or("none"),
            record
                .blocking_predecessor_id
                .as_ref()
                .map(|id| id.as_str())
                .unwrap_or("none"),
        ));
    }
    lines.join("\n")
}

/// Overall run state: a single terminal failure dominates, otherwise work is in flight.
pub fn overall_state(records: &[JobRecord]) -> &'static str {
    if records
        .iter()
        .any(|record| record.state == JobState::VerificationFailed)
    {
        return "failed";
    }
    if !records.is_empty()
        && records
            .iter()
            .all(|record| record.state == JobState::Verified)
    {
        return "verified";
    }
    if records.iter().any(|record| record.terminal_failure) {
        return "blocked";
    }
    "running"
}

/// Job kinds the queue schedules, in the order the plan introduces them.
pub fn schedulable_kinds() -> [JobKind; 5] {
    [
        JobKind::Witness,
        JobKind::Prepare,
        JobKind::LeafProof,
        JobKind::LeafVerification,
        JobKind::NativeAggregate,
    ]
}

/// Aging window used when the caller does not override it.
pub fn default_aging_seconds() -> i64 {
    DEFAULT_AGING_SECONDS
}
