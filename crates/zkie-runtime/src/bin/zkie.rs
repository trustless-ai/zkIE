//! `zkie` — resumable, resource-aware proving queue.
//!
//! `status` and the internal `worker` entry point are wired end to end. The commands
//! that drive the queue (`plan`, `prepare`, `prove-queue`, `verify`) still need the
//! backend registry that lands with the calibration work, so they report that
//! explicitly rather than pretending to run.

use std::path::Path;
use std::process::ExitCode;

use zkie_runtime::{
    default_aging_seconds, overall_state, parse_args, render_status, CliCommand, RunConfig, RunDb,
    WorkerMeasurements, WorkerOutcome, WorkerResult, WorkerSpec,
};

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (command, config) = match parse_args(args) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("zkie: {error}");
            return ExitCode::from(2);
        }
    };
    match execute(command, &config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("zkie: {message}");
            ExitCode::FAILURE
        }
    }
}

fn execute(command: CliCommand, config: &RunConfig) -> Result<(), String> {
    match command {
        CliCommand::Status => status(config),
        CliCommand::Worker => worker(config),
        other => Err(format!(
            "`{}` is not wired to the queue engine yet",
            other.as_str()
        )),
    }
}

fn status(config: &RunConfig) -> Result<(), String> {
    let run_dir = config
        .run_dir
        .as_deref()
        .ok_or_else(|| "status requires --run-dir".to_owned())?;
    let db = open_run(run_dir)?;
    let records = db.jobs().map_err(|error| error.to_string())?;
    println!("state: {}", overall_state(&records));
    println!("{}", render_status(&records, default_aging_seconds()));
    Ok(())
}

fn worker(config: &RunConfig) -> Result<(), String> {
    let spec_path = config
        .spec_path
        .as_deref()
        .ok_or_else(|| "worker requires --spec".to_owned())?;
    let result_path = config
        .result_path
        .as_deref()
        .ok_or_else(|| "worker requires --result".to_owned())?;
    let text = std::fs::read_to_string(spec_path).map_err(|error| error.to_string())?;
    let spec = WorkerSpec::decode(&text).map_err(|error| error.to_string())?;
    // The proving backends are registered by the calibration work; until then a worker
    // reports a typed failure instead of silently reporting success.
    let result = WorkerResult::failed(&spec, "backend-unavailable", WorkerMeasurements::default());
    let encoded = result.encode().map_err(|error| error.to_string())?;
    std::fs::write(result_path, encoded).map_err(|error| error.to_string())?;
    match result.outcome {
        WorkerOutcome::Succeeded => Ok(()),
        WorkerOutcome::Failed { code } => Err(format!("worker failed: {code}")),
    }
}

fn open_run(run_dir: &Path) -> Result<RunDb, String> {
    let run_id = run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "run directory must have a name".to_owned())?;
    RunDb::open(run_dir.join("run.sqlite"), run_id).map_err(|error| error.to_string())
}
