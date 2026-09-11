use std::fs;
use std::path::PathBuf;

use zkie_runtime::{
    default_aging_seconds, overall_state, parse_args, render_status, CliCommand, CliError, JobKind,
    JobState, RunConfig, RunDb, GIB,
};

fn temp_db(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-cli-{label}-{}-{:?}.sqlite",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = fs::remove_file(&path);
    path
}

#[test]
fn the_documented_prove_queue_command_converts_gib_to_exact_bytes() {
    let (command, config) = parse_args([
        "prove-queue",
        "--memory-budget-gib",
        "400",
        "--memory-hard-limit-gib",
        "440",
        "--cpu-budget-cores",
        "56",
        "--aggregation-fan-in",
        "4",
    ])
    .unwrap();

    assert_eq!(command, CliCommand::ProveQueue);
    assert_eq!(config.memory_budget_bytes, 429_496_729_600);
    assert_eq!(config.memory_hard_limit_bytes, 472_446_402_560);
    assert_eq!(config.memory_budget_bytes, 400 * GIB);
    assert_eq!(config.cpu_budget_cores, 56);
    assert_eq!(config.aggregation_fan_in, 4);
    assert!(config.cuda_devices.is_empty());
    assert!(!config.allow_best_effort_hard_limit);
    assert_eq!(config.memory_limits().unwrap().hard_limit_bytes, 440 * GIB);

    assert_eq!(RunConfig::defaults(), config);
    assert_eq!(default_aging_seconds(), 600);
    for (text, expected) in [
        ("plan", CliCommand::Plan),
        ("prepare", CliCommand::Prepare),
        ("status", CliCommand::Status),
        ("verify", CliCommand::Verify),
        ("worker", CliCommand::Worker),
    ] {
        assert_eq!(CliCommand::parse(text), Some(expected));
        assert_eq!(expected.as_str(), text);
    }
}

#[test]
fn invalid_budgets_devices_and_fan_in_are_rejected() {
    assert_eq!(
        parse_args(["prove-queue", "--memory-hard-limit-gib", "400"]).unwrap_err(),
        CliError::HardLimitNotAboveAdmission
    );
    assert_eq!(
        parse_args(["prove-queue", "--cpu-budget-cores", "0"]).unwrap_err(),
        CliError::ZeroCpuBudget
    );
    assert_eq!(
        parse_args(["prove-queue", "--aggregation-fan-in", "1"]).unwrap_err(),
        CliError::InvalidFanIn(1)
    );
    assert_eq!(
        parse_args(["prove-queue", "--aggregation-fan-in", "17"]).unwrap_err(),
        CliError::InvalidFanIn(17)
    );
    assert_eq!(
        parse_args(["prove-queue", "--cuda-device", "0", "--cuda-device", "0"]).unwrap_err(),
        CliError::DuplicateCudaDevice(0)
    );
    assert_eq!(
        parse_args(["prove-queue", "--cuda-device", "1", "--cuda-device", "2"])
            .unwrap()
            .1
            .cuda_devices,
        vec![1, 2]
    );
    assert_eq!(
        parse_args::<[&str; 0], &str>([]).unwrap_err(),
        CliError::MissingCommand
    );
    assert_eq!(
        parse_args(["explode"]).unwrap_err(),
        CliError::UnknownCommand("explode".to_owned())
    );
    assert_eq!(
        parse_args(["status", "--turbo"]).unwrap_err(),
        CliError::UnknownArgument("--turbo".to_owned())
    );
    assert_eq!(
        parse_args(["status", "--cpu-budget-cores"]).unwrap_err(),
        CliError::MissingValue("--cpu-budget-cores")
    );
    assert_eq!(
        parse_args(["status", "--cpu-budget-cores", "many"]).unwrap_err(),
        CliError::InvalidInteger("--cpu-budget-cores", "many".to_owned())
    );
}

#[test]
fn the_best_effort_fallback_is_parsed_and_visible_in_the_configuration() {
    let (_, config) = parse_args(["prove-queue", "--allow-best-effort-hard-limit"]).unwrap();
    assert!(config.allow_best_effort_hard_limit);
    assert!(!RunConfig::defaults().allow_best_effort_hard_limit);
}

#[test]
fn status_output_is_stable_and_reports_every_failure_field() {
    let path = temp_db("status");
    let mut db = RunDb::open(&path, "cli-status").unwrap();
    let leaf = db.insert_job("leaf", JobKind::LeafProof).unwrap();
    let aggregate = db
        .insert_job("aggregate", JobKind::NativeAggregate)
        .unwrap();
    db.add_dependency(&leaf, &aggregate).unwrap();
    db.refresh_readiness().unwrap();

    let records = [db.job(&leaf).unwrap(), db.job(&aggregate).unwrap()];
    assert_eq!(overall_state(&records), "running");

    let rendered = render_status(&records, default_aging_seconds());
    assert!(rendered.contains("aging_seconds=600"));
    assert!(rendered.contains("job leaf kind=leaf-proof state=ready attempts=0"));
    assert!(rendered.contains("blocked_by=none"));
    assert!(rendered.contains("job aggregate kind=native-aggregate state=pending"));

    // A dependent whose predecessor is blocked is reported as blocked, not running.
    let raw = rusqlite::Connection::open(&path).unwrap();
    raw.execute(
        "UPDATE jobs SET state='verification-failed', terminal_failure=1 WHERE job_id=?1",
        [leaf.as_str()],
    )
    .unwrap();
    drop(raw);
    let records = [db.job(&leaf).unwrap(), db.job(&aggregate).unwrap()];
    assert_eq!(overall_state(&records), "failed");
    let rendered = render_status(&records, default_aging_seconds());
    assert!(rendered.contains("state=verification-failed"));

    // With every job verified the run reports success.
    let raw = rusqlite::Connection::open(&path).unwrap();
    raw.execute("UPDATE jobs SET state='verified'", []).unwrap();
    drop(raw);
    let records = [db.job(&leaf).unwrap(), db.job(&aggregate).unwrap()];
    assert_eq!(overall_state(&records), "verified");
    assert_eq!(records[0].state, JobState::Verified);

    drop(db);
    let _ = fs::remove_file(path);
}
