mod common;

use std::{path::Path, time::Instant};
use zkie_verified::{verify, Model, ModelId, Statement, MAX_PROOF_BYTES};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() == 3 && args[1] == "id" {
        let model: Model = common::read_json(Path::new(&args[2]))?;
        println!("{}", model.id()?.to_hex());
        return Ok(());
    }
    if args.len() != 6 {
        return Err("usage: zkie-verify MODEL.json TRUSTED_MODEL_ID EXPECTED_REQUEST.json STATEMENT.json PROOF.bin\n       zkie-verify id MODEL.json".into());
    }
    let model: Model = common::read_json(Path::new(&args[1]))?;
    let trusted_id = ModelId::from_hex(args[2].to_str().ok_or("model id must be UTF-8")?)?;
    let expected: common::Request = common::read_json(Path::new(&args[3]))?;
    let statement: Statement = common::read_json(Path::new(&args[4]))?;
    // Independently expected request data prevents accepting an unrelated valid
    // inference. Freshness/replay policy remains the caller's responsibility.
    if statement.input != expected.input || statement.context != expected.context {
        return Err("statement does not match the expected request".into());
    }
    let proof = common::read_bounded(Path::new(&args[5]), MAX_PROOF_BYTES)?;
    let start = Instant::now();
    verify(&model, trusted_id, &statement, &proof)?;
    println!(
        "{}",
        serde_json::json!({"verified": true, "model_id": trusted_id.to_hex(),
        "output": statement.output, "proof_bytes": proof.len(),
        "verify_us": start.elapsed().as_micros()})
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("verification failed: {error}");
        std::process::exit(1);
    }
}
