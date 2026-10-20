mod common;

use std::{fs, io::Write, path::Path, time::Instant};
use zkie_verified::{evaluate, prove, Model};

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        return Err("usage: zkie-prove MODEL.json REQUEST.json NEW_OUTPUT_DIRECTORY".into());
    }
    let model: Model = common::read_json(Path::new(&args[1]))?;
    let request: common::Request = common::read_json(Path::new(&args[2]))?;
    let id = model.id()?;
    let start = Instant::now();
    let expected = evaluate(&model, request.input)?;
    let evaluate_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let (statement, proof) = prove(&model, request.input, request.context)?;
    let prove_us = start.elapsed().as_micros();
    if statement.output != expected {
        return Err("executor/prover output mismatch".into());
    }
    // Never overwrite a previous bundle. A failed write may leave a partial new
    // directory; verification still requires all complete validated artifacts.
    let out = Path::new(&args[3]);
    fs::create_dir(out)?;
    write_new(&out.join("model.json"), &serde_json::to_vec_pretty(&model)?)?;
    write_new(
        &out.join("statement.json"),
        &serde_json::to_vec_pretty(&statement)?,
    )?;
    write_new(&out.join("proof.bin"), &proof)?;
    println!(
        "{}",
        serde_json::json!({"model_id": id.to_hex(), "output": statement.output,
        "proof_bytes": proof.len(), "evaluate_ns": evaluate_ns, "prove_us": prove_us})
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("proof generation failed: {error}");
        std::process::exit(1);
    }
}
