use serde::Deserialize;
use std::path::Path;
use zkie_verified::{matmul, ModelId};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    m: usize,
    input: Vec<i32>,
    context: [u8; 32],
}

#[cfg(feature = "prover")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Weights {
    k: usize,
    n: usize,
    weights: Vec<i32>,
}

#[cfg(feature = "prover")]
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() == 7 && args[1] == "verify" {
        let key = matmul::VerifierKey::from_bytes(&read_bounded(Path::new(&args[2]), 4096)?)?;
        let id = ModelId::from_hex(args[3].to_str().ok_or("id must be UTF-8")?)?.0;
        let expected: Request =
            serde_json::from_slice(&read_bounded(Path::new(&args[4]), 16 << 20)?)?;
        let statement: matmul::Statement =
            serde_json::from_slice(&read_bounded(Path::new(&args[5]), 32 << 20)?)?;
        if expected.m != statement.m
            || expected.input != statement.input
            || expected.context != statement.context
        {
            return Err("request mismatch".into());
        }
        let bytes = read_bounded(Path::new(&args[6]), 4 << 20)?;
        let start = std::time::Instant::now();
        matmul::verify(&key, id, &statement, &bytes)?;
        println!(
            "{}",
            serde_json::json!({"verified":true,"output":statement.output,
            "key_bytes":key.to_bytes()?.len(),"proof_bytes":bytes.len(),"verify_us":start.elapsed().as_micros()})
        );
        return Ok(());
    }
    #[cfg(feature = "prover")]
    if (args.len() == 4 && args[1] == "prepare") || (args.len() == 5 && args[1] == "prove") {
        let w: Weights = serde_json::from_slice(&read_bounded(Path::new(&args[2]), 16 << 20)?)?;
        let start = std::time::Instant::now();
        let (pk, key) = matmul::prepare(w.k, w.n, w.weights)?;
        let prepare_us = start.elapsed().as_micros();
        let output = Path::new(args.last().unwrap());
        std::fs::create_dir(output)?;
        write_new(&output.join("key.bin"), &key.to_bytes()?)?;
        let mut result = serde_json::json!({"model_id":ModelId(key.id()?).to_hex(),
            "key_bytes":key.to_bytes()?.len(),"prepare_us":prepare_us});
        if args[1] == "prove" {
            let request: Request =
                serde_json::from_slice(&read_bounded(Path::new(&args[3]), 16 << 20)?)?;
            let start = std::time::Instant::now();
            let (statement, proof) = pk.prove(request.m, request.input, request.context)?;
            let prove_us = start.elapsed().as_micros();
            write_new(
                &output.join("statement.json"),
                &serde_json::to_vec(&statement)?,
            )?;
            write_new(&output.join("proof.bin"), &proof)?;
            result["output"] = serde_json::json!(statement.output);
            result["proof_bytes"] = serde_json::json!(proof.len());
            result["prove_us"] = serde_json::json!(prove_us);
        }
        println!("{result}");
        return Ok(());
    }
    Err("usage: zkie-linear verify KEY.bin TRUSTED_ID EXPECTED_REQUEST.json STATEMENT.json PROOF.bin\nwith prover feature: zkie-linear prepare WEIGHTS.json NEW_DIRECTORY\n                     zkie-linear prove WEIGHTS.json REQUEST.json NEW_DIRECTORY".into())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("committed linear proof failed: {e}");
        std::process::exit(1);
    }
}

fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("input exceeds size limit".into());
    }
    Ok(bytes)
}
