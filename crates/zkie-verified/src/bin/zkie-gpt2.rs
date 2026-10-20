use serde::Deserialize;
use std::path::Path;
use zkie_verified::{graph, ModelId};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    input: Vec<i64>,
    context: [u8; 32],
}
fn read(path: &Path, limit: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("input exceeds limit".into());
    }
    Ok(bytes)
}
#[cfg(feature = "prover")]
fn write(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
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
        let key = graph::VerifierKey::from_bytes(&read(Path::new(&args[2]), 16 << 20)?)?;
        let id = ModelId::from_hex(args[3].to_str().ok_or("id must be UTF-8")?)?.0;
        let expected: Request = serde_json::from_slice(&read(Path::new(&args[4]), 32 << 20)?)?;
        let statement: graph::Statement =
            serde_json::from_slice(&read(Path::new(&args[5]), 64 << 20)?)?;
        if expected.input != statement.input || expected.context != statement.context {
            return Err("request mismatch".into());
        }
        let bytes = read(Path::new(&args[6]), 512 << 20)?;
        let start = std::time::Instant::now();
        graph::verify(&key, id, &statement, &bytes)?;
        println!(
            "{}",
            serde_json::json!({"verified":true,"output":statement.output,"proof_bytes":bytes.len(),"verify_us":start.elapsed().as_micros()})
        );
        return Ok(());
    }
    #[cfg(feature = "prover")]
    if (args.len() == 5 && args[1] == "prepare")
        || (args.len() == 6 && args[1] == "prove")
        || (args.len() == 4 && args[1] == "inspect")
    {
        let config: zkie_verified::transformer::Config =
            serde_json::from_slice(&read(Path::new(&args[2]), 4096)?)?;
        let (program, constants) =
            zkie_verified::transformer::compile_directory(&config, Path::new(&args[3]))?;
        program.validate()?;
        let coefficients: usize = constants.iter().map(|(_, v)| v.len()).sum();
        if args[1] == "inspect" {
            println!(
                "{}",
                serde_json::json!({"valid":true,"nodes":program.nodes.len(),"constant_coefficients":coefficients,"configuration":config,"proof_executed":false})
            );
            return Ok(());
        }
        let start = std::time::Instant::now();
        let (pk, key) = graph::prepare(program, constants)?;
        let prepare_us = start.elapsed().as_micros();
        let output = Path::new(args.last().unwrap());
        std::fs::create_dir(output)?;
        write(&output.join("key.bin"), &key.to_bytes()?)?;
        let mut result = serde_json::json!({"model_id":ModelId(key.id()?).to_hex(),"key_bytes":key.to_bytes()?.len(),"prepare_us":prepare_us});
        if args[1] == "prove" {
            let request: Request = serde_json::from_slice(&read(Path::new(&args[4]), 32 << 20)?)?;
            let start = std::time::Instant::now();
            let (s, proof) = graph::prove(&pk, request.input, request.context)?;
            write(&output.join("statement.json"), &serde_json::to_vec(&s)?)?;
            write(&output.join("proof.bin"), &proof)?;
            result["output"] = serde_json::json!(s.output);
            result["proof_bytes"] = serde_json::json!(proof.len());
            result["prove_us"] = serde_json::json!(start.elapsed().as_micros());
        }
        println!("{result}");
        return Ok(());
    }
    Err("usage: zkie-gpt2 verify KEY.bin TRUSTED_ID EXPECTED_REQUEST.json STATEMENT.json PROOF.bin\nwith prover: zkie-gpt2 inspect CONFIG.json WEIGHTS_DIRECTORY\n             zkie-gpt2 prepare CONFIG.json WEIGHTS_DIRECTORY NEW_DIRECTORY\n             zkie-gpt2 prove CONFIG.json WEIGHTS_DIRECTORY REQUEST.json NEW_DIRECTORY".into())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("GPT-2 proof failed: {e}");
        std::process::exit(1);
    }
}
