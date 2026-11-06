//! B2 independent verifier process.
//!
//! Usage: verify_bundle <bundle-dir> [model-dir]
//! Reads <bundle-dir>/bundle.bin and <bundle-dir>/proof.bin; for A-bearing
//! bundles it also reads the model distribution's tables.meta and A table
//! files from <model-dir> (or ZKIE_MODEL_DIR). No prover state or store is
//! loaded.

use std::path::Path;

fn main() {
    let dir = match std::env::args().nth(1) {
        Some(d) => d,
        None => {
            eprintln!("usage: verify_bundle <bundle-dir>");
            std::process::exit(2);
        }
    };
    let model_dir = std::env::args().nth(2)
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("ZKIE_MODEL_DIR").map(std::path::PathBuf::from));
    let ok = zkie_ops::bundle::verify_bundle_dir_with_model(
        Path::new(&dir), model_dir.as_deref());
    println!("{}", if ok { "ACCEPT" } else { "REJECT" });
    std::process::exit(if ok { 0 } else { 1 });
}
