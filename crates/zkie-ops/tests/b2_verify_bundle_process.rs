//! B2: the independent verifier process reads only bundle.bin + proof.bin and
//! fails closed on missing/corrupt artifacts. A valid bundle acceptance is
//! covered by the in-crate b2_public_bundle_independent_verifier_and_tamper_matrix
//! test (same verify_bundle_dir entry the binary calls).

use std::path::Path;
use std::process::Command;

fn run(dir: &Path) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_verify_bundle"))
        .arg(dir)
        .output()
        .expect("spawn verify_bundle");
    (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn verify_bundle_process_rejects_missing_and_corrupt_artifacts() {
    let dir = std::env::temp_dir()
        .join(format!("zkie-b2-bin-missing-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (code, out) = run(&dir);
    assert_eq!(code, Some(1));
    assert!(out.contains("REJECT"));

    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("bundle.bin"), b"not a bundle").unwrap();
    std::fs::write(dir.join("proof.bin"), b"not a proof").unwrap();
    let (code, out) = run(&dir);
    assert_eq!(code, Some(1));
    assert!(out.contains("REJECT"));

    // A well-formed bundle with a corrupt proof must also reject (the proof
    // decoder is bounded and the submission JSON/bytes never panic).
    let bundle_dir = std::env::temp_dir()
        .join(format!("zkie-b2-bin-proof-{}", std::process::id()));
    std::fs::create_dir_all(&bundle_dir).unwrap();
    std::fs::write(bundle_dir.join("bundle.bin"), b"not a bundle").unwrap();
    let good_proof = vec![0u8; 32];
    std::fs::write(bundle_dir.join("proof.bin"), &good_proof).unwrap();
    let (code, out) = run(&bundle_dir);
    assert_eq!(code, Some(1));
    assert!(out.contains("REJECT"));
}
