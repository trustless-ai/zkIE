//! B2: prove the full Gemma 3 graph under a LanguageModel statement and write
//! the public verification bundle (bundle.bin + proof.bin). The independent
//! verifier process is `verify_bundle <bundle-dir>` (zkie-ops).

use std::path::Path;
use std::time::Instant;

use zkie_core::pcs::whir::Whir;
use zkie_models_gemma3::build_gemma3;
use zkie_ops::bundle::{
    index_shapes, public_execution, verify_bundle_dir_with_model, write_bundle, PublicBundle,
};
use zkie_ops::compose::{
    commit_model_weights, evaluate_ops, prove_committed_shard_dag_bound, PublicStatement,
    Store, TrustedModelManifest,
};

fn main() {
    let m = std::env::var("GEMMA_SEQ").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let dir = "models/gemma3/weights";
    let bundle_dir = std::env::var("ZKIE_BUNDLE_DIR")
        .unwrap_or_else(|_| "/tmp/zkie-gemma3-bundle".to_string());
    let cache_dir = std::env::var("ZKIE_CACHE_DIR")
        .unwrap_or_else(|_| "/tmp/zkie-b2-cache-gemma3".to_string());
    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, _logits, tokens_t, _t) = build_gemma3(&mut store, &mut ops, dir, m, 16);
    evaluate_ops(&mut store, &ops);
    let cached = commit_model_weights(&store, &ops, &cache_dir);
    let mut model = TrustedModelManifest::from_offline_commitment(
        &cached, &store.owned_slices(), &ops).with_index_shapes(&index_shapes(&store));
    match zkie_ops::tables::try_install_a_tables(Path::new(dir), &ops, &mut model) {
        Ok(_) => {}
        Err(e) => { eprintln!("A-table sidecar rejected: {e}"); std::process::exit(2); }
    }
    let execution = match public_execution(&store, &ops, &cached, &model) {
        Some(e) => e,
        None => {
            eprintln!("graph contains an op without a bound reverse contract; failing closed");
            std::process::exit(2);
        }
    };
    let tokens = store.idx.get(tokens_t).cloned().unwrap_or_default();
    if tokens.len() != m || m < 2 {
        eprintln!("expected a {m}-token public sequence, got {}", tokens.len());
        std::process::exit(2);
    }
    let stmt = PublicStatement::LanguageModel {
        prompt: tokens[..1].to_vec(),
        response: tokens[1..].to_vec(),
        next_token: None,
    };
    let whir = Whir::new_testing(5);
    let t0 = Instant::now();
    let proof = prove_committed_shard_dag_bound(
        &mut store, &ops, 1, &whir, &cached, 0, 0, &stmt, &model, &execution)
        .expect("Gemma3 bound proof");
    let elapsed = t0.elapsed();
    let bytes = bincode::serialize(&proof).expect("serialize proof");
    let bundle = PublicBundle::new(ops.clone(), 1, stmt, model, execution);
    write_bundle(Path::new(&bundle_dir), &bundle, &bytes).expect("write bundle");
    let model_dir = Path::new(dir).join("tables.meta").exists().then_some(Path::new(dir));
    let ok = verify_bundle_dir_with_model(Path::new(&bundle_dir), model_dir);
    println!("gemma3 bundle={bundle_dir} prove={elapsed:?} proof_bytes={} independent_verify={ok}",
        bytes.len());
}
