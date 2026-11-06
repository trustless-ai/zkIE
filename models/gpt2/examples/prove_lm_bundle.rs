//! B2: prove the full GPT-2 graph under a LanguageModel statement and write
//! the public verification bundle. LayerNormCentered is covered by the LNC
//! reverse reduction; this example exits 2 only if public execution
//! extraction fails closed for another reason.

use std::path::Path;
use std::time::Instant;

use zkie_core::pcs::whir::Whir;
use zkie_models_gpt2::build_gpt2;
use zkie_ops::bundle::{
    index_shapes, public_execution, verify_bundle_dir_with_model, write_bundle, PublicBundle,
};
use zkie_ops::compose::{
    commit_model_weights, evaluate_ops, prove_committed_shard_dag_bound, PublicStatement,
    Store, TrustedModelManifest,
};

fn main() {
    let m = std::env::var("M").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let dir = "models/gpt2/weights";
    let bundle_dir = std::env::var("ZKIE_BUNDLE_DIR")
        .unwrap_or_else(|_| "/tmp/zkie-gpt2-bundle".to_string());
    let cache_dir = std::env::var("ZKIE_CACHE_DIR")
        .unwrap_or_else(|_| "/tmp/zkie-b2-cache-gpt2".to_string());
    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, _logits, tokens_t, _t) = build_gpt2(&mut store, &mut ops, dir, m, 16);
    evaluate_ops(&mut store, &ops);
    let cached = commit_model_weights(&store, &ops, &cache_dir);
    let mut model = TrustedModelManifest::from_offline_commitment(
        &cached, &store.owned_slices(), &ops).with_index_shapes(&index_shapes(&store));
    let stats = std::env::var("ZKIE_MODEL_STATS").is_ok();
    let print_stats = |tag: &str, model: &TrustedModelManifest| {
        let t = model.transcript_bytes().expect("manifest transcript");
        let mut h = 0xcbf29ce484222325u64;
        for &b in &t { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
        let k = zkie_ops::compose::opening_upper_bound(&ops, model)
            .expect("opening upper bound");
        println!("gpt2 {tag}: transcript_len={} transcript_fnv={:016x} K={k} a_tables={}",
            t.len(), h, model.a_tables.len());
    };
    if stats { print_stats("pre-A", &model); }
    match zkie_ops::tables::try_install_a_tables(Path::new(dir), &ops, &mut model) {
        Ok(_) => {}
        Err(e) => { eprintln!("A-table sidecar rejected: {e}"); std::process::exit(2); }
    }
    if stats { print_stats("post-A", &model); return; }
    if std::env::var("ZKIE_BUDGET_ESTIMATE").is_ok() {
        let ub = zkie_ops::compose::opening_upper_bound(&ops, &model)
            .expect("opening upper bound");
        println!("gpt2 opening upper bound = {ub} (cap 64)");
        return;
    }
    let execution = match public_execution(&store, &ops, &cached, &model) {
        Some(e) => e,
        None => {
            eprintln!("GPT-2 public execution extraction failed closed; no bundle written");
            std::process::exit(2);
        }
    };
    let tokens = store.idx.get(tokens_t).cloned().unwrap_or_default();
    if tokens.len() != m || m < 2 {
        eprintln!("expected a {m}-token public sequence, got {}", tokens.len());
        std::process::exit(2);
    }
    // The real prompt for the canonical input is "The sky is" (3 GPT-2
    // tokens); the teacher-forcing check runs from prompt_len-1. Defaults to
    // the canonical prompt; PROMPT_LEN overrides for other inputs.
    let prompt_len = std::env::var("PROMPT_LEN").ok()
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(3);
    if prompt_len == 0 || prompt_len >= m {
        eprintln!("prompt_len={prompt_len} out of range for m={m}");
        std::process::exit(2);
    }
    // A reduced-layer model cannot be expected to satisfy canonical GPT-2
    // teacher forcing; ZKIE_GENERIC_STATEMENT exercises the same proof and
    // verifier machinery with a Generic statement for resource-constrained
    // e2e runs.
    let stmt = if std::env::var("ZKIE_GENERIC_STATEMENT").is_ok() {
        PublicStatement::Generic
    } else {
        PublicStatement::LanguageModel {
            prompt: tokens[..prompt_len].to_vec(),
            response: tokens[prompt_len..].to_vec(),
            next_token: None,
        }
    };
    let whir = Whir::new_testing(5);
    let t0 = Instant::now();
    let proof = prove_committed_shard_dag_bound(
        &mut store, &ops, 1, &whir, &cached, 0, 0, &stmt, &model, &execution)
        .expect("GPT-2 bound proof");
    let elapsed = t0.elapsed();
    let bytes = bincode::serialize(&proof).expect("serialize proof");
    let bundle = PublicBundle::new(ops.clone(), 1, stmt, model.clone(), execution);
    write_bundle(Path::new(&bundle_dir), &bundle, &bytes).expect("write bundle");
    let model_dir = Path::new(dir).join("tables.meta").exists().then_some(Path::new(dir));
    let t_verify = Instant::now();
    let ok = verify_bundle_dir_with_model(Path::new(&bundle_dir), model_dir);
    let verify_ms = t_verify.elapsed().as_secs_f64() * 1e3;
    let transcript = model.transcript_bytes().expect("manifest transcript");
    let mut fnv = 0xcbf29ce484222325u64;
    for &b in &transcript { fnv ^= b as u64; fnv = fnv.wrapping_mul(0x100000001b3); }
    println!("gpt2 bundle={bundle_dir} prove={elapsed:?} proof_bytes={} transcript_bytes={} transcript_fnv={:016x} independent_verify={ok} verify_ms={verify_ms:.3}",
        bytes.len(), transcript.len(), fnv);
}
