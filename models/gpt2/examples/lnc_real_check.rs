//! Real-weight LNC coverage: pick every LayerNormCentered op out of the real
//! GPT-2 graph, keep their real activations/weights from a full forward, and
//! prove/verify one bound proof over all of them. This isolates the LNC
//! operator from the rest of the graph: the x/w/b tensors are the real
//! GPT-2 values, the rsqrt table is the real 2^23-entry table, and the only
//! public data are the 25 input activations and 25 outputs.
//!
//! Usage: cargo run --release -p zkie-models-gpt2 --example lnc_real_check

use std::collections::BTreeMap;
use std::time::Instant;

use zkie_core::pcs::whir::Whir;
use zkie_models_gpt2::build_gpt2;
use zkie_ops::compose::{
    commit_model_weights, evaluate_ops, prove_committed_shard_dag_bound,
    verify_committed_shard_dag_bound, Op, PublicStatement, Store, TrustedExecution,
    TrustedModelManifest,
};

fn main() {
    let m = std::env::var("M").ok().and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16usize);
    let mut store = Store::new();
    let mut all = Vec::new();
    let _ = build_gpt2(&mut store, &mut all, "models/gpt2/weights", m, 16);
    evaluate_ops(&mut store, &all);
    let limit = std::env::var("LNC_LIMIT").ok()
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX);
    let lnc_ops: Vec<Op> = all.iter()
        .filter(|o| matches!(o, Op::LayerNormCentered { .. }))
        .take(limit)
        .cloned().collect();
    if lnc_ops.is_empty() {
        eprintln!("no LayerNormCentered ops in the graph");
        std::process::exit(2);
    }
    let mut inputs: BTreeMap<usize, Vec<zkie_core::common::field::Goldilocks>> = BTreeMap::new();
    let mut outputs: BTreeMap<usize, Vec<zkie_core::common::field::Goldilocks>> = BTreeMap::new();
    for op in &lnc_ops {
        if let Op::LayerNormCentered { x, out, .. } = op {
            inputs.insert(*x, store.get(*x).to_vec());
            outputs.insert(*out, store.get(*out).to_vec());
        }
    }
    let cached = commit_model_weights(&store, &lnc_ops, "/tmp/zkie-lnc-real-cache");
    let model = TrustedModelManifest::from_offline_commitment(
        &cached, &store.owned_slices(), &lnc_ops)
        .with_index_shapes(&BTreeMap::new());
    let execution = TrustedExecution { inputs, outputs };
    let whir = Whir::new_testing(5);
    let t0 = Instant::now();
    let proof = prove_committed_shard_dag_bound(
        &mut store, &lnc_ops, 1, &whir, &cached, 0, 0,
        &PublicStatement::Generic, &model, &execution)
        .expect("real-weight LNC bound proof");
    let prove = t0.elapsed();
    let bytes = bincode::serialize(&proof).expect("serialize");
    let ok = verify_committed_shard_dag_bound(
        &lnc_ops, 1, &whir, &proof, &PublicStatement::Generic, &model, &execution);
    println!("LNC real-weights ops={} m={} prove={:?} bytes={} verify={}",
        lnc_ops.len(), m, prove, bytes.len(), ok);
    if !ok { std::process::exit(1); }
}
