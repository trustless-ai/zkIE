//! How large is a REAL proof on zkIE HEAD, and where does its size come from?
//!
//! Runs the field-element sizer (`crates/zkie-ops/src/proof_size.rs`) on 12-layer
//! GPT-2 at seq `M`, across several shard granularities, and prints the
//! per-bucket and per-op breakdown plus the whole-model verify.
//!
//! Reports FIELD ELEMENTS, not a serialised size. Constraints in a wrap circuit
//! track field elements; `bincode` framing does not — and `bincode` on the
//! sharded proof additionally drags in `cross_tensors` (the raw witness a
//! verifier never receives). Do not quote these bytes as calldata.
//!
//! The seq and the granularity list are overridable:
//!   M=512 VERIFY=1 SHARDS="2114,1436,718,176" cargo run --release --example proof_size_gpt2

use zkie_core::common::fixed_point::{to_i32, to_i64};
use zkie_models_gpt2::{build_gpt2, load_i32};
use zkie_ops::compose::{prove_shard_dag, verify_shard_dag, Store};
use zkie_ops::proof_size::shard_dag_proof_size;

const VOCAB: usize = 50257;
const VOCAB_PAD: usize = 65536;

fn peak_rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            return v.trim().trim_end_matches(" kB").parse().unwrap_or(0);
        }
    }
    0
}

fn main() {
    let m: usize = std::env::var("M").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let verify_once: bool = std::env::var("VERIFY").map(|v| v == "1").unwrap_or(false);
    let dir = "models/gpt2/weights";

    // Build once to learn the op count (for the whole-model granularity and the
    // default sweep); each proof below rebuilds its own store to avoid relying
    // on prover state surviving across granularities.
    let n_ops = {
        let mut store = Store::new();
        let mut ops = Vec::new();
        let _ = build_gpt2(&mut store, &mut ops, dir, m, 16);
        ops.len()
    };

    let gt = load_i32(&format!("{dir}/gt_argmax_{}i32.bin", if m <= 16 { "" } else { "512_" }));

    let grans: Vec<usize> = match std::env::var("SHARDS") {
        Ok(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => vec![n_ops, 1436, 718, 176],
    };

    println!("model=gpt2 seq={m} ops={n_ops} granularities={grans:?}");
    println!(
        "{:>10} {:>7} {:>11} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>13} {:>9}",
        "ops/shard", "shards", "op_fe", "claims_fe", "binds_fe", "cross_fe", "wit_fe", "tbl_fe",
        "tids", "total_bytes", "prove_s"
    );

    let mut first = true;
    for ops_per_shard in grans {
        let mut store = Store::new();
        let mut ops = Vec::new();
        let (_x0, logits, _tokens, _t) = build_gpt2(&mut store, &mut ops, dir, m, 16);

        let t0 = std::time::Instant::now();
        let proof = prove_shard_dag(&mut store, &ops, ops_per_shard);
        let prove_t = t0.elapsed().as_secs_f64();

        let sz = shard_dag_proof_size(&proof);
        println!(
            "{:>10} {:>7} {:>11} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>13} {:>9.1}",
            ops_per_shard, sz.shards, sz.op_proof_fe, sz.claims_fe, sz.shard_binds_fe,
            sz.cross_binds_fe, sz.witness_cols_fe, sz.table_cols_fe, sz.tensor_ids,
            sz.total_bytes(), prove_t
        );

        // The accounting must close: total_fe is exactly the sum of its buckets.
        assert_eq!(
            sz.total_fe(),
            sz.op_proof_fe + sz.claims_fe + sz.shard_binds_fe + sz.cross_binds_fe
                + sz.witness_cols_fe + sz.table_cols_fe,
            "accounting must close"
        );
        // Every op must have been visited exactly once (column buckets aside).
        let op_visits: usize = sz.count_by_op.iter()
            .filter(|(k, _)| **k != "WitnessColumn" && **k != "TableColumn")
            .map(|(_, c)| *c)
            .sum();
        assert_eq!(op_visits, ops.len(), "every op must be sized exactly once");

        if first {
            first = false;
            if verify_once {
                let ok = verify_shard_dag(&store.owned_slices(), &ops, ops_per_shard, &proof);
                let lg = store.get(logits);
                let mut matches = 0;
                for i in 0..m {
                    let mut best = 0usize;
                    let mut best_v = lg[i * VOCAB_PAD];
                    for j in 1..VOCAB {
                        let v = lg[i * VOCAB_PAD + j];
                        if to_i64(v) > to_i64(best_v) {
                            best_v = v;
                            best = j;
                        }
                    }
                    if best as i32 == to_i32(gt[i]) {
                        matches += 1;
                    }
                }
                println!("  verified={ok}, argmax {matches}/{m}");
                assert!(ok, "verify failed");
            }
            println!("  per bucket (fe, share of total):");
            let total = sz.total_fe().max(1);
            let mut rows: Vec<(&&str, &usize)> = sz.fe_by_op.iter().collect();
            rows.sort_by(|a, b| b.1.cmp(a.1));
            for (name, fe) in rows {
                let n = sz.count_by_op.get(*name).copied().unwrap_or(0);
                let per = if n > 0 { *fe as f64 / n as f64 } else { 0.0 };
                println!(
                    "    {:<20} {:>6} ops {:>12} fe  {:>8.0} fe/op  {:>5.1}%",
                    name, n, fe, per, 100.0 * *fe as f64 / total as f64
                );
            }
            println!("  peak RSS {} MiB", peak_rss_kb() / 1024);
        }
    }
}
