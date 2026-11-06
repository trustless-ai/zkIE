//! O6 measurement: reuse the forward Projection product h = x @ w in the
//! reverse proof instead of recomputing it.
//!
//! Run with `ZKIE_O6_CACHE_MB=0` (baseline: proof recomputes h) and with the
//! default 256 MB budget (proof reuses the forward h). Proof bytes must be
//! identical in both modes.

use zkie_core::common::field::Goldilocks;
use zkie_core::common::fixed_point::from_i64;
use zkie_core::pcs::whir::Whir;
use zkie_ops::compose::{
    commit_model_weights, evaluate_ops, prove_committed_shard_dag_bound,
    verify_committed_shard_dag_bound, Op, PublicStatement, Store, TrustedExecution,
    TrustedModelManifest,
};

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn rss_hwm_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn main() {
    let (m, k, n) = (256usize, 1024usize, 256usize);
    let shift = 8u32;
    // Bounded fixed-point inputs: the Projection remainder/aux contract
    // requires the products to stay in a small signed range.
    let xv: Vec<Goldilocks> = (0..m * k).map(|i| from_i64((i as i64 % 5) - 2)).collect();
    let wv: Vec<Goldilocks> = (0..k * n).map(|i| from_i64((i as i64 % 7) - 3)).collect();
    let bv: Vec<Goldilocks> = (0..m * n).map(|i| from_i64((i as i64 % 7) - 3)).collect();

    let mut store = Store::new();
    let x = store.push(xv);
    let w = store.push(wv);
    let bias = store.push(bv);
    let out = store.push(vec![]);
    let rem = store.push(vec![]);
    let ops = vec![Op::Projection { x, w, bias, out, rem, m, k, n, shift }];

    // Forward once to obtain the expected public output; the cache is off
    // during this call, so it counts as an ordinary forward.
    evaluate_ops(&mut store, &ops);
    let expected = store.get(out).to_vec();
    store.o6_metrics_reset();

    let path = std::env::temp_dir().join("zkie-o6-projection-example");
    let cache = commit_model_weights(&store, &ops, path.to_str().unwrap());
    let model = TrustedModelManifest::from_offline_commitment(&cache, &store.owned_slices(), &ops);
    let execution = TrustedExecution {
        inputs: [(x, store.get(x).to_vec())].into(),
        outputs: [(out, expected)].into(),
    };
    let whir = Whir::with_params(5, 90, 17);
    let t = std::time::Instant::now();
    let proof = prove_committed_shard_dag_bound(
        &mut store, &ops, 1, &whir, &cache, 0, 0,
        &PublicStatement::Generic, &model, &execution,
    )
    .expect("O6 projection proof");
    let ms = t.elapsed().as_secs_f64() * 1e3;
    assert!(verify_committed_shard_dag_bound(
        &ops, 1, &whir, &proof, &PublicStatement::Generic, &model, &execution
    ));
    let bytes = bincode::serialize(&proof).unwrap();
    let (fwd, recompute, hits) = store.o6_metrics();
    let (cached, cached_bytes) = store.projection_cache_stats();
    let cache_mb = std::env::var("ZKIE_O6_CACHE_MB").unwrap_or_else(|_| "256".into());
    println!(
        "O6 cache_mb={} m={} k={} n={} fwd_mm={} proof_mm={} hits={} cached={} cached_bytes={} ms={:.3} proof_len={} proof_hash={:016x} hwm_kb={}",
        cache_mb, m, k, n, fwd, recompute, hits, cached, cached_bytes, ms,
        bytes.len(), fnv(&bytes), rss_hwm_kb()
    );
}
