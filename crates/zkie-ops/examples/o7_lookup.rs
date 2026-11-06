//! O7 measurement: two Lookups into the SAME trusted static table. Per-proof
//! table commitment sharing landed first; deep O7 (b) aggregated the
//! multiplicity (M = m1 + m2) and shared one table-side fraction tree with a
//! common alpha/beta; (c) reduced the table opening to one group-level copy.
//! Step (d) reworks this harness into the full before/after measurement;
//! until then its counter shows the current commit count.
//!
//! Run with `ZKIE_O7_TABLE_CACHE=0` and with the default cache. Proof bytes
//! must be identical.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::pcs::whir::Whir;
use zkie_ops::compose::{
    commit_model_weights, evaluate_ops, prove_committed_shard_dag_bound,
    verify_committed_shard_dag_bound, Op, PublicStatement, Store, TensorKey, TrustedExecution,
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
    let len = 1usize << 18;
    let table: Vec<Goldilocks> = (0..len).map(|i| from_i64(1000 + (i % 1000) as i64)).collect();
    let idx1: Vec<u32> = (0..len).map(|i| ((i * 2654435761) & (len - 1)) as u32).collect();
    let idx2: Vec<u32> = (0..len).map(|i| ((i * 40503 + 7) & (len - 1)) as u32).collect();

    let mut store = Store::new();
    let t = store.push(table);
    let i1 = store.push_idx(idx1);
    let i2 = store.push_idx(idx2);
    let o1 = store.push(vec![]);
    let o2 = store.push(vec![]);
    let ops = vec![
        Op::Lookup { idx: i1, out: o1, table: t },
        Op::Lookup { idx: i2, out: o2, table: t },
    ];
    evaluate_ops(&mut store, &ops);
    let out1 = store.get(o1).to_vec();
    let out2 = store.get(o2).to_vec();
    store.o7_metrics_reset();

    let path = std::env::temp_dir().join("zkie-o7-lookup-example");
    let cache = commit_model_weights(&store, &ops, path.to_str().unwrap());
    let model = TrustedModelManifest::from_offline_commitment(&cache, &store.owned_slices(), &ops)
        .with_index_shapes(&[(i1, len), (i2, len)].into());
    let k1 = TensorKey::Index(i1).encoded().unwrap();
    let k2 = TensorKey::Index(i2).encoded().unwrap();
    let g1: Vec<Goldilocks> = store.idx[i1].iter().map(|&v| Goldilocks::from_u64(v as u64)).collect();
    let g2: Vec<Goldilocks> = store.idx[i2].iter().map(|&v| Goldilocks::from_u64(v as u64)).collect();
    let execution = TrustedExecution {
        inputs: [(k1, g1), (k2, g2)].into(),
        outputs: [(o1, out1), (o2, out2)].into(),
    };
    let whir = Whir::with_params(5, 90, 17);
    let t0 = std::time::Instant::now();
    let proof = prove_committed_shard_dag_bound(
        &mut store, &ops, 2, &whir, &cache, 0, 0,
        &PublicStatement::Generic, &model, &execution,
    )
    .expect("O7 same-table lookup proof");
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    assert!(verify_committed_shard_dag_bound(
        &ops, 2, &whir, &proof, &PublicStatement::Generic, &model, &execution
    ));
    let bytes = bincode::serialize(&proof).unwrap();
    let cache_setting = std::env::var("ZKIE_O7_TABLE_CACHE").unwrap_or_else(|_| "1".into());
    println!(
        "O7 table_cache={} len={} commits={} ms={:.3} proof_len={} proof_hash={:016x} hwm_kb={}",
        cache_setting, len, store.o7_table_commits(), ms, bytes.len(), fnv(&bytes), rss_hwm_kb()
    );
}
