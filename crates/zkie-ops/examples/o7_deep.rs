//! Deep O7 (d): N same-table static Lookups over one synthetic silu-like
//! table (default 2^24 entries, 2^18 witness rows per lookup).
//!
//! Measures the table-side work that deep O7 aggregates: table-side fraction
//! trees, table openings, padded-table scan bytes (commit scan + fraction-tree
//! build + leaf/root evaluation), proof bytes, prove/verify time and HWM.
//! The same source runs against the 184bf82 baseline worktree (per-lookup
//! table trees/openings) and the current deep (b)+(c) code (one shared tree
//! and one group opening).
//!
//! Env:
//!   ZKIE_O7_N            number of same-table Lookups (default 2, range 2..=4)
//!   ZKIE_O7_TABLE_BITS   static table log2 size (default 24)
//!   ZKIE_O7_IDX_BITS     witness rows per lookup log2 (default 18)
//!   ZKIE_O7_DUMP         optional path to write the serialized proof bytes

use std::time::Instant;
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::pcs::whir::Whir;
use zkie_ops::compose::{
    evaluate_ops, prove_committed_shard_dag_bound, verify_committed_shard_dag_bound, Op,
    PublicStatement, Store, TensorKey, TrustedExecution, TrustedModelManifest,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

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

/// Fixed-point siLU-like table: t[i] = round(2^16 * silu((i - len/2)/4096)).
/// Deterministic and version-independent (same values in every worktree).
fn silu_table(bits: usize) -> Vec<Goldilocks> {
    let len = 1usize << bits;
    let center = (len / 2) as f64;
    (0..len)
        .map(|i| {
            let x = (i as f64 - center) / 4096.0;
            let y = x / (1.0 + (-x).exp());
            from_i64((y * 65536.0).round() as i64)
        })
        .collect()
}

fn main() {
    let n = env_usize("ZKIE_O7_N", 2).clamp(2, 4);
    let table_bits = env_usize("ZKIE_O7_TABLE_BITS", 24);
    let idx_bits = env_usize("ZKIE_O7_IDX_BITS", 18);
    let table_len = 1usize << table_bits;
    let idx_len = 1usize << idx_bits;

    let mut store = Store::new();
    let t = store.push(silu_table(table_bits));
    let mask = (table_len - 1) as u64;
    let mut idxs = Vec::with_capacity(n);
    let mut outs = Vec::with_capacity(n);
    for j in 0..n {
        // Distinct deterministic permutations of the table domain.
        let mul = (2 * j + 1) as u64;
        let off = (j as u64 * 7919) & mask;
        let idx: Vec<u32> = (0..idx_len)
            .map(|i| ((i as u64 * mul + off) & mask) as u32)
            .collect();
        idxs.push(store.push_idx(idx));
        outs.push(store.push(vec![]));
    }
    let ops: Vec<Op> = (0..n)
        .map(|j| Op::Lookup { idx: idxs[j], out: outs[j], table: t })
        .collect();
    evaluate_ops(&mut store, &ops);

    let shapes: std::collections::BTreeMap<usize, usize> = idxs
        .iter()
        .map(|&i| (i, idx_len))
        .collect();
    let model = TrustedModelManifest::from_offline_commitment(&[], &store.owned_slices(), &ops)
        .with_index_shapes(&shapes);
    let mut inputs = std::collections::BTreeMap::new();
    for &i in &idxs {
        let vals: Vec<Goldilocks> = store.idx[i]
            .iter()
            .map(|&v| Goldilocks::from_u64(v as u64))
            .collect();
        inputs.insert(TensorKey::Index(i).encoded().unwrap(), vals);
    }
    let mut outputs = std::collections::BTreeMap::new();
    for &o in &outs {
        outputs.insert(o, store.get(o).to_vec());
    }
    let execution = TrustedExecution { inputs, outputs };
    let whir = Whir::with_params(5, 90, 17);

    store.o7_metrics_reset();
    let t0 = Instant::now();
    let proof = prove_committed_shard_dag_bound(
        &mut store, &ops, 1, &whir, &[], 0, 0,
        &PublicStatement::Generic, &model, &execution,
    )
    .expect("deep O7 proof");
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
    let bytes = bincode::serialize(&proof).expect("serialize proof");
    let t1 = Instant::now();
    assert!(
        verify_committed_shard_dag_bound(
            &ops, 1, &whir, &proof, &PublicStatement::Generic, &model, &execution,
        ),
        "deep O7 proof must verify"
    );
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
    let (trees, openings, scan_bytes) = store.o7_deep_metrics();
    if let Ok(path) = std::env::var("ZKIE_O7_DUMP") {
        std::fs::write(path, &bytes).expect("write proof dump");
    }
    println!(
        "O7DEEP n={} table_bits={} idx_bits={} table_bytes={} commits={} trees={} openings={} scan_bytes={} proof_len={} proof_hash={:016x} prove_ms={:.1} verify_ms={:.1} hwm_kb={}",
        n,
        table_bits,
        idx_bits,
        table_len * std::mem::size_of::<Goldilocks>(),
        store.o7_table_commits(),
        trees,
        openings,
        scan_bytes,
        bytes.len(),
        fnv(&bytes),
        prove_ms,
        verify_ms,
        rss_hwm_kb(),
    );
}
