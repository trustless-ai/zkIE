//! O1 measurement: one real-shape Argmax (m rows x 65536 logits, the GPT-2
//! final head shape). Isolates the per-token selector-opening cost that O1
//! replaces with the public-selector sumcheck. The same source runs on the
//! pre-O1 worktree (`selector_openings()` non-empty) and the O1 tree (empty).
//!
//! Env: ZKIE_O1_M (default 16), ZKIE_O1_N (default 65536).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::pcs::whir::Whir;
use zkie_ops::compose::{
    evaluate_ops, prove_committed_shard_dag_bound, verify_committed_shard_dag_bound,
    Op, PublicStatement, Store, TensorKey, TrustedExecution, TrustedModelManifest,
};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn status_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn main() {
    let m = std::env::var("ZKIE_O1_M").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let n = std::env::var("ZKIE_O1_N").ok().and_then(|v| v.parse().ok()).unwrap_or(65536usize);
    let mut store = Store::new();
    let x = store.push(
        (0..m * n).map(|i| from_i64((((i as u64 * 37 + 11) % 101) as i64) - 50)).collect(),
    );
    let sel = store.push(vec![]);
    let thr = store.push(vec![]);
    let gate = store.push(vec![]);
    let d1 = store.push(vec![]);
    let d2 = store.push(vec![]);
    let out = store.push_idx(vec![]);
    let ops = vec![Op::Argmax { x, sel, thr, gate, d1, d2, out, m, n }];
    evaluate_ops(&mut store, &ops);
    let model = TrustedModelManifest::from_offline_commitment(&[], &store.owned_slices(), &ops)
        .with_index_shapes(&[(out, m)].into());
    let out_key = TensorKey::Index(out).encoded().unwrap();
    let out_vals: Vec<Goldilocks> =
        store.idx[out].iter().map(|&v| Goldilocks::from_u64(v as u64)).collect();
    let execution = TrustedExecution {
        inputs: [(x, store.get(x).to_vec())].into(),
        outputs: [
            (sel, store.get(sel).to_vec()),
            (thr, store.get(thr).to_vec()),
            (gate, store.get(gate).to_vec()),
            (d1, store.get(d1).to_vec()),
            (d2, store.get(d2).to_vec()),
            (out_key, out_vals),
        ].into(),
    };
    let whir = Whir::new_testing(5);
    let rss_before = status_kb("VmRSS:");
    let (a0, b0) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    let t0 = Instant::now();
    let proof = prove_committed_shard_dag_bound(
        &mut store, &ops, 1, &whir, &[], 0, 0,
        &PublicStatement::Generic, &model, &execution,
    )
    .expect("Argmax O1 proof");
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
    let (a1, b1) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    let bytes = bincode::serialize(&proof).expect("serialize proof");
    let openings = proof.selector_openings();
    let sel_groups = openings.len();
    let sel_points: usize = openings.iter().map(|s| s.points.len()).sum();
    let sel_bytes: usize = openings.iter()
        .map(|s| bincode::serialize(s).map(|b| b.len()).unwrap_or(0)).sum();
    let t1 = Instant::now();
    let ok = verify_committed_shard_dag_bound(
        &ops, 1, &whir, &proof, &PublicStatement::Generic, &model, &execution,
    );
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
    let rss_after = status_kb("VmRSS:");
    let hwm = status_kb("VmHWM:");
    println!(
        "O1-ARGMAX m={} n={} proof_bytes={} sel_groups={} sel_points={} sel_bytes={} prove_ms={:.1} verify_ms={:.3} verify={} allocs={} alloc_bytes={} rss_before_kb={} rss_after_kb={} hwm_kb={}",
        m, n, bytes.len(), sel_groups, sel_points, sel_bytes, prove_ms, verify_ms, ok,
        a1 - a0, b1 - b0, rss_before, rss_after, hwm,
    );
    assert!(ok, "proof must verify");
}
