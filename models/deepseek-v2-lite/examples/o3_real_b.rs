//! O3 real B-proof allocation/CPU/RSS measurement.
//!
//! Runs only phase B of the real 64-expert core chain (routing plan, gate
//! GatherMatMul, gate projection) and reports process allocation counts, CPU
//! time and RSS around the proof call. The same source builds against the
//! pre-O3/prior-O3 tree for a before/after comparison.
//!
//! Env: DS_WEIGHTS, DS_LAYER, ZKIE_CACHE (same as the core-chain examples).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::{from_i32, to_i64};
use zkie_core::common::weights_io::WeightMmap;
use zkie_core::pcs::whir::Whir;
use zkie_ops::compose::{
    commit_model_weights, evaluate_ops, prove_committed_shard_dag_bound,
    verify_committed_shard_dag_bound_bytes, Op, PublicStatement, Store, TensorKey,
    TrustedExecution, TrustedModelManifest,
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

fn snapshot() -> (usize, usize) {
    (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed))
}

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

fn cpu_ms() -> f64 {
    let tck = std::process::Command::new("getconf")
        .arg("CLK_TCK")
        .output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(100.0);
    let Ok(s) = std::fs::read_to_string("/proc/self/stat") else { return 0.0; };
    let t = s.replace(')', " ");
    let f: Vec<&str> = t.split_whitespace().collect();
    let utime: f64 = f.get(13).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let stime: f64 = f.get(14).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    (utime + stime) * 1000.0 / tck
}

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn to_g(v: &[u32]) -> Vec<Goldilocks> {
    v.iter().map(|&x| Goldilocks::from_u64(x as u64)).collect()
}

const H: usize = 2048;
const N: usize = 64;
const K: usize = 6;
const MOE_PAD: usize = 2048;

fn main() {
    let dir = std::env::var("DS_WEIGHTS")
        .unwrap_or_else(|_| "/data/zkIE/models/deepseek-v2-lite/weights".to_string());
    let layer: usize = std::env::var("DS_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let cache = std::env::var("ZKIE_CACHE")
        .unwrap_or_else(|_| "/dev/shm/zkie-core-chain-cache".to_string());
    let m = 1usize;
    let s = K * m;
    let s_pad = s.next_power_of_two();
    let scores_v = load_i32(&format!("{dir}/L{layer}_core_scores_i32.bin"));
    let act_v = load_i32(&format!("{dir}/L{layer}_core_act_i32.bin"));
    assert_eq!(scores_v.len(), m * N, "real router scores length");
    assert_eq!(act_v.len(), m * H, "real activation length");

    // Phase A forward only: derive the real routing arrays used by B.
    let mut sa = Store::new();
    let scores_a = sa.push(scores_v.clone());
    let sel = sa.push(vec![]);
    let thr = sa.push(vec![]);
    let gate = sa.push(vec![]);
    let d1 = sa.push(vec![]);
    let d2 = sa.push(vec![]);
    let token_id = sa.push_idx(vec![]);
    let expert_of = sa.push_idx(vec![]);
    let tok_pad = sa.push_idx(vec![]);
    let exp_pad = sa.push_idx(vec![]);
    let ops_a = vec![
        Op::TopKSelect { x: scores_a, sel, thr, gate, d1, d2, m, n: N, k: K },
        Op::Scatter { sel, token_id, expert_of, m, n: N },
        Op::PadIdx { base: token_id, out: tok_pad, len: s_pad, base_len: s },
        Op::PadIdx { base: expert_of, out: exp_pad, len: s_pad, base_len: s },
    ];
    evaluate_ops(&mut sa, &ops_a);
    let tok_v = sa.idx[tok_pad].clone();
    let exp_v = sa.idx[exp_pad].clone();

    // Phase B: real 64-expert gate FFN.
    let mut sb = Store::new();
    let tok_b = sb.push_idx(tok_v.clone());
    let exp_b = sb.push_idx(exp_v.clone());
    let mask_v: Vec<Goldilocks> = (0..s_pad)
        .map(|i| if i < s { Goldilocks::ONE } else { Goldilocks::ZERO })
        .collect();
    let mask = sb.push(mask_v.clone());
    let act = sb.push(act_v.clone());
    let idx_seg = sb.push_idx(vec![]);
    let seg = sb.push(vec![]);
    let seg_z = sb.push(vec![]);
    let g_raw = sb.push(vec![]);
    let g = sb.push(vec![]);
    let wmap = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_experts_gate_i32.bin"))
        .unwrap_or_else(|e| panic!("open expert gate weights: {e}")));
    let w = sb.push_mmap(wmap, 0, N * H * MOE_PAD);
    let ops_b = vec![
        Op::IdxBroadcast { base: tok_b, out: idx_seg, b: H },
        Op::Lookup { idx: idx_seg, out: seg, table: act },
        Op::BroadcastScale { x: seg, scale: mask, out: seg_z, h: H },
        Op::GatherMatMul { a: seg_z, expert_of: exp_b, w, c: g_raw, s: s_pad, k: H, n: MOE_PAD, n_experts: N },
        Op::Scale { x: g_raw, out: g, factor: 1, shift: 16 },
    ];
    evaluate_ops(&mut sb, &ops_b);
    let g_v = sb.get(g).to_vec();
    let kb_tok = TensorKey::Index(tok_b).encoded().unwrap();
    let kb_exp = TensorKey::Index(exp_b).encoded().unwrap();
    let execution_b = TrustedExecution {
        inputs: [(kb_tok, to_g(&tok_v)), (kb_exp, to_g(&exp_v)), (mask, mask_v.clone())].into(),
        outputs: [(g, g_v.clone())].into(),
    };
    let cache_b = format!("{cache}-b");
    let t_commit = Instant::now();
    let cached_b = commit_model_weights(&sb, &ops_b, &cache_b);
    let model_b = TrustedModelManifest::from_offline_commitment(&cached_b, &sb.owned_slices(), &ops_b)
        .with_index_shapes(&[(tok_b, s_pad), (exp_b, s_pad), (idx_seg, s_pad * H)].into());
    let commit_ms = t_commit.elapsed().as_secs_f64() * 1e3;

    let whir = Whir::new_testing(5);
    let rss_before = status_kb("VmRSS:");
    let cpu0 = cpu_ms();
    let (a0, b0) = snapshot();
    let t_prove = Instant::now();
    let proof = prove_committed_shard_dag_bound(&mut sb, &ops_b, 1, &whir, &cached_b, 0, 0,
        &PublicStatement::Generic, &model_b, &execution_b)
        .expect("phase B real FFN proof");
    let prove_ms = t_prove.elapsed().as_secs_f64() * 1e3;
    let (a1, b1) = snapshot();
    let cpu = cpu_ms() - cpu0;
    let rss_after = status_kb("VmRSS:");
    let hwm = status_kb("VmHWM:");

    let bytes = bincode::serialize(&proof).expect("serialize B");
    let t_verify = Instant::now();
    let verify = verify_committed_shard_dag_bound_bytes(
        &bytes, &ops_b, 1, &whir, &PublicStatement::Generic, &model_b, &execution_b);
    let verify_ms = t_verify.elapsed().as_secs_f64() * 1e3;
    println!(
        "O3-real-B threads={} commit_ms={:.1} prove_ms={:.1} cpu_ms={:.1} allocs={} alloc_bytes={} rss_before_kb={} rss_after_kb={} hwm_kb={} proof_bytes={} verify_ms={:.3} verify={}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string()),
        commit_ms, prove_ms, cpu, a1 - a0, b1 - b0,
        rss_before, rss_after, hwm, bytes.len(), verify_ms, verify);
    assert!(verify, "real B proof must verify");
}
