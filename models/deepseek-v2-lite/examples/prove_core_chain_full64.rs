//! S7 full 64-expert core-chain real-data proofs (bound/bytes entry),
//! four-statement segmented shape: routing A, gate projection B, full expert
//! FFN D, router-gate + Gather C.
//!
//! The single-statement chain cannot certify its internal routing INDEX
//! intermediates: IdxBroadcast requires the base to be verifier-replayable
//! or certified public u32 data, and replacing that with random-point claims
//! would drop the pointwise u32 / b*base+j overflow contract. Instead the
//! chain is split at the routing arrays, which are PUBLIC data in the S7
//! statement:
//!
//!   A: scores -> TopKSelect -> Scatter -> PadIdx
//!      public outputs: thr, gate, d1, d2, token_id_pad, expert_of_pad
//!   B: token_id_pad/expert_of_pad (public index inputs) -> IdxBroadcast
//!      -> Lookup(real post-attn activation table) -> BroadcastScale(mask)
//!      -> GatherMatMul(full real 64-expert gate weights) -> Scale(shift 16)
//!      public output: g
//!   D: full real 64-expert FFN from g and the public routing arrays:
//!      IdxBroadcast -> Lookup(real activation) -> BroadcastScale(mask)
//!      -> GatherMatMul(up) -> Scale16 -> GeluIndex(2^24)
//!      -> Lookup(real silu 2^24) -> ScaleVec(shift 16)
//!      -> GatherMatMul(down) -> Scale16
//!      public output: d (g is B's proven output, caller-bound)
//!   C: scores -> TopKSelect -> Scatter -> PadIdx -> CombineIdx
//!      -> Lookup(table = op-written TopK gate) -> BroadcastScale(d, gate_pos)
//!      -> Scale16 -> Gather(ones) -> out
//!      public outputs: thr, gate, d1, d2, out
//!
//! IdxBroadcast in B/D uses the existing verifier replay path, which checks
//! the public index values are < 2^32 and uses checked u32 arithmetic for
//! b*base+j. No reduction is added and no constraint is bypassed.
//!
//! Real inputs: scripts/ds_layer_core_inputs.py exports the layer-L post-attn
//! activation and the Q16 router scores with the model's fixed-point recipe.
//!
//! BOUNDARY (NOT gist S7 acceptance): this is a four-statement segmented
//! real-data experiment. Each verifier ingests the full routing arrays as
//! public data, which the gist forbids for the final S7 construction, and the
//! cross-statement links (A output arrays = B/D/C input arrays, B's g = D's
//! g, D's d = C's d, A's gate = C's gate) are caller-trusted equalities, not
//! one proof. The B/D weight roots use the full real 64-expert 2^28 tensors
//! directly (no 8-expert slice identity assumption; the trusted manifest
//! still binds the offline-committed bytes). C exercises the router-gate
//! consistency positive path: the Gather weight enters through
//! BroadcastScale.scale from an op-written TopK-gate Lookup, which the staged
//! verifier accepts; the static-forged-gate-table counterexample remains
//! fail-closed (unit-level regression). No single-statement S7 proof was
//! achieved.

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

const H: usize = 2048;
const N: usize = 64;
const K: usize = 6;
const MOE_PAD: usize = 2048;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn peak_rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            return v.trim().trim_end_matches(" kB").parse().unwrap_or(0);
        }
    }
    0
}

fn to_g(v: &[u32]) -> Vec<Goldilocks> {
    v.iter().map(|&x| Goldilocks::from_u64(x as u64)).collect()
}

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
    let max_abs = scores_v.iter().map(|v| { let x = to_i64(*v); if x < 0 { -x } else { x } }).max().unwrap();
    let max_v = scores_v.iter().map(|v| to_i64(*v)).max().unwrap();
    println!("layer={layer} seq=1 k={K} real scores: max|.|={max_abs} max={max_v}");

    // ---------------- Phase A: selection + routing + padding ----------------
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
    let t0 = Instant::now();
    evaluate_ops(&mut sa, &ops_a);
    println!("A forward {:?}", t0.elapsed());
    let thr_v = sa.get(thr).to_vec();
    let gate_v = sa.get(gate).to_vec();
    let d1_v = sa.get(d1).to_vec();
    let d2_v = sa.get(d2).to_vec();
    let tok_v = sa.idx[tok_pad].clone();
    let exp_v = sa.idx[exp_pad].clone();
    println!("sel/route: thr={:?} experts={:?} token_pad={:?} expert_pad={:?}",
        thr_v.iter().map(|v| to_i64(*v)).collect::<Vec<_>>(), exp_v,
        tok_v, exp_v);
    let tk_key = TensorKey::Index(tok_pad).encoded().unwrap();
    let ek_key = TensorKey::Index(exp_pad).encoded().unwrap();
    let execution_a = TrustedExecution {
        inputs: [(scores_a, scores_v.clone())].into(),
        outputs: [
            (thr, thr_v.clone()), (gate, gate_v.clone()), (d1, d1_v.clone()),
            (d2, d2_v.clone()), (tk_key, to_g(&tok_v)), (ek_key, to_g(&exp_v)),
        ].into(),
    };
    let cache_a = format!("{cache}-a");
    let t1 = Instant::now();
    let cached_a = commit_model_weights(&sa, &ops_a, &cache_a);
    let t1_commit = t1.elapsed();
    let model_a = TrustedModelManifest::from_offline_commitment(&cached_a, &sa.owned_slices(), &ops_a)
        .with_index_shapes(&[(token_id, s), (expert_of, s)].into());
    let whir = Whir::new_testing(5);
    let proof_a = prove_committed_shard_dag_bound(&mut sa, &ops_a, 1, &whir, &cached_a, 0, 0,
        &PublicStatement::Generic, &model_a, &execution_a)
        .expect("phase A real routing proof");
    let t1_prove = t1.elapsed() - t1_commit;
    let bytes_a = bincode::serialize(&proof_a).expect("serialize A");
    println!("A commit {:?} prove {:?} bytes {}", t1_commit, t1_prove, bytes_a.len());
    assert!(verify_committed_shard_dag_bound_bytes(&bytes_a, &ops_a, 1, &whir,
        &PublicStatement::Generic, &model_a, &execution_a), "phase A must verify");
    // Real-data router range: the TopKSelect proof carries XRange/ThrRange.
    let mut bad_scores = scores_v.clone();
    bad_scores[0] = bad_scores[0] + Goldilocks::from_u64(1u64 << 22);
    let mut sb = Store::new();
    let sc = sb.push(bad_scores.clone());
    let sel_b = sb.push(vec![]); let thr_b = sb.push(vec![]); let gate_b = sb.push(vec![]);
    let d1_b = sb.push(vec![]); let d2_b = sb.push(vec![]);
    let ops_r = vec![Op::TopKSelect { x: sc, sel: sel_b, thr: thr_b, gate: gate_b, d1: d1_b, d2: d2_b, m, n: N, k: K }];
    evaluate_ops(&mut sb, &ops_r);
    let cached_r = commit_model_weights(&sb, &ops_r, &format!("{cache}-range"));
    let model_r = TrustedModelManifest::from_offline_commitment(&cached_r, &sb.owned_slices(), &ops_r);
    // Correct public outputs for the tampered forward, so the only reason the
    // prover can refuse is the out-of-contract router score (range aux).
    let exec_r = TrustedExecution {
        inputs: [(sc, bad_scores)].into(),
        outputs: [
            (thr_b, sb.get(thr_b).to_vec()), (gate_b, sb.get(gate_b).to_vec()),
            (d1_b, sb.get(d1_b).to_vec()), (d2_b, sb.get(d2_b).to_vec()),
        ].into(),
    };
    let _ = &exec_r;
    assert!(prove_committed_shard_dag_bound(&mut sb, &ops_r, 1, &whir, &cached_r, 0, 0,
        &PublicStatement::Generic, &model_r, &exec_r).is_none(),
        "out-of-range router score must fail closed at proof level");
    println!("A negative rejected: router score above 2^22 (proof-level range)");

    // ---------------- Phase B: replay routing + expert FFN gate ----------------
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
    let t2 = Instant::now();
    evaluate_ops(&mut sb, &ops_b);
    let g_v = sb.get(g).to_vec();
    println!("B forward {:?} g_len={}", t2.elapsed(), g_v.len());
    let g_ints: Vec<i64> = g_v.iter().map(|&v| to_i64(v)).collect();
    let g_min = *g_ints.iter().min().unwrap();
    let g_max = *g_ints.iter().max().unwrap();
    let w_min = g_min + (1i64 << 23);
    let w_max = g_max + (1i64 << 23);
    let hi_min = w_min - ((1i64 << 24) - 1);
    let hi_max = w_max - ((1i64 << 24) - 1);
    println!("B gate range [{g_min},{g_max}] w=[{w_min},{w_max}] high=[{hi_min},{hi_max}]");
    assert!(g_min >= -(1i64 << 24) && g_max < (1i64 << 24), "real g outside 25-bit x domain");
    assert!(w_min >= -(1i64 << 24) && w_max < (1i64 << 24), "real w outside 25-bit ReLU domain");
    assert!(hi_min >= -(1i64 << 24) && hi_max < (1i64 << 24), "real high outside 25-bit ReLU domain");
    let kb_tok = TensorKey::Index(tok_b).encoded().unwrap();
    let kb_exp = TensorKey::Index(exp_b).encoded().unwrap();
    let execution_b = TrustedExecution {
        inputs: [
            (kb_tok, to_g(&tok_v)), (kb_exp, to_g(&exp_v)), (mask, mask_v.clone()),
        ].into(),
        outputs: [(g, g_v.clone())].into(),
    };
    let cache_b = format!("{cache}-b");
    let t3 = Instant::now();
    let cached_b = commit_model_weights(&sb, &ops_b, &cache_b);
    let model_b = TrustedModelManifest::from_offline_commitment(&cached_b, &sb.owned_slices(), &ops_b)
        .with_index_shapes(&[(tok_b, s_pad), (exp_b, s_pad), (idx_seg, s_pad * H)].into());
    let proof_b = prove_committed_shard_dag_bound(&mut sb, &ops_b, 1, &whir, &cached_b, 0, 0,
        &PublicStatement::Generic, &model_b, &execution_b)
        .expect("phase B real FFN proof");
    let bytes_b = bincode::serialize(&proof_b).expect("serialize B");
    let t4 = Instant::now();
    assert!(verify_committed_shard_dag_bound_bytes(&bytes_b, &ops_b, 1, &whir,
        &PublicStatement::Generic, &model_b, &execution_b), "phase B must verify");
    println!("B commit+prove {:?} verify {:?} bytes {} peak_rss={}kB",
        t3.elapsed(), t4.elapsed(), bytes_b.len(), peak_rss_kb());

    // ---------------- real-data malicious regressions (bytes entry) ----------------
    let mut bad_a = execution_a.clone();
    bad_a.outputs.get_mut(&gate).unwrap()[0] = bad_a.outputs[&gate][0] + Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_a, &ops_a, 1, &whir,
        &PublicStatement::Generic, &model_a, &bad_a), "A tampered gate must be rejected");
    println!("negative rejected: A tampered gate");
    let mut bad_a2 = execution_a.clone();
    bad_a2.inputs.get_mut(&scores_a).unwrap()[0] = bad_a2.inputs[&scores_a][0] + Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_a, &ops_a, 1, &whir,
        &PublicStatement::Generic, &model_a, &bad_a2), "A tampered score input must be rejected");
    println!("negative rejected: A tampered router score input");
    let mut bad_b_inputs = execution_b.clone();
    let mut bad_exp = to_g(&exp_v);
    bad_exp[0] = Goldilocks::from_u64(exp_v[1] as u64);
    bad_b_inputs.inputs.insert(kb_exp, bad_exp);
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_b, &ops_b, 1, &whir,
        &PublicStatement::Generic, &model_b, &bad_b_inputs),
        "B tampered (duplicate) public routing array must be rejected");
    println!("negative rejected: B tampered public routing array");
    let mut bad_b_out = execution_b.clone();
    bad_b_out.outputs.get_mut(&g).unwrap()[0] = bad_b_out.outputs[&g][0] + Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_b, &ops_b, 1, &whir,
        &PublicStatement::Generic, &model_b, &bad_b_out), "B tampered g must be rejected");
    println!("negative rejected: B tampered g output");
    // Wrong expert weight root: substitute an unrelated committed root.
    let mut small = Store::new();
    let xa = small.push(vec![Goldilocks::ONE; 32]);
    let xw = small.push((0..32).map(|i| Goldilocks::from_u64(i as u64 + 1)).collect());
    let xc = small.push(vec![]);
    let xops = vec![Op::MatMul { a: xa, b: xw, c: xc, m: 4, k: 8, n: 4 }];
    let xcached = commit_model_weights(&small, &xops, &format!("{cache}-small"));
    let mut bad_cached = cached_b.clone();
    bad_cached[0].commitment = xcached[0].commitment.clone();
    let bad_model = TrustedModelManifest::from_offline_commitment(&bad_cached, &sb.owned_slices(), &ops_b)
        .with_index_shapes(&[(tok_b, s_pad), (exp_b, s_pad), (idx_seg, s_pad * H)].into());
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_b, &ops_b, 1, &whir,
        &PublicStatement::Generic, &bad_model, &execution_b), "wrong expert root must be rejected");
    println!("negative rejected: wrong expert weight root");
    // Wrong activation table root (cheap committed table).
    let mut act2 = act_v.clone();
    act2[0] = act2[0] + Goldilocks::ONE;
    let mut s2 = Store::new();
    let tok2 = s2.push_idx(tok_v.clone());
    let exp2 = s2.push_idx(exp_v.clone());
    let mask2 = s2.push(mask_v.clone());
    let act2_t = s2.push(act2);
    let idx2 = s2.push_idx(vec![]); let seg2 = s2.push(vec![]); let segz2 = s2.push(vec![]);
    let graw2 = s2.push(vec![]); let g2 = s2.push(vec![]);
    let wmap2 = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_experts_gate_i32.bin")).unwrap());
    let w2 = s2.push_mmap(wmap2, 0, N * H * MOE_PAD);
    let ops2 = vec![
        Op::IdxBroadcast { base: tok2, out: idx2, b: H },
        Op::Lookup { idx: idx2, out: seg2, table: act2_t },
        Op::BroadcastScale { x: seg2, scale: mask2, out: segz2, h: H },
        Op::GatherMatMul { a: segz2, expert_of: exp2, w: w2, c: graw2, s: s_pad, k: H, n: MOE_PAD, n_experts: N },
        Op::Scale { x: graw2, out: g2, factor: 1, shift: 16 },
    ];
    // Fill the store shapes (graw2/g2/segz2 lengths) exactly as the honest B
    // path does before committing/inspecting the manifest; without this the
    // Scale input shape is missing and manifest construction panics instead of
    // reaching the verifier. The tampered activation table root remains the
    // only difference from cached_b, so the verifier must reject.
    evaluate_ops(&mut s2, &ops2);
    let cached2 = commit_model_weights(&s2, &ops2, &format!("{cache}-table"));
    let model2 = TrustedModelManifest::from_offline_commitment(&cached2, &s2.owned_slices(), &ops2)
        .with_index_shapes(&[(tok2, s_pad), (exp2, s_pad), (idx2, s_pad * H)].into());
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_b, &ops_b, 1, &whir,
        &PublicStatement::Generic, &model2, &execution_b), "wrong activation table root must be rejected");
    println!("negative rejected: wrong activation table root");
    // ---------------- Phase D: full 64-expert FFN (real up/down) ----------------
    // A/B/D/C segmented real chain. D proves the full expert FFN from the
    // public routing/activation segment and B's proven g, using the real
    // 64-expert up/down weights (2^28 slots each) and the real silu 2^24
    // table. The output d is the caller d consumed by C.
    let mut sd = Store::new();
    let tok_d = sd.push_idx(tok_v.clone());
    let exp_d = sd.push_idx(exp_v.clone());
    let mask_d = sd.push(mask_v.clone());
    let act_d = sd.push(act_v.clone());
    let idx_d = sd.push_idx(vec![]);
    let seg_d = sd.push(vec![]);
    let segz_d = sd.push(vec![]);
    let g_d = sd.push(g_v.clone());
    let wmap_u = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_experts_up_i32.bin"))
        .unwrap_or_else(|e| panic!("open expert up weights: {e}")));
    let wu_d = sd.push_mmap(wmap_u, 0, N * H * MOE_PAD);
    let uraw_d = sd.push(vec![]);
    let u_d = sd.push(vec![]);
    let silu_v = load_i32(&format!("{dir}/silu_table_i32.bin"));
    assert_eq!(silu_v.len(), 1usize << 24, "real silu table length");
    let silu_t = sd.push(silu_v);
    let sidx_d = sd.push_idx(vec![]);
    let silu_d = sd.push(vec![]);
    let wmap_d = Arc::new(WeightMmap::open(&format!("{dir}/L{layer}_experts_down_i32.bin"))
        .unwrap_or_else(|e| panic!("open expert down weights: {e}")));
    let wd_d = sd.push_mmap(wmap_d, 0, N * H * MOE_PAD);
    let ffn_d = sd.push(vec![]);
    let draw_d = sd.push(vec![]);
    let out_d = sd.push(vec![]);
    let ops_d = vec![
        Op::IdxBroadcast { base: tok_d, out: idx_d, b: H },
        Op::Lookup { idx: idx_d, out: seg_d, table: act_d },
        Op::BroadcastScale { x: seg_d, scale: mask_d, out: segz_d, h: H },
        Op::GatherMatMul { a: segz_d, expert_of: exp_d, w: wu_d, c: uraw_d, s: s_pad, k: H, n: MOE_PAD, n_experts: N },
        Op::Scale { x: uraw_d, out: u_d, factor: 1, shift: 16 },
        Op::GeluIndex { x: g_d, out: sidx_d, offset: 1 << 23, table_len: 1 << 24 },
        Op::Lookup { idx: sidx_d, out: silu_d, table: silu_t },
        Op::ScaleVec { x: silu_d, scale: u_d, out: ffn_d, shift: 16 },
        Op::GatherMatMul { a: ffn_d, expert_of: exp_d, w: wd_d, c: draw_d, s: s_pad, k: MOE_PAD, n: H, n_experts: N },
        Op::Scale { x: draw_d, out: out_d, factor: 1, shift: 16 },
    ];
    evaluate_ops(&mut sd, &ops_d);
    let d_v = sd.get(out_d).to_vec();
    assert_eq!(sd.get(g_d), g_v.as_slice(), "D g must equal B's proven g");
    let silu_max = sd.get(silu_d).iter().map(|&v| to_i64(v).abs()).max().unwrap_or(0);
    let u_max = sd.get(u_d).iter().map(|&v| to_i64(v).abs()).max().unwrap_or(0);
    println!("D ScaleVec domain |silu|max={} |u|max={} (contract < 2^24)", silu_max, u_max);
    assert!(silu_max < (1i64 << 24) && u_max < (1i64 << 24));
    // Identity table for the Scale/ScaleVec shift>0 push arms.
    let _ = sd.push((0..(1usize << 16)).map(|i| Goldilocks::from_u64(i as u64)).collect());
    let cache_d = format!("{cache}-d");
    let t_d = Instant::now();
    let cached_d = commit_model_weights(&sd, &ops_d, &cache_d);
    let model_d = TrustedModelManifest::from_offline_commitment(&cached_d, &sd.owned_slices(), &ops_d)
        .with_index_shapes(&[
            (tok_d, s_pad), (exp_d, s_pad), (idx_d, s_pad * H), (sidx_d, s_pad * MOE_PAD),
        ].into());
    let exec_d = TrustedExecution {
        inputs: [
            (TensorKey::Index(tok_d).encoded().unwrap(), to_g(&tok_v)),
            (TensorKey::Index(exp_d).encoded().unwrap(), to_g(&exp_v)),
            (mask_d, mask_v.clone()), (g_d, g_v.clone()),
        ].into(),
        outputs: [(out_d, d_v.clone())].into(),
    };
    let proof_d = prove_committed_shard_dag_bound(&mut sd, &ops_d, 1, &whir, &cached_d, 0, 0,
        &PublicStatement::Generic, &model_d, &exec_d)
        .expect("phase D full 64-expert FFN proof");
    let bytes_d = bincode::serialize(&proof_d).expect("serialize D");
    let t_dv = Instant::now();
    assert!(verify_committed_shard_dag_bound_bytes(&bytes_d, &ops_d, 1, &whir,
        &PublicStatement::Generic, &model_d, &exec_d), "phase D must verify");
    println!("D commit+prove {:?} verify {:?} bytes {} peak_rss={}kB",
        t_d.elapsed(), t_dv.elapsed(), bytes_d.len(), peak_rss_kb());

    // D negatives: output / g / mask / routing swaps and drops.
    let mut bad_d_out = exec_d.clone();
    bad_d_out.outputs.get_mut(&out_d).unwrap()[0] += Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_d, &ops_d, 1, &whir,
        &PublicStatement::Generic, &model_d, &bad_d_out), "D tampered output must be rejected");
    let mut bad_d_g = exec_d.clone();
    bad_d_g.inputs.get_mut(&g_d).unwrap()[0] += Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_d, &ops_d, 1, &whir,
        &PublicStatement::Generic, &model_d, &bad_d_g), "D tampered g must be rejected");
    let mut bad_d_mask = exec_d.clone();
    bad_d_mask.inputs.get_mut(&mask_d).unwrap()[0] += Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_d, &ops_d, 1, &whir,
        &PublicStatement::Generic, &model_d, &bad_d_mask), "D tampered mask must be rejected");
    let mut bad_d_exp = exec_d.clone();
    let mut swapped = to_g(&exp_v);
    swapped.swap(0, 1);
    bad_d_exp.inputs.insert(TensorKey::Index(exp_d).encoded().unwrap(), swapped);
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_d, &ops_d, 1, &whir,
        &PublicStatement::Generic, &model_d, &bad_d_exp), "D swapped experts must be rejected");
    let mut bad_d_drop = exec_d.clone();
    let mut dropped = to_g(&exp_v);
    dropped[0] = dropped[1];
    bad_d_drop.inputs.insert(TensorKey::Index(exp_d).encoded().unwrap(), dropped);
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_d, &ops_d, 1, &whir,
        &PublicStatement::Generic, &model_d, &bad_d_drop), "D dropped/duplicate route must be rejected");
    println!("D negatives rejected: out/g/mask tamper, swapped experts, dropped route");
    // Wrong up/down weight root.
    let mut bad_cached_d = cached_d.clone();
    bad_cached_d[0].commitment = xcached[0].commitment.clone();
    let bad_model_d = TrustedModelManifest::from_offline_commitment(&bad_cached_d, &sd.owned_slices(), &ops_d)
        .with_index_shapes(&[
            (tok_d, s_pad), (exp_d, s_pad), (idx_d, s_pad * H), (sidx_d, s_pad * MOE_PAD),
        ].into());
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_d, &ops_d, 1, &whir,
        &PublicStatement::Generic, &bad_model_d, &exec_d), "wrong up/down expert root must be rejected");
    println!("negative rejected: wrong up/down expert root");
    // Wrong real silu table root.
    sd.get_mut(silu_t)[0] += Goldilocks::ONE;
    let bad_silu_model = TrustedModelManifest::from_offline_commitment(&cached_d, &sd.owned_slices(), &ops_d)
        .with_index_shapes(&[
            (tok_d, s_pad), (exp_d, s_pad), (idx_d, s_pad * H), (sidx_d, s_pad * MOE_PAD),
        ].into());
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_d, &ops_d, 1, &whir,
        &PublicStatement::Generic, &bad_silu_model, &exec_d), "wrong real silu root must be rejected");
    println!("negative rejected: wrong real silu table root");

    // ---------------- Phase C: router-gate consistency chain + Gather ----------------
    let mut scc = Store::new();
    let scores_c = scc.push(scores_v.clone());
    let sel_c = scc.push(vec![]);
    let thr_c = scc.push(vec![]);
    let gate_c = scc.push(vec![]);
    let d1_c = scc.push(vec![]);
    let d2_c = scc.push(vec![]);
    let tok_c = scc.push_idx(vec![]);
    let exp_c = scc.push_idx(vec![]);
    let tokp_c = scc.push_idx(vec![]);
    let expp_c = scc.push_idx(vec![]);
    let idxg_c = scc.push_idx(vec![]);
    let gpos_c = scc.push(vec![]);
    let d_c = scc.push(d_v.clone());
    let wraw_c = scc.push(vec![]);
    let w_c = scc.push(vec![]);
    let ones_c = scc.push(vec![Goldilocks::ONE; s_pad]);
    let out_c = scc.push(vec![]);
    let ops_c = vec![
        Op::TopKSelect { x: scores_c, sel: sel_c, thr: thr_c, gate: gate_c, d1: d1_c, d2: d2_c, m, n: N, k: K },
        Op::Scatter { sel: sel_c, token_id: tok_c, expert_of: exp_c, m, n: N },
        Op::PadIdx { base: tok_c, out: tokp_c, len: s_pad, base_len: s },
        Op::PadIdx { base: exp_c, out: expp_c, len: s_pad, base_len: s },
        Op::CombineIdx { a: tokp_c, b: expp_c, out: idxg_c, scale: N },
        Op::Lookup { idx: idxg_c, out: gpos_c, table: gate_c },
        Op::BroadcastScale { x: d_c, scale: gpos_c, out: wraw_c, h: H },
        Op::Scale { x: wraw_c, out: w_c, factor: 1, shift: 16 },
        Op::Gather { weighted: w_c, token_id: tokp_c, gate: ones_c, out: out_c, m, h: H },
    ];
    let t_c = Instant::now();
    evaluate_ops(&mut scc, &ops_c);
    let out_v = scc.get(out_c).to_vec();
    assert_eq!(scc.get(thr_c), thr_v.as_slice(), "C threshold matches A");
    assert_eq!(scc.get(gate_c), gate_v.as_slice(), "C gate matches A");
    assert_eq!(scc.get(d1_c), d1_v.as_slice(), "C d1 matches A");
    assert_eq!(scc.get(d2_c), d2_v.as_slice(), "C d2 matches A");
    let exec_c = TrustedExecution {
        inputs: [
            (scores_c, scores_v.clone()), (d_c, d_v.clone()),
            (ones_c, vec![Goldilocks::ONE; s_pad]),
        ].into(),
        outputs: [
            (thr_c, thr_v.clone()), (gate_c, gate_v.clone()),
            (d1_c, d1_v.clone()), (d2_c, d2_v.clone()), (out_c, out_v.clone()),
        ].into(),
    };
    let cache_c = format!("{cache}-c");
    let cached_c = commit_model_weights(&scc, &ops_c, &cache_c);
    let model_c = TrustedModelManifest::from_offline_commitment(&cached_c, &scc.owned_slices(), &ops_c)
        .with_index_shapes(&[
            (tok_c, s), (exp_c, s), (tokp_c, s_pad), (expp_c, s_pad), (idxg_c, s_pad),
        ].into());
    let t_cp = Instant::now();
    let proof_c = prove_committed_shard_dag_bound(&mut scc, &ops_c, 1, &whir, &cached_c, 0, 0,
        &PublicStatement::Generic, &model_c, &exec_c)
        .expect("phase C router-gate + Gather proof");
    let bytes_c = bincode::serialize(&proof_c).expect("serialize C");
    let t_cv = Instant::now();
    assert!(verify_committed_shard_dag_bound_bytes(&bytes_c, &ops_c, 1, &whir,
        &PublicStatement::Generic, &model_c, &exec_c), "phase C must verify");
    println!("C forward {:?} commit+prove {:?} verify {:?} bytes {} out_len={}",
        t_c.elapsed(), t_cp.elapsed(), t_cv.elapsed(), bytes_c.len(), out_v.len());
    let mut bad_c_d = exec_c.clone();
    bad_c_d.inputs.get_mut(&d_c).unwrap()[0] += Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_c, &ops_c, 1, &whir,
        &PublicStatement::Generic, &model_c, &bad_c_d), "C tampered d must be rejected");
    let mut bad_c_score = exec_c.clone();
    bad_c_score.inputs.get_mut(&scores_c).unwrap()[0] += Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_c, &ops_c, 1, &whir,
        &PublicStatement::Generic, &model_c, &bad_c_score), "C tampered scores must be rejected");
    let mut bad_c_out = exec_c.clone();
    bad_c_out.outputs.get_mut(&out_c).unwrap()[0] += Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_c, &ops_c, 1, &whir,
        &PublicStatement::Generic, &model_c, &bad_c_out), "C tampered output must be rejected");
    let mut bad_c_gate = exec_c.clone();
    bad_c_gate.outputs.get_mut(&gate_c).unwrap()[0] += Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_c, &ops_c, 1, &whir,
        &PublicStatement::Generic, &model_c, &bad_c_gate), "C tampered gate must be rejected");
    println!("C negatives rejected: d/score/out/gate tamper");
    if let Ok(prefix) = std::env::var("ZKIE_S7_DUMP_PREFIX") {
        std::fs::write(format!("{prefix}-a.bin"), &bytes_a).expect("dump A");
        std::fs::write(format!("{prefix}-b.bin"), &bytes_b).expect("dump B");
        std::fs::write(format!("{prefix}-d.bin"), &bytes_d).expect("dump D");
        std::fs::write(format!("{prefix}-c.bin"), &bytes_c).expect("dump C");
    }
    println!("S7-CHAIN-OK layer={layer} A_bytes={} B_bytes={} D_bytes={} C_bytes={} peak_rss={}kB",
        bytes_a.len(), bytes_b.len(), bytes_d.len(), bytes_c.len(), peak_rss_kb());
}
