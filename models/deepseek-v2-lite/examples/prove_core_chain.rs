//! SEGMENTED real-data proofs (bound/bytes entry), three-statement shape.
//! This is NOT the complete S7 core chain and NOT S7 acceptance.
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
//!      -> GatherMatMul(real expert gate weights) -> Scale(shift 16)
//!      public outputs: g
//!   C: scores + caller-supplied real expert-FFN down projection d ->
//!      TopKSelect -> Scatter -> PadIdx -> CombineIdx
//!      -> Lookup(table = the DYNAMIC TopK gate) -> BroadcastScale(d, gate)
//!      -> Scale(shift 16) -> Gather -> out
//!      public outputs: thr, gate, d1, d2, out
//!
//! IdxBroadcast in B uses the existing verifier replay path, which checks
//! the public index values are < 2^32 and uses checked u32 arithmetic for
//! b*base+j. No reduction is added and no constraint is bypassed.
//!
//! C proves the router-gate consistency chain (the Lookup table is the same
//! tensor as the TopKSelect gate, authenticated by the returned table claim)
//! and the Gather scatter-add over the real router scores and real d.
//! LIMITATION (GeluIndex, fail-closed): the expert FFN bound proof (gate/up/
//! silu/down through d) is blocked because the real GeluIndex silu table has
//! 2^24 entries while `prove_reverse_gelu_index` accepts only
//! table_len <= RELU_BOUND = 2^22. d is therefore computed by the caller with
//! the model's exact forward ops and supplied as a caller-trusted public
//! input; C does not prove d's FFN provenance. The limit is a separate
//! reduction item and stays fail-closed.
//! TRUSTED-OFFLINE NOTE: the composition verifier authenticates the 8-expert
//! slice TENSOR commitment and the public slice index mapping, but it does
//! NOT prove that this slice equals the corresponding rows of the full
//! 64-expert model. That identity must be established by a trusted offline
//! manifest / external modeling (here: the explicit slice_experts list
//! applied to the real L*_experts_gate_i32.bin file).
//!
//! BOUNDARY (not gist S7 acceptance): the verifiers of B and C ingest the
//! full routing arrays as public data, which the gist forbids for the final
//! S7 construction. The A/B/C links are caller-trusted external statement
//! values (the SAME arrays/outputs populate A's outputs, B's inputs, and C's
//! inputs, bound item-by-item below), and C's d is caller-computed; it is a
//! segmented real-data experiment, not a single-statement S7 proof.
//!
//! Real inputs: scripts/ds_layer_core_inputs.py exports the layer-L post-attn
//! activation and the Q16 router scores with the model's fixed-point recipe.

use std::sync::Arc;
use std::time::Instant;

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::{from_i32, to_i64};
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

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Real expert-gate weight slice: gather the requested real expert blocks
/// from the 64-expert file. The full 2^28 matrix opening OOMs on the
/// 495GB machine (reported separately); the slice keeps the SAME real
/// weights for the routed experts.
fn load_expert_slice_file(path: &str, experts: &[usize]) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let block = H * MOE_PAD * 4;
    let mut out = Vec::with_capacity(experts.len() * H * MOE_PAD);
    for &e in experts {
        let off = e * block;
        out.extend(bytes[off..off + block].chunks_exact(4)
            .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]]))));
    }
    out
}

fn load_expert_slice(dir: &str, layer: usize, experts: &[usize]) -> Vec<Goldilocks> {
    load_expert_slice_file(&format!("{dir}/L{layer}_experts_gate_i32.bin"), experts)
}

fn ops_a_input(ops: &[Op]) -> Option<usize> {
    ops.iter().find_map(|op| if let Op::TopKSelect { x, .. } = op { Some(*x) } else { None })
}
fn ops_a_output_field(ops: &[Op], i: usize) -> Option<usize> {
    ops.iter().find_map(|op| if let Op::TopKSelect { thr, gate, d1, d2, .. } = op {
        Some([*thr, *gate, *d1, *d2][i])
    } else { None })
}
fn ops_b_mask_input(ops: &[Op]) -> Option<usize> {
    ops.iter().find_map(|op| if let Op::BroadcastScale { scale, .. } = op { Some(*scale) } else { None })
}
fn ops_b_output(ops: &[Op]) -> Option<usize> {
    ops.iter().rev().find_map(|op| if let Op::Scale { out, .. } = op { Some(*out) } else { None })
}

/// Caller-trusted external statement for the two segments. The verifier
/// below derives BOTH executions from this value alone: A's public outputs
/// (thr/gate/d1/d2 + original routing arrays) and B's public inputs (sliced
/// routing + mask) and outputs (g). No prover-local value is consulted.
#[derive(Clone)]
struct CoreStatement {
    scores: Vec<Goldilocks>,
    thr: Vec<Goldilocks>,
    gate: Vec<Goldilocks>,
    d1: Vec<Goldilocks>,
    d2: Vec<Goldilocks>,
    tok_routing: Vec<u32>,
    exp_routing: Vec<u32>,
    mask: Vec<Goldilocks>,
    g: Vec<Goldilocks>,
    /// Caller-computed real expert-FFN down-projection output (C input).
    /// NOT proven by any segment: the FFN bound proof is blocked at the real
    /// GeluIndex table_len = 2^24 > RELU_BOUND = 2^22 (fail-closed), reported
    /// to the TL. C consumes it as a caller-trusted public input.
    d: Vec<Goldilocks>,
    /// Router-gate chain + Gather output over `d` (C output).
    out: Vec<Goldilocks>,
    /// Real expert ids for the 8 slice rows (6 routed + 2 real padding).
    slice_experts: Vec<usize>,
    /// Per padded position: slice row index for exp_routing[i].
    slice: Vec<u32>,
    s: usize,
}

/// Derive the slice table from the statement's routing and slice identity.
/// Returns None on unknown routed experts, duplicate slice identities, or a
/// padding position that is not routed to slice row 6.
fn derive_slice(exp_routing: &[u32], slice_experts: &[usize], s: usize) -> Option<Vec<u32>> {
    if slice_experts.len() != 8 || exp_routing.len() < s { return None; }
    let mut seen = std::collections::BTreeSet::new();
    for &e in slice_experts { if !seen.insert(e) { return None; } }
    let mut out = Vec::with_capacity(exp_routing.len());
    for (i, &e) in exp_routing.iter().enumerate() {
        if i < s {
            let pos = slice_experts.iter().position(|&x| x == e as usize)?;
            if pos >= s { return None; }
            out.push(pos as u32);
        } else {
            if e != 0 { return None; }
            out.push(6);
        }
    }
    Some(out)
}

fn reverse_keys(ops: &[Op]) -> Option<(usize, usize)> {
    let mut tok = None;
    let mut exp = None;
    for op in ops {
        match op {
            Op::PadIdx { out, .. } => {
                let k = TensorKey::Index(*out).encoded()?;
                if tok.is_none() { tok = Some(k); } else { exp = Some(k); }
            }
            Op::IdxBroadcast { base, .. } => tok = Some(TensorKey::Index(*base).encoded()?),
            Op::GatherMatMul { expert_of, .. } => exp = Some(TensorKey::Index(*expert_of).encoded()?),
            _ => {}
        }
    }
    Some((tok?, exp?))
}

fn ops_c_scores_input(ops: &[Op]) -> Option<usize> {
    ops.iter().find_map(|op| if let Op::TopKSelect { x, .. } = op { Some(*x) } else { None })
}
fn ops_c_d_input(ops: &[Op]) -> Option<usize> {
    ops.iter().find_map(|op| if let Op::BroadcastScale { x, .. } = op { Some(*x) } else { None })
}
fn ops_c_out(ops: &[Op]) -> Option<usize> {
    ops.iter().find_map(|op| if let Op::Gather { out, .. } = op { Some(*out) } else { None })
}
fn ops_c_ones_input(ops: &[Op]) -> Option<usize> {
    ops.iter().find_map(|op| if let Op::Gather { gate, .. } = op { Some(*gate) } else { None })
}

/// Independent composition verifier: consumes ONLY the caller-trusted
/// statement, the three proof byte strings, the models and their op graphs.
/// It derives each segment's execution item-by-item, rejects an invalid
/// slice identity (unknown expert, duplicate identity, wrong padding row) and
/// segment mismatches. No cryptographic reduction is added. Segment C proves
/// the router-gate consistency chain + Gather over the caller-supplied real
/// down-projection `d`; `d` itself is NOT proven (the FFN bound proof is
/// blocked at GeluIndex table_len 2^24 > 2^22, fail-closed).
fn verify_segmented(
    stmt: &CoreStatement,
    ops_a: &[Op], model_a: &TrustedModelManifest, bytes_a: &[u8],
    ops_b: &[Op], model_b: &TrustedModelManifest, bytes_b: &[u8],
    ops_c: &[Op], model_c: &TrustedModelManifest, bytes_c: &[u8],
    whir: &Whir,
) -> bool {
    if stmt.tok_routing.len() != stmt.exp_routing.len() { return false; }
    let slice = match derive_slice(&stmt.exp_routing, &stmt.slice_experts, stmt.s) { Some(v) => v, None => return false };
    if slice != stmt.slice { return false; } // caller statement's slice identity must match its routing
    let (tk_a, ek_a) = match reverse_keys(ops_a) { Some(k) => k, None => return false };
    let (tk_b, ek_b) = match reverse_keys(ops_b) { Some(k) => k, None => return false };
    let a_in = match ops_a_input(ops_a) { Some(v) => v, None => return false };
    let a0 = match ops_a_output_field(ops_a, 0) { Some(v) => v, None => return false };
    let a1 = match ops_a_output_field(ops_a, 1) { Some(v) => v, None => return false };
    let a2 = match ops_a_output_field(ops_a, 2) { Some(v) => v, None => return false };
    let a3 = match ops_a_output_field(ops_a, 3) { Some(v) => v, None => return false };
    let b_mask = match ops_b_mask_input(ops_b) { Some(v) => v, None => return false };
    let b_out = match ops_b_output(ops_b) { Some(v) => v, None => return false };
    let exec_a = TrustedExecution {
        inputs: [(a_in, stmt.scores.clone())].into(),
        outputs: [
            (a0, stmt.thr.clone()), (a1, stmt.gate.clone()),
            (a2, stmt.d1.clone()), (a3, stmt.d2.clone()),
            (tk_a, to_g(&stmt.tok_routing)),
            (ek_a, to_g(&stmt.exp_routing)),
        ].into(),
    };
    let exec_b = TrustedExecution {
        inputs: [
            (tk_b, to_g(&stmt.tok_routing)),
            (ek_b, to_g(&slice)),
            (b_mask, stmt.mask.clone()),
        ].into(),
        outputs: [(b_out, stmt.g.clone())].into(),
    };
    let c_scores = match ops_c_scores_input(ops_c) { Some(v) => v, None => return false };
    let c_d = match ops_c_d_input(ops_c) { Some(v) => v, None => return false };
    let c_ones = match ops_c_ones_input(ops_c) { Some(v) => v, None => return false };
    let c_out = match ops_c_out(ops_c) { Some(v) => v, None => return false };
    let c_s_pad = stmt.s.next_power_of_two();
    let c0 = match ops_a_output_field(ops_c, 0) { Some(v) => v, None => return false };
    let c1 = match ops_a_output_field(ops_c, 1) { Some(v) => v, None => return false };
    let c2 = match ops_a_output_field(ops_c, 2) { Some(v) => v, None => return false };
    let c3 = match ops_a_output_field(ops_c, 3) { Some(v) => v, None => return false };
    let exec_c = TrustedExecution {
        inputs: [
            (c_scores, stmt.scores.clone()),
            (c_d, stmt.d.clone()),
            (c_ones, vec![Goldilocks::ONE; c_s_pad]),
        ].into(),
        outputs: [
            (c0, stmt.thr.clone()), (c1, stmt.gate.clone()),
            (c2, stmt.d1.clone()), (c3, stmt.d2.clone()),
            (c_out, stmt.out.clone()),
        ].into(),
    };
    verify_committed_shard_dag_bound_bytes(bytes_a, ops_a, 1, whir, &PublicStatement::Generic, model_a, &exec_a)
        && verify_committed_shard_dag_bound_bytes(bytes_b, ops_b, 1, whir, &PublicStatement::Generic, model_b, &exec_b)
        && verify_committed_shard_dag_bound_bytes(bytes_c, ops_c, 1, whir, &PublicStatement::Generic, model_c, &exec_c)
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
    let silu_v = load_i32(&format!("{dir}/silu_table_i32.bin"));
    assert_eq!(scores_v.len(), m * N, "real router scores length");
    assert_eq!(act_v.len(), m * H, "real activation length");
    assert_eq!(silu_v.len(), 1 << 24, "real silu table length");
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
    // Trusted external statement binding: A's public routing outputs and B's
    // public routing inputs are the SAME caller-owned arrays; the proofs only
    // report values, so the binding is checked here item by item.
    assert_eq!(execution_a.outputs.get(&tk_key), Some(&to_g(&tok_v)));
    assert_eq!(execution_a.outputs.get(&ek_key), Some(&to_g(&exp_v)));
    let stmt_tok = tok_v.clone();
    let stmt_exp = exp_v.clone();
    let cache_a = format!("{cache}-a");
    let t1 = Instant::now();
    let cached_a = commit_model_weights(&sa, &ops_a, &cache_a);
    let model_a = TrustedModelManifest::from_offline_commitment(&cached_a, &sa.owned_slices(), &ops_a)
        .with_index_shapes(&[(token_id, s), (expert_of, s)].into());
    let whir = Whir::new_testing(5);
    let proof_a = prove_committed_shard_dag_bound(&mut sa, &ops_a, 1, &whir, &cached_a, 0, 0,
        &PublicStatement::Generic, &model_a, &execution_a)
        .expect("phase A real routing proof");
    let bytes_a = bincode::serialize(&proof_a).expect("serialize A");
    println!("A commit {:?} prove {:?} bytes {}", t1.elapsed(), t1.elapsed(), bytes_a.len());
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
    let mut exec_r = TrustedExecution {
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
    // Real experts actually routed in this row, ascending (scatter order).
    let mut routed: Vec<usize> = exp_v.iter().take(s).map(|&e| e as usize).collect();
    routed.sort_unstable();
    routed.dedup();
    assert_eq!(routed.len(), s, "routed experts must be distinct");
    // Slice = the routed real experts + real experts 6,7 for masked padding.
    const PAD_EXPERTS: [usize; 2] = [6, 7];
    let slice_experts: Vec<usize> = routed.iter().copied().chain(PAD_EXPERTS).collect();
    let slice_of = |i: usize| -> u32 {
        if i < s { routed.iter().position(|&x| x == exp_v[i] as usize).unwrap() as u32 }
        else { 6 }
    };
    let exp_slice: Vec<u32> = (0..exp_v.len()).map(slice_of).collect();
    let tok_b = sb.push_idx(tok_v.clone());
    let exp_b = sb.push_idx(exp_slice.clone());
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
    // Real 8-expert slice: the routed experts plus real experts 6,7 for the
    // masked padding rows. The public statement's slice table binds the
    // original routing ids to these slice indices.
    let w = sb.push(load_expert_slice(&dir, layer, &slice_experts));
    let ops_b = vec![
        Op::IdxBroadcast { base: tok_b, out: idx_seg, b: H },
        Op::Lookup { idx: idx_seg, out: seg, table: act },
        Op::BroadcastScale { x: seg, scale: mask, out: seg_z, h: H },
        Op::GatherMatMul { a: seg_z, expert_of: exp_b, w, c: g_raw, s: s_pad, k: H, n: MOE_PAD, n_experts: 8 },
        Op::Scale { x: g_raw, out: g, factor: 1, shift: 16 },
    ];
    let t2 = Instant::now();
    evaluate_ops(&mut sb, &ops_b);
    let g_v = sb.get(g).to_vec();
    println!("B forward {:?} g_len={}", t2.elapsed(), g_v.len());
    let kb_tok = TensorKey::Index(tok_b).encoded().unwrap();
    let kb_exp = TensorKey::Index(exp_b).encoded().unwrap();
    let execution_b = TrustedExecution {
        inputs: [
            (kb_tok, to_g(&tok_v)), (kb_exp, to_g(&exp_slice)), (mask, mask_v.clone()),
        ].into(),
        outputs: [(g, g_v.clone())].into(),
    };

    // ---------------- Phase C input: real expert-FFN down projection ----------------
    // Caller-computed real d using the model's exact ops (ScaleVec shift=16),
    // forward only. The FFN BOUND proof is blocked at the real GeluIndex
    // table_len = 2^24 > RELU_BOUND = 2^22 (fail-closed); reported to the TL.
    // C consumes d as a caller-trusted public input.
    let mut sd = Store::new();
    let tok_d = sd.push_idx(tok_v.clone());
    let exp_d = sd.push_idx(exp_slice.clone());
    let mask_d = sd.push(mask_v.clone());
    let act_d = sd.push(act_v.clone());
    let idx_d = sd.push_idx(vec![]);
    let seg_d = sd.push(vec![]);
    let segz_d = sd.push(vec![]);
    let graw_d = sd.push(vec![]);
    let g_d = sd.push(vec![]);
    let wg_d = sd.push(load_expert_slice(&dir, layer, &slice_experts));
    let wu_d = sd.push(load_expert_slice_file(&format!("{dir}/L{layer}_experts_up_i32.bin"), &slice_experts));
    let uraw_d = sd.push(vec![]);
    let u_d = sd.push(vec![]);
    let silu_t_d = sd.push(silu_v.clone());
    let sidx_d = sd.push_idx(vec![]);
    let silu_d = sd.push(vec![]);
    let ffn_d = sd.push(vec![]);
    let wd_d = sd.push(load_expert_slice_file(&format!("{dir}/L{layer}_experts_down_i32.bin"), &slice_experts));
    let draw_d = sd.push(vec![]);
    let d_t = sd.push(vec![]);
    let ops_d = vec![
        Op::IdxBroadcast { base: tok_d, out: idx_d, b: H },
        Op::Lookup { idx: idx_d, out: seg_d, table: act_d },
        Op::BroadcastScale { x: seg_d, scale: mask_d, out: segz_d, h: H },
        Op::GatherMatMul { a: segz_d, expert_of: exp_d, w: wg_d, c: graw_d, s: s_pad, k: H, n: MOE_PAD, n_experts: 8 },
        Op::Scale { x: graw_d, out: g_d, factor: 1, shift: 16 },
        Op::GatherMatMul { a: segz_d, expert_of: exp_d, w: wu_d, c: uraw_d, s: s_pad, k: H, n: MOE_PAD, n_experts: 8 },
        Op::Scale { x: uraw_d, out: u_d, factor: 1, shift: 16 },
        Op::GeluIndex { x: g_d, out: sidx_d, offset: 1 << 23, table_len: 1 << 24 },
        Op::Lookup { idx: sidx_d, out: silu_d, table: silu_t_d },
        Op::ScaleVec { x: silu_d, scale: u_d, out: ffn_d, shift: 16 },
        Op::GatherMatMul { a: ffn_d, expert_of: exp_d, w: wd_d, c: draw_d, s: s_pad, k: MOE_PAD, n: H, n_experts: 8 },
        Op::Scale { x: draw_d, out: d_t, factor: 1, shift: 16 },
    ];
    let t_d = Instant::now();
    evaluate_ops(&mut sd, &ops_d);
    let d_v = sd.get(d_t).to_vec();
    // Project ② step 2 domain probe: the real gate projection is the input of
    // GeluIndex(offset=2^23, table_len=2^24). Record its signed range and the
    // derived w=x+2^23 / high=w-(2^24-1) ranges; the 25-bit contract needs
    // all three inside [-2^24, 2^24).
    let g_vals: Vec<i64> = sd.get(g_d).iter().map(|&v| to_i64(v)).collect();
    let g_min = *g_vals.iter().min().unwrap();
    let g_max = *g_vals.iter().max().unwrap();
    let w_min = g_min + (1i64 << 23);
    let w_max = g_max + (1i64 << 23);
    let hi_min = w_min - ((1i64 << 24) - 1);
    let hi_max = w_max - ((1i64 << 24) - 1);
    println!("D gate range [{g_min},{g_max}] w=[{w_min},{w_max}] high=[{hi_min},{hi_max}]");
    assert!(g_min >= -(1i64 << 24) && g_max < (1i64 << 24), "real g outside 25-bit x domain");
    assert!(w_min >= -(1i64 << 24) && w_max < (1i64 << 24), "real w outside 25-bit ReLU domain");
    assert!(hi_min >= -(1i64 << 24) && hi_max < (1i64 << 24), "real high outside 25-bit ReLU domain");
    println!("D forward {:?} d_len={} (caller-computed; full FFN tail blocked at ScaleVec shift=16)", t_d.elapsed(), d_v.len());

    // ---------------- Phase D (project ② step 2b): full FFN tail ----------
    // Trust boundary: g is B's proven gate projection (asserted equal below);
    // segz is the real activation segment and exp the real public routing
    // slice; both are caller-trusted public inputs. The segment proves
    //   GatherMatMul(up) -> Scale16(u) -> GeluIndex(offset 2^23, len 2^24)
    //   -> Lookup(real silu 2^24) -> ScaleVec(shift 16) ->
    //   GatherMatMul(down) -> Scale16(out)
    // with the real up/down expert weights and the new rounding-tail
    // ScaleVec reduction. The output d must equal the caller d consumed by C.
    assert_eq!(sd.get(g_d), sb.get(g), "D gate input must equal B's proven g");
    let ops_ffn = vec![
        Op::GatherMatMul { a: segz_d, expert_of: exp_d, w: wu_d, c: uraw_d, s: s_pad, k: H, n: MOE_PAD, n_experts: 8 },
        Op::Scale { x: uraw_d, out: u_d, factor: 1, shift: 16 },
        Op::GeluIndex { x: g_d, out: sidx_d, offset: 1 << 23, table_len: 1 << 24 },
        Op::Lookup { idx: sidx_d, out: silu_d, table: silu_t_d },
        Op::ScaleVec { x: silu_d, scale: u_d, out: ffn_d, shift: 16 },
        Op::GatherMatMul { a: ffn_d, expert_of: exp_d, w: wd_d, c: draw_d, s: s_pad, k: MOE_PAD, n: H, n_experts: 8 },
        Op::Scale { x: draw_d, out: d_t, factor: 1, shift: 16 },
    ];
    let t_ff_fwd = Instant::now();
    evaluate_ops(&mut sd, &ops_ffn);
    let d_ffn = sd.get(d_t).to_vec();
    assert_eq!(d_ffn, d_v, "proven FFN output must equal the caller d used by C");
    let silu_max = sd.get(silu_d).iter().map(|&v| to_i64(v).abs()).max().unwrap_or(0);
    let u_max = sd.get(u_d).iter().map(|&v| to_i64(v).abs()).max().unwrap_or(0);
    println!("FFN ScaleVec domain |silu|max={} |u|max={} (contract < 2^24)", silu_max, u_max);
    assert!(silu_max < (1i64 << 24) && u_max < (1i64 << 24),
        "real ScaleVec operands must stay in the 25-bit signed domain");
    // The ScaleVec shift>0 push arm references the identity table of length
    // 2^shift; provide the trusted identity table for the manifest.
    let _id16 = sd.push((0..(1usize << 16)).map(|i| Goldilocks::from_u64(i as u64)).collect());
    let cache_ffn = format!("{cache}-ffn");
    let cached_ffn = commit_model_weights(&sd, &ops_ffn, &cache_ffn);
    let sidx_len = sd.idx[sidx_d].len();
    let model_ffn = TrustedModelManifest::from_offline_commitment(&cached_ffn, &sd.owned_slices(), &ops_ffn)
        .with_index_shapes(&[(exp_d, s_pad), (sidx_d, sidx_len)].into());
    let exp_values = to_g(&sd.idx[exp_d]);
    let exec_ffn = TrustedExecution {
        inputs: [
            (segz_d, sd.get(segz_d).to_vec()),
            (g_d, sd.get(g_d).to_vec()),
            (TensorKey::Index(exp_d).encoded().unwrap(), exp_values),
        ].into(),
        outputs: [(d_t, d_ffn.clone())].into(),
    };
    let t_ff = Instant::now();
    let proof_ffn = prove_committed_shard_dag_bound(&mut sd, &ops_ffn, 1, &whir, &cached_ffn, 0, 0,
        &PublicStatement::Generic, &model_ffn, &exec_ffn)
        .expect("phase D full FFN proof");
    let bytes_ffn = bincode::serialize(&proof_ffn).expect("serialize FFN");
    let ff_prove_t = t_ff.elapsed();
    let t_ff_v = Instant::now();
    assert!(verify_committed_shard_dag_bound_bytes(&bytes_ffn, &ops_ffn, 1, &whir,
        &PublicStatement::Generic, &model_ffn, &exec_ffn), "phase D must verify");
    println!("FFN commit+prove {:?} verify {:?} bytes {} hash={:016x} d_len={}",
        ff_prove_t, t_ff_v.elapsed(), bytes_ffn.len(), fnv(&bytes_ffn), d_ffn.len());
    let mut bad_ffn_out = exec_ffn.clone();
    bad_ffn_out.outputs.get_mut(&d_t).unwrap()[0] += Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_ffn, &ops_ffn, 1, &whir,
        &PublicStatement::Generic, &model_ffn, &bad_ffn_out));
    for bad_input in [segz_d, g_d] {
        let mut bad = exec_ffn.clone();
        bad.inputs.get_mut(&bad_input).unwrap()[0] += Goldilocks::ONE;
        assert!(!verify_committed_shard_dag_bound_bytes(&bytes_ffn, &ops_ffn, 1, &whir,
            &PublicStatement::Generic, &model_ffn, &bad));
    }
    let mut bad_exp = exec_ffn.clone();
    let ek = TensorKey::Index(exp_d).encoded().unwrap();
    bad_exp.inputs.get_mut(&ek).unwrap()[0] += Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_ffn, &ops_ffn, 1, &whir,
        &PublicStatement::Generic, &model_ffn, &bad_exp));
    println!("negative rejected: FFN u source / silu source / out tamper");
    // Wrong real silu table root: mutate the real table and rebuild the
    // trusted manifest. The original proof cannot authenticate against it.
    sd.get_mut(silu_t_d)[0] += Goldilocks::ONE;
    let bad_model_ffn = TrustedModelManifest::from_offline_commitment(&cached_ffn, &sd.owned_slices(), &ops_ffn)
        .with_index_shapes(&[(exp_d, s_pad), (sidx_d, sidx_len)].into());
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_ffn, &ops_ffn, 1, &whir,
        &PublicStatement::Generic, &bad_model_ffn, &exec_ffn));
    println!("negative rejected: wrong real silu table root");

    // ---------------- Phase C: router-gate consistency chain + Gather ----------------
    // TopKSelect re-derives the router gate from the same real scores; the
    // Lookup table is the DYNAMIC TopK gate tensor (not a static table), so
    // the reduction ties gate_pos to the router output. d is the real
    // down-projection supplied as a caller-trusted public input.
    let mut sc = Store::new();
    let scores_c = sc.push(scores_v.clone());
    let sel_c = sc.push(vec![]);
    let thr_c = sc.push(vec![]);
    let gate_c = sc.push(vec![]);
    let d1_c = sc.push(vec![]);
    let d2_c = sc.push(vec![]);
    let tok_c = sc.push_idx(vec![]);
    let exp_c = sc.push_idx(vec![]);
    let tokp_c = sc.push_idx(vec![]);
    let expp_c = sc.push_idx(vec![]);
    let idxg_c = sc.push_idx(vec![]);
    let gpos_c = sc.push(vec![]);
    let d_c = sc.push(d_v.clone());
    let wraw_c = sc.push(vec![]);
    let w_c = sc.push(vec![]);
    let ones_c = sc.push(vec![Goldilocks::ONE; s_pad]);
    let out_c = sc.push(vec![]);
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
    evaluate_ops(&mut sc, &ops_c);
    let c_fwd_t = t_c.elapsed();
    let out_v = sc.get(out_c).to_vec();
    assert_eq!(sc.get(thr_c), thr_v.as_slice(), "C router threshold matches A");
    assert_eq!(sc.get(gate_c), gate_v.as_slice(), "C router gate matches A");
    assert_eq!(sc.get(d1_c), d1_v.as_slice(), "C d1 matches A");
    assert_eq!(sc.get(d2_c), d2_v.as_slice(), "C d2 matches A");
    let exec_c = TrustedExecution {
        inputs: [
            (scores_c, scores_v.clone()),
            (d_c, d_v.clone()),
            (ones_c, vec![Goldilocks::ONE; s_pad]),
        ].into(),
        outputs: [
            (thr_c, thr_v.clone()), (gate_c, gate_v.clone()),
            (d1_c, d1_v.clone()), (d2_c, d2_v.clone()),
            (out_c, out_v.clone()),
        ].into(),
    };
    let cache_c = format!("{cache}-c");
    let cached_c = commit_model_weights(&sc, &ops_c, &cache_c);
    let model_c = TrustedModelManifest::from_offline_commitment(&cached_c, &sc.owned_slices(), &ops_c)
        .with_index_shapes(&[
            (tok_c, s), (exp_c, s), (tokp_c, s_pad), (expp_c, s_pad), (idxg_c, s_pad),
        ].into());
    let t_cp = Instant::now();
    let proof_c = prove_committed_shard_dag_bound(&mut sc, &ops_c, 1, &whir, &cached_c, 0, 0,
        &PublicStatement::Generic, &model_c, &exec_c)
        .expect("phase C router-gate + Gather proof");
    let bytes_c = bincode::serialize(&proof_c).expect("serialize C");
    let t_cv = Instant::now();
    assert!(verify_committed_shard_dag_bound_bytes(&bytes_c, &ops_c, 1, &whir,
        &PublicStatement::Generic, &model_c, &exec_c), "phase C must verify");
    let c_verify_t = t_cv.elapsed();
    println!("C forward {:?} commit+prove {:?} verify {:?} bytes {} out_len={}",
        c_fwd_t, t_cp.elapsed(), c_verify_t, bytes_c.len(), out_v.len());

    // Same trusted statement on the B side: the prover recomputed these
    // arrays from its own forward, so also assert they equal the A outputs
    // used above (item by item), never comparing proof self-reported fields.
    assert_eq!(to_g(&stmt_tok), to_g(&tok_v));
    assert_eq!(to_g(&stmt_exp), to_g(&exp_v));
    // Item-by-item external-statement binding through the public remap:
    // B's public routing input is the remap of A's public routing output.
    let bound: Vec<u32> = (0..stmt_exp.len()).map(slice_of).collect();
    assert_eq!(bound, exp_slice);
    assert_eq!(sb.idx[tok_b], stmt_tok);
    assert_eq!(sb.idx[exp_b], bound);
    assert_eq!(execution_b.inputs.get(&kb_tok), Some(&to_g(&stmt_tok)));
    assert_eq!(execution_b.inputs.get(&kb_exp), Some(&to_g(&bound)));
    // Caller-trusted external statement: the ONLY input to the composition
    // verifier below (prover-local arrays are not consulted by it).
    let stmt = CoreStatement {
        scores: scores_v.clone(), thr: thr_v.clone(), gate: gate_v.clone(),
        d1: d1_v.clone(), d2: d2_v.clone(),
        tok_routing: tok_v.clone(), exp_routing: exp_v.clone(),
        mask: mask_v.clone(), g: g_v.clone(),
        d: d_v.clone(), out: out_v.clone(),
        slice_experts: slice_experts.clone(), slice: exp_slice.clone(), s,
    };
    let cache_b = format!("{cache}-b");
    let t3 = Instant::now();
    let cached_b = commit_model_weights(&sb, &ops_b, &cache_b);
    let model_b = TrustedModelManifest::from_offline_commitment(&cached_b, &sb.owned_slices(), &ops_b)
        .with_index_shapes(&[(tok_b, s_pad), (exp_b, s_pad), (idx_seg, s_pad * H)].into());
    sb.o7_metrics_reset();
    let proof_b = prove_committed_shard_dag_bound(&mut sb, &ops_b, 1, &whir, &cached_b, 0, 0,
        &PublicStatement::Generic, &model_b, &execution_b)
        .expect("phase B real FFN proof");
    let bytes_b = bincode::serialize(&proof_b).expect("serialize B");
    let b_prove_t = t3.elapsed();
    let (o7_trees, o7_openings, o7_scan) = sb.o7_deep_metrics();
    let o7_commits = sb.o7_table_commits();
    let t4 = Instant::now();
    assert!(verify_committed_shard_dag_bound_bytes(&bytes_b, &ops_b, 1, &whir,
        &PublicStatement::Generic, &model_b, &execution_b), "phase B must verify");
    // Capture the B verify time HERE: the print below happens after the
    // composition/B2 tests and must not fold their cost into this number.
    let b_verify_t = t4.elapsed();
    // Independent composition verification (caller statement + both proofs
    // + models + ops only).
    assert!(verify_segmented(&stmt, &ops_a, &model_a, &bytes_a, &ops_b, &model_b, &bytes_b, &ops_c, &model_c, &bytes_c, &whir),
        "segmented honest composition must verify");
    println!("composition honest verify ok");
    let mut bad_remap = stmt.clone();
    bad_remap.slice_experts[1] = bad_remap.slice_experts[0];
    assert!(!verify_segmented(&bad_remap, &ops_a, &model_a, &bytes_a, &ops_b, &model_b, &bytes_b, &ops_c, &model_c, &bytes_c, &whir),
        "duplicate slice identity must be rejected");
    let mut bad_slice = stmt.clone();
    bad_slice.slice[0] = bad_slice.slice[1];
    assert!(!verify_segmented(&bad_slice, &ops_a, &model_a, &bytes_a, &ops_b, &model_b, &bytes_b, &ops_c, &model_c, &bytes_c, &whir),
        "statement slice/routing mismatch must be rejected");
    println!("negative rejected: invalid slice identity / slice mismatch");
    // Cross-segment: B2 is INDIVIDUALLY valid for statement S2 (routing swap)
    // but the composition with A1 (statement S1) must be rejected.
    let mut stmt2 = stmt.clone();
    stmt2.exp_routing.swap(0, 1);
    stmt2.slice = derive_slice(&stmt2.exp_routing, &stmt2.slice_experts, s).expect("valid slice2");
    let mut s2 = Store::new();
    let t2 = s2.push_idx(stmt2.tok_routing.clone());
    let e2 = s2.push_idx(stmt2.slice.clone());
    let m2 = s2.push(stmt2.mask.clone());
    let a2 = s2.push(act_v.clone());
    let i2 = s2.push_idx(vec![]); let sg2 = s2.push(vec![]); let sz2 = s2.push(vec![]);
    let gr2 = s2.push(vec![]); let g2t = s2.push(vec![]);
    let w2s = s2.push(load_expert_slice(&dir, layer, &slice_experts));
    let ops2 = vec![
        Op::IdxBroadcast { base: t2, out: i2, b: H },
        Op::Lookup { idx: i2, out: sg2, table: a2 },
        Op::BroadcastScale { x: sg2, scale: m2, out: sz2, h: H },
        Op::GatherMatMul { a: sz2, expert_of: e2, w: w2s, c: gr2, s: s_pad, k: H, n: MOE_PAD, n_experts: 8 },
        Op::Scale { x: gr2, out: g2t, factor: 1, shift: 16 },
    ];
    evaluate_ops(&mut s2, &ops2);
    stmt2.g = s2.get(g2t).to_vec();
    let exec_b2 = TrustedExecution {
        inputs: [
            (TensorKey::Index(t2).encoded().unwrap(), to_g(&stmt2.tok_routing)),
            (TensorKey::Index(e2).encoded().unwrap(), to_g(&stmt2.slice)),
            (m2, stmt2.mask.clone()),
        ].into(),
        outputs: [(g2t, stmt2.g.clone())].into(),
    };
    let proof_b2 = prove_committed_shard_dag_bound(&mut s2, &ops2, 1, &whir, &cached_b, 0, 0,
        &PublicStatement::Generic, &model_b, &exec_b2).expect("B2 individually valid proof");
    let bytes_b2 = bincode::serialize(&proof_b2).expect("serialize B2");
    assert!(verify_committed_shard_dag_bound_bytes(&bytes_b2, &ops2, 1, &whir,
        &PublicStatement::Generic, &model_b, &exec_b2), "B2 must be individually valid");
    assert!(!verify_segmented(&stmt, &ops_a, &model_a, &bytes_a, &ops2, &model_b, &bytes_b2, &ops_c, &model_c, &bytes_c, &whir),
        "cross-segment A1 + B2 mismatch must be rejected");
    println!("negative rejected: cross-segment individually-valid A1+B2 mismatch");
    println!("B commit+prove {:?} verify {:?} bytes {} peak_rss={}kB o7_commits={} o7_trees={} o7_openings={} o7_scan_bytes={}",
        b_prove_t, b_verify_t, bytes_b.len(), peak_rss_kb(), o7_commits, o7_trees, o7_openings, o7_scan);

    // S9 2d: the former O2 route-A measurement hook consumed the weight
    // opening's points/claimed values as base-field Goldilocks. Those fields
    // are EF after the 2b-2b migration, and the one-time measurement is
    // obsolete; the O2 results are recorded in docs. Removed instead of
    // converting, which would have materialized an EF copy of the real table.

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
    let mut bad_exp = to_g(&exp_slice);
    bad_exp[0] = Goldilocks::from_u64(exp_slice[1] as u64);
    bad_b_inputs.inputs.insert(kb_exp, bad_exp);
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_b, &ops_b, 1, &whir,
        &PublicStatement::Generic, &model_b, &bad_b_inputs),
        "B tampered (duplicate) public routing array must be rejected");
    println!("negative rejected: B tampered public routing array");
    // Cross-segment tamper: A's public routing output altered in a fresh
    // execution (the proof bytes stay honest).
    let mut bad_a_route = execution_a.clone();
    // The token column is all-zero for seq=1, so tamper the EXPERT routing
    // output (26 vs 30) to make the statement mismatch observable.
    let mut bad_exp_out = to_g(&stmt_exp);
    bad_exp_out.swap(0, 1);
    bad_a_route.outputs.insert(ek_key, bad_exp_out);
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_a, &ops_a, 1, &whir,
        &PublicStatement::Generic, &model_a, &bad_a_route),
        "cross-segment tampered A routing output must be rejected");
    println!("negative rejected: cross-segment A routing output tamper");
    // Cross-segment mismatch: feed B an execution whose routing inputs are a
    // different routing table (swap two experts), while the proof bytes were
    // made for the statement routing.
    let mut cross = execution_b.clone();
    let mut cross_exp = exp_slice.clone();
    cross_exp.swap(0, 1);
    cross.inputs.insert(kb_exp, to_g(&cross_exp));
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_b, &ops_b, 1, &whir,
        &PublicStatement::Generic, &model_b, &cross),
        "cross-segment routing mismatch must be rejected at the bytes entry");
    println!("negative rejected: cross-segment B routing mismatch");
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
    let exp2 = s2.push_idx(exp_slice.clone());
    let mask2 = s2.push(mask_v.clone());
    let act2_t = s2.push(act2);
    let idx2 = s2.push_idx(vec![]); let seg2 = s2.push(vec![]); let segz2 = s2.push(vec![]);
    let graw2 = s2.push(vec![]); let g2 = s2.push(vec![]);
    let w2 = s2.push(load_expert_slice(&dir, layer, &slice_experts));
    let ops2 = vec![
        Op::IdxBroadcast { base: tok2, out: idx2, b: H },
        Op::Lookup { idx: idx2, out: seg2, table: act2_t },
        Op::BroadcastScale { x: seg2, scale: mask2, out: segz2, h: H },
        Op::GatherMatMul { a: segz2, expert_of: exp2, w: w2, c: graw2, s: s_pad, k: H, n: MOE_PAD, n_experts: 8 },
        Op::Scale { x: graw2, out: g2, factor: 1, shift: 16 },
    ];
    // Same real weight batch (identical slice), different activation table
    // root: the verifier must reject the honest proof against this model.
    evaluate_ops(&mut s2, &ops2);
    let model2 = TrustedModelManifest::from_offline_commitment(&cached_b, &s2.owned_slices(), &ops2)
        .with_index_shapes(&[(tok2, s_pad), (exp2, s_pad), (idx2, s_pad * H)].into());
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_b, &ops_b, 1, &whir,
        &PublicStatement::Generic, &model2, &execution_b), "wrong activation table root must be rejected");
    println!("negative rejected: wrong activation table root");

    // ---------------- Phase C negatives (real router scores + real d) ----------------
    let mut bad_c_d = exec_c.clone();
    bad_c_d.inputs.get_mut(&d_c).unwrap()[0] = bad_c_d.inputs[&d_c][0] + Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_c, &ops_c, 1, &whir,
        &PublicStatement::Generic, &model_c, &bad_c_d), "C tampered d input must be rejected");
    println!("negative rejected: C tampered d input");
    let mut bad_c_scores = exec_c.clone();
    bad_c_scores.inputs.get_mut(&scores_c).unwrap()[0] = bad_c_scores.inputs[&scores_c][0] + Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_c, &ops_c, 1, &whir,
        &PublicStatement::Generic, &model_c, &bad_c_scores), "C tampered score input must be rejected");
    println!("negative rejected: C tampered router score input");
    let mut bad_c_out = exec_c.clone();
    bad_c_out.outputs.get_mut(&out_c).unwrap()[0] = bad_c_out.outputs[&out_c][0] + Goldilocks::ONE;
    assert!(!verify_committed_shard_dag_bound_bytes(&bytes_c, &ops_c, 1, &whir,
        &PublicStatement::Generic, &model_c, &bad_c_out), "C tampered gather output must be rejected");
    println!("negative rejected: C tampered gather output");
    let mut bad_stmt_d = stmt.clone();
    bad_stmt_d.d[0] = bad_stmt_d.d[0] + Goldilocks::ONE;
    assert!(!verify_segmented(&bad_stmt_d, &ops_a, &model_a, &bytes_a, &ops_b, &model_b, &bytes_b,
        &ops_c, &model_c, &bytes_c, &whir), "statement d tamper must be rejected");
    println!("negative rejected: cross-segment statement d tamper");
    let mut bad_stmt_out = stmt.clone();
    bad_stmt_out.out[0] = bad_stmt_out.out[0] + Goldilocks::ONE;
    assert!(!verify_segmented(&bad_stmt_out, &ops_a, &model_a, &bytes_a, &ops_b, &model_b, &bytes_b,
        &ops_c, &model_c, &bytes_c, &whir), "statement out tamper must be rejected");
    println!("negative rejected: cross-segment statement out tamper");

    // Static gate table in the real-graph chain: the Lookup table is NOT the
    // TopK gate tensor. The dynamic-table mode must fail closed; if this
    // verifies, the router-gate consistency chain is NOT enforced for the
    // BroadcastScale path and must be reported to the TL.
    let mut bad_gate_v = gate_v.clone();
    bad_gate_v[0] = bad_gate_v[0] + Goldilocks::ONE;
    let mut sbad = Store::new();
    let scores_bad = sbad.push(scores_v.clone());
    let sel_bad = sbad.push(vec![]);
    let thr_bad = sbad.push(vec![]);
    let gate_bad = sbad.push(vec![]);
    let d1_bad = sbad.push(vec![]);
    let d2_bad = sbad.push(vec![]);
    let tok_bad = sbad.push_idx(vec![]);
    let exp_bad = sbad.push_idx(vec![]);
    let tokp_bad = sbad.push_idx(vec![]);
    let expp_bad = sbad.push_idx(vec![]);
    let idxg_bad = sbad.push_idx(vec![]);
    let gpos_bad = sbad.push(vec![]);
    let d_bad = sbad.push(d_v.clone());
    let wraw_bad = sbad.push(vec![]);
    let w_bad = sbad.push(vec![]);
    let ones_bad = sbad.push(vec![Goldilocks::ONE; s_pad]);
    let out_bad = sbad.push(vec![]);
    let bad_gate_t = sbad.push(bad_gate_v);
    let ops_bad = vec![
        Op::TopKSelect { x: scores_bad, sel: sel_bad, thr: thr_bad, gate: gate_bad, d1: d1_bad, d2: d2_bad, m, n: N, k: K },
        Op::Scatter { sel: sel_bad, token_id: tok_bad, expert_of: exp_bad, m, n: N },
        Op::PadIdx { base: tok_bad, out: tokp_bad, len: s_pad, base_len: s },
        Op::PadIdx { base: exp_bad, out: expp_bad, len: s_pad, base_len: s },
        Op::CombineIdx { a: tokp_bad, b: expp_bad, out: idxg_bad, scale: N },
        Op::Lookup { idx: idxg_bad, out: gpos_bad, table: bad_gate_t },
        Op::BroadcastScale { x: d_bad, scale: gpos_bad, out: wraw_bad, h: H },
        Op::Scale { x: wraw_bad, out: w_bad, factor: 1, shift: 16 },
        Op::Gather { weighted: w_bad, token_id: tokp_bad, gate: ones_bad, out: out_bad, m, h: H },
    ];
    evaluate_ops(&mut sbad, &ops_bad);
    let out_bad_v = sbad.get(out_bad).to_vec();
    let exec_bad = TrustedExecution {
        inputs: [
            (scores_bad, scores_v.clone()), (d_bad, d_v.clone()),
            (ones_bad, vec![Goldilocks::ONE; s_pad]),
        ].into(),
        outputs: [
            (thr_bad, thr_v.clone()), (gate_bad, gate_v.clone()),
            (d1_bad, d1_v.clone()), (d2_bad, d2_v.clone()),
            (out_bad, out_bad_v),
        ].into(),
    };
    let cached_bad = commit_model_weights(&sbad, &ops_bad, &format!("{cache}-cbad"));
    let model_bad = TrustedModelManifest::from_offline_commitment(&cached_bad, &sbad.owned_slices(), &ops_bad)
        .with_index_shapes(&[
            (tok_bad, s), (exp_bad, s), (tokp_bad, s_pad), (expp_bad, s_pad), (idxg_bad, s_pad),
        ].into());
    assert!(prove_committed_shard_dag_bound(&mut sbad, &ops_bad, 1, &whir, &cached_bad, 0, 0,
        &PublicStatement::Generic, &model_bad, &exec_bad).is_none(),
        "static gate table in the real-graph chain must fail closed");
    println!("negative rejected: C static gate table fails closed (router-gate consistency)");
    if let Ok(prefix) = std::env::var("ZKIE_O2_DUMP_PREFIX") {
        std::fs::write(format!("{prefix}-a.bin"), &bytes_a).expect("dump A bytes");
        std::fs::write(format!("{prefix}-b.bin"), &bytes_b).expect("dump B bytes");
        std::fs::write(format!("{prefix}-c.bin"), &bytes_c).expect("dump C bytes");
        std::fs::write(format!("{prefix}-ffn.bin"), &bytes_ffn).expect("dump FFN bytes");
    }
    println!("SEGMENT-OK layer={layer} (A routing + B gate projection + D gelu/silu 2^24 prefix + C router-gate/Gather over caller-trusted real d) A_bytes={} B_bytes={} C_bytes={} FFN_bytes={} FFN_hash={:016x}",
        bytes_a.len(), bytes_b.len(), bytes_c.len(), bytes_ffn.len(), fnv(&bytes_ffn));
}
