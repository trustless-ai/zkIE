//! How large is a shard-DAG proof, and which part of it is large?
//!
//! This walks the [`ShardDagProof`] structure and counts **field elements**
//! rather than serialising. That is deliberate and it is the more useful
//! quantity: wrapping the transcript in a SNARK costs constraints roughly in
//! proportion to its field elements, while a `bincode` framing byte is not
//! in-circuit. The number this reports is therefore an input to wrap cost,
//! **not** a wire size, and callers must not quote it as one. (Since the proof
//! structs now derive `Serialize`, `bincode::serialize(&proof).len()` is easy to
//! reach for and badly wrong — it drags in `cross_tensors`, the raw witness a
//! verifier never receives. This file exists to give the honest number.)
//!
//! ## The completeness guard
//!
//! An incomplete walk silently under-reports, and an under-reported proof size
//! is worse than no measurement. So **every struct below is destructured without
//! `..`, and the `OpProof` match carries no wildcard.** Adding a field or a
//! variant then fails to compile instead of being silently counted as zero.
//!
//! ## Ported to zkIE HEAD (`@9eaa8f1`) — what changed since the Oct sizer
//!
//! The prover's proof structure moved on since this sizer was first written
//! (Oct `@7e1678e`). The port re-derives every counter against the current
//! structs; the material changes are:
//!   * `OpProof::Lookup` is now payload-free — the per-op lookup proof moved to
//!     the shard-level `witness_columns` / `table_columns` (counted in
//!     `add_shard`), so lookups are batched per shard rather than per op.
//!   * `FractionalProof` gained `full_r`; `SoftmaxRoundedProof`,
//!     `Layernorm*`, `RmsNormProof`, `RoPEProof` and `TopKSelectProof` were
//!     restructured (the lookup/range sub-proofs they used to carry are gone or
//!     replaced by `RangeDecompProof`).
//!   * New ops: `Argmax`, `Scatter`, `Gather`, `BatchedMatMul`, `GatherMatMul`,
//!     and four payload-free index ops (`IdxBroadcast`/`CombineIdx`/`PadIdx`/
//!     `BroadcastScale`).
//!
//! ## The accounting rule: count only what the verifier reads
//!
//! The honest proof size is the data the verifier consumes, not every field the
//! prover happens to park in the (serializable) proof struct. The Oct sizer
//! applied this already — it excluded `cross_tensors` (raw witness). On HEAD the
//! same trap reappears inside the new lookup machinery: a `WitnessColumn` /
//! `TableColumn` carries the full lookup columns (`x`/`y`/`num`/`m`/`point`),
//! but the verifier (`verify_shard_precomputed` → `verify_lookup_split`) reads
//! **only each column's `.proof`** (a `FractionalProof`) plus its `table` id.
//! Likewise `TopKSelectProof::sel` (the selection matrix) and `ScatterAddProof`'s
//! gather columns are witness: `verify_topk` / `verify_argmax` /
//! `verify_scatter_add` never read them (argmax touches `topk.sel` only in the
//! `sel_evals.is_empty()` fallback, which the real proof does not take).
//! Counting those raw columns inflates the figure ~400x — the very bincode trap
//! this file exists to avoid — so they are destructured to `_` with the reason.
//!
//! Within what the verifier *does* read: a `Goldilocks` is one field element; a
//! transmitted `Vec<Goldilocks>` (sumcheck rounds, final evals, opening points,
//! logup challenge vectors `r`/`r_*`/`*_ch`) counts by length; Fiat–Shamir
//! challenges the verifier re-derives count zero. `usize`/`T` tensor ids and
//! `u32` output indices are reported under `tensor_ids` (4 bytes), not folded
//! into the field-element total.
//!
//! The GPT-2-exercised arms (Projection/Softmax/MatMul/Add/LayerNormCentered/
//! lookup columns/Argmax) are validated by the `proof_size_gpt2` run + verify.
//! The arms for ops GPT-2 does not reach (Scatter/Gather/BatchedMatMul/
//! GatherMatMul/RoPE/RmsNorm/Layernorm chain/standalone TopK) count their
//! verify-read proof objects (sumchecks/fractionals) and exclude their witness
//! vectors; those are marked `// FLAG` for the prover author to confirm the
//! read-set, since no GPT-2 run exercises them.

use std::collections::BTreeMap;

use zkie_core::common::batched_matmul::{BatchedMatmulProof, GatherMatMulProof};
use zkie_core::common::field::Goldilocks;
use zkie_core::common::logup_gkr::{FractionalLayer, FractionalProof, PermutationProof};
use zkie_core::common::matmul::MatmulProof;
use zkie_core::common::same_poly::SamePolyProof;
use zkie_core::common::sumcheck::{RoundPoly, SumcheckProof, VirtualProof};

use crate::compose::{OpProof, OpShardProof, ShardDagProof, TableColumn, WitnessColumn};
use crate::gather::ScatterAddProof;
use crate::layer_norm_centered::LayerNormCenteredProof;
use crate::layernorm_chain::LayernormChainProof;
use crate::projection::ProjectionProof;
use crate::rms_norm::RmsNormProof;
use crate::rope::RoPEProof;
use crate::scatter::ScatterProof;
use crate::softmax_scaled::SoftmaxRoundedProof;
use crate::topk::{ArgmaxProof, RangeDecompProof, TopKSelectProof};

/// A Goldilocks element is 8 bytes on the wire and one value in-circuit.
pub const FE_BYTES: usize = 8;

/// Where a proof's field elements are.
#[derive(Default, Debug, Clone)]
pub struct ProofSize {
    /// Field elements inside the per-op proofs.
    pub op_proof_fe: usize,
    /// Field elements in the raw `(tensor, point, eval)` claims. The verifier
    /// reads these (`verify_shard_dag` touches `shard.claims`), so they count.
    pub claims_fe: usize,
    /// Field elements in the within-shard `same_poly` bindings.
    pub shard_binds_fe: usize,
    /// Field elements in the cross-shard `same_poly` bindings.
    pub cross_binds_fe: usize,
    /// Field elements in the shard-level lookup witness columns (HEAD).
    pub witness_cols_fe: usize,
    /// Field elements in the shard-level lookup table columns (HEAD).
    pub table_cols_fe: usize,
    /// Tensor ids and `u32` output indices. Indices rather than field elements,
    /// reported separately rather than folded into the total.
    pub tensor_ids: usize,
    /// Field elements per bucket (`OpProof` variant, plus the two column kinds).
    pub fe_by_op: BTreeMap<&'static str, usize>,
    /// How many of each bucket appeared.
    pub count_by_op: BTreeMap<&'static str, usize>,
    pub shards: usize,
}

impl ProofSize {
    /// Total field elements. Excludes tensor ids, which are not field elements.
    pub fn total_fe(&self) -> usize {
        self.op_proof_fe
            + self.claims_fe
            + self.shard_binds_fe
            + self.cross_binds_fe
            + self.witness_cols_fe
            + self.table_cols_fe
    }
    /// Total bytes, counting one field element as 8 and one tensor id as 4.
    /// An information-content figure, **not** a serialised size.
    pub fn total_bytes(&self) -> usize {
        self.total_fe() * FE_BYTES + self.tensor_ids * 4
    }
}

// ---- leaves -----------------------------------------------------------------

fn round_poly_fe(p: &RoundPoly) -> usize {
    let RoundPoly { c0: _, c1: _, c2: _ } = p;
    3
}

fn sumcheck_fe(p: &SumcheckProof) -> usize {
    let SumcheckProof { rounds, f_eval: _, h_eval: _ } = p;
    rounds.iter().map(round_poly_fe).sum::<usize>() + 2
}

fn virtual_fe(p: &VirtualProof) -> usize {
    let VirtualProof { rounds, final_evals } = p;
    rounds.iter().map(Vec::len).sum::<usize>() + final_evals.len()
}

fn same_poly_fe(p: &SamePolyProof) -> usize {
    let SamePolyProof { coeffs, proof, merged_point, merged_eval: _ } = p;
    coeffs.len() + virtual_fe(proof) + merged_point.len() + 1
}

fn matmul_fe(p: &MatmulProof) -> usize {
    let MatmulProof { claimed: _, sumcheck } = p;
    1 + sumcheck_fe(sumcheck)
}

fn fractional_layer_fe(l: &FractionalLayer) -> usize {
    let FractionalLayer { r, c: _, proof } = l;
    r.len() + 1 + virtual_fe(proof)
}

fn fractional_fe(p: &FractionalProof) -> usize {
    // HEAD: `full_r` (the merged challenge point) is now carried explicitly.
    let FractionalProof { layers, final_num: _, final_den: _, full_r } = p;
    layers.iter().map(fractional_layer_fe).sum::<usize>() + 2 + full_r.len()
}

// ---- composite op proofs ----------------------------------------------------

fn projection_fe(p: &ProjectionProof) -> usize {
    // Unchanged since Oct — kept identical, including the two transmitted claims.
    let ProjectionProof { matmul, affine, frac, alpha: _, beta: _, pt, u, v, ch } = p;
    matmul_fe(matmul) + virtual_fe(affine) + fractional_fe(frac) + 2 + pt.len() + u.len() + v.len() + ch.len()
}

fn softmax_fe(p: &SoftmaxRoundedProof) -> usize {
    // HEAD: the lookup `FractionalProof` and alpha/beta are gone; a transmitted
    // `claimed` scalar replaces them.
    let SoftmaxRoundedProof { row_sum, rescale, r_sum, sum_ch, r_scale, claimed: _ } = p;
    virtual_fe(row_sum) + virtual_fe(rescale) + r_sum.len() + sum_ch.len() + r_scale.len() + 1
}

fn layernorm_chain_fe(p: &LayernormChainProof) -> usize {
    // HEAD: `rsqrt`/alpha/beta dropped; a transmitted `mean_claim` added.
    let LayernormChainProof { mean_sq, out, r_mean, mean_ch, r_out, mean_claim: _ } = p;
    virtual_fe(mean_sq) + virtual_fe(out) + r_mean.len() + mean_ch.len() + r_out.len() + 1
}

fn layer_norm_centered_fe(p: &LayerNormCenteredProof) -> usize {
    // HEAD: `rstd`/alpha/beta dropped.
    let LayerNormCenteredProof { mean, centered, var, out, r_mean, mean_ch, centered_r, r_var, var_ch, r_out } = p;
    virtual_fe(mean) + virtual_fe(centered) + virtual_fe(var) + virtual_fe(out)
        + r_mean.len() + mean_ch.len() + centered_r.len() + r_var.len() + var_ch.len() + r_out.len()
}

fn rms_norm_fe(p: &RmsNormProof) -> usize {
    // HEAD: collapsed to three sumchecks + three challenge vectors.
    let RmsNormProof { trunc, out, rem_split, trunc_ch, r_out, rem_split_ch } = p;
    virtual_fe(trunc) + virtual_fe(out) + virtual_fe(rem_split) + trunc_ch.len() + r_out.len() + rem_split_ch.len()
}

fn rope_fe(p: &RoPEProof) -> usize {
    // HEAD: the two fractional range proofs and their alpha/beta are gone.
    let RoPEProof { first, second, r_first, r_second, r_out } = p;
    virtual_fe(first) + virtual_fe(second) + r_first.len() + r_second.len() + r_out.len()
}

fn range_decomp_fe(p: &RangeDecompProof) -> usize {
    // New on HEAD: three limb range proofs + a recomposition sumcheck.
    let RangeDecompProof { c0_range, c1_range, c2_range, recompose, a0: _, b0: _, a1: _, b1: _, a2: _, b2: _, r } = p;
    fractional_fe(c0_range) + fractional_fe(c1_range) + fractional_fe(c2_range) + virtual_fe(recompose) + r.len()
}

fn topk_fe(p: &TopKSelectProof) -> usize {
    // `verify_topk` reads the four sumchecks against the single point `r`, the
    // two lookup fractional chains, and the range decomposition. It never reads
    // `r_row`, the scalar bounds, `range_bits`, or `sel` (the selection matrix —
    // witness; only `verify_argmax`'s empty-`sel_evals` fallback touches it).
    let TopKSelectProof {
        gate_rel, d1_rel, d2_rel, row_sum, sel_range, d1_range, d2_decomp, r,
        r_row: _, a_sel: _, b_sel: _, a_d1: _, b_d1: _, range_bits: _, sel: _,
    } = p;
    virtual_fe(gate_rel) + virtual_fe(d1_rel) + virtual_fe(d2_rel) + virtual_fe(row_sum)
        + fractional_fe(sel_range) + fractional_fe(d1_range) + range_decomp_fe(d2_decomp)
        + r.len()
}

fn permutation_fe(p: &PermutationProof) -> usize {
    let PermutationProof { a_side, b_side, alpha: _, a, b } = p;
    fractional_fe(a_side) + fractional_fe(b_side) + a.len() + b.len()
}

fn batched_matmul_fe(p: &BatchedMatmulProof) -> usize {
    // Like `matmul_fe`, but the sumcheck is a `VirtualProof` here.
    let BatchedMatmulProof { claimed: _, sumcheck } = p;
    1 + virtual_fe(sumcheck)
}

fn gather_matmul_fe(p: &GatherMatMulProof) -> usize {
    // FLAG: not exercised by GPT-2. `claimed` + the sumcheck are proof; `counts`
    // and `table` look like the gathered table witness, excluded pending the
    // author confirming `bmm::verify_gather_matmul`'s read-set.
    let GatherMatMulProof { claimed: _, sumcheck, counts: _, table: _ } = p;
    1 + virtual_fe(sumcheck)
}

/// `(field elements, u32 output indices)` for an argmax proof.
fn argmax_fe(p: &ArgmaxProof) -> (usize, usize) {
    let ArgmaxProof { topk, out, sel_evals } = p;
    (topk_fe(topk) + sel_evals.len(), out.len())
}

/// `(field elements, u32 token-id indices)` for a scatter-add / gather proof.
/// FLAG: not exercised by GPT-2. `verify_scatter_add` reads the two lookup sides
/// and the two product sumchecks; the gather columns (`gamma`/`num_*`/`den_*`/
/// `weighted`/`gate`/`out`/`token_id`) are witness it does not read. Author to
/// confirm `r_prod_*` aren't separately needed as opening points.
fn scatter_add_fe(p: &ScatterAddProof) -> (usize, usize) {
    let ScatterAddProof {
        source_side, target_side, alpha: _, gamma: _,
        num_source: _, den_source: _, num_target: _, den_target: _,
        prod_source, prod_target, r_prod_source: _, r_prod_target: _,
        weighted: _, gate: _, out: _, token_id: _,
    } = p;
    let fe = fractional_fe(source_side) + fractional_fe(target_side)
        + virtual_fe(prod_source) + virtual_fe(prod_target);
    (fe, 0)
}

fn scatter_fe(p: &ScatterProof) -> usize {
    let ScatterProof { perm } = p;
    permutation_fe(perm)
}

/// `(variant name, field elements, u32 index count)` for one op proof.
///
/// EXHAUSTIVE with no wildcard: a new `OpProof` variant will not compile until
/// it is sized here. The payload-free variants contribute zero fe because they
/// have no proof of their own, not because they were skipped.
pub fn op_proof_fe(p: &OpProof) -> (&'static str, usize, usize) {
    match p {
        OpProof::Transpose => ("Transpose", 0, 0),
        OpProof::Scale => ("Scale", 0, 0),
        OpProof::ScaleVec => ("ScaleVec", 0, 0),
        OpProof::ScaleGate => ("ScaleGate", 0, 0),
        OpProof::Relu => ("Relu", 0, 0),
        OpProof::SoftmaxIndex => ("SoftmaxIndex", 0, 0),
        OpProof::GeluIndex => ("GeluIndex", 0, 0),
        OpProof::StableSoftmaxIndex => ("StableSoftmaxIndex", 0, 0),
        // HEAD: Lookup carries no payload; its logup proof is now a shard-level column.
        OpProof::Lookup => ("Lookup", 0, 0),
        OpProof::IdxBroadcast => ("IdxBroadcast", 0, 0),
        OpProof::CombineIdx => ("CombineIdx", 0, 0),
        OpProof::PadIdx => ("PadIdx", 0, 0),
        OpProof::BroadcastScale => ("BroadcastScale", 0, 0),
        OpProof::MatMul(mm, a, b, c) => ("MatMul", matmul_fe(mm) + a.len() + b.len() + c.len(), 0),
        OpProof::Projection(pp) => ("Projection", projection_fe(pp), 0),
        OpProof::Add(vp, r) => ("Add", virtual_fe(vp) + r.len(), 0),
        OpProof::Softmax(sp) => ("Softmax", softmax_fe(sp), 0),
        OpProof::Layernorm(lp) => ("Layernorm", layernorm_chain_fe(lp), 0),
        OpProof::LayerNormCentered(lp) => ("LayerNormCentered", layer_norm_centered_fe(lp), 0),
        OpProof::RmsNorm(rp) => ("RmsNorm", rms_norm_fe(rp), 0),
        OpProof::RoPE(rp) => ("RoPE", rope_fe(rp), 0),
        OpProof::TopKSelect(tp) => ("TopKSelect", topk_fe(tp), 0),
        OpProof::Argmax(ap) => {
            let (fe, idx) = argmax_fe(ap);
            ("Argmax", fe, idx)
        }
        OpProof::Scatter(sp) => ("Scatter", scatter_fe(sp), 0),
        OpProof::Gather(gp) => {
            let (fe, idx) = scatter_add_fe(gp);
            ("Gather", fe, idx)
        }
        OpProof::BatchedMatMul(bp, a, b, c) => ("BatchedMatMul", batched_matmul_fe(bp) + a.len() + b.len() + c.len(), 0),
        OpProof::GatherMatMul(gp, a, b, c) => ("GatherMatMul", gather_matmul_fe(gp) + a.len() + b.len() + c.len(), 0),
    }
}

/// `(field elements, tensor ids)` for one lookup witness column (HEAD).
/// `verify_lookup_split` reads only `.proof`; `x`/`y`/`num`/`point`/leaves are
/// the raw lookup column (witness) the verifier never reads. `table` is read for
/// matching (one tensor id).
fn witness_column_fe(c: &WitnessColumn) -> (usize, usize) {
    let WitnessColumn { proof, x: _, y: _, num: _, point: _, num_leaf: _, den_leaf: _, alpha: _, beta: _, table: _ } = c;
    (fractional_fe(proof), 1)
}

/// `(field elements, tensor ids)` for one lookup table column (HEAD).
/// As above: only `.proof` is verifier-read; `m` (multiplicities)/`point`/leaves
/// are witness. `table` is a tensor id.
fn table_column_fe(c: &TableColumn) -> (usize, usize) {
    let TableColumn { proof, table: _, m: _, alpha: _, beta: _, point: _, num_leaf: _, den_leaf: _ } = c;
    (fractional_fe(proof), 1)
}

fn add_shard(acc: &mut ProofSize, s: &OpShardProof) {
    let OpShardProof { ops, bound, binds, claims, witness_columns, table_columns } = s;
    for op in ops {
        let (name, fe, idx) = op_proof_fe(op);
        acc.op_proof_fe += fe;
        acc.tensor_ids += idx;
        *acc.fe_by_op.entry(name).or_insert(0) += fe;
        *acc.count_by_op.entry(name).or_insert(0) += 1;
    }
    acc.tensor_ids += bound.len();
    acc.shard_binds_fe += binds.iter().map(same_poly_fe).sum::<usize>();
    for (_t, pt, _eval) in claims {
        acc.tensor_ids += 1;
        acc.claims_fe += pt.len() + 1;
    }
    for wc in witness_columns {
        let (fe, idx) = witness_column_fe(wc);
        acc.witness_cols_fe += fe;
        acc.tensor_ids += idx;
        *acc.fe_by_op.entry("WitnessColumn").or_insert(0) += fe;
        *acc.count_by_op.entry("WitnessColumn").or_insert(0) += 1;
    }
    for tc in table_columns {
        let (fe, idx) = table_column_fe(tc);
        acc.table_cols_fe += fe;
        acc.tensor_ids += idx;
        *acc.fe_by_op.entry("TableColumn").or_insert(0) += fe;
        *acc.count_by_op.entry("TableColumn").or_insert(0) += 1;
    }
}

/// Walk a whole shard-DAG proof and report where its size is.
pub fn shard_dag_proof_size(p: &ShardDagProof) -> ProofSize {
    let ShardDagProof { shards, cross_tensors, cross_binds } = p;
    let mut acc = ProofSize { shards: shards.len(), ..Default::default() };
    for s in shards {
        add_shard(&mut acc, s);
    }
    acc.tensor_ids += cross_tensors.len();
    acc.cross_binds_fe += cross_binds.iter().map(same_poly_fe).sum::<usize>();
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkie_core::common::field::PrimeCharacteristicRing;

    /// The leaf counters must agree with a hand count on a known shape, so a
    /// refactor that changes what `virtual_fe` includes cannot pass silently.
    #[test]
    fn leaf_counters_match_a_hand_count() {
        let vp = VirtualProof {
            rounds: vec![vec![Goldilocks::ZERO; 3], vec![Goldilocks::ZERO; 3]],
            final_evals: vec![Goldilocks::ZERO; 4],
        };
        assert_eq!(virtual_fe(&vp), 3 + 3 + 4, "2 rounds of 3 plus 4 final evals");

        let sp = SamePolyProof {
            coeffs: vec![Goldilocks::ZERO; 5],
            proof: vp,
            merged_point: vec![Goldilocks::ZERO; 2],
            merged_eval: Goldilocks::ZERO,
        };
        assert_eq!(same_poly_fe(&sp), 5 + 10 + 2 + 1);
    }

    /// The payload-free variants must contribute zero field elements. If one
    /// gains a payload this stops compiling (the match destructures it).
    #[test]
    fn the_proofless_ops_contribute_zero() {
        let free = [
            OpProof::Transpose, OpProof::Scale, OpProof::ScaleVec, OpProof::ScaleGate,
            OpProof::Relu, OpProof::SoftmaxIndex, OpProof::GeluIndex, OpProof::StableSoftmaxIndex,
            OpProof::Lookup, OpProof::IdxBroadcast, OpProof::CombineIdx, OpProof::PadIdx,
            OpProof::BroadcastScale,
        ];
        for p in &free {
            let (_name, fe, idx) = op_proof_fe(p);
            assert_eq!(fe, 0, "a payload-free variant cannot carry field elements");
            assert_eq!(idx, 0, "a payload-free variant cannot carry indices");
        }
    }

    /// total_fe must equal the sum of its buckets, so a new bucket cannot be
    /// added to the struct and left out of the total.
    #[test]
    fn total_is_the_sum_of_the_buckets() {
        let s = ProofSize {
            op_proof_fe: 7, claims_fe: 11, shard_binds_fe: 13, cross_binds_fe: 17,
            witness_cols_fe: 19, table_cols_fe: 23, tensor_ids: 100, ..Default::default()
        };
        assert_eq!(s.total_fe(), 7 + 11 + 13 + 17 + 19 + 23);
        assert_eq!(s.total_bytes(), 90 * FE_BYTES + 100 * 4);
    }
}
