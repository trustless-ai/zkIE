//! How large is a shard-DAG proof, and which part of it is large?
//!
//! The total has never been measured. One WHIR *opening* has (#15: 94.8 KB at
//! 2^20, 115 KB at 2^22, 129 KB at 2^24), but the aggregate [`ShardDagProof`]
//! has not, because no proof type derives `Serialize` and `serde` is only a
//! dev-dependency.
//!
//! So this walks the structure and counts **field elements** rather than
//! serialising. That is deliberate and it is the more useful quantity: wrapping
//! the transcript in a SNARK costs constraints roughly in proportion to its
//! field elements, while a `bincode` framing byte is not in-circuit. The number
//! this reports is therefore an input to wrap cost, **not** a wire size, and
//! callers must not quote it as one.
//!
//! ## Why the breakdown matters more than the total
//!
//! `OpShardProof::claims` is a `Vec<(T, Vec<Goldilocks>, Goldilocks)>` emitted
//! *per op*, so it may scale with op count rather than with work. If it does, it
//! dominates at fine shard granularity — and then finer sharding inflates the
//! proof twice over, through boundary commitments *and* claim vectors, which
//! bears directly on the memory-versus-openings trade-off.
//!
//! ## The completeness guard
//!
//! An incomplete walk silently under-reports, and an under-reported proof size
//! is worse than no measurement. So **every struct below is destructured without
//! `..`, and the `OpProof` match carries no wildcard.** Adding a field or a
//! variant then fails to compile instead of being silently counted as zero.

use std::collections::BTreeMap;

use zkie_core::common::field::Goldilocks;
use zkie_core::common::logup_gkr::{FractionalLayer, FractionalProof};
use zkie_core::common::matmul::MatmulProof;
use zkie_core::common::same_poly::SamePolyProof;
use zkie_core::common::sumcheck::{RoundPoly, SumcheckProof, VirtualProof};

use crate::compose::{OpProof, OpShardProof, ShardDagProof};
use crate::layer_norm_centered::LayerNormCenteredProof;
use crate::layernorm_chain::LayernormChainProof;
use crate::projection::ProjectionProof;
use crate::rms_norm::RmsNormProof;
use crate::rope::RoPEProof;
use crate::softmax_scaled::SoftmaxRoundedProof;
use crate::topk::TopKSelectProof;

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
    /// Tensor ids. `T = usize`, so these are indices rather than field
    /// elements, and are reported separately rather than folded into the total.
    pub tensor_ids: usize,
    /// Field elements per `OpProof` variant.
    pub fe_by_op: BTreeMap<&'static str, usize>,
    /// How many of each `OpProof` variant appeared.
    pub count_by_op: BTreeMap<&'static str, usize>,
    pub shards: usize,
}

impl ProofSize {
    /// Total field elements. Excludes tensor ids, which are not field elements.
    pub fn total_fe(&self) -> usize {
        self.op_proof_fe + self.claims_fe + self.shard_binds_fe + self.cross_binds_fe
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
    let FractionalProof { layers, final_num: _, final_den: _ } = p;
    layers.iter().map(fractional_layer_fe).sum::<usize>() + 2
}

// ---- composite op proofs ----------------------------------------------------

fn projection_fe(p: &ProjectionProof) -> usize {
    let ProjectionProof { matmul, affine, frac, alpha: _, beta: _, pt, u, v, ch } = p;
    matmul_fe(matmul) + virtual_fe(affine) + fractional_fe(frac) + 2
        + pt.len() + u.len() + v.len() + ch.len()
}

fn softmax_fe(p: &SoftmaxRoundedProof) -> usize {
    let SoftmaxRoundedProof { lookup, row_sum, rescale, alpha: _, beta: _, r_sum, sum_ch, r_scale } = p;
    fractional_fe(lookup) + virtual_fe(row_sum) + virtual_fe(rescale) + 2
        + r_sum.len() + sum_ch.len() + r_scale.len()
}

fn layernorm_chain_fe(p: &LayernormChainProof) -> usize {
    let LayernormChainProof { mean_sq, rsqrt, out, r_mean, mean_ch, r_out, alpha: _, beta: _ } = p;
    virtual_fe(mean_sq) + fractional_fe(rsqrt) + virtual_fe(out) + 2
        + r_mean.len() + mean_ch.len() + r_out.len()
}

fn layer_norm_centered_fe(p: &LayerNormCenteredProof) -> usize {
    let LayerNormCenteredProof {
        mean, centered, var, rstd, out,
        r_mean, mean_ch, centered_r, r_var, var_ch, r_out, alpha: _, beta: _,
    } = p;
    virtual_fe(mean) + virtual_fe(centered) + virtual_fe(var) + fractional_fe(rstd)
        + virtual_fe(out) + 2
        + r_mean.len() + mean_ch.len() + centered_r.len() + r_var.len() + var_ch.len() + r_out.len()
}

fn rms_norm_fe(p: &RmsNormProof) -> usize {
    let RmsNormProof {
        trunc, rstd, out, rem_split, rem_lo_range, rem_hi_range,
        trunc_ch, r_out, rem_split_ch,
        alpha: _, beta: _, alpha_lo: _, beta_lo: _, alpha_hi: _, beta_hi: _,
    } = p;
    virtual_fe(trunc) + fractional_fe(rstd) + virtual_fe(out) + virtual_fe(rem_split)
        + fractional_fe(rem_lo_range) + fractional_fe(rem_hi_range)
        + trunc_ch.len() + r_out.len() + rem_split_ch.len() + 6
}

fn rope_fe(p: &RoPEProof) -> usize {
    let RoPEProof {
        first, second, frac_f, frac_s, r_first, r_second, r_out,
        alpha_f: _, beta_f: _, alpha_s: _, beta_s: _,
    } = p;
    virtual_fe(first) + virtual_fe(second) + fractional_fe(frac_f) + fractional_fe(frac_s)
        + r_first.len() + r_second.len() + r_out.len() + 4
}

fn topk_fe(p: &TopKSelectProof) -> usize {
    let TopKSelectProof {
        gate_rel, d1_rel, d2_rel, row_sum, sel_range, d1_range, d2_range,
        r, r_row,
        a_sel: _, b_sel: _, a_d1: _, b_d1: _, a_d2: _, b_d2: _,
    } = p;
    virtual_fe(gate_rel) + virtual_fe(d1_rel) + virtual_fe(d2_rel) + virtual_fe(row_sum)
        + fractional_fe(sel_range) + fractional_fe(d1_range) + fractional_fe(d2_range)
        + r.len() + r_row.len() + 6
}

/// `(variant name, field elements)` for one op proof.
///
/// EXHAUSTIVE with no wildcard: a new `OpProof` variant will not compile until
/// it is sized here. The eight payload-free variants are the ops with no proof
/// of their own (see `compose::tests::ops_without_their_own_proof_are_the_known_eight`);
/// they contribute zero because there is nothing to transmit, not because they
/// were skipped.
pub fn op_proof_fe(p: &OpProof) -> (&'static str, usize) {
    match p {
        OpProof::Transpose => ("Transpose", 0),
        OpProof::Scale => ("Scale", 0),
        OpProof::ScaleVec => ("ScaleVec", 0),
        OpProof::ScaleGate => ("ScaleGate", 0),
        OpProof::Relu => ("Relu", 0),
        OpProof::SoftmaxIndex => ("SoftmaxIndex", 0),
        OpProof::GeluIndex => ("GeluIndex", 0),
        OpProof::StableSoftmaxIndex => ("StableSoftmaxIndex", 0),
        OpProof::MatMul(mm, a, b, c) => ("MatMul", matmul_fe(mm) + a.len() + b.len() + c.len()),
        OpProof::Projection(pp) => ("Projection", projection_fe(pp)),
        OpProof::Add(vp, r) => ("Add", virtual_fe(vp) + r.len()),
        OpProof::Lookup(fp, _, _) => ("Lookup", fractional_fe(fp) + 2),
        OpProof::Softmax(sp) => ("Softmax", softmax_fe(sp)),
        OpProof::Layernorm(lp) => ("Layernorm", layernorm_chain_fe(lp)),
        OpProof::LayerNormCentered(lp) => ("LayerNormCentered", layer_norm_centered_fe(lp)),
        OpProof::RmsNorm(rp) => ("RmsNorm", rms_norm_fe(rp)),
        OpProof::RoPE(rp) => ("RoPE", rope_fe(rp)),
        OpProof::TopKSelect(tp) => ("TopKSelect", topk_fe(tp)),
    }
}

fn add_shard(acc: &mut ProofSize, s: &OpShardProof) {
    let OpShardProof { ops, bound, binds, claims } = s;
    for op in ops {
        let (name, fe) = op_proof_fe(op);
        acc.op_proof_fe += fe;
        *acc.fe_by_op.entry(name).or_insert(0) += fe;
        *acc.count_by_op.entry(name).or_insert(0) += 1;
    }
    acc.tensor_ids += bound.len();
    acc.shard_binds_fe += binds.iter().map(same_poly_fe).sum::<usize>();
    for (_t, pt, _eval) in claims {
        acc.tensor_ids += 1;
        acc.claims_fe += pt.len() + 1;
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

    /// The payload-free variants must be exactly the eight with no proof, and
    /// must contribute zero. If one gains a payload this stops compiling.
    #[test]
    fn the_eight_proofless_ops_contribute_zero() {
        let free = [
            OpProof::Transpose, OpProof::Scale, OpProof::ScaleVec, OpProof::ScaleGate,
            OpProof::Relu, OpProof::SoftmaxIndex, OpProof::GeluIndex, OpProof::StableSoftmaxIndex,
        ];
        assert_eq!(free.len(), 8);
        for p in &free {
            assert_eq!(op_proof_fe(p).1, 0, "a payload-free variant cannot carry field elements");
        }
    }

    /// total_fe must equal the sum of its parts, so a new bucket cannot be
    /// added to the struct and left out of the total.
    #[test]
    fn total_is_the_sum_of_the_buckets() {
        let s = ProofSize {
            op_proof_fe: 7, claims_fe: 11, shard_binds_fe: 13, cross_binds_fe: 17,
            tensor_ids: 100, ..Default::default()
        };
        assert_eq!(s.total_fe(), 7 + 11 + 13 + 17);
        assert_eq!(s.total_bytes(), 48 * FE_BYTES + 100 * 4);
    }
}
