//! Per-position (batched) matmul: `g[pos][j] = sum_d seg[pos][d] * w[pos][d][j]`,
//! where each position `pos` carries its own `K x N` weight matrix (already
//! gathered from the per-expert weights).
//!
//! The ordinary single-contraction sumcheck does not apply here: with a shared
//! weight the identity `C~(u,v) = sum_d A~(u,d) B~(d,v)` is multilinear in `u`
//! because `B~` does not depend on `u`; with a per-position weight both factors
//! depend on `u`, so that identity is degree-2 in `u` and is not the MLE of `g`.
//!
//! The sound reduction is an eq-selector virtual sumcheck over the `(S, K)`
//! domain: fix the output variable `j = v`, then
//!
//!   G~(u,v) = sum_{pos,d} seg[pos][d] * w[pos][d][v] * eq(pos, u).
//!
//! This is a single triple-product virtual sumcheck (domain `S*K`), and it
//! reduces to claims on `seg` (at the full challenge) and `w` (at `v || ch`),
//! plus the public `eq(pos, u)` selector.

use std::borrow::Cow;

use crate::common::field::{Ef, Goldilocks, PrimeCharacteristicRing};
use crate::common::ef_outer;
use crate::common::mle;
use crate::common::sumcheck::{prove_virtual, verify_virtual, virtual_eval, VirtualProof};
use crate::common::transcript::Transcript;
use crate::common::weights_io::WeightMmap;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct BatchedMatmulProof {
    pub claimed: Goldilocks,
    pub sumcheck: VirtualProof,
}

fn broadcast_eq(eq_u: &[Goldilocks], k: usize) -> Vec<Goldilocks> {
    let s = eq_u.len();
    let mut out = vec![Goldilocks::ZERO; s * k];
    for pos in 0..s {
        for d in 0..k {
            out[pos * k + d] = eq_u[pos];
        }
    }
    out
}

fn terms() -> Vec<(Goldilocks, Vec<usize>)> {
    vec![(Goldilocks::ONE, vec![0usize, 1, 2])]
}

/// Prove `g[pos][j] = sum_d seg[pos][d] * w[pos][d][j]` at the random MLE point
/// `(u, v)`.
///
/// Layouts (row-major): `seg` = `S x K`, `w` = `S x K x N`, `g` = `S x N`.
/// `u` fixes the `S` position variables and `v` the `N` output variables
/// (statement-bound, squeezed by the caller). S1: the `S + K` sumcheck
/// challenges `ch` over the `(pos, d)` domain are derived per-round from the
/// sumcheck's own round messages and returned. `S`, `K`, `N` must be powers of
/// two.
pub fn prove(
    seg: &[Goldilocks],
    w: &[Goldilocks],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    transcript: &mut Transcript,
) -> (BatchedMatmulProof, Vec<Goldilocks>) {
    assert_eq!(seg.len(), s * k);
    assert_eq!(w.len(), s * k * n);
    assert_eq!(g.len(), s * n);
    assert_eq!(u.len(), s.trailing_zeros() as usize);
    assert_eq!(v.len(), n.trailing_zeros() as usize);

    let w_v = mle::partial_eval(w, v); // S x K, index pos*K + d
    let eq_u = mle::eq_evals(u); // S
    let eq_b = broadcast_eq(&eq_u, k); // S x K

    let mut gp = v.to_vec();
    gp.extend_from_slice(u);
    let claimed = mle::eval(g, &gp);

    let mles: [&[Goldilocks]; 3] = [seg, &w_v, &eq_b];
    let (sumcheck, ch) = prove_virtual(&mles, &terms(), claimed, transcript);
    (BatchedMatmulProof { claimed, sumcheck }, ch)
}

/// Verify the sumcheck with the S1 transcript replay, recomputing the public
/// `eq(pos, u)` selector. Returns the derived `ch`. The `seg`/`w` endpoint
/// evals carried by the proof are bound to their commitments by the caller
/// (op layer), matching the ordinary-matmul convention.
pub fn verify(
    proof: &BatchedMatmulProof,
    s: usize,
    k: usize,
    u: &[Goldilocks],
    transcript: &mut Transcript,
) -> (bool, Vec<Goldilocks>) {
    // O4: for the row-major (pos*K + d) layout with power-of-two S and K, the
    // public broadcast selector eq_b[pos*K + d] = eq_u[pos] is independent of
    // the low d variables. Its MLE at the sumcheck challenge ch is therefore
    // eq_u evaluated at the high position variables ch[logK..]; the verifier
    // needs no S*K expansion. Non-power-of-two shapes fail closed: padding
    // must already be expressed by the caller's masked/zeroed domain, not by
    // truncating coordinates here.
    if !s.is_power_of_two() || !k.is_power_of_two() {
        return (false, Vec::new());
    }
    if u.len() != s.trailing_zeros() as usize {
        return (false, Vec::new());
    }
    let (ok, ch, final_claim) = verify_virtual(
        &proof.sumcheck,
        &terms(),
        proof.claimed,
        (s * k).trailing_zeros() as usize,
        transcript,
    );
    if !ok {
        return (false, ch);
    }
    let k_bits = k.trailing_zeros() as usize;
    if ch.len() != (s * k).trailing_zeros() as usize || ch.len() < k_bits {
        return (false, ch);
    }
    // Closed-form point evaluation of the public broadcast selector: the
    // verifier does not materialize eq_u or the S*K column. Padding is part of
    // the masked/zeroed power-of-two domain chosen by the caller; the
    // challenge coordinates are never truncated.
    let eq_final = mle::eq_poly(u, &ch[k_bits..]);
    let final_evals = vec![
        proof.sumcheck.final_evals[0],
        proof.sumcheck.final_evals[1],
        eq_final,
    ];
    (final_claim == virtual_eval(&terms(), &final_evals), ch)
}

/// Plain per-position matmul forward (also the test oracle).
pub fn forward(
    seg: &[Goldilocks],
    w: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
    let mut g = vec![Goldilocks::ZERO; s * n];
    for pos in 0..s {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for d in 0..k {
                acc = acc + seg[pos * k + d] * w[pos * k * n + d * n + j];
            }
            g[pos * n + j] = acc;
        }
    }
    g
}

/// `w_gathered[pos][d][j] = w_all[expert_of[pos]][d][j]`, with `w_all` laid out
/// expert-major as `n_experts x K x N`. Plain forward only: the *proof* of this
/// gather is the existing `Lookup` op (`out = table[idx]`), so it is not
/// duplicated here.
pub fn gather_weights(
    expert_of: &[u32],
    w_all: &[Goldilocks],
    s: usize,
    n_experts: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    let mut w = vec![Goldilocks::ZERO; s * k * n];
    for pos in 0..s {
        let base = expert_of[pos] as usize * k * n;
        for d in 0..k {
            for j in 0..n {
                w[pos * k * n + d * n + j] = w_all[base + d * n + j];
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    #[test]
    fn batched_matmul_roundtrip() {
        let mut rng = XorShift64::new(0xbeef);
        let (s, k, n, n_experts) = (4usize, 4usize, 2usize, 3usize);

        let mut w_all = vec![Goldilocks::ZERO; n_experts * k * n];
        for i in 0..w_all.len() {
            w_all[i] = Goldilocks::from_u64((i as u64 % 7) + 1);
        }
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|i| Goldilocks::from_u64((i as u64 % 11) + 1))
            .collect();

        let w = gather_weights(&expert_of, &w_all, s, n_experts, k, n);
        let g = forward(&seg, &w, s, k, n);

        // dense reference oracle
        let mut reference = vec![Goldilocks::ZERO; s * n];
        for pos in 0..s {
            let e = expert_of[pos] as usize;
            for j in 0..n {
                let mut acc = Goldilocks::ZERO;
                for d in 0..k {
                    acc = acc + seg[pos * k + d] * w_all[e * k * n + d * n + j];
                }
                reference[pos * n + j] = acc;
            }
        }
        assert_eq!(g, reference);

        let mut tp = Transcript::new(b"zkie-bmm-test");
        tp.absorb_u64(0xbeef);
        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let (proof, ch) = prove(&seg, &w, &g, s, k, n, &u, &v, &mut tp);

        let mut tv = Transcript::new(b"zkie-bmm-test");
        tv.absorb_u64(0xbeef);
        for _ in 0..s.trailing_zeros() {
            tv.field();
        }
        for _ in 0..n.trailing_zeros() {
            tv.field();
        }
        let (ok, ch_v) = verify(&proof, s, k, &u, &mut tv);
        assert!(ok);
        assert_eq!(ch_v, ch);

        // endpoint evals match the committed tensors at the expected points
        let w_v = mle::partial_eval(&w, &v);
        assert_eq!(proof.sumcheck.final_evals[0], mle::eval(&seg, &ch));
        assert_eq!(proof.sumcheck.final_evals[1], mle::eval(&w_v, &ch));

        // tamper the claimed output value -> reject
        let mut bad = proof;
        bad.claimed = bad.claimed + Goldilocks::ONE;
        let mut tv2 = Transcript::new(b"zkie-bmm-test");
        tv2.absorb_u64(0xbeef);
        for _ in 0..s.trailing_zeros() {
            tv2.field();
        }
        for _ in 0..n.trailing_zeros() {
            tv2.field();
        }
        let (ok2, _) = verify(&bad, s, k, &u, &mut tv2);
        assert!(!ok2);
    }
}


/// Stream the `partial_eval(W_gathered, v)` result from the per-expert weights,
/// without materializing `W_gathered = S x K x N`. This is the memory-friendly
/// replacement for `partial_eval(gather_weights(...), v)` used by the seq512 path.
pub fn gather_w_v(
    expert_of: &[u32],
    w_all: &[Goldilocks],
    s: usize,
    n_experts: usize,
    k: usize,
    n: usize,
    v: &[Goldilocks],
) -> Vec<Goldilocks> {
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    let eq_v = mle::eq_evals(v); // n elements
    let mut w_v = vec![Goldilocks::ZERO; s * k];
    for pos in 0..s {
        let base = expert_of[pos] as usize * k * n;
        for d in 0..k {
            let mut acc = Goldilocks::ZERO;
            for j in 0..n {
                acc = acc + w_all[base + d * n + j] * eq_v[j];
            }
            w_v[pos * k + d] = acc;
        }
    }
    w_v
}

/// Source of the expert-major weight `W_all` for streaming GatherMatMul
/// evaluation. Each expert block `W_all[e]` is `K x N` row-major
/// (`d*N + j`). An owned flat buffer is borrowed in place; an mmap-backed i32
/// tensor converts exactly one `K*N` block at a time, so the full weight is
/// never resident (O5 / streaming-gather design).
#[derive(Clone, Copy)]
pub enum WeightBlocks<'a> {
    Slice(&'a [Goldilocks]),
    Mmap { mmap: &'a WeightMmap, start: usize, len: usize },
}

impl<'a> WeightBlocks<'a> {
    /// Element count of the full expert-major weight (`n_experts * K * N`).
    pub fn len(&self) -> usize {
        match *self {
            WeightBlocks::Slice(s) => s.len(),
            WeightBlocks::Mmap { len, .. } => len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Expert block `e` as a `K*N` slice (mmap blocks are converted on demand
    /// and freed when the returned `Cow` is dropped).
    pub fn block(&self, e: usize, k: usize, n: usize) -> Cow<'a, [Goldilocks]> {
        let per = k * n;
        let lo = e * per;
        let hi = lo + per;
        assert!(hi <= self.len(), "expert weight block out of range");
        match *self {
            WeightBlocks::Slice(s) => Cow::Borrowed(&s[lo..hi]),
            WeightBlocks::Mmap { mmap, start, .. } => {
                Cow::Owned(mmap.read_goldilocks(start + lo, per))
            }
        }
    }
}

/// One streaming pass over the per-expert weight blocks.
pub struct StreamedWeights {
    /// `w_v[pos*K + d] = sum_j W_all[expert_of[pos]][d][j] * eq(j, v)`.
    pub w_v: Vec<Goldilocks>,
    /// `row_dots[e*K + d] = sum_j W_all[e][d][j] * eq(j, v)`.
    row_dots: Vec<Goldilocks>,
    experts_read: Vec<bool>,
    n_experts: usize,
    k: usize,
}

impl StreamedWeights {
    /// `table[e] = MLE(W_all[e], (ch_d, v))`, recovered from the row folds
    /// without touching the weight again. Returns `None` unless every expert
    /// block was read (i.e. the pass used `all_experts = true`) and the point
    /// arity matches the contraction dimension.
    pub fn table_evals(&self, ch_d: &[Goldilocks]) -> Option<Vec<Goldilocks>> {
        if ch_d.len() != self.k.trailing_zeros() as usize {
            return None;
        }
        if self.experts_read.iter().any(|&r| !r) {
            return None;
        }
        let eq_d = mle::eq_evals(ch_d);
        let mut out = Vec::with_capacity(self.n_experts);
        for e in 0..self.n_experts {
            let mut acc = Goldilocks::ZERO;
            for d in 0..self.k {
                acc = acc + eq_d[d] * self.row_dots[e * self.k + d];
            }
            out.push(acc);
        }
        Some(out)
    }
}

/// S9 EF version of [`StreamedWeights`]: `v` is an EF point, `row_dots` and
/// `w_v` are EF, and the per-expert table terminals are recovered at an EF
/// `ch_d` point.
pub struct StreamedWeightsEf {
    pub w_v: Vec<Ef>,
    row_dots: Vec<Ef>,
    experts_read: Vec<bool>,
    n_experts: usize,
    k: usize,
}

impl StreamedWeightsEf {
    pub fn table_evals(&self, ch_d: &[Ef]) -> Option<Vec<Ef>> {
        if ch_d.len() != self.k.trailing_zeros() as usize {
            return None;
        }
        if self.experts_read.iter().any(|&r| !r) {
            return None;
        }
        let eq_d = ef_outer::eq_evals(ch_d);
        let mut out = Vec::with_capacity(self.n_experts);
        for e in 0..self.n_experts {
            let mut acc = Ef::ZERO;
            for d in 0..self.k {
                acc = acc + eq_d[d] * self.row_dots[e * self.k + d];
            }
            out.push(acc);
        }
        Some(out)
    }
}

/// EF streaming pass: identical block traversal to [`stream_weights`], with
/// the `v` fold and all derived values in EF.
#[allow(clippy::too_many_arguments)]
pub fn stream_weights_ef(
    expert_of: &[u32],
    weights: &WeightBlocks<'_>,
    s: usize,
    n_experts: usize,
    k: usize,
    n: usize,
    v: &[Ef],
    all_experts: bool,
) -> StreamedWeightsEf {
    assert_eq!(expert_of.len(), s);
    assert_eq!(weights.len(), n_experts * k * n, "expert weight shape mismatch");
    assert_eq!(v.len(), n.trailing_zeros() as usize, "seq point arity mismatch");
    let eq_v = ef_outer::eq_evals(v);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
    for (pos, &e) in expert_of.iter().enumerate() {
        assert!((e as usize) < n_experts, "expert id out of range");
        buckets[e as usize].push(pos);
    }
    let mut w_v = vec![Ef::ZERO; s * k];
    let mut row_dots = vec![Ef::ZERO; n_experts * k];
    let mut experts_read = vec![false; n_experts];
    for e in 0..n_experts {
        if !all_experts && buckets[e].is_empty() {
            continue;
        }
        let block = weights.block(e, k, n);
        debug_assert_eq!(block.len(), k * n);
        for d in 0..k {
            let row = &block[d * n..(d + 1) * n];
            let mut acc = Ef::ZERO;
            for j in 0..n {
                acc = acc + Ef::from(row[j]) * eq_v[j];
            }
            row_dots[e * k + d] = acc;
        }
        for &pos in &buckets[e] {
            w_v[pos * k..(pos + 1) * k].copy_from_slice(&row_dots[e * k..(e + 1) * k]);
        }
        experts_read[e] = true;
    }
    StreamedWeightsEf { w_v, row_dots, experts_read, n_experts, k }
}

/// Fold each expert's `K x N` block against `eq(j, v)` once, then emit
/// `w_v[pos][d]` for every routed position (streaming, no `S x K x N` gather).
/// With `all_experts = true` every block is read so the per-expert `table`
/// terminals can be derived afterwards from the stored row folds; with
/// `all_experts = false` only experts that own a position are read (forward).
pub fn stream_weights(
    expert_of: &[u32],
    weights: &WeightBlocks<'_>,
    s: usize,
    n_experts: usize,
    k: usize,
    n: usize,
    v: &[Goldilocks],
    all_experts: bool,
) -> StreamedWeights {
    assert_eq!(expert_of.len(), s);
    assert_eq!(weights.len(), n_experts * k * n, "expert weight shape mismatch");
    assert_eq!(v.len(), n.trailing_zeros() as usize, "seq point arity mismatch");
    let eq_v = mle::eq_evals(v);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
    for (pos, &e) in expert_of.iter().enumerate() {
        assert!((e as usize) < n_experts, "expert id out of range");
        buckets[e as usize].push(pos);
    }
    let mut w_v = vec![Goldilocks::ZERO; s * k];
    let mut row_dots = vec![Goldilocks::ZERO; n_experts * k];
    let mut experts_read = vec![false; n_experts];
    for e in 0..n_experts {
        if !all_experts && buckets[e].is_empty() {
            continue;
        }
        let block = weights.block(e, k, n);
        debug_assert_eq!(block.len(), k * n);
        for d in 0..k {
            let row = &block[d * n..(d + 1) * n];
            let mut acc = Goldilocks::ZERO;
            for j in 0..n {
                acc = acc + row[j] * eq_v[j];
            }
            row_dots[e * k + d] = acc;
        }
        for &pos in &buckets[e] {
            w_v[pos * k..(pos + 1) * k].copy_from_slice(&row_dots[e * k..(e + 1) * k]);
        }
        experts_read[e] = true;
    }
    StreamedWeights {
        w_v,
        row_dots,
        experts_read,
        n_experts,
        k,
    }
}

/// Streaming per-position matmul forward from a weight-block source:
/// `g[pos][j] = sum_d seg[pos][d] * W_all[expert_of[pos]][d][j]`, reading one
/// `K x N` block per used expert (no `S x K x N` materialization). Identical
/// field arithmetic to [`gather_matmul_forward`].
pub fn gather_matmul_forward_streaming(
    seg: &[Goldilocks],
    expert_of: &[u32],
    weights: &WeightBlocks<'_>,
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
) -> Vec<Goldilocks> {
    assert_eq!(seg.len(), s * k);
    assert_eq!(expert_of.len(), s);
    assert_eq!(weights.len(), n_experts * k * n, "expert weight shape mismatch");
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
    for (pos, &e) in expert_of.iter().enumerate() {
        assert!((e as usize) < n_experts, "expert id out of range");
        buckets[e as usize].push(pos);
    }
    let mut g = vec![Goldilocks::ZERO; s * n];
    for e in 0..n_experts {
        if buckets[e].is_empty() {
            continue;
        }
        let block = weights.block(e, k, n);
        for &pos in &buckets[e] {
            let seg_row = &seg[pos * k..(pos + 1) * k];
            let dst = &mut g[pos * n..(pos + 1) * n];
            for j in 0..n {
                let mut acc = Goldilocks::ZERO;
                for d in 0..k {
                    acc = acc + seg_row[d] * block[d * n + j];
                }
                dst[j] = acc;
            }
        }
    }
    g
}

/// `counts[e] = sum_{pos: expert_of[pos]=e} eq(pos, ch_pos)`. Computed by the
/// verifier from the (already authenticated) `expert_of`.
pub fn expert_counts(expert_of: &[u32], n_experts: usize, ch_pos: &[Goldilocks]) -> Vec<Goldilocks> {
    let eq_pos = mle::eq_evals(ch_pos); // s elements
    let mut counts = vec![Goldilocks::ZERO; n_experts];
    for (pos, &e) in expert_of.iter().enumerate() {
        counts[e as usize] = counts[e as usize] + eq_pos[pos];
    }
    counts
}

/// `table[e] = MLE_eval(W_all[e], (ch_d, v))`, the MLE of the `K x N` block of
/// expert `e` at the point `(ch_d, v)`. These are the n openings of the committed
/// `W_all` (batchable).
pub fn expert_table_evals(
    w_all: &[Goldilocks],
    n_experts: usize,
    k: usize,
    n: usize,
    ch_d: &[Goldilocks],
    v: &[Goldilocks],
) -> Vec<Goldilocks> {
    assert_eq!(w_all.len(), n_experts * k * n);
    let mut point = v.to_vec();
    point.extend_from_slice(ch_d); // [j = v (LSB), d = ch_d (MSB)]
    let mut table = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let block = &w_all[e * k * n..(e + 1) * k * n];
        table.push(mle::eval(block, &point));
    }
    table
}

/// The linear-combination binding: `W~_gathered(ch_pos, ch_d, v) = sum_e counts[e] * table[e]`.
pub fn gather_open(counts: &[Goldilocks], table: &[Goldilocks]) -> Goldilocks {
    assert_eq!(counts.len(), table.len());
    let mut acc = Goldilocks::ZERO;
    for i in 0..counts.len() {
        acc = acc + counts[i] * table[i];
    }
    acc
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    #[test]
    fn gather_opening_linear_combination() {
        let mut rng = XorShift64::new(0x5151);
        let (s, k, n, n_experts) = (8usize, 4usize, 2usize, 4usize);

        let w_all: Vec<Goldilocks> = (0..n_experts * k * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1))
            .collect();
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();

        let ch_d: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
        // The prover and verifier both derive (u, v) from the same transcript
        // prefix, exactly like the op layer does.
        let mut tp = Transcript::new(b"zkie-bmm-mmap-test");
        tp.absorb_u64(0x8181);
        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let ch_pos: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| rng.field()).collect();

        // materialized reference
        let w_gathered = gather_weights(&expert_of, &w_all, s, n_experts, k, n);
        let w_v_ref = mle::partial_eval(&w_gathered, &v);

        // streaming
        let w_v = gather_w_v(&expert_of, &w_all, s, n_experts, k, n, &v);
        assert_eq!(w_v, w_v_ref);

        // weight claim == linear combination over experts
        let mut ch = ch_d.clone();
        ch.extend_from_slice(&ch_pos);
        let weight_claim = mle::eval(&w_v, &ch);

        let counts = expert_counts(&expert_of, n_experts, &ch_pos);
        let table = expert_table_evals(&w_all, n_experts, k, n, &ch_d, &v);
        let bound = gather_open(&counts, &table);

        assert_eq!(weight_claim, bound);
    }
}


/// Streaming per-position matmul forward: `g[pos][j] = sum_d seg[pos][d] * W_all[expert_of[pos]][d][j]`,
/// reading one `K x N` block of `W_all` at a time (no `S x K x N` materialization).
pub fn gather_matmul_forward(
    seg: &[Goldilocks],
    expert_of: &[u32],
    w_all: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
) -> Vec<Goldilocks> {
    assert_eq!(seg.len(), s * k);
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    let mut g = vec![Goldilocks::ZERO; s * n];
    for pos in 0..s {
        let base = expert_of[pos] as usize * k * n;
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for d in 0..k {
                acc = acc + seg[pos * k + d] * w_all[base + d * n + j];
            }
            g[pos * n + j] = acc;
        }
    }
    g
}

/// Streaming prove: same eq-selector sumcheck as `prove`, but `w_v` is computed on
/// the fly from the per-expert weights instead of a materialized `S x K x N` gather.
/// Returns the proof and the derived `ch`.
pub fn prove_streaming(
    seg: &[Goldilocks],
    expert_of: &[u32],
    w_all: &[Goldilocks],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    transcript: &mut Transcript,
) -> (BatchedMatmulProof, Vec<Goldilocks>) {
    assert_eq!(seg.len(), s * k);
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    assert_eq!(g.len(), s * n);

    let w_v = gather_w_v(expert_of, w_all, s, n_experts, k, n, v);
    prove_with_w_v(seg, &w_v, g, s, k, n, u, v, transcript)
}

/// Prove the eq-selector sumcheck from an already-computed weight fold `w_v`
/// (`S x K`). Shared by the materialized and the streaming gather variants, so
/// both produce the identical sumcheck/transcript for the same inputs.
pub fn prove_with_w_v(
    seg: &[Goldilocks],
    w_v: &[Goldilocks],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    transcript: &mut Transcript,
) -> (BatchedMatmulProof, Vec<Goldilocks>) {
    assert_eq!(seg.len(), s * k);
    assert_eq!(w_v.len(), s * k);
    assert_eq!(g.len(), s * n);

    let eq_u = mle::eq_evals(u);
    let eq_b = broadcast_eq(&eq_u, k);

    let mut gp = v.to_vec();
    gp.extend_from_slice(u);
    let claimed = mle::eval(g, &gp);

    let mles: [&[Goldilocks]; 3] = [seg, w_v, &eq_b];
    let (sumcheck, ch) = prove_virtual(&mles, &terms(), claimed, transcript);
    (BatchedMatmulProof { claimed, sumcheck }, ch)
}

/// Streaming verify: the ordinary eq-selector sumcheck (S1 transcript replay),
/// plus the weight claim bound by the linear combination
/// `final_evals[1] == sum_e counts[e] * table[e]`.
pub fn verify_streaming(
    proof: &BatchedMatmulProof,
    expert_of: &[u32],
    w_all: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    transcript: &mut Transcript,
) -> (bool, Vec<Goldilocks>) {
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    let (ok, ch) = verify(proof, s, k, u, transcript);
    if !ok {
        return (false, ch);
    }
    let k_bits = k.trailing_zeros() as usize;
    let ch_d = &ch[..k_bits];
    let ch_pos = &ch[k_bits..];
    let counts = expert_counts(expert_of, n_experts, ch_pos);
    let table = expert_table_evals(w_all, n_experts, k, n, ch_d, v);
    (proof.sumcheck.final_evals[1] == gather_open(&counts, &table), ch)
}

#[cfg(test)]
mod streaming_matmul_tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    #[test]
    fn streaming_matches_materialized() {
        let mut rng = XorShift64::new(0x6161);
        let (s, k, n, n_experts) = (8usize, 4usize, 2usize, 4usize);

        let w_all: Vec<Goldilocks> = (0..n_experts * k * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1))
            .collect();
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|i| Goldilocks::from_u64((i as u64 % 11) + 1))
            .collect();

        // streaming forward == materialized forward
        let g_stream = gather_matmul_forward(&seg, &expert_of, &w_all, s, k, n, n_experts);
        let w_gathered = gather_weights(&expert_of, &w_all, s, n_experts, k, n);
        let g_mat = forward(&seg, &w_gathered, s, k, n);
        assert_eq!(g_stream, g_mat);

        let mut tp = Transcript::new(b"zkie-bmm-stream-test");
        tp.absorb_u64(0x6161);
        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let (proof, ch) =
            prove_streaming(&seg, &expert_of, &w_all, &g_stream, s, k, n, n_experts, &u, &v, &mut tp);

        let mut tv = Transcript::new(b"zkie-bmm-stream-test");
        tv.absorb_u64(0x6161);
        for _ in 0..s.trailing_zeros() {
            tv.field();
        }
        for _ in 0..n.trailing_zeros() {
            tv.field();
        }
        let (ok, ch_v) = verify_streaming(&proof, &expert_of, &w_all, s, k, n, n_experts, &u, &v, &mut tv);
        assert!(ok);
        assert_eq!(ch_v, ch);

        // tamper the weight claim -> reject
        let mut bad = proof;
        bad.sumcheck.final_evals[1] = bad.sumcheck.final_evals[1] + Goldilocks::ONE;
        let mut tv2 = Transcript::new(b"zkie-bmm-stream-test");
        tv2.absorb_u64(0x6161);
        for _ in 0..s.trailing_zeros() {
            tv2.field();
        }
        for _ in 0..n.trailing_zeros() {
            tv2.field();
        }
        let (ok2, _) = verify_streaming(&bad, &expert_of, &w_all, s, k, n, n_experts, &u, &v, &mut tv2);
        assert!(!ok2);
    }

    /// O5/B: the same block stream must produce the same `w_v`, per-expert
    /// terminals and forward output from (a) an owned slice and (b) an
    /// mmap-backed i32 weight, and the streaming gather proof must verify
    /// against the same verifier.
    #[test]
    fn streaming_from_mmap_matches_slice() {
        use crate::common::fixed_point::from_i32;
        use std::io::Write;
        let mut rng = XorShift64::new(0x8181);
        let (s, k, n, n_experts) = (8usize, 4usize, 2usize, 4usize);
        let raw: Vec<i32> = (0..n_experts * k * n).map(|i| (i as i32 % 7) - 3).collect();
        let w_all: Vec<Goldilocks> = raw.iter().map(|&v| from_i32(v)).collect();
        let experts: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|i| Goldilocks::from_u64((i as u64 % 11) + 1))
            .collect();
        let ch_d: Vec<Goldilocks> = (0..k.trailing_zeros() as usize).map(|_| rng.field()).collect();
        // The prover and verifier both derive (u, v) from the same transcript
        // prefix, exactly like the op layer does.
        let mut tp = Transcript::new(b"zkie-bmm-mmap-test");
        tp.absorb_u64(0x8181);
        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| tp.field()).collect();

        // mmap-backed source
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("zkie-bmm-mmap-{}-{stamp}.bin", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            for x in &raw {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        let mmap = WeightMmap::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let from_mmap = WeightBlocks::Mmap { mmap: &mmap, start: 0, len: raw.len() };
        let from_slice = WeightBlocks::Slice(&w_all);

        // Forward: identical output.
        let g_mmap = gather_matmul_forward_streaming(&seg, &experts, &from_mmap, s, k, n, n_experts);
        let g_slice = gather_matmul_forward_streaming(&seg, &experts, &from_slice, s, k, n, n_experts);
        assert_eq!(g_mmap, g_slice);
        let w_gathered = gather_weights(&experts, &w_all, s, n_experts, k, n);
        assert_eq!(g_slice, forward(&seg, &w_gathered, s, k, n));

        // Weight stream: w_v and every expert terminal (all experts read).
        let sm = stream_weights(&experts, &from_mmap, s, n_experts, k, n, &v, true);
        let ss = stream_weights(&experts, &from_slice, s, n_experts, k, n, &v, true);
        assert_eq!(sm.w_v, ss.w_v);
        assert_eq!(sm.w_v, gather_w_v(&experts, &w_all, s, n_experts, k, n, &v));
        assert_eq!(
            sm.table_evals(&ch_d).unwrap(),
            expert_table_evals(&w_all, n_experts, k, n, &ch_d, &v)
        );

        // Streaming gather proof from the mmap source verifies with the same
        // verifier as the materialized path.
        let (proof, ch) =
            prove_gather_matmul_streaming(&seg, &experts, &from_mmap, &g_mmap, s, k, n, n_experts, &u, &v, &mut tp);
        let mut tv = Transcript::new(b"zkie-bmm-mmap-test");
        tv.absorb_u64(0x8181);
        for _ in 0..s.trailing_zeros() {
            tv.field();
        }
        for _ in 0..n.trailing_zeros() {
            tv.field();
        }
        let (ok, ch_v) = verify_gather_matmul(&proof, s, k, &u, &mut tv);
        assert!(ok);
        assert_eq!(ch_v, ch);
        // Tampered terminal -> reject.
        let mut bad = proof;
        bad.table[0] = bad.table[0] + Goldilocks::ONE;
        let mut tv2 = Transcript::new(b"zkie-bmm-mmap-test");
        tv2.absorb_u64(0x8181);
        for _ in 0..s.trailing_zeros() { tv2.field(); }
        for _ in 0..n.trailing_zeros() { tv2.field(); }
        assert!(!verify_gather_matmul(&bad, s, k, &u, &mut tv2).0);
    }
}


/// Self-contained streaming gather-matmul proof: the eq-selector sumcheck plus the
/// linear-combination terms that bind the weight claim to the per-expert weights.
/// `table[e] = MLE_eval(W_all[e], (ch_d, v))` is opened from the committed `W_all`
/// on the committed path; `counts[e] = sum_{pos->e} eq(pos, ch_pos)` is recomputed
/// from the authenticated `expert_of`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GatherMatMulProof {
    pub claimed: Goldilocks,
    pub sumcheck: VirtualProof,
    pub counts: Vec<Goldilocks>,
    pub table: Vec<Goldilocks>,
}

pub fn prove_gather_matmul(
    seg: &[Goldilocks],
    expert_of: &[u32],
    w_all: &[Goldilocks],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    transcript: &mut Transcript,
) -> (GatherMatMulProof, Vec<Goldilocks>) {
    let (matmul, ch) = prove_streaming(seg, expert_of, w_all, g, s, k, n, n_experts, u, v, transcript);
    let k_bits = k.trailing_zeros() as usize;
    let ch_d = &ch[..k_bits];
    let ch_pos = &ch[k_bits..];
    let counts = expert_counts(expert_of, n_experts, ch_pos);
    let table = expert_table_evals(w_all, n_experts, k, n, ch_d, v);
    (
        GatherMatMulProof {
            claimed: matmul.claimed,
            sumcheck: matmul.sumcheck,
            counts,
            table,
        },
        ch,
    )
}

/// Streaming variant of [`prove_gather_matmul`]: the same eq-selector
/// sumcheck, with `w_v` and the per-expert terminals produced by one streaming
/// pass over the weight blocks (`WeightBlocks`), never materializing
/// `W_gathered` or the full flat `W_all`.
#[allow(clippy::too_many_arguments)]
pub fn prove_gather_matmul_streaming(
    seg: &[Goldilocks],
    expert_of: &[u32],
    weights: &WeightBlocks<'_>,
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    n_experts: usize,
    u: &[Goldilocks],
    v: &[Goldilocks],
    transcript: &mut Transcript,
) -> (GatherMatMulProof, Vec<Goldilocks>) {
    assert_eq!(g.len(), s * n);
    let streamed = stream_weights(expert_of, weights, s, n_experts, k, n, v, true);
    let (matmul, ch) = prove_with_w_v(seg, &streamed.w_v, g, s, k, n, u, v, transcript);
    let k_bits = k.trailing_zeros() as usize;
    let ch_d = &ch[..k_bits];
    let ch_pos = &ch[k_bits..];
    let counts = expert_counts(expert_of, n_experts, ch_pos);
    let table = streamed
        .table_evals(ch_d)
        .expect("streaming pass read every expert block");
    (
        GatherMatMulProof {
            claimed: matmul.claimed,
            sumcheck: matmul.sumcheck,
            counts,
            table,
        },
        ch,
    )
}

pub fn verify_gather_matmul(
    proof: &GatherMatMulProof,
    s: usize,
    k: usize,
    u: &[Goldilocks],
    transcript: &mut Transcript,
) -> (bool, Vec<Goldilocks>) {
    let (ok, ch) = verify(
        &BatchedMatmulProof {
            claimed: proof.claimed,
            sumcheck: proof.sumcheck.clone(),
        },
        s,
        k,
        u,
        transcript,
    );
    if !ok {
        return (false, ch);
    }
    (proof.sumcheck.final_evals[1] == gather_open(&proof.counts, &proof.table), ch)
}

#[cfg(test)]
mod gather_matmul_proof_tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    #[test]
    fn gather_verify_uses_closed_form_broadcast_eq() {
        let mut rng = XorShift64::new(0x7172);
        let (s, k, n, n_experts) = (8usize, 4usize, 2usize, 4usize);
        let w_all: Vec<Goldilocks> = (0..n_experts * k * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1)).collect();
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|i| Goldilocks::from_u64((i as u64 % 11) + 1)).collect();
        let g = gather_matmul_forward(&seg, &expert_of, &w_all, s, k, n, n_experts);
        let mut tp = Transcript::new(b"o4-gather");
        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize)
            .map(|_| rng.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize)
            .map(|_| rng.field()).collect();
        let (proof, ch) = prove_gather_matmul(
            &seg, &expert_of, &w_all, &g, s, k, n, n_experts, &u, &v, &mut tp);
        let mut tv = Transcript::new(b"o4-gather");
        let (ok, ch_v) = verify_gather_matmul(&proof, s, k, &u, &mut tv);
        assert!(ok);
        assert_eq!(ch_v, ch);
        let k_bits = k.trailing_zeros() as usize;
        let eq_u = mle::eq_evals(&u);
        let full = broadcast_eq(&eq_u, k);
        assert_eq!(mle::eval(&full, &ch), mle::eq_poly(&u, &ch[k_bits..]));
    }

    #[test]
    fn gather_matmul_self_contained() {
        let mut rng = XorShift64::new(0x7171);
        let (s, k, n, n_experts) = (8usize, 4usize, 2usize, 4usize);
        let w_all: Vec<Goldilocks> = (0..n_experts * k * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1))
            .collect();
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|i| Goldilocks::from_u64((i as u64 % 11) + 1))
            .collect();
        let g = gather_matmul_forward(&seg, &expert_of, &w_all, s, k, n, n_experts);

        let mut tp = Transcript::new(b"zkie-bmm-gather-test");
        tp.absorb_u64(0x7171);
        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| tp.field()).collect();
        let (proof, ch) =
            prove_gather_matmul(&seg, &expert_of, &w_all, &g, s, k, n, n_experts, &u, &v, &mut tp);

        let mut tv = Transcript::new(b"zkie-bmm-gather-test");
        tv.absorb_u64(0x7171);
        for _ in 0..s.trailing_zeros() {
            tv.field();
        }
        for _ in 0..n.trailing_zeros() {
            tv.field();
        }
        let (ok, ch_v) = verify_gather_matmul(&proof, s, k, &u, &mut tv);
        assert!(ok);
        assert_eq!(ch_v, ch);

        // tamper the weight claim (final_evals[1]) -> reject
        let mut bad = proof;
        bad.sumcheck.final_evals[1] = bad.sumcheck.final_evals[1] + Goldilocks::ONE;
        let mut tv2 = Transcript::new(b"zkie-bmm-gather-test");
        tv2.absorb_u64(0x7171);
        for _ in 0..s.trailing_zeros() {
            tv2.field();
        }
        for _ in 0..n.trailing_zeros() {
            tv2.field();
        }
        let (ok2, _) = verify_gather_matmul(&bad, s, k, &u, &mut tv2);
        assert!(!ok2);
    }
}

#[cfg(test)]
mod o4_tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    /// O4: the closed form eq_u(ch[logK..]) must equal the full S*K broadcast
    /// column evaluated at ch, for several power-of-two shapes and random
    /// points (including K=1 and S=1).
    #[test]
    fn broadcast_eq_closed_form_matches_expansion() {
        let mut rng = XorShift64::new(0x04A4);
        for (s, k) in [(1usize, 1usize), (2, 4), (8, 16), (32, 64), (4, 1), (1, 8)] {
            let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize)
                .map(|_| rng.field())
                .collect();
            let sk_bits = (s * k).trailing_zeros() as usize;
            let ch: Vec<Goldilocks> = (0..sk_bits).map(|_| rng.field()).collect();
            let eq_u = mle::eq_evals(&u);
            let full = broadcast_eq(&eq_u, k);
            let full_eval = mle::eval(&full, &ch);
            let k_bits = k.trailing_zeros() as usize;
            let closed = mle::eq_poly(&u, &ch[k_bits..]);
            assert_eq!(full_eval, closed, "closed form mismatch for s={s}, k={k}");
        }
    }

    /// A masked/zero-padded power-of-two domain: six real positions are
    /// embedded in S=8 with zero rows, exactly as the op layer pads S. The
    /// closed form uses all three position bits; it must not truncate the
    /// challenge to the six-real-bit prefix.
    #[test]
    fn closed_form_handles_zero_padded_rows() {
        use crate::common::field::ChallengeSource;
        let mut rng = XorShift64::new(0x04A5);
        let (s_pad, actual, k, n) = (8usize, 6usize, 4usize, 2usize);
        let mut seg = vec![Goldilocks::ZERO; s_pad * k];
        let mut w = vec![Goldilocks::ZERO; s_pad * k * n];
        for pos in 0..actual {
            for d in 0..k {
                seg[pos * k + d] = rng.field();
                for j in 0..n {
                    w[(pos * k + d) * n + j] = rng.field();
                }
            }
        }
        let g = forward(&seg, &w, s_pad, k, n);
        let u: Vec<Goldilocks> = (0..s_pad.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let (proof, ch) = prove(&seg, &w, &g, s_pad, k, n, &u, &v, &mut Transcript::new(b"o4-pad"));
        let (ok, ch_v) = verify(&proof, s_pad, k, &u, &mut Transcript::new(b"o4-pad"));
        assert!(ok, "zero-padded domain must verify");
        assert_eq!(ch_v, ch);
        let eq_u = mle::eq_evals(&u);
        let full = broadcast_eq(&eq_u, k);
        let k_bits = k.trailing_zeros() as usize;
        assert_eq!(mle::eval(&full, &ch), mle::eq_poly(&u, &ch[k_bits..]));
        // The six-real-bit prefix (mechanically truncated coordinates) is not
        // the selector used by the padded proof.
        let truncated = mle::eq_poly(&u[..actual.trailing_zeros() as usize], &ch[k_bits..][..actual.trailing_zeros() as usize]);
        let pad = mle::eq_poly(&u[actual.trailing_zeros() as usize..], &ch[k_bits + actual.trailing_zeros() as usize..]);
        assert_eq!(mle::eq_poly(&u, &ch[k_bits..]), truncated * pad,
            "padded closed form must factor through the extra valid bits");
    }

    /// Non-power-of-two layouts are outside the closed-form contract and must
    /// fail closed rather than truncating coordinates.
    #[test]
    fn verify_rejects_non_power_of_two_shapes() {
        let proof = BatchedMatmulProof {
            claimed: Goldilocks::ZERO,
            sumcheck: VirtualProof { rounds: Vec::new(), final_evals: Vec::new() },
        };
        let mut t = Transcript::new(b"o4-shape");
        let (ok, _) = verify(&proof, 3, 4, &[], &mut t);
        assert!(!ok, "s=3 must fail closed");
        let mut t = Transcript::new(b"o4-shape");
        let (ok, _) = verify(&proof, 4, 3, &[], &mut t);
        assert!(!ok, "k=3 must fail closed");
    }
}

