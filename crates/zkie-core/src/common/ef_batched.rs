//! S9: per-position (batched) MatMul reduction over the quadratic extension
//! field. Mirrors `common::batched_matmul`; tensors stay base-field, the
//! output-column point, eq selector, claimed value and sum-check messages are
//! EF.

use crate::common::ef_outer::{self as efo, EfVirtualProof};
use crate::common::field::{Ef, Goldilocks, PrimeCharacteristicRing};
use crate::common::transcript::Transcript;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct EfBatchedMatmulProof {
    pub claimed: Ef,
    pub sumcheck: EfVirtualProof,
}

fn terms() -> Vec<(Ef, Vec<usize>)> {
    vec![(Ef::ONE, vec![0usize, 1, 2])]
}

fn broadcast_eq(eq_u: &[Ef], k: usize) -> Vec<Ef> {
    let s = eq_u.len();
    let mut out = vec![Ef::ZERO; s * k];
    for pos in 0..s {
        for d in 0..k {
            out[pos * k + d] = eq_u[pos];
        }
    }
    out
}

/// Prove `g[pos][j] = sum_d seg[pos][d]·w[pos][d][j]` at EF point `(u,v)`.
#[allow(clippy::too_many_arguments)]
pub fn prove(
    seg: &[Goldilocks],
    w: &[Goldilocks],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    u: &[Ef],
    v: &[Ef],
    transcript: &mut Transcript,
) -> (EfBatchedMatmulProof, Vec<Ef>) {
    assert_eq!(seg.len(), s * k);
    assert_eq!(w.len(), s * k * n);
    assert_eq!(g.len(), s * n);
    assert_eq!(u.len(), s.trailing_zeros() as usize);
    assert_eq!(v.len(), n.trailing_zeros() as usize);
    let w_v = efo::partial_eval_at_ef(w, v).expect("W restriction");
    let eq_u = efo::eq_evals(u);
    let eq_b = broadcast_eq(&eq_u, k);
    let mut gp = v.to_vec();
    gp.extend_from_slice(u);
    let claimed = efo::mle_eval_at_ef(g, &gp).expect("G evaluation");
    let seg_ef: Vec<Ef> = seg.iter().map(|&x| Ef::from(x)).collect();
    let mles: [&[Ef]; 3] = [seg_ef.as_slice(), &w_v, &eq_b];
    let (sumcheck, ch) = efo::prove_virtual(&mles, &terms(), claimed, transcript);
    (EfBatchedMatmulProof { claimed, sumcheck }, ch)
}

/// Prove using the per-position folded weight matrix `w_v` from the streaming
/// pass (`S x K`, EF) instead of materializing/re-folding the full `S x K x N`
/// weight tensor.
#[allow(clippy::too_many_arguments)]
pub fn prove_with_w_v(
    seg: &[Goldilocks],
    w_v: &[Ef],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    u: &[Ef],
    v: &[Ef],
    transcript: &mut Transcript,
) -> (EfBatchedMatmulProof, Vec<Ef>) {
    assert_eq!(seg.len(), s * k);
    assert_eq!(w_v.len(), s * k);
    assert_eq!(g.len(), s * n);
    assert_eq!(u.len(), s.trailing_zeros() as usize);
    assert_eq!(v.len(), n.trailing_zeros() as usize);
    let eq_u = efo::eq_evals(u);
    let eq_b = broadcast_eq(&eq_u, k);
    let mut gp = v.to_vec();
    gp.extend_from_slice(u);
    let claimed = efo::mle_eval_at_ef(g, &gp).expect("G evaluation");
    let seg_ef: Vec<Ef> = seg.iter().map(|&x| Ef::from(x)).collect();
    let mles: [&[Ef]; 3] = [seg_ef.as_slice(), w_v, &eq_b];
    let (sumcheck, ch) = efo::prove_virtual(&mles, &terms(), claimed, transcript);
    (EfBatchedMatmulProof { claimed, sumcheck }, ch)
}

/// Verify the EF batched-matmul sum-check. Recomputes the public eq selector
/// at the derived challenge vector (closed form, no S·K expansion).
pub fn verify(
    proof: &EfBatchedMatmulProof,
    s: usize,
    k: usize,
    u: &[Ef],
    transcript: &mut Transcript,
) -> (bool, Vec<Ef>) {
    if !s.is_power_of_two() || !k.is_power_of_two() {
        return (false, Vec::new());
    }
    if u.len() != s.trailing_zeros() as usize {
        return (false, Vec::new());
    }
    let eq_u = efo::eq_evals(u);
    let (ok, ch, final_claim) = efo::verify_virtual(
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
    let Some(eq_final) = efo::eval_ef_slice(&eq_u, &ch[k_bits..]) else {
        return (false, ch);
    };
    if proof.sumcheck.final_evals.len() < 3
        || proof.sumcheck.final_evals[2] != eq_final
    {
        return (false, ch);
    }
    let fe = &proof.sumcheck.final_evals;
    if final_claim != efo::virtual_eval(&terms(), &[fe[0], fe[1], eq_final]) {
        return (false, ch);
    }
    (true, ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::XorShift64;

    #[test]
    fn batched_matmul_ef_roundtrip_and_tamper() {
        let (s, k, n) = (4usize, 4usize, 4usize);
        let mut rng = XorShift64::new(0x9501);
        let seg: Vec<Goldilocks> = (0..s * k).map(|_| rng.field()).collect();
        let w: Vec<Goldilocks> = (0..s * k * n).map(|_| rng.field()).collect();
        let mut g = vec![Goldilocks::ZERO; s * n];
        for pos in 0..s {
            for j in 0..n {
                let mut acc = Goldilocks::ZERO;
                for d in 0..k {
                    acc += seg[pos * k + d] * w[(pos * k + d) * n + j];
                }
                g[pos * n + j] = acc;
            }
        }
        let u: Vec<Ef> = (0..2).map(|_| Ef::new([rng.field(), rng.field()])).collect();
        let v: Vec<Ef> = (0..2).map(|_| Ef::new([rng.field(), rng.field()])).collect();
        let mut tp = Transcript::new(b"ef-bmm");
        let (proof, ch) = prove(&seg, &w, &g, s, k, n, &u, &v, &mut tp);
        let mut tv = Transcript::new(b"ef-bmm");
        let (ok, ch2) = verify(&proof, s, k, &u, &mut tv);
        assert!(ok);
        assert_eq!(ch, ch2);
        let mut bad = proof.clone();
        bad.sumcheck.final_evals[0] += Ef::ONE;
        let mut tv = Transcript::new(b"ef-bmm");
        assert!(!verify(&bad, s, k, &u, &mut tv).0);
    }
}
