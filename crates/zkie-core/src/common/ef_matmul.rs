//! S9: MatMul sum-check over the quadratic extension field.
//!
//! The reduction is identical to `common::matmul`, but the restricted
//! tensors, claimed value, round messages, and terminal evaluations live in
//! EF. Tensors and public execution values remain base-field; the EF point
//! restriction is `ef_outer::partial_eval_at_ef`.

use crate::common::ef_outer::{self, EfSumcheckProof, EfRoundPoly};
use crate::common::field::{Ef, Goldilocks};
use crate::common::transcript::Transcript;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct EfMatmulProof {
    pub claimed: Ef,
    pub sumcheck: EfSumcheckProof,
}

/// Prove `C = A @ B` at EF claim points `u` (rows of A^T) and `v` (cols of B).
/// `at` is A stored transposed row-major (`k x m`); `b` is `k x n`;
/// `c` is `m x n`.
pub fn prove(
    at: &[Goldilocks],
    b: &[Goldilocks],
    c: &[Goldilocks],
    m: usize,
    k: usize,
    n: usize,
    u: &[Ef],
    v: &[Ef],
    transcript: &mut Transcript,
) -> (EfMatmulProof, Vec<Ef>) {
    assert_eq!(at.len(), k * m);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    assert_eq!(u.len(), m.trailing_zeros() as usize);
    assert_eq!(v.len(), n.trailing_zeros() as usize);
    let a_restricted = ef_outer::partial_eval_at_ef(at, u).expect("A restriction");
    let b_restricted = ef_outer::partial_eval_at_ef(b, v).expect("B restriction");
    debug_assert_eq!(a_restricted.len(), k);
    debug_assert_eq!(b_restricted.len(), k);
    let mut c_point = v.to_vec();
    c_point.extend_from_slice(u);
    let claimed = ef_outer::mle_eval_at_ef(c, &c_point).expect("C evaluation");
    let (sumcheck, challenges) =
        ef_outer::prove_sumcheck_ef(&a_restricted, &b_restricted, claimed, transcript);
    (EfMatmulProof { claimed, sumcheck }, challenges)
}

/// Verify the MatMul sum-check. `num_rounds` is the statement-derived
/// `k.trailing_zeros()`.
pub fn verify(
    proof: &EfMatmulProof,
    num_rounds: usize,
    transcript: &mut Transcript,
) -> (bool, Vec<Ef>) {
    ef_outer::verify_sumcheck(&proof.sumcheck, proof.claimed, num_rounds, transcript)
}

/// Convenience: the degree-2 round polynomial type re-exported for callers
/// that carry MatMul proofs alongside virtual proofs.
pub type RoundPoly = EfRoundPoly;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::{PrimeCharacteristicRing, XorShift64};

    #[test]
    fn matmul_ef_roundtrip_and_tamper() {
        let (m, k, n) = (4usize, 8usize, 4usize);
        let mut rng = XorShift64::new(0x9101);
        let at: Vec<Goldilocks> = (0..k * m).map(|_| rng.field()).collect();
        let b: Vec<Goldilocks> = (0..k * n).map(|_| rng.field()).collect();
        let mut c = vec![Goldilocks::ZERO; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = Goldilocks::ZERO;
                for kk in 0..k {
                    acc += at[kk * m + i] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        let u: Vec<Ef> = (0..2).map(|_| Ef::new([rng.field(), rng.field()])).collect();
        let v: Vec<Ef> = (0..2).map(|_| Ef::new([rng.field(), rng.field()])).collect();
        let mut tp = Transcript::new(b"ef-matmul");
        let (proof, ch) = prove(&at, &b, &c, m, k, n, &u, &v, &mut tp);
        let mut tv = Transcript::new(b"ef-matmul");
        let (ok, ch2) = verify(&proof, k.trailing_zeros() as usize, &mut tv);
        assert!(ok);
        assert_eq!(ch, ch2);
        let mut bad = proof.clone();
        bad.sumcheck.rounds[0].c0 += Ef::ONE;
        let mut tv = Transcript::new(b"ef-matmul");
        assert!(!verify(&bad, k.trailing_zeros() as usize, &mut tv).0);
    }
}
