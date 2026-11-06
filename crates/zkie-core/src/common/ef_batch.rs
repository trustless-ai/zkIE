//! S9: sum-of-products sum-check over the quadratic extension field.
//!
//! This is the arithmetic half of the multi-point batch-opening `W_t`
//! reduction. The WHIR opening itself is unchanged at the PCS boundary; the
//! EF batch-opening wrapper is added when the composer migrates.

use crate::common::ef_outer::EfRoundPoly;
use crate::common::field::{Ef, Field, Goldilocks, PrimeCharacteristicRing};
use crate::common::transcript::Transcript;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EfSumcheckProofBatch {
    pub rounds: Vec<EfRoundPoly>,
    pub b_eval: Ef,
}

fn efg(x: Goldilocks) -> Ef {
    Ef::from(x)
}

fn ef_u64(x: u64) -> Ef {
    Ef::from(Goldilocks::from_u64(x))
}

fn absorb_round_poly(t: &mut Transcript, rp: &EfRoundPoly) {
    t.absorb_ef(rp.c0);
    t.absorb_ef(rp.c1);
    t.absorb_ef(rp.c2);
}

fn round_poly_batch(w_bufs: &[Vec<Ef>], v_bufs: &[Vec<Ef>]) -> EfRoundPoly {
    let n = w_bufs.len();
    let half = w_bufs[0].len() / 2;
    let mut p0 = Ef::ZERO;
    let mut p1 = Ef::ZERO;
    let mut p2 = Ef::ZERO;
    for i in 0..n {
        let w = &w_bufs[i];
        let v = &v_bufs[i];
        for s in 0..half {
            let (w0, w1) = (w[2 * s], w[2 * s + 1]);
            let (v0, v1) = (v[2 * s], v[2 * s + 1]);
            p0 = p0 + w0 * v0;
            p1 = p1 + w1 * v1;
            let w2 = ef_u64(2) * w1 - w0;
            let v2 = ef_u64(2) * v1 - v0;
            p2 = p2 + w2 * v2;
        }
    }
    let inv2 = ef_u64(2).inverse();
    let c0 = p0;
    let c1 = (ef_u64(4) * p1 - p2 - ef_u64(3) * p0) * inv2;
    let c2 = (p2 - ef_u64(2) * p1 + p0) * inv2;
    EfRoundPoly { c0, c1, c2 }
}

fn fold(buf: &mut Vec<Ef>, p: Ef) {
    let half = buf.len() / 2;
    for i in 0..half {
        let a = buf[2 * i];
        let b = buf[2 * i + 1];
        buf[i] = a + p * (b - a);
    }
    buf.truncate(half);
}

/// Prove `sum_x sum_i weights[i][x]·values[i][x] == claimed_sum` over EF.
pub fn prove_sum_of_products(
    weights: &[Vec<Ef>],
    values: &[Vec<Ef>],
    claimed_sum: Ef,
    transcript: &mut Transcript,
) -> (EfSumcheckProofBatch, Vec<Ef>) {
    prove_sum_of_products_owned(weights.to_vec(), values.to_vec(), claimed_sum, transcript)
}

/// Ownership-taking variant: the caller's weight/value buffers become the
/// folding buffers, so no full column copies are made. This is the path used
/// by the pooled aux openings, where columns can be large.
pub fn prove_sum_of_products_owned(
    mut w_bufs: Vec<Vec<Ef>>,
    mut v_bufs: Vec<Vec<Ef>>,
    claimed_sum: Ef,
    transcript: &mut Transcript,
) -> (EfSumcheckProofBatch, Vec<Ef>) {
    let n = w_bufs.len();
    assert!(n > 0, "empty batch");
    assert_eq!(v_bufs.len(), n);
    let t = w_bufs[0].len().trailing_zeros() as usize;
    assert!(w_bufs[0].len().is_power_of_two());
    for i in 0..n {
        assert_eq!(w_bufs[i].len(), v_bufs[i].len(), "batch column length mismatch");
        assert_eq!(w_bufs[i].len(), w_bufs[0].len(), "batch arity mismatch");
    }
    transcript.absorb_ef(claimed_sum);
    let mut rounds = Vec::with_capacity(t);
    let mut challenges = Vec::with_capacity(t);
    for _ in 0..t {
        let rp = round_poly_batch(&w_bufs, &v_bufs);
        rounds.push(rp);
        absorb_round_poly(transcript, &rp);
        let r = transcript.ef_field();
        challenges.push(r);
        for i in 0..n {
            fold(&mut w_bufs[i], r);
            fold(&mut v_bufs[i], r);
        }
    }
    let b_eval = (0..n).fold(Ef::ZERO, |acc, i| acc + w_bufs[i][0] * v_bufs[i][0]);
    (EfSumcheckProofBatch { rounds, b_eval }, challenges)
}

/// Verify the EF sum-of-products sum-check. Folding `b_eval` into the opened
/// evaluations is the caller's job, as in the base-field version.
pub fn verify_sum_of_products(
    proof: &EfSumcheckProofBatch,
    claimed_sum: Ef,
    num_rounds: usize,
    transcript: &mut Transcript,
) -> (bool, Vec<Ef>) {
    if proof.rounds.len() != num_rounds {
        return (false, Vec::new());
    }
    transcript.absorb_ef(claimed_sum);
    let mut prev = claimed_sum;
    let mut challenges = Vec::with_capacity(num_rounds);
    for rp in &proof.rounds {
        absorb_round_poly(transcript, rp);
        let r = transcript.ef_field();
        challenges.push(r);
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2;
        if p0 + p1 != prev {
            return (false, challenges);
        }
        prev = rp.eval(r);
    }
    (prev == proof.b_eval, challenges)
}

/// Lift base-field buffers into the EF batch sum-check input shape.
pub fn lift_columns(cols: &[&[Goldilocks]]) -> Vec<Vec<Ef>> {
    cols.iter().map(|c| c.iter().map(|&x| efg(x)).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::{PrimeCharacteristicRing, XorShift64};

    #[test]
    fn sum_of_products_ef_roundtrip_and_tamper() {
        let mut rng = XorShift64::new(0x9201);
        let t = 5usize;
        let n = 4usize;
        let weights: Vec<Vec<Ef>> = (0..n).map(|_| {
            (0..1 << t).map(|_| Ef::new([rng.field(), rng.field()])).collect()
        }).collect();
        let values: Vec<Vec<Ef>> = (0..n).map(|_| {
            (0..1 << t).map(|_| Ef::new([rng.field(), rng.field()])).collect()
        }).collect();
        let claimed = (0..n).fold(Ef::ZERO, |acc, i| {
            acc + weights[i].iter().zip(&values[i])
                .fold(Ef::ZERO, |a, (&w, &v)| a + w * v)
        });
        let mut tp = Transcript::new(b"ef-sop");
        let (proof, ch) = prove_sum_of_products(&weights, &values, claimed, &mut tp);
        let mut tv = Transcript::new(b"ef-sop");
        let (ok, ch2) = verify_sum_of_products(&proof, claimed, t, &mut tv);
        assert!(ok);
        assert_eq!(ch, ch2);
        let mut bad = proof.clone();
        bad.rounds[0].c1 += Ef::ONE;
        let mut tv = Transcript::new(b"ef-sop");
        assert!(!verify_sum_of_products(&bad, claimed, t, &mut tv).0);
    }
}
