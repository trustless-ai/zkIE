//! S9: outer reductions over the quadratic extension field.
//!
//! The base proof system samples challenges in Goldilocks (`|F| ≈ 2^64`),
//! which caps the outer sum-check / virtual / same-poly error near 2^-52.
//! The S9 budget moves only the *outer challenge arithmetic* to
//! `Ef = BinomialExtensionField<Goldilocks, 2>` (`|Ef| ≈ 2^128`); tensors,
//! commitments, and public execution values stay in the base field.
//!
//! Wire representation: every `Ef` value is serialized/absorbed as its two
//! base-field lanes (p3's array serialization), so the verifier and the hash
//! transcript see canonical base-field data. An MLE over base-field values is
//! evaluated at an EF point by folding base values with EF weights
//! ([`mle_eval_at_ef`]), exactly as the PCS boundary does with
//! `to_ef_point`.
//!
//! These routines are being introduced as a separate core layer (S9 step 2a);
//! the existing base-field routines are untouched until the composers are
//! migrated in the next sub-step.

use crate::common::field::{Ef, Field, Goldilocks, PrimeCharacteristicRing};
use crate::common::transcript::Transcript;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EfRoundPoly {
    pub c0: Ef,
    pub c1: Ef,
    pub c2: Ef,
}

impl EfRoundPoly {
    #[inline]
    pub fn eval(&self, x: Ef) -> Ef {
        self.c0 + self.c1 * x + self.c2 * x * x
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EfSumcheckProof {
    pub rounds: Vec<EfRoundPoly>,
    pub f_eval: Ef,
    pub h_eval: Ef,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EfVirtualProof {
    pub rounds: Vec<Vec<Ef>>,
    pub final_evals: Vec<Ef>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct EfSamePolyProof {
    pub gamma: Ef,
    pub proof: EfVirtualProof,
    pub merged_point: Vec<Ef>,
}

#[inline]
fn efg(x: Goldilocks) -> Ef {
    Ef::from(x)
}

#[inline]
fn ef_u64(x: u64) -> Ef {
    Ef::from(Goldilocks::from_u64(x))
}

#[inline]
fn ef_i64(x: i64) -> Ef {
    if x >= 0 { ef_u64(x as u64) } else { -ef_u64((-x) as u64) }
}

fn absorb_round_poly(t: &mut Transcript, rp: &EfRoundPoly) {
    t.absorb_ef(rp.c0);
    t.absorb_ef(rp.c1);
    t.absorb_ef(rp.c2);
}

fn round_poly(f_buf: &[Ef], h_buf: &[Ef]) -> EfRoundPoly {
    let half = f_buf.len() / 2;
    let mut y0 = Ef::ZERO;
    let mut y1 = Ef::ZERO;
    let mut y2 = Ef::ZERO;
    for s in 0..half {
        let f0 = f_buf[2 * s];
        let f1 = f_buf[2 * s + 1];
        let h0 = h_buf[2 * s];
        let h1 = h_buf[2 * s + 1];
        y0 = y0 + f0 * h0;
        y1 = y1 + f1 * h1;
        let f2 = ef_u64(2) * f1 - f0;
        let h2 = ef_u64(2) * h1 - h0;
        y2 = y2 + f2 * h2;
    }
    let inv2 = ef_u64(2).inverse();
    let c0 = y0;
    let c1 = (ef_u64(4) * y1 - y2 - ef_u64(3) * y0) * inv2;
    let c2 = (y2 - ef_u64(2) * y1 + y0) * inv2;
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

/// EF sum-check for `sum_x f(x)·h(x)` over EF-valued buffers.
pub fn prove_sumcheck_ef(
    f: &[Ef],
    h: &[Ef],
    claimed_sum: Ef,
    transcript: &mut Transcript,
) -> (EfSumcheckProof, Vec<Ef>) {
    assert_eq!(f.len(), h.len(), "sumcheck f/h length mismatch");
    assert!(f.len().is_power_of_two(), "sumcheck length must be a power of two");
    let t = f.len().trailing_zeros() as usize;
    transcript.absorb_ef(claimed_sum);
    let mut f_buf = f.to_vec();
    let mut h_buf = h.to_vec();
    let mut rounds = Vec::with_capacity(t);
    let mut challenges = Vec::with_capacity(t);
    for _ in 0..t {
        let rp = round_poly(&f_buf, &h_buf);
        rounds.push(rp);
        absorb_round_poly(transcript, &rp);
        let r = transcript.ef_field();
        challenges.push(r);
        fold(&mut f_buf, r);
        fold(&mut h_buf, r);
    }
    (
        EfSumcheckProof {
            rounds,
            f_eval: f_buf[0],
            h_eval: h_buf[0],
        },
        challenges,
    )
}

/// EF sum-check for base-field buffers (lifted to EF).
pub fn prove_sumcheck(
    f: &[Goldilocks],
    h: &[Goldilocks],
    claimed_sum: Ef,
    transcript: &mut Transcript,
) -> (EfSumcheckProof, Vec<Ef>) {
    let f_ef: Vec<Ef> = f.iter().map(|&x| efg(x)).collect();
    let h_ef: Vec<Ef> = h.iter().map(|&x| efg(x)).collect();
    prove_sumcheck_ef(&f_ef, &h_ef, claimed_sum, transcript)
}

/// EF sum-check verification. Returns `(ok, challenges)`; `ok` includes the
/// final `f(r)·h(r)` relation.
pub fn verify_sumcheck(
    proof: &EfSumcheckProof,
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
    (prev == proof.f_eval * proof.h_eval, challenges)
}

fn interpolate(vals: &[Ef], x: Ef, d: usize) -> Ef {
    let mut out = Ef::ZERO;
    for k in 0..=d {
        let mut term = vals[k];
        for j in 0..=d {
            if j != k {
                let num = x - ef_u64(j as u64);
                let den = ef_u64(k as u64) - ef_u64(j as u64);
                term = term * num * den.inverse();
            }
        }
        out = out + term;
    }
    out
}

/// Evaluate a virtual polynomial `sum_i coeff_i · prod_{j in term_i} evals[j]`
/// over EF values.
pub fn virtual_eval(terms: &[(Ef, Vec<usize>)], evals: &[Ef]) -> Ef {
    let mut out = Ef::ZERO;
    for (coeff, idxs) in terms {
        let mut prod = *coeff;
        for &j in idxs {
            prod = prod * evals[j];
        }
        out = out + prod;
    }
    out
}

/// EF virtual sum-check. All MLE columns are supplied already lifted to EF.
pub fn prove_virtual(
    mles: &[&[Ef]],
    terms: &[(Ef, Vec<usize>)],
    claimed: Ef,
    transcript: &mut Transcript,
) -> (EfVirtualProof, Vec<Ef>) {
    assert!(!mles.is_empty(), "virtual proof needs at least one MLE");
    let n = mles[0].len();
    assert!(n.is_power_of_two(), "virtual domain must be a power of two");
    for m in mles {
        assert_eq!(m.len(), n, "virtual MLE lengths differ");
    }
    let max_deg = terms.iter().map(|(_, idxs)| idxs.len()).max().unwrap_or(0);
    assert!(max_deg >= 1, "virtual terms must have degree at least 1");
    transcript.absorb_ef(claimed);
    let mut bufs: Vec<Vec<Ef>> = mles.iter().map(|m| m.to_vec()).collect();
    let t = n.trailing_zeros() as usize;
    let mut rounds = Vec::with_capacity(t);
    let mut challenges = Vec::with_capacity(t);
    for _ in 0..t {
        let half = bufs[0].len() / 2;
        let mut pvals = vec![Ef::ZERO; max_deg + 1];
        for s in 0..half {
            for (coeff, idxs) in terms {
                for k in 0..=max_deg {
                    let mut prod = *coeff;
                    for &j in idxs {
                        let f0 = bufs[j][2 * s];
                        let f1 = bufs[j][2 * s + 1];
                        let fk = ef_u64(k as u64) * f1 - ef_i64(k as i64 - 1) * f0;
                        prod = prod * fk;
                    }
                    pvals[k] = pvals[k] + prod;
                }
            }
        }
        for &p in &pvals {
            transcript.absorb_ef(p);
        }
        let r = transcript.ef_field();
        rounds.push(pvals);
        challenges.push(r);
        for buf in bufs.iter_mut() {
            fold(buf, r);
        }
    }
    let final_evals = bufs.iter().map(|b| b[0]).collect();
    (EfVirtualProof { rounds, final_evals }, challenges)
}

/// EF virtual verification. Returns `(ok, challenges, final_claim)`.
pub fn verify_virtual(
    proof: &EfVirtualProof,
    terms: &[(Ef, Vec<usize>)],
    claimed: Ef,
    num_rounds: usize,
    transcript: &mut Transcript,
) -> (bool, Vec<Ef>, Ef) {
    let max_deg = terms.iter().map(|(_, idxs)| idxs.len()).max().unwrap_or(0);
    // A degree-0 virtual has no soundness and would index pvals[1] below.
    if max_deg == 0 || proof.rounds.len() != num_rounds {
        return (false, Vec::new(), Ef::ZERO);
    }
    transcript.absorb_ef(claimed);
    let mut current = claimed;
    let mut challenges = Vec::with_capacity(num_rounds);
    for pvals in &proof.rounds {
        if pvals.len() != max_deg + 1 {
            return (false, challenges, Ef::ZERO);
        }
        for &p in pvals {
            transcript.absorb_ef(p);
        }
        let r = transcript.ef_field();
        challenges.push(r);
        if pvals[0] + pvals[1] != current {
            return (false, challenges, Ef::ZERO);
        }
        current = interpolate(pvals, r, max_deg);
    }
    // Bind the carried final evaluations: the virtual polynomial at the
    // derived challenge vector must equal the replay's final claim. This
    // rejects the S8 dead-field malleability for EF virtual proofs.
    let max_idx = terms.iter().flat_map(|(_, idxs)| idxs.iter()).copied().max().unwrap_or(0);
    if proof.final_evals.len() <= max_idx
        || virtual_eval(terms, &proof.final_evals) != current
    {
        return (false, challenges, current);
    }
    (true, challenges, current)
}

fn absorb_claims(transcript: &mut Transcript, claims: &[(Vec<Ef>, Ef)]) {
    transcript.absorb(b"zkie-ef-same-poly-claims-v1");
    transcript.absorb_u64(claims.len() as u64);
    for (point, value) in claims {
        transcript.absorb_u64(point.len() as u64);
        transcript.absorb_efs(point);
        transcript.absorb_ef(*value);
    }
}

/// Evaluate the equality polynomial at an EF point: `prod_j (x_j r_j +
/// (1−x_j)(1−r_j))`.
pub fn eq_poly(r: &[Ef], x: &[Ef]) -> Option<Ef> {
    if r.len() != x.len() {
        return None;
    }
    let mut acc = Ef::ONE;
    for (&ri, &xi) in r.iter().zip(x) {
        acc = acc * (xi * ri + (Ef::ONE - xi) * (Ef::ONE - ri));
    }
    Some(acc)
}

/// Equality-polynomial values over the boolean hypercube, lifted to EF.
pub fn eq_evals(r: &[Ef]) -> Vec<Ef> {
    let d = r.len();
    if d >= usize::BITS as usize {
        return Vec::new();
    }
    let n = 1usize << d;
    let mut eq = vec![Ef::ONE; n];
    let mut size = 1usize;
    for &rj in r {
        let one_minus = Ef::ONE - rj;
        for i in 0..size {
            let v = eq[i];
            eq[i] = v * one_minus;
            eq[i + size] = v * rj;
        }
        size <<= 1;
    }
    eq
}

/// Partially evaluate a base-field MLE at an EF prefix, returning the
/// EF-valued residual table used by the MatMul/batch-opening reductions.
pub fn partial_eval_at_ef(values: &[Goldilocks], fix: &[Ef]) -> Option<Vec<Ef>> {
    let t = values.len().trailing_zeros() as usize;
    if !values.len().is_power_of_two() || fix.len() > t {
        return None;
    }
    let mut buf: Vec<Ef> = values.iter().map(|&v| efg(v)).collect();
    for &p in fix {
        let half = buf.len() / 2;
        for i in 0..half {
            let a = buf[2 * i];
            let b = buf[2 * i + 1];
            buf[i] = a + p * (b - a);
        }
        buf.truncate(half);
    }
    Some(buf)
}

/// Evaluate a base-field MLE at an EF point: fold base values with EF
/// weights. The result is an EF element; when every point coordinate lies in
/// the base subfield (second lane zero) it equals the base-field evaluation.
pub fn mle_eval_at_ef(values: &[Goldilocks], point: &[Ef]) -> Option<Ef> {
    let t = point.len();
    if t >= usize::BITS as usize || values.len() != 1usize.checked_shl(t as u32)? {
        return None;
    }
    let mut buf: Vec<Ef> = values.iter().map(|&v| efg(v)).collect();
    let mut size = buf.len();
    for &p in point {
        let half = size / 2;
        for i in 0..half {
            let a = buf[2 * i];
            let b = buf[2 * i + 1];
            buf[i] = a + p * (b - a);
        }
        size = half;
    }
    Some(buf[0])
}

/// Evaluate an EF-valued MLE at an EF point (length-checked).
pub fn eval_ef_slice(values: &[Ef], point: &[Ef]) -> Option<Ef> {
    let t = point.len();
    if t >= usize::BITS as usize || values.len() != 1usize.checked_shl(t as u32)? {
        return None;
    }
    let mut buf = values.to_vec();
    let mut size = buf.len();
    for &p in point {
        let half = size / 2;
        for i in 0..half {
            let a = buf[2 * i];
            let b = buf[2 * i + 1];
            buf[i] = a + p * (b - a);
        }
        size = half;
    }
    Some(buf[0])
}

/// EF same-poly merge over a base-field MLE.
pub fn prove_same_poly(
    f: &[Goldilocks],
    claims: &[(Vec<Ef>, Ef)],
    transcript: &mut Transcript,
) -> (EfSamePolyProof, Ef) {
    let f_ef: Vec<Ef> = f.iter().map(|&x| efg(x)).collect();
    prove_same_poly_ef(&f_ef, claims, transcript)
}

/// EF same-poly merge over an EF-valued MLE (e.g. a LogUp fraction column).
pub fn prove_same_poly_ef(
    f: &[Ef],
    claims: &[(Vec<Ef>, Ef)],
    transcript: &mut Transcript,
) -> (EfSamePolyProof, Ef) {
    assert!(!claims.is_empty(), "same_poly needs at least one claim");
    let t = f.len().trailing_zeros() as usize;
    for (r_i, _) in claims {
        assert_eq!(r_i.len(), t, "same_poly claim arity mismatch");
    }
    absorb_claims(transcript, claims);
    let gamma = transcript.ef_field();
    let mut coeffs = Vec::with_capacity(claims.len());
    let mut acc = Ef::ONE;
    for _ in claims {
        coeffs.push(acc);
        acc = acc * gamma;
    }
    let eqs: Vec<Vec<Ef>> = claims.iter().map(|(r_i, _)| eq_evals(r_i)).collect();
    let mut mles: Vec<&[Ef]> = Vec::with_capacity(1 + eqs.len());
    mles.push(f);
    for e in &eqs {
        mles.push(e.as_slice());
    }
    let mut terms: Vec<(Ef, Vec<usize>)> = Vec::new();
    let mut claimed = Ef::ZERO;
    for (i, (_, y_i)) in claims.iter().enumerate() {
        terms.push((coeffs[i], vec![0usize, i + 1]));
        claimed = claimed + coeffs[i] * *y_i;
    }
    let (proof, merged_point) = prove_virtual(&mles, &terms, claimed, transcript);
    let merged_eval = proof.final_evals[0];
    (EfSamePolyProof { gamma, proof, merged_point }, merged_eval)
}

/// Verify an EF same-poly merge and return the authenticated merged claim.
pub fn verify_same_poly(
    proof: &EfSamePolyProof,
    claims: &[(Vec<Ef>, Ef)],
    transcript: &mut Transcript,
) -> Option<(Vec<Ef>, Ef)> {
    if claims.is_empty() {
        return None;
    }
    let t = claims[0].0.len();
    if claims.iter().any(|(r_i, _)| r_i.len() != t) {
        return None;
    }
    absorb_claims(transcript, claims);
    let gamma = transcript.ef_field();
    if gamma != proof.gamma || gamma == Ef::ZERO || gamma == Ef::ONE {
        return None;
    }
    let mut coeffs = Vec::with_capacity(claims.len());
    let mut acc = Ef::ONE;
    for _ in claims {
        coeffs.push(acc);
        acc = acc * gamma;
    }
    for i in 0..claims.len() {
        for j in (i + 1)..claims.len() {
            if claims[i].0 == claims[j].0 && claims[i].1 != claims[j].1 {
                return None;
            }
        }
    }
    let mut terms: Vec<(Ef, Vec<usize>)> = Vec::new();
    let mut claimed = Ef::ZERO;
    for (i, (_, y_i)) in claims.iter().enumerate() {
        terms.push((coeffs[i], vec![0usize, i + 1]));
        claimed = claimed + coeffs[i] * *y_i;
    }
    let (ok, r, final_claim) = verify_virtual(&proof.proof, &terms, claimed, t, transcript);
    if !ok || r != proof.merged_point {
        return None;
    }
    let mut w = Ef::ZERO;
    for (i, (r_i, _)) in claims.iter().enumerate() {
        w = w + coeffs[i] * eq_poly(r_i, &r)?;
    }
    if w == Ef::ZERO {
        return None;
    }
    Some((r, final_claim * w.inverse()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::XorShift64;
    use crate::common::mle;

    fn ef_rng(rng: &mut XorShift64) -> Ef {
        Ef::new([rng.field(), rng.field()])
    }

    fn ef_point(rng: &mut XorShift64, n: usize) -> Vec<Ef> {
        (0..n).map(|_| ef_rng(rng)).collect()
    }

    #[test]
    fn sumcheck_ef_roundtrip_and_tamper() {
        let mut rng = XorShift64::new(0x9001);
        let t = 6usize;
        let f: Vec<Goldilocks> = (0..1 << t).map(|_| rng.field()).collect();
        let h: Vec<Goldilocks> = (0..1 << t).map(|_| rng.field()).collect();
        let claimed = f.iter().zip(&h).fold(Ef::ZERO, |acc, (&a, &b)| {
            acc + efg(a) * efg(b)
        });
        let mut tp = Transcript::new(b"ef-sumcheck");
        let (proof, ch) = prove_sumcheck(&f, &h, claimed, &mut tp);
        let mut tv = Transcript::new(b"ef-sumcheck");
        let (ok, ch2) = verify_sumcheck(&proof, claimed, t, &mut tv);
        assert!(ok);
        assert_eq!(ch, ch2);
        let mut bad = proof.clone();
        bad.rounds[0].c0 += Ef::ONE;
        let mut tv = Transcript::new(b"ef-sumcheck");
        assert!(!verify_sumcheck(&bad, claimed, t, &mut tv).0);
    }

    #[test]
    fn virtual_ef_roundtrip_and_tamper() {
        let mut rng = XorShift64::new(0x9002);
        let n = 32usize;
        let a: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let c: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let a_ef: Vec<Ef> = a.iter().map(|&x| efg(x)).collect();
        let c_ef: Vec<Ef> = c.iter().map(|&x| efg(x)).collect();
        // Non-degenerate linear relation: sum eq*(a - c) with the honest
        // claimed sum; the eq column has a non-zero coefficient.
        let claimed = a.iter().zip(&c).fold(Ef::ZERO, |acc, (&x, &y)| {
            acc + efg(x) - efg(y)
        });
        let terms = vec![
            (Ef::ONE, vec![0usize, 1]),
            (-Ef::ONE, vec![0, 2]),
        ];
        let mles: Vec<&[Ef]> = vec![&[Ef::ONE; 32], &a_ef, &c_ef];
        let mut tp = Transcript::new(b"ef-virtual");
        let (proof, ch) = prove_virtual(&mles, &terms, claimed, &mut tp);
        let mut tv = Transcript::new(b"ef-virtual");
        let (ok, ch2, fin) = verify_virtual(&proof, &terms, claimed, 5, &mut tv);
        assert!(ok);
        assert_eq!(ch, ch2);
        assert_eq!(virtual_eval(&terms, &proof.final_evals), fin);
        let mut bad = proof.clone();
        bad.final_evals[0] += Ef::ONE;
        let mut tv = Transcript::new(b"ef-virtual");
        assert!(!verify_virtual(&bad, &terms, claimed, 5, &mut tv).0);
    }

    #[test]
    fn same_poly_ef_roundtrip_and_tamper() {
        let mut rng = XorShift64::new(0x9003);
        let t = 6usize;
        let f: Vec<Goldilocks> = (0..1 << t).map(|_| rng.field()).collect();
        let claims: Vec<(Vec<Ef>, Ef)> = (0..3).map(|_| {
            let r = ef_point(&mut rng, t);
            let y = mle_eval_at_ef(&f, &r).unwrap();
            (r, y)
        }).collect();
        let mut tp = Transcript::new(b"ef-same-poly");
        let (proof, merged) = prove_same_poly(&f, &claims, &mut tp);
        let mut tv = Transcript::new(b"ef-same-poly");
        let (r, v) = verify_same_poly(&proof, &claims, &mut tv).unwrap();
        assert_eq!(r, proof.merged_point);
        assert_eq!(v, mle_eval_at_ef(&f, &r).unwrap());
        assert_eq!(merged, mle_eval_at_ef(&f, &r).unwrap());
        let mut bad = claims.clone();
        bad[0].1 += Ef::ONE;
        let mut tv = Transcript::new(b"ef-same-poly");
        assert!(verify_same_poly(&proof, &bad, &mut tv).is_none());
    }

    #[test]
    fn ef_mle_eval_matches_base_on_subfield_points() {
        let mut rng = XorShift64::new(0x9004);
        let t = 5usize;
        let f: Vec<Goldilocks> = (0..1 << t).map(|_| rng.field()).collect();
        let base_pt: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let ef_pt: Vec<Ef> = base_pt.iter().map(|&x| efg(x)).collect();
        assert_eq!(mle_eval_at_ef(&f, &ef_pt).unwrap(), efg(mle::eval(&f, &base_pt)));
    }

    #[test]
    fn ef_wire_encoding_is_two_base_lanes() {
        let x = Ef::new([Goldilocks::from_u64(7), Goldilocks::from_u64(9)]);
        let bytes = bincode::serialize(&x).unwrap();
        assert_eq!(bytes.len(), 16, "EF wire must be two base-field lanes");
        let back: Ef = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back, x);
        let rp = EfRoundPoly { c0: x, c1: Ef::ONE, c2: Ef::ZERO };
        let rb = bincode::serialize(&rp).unwrap();
        assert_eq!(rb.len(), 48, "degree-2 EF round message is three EF values");
        let rback: EfRoundPoly = bincode::deserialize(&rb).unwrap();
        assert_eq!(rback, rp);
    }

    #[test]
    fn transcript_ef_absorption_is_deterministic_and_bound() {
        let mut a = Transcript::new(b"ef-transcript");
        let mut b = Transcript::new(b"ef-transcript");
        let x = Ef::new([Goldilocks::from_u64(7), Goldilocks::from_u64(9)]);
        a.absorb_ef(x);
        b.absorb_ef(x);
        assert_eq!(a.ef_field(), b.ef_field());
        let mut c = Transcript::new(b"ef-transcript");
        c.absorb_ef(Ef::new([Goldilocks::from_u64(7), Goldilocks::from_u64(10)]));
        assert_ne!(a.ef_field(), c.ef_field());
    }
}
