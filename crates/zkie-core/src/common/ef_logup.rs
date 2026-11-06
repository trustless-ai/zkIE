//! S9: LogUp / fractional GKR over the quadratic extension field.
//!
//! Mirrors `common::logup_gkr` with EF challenges (`alpha`, `beta`, per-layer
//! `gamma`), EF-valued fraction columns, EF virtual/same-poly messages, and EF
//! derived leaf/root claims. Tensors and multiplicities stay base-field; the
//! column builders lift table/activation values to EF.

use crate::common::ef_outer::{
    self, eq_poly, prove_same_poly_ef, prove_virtual, verify_same_poly, verify_virtual,
    virtual_eval, EfSamePolyProof, EfVirtualProof,
};
use crate::common::field::{Ef, Field, Goldilocks, PrimeCharacteristicRing};
use crate::common::transcript::Transcript;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct EfFractionalLayer {
    pub gamma: Ef,
    pub proof: EfVirtualProof,
    pub s: Vec<Ef>,
    pub num_merge: EfSamePolyProof,
    pub den_merge: EfSamePolyProof,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct EfFractionalProof {
    pub layers: Vec<EfFractionalLayer>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EfFractionalChainClaims {
    pub q_num: Vec<Ef>,
    pub num_leaf: Ef,
    pub q_den: Vec<Ef>,
    pub den_leaf: Ef,
    pub final_num: Ef,
    pub final_den: Ef,
}

fn efg(x: Goldilocks) -> Ef {
    Ef::from(x)
}

fn ef_u64(x: u64) -> Ef {
    Ef::from(Goldilocks::from_u64(x))
}

fn build_terms(gamma: Ef, a: Ef, b: Ef) -> Vec<(Ef, Vec<usize>)> {
    let neg_a = -a;
    let neg_gb = -(gamma * b);
    vec![
        (Ef::ONE, vec![0usize, 2, 5]),
        (Ef::ONE, vec![0, 3, 4]),
        (neg_a, vec![0]),
        (gamma, vec![1, 4, 5]),
        (neg_gb, vec![1]),
    ]
}

/// Prove `sum_i num[i]/den[i]` via the fraction-addition tree over EF.
pub fn prove_fractional(num: Vec<Ef>, den: Vec<Ef>, transcript: &mut Transcript) -> EfFractionalProof {
    assert_eq!(num.len(), den.len());
    assert!(num.len().is_power_of_two());
    assert!(num.len() >= 2, "fractional domain must have at least 2 entries");
    let t = num.len().trailing_zeros() as usize;

    let mut tn = vec![num];
    let mut td = vec![den];
    while tn.last().unwrap().len() > 1 {
        let half = tn.last().unwrap().len() / 2;
        let (nl, nh) = tn.last().unwrap().split_at(half);
        let (dl, dh) = td.last().unwrap().split_at(half);
        let next_n: Vec<Ef> = (0..half).map(|i| nl[i] * dh[i] + nh[i] * dl[i]).collect();
        let next_d: Vec<Ef> = (0..half).map(|i| dl[i] * dh[i]).collect();
        tn.push(next_n);
        td.push(next_d);
    }
    let mut a = tn[t][0];
    let mut b = td[t][0];
    let mut q_num: Vec<Ef> = Vec::new();
    let mut q_den: Vec<Ef> = Vec::new();
    let mut layers = Vec::with_capacity(t);
    for k in (1..=t).rev() {
        let gamma = transcript.ef_field();
        let eq_num = ef_outer::eq_evals(&q_num);
        let eq_den = ef_outer::eq_evals(&q_den);
        let half = 1usize << (t - k);
        let num_k1 = &tn[k - 1];
        let den_k1 = &td[k - 1];
        let mles: Vec<&[Ef]> = vec![
            &eq_num,
            &eq_den,
            &num_k1[..half],
            &num_k1[half..],
            &den_k1[..half],
            &den_k1[half..],
        ];
        let terms = build_terms(gamma, a, b);
        let (proof, s) = prove_virtual(&mles, &terms, Ef::ZERO, transcript);
        let fe = &proof.final_evals;
        transcript.absorb(b"zkie-ef-logup-child-evals-v1");
        transcript.absorb_efs(&fe[2..6]);
        let mut p0 = s.clone();
        p0.push(Ef::ZERO);
        let mut p1 = s.clone();
        p1.push(Ef::ONE);
        let (num_merge, a_next) =
            prove_same_poly_ef(num_k1, &[(p0.clone(), fe[2]), (p1.clone(), fe[3])], transcript);
        let (den_merge, b_next) =
            prove_same_poly_ef(den_k1, &[(p0, fe[4]), (p1, fe[5])], transcript);
        a = a_next;
        b = b_next;
        q_num = num_merge.merged_point.clone();
        q_den = den_merge.merged_point.clone();
        layers.push(EfFractionalLayer { gamma, proof, s, num_merge, den_merge });
    }
    EfFractionalProof { layers }
}

/// Replay the layered EF fractional chain. Structurally identical to the
/// base-field S3 replay, with EF eq columns, challenges, and claims.
pub fn verify_fractional_chain_core(
    proof: &EfFractionalProof,
    t: usize,
    transcript: &mut Transcript,
) -> Option<EfFractionalChainClaims> {
    if t == 0 || proof.layers.len() != t {
        return None;
    }
    let mut a = Ef::ZERO;
    let mut b = Ef::ZERO;
    let mut q_num: Vec<Ef> = Vec::new();
    let mut q_den: Vec<Ef> = Vec::new();
    let mut root: Option<(Ef, Ef)> = None;
    for (li, layer) in proof.layers.iter().enumerate() {
        let gamma = transcript.ef_field();
        if gamma != layer.gamma {
            return None;
        }
        let arity = li;
        let fe = &layer.proof.final_evals;
        if layer.s.len() != arity || layer.proof.rounds.len() != arity || fe.len() != 6 {
            return None;
        }
        let eq0 = eq_poly(&q_num, &layer.s)?;
        let eq1 = eq_poly(&q_den, &layer.s)?;
        if fe[0] != eq0 || fe[1] != eq1 {
            return None;
        }
        if li == 0 {
            a = fe[2] * fe[5] + fe[3] * fe[4];
            b = fe[4] * fe[5];
            root = Some((a, b));
        }
        let terms = build_terms(gamma, a, b);
        let (ok, s, final_claim) =
            verify_virtual(&layer.proof, &terms, Ef::ZERO, arity, transcript);
        if !ok || s != layer.s {
            return None;
        }
        if final_claim != virtual_eval(&terms, &[eq0, eq1, fe[2], fe[3], fe[4], fe[5]]) {
            return None;
        }
        transcript.absorb(b"zkie-ef-logup-child-evals-v1");
        transcript.absorb_efs(&fe[2..6]);
        let mut p0 = s.clone();
        p0.push(Ef::ZERO);
        let mut p1 = s.clone();
        p1.push(Ef::ONE);
        let num_claims = vec![(p0.clone(), fe[2]), (p1.clone(), fe[3])];
        let den_claims = vec![(p0, fe[4]), (p1, fe[5])];
        let (q_next, a_next) = verify_same_poly(&layer.num_merge, &num_claims, transcript)?;
        let (qd_next, b_next) = verify_same_poly(&layer.den_merge, &den_claims, transcript)?;
        a = a_next;
        b = b_next;
        q_num = q_next;
        q_den = qd_next;
    }
    let (final_num, final_den) = root?;
    if final_den == Ef::ZERO {
        return None;
    }
    Some(EfFractionalChainClaims {
        q_num, num_leaf: a, q_den, den_leaf: b, final_num, final_den,
    })
}

/// Chain replay plus the root ratio check `final_num == claimed·final_den`.
pub fn verify_fractional(
    proof: &EfFractionalProof,
    claimed: Ef,
    t: usize,
    transcript: &mut Transcript,
) -> bool {
    match verify_fractional_chain_core(proof, t, transcript) {
        Some(c) => c.final_den != Ef::ZERO && c.final_num == claimed * c.final_den,
        None => false,
    }
}

/// Prover-side helper: read the leaf points/evals out of a freshly proven EF
/// chain (never used on untrusted proofs).
pub fn fractional_leaf(proof: &EfFractionalProof) -> (Vec<Ef>, Ef, Vec<Ef>, Ef) {
    match proof.layers.last() {
        Some(layer) => (
            layer.num_merge.merged_point.clone(),
            layer.num_merge.proof.final_evals[0],
            layer.den_merge.merged_point.clone(),
            layer.den_merge.proof.final_evals[0],
        ),
        None => (Vec::new(), Ef::ZERO, Vec::new(), Ef::ZERO),
    }
}

/// EF permutation proof: two fractional sides plus the EF alpha. The
/// permutation columns `a`/`b` are tensor data and stay base-field; the
/// verifier never consumes them (they are prover-side context).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct EfPermutationProof {
    pub a_side: EfFractionalProof,
    pub b_side: EfFractionalProof,
    pub alpha: Ef,
    pub a: Vec<Ef>,
    pub b: Vec<Ef>,
}

/// Permutation-side LogUp columns over EF: `1/(α−value[i])`.
pub fn build_perm_num_den(value: &[Ef], alpha: Ef) -> (Vec<Ef>, Vec<Ef>) {
    let total = value.len().next_power_of_two().max(32);
    let mut num = vec![Ef::ZERO; total];
    let mut den = vec![alpha; total];
    for i in 0..value.len() {
        num[i] = Ef::ONE;
        den[i] = alpha - value[i];
    }
    (num, den)
}

pub fn prove_permutation(
    a: &[Ef],
    b: &[Ef],
    transcript: &mut Transcript,
) -> EfPermutationProof {
    assert_eq!(a.len(), b.len());
    let alpha = transcript.ef_field();
    let (num_a, den_a) = build_perm_num_den(a, alpha);
    let (num_b, den_b) = build_perm_num_den(b, alpha);
    let a_side = prove_fractional(num_a, den_a, transcript);
    let b_side = prove_fractional(num_b, den_b, transcript);
    EfPermutationProof { a_side, b_side, alpha, a: a.to_vec(), b: b.to_vec() }
}

pub fn verify_lookup_split(
    witness: &EfFractionalProof,
    table: &EfFractionalProof,
    t_witness: usize,
    t_table: usize,
    transcript: &mut Transcript,
) -> bool {
    let wc = match verify_fractional_chain_core(witness, t_witness, transcript) {
        Some(c) => c,
        None => return false,
    };
    if wc.final_den == Ef::ZERO {
        return false;
    }
    let tc = match verify_fractional_chain_core(table, t_table, transcript) {
        Some(c) => c,
        None => return false,
    };
    if tc.final_den == Ef::ZERO {
        return false;
    }
    wc.final_num * tc.final_den == tc.final_num * wc.final_den
}

pub fn verify_permutation(
    proof: &EfPermutationProof,
    t: usize,
    transcript: &mut Transcript,
) -> bool {
    let alpha = transcript.ef_field();
    if alpha != proof.alpha {
        return false;
    }
    verify_lookup_split(&proof.a_side, &proof.b_side, t, t, transcript)
}

/// Multiplicity of each table entry for one routing array.
pub fn lookup_multiplicity(x: &[u32], t: usize) -> Vec<u64> {
    let mut m = vec![0u64; t];
    for &i in x {
        if (i as usize) < t {
            m[i as usize] += 1;
        }
    }
    m
}

/// Merge multiplicities from several lookups into one table-side column.
pub fn merge_multiplicity(xs: &[&[u32]], t: usize) -> Vec<u64> {
    let mut m = vec![0u64; t];
    for x in xs {
        for &i in *x {
            if (i as usize) < t {
                m[i as usize] += 1;
            }
        }
    }
    m
}

/// Batched witness-side EF columns: flatten all `(x,y)` pairs into one
/// column, padded to at least 2^5 (the WHIR folding-factor floor), and prove
/// the fraction tree. Returns the proof plus the flattened x/y/num/den
/// columns for the caller's openings.
pub fn prove_lookup_witness_batch_columns(
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    alpha: Ef,
    beta: Ef,
    transcript: &mut Transcript,
) -> (EfFractionalProof, Vec<Ef>, Vec<Ef>, Vec<Ef>, Vec<Ef>) {
    assert_eq!(xs.len(), ys.len());
    let n: usize = xs.iter().map(|x| x.len()).sum();
    let total = n.next_power_of_two().max(32);
    let mut num = vec![Ef::ZERO; total];
    let mut x_flat = vec![Ef::ZERO; total];
    let mut y_flat = vec![Ef::ZERO; total];
    let mut den = vec![Ef::ZERO; total];
    let mut pos = 0usize;
    for (x, y) in xs.iter().zip(ys.iter()) {
        for i in 0..x.len() {
            num[pos] = Ef::ONE;
            x_flat[pos] = ef_u64(x[i] as u64);
            y_flat[pos] = efg(y[i]);
            pos += 1;
        }
    }
    for j in 0..total {
        den[j] = alpha + x_flat[j] + beta * y_flat[j];
    }
    let proof = prove_fractional(num.clone(), den.clone(), transcript);
    (proof, x_flat, y_flat, num, den)
}

/// Witness-side LogUp columns over EF: `1/(α+x_i+β·y_i)`.
pub fn build_lookup_witness_num_den(
    x: &[u32],
    y: &[Goldilocks],
    alpha: Ef,
    beta: Ef,
) -> (Vec<Ef>, Vec<Ef>) {
    let n = x.len();
    let total = n.next_power_of_two();
    let mut num = vec![Ef::ZERO; total];
    let mut den = vec![Ef::ONE; total];
    for i in 0..n {
        num[i] = Ef::ONE;
        den[i] = alpha + ef_u64(x[i] as u64) + beta * efg(y[i]);
    }
    (num, den)
}

/// Table-side LogUp columns over EF: `m_j/(α+j+β·table[j])`.
pub fn build_lookup_table_num_den(
    m: &[u64],
    table: &[Goldilocks],
    alpha: Ef,
    beta: Ef,
) -> (Vec<Ef>, Vec<Ef>) {
    let t = table.len();
    let total = t.next_power_of_two();
    let mut num = vec![Ef::ZERO; total];
    let mut den = vec![Ef::ZERO; total];
    for j in 0..total {
        num[j] = if j < t { ef_u64(m[j]) } else { Ef::ZERO };
        let tj = if j < t { table[j] } else { Goldilocks::ZERO };
        den[j] = alpha + ef_u64(j as u64) + beta * efg(tj);
    }
    (num, den)
}

#[cfg(test)]
mod tests {
    use crate::common::field::{PrimeCharacteristicRing, XorShift64};

    use super::*;

    fn ef_rng(rng: &mut XorShift64) -> Ef {
        Ef::new([rng.field(), rng.field()])
    }

    #[test]
    fn lookup_split_ef_identity_and_zero_den_rejected() {
        let mut rng = XorShift64::new(0x9302);
        let x: Vec<u32> = vec![0, 1, 1, 3];
        let table: Vec<Goldilocks> = (0..4).map(|i| Goldilocks::from_u64(100 + i)).collect();
        let y: Vec<Goldilocks> = x.iter()
            .map(|&i| table[i as usize]).collect();
        let alpha = ef_rng(&mut rng);
        let beta = ef_rng(&mut rng);
        let m = lookup_multiplicity(&x, table.len());
        let (wn, wd) = build_lookup_witness_num_den(&x, &y, alpha, beta);
        let (tn, td) = build_lookup_table_num_den(&m, &table, alpha, beta);
        let wt = wn.len().trailing_zeros() as usize;
        let tt = tn.len().trailing_zeros() as usize;
        let mut tp = Transcript::new(b"ef-lookup-split");
        let wproof = prove_fractional(wn, wd, &mut tp);
        let tproof = prove_fractional(tn, td, &mut tp);
        let mut tv = Transcript::new(b"ef-lookup-split");
        let wc = verify_fractional_chain_core(&wproof, wt, &mut tv).unwrap();
        let tc = verify_fractional_chain_core(&tproof, tt, &mut tv).unwrap();
        assert_eq!(wc.final_num * tc.final_den, tc.final_num * wc.final_den,
            "witness/table LogUp roots must satisfy the split identity");

        // A zero root denominator is rejected (no vacuous 0/0).
        let zero_den = vec![Ef::ZERO; 4];
        let mut tp = Transcript::new(b"ef-logup-zero-den");
        let zp = prove_fractional(vec![Ef::ONE; 4], zero_den, &mut tp);
        let mut tv = Transcript::new(b"ef-logup-zero-den");
        assert!(verify_fractional_chain_core(&zp, 2, &mut tv).is_none());
    }

    #[test]
    fn fractional_ef_roundtrip_and_tamper() {
        let mut rng = XorShift64::new(0x9301);
        let t = 5usize;
        let n = 1usize << t;
        let num: Vec<Ef> = (0..n).map(|_| ef_rng(&mut rng)).collect();
        let den: Vec<Ef> = (0..n).map(|_| {
            let d = ef_rng(&mut rng);
            if d == Ef::ZERO { Ef::ONE } else { d }
        }).collect();
        // True ratio of the tree root.
        let mut tn = vec![num.clone()];
        let mut td = vec![den.clone()];
        while tn.last().unwrap().len() > 1 {
            let half = tn.last().unwrap().len() / 2;
            let (nl, nh) = tn.last().unwrap().split_at(half);
            let (dl, dh) = td.last().unwrap().split_at(half);
            tn.push((0..half).map(|i| nl[i] * dh[i] + nh[i] * dl[i]).collect());
            td.push((0..half).map(|i| dl[i] * dh[i]).collect());
        }
        let claimed = tn[t][0] * td[t][0].inverse();
        let mut tp = Transcript::new(b"ef-fractional");
        let proof = prove_fractional(num, den, &mut tp);
        let mut tv = Transcript::new(b"ef-fractional");
        let claims = verify_fractional_chain_core(&proof, t, &mut tv).expect("honest EF chain");
        assert_eq!(claims.final_num, tn[t][0]);
        assert_eq!(claims.final_den, td[t][0]);
        assert_eq!(claims.final_num, claimed * claims.final_den);
        let (qn, vn, qd, vd) = fractional_leaf(&proof);
        assert_eq!(qn.len(), claims.q_num.len());
        assert_eq!(qd.len(), claims.q_den.len());
        let _ = (vn, vd);
        let mut bad = proof;
        bad.layers[0].gamma += Ef::ONE;
        let mut tv = Transcript::new(b"ef-fractional");
        assert!(verify_fractional_chain_core(&bad, t, &mut tv).is_none());
    }
}
