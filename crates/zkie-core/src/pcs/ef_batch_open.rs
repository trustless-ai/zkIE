//! S9: multi-point batch-opening verifier over EF challenge points.
//!
//! The committed tensors and Merkle roots stay base-field. The batch
//! `sum-of-products` arithmetic, the Fiat–Shamir γ, the evaluation point and
//! the opened evaluations live in `Ef`. The PCS boundary opens the base-field
//! MLEs at the EF point (`Whir::open_batch_multi_ef` / `verify_batch_multi_ef`).

use crate::common::ef_batch::{self, EfSumcheckProofBatch};
use crate::common::ef_outer::{self, eq_poly};
use crate::common::field::{Ef, Goldilocks, PrimeCharacteristicRing};
use crate::common::transcript::Transcript;
use crate::pcs::whir::{Commitment, OpeningProtocol, Proof, ProverData, Whir};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct EfBatchOpeningProof {
    pub sumcheck: EfSumcheckProofBatch,
    pub open: Proof,
    pub evals: Vec<Ef>,
}

/// Unique tables in first-appearance order plus each claim's group index.
fn claim_groups(table_idxs: &[usize]) -> (Vec<usize>, Vec<usize>) {
    // First-appearance order, O(n log n): map table id to its group slot.
    let mut index_of = std::collections::BTreeMap::new();
    let mut unique = Vec::new();
    let mut group_of = Vec::with_capacity(table_idxs.len());
    for &t in table_idxs {
        match index_of.get(&t) {
            Some(&i) => group_of.push(i),
            None => {
                let i = unique.len();
                unique.push(t);
                index_of.insert(t, i);
                group_of.push(i);
            }
        }
    }
    (unique, group_of)
}

/// Build the per-unique-table equality weights `W_t(X) = sum_i gamma^i eq(r_i,X)`
/// over the global claim order. When all claims of one table share a coordinate
/// prefix, fold that prefix once and combine it with the gamma-weighted suffix
/// sum (outer product), instead of materializing one full eq vector per claim.
fn build_batch_weights(
    unique: usize,
    table_len: usize,
    group_of: &[usize],
    points: &[Vec<Ef>],
    gamma: Ef,
) -> Result<Vec<Vec<Ef>>, String> {
    let n = points.len();
    if group_of.len() != n || n == 0 {
        return Err("EF batch-open grouping mismatch".to_owned());
    }
    let d = points[0].len();
    if d >= usize::BITS as usize || points.iter().any(|p| p.len() != d)
        || table_len != 1usize.checked_shl(d as u32).ok_or_else(|| "EF batch-open arity overflow".to_owned())?
    {
        return Err("EF batch-open point arity mismatch".to_owned());
    }
    let mut gamma_pows = Vec::with_capacity(n);
    let mut gp = Ef::ONE;
    for _ in 0..n { gamma_pows.push(gp); gp = gp * gamma; }

    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); unique];
    for (i, &g) in group_of.iter().enumerate() {
        groups.get_mut(g).ok_or_else(|| "EF batch-open group out of range".to_owned())?.push(i);
    }
    let mut weights: Vec<Vec<Ef>> = (0..unique).map(|_| vec![Ef::ZERO; table_len]).collect();
    for (g, idxs) in groups.iter().enumerate() {
        if idxs.is_empty() { continue; }
        let mut lcp = 0usize;
        while lcp < d {
            let c = points[idxs[0]][lcp];
            if idxs.iter().any(|&i| points[i][lcp] != c) { break; }
            lcp += 1;
        }
        if idxs.len() > 1 && lcp > 0 {
            let pref_len = 1usize.checked_shl(lcp as u32).ok_or_else(|| "EF batch-open prefix overflow".to_owned())?;
            let eq_pref = ef_outer::eq_evals(&points[idxs[0]][..lcp]);
            if eq_pref.len() != pref_len { return Err("EF batch-open eq length mismatch".to_owned()); }
            let suf_len = table_len / pref_len;
            let mut suffix = vec![Ef::ZERO; suf_len];
            for &i in idxs {
                let eq = ef_outer::eq_evals(&points[i][lcp..]);
                if eq.len() != suf_len { return Err("EF batch-open eq length mismatch".to_owned()); }
                let scale = gamma_pows[i];
                for (acc, &e) in suffix.iter_mut().zip(eq.iter()) {
                    *acc = *acc + scale * e;
                }
            }
            let w = &mut weights[g];
            for (s, &sv) in suffix.iter().enumerate() {
                let base = s << lcp;
                for (acc, &pv) in w[base..base + pref_len].iter_mut().zip(eq_pref.iter()) {
                    *acc = *acc + pv * sv;
                }
            }
        } else {
            let w = &mut weights[g];
            for &i in idxs {
                let eq = ef_outer::eq_evals(&points[i]);
                if eq.len() != table_len { return Err("EF batch-open eq length mismatch".to_owned()); }
                let scale = gamma_pows[i];
                for (acc, &e) in w.iter_mut().zip(eq.iter()) {
                    *acc = *acc + scale * e;
                }
            }
        }
    }
    Ok(weights)
}

/// Prove a multi-point batch opening at EF points. `values` holds exactly one
/// base-field tensor per unique table, in first-appearance order.
#[allow(clippy::too_many_arguments)]
pub fn prove_batch_open_committed_ef(
    whir: &Whir,
    commitment: &Commitment,
    protocol: &OpeningProtocol,
    num_tables: usize,
    prover_data: ProverData,
    table_idxs: &[usize],
    values: Vec<Vec<Goldilocks>>,
    points: &[Vec<Ef>],
    claimed: &[Ef],
) -> Result<EfBatchOpeningProof, String> {
    let refs: Vec<&[Goldilocks]> = values.iter().map(Vec::as_slice).collect();
    prove_batch_open_committed_ef_slices(whir, commitment, protocol, num_tables,
        prover_data, table_idxs, &refs, points, claimed)
}

/// Borrowed-column variant used by pooled openings: no base-field column
/// copies; each unique column is converted to EF once for the sumcheck.
#[allow(clippy::too_many_arguments)]
pub fn prove_batch_open_committed_ef_slices(
    whir: &Whir,
    commitment: &Commitment,
    protocol: &OpeningProtocol,
    num_tables: usize,
    prover_data: ProverData,
    table_idxs: &[usize],
    values: &[&[Goldilocks]],
    points: &[Vec<Ef>],
    claimed: &[Ef],
) -> Result<EfBatchOpeningProof, String> {
    let n = points.len();
    if table_idxs.len() != n || claimed.len() != n || n == 0 {
        return Err("EF batch-open length mismatch".to_owned());
    }
    let (unique, group_of) = claim_groups(table_idxs);
    if values.len() != unique.len() {
        return Err("EF batch-open unique table mismatch".to_owned());
    }
    let d = points[0].len();
    if points.iter().any(|p| p.len() != d) {
        return Err("EF batch-open point arity mismatch".to_owned());
    }
    let table_len = 1usize
        .checked_shl(d as u32)
        .ok_or_else(|| "EF batch-open arity overflow".to_owned())?;
    if values.iter().any(|v| v.len() != table_len) {
        return Err("EF batch-open tensor/point arity mismatch".to_owned());
    }
    if table_idxs.iter().any(|&t| t >= num_tables) {
        return Err("EF batch-open table index out of range".to_owned());
    }

    let mut t = Transcript::new(b"zkie-batch-open");
    t.absorb(&bincode::serialize(commitment).map_err(|e| e.to_string())?);
    for (pt, c) in points.iter().zip(claimed.iter()) {
        t.absorb_efs(pt);
        t.absorb_ef(*c);
    }
    let gamma = t.ef_field();

    let trace = std::env::var("ZKIE_TRACE_PROVE").is_ok();
    let t_w = std::time::Instant::now();
    // W_t for each unique table. Claims of one table that share a coordinate
    // prefix (the expert-major weight layout has all expert claims on the
    // same inner (n, k) point) reuse that prefix's eq vector: W_t is the
    // outer product of eq(prefix) with the gamma-weighted suffix sum. This
    // lowers generation from O(claims x 2^d) to O(2^d) for those groups; the
    // values are identical because field arithmetic is exact.
    let weights = build_batch_weights(unique.len(), table_len, &group_of, points, gamma)?;
    let mut gamma_pow = Ef::ONE;
    let mut claimed_sum = Ef::ZERO;
    for i in 0..n {
        claimed_sum = claimed_sum + gamma_pow * claimed[i];
        gamma_pow = gamma_pow * gamma;
    }
    if trace {
        eprintln!("BATCH weights_ms={:.1} claims={} unique={} table_len={}",
            t_w.elapsed().as_secs_f64() * 1e3, n, unique.len(), table_len);
    }
    let t_v = std::time::Instant::now();
    let values_ef: Vec<Vec<Ef>> = values
        .iter()
        .map(|v| v.iter().map(|&x| Ef::from(x)).collect())
        .collect();
    if trace {
        eprintln!("BATCH values_ef_ms={:.1}", t_v.elapsed().as_secs_f64() * 1e3);
    }
    let t_s = std::time::Instant::now();
    let (sumcheck, challenges) =
        ef_batch::prove_sum_of_products_owned(weights, values_ef, claimed_sum, &mut t);
    if trace {
        eprintln!("BATCH sumcheck_ms={:.1}", t_s.elapsed().as_secs_f64() * 1e3);
    }
    let t_o = std::time::Instant::now();
    let (open, evals) = whir.open_batch_multi_ef(prover_data, protocol, num_tables, &challenges);
    if trace {
        eprintln!("BATCH whir_open_ms={:.1} num_tables={}",
            t_o.elapsed().as_secs_f64() * 1e3, num_tables);
    }
    Ok(EfBatchOpeningProof { sumcheck, open, evals })
}

/// Verify an EF multi-point batch opening. Reads no committed tensor data.
#[allow(clippy::too_many_arguments)]
pub fn verify_batch_open_committed_ef(
    whir: &Whir,
    commitment: &Commitment,
    protocol: &OpeningProtocol,
    num_tables: usize,
    table_idxs: &[usize],
    points: &[Vec<Ef>],
    claimed: &[Ef],
    proof: &EfBatchOpeningProof,
) -> bool {
    let n = table_idxs.len();
    if n != points.len() || n != claimed.len() || n == 0 {
        return false;
    }
    let d = points[0].len();
    if points.iter().any(|p| p.len() != d) {
        return false;
    }
    if table_idxs.iter().any(|&t| t >= num_tables) {
        return false;
    }
    let (unique, group_of) = claim_groups(table_idxs);

    let mut t = Transcript::new(b"zkie-batch-open");
    let Ok(commitment_bytes) = bincode::serialize(commitment) else { return false };
    t.absorb(&commitment_bytes);
    for (pt, c) in points.iter().zip(claimed.iter()) {
        t.absorb_efs(pt);
        t.absorb_ef(*c);
    }
    let gamma = t.ef_field();

    let mut claimed_sum = Ef::ZERO;
    let mut gamma_pow = Ef::ONE;
    for i in 0..n {
        claimed_sum = claimed_sum + gamma_pow * claimed[i];
        gamma_pow = gamma_pow * gamma;
    }
    let (rounds_ok, challenges) =
        ef_batch::verify_sum_of_products(&proof.sumcheck, claimed_sum, d, &mut t);
    if !rounds_ok {
        return false;
    }
    let evals = match whir.verify_batch_multi_ef(commitment, &proof.open, protocol, num_tables, &challenges) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if evals != proof.evals {
        return false;
    }
    let mut w_at_r = vec![Ef::ZERO; unique.len()];
    let mut gamma_pow = Ef::ONE;
    for i in 0..n {
        let Some(eq) = eq_poly(&points[i], &challenges) else { return false };
        w_at_r[group_of[i]] = w_at_r[group_of[i]] + gamma_pow * eq;
        gamma_pow = gamma_pow * gamma;
    }
    let mut expected = Ef::ZERO;
    for g in 0..unique.len() {
        expected = expected + w_at_r[g] * evals[unique[g]];
    }
    proof.sumcheck.b_eval == expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::XorShift64;
    use crate::pcs::whir::Whir;

    fn ef_eval_at(values: &[Goldilocks], point: &[Ef]) -> Ef {
        // EF evaluation of a base-field MLE via the core helper.
        crate::common::ef_outer::mle_eval_at_ef(values, point).unwrap()
    }

    #[test]
    fn shared_prefix_weights_match_direct() {
        let mut rng = XorShift64::new(0x9402);
        let d = 8usize;
        let gamma = Ef::new([rng.field(), rng.field()]);
        let shared: Vec<Ef> = (0..5).map(|_| Ef::new([rng.field(), rng.field()])).collect();
        let mut points: Vec<Vec<Ef>> = Vec::new();
        for e in 0..4usize {
            let mut p = shared.clone();
            for b in 0..3 {
                p.push(if (e >> b) & 1 == 1 { Ef::ONE } else { Ef::ZERO });
            }
            points.push(p);
        }
        let group_of = vec![0usize; points.len()];
        let got = build_batch_weights(1, 1 << d, &group_of, &points, gamma).unwrap();
        let mut want = vec![Ef::ZERO; 1 << d];
        let mut gp = Ef::ONE;
        for p in &points {
            let eq = ef_outer::eq_evals(p);
            for (a, &e) in want.iter_mut().zip(eq.iter()) { *a = *a + gp * e; }
            gp = gp * gamma;
        }
        assert_eq!(got[0], want);
        // All claims at one point: the whole point is the shared prefix.
        let same = vec![points[0].clone(); 5];
        let got_same = build_batch_weights(1, 1 << d, &vec![0usize; 5], &same, gamma).unwrap();
        let mut gp = Ef::ONE;
        let mut sum = Ef::ZERO;
        for _ in 0..5 { sum = sum + gp; gp = gp * gamma; }
        let eq = ef_outer::eq_evals(&points[0]);
        for (a, &e) in got_same[0].iter().zip(eq.iter()) { assert_eq!(*a, sum * e); }
        // No shared prefix: two tables, claims interleaved; fallback path.
        let mut p2 = points.clone();
        p2[0][0] = Ef::new([rng.field(), rng.field()]);
        p2[2][1] = Ef::new([rng.field(), rng.field()]);
        let group_of2 = vec![0usize, 1, 0, 1];
        let got2 = build_batch_weights(2, 1 << d, &group_of2, &p2, gamma).unwrap();
        for g in 0..2 {
            let mut want = vec![Ef::ZERO; 1 << d];
            let mut gp = Ef::ONE;
            for (i, p) in p2.iter().enumerate() {
                if group_of2[i] == g {
                    let eq = ef_outer::eq_evals(p);
                    for (a, &e) in want.iter_mut().zip(eq.iter()) { *a = *a + gp * e; }
                }
                gp = gp * gamma;
            }
            assert_eq!(got2[g], want);
        }
    }

    #[test]
    fn ef_batch_open_roundtrip_and_tamper() {
        let mut rng = XorShift64::new(0x9401);
        let d = 5usize;
        let n = 3usize;
        let tables: Vec<Vec<Goldilocks>> = (0..n)
            .map(|_| (0..1 << d).map(|_| rng.field()).collect())
            .collect();
        let refs: Vec<&[Goldilocks]> = tables.iter().map(Vec::as_slice).collect();
        let base = Whir::new_testing(d);
        let (commitment, prover_data, protocol, batch_whir) = base.commit_batch(&refs);
        let point: Vec<Ef> = (0..d).map(|_| Ef::new([rng.field(), rng.field()])).collect();
        let claimed: Vec<Ef> = tables.iter().map(|t| ef_eval_at(t, &point)).collect();
        let table_idxs = vec![0usize, 1, 2, 0];
        let points = vec![point.clone(), point.clone(), point.clone(), point.clone()];
        let claimed = vec![claimed[0], claimed[1], claimed[2], claimed[0]];
        let values = vec![tables[0].clone(), tables[1].clone(), tables[2].clone()];
        let proof = prove_batch_open_committed_ef(
            &batch_whir, &commitment, &protocol, n, prover_data,
            &table_idxs, values, &points, &claimed,
        ).expect("prove EF batch open");
        assert!(verify_batch_open_committed_ef(
            &batch_whir, &commitment, &protocol, n, &table_idxs, &points, &claimed, &proof));
        assert_eq!(proof.evals.len(), n);
        // The WHIR-level single-table EF open/verify roundtrip is the direct
        // EF-point opening: it must equal the EF MLE evaluation of the F table.
        let (single_commit, single_pd, single_protocol) = base.commit(&tables[0]);
        let (op, opened) = base.open_ef(single_pd, &single_protocol, &point);
        assert_eq!(opened, ef_eval_at(&tables[0], &point));
        assert_eq!(base.verify_ef(&single_commit, &op, &single_protocol, &point).unwrap(), opened);

        let mut bad = proof.clone();
        bad.sumcheck.rounds[0].c0 += Ef::ONE;
        assert!(!verify_batch_open_committed_ef(
            &batch_whir, &commitment, &protocol, n, &table_idxs, &points, &claimed, &bad));
        let mut bad2 = proof.clone();
        bad2.evals[0] += Ef::ONE;
        assert!(!verify_batch_open_committed_ef(
            &batch_whir, &commitment, &protocol, n, &table_idxs, &points, &claimed, &bad2));
    }
}
