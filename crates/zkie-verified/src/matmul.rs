//! Reusable committed weights + sequential sumcheck + a WHIR terminal opening.
//! Verification never receives weights or intermediate activations.
use crate::{
    backend::F,
    model::signed,
    whir_backend::{self as wb, Challenge as E},
    Error,
};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_sumcheck::PrescribedPointPcs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::panic::{catch_unwind, AssertUnwindSafe};

const DOMAIN: &[u8] = b"zkie/committed-matmul/v2\0";
const MAGIC: &[u8; 8] = b"ZKIEMM02";
const LIMIT: usize = 1 << 20;
const PROOF_LIMIT: usize = 4 << 20;

#[derive(Clone, Serialize, Deserialize)]
pub struct VerifierKey {
    version: u32,
    k: usize,
    n: usize,
    root: wb::Commitment,
}

impl VerifierKey {
    pub fn dimensions(&self) -> (usize, usize) {
        (self.k, self.n)
    }
    fn validate(&self) -> Result<(), Error> {
        if self.root.num_roots() != 1
            || self.version != 2
            || !self.k.is_power_of_two()
            || !self.n.is_power_of_two()
            || self.k.checked_mul(self.n).is_none_or(|n| n > LIMIT)
        {
            return Err(Error::InvalidModel);
        }
        Ok(())
    }
    fn vars(&self) -> usize {
        (self.k * self.n).trailing_zeros().max(5) as usize
    }
    pub fn id(&self) -> Result<[u8; 32], Error> {
        self.validate()?;
        let mut h = Sha256::new();
        h.update(DOMAIN);
        h.update(self.to_bytes()?);
        Ok(h.finalize().into())
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        self.validate()?;
        postcard::to_allocvec(self).map_err(|_| Error::ProofEncoding)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > 4096 {
            return Err(Error::ProofTooLarge);
        }
        let (key, tail): (Self, _) =
            postcard::take_from_bytes(bytes).map_err(|_| Error::ProofEncoding)?;
        key.validate()?;
        if !tail.is_empty() || key.to_bytes()? != bytes {
            return Err(Error::ProofEncoding);
        }
        Ok(key)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statement {
    pub m: usize,
    pub input: Vec<i32>,
    pub output: Vec<i64>,
    pub context: [u8; 32],
}

fn validate_statement(key: &VerifierKey, s: &Statement) -> Result<(), Error> {
    key.validate()?;
    if !s.m.is_power_of_two()
        || s.m
            .checked_mul(key.k)
            .is_none_or(|n| n > LIMIT || n != s.input.len())
        || s.m
            .checked_mul(key.n)
            .is_none_or(|n| n > LIMIT || n != s.output.len())
        || s.input.iter().any(|x| !(-32767..=32767).contains(x))
        || s.output
            .iter()
            .any(|y| y.unsigned_abs() > key.k as u64 * 32767 * 32767)
    {
        return Err(Error::InvalidStatement);
    }
    Ok(())
}

fn field64(x: i64) -> F {
    if x < 0 {
        F::ZERO - F::from_u64(x.unsigned_abs())
    } else {
        F::from_u64(x as u64)
    }
}

fn transcript(key: &VerifierKey, s: &Statement, pcs: &wb::Pcs) -> Result<wb::Challenger, Error> {
    let mut h = Sha256::new();
    h.update(DOMAIN);
    h.update(key.id()?);
    h.update(postcard::to_allocvec(s).map_err(|_| Error::ProofEncoding)?);
    let mut c = wb::challenger(pcs, &h.finalize());
    // PrescribedPointPcs requires the caller to absorb the commitment before
    // deriving the prescribed point. Do this explicitly, not just via its id.
    c.observe(key.root.clone());
    Ok(c)
}

fn observe(c: &mut wb::Challenger, values: &[E]) {
    for value in values {
        c.observe_slice(<E as BasedVectorSpace<F>>::as_basis_coefficients_slice(
            value,
        ));
    }
}

fn challenges(c: &mut wb::Challenger, bits: usize) -> Vec<E> {
    (0..bits).map(|_| c.sample_algebra_element()).collect()
}

fn fold(values: &mut Vec<E>, r: E) {
    for i in 0..values.len() / 2 {
        values[i] = values[2 * i] + r * (values[2 * i + 1] - values[2 * i]);
    }
    values.truncate(values.len() / 2);
}

fn eval(values: impl IntoIterator<Item = F>, point: &[E]) -> E {
    let mut v: Vec<E> = values.into_iter().map(E::from).collect();
    for &r in point {
        fold(&mut v, r);
    }
    v[0]
}

fn polynomial(p: &[E; 3], r: E) -> E {
    // Lagrange interpolation at 0,1,2, not prover-selected coefficients.
    let half = E::from(F::from_u64(2).inverse());
    p[0] + r * (p[1] - p[0]) + r * (r - E::ONE) * (p[2] - p[1] - p[1] + p[0]) * half
}

#[derive(Serialize, Deserialize)]
struct Proof {
    rounds: Vec<[E; 3]>,
    opening: wb::Opening,
}

pub fn verify(
    key: &VerifierKey,
    trusted_id: [u8; 32],
    s: &Statement,
    bytes: &[u8],
) -> Result<(), Error> {
    validate_statement(key, s)?;
    if key.id()? != trusted_id {
        return Err(Error::ModelIdentity);
    }
    if bytes.len() > PROOF_LIMIT {
        return Err(Error::ProofTooLarge);
    }
    if !bytes.starts_with(MAGIC) {
        return Err(Error::ProofEncoding);
    }
    catch_unwind(AssertUnwindSafe(|| {
        let payload = &bytes[MAGIC.len()..];
        let (proof, tail): (Proof, _) =
            postcard::take_from_bytes(payload).map_err(|_| Error::ProofEncoding)?;
        if !tail.is_empty()
            || proof.rounds.len() != key.k.trailing_zeros() as usize
            || postcard::to_allocvec(&proof).map_err(|_| Error::ProofEncoding)? != payload
        {
            return Err(Error::ProofEncoding);
        }
        let pcs = wb::pcs(key.vars())?;
        let mut c = transcript(key, s, &pcs)?;
        let u = challenges(&mut c, s.m.trailing_zeros() as usize);
        let v = challenges(&mut c, key.n.trailing_zeros() as usize);
        let mut output_point = v.clone();
        output_point.extend_from_slice(&u);
        let mut claim = eval(s.output.iter().copied().map(field64), &output_point);
        let mut r = Vec::new();
        for p in &proof.rounds {
            if p[0] + p[1] != claim {
                return Err(Error::InvalidProof);
            }
            observe(&mut c, p);
            let challenge: E = c.sample_algebra_element();
            claim = polynomial(p, challenge);
            r.push(challenge);
        }
        let mut a_point = r.clone();
        a_point.extend_from_slice(&u);
        let a_eval = eval(s.input.iter().copied().map(signed), &a_point);
        let mut w_point = v;
        w_point.extend_from_slice(&r);
        let evaluations = pcs
            .verify_at(
                &key.root,
                &proof.opening,
                &wb::protocol(key.vars()),
                &[wb::point(&w_point, key.vars())],
                &mut c,
            )
            .map_err(|_| Error::InvalidProof)?;
        let w_eval = evaluations
            .first()
            .and_then(|e| e.current().first())
            .ok_or(Error::InvalidProof)?;
        if claim != a_eval * *w_eval {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }))
    .unwrap_or(Err(Error::InvalidProof))
}

#[cfg(feature = "prover")]
pub struct ProvingKey {
    key: VerifierKey,
    weights: Vec<i32>,
    data: wb::Data,
}

#[cfg(feature = "prover")]
pub fn prepare(k: usize, n: usize, weights: Vec<i32>) -> Result<(ProvingKey, VerifierKey), Error> {
    use p3_commit::MultilinearPcs;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_sumcheck::layout::{Layout, SuffixProver, Table};
    if !k.is_power_of_two()
        || !n.is_power_of_two()
        || k.checked_mul(n)
            .is_none_or(|len| len > LIMIT || len != weights.len())
        || weights.iter().any(|w| !(-32767..=32767).contains(w))
    {
        return Err(Error::InvalidModel);
    }
    let vars = (k * n).trailing_zeros().max(5) as usize;
    let pcs = wb::pcs(vars)?;
    let mut values: Vec<F> = weights.iter().copied().map(signed).collect();
    values.resize(1 << vars, F::ZERO);
    let witness = SuffixProver::<F, E>::new_witness(
        vec![Table::new(RowMajorMatrix::new(values, 1 << vars))],
        5,
    );
    let (root, data) = pcs.commit(witness, &mut wb::challenger(&pcs, DOMAIN));
    let key = VerifierKey {
        version: 2,
        k,
        n,
        root,
    };
    Ok((
        ProvingKey {
            key: key.clone(),
            weights,
            data,
        },
        key,
    ))
}

#[cfg(feature = "prover")]
impl ProvingKey {
    pub fn prove(
        &self,
        m: usize,
        input: Vec<i32>,
        context: [u8; 32],
    ) -> Result<(Statement, Vec<u8>), Error> {
        let k = self.key.k;
        let n = self.key.n;
        // Validate dimensions and range before any indexing/allocation or dot products.
        let mut s = Statement {
            m,
            input,
            output: vec![],
            context,
        };
        if m.checked_mul(n).is_none_or(|len| len > LIMIT) {
            return Err(Error::InvalidStatement);
        }
        s.output.resize(m * n, 0);
        validate_statement(&self.key, &s)?;
        for i in 0..m {
            for j in 0..n {
                s.output[i * n + j] = (0..k)
                    .map(|t| i64::from(s.input[i * k + t]) * i64::from(self.weights[t * n + j]))
                    .sum();
            }
        }
        let pcs = wb::pcs(self.key.vars())?;
        let mut c = transcript(&self.key, &s, &pcs)?;
        let u = challenges(&mut c, m.trailing_zeros() as usize);
        let v = challenges(&mut c, n.trailing_zeros() as usize);
        let mut a: Vec<E> = (0..k)
            .map(|t| eval((0..m).map(|i| signed(s.input[i * k + t])), &u))
            .collect();
        let mut b: Vec<E> = (0..k)
            .map(|t| {
                eval(
                    self.weights[t * n..(t + 1) * n].iter().copied().map(signed),
                    &v,
                )
            })
            .collect();
        let mut rounds = Vec::new();
        let mut r = Vec::new();
        while a.len() > 1 {
            let mut p = [E::ZERO; 3];
            for i in 0..a.len() / 2 {
                for (z, total) in p.iter_mut().enumerate() {
                    let z = E::from(F::from_usize(z));
                    *total += (a[2 * i] + z * (a[2 * i + 1] - a[2 * i]))
                        * (b[2 * i] + z * (b[2 * i + 1] - b[2 * i]));
                }
            }
            observe(&mut c, &p);
            let ch: E = c.sample_algebra_element();
            fold(&mut a, ch);
            fold(&mut b, ch);
            rounds.push(p);
            r.push(ch);
        }
        let mut point = v;
        point.extend_from_slice(&r);
        let opening = pcs.open_at(
            self.data.clone(),
            &wb::protocol(self.key.vars()),
            &[wb::point(&point, self.key.vars())],
            &mut c,
        );
        let proof = Proof { rounds, opening };
        let mut bytes = MAGIC.to_vec();
        bytes.extend(postcard::to_allocvec(&proof).map_err(|_| Error::ProofEncoding)?);
        if bytes.len() > PROOF_LIMIT {
            return Err(Error::ProofTooLarge);
        }
        Ok((s, bytes))
    }
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::*;
    #[test]
    fn reusable_key_verifies_without_weights_across_shapes() {
        for (m, k, n) in [(1, 1, 1), (2, 4, 2), (4, 8, 4), (1, 32, 8)] {
            let weights = (0..k * n).map(|i| i as i32 % 13 - 6).collect();
            let (pk, key) = prepare(k, n, weights).unwrap();
            let id = key.id().unwrap();
            let serialized = key.to_bytes().unwrap();
            assert!(serialized.len() < 256);
            let vk = VerifierKey::from_bytes(&serialized).unwrap();
            for seed in 0..2 {
                let (s, bytes) = pk
                    .prove(
                        m,
                        (0..m * k).map(|i| i as i32 % 7 - 3).collect(),
                        [seed; 32],
                    )
                    .unwrap();
                verify(&vk, id, &s, &bytes).unwrap();
                let mut bad = s.clone();
                bad.output[0] += 1;
                assert!(verify(&vk, id, &bad, &bytes).is_err());
                bad = s.clone();
                bad.context[0] ^= 1;
                assert!(verify(&vk, id, &bad, &bytes).is_err());
                bad = s.clone();
                bad.input[0] += 1;
                assert!(verify(&vk, id, &bad, &bytes).is_err());
                let mut proof: Proof = postcard::from_bytes(&bytes[8..]).unwrap();
                if let Some(round) = proof.rounds.first_mut() {
                    round[0] += E::ONE;
                    let mut forged = MAGIC.to_vec();
                    forged.extend(postcard::to_allocvec(&proof).unwrap());
                    assert!(verify(&vk, id, &s, &forged).is_err());
                }
            }
        }
    }
    #[test]
    fn rejects_shapes_keys_and_truncated_messages() {
        assert!(prepare(3, 2, vec![0; 6]).is_err());
        assert!(prepare(2, 2, vec![32768; 4]).is_err());
        let (pk, key) = prepare(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert!(pk.prove(3, vec![0; 6], [0; 32]).is_err());
        let (s, bytes) = pk.prove(2, vec![1, 2, 3, 4], [0; 32]).unwrap();
        for cut in [0, 7, 8, 16, bytes.len() - 1] {
            assert!(verify(&key, key.id().unwrap(), &s, &bytes[..cut]).is_err());
        }
        assert!(verify(&key, [0; 32], &s, &bytes).is_err());
        let (_, other) = prepare(2, 2, vec![4, 3, 2, 1]).unwrap();
        assert!(verify(&other, other.id().unwrap(), &s, &bytes).is_err());
    }
}
