#[cfg(feature = "prover")]
use crate::backend::F;
// Committed multi-layer fixed-point inference. No activation or weight arrays
// are passed to verification. All constraints are derived from the pinned key.
use crate::{
    model::signed,
    oracle::{self as o, Coordinate as C, Factor as X, Relation, Tensor, View},
    whir_backend::{self as wb, Challenge as E},
    Error,
};
use p3_challenger::CanObserve;
use p3_field::{Field, PrimeCharacteristicRing};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::panic::{catch_unwind, AssertUnwindSafe};

const DOMAIN: &[u8] = b"zkie/committed-dense/v2\0";
const MAGIC: &[u8; 8] = b"ZKIEDN02";
#[derive(Clone, Serialize, Deserialize)]
struct LayerKey {
    k: usize,
    n: usize,
    shift: u32,
    relu: bool,
    w: Tensor,
    b: Tensor,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct VerifierKey {
    version: u32,
    layers: Vec<LayerKey>,
}
impl VerifierKey {
    fn validate(&self) -> Result<(), Error> {
        if self.version != 2 || self.layers.is_empty() || self.layers.len() > 32 {
            return Err(Error::InvalidModel);
        }
        let mut previous = self.layers[0].k;
        for l in &self.layers {
            l.w.validate()?;
            l.b.validate()?;
            if !l.k.is_power_of_two()
                || !l.n.is_power_of_two()
                || l.k > 65536
                || l.k != previous
                || l.n > 65536
                || l.k
                    .checked_mul(l.n)
                    .is_none_or(|v| v > (1 << 20) || v != l.w.len)
                || l.b.len != l.n
                || !(1..=16).contains(&l.shift)
            {
                return Err(Error::InvalidModel);
            }
            previous = l.n;
        }
        Ok(())
    }
    pub fn id(&self) -> Result<[u8; 32], Error> {
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
        if bytes.len() > (1 << 16) {
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
    pub output: Vec<i32>,
    pub context: [u8; 32],
}
fn validate_statement(key: &VerifierKey, s: &Statement) -> Result<(), Error> {
    key.validate()?;
    if !s.m.is_power_of_two()
        || s.m
            .checked_mul(key.layers[0].k)
            .is_none_or(|n| n > (1 << 20) || n != s.input.len())
        || s.m
            .checked_mul(key.layers.last().unwrap().n)
            .is_none_or(|n| n > (1 << 20) || n != s.output.len())
        || key
            .layers
            .iter()
            .any(|l| s.m.checked_mul(l.n).is_none_or(|n| n > (1 << 20)))
    {
        return Err(Error::InvalidStatement);
    }
    Ok(())
}
#[derive(Serialize, Deserialize)]
struct LayerProof {
    matmul: o::MatmulProof,
    relations: Vec<o::RelationProof>,
}
#[derive(Serialize, Deserialize)]
struct Proof {
    activations: Vec<Tensor>,
    input: wb::Opening,
    layers: Vec<LayerProof>,
    output: wb::Opening,
}

fn transcript(
    key: &VerifierKey,
    s: &Statement,
    tensors: &[Tensor],
) -> Result<wb::Challenger, Error> {
    let mut h = Sha256::new();
    h.update(DOMAIN);
    h.update(key.id()?);
    h.update(postcard::to_allocvec(s).map_err(|_| Error::ProofEncoding)?);
    h.update(postcard::to_allocvec(tensors).map_err(|_| Error::ProofEncoding)?);
    let mut c = wb::challenger(&wb::pcs(5)?, &h.finalize());
    for t in tensors {
        t.validate()?;
        c.observe(t.root.clone());
    }
    Ok(c)
}
struct Ids {
    input: usize,
    w: usize,
    b: usize,
    raw: usize,
    q: usize,
    rem: usize,
    qbits: usize,
    rbits: usize,
    out: usize,
}
fn ids(index: usize, start: usize, input: usize, relu: bool) -> Ids {
    Ids {
        input,
        w: index * 2,
        b: index * 2 + 1,
        raw: start,
        q: start + 1,
        rem: start + 2,
        qbits: start + 3,
        rbits: start + 4,
        out: if relu { start + 5 } else { start + 1 },
    }
}
fn factors(ids: &[usize], log: usize) -> Vec<X> {
    ids.iter()
        .map(|&i| X::Oracle(View::identity(i, log)))
        .collect()
}
pub(crate) fn relation(
    log: usize,
    mut fs: Vec<X>,
    terms: Vec<(E, Vec<usize>)>,
    c: &mut wb::Challenger,
) -> Relation {
    let eq = fs.len();
    fs.push(X::Eq {
        point: o::sample(c, log),
        start: 0,
    });
    Relation {
        log,
        factors: fs,
        terms: terms
            .into_iter()
            .map(|(a, mut vars)| {
                vars.push(eq);
                (a, vars)
            })
            .collect(),
    }
}
pub(crate) fn range(
    tensor: usize,
    bits: usize,
    width: usize,
    offset: u64,
    log: usize,
    c: &mut wb::Challenger,
) -> Vec<Relation> {
    let columns = width.next_power_of_two();
    let column_bits = columns.trailing_zeros() as usize;
    let total = log + column_bits;
    let boolean = relation(
        total,
        factors(&[bits], total),
        vec![(E::ONE, vec![0, 0]), (-E::ONE, vec![0])],
        c,
    );
    let mut fs = factors(&[bits], total);
    fs.push(X::Oracle(View::identity(tensor, log)));
    fs.push(X::Powers {
        start: log,
        bits: column_bits,
    });
    fs.push(X::Eq {
        point: o::sample(c, log),
        start: 0,
    });
    let inv = E::from_usize(columns).inverse();
    let reconstruction = Relation {
        log: total,
        factors: fs,
        terms: vec![
            (E::ONE, vec![0, 2, 3]),
            (-inv, vec![1, 3]),
            (-E::from_u64(offset) * inv, vec![3]),
        ],
    };
    let mut out = vec![boolean, reconstruction];
    if width != columns {
        let fs = vec![
            X::Oracle(View::identity(bits, total)),
            X::High {
                start: log,
                bits: column_bits,
                boundary: width,
            },
        ];
        out.push(relation(total, fs, vec![(E::ONE, vec![0, 1])], c));
    }
    out
}
fn constraints(l: &LayerKey, id: &Ids, m: usize, c: &mut wb::Challenger) -> Vec<Relation> {
    let log = (m * l.n).trailing_zeros() as usize;
    let mut fs = factors(&[id.raw, id.q, id.rem], log);
    fs.push(X::Oracle(View::identity(
        id.b,
        l.n.trailing_zeros() as usize,
    )));
    let scale = E::from_u64(1u64 << l.shift);
    let round = relation(
        log,
        fs,
        vec![
            (E::ONE, vec![0]),
            (-scale, vec![1]),
            (-E::ONE, vec![2]),
            (scale, vec![3]),
            (E::from_u64(1u64 << (l.shift - 1)), vec![]),
        ],
        c,
    );
    let mut result = vec![round];
    result.extend(range(id.q, id.qbits, 32, 1u64 << 31, log, c));
    result.extend(range(id.rem, id.rbits, l.shift as usize, 0, log, c));
    if l.relu {
        let mut sign = View::identity(id.qbits, log);
        sign.coordinates.extend((0..5).map(|_| C::Bit(true)));
        let mut fs = factors(&[id.out, id.q], log);
        fs.push(X::Oracle(sign));
        result.push(relation(
            log,
            fs,
            vec![(E::ONE, vec![0]), (-E::ONE, vec![1, 2])],
            c,
        ));
    }
    result
}
fn tensor_layout(
    key: &VerifierKey,
    m: usize,
    activations: &[Tensor],
) -> Result<(Vec<Tensor>, Vec<Ids>), Error> {
    let mut tensors = Vec::new();
    for l in &key.layers {
        tensors.extend([l.w.clone(), l.b.clone()]);
    }
    let mut expected = vec![m * key.layers[0].k];
    let mut all_ids = Vec::new();
    let mut input = tensors.len();
    for (i, l) in key.layers.iter().enumerate() {
        let n = m * l.n;
        let start = tensors.len() + expected.len();
        let id = ids(i, start, input, l.relu);
        expected.extend([n, n, n, n * 32, n * (l.shift as usize).next_power_of_two()]);
        if l.relu {
            expected.push(n);
        }
        input = id.out;
        all_ids.push(id);
    }
    if activations.len() != expected.len()
        || activations.iter().zip(expected).any(|(a, n)| a.len != n)
    {
        return Err(Error::InvalidProof);
    }
    tensors.extend_from_slice(activations);
    for t in &tensors {
        t.validate()?;
    }
    Ok((tensors, all_ids))
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
    if bytes.len() > (64 << 20) {
        return Err(Error::ProofTooLarge);
    }
    if !bytes.starts_with(MAGIC) {
        return Err(Error::ProofEncoding);
    }
    catch_unwind(AssertUnwindSafe(|| {
        let (proof, tail): (Proof, _) =
            postcard::take_from_bytes(&bytes[8..]).map_err(|_| Error::ProofEncoding)?;
        if !tail.is_empty()
            || proof.layers.len() != key.layers.len()
            || postcard::to_allocvec(&proof).map_err(|_| Error::ProofEncoding)? != bytes[8..]
        {
            return Err(Error::ProofEncoding);
        }
        let (tensors, all_ids) = tensor_layout(key, s.m, &proof.activations)?;
        let mut c = transcript(key, s, &tensors)?;
        let input = key.layers.len() * 2;
        let point = o::sample(&mut c, tensors[input].len.trailing_zeros() as usize);
        if o::verify_open(&tensors[input], &point, &proof.input, &mut c)?
            != o::eval(s.input.iter().copied().map(signed), &point)
        {
            return Err(Error::InvalidProof);
        }
        for ((l, id), p) in key.layers.iter().zip(&all_ids).zip(&proof.layers) {
            o::verify_matmul(
                &tensors,
                [id.input, id.w, id.raw],
                [s.m, l.k, l.n],
                &p.matmul,
                &mut c,
            )?;
            let relations = constraints(l, id, s.m, &mut c);
            if relations.len() != p.relations.len() {
                return Err(Error::InvalidProof);
            }
            for (d, p) in relations.iter().zip(&p.relations) {
                o::verify_relation(d, &tensors, p, &mut c)?;
            }
        }
        let out = all_ids.last().unwrap().out;
        let point = o::sample(&mut c, tensors[out].len.trailing_zeros() as usize);
        if o::verify_open(&tensors[out], &point, &proof.output, &mut c)?
            != o::eval(s.output.iter().copied().map(signed), &point)
        {
            return Err(Error::InvalidProof);
        }
        Ok(())
    }))
    .unwrap_or(Err(Error::InvalidProof))
}

#[cfg(feature = "prover")]
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub k: usize,
    pub n: usize,
    pub shift: u32,
    pub relu: bool,
    pub weights: Vec<i32>,
    pub bias: Vec<i32>,
}
#[cfg(feature = "prover")]
pub struct ProvingKey {
    key: VerifierKey,
    data: Vec<std::sync::Arc<o::Data>>,
    layers: Vec<Layer>,
}
#[cfg(feature = "prover")]
pub fn prepare(layers: Vec<Layer>) -> Result<(ProvingKey, VerifierKey), Error> {
    let mut keys = Vec::new();
    let mut data = Vec::new();
    if layers.is_empty() || layers.len() > 32 {
        return Err(Error::InvalidModel);
    }
    for l in &layers {
        if l.k == 0
            || l.n == 0
            || l.k
                .checked_mul(l.n)
                .is_none_or(|n| n > (1 << 20) || n != l.weights.len())
            || l.n != l.bias.len()
            || !l.k.is_power_of_two()
            || !l.n.is_power_of_two()
            || !(1..=16).contains(&l.shift)
            || l.weights.iter().any(|w| !(-32767..=32767).contains(w))
        {
            return Err(Error::InvalidModel);
        }
        let w = o::commit(l.weights.iter().copied().map(signed).collect())?;
        let b = o::commit(l.bias.iter().copied().map(signed).collect())?;
        keys.push(LayerKey {
            k: l.k,
            n: l.n,
            shift: l.shift,
            relu: l.relu,
            w: w.tensor.clone(),
            b: b.tensor.clone(),
        });
        data.extend([w, b]);
    }
    let key = VerifierKey {
        version: 2,
        layers: keys,
    };
    key.validate()?;
    Ok((
        ProvingKey {
            key: key.clone(),
            data,
            layers,
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
        let mut s = Statement {
            m,
            input,
            output: vec![],
            context,
        };
        let out_len = m
            .checked_mul(self.key.layers.last().unwrap().n)
            .filter(|&n| n <= 1 << 20)
            .ok_or(Error::InvalidStatement)?;
        s.output.resize(out_len, 0);
        validate_statement(&self.key, &s)?;
        let mut data = self.data.clone();
        let input_id = data.len();
        data.push(o::commit(s.input.iter().copied().map(signed).collect())?);
        let mut values = s.input.clone();
        for l in &self.layers {
            let len = m * l.n;
            let mut raw = vec![0i64; len];
            let mut q = vec![0i32; len];
            let mut rem = vec![0i32; len];
            for i in 0..m {
                for j in 0..l.n {
                    let index = i * l.n + j;
                    raw[index] = (0..l.k)
                        .map(|k| i64::from(values[i * l.k + k]) * i64::from(l.weights[k * l.n + j]))
                        .sum();
                    let scale = 1i64 << l.shift;
                    q[index] = i32::try_from(
                        (raw[index] + scale / 2).div_euclid(scale) + i64::from(l.bias[j]),
                    )
                    .map_err(|_| Error::InvalidStatement)?;
                    rem[index] = (raw[index] + scale / 2).rem_euclid(scale) as i32;
                }
            }
            let raw_field = raw
                .iter()
                .map(|&x| {
                    if x < 0 {
                        F::ZERO - F::from_u64(x.unsigned_abs())
                    } else {
                        F::from_u64(x as u64)
                    }
                })
                .collect();
            let qbits = (0..32)
                .flat_map(|bit| {
                    q.iter().map(move |&x| {
                        F::from_bool((((i64::from(x) + (1i64 << 31)) as u32) >> bit) & 1 == 1)
                    })
                })
                .collect();
            let rbits = (0..(l.shift as usize).next_power_of_two())
                .flat_map(|bit| {
                    rem.iter()
                        .map(move |&x| F::from_bool((x as u32 >> bit) & 1 == 1))
                })
                .collect();
            data.extend([
                o::commit(raw_field)?,
                o::commit(q.iter().copied().map(signed).collect())?,
                o::commit(rem.iter().copied().map(signed).collect())?,
                o::commit(qbits)?,
                o::commit(rbits)?,
            ]);
            values = q
                .into_iter()
                .map(|x| if l.relu { x.max(0) } else { x })
                .collect();
            if l.relu {
                data.push(o::commit(values.iter().copied().map(signed).collect())?);
            }
        }
        s.output = values;
        let activations: Vec<Tensor> = data[input_id..].iter().map(|d| d.tensor.clone()).collect();
        let (tensors, all_ids) = tensor_layout(&self.key, m, &activations)?;
        let mut c = transcript(&self.key, &s, &tensors)?;
        let point = o::sample(&mut c, data[input_id].tensor.len.trailing_zeros() as usize);
        let input = o::open(&data[input_id], &point, &mut c)?.0;
        let mut proofs = Vec::new();
        for (l, id) in self.key.layers.iter().zip(&all_ids) {
            let matmul = o::prove_matmul(&data, [id.input, id.w, id.raw], [m, l.k, l.n], &mut c)?;
            let relations = constraints(l, id, m, &mut c)
                .iter()
                .map(|d| o::prove_relation(d, &data, &mut c))
                .collect::<Result<Vec<_>, _>>()?;
            proofs.push(LayerProof { matmul, relations });
        }
        let out = all_ids.last().unwrap().out;
        let point = o::sample(&mut c, data[out].tensor.len.trailing_zeros() as usize);
        let output = o::open(&data[out], &point, &mut c)?.0;
        let mut bytes = MAGIC.to_vec();
        bytes.extend(
            postcard::to_allocvec(&Proof {
                activations,
                input,
                layers: proofs,
                output,
            })
            .map_err(|_| Error::ProofEncoding)?,
        );
        if bytes.len() > 64 << 20 {
            return Err(Error::ProofTooLarge);
        }
        Ok((s, bytes))
    }
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::*;
    #[test]
    fn multi_layer_inference_has_no_witness_in_verifier() {
        let layers = vec![
            Layer {
                k: 2,
                n: 4,
                shift: 4,
                relu: true,
                weights: vec![8, -4, 2, 7, -8, 8, 1, -3],
                bias: vec![1, -2, 3, 0],
            },
            Layer {
                k: 4,
                n: 2,
                shift: 3,
                relu: false,
                weights: vec![1, 2, 3, 4, 5, 6, 7, 8],
                bias: vec![-4, 2],
            },
        ];
        let (pk, key) = prepare(layers).unwrap();
        let (s, proof) = pk.prove(2, vec![17, -9, -1, 0], [4; 32]).unwrap();
        let bytes = key.to_bytes().unwrap();
        assert!(bytes.len() < 512);
        let key = VerifierKey::from_bytes(&bytes).unwrap();
        verify(&key, key.id().unwrap(), &s, &proof).unwrap();
        let mut wrong = s.clone();
        wrong.output[0] += 1;
        assert!(verify(&key, key.id().unwrap(), &wrong, &proof).is_err());
        let mut decoded: Proof = postcard::from_bytes(&proof[8..]).unwrap();
        decoded.layers[0].relations.pop();
        let mut bad = MAGIC.to_vec();
        bad.extend(postcard::to_allocvec(&decoded).unwrap());
        assert!(verify(&key, key.id().unwrap(), &s, &bad).is_err());
        let mut decoded: Proof = postcard::from_bytes(&proof[8..]).unwrap();
        decoded.activations.swap(1, 2);
        let mut bad = MAGIC.to_vec();
        bad.extend(postcard::to_allocvec(&decoded).unwrap());
        assert!(verify(&key, key.id().unwrap(), &s, &bad).is_err());
    }
}
