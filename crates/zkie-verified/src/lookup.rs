//! Authenticated LogUp lookup with pre-challenge multiplicities and a fully
//! constrained fraction tree. Public API inputs are commitments, not witnesses.
use crate::{
    backend::F,
    oracle::{self as o, Coordinate as C, Factor as X, Relation, Tensor, View},
    whir_backend::{self as wb, Challenge as E},
    Error,
};
use p3_challenger::{CanObserve, FieldChallenger};
#[cfg(feature = "prover")]
use p3_field::PrimeField64;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::panic::{catch_unwind, AssertUnwindSafe};
const D: usize = <E as BasedVectorSpace<F>>::DIMENSION;
const MAGIC: &[u8; 8] = b"ZKIELU01";

/// These roots must be pinned by a model/input statement or connected to other
/// authenticated graph constraints. A prover's unendorsed roots prove no model.
#[derive(Clone, Serialize, Deserialize)]
pub struct Spec {
    pub indices: usize,
    pub values: usize,
    pub table: usize,
    pub actual_table_len: usize,
}
impl Spec {
    fn validate(&self, tensors: &[Tensor]) -> Result<(), Error> {
        if tensors.len() > 65536 {
            return Err(Error::InvalidModel);
        }
        let x = tensors.get(self.indices).ok_or(Error::InvalidModel)?;
        let y = tensors.get(self.values).ok_or(Error::InvalidModel)?;
        let table = tensors.get(self.table).ok_or(Error::InvalidModel)?;
        for t in tensors {
            t.validate()?;
        }
        if x.len != y.len || self.actual_table_len == 0 || self.actual_table_len > table.len {
            return Err(Error::InvalidModel);
        }
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
pub(crate) struct Proof {
    counts: Tensor,
    aux: Vec<Tensor>,
    relations: Vec<o::RelationProof>,
    terminal: Vec<wb::Opening>,
}
struct Node {
    len: usize,
    n: [usize; D],
    d: [usize; D],
}
fn basis(i: usize) -> E {
    <E as BasedVectorSpace<F>>::ith_basis_element(i).unwrap()
}
type Atom = Vec<(E, View)>;
fn atom(ids: &[usize; D], log: usize, half: Option<bool>) -> Atom {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| {
            let mut view = View::identity(id, log);
            if let Some(bit) = half {
                view.coordinates.push(C::Bit(bit));
            }
            (basis(i), view)
        })
        .collect()
}
fn equation(
    log: usize,
    atoms: &[Atom],
    terms: &[(E, Vec<usize>)],
    c: &mut wb::Challenger,
) -> Relation {
    let mut fs = Vec::new();
    let mut references = Vec::new();
    for a in atoms {
        references.push(
            a.iter()
                .map(|(coef, v)| {
                    let id = fs.len();
                    fs.push(X::Oracle(v.clone()));
                    (*coef, id)
                })
                .collect::<Vec<_>>(),
        );
    }
    let mut expanded = Vec::new();
    for (coefficient, vars) in terms {
        let mut parts = vec![(*coefficient, vec![])];
        for &v in vars {
            parts = parts
                .into_iter()
                .flat_map(|(c, indices)| {
                    references[v].iter().map(move |&(a, i)| {
                        let mut is = indices.clone();
                        is.push(i);
                        (c * a, is)
                    })
                })
                .collect();
        }
        expanded.extend(parts);
    }
    finish(log, fs, expanded, c)
}
fn finish(
    log: usize,
    mut factors: Vec<X>,
    terms: Vec<(E, Vec<usize>)>,
    c: &mut wb::Challenger,
) -> Relation {
    let eq = factors.len();
    factors.push(X::Eq {
        point: o::sample(c, log),
        start: 0,
    });
    Relation {
        log,
        factors,
        terms: terms
            .into_iter()
            .map(|(a, mut ids)| {
                ids.push(eq);
                (a, ids)
            })
            .collect(),
    }
}
fn layout(
    base: &[Tensor],
    counts: &Tensor,
    aux: &[Tensor],
    spec: &Spec,
) -> Result<(Vec<Tensor>, [Vec<Node>; 2]), Error> {
    if counts.len != base[spec.table].len {
        return Err(Error::InvalidProof);
    }
    counts.validate()?;
    let mut tensors = base.to_vec();
    tensors.push(counts.clone());
    let mut cursor = 0;
    let mut trees = [Vec::new(), Vec::new()];
    for (tree, mut len) in trees
        .iter_mut()
        .zip([base[spec.indices].len, base[spec.table].len])
    {
        loop {
            let mut n = [0; D];
            let mut d = [0; D];
            for ids in [&mut n, &mut d] {
                for slot in ids {
                    let tensor = aux.get(cursor).ok_or(Error::InvalidProof)?;
                    if tensor.len != len {
                        return Err(Error::InvalidProof);
                    }
                    tensor.validate()?;
                    *slot = tensors.len();
                    tensors.push(tensor.clone());
                    cursor += 1;
                }
            }
            tree.push(Node { len, n, d });
            if len == 1 {
                break;
            }
            len /= 2;
        }
    }
    if cursor != aux.len() {
        return Err(Error::InvalidProof);
    }
    Ok((tensors, trees))
}
fn constraints(
    spec: &Spec,
    base: &[Tensor],
    trees: &[Vec<Node>; 2],
    alpha: E,
    beta: E,
    c: &mut wb::Challenger,
) -> Vec<Relation> {
    let mut relations = Vec::new();
    let left = &trees[0][0];
    let right = &trees[1][0];
    let xl = left.len.trailing_zeros() as usize;
    let tl = right.len.trailing_zeros() as usize;
    relations.push(equation(
        xl,
        &[atom(&left.n, xl, None)],
        &[(E::ONE, vec![0]), (-E::ONE, vec![])],
        c,
    ));
    relations.push(equation(
        xl,
        &[
            atom(&left.d, xl, None),
            vec![(E::ONE, View::identity(spec.indices, xl))],
            vec![(E::ONE, View::identity(spec.values, xl))],
        ],
        &[
            (E::ONE, vec![0]),
            (-E::ONE, vec![1]),
            (-beta, vec![2]),
            (-alpha, vec![]),
        ],
        c,
    ));
    let mut fs: Vec<X> = atom(&right.n, tl, None)
        .into_iter()
        .map(|(_, v)| X::Oracle(v))
        .collect();
    fs.push(X::Oracle(View::identity(base.len(), tl)));
    fs.push(X::Prefix {
        start: 0,
        bits: tl,
        limit: spec.actual_table_len,
    });
    let mut terms: Vec<_> = (0..D).map(|i| (basis(i), vec![i])).collect();
    terms.push((-E::ONE, vec![D, D + 1]));
    relations.push(finish(tl, fs, terms, c));
    let mut fs: Vec<X> = atom(&right.d, tl, None)
        .into_iter()
        .map(|(_, v)| X::Oracle(v))
        .collect();
    fs.push(X::Oracle(View::identity(spec.table, tl)));
    fs.push(X::Index { start: 0, bits: tl });
    let mut terms: Vec<_> = (0..D).map(|i| (basis(i), vec![i])).collect();
    terms.extend([(-beta, vec![D]), (-E::ONE, vec![D + 1]), (-alpha, vec![])]);
    relations.push(finish(tl, fs, terms, c));
    for tree in trees {
        for pair in tree.windows(2) {
            let child = &pair[0];
            let parent = &pair[1];
            let log = parent.len.trailing_zeros() as usize;
            relations.push(equation(
                log,
                &[
                    atom(&parent.n, log, None),
                    atom(&child.n, log, Some(false)),
                    atom(&child.d, log, Some(true)),
                    atom(&child.n, log, Some(true)),
                    atom(&child.d, log, Some(false)),
                ],
                &[
                    (E::ONE, vec![0]),
                    (-E::ONE, vec![1, 2]),
                    (-E::ONE, vec![3, 4]),
                ],
                c,
            ));
            relations.push(equation(
                log,
                &[
                    atom(&parent.d, log, None),
                    atom(&child.d, log, Some(false)),
                    atom(&child.d, log, Some(true)),
                ],
                &[(E::ONE, vec![0]), (-E::ONE, vec![1, 2])],
                c,
            ));
        }
    }
    relations
}
fn terminal_ids(trees: &[Vec<Node>; 2]) -> Vec<usize> {
    trees
        .iter()
        .flat_map(|tree| {
            let n = tree.last().unwrap();
            n.n.into_iter().chain(n.d)
        })
        .collect()
}
fn terminal_check(values: &[F]) -> Result<(), Error> {
    if values.len() != 4 * D {
        return Err(Error::InvalidProof);
    }
    let v: Vec<E> = values
        .chunks_exact(D)
        .map(|parts| <E as BasedVectorSpace<F>>::from_basis_coefficients_slice(parts).unwrap())
        .collect();
    if v[1] == E::ZERO || v[3] == E::ZERO || v[0] * v[3] != v[2] * v[1] {
        return Err(Error::InvalidProof);
    }
    Ok(())
}
pub(crate) fn verify_core(
    spec: &Spec,
    base: &[Tensor],
    proof: &Proof,
    c: &mut wb::Challenger,
) -> Result<(), Error> {
    spec.validate(base)?;
    c.observe(proof.counts.root.clone());
    let alpha: E = c.sample_algebra_element();
    let beta: E = c.sample_algebra_element();
    let (tensors, trees) = layout(base, &proof.counts, &proof.aux, spec)?;
    for t in &proof.aux {
        c.observe(t.root.clone());
    }
    let relations = constraints(spec, base, &trees, alpha, beta, c);
    if relations.len() != proof.relations.len() {
        return Err(Error::InvalidProof);
    }
    for (r, p) in relations.iter().zip(&proof.relations) {
        o::verify_relation(r, &tensors, p, c)?;
    }
    let ids = terminal_ids(&trees);
    if ids.len() != proof.terminal.len() {
        return Err(Error::InvalidProof);
    }
    let mut values = Vec::new();
    for (&id, p) in ids.iter().zip(&proof.terminal) {
        let value = o::verify_open(&tensors[id], &[], p, c)?;
        let components = <E as BasedVectorSpace<F>>::as_basis_coefficients_slice(&value);
        if components[1..].iter().any(|&f| f != F::ZERO) {
            return Err(Error::InvalidProof);
        }
        values.push(components[0]);
    }
    terminal_check(&values)
}
fn transcript(spec: &Spec, tensors: &[Tensor], context: [u8; 32]) -> Result<wb::Challenger, Error> {
    spec.validate(tensors)?;
    let mut h = Sha256::new();
    h.update(b"zkie/lookup/v1\0");
    h.update(context);
    h.update(postcard::to_allocvec(spec).map_err(|_| Error::ProofEncoding)?);
    h.update(postcard::to_allocvec(tensors).map_err(|_| Error::ProofEncoding)?);
    let mut c = wb::challenger(&wb::pcs(5)?, &h.finalize());
    for t in tensors {
        c.observe(t.root.clone());
    }
    Ok(c)
}
pub fn verify(
    spec: &Spec,
    tensors: &[Tensor],
    context: [u8; 32],
    bytes: &[u8],
) -> Result<(), Error> {
    if bytes.len() > 64 << 20 {
        return Err(Error::ProofTooLarge);
    }
    if !bytes.starts_with(MAGIC) {
        return Err(Error::ProofEncoding);
    }
    catch_unwind(AssertUnwindSafe(|| {
        let (proof, tail): (Proof, _) =
            postcard::take_from_bytes(&bytes[8..]).map_err(|_| Error::ProofEncoding)?;
        if !tail.is_empty()
            || postcard::to_allocvec(&proof).map_err(|_| Error::ProofEncoding)? != bytes[8..]
        {
            return Err(Error::ProofEncoding);
        }
        verify_core(
            spec,
            tensors,
            &proof,
            &mut transcript(spec, tensors, context)?,
        )
    }))
    .unwrap_or(Err(Error::InvalidProof))
}

#[cfg(feature = "prover")]
pub struct CommittedTensor(std::sync::Arc<o::Data>);
#[cfg(feature = "prover")]
impl CommittedTensor {
    pub fn commitment(&self) -> Tensor {
        self.0.tensor.clone()
    }
}
#[cfg(feature = "prover")]
pub fn commit(values: &[i64]) -> Result<CommittedTensor, Error> {
    if values.iter().any(|&x| x.unsigned_abs() >= 1 << 62) {
        return Err(Error::InvalidStatement);
    }
    Ok(CommittedTensor(o::commit(
        values
            .iter()
            .map(|&x| {
                if x < 0 {
                    F::ZERO - F::from_u64(x.unsigned_abs())
                } else {
                    F::from_u64(x as u64)
                }
            })
            .collect(),
    )?))
}
#[cfg(feature = "prover")]
fn append_tree(
    mut num: Vec<E>,
    mut den: Vec<E>,
    data: &mut Vec<std::sync::Arc<o::Data>>,
) -> Result<(), Error> {
    loop {
        for values in [&num, &den] {
            for i in 0..D {
                data.push(o::commit(
                    values
                        .iter()
                        .map(|v| <E as BasedVectorSpace<F>>::as_basis_coefficients_slice(v)[i])
                        .collect(),
                )?);
            }
        }
        if num.len() == 1 {
            break;
        }
        let half = num.len() / 2;
        let next_num = (0..half)
            .map(|i| num[i] * den[i + half] + num[i + half] * den[i])
            .collect();
        let next_den = (0..half).map(|i| den[i] * den[i + half]).collect();
        num = next_num;
        den = next_den;
    }
    Ok(())
}
#[cfg(feature = "prover")]
pub(crate) fn prove_core(
    spec: &Spec,
    base: &[std::sync::Arc<o::Data>],
    c: &mut wb::Challenger,
) -> Result<Proof, Error> {
    let roots: Vec<Tensor> = base.iter().map(|d| d.tensor.clone()).collect();
    spec.validate(&roots)?;
    let x = &base[spec.indices].values;
    let y = &base[spec.values].values;
    let table = &base[spec.table].values;
    let mut counts = vec![F::ZERO; table.len()];
    for index in x {
        if let Ok(i) = usize::try_from(index.as_canonical_u64()) {
            if i < spec.actual_table_len {
                counts[i] += F::ONE;
            }
        }
    }
    let count_data = o::commit(counts.clone())?;
    c.observe(count_data.tensor.root.clone());
    let alpha: E = c.sample_algebra_element();
    let beta: E = c.sample_algebra_element();
    let mut data = base.to_vec();
    data.push(count_data.clone());
    let start = data.len();
    append_tree(
        vec![E::ONE; x.len()],
        x.iter()
            .zip(y)
            .map(|(&x, &y)| alpha + E::from(x) + beta * E::from(y))
            .collect(),
        &mut data,
    )?;
    append_tree(
        counts
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i < spec.actual_table_len {
                    E::from(v)
                } else {
                    E::ZERO
                }
            })
            .collect(),
        table
            .iter()
            .enumerate()
            .map(|(i, &v)| alpha + E::from_usize(i) + beta * E::from(v))
            .collect(),
        &mut data,
    )?;
    let aux: Vec<Tensor> = data[start..].iter().map(|d| d.tensor.clone()).collect();
    for t in &aux {
        c.observe(t.root.clone());
    }
    let (_, trees) = layout(&roots, &count_data.tensor, &aux, spec)?;
    let relations = constraints(spec, &roots, &trees, alpha, beta, c)
        .iter()
        .map(|r| o::prove_relation(r, &data, c))
        .collect::<Result<Vec<_>, _>>()?;
    let terminal = terminal_ids(&trees)
        .iter()
        .map(|&i| o::open(&data[i], &[], c).map(|p| p.0))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Proof {
        counts: count_data.tensor.clone(),
        aux,
        relations,
        terminal,
    })
}
#[cfg(feature = "prover")]
pub fn prove(
    spec: &Spec,
    tensors: &[CommittedTensor],
    context: [u8; 32],
) -> Result<Vec<u8>, Error> {
    let data: Vec<_> = tensors.iter().map(|t| t.0.clone()).collect();
    let roots: Vec<_> = tensors.iter().map(CommittedTensor::commitment).collect();
    let proof = prove_core(spec, &data, &mut transcript(spec, &roots, context)?)?;
    let mut bytes = MAGIC.to_vec();
    bytes.extend(postcard::to_allocvec(&proof).map_err(|_| Error::ProofEncoding)?);
    if bytes.len() > 64 << 20 {
        return Err(Error::ProofTooLarge);
    }
    Ok(bytes)
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::*;
    #[test]
    fn lookup_binds_counts_layers_roots_and_non_power_of_two_table_domain() {
        let spec = Spec {
            indices: 0,
            values: 1,
            table: 2,
            actual_table_len: 5,
        };
        let tensors = vec![
            commit(&[0, 4, 2, 2]).unwrap(),
            commit(&[11, 55, 33, 33]).unwrap(),
            commit(&[11, 22, 33, 44, 55, 0, 0, 0]).unwrap(),
        ];
        let roots: Vec<_> = tensors.iter().map(CommittedTensor::commitment).collect();
        let bytes = prove(&spec, &tensors, [1; 32]).unwrap();
        verify(&spec, &roots, [1; 32], &bytes).unwrap();
        assert!(verify(&spec, &roots, [2; 32], &bytes).is_err());
        let mut p: Proof = postcard::from_bytes(&bytes[8..]).unwrap();
        p.relations.clear();
        let mut bad = MAGIC.to_vec();
        bad.extend(postcard::to_allocvec(&p).unwrap());
        assert!(verify(&spec, &roots, [1; 32], &bad).is_err());
        let bad = vec![
            commit(&[0, 4, 2, 2]).unwrap(),
            commit(&[99, 55, 33, 33]).unwrap(),
            commit(&[11, 22, 33, 44, 55, 0, 0, 0]).unwrap(),
        ];
        let roots: Vec<_> = bad.iter().map(CommittedTensor::commitment).collect();
        let bytes = prove(&spec, &bad, [1; 32]).unwrap();
        assert!(verify(&spec, &roots, [1; 32], &bytes).is_err());
        let padded = vec![
            commit(&[7, 4, 2, 2]).unwrap(),
            commit(&[0, 55, 33, 33]).unwrap(),
            commit(&[11, 22, 33, 44, 55, 0, 0, 0]).unwrap(),
        ];
        let roots: Vec<_> = padded.iter().map(CommittedTensor::commitment).collect();
        let bytes = prove(&spec, &padded, [1; 32]).unwrap();
        assert!(verify(&spec, &roots, [1; 32], &bytes).is_err());
    }
}
