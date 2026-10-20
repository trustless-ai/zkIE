//! Committed tensor constraints. Descriptors are built by the trusted compiler,
//! never supplied by the proof. Every terminal tensor claim gets a PCS opening.
use crate::{
    backend::F,
    whir_backend::{self as wb, Challenge as E},
    Error,
};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_sumcheck::PrescribedPointPcs;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub(crate) len: usize,
    pub(crate) root: wb::Commitment,
}
impl Tensor {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn validate(&self) -> Result<(), Error> {
        if !self.len.is_power_of_two() || self.len > (1 << 27) || self.root.num_roots() != 1 {
            return Err(Error::InvalidModel);
        }
        Ok(())
    }
    pub fn vars(&self) -> usize {
        self.len.trailing_zeros().max(5) as usize
    }
}
#[cfg(feature = "prover")]
pub(crate) struct Data {
    pub tensor: Tensor,
    pub values: Vec<F>,
    pub pcs: wb::Data,
}

#[cfg(feature = "prover")]
pub(crate) fn commit(values: Vec<F>) -> Result<std::sync::Arc<Data>, Error> {
    use p3_commit::MultilinearPcs;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_sumcheck::layout::{Layout, SuffixProver, Table};
    if !values.len().is_power_of_two() || values.len() > (1 << 27) {
        return Err(Error::InvalidModel);
    }
    let vars = values.len().trailing_zeros().max(5) as usize;
    let pcs = wb::pcs(vars)?;
    let mut padded = values.clone();
    padded.resize(1 << vars, F::ZERO);
    let witness = SuffixProver::<F, E>::new_witness(
        vec![Table::new(RowMajorMatrix::new(padded, 1 << vars))],
        5,
    );
    let (root, data) = pcs.commit(witness, &mut wb::challenger(&pcs, b"zkie/tensor/v1"));
    Ok(std::sync::Arc::new(Data {
        tensor: Tensor {
            len: values.len(),
            root,
        },
        values,
        pcs: data,
    }))
}

pub(crate) fn absorb(c: &mut wb::Challenger, values: &[E]) {
    for value in values {
        c.observe_slice(<E as BasedVectorSpace<F>>::as_basis_coefficients_slice(
            value,
        ));
    }
}
pub(crate) fn sample(c: &mut wb::Challenger, bits: usize) -> Vec<E> {
    (0..bits).map(|_| c.sample_algebra_element()).collect()
}
fn domain(c: &mut wb::Challenger, pcs: &wb::Pcs) {
    use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
    let mut d = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut d);
    d.observe_domain_separator(c);
}
#[cfg(feature = "prover")]
pub(crate) fn open(
    data: &Data,
    point: &[E],
    c: &mut wb::Challenger,
) -> Result<(wb::Opening, E), Error> {
    let pcs = wb::pcs(data.tensor.vars())?;
    domain(c, &pcs);
    let proof = pcs.open_at(
        data.pcs.clone(),
        &wb::protocol(data.tensor.vars()),
        &[wb::point(point, data.tensor.vars())],
        c,
    );
    let value = proof.evals[0].current()[0];
    Ok((proof, value))
}
pub(crate) fn verify_open(
    tensor: &Tensor,
    point: &[E],
    proof: &wb::Opening,
    c: &mut wb::Challenger,
) -> Result<E, Error> {
    tensor.validate()?;
    if point.len() != tensor.len.trailing_zeros() as usize {
        return Err(Error::InvalidProof);
    }
    let pcs = wb::pcs(tensor.vars())?;
    domain(c, &pcs);
    let evals = pcs
        .verify_at(
            &tensor.root,
            proof,
            &wb::protocol(tensor.vars()),
            &[wb::point(point, tensor.vars())],
            c,
        )
        .map_err(|_| Error::InvalidProof)?;
    evals
        .first()
        .and_then(|e| e.current().first())
        .copied()
        .ok_or(Error::InvalidProof)
}
pub(crate) fn fold(v: &mut Vec<E>, r: E) {
    for i in 0..v.len() / 2 {
        v[i] = v[2 * i] + r * (v[2 * i + 1] - v[2 * i]);
    }
    v.truncate(v.len() / 2);
}
pub(crate) fn eval(v: impl IntoIterator<Item = F>, point: &[E]) -> E {
    let mut v: Vec<E> = v.into_iter().map(E::from).collect();
    for &r in point {
        fold(&mut v, r);
    }
    v[0]
}
fn interpolate(values: &[E], r: E) -> E {
    let mut value = E::ZERO;
    for (i, &yi) in values.iter().enumerate() {
        let mut t = yi;
        for j in 0..values.len() {
            if i != j {
                t *= (r - E::from_usize(j)) / (E::from_usize(i) - E::from_usize(j));
            }
        }
        value += t;
    }
    value
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) enum Coordinate {
    Var(usize),
    Bit(bool),
}
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct View {
    pub tensor: usize,
    pub coordinates: Vec<Coordinate>,
}
impl View {
    pub fn identity(tensor: usize, log: usize) -> Self {
        Self {
            tensor,
            coordinates: (0..log).map(Coordinate::Var).collect(),
        }
    }
    fn point(&self, r: &[E]) -> Vec<E> {
        self.coordinates
            .iter()
            .map(|c| match c {
                Coordinate::Var(i) => r[*i],
                Coordinate::Bit(b) => E::from_bool(*b),
            })
            .collect()
    }
    #[cfg(feature = "prover")]
    fn index(&self, index: usize) -> usize {
        self.coordinates
            .iter()
            .enumerate()
            .map(|(b, c)| {
                let bit = match c {
                    Coordinate::Var(i) => (index >> i) & 1,
                    Coordinate::Bit(b) => usize::from(*b),
                };
                bit << b
            })
            .sum()
    }
}
#[derive(Clone)]
pub(crate) enum Factor {
    Oracle(View),
    Index {
        start: usize,
        bits: usize,
    },
    Prefix {
        start: usize,
        bits: usize,
        limit: usize,
    },
    Eq {
        point: Vec<E>,
        start: usize,
    },
    Powers {
        start: usize,
        bits: usize,
    },
    High {
        start: usize,
        bits: usize,
        boundary: usize,
    },
}
impl Factor {
    fn public_eval(&self, r: &[E]) -> E {
        match self {
            Self::Index { start, bits } => {
                (0..*bits).map(|i| r[start + i] * E::from_u64(1 << i)).sum()
            }
            Self::Prefix { start, bits, limit } => prefix(*limit, &r[*start..*start + *bits]),
            Self::Eq { point, start } => point
                .iter()
                .zip(&r[*start..])
                .map(|(&a, &b)| a * b + (E::ONE - a) * (E::ONE - b))
                .product(),
            Self::Powers { start, bits } => (0..*bits)
                .map(|i| E::ONE + r[start + i] * (E::from_u64(1u64 << (1 << i)) - E::ONE))
                .product(),
            Self::High {
                start,
                bits,
                boundary,
            } => eval(
                (0..1usize << bits).map(|i| F::from_bool(i >= *boundary)),
                &r[*start..*start + *bits],
            ),
            Self::Oracle(_) => unreachable!("authenticated separately"),
        }
    }
}
pub(crate) struct Relation {
    pub log: usize,
    pub factors: Vec<Factor>,
    pub terms: Vec<(E, Vec<usize>)>,
}
impl Relation {
    fn validate(&self, tensors: &[Tensor]) -> Result<(), Error> {
        if self.log > 27
            || self.terms.is_empty()
            || self
                .terms
                .iter()
                .any(|(_, f)| f.len() > 3 || f.iter().any(|&i| i >= self.factors.len()))
        {
            return Err(Error::InvalidModel);
        }
        for f in &self.factors {
            match f {
                Factor::Oracle(v) => {
                    let t = tensors.get(v.tensor).ok_or(Error::InvalidModel)?;
                    t.validate()?;
                    if v.coordinates.len() != t.len.trailing_zeros() as usize {
                        return Err(Error::InvalidModel);
                    }
                    let mut seen = std::collections::HashSet::new();
                    for c in &v.coordinates {
                        if let Coordinate::Var(i) = c {
                            if *i >= self.log || !seen.insert(i) {
                                return Err(Error::InvalidModel);
                            }
                        }
                    }
                }
                Factor::Index { start, bits } | Factor::Prefix { start, bits, .. } => {
                    if *bits > 27 || start + bits > self.log {
                        return Err(Error::InvalidModel);
                    }
                }
                Factor::Eq { point, start } => {
                    if start + point.len() > self.log {
                        return Err(Error::InvalidModel);
                    }
                }
                Factor::Powers { start, bits } | Factor::High { start, bits, .. } => {
                    if *bits > 6 || start + bits > self.log {
                        return Err(Error::InvalidModel);
                    }
                }
            }
        }
        Ok(())
    }
    fn degree(&self) -> usize {
        self.terms.iter().map(|(_, f)| f.len()).max().unwrap()
    }
    fn combine(&self, values: &[E]) -> E {
        self.terms
            .iter()
            .map(|(c, fs)| fs.iter().fold(*c, |c, &i| c * values[i]))
            .sum()
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct RelationProof {
    pub rounds: Vec<Vec<E>>,
    pub openings: Vec<wb::Opening>,
}
#[cfg(feature = "prover")]
pub(crate) fn prove_relation(
    desc: &Relation,
    data: &[std::sync::Arc<Data>],
    c: &mut wb::Challenger,
) -> Result<RelationProof, Error> {
    desc.validate(&data.iter().map(|d| d.tensor.clone()).collect::<Vec<_>>())?;
    let mut factors: Vec<Vec<E>> = desc
        .factors
        .iter()
        .map(|f| {
            (0..1usize << desc.log)
                .map(|i| match f {
                    Factor::Oracle(v) => E::from(data[v.tensor].values[v.index(i)]),
                    _ => f.public_eval(
                        &(0..desc.log)
                            .map(|b| E::from_bool((i >> b) & 1 == 1))
                            .collect::<Vec<_>>(),
                    ),
                })
                .collect()
        })
        .collect();
    let mut rounds = Vec::new();
    let mut point = Vec::new();
    for _ in 0..desc.log {
        let mut message = vec![E::ZERO; desc.degree() + 1];
        for i in 0..factors[0].len() / 2 {
            for (z, sum) in message.iter_mut().enumerate() {
                let z = E::from_usize(z);
                let v: Vec<E> = factors
                    .iter()
                    .map(|f| f[2 * i] + z * (f[2 * i + 1] - f[2 * i]))
                    .collect();
                *sum += desc.combine(&v);
            }
        }
        absorb(c, &message);
        let r: E = c.sample_algebra_element();
        for f in &mut factors {
            fold(f, r);
        }
        rounds.push(message);
        point.push(r);
    }
    let mut openings = Vec::new();
    for f in &desc.factors {
        if let Factor::Oracle(v) = f {
            openings.push(open(&data[v.tensor], &v.point(&point), c)?.0);
        }
    }
    Ok(RelationProof { rounds, openings })
}
pub(crate) fn verify_relation(
    desc: &Relation,
    tensors: &[Tensor],
    proof: &RelationProof,
    c: &mut wb::Challenger,
) -> Result<(), Error> {
    desc.validate(tensors)?;
    if proof.rounds.len() != desc.log
        || proof.openings.len()
            != desc
                .factors
                .iter()
                .filter(|f| matches!(f, Factor::Oracle(_)))
                .count()
    {
        return Err(Error::InvalidProof);
    }
    let mut claim = E::ZERO;
    let mut point = Vec::new();
    for message in &proof.rounds {
        if message.len() != desc.degree() + 1 || message[0] + message[1] != claim {
            return Err(Error::InvalidProof);
        }
        absorb(c, message);
        let r: E = c.sample_algebra_element();
        claim = interpolate(message, r);
        point.push(r);
    }
    let mut openings = proof.openings.iter();
    let mut values = Vec::new();
    for f in &desc.factors {
        values.push(match f {
            Factor::Oracle(v) => verify_open(
                &tensors[v.tensor],
                &v.point(&point),
                openings.next().ok_or(Error::InvalidProof)?,
                c,
            )?,
            _ => f.public_eval(&point),
        });
    }
    if claim != desc.combine(&values) {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub(crate) struct MatmulProof {
    output: wb::Opening,
    rounds: Vec<[E; 3]>,
    left: wb::Opening,
    right: wb::Opening,
}
#[cfg(feature = "prover")]
pub(crate) fn prove_matmul(
    data: &[std::sync::Arc<Data>],
    ids: [usize; 3],
    shape: [usize; 3],
    c: &mut wb::Challenger,
) -> Result<MatmulProof, Error> {
    let [a, b, out] = ids;
    let [m, k, n] = shape;
    let u = sample(c, m.trailing_zeros() as usize);
    let v = sample(c, n.trailing_zeros() as usize);
    let mut out_point = v.clone();
    out_point.extend_from_slice(&u);
    let output = open(&data[out], &out_point, c)?.0;
    let mut av: Vec<E> = (0..k)
        .map(|t| eval((0..m).map(|i| data[a].values[i * k + t]), &u))
        .collect();
    let mut bv: Vec<E> = (0..k)
        .map(|t| eval(data[b].values[t * n..(t + 1) * n].iter().copied(), &v))
        .collect();
    let mut rounds = Vec::new();
    let mut r = Vec::new();
    while av.len() > 1 {
        let mut p = [E::ZERO; 3];
        for i in 0..av.len() / 2 {
            for (z, sum) in p.iter_mut().enumerate() {
                let z = E::from_usize(z);
                *sum += (av[2 * i] + z * (av[2 * i + 1] - av[2 * i]))
                    * (bv[2 * i] + z * (bv[2 * i + 1] - bv[2 * i]));
            }
        }
        absorb(c, &p);
        let ch: E = c.sample_algebra_element();
        fold(&mut av, ch);
        fold(&mut bv, ch);
        rounds.push(p);
        r.push(ch);
    }
    let mut ap = r.clone();
    ap.extend_from_slice(&u);
    let mut bp = v;
    bp.extend_from_slice(&r);
    let left = open(&data[a], &ap, c)?.0;
    let right = open(&data[b], &bp, c)?.0;
    Ok(MatmulProof {
        output,
        rounds,
        left,
        right,
    })
}
pub(crate) fn verify_matmul(
    tensors: &[Tensor],
    ids: [usize; 3],
    shape: [usize; 3],
    proof: &MatmulProof,
    c: &mut wb::Challenger,
) -> Result<(), Error> {
    let [a, b, out] = ids;
    let [m, k, n] = shape;
    if [m, k, n].iter().any(|x| !x.is_power_of_two())
        || tensors[a].len != m * k
        || tensors[b].len != k * n
        || tensors[out].len != m * n
        || proof.rounds.len() != k.trailing_zeros() as usize
    {
        return Err(Error::InvalidProof);
    }
    let u = sample(c, m.trailing_zeros() as usize);
    let v = sample(c, n.trailing_zeros() as usize);
    let mut op = v.clone();
    op.extend_from_slice(&u);
    let mut claim = verify_open(&tensors[out], &op, &proof.output, c)?;
    let mut r = Vec::new();
    for p in &proof.rounds {
        if p[0] + p[1] != claim {
            return Err(Error::InvalidProof);
        }
        absorb(c, p);
        let ch: E = c.sample_algebra_element();
        claim = interpolate(p, ch);
        r.push(ch);
    }
    let mut ap = r.clone();
    ap.extend_from_slice(&u);
    let mut bp = v;
    bp.extend_from_slice(&r);
    let left = verify_open(&tensors[a], &ap, &proof.left, c)?;
    let right = verify_open(&tensors[b], &bp, &proof.right, c)?;
    if claim != left * right {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::*;
    #[test]
    fn authenticated_boolean_constraints_reject_false_witness_and_missing_openings() {
        let data = vec![commit(vec![F::ZERO, F::ONE, F::ONE, F::ZERO]).unwrap()];
        let pcs = wb::pcs(5).unwrap();
        let mut c = wb::challenger(&pcs, b"oracle-test");
        c.observe(data[0].tensor.root.clone());
        let gamma = sample(&mut c, 2);
        let desc = Relation {
            log: 2,
            factors: vec![
                Factor::Oracle(View::identity(0, 2)),
                Factor::Eq {
                    point: gamma,
                    start: 0,
                },
            ],
            terms: vec![(E::ONE, vec![0, 0, 1]), (-E::ONE, vec![0, 1])],
        };
        let mut vc = c.clone();
        let mut proof = prove_relation(&desc, &data, &mut c).unwrap();
        verify_relation(&desc, &[data[0].tensor.clone()], &proof, &mut vc).unwrap();
        proof.openings.clear();
        assert!(verify_relation(&desc, &[data[0].tensor.clone()], &proof, &mut vc).is_err());
        let bad = vec![commit(vec![F::ZERO, F::from_u32(2), F::ONE, F::ZERO]).unwrap()];
        let mut c = wb::challenger(&pcs, b"bad-oracle-test");
        c.observe(bad[0].tensor.root.clone());
        let mut vc = c.clone();
        let proof = prove_relation(&desc, &bad, &mut c).unwrap();
        assert!(verify_relation(&desc, &[bad[0].tensor.clone()], &proof, &mut vc).is_err());
    }
}

fn prefix(limit: usize, r: &[E]) -> E {
    if limit == 0 {
        return E::ZERO;
    }
    if limit >= 1usize << r.len() {
        return E::ONE;
    }
    let half = 1usize << (r.len() - 1);
    let top = r[r.len() - 1];
    if limit <= half {
        (E::ONE - top) * prefix(limit, &r[..r.len() - 1])
    } else {
        E::ONE - top + top * prefix(limit - half, &r[..r.len() - 1])
    }
}
