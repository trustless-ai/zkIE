//! Pinned integer computation graphs. The verifier receives commitments and
//! public endpoints, never model weights or intermediate activation arrays.
use crate::{
    backend::F,
    dense, lookup,
    oracle::{self as o, Coordinate as C, Factor as X, Tensor, View},
    whir_backend::{self as wb, Challenge as E},
    Error,
};
use p3_challenger::CanObserve;
use p3_field::{Field, PrimeCharacteristicRing};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::panic::{catch_unwind, AssertUnwindSafe};
const DOMAIN: &[u8] = b"zkie/integer-graph/v1\0";
const MAGIC: &[u8; 8] = b"ZKIEGR01";
const MODULUS: u128 = 18446744069414584321;
const MAX_NODES: usize = 32768;
const MAX_PROOF_BYTES: usize = 512 << 20;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Range {
    Signed(u8),
    Unsigned(u8),
}
impl Range {
    fn width(self) -> usize {
        match self {
            Self::Signed(b) | Self::Unsigned(b) => b as usize,
        }
    }
    fn validate(self) -> Result<(), Error> {
        let b = self.width();
        if b == 0 || b > 62 {
            return Err(Error::InvalidModel);
        }
        Ok(())
    }
    fn offset(self) -> u64 {
        match self {
            Self::Signed(b) => 1u64 << (b - 1),
            Self::Unsigned(_) => 0,
        }
    }
    fn bound(self) -> u128 {
        match self {
            Self::Signed(b) => 1u128 << (b - 1),
            Self::Unsigned(b) => (1u128 << b) - 1,
        }
    }
    fn contains(self, x: i64) -> bool {
        match self {
            Self::Signed(b) => -(1i128 << (b - 1)) <= x as i128 && (x as i128) < (1i128 << (b - 1)),
            Self::Unsigned(b) => x >= 0 && (x as u64) < (1u64 << b),
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub enum Bit {
    Variable(usize),
    Fixed(bool),
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Source {
    pub tensor: usize,
    pub coordinates: Vec<Bit>,
}
impl Source {
    pub fn identity(tensor: usize, len: usize) -> Self {
        Self {
            tensor,
            coordinates: (0..len.trailing_zeros() as usize)
                .map(Bit::Variable)
                .collect(),
        }
    }
    fn view(&self) -> View {
        View {
            tensor: self.tensor,
            coordinates: self
                .coordinates
                .iter()
                .map(|b| match b {
                    Bit::Variable(i) => C::Var(*i),
                    Bit::Fixed(b) => C::Bit(*b),
                })
                .collect(),
        }
    }
    fn validate(&self, nodes: &[Node], before: usize, log: usize) -> Result<(), Error> {
        let node = nodes
            .get(self.tensor)
            .filter(|_| self.tensor < before)
            .ok_or(Error::InvalidModel)?;
        if self.coordinates.len() != node.len.trailing_zeros() as usize {
            return Err(Error::InvalidModel);
        }
        let mut seen = vec![false; log];
        for b in &self.coordinates {
            if let Bit::Variable(i) = b {
                if *i >= log || seen[*i] {
                    return Err(Error::InvalidModel);
                }
                seen[*i] = true;
            }
        }
        Ok(())
    }
    #[cfg(feature = "prover")]
    fn index(&self, index: usize) -> usize {
        self.coordinates
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let v = match b {
                    Bit::Variable(j) => (index >> j) & 1,
                    Bit::Fixed(b) => usize::from(*b),
                };
                v << i
            })
            .sum()
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub enum Operation {
    Input,
    Constant,
    Linear {
        terms: Vec<(i64, Source)>,
        constant: i64,
    },
    Product {
        a: Source,
        b: Source,
    },
    Matmul {
        a: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    /// floor((x + divisor/2)/divisor) for nearest, floor(x/divisor) otherwise.
    Round {
        input: usize,
        divisor: u64,
        nearest: bool,
    },
    Relu {
        input: usize,
    },
    SumRows {
        input: usize,
        columns: usize,
        actual_columns: usize,
    },
    DividePositive {
        numerator: usize,
        denominator: usize,
    },
    Lookup {
        indices: usize,
        table: usize,
        actual_len: usize,
    },
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Node {
    pub len: usize,
    pub range: Range,
    pub operation: Operation,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Program {
    pub nodes: Vec<Node>,
    pub output: usize,
}
impl Program {
    pub fn validate(&self) -> Result<(), Error> {
        if self.nodes.is_empty() || self.nodes.len() > MAX_NODES || self.output >= self.nodes.len()
        {
            return Err(Error::InvalidModel);
        }
        let mut inputs = 0;
        for (i, node) in self.nodes.iter().enumerate() {
            node.range.validate()?;
            if !node.len.is_power_of_two()
                || node.len > (1 << 26)
                || (!matches!(node.operation, Operation::Constant)
                    && node
                        .len
                        .checked_mul(node.range.width().next_power_of_two())
                        .is_none_or(|n| n > (1 << 27)))
            {
                return Err(Error::InvalidModel);
            }
            let get = |id: usize| {
                self.nodes
                    .get(id)
                    .filter(|_| id < i)
                    .ok_or(Error::InvalidModel)
            };
            let mut bound = node.range.bound();
            match &node.operation {
                Operation::Input => inputs += 1,
                Operation::Constant => {}
                Operation::Linear { terms, constant } => {
                    if terms.is_empty() || terms.len() > 16 {
                        return Err(Error::InvalidModel);
                    }
                    bound += constant.unsigned_abs() as u128;
                    for (a, s) in terms {
                        s.validate(&self.nodes, i, node.len.trailing_zeros() as usize)?;
                        bound = bound
                            .checked_add((*a).unsigned_abs() as u128 * get(s.tensor)?.range.bound())
                            .ok_or(Error::InvalidModel)?;
                    }
                }
                Operation::Product { a, b } => {
                    a.validate(&self.nodes, i, node.len.trailing_zeros() as usize)?;
                    b.validate(&self.nodes, i, node.len.trailing_zeros() as usize)?;
                    bound += get(a.tensor)?.range.bound() * get(b.tensor)?.range.bound();
                }
                Operation::Matmul { a, b, m, k, n } => {
                    if !m.is_power_of_two()
                        || !k.is_power_of_two()
                        || !n.is_power_of_two()
                        || m.checked_mul(*k) != Some(get(*a)?.len)
                        || k.checked_mul(*n) != Some(get(*b)?.len)
                        || m.checked_mul(*n) != Some(node.len)
                    {
                        return Err(Error::InvalidModel);
                    }
                    bound = bound
                        .checked_add(
                            get(*a)?
                                .range
                                .bound()
                                .checked_mul(get(*b)?.range.bound())
                                .and_then(|v| v.checked_mul(*k as u128))
                                .ok_or(Error::InvalidModel)?,
                        )
                        .ok_or(Error::InvalidModel)?;
                }
                Operation::Round { input, divisor, .. } => {
                    if get(*input)?.len != node.len || *divisor == 0 || *divisor > (1 << 32) {
                        return Err(Error::InvalidModel);
                    }
                    let width = (64 - (divisor - 1).leading_zeros() as usize).max(1);
                    let rmax = (1u128 << width) - 1;
                    let half = if matches!(node.operation, Operation::Round { nearest: true, .. }) {
                        divisor / 2
                    } else {
                        0
                    };
                    bound = (bound * (*divisor as u128)
                        + get(*input)?.range.bound()
                        + half as u128
                        + rmax)
                        .max(2 * rmax + *divisor as u128 - 1);
                }
                Operation::Relu { input } => {
                    let x = get(*input)?;
                    if x.len != node.len
                        || !matches!(x.range, Range::Signed(_))
                        || matches!(x.operation, Operation::Constant)
                    {
                        return Err(Error::InvalidModel);
                    }
                    bound += x.range.bound();
                }
                Operation::SumRows {
                    input,
                    columns,
                    actual_columns,
                } => {
                    let x = get(*input)?;
                    if !columns.is_power_of_two()
                        || *actual_columns == 0
                        || actual_columns > columns
                        || node.len.checked_mul(*columns) != Some(x.len)
                    {
                        return Err(Error::InvalidModel);
                    }
                    bound += x.range.bound() * (*actual_columns as u128);
                }
                Operation::DividePositive {
                    numerator,
                    denominator,
                } => {
                    let a = get(*numerator)?;
                    let b = get(*denominator)?;
                    if a.len != node.len
                        || b.len != node.len
                        || !matches!(a.range, Range::Unsigned(_))
                        || !matches!(b.range, Range::Unsigned(1..=60))
                        || !matches!(node.range, Range::Unsigned(_))
                    {
                        return Err(Error::InvalidModel);
                    }
                    bound = 2 * a.range.bound()
                        + 2 * b.range.bound() * node.range.bound()
                        + 8 * b.range.bound()
                        + 1;
                }
                Operation::Lookup {
                    indices,
                    table,
                    actual_len,
                } => {
                    if get(*indices)?.len != node.len
                        || *actual_len == 0
                        || *actual_len > get(*table)?.len
                    {
                        return Err(Error::InvalidModel);
                    }
                }
            }
            if bound >= MODULUS {
                return Err(Error::InvalidModel);
            }
        }
        if inputs != 1 {
            return Err(Error::InvalidModel);
        }
        Ok(())
    }
    fn input(&self) -> usize {
        self.nodes
            .iter()
            .position(|n| matches!(n.operation, Operation::Input))
            .unwrap()
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct VerifierKey {
    program: Program,
    constants: Vec<(usize, Tensor)>,
}
impl VerifierKey {
    fn validate(&self) -> Result<(), Error> {
        self.program.validate()?;
        let expected: Vec<_> = self
            .program
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.operation, Operation::Constant))
            .map(|(i, _)| i)
            .collect();
        if expected != self.constants.iter().map(|(i, _)| *i).collect::<Vec<_>>() {
            return Err(Error::InvalidModel);
        }
        for (id, t) in &self.constants {
            t.validate()?;
            if t.len != self.program.nodes[*id].len {
                return Err(Error::InvalidModel);
            }
        }
        Ok(())
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        self.validate()?;
        postcard::to_allocvec(self).map_err(|_| Error::ProofEncoding)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > 16 << 20 {
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
    pub fn id(&self) -> Result<[u8; 32], Error> {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        h.update(self.to_bytes()?);
        Ok(h.finalize().into())
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statement {
    pub input: Vec<i64>,
    pub output: Vec<i64>,
    pub context: [u8; 32],
}
fn validate_statement(key: &VerifierKey, s: &Statement) -> Result<(), Error> {
    key.validate()?;
    for (id, v) in [
        (key.program.input(), &s.input),
        (key.program.output, &s.output),
    ] {
        let node = &key.program.nodes[id];
        if node.len != v.len() || v.iter().any(|&x| !node.range.contains(x)) {
            return Err(Error::InvalidStatement);
        }
    }
    Ok(())
}
#[derive(Serialize, Deserialize)]
enum GateProof {
    None,
    Relation(o::RelationProof),
    Matmul(Box<o::MatmulProof>),
    Round(Vec<o::RelationProof>),
    Lookup(lookup::Proof),
}
#[derive(Serialize, Deserialize)]
struct Proof {
    tensors: Vec<Tensor>,
    anchors: Vec<wb::Opening>,
    ranges: Vec<Vec<o::RelationProof>>,
    gates: Vec<GateProof>,
}
struct Layout {
    bits: Vec<Option<usize>>,
    round: Vec<Option<(usize, usize, usize, usize)>>,
    lengths: Vec<usize>,
}
fn layout(program: &Program) -> Layout {
    let mut lengths: Vec<_> = program.nodes.iter().map(|n| n.len).collect();
    let mut bits = vec![None; program.nodes.len()];
    let mut round = vec![None; program.nodes.len()];
    for (id, node) in program.nodes.iter().enumerate() {
        if !matches!(node.operation, Operation::Constant) {
            bits[id] = Some(lengths.len());
            lengths.push(node.len * node.range.width().next_power_of_two());
        }
        if let Operation::Round { divisor, .. } = node.operation {
            let width = (64 - (divisor - 1).leading_zeros() as usize).max(1);
            let r = lengths.len();
            lengths.extend([
                node.len,
                node.len,
                node.len * width.next_power_of_two(),
                node.len * width.next_power_of_two(),
            ]);
            round[id] = Some((r, r + 1, r + 2, r + 3));
        }
        if let Operation::DividePositive { denominator, .. } = node.operation {
            let width = program.nodes[denominator].range.width() + 1;
            let r = lengths.len();
            lengths.extend([
                node.len,
                node.len,
                node.len * width.next_power_of_two(),
                node.len * width.next_power_of_two(),
            ]);
            round[id] = Some((r, r + 1, r + 2, r + 3));
        }
    }
    Layout {
        bits,
        round,
        lengths,
    }
}
fn transcript(key: &VerifierKey, s: &Statement, roots: &[Tensor]) -> Result<wb::Challenger, Error> {
    let mut h = Sha256::new();
    h.update(DOMAIN);
    h.update(key.id()?);
    h.update(postcard::to_allocvec(s).map_err(|_| Error::ProofEncoding)?);
    h.update(postcard::to_allocvec(roots).map_err(|_| Error::ProofEncoding)?);
    let mut c = wb::challenger(&wb::pcs(5)?, &h.finalize());
    for t in roots {
        c.observe(t.root.clone());
    }
    Ok(c)
}
fn f(x: i64) -> F {
    if x < 0 {
        -F::from_u64(x.unsigned_abs())
    } else {
        F::from_u64(x as u64)
    }
}
fn e(x: i64) -> E {
    E::from(f(x))
}
fn gate_relation(
    node: &Node,
    id: usize,
    program: &Program,
    l: &Layout,
    c: &mut wb::Challenger,
) -> Option<o::Relation> {
    let log = node.len.trailing_zeros() as usize;
    let mut fs = vec![X::Oracle(View::identity(id, log))];
    let mut terms = vec![(E::ONE, vec![0])];
    match &node.operation {
        Operation::Linear {
            terms: xs,
            constant,
        } => {
            for (a, s) in xs {
                let i = fs.len();
                fs.push(X::Oracle(s.view()));
                terms.push((-e(*a), vec![i]));
            }
            terms.push((-e(*constant), vec![]));
        }
        Operation::Product { a, b } => {
            fs.push(X::Oracle(a.view()));
            fs.push(X::Oracle(b.view()));
            terms.push((-E::ONE, vec![1, 2]));
        }
        Operation::Relu { input } => {
            fs.push(X::Oracle(View::identity(*input, log)));
            let width = program.nodes[*input].range.width();
            let mut view = View::identity(l.bits[*input].unwrap(), log);
            view.coordinates.extend(
                (0..width.next_power_of_two().trailing_zeros())
                    .map(|b| C::Bit(((width - 1) >> b) & 1 == 1)),
            );
            fs.push(X::Oracle(view));
            terms.push((-E::ONE, vec![1, 2]));
        }
        Operation::SumRows {
            input,
            columns,
            actual_columns,
        } => {
            let cb = columns.trailing_zeros() as usize;
            let total = log + cb;
            let eq = X::Eq {
                point: o::sample(c, log),
                start: cb,
            };
            let out = View {
                tensor: id,
                coordinates: (cb..total).map(C::Var).collect(),
            };
            return Some(o::Relation {
                log: total,
                factors: vec![
                    X::Oracle(View::identity(*input, total)),
                    X::Oracle(out),
                    X::Prefix {
                        start: 0,
                        bits: cb,
                        limit: *actual_columns,
                    },
                    eq,
                ],
                terms: vec![
                    (E::ONE, vec![0, 2, 3]),
                    (-E::from_usize(*columns).inverse(), vec![1, 3]),
                ],
            });
        }
        _ => return None,
    }
    Some(dense::relation(log, fs, terms, c))
}
fn round_relations(
    node: &Node,
    id: usize,
    aux: (usize, usize, usize, usize),
    c: &mut wb::Challenger,
) -> Vec<o::Relation> {
    let Operation::Round {
        input,
        divisor,
        nearest,
    } = node.operation
    else {
        unreachable!()
    };
    let (r, d, rb, db) = aux;
    let log = node.len.trailing_zeros() as usize;
    let width = (64 - (divisor - 1).leading_zeros() as usize).max(1);
    let fs = [input, id, r, d]
        .into_iter()
        .map(|i| X::Oracle(View::identity(i, log)))
        .collect::<Vec<_>>();
    let mut out = vec![
        dense::relation(
            log,
            fs.clone(),
            vec![
                (E::ONE, vec![0]),
                (-E::from_u64(divisor), vec![1]),
                (-E::ONE, vec![2]),
                (E::from_u64(if nearest { divisor / 2 } else { 0 }), vec![]),
            ],
            c,
        ),
        dense::relation(
            log,
            fs,
            vec![
                (E::ONE, vec![2]),
                (E::ONE, vec![3]),
                (-E::from_u64(divisor - 1), vec![]),
            ],
            c,
        ),
    ];
    out.extend(dense::range(r, rb, width, 0, log, c));
    out.extend(dense::range(d, db, width, 0, log, c));
    out
}
fn divide_relations(
    node: &Node,
    id: usize,
    program: &Program,
    aux: (usize, usize, usize, usize),
    c: &mut wb::Challenger,
) -> Vec<o::Relation> {
    let Operation::DividePositive {
        numerator,
        denominator,
    } = node.operation
    else {
        unreachable!()
    };
    let (r, d, rb, db) = aux;
    let log = node.len.trailing_zeros() as usize;
    let width = program.nodes[denominator].range.width() + 1;
    let fs = [numerator, denominator, id, r, d]
        .into_iter()
        .map(|i| X::Oracle(View::identity(i, log)))
        .collect::<Vec<_>>();
    let mut rs = vec![
        dense::relation(
            log,
            fs.clone(),
            vec![
                (E::TWO, vec![0]),
                (E::ONE, vec![1]),
                (-E::TWO, vec![1, 2]),
                (-E::ONE, vec![3]),
            ],
            c,
        ),
        dense::relation(
            log,
            fs,
            vec![
                (E::ONE, vec![3]),
                (E::ONE, vec![4]),
                (-E::TWO, vec![1]),
                (E::ONE, vec![]),
            ],
            c,
        ),
    ];
    rs.extend(dense::range(r, rb, width, 0, log, c));
    rs.extend(dense::range(d, db, width, 0, log, c));
    rs
}
/// The expected model id must come from the verifier's own configuration.
pub fn verify(
    key: &VerifierKey,
    expected_id: [u8; 32],
    s: &Statement,
    bytes: &[u8],
) -> Result<(), Error> {
    validate_statement(key, s)?;
    if key.id()? != expected_id {
        return Err(Error::InvalidModel);
    }
    if bytes.len() > MAX_PROOF_BYTES {
        return Err(Error::ProofTooLarge);
    }
    catch_unwind(AssertUnwindSafe(|| {
        if bytes.get(..8) != Some(MAGIC) {
            return Err(Error::ProofEncoding);
        }
        let (p, tail): (Proof, _) =
            postcard::take_from_bytes(&bytes[8..]).map_err(|_| Error::ProofEncoding)?;
        if !tail.is_empty()
            || postcard::to_allocvec(&p).map_err(|_| Error::ProofEncoding)? != bytes[8..]
        {
            return Err(Error::ProofEncoding);
        }
        let l = layout(&key.program);
        if p.tensors.len() != l.lengths.len()
            || p.anchors.len() != 2
            || p.ranges.len() != key.program.nodes.len()
            || p.gates.len() != key.program.nodes.len()
        {
            return Err(Error::InvalidProof);
        }
        for (t, len) in p.tensors.iter().zip(&l.lengths) {
            t.validate()?;
            if t.len != *len {
                return Err(Error::InvalidProof);
            }
        }
        for (id, t) in &key.constants {
            if postcard::to_allocvec(t).unwrap() != postcard::to_allocvec(&p.tensors[*id]).unwrap()
            {
                return Err(Error::InvalidProof);
            }
        }
        let mut c = transcript(key, s, &p.tensors)?;
        for ((id, values), opening) in [
            (key.program.input(), &s.input),
            (key.program.output, &s.output),
        ]
        .into_iter()
        .zip(&p.anchors)
        {
            let point = o::sample(&mut c, values.len().trailing_zeros() as usize);
            if o::verify_open(&p.tensors[id], &point, opening, &mut c)?
                != o::eval(values.iter().copied().map(f), &point)
            {
                return Err(Error::InvalidProof);
            }
        }
        for (id, node) in key.program.nodes.iter().enumerate() {
            let rs = if let Some(bits) = l.bits[id] {
                dense::range(
                    id,
                    bits,
                    node.range.width(),
                    node.range.offset(),
                    node.len.trailing_zeros() as usize,
                    &mut c,
                )
            } else {
                vec![]
            };
            if rs.len() != p.ranges[id].len() {
                return Err(Error::InvalidProof);
            }
            for (r, proof) in rs.iter().zip(&p.ranges[id]) {
                o::verify_relation(r, &p.tensors, proof, &mut c)?;
            }
            match (&node.operation, &p.gates[id]) {
                (Operation::Input | Operation::Constant, GateProof::None) => {}
                (
                    Operation::Linear { .. }
                    | Operation::Product { .. }
                    | Operation::Relu { .. }
                    | Operation::SumRows { .. },
                    GateProof::Relation(proof),
                ) => {
                    o::verify_relation(
                        &gate_relation(node, id, &key.program, &l, &mut c).unwrap(),
                        &p.tensors,
                        proof,
                        &mut c,
                    )?;
                }
                (Operation::Matmul { a, b, m, k, n }, GateProof::Matmul(proof)) => {
                    o::verify_matmul(&p.tensors, [*a, *b, id], [*m, *k, *n], proof, &mut c)?
                }
                (Operation::Round { .. }, GateProof::Round(proofs)) => {
                    let rs = round_relations(node, id, l.round[id].unwrap(), &mut c);
                    if rs.len() != proofs.len() {
                        return Err(Error::InvalidProof);
                    }
                    for (r, proof) in rs.iter().zip(proofs) {
                        o::verify_relation(r, &p.tensors, proof, &mut c)?;
                    }
                }
                (Operation::DividePositive { .. }, GateProof::Round(proofs)) => {
                    let rs = divide_relations(node, id, &key.program, l.round[id].unwrap(), &mut c);
                    if rs.len() != proofs.len() {
                        return Err(Error::InvalidProof);
                    }
                    for (r, proof) in rs.iter().zip(proofs) {
                        o::verify_relation(r, &p.tensors, proof, &mut c)?;
                    }
                }
                (
                    Operation::Lookup {
                        indices,
                        table,
                        actual_len,
                    },
                    GateProof::Lookup(proof),
                ) => lookup::verify_core(
                    &lookup::Spec {
                        indices: *indices,
                        values: id,
                        table: *table,
                        actual_table_len: *actual_len,
                    },
                    &p.tensors,
                    proof,
                    &mut c,
                )?,
                _ => return Err(Error::InvalidProof),
            }
        }
        Ok(())
    }))
    .unwrap_or(Err(Error::InvalidProof))
}

#[cfg(feature = "prover")]
pub struct ProvingKey {
    key: VerifierKey,
    constants: Vec<(usize, std::sync::Arc<o::Data>, Vec<i64>)>,
}
#[cfg(feature = "prover")]
pub fn prepare(
    program: Program,
    constants: Vec<(usize, Vec<i64>)>,
) -> Result<(ProvingKey, VerifierKey), Error> {
    program.validate()?;
    let mut data = Vec::new();
    for (id, v) in constants {
        let node = program.nodes.get(id).ok_or(Error::InvalidModel)?;
        if !matches!(node.operation, Operation::Constant)
            || v.len() != node.len
            || v.iter().any(|&x| !node.range.contains(x))
        {
            return Err(Error::InvalidModel);
        }
        data.push((id, o::commit(v.iter().copied().map(f).collect())?, v));
    }
    let key = VerifierKey {
        program,
        constants: data
            .iter()
            .map(|(id, d, _)| (*id, d.tensor.clone()))
            .collect(),
    };
    key.validate()?;
    Ok((
        ProvingKey {
            key: key.clone(),
            constants: data,
        },
        key,
    ))
}
#[cfg(feature = "prover")]
fn bit_data(values: &[i64], width: usize, offset: u64) -> Result<std::sync::Arc<o::Data>, Error> {
    let cols = width.next_power_of_two();
    o::commit(
        (0..cols)
            .flat_map(|b| {
                values.iter().map(move |&x| {
                    F::from_bool(b < width && (((x as i128 + offset as i128) as u64 >> b) & 1) == 1)
                })
            })
            .collect(),
    )
}
#[cfg(feature = "prover")]
fn evaluate(pk: &ProvingKey, input: &[i64]) -> Result<Vec<Vec<i64>>, Error> {
    let mut values: Vec<Vec<i64>> = Vec::new();
    for (id, node) in pk.key.program.nodes.iter().enumerate() {
        let result = match &node.operation {
            Operation::Input => input.to_vec(),
            Operation::Constant => pk
                .constants
                .iter()
                .find(|(i, _, _)| *i == id)
                .ok_or(Error::InvalidModel)?
                .2
                .clone(),
            Operation::Linear { terms, constant } => (0..node.len)
                .map(|i| {
                    let x = terms
                        .iter()
                        .map(|(a, s)| *a as i128 * values[s.tensor][s.index(i)] as i128)
                        .sum::<i128>()
                        + *constant as i128;
                    i64::try_from(x).map_err(|_| Error::InvalidStatement)
                })
                .collect::<Result<Vec<_>, _>>()?,
            Operation::Product { a, b } => (0..node.len)
                .map(|i| {
                    i64::try_from(
                        values[a.tensor][a.index(i)] as i128 * values[b.tensor][b.index(i)] as i128,
                    )
                    .map_err(|_| Error::InvalidStatement)
                })
                .collect::<Result<Vec<_>, _>>()?,
            Operation::Matmul { a, b, m: _, k, n } => (0..node.len)
                .map(|i| {
                    let row = i / n;
                    let col = i % n;
                    let x = (0..*k)
                        .map(|j| values[*a][row * k + j] as i128 * values[*b][j * n + col] as i128)
                        .sum::<i128>();
                    i64::try_from(x).map_err(|_| Error::InvalidStatement)
                })
                .collect::<Result<Vec<_>, _>>()?,
            Operation::Round {
                input,
                divisor,
                nearest,
            } => values[*input]
                .iter()
                .map(|&x| {
                    ((x as i128 + if *nearest { *divisor as i128 / 2 } else { 0 })
                        .div_euclid(*divisor as i128)) as i64
                })
                .collect(),
            Operation::Relu { input } => values[*input].iter().map(|&x| x.max(0)).collect(),
            Operation::SumRows {
                input,
                columns,
                actual_columns,
            } => values[*input]
                .chunks_exact(*columns)
                .map(|row| {
                    i64::try_from(
                        row[..*actual_columns]
                            .iter()
                            .map(|&x| x as i128)
                            .sum::<i128>(),
                    )
                    .map_err(|_| Error::InvalidStatement)
                })
                .collect::<Result<Vec<_>, _>>()?,
            Operation::DividePositive {
                numerator,
                denominator,
            } => values[*numerator]
                .iter()
                .zip(&values[*denominator])
                .map(|(&a, &b)| {
                    if b <= 0 {
                        return Err(Error::InvalidStatement);
                    }
                    Ok(((2 * a as i128 + b as i128) / (2 * b as i128)) as i64)
                })
                .collect::<Result<Vec<_>, _>>()?,
            Operation::Lookup {
                indices,
                table,
                actual_len,
            } => values[*indices]
                .iter()
                .map(|&i| {
                    if i < 0 || i as usize >= *actual_len {
                        return Err(Error::InvalidStatement);
                    }
                    Ok(values[*table][i as usize])
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        if result.len() != node.len || result.iter().any(|&x| !node.range.contains(x)) {
            return Err(Error::InvalidStatement);
        }
        values.push(result);
    }
    Ok(values)
}
#[cfg(feature = "prover")]
pub fn prove(
    pk: &ProvingKey,
    input: Vec<i64>,
    context: [u8; 32],
) -> Result<(Statement, Vec<u8>), Error> {
    let values = evaluate(pk, &input)?;
    prove_values(pk, input, context, values)
}
#[cfg(feature = "prover")]
fn prove_values(
    pk: &ProvingKey,
    input: Vec<i64>,
    context: [u8; 32],
    values: Vec<Vec<i64>>,
) -> Result<(Statement, Vec<u8>), Error> {
    let program = &pk.key.program;
    let s = Statement {
        input,
        output: values[program.output].clone(),
        context,
    };
    validate_statement(&pk.key, &s)?;
    let l = layout(program);
    let mut data = Vec::new();
    for (id, v) in values.iter().enumerate() {
        data.push(
            if let Some((_, d, _)) = pk.constants.iter().find(|(i, _, _)| *i == id) {
                d.clone()
            } else {
                o::commit(v.iter().copied().map(f).collect())?
            },
        );
    }
    for (id, node) in program.nodes.iter().enumerate() {
        if l.bits[id].is_some() {
            data.push(bit_data(
                &values[id],
                node.range.width(),
                node.range.offset(),
            )?);
        }
        if let Operation::Round {
            input,
            divisor,
            nearest,
        } = node.operation
        {
            let rem: Vec<i64> = values[input]
                .iter()
                .zip(&values[id])
                .map(|(&x, &q)| {
                    (x as i128 + if nearest { divisor as i128 / 2 } else { 0 }
                        - divisor as i128 * q as i128) as i64
                })
                .collect();
            let delta: Vec<i64> = rem.iter().map(|&r| divisor as i64 - 1 - r).collect();
            let width = (64 - (divisor - 1).leading_zeros() as usize).max(1);
            data.push(o::commit(rem.iter().copied().map(f).collect())?);
            data.push(o::commit(delta.iter().copied().map(f).collect())?);
            data.push(bit_data(&rem, width, 0)?);
            data.push(bit_data(&delta, width, 0)?);
        }
        if let Operation::DividePositive {
            numerator,
            denominator,
        } = node.operation
        {
            let rem: Vec<i64> = values[numerator]
                .iter()
                .zip(&values[denominator])
                .zip(&values[id])
                .map(|((&a, &b), &q)| {
                    (2 * a as i128 + b as i128 - 2 * b as i128 * q as i128) as i64
                })
                .collect();
            let delta: Vec<i64> = values[denominator]
                .iter()
                .zip(&rem)
                .map(|(&b, &r)| 2 * b - 1 - r)
                .collect();
            let width = program.nodes[denominator].range.width() + 1;
            data.push(o::commit(rem.iter().copied().map(f).collect())?);
            data.push(o::commit(delta.iter().copied().map(f).collect())?);
            data.push(bit_data(&rem, width, 0)?);
            data.push(bit_data(&delta, width, 0)?);
        }
    }
    let tensors: Vec<_> = data.iter().map(|d| d.tensor.clone()).collect();
    let mut c = transcript(&pk.key, &s, &tensors)?;
    let mut anchors = Vec::new();
    for id in [program.input(), program.output] {
        let point = o::sample(&mut c, program.nodes[id].len.trailing_zeros() as usize);
        anchors.push(o::open(&data[id], &point, &mut c)?.0);
    }
    let mut ranges = Vec::new();
    let mut gates = Vec::new();
    for (id, node) in program.nodes.iter().enumerate() {
        let rs = if let Some(bits) = l.bits[id] {
            dense::range(
                id,
                bits,
                node.range.width(),
                node.range.offset(),
                node.len.trailing_zeros() as usize,
                &mut c,
            )
        } else {
            vec![]
        };
        ranges.push(
            rs.iter()
                .map(|r| o::prove_relation(r, &data, &mut c))
                .collect::<Result<Vec<_>, _>>()?,
        );
        gates.push(match &node.operation {
            Operation::Input | Operation::Constant => GateProof::None,
            Operation::Linear { .. }
            | Operation::Product { .. }
            | Operation::Relu { .. }
            | Operation::SumRows { .. } => GateProof::Relation(o::prove_relation(
                &gate_relation(node, id, program, &l, &mut c).unwrap(),
                &data,
                &mut c,
            )?),
            Operation::Matmul { a, b, m, k, n } => GateProof::Matmul(Box::new(o::prove_matmul(
                &data,
                [*a, *b, id],
                [*m, *k, *n],
                &mut c,
            )?)),
            Operation::Round { .. } => GateProof::Round(
                round_relations(node, id, l.round[id].unwrap(), &mut c)
                    .iter()
                    .map(|r| o::prove_relation(r, &data, &mut c))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Operation::DividePositive { .. } => GateProof::Round(
                divide_relations(node, id, program, l.round[id].unwrap(), &mut c)
                    .iter()
                    .map(|r| o::prove_relation(r, &data, &mut c))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Operation::Lookup {
                indices,
                table,
                actual_len,
            } => GateProof::Lookup(lookup::prove_core(
                &lookup::Spec {
                    indices: *indices,
                    values: id,
                    table: *table,
                    actual_table_len: *actual_len,
                },
                &data,
                &mut c,
            )?),
        });
    }
    let p = Proof {
        tensors,
        anchors,
        ranges,
        gates,
    };
    let mut bytes = MAGIC.to_vec();
    bytes.extend(postcard::to_allocvec(&p).map_err(|_| Error::ProofEncoding)?);
    if bytes.len() > MAX_PROOF_BYTES {
        return Err(Error::ProofTooLarge);
    }
    Ok((s, bytes))
}
#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::*;
    fn fixture() -> (ProvingKey, VerifierKey) {
        let nodes = vec![
            Node {
                len: 4,
                range: Range::Signed(5),
                operation: Operation::Input,
            },
            Node {
                len: 4,
                range: Range::Signed(4),
                operation: Operation::Constant,
            },
            Node {
                len: 4,
                range: Range::Signed(10),
                operation: Operation::Matmul {
                    a: 0,
                    b: 1,
                    m: 2,
                    k: 2,
                    n: 2,
                },
            },
            Node {
                len: 4,
                range: Range::Signed(10),
                operation: Operation::Round {
                    input: 2,
                    divisor: 3,
                    nearest: true,
                },
            },
            Node {
                len: 4,
                range: Range::Unsigned(3),
                operation: Operation::Linear {
                    terms: vec![(1, Source::identity(3, 4))],
                    constant: 3,
                },
            },
            Node {
                len: 8,
                range: Range::Signed(8),
                operation: Operation::Constant,
            },
            Node {
                len: 4,
                range: Range::Signed(8),
                operation: Operation::Lookup {
                    indices: 4,
                    table: 5,
                    actual_len: 7,
                },
            },
        ];
        prepare(
            Program { nodes, output: 6 },
            vec![
                (1, vec![1, 0, 0, 1]),
                (5, vec![11, 22, 33, 44, 55, 66, 77, 0]),
            ],
        )
        .unwrap()
    }
    #[test]
    fn connected_integer_graph_rejects_modified_execution_and_missing_constraints() {
        let (pk, key) = fixture();
        let input = vec![-4, -1, 2, 8];
        let (s, bytes) = prove(&pk, input.clone(), [7; 32]).unwrap();
        assert_eq!(s.output, vec![33, 44, 55, 77]);
        verify(&key, key.id().unwrap(), &s, &bytes).unwrap();
        let mut other = s.clone();
        other.context[0] ^= 1;
        assert!(verify(&key, key.id().unwrap(), &other, &bytes).is_err());
        let mut values = evaluate(&pk, &input).unwrap();
        values[3][0] += 1;
        let (s, bad) = prove_values(&pk, input, [7; 32], values).unwrap();
        assert!(verify(&key, key.id().unwrap(), &s, &bad).is_err());
        let mut p: Proof = postcard::from_bytes(&bytes[8..]).unwrap();
        p.ranges[3].clear();
        let mut bad = MAGIC.to_vec();
        bad.extend(postcard::to_allocvec(&p).unwrap());
        assert!(verify(&key, key.id().unwrap(), &s, &bad).is_err());
    }
    #[test]
    fn static_integer_alias_guards_reject_unsafe_profiles() {
        let mut p = fixture().1.program;
        p.nodes[0].range = Range::Signed(62);
        p.nodes[1].range = Range::Signed(62);
        assert!(p.validate().is_err());
    }
}

#[cfg(all(test, feature = "prover"))]
mod transformer_gate_attacks {
    use super::*;
    fn fixture() -> (ProvingKey, VerifierKey) {
        let nodes = vec![
            Node {
                len: 4,
                range: Range::Signed(6),
                operation: Operation::Input,
            },
            Node {
                len: 4,
                range: Range::Unsigned(6),
                operation: Operation::Relu { input: 0 },
            },
            Node {
                len: 4,
                range: Range::Unsigned(3),
                operation: Operation::Constant,
            },
            Node {
                len: 4,
                range: Range::Unsigned(6),
                operation: Operation::DividePositive {
                    numerator: 1,
                    denominator: 2,
                },
            },
            Node {
                len: 1,
                range: Range::Unsigned(8),
                operation: Operation::SumRows {
                    input: 3,
                    columns: 4,
                    actual_columns: 3,
                },
            },
            Node {
                len: 4,
                range: Range::Signed(6),
                operation: Operation::Round {
                    input: 0,
                    divisor: 3,
                    nearest: true,
                },
            },
            Node {
                len: 4,
                range: Range::Signed(7),
                operation: Operation::Linear {
                    terms: vec![(1, Source::identity(3, 4)), (1, Source::identity(5, 4))],
                    constant: 0,
                },
            },
        ];
        prepare(Program { nodes, output: 6 }, vec![(2, vec![1, 2, 3, 7])]).unwrap()
    }
    #[test]
    fn sign_non_power_of_two_remainders_and_masked_reductions_are_constrained() {
        let (pk, key) = fixture();
        let input = vec![-17, -4, 5, 17];
        let (s, bytes) = prove(&pk, input.clone(), [2; 32]).unwrap();
        assert_eq!(s.output, vec![-6, -1, 4, 8]);
        verify(&key, key.id().unwrap(), &s, &bytes).unwrap();
        // Each modified prover regenerates all auxiliary commitments/proofs.
        // The output alone cannot distinguish the hidden row-reduction attack.
        for id in [1, 3, 4, 5] {
            let mut v = evaluate(&pk, &input).unwrap();
            v[id][0] += 1;
            let (s, bad) = prove_values(&pk, input.clone(), [2; 32], v).unwrap();
            assert!(
                verify(&key, key.id().unwrap(), &s, &bad).is_err(),
                "accepted forged node {id}"
            );
        }
        let mut p: Proof = postcard::from_bytes(&bytes[8..]).unwrap();
        p.gates[3] = GateProof::None;
        let mut bad = MAGIC.to_vec();
        bad.extend(postcard::to_allocvec(&p).unwrap());
        assert!(verify(&key, key.id().unwrap(), &s, &bad).is_err());
    }
    #[test]
    fn zero_softmax_denominator_cannot_be_hidden_by_a_modified_prover() {
        let (_, key) = fixture();
        let (pk, key) = prepare(key.program, vec![(2, vec![0; 4])]).unwrap();
        let input = vec![-17, -4, 5, 17];
        assert!(prove(&pk, input.clone(), [0; 32]).is_err());
        let values = vec![
            input.clone(),
            vec![0, 0, 5, 17],
            vec![0; 4],
            vec![0; 4],
            vec![0],
            vec![-6, -1, 2, 6],
            vec![-6, -1, 2, 6],
        ];
        let (s, bad) = prove_values(&pk, input, [0; 32], values).unwrap();
        assert!(verify(&key, key.id().unwrap(), &s, &bad).is_err());
    }
    #[test]
    fn malformed_programs_reject_before_proving() {
        let (_, key) = fixture();
        let mut p = key.program.clone();
        p.nodes[1].operation = Operation::Relu { input: 2 };
        assert!(p.validate().is_err());
        let mut p = key.program.clone();
        p.nodes[5].operation = Operation::Round {
            input: 0,
            divisor: 0,
            nearest: true,
        };
        assert!(p.validate().is_err());
        let mut p = key.program.clone();
        p.nodes[6].operation = Operation::Linear {
            terms: vec![(
                1,
                Source {
                    tensor: 0,
                    coordinates: vec![Bit::Variable(0), Bit::Variable(0)],
                },
            )],
            constant: 0,
        };
        assert!(p.validate().is_err());
        let mut p = key.program.clone();
        p.nodes[4].operation = Operation::SumRows {
            input: 3,
            columns: 4,
            actual_columns: 5,
        };
        assert!(p.validate().is_err());
        let mut p = key.program;
        p.nodes[1].operation = Operation::Relu { input: 0 };
        p.nodes[0].operation = Operation::Constant;
        p.nodes[2].operation = Operation::Input;
        assert!(p.validate().is_err());
    }
}
