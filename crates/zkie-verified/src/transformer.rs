//! GPT-2 pre-norm Transformer compiler for the authenticated integer graph.
//! Public input is the precomputed embedding matrix; logits are the output.
use crate::{
    graph::{self, Bit, Node, Operation as Op, Program, Range as R, Source},
    Error,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub sequence: usize,
    pub width: usize,
    pub real_width: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_width: usize,
    pub layers: usize,
    pub vocabulary: usize,
    pub gelu_offset: i64,
    pub exp_offset: i64,
}
impl Config {
    pub fn gpt2(sequence: usize) -> Self {
        Self {
            sequence,
            width: 1024,
            real_width: 768,
            hidden: 4096,
            heads: 12,
            head_width: 64,
            layers: 12,
            vocabulary: 65536,
            gelu_offset: 1 << 23,
            exp_offset: 1 << 21,
        }
    }
    pub fn validate(&self) -> Result<(), Error> {
        if !self.sequence.is_power_of_two()
            || self.sequence > 32
            || !self.width.is_power_of_two()
            || self.width > 1024
            || self.real_width == 0
            || self.real_width > self.width
            || !self.hidden.is_power_of_two()
            || self.hidden > 4096
            || self.heads == 0
            || self.heads > 16
            || !self.head_width.is_power_of_two()
            || self.head_width > 64
            || self
                .heads
                .checked_mul(self.head_width)
                .is_none_or(|n| n > self.width)
            || self.layers == 0
            || self.layers > 12
            || !self.vocabulary.is_power_of_two()
            || self.vocabulary > 65536
            || self.gelu_offset < 0
            || self.exp_offset < 0
            || self.gelu_offset > 1 << 24
            || self.exp_offset > 1 << 24
        {
            return Err(Error::InvalidModel);
        }
        Ok(())
    }
}
pub type CompiledModel = (Program, Vec<(usize, Vec<i64>)>);

struct Builder {
    nodes: Vec<Node>,
    constants: Vec<(usize, Vec<i64>)>,
}
impl Builder {
    fn push(&mut self, len: usize, range: R, operation: Op) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            len,
            range,
            operation,
        });
        id
    }
    fn constant(&mut self, v: Vec<i64>) -> usize {
        let mut bits = 1;
        while bits < 62
            && v.iter()
                .any(|&x| x < -(1i64 << (bits - 1)) || x >= (1i64 << (bits - 1)))
        {
            bits += 1;
        }
        let id = self.push(v.len(), R::Signed(bits), Op::Constant);
        self.constants.push((id, v));
        id
    }
    fn source(&self, id: usize) -> Source {
        Source::identity(id, self.nodes[id].len)
    }
    fn linear(&mut self, len: usize, r: R, terms: Vec<(i64, Source)>, constant: i64) -> usize {
        self.push(len, r, Op::Linear { terms, constant })
    }
    fn affine(&mut self, x: usize, r: R, a: i64, b: i64) -> usize {
        self.linear(self.nodes[x].len, r, vec![(a, self.source(x))], b)
    }
    fn add(&mut self, a: usize, b: usize, r: R) -> usize {
        self.linear(
            self.nodes[a].len,
            r,
            vec![(1, self.source(a)), (1, self.source(b))],
            0,
        )
    }
    fn round(&mut self, x: usize, d: u64, r: R, nearest: bool) -> usize {
        self.push(
            self.nodes[x].len,
            r,
            Op::Round {
                input: x,
                divisor: d,
                nearest,
            },
        )
    }
    fn product(&mut self, len: usize, a: Source, b: Source, r: R) -> usize {
        self.push(len, r, Op::Product { a, b })
    }
    fn relu(&mut self, x: usize, r: R) -> usize {
        self.push(self.nodes[x].len, r, Op::Relu { input: x })
    }
    fn rows(&self, id: usize, columns: usize) -> Source {
        let cb = columns.trailing_zeros() as usize;
        Source {
            tensor: id,
            coordinates: (cb..cb + self.nodes[id].len.trailing_zeros() as usize)
                .map(Bit::Variable)
                .collect(),
        }
    }
    fn cols(&self, id: usize) -> Source {
        self.source(id)
    }
    fn matmul(&mut self, a: usize, b: usize, m: usize, k: usize, n: usize) -> usize {
        self.push(m * n, R::Signed(62), Op::Matmul { a, b, m, k, n })
    }
    fn projection(
        &mut self,
        x: usize,
        w: usize,
        b: Option<usize>,
        m: usize,
        k: usize,
        n: usize,
    ) -> usize {
        let raw = self.matmul(x, w, m, k, n);
        let q = self.round(raw, 1 << 16, R::Signed(25), true);
        if let Some(b) = b {
            self.linear(
                m * n,
                R::Signed(25),
                vec![(1, self.source(q)), (1, self.cols(b))],
                0,
            )
        } else {
            q
        }
    }
    fn clamp(&mut self, x: usize, limit: i64) -> usize {
        let positive = self.relu(x, R::Signed(36));
        let shifted = self.affine(positive, R::Signed(36), 1, -limit);
        let excess = self.relu(shifted, R::Signed(36));
        let width = (64 - (limit as u64).leading_zeros()).max(1) as u8;
        self.linear(
            self.nodes[x].len,
            R::Unsigned(width),
            vec![(1, self.source(positive)), (-1, self.source(excess))],
            0,
        )
    }
    fn index(&mut self, x: usize, offset: i64, len: usize) -> usize {
        let shifted = self.affine(x, R::Signed(36), 1, offset);
        self.clamp(shifted, (len - 1) as i64)
    }
    fn lookup(&mut self, x: usize, table: usize, actual_len: usize, r: R) -> usize {
        self.push(
            self.nodes[x].len,
            r,
            Op::Lookup {
                indices: x,
                table,
                actual_len,
            },
        )
    }
    fn sum(&mut self, x: usize, columns: usize, actual_columns: usize, r: R) -> usize {
        self.push(
            self.nodes[x].len / columns,
            r,
            Op::SumRows {
                input: x,
                columns,
                actual_columns,
            },
        )
    }
    fn rsqrt_index(&mut self, var: usize) -> usize {
        let raw = self.round(var, 1 << 18, R::Unsigned(34), true);
        let a = self.affine(raw, R::Signed(36), 1, -(1 << 20));
        let a = self.relu(a, R::Signed(36));
        let b = self.round(a, 256, R::Signed(36), false);
        let t = self.affine(raw, R::Signed(36), 1, -((1 << 20) + (1i64 << 30)));
        let t = self.relu(t, R::Signed(36));
        let u = self.round(t, 256, R::Signed(36), false);
        let v = self.round(t, 65536, R::Signed(36), false);
        self.linear(
            self.nodes[var].len,
            R::Unsigned(24),
            vec![
                (1, self.source(raw)),
                (-1, self.source(a)),
                (1, self.source(b)),
                (-1, self.source(u)),
                (1, self.source(v)),
            ],
            0,
        )
    }
    fn norm(
        &mut self,
        x: usize,
        w: usize,
        b: usize,
        table: usize,
        actual: usize,
        c: &Config,
    ) -> usize {
        let len = c.sequence * c.width;
        let sum = self.sum(x, c.width, c.real_width, R::Signed(36));
        let mean = self.round(sum, c.real_width as u64, R::Signed(25), true);
        let centered = self.linear(
            len,
            R::Signed(26),
            vec![(1, self.source(x)), (-1, self.rows(mean, c.width))],
            0,
        );
        let square = self.product(
            len,
            self.source(centered),
            self.source(centered),
            R::Unsigned(51),
        );
        let total = self.sum(square, c.width, c.real_width, R::Unsigned(61));
        let var = self.round(total, c.real_width as u64, R::Unsigned(51), true);
        let idx = self.rsqrt_index(var);
        let rstd = self.lookup(idx, table, actual, R::Signed(27));
        let normalized = self.product(
            len,
            self.source(centered),
            self.rows(rstd, c.width),
            R::Signed(40),
        );
        let raw = self.product(len, self.source(normalized), self.cols(w), R::Signed(62));
        let q = self.round(raw, 1 << 32, R::Signed(29), true);
        self.linear(
            len,
            R::Signed(25),
            vec![(1, self.source(q)), (1, self.cols(b))],
            0,
        )
    }
    fn row_max(&mut self, x: usize, columns: usize) -> usize {
        let mut value = x;
        let mut n = columns;
        while n > 1 {
            let outlen = self.nodes[value].len / 2;
            let log = outlen.trailing_zeros() as usize;
            let source = |odd| Source {
                tensor: value,
                coordinates: std::iter::once(Bit::Fixed(odd))
                    .chain((0..log).map(Bit::Variable))
                    .collect(),
            };
            let even = source(false);
            let odd = source(true);
            let diff = self.linear(outlen, R::Signed(34), vec![(1, odd), (-1, even.clone())], 0);
            let positive = self.relu(diff, R::Signed(34));
            value = self.linear(
                outlen,
                R::Signed(33),
                vec![(1, even), (1, self.source(positive))],
                0,
            );
            n /= 2;
        }
        value
    }
    fn transpose(&mut self, x: usize, rows: usize, cols: usize) -> usize {
        let rb = rows.trailing_zeros() as usize;
        let cb = cols.trailing_zeros() as usize;
        let coordinates = (0..cb)
            .map(|i| Bit::Variable(rb + i))
            .chain((0..rb).map(Bit::Variable))
            .collect();
        self.linear(
            rows * cols,
            self.nodes[x].range,
            vec![(
                1,
                Source {
                    tensor: x,
                    coordinates,
                },
            )],
            0,
        )
    }
}
/// Compile exactly the padded GPT-2 architecture used by this repository.
/// The loader returns signed integers from named little-endian i32 files. The
/// returned program and constants can be persisted without doing any proving.
pub fn compile(
    c: &Config,
    mut load: impl FnMut(&str) -> Result<Vec<i64>, Error>,
) -> Result<CompiledModel, Error> {
    c.validate()?;
    let mut b = Builder {
        nodes: vec![],
        constants: vec![],
    };
    let mut get = |name: &str, expected: Option<usize>| {
        let v = load(name)?;
        if v.is_empty() || expected.is_some_and(|n| n != v.len()) {
            return Err(Error::InvalidModel);
        }
        Ok(v)
    };
    let mut table = |b: &mut Builder, name: &str| {
        let mut v = get(name, None)?;
        let actual = v.len();
        let padded = actual
            .checked_next_power_of_two()
            .ok_or(Error::InvalidModel)?;
        if padded > 1 << 26 {
            return Err(Error::InvalidModel);
        }
        v.resize(padded, 0);
        Ok((b.constant(v), actual))
    };
    let input = b.push(c.sequence * c.width, R::Signed(25), Op::Input);
    let (exp, exp_len) = table(&mut b, "exp_table")?;
    let (gelu, gelu_len) = table(&mut b, "gelu_table")?;
    let (rsqrt, rsqrt_len) = table(&mut b, "rsqrt_table")?;
    let mask = b.constant(
        (0..c.sequence * c.sequence)
            .map(|i| {
                if i % c.sequence > i / c.sequence {
                    -(1 << 30)
                } else {
                    0
                }
            })
            .collect(),
    );
    let mut current = input;
    for layer in 0..c.layers {
        let prefix = format!("L{layer}_");
        let mut weight = |name: &str, len| get(&format!("{prefix}{name}"), Some(len));
        let q = weight("q_w", c.width * c.width)?;
        let k = weight("k_w", c.width * c.width)?;
        let v = weight("v_w", c.width * c.width)?;
        let out = weight("o_proj_w", c.width * c.width)?;
        let qb = weight("q_b", c.width)?;
        let kb = weight("k_b", c.width)?;
        let vb = weight("v_b", c.width)?;
        let ob = b.constant(weight("o_proj_b", c.width)?);
        let ln1w = b.constant(weight("ln1_w", c.width)?);
        let ln1b = b.constant(weight("ln1_b", c.width)?);
        let h = b.norm(current, ln1w, ln1b, rsqrt, rsqrt_len, c);
        let mut heads = Vec::new();
        for head in 0..c.heads {
            let start = head * c.head_width;
            let slice = |w: &[i64]| {
                (0..c.width)
                    .flat_map(|row| {
                        w[row * c.width + start..row * c.width + start + c.head_width]
                            .iter()
                            .copied()
                    })
                    .collect::<Vec<_>>()
            };
            let qw = b.constant(slice(&q));
            let kw = b.constant(slice(&k));
            let vw = b.constant(slice(&v));
            let ow = b.constant(out[start * c.width..(start + c.head_width) * c.width].to_vec());
            let qbias = b.constant(qb[start..start + c.head_width].to_vec());
            let kbias = b.constant(kb[start..start + c.head_width].to_vec());
            let vbias = b.constant(vb[start..start + c.head_width].to_vec());
            let qh = b.projection(h, qw, Some(qbias), c.sequence, c.width, c.head_width);
            let kh = b.projection(h, kw, Some(kbias), c.sequence, c.width, c.head_width);
            let vh = b.projection(h, vw, Some(vbias), c.sequence, c.width, c.head_width);
            let qs = b.affine(qh, R::Signed(42), 23170, 0);
            let qs = b.round(qs, 1 << 16, R::Signed(25), true);
            let ks = b.affine(kh, R::Signed(42), 23170, 0);
            let ks = b.round(ks, 1 << 16, R::Signed(25), true);
            let kt = b.transpose(ks, c.sequence, c.head_width);
            let score = b.matmul(qs, kt, c.sequence, c.head_width, c.sequence);
            let score = b.round(score, 1 << 16, R::Signed(30), true);
            let masked = b.add(score, mask, R::Signed(33));
            let max = b.row_max(masked, c.sequence);
            let diff = b.linear(
                c.sequence * c.sequence,
                R::Signed(34),
                vec![(1, b.source(masked)), (-1, b.rows(max, c.sequence))],
                0,
            );
            let idx = b.index(diff, c.exp_offset, exp_len);
            let expv = b.lookup(idx, exp, exp_len, R::Unsigned(17));
            let sums = b.sum(expv, c.sequence, c.sequence, R::Unsigned(28));
            let denom = b.linear(
                c.sequence * c.sequence,
                R::Unsigned(28),
                vec![(1, b.rows(sums, c.sequence))],
                0,
            );
            let numerator = b.affine(expv, R::Unsigned(34), 1 << 16, 0);
            let probs = b.push(
                c.sequence * c.sequence,
                R::Unsigned(17),
                Op::DividePositive {
                    numerator,
                    denominator: denom,
                },
            );
            let attn = b.matmul(probs, vh, c.sequence, c.sequence, c.head_width);
            let attn = b.round(attn, 1 << 16, R::Signed(25), true);
            heads.push(b.projection(attn, ow, None, c.sequence, c.head_width, c.width));
        }
        let mut attn = heads[0];
        for &head in &heads[1..] {
            attn = b.add(attn, head, R::Signed(25));
        }
        let attn = b.linear(
            c.sequence * c.width,
            R::Signed(25),
            vec![(1, b.source(attn)), (1, b.cols(ob))],
            0,
        );
        let residual = b.add(current, attn, R::Signed(25));
        let ln2w = b.constant(weight("ln2_w", c.width)?);
        let ln2b = b.constant(weight("ln2_b", c.width)?);
        let h = b.norm(residual, ln2w, ln2b, rsqrt, rsqrt_len, c);
        let fcw = b.constant(weight("fc_w", c.width * c.hidden)?);
        let fcb = b.constant(weight("fc_b", c.hidden)?);
        let fc = b.projection(h, fcw, Some(fcb), c.sequence, c.width, c.hidden);
        let idx = b.index(fc, c.gelu_offset, gelu_len);
        let act = b.lookup(idx, gelu, gelu_len, R::Signed(25));
        let pw = b.constant(weight("proj_w", c.hidden * c.width)?);
        let pb = b.constant(weight("proj_b", c.width)?);
        let proj = b.projection(act, pw, Some(pb), c.sequence, c.hidden, c.width);
        current = b.add(residual, proj, R::Signed(25));
    }
    let w = b.constant(get("ln_f_w", Some(c.width))?);
    let bias = b.constant(get("ln_f_b", Some(c.width))?);
    let h = b.norm(current, w, bias, rsqrt, rsqrt_len, c);
    let lm = b.constant(get("lm_head_w", Some(c.width * c.vocabulary))?);
    let output = b.matmul(h, lm, c.sequence, c.width, c.vocabulary);
    Ok((
        Program {
            nodes: b.nodes,
            output,
        },
        b.constants,
    ))
}
/// Strict file loader: rejects partial integers and unexpected path components.
pub fn compile_directory(c: &Config, dir: &std::path::Path) -> Result<CompiledModel, Error> {
    compile(c, |name| {
        if !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err(Error::InvalidModel);
        }
        let path = dir.join(format!("{name}_i32.bin"));
        let meta = std::fs::metadata(&path).map_err(|_| Error::InvalidModel)?;
        if meta.len() > 1 << 29 || meta.len() % 4 != 0 {
            return Err(Error::InvalidModel);
        }
        let bytes = std::fs::read(path).map_err(|_| Error::InvalidModel)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()) as i64)
            .collect())
    })
}
/// Convenience preparation; commitment generation is intentionally explicit.
pub fn prepare(
    c: &Config,
    dir: &std::path::Path,
) -> Result<(graph::ProvingKey, graph::VerifierKey), Error> {
    let (p, constants) = compile_directory(c, dir)?;
    graph::prepare(p, constants)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_nonzero_transformer_matches_hand_computed_integer_execution() {
        let c = Config {
            sequence: 2,
            width: 2,
            real_width: 2,
            hidden: 2,
            heads: 1,
            head_width: 2,
            layers: 1,
            vocabulary: 2,
            gelu_offset: 2,
            exp_offset: 2,
        };
        let (p, constants) = compile(&c, |name| {
            Ok(match name {
                "exp_table" => vec![1, 2, 3, 4, 5],
                "gelu_table" => vec![-2, -1, 0, 1, 2, 3, 4, 5],
                "rsqrt_table" => vec![65536],
                n if n.ends_with("_b") => vec![0, 0],
                n if n.contains("ln") => vec![65536, 65536],
                _ => vec![65536, 0, 0, 65536],
            })
        })
        .unwrap();
        for end in 1..=p.nodes.len() {
            let prefix = Program {
                nodes: p.nodes[..end].to_vec(),
                output: end - 1,
            };
            if let Err(e) = prefix.validate() {
                panic!("invalid graph prefix at node {}: {:?}", end - 1, e);
            }
        }
        let (pk, key) = graph::prepare(p, constants).unwrap();
        let (s, proof) = graph::prove(&pk, vec![2, -2, 4, -4], [8; 32]).unwrap();
        assert_eq!(s.output, vec![458752, -524288, 655360, -720896]);
        graph::verify(&key, key.id().unwrap(), &s, &proof).unwrap();
        let mut changed = s.clone();
        changed.output[0] += 1;
        assert!(graph::verify(&key, key.id().unwrap(), &changed, &proof).is_err());
    }
}
