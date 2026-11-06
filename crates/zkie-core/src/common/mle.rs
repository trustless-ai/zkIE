//! Multilinear-extension evaluation over the Goldilocks field.
//!
//! Conventions: variable `0` is the least-significant bit of the flattened
//! index; `partial_eval` fixes the *first* `fix.len()` variables.

use crate::common::field::{Goldilocks, PrimeCharacteristicRing};

pub fn eval(values: &[Goldilocks], point: &[Goldilocks]) -> Goldilocks {
    let t = point.len();
    assert_eq!(values.len(), 1 << t, "mle eval: length mismatch");
    let mut buf = values.to_vec();
    let mut size = values.len();
    for &p in point {
        let half = size / 2;
        for i in 0..half {
            let a = buf[2 * i];
            let b = buf[2 * i + 1];
            buf[i] = a + p * (b - a);
        }
        size = half;
    }
    buf[0]
}

pub fn partial_eval(values: &[Goldilocks], fix: &[Goldilocks]) -> Vec<Goldilocks> {
    assert!(fix.len() <= values.len().trailing_zeros() as usize);
    let mut buf = values.to_vec();
    for &p in fix {
        let half = buf.len() / 2;
        for i in 0..half {
            let a = buf[2 * i];
            let b = buf[2 * i + 1];
            buf[i] = a + p * (b - a);
        }
        buf.truncate(half);
    }
    buf
}

/// Evaluate the equality (Lagrange-basis) polynomial `eq(x, r)` at every
/// hypercube point `x = i` for a fixed `r`: `eq_i(r) = prod_j (bit_j(i) * r_j +
/// (1 - bit_j(i)) * (1 - r_j))`. Used as the random selector in the
/// zero-check sumchecks.
pub fn eq_evals(r: &[Goldilocks]) -> Vec<Goldilocks> {
    let d = r.len();
    let n = 1 << d;
    // Doubling construction: after processing bit j, `eq[..2^j]` holds the
    // product over bits 0..j. Each step derives the upper half from the lower
    // half, so the total work is O(2^d) field multiplications instead of
    // O(d * 2^d). The values are identical to the direct product (exact field
    // arithmetic), so proofs and transcripts are unchanged.
    let mut eq = vec![Goldilocks::ONE; n];
    let mut size = 1usize;
    for &rj in r.iter().take(d) {
        let one_minus = Goldilocks::ONE - rj;
        for i in 0..size {
            let v = eq[i];
            eq[i] = v * one_minus;
            eq[i + size] = v * rj;
        }
        size <<= 1;
    }
    eq
}

/// Evaluate the equality polynomial `eq(r, x) = prod_j (x_j r_j + (1-x_j)(1-r_j))`
/// at a single point `x`. Used by the multi-point batch opening.
pub fn eq_poly(r: &[Goldilocks], x: &[Goldilocks]) -> Goldilocks {
    assert_eq!(r.len(), x.len());
    r.iter().zip(x).fold(Goldilocks::ONE, |acc, (&ri, &xi)| {
        acc * (xi * ri + (Goldilocks::ONE - xi) * (Goldilocks::ONE - ri))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::XorShift64;

    #[test]
    fn partial_then_full_matches_full() {
        let mut rng = XorShift64::new(2);
        let values: Vec<Goldilocks> = (0..16).map(|_| rng.field()).collect();
        let point: Vec<Goldilocks> = (0..4).map(|_| rng.field()).collect();

        let rest = partial_eval(&values, &point[..2]);
        assert_eq!(rest.len(), 4);
        let via_partial = eval(&rest, &point[2..]);
        let full = eval(&values, &point);
        assert_eq!(via_partial, full);
    }

    /// The O(2^d) doubling construction must match the direct product formula
    /// bit-for-bit (it is a pure reordering of exact field multiplications).
    #[test]
    fn eq_evals_matches_direct_product() {
        let mut rng = XorShift64::new(0x9e_e1);
        for d in 0..=8usize {
            let r: Vec<Goldilocks> = (0..d).map(|_| rng.field()).collect();
            let fast = eq_evals(&r);
            assert_eq!(fast.len(), 1 << d);
            for (i, &got) in fast.iter().enumerate() {
                let mut want = Goldilocks::ONE;
                for (j, &rj) in r.iter().enumerate() {
                    let one_minus = Goldilocks::ONE - rj;
                    want = want * if (i >> j) & 1 == 1 { rj } else { one_minus };
                }
                assert_eq!(got, want, "d={d} i={i}");
            }
            // And each entry agrees with the single-point eq polynomial at the
            // corresponding boolean vertex.
            for (i, &got) in fast.iter().enumerate() {
                let xb: Vec<Goldilocks> = (0..d)
                    .map(|j| if (i >> j) & 1 == 1 { Goldilocks::ONE } else { Goldilocks::ZERO })
                    .collect();
                assert_eq!(got, eq_poly(&r, &xb), "point eq mismatch d={d} i={i}");
            }
        }
    }
}
