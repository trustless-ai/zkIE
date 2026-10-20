use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::PrimeCharacteristicRing;

pub(crate) const PUBLIC_VALUES: usize = 48;
pub(crate) const TRACE_LOG_ROWS: usize = 3;
pub(crate) const WIDTH: usize = 70;
pub(crate) const Q_START: usize = 16;
pub(crate) const REM_START: usize = 52;
pub(crate) const H_START: usize = 68;

pub(crate) struct InferenceAir;

impl<F> BaseAir<F> for InferenceAir {
    fn width(&self) -> usize {
        WIDTH
    }
    fn num_public_values(&self) -> usize {
        PUBLIC_VALUES
    }
    fn max_constraint_degree(&self) -> Option<usize> {
        Some(2)
    }
}

/// Little-endian, range-constrained bit decomposition. Signed values use a
/// positive offset, avoiding ambiguous field-to-signed-integer conversion.
fn bits<AB: AirBuilder>(row: &[AB::Var], start: usize, len: usize) -> AB::Expr {
    (0..len).fold(AB::Expr::ZERO, |sum, i| {
        sum + row[start + i] * AB::F::from_u64(1 << i)
    })
}

impl<AB: AirBuilder> Air<AB> for InferenceAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let public = builder.public_values().to_vec();
        for &b in &row[..H_START] {
            builder.assert_bool(b);
        }
        let x: [AB::Expr; 2] =
            std::array::from_fn(|i| bits::<AB>(row, i * 8, 8) - AB::F::from_u64(128));
        let q: [AB::Expr; 4] =
            std::array::from_fn(|i| bits::<AB>(row, Q_START + i * 9, 9) - AB::F::from_u64(256));
        for i in 0..2 {
            builder.assert_eq(x[i].clone(), public[44 + i]);
            builder.assert_eq(q[2 + i].clone(), public[46 + i]);
            // The top bit of q+256 is one iff q >= 0. This constrains ReLU,
            // including its signed range and the connection to the next layer.
            builder.assert_eq(row[H_START + i], q[i].clone() * row[Q_START + i * 9 + 8]);
        }
        for layer in 0..2 {
            for j in 0..2 {
                let inputs: [AB::Expr; 2] = if layer == 0 {
                    x.clone()
                } else {
                    std::array::from_fn(|i| row[H_START + i].into())
                };
                let w0: AB::Expr = public[32 + layer * 4 + j * 2].into();
                let w1: AB::Expr = public[32 + layer * 4 + j * 2 + 1].into();
                let dot = inputs[0].clone() * w0 + inputs[1].clone() * w1;
                let rem = bits::<AB>(row, REM_START + (layer * 2 + j) * 4, 4);
                let bias: AB::Expr = public[40 + layer * 2 + j].into();
                // dot + half = (q - bias)*16 + rem, 0 <= rem < 16.
                builder.assert_eq(
                    dot + AB::F::from_u64(8),
                    (q[layer * 2 + j].clone() - bias) * AB::F::from_u64(16) + rem,
                );
            }
        }
    }
}
