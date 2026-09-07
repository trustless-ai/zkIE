//! `LookupRangeCheckChip`: bounds a value to `n_bits` by splitting it into
//! `LIMB_BITS`-wide limbs and looking each one up in a fixed byte table.
//!
//! Same guarantee as [`crate::chips::range_check::RangeCheckChip`], at
//! `n_bits / LIMB_BITS` rows per value instead of `n_bits`. That difference
//! is what makes it usable for per-operand bounds: a length-264 dot product
//! needs two bounds per element, which costs 4,224 rows here against 33,792
//! by bit decomposition.
//!
//! The bit-decomposition chip is still the right tool for one-off witnesses
//! such as `q`/`r`/`slack`, where the count is fixed per instance and
//! `n_bits` need not be a multiple of `LIMB_BITS`.

use crate::field_convert::Fr;
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::plonk::{
    Advice, Column, ConstraintSystem, ErrorFront, Expression, Selector, TableColumn,
};
use halo2_proofs::poly::Rotation;

/// Width of one looked-up limb. 8 keeps the table at 256 rows, so the table
/// never forces a circuit larger than the rest of the design already needs.
pub const LIMB_BITS: usize = 8;

const LIMB_VALUES: u64 = 1 << LIMB_BITS;

#[derive(Clone, Debug)]
pub struct LookupRangeCheckConfig {
    pub value: Column<Advice>,
    pub limbs: Column<Advice>,
    pub s_limb: Selector,
    pub s_sum: Selector,
    pub table: TableColumn,
    pub n_bits: usize,
}

pub struct LookupRangeCheckChip {
    config: LookupRangeCheckConfig,
}

impl LookupRangeCheckChip {
    /// `n_bits` must be a multiple of [`LIMB_BITS`].
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        value: Column<Advice>,
        limbs: Column<Advice>,
        n_bits: usize,
    ) -> LookupRangeCheckConfig {
        assert!(
            n_bits > 0 && n_bits.is_multiple_of(LIMB_BITS),
            "n_bits must be a positive multiple of {LIMB_BITS}"
        );
        meta.enable_equality(limbs);

        // Complex, because it appears inside a lookup expression.
        let s_limb = meta.complex_selector();
        let table = meta.lookup_table_column();
        meta.lookup("limb is in the byte table", |meta| {
            let s = meta.query_selector(s_limb);
            let limb = meta.query_advice(limbs, Rotation::cur());
            // On a disabled row the expression collapses to 0, which the
            // table always contains, so unrelated rows are unconstrained.
            vec![(s * limb, table)]
        });

        let n_limbs = n_bits / LIMB_BITS;
        let s_sum = meta.selector();
        meta.create_gate("limbs recompose to value", |meta| {
            let value = meta.query_advice(value, Rotation::cur());
            let s_sum = meta.query_selector(s_sum);
            let mut sum = Expression::Constant(Fr::zero());
            let mut coeff = Fr::one();
            let base = Fr::from(LIMB_VALUES);
            for i in 0..n_limbs {
                // `assign` puts limb `i` at row `i` with `s_sum` enabled at
                // row `n_limbs - 1`, mirroring `RangeCheckChip`'s layout.
                let rotation = Rotation(-((n_limbs - 1 - i) as i32));
                let limb = meta.query_advice(limbs, rotation);
                sum = sum + limb * Expression::Constant(coeff);
                coeff *= base;
            }
            vec![s_sum * (sum - value)]
        });

        LookupRangeCheckConfig {
            value,
            limbs,
            s_limb,
            s_sum,
            table,
            n_bits,
        }
    }

    pub fn construct(config: LookupRangeCheckConfig) -> Self {
        LookupRangeCheckChip { config }
    }

    /// Loads the byte table. Must be called once per circuit synthesis, and
    /// once per configured chip, independently of any [`Self::assign`] calls.
    pub fn load_table(&self, mut layouter: impl Layouter<Fr>) -> Result<(), ErrorFront> {
        layouter.assign_table(
            || "byte range table",
            |mut table| {
                for i in 0..LIMB_VALUES {
                    table.assign_cell(
                        || format!("byte {i}"),
                        self.config.table,
                        i as usize,
                        || Value::known(Fr::from(i)),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Witnesses the limbs of `raw_value` and returns the cell holding
    /// `value`, which the caller must copy-constrain to the cell it means to
    /// bound -- on its own this only constrains a cell in this chip's region.
    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fr>,
        value: Value<Fr>,
        raw_value: Value<i128>,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        let n_limbs = self.config.n_bits / LIMB_BITS;
        layouter.assign_region(
            || "lookup range check",
            |mut region| {
                for i in 0..n_limbs {
                    self.config.s_limb.enable(&mut region, i)?;
                    let limb =
                        raw_value.map(|v| ((v >> (LIMB_BITS * i)) as u64) & (LIMB_VALUES - 1));
                    region.assign_advice(
                        || format!("limb {i}"),
                        self.config.limbs,
                        i,
                        || limb.map(Fr::from),
                    )?;
                }
                self.config.s_sum.enable(&mut region, n_limbs - 1)?;
                region.assign_advice(|| "value", self.config.value, n_limbs - 1, || value)
            },
        )
    }
}
