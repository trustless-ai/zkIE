//! Canonical BN254 Poseidon commitments for shard boundary tensors.

use ark_ff::{BigInteger, PrimeField as ArkPrimeField};
use halo2_proofs::circuit::{AssignedCell, Layouter, Value};
use halo2_proofs::halo2curves::bn256::Fr;
use halo2_proofs::halo2curves::ff::PrimeField;
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, ErrorFront, Fixed, Selector};
use halo2_proofs::poly::Rotation;
use light_poseidon::{parameters::bn254_x5, Poseidon, PoseidonHasher};

use crate::field_convert::i64_to_fr;
use crate::fixed_point::I18;

pub const BOUNDARY_COMMITMENT_SCHEMA_VERSION: u32 = 1;
pub const MAX_BOUNDARY_ELEMENTS: usize = 1 << 13;
pub const MAX_BOUNDARY_IDENTITIES: usize = 64;
pub const MAX_BOUNDARY_ID_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryRole {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryDType {
    I18,
}

/// The compiler currently guarantees a flat element count only. This must not
/// be interpreted as a claimed higher-rank tensor shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryShape {
    FlatElementCountOnly { element_count: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryDescriptor {
    role: BoundaryRole,
    register_id: String,
    edge_ids: Vec<String>,
    graph_output_names: Vec<String>,
    dtype: BoundaryDType,
    shape: BoundaryShape,
    quantization_scale: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundaryCommitmentError {
    EmptyTensor,
    TooManyElements,
    TooManyIdentities,
    InvalidIdentity,
    NonCanonicalIdentities,
    LengthMismatch { expected: usize, actual: usize },
    PoseidonParameters,
}

impl std::fmt::Display for BoundaryCommitmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BoundaryCommitmentError {}

impl BoundaryDescriptor {
    pub fn flat_i18(
        role: BoundaryRole,
        register_id: impl Into<String>,
        edge_ids: Vec<String>,
        graph_output_names: Vec<String>,
        element_count: usize,
        quantization_scale: u64,
    ) -> Result<Self, BoundaryCommitmentError> {
        let descriptor = Self {
            role,
            register_id: register_id.into(),
            edge_ids,
            graph_output_names,
            dtype: BoundaryDType::I18,
            shape: BoundaryShape::FlatElementCountOnly { element_count },
            quantization_scale,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn role(&self) -> BoundaryRole {
        self.role
    }
    pub fn register_id(&self) -> &str {
        &self.register_id
    }
    pub fn edge_ids(&self) -> &[String] {
        &self.edge_ids
    }
    pub fn graph_output_names(&self) -> &[String] {
        &self.graph_output_names
    }
    pub fn dtype(&self) -> BoundaryDType {
        self.dtype
    }
    pub fn shape(&self) -> BoundaryShape {
        self.shape
    }
    pub fn element_count(&self) -> usize {
        match self.shape {
            BoundaryShape::FlatElementCountOnly { element_count } => element_count,
        }
    }
    pub fn quantization_scale(&self) -> u64 {
        self.quantization_scale
    }

    pub fn validate(&self) -> Result<(), BoundaryCommitmentError> {
        if self.element_count() == 0 {
            return Err(BoundaryCommitmentError::EmptyTensor);
        }
        if self.element_count() > MAX_BOUNDARY_ELEMENTS {
            return Err(BoundaryCommitmentError::TooManyElements);
        }
        if self
            .edge_ids
            .len()
            .saturating_add(self.graph_output_names.len())
            > MAX_BOUNDARY_IDENTITIES
        {
            return Err(BoundaryCommitmentError::TooManyIdentities);
        }
        if !valid_id(&self.register_id)
            || self
                .edge_ids
                .iter()
                .chain(&self.graph_output_names)
                .any(|id| !valid_id(id))
        {
            return Err(BoundaryCommitmentError::InvalidIdentity);
        }
        if !strictly_sorted(&self.edge_ids) || !strictly_sorted(&self.graph_output_names) {
            return Err(BoundaryCommitmentError::NonCanonicalIdentities);
        }
        Ok(())
    }

    /// Field sequence committed before tensor values. Every variable-length
    /// component is count- or length-bound and every semantic role is tagged.
    pub fn canonical_fields(&self) -> Vec<Fr> {
        let mut fields = vec![
            Fr::from(BOUNDARY_COMMITMENT_SCHEMA_VERSION as u64),
            Fr::from(match self.role {
                BoundaryRole::Input => 1,
                BoundaryRole::Output => 2,
            }),
            Fr::from(match self.dtype {
                BoundaryDType::I18 => 1,
            }),
            Fr::from(1), // explicit FlatElementCountOnly shape kind
            Fr::from(self.element_count() as u64),
            Fr::from(self.quantization_scale),
        ];
        fields.extend(identity_fields(&self.register_id));
        fields.push(Fr::from(self.edge_ids.len() as u64));
        for id in &self.edge_ids {
            fields.extend(identity_fields(id));
        }
        fields.push(Fr::from(self.graph_output_names.len() as u64));
        for id in &self.graph_output_names {
            fields.extend(identity_fields(id));
        }
        fields
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_BOUNDARY_ID_BYTES
}
fn strictly_sorted(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn identity_fields(value: &str) -> [Fr; 2] {
    let mut bytes = b"zkie.boundary-identity.v1\0".to_vec();
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    let digest = blake3::hash(&bytes);
    let raw = digest.as_bytes();
    [limb_to_fr(&raw[..16]), limb_to_fr(&raw[16..])]
}

fn limb_to_fr(bytes: &[u8]) -> Fr {
    let mut repr = [0_u8; 32];
    repr[..bytes.len()].copy_from_slice(bytes);
    Option::from(Fr::from_repr(repr.into())).expect("128-bit limbs fit BN254 Fr")
}

fn ark_to_halo(value: ark_bn254::Fr) -> Fr {
    let bytes = value.into_bigint().to_bytes_le();
    let mut repr = [0_u8; 32];
    repr[..bytes.len()].copy_from_slice(&bytes);
    Option::from(Fr::from_repr(repr.into())).expect("same BN254 scalar modulus")
}

fn halo_to_ark(value: Fr) -> ark_bn254::Fr {
    ark_bn254::Fr::from_le_bytes_mod_order(value.to_repr().as_ref())
}

/// Circom-compatible BN254 x5 Poseidon with two field inputs. The byte KAT in
/// tests uses big-endian input/output bytes; field representations here are canonical.
pub fn poseidon_compress_native(left: Fr, right: Fr) -> Fr {
    let mut poseidon = Poseidon::<ark_bn254::Fr>::new_circom(2)
        .expect("pinned light-poseidon provides width-3 BN254 parameters");
    ark_to_halo(
        poseidon
            .hash(&[halo_to_ark(left), halo_to_ark(right)])
            .expect("exactly two canonical field inputs"),
    )
}

pub fn commit_boundary_native(
    descriptor: &BoundaryDescriptor,
    values: &[I18],
) -> Result<Fr, BoundaryCommitmentError> {
    descriptor.validate()?;
    if values.len() != descriptor.element_count() {
        return Err(BoundaryCommitmentError::LengthMismatch {
            expected: descriptor.element_count(),
            actual: values.len(),
        });
    }
    let protocol_domain = identity_fields("zkie.poseidon-boundary-fold.v1")[0];
    let mut state = protocol_domain;
    for field in descriptor.canonical_fields() {
        state = poseidon_compress_native(state, field);
    }
    state = poseidon_compress_native(state, Fr::from(values.len() as u64));
    for value in values {
        state = poseidon_compress_native(state, i64_to_fr(value.raw()));
    }
    Ok(state)
}

#[derive(Clone, Debug)]
pub struct PoseidonBoundaryConfig {
    state: [Column<Advice>; 3],
    round_constants: [Column<Fixed>; 3],
    _constant: Column<Fixed>,
    full_round: Selector,
    partial_round: Selector,
}

#[derive(Clone, Debug)]
pub struct PoseidonBoundaryChip {
    config: PoseidonBoundaryConfig,
}

impl PoseidonBoundaryChip {
    pub fn configure(meta: &mut ConstraintSystem<Fr>) -> PoseidonBoundaryConfig {
        let state = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        for column in state {
            meta.enable_equality(column);
        }
        let round_constants = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        let constant = meta.fixed_column();
        meta.enable_constant(constant);
        let full_round = meta.selector();
        let partial_round = meta.selector();
        let params = poseidon_parameters();
        let mds = params.mds;
        meta.create_gate("poseidon boundary full round", |meta| {
            let q = meta.query_selector(full_round);
            let current: Vec<_> = state
                .iter()
                .zip(&round_constants)
                .map(|(state, rc)| {
                    meta.query_advice(*state, Rotation::cur())
                        + meta.query_fixed(*rc, Rotation::cur())
                })
                .collect();
            let sboxed: Vec<_> = current.into_iter().map(pow5).collect();
            state
                .iter()
                .enumerate()
                .map(|(row, next)| {
                    let expected = sboxed.iter().enumerate().fold(
                        halo2_proofs::plonk::Expression::Constant(Fr::zero()),
                        |sum, (column, value)| sum + value.clone() * ark_to_halo(mds[row][column]),
                    );
                    q.clone() * (meta.query_advice(*next, Rotation::next()) - expected)
                })
                .collect::<Vec<_>>()
        });
        let params = poseidon_parameters();
        let mds = params.mds;
        meta.create_gate("poseidon boundary partial round", |meta| {
            let q = meta.query_selector(partial_round);
            let current: Vec<_> = state
                .iter()
                .zip(&round_constants)
                .map(|(state, rc)| {
                    meta.query_advice(*state, Rotation::cur())
                        + meta.query_fixed(*rc, Rotation::cur())
                })
                .collect();
            let sboxed = [
                pow5(current[0].clone()),
                current[1].clone(),
                current[2].clone(),
            ];
            state
                .iter()
                .enumerate()
                .map(|(row, next)| {
                    let expected = sboxed.iter().enumerate().fold(
                        halo2_proofs::plonk::Expression::Constant(Fr::zero()),
                        |sum, (column, value)| sum + value.clone() * ark_to_halo(mds[row][column]),
                    );
                    q.clone() * (meta.query_advice(*next, Rotation::next()) - expected)
                })
                .collect::<Vec<_>>()
        });
        PoseidonBoundaryConfig {
            state,
            round_constants,
            _constant: constant,
            full_round,
            partial_round,
        }
    }

    pub fn construct(config: PoseidonBoundaryConfig) -> Self {
        Self { config }
    }

    pub fn hash_pair(
        &self,
        mut layouter: impl Layouter<Fr>,
        left: &AssignedCell<Fr, Fr>,
        right: &AssignedCell<Fr, Fr>,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        let params = poseidon_parameters();
        let half_full = params.full_rounds / 2;
        let partial_end = half_full + params.partial_rounds;
        layouter.assign_region(
            || "constrained circom poseidon two-input compression",
            |mut region| {
                let zero = region.assign_advice(
                    || "domain tag",
                    self.config.state[0],
                    0,
                    || Value::known(Fr::zero()),
                )?;
                region.constrain_constant(zero.cell(), Fr::zero())?;
                let left_copy = region.assign_advice(
                    || "left input",
                    self.config.state[1],
                    0,
                    || left.value().copied(),
                )?;
                region.constrain_equal(left.cell(), left_copy.cell())?;
                let right_copy = region.assign_advice(
                    || "right input",
                    self.config.state[2],
                    0,
                    || right.value().copied(),
                )?;
                region.constrain_equal(right.cell(), right_copy.cell())?;
                let mut cells = [zero, left_copy, right_copy];
                for round in 0..(params.full_rounds + params.partial_rounds) {
                    if round < half_full || round >= partial_end {
                        self.config.full_round.enable(&mut region, round)?;
                    } else {
                        self.config.partial_round.enable(&mut region, round)?;
                    }
                    let mut after_ark = Vec::with_capacity(3);
                    for (column, cell) in cells.iter().enumerate() {
                        let rc = ark_to_halo(params.ark[round * 3 + column]);
                        region.assign_fixed(
                            || "round constant",
                            self.config.round_constants[column],
                            round,
                            || Value::known(rc),
                        )?;
                        after_ark.push(cell.value().copied() + Value::known(rc));
                    }
                    let sboxed = if round < half_full || round >= partial_end {
                        [
                            value_pow5(after_ark[0]),
                            value_pow5(after_ark[1]),
                            value_pow5(after_ark[2]),
                        ]
                    } else {
                        [value_pow5(after_ark[0]), after_ark[1], after_ark[2]]
                    };
                    let next_values: [Value<Fr>; 3] = std::array::from_fn(|row| {
                        (0..3).fold(Value::known(Fr::zero()), |sum, column| {
                            sum + sboxed[column]
                                * Value::known(ark_to_halo(params.mds[row][column]))
                        })
                    });
                    let next0 = region.assign_advice(
                        || "next state 0",
                        self.config.state[0],
                        round + 1,
                        || next_values[0],
                    )?;
                    let next1 = region.assign_advice(
                        || "next state 1",
                        self.config.state[1],
                        round + 1,
                        || next_values[1],
                    )?;
                    let next2 = region.assign_advice(
                        || "next state 2",
                        self.config.state[2],
                        round + 1,
                        || next_values[2],
                    )?;
                    cells = [next0, next1, next2];
                }
                Ok(cells[0].clone())
            },
        )
    }

    pub fn assign_constant(
        &self,
        mut layouter: impl Layouter<Fr>,
        value: Fr,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        layouter.assign_region(
            || "poseidon constant",
            |mut region| {
                let cell = region.assign_advice(
                    || "constant",
                    self.config.state[0],
                    0,
                    || Value::known(value),
                )?;
                region.constrain_constant(cell.cell(), value)?;
                Ok(cell)
            },
        )
    }

    pub fn assign_public_value(
        &self,
        mut layouter: impl Layouter<Fr>,
        value: Fr,
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        layouter.assign_region(
            || "leaf public advice",
            |mut region| {
                region.assign_advice(
                    || "public value",
                    self.config.state[0],
                    0,
                    || Value::known(value),
                )
            },
        )
    }

    pub fn constrain_equal_to_constant(
        &self,
        mut layouter: impl Layouter<Fr>,
        cell: &AssignedCell<Fr, Fr>,
        value: Fr,
    ) -> Result<(), ErrorFront> {
        let constant = self.assign_constant(layouter.namespace(|| "expected constant"), value)?;
        layouter.assign_region(
            || "bind assigned cell to configured constant",
            |mut region| region.constrain_equal(cell.cell(), constant.cell()),
        )
    }

    /// Commits the exact canonical assembler cells. Every value is copied with
    /// an equality constraint before entering a fully constrained permutation.
    pub fn commit_assigned(
        &self,
        mut layouter: impl Layouter<Fr>,
        descriptor: &BoundaryDescriptor,
        values: &[AssignedCell<Fr, Fr>],
    ) -> Result<AssignedCell<Fr, Fr>, ErrorFront> {
        if descriptor.validate().is_err() || values.len() != descriptor.element_count() {
            return Err(ErrorFront::Synthesis);
        }
        let domain = identity_fields("zkie.poseidon-boundary-fold.v1")[0];
        let mut state =
            self.assign_constant(layouter.namespace(|| "boundary protocol domain"), domain)?;
        for (index, field) in descriptor.canonical_fields().into_iter().enumerate() {
            let field =
                self.assign_constant(layouter.namespace(|| format!("metadata {index}")), field)?;
            state = self.hash_pair(
                layouter.namespace(|| format!("metadata fold {index}")),
                &state,
                &field,
            )?;
        }
        let count = self.assign_constant(
            layouter.namespace(|| "boundary value count"),
            Fr::from(values.len() as u64),
        )?;
        state = self.hash_pair(layouter.namespace(|| "value count fold"), &state, &count)?;
        for (index, value) in values.iter().enumerate() {
            state = self.hash_pair(
                layouter.namespace(|| format!("value fold {index}")),
                &state,
                value,
            )?;
        }
        Ok(state)
    }
}

fn poseidon_parameters() -> light_poseidon::PoseidonParameters<ark_bn254::Fr> {
    bn254_x5::get_poseidon_parameters(3).expect("pinned width-3 parameters")
}

fn pow5(value: halo2_proofs::plonk::Expression<Fr>) -> halo2_proofs::plonk::Expression<Fr> {
    let square = value.clone() * value.clone();
    square.clone() * square * value
}

fn value_pow5(value: Value<Fr>) -> Value<Fr> {
    value.map(|value| value.square().square() * value)
}

#[cfg(test)]
mod circuit_tests {
    use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Advice, Circuit, Column, ConstraintSystem, ErrorFront, Instance};

    use super::*;

    #[derive(Clone)]
    struct PairConfig {
        poseidon: PoseidonBoundaryConfig,
        input: Column<Advice>,
        instance: Column<Instance>,
    }

    #[derive(Clone)]
    struct PairCircuit {
        left: Fr,
        right: Fr,
    }

    impl Circuit<Fr> for PairCircuit {
        type Config = PairConfig;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();
        fn without_witnesses(&self) -> Self {
            self.clone()
        }
        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let input = meta.advice_column();
            let instance = meta.instance_column();
            meta.enable_equality(input);
            meta.enable_equality(instance);
            PairConfig {
                poseidon: PoseidonBoundaryChip::configure(meta),
                input,
                instance,
            }
        }
        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let inputs = layouter.assign_region(
                || "canonical source cells",
                |mut region| {
                    let left = region.assign_advice(
                        || "left",
                        config.input,
                        0,
                        || Value::known(self.left),
                    )?;
                    let right = region.assign_advice(
                        || "right",
                        config.input,
                        1,
                        || Value::known(self.right),
                    )?;
                    Ok((left, right))
                },
            )?;
            let output = PoseidonBoundaryChip::construct(config.poseidon).hash_pair(
                layouter.namespace(|| "poseidon"),
                &inputs.0,
                &inputs.1,
            )?;
            layouter.constrain_instance(output.cell(), config.instance, 0)
        }
    }

    #[test]
    fn circuit_matches_native_and_rejects_changed_public_hash() {
        let circuit = PairCircuit {
            left: Fr::from(7),
            right: Fr::from(11),
        };
        let expected = poseidon_compress_native(circuit.left, circuit.right);
        MockProver::run(8, &circuit, vec![vec![expected]])
            .unwrap()
            .assert_satisfied();
        assert!(
            MockProver::run(8, &circuit, vec![vec![expected + Fr::one()]])
                .unwrap()
                .verify()
                .is_err()
        );
    }
}
