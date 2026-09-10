use std::collections::HashMap;

use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};

use crate::assembler::{
    AssemblerChip, AssemblerConfig, AssemblerInstruction, AssemblerProgram, RegisterRef,
};
use crate::chips::layer_norm::RsqrtDomain;
use crate::chips::poseidon_boundary::{
    BoundaryDescriptor, BoundaryRole, PoseidonBoundaryChip, PoseidonBoundaryConfig,
    BOUNDARY_COMMITMENT_SCHEMA_VERSION,
};
use crate::field_convert::Fr;
use crate::isa::{EltwiseOp, Instruction, ReduceOp};

/// All runtime data that influences an assembler circuit's configured shape.
/// Witness values themselves are intentionally excluded.
#[derive(Clone, Default)]
pub struct AssemblerCircuitParams {
    pub instructions: Vec<AssemblerInstruction>,
    pub rms_norm_domains: HashMap<(usize, u64), RsqrtDomain>,
    input_shapes: Vec<usize>,
    weight_shapes: Vec<usize>,
    public_output_indices: Vec<usize>,
    boundary_inputs: Vec<BoundaryDescriptor>,
    boundary_outputs: Vec<(usize, BoundaryDescriptor)>,
    leaf_prefix_len: usize,
    fixed_public_weights: Vec<Vec<i64>>,
}

impl AssemblerCircuitParams {
    fn from_program(
        program: &AssemblerProgram,
        rms_norm_domains: HashMap<(usize, u64), RsqrtDomain>,
        public_output_indices: Vec<usize>,
    ) -> Self {
        Self {
            instructions: program.instructions.clone(),
            rms_norm_domains,
            input_shapes: program.input_values.iter().map(Vec::len).collect(),
            weight_shapes: program.weight_values.iter().map(Vec::len).collect(),
            public_output_indices,
            boundary_inputs: Vec::new(),
            boundary_outputs: Vec::new(),
            leaf_prefix_len: 0,
            fixed_public_weights: Vec::new(),
        }
    }

    /// A versioned, canonical BLAKE3 digest over every configuration and
    /// layout input. It is a shape identifier, not a witness commitment.
    pub fn shape_digest(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_shape_bytes()).as_bytes()
    }

    fn canonical_shape_bytes(&self) -> Vec<u8> {
        let mut bytes = b"zkie.assembler-circuit-shape.v3\0".to_vec();
        if self.boundary_inputs.is_empty() && self.boundary_outputs.is_empty() {
            bytes.extend_from_slice(b"public-instances-v1:inputs,weights,selected-outputs\0");
        } else {
            bytes.extend_from_slice(
                b"public-instances-v2:leaf,input-commitments,output-commitments,weights\0",
            );
        }
        encode_usizes(&mut bytes, &self.input_shapes);
        encode_usizes(&mut bytes, &self.weight_shapes);
        encode_usizes(&mut bytes, &self.public_output_indices);
        encode_usize(&mut bytes, self.leaf_prefix_len);
        encode_usize(&mut bytes, self.boundary_inputs.len());
        for descriptor in &self.boundary_inputs {
            encode_boundary_descriptor(&mut bytes, descriptor);
        }
        encode_usize(&mut bytes, self.boundary_outputs.len());
        for (index, descriptor) in &self.boundary_outputs {
            encode_usize(&mut bytes, *index);
            encode_boundary_descriptor(&mut bytes, descriptor);
        }
        encode_usize(&mut bytes, self.fixed_public_weights.len());
        for tensor in &self.fixed_public_weights {
            encode_usize(&mut bytes, tensor.len());
            for value in tensor {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        encode_usize(&mut bytes, self.instructions.len());
        for instruction in &self.instructions {
            encode_instruction(&mut bytes, instruction);
        }
        let mut domains: Vec<_> = self.rms_norm_domains.iter().collect();
        domains.sort_unstable_by_key(|(key, _)| **key);
        encode_usize(&mut bytes, domains.len());
        for ((dim, epsilon), domain) in domains {
            encode_usize(&mut bytes, *dim);
            bytes.extend_from_slice(&epsilon.to_le_bytes());
            match domain {
                RsqrtDomain::Range { min, max, n } => {
                    bytes.push(0);
                    bytes.extend_from_slice(&min.to_bits().to_le_bytes());
                    bytes.extend_from_slice(&max.to_bits().to_le_bytes());
                    encode_usize(&mut bytes, *n);
                }
                RsqrtDomain::RawAnchors(anchors) => {
                    bytes.push(1);
                    encode_usize(&mut bytes, anchors.len());
                    for anchor in anchors {
                        bytes.extend_from_slice(&anchor.to_le_bytes());
                    }
                }
            }
        }
        bytes
    }
}

fn encode_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}
fn encode_usizes(bytes: &mut Vec<u8>, values: &[usize]) {
    encode_usize(bytes, values.len());
    for value in values {
        encode_usize(bytes, *value);
    }
}
fn encode_string(bytes: &mut Vec<u8>, value: &str) {
    encode_usize(bytes, value.len());
    bytes.extend_from_slice(value.as_bytes());
}
fn encode_boundary_descriptor(bytes: &mut Vec<u8>, descriptor: &BoundaryDescriptor) {
    bytes.extend_from_slice(&BOUNDARY_COMMITMENT_SCHEMA_VERSION.to_le_bytes());
    bytes.push(match descriptor.role() {
        BoundaryRole::Input => 1,
        BoundaryRole::Output => 2,
    });
    encode_string(bytes, descriptor.register_id());
    encode_usize(bytes, descriptor.edge_ids().len());
    for value in descriptor.edge_ids() {
        encode_string(bytes, value);
    }
    encode_usize(bytes, descriptor.graph_output_names().len());
    for value in descriptor.graph_output_names() {
        encode_string(bytes, value);
    }
    encode_usize(bytes, descriptor.element_count());
    bytes.extend_from_slice(&descriptor.quantization_scale().to_le_bytes());
}
fn encode_instruction(bytes: &mut Vec<u8>, instruction: &AssemblerInstruction) {
    encode_usize(bytes, instruction.inputs.len());
    for input in &instruction.inputs {
        match input {
            RegisterRef::Input(index) => {
                bytes.push(0);
                encode_usize(bytes, *index);
            }
            RegisterRef::Weight(index) => {
                bytes.push(1);
                encode_usize(bytes, *index);
            }
            RegisterRef::Virtual(index) => {
                bytes.push(2);
                encode_usize(bytes, *index);
            }
        }
    }
    match &instruction.instruction {
        Instruction::DotGeneral {
            m,
            n,
            k,
            batch_dims,
            trans_a,
            trans_b,
        } => {
            bytes.push(0);
            encode_usizes(bytes, &[*m, *n, *k]);
            encode_usizes(bytes, batch_dims);
            bytes.push(u8::from(*trans_a));
            bytes.push(u8::from(*trans_b));
        }
        Instruction::Softmax { axis_dim } => {
            bytes.push(1);
            encode_usize(bytes, *axis_dim);
        }
        Instruction::Gelu => bytes.push(2),
        Instruction::LayerNorm { dim, epsilon_milli } => {
            bytes.push(3);
            encode_usize(bytes, *dim);
            bytes.extend_from_slice(&epsilon_milli.to_le_bytes());
        }
        Instruction::RmsNorm { dim, epsilon_milli } => {
            bytes.push(4);
            encode_usize(bytes, *dim);
            bytes.extend_from_slice(&epsilon_milli.to_le_bytes());
        }
        Instruction::Eltwise { op } => {
            let op = match op {
                EltwiseOp::Add => 0_u8,
                EltwiseOp::Mul => 1,
                EltwiseOp::Relu => 2,
            };
            bytes.extend_from_slice(&[5, op]);
        }
        Instruction::Reduce { op, axis } => {
            let op = match op {
                ReduceOp::Sum => 0_u8,
                ReduceOp::Mean => 1,
            };
            bytes.extend_from_slice(&[6, op]);
            encode_usize(bytes, *axis);
        }
        Instruction::EmbedLookup {
            table_size,
            embed_dim,
        } => {
            bytes.push(7);
            encode_usizes(bytes, &[*table_size, *embed_dim]);
        }
        Instruction::PatchEmbed {
            patch_len,
            embed_dim,
        } => {
            bytes.push(8);
            encode_usizes(bytes, &[*patch_len, *embed_dim]);
        }
    }
}

/// One concrete circuit type whose Halo2 configuration is selected from its
/// runtime [`AssemblerCircuitParams`].
#[derive(Clone)]
pub struct AssemblerCircuit {
    params: AssemblerCircuitParams,
    program: AssemblerProgram,
    has_known_witnesses: bool,
    leaf_prefix: Vec<Fr>,
}

impl AssemblerCircuit {
    pub fn new(
        program: AssemblerProgram,
        rms_norm_domains: HashMap<(usize, u64), RsqrtDomain>,
    ) -> Self {
        let public_output_indices = program
            .instructions
            .len()
            .checked_sub(1)
            .into_iter()
            .collect();
        Self::new_with_public_outputs(program, rms_norm_domains, public_output_indices)
            .expect("default final output layout is valid")
    }

    pub fn new_with_public_outputs(
        program: AssemblerProgram,
        rms_norm_domains: HashMap<(usize, u64), RsqrtDomain>,
        public_output_indices: Vec<usize>,
    ) -> Result<Self, String> {
        if public_output_indices
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || public_output_indices
                .iter()
                .any(|index| *index >= program.instructions.len())
        {
            return Err(
                "public output indices must be strictly sorted, unique, and in range".into(),
            );
        }
        let params =
            AssemblerCircuitParams::from_program(&program, rms_norm_domains, public_output_indices);
        Ok(Self {
            params,
            program,
            has_known_witnesses: true,
            leaf_prefix: Vec::new(),
        })
    }

    pub fn new_with_boundary_commitments(
        program: AssemblerProgram,
        rms_norm_domains: HashMap<(usize, u64), RsqrtDomain>,
        boundary_inputs: Vec<BoundaryDescriptor>,
        boundary_outputs: Vec<(usize, BoundaryDescriptor)>,
        leaf_prefix: Vec<Fr>,
    ) -> Result<Self, String> {
        let poseidon_work = boundary_inputs
            .iter()
            .chain(boundary_outputs.iter().map(|(_, descriptor)| descriptor))
            .try_fold(0usize, |sum, descriptor| {
                sum.checked_add(descriptor.canonical_fields().len())?
                    .checked_add(descriptor.element_count())?
                    .checked_add(2)
            });
        if boundary_inputs.len() != program.input_values.len()
            || boundary_inputs
                .iter()
                .zip(&program.input_values)
                .any(|(descriptor, values)| {
                    descriptor.role() != BoundaryRole::Input
                        || descriptor.element_count() != values.len()
                        || descriptor.validate().is_err()
                })
            || boundary_outputs
                .windows(2)
                .any(|pair| pair[0].0 >= pair[1].0)
            || boundary_outputs.iter().any(|(index, descriptor)| {
                *index >= program.instructions.len()
                    || descriptor.role() != BoundaryRole::Output
                    || descriptor.validate().is_err()
            })
            || leaf_prefix.len() != 17
            || poseidon_work.is_none_or(|work| work > 12_000)
        {
            return Err("invalid boundary commitment public layout".into());
        }
        let mut params =
            AssemblerCircuitParams::from_program(&program, rms_norm_domains, Vec::new());
        params.boundary_inputs = boundary_inputs;
        params.boundary_outputs = boundary_outputs;
        params.leaf_prefix_len = leaf_prefix.len();
        params.fixed_public_weights = program
            .weight_values
            .iter()
            .map(|tensor| tensor.iter().map(|value| value.raw()).collect())
            .collect();
        Ok(Self {
            params,
            program,
            has_known_witnesses: true,
            leaf_prefix,
        })
    }

    pub fn empty() -> Self {
        Self::new(
            AssemblerProgram {
                instructions: vec![],
                input_values: vec![],
                weight_values: vec![],
            },
            HashMap::new(),
        )
    }

    pub fn params(&self) -> &AssemblerCircuitParams {
        &self.params
    }

    pub fn has_known_witnesses(&self) -> bool {
        self.has_known_witnesses
    }
}

#[derive(Clone)]
pub struct AssemblerCircuitConfig {
    assembler: AssemblerConfig,
    poseidon: PoseidonBoundaryConfig,
}

impl Circuit<Fr> for AssemblerCircuit {
    type Config = AssemblerCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = AssemblerCircuitParams;

    fn without_witnesses(&self) -> Self {
        Self {
            params: self.params.clone(),
            program: self.program.clone(),
            has_known_witnesses: false,
            leaf_prefix: self.leaf_prefix.clone(),
        }
    }

    fn params(&self) -> Self::Params {
        self.params.clone()
    }

    fn configure_with_params(
        meta: &mut ConstraintSystem<Fr>,
        params: Self::Params,
    ) -> Self::Config {
        let assembler = AssemblerChip::configure_with_rms_norm_domains_and_public_instances(
            meta,
            &params.instructions,
            &params.rms_norm_domains,
        );
        let poseidon = PoseidonBoundaryChip::configure(meta);
        AssemblerCircuitConfig {
            assembler,
            poseidon,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        Self::configure_with_params(meta, Self::Params::default())
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let public_instance = config
            .assembler
            .public_instance
            .expect("AssemblerCircuit always configures public instances");
        let poseidon = PoseidonBoundaryChip::construct(config.poseidon);
        let chip = AssemblerChip::construct(config.assembler);
        chip.load_rms_norm_tables(layouter.namespace(|| "assembler rms norm tables"))?;
        let assigned = chip
            .assign_with_cells_with_witnesses(
                layouter.namespace(|| "assembler program"),
                &self.program,
                self.has_known_witnesses,
            )
            .map_err(|_| ErrorFront::Synthesis)?;
        let mut offset = 0;
        if !self.params.boundary_inputs.is_empty() || !self.params.boundary_outputs.is_empty() {
            for (index, value) in self.leaf_prefix.iter().enumerate() {
                let cell = poseidon.assign_public_value(
                    layouter.namespace(|| format!("leaf identity {index}")),
                    *value,
                )?;
                layouter.constrain_instance(cell.cell(), public_instance, offset)?;
                offset += 1;
            }
            for (index, descriptor) in self.params.boundary_inputs.iter().enumerate() {
                for (limb_index, field) in
                    descriptor.public_binding_fields().into_iter().enumerate()
                {
                    let cell = poseidon.assign_constant(
                        layouter.namespace(|| {
                            format!("input boundary {index} descriptor limb {limb_index}")
                        }),
                        field,
                    )?;
                    layouter.constrain_instance(cell.cell(), public_instance, offset)?;
                    offset += 1;
                }
                let tensor = assigned.inputs.get(index).ok_or(ErrorFront::Synthesis)?;
                let commitment = poseidon.commit_assigned(
                    layouter.namespace(|| format!("input boundary {index}")),
                    descriptor,
                    &tensor.cells,
                )?;
                layouter.constrain_instance(commitment.cell(), public_instance, offset)?;
                offset += 1;
            }
            let output_count = poseidon.assign_constant(
                layouter.namespace(|| "output boundary count"),
                Fr::from(self.params.boundary_outputs.len() as u64),
            )?;
            layouter.constrain_instance(output_count.cell(), public_instance, offset)?;
            offset += 1;
            for (index, descriptor) in &self.params.boundary_outputs {
                for (limb_index, field) in
                    descriptor.public_binding_fields().into_iter().enumerate()
                {
                    let cell = poseidon.assign_constant(
                        layouter.namespace(|| {
                            format!("output boundary {index} descriptor limb {limb_index}")
                        }),
                        field,
                    )?;
                    layouter.constrain_instance(cell.cell(), public_instance, offset)?;
                    offset += 1;
                }
                let tensor = assigned.virtuals.get(*index).ok_or(ErrorFront::Synthesis)?;
                if tensor.cells.len() != descriptor.element_count() {
                    return Err(ErrorFront::Synthesis);
                }
                let commitment = poseidon.commit_assigned(
                    layouter.namespace(|| format!("output boundary {index}")),
                    descriptor,
                    &tensor.cells,
                )?;
                layouter.constrain_instance(commitment.cell(), public_instance, offset)?;
                offset += 1;
            }
            let weight_count = self
                .program
                .weight_values
                .iter()
                .map(Vec::len)
                .sum::<usize>();
            let count_cell = poseidon.assign_constant(
                layouter.namespace(|| "public weight count"),
                Fr::from(weight_count as u64),
            )?;
            layouter.constrain_instance(count_cell.cell(), public_instance, offset)?;
            offset += 1;
            for (tensor, expected) in assigned
                .weights
                .iter()
                .zip(&self.params.fixed_public_weights)
            {
                for (cell, expected) in tensor.cells.iter().zip(expected) {
                    poseidon.constrain_equal_to_constant(
                        layouter.namespace(|| "fixed public-model weight"),
                        cell,
                        crate::field_convert::i64_to_fr(*expected),
                    )?;
                    layouter.constrain_instance(cell.cell(), public_instance, offset)?;
                    offset += 1;
                }
            }
            return Ok(());
        }
        for tensor in assigned.inputs.iter().chain(&assigned.weights) {
            for cell in &tensor.cells {
                layouter.constrain_instance(cell.cell(), public_instance, offset)?;
                offset += 1;
            }
        }
        for index in &self.params.public_output_indices {
            let tensor = assigned.virtuals.get(*index).ok_or(ErrorFront::Synthesis)?;
            for cell in &tensor.cells {
                layouter.constrain_instance(cell.cell(), public_instance, offset)?;
                offset += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{Advice, Circuit, Column, ConstraintSystem, ErrorFront};

    use crate::assembler::{AssemblerInstruction, AssemblerProgram, RegisterRef};
    use crate::fixed_point::I18;
    use crate::isa::Instruction;

    use super::{AssemblerChip, AssemblerCircuit, AssemblerConfig, Fr};

    fn dot_program(k: usize) -> AssemblerProgram {
        AssemblerProgram {
            instructions: vec![AssemblerInstruction {
                instruction: Instruction::DotGeneral {
                    m: 1,
                    n: 1,
                    k,
                    batch_dims: vec![],
                    trans_a: false,
                    trans_b: false,
                },
                inputs: vec![RegisterRef::Input(0), RegisterRef::Weight(0)],
            }],
            input_values: vec![vec![I18::from_raw(1_000_000_000_000_000_000); k]],
            weight_values: vec![vec![I18::from_raw(1_000_000_000_000_000_000); k]],
        }
    }

    fn dot_instances(k: usize) -> Vec<Vec<Fr>> {
        let one = crate::field_convert::i64_to_fr(1_000_000_000_000_000_000);
        let output =
            crate::field_convert::i64_to_fr(i64::try_from(k).unwrap() * 1_000_000_000_000_000_000);
        vec![(0..k)
            .map(|_| one)
            .chain((0..k).map(|_| one))
            .chain([output])
            .collect()]
    }

    #[test]
    fn runtime_params_configure_distinct_program_shapes() {
        let k2 = AssemblerCircuit::new(dot_program(2), Default::default());
        let k3 = AssemblerCircuit::new(dot_program(3), Default::default());

        MockProver::run(10, &k2, dot_instances(2))
            .unwrap()
            .assert_satisfied();
        MockProver::run(10, &k3, dot_instances(3))
            .unwrap()
            .assert_satisfied();
        assert_ne!(k2.params().shape_digest(), k3.params().shape_digest());
    }

    #[test]
    fn without_witnesses_keeps_shape_and_marks_witnesses_unknown() {
        let circuit = AssemblerCircuit::new(dot_program(2), Default::default());
        let blank = circuit.without_witnesses();

        assert_eq!(
            circuit.params().shape_digest(),
            blank.params().shape_digest()
        );
        assert!(!blank.has_known_witnesses());
        assert_eq!(blank.program.input_values.len(), 1);
        assert_eq!(blank.program.input_values[0].len(), 2);
        assert_eq!(blank.program.weight_values.len(), 1);
        assert_eq!(blank.program.weight_values[0].len(), 2);
    }

    #[test]
    fn public_output_selection_is_validated_and_shape_bound() {
        let mut program = dot_program(2);
        program.instructions.push(AssemblerInstruction {
            instruction: Instruction::Eltwise {
                op: crate::isa::EltwiseOp::Add,
            },
            inputs: vec![RegisterRef::Virtual(0), RegisterRef::Virtual(0)],
        });
        let first =
            AssemblerCircuit::new_with_public_outputs(program.clone(), Default::default(), vec![0])
                .unwrap();
        let second =
            AssemblerCircuit::new_with_public_outputs(program.clone(), Default::default(), vec![1])
                .unwrap();
        assert_ne!(
            first.params().shape_digest(),
            second.params().shape_digest()
        );
        assert!(AssemblerCircuit::new_with_public_outputs(
            program.clone(),
            Default::default(),
            vec![0, 0]
        )
        .is_err());
        assert!(
            AssemblerCircuit::new_with_public_outputs(program, Default::default(), vec![2])
                .is_err()
        );
    }

    #[test]
    fn plain_configure_is_an_empty_shape_compatibility_fallback() {
        let mut constraint_system = ConstraintSystem::default();
        let _ = AssemblerCircuit::configure(&mut constraint_system);
    }

    #[test]
    fn boundary_commitments_use_exact_assembler_cells() {
        use crate::chips::poseidon_boundary::{
            commit_boundary_native, BoundaryDescriptor, BoundaryRole,
        };
        let program = dot_program(2);
        let input_descriptor = BoundaryDescriptor::flat_i18(
            BoundaryRole::Input,
            "graph-input:x",
            Vec::new(),
            Vec::new(),
            2,
            1_000_000_000_000_000_000,
        )
        .unwrap();
        let output_descriptor = BoundaryDescriptor::flat_i18(
            BoundaryRole::Output,
            "virtual:0",
            Vec::new(),
            vec!["y".into()],
            1,
            1_000_000_000_000_000_000,
        )
        .unwrap();
        let input_commitment =
            commit_boundary_native(&input_descriptor, &program.input_values[0]).unwrap();
        let input_binding = input_descriptor.public_binding_fields();
        let output = [I18::from_raw(2_000_000_000_000_000_000)];
        let output_commitment = commit_boundary_native(&output_descriptor, &output).unwrap();
        let output_binding = output_descriptor.public_binding_fields();
        let mut altered_program = program.clone();
        altered_program.weight_values[0][0] = I18::from_raw(2_000_000_000_000_000_000);
        let altered = AssemblerCircuit::new_with_boundary_commitments(
            altered_program,
            Default::default(),
            vec![input_descriptor.clone()],
            vec![(0, output_descriptor.clone())],
            vec![Fr::from(42); 17],
        )
        .unwrap();
        let circuit = AssemblerCircuit::new_with_boundary_commitments(
            program,
            Default::default(),
            vec![input_descriptor],
            vec![(0, output_descriptor)],
            vec![Fr::from(42); 17],
        )
        .unwrap();
        assert_ne!(
            circuit.params().shape_digest(),
            altered.params().shape_digest(),
            "fixed public-model weights must produce a distinct key identity",
        );
        let one = crate::field_convert::i64_to_fr(1_000_000_000_000_000_000);
        let instances = vec![(0..17)
            .map(|_| Fr::from(42))
            .chain([
                input_binding[0],
                input_binding[1],
                input_commitment,
                Fr::from(1),
                output_binding[0],
                output_binding[1],
                output_commitment,
                Fr::from(2),
                one,
                one,
            ])
            .collect()];
        MockProver::run(12, &circuit, instances)
            .unwrap()
            .assert_satisfied();
    }

    #[derive(Clone)]
    struct CellConsumerConfig {
        assembler: AssemblerConfig,
        consumer: Column<Advice>,
    }

    #[derive(Clone)]
    struct CellConsumerCircuit {
        program: AssemblerProgram,
        consumer_value: I18,
    }

    impl Circuit<Fr> for CellConsumerCircuit {
        type Config = CellConsumerConfig;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let consumer = meta.advice_column();
            meta.enable_equality(consumer);
            CellConsumerConfig {
                assembler: AssemblerChip::configure(meta, &dot_program(2).instructions),
                consumer,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let assigned = AssemblerChip::construct(config.assembler)
                .assign_with_cells(layouter.namespace(|| "assembler"), &self.program)
                .map_err(|_| ErrorFront::Synthesis)?;
            let consumer = layouter.assign_region(
                || "external consumer",
                |mut region| {
                    region.assign_advice(
                        || "consumer value",
                        config.consumer,
                        0,
                        || Value::known(crate::field_convert::i64_to_fr(self.consumer_value.raw())),
                    )
                },
            )?;
            layouter.assign_region(
                || "consumer uses canonical virtual cell",
                |mut region| {
                    region.constrain_equal(consumer.cell(), assigned.virtuals[0].cells[0].cell())
                },
            )
        }
    }

    #[test]
    fn returned_virtual_cell_rejects_an_inconsistent_consumer_assignment() {
        let circuit = CellConsumerCircuit {
            program: dot_program(2),
            // The dot product is 1 * 1 + 1 * 1 = 2, hand-derived in I18 raw units.
            consumer_value: I18::from_raw(3_000_000_000_000_000_000),
        };
        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
