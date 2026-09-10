//! Immutable binding of one validated partition-plan shard into local assembler indices.

use std::collections::{HashMap, HashSet};
use std::fmt;

use zkie_core::assembler::{AssemblerInstruction, AssemblerProgram, RegisterRef};
use zkie_core::fixed_point::I18;
use zkie_types::Digest32;

use crate::dag::{build_dag, EdgeId, PartitionPlan, ShardSpec};
use crate::graph_compiler::{CompiledProgram, Register};
use crate::onnx_parser::OnnxParseError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryTensorSpec {
    register: Register,
    element_count: usize,
}

impl BoundaryTensorSpec {
    /// Describes only a canonical flat element count. The current compiler
    /// does not claim to know a higher-rank tensor shape here.
    pub fn flat(register: Register, element_count: usize) -> Result<Self, ShardBindingError> {
        if element_count == 0 || matches!(register, Register::Weight(_)) {
            return Err(ShardBindingError::InvalidBoundaryLayout {
                message: "boundary specs require a non-weight register and nonzero length".into(),
            });
        }
        Ok(Self {
            register,
            element_count,
        })
    }

    pub fn register(&self) -> &Register {
        &self.register
    }

    pub fn element_count(&self) -> usize {
        self.element_count
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryLayout {
    specs: Vec<BoundaryTensorSpec>,
}

impl BoundaryLayout {
    pub fn new(mut specs: Vec<BoundaryTensorSpec>) -> Result<Self, ShardBindingError> {
        specs.sort_unstable_by(|left, right| {
            register_key(&left.register).cmp(&register_key(&right.register))
        });
        if specs
            .windows(2)
            .any(|pair| pair[0].register == pair[1].register)
        {
            return Err(ShardBindingError::DuplicateBoundary);
        }
        Ok(Self { specs })
    }

    pub fn specs(&self) -> &[BoundaryTensorSpec] {
        &self.specs
    }

    pub fn digest(&self) -> Digest32 {
        let mut bytes = b"zkie.boundary-layout.v1\0".to_vec();
        bytes.extend_from_slice(&(self.specs.len() as u64).to_le_bytes());
        for spec in &self.specs {
            match &spec.register {
                Register::GraphInput(name) => {
                    bytes.push(0);
                    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
                    bytes.extend_from_slice(name.as_bytes());
                }
                Register::Virtual(index) => {
                    bytes.push(1);
                    bytes.extend_from_slice(&(*index as u64).to_le_bytes());
                }
                Register::Weight(_) => unreachable!("constructor rejects weight boundaries"),
            }
            bytes.extend_from_slice(&(spec.element_count as u64).to_le_bytes());
        }
        Digest32::new(*blake3::hash(&bytes).as_bytes())
    }

    fn get(&self, register: &Register) -> Result<&BoundaryTensorSpec, ShardBindingError> {
        self.specs
            .iter()
            .find(|spec| &spec.register == register)
            .ok_or_else(|| ShardBindingError::MissingBoundaryShape {
                register: register.clone(),
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryDescriptor {
    register: Register,
    element_count: usize,
    edge_ids: Vec<EdgeId>,
    graph_output_names: Vec<String>,
}

impl BoundaryDescriptor {
    pub fn register(&self) -> &Register {
        &self.register
    }
    pub fn element_count(&self) -> usize {
        self.element_count
    }
    pub fn edge_ids(&self) -> &[EdgeId] {
        &self.edge_ids
    }
    pub fn graph_output_names(&self) -> &[String] {
        &self.graph_output_names
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundShardProgram {
    assembler_program: AssemblerProgram,
    input_boundaries: Vec<BoundaryDescriptor>,
    output_boundaries: Vec<BoundaryDescriptor>,
    global_to_local: HashMap<usize, usize>,
    weight_names: Vec<String>,
}

impl BoundShardProgram {
    pub fn assembler_program(&self) -> &AssemblerProgram {
        &self.assembler_program
    }
    pub fn input_boundaries(&self) -> &[BoundaryDescriptor] {
        &self.input_boundaries
    }
    pub fn output_boundaries(&self) -> &[BoundaryDescriptor] {
        &self.output_boundaries
    }
    pub fn global_to_local(&self) -> &HashMap<usize, usize> {
        &self.global_to_local
    }
    pub fn weight_names(&self) -> &[String] {
        &self.weight_names
    }
}

#[derive(Debug)]
pub enum ShardBindingError {
    UnknownShard {
        shard_id: usize,
    },
    PlanProgramMismatch,
    BoundaryLayoutMismatch,
    InvalidBoundaryLayout {
        message: String,
    },
    DuplicateBoundary,
    MissingBoundaryShape {
        register: Register,
    },
    MissingBoundary {
        register: Register,
    },
    ExtraBoundary {
        register: Register,
    },
    MissingGraphInput {
        name: String,
    },
    BoundaryLengthMismatch {
        register: Register,
        expected: usize,
        actual: usize,
    },
    LaterOrOutsideReference {
        instruction: usize,
        register: Register,
    },
    OutputOutsideProgram {
        register: Register,
    },
    Weight(OnnxParseError),
}

impl fmt::Display for ShardBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ShardBindingError {}

impl From<OnnxParseError> for ShardBindingError {
    fn from(value: OnnxParseError) -> Self {
        Self::Weight(value)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bind_shard_program(
    program: &CompiledProgram,
    plan: &PartitionPlan,
    shard_id: usize,
    program_model_digest: Digest32,
    boundary_layout: &BoundaryLayout,
    graph_inputs: &HashMap<String, Vec<I18>>,
    incoming: &HashMap<Register, Vec<I18>>,
) -> Result<BoundShardProgram, ShardBindingError> {
    for (instruction, compiled) in program.instructions.iter().enumerate() {
        if let Some(register) = compiled.inputs.iter().find(
            |register| matches!(register, Register::Virtual(producer) if *producer >= instruction),
        ) {
            return Err(ShardBindingError::LaterOrOutsideReference {
                instruction,
                register: register.clone(),
            });
        }
    }
    if program_model_digest != plan.model_digest() || !plan.matches_program(program) {
        return Err(ShardBindingError::PlanProgramMismatch);
    }
    if plan.boundary_layout_digest() != Some(boundary_layout.digest()) {
        return Err(ShardBindingError::BoundaryLayoutMismatch);
    }
    let planned = plan
        .shards()
        .get(shard_id)
        .ok_or(ShardBindingError::UnknownShard { shard_id })?;
    let dag_shard = plan
        .dag()
        .shards
        .get(shard_id)
        .ok_or(ShardBindingError::PlanProgramMismatch)?;
    if planned.id() != shard_id as u64
        || dag_shard.id != shard_id
        || planned.range() != dag_shard.range
        || planned.name() != dag_shard.name
    {
        return Err(ShardBindingError::PlanProgramMismatch);
    }
    let specs = plan
        .shards()
        .iter()
        .map(|shard| ShardSpec {
            name: shard.name().into(),
            range: shard.range(),
        })
        .collect::<Vec<_>>();
    let rebuilt = build_dag(program, &specs).map_err(|_| ShardBindingError::PlanProgramMismatch)?;
    if &rebuilt != plan.dag() {
        return Err(ShardBindingError::PlanProgramMismatch);
    }

    let range = planned.range();
    let expected_incoming = dag_shard.inputs.iter().cloned().collect::<HashSet<_>>();
    for register in &expected_incoming {
        if !incoming.contains_key(register) {
            return Err(ShardBindingError::MissingBoundary {
                register: register.clone(),
            });
        }
    }
    if let Some(register) = incoming
        .keys()
        .find(|register| !expected_incoming.contains(*register))
    {
        return Err(ShardBindingError::ExtraBoundary {
            register: register.clone(),
        });
    }

    let mut external = HashSet::new();
    for (offset, instruction) in program.instructions[range.clone()].iter().enumerate() {
        let global = range.start + offset;
        for register in &instruction.inputs {
            match register {
                Register::GraphInput(_) => {
                    external.insert(register.clone());
                }
                Register::Weight(_) => {}
                Register::Virtual(source) if *source < range.start => {
                    if !expected_incoming.contains(register) {
                        return Err(ShardBindingError::PlanProgramMismatch);
                    }
                    external.insert(register.clone());
                }
                Register::Virtual(source) if *source < global && range.contains(source) => {}
                Register::Virtual(_) => {
                    return Err(ShardBindingError::LaterOrOutsideReference {
                        instruction: global,
                        register: register.clone(),
                    });
                }
            }
        }
    }
    let mut external = external.into_iter().collect::<Vec<_>>();
    external.sort_unstable_by(|left, right| register_key(left).cmp(&register_key(right)));

    let mut input_values = Vec::with_capacity(external.len());
    let mut input_index = HashMap::new();
    let mut input_boundaries = Vec::with_capacity(external.len());
    for register in external {
        let spec = boundary_layout.get(&register)?;
        let values = match &register {
            Register::GraphInput(name) => graph_inputs
                .get(name)
                .ok_or_else(|| ShardBindingError::MissingGraphInput { name: name.clone() })?,
            Register::Virtual(_) => {
                incoming
                    .get(&register)
                    .ok_or_else(|| ShardBindingError::MissingBoundary {
                        register: register.clone(),
                    })?
            }
            Register::Weight(_) => unreachable!(),
        };
        if values.len() != spec.element_count {
            return Err(ShardBindingError::BoundaryLengthMismatch {
                register,
                expected: spec.element_count,
                actual: values.len(),
            });
        }
        let index = input_values.len();
        input_index.insert(register.clone(), index);
        input_values.push(values.clone());
        input_boundaries.push(descriptor(
            plan,
            shard_id,
            register,
            spec.element_count,
            false,
        ));
    }

    let global_to_local = range
        .clone()
        .enumerate()
        .map(|(local, global)| (global, local))
        .collect::<HashMap<_, _>>();
    let mut weight_names = Vec::new();
    let mut weight_index = HashMap::new();
    let mut weight_values = Vec::new();
    let mut instructions = Vec::with_capacity(range.len());
    for (offset, compiled) in program.instructions[range.clone()].iter().enumerate() {
        let global = range.start + offset;
        let inputs = compiled
            .inputs
            .iter()
            .map(|register| match register {
                Register::GraphInput(_) => Ok(RegisterRef::Input(input_index[register])),
                Register::Weight(name) => {
                    let index = if let Some(index) = weight_index.get(name) {
                        *index
                    } else {
                        let index = weight_values.len();
                        weight_values.push(program.weight_as_i18(name)?);
                        weight_names.push(name.clone());
                        weight_index.insert(name.clone(), index);
                        index
                    };
                    Ok(RegisterRef::Weight(index))
                }
                Register::Virtual(source) if range.contains(source) && *source < global => {
                    Ok(RegisterRef::Virtual(global_to_local[source]))
                }
                Register::Virtual(source) if *source < range.start => {
                    Ok(RegisterRef::Input(input_index[register]))
                }
                _ => Err(ShardBindingError::LaterOrOutsideReference {
                    instruction: global,
                    register: register.clone(),
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        instructions.push(AssemblerInstruction {
            instruction: compiled.instruction.clone(),
            inputs,
        });
    }

    let mut outputs = dag_shard.outputs.iter().cloned().collect::<HashSet<_>>();
    let mut graph_output_names = HashMap::<Register, Vec<String>>::new();
    for (name, register) in &program.graph_outputs {
        let Register::Virtual(index) = register else {
            return Err(ShardBindingError::OutputOutsideProgram {
                register: register.clone(),
            });
        };
        if *index >= program.instructions.len() {
            return Err(ShardBindingError::OutputOutsideProgram {
                register: register.clone(),
            });
        }
        if range.contains(index) {
            outputs.insert(register.clone());
            graph_output_names
                .entry(register.clone())
                .or_default()
                .push(name.clone());
        }
    }
    let mut outputs = outputs.into_iter().collect::<Vec<_>>();
    outputs.sort_unstable_by(|left, right| register_key(left).cmp(&register_key(right)));
    let output_boundaries = outputs
        .into_iter()
        .map(|register| {
            let spec = boundary_layout.get(&register)?;
            let mut descriptor =
                descriptor(plan, shard_id, register.clone(), spec.element_count, true);
            descriptor.graph_output_names =
                graph_output_names.remove(&register).unwrap_or_default();
            descriptor.graph_output_names.sort_unstable();
            Ok(descriptor)
        })
        .collect::<Result<Vec<_>, ShardBindingError>>()?;

    Ok(BoundShardProgram {
        assembler_program: AssemblerProgram {
            instructions,
            input_values,
            weight_values,
        },
        input_boundaries,
        output_boundaries,
        global_to_local,
        weight_names,
    })
}

fn descriptor(
    plan: &PartitionPlan,
    shard_id: usize,
    register: Register,
    element_count: usize,
    producer_side: bool,
) -> BoundaryDescriptor {
    let mut edge_ids = plan
        .dag()
        .edges
        .iter()
        .filter(|edge| {
            edge.register == register
                && if producer_side {
                    edge.producer == shard_id
                } else {
                    edge.consumer == shard_id
                }
        })
        .map(|edge| edge.id.clone())
        .collect::<Vec<_>>();
    edge_ids.sort_unstable();
    BoundaryDescriptor {
        register,
        element_count,
        edge_ids,
        graph_output_names: Vec::new(),
    }
}

fn register_key(register: &Register) -> (u8, &str, usize) {
    match register {
        Register::GraphInput(name) => (0, name, 0),
        Register::Virtual(index) => (1, "", *index),
        Register::Weight(name) => (2, name, 0),
    }
}
