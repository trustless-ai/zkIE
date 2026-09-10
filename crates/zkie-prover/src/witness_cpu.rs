//! Deterministic host execution for the zkIE instruction subset proven by the assembler.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use zkie_compiler::dag::Shard;
use zkie_compiler::graph_compiler::{CompiledProgram, Register};
use zkie_compiler::onnx_parser::OnnxParseError;
use zkie_core::chips::layer_norm::{rsqrt_f64, RsqrtDomain};
use zkie_core::chips::lookup::{build_domain, build_domain_from_raw};
use zkie_core::fixed_point::{requantize_mul, requantize_raw, I18};
use zkie_core::isa::{EltwiseOp, Instruction, ReduceOp};
use zkie_types::{
    Digest32, ExecutionBackendId, ModelVisibility, ProofFlavorId, ResourceRequest, RunIdentity,
};

use crate::{
    BackendCapabilities, BackendError, CapabilityId, ShardIdentity, WitnessArtifact,
    WitnessBackend, WitnessJob,
};

const INPUT_SCHEMA_VERSION: u32 = 1;
const ARTIFACT_SCHEMA_VERSION: u32 = 1;
const CIRCUIT_BINDING_VERSION: &[u8] = b"zkie.cpu-witness-circuit-binding.v1\0";
const SUPPORTED_PROOF_FLAVOR: &str = "halo2-kzg-bn256-shplonk-v1";

pub const MAX_CPU_WITNESS_INPUT_BYTES: usize = 1 << 20;
pub const MAX_CPU_WITNESS_ARTIFACT_BYTES: usize = 8 << 20;
pub const MAX_CPU_WITNESS_VALUES_PER_TENSOR: usize = 1 << 16;
pub const MAX_CPU_WITNESS_DOMAIN_POINTS: usize = 1 << 16;
pub const MAX_CPU_WITNESS_TOTAL_INPUT_VALUES: usize = 1 << 18;
pub const MAX_CPU_WITNESS_TOTAL_ARTIFACT_VALUES: usize = 1 << 20;
pub const MAX_CPU_WITNESS_TENSORS: usize = 1 << 10;
pub const MAX_CPU_WITNESS_INSTRUCTIONS: usize = 1 << 10;
pub const MAX_CPU_WITNESS_OPERAND_REFERENCES: usize = 1 << 14;

/// Absolute allocation and serialization limits for one backend instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuWitnessLimits {
    pub input_bytes: usize,
    pub artifact_bytes: usize,
    pub values_per_tensor: usize,
    pub total_input_values: usize,
    pub total_artifact_values: usize,
    pub tensor_count: usize,
    pub domain_points: usize,
    pub instruction_count: usize,
    pub operand_references: usize,
}

impl CpuWitnessLimits {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_bytes: usize,
        artifact_bytes: usize,
        values_per_tensor: usize,
        total_input_values: usize,
        total_artifact_values: usize,
        tensor_count: usize,
        domain_points: usize,
        instruction_count: usize,
        operand_references: usize,
    ) -> Result<Self, BackendError> {
        let limits = Self {
            input_bytes,
            artifact_bytes,
            values_per_tensor,
            total_input_values,
            total_artifact_values,
            tensor_count,
            domain_points,
            instruction_count,
            operand_references,
        };
        if [
            input_bytes,
            artifact_bytes,
            values_per_tensor,
            total_input_values,
            total_artifact_values,
            tensor_count,
            domain_points,
            instruction_count,
            operand_references,
        ]
        .contains(&0)
            || values_per_tensor > total_input_values
            || values_per_tensor > total_artifact_values
        {
            return Err(BackendError::Resource {
                message: "CPU witness limits must be nonzero and internally consistent".into(),
            });
        }
        limits.estimated_ram_bytes()?;
        Ok(limits)
    }

    fn estimated_ram_bytes(&self) -> Result<u64, BackendError> {
        let bytes = self
            .input_bytes
            .checked_add(self.artifact_bytes)
            .and_then(|v| v.checked_add(self.total_input_values.checked_mul(16)?))
            .and_then(|v| v.checked_add(self.total_artifact_values.checked_mul(16)?))
            .and_then(|v| v.checked_add(self.domain_points.checked_mul(32)?))
            .and_then(|v| v.checked_add(self.input_bytes.checked_mul(1)?))
            .and_then(|v| v.checked_add(self.artifact_bytes.checked_mul(2)?))
            .and_then(|v| v.checked_add(self.total_input_values.checked_mul(16)?))
            .and_then(|v| v.checked_add(self.total_artifact_values.checked_mul(32)?))
            .and_then(|v| v.checked_add(self.domain_points.checked_mul(64)?))
            .and_then(|v| v.checked_add(self.tensor_count.checked_mul(512)?))
            .and_then(|v| v.checked_add(self.instruction_count.checked_mul(512)?))
            .and_then(|v| v.checked_add(self.operand_references.checked_mul(128)?))
            .and_then(|v| v.checked_add(32 << 20))
            .ok_or_else(|| BackendError::Resource {
                message: "CPU witness resource estimate overflow".into(),
            })?;
        u64::try_from(bytes).map_err(|_| BackendError::Resource {
            message: "CPU witness resource estimate does not fit u64".into(),
        })
    }
}

impl Default for CpuWitnessLimits {
    fn default() -> Self {
        Self {
            input_bytes: MAX_CPU_WITNESS_INPUT_BYTES,
            artifact_bytes: MAX_CPU_WITNESS_ARTIFACT_BYTES,
            values_per_tensor: MAX_CPU_WITNESS_VALUES_PER_TENSOR,
            total_input_values: MAX_CPU_WITNESS_TOTAL_INPUT_VALUES,
            total_artifact_values: MAX_CPU_WITNESS_TOTAL_ARTIFACT_VALUES,
            tensor_count: MAX_CPU_WITNESS_TENSORS,
            domain_points: MAX_CPU_WITNESS_DOMAIN_POINTS,
            instruction_count: MAX_CPU_WITNESS_INSTRUCTIONS,
            operand_references: MAX_CPU_WITNESS_OPERAND_REFERENCES,
        }
    }
}

/// Concrete values supplied to one shard execution. Values are already quantized I18.
#[derive(Clone, Debug, Default)]
pub struct CpuWitnessInputs {
    graph_inputs: HashMap<String, Vec<I18>>,
    virtual_inputs: BTreeMap<usize, Vec<I18>>,
}

impl CpuWitnessInputs {
    pub fn new(
        graph_inputs: HashMap<String, Vec<I18>>,
        virtual_inputs: BTreeMap<usize, Vec<I18>>,
    ) -> Self {
        Self {
            graph_inputs,
            virtual_inputs,
        }
    }

    pub fn graph_inputs(&self) -> &HashMap<String, Vec<I18>> {
        &self.graph_inputs
    }

    pub fn virtual_inputs(&self) -> &BTreeMap<usize, Vec<I18>> {
        &self.virtual_inputs
    }
}

/// The exact register state needed to lower this shard into a proof circuit.
#[derive(Clone, Debug)]
pub struct CpuWitnessArtifact {
    shard: Shard,
    shard_identity: ShardIdentity,
    run_identity: RunIdentity,
    run_identity_digest: Digest32,
    inputs: HashMap<Register, Vec<I18>>,
    virtuals: BTreeMap<usize, Vec<I18>>,
    outputs: BTreeMap<usize, Vec<I18>>,
    record_limit: usize,
}

impl CpuWitnessArtifact {
    pub fn shard(&self) -> &Shard {
        &self.shard
    }

    pub fn circuit_digest(&self) -> Digest32 {
        self.shard_identity.circuit_digest()
    }

    pub fn shard_identity(&self) -> &ShardIdentity {
        &self.shard_identity
    }

    pub fn run_identity(&self) -> &RunIdentity {
        &self.run_identity
    }

    pub fn run_identity_digest(&self) -> Digest32 {
        self.run_identity_digest
    }

    pub fn input_value(&self, register: &Register) -> Option<&[I18]> {
        self.inputs.get(register).map(Vec::as_slice)
    }

    pub fn virtual_value(&self, index: usize) -> Option<&[I18]> {
        self.virtuals.get(&index).map(Vec::as_slice)
    }

    pub fn output_value(&self, index: usize) -> Option<&[I18]> {
        self.outputs.get(&index).map(Vec::as_slice)
    }

    pub fn virtuals(&self) -> &BTreeMap<usize, Vec<I18>> {
        &self.virtuals
    }

    pub fn outputs(&self) -> &BTreeMap<usize, Vec<I18>> {
        &self.outputs
    }
}

/// CPU backend for the exact fixed-point subset currently dispatched by `AssemblerChip`.
pub struct ZkieIsaCpuWitnessBackend {
    program: Arc<CompiledProgram>,
    shard: Shard,
    shard_identity: ShardIdentity,
    run_identity: RunIdentity,
    rms_norm_tables: HashMap<(usize, u64), BTreeMap<i64, I18>>,
    weights: HashMap<String, Vec<I18>>,
    required_graph_inputs: BTreeSet<String>,
    required_virtual_inputs: BTreeSet<usize>,
    artifact_record_count: usize,
    limits: CpuWitnessLimits,
}

impl ZkieIsaCpuWitnessBackend {
    pub fn new(
        program: Arc<CompiledProgram>,
        shard: Shard,
        rms_norm_domains: HashMap<(usize, u64), RsqrtDomain>,
        run_identity: RunIdentity,
    ) -> Result<Self, BackendError> {
        Self::with_limits(
            program,
            shard,
            rms_norm_domains,
            run_identity,
            CpuWitnessLimits::default(),
        )
    }

    pub fn with_limits(
        program: Arc<CompiledProgram>,
        shard: Shard,
        rms_norm_domains: HashMap<(usize, u64), RsqrtDomain>,
        run_identity: RunIdentity,
        limits: CpuWitnessLimits,
    ) -> Result<Self, BackendError> {
        CpuWitnessLimits::new(
            limits.input_bytes,
            limits.artifact_bytes,
            limits.values_per_tensor,
            limits.total_input_values,
            limits.total_artifact_values,
            limits.tensor_count,
            limits.domain_points,
            limits.instruction_count,
            limits.operand_references,
        )?;
        validate_run_identity(&run_identity)?;
        if shard.range.start > shard.range.end || shard.range.end > program.instructions.len() {
            return Err(BackendError::InvalidJob {
                message: format!(
                    "shard range {:?} is outside {} instructions",
                    shard.range,
                    program.instructions.len()
                ),
            });
        }
        if shard.inputs.iter().collect::<HashSet<_>>().len() != shard.inputs.len()
            || shard.outputs.iter().collect::<HashSet<_>>().len() != shard.outputs.len()
        {
            return Err(BackendError::InvalidJob {
                message: "shard boundary registers must be unique".into(),
            });
        }
        for register in &shard.inputs {
            let Register::Virtual(index) = register else {
                return Err(BackendError::InvalidJob {
                    message: format!("shard boundary contains non-virtual register {register:?}"),
                });
            };
            if *index >= shard.range.start {
                return Err(BackendError::InvalidJob {
                    message: format!("shard input {register:?} is not produced before its range"),
                });
            }
        }
        for register in &shard.outputs {
            let Register::Virtual(index) = register else {
                return Err(BackendError::InvalidJob {
                    message: format!("shard boundary contains non-virtual register {register:?}"),
                });
            };
            if !shard.range.contains(index) {
                return Err(BackendError::InvalidJob {
                    message: format!("shard output {register:?} is not produced inside its range"),
                });
            }
        }
        let analysis = analyze_program(&program, &shard, &rms_norm_domains, &limits)?;
        let circuit_digest = derive_circuit_digest(
            &program,
            &shard,
            &rms_norm_domains,
            &analysis.weights,
            &analysis.rms_norm_tables,
        );
        let shard_identity = ShardIdentity::new(
            u64::try_from(shard.id).map_err(|_| BackendError::InvalidJob {
                message: "configured shard id does not fit u64".into(),
            })?,
            shard.name.clone(),
            circuit_digest,
        )?;
        Ok(Self {
            program,
            shard,
            shard_identity,
            run_identity,
            rms_norm_tables: analysis.rms_norm_tables,
            weights: analysis.weights,
            required_graph_inputs: analysis.required_graph_inputs,
            required_virtual_inputs: analysis.required_virtual_inputs,
            artifact_record_count: analysis.artifact_record_count,
            limits,
        })
    }

    pub fn circuit_digest(&self) -> Digest32 {
        self.shard_identity.circuit_digest()
    }

    pub fn shard_identity(&self) -> &ShardIdentity {
        &self.shard_identity
    }

    pub fn run_identity(&self) -> &RunIdentity {
        &self.run_identity
    }

    pub fn execute(&self, supplied: &CpuWitnessInputs) -> Result<CpuWitnessArtifact, BackendError> {
        self.validate_supplied_inputs(supplied)?;
        let mut registers = HashMap::<Register, Vec<I18>>::new();
        let mut boundary_inputs = HashMap::<Register, Vec<I18>>::new();
        let mut virtuals = BTreeMap::new();
        let mut generated_values = 0usize;

        for index in self.shard.range.clone() {
            let compiled = &self.program.instructions[index];
            for register in &compiled.inputs {
                if registers.contains_key(register) {
                    continue;
                }
                let values = self.resolve_external(register, index, supplied)?;
                boundary_inputs.insert(register.clone(), values.clone());
                registers.insert(register.clone(), values);
            }
            let operands = compiled
                .inputs
                .iter()
                .map(|register| {
                    registers.get(register).map(Vec::as_slice).ok_or_else(|| {
                        BackendError::MissingRegister {
                            register: format!("{register:?}"),
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output = self.execute_instruction(index, &compiled.instruction, &operands)?;
            if output.len() > self.limits.values_per_tensor {
                return Err(BackendError::Resource {
                    message: format!("Virtual({index}) exceeds tensor value limit"),
                });
            }
            generated_values = generated_values.checked_add(output.len()).ok_or_else(|| {
                BackendError::Resource {
                    message: "generated witness value count overflow".into(),
                }
            })?;
            if generated_values > self.limits.total_artifact_values {
                return Err(BackendError::Resource {
                    message: "generated witness values exceed artifact limit".into(),
                });
            }
            registers.insert(Register::Virtual(index), output.clone());
            virtuals.insert(index, output);
        }

        let boundary_value_count = boundary_inputs.values().map(Vec::len).sum::<usize>();
        let output_value_count = self
            .required_outputs()
            .iter()
            .filter_map(|index| virtuals.get(index))
            .map(Vec::len)
            .sum::<usize>();
        let artifact_value_count = boundary_value_count
            .checked_add(generated_values)
            .and_then(|count| count.checked_add(output_value_count))
            .ok_or_else(|| BackendError::Resource {
                message: "artifact witness value count overflow".into(),
            })?;
        if artifact_value_count > self.limits.total_artifact_values {
            return Err(BackendError::Resource {
                message: "artifact witness values exceed configured limit".into(),
            });
        }
        let mut outputs = BTreeMap::new();
        for register in &self.shard.outputs {
            let Register::Virtual(index) = register else {
                unreachable!("constructor validates shard boundary registers")
            };
            let values =
                registers
                    .get(register)
                    .cloned()
                    .ok_or_else(|| BackendError::MissingRegister {
                        register: format!("{register:?}"),
                    })?;
            outputs.insert(*index, values);
        }
        let artifact_record_count = validate_artifact_record_count(
            boundary_inputs.len(),
            virtuals.len(),
            outputs.len(),
            self.limits.tensor_count,
        )?;
        if artifact_record_count != self.artifact_record_count {
            return Err(BackendError::InvalidJob {
                message: "CPU witness artifact record coverage changed during execution".into(),
            });
        }
        Ok(CpuWitnessArtifact {
            shard: self.shard.clone(),
            shard_identity: self.shard_identity.clone(),
            run_identity_digest: self.run_identity.canonical_digest(),
            run_identity: self.run_identity.clone(),
            inputs: boundary_inputs,
            virtuals,
            outputs,
            record_limit: self.limits.tensor_count,
        })
    }

    fn validate_supplied_inputs(&self, supplied: &CpuWitnessInputs) -> Result<(), BackendError> {
        let graph_keys = supplied
            .graph_inputs
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let virtual_keys = supplied
            .virtual_inputs
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        if let Some(name) = self.required_graph_inputs.difference(&graph_keys).next() {
            return Err(BackendError::MissingRegister {
                register: format!("GraphInput({name:?})"),
            });
        }
        if let Some(index) = self
            .required_virtual_inputs
            .difference(&virtual_keys)
            .next()
        {
            return Err(BackendError::MissingRegister {
                register: format!("Virtual({index})"),
            });
        }
        if graph_keys != self.required_graph_inputs || virtual_keys != self.required_virtual_inputs
        {
            return Err(BackendError::InvalidJob {
                message: "concrete inputs contain extraneous registers".into(),
            });
        }
        validate_tensor_collection(
            supplied
                .graph_inputs
                .values()
                .chain(supplied.virtual_inputs.values()),
            self.limits.tensor_count,
            self.limits.values_per_tensor,
            self.limits.total_input_values,
            "CPU witness inputs",
        )
    }

    fn resolve_external(
        &self,
        register: &Register,
        instruction: usize,
        supplied: &CpuWitnessInputs,
    ) -> Result<Vec<I18>, BackendError> {
        match register {
            Register::GraphInput(name) => {
                supplied.graph_inputs.get(name).cloned().ok_or_else(|| {
                    BackendError::MissingRegister {
                        register: format!("GraphInput({name:?})"),
                    }
                })
            }
            Register::Weight(name) => {
                self.weights
                    .get(name)
                    .cloned()
                    .ok_or_else(|| BackendError::MissingRegister {
                        register: format!("Weight({name:?})"),
                    })
            }
            Register::Virtual(index) if *index >= instruction => {
                Err(BackendError::ForwardRegister {
                    instruction,
                    register: format!("Virtual({index})"),
                })
            }
            Register::Virtual(index) if *index < self.shard.range.start => {
                if !self.shard.inputs.contains(register) {
                    return Err(BackendError::MissingRegister {
                        register: format!("undeclared Virtual({index}) shard input"),
                    });
                }
                supplied.virtual_inputs.get(index).cloned().ok_or_else(|| {
                    BackendError::MissingRegister {
                        register: format!("Virtual({index})"),
                    }
                })
            }
            Register::Virtual(index) => Err(BackendError::MissingRegister {
                register: format!("Virtual({index}) was not produced"),
            }),
        }
    }

    fn execute_instruction(
        &self,
        index: usize,
        instruction: &Instruction,
        operands: &[&[I18]],
    ) -> Result<Vec<I18>, BackendError> {
        match instruction {
            Instruction::DotGeneral {
                m,
                n,
                k,
                batch_dims,
                trans_a,
                trans_b,
            } => self.dot_general(index, operands, *m, *n, *k, batch_dims, *trans_a, *trans_b),
            Instruction::Eltwise {
                op: op @ (EltwiseOp::Add | EltwiseOp::Mul),
            } => self.eltwise(index, operands, op),
            Instruction::RmsNorm { dim, epsilon_milli } => {
                self.rms_norm(index, operands, *dim, *epsilon_milli)
            }
            other => Err(BackendError::UnsupportedInstruction {
                message: format!("{other:?}"),
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dot_general(
        &self,
        index: usize,
        operands: &[&[I18]],
        m: usize,
        n: usize,
        k: usize,
        batch_dims: &[usize],
        trans_a: bool,
        trans_b: bool,
    ) -> Result<Vec<I18>, BackendError> {
        if !batch_dims.is_empty() {
            return Err(BackendError::UnsupportedInstruction {
                message: "DotGeneral with batch dimensions".into(),
            });
        }
        expect_operands(index, "DotGeneral", operands, 2)?;
        let expected_a = checked_len(index, m, k)?;
        let expected_b = checked_len(index, k, n)?;
        if operands[0].len() != expected_a || operands[1].len() != expected_b {
            return shape_error(
                index,
                format!(
                    "DotGeneral expected ({expected_a}, {expected_b}) elements, got ({}, {})",
                    operands[0].len(),
                    operands[1].len()
                ),
            );
        }
        let out_len = checked_len(index, m, n)?;
        let mut output = Vec::with_capacity(out_len);
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0_i128;
                for inner in 0..k {
                    let a_index = if trans_a {
                        inner * m + row
                    } else {
                        row * k + inner
                    };
                    let b_index = if trans_b {
                        column * k + inner
                    } else {
                        inner * n + column
                    };
                    let product = i128::from(operands[0][a_index].raw())
                        * i128::from(operands[1][b_index].raw());
                    sum = sum
                        .checked_add(product)
                        .ok_or_else(|| overflow(index, "dot accumulation"))?;
                }
                output.push(
                    requantize_raw(sum)
                        .map_err(|error| overflow(index, error.to_string()))?
                        .0,
                );
            }
        }
        Ok(output)
    }

    fn eltwise(
        &self,
        index: usize,
        operands: &[&[I18]],
        op: &EltwiseOp,
    ) -> Result<Vec<I18>, BackendError> {
        expect_operands(index, "Eltwise", operands, 2)?;
        let a_len = operands[0].len();
        let b_len = operands[1].len();
        let out_len = a_len.max(b_len);
        if a_len == 0
            || b_len == 0
            || !out_len.is_multiple_of(a_len)
            || !out_len.is_multiple_of(b_len)
        {
            return shape_error(
                index,
                format!("Eltwise operands are not broadcast-compatible: {a_len}, {b_len}"),
            );
        }
        (0..out_len)
            .map(|position| {
                let a = operands[0][position % a_len];
                let b = operands[1][position % b_len];
                match op {
                    EltwiseOp::Add => a
                        .raw()
                        .checked_add(b.raw())
                        .map(I18::from_raw)
                        .ok_or_else(|| overflow(index, "eltwise add")),
                    EltwiseOp::Mul => requantize_mul(a, b)
                        .map(|(value, _)| value)
                        .map_err(|error| overflow(index, error.to_string())),
                    EltwiseOp::Relu => unreachable!("caller filters supported eltwise ops"),
                }
            })
            .collect()
    }

    fn rms_norm(
        &self,
        index: usize,
        operands: &[&[I18]],
        dim: usize,
        epsilon_milli: u64,
    ) -> Result<Vec<I18>, BackendError> {
        expect_operands(index, "RmsNorm", operands, 2)?;
        if dim == 0 || operands[0].len() != dim || operands[1].len() != dim {
            return shape_error(
                index,
                format!(
                    "RmsNorm dim {dim} requires two {dim}-element tensors, got {} and {}",
                    operands[0].len(),
                    operands[1].len()
                ),
            );
        }
        let squares = operands[0]
            .iter()
            .map(|value| {
                requantize_mul(*value, *value)
                    .map(|(square, _)| square)
                    .map_err(|error| overflow(index, error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sum_raw = squares.iter().try_fold(0_i64, |sum, value| {
            sum.checked_add(value.raw())
                .ok_or_else(|| overflow(index, "RmsNorm square sum"))
        })?;
        let reciprocal =
            I18::from_f64(1.0 / dim as f64).map_err(|error| overflow(index, error.to_string()))?;
        let mean = requantize_mul(I18::from_raw(sum_raw), reciprocal)
            .map_err(|error| overflow(index, error.to_string()))?
            .0;
        let epsilon = I18::from_f64(epsilon_milli as f64 / 1000.0)
            .map_err(|error| overflow(index, error.to_string()))?;
        let variance_plus_epsilon = mean
            .raw()
            .checked_add(epsilon.raw())
            .map(I18::from_raw)
            .ok_or_else(|| overflow(index, "RmsNorm variance plus epsilon"))?;
        let domain = self
            .rms_norm_tables
            .get(&(dim, epsilon_milli))
            .ok_or_else(|| BackendError::InvalidJob {
                message: format!("missing RMSNorm domain for ({dim}, {epsilon_milli})"),
            })?;
        let rsqrt = domain
            .get(&variance_plus_epsilon.raw())
            .copied()
            .ok_or_else(|| BackendError::InvalidJob {
                message: format!(
                    "RMSNorm input raw {} is absent from its lookup domain",
                    variance_plus_epsilon.raw()
                ),
            })?;
        operands[0]
            .iter()
            .zip(operands[1])
            .map(|(x, weight)| {
                let normalized = requantize_mul(*x, rsqrt)
                    .map_err(|error| overflow(index, error.to_string()))?
                    .0;
                requantize_mul(normalized, *weight)
                    .map(|(value, _)| value)
                    .map_err(|error| overflow(index, error.to_string()))
            })
            .collect()
    }

    fn validate_job(&self, job: &WitnessJob) -> Result<(), BackendError> {
        if job.run_identity() != &self.run_identity
            || job.shard() != &self.shard_identity
            || job.expected_circuit_digest() != self.circuit_digest()
        {
            return Err(BackendError::InvalidJob {
                message: "witness job does not match configured shard/circuit".into(),
            });
        }
        Ok(())
    }

    /// Loads a generated artifact, verifies its file digest and reconstructs trusted I18 state.
    pub fn load_artifact(
        &self,
        artifact: &WitnessArtifact,
    ) -> Result<CpuWitnessArtifact, BackendError> {
        if artifact.shard() != &self.shard_identity
            || artifact.circuit_digest() != self.circuit_digest()
        {
            return Err(BackendError::InvalidJob {
                message: "witness artifact metadata does not match configured backend".into(),
            });
        }
        let bytes = read_bounded(artifact.path(), self.limits.artifact_bytes)?;
        if Digest32::new(*blake3::hash(&bytes).as_bytes()) != artifact.digest() {
            return Err(BackendError::InvalidJob {
                message: "witness artifact file digest mismatch".into(),
            });
        }
        let wire: CpuWitnessArtifactWire =
            serde_json::from_slice(&bytes).map_err(|error| BackendError::Serialization {
                format: "zkie-cpu-witness-artifact-json-v1".into(),
                message: error.to_string(),
            })?;
        self.decode_artifact_wire(wire)
    }

    fn decode_artifact_wire(
        &self,
        wire: CpuWitnessArtifactWire,
    ) -> Result<CpuWitnessArtifact, BackendError> {
        if wire.schema_version != ARTIFACT_SCHEMA_VERSION
            || wire.run_identity != self.run_identity
            || wire.run_identity_digest != wire.run_identity.canonical_digest()
            || wire.run_identity_digest != self.run_identity.canonical_digest()
            || wire.shard_identity != self.shard_identity
            || wire.circuit_digest != self.circuit_digest()
        {
            return Err(BackendError::InvalidJob {
                message: "CPU witness artifact identity/schema mismatch".into(),
            });
        }

        let mut inputs = HashMap::new();
        for entry in wire.inputs {
            let register = entry.register.into_register();
            let values = entry.raw_values.into_iter().map(I18::from_raw).collect();
            if inputs.insert(register.clone(), values).is_some() {
                return Err(BackendError::InvalidJob {
                    message: format!("duplicate artifact input register {register:?}"),
                });
            }
        }
        let virtuals = into_i18_indexed(wire.virtuals, "virtual")?;
        let outputs = into_i18_indexed(wire.outputs, "output")?;
        let artifact_record_count = validate_artifact_record_count(
            inputs.len(),
            virtuals.len(),
            outputs.len(),
            self.limits.tensor_count,
        )?;
        if artifact_record_count != self.artifact_record_count {
            return Err(BackendError::InvalidJob {
                message: "CPU witness artifact record count mismatch".into(),
            });
        }
        validate_tensor_collection(
            inputs
                .values()
                .chain(virtuals.values())
                .chain(outputs.values()),
            self.limits.tensor_count,
            self.limits.values_per_tensor,
            self.limits.total_artifact_values,
            "CPU witness artifact",
        )?;

        let expected_inputs = self.expected_artifact_inputs();
        if inputs.keys().cloned().collect::<HashSet<_>>() != expected_inputs
            || virtuals.keys().copied().collect::<BTreeSet<_>>()
                != self.shard.range.clone().collect()
            || outputs.keys().copied().collect::<BTreeSet<_>>() != self.required_outputs()
        {
            return Err(BackendError::InvalidJob {
                message: "CPU witness artifact register coverage mismatch".into(),
            });
        }
        for (index, output) in &outputs {
            if virtuals.get(index) != Some(output) {
                return Err(BackendError::InvalidJob {
                    message: format!("artifact output Virtual({index}) differs from its virtual"),
                });
            }
        }
        let supplied = CpuWitnessInputs::new(
            inputs
                .iter()
                .filter_map(|(register, values)| match register {
                    Register::GraphInput(name) => Some((name.clone(), values.clone())),
                    _ => None,
                })
                .collect(),
            inputs
                .iter()
                .filter_map(|(register, values)| match register {
                    Register::Virtual(index) => Some((*index, values.clone())),
                    _ => None,
                })
                .collect(),
        );
        let recomputed = self.execute(&supplied)?;
        if recomputed.inputs != inputs
            || recomputed.virtuals != virtuals
            || recomputed.outputs != outputs
        {
            return Err(BackendError::InvalidJob {
                message: "CPU witness artifact values do not match deterministic execution".into(),
            });
        }
        Ok(CpuWitnessArtifact {
            shard: self.shard.clone(),
            shard_identity: wire.shard_identity,
            run_identity: wire.run_identity,
            run_identity_digest: wire.run_identity_digest,
            inputs,
            virtuals,
            outputs,
            record_limit: self.limits.tensor_count,
        })
    }

    fn expected_artifact_inputs(&self) -> HashSet<Register> {
        self.required_graph_inputs
            .iter()
            .cloned()
            .map(Register::GraphInput)
            .chain(self.weights.keys().cloned().map(Register::Weight))
            .chain(
                self.required_virtual_inputs
                    .iter()
                    .copied()
                    .map(Register::Virtual),
            )
            .collect()
    }

    fn required_outputs(&self) -> BTreeSet<usize> {
        self.shard
            .outputs
            .iter()
            .filter_map(|register| match register {
                Register::Virtual(index) => Some(*index),
                _ => None,
            })
            .collect()
    }
}

impl WitnessBackend for ZkieIsaCpuWitnessBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(
            ExecutionBackendId::parse("zkie-isa-cpu-v1").expect("static id is valid"),
            vec![
                ProofFlavorId::parse("halo2-kzg-bn256-shplonk-v1").expect("static flavor is valid")
            ],
            vec![CapabilityId::parse("deterministic-fixed-point-witness-v1")
                .expect("static capability is valid")],
        )
        .expect("static capabilities are valid")
    }

    fn estimate_resources(&self, job: WitnessJob) -> Result<ResourceRequest, BackendError> {
        self.validate_job(&job)?;
        ResourceRequest::new(1, self.limits.estimated_ram_bytes()?, 0, 0).map_err(|error| {
            BackendError::Resource {
                message: error.to_string(),
            }
        })
    }

    fn generate(&self, job: WitnessJob, output: PathBuf) -> Result<WitnessArtifact, BackendError> {
        self.validate_job(&job)?;
        let bytes = read_bounded(job.witness_input_path(), self.limits.input_bytes)?;
        let wire: CpuWitnessInputWire =
            serde_json::from_slice(&bytes).map_err(|error| BackendError::Serialization {
                format: "zkie-cpu-witness-input-json-v1".into(),
                message: error.to_string(),
            })?;
        if wire.schema_version != INPUT_SCHEMA_VERSION {
            return Err(BackendError::Serialization {
                format: "zkie-cpu-witness-input-json-v1".into(),
                message: format!("unsupported schema version {}", wire.schema_version),
            });
        }
        let supplied = CpuWitnessInputs::new(
            wire.graph_inputs
                .into_iter()
                .map(|(name, values)| (name, values.into_iter().map(I18::from_raw).collect()))
                .collect(),
            wire.virtual_inputs
                .into_iter()
                .map(|(index, values)| (index, values.into_iter().map(I18::from_raw).collect()))
                .collect(),
        );
        let artifact = self.execute(&supplied)?;
        let encoded = artifact.to_json()?;
        if encoded.len() > self.limits.artifact_bytes {
            return Err(BackendError::Resource {
                message: format!(
                    "CPU witness artifact is {} bytes, limit is {}",
                    encoded.len(),
                    self.limits.artifact_bytes
                ),
            });
        }
        let digest = write_exclusive_and_hash(&output, &encoded, self.limits.artifact_bytes)?;
        WitnessArtifact::new(output, digest, job.shard().clone(), self.circuit_digest())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CpuWitnessInputWire {
    schema_version: u32,
    graph_inputs: HashMap<String, Vec<i64>>,
    #[serde(default)]
    virtual_inputs: BTreeMap<usize, Vec<i64>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CpuWitnessArtifactWire {
    schema_version: u32,
    run_identity: RunIdentity,
    run_identity_digest: Digest32,
    shard_identity: ShardIdentity,
    circuit_digest: Digest32,
    inputs: Vec<RegisterValueWire>,
    virtuals: Vec<IndexedValueWire>,
    outputs: Vec<IndexedValueWire>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexedValueWire {
    index: usize,
    raw_values: Vec<i64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisterValueWire {
    register: RegisterWire,
    raw_values: Vec<i64>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RegisterWire {
    GraphInput { name: String },
    Weight { name: String },
    Virtual { index: usize },
}

impl RegisterWire {
    fn from_register(register: &Register) -> Self {
        match register {
            Register::GraphInput(name) => Self::GraphInput { name: name.clone() },
            Register::Weight(name) => Self::Weight { name: name.clone() },
            Register::Virtual(index) => Self::Virtual { index: *index },
        }
    }

    fn into_register(self) -> Register {
        match self {
            Self::GraphInput { name } => Register::GraphInput(name),
            Self::Weight { name } => Register::Weight(name),
            Self::Virtual { index } => Register::Virtual(index),
        }
    }

    fn sort_key(&self) -> (u8, &str, usize) {
        match self {
            Self::GraphInput { name } => (0, name, 0),
            Self::Weight { name } => (1, name, 0),
            Self::Virtual { index } => (2, "", *index),
        }
    }
}

impl CpuWitnessArtifact {
    fn to_json(&self) -> Result<Vec<u8>, BackendError> {
        validate_artifact_record_count(
            self.inputs.len(),
            self.virtuals.len(),
            self.outputs.len(),
            self.record_limit,
        )?;
        let mut inputs = self
            .inputs
            .iter()
            .map(|(register, values)| RegisterValueWire {
                register: RegisterWire::from_register(register),
                raw_values: values.iter().map(I18::raw).collect(),
            })
            .collect::<Vec<_>>();
        inputs.sort_unstable_by(|left, right| {
            left.register.sort_key().cmp(&right.register.sort_key())
        });
        let raw_indexed = |values: &BTreeMap<usize, Vec<I18>>| {
            values
                .iter()
                .map(|(index, tensor)| IndexedValueWire {
                    index: *index,
                    raw_values: tensor.iter().map(I18::raw).collect(),
                })
                .collect()
        };
        serde_json::to_vec(&CpuWitnessArtifactWire {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            run_identity: self.run_identity.clone(),
            run_identity_digest: self.run_identity_digest,
            shard_identity: self.shard_identity.clone(),
            circuit_digest: self.circuit_digest(),
            inputs,
            virtuals: raw_indexed(&self.virtuals),
            outputs: raw_indexed(&self.outputs),
        })
        .map_err(|error| BackendError::Serialization {
            format: "zkie-cpu-witness-artifact-json-v1".into(),
            message: error.to_string(),
        })
    }
}

fn expect_operands(
    instruction: usize,
    name: &str,
    operands: &[&[I18]],
    expected: usize,
) -> Result<(), BackendError> {
    if operands.len() == expected {
        Ok(())
    } else {
        shape_error(
            instruction,
            format!(
                "{name} expected {expected} operands, got {}",
                operands.len()
            ),
        )
    }
}

fn checked_len(instruction: usize, left: usize, right: usize) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| overflow(instruction, "shape element count"))
}

fn shape_error<T>(instruction: usize, message: String) -> Result<T, BackendError> {
    Err(BackendError::ShapeMismatch {
        instruction,
        message,
    })
}

fn overflow(instruction: usize, message: impl Into<String>) -> BackendError {
    BackendError::ArithmeticOverflow {
        instruction,
        message: message.into(),
    }
}

struct ProgramAnalysis {
    weights: HashMap<String, Vec<I18>>,
    required_graph_inputs: BTreeSet<String>,
    required_virtual_inputs: BTreeSet<usize>,
    rms_norm_tables: HashMap<(usize, u64), BTreeMap<i64, I18>>,
    artifact_record_count: usize,
}

fn analyze_program(
    program: &CompiledProgram,
    shard: &Shard,
    domains: &HashMap<(usize, u64), RsqrtDomain>,
    limits: &CpuWitnessLimits,
) -> Result<ProgramAnalysis, BackendError> {
    let instruction_count = shard.range.len();
    if instruction_count > limits.instruction_count {
        return Err(BackendError::Resource {
            message: format!(
                "CPU witness shard has {instruction_count} instructions, limit is {}",
                limits.instruction_count
            ),
        });
    }
    let mut weights = HashMap::new();
    let mut graph_inputs = BTreeSet::new();
    let mut virtual_inputs = BTreeSet::new();
    let mut domain_keys = BTreeSet::new();
    let declared_graph_inputs = program.graph_inputs.iter().collect::<HashSet<_>>();
    let mut operand_references = 0usize;

    for (index, compiled) in program.instructions[shard.range.clone()].iter().enumerate() {
        let absolute_index = shard.range.start + index;
        operand_references = operand_references
            .checked_add(compiled.inputs.len())
            .ok_or_else(|| BackendError::Resource {
                message: "CPU witness operand reference count overflow".into(),
            })?;
        if let Instruction::DotGeneral { batch_dims, .. } = &compiled.instruction {
            operand_references = operand_references
                .checked_add(batch_dims.len())
                .ok_or_else(|| BackendError::Resource {
                    message: "CPU witness operand metadata count overflow".into(),
                })?;
        }
        if operand_references > limits.operand_references {
            return Err(BackendError::Resource {
                message: format!(
                    "CPU witness shard has {operand_references} operand metadata entries, limit is {}",
                    limits.operand_references
                ),
            });
        }
        if let Instruction::DotGeneral { m, n, k, .. } = &compiled.instruction {
            if *m == 0 || *n == 0 || *k == 0 {
                return shape_error(
                    absolute_index,
                    "DotGeneral dimensions must be nonzero".into(),
                );
            }
            checked_len(absolute_index, *m, *k)?;
            checked_len(absolute_index, *k, *n)?;
            let output_len = checked_len(absolute_index, *m, *n)?;
            if output_len > limits.values_per_tensor {
                return Err(BackendError::Resource {
                    message: format!(
                        "DotGeneral output has {output_len} values, limit is {}",
                        limits.values_per_tensor
                    ),
                });
            }
        }
        if let Instruction::RmsNorm { dim, epsilon_milli } = compiled.instruction {
            if dim == 0 {
                return shape_error(absolute_index, "RmsNorm dim must be nonzero".into());
            }
            if dim > limits.values_per_tensor {
                return Err(BackendError::Resource {
                    message: format!("RmsNorm dim {dim} exceeds tensor value limit"),
                });
            }
            domain_keys.insert((dim, epsilon_milli));
        }
        for register in &compiled.inputs {
            match register {
                Register::GraphInput(name) => {
                    if !declared_graph_inputs.contains(name) {
                        return Err(BackendError::MissingRegister {
                            register: format!("GraphInput({name:?})"),
                        });
                    }
                    graph_inputs.insert(name.clone());
                }
                Register::Weight(name) => {
                    if !weights.contains_key(name) {
                        let values = program.weight_as_i18(name).map_err(|error| match error {
                            OnnxParseError::WeightNotFound(_) => BackendError::MissingRegister {
                                register: format!("Weight({name:?})"),
                            },
                            OnnxParseError::WeightOutOfRange { .. } => {
                                overflow(absolute_index, error.to_string())
                            }
                            _ => BackendError::InvalidJob {
                                message: format!("invalid weight {name:?}: {error}"),
                            },
                        })?;
                        if values.len() > limits.values_per_tensor {
                            return Err(BackendError::Resource {
                                message: format!("weight {name:?} exceeds tensor value limit"),
                            });
                        }
                        weights.insert(name.clone(), values);
                    }
                }
                Register::Virtual(producer) if *producer < shard.range.start => {
                    virtual_inputs.insert(*producer);
                }
                Register::Virtual(producer) if *producer >= absolute_index => {
                    return Err(BackendError::ForwardRegister {
                        instruction: absolute_index,
                        register: format!("Virtual({producer})"),
                    });
                }
                Register::Virtual(_) => {}
            }
        }
    }

    let declared_virtual_inputs = shard
        .inputs
        .iter()
        .filter_map(|register| match register {
            Register::Virtual(index) => Some(*index),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if virtual_inputs != declared_virtual_inputs {
        return Err(BackendError::InvalidJob {
            message: "shard input boundary does not exactly match external virtual uses".into(),
        });
    }
    let input_record_count = weights
        .len()
        .checked_add(graph_inputs.len())
        .and_then(|count| count.checked_add(virtual_inputs.len()))
        .ok_or_else(|| BackendError::Resource {
            message: "CPU witness input record count overflow".into(),
        })?;
    let artifact_record_count = validate_artifact_record_count(
        input_record_count,
        shard.range.len(),
        shard.outputs.len(),
        limits.tensor_count,
    )?;
    let weight_values = weights
        .values()
        .map(Vec::len)
        .try_fold(0usize, |sum, len| {
            sum.checked_add(len).ok_or_else(|| BackendError::Resource {
                message: "CPU witness weight value count overflow".into(),
            })
        })?;
    if weight_values > limits.total_artifact_values {
        return Err(BackendError::Resource {
            message: "CPU witness weights exceed total artifact value limit".into(),
        });
    }
    let configured_keys = domains.keys().copied().collect::<BTreeSet<_>>();
    if configured_keys != domain_keys {
        return Err(BackendError::InvalidJob {
            message: "RMSNorm domains do not exactly match instructions in shard".into(),
        });
    }
    let total_domain_points = domains.values().try_fold(0usize, |sum, domain| {
        let count = match domain {
            RsqrtDomain::Range { n, .. } => *n,
            RsqrtDomain::RawAnchors(points) => points.len(),
        };
        sum.checked_add(count)
            .ok_or_else(|| BackendError::Resource {
                message: "RMSNorm total domain point count overflow".into(),
            })
    })?;
    if total_domain_points > limits.domain_points {
        return Err(BackendError::Resource {
            message: format!(
                "RMSNorm domains have {total_domain_points} points, limit is {}",
                limits.domain_points
            ),
        });
    }
    let rms_norm_tables = domains
        .iter()
        .map(|(key, domain)| Ok((*key, materialize_domain(domain, limits.domain_points)?)))
        .collect::<Result<HashMap<_, _>, BackendError>>()?;
    Ok(ProgramAnalysis {
        weights,
        required_graph_inputs: graph_inputs,
        required_virtual_inputs: virtual_inputs,
        rms_norm_tables,
        artifact_record_count,
    })
}

fn validate_artifact_record_count(
    input_records: usize,
    virtual_records: usize,
    output_records: usize,
    limit: usize,
) -> Result<usize, BackendError> {
    let count = input_records
        .checked_add(virtual_records)
        .and_then(|count| count.checked_add(output_records))
        .ok_or_else(|| BackendError::Resource {
            message: "CPU witness artifact record count overflow".into(),
        })?;
    if count > limit {
        return Err(BackendError::Resource {
            message: format!(
                "CPU witness artifact has {count} records, configured limit is {limit}"
            ),
        });
    }
    Ok(count)
}

fn materialize_domain(
    domain: &RsqrtDomain,
    max_points: usize,
) -> Result<BTreeMap<i64, I18>, BackendError> {
    let point_count = match domain {
        RsqrtDomain::Range { n, .. } => *n,
        RsqrtDomain::RawAnchors(points) => points.len(),
    };
    if point_count == 0 {
        return Err(BackendError::InvalidJob {
            message: "RMSNorm lookup domain is empty".into(),
        });
    }
    if point_count > max_points {
        return Err(BackendError::Resource {
            message: format!("RMSNorm domain has {point_count} points, limit is {max_points}"),
        });
    }
    match domain {
        RsqrtDomain::Range { min, max, .. }
            if !min.is_finite()
                || !max.is_finite()
                || *min <= 0.0
                || min > max
                || I18::from_f64(*min).is_err()
                || I18::from_f64(*max).is_err()
                || I18::from_f64(rsqrt_f64(*min)).is_err() =>
        {
            return Err(BackendError::InvalidJob {
                message: "invalid RMSNorm range domain".into(),
            });
        }
        RsqrtDomain::RawAnchors(points)
            if points.iter().any(|raw| {
                *raw <= 0 || I18::from_f64(rsqrt_f64(I18::from_raw(*raw).to_f64())).is_err()
            }) =>
        {
            return Err(BackendError::InvalidJob {
                message: "invalid RMSNorm raw-anchor domain".into(),
            });
        }
        _ => {}
    }
    let (points, values) = match domain {
        RsqrtDomain::Range { min, max, n } => build_domain(rsqrt_f64, *min, *max, *n),
        RsqrtDomain::RawAnchors(points) => build_domain_from_raw(rsqrt_f64, points),
    };
    let mut table = BTreeMap::new();
    for (point, value) in points.into_iter().zip(values) {
        if let Some(previous) = table.insert(point.raw(), value) {
            if previous != value {
                return Err(BackendError::InvalidJob {
                    message: format!(
                        "RMSNorm domain maps raw input {} to conflicting outputs",
                        point.raw()
                    ),
                });
            }
        }
    }
    Ok(table)
}

fn validate_run_identity(run: &RunIdentity) -> Result<(), BackendError> {
    if run.proof_flavor.as_str() != SUPPORTED_PROOF_FLAVOR {
        return Err(BackendError::InvalidJob {
            message: format!("unsupported proof flavor {}", run.proof_flavor),
        });
    }
    if run.model_visibility != ModelVisibility::PublicModel {
        return Err(BackendError::InvalidJob {
            message: "private-model witness generation is not supported".into(),
        });
    }
    Ok(())
}

fn validate_tensor_collection<'a>(
    tensors: impl Iterator<Item = &'a Vec<I18>>,
    max_tensors: usize,
    max_per_tensor: usize,
    max_total: usize,
    label: &str,
) -> Result<(), BackendError> {
    let mut count = 0usize;
    let mut total = 0usize;
    for tensor in tensors {
        count = count.checked_add(1).ok_or_else(|| BackendError::Resource {
            message: format!("{label} tensor count overflow"),
        })?;
        if count > max_tensors || tensor.len() > max_per_tensor {
            return Err(BackendError::Resource {
                message: format!("{label} exceeds per-tensor/count limits"),
            });
        }
        total = total
            .checked_add(tensor.len())
            .ok_or_else(|| BackendError::Resource {
                message: format!("{label} value count overflow"),
            })?;
        if total > max_total {
            return Err(BackendError::Resource {
                message: format!("{label} exceeds total value limit"),
            });
        }
    }
    Ok(())
}

fn into_i18_indexed(
    raw: Vec<IndexedValueWire>,
    label: &str,
) -> Result<BTreeMap<usize, Vec<I18>>, BackendError> {
    let mut values = BTreeMap::new();
    for entry in raw {
        if values
            .insert(
                entry.index,
                entry.raw_values.into_iter().map(I18::from_raw).collect(),
            )
            .is_some()
        {
            return Err(BackendError::InvalidJob {
                message: format!("duplicate artifact {label} index {}", entry.index),
            });
        }
    }
    Ok(values)
}

fn derive_circuit_digest(
    program: &CompiledProgram,
    shard: &Shard,
    domains: &HashMap<(usize, u64), RsqrtDomain>,
    weights: &HashMap<String, Vec<I18>>,
    tables: &HashMap<(usize, u64), BTreeMap<i64, I18>>,
) -> Digest32 {
    let mut bytes = CIRCUIT_BINDING_VERSION.to_vec();
    encode_usize(&mut bytes, shard.id);
    encode_text(&mut bytes, &shard.name);
    encode_usize(&mut bytes, shard.range.start);
    encode_usize(&mut bytes, shard.range.end);
    encode_registers(&mut bytes, &shard.inputs);
    encode_registers(&mut bytes, &shard.outputs);
    for (index, compiled) in program.instructions[shard.range.clone()].iter().enumerate() {
        encode_usize(&mut bytes, shard.range.start + index);
        encode_instruction(&mut bytes, &compiled.instruction);
        encode_registers(&mut bytes, &compiled.inputs);
    }
    let mut weights = weights.iter().collect::<Vec<_>>();
    weights.sort_unstable_by_key(|(name, _)| name.as_str());
    encode_usize(&mut bytes, weights.len());
    for (name, values) in weights {
        encode_text(&mut bytes, name);
        encode_usize(&mut bytes, values.len());
        for value in values {
            bytes.extend_from_slice(&value.raw().to_le_bytes());
        }
    }
    let mut domains = domains.iter().collect::<Vec<_>>();
    domains.sort_unstable_by_key(|(key, _)| **key);
    encode_usize(&mut bytes, domains.len());
    for (key @ (dim, epsilon), domain) in domains {
        encode_usize(&mut bytes, *dim);
        bytes.extend_from_slice(&epsilon.to_le_bytes());
        match domain {
            RsqrtDomain::Range { min, max, n } => {
                bytes.push(0);
                bytes.extend_from_slice(&min.to_bits().to_le_bytes());
                bytes.extend_from_slice(&max.to_bits().to_le_bytes());
                encode_usize(&mut bytes, *n);
            }
            RsqrtDomain::RawAnchors(points) => {
                bytes.push(1);
                encode_usize(&mut bytes, points.len());
                for point in points {
                    bytes.extend_from_slice(&point.to_le_bytes());
                }
            }
        }
        let table = &tables[key];
        encode_usize(&mut bytes, table.len());
        for (input, output) in table {
            bytes.extend_from_slice(&input.to_le_bytes());
            bytes.extend_from_slice(&output.raw().to_le_bytes());
        }
    }
    Digest32::new(*blake3::hash(&bytes).as_bytes())
}

fn encode_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn encode_text(bytes: &mut Vec<u8>, value: &str) {
    encode_usize(bytes, value.len());
    bytes.extend_from_slice(value.as_bytes());
}

fn encode_registers(bytes: &mut Vec<u8>, registers: &[Register]) {
    encode_usize(bytes, registers.len());
    for register in registers {
        match register {
            Register::GraphInput(name) => {
                bytes.push(0);
                encode_text(bytes, name);
            }
            Register::Weight(name) => {
                bytes.push(1);
                encode_text(bytes, name);
            }
            Register::Virtual(index) => {
                bytes.push(2);
                encode_usize(bytes, *index);
            }
        }
    }
}

fn encode_instruction(bytes: &mut Vec<u8>, instruction: &Instruction) {
    match instruction {
        Instruction::DotGeneral {
            m,
            n,
            k,
            batch_dims,
            trans_a,
            trans_b,
        } => {
            bytes.push(0);
            for value in [*m, *n, *k] {
                encode_usize(bytes, value);
            }
            encode_usize(bytes, batch_dims.len());
            for value in batch_dims {
                encode_usize(bytes, *value);
            }
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
            bytes.push(5);
            bytes.push(match op {
                EltwiseOp::Add => 0,
                EltwiseOp::Mul => 1,
                EltwiseOp::Relu => 2,
            });
        }
        Instruction::Reduce { op, axis } => {
            bytes.push(6);
            bytes.push(match op {
                ReduceOp::Sum => 0,
                ReduceOp::Mean => 1,
            });
            encode_usize(bytes, *axis);
        }
        Instruction::EmbedLookup {
            table_size,
            embed_dim,
        } => {
            bytes.push(7);
            encode_usize(bytes, *table_size);
            encode_usize(bytes, *embed_dim);
        }
        Instruction::PatchEmbed {
            patch_len,
            embed_dim,
        } => {
            bytes.push(8);
            encode_usize(bytes, *patch_len);
            encode_usize(bytes, *embed_dim);
        }
    }
}

fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>, BackendError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options
        .open(path)
        .map_err(|error| io_error("open_regular", path, error))?;
    let metadata = file
        .metadata()
        .map_err(|error| io_error("metadata", path, error))?;
    if !metadata.file_type().is_file() {
        return Err(BackendError::Io {
            operation: "open_regular".into(),
            path: path.to_path_buf(),
            kind: "InvalidInput".into(),
            message: "bounded input must be a regular file".into(),
        });
    }
    let length = metadata.len();
    if length > max_bytes as u64 {
        return Err(BackendError::Resource {
            message: format!("{} is {length} bytes, limit is {max_bytes}", path.display()),
        });
    }
    let mut bytes = Vec::with_capacity(length as usize);
    Read::by_ref(&mut file)
        .take((max_bytes as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| io_error("read", path, error))?;
    if bytes.len() > max_bytes {
        return Err(BackendError::Resource {
            message: format!("{} exceeds {max_bytes} bytes", path.display()),
        });
    }
    Ok(bytes)
}

fn write_exclusive_and_hash(
    path: &Path,
    bytes: &[u8],
    max_bytes: usize,
) -> Result<Digest32, BackendError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path.file_name().ok_or_else(|| BackendError::Io {
        operation: "create_temp".into(),
        path: path.to_path_buf(),
        kind: "InvalidInput".into(),
        message: "witness output path has no file name".into(),
    })?;
    let (temp_path, mut file) = create_unique_temp(parent, file_name, path)?;
    let mut temp_guard = TempFileGuard(Some(temp_path.clone()));
    file.write_all(bytes)
        .map_err(|error| io_error("write", &temp_path, error))?;
    file.flush()
        .map_err(|error| io_error("flush", &temp_path, error))?;
    file.sync_all()
        .map_err(|error| io_error("sync_all", &temp_path, error))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| io_error("seek", &temp_path, error))?;
    let mut persisted = Vec::with_capacity(bytes.len());
    Read::by_ref(&mut file)
        .take((max_bytes as u64).saturating_add(1))
        .read_to_end(&mut persisted)
        .map_err(|error| io_error("read_back", &temp_path, error))?;
    if persisted.len() > max_bytes || persisted != bytes {
        return Err(BackendError::Io {
            operation: "verify_write".into(),
            path: temp_path,
            kind: "DataIntegrity".into(),
            message: "persisted witness bytes differ from encoded artifact".into(),
        });
    }
    let digest = Digest32::new(*blake3::hash(&persisted).as_bytes());
    drop(file);
    fs::hard_link(&temp_path, path).map_err(|error| io_error("publish_no_clobber", path, error))?;
    if let Err(error) = fs::remove_file(&temp_path) {
        let _ = fs::remove_file(path);
        return Err(io_error("remove_temp", &temp_path, error));
    }
    temp_guard.0 = None;
    if let Err(error) = sync_parent_directory(parent) {
        let _ = fs::remove_file(path);
        return Err(io_error("sync_parent", parent, error));
    }
    Ok(digest)
}

struct TempFileGuard(Option<PathBuf>);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_file(path);
        }
    }
}

fn create_unique_temp(
    parent: &Path,
    file_name: &std::ffi::OsStr,
    output_path: &Path,
) -> Result<(PathBuf, File), BackendError> {
    for _ in 0..32 {
        let mut random = [0u8; 16];
        OsRng
            .try_fill_bytes(&mut random)
            .map_err(|error| BackendError::Io {
                operation: "random_temp_name".into(),
                path: output_path.to_path_buf(),
                kind: "Other".into(),
                message: error.to_string(),
            })?;
        let mut suffix = String::with_capacity(32);
        for byte in random {
            use std::fmt::Write as _;
            let _ = write!(suffix, "{byte:02x}");
        }
        let temp_path = parent.join(format!(
            ".{}.zkie-tmp-{suffix}",
            file_name.to_string_lossy()
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&temp_path) {
            Ok(file) => return Ok((temp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_error("create_temp", &temp_path, error)),
        }
    }
    Err(BackendError::Io {
        operation: "create_temp".into(),
        path: output_path.to_path_buf(),
        kind: "AlreadyExists".into(),
        message: "could not reserve a unique witness temporary file".into(),
    })
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> std::io::Result<()> {
    match File::open(parent)?.sync_all() {
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::Unsupported | std::io::ErrorKind::InvalidInput
            ) =>
        {
            Ok(())
        }
        result => result,
    }
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> std::io::Result<()> {
    Ok(())
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> BackendError {
    BackendError::Io {
        operation: operation.into(),
        path: path.to_path_buf(),
        kind: format!("{:?}", error.kind()),
        message: error.to_string(),
    }
}
