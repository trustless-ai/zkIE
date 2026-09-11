//! Deterministic resource-targeted partition planning.

use std::collections::HashSet;
use std::fmt;
use std::ops::Range;

use zkie_types::Digest32;

use super::model::{build_dag, BuildDagError, Dag, EdgeKind, ShardSpec};
use crate::graph_compiler::{CompiledInstruction, CompiledProgram, Register};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstructionEstimate {
    ram_bytes: u64,
    required_k: u32,
}

impl InstructionEstimate {
    pub fn new(ram_bytes: u64, required_k: u32) -> Result<Self, PartitionError> {
        Ok(Self {
            ram_bytes,
            required_k,
        })
    }
    pub fn ram_bytes(&self) -> u64 {
        self.ram_bytes
    }
    pub fn required_k(&self) -> u32 {
        self.required_k
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardEstimate {
    ram_bytes: u64,
    required_k: u32,
}

impl ShardEstimate {
    pub fn new(ram_bytes: u64, required_k: u32) -> Result<Self, PartitionError> {
        Ok(Self {
            ram_bytes,
            required_k,
        })
    }
    pub fn ram_bytes(&self) -> u64 {
        self.ram_bytes
    }
    pub fn required_k(&self) -> u32 {
        self.required_k
    }
}

pub trait InstructionCostModel {
    fn identity(&self) -> &str;
    fn version(&self) -> u32;
    fn estimate_instruction(
        &self,
        instruction: &CompiledInstruction,
    ) -> Result<InstructionEstimate, PartitionError>;
    fn estimate_shard(
        &self,
        instructions: &[CompiledInstruction],
    ) -> Result<ShardEstimate, PartitionError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryHint {
    boundary: usize,
    priority: u32,
    label: String,
}

impl BoundaryHint {
    pub fn new(
        boundary: usize,
        priority: u32,
        label: impl Into<String>,
    ) -> Result<Self, PartitionError> {
        let label = label.into();
        if boundary == 0 || label.is_empty() {
            return Err(PartitionError::InvalidHint {
                message: "hint boundary must be nonzero and label nonempty".into(),
            });
        }
        Ok(Self {
            boundary,
            priority,
            label,
        })
    }
    pub fn boundary(&self) -> usize {
        self.boundary
    }
    pub fn priority(&self) -> u32 {
        self.priority
    }
    pub fn label(&self) -> &str {
        &self.label
    }
}

pub struct PartitionRequest<M> {
    target_shard_ram_bytes: u64,
    max_k: u32,
    model_digest: Digest32,
    compiler_version: String,
    cost_model: M,
    boundary_hints: Vec<BoundaryHint>,
    boundary_layout_digest: Option<Digest32>,
}

impl<M> PartitionRequest<M> {
    pub fn new(
        target_shard_ram_bytes: u64,
        max_k: u32,
        model_digest: Digest32,
        compiler_version: impl Into<String>,
        cost_model: M,
    ) -> Result<Self, PartitionError> {
        let compiler_version = compiler_version.into();
        if target_shard_ram_bytes == 0 || max_k == 0 {
            return Err(PartitionError::ZeroBudget);
        }
        if compiler_version.is_empty() {
            return Err(PartitionError::InvalidRequest {
                message: "compiler version must be nonempty".into(),
            });
        }
        Ok(Self {
            target_shard_ram_bytes,
            max_k,
            model_digest,
            compiler_version,
            cost_model,
            boundary_hints: Vec::new(),
            boundary_layout_digest: None,
        })
    }

    pub fn with_boundary_hints(
        mut self,
        mut hints: Vec<BoundaryHint>,
    ) -> Result<Self, PartitionError> {
        let mut positions = HashSet::new();
        if hints.iter().any(|hint| !positions.insert(hint.boundary)) {
            return Err(PartitionError::InvalidHint {
                message: "duplicate hint boundary".into(),
            });
        }
        hints.sort_unstable_by(|left, right| {
            left.boundary
                .cmp(&right.boundary)
                .then_with(|| left.label.cmp(&right.label))
        });
        self.boundary_hints = hints;
        Ok(self)
    }

    pub fn with_boundary_layout_digest(mut self, digest: Digest32) -> Self {
        self.boundary_layout_digest = Some(digest);
        self
    }

    pub fn target_shard_ram_bytes(&self) -> u64 {
        self.target_shard_ram_bytes
    }
    pub fn max_k(&self) -> u32 {
        self.max_k
    }
    pub fn model_digest(&self) -> Digest32 {
        self.model_digest
    }
    pub fn compiler_version(&self) -> &str {
        &self.compiler_version
    }
    pub fn cost_model(&self) -> &M {
        &self.cost_model
    }
    pub fn boundary_hints(&self) -> &[BoundaryHint] {
        &self.boundary_hints
    }
    pub fn boundary_layout_digest(&self) -> Option<Digest32> {
        self.boundary_layout_digest
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedShard {
    id: u64,
    name: String,
    range: Range<usize>,
    estimate: ShardEstimate,
    instruction_estimates: Vec<InstructionEstimate>,
}

impl PlannedShard {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn estimate(&self) -> ShardEstimate {
        self.estimate
    }
    pub fn instruction_estimates(&self) -> &[InstructionEstimate] {
        &self.instruction_estimates
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PartitionPlan {
    shards: Vec<PlannedShard>,
    dag: Dag,
    model_digest: Digest32,
    boundary_layout_digest: Option<Digest32>,
    digest: Digest32,
    program_snapshot: CompiledProgram,
}

impl PartitionPlan {
    pub fn shards(&self) -> &[PlannedShard] {
        &self.shards
    }
    pub fn dag(&self) -> &Dag {
        &self.dag
    }
    pub fn digest(&self) -> Digest32 {
        self.digest
    }
    pub fn model_digest(&self) -> Digest32 {
        self.model_digest
    }
    pub fn boundary_layout_digest(&self) -> Option<Digest32> {
        self.boundary_layout_digest
    }
    pub(crate) fn matches_program(&self, program: &CompiledProgram) -> bool {
        &self.program_snapshot == program
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartitionError {
    ZeroBudget,
    EmptyProgram,
    EmptyPartition,
    InvalidRequest {
        message: String,
    },
    InvalidHint {
        message: String,
    },
    EstimateOverflow,
    UnsplittableInstruction {
        instruction: usize,
        estimate: InstructionEstimate,
    },
    InvalidVirtualReference {
        consumer_instruction: usize,
        register: Register,
    },
    Dag {
        message: String,
    },
    InconsistentDag {
        message: String,
    },
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PartitionError {}

pub struct PartitionPlanner;

impl PartitionPlanner {
    pub fn plan<M: InstructionCostModel>(
        program: &CompiledProgram,
        request: &PartitionRequest<M>,
    ) -> Result<PartitionPlan, PartitionError> {
        if program.instructions.is_empty() {
            return Err(PartitionError::EmptyProgram);
        }
        validate_virtual_references(program)?;
        if request.cost_model.identity().is_empty() {
            return Err(PartitionError::InvalidRequest {
                message: "cost model identity must be nonempty".into(),
            });
        }
        if request
            .boundary_hints
            .iter()
            .any(|hint| hint.boundary >= program.instructions.len())
        {
            return Err(PartitionError::InvalidHint {
                message: "hint boundary is outside the program".into(),
            });
        }

        let instruction_estimates = program
            .instructions
            .iter()
            .map(|instruction| request.cost_model.estimate_instruction(instruction))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ranges = Vec::new();
        let mut start = 0usize;
        while start < program.instructions.len() {
            let single = request
                .cost_model
                .estimate_shard(&program.instructions[start..start + 1])?;
            if range_exceeds(&instruction_estimates[start..start + 1], single, request)? {
                return Err(PartitionError::UnsplittableInstruction {
                    instruction: start,
                    estimate: instruction_estimates[start],
                });
            }
            let mut accepted_estimates = vec![(start + 1, single)];
            let mut end = start + 1;
            while end < program.instructions.len() {
                let candidate = request
                    .cost_model
                    .estimate_shard(&program.instructions[start..end + 1])?;
                if range_exceeds(&instruction_estimates[start..end + 1], candidate, request)? {
                    break;
                }
                accepted_estimates.push((end + 1, candidate));
                end += 1;
            }
            if end < program.instructions.len() {
                if let Some(hint) = best_hint(&request.boundary_hints, start, end) {
                    end = hint.boundary;
                }
            }
            if end <= start {
                return Err(PartitionError::EmptyPartition);
            }
            let estimate = accepted_estimates
                .iter()
                .find(|(boundary, _)| *boundary == end)
                .map(|(_, estimate)| *estimate)
                .ok_or_else(|| PartitionError::InconsistentDag {
                    message: "selected boundary lacks an accepted estimate".into(),
                })?;
            ranges.push((start..end, estimate));
            start = end;
        }

        let specs = ranges
            .iter()
            .enumerate()
            .map(|(id, (range, _))| ShardSpec {
                name: format!("shard-{id}"),
                range: range.clone(),
            })
            .collect::<Vec<_>>();
        let dag = build_dag(program, &specs).map_err(map_dag_error)?;
        validate_dag(&dag, program.instructions.len())?;
        let shards = ranges
            .into_iter()
            .enumerate()
            .map(|(id, (range, estimate))| PlannedShard {
                id: id as u64,
                name: format!("shard-{id}"),
                instruction_estimates: instruction_estimates[range.clone()].to_vec(),
                range,
                estimate,
            })
            .collect::<Vec<_>>();
        if shards.is_empty() {
            return Err(PartitionError::EmptyPartition);
        }
        let digest = plan_digest(request, &shards, &dag);
        Ok(PartitionPlan {
            shards,
            dag,
            model_digest: request.model_digest,
            boundary_layout_digest: request.boundary_layout_digest,
            digest,
            program_snapshot: program.clone(),
        })
    }
}

fn range_exceeds<M>(
    instructions: &[InstructionEstimate],
    shard: ShardEstimate,
    request: &PartitionRequest<M>,
) -> Result<bool, PartitionError> {
    let instruction_ram = instructions.iter().try_fold(0_u64, |sum, estimate| {
        sum.checked_add(estimate.ram_bytes)
            .ok_or(PartitionError::EstimateOverflow)
    })?;
    let instruction_k = instructions
        .iter()
        .map(|estimate| estimate.required_k)
        .max()
        .unwrap_or(0);
    Ok(instruction_ram > request.target_shard_ram_bytes
        || instruction_k > request.max_k
        || shard.ram_bytes > request.target_shard_ram_bytes
        || shard.required_k > request.max_k)
}

fn best_hint(hints: &[BoundaryHint], start: usize, end: usize) -> Option<&BoundaryHint> {
    hints
        .iter()
        .filter(|hint| hint.boundary > start && hint.boundary <= end)
        .max_by(|left, right| {
            left.priority
                .cmp(&right.priority)
                .then_with(|| left.boundary.cmp(&right.boundary))
                .then_with(|| right.label.cmp(&left.label))
        })
}

fn validate_virtual_references(program: &CompiledProgram) -> Result<(), PartitionError> {
    for (consumer, instruction) in program.instructions.iter().enumerate() {
        for register in &instruction.inputs {
            if let Register::Virtual(producer) = register {
                if *producer >= consumer {
                    return Err(PartitionError::InvalidVirtualReference {
                        consumer_instruction: consumer,
                        register: register.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn map_dag_error(error: BuildDagError) -> PartitionError {
    match error {
        BuildDagError::UnknownRegister {
            consumer_instruction,
            register,
        }
        | BuildDagError::ForwardRegister {
            consumer_instruction,
            register,
        } => PartitionError::InvalidVirtualReference {
            consumer_instruction,
            register,
        },
        other => PartitionError::Dag {
            message: other.to_string(),
        },
    }
}

fn validate_dag(dag: &Dag, instruction_count: usize) -> Result<(), PartitionError> {
    let covered = dag
        .shards
        .iter()
        .map(|shard| shard.range.len())
        .sum::<usize>();
    if covered != instruction_count
        || dag
            .edges
            .iter()
            .any(|edge| edge.producer >= edge.consumer || edge.consumer >= dag.shards.len())
    {
        return Err(PartitionError::InconsistentDag {
            message: "derived DAG coverage or direction is invalid".into(),
        });
    }
    Ok(())
}

fn plan_digest<M: InstructionCostModel>(
    request: &PartitionRequest<M>,
    shards: &[PlannedShard],
    dag: &Dag,
) -> Digest32 {
    let mut bytes = b"zkie.partition-plan.v1\0".to_vec();
    bytes.extend_from_slice(request.model_digest.as_bytes());
    encode_text(&mut bytes, &request.compiler_version);
    encode_u64(&mut bytes, request.target_shard_ram_bytes);
    encode_u32(&mut bytes, request.max_k);
    encode_text(&mut bytes, request.cost_model.identity());
    encode_u32(&mut bytes, request.cost_model.version());
    match request.boundary_layout_digest {
        Some(digest) => {
            bytes.push(1);
            bytes.extend_from_slice(digest.as_bytes());
        }
        None => bytes.push(0),
    }
    encode_u64(&mut bytes, request.boundary_hints.len() as u64);
    for hint in &request.boundary_hints {
        encode_u64(&mut bytes, hint.boundary as u64);
        encode_u32(&mut bytes, hint.priority);
        encode_text(&mut bytes, &hint.label);
    }
    encode_u64(&mut bytes, shards.len() as u64);
    for shard in shards {
        encode_u64(&mut bytes, shard.id);
        encode_text(&mut bytes, &shard.name);
        encode_u64(&mut bytes, shard.range.start as u64);
        encode_u64(&mut bytes, shard.range.end as u64);
        encode_estimate(&mut bytes, shard.estimate);
        encode_u64(&mut bytes, shard.instruction_estimates.len() as u64);
        for estimate in &shard.instruction_estimates {
            encode_u64(&mut bytes, estimate.ram_bytes);
            encode_u32(&mut bytes, estimate.required_k);
        }
    }
    encode_u64(&mut bytes, dag.edges.len() as u64);
    let mut edges = dag.edges.iter().collect::<Vec<_>>();
    edges.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    for edge in edges {
        encode_text(&mut bytes, edge.id.as_str());
        encode_u64(&mut bytes, edge.producer as u64);
        encode_u64(&mut bytes, edge.consumer as u64);
        encode_register(&mut bytes, &edge.register);
        bytes.push(match edge.kind {
            EdgeKind::Sequential => 0,
            EdgeKind::Broadcast => 1,
        });
    }
    Digest32::new(*blake3::hash(&bytes).as_bytes())
}

fn encode_estimate(bytes: &mut Vec<u8>, estimate: ShardEstimate) {
    encode_u64(bytes, estimate.ram_bytes);
    encode_u32(bytes, estimate.required_k);
}
fn encode_register(bytes: &mut Vec<u8>, register: &Register) {
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
            encode_u64(bytes, *index as u64);
        }
    }
}
fn encode_text(bytes: &mut Vec<u8>, value: &str) {
    encode_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}
fn encode_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn encode_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
