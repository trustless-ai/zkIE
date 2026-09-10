//! Deterministic native aggregation plans and verified frontier claims.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroU8;

use halo2_proofs::halo2curves::bn256::Fr;
use halo2_proofs::halo2curves::ff::PrimeField;
use serde::{Deserialize, Serialize};
use zkie_compiler::dag::{Dag, PartitionPlan};
use zkie_compiler::graph_compiler::Register;
use zkie_core::chips::poseidon_boundary::{BoundaryDescriptor, BoundaryRole};
use zkie_types::{Digest32, ExecutionBackendId, ModelVisibility, ProofFlavorId, RunIdentity};

use crate::{LeafStatement, VerifiedProof, LEAF_STATEMENT_SCHEMA_VERSION};

const MIN_FAN_IN: u8 = 2;
const MAX_FAN_IN: u8 = 16;
const MAX_LEAVES: usize = 1 << 16;
const MAX_EDGES: usize = 1 << 20;
const MAX_LEVELS: usize = 64;
const MAX_MANIFEST_BYTES: usize = 1 << 20;
const MAX_PUBLIC_BOUNDARY_DECLARATIONS: usize = 1 << 12;
const MAX_PUBLIC_BOUNDARY_WIRE_BYTES: usize = MAX_MANIFEST_BYTES - (4 << 10);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AggregationChildId {
    Leaf(u64),
    Node { level: u32, index: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregationNode {
    id: AggregationChildId,
    children: Vec<AggregationChildId>,
    plan_digest: Digest32,
}

impl AggregationNode {
    pub fn id(&self) -> &AggregationChildId {
        &self.id
    }
    pub fn children(&self) -> &[AggregationChildId] {
        &self.children
    }
    pub fn actual_arity(&self) -> u8 {
        self.children.len() as u8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregationPlan {
    partition_digest: Digest32,
    fan_in: NonZeroU8,
    leaf_count: usize,
    levels: Vec<Vec<AggregationNode>>,
    root: AggregationChildId,
    leaf_expectations: Option<LeafExpectationSet>,
    digest: Digest32,
}

impl AggregationPlan {
    pub fn partition_digest(&self) -> Digest32 {
        self.partition_digest
    }
    pub fn fan_in(&self) -> NonZeroU8 {
        self.fan_in
    }
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }
    pub fn levels(&self) -> &[Vec<AggregationNode>] {
        &self.levels
    }
    pub fn root(&self) -> &AggregationChildId {
        &self.root
    }
    pub fn digest(&self) -> Digest32 {
        self.digest
    }

    /// Binds the aggregation topology to the trusted circuit and key identity
    /// admitted for every leaf. Unbound plans can be inspected, but cannot
    /// admit proofs or produce manifests.
    pub fn bind_leaf_expectations(
        mut self,
        expectations: &LeafExpectationSet,
    ) -> Result<Self, AggregationError> {
        if expectations.partition_digest != self.partition_digest
            || expectations.leaves.len() != self.leaf_count
        {
            return Err(AggregationError::InvalidLeafExpectations);
        }
        self.leaf_expectations = Some(expectations.clone());
        self.digest = aggregation_plan_digest(
            self.partition_digest,
            self.fan_in,
            self.leaf_count,
            &self.levels,
            &self.root,
            Some(expectations.digest),
        );
        for level in &mut self.levels {
            for node in level {
                node.plan_digest = self.digest;
            }
        }
        Ok(self)
    }

    pub fn leaf_expectation_set_digest(&self) -> Option<Digest32> {
        self.leaf_expectations.as_ref().map(|set| set.digest)
    }

    fn trusted_leaf_expectations<'a>(
        &'a self,
        partition: &PartitionPlan,
    ) -> Result<&'a LeafExpectationSet, AggregationError> {
        let expectations = self
            .leaf_expectations
            .as_ref()
            .ok_or(AggregationError::InvalidLeafExpectations)?;
        let expected_plan_digest = aggregation_plan_digest(
            self.partition_digest,
            self.fan_in,
            self.leaf_count,
            &self.levels,
            &self.root,
            Some(expectations.digest),
        );
        if self.partition_digest != partition.digest()
            || expectations.partition_digest != partition.digest()
            || expectations.leaves.len() != partition.shards().len()
            || leaf_expectation_set_digest(expectations.partition_digest, &expectations.leaves)
                != expectations.digest
            || expected_plan_digest != self.digest
        {
            return Err(AggregationError::InvalidLeafExpectations);
        }
        Ok(expectations)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafExpectation {
    shard_id: u64,
    shard_name: String,
    circuit_digest: Digest32,
    verification_key_digest: Digest32,
    execution_backend: ExecutionBackendId,
    proof_flavor: ProofFlavorId,
    public_inputs: Vec<BoundaryDescriptor>,
    public_outputs: Vec<BoundaryDescriptor>,
}

impl LeafExpectation {
    pub fn new(
        shard_id: u64,
        shard_name: String,
        circuit_digest: Digest32,
        verification_key_digest: Digest32,
        execution_backend: ExecutionBackendId,
        proof_flavor: ProofFlavorId,
    ) -> Result<Self, AggregationError> {
        if shard_name.is_empty() || !shard_name.is_ascii() || shard_name.len() > 4096 {
            return Err(AggregationError::InvalidLeafExpectations);
        }
        Ok(Self {
            shard_id,
            shard_name,
            circuit_digest,
            verification_key_digest,
            execution_backend,
            proof_flavor,
            public_inputs: Vec::new(),
            public_outputs: Vec::new(),
        })
    }

    /// Declares the graph-level public boundaries that this trusted leaf must
    /// expose. Their values remain proof-bound; the descriptors are fixed by
    /// the operator-controlled expectation set.
    pub fn with_public_io(
        mut self,
        public_inputs: Vec<BoundaryDescriptor>,
        public_outputs: Vec<BoundaryDescriptor>,
    ) -> Result<Self, AggregationError> {
        if public_inputs
            .len()
            .checked_add(public_outputs.len())
            .is_none_or(|count| count > MAX_PUBLIC_BOUNDARY_DECLARATIONS)
            || public_inputs.iter().any(|descriptor| {
                descriptor.validate().is_err()
                    || descriptor.role() != BoundaryRole::Input
                    || !descriptor.register_id().starts_with("graph-input:")
                    || !descriptor.edge_ids().is_empty()
                    || !descriptor.graph_output_names().is_empty()
            })
            || public_outputs.iter().any(|descriptor| {
                descriptor.validate().is_err()
                    || descriptor.role() != BoundaryRole::Output
                    || descriptor.graph_output_names().is_empty()
            })
        {
            return Err(AggregationError::InvalidLeafExpectations);
        }
        let input_ids = public_inputs
            .iter()
            .map(|descriptor| descriptor.register_id())
            .collect::<BTreeSet<_>>();
        let output_names = public_outputs
            .iter()
            .flat_map(|descriptor| descriptor.graph_output_names())
            .collect::<BTreeSet<_>>();
        if input_ids.len() != public_inputs.len()
            || output_names.len()
                != public_outputs
                    .iter()
                    .map(|descriptor| descriptor.graph_output_names().len())
                    .sum::<usize>()
        {
            return Err(AggregationError::InvalidLeafExpectations);
        }
        self.public_inputs = public_inputs;
        self.public_outputs = public_outputs;
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafExpectationSet {
    partition_digest: Digest32,
    leaves: BTreeMap<u64, LeafExpectation>,
    digest: Digest32,
}

impl LeafExpectationSet {
    pub fn new(
        partition: &PartitionPlan,
        expectations: Vec<LeafExpectation>,
    ) -> Result<Self, AggregationError> {
        if expectations.len() != partition.shards().len() || expectations.len() > MAX_LEAVES {
            return Err(AggregationError::InvalidLeafExpectations);
        }
        let public_boundary_budget =
            expectations
                .iter()
                .try_fold((0_usize, 0_usize), |(count, bytes), expectation| {
                    let mut boundaries = expectation
                        .public_inputs
                        .iter()
                        .chain(&expectation.public_outputs);
                    boundaries.try_fold((count, bytes), |(count, bytes), descriptor| {
                        let count = count.checked_add(1)?;
                        let bytes =
                            bytes.checked_add(public_boundary_wire_upper_bound(descriptor)?)?;
                        (count <= MAX_PUBLIC_BOUNDARY_DECLARATIONS
                            && bytes <= MAX_PUBLIC_BOUNDARY_WIRE_BYTES)
                            .then_some((count, bytes))
                    })
                });
        if public_boundary_budget.is_none() {
            return Err(AggregationError::InvalidLeafExpectations);
        }
        let leaves = expectations
            .into_iter()
            .map(|expectation| (expectation.shard_id, expectation))
            .collect::<BTreeMap<_, _>>();
        if leaves.len() != partition.shards().len()
            || partition.shards().iter().any(|shard| {
                leaves
                    .get(&shard.id())
                    .is_none_or(|expected| expected.shard_name != shard.name())
            })
        {
            return Err(AggregationError::InvalidLeafExpectations);
        }
        let partition_digest = partition.digest();
        let digest = leaf_expectation_set_digest(partition_digest, &leaves);
        Ok(Self {
            partition_digest,
            leaves,
            digest,
        })
    }

    pub fn digest(&self) -> Digest32 {
        self.digest
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AggregationError {
    InvalidFanIn { fan_in: u8 },
    InvalidPartition,
    ResourceLimit,
    InvalidLeafExpectations,
    UnsupportedModelVisibility,
    LeafIdentityMismatch { shard: u64 },
    MissingBoundary { edge: String, shard: u64 },
    ExtraBoundary { edge: String, shard: u64 },
    EndpointMismatch { edge: String, shard: u64 },
    BoundaryCommitmentMismatch { node: String, edge: String },
    DuplicateEdgeClosure { node: String, edge: String },
    ChildOrderMismatch { node: String },
    OverlappingShard { node: String, shard: u64 },
    MissingLeaf { shard: u64 },
    OpenInternalEdge { edge: String },
    AggregationIdentityMismatch,
    RecursiveProofUnavailable,
    MalformedStatement { shard: u64 },
    MalformedManifest,
}

impl std::fmt::Display for AggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for AggregationError {}

pub fn plan_aggregation(
    partition: &PartitionPlan,
    fan_in: NonZeroU8,
) -> Result<AggregationPlan, AggregationError> {
    let fan = fan_in.get();
    if !(MIN_FAN_IN..=MAX_FAN_IN).contains(&fan) {
        return Err(AggregationError::InvalidFanIn { fan_in: fan });
    }
    let shards = partition.shards();
    if shards.is_empty()
        || shards.len() > MAX_LEAVES
        || partition.dag().edges.len() > MAX_EDGES
        || shards
            .iter()
            .enumerate()
            .any(|(index, shard)| shard.id() != index as u64)
        || partition
            .dag()
            .edges
            .iter()
            .any(|edge| edge.producer >= edge.consumer || edge.consumer >= shards.len())
    {
        return Err(AggregationError::InvalidPartition);
    }
    let mut current = shards
        .iter()
        .map(|shard| AggregationChildId::Leaf(shard.id()))
        .collect::<Vec<_>>();
    let mut levels = Vec::new();
    while current.len() > 1 {
        if levels.len() >= MAX_LEVELS {
            return Err(AggregationError::ResourceLimit);
        }
        let level_index =
            u32::try_from(levels.len()).map_err(|_| AggregationError::ResourceLimit)?;
        let mut level = Vec::new();
        let mut next = Vec::new();
        for children in current.chunks(fan as usize) {
            if children.len() == 1 {
                next.push(children[0].clone());
                continue;
            }
            let index = u32::try_from(level.len()).map_err(|_| AggregationError::ResourceLimit)?;
            let id = AggregationChildId::Node {
                level: level_index,
                index,
            };
            level.push(AggregationNode {
                id: id.clone(),
                children: children.to_vec(),
                plan_digest: Digest32::default(),
            });
            next.push(id);
        }
        if level.is_empty() || next.len() >= current.len() {
            return Err(AggregationError::InvalidPartition);
        }
        levels.push(level);
        current = next;
    }
    let root = current.pop().ok_or(AggregationError::InvalidPartition)?;
    let digest = aggregation_plan_digest(
        partition.digest(),
        fan_in,
        shards.len(),
        &levels,
        &root,
        None,
    );
    for level in &mut levels {
        for node in level {
            node.plan_digest = digest;
        }
    }
    Ok(AggregationPlan {
        partition_digest: partition.digest(),
        fan_in,
        leaf_count: shards.len(),
        levels,
        root,
        leaf_expectations: None,
        digest,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EdgeFrontier {
    producer: Option<Fr>,
    consumer: Option<Fr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LeafBinding {
    proof_digest: Digest32,
    statement_digest: Digest32,
    circuit_digest: Digest32,
    verification_key_digest: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PublicBoundaryBinding {
    descriptor: BoundaryDescriptor,
    commitment: Fr,
}

/// A request-local index built once from the trusted partition DAG.
///
/// Keeping this separate from serialized claims prevents attacker-controlled
/// manifests from supplying lookup data while avoiding repeated whole-DAG
/// scans for every admitted leaf and aggregation node.
struct DagIndex<'a> {
    dag: &'a Dag,
    edges_by_id: HashMap<&'a str, usize>,
    incident_edges: Vec<Vec<usize>>,
}

impl<'a> DagIndex<'a> {
    fn new(dag: &'a Dag, shard_count: usize) -> Result<Self, AggregationError> {
        if shard_count == 0 || shard_count > MAX_LEAVES || dag.edges.len() > MAX_EDGES {
            return Err(AggregationError::ResourceLimit);
        }
        let mut edges_by_id = HashMap::with_capacity(dag.edges.len());
        let mut incident_edges = vec![Vec::new(); shard_count];
        for (edge_index, edge) in dag.edges.iter().enumerate() {
            if edge.producer >= shard_count
                || edge.consumer >= shard_count
                || edge.producer >= edge.consumer
                || edges_by_id.insert(edge.id.as_str(), edge_index).is_some()
            {
                return Err(AggregationError::InvalidPartition);
            }
            incident_edges[edge.producer].push(edge_index);
            incident_edges[edge.consumer].push(edge_index);
        }
        Ok(Self {
            dag,
            edges_by_id,
            incident_edges,
        })
    }

    fn edge(&self, edge_id: &str) -> Option<&'a zkie_compiler::dag::Edge> {
        self.edges_by_id
            .get(edge_id)
            .map(|index| &self.dag.edges[*index])
    }

    fn incident(&self, shard: usize) -> impl Iterator<Item = &'a zkie_compiler::dag::Edge> + '_ {
        self.incident_edges[shard]
            .iter()
            .map(|index| &self.dag.edges[*index])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedClaim {
    id: AggregationChildId,
    aggregation_digest: Digest32,
    run_identity_digest: Digest32,
    partition_digest: Digest32,
    model_digest: Digest32,
    weights_digest: Digest32,
    proof_flavor: ProofFlavorId,
    covered_shards: BTreeSet<u64>,
    frontier: BTreeMap<String, EdgeFrontier>,
    closed_edges: BTreeSet<String>,
    public_inputs: Vec<PublicBoundaryBinding>,
    public_outputs: Vec<PublicBoundaryBinding>,
    leaves: BTreeMap<u64, LeafBinding>,
    digest: Digest32,
}

impl VerifiedClaim {
    pub fn closed_edge_count(&self) -> usize {
        self.closed_edges.len()
    }
    pub fn frontier_count(&self) -> usize {
        self.frontier.len()
    }
    pub fn digest(&self) -> Digest32 {
        self.digest
    }
}

pub fn admit_verified_leaf(
    aggregation: &AggregationPlan,
    partition: &PartitionPlan,
    expected_run: &RunIdentity,
    proof: VerifiedProof,
) -> Result<VerifiedClaim, AggregationError> {
    let expectations = aggregation.trusted_leaf_expectations(partition)?;
    let dag_index = DagIndex::new(partition.dag(), partition.shards().len())?;
    admit_verified_leaf_indexed(
        aggregation,
        partition,
        expected_run,
        proof,
        expectations,
        &dag_index,
    )
}

fn admit_verified_leaf_indexed(
    aggregation: &AggregationPlan,
    partition: &PartitionPlan,
    expected_run: &RunIdentity,
    proof: VerifiedProof,
    expectations: &LeafExpectationSet,
    index: &DagIndex<'_>,
) -> Result<VerifiedClaim, AggregationError> {
    validate_public_model(expected_run)?;
    let shard_id = proof.shard().id();
    let shard_index = usize::try_from(shard_id)
        .map_err(|_| AggregationError::LeafIdentityMismatch { shard: shard_id })?;
    let planned = partition
        .shards()
        .get(shard_index)
        .ok_or(AggregationError::LeafIdentityMismatch { shard: shard_id })?;
    let expected_leaf = expectations
        .leaves
        .get(&shard_id)
        .ok_or(AggregationError::LeafIdentityMismatch { shard: shard_id })?;
    if aggregation.partition_digest != partition.digest()
        || expected_run.partition_plan_digest != partition.digest()
        || expected_run.aggregation_plan_digest != aggregation.digest
        || expected_run.aggregation_fan_in != u32::from(aggregation.fan_in.get())
        || expected_run.public_input_schema_version != LEAF_STATEMENT_SCHEMA_VERSION
        || expected_run.model_graph_digest != partition.model_digest()
        || proof.run_identity_digest() != expected_run.canonical_digest()
        || proof.shard().name() != planned.name()
        || planned.id() != shard_id
        || expected_leaf.shard_id != shard_id
        || expected_leaf.shard_name != planned.name()
        || expected_leaf.circuit_digest != proof.circuit_digest()
        || expected_leaf.verification_key_digest != proof.verification_key_digest()
        || &expected_leaf.execution_backend != proof.execution_backend()
        || &expected_leaf.proof_flavor != proof.proof_flavor()
    {
        return Err(AggregationError::LeafIdentityMismatch { shard: shard_id });
    }
    let statement = LeafStatement::decode(proof.public_statement())
        .map_err(|_| AggregationError::MalformedStatement { shard: shard_id })?;
    if statement.shard_id() != shard_id
        || statement.shard_name() != planned.name()
        || statement.circuit_digest() != proof.circuit_digest()
        || statement.partition_digest() != partition.digest()
        || statement.model_digest() != expected_run.model_graph_digest
        || statement.weights_digest() != expected_run.weights_digest
        || statement.proof_flavor() != &expected_run.proof_flavor
        || proof.proof_flavor() != &expected_run.proof_flavor
        || statement.verification_key_digest() != proof.verification_key_digest()
    {
        return Err(AggregationError::LeafIdentityMismatch { shard: shard_id });
    }
    let public_inputs = statement
        .input_claims()
        .iter()
        .filter(|claim| is_public_graph_input(claim.descriptor()))
        .map(|claim| PublicBoundaryBinding {
            descriptor: claim.descriptor().clone(),
            commitment: claim.commitment(),
        })
        .collect::<Vec<_>>();
    let public_outputs = statement
        .output_claims()
        .iter()
        .filter(|claim| !claim.descriptor().graph_output_names().is_empty())
        .map(|claim| PublicBoundaryBinding {
            descriptor: claim.descriptor().clone(),
            commitment: claim.commitment(),
        })
        .collect::<Vec<_>>();
    if !descriptors_match(&public_inputs, &expected_leaf.public_inputs)
        || !descriptors_match(&public_outputs, &expected_leaf.public_outputs)
    {
        return Err(AggregationError::LeafIdentityMismatch { shard: shard_id });
    }
    let mut frontier = BTreeMap::new();
    for (role_is_output, claims) in [
        (false, statement.input_claims()),
        (true, statement.output_claims()),
    ] {
        for claim in claims {
            let is_public = if role_is_output {
                !claim.descriptor().graph_output_names().is_empty()
            } else {
                is_public_graph_input(claim.descriptor())
            };
            if claim.descriptor().edge_ids().is_empty() && !is_public {
                return Err(AggregationError::ExtraBoundary {
                    edge: claim.descriptor().register_id().to_owned(),
                    shard: shard_id,
                });
            }
            for edge_id in claim.descriptor().edge_ids() {
                let edge = index
                    .edge(edge_id)
                    .ok_or_else(|| AggregationError::ExtraBoundary {
                        edge: edge_id.clone(),
                        shard: shard_id,
                    })?;
                let expected_endpoint = if role_is_output {
                    edge.producer
                } else {
                    edge.consumer
                };
                if expected_endpoint != shard_index
                    || claim.descriptor().register_id() != register_id(&edge.register)
                {
                    return Err(AggregationError::EndpointMismatch {
                        edge: edge_id.clone(),
                        shard: shard_id,
                    });
                }
                let entry = frontier.entry(edge_id.clone()).or_insert(EdgeFrontier {
                    producer: None,
                    consumer: None,
                });
                let slot = if role_is_output {
                    &mut entry.producer
                } else {
                    &mut entry.consumer
                };
                if slot.replace(claim.commitment()).is_some() {
                    return Err(AggregationError::ExtraBoundary {
                        edge: edge_id.clone(),
                        shard: shard_id,
                    });
                }
            }
        }
    }
    for edge in index.incident(shard_index) {
        let entry =
            frontier
                .get(edge.id.as_str())
                .ok_or_else(|| AggregationError::MissingBoundary {
                    edge: edge.id.as_str().into(),
                    shard: shard_id,
                })?;
        if (edge.producer == shard_index) != entry.producer.is_some()
            || (edge.consumer == shard_index) != entry.consumer.is_some()
        {
            return Err(AggregationError::EndpointMismatch {
                edge: edge.id.as_str().into(),
                shard: shard_id,
            });
        }
    }
    let mut claim = VerifiedClaim {
        id: AggregationChildId::Leaf(shard_id),
        aggregation_digest: aggregation.digest,
        run_identity_digest: expected_run.canonical_digest(),
        partition_digest: partition.digest(),
        model_digest: expected_run.model_graph_digest,
        weights_digest: expected_run.weights_digest,
        proof_flavor: expected_run.proof_flavor.clone(),
        covered_shards: BTreeSet::from([shard_id]),
        frontier,
        closed_edges: BTreeSet::new(),
        public_inputs,
        public_outputs,
        leaves: BTreeMap::from([(
            shard_id,
            LeafBinding {
                proof_digest: proof.proof_digest(),
                statement_digest: proof.public_statement_digest(),
                circuit_digest: proof.circuit_digest(),
                verification_key_digest: proof.verification_key_digest(),
            },
        )]),
        digest: Digest32::default(),
    };
    claim.digest = claim_digest(&claim);
    Ok(claim)
}

pub fn merge_verified_claims(
    node: &AggregationNode,
    children: &[VerifiedClaim],
    dag: &Dag,
) -> Result<VerifiedClaim, AggregationError> {
    let shard_count = dag.shards.len();
    let index = DagIndex::new(dag, shard_count)?;
    merge_verified_claims_indexed(node, children, &index)
}

fn merge_verified_claims_indexed(
    node: &AggregationNode,
    children: &[VerifiedClaim],
    index: &DagIndex<'_>,
) -> Result<VerifiedClaim, AggregationError> {
    let node_name = child_name(&node.id);
    if children.len() != node.children.len()
        || children
            .iter()
            .zip(&node.children)
            .any(|(claim, expected)| &claim.id != expected)
    {
        return Err(AggregationError::ChildOrderMismatch { node: node_name });
    }
    let first = children
        .first()
        .ok_or_else(|| AggregationError::ChildOrderMismatch {
            node: child_name(&node.id),
        })?;
    if first.aggregation_digest != node.plan_digest
        || children.iter().any(|claim| {
            claim.aggregation_digest != first.aggregation_digest
                || claim.run_identity_digest != first.run_identity_digest
                || claim.partition_digest != first.partition_digest
                || claim.model_digest != first.model_digest
                || claim.weights_digest != first.weights_digest
                || claim.proof_flavor != first.proof_flavor
        })
    {
        return Err(AggregationError::AggregationIdentityMismatch);
    }
    let mut covered = BTreeSet::new();
    let mut frontier = BTreeMap::<String, EdgeFrontier>::new();
    let mut closed = BTreeSet::new();
    let mut public_inputs = BTreeMap::new();
    let mut public_outputs = BTreeMap::new();
    let mut public_output_names = BTreeSet::new();
    let mut leaves = BTreeMap::new();
    for child in children {
        for shard in &child.covered_shards {
            if !covered.insert(*shard) {
                return Err(AggregationError::OverlappingShard {
                    node: child_name(&node.id),
                    shard: *shard,
                });
            }
        }
        for edge in &child.closed_edges {
            if !closed.insert(edge.clone()) {
                return Err(AggregationError::DuplicateEdgeClosure {
                    node: child_name(&node.id),
                    edge: edge.clone(),
                });
            }
        }
        for (edge, incoming) in &child.frontier {
            let entry = frontier.entry(edge.clone()).or_insert(EdgeFrontier {
                producer: None,
                consumer: None,
            });
            for (target, value) in [
                (&mut entry.producer, incoming.producer),
                (&mut entry.consumer, incoming.consumer),
            ] {
                if let Some(value) = value {
                    if target.replace(value).is_some() {
                        return Err(AggregationError::ExtraBoundary {
                            edge: edge.clone(),
                            shard: 0,
                        });
                    }
                }
            }
        }
        for (shard, binding) in &child.leaves {
            leaves.insert(*shard, binding.clone());
        }
        merge_public_inputs(&mut public_inputs, &child.public_inputs, &node_name)?;
        merge_public_outputs(
            &mut public_outputs,
            &mut public_output_names,
            &child.public_outputs,
            &node_name,
        )?;
        let public_boundary_budget = public_inputs
            .values()
            .chain(public_outputs.values())
            .try_fold((0_usize, 0_usize), |(count, bytes), binding| {
                Some((
                    count.checked_add(1)?,
                    bytes.checked_add(public_boundary_wire_upper_bound(&binding.descriptor)?)?,
                ))
            });
        if public_boundary_budget.is_none_or(|(count, bytes)| {
            count > MAX_PUBLIC_BOUNDARY_DECLARATIONS || bytes > MAX_PUBLIC_BOUNDARY_WIRE_BYTES
        }) {
            return Err(AggregationError::ResourceLimit);
        }
    }
    let frontier_edges = frontier.keys().cloned().collect::<Vec<_>>();
    for edge_id in frontier_edges {
        let edge = index
            .edge(&edge_id)
            .ok_or_else(|| AggregationError::ExtraBoundary {
                edge: edge_id.clone(),
                shard: 0,
            })?;
        let producer = covered.contains(&(edge.producer as u64));
        let consumer = covered.contains(&(edge.consumer as u64));
        if producer && consumer {
            if closed.contains(&edge_id) {
                if frontier.contains_key(&edge_id) {
                    return Err(AggregationError::DuplicateEdgeClosure {
                        node: child_name(&node.id),
                        edge: edge_id.clone(),
                    });
                }
                continue;
            }
            let endpoints =
                frontier
                    .remove(&edge_id)
                    .ok_or_else(|| AggregationError::MissingBoundary {
                        edge: edge_id.clone(),
                        shard: edge.consumer as u64,
                    })?;
            if endpoints.producer.is_none() || endpoints.consumer.is_none() {
                return Err(AggregationError::MissingBoundary {
                    edge: edge_id.clone(),
                    shard: edge.consumer as u64,
                });
            }
            if endpoints.producer != endpoints.consumer {
                return Err(AggregationError::BoundaryCommitmentMismatch {
                    node: child_name(&node.id),
                    edge: edge_id.clone(),
                });
            }
            closed.insert(edge_id);
        }
    }
    let public_inputs = public_inputs.into_values().collect();
    let public_outputs = public_outputs.into_values().collect();
    let mut claim = VerifiedClaim {
        id: node.id.clone(),
        aggregation_digest: first.aggregation_digest,
        run_identity_digest: first.run_identity_digest,
        partition_digest: first.partition_digest,
        model_digest: first.model_digest,
        weights_digest: first.weights_digest,
        proof_flavor: first.proof_flavor.clone(),
        covered_shards: covered,
        frontier,
        closed_edges: closed,
        public_inputs,
        public_outputs,
        leaves,
        digest: Digest32::default(),
    };
    claim.digest = claim_digest(&claim);
    Ok(claim)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinalArtifactKind {
    NativeVerifiedManifest,
    RecursiveRootProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedManifest {
    kind: FinalArtifactKind,
    aggregation_digest: Digest32,
    partition_digest: Digest32,
    root_claim_digest: Digest32,
    leaf_count: usize,
    model_digest: Digest32,
    weights_digest: Digest32,
    proof_flavor: ProofFlavorId,
    public_inputs: Vec<PublicBoundaryWire>,
    public_outputs: Vec<PublicBoundaryWire>,
    manifest_digest: Digest32,
}

impl VerifiedManifest {
    pub fn kind(&self) -> FinalArtifactKind {
        self.kind
    }
    pub fn require_kind(&self, kind: FinalArtifactKind) -> Result<(), AggregationError> {
        if kind == FinalArtifactKind::RecursiveRootProof {
            return Err(AggregationError::RecursiveProofUnavailable);
        }
        if self.kind != kind {
            return Err(AggregationError::AggregationIdentityMismatch);
        }
        Ok(())
    }
    pub fn digest(&self) -> Digest32 {
        self.manifest_digest
    }
    pub fn public_input_count(&self) -> usize {
        self.public_inputs.len()
    }
    pub fn public_output_count(&self) -> usize {
        self.public_outputs.len()
    }
    pub fn encode(&self) -> Result<Vec<u8>, AggregationError> {
        let bytes = serde_json::to_vec(&ManifestWire::from(self))
            .map_err(|_| AggregationError::MalformedManifest)?;
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(AggregationError::ResourceLimit);
        }
        Ok(bytes)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWire {
    schema_version: u32,
    kind: FinalArtifactKind,
    aggregation_digest: Digest32,
    partition_digest: Digest32,
    root_claim_digest: Digest32,
    leaf_count: u64,
    model_digest: Digest32,
    weights_digest: Digest32,
    proof_flavor: ProofFlavorId,
    public_inputs: Vec<PublicBoundaryWire>,
    public_outputs: Vec<PublicBoundaryWire>,
    manifest_digest: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicBoundaryWire {
    role: u8,
    dtype: u8,
    register_id: String,
    edge_ids: Vec<String>,
    graph_output_names: Vec<String>,
    element_count: u64,
    quantization_scale: u64,
    commitment: Digest32,
}

impl From<&PublicBoundaryBinding> for PublicBoundaryWire {
    fn from(value: &PublicBoundaryBinding) -> Self {
        Self {
            role: match value.descriptor.role() {
                BoundaryRole::Input => 1,
                BoundaryRole::Output => 2,
            },
            dtype: 1,
            register_id: value.descriptor.register_id().to_owned(),
            edge_ids: value.descriptor.edge_ids().to_vec(),
            graph_output_names: value.descriptor.graph_output_names().to_vec(),
            element_count: value.descriptor.element_count() as u64,
            quantization_scale: value.descriptor.quantization_scale(),
            commitment: field_encoding(value.commitment),
        }
    }
}

impl From<&VerifiedManifest> for ManifestWire {
    fn from(value: &VerifiedManifest) -> Self {
        Self {
            schema_version: 2,
            kind: value.kind,
            aggregation_digest: value.aggregation_digest,
            partition_digest: value.partition_digest,
            root_claim_digest: value.root_claim_digest,
            leaf_count: value.leaf_count as u64,
            model_digest: value.model_digest,
            weights_digest: value.weights_digest,
            proof_flavor: value.proof_flavor.clone(),
            public_inputs: value.public_inputs.clone(),
            public_outputs: value.public_outputs.clone(),
            manifest_digest: value.manifest_digest,
        }
    }
}

pub fn finalize_native_manifest(
    aggregation: &AggregationPlan,
    partition: &PartitionPlan,
    expected_run: &RunIdentity,
    root: VerifiedClaim,
) -> Result<VerifiedManifest, AggregationError> {
    aggregation.trusted_leaf_expectations(partition)?;
    validate_public_model(expected_run)?;
    if root.id != aggregation.root
        || root.aggregation_digest != aggregation.digest
        || root.partition_digest != partition.digest()
        || expected_run.aggregation_plan_digest != aggregation.digest
        || expected_run.partition_plan_digest != partition.digest()
    {
        return Err(AggregationError::AggregationIdentityMismatch);
    }
    for shard in partition.shards() {
        if !root.covered_shards.contains(&shard.id()) {
            return Err(AggregationError::MissingLeaf { shard: shard.id() });
        }
    }
    if let Some(edge) = partition
        .dag()
        .edges
        .iter()
        .find(|edge| !root.closed_edges.contains(edge.id.as_str()))
    {
        return Err(AggregationError::OpenInternalEdge {
            edge: edge.id.as_str().into(),
        });
    }
    if let Some(edge) = root.frontier.keys().next() {
        return Err(AggregationError::OpenInternalEdge { edge: edge.clone() });
    }
    if root.model_digest != expected_run.model_graph_digest
        || root.weights_digest != expected_run.weights_digest
        || root.proof_flavor != expected_run.proof_flavor
        || root.run_identity_digest != expected_run.canonical_digest()
        || root.leaves.len() != partition.shards().len()
    {
        return Err(AggregationError::AggregationIdentityMismatch);
    }
    let mut manifest = VerifiedManifest {
        kind: FinalArtifactKind::NativeVerifiedManifest,
        aggregation_digest: aggregation.digest,
        partition_digest: partition.digest(),
        root_claim_digest: root.digest,
        leaf_count: root.leaves.len(),
        model_digest: root.model_digest,
        weights_digest: root.weights_digest,
        proof_flavor: root.proof_flavor,
        public_inputs: root
            .public_inputs
            .iter()
            .map(PublicBoundaryWire::from)
            .collect(),
        public_outputs: root
            .public_outputs
            .iter()
            .map(PublicBoundaryWire::from)
            .collect(),
        manifest_digest: Digest32::default(),
    };
    manifest.manifest_digest = manifest_digest(&manifest);
    let _ = manifest.encode()?;
    Ok(manifest)
}

pub fn build_native_manifest(
    aggregation: &AggregationPlan,
    partition: &PartitionPlan,
    expected_run: &RunIdentity,
    proofs: Vec<VerifiedProof>,
) -> Result<VerifiedManifest, AggregationError> {
    let expectations = aggregation.trusted_leaf_expectations(partition)?;
    validate_public_model(expected_run)?;
    let dag_index = DagIndex::new(partition.dag(), partition.shards().len())?;
    if proofs.len() != partition.shards().len() {
        let mut supplied = vec![false; partition.shards().len()];
        for proof in &proofs {
            if let Ok(shard) = usize::try_from(proof.shard().id()) {
                if let Some(slot) = supplied.get_mut(shard) {
                    *slot = true;
                }
            }
        }
        let missing = partition
            .shards()
            .iter()
            .find(|shard| !supplied[shard.id() as usize])
            .map(|shard| shard.id())
            .unwrap_or(0);
        return Err(AggregationError::MissingLeaf { shard: missing });
    }
    let mut claims = BTreeMap::new();
    for (index, proof) in proofs.into_iter().enumerate() {
        if proof.shard().id() != index as u64 {
            return Err(AggregationError::ChildOrderMismatch {
                node: "leaf-order".into(),
            });
        }
        let claim = admit_verified_leaf_indexed(
            aggregation,
            partition,
            expected_run,
            proof,
            expectations,
            &dag_index,
        )?;
        if claims.insert(claim.id.clone(), claim).is_some() {
            return Err(AggregationError::OverlappingShard {
                node: "leaf-order".into(),
                shard: index as u64,
            });
        }
    }
    for level in &aggregation.levels {
        for node in level {
            let children = node
                .children
                .iter()
                .map(|id| {
                    claims
                        .remove(id)
                        .ok_or_else(|| AggregationError::ChildOrderMismatch {
                            node: child_name(&node.id),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let claim = merge_verified_claims_indexed(node, &children, &dag_index)?;
            claims.insert(node.id.clone(), claim);
        }
    }
    let root = claims
        .remove(&aggregation.root)
        .ok_or(AggregationError::AggregationIdentityMismatch)?;
    if !claims.is_empty() {
        return Err(AggregationError::AggregationIdentityMismatch);
    }
    finalize_native_manifest(aggregation, partition, expected_run, root)
}

pub fn verify_native_manifest(
    bytes: &[u8],
    requested_kind: FinalArtifactKind,
    aggregation: &AggregationPlan,
    partition: &PartitionPlan,
    expected_run: &RunIdentity,
    proofs: Vec<VerifiedProof>,
) -> Result<VerifiedManifest, AggregationError> {
    if requested_kind == FinalArtifactKind::RecursiveRootProof {
        return Err(AggregationError::RecursiveProofUnavailable);
    }
    aggregation.trusted_leaf_expectations(partition)?;
    validate_public_model(expected_run)?;
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(AggregationError::ResourceLimit);
    }
    let wire: ManifestWire =
        serde_json::from_slice(bytes).map_err(|_| AggregationError::MalformedManifest)?;
    let leaf_count =
        usize::try_from(wire.leaf_count).map_err(|_| AggregationError::ResourceLimit)?;
    if wire.schema_version != 2
        || wire.kind != FinalArtifactKind::NativeVerifiedManifest
        || wire.kind != requested_kind
        || leaf_count > MAX_LEAVES
    {
        return Err(AggregationError::MalformedManifest);
    }
    let expected = build_native_manifest(aggregation, partition, expected_run, proofs)?;
    if wire.aggregation_digest != expected.aggregation_digest
        || wire.partition_digest != expected.partition_digest
        || wire.root_claim_digest != expected.root_claim_digest
        || leaf_count != expected.leaf_count
        || wire.model_digest != expected.model_digest
        || wire.weights_digest != expected.weights_digest
        || wire.proof_flavor != expected.proof_flavor
        || wire.public_inputs != expected.public_inputs
        || wire.public_outputs != expected.public_outputs
        || wire.manifest_digest != expected.manifest_digest
        || manifest_digest(&expected) != expected.manifest_digest
    {
        return Err(AggregationError::AggregationIdentityMismatch);
    }
    Ok(expected)
}

fn manifest_digest(manifest: &VerifiedManifest) -> Digest32 {
    let mut bytes = b"zkie.native-verified-manifest.v2\0".to_vec();
    bytes.push(match manifest.kind {
        FinalArtifactKind::NativeVerifiedManifest => 1,
        FinalArtifactKind::RecursiveRootProof => 2,
    });
    for digest in [
        manifest.aggregation_digest,
        manifest.partition_digest,
        manifest.root_claim_digest,
        manifest.model_digest,
        manifest.weights_digest,
    ] {
        bytes.extend_from_slice(digest.as_bytes());
    }
    bytes.extend_from_slice(&(manifest.leaf_count as u64).to_le_bytes());
    bytes.extend_from_slice(&(manifest.proof_flavor.as_str().len() as u64).to_le_bytes());
    bytes.extend_from_slice(manifest.proof_flavor.as_str().as_bytes());
    encode_public_boundary_wires(&mut bytes, b"public-inputs\0", &manifest.public_inputs);
    encode_public_boundary_wires(&mut bytes, b"public-outputs\0", &manifest.public_outputs);
    Digest32::new(*blake3::hash(&bytes).as_bytes())
}

fn field_encoding(value: Fr) -> Digest32 {
    let representation = value.to_repr();
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(representation.as_ref());
    Digest32::new(bytes)
}

fn encode_public_boundary_wires(
    bytes: &mut Vec<u8>,
    domain: &[u8],
    boundaries: &[PublicBoundaryWire],
) {
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&(boundaries.len() as u64).to_le_bytes());
    for boundary in boundaries {
        bytes.push(boundary.role);
        bytes.push(boundary.dtype);
        encode_text(bytes, &boundary.register_id);
        encode_text_list(bytes, &boundary.edge_ids);
        encode_text_list(bytes, &boundary.graph_output_names);
        bytes.extend_from_slice(&boundary.element_count.to_le_bytes());
        bytes.extend_from_slice(&boundary.quantization_scale.to_le_bytes());
        bytes.extend_from_slice(boundary.commitment.as_bytes());
    }
}

fn encode_boundary_descriptor(bytes: &mut Vec<u8>, descriptor: &BoundaryDescriptor) {
    bytes.push(match descriptor.role() {
        BoundaryRole::Input => 1,
        BoundaryRole::Output => 2,
    });
    bytes.push(1); // I18
    encode_text(bytes, descriptor.register_id());
    encode_text_list(bytes, descriptor.edge_ids());
    encode_text_list(bytes, descriptor.graph_output_names());
    bytes.extend_from_slice(&(descriptor.element_count() as u64).to_le_bytes());
    bytes.extend_from_slice(&descriptor.quantization_scale().to_le_bytes());
}

/// Conservative upper bound for one JSON wire entry. ASCII identities may
/// expand to six bytes each when JSON-escaped; the fixed allowance covers
/// field names, integers, separators, and the 32-byte commitment array.
fn public_boundary_wire_upper_bound(descriptor: &BoundaryDescriptor) -> Option<usize> {
    let mut identities = std::iter::once(descriptor.register_id())
        .chain(descriptor.edge_ids().iter().map(String::as_str))
        .chain(descriptor.graph_output_names().iter().map(String::as_str));
    let (identity_count, identity_bytes) =
        identities.try_fold((0_usize, 0_usize), |(count, bytes), identity| {
            Some((count.checked_add(1)?, bytes.checked_add(identity.len())?))
        })?;
    512_usize
        .checked_add(identity_count.checked_mul(16)?)?
        .checked_add(identity_bytes.checked_mul(6)?)
}

fn encode_public_bindings(
    bytes: &mut Vec<u8>,
    domain: &[u8],
    boundaries: &[PublicBoundaryBinding],
) {
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&(boundaries.len() as u64).to_le_bytes());
    for boundary in boundaries {
        encode_boundary_descriptor(bytes, &boundary.descriptor);
        bytes.extend_from_slice(boundary.commitment.to_repr().as_ref());
    }
}

fn encode_text(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn encode_text_list(bytes: &mut Vec<u8>, values: &[String]) {
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        encode_text(bytes, value);
    }
}

fn register_id(register: &Register) -> String {
    match register {
        Register::GraphInput(name) => format!("graph-input:{name}"),
        Register::Virtual(index) => format!("virtual:{index}"),
        Register::Weight(name) => format!("weight:{name}"),
    }
}

fn validate_public_model(run: &RunIdentity) -> Result<(), AggregationError> {
    if run.model_visibility != ModelVisibility::PublicModel {
        return Err(AggregationError::UnsupportedModelVisibility);
    }
    Ok(())
}

fn is_public_graph_input(descriptor: &BoundaryDescriptor) -> bool {
    descriptor.role() == BoundaryRole::Input
        && descriptor.register_id().starts_with("graph-input:")
        && descriptor.edge_ids().is_empty()
        && descriptor.graph_output_names().is_empty()
}

fn descriptors_match(actual: &[PublicBoundaryBinding], expected: &[BoundaryDescriptor]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| &actual.descriptor == expected)
}

fn merge_public_inputs(
    target: &mut BTreeMap<String, PublicBoundaryBinding>,
    incoming: &[PublicBoundaryBinding],
    node: &str,
) -> Result<(), AggregationError> {
    for binding in incoming {
        let key = binding.descriptor.register_id().to_owned();
        if let Some(existing) = target.get(&key) {
            if existing != binding {
                return Err(AggregationError::BoundaryCommitmentMismatch {
                    node: node.to_owned(),
                    edge: key,
                });
            }
        } else {
            target.insert(key, binding.clone());
        }
    }
    Ok(())
}

fn merge_public_outputs(
    target: &mut BTreeMap<String, PublicBoundaryBinding>,
    claimed_names: &mut BTreeSet<String>,
    incoming: &[PublicBoundaryBinding],
    node: &str,
) -> Result<(), AggregationError> {
    for binding in incoming {
        if let Some(name) = binding
            .descriptor
            .graph_output_names()
            .iter()
            .find(|name| claimed_names.contains(*name))
        {
            return Err(AggregationError::DuplicateEdgeClosure {
                node: node.to_owned(),
                edge: format!("graph-output:{name}"),
            });
        }
        claimed_names.extend(binding.descriptor.graph_output_names().iter().cloned());
        let key = binding
            .descriptor
            .graph_output_names()
            .first()
            .ok_or(AggregationError::AggregationIdentityMismatch)?
            .clone();
        if target.insert(key, binding.clone()).is_some() {
            return Err(AggregationError::AggregationIdentityMismatch);
        }
    }
    Ok(())
}

fn child_name(id: &AggregationChildId) -> String {
    match id {
        AggregationChildId::Leaf(shard) => format!("leaf-{shard}"),
        AggregationChildId::Node { level, index } => format!("node-{level}-{index}"),
    }
}

fn claim_digest(claim: &VerifiedClaim) -> Digest32 {
    let mut bytes = b"zkie.verified-aggregation-claim.v2\0".to_vec();
    encode_child_id(&mut bytes, &claim.id);
    for digest in [
        claim.aggregation_digest,
        claim.run_identity_digest,
        claim.partition_digest,
        claim.model_digest,
        claim.weights_digest,
    ] {
        bytes.extend_from_slice(digest.as_bytes());
    }
    bytes.extend_from_slice(&(claim.proof_flavor.as_str().len() as u64).to_le_bytes());
    bytes.extend_from_slice(claim.proof_flavor.as_str().as_bytes());
    bytes.extend_from_slice(b"covered\0");
    bytes.extend_from_slice(&(claim.covered_shards.len() as u64).to_le_bytes());
    for shard in &claim.covered_shards {
        bytes.extend_from_slice(&shard.to_le_bytes());
    }
    bytes.extend_from_slice(b"frontier\0");
    bytes.extend_from_slice(&(claim.frontier.len() as u64).to_le_bytes());
    for (edge, endpoints) in &claim.frontier {
        bytes.extend_from_slice(&(edge.len() as u64).to_le_bytes());
        bytes.extend_from_slice(edge.as_bytes());
        for value in [endpoints.producer, endpoints.consumer] {
            bytes.push(u8::from(value.is_some()));
            if let Some(value) = value {
                bytes.extend_from_slice(value.to_repr().as_ref());
            }
        }
    }
    bytes.extend_from_slice(b"closed\0");
    bytes.extend_from_slice(&(claim.closed_edges.len() as u64).to_le_bytes());
    for edge in &claim.closed_edges {
        bytes.extend_from_slice(&(edge.len() as u64).to_le_bytes());
        bytes.extend_from_slice(edge.as_bytes());
    }
    encode_public_bindings(&mut bytes, b"public-inputs\0", &claim.public_inputs);
    encode_public_bindings(&mut bytes, b"public-outputs\0", &claim.public_outputs);
    bytes.extend_from_slice(b"leaves\0");
    bytes.extend_from_slice(&(claim.leaves.len() as u64).to_le_bytes());
    for (shard, leaf) in &claim.leaves {
        bytes.extend_from_slice(&shard.to_le_bytes());
        for digest in [
            leaf.proof_digest,
            leaf.statement_digest,
            leaf.circuit_digest,
            leaf.verification_key_digest,
        ] {
            bytes.extend_from_slice(digest.as_bytes());
        }
    }
    Digest32::new(*blake3::hash(&bytes).as_bytes())
}

fn aggregation_plan_digest(
    partition_digest: Digest32,
    fan_in: NonZeroU8,
    leaf_count: usize,
    levels: &[Vec<AggregationNode>],
    root: &AggregationChildId,
    leaf_expectation_set_digest: Option<Digest32>,
) -> Digest32 {
    let mut bytes = b"zkie.native-aggregation-plan.v2\0".to_vec();
    bytes.extend_from_slice(partition_digest.as_bytes());
    bytes.push(u8::from(leaf_expectation_set_digest.is_some()));
    if let Some(digest) = leaf_expectation_set_digest {
        bytes.extend_from_slice(digest.as_bytes());
    }
    bytes.push(fan_in.get());
    bytes.extend_from_slice(&(leaf_count as u64).to_le_bytes());
    bytes.extend_from_slice(&(levels.len() as u64).to_le_bytes());
    for level in levels {
        bytes.extend_from_slice(&(level.len() as u64).to_le_bytes());
        for node in level {
            encode_child_id(&mut bytes, &node.id);
            bytes.push(node.actual_arity());
            for child in &node.children {
                encode_child_id(&mut bytes, child);
            }
        }
    }
    encode_child_id(&mut bytes, root);
    Digest32::new(*blake3::hash(&bytes).as_bytes())
}

fn leaf_expectation_set_digest(
    partition_digest: Digest32,
    leaves: &BTreeMap<u64, LeafExpectation>,
) -> Digest32 {
    let mut bytes = b"zkie.leaf-expectation-set.v1\0".to_vec();
    bytes.extend_from_slice(partition_digest.as_bytes());
    bytes.extend_from_slice(&(leaves.len() as u64).to_le_bytes());
    for (shard_id, leaf) in leaves {
        bytes.extend_from_slice(&shard_id.to_le_bytes());
        bytes.extend_from_slice(&(leaf.shard_name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(leaf.shard_name.as_bytes());
        bytes.extend_from_slice(leaf.circuit_digest.as_bytes());
        bytes.extend_from_slice(leaf.verification_key_digest.as_bytes());
        bytes.extend_from_slice(&(leaf.execution_backend.as_str().len() as u64).to_le_bytes());
        bytes.extend_from_slice(leaf.execution_backend.as_str().as_bytes());
        bytes.extend_from_slice(&(leaf.proof_flavor.as_str().len() as u64).to_le_bytes());
        bytes.extend_from_slice(leaf.proof_flavor.as_str().as_bytes());
        bytes.extend_from_slice(&(leaf.public_inputs.len() as u64).to_le_bytes());
        for descriptor in &leaf.public_inputs {
            encode_boundary_descriptor(&mut bytes, descriptor);
        }
        bytes.extend_from_slice(&(leaf.public_outputs.len() as u64).to_le_bytes());
        for descriptor in &leaf.public_outputs {
            encode_boundary_descriptor(&mut bytes, descriptor);
        }
    }
    Digest32::new(*blake3::hash(&bytes).as_bytes())
}

fn encode_child_id(bytes: &mut Vec<u8>, id: &AggregationChildId) {
    match id {
        AggregationChildId::Leaf(shard) => {
            bytes.push(0);
            bytes.extend_from_slice(&shard.to_le_bytes());
        }
        AggregationChildId::Node { level, index } => {
            bytes.push(1);
            bytes.extend_from_slice(&level.to_le_bytes());
            bytes.extend_from_slice(&index.to_le_bytes());
        }
    }
}
