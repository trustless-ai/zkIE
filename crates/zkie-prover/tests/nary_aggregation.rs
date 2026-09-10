use std::collections::HashMap;
use std::num::NonZeroU8;
use std::path::PathBuf;

use halo2_proofs::halo2curves::bn256::Fr;
use zkie_compiler::dag::{
    InstructionCostModel, InstructionEstimate, PartitionError, PartitionPlanner, PartitionRequest,
    ShardEstimate,
};
use zkie_compiler::graph_compiler::{CompiledInstruction, CompiledProgram, Register};
use zkie_core::chips::poseidon_boundary::{
    commit_boundary_native, BoundaryDescriptor, BoundaryRole,
};
use zkie_core::fixed_point::I18;
use zkie_core::isa::{EltwiseOp, Instruction};
use zkie_prover::{
    admit_verified_leaf, build_native_manifest, finalize_native_manifest, merge_verified_claims,
    plan_aggregation, verify_native_manifest, verify_proof, AggregationError, BackendCapabilities,
    BackendError, BoundaryClaim, CapabilityId, CryptoVerificationReceipt,
    CryptoVerificationRequest, ExecutionBackendId, FinalArtifactKind, KeyIdentity,
    KeyMaterialStore, LeafExpectation, LeafExpectationSet, LeafStatement, ModelVisibility,
    PrepareJob, PreparedKeys, ProofBackend, ProofFlavorId, ProofJob, ResourceRequest, RunIdentity,
    ShardIdentity, UnverifiedProof, VerificationError, VerificationExpectation, VerificationJob,
    WitnessArtifact,
};
use zkie_types::Digest32;

#[derive(Clone)]
struct UnitCost;
impl InstructionCostModel for UnitCost {
    fn identity(&self) -> &str {
        "aggregation-test-unit"
    }
    fn version(&self) -> u32 {
        1
    }
    fn estimate_instruction(
        &self,
        _: &CompiledInstruction,
    ) -> Result<InstructionEstimate, PartitionError> {
        InstructionEstimate::new(1, 1)
    }
    fn estimate_shard(
        &self,
        instructions: &[CompiledInstruction],
    ) -> Result<ShardEstimate, PartitionError> {
        ShardEstimate::new(instructions.len() as u64, 1)
    }
}

fn independent_plan(leaves: usize) -> zkie_compiler::dag::PartitionPlan {
    let instructions = (0..leaves)
        .map(|index| CompiledInstruction {
            instruction: Instruction::Eltwise { op: EltwiseOp::Add },
            inputs: vec![
                Register::GraphInput(format!("x-{index}")),
                Register::GraphInput(format!("y-{index}")),
            ],
            output_name: format!("z-{index}"),
        })
        .collect();
    let program = CompiledProgram {
        instructions,
        weights: HashMap::new(),
        graph_inputs: (0..leaves)
            .flat_map(|index| [format!("x-{index}"), format!("y-{index}")])
            .collect(),
        graph_outputs: vec![("z".into(), Register::Virtual(leaves - 1))],
    };
    let request =
        PartitionRequest::new(1, 1, Digest32::new([7; 32]), "compiler-v1", UnitCost).unwrap();
    PartitionPlanner::plan(&program, &request).unwrap()
}

#[test]
fn fan_in_four_groups_ten_leaves_without_dummy_children() {
    let plan = plan_aggregation(&independent_plan(10), NonZeroU8::new(4).unwrap()).unwrap();
    assert_eq!(
        plan.levels()[0]
            .iter()
            .map(|node| node.actual_arity())
            .collect::<Vec<_>>(),
        vec![4, 4, 2]
    );
    assert_eq!(plan.levels()[1].len(), 1);
    assert_eq!(plan.levels()[1][0].actual_arity(), 3);
    assert_eq!(plan.leaf_count(), 10);
}

#[test]
fn fan_in_is_validated_and_digest_bound() {
    let partition = independent_plan(10);
    let two = plan_aggregation(&partition, NonZeroU8::new(2).unwrap()).unwrap();
    let eight = plan_aggregation(&partition, NonZeroU8::new(8).unwrap()).unwrap();
    assert_ne!(two.digest(), eight.digest());
    assert_eq!(
        plan_aggregation(&partition, NonZeroU8::new(1).unwrap()).unwrap_err(),
        AggregationError::InvalidFanIn { fan_in: 1 }
    );
    assert_eq!(
        plan_aggregation(&partition, NonZeroU8::new(17).unwrap()).unwrap_err(),
        AggregationError::InvalidFanIn { fan_in: 17 }
    );
}

struct ReceiptBackend;
impl ProofBackend for ReceiptBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(
            ExecutionBackendId::parse("receipt-test").unwrap(),
            vec![ProofFlavorId::parse("halo2-test-v1").unwrap()],
            vec![CapabilityId::parse("cryptographic-verification").unwrap()],
        )
        .unwrap()
    }
    fn estimate_resources(&self, _: ProofJob) -> Result<ResourceRequest, BackendError> {
        Ok(ResourceRequest::new(1, 1, 0, 0).unwrap())
    }
    fn prepare(
        &self,
        _: PrepareJob,
        _: &dyn KeyMaterialStore,
    ) -> Result<PreparedKeys, BackendError> {
        unreachable!()
    }
    fn prove(&self, _: ProofJob, _: PathBuf) -> Result<UnverifiedProof, BackendError> {
        unreachable!()
    }
    fn verify_cryptographically(
        &self,
        request: CryptoVerificationRequest,
    ) -> Result<CryptoVerificationReceipt, VerificationError> {
        Ok(CryptoVerificationReceipt::new(
            request.request_binding_digest(),
        ))
    }
}

fn d(byte: u8) -> Digest32 {
    Digest32::new([byte; 32])
}

fn chain_plan() -> zkie_compiler::dag::PartitionPlan {
    let program = CompiledProgram {
        instructions: vec![
            CompiledInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![
                    Register::GraphInput("x".into()),
                    Register::GraphInput("y".into()),
                ],
                output_name: "broadcast".into(),
            },
            CompiledInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![Register::Virtual(0), Register::GraphInput("y".into())],
                output_name: "middle".into(),
            },
            CompiledInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![Register::Virtual(0), Register::Virtual(1)],
                output_name: "z".into(),
            },
        ],
        weights: HashMap::new(),
        graph_inputs: vec!["x".into(), "y".into()],
        graph_outputs: vec![("z".into(), Register::Virtual(2))],
    };
    let request = PartitionRequest::new(1, 1, d(7), "compiler-v1", UnitCost).unwrap();
    PartitionPlanner::plan(&program, &request).unwrap()
}

fn run_for(
    partition: &zkie_compiler::dag::PartitionPlan,
    aggregation_digest: Digest32,
) -> RunIdentity {
    RunIdentity {
        model_graph_digest: partition.model_digest(),
        weights_digest: d(8),
        compiler_digest: d(9),
        isa_digest: d(10),
        quantization_digest: d(11),
        partition_plan_digest: partition.digest(),
        aggregation_plan_digest: aggregation_digest,
        proof_flavor: ProofFlavorId::parse("halo2-test-v1").unwrap(),
        model_visibility: ModelVisibility::PublicModel,
        aggregation_fan_in: 4,
        public_input_schema_version: 2,
    }
}

fn verified_leaf(
    partition: &zkie_compiler::dag::PartitionPlan,
    run: &RunIdentity,
    shard_index: usize,
    corrupt_input_edge: Option<&str>,
) -> zkie_prover::VerifiedProof {
    verified_leaf_with_identity(
        partition,
        run,
        shard_index,
        corrupt_input_edge,
        d(30 + shard_index as u8),
        d(40 + shard_index as u8),
    )
}

fn verified_leaf_with_identity(
    partition: &zkie_compiler::dag::PartitionPlan,
    run: &RunIdentity,
    shard_index: usize,
    corrupt_input_edge: Option<&str>,
    circuit: Digest32,
    vk: Digest32,
) -> zkie_prover::VerifiedProof {
    verified_leaf_with_public_io(
        partition,
        run,
        shard_index,
        corrupt_input_edge,
        circuit,
        vk,
        Vec::new(),
        Vec::new(),
    )
}

#[allow(clippy::too_many_arguments)]
fn verified_leaf_with_public_io(
    partition: &zkie_compiler::dag::PartitionPlan,
    run: &RunIdentity,
    shard_index: usize,
    corrupt_input_edge: Option<&str>,
    circuit: Digest32,
    vk: Digest32,
    mut public_inputs: Vec<BoundaryClaim>,
    mut public_outputs: Vec<BoundaryClaim>,
) -> zkie_prover::VerifiedProof {
    let shard = &partition.dag().shards[shard_index];
    let shard_identity = ShardIdentity::new(shard.id as u64, shard.name.clone(), circuit).unwrap();
    let mut input_claims = Vec::new();
    for register in &shard.inputs {
        let mut edge_ids = partition
            .dag()
            .edges
            .iter()
            .filter(|edge| edge.consumer == shard_index && &edge.register == register)
            .map(|edge| edge.id.as_str().to_owned())
            .collect::<Vec<_>>();
        edge_ids.sort();
        let index = match register {
            Register::Virtual(index) => *index,
            _ => unreachable!(),
        };
        let corrupt = edge_ids
            .iter()
            .any(|edge| Some(edge.as_str()) == corrupt_input_edge);
        let descriptor = BoundaryDescriptor::flat_i18(
            BoundaryRole::Input,
            format!("virtual:{index}"),
            edge_ids,
            Vec::new(),
            1,
            1_000_000_000_000_000_000,
        )
        .unwrap();
        let commitment = commit_boundary_native(
            &descriptor,
            &[I18::from_raw(100 + index as i64 + i64::from(corrupt))],
        )
        .unwrap();
        input_claims.push(BoundaryClaim::new(descriptor, commitment).unwrap());
    }
    input_claims.append(&mut public_inputs);
    let mut output_claims = Vec::new();
    for register in &shard.outputs {
        let mut edge_ids = partition
            .dag()
            .edges
            .iter()
            .filter(|edge| edge.producer == shard_index && &edge.register == register)
            .map(|edge| edge.id.as_str().to_owned())
            .collect::<Vec<_>>();
        edge_ids.sort();
        let index = match register {
            Register::Virtual(index) => *index,
            _ => unreachable!(),
        };
        let descriptor = BoundaryDescriptor::flat_i18(
            BoundaryRole::Output,
            format!("virtual:{index}"),
            edge_ids,
            Vec::new(),
            1,
            1_000_000_000_000_000_000,
        )
        .unwrap();
        let commitment =
            commit_boundary_native(&descriptor, &[I18::from_raw(100 + index as i64)]).unwrap();
        output_claims.push(BoundaryClaim::new(descriptor, commitment).unwrap());
    }
    output_claims.append(&mut public_outputs);
    let statement = LeafStatement::new(
        shard.id as u64,
        shard.name.clone(),
        circuit,
        partition.digest(),
        partition.model_digest(),
        run.weights_digest,
        run.proof_flavor.clone(),
        vk,
        input_claims,
        output_claims,
        Vec::new(),
    )
    .unwrap()
    .encode()
    .unwrap();
    let proof_bytes = format!("proof-{shard_index}").into_bytes();
    let manifest = format!("manifest-{shard_index}").into_bytes();
    let key = KeyIdentity::new(run.proof_flavor.clone(), circuit, vk).unwrap();
    let job = ProofJob::new(
        run.clone(),
        shard_identity.clone(),
        WitnessArtifact::new(
            "w".into(),
            d(60 + shard_index as u8),
            shard_identity.clone(),
            circuit,
        )
        .unwrap(),
        PreparedKeys::new(key.clone(), key).unwrap(),
    )
    .unwrap();
    let proof = UnverifiedProof::new(
        "p".into(),
        Digest32::new(*blake3::hash(&proof_bytes).as_bytes()),
        statement.clone(),
        circuit,
        vk,
        run.proof_flavor.clone(),
        ExecutionBackendId::parse("receipt-test").unwrap(),
        shard_identity,
        manifest.clone(),
        run.canonical_digest(),
    )
    .unwrap();
    let expectation = VerificationExpectation::from_proof_job(
        &job,
        ExecutionBackendId::parse("receipt-test").unwrap(),
        statement,
        Digest32::new(*blake3::hash(&proof_bytes).as_bytes()),
        manifest,
    )
    .unwrap();
    verify_proof(
        &ReceiptBackend,
        VerificationJob::new(expectation, proof, proof_bytes).unwrap(),
    )
    .unwrap()
}

fn leaf_expectations(partition: &zkie_compiler::dag::PartitionPlan) -> LeafExpectationSet {
    let backend = ExecutionBackendId::parse("receipt-test").unwrap();
    let flavor = ProofFlavorId::parse("halo2-test-v1").unwrap();
    LeafExpectationSet::new(
        partition,
        partition
            .shards()
            .iter()
            .map(|shard| {
                LeafExpectation::new(
                    shard.id(),
                    shard.name().to_owned(),
                    d(30 + shard.id() as u8),
                    d(40 + shard.id() as u8),
                    backend.clone(),
                    flavor.clone(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}

fn public_input_descriptor(name: &str) -> BoundaryDescriptor {
    BoundaryDescriptor::flat_i18(
        BoundaryRole::Input,
        format!("graph-input:{name}"),
        Vec::new(),
        Vec::new(),
        1,
        1_000_000_000_000_000_000,
    )
    .unwrap()
}

fn public_output_descriptor(name: &str) -> BoundaryDescriptor {
    BoundaryDescriptor::flat_i18(
        BoundaryRole::Output,
        "virtual:2",
        Vec::new(),
        vec![name.to_owned()],
        1,
        1_000_000_000_000_000_000,
    )
    .unwrap()
}

fn public_io_expectations(
    partition: &zkie_compiler::dag::PartitionPlan,
    input: &BoundaryDescriptor,
    output: &BoundaryDescriptor,
) -> LeafExpectationSet {
    let backend = ExecutionBackendId::parse("receipt-test").unwrap();
    let flavor = ProofFlavorId::parse("halo2-test-v1").unwrap();
    LeafExpectationSet::new(
        partition,
        partition
            .shards()
            .iter()
            .map(|shard| {
                let expectation = LeafExpectation::new(
                    shard.id(),
                    shard.name().to_owned(),
                    d(30 + shard.id() as u8),
                    d(40 + shard.id() as u8),
                    backend.clone(),
                    flavor.clone(),
                )
                .unwrap();
                match shard.id() {
                    0 => expectation.with_public_io(vec![input.clone()], Vec::new()),
                    2 => expectation.with_public_io(Vec::new(), vec![output.clone()]),
                    _ => Ok(expectation),
                }
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn leaf_expectations_reject_a_globally_oversized_public_io_plan() {
    let partition = independent_plan(2);
    let backend = ExecutionBackendId::parse("receipt-test").unwrap();
    let flavor = ProofFlavorId::parse("halo2-test-v1").unwrap();
    let first_inputs = (0..4096)
        .map(|index| public_input_descriptor(&format!("input-{index:04}")))
        .collect();
    let expectations = vec![
        LeafExpectation::new(
            0,
            partition.shards()[0].name().to_owned(),
            d(30),
            d(40),
            backend.clone(),
            flavor.clone(),
        )
        .unwrap()
        .with_public_io(first_inputs, Vec::new())
        .unwrap(),
        LeafExpectation::new(
            1,
            partition.shards()[1].name().to_owned(),
            d(31),
            d(41),
            backend,
            flavor,
        )
        .unwrap()
        .with_public_io(vec![public_input_descriptor("one-too-many")], Vec::new())
        .unwrap(),
    ];

    assert_eq!(
        LeafExpectationSet::new(&partition, expectations),
        Err(AggregationError::InvalidLeafExpectations),
    );
}

#[test]
fn public_io_is_trusted_preserved_and_manifest_tampering_is_rejected() {
    let partition = chain_plan();
    let input = public_input_descriptor("x");
    let output = public_output_descriptor("z");
    let expectations = public_io_expectations(&partition, &input, &output);
    let aggregation = plan_aggregation(&partition, NonZeroU8::new(4).unwrap())
        .unwrap()
        .bind_leaf_expectations(&expectations)
        .unwrap();
    let run = run_for(&partition, aggregation.digest());
    let input_claim = BoundaryClaim::new(input.clone(), Fr::from(123)).unwrap();
    let output_claim = BoundaryClaim::new(output.clone(), Fr::from(456)).unwrap();
    let proofs = vec![
        verified_leaf_with_public_io(
            &partition,
            &run,
            0,
            None,
            d(30),
            d(40),
            vec![input_claim.clone()],
            Vec::new(),
        ),
        verified_leaf(&partition, &run, 1, None),
        verified_leaf_with_public_io(
            &partition,
            &run,
            2,
            None,
            d(32),
            d(42),
            Vec::new(),
            vec![output_claim.clone()],
        ),
    ];
    let manifest = build_native_manifest(&aggregation, &partition, &run, proofs).unwrap();
    assert_eq!(manifest.public_input_count(), 1);
    assert_eq!(manifest.public_output_count(), 1);

    let missing = verified_leaf(&partition, &run, 0, None);
    assert_eq!(
        admit_verified_leaf(&aggregation, &partition, &run, missing),
        Err(AggregationError::LeafIdentityMismatch { shard: 0 }),
    );

    let wrong_route = public_output_descriptor("wrong-z");
    let wrong_route = BoundaryClaim::new(wrong_route, Fr::from(456)).unwrap();
    let wrong = verified_leaf_with_public_io(
        &partition,
        &run,
        2,
        None,
        d(32),
        d(42),
        Vec::new(),
        vec![wrong_route],
    );
    assert_eq!(
        admit_verified_leaf(&aggregation, &partition, &run, wrong),
        Err(AggregationError::LeafIdentityMismatch { shard: 2 }),
    );

    let mut encoded: serde_json::Value =
        serde_json::from_slice(&manifest.encode().unwrap()).unwrap();
    encoded["public_inputs"][0]["commitment"] = serde_json::to_value(d(77)).unwrap();
    let tampered = serde_json::to_vec(&encoded).unwrap();
    let fresh = vec![
        verified_leaf_with_public_io(
            &partition,
            &run,
            0,
            None,
            d(30),
            d(40),
            vec![input_claim],
            Vec::new(),
        ),
        verified_leaf(&partition, &run, 1, None),
        verified_leaf_with_public_io(
            &partition,
            &run,
            2,
            None,
            d(32),
            d(42),
            Vec::new(),
            vec![output_claim],
        ),
    ];
    assert_eq!(
        verify_native_manifest(
            &tampered,
            FinalArtifactKind::NativeVerifiedManifest,
            &aggregation,
            &partition,
            &run,
            fresh,
        ),
        Err(AggregationError::AggregationIdentityMismatch),
    );
}

#[test]
fn unbound_topology_plan_is_rejected_at_every_manifest_boundary() {
    let partition = chain_plan();
    let aggregation = plan_aggregation(&partition, NonZeroU8::new(4).unwrap()).unwrap();
    let run = run_for(&partition, aggregation.digest());

    assert_eq!(
        admit_verified_leaf(
            &aggregation,
            &partition,
            &run,
            verified_leaf(&partition, &run, 0, None),
        ),
        Err(AggregationError::InvalidLeafExpectations),
    );
    assert_eq!(
        build_native_manifest(&aggregation, &partition, &run, Vec::new()),
        Err(AggregationError::InvalidLeafExpectations),
    );
    assert_eq!(
        verify_native_manifest(
            b"{}",
            FinalArtifactKind::NativeVerifiedManifest,
            &aggregation,
            &partition,
            &run,
            Vec::new(),
        ),
        Err(AggregationError::InvalidLeafExpectations),
    );
}

fn trusted_aggregation(
    partition: &zkie_compiler::dag::PartitionPlan,
    fan_in: u8,
) -> zkie_prover::AggregationPlan {
    plan_aggregation(partition, NonZeroU8::new(fan_in).unwrap())
        .unwrap()
        .bind_leaf_expectations(&leaf_expectations(partition))
        .unwrap()
}

#[test]
fn rejects_an_independently_valid_alternate_leaf_identity() {
    let partition = chain_plan();
    let expectations = leaf_expectations(&partition);
    let aggregation = plan_aggregation(&partition, NonZeroU8::new(4).unwrap())
        .unwrap()
        .bind_leaf_expectations(&expectations)
        .unwrap();
    let run = run_for(&partition, aggregation.digest());
    let expected = verified_leaf(&partition, &run, 0, None);
    let alternate = verified_leaf_with_identity(&partition, &run, 0, None, d(90), d(91));

    assert!(admit_verified_leaf(&aggregation, &partition, &run, expected).is_ok());
    assert_eq!(
        admit_verified_leaf(&aggregation, &partition, &run, alternate),
        Err(AggregationError::LeafIdentityMismatch { shard: 0 })
    );
}

#[test]
fn merge_rejects_siblings_admitted_under_different_complete_run_identities() {
    let partition = chain_plan();
    let aggregation = trusted_aggregation(&partition, 4);
    let run = run_for(&partition, aggregation.digest());
    let mut other_run = run.clone();
    other_run.compiler_digest = d(99);
    let claims = vec![
        admit_verified_leaf(
            &aggregation,
            &partition,
            &run,
            verified_leaf(&partition, &run, 0, None),
        )
        .unwrap(),
        admit_verified_leaf(
            &aggregation,
            &partition,
            &other_run,
            verified_leaf(&partition, &other_run, 1, None),
        )
        .unwrap(),
        admit_verified_leaf(
            &aggregation,
            &partition,
            &run,
            verified_leaf(&partition, &run, 2, None),
        )
        .unwrap(),
    ];

    assert_eq!(
        merge_verified_claims(&aggregation.levels()[0][0], &claims, partition.dag()),
        Err(AggregationError::AggregationIdentityMismatch),
    );
}

#[test]
fn verified_claims_close_each_edge_once_and_publish_only_native_manifest() {
    let partition = chain_plan();
    let aggregation = trusted_aggregation(&partition, 4);
    let run = run_for(&partition, aggregation.digest());
    let claims = (0..3)
        .map(|shard| {
            admit_verified_leaf(
                &aggregation,
                &partition,
                &run,
                verified_leaf(&partition, &run, shard, None),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let root =
        merge_verified_claims(&aggregation.levels()[0][0], &claims, partition.dag()).unwrap();
    assert_eq!(root.closed_edge_count(), partition.dag().edges.len());
    assert_eq!(root.frontier_count(), 0);
    let manifest = finalize_native_manifest(&aggregation, &partition, &run, root).unwrap();
    assert_eq!(manifest.kind(), FinalArtifactKind::NativeVerifiedManifest);
    let encoded = manifest.encode().unwrap();
    let fresh_proofs = (0..3)
        .map(|shard| verified_leaf(&partition, &run, shard, None))
        .collect();
    assert_eq!(
        verify_native_manifest(
            &encoded,
            FinalArtifactKind::NativeVerifiedManifest,
            &aggregation,
            &partition,
            &run,
            fresh_proofs,
        )
        .unwrap()
        .digest(),
        manifest.digest(),
    );
    assert_eq!(
        manifest
            .require_kind(FinalArtifactKind::RecursiveRootProof)
            .unwrap_err(),
        AggregationError::RecursiveProofUnavailable,
    );
}

#[test]
fn native_manifest_rejects_missing_leaf_changed_fan_in_and_recursive_request() {
    let partition = chain_plan();
    let aggregation = trusted_aggregation(&partition, 4);
    let run = run_for(&partition, aggregation.digest());
    let incomplete = (0..2)
        .map(|shard| verified_leaf(&partition, &run, shard, None))
        .collect();
    assert_eq!(
        build_native_manifest(&aggregation, &partition, &run, incomplete).unwrap_err(),
        AggregationError::MissingLeaf { shard: 2 },
    );
    let proofs = (0..3)
        .map(|shard| verified_leaf(&partition, &run, shard, None))
        .collect();
    let changed = trusted_aggregation(&partition, 2);
    assert!(matches!(
        build_native_manifest(&changed, &partition, &run, proofs),
        Err(AggregationError::LeafIdentityMismatch { .. })
            | Err(AggregationError::AggregationIdentityMismatch)
    ));
    assert_eq!(
        verify_native_manifest(
            b"{}",
            FinalArtifactKind::RecursiveRootProof,
            &aggregation,
            &partition,
            &run,
            Vec::new(),
        )
        .unwrap_err(),
        AggregationError::RecursiveProofUnavailable,
    );
}

#[test]
fn leaf_admission_rejects_run_with_non_leaf_statement_schema() {
    let partition = chain_plan();
    let aggregation = trusted_aggregation(&partition, 4);
    let mut run = run_for(&partition, aggregation.digest());
    run.public_input_schema_version = 1;
    let proof = verified_leaf(&partition, &run, 0, None);

    assert_eq!(
        admit_verified_leaf(&aggregation, &partition, &run, proof),
        Err(AggregationError::LeafIdentityMismatch { shard: 0 })
    );
}

#[test]
fn native_aggregation_rejects_private_model_runs_until_weight_commitments_exist() {
    let partition = chain_plan();
    let aggregation = trusted_aggregation(&partition, 4);
    let mut run = run_for(&partition, aggregation.digest());
    run.model_visibility = ModelVisibility::PrivateModel;
    let proof = verified_leaf(&partition, &run, 0, None);

    assert_eq!(
        admit_verified_leaf(&aggregation, &partition, &run, proof),
        Err(AggregationError::UnsupportedModelVisibility),
    );
}

#[test]
fn merge_rejects_corrupt_edge_and_child_reordering() {
    let partition = chain_plan();
    let aggregation = trusted_aggregation(&partition, 4);
    let run = run_for(&partition, aggregation.digest());
    let edge = partition.dag().edges[0].id.as_str();
    let mut claims = (0..3)
        .map(|shard| {
            let corrupt = (shard == partition.dag().edges[0].consumer).then_some(edge);
            admit_verified_leaf(
                &aggregation,
                &partition,
                &run,
                verified_leaf(&partition, &run, shard, corrupt),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        merge_verified_claims(&aggregation.levels()[0][0], &claims, partition.dag()),
        Err(AggregationError::BoundaryCommitmentMismatch { .. })
    ));
    claims.swap(0, 1);
    assert!(matches!(
        merge_verified_claims(&aggregation.levels()[0][0], &claims, partition.dag()),
        Err(AggregationError::ChildOrderMismatch { .. })
    ));
}
