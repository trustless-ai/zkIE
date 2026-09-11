use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::num::NonZeroU8;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use zkie_compiler::dag::{
    InstructionCostModel, InstructionEstimate, PartitionError, PartitionPlanner, PartitionRequest,
    ShardEstimate,
};
use zkie_compiler::graph_compiler::{compile_model, CompiledInstruction, Register};
use zkie_compiler::onnx::{
    tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto, TensorShapeProto, TypeProto,
    ValueInfoProto,
};
use zkie_compiler::shard_binding::{
    bind_shard_program, BoundShardProgram, BoundaryDescriptor as CompilerBoundaryDescriptor,
    BoundaryLayout, BoundaryTensorSpec,
};
use zkie_core::chips::poseidon_boundary::{BoundaryDescriptor, BoundaryRole};
use zkie_core::fixed_point::{I18, SCALE_18};
use zkie_prover::{
    build_native_manifest, plan_aggregation, verify_proof, AggregationError, BackendError,
    BoundaryShapeManifest, CpuWitnessInputs, Digest32, FinalArtifactKind, Halo2KzgCpuBackend,
    KeyIdentity, KeyMaterialStore, KeyWriteOutcome, LeafExpectation, LeafExpectationSet,
    LeafStatement, ModelVisibility, PrepareJob, ProofBackend, ProofFlavorId, ProofJob, RunIdentity,
    SrsPolicy, VerificationExpectation, VerificationJob, VerifiedProof, WitnessBackend, WitnessJob,
    ZkieIsaCpuWitnessBackend, LEAF_STATEMENT_SCHEMA_VERSION,
};

const ONNX_FLOAT: i32 = 1;
const CIRCUIT_K: u32 = 13;

#[derive(Clone)]
struct UnitCost;

impl InstructionCostModel for UnitCost {
    fn identity(&self) -> &str {
        "sharded-kzg-test-unit"
    }

    fn version(&self) -> u32 {
        1
    }

    fn estimate_instruction(
        &self,
        _: &CompiledInstruction,
    ) -> Result<InstructionEstimate, PartitionError> {
        InstructionEstimate::new(1, CIRCUIT_K)
    }

    fn estimate_shard(
        &self,
        instructions: &[CompiledInstruction],
    ) -> Result<ShardEstimate, PartitionError> {
        ShardEstimate::new(instructions.len() as u64, CIRCUIT_K)
    }
}

#[derive(Default)]
struct MemoryKeyStore(Mutex<HashMap<KeyIdentity, Vec<u8>>>);

impl KeyMaterialStore for MemoryKeyStore {
    fn read(&self, identity: &KeyIdentity) -> Result<Option<Vec<u8>>, BackendError> {
        Ok(self.0.lock().unwrap().get(identity).cloned())
    }

    fn write_if_absent(
        &self,
        identity: &KeyIdentity,
        bytes: &[u8],
    ) -> Result<KeyWriteOutcome, BackendError> {
        let mut values = self.0.lock().unwrap();
        match values.get(identity) {
            Some(existing) if existing == bytes => Ok(KeyWriteOutcome::AlreadyPresentIdentical),
            Some(_) => Err(BackendError::KeyConflict),
            None => {
                values.insert(identity.clone(), bytes.to_vec());
                Ok(KeyWriteOutcome::Inserted)
            }
        }
    }
}

fn digest(byte: u8) -> Digest32 {
    Digest32::new([byte; 32])
}

fn node(name: &str, inputs: &[&str], output: &str) -> NodeProto {
    NodeProto {
        name: name.into(),
        op_type: "Add".into(),
        input: inputs.iter().map(|input| (*input).into()).collect(),
        output: vec![output.into()],
        ..Default::default()
    }
}

fn value(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: ONNX_FLOAT,
                shape: Some(TensorShapeProto {
                    dim: vec![tensor_shape_proto::Dimension {
                        value: Some(tensor_shape_proto::dimension::Value::DimValue(2)),
                        ..Default::default()
                    }],
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn supported_broadcast_model() -> ModelProto {
    ModelProto {
        graph: Some(GraphProto {
            input: vec![value("x"), value("y")],
            output: vec![value("z")],
            value_info: vec![value("broadcast"), value("middle")],
            node: vec![
                node("produce-broadcast", &["x", "y"], "broadcast"),
                node("consume-first", &["y", "broadcast"], "middle"),
                node("consume-broadcast-again", &["broadcast", "middle"], "z"),
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-sharded-kzg-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    if path.exists() {
        fs::remove_dir_all(&path).unwrap();
    }
    fs::create_dir(&path).unwrap();
    path
}

fn raw_json(values: &[I18]) -> Vec<i64> {
    values.iter().map(|value| value.raw()).collect()
}

fn direct_graph_inputs(
    instructions: &[CompiledInstruction],
    values: &HashMap<String, Vec<I18>>,
) -> HashMap<String, Vec<I18>> {
    instructions
        .iter()
        .flat_map(|instruction| instruction.inputs.iter())
        .filter_map(|register| match register {
            Register::GraphInput(name) => Some((name.clone(), values[name].clone())),
            _ => None,
        })
        .collect()
}

fn run_identity(
    partition: &zkie_compiler::dag::PartitionPlan,
    aggregation_digest: Digest32,
) -> RunIdentity {
    RunIdentity {
        model_graph_digest: partition.model_digest(),
        weights_digest: digest(2),
        compiler_digest: digest(3),
        isa_digest: digest(4),
        quantization_digest: digest(5),
        partition_plan_digest: partition.digest(),
        aggregation_plan_digest: aggregation_digest,
        proof_flavor: ProofFlavorId::parse("halo2-kzg-bn256-shplonk-v1").unwrap(),
        model_visibility: ModelVisibility::PublicModel,
        aggregation_fan_in: 4,
        public_input_schema_version: LEAF_STATEMENT_SCHEMA_VERSION,
    }
}

fn boundary_shapes(
    direct: &HashMap<String, Vec<I18>>,
    incoming: &HashMap<Register, Vec<I18>>,
) -> BoundaryShapeManifest {
    BoundaryShapeManifest::new(
        direct
            .iter()
            .map(|(name, values)| (name.clone(), values.len()))
            .collect(),
        incoming
            .iter()
            .map(|(register, values)| match register {
                Register::Virtual(index) => (*index, values.len()),
                _ => unreachable!(),
            })
            .collect(),
    )
    .unwrap()
}

fn concrete_inputs(
    direct: HashMap<String, Vec<I18>>,
    incoming: &HashMap<Register, Vec<I18>>,
) -> CpuWitnessInputs {
    CpuWitnessInputs::new(
        direct,
        incoming
            .iter()
            .map(|(register, values)| match register {
                Register::Virtual(index) => (*index, values.clone()),
                _ => unreachable!(),
            })
            .collect(),
    )
}

struct PreparedLeaf {
    bound: BoundShardProgram,
    direct: HashMap<String, Vec<I18>>,
    incoming: HashMap<Register, Vec<I18>>,
    key: KeyIdentity,
    circuit_digest: Digest32,
}

fn public_descriptor(
    descriptor: &CompilerBoundaryDescriptor,
    role: BoundaryRole,
) -> BoundaryDescriptor {
    let register_id = match descriptor.register() {
        Register::GraphInput(name) => format!("graph-input:{name}"),
        Register::Virtual(index) => format!("virtual:{index}"),
        Register::Weight(_) => unreachable!("bound shard boundaries cannot be weights"),
    };
    BoundaryDescriptor::flat_i18(
        role,
        register_id,
        descriptor
            .edge_ids()
            .iter()
            .map(|edge| edge.as_str().to_owned())
            .collect(),
        descriptor.graph_output_names().to_vec(),
        descriptor.element_count(),
        SCALE_18 as u64,
    )
    .unwrap()
}

fn trusted_public_io(
    bound: &BoundShardProgram,
) -> (Vec<BoundaryDescriptor>, Vec<BoundaryDescriptor>) {
    let inputs = bound
        .input_boundaries()
        .iter()
        .filter(|descriptor| matches!(descriptor.register(), Register::GraphInput(_)))
        .map(|descriptor| public_descriptor(descriptor, BoundaryRole::Input))
        .collect();
    let outputs = bound
        .output_boundaries()
        .iter()
        .filter(|descriptor| !descriptor.graph_output_names().is_empty())
        .map(|descriptor| public_descriptor(descriptor, BoundaryRole::Output))
        .collect();
    (inputs, outputs)
}

fn prove_and_independently_verify(
    dir: &Path,
    label: &str,
    cpu: Arc<ZkieIsaCpuWitnessBackend>,
    bound: &zkie_compiler::shard_binding::BoundShardProgram,
    concrete: CpuWitnessInputs,
    key: &KeyIdentity,
    store: &MemoryKeyStore,
) -> (
    VerifiedProof,
    VerifiedProof,
    VerifiedProof,
    zkie_prover::CpuWitnessArtifact,
    u64,
) {
    let input_path = dir.join(format!("{label}-input.json"));
    fs::write(
        &input_path,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "graph_inputs": concrete
                .graph_inputs()
                .iter()
                .map(|(name, values)| (name, raw_json(values)))
                .collect::<HashMap<_, _>>(),
            "virtual_inputs": concrete
                .virtual_inputs()
                .iter()
                .map(|(index, values)| (index, raw_json(values)))
                .collect::<BTreeMap<_, _>>(),
        }))
        .unwrap(),
    )
    .unwrap();
    let witness = cpu
        .generate(
            WitnessJob::new(
                cpu.run_identity().clone(),
                cpu.shard_identity().clone(),
                input_path,
                cpu.circuit_digest(),
            )
            .unwrap(),
            dir.join(format!("{label}-witness.json")),
        )
        .unwrap();
    let executed = cpu.load_artifact(&witness).unwrap();

    let prover = Halo2KzgCpuBackend::new_with_boundary_commitments(
        Arc::clone(&cpu),
        bound,
        HashMap::new(),
        CIRCUIT_K,
        SrsPolicy::DevelopmentGenerate,
    )
    .unwrap();
    let keys = prover
        .prepare(
            PrepareJob::new_development(
                cpu.run_identity().clone(),
                cpu.shard_identity().clone(),
                CIRCUIT_K,
                key.clone(),
            )
            .unwrap(),
            store,
        )
        .unwrap();
    let proof_job = ProofJob::new(
        cpu.run_identity().clone(),
        cpu.shard_identity().clone(),
        witness.clone(),
        keys.clone(),
    )
    .unwrap();
    let unverified = prover
        .prove(proof_job.clone(), dir.join(format!("{label}-proof.bin")))
        .unwrap();
    let proof_bytes = fs::read(unverified.proof_path()).unwrap();
    let proof_size = proof_bytes.len() as u64;

    let verifier = Halo2KzgCpuBackend::new_with_boundary_commitments(
        cpu,
        bound,
        HashMap::new(),
        CIRCUIT_K,
        SrsPolicy::DevelopmentGenerate,
    )
    .unwrap();
    let verifier_keys = verifier
        .prepare(
            PrepareJob::new_development(
                proof_job.run_identity().clone(),
                proof_job.shard().clone(),
                CIRCUIT_K,
                keys.verification_key().clone(),
            )
            .unwrap(),
            store,
        )
        .unwrap();
    let verifier_job = ProofJob::new(
        proof_job.run_identity().clone(),
        proof_job.shard().clone(),
        witness,
        verifier_keys,
    )
    .unwrap();
    let expectation = VerificationExpectation::from_proof_job(
        &verifier_job,
        verifier.capabilities().execution_backend().clone(),
        unverified.public_statement().to_vec(),
        unverified.proof_digest(),
        unverified.artifact_manifest().to_vec(),
    )
    .unwrap();
    let verified = verify_proof(
        &verifier,
        VerificationJob::new(expectation, unverified.clone(), proof_bytes.clone()).unwrap(),
    )
    .unwrap();
    let second_expectation = VerificationExpectation::from_proof_job(
        &verifier_job,
        verifier.capabilities().execution_backend().clone(),
        unverified.public_statement().to_vec(),
        unverified.proof_digest(),
        unverified.artifact_manifest().to_vec(),
    )
    .unwrap();
    let second_verified = verify_proof(
        &verifier,
        VerificationJob::new(second_expectation, unverified.clone(), proof_bytes.clone()).unwrap(),
    )
    .unwrap();
    let third_expectation = VerificationExpectation::from_proof_job(
        &verifier_job,
        verifier.capabilities().execution_backend().clone(),
        unverified.public_statement().to_vec(),
        unverified.proof_digest(),
        unverified.artifact_manifest().to_vec(),
    )
    .unwrap();
    let third_verified = verify_proof(
        &verifier,
        VerificationJob::new(third_expectation, unverified, proof_bytes).unwrap(),
    )
    .unwrap();
    (
        verified,
        second_verified,
        third_verified,
        executed,
        proof_size,
    )
}

#[test]
fn supported_broadcast_graph_proves_every_shard_and_rejects_wrong_composition() {
    let compiled = Arc::new(compile_model(&supported_broadcast_model()).unwrap());
    let layout = BoundaryLayout::new(
        [
            Register::GraphInput("x".into()),
            Register::GraphInput("y".into()),
        ]
        .into_iter()
        .chain((0..3).map(Register::Virtual))
        .map(|register| BoundaryTensorSpec::flat(register, 2).unwrap())
        .collect(),
    )
    .unwrap();
    let request = PartitionRequest::new(1, CIRCUIT_K, digest(1), "compiler-v1", UnitCost)
        .unwrap()
        .with_boundary_layout_digest(layout.digest());
    let partition = PartitionPlanner::plan(&compiled, &request).unwrap();
    let repeated = PartitionPlanner::plan(&compiled, &request).unwrap();
    assert_eq!(partition, repeated);
    assert_eq!(partition.shards().len(), 3);
    assert_eq!(partition.dag().edges.len(), 3);
    assert_eq!(
        partition
            .dag()
            .edges
            .iter()
            .filter(|edge| edge.register == Register::Virtual(0))
            .count(),
        2
    );

    let unbound_aggregation = plan_aggregation(&partition, NonZeroU8::new(4).unwrap()).unwrap();
    assert_eq!(unbound_aggregation.levels().len(), 1);
    assert_eq!(unbound_aggregation.levels()[0][0].actual_arity(), 3);
    let provisional_run = run_identity(&partition, unbound_aggregation.digest());
    let graph_values = HashMap::from([
        ("x".into(), vec![I18::from_raw(10), I18::from_raw(20)]),
        ("y".into(), vec![I18::from_raw(30), I18::from_raw(40)]),
    ]);
    let dir = temp_dir();
    let store = MemoryKeyStore::default();
    let mut virtuals = BTreeMap::<usize, Vec<I18>>::new();
    let mut prepared = Vec::new();
    let mut expectations = Vec::new();

    // Circuit and verification-key identities exist before the final aggregation digest. Prepare
    // those trusted identities first, then bind them into the N=4 plan.
    for (shard_index, shard) in partition.dag().shards.iter().enumerate() {
        let direct =
            direct_graph_inputs(&compiled.instructions[shard.range.clone()], &graph_values);
        let incoming = shard
            .inputs
            .iter()
            .map(|register| match register {
                Register::Virtual(index) => (register.clone(), virtuals[index].clone()),
                _ => unreachable!(),
            })
            .collect::<HashMap<_, _>>();
        let bound = bind_shard_program(
            &compiled,
            &partition,
            shard_index,
            partition.model_digest(),
            &layout,
            &graph_values,
            &incoming,
        )
        .unwrap();
        let cpu = Arc::new(
            ZkieIsaCpuWitnessBackend::new(
                Arc::clone(&compiled),
                shard.clone(),
                HashMap::new(),
                provisional_run.clone(),
                boundary_shapes(&direct, &incoming),
            )
            .unwrap(),
        );
        let executed = cpu
            .execute(&concrete_inputs(direct.clone(), &incoming))
            .unwrap();
        for (index, values) in executed.outputs() {
            virtuals.insert(*index, values.clone());
        }
        let backend = Halo2KzgCpuBackend::new_with_boundary_commitments(
            Arc::clone(&cpu),
            &bound,
            HashMap::new(),
            CIRCUIT_K,
            SrsPolicy::DevelopmentGenerate,
        )
        .unwrap();
        let keys = backend
            .prepare(
                PrepareJob::new_development_generate(
                    provisional_run.clone(),
                    cpu.shard_identity().clone(),
                    CIRCUIT_K,
                )
                .unwrap(),
                &store,
            )
            .unwrap();
        let (public_inputs, public_outputs) = trusted_public_io(&bound);
        expectations.push(
            LeafExpectation::new(
                shard.id as u64,
                shard.name.clone(),
                cpu.circuit_digest(),
                keys.verification_key().key_digest(),
                backend.capabilities().execution_backend().clone(),
                provisional_run.proof_flavor.clone(),
            )
            .unwrap()
            .with_public_io(public_inputs, public_outputs)
            .unwrap(),
        );
        prepared.push(PreparedLeaf {
            bound,
            direct,
            incoming,
            key: keys.verification_key().clone(),
            circuit_digest: cpu.circuit_digest(),
        });
    }
    let expectation_set = LeafExpectationSet::new(&partition, expectations).unwrap();
    let aggregation = unbound_aggregation
        .bind_leaf_expectations(&expectation_set)
        .unwrap();
    assert_eq!(
        aggregation.leaf_expectation_set_digest(),
        Some(expectation_set.digest())
    );
    let run = run_identity(&partition, aggregation.digest());

    let mut root_proofs = Vec::new();
    let mut invalid_proofs = Vec::new();
    let mut wrong_consumer = None;
    let mut proof_sizes = Vec::new();

    for (shard_index, (shard, prepared)) in partition
        .dag()
        .shards
        .iter()
        .zip(prepared.iter())
        .enumerate()
    {
        let cpu = Arc::new(
            ZkieIsaCpuWitnessBackend::new(
                Arc::clone(&compiled),
                shard.clone(),
                HashMap::new(),
                run.clone(),
                boundary_shapes(&prepared.direct, &prepared.incoming),
            )
            .unwrap(),
        );
        assert_eq!(cpu.circuit_digest(), prepared.circuit_digest);
        let concrete = concrete_inputs(prepared.direct.clone(), &prepared.incoming);
        let (verified, root_copy, invalid_copy, executed, proof_size) =
            prove_and_independently_verify(
                &dir,
                &format!("shard-{shard_index}"),
                Arc::clone(&cpu),
                &prepared.bound,
                concrete,
                &prepared.key,
                &store,
            );
        root_proofs.push(root_copy);
        if shard_index < 2 {
            invalid_proofs.push(invalid_copy);
        }
        proof_sizes.push(proof_size);
        for (index, values) in executed.outputs() {
            virtuals.insert(*index, values.clone());
        }

        let statement = LeafStatement::decode(verified.public_statement()).unwrap();
        if shard_index == 0 {
            assert!(statement.input_claims().iter().any(|claim| {
                claim.descriptor().register_id() == "graph-input:x"
                    && claim.descriptor().edge_ids().is_empty()
            }));
        }
        if shard_index == 2 {
            assert!(statement.output_claims().iter().any(|claim| {
                claim.descriptor().register_id() == "virtual:2"
                    && claim.descriptor().graph_output_names() == ["z"]
            }));
            let mut wrong_incoming = prepared.incoming.clone();
            wrong_incoming.get_mut(&Register::Virtual(0)).unwrap()[0] = I18::from_raw(999);
            let wrong_bound = bind_shard_program(
                &compiled,
                &partition,
                shard_index,
                partition.model_digest(),
                &layout,
                &graph_values,
                &wrong_incoming,
            )
            .unwrap();
            let wrong_concrete = CpuWitnessInputs::new(
                prepared.direct.clone(),
                wrong_incoming
                    .iter()
                    .map(|(register, values)| match register {
                        Register::Virtual(index) => (*index, values.clone()),
                        _ => unreachable!(),
                    })
                    .collect(),
            );
            wrong_consumer = Some(
                prove_and_independently_verify(
                    &dir,
                    "shard-2-wrong-input",
                    cpu,
                    &wrong_bound,
                    wrong_concrete,
                    &prepared.key,
                    &store,
                )
                .0,
            );
        }
    }

    let claims = root_proofs
        .into_iter()
        .map(|proof| {
            zkie_prover::admit_verified_leaf(&aggregation, &partition, &run, proof).unwrap()
        })
        .collect::<Vec<_>>();
    let root =
        zkie_prover::merge_verified_claims(&aggregation.levels()[0][0], &claims, partition.dag())
            .unwrap();
    assert_eq!(root.closed_edge_count(), partition.dag().edges.len());
    assert_eq!(root.frontier_count(), 0);
    let manifest =
        zkie_prover::finalize_native_manifest(&aggregation, &partition, &run, root).unwrap();
    assert_eq!(manifest.kind(), FinalArtifactKind::NativeVerifiedManifest);
    assert_eq!(manifest.public_input_count(), 2);
    assert_eq!(manifest.public_output_count(), 1);
    let manifest_bytes = manifest.encode().unwrap();
    let manifest_json: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(manifest_json["leaf_count"], 3);
    assert!(proof_sizes.iter().all(|size| *size > 0));
    println!(
        "shards=3 proof_bytes={proof_sizes:?} manifest_bytes={}",
        manifest_bytes.len()
    );

    invalid_proofs.push(wrong_consumer.unwrap());
    assert!(matches!(
        build_native_manifest(&aggregation, &partition, &run, invalid_proofs),
        Err(AggregationError::BoundaryCommitmentMismatch { .. })
    ));

    fs::remove_dir_all(dir).unwrap();
}
