use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use zkie_compiler::circuit_binding::to_assembler_program;
use zkie_compiler::dag::Shard;
use zkie_compiler::graph_compiler::{CompiledInstruction, CompiledProgram, Register};
use zkie_compiler::onnx_parser::WeightTensor;
use zkie_core::fixed_point::I18;
use zkie_core::isa::{EltwiseOp, Instruction};
use zkie_prover::{
    verify_proof, BackendError, BoundaryShapeManifest, Digest32, Halo2KzgCpuBackend, KeyIdentity,
    KeyMaterialStore, KeyWriteOutcome, ModelVisibility, PrepareJob, ProofBackend, ProofFlavorId,
    ProofJob, RunIdentity, ShardIdentity, SrsPolicy, VerificationError, VerificationExpectation,
    VerificationJob, WitnessBackend, WitnessJob, ZkieIsaCpuWitnessBackend,
};

#[derive(Default)]
struct Store(Mutex<HashMap<KeyIdentity, Vec<u8>>>);

impl KeyMaterialStore for Store {
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

fn d(byte: u8) -> Digest32 {
    Digest32::new([byte; 32])
}

fn run(visibility: ModelVisibility) -> RunIdentity {
    RunIdentity {
        model_graph_digest: d(1),
        weights_digest: d(2),
        compiler_digest: d(3),
        isa_digest: d(4),
        quantization_digest: d(5),
        partition_plan_digest: d(6),
        aggregation_plan_digest: d(7),
        proof_flavor: ProofFlavorId::parse("halo2-kzg-bn256-shplonk-v1").unwrap(),
        model_visibility: visibility,
        aggregation_fan_in: 2,
        public_input_schema_version: 1,
    }
}

fn fixture(
    run_identity: RunIdentity,
) -> (
    Arc<ZkieIsaCpuWitnessBackend>,
    zkie_core::assembler::AssemblerProgram,
    HashMap<(usize, u64), zkie_core::chips::layer_norm::RsqrtDomain>,
) {
    let compiled = CompiledProgram {
        instructions: vec![CompiledInstruction {
            instruction: Instruction::Eltwise { op: EltwiseOp::Add },
            inputs: vec![
                Register::GraphInput("x".into()),
                Register::Weight("b".into()),
            ],
            output_name: "y".into(),
        }],
        weights: HashMap::from([(
            "b".into(),
            WeightTensor {
                shape: vec![2],
                data: vec![0.25, -0.5],
            },
        )]),
        graph_inputs: vec!["x".into()],
        graph_outputs: vec![("y".into(), Register::Virtual(0))],
    };
    let shard = Shard {
        id: 9,
        name: "leaf-9".into(),
        range: 0..1,
        inputs: vec![],
        outputs: vec![Register::Virtual(0)],
    };
    let inputs = HashMap::from([(
        "x".into(),
        vec![I18::from_f64(1.0).unwrap(), I18::from_f64(2.0).unwrap()],
    )]);
    let assembler = to_assembler_program(&compiled, &inputs).unwrap();
    let cpu = Arc::new(
        ZkieIsaCpuWitnessBackend::new(
            Arc::new(compiled),
            shard,
            HashMap::new(),
            run_identity,
            BoundaryShapeManifest::new(BTreeMap::from([("x".into(), 2)]), BTreeMap::new()).unwrap(),
        )
        .unwrap(),
    );
    (cpu, assembler, HashMap::new())
}

fn temp_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "zkie-halo2-{label}-{}-{}",
        std::process::id(),
        rand_core::OsRng.next_u64()
    ));
    fs::create_dir(&path).unwrap();
    path
}

use rand_core::RngCore;

fn write_witness(cpu: &ZkieIsaCpuWitnessBackend, dir: &Path) -> zkie_prover::WitnessArtifact {
    write_witness_values(
        cpu,
        dir,
        "witness.json",
        [1_000_000_000_000_000_000, 2_000_000_000_000_000_000],
    )
}

fn write_witness_values(
    cpu: &ZkieIsaCpuWitnessBackend,
    dir: &Path,
    output: &str,
    values: [i64; 2],
) -> zkie_prover::WitnessArtifact {
    let input = dir.join("input.json");
    fs::write(
        &input,
        serde_json::json!({"schema_version":1,"graph_inputs":{"x":values},"virtual_inputs":{}})
            .to_string(),
    )
    .unwrap();
    let job = WitnessJob::new(
        cpu.run_identity().clone(),
        cpu.shard_identity().clone(),
        input,
        cpu.circuit_digest(),
    )
    .unwrap();
    cpu.generate(job, dir.join(output)).unwrap()
}

fn ready(label: &str) -> (PathBuf, Halo2KzgCpuBackend, ProofJob) {
    let dir = temp_dir(label);
    let (cpu, assembler, domains) = fixture(run(ModelVisibility::PublicModel));
    let witness = write_witness(&cpu, &dir);
    let backend = Halo2KzgCpuBackend::new(
        cpu.clone(),
        assembler,
        domains,
        13,
        SrsPolicy::DevelopmentGenerate,
    )
    .unwrap();
    let prepare = PrepareJob::new_development_generate(
        cpu.run_identity().clone(),
        cpu.shard_identity().clone(),
        13,
    )
    .unwrap();
    let keys = backend.prepare(prepare, &Store::default()).unwrap();
    let job = ProofJob::new(
        cpu.run_identity().clone(),
        cpu.shard_identity().clone(),
        witness,
        keys,
    )
    .unwrap();
    (dir, backend, job)
}

#[test]
fn development_roundtrip_rereads_artifact_and_rejects_tampering() {
    let dir = temp_dir("roundtrip");
    let (cpu, assembler, domains) = fixture(run(ModelVisibility::PublicModel));
    let witness = write_witness(&cpu, &dir);
    let backend = Halo2KzgCpuBackend::new(
        cpu.clone(),
        assembler,
        domains,
        13,
        SrsPolicy::DevelopmentGenerate,
    )
    .unwrap();
    let prepare = PrepareJob::new_development_generate(
        cpu.run_identity().clone(),
        cpu.shard_identity().clone(),
        13,
    )
    .unwrap();
    let keys = backend.prepare(prepare, &Store::default()).unwrap();
    let proof_job = ProofJob::new(
        cpu.run_identity().clone(),
        cpu.shard_identity().clone(),
        witness,
        keys,
    )
    .unwrap();
    let proof = backend
        .prove(proof_job.clone(), dir.join("proof.bin"))
        .unwrap();
    fs::remove_file(proof_job.witness().path()).unwrap();
    let expected = VerificationExpectation::from_proof_job(
        &proof_job,
        backend.capabilities().execution_backend().clone(),
        proof.public_statement().to_vec(),
        proof.proof_digest(),
        proof.artifact_manifest().to_vec(),
    )
    .unwrap();
    let bytes = fs::read(proof.proof_path()).unwrap();
    verify_proof(
        &backend,
        VerificationJob::new(expected, proof.clone(), bytes.clone()).unwrap(),
    )
    .unwrap();

    let mut appended = bytes;
    appended.extend_from_slice(b"trailing");
    let appended_digest = Digest32::new(*blake3::hash(&appended).as_bytes());
    let appended_proof = zkie_prover::UnverifiedProof::new(
        proof.proof_path().clone(),
        appended_digest,
        proof.public_statement().to_vec(),
        proof.circuit_digest(),
        proof.verification_key_digest(),
        proof.proof_flavor().clone(),
        proof.execution_backend().clone(),
        proof.shard().clone(),
        proof.artifact_manifest().to_vec(),
        proof.run_identity_digest(),
    )
    .unwrap();
    let appended_expected = VerificationExpectation::from_proof_job(
        &proof_job,
        backend.capabilities().execution_backend().clone(),
        proof.public_statement().to_vec(),
        appended_digest,
        proof.artifact_manifest().to_vec(),
    )
    .unwrap();
    assert!(matches!(
        verify_proof(
            &backend,
            VerificationJob::new(appended_expected, appended_proof, appended).unwrap()
        ),
        Err(VerificationError::CryptographicVerificationFailed { .. })
    ));

    let mut tampered = fs::read(proof.proof_path()).unwrap();
    let middle = tampered.len() / 2;
    tampered[middle] ^= 0xff;
    fs::write(proof.proof_path(), &tampered).unwrap();
    assert!(matches!(
        backend.verify(&proof),
        Err(VerificationError::ProofDigestMismatch)
    ));
    let tampered_digest = Digest32::new(*blake3::hash(&tampered).as_bytes());
    let tampered_proof = zkie_prover::UnverifiedProof::new(
        proof.proof_path().clone(),
        tampered_digest,
        proof.public_statement().to_vec(),
        proof.circuit_digest(),
        proof.verification_key_digest(),
        proof.proof_flavor().clone(),
        proof.execution_backend().clone(),
        proof.shard().clone(),
        proof.artifact_manifest().to_vec(),
        proof.run_identity_digest(),
    )
    .unwrap();
    let tampered_expected = VerificationExpectation::from_proof_job(
        &proof_job,
        backend.capabilities().execution_backend().clone(),
        proof.public_statement().to_vec(),
        tampered_digest,
        proof.artifact_manifest().to_vec(),
    )
    .unwrap();
    assert!(matches!(
        verify_proof(
            &backend,
            VerificationJob::new(tampered_expected, tampered_proof, tampered).unwrap()
        ),
        Err(VerificationError::CryptographicVerificationFailed { .. })
    ));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn private_model_and_development_srs_for_production_are_rejected() {
    assert!(matches!(
        Halo2KzgCpuBackend::validate_model_visibility(&run(ModelVisibility::PrivateModel)),
        Err(BackendError::UnsupportedModelVisibility)
    ));

    let (cpu, assembler, domains) = fixture(run(ModelVisibility::PublicModel));
    let backend = Halo2KzgCpuBackend::new(
        cpu.clone(),
        assembler,
        domains,
        13,
        SrsPolicy::DevelopmentGenerate,
    )
    .unwrap();
    let production = PrepareJob::new(
        cpu.run_identity().clone(),
        cpu.shard_identity().clone(),
        13,
        KeyIdentity::new(
            cpu.run_identity().proof_flavor.clone(),
            cpu.circuit_digest(),
            d(77),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        backend.prepare(production, &Store::default()),
        Err(BackendError::NonProductionSrs)
    ));
}

#[test]
fn production_srs_missing_or_wrong_digest_fails_closed() {
    let dir = temp_dir("srs");
    let (cpu, assembler, domains) = fixture(run(ModelVisibility::PublicModel));
    assert!(matches!(
        Halo2KzgCpuBackend::new(
            cpu.clone(),
            assembler.clone(),
            domains.clone(),
            8,
            SrsPolicy::ProductionExisting {
                path: dir.join("missing.srs"),
                expected_source_digest: d(42)
            }
        ),
        Err(BackendError::MissingSrs { .. })
    ));
    let path = dir.join("bad.srs");
    fs::write(&path, b"not an srs").unwrap();
    assert!(matches!(
        Halo2KzgCpuBackend::new(
            cpu.clone(),
            assembler.clone(),
            domains.clone(),
            8,
            SrsPolicy::ProductionExisting {
                path,
                expected_source_digest: d(42)
            }
        ),
        Err(BackendError::SrsDigestMismatch)
    ));
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let target = dir.join("target.srs");
        fs::write(&target, b"untrusted").unwrap();
        let link = dir.join("linked.srs");
        symlink(&target, &link).unwrap();
        assert!(matches!(
            Halo2KzgCpuBackend::new(
                fixture(run(ModelVisibility::PublicModel)).0,
                assembler,
                HashMap::new(),
                8,
                SrsPolicy::ProductionExisting {
                    path: link,
                    expected_source_digest: Digest32::new(*blake3::hash(b"untrusted").as_bytes())
                }
            ),
            Err(BackendError::Io { .. })
        ));
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn oversized_k_is_rejected_before_srs_allocation() {
    let (cpu, assembler, domains) = fixture(run(ModelVisibility::PublicModel));
    assert!(matches!(
        Halo2KzgCpuBackend::new(cpu, assembler, domains, 31, SrsPolicy::DevelopmentGenerate),
        Err(BackendError::InvalidJob { .. })
    ));
}

#[test]
fn unsupported_public_input_schema_is_rejected() {
    let mut identity = run(ModelVisibility::PublicModel);
    identity.public_input_schema_version = 2;
    let (cpu, assembler, domains) = fixture(identity);
    assert!(matches!(
        Halo2KzgCpuBackend::new(cpu, assembler, domains, 13, SrsPolicy::DevelopmentGenerate),
        Err(BackendError::InvalidJob { .. })
    ));
}

#[test]
fn constructor_rejects_a_template_for_different_circuit_semantics() {
    let (cpu, mut assembler, domains) = fixture(run(ModelVisibility::PublicModel));
    assembler.instructions[0].instruction = Instruction::Eltwise { op: EltwiseOp::Mul };
    assert!(matches!(
        Halo2KzgCpuBackend::new(cpu, assembler, domains, 13, SrsPolicy::DevelopmentGenerate),
        Err(BackendError::InvalidJob { .. })
    ));
}

#[test]
fn boundary_shape_manifest_changes_circuit_identity_and_rejects_the_other_template() {
    let make = |len: usize| {
        let compiled = CompiledProgram {
            instructions: vec![CompiledInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![
                    Register::GraphInput("a".into()),
                    Register::GraphInput("b".into()),
                ],
                output_name: "y".into(),
            }],
            weights: HashMap::new(),
            graph_inputs: vec!["a".into(), "b".into()],
            graph_outputs: vec![("y".into(), Register::Virtual(0))],
        };
        let shard = Shard {
            id: 10,
            name: "shape-boundary".into(),
            range: 0..1,
            inputs: vec![],
            outputs: vec![Register::Virtual(0)],
        };
        let inputs = HashMap::from([
            ("a".into(), vec![I18::from_raw(1); len]),
            ("b".into(), vec![I18::from_raw(2); len]),
        ]);
        let assembler = to_assembler_program(&compiled, &inputs).unwrap();
        let manifest = BoundaryShapeManifest::new(
            BTreeMap::from([("a".into(), len), ("b".into(), len)]),
            BTreeMap::new(),
        )
        .unwrap();
        let cpu = Arc::new(
            ZkieIsaCpuWitnessBackend::new(
                Arc::new(compiled),
                shard,
                HashMap::new(),
                run(ModelVisibility::PublicModel),
                manifest,
            )
            .unwrap(),
        );
        (cpu, assembler)
    };
    let (len_two, template_two) = make(2);
    let (len_four, _) = make(4);
    assert_ne!(len_two.circuit_digest(), len_four.circuit_digest());
    assert!(matches!(
        Halo2KzgCpuBackend::new(
            len_four,
            template_two,
            HashMap::new(),
            13,
            SrsPolicy::DevelopmentGenerate,
        ),
        Err(BackendError::InvalidJob { .. })
    ));
}

#[test]
fn prepare_rejects_wrong_k_key_and_corrupt_persisted_bytes_before_deserialization() {
    let (cpu, assembler, domains) = fixture(run(ModelVisibility::PublicModel));
    let backend = Halo2KzgCpuBackend::new(
        cpu.clone(),
        assembler,
        domains,
        13,
        SrsPolicy::DevelopmentGenerate,
    )
    .unwrap();
    let wrong_k = PrepareJob::new_development_generate(
        cpu.run_identity().clone(),
        cpu.shard_identity().clone(),
        12,
    )
    .unwrap();
    assert!(matches!(
        backend.prepare(wrong_k, &Store::default()),
        Err(BackendError::KeyMetadataMismatch { .. })
    ));
    let wrong_key = KeyIdentity::new(
        cpu.run_identity().proof_flavor.clone(),
        cpu.circuit_digest(),
        d(99),
    )
    .unwrap();
    let wrong_key_job = PrepareJob::new_development(
        cpu.run_identity().clone(),
        cpu.shard_identity().clone(),
        13,
        wrong_key,
    )
    .unwrap();
    assert!(matches!(
        backend.prepare(wrong_key_job, &Store::default()),
        Err(BackendError::MissingKey { .. })
    ));

    let store = Store::default();
    let generated = backend
        .prepare(
            PrepareJob::new_development_generate(
                cpu.run_identity().clone(),
                cpu.shard_identity().clone(),
                13,
            )
            .unwrap(),
            &store,
        )
        .unwrap();
    let identity = generated.proving_key().clone();
    store
        .0
        .lock()
        .unwrap()
        .insert(identity.clone(), b"bad key bytes".to_vec());
    let (second_cpu, second_assembler, second_domains) = fixture(cpu.run_identity().clone());
    let second = Halo2KzgCpuBackend::new(
        second_cpu.clone(),
        second_assembler,
        second_domains,
        13,
        SrsPolicy::DevelopmentGenerate,
    )
    .unwrap();
    let job = PrepareJob::new_development(
        second_cpu.run_identity().clone(),
        second_cpu.shard_identity().clone(),
        13,
        identity,
    )
    .unwrap();
    assert!(matches!(
        second.prepare(job, &store),
        Err(BackendError::KeyDigestMismatch)
    ));
}

#[test]
fn persisted_key_package_prepares_in_a_second_backend_without_keygen() {
    let identity = run(ModelVisibility::PublicModel);
    let (first_cpu, first_assembler, first_domains) = fixture(identity.clone());
    let first = Halo2KzgCpuBackend::new(
        first_cpu.clone(),
        first_assembler,
        first_domains,
        13,
        SrsPolicy::DevelopmentGenerate,
    )
    .unwrap();
    assert_eq!(first.keygen_invocations(), 0);
    let store = Store::default();
    let keys = first
        .prepare(
            PrepareJob::new_development_generate(
                first_cpu.run_identity().clone(),
                first_cpu.shard_identity().clone(),
                13,
            )
            .unwrap(),
            &store,
        )
        .unwrap();
    assert_eq!(first.keygen_invocations(), 1);
    let cached = first
        .prepare(
            PrepareJob::new_development_generate(
                first_cpu.run_identity().clone(),
                first_cpu.shard_identity().clone(),
                13,
            )
            .unwrap(),
            &store,
        )
        .unwrap();
    assert_eq!(cached, keys);
    assert_eq!(first.keygen_invocations(), 1);

    let (second_cpu, second_assembler, second_domains) = fixture(identity);
    let second = Halo2KzgCpuBackend::new(
        second_cpu.clone(),
        second_assembler,
        second_domains,
        13,
        SrsPolicy::DevelopmentGenerate,
    )
    .unwrap();
    let loaded = second
        .prepare(
            PrepareJob::new_development(
                second_cpu.run_identity().clone(),
                second_cpu.shard_identity().clone(),
                13,
                keys.proving_key().clone(),
            )
            .unwrap(),
            &store,
        )
        .unwrap();
    assert_eq!(loaded, keys);
    assert_eq!(second.keygen_invocations(), 0);
}

#[test]
fn prove_rejects_crossed_run_and_no_clobber_outputs() {
    let (dir, backend, job) = ready("bindings");
    let mut crossed_run = job.run_identity().clone();
    crossed_run.compiler_digest = d(88);
    let crossed = ProofJob::new(
        crossed_run,
        job.shard().clone(),
        job.witness().clone(),
        job.prepared_keys().clone(),
    )
    .unwrap();
    assert!(matches!(
        backend.prove(crossed, dir.join("crossed.bin")),
        Err(BackendError::InvalidJob { .. })
    ));

    let existing = dir.join("existing.bin");
    fs::write(&existing, b"keep").unwrap();
    assert!(matches!(
        backend.prove(job.clone(), existing.clone()),
        Err(BackendError::Io { .. })
    ));
    assert_eq!(fs::read(&existing).unwrap(), b"keep");
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let link = dir.join("link.bin");
        symlink(&existing, &link).unwrap();
        assert!(matches!(
            backend.prove(job, link),
            Err(BackendError::Io { .. })
        ));
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn crossed_shard_circuit_flavor_and_missing_proof_are_rejected() {
    let (dir, backend, job) = ready("crossed");
    let proof = backend.prove(job.clone(), dir.join("proof.bin")).unwrap();
    fs::remove_file(proof.proof_path()).unwrap();
    assert!(matches!(
        backend.verify(&proof),
        Err(VerificationError::MissingProofArtifact)
    ));

    let other_shard = ShardIdentity::new(10, "other".into(), job.shard().circuit_digest()).unwrap();
    assert!(matches!(
        ProofJob::new(
            job.run_identity().clone(),
            other_shard,
            job.witness().clone(),
            job.prepared_keys().clone()
        ),
        Err(BackendError::InvalidJob { .. })
    ));
    let other_circuit = d(44);
    let circuit_shard =
        ShardIdentity::new(job.shard().id(), job.shard().name().into(), other_circuit).unwrap();
    let circuit_witness = zkie_prover::WitnessArtifact::new(
        job.witness().path().clone(),
        job.witness().digest(),
        circuit_shard.clone(),
        other_circuit,
    )
    .unwrap();
    let circuit_key = KeyIdentity::new(
        job.run_identity().proof_flavor.clone(),
        other_circuit,
        d(45),
    )
    .unwrap();
    let circuit_keys = zkie_prover::PreparedKeys::new(circuit_key.clone(), circuit_key).unwrap();
    let circuit_job = ProofJob::new(
        job.run_identity().clone(),
        circuit_shard,
        circuit_witness,
        circuit_keys,
    )
    .unwrap();
    assert!(matches!(
        backend.prove(circuit_job, dir.join("circuit.bin")),
        Err(BackendError::InvalidJob { .. })
    ));

    let mut wrong_flavor = job.run_identity().clone();
    wrong_flavor.proof_flavor = ProofFlavorId::parse("other-proof-v1").unwrap();
    assert!(matches!(
        ProofJob::new(
            wrong_flavor,
            job.shard().clone(),
            job.witness().clone(),
            job.prepared_keys().clone()
        ),
        Err(BackendError::InvalidJob { .. })
    ));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn proof_for_witness_a_cannot_verify_against_public_statement_b() {
    let (dir, backend, job_a) = ready("substitution");
    let proof_a = backend
        .prove(job_a.clone(), dir.join("proof-a.bin"))
        .unwrap();
    let witness_b = write_witness_values(
        backend.witness_backend(),
        &dir,
        "witness-b.json",
        [3_000_000_000_000_000_000, 4_000_000_000_000_000_000],
    );
    let statement_b = backend.statement_for(&witness_b).unwrap();
    let forged = zkie_prover::UnverifiedProof::new(
        proof_a.proof_path().clone(),
        proof_a.proof_digest(),
        statement_b.clone(),
        proof_a.circuit_digest(),
        proof_a.verification_key_digest(),
        proof_a.proof_flavor().clone(),
        proof_a.execution_backend().clone(),
        proof_a.shard().clone(),
        proof_a.artifact_manifest().to_vec(),
        proof_a.run_identity_digest(),
    )
    .unwrap();
    let expectation = VerificationExpectation::from_proof_job(
        &job_a,
        backend.capabilities().execution_backend().clone(),
        statement_b,
        proof_a.proof_digest(),
        proof_a.artifact_manifest().to_vec(),
    )
    .unwrap();
    let bytes = fs::read(proof_a.proof_path()).unwrap();
    assert!(matches!(
        verify_proof(
            &backend,
            VerificationJob::new(expectation, forged, bytes).unwrap()
        ),
        Err(VerificationError::CryptographicVerificationFailed { .. })
    ));
    fs::remove_dir_all(dir).unwrap();
}
