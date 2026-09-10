use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};
use zkie_prover::*;

struct FakeBackend(AtomicUsize);
impl FakeBackend {
    fn new() -> Self {
        Self(AtomicUsize::new(0))
    }
}
impl ProofBackend for FakeBackend {
    fn capabilities(&self) -> BackendCapabilities {
        capabilities()
    }
    fn estimate_resources(&self, job: ProofJob) -> Result<ResourceRequest, BackendError> {
        assert_eq!(job.witness().circuit_digest(), job.shard().circuit_digest());
        Ok(ResourceRequest::new(1, 1, 0, 0).unwrap())
    }
    fn prepare(
        &self,
        job: PrepareJob,
        _: &dyn KeyMaterialStore,
    ) -> Result<PreparedKeys, BackendError> {
        assert_eq!(
            job.key_identity().unwrap().circuit_digest(),
            job.shard().circuit_digest()
        );
        Ok(keys())
    }
    fn prove(&self, job: ProofJob, output: PathBuf) -> Result<UnverifiedProof, BackendError> {
        assert_eq!(job.witness().circuit_digest(), job.shard().circuit_digest());
        assert!(!output.as_os_str().is_empty());
        unreachable!()
    }
    fn verify_cryptographically(
        &self,
        proof: CryptoVerificationRequest,
    ) -> Result<CryptoVerificationReceipt, VerificationError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        assert_eq!(proof.proof_bytes(), b"proof");
        assert_eq!(
            proof.verification_key().key_digest(),
            keys().verification_key().key_digest()
        );
        assert_eq!(proof.circuit_digest(), digest(3));
        assert_eq!(proof.proof_flavor(), &flavor());
        Ok(CryptoVerificationReceipt::new(
            proof.request_binding_digest(),
        ))
    }
}

struct FakeWitness;
impl WitnessBackend for FakeWitness {
    fn capabilities(&self) -> BackendCapabilities {
        capabilities()
    }
    fn estimate_resources(&self, job: WitnessJob) -> Result<ResourceRequest, BackendError> {
        assert_eq!(job.expected_circuit_digest(), job.shard().circuit_digest());
        Ok(ResourceRequest::new(1, 1, 0, 0).unwrap())
    }
    fn generate(&self, job: WitnessJob, output: PathBuf) -> Result<WitnessArtifact, BackendError> {
        WitnessArtifact::new(
            output,
            digest(20),
            job.shard().clone(),
            job.expected_circuit_digest(),
        )
    }
}

#[test]
fn external_backend_can_verify_only_through_library_owned_facade() {
    let backend = FakeBackend::new();
    let _: &dyn ProofBackend = &backend;
    let _: &dyn WitnessBackend = &FakeWitness;
    let verified = verify_proof(&backend, verification_job()).unwrap();
    assert_eq!(backend.0.load(Ordering::SeqCst), 1);
    assert_eq!(verified.proof_digest(), digest_of(b"proof"));
    assert_eq!(verified.shard().id(), 4);
    assert_eq!(verified.public_statement(), b"statement");
    assert_eq!(verified.artifact_manifest(), b"manifest");
}

struct StaleReceiptBackend;
impl ProofBackend for StaleReceiptBackend {
    fn capabilities(&self) -> BackendCapabilities {
        capabilities()
    }
    fn estimate_resources(&self, _: ProofJob) -> Result<ResourceRequest, BackendError> {
        Ok(ResourceRequest::new(1, 1, 0, 0).unwrap())
    }
    fn prepare(
        &self,
        _: PrepareJob,
        _: &dyn KeyMaterialStore,
    ) -> Result<PreparedKeys, BackendError> {
        Ok(keys())
    }
    fn prove(&self, _: ProofJob, _: PathBuf) -> Result<UnverifiedProof, BackendError> {
        unreachable!()
    }
    fn verify_cryptographically(
        &self,
        _: CryptoVerificationRequest,
    ) -> Result<CryptoVerificationReceipt, VerificationError> {
        Ok(CryptoVerificationReceipt::new(digest(99)))
    }
}

struct CaptureBackend(Mutex<Option<serde_json::Value>>);
impl CaptureBackend {
    fn new() -> Self {
        Self(Mutex::new(None))
    }
    fn wire(&self) -> serde_json::Value {
        self.0.lock().unwrap().clone().unwrap()
    }
}
impl ProofBackend for CaptureBackend {
    fn capabilities(&self) -> BackendCapabilities {
        capabilities()
    }
    fn estimate_resources(&self, _: ProofJob) -> Result<ResourceRequest, BackendError> {
        Ok(ResourceRequest::new(1, 1, 0, 0).unwrap())
    }
    fn prepare(
        &self,
        _: PrepareJob,
        _: &dyn KeyMaterialStore,
    ) -> Result<PreparedKeys, BackendError> {
        Ok(keys())
    }
    fn prove(&self, _: ProofJob, _: PathBuf) -> Result<UnverifiedProof, BackendError> {
        unreachable!()
    }
    fn verify_cryptographically(
        &self,
        r: CryptoVerificationRequest,
    ) -> Result<CryptoVerificationReceipt, VerificationError> {
        let binding = r.request_binding_digest();
        *self.0.lock().unwrap() = Some(serde_json::to_value(&r).unwrap());
        Ok(CryptoVerificationReceipt::new(binding))
    }
}
#[test]
fn crypto_request_serde_rejects_invalid_wire_states() {
    let backend = CaptureBackend::new();
    verify_proof(&backend, verification_job()).unwrap();
    let valid = backend.wire();
    let mut empty = valid.clone();
    empty["proof_bytes"] = serde_json::json!([]);
    let mut oversized = valid.clone();
    oversized["proof_bytes"] = serde_json::json!(vec![0u8; (1 << 20) + 1]);
    let mut bad_digest = valid.clone();
    bad_digest["proof"]["proof_digest"] = serde_json::to_value(digest(99)).unwrap();
    let mut cross = valid;
    cross["verification_key"]["circuit_digest"] = serde_json::to_value(digest(99)).unwrap();
    for wire in [empty, oversized, bad_digest, cross] {
        assert!(serde_json::from_value::<CryptoVerificationRequest>(wire).is_err());
    }
}

enum ReceiptSwap {
    Backend,
    Shard,
}
struct SwappedReceiptBackend(ReceiptSwap);
impl ProofBackend for SwappedReceiptBackend {
    fn capabilities(&self) -> BackendCapabilities {
        capabilities()
    }
    fn estimate_resources(&self, _: ProofJob) -> Result<ResourceRequest, BackendError> {
        Ok(ResourceRequest::new(1, 1, 0, 0).unwrap())
    }
    fn prepare(
        &self,
        _: PrepareJob,
        _: &dyn KeyMaterialStore,
    ) -> Result<PreparedKeys, BackendError> {
        Ok(keys())
    }
    fn prove(&self, _: ProofJob, _: PathBuf) -> Result<UnverifiedProof, BackendError> {
        unreachable!()
    }
    fn verify_cryptographically(
        &self,
        request: CryptoVerificationRequest,
    ) -> Result<CryptoVerificationReceipt, VerificationError> {
        let mut wire = serde_json::to_value(&request).unwrap();
        match self.0 {
            ReceiptSwap::Backend => {
                wire["proof"]["execution_backend"] = serde_json::json!("alternate-worker")
            }
            ReceiptSwap::Shard => {
                wire["proof"]["shard"]["id"] = serde_json::json!(999u64);
                wire["proof"]["shard"]["name"] = serde_json::json!("alternate-shard");
            }
        }
        let alternate: CryptoVerificationRequest = serde_json::from_value(wire).unwrap();
        Ok(CryptoVerificationReceipt::new(
            alternate.request_binding_digest(),
        ))
    }
}
#[test]
fn stale_receipt_with_only_execution_backend_changed_is_rejected() {
    assert_eq!(
        verify_proof(
            &SwappedReceiptBackend(ReceiptSwap::Backend),
            verification_job()
        )
        .unwrap_err(),
        VerificationError::ReceiptBindingMismatch
    );
}
#[test]
fn stale_receipt_with_only_shard_changed_is_rejected() {
    assert_eq!(
        verify_proof(
            &SwappedReceiptBackend(ReceiptSwap::Shard),
            verification_job()
        )
        .unwrap_err(),
        VerificationError::ReceiptBindingMismatch
    );
}
#[test]
fn stale_or_crossed_receipt_cannot_mint_verified_proof() {
    assert!(matches!(
        verify_proof(&StaleReceiptBackend, verification_job()),
        Err(VerificationError::ReceiptBindingMismatch)
    ));
}

struct UnsupportedBackend;
impl ProofBackend for UnsupportedBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(
            backend_id(),
            vec![ProofFlavorId::parse("other-v1").unwrap()],
            vec![],
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
        Ok(keys())
    }
    fn prove(&self, _: ProofJob, _: PathBuf) -> Result<UnverifiedProof, BackendError> {
        unreachable!()
    }
    fn verify_cryptographically(
        &self,
        _: CryptoVerificationRequest,
    ) -> Result<CryptoVerificationReceipt, VerificationError> {
        unreachable!()
    }
}
#[test]
fn unsupported_flavor_is_exact_and_never_calls_crypto() {
    assert_eq!(
        verify_proof(&UnsupportedBackend, verification_job()).unwrap_err(),
        VerificationError::UnsupportedProofFlavor
    );
}

#[test]
fn original_expectation_rejects_coherent_substitution_and_exact_bytes_mismatch() {
    let expected = expectation_for(&proof_job(), b"statement", b"proof", b"manifest");
    let replacement = unverified_for(
        &alternate_proof_job(),
        b"other",
        b"other-statement",
        b"other-manifest",
    );
    let job = VerificationJob::new(expected.clone(), replacement, b"other".to_vec()).unwrap();
    let backend = FakeBackend::new();
    assert!(matches!(
        verify_proof(&backend, job),
        Err(VerificationError::ProofFlavorMismatch)
    ));
    assert_eq!(backend.0.load(Ordering::SeqCst), 0);
    let job = VerificationJob::new(expected, unverified(), b"other".to_vec()).unwrap();
    assert!(matches!(
        verify_proof(&FakeBackend::new(), job),
        Err(VerificationError::ProofDigestMismatch)
    ));
}

#[test]
fn each_preflight_identity_mutation_has_exact_error_and_skips_crypto() {
    let expected = expectation_for(&proof_job(), b"statement", b"proof", b"manifest");
    let cases: Vec<(UnverifiedProof, VerificationError)> = vec![
        (
            UnverifiedProof::new(
                PathBuf::from("p"),
                digest_of(b"proof"),
                b"statement".to_vec(),
                digest(3),
                keys().verification_key().key_digest(),
                ProofFlavorId::parse("other-v1").unwrap(),
                backend_id(),
                shard(),
                b"manifest".to_vec(),
                run_identity().canonical_digest(),
            )
            .unwrap(),
            VerificationError::ProofFlavorMismatch,
        ),
        (
            UnverifiedProof::new(
                PathBuf::from("p"),
                digest_of(b"proof"),
                b"statement".to_vec(),
                digest(3),
                keys().verification_key().key_digest(),
                flavor(),
                ExecutionBackendId::parse("other-worker").unwrap(),
                shard(),
                b"manifest".to_vec(),
                run_identity().canonical_digest(),
            )
            .unwrap(),
            VerificationError::ExecutionBackendMismatch,
        ),
        (
            UnverifiedProof::new(
                PathBuf::from("p"),
                digest_of(b"proof"),
                b"changed".to_vec(),
                digest(3),
                keys().verification_key().key_digest(),
                flavor(),
                backend_id(),
                shard(),
                b"manifest".to_vec(),
                run_identity().canonical_digest(),
            )
            .unwrap(),
            VerificationError::PublicStatementDigestMismatch,
        ),
        (
            UnverifiedProof::new(
                PathBuf::from("p"),
                digest_of(b"proof"),
                b"statement".to_vec(),
                digest(3),
                keys().verification_key().key_digest(),
                flavor(),
                backend_id(),
                shard(),
                b"changed-manifest".to_vec(),
                run_identity().canonical_digest(),
            )
            .unwrap(),
            VerificationError::ArtifactDigestMismatch,
        ),
        (
            UnverifiedProof::new(
                PathBuf::from("p"),
                digest_of(b"proof"),
                b"statement".to_vec(),
                digest(3),
                keys().verification_key().key_digest(),
                flavor(),
                backend_id(),
                shard(),
                b"manifest".to_vec(),
                digest(99),
            )
            .unwrap(),
            VerificationError::RunIdentityDigestMismatch,
        ),
    ];
    for (proof, expected_error) in cases {
        let backend = FakeBackend::new();
        let result = verify_proof(
            &backend,
            VerificationJob::new(expected.clone(), proof, b"proof".to_vec()).unwrap(),
        );
        assert_eq!(result.unwrap_err(), expected_error);
        assert_eq!(backend.0.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn circuit_vk_shard_and_proof_byte_mismatches_are_exact_and_skip_crypto() {
    let expected = expectation_for(&proof_job(), b"statement", b"proof", b"manifest");
    let inputs = [
        (
            unverified_for(&alternate_proof_job(), b"proof", b"statement", b"manifest"),
            VerificationError::ProofFlavorMismatch,
            b"proof".to_vec(),
        ),
        (
            UnverifiedProof::new(
                PathBuf::from("p"),
                digest_of(b"proof"),
                b"statement".to_vec(),
                digest(3),
                digest(44),
                flavor(),
                backend_id(),
                shard(),
                b"manifest".to_vec(),
                run_identity().canonical_digest(),
            )
            .unwrap(),
            VerificationError::VerificationKeyDigestMismatch,
            b"proof".to_vec(),
        ),
        (
            UnverifiedProof::new(
                PathBuf::from("p"),
                digest_of(b"proof"),
                b"statement".to_vec(),
                digest(3),
                keys().verification_key().key_digest(),
                flavor(),
                backend_id(),
                ShardIdentity::new(99, "other".into(), digest(3)).unwrap(),
                b"manifest".to_vec(),
                run_identity().canonical_digest(),
            )
            .unwrap(),
            VerificationError::ShardIdentityMismatch,
            b"proof".to_vec(),
        ),
        (
            unverified(),
            VerificationError::ProofDigestMismatch,
            b"wrong".to_vec(),
        ),
    ];
    for (proof, error, bytes) in inputs {
        let backend = FakeBackend::new();
        assert_eq!(
            verify_proof(
                &backend,
                VerificationJob::new(expected.clone(), proof, bytes).unwrap()
            )
            .unwrap_err(),
            error
        );
        assert_eq!(backend.0.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn circuit_mismatch_is_independent_of_flavor_and_key_flavor() {
    let expected = expectation_for(&proof_job(), b"statement", b"proof", b"manifest");
    let circuit = digest(9);
    let shard = ShardIdentity::new(4, "encoder".into(), circuit).unwrap();
    let key = KeyIdentity::new(flavor(), circuit, keys().verification_key().key_digest()).unwrap();
    let proof = UnverifiedProof::new(
        PathBuf::from("p"),
        digest_of(b"proof"),
        b"statement".to_vec(),
        circuit,
        key.key_digest(),
        flavor(),
        backend_id(),
        shard,
        b"manifest".to_vec(),
        run_identity().canonical_digest(),
    )
    .unwrap();
    let backend = FakeBackend::new();
    assert_eq!(
        verify_proof(
            &backend,
            VerificationJob::new(expected, proof, b"proof".to_vec()).unwrap()
        )
        .unwrap_err(),
        VerificationError::CircuitDigestMismatch
    );
    assert_eq!(backend.0.load(Ordering::SeqCst), 0);
}

struct MalformedBackend;
impl ProofBackend for MalformedBackend {
    fn capabilities(&self) -> BackendCapabilities {
        capabilities()
    }
    fn estimate_resources(&self, _: ProofJob) -> Result<ResourceRequest, BackendError> {
        Ok(ResourceRequest::new(1, 1, 0, 0).unwrap())
    }
    fn prepare(
        &self,
        _: PrepareJob,
        _: &dyn KeyMaterialStore,
    ) -> Result<PreparedKeys, BackendError> {
        Ok(keys())
    }
    fn prove(&self, _: ProofJob, _: PathBuf) -> Result<UnverifiedProof, BackendError> {
        unreachable!()
    }
    fn verify_cryptographically(
        &self,
        _: CryptoVerificationRequest,
    ) -> Result<CryptoVerificationReceipt, VerificationError> {
        Err(VerificationError::MalformedProofArtifact)
    }
}
#[test]
fn backend_malformed_artifact_propagates_without_verified_proof() {
    assert_eq!(
        verify_proof(&MalformedBackend, verification_job()).unwrap_err(),
        VerificationError::MalformedProofArtifact
    );
}

#[test]
fn preflight_rejects_manifest_statement_backend_and_oversized_bytes() {
    let expected = expectation_for(&proof_job(), b"statement", b"proof", b"manifest");
    let proof = UnverifiedProof::new(
        "p".into(),
        digest_of(b"proof"),
        b"statement".to_vec(),
        digest(3),
        keys().verification_key().key_digest(),
        flavor(),
        ExecutionBackendId::parse("other").unwrap(),
        shard(),
        b"manifest".to_vec(),
        run_identity().canonical_digest(),
    )
    .unwrap();
    assert!(matches!(
        verify_proof(
            &FakeBackend::new(),
            VerificationJob::new(expected.clone(), proof, b"proof".to_vec()).unwrap()
        ),
        Err(VerificationError::ExecutionBackendMismatch)
    ));
    assert!(UnverifiedProof::new(
        "p".into(),
        digest_of(b"proof"),
        vec![0; (1 << 20) + 1],
        digest(3),
        keys().verification_key().key_digest(),
        flavor(),
        backend_id(),
        shard(),
        b"manifest".to_vec(),
        run_identity().canonical_digest()
    )
    .is_err());
    assert!(VerificationExpectation::from_proof_job(
        &proof_job(),
        backend_id(),
        b"statement".to_vec(),
        digest_of(b"proof"),
        vec![0; (1 << 20) + 1]
    )
    .is_err());
}

#[test]
fn constructors_and_serde_reject_cross_field_confusion() {
    assert!(WitnessJob::new(run_identity(), shard(), PathBuf::from("in"), digest(9)).is_err());
    assert!(PreparedKeys::new(
        key_identity(b"pk"),
        KeyIdentity::new(
            ProofFlavorId::parse("other-v1").unwrap(),
            digest(3),
            digest_of(b"vk")
        )
        .unwrap(),
    )
    .is_err());
    let valid = WitnessJob::new(run_identity(), shard(), PathBuf::from("in"), digest(3)).unwrap();
    let json = serde_json::to_string(&valid).unwrap();
    assert_eq!(serde_json::from_str::<WitnessJob>(&json).unwrap(), valid);
    round_trip(&WitnessArtifact::new(PathBuf::from("w"), digest(20), shard(), digest(3)).unwrap());
    round_trip(&PrepareJob::new(run_identity(), shard(), 10, key_identity(b"key")).unwrap());
    round_trip(&keys());
    round_trip(&proof_job());
    round_trip(&unverified());
    let mut invalid = serde_json::to_value(&valid).unwrap();
    invalid["expected_circuit_digest"] = serde_json::to_value(digest(9)).unwrap();
    assert!(serde_json::from_value::<WitnessJob>(invalid).is_err());
}

#[test]
fn capabilities_are_generic_and_validate_flavors() {
    assert!(BackendCapabilities::new(backend_id(), vec![], vec![]).is_err());
    assert!(BackendCapabilities::new(backend_id(), vec![flavor(), flavor()], vec![]).is_err());
    assert!(BackendCapabilities::new(
        backend_id(),
        vec![flavor()],
        vec![CapabilityId::parse("remote-worker").unwrap()]
    )
    .is_ok());
}

fn digest(n: u8) -> Digest32 {
    Digest32::new([n; 32])
}
fn digest_of(b: &[u8]) -> Digest32 {
    Digest32::new(*blake3::hash(b).as_bytes())
}
fn flavor() -> ProofFlavorId {
    ProofFlavorId::parse("test-v1").unwrap()
}
fn backend_id() -> ExecutionBackendId {
    ExecutionBackendId::parse("fake-worker").unwrap()
}
fn run_identity() -> RunIdentity {
    RunIdentity {
        model_graph_digest: digest(1),
        weights_digest: digest(2),
        compiler_digest: digest(3),
        isa_digest: digest(4),
        quantization_digest: digest(5),
        partition_plan_digest: digest(6),
        aggregation_plan_digest: digest(7),
        proof_flavor: flavor(),
        model_visibility: ModelVisibility::PublicModel,
        aggregation_fan_in: 2,
        public_input_schema_version: 1,
    }
}
fn shard() -> ShardIdentity {
    ShardIdentity::new(4, "encoder".into(), digest(3)).unwrap()
}
fn key_identity(b: &[u8]) -> KeyIdentity {
    KeyIdentity::new(flavor(), digest(3), digest_of(b)).unwrap()
}
fn keys() -> PreparedKeys {
    let key = key_identity(b"key");
    PreparedKeys::new(key.clone(), key).unwrap()
}
fn proof_job() -> ProofJob {
    ProofJob::new(
        run_identity(),
        shard(),
        WitnessArtifact::new(PathBuf::from("w"), digest(20), shard(), digest(3)).unwrap(),
        keys(),
    )
    .unwrap()
}
fn alternate_proof_job() -> ProofJob {
    let f = ProofFlavorId::parse("alt-v1").unwrap();
    let s = ShardIdentity::new(8, "alt".into(), digest(9)).unwrap();
    let run = RunIdentity {
        proof_flavor: f.clone(),
        ..run_identity()
    };
    let k = KeyIdentity::new(f, digest(9), digest_of(b"alt")).unwrap();
    ProofJob::new(
        run,
        s.clone(),
        WitnessArtifact::new(PathBuf::from("w"), digest(21), s.clone(), digest(9)).unwrap(),
        PreparedKeys::new(k.clone(), k).unwrap(),
    )
    .unwrap()
}
fn expectation_for(j: &ProofJob, s: &[u8], p: &[u8], m: &[u8]) -> VerificationExpectation {
    VerificationExpectation::from_proof_job(j, backend_id(), s.to_vec(), digest_of(p), m.to_vec())
        .unwrap()
}
fn unverified_for(j: &ProofJob, p: &[u8], s: &[u8], m: &[u8]) -> UnverifiedProof {
    UnverifiedProof::new(
        PathBuf::from("proof"),
        digest_of(p),
        s.to_vec(),
        j.shard().circuit_digest(),
        j.prepared_keys().verification_key().key_digest(),
        j.run_identity().proof_flavor.clone(),
        backend_id(),
        j.shard().clone(),
        m.to_vec(),
        j.run_identity().canonical_digest(),
    )
    .unwrap()
}
fn unverified() -> UnverifiedProof {
    unverified_for(&proof_job(), b"proof", b"statement", b"manifest")
}
fn verification_job() -> VerificationJob {
    VerificationJob::new(
        expectation_for(&proof_job(), b"statement", b"proof", b"manifest"),
        unverified(),
        b"proof".to_vec(),
    )
    .unwrap()
}
fn capabilities() -> BackendCapabilities {
    BackendCapabilities::new(backend_id(), vec![flavor()], vec![]).unwrap()
}

// Foundation regressions retained from Task 1/3 before verification hardening.
#[test]
fn resource_reservation_preserves_per_device_vram() {
    let capacity = ResourceCapacity::new(4, 8 << 30, 2, 16 << 30).unwrap();
    let remaining = capacity
        .checked_reserve(&ResourceRequest::new(1, 1 << 30, 1, 16 << 30).unwrap())
        .unwrap();
    assert_eq!(remaining.gpu_count, 1);
    assert_eq!(remaining.gpu_vram_bytes_per_device, 16 << 30);
}
#[test]
fn type_deserialization_keeps_identifier_and_resource_validation() {
    assert!(serde_json::from_str::<ProofFlavorId>(r#""""#).is_err());
    assert!(serde_json::from_str::<ExecutionBackendId>(r#""halo2-cudá""#).is_err());
    assert!(serde_json::from_str::<ResourceRequest>(
        r#"{"cpu_cores":0,"ram_bytes":0,"gpu_count":0,"gpu_vram_bytes_per_device":0}"#
    )
    .is_err());
}
#[test]
fn run_identity_digest_binds_every_field() {
    let base = run_identity();
    let original = base.canonical_digest();
    for changed in [
        RunIdentity {
            model_graph_digest: digest(9),
            ..base.clone()
        },
        RunIdentity {
            weights_digest: digest(9),
            ..base.clone()
        },
        RunIdentity {
            compiler_digest: digest(9),
            ..base.clone()
        },
        RunIdentity {
            isa_digest: digest(9),
            ..base.clone()
        },
        RunIdentity {
            quantization_digest: digest(9),
            ..base.clone()
        },
        RunIdentity {
            partition_plan_digest: digest(9),
            ..base.clone()
        },
        RunIdentity {
            aggregation_plan_digest: digest(9),
            ..base.clone()
        },
        RunIdentity {
            proof_flavor: ProofFlavorId::parse("other-v1").unwrap(),
            ..base.clone()
        },
        RunIdentity {
            model_visibility: ModelVisibility::PrivateModel,
            ..base.clone()
        },
        RunIdentity {
            aggregation_fan_in: 3,
            ..base.clone()
        },
        RunIdentity {
            public_input_schema_version: 2,
            ..base.clone()
        },
    ] {
        assert_ne!(original, changed.canonical_digest());
    }
}
#[test]
fn nested_identity_and_capability_serde_rejects_invalid_values() {
    assert!(serde_json::from_str::<CapabilityId>(r#""""#).is_err());
    assert!(serde_json::from_str::<CapabilityId>(r#""é""#).is_err());
    assert!(serde_json::from_str::<ShardIdentity>(r#"{"id":1,"name":"","circuit_digest":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}"#).is_err());
    assert!(BackendCapabilities::new(backend_id(), vec![flavor(), flavor()], vec![]).is_err());
}

fn round_trip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug + Clone,
{
    assert_eq!(
        serde_json::from_str::<T>(&serde_json::to_string(value).unwrap()).unwrap(),
        value.clone()
    );
}
