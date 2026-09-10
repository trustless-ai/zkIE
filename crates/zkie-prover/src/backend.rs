//! Halo2-free, process-boundary-safe witness and proof backend contracts.
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf, sync::Arc};
use thiserror::Error;
use zkie_types::{Digest32, ExecutionBackendId, ProofFlavorId, ResourceRequest, RunIdentity};

const MAX_TEXT: usize = 4096;
const MAX_BYTES: usize = 1 << 20;
fn digest(bytes: &[u8]) -> Digest32 {
    Digest32::new(*blake3::hash(bytes).as_bytes())
}
fn valid_text(s: &str) -> Result<(), BackendError> {
    if s.is_empty() || s.len() > MAX_TEXT {
        Err(BackendError::InvalidJob {
            message: "empty or oversized text".into(),
        })
    } else {
        Ok(())
    }
}
fn valid_bytes(b: &[u8]) -> Result<(), BackendError> {
    if b.len() > MAX_BYTES {
        Err(BackendError::InvalidJob {
            message: "oversized byte field".into(),
        })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct CapabilityId(String);
impl CapabilityId {
    pub fn parse(value: impl Into<String>) -> Result<Self, BackendError> {
        let v = value.into();
        valid_text(&v)?;
        if !v.is_ascii() {
            return Err(BackendError::InvalidJob {
                message: "non-ASCII capability".into(),
            });
        }
        Ok(Self(v))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl<'de> Deserialize<'de> for CapabilityId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d).and_then(|v| Self::parse(v).map_err(serde::de::Error::custom))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ShardIdentity {
    id: u64,
    name: String,
    circuit_digest: Digest32,
}
#[derive(Deserialize)]
struct ShardIdentityWire {
    id: u64,
    name: String,
    circuit_digest: Digest32,
}
impl<'de> Deserialize<'de> for ShardIdentity {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        ShardIdentityWire::deserialize(d).and_then(|w| {
            Self::new(w.id, w.name, w.circuit_digest).map_err(serde::de::Error::custom)
        })
    }
}
impl ShardIdentity {
    pub fn new(id: u64, name: String, circuit_digest: Digest32) -> Result<Self, BackendError> {
        valid_text(&name)?;
        Ok(Self {
            id,
            name,
            circuit_digest,
        })
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn circuit_digest(&self) -> Digest32 {
        self.circuit_digest
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyIdentity {
    proof_flavor: ProofFlavorId,
    circuit_digest: Digest32,
    key_digest: Digest32,
}
impl KeyIdentity {
    pub fn new(
        proof_flavor: ProofFlavorId,
        circuit_digest: Digest32,
        key_digest: Digest32,
    ) -> Result<Self, BackendError> {
        Ok(Self {
            proof_flavor,
            circuit_digest,
            key_digest,
        })
    }
    pub fn key_digest(&self) -> Digest32 {
        self.key_digest
    }
    pub fn proof_flavor(&self) -> &ProofFlavorId {
        &self.proof_flavor
    }
    pub fn circuit_digest(&self) -> Digest32 {
        self.circuit_digest
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WitnessJob {
    run_identity: RunIdentity,
    shard: ShardIdentity,
    witness_input_path: PathBuf,
    expected_circuit_digest: Digest32,
}
#[derive(Deserialize)]
struct WitnessJobWire {
    run_identity: RunIdentity,
    shard: ShardIdentity,
    witness_input_path: PathBuf,
    expected_circuit_digest: Digest32,
}
impl WitnessJob {
    pub fn new(
        run_identity: RunIdentity,
        shard: ShardIdentity,
        witness_input_path: PathBuf,
        expected_circuit_digest: Digest32,
    ) -> Result<Self, BackendError> {
        valid_text(&witness_input_path.to_string_lossy())?;
        if shard.circuit_digest() != expected_circuit_digest {
            return Err(BackendError::InvalidJob {
                message: "witness circuit mismatch".into(),
            });
        }
        Ok(Self {
            run_identity,
            shard,
            witness_input_path,
            expected_circuit_digest,
        })
    }
    pub fn run_identity(&self) -> &RunIdentity {
        &self.run_identity
    }
    pub fn shard(&self) -> &ShardIdentity {
        &self.shard
    }
    pub fn witness_input_path(&self) -> &PathBuf {
        &self.witness_input_path
    }
    pub fn expected_circuit_digest(&self) -> Digest32 {
        self.expected_circuit_digest
    }
}
impl TryFrom<WitnessJobWire> for WitnessJob {
    type Error = BackendError;
    fn try_from(w: WitnessJobWire) -> Result<Self, Self::Error> {
        Self::new(
            w.run_identity,
            w.shard,
            w.witness_input_path,
            w.expected_circuit_digest,
        )
    }
}
impl<'de> Deserialize<'de> for WitnessJob {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        WitnessJobWire::deserialize(d)
            .and_then(|w| Self::try_from(w).map_err(serde::de::Error::custom))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WitnessArtifact {
    path: PathBuf,
    digest: Digest32,
    shard: ShardIdentity,
    circuit_digest: Digest32,
}
#[derive(Deserialize)]
struct WitnessArtifactWire {
    path: PathBuf,
    digest: Digest32,
    shard: ShardIdentity,
    circuit_digest: Digest32,
}
impl WitnessArtifact {
    pub fn new(
        path: PathBuf,
        digest: Digest32,
        shard: ShardIdentity,
        circuit_digest: Digest32,
    ) -> Result<Self, BackendError> {
        valid_text(&path.to_string_lossy())?;
        if shard.circuit_digest() != circuit_digest {
            return Err(BackendError::InvalidJob {
                message: "witness artifact circuit mismatch".into(),
            });
        }
        Ok(Self {
            path,
            digest,
            shard,
            circuit_digest,
        })
    }
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
    pub fn digest(&self) -> Digest32 {
        self.digest
    }
    pub fn shard(&self) -> &ShardIdentity {
        &self.shard
    }
    pub fn circuit_digest(&self) -> Digest32 {
        self.circuit_digest
    }
}
impl TryFrom<WitnessArtifactWire> for WitnessArtifact {
    type Error = BackendError;
    fn try_from(w: WitnessArtifactWire) -> Result<Self, Self::Error> {
        Self::new(w.path, w.digest, w.shard, w.circuit_digest)
    }
}
impl<'de> Deserialize<'de> for WitnessArtifact {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        WitnessArtifactWire::deserialize(d)
            .and_then(|w| Self::try_from(w).map_err(serde::de::Error::custom))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PrepareJob {
    run_identity: RunIdentity,
    shard: ShardIdentity,
    circuit_k: u32,
    key_identity: KeyIdentity,
}
#[derive(Deserialize)]
struct PrepareJobWire {
    run_identity: RunIdentity,
    shard: ShardIdentity,
    circuit_k: u32,
    key_identity: KeyIdentity,
}
impl PrepareJob {
    pub fn new(
        run_identity: RunIdentity,
        shard: ShardIdentity,
        circuit_k: u32,
        key_identity: KeyIdentity,
    ) -> Result<Self, BackendError> {
        if circuit_k == 0
            || run_identity.proof_flavor != key_identity.proof_flavor
            || shard.circuit_digest() != key_identity.circuit_digest
        {
            return Err(BackendError::InvalidJob {
                message: "invalid prepare identity".into(),
            });
        }
        Ok(Self {
            run_identity,
            shard,
            circuit_k,
            key_identity,
        })
    }
    pub fn run_identity(&self) -> &RunIdentity {
        &self.run_identity
    }
    pub fn shard(&self) -> &ShardIdentity {
        &self.shard
    }
    pub fn circuit_k(&self) -> u32 {
        self.circuit_k
    }
    pub fn key_identity(&self) -> &KeyIdentity {
        &self.key_identity
    }
}
impl TryFrom<PrepareJobWire> for PrepareJob {
    type Error = BackendError;
    fn try_from(w: PrepareJobWire) -> Result<Self, Self::Error> {
        Self::new(w.run_identity, w.shard, w.circuit_k, w.key_identity)
    }
}
impl<'de> Deserialize<'de> for PrepareJob {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        PrepareJobWire::deserialize(d)
            .and_then(|w| Self::try_from(w).map_err(serde::de::Error::custom))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PreparedKeys {
    proving_key: KeyIdentity,
    verification_key: KeyIdentity,
}
#[derive(Deserialize)]
struct PreparedKeysWire {
    proving_key: KeyIdentity,
    verification_key: KeyIdentity,
}
impl PreparedKeys {
    pub fn new(
        proving_key: KeyIdentity,
        verification_key: KeyIdentity,
    ) -> Result<Self, BackendError> {
        if proving_key.proof_flavor != verification_key.proof_flavor
            || proving_key.circuit_digest != verification_key.circuit_digest
        {
            return Err(BackendError::InvalidJob {
                message: "PK/VK identity mismatch".into(),
            });
        }
        Ok(Self {
            proving_key,
            verification_key,
        })
    }
    pub fn verification_key(&self) -> &KeyIdentity {
        &self.verification_key
    }
    pub fn proving_key(&self) -> &KeyIdentity {
        &self.proving_key
    }
}
impl TryFrom<PreparedKeysWire> for PreparedKeys {
    type Error = BackendError;
    fn try_from(w: PreparedKeysWire) -> Result<Self, Self::Error> {
        Self::new(w.proving_key, w.verification_key)
    }
}
impl<'de> Deserialize<'de> for PreparedKeys {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        PreparedKeysWire::deserialize(d)
            .and_then(|w| Self::try_from(w).map_err(serde::de::Error::custom))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProofJob {
    run_identity: RunIdentity,
    shard: ShardIdentity,
    witness: WitnessArtifact,
    prepared_keys: PreparedKeys,
}
#[derive(Deserialize)]
struct ProofJobWire {
    run_identity: RunIdentity,
    shard: ShardIdentity,
    witness: WitnessArtifact,
    prepared_keys: PreparedKeys,
}
impl ProofJob {
    pub fn new(
        run_identity: RunIdentity,
        shard: ShardIdentity,
        witness: WitnessArtifact,
        prepared_keys: PreparedKeys,
    ) -> Result<Self, BackendError> {
        let c = shard.circuit_digest();
        if witness.shard != shard
            || witness.circuit_digest != c
            || prepared_keys.proving_key.circuit_digest != c
            || run_identity.proof_flavor != prepared_keys.proving_key.proof_flavor
        {
            return Err(BackendError::InvalidJob {
                message: "proof job identity mismatch".into(),
            });
        }
        Ok(Self {
            run_identity,
            shard,
            witness,
            prepared_keys,
        })
    }
    pub fn shard(&self) -> &ShardIdentity {
        &self.shard
    }
    pub fn prepared_keys(&self) -> &PreparedKeys {
        &self.prepared_keys
    }
    pub fn run_identity(&self) -> &RunIdentity {
        &self.run_identity
    }
    pub fn witness(&self) -> &WitnessArtifact {
        &self.witness
    }
}
impl TryFrom<ProofJobWire> for ProofJob {
    type Error = BackendError;
    fn try_from(w: ProofJobWire) -> Result<Self, Self::Error> {
        Self::new(w.run_identity, w.shard, w.witness, w.prepared_keys)
    }
}
impl<'de> Deserialize<'de> for ProofJob {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        ProofJobWire::deserialize(d)
            .and_then(|w| Self::try_from(w).map_err(serde::de::Error::custom))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct UnverifiedProof {
    proof_path: PathBuf,
    proof_digest: Digest32,
    public_statement: Vec<u8>,
    public_statement_digest: Digest32,
    circuit_digest: Digest32,
    verification_key_digest: Digest32,
    proof_flavor: ProofFlavorId,
    execution_backend: ExecutionBackendId,
    shard: ShardIdentity,
    artifact_manifest: Vec<u8>,
    artifact_digest: Digest32,
    run_identity_digest: Digest32,
}
#[derive(Deserialize)]
struct UnverifiedProofWire {
    proof_path: PathBuf,
    proof_digest: Digest32,
    public_statement: Vec<u8>,
    public_statement_digest: Digest32,
    circuit_digest: Digest32,
    verification_key_digest: Digest32,
    proof_flavor: ProofFlavorId,
    execution_backend: ExecutionBackendId,
    shard: ShardIdentity,
    artifact_manifest: Vec<u8>,
    artifact_digest: Digest32,
    run_identity_digest: Digest32,
}
impl UnverifiedProof {
    #[allow(clippy::too_many_arguments)] // The serialized proof boundary has these independent cryptographic bindings.
    pub fn new(
        proof_path: PathBuf,
        proof_digest: Digest32,
        public_statement: Vec<u8>,
        circuit_digest: Digest32,
        verification_key_digest: Digest32,
        proof_flavor: ProofFlavorId,
        execution_backend: ExecutionBackendId,
        shard: ShardIdentity,
        artifact_manifest: Vec<u8>,
        run_identity_digest: Digest32,
    ) -> Result<Self, BackendError> {
        valid_text(&proof_path.to_string_lossy())?;
        valid_bytes(&public_statement)?;
        valid_bytes(&artifact_manifest)?;
        if artifact_manifest.is_empty() {
            return Err(BackendError::InvalidJob {
                message: "missing artifact manifest".into(),
            });
        }
        if shard.circuit_digest() != circuit_digest {
            return Err(BackendError::InvalidJob {
                message: "proof shard circuit mismatch".into(),
            });
        }
        let public_statement_digest = digest(&public_statement);
        let artifact_digest = digest(&artifact_manifest);
        Ok(Self {
            proof_path,
            proof_digest,
            public_statement,
            public_statement_digest,
            circuit_digest,
            verification_key_digest,
            proof_flavor,
            execution_backend,
            shard,
            artifact_manifest,
            artifact_digest,
            run_identity_digest,
        })
    }
    pub fn proof_path(&self) -> &PathBuf {
        &self.proof_path
    }
    pub fn proof_digest(&self) -> Digest32 {
        self.proof_digest
    }
    pub fn public_statement(&self) -> &[u8] {
        &self.public_statement
    }
    pub fn circuit_digest(&self) -> Digest32 {
        self.circuit_digest
    }
    pub fn verification_key_digest(&self) -> Digest32 {
        self.verification_key_digest
    }
    pub fn proof_flavor(&self) -> &ProofFlavorId {
        &self.proof_flavor
    }
    pub fn execution_backend(&self) -> &ExecutionBackendId {
        &self.execution_backend
    }
    pub fn shard(&self) -> &ShardIdentity {
        &self.shard
    }
    pub fn artifact_manifest(&self) -> &[u8] {
        &self.artifact_manifest
    }
    pub fn run_identity_digest(&self) -> Digest32 {
        self.run_identity_digest
    }
    pub fn public_statement_digest(&self) -> Digest32 {
        self.public_statement_digest
    }
    pub fn artifact_digest(&self) -> Digest32 {
        self.artifact_digest
    }
}
impl TryFrom<UnverifiedProofWire> for UnverifiedProof {
    type Error = BackendError;
    fn try_from(w: UnverifiedProofWire) -> Result<Self, Self::Error> {
        let p = Self::new(
            w.proof_path,
            w.proof_digest,
            w.public_statement,
            w.circuit_digest,
            w.verification_key_digest,
            w.proof_flavor,
            w.execution_backend,
            w.shard,
            w.artifact_manifest,
            w.run_identity_digest,
        )?;
        if p.public_statement_digest != w.public_statement_digest
            || p.artifact_digest != w.artifact_digest
        {
            Err(BackendError::InvalidJob {
                message: "proof digest metadata mismatch".into(),
            })
        } else {
            Ok(p)
        }
    }
}
impl<'de> Deserialize<'de> for UnverifiedProof {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        UnverifiedProofWire::deserialize(d)
            .and_then(|w| Self::try_from(w).map_err(serde::de::Error::custom))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationExpectation {
    proof_digest: Digest32,
    artifact_digest: Digest32,
    run_identity_digest: Digest32,
    public_statement_digest: Digest32,
    circuit_digest: Digest32,
    verification_key_digest: Digest32,
    proof_flavor: ProofFlavorId,
    execution_backend: ExecutionBackendId,
    shard: ShardIdentity,
}
impl VerificationExpectation {
    pub fn from_proof_job(
        job: &ProofJob,
        execution_backend: ExecutionBackendId,
        statement: Vec<u8>,
        proof_digest: Digest32,
        manifest: Vec<u8>,
    ) -> Result<Self, BackendError> {
        valid_bytes(&statement)?;
        valid_bytes(&manifest)?;
        if manifest.is_empty() {
            return Err(BackendError::InvalidJob {
                message: "missing artifact manifest".into(),
            });
        }
        Ok(Self {
            proof_digest,
            artifact_digest: digest(&manifest),
            run_identity_digest: job.run_identity.canonical_digest(),
            public_statement_digest: digest(&statement),
            circuit_digest: job.shard.circuit_digest(),
            verification_key_digest: job.prepared_keys.verification_key.key_digest(),
            proof_flavor: job.run_identity.proof_flavor.clone(),
            execution_backend,
            shard: job.shard.clone(),
        })
    }
}
pub struct VerificationJob {
    expectation: VerificationExpectation,
    proof: UnverifiedProof,
    proof_bytes: Vec<u8>,
}
impl VerificationJob {
    pub fn new(
        expectation: VerificationExpectation,
        proof: UnverifiedProof,
        proof_bytes: Vec<u8>,
    ) -> Result<Self, BackendError> {
        valid_bytes(&proof_bytes)?;
        if proof_bytes.is_empty() {
            return Err(BackendError::InvalidJob {
                message: "missing proof artifact".into(),
            });
        }
        Ok(Self {
            expectation,
            proof,
            proof_bytes,
        })
    }
}
#[derive(Serialize)]
pub struct CryptoVerificationRequest {
    proof: Arc<UnverifiedProof>,
    proof_bytes: Vec<u8>,
    verification_key: KeyIdentity,
}
impl CryptoVerificationRequest {
    pub fn proof_bytes(&self) -> &[u8] {
        &self.proof_bytes
    }
    pub fn public_statement(&self) -> &[u8] {
        &self.proof.public_statement
    }
    pub fn artifact_manifest(&self) -> &[u8] {
        &self.proof.artifact_manifest
    }
    pub fn verification_key(&self) -> &KeyIdentity {
        &self.verification_key
    }
    pub fn circuit_digest(&self) -> Digest32 {
        self.proof.circuit_digest
    }
    pub fn proof_flavor(&self) -> &ProofFlavorId {
        &self.proof.proof_flavor
    }
    pub fn execution_backend(&self) -> &ExecutionBackendId {
        &self.proof.execution_backend
    }
    pub fn shard(&self) -> &ShardIdentity {
        &self.proof.shard
    }
    pub fn run_identity_digest(&self) -> Digest32 {
        self.proof.run_identity_digest
    }
    pub fn request_binding_digest(&self) -> Digest32 {
        let mut h = blake3::Hasher::new();
        h.update(b"zkie.crypto-request.v1");
        h.update(&(self.proof_flavor().as_str().len() as u64).to_le_bytes());
        h.update(self.proof_flavor().as_str().as_bytes());
        h.update(&(self.execution_backend().as_str().len() as u64).to_le_bytes());
        h.update(self.execution_backend().as_str().as_bytes());
        h.update(&self.shard().id().to_le_bytes());
        h.update(&(self.shard().name().len() as u64).to_le_bytes());
        h.update(self.shard().name().as_bytes());
        h.update(self.shard().circuit_digest().as_bytes());
        h.update(&(self.verification_key.proof_flavor().as_str().len() as u64).to_le_bytes());
        h.update(self.verification_key.proof_flavor().as_str().as_bytes());
        h.update(self.verification_key.circuit_digest().as_bytes());
        for d in [
            self.proof.proof_digest,
            self.proof.artifact_digest,
            self.proof.public_statement_digest,
            self.proof.circuit_digest,
            self.proof.verification_key_digest,
            self.proof.run_identity_digest,
            self.verification_key.key_digest,
        ] {
            h.update(d.as_bytes());
        }
        h.update(&(self.proof.public_statement.len() as u64).to_le_bytes());
        h.update(&self.proof.public_statement);
        h.update(&(self.proof.artifact_manifest.len() as u64).to_le_bytes());
        h.update(&self.proof.artifact_manifest);
        h.update(&(self.proof_bytes.len() as u64).to_le_bytes());
        h.update(&self.proof_bytes);
        Digest32::new(*h.finalize().as_bytes())
    }
}
#[derive(Deserialize)]
struct CryptoVerificationRequestWire {
    proof: UnverifiedProof,
    proof_bytes: Vec<u8>,
    verification_key: KeyIdentity,
}
impl CryptoVerificationRequest {
    fn new(
        proof: Arc<UnverifiedProof>,
        proof_bytes: Vec<u8>,
        verification_key: KeyIdentity,
    ) -> Result<Self, BackendError> {
        valid_bytes(&proof_bytes)?;
        if proof_bytes.is_empty()
            || digest(&proof_bytes) != proof.proof_digest
            || proof.proof_flavor != verification_key.proof_flavor
            || proof.circuit_digest != verification_key.circuit_digest
            || proof.verification_key_digest != verification_key.key_digest
        {
            return Err(BackendError::InvalidJob {
                message: "invalid crypto verification request".into(),
            });
        }
        Ok(Self {
            proof,
            proof_bytes,
            verification_key,
        })
    }
}
impl<'de> Deserialize<'de> for CryptoVerificationRequest {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        CryptoVerificationRequestWire::deserialize(d).and_then(|w| {
            Self::new(Arc::new(w.proof), w.proof_bytes, w.verification_key)
                .map_err(serde::de::Error::custom)
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CryptoVerificationReceipt {
    request_binding_digest: Digest32,
}
impl CryptoVerificationReceipt {
    pub fn new(request_binding_digest: Digest32) -> Self {
        Self {
            request_binding_digest,
        }
    }
    pub fn request_binding_digest(&self) -> Digest32 {
        self.request_binding_digest
    }
}
#[derive(Debug)]
pub struct VerifiedProof {
    proof: Arc<UnverifiedProof>,
}
impl VerifiedProof {
    pub fn proof_digest(&self) -> Digest32 {
        self.proof.proof_digest
    }
    pub fn shard(&self) -> &ShardIdentity {
        &self.proof.shard
    }
    pub fn proof_path(&self) -> &PathBuf {
        &self.proof.proof_path
    }
    pub fn public_statement(&self) -> &[u8] {
        &self.proof.public_statement
    }
    pub fn public_statement_digest(&self) -> Digest32 {
        self.proof.public_statement_digest
    }
    pub fn artifact_manifest(&self) -> &[u8] {
        &self.proof.artifact_manifest
    }
    pub fn artifact_digest(&self) -> Digest32 {
        self.proof.artifact_digest
    }
    pub fn run_identity_digest(&self) -> Digest32 {
        self.proof.run_identity_digest
    }
    pub fn circuit_digest(&self) -> Digest32 {
        self.proof.circuit_digest
    }
    pub fn verification_key_digest(&self) -> Digest32 {
        self.proof.verification_key_digest
    }
    pub fn proof_flavor(&self) -> &ProofFlavorId {
        &self.proof.proof_flavor
    }
    pub fn execution_backend(&self) -> &ExecutionBackendId {
        &self.proof.execution_backend
    }
}

pub fn verify_proof(
    backend: &dyn ProofBackend,
    job: VerificationJob,
) -> Result<VerifiedProof, VerificationError> {
    let caps = backend.capabilities();
    caps.validate()
        .map_err(|e| VerificationError::InvalidCapability {
            message: e.to_string(),
        })?;
    if !caps.proof_flavors.contains(&job.expectation.proof_flavor) {
        return Err(VerificationError::UnsupportedProofFlavor);
    }
    let VerificationJob {
        expectation,
        proof,
        proof_bytes,
    } = job;
    let e = &expectation;
    let p = &proof;
    macro_rules! ck {
        ($a:ident,$v:ident,$err:ident) => {
            if e.$a != p.$a {
                return Err(VerificationError::$err);
            }
        };
    }
    ck!(proof_flavor, proof_flavor, ProofFlavorMismatch);
    ck!(
        execution_backend,
        execution_backend,
        ExecutionBackendMismatch
    );
    ck!(circuit_digest, circuit_digest, CircuitDigestMismatch);
    ck!(
        verification_key_digest,
        verification_key_digest,
        VerificationKeyDigestMismatch
    );
    ck!(
        run_identity_digest,
        run_identity_digest,
        RunIdentityDigestMismatch
    );
    if e.shard != p.shard {
        return Err(VerificationError::ShardIdentityMismatch);
    }
    if e.public_statement_digest != p.public_statement_digest {
        return Err(VerificationError::PublicStatementDigestMismatch);
    }
    if e.artifact_digest != p.artifact_digest {
        return Err(VerificationError::ArtifactDigestMismatch);
    }
    if proof_bytes.is_empty() {
        return Err(VerificationError::MissingProofArtifact);
    }
    if p.artifact_manifest.is_empty() {
        return Err(VerificationError::MissingArtifactManifest);
    }
    if digest(&proof_bytes) != p.proof_digest || e.proof_digest != p.proof_digest {
        return Err(VerificationError::ProofDigestMismatch);
    }
    let proof = Arc::new(proof);
    let verified_proof = proof.clone();
    let pre = CryptoVerificationRequest::new(
        proof,
        proof_bytes,
        KeyIdentity::new(
            e.proof_flavor.clone(),
            e.circuit_digest,
            e.verification_key_digest,
        )
        .map_err(|_| VerificationError::MalformedProofArtifact)?,
    )
    .map_err(|_| VerificationError::MalformedProofArtifact)?;
    let binding = pre.request_binding_digest();
    let receipt = backend.verify_cryptographically(pre)?;
    if receipt.request_binding_digest() != binding {
        return Err(VerificationError::ReceiptBindingMismatch);
    }
    Ok(VerifiedProof {
        proof: verified_proof,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BackendCapabilities {
    execution_backend: ExecutionBackendId,
    proof_flavors: Vec<ProofFlavorId>,
    capabilities: Vec<CapabilityId>,
}
#[derive(Deserialize)]
struct BackendCapabilitiesWire {
    execution_backend: ExecutionBackendId,
    proof_flavors: Vec<ProofFlavorId>,
    capabilities: Vec<CapabilityId>,
}
impl<'de> Deserialize<'de> for BackendCapabilities {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        BackendCapabilitiesWire::deserialize(d).and_then(|w| {
            Self::new(w.execution_backend, w.proof_flavors, w.capabilities)
                .map_err(serde::de::Error::custom)
        })
    }
}
impl BackendCapabilities {
    pub fn new(
        execution_backend: ExecutionBackendId,
        proof_flavors: Vec<ProofFlavorId>,
        capabilities: Vec<CapabilityId>,
    ) -> Result<Self, BackendError> {
        let s = Self {
            execution_backend,
            proof_flavors,
            capabilities,
        };
        s.validate()?;
        Ok(s)
    }
    fn validate(&self) -> Result<(), BackendError> {
        if self.proof_flavors.is_empty()
            || self.proof_flavors.iter().collect::<HashSet<_>>().len() != self.proof_flavors.len()
        {
            Err(BackendError::InvalidJob {
                message: "flavors must be nonempty and unique".into(),
            })
        } else if self
            .capabilities
            .iter()
            .any(|c| CapabilityId::parse(c.as_str()).is_err())
        {
            Err(BackendError::InvalidJob {
                message: "invalid capability".into(),
            })
        } else {
            Ok(())
        }
    }
    pub fn execution_backend(&self) -> &ExecutionBackendId {
        &self.execution_backend
    }
    pub fn proof_flavors(&self) -> &[ProofFlavorId] {
        &self.proof_flavors
    }
    pub fn capabilities(&self) -> &[CapabilityId] {
        &self.capabilities
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyWriteOutcome {
    Inserted,
    AlreadyPresentIdentical,
}
pub trait KeyMaterialStore: Send + Sync {
    fn read(&self, identity: &KeyIdentity) -> Result<Option<Vec<u8>>, BackendError>;
    /// Linearizable no-clobber: inserts only absent bytes; same bytes report AlreadyPresentIdentical; differing bytes fail.
    fn write_if_absent(
        &self,
        identity: &KeyIdentity,
        bytes: &[u8],
    ) -> Result<KeyWriteOutcome, BackendError>;
}
pub trait WitnessBackend: Send + Sync {
    fn capabilities(&self) -> BackendCapabilities;
    fn estimate_resources(&self, job: WitnessJob) -> Result<ResourceRequest, BackendError>;
    fn generate(&self, job: WitnessJob, output: PathBuf) -> Result<WitnessArtifact, BackendError>;
}
pub trait ProofBackend: Send + Sync {
    fn capabilities(&self) -> BackendCapabilities;
    fn estimate_resources(&self, job: ProofJob) -> Result<ResourceRequest, BackendError>;
    fn prepare(
        &self,
        job: PrepareJob,
        store: &dyn KeyMaterialStore,
    ) -> Result<PreparedKeys, BackendError>;
    fn prove(&self, job: ProofJob, output: PathBuf) -> Result<UnverifiedProof, BackendError>;
    fn verify_cryptographically(
        &self,
        request: CryptoVerificationRequest,
    ) -> Result<CryptoVerificationReceipt, VerificationError>;
}
#[non_exhaustive]
#[derive(Debug, Error, Serialize, Deserialize)]
pub enum BackendError {
    #[error("invalid job: {message}")]
    InvalidJob { message: String },
    #[error("I/O {operation} at {path}: {kind}: {message}")]
    Io {
        operation: String,
        path: PathBuf,
        kind: String,
        message: String,
    },
    #[error("serialization {format}: {message}")]
    Serialization { format: String, message: String },
    #[error("key digest mismatch")]
    KeyDigestMismatch,
    #[error("key conflict")]
    KeyConflict,
    #[error("proving failed: {message}")]
    ProvingFailed { message: String },
    #[error("resource error: {message}")]
    Resource { message: String },
    #[error("missing key: {identity:?}")]
    MissingKey { identity: KeyIdentity },
    #[error("unsupported instruction: {message}")]
    UnsupportedInstruction { message: String },
    #[error("missing register {register}")]
    MissingRegister { register: String },
    #[error("instruction {instruction} has forward register {register}")]
    ForwardRegister {
        instruction: usize,
        register: String,
    },
    #[error("instruction {instruction} shape mismatch: {message}")]
    ShapeMismatch { instruction: usize, message: String },
    #[error("instruction {instruction} arithmetic overflow: {message}")]
    ArithmeticOverflow { instruction: usize, message: String },
}
#[non_exhaustive]
#[derive(Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationError {
    #[error("proof flavor mismatch")]
    ProofFlavorMismatch,
    #[error("execution backend mismatch")]
    ExecutionBackendMismatch,
    #[error("circuit digest mismatch")]
    CircuitDigestMismatch,
    #[error("VK digest mismatch")]
    VerificationKeyDigestMismatch,
    #[error("statement digest mismatch")]
    PublicStatementDigestMismatch,
    #[error("proof digest mismatch")]
    ProofDigestMismatch,
    #[error("artifact manifest digest mismatch")]
    ArtifactDigestMismatch,
    #[error("run identity mismatch")]
    RunIdentityDigestMismatch,
    #[error("shard mismatch")]
    ShardIdentityMismatch,
    #[error("unsupported proof flavor")]
    UnsupportedProofFlavor,
    #[error("invalid capability: {message}")]
    InvalidCapability { message: String },
    #[error("cryptographic verification failed: {message}")]
    CryptographicVerificationFailed { message: String },
    #[error("crypto receipt binding mismatch")]
    ReceiptBindingMismatch,
    #[error("missing proof artifact")]
    MissingProofArtifact,
    #[error("missing artifact manifest")]
    MissingArtifactManifest,
    #[error("malformed proof artifact")]
    MalformedProofArtifact,
}
#[cfg(test)]
pub(crate) struct InMemoryKeyMaterialStore(
    std::sync::Mutex<std::collections::HashMap<KeyIdentity, Vec<u8>>>,
);
#[cfg(test)]
impl InMemoryKeyMaterialStore {
    pub(crate) fn new() -> Self {
        Self(std::sync::Mutex::new(std::collections::HashMap::new()))
    }
}

#[cfg(test)]
impl KeyMaterialStore for InMemoryKeyMaterialStore {
    fn read(&self, i: &KeyIdentity) -> Result<Option<Vec<u8>>, BackendError> {
        let v = self
            .0
            .lock()
            .map_err(|_| BackendError::KeyConflict)?
            .get(i)
            .cloned();
        if let Some(ref b) = v {
            if digest(b) != i.key_digest {
                return Err(BackendError::KeyDigestMismatch);
            }
        }
        Ok(v)
    }
    fn write_if_absent(&self, i: &KeyIdentity, b: &[u8]) -> Result<KeyWriteOutcome, BackendError> {
        if digest(b) != i.key_digest {
            return Err(BackendError::KeyDigestMismatch);
        }
        let mut m = self.0.lock().map_err(|_| BackendError::KeyConflict)?;
        match m.get(i) {
            None => {
                m.insert(i.clone(), b.to_vec());
                Ok(KeyWriteOutcome::Inserted)
            }
            Some(old) if old == b => Ok(KeyWriteOutcome::AlreadyPresentIdentical),
            Some(_) => Err(BackendError::KeyConflict),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    #[test]
    fn key_store_hashes_reads_writes_and_rejects_concurrent_conflicts() {
        let store = Arc::new(InMemoryKeyMaterialStore::new());
        let bytes = b"key".to_vec();
        let identity = KeyIdentity::new(
            ProofFlavorId::parse("test-v1").unwrap(),
            Digest32::new([1; 32]),
            digest(&bytes),
        )
        .unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let first_store = store.clone();
        let first_identity = identity.clone();
        let first_barrier = barrier.clone();
        let first_bytes = bytes.clone();
        let first = thread::spawn(move || {
            first_barrier.wait();
            first_store.write_if_absent(&first_identity, &first_bytes)
        });
        let other_store = store.clone();
        let other_identity = identity.clone();
        let other_barrier = barrier.clone();
        let other_bytes = bytes.clone();
        let second = thread::spawn(move || {
            other_barrier.wait();
            other_store.write_if_absent(&other_identity, &other_bytes)
        });
        barrier.wait();
        let outcomes = [
            first.join().unwrap().unwrap(),
            second.join().unwrap().unwrap(),
        ];
        assert!(outcomes.contains(&KeyWriteOutcome::Inserted));
        assert!(outcomes.contains(&KeyWriteOutcome::AlreadyPresentIdentical));
        // Different bytes cannot satisfy this identity's digest; rejection occurs before lock acquisition.
        assert!(matches!(
            store.write_if_absent(&identity, b"other"),
            Err(BackendError::KeyDigestMismatch)
        ));
        assert_eq!(store.read(&identity).unwrap(), Some(bytes));
        store
            .0
            .lock()
            .unwrap()
            .insert(identity.clone(), b"corrupt".to_vec());
        assert!(matches!(
            store.read(&identity),
            Err(BackendError::KeyDigestMismatch)
        ));
    }
}
