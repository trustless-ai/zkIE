use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Path, PathBuf};

use rustix::fs::{
    fchmod, fstat, fsync, mkdirat, open, openat, renameat_with, statat, unlinkat, AtFlags, Dir,
    FileType, Mode, OFlags, RenameFlags,
};
use rustix::io::{dup, Errno};
use thiserror::Error;
use zkie_types::{Digest32, ExecutionBackendId, ProofFlavorId};

use crate::{AttemptId, JobId, JobKind};

#[cfg(test)]
static BEFORE_LINK_HOOK: std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>> =
    std::sync::Mutex::new(None);

const MAGIC: &[u8; 8] = b"ZKIEOBJ2";
const DIGEST_DOMAIN: &[u8] = b"zkie-content-store-v1\0";
const PROOF_IDENTITY_DOMAIN: &[u8] = b"zkie-proof-identity-v1\0";
const PROOF_ATTESTATION_DOMAIN: &[u8] = b"zkie-proof-attestation-v1\0";
const KEY_IDENTITY_DOMAIN: &[u8] = b"zkie-key-identity-v1\0";
const KEY_ATTESTATION_DOMAIN: &[u8] = b"zkie-key-attestation-v1\0";
const MAX_METADATA_BYTES: usize = 4096;
const STAGE_PREFIX: &str = "stage-";
const STAGE_SUFFIX: &str = ".tmp";
const STORE_ID_NAME: &str = ".zkie-store-id";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactMetadata {
    artifact_id: String,
    content_digest: Digest32,
    size: u64,
    proof_identity_digest: Option<Digest32>,
}

impl ArtifactMetadata {
    pub fn new(
        artifact_id: impl Into<String>,
        content_digest: Digest32,
        size: u64,
    ) -> Result<Self, StoreError> {
        let artifact_id = artifact_id.into();
        validate_identifier(&artifact_id)?;
        Ok(Self {
            artifact_id,
            content_digest,
            size,
            proof_identity_digest: None,
        })
    }

    pub fn new_proof(
        artifact_id: impl Into<String>,
        content_digest: Digest32,
        size: u64,
        proof_identity_digest: Digest32,
    ) -> Result<Self, StoreError> {
        let mut metadata = Self::new(artifact_id, content_digest, size)?;
        metadata.proof_identity_digest = Some(proof_identity_digest);
        Ok(metadata)
    }

    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    pub fn proof_identity_digest(&self) -> Option<Digest32> {
        self.proof_identity_digest
    }

    pub fn content_digest(&self) -> Digest32 {
        self.content_digest
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactRole {
    Witness,
    PreparedMaterial,
    LeafProof,
    NativeVerifiedManifest,
}

impl ArtifactRole {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Witness => "witness",
            Self::PreparedMaterial => "prepared-material",
            Self::LeafProof => "leaf-proof",
            Self::NativeVerifiedManifest => "native-verified-manifest",
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::Witness => 1,
            Self::PreparedMaterial => 2,
            Self::LeafProof => 3,
            Self::NativeVerifiedManifest => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationArity {
    NotApplicable,
    Actual(u32),
}

impl AggregationArity {
    fn encode_into(self, output: &mut Vec<u8>) -> Result<(), StoreError> {
        match self {
            Self::NotApplicable => output.push(0),
            Self::Actual(arity) if arity >= 2 => {
                output.push(1);
                output.extend_from_slice(&arity.to_le_bytes());
            }
            Self::Actual(_) => return Err(StoreError::InvalidMetadata),
        }
        Ok(())
    }

    pub(crate) fn audit_value(self) -> (i64, Option<u32>) {
        match self {
            Self::NotApplicable => (0, None),
            Self::Actual(value) => (1, Some(value)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyMetadata {
    content_digest: Digest32,
    size: u64,
    srs_source_digest: Digest32,
    proof_flavor: ProofFlavorId,
    circuit_digest: Digest32,
    k: u32,
    aggregation_arity: u32,
    key_identity_digest: Option<Digest32>,
}

impl KeyMetadata {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        content_digest: Digest32,
        size: u64,
        srs_source_digest: Digest32,
        proof_flavor: ProofFlavorId,
        circuit_digest: Digest32,
        k: u32,
        aggregation_arity: u32,
    ) -> Result<Self, StoreError> {
        validate_identifier(proof_flavor.as_str())?;
        if aggregation_arity == 0 {
            return Err(StoreError::InvalidMetadata);
        }
        Ok(Self {
            content_digest,
            size,
            srs_source_digest,
            proof_flavor,
            circuit_digest,
            k,
            aggregation_arity,
            key_identity_digest: None,
        })
    }

    pub fn bind_identity(mut self, key_identity_digest: Digest32) -> Self {
        self.key_identity_digest = Some(key_identity_digest);
        self
    }

    pub fn key_identity_digest(&self) -> Option<Digest32> {
        self.key_identity_digest
    }

    pub fn srs_source_digest(&self) -> Digest32 {
        self.srs_source_digest
    }
    pub fn content_digest(&self) -> Digest32 {
        self.content_digest
    }
    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn proof_flavor(&self) -> &ProofFlavorId {
        &self.proof_flavor
    }
    pub fn circuit_digest(&self) -> Digest32 {
        self.circuit_digest
    }
    pub fn k(&self) -> u32 {
        self.k
    }
    pub fn aggregation_arity(&self) -> u32 {
        self.aggregation_arity
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyRole {
    ProvingAndVerifyingKey,
}

impl KeyRole {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ProvingAndVerifyingKey => "proving-and-verifying-key",
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::ProvingAndVerifyingKey => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedKeyIdentity {
    role: KeyRole,
    job_kind: JobKind,
    run_id: String,
    job: JobId,
    attempt: AttemptId,
    metadata: KeyMetadata,
}

impl TrustedKeyIdentity {
    pub fn new(
        role: KeyRole,
        job_kind: JobKind,
        run_id: &str,
        job: &JobId,
        attempt: &AttemptId,
        metadata: KeyMetadata,
    ) -> Result<Self, StoreError> {
        validate_identifier(run_id)?;
        if job_kind != JobKind::Prepare {
            return Err(StoreError::InvalidMetadata);
        }
        let identity = Self {
            role,
            job_kind,
            run_id: run_id.to_owned(),
            job: job.clone(),
            attempt: attempt.clone(),
            metadata,
        };
        identity.canonical_bytes()?;
        Ok(identity)
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, StoreError> {
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(KEY_IDENTITY_DOMAIN);
        bytes.push(self.role.tag());
        write_string(&mut bytes, self.job_kind.as_str())?;
        write_string(&mut bytes, &self.run_id)?;
        write_string(&mut bytes, self.job.as_str())?;
        write_string(&mut bytes, self.attempt.as_str())?;
        bytes.extend_from_slice(self.metadata.content_digest.as_bytes());
        bytes.extend_from_slice(&self.metadata.size.to_le_bytes());
        bytes.extend_from_slice(self.metadata.srs_source_digest.as_bytes());
        write_string(&mut bytes, self.metadata.proof_flavor.as_str())?;
        bytes.extend_from_slice(self.metadata.circuit_digest.as_bytes());
        bytes.extend_from_slice(&self.metadata.k.to_le_bytes());
        bytes.extend_from_slice(&self.metadata.aggregation_arity.to_le_bytes());
        Ok(bytes)
    }

    pub fn digest(&self) -> Digest32 {
        Digest32::new(
            *blake3::hash(
                &self
                    .canonical_bytes()
                    .expect("validated key identity remains encodable"),
            )
            .as_bytes(),
        )
    }

    fn attestation_digest(&self, object_digest: Digest32) -> Digest32 {
        key_attestation_digest(
            self.digest(),
            object_digest,
            &self.run_id,
            &self.job,
            &self.attempt,
        )
    }

    pub(crate) fn binding(&self) -> (&str, &JobId, &AttemptId) {
        (&self.run_id, &self.job, &self.attempt)
    }

    pub(crate) fn role(&self) -> &'static str {
        self.role.as_str()
    }

    pub(crate) fn job_kind(&self) -> &'static str {
        self.job_kind.as_str()
    }

    pub(crate) fn metadata(&self) -> &KeyMetadata {
        &self.metadata
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObjectMetadata {
    Artifact(ArtifactMetadata),
    Key(KeyMetadata),
}

impl ObjectMetadata {
    fn content_digest(&self) -> Digest32 {
        match self {
            Self::Artifact(value) => value.content_digest,
            Self::Key(value) => value.content_digest,
        }
    }

    pub fn size(&self) -> u64 {
        match self {
            Self::Artifact(value) => value.size,
            Self::Key(value) => value.size,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedProofIdentity {
    artifact_role: ArtifactRole,
    job_kind: JobKind,
    k: u32,
    aggregation_arity: AggregationArity,
    run_id: String,
    job: JobId,
    attempt: AttemptId,
    proof_content_digest: Digest32,
    artifact_manifest_digest: Digest32,
    run_identity_digest: Digest32,
    public_statement_digest: Digest32,
    circuit_digest: Digest32,
    verifying_key_digest: Digest32,
    srs_source_digest: Digest32,
    shard_identity_digest: Digest32,
    witness_artifact_digest: Digest32,
    proof_flavor: ProofFlavorId,
    execution_backend: ExecutionBackendId,
}

impl TrustedProofIdentity {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        artifact_role: ArtifactRole,
        job_kind: JobKind,
        k: u32,
        aggregation_arity: AggregationArity,
        run_id: &str,
        job: &JobId,
        attempt: &AttemptId,
        proof_content_digest: Digest32,
        artifact_manifest_digest: Digest32,
        run_identity_digest: Digest32,
        public_statement_digest: Digest32,
        circuit_digest: Digest32,
        verifying_key_digest: Digest32,
        srs_source_digest: Digest32,
        shard_identity_digest: Digest32,
        witness_artifact_digest: Digest32,
        proof_flavor: ProofFlavorId,
        execution_backend: ExecutionBackendId,
    ) -> Result<Self, StoreError> {
        validate_identifier(proof_flavor.as_str())?;
        validate_identifier(execution_backend.as_str())?;
        validate_identifier(run_id)?;
        if !matches!(
            (artifact_role, job_kind, aggregation_arity),
            (
                ArtifactRole::Witness,
                JobKind::Witness,
                AggregationArity::NotApplicable
            ) | (
                ArtifactRole::PreparedMaterial,
                JobKind::Prepare,
                AggregationArity::NotApplicable
            ) | (
                ArtifactRole::LeafProof,
                JobKind::LeafProof,
                AggregationArity::NotApplicable
            ) | (
                ArtifactRole::NativeVerifiedManifest,
                JobKind::NativeAggregate,
                AggregationArity::Actual(_)
            )
        ) {
            return Err(StoreError::InvalidMetadata);
        }
        let identity = Self {
            artifact_role,
            job_kind,
            k,
            aggregation_arity,
            run_id: run_id.to_owned(),
            job: job.clone(),
            attempt: attempt.clone(),
            proof_content_digest,
            artifact_manifest_digest,
            run_identity_digest,
            public_statement_digest,
            circuit_digest,
            verifying_key_digest,
            srs_source_digest,
            shard_identity_digest,
            witness_artifact_digest,
            proof_flavor,
            execution_backend,
        };
        identity.canonical_bytes()?;
        Ok(identity)
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, StoreError> {
        let mut bytes = Vec::with_capacity(512);
        bytes.extend_from_slice(PROOF_IDENTITY_DOMAIN);
        bytes.push(self.artifact_role.tag());
        write_string(&mut bytes, self.job_kind.as_str())?;
        bytes.extend_from_slice(&self.k.to_le_bytes());
        self.aggregation_arity.encode_into(&mut bytes)?;
        write_string(&mut bytes, &self.run_id)?;
        write_string(&mut bytes, self.job.as_str())?;
        write_string(&mut bytes, self.attempt.as_str())?;
        for digest in [
            self.proof_content_digest,
            self.artifact_manifest_digest,
            self.run_identity_digest,
            self.public_statement_digest,
            self.circuit_digest,
            self.verifying_key_digest,
            self.srs_source_digest,
            self.shard_identity_digest,
            self.witness_artifact_digest,
        ] {
            bytes.extend_from_slice(digest.as_bytes());
        }
        write_string(&mut bytes, self.proof_flavor.as_str())?;
        write_string(&mut bytes, self.execution_backend.as_str())?;
        Ok(bytes)
    }

    pub fn digest(&self) -> Digest32 {
        Digest32::new(
            *blake3::hash(
                &self
                    .canonical_bytes()
                    .expect("validated proof identity remains encodable"),
            )
            .as_bytes(),
        )
    }

    fn attestation_digest(&self, object_digest: Digest32) -> Digest32 {
        proof_attestation_digest(
            self.digest(),
            object_digest,
            &self.run_id,
            &self.job,
            &self.attempt,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn verifier_id(&self) -> &str {
        self.execution_backend.as_str()
    }

    pub(crate) fn binding(&self) -> (&str, &JobId, &AttemptId) {
        (&self.run_id, &self.job, &self.attempt)
    }

    pub(crate) fn audit(&self) -> ProofIdentityAudit<'_> {
        ProofIdentityAudit(self)
    }
}

pub(crate) struct ProofIdentityAudit<'a>(&'a TrustedProofIdentity);

impl ProofIdentityAudit<'_> {
    pub(crate) fn role(&self) -> &'static str {
        self.0.artifact_role.as_str()
    }
    pub(crate) fn job_kind(&self) -> &'static str {
        self.0.job_kind.as_str()
    }
    pub(crate) fn k(&self) -> u32 {
        self.0.k
    }
    pub(crate) fn aggregation_arity(&self) -> (i64, Option<u32>) {
        self.0.aggregation_arity.audit_value()
    }
    pub(crate) fn artifact_manifest_digest(&self) -> Digest32 {
        self.0.artifact_manifest_digest
    }
    pub(crate) fn run_identity_digest(&self) -> Digest32 {
        self.0.run_identity_digest
    }
    pub(crate) fn public_statement_digest(&self) -> Digest32 {
        self.0.public_statement_digest
    }
    pub(crate) fn circuit_digest(&self) -> Digest32 {
        self.0.circuit_digest
    }
    pub(crate) fn verifying_key_digest(&self) -> Digest32 {
        self.0.verifying_key_digest
    }
    pub(crate) fn srs_source_digest(&self) -> Digest32 {
        self.0.srs_source_digest
    }
    pub(crate) fn shard_identity_digest(&self) -> Digest32 {
        self.0.shard_identity_digest
    }
    pub(crate) fn witness_artifact_digest(&self) -> Digest32 {
        self.0.witness_artifact_digest
    }
    pub(crate) fn proof_flavor(&self) -> &str {
        self.0.proof_flavor.as_str()
    }
    pub(crate) fn execution_backend(&self) -> &str {
        self.0.execution_backend.as_str()
    }
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("store I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("store filesystem operation failed: {0}")]
    Fs(#[from] Errno),
    #[error("atomic descriptor-relative storage is unsupported on this platform")]
    UnsupportedPlatform,
    #[error("invalid object metadata")]
    InvalidMetadata,
    #[error("staged metadata did not match the typed expectation")]
    MetadataMismatch,
    #[error("object content digest or size did not match")]
    ContentMismatch,
    #[error("payload exceeds its declared size")]
    PayloadTooLarge,
    #[error("object metadata belongs to the other store kind")]
    WrongStoreKind,
    #[error("symbolic link or non-regular object rejected")]
    UnsafeFileType,
    #[error("an existing object conflicts with digest {digest}")]
    ExistingObjectConflict { digest: Digest32 },
    #[error("object does not exist for digest {digest}")]
    NotFound { digest: Digest32 },
    #[error("validated file identity changed before publication")]
    FileIdentityChanged,
    #[error("trusted proof identity does not match the exact published artifact")]
    ProofIdentityMismatch,
    #[error("trusted key identity does not match the exact published key object")]
    KeyIdentityMismatch,
    #[error("the named store root or objects directory no longer matches the pinned store")]
    StoreLocatorChanged,
}

pub trait ContentStore {
    fn stage(&self, expected: &ObjectMetadata) -> Result<StagedObject, StoreError>;
    fn validate(&self, staged: StagedObject) -> Result<ValidatedObject, StoreError>;
    fn publish(&self, validated: ValidatedObject) -> Result<PublishedObject, StoreError>;
    fn open_verified(&self, digest: Digest32) -> Result<File, StoreError>;
}

#[derive(Debug)]
pub struct StagedObject {
    name: String,
    expected: ObjectMetadata,
    file: File,
    written: u64,
}

impl Write for StagedObject {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let requested = u64::try_from(bytes.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, StoreError::PayloadTooLarge)
        })?;
        let next = self.written.checked_add(requested).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, StoreError::PayloadTooLarge)
        })?;
        if next > self.expected.size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                StoreError::PayloadTooLarge,
            ));
        }
        let count = self.file.write(bytes)?;
        self.written = self.written.checked_add(count as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, StoreError::PayloadTooLarge)
        })?;
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[derive(Debug)]
pub struct ValidatedObject {
    name: String,
    metadata: ObjectMetadata,
    digest: Digest32,
    file: File,
    identity: FileIdentity,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct PublishedObject {
    name: String,
    metadata: ObjectMetadata,
    digest: Digest32,
    file: File,
    root: PathBuf,
    root_fd: OwnedFd,
    objects_fd: OwnedFd,
    store_identity_digest: Digest32,
}

impl PublishedObject {
    pub fn metadata(&self) -> &ObjectMetadata {
        &self.metadata
    }
    pub fn digest(&self) -> Digest32 {
        self.digest
    }

    #[allow(dead_code)]
    pub(crate) fn artifact_id(&self) -> Option<&str> {
        match &self.metadata {
            ObjectMetadata::Artifact(value) => Some(value.artifact_id()),
            ObjectMetadata::Key(_) => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn store_identity_digest(&self) -> Digest32 {
        self.store_identity_digest
    }

    #[allow(dead_code)]
    pub(crate) fn ensure_verified(&self) -> Result<(), StoreError> {
        let current = open_regular_at(&self.objects_fd, &self.name, OFlags::RDONLY)?;
        if file_identity(&current)? != file_identity(&self.file)? {
            return Err(StoreError::FileIdentityChanged);
        }
        let (metadata, digest) = verify_envelope(&current, Some(&self.metadata))?;
        if metadata != self.metadata || digest != self.digest {
            return Err(StoreError::ContentMismatch);
        }
        Ok(())
    }

    pub(crate) fn ensure_pinned_file_identity(&self) -> Result<(), StoreError> {
        let current = open_regular_at(&self.objects_fd, &self.name, OFlags::RDONLY)?;
        if file_identity(&current)? != file_identity(&self.file)? {
            return Err(StoreError::FileIdentityChanged);
        }
        Ok(())
    }

    pub(crate) fn ensure_current_locator(&self) -> Result<(), StoreError> {
        let named_root = open(
            &self.root,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )?;
        if FileIdentity::from_stat(&fstat(&named_root)?)
            != FileIdentity::from_stat(&fstat(&self.root_fd)?)
        {
            return Err(StoreError::StoreLocatorChanged);
        }
        let named_objects = statat(&named_root, "objects", AtFlags::SYMLINK_NOFOLLOW)?;
        if !is_directory_mode(named_objects.st_mode)
            || FileIdentity::from_stat(&named_objects)
                != FileIdentity::from_stat(&fstat(&self.objects_fd)?)
        {
            return Err(StoreError::StoreLocatorChanged);
        }
        Ok(())
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VerifiedArtifact {
    identity: TrustedProofIdentity,
    published: PublishedObject,
}

#[allow(dead_code)]
impl VerifiedArtifact {
    pub(crate) fn attest(
        identity: TrustedProofIdentity,
        published: PublishedObject,
    ) -> Result<Self, StoreError> {
        published.ensure_verified()?;
        match published.metadata() {
            ObjectMetadata::Artifact(metadata)
                if metadata.content_digest == identity.proof_content_digest
                    && metadata.proof_identity_digest == Some(identity.digest()) => {}
            ObjectMetadata::Artifact(_) | ObjectMetadata::Key(_) => {
                return Err(StoreError::ProofIdentityMismatch)
            }
        }
        Ok(Self {
            identity,
            published,
        })
    }

    pub(crate) fn binding(
        &self,
    ) -> (
        &str,
        &JobId,
        &AttemptId,
        &TrustedProofIdentity,
        &PublishedObject,
    ) {
        let (run, job, attempt) = self.identity.binding();
        (run, job, attempt, &self.identity, &self.published)
    }

    pub(crate) fn attestation_digest(&self) -> Digest32 {
        self.identity.attestation_digest(self.published.digest())
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VerifiedKey {
    identity: TrustedKeyIdentity,
    published: PublishedObject,
}

#[allow(dead_code)]
impl VerifiedKey {
    pub(crate) fn attest(
        identity: TrustedKeyIdentity,
        published: PublishedObject,
    ) -> Result<Self, StoreError> {
        published.ensure_verified()?;
        match published.metadata() {
            ObjectMetadata::Key(metadata)
                if metadata.content_digest == identity.metadata.content_digest
                    && metadata.size == identity.metadata.size
                    && metadata.srs_source_digest == identity.metadata.srs_source_digest
                    && metadata.proof_flavor == identity.metadata.proof_flavor
                    && metadata.circuit_digest == identity.metadata.circuit_digest
                    && metadata.k == identity.metadata.k
                    && metadata.aggregation_arity == identity.metadata.aggregation_arity
                    && metadata.key_identity_digest == Some(identity.digest()) => {}
            ObjectMetadata::Artifact(_) | ObjectMetadata::Key(_) => {
                return Err(StoreError::KeyIdentityMismatch);
            }
        }
        Ok(Self {
            identity,
            published,
        })
    }

    pub(crate) fn binding(
        &self,
    ) -> (
        &str,
        &JobId,
        &AttemptId,
        &TrustedKeyIdentity,
        &PublishedObject,
    ) {
        let (run, job, attempt) = self.identity.binding();
        (run, job, attempt, &self.identity, &self.published)
    }

    pub(crate) fn attestation_digest(&self) -> Digest32 {
        self.identity.attestation_digest(self.published.digest())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub removed: usize,
    pub quarantined: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StoreKind {
    Artifact,
    Key,
}

#[derive(Debug)]
struct StoreCore {
    root: PathBuf,
    #[allow(dead_code)]
    root_fd: OwnedFd,
    staging_fd: OwnedFd,
    objects_fd: OwnedFd,
    quarantine_fd: OwnedFd,
    store_identity_digest: Digest32,
    kind: StoreKind,
}

#[derive(Debug)]
pub struct ArtifactStore(StoreCore);
#[derive(Debug)]
pub struct KeyStore(StoreCore);

macro_rules! impl_store {
    ($name:ident, $kind:expr) => {
        impl $name {
            pub fn open(root: impl AsRef<Path>) -> Result<Self, StoreError> {
                StoreCore::open(root.as_ref(), $kind).map(Self)
            }
            pub fn recover_stale(&self) -> Result<RecoveryReport, StoreError> {
                self.0.recover_stale()
            }
            pub(crate) fn reopen_persisted(
                &self,
                store_identity: Digest32,
                digest: Digest32,
            ) -> Result<(File, ObjectMetadata), StoreError> {
                self.0.ensure_current_locator()?;
                if self.0.store_identity_digest()? != store_identity {
                    return Err(StoreError::StoreLocatorChanged);
                }
                self.0.open_verified_with_metadata(digest)
            }
        }
        impl ContentStore for $name {
            fn stage(&self, expected: &ObjectMetadata) -> Result<StagedObject, StoreError> {
                self.0.stage(expected)
            }
            fn validate(&self, staged: StagedObject) -> Result<ValidatedObject, StoreError> {
                self.0.validate(staged)
            }
            fn publish(&self, validated: ValidatedObject) -> Result<PublishedObject, StoreError> {
                self.0.publish(validated)
            }
            fn open_verified(&self, digest: Digest32) -> Result<File, StoreError> {
                self.0.open_verified(digest)
            }
        }
    };
}

impl_store!(ArtifactStore, StoreKind::Artifact);
impl_store!(KeyStore, StoreKind::Key);

impl StoreCore {
    fn open(root: &Path, kind: StoreKind) -> Result<Self, StoreError> {
        #[cfg(not(unix))]
        return Err(StoreError::UnsupportedPlatform);
        #[cfg(unix)]
        {
            if fs::symlink_metadata(root).is_ok_and(|value| value.file_type().is_symlink()) {
                return Err(StoreError::UnsafeFileType);
            }
            fs::create_dir_all(root)?;
            let root = fs::canonicalize(root)?;
            let root_fd = open(
                &root,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )?;
            let staging_fd = ensure_owned_dir(&root_fd, "staging")?;
            let objects_fd = ensure_owned_dir(&root_fd, "objects")?;
            let quarantine_fd = ensure_owned_dir(&root_fd, "quarantine")?;
            let store_identity_digest = ensure_store_identity(&root_fd)?;
            Ok(Self {
                root,
                root_fd,
                staging_fd,
                objects_fd,
                quarantine_fd,
                store_identity_digest,
                kind,
            })
        }
    }

    fn stage(&self, expected: &ObjectMetadata) -> Result<StagedObject, StoreError> {
        self.require_kind(expected)?;
        let metadata = encode_metadata(expected)?;
        loop {
            let name = random_name(STAGE_PREFIX, STAGE_SUFFIX)?;
            match openat(
                &self.staging_fd,
                &name,
                OFlags::RDWR | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::RUSR | Mode::WUSR,
            ) {
                Ok(fd) => {
                    let mut file = File::from(fd);
                    file.write_all(MAGIC)?;
                    file.write_all(&(metadata.len() as u32).to_le_bytes())?;
                    file.write_all(&metadata)?;
                    return Ok(StagedObject {
                        name,
                        expected: expected.clone(),
                        file,
                        written: 0,
                    });
                }
                Err(Errno::EXIST) => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }

    fn validate(&self, mut staged: StagedObject) -> Result<ValidatedObject, StoreError> {
        let name = staged.name.clone();
        let result = (|| {
            if staged.written != staged.expected.size() {
                return Err(StoreError::ContentMismatch);
            }
            staged.file.flush()?;
            staged.file.sync_all()?;
            fchmod(staged.file.as_fd(), Mode::RUSR)?;
            staged.file.sync_all()?;
            let identity = file_identity(&staged.file)?;
            let named = statat(&self.staging_fd, &staged.name, AtFlags::SYMLINK_NOFOLLOW)?;
            if !is_regular_mode(named.st_mode) || FileIdentity::from_stat(&named) != identity {
                return Err(StoreError::FileIdentityChanged);
            }
            let (metadata, digest) = verify_envelope(&staged.file, Some(&staged.expected))?;
            self.require_kind(&metadata)?;
            Ok(ValidatedObject {
                name: staged.name,
                metadata,
                digest,
                file: staged.file,
                identity,
            })
        })();
        if result.is_err() {
            let _ = unlinkat(&self.staging_fd, &name, AtFlags::empty());
            let _ = fsync(&self.staging_fd);
        }
        result
    }

    fn publish(&self, validated: ValidatedObject) -> Result<PublishedObject, StoreError> {
        let (metadata, digest) = verify_envelope(&validated.file, Some(&validated.metadata))?;
        if digest != validated.digest || file_identity(&validated.file)? != validated.identity {
            return Err(StoreError::FileIdentityChanged);
        }
        let named = statat(&self.staging_fd, &validated.name, AtFlags::SYMLINK_NOFOLLOW)?;
        if !is_regular_mode(named.st_mode) || FileIdentity::from_stat(&named) != validated.identity
        {
            return Err(StoreError::FileIdentityChanged);
        }
        #[cfg(test)]
        if let Some(hook) = BEFORE_LINK_HOOK
            .lock()
            .expect("publish hook poisoned")
            .take()
        {
            hook();
        }
        // Publication never links the worker-owned inode into the durable object store. A worker
        // may retain a writable descriptor even after validation and chmod; copying from the pinned
        // validated descriptor gives the scheduler a fresh inode that such a descriptor cannot
        // mutate after the database commit.
        let publication_name = random_name(STAGE_PREFIX, STAGE_SUFFIX)?;
        let publication = match self.copy_for_publication(&validated, &publication_name) {
            Ok(file) => file,
            Err(error) => {
                let _ = unlinkat(&self.staging_fd, &publication_name, AtFlags::empty());
                return Err(error);
            }
        };
        let final_name = digest.to_string();
        match renameat_with(
            &self.staging_fd,
            &publication_name,
            &self.objects_fd,
            &final_name,
            RenameFlags::NOREPLACE,
        ) {
            Ok(()) => {}
            Err(Errno::EXIST) => {
                unlinkat(&self.staging_fd, &publication_name, AtFlags::empty())?;
                let existing = open_regular_at(&self.objects_fd, &final_name, OFlags::RDONLY)?;
                let (existing_metadata, existing_digest) =
                    verify_envelope(&existing, Some(&metadata))?;
                if existing_metadata != metadata || existing_digest != digest {
                    return Err(StoreError::ExistingObjectConflict { digest });
                }
                unlinkat(&self.staging_fd, &validated.name, AtFlags::empty())?;
                fsync(&self.staging_fd)?;
                return self.published(final_name, metadata, digest, existing);
            }
            Err(error) => {
                let _ = unlinkat(&self.staging_fd, &publication_name, AtFlags::empty());
                return Err(error.into());
            }
        }
        let final_file = match open_regular_at(&self.objects_fd, &final_name, OFlags::RDONLY) {
            Ok(file) => file,
            Err(error) => {
                self.remove_failed_publication(&final_name);
                return Err(error);
            }
        };
        if file_identity(&final_file)? != file_identity(&publication)? {
            self.remove_failed_publication(&final_name);
            return Err(StoreError::FileIdentityChanged);
        }
        let final_digest = match verify_envelope(&final_file, Some(&metadata)) {
            Ok((_, digest)) => digest,
            Err(error) => {
                self.remove_failed_publication(&final_name);
                return Err(error);
            }
        };
        if final_digest != digest {
            self.remove_failed_publication(&final_name);
            return Err(StoreError::ContentMismatch);
        }
        fsync(&self.objects_fd)?;
        unlinkat(&self.staging_fd, &validated.name, AtFlags::empty())?;
        fsync(&self.staging_fd)?;
        self.published(final_name, metadata, digest, final_file)
    }

    fn copy_for_publication(
        &self,
        validated: &ValidatedObject,
        publication_name: &str,
    ) -> Result<File, StoreError> {
        let fd = openat(
            &self.staging_fd,
            publication_name,
            OFlags::RDWR | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::RUSR | Mode::WUSR,
        )?;
        let mut destination = File::from(fd);
        let mut source = validated.file.try_clone()?;
        source.seek(SeekFrom::Start(0))?;
        let metadata_len = u64::try_from(encode_metadata(&validated.metadata)?.len())
            .map_err(|_| StoreError::InvalidMetadata)?;
        let mut remaining = 12_u64
            .checked_add(metadata_len)
            .and_then(|value| value.checked_add(validated.metadata.size()))
            .ok_or(StoreError::InvalidMetadata)?;
        let mut buffer = [0_u8; 64 * 1024];
        while remaining != 0 {
            let wanted = remaining.min(buffer.len() as u64) as usize;
            let count = source.read(&mut buffer[..wanted])?;
            if count == 0 {
                return Err(StoreError::ContentMismatch);
            }
            destination.write_all(&buffer[..count])?;
            remaining -= count as u64;
        }
        destination.flush()?;
        destination.sync_all()?;
        fchmod(destination.as_fd(), Mode::RUSR)?;
        destination.sync_all()?;
        let (metadata, digest) = verify_envelope(&destination, Some(&validated.metadata))?;
        if metadata != validated.metadata || digest != validated.digest {
            return Err(StoreError::ContentMismatch);
        }
        Ok(destination)
    }

    fn published(
        &self,
        name: String,
        metadata: ObjectMetadata,
        digest: Digest32,
        file: File,
    ) -> Result<PublishedObject, StoreError> {
        Ok(PublishedObject {
            name,
            metadata,
            digest,
            file,
            root: self.root.clone(),
            root_fd: dup(&self.root_fd)?,
            objects_fd: dup(&self.objects_fd)?,
            store_identity_digest: self.store_identity_digest,
        })
    }

    fn open_verified(&self, digest: Digest32) -> Result<File, StoreError> {
        self.open_verified_with_metadata(digest)
            .map(|value| value.0)
    }

    fn open_verified_with_metadata(
        &self,
        digest: Digest32,
    ) -> Result<(File, ObjectMetadata), StoreError> {
        let name = digest.to_string();
        let file = match open_regular_at(&self.objects_fd, &name, OFlags::RDONLY) {
            Ok(file) => file,
            Err(StoreError::Fs(Errno::NOENT)) => return Err(StoreError::NotFound { digest }),
            Err(error) => return Err(error),
        };
        let (metadata, actual) = verify_envelope(&file, None)?;
        self.require_kind(&metadata)?;
        if actual != digest {
            return Err(StoreError::ContentMismatch);
        }
        let mut file = file;
        seek_payload(&mut file)?;
        Ok((file, metadata))
    }

    fn recover_stale(&self) -> Result<RecoveryReport, StoreError> {
        let mut report = RecoveryReport::default();
        let directory = Dir::read_from(&self.staging_fd)?;
        for entry in directory {
            let entry = entry?;
            let name = entry.file_name();
            let bytes = name.to_bytes();
            if bytes == b"." || bytes == b".." {
                continue;
            }
            let safe_regular = entry.file_type() == FileType::RegularFile
                && std::str::from_utf8(bytes).is_ok_and(is_stage_name);
            if safe_regular {
                unlinkat(&self.staging_fd, name, AtFlags::empty())?;
                report.removed += 1;
            } else {
                loop {
                    let destination = random_name("quarantine-", "")?;
                    match renameat_with(
                        &self.staging_fd,
                        name,
                        &self.quarantine_fd,
                        &destination,
                        RenameFlags::NOREPLACE,
                    ) {
                        Ok(()) => {
                            report.quarantined += 1;
                            break;
                        }
                        Err(Errno::EXIST) => continue,
                        Err(error) => return Err(error.into()),
                    }
                }
            }
        }
        fsync(&self.staging_fd)?;
        if report.quarantined != 0 {
            fsync(&self.quarantine_fd)?;
        }
        Ok(report)
    }

    fn require_kind(&self, metadata: &ObjectMetadata) -> Result<(), StoreError> {
        if matches!(
            (self.kind, metadata),
            (StoreKind::Artifact, ObjectMetadata::Artifact(_))
                | (StoreKind::Key, ObjectMetadata::Key(_))
        ) {
            Ok(())
        } else {
            Err(StoreError::WrongStoreKind)
        }
    }

    fn store_identity_digest(&self) -> Result<Digest32, StoreError> {
        Ok(self.store_identity_digest)
    }

    fn ensure_current_locator(&self) -> Result<(), StoreError> {
        let named_root = open(
            &self.root,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )?;
        if FileIdentity::from_stat(&fstat(&named_root)?)
            != FileIdentity::from_stat(&fstat(&self.root_fd)?)
        {
            return Err(StoreError::StoreLocatorChanged);
        }
        let named_objects = statat(&named_root, "objects", AtFlags::SYMLINK_NOFOLLOW)?;
        if !is_directory_mode(named_objects.st_mode)
            || FileIdentity::from_stat(&named_objects)
                != FileIdentity::from_stat(&fstat(&self.objects_fd)?)
        {
            return Err(StoreError::StoreLocatorChanged);
        }
        Ok(())
    }

    fn remove_failed_publication(&self, name: &str) {
        let _ = unlinkat(&self.objects_fd, name, AtFlags::empty());
        let _ = fsync(&self.objects_fd);
    }
}

fn ensure_owned_dir(root: &OwnedFd, name: &str) -> Result<OwnedFd, StoreError> {
    match mkdirat(root, name, Mode::RWXU) {
        Ok(()) | Err(Errno::EXIST) => {}
        Err(error) => return Err(error.into()),
    }
    Ok(openat(
        root,
        name,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )?)
}

fn open_regular_at(dir: &OwnedFd, name: &str, access: OFlags) -> Result<File, StoreError> {
    let fd = openat(
        dir,
        name,
        access | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )?;
    let stat = fstat(&fd)?;
    if !is_regular_mode(stat.st_mode) {
        return Err(StoreError::UnsafeFileType);
    }
    Ok(File::from(fd))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

impl FileIdentity {
    fn from_stat(stat: &rustix::fs::Stat) -> Self {
        Self {
            device: stat.st_dev as u64,
            inode: stat.st_ino,
        }
    }
}

fn file_identity(file: &File) -> Result<FileIdentity, StoreError> {
    Ok(FileIdentity::from_stat(&fstat(file)?))
}

fn is_regular_mode(mode: rustix::fs::RawMode) -> bool {
    FileType::from_raw_mode(mode) == FileType::RegularFile
}

fn is_directory_mode(mode: rustix::fs::RawMode) -> bool {
    FileType::from_raw_mode(mode) == FileType::Directory
}

fn ensure_store_identity(root: &OwnedFd) -> Result<Digest32, StoreError> {
    let mut marker = match openat(
        root,
        STORE_ID_NAME,
        OFlags::RDWR | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    ) {
        Ok(fd) => {
            let mut file = File::from(fd);
            let mut random = [0_u8; 32];
            getrandom::getrandom(&mut random).map_err(|error| {
                io::Error::other(format!("secure randomness unavailable: {error}"))
            })?;
            file.write_all(&random)?;
            file.sync_all()?;
            fchmod(file.as_fd(), Mode::RUSR)?;
            file.sync_all()?;
            fsync(root)?;
            file
        }
        Err(Errno::EXIST) => open_regular_at(root, STORE_ID_NAME, OFlags::RDONLY)?,
        Err(error) => return Err(error.into()),
    };
    marker.seek(SeekFrom::Start(0))?;
    let mut identity = [0_u8; 32];
    marker
        .read_exact(&mut identity)
        .map_err(|_| StoreError::StoreLocatorChanged)?;
    let mut trailing = [0_u8; 1];
    if marker.read(&mut trailing)? != 0 {
        return Err(StoreError::StoreLocatorChanged);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"zkie-store-identity-v1\0");
    hasher.update(&identity);
    Ok(Digest32::new(*hasher.finalize().as_bytes()))
}

fn verify_envelope(
    file: &File,
    expected: Option<&ObjectMetadata>,
) -> Result<(ObjectMetadata, Digest32), StoreError> {
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(0))?;
    let mut magic = [0; 8];
    file.read_exact(&mut magic)
        .map_err(|_| StoreError::MetadataMismatch)?;
    if &magic != MAGIC {
        return Err(StoreError::MetadataMismatch);
    }
    let mut raw_len = [0; 4];
    file.read_exact(&mut raw_len)
        .map_err(|_| StoreError::MetadataMismatch)?;
    let metadata_len = u32::from_le_bytes(raw_len) as usize;
    if metadata_len > MAX_METADATA_BYTES {
        return Err(StoreError::MetadataMismatch);
    }
    let mut raw_metadata = vec![0; metadata_len];
    file.read_exact(&mut raw_metadata)
        .map_err(|_| StoreError::MetadataMismatch)?;
    let metadata = decode_metadata(&raw_metadata)?;
    if expected.is_some_and(|value| value != &metadata) {
        return Err(StoreError::MetadataMismatch);
    }
    let offset = 12_u64 + metadata_len as u64;
    if file.metadata()?.len().checked_sub(offset) != Some(metadata.size()) {
        return Err(StoreError::ContentMismatch);
    }
    let mut content_hasher = blake3::Hasher::new();
    let mut object_hasher = blake3::Hasher::new();
    object_hasher.update(DIGEST_DOMAIN);
    object_hasher.update(&raw_metadata);
    let mut remaining = metadata.size();
    let mut buffer = [0; 64 * 1024];
    while remaining != 0 {
        let wanted = remaining.min(buffer.len() as u64) as usize;
        let count = file.read(&mut buffer[..wanted])?;
        if count == 0 {
            return Err(StoreError::ContentMismatch);
        }
        content_hasher.update(&buffer[..count]);
        object_hasher.update(&buffer[..count]);
        remaining -= count as u64;
    }
    if Digest32::new(*content_hasher.finalize().as_bytes()) != metadata.content_digest() {
        return Err(StoreError::ContentMismatch);
    }
    Ok((
        metadata,
        Digest32::new(*object_hasher.finalize().as_bytes()),
    ))
}

fn seek_payload(file: &mut File) -> Result<(), StoreError> {
    file.seek(SeekFrom::Start(8))?;
    let mut raw_len = [0; 4];
    file.read_exact(&mut raw_len)?;
    file.seek(SeekFrom::Start(12 + u32::from_le_bytes(raw_len) as u64))?;
    Ok(())
}

fn encode_metadata(metadata: &ObjectMetadata) -> Result<Vec<u8>, StoreError> {
    let mut output = Vec::new();
    match metadata {
        ObjectMetadata::Artifact(value) => {
            output.push(1);
            output.extend_from_slice(value.content_digest.as_bytes());
            output.extend_from_slice(&value.size.to_le_bytes());
            write_string(&mut output, &value.artifact_id)?;
            match value.proof_identity_digest {
                Some(digest) => {
                    output.push(1);
                    output.extend_from_slice(digest.as_bytes());
                }
                None => output.push(0),
            }
        }
        ObjectMetadata::Key(value) => {
            output.push(2);
            output.extend_from_slice(value.content_digest.as_bytes());
            output.extend_from_slice(&value.size.to_le_bytes());
            output.extend_from_slice(value.srs_source_digest.as_bytes());
            write_string(&mut output, value.proof_flavor.as_str())?;
            output.extend_from_slice(value.circuit_digest.as_bytes());
            output.extend_from_slice(&value.k.to_le_bytes());
            output.extend_from_slice(&value.aggregation_arity.to_le_bytes());
            match value.key_identity_digest {
                Some(digest) => {
                    output.push(1);
                    output.extend_from_slice(digest.as_bytes());
                }
                None => output.push(0),
            }
        }
    }
    if output.len() > MAX_METADATA_BYTES {
        return Err(StoreError::InvalidMetadata);
    }
    Ok(output)
}

fn decode_metadata(bytes: &[u8]) -> Result<ObjectMetadata, StoreError> {
    let mut cursor = Cursor::new(bytes);
    let tag = cursor.byte()?;
    let content_digest = cursor.digest()?;
    let size = cursor.u64()?;
    let result = match tag {
        1 => {
            let artifact_id = cursor.string()?;
            let identity_digest = match cursor.byte()? {
                0 => None,
                1 => Some(cursor.digest()?),
                _ => return Err(StoreError::MetadataMismatch),
            };
            ObjectMetadata::Artifact(match identity_digest {
                Some(identity) => {
                    ArtifactMetadata::new_proof(artifact_id, content_digest, size, identity)?
                }
                None => ArtifactMetadata::new(artifact_id, content_digest, size)?,
            })
        }
        2 => {
            let mut metadata = KeyMetadata::new(
                content_digest,
                size,
                cursor.digest()?,
                ProofFlavorId::parse(cursor.string()?).map_err(|_| StoreError::MetadataMismatch)?,
                cursor.digest()?,
                cursor.u32()?,
                cursor.u32()?,
            )?;
            metadata.key_identity_digest = match cursor.byte()? {
                0 => None,
                1 => Some(cursor.digest()?),
                _ => return Err(StoreError::MetadataMismatch),
            };
            ObjectMetadata::Key(metadata)
        }
        _ => return Err(StoreError::MetadataMismatch),
    };
    if cursor.remaining() != 0 {
        return Err(StoreError::MetadataMismatch);
    }
    Ok(result)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8], StoreError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(StoreError::MetadataMismatch)?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or(StoreError::MetadataMismatch)?;
        self.offset = end;
        Ok(result)
    }
    fn byte(&mut self) -> Result<u8, StoreError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, StoreError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| StoreError::MetadataMismatch)?,
        ))
    }
    fn u64(&mut self) -> Result<u64, StoreError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| StoreError::MetadataMismatch)?,
        ))
    }
    fn digest(&mut self) -> Result<Digest32, StoreError> {
        Ok(Digest32::new(
            self.take(32)?
                .try_into()
                .map_err(|_| StoreError::MetadataMismatch)?,
        ))
    }
    fn string(&mut self) -> Result<String, StoreError> {
        let len = self.u32()? as usize;
        let value = std::str::from_utf8(self.take(len)?)
            .map_err(|_| StoreError::MetadataMismatch)?
            .to_owned();
        validate_identifier(&value).map_err(|_| StoreError::MetadataMismatch)?;
        Ok(value)
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<(), StoreError> {
    validate_identifier(value)?;
    let len = u32::try_from(value.len()).map_err(|_| StoreError::InvalidMetadata)?;
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

pub(crate) fn proof_attestation_digest(
    identity_digest: Digest32,
    object_digest: Digest32,
    run_id: &str,
    job: &JobId,
    attempt: &AttemptId,
) -> Digest32 {
    attestation_digest(
        PROOF_ATTESTATION_DOMAIN,
        identity_digest,
        object_digest,
        run_id,
        job,
        attempt,
    )
}

pub(crate) fn key_attestation_digest(
    identity_digest: Digest32,
    object_digest: Digest32,
    run_id: &str,
    job: &JobId,
    attempt: &AttemptId,
) -> Digest32 {
    attestation_digest(
        KEY_ATTESTATION_DOMAIN,
        identity_digest,
        object_digest,
        run_id,
        job,
        attempt,
    )
}

fn attestation_digest(
    domain: &[u8],
    identity_digest: Digest32,
    object_digest: Digest32,
    run_id: &str,
    job: &JobId,
    attempt: &AttemptId,
) -> Digest32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(identity_digest.as_bytes());
    hasher.update(object_digest.as_bytes());
    hash_string(&mut hasher, run_id);
    hash_string(&mut hasher, job.as_str());
    hash_string(&mut hasher, attempt.as_str());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn validate_identifier(value: &str) -> Result<(), StoreError> {
    if value.is_empty()
        || value.len() > 256
        || !value.is_ascii()
        || value.chars().any(char::is_control)
    {
        Err(StoreError::InvalidMetadata)
    } else {
        Ok(())
    }
}

fn random_name(prefix: &str, suffix: &str) -> Result<String, StoreError> {
    let mut random = [0_u8; 32];
    getrandom::getrandom(&mut random)
        .map_err(|error| io::Error::other(format!("secure randomness unavailable: {error}")))?;
    let hex: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(format!("{prefix}{hex}{suffix}"))
}

fn is_stage_name(name: &str) -> bool {
    name.strip_prefix(STAGE_PREFIX)
        .and_then(|value| value.strip_suffix(STAGE_SUFFIX))
        .is_some_and(|value| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

#[cfg(test)]
mod store_tests {
    use super::*;

    fn test_digest(label: &str) -> Digest32 {
        Digest32::new(*blake3::hash(label.as_bytes()).as_bytes())
    }

    #[test]
    fn proof_identity_digest_commits_to_every_semantic_field() {
        let job = JobId::new("run:aggregate").unwrap();
        let attempt = AttemptId::new("run:aggregate:attempt:1").unwrap();
        let base = TrustedProofIdentity::new(
            ArtifactRole::NativeVerifiedManifest,
            JobKind::NativeAggregate,
            19,
            AggregationArity::Actual(4),
            "run",
            &job,
            &attempt,
            test_digest("proof"),
            test_digest("manifest"),
            test_digest("run-identity"),
            test_digest("statement"),
            test_digest("circuit"),
            test_digest("vk"),
            test_digest("srs"),
            test_digest("shard"),
            test_digest("witness"),
            ProofFlavorId::parse("flavor").unwrap(),
            ExecutionBackendId::parse("backend").unwrap(),
        )
        .unwrap();
        let expected = base.digest();
        let variants = [
            TrustedProofIdentity {
                artifact_role: ArtifactRole::LeafProof,
                ..base.clone()
            },
            TrustedProofIdentity {
                job_kind: JobKind::LeafProof,
                ..base.clone()
            },
            TrustedProofIdentity {
                k: 20,
                ..base.clone()
            },
            TrustedProofIdentity {
                aggregation_arity: AggregationArity::Actual(8),
                ..base.clone()
            },
            TrustedProofIdentity {
                run_id: "other-run".to_owned(),
                ..base.clone()
            },
            TrustedProofIdentity {
                job: JobId::new("run:other-job").unwrap(),
                ..base.clone()
            },
            TrustedProofIdentity {
                attempt: AttemptId::new("run:aggregate:attempt:2").unwrap(),
                ..base.clone()
            },
            TrustedProofIdentity {
                proof_content_digest: test_digest("other-proof"),
                ..base.clone()
            },
            TrustedProofIdentity {
                artifact_manifest_digest: test_digest("other-manifest"),
                ..base.clone()
            },
            TrustedProofIdentity {
                run_identity_digest: test_digest("other-run-identity"),
                ..base.clone()
            },
            TrustedProofIdentity {
                public_statement_digest: test_digest("other-statement"),
                ..base.clone()
            },
            TrustedProofIdentity {
                circuit_digest: test_digest("other-circuit"),
                ..base.clone()
            },
            TrustedProofIdentity {
                verifying_key_digest: test_digest("other-vk"),
                ..base.clone()
            },
            TrustedProofIdentity {
                srs_source_digest: test_digest("other-srs"),
                ..base.clone()
            },
            TrustedProofIdentity {
                shard_identity_digest: test_digest("other-shard"),
                ..base.clone()
            },
            TrustedProofIdentity {
                witness_artifact_digest: test_digest("other-witness"),
                ..base.clone()
            },
            TrustedProofIdentity {
                proof_flavor: ProofFlavorId::parse("other-flavor").unwrap(),
                ..base.clone()
            },
            TrustedProofIdentity {
                execution_backend: ExecutionBackendId::parse("other-backend").unwrap(),
                ..base.clone()
            },
        ];
        for variant in variants {
            assert_ne!(variant.digest(), expected);
        }
    }

    #[test]
    fn swap_of_named_stage_cannot_change_pinned_bytes_copied_for_publication() {
        let root =
            std::env::temp_dir().join(format!("zkie-runtime-publish-hook-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let store = ArtifactStore::open(&root).unwrap();
        let bytes = b"proof";
        let metadata = ObjectMetadata::Artifact(
            ArtifactMetadata::new(
                "proof",
                Digest32::new(*blake3::hash(bytes).as_bytes()),
                bytes.len() as u64,
            )
            .unwrap(),
        );
        let mut staged = store.stage(&metadata).unwrap();
        staged.write_all(bytes).unwrap();
        let validated = store.validate(staged).unwrap();
        let hook_root = root.clone();
        *BEFORE_LINK_HOOK.lock().unwrap() = Some(Box::new(move || {
            let named = fs::read_dir(hook_root.join("staging"))
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            fs::rename(&named, hook_root.join("displaced")).unwrap();
            std::os::unix::fs::symlink(hook_root.join("displaced"), named).unwrap();
        }));

        let published = store.publish(validated).unwrap();
        let mut file = store.open_verified(published.digest()).unwrap();
        let mut actual = Vec::new();
        file.read_to_end(&mut actual).unwrap();
        assert_eq!(actual, bytes);
        let _ = fs::remove_dir_all(root);
    }
}
