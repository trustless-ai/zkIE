//! Real BN256 KZG/SHPLONK CPU leaf proving adapter.
//!
//! The current backend contract passes the key store only to `prepare`, so
//! deserialized keys are held in a process-local, identity-keyed cache after
//! authoritative store validation. `prove` and `verify` never regenerate or
//! silently fall back when that exact cache entry is absent.
//!
//! Schema v2 exposes circuit-constrained commitments for boundary inputs and
//! selected shard outputs. Used weights remain raw public instances, so this
//! POC does not provide model confidentiality. A private-model backend needs
//! an in-circuit weight commitment or fixed-weight design.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use halo2_proofs::halo2curves::bn256::{Bn256, G1Affine};
use halo2_proofs::plonk::{create_proof, keygen_pk, keygen_vk, pk_read, ProvingKey};
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, Transcript, TranscriptReadBuffer,
    TranscriptWriterBuffer,
};
use halo2_proofs::SerdeFormat;
use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use zkie_compiler::graph_compiler::Register;
use zkie_compiler::shard_binding::{
    BoundShardProgram, BoundaryDescriptor as CompilerBoundaryDescriptor,
};
use zkie_core::assembler::AssemblerProgram;
use zkie_core::chips::layer_norm::RsqrtDomain;
use zkie_core::chips::poseidon_boundary::{commit_boundary_native, BoundaryDescriptor};
use zkie_core::field_convert::{i64_to_fr, Fr};
use zkie_core::fixed_point::I18;
use zkie_core::program_circuit::AssemblerCircuit;
use zkie_types::{Digest32, ExecutionBackendId, ModelVisibility, ProofFlavorId, ResourceRequest};

use crate::{
    BackendCapabilities, BackendError, BoundaryClaim, CapabilityId, CryptoVerificationReceipt,
    CryptoVerificationRequest, KeyIdentity, KeyMaterialStore, LeafStatement, PrepareJob,
    PreparedKeys, ProofBackend, ProofJob, UnverifiedProof, VerificationError,
    ZkieIsaCpuWitnessBackend, LEAF_STATEMENT_SCHEMA_VERSION,
};

const FLAVOR: &str = "halo2-kzg-bn256-shplonk-v1";
const BACKEND: &str = "halo2-cpu";
const KEY_MAGIC: &[u8] = b"zkie-halo2-key-v2\0";
const MAX_SRS_BYTES: usize = 512 << 20;
const MAX_KEY_BYTES: usize = 512 << 20;
const MAX_PROOF_BYTES: usize = 16 << 20;
const PUBLIC_STATEMENT_MAGIC: &[u8] = b"zkie.public-instances.v1\0";
const PUBLIC_STATEMENT_SCHEMA: u32 = 1;
const MAX_PUBLIC_VALUES: usize = ((1 << 20) - 80) / 8;
const MAX_CIRCUIT_K: u32 = 20;

fn core_boundary_descriptor(
    descriptor: &CompilerBoundaryDescriptor,
    role: zkie_core::chips::poseidon_boundary::BoundaryRole,
) -> Result<BoundaryDescriptor, BackendError> {
    let register_id = match descriptor.register() {
        Register::GraphInput(name) => format!("graph-input:{name}"),
        Register::Virtual(index) => format!("virtual:{index}"),
        Register::Weight(_) => {
            return Err(BackendError::InvalidJob {
                message: "weights cannot be shard boundary descriptors".into(),
            })
        }
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
        u64::try_from(zkie_core::fixed_point::SCALE_18).expect("I18 scale fits u64"),
    )
    .map_err(|error| BackendError::InvalidJob {
        message: error.to_string(),
    })
}

#[derive(Clone, Debug)]
pub enum SrsPolicy {
    /// The expected digest is a trust anchor and must come from immutable,
    /// operator-controlled deployment configuration, never the proof request.
    ProductionExisting {
        path: PathBuf,
        expected_source_digest: Digest32,
    },
    DevelopmentGenerate,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    identity: KeyIdentity,
    k: u32,
    srs_digest: Digest32,
    production: bool,
}

struct CacheEntry {
    params: ParamsKZG<Bn256>,
    pk: ProvingKey<G1Affine>,
    metadata: KeyMetadata,
}

struct SrsState {
    params: Option<ParamsKZG<Bn256>>,
    digest: Option<Digest32>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyMetadata {
    schema_version: u32,
    proof_flavor: ProofFlavorId,
    circuit_digest: Digest32,
    k: u32,
    srs_digest: Digest32,
    production: bool,
    assembler_shape_digest: [u8; 32],
    public_statement_schema_version: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProofManifest {
    schema_version: u32,
    proof_flavor: ProofFlavorId,
    execution_backend: ExecutionBackendId,
    circuit_digest: Digest32,
    key_digest: Digest32,
    run_identity_digest: Digest32,
    shard_id: u64,
    shard_name: String,
    witness_digest: Digest32,
    k: u32,
    srs_digest: Digest32,
    production: bool,
    assembler_shape_digest: [u8; 32],
    public_statement_schema_version: u32,
}

pub struct Halo2KzgCpuBackend {
    witness_backend: Arc<ZkieIsaCpuWitnessBackend>,
    template: AssemblerCircuit,
    domains: HashMap<(usize, u64), RsqrtDomain>,
    k: u32,
    production: bool,
    srs: Mutex<SrsState>,
    prepare_lock: Mutex<()>,
    cache: Mutex<HashMap<CacheKey, Arc<CacheEntry>>>,
    keygen_invocations: AtomicUsize,
    boundary_inputs: Option<Vec<BoundaryDescriptor>>,
    boundary_outputs: Option<Vec<(usize, BoundaryDescriptor)>>,
    statement_schema: u32,
    trusted_public_weights: Vec<I18>,
}

impl Halo2KzgCpuBackend {
    pub fn validate_model_visibility(run: &zkie_types::RunIdentity) -> Result<(), BackendError> {
        if run.model_visibility == ModelVisibility::PrivateModel {
            Err(BackendError::UnsupportedModelVisibility)
        } else {
            Ok(())
        }
    }

    pub fn new(
        witness_backend: Arc<ZkieIsaCpuWitnessBackend>,
        template_program: AssemblerProgram,
        domains: HashMap<(usize, u64), RsqrtDomain>,
        k: u32,
        policy: SrsPolicy,
    ) -> Result<Self, BackendError> {
        Self::build(
            witness_backend,
            template_program,
            domains,
            k,
            policy,
            PUBLIC_STATEMENT_SCHEMA,
            None,
            None,
        )
    }

    pub fn new_with_boundary_commitments(
        witness_backend: Arc<ZkieIsaCpuWitnessBackend>,
        bound_shard: &BoundShardProgram,
        domains: HashMap<(usize, u64), RsqrtDomain>,
        k: u32,
        policy: SrsPolicy,
    ) -> Result<Self, BackendError> {
        if bound_shard.partition_plan_digest()
            != witness_backend.run_identity().partition_plan_digest
            || bound_shard.shard_id() != witness_backend.shard_identity().id()
            || bound_shard.shard_name() != witness_backend.shard_identity().name()
            || bound_shard.shard_range() != witness_backend.configured_shard().range
        {
            return Err(BackendError::InvalidJob {
                message: "bound shard partition/shard identity mismatch".into(),
            });
        }
        let boundary_inputs = bound_shard
            .input_boundaries()
            .iter()
            .map(|descriptor| {
                core_boundary_descriptor(
                    descriptor,
                    zkie_core::chips::poseidon_boundary::BoundaryRole::Input,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let boundary_outputs = bound_shard
            .output_boundaries()
            .iter()
            .map(|descriptor| {
                let global = match descriptor.register() {
                    Register::Virtual(index) => *index,
                    _ => {
                        return Err(BackendError::InvalidJob {
                            message: "shard output boundary is not virtual".into(),
                        })
                    }
                };
                let local = *bound_shard.global_to_local().get(&global).ok_or_else(|| {
                    BackendError::InvalidJob {
                        message: "shard output is not mapped to a local assembler cell".into(),
                    }
                })?;
                Ok((
                    local,
                    core_boundary_descriptor(
                        descriptor,
                        zkie_core::chips::poseidon_boundary::BoundaryRole::Output,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        Self::build(
            witness_backend,
            bound_shard.assembler_program().clone(),
            domains,
            k,
            policy,
            LEAF_STATEMENT_SCHEMA_VERSION,
            Some(boundary_inputs),
            Some(boundary_outputs),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        witness_backend: Arc<ZkieIsaCpuWitnessBackend>,
        template_program: AssemblerProgram,
        domains: HashMap<(usize, u64), RsqrtDomain>,
        k: u32,
        policy: SrsPolicy,
        statement_schema: u32,
        boundary_inputs: Option<Vec<BoundaryDescriptor>>,
        boundary_outputs: Option<Vec<(usize, BoundaryDescriptor)>>,
    ) -> Result<Self, BackendError> {
        if k == 0
            || k > MAX_CIRCUIT_K
            || witness_backend.run_identity().proof_flavor.as_str() != FLAVOR
            || witness_backend.run_identity().public_input_schema_version != statement_schema
        {
            return Err(BackendError::InvalidJob {
                message: "unsupported flavor or k".into(),
            });
        }
        Self::validate_model_visibility(witness_backend.run_identity())?;
        let boundary_template_program = template_program.clone();
        let trusted_public_weights = template_program
            .weight_values
            .iter()
            .flatten()
            .copied()
            .collect();
        let raw_template =
            witness_backend.validate_assembler_template(template_program, &domains)?;
        let template = match (&boundary_inputs, &boundary_outputs) {
            (Some(inputs), Some(outputs)) => AssemblerCircuit::new_with_boundary_commitments(
                boundary_template_program,
                domains.clone(),
                inputs.clone(),
                outputs.clone(),
                vec![Fr::from(0); 17],
            )
            .map_err(|message| BackendError::InvalidJob { message })?,
            (None, None) => raw_template,
            _ => {
                return Err(BackendError::InvalidJob {
                    message: "incomplete boundary layout".into(),
                })
            }
        };
        let (params, srs_digest, production) = match policy {
            SrsPolicy::DevelopmentGenerate => (None, None, false),
            SrsPolicy::ProductionExisting {
                path,
                expected_source_digest,
            } => {
                let bytes = match read_bounded(&path, MAX_SRS_BYTES) {
                    Err(BackendError::Io { kind, .. }) if kind == "NotFound" => {
                        return Err(BackendError::MissingSrs { path });
                    }
                    result => result?,
                };
                if digest(&bytes) != expected_source_digest {
                    return Err(BackendError::SrsDigestMismatch);
                }
                if bytes.len() < 4
                    || u32::from_le_bytes(bytes[..4].try_into().expect("four bytes")) != k
                {
                    return Err(BackendError::KeyMetadataMismatch {
                        message: "SRS encoded k mismatch".into(),
                    });
                }
                let mut cursor = Cursor::new(bytes);
                let params = ParamsKZG::<Bn256>::read(&mut cursor)
                    .map_err(|e| serialization("kzg-srs", e))?;
                if params.k() != k || cursor.position() != cursor.get_ref().len() as u64 {
                    return Err(BackendError::KeyMetadataMismatch {
                        message: "SRS k/trailing bytes mismatch".into(),
                    });
                }
                (Some(params), Some(expected_source_digest), true)
            }
        };
        Ok(Self {
            witness_backend,
            template,
            domains,
            k,
            production,
            srs: Mutex::new(SrsState {
                params,
                digest: srs_digest,
            }),
            prepare_lock: Mutex::new(()),
            cache: Mutex::new(HashMap::new()),
            keygen_invocations: AtomicUsize::new(0),
            boundary_inputs,
            boundary_outputs,
            statement_schema,
            trusted_public_weights,
        })
    }

    pub fn keygen_invocations(&self) -> usize {
        self.keygen_invocations.load(Ordering::Relaxed)
    }

    pub fn witness_backend(&self) -> &ZkieIsaCpuWitnessBackend {
        &self.witness_backend
    }

    pub fn statement_for(&self, witness: &crate::WitnessArtifact) -> Result<Vec<u8>, BackendError> {
        let (program, values) = self
            .witness_backend
            .load_assembler_program_and_public_values(witness)?;
        if self.statement_schema == PUBLIC_STATEMENT_SCHEMA {
            return encode_public_statement(&values, self.template.params().shape_digest());
        }
        let cache = self.cache.lock().map_err(|_| BackendError::Resource {
            message: "key cache poisoned".into(),
        })?;
        let mut identities = cache.keys().map(|key| key.identity.clone());
        let identity = identities.next().ok_or_else(|| BackendError::InvalidJob {
            message: "statement construction requires one prepared key".into(),
        })?;
        if identities.next().is_some() {
            return Err(BackendError::InvalidJob {
                message: "statement construction is ambiguous across prepared keys".into(),
            });
        }
        drop(cache);
        self.leaf_statement(&identity, &program, &values)?
            .encode()
            .map_err(|error| BackendError::Serialization {
                format: "zkie-leaf-statement-v1".into(),
                message: error.to_string(),
            })
    }

    pub fn statement_for_key(
        &self,
        witness: &crate::WitnessArtifact,
        identity: &KeyIdentity,
    ) -> Result<Vec<u8>, BackendError> {
        if self.statement_schema != LEAF_STATEMENT_SCHEMA_VERSION {
            return self.statement_for(witness);
        }
        self.cached_entry(identity)?;
        let (program, values) = self
            .witness_backend
            .load_assembler_program_and_public_values(witness)?;
        self.leaf_statement(identity, &program, &values)?
            .encode()
            .map_err(|error| BackendError::Serialization {
                format: "zkie-leaf-statement-v1".into(),
                message: error.to_string(),
            })
    }

    fn leaf_statement(
        &self,
        identity: &KeyIdentity,
        program: &AssemblerProgram,
        public_values: &[I18],
    ) -> Result<LeafStatement, BackendError> {
        let inputs = self
            .boundary_inputs
            .as_ref()
            .ok_or_else(|| BackendError::InvalidJob {
                message: "missing trusted boundary input layout".into(),
            })?;
        let outputs = self
            .boundary_outputs
            .as_ref()
            .ok_or_else(|| BackendError::InvalidJob {
                message: "missing trusted boundary output layout".into(),
            })?;
        if inputs.len() != program.input_values.len() {
            return Err(BackendError::InvalidJob {
                message: "boundary input layout mismatch".into(),
            });
        }
        let input_count = program.input_values.iter().map(Vec::len).sum::<usize>();
        let weight_count = program.weight_values.iter().map(Vec::len).sum::<usize>();
        let output_count = outputs
            .iter()
            .try_fold(0usize, |sum, (_, descriptor)| {
                sum.checked_add(descriptor.element_count())
            })
            .ok_or_else(|| BackendError::Resource {
                message: "boundary output count overflow".into(),
            })?;
        let expected_count = input_count
            .checked_add(weight_count)
            .and_then(|count| count.checked_add(output_count))
            .ok_or_else(|| BackendError::Resource {
                message: "public statement value count overflow".into(),
            })?;
        if public_values.len() != expected_count {
            return Err(BackendError::InvalidJob {
                message: "public boundary value layout mismatch".into(),
            });
        }
        let input_claims = inputs
            .iter()
            .zip(&program.input_values)
            .map(|(descriptor, values)| {
                BoundaryClaim::new(
                    descriptor.clone(),
                    commit_boundary_native(descriptor, values).map_err(|error| {
                        BackendError::InvalidJob {
                            message: error.to_string(),
                        }
                    })?,
                )
                .map_err(|error| BackendError::InvalidJob {
                    message: error.to_string(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut output_offset = input_count + weight_count;
        let output_claims = outputs
            .iter()
            .map(|(_, descriptor)| {
                let end = output_offset
                    .checked_add(descriptor.element_count())
                    .ok_or_else(|| BackendError::Resource {
                        message: "boundary output offset overflow".into(),
                    })?;
                let values = public_values.get(output_offset..end).ok_or_else(|| {
                    BackendError::InvalidJob {
                        message: "missing boundary output values".into(),
                    }
                })?;
                output_offset = end;
                BoundaryClaim::new(
                    descriptor.clone(),
                    commit_boundary_native(descriptor, values).map_err(|error| {
                        BackendError::InvalidJob {
                            message: error.to_string(),
                        }
                    })?,
                )
                .map_err(|error| BackendError::InvalidJob {
                    message: error.to_string(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let public_weights = program.weight_values.iter().flatten().copied().collect();
        let run = self.witness_backend.run_identity();
        LeafStatement::new(
            self.witness_backend.shard_identity().id(),
            self.witness_backend.shard_identity().name().into(),
            self.witness_backend.circuit_digest(),
            run.partition_plan_digest,
            run.model_graph_digest,
            run.weights_digest,
            flavor(),
            identity.key_digest(),
            input_claims,
            output_claims,
            public_weights,
        )
        .map_err(|error| BackendError::InvalidJob {
            message: error.to_string(),
        })
    }

    /// Convenience verifier that always rereads and digests the named artifact.
    pub fn verify(&self, proof: &UnverifiedProof) -> Result<(), VerificationError> {
        let bytes = read_bounded(proof.proof_path(), MAX_PROOF_BYTES)
            .map_err(|_| VerificationError::MissingProofArtifact)?;
        if digest(&bytes) != proof.proof_digest() {
            return Err(VerificationError::ProofDigestMismatch);
        }
        let manifest: ProofManifest = serde_json::from_slice(proof.artifact_manifest())
            .map_err(|_| VerificationError::MalformedProofArtifact)?;
        let request = CryptoVerificationRequest::new(
            Arc::new(proof.clone()),
            bytes,
            KeyIdentity::new(
                proof.proof_flavor().clone(),
                proof.circuit_digest(),
                proof.verification_key_digest(),
            )
            .map_err(|_| VerificationError::MalformedProofArtifact)?,
            manifest.witness_digest,
        )
        .map_err(|_| VerificationError::MalformedProofArtifact)?;
        self.verify_cryptographically(request).map(|_| ())
    }

    fn cache_key(&self, identity: KeyIdentity, srs_digest: Digest32) -> CacheKey {
        CacheKey {
            identity,
            k: self.k,
            srs_digest,
            production: self.production,
        }
    }

    fn expected_metadata(&self, srs_digest: Digest32) -> KeyMetadata {
        KeyMetadata {
            schema_version: 1,
            proof_flavor: flavor(),
            circuit_digest: self.witness_backend.circuit_digest(),
            k: self.k,
            srs_digest,
            production: self.production,
            assembler_shape_digest: self.template.params().shape_digest(),
            public_statement_schema_version: self.statement_schema,
        }
    }

    fn install_authoritative_package(
        &self,
        identity: KeyIdentity,
        authoritative: Vec<u8>,
    ) -> Result<PreparedKeys, BackendError> {
        if authoritative.len() > MAX_KEY_BYTES || digest(&authoritative) != identity.key_digest() {
            return Err(BackendError::KeyDigestMismatch);
        }
        let metadata = read_key_metadata(&authoritative)?;
        let expected = self.expected_metadata(metadata.srs_digest);
        if metadata.schema_version != expected.schema_version
            || metadata.proof_flavor != expected.proof_flavor
            || metadata.circuit_digest != expected.circuit_digest
            || metadata.k != expected.k
            || metadata.production != expected.production
            || metadata.assembler_shape_digest != expected.assembler_shape_digest
            || metadata.public_statement_schema_version != expected.public_statement_schema_version
        {
            return Err(BackendError::KeyMetadataMismatch {
                message: "flavor/circuit/k/SRS provenance mismatch".into(),
            });
        }
        {
            let state = self.srs.lock().map_err(|_| BackendError::Resource {
                message: "SRS state poisoned".into(),
            })?;
            if state
                .digest
                .is_some_and(|digest| digest != metadata.srs_digest)
            {
                return Err(BackendError::SrsDigestMismatch);
            }
        }
        let entry = decode_key_package(&authoritative, &expected, &self.template)?;
        {
            let mut state = self.srs.lock().map_err(|_| BackendError::Resource {
                message: "SRS state poisoned".into(),
            })?;
            if state.digest.is_none() {
                state.digest = Some(metadata.srs_digest);
                state.params = Some(entry.params.clone());
            }
        }
        let cache_key = self.cache_key(identity.clone(), metadata.srs_digest);
        self.cache
            .lock()
            .map_err(|_| BackendError::Resource {
                message: "key cache poisoned".into(),
            })?
            .insert(cache_key, Arc::new(entry));
        PreparedKeys::new(identity.clone(), identity)
    }

    fn cached_entry(&self, identity: &KeyIdentity) -> Result<Arc<CacheEntry>, BackendError> {
        self.cache
            .lock()
            .map_err(|_| BackendError::Resource {
                message: "key cache poisoned".into(),
            })?
            .iter()
            .find(|(key, _)| {
                key.identity == *identity && key.k == self.k && key.production == self.production
            })
            .map(|(_, entry)| Arc::clone(entry))
            .ok_or_else(|| BackendError::MissingKey {
                identity: identity.clone(),
            })
    }

    fn validate_common(
        &self,
        run: &zkie_types::RunIdentity,
        shard: &crate::ShardIdentity,
    ) -> Result<(), BackendError> {
        if run != self.witness_backend.run_identity()
            || shard != self.witness_backend.shard_identity()
            || shard.circuit_digest() != self.witness_backend.circuit_digest()
            || run.proof_flavor.as_str() != FLAVOR
            || run.public_input_schema_version != self.statement_schema
        {
            return Err(BackendError::InvalidJob {
                message: "Halo2 backend run/shard/circuit/flavor mismatch".into(),
            });
        }
        Ok(())
    }
}

impl ProofBackend for Halo2KzgCpuBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(
            backend_id(),
            vec![flavor()],
            vec![CapabilityId::parse("bn256-kzg-shplonk-leaf-v1").expect("static capability")],
        )
        .expect("static capabilities")
    }

    fn estimate_resources(&self, job: ProofJob) -> Result<ResourceRequest, BackendError> {
        self.validate_common(job.run_identity(), job.shard())?;
        ResourceRequest::new(1, 1 << 30, 0, 0).map_err(|e| BackendError::Resource {
            message: e.to_string(),
        })
    }

    fn prepare(
        &self,
        job: PrepareJob,
        store: &dyn KeyMaterialStore,
    ) -> Result<PreparedKeys, BackendError> {
        self.validate_common(job.run_identity(), job.shard())?;
        if job.circuit_k() != self.k {
            return Err(BackendError::KeyMetadataMismatch {
                message: "prepare k mismatch".into(),
            });
        }
        if job.requires_production() && !self.production {
            return Err(BackendError::NonProductionSrs);
        }
        if self.production && job.key_identity().is_none() {
            return Err(BackendError::InvalidJob {
                message: "production prepare requires an explicit key identity".into(),
            });
        }
        let _prepare = self
            .prepare_lock
            .lock()
            .map_err(|_| BackendError::Resource {
                message: "prepare lock poisoned".into(),
            })?;

        if let Some(identity) = job.key_identity() {
            if self
                .cache
                .lock()
                .map_err(|_| BackendError::Resource {
                    message: "key cache poisoned".into(),
                })?
                .keys()
                .any(|key| {
                    key.identity == *identity
                        && key.k == self.k
                        && key.production == self.production
                })
            {
                return PreparedKeys::new(identity.clone(), identity.clone());
            }
            let authoritative = store
                .read(identity)?
                .ok_or_else(|| BackendError::MissingKey {
                    identity: identity.clone(),
                })?;
            return self.install_authoritative_package(identity.clone(), authoritative);
        }

        let current_srs_digest = self
            .srs
            .lock()
            .map_err(|_| BackendError::Resource {
                message: "SRS state poisoned".into(),
            })?
            .digest;
        if let Some(srs_digest) = current_srs_digest {
            if let Some(identity) = self
                .cache
                .lock()
                .map_err(|_| BackendError::Resource {
                    message: "key cache poisoned".into(),
                })?
                .keys()
                .find(|key| {
                    key.k == self.k
                        && key.production == self.production
                        && key.srs_digest == srs_digest
                })
                .map(|key| key.identity.clone())
            {
                return PreparedKeys::new(identity.clone(), identity);
            }
        }

        let (params, srs_digest) = {
            let mut state = self.srs.lock().map_err(|_| BackendError::Resource {
                message: "SRS state poisoned".into(),
            })?;
            if state.params.is_none() {
                let params = ParamsKZG::<Bn256>::setup(self.k, OsRng);
                let srs_digest = digest(&serialize_params(&params)?);
                state.params = Some(params);
                state.digest = Some(srs_digest);
            }
            (
                state.params.clone().expect("initialized above"),
                state.digest.expect("initialized above"),
            )
        };
        self.keygen_invocations.fetch_add(1, Ordering::Relaxed);
        let vk = keygen_vk(&params, &self.template).map_err(|e| proving(e.to_string()))?;
        let pk = keygen_pk(&params, vk, &self.template).map_err(|e| proving(e.to_string()))?;
        let generated_package =
            encode_key_package(&self.expected_metadata(srs_digest), &params, &pk)?;
        let identity = KeyIdentity::new(
            flavor(),
            self.witness_backend.circuit_digest(),
            digest(&generated_package),
        )?;
        store.write_if_absent(&identity, &generated_package)?;
        let authoritative = store
            .read(&identity)?
            .ok_or_else(|| BackendError::MissingKey {
                identity: identity.clone(),
            })?;
        self.install_authoritative_package(identity, authoritative)
    }

    fn prove(&self, job: ProofJob, output: PathBuf) -> Result<UnverifiedProof, BackendError> {
        self.validate_common(job.run_identity(), job.shard())?;
        let identity = job.prepared_keys().proving_key();
        if job.witness().shard() != job.shard()
            || job.witness().circuit_digest() != self.witness_backend.circuit_digest()
            || job.prepared_keys().verification_key() != identity
        {
            return Err(BackendError::InvalidJob {
                message: "proof job witness/key mismatch".into(),
            });
        }
        let entry = self.cached_entry(identity)?;
        let (program, public_values) = self
            .witness_backend
            .load_assembler_program_and_public_values(job.witness())?;
        let (circuit, public_statement, instance_values) =
            if self.statement_schema == LEAF_STATEMENT_SCHEMA_VERSION {
                let statement = self.leaf_statement(identity, &program, &public_values)?;
                let circuit = AssemblerCircuit::new_with_boundary_commitments(
                    program,
                    self.domains.clone(),
                    self.boundary_inputs
                        .clone()
                        .expect("schema validated at construction"),
                    self.boundary_outputs
                        .clone()
                        .expect("schema validated at construction"),
                    statement.instance_prefix(),
                )
                .map_err(|message| BackendError::InvalidJob { message })?;
                let encoded = statement
                    .encode()
                    .map_err(|error| BackendError::Serialization {
                        format: "zkie-leaf-statement-v1".into(),
                        message: error.to_string(),
                    })?;
                (circuit, encoded, statement.instances())
            } else {
                let circuit = AssemblerCircuit::new_with_public_outputs(
                    program,
                    self.domains.clone(),
                    self.witness_backend.public_output_indices(),
                )
                .map_err(|message| BackendError::InvalidJob { message })?;
                let encoded =
                    encode_public_statement(&public_values, self.template.params().shape_digest())?;
                let instances = public_values
                    .iter()
                    .map(|value| i64_to_fr(value.raw()))
                    .collect();
                (circuit, encoded, instances)
            };
        if circuit.params().shape_digest() != self.template.params().shape_digest() {
            return Err(BackendError::InvalidJob {
                message: "witness circuit shape differs from prepared circuit".into(),
            });
        }
        let manifest = serde_json::to_vec(&ProofManifest {
            schema_version: 1,
            proof_flavor: flavor(),
            execution_backend: backend_id(),
            circuit_digest: self.witness_backend.circuit_digest(),
            key_digest: identity.key_digest(),
            run_identity_digest: job.run_identity().canonical_digest(),
            shard_id: job.shard().id(),
            shard_name: job.shard().name().into(),
            witness_digest: job.witness().digest(),
            k: self.k,
            srs_digest: entry.metadata.srs_digest,
            production: self.production,
            assembler_shape_digest: self.template.params().shape_digest(),
            public_statement_schema_version: self.statement_schema,
        })
        .map_err(|e| serialization("halo2-proof-manifest-v1", e))?;
        let instances = vec![vec![instance_values]];
        let temp = create_temp_sibling(&output)?;
        let result = (|| {
            let file = temp.1;
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(file);
            bind_manifest(&mut transcript, &manifest)
                .map_err(|e| serialization("halo2-proof-transcript-binding", e))?;
            create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
                &entry.params,
                &entry.pk,
                std::slice::from_ref(&circuit),
                instances.as_slice(),
                OsRng,
                &mut transcript,
            )
            .map_err(|e| proving(e.to_string()))?;
            let file = transcript.finalize();
            file.sync_all()
                .map_err(|e| io_error("sync proof", &temp.0, e))?;
            fs::hard_link(&temp.0, &output)
                .map_err(|e| io_error("publish proof without clobber", &output, e))?;
            sync_parent_dir(&output)?;
            Ok::<(), BackendError>(())
        })();
        match result {
            Ok(()) => {
                fs::remove_file(&temp.0)
                    .map_err(|error| io_error("remove published proof temp", &temp.0, error))?;
                sync_parent_dir(&output)?;
            }
            Err(error) => {
                let _ = fs::remove_file(&temp.0);
                return Err(error);
            }
        }
        let proof_bytes = read_bounded(&output, MAX_PROOF_BYTES)?;
        let proof_digest = digest(&proof_bytes);
        UnverifiedProof::new(
            output,
            proof_digest,
            public_statement,
            self.witness_backend.circuit_digest(),
            identity.key_digest(),
            flavor(),
            backend_id(),
            job.shard().clone(),
            manifest,
            job.run_identity().canonical_digest(),
        )
    }

    fn verify_cryptographically(
        &self,
        request: CryptoVerificationRequest,
    ) -> Result<CryptoVerificationReceipt, VerificationError> {
        if request.proof_flavor().as_str() != FLAVOR
            || request.execution_backend().as_str() != BACKEND
            || request.circuit_digest() != self.witness_backend.circuit_digest()
            || request.shard() != self.witness_backend.shard_identity()
            || request.run_identity_digest()
                != self.witness_backend.run_identity().canonical_digest()
        {
            return Err(VerificationError::MalformedProofArtifact);
        }
        let manifest: ProofManifest = serde_json::from_slice(request.artifact_manifest())
            .map_err(|_| VerificationError::MalformedProofArtifact)?;
        if manifest.schema_version != 1
            || manifest.proof_flavor.as_str() != FLAVOR
            || manifest.execution_backend.as_str() != BACKEND
            || manifest.circuit_digest != self.witness_backend.circuit_digest()
            || manifest.key_digest != request.verification_key().key_digest()
            || manifest.run_identity_digest
                != self.witness_backend.run_identity().canonical_digest()
            || manifest.shard_id != self.witness_backend.shard_identity().id()
            || manifest.shard_name != self.witness_backend.shard_identity().name()
            || manifest.k != self.k
            || manifest.production != self.production
            || manifest.assembler_shape_digest != self.template.params().shape_digest()
            || manifest.public_statement_schema_version != self.statement_schema
            || manifest.witness_digest != request.expected_witness_artifact_digest()
        {
            return Err(VerificationError::MalformedProofArtifact);
        }
        let entry = self.cached_entry(request.verification_key()).map_err(|_| {
            VerificationError::CryptographicVerificationFailed {
                message: "exact prepared key is absent from process cache".into(),
            }
        })?;
        if manifest.srs_digest != entry.metadata.srs_digest {
            return Err(VerificationError::MalformedProofArtifact);
        }
        let verifier_params = entry.params.verifier_params();
        let strategy = SingleStrategy::new(&verifier_params);
        let public_values = if self.statement_schema == LEAF_STATEMENT_SCHEMA_VERSION {
            let statement = LeafStatement::decode(request.public_statement())
                .map_err(|_| VerificationError::MalformedProofArtifact)?;
            // The request/manifest remains the trusted provenance source; a
            // self-described statement cannot replace any expected identity.
            let trusted_inputs = self
                .boundary_inputs
                .as_ref()
                .ok_or(VerificationError::MalformedProofArtifact)?;
            let trusted_outputs = self
                .boundary_outputs
                .as_ref()
                .ok_or(VerificationError::MalformedProofArtifact)?;
            if statement
                .input_claims()
                .iter()
                .map(BoundaryClaim::descriptor)
                .ne(trusted_inputs.iter())
                || statement
                    .output_claims()
                    .iter()
                    .map(BoundaryClaim::descriptor)
                    .ne(trusted_outputs.iter().map(|(_, descriptor)| descriptor))
            {
                return Err(VerificationError::MalformedProofArtifact);
            }
            if statement.public_weights() != self.trusted_public_weights {
                return Err(VerificationError::MalformedProofArtifact);
            }
            // Identity equality is checked without witness reconstruction by
            // rebuilding the statement from trusted request expectations.
            let trusted_prefix = LeafStatement::new(
                self.witness_backend.shard_identity().id(),
                self.witness_backend.shard_identity().name().into(),
                self.witness_backend.circuit_digest(),
                self.witness_backend.run_identity().partition_plan_digest,
                self.witness_backend.run_identity().model_graph_digest,
                self.witness_backend.run_identity().weights_digest,
                flavor(),
                request.verification_key().key_digest(),
                statement.input_claims().to_vec(),
                statement.output_claims().to_vec(),
                statement.public_weights().to_vec(),
            )
            .map_err(|_| VerificationError::MalformedProofArtifact)?;
            if trusted_prefix != statement {
                return Err(VerificationError::MalformedProofArtifact);
            }
            statement.instances()
        } else {
            decode_public_statement(
                request.public_statement(),
                self.template.params().shape_digest(),
            )?
        };
        let instances = vec![vec![public_values]];
        let consumed = Arc::new(AtomicUsize::new(0));
        let reader = CountingReader {
            bytes: request.proof_bytes(),
            position: 0,
            consumed: Arc::clone(&consumed),
        };
        let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(reader);
        bind_manifest(&mut transcript, request.artifact_manifest()).map_err(|e| {
            VerificationError::CryptographicVerificationFailed {
                message: e.to_string(),
            }
        })?;
        halo2_proofs::plonk::verify_proof::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<Bn256>,
            _,
            _,
            _,
        >(
            &verifier_params,
            entry.pk.get_vk(),
            strategy,
            instances.as_slice(),
            &mut transcript,
        )
        .map_err(|e| VerificationError::CryptographicVerificationFailed {
            message: e.to_string(),
        })?;
        drop(transcript);
        if consumed.load(Ordering::Relaxed) != request.proof_bytes().len() {
            return Err(VerificationError::CryptographicVerificationFailed {
                message: "proof contains trailing bytes".into(),
            });
        }
        Ok(CryptoVerificationReceipt::new(
            request.request_binding_digest(),
        ))
    }
}

struct CountingReader<'a> {
    bytes: &'a [u8],
    position: usize,
    consumed: Arc<AtomicUsize>,
}

impl Read for CountingReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let remaining = &self.bytes[self.position..];
        let count = remaining.len().min(output.len());
        output[..count].copy_from_slice(&remaining[..count]);
        self.position += count;
        self.consumed.store(self.position, Ordering::Relaxed);
        Ok(count)
    }
}

fn encode_key_package(
    metadata: &KeyMetadata,
    params: &ParamsKZG<Bn256>,
    pk: &ProvingKey<G1Affine>,
) -> Result<Vec<u8>, BackendError> {
    let meta =
        serde_json::to_vec(metadata).map_err(|e| serialization("halo2-key-metadata-v1", e))?;
    let params = serialize_params(params)?;
    let mut pk_bytes = Vec::new();
    pk.write(&mut pk_bytes, SerdeFormat::Processed)
        .map_err(|e| serialization("halo2-pk", e))?;
    let mut out =
        Vec::with_capacity(KEY_MAGIC.len() + 20 + meta.len() + params.len() + pk_bytes.len());
    out.extend_from_slice(KEY_MAGIC);
    out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    out.extend_from_slice(&meta);
    out.extend_from_slice(&(params.len() as u64).to_le_bytes());
    out.extend_from_slice(&params);
    out.extend_from_slice(&(pk_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&pk_bytes);
    if out.len() > MAX_KEY_BYTES {
        return Err(BackendError::Resource {
            message: "key package too large".into(),
        });
    }
    Ok(out)
}

fn decode_key_package(
    bytes: &[u8],
    expected: &KeyMetadata,
    circuit: &AssemblerCircuit,
) -> Result<CacheEntry, BackendError> {
    let mut cursor = Cursor::new(bytes);
    let mut magic = vec![0; KEY_MAGIC.len()];
    cursor
        .read_exact(&mut magic)
        .map_err(|e| serialization("halo2-key-package", e))?;
    if magic != KEY_MAGIC {
        return Err(BackendError::KeyMetadataMismatch {
            message: "bad key package magic".into(),
        });
    }
    let meta_len = read_u32(&mut cursor)? as usize;
    if meta_len > 64 << 10 {
        return Err(BackendError::KeyMetadataMismatch {
            message: "oversized metadata".into(),
        });
    }
    let mut meta = vec![0; meta_len];
    cursor
        .read_exact(&mut meta)
        .map_err(|e| serialization("halo2-key-metadata", e))?;
    let actual: KeyMetadata =
        serde_json::from_slice(&meta).map_err(|e| serialization("halo2-key-metadata-v1", e))?;
    if actual.schema_version != expected.schema_version
        || actual.proof_flavor != expected.proof_flavor
        || actual.circuit_digest != expected.circuit_digest
        || actual.k != expected.k
        || actual.srs_digest != expected.srs_digest
        || actual.production != expected.production
        || actual.assembler_shape_digest != expected.assembler_shape_digest
        || actual.public_statement_schema_version != expected.public_statement_schema_version
    {
        return Err(BackendError::KeyMetadataMismatch {
            message: "flavor/circuit/k/SRS provenance mismatch".into(),
        });
    }
    let params_len =
        usize::try_from(read_u64(&mut cursor)?).map_err(|_| BackendError::Resource {
            message: "SRS length does not fit this platform".into(),
        })?;
    if params_len > MAX_SRS_BYTES {
        return Err(BackendError::Resource {
            message: "oversized SRS".into(),
        });
    }
    let mut params_bytes = vec![0; params_len];
    cursor
        .read_exact(&mut params_bytes)
        .map_err(|e| serialization("halo2-key-params", e))?;
    if digest(&params_bytes) != expected.srs_digest {
        return Err(BackendError::SrsDigestMismatch);
    }
    let pk_len = usize::try_from(read_u64(&mut cursor)?).map_err(|_| BackendError::Resource {
        message: "proving-key length does not fit this platform".into(),
    })?;
    if pk_len > MAX_KEY_BYTES {
        return Err(BackendError::Resource {
            message: "oversized proving key".into(),
        });
    }
    let mut pk_bytes = vec![0; pk_len];
    cursor
        .read_exact(&mut pk_bytes)
        .map_err(|e| serialization("halo2-key-pk", e))?;
    if cursor.position() != bytes.len() as u64 {
        return Err(BackendError::KeyMetadataMismatch {
            message: "trailing key bytes".into(),
        });
    }
    let mut params_cursor = Cursor::new(params_bytes);
    let params =
        ParamsKZG::<Bn256>::read(&mut params_cursor).map_err(|e| serialization("halo2-srs", e))?;
    if params.k() != expected.k || params_cursor.position() != params_cursor.get_ref().len() as u64
    {
        return Err(BackendError::KeyMetadataMismatch {
            message: "deserialized SRS k mismatch".into(),
        });
    }
    let mut pk_cursor = Cursor::new(pk_bytes);
    let pk = pk_read::<G1Affine, _, AssemblerCircuit>(
        &mut pk_cursor,
        SerdeFormat::Processed,
        expected.k,
        circuit,
        true,
    )
    .map_err(|e| serialization("halo2-pk", e))?;
    if pk_cursor.position() != pk_cursor.get_ref().len() as u64 {
        return Err(BackendError::KeyMetadataMismatch {
            message: "trailing proving key bytes".into(),
        });
    }
    Ok(CacheEntry {
        params,
        pk,
        metadata: expected.clone(),
    })
}

fn read_key_metadata(bytes: &[u8]) -> Result<KeyMetadata, BackendError> {
    let mut cursor = Cursor::new(bytes);
    let mut magic = vec![0; KEY_MAGIC.len()];
    cursor
        .read_exact(&mut magic)
        .map_err(|e| serialization("halo2-key-package", e))?;
    if magic != KEY_MAGIC {
        return Err(BackendError::KeyMetadataMismatch {
            message: "bad key package magic".into(),
        });
    }
    let meta_len = read_u32(&mut cursor)? as usize;
    if meta_len > 64 << 10 || meta_len > bytes.len().saturating_sub(KEY_MAGIC.len() + 4) {
        return Err(BackendError::KeyMetadataMismatch {
            message: "oversized or truncated metadata".into(),
        });
    }
    let mut meta = vec![0; meta_len];
    cursor
        .read_exact(&mut meta)
        .map_err(|e| serialization("halo2-key-metadata", e))?;
    serde_json::from_slice(&meta).map_err(|e| serialization("halo2-key-metadata-v1", e))
}

fn serialize_params(params: &ParamsKZG<Bn256>) -> Result<Vec<u8>, BackendError> {
    let mut bytes = Vec::new();
    params
        .write(&mut bytes)
        .map_err(|e| serialization("kzg-srs", e))?;
    Ok(bytes)
}
fn bind_manifest<T: Transcript<G1Affine, Challenge255<G1Affine>>>(
    transcript: &mut T,
    manifest: &[u8],
) -> std::io::Result<()> {
    for chunk in blake3::hash(manifest).as_bytes().chunks_exact(8) {
        transcript.common_scalar(Fr::from(u64::from_le_bytes(
            chunk.try_into().expect("eight-byte chunk"),
        )))?;
    }
    Ok(())
}

fn encode_public_statement(
    values: &[I18],
    shape_digest: [u8; 32],
) -> Result<Vec<u8>, BackendError> {
    if values.len() > MAX_PUBLIC_VALUES {
        return Err(BackendError::Resource {
            message: "public statement has too many values".into(),
        });
    }
    let mut bytes = Vec::with_capacity(PUBLIC_STATEMENT_MAGIC.len() + 44 + values.len() * 8);
    bytes.extend_from_slice(PUBLIC_STATEMENT_MAGIC);
    bytes.extend_from_slice(&PUBLIC_STATEMENT_SCHEMA.to_le_bytes());
    bytes.extend_from_slice(&shape_digest);
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.raw().to_le_bytes());
    }
    Ok(bytes)
}

fn decode_public_statement(
    bytes: &[u8],
    expected_shape_digest: [u8; 32],
) -> Result<Vec<Fr>, VerificationError> {
    let version_end = PUBLIC_STATEMENT_MAGIC.len() + 4;
    let shape_end = version_end + 32;
    let header = shape_end + 8;
    if bytes.len() < header || &bytes[..PUBLIC_STATEMENT_MAGIC.len()] != PUBLIC_STATEMENT_MAGIC {
        return Err(VerificationError::MalformedProofArtifact);
    }
    if u32::from_le_bytes(
        bytes[PUBLIC_STATEMENT_MAGIC.len()..version_end]
            .try_into()
            .map_err(|_| VerificationError::MalformedProofArtifact)?,
    ) != PUBLIC_STATEMENT_SCHEMA
        || bytes[version_end..shape_end] != expected_shape_digest
    {
        return Err(VerificationError::MalformedProofArtifact);
    }
    let count = usize::try_from(u64::from_le_bytes(
        bytes[shape_end..header]
            .try_into()
            .map_err(|_| VerificationError::MalformedProofArtifact)?,
    ))
    .map_err(|_| VerificationError::MalformedProofArtifact)?;
    let expected_len = header
        .checked_add(
            count
                .checked_mul(8)
                .ok_or(VerificationError::MalformedProofArtifact)?,
        )
        .ok_or(VerificationError::MalformedProofArtifact)?;
    if count > MAX_PUBLIC_VALUES || bytes.len() != expected_len {
        return Err(VerificationError::MalformedProofArtifact);
    }
    bytes[header..]
        .chunks_exact(8)
        .map(|chunk| {
            let raw = i64::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| VerificationError::MalformedProofArtifact)?,
            );
            Ok(i64_to_fr(raw))
        })
        .collect()
}
fn read_u32(r: &mut impl Read) -> Result<u32, BackendError> {
    let mut b = [0; 4];
    r.read_exact(&mut b)
        .map_err(|e| serialization("halo2-key-package", e))?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> Result<u64, BackendError> {
    let mut b = [0; 8];
    r.read_exact(&mut b)
        .map_err(|e| serialization("halo2-key-package", e))?;
    Ok(u64::from_le_bytes(b))
}
fn digest(bytes: &[u8]) -> Digest32 {
    Digest32::new(*blake3::hash(bytes).as_bytes())
}
fn flavor() -> ProofFlavorId {
    ProofFlavorId::parse(FLAVOR).expect("static flavor")
}
fn backend_id() -> ExecutionBackendId {
    ExecutionBackendId::parse(BACKEND).expect("static backend")
}
fn proving(message: String) -> BackendError {
    BackendError::ProvingFailed { message }
}
fn serialization(format: &str, e: impl std::fmt::Display) -> BackendError {
    BackendError::Serialization {
        format: format.into(),
        message: e.to_string(),
    }
}
fn io_error(operation: &str, path: &Path, e: std::io::Error) -> BackendError {
    BackendError::Io {
        operation: operation.into(),
        path: path.into(),
        kind: format!("{:?}", e.kind()),
        message: e.to_string(),
    }
}
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, BackendError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options
        .open(path)
        .map_err(|e| io_error("open regular", path, e))?;
    let meta = file.metadata().map_err(|e| io_error("metadata", path, e))?;
    if !meta.file_type().is_file() {
        return Err(BackendError::Io {
            operation: "open regular".into(),
            path: path.into(),
            kind: "InvalidInput".into(),
            message: "bounded artifact must be a regular file".into(),
        });
    }
    if meta.len() > limit as u64 {
        return Err(BackendError::Resource {
            message: "artifact is oversized".into(),
        });
    }
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    Read::by_ref(&mut file)
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| io_error("read", path, e))?;
    if bytes.len() > limit {
        return Err(BackendError::Resource {
            message: "artifact oversized".into(),
        });
    }
    Ok(bytes)
}
fn create_temp_sibling(output: &Path) -> Result<(PathBuf, File), BackendError> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    if fs::symlink_metadata(output).is_ok() {
        return Err(BackendError::Io {
            operation: "publish proof without clobber".into(),
            path: output.into(),
            kind: "AlreadyExists".into(),
            message: "output exists".into(),
        });
    }
    for _ in 0..32 {
        let path = parent.join(format!(".zkie-proof-{:016x}.tmp", OsRng.next_u64()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(io_error("create proof temp", &path, e)),
        }
    }
    Err(BackendError::Resource {
        message: "could not allocate proof temp file".into(),
    })
}

fn sync_parent_dir(path: &Path) -> Result<(), BackendError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| io_error("sync parent directory", parent, error))
}
