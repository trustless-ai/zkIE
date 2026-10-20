//! A deliberately bounded, non-ZK inference protocol. See `docs/security/`.
//!
//! Verification takes only a trusted model id, its public manifest, the public
//! statement and serialized proof. No witness, forward executor or prover data
//! is part of that API. Prover code is behind the optional `prover` feature.

mod air;
mod backend;
mod model;
#[cfg(feature = "prover")]
mod prover;

pub use model::{Model, ModelId, Statement, VERSION};
#[cfg(feature = "prover")]
pub use prover::{evaluate, prove};

use backend::{config, StarkProof};
use p3_uni_stark::verify as stark_verify;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Fixed framing and limits are protocol parameters, not chosen by a proof.
pub const MAX_PROOF_BYTES: usize = 1 << 20;
pub(crate) const MAGIC: &[u8; 8] = b"ZKIESTK2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidModel,
    InvalidStatement,
    ModelIdentity,
    ProofEncoding,
    ProofTooLarge,
    InvalidProof,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

/// The expected id must come from the verifier's trust policy, independently of
/// the untrusted statement/proof bundle. Parsing a model id is not endorsing it.
pub fn verify(
    model: &Model,
    trusted_model: ModelId,
    statement: &Statement,
    proof: &[u8],
) -> Result<(), Error> {
    model.validate()?;
    statement.validate()?;
    if model.id()? != trusted_model || statement.model_id != trusted_model {
        return Err(Error::ModelIdentity);
    }
    if proof.len() > MAX_PROOF_BYTES {
        return Err(Error::ProofTooLarge);
    }
    if proof.len() <= MAGIC.len() || !proof.starts_with(MAGIC) {
        return Err(Error::ProofEncoding);
    }
    // Defense in depth at the dependency boundary: malformed adversarial proof
    // shapes must fail closed even if the backend gains an unchecked index.
    // This is not a substitute for length/shape checks or an OS memory limit.
    catch_unwind(AssertUnwindSafe(|| {
        let payload = &proof[MAGIC.len()..];
        let (decoded, remaining): (StarkProof, _) =
            postcard::take_from_bytes(payload).map_err(|_| Error::ProofEncoding)?;
        if !remaining.is_empty()
            || decoded.degree_bits != air::TRACE_LOG_ROWS
            || postcard::to_allocvec(&decoded).map_err(|_| Error::ProofEncoding)? != payload
        {
            return Err(Error::ProofEncoding);
        }
        stark_verify(
            &config(),
            &air::InferenceAir,
            &decoded,
            &model.public_values(statement),
        )
        .map_err(|_| Error::InvalidProof)
    }))
    .unwrap_or(Err(Error::InvalidProof))
}

#[cfg(all(test, feature = "prover"))]
mod security_tests;

pub mod matmul;
mod whir_backend;

mod oracle;

pub mod dense;

pub use oracle::Tensor as TensorCommitment;
pub mod lookup;

pub mod graph;

#[cfg(feature = "prover")]
pub mod transformer;
