//! Shared, serializable identities and resource descriptions for prover backends.

mod identity;
mod resource;

pub use identity::{
    Digest32, DigestParseError, ExecutionBackendId, IdentifierError, ModelVisibility,
    ProofFlavorId, RunIdentity,
};
pub use resource::{HardwareProfile, ResourceCapacity, ResourceRequest, ResourceValidationError};
