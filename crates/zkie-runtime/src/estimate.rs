use thiserror::Error;
use zkie_types::{Digest32, ExecutionBackendId, ProofFlavorId, ResourceRequest};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReservationKey {
    pub circuit_digest: Digest32,
    pub k: u32,
    pub proof_flavor: ProofFlavorId,
    pub execution_backend: ExecutionBackendId,
    pub hardware_profile: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub key: ReservationKey,
    pub resources: ResourceRequest,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum EstimateError {
    #[error("resource estimate arithmetic overflow")]
    Overflow,
}

pub fn calibrated_reservation(
    static_request: ResourceRequest,
    max_observed_peak_bytes: Option<u64>,
) -> Result<ResourceRequest, EstimateError> {
    let Some(peak) = max_observed_peak_bytes else {
        return Ok(static_request);
    };
    let calibrated_ram = u64::try_from((u128::from(peak) * 115).div_ceil(100))
        .map_err(|_| EstimateError::Overflow)?;
    Ok(ResourceRequest {
        ram_bytes: static_request.ram_bytes.max(calibrated_ram),
        ..static_request
    })
}
