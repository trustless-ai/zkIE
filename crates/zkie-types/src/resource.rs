use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResourceRequest {
    pub cpu_cores: u32,
    pub ram_bytes: u64,
    pub gpu_count: u32,
    pub gpu_vram_bytes_per_device: u64,
}

impl ResourceRequest {
    pub fn new(
        cpu_cores: u32,
        ram_bytes: u64,
        gpu_count: u32,
        gpu_vram_bytes_per_device: u64,
    ) -> Result<Self, ResourceValidationError> {
        validate_request(cpu_cores, gpu_count, gpu_vram_bytes_per_device)?;
        Ok(Self {
            cpu_cores,
            ram_bytes,
            gpu_count,
            gpu_vram_bytes_per_device,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResourceCapacity {
    pub cpu_cores: u32,
    pub ram_bytes: u64,
    pub gpu_count: u32,
    pub gpu_vram_bytes_per_device: u64,
}

#[derive(Deserialize)]
struct ResourceWire {
    cpu_cores: u32,
    ram_bytes: u64,
    gpu_count: u32,
    gpu_vram_bytes_per_device: u64,
}

impl<'de> Deserialize<'de> for ResourceRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ResourceWire::deserialize(deserializer)?;
        Self::new(
            wire.cpu_cores,
            wire.ram_bytes,
            wire.gpu_count,
            wire.gpu_vram_bytes_per_device,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for ResourceCapacity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ResourceWire::deserialize(deserializer)?;
        Self::new(
            wire.cpu_cores,
            wire.ram_bytes,
            wire.gpu_count,
            wire.gpu_vram_bytes_per_device,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ResourceCapacity {
    pub fn new(
        cpu_cores: u32,
        ram_bytes: u64,
        gpu_count: u32,
        gpu_vram_bytes_per_device: u64,
    ) -> Result<Self, ResourceValidationError> {
        validate_gpu_resources(gpu_count, gpu_vram_bytes_per_device)?;
        Ok(Self {
            cpu_cores,
            ram_bytes,
            gpu_count,
            gpu_vram_bytes_per_device,
        })
    }

    pub fn fits(&self, request: &ResourceRequest) -> bool {
        self.cpu_cores >= request.cpu_cores
            && self.ram_bytes >= request.ram_bytes
            && self.gpu_count >= request.gpu_count
            && self.gpu_vram_bytes_per_device >= request.gpu_vram_bytes_per_device
    }

    pub fn checked_reserve(&self, request: &ResourceRequest) -> Option<Self> {
        self.fits(request).then(|| {
            let gpu_count = self.gpu_count - request.gpu_count;
            Self {
                cpu_cores: self.cpu_cores - request.cpu_cores,
                ram_bytes: self.ram_bytes - request.ram_bytes,
                gpu_count,
                gpu_vram_bytes_per_device: if gpu_count != 0 {
                    self.gpu_vram_bytes_per_device
                } else {
                    0
                },
            }
        })
    }
}

/// Stable worker hardware metadata paired with the resources schedulable on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub name: String,
    pub capacity: ResourceCapacity,
}

fn validate_request(
    cpu_cores: u32,
    gpu_count: u32,
    gpu_vram_bytes_per_device: u64,
) -> Result<(), ResourceValidationError> {
    if cpu_cores == 0 {
        return Err(ResourceValidationError::ZeroCpuCores);
    }
    validate_gpu_resources(gpu_count, gpu_vram_bytes_per_device)
}

fn validate_gpu_resources(
    gpu_count: u32,
    gpu_vram_bytes_per_device: u64,
) -> Result<(), ResourceValidationError> {
    if (gpu_count == 0) != (gpu_vram_bytes_per_device == 0) {
        return Err(ResourceValidationError::InconsistentGpuResources);
    }
    (gpu_count as u64)
        .checked_mul(gpu_vram_bytes_per_device)
        .ok_or(ResourceValidationError::ArithmeticOverflow)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceValidationError {
    ZeroCpuCores,
    InconsistentGpuResources,
    ArithmeticOverflow,
}

impl fmt::Display for ResourceValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ZeroCpuCores => "resource CPU cores must be non-zero",
            Self::InconsistentGpuResources => {
                "GPU count and per-device VRAM must both be zero or both be non-zero"
            }
            Self::ArithmeticOverflow => "resource total overflows its integer representation",
        })
    }
}

impl Error for ResourceValidationError {}
