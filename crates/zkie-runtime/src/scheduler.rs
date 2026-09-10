use std::cmp::Ordering;
use std::collections::BTreeSet;

use thiserror::Error;
use zkie_types::{ResourceCapacity, ResourceRequest};

use crate::{DbError, JobId, JobState, RunDb};

pub const DEFAULT_AGING_SECONDS: i64 = 600;

pub trait SchedulerClock {
    fn now_unix_seconds(&self) -> i64;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub capacity: ResourceCapacity,
    pub aging_seconds: i64,
}

impl SchedulerConfig {
    pub fn new(capacity: ResourceCapacity) -> Self {
        Self {
            capacity,
            aging_seconds: DEFAULT_AGING_SECONDS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulableJob {
    pub job_id: JobId,
    pub request: ResourceRequest,
    pub ready_since_unix_seconds: i64,
}

impl SchedulableJob {
    pub fn new(job_id: JobId, request: ResourceRequest, ready_since_unix_seconds: i64) -> Self {
        Self {
            job_id,
            request,
            ready_since_unix_seconds,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulerDecision {
    pub selected: Vec<JobId>,
    pub remaining: ResourceCapacity,
    pub reserved_for: Option<JobId>,
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error(transparent)]
    Database(#[from] DbError),
    #[error("duplicate scheduler candidate {0}")]
    DuplicateCandidate(JobId),
}

pub struct Scheduler<C> {
    config: SchedulerConfig,
    clock: C,
}

impl<C: SchedulerClock> Scheduler<C> {
    pub fn new(config: SchedulerConfig, clock: C) -> Self {
        Self { config, clock }
    }

    pub fn decide(
        &self,
        db: &mut RunDb,
        candidates: &[SchedulableJob],
        available: ResourceCapacity,
    ) -> Result<SchedulerDecision, SchedulerError> {
        let mut candidate_ids = BTreeSet::new();
        for candidate in candidates {
            if !candidate_ids.insert(&candidate.job_id) {
                return Err(SchedulerError::DuplicateCandidate(candidate.job_id.clone()));
            }
        }
        db.refresh_readiness()?;
        let gpu_count = available.gpu_count.min(self.config.capacity.gpu_count);
        let available = ResourceCapacity {
            cpu_cores: available.cpu_cores.min(self.config.capacity.cpu_cores),
            ram_bytes: available.ram_bytes.min(self.config.capacity.ram_bytes),
            gpu_count,
            gpu_vram_bytes_per_device: if gpu_count == 0 {
                0
            } else {
                available
                    .gpu_vram_bytes_per_device
                    .min(self.config.capacity.gpu_vram_bytes_per_device)
            },
        };
        let mut eligible = Vec::new();
        for candidate in candidates {
            let record = db.job(&candidate.job_id)?;
            if record.state == JobState::Ready
                && db.required_predecessors_verified(&candidate.job_id)?
            {
                eligible.push(candidate);
            }
        }
        eligible.sort_by(|left, right| {
            left.ready_since_unix_seconds
                .cmp(&right.ready_since_unix_seconds)
                .then_with(|| left.job_id.cmp(&right.job_id))
        });

        let now = self.clock.now_unix_seconds();
        let aged = eligible.iter().copied().find(|candidate| {
            self.config.capacity.fits(&candidate.request)
                && now.saturating_sub(candidate.ready_since_unix_seconds)
                    >= self.config.aging_seconds
        });
        if let Some(aged) = aged {
            if !available.fits(&aged.request) {
                return Ok(SchedulerDecision {
                    selected: Vec::new(),
                    remaining: available,
                    reserved_for: Some(aged.job_id.clone()),
                });
            }
        }

        let mut remaining = available;
        let mut selected = Vec::new();
        if let Some(aged) = aged {
            remaining = remaining
                .checked_reserve(&aged.request)
                .expect("aged request was checked to fit");
            selected.push(aged.job_id.clone());
            eligible.retain(|candidate| candidate.job_id != aged.job_id);
        }

        loop {
            let Some(best_index) = eligible
                .iter()
                .enumerate()
                .filter(|(_, candidate)| remaining.fits(&candidate.request))
                .min_by(|(_, left), (_, right)| compare_fit(left, right, &remaining))
                .map(|(index, _)| index)
            else {
                break;
            };
            let best = eligible.remove(best_index);
            remaining = remaining
                .checked_reserve(&best.request)
                .expect("best-fit request was checked to fit");
            selected.push(best.job_id.clone());
        }

        Ok(SchedulerDecision {
            selected,
            remaining,
            reserved_for: None,
        })
    }
}

fn compare_fit(
    left: &SchedulableJob,
    right: &SchedulableJob,
    available: &ResourceCapacity,
) -> Ordering {
    fit_score(&left.request, available)
        .cmp(&fit_score(&right.request, available))
        .then_with(|| {
            left.ready_since_unix_seconds
                .cmp(&right.ready_since_unix_seconds)
        })
        .then_with(|| left.job_id.cmp(&right.job_id))
}

fn fit_score(request: &ResourceRequest, available: &ResourceCapacity) -> u128 {
    const SCALE: u128 = 1_000_000_000_000;
    let mut score = normalized_leftover(
        u128::from(available.ram_bytes - request.ram_bytes),
        u128::from(available.ram_bytes),
        SCALE,
    ) + normalized_leftover(
        u128::from(available.cpu_cores - request.cpu_cores),
        u128::from(available.cpu_cores),
        SCALE,
    );
    if available.gpu_count != 0 {
        score += normalized_leftover(
            u128::from(available.gpu_count - request.gpu_count),
            u128::from(available.gpu_count),
            SCALE,
        );
        if request.gpu_count != 0 {
            score += normalized_leftover(
                u128::from(available.gpu_vram_bytes_per_device - request.gpu_vram_bytes_per_device),
                u128::from(available.gpu_vram_bytes_per_device),
                SCALE,
            );
        }
    }
    score
}

fn normalized_leftover(leftover: u128, capacity: u128, scale: u128) -> u128 {
    if capacity == 0 {
        0
    } else {
        leftover * scale / capacity
    }
}
