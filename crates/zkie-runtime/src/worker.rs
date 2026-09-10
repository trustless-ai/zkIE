//! Immutable worker protocol between the scheduler and an isolated attempt process.
//!
//! The scheduler writes a [`WorkerSpec`] atomically and launches
//! `zkie worker --spec <abs> --result <abs>`. The worker only reads that spec, writes
//! staged output under `staged_output_dir`, and returns a [`WorkerResult`]; it never
//! touches SQLite. Every message is versioned and canonical so a stale or foreign result
//! is rejected instead of silently accepted.
//!
//! The encoding is a deterministic, line-oriented key/value text format rather than JSON,
//! which keeps the runtime crate free of a serialization dependency while staying strict:
//! the schema version is explicit, values are ASCII and control-free, and the decoder
//! rejects anything it does not fully understand.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::{AttemptId, JobId};
use zkie_types::{Digest32, ResourceRequest};

pub const WORKER_SCHEMA_VERSION: u32 = 1;
const SPEC_MAGIC: &str = "zkie-worker-spec";
const RESULT_MAGIC: &str = "zkie-worker-result";
const MAX_FIELD_BYTES: usize = 4096;
const MAX_FIELDS: usize = 64;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("unsupported worker schema version {0}")]
    UnsupportedSchemaVersion(u32),
    #[error("malformed worker message: {0}")]
    Malformed(&'static str),
    #[error("worker output directory must be absolute: {0}")]
    RelativeOutputDir(PathBuf),
    #[error("worker job needs at least one CPU thread")]
    ZeroThreads,
    #[error("duplicate CUDA device {0}")]
    DuplicateCudaDevice(u32),
    #[error("worker result belongs to attempt {actual} but the spec expects {expected}")]
    AttemptMismatch {
        expected: AttemptId,
        actual: AttemptId,
    },
    #[error("worker result belongs to job {actual} but the spec expects {expected}")]
    JobMismatch { expected: JobId, actual: JobId },
    #[error("worker result run digest does not match the spec")]
    RunDigestMismatch,
    #[error("worker launch failed: {0}")]
    Launch(String),
    #[error("worker protocol I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("worker message carries an invalid identifier: {0}")]
    Identifier(#[from] crate::StateError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkerJobKind {
    Witness,
    Prepare,
    LeafProof,
    LeafVerification,
    NativeAggregate,
}

impl WorkerJobKind {
    pub const ALL: [Self; 5] = [
        Self::Witness,
        Self::Prepare,
        Self::LeafProof,
        Self::LeafVerification,
        Self::NativeAggregate,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Witness => "witness",
            Self::Prepare => "prepare",
            Self::LeafProof => "leaf-proof",
            Self::LeafVerification => "leaf-verification",
            Self::NativeAggregate => "native-aggregate",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == value)
    }
}

/// The job payload: a small, canonical set of ASCII key/value fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerJob {
    pub kind: WorkerJobKind,
    fields: BTreeMap<String, String>,
}

impl WorkerJob {
    pub fn new(
        kind: WorkerJobKind,
        fields: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Self, ProtocolError> {
        let mut canonical = BTreeMap::new();
        for (key, value) in fields {
            validate_token(&key)?;
            validate_value(&value)?;
            canonical.insert(key, value);
        }
        if canonical.len() > MAX_FIELDS {
            return Err(ProtocolError::Malformed("too many job fields"));
        }
        Ok(Self {
            kind,
            fields: canonical,
        })
    }

    pub fn field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }

    pub fn fields(&self) -> impl Iterator<Item = (&str, &str)> {
        self.fields
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerSpec {
    pub schema_version: u32,
    pub run_digest: Digest32,
    pub job_id: JobId,
    pub attempt_id: AttemptId,
    pub job: WorkerJob,
    pub staged_output_dir: PathBuf,
    pub rayon_threads: u32,
    pub cuda_devices: Vec<u32>,
}

impl WorkerSpec {
    /// Rejects anything the worker could not honour safely.
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.schema_version != WORKER_SCHEMA_VERSION {
            return Err(ProtocolError::UnsupportedSchemaVersion(self.schema_version));
        }
        if !self.staged_output_dir.is_absolute() {
            return Err(ProtocolError::RelativeOutputDir(
                self.staged_output_dir.clone(),
            ));
        }
        if self.rayon_threads == 0 {
            return Err(ProtocolError::ZeroThreads);
        }
        let mut seen = Vec::new();
        for device in &self.cuda_devices {
            if seen.contains(device) {
                return Err(ProtocolError::DuplicateCudaDevice(*device));
            }
            seen.push(*device);
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<String, ProtocolError> {
        self.validate()?;
        let mut text = String::new();
        text.push_str(&format!("{SPEC_MAGIC} {}\n", self.schema_version));
        text.push_str(&format!("run_digest {}\n", self.run_digest));
        text.push_str(&format!("job_id {}\n", self.job_id));
        text.push_str(&format!("attempt_id {}\n", self.attempt_id));
        text.push_str(&format!("job_kind {}\n", self.job.kind.as_str()));
        for (key, value) in self.job.fields() {
            text.push_str(&format!("job_field {key}={value}\n"));
        }
        text.push_str(&format!(
            "output_dir {}\n",
            self.staged_output_dir.display()
        ));
        text.push_str(&format!("rayon_threads {}\n", self.rayon_threads));
        for device in &self.cuda_devices {
            text.push_str(&format!("cuda_device {device}\n"));
        }
        Ok(text)
    }

    pub fn decode(text: &str) -> Result<Self, ProtocolError> {
        let mut lines = text.lines();
        let (magic, version) = split_pair(lines.next().ok_or_else(malformed)?)?;
        if magic != SPEC_MAGIC {
            return Err(ProtocolError::Malformed("unexpected message magic"));
        }
        let schema_version = parse_u32(version)?;
        if schema_version != WORKER_SCHEMA_VERSION {
            return Err(ProtocolError::UnsupportedSchemaVersion(schema_version));
        }
        let mut run_digest = None;
        let mut job_id = None;
        let mut attempt_id = None;
        let mut kind = None;
        let mut fields = Vec::new();
        let mut output_dir = None;
        let mut rayon_threads = None;
        let mut cuda_devices = Vec::new();
        for line in lines {
            let (key, value) = split_pair(line)?;
            match key {
                "run_digest" => run_digest = Some(parse_digest(value)?),
                "job_id" => job_id = Some(JobId::new(value)?),
                "attempt_id" => attempt_id = Some(AttemptId::new(value)?),
                "job_kind" => {
                    kind = Some(
                        WorkerJobKind::parse(value)
                            .ok_or(ProtocolError::Malformed("unknown job kind"))?,
                    );
                }
                "job_field" => {
                    let (name, raw) = value
                        .split_once('=')
                        .ok_or(ProtocolError::Malformed("malformed job field"))?;
                    fields.push((name.to_owned(), raw.to_owned()));
                }
                "output_dir" => output_dir = Some(PathBuf::from(value)),
                "rayon_threads" => rayon_threads = Some(parse_u32(value)?),
                "cuda_device" => cuda_devices.push(parse_u32(value)?),
                _ => return Err(ProtocolError::Malformed("unknown spec key")),
            }
        }
        let spec = Self {
            schema_version,
            run_digest: run_digest.ok_or_else(malformed)?,
            job_id: job_id.ok_or_else(malformed)?,
            attempt_id: attempt_id.ok_or_else(malformed)?,
            job: WorkerJob::new(kind.ok_or_else(malformed)?, fields)?,
            staged_output_dir: output_dir.ok_or_else(malformed)?,
            rayon_threads: rayon_threads.ok_or_else(malformed)?,
            cuda_devices,
        };
        spec.validate()?;
        Ok(spec)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerOutcome {
    Succeeded,
    /// A typed, sanitized failure code; never free-form worker text.
    Failed {
        code: String,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkerMeasurements {
    pub peak_resident_bytes: u64,
    pub wall_millis: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerResult {
    pub schema_version: u32,
    pub run_digest: Digest32,
    pub job_id: JobId,
    pub attempt_id: AttemptId,
    pub outcome: WorkerOutcome,
    pub measurements: WorkerMeasurements,
}

impl WorkerResult {
    pub fn succeeded(spec: &WorkerSpec, measurements: WorkerMeasurements) -> Self {
        Self::for_spec(spec, WorkerOutcome::Succeeded, measurements)
    }

    pub fn failed(spec: &WorkerSpec, code: &str, measurements: WorkerMeasurements) -> Self {
        Self::for_spec(
            spec,
            WorkerOutcome::Failed {
                code: sanitize_code(code),
            },
            measurements,
        )
    }

    fn for_spec(
        spec: &WorkerSpec,
        outcome: WorkerOutcome,
        measurements: WorkerMeasurements,
    ) -> Self {
        Self {
            schema_version: spec.schema_version,
            run_digest: spec.run_digest,
            job_id: spec.job_id.clone(),
            attempt_id: spec.attempt_id.clone(),
            outcome,
            measurements,
        }
    }

    pub fn encode(&self) -> Result<String, ProtocolError> {
        if self.schema_version != WORKER_SCHEMA_VERSION {
            return Err(ProtocolError::UnsupportedSchemaVersion(self.schema_version));
        }
        let (status, code) = match &self.outcome {
            WorkerOutcome::Succeeded => ("succeeded", String::new()),
            WorkerOutcome::Failed { code } => ("failed", sanitize_code(code)),
        };
        Ok(format!(
            "{RESULT_MAGIC} {}\nrun_digest {}\njob_id {}\nattempt_id {}\nstatus {}\ncode {}\npeak_resident_bytes {}\nwall_millis {}\n",
            self.schema_version,
            self.run_digest,
            self.job_id,
            self.attempt_id,
            status,
            code,
            self.measurements.peak_resident_bytes,
            self.measurements.wall_millis,
        ))
    }

    pub fn decode(text: &str) -> Result<Self, ProtocolError> {
        let mut lines = text.lines();
        let (magic, version) = split_pair(lines.next().ok_or_else(malformed)?)?;
        if magic != RESULT_MAGIC {
            return Err(ProtocolError::Malformed("unexpected message magic"));
        }
        let schema_version = parse_u32(version)?;
        if schema_version != WORKER_SCHEMA_VERSION {
            return Err(ProtocolError::UnsupportedSchemaVersion(schema_version));
        }
        let mut run_digest = None;
        let mut job_id = None;
        let mut attempt_id = None;
        let mut status = None;
        let mut code = None;
        let mut peak = None;
        let mut wall = None;
        for line in lines {
            let (key, value) = split_pair(line)?;
            match key {
                "run_digest" => run_digest = Some(parse_digest(value)?),
                "job_id" => job_id = Some(JobId::new(value)?),
                "attempt_id" => attempt_id = Some(AttemptId::new(value)?),
                "status" => status = Some(value.to_owned()),
                "code" => code = Some(value.to_owned()),
                "peak_resident_bytes" => peak = Some(parse_u64(value)?),
                "wall_millis" => wall = Some(parse_u64(value)?),
                _ => return Err(ProtocolError::Malformed("unknown result key")),
            }
        }
        let outcome = match status.as_deref() {
            Some("succeeded") => WorkerOutcome::Succeeded,
            Some("failed") => WorkerOutcome::Failed {
                code: sanitize_code(code.as_deref().unwrap_or("unknown")),
            },
            _ => return Err(ProtocolError::Malformed("unknown worker status")),
        };
        Ok(Self {
            schema_version,
            run_digest: run_digest.ok_or_else(malformed)?,
            job_id: job_id.ok_or_else(malformed)?,
            attempt_id: attempt_id.ok_or_else(malformed)?,
            outcome,
            measurements: WorkerMeasurements {
                peak_resident_bytes: peak.ok_or_else(malformed)?,
                wall_millis: wall.ok_or_else(malformed)?,
            },
        })
    }

    /// Accepts the result only when it provably belongs to this spec.
    pub fn verify_against(&self, spec: &WorkerSpec) -> Result<(), ProtocolError> {
        if self.run_digest != spec.run_digest {
            return Err(ProtocolError::RunDigestMismatch);
        }
        if self.job_id != spec.job_id {
            return Err(ProtocolError::JobMismatch {
                expected: spec.job_id.clone(),
                actual: self.job_id.clone(),
            });
        }
        if self.attempt_id != spec.attempt_id {
            return Err(ProtocolError::AttemptMismatch {
                expected: spec.attempt_id.clone(),
                actual: self.attempt_id.clone(),
            });
        }
        Ok(())
    }
}

/// Builds a spec, deriving thread and device placement from the reserved resources.
pub fn worker_spec(
    run_digest: Digest32,
    job_id: JobId,
    attempt_id: AttemptId,
    job: WorkerJob,
    staged_output_dir: PathBuf,
    request: &ResourceRequest,
) -> Result<WorkerSpec, ProtocolError> {
    let spec = WorkerSpec {
        schema_version: WORKER_SCHEMA_VERSION,
        run_digest,
        job_id,
        attempt_id,
        job,
        staged_output_dir,
        rayon_threads: request.cpu_cores,
        cuda_devices: (0..request.gpu_count).collect(),
    };
    spec.validate()?;
    Ok(spec)
}

/// Runs one attempt to completion. Implementations must not touch the run database.
pub trait WorkerLauncher {
    fn launch(
        &mut self,
        spec: &WorkerSpec,
        result_path: &Path,
    ) -> Result<WorkerResult, ProtocolError>;
}

/// Launches `zkie worker --spec <abs> --result <abs>` in its own process group.
pub struct ProcessWorkerLauncher {
    binary: PathBuf,
}

impl ProcessWorkerLauncher {
    pub fn new(binary: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
        }
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }
}

impl WorkerLauncher for ProcessWorkerLauncher {
    fn launch(
        &mut self,
        spec: &WorkerSpec,
        result_path: &Path,
    ) -> Result<WorkerResult, ProtocolError> {
        spec.validate()?;
        let spec_path = spec.staged_output_dir.join("worker-spec.txt");
        std::fs::write(&spec_path, spec.encode()?)?;
        let mut command = std::process::Command::new(&self.binary);
        command
            .arg("worker")
            .arg("--spec")
            .arg(&spec_path)
            .arg("--result")
            .arg(result_path)
            .env("RAYON_NUM_THREADS", spec.rayon_threads.to_string())
            .env(
                "CUDA_VISIBLE_DEVICES",
                spec.cuda_devices
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let status = command.status()?;
        if !status.success() {
            return Err(ProtocolError::Launch(format!(
                "worker exited with {status}"
            )));
        }
        let text = std::fs::read_to_string(result_path)?;
        let result = WorkerResult::decode(&text)?;
        result.verify_against(spec)?;
        Ok(result)
    }
}

fn malformed() -> ProtocolError {
    ProtocolError::Malformed("missing required field")
}

fn split_pair(line: &str) -> Result<(&str, &str), ProtocolError> {
    let (key, value) = line
        .split_once(' ')
        .ok_or(ProtocolError::Malformed("expected a key/value line"))?;
    if key.is_empty() {
        return Err(ProtocolError::Malformed("empty message key"));
    }
    Ok((key, value.trim_end()))
}

fn validate_token(value: &str) -> Result<(), ProtocolError> {
    if value.is_empty()
        || value.len() > 64
        || !value.is_ascii()
        || value.chars().any(|c| c.is_control() || c == ' ')
    {
        return Err(ProtocolError::Malformed("invalid job field name"));
    }
    Ok(())
}

fn validate_value(value: &str) -> Result<(), ProtocolError> {
    if value.len() > MAX_FIELD_BYTES
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.starts_with(' ')
    {
        return Err(ProtocolError::Malformed("invalid job field value"));
    }
    Ok(())
}

/// Worker failure text never survives verbatim; only a bounded token does.
pub fn sanitize_code(code: &str) -> String {
    let token: String = code
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .take(64)
        .collect();
    if token.is_empty() {
        "unknown".to_owned()
    } else {
        token
    }
}

fn parse_u32(value: &str) -> Result<u32, ProtocolError> {
    value
        .parse::<u32>()
        .map_err(|_| ProtocolError::Malformed("invalid integer"))
}

fn parse_u64(value: &str) -> Result<u64, ProtocolError> {
    value
        .parse::<u64>()
        .map_err(|_| ProtocolError::Malformed("invalid integer"))
}

fn parse_digest(value: &str) -> Result<Digest32, ProtocolError> {
    use std::str::FromStr;
    Digest32::from_str(value).map_err(|_| ProtocolError::Malformed("invalid digest"))
}
