use std::fmt;

use thiserror::Error;

const MAX_ID_BYTES: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttemptId(String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerificationRecordId(String);

macro_rules! identifier {
    ($name:ident) => {
        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, StateError> {
                let value = value.into();
                if value.is_empty()
                    || value.len() > MAX_ID_BYTES
                    || !value.is_ascii()
                    || value.chars().any(char::is_control)
                {
                    return Err(StateError::InvalidIdentifier);
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

identifier!(JobId);
identifier!(AttemptId);
identifier!(VerificationRecordId);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Witness,
    Prepare,
    LeafProof,
    LeafVerification,
    NativeAggregate,
}

impl JobKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Witness => "witness",
            Self::Prepare => "prepare",
            Self::LeafProof => "leaf-proof",
            Self::LeafVerification => "leaf-verification",
            Self::NativeAggregate => "native-aggregate",
        }
    }
}

impl TryFrom<&str> for JobKind {
    type Error = UnknownEnumValue;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "witness" => Ok(Self::Witness),
            "prepare" => Ok(Self::Prepare),
            "leaf-proof" => Ok(Self::LeafProof),
            "leaf-verification" => Ok(Self::LeafVerification),
            "native-aggregate" => Ok(Self::NativeAggregate),
            _ => Err(UnknownEnumValue(value.to_owned())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobState {
    Pending,
    Ready,
    Witnessing,
    WitnessReady,
    Preparing,
    Prepared,
    Proving,
    Proved,
    Verifying,
    Verified,
    ExecutionFailed,
    VerificationFailed,
    ResourceExceeded,
    Interrupted,
    Requeued,
    Blocked,
}

impl JobState {
    pub fn transition(self, event: JobEvent) -> Result<Self, TransitionError> {
        use JobEvent as E;
        use JobState as S;

        let next = match (&self, &event) {
            (S::Pending | S::Requeued, E::DependenciesSatisfied) => S::Ready,
            (S::Pending | S::Requeued, E::DependencyBlocked { .. }) => S::Blocked,
            (S::Ready, E::StartWitnessing) => S::Witnessing,
            (S::Witnessing, E::WitnessCompleted) => S::WitnessReady,
            (S::Ready | S::WitnessReady, E::StartPreparing) => S::Preparing,
            (S::Preparing, E::PreparationCompleted) => S::Prepared,
            (S::Ready | S::Preparing, E::StartProving) => S::Proving,
            (S::Proving, E::ProofProduced) => S::Proved,
            (S::Ready | S::WitnessReady | S::Preparing | S::Proved, E::StartVerification) => {
                S::Verifying
            }
            (
                S::WitnessReady | S::Prepared | S::Proved | S::Verifying,
                E::VerificationSucceeded {
                    verification_record: Some(_),
                },
            ) => S::Verified,
            (
                S::Witnessing | S::Preparing | S::Proving | S::Verifying,
                E::ExecutionFailed(failure),
            ) if failure.kind == FailureKind::Execution => S::ExecutionFailed,
            (
                S::WitnessReady | S::Prepared | S::Proved | S::Verifying,
                E::VerificationFailed(failure),
            ) if failure.kind == FailureKind::Verification => S::VerificationFailed,
            (
                S::Witnessing | S::Preparing | S::Proving | S::Verifying,
                E::ResourceExceeded(failure),
            ) if failure.kind == FailureKind::ResourceExceeded => S::ResourceExceeded,
            (S::Witnessing | S::Preparing | S::Proving | S::Verifying, E::Interrupted(failure))
                if failure.kind == FailureKind::Interrupted =>
            {
                S::Interrupted
            }
            (S::ExecutionFailed, E::RetryExecution) => S::Requeued,
            (S::ResourceExceeded, E::RetryResourceExceeded) => S::Requeued,
            (S::Interrupted, E::RecoveryRequeue) => S::Requeued,
            _ => {
                return Err(TransitionError {
                    from: self,
                    event: event.name(),
                })
            }
        };
        Ok(next)
    }

    pub fn is_permanent_failure(self) -> bool {
        matches!(self, Self::VerificationFailed | Self::Blocked)
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Ready => "ready",
            Self::Witnessing => "witnessing",
            Self::WitnessReady => "witness-ready",
            Self::Preparing => "preparing",
            Self::Prepared => "prepared",
            Self::Proving => "proving",
            Self::Proved => "proved",
            Self::Verifying => "verifying",
            Self::Verified => "verified",
            Self::ExecutionFailed => "execution-failed",
            Self::VerificationFailed => "verification-failed",
            Self::ResourceExceeded => "resource-exceeded",
            Self::Interrupted => "interrupted",
            Self::Requeued => "requeued",
            Self::Blocked => "blocked",
        }
    }

    pub(crate) fn is_active(self) -> bool {
        matches!(
            self,
            Self::Witnessing | Self::Preparing | Self::Proving | Self::Verifying
        )
    }
}

impl TryFrom<&str> for JobState {
    type Error = UnknownEnumValue;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "pending" => Ok(Self::Pending),
            "ready" => Ok(Self::Ready),
            "witnessing" => Ok(Self::Witnessing),
            "witness-ready" => Ok(Self::WitnessReady),
            "preparing" => Ok(Self::Preparing),
            "prepared" => Ok(Self::Prepared),
            "proving" => Ok(Self::Proving),
            "proved" => Ok(Self::Proved),
            "verifying" => Ok(Self::Verifying),
            "verified" => Ok(Self::Verified),
            "execution-failed" => Ok(Self::ExecutionFailed),
            "verification-failed" => Ok(Self::VerificationFailed),
            "resource-exceeded" => Ok(Self::ResourceExceeded),
            "interrupted" => Ok(Self::Interrupted),
            "requeued" => Ok(Self::Requeued),
            "blocked" => Ok(Self::Blocked),
            _ => Err(UnknownEnumValue(value.to_owned())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    Execution,
    Verification,
    ResourceExceeded,
    Dependency,
    Interrupted,
}

impl FailureKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Execution => "execution",
            Self::Verification => "verification",
            Self::ResourceExceeded => "resource-exceeded",
            Self::Dependency => "dependency",
            Self::Interrupted => "interrupted",
        }
    }

    pub(crate) fn summary(self) -> &'static str {
        match self {
            Self::Execution => "execution failed",
            Self::Verification => "verification failed",
            Self::ResourceExceeded => "resource limit exceeded",
            Self::Dependency => "dependency failed",
            Self::Interrupted => "interrupted",
        }
    }
}

impl TryFrom<&str> for FailureKind {
    type Error = UnknownEnumValue;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "execution" => Ok(Self::Execution),
            "verification" => Ok(Self::Verification),
            "resource-exceeded" => Ok(Self::ResourceExceeded),
            "dependency" => Ok(Self::Dependency),
            "interrupted" => Ok(Self::Interrupted),
            _ => Err(UnknownEnumValue(value.to_owned())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureStage {
    Witness,
    Prepare,
    Prove,
    Verify,
    Aggregate,
    Scheduler,
}

impl FailureStage {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Witness => "witness",
            Self::Prepare => "prepare",
            Self::Prove => "prove",
            Self::Verify => "verify",
            Self::Aggregate => "aggregate",
            Self::Scheduler => "scheduler",
        }
    }
}

impl TryFrom<&str> for FailureStage {
    type Error = UnknownEnumValue;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "witness" => Ok(Self::Witness),
            "prepare" => Ok(Self::Prepare),
            "prove" => Ok(Self::Prove),
            "verify" => Ok(Self::Verify),
            "aggregate" => Ok(Self::Aggregate),
            "scheduler" => Ok(Self::Scheduler),
            _ => Err(UnknownEnumValue(value.to_owned())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureCode {
    WorkerExit,
    InvalidResult,
    ResourceLimit,
    PredecessorFailed,
    SchedulerRestart,
}

impl FailureCode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::WorkerExit => "worker_exit",
            Self::InvalidResult => "invalid_result",
            Self::ResourceLimit => "resource_limit",
            Self::PredecessorFailed => "predecessor_failed",
            Self::SchedulerRestart => "scheduler_restart",
        }
    }
}

impl TryFrom<&str> for FailureCode {
    type Error = UnknownEnumValue;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "worker_exit" => Ok(Self::WorkerExit),
            "invalid_result" => Ok(Self::InvalidResult),
            "resource_limit" => Ok(Self::ResourceLimit),
            "predecessor_failed" => Ok(Self::PredecessorFailed),
            "scheduler_restart" => Ok(Self::SchedulerRestart),
            _ => Err(UnknownEnumValue(value.to_owned())),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailureRecord {
    kind: FailureKind,
    stage: FailureStage,
    code: FailureCode,
    summary: &'static str,
}

impl FailureRecord {
    pub fn new(kind: FailureKind, stage: FailureStage, code: FailureCode) -> Self {
        Self {
            kind,
            stage,
            code,
            summary: kind.summary(),
        }
    }

    pub fn from_untrusted_message(
        kind: FailureKind,
        stage: FailureStage,
        code: FailureCode,
        _message: impl AsRef<str>,
    ) -> Self {
        Self {
            kind,
            stage,
            code,
            summary: "[REDACTED]",
        }
    }

    pub fn kind(&self) -> FailureKind {
        self.kind
    }

    pub fn stage(&self) -> FailureStage {
        self.stage
    }

    pub fn code(&self) -> FailureCode {
        self.code
    }

    pub fn summary(&self) -> &str {
        self.summary
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobEvent {
    DependenciesSatisfied,
    DependencyBlocked {
        predecessor: JobId,
    },
    StartWitnessing,
    WitnessCompleted,
    StartPreparing,
    PreparationCompleted,
    StartProving,
    ProofProduced,
    StartVerification,
    VerificationSucceeded {
        verification_record: Option<VerificationRecordId>,
    },
    ExecutionFailed(FailureRecord),
    VerificationFailed(FailureRecord),
    ResourceExceeded(FailureRecord),
    Interrupted(FailureRecord),
    RetryExecution,
    RetryResourceExceeded,
    RecoveryRequeue,
}

impl JobEvent {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::DependenciesSatisfied => "dependencies-satisfied",
            Self::DependencyBlocked { .. } => "dependency-blocked",
            Self::StartWitnessing => "start-witnessing",
            Self::WitnessCompleted => "witness-completed",
            Self::StartPreparing => "start-preparing",
            Self::PreparationCompleted => "preparation-completed",
            Self::StartProving => "start-proving",
            Self::ProofProduced => "proof-produced",
            Self::StartVerification => "start-verification",
            Self::VerificationSucceeded { .. } => "verification-succeeded",
            Self::ExecutionFailed(_) => "execution-failed",
            Self::VerificationFailed(_) => "verification-failed",
            Self::ResourceExceeded(_) => "resource-exceeded",
            Self::Interrupted(_) => "interrupted",
            Self::RetryExecution => "retry-execution",
            Self::RetryResourceExceeded => "retry-resource-exceeded",
            Self::RecoveryRequeue => "recovery-requeue",
        }
    }

    pub(crate) fn failure(&self) -> Option<&FailureRecord> {
        match self {
            Self::ExecutionFailed(failure)
            | Self::VerificationFailed(failure)
            | Self::ResourceExceeded(failure)
            | Self::Interrupted(failure) => Some(failure),
            _ => None,
        }
    }

    pub(crate) fn blocking_predecessor(&self) -> Option<&JobId> {
        match self {
            Self::DependencyBlocked { predecessor } => Some(predecessor),
            _ => None,
        }
    }

    pub(crate) fn verification_record(&self) -> Option<&VerificationRecordId> {
        match self {
            Self::VerificationSucceeded {
                verification_record,
            } => verification_record.as_ref(),
            _ => None,
        }
    }

    pub(crate) fn starts_attempt(&self) -> bool {
        matches!(
            self,
            Self::StartWitnessing
                | Self::StartPreparing
                | Self::StartProving
                | Self::StartVerification
        )
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("illegal transition from {from:?} via {event}")]
pub struct TransitionError {
    pub from: JobState,
    pub event: &'static str,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum StateError {
    #[error("invalid identifier")]
    InvalidIdentifier,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("unknown enum value {0}")]
pub struct UnknownEnumValue(pub String);
