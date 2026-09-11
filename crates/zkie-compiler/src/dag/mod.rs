//! Generic, model-agnostic DAG of independently-provable "shards" over a
//! `CompiledProgram`, plus a pluggable `Prover` and a
//! commitment-consistency `Linker`. See
//! `docs/superpowers/specs/2026-07-27-zkie-dag-sharding-aggregation-design.md`.

pub mod linker;
pub mod model;
pub mod partition;
pub mod prover;

pub use linker::{link, LinkError};
pub use model::{build_dag, BuildDagError, Dag, Edge, EdgeId, EdgeKind, Shard, ShardSpec};
pub use partition::*;
pub use prover::{Commitment, MockProver, Prover, ShardProof, Witness};
