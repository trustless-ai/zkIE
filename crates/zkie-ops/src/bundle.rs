//! B2: public verification bundle for the independent verifier.
//!
//! The prover writes exactly two files: `bundle.bin` (trusted graph, public
//! statement, root-only model manifest, caller-owned execution I/O) and
//! `proof.bin` (the BoundExecutionProof bytes). The verifier binary loads
//! only these files — no weights, no prover caches, no store — and returns
//! the bound verifier's decision. The manifest is the offline commit's
//! root-only view; committed tensor data never enters the bundle. Weight
//! authenticity comes exclusively from the manifest roots/commitments, not
//! from bundle data.

use crate::compose::{
    verify_committed_shard_dag_bound_bytes, CachedWeightBatch, Op, PublicStatement, Store,
    TensorKey, TrustedExecution, TrustedModelManifest,
};
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// O7-deep M1 wire version: the proof carries per-arity aux group
/// descriptors (`AUX_LAYOUT_VERSION = 1` in compose.rs). Old bundles
/// (version 1) are rejected; there is no silent reinterpretation.
/// M4 adds per-table internal lookup batch openings: version 4.
/// P1 adds the carried A-table registry and digest absorption (v5).
/// P2 moves GeluIndex/Relu bit openings into the pooled aux plan:
/// version 6; older versions are rejected.
pub const BUNDLE_VERSION: u32 = 6;
/// Hard cap on each serialized bundle file before decoding.
pub const MAX_BUNDLE_FILE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
pub struct PublicBundle {
    pub version: u32,
    pub ops: Vec<Op>,
    pub ops_per_shard: usize,
    pub statement: PublicStatement,
    pub model: TrustedModelManifest,
    pub execution: TrustedExecution,
}

impl PublicBundle {
    pub fn new(ops: Vec<Op>, ops_per_shard: usize, statement: PublicStatement,
        model: TrustedModelManifest, execution: TrustedExecution) -> Self
    {
        Self { version: BUNDLE_VERSION, ops, ops_per_shard, statement, model, execution }
    }

    pub fn write(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let bytes = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(dir.join("bundle.bin"), bytes)
    }

    pub fn read(dir: &Path) -> Option<Self> {
        let path = dir.join("bundle.bin");
        let len = std::fs::metadata(&path).ok()?.len();
        if len > MAX_BUNDLE_FILE_BYTES { return None; }
        let bytes = std::fs::read(&path).ok()?;
        let bundle: Self = bincode::deserialize(&bytes).ok()?;
        if bundle.version != BUNDLE_VERSION { return None; }
        Some(bundle)
    }
}

pub fn write_proof(dir: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("proof.bin"), bytes)
}

pub fn read_proof(dir: &Path) -> Option<Vec<u8>> {
    let path = dir.join("proof.bin");
    let len = std::fs::metadata(&path).ok()?.len();
    if len > MAX_BUNDLE_FILE_BYTES { return None; }
    std::fs::read(&path).ok()
}

/// Write both public artifacts; the prover-side store/weights stay behind.
pub fn write_bundle(dir: &Path, bundle: &PublicBundle, proof: &[u8]) -> std::io::Result<()> {
    bundle.write(dir)?;
    write_proof(dir, proof)
}

/// Independent verifier: reads only bundle.bin + proof.bin. A-bearing
/// bundles additionally need the model distribution directory (sidecar +
/// table files) from `ZKIE_MODEL_DIR`; it is never read from the proof.
pub fn verify_bundle_dir(dir: &Path) -> bool {
    let model_dir = std::env::var_os("ZKIE_MODEL_DIR").map(std::path::PathBuf::from);
    verify_bundle_dir_with_model(dir, model_dir.as_deref())
}

/// Verifier entry with an explicit caller-owned model directory. The
/// manifest copy inside the bundle is compared against the distribution
/// sidecar; a mismatch rejects before proof verification. Non-A bundles do
/// not require a model directory.
pub fn verify_bundle_dir_with_model(dir: &Path, model_dir: Option<&Path>) -> bool {
    let mut bundle = match PublicBundle::read(dir) { Some(b) => b, None => return false };
    let proof = match read_proof(dir) { Some(p) => p, None => return false };
    match model_dir {
        Some(model_dir) => {
            // Caller-owned model distribution is the authoritative A set: the
            // bundle's caller copy must match it exactly, and an A-bearing
            // graph without a usable sidecar rejects.
            let entries = match crate::tables::load_tables_meta(&model_dir.join("tables.meta")) {
                Ok(v) => v,
                Err(_) => return false,
            };
            if entries != bundle.model.a_tables { return false; }
            let loaded = match crate::tables::load_a_tables(model_dir, &bundle.ops, &bundle.model) {
                Ok(v) => v,
                Err(_) => return false,
            };
            if !loaded.is_empty() && loaded.install(&mut bundle.model).is_none() { return false; }
        }
        None => {
            // Without the distribution we can only verify a bundle that
            // carries no A registry. A-bearing bundles require the caller's
            // model directory.
            if !bundle.model.a_tables.is_empty() { return false; }
        }
    }
    // The bound verifier rebuilds its WHIR parameters from VERIFIER_SECURITY
    // and each opening's arity; the `whir` argument is part of the legacy
    // signature and is ignored (`let _ = whir` in the verifier). Pass a
    // policy-default instance only to satisfy that signature.
    let whir = zkie_core::pcs::whir::Whir::new_testing(5);
    verify_committed_shard_dag_bound_bytes(
        &proof, &bundle.ops, bundle.ops_per_shard, &whir,
        &bundle.statement, &bundle.model, &bundle.execution,
    )
}

fn execution_value(store: &Store, key: usize) -> Option<Vec<Goldilocks>> {
    let tag = 1usize << (usize::BITS - 1);
    if key & tag != 0 {
        let id = key & !tag;
        let idx = store.idx.get(id)?;
        Some(idx.iter().map(|&v| Goldilocks::from_u64(v as u64)).collect())
    } else {
        Some(store.get(key).to_vec())
    }
}

/// B2 helper: derive the caller-owned public execution from the trusted graph
/// and the evaluated store. Inputs are graph reads that are neither written
/// internally nor committed weights (reverse IO already excludes trusted
/// static tables); outputs are the single Argmax's columns for language
/// models, or the last op's writes for a Generic graph. Fails closed if any
/// op lacks a reverse IO contract.
pub fn public_execution(store: &Store, ops: &[Op], cached: &[CachedWeightBatch],
    model: &TrustedModelManifest) -> Option<TrustedExecution>
{
    let tag = 1usize << (usize::BITS - 1);
    let mut weights: BTreeSet<usize> = BTreeSet::new();
    for b in cached {
        for &id in &b.table_ids { if id & tag == 0 { weights.insert(id); } }
    }
    let mut reads: BTreeSet<usize> = BTreeSet::new();
    let mut writes: BTreeSet<usize> = BTreeSet::new();
    for op in ops {
        let (r, w) = crate::compose::reverse_op_io(op)?;
        reads.extend(r);
        writes.extend(w);
    }
    let mut inputs = BTreeMap::new();
    for &k in reads.difference(&writes) {
        if weights.contains(&k) || model.a_tables.contains_key(&k) { continue; }
        inputs.insert(k, execution_value(store, k)?);
    }
    let argmax: Vec<usize> = ops.iter().enumerate()
        .filter(|(_, op)| matches!(op, Op::Argmax { .. }))
        .map(|(i, _)| i).collect();
    let src = if argmax.len() == 1 { argmax[0] } else { ops.len().checked_sub(1)? };
    let (_, outs) = crate::compose::reverse_op_io(&ops[src])?;
    let mut outputs = BTreeMap::new();
    for k in outs { outputs.insert(k, execution_value(store, k)?); }
    Some(TrustedExecution { inputs, outputs })
}

/// B2 helper: trusted index-tensor schema (lengths) for the manifest. Values
/// never enter the manifest; only the caller-owned shape metadata.
pub fn index_shapes(store: &Store) -> BTreeMap<usize, usize> {
    let mut shapes = BTreeMap::new();
    for (id, v) in store.idx.iter().enumerate() {
        if !v.is_empty() { shapes.insert(id, v.len()); }
    }
    shapes
}
