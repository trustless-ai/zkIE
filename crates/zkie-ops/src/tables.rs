//! P1 Phase 2: `tables.meta` sidecar loading and A-table authentication.
//!
//! The sidecar is a JSON document distributed with the model:
//!
//! ```json
//! {
//!   "version": 1,
//!   "tables": [
//!     { "table_id": 5, "kind": "file", "length": 7340032, "arity": 23,
//!       "file_path": "rsqrt_table_i32.bin", "digest": [1,2,3,4,5,6,7,8] },
//!     { "table_id": 4, "kind": "compute", "length": 2097152, "arity": 21,
//!       "file_path": null, "digest": [9,10,11,12,13,14,15,16] }
//!   ]
//! }
//! ```
//!
//! A tables are public function tables. Their values are computed by the
//! verifier (public formula + graph parameters) or loaded from a file under
//! the model directory; the 32-byte digest anchors them to the independently
//! distributed sidecar. The prover never supplies a digest. Range/identity
//! tables (row `t = t`) are not listed here; they are generated closed-form.
//!
//! Set-level rule: if any entry carries a digest, every entry must carry one
//! and every computed/loaded digest must match. A `file` entry always
//! requires a digest. A `compute` entry with no sidecar digest is allowed only
//! when the whole A set has no digest lines; its computed digest is still
//! recorded in the verifier-owned map used for transcript absorption.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, PrimeField64};
use zkie_core::common::fixed_point::from_i32;

use crate::compose::{ATableEntry, ATableKind, Op, TrustedModelManifest, T};

/// Current sidecar schema version.
pub const TABLES_META_VERSION: u32 = 1;
/// Digest domain separator (the hash is domain-separated and independent of
/// the transcript hash).
pub const A_TABLE_DIGEST_DOMAIN: &[u8] = b"zkie-a-table-v1";

#[derive(Debug)]
pub enum TableMetaError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedVersion(u32),
    DuplicateTable(usize),
    InvalidEntry(String),
    MissingSidecar(PathBuf),
    WrongDigest(usize),
    UnsupportedCompute(usize),
}

impl fmt::Display for TableMetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableMetaError::Io(e) => write!(f, "tables.meta I/O error: {e}"),
            TableMetaError::Json(e) => write!(f, "tables.meta JSON error: {e}"),
            TableMetaError::UnsupportedVersion(v) => write!(f, "unsupported tables.meta version {v}"),
            TableMetaError::DuplicateTable(id) => write!(f, "duplicate tables.meta entry for table {id}"),
            TableMetaError::InvalidEntry(msg) => write!(f, "invalid tables.meta entry: {msg}"),
            TableMetaError::MissingSidecar(p) => write!(f, "missing tables.meta sidecar {}", p.display()),
            TableMetaError::WrongDigest(id) => write!(f, "A table {id} digest mismatch"),
            TableMetaError::UnsupportedCompute(id) => write!(f, "A table {id} has no supported compute formula"),
        }
    }
}

impl std::error::Error for TableMetaError {}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TablesMetaFile {
    version: u32,
    tables: Vec<TableMetaEntry>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TableMetaEntry {
    table_id: usize,
    kind: ATableKind,
    length: usize,
    arity: usize,
    #[serde(default)]
    file_path: Option<String>,
    #[serde(default)]
    digest: Option<[u32; 8]>,
}

/// Parse a sidecar from text. Duplicate ids, malformed entries and unknown
/// fields are rejected; an empty `tables` list is valid.
pub fn parse_tables_meta_str(text: &str)
    -> Result<BTreeMap<usize, ATableEntry>, TableMetaError>
{
    let file: TablesMetaFile = serde_json::from_str(text)
        .map_err(TableMetaError::Json)?;
    if file.version != TABLES_META_VERSION {
        return Err(TableMetaError::UnsupportedVersion(file.version));
    }
    let mut out = BTreeMap::new();
    for e in file.tables {
        if e.length == 0 || e.arity == 0 || e.arity >= usize::BITS as usize {
            return Err(TableMetaError::InvalidEntry(format!(
                "table {} has invalid length/arity ({}/{})", e.table_id, e.length, e.arity)));
        }
        let padded = 1usize.checked_shl(e.arity as u32)
            .ok_or_else(|| TableMetaError::InvalidEntry(
                format!("table {} arity {} overflows", e.table_id, e.arity)))?;
        if e.length > padded {
            return Err(TableMetaError::InvalidEntry(format!(
                "table {} length {} exceeds padded arity {}", e.table_id, e.length, padded)));
        }
        match e.kind {
            ATableKind::File => {
                let path = e.file_path.as_deref().ok_or_else(|| {
                    TableMetaError::InvalidEntry(format!(
                        "file table {} has no file_path", e.table_id))
                })?;
                if path.is_empty() || path.starts_with('/')
                    || path.split('/').any(|c| c == "..")
                {
                    return Err(TableMetaError::InvalidEntry(format!(
                        "file table {} has an unsafe file_path", e.table_id)));
                }
            }
            ATableKind::Compute => {
                if e.file_path.is_some() {
                    return Err(TableMetaError::InvalidEntry(format!(
                        "compute table {} must not carry file_path", e.table_id)));
                }
            }
        }
        let entry = ATableEntry {
            kind: e.kind,
            length: e.length,
            arity: e.arity,
            file_path: e.file_path,
            digest: e.digest,
        };
        if out.insert(e.table_id, entry).is_some() {
            return Err(TableMetaError::DuplicateTable(e.table_id));
        }
    }
    Ok(out)
}

/// Load `<model_dir>/tables.meta`. A missing file is an empty registry; callers
/// that know the graph contains required A tables reject separately.
pub fn load_tables_meta(path: &Path)
    -> Result<BTreeMap<usize, ATableEntry>, TableMetaError>
{
    match std::fs::read_to_string(path) {
        Ok(text) => parse_tables_meta_str(&text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(TableMetaError::Io(e)),
    }
}

/// Digest of a logical A table: SHA-256 over a domain tag, the table identity
/// and shape, and every canonical field element (8 bytes LE). The 32-byte
/// output is returned as eight little-endian 32-bit words.
pub fn hash_a_table(table_id: T, length: usize, arity: usize, values: &[Goldilocks]) -> [u32; 8] {
    assert_eq!(values.len(), length, "A digest must cover the logical table");
    let mut h = Sha256::new();
    h.update(A_TABLE_DIGEST_DOMAIN);
    h.update((table_id as u64).to_le_bytes());
    h.update((length as u64).to_le_bytes());
    h.update((arity as u64).to_le_bytes());
    for v in values {
        h.update(v.as_canonical_u64().to_le_bytes());
    }
    let bytes: [u8; 32] = h.finalize().into();
    let mut words = [0u32; 8];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    }
    words
}

/// A-table values and digests loaded by the caller from the model
/// distribution. `values` are zero-padded to `2^arity`.
pub struct LoadedATables {
    pub entries: BTreeMap<T, ATableEntry>,
    pub values: BTreeMap<T, Arc<Vec<Goldilocks>>>,
    pub digests: BTreeMap<T, [u32; 8]>,
}

impl LoadedATables {
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Install into a manifest: removes the A ids from the PCS root path and
    /// stores the runtime values/digests used by the reductions.
    pub fn install(self, model: &mut TrustedModelManifest) -> Option<()> {
        model.install_a_tables(self.entries, self.values, self.digests)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AUsage { Exp, FileOnly, LookupFile }

/// A-table ids implied by the trusted graph. `Softmax` exp is compute; norm
/// rsqrt and RoPE cos/sin are files; a `Lookup` table is an A table only when
/// the sidecar lists it (otherwise it stays a B table).
fn a_usage(ops: &[Op]) -> BTreeMap<T, AUsage> {
    let mut u = BTreeMap::new();
    for op in ops {
        match op {
            Op::Softmax { table, .. } => { u.insert(*table, AUsage::Exp); }
            Op::RmsNorm { rsqrt_table, .. } => { u.insert(*rsqrt_table, AUsage::FileOnly); }
            Op::Layernorm { rsqrt_table, .. } => { u.insert(*rsqrt_table, AUsage::FileOnly); }
            Op::LayerNormCentered { rsqrt_table, .. } => { u.insert(*rsqrt_table, AUsage::FileOnly); }
            Op::RoPE { cos, sin, .. } => {
                u.insert(*cos, AUsage::FileOnly);
                u.insert(*sin, AUsage::FileOnly);
            }
            _ => {}
        }
    }
    for op in ops {
        if let Op::Lookup { table, .. } = op {
            u.entry(*table).or_insert(AUsage::LookupFile);
        }
    }
    u
}

fn field_writes(ops: &[Op]) -> BTreeSet<T> {
    let tag = if usize::BITS == 64 { 1usize << (usize::BITS - 1) } else { 0 };
    let mut w = BTreeSet::new();
    for op in ops {
        // Use the field/index-encoded wire writes: raw `op.writes()` mixes the
        // two namespaces and can alias an index output onto an A table id.
        let writes = match crate::compose::reverse_op_io(op) {
            Some((_, ws)) => ws,
            None => op.writes(),
        };
        for t in writes { if t & tag == 0 { w.insert(t); } }
    }
    w
}

fn padded_arity(length: usize) -> Option<usize> {
    if length == 0 { return None; }
    Some(length.checked_next_power_of_two()?.max(32).trailing_zeros() as usize)
}

fn load_file_values(dir: &Path, entry: &ATableEntry) -> Result<Vec<Goldilocks>, TableMetaError> {
    let rel = entry.file_path.as_deref().ok_or_else(|| TableMetaError::InvalidEntry(
        "file entry without path".to_string()))?;
    let path = dir.join(rel);
    let bytes = std::fs::read(&path).map_err(TableMetaError::Io)?;
    let want = entry.length.checked_mul(4).ok_or_else(|| TableMetaError::InvalidEntry(
        "table length overflow".to_string()))?;
    if bytes.len() != want {
        return Err(TableMetaError::InvalidEntry(format!(
            "table {} file length {} != {} entries", path.display(), bytes.len(), entry.length)));
    }
    Ok(bytes.chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect())
}

/// Compute the public exp table used by the attention softmax: index
/// `i` maps to `x = (i - 2^21) / 2^16` in `[-32, 0]`, value
/// `round(exp(x) * 2^16)`, clipped to the non-negative i32 range.
pub(crate) fn compute_exp_values(entry: &ATableEntry) -> Result<Vec<Goldilocks>, TableMetaError> {
    if entry.length != (1usize << 21) || entry.arity != 21 {
        return Err(TableMetaError::UnsupportedCompute(0));
    }
    let scale = 65536.0f64;
    let offset = (1usize << 21) as f64;
    Ok((0..entry.length).map(|i| {
        let x = ((i as f64) - offset) / scale;
        let v = (x.exp() * scale).round_ties_even().clamp(0.0, i32::MAX as f64) as i64;
        zkie_core::common::fixed_point::from_i64(v)
    }).collect())
}

/// Load and authenticate the A-table set for `model` from `model_dir`.
///
/// Every id required by the graph must be present; every file entry must
/// exist, have the exact logical length, and match its digest. Range/identity
/// tables are not entries. The returned values are padded to `2^arity`.
pub fn load_a_tables(model_dir: &Path, ops: &[Op], model: &TrustedModelManifest)
    -> Result<LoadedATables, TableMetaError>
{
    let sidecar = model_dir.join("tables.meta");
    let usage = a_usage(ops);
    let required: Vec<T> = usage.iter()
        .filter(|(_, u)| !matches!(u, AUsage::LookupFile))
        .map(|(&id, _)| id).collect();
    if !sidecar.exists() {
        if required.is_empty() {
            return Ok(LoadedATables { entries: BTreeMap::new(), values: BTreeMap::new(), digests: BTreeMap::new() });
        }
        return Err(TableMetaError::MissingSidecar(sidecar));
    }
    let entries = load_tables_meta(&sidecar)?;
    let writes = field_writes(ops);
    for &id in &required {
        if !entries.contains_key(&id) {
            return Err(TableMetaError::InvalidEntry(format!(
                "missing A table {id} required by the graph")));
        }
    }
    // Digest mode is a set-level property.
    let digest_mode = entries.values().any(|e| e.digest.is_some());
    if digest_mode && entries.values().any(|e| e.digest.is_none()) {
        return Err(TableMetaError::InvalidEntry(
            "set-level all-or-none: some A entries carry a digest and some do not".to_string()));
    }
    if !digest_mode && entries.values().any(|e| e.kind == ATableKind::File) {
        return Err(TableMetaError::InvalidEntry(
            "file A entries require a digest".to_string()));
    }
    let mut values = BTreeMap::new();
    let mut digests = BTreeMap::new();
    for (&id, entry) in &entries {
        let use_kind = match usage.get(&id) {
            Some(u) => *u,
            None => return Err(TableMetaError::InvalidEntry(format!(
                "table {id} is not an A table implied by the trusted graph"))),
        };
        if use_kind == AUsage::LookupFile && (writes.contains(&id) || model.is_identity_table_id(id)) {
            return Err(TableMetaError::InvalidEntry(format!(
                "table {id} is dynamic/identity and cannot be an A lookup table")));
        }
        let expected_logical = model.table_logical_length(id).ok_or_else(|| TableMetaError::InvalidEntry(
            format!("table {id} has no trusted lookup length")))?;
        if entry.length != expected_logical {
            return Err(TableMetaError::InvalidEntry(format!(
                "table {id} length {} != trusted {}", entry.length, expected_logical)));
        }
        let arity = padded_arity(entry.length).ok_or_else(|| TableMetaError::InvalidEntry(
            format!("table {id} has invalid length {}", entry.length)))?;
        if entry.arity != arity {
            return Err(TableMetaError::InvalidEntry(format!(
                "table {id} arity {} != padded {}", entry.arity, arity)));
        }
        let (logical_values, kind_ok) = match (use_kind, entry.kind) {
            (AUsage::Exp, ATableKind::Compute) => (compute_exp_values(entry)?, true),
            (AUsage::Exp, ATableKind::File) => (load_file_values(model_dir, entry)?, true),
            (AUsage::FileOnly | AUsage::LookupFile, ATableKind::File) =>
                (load_file_values(model_dir, entry)?, true),
            (_, _) => (Vec::new(), false),
        };
        if !kind_ok {
            return Err(TableMetaError::InvalidEntry(format!(
                "table {id} has a kind not allowed by its graph role")));
        }
        let digest = hash_a_table(id, entry.length, entry.arity, &logical_values);
        if let Some(want) = entry.digest {
            if want != digest { return Err(TableMetaError::WrongDigest(id)); }
        }
        let padded = 1usize.checked_shl(entry.arity as u32).ok_or_else(|| TableMetaError::InvalidEntry(
            format!("table {id} arity overflow")))?;
        let mut vals = logical_values;
        vals.resize(padded, Goldilocks::ZERO);
        values.insert(id, Arc::new(vals));
        digests.insert(id, digest);
    }
    Ok(LoadedATables { entries, values, digests })
}

/// Convenience for model builders/tests: install A tables when the model
/// distribution ships a sidecar; return `false` when it does not (legacy
/// non-A manifest). A present sidecar whose contents fail authentication is
/// an error, never a silent fallback.
pub fn try_install_a_tables(model_dir: &Path, ops: &[Op], model: &mut TrustedModelManifest)
    -> Result<bool, TableMetaError>
{
    if !model_dir.join("tables.meta").exists() { return Ok(false); }
    let loaded = load_a_tables(model_dir, ops, model)?;
    if loaded.is_empty() { return Ok(false); }
    loaded.install(model).ok_or_else(|| TableMetaError::InvalidEntry(
        "A table install rejected the loaded set".to_string()))?;
    Ok(true)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zkie-tables-meta-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parses_compute_and_file_entries() {
        let text = r#"{
            "version": 1,
            "tables": [
                {"table_id": 7, "kind": "file", "length": 7340032, "arity": 23,
                 "file_path": "tables/rsqrt_table_i32.bin",
                 "digest": [1,2,3,4,5,6,7,8]},
                {"table_id": 9, "kind": "compute", "length": 2097152, "arity": 21,
                 "file_path": null, "digest": null}
            ]
        }"#;
        let map = parse_tables_meta_str(text).expect("valid sidecar");
        assert_eq!(map.len(), 2);
        let f = &map[&7];
        assert_eq!(f.kind, ATableKind::File);
        assert_eq!(f.length, 7340032);
        assert_eq!(f.arity, 23);
        assert_eq!(f.digest, Some([1, 2, 3, 4, 5, 6, 7, 8]));
        assert_eq!(map[&9].kind, ATableKind::Compute);
        assert_eq!(map[&9].digest, None);
    }

    #[test]
    fn missing_sidecar_is_empty_registry() {
        let dir = temp_dir("missing");
        assert!(load_tables_meta(&dir.join("tables.meta")).expect("missing file is empty").is_empty());
    }

    #[test]
    fn bad_json_is_rejected() {
        let dir = temp_dir("bad");
        std::fs::write(dir.join("tables.meta"), "{not json").unwrap();
        assert!(matches!(load_tables_meta(&dir.join("tables.meta")), Err(TableMetaError::Json(_))));
    }

    #[test]
    fn duplicate_table_ids_are_rejected() {
        let text = r#"{"version":1,"tables":[
            {"table_id":3,"kind":"compute","length":32,"arity":5,
             "file_path":null,"digest":null},
            {"table_id":3,"kind":"compute","length":32,"arity":5,
             "file_path":null,"digest":null}
        ]}"#;
        assert!(matches!(parse_tables_meta_str(text),
            Err(TableMetaError::DuplicateTable(3))));
    }

    #[test]
    fn file_entry_requires_safe_path() {
        let no_path = r#"{"version":1,"tables":[
            {"table_id":1,"kind":"file","length":32,"arity":5,
             "file_path":null,"digest":null}
        ]}"#;
        assert!(matches!(parse_tables_meta_str(no_path),
            Err(TableMetaError::InvalidEntry(_))));
        let escaping = r#"{"version":1,"tables":[
            {"table_id":1,"kind":"file","length":32,"arity":5,
             "file_path":"../secret.bin","digest":null}
        ]}"#;
        assert!(matches!(parse_tables_meta_str(escaping),
            Err(TableMetaError::InvalidEntry(_))));
    }

    #[test]
    fn unknown_version_and_fields_are_rejected() {
        let v2 = r#"{"version":2,"tables":[]}"#;
        assert!(matches!(parse_tables_meta_str(v2),
            Err(TableMetaError::UnsupportedVersion(2))));
        let unknown = r#"{"version":1,"tables":[],"extra":1}"#;
        assert!(matches!(parse_tables_meta_str(unknown),
            Err(TableMetaError::Json(_))));
    }

    #[test]
    fn digest_is_deterministic_and_shape_bound() {
        let vals: Vec<Goldilocks> = (0..32).map(|i| Goldilocks::from_u64(i)).collect();
        let a = hash_a_table(3, 32, 5, &vals);
        assert_eq!(a, hash_a_table(3, 32, 5, &vals));
        assert_ne!(a, hash_a_table(4, 32, 5, &vals));
        assert_ne!(a, hash_a_table(3, 32, 6, &vals));
    }

    #[test]
    fn file_values_load_and_digest_check() {
        let dir = temp_dir("fileload");
        let vals: Vec<i32> = (0..32).map(|i| i - 16).collect();
        let mut bytes = Vec::new();
        for v in &vals { bytes.extend_from_slice(&v.to_le_bytes()); }
        std::fs::write(dir.join("t.bin"), &bytes).unwrap();
        let logical: Vec<Goldilocks> = vals.iter().map(|&v| from_i32(v)).collect();
        let digest = hash_a_table(1, 32, 5, &logical);
        let entry = ATableEntry {
            kind: ATableKind::File, length: 32, arity: 5,
            file_path: Some("t.bin".to_string()), digest: Some(digest),
        };
        let got = load_file_values(&dir, &entry).unwrap();
        assert_eq!(got, logical);
        assert_eq!(hash_a_table(1, 32, 5, &got), digest);
        let bad = ATableEntry { digest: Some([0u32; 8]), ..entry };
        assert!(matches!(load_file_values(&dir, &bad), Ok(_))); // loader does not hash
    }

    #[test]
    fn compute_exp_matches_public_formula() {
        let entry = ATableEntry {
            kind: ATableKind::Compute, length: 1 << 21, arity: 21,
            file_path: None, digest: None,
        };
        let vals = compute_exp_values(&entry).unwrap();
        assert_eq!(vals.len(), 1 << 21);
        assert_eq!(vals[0], Goldilocks::ZERO); // exp(-32) rounds to 0
        let last = vals.last().unwrap().as_canonical_u64();
        assert!(last > 65500 && last <= 65536, "last={last}");
        assert!(compute_exp_values(&ATableEntry { length: 32, arity: 5, ..entry }).is_err());
    }
}
