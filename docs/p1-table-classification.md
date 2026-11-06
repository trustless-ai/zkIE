# P1 table classification (A: public function tables / B: weight-derived tables)

Status: design inventory for TL review, 2026-11-06. No code changes in this
document. Authority for mechanism/path choices: local FlashProve fork
`/Users/shakku/trustless-ai/FlashProve`, HEAD `289dd85f` (not the network
copy). FlashProve file references below are to that fork.

## 0. TL decisions recorded

1. **Gelu function table -> A (load file + digest, A10).** It is a public
   function table in the same class as FP's SiLU/gate. The offline fixed-point
   artifact uses the second path. If a public bit-exact generator spec is
   published later, it can move to the first path (compute in verifier).
2. **rsqrt -> A, load file + digest** (second path). FP has no rsqrt table;
   FP explicitly permits "takes them as data (what then fixes them: R15.3)".
   Computing it in the verifier remains an option after
   `scripts/extract_rsqrt_piecewise.py` is declared a public bit-exact spec.
3. **A-table digest source -> model-distribution sidecar** (`tables.meta`,
   parameter-file style). The model distribution ships the sidecar and table
   files; the caller loads them into the manifest A-registry. The prover never
   writes a digest. The manifest inside the bundle is a caller copy, not the
   only source; the ultimate anchor is the model distribution.
4. **RoPE cos/sin -> A, load file + digest.** They move from execution inputs
   to A-registry table binding: RoPE claims are evaluated against the
   digest-verified table values. FP's `ropeC` compute path (first path) is a
   possible follow-up, not the initial implementation.

Additional standing rules: A tables are never supplied with a prover-carried
hash; B tables keep their existing commitments; dynamic op-written tables keep
producer-claim authentication.

## 1. Full table inventory

Registry paths in zkIE (`crates/zkie-ops/src/compose.rs`):
`identity_tables` (closed-form range tables), `lookup_roots` (individual
static table commitments), `static_table_commits` (M4a once-per-proof static
commit), `static_lookup_groups`/`LookupGroups` (static `Op::Lookup` tables),
`op_field_writes` (dynamic op-written tables), and `model.batches`
(`commit_model_weights` weight commitments).

| # | table / purpose | source op (models) | current registration / authentication | class | suggested path | reason / FlashProve reference |
|---|---|---|---|---|---|---|
| A1 | Softmax exp table (`e = exp_table[idx]`, padded 2^21 in GPT-2/Gemma3/DeepSeek; model-specific in TimesFM) | `Op::Softmax.table` (GPT-2, TimesFM, Gemma3, DeepSeek-V2-Lite) | `lookup_roots` single-table root + `static_table_commits` (M4a) + internal per-table batch opening | A | **compute in verifier + digest** | Public exp function. FP `verifier/src/vf_attention_shape.h:206` builds `Tt` with `nearbyint(ldexp(1,S)*exp(-u/2^rx))`; `verifier/src/cp_verify.cpp:2010` hashes `AF[0].Tt` with `earlier_hash_h_digest(hh,72,...)` -> `h.hTexp` -> `tblHeld(3,"table-digest-exp")`; FORMAT R15.3 |
| A2 | StableSoftmaxIndex "exp" table | `Op::StableSoftmaxIndex` has only `table_len`; the exp values come from the following `Op::Softmax.table` | No table of its own; reuses A1 | A | **same as A1** | SSI computes index/clamp/relu only; FP's softmax table is the same public table path |
| A3 | RmsNorm rsqrt table (7,340,032 entries -> padded 2^23) | `Op::RmsNorm.rsqrt_table` (Gemma3, DeepSeek, TimesFM) | `lookup_roots` + M4a static commit + internal per-table batch opening | A | **load file + digest** (decision 2) | FP has no rsqrt table; FP's second path "takes them as data (R15.3)" applies. zkIE generator: `scripts/extract_rsqrt_piecewise.py` (piecewise f64 `1/sqrt`, `np.round` ties-to-even, scale 2^16) |
| A4 | LayerNormCentered rsqrt table | `Op::LayerNormCentered.rsqrt_table` (GPT-2, TimesFM) | `lookup_roots` + M4a | A | **load file + digest** | Same as A3 |
| A5 | legacy Layernorm rsqrt table | `Op::Layernorm.rsqrt_table` (legacy/tests) | `lookup_roots` | A | **load file + digest** | Same as A3 |
| A6 | RoPE cos/sin tables | `Op::RoPE.cos/sin` (Gemma3, DeepSeek, TimesFM RoPE) | Currently execution inputs: reverse claims are checked against caller execution values; not in `lookup_roots`/`static_table_commits` | A | **load file + digest** (decision 4), claims evaluate against verified table values | FP computes rotary tables (`cp_verify.cpp:1609 ropeC`) and hashes them into `h.hRope` (`:1426`, `tblHeld(2,"table-digest-rope")`); FP's compute path is a later option |
| A7 | RoPE range table | `Op::RoPE.range_table` (models pass `limb_t`, 2^16 identity) | `identity_tables`, no commitment/opening | A | **compute (closed form, row t=t)** | 11-1 *Public tables*: "A range table has no listed rows: row t is the number t" |
| A8 | RmsNorm limb table | `Op::RmsNorm.limb_table` (2^16 identity) | `identity_tables` | A | **compute (closed form)** | 11-1 ranges of 16 bits |
| A9 | SoftmaxIndex/GeluIndex/Relu/Scale/ScaleVec range tables | `Op::SoftmaxIndex.table_len`, `Op::GeluIndex.table_len`, `Op::Relu` (2^22), `Op::Scale/ScaleVec` (2^shift) | `ensure_public_tables`/`identity_tables` (+2^22 identity tables for SoftmaxIndex/GeluIndex) | A | **compute (closed form)** | 11-1 ranges; row t=t; no digest needed |
| A10 | Gelu function table (2^24) | `Op::Lookup.table = gelu_t` (GPT-2, Gemma3) | `lookup_roots` + `LookupGroups` | A | **load file + digest** (decision 1) | Public function table; FP gate/SiLU is the same class but zkIE's tanh-GELU has no FP counterpart; offline generator `scripts/extract_gpt2.py` (0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))), scale 2^16). Future: public bit-exact spec -> compute path |
| A11 | SiLU function table (TimesFM 2^20; DeepSeek model-specific) | `Op::Lookup.table = silu_t` (DeepSeek, TimesFM) | `lookup_roots` + `LookupGroups` | A | **compute in verifier + digest** | FP `gateTableEntry` (`cp_verify.cpp:202`, double round) + hashed `h.hTs` (`:1418`, `tblHeld(0,"table-digest-silu")`); FORMAT R15.3 |
| A12 | Sigmoid table (reserved; no current zkIE model) | future `Op::Lookup.table` | — | A | **compute in verifier + digest** if added | FP `sigmoidTableEntry` (`cp_verify.cpp:183`) + `h.hSig` (`:1426`); R15.3 |
| B1 | Embedding / wte table | `Op::Lookup.table`: GPT-2/Gemma3 `wte_owned`, DeepSeek `embed_tokens` clone | `lookup_roots` + `LookupGroups` | **B** | keep committed | Weight-derived; FP 11-5 committed matrix, embedding kind 12 |
| B2 | LM head / output head table | GPT-2 `lm_head` used as Lookup table; Gemma3/DeepSeek head weights | `lookup_roots`/`LookupGroups`; or `model.batches` for MatMul b | **B** | keep committed | FP 11-5 output head kind 13 |
| B3 | Dense weights: MatMul b, Projection w/bias, BatchedMatMul w, RmsNorm w, LNC w/b, Layernorm w/b | dense/attention/FFN ops | `commit_model_weights` -> `model.batches` + weight batch opening | **B** | keep committed | FP 11-5 committed matrix W units |
| B4 | Expert/router weights: GatherMatMul/BatchedMatMul `eg/eu/ed`, router | DeepSeek/Gemma3 MoE | weight batches; GatherMatMul expert slices bound by offline manifest | **B** | keep committed | FP 11-5 kinds 7-11 (routed/shared experts, router) |
| B5 | Dynamic op-written tables (activation/routing derived) | `Op::Lookup.table` where `op_field_writes(ops)` contains the id (e.g. DeepSeek `table: x`, `table: gate`) | No `lookup_roots`; dynamic Lookup returns producer `EfClaim`, authenticated by the producing op's committed cells/claims | **B (must remain authenticated)** | keep producer-claim path | Not a public function; derived from committed activations/weights. FP has no "dynamic activation table"; principle is committed cells + claims (11-5 spirit) |

Note: `StableSoftmaxIndex` itself has no exp table; `GeluIndex` itself has only
identity range tables (A9); the gelu function-value table is the following
`Op::Lookup.table` (A10).

## 2. Mechanism conclusions

### 2.1 Two A-table paths (mirroring FP)

**Path 1 - compute in verifier.** The verifier computes the table to the bit
from public mathematics and public parameters, hashes it with the
domain-separated digest (FP uses the second hash function
`K(72; table)`, FORMAT R15.3), stores the digest in verifier-owned header
words (FP `hTs/hSig/hTexp`), and compares it with the parameter/sidecar line
when present (mismatch rejects). Used by A1 (exp), A11 (SiLU), A12 (sigmoid
if ever added); RoPE A6 may move here later.

**Path 2 - load file + digest.** The A table is an offline file. The verifier
loads it, checks length/arity/shape, streams a hash, and compares it with the
digest from `tables.meta` (model distribution sidecar). Missing/mismatched
file rejects before proof verification; table values are cached. Used by
A3-A5 (rsqrt), A10 (Gelu), and A6 (RoPE cos/sin) per decisions 1, 2, 4.

**Range/identity tables** (A7-A9) are computed closed-form (`row t = t`);
they need no file or 32-byte digest (an optional generator-version digest may
be recorded).

### 2.2 Digest channel and independence

- **Sidecar**: each model distribution ships `tables.meta`, parameter-file
  style, with one entry per A table: table id, kind (`compute` or `file`),
  length, arity, file path (for `file`), 32-byte digest (8 x 32-bit words,
  FP style), and hash domain. Model files are distributed with the sidecar.
- **Caller path**: the caller passes the model directory/sidecar to the
  verifier; the verifier loads it into the manifest A-registry. The manifest
  inside `bundle.bin` is a caller copy for reproducibility, **not the only
  source**; the verifier must take digests from the distribution sidecar and
  reject a bundle whose copy disagrees. The prover never writes a digest.
- **Commitment parameter set**: the A digest set enters the configuration
  digest / commitment parameters before challenges, matching FP's chain:
  `earlier_hash_h_digest(...,72,...)` -> verifier-owned header word ->
  parameters/statement -> configuration digest (R4.9) -> commitment ID
  (R4.11). A digest mismatch is fatal; the prover cannot substitute a table
  or a hash.
- **Set-level all-or-none (集合级全或无)**: if ANY A-table entry carries a
  digest, then ALL A-table entries (both `compute` and `file`) must carry a
  digest and every one must match. A `compute` entry whose sidecar has no
  digest line is allowed only when the whole A set has no digest lines; the
  verifier still computes the table and records its digest in the
  verifier-owned header slot (so the header is always complete). A `file`
  entry requires a digest, so the presence of any `file` entry forces the
  digest-bearing mode for the whole set. This is the set-level generalization
  of FP 11-1 "Either all 32 lines are present or none" (the FP parameter
  file's four tables are all-or-none). A table registry entry not justified
  by the trusted graph rejects.

### 2.3 Verifier: load + hash + async parallel

1. Parse `tables.meta`; validate table ids/kinds/lengths/arities against the
   trusted op graph and manifest; load A-registry.
2. For `file` entries: open the table file under the model directory; check
   byte length/shape; stream-hash; compare digest; cache `digest -> values`.
3. For `compute` entries: compute the table (double-precision formula and
   rounding as specified); hash; compare optional sidecar digest; cache.
4. Hash/load tables concurrently (rayon/task pool) with bounded concurrency
   and memory; fail-closed on missing/short/mismatched/duplicate entries.
5. Absorb the A digest set into the transcript/config before any challenge.
6. Use the cached A table values in the reverse reductions; no A-table PCS
   opening or commitment is read from the proof.
7. Range/identity tables are generated locally (no file I/O).

### 2.4 Proof / manifest / bundle landing points

- `TrustedModelManifest`: add A-table registry/digest fields (serde schema);
  keep `lookup_roots` only for B static tables and other non-A tables.
- `BUNDLE_VERSION`: 4 -> **5**; `verify_bundle_dir` checks the version,
  resolves the model-distribution sidecar, loads/validates A tables, then
  calls the proof verifier with the loaded A tables. Old versions reject.
- Prover/proof paths: A tables lose `lookup_roots`, `static_table_commits`,
  `internal_table_openings`, and `LookupGroups` table openings; M4a
  per-table batch openings remain for B static lookup tables
  (embedding/head) if applicable. `opening_upper_bound` and `proof_budget`
  drop A-table openings; expected K decreases further.
- Tests (future implementation): sidecar parse; missing file; wrong digest;
  wrong length/arity; async determinism; bundle version rejection; A table
  values used in lookups; B commitments unchanged; dynamic claims unchanged.

### 2.5 Security boundaries

- A: trust = digest collision resistance (FlashProve second hash K/TK or
  zkIE's selected domain-separated hash) + the independent model-distribution
  sidecar + the configuration/commitment chain. No prover-supplied hash.
- B: trust = existing PCS commitments (lookup roots, weight batches,
  LookupGroups). B tables never move to digest-only.
- Dynamic B5: trust = producer op's committed cells/claims; do not classify
  as A even if the values happen to look like a function table.

## 3. FlashProve references (local fork 289dd85f)

- `verifier/src/vf_attention_shape.h:206` - softmax exp table (`Tt`).
- `verifier/src/cp_verify.cpp:183` - `sigmoidTableEntry`.
- `verifier/src/cp_verify.cpp:202` - `gateTableEntry` (SiLU/gate).
- `verifier/src/cp_verify.cpp:1418`, `:1426` - table hashing
  (`earlier_hash_h_digest(...,72,...)`; `h.hTs`, `h.hSig`, `h.hRope`).
- `verifier/src/cp_verify.cpp:1609` - RoPE `ropeC`; FE integer recurrence.
- `verifier/src/cp_verify.cpp:1734` - SiLU `tblHeld(0,...)`.
- `verifier/src/cp_verify.cpp:2010` - exp `tblHeld(3,...)`.
- `docs/format/11-1-machinery.md` - *Public tables*: range table row t=t;
  SiLU/sigmoid/rotary/softmax public tables; `tsilu/tsig/trope/texp` digests;
  verifier computes or takes as data and rejects digest mismatch.
- `docs/format/FORMAT.md` - **R15.3** second hash `K(72;table)` / 32-byte
  digest; **R4.9** configuration digest; **R4.11** commitment ID; 12.6
  second hash; 13.4 step 4 / 17.1 item 8-9 / 17.2 item 7 public-table
  handling.
- `docs/format/11-5-weights-in-the-table.md` - committed weight matrix W;
  embedding/head kinds 12/13; expert/router kinds 7-11; weight claims.

## 4. Decision status

Resolved by TL: Gelu (A, load+digest), rsqrt (A, load+digest), digest source
(model-distribution `tables.meta` sidecar), RoPE cos/sin (A, load+digest,
migrated from execution inputs). Open for implementation time: optional
compute paths for rsqrt/Gelu/RoPE after public bit-exact generator specs; the
exact sidecar hash algorithm/domain ids; and any future A-table not listed
here. B tables and dynamic producer claims are unchanged.

## 5. Implementation design (pending boss final review; no code)

Status: design only. The boss's final review must approve the classification
above; if the classification changes, this section is updated first. No code,
API, or proof-wire change has been made by this document.

### 5.a TrustedModelManifest A-registry (schema level)

Proposed schema (names may be adjusted during implementation; the fields and
semantics are the design):

```rust
pub enum ATableKind { Compute, File }

#[derive(Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct ATableEntry {
    pub table_id: T,                 // trusted store tensor id
    pub kind: ATableKind,            // Compute (path 1) or File (path 2)
    pub length: usize,               // logical (unpadded) table length
    pub arity: usize,                // padded 2^arity
    pub file_path: Option<String>,   // relative to the model dir; None for Compute
    pub digest_words: Option<[u32; 8]>, // 32-byte digest, 8 low-first words (FP style)
    pub hash_domain: u32,            // FP K domain 72 or a zkIE domain id
}

pub struct TrustedModelManifest {
    // ... existing fields ...
    #[serde(default)]
    pub a_tables: Vec<ATableEntry>,  // canonical order: ascending table_id
}
```

- `a_tables` is sorted by `table_id`; duplicate ids, unknown ids, a `File`
  entry without a digest, or a kind/length/arity that disagrees with the
  trusted graph reject.
- **Serde compatibility**: the new field uses `#[serde(default)]`. The
  bundle wire version is bumped (5.c), so a v4 bundle is rejected before
  proof decoding; code-built manifests and synthetic tests get an empty
  registry by default. A trusted graph that contains A tables with an empty
  (or partial) registry rejects at load time; it is never treated as "no
  tables".
- `lookup_lengths` remains as trusted shape metadata; `identity_tables`
  remains for closed-form range tables. A ids are removed from
  `lookup_roots` (5.d). The bundle's manifest carries a caller copy of
  `a_tables`; the authoritative digest source is the sidecar (5.c).
- **Absorption position/order**: in `execution_seeds`, after
  `transcript.absorb(&bincode::serialize(model))` and before the execution,
  aux descriptors, aux groups and statement, add a canonical absorption:
  tag `b"zkie-p1-a-tables-v1"`, entry count, then for each entry in ascending
  `table_id`: table id, kind, length, arity, digest words, hash domain. File
  paths are not absorbed (the digest is the security-relevant value; paths
  are confined under the model dir). Every challenge in the reverse DAG comes
  after this position, so the A set is bound into the transcript/config the
  same way FP binds public table digests before its challenges (FP
  `cp_verify.cpp:1418`, `:1426`, `:2010`; FORMAT R15.3/R4.9).
- The verifier first checks that the manifest copy in the bundle equals the
  sidecar registry; a mismatch rejects before proof verification.

### 5.b Verifier load pipeline

Proposed functions (new module, e.g. `crates/zkie-ops/src/tables.rs`):

```rust
pub struct LoadedATables {
    pub entries: BTreeMap<T,ATableEntry>,
    pub values: BTreeMap<T, std::sync::Arc<Vec<Goldilocks>>>,
    pub digests: BTreeMap<T, [u32; 8]>,
}

pub fn load_a_tables(model_dir: &Path, model: &TrustedModelManifest,
                     ops: &[Op]) -> Result<LoadedATables, ATableError>;

fn parse_tables_meta(path: &Path) -> Result<Vec<ATableEntry>, ATableError>;
fn validate_a_entries(ops: &[Op], model: &TrustedModelManifest,
                      entries: &[ATableEntry]) -> Result<(), ATableError>;
fn load_file_entry(entry: &ATableEntry, model_dir: &Path)
                   -> Result<(Vec<Goldilocks>, [u32; 8]), ATableError>;
fn compute_entry(entry: &ATableEntry, model: &TrustedModelManifest)
                 -> Result<(Vec<Goldilocks>, [u32; 8]), ATableError>;
fn hash_a_table(entry: &ATableEntry, values: &[Goldilocks])
                -> Result<[u32; 8], ATableError>;
fn absorb_a_digests(t: &mut Transcript, tables: &LoadedATables);
```

Order (every step fail-closed):

1. Parse `<model_dir>/tables.meta` (default path; parameter-file style, one
   entry per A table: id/kind/length/arity/path/digest words/domain). Reject
   malformed lines, duplicate ids, missing/extra entries.
2. Validate against the trusted graph/model: ids are exactly the A-table ids
   implied by the ops and the classification (exp/rsqrt/Gelu/RoPE-plus any
   future A table), kinds follow the decisions (A1/A11 compute; A3-A6/A10
   file; A12 compute if added), logical lengths and padded arities match
   `lookup_lengths`, and the set-level all-or-none digest rule holds.
3. Load/compute concurrently (rayon or a bounded task pool):
   - `File`: open the path confined under `model_dir`, check byte length and
     shape, stream-hash (BLAKE2b `K(72; table)`/`TK` or the zkIE domain), and
     compare with `digest_words` when the set is in digest-bearing mode.
   - `Compute`: compute the table to the bit from the public formula and
     public parameters (exp: FP `vf_attention_shape.h:206`; SiLU:
     `cp_verify.cpp:202`; sigmoid: `:183`; if possible later, RoPE
     `ropeC` at `:1609` and rsqrt/Gelu after public generator specs), hash
     it, and compare when the set carries digest lines. Compute entries
     always record their computed digest in `LoadedATables.digests` even
     when the sidecar has no line (complete verifier-owned header).
4. Check values length/arity; cache by table id and digest; reject duplicate
   digests for different ids (or record and flag), path escapes, short
   reads, unknown entries.
5. Absorb the A digest set with `absorb_a_digests` before the proof's
   challenges (5.a); then use `LoadedATables.values` in the reverse
   reductions. No A table value or digest is read from the proof.
6. Range/identity tables (A7-A9, A2's absence of a table) are generated by
   the existing closed-form paths; they need no file/sidecar entry (an
   optional generator-version digest may be recorded).

FP correspondence: `verifier/src/vf_attention_shape.h:206` (compute exp),
`verifier/src/cp_verify.cpp:183` (sigmoid), `:202` (SiLU/gate), `:1418`
(SiLU hash `h.hTs`), `:1426` (sigmoid/rope hashes `h.hSig`/`h.hRope`),
`:1609` (ropeC/FE recurrence), `:1734` (`tblHeld(0)`), `:2010`
(`tblHeld(3)`); FORMAT R15.3 for the second hash and 32-byte digest.

### 5.c Bundle v5 migration points

- `crates/zkie-ops/src/bundle.rs`: `BUNDLE_VERSION` 4 -> **5**. `PublicBundle`
  carries the manifest with the caller copy of `a_tables`; the model
  distribution sidecar is authoritative for digests.
- `verify_bundle_dir` order:
  1. read/validate the bundle version; `version != 5` -> reject before
     decoding the proof (v4 and older are rejected; no silent
     reinterpretation);
  2. read the bundle manifest and proof;
  3. resolve the model directory/sidecar (required when the trusted graph
     contains A tables; default filename `tables.meta`). The model directory
     comes from the caller (CLI/example argument or `ZKIE_MODEL_DIR`), never
     from the proof. Missing sidecar for an A-bearing graph rejects;
     absent sidecar is allowed only when the graph has no A entries (e.g.
     only range/identity tables);
  4. compare the bundle manifest's A-registry copy with the sidecar; a
     mismatch rejects (the prover cannot substitute a digest or path);
  5. `load_a_tables` -> load/compute/hash/compare (5.b);
  6. run `execution_seeds` with the A digest set absorbed, then the proof
     verifier; A table values come only from `LoadedATables`.
- Sidecar parsing/defaults: parameter-file style; `File` paths are relative
  to `model_dir` and confined (no absolute paths, no `..` escape); entries
  are canonicalized by id; unknown/duplicate/malformed entries reject. A
  sidecar with any digest line makes the whole A set digest-bearing
  (set-level all-or-none); compute entries without sidecar lines are allowed
  only when the whole set has no digest lines, and their computed digest is
  still recorded in the verifier header.
- Bundle tests (future): v4 reject; missing/malformed sidecar; digest
  mismatch; path escape; id/kind/length/arity mismatch; set-level all-or-none
  violations; async/parallel determinism; B tables unchanged.
- FP correspondence: FORMAT R4.9 (configuration digest), R4.11 (commitment
  ID), R15.3 (second hash/digest lines); 17.1 item 8-9 and 17.2 item 7 for
  parameter-file/table-digest handling; `cp_verify.cpp:1418/1426/2010` for
  verifier-owned digest words.

### 5.d A-table opening removal list

With A tables public (compute or file+digest), the following commitment and
opening paths are removed for A ids; B ids keep their mechanics:

- `from_offline_commitment`: do not insert A ids into `lookup_roots` (keep
  `lookup_lengths` and `identity_tables`); B roots stay. A ids are still
  collected in the A registry (5.a).
- `prepare_static_lookup_commits`: skip A ids (Softmax exp, RmsNorm/LNC/
  legacy rsqrt, Gelu/SiLU; RoPE cos/sin are not in this list today but are A
  after P1). B static tables (embedding/head) and any non-A static table stay
  once-per-proof committed.
- `internal_lookup_prove` / `internal_lookup_verify`: for A ids, use
  `LoadedATables` values (prover from its model files/store, verifier from
  the sidecar-verified values); do not read `static_table_commits`, do not
  emit/verify `table_opening`, do not call `record_table`. Identity tables
  keep their closed-form branch; B internal tables keep the M4a path.
- `build_internal_table_openings` / `AuxOpenRecorder`: no A table claims;
  `internal_table_openings` remains for B internal tables only (possibly
  empty in the current models).
- `static_lookup_groups` / `prepare_lookup_group_aux` / `prove_lookup_groups`
  / verify: exclude A tables (Gelu, SiLU, future A Lookup tables); keep B
  static tables (embedding/head) and their group table opening.
- `public_execution` / reverse claims for RoPE cos/sin: remove A cos/sin ids
  from execution inputs; authenticate RoPE claims against the digest-verified
  A table values (prover uses the same public model files); B inputs and
  dynamic producer claims unchanged.
- `proof_budget` / `opening_upper_bound`: remove A internal-table batch
  openings and A LookupGroups table openings; exclude A ids from
  `internal_lookup_member_counts`. Relu/GeluIndex range bit openings remain
  (they prove the index range witnesses, not the Gelu function table);
  B static/weight/dynamic openings remain.
- Dynamic B5 (`op_field_writes`) and weight batches are untouched.
- Tests to update: A ids absent from `lookup_roots`/`static_table_commits`/
  `LookupGroups`; B ids present; K/budget expectations; digest negative
  tests; v5 bundle tests.

FP correspondence: 11-1 *Public tables* (range/table public, digest lines),
FORMAT R15.3; B tables follow 11-5 *weights in the stored table*.
