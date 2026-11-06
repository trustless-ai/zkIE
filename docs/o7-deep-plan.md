# O7-deep: pooled aux/table openings for monolithic B2 proofs

Status: M1 implemented (see `docs/o7-deep-m1.md`); M2 next. Original design
note below. Chosen by TL
after the B2 real-model blocker: `security_policy_ok` rejects
`ProofBudget.openings > MAX_EFFECTIVE_OPENINGS = 64`, and a monolithic
Gemma3/DeepSeek proof has hundreds of per-op PCS openings (RmsNorm alone: 9
openings/op x 108+ ops), so it cannot verify at 97 bits.

## 1. Opening inventory today

| source | count | notes |
|---|---|---|
| per-op aux commitments | 1-9 per op | RmsNorm 9, Softmax up to 6, StableSoftmaxIndex 1, Projection/Scale/Argmax 1, Lookup 2-3, TopK/Scatter/Gather several |
| static table openings | 1 per lookup op | every non-identity lookup opens its table at its own leaf point (the commit is cached, the opening is not shared) |
| weight batches | 1 per cached batch | already grouped by size; ~5-20 per model |
| base openings | identity tables / gated paths | small |

The A1 budget counts every opening in `ProofBudget.openings`; full-model
graphs are dominated by the per-op aux and per-lookup table openings.

## 2. Design

### 2.1 Per-arity global aux commitments

Replace per-op aux commitments with a deterministic global layout derived
from the trusted graph and manifest:

- Compute every aux descriptor exactly as today (`*_aux_specs`), then group
  descriptors by `(arity)`. Within a group, concatenate their columns in
  ascending `(op_index, role_rank, column)` order; all columns are padded to
  `2^arity`.
- Commit each group once with `Whir::commit_batch`; the proof carries one
  `AuxGroupDescriptor { arity, columns, commitment, layout_version }` per
  group (first-pass arrays at arities 5/…/16/23; ~4 groups per model).
- A deterministic `AuxPlan` maps `(op_index, role, col) -> group + offset`.
  The verifier rebuilds the same plan from `ops`/`model` and validates the
  group descriptors against it (the existing `validate_aux_descriptors`
  generalizes to a layout check). Roots are absorbed by `execution_seeds`
  exactly as today.
- Each reduction keeps its column slices but opens them against the group
  commitment/offset instead of a per-op commitment.

### 2.2 Deferred pooled openings

PCS opening proofs do not feed the transcript (all challenges are derived
from the virtual sumchecks/absorptions that already precede them), so
openings can be deferred without changing Fiat-Shamir order:

- During the reverse DAG, each reduction records
  `AuxOpenClaim { group, col, point, value }` and returns no opening proof.
- A trailing proof component holds one `EfBatchOpeningProof` per aux group,
  covering all recorded `(column, point, value)` claims. The PCS batch API
  already supports repeated tables at different points (O5 `claim_groups`).
- The verifier replays the same claims from its own reductions and checks the
  pooled opening per group. Weight claims keep the existing per-batch
  grouping; static lookup tables get one pooled opening per trusted table
  (one table index repeated at every per-op leaf point) instead of one per
  lookup op.
- The proof's trailing sentinel step (`LookupGroups`-style) carries the
  pooled aux/table opening sections; the per-op steps carry only their virtual
  proofs and claim lists (already the case for aux values).

### 2.3 Per-table LogUp aggregation (performance)

For every trusted static table shared by multiple lookups (rsqrt table across
all RmsNorm ops; exp/gelu/silu; limb identity tables are closed-form):

- Derive one shared `(alpha, beta)` per table after the aux absorption (the
  existing `derive_lookup_group_ab` pattern).
- Commit one aggregate multiplicity column `M = Σ_op m_op` per table per
  proof (2^23 for the rsqrt table, instead of one per RmsNorm op) and build
  the table-side fraction tree once.
- Each member proves its witness fraction against the shared table tree via
  the existing group identity `Σ_op m_op(q) = M(q)`; the padding-zero
  virtual stays per member.
- This removes the O(RSQRT_TABLE_LEN) table work and the per-op multiplicity
  commitment from every RmsNorm op (the dominant B2 cost), and removes the
  per-op table opening.

### 2.4 Budget

After pooling, `ProofBudget.openings` = aux groups (~4) + non-identity static
tables (~3-5) + weight batches (~5-20) + closed-form/gated base openings
(few). Gemma3/DeepSeek estimate: K ≈ 25-40, under the 64 cap at 97-bit
per-opening security. `weighted`/`merge` accounting is unchanged per
virtual/claim; the pooled batch openings charge their claims' merge units as
today.

## 3. Wire/API changes

- `BoundExecutionProof.aux: Vec<AuxDescriptor>` becomes
  `aux_groups: Vec<AuxGroupDescriptor>` (per-arity commitments) plus the
  deterministic plan; version bump and explicit rejection of the old layout
  (no silent reinterpretation).
- Each reduction's aux fields become `(group, offset, columns)` slices; the
  public `PreparedAux`/`AuxDescriptor` types are replaced by the plan-aware
  variants inside the crate. Bundle format version bumps accordingly.
- The per-op `Openings` fields are removed from reduction proofs and replaced
  by the pooled sections; proof bytes should shrink (one opening per group
  instead of per op).

## 4. Staged implementation

1. M1: layout planner (`AuxPlan`), per-arity commits, descriptor validation,
   bundle/aux wire version bump; no reduction migrated yet (tests pin the
   plan).
2. M2: migrate the B1 aux roles (RmsNorm/RoPE/StableSoftmaxIndex/Softmax/
   ScaleGate) to plan slices + deferred claims; pooled aux opening section.
3. M3: migrate the remaining aux roles (Projection/Scale/Argmax/Lookup/
   TopKSelect/Scatter/Gather/…).
4. M4: per-table LogUp aggregation for shared static tables (rsqrt first).
5. M5: budget/wire finalization, full-model Gemma3/DeepSeek runs with the
   independent verifier, evidence and docs.

Each stage is an independent commit with the full ops suite plus a synthetic
many-op graph that crosses the 64-opening line today and must verify after
pooling. The old monolithic wire is rejected after M1 (version bump); the
segmented S9 proofs remain reproducible from their tagged commits.

## 5. Security notes

- The aux group commitments are absorbed before any challenge exactly like the
  per-op commitments, so the deferred opening proofs cannot influence the
  transcript.
- The verifier derives the aux plan (shapes, roles, lengths) from the trusted
  graph; a proof cannot add/move/retype columns.
- Pooling changes only how openings are batched; each individual claim is
  still checked at its transcript-derived point with its reduction-derived
  value, so the per-opening 97-bit WHIR security and the A1 aggregate budget
  semantics are preserved.

Estimated effort: M1-M2 are the critical path; M3 is mechanical but touches
every reduction; M4 removes the RmsNorm cost explosion. Requesting TL
approval of this plan/wire change before M1 starts.

## 6. Hard-gate audit: do PCS openings touch the main transcript?

Result: **no**. Checked line by line:

- `crates/zkie-core/src/pcs/ef_batch_open.rs`
  - `prove_batch_open_committed_ef` (`:40`) and
    `verify_batch_open_committed_ef` (`:109`) take **no transcript
    argument**. They construct a local `Transcript::new(b"zkie-batch-open")`
    (`:73`, `:132`) and absorb only the commitment, the claim points and the
    claimed values into it. The main reverse-DAG transcript is not passed in
    and cannot be mutated.
  - The PCS call itself (`whir.open_batch_multi_ef`) receives the local
    challenges, not the main transcript.
- `crates/zkie-core/src/pcs/whir.rs`
  - The file has no `Transcript` type at all (`grep Transcript whir.rs` is
    empty). `open_ef` (`:759`), `verify_ef` (`:781`),
    `open_batch_multi_ef` (`:932`) and `verify_batch_multi_ef` (`:953`) use
    `fresh_challenger()` (`:454`), a fresh WHIR challenger seeded by the PCS
    domain separator. The commitment/opening Fiat-Shamir is entirely inside
    the PCS.
- In `compose.rs`, the main transcript is only advanced by the virtual
  sumchecks, `same_poly`, and the explicit `absorb_*` calls before them.
  Every opening call (`open_aux`, `prove_batch_open_committed_ef`,
  `verify_aux_open`, table/weight openings) is transcript-free.

Therefore deferring the opening proofs to a trailing pooled section does not
change any challenge derivation: the reduction's points/values are already
fixed by the preceding virtual proofs/absorptions, and the pooled opening is a
transcript-independent PCS proof over those fixed claims. The batch-opening
local transcript still binds the commitment + points + claimed values, so the
deferred proof cannot be re-pointed after the fact.

Conclusion: the deferral pillar holds; no mirror absorption is needed. This
audit is the M2 pre-gate and is recorded here as required by the TL.
