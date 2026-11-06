# O7-deep M1 — AuxPlan layout + per-arity global aux commitments

Date: 2026-11-05. Status: implemented (this commit); M2 (reduction
migration + pooled openings) is next. Design: `docs/o7-deep-plan.md`.

## What M1 adds

- `aux_specs(ops, model)`: the single deterministic source of truth for the
  aux descriptor list (op_index, role, len, arity, columns), shared by the
  old per-op validator and the new plan.
- `AuxPlan`: groups specs by arity (ascending); within a group, columns are
  ordered by `(op_index, role_rank, column)`. `plan.slots[(op, role, col)] =
  (group, offset)`; `plan.groups = [(arity, total_columns)]`.
- `AuxGroupDescriptor { arity, columns, commitment, layout_version }` and
  `AUX_LAYOUT_VERSION = 1`.
- `commit_aux_groups[_from]`: concatenates the same prepared aux columns the
  per-op path uses, pads each to `2^arity`, and commits one WHIR batch per
  arity group.
- `validate_aux_plan`: the verifier rebuilds the plan from `ops`/`model`
  and requires exact `(arity, columns, layout_version)` agreement; the group
  commitments are absorbed into `execution_seeds` before any challenge.
- Wire: `BoundExecutionProof` now carries `aux_groups` + `aux_layout`;
  `BUNDLE_VERSION` is 2. Old bundles/proofs (version 1 / missing fields) are
  rejected by the version check or serde; there is no silent reinterpretation.

## Gate evidence

- Ops suite: 202 passed, 0 failed, 0 ignored (`cargo test -p zkie-ops
  --release --lib`).
- Synthetic 6×RmsNorm graph (m=4, d=8, n_real=5):
  - per-op wire: `proof_budget.openings = 79 > MAX_EFFECTIVE_OPENINGS = 64`
    → `verify_committed_shard_dag_bound` rejects;
  - plan: `aux_groups = 2`, total padded group columns = 1818 (arities 5 and
    16, dominated by the 65536-entry identity limb tables);
  - a pooled batch opening of a plan column against its per-arity group
    commitment verifies;
  - `aux_layout` and `aux_groups[0].columns` tampering are rejected.
- The test also prints the measured M1 numbers:
  `O7-M1 synthetic: old_K=79 aux_groups=2 group_cols=1818`.

The synthetic full proof is still rejected because the reductions continue to
use per-op aux openings in M1 (as planned); the **aggregated** group opening
is what verifies. M2 migrates the reductions and deferred claims so the full
proof's K becomes the group/table/weight count.

## M2 status

The M2 gate is met for the synthetic multi-op RmsNorm graph; see
`docs/o7-deep-m2.md` (pooled K=9, full proof verifies). RoPE/SSI/ScaleGate/
Softmax migrations remain in M2b.

## M2 target (original note)

- Replace per-op `open_aux` calls with recorded `AuxOpenClaim { group, col,
  point, value }` + one pooled `EfBatchOpeningProof` per group.
- Keep the per-op virtual proofs unchanged; only the opening section moves.
- Re-run the synthetic graph and require the full proof to verify under the
  64 cap; then the S9 segmented baselines (wire changed again) and the B2
  real-model gate.
