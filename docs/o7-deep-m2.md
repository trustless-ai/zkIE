# O7-deep M2 — deferred pooled aux openings (B1: RmsNorm first)

Date: 2026-11-05. Status: M2 complete for the B1 roles (RmsNorm, RoPE,
StableSoftmaxIndex, ScaleGate, Softmax; Softmax's shared Lookup pairs stay
with M3). Design:
`docs/o7-deep-plan.md`; M1: `docs/o7-deep-m1.md`.

## Mechanism

- `AuxOpenClaim { group, col, point, value }` + `AuxOpenRecorder`: every
  migrated reduction records the exact claims it would have opened, mapped
  through the AuxPlan slot table.
- Claim order is canonicalized by the serialized `(col, point, value)`
  tuple, so the prover's and verifier's reduction-local recording orders do
  not matter; the batched PCS opening binds the ordered list.
- `AuxGroupOpening { group, proof }`: the trailing pooled section of
  `BoundExecutionProof` (`aux_pooled`), one `EfBatchOpeningProof` per group
  with claims. The verifier reconstructs the same claims from its own
  reductions and verifies against the per-arity group commitment (M1)
  reconstructed from `aux_groups`.
- Migrated B1 roles: RmsNorm (row/out + five internal pair/multiplicity
  openings), RoPE (out witness + rem_f/rem_s internal pair/multiplicity),
  StableSoftmaxIndex (the single witness batch), ScaleGate (out + remainder),
  Softmax (out/sum/e/rem; its shared Lookup pairs stay per-op for M3).
  Table openings (rsqrt/silu/identity) stay per-op for M4. Virtual proofs are
  untouched.
- `AUX_LAYOUT_VERSION = 2`, `BUNDLE_VERSION = 3`; M1 proofs/bundles are
  rejected.

## Gate evidence

- Synthetic 6×RmsNorm graph (m=4, d=8, n_real=5), bound path:
  - full pooled proof **verifies** under the 64-opening cap;
  - measured `O7-M2 synthetic: pooled_K=9 aux_groups=2 group_cols=1818`
    (K=9 = 2 aux groups + 6 rsqrt table openings + 1 weight batch);
  - pre-M2 per-op K was 79 for the same graph (rejected).
- Ops suite: **202 passed, 0 failed, 0 ignored**. RmsNorm attack test
  (forged out claim) still green; direct RmsNorm path keeps per-op openings
  (recorder = None) and is unchanged.
- Layout/version tampering still rejected; `aux_pooled` claims are checked
  against the verifier-reconstructed AuxPlan.

## Remaining

- M3: Projection/Scale/Argmax/Lookup/TopK/Scatter/Gather and the Softmax
  shared Lookup pairs. The non-migrated roles keep their per-op aux openings
  and budget charges; the pooled section only carries the migrated B1 claims.
- M4: per-table LogUp aggregation (rsqrt first) removes the 6 rsqrt table
  openings in the synthetic graph and the O(2^23) per-op cost.
- M5: final wire/budget docs, full64/B2 real-model evidence on the pooled
  wire.
