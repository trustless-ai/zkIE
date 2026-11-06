# O4 continuation record — closed-form broadcast eq in the batched verifier

State: `7597a0a` + this O4 change on `fix/soundness-repair` (test machine
`/data/zkie-soundness-fix`). O4 is NOT complete pending TL review.

Update 2026-11-06: the verifier now uses the scalar `eq_poly` closed form
without allocating `eq_u`; padded-domain and GatherMatMul tests were added.
See the final section.

## Problem (gist O4)

`batched_matmul::verify` rebuilt the public broadcast selector
`eq_b[pos*K + d] = eq_u[pos]` as an `S x K` vector and evaluated its MLE at
the sumcheck challenge. The column depends only on position, not on the
contraction index `d`, so the expansion and the O(S*K) evaluation are
redundant verifier work.

## Implementation

- For the row-major layout `index = pos*K + d` with power-of-two `S` and `K`,
  the MLE of the broadcast selector at `ch` is
  `eq_u(ch[logK..])` — the low `logK` challenge variables multiply a constant
  with respect to `d` and factor out. `verify` now computes this closed form
  directly; no `S*K` buffer is allocated.
- Verifier guards: non-power-of-two `S`/`K` and a wrong `u` length fail
  closed; the challenge length is checked before slicing. Padding remains the
  caller's responsibility (zeroed/masked `seg` domain), as in the GatherMatMul
  path; coordinates are never mechanically truncated.
- `prove` is unchanged: the prover still materializes the public `eq_b`
  column for the virtual sumcheck. Only the verifier-side public structural
  value is optimized; witness evaluations are untouched.
- No transcript/authentication relation change.

## Measurements

Dedicated example `crates/zkie-core/examples/o4_verify_alloc.rs` (counting
allocator, honest proof, `S=512, K=2048, N=8`, release):

| verifier | result | allocations | allocated bytes | time |
|---|---|---|---|---|
| pre-O4 reference (`S*K` expansion) | ok | 9 | 16.78 MB | 4.364 ms |
| current (closed form) | ok | 8 | 8,488 B | 0.039 ms |

Same verification result and challenges. The allocation count is essentially
unchanged; the large allocation and its O(S*K) evaluation are removed.

Real segmented A/B/C (`S_pad=8, K=2048`, so `S*K=16K`): proof bytes identical
to the pre-O4 dumps (A 241,874; B 462,836; C 355,183 B); verify times ~15.6 ms
(C) and ~29.7 ms (B), dominated by other work. Full64: `CORE-CHAIN-OK`,
B 510,969 B, verify 153.9 ms (pre-O4 154.7 ms), 64 claims / one `W_t` opening.

## Regression

- `o4_tests::broadcast_eq_closed_form_matches_expansion`: closed form vs the
  full expansion for `(S,K) = (1,1), (2,4), (8,16), (32,64), (4,1), (1,8)` at
  random points.
- `o4_tests::verify_rejects_non_power_of_two_shapes`: fail closed for S=3 or
  K=3.
- `streaming` tests 3/3; `zkie-ops --lib` 158/0; `zkie-core --lib` release
  72/0, 1 ignored (pow-32 `whir_prescribed` skipped as grind-only; green on
  `a8900ad`).
- Real A/B/C proofs byte-identical; full64 `CORE-CHAIN-OK`.
- No transcript/security change; proof size unchanged; security 90-bit.

## Remaining

O4 awaits TL review. O6/O7 are not started; S7 is not accepted; S8/S9 remain
postponed.

## Scalar closed-form pass (2026-11-06)

The first O4 pass (`d8ec50c`) removed the `S*K` broadcast expansion but still
allocated `eq_u` (length `S`) and evaluated its MLE at `ch[logK..]`, i.e. O(S)
verifier work. This pass uses the scalar equality polynomial directly:

- `verify` computes `eq_final = mle::eq_poly(u, &ch[k_bits..])`; it never
  builds `eq_u` or the `S*K` column. The row-major layout
  `index = pos*K + d` with power-of-two `S,K` makes the low `logK` challenge
  variables multiply the constant `d` block, so the broadcast selector's MLE is
  exactly `eq(u, ch[logK..])`.
- Padding is represented by the caller's masked/zeroed power-of-two domain:
  a test embeds 6 real rows in `S=8`, verifies the proof, and checks the
  closed form factors through the extra valid bits rather than truncating the
  challenge to the six-real-bit prefix. Un-padded non-power-of-two `S`/`K`
  still fail closed; no coordinate is mechanically dropped.
- The GatherMatMul verifier reaches the same closed form through
  `verify_gather_matmul -> verify_streaming -> verify`; a dedicated test
  checks its derived challenge against the full `S*K` expansion.
- Prover and transcript are unchanged; only the verifier's public structural
  evaluation changes.

### Large verifier example

`crates/zkie-core/examples/o4_verify_alloc.rs`, `S=512, K=2048, N=8`, release,
5 runs (median reported; allocation counts are stable):

| verifier | result | allocations | allocated bytes | time |
|---|---|---|---:|---:|
| pre-O4 reference (`S*K` expansion) | ok | 9 | 16,781,608 | 4.41 ms |
| first O4 pass (eq_u vector + MLE eval) | ok | 8 | 8,488 B | 0.039 ms |
| scalar closed form (this pass) | ok | 6 | 296 B | 0.026 ms |

Proof length is 848 B with hash `696b42b19f930f0e`; both verifier paths return
the same result and challenges. The scalar form removes the last O(S)
allocation/work from the selector check; the remaining 296 B are the verifier's
ordinary small proof/transcript buffers.

### Regression

- `o4_tests::broadcast_eq_closed_form_matches_expansion`:
  `(S,K)=(1,1),(2,4),(8,16),(32,64),(4,1),(1,8)` random points, scalar closed
  form vs full expansion.
- `o4_tests::closed_form_handles_zero_padded_rows`: 6 real rows in `S=8`
  zero-padded domain, honest proof verifies, closed form equals the full
  expansion and factors through the padding bits.
- `o4_tests::verify_rejects_non_power_of_two_shapes`: un-padded non-power
  shapes fail closed.
- `gather_verify_uses_closed_form_broadcast_eq`: GatherMatMul verifier path and
  selector equality against the full expansion.
- `zkie-ops` release: 234 passed, 0 failed.
- `zkie-core` release full suite: 94 passed, 0 failed, 1 ignored (837.5 s).
- Real 64-expert B proof under the new verifier: `verify=true`, proof bytes
  3,081,547 unchanged, verify 181.9 ms; proof generation/transcript untouched.
- No transcript/authentication/proof-size/security change.
