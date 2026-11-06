# O6 continuation record — split witness/proof work; reuse Projection matmul

State: `0eec859` + the committed-split completion on `fix/soundness-repair`
(test machine `/data/zkie-soundness-fix`). The earlier O6 Projection cache
(`0cc1f50`) is included below; the committed-path split and its measurements
are the completion pass. O6 awaits TL review.

## Inventory: where the gist's pattern actually lives

- The "run a plain proof first, then re-prove the fractional sections with
  commitment-derived challenges" pattern is in the **retired committed path**:
  `prove_committed_shard_dag_inner` calls
  `prove_shard_dag_inner(..., FracMode::Committed)`, whose permutation/gather
  fractional proofs run on scratch transcripts and are re-proven later on the
  committed transcripts (comments in `prove_shard_precomputed`). The committed
  verifier permanently rejects (`verify_committed_shard_dag` is retired), so no
  accepted proof exercises that path. The first pass left it untouched; the
  completion pass below performs the dedicated split.
- The **active bound path** already has the intended split: forward/aux witness
  construction and commitments happen first, then the transcript/challenges are
  derived, then the reverse DAG proof is produced. It does not build a plain
  proof and discard it.
- The remaining O6 redundancy on the active path is the Projection proof:
  the forward computes `h = x @ w` to produce `out`/`rem` and then drops it,
  and `prove_reverse_projection` recomputed the same product; the weight was
  also materialized twice.

## Implementation (active bound path)

- `Store` has a bounded per-store cache of the forward Projection product
  `h = x @ w`, keyed by the Projection output tensor id, with a byte budget.
  The bound prover enables it (default 256 MB; `ZKIE_O6_CACHE_MB=0` disables it
  for baseline measurement or memory pressure). The forward moves `h` into the
  cache without copying when the budget allows; otherwise it is dropped and
  the proof recomputes it (controlled fallback).
- `prove_reverse_projection` reuses the cached `h` when present (one cache
  hit), otherwise recomputes it once; the weight is materialized exactly once
  and shared with the matmul proof.
- Per-store metrics count forward matmul calls, reverse-proof recomputes and
  cache hits, plus cached tensors/bytes. These are measurement counters only;
  no proof value, transcript message or authentication relation changes.
- No all-model activation residency is introduced: only Projection `h`
  buffers are cached, bounded by the byte budget.

## Measurements

`crates/zkie-ops/examples/o6_projection.rs`, one real Projection
`m=256, k=1024, n=256, shift=8` with bounded fixed-point inputs (release):

| mode | fwd mm | proof mm | hits | cached | bytes | wall | proof bytes / hash |
|---|---|---|---|---|---|---|---|
| `ZKIE_O6_CACHE_MB=0` | 1 | 1 | 0 | 0 | 0 | 147.9-157.0 ms | 275,815 / `9ce6b98fa9199322` |
| default 256 MB | 1 | 0 | 1 | 1 | 524,288 | 153.8-155.0 ms | 275,815 / `9ce6b98fa9199322` |

The repeated matmul product is eliminated (two computations -> one). The wall
time is dominated by the Projection aux/WHIR proof at these sizes and the
difference is within run-to-run host noise; the metric that matters is the
single computation and the identical proof bytes. Cached `h` adds 512 KiB,
within the budget; VmHWM stayed ~49 MB in both runs.

These figures are from the original `0cc1f50` pass. After the intervening O2/
O5/O7 protocol changes, the same example at `0eec859`+this pass reports
385,047 B / `b92348c847b6b214` in both cache modes (see the completion pass
below); the cache-on/off identity is the invariant.

## Regression

- New `o6_projection_h_cache_reuses_forward_product`: the bound prover computes
  the forward product once (`fwd=1`), the proof reuses it (`recompute=0`,
  `hit=1`), the cache holds one tensor, and the proof verifies.
- `zkie-ops --lib`: 159/0 (158 + the new O6 test); all existing negative tests
  remain green (wrong boundary/claims, tampered witnesses, retired committed
  verifier rejection).
- Real segmented A/B/C: proof bytes identical to the pre-O6 dumps
  (241,874 / 462,836 / 355,183 B); verify 15.5 ms (C) / 29.4 ms (B). The real
  chain has no Projection ops, so O6 does not alter it.
- Real full64: `CORE-CHAIN-OK`; 64 claims, one `W_t` opening; B 510,969 B;
  verify 154.8 ms.
- No transcript/authentication/security change; proof bytes unchanged; 90-bit.

## Completion pass — committed witness/proof split

The earlier pass removed the active bound path's duplicate Projection product.
This pass removes the remaining discarded work in the retired committed path
and applies the same ordered split there:

```
build witness columns
  -> commit alpha/gamma-independent columns, absorb roots
  -> derive alpha/gamma per op
  -> build and commit the dependent num/den columns, absorb roots
  -> generate each final fractional proof / product sumcheck exactly once
```

- Scatter: the committed-mode shard stage now returns `permutation_witness`
  (`a`/`b` columns only). It creates no scratch transcript and no
  `prove_permutation`; the committed perm section commits `a`/`b`, derives
  alpha, builds num/den, and proves the two sides once.
- Gather: the committed-mode shard stage keeps only
  `weighted/gate/out/token_id`. It does not derive a scratch alpha/gamma, does
  not build num/den, and does not run either product sumcheck. The committed
  gather section commits the independent columns, derives alpha/gamma, builds
  and commits num/den, then runs the two fractional proofs and the two product
  sumchecks exactly once on the committed transcript. The product sumchecks
  therefore consume the committed gamma/num columns, not early-challenge
  columns.
- Lookup/table side: the committed-mode shard stage builds only the witness
  `x`/`y`/`num` columns and the multiplicity `m`; it no longer squeezes
  alpha/beta on the shard transcript and no longer generates the witness or
  table fractional proof. The committed lookup section absorbs the committed
  witness/table roots, derives alpha/beta, builds the den columns, and
  generates the witness-side and table-side proofs exactly once.
- The committed columns are moved into the batch openings and are not kept in
  the serialized proof; the prover-side copies are dropped before the openings.
- `Store::o6_split_metrics()` reports `(perm fracs, gather fracs, gather
  products, lookup-witness fracs, lookup-table fracs, discarded fracs,
  discarded ns)`. A committed proof resets the counters at its start. The
  discarded counters are zero on the honest path, and the regression tests
  assert this.
- Projection `h = x @ w` reuse now covers the committed shard path too
  (`prove_projection_with_h`). `enable_projection_cache` starts a fresh shard
  lifetime (drops retained products), and a re-run forward refreshes an
  existing entry, so a reused Store cannot feed a stale product into a proof.
  The budget/fallback contract is unchanged (`ZKIE_O6_CACHE_MB`, default
  256 MB; `0` disables).

### Measurements

Release build on the test machine. One Scatter (`m=1024, n=1024, k=4`,
`s=4096`) and one Gather (`s=4096, h=64`, source domain 2^18, target domain
2^16), direct committed-mode construction and end-to-end
`prove_committed_shard_dag`:

| metric | baseline `0eec859` | this pass |
|---|---|---|
| direct Scatter committed construction | 30.3 ms | 1.05-1.37 ms |
| direct Gather committed construction | 131.9 ms | 1.02-1.11 ms |
| end-to-end committed proof | 422.2 ms | 337.5-357.1 ms |
| proof bytes | 834,170 | 832,954 |
| proof hash | `9df6fb008ca8e911` | `7aee62a5ed351339` |
| RSS after proof | 98,924 KiB | 62,940-62,968 KiB (uncontended runs) |
| VmHWM | 102,424 KiB | 82,116-83,748 KiB (uncontended runs) |

Baseline internal instrumentation (temporary, parent tree only):
direct Scatter scratch permutation 29.3 ms; direct Gather scratch fractional
proofs 113.6 ms and early product sumchecks 15.5 ms. On the end-to-end path the
same scratch work was 15.0 ms (permutation) + 79.0 ms (gather fractional) +
11.7 ms (early products). Per Scatter+Gather, the baseline generated 8
fractional subtasks (2 Scratch perm + 2 Scratch gather, then the same 2+2
final subtasks) and 2 early product sumchecks. The split generates the final
2 perm + 2 gather fractional subtasks and 2 product sumchecks once, and
discards nothing:

```
O6-SPLIT-METRICS perm_fracs=2 gather_fracs=2 gather_products=2 lookup_witness_fracs=0 lookup_table_fracs=0 discarded=0 discarded_ns=0
```

Lookup/table side (`idx_len=65536`, table length 4096, same example): the old
shard stage's direct witness-side fractional proof costs 31.8-32.3 ms and the
table-side proof 10.5-14.1 ms; both were generated there and generated again
in the committed section. A committed proof with one Lookup now reports
`lookup_witness_fracs=1 lookup_table_fracs=1 discarded=0`; the two proofs are
generated only after the committed roots, once each.

The proof bytes/hash change is confined to the legacy committed format (the
S1 protocol version changes when the lookup alpha/beta and gather product
challenges move to the committed transcript). The committed verifier remains
permanently closed; no accepted proof uses this format. The active bound path
is unchanged.

Projection cross-check (`o6_projection`, same machine): cache-on and cache-off
produce the same proof, 385,047 B / `b92348c847b6b214`; cache-on
`fwd_mm=1 proof_mm=0 hits=1 cached=1` (524,288 B), cache-off
`fwd_mm=1 proof_mm=1 hits=0 cached=0`; VmHWM 78,808 KiB on / 76,272 KiB off
(the retained 512 KiB product plus allocator pages).

### Regression

- `cargo build --workspace --release` passes.
- `zkie-ops` release: 237 passed / 0 failed (234 prior + the three new O6
  split tests: scatter/gather split, lookup split, committed Projection
  reuse); the malicious/negative tests remain green.
- `zkie-core` release: 94 passed / 0 failed / 1 ignored (837.67 s), full suite
  including the lookup fractional tests after the witness-only refactor.

## Remaining

O6 awaits TL review. O7 is not started; S7 is not accepted; S8/S9 remain
postponed. The retired committed verifier stays permanently closed; this pass
changes only proof generation in that non-deployed path.
