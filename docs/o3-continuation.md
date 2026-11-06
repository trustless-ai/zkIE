# O3 continuation record — virtual sumcheck allocation removal

State: `a8900ad` + this O3 change on `fix/soundness-repair` (test machine
`/data/zkie-soundness-fix`). O3 is NOT complete pending TL review.

Update 2026-11-06: a second pass removes the remaining per-round fold
accumulator allocations by reusing one cache-line-padded scratch buffer per
proof call; see the final section of this file for implementation, small-case
and real-B evidence.

## Baseline (gist O3)

`prove_virtual`'s round loop built the round polynomial with a parallel
`map` that allocated one `Vec<Goldilocks>` (length `max_deg+1`) per element
pair, `half` times per round, then `reduce`d the vectors. The same pattern
was in `prove_virtual_with_challenges`. A dedicated measurement example
(`crates/zkie-core/examples/o3_virtual_alloc.rs`, global counting allocator,
`n = 2^18`, 3 MLEs, terms of degrees 1/2/3, release) measured the pre-O3
reference copy:

| threads | allocations | allocated bytes | time |
|---|---|---|---|
| 64 | 384,535 | 18.60 MB | 29.63 ms |
| 1 | 262,278 | 14.69 MB | 37.89 ms |

## Implementation

- `map + reduce` → Rayon `fold` with one reused accumulator
  (`Vec<Goldilocks>` of length `max_deg+1`) per worker/folder, then the same
  `reduce`.
- `.with_min_len(half / (threads * 4))` bounds the folder count so the
  optimization is not lost to per-split accumulator allocation on many cores.
- Applied to both `prove_virtual` and `prove_virtual_with_challenges`.
- Field addition is exact and associative; each element's contribution is
  still accumulated into the same per-`k` slots; the round messages are
  absorbed in the same order and the proof bytes are unchanged.

## Measurements after

| threads | allocations | allocated bytes | time |
|---|---|---|---|
| 64 | 5,637 | 6.47 MB | 7.82 ms |
| 1 | 141 | 6.30 MB | 33.93 ms |

At 64 threads: ~68x fewer allocations and ~3.8x faster in the microbenchmark.
The proof is 760 bytes with hash `bafebfd2fedfcd97` for optimized vs
reference and for 1 vs 64 threads (field-exact, thread-count independent).

## Regression

- `virtual_tests` 4/4, including
  `virtual_fold_matches_map_reduce_reference`: round polys, final evals,
  challenges and serialized proof bytes equal to the pre-O3 reference (both
  transcript and fixed-challenge paths).
- `zkie-ops --lib`: 158/0.
- `zkie-core --lib` (release, `--skip whir_prescribed`): 70/0, 1 ignored.
  The pow-32 `whir_prescribed` test exercises the deterministic PoW grind, not
  this sumcheck change; it was green in 836 s on `a8900ad`.
- Real A/B/C proof bytes are identical to the pre-O3 dumps (A 241,874; B
  462,836; C 355,183 B), so O3 changes neither proof bytes nor size.
- Real full64: `claims=64 unique=1 moved_bytes=2,147,483,648 cloned_bytes=0`;
  `CORE-CHAIN-OK`; B proof 510,969 B; verify 154.7 ms. (B prove wall time is
  WHIR-opening dominated and host-load dependent; O3's microbenchmark gain is
  not expected to move it materially.)
- No transcript/authentication/security-relation change; security 90-bit.

## Hygiene

Removed the accidentally vendored OMC session state
(`.omc/state/hud-state.json`, `hud-stdin-cache.json`) and registry metadata
(`.cargo-ok`, `.cargo_vcs_info.json`) from the vendored p3-challenger; the
vendor directory now contains only the upstream source, README/CHANGELOG and
our grind patch. Added `.omc/` to `.gitignore`.

## Remaining

O3 awaits TL review. O4/O6/O7 are not started; S7 is not accepted; S8/S9
remain postponed.

## Reused chunk accumulators (2026-11-06)

The first O3 pass (`7597a0a`) replaced the per-element `map` allocation with a
Rayon `fold` accumulator per worker/folder. This pass removes the remaining
per-round per-folder accumulator allocations as well:

- `prove_virtual` / `prove_virtual_with_challenges` allocate one `scratch`
  buffer per proof call (`threads * 4` slots). Each slot is padded to 16
  `Goldilocks` (128 B) to keep chunk accumulators off shared cache lines; the
  first `max_deg + 1` fields form the accumulator. Chunks are reduced in order
  into the round's `pvals`; `scratch` is zeroed and reused every round, so the
  round loop allocates no per-worker vectors.
- The MLE folding buffers (`bufs`) are still allocated once and folded /
  truncated in place; the per-round `pvals` Vec remains because it is proof
  data (absorbed in the unchanged order).
- Exact field arithmetic; same coefficients, transcript message order, proof
  bytes and security relation.

### Small case (n = 2^18, 3 MLEs, terms of degrees 1/2/3, release)

Allocation counts are from the example's counting global allocator; CPU is
process user+system time (10 ms `CLK_TCK` resolution). RSS is `/proc` VmRSS
sampled before/after the call (not a peak).

| threads | implementation | allocs | alloc bytes | wall ms | CPU ms | RSS before -> after KiB |
|---:|---|---:|---:|---:|---:|---:|
| 1 | pre-O3 map | 262,278 | 14,686,288 | 40.44 | 40 | 16,820 -> 12,856 |
| 1 | prior O3 fold | 141 | 6,296,416 | 34.27 | 40 | 14,516 -> 16,820 |
| 1 | new scratch | 26 | 6,293,216 | 33.51 | 30 | 14,576 -> 16,820 |
| 8 | pre-O3 map | 270,286 | 14,942,544 | 19.04 | 140 | 17,172 -> 13,228 |
| 8 | prior O3 fold | 901 | 6,320,736 | 7.06 | 30 | 14,996 -> 17,172 |
| 8 | new scratch | 26 | 6,296,800 | 8.31 | 40 | 8,904 -> 9,100 |
| 64 | pre-O3 map | 388,559 | 18,727,280 | 29.38 | 1,680 | 11,816 -> 18,000 |
| 64 | prior O3 fold | 5,637 | 6,472,288 | 6.26 | 210 | 11,684 -> 11,816 |
| 64 | new scratch | 26 | 6,325,472 | 6.94 | 200 | 11,604 -> 17,784 |

New scratch vs pre-O3 map: 10k-15k x fewer allocations; vs prior O3 fold:
5.4x/34.7x/217x fewer at 1/8/64 threads. Wall time is comparable (within
run-to-run noise; at 64 threads the new path is 6.94 ms vs 6.26 ms in this
sample, while repeated runs ranged around 7.0 vs 7.4 ms). The change targets
allocator pressure, not field arithmetic. The small case is in-memory, so
page-cache cold/warm does not apply; the real-B table below is the cold/warm
evidence.

All rows produce `proof_len=760` and `proof_hash=bafebfd2fedfcd97`; the
optimized and pre-O3 reference proofs and the 1/8/64-thread proofs are
byte-identical.

### Real B proof (layer 10, 64-expert gate FFN)

Real boundary weights and routing; proof-only snapshot (commit excluded), so
`allocs`/`cpu_ms`/RSS cover `prove_committed_shard_dag_bound` only. Wall and
whole-process I/O are from `o5_measure.py`.

| run | cache | prove ms | CPU ms | allocs | alloc bytes | RSS before -> after KiB | HWM KiB | proof bytes | verify ms | wall s | maxrss KiB | minflt | majflt | read bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prior O3 | cold | 123,140.6 | 6,270,990 | 88,551,738 | 56,986,471,296 | 1,123,704 -> 1,139,808 | 3,837,924 | 3,081,547 | 180.771 | 134.244 | 3,837,924 | 239,290 | 8,275 | 1,078,267,904 |
| prior O3 | warm | 120,910.2 | 6,270,600 | 88,551,741 | 56,986,475,087 | 1,123,732 -> 1,144,012 | 3,837,980 | 3,081,547 | 182.259 | 131.136 | 3,837,980 | 239,222 | 0 | 0 |
| new scratch | cold | 122,879.7 | 6,251,680 | 88,551,719 | 56,986,446,609 | 1,123,332 -> 1,143,956 | 3,838,440 | 3,081,547 | 181.197 | 134.266 | 3,838,440 | 236,628 | 8,210 | 1,078,263,808 |
| new scratch | warm | 122,712.5 | 6,248,820 | 88,551,735 | 56,986,467,249 | 1,123,396 -> 1,144,060 | 3,838,236 | 3,081,547 | 180.666 | 132.936 | 3,838,236 | 290,522 | 0 | 0 |

The B path does not exercise the base-field `sumcheck::prove_virtual`: its
Lookup/GatherMatMul/BroadcastScale reductions use the EF virtual sumcheck
(`ef_outer`), and Scale uses direct evaluations. Whole-B allocation count is
therefore dominated by other code and is unchanged (within 19 allocs,
-0.31% CPU, +0.02% wall), with identical proof bytes, RSS and verification.
This is a no-regression check, not a place where the base-sumcheck allocation
change can show up. The EF virtual sumcheck has its own allocation profile and
was outside this task's scope. The cold/warm split confirms the cache is not
masking anything: warm has `read_bytes=0` and `majflt=0`.

### Regression

- `virtual_fold_matches_map_reduce_reference` (zkie-core) passes; it compares
  the serialized proof (round coefficients, final evals, challenges) against
  the pre-O3 map/reduce reference for both transcript and fixed-challenge
  paths.
- Proof bytes/hash are identical at 1, 8 and 64 threads (see tables).
- `zkie-ops` release: 234 passed, 0 failed.
- `zkie-core` release full suite (including the 32-bit PoW test): 92 passed,
  0 failed, 1 ignored (837.6 s). The new `virtual_thread_count_outputs_identical`
  test is included.
- No transcript order, proof size, degree/relation, or security change.
