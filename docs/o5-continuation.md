# O5 continuation record (W_t grouping + streaming Gather)

State: `3e49e59` + this O5 step on `fix/soundness-repair` (test machine
`/data/zkie-soundness-fix`, remote origin trustless-ai/zkIE). S7 is NOT
accepted. No security/transcript change is introduced by O5.

## Landed in this step (A + B)

### A. Combined equality W_t grouping (batch opening)

- `claim_groups(table_idxs)` (crates/zkie-core/src/pcs/batch_open.rs): unique
  tables in order of first appearance plus each claim's group position, shared
  by prover and verifier.
- Prover `prove_batch_open_committed_with_data`: one combined equality weight
  per unique table,
  `W_t(X) = sum_{i: table_i = t} gamma^i * eq(r_i, X)`,
  built by adding scaled eq vectors; `prove_sum_of_products_owned` runs over
  one polynomial pair per unique table. The global gamma powers, `claimed_sum`,
  and the Fiat-Shamir absorption order are unchanged (gamma powers stay global
  over the original claim order; only the sumcheck polynomial count changes).
- Verifier `verify_batch_open_committed`: derives the identical grouping from
  the public `table_idxs` and checks
  `b_eval == sum_g W_g(r) * evals[unique[g]]`, i.e. the same randomized
  gamma-combination of the same claims against the same committed tables.
  The WHIR multi-open still opens every batch table at the fresh point, so the
  unique-table indices are the batch indices. Added fail-closed checks for
  per-claim point arity and out-of-range table indices (previously a panic).
- New `prove_batch_open_committed_unique_owned` takes exactly one owned tensor
  per unique table (first-appearance order) and MOVES it; the legacy
  per-claim entry points keep their signatures and dedupe internally.
- `prove_sum_of_products_owned` (sumcheck.rs): the folding pass takes the
  polynomial buffers by value, so a multi-GB table is not cloned inside the
  sumcheck.
- `mle::eq_evals`: O(2^d) doubling construction instead of O(d*2^d); the
  values are bit-identical to the direct product (dedicated test), so no
  proof/transcript change.
- Bound weight-opening loop (compose.rs): groups claims by table, moves each
  unique padded buffer, drops unreferenced tables before opening, and prints
  `O5-metrics batch=.. claims=.. unique=.. moved_bytes=.. cloned_bytes=0
  tables=..`.
- Tests: `same_table_multi_claim_grouping` (grouped and per-claim provers
  agree on sumcheck + evals; tampered claimed value, tampered table index,
  out-of-range index, tampered b_eval rejected); all existing batch-opening
  tests stay green.

### B. Streaming GatherMatMul

- `WeightBlocks::{Slice, Mmap}` (batched_matmul.rs) + `Store::weight_blocks`:
  a shared owned/mmap handle that yields one `K x N` expert block at a time.
  Owned data is borrowed; mmap data converts one block per expert and frees it
  immediately, so the whole expert-major W_all is never materialized for
  evaluation.
- `stream_weights`: for each expert block compute the row folds
  `row_dots[e][d] = sum_j W_all[e][d][j] * eq(j, v)` once, then
  `w_v[pos][d] = row_dots[expert_of[pos]][d]`; the per-expert terminal
  `table_eval[e] = MLE(W_all[e], (ch_d, v))` is recovered from the stored row
  folds after the sumcheck (`table_evals`), with no second weight pass.
- `gather_matmul_forward_streaming` (forward), `prove_with_w_v` (the existing
  eq-selector sumcheck), `prove_gather_matmul_streaming` (self-contained path).
- Forward, plain prove and the bound reverse reduction no longer materialize
  W_all. The bound weight claim is still `(counts_e, table_eval[e])`, bound by
  opening `W_all[e]` at the n points `(e, ch_d, v)` batched with the
  linear-combination check. No new reduction; transcript and authentication
  semantics are unchanged; out-of-range expert ids fail closed.
- Tests: `streaming_from_mmap_matches_slice` (mmap vs owned forward, w_v and
  per-expert terminals agree with the materialized reference; the streamed
  proof verifies; a tampered terminal is rejected) and
  `s7_gather_matmul_mmap_weight_bound_roundtrip` (bound path with an
  mmap-backed weight; honest verify + bytes verify; tampered weight terminal
  rejected).

## Test gate (test machine)

- `cargo test -p zkie-ops --lib`: 154 passed, 0 failed (153 baseline + the new
  mmap bound regression).
- `cargo test -p zkie-core --lib`: 69 passed, 0 failed, 1 ignored (includes the
  new `eq_evals_matches_direct_product` and `same_table_multi_claim_grouping`).
- Real segmented example (`prove_core_chain`, 8-expert real slice): composition
  honest verify, invalid slice identity / slice mismatch rejected,
  individually-valid A1+B2 mismatch rejected, all eight real-data negatives
  rejected, SEGMENT-OK.

## Measurements (real layer-10 segmented example, 8-expert slice, 90-bit)

Byte-size note (O2 determinism follow-up): the proof byte counts reported
below were taken before the p3-challenger ordered-grind fix. With the upstream
nondeterministic PoW witness, sizes varied by ±~0.1% across runs on identical
inputs (e.g. B 464,564/462,132/461,940 B; C 354,735/355,119/355,247 B; every
variant verified). After the fix (`[patch.crates-io]` vendored p3-challenger
with ordered `find_first`) proof bytes are deterministic; the time/RSS
measurements below are unaffected by the byte nondeterminism itself.

Cold-read metering fix: `scripts/o5_measure.py` records page-cache residency
with `fincore --bytes` immediately before and after each run, samples
`/proc/<pid>/{io,status,stat}`, and reports child rusage. A run is classified
cold only when `fincore_before=0` and warm when it equals the probed file size.
The earlier full-64 RESULT had `read_bytes=0` because that run was warm and had
no cache-state probe; zero read bytes was not cold evidence. On this kernel the
cold mmap path also reports the file bytes in `read_bytes`, so the RESULT line
reports `read_bytes`, `ru_inblock` and `majflt` together with `fincore`
before/after. `/dev/shm` (ZKIE_CACHE) is tmpfs and is NOT dropped.

Exact command (cold: drop caches first; warm: rerun immediately):
```
cd /data/zkie-soundness-fix
export DS_WEIGHTS=/data/zkIE/models/deepseek-v2-lite/weights
export ZKIE_CACHE=/dev/shm/zkie-core-chain-cache DS_LAYER=10
W=$DS_WEIGHTS/L10_experts_gate_i32.bin
sync; echo 3 > /proc/sys/vm/drop_caches
python3 scripts/o5_measure.py target/release/examples/prove_core_chain \
  logs/o5-cold-seg-73712b8.log "$W"
python3 scripts/o5_measure.py target/release/examples/prove_core_chain \
  logs/o5-warm-seg-73712b8.log "$W"
```

Cold (fincore 0 -> 1,073,741,824 B): wall 12.207 s; maxrss 2,450,004 kB;
minflt 67,858; majflt 9; inblock 2,107,840 (512-B blocks); sampled read_bytes
1,078,497,280 B (the full `L10_experts_gate_i32.bin` read by `load_expert_slice`,
plus ~4.7 MB binary/libs). Warm (fincore_before=1 GiB): wall 11.147 s;
maxrss 2,449,624 kB; majflt 0; inblock 0; read_bytes 0. Both:
`claims=8 unique=1 moved_bytes=268,435,456 cloned_bytes=0`; open 3.12 s;
B commit+prove 5.32/5.29 s; verify 29.6/30.0 ms; all eight negatives rejected;
SEGMENT-OK. Logs:
`/data/zkie-soundness-fix/logs/o5-cold-seg-73712b8.log`,
`/data/zkie-soundness-fix/logs/o5-warm-seg-73712b8.log`.

Before this step (same example, documented baseline): per-claim clone
+2.10 GB, open 17.2 s, VmHWM 10.0-10.3 GB, B commit+prove 19.9 s. After:
cloned_bytes=0, 256 MiB moved once, open 3.12-3.18 s, VmHWM 2.42-2.45 GB,
B commit+prove 5.25-5.32 s. Same 90-bit security, same real inputs, same
authentication scope.

## S7 full 64-expert weight regression (previously OOM-killed)

Harness (committed): `models/deepseek-v2-lite/examples/prove_core_chain_full64.rs`,
the full-64 variant of the segmented real-data experiment: a single
2^28-element real expert-gate weight, mmap-backed (`WeightMmap` +
`Store::push_mmap`), 64 claims on one table; same real layer-10 data, 90-bit,
cache in /dev/shm. It is a memory/protocol regression harness, NOT S7
acceptance (segmented A/B; B reads the full routing arrays as public data).

Command (cold; warm is the same line rerun without the drop):
```
cd /data/zkie-soundness-fix
export DS_WEIGHTS=/data/zkIE/models/deepseek-v2-lite/weights
export ZKIE_CACHE=/dev/shm/zkie-core-chain-cache DS_LAYER=10
W=$DS_WEIGHTS/L10_experts_gate_i32.bin
sync; echo 3 > /proc/sys/vm/drop_caches
python3 scripts/o5_measure.py target/release/examples/prove_core_chain_full64 \
  logs/o5-cold-full64-73712b8.log "$W"
python3 scripts/o5_measure.py target/release/examples/prove_core_chain_full64 \
  logs/o5-warm-full64-73712b8.log "$W"
```

Cold (fincore 0 -> 1,073,741,824 B): wall 162.664 s; maxrss 15,998,568 kB;
minflt 248,345; majflt 8,209; inblock 2,107,584; sampled read_bytes
1,078,366,208 B. Warm (fincore_before=1 GiB): wall 148.303 s; maxrss
15,997,612 kB; majflt 0; inblock 0; read_bytes 0. Both:
- `O5-metrics batch=0 claims=64 unique=1 moved_bytes=2,147,483,648
  cloned_bytes=0 tables=1`
- `O5-phases mat_ms=690.6/693.6 restore_ms=3300.7/3263.8
  open_ms=136099.4/122529.9 rss_now_mat=3,178,036/3,177,752
  rss_now_restore=11,808,468/11,808,248 rss_now_open=1,192,436/1,192,256
  hwm=15,998,568/15,997,612` (kB; cold/warm; VmHWM is cumulative)
- B commit+prove 152.45/137.84 s, verify 155.4/152.7 ms, proof
  514,681/513,209 B, peak RSS 15,998,568/15,997,612 kB (15.26 GiB).
- Harness negatives rejected: A gate/score tamper, B public routing array,
  B g output, wrong expert root, wrong activation table root;
  `CORE-CHAIN-OK`.

The cold-vs-warm wall delta (+14.4 s) is the 1 GiB weight read; no security
parameter changes. Proof byte size varies run to run (WHIR prover randomness),
so sizes are reported per run. Logs for these runs:
`/data/zkie-soundness-fix/logs/o5-cold-full64-73712b8.log`,
`/data/zkie-soundness-fix/logs/o5-warm-full64-73712b8.log`.

### Committed harness rerun (reproducible)

The harness is committed at
`models/deepseek-v2-lite/examples/prove_core_chain_full64.rs`. The
wrong-activation-table-root negative now evaluates `s2` before manifest
construction, so the Scale input shape is present and the verifier rejects the
tampered table (previously it panicked with "missing trusted Scale input
shape"). Build and run (cold; warm is the same meter line without the drop;
host load average was ~35 at start, so timings are not comparable to the
low-load runs above):
```
cd /data/zkie-soundness-fix
cargo build --release -p zkie-models-deepseek-v2-lite --example prove_core_chain_full64
export DS_WEIGHTS=/data/zkIE/models/deepseek-v2-lite/weights
export ZKIE_CACHE=/dev/shm/zkie-core-chain-cache DS_LAYER=10
W=$DS_WEIGHTS/L10_experts_gate_i32.bin
sync; echo 3 > /proc/sys/vm/drop_caches
python3 scripts/o5_measure.py target/release/examples/prove_core_chain_full64 \
  logs/o5-cold-full64-committed.log "$W"
python3 scripts/o5_measure.py target/release/examples/prove_core_chain_full64 \
  logs/o5-warm-full64-committed.log "$W"
```

Cold (fincore 0 -> 1 GiB): read_bytes 1,078,366,208; majflt 8,209; wall
177.432 s; maxrss 15,998,544 kB; open 150.6 s; B 166.9 s / 156.1 ms; proof
512,953 B. Warm (fincore_before=1 GiB): read_bytes 0; wall 190.256 s; maxrss
15,998,592 kB; open 164.5 s; B 179.9 s / 154.0 ms; proof 513,081 B. Both:
claims=64 unique=1 moved 2,147,483,648 cloned=0; all six harness negatives
rejected including wrong activation table root; `CORE-CHAIN-OK`. Logs:
`/data/zkie-soundness-fix/logs/o5-cold-full64-committed.log`,
`/data/zkie-soundness-fix/logs/o5-warm-full64-committed.log`.

The earlier `/tmp/prove_core_chain.rs` was overwritten by an orphaned shell from
the stopped continuation agent at 00:38; the source was then fixed (`s2`
evaluate_ops) and committed, `prove_core_chain.rs` restored to `73712b8`, and
the binaries rebuilt. The 73712b8 logs above were produced by the separate
retained binary before the overwrite and are unaffected.

Previous state: OOM-killed at anon-rss 502,710,376 kB (dmesg) on the 495 GB
machine. The staged RSS shows the peak window is the W_t construction after
the prover-data restore: restore VmRSS 11.26 GiB, then one 2 GiB table buffer +
one 2 GiB W_t + one 2 GiB transient eq buffer; by the opening sample current
VmRSS is 1.14 GiB and VmHWM stays at 15.26 GiB. The opening (122-136 s)
dominates the time, not memory.

## Remaining / acceptance reminders

- S7 is still NOT accepted: the segmented B verifier ingests the full routing
  table as public data, and the slice-weight identity vs the full 64-expert
  model is a trusted-offline assumption. O5 changes none of that. The full
  64-expert case is now a working memory regression, not S7 acceptance.
- Next: S7 router-gate consistency chain + Gather; then O2/O3/O4/O6/O7; S8/S9
  postponed.

## Resume commands (test machine)

```
cd /data/zkie-soundness-fix
cargo test -p zkie-ops --lib
DS_WEIGHTS=/data/zkIE/models/deepseek-v2-lite/weights \
ZKIE_CACHE=/dev/shm/zkie-core-chain-cache \
cargo run --release -p zkie-models-deepseek-v2-lite --example prove_core_chain
```
