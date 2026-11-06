# O7 closeout — evidence and acceptance items

State: `5dd539e` + this closeout (new tests, an O7 metrics print in the real-B
example, docs; no protocol/reduction/transcript change) on
`fix/soundness-repair`. O7 awaits TL review.

The deep O7 mechanism record is `docs/o7-deep-continuation.md`; this file
collects the closeout evidence the TL asked for: targeted negatives,
aggregation accounting, same-table before/after statistics under cold and warm
page cache, the real-chain comparison, and full regression.

## 1. Negative regression

All of the following are in the `zkie-ops` release suite; every case rejects
(or fails closed at proof time) and no case panics.

| attack | test | result |
|---|---|---|
| wrong static table root | `gelu_silu_lookup_wrong_root_and_index_tamper_rejected` | reject |
| table opening moved off the derived leaf point | `s2_table_leaf_point_swap_rejected` | reject |
| wrong group table id / empty member list / out-of-range member / empty group tree / duplicate group | `s8_lookup_group_wire_mutations_reject_without_panic` | reject |
| **missing Lookup step** (one member dropped) | `o7_missing_lookup_step_rejected` | reject |
| **missing group step** | `o7_missing_lookup_step_rejected` | reject |
| per-op multiplicity active entry zeroed | `o7b_zeroed_active_multiplicity_rejected` | prover fail-closed + verifier reject |
| **forged aggregate multiplicity** (sum violation, swapped M commitment, forged M identity eval) | `o7b_forged_aggregate_multiplicity_rejected` | fail-closed + reject |
| forged padded identity row | `lookup_rejects_forged_padded_identity_row` | reject |
| **cross-group misconnection** (swap two group proofs) | `o7_cross_group_swap_rejected` | reject |
| cross-table structural merge | `o7b_cross_table_merge_rejected` | reject |
| dynamic table mixed into a static group | `o7b_dynamic_table_not_grouped` | reject |

The bold rows are the new closeout negative coverage (the N=4 accounting
test is in section 2). The rest are the existing negative suite from deep
O7 / S2 / S8. Mapping to the requested directed
cases: tampered group multiplicity -> `o7b_forged_aggregate_multiplicity_rejected`
and `o7b_zeroed_active_multiplicity_rejected`; forged group identity ->
the identity-eval/commitment legs of `o7b_forged_aggregate_multiplicity_rejected`
plus the truncated-identity wire case in `s8_lookup_group_wire_mutations_reject_without_panic`;
cross-group misconnection -> `o7_cross_group_swap_rejected` and
`o7b_cross_table_merge_rejected`; missing lookup -> the new
`o7_missing_lookup_step_rejected`.

## 2. Aggregation accounting

One group is one trusted table id. The aggregate multiplicity column is one
`LookupMultiplicityAgg` aux descriptor per group; each member keeps its own
per-op multiplicity and witness tree.

| case | members | table groups | Agg columns | per-op multiplicity cols | per-op witness trees | table-side trees | table openings | table commits |
|---|---|---|---|---|---|---|---|---|
| synthetic N=2 same table | 2 | 1 | 1 | 2 | 2 | 1 | 1 | 1 |
| synthetic N=4 same table | 4 | 1 | 1 | 4 | 4 | 1 | 1 | 1 |
| real B (singleton) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

- `o7_same_table_lookups_share_table_commit` asserts the N=2 row (one table
  commit, one group `[0,1]`, two per-op Lookup steps, one Agg descriptor, two
  per-op multiplicity descriptors).
- `o7_four_same_table_group_counts` asserts the N=4 row (one table-side tree,
  one opening, one table commit, one Agg descriptor, four per-op multiplicity
  descriptors, four witness steps, group members `[0,1,2,3]`).
- The N=2/N=4 table-side tree/opening numbers are also the `o7_deep` example
  counters (`trees=1 openings=1`), independent of the proof-shape assertions.

## 3. Same-table case: aggregation before/after, cold and warm

Harness: `crates/zkie-ops/examples/o7_deep.rs`; synthetic deterministic
siLU-like table; same source runs on the pre-aggregation baseline worktree
`184bf82` and the deep O7 worktree `1450027`. Table is generated in memory.
Cold = `drop_caches` immediately before the run; warm = immediate rerun.

`N=2, table=2^24, idx=2^18` (primary same-table case):

| version | cold/warm | trees | openings | scan bytes | proof bytes | prove ms | verify ms | HWM kB |
|---|---|---|---|---|---|---|---|---|
| 184bf82 | cold | 2 | 2 | 671,088,640 | 979,465 | 10,166.3 | 70.8 | 3,628,396 |
| 184bf82 | warm | 2 | 2 | 671,088,640 | 979,465 | 10,160.0 | 70.9 | 3,630,540 |
| deep 1450027 | cold | 1 | 1 | 536,870,912 | 935,169 | 9,477.8 | 71.3 | 4,433,348 |
| deep 1450027 | warm | 1 | 1 | 536,870,912 | 935,169 | 9,478.6 | 72.1 | 4,433,500 |

Deep vs baseline: trees/openings 2 -> 1; scan bytes -20.0%; proof -4.5%;
prove time -6.8%; verify within noise; HWM +805 MB (shared group/table state
retained), reported as a real trade-off.

`N=4, table=2^20, idx=2^16` (scaling support):

| version | cold/warm | trees | openings | scan bytes | proof bytes | prove ms | verify ms | HWM kB |
|---|---|---|---|---|---|---|---|---|
| 184bf82 | cold | 4 | 4 | 75,497,472 | 1,526,585 | 1,506.6 | 66.2 | 337,540 |
| 184bf82 | warm | 4 | 4 | 75,497,472 | 1,526,585 | 1,539.4 | 66.8 | 334,268 |
| deep 1450027 | cold | 1 | 1 | 33,554,432 | 1,216,289 | 1,152.8 | 55.5 | 382,292 |
| deep 1450027 | warm | 1 | 1 | 33,554,432 | 1,216,289 | 1,159.1 | 55.4 | 382,224 |

Deep vs baseline: trees/openings 4 -> 1; scan bytes -55.6%; proof -20.3%;
prove time -23.5%; verify -16.2%; HWM +44 MiB.

Cold/warm accounting: the synthetic table is generated in memory, so the only
cold I/O is the binary/libs. Cold runs read 4.46-4.62 MB and took 4-5 major
faults; warm runs read 0 B and took 0 major faults. Proof times agree within
~2% run-to-run (table-side work is compute-bound, not file-cache-bound). The
cold/warm split is therefore reported as an honest null file-cache signal for
this case, with the real-chain split below carrying the I/O-sensitive data.

## 4. Real chain: B segment before/after, cold and warm

Harness: `models/deepseek-v2-lite/examples/prove_core_chain.rs`, phase B
(real routed experts, static real activation table, real expert gate weights).
Same B source on `184bf82` and `1450027`; O7 metrics are reset immediately
before the B proof. Cold = `drop_caches` before the run; warm = immediate
rerun. Probe file is the real expert-gate weight file (1 GiB).

| version | mode | trees | openings | scan bytes | proof bytes | B prove ms | B verify ms | process wall s | HWM kB | read_bytes | majflt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 184bf82 | cold | 1 | 1 | 49,152 | 462,836 | 5,331.4 | 29.47 | 16.815 | 3,469,828 | 3,292,610,560 | 8 |
| 184bf82 | warm | 1 | 1 | 49,152 | 462,836 | 5,283.9 | 29.60 | 13.304 | 3,470,080 | 0 | 0 |
| deep 1450027 | cold | 1 | 1 | 65,536 | 491,978 | 5,392.6 | 30.51 | 16.404 | 3,469,252 | 3,292,561,408 | 9 |
| deep 1450027 | warm | 1 | 1 | 65,536 | 491,978 | 5,455.1 | 31.32 | 13.616 | 3,468,952 | 0 | 0 |

The real B segment has exactly one static Lookup, i.e. a singleton group.
Aggregation therefore cannot reduce its tree/opening count (1 -> 1); the
measured cost is the aggregate-M/group-identity overhead: scan bytes +33.3%,
proof bytes +6.3%, B prove +1.2/+3.2% (cold/warm), B verify +3.5/+5.8%. This
is the honest singleton cost, not a claimed win; the multi-lookup win is in
the N=2/N=4 same-table table above.

The cold/warm split is visible at process level: cold reads 3.29 GB from the
weight file (8-9 major faults), warm reads 0 B and takes 0 major faults. B's
`t3` timer starts after the input/weight loading, so the B proof time itself
is compute-bound and does not change with page cache; the overall process wall
drops ~20% cold -> warm (16.8 -> 13.3 s baseline, 16.4 -> 13.6 s deep).

Versions: these two columns are `184bf82` (pre-deep O7) and `1450027` (deep
O7 (d)); later accepted stages (O5/O2/O3/O4/O6/P2) are excluded from this
aggregation comparison. Current HEAD keeps the same mechanism and is covered
by the full regression below.

Full64 has the same singleton static lookup. Its proof-byte record is
510,969 B at the pre-deep O7 baseline and 540,239 B at deep (d)
(`docs/o7-deep-continuation.md`), with the same 1 tree / 1 opening shape.
Full64 cold/warm would take ~4 minutes per proof and is not repeated here;
the B segment above carries the cold/warm and scan-byte evidence.

## 5. Cold/warm logs

- Synthetic: `/data/zkie-soundness-fix/logs/o7syn-{base,deep}-N{2,4}-T{24,20}-{cold,warm}.log`
- Real B: `/data/zkie-o7-baseline/logs/o7real-base-{cold,warm}.log`,
  `/data/zkie-o7-deep/logs/o7real-deep-{cold,warm}.log`
- Worktrees: `/data/zkie-o7-baseline` (184bf82, instrumentation counters),
  `/data/zkie-o7-deep` (1450027).

## 6. Regression

- `cargo build --workspace --release`: pass.
- `zkie-ops` release: 240 passed / 0 failed (237 + three new closeout tests).
- `zkie-core` release: 94 passed / 0 failed / 1 ignored (837.68 s; core is
  unchanged by this closeout, full suite rerun for the gate).
- The existing malicious/negative suite is green; no protocol, transcript,
  commitment, or verifier change in this closeout.

## Remaining

O7 awaits TL review. O7 deep (a)-(d) remains the mechanism of record; this
closeout adds tests and evidence only. S7 remains unaccepted; S8/S9 postponed.
