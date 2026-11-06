# S7 full 64-expert retry (step 3)

Date: 2026-11-04. Status: implementation and real-data retest complete,
**pending the TL's final S7 judgment — this document does not declare S7
accepted.** The single-statement S7 construction was not achieved; the chain
below is segmented, and its exact trust boundaries are listed in
"Honest boundaries".

Companion docs: `docs/gelu-big-continuation.md` (steps 1, 2a, 2b) and the S7
section of `docs/soundness-fix-tracker.md`.

## Result

The real layer-10 DeepSeek-V2-Lite core chain now runs end to end as four
bound/bytes-entry statements over the full 64-expert model:

| phase | content | proof bytes | verify |
|---|---|---|---|
| A | `TopKSelect -> Scatter -> PadIdx` (real scores) | 241,874 | not separately timed |
| B | routing replay + activation lookup + full real 64-expert gate projection | 540,239 | 159 ms |
| D | full real 64-expert FFN, including silu 2^24 and `ScaleVec` shift 16 | 1,779,403 | 339 ms |
| C | router-gate consistency + Gather into the router output | 355,183 | 15.5 ms |

All four honest proofs verify through the formal bytes entry at 90-bit WHIR
security. All real-data negatives listed below are rejected. Two clean runs
are byte-identical in all four dumps (see "Determinism and baselines").

## Chain

```
A: scores --TopKSelect--> sel, thr, gate, d1, d2
           --Scatter--> token_id, expert_of
           --PadIdx--> token_id_pad, expert_of_pad          (public outputs)

B: token_id_pad/expert_of_pad (public index inputs)
   -> IdxBroadcast -> Lookup(real post-attn activation) -> BroadcastScale(mask)
   -> GatherMatMul(full real 64-expert gate W, 2^28 slots) -> Scale(16)
   => g

D: token_id_pad/expert_of_pad, mask, g (B's proven output), real activation
   -> IdxBroadcast -> Lookup(real activation) -> BroadcastScale(mask)
   -> GatherMatMul(full real 64-expert up W, 2^28 slots) -> Scale(16)
   -> GeluIndex(offset 2^23, table_len 2^24)
   -> Lookup(real silu 2^24 table)
   -> ScaleVec(shift 16, precommitted remainder bits, ties-up)
   -> GatherMatMul(full real 64-expert down W, 2^28 slots) -> Scale(16)
   => d

C: scores -> TopKSelect -> Scatter -> PadIdx
   -> CombineIdx -> Lookup(table = op-written TopK gate)   (gate consistency)
   -> BroadcastScale(d, gate_pos) -> Scale(16) -> Gather(ones) -> out
   => thr, gate, d1, d2, out
```

Real data: `scripts/ds_layer_core_inputs.py`, layer 10, seq=1, token 549,
top-6 experts `[13,42,47,9,2,31]`. The route/padding arrays are public data in
every statement. The D phase recomputes the full FFN from the real up/down
weights; `g` is asserted equal to B's proven output and `d` is asserted equal
to C's caller input. The gate-weight consistency is checked in C: the
`BroadcastScale.scale` input is produced by a Lookup whose table is the
op-written TopK gate tensor, so the weight applied to each down-projection
row is the router gate from the same statement.

Real-domain measurements asserted by the harness: gate `g` in
`[-232482, 171093]`, shifted `w = g + 2^23` in `[8156126, 8559701]`, and
`high = w - (2^24 - 1)` in `[-8621089, -8217514]`, all inside the 25-bit
ReLU/GeluIndex domain; `|silu|max = 159381`, `|u|max = 231507`, both inside
the `ScaleVec` `2^24` operand contract.

## Gist S7 acceptance mapping

| gist S7 item | real-chain negative | ops-level regression |
|---|---|---|
| 等长但伪造的 counts/table 拒绝 | Equal-length routing-array forgery is the real-data counterpart: `B tampered (duplicate) public routing array` and `wrong expert weight root` are rejected. The proof no longer carries a free counts/table vector (counts are derived from the authenticated routing aux; weight terminals are locked by the manifest openings), so the literal proof-field attack is exercised at the ops level. | `s7_equal_length_forged_counts_table_rejected` (forged weight table, forged counts-side aux opening, and a consistent rewrite that breaks the matmul sumcheck) |
| 错专家根拒绝 | `wrong expert weight root` (B, unrelated committed root), `wrong up/down expert root` (D) | `s7_swapped_expert_weights_rejected` |
| 交换专家拒绝 | `D swapped experts` (swap two routing entries) | `s7_swapped_expert_weights_rejected` |
| 重复路由拒绝 | `B tampered (duplicate) public routing array`; `D dropped/duplicate route` | `s7_routing_relation_rejects_duplicate_dropped_and_out_of_range` |
| 删除活跃对拒绝 | `D dropped route` | `s7_routing_relation_rejects_duplicate_dropped_and_out_of_range` |
| 篡改 gate 拒绝 | `A tampered gate`, `A tampered router score input`, `C tampered gate`, and the out-of-range router score fail-closed case | `s7c_gate_chain_static_table_via_scale_slot_rejected`, `s7c_gather_gate_static_via_broadcast_rejected`; real-data static bad gate table finding (`logs/s7-c-realdata-finding-20261104.log`) stays fail-closed |
| 小 MoE 通过 | — | `s7_gather_matmul_bound_roundtrip`, `s7_gather_matmul_mmap_weight_bound_roundtrip` |
| 单层真实 DeepSeek 通过 | A/B/D/C honest chain passes on the real layer-10 data (segmented; see boundaries) | — |

D also rejects tampered output, tampered `g`, tampered mask, and a wrong real
silu table root. B also rejects a tampered activation table root and a
tampered `g` output. C also rejects tampered `d`, scores, and output.

## 64-expert memory regression

The earlier 8-expert-slice workaround existed because the 64-expert (2^28)
gate opening was OOM-killed (`anon-rss` 502,710,376 kB). The O5 streaming
work plus the step-3 harness now prove the full real tensors directly; the
harness no longer assumes a weight-slice identity to fit in memory.

| phase | peak RSS (run 2; run 1 in parentheses) |
|---|---|
| B (one 2^28 gate tensor) | 15,998,664 kB (16,000,432 kB) |
| D (two 2^28 up/down tensors) | 20,069,632 kB (20,050,632 kB) |

Machine: 495 GB RAM, 64 cores. No OOM; RSS stays well inside the machine.
The O5 cold/warm metering lines (`O5-phases mat/restore/open`) are retained in
the log for the memory regression record.

## Determinism and baselines

Two clean runs produce identical proof dumps (`cmp` on A/B/D/C), and a third
run of the final committed source reproduces the same four SHA-256 values.
Baselines:

| proof | bytes | SHA-256 (identical in both runs) | commit+prove (run 2) | verify (run 2) |
|---|---|---|---|---|
| A | 241,874 | `3ad3fb4bd4f187f78afff24b03f74b5347fa7f723f5ff469320803f98d3dac66` | 0.15–0.18 s | not separately timed |
| B | 540,239 | `e3e68911b07a6ae55713b7639831e06bd5558ad01216dd8375d6ac196f4933c8` | 241.4 s | 159 ms |
| D | 1,779,403 | `4a138b1c38328b152baab7f8c3ac5a923ec835b2c01d83b474cd8b7b413880c7` | 487.7 s | 339 ms |
| C | 355,183 | `51a68635503991d85d5c0e2faf13a24d902ef52007009b813455ca376bf2f7f1` | 204 ms | 15.5 ms |

Run 1 was executed concurrently with the `zkie-core` test suite (1,211.63 s),
so its B/D prove and verify times are load-inflated (B 390.6 s, D 811.4 s);
run 2 is the uncontended baseline and matches the earlier debug run
(B 233.6 s, D 482.1 s). The proof bytes are identical in both runs, so the
determinism claim does not depend on timing. Run 3 of the committed source
(uncontended) reproduced all four hashes with B 237.9 s / D 485.8 s and peak
RSS 15,999,872 / 20,036,096 kB. A verify is exercised but not separately
timed.

Security level is unchanged at 90-bit (`Whir::new_testing(5)`). A 241,874,
B 540,239, and C 355,183 match the earlier experiments; D grows from the
8-expert-slice 1,359,818 to 1,779,403 with the full 64-expert up/down
weights inside the statement. These four are the new baselines.

## Honest boundaries

1. **Segmented, not single-statement.** Each of A, B, D, C is a separate
   bound proof. The links `A outputs = B/D/C inputs`, `B's g = D's g`,
   `D's d = C's d`, and `A's gate = C's gate` are caller-trusted equalities
   over shared public arrays, not one proof. A single statement is still
   blocked by the internal routing index intermediates: `IdxBroadcast`
   requires a verifier-replayable or certified public base, and replacing it
   with random-point claims would drop the pointwise u32 / `b*base+j`
   overflow contract (the earlier relaxation attempt was reverted at the
   TL's request). **No single-statement S7 proof was achieved.**
2. **The verifiers read the full routing arrays.** In A/B/D/C the routing
   arrays are public statement data; the gist's final S7 construction forbids
   this. The ops-level S7 aux path (verifier-derived counts from the one-hot
   routing aux, manifest-backed expert terminals, Scatter order/LogUp
   uniqueness) is covered by the synthetic regressions, but the real layer
   chain does not yet exercise it as one statement.
3. **Weight identity is a trusted-manifest assumption.** B and D use the
   full real `L10_experts_{gate,up,down}_i32.bin` tensors directly, so the
   earlier 8-expert *slice* identity caveat does not apply to these roots;
   the verifier authenticates the offline-committed bytes, not that those
   bytes are the model's canonical weights. Associating the manifest with
   the real model remains offline-trusted, as everywhere else in the bound
   API.
4. **Gate consistency is the positive path.** C exercises the real MoE
   pattern (router gate enters through `BroadcastScale.scale` from an
   op-written TopK-gate Lookup) and it verifies. The static-forged-gate-table
   counterexample is fail-closed by the staged verifier rule, demonstrated at
   the unit level and by the earlier real-data finding; the rule deliberately
   only traverses the weighted-side `Scale`/`ScaleVec` chain, so other ops
   mixed into that chain remain outside the fix.
5. **Real trajectory caveat.** The inputs come from the float64 reference
   model (bf16 weights) quantized at the Q16 boundary, not a full fixed-point
   model trajectory. This is the same data provenance as the prior segments.
6. **S7 status.** S7 remains incomplete in the gist sense. The TL judges
   whether the evidence above constitutes acceptance; this step does not
   self-declare it. S8/S9 remain postponed.

## Tests

- `zkie-ops --lib`: 170 passed, 0 failed.
- `zkie-core --lib`: 73 passed, 0 failed, 1 ignored.
- Real-chain harness: A/B/D/C honest verification, all negatives above,
  double-run dump comparison.

## Logs

`/data/zkie-soundness-fix/logs/s7-final-run{1,2,3}.log`, with dumps
`s7final-run{1,2,3}-{a,b,d,c}.bin`.
