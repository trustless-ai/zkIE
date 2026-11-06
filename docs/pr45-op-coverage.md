# PR#45 B1 — four-model op coverage inventory (current head)

Date: 2026-11-05. Scope: the real graph builders in `models/*/src/lib.rs` and
the bound verifier's reverse whitelist (`reverse_op_io`, compose.rs). This is
the evidence requested before wiring the five missing reverse reductions. It
does not claim model-level acceptance: B2 (statement migration + independent
verifier) and B3 (single-statement MoE) remain open.

## 1. Op use by the four model builders

Counts are `Op::` occurrences in each model crate (builders; examples repeat
only a subset).

| op | GPT-2 | Gemma3 | DeepSeek-V2-Lite | TimesFM | reverse whitelist |
|---|---|---|---|---|---|
| Add | 6 | 3 | 6 | 9 | yes |
| MatMul | 3 | 3 | 7 | 2 | yes |
| Relu | 0 | 0 | 0 | 1 | yes |
| Transpose | 1 | 1 | 2 | 1 | yes |
| Projection | 6 | 7 | 13 | 12 | yes |
| Lookup | 2 | 2 | 19 | 2 | yes |
| GeluIndex | 1 | 1 | 6 | 2 | yes |
| Scale | 4 | 3 | 23 | 3 | yes |
| ScaleVec | 0 | 1 | 6 | 1 | yes |
| Argmax | 1 | 1 | 1 | 0 | yes |
| TopKSelect | 0 | 0 | 11 | 0 | yes |
| Scatter | 0 | 0 | 6 | 0 | yes |
| BroadcastScale | 0 | 0 | 14 | 0 | yes |
| Gather | 0 | 0 | 6 | 0 | yes |
| GatherMatMul | 0 | 0 | 16 | 0 | yes |
| CombineIdx | 0 | 0 | 4 | 0 | yes |
| IdxBroadcast | 0 | 0 | 9 | 0 | yes |
| PadIdx | 0 | 0 | 13 | 0 | yes |
| **Softmax** | 1 | 1 | 2 | 1 | **no (fail-closed)** |
| **StableSoftmaxIndex** | 1 | 1 | 2 | 1 | **no (fail-closed)** |
| SoftmaxIndex | 0 | 0 | 0 | 0 | no (not used) |
| **RmsNorm** | 0 | 7 | 4 | 1 | **no (fail-closed)** |
| **RoPE** | 0 | 2 | 2 | 0 | **no (fail-closed)** |
| **ScaleGate** | 0 | 0 | 0* | 0 | **no (fail-closed)** |
| LayerNormCentered | 3 | 0 | 0 | 1 | **no (fail-closed)** |
| Layernorm | 0 | 0 | 0 | 0 | no (not used) |
| BatchedMatMul | 0 | 0 | 0 | 0 | no (not used) |

\* ScaleGate is not emitted by the current DeepSeek builder; it is retained
in `Op`/`OpProof::ScaleGate` for the router-gate path that B3 needs. The B1
priority still includes it because B3's single-statement MoE chain requires
`Softmax → ScaleGate` router normalization.

## 2. Status matrix

Columns: **code** = an op-level proof module/OpProof exists; **bound** = the
bound any-op verifier accepts it today; **real model** = a full real-model
statement through the bound entry (B2/B3, not yet).

| op | code | bound entry | real model layer | action |
|---|---|---|---|---|
| Add / MatMul / Transpose / Projection / Lookup / GeluIndex / Scale / ScaleVec | yes | yes | no (B2) | — |
| Relu | yes | yes | TimesFM (B2) | — |
| Argmax / TopKSelect / Scatter / BroadcastScale / Gather / GatherMatMul / CombineIdx / IdxBroadcast / PadIdx | yes | yes | DeepSeek segmented (B3 for single statement) | — |
| Softmax | softmax_scaled.rs | yes (`908e00b`) | all four models | done (B1) |
| ScaleGate | OpProof::ScaleGate | yes (`c47ed80`) | B3 router | done (B1) |
| StableSoftmaxIndex | softmax_scaled.rs | yes (`74088e4`) | all four models | done (B1) |
| RmsNorm | rms_norm.rs | yes (`990b709`) | Gemma3/DeepSeek/TimesFM | done (B1) |
| RoPE | rope.rs | yes (`2346cf3`) | Gemma3/DeepSeek | done (B1) |
| LayerNormCentered | layer_norm_centered.rs | yes (LNC reverse) | GPT-2/TimesFM | B1 op #6; mean/var/rsqrt/out reduction on the pooled aux wire (`docs/lnc-reverse.md`) |
| SoftmaxIndex / Layernorm / BatchedMatMul | some | no | none of the four | out of scope; fail-closed |

## 2b. Softmax row-sum window: measured max S and B=21

The reverse Softmax normalization is

    out(i,j) = round(e(i,j) * 2^16 / S(i)),     S(i) = sum_j e(i,j).

The proof commits `rem = e*2^16 - out*S + 2^(B-1)` with B Boolean columns
and enforces `rem in [0, 2^B)` through

    out*S - 2^16*e - rem + 2^(B-1) = 0.

The forward output scale stays **2^16**; only the committed remainder window
is widened. The true rounding error is at most `S/2` in absolute value, so the
half-offset window contains it when `|S| <= 2^B`; a row outside that bound
fails closed at aux preparation.

**Measured widths** (full real-graph scan, `models/*/examples/softmax_scan.rs`):

| graph | Softmax ops | max abs S | location |
|---|---|---|---|
| GPT-2, M=16 | 144 | 573,417 | op 142, row 15 |
| TimesFM, SEQ=16 | 320 | 674,671 | op 328, row 15 |

Global `max|S| = 674,671`, so `ceil(log2(max)) = 20`. Per the TL's
measure-then-width rule with one bit of margin, `SOFTMAX_REM_BITS = 21`
(`S <= 2^21 = 2,097,152`). The earlier values 227,312 (GPT-2 op12 row8) and
164,266 (TimesFM op29 row4) were the first failures under the 17-bit window,
not the maxima; a 21-bit window covers both.

**Exact candidate count.** The remainder condition is equivalent to

    out in ( (X - 2^(B-1))/S, (X + 2^(B-1))/S ],  X = e*2^16,

a half-open interval of length `2^B/S`. It contains at most `ceil(2^B/S)`
consecutive integers. With B=21 the "at most two candidates" property holds
for `S >= 2^20 = 1,048,576`; rows with smaller S can have more candidates
(e.g. S=2^16 has up to 32). This is a property of a fixed remainder window:
covering the measured max S=674,671 and keeping the at-most-two threshold at
S=2^16 are incompatible for a single B. Real rows include very small S
(min observed 0), so the uniqueness statement from the 17-bit window
(`S >= 2^16 -> at most 2`) does not carry over unchanged to B=21. The
relation and the committed out still bind one value, but the fixed-point
rounding is not selected uniquely by the window alone for small S. This is
flagged for the TL; no domain contract is changed in this commit.

Regression `softmax_rowsum_window_measured`: S=86,741 (original failure),
S=674,671 (measured max), and S=2^16 prove; S=2^21+1 with an error of exactly
+2^20 (remainder 2^21) fails closed.

## 2c. Measured fixed-point range limits

Real graph boundaries are not guessable from the op contract; each ranged
operator must be sized from measured extremes with one bit of margin. The
scanners live in `models/*/examples/softmax_scan.rs` (Softmax, SSI, LNC).
Measured over the full GPT-2 (M=16) and TimesFM (SEQ=16) graphs:

| quantity | constant | coverage | measured max | margin |
|---|---|---|---|---|
| Softmax row sum S | `SOFTMAX_REM_BITS=21` | 2^21 = 2,097,152 | 674,671 (TimesFM op328 row15; GPT-2 573,417 op142 row15) | 3.1x |
| SSI `mdiff = rmax-masked` | `SSI_RANGE_BITS=32` | 2^32 = 4,294,967,296 | 1,075,866,322 (GPT-2 op2090) | 4.0x |
| SSI `tr_w = trunc-w` | `SSI_RANGE_BITS=32` | 2^32 | 1,073,769,170 (GPT-2 op2090) | 4.0x |
| SSI `diff = trunc-idx` | `SSI_RANGE_BITS=32` | 2^32 | 1 | — |
| LNC mean | 16-bit signed raw + 16-bit rem | mean in [-2^31, 2^31) | 319,512 (GPT-2) / 22,822 (TimesFM) | >6000x |
| LNC out pin `out-b+2^31` | 32-bit limbs | 2^32 | 2,161,616,942 (GPT-2) / 2,148,437,433 (TimesFM) | 1.99x |
| LNC out remainder | 2x16-bit limbs | 2^32 | range by construction | — |
| Argmax `d2` gap | `ARGMAX_D2_BITS=43` | 2^43 = 8,796,093,022,208 | 2,348,020,452,072 (GPT-2 op2117) | 3.7x |
| Argmax `tie` gap | `ARGMAX_TIE_BITS=43` | 2^43 | 2,348,020,452,072 (same location) | 3.7x |

Every widening keeps the Boolean decomposition canonical and charges one
extra column/claim (Softmax: +1 remainder bit; SSI: +3 range columns per
bit). The PCS opening count K is unchanged by these width changes.

**M4a: per-table internal lookup commitments/openings.** All non-identity
internal lookup tables (rsqrt for LNC/RmsNorm, exp for Softmax, range tables)
are now committed once per proof (`prepare_static_lookup_commits`), and every
per-op table opening is deferred into ONE batch opening per table
(`BoundExecutionProof.internal_table_openings`, bundle version 4). In pooled
mode a reduction records `(table_id, leaf point, leaf value)`; the verifier
replays the same claims and checks one batch opening per table against the
trusted root. This removes the per-op table commitments and openings but
keeps the per-op LogUp witness/table fraction trees and multiplicity columns;
M4b (aggregate multiplicity `M = Σ m_op` and a shared table fraction tree)
remains for the memory/time reduction.

Verifier-derived opening upper bound
(`opening_upper_bound`, conservative): pooled aux groups + one internal table
batch per shared table + two openings per static lookup group + one selector
opening per Argmax + token binding + weight batches + Relu/Gelu bit openings.
Measured: **GPT-2 51, TimesFM 48**, both below 64. (The S6 selector proof is
one batch opening per Argmax with m claims, not m openings.)

**Full-graph K implication.** Before M4a, per-op static-table openings put the monolithic B2 proof over
the 64-opening cap by construction: GPT-2 had 144 Softmax exp-table + 25 LNC
rsqrt openings (>=169) and TimesFM 320 + 20 (>=340). M4a collapses each
table's openings to one, bringing the conservative bounds to 51 and 48.

**Lesson (recorded per TL):** the first failure under a given width is not
the maximum. A full-graph scan is required before sizing a range constant;
the 17-bit Softmax failure at S=227,312 hid rows up to 674,671; the SSI
22-bit failure at mdiff=7.3M hid GPT-2 values above 2^30; and the Argmax
23-bit failure at gap=18,990,631,562 hid a real maximum of
2,348,020,452,072. Measured maxima plus one bit of margin, with a scanner
and a boundary regression per operator, is the standing procedure.

## 3. Implementation constraints for B1

- New reductions must be EF (claim points/values/terminals), integrate with
  `ReverseReduction`/`ReverseDagProof`, and return verifier-derived claims for
  inputs and aux columns; they must not re-open the legacy base verifier.
- Relations must be aligned with the forward semantics in compose.rs
  (`apply_op`): rounding/shift, masks, index clamping, row normalization and
  endpoints. Tag/label checks alone are insufficient.
- Aux columns are precommitted and descriptor-pinned like the existing
  reductions; endpoints must be authenticated by aux/weight openings.
- Until each op is implemented and tested, `reverse_op_io` must keep
  returning `None` for it (fail-closed). The old verifier stays closed.

## 4. B2/B3 dependencies

- B2 must move GPT-2/Gemma3/DeepSeek examples to `LanguageModel` and run an
  independent verifier process; TimesFM stays `Generic`. The current examples
  are still `Generic` and same-process.
- B3 needs the Softmax/ScaleGate reductions plus a single-statement MoE
  route→TopK→Scatter→expert→gate→Gather chain. This inventory is not B3
  acceptance.

## 5. B1 closure and B2 bundle status

All five priority ops now have bound reverse reductions on
`fix/soundness-repair` (Softmax `908e00b`, ScaleGate `c47ed80`,
StableSoftmaxIndex `74088e4`, RmsNorm `990b709`, RoPE `2346cf3`), and
LayerNormCentered (op #6) is covered by the LNC reduction described in
`docs/lnc-reverse.md`. Each
reduction commits its witnesses before the transcript challenges, proves the
forward rounding/range/index semantics with EF virtuals, and authenticates
the endpoints against the committed tensors/tables; the old unsafe entries
remain permanently closed.

B2 adds the public bundle + independent verifier contract (`src/bundle.rs`,
`src/bin/verify_bundle.rs`):

- `bundle.bin` carries only the trusted op graph, the public statement, the
  root-only model manifest and the caller-owned execution I/O; `proof.bin`
  carries the bound proof. The verifier process loads exactly these two files
  and nothing else (no weights, store or prover caches). **bundle.bin 不含任何
  权重数据**：权重只以 manifest 中的 commitment root 形式出现，roots 是唯一
  的权重认证源；`verify_bundle_dir` 的 whir 参数是遗留签名占位，verifier
  内部按 `VERIFIER_SECURITY`/各 opening arity 重建 WHIR。
- The token binding accepts both canonical embedding layouts: column-major
  (`d*base + token`, GPT-2's transposed lm_head) and row-major
  (`token*base + d`, Gemma3/DeepSeek embed_tokens). Prompt, response, output
  and embedding-token tampering are rejected, as are model-root changes and
  argmax/tie forgeries (test
  `b2_public_bundle_independent_verifier_and_tamper_matrix`).
- The independent process test covers missing/corrupt artifacts and the
  fail-closed binary contract.

Remaining B2 work for the four real-model examples is the per-model public
execution assembly (input tables, execution outputs) and full-graph proving.
GPT-2/TimesFM still hit the out-of-scope fail-closed `LayerNormCentered` gap;
Gemma3/DeepSeek are fully covered by the five B1 reductions. The per-op
RmsNorm rsqrt table side is O(2^23); cross-op aggregation is the deeper O7/B3
work agreed with the TL.
