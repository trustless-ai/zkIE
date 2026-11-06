# S9 step 1 — end-to-end soundness budget inventory

Date: 2026-11-04. Status: **inventory and error-budget derivation only; no
protocol changes, no mitigation selected.** S9 is not accepted. Step 2 (after
TL review) decides between extension-field challenges and independent
repetition, and adds verifier-enforced budget caps plus measured costs.

Scope: the gist S9 item (“WHIR 90-bit ≠ whole-protocol 90-bit”). We enumerate
every reduction in the current bound proof, its field, degree, number of
rounds, batching checks, and repetitions, then derive the overall acceptance
error under the current Goldilocks challenge field. All counts are derived
from the verifier code and the real layer-10 A/B/D/C statement dimensions;
they are charging-rule estimates, not an attack measurement (the gist
explicitly says an exact attack cost must not be declared from field size
alone).

## 1. Fields and what “90-bit” currently covers

- Base field `F = Goldilocks`, `p = 2^64 − 2^32 + 1 ≈ 2^64`.
- WHIR works over the quadratic extension `EF = BinomialExtensionField<F, 2>`,
  `|EF| ≈ 2^128` (`crates/zkie-core/src/pcs/whir.rs`, `type EF`).
- `security_level = 90` is a **per-WHIR-opening** configuration
  (`Whir::try_with_params`, WHIR capacity-bound analysis and PoW). It does
  not include the outer GKR/sum-check/LogUp reductions, and it does not cover
  the union over many openings in one proof.
- `to_ef_point` embeds each base-field coordinate into EF. This changes the
  arithmetic field but adds **no entropy**: a challenge sampled in `F` has at
  most 64 bits of randomness even when consumed as an EF element.

The current verifier policy (S8, `f67b807`) fixes outer WHIR reconstruction to
`VERIFIER_SECURITY = 90`, rejects proof-carried parameters that would lower
it, and caps `num_variables ≤ 30`, `pow_bits ≤ 128`. It does **not** yet
enforce an aggregate error budget over the outer reductions.

## 2. Charging rules

For a non-zero polynomial of total degree `d` over a field of size `q`, one
uniform challenge hits a root with probability at most `d/q`
(Schwartz–Zippel). We charge every randomized check as follows:

| check | charge |
|---|---|
| Sum-check / virtual sum-check, `r` rounds, max degree `d` | `r·d / q` |
| Random linear combination / batching of `N` equations (coefficient γ) | `N / q` |
| Final deterministic identity (transcript replay, point/value equality, zero-denominator test) | `0` |
| WHIR opening at configured security `s` | `2^-s` |
| Merkle commitment collision / hash binding | assumed; not counted (trusted hash assumption) |

The transcript is a Fiat–Shamir random-oracle heuristic: challenges are
treated as independent uniform. This is the standard ROM heuristic used by
the existing S1–S4 arguments; it is not a proof, and the budget below is
conditional on it.

The union bound over all randomized checks in one proof is the sum of the
charges. Over the A/B/D/C statements, the acceptance probability of a false
composite statement is bounded by the sum over statements (each statement is
verified independently).

All outer reductions currently sample in `F`, so `q = p ≈ 2^64`. Only WHIR
openings use `EF` and the configured WHIR security.

## 3. Primitive inventory

Degrees are the maximum total degree of the polynomial represented by each
proof system's `terms`; `a` is the padded multilinear arity (log2 domain).

| # | primitive | field | degree | rounds | batching | charge per call |
|---|---|---|---|---|---|---|
| 1 | `sumcheck` (MatMul `f·h`) | F | 2 | `log2(k)` | — | `2·log2(k)/p` |
| 2 | `verify_virtual` (eq-weighted relation) | F | ≤3 | `a` | — | `3a/p` |
| 3 | `verify_same_poly` | F | 2 | `a+1` | `N` claims | `2(a+1)/p + N/p` |
| 4 | fractional/LogUp layer `li` | F | 3 | `li` | 2 merges × 2 claims | `3li/p + 2·(2(li+1))/p + 4/p` |
| 5 | TopK tie/rank/range virtuals | F | ≤3 | `a` | — | `3a/p` each |
| 6 | Relu/Gelu range decomposition | F | 2 | `a_r` | — | `2a_r/p` each |
| 7 | Scale/ScaleVec remainder virtual | F | ≤3 | `a` | — | `3a/p` |
| 8 | Gather/GatherMatMul routing + sum virtuals | F | 2 | `a` | — | `2a/p` each |
| 9 | deep-O7 group identity virtual | F | 2 | `a_table` | — | `2a_table/p` |
| 10 | batch-opening `W_t` sum-of-products | F | 2 | `d` (point arity) | — | `2d/p` |
| 11 | WHIR opening (aux / weight / table / selector / token) | EF | — | — | — | `2^-90` per opening |
| 12 | Merkle/hash binding | — | — | — | — | assumed |

### 3.1 Exact per-op virtual counts (from the verifier code)

- **TopKSelect** (`verify_reverse_topk`): 15 virtuals at arity `a`:
  `bools, row_sums, gate, d1, d2s, att_sel, att_gap, broadcast,
  x_range, thr_range, tie_zero, tie_bits, tie_rank, tie_row, tie_less`.
  Charge `≤ 15·3a/p` (conservative degree 3; several are degree 2).
- **Scatter** (`verify_reverse_scatter`): 8 virtuals at arity `a`:
  `bools, mask_tok, mask_exp, counts, order_key_cons, order_key_bits,
  order_d_rel, order_asc`. Charge `≤ 8·3a/p`.
- **Gather** (`verify_reverse_gather`): 2 virtuals at `a`
  (`routing, sum`). Charge `4a/p` (degree 2).
- **GatherMatMul** (`verify_reverse_gather_matmul`): MatMul sum-check plus
  `routing` and `sum` virtuals. Charge `2·log2(k)/p + 4a/p`.
- **Projection**: MatMul sum-check plus booleanity virtual at `a`.
  Charge `2·log2(k)/p + 3a/p`.
- **Relu**: two virtuals (`output` at arity `a`, `range` at `a_r`).
  Charge `2a/p + 2a_r/p` (degree 2).
- **GeluIndex**: upper and lower Relu reductions (2× the Relu charge) plus
  the 25-bit `x_range` virtual at `a_r`. Charge
  `4a/p + 7a_r/p` (degree-3 conservative for `x_range`: `4a + 4a_r` from the
  two Relu proofs plus `3a_r`).
- **Scale (shift > 0)**: remainder/booleanity virtual at `a`. Charge `3a/p`.
- **ScaleVec (shift > 0)**: product virtual at `a` (degree 3), booleanity
  is folded into the gamma-weighted terms, plus one `same_poly` bind with
  arity `a+1`. Charge `3a/p + 2(a+1)/p + 2/p`.
- **Lookup** (ungrouped or group member): the witness-side fractional tree
  has depth `t_w`; the table-side tree has depth `t_t`. A fractional tree of
  depth `t` charges
  `F(t) = Σ_{li=0}^{t-1} [3li + 4(li+1) + 4] = 7t(t−1)/2 + 8t`
  field-scaled round checks (the `+4` per layer is the two `same_poly`
  batching checks; the two merges each have two claims).
  Plus the static-group identity at arity `t` (`2t/p`) and the group table
  tree once per static table.
- **Deep-O7 group relation**: the division-free relation check is
  deterministic; the randomized parts are the group identity virtual and the
  shared table fractional tree (already counted above), plus one group-level
  table opening and one aggregate opening (WHIR).
- **Batch opening** (`verify_batch_open_committed`): one sum-of-products
  sum-check of degree 2 over the point arity `d` (`2d/p`) + one WHIR opening.

## 4. Worked example: real layer-10 D proof (full 64-expert FFN)

Dimensions: `s_pad = 8`, `H = 2048`, `MOE_PAD = 2048`, 64 experts; activation
table 2048, real silu table `2^24`; up/down weights `2^28` each; two Lookups
(activation and silu), each idx column length `s_pad·H = 2^14`.

Using the charging rules above:

| component | count | charge (weighted rounds) |
|---|---|---|
| Lookup witness trees (2 × depth 14) | `2·F(14)` | 1,498 |
| Activation table tree (depth 11) | `F(11)` | 473 |
| Silu table tree (depth 24) | `F(24)` | 2,124 |
| Group identities (arity 11 + 24) | `2·11 + 2·24` | 70 |
| Lookup padding virtuals (multiplicity arity 14) | `2 × 3·14` | 84 |
| GatherMatMul up + down (MatMul 11 + 2 virtuals ×3) | 2 × (22 + 6) | 56 |
| Scale16 up + out (arity 14) | 2 × 28 | 56 |
| GeluIndex (2 Relu + x_range; a=14, a_r=19) | 4·14 + 7·19 | 189 |
| ScaleVec shift 16 (arity 14) | `3·14 + 2·15 + 2` | 74 |
| Weight batch openings up/down (arity 28) | 2 × 2·28 | 112 |
| **D total (base field)** | | **≈ 4,736** |

`ε_D^base ≤ 4,736 / p ≈ 2^-51.8`.

Segments A/B/C are smaller (estimated `2^-55` … `2^-57`): A is dominated by
15 TopK virtuals at arity 6 and 8 Scatter virtuals at arity 3; B by the
MatMul sum-check (11 rounds), the GatherMatMul virtuals, and one weight
batch opening at arity 28; C by TopK (arity 5) + Scatter (arity 3) + Gather
(arity 3) + Scale/broadcast virtuals. Summing the four independent statement
acceptance errors gives a conservative whole-chain estimate

```
ε_base(total) ≲ 2^-51.5   (dominated by D)
```

with the caveat that the exact counts should be instrumented in step 2 and
the formula re-run. The important invariant is the shape, not the last digit:
`ε_base ≈ (weighted round count) / 2^64`, and the weighted count is in the
low thousands, so the outer protocol is roughly **2^-52**, not 2^-90.

## 5. WHIR openings and the union bound

Each opening is configured at `security_level = 90`. A proof performs many
openings: per aux descriptor (projection bits, scale remainders, routing,
TopK witness/thr/range/thr-range/tie, scatter routing/order, gather routing,
Gelu range, ScaleVec remainder), per weight batch, per group aggregate/table,
per Argmax selector position, and the token binding. For D the count is
`K ≈ 12–20`; larger models or statements can have more. The union bound over
openings is

```
ε_PCS ≤ K · 2^-90.
```

For `K = 16`, `ε_PCS ≈ 2^-86`; for `K = 256`, `≈ 2^-82`. This is already
below 90 bits even before the outer reductions. To claim an overall 90-bit
PCS bound, the per-opening WHIR security must be set to
`90 + ceil(log2 K_max)`, which increases PoW bits (and some proof work) but
not the outer budget.

## 6. Current overall assessment

```
ε_total ≲ ε_base + ε_PCS + ε_FS/hash
        ≲ 2^-51.5 + 2^-85 + assumed hash term
        ≈ 2^-51.5
```

The bottleneck is the **base-field outer reductions** (sum-check, virtual
relations, LogUp, tie/range, batch-opening sum-check): the weighted round
count is about `2^12.2`, so `64 − 12.2 ≈ 51.8` bits, i.e. about 2^-52.
WHIR's 90-bit extension
field setting cannot lift this; increasing WHIR PoW cannot compensate for an
unrelated outer error (gist S9). Therefore, **the current stack is not an
end-to-end 90-bit argument**, even though the PCS is configured at 90-bit.
This is a budget statement, not a demonstrated forgery.

## 7. Verifier-enforced parameter policy (current and proposed)

Currently enforced (S8 `f67b807`):

- WHIR reconstruction security fixed at 90 for aux/weight paths; proof-carried
  WHIR parameters (`WhirParams`) are policy-gated and cannot lower it.
- `num_variables ≤ 30`, `security_level ∈ [min, 256]`, `pow_bits ≤ 128`,
  `pow_bits ≥ required_pow_bits(num_variables, 90)`.
- Proof byte cap 512 MiB, bounded decode, aux/step caps, checked dimension
  arithmetic.

Proposed for step 2 (not implemented here):

- Aggregate budget cap: `W = Σ_i (r_i·d_i + c_i) ≤ 2^26` per proof (and per
  statement), checked by the verifier from the verified op dimensions and aux
  specs. Under EF challenges this gives `≤ 2^-102`; under the current base
  field it caps the damage at `≤ 2^-38`.
- Opening-count cap `K ≤ 2^12`, with per-opening WHIR security
  `90 + ceil(log2 K_max)` when a whole-protocol 90-bit claim is desired.
- Keep S8 caps/limits as the resource envelope; the budget cap is an
  additional security parameter, not a replacement.

## 8. Mitigation options for step 2 (analysis only — no decision)

### Option A: move outer challenges to EF

Change the outer sum-check/virtual/same_poly/LogUp/tie/range challenges (and
the batching γ) to be sampled in `EF` while tensors and evaluations remain in
`F`. `to_ef_point` already exists at the PCS boundary; MLE evaluation over `F`
at an EF point is a linear combination of base-field values with EF weights.
Charging uses `|EF| ≈ 2^128`:

```
ε_base ≈ W / 2^128,  W ≈ 2^12 → ≈ 2^-116.
```

with the proposed `W ≤ 2^26` cap giving `≥ 2^-102`. Interfaces affected:
sumcheck round polynomials and final evaluations (EF), same_poly merged point
and value (EF), LogUp α/β/γ and leaf claims (EF), transcript absorbs EF
elements (2 base-field lanes for a quadratic extension), and the point
conversion at every PCS call. Tensors/commitments can stay in `F`.

Costs to measure in step 2: prove/verify wall time, proof size for outer
round messages/final evals (roughly 2× those fields), and transcript
sensitivity. WHIR proof sizes dominate total bytes, so the relative size
increase should be small; arithmetic cost is expected to be a small multiple
(quadratic-extension multiplication is ~3 base multiplications).

### Option B: independent repetition

Run the full outer proof `k` times with independent Fiat–Shamir transcripts
(i.e. independent verifier randomness), and require all `k` instances to
accept. Under independence, a false statement with per-run error `ε` has
overall error `ε^k`:

```
ε ≈ 2^-52, k=2 → ≈ 2^-104.
```

This doubles proof size, prover time, and verifier time for the outer parts
(WHIR openings can be shared only if they are statement-identical; sharing
would reintroduce correlated randomness). It does not require changing the
field interfaces. The gist’s warning applies: duplicating bytes or reusing
the same transcript is not independent repetition and proves nothing.

### Option C: combination / hybrid

If step 2 wants a margin at lower arithmetic cost, EF challenges for the
high-count reductions (fractional/LogUp/range trees) plus independent
repetition for the remaining scalar checks is possible, but must be specified
as one budget with independent randomness, not “repeat twice”.

PoW cannot replace either option: it is a PCS grinding cost, not a remedy for
the outer reduction error.

## 9. Adjacent S8 observation (out of S9 scope)

The expanded S8 malformed-proof fuzz (commit `06dedb5`, tests only) showed
that a small number of single-byte mutations are still accepted. The affected
bytes identified so far are proof data the verifier deliberately does not
consume in these paths:

- `SamePolyProof.proof.final_evals` — the verifier derives the endpoint weight
  in closed form and never reads these stored evals (S4 design). This matches
  the repeated small offsets inside the reverse steps of the O7 fixture.
- `BoundExecutionProof.token_binding` — ignored for `PublicStatement::Generic`
  (it is checked only for `LanguageModel`); this matches the large accepted
  block at the end of the serialized `Generic` proof.

This is **non-canonical proof encoding (malleability), not an extra
acceptance path**: no verified relation depends on those bytes. It is recorded
here so step 2/TL can decide whether to canonicalize/reject dead fields in a
separate hardening item; it does not change the budget above.

## 9b. Step 2a status — EF core layer (committed separately)

The TL approved Option A and the step-2 hardening list. Step 2a introduces the
extension-field outer layer without changing any existing proof/transcript
bytes:

- `crates/zkie-core/src/common/ef_outer.rs`: EF sum-check (degree-2 round
  polynomials, `f·h` terminal), EF virtual sum-check (degree ≤3 terms,
  `rounds: Vec<Vec<Ef>>`, `final_evals: Vec<Ef>`), EF `same_poly` (gamma and
  merged point/value in EF), EF equality-polynomial helpers, and
  `mle_eval_at_ef` (base-field MLE folded with EF weights).
- `Transcript::absorb_ef` absorbs each EF element as its two base-field lanes;
  `Transcript::ef_field` samples two independent base squeezes. The
  `ChallengeSource` trait gained default `ef_field`/`ef_field_vec` so the test
  PRNG remains usable.
- EF `verify_virtual` now binds the carried `final_evals` against the
  replayed final claim (`virtual_eval(terms, final_evals) == final_claim`),
  closing the same dead-field malleability on the EF path.
- Tests: 6/6 (`sumcheck_ef_roundtrip_and_tamper`,
  `virtual_ef_roundtrip_and_tamper`, `same_poly_ef_roundtrip_and_tamper`,
  `ef_mle_eval_matches_base_on_subfield_points`, `ef_wire_encoding_is_two_base_lanes`,
  `transcript_ef_absorption_is_deterministic_and_bound`). Existing ops suite
  175/0; core suite re-run for the step baseline.
- Not yet wired into composers: the base-field sum-check/virtual/same_poly
  remain the production path until step 2b/2c. No proof bytes changed.

## 9c. Step 2b-1 status — EF MatMul / W_t / LogUp core

Step 2b is split into sub-steps because the composer migration touches every
outer reduction. Sub-step 2b-1 adds the remaining EF algebra cores, still
without changing any production path or proof byte:

- `common/ef_matmul.rs`: EF MatMul sum-check (`claimed`, round messages and
  terminals in EF; A/B/C tensors stay F, restricted with
  `partial_eval_at_ef`; C evaluated with `mle_eval_at_ef`).
- `common/ef_batch.rs`: EF sum-of-products sum-check for the batch-opening
  `W_t` arithmetic (weights/values EF; `b_eval` EF).
- `common/ef_logup.rs`: EF fractional/LogUp chain with EF α/β/γ and leaf
  claims; EF witness/table/permutation column builders; EF leaf helper;
  split witness/table identity and zero-root-denominator rejection.
- `ef_outer.rs`: `prove_sumcheck_ef`, `partial_eval_at_ef`, and
  `prove_same_poly_ef` (same-poly over EF-valued fraction columns).

Tests: 10/10 EF (`ef_` filter; adds MatMul, batch sum-of-products, EF LogUp
roundtrip/tamper, lookup split identity + zero-den rejection). Existing ops
175/0 and the base-field protocols are untouched; no proof bytes changed.
Composer wiring, EF PCS boundary/batch-opening wrapper, budget caps,
dead-field canonicalization, and A/B/D/C measurements remain for 2b-2/2b-3.

## 9d. Step 2b-2a status — EF PCS boundary and batch opening

Sub-step 2b-2a adds the EF end of the committed-data boundary; tensors and
Merkle roots stay base-field:

- `Whir::open_ef` / `verify_ef`: open/verify a base-field committed MLE at an
  EF challenge point (LSB→MSB coordinate reversal, no extra randomness). The
  returned evaluation is EF and equals `mle_eval_at_ef` of the F table.
- `Whir::open_batch_multi_ef` / `verify_batch_multi_ef`: the same for a batch
  of tables opened at one EF point.
- `pcs/ef_batch_open.rs`: `EfBatchOpeningProof` plus
  `prove/verify_batch_open_committed_ef`. Fiat–Shamir absorbs the commitment
  and EF points/claims as two-lane values, γ is an EF squeeze, the
  sum-of-products `W_t` sumcheck is EF (`ef_batch`), the WHIR opening runs at
  the EF challenges, and the verifier reconstructs `Σ W_g(r)·eval_g` in EF.
- Test: honest roundtrip, duplicate-table grouping, EF open matches EF MLE
  evaluation, tampered sumcheck/evals rejected.

Existing base-field paths remain untouched; no proof bytes changed. Composer
claim/reverse migration, budget caps, dead-field canonicalization and A/B/D/C
measurements remain for 2b-2b/2b-2c.

## 9e. Step 2b-2b status — composer reverse layer on EF

Sub-step 2b-2b migrates the composer's outer claim/reduction layer to EF.
The committed data plane is unchanged: tensors, Merkle commitments, table
contents and public execution values stay in the base field.

- `EfClaim` (tensor id, EF point, EF value) replaces `EvalClaim` across
  `reverse_setup` / `prove_reverse_dag` / `verify_reverse_dag` and every
  per-op reverse prover/verifier (Gather, TopKSelect, BroadcastScale,
  ScaleVec[Shifted], Scatter, GeluIndex, Scale, Argmax, GatherMatMul,
  Lookup, Projection, Relu, and the reduction steps).
- `ReverseReduction` scalar variants (Add, CombineIdx, IdxBroadcast,
  PadIdx) and all reverse proof structs carry EF points, claimed values and
  sumcheck terminals. Round messages use the EF cores from 2a/2b-1;
  `common/ef_batched.rs` adds the per-position batched MatMul reduction and
  `StreamedWeightsEf` streams the folded weight matrix in EF.
- The LogUp fractional chain (α, β, γ, per-layer claims and leaf values),
  tie/rank/signed-range/remainder relations, and the O7 deep weight groups
  are EF throughout.
- `execution_seeds` lifts each public F tensor by evaluating it at the EF
  challenge point (`mle_eval_at_ef`), producing EF claims; transcript
  absorption of EF values uses the two-lane encoding from 2a.
- PCS boundary: aux, weight-batch, selector and token openings all use
  `prove/verify_batch_open_committed_ef`. Openings are checked against the
  original base-field commitments at the EF challenge points, so EF proofs
  cannot re-point a base commitment; the verifier never trusts an EF value
  without replaying the corresponding opening.
- The legacy committed-DAG verifier was already permanently closed; its
  dead inner verifier is deleted in this step. The legacy prover keeps a
  base-field token-evaluation helper because its output is never accepted
  (the byte entry rejects before decoding).
- The S8 verifier caps run unchanged before any EF challenge derivation.
  All S1–S8 negative and fuzz tests pass unchanged, including the
  `s8_*_reject_without_panic` wire-mutation tests.

Verification: `cargo test -p zkie-core --release` 88 passed / 1 ignored;
`cargo test -p zkie-ops --release` 175 passed (all S1–S8 negatives included);
`cargo check --workspace --release` clean. Verifier budget caps, dead-field
canonicalization and the real A/B/D/C before/after measurements remain for
2b-2c/2b-2d.

## 9f. Step 2b-2c status — verifier caps and dead-field canonicalization

Sub-step 2c makes the S9 budget verifier-enforced and removes the remaining
wire malleability.

- `ProofBudget` walks the verified proof structure and accumulates an upper
  bound on `W = Σ(r·d + c)`: every EF virtual/sumcheck round is charged at
  degree 3, same-poly and batch sum-of-products at degree 2, fractional trees
  per layer (including both merges and their batching), permutations on both
  sides, and each opening's sum-of-products rounds. `verify_committed_shard_
  dag_bound` rejects `W > 2^26` and `openings > 2^12` before any challenging
  or PCS work beyond the structural caps.
- Per-opening security: the protocol's WHIR instances are provisioned at
  `VERIFIER_SECURITY = 96` (= 90 + ceil(log2 64)); the verifier additionally
  requires `security ≥ 90 + ceil(log2 K)` for the proof's actual opening
  count `K`. A proof therefore needs `K ≤ 64` under the current provisioning;
  the `2^12` hard cap bounds decoder/PCS work, and statements with more
  openings fail closed instead of degrading below 90 bits. Raising the
  provisioned constant to 102 would cover the full 2^12 cap at higher PoW.
- All protocol-side WHIR constructions (aux, weight commit, table, selector,
  token) use the same 96-bit constant; the weight-cache filename includes
  the security so a 90-bit cache can never be reused with a 96-bit manifest.
- Dead-field canonicalization:
  - base `SamePolyProof` stored final evaluations are not consulted by the
    verifier; the prover now emits them as zero and `verify_same_poly`
    rejects any non-zero value. The prover-side `fractional_leaf` helper
    recovers LogUp leaf values from the merge sum-check and the recorded
    challenge vector instead of the dead field.
  - under a `Generic` statement the token binding is never consulted; the
    prover omits it and the verifier rejects any proof that carries one.
  - EF `EfSamePolyProof` needs no change: `ef_outer::verify_virtual` already
    binds `final_evals` into the checked virtual identity, so the field is
    consumed rather than dead.

Verification: `cargo test -p zkie-ops --release` 176 passed / 0 failed (all
S1–S8 negatives and S8 fuzz included with the caps active; the new
`s9_budget_caps_cover_honest_fixture` asserts the honest fixture is inside
both caps and inside the provisioned security); `cargo test -p
zkie-core --release` 88 passed / 1 ignored, plus the `same_poly` suite
10/10 including the new `s9_nonzero_final_evals_rejected`;
`cargo check --workspace --release` clean. Because WHIR security changed
90→96, all proof bytes change; the new baseline and EF cost measurements
are step 2d.

## 9g. Step 2d — large-weight sub-table split and 96-bit A/B/D/C baseline
(historical; superseded by WAVE-1 A1's 97-bit allocation in §9h)

**Why the split.** The first 96-bit full64 attempt showed that the cost is
the WHIR **opening** grind, not commit/cache: B's 7,195.5 s proof time was
7,180.3 s of `open_ms`, while the 6.4 GB cache commit took under 3 s and the
restore 3.26 s. A single 2^28-slot opening at 96 bits is ~2 h wall on this
machine, so D (two such openings) would be ~4 h and the run was terminated
at 174.25 cumulative CPU-hours under the TL's 168-hour abort threshold. That
measurement motivated the boss-approved split below.

**Split construction.** Every committed weight tensor longer than 2^24 is
split into independent 2^24-element sub-tables, each with its own
commitment/root and opening (chunk metadata `chunk_bits`/`chunk_index` in
the manifest and cache). For a full claim at `r = (r_low, r_high)` the
verifier reassembles

```
f(r) = Σ_h eq(r_high, bits(h)) · f_h(r_low)
```

using weights it recomputes from the public claim point. Each `f_h(r_low)`
is the PCS-authenticated opening value at the chunk point; the verifier
checks both every opening and the weighted sum back to the original leaf.
Unsplit claims keep the exact-value check. The prover materializes each
parent tensor once per proof; the cache key still includes the WHIR
security. `s9_weight_subtable_split_grouping_reassembles_mle` exercises the
16-chunk math (including missing/duplicate-chunk rejection) without a large
PCS run.

Real full64 measurement (DeepSeek-V2-Lite layer 10, seq=1, 64 experts; same
495 GB / 64-core machine). "Before" is the recorded S7 baseline
(`docs/s7-full-continuation.md`: base-field outer challenges, WHIR 90).
"After" is this branch (EF outer challenges, WHIR 96, split sub-tables).
`K`/`W` are the verifier's `ProofBudget` values printed by the r3 run;
`req` is `90 + ceil(log2 K)`.

| phase | before bytes | after bytes | before prove | after prove | before verify | after verify | K | W | req | after peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 241,874 | 290,618 (+20.1%) | 0.15–0.18 s | 0.134 s | not separately timed | not separately timed | 7 | 2,936 | 93 | — |
| B | 540,239 | 2,929,073 (+442%) | 241.4 s | **560.0 s** | 159 ms | 179.1 ms | 22 | 3,090 | 95 | 4.81 GB |
| D | 1,779,403 | 6,641,018 (+273%) | 487.7 s | **1,170.3 s** | 339 ms | 400.3 ms | 48 | 9,102 | 96 | 10.92 GB |
| C | 355,183 | 408,297 (+14.9%) | 204 ms | 276.8 ms | 15.5 ms | 17.3 ms | 11 | 3,676 | 94 | — |

The full A/B/D/C run takes ~29.5 min (B 560 s + D 1,170 s + A/C/negatives),
versus >4 h for a single unsplit 96-bit D proof. The price is proof size:
independent sub-table roots/openings grow B 5.4× and D 3.7× (A/C grow only
from EF/96). Each nv24 opening grinds ~18 s; the remaining cost is the sum
over 32 D sub-openings, not one huge grind. Peak RSS improves because each
opening only holds a 2^24 table: B 16.0 GB → 4.81 GB and D 20.1 GB →
10.92 GB (before values from `docs/s7-full-continuation.md`).

**Determinism.** Three clean runs — r1 (unoptimized parent materialization),
r2 (parent cached once per proof), r3 (final code with budget logging) —
produce byte-identical dumps for all four phases:

| proof | bytes | SHA-256 (all three runs) |
|---|---|---|
| A | 290,618 | `a5257cae12316074763dea63e461b7d35bc87ba39d215d51b925ab68ee14de60` |
| B | 2,929,073 | `c5c5b49d366b25d73e669bf21ed35a57181252d7fcba5dd54fa40431345b4313` |
| D | 6,641,018 | `7015aa86d7f0d0d6e986ef9803308a687c05c198b9acc0b3139193a35d5e05fb` |
| C | 408,297 | `03227ea90d5e1c8aa392c404ed3462defeab142d59f5b928b79cee823fd3f566` |

All real-data negatives are rejected (tampered A gate/scores, B routing/g,
wrong expert/activation roots, D out/g/mask/swapped/dropped/up-down/silu
roots, C d/scores/out/gate), and the harness ends with `S7-CHAIN-OK`.

**End-to-end ≥90-bit argument for accepted proofs.**

- Outer reductions (sum-check, virtual, LogUp/fractional, tie/rank/range,
  same-poly, batch-opening sumcheck) sample EF challenges: `|EF| ≈ 2^128`.
  The verifier caps the conservative weighted budget `W = Σ(r·d + c)` at
  `2^26`; therefore `ε_outer ≤ W/2^128 ≤ 2^-102`. The real chain is far
  below the cap (D: W = 9,102 ≈ 2^13.2 → `ε_outer ≤ 2^-114.8`).
- PCS: each opening is provisioned at 96 bits, and the verifier rejects a
  proof unless `96 ≥ 90 + ceil(log2 K)` for its actual opening count K ≤ 64
  (hard wire cap `K ≤ 2^12`; larger K fails closed rather than weakening the
  bound). Thus `ε_PCS ≤ K·2^-96 ≤ 2^-90`. Real counts:
  A 7 → 2^-93.2, B 22 → 2^-91.5, D 48 → 2^-90.4, C 11 → 2^-92.5.
- Union: `ε_total ≤ 2^-102 + 2^-90 + ε_FS/hash < 2^-89.99`, i.e. at least
  90 bits up to the Fiat–Shamir/hash assumption used throughout S1–S8.
  Raising the provisioned constant to 102 would cover the full `2^12`
  opening cap while keeping the same 2^-90 PCS bound.

Verification: `cargo test -p zkie-ops --release` 177 passed / 0 failed (all
S1–S8 negatives, S8 fuzz, the 2c budget test, and the new split-math test);
`cargo test -p zkie-core --release` 88 passed / 1 ignored; `cargo check
--workspace --release` clean. S9 is not accepted pending the TL's final
review; this document does not declare acceptance.

## 9h. WAVE-1 A1/A2 — whole-protocol error allocation and γ-merge accounting

PR#45 A1/A2. This section supersedes the 96-bit totals in §9g; §9g remains
as the historical 2d measurement. No reduction or transcript semantics
changed — only the security constant, the verifier budget/policy checks, and
the γ-merge charge.

### A1 — exact allocation

- `|EF| = (2^64 − 2^32 + 1)^2`
  `= 2^128 − 2^97 + 2^65 + 2^64 − 2^33 + 1`
  `= 340282366762482138490186164457219031041` (exact, fits u128; the code
  never approximates it by 2^128).
- `VERIFIER_SECURITY = 97`. Effective opening cap `K ≤ 64`; the existing
  `MAX_PROOF_OPENINGS = 2^12` remains only a wire/resource prefilter.
- Two explicit halves plus the exact combined bound:
  - PCS: `ε_PCS ≤ K·2^-97 ≤ 64·2^-97 = 2^-91`.
  - Outer + γ-merge: `S = weighted + merge ≤ 2^26`, hence
    `ε_outer+merge ≤ S/|EF| ≤ 2^26/|EF| < 2^-101`. The verifier also checks
    the half explicitly as `S·2^91 ≤ |EF|`.
  - Combined, computed in exact integer arithmetic:
    `K·2^-97 + S/|EF| ≤ 2^-90` ⇔ `K + S·2^97/|EF| ≤ 128`. The code checks
    `K + ceil(S·2^97/|EF|) ≤ 128` (ceiling is stricter); checked u128
    arithmetic, overflow → reject. This does not rely on the halves being
    independently acceptable.
- Propagation: all aux/weight/table/selector/token WHIR constructions use
  `VERIFIER_SECURITY`; the weight-cache filename includes `s97`; O7 static
  lookup commits are rebuilt in memory at 97 bits; `trusted_batch_whir`
  rejects manifests below 97, and a wire-level test mutates a manifest batch
  to 96 and requires rejection. Legacy 96-bit caches/manifests cannot be
  reused.
- Tests: exact field constant; K=1/63/64 pass; K=65 rejected by the
  effective cap; K=127/128 exercise the exact combined bound; W cap and
  outer half boundaries; 96-bit manifest rejected; honest fixture inside all
  caps. `ops` 181/0 after the additions (including the tie_zero/tie_less
  degree-charge regression).

97-bit real full64 rerun (r4, final corrected budget code; same
harness/inputs; EF outer, split sub-tables):

| phase | bytes | commit+prove | verify | K | weighted | merge | S | required | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| A | 291,042 | 0.137 s prove | — | 7 | 7,122 | 592 | 7,714 | 93 | — |
| B | 2,937,745 | 602.1 s | 179.4 ms | 22 | 21,074 | 1,231 | 22,305 | 95 | 4.81 GB |
| D | 6,645,146 | 1,248.8 s | 403.7 ms | 48 | 16,806,938 | 2,551 | 16,809,489 | 96 | 10.92 GB |
| C | 410,385 | 285.5 ms | 17.7 ms | 11 | 8,614 | 689 | 9,303 | 94 | — |

D's `weighted` is dominated by the A2 LogUp α/β term for the real 2^24 silu
table (`N = 2^24`); `S ≤ 2^26` still holds. For D:
`ε_PCS ≤ 48·2^-97 = 2^-91.42`, `ε_outer+merge ≤ 16,809,489/|EF| < 2^-103.9`,
so the exact total is below `2^-90` with margin; A/B/C are smaller
(`K = 7/22/11`).

All real-data negatives remain rejected and the harness ends with
`S7-CHAIN-OK`. Raising the per-opening security 96→97 changes proof bytes
(A/B/D/C ≈ +424/+8,672/+4,128/+2,088 bytes) and adds ~7–8% prove time on the
large phases (B 560→605 s, D 1,170→1,259 s); the split keeps the total run in
the tens of minutes. Four 97-bit runs (r1–r4; r4 is the final corrected
budget code with per-call degree charging) produce byte-identical dumps:

| proof | SHA-256 (all four 97-bit runs) |
|---|---|
| A | `f857facf7d3001ff86e508c9a98b617dad88f59531f140149b250087b4983a9e` |
| B | `cb5ea97d6c95a95e2e15cdaef038ce33a96d3ead44e9b32fddb4888650e3cbd1` |
| D | `5e99fa7690b3cd756a1a2ee1d47f1a48449e8c523b087d934a6e91383e7cdd3f` |
| C | `9c3194577fa1f180772df4e118b3b02af7f772ec75cdcb5bef14975c7ce0d5b7` |

### A2 — batch-opening γ merge and budget audit

`ProofBudget::opening` now charges, per batch opening, the γ-power merge
polynomial degree `n−1` in addition to the sum-of-products sumcheck and the
one PCS opening, where `n` is the number of claims in that opening (points =
claimed values). Examples: A/B/D/C carry `merge = 592/1,231/2,551/689`
weighted units. Zero/one-claim openings add zero. Counts saturate/reject on
overflow, and the verifier's existing length/index/point checks reject
malformed claim sets before the opening is accepted. The same audit also
charges the γ-separated term batching inside each virtual relation with the
true invariant **γ-degree ≤ 4·|final_evals|** (verified for all 33
constructors; this is not a term-count bound — `scatter_counts_terms` has
m·n terms but γ-degree 0 and x-degree 2, covered by the default 3), and the
LogUp α/β collision term `N = max(index length, table length)` per Lookup op
(manifest/offline lengths; accepted graphs materialize these shapes during
aux validation). The per-round x-degree is charged per call site: default 3,
`tie_zero` 4 (4-factor terms), `tie_less` 5 (5-factor terms); all other
constructors have max_deg ≤3.

Verifier call ↔ budget item audit:

| verifier work | budget charge | coverage |
|---|---|---|
| EF virtual/sumcheck rounds | `deg·rounds + 4·|final_evals|`; `deg` is the real per-round x-degree at the call site (default 3, `tie_zero` 4, `tie_less` 5). The γ invariant is **γ-degree ≤ 4·|final_evals|** for every production constructor (not a term-count bound: `scatter_counts_terms` has m·n terms but γ-degree 0 and x-degree 2, already inside the default 3) | `weighted` |
| MatMul / batch sum-of-products | `2·rounds` | `weighted` |
| same_poly merge | virtual charge + `final_evals.len()` | `weighted`; its `final_evals` length is constrained by downstream consumers, so no separate mechanical charge is added |
| LogUp fractional tree | per layer: virtual charge, both same_poly merges, `s.len()` batching, +8 | `weighted` |
| LogUp α/β collision | `N = max(index len, table len)` per Lookup op (manifest/offline lengths) | `weighted` (A2 audit; N ≤ 2^26 by the W cap) |
| tie / range / rank γ-separated terms | covered by the virtual γ-term charge | `weighted` |
| permutation | both fractional sides + `|a|+|b|` | `weighted` |
| batch-opening γ merge (`W_t`, aux/weight/sel/token) | `n−1` per opening | `merge` (A2) |
| WHIR opening | `2^-97` each, K ≤ 64 | `ε_PCS` |
| leaf/public execution evaluation | 0: the public execution tensors are deterministic; the sampled point/value becomes a claim that is reduced/authenticated, not a standalone probabilistic check | — |
| Fiat–Shamir / hash | assumed | stated assumption |

Tests: budget grows with claims at fixed rounds/openings (1/2/64 → merge
0/1/63); same tensor multi-point and duplicate requests are covered by the
weight/sel grouping tests; malformed lengths/points/indices are rejected by
the existing S8 structure-mutation tests; exact total-bound boundaries
(K=127/128) are tested.

## 10. Acceptance status

Step 1 delivers only this inventory and budget. The S9 acceptance items

- auditable end-to-end ≥90-bit parameter basis,
- verifier-enforced parameter caps, and
- measured extension-field cost

are **not yet met** by the current base-field protocol. Step 2, after TL
review, will choose Option A or B (or a specified hybrid), implement the
verifier checks and the size/round caps, measure prove/verify/size costs on
the real A/B/D/C chain, and then submit the end-to-end argument for review.

No protocol, transcript, or proof-byte behavior was changed by this step.

## Post-fix full64 segmented rerun (2026-11-05, same_poly out-binding fixes)

Baseline invalidated by the eight per-op out-witness fixes
(`fix/soundness-repair` 6b4a4a0..abb1751): aux layouts, transcript inputs and
wire bytes changed. Rerun: DeepSeek-V2-Lite layer 10, seq=1, K=6/64 experts,
four-statement segmented shape A/B/D/C, real weights and inputs, on the test
machine, twice with different proof-dump prefixes.

Double-run result: A/B/D/C proof bytes are **identical**; all real-data
negative cases in the harness are rejected; `S7-CHAIN-OK layer=10`.
Command: `ZKIE_S9_BUDGET_LOG=1 ZKIE_S7_DUMP_PREFIX=... cargo run --release -p
zkie-models-deepseek-v2-lite --example prove_core_chain_full64`.

| Statement | prove/commit | verify | proof bytes | K (openings) | W (weighted) | merge | outer (W+merge) | S (required) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 129.1 ms | — | 291,042 | 7 | 7,122 | 592 | 7,714 | 93 |
| B | 596.8 s | 198.1 ms | 2,993,256 | 23 | 21,102 | 1,232 | 22,334 | 95 |
| D | 1282.4 s | 401.5 ms | 6,712,889 | 49 | 16,806,966 | 2,554 | 16,809,520 | 96 |
| C | 287.9 ms (prove) | 20.0 ms | 470,760 | 12 | 8,642 | 690 | 9,332 | 94 |

Peak RSS: B 4,808,732 kB; D 10,957,896 kB (run 1) / 10,958,044 kB (run 2).

Notes: D is at K=49 of the 64 effective-opening cap; the added out-witness
commitments consumed headroom but the segmented statements still verify.
Monolithic Gemma3/DeepSeek B2 remains blocked by the cap (see
docs/b2-evidence/b2-real-model-status.md); O7-deep M1 is the aggregation path.
