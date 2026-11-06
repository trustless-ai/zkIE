# Audit: `same_poly` output-claim binding (CRITICAL, systemic)

2026-11-05. Scope: reverse reductions in `crates/zkie-ops/src/compose.rs`.
Trigger: TL deep audit of RmsNorm (990b709) found Q5 endpoint binding
unsound. TL asked for an immediate systemic self-audit of existing
reductions, no production-code changes, and a conclusion table.

Status: **systemic hole confirmed**. Production code is unchanged in this
commit; only two `#[ignore]` red tests and this document are added.
O7-deep M1 is paused.

## 1. The contract `verify_same_poly` does not enforce

`zkie_core::common::ef_outer::verify_same_poly(proof, claims, transcript)
-> Option<(Vec<Ef>, Ef)>` returns the merged claim `(r, f(r))`.
It verifies the sumcheck that merges values claimed for one polynomial `f`
at several points; it **never checks that those claimed values are values
of the real `f`**, and the returned `(r, f(r))` is authenticated only if
the caller checks it against `f`'s commitment (or `f` is otherwise
public). The comments and the function signature say exactly this.

Consequence: calling `verify_same_poly(...).is_none()` and discarding the
returned claim over an **uncommitted** `f` accepts an arbitrary claimed
value for `f`. For any desired output claim `(point, forged)`, a malicious
prover:

1. keeps the honest virtual/sumcheck proofs;
2. builds `f' = f + δ`, where `δ` is multilinear with
   `δ(point) = forged - f(point)` and `δ(z) = 0` at the virtual endpoint
   `z` (basis `chi_0 = Π(1-X_j)`, `chi_1 = Π X_j`; a 2x2 solve, generalizes
   to any set of distinct claim points);
3. proves `same_poly` over `f'` for the claims `(point, forged), (z, fe)`;
4. the verifier accepts and returns `forged` as the caller's out claim.

If auxiliary proofs follow the bind and are transcript-dependent, the
prover regenerates them under the forged transcript state (the transcript
is public); that is what the RmsNorm red test does.

## 2. Reproducible evidence (red tests, `#[ignore]`d to keep CI green)

    cargo test -p zkie-ops --release --lib audit_ -- --ignored --nocapture

Result on the current commit:

    AUDIT bind_ok=true
    AUDIT FAILURE: forged ScaleVec output claim was ACCEPTED
    test compose::tests::audit_scale_vec_forged_output_claim_must_reject ... FAILED

    AUDIT FAILURE: forged RmsNorm output claim was ACCEPTED
    test compose::tests::audit_rms_norm_forged_output_claim_must_reject ... FAILED

* `audit_scale_vec_forged_output_claim_must_reject`: existing accepted B1
  operator (ScaleVec shift 0). Honest proof + fake interpolating `f'` +
  fake bind; verifier accepts `value + 1`.
* `audit_rms_norm_forged_output_claim_must_reject`: RmsNorm (990b709).
  Same attack, plus the three post-bind LogUp proofs regenerated under the
  forged transcript. Reusing the honest LogUp proofs is rejected — that is
  transcript divergence, **not** binding; a real prover regenerates them.

The tests panic iff the forged claim is accepted, so they are red today
and become green once the fix lands.

## 3. Conclusion table (all reverse reductions)

Verdicts:
* **HOLE** — caller out value bound only by a discarded `same_poly` over
  the uncommitted out MLE.
* **OK** — out value enters a same-point/global equation tied to
  committed or upstream-authenticated data, a committed-aux opening, or an
  explicit claimed sum whose final evaluations are opened against a
  commitment.

| Operator | verify function / same_poly site | out binding | Verdict |
|---|---|---|---|
| BroadcastScale | `verify_reverse_broadcast_scale` / 2188 | discarded merge over out MLE | **HOLE** |
| ScaleVec (shift 0) | `verify_reverse_scale_vec` / 2248 | discarded merge over out MLE | **HOLE** (red test) |
| ScaleVecShifted | `verify_reverse_scale_vec_shifted` / 2351 | discarded merge over out MLE; remainder bits are committed, out is not | **HOLE** |
| ScaleGate | `verify_reverse_scale_gate` / 2497 | discarded merge over out MLE | **HOLE** |
| Softmax | `verify_reverse_softmax` / 2690 | discarded merge over out MLE (`fs[1]`) | **HOLE** |
| StableSoftmaxIndex | `verify_reverse_stable_softmax_index` / 2989 | discarded merge over idx_f/out MLE | **HOLE** |
| RmsNorm | `verify_reverse_rms_norm` / 3876 | discarded merge over out MLE; `RmsNormOutWitness` commits only rem_lo/rem_hi | **HOLE** (red test; F1) |
| RoPE | `verify_reverse_rope` / 4282 | discarded 5-claim merge over out MLE; `v0/v1` are proof-supplied and can be set `= value`, then `f'` interpolates the distinct points | **HOLE** |
| Scale | `verify_reverse_scale` / 5169 | `value` in rounding equation; remainder bits aux-opened + boolean/range; x claim derived | OK |
| Relu | `verify_reverse_relu` / 7863 | explicit claimed sum; final x eval returned as upstream claim, output bit column opened against `bit_commitment` | OK |
| GeluIndex | `verify_reverse_gelu_index` / 5062 | two relu claimed sums; `lower_at_output` pinned by the lower relu; committed range bits | OK |
| Projection | `verify_reverse_projection` / 7763 | `matmul.claimed - rem + half == 2^shift*(value - bias)`; remainder range-proven | OK |
| MatMul | inline / 8678 | `proof.claimed == value` + EF matmul sumcheck | OK |
| Add | inline / 8671 | `av + bv == value`, same point; inputs authenticated upstream | OK |
| Transpose | inline / 8427 | point permutation, value carried | OK |
| CombineIdx / IdxBroadcast / PadIdx | inline / 8611-8670 | trusted-execution replay recomputes the value, or linear same-point relations | OK |
| Lookup | `verify_reverse_lookup` / 7614 | `value` opened against committed pairs column at the claim point; table leaf committed/identity | OK |
| Argmax | `verify_reverse_argmax` / 5476 | outputs opened against committed argmax aux at the claim points | OK |
| Gather | `verify_reverse_gather` / 603 | explicit claimed sum `v`; final evals checked against opened committed routing/selector aux | OK |
| TopKSelect | `verify_reverse_topk` / 1688 | precommitted witness columns; explicit sums/openings | OK |
| Scatter | `verify_reverse_scatter` / 4714 | precommitted witness/index columns; openings | OK |
| GatherMatMul | `verify_reverse_gather_matmul` / 5856 | bmm sumcheck + committed index columns | OK |

Related sites with the same anti-pattern (no discarded per-op same_poly,
but a merge over an uncommitted polynomial):

| Site | Situation | Assessment |
|---|---|---|
| `verify_reverse_dag_inner` / 8420 | multi-consumer merge over `reverse_tensor_values` (uncommitted store tensor); the merged `(p,v)` is checked against `stored.point/value` and used as the step's out claim | Same root cause. Exploitability depends on the consuming reduction: with any **HOLE** op it composes directly; with OK ops the reduction still pins the relation, but multi-consumer consistency is not actually proven. Fix in the same program; add an attack test. |
| `verify_shard_precomputed_inner` / 12262 | claim groups merged over locally built claims with `verify_same_poly(...).is_none()`; labels (points/values) come from verified sumchecks, not from a committed merged polynomial | Merge itself is vacuous; claims have independent sumcheck/opening paths. Exploit not demonstrated; flag for follow-up audit. |
| `committed_cross_bind` / 12611 | every claim is opened against the boundary commitment **before** the merge (S4 comment) | OK: openings are the authentication; merge is redundant consistency. |
| Projection/residual chains / 8922, 9029, 9205, 9217 | merges over `ys/xs/ps` recomputed by the verifier from the committed input `x` | OK: underlying values are verifier-recomputed. |

## 4. Impact

The hole exists in eight reverse reductions, including two already
accepted B1 operators (ScaleVec in the S-series base set; RmsNorm 990b709)
and three accepted WAVE-2 ops (ScaleGate/Softmax/StableSoftmaxIndex, plus
RoPE 2346cf3). Any route that uses these reverse verifiers with a
caller/prover-supplied output seed can have that seed forged while the
returned input claims remain genuine. Whether a specific bundle route
exposes the forged claim depends on how its output seed is bound; that is
a separate audit. The op-level acceptance properties claimed in the
S-series / B1 reviews are not sound until this is fixed.

## 5. Fix directions (for TL/boss decision)

1. **API**: make the authenticated-merge requirement impossible to
   ignore. Either remove the bare `verify_same_poly` from verifier-facing
   code and expose `verify_same_poly_committed(commitment, opening, ...)`,
   or return the merged claim and force every call site to consume it.
   Add a CI lint against `verify_same_poly(...).is_none()` in `zkie-ops`.
2. **Per-op**: commit the out-value column (or its range-decomposed
   limbs) in the aux witness and open it at the merged point `r`, checking
   `f(r) == opened`. For RmsNorm this is TL F1 (out/32-bit decomposition in
   `RmsNormOutWitness`, consume the merged claim) plus F2 (range-decompose
   out via `out_lo/out_hi` limb lookups, unique solution of the rounding
   relation).
3. **DAG merge**: for uncommitted multi-consumer tensors, either commit
   the tensor (boundary commit) or derive the merged claim from the
   verifier's own recomputation; do not accept a merge over store values.
4. **Regression**: enable the two red tests after the fix and add an
   attack test per affected op (same construction; regenerate any
   transcript-dependent post-bind proofs).

## 6. Fix status (2026-11-05)

All eight HOLE reductions are fixed, each with a red->green attack test and
the per-op out witness committed before the challenge:

| Commit | Op |
|---|---|
| 6b4a4a0 | ScaleVec (shift 0) |
| 2ad9c45 | BroadcastScale |
| 26b4083 | ScaleVecShifted |
| cbaf2ca | ScaleGate |
| c4d8c1b | Softmax |
| d285d52 | StableSoftmaxIndex |
| 9852c59 | RoPE |
| b555a89 | RmsNorm (F1 + F2 + LOW items) |

Follow-up `abb1751`: multi-consumer DAG attack test, CI lint
(`scripts/check_same_poly_binding.py` + `same_poly_discard_lint`), and this
status section. Ops suite: 201 passed, 0 ignored.

The fix pattern is uniform: the out column (or its limbs) is part of a
committed aux batch absorbed into the transcript **before** the virtual
challenge; the `verify_same_poly` merged claim is consumed and opened at
the merged point and at the virtual endpoint, so gamma separation forces
every claim value to equal the committed out column's evaluation.

The sixth B1 operator, LayerNormCentered (`ReverseReduction::Lnc`), is the
first op designed directly on this pattern: its 69-column out witness
(`out, rem_lo/hi, o_lo/hi, 64 bits`) is committed before the challenges and
the merged claim is consumed by both the direct opening and the pooled aux
recorder. Attack/tamper evidence in `docs/lnc-reverse.md`.

RmsNorm F2 deviation from the literal `out_lo/out_hi` wording: RmsNorm's
out is signed, so the witness pins `out + 2^31` to `[0, 2^32)` through two
16-bit limbs (identity-table lookups) and the extra out-virtual equation
`out + 2^31 = out_lo + 2^16*out_hi`. This is the signed-32-bit equivalent
and still gives a unique rounded solution.

Multi-consumer DAG merge: per TL review, the per-op bind now authenticates
the merge transitively; no DAG code change is made. The requested attack
test `audit_multi_consumer_forged_claim_must_reject` (two consumers of one
intermediate, one seed forged) is green. The CI lint
`scripts/check_same_poly_binding.py` and the in-tree
`same_poly_discard_lint` test reject unmarked
`verify_same_poly(...).is_none()` discards; the remaining legitimate sites
carry `LINT-ALLOW` justifications.

Wire/transcript changed for every proof containing the fixed ops, so the
old segmented baselines are invalid and are being re-run (full64 A/B/D/C,
A1 K/W/S, negative matrix, double-run byte determinism).
