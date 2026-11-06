# S7c implementation plan — Scatter ↔ router selector, top-k, uniqueness, gate, real layer

Status of S7 at this commit: **incomplete**. Implemented and reviewed-in-stage:

- GatherMatMul: expert weights + `ExpertRouting` counts (S7a).
- Gather: authenticated one-hot `GatherRouting` token matrix; verifier opens
  committed columns only at transcript-derived points (S7b batch 4).
- Linear index ops (CombineIdx/IdxBroadcast/PadIdx) with exact shapes,
  u32-domain contract and replay/certification rules (S7b batches 1–3).
- **TopKSelect**: wired with pre-committed witness/thr aux, whole-column
  broadcast, attainment and a strict-tie fail-closed boundary (see status
  below; the deterministic smaller-index tie rule is still pending).
- **Fail-closed**: Scatter, Softmax, BatchedMatMul, BroadcastScale, ScaleVec,
  and every other op without a reverse reduction. A `Scatter` graph is refused
  (`s7c_scatter_still_fail_closed`); the gate-weighted tail (BroadcastScale)
  is refused (`s7b_gate_weighted_tail_fail_closed`).

The public-index-replay prototype was removed (review: it requires the
selector/routing to be public and recomputed by the verifier, which the gist
forbids). Everything below keeps the routing virtual and authenticated by
proof reductions.

## Dependency order

### 1. TopKSelect reverse reduction (router authentication) — first

`Op::TopKSelect { x, sel, thr, gate, d1, d2, m, n, k }`. Relations (from
`topk.rs`, same S5 pattern as the Argmax reduction):

- `sel ∈ {0,1}` and `Σ_e sel[t,e] = k` per row (eq-weighted booleanity + row
  sum virtuals, claim k).
- `gate = x·sel` (eq-weighted whole-column virtual; endpoints gate/x/sel).
- `d1 = sel·(x − thr)`, `d2 = (1−sel)·(thr − x)` with `thr` row-broadcast
  (eq-weighted virtuals, as in `verify_reverse_argmax`'s d1/d2).
- `d1, d2 ≥ 0` ranges (d1 aux bit decomposition; d2 range decomposition).
- Returned claims: `x` at the virtual endpoints; outputs `sel/thr/gate/d1/d2`
  authenticated as seeds/downstream claims.

This makes `sel` a *claimed, authenticated* column — the anchor Scatter needs.
No verifier read of the selector table: only virtual endpoints and aux
openings at derived points.

Implementation status (review of 978f3f8):
1. DONE: `TopKSelectWitness`/`TopKSelectThr` aux batches; the eq-weighted
   identity Σ eq(q,pos)·thr_b(pos) = thr~(q_hi) has claim thr~(q_hi) supplied
   by the verified thr-aux opening at q_hi = q[lgn..], proving whole-column
   broadcast. Forged per-cell thr_b witnesses are rejected.
2. STAGED BOUNDARY ONLY: `d2s = (1-sel)(thr_b-x-1)` range-proves unselected
   x < thr, so boundary ties fail closed; the malicious x=[5,5], k=1 selection
   sel=[0,1] is rejected. The FINAL requirement remains the deterministic
   smaller-expert-index tie-break of `topk_forward`, to be generated and
   verified on real tied inputs (positive + negative tests). S7 is not
   complete until this lands.
3. DONE: `att` column (Boolean, one per row, att·(1-sel)=0, att·(x-thr_b)=0)
   proves threshold attainment; with d1/d2s signs this forces thr to be the
   k-th largest. The x=[10,0], k=1, thr=5 counterexample is rejected.
3b. DONE (new P0): the 16-bit field gaps alone do NOT imply the to_i64 signed
   order (A=2^63, B=2^63-1 with A-B=1 mod P). Verifier-authenticated signed
   range proofs (23-bit decomposition of value+2^22) for x and the thr_b
   column, with pre-committed aux bound before seeds, exclude the wrap.
   Contract: x, thr in [-2^22, 2^22); out-of-range fails closed. Real
   DeepSeek extracted activations (layers 1-26, seq=1) give max|logits16| =
   248167 and max Q16 probability 11937, both below 2^22; the real layer
   proof has NOT been run, so no real-model support is claimed.
4. DONE: both TopK aux descriptors/roots are bound in `execution_seeds`
   before output claim points and challenges; the reduction commits nothing
   itself. Regressions assert seed dependence on the witness root.

### 2. Scatter routing ↔ selector reduction

`Op::Scatter { sel, token_id, expert_of, m, n }`, with `sel` carried as the
TopKSelect claim (fail-closed if the producer is unsupported).

- Aux batch `ScatterRouting`: token one-hot matrix `R[t,pos]` (m columns) and
  expert one-hot `X[e,pos]` (n columns) over the padded position domain
  (s = len(token_id), from the manifest exact shape).
- Eq-weighted virtuals (claim 0):
  - booleanity of every R/X column;
  - `Σ_t R[t,pos] = Σ_e X[e,pos] = mask` (mask = 1[pos<s]); padding selects
    nothing;
  - ties `token_id(pos) = Σ_t t·R[t,pos]`, `expert_of(pos) = Σ_e e·X[e,pos]`
    (endpoints become claims on the token/expert columns; the counts side is
    therefore virtual, never read by the verifier).
- Counts/multiset identity (NO position weight): for random
  `q = (q_e, q_t)` over the (token, expert) domain,
  `Σ_pos R'(pos)·X'(pos) = sel~(q_e || q_t)`, where
  `R'(pos) = Σ_t eq(q_t,t)·R[t,pos]`, `X'` likewise. The unweighted sum is
  exactly the MLE of the routing count matrix at `(q_t, q_e)`, so equality
  with the sel claim forces `count == sel` pointwise. Endpoints are aux
  openings at the sumcheck point; the RHS is the returned sel claim, later
  authenticated by the TopKSelect committed column. No `eq(z,pos)` factor and
  no `mask~(z)` division appear.
- Consequences: active-pair uniqueness follows from sel Boolean; `k` from the
  TopKSelect statement gives exactly-k routes; dropped routes fail the
  row-sum/counts identity; duplicates cannot occur in a Boolean selector.
- Outputs token_id/expert_of remain claims whose points are checked against
  the committed routing columns by aux openings (no table read).

### 2a. Scatter reduction protocol (implementation-ready)

`Op::Scatter { sel, token_id, expert_of, m, n }` reads `sel` (field, m*n,
row-major flat index `t*n + e`, e = low bits) and writes the index tensors
`token_id`/`expert_of` (length `s` = number of active pairs, from the trusted
manifest; `scatter_forward` enumerates experts outer / tokens inner). The
verifier never reads the routing arrays.

Aux `ScatterRouting` (pre-committed, like the TopK aux): `m` token one-hot
columns `R_t[pos]` and `n` expert one-hot columns `X_e[pos]` over the padded
position domain (arity `A = max(ceil_log2 s, 5)`), descriptor derived from the
graph + manifest and bound in `execution_seeds` before output points.

Reduction (all eq-weighted, transcript order fixed):

1. Booleanity (claim 0, gamma-combined): `eq*(R_t - R_t^2)` for all m columns
   and `eq*(X_e - X_e^2)` for all n columns.
2. Mask (claim `(g0+g1) * mask~(q2)`, verifier-computable closed form
   `mask~(z) = sum_{pos<s} eq_evals(z)[pos]`): `g0*eq*(sum_t R_t)` +
   `g1*eq*(sum_e X_e) = (g0+g1)*mask~(q2)`. Pins the support to exactly
   `[0, s)` (padding selects nothing) and gives each position exactly one
   token and one expert.
3. Output ties (direct checks at the claim points `z_tok`, `z_exp`, length
   `lb = ceil_log2 s`): `sum_t t*R_t~(pad(z_tok)) == v_tok` and
   `sum_e e*X_e~(pad(z_exp)) == v_exp`. Random points, aux committed before
   them, so the one-hot columns are pinned pointwise to the claimed public /
   downstream token and expert arrays (values in range by Boolean one-hot).
4. Counts/multiset identity (claim `counts_total`, prover-supplied but
   forced by the sumcheck): with `q_t`, `q_e` random and NO position weight,
   `sum_pos R'(pos) * X'(pos) = sel~(q_e || q_t)`, where
   `R'(pos) = sum_t eq(q_t,t) R_t(pos)`, `X'` likewise, implemented as the
   virtual `sum_{t,e} eq_t[t] eq_e[e] * R_t * X_e` with claim
   `counts_total` (the MLE of the count matrix at `(q_t, q_e)`). The verifier
   returns `EvalClaim { sel, q_e || q_t, counts_total }`; no verifier table
   read and no mask division. The claim is then
   authenticated by the producing TopKSelect reduction (its committed sel
   column opened at the same point), which is what rejects duplicated,
   dropped or non-active routes: by Schwartz-Zippel the routing pair multiset
   must equal the authenticated sel support exactly.

Consequences after composition with the TopKSelect reduction: exactly `k`
routes per token; each active `(token, expert)` pair appears exactly once
(positions are distinct, each has a unique pair, and the multiset identity
matches the Boolean sel); duplicate or dropped routes make the counts
identity fail against the TopK sel opening; token/expert values stay in
`[0,m) x [0,n)` by construction. `s` must be a power of two for the outputs
to be public seeds at the bound entry (execution_seeds requires power-of-two
public lengths); otherwise fail closed.

### 2b. Scatter expert-major order proof — IMPLEMENTED

The multiset identity is insufficient (TL review f48fcfd): a permuted routing
array with the same (token, expert) multiset is accepted. The forward contract
requires the active pairs in column-major key order key = e*m + t, and a formal
bound counterexample confirms the pre-fix verifier accepted a swap.

Gadget (compatible with the existing LogUp permutation argument):

1. New pre-committed aux `ScatterOrder` over the padded flat position domain:
   - `k[pos]` = key of the pair at pos (0 on padding), with a B-bit
     decomposition, B = log2(m*n), masked to pos < s. This gives the key range
     [0, m*n) for active positions.
   - `k_next[pos]` = cyclic shift of k: k_next[pos] = k[pos-1] for pos >= 1,
     k_next[0] = k[2^A - 1].
   - `d[pos]` = k_next[pos] - k[pos] - 1, committed, with a masked bit
     decomposition proving d >= 0 for 1 <= pos < s (strict increase on the
     real sequence; padding unconstrained).
   - key-consistency virtual (claim 0): k = m * sum_e e*X_e + sum_t t*R_t,
     endpoints opened from the routing aux; this binds k to the same one-hot
     columns the tie checks bind to the arrays.
   - shift bindings: eq-weighted virtuals k_next - k_shift = 0 over the
     committed columns, where k_shift is the pair-permutation gadget below.

2. Shift via pair-multiset permutation (no shift sumcheck needed): sample gamma,
   build a[p] = k_next[p] + gamma*p and b[q] = k[q] + gamma*((q+1) mod 2^A),
   and prove multiset equality with the LogUp permutation argument
   (prove_permutation/verify_permutation). The deterministic second component
   makes the matching unique, so multiset equality forces k_next[p] = k[p-1]
   pointwise (cyclic). The verifier additionally checks the encodings at the
   fractional leaf points against openings of the committed k/k_next columns
   (the identity-index MLE is a closed form the verifier computes), which binds
   the carried a/b vectors to the commitments (Schwartz-Zippel). This is the
   new primitive to implement and review; it needs aux openings at the
   fractional leaf points but no changes to the sumcheck core.

3. Consequences: the returned pairs are distinct values in [0, m*n), strictly
   increasing, and (by the counts identity + sel Booleanity) have exactly the
   sel support as their multiset. A strictly increasing sequence with that
   multiset is unique and equals scatter_forward's expert-major enumeration.
   Padding positions carry no pair (mask), so the compressed length is s.

Until steps 1-3 are implemented and a bound malicious witness for the swapped
order is rejected, Scatter remains fail-closed (commit 5df8e59). The formal
pre-fix counterexample already reproduced: with m=n=2, sel all ones and the
positions 0/2 swapped, the old bound verifier returned true.

### 2c. Tie-break formalization (topk_forward `then(a.cmp(b))`) — IMPLEMENTED

As-built summary: `TopKSelectTie` is a pre-committed aux batch (root bound in
`execution_seeds` before any challenge); its zero/bits/rank/row/less virtuals
and two pair-multiset permutations all take fresh transcript challenges, and
the tie-aux openings are checked against the virtual endpoints. The witness
batch keeps `d2 = (1-sel)*(thr_b-x)` (non-strict); the old `-1` strict gap is
gone. Honest tied inputs select exactly the smaller-index prefix and verify;
forged t_b, wrong rank and c_r=0 witnesses are rejected by the row-sum /
rank-recurrence / selection relations.

Forward rule (verified against `topk_forward`): per row, sort by value
descending and by ascending column index on ties; select the first k; thr is
the k-th value. Consequence: among the tied set `T = {e : x[e] = thr}` the
selected ones are exactly the smallest-column prefix of T.

Definitions (per row r, column e; flat position pos = r*n + e):
- `T[pos] = [x[pos] = thr_b[pos]]`. With the non-strict gap witnesses
  `d1 = sel*(x - thr_b) >= 0` and `d2 = (1-sel)*(thr_b - x) >= 0`, this is
  `T = sel*Z(d1) + (1-sel)*Z(d2)`. As built, `Z(d)` is proven with a
  committed inverse witness per gap column: `T1*d1 = 0`,
  `(1-T1)*(d1*w1 - 1) = 0`, `T2*d2 = 0`, `(1-T2)*(d2*w2 - 1) = 0`, and
  `T = sel*T1 + (1-sel)*T2` (one gamma-combined eq-weighted virtual). The
  zero-test has degree 4 and avoids the 2^16 expansion of `prod (1-bit_b)`;
  w1/w2 are the field inverses when the gap is nonzero and 0 otherwise, and
  all columns live in the pre-committed `TopKSelectTie` aux.
- 0-based tied rank: `P[pos] = #{e' < e : T[r,e'] = 1}` (the first tied
  position has rank 0).
- `c_r = sum_e T[r,e]*S[r,e]` = number of selected tied positions in row r;

  committed as the flat broadcast column `t_b[pos] = c_r`.

Corrected selection condition (off-by-one fixed):
  `T[pos] * (S[pos] - less(P[pos], t_b[pos])) = 0`,
  with `less(P,C)` the strict integer comparison over B = log2(n)+1 bits
  (`P < C`). With 0-based P this selects exactly ranks 0..c_r-1, i.e. c_r
  positions. Alternatives checked and rejected: 1-based P with strict
  `P < c_r` selects only c_r-1 (off-by-one; exhaustive enumeration fails);
  1-based needs `P <= c_r` or cutoff c_r+1 with strict less.

Boundary cases (exhaustive enumeration n in {2,4,8}, all rows over {0,1,2}):
- `c_r = 0`: selects no tied position. It cannot occur for k >= 1 with the
  attainment relation (`att` marks a selected tied position), and if a
  prover tried it the selection rule plus att would be contradictory; it is
  rejected, not silently accepted.
- `c_r = |T_r|`: all tied positions are selected; ranks 0..|T_r|-1 < c_r.
- `c_r > |T_r|` is impossible because `c_r = sum T*S <= sum T`; a forged
  t_b above the tied count fails the row-sum relation.
- Verified: the 0-based strict-less rule matched `topk_forward` on 52,830
  exhaustive cases (0 failures); the 1-based variant failed 52,812 cases.

Rank reset and padding:
- The flat position domain is exactly `m*n = 2^A` (TopK requires m,n powers
  of two and m*n >= 32; `A = log2(m*n)`), so there are NO padding positions
  and no padding rows in the tie gadget. `dom = 1 << arity == m*n` is
  asserted; otherwise fail closed.
- `E0[pos] = [e(pos) = 0]` (verifier computes `E0~(z) = prod_{b<log2 n}(1-z_b)`).
  Recurrence: `E0 * P = 0`; `(1-E0) * (P - P_prev - T_prev) = 0`, where
  `P_prev[pos] = P[pos-1]` and `T_prev[pos] = T[pos-1]` are cyclic shifts
  bound by the same pair-multiset LogUp permutation used by ScatterOrder.
  For e >= 1 the shift stays inside the row; the cyclic wrap (pos 0) is
  overridden by `E0 * P = 0`, so ranks reset exactly at row starts and never
  leak across rows.
- `t_b` is a flat column; `c_r = sum_e T*S` is enforced by the row-sum
  identity `eq_rows*(T*S - E0*t_b) = 0`: weighting the broadcast value by E0
  reads it once per row, so the identity forces `t_b(r) = sum_e T*S` (without
  E0 it would force `n*t_b`). The virtual's `sel` endpoint is bound by the
  witness-batch opening at the same point; `t_b` and `T` endpoints by the
  tie-aux opening.
- Row-constancy of the broadcast cutoff is a SEPARATE identity in the same
  virtual, with its own full-domain eq weight:
  `gamma*eq_full*(1-E0)*(t_b - t_b_prev) = 0`, where `t_b_prev` is bound to
  the cyclic shift by a third pair-multiset permutation (same primitive as
  P_prev/T_prev). A row-constant weight would only constrain the row sum of
  the difference, which telescopes and is satisfied whenever the row's first
  and last values agree; the full eq makes any pointwise difference a
  nonzero polynomial in the fresh challenge. Without this, a prover could
  set a per-position cutoff and select an arbitrary tied subset.
- The comparison uses the committed t_b bits per position with the MSB-first
  prefix recurrence `E_{B-1} = 1`, `E_b = E_{b+1}*eq_{b+1}` and
  `less = sum_b (1-P_b)*C_b*E_b` (degree <= 5).

### 3. Gate consistency chain — IMPLEMENTED (dynamic router-gate table; see tracker)

`idx_gate = token_id·n + expert_of` via `CombineIdx` (linear; already
supported), `gate_pos = gate_table[idx_gate]` via `Lookup` (already
supported; table root from the model), `weighted = d·gate_pos` via
`ScaleVec`/`BroadcastScale` (see 4), and `Gather` uses `gate_pos`. The chain
ties the gate used by the scatter-add to the router's `gate` column at the
same `(token_id, expert_of)` pair: `gate_pos` claim → Lookup → idx_gate →
CombineIdx → token_id/expert_of claims → Scatter routing → TopKSelect sel,
with `gate` from the same TopKSelect reduction.

### 4. Multiplicative tail reductions (ScaleVec / BroadcastScale) — DONE (best effort)

Implemented as exact-product eq-weighted virtuals with S4 same_poly output
binding: BroadcastScale (out[i]=x[i]*scale[i/h]) and ScaleVec shift 0
(out[i]=x[i]*scale[i%s]); ScaleVec shift>0 rounding remains fail-closed.
The gate-consistency chain uses the router gate itself as the dynamic Lookup table
(same tensor id), authenticated through the returned table claim merged with the
public gate output; static tables feeding a Gather gate fail closed.


Needed for `weighted = d·gate_pos`. Reverse pattern: eq-weighted product
virtual with claim 0 at a fresh point; the *output* endpoint must be
consistent with the op's output claim — merge the endpoint claim with the
merged output claim via `same_poly` (S4 machinery) so the relation and the
claimed value refer to the same polynomial; return claims on x/scale at the
virtual endpoint. Rounding (ScaleVec) reuses the committed remainder aux
(`ScaleRemainder`).

### 5. Real DeepSeek single layer — SEGMENTED REAL-DATA EXPERIMENT (not S7)

Status: a real-data core-chain experiment is implemented as TWO bound
statements; it is explicitly NOT gist S7 acceptance.

- Full-layer blocker: RmsNorm/RoPE/Softmax/StableSoftmaxIndex/ScaleVec-shift16
  remain fail-closed in the bound path (S5-family devices). No new reductions
  were added for them.
- Routing blocker: the single-statement core chain stops at IdxBroadcast.
  Its reduction needs the base index tensor to be verifier-replayable or
  certified public u32 data; internal routing intermediates are neither, and
  random-point claims cannot replace the pointwise u32 / b*base+j overflow
  contract. That branch stays fail-closed (the attempted relaxation was
  reverted).
- Segmentation: A = TopKSelect -> Scatter -> PadIdx, publishing the padded
  routing arrays as public index outputs; B = IdxBroadcast -> Lookup(real
  post-attn activation) -> BroadcastScale(mask) -> GatherMatMul(real expert
  gate weights) -> Scale(shift 16), taking those arrays as public index
  inputs. B's IdxBroadcast uses the existing verifier replay path (u32
  checked arithmetic), so no constraint is bypassed. The A/B link is the
  SAME caller-owned external statement bound item-by-item; the proofs'
  self-reported fields are never used for the link. The bytes entry is used
  for both segments and for the cross-segment tamper tests.
- Gist limitation (recorded): B's verifier ingests the full routing table,
  which the gist forbids for the final S7 construction. Public routing,
  segmented proofs, and replay are acceptable only as a bounded real-data
  experiment; S7 remains incomplete.
- Real data: layer 10, seq=1, token 549 (first extracted token).
  `scripts/ds_layer_core_inputs.py` runs the FULL reference forward: every
  prior layer includes its FFN/MoE (dense layer 0, routed+shared experts for
  1..LAYER-1); at LAYER it exports the post-attention RMSNorm MoE input. This
  trajectory is a FLOAT64 REFERENCE model over bf16 weights, quantized at the
  Q16 boundary; it is NOT the fixed-point full-model trajectory. The
  file `L*_post_attn_norm_i32.bin` is the RMSNorm WEIGHT (all-positive
  ~0.09-0.91), not an activation. Measured (corrected): activation
  max|.|=835232, max score 28854, max|logits16| 272985 < 2^22; top-6 =
  [13,42,47,9,2,31]. The TopKSelectXRange/ThrRange relations enforce the
  contract inside the proof; an out-of-range real score fails closed.
- Scope (NOT the complete S7 core chain): B stops at the expert GATE
  PROJECTION. It does not include the router gate consistency chain
  (CombineIdx -> Lookup gate_pos over the dynamic router table ->
  BroadcastScale -> Gather) and has no Gather reduction.
- Malicious regressions (bytes entry, real data): tampered A gate/score
  input, tampered B public routing array and g output, cross-segment A
  routing output tamper, cross-segment B routing mismatch, wrong expert
  weight root, wrong activation table root. Aux-level forgeries (committed
  counts/table, duplicate/deleted routes inside the committed routing) are
  covered by the synthetic S7 regressions, not by this public-API experiment.
- Measured cost (test machine, 90-bit security, corrected data): A proof
  241,874 bytes, ~0.17 s prove; B (real 8-expert slice, 2^25 weight)
  commit+prove 19.9 s, verify 30 ms, proof 463,092 bytes, peak RSS ~10.0 GB;
  all eight negative classes rejected. The FULL 64-expert (2^28-element)
  gate weight opening is a reported memory blocker: OOM-killed at anon-rss
  502,710,376 kB (dmesg), above the 495 GB machine. The run uses a REAL
  8-expert slice built from the ACTUAL routed experts of this row
  ([13,42,47,9,2,31]) plus real experts 6,7 for the masked padding rows,
  with an explicit public slice table; the external statement binds A's
  original routing output to B's sliced public input item-by-item. The
  composition verifier authenticates the slice TENSOR commitment and the
  public slice index mapping, but NOT that the slice equals the
  corresponding rows of the full 64-expert model: that identity is a
  trusted-offline manifest/external-modeling assumption (explicit
  `slice_experts` applied to the real weight file). This is a real-weight
  segmented experiment, not S7 acceptance.

## Acceptance gate

S7 is complete only when, on the real layer, the gist's S7 acceptance list
passes: equal-length forged counts/table, wrong expert roots, swapped
experts, duplicated routes, dropped active pairs, tampered gate all rejected,
and the honest small MoE plus the real DeepSeek single layer both verify.
Until then every incomplete op stays fail-closed and S7 is reported as
incomplete; S8/S9 are not started.
