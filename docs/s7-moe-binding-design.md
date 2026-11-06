# S7 · MoE expert binding in the bound reverse path — protocol design

Status: **STAGED / PARTIAL — S7 is NOT complete.** The Gather reduction now
uses an authenticated one-hot routing matrix (committed aux, opened only at
transcript-derived points) — the verifier never reads the routing table.
A verifier-side public-index replay prototype was rejected by review and
removed; Scatter keeps failing closed until its permutation/compaction
reduction lands.

Status (earlier): **STAGED / PARTIAL — S7 is NOT complete.** The GatherMatMul
expert-FFN core is implemented and tested on an isolated small MoE graph;
the linear index ops (CombineIdx/IdxBroadcast/PadIdx) are implemented and
tested as the first S7b batch. Scatter/Gather and the gate-weighted tail remain fail-closed. The
green zkie-ops suite proves no regressions; it is not S7 acceptance, and no
real DeepSeek layer has been proven. Remaining work: S7b (see the dependency
table and the tracker entry).

## Threat

The plain GatherMatMul proof carries free `counts`/`table` vectors and only
checks `sumcheck.final_evals[1] == <counts, table>`. Nothing ties `counts` to
the routing actually used by the matmul, nor `table[e]` to the committed expert
weights. A prover can replace an E-entry counts/table pair with the shortest
consistent fabrication and still pass.

## Trusted inputs

- Op graph (statement): `GatherMatMul { a, expert_of, w, c, s, k, n, n_experts }`
  fixes the expert count, shapes and tensor IDs; `absorb_statement` binds them
  into every challenge stream.
- Trusted model manifest: the batch commitment for `w` (expert-major
  `n_experts × k × n`) is the only accepted root for expert weights.
- Public execution: inputs `a`, `expert_of` (index tensor) and output `c`.

## Protocol (GatherMatMul reverse reduction)

Let `A = 2^arity` be the padded position domain (`arity = max(5, log2 s)`).

Prover precommits one aux batch, role `ExpertRouting`, with `E = n_experts`
columns `sel_e` over positions: `sel_e[pos] = 1[expert_of[pos] = e]`,
zero-padded. The descriptor (op index, role, len = s, arity, columns = E) is
derived from the graph by the verifier and bound into the aux commitment.

Challenges (both sides, same order):
1. `u ∈ F^{log2 s}`, `v ∈ F^{log2 n}` — position/output evaluation points.
2. Replay the batched-matmul sumcheck (`bmm::verify`) on the output claim
   `c(v, u)`: yields `ch = (ch_d ∈ F^{log2 k}, ch_pos ∈ F^{log2 s})` and the
   authenticated terminals `fe_a = a~(ch)`, `fe_w = W~_gathered(ch)`.
3. `q ← F^arity`, `γ ← F`; eq-weighted routing virtual (claim 0) over
   `[eq_q, sel_0..sel_{E-1}, expert_of_pad, mask]`:

   ```
   Σ_e γ^e · eq · sel_e·(sel_e − 1)          booleanity
   + γ^E · eq · (Σ_e e·sel_e − expert_of)    tie to the routing column
   + γ^{E+1} · eq · (Σ_e sel_e − mask)       exactly one expert per valid position
   ```

   `mask[pos] = 1[pos < s]` (effective length / padding), closed-form in the
   verifier. Claim 0 forces, by Schwartz–Zippel at the random eq point:
   every `sel_e` is Boolean, each valid position selects exactly one expert,
   padding selects none, and `expert_of(pos) = Σ_e e·sel_e(pos)` everywhere.
   This is what rules out duplicated routes (two selectors set), dropped
   routes (no selector on a valid position) and out-of-range experts
   (`expert_of ≥ E` cannot equal the tie for an E-column one-hot).

   The virtual's endpoint evals are authenticated as: `eq_q(z)` closed-form;
   `sel_e(z)` by the aux batch opening at `z`; `mask(z)` closed-form;
   `expert_of~(z)` is returned as a **claim on the public/upstream
   `expert_of` tensor** at `z` and authenticated by the reverse chain or the
   public-input leaf.

4. `counts_e = sel_e(ch_pos)` — read from the aux batch opening at `ch_pos`.
   The verifier derives counts; the proof never carries them.
5. `table_e` — the E weight-side terminal values. The verifier derives the
   sampled points `p_e = v ‖ ch_d ‖ bits(e)` (expert-major layout of `w`) and
   returns one claim per expert `(w, p_e, table_e)`. Those claims are grouped
   into the manifest weight batch by `group_reverse_weights` and opened by the
   existing batch-opening path, so `table_e` is forced to the committed
   expert block evaluation at the verifier-chosen point. Swapping expert
   blocks/roots changes the value at `p_e` and fails the opening.
6. Weight terminal check: `fe_w == Σ_e counts_e · table_e`. With `fe_w`
   locked by the sumcheck and both factors authenticated, neither counts nor
   table can be fabricated (equal-length forgeries included).

The aux batch is opened once at the two derived point sets (E entries at
`ch_pos`, E at `z`), in a verifier-rebuilt order.

## Dependency table / fail-closed boundary

| Op | Reverse reduction | S7 state |
|---|---|---|
| GatherMatMul | routing + expert weights + activation | implemented (isolated core only) |
| MatMul / Projection / Add / Scale / ScaleVec? | dense branch | pre-existing (MatMul, Projection, Add, Scale) |
| Scatter | permutation between sel and (token_id, expert_of) | fail-closed |
| Gather | weighted histogram scatter-add | fail-closed |
| BatchedMatMul, CombineIdx, IdxBroadcast, PadIdx, BroadcastScale, Softmax, TopKSelect | – | fail-closed |

Consequences for the DeepSeek MoE layer:

- The expert FFN core (`seg_z @ W[expert_of]`, three GatherMatMuls) is
  covered: routing, expert count/shape/ID, expert weights.
- Shared experts are dense `Projection`/`MatMul` ops (separate branch) — no
  routing enters them.
- `top-k` exactly, `(token, expert)` active-pair uniqueness across positions,
  `token_id`↔`expert_of` consistency, and the router gate (`TopKSelect` sel →
  `Scatter` → gate gather → `BroadcastScale`) live in Scatter/Gather/
  CombineIdx/Lookup/BroadcastScale, which have no reverse reduction yet. Any
  graph containing them is refused by the bound path (fail-closed), so a
  tampered gate cannot yield an accepted proof; the positive binding is
  deferred to S7b.
- The plain (non-committed) shard verifier is a demonstrator; the bound
  committed entry is the sound one. The old committed API remains
  permanently fail-closed (verifier returns false) and is not re-enabled.

## Integer/index domain contract (S7b)

Index tensors hold u32 values. Every index relation is checked exactly over
the Goldilocks field (exact for values < p), so the field relation coincides
with exact u32 arithmetic only under the contract: all index values are u32,
`scale`/`b` < 2^32, and the result fits u32 (no wraparound). The forward path
enforces the contract with checked arithmetic (fail closed), the reverse
reductions reject out-of-domain constants, and the bound verifier
range-checks every public index tensor. Intermediate index tensors produced by
arithmetic relations (CombineIdx/IdxBroadcast) are NOT certifiable and fail
closed unless they are public; PadIdx copies propagate certification, and
non-power-of-two public index tensors are zero-extended consistently in the
leaf check. A dedicated per-column range binding for arithmetic
intermediates remains S7b work. PadIdx pins the base tensor's exact
length to `base_len`, so the zero-extension identity is exact (no unverified
padding entries).

## Tests

- `s7_gather_matmul_bound_roundtrip` — honest small MoE core via bytes entry.
- `s7_equal_length_forged_counts_table_rejected` — fabricated E-entry tables
  (and matching weight-terminal edits) rejected.
- `s7_swapped_expert_weights_rejected` — swapped expert claim values /
  opening entries rejected.
- `s7_duplicate_route_rejected`, `s7_dropped_route_rejected` — malicious
  routing aux witness rejected by the booleanity / one-hot virtuals.
- `s7_gate_weighted_chain_fail_closed` — a chain with the gate tail
  (BroadcastScale/Gather) is refused by the bound path.
