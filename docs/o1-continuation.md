# O1 continuation — Argmax public-selector reduction

State: `4c4f145` + this O1 change on `fix/soundness-repair`. O1 awaits TL
review. Active bound path only; the retired committed verifier remains closed
(O6 scope).

## Implementation

The per-token `BoundSelOpening` in the active bound path is replaced by a
public-selector reduction inside the Argmax reverse proof.

For public tokens `t_i` (read from the caller-owned execution outputs, never
from the proof), define

```
E(u,v) = Σ_i eq(u, bits(i)) · eq(v, bits(t_i))
```

so at integer coordinates `E(i,j) = 1` iff `j == t_i`. One EF virtual sumcheck
over the `m×n` domain proves the two relations

```
Σ eq(r_row,i) · sel(i,j) · E(i,j)       = 1
Σ eq(r_row,i) · logits(i,j) · E(i,j)    = M(r_row)
```

with a single random challenge row point `r_row` and one batching challenge
gamma:

```
F = eq_rows · E · (sel + gamma · logits)
claim = 1 + gamma · M(r_row)
```

- The first relation binds the public token list to the committed one-hot
  selector (the same ArgmaxWitness aux column 0 the reverse reduction already
  binds). It preserves the existing tie rule: a tied second maximum has
  `sel(i,t_i)=0`, so the relation fails at the random row point. No change to
  the `before`/`tie` witness or the max/range relations.
- The second relation binds the public token's logit to the max value
  `M(r_row)`. `M` is supplied as a new `thr` claim at `r_row`; the reverse DAG
  authenticates it through the existing max/range witness and the same-poly/
  seed path.
- The logits endpoint `logits(z_sel)` becomes a new `x` claim, which continues
  to reduce through lm_head exactly like the existing Argmax `x` claims.
- The selector value `sel(z_sel)` is added to the existing aux opening list.
- The verifier evaluates `E(z_sel)` from the public token list in
  `O(m(log m + log v))`; it never builds the `m×v` table.
- The prover builds one dense `E` column of `m·n` EF entries. This is one
  tensor, not a per-point equality-vector rebuild; it is released after the
  sumcheck (the cc48f6f multi-point batch-opening path is not restored).

Proof/wire changes:
- `ReverseArgmaxProof` gains `selector: ArgmaxSelectorProof { m_eval, sumcheck }`
  (`topk.rs`).
- `BoundExecutionProof.sel_openings` is retained as a legacy wire field but is
  always empty on honest proofs; a non-empty value is rejected as
  non-canonical. `BoundSelOpening` remains only for decoding/rejection.
- Helpers: `topk::public_selector_eval`, `topk::ArgmaxSelectorProof`.
- `prove_bound_token_binding` now only builds the embedding token binding.
- Max/range witness, selector booleanity, tie bits, sequence checks and the
  token binding are unchanged.

## Security / negative regression

- New `argmax_public_selector_replaces_token_openings`:
  - forged maximum (`selector.m_eval += 1`) rejects;
  - forged logits endpoint (`selector.sumcheck.final_evals[3] += 1`) rejects;
  - wrong public token (execution argmax value changed) rejects;
  - honest proof verifies, `sel_openings` is empty, and the Argmax step carries
    the selector sumcheck.
- `committed_argmax_chain_binds_all_outputs_and_tie_rule`: tampering any of
  sel/thr/gate/d1/d2/out rejects; forging the public argmax to the tied second
  maximum rejects.
- `s6_tie_rule_rejected`, `s6_wrong_prompt_rejected`,
  `s6_response_tampered_rejected`, `s6_length_mismatch_rejected`,
  `s6_out_of_vocab_rejected` stay green.
- `s8_lookup_group_wire_mutations_reject_without_panic` and the generic
  bound-wire mutation suite stay green; a non-empty legacy `sel_openings`
  field is rejected.

## Measurements

Harness: `crates/zkie-ops/examples/o1_argmax_metrics.rs`, one Argmax with the
GPT-2 head row count (`m=16`) and `n=4096` / `n=16384` (the full GPT-2 head is
`n=65536`; the full 12-layer graph is dominated by non-Argmax aux and is not
used to isolate this change). The same source runs on the pre-O1 worktree
`4c4f145`. Cold = `drop_caches` immediately before; warm = immediate rerun.
The logits table is generated in memory, so cold/warm differs only in
binary/library page-in (read_bytes 4.55-4.75 MB, majflt 6 cold; 0/0 warm).

### `m=16, n=4096`

| metric | pre-O1 `4c4f145` | O1 |
|---|---|---|
| selector groups / points / bytes | 1 / 16 / 136,101 | 0 / 0 / 0 |
| proof bytes | 413,158 | 282,225 (-31.7%) |
| prove ms (cold / warm) | 12,738.0 / 12,700.4 | 9,511.1 / 9,520.8 (-25.3% / -25.0%) |
| verify ms (cold / warm) | 27.729 / 27.350 | 20.612 / 20.451 (-25.7% / -25.2%) |
| allocations | 8,398,759 | 5,602,666 (-33.3%) |
| allocated bytes | 4,784,876,974 | 3,719,234,570 (-22.3%) |
| VmHWM kB (cold / warm) | 989,868 / 990,316 | 883,000 / 883,480 (-10.8%) |
| process wall s (cold / warm) | 12.805 / 12.757 | 9.590 / 9.586 (-25.1% / -24.8%) |
| RSS after kB (cold / warm) | 304,208 / 304,280 | 327,920 / 328,008 (+7.8%) |

### `m=16, n=16384` (scaling)

| metric | pre-O1 `4c4f145` | O1 |
|---|---|---|
| selector groups / points / bytes | 1 / 16 / 148,061 | 0 / 0 / 0 |
| proof bytes | 455,742 | 309,153 (-32.2%) |
| prove ms (cold / warm) | 47,976.5 / 47,891.3 | 36,316.3 / 36,357.4 (-24.3% / -24.1%) |
| verify ms (cold / warm) | 79.156 / 79.550 | 66.004 / 65.178 (-16.6% / -18.1%) |
| allocations | 33,271,244 | 22,189,802 (-33.3%) |
| allocated bytes | 19,019,354,934 | 14,794,998,058 (-22.2%) |
| VmHWM kB (cold / warm) | 3,948,164 / 3,949,444 | 3,523,744 / 3,523,068 (-10.8%) |
| process wall s (cold / warm) | 48.129 / 48.030 | 36.487 / 36.482 (-24.2% / -24.0%) |
| RSS after kB (cold / warm) | 1,079,468 / 1,079,596 | 1,275,312 / 1,275,444 (+18.1%) |

The selector-opening proof bytes (136-148 kB) disappear; total proof bytes drop
by 130-147 kB (-32%). Accounting: n=4096 removes 136,101 B of selector opening
and adds ~5.2 kB of selector sumcheck/endpoint/claims (net -130,933 B); n=16384
removes 148,061 B and adds ~1.5 kB (net -146,589 B). The prover's dense E column and the
existing logits/selector columns are the remaining allocation cost; the
verifier side is O(m(log m+log v)) and never materializes E. VmHWM drops ~11%
in both cases. RSS-after is higher by 7.8% / 18.1% (the dense E column is
retained in allocator pages after the proof); this is reported as measured,
not hidden. Cold/warm proof times are stable (compute-bound).

Logs: `/data/zkie-soundness-fix/logs/o1-syn-{base,cur}-n{4096,16384}-{cold,warm}.log`.
Pre-O1 worktree: `/data/zkie-o1-baseline` (4c4f145).

The GPT-2 M=16 opening upper bound is 25 both before and after O1
(`ZKIE_BUDGET_ESTIMATE=1` on `prove_lm_bundle`), well under the cap 64; the
two new residual claims (logits endpoint and max value) do not change the
provisioning.

## Regression

- `cargo build --workspace --release`: pass.
- `zkie-ops` release: 240 passed / 0 failed on the final tree.
- `zkie-core` release: 94 passed / 0 failed / 1 ignored (858.06 s on the
  final tree; core itself is unchanged by O1).

## Scope / remaining

The active bound path is covered. `prove_committed_shard_dag` (legacy
committed path) still has its `SelOpening` list, but its verifier
(`verify_committed_shard_dag`) is permanently closed and the path is retired
under O6; O1 does not revive or extend it. O1 awaits TL review.
