# LayerNormCentered reverse reduction (6th B1 operator)

Status: implemented in `crates/zkie-ops/src/compose.rs`; ops suite green
(209 tests incl. 6 LNC tests). Real-model evidence below.

## 1. Forward semantics (aligned with `layer_norm_centered.rs`)

Per row with `n_real <= d` real columns:

    mean  = round(sum_j x_j / n_real)
    c_j   = x_j - mean
    var   = round(sum_j c_j^2 / n_real)
    s_idx = rsqrt_index(var)              # public piecewise map (same as RmsNorm)
    rstd  = rsqrt_table[s_idx]
    raw   = c_j * rstd * w_j
    out   = round_div(raw, 2^32) + b_j
    rem   = raw - (out - b_j)*2^32 + 2^31  in [0, 2^32)

`sum_x/rem_mean` and `sum_var/rem_var` are the exact truncation witnesses:
`sum_x = n*mean + rem_mean - n/2`, `sum_var = n*var + rem_var - n/2`.

## 2. Aux witnesses (all committed before challenges, via `aux_specs`/AuxPlan)

| Role | len | columns |
|---|---|---|
| `LncRowWitness` | m | full RMS row layout (`var` as `ms`, `sum_var` as `sum_sq`) + `mean, sum_x, rem_mean, delta, rem/delta bits`, and a bounded mean decomposition (`mean = 2^16*raw + rem - 2^15`, signed 16-bit raw + 16-bit rem + Boolean bits) |
| `LncOutWitness` | m*d | `out, rem_lo, rem_hi, o_lo, o_hi` + four 16-bit decompositions (`rem_lo/rem_hi` from `rem`, `o_lo/o_hi` from `out-b+2^31`) |
| `LncRsqrtPairs` | m | `(s_idx, rstd)` |
| `LncRsqrtMultiplicity` | table_len | counts per table entry |

The out witness is a 69-column committed batch; column 0 is the full out
column used by the `same_poly` bind.

## 3. Relations

* **Row virtual** (`lnc_row_terms`): the complete RMS row relation set
  (mean-square truncation, rem range, the piecewise `rsqrt_index` map, rounding
  raw) plus:
  * `sum_x - n*mean + n/2 - rem_mean = 0` (mean truncation),
  * rem/delta bit decompositions and `rem_mean + delta = n-1`,
  * the bounded mean decomposition `mean = 2^16*raw + rem - 2^15`, with
    signed 16-bit raw (raw+2^15 bits), 16-bit rem, and Booleanity. Together
    with the rem range this makes the mean quotient unique (TL audit HIGH),
  * Booleanity of every new bit.
  This pins `s_idx` to `rsqrt_index(var)`, so the lookup cannot select a wrong
  table entry.
* **Mean virtual**: `sum_j rho_j x_ij = sum_x_i` over the m*d domain, with the
  row-eq broadcast and `eq_i` sampled on the row axis. The `sum_x_b` column is
  constant across d, hence the `d_inv` factor. The verifier pins the
  prover-supplied rho column by recomputing `sum_{j<n_real} eq(z[..lg_d])_j`
  and checking the rho final evaluation (TL audit CRITICAL; same as RmsNorm).
* **Var virtual**: `sum_j rho_j (x_ij - mean_i)^2 = n*var_i + rem_var_i - n/2`,
  with `mean_b/var_b/rem_var_b` opened at the row point and the same rho
  final-evaluation check. Its term pencil has max degree 4, charged as 4 in
  the budget arm.
* **Out virtual**: `(x - mean_b)*rstd_b*w + 2^31 = 2^32*(out-b) + rem_lo +
  2^16 rem_hi`, the signed pin `out-b+2^31 = o_lo + 2^16 o_hi`, and the four
  16-bit Boolean decompositions. The 32-bit remainder range plus the pin gives
  a unique rounded solution.
* **Bind**: `same_poly` merges the caller claim `(point, value)` with the
  virtual endpoint claim on the committed out column; the merged claim is
  opened at the merged point (direct path) or recorded into the pooled aux
  opening (bound path). `value` is therefore authenticated against the
  committed out column, and the out virtual ties that column to the committed
  x/w/b.

## 4. Evidence

* `b1_lnc_reverse_roundtrip` — honest direct + bound/pooled roundtrip,
  forged caller value rejection, `K <= 64` (synthetic shape K=3).
* `audit_lnc_forged_output_claim_must_reject` — malicious prover keeps all
  honest virtual proofs, commits the honest out witness, and replaces the bind
  with `f' = out + a*chi0 + b*chi1` satisfying `f'(point)=forged`,
  `f'(z_out)=fo[6]`; post-bind rsqrt lookup regenerated under the forged
  transcript. Red (accepted) with the merged-claim check disabled, green
  (rejected) with the fix. See the tamper test for the exact gates.
* `lnc_tamper_matrix_must_reject` — prover rejects mutated mean/sum_x/
  rem_mean/mean decomposition/var/rem_var/rstd/s_idx/bits, out limbs/bits/
  value, rsqrt pair and multiplicity; verifier rejects mutated mean/var/out/
  limb/ones evals, the mean/var rho evals, row mean/ms evals, and a mutated
  rsqrt index; the pooled bound verifier rejects a mutated LNC step and a
  tampered aux descriptor.
* `audit_lnc_forged_rho_must_reject` — TL audit regression: row aux forged to
  mean+1 and sum_x+n, the out witness recomputed, and the mean/var virtuals
  built with fake rho columns solved to zero the sums. Accepted when the rho
  gates are disabled (red evidence), rejected with the gates enabled.
* `lnc_real_shape_roundtrip_pooled` — GPT-2/TimesFM shape
  `(m,d,n_real)=(16,1024,768)` (non-power-of-two n_real), direct + pooled,
  K=4.

## 5. Budget

The `ReverseReduction::Lnc` arm charges the row/mean/var/out virtuals, the
same_poly merge, the rsqrt fractional pair + padding virtual, the per-op
openings only for `aux_layout < 2`, the table base opening, and the pooled
group opening (charged with the group's full column count). Synthetic K=3,
real-shape K=4; both below `MAX_EFFECTIVE_OPENINGS=64`.

## 6. Real-model evidence

- **Wiring**: `reverse_op_io` accepts `LayerNormCentered`, so the B2
  `public_execution` boundary no longer exits 2 at this op for GPT-2 or
  TimesFM. Both full graphs pass the LNC path and now stop later at the
  Softmax row-sum boundary (see `docs/b2-evidence/b2-real-model-status.md`).
- **Real weights, real activations, real rsqrt table**:
  `models/gpt2/examples/lnc_real_check.rs` and
  `models/timesfm/examples/lnc_real_check.rs` take every LNC op out of the
  real graph, keep the real input activations/weights/outputs, and prove one
  bound statement over them. `LNC_LIMIT=N` restricts the prefix for a quick
  check.
  - GPT-2 (1 op, m=16, real 2^23-entry table, post-audit build):
    prove 28.4 s, proof 940,094 bytes, verify=true, budget K=5.
  - TimesFM (1 op, post-audit build): prove 33.8 s, proof 957,312 bytes,
    verify=true, K=5.
  - The all-ops runs (25 GPT-2 / 20 TimesFM LNC ops in one proof) are the
    same harness without `LNC_LIMIT`; they are long because each op carries
    its own rsqrt table lookup (the M4 per-table aggregation target).
