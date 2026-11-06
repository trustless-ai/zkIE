# GeluIndex big-table reduction (project ②) — continuation record

Plan: `/tmp/tl_plan_deep_o7_and_geluindex.txt` (approved). Step 1 is
implemented; steps 2 (real silu 2^24 table + FFN segment) and 3 (full 64-expert
S7 retry) remain and await TL acceptance of step 1.

## Step 1 — parameterized ReLU + 25-bit GeluIndex domain witness

### ReLU parameterization

`relu_range_terms(gamma, bits)`, `prove_reverse_relu(..., bits, ...)` and
`verify_reverse_relu(..., bits, ...)` now take the bit width. The Relu op
keeps `RELU_BITS = 23`, `RELU_BOUND = 2^22`; its columns, transcript order and
proof bytes are unchanged (verified by the unchanged real A/B/C and full64
bytes and the existing Relu tests). GeluIndex uses `GELU_BITS = 25`,
`GELU_BOUND = 2^24`, i.e. the signed domain `[-2^24, 2^24)`.

### Domain contract

`prove/verify_reverse_gelu_index` now require:

- `table_len` a power of two and `table_len <= 2^24`;
- `offset.unsigned_abs() <= 2^24`;
- input length a power of two with `point.len() == log2(input.len())`;
- per-row `w = x + offset` and `high = w - (table_len - 1)` both inside
  `[-2^24, 2^24)` (the two 25-bit ReLU inputs).

`Relu` itself remains unchanged and still enforces its 23-bit domain.

### GeluIndexXRange aux (precommitted)

One `AuxRole::GeluIndexXRange` aux per GeluIndex op:

- `GELU_XRANGE_BITS = 25` boolean columns over the padded input domain,
  decomposing `x + 2^24` (zero-padding rows encode `0 + 2^24`, i.e. bit 24
  set);
- the descriptor carries `len = input length`, `arity = max(log2 len, 5)` and
  is part of the aux set absorbed by `execution_seeds`, so its commitment is
  fixed before every GeluIndex challenge (TopKSelectXRange pattern);
- `validate_aux_descriptors` includes the exact GeluIndex group specs, and
  the bound prover prepares it after the forward pass. The prover checks the
  committed bits against the actual input (and padding) and fails closed on
  any mismatch. The plain path (no model/aux) builds and commits the same
  witness inside the proof, absorbing the commitment before the range
  challenge, so that path stays sound too; the bound path requires the
  precommitted descriptor (`model.is_some()` + missing aux -> fail closed).

### GeluIndex proof / Fiat–Shamir order

`ReverseGeluIndexProof` additionally carries `range_commitment`, the
`x_range` virtual proof, the opened bit values and the batch opening. Order
inside the reduction:

```
absorb lower_at_output
absorb b"zkie-gelu-range-v1" + range commitment
q = field_vec(arity); gamma = field()
x_range virtual: eq(q,x)*(x + 2^24 - sum_b 2^b bits_b) == 0
                 + gamma-weighted booleanity of every bit (claim 0)
open the 25 bit columns at the range point; match the virtual final evals
upper ReLU (25 bits); absorb its input claims
lower ReLU (25 bits)
return upper/lower x claims + the x claim from the range virtual
```

The range virtual's claim-0 sumcheck is sound because the booleanity terms
vanish on the Boolean cube; the weighted identity forces `x + 2^24` to be the
25-bit integer represented by the committed bits, hence `x ∈ [-2^24, 2^24)`.
The sumcheck final eval at the random point is not required to be zero (the
virtual polynomial is not identically zero as a polynomial), only consistent
with `virtual_eval`; this is the same convention as the other range proofs.

### Wrap attack

The attack is `x = -(2^24+1)` (so `to_i64(x)` is outside the domain) with
`offset = 2^24+1`, making `x + offset mod p = 0` land inside the ReLU domain.
There is no 25-bit decomposition of `x + 2^24 = -1`; `build_gelu_range_aux`
returns None, and a proof built with a witness for a different input is
rejected by the prover's bit-consistency check. On the verifier side the
precommitted bits are opened at the range point and tied to the `x_range`
virtual final evals, so a forged x eval cannot be re-pointed at the wrapped
value; the test tampers the range x eval and verifier rejects.

### Tests and regression

- `zkie-ops --lib`: 167/0.
- New tests:
  - `gelu_index_signed_domain_boundaries_roundtrip`: `±2^24` domain edges,
    `table_len = 2^24`, `offset = 2^24` and their combination on in-range
    inputs.
  - `gelu_index_domain_wrap_and_contract_rejected`: the wrap input (prover
    fail-closed), out-of-domain `-2^24-1` / `2^24` (no 25-bit witness),
    non-power-of-two and `> 2^24` table_len, `|offset| > 2^24` (prover and
    verifier reject), and a forged range x eval (verifier rejects).
  - `gelu_index_big_table_bound_roundtrip_and_oob_rejected`: bound proof with
    `table_len = 2^24`, `offset = 2^23` (the real silu parameters), the
    `GeluIndexXRange` auxiliary present, two proofs byte-identical
    (determinism), and an out-of-domain bound input fails closed.
- Existing `gelu`/`relu` suites green; the existing committed
  Projection→GeluIndex→Lookup→Projection chain test green and now asserts the
  precommitted `GeluIndexXRange` descriptor.
- Real A/B/C: A 241,874 / B 491,978 / C 355,183 bytes, unchanged; full64
  `CORE-CHAIN-OK`, B 540,239, verify 154.7 ms. These proofs contain no
  GeluIndex reduction (the segmented FFN's GeluIndex at 2^24 is still only
  forward-evaluated in step 1), and the 23-bit Relu path is byte-identical.
- Deterministic PoW and 90-bit security unchanged.

## Step 2a — real silu 2^24 prefix (accepted, `4c4218a`)

A D phase proved `GeluIndex(offset=2^23, table_len=2^24) ->
Lookup(real L10 silu 2^24 static table)` with `g = B.g` as a caller-trusted
input, the precommitted GeluIndexXRange witness and the deep-O7 shared table
commit. Real domain: `g ∈ [-232482,171093]`, `w ∈ [8156126,8559701]`,
`high ∈ [-8621089,-8217514]`, all inside `[-2^24,2^24)`. Prefix proof:
908,637 bytes, hash `b2e058f3b527f805` (deterministic). The full FFN tail was
blocked at ScaleVec shift 16 and escalated; the boss approved the extension.

## Step 2b — ScaleVec shift>0 reduction + full FFN D segment

### Reduction

- Domain contract: position length `L >= 32` and a power of two; the scale
  period is a power of two dividing `L`; `1 <= shift <= 31`; real ScaleVec
  operands (silu and u) must be in the 25-bit signed domain `[-2^24,2^24)`
  (checked when the precommitted witness is built; the verifier relies on the
  authenticated input claims exactly as the existing Scale reduction does).
- Precommitted `ScaleVecRemainder` aux (ScaleRemainder generalized to a
  tensor/periodic scale): `shift` bit columns of
  `rem = x*scale_b - out*2^shift + 2^(shift-1)`, requiring
  `0 <= rem < 2^shift`. This is ties-up rounding, identical to `Op::Scale`.
- The bound reduction proves one eq-weighted virtual over the position domain:
  `eq*(x*scale_b - out*2^shift + 2^(shift-1) - sum_j 2^j bits_j) = 0`
  plus gamma-weighted booleanity of every remainder bit (claim 0). The
  precommitted bits are opened at the virtual point; the output endpoint is
  bound by S4 same_poly; the x claim is returned in full and the scale claim
  at `z[..lgs]` (shift-0 convention). The shift-0 path keeps its existing
  proof variant and bytes (the new variant is separate).
- Range-check realization: the bound reverse DAG has no legacy
  `lookup_by_table` channel, so the `[0,2^shift)` identity-table range check
  is realized by the precommitted bit witness + booleanity (the same range
  predicate); the trusted identity table of size `2^shift` remains present in
  the manifest via the existing ScaleVec push arm.

### Reduction tests

`zkie-ops --lib`: 169/0. New `scale_vec_shift16_roundtrip_and_negatives`:
honest shift-16 roundtrip including an exact tie (`1*2^15`), byte-identical
double proof, tampered scale/out/remainder-bit rejection, and ties-down /
rho-out-of-range inputs fail closed at the precommitted witness. The shift-0
test (`s7c_scale_vec_bound_roundtrip`) is unchanged and green.

### Full FFN D segment (real data)

Ops, using the real L10 up/down expert weights and real silu table:

```
GatherMatMul(up, w_u) -> Scale16(u) -> GeluIndex(offset 2^23, len 2^24)
  -> Lookup(real silu 2^24) -> ScaleVec(silu, u, shift 16)
  -> GatherMatMul(down, w_d) -> Scale16(out)
```

Trust boundary: `g` is B's proven gate output (asserted equal); `segz` (the
real activation segment) and `exp` (real public routing slice) are
caller-trusted inputs. The output `d` is asserted equal to the caller d used
by C.

Real ScaleVec operand range: `|silu|max = 159,381`, `|u|max = 231,507`, both
inside `2^24`; the example asserts this.

| quantity | value |
|---|---|
| FFN commit + prove | 15.95–16.28 s |
| FFN verify | 92.9–96.5 ms |
| FFN proof bytes | 1,359,818 |
| FFN proof hash (two runs) | `cbd9f41ab8a3469d` both |
| FFN proof dump SHA-256 | `c15b13220910062a5fe6324ea31f89587159f63ac3391fb7e19a6f7f0e3fdd52` both (`cmp` clean) |
| d length | 16,384 |

Negatives: tampered `segz`/`g` (u/silu sources), tampered routing slice, and
tampered output `d` are rejected; a wrong real silu-table root (manifest
rebuilt over a mutated 2^24 table) is rejected. Unit tests additionally cover
tampered ScaleVec scale/out/remainder and GeluIndex index/range derivation.

### Baselines

- Real A/B/C unchanged: 241,874 / 491,978 / 355,183.
- Full64 unchanged: B 540,239, `CORE-CHAIN-OK`.
- The full FFN D proof is the new baseline: 1,359,818 bytes, deterministic
  across two runs.
- Logs: `/data/zkie-soundness-fix/logs/scalevec-ops.log`,
  `gelu2b-fullffn-run{1,2}.log`, `gelu2b-full64.log`.

## Remaining

Step 3 (S7 full 64-expert retry: full FFN + routing chain + gate consistency,
gist S7 negatives, memory regression) is next after TL review of step 2b. S7
is not accepted until TL final review; S8/S9 stay postponed. Deep O7 (a)–(d)
is accepted.
