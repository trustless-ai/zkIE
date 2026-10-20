# GPT-2 authenticated integer execution

The new `zkie-verified` route compiles the repository's padded GPT-2 architecture
into a model-pinned integer graph and verifies every interior tensor through
polynomial commitments. This is separate from the legacy `verify_shard` route.
The new verifier never loads weights, activations or the old forward evaluator.

Implementation and bounded full-architecture acceptance are available. Execution
with actual GPT-2 weights has **not** been accepted: those weights were unavailable
in this session. No full-size prover time, memory requirement, proof size or
floating-point model accuracy is claimed. The small-model results below cannot
be extrapolated into a production GPT-2 performance claim.

## Proven statement and trust boundary

Given a verifier-pinned key digest, public embedding matrix, public logits and
request context, the proof asserts that executing the key's exact typed integer
program and committed constants produces those logits. The key commits to all
operations, shapes, constant/table roots, declared numeric ranges and output node.
The digest must be obtained independently of the untrusted prover's response.

The input boundary is a precomputed embedding matrix, row-major, including any
positional embedding supplied by the application. Tokenization, the embedding
lookup from token IDs, provenance of the input text, model training, autoregressive
sampling and KV-cache correctness are outside this statement. Output is the
padded vocabulary's **raw integer matrix product** (Q32 for two Q16 operands),
matching the repository's final unrounded `MatMul`; it is not decoded text.

Key preparation checks constant shapes and integer ranges, then commits them.
An application endorses that key once and distributes its digest through its own
trusted channel. Verifying a proof does not establish that supplied weight/table
files are OpenAI's GPT-2, that a table equals an ideal transcendental function,
or that quantization preserves floating-point accuracy. Those are separate model
release/approximation validations. There is no witness-hiding claim.

## Architecture and arithmetic

`transformer::Config::gpt2(sequence)` selects 12 pre-norm blocks, 12 heads of 64,
1024 padded columns / 768 real LayerNorm columns, FFN width 4096 and padded
vocabulary 65536. Supported sequence lengths are powers of two up to **32**;
this limit follows the current bit-tensor resource cap, not GPT-2's original
context window. The smaller configurable architecture is used only for tests.

The compiler lowers these paths into authenticated gates:

| Path | Constraints |
| --- | --- |
| Projection | Committed matrix product, signed quotient, bounded remainder, bias broadcast |
| Q/K scale and transpose | Integer multiply, nearest divide by 65536, coordinate permutation |
| Attention scores | Committed contraction, nearest divide, committed causal mask |
| Stable Softmax | Proven pairwise row max, difference/clamp, committed exp lookup, row sum, rounded positive division |
| LayerNorm | First-768 mean and variance reductions, centering, piecewise rsqrt index, committed lookup, two products, nearest divide by 2^32, bias |
| GELU | Signed offset and clamp, committed table membership |
| Residual and head sum | Integer affine gates |
| Final logits | Committed unrounded matrix product |

Nearest rounding is `floor((x + floor(d/2))/d)`, including negative ties. All
quotient equations enforce `0 <= remainder < d`. Non-power-of-two divisors use a
second range-checked witness `d-1-remainder`. Softmax's variable divisor uses
`r=2*numerator+denominator-2*q*denominator` and
`delta=2*denominator-1-r`; both are unsigned. These equations simultaneously
exclude a zero denominator and enforce nearest rounding.

Signedness/ranges are explicit in the program. Input and usual activations use
25 signed bits. Centering, squares, reductions and products have wider profiles;
LayerNorm's normalized intermediate is capped at 40 signed bits. A model/input
outside a profile is rejected. The compiler's range choices are a restricted,
alias-free integer semantics; they are not an assertion that every legacy i32
input is supported. Every arithmetic gate has a static absolute residual bound
strictly below the Goldilocks modulus. Unsafe profiles fail validation.

Constant tables may have a non-power-of-two true length; their committed padding
cannot be addressed. The model digest binds their contents and true length.
LayerNorm's piecewise rsqrt indexing follows `compose::rsqrt_index`, including
both thresholds and floor division in the coarse branches. Padded LayerNorm
columns are centered too, while reductions use only the configured real prefix.

## Transcript and proof structure

All main, bit and quotient-remainder roots are committed before arithmetic
challenges. Public input/output are anchored through random-point openings.
Each gate descriptor is reconstructed from the key, and every terminal witness
polynomial claim is checked with WHIR. Lookup multiplicities are committed before
compression challenges; the complete fraction tree and its terminal denominator
and cross product are authenticated. No challenge, gate list or claim is accepted
merely because a prover supplied it.

The reduction backend uses Goldilocks^5, Poseidon2 and upstream WHIR 0.7.0's
requested-128-bit profile, with no proof-of-work. This is a parameter setting,
not an audited whole-system 128-bit security estimate: hash/PCS assumptions,
Fiat-Shamir, all reductions and their composition still need independent review.
The original tiny STARK baseline has its separately documented security budget.

Canonical postcard encoding, expected tensor/round/opening counts, shape checks,
resource limits, independently pinned model IDs, and panic containment surround
verification. Keys are capped at 16 MiB; graph proofs at 512 MiB. The prover also
rejects an encoded graph proof above that limit. There is no promise that a
full-size proof fits this cap. The implementation retains witness/PCS data in
memory and currently opens constraints separately; it is a correctness-oriented
prototype, with no streaming, batched-opening or production memory claim.

## Running it

Build/test everything on bounded fixtures without downloading model files:

```sh
bash scripts/check_verified.sh
```

The script checks prover-enabled and verifier-only builds with strict Clippy,
runs 16 unit tests and 3 numerical integration tests, and runs three
independent-process acceptance scripts.
CI's existing verified-inference job invokes this script too.

For actual files, first inspect the program without committing or proving:

```sh
cargo +1.94.1 build --release -p zkie-verified --features prover --bins --locked
target/release/zkie-gpt2 inspect examples/gpt2-verified-q16.json /path/to/weights
```

The loader uses the repository's names, e.g. `L0_q_w_i32.bin`,
`L0_ln1_w_i32.bin`, `gelu_table_i32.bin`, `rsqrt_table_i32.bin`,
`exp_table_i32.bin`, `ln_f_w_i32.bin`, `lm_head_w_i32.bin`. Matrices are
row-major little-endian i32. Files with partial integers or wrong sizes reject.
All paths come from the explicitly supplied directory. The request JSON contains
`input` (signed integers) and `context` (32 bytes); output arrays are not a
request input. No files are downloaded automatically.

Preparation is an explicit, potentially expensive operation on real weights:

```sh
target/release/zkie-gpt2 prepare examples/gpt2-verified-q16.json /path/to/weights trusted-model
target/release/zkie-gpt2 prove examples/gpt2-verified-q16.json /path/to/weights request.json result
cargo +1.94.1 build --release -p zkie-verified --no-default-features --bin zkie-gpt2 --target-dir target/verified-verifier --locked
target/verified-verifier/release/zkie-gpt2 verify trusted-model/key.bin TRUSTED_MODEL_ID request.json result/statement.json result/proof.bin
```

The application pins the ID printed during trusted preparation, rather than
trusting the ID printed alongside an arbitrary proof. Output directories must
not already exist. The library can reuse an in-memory proving key; separate CLI
invocations currently prepare the commitments again.

## Acceptance evidence (2026-10-20)

`accept_gpt2.py` creates a **synthetic** two-layer, two-head graph with sequence 2,
width 4, real LayerNorm width 3, head width 2 and vocabulary 4. It uses nonzero
signed inputs, identity Q16 matrices and explicitly specified test tables. An
independent Python integer implementation computes attention, masking, maxima,
normalization, rounding, residuals and final logits without invoking Rust's
forward evaluator. All eight logits agree. The true exp table has five entries,
so lookup padding is exercised as well.

The script prepares the trusted key separately, generates a proof, copies only
key/request/statement/proof and the verifier-only binary to a consumer directory,
and deletes the producer directory including all weight files. Verification
succeeds there. Modified input, output, context, proof corruption/truncation/
trailing bytes, wrong trusted ID and truncated key all reject. A partial-i32
weight file is also rejected during inspection.

One observed local run (single sample, not a benchmark):

| Measurement | Result |
| --- | ---: |
| Graph nodes / constant coefficients | 327 / 317 |
| Verification key | 7,389 bytes |
| Proof | 9,267,882 bytes |
| Prepare | 2.86 ms |
| Prove | 2.65 s |
| Verify | 296 ms |

Modified-prover Rust regressions regenerate auxiliary commitments/proofs after
forging a quotient, ReLU result, hidden row reduction or rounded division. All
reject; a forged zero Softmax denominator also rejects. Other regressions cover
missing constraints/openings, lookup tree/true-domain binding, invalid views,
unsafe integer profiles, serialization and strict feature separation.

## What remains before a full-size deployment claim

Actual model artifacts and their independent hashes/quantization specification
must be supplied and endorsed. Run a full-size differential execution on them,
then measure memory/proof size/time under an explicit resource budget. The
current per-constraint openings and large lookup tables need scalability work
if that run exceeds the resource caps. This session did not run such a test.
Independent cryptographic review, an aggregate soundness analysis, input/token
provenance if required by the application, and any privacy guarantees remain
separate work. Passing the bounded acceptance suite does not satisfy those gates.

## Code navigation

- `crates/zkie-verified/src/transformer.rs`: architecture, numeric profiles, strict weight loader.
- `crates/zkie-verified/src/graph.rs`: pinned program/key, alias guards, interpreter, proof layout and independent verifier.
- `crates/zkie-verified/src/oracle.rs`: multilinear views, authenticated sumchecks, tensor openings and matrix contractions.
- `crates/zkie-verified/src/lookup.rs`: pre-challenge multiplicities and authenticated fraction trees.
- `crates/zkie-verified/src/whir_backend.rs`: extension field, PCS parameters and transcript domain separation.
- `crates/zkie-verified/src/bin/zkie-gpt2.rs`: producer/consumer CLI boundary.
- `scripts/accept_gpt2.py`: independent reference and consumer-only process tests.
- `scripts/check_verified.sh`: bounded acceptance entry point.
