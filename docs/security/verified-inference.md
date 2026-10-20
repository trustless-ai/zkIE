# Verified inference baseline

For the subsequent committed-graph and GPT-2 compiler work, see
[the current protocol and acceptance report](gpt2-verified.md). This file
records the earlier bounded STARK milestone.

Status: the first bounded milestone passes local acceptance. This document
defines its gates; it does not claim that the legacy Transformer/GKR route is hardened.

## Statement and trust

The first closed loop proves one bounded, two-layer fixed-point dense network
with a ReLU between layers. Public inputs, public outputs, the complete public
model manifest, protocol version and request context are bound to the proof.
The verifier must receive an independently trusted model identifier; an identifier
supplied only alongside a proof is not a trust anchor.

All integer domains, dimensions, rounding and encoding are versioned. The initial
profile uses small signed integer inputs/weights and a fixed fractional-bit count;
values outside the supported domain are rejected. Rounding is round-half-up,
including negative ties. ReLU is max(0, q) on the signed rounded integer.

The baseline uses Plonky3's uni-STARK/FRI rather than importing the legacy
prover-selected sumcheck challenges. It makes no zero-knowledge claim, uses no
secret setup, and is not a claim of end-to-end production security. A configured
FRI query budget is not an independently audited whole-system security estimate.

The public model is small and travels in reusable verification material. Thus the
baseline verifier still reads small public model coefficients; it must not read
external model weight files, private activations or prover data, nor execute a
forward pass. Large-model verification via a reusable committed weight root and
GKR reductions remains a later milestone, not an implemented property of this
baseline.

## Acceptance gates and commits

A. Specify the statement, trust boundary and acceptance gates (this document).
B. Repair legacy fractional-tree root/shape/denominator validation with malicious
   proof regression tests. Keep legacy security limitations explicitly documented.
C. Add a separately scoped, bounded arithmetic relation, pinned proof backend,
   canonical statement binding and serialized proof verification. Test invalid
   public statements and deliberately dishonest witnesses.
D. Add separate prover/verifier executables and process-level acceptance tests.
   Prover support must be an optional feature absent from the verifier-only build.
E. Add an independent integer reference, boundary/domain tests, malformed-proof
   tests, reproducible lightweight measurements and CI coverage. Publish the exact
   results and remaining limitations. Commit every completed stage.

No large-model benchmark, CUDA dependency or training run is required for these
gates. Expanding to Transformer operators, succinct committed model weights,
MoE and chain verification is not implied by passing these small-network gates.

## Evidence required

- Honest proof accepts after a byte serialization round trip.
- Changing any input, output, model, protocol or request binding rejects.
- Constraints reject a dishonest matrix product, rounding remainder and ReLU.
- Empty/truncated/oversized/malformed/trailing proof data rejects without panic.
- A verifier-only binary verifies in a new process with only model verification
  material, statement and proof; no model weight directory or witness is present.
- Numerical reference comparisons distinguish exact fixed-point correctness from
  agreement with a floating-point model.
- Measurements separate ordinary evaluation, proving, verification and proof size.

## Protocol references

- Plonky3 0.7.0 uni-STARK implementation and tests:
  https://github.com/Plonky3/Plonky3/tree/v0.7.0/uni-stark
- Sumcheck transcript sequencing:
  https://microsoft.github.io/vega-prover/building-blocks/sumcheck.html
- WHIR (future committed-weight/reduction integration):
  https://eprint.iacr.org/2024/1586

## Stage B evidence

`cargo +1.94.1 test --release -p zkie-core --lib common::logup_gkr --locked`:
6 passed. New regressions reject missing layers, forged/zero roots, bad challenge
lengths, zero denominators, invalid input lengths and out-of-range lookup indices.
The terminal fraction is now checked against the recomputed tree root. This repair
alone does not fix legacy prover-selected challenges or make that route succinct.
