# Small-network protocol v2

## Scope and statement

The profile is fixed in code: dense 2→2, ReLU, dense 2→2. It is not configurable
by an untrusted proof. The public model has Q4 weights in [-8,8] and Q4 biases
in [-16,16]. Inputs are signed Q4 integers in [-128,127]. For each dense layer:

    raw[j] = sum_i(input[i] * weight[layer][j][i])
    q[j] = floor((raw[j] + 8) / 16) + bias[layer][j]
    hidden[j] = max(0, q[j])  # only after the first layer

Outputs are the second layer's q values. Public data consists of the model,
input, output, version and caller-provided 32-byte request context. There is no
proof of real-world input origin, freshness or output quality. Applications must
compare the context with their pending request and implement replay policy.

The client must obtain an expected ModelId independently. ModelId is SHA-256 of
a domain-separated canonical encoding: profile domain string (including its NUL
terminator), u32-le version, u8 fractional bits, eight i32-le weights in
[layer][output][input] order, then four i32-le biases in [layer][output] order.
Changing JSON whitespace/key order does not change the identity. Changing a
coefficient or profile does. The verifier validates the model's bounds before
encoding and compares both model and statement identity to the trusted id.

## Arithmetic relation

The AIR has 70 columns and eight rows. Each row independently describes the same
inference statement; repeating rows is a convenient tiny trace layout, not batch
inference or hiding. Columns 0..16 encode two inputs using eight boolean bits
each, with offset 128. Columns 16..52 encode four q values using nine boolean
bits each, with offset 256. Columns 52..68 encode four remainders using four
boolean bits each. Columns 68..70 are the ReLU outputs.

Every bit is constrained boolean. For each dot product:

    raw + 8 = (q - bias) * 16 + remainder,  0 <= remainder < 16

The high bit of q+256 equals one iff q is nonnegative. Thus
`hidden = q * high_bit(q+256)` constrains ReLU and binds its result into the
second dot product. Public inputs are equated to the decoded input columns;
public outputs are equated to the final q values. No intermediate value is an
unconstrained, independently supplied evaluation.

With bounded public coefficients, first-layer |raw| <= 2048. Every admissible
first-layer output is in [-144,144], so hidden is in [0,144]; second-layer
|raw| <= 2304 and output is in [-160,160]. These are within the nine-bit signed
domain. Even using the looser witness bounds hidden <=255, every constrained
rounding equation has absolute integer residual <9000. Consequently a residual
zero in Goldilocks is zero over the integers: there is no modular-wrap alias.
The remainder equation uniquely determines the specified round-half-up result,
including negative ties. All AIR constraints have degree at most two.

These bounds plus boolean decompositions establish that an exactly satisfied
AIR row gives the stated integer computation. This is an algebraic argument,
not a machine-checked theorem about the Rust implementation.

## Commitment and transcript

The proof uses pinned Plonky3 0.7.0 uni-STARK, TwoAdicFriPcs, Goldilocks,
degree-two extension challenges, Poseidon2Goldilocks<16> with upstream default
constants, an eight-field-element Merkle digest, blowup 8, 128 FRI queries,
final polynomial length 1, binary folding and no proof of work. The decoder
enforces the eight-row degree and this configuration is not proof-controlled.

The 48 public field elements are a 32-byte statement digest, the eight weights,
four biases, two inputs and two outputs. The digest hashes the profile domain,
statement version, ModelId, context, input and output in fixed order. Bytes are
embedded separately; signed small integers use canonical field embedding.

Plonky3's prover/verifier absorb the trace commitment and all public values
before sampling the AIR constraint combination challenge. Quotient commitments
are absorbed before the out-of-domain challenge; FRI has its own sequential
transcript. The verifier reconstructs every challenge from that transcript.
There is no prover-supplied XorShift challenge vector in this route.

Assumptions include SHA-256 identity binding, Poseidon2 collision resistance and
transcript security, FRI/DEEP-ALI soundness, correct backend implementation and
correct compilation. The raw query count does not establish 128-bit end-to-end
security; security estimates must account for extension size, constraint batching,
committed columns and concrete proximity bounds. No new audited-security claim
is made. The workspace's vendored p3-merkle-tree patch is part of the trusted
dependency surface and needs independent review alongside upstream dependencies.

## Encoding, separation and remaining costs

Proofs have an eight-byte `ZKIESTK2` header and canonical postcard payload, capped
at 1 MiB; trailing or noncanonical encodings and wrong degree are rejected.
Dependency decoding/verification panics fail closed via catch_unwind, which does
not catch process abort/OOM. Services still need OS resource limits. Format and
backend versions are intentionally pinned; upgrades need a new profile/version
and compatibility review.

`verify(model, trusted_id, statement, proof_bytes)` has no executor or witness
argument. Prover support is an opt-in Cargo feature. Verification material
contains the 12 small public coefficients: this baseline has not achieved
weight-size-independent verification of a large committed model. Likewise it
does not prove original floating-point behavior, tokenization or decoding.

## Stage C validation

Release-mode unit tests exercise an honest serialization round trip, altered
input/output/context/version/model binding, and modified-prover construction
of invalid boolean bits, rounding outputs, remainders and ReLU witnesses.
Stage D/E add independent-process, parser and numerical-reference evidence.


## Accepted v2 parameter gate

Stage E upgraded the development profile from 64 to 128 FRI queries and changed
its version/domain/header to v2. Plonky3's estimator includes the symbolically
counted 78 degree-two constraints and 72 batched base-field functions. Using a
conservative 127-bit extension-size input and a 128-bit collision cap yields
103 unique-decoding bits / 101 list-decoding bits; a regression requires >=96
estimated backend bits. This is conditional parameter evidence, not an
independent whole-system security proof. See `acceptance-report.md` for exact
local tests, measurements, limitations and reproducibility commands.
