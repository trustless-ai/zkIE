# Authenticated integer graphs

The `zkie_verified::graph` API pins the entire typed program, tensor shapes,
numeric ranges, constants' commitments and output node in one verifier key.
The consumer must pin its expected key digest independently. Key preparation
endorses the supplied constants and checks their declared ranges; the proof does
not establish that those constants were trained by any particular organization.

The public statement contains one input vector, one output vector and a 32-byte
application context. Every other array is represented by a committed tensor.
The verifier authenticates public endpoints at transcript-derived points,
checks every dynamic tensor's bit decomposition, then checks all program gates.
Neither constant arrays nor intermediate arrays are verifier inputs.

Supported gates: affine combinations over coordinate views (slices, broadcasts,
transposes), elementwise products, matrix products, floor/nearest integer division
by a positive constant, and lookup in the first `actual_len` entries of a table.
Nearest division means floor((x + floor(d/2))/d), including for negative x.
Division proves both the equation and 0 <= remainder < d; non-power-of-two d is
handled by a second unsigned complement witness. Padded lookup entries cannot
be addressed. Constants and all main/range/remainder roots are bound before
arithmetic challenges. Lookup multiplicities and fraction-tree challenges follow
the lookup protocol's additional commitment rounds.

Program validation bounds each integer constraint's absolute residual below the
Goldilocks modulus. This prevents a nonzero integer residual from vanishing in
the field. Operations with unsafe declared profiles are rejected before proving
or verification. Range violations are rejected by the honest prover and also
constrained cryptographically for a modified prover.

Tests include a connected matmul / non-power-of-two rounding / affine / lookup
pipeline, a modified intermediate quotient, omitted constraints and unsafe range
profiles. This stage is infrastructure for the Transformer compiler, not evidence
of a full GPT-2 model execution. Proofs are not claimed to hide witnesses or to
have an independently audited aggregate soundness bound. The WHIR backend uses
its requested-128-bit profile and a degree-five extension field.
