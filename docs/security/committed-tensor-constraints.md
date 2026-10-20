# Committed tensor constraints and dense graphs

The `oracle` compiler backend authenticates *every* terminal evaluation against a
WHIR tensor root. Constraint descriptors, view coordinate maps, degrees, message
counts and required openings come from trusted model code, never a proof's own
binding list. Views allow only constants and distinct selected coordinates;
diagonal/repeated-coordinate substitutions are rejected to preserve multilinearity.

For an elementwise relation R, all tensor commitments precede an extension-field
point gamma. A sequential sumcheck proves sum_x eq(gamma,x) R(x) = 0. Messages
are absorbed before deriving each round challenge. The verifier evaluates public
coefficient polynomials itself and obtains tensor evaluations exclusively from
PCS openings. No raw tensor/witness array is an argument to verification.

The dense graph route composes committed matmuls, exact round-half-up with bias,
signed 32-bit output decomposition, unsigned remainder decomposition, unused-high-
bit checks and optional ReLU. All roots are bound into the model/request transcript
before constraints are sampled. Public input/output are independently anchored to
their tensor roots by authenticated random evaluations. Shape and proof counts
are derived from the pinned key, so omitted constraints and swapped tensors reject.

Weights are bounded signed 16-bit values at endorsed preparation; contraction
dimensions <=65536 and signed-32-bit activations keep raw dot products below 2^62.
Shifts are 1..16, biases are signed 32-bit. Rounding residual equations remain
strictly below the field modulus, so the modular identities imply the intended
integer equations. Output overflow is rejected instead of silently truncating.
These bounds are a supported dense profile, not a statement that all GPT-2
normalization/index operators have already been lowered into this route.

Tests include an authenticated boolean constraint with a deliberately false
witness, two dense layers with different dimensions/shifts and a non-power-of-two
remainder bit count, changed output, omitted relation and swapped activation root.
The generic primitive is the foundation for the GPT-2 graph migration. Lookup,
Softmax and LayerNorm lowering still require their own accepted constraints.
