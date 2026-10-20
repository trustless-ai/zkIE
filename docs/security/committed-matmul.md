# Committed matrix multiplication, v2

This route implements C = A W using a reusable WHIR weight commitment and a
sequential Fiat–Shamir sumcheck over a degree-five Goldilocks extension field.
It is separate from the legacy base-field/XorShift reductions.

`prepare(k,n,weights)` validates and commits W once, returning a reusable in-memory
proving key and serializable verifier key. The latter contains only version,
dimensions and one Merkle root, not weights or Merkle prover data. Clients pin
the canonical key's SHA-256 identity independently. Preparation is deterministic
and transparent, not a secret setup. The publisher must endorse the correct
model/quantization; a key accepted only because a prover supplied it is unsafe.

`ProvingKey::prove(m,input,context)` returns a public input/output statement and
proof bytes. `verify(key,trusted_id,statement,bytes)` reads no weights, does not
multiply matrices and does not create PCS openings. The small 2→2→2 STARK API
remains available independently.

## Protocol

Commit W before any inference challenges. The outer transcript binds the full
statement, context, canonical key identity, WHIR domain/parameters and weight
root. Sample output row/column points u,v in the extension field. Starting from
the verifier's own evaluation of public C at (v,u), reduce

    C~(u,v) = sum_t A~(u,t) W~(t,v).

Each sumcheck message contains polynomial evaluations at 0,1,2. The verifier
checks p(0)+p(1)=previous claim, absorbs that message, then derives the next
extension-field challenge. There is no proof-supplied challenge or initial claim.
The final A evaluation comes from the public input; W's evaluation is authenticated
by a prescribed-point WHIR opening in the *same* transcript. The terminal product
must equal the accumulated sumcheck claim.

Points use explicitly little-endian coordinates in our MLE evaluation, converted
to the upstream big-endian order once at the PCS boundary. Weight padding to the
minimum WHIR arity is zero in preparation; extra coordinates are fixed to zero.
There are no trust-dependent interior evaluation arrays in the verifier API.

Shapes must be powers of two; each input/output/weight tensor is limited to 2^20
elements. Public inputs and prepared weights are in [-32767,32767]. Outputs are
signed i64 with |C_ij| <= k*32767^2. For honestly endorsed prepared models these
bounds are <p/2, so equality also has the intended integer-dot-product meaning.
The commitment does not independently certify a malicious publisher's weight
range; publication/preparation validation belongs to model endorsement.

WHIR uses a 128-bit requested UniqueDecoding profile, zero PoW, folding factor 32,
Goldilocks^5 challenges and upstream Poseidon2 constants. Requested security is
not an independent end-to-end audit. The additional reduction error is bounded
in the interactive model by (log2(m)+log2(n)+2log2(k))/|E|; non-interactive security
also relies on the Fiat–Shamir/random-oracle treatment and WHIR composition.

Verification work is linear in public A and C plus logarithmic sumcheck and PCS
verification; it is not linear in the private/resident weight array. Proof/key
decoders impose size/canonical/shape checks and fail closed on backend panics,
but service OS resource limits remain necessary. No ZK/privacy claim is made.

## Stage F evidence

Targeted release tests cover reusable keys for (m,k,n) = (1,1,1), (2,4,2),
(4,8,4), (1,32,8), two contexts per key, altered input/output/context/root,
changed sumcheck messages, invalid ranges/shapes and truncated encodings.
Serialized verifier keys in these cases are under 256 bytes.

This is a raw matrix-product primitive. Rounding, nonlinearity and graph
composition still require authenticated constraints; this document does not
claim that a full Transformer now uses this verifier.

The v2 reduction profile upgrades to Goldilocks^5 before full-model batching,
so the algebraic error budget does not depend on a nearly exhausted 128-bit
challenge field. The hash/PCS assumptions still cap security; extension size
alone is not a whole-protocol security estimate. v1 keys/proofs are rejected.
