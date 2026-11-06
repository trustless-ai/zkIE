# Reverse-claim proof schedule (work in progress)

This document describes the intended committed verifier schedule. It is not a
claim that all operators or the complete protocol have been migrated or audited.
The current bound proof supports Add, MatMul, Projection, Scale, Transpose,
bounded ReLU, GeluIndex, Lookup, Argmax, the GatherMatMul expert-FFN core,
the Gather scatter-add with authenticated one-hot routing, Scatter with the
expert-major order proof (key column + masked strict-increase + LogUp shift
permutation), the linear index ops
CombineIdx/IdxBroadcast/PadIdx (CombineIdx also accepts bound intermediate
indices authenticated by their producer), the exact-product tail ops BroadcastScale and
ScaleVec (shift 0), and TopKSelect with pre-committed witness aux,
whole-column threshold broadcast, threshold attainment, verifier-authenticated
signed range proofs for x and thr_b (contract [-2^22, 2^22), out-of-range
fails closed) and a strict-tie fail-closed boundary (staged parts of S7; S7
itself remains incomplete; the real DeepSeek single layer has not been run). The
bound entry also supports the LanguageModel public statement (S6). Still fail
closed: BatchedMatMul, Softmax,
SoftmaxIndex, StableSoftmaxIndex, LayerNorm/RmsNorm/RoPE, ScaleVec with rounding
shifts, plus every op whose index inputs are not authenticated. TopKSelect's final acceptance still requires the gist/forward
deterministic smaller-index tie-break on real tied inputs (the current path
refuses boundary ties); Scatter routing and the real DeepSeek single layer are
not done. Gate consistency (TL hardening, 2026-11-04): a static (non-op-written)
Lookup table anywhere in the Gather.gate producer closure, or in the
BroadcastScale scale slot reached from Gather.weighted through Scale/ScaleVec,
now fails closed; this covers the real MoE graph pattern (the router gate enters
via BroadcastScale.scale). Limitation: only the weighted-side Scale/ScaleVec
chain is traversed; other ops (e.g. Add) mixed into that chain are outside this
fix. The real silu GeluIndex (table_len 2^24 > RELU_BOUND 2^22) remains
fail-closed and is a separate reduction item. GPT-2 support therefore still requires the
norm and softmax operators. The legacy committed verifier permanently rejects.
Model support must not be reported complete until the real graph has been
migrated and tested.

1. The caller supplies an independently trusted model manifest (op graph, tensor
   layout, fixed weight and lookup commitments) and authenticated public input
   and output data. Dynamic witness roots never replace model identities.
2. The prover computes virtual intermediates, then commits graph-ordered
   auxiliary witnesses needed by the supported ops. Each descriptor fixes the
   producing op, purpose, encoding, original shape, padded arity, column count,
   90-bit WHIR configuration and root. The verifier derives the expected list
   from the graph, rejects missing, duplicate, reordered or extra descriptors,
   and absorbs all descriptors and roots before sampling any output points.
   Remainder bits, lookup witness columns and ReLU sign bits are auxiliaries;
   activation tensors themselves remain virtual.
3. The transcript binds the full graph, trusted model identity, public execution
   data, and all auxiliary descriptors. It then derives output evaluation
   points. Public output vectors or independently committed outputs authenticate
   the resulting initial claims. A language-model implementation must instead
   bind the public token/argmax statement without exposing internal logits.
4. Process producers in deterministic reverse-topological order. Bind each
   ordered incoming claim list and merge claims before reducing the exact merged
   output point/value through its producing op. Bind terminal input claims before
   deriving later challenges. A lookup/remainder relation must authenticate the
   same precommitted auxiliary column at each derived point, and bind it to the
   op's actual input and output claims. Unsupported reductions fail closed.
5. Authenticate every terminal leaf against the caller's public input or the
   exact manifest-identified weight/table commitment. Batch only compatible
   same-shape columns and verify the exact required point/value/index multiset.

ReLU's bounded integer contract is x in [-2^22, 2^22 - 1]. Its current bit
witness proves the offset encoding x + 2^22 and uses the top bit as the sign
selector, while keeping x and the output virtual. The 90-bit setting above is
the WHIR configuration, not a claim that all base-field algebraic reductions
achieve 90-bit soundness. Legacy proof bytes are not compatible with this path.

For GeluIndex with w=x+offset and L=table_len-1, both w and w-L must lie in
[-2^22, 2^22-1], equivalently w in [-2^22+L, 2^22-1]. Offsets and table
lengths are also bounded by 2^22. Inputs outside this domain fail closed; the
actual GPT-2 activation range has not yet been validated against this contract.
