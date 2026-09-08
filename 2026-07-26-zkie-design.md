# zkIE — Zero-Knowledge Inference Engine

## Design Spec v1.0

**Date**: 2026-07-26
**Status**: Approved
**Project**: zkIE (Zero-Knowledge Inference Engine)

---

## 1. Project Overview

### 1.1 Positioning

SP1 is to RISC-V what zkIE is to AI inference: a modular, chip-based ZK proof system
where each "instruction" is an AI primitive (matmul, softmax, layer norm, etc.) rather
than a CPU opcode. The system produces EVM-verifiable proofs via Halo2 + KZG (BN254).

### 1.2 Deliverables (Phase 1)

1. **zkie-core** — AI primitive ISA with modular Halo2 chip implementations
2. **zkie-compiler** — ONNX graph → ISA instruction sequence compiler
3. **zkie-cli** — `zkie prove` / `zkie verify` / `zkie evm-verifier`
4. **ZkIEVerifier.sol** — ERC-8274 `IProofVerifier` compatible on-chain verifier
5. **End-to-end proof**: TimesFM 1.0 (200M params) inference verified on local Anvil chain

### 1.3 Non-Goals (Phase 1)

- Private weights/model (weights are public in the circuit)
- Training proofs (inference only)
- GPU acceleration (CPU proving first)
- Distributed proving
- MLIR integration (ONNX path only)
- Fused ATTENTION chip (v1 uses DOT_GENERAL + SOFTMAX + ELTWISE composition)

---

## 2. Architecture

### 2.1 Two-Layer Design

```
Layer 2: Model Compiler (zkie-compiler)
  - Parses ONNX graph via oxionnx-ops
  - Maps ONNX ops 1:1 to ISA instructions
  - Topological sort + virtual register allocation
  - Output: CompiledProgram { instructions,…2000 tokens truncated…300k gas (pairing check)

---

## 7. Implementation Plan

### Phase 1: Foundation (core abstractions)
- `fixed_point.rs` — I18/I36 fixed-point system
- `tensor.rs` — Tensor type with shape tracking
- `chip.rs` + `isa.rs` — ISA definition + Chip trait
- `lookup.rs` — Shared lookup table utilities
- Unit tests for all

### Phase 2: Chip Implementation (easiest to hardest)
- ELTWISE (simplest, validates Halo2 pipeline)
- REDUCE
- GELU (lookup-based)
- EMBED_LOOKUP
- SOFTMAX (lookup + DOT_GENERAL composition)
- LAYER_NORM
- DOT_GENERAL (core, most complex)
- PATCH_EMBED
- Independent unit tests per chip

### Phase 3: Compiler
- ONNX parsing
- Op → instruction mapping
- Graph compilation (topological sort + register allocation)
- End-to-end test with a tiny model

### Phase 4: TimesFM Integration
- Export TimesFM 1.0 to ONNX
- Compile TimesFM ONNX → ISA
- Circuit assembly + proof generation
- Verify inference correctness (compare with PyTorch output)

### Phase 5: On-Chain Verification
- Generate verifier.sol via `halo2-solidity-verifier`
- Implement ERC-8274 ZkIEVerifier wrapper
- Deploy to Anvil with Foundry
- Submit proof on-chain → verify passes

---

## 8. Success Criteria (Phase 1)

1. `zkie prove timesfm.onnx input.json` produces a valid Halo2 proof
2. `zkie verify proof.json vk.key` returns `true`
3. `zkie evm-verifier vk.key` generates `verifier.sol`
4. `ZkIEVerifier.sol` deploys to Anvil
5. On-chain `verify(proof)` returns `true`
6. Inference output matches reference PyTorch output (within quantization tolerance)

---

## 9. Future Directions (Post Phase 1)

- Fused ATTENTION chip (efficiency optimization)
- MLIR integration (PyTorch → Torch-MLIR → zkie dialect)
- Model weight privacy (weights as private witnesses)
- Recursive proof composition (proof of multiple inferences)
- GPU-accelerated proving
- Distributed proving across machines
- Gemma 4 / GPT-2 / Llama model support
