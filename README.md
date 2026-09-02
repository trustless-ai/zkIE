> ⚠️ **Prototype notice:** This is a prototype and does not implement full functionality.

# zkIE — Zero-Knowledge Inference Engine

**Take any ONNX model. Shard by layer. Prove in parallel. Verify on-chain.**

zkIE is a ZK proving system that compiles ONNX models into a purpose-built ISA of AI primitives, shards the workload for parallel proving across distributed workers, and aggregates shard proofs into a single on-chain-verifiable proof.

For the full design, see the [zkIE design notes](https://gist.github.com/JimmyShi22/23f4f1d3473ea1ed20db130f76390e76).

## Workspace

- `crates/zkie-core` — ZK-friendly ISA and proof stack.
- `crates/zkie-compiler` — ONNX parsing, pattern matching, compilation, and sharding.
- `engines/timesfm` — TimesFM partitioning engine.

## Build

```sh
cargo build
```

## Status

Prototype / work in progress.
