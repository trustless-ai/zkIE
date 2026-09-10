> ⚠️ **Prototype notice:** This is a prototype and does not implement full functionality.

# zkIE — Zero-Knowledge Inference Engine

**Take any ONNX model. Shard by layer. Prove in parallel. Verify on-chain.**

zkIE is a ZK proving system that compiles ONNX models into a purpose-built ISA of AI primitives, shards the workload for parallel proving across distributed workers, and aggregates shard proofs into a single on-chain-verifiable proof.

For the full design, see the [zkIE design notes](https://gist.github.com/JimmyShi22/23f4f1d3473ea1ed20db130f76390e76).

## Workspace

- `crates/zkie-core` — ZK-friendly ISA and proof stack.
- `crates/zkie-types` — shared run, shard, key and resource identities.
- `crates/zkie-compiler` — ONNX parsing, pattern matching, compilation, and sharding.
- `crates/zkie-prover` — witness generation, proof backends and native aggregation.
- `crates/zkie-runtime` — durable run state, atomic artifact/key stores, the
  resource-aware scheduler, memory enforcement, and the resumable `zkie` queue.
- `engines/timesfm` — TimesFM partitioning engine.

## Resource-aware proving queue

`zkie-runtime` persists every job in SQLite and only unlocks a job once all of its
predecessors are `Verified`. Reservations are `max(static, ceil(peak * 1.15))` per
`circuit + k + flavor + backend + hardware` identity; admission freezes above the
admission budget and a runaway attempt is terminated after three consecutive samples
above the hard limit. Proof bytes and keys are published through a content-addressed
store before the run state is allowed to move to `Verified`.

```sh
# 400 GiB admission, 440 GiB hard limit, 56 cores, aggregation fan-in 4
zkie prove-queue --run-dir /var/lib/zkie/run-1 \
  --memory-budget-gib 400 --memory-hard-limit-gib 440 \
  --cpu-budget-cores 56 --aggregation-fan-in 4

zkie status --run-dir /var/lib/zkie/run-1
```

`status` reports the overall state plus every job's stage, typed failure code and
retry count. The queue-driving commands still need the proving backend registry
wired from `zkie-prover`; see
`docs/superpowers/specs/2026-09-09-zkie-scheduler-calibration-results.md` for the
current boundary.

## Build

```sh
cargo build
```

## Status

Prototype / work in progress.
