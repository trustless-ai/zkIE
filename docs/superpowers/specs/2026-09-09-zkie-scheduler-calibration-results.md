# zkIE scheduler calibration

Status: **procedure ready, measurements not yet taken.**

The scheduler ships with conservative defaults and a calibration procedure. This file
records what has been established by tests, and exactly what still has to be measured on
the calibration machine before the resource history is seeded.

## What is already established

- Admission budget 400 GiB, hard limit 440 GiB, 56 logical cores and aggregation fan-in 4
  are the defaults the CLI parses, and the exact byte conversion is covered by tests
  (`crates/zkie-runtime/tests/cli_config.rs`).
- A reservation is `max(static_request, ceil(max_observed_peak * 115 / 100))`, computed
  with checked integer arithmetic, and the history is keyed by the complete identity
  `circuit digest + k + proof flavor + execution backend + hardware profile`
  (`crates/zkie-runtime/src/estimate.rs`, `tests/scheduler_resources.rs`).
- The scheduler never exceeds RAM, CPU, GPU count or per-device VRAM, ages jobs at
  exactly 600 seconds, and retries execution twice and resource-exceeded work three
  times with the observed peak persisted before the retry decision.
- Memory enforcement keeps the scheduler outside the run cgroup, moves each worker into
  it, and freezes admission above the budget; the portable `ps`/`/proc` fallback is
  recorded as `best-effort` in run metadata and only used when the operator passes
  `--allow-best-effort-hard-limit`.

## Procedure on the calibration machine

Work from the checkout directory; never run a build-tree artifact directly.

```bash
CARGO_TARGET_DIR=/data/jimmyshi/cargo-build/zkie cargo install --locked \
  --path crates/zkie-runtime --root /data/jimmyshi/zkie-install
/data/jimmyshi/zkie-install/bin/zkie --version || true
```

1. One-shard baseline with 56 threads, 400 GiB admission and 440 GiB hard limit. Record
   the circuit digest, `k`, backend, flavor, static reservation, peak `memory.current`,
   worker peak RSS, wall time, proof size, verification time and key load time.
2. Repeat the same FinText shard three times sequentially to populate resource history,
   then run resource-compatible distinct shards concurrently. Confirm the reservation
   becomes `max(static, max_observed * 1.15)` and that the scheduler never admits more
   than 400 GiB or 56 logical cores.
3. Interrupt one worker below the hard limit and confirm `status` reports a requeued
   attempt rather than a cryptographic failure.
4. Exercise the hard-limit path only with synthetic memory workers capped far below host
   capacity, and confirm the recorded outcome is `resource-exceeded` with the observed
   peak and signal/exit status, never `verification-failed`.

## What is explicitly not generalised

FinText measurements must not be extrapolated to TimesFM: the two circuits have
different shapes, `k` and memory profiles, and the history key keeps them separate for
exactly this reason. A TimesFM number requires a TimesFM circuit measurement.

## Blocking factors

- The current development host is macOS, so the delegated-cgroup path cannot be
  exercised here; it is compiled under `cfg(target_os = "linux")` and its pure logic is
  covered by the fake-monitor tests in `crates/zkie-runtime/tests/resource_monitor.rs`.
- The `prove-queue`, `plan`, `prepare` and `verify` commands still need the backend
  registry wired to the proving crate before a real end-to-end calibration run can be
  driven from the CLI; `status` and `worker` are already wired.
- The EuroHPC/Deucalion voucher reply should request the CUDA module stack and the
  PyTorch/ONNX runtime used for the inference under proof, in addition to the proving
  stack, because the GPU partition is where the memory wall is expected to move.
