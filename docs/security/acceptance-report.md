# Verified inference acceptance report

For the subsequent committed-graph and GPT-2 compiler work, see
[the current protocol and acceptance report](gpt2-verified.md). This file
records the earlier bounded STARK milestone.

Date: 2026-10-20. Base: upstream/fork `39d8df9`. Scope: the first closed-loop
milestone defined in `verified-inference.md`, implemented as `zkie-verified`.
All code changes and tests were made in the user's fork, not the earlier analysis
clone. No model download, training, Transformer benchmark or CUDA run was made.

## Outcome

The bounded Q4 2→2→2/ReLU network has a portable, canonical proof format and an
independently compiled verifier-only executable. That executable accepts the
honest computation after the producer's directory is deleted. It receives only
public model verification material, an independently expected model id/request,
the public statement and proof. It has no forward executor API or witness input.

The model's small public coefficients are still in its verification material;
this is not verification independent of a large model's weight size. It is a
separate STARK baseline, not a rewrite/hardening of the legacy GKR Transformer
proof route. A production deployment or completion of Transformer/MoE support
is not part of the passed milestone.

## Commits by stage

| Stage | Commit | Result |
| --- | --- | --- |
| A | 1a53cb1 | Explicit statement, trust boundary and acceptance gates |
| B | 2177d9d | Complete fractional-tree/root/nonzero-denominator checks and regressions |
| C | a6ec8c6 | Bounded integer AIR, statement/model binding, serialized STARK proof API |
| D | ec79cb9 | Separate prover/verifier binaries and independent-process acceptance |
| E | This commit | v2 parameters, concrete budget gate, numerical/parser coverage, CI and this report |

The v1 development profile used 64 FRI queries. The full backend estimator
reported only 87 bits after AIR and 72-column batching were included, below the
96-bit gate. The accepted profile is v2 with 128 queries; its domain, frame header,
model version and fixture identity were changed. v1 bytes are rejected. No old
proof compatibility is implied by the development commits.

## Checks run

1. `cargo +1.94.1 test --release -p zkie-core --lib common::logup_gkr --locked`:
   6 passed, including three new regression groups. Repairs are deliberately
   limited; legacy challenge generation and other combination issues remain.
2. `cargo +1.94.1 test --release --workspace --lib --locked`:
   64 tests passed (24 core, 3 engine, 37 ops; other libraries have zero tests
   under default features). Core took 286 seconds with the existing WHIR/PoW
   tests and concurrent Rayon work; nothing was skipped. Existing legacy
   warnings, including a duplicate test attribute, remain.
3. `bash scripts/check_verified.sh`: format check, 8 new tests, strict Clippy,
   separate prover/verifier-only builds and cross-process acceptance passed.
   The new tests include 196,608 input pairs over three models against an
   independent i128/truncation-based reference, boundary/tie proof cases,
   modified-prover false witnesses, statement/model/context/version changes,
   malformed dimensions/openings, truncated/trailing/oversized encodings and
   512 deterministic malformed byte strings. These are finite tests, not a
   security theorem or exhaustive exploration of the model parameter space.
4. The independent-process Python acceptance uses a separate SHA-256 manifest
   encoder and exact Fraction-based evaluator. It checks expected request input
   and context, rejects an untrusted id and altered output, and checks invalid
   files return exit code 1. Generated files stay in a temporary directory.

CI was configured to run the new acceptance job and retain the workspace library
job. The remote GitHub Actions execution has not been observed: these are local
results, not a claim that remote CI is green. Rust is pinned to 1.94.1.

## Concrete backend budget

The regression uses Plonky3 0.7.0's `ProvenSecurity` estimator with eight trace
rows, 70 trace columns, 78 degree-two AIR constraints, no next-row opening,
72 batched base-field functions including quotient columns, conservative
127-bit quadratic-extension field size, 128-bit hash collision cap, blowup 8,
128 FRI queries and zero PoW. It reports unique-decoding 103 bits and
list-decoding 101 bits. The checked gate is >=96 bits.

This is conditional backend parameter accounting from an upstream estimator.
It does not independently prove the estimator, the Fiat–Shamir transformation,
hash instantiation, Rust/AIR correspondence or absence of implementation bugs.
The bounded-integer/no-wrap and rounding argument is in
`small-network-protocol.md`. An independent cryptographic review remains needed.

## Lightweight measurements

Local macOS arm64, release, Rust 1.94.1, fixed fixture, seven fresh processes
per prover/verifier measurement after the workspace suite completed. Exact CPU
model and hardware isolation were not recorded, so these numbers are diagnostic
and not a hardware-normalized performance claim.

| Measurement | Result |
| --- | ---: |
| Output for input [17,-9] | [9,-1] |
| Encoded proof | 84,670 bytes |
| Ordinary integer evaluation, median | 167 ns |
| Proving including witness/configuration/serialization, median | 2.679 ms |
| Verification including decode/configuration, median | 2.056 ms |

Timers are inside the binaries and exclude process startup and file IO. Tiny
evaluation timing is close to timer resolution; use it only to see the order
of magnitude. There is no separate measurement of STARK internal commit/open
stages, peak RSS or amortized large-model costs. Verification is much more
expensive than directly evaluating this tiny network; no economic speedup is
claimed. The purpose is an independently checkable protocol baseline.

## Reproduce and continue

Run `bash scripts/check_verified.sh`. Fixture identity:

    1143edd7832d9a74c700a73bbd23c52db6efaa33352d96c268c90797fed87428

An application must pin its own expected model id and compare the context to
its pending request. Context binding alone does not enforce freshness/replay
policy. Services must enforce OS time/memory limits: encoded byte bounds and
catch_unwind do not prevent process abort/OOM.

Next milestones: formal/independent protocol review; a reusable committed model
root and witness-free GKR reductions; missing Transformer operator constraints;
then complete-model and dynamic MoE graph coverage. Privacy, original floating-
point equivalence and chain verification are separate future requirements.
