#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Bounded fixtures only; no model download or large-model benchmark.
cargo +1.94.1 fmt -p zkie-verified --check
cargo +1.94.1 test --release -p zkie-verified --features prover --locked
cargo +1.94.1 clippy --release -p zkie-verified --all-targets --features prover --locked -- -D warnings
cargo +1.94.1 clippy --release -p zkie-verified --all-targets --no-default-features --locked -- -D warnings
cargo +1.94.1 build --release -p zkie-verified --features prover --bins --locked
# A separate target directory prevents accidentally testing the prover-enabled
# library artifact under the name of a verifier-only build.
cargo +1.94.1 build --release -p zkie-verified --no-default-features \
    --bin zkie-verify --bin zkie-linear --bin zkie-gpt2 --target-dir target/verified-verifier --locked
python3 scripts/accept_verified.py target/release/zkie-prove \
    target/verified-verifier/release/zkie-verify

python3 scripts/accept_linear.py target/release/zkie-linear \
    target/verified-verifier/release/zkie-linear

python3 scripts/accept_gpt2.py target/release/zkie-gpt2 \
    target/verified-verifier/release/zkie-gpt2
