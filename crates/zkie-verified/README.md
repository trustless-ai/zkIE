# zkie-verified

A bounded, non-ZK fixed-point network proving baseline. This is deliberately
separate from the legacy Transformer/GKR implementation. It does not make that
implementation sound or make its published benchmarks apply to this protocol.

The relation is a 2-input, 2-hidden, 2-output dense network with ReLU between
layers, Q4 scaling and round-half-up. The verifier reads a small public model
manifest, a separately trusted model id, the public statement and proof bytes.
It does not run the network or accept activations/prover data.

Public coefficients are bounded to weights [-8,8], biases [-16,16]; inputs are
[-128,127]. The exact relation and trust boundary are documented in
`docs/security/small-network-protocol.md` at the workspace root.

```sh
cargo +1.94.1 test --release -p zkie-verified --features prover --locked
```

The default build exposes verification only. Enable `prover` to call `evaluate`
or `prove`. Model/statement types support strict JSON; proofs use versioned
binary framing with canonical postcard encoding and a 1 MiB encoded size cap.

No secret setup and no zero-knowledge claim. This small-network baseline has not
received an independent cryptographic audit. It is not a generic ONNX compiler,
Transformer implementation, committed-large-weight verifier or chain verifier.


## Separate processes

Build the prover and a genuinely verifier-only target separately:

```sh
cargo +1.94.1 build --release -p zkie-verified --features prover --bins --locked
cargo +1.94.1 build --release -p zkie-verified --no-default-features --bin zkie-verify --target-dir target/verified-verifier --locked
python3 scripts/accept_verified.py target/release/zkie-prove target/verified-verifier/release/zkie-verify
```

To use the fixture manually, from the repository root:

```sh
target/release/zkie-prove crates/zkie-verified/fixtures/model.json crates/zkie-verified/fixtures/request.json /tmp/zkie-new-bundle

target/verified-verifier/release/zkie-verify /tmp/zkie-new-bundle/model.json 1143edd7832d9a74c700a73bbd23c52db6efaa33352d96c268c90797fed87428 crates/zkie-verified/fixtures/request.json /tmp/zkie-new-bundle/statement.json /tmp/zkie-new-bundle/proof.bin
```

The output directory must not exist. The trusted id above is specific to the
checked-in fixture, not an instruction to trust ids emitted by any prover.
The verifier also compares input/context to the independently expected request.
The acceptance script deletes the producer directory before running a copied
verifier-only binary with just public files and proof in a separate directory.
It also checks incorrect outputs/requests/model ids and malformed proof files.
