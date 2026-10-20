#!/usr/bin/env python3
"""Independent-process acceptance; stdlib only, bounded tiny-network work.

Usage: python3 scripts/accept_verified.py PROVER_BINARY VERIFIER_ONLY_BINARY
Build the second binary with --no-default-features in a separate target dir.
No generated artifacts are written outside this script's temporary directory.
"""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import statistics
import subprocess
import sys
import tempfile


def model_id(model):
    """Independent implementation of the specified canonical model encoding."""
    h = hashlib.sha256(b"zkie-verified/dense2x2-q4-relu/stark-v2\0")
    h.update(struct.pack("<IB", model["version"], model["fractional_bits"]))
    for layer in model["weights"]:
        for row in layer:
            for w in row:
                h.update(struct.pack("<i", w))
    for row in model["biases"]:
        for b in row:
            h.update(struct.pack("<i", b))
    return h.hexdigest()


def reference(model, values):
    """Exact rational reference; independent of the Rust trace builder."""
    from fractions import Fraction
    for layer in range(2):
        result = []
        for row, bias in zip(model["weights"][layer], model["biases"][layer]):
            scaled = sum(Fraction(x * w, 16) for x, w in zip(values, row))
            rounded = (scaled + Fraction(1, 2)).__floor__() + bias
            result.append(max(0, rounded) if layer == 0 else rounded)
        values = result
    return values


def command(args, cwd, success=True):
    p = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True, timeout=30)
    if success and p.returncode != 0:
        raise AssertionError(f"command failed ({p.returncode}): {p.stderr}")
    if not success and p.returncode != 1:
        raise AssertionError(f"invalid proof/request should exit 1, got {p.returncode}: {p.stderr}")
    return p.stdout


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    prover, verifier = (Path(p).resolve(strict=True) for p in sys.argv[1:])
    fixtures = Path(__file__).resolve().parents[1] / "crates/zkie-verified/fixtures"
    model = json.loads((fixtures / "model.json").read_text())
    request = json.loads((fixtures / "request.json").read_text())
    expected_id = model_id(model)
    with tempfile.TemporaryDirectory(prefix="zkie-verified-") as temp:
        temp = Path(temp)
        producer = temp / "producer"
        consumer = temp / "consumer"
        producer.mkdir()
        consumer.mkdir()
        (producer / "model.json").write_text(json.dumps(model))
        (producer / "request.json").write_text(json.dumps(request))
        proving_samples = [json.loads(command([prover, "model.json", "request.json", f"bundle{i}"], producer))
                           for i in range(7)]
        proving = proving_samples[0]
        assert proving["model_id"] == expected_id
        assert proving["output"] == reference(model, request["input"])
        # Transfer only the documented public verification inputs and proof.
        for name in ("model.json", "statement.json", "proof.bin"):
            shutil.copyfile(producer / "bundle0" / name, consumer / name)
        (consumer / "request.json").write_text(json.dumps(request))
        # A local binary in an otherwise empty directory, with producer artifacts
        # gone, cannot find a relative weights/activation tree accidentally.
        shutil.copy2(verifier, consumer / "verify")
        shutil.rmtree(producer)
        args = [consumer / "verify", "model.json", expected_id, "request.json", "statement.json", "proof.bin"]
        verified_samples = [json.loads(command(args, consumer)) for _ in range(7)]
        verified = verified_samples[0]
        assert verified["verified"] is True and verified["output"] == proving["output"]
        assert command([consumer / "verify", "id", "model.json"], consumer).strip() == expected_id
        statement = json.loads((consumer / "statement.json").read_text())
        bad = dict(statement, output=[statement["output"][0] + 1, statement["output"][1]])
        (consumer / "statement.json").write_text(json.dumps(bad))
        command(args, consumer, success=False)
        (consumer / "statement.json").write_text(json.dumps(statement))
        wrong_request = dict(request, context=[8] * 32)
        (consumer / "request.json").write_text(json.dumps(wrong_request))
        command(args, consumer, success=False)
        (consumer / "request.json").write_text(json.dumps(request))
        command(args[:2] + ["00" * 32] + args[3:], consumer, success=False)
        proof = (consumer / "proof.bin").read_bytes()
        for malformed in (b"", proof[:10], proof + b"trailing", b"x" * ((1 << 20) + 1)):
            (consumer / "proof.bin").write_bytes(malformed)
            command(args, consumer, success=False)
        print(json.dumps({"acceptance": "passed", "independent_reference": True,
                          "model_id": expected_id, "prover": proving, "verifier": verified,
                          "measurements": {"samples": 7,
                              "evaluate_ns_median": statistics.median(p["evaluate_ns"] for p in proving_samples),
                              "prove_us_median": statistics.median(p["prove_us"] for p in proving_samples),
                              "verify_us_median": statistics.median(v["verify_us"] for v in verified_samples)}}, indent=2))


if __name__ == "__main__":
    main()
