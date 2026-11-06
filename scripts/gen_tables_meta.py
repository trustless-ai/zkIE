#!/usr/bin/env python3
"""Generate the model-distribution `tables.meta` sidecar for A tables.

This script is deliberately independent of the prover: it reads the public
function-table files and writes their 32-byte digests. The prover never writes
a digest. For GPT-2 the store ids are fixed by the builder push order:
exp=4, rsqrt=5, gelu=6 (wte/token_emb/pos_emb/x0 precede them).

Usage:
  python scripts/gen_tables_meta.py --model-dir models/gpt2/weights \
      --exp-id 4 --rsqrt-id 5 --gelu-id 6
"""
import argparse
import hashlib
import json
import os

P = (1 << 64) - (1 << 32) + 1
DOMAIN = b"zkie-a-table-v1"


def canonical_i32_words(path):
    with open(path, "rb") as f:
        b = f.read()
    if len(b) % 4:
        raise SystemExit(f"{path}: length is not a multiple of 4")
    vals = []
    for i in range(0, len(b), 4):
        v = int.from_bytes(b[i:i + 4], "little", signed=True)
        vals.append(v if v >= 0 else P - (-v))
    return vals, len(b) // 4


def digest(table_id, length, arity, values):
    h = hashlib.sha256()
    h.update(DOMAIN)
    h.update(int(table_id).to_bytes(8, "little"))
    h.update(int(length).to_bytes(8, "little"))
    h.update(int(arity).to_bytes(8, "little"))
    for v in values:
        h.update(int(v).to_bytes(8, "little"))
    raw = h.digest()
    return [int.from_bytes(raw[4 * i:4 * i + 4], "little") for i in range(8)]


def arity(length):
    if length == 0:
        raise SystemExit("empty table")
    return max(32, 1 << (length - 1).bit_length()).bit_length() - 1


def entry(table_id, kind, path, model_dir, file_path):
    name = os.path.join(model_dir, path)
    values, length = canonical_i32_words(name)
    a = arity(length)
    e = {
        "table_id": table_id,
        "kind": kind,
        "length": length,
        "arity": a,
        "file_path": file_path,
        "digest": digest(table_id, length, a, values),
    }
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/gpt2/weights")
    ap.add_argument("--exp-id", type=int, default=4)
    ap.add_argument("--rsqrt-id", type=int, default=5)
    ap.add_argument("--gelu-id", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.model_dir, "tables.meta")
    tables = [
        entry(args.exp_id, "compute", "exp_table_i32.bin", args.model_dir, None),
        entry(args.rsqrt_id, "file", "rsqrt_table_i32.bin", args.model_dir,
              "rsqrt_table_i32.bin"),
        entry(args.gelu_id, "file", "gelu_table_i32.bin", args.model_dir,
              "gelu_table_i32.bin"),
    ]
    with open(out, "w") as f:
        json.dump({"version": 1, "tables": tables}, f, indent=2)
        f.write("\n")
    print(f"wrote {out}: " + ", ".join(
        f"{t['table_id']}/{t['kind']}/{t['length']}/2^{t['arity']}" for t in tables))


if __name__ == "__main__":
    main()
