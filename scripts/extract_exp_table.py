#!/usr/bin/env python3
"""Generate the quantized exp lookup table for the attention softmax.

The softmax is computed in the numerically-stable form `softmax(x) =
softmax(x - max)`, so the exp input lives in `(-inf, 0]` and the exp output is
in `(0, 1]`. The table maps `x` in [-32, 0] (scale 2^16) to `exp(x)` at scale
2^16; below -32 the value is clipped to 0.

The table is per-model: it is written to `models/<model>/weights/`, which is
where that model's op-graph builder loads it from (e.g.
`models/gpt2/src/lib.rs`). Pass the model name as the first argument or via
`MODEL=...`; it defaults to gpt2.

Usage: python scripts/extract_exp_table.py [model]
"""
import os
import sys

import numpy as np

SCALE = 65536.0
OFFSET = 1 << 21       # x in [-32, 0] at scale 2^16
TABLE_SIZE = 1 << 21


def model_out_dir() -> str:
    model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "gpt2")
    path = os.path.join("models", model, "weights")
    os.makedirs(path, exist_ok=True)
    return path


def main() -> None:
    idx = np.arange(TABLE_SIZE, dtype=np.float64)
    x = (idx - OFFSET) / SCALE
    table = np.clip(np.round(np.exp(x) * SCALE), 0, 2**31 - 1).astype(np.int32)
    path = os.path.join(model_out_dir(), "exp_table_i32.bin")
    table.tofile(path)
    print(f"wrote {TABLE_SIZE} exp entries (x in [-32, 0]) to {path}")


if __name__ == "__main__":
    main()
