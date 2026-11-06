#!/usr/bin/env python3
"""Verify real DeepSeek-V2-Lite router score ranges against the TopKSelect
signed-range contract [-2^22, 2^22).

For every layer that has a real post-attention-norm activation and a router
weight, compute the Q16 logits and Q16 softmax probabilities and report:
  - max |logit| in Q16 units (the fixed-point value to_i64 would compare)
  - max softmax probability in Q16 units (the value the reference top-k uses)
  - max of the stored L*_gate_i32.bin (selected probabilities)

Usage:
  DS_WEIGHTS=models/deepseek-v2-lite/weights \
  GPT_WEIGHTS=models/gpt2/weights \
  python3 scripts/ds_router_range.py
"""
import os
import sys
import numpy as np

DS = os.environ.get("DS_WEIGHTS", "models/deepseek-v2-lite/weights")
GPT = os.environ.get("GPT_WEIGHTS", "models/gpt2/weights")
H = 2048
N = 64
F = 1 << 16
LIMIT = 1 << 22


def li(name):
    return np.fromfile(os.path.join(DS, name), dtype="<i4").astype(np.float64)


def rd(a, b):
    q = np.floor(a / b)
    r = a - q * b
    return q + (r * 2 >= b)


def main():
    exp_t = np.fromfile(os.path.join(GPT, "exp_table_i32.bin"), dtype="<i4").astype(np.float64)
    rows = []
    for L in range(1, 61):
        norm_f = os.path.join(DS, "L%d_post_attn_norm_i32.bin" % L)
        router_f = os.path.join(DS, "L%d_router_i32.bin" % L)
        if not (os.path.exists(norm_f) and os.path.exists(router_f)):
            continue
        norm = li("L%d_post_attn_norm_i32.bin" % L)
        seq = norm.size // H
        h2 = norm.reshape(seq, H)
        router = li("L%d_router_i32.bin" % L).reshape(H, N)
        logits16 = rd(h2 @ router, F)
        row_max = logits16.max(-1, keepdims=True)
        shifted = np.clip((logits16 - row_max) + (1 << 21), 0, (1 << 21) - 1).astype(np.int64)
        ev = exp_t[shifted]
        s = rd(ev * F, ev.sum(-1, keepdims=True))
        gate_max = float("nan")
        gate_file = os.path.join(DS, "L%d_gate_i32.bin" % L)
        if os.path.exists(gate_file):
            gate_max = np.max(li("L%d_gate_i32.bin" % L))
        rows.append((L, seq, np.max(np.abs(logits16)), np.max(s)))
        print("L%-3d seq=%-4d max|logits16|=%-12.0f max_prob_q16=%-8.0f gate_max=%-8.0f"
              % (L, seq, np.max(np.abs(logits16)), np.max(s), gate_max))
    if not rows:
        print("NO REAL ROUTER DATA FOUND")
        sys.exit(2)
    max_logit = max(r[2] for r in rows)
    max_prob = max(r[3] for r in rows)
    print("SUMMARY layers=%d  max|logits16|=%.0f  max_prob_q16=%.0f  LIMIT=%d (-2^22 excluded)"
          % (len(rows), max_logit, max_prob, LIMIT))
    print("logits-in-range=%s probabilities-in-range=%s"
          % (max_logit < LIMIT, max_prob < LIMIT))


if __name__ == "__main__":
    main()
