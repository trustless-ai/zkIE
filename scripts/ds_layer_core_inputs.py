#!/usr/bin/env python3
"""Export the real DeepSeek-V2-Lite inputs for the S7 core-chain proof.

Runs the float64 reference forward (the same one used to validate the
fixed-point extractor) for a single token up to the chosen MoE layer, and
writes:

  L{L}_core_act_i32.bin     real post-attention RMSNorm activation, Q16 int32
  L{L}_core_scores_i32.bin  real router softmax scores, Q16 int32, computed
                            with the model's exact fixed-point recipe:
                            logits16 = rd(act @ router, 2^16),
                            idx = clip(logits16 - row_max + 2^21, 0, 2^21-1),
                            ev = exp_table[idx], scores = rd(ev*2^16, sum(ev))
  L{L}_core_logits16_i32.bin (diagnostic) Q16 router logits

The trajectory is a FLOAT64 REFERENCE model over bf16 weights, quantized once
at the Q16 contract boundary; it is NOT the fixed-point full-model trajectory
(the fixed-point forward of the whole model is not computed here). The router
scores from the quantized activation/weights are then computed exactly the
way the model's Softmax op computes them. The extracted
token is the first token of the existing tokens_i32.bin, so the input matches
the extracted model artifacts.

Usage:
  DS_LAYER=10 DS_WEIGHTS=/data/zkIE/.../weights \
  DS_RAW=/data/zkIE/.../raw GPT_WEIGHTS=/data/zkIE/models/gpt2/weights \
  python3 scripts/ds_layer_core_inputs.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_deepseek as E

LAYER = int(os.environ.get("DS_LAYER", "10"))
DS_WEIGHTS = os.environ.get("DS_WEIGHTS", "models/deepseek-v2-lite/weights")
DS_RAW = os.environ.get("DS_RAW", "models/deepseek-v2-lite/raw")
GPT_WEIGHTS = os.environ.get("GPT_WEIGHTS", "models/gpt2/weights")
E.RAW = DS_RAW
E.OUT = DS_WEIGHTS

H, N = E.H, E.N_ROUTED
HEADS, QK_NOPE, QK_ROPE, V_HEAD, KV_LORA = (
    E.HEADS, E.QK_NOPE, E.QK_ROPE, E.V_HEAD, E.KV_LORA)
ROPE_BASE = E.ROPE_BASE
F = 1 << 16
OFF = 1 << 21
TLEN = 1 << 21


def rms_norm(x, w, eps=1e-6):
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    return np.concatenate([x[..., :half] * cos - x[..., half:] * sin,
                           x[..., half:] * cos + x[..., :half] * sin], axis=-1)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def rd(a, b):
    a = np.asarray(a, dtype=np.int64)
    q = a // b  # floor division (b > 0)
    r = a - q * b
    return q + (r * 2 >= b)


def main():
    tokens = np.fromfile(os.path.join(DS_WEIGHTS, "tokens_i32.bin"), dtype="<i4")
    tok = int(tokens[0])
    print("token id %d, layer %d, seq 1" % (tok, LAYER))
    S = E.Sharded()
    emb = S.arr("model.embed_tokens.weight").astype(np.float64)
    x = emb[[tok]]  # [1, 2048]
    seq = x.shape[0]
    mscale = 0.1 * 0.707 * np.log(40.0) + 1.0
    softmax_scale = (QK_NOPE + QK_ROPE) ** -0.5 * mscale * mscale
    pos = np.arange(seq, dtype=np.float64)[:, None]
    inv = np.arange(QK_ROPE // 2, dtype=np.float64)[None, :]
    theta = pos * np.power(ROPE_BASE, -2.0 * inv / QK_ROPE)
    cos, sin = np.cos(theta), np.sin(theta)
    h = None
    for L in range(LAYER + 1):
        p = "model.layers.%d." % L
        in_norm = S.arr(p + "input_layernorm.weight").astype(np.float64)
        post_norm = S.arr(p + "post_attention_layernorm.weight").astype(np.float64)
        q_w = S.arr(p + "self_attn.q_proj.weight").astype(np.float64)
        kva_w = S.arr(p + "self_attn.kv_a_proj_with_mqa.weight").astype(np.float64)
        kva_ln = S.arr(p + "self_attn.kv_a_layernorm.weight").astype(np.float64)
        kvb_w = S.arr(p + "self_attn.kv_b_proj.weight").astype(np.float64)
        o_w = S.arr(p + "self_attn.o_proj.weight").astype(np.float64)
        residual = x
        hh = rms_norm(x, in_norm)
        qq = (hh @ q_w.T).reshape(seq, HEADS, QK_NOPE + QK_ROPE)
        q_nope, q_pe = qq[..., :QK_NOPE], qq[..., QK_NOPE:]
        ckv = hh @ kva_w.T
        compressed_kv, k_pe = ckv[..., :KV_LORA], ckv[..., KV_LORA:KV_LORA + QK_ROPE]
        kv = (kvb_w @ rms_norm(compressed_kv, kva_ln).T).T
        kv = kv.reshape(seq, HEADS, QK_NOPE + V_HEAD)
        k_nope, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]
        q_pe = apply_rope(q_pe, cos[:, None, :], sin[:, None, :])
        k_pe = apply_rope(k_pe[:, None, :], cos[:, None, :], sin[:, None, :])
        query = np.concatenate([q_nope, q_pe], axis=-1)
        key = np.concatenate([k_nope, np.broadcast_to(k_pe, (seq, HEADS, QK_ROPE))], axis=-1)
        sc = np.einsum("ihd,jhd->ihj", query, key) * softmax_scale
        mask = np.triu(np.full((seq, seq), -1e30), k=1)
        probs = softmax(sc + mask[:, None, :], axis=-1)
        attn = np.einsum("ihk,khd->ihd", probs, v).reshape(seq, HEADS * V_HEAD) @ o_w.T
        x = residual + attn
        h = rms_norm(x, post_norm)
        if L == LAYER:
            break
        # Full FFN/MoE of this layer (the reference trajectory needs it for
        # the next layer's input; omitting it would make h at LAYER wrong).
        residual = x
        if L == 0:
            gw = S.arr(p + "mlp.gate_proj.weight").astype(np.float64)
            uw = S.arr(p + "mlp.up_proj.weight").astype(np.float64)
            dw = S.arr(p + "mlp.down_proj.weight").astype(np.float64)
            mlp = silu(h @ gw.T) * (h @ uw.T)
            mlp = mlp @ dw.T
        else:
            router = S.arr(p + "mlp.gate.weight").astype(np.float64)
            sg = S.arr(p + "mlp.shared_experts.gate_proj.weight").astype(np.float64)
            su = S.arr(p + "mlp.shared_experts.up_proj.weight").astype(np.float64)
            sd = S.arr(p + "mlp.shared_experts.down_proj.weight").astype(np.float64)
            logits = h @ router.T
            scores = softmax(logits, axis=-1)
            topi = np.argsort(-scores, axis=-1)[:, :E.TOPK]
            topw = np.take_along_axis(scores, topi, axis=-1)
            mlp = np.zeros_like(h)
            for e in range(N):
                eg = S.arr(p + "mlp.experts.%d.gate_proj.weight" % e).astype(np.float64)
                eu = S.arr(p + "mlp.experts.%d.up_proj.weight" % e).astype(np.float64)
                ed = S.arr(p + "mlp.experts.%d.down_proj.weight" % e).astype(np.float64)
                for t in range(seq):
                    for kk in range(E.TOPK):
                        if topi[t, kk] == e:
                            out = silu(h[t] @ eg.T) * (h[t] @ eu.T)
                            mlp[t] += topw[t, kk] * (out @ ed.T)
            sh = silu(h @ sg.T) * (h @ su.T)
            mlp = mlp + sh @ sd.T
        x = residual + mlp
    act = np.clip(np.round(h.reshape(-1) * E.SCALE), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
    router = np.fromfile(os.path.join(DS_WEIGHTS, "L%d_router_i32.bin" % LAYER),
                         dtype="<i4").astype(np.int64).reshape(H, N)
    dot = act.astype(np.int64) @ router  # [N]
    logits16 = rd(dot, F).astype(np.int64)
    row_max = logits16.max()
    shifted = np.clip(logits16 - row_max + OFF, 0, TLEN - 1).astype(np.int64)
    exp_t = np.fromfile(os.path.join(GPT_WEIGHTS, "exp_table_i32.bin"), dtype="<i4").astype(np.int64)
    ev = exp_t[shifted]
    scores = rd(ev * F, ev.sum()).astype(np.int32)
    np.asarray(act, dtype=np.int32).tofile(os.path.join(DS_WEIGHTS, "L%d_core_act_i32.bin" % LAYER))
    np.asarray(scores, dtype=np.int32).tofile(os.path.join(DS_WEIGHTS, "L%d_core_scores_i32.bin" % LAYER))
    np.asarray(logits16, dtype=np.int32).tofile(os.path.join(DS_WEIGHTS, "L%d_core_logits16_i32.bin" % LAYER))
    order_s = np.argsort(-scores.astype(np.int64), kind="stable")[:E.TOPK]
    order_l = np.argsort(-logits16, kind="stable")[:E.TOPK]
    print("act: max|.|=%d  logits16: max|.|=%d  max score=%d" % (
        int(np.abs(act.astype(np.int64)).max()), int(np.abs(logits16).max()), int(scores.max())))
    print("topk by scores : %s  values %s" % (order_s.tolist(), scores[order_s].tolist()))
    print("topk by logits : %s  values %s" % (order_l.tolist(), logits16[order_l].tolist()))
    if order_s.tolist() != order_l.tolist():
        print("WARNING: score/logit top-k differ (fixed-point softmax rounding tie)")


if __name__ == "__main__":
    main()
