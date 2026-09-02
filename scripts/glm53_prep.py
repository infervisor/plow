#!/usr/bin/env python3
"""Build a plow GLM-5.3 checkpoint overlay without copying the 306 GB snapshot."""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import glm52_prep as P


FULL = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43}


def matrix(idx, name):
    return P.dequant_blockfp8(idx, name) if idx[name][3] == "F8_E4M3" else P.load_tensor(idx, name)


def add_bf16(w, name, value):
    shape = list(value.shape)
    w.add(name, "BF16", shape, int(np.prod(shape)) * 2, P.p_bf16(value))


def add_f32(w, name, value):
    value = value.float().cpu().numpy()
    shape = list(value.shape)
    w.add(name, "F32", shape, int(np.prod(shape)) * 4, P.p_f32(value))


def add_layer(idx, w, layer):
    p = f"model.language_model.layers.{layer}."
    for stage in ("attn", "ffn"):
        name = p + f"hc_{stage}_fn"
        add_f32(w, name, P.load_tensor(idx, name))

    if layer in FULL:
        a = p + "self_attn."
        nh, dk, qn, vd, ql = 64, 512, 256, 256, 1536
        # q_b_proj/kv_b_proj are stored in DIFFERENT dtypes in this checkpoint (fp8 vs bf16,
        # unlike GLM-5.2 where both are fp8) -- unify to float32 before the einsum, matching
        # glm52_prep.py's own derive-in-float convention.
        qb = matrix(idx, a + "q_b_proj.weight").view(nh, qn, ql).float()
        kvb = matrix(idx, a + "kv_b_proj.weight").view(nh, qn + vd, dk).float()
        kn, value = kvb[:, :qn, :], kvb[:, qn:, :]
        wqa = __import__("torch").einsum("hpl,hpk->hlk", kn, qb).contiguous()
        wuv = value.transpose(-1, -2).contiguous()
        kva = matrix(idx, a + "kv_a_proj_with_mqa.weight")
        add_bf16(w, a + "q_a_proj.weight", matrix(idx, a + "q_a_proj.weight"))
        add_bf16(w, a + "derived.q_absorb.weight", wqa.reshape(nh * dk, ql))
        add_bf16(w, a + "derived.kv_a_latent.weight", kva[:dk])
        add_bf16(w, a + "derived.v_absorb.weight", wuv.reshape(nh * dk, vd))
        add_bf16(w, a + "o_proj.weight", matrix(idx, a + "o_proj.weight"))
        wp = a + "indexer.weights_proj.weight"
        add_f32(w, wp, P.load_tensor(idx, wp))

    else:
        # KDA layer. devgen's declare_kda_weights (kda.rs) declares q/k/v_conv1d.weight and
        # o_norm.weight as F32 -- true for K3's checkpoint (what that code was written
        # against) but NOT this one: this checkpoint stores all four as BF16. Overriding
        # these exact checkpoint names in the sidecar (processed last, so it wins on the
        # name collision -- same mechanism the hc_*_fn tensors above rely on) upcasts them
        # to real F32 so the emitted F32-typed tensor handle gets a byte range that actually
        # matches its declared size, instead of a half-sized BF16 range read/written as if
        # it were F32.
        a = p + "self_attn."
        for name in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "o_norm.weight"):
            add_f32(w, a + name, P.load_tensor(idx, a + name))

    if layer >= 3:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = p + f"mlp.shared_experts.{proj}.weight"
            add_bf16(w, name, matrix(idx, name))


def link_snapshot(model, out):
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(model):
        if not (name.endswith(".safetensors") or name.endswith(".json")):
            continue
        dst = os.path.join(out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(model, name), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    link_snapshot(args.model, args.out)
    idx = P._index_shards(args.model)
    writer = P.STWriter()
    for layer in range(45):
        add_layer(idx, writer, layer)
        print(f"[glm53-prep] described layer {layer}/44", flush=True)
    sidecar = os.path.join(args.out, "zz-plow-glm53-derived.safetensors")
    writer.flush(sidecar)
    with open(os.path.join(args.out, "plow-glm53-prep.json"), "w") as f:
        json.dump({"source": os.path.realpath(args.model), "layers": 45, "full_attention": sorted(FULL)}, f)
    print(f"[glm53-prep] wrote {os.path.getsize(sidecar)/1e9:.2f} GB -> {sidecar}")


if __name__ == "__main__":
    main()
