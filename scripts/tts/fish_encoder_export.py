#!/usr/bin/env python3
"""Export the Fish S2 Pro codec ENCODER (modded DAC, fishaudio/s2-pro codec.pth) for plowc: the
reference-audio encoder of voice cloning (crates/devgen/src/fish_encoder.rs).

Writes OUT_DIR/model.safetensors (fp32, torch layouts, weight norm folded in fp64) and
OUT_DIR/config.json (model_type fish_dac_encoder). The official server encodes in fp32 with the
weight norm unfolded; an fp32 fold reproduces its codes exactly, a bf16 one does not.
  enc.c0.{w,b}                 CausalConv k7 1 -> 64
  enc.b{i}.ru{j}.{a1, c1.w, c1.b, a2, c2.w, c2.b}   residual units (dilation 1, 3, 9)
  enc.b{i}.snake / .down.{w,b} Snake, CausalConv k=2s stride s, C -> 2C
  xf.{l}.* / xf.norm           window-512 causal transformer after the last stage (wqkv q/k rows
                               permuted to half-split RoPE; wo_s / w2_s = LayerScale-folded)
  enc.snake / enc.out.{w,b}    Snake, CausalConv k3
  dn{u}.conv.{w,b}             CausalConv k2 s2 (no weight norm)
  dn{u}.cnx.*                  ConvNeXt (dw k7, LayerNorm, pw1, GELU, gamma-folded pw2 = 2.w_s/2.b_s)
  pre.{l}.* / pre.norm         window-128 causal transformer
  vq{k}.win / vq{k}.bin        in_proj [8][1024] + bias
  vq{k}.C                      L2-normalized codebook [size][8]: argmax of in_proj(r) . C is the
                               official cosine nearest neighbour
  vq{k}.T                      codebook @ out_proj^T + bias [size][1024]: subtracted from the residual

  HF_HOME=... python scripts/tts/fish_encoder_export.py OUT_DIR REVISION
"""
import json
import os
import sys

import torch

REPO = "models--fishaudio--s2-pro"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    from safetensors.torch import save_file

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    src = os.path.join(hub, REPO, "snapshots", sys.argv[2])
    sd = torch.load(f"{src}/codec.pth", map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    T = {}

    def put(name, t):
        T[name] = t.detach().double().float().contiguous()

    def wn(p):
        if f"{p}.weight_g" in sd:
            g, v = sd[f"{p}.weight_g"].double(), sd[f"{p}.weight_v"].double()
        elif f"{p}.parametrizations.weight.original0" in sd:
            g = sd[f"{p}.parametrizations.weight.original0"].double()
            v = sd[f"{p}.parametrizations.weight.original1"].double()
        else:
            return sd[f"{p}.weight"].double()
        return g * v / v.norm(dim=tuple(range(1, v.dim())), keepdim=True)

    dim, heads = 1024, 16
    hd = dim // heads
    perm = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])

    def transformer(pm, o):
        n = 0
        while f"{pm}.layers.{n}.attention.wqkv.weight" in sd:
            p, q = f"{pm}.layers.{n}", f"{o}.{n}"
            wqkv = sd[f"{p}.attention.wqkv.weight"].double()
            assert wqkv.shape == (3 * dim, dim), wqkv.shape
            qk = wqkv[:2 * dim].reshape(2 * heads, hd, dim)[:, perm].reshape(2 * dim, dim)
            put(f"{q}.wqkv", torch.cat([qk, wqkv[2 * dim:]]))
            put(f"{q}.attn_norm", sd[f"{p}.attention_norm.weight"])
            put(f"{q}.ffn_norm", sd[f"{p}.ffn_norm.weight"])
            put(f"{q}.w1", sd[f"{p}.feed_forward.w1.weight"])
            put(f"{q}.w3", sd[f"{p}.feed_forward.w3.weight"])
            put(f"{q}.wo_s", sd[f"{p}.attention_layer_scale.gamma"].double()[:, None]
                * sd[f"{p}.attention.wo.weight"].double())
            put(f"{q}.w2_s", sd[f"{p}.ffn_layer_scale.gamma"].double()[:, None]
                * sd[f"{p}.feed_forward.w2.weight"].double())
            n += 1
        put(f"{o}.norm", sd[f"{pm}.norm.weight"])
        return n

    e = "encoder.block"
    put("enc.c0.w", wn(f"{e}.0.conv"))
    put("enc.c0.b", sd[f"{e}.0.conv.bias"])
    strides = []
    i = 0
    while f"{e}.{1 + i}.block.4.conv.bias" in sd:
        p, o = f"{e}.{1 + i}.block", f"enc.b{i}"
        for j in range(3):
            r = f"{p}.{j}.block"
            put(f"{o}.ru{j}.a1", sd[f"{r}.0.alpha"].reshape(-1))
            put(f"{o}.ru{j}.c1.w", wn(f"{r}.1.conv"))
            put(f"{o}.ru{j}.c1.b", sd[f"{r}.1.conv.bias"])
            put(f"{o}.ru{j}.a2", sd[f"{r}.2.alpha"].reshape(-1))
            put(f"{o}.ru{j}.c2.w", wn(f"{r}.3.conv"))
            put(f"{o}.ru{j}.c2.b", sd[f"{r}.3.conv.bias"])
        w = wn(f"{p}.4.conv")
        # The fused input Snake of a widening conv reads a max(cin, cout) alpha (only cin used).
        a = sd[f"{p}.3.alpha"].reshape(-1).double()
        put(f"{o}.snake", torch.cat([a, torch.ones(w.shape[0] - a.shape[0], dtype=a.dtype)]))
        strides.append(w.shape[2] // 2)
        put(f"{o}.down.w", w)
        put(f"{o}.down.b", sd[f"{p}.4.conv.bias"])
        i += 1
    xf_layers = transformer(f"{e}.{i}.block.5", "xf")
    put("enc.snake", sd[f"{e}.{1 + i}.alpha"].reshape(-1))
    put("enc.out.w", wn(f"{e}.{2 + i}.conv"))
    put("enc.out.b", sd[f"{e}.{2 + i}.conv.bias"])

    q = "quantizer"
    downs = 0
    while f"{q}.downsample.{downs}.0.conv.weight" in sd:
        p, o = f"{q}.downsample.{downs}", f"dn{downs}"
        put(f"{o}.conv.w", sd[f"{p}.0.conv.weight"])
        put(f"{o}.conv.b", sd[f"{p}.0.conv.bias"])
        put(f"{o}.cnx.dw.w", sd[f"{p}.1.dwconv.conv.weight"])
        put(f"{o}.cnx.dw.b", sd[f"{p}.1.dwconv.conv.bias"])
        put(f"{o}.cnx.ln.w", sd[f"{p}.1.norm.weight"])
        put(f"{o}.cnx.ln.b", sd[f"{p}.1.norm.bias"])
        put(f"{o}.cnx.1.w", sd[f"{p}.1.pwconv1.weight"])
        put(f"{o}.cnx.1.b", sd[f"{p}.1.pwconv1.bias"])
        gamma = sd[f"{p}.1.gamma"].double()
        put(f"{o}.cnx.2.w_s", gamma[:, None] * sd[f"{p}.1.pwconv2.weight"].double())
        put(f"{o}.cnx.2.b_s", gamma * sd[f"{p}.1.pwconv2.bias"].double())
        downs += 1
    pre_layers = transformer(f"{q}.pre_module", "pre")

    sizes = []
    vqs = [f"{q}.semantic_quantizer.quantizers.0"] + [
        f"{q}.quantizer.quantizers.{k}" for k in range(9)]
    for k, p in enumerate(vqs):
        cb = sd[f"{p}.codebook.weight"].double()
        put(f"vq{k}.win", wn(f"{p}.in_proj")[:, :, 0])
        put(f"vq{k}.bin", sd[f"{p}.in_proj.bias"])
        put(f"vq{k}.C", cb / cb.norm(dim=1, keepdim=True).clamp_min(1e-12))
        put(f"vq{k}.T", cb @ wn(f"{p}.out_proj")[:, :, 0].t() + sd[f"{p}.out_proj.bias"].double()[None, :])
        sizes.append(cb.shape[0])

    save_file(T, os.path.join(out, "model.safetensors"))
    config = dict(
        model_type="fish_dac_encoder", sampling_rate=44100, frame_samples=2048,
        codebook_sizes=sizes, codebook_dim=T["vq0.win"].shape[0], channels=T["enc.c0.w"].shape[0],
        strides=strides, kernel=7, dilations=[1, 3, 9], out_kernel=T["enc.out.w"].shape[2],
        hidden=dim, heads=heads, head_dim=hd, intermediate=T["xf.0.w1"].shape[0], eps=1e-5,
        rope_theta=1e4, transformer=dict(layers=xf_layers, window=512),
        downsample=dict(count=downs, stride=2, convnext_kernel=7, ln_eps=1e-6),
        pre=dict(layers=pre_layers, window=128),
    )
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"wrote {out}: {len(T)} tensors")


if __name__ == "__main__":
    main()
