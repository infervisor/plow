#!/usr/bin/env python3
"""Export the Fish S2 Pro codec decoder (modded DAC, fishaudio/s2-pro codec.pth) for plowc.

Writes OUT_DIR/model.safetensors (fp32, torch layouts, weight norm folded in fp64) and
OUT_DIR/config.json. Decode path only:
  vq.T{k}            [4096][1024]   codebook_k @ out_proj_k^T + bias_k (row = code id; residual
                                    tables padded with their last row = the official clamp); the
                                    decoder input is sum_k vq.T{k}[code_k]
  post.layers.N.*    window-128 causal transformer; wqkv q/k rows permuted per head from
                     interleaved RoPE pairs to half-split ([evens, odds], as scripts/tts/fish_prep.py);
                     wo_s / w2_s = LayerScale-folded wo / w2
  up{i}.convt.{w,b}  CausalTransConv1d(k=2, s=2), torch [Cin][Cout][k]
  up{i}.cnx.*        ConvNeXt: dw k7 (+b), LayerNorm (w, b), pw1 (+b), pw2 (+b), gamma; 2.w_s / 2.b_s =
                     gamma-folded pw2
  dec.in.{w,b}       CausalConv k7 1024->1536
  dec.blk{i}.snake / .up.{w,b} / .ru{j}.{a1, c1.w, c1.b, a2, c2.w, c2.b}
  dec.out.snake / dec.out.{w,b}   then tanh

  HF_HOME=... python scripts/tts/fish_codec_export.py OUT_DIR REVISION
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
        """Weight-normed conv weight at prefix p (old weight_g/v or parametrizations), folded."""
        if f"{p}.weight_g" in sd:
            g, v = sd[f"{p}.weight_g"].double(), sd[f"{p}.weight_v"].double()
        elif f"{p}.parametrizations.weight.original0" in sd:
            g = sd[f"{p}.parametrizations.weight.original0"].double()
            v = sd[f"{p}.parametrizations.weight.original1"].double()
        else:
            return sd[f"{p}.weight"].double()
        dims = tuple(range(1, v.dim()))
        return g * v / v.norm(dim=dims, keepdim=True)

    q = "quantizer"
    sizes = []
    vqs = [f"{q}.semantic_quantizer.quantizers.0"] + [
        f"{q}.quantizer.quantizers.{k}" for k in range(9)]
    for k, p in enumerate(vqs):
        cb = sd[f"{p}.codebook.weight"].double()
        w = wn(f"{p}.out_proj")[:, :, 0]
        t = cb @ w.t() + sd[f"{p}.out_proj.bias"].double()[None, :]
        # The fast AR samples 4096 classes for every codebook; the codec clamps residual codes to
        # the last row (rvq.py), which a table padded with that row does without a clamp.
        t = torch.cat([t, t[-1:].expand(sizes[0] - t.shape[0], -1)]) if sizes else t
        put(f"vq.T{k}", t)
        sizes.append(t.shape[0])

    pm = f"{q}.post_module"
    dim, heads = 1024, 16
    hd = dim // heads
    perm = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])
    layers = 0
    while f"{pm}.layers.{layers}.attention.wqkv.weight" in sd:
        p, o = f"{pm}.layers.{layers}", f"post.layers.{layers}"
        wqkv = sd[f"{p}.attention.wqkv.weight"].double()
        assert wqkv.shape == (3 * dim, dim), wqkv.shape
        qk = wqkv[:2 * dim].reshape(2 * heads, hd, dim)[:, perm].reshape(2 * dim, dim)
        put(f"{o}.wqkv", torch.cat([qk, wqkv[2 * dim:]]))
        put(f"{o}.wo", sd[f"{p}.attention.wo.weight"])
        put(f"{o}.attn_norm", sd[f"{p}.attention_norm.weight"])
        put(f"{o}.ffn_norm", sd[f"{p}.ffn_norm.weight"])
        put(f"{o}.ls_attn", sd[f"{p}.attention_layer_scale.gamma"])
        put(f"{o}.ls_ffn", sd[f"{p}.ffn_layer_scale.gamma"])
        for s in ("w1", "w2", "w3"):
            put(f"{o}.{s}", sd[f"{p}.feed_forward.{s}.weight"])
        # LayerScale folded into the projection rows (the lowering's weights).
        put(f"{o}.wo_s", sd[f"{p}.attention_layer_scale.gamma"].double()[:, None] * sd[f"{p}.attention.wo.weight"].double())
        put(f"{o}.w2_s", sd[f"{p}.ffn_layer_scale.gamma"].double()[:, None] * sd[f"{p}.feed_forward.w2.weight"].double())
        layers += 1
    put("post.norm", sd[f"{pm}.norm.weight"])

    ups = 0
    while f"{q}.upsample.{ups}.0.conv.weight" in sd:
        p, o = f"{q}.upsample.{ups}", f"up{ups}"
        put(f"{o}.convt.w", sd[f"{p}.0.conv.weight"])
        put(f"{o}.convt.b", sd[f"{p}.0.conv.bias"])
        put(f"{o}.cnx.dw.w", sd[f"{p}.1.dwconv.conv.weight"])
        put(f"{o}.cnx.dw.b", sd[f"{p}.1.dwconv.conv.bias"])
        put(f"{o}.cnx.ln.w", sd[f"{p}.1.norm.weight"])
        put(f"{o}.cnx.ln.b", sd[f"{p}.1.norm.bias"])
        for s in ("pwconv1", "pwconv2"):
            put(f"{o}.cnx.{s[-1]}.w", sd[f"{p}.1.{s}.weight"])
            put(f"{o}.cnx.{s[-1]}.b", sd[f"{p}.1.{s}.bias"])
        put(f"{o}.cnx.gamma", sd[f"{p}.1.gamma"])
        gamma = sd[f"{p}.1.gamma"].double()
        put(f"{o}.cnx.2.w_s", gamma[:, None] * sd[f"{p}.1.pwconv2.weight"].double())
        put(f"{o}.cnx.2.b_s", gamma * sd[f"{p}.1.pwconv2.bias"].double())
        ups += 1

    d = "decoder.model"
    put("dec.in.w", wn(f"{d}.0.conv"))
    put("dec.in.b", sd[f"{d}.0.conv.bias"])
    blocks = []
    i = 0
    while f"{d}.{1 + i}.block.1.conv.bias" in sd:
        p, o = f"{d}.{1 + i}.block", f"dec.blk{i}"
        w = wn(f"{p}.1.conv")
        cin, cout, k = w.shape
        blocks.append(dict(cin=cin, cout=cout, kernel=k, stride=k // 2, dilations=[1, 3, 9]))
        put(f"{o}.snake", sd[f"{p}.0.alpha"].reshape(-1))
        put(f"{o}.up.w", w)
        put(f"{o}.up.b", sd[f"{p}.1.conv.bias"])
        for j in range(3):
            r = f"{p}.{2 + j}.block"
            put(f"{o}.ru{j}.a1", sd[f"{r}.0.alpha"].reshape(-1))
            put(f"{o}.ru{j}.c1.w", wn(f"{r}.1.conv"))
            put(f"{o}.ru{j}.c1.b", sd[f"{r}.1.conv.bias"])
            put(f"{o}.ru{j}.a2", sd[f"{r}.2.alpha"].reshape(-1))
            put(f"{o}.ru{j}.c2.w", wn(f"{r}.3.conv"))
            put(f"{o}.ru{j}.c2.b", sd[f"{r}.3.conv.bias"])
        i += 1
    put("dec.out.snake", sd[f"{d}.{1 + i}.alpha"].reshape(-1))
    put("dec.out.w", wn(f"{d}.{2 + i}.conv"))
    put("dec.out.b", sd[f"{d}.{2 + i}.conv.bias"])

    save_file(T, os.path.join(out, "model.safetensors"))
    config = dict(
        model_type="fish_dac", sampling_rate=44100, frame_samples=2048, frame_codes=len(sizes),
        codebook_sizes=sizes, latent_dim=dim,
        transformer=dict(layers=layers, dim=dim, heads=heads, head_dim=hd,
                         intermediate=T["post.layers.0.w1"].shape[0], eps=1e-5, window=128,
                         rope_theta=1e4),
        upsample=[dict(stride=2, kernel=2, convnext_kernel=7, ln_eps=1e-6)] * ups,
        decoder=dict(in_kernel=7, in_channels=dim, channels=T["dec.in.w"].shape[0], blocks=blocks,
                     out_kernel=7),
    )
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"wrote {out}: {len(T)} tensors")


if __name__ == "__main__":
    main()
