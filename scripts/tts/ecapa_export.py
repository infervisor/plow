#!/usr/bin/env python3
"""Export the speechbrain ECAPA-TDNN speaker embedding (spkrec-ecapa-voxceleb) for plowrt's host port.

    python scripts/tts/ecapa_export.py OUT_DIR [--ckpt embedding_model.ckpt]

Writes OUT_DIR/model.safetensors (all F32) and OUT_DIR/config.json, nothing else. Consumer:
crates/plowrt/src/tts/ecapa.rs. Layout (C = 1024, M = 3072, A = 128, E = 192):

  Conv weights are [Cout, K, Cin] (torch's [Cout, Cin, K] with the tap axis moved before Cin); 1x1 convs are
  [Cout, 1, Cin]. Every TDNN block is conv -> ReLU -> eval BatchNorm, so BN cannot fold into the conv; it is
  stored as `<p>.bn.scale` = gamma / sqrt(var + 1e-5) and `<p>.bn.shift` = beta - mean * scale.

  blocks.0.conv.{weight [C,5,80], bias}, blocks.0.bn.{scale, shift}
  blocks.{1,2,3}.tdnn1.{conv.*, bn.*}               1x1 C -> C
  blocks.{1,2,3}.res2net.{0..6}.{conv.*, bn.*}       [C/8, 3, C/8], dilation = block index + 1
  blocks.{1,2,3}.tdnn2.{conv.*, bn.*}               1x1 C -> C
  blocks.{1,2,3}.se.conv1.{weight [A,1,C], bias}, blocks.{1,2,3}.se.conv2.{weight [C,1,A], bias}
  mfa.{conv.*, bn.*}                                1x1 3C -> M
  asp.tdnn.conv.weight_{x,mean,std} [A,1,M]         the 3M-wide input [x, mean, std] split by column block
  asp.tdnn.conv.bias, asp.tdnn.bn.{scale, shift}
  asp.conv.{weight [M,1,A], bias}
  fc.{weight [E,1,2M], bias}                        asp_bn (eval, no activation before fc) folded in
"""
import argparse
import glob
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

BN_EPS = 1e-5
CHANNELS = [1024, 1024, 1024, 1024, 3072]
KERNELS = [5, 3, 3, 3, 1]
DILATIONS = [1, 2, 3, 4, 1]


def default_ckpt():
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/*/embedding_model.ckpt")))
    return hits[0] if hits else None


def bn_affine(sd, p):
    scale = sd[f"{p}.weight"] / torch.sqrt(sd[f"{p}.running_var"] + BN_EPS)
    return scale, sd[f"{p}.bias"] - sd[f"{p}.running_mean"] * scale


def conv_w(w):
    return w.permute(0, 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ckpt", default=default_ckpt())
    a = ap.parse_args()
    sd = {k: v.double() for k, v in torch.load(a.ckpt, map_location="cpu", weights_only=True).items()
          if not k.endswith("num_batches_tracked")}
    out = {}

    def tdnn(src, dst):
        out[f"{dst}.conv.weight"] = conv_w(sd[f"{src}.conv.conv.weight"])
        out[f"{dst}.conv.bias"] = sd[f"{src}.conv.conv.bias"]
        out[f"{dst}.bn.scale"], out[f"{dst}.bn.shift"] = bn_affine(sd, f"{src}.norm.norm")

    def conv(src, dst):
        out[f"{dst}.weight"] = conv_w(sd[f"{src}.conv.weight"])
        out[f"{dst}.bias"] = sd[f"{src}.conv.bias"]

    tdnn("blocks.0", "blocks.0")
    for b in (1, 2, 3):
        tdnn(f"blocks.{b}.tdnn1", f"blocks.{b}.tdnn1")
        for i in range(7):
            tdnn(f"blocks.{b}.res2net_block.blocks.{i}", f"blocks.{b}.res2net.{i}")
        tdnn(f"blocks.{b}.tdnn2", f"blocks.{b}.tdnn2")
        conv(f"blocks.{b}.se_block.conv1", f"blocks.{b}.se.conv1")
        conv(f"blocks.{b}.se_block.conv2", f"blocks.{b}.se.conv2")
    tdnn("mfa", "mfa")
    m = CHANNELS[-1]
    w = conv_w(sd["asp.tdnn.conv.conv.weight"])
    for i, n in enumerate(("x", "mean", "std")):
        out[f"asp.tdnn.conv.weight_{n}"] = w[:, :, i * m:(i + 1) * m]
    out["asp.tdnn.conv.bias"] = sd["asp.tdnn.conv.conv.bias"]
    out["asp.tdnn.bn.scale"], out["asp.tdnn.bn.shift"] = bn_affine(sd, "asp.tdnn.norm.norm")
    conv("asp.conv", "asp.conv")
    s, t = bn_affine(sd, "asp_bn.norm")
    fw = sd["fc.conv.weight"][:, :, 0]
    out["fc.weight"] = (fw * s[None, :])[:, None, :]
    out["fc.bias"] = sd["fc.conv.bias"] + fw @ t

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in out.items()}, str(od / "model.safetensors"))
    cfg = {"arch": "speechbrain.ECAPA_TDNN", "source": str(a.ckpt), "sample_rate_in": 24000, "sample_rate": 16000,
           "n_mels": 80, "n_fft": 400, "hop": 160, "channels": CHANNELS, "kernel_sizes": KERNELS,
           "dilations": DILATIONS, "res2net_scale": 8, "se_channels": 128, "attention_channels": 128,
           "lin_neurons": 192, "bn_eps": BN_EPS}
    (od / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"wrote {len(out)} tensors -> {od}")


if __name__ == "__main__":
    main()
