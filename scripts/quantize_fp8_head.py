#!/usr/bin/env python3
"""Quantize the tied lm_head (embed_tokens) to per-row fp8 for PLOW_FP8_HEAD=1.

The fp8 checkpoint produced for the body weights does NOT carry the embedding
table, so a blob emitted with PLOW_FP8_HEAD=1 binds `fp8/<prefix>embed_tokens.*`
to nothing and the engine memory-faults at the first head GEMV (plowrt does not
refuse the missing tensor — see the task list). This writes the missing shard.

Convention matches the body quantizer exactly (verified against
layers.0.q_proj: scale = amax(row)/448, W8 = e4m3(W/scale), per-OUTPUT-row):
  fp8/<prefix>embed_tokens.weight        F8_E4M3 [vocab, hidden]
  fp8/<prefix>embed_tokens.weight_scale  F32     [vocab]

The embedding LOOKUP stays bf16 (reads the original table); only the head GEMV
reads this twin. MEASURED (Gemma-4-12B, MI300X, occ4 + both NRN folds):
11.744 -> 11.336 ms/token @4k, 11.860 -> 11.416 @8k — the head is a 2 GB bf16
read per token and this halves it. Own reporting row: vLLM's fp8 recipe keeps
lm_head bf16, so quality is NOT bit-comparable (greedy outputs stay factually
identical on the serve gate; free-form tails can flip a token).

Usage:
  python3 scripts/quantize_fp8_head.py <bf16_checkpoint_dir> <fp8_dir> \
      [--name model.language_model.embed_tokens.weight]
"""
import argparse
import glob
import json
import os
import struct

import numpy as np
import torch


def load_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("fp8_dir")
    ap.add_argument("--name", default="model.language_model.embed_tokens.weight")
    ap.add_argument("--chunk", type=int, default=16384)
    args = ap.parse_args()

    src = None
    for shard in sorted(glob.glob(os.path.join(args.ckpt_dir, "*.safetensors"))):
        h, base = load_header(shard)
        if args.name in h:
            src = (shard, h[args.name], base)
            break
    assert src, f"{args.name} not found in {args.ckpt_dir}"
    shard, meta, base = src
    assert meta["dtype"] == "BF16", meta["dtype"]
    v, hid = meta["shape"]
    o0, _ = meta["data_offsets"]
    print(f"quantizing {args.name} [{v}, {hid}] from {shard}")

    w8 = np.empty((v, hid), dtype=np.uint8)
    scales = np.empty(v, dtype=np.float32)
    with open(shard, "rb") as f:
        for r0 in range(0, v, args.chunk):
            r1 = min(r0 + args.chunk, v)
            f.seek(base + o0 + r0 * hid * 2)
            buf = f.read((r1 - r0) * hid * 2)
            t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16)
            t = t.reshape(r1 - r0, hid).to(torch.float32)
            sc = (t.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
            w8[r0:r1] = (t / sc[:, None]).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            scales[r0:r1] = sc.numpy()

    out = os.path.join(args.fp8_dir, "model-head.safetensors")
    n1, n2 = f"fp8/{args.name}", f"fp8/{args.name}_scale"
    sz1, sz2 = v * hid, v * 4
    hdr = {
        n1: {"dtype": "F8_E4M3", "shape": [v, hid], "data_offsets": [0, sz1]},
        n2: {"dtype": "F32", "shape": [v], "data_offsets": [sz1, sz1 + sz2]},
    }
    hb = json.dumps(hdr).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(w8.tobytes())
        f.write(scales.tobytes())
    print(f"wrote {out} ({8 + len(hb) + sz1 + sz2} bytes)")


if __name__ == "__main__":
    main()
