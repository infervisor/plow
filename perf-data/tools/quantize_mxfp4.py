#!/usr/bin/env python3
"""Quantize the dense projections of a Gemma-4 / Qwen / Llama bf16 checkpoint to OCP MXFP4 weight
twins for the w4a16 decode ops (dev_isa.h PLOW_DOP_GEMV_MXFP4 91 / GEMV_GLU_MXFP4 92 /
GEMM_MXFP4 93; devgen `PLOW_MXFP4=1` on the dense family).

FORMAT (byte-identical to the GPT-OSS checkpoint the same ops consume):
  mxfp4/<name>        u8 [N, K/2]   e2m1 codes, LOW nibble = even k, sign in bit 3,
                                     magnitude LUT {0, .5, 1, 1.5, 2, 3, 4, 6}
  mxfp4/<name>_scale  u8 [N, K/32]  E8M0: scale = 2^(byte - 127), one per 32-K block
METHOD: per (row, 32-block) power-of-two scale chosen so |w/scale| <= 6 with no clipping:
  amax = m * 2^e0 (m in [1,2)) -> scale = 2^(e0-1) if m > 1.5 else 2^(e0-2), i.e. amax/scale in
  (3, 6]; codes = round-to-nearest onto the e2m1 grid (ties down). All-zero block -> scale 2^-3,
  codes 0. Same helpers / projection set as quantize_fp8.py; norms stay bf16, and so does the
  embedding TABLE — a tied model's head twin is a second copy the decode GEMV reads, not a
  replacement for the row lookup.

Usage: quantize_mxfp4.py <src-model-dir> <out-dir> [prefix] [--extra lm_head.weight]
  prefix default "model.language_model."; GPT-OSS: prefix "model." --extra lm_head.weight (attention
  projections + untied head; its experts are MXFP4 in the checkpoint already).
  A TIED checkpoint (Gemma, Qwen) also gets its embed/lm_head quantized, because devgen turns
  PLOW_MX4_HEAD on by default under --mxfp4; --no-tied-head opts out. To add that head to an
  ALREADY-quantized twin without rewriting it: --no-layers --out-name head_mx4.safetensors
  --extra <prefix>embed_tokens.weight — the loader mmaps every *.safetensors in the twin dir.
"""
import os
import struct
import json
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_fp8 import open_sources, source_chunks, raw_bytes, PROJS, GDN_PROJS, EXPERT_PROJS  # noqa: E402
import torch  # noqa: E402

MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def quant_rows(w):
    """w [nr, K] f32 -> (packed u8 [nr, K/2], scale u8 [nr, K/32])."""
    nr, K = w.shape
    if K % 32:
        raise ValueError(f"K={K} is not a multiple of 32")
    blk = w.view(nr, K // 32, 32)
    amax = blk.abs().amax(dim=2)
    mant, ex = torch.frexp(amax)  # amax = mant * 2^ex, mant in [0.5, 1)
    e = ex - 2 - (mant <= 0.75).to(ex.dtype)  # amax/2^e in (3, 6]
    e = e.clamp(-127, 127)
    scale = torch.exp2(e.to(torch.float32))
    v = (blk / scale.unsqueeze(2)).clamp(-6.0, 6.0)
    idx = torch.bucketize(v.abs(), MID)
    code = idx | ((v < 0).to(torch.int64) << 3)
    code = code.view(nr, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    sbyte = (e + 127).to(torch.uint8)
    return packed, sbyte


def build_plan(weight_map, shards, prefix, layers):
    plan = []
    for l in range(layers):
        for proj in PROJS + GDN_PROJS + EXPERT_PROJS:
            name = f"{prefix}layers.{l}.{proj}"
            if name not in weight_map:
                continue
            hdr, _, _, _ = shards[weight_map[name]]
            shape = list(hdr[name]["shape"])
            if len(shape) == 3:  # fused experts [E][N][K] -> flat rows [E*N][K] (ops 147-150 index e*N)
                shape = [shape[0] * shape[1], shape[2]]
            if len(shape) != 2:
                raise ValueError(f"{name}: expected a 2-D or 3-D weight, got {shape}")
            N, K = shape
            plan.append((f"mxfp4/{name}", f"mxfp4/{name}_scale", name, N, K))
    return plan


def add_top_level(plan, weight_map, shards, names):
    """Extra 2-D tensors outside the layer stack (e.g. an untied lm_head.weight [vocab, hidden])."""
    for name in names:
        if name not in weight_map:
            raise ValueError(f"{name}: not in checkpoint")
        hdr, _, _, _ = shards[weight_map[name]]
        N, K = hdr[name]["shape"]
        plan.append((f"mxfp4/{name}", f"mxfp4/{name}_scale", name, N, K))


def tied(src_dir):
    """`tie_word_embeddings` from config.json (nested under `text_config` on multimodal exports)."""
    path = os.path.join(src_dir, "config.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        cfg = json.load(f)
    return bool(cfg.get("text_config", cfg).get("tie_word_embeddings", False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("prefix", nargs="?", default="model.language_model.")
    parser.add_argument("--extra", action="append", default=[],
                        help="extra 2-D tensor to quantize (repeatable), e.g. lm_head.weight")
    parser.add_argument("--no-layers", action="store_true",
                        help="quantize only --extra tensors (e.g. a tied lm_head twin on its own)")
    parser.add_argument("--no-tied-head", action="store_true",
                        help="skip the tied embed/lm_head twin devgen's PLOW_MX4_HEAD default expects")
    parser.add_argument("--out-name", default="model.safetensors",
                        help="output file name inside <out-dir>")
    args = parser.parse_args()
    weight_map, shards = open_sources(args.src_dir)
    layers = 1 + max(int(k.split("layers.")[1].split(".")[0]) for k in weight_map if "layers." in k)
    plan = [] if args.no_layers else build_plan(weight_map, shards, args.prefix, layers)
    extra = list(args.extra)
    # TIED embed/lm_head (Gemma, Qwen). devgen turns PLOW_MX4_HEAD on by default under --mxfp4,
    # so the twin has to carry the head or the blob fails to load. The bf16 table stays in the
    # base checkpoint and the EMBED lookup still reads it — only the decode head GEMV reads this.
    if not args.no_layers and not args.no_tied_head and tied(args.src_dir):
        head = f"{args.prefix}embed_tokens.weight"
        if head in weight_map and head not in extra:
            extra.append(head)
    add_top_level(plan, weight_map, shards, extra)
    if not plan:
        raise ValueError("no supported projection weights found")
    print(f"src {args.src_dir}: {len(weight_map)} tensors, {layers} layers, {len(plan)} projections")

    meta, off = {}, 0
    for wname, sname, _src, N, K in plan:
        meta[wname] = {"dtype": "U8", "shape": [N, K // 2], "data_offsets": [off, off + N * K // 2]}
        off += N * K // 2
        meta[sname] = {"dtype": "U8", "shape": [N, K // 32], "data_offsets": [off, off + N * K // 32]}
        off += N * K // 32
    meta["__metadata__"] = {"format": "mxfp4", "quantizer": "perf-data/tools/quantize_mxfp4.py",
                            "scale_rule": "2^e with amax/2^e in (3,6], e2m1 RTN"}
    hdr_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    hdr_bytes += b" " * ((-len(hdr_bytes)) % 8)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, args.out_name)
    written = 0
    with open(out, "wb") as o:
        o.write(struct.pack("<Q", len(hdr_bytes)))
        o.write(hdr_bytes)
        for i, (wname, sname, src_name, N, K) in enumerate(plan):
            scale_parts = []
            for w, _raw in source_chunks(src_name, N, K, weight_map, shards):
                packed, sbyte = quant_rows(w)
                o.write(raw_bytes(packed))
                scale_parts.append(raw_bytes(sbyte))
            for sb in scale_parts:
                o.write(sb)
            written += N * K // 2 + N * K // 32
            if i % 40 == 0:
                print(f"  [{i + 1}/{len(plan)}] {src_name}  N={N} K={K}", flush=True)
    for _, _, mm, backing in shards.values():
        mm.close()
        backing.close()
    print(f"done: {out}  ({written / 1e9:.2f} GB over {len(plan)} projections)")


if __name__ == "__main__":
    main()
