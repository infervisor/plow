#!/usr/bin/env python3
"""Read safetensors HEADERS only (mmap, no bulk load) for Kimi-K3 KDA layers.

Usage: kda_hdr.py <layer_idx> [<layer_idx> ...]
"""
import json
import os
import struct
import sys

SNAP = os.environ.get(
    "K3_SNAP",
    "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
)


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def index():
    with open(os.path.join(SNAP, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def main():
    wm = index()
    layers = [int(a) for a in sys.argv[1:]] or [0]
    hdrs = {}
    for li in layers:
        pre = f"language_model.model.layers.{li}.self_attn."
        keys = sorted(k for k in wm if k.startswith(pre))
        print(f"=== layer {li}: {len(keys)} self_attn tensors, shard {wm[keys[0]]}")
        for k in keys:
            sh = wm[k]
            if sh not in hdrs:
                hdrs[sh] = header(os.path.join(SNAP, sh))
            h, _ = hdrs[sh]
            m = h[k]
            print(f"  {k[len(pre):]:28s} {m['dtype']:6s} {m['shape']}")


if __name__ == "__main__":
    main()
