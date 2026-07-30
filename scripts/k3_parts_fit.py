#!/usr/bin/env python3
"""k3_parts_fit.py <parts.npz> <plow_logits.bin> [name,name,...]

Least-squares plow's logit row onto the logit images of the reference's OWN
per-component hidden vectors. `logits = W @ (gamma * m) / rms(m)` is linear in `m`
up to one positive scalar, so a component whose fitted coefficient is not 1 (after
normalising the model's residual-stream terms to 1) is the one plow got wrong.
"""
import json
import mmap
import os
import struct
import sys

import numpy as np

SNAP = os.environ.get(
    "K3_DIR",
    "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
)


def raw(name):
    with open(os.path.join(SNAP, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][name]
    p = os.path.join(SNAP, shard)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    m = h[name]
    lo, hi = m["data_offsets"]
    f = open(p, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    return mm[8 + n + lo : 8 + n + hi], m


z = np.load(sys.argv[1])
rawl = np.fromfile(sys.argv[2], np.uint16)
got = (rawl.astype(np.uint32) << 16).view(np.float32)
keys = sys.argv[3].split(",") if len(sys.argv) > 3 else [
    k for k in z.files if k == "embed" or "_ffn" in k or "_attn" in k
]

gb, _ = raw("language_model.model.norm.weight")
gamma = (np.frombuffer(gb, np.uint16).astype(np.uint32) << 16).view(np.float32)
hb, hm = raw("language_model.lm_head.weight")
head = np.frombuffer(hb, np.uint16).reshape(*hm["shape"])

B = np.stack([z[k] * gamma for k in keys])
A = np.empty((len(keys), head.shape[0]), np.float32)
for i in range(0, head.shape[0], 16384):
    w = (head[i : i + 16384].astype(np.uint32) << 16).view(np.float32)
    A[:, i : i + 16384] = B @ w.T
M = A.T
c, *_ = np.linalg.lstsq(M, got, rcond=None)
r = got - M @ c
# normalise so the first non-embed component is 1
base = c[1] if len(c) > 1 else c[0]
print(f"{'component':16s} {'coef':>12s} {'/first':>9s}  |v|")
for k, v in zip(keys, c):
    print(f"{k:16s} {v:12.4f} {v/base:9.4f}  {np.linalg.norm(z[k]):.4f}")
print(f"residual rel = {np.sqrt((r**2).mean())/got.std():.4e}")
