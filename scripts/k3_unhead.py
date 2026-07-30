#!/usr/bin/env python3
"""k3_unhead.py — recover the final normed hidden row from a logit dump, then peel the
model-level AttnRes to recover plow's PREFIX SUM, and report it against the reference's.

The control is built in: the same inversion is run on the reference's own logits, whose
hidden row is known, so the recovery error is measured rather than assumed.

  python3 scripts/k3_unhead.py <ref.h.npy> <plow_logits.bin> [n_rows]
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
EPS = 1e-5


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


tr = np.load(sys.argv[1])
rl = np.fromfile(sys.argv[2], np.uint16)
got = (rl.astype(np.uint32) << 16).view(np.float32)
R = int(sys.argv[3]) if len(sys.argv) > 3 else 40000

hb, hm = raw("language_model.lm_head.weight")
head = np.frombuffer(hb, np.uint16).reshape(*hm["shape"])
gb, _ = raw("language_model.model.norm.weight")
gamma = (np.frombuffer(gb, np.uint16).astype(np.uint32) << 16).view(np.float32)
ob, _ = raw("language_model.model.output_attn_res_norm.weight")
pb, _ = raw("language_model.model.output_attn_res_proj.weight")
onm = (np.frombuffer(ob, np.uint16).astype(np.uint32) << 16).view(np.float32)
opj = (np.frombuffer(pb, np.uint16).astype(np.uint32) << 16).view(np.float32)
score_w = onm * opj

rng = np.random.default_rng(0)
idx = np.sort(rng.choice(head.shape[0], R, replace=False))
A = (head[idx].astype(np.uint32) << 16).view(np.float32)
G = A.T @ A + 1e-2 * np.eye(A.shape[1], dtype=np.float32)
L = np.linalg.cholesky(G.astype(np.float64))


def solve(logits):
    b = A.T.astype(np.float64) @ logits[idx].astype(np.float64)
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


e = tr[0]
prefix_ref = tr[-2]
xn_ref = tr[-1]

# control: invert the reference's own logits
ref_logits = np.empty(head.shape[0], np.float32)
for i in range(0, head.shape[0], 16384):
    w = (head[i : i + 16384].astype(np.uint32) << 16).view(np.float32)
    ref_logits[i : i + 16384] = w @ xn_ref
xr = solve(ref_logits)
print(f"control: |xn_rec - xn_ref| / |xn_ref| = "
      f"{np.linalg.norm(xr - xn_ref)/np.linalg.norm(xn_ref):.3e}")

xp = solve(got.astype(np.float64))
u = xp / gamma            # = m / rms(m)


def probs(v0, v1):
    k0 = v0 / np.sqrt((v0 ** 2).mean() + EPS)
    k1 = v1 / np.sqrt((v1 ** 2).mean() + EPS)
    s = np.array([k0 @ score_w, k1 @ score_w])
    s = np.exp(s - s.max())
    return s / s.sum()


p = probs(e, prefix_ref)
for _ in range(50):
    c = np.sqrt((u ** 2).mean()) ** -1 * 1.0
    # m = c*u ; solve for c so that rms(m) = c*rms(u) is consistent: any c works for the
    # direction, so fix c by requiring prefix = (m - p0 e)/p1 to reproduce p.
    # Sweep c on a log grid, keep the fixed point.
    break
best = None
for c in np.exp(np.linspace(np.log(0.05), np.log(500.0), 4000)):
    m = c * u
    pr = (m - p[0] * e) / p[1]
    q = probs(e, pr)
    d = abs(q[0] - p[0])
    if best is None or d < best[0]:
        best = (d, c, pr, q)
    p2 = q
_, c, prefix_plow, q = best
# one refinement pass with the converged probs
p = q
best = None
for c in np.exp(np.linspace(np.log(0.05), np.log(500.0), 4000)):
    m = c * u
    pr = (m - p[0] * e) / p[1]
    qq = probs(e, pr)
    d = abs(qq[0] - p[0])
    if best is None or d < best[0]:
        best = (d, c, pr, qq)
_, c, prefix_plow, q = best
print(f"model AttnRes probs: ref {probs(e, prefix_ref)}  plow {q}")
pr, pf = prefix_plow, prefix_ref
print(f"|prefix| ref {np.linalg.norm(pf):.4f}  plow {np.linalg.norm(pr):.4f}  "
      f"ratio {np.linalg.norm(pr)/np.linalg.norm(pf):.4f}  "
      f"cos {pr@pf/np.linalg.norm(pr)/np.linalg.norm(pf):.4f}")

# per-component decomposition against the reference's parts
pz = sys.argv[1].replace(".h.npy", ".parts.npz")
if os.path.exists(pz):
    z = np.load(pz)
    keys = [k for k in z.files if k != "embed" and "_ffn" not in k] + \
           [k for k in z.files if "_ffn0" in k]
    B = np.stack([z[k] for k in keys]).T
    coef, *_ = np.linalg.lstsq(B, pr, rcond=None)
    res = pr - B @ coef
    print("prefix_plow in the reference's component basis:")
    for k, cf in zip(keys, coef):
        print(f"   {k:12s} {cf: .4f}   |v|={np.linalg.norm(z[k]):.4f}")
    print(f"   residual |r|/|prefix| = {np.linalg.norm(res)/np.linalg.norm(pr):.4f}")
