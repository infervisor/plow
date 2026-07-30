#!/usr/bin/env python3
"""k3_logit_cmp.py <ref.f32> <plow_logits.bin> — elementwise, not by argmax."""
import sys

import numpy as np

ref = np.fromfile(sys.argv[1], np.float32)
raw = np.fromfile(sys.argv[2], np.uint16)
got = (raw.astype(np.uint32) << 16).view(np.float32)
assert ref.shape == got.shape, (ref.shape, got.shape)

d = got - ref
sc = np.abs(ref).max()
r = np.corrcoef(ref, got)[0, 1]
# cosine on the mean-centred logit vector — the only part that survives softmax
rc, gc = ref - ref.mean(), got - got.mean()
cos = float(rc @ gc / (np.linalg.norm(rc) * np.linalg.norm(gc)))
print(f"ref  |max|={sc:.4f} std={ref.std():.4f} top1={int(ref.argmax())}")
print(f"plow |max|={np.abs(got).max():.4f} std={got.std():.4f} top1={int(got.argmax())}")
print(f"max|d| = {np.abs(d).max():.4e}   rel(max) = {np.abs(d).max() / sc:.4e}")
print(f"rms|d| = {np.sqrt((d ** 2).mean()):.4e}   rel(rms) = "
      f"{np.sqrt((d**2).mean()) / ref.std():.4e}")
print(f"pearson = {r:.6f}   centred-cosine = {cos:.6f}")
t = np.argsort(-ref)[:8]
print("ref top8 :", [(int(i), round(float(ref[i]), 3)) for i in t])
print("plow @same:", [(int(i), round(float(got[i]), 3)) for i in t])
g = np.argsort(-got)[:8]
print("plow top8:", [(int(i), round(float(got[i]), 3)) for i in g])
