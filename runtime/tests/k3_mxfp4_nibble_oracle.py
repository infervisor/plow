#!/usr/bin/env python3
# k3_mxfp4_nibble_oracle.py — settle the ONE unverified bit of plow's MXFP4 encoding. [K3-MXFP4]
#
# `perf-data/kimi-k3-kernel-gap.md` §4c: everything about plow's mxfp4 layout was verified
# byte-exact against the Kimi-K3 checkpoint — `weight_packed [N, K/2]`, `weight_scale [N, K/32]`,
# group 32 along K, E8M0 bias 127 confirmed empirically from the scale bytes — EXCEPT the nibble
# order, and that section is explicit that the data provably CANNOT settle it:
#
#   "The measured low/high nibble histograms are statistically identical [...] so the data cannot
#    distinguish the two orders — a nibble swap permutes elements within a byte and leaves every
#    per-block multiset unchanged. This is the one MXFP4 fact that is INFERRED rather than
#    verified. It costs one comparison at bringup [...] Do it once; if it is wrong, every mxfp4
#    number is garbage in a way that looks like 'the model is just bad'."
#
# This is that comparison, and it is a GEMV rather than a dequant diff because a GEMV is what the
# kernel actually does: `fp4_to_bf16v8x4` decodes and accumulates in one instruction, so a dequant
# helper written on the host could agree with the doc and disagree with the hardware.
#
# It emits BOTH references — element 2i in the LOW nibble (plow's claim, and
# `compressed_tensors.pack_fp4_to_uint8`'s convention) and element 2i in the HIGH nibble — off ONE
# real Kimi-K3 expert tensor. The device answer must match exactly one of them, and the harness
# says which. A swap is not a small error: it permutes weights across the K axis in pairs.
#
#   K3_DIR=<snapshot>  K3_MXLAYER=1  K3_MXEXPERT=0
#   python3 k3_mxfp4_nibble_oracle.py <out.bin>

import json
import os
import struct
import sys
import mmap

import numpy as np

MAGIC = 0x4D584E31  # "MXN1"

SNAP = os.environ.get(
    "K3_DIR",
    "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
)
LAYER = int(os.environ.get("K3_MXLAYER", "1"))
EXPERT = int(os.environ.get("K3_MXEXPERT", "0"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "mxnib_fixture.bin"

_MM = {}


def _mm(path):
    if path not in _MM:
        f = open(path, "rb")
        _MM[path] = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    return _MM[path]


with open(os.path.join(SNAP, "model.safetensors.index.json")) as f:
    WM = json.load(f)["weight_map"]
_H = {}


def load_u8(name):
    shard = WM[name]
    p = os.path.join(SNAP, shard)
    if shard not in _H:
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _H[shard] = (json.loads(fh.read(n)), 8 + n)
    hdr, base = _H[shard]
    m = hdr[name]
    assert m["dtype"] == "U8", m["dtype"]
    lo, hi = m["data_offsets"]
    a = np.frombuffer(_mm(p), dtype=np.uint8, count=hi - lo, offset=base + lo)
    return a.reshape(*m["shape"])


PFX = f"language_model.model.layers.{LAYER}.block_sparse_moe.experts.{EXPERT}.w1."
packed = load_u8(PFX + "weight_packed")   # [N, K/2]
scale = load_u8(PFX + "weight_scale")     # [N, K/32] E8M0
N, KH = packed.shape
K = KH * 2
assert scale.shape == (N, K // 32), (scale.shape, N, K // 32)
print(f"K3 mxfp4 nibble probe: layer {LAYER} expert {EXPERT} w1  packed{list(packed.shape)} "
      f"scale{list(scale.shape)} -> N={N} K={K}")

# E8M0, BIAS 127. the design notes §2: "E8M0 scales are biased by 127. Neutral = 127, not
# 0. Byte 0 means 2^-127 and flushes the block to zero." Confirmed empirically from THIS
# checkpoint in the gap doc §4b: bytes span 112-122, no byte is 0x00 and none is 0xFF.
assert scale.min() > 0 and scale.max() < 255, (scale.min(), scale.max())
_smin, _smax = int(scale.min()), int(scale.max())
print(f"  E8M0 bytes in [{_smin}, {_smax}] -> 2^[{_smin - 127}, {_smax - 127}]"
      f"; zero bytes {int((scale == 0).sum())}, 0xFF {int((scale == 255).sum())}")

# OCP e2m1: 1 sign, 2 exponent, 1 mantissa. Eight magnitudes, no inf, no NaN.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1 = np.concatenate([E2M1, -E2M1])      # index by the raw 4 bits

lo_nib = (packed & 0x0F).astype(np.uint8)
hi_nib = (packed >> 4).astype(np.uint8)

# The two candidate readings, interleaved back onto the K axis.
w_lo = np.empty((N, K), dtype=np.float32)   # element 2i = LOW nibble  (plow / compressed-tensors)
w_lo[:, 0::2] = E2M1[lo_nib]
w_lo[:, 1::2] = E2M1[hi_nib]
w_hi = np.empty((N, K), dtype=np.float32)   # element 2i = HIGH nibble (the swap)
w_hi[:, 0::2] = E2M1[hi_nib]
w_hi[:, 1::2] = E2M1[lo_nib]

sc = np.ldexp(np.ones((N, K // 32), dtype=np.float32), scale.astype(np.int32) - 127)
sc = np.repeat(sc, 32, axis=1)
w_lo *= sc
w_hi *= sc

# The multisets are identical per 32-block, which is exactly why bytes alone cannot settle this.
assert np.allclose(np.sort(w_lo, axis=1), np.sort(w_hi, axis=1)), \
    "the two readings differ as multisets — then the histograms in the gap doc would have settled it"
d = np.abs(w_lo - w_hi)
print(f"  the two readings differ on {100.0 * (d > 0).mean():.1f}% of elements, "
      f"max |diff| {d.max():.4f} — same multiset per block, different POSITIONS")

rng = np.random.default_rng(0xB4)
x = rng.standard_normal(K).astype(np.float32)
# The device consumes bf16. Round here or the residual measures the rounding, not the layout.
x = (x.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)

y_lo = w_lo @ x
y_hi = w_hi @ x
rel = np.linalg.norm(y_lo - y_hi) / np.linalg.norm(y_lo)
print(f"  |y_lo| {np.linalg.norm(y_lo):.3f}  |y_hi| {np.linalg.norm(y_hi):.3f}  "
      f"rel difference {rel:.3e}")
assert rel > 0.1, "the two orders give the same GEMV — this probe cannot decide anything"


def w_bf(f, a):
    u = (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    f.write(np.ascontiguousarray(u).tobytes())


with open(OUT, "wb") as f:
    f.write(struct.pack("<4i", MAGIC, N, K, 0))
    w_bf(f, x)
    f.write(np.ascontiguousarray(packed).tobytes())
    f.write(np.ascontiguousarray(scale).tobytes())
    f.write(np.ascontiguousarray(y_lo.astype(np.float32)).tobytes())
    f.write(np.ascontiguousarray(y_hi.astype(np.float32)).tobytes())
    sz = f.tell()
print(f"wrote {OUT}  ({sz / 1e6:.1f} MB)")
