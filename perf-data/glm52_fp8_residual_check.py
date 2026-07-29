#!/usr/bin/env python3
"""(a) numeric gate, offline: is the block-fp8 form the kernel will now read the SAME weight the
bf16 the prep wrote was standing in for?

The prep computes w_bf16 = round_bf16(fp8 * scale). GEMV_FP8_BLK computes fp8 * scale in f32. So
the two differ by exactly one bf16 rounding of the product and nothing else — no requantisation is
involved anywhere. This measures that difference on real tensors."""
import json, os, struct, sys, glob
import numpy as np

SRC = glob.glob('/home/lava/.cache/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/*/')[0]
PREP = '/home/lava/models/GLM-5.2-plow'

def index(d):
    idx = {}
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith('model-') and fn.endswith('.safetensors')):
            continue
        p = os.path.join(d, fn)
        with open(p, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k == '__metadata__' or k in idx:
                continue
            idx[k] = (p, 8 + n + v['data_offsets'][0], 8 + n + v['data_offsets'][1],
                      v['dtype'], v['shape'])
    return idx

def rd(idx, name):
    p, a, b, dt, shape = idx[name]
    with open(p, 'rb') as f:
        f.seek(a)
        raw = f.read(b - a)
    if dt == 'F8_E4M3':
        return np.frombuffer(raw, np.uint8).reshape(shape), dt
    if dt == 'F32':
        return np.frombuffer(raw, np.float32).reshape(shape), dt
    if dt == 'BF16':
        u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape), dt
    raise ValueError(dt)

# e4m3fn decode table (1-4-3, bias 7, no inf, NaN = 0x7F/0xFF)
_T = np.zeros(256, np.float32)
for byte in range(256):
    s = -1.0 if byte & 0x80 else 1.0
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    if e == 0:
        v = m / 8.0 * 2.0 ** -6
    elif e == 0xF and m == 0x7:
        v = np.nan
    else:
        v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
    _T[byte] = s * v

def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    # round-to-nearest-even, the same rule torch .to(bfloat16) uses
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) & 0xFFFF0000).view(np.float32)

si, pi = index(SRC), index(PREP)
print(f"{'tensor':52s} {'max|fp8*s - bf16|':>18s} {'rel(max|w|)':>12s} {'== bf16 round?':>15s}")
for name in ['model.layers.3.self_attn.o_proj.weight',
             'model.layers.3.mlp.shared_experts.gate_proj.weight',
             'model.layers.3.mlp.shared_experts.up_proj.weight',
             'model.layers.3.mlp.shared_experts.down_proj.weight',
             'model.layers.40.self_attn.o_proj.weight',
             'model.layers.77.mlp.shared_experts.down_proj.weight']:
    w8, _ = rd(si, name)
    sc, _ = rd(si, name + '_scale_inv')
    ref, _ = rd(pi, name)
    N, K = w8.shape
    # exact kernel arithmetic: f32 product of the fp8 value and its [128,128] block scale
    got = _T[w8] * sc[np.arange(N) // 128][:, np.arange(K) // 128]
    d = np.abs(got - ref)
    scale = np.abs(ref).max()
    exact = bool(np.array_equal(bf16(got), ref))
    print(f"{name:52s} {d.max():18.3e} {d.max()/scale:12.3e} {str(exact):>15s}")
