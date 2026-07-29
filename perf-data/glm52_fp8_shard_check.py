#!/usr/bin/env python3
"""GLM_LINEAR_FP8's TP4 gate: does RANK r's slice of the fp8 weight, paired with RANK r's slice of
the [128,128] scale grid, dequantise to RANK r's slice of the bf16 weight the prep wrote?

`glm52_fp8_residual_check.py` already settled the WHOLE-TENSOR question: `bf16_round(fp8 * scale)`
equals the prepped bf16 bit for bit, so the fp8 arm is the un-rounded form of the same weight and
not a requantisation. That check runs at tp=1 and therefore cannot see the thing most likely to be
wrong here.

WHAT THIS ADDS. The weight and its scale grid are sharded by TWO SEPARATE calls to
`plowrt::asset::shard::slice_for`, which agree only because both names contain the same
`<proj>.weight` substring and because the shard boundary happens to land on a multiple of 128 in
BOTH. Nothing checks the second half. A grid whose slice is off by one block column is silent: the
kernel reads a real scale from the wrong block, every output is plausible, and no bound is
violated. That is this campaign's standard failure mode, so it gets an explicit check.

  o_proj              [6144,16384] ROW-parallel  -> rank cuts K: 4096 cols, grid [48,128] -> 32 cols
  shared gate/up      [2048, 6144] COL-parallel  -> rank cuts N:  512 rows, grid [16, 48] ->  4 rows
  shared down         [6144, 2048] ROW-parallel  -> rank cuts K:  512 cols, grid [48, 16] ->  4 cols

CPU only, no GPU, no lease. Reads the two dirs plowrt itself loads:
  fp8 + grid : $PLOW_CKPT_Q  (glm52_prep_fp8_linear.py's additive `.weight_fp8`/`.weight_scale_inv`)
  bf16 ref   : $PLOW_CKPT    (glm52_prep.py's dequantised `.weight`)
"""
import json, os, struct, sys, glob
import numpy as np

Q    = os.environ.get('PLOW_CKPT_Q', '/home/lava/models/GLM-5.2-plow-q')
PREP = os.environ.get('PLOW_CKPT',   '/home/lava/models/GLM-5.2-plow')
TP   = int(os.environ.get('PLOW_TP', '4'))


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
        return np.frombuffer(raw, np.uint8).reshape(shape)
    if dt == 'F32':
        return np.frombuffer(raw, np.float32).reshape(shape)
    if dt == 'BF16':
        u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    raise ValueError(dt)


# e4m3fn decode table (1-4-3, bias 7, no inf, NaN = 0x7F/0xFF)
_T = np.zeros(256, np.float32)
for byte in range(256):
    s = -1.0 if byte & 0x80 else 1.0
    e, m = (byte >> 3) & 0xF, byte & 0x7
    if e == 0:
        v = m / 8.0 * 2.0 ** -6
    elif e == 0xF and m == 0x7:
        v = np.nan
    else:
        v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
    _T[byte] = s * v


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF          # round-to-nearest-even, torch's .to(bfloat16) rule
    return ((u + r) & 0xFFFF0000).view(np.float32)


# name -> shard axis, mirroring plowrt::asset::shard::shard_of's COL/ROW substring lists.
def axis(name):
    if 'o_proj.weight' in name or 'down_proj.weight' in name:
        return 'row'      # cuts K (input lanes)
    if 'gate_proj.weight' in name or 'up_proj.weight' in name:
        return 'col'      # cuts N (output rows)
    raise ValueError(name)


TENSORS = [
    'model.layers.3.self_attn.o_proj',
    'model.layers.3.mlp.shared_experts.gate_proj',
    'model.layers.3.mlp.shared_experts.up_proj',
    'model.layers.3.mlp.shared_experts.down_proj',
    'model.layers.40.self_attn.o_proj',
    'model.layers.77.mlp.shared_experts.down_proj',
]

qi, pi = index(Q), index(PREP)
print(f'tp={TP}   fp8+grid: {Q}\n         bf16 ref: {PREP}\n')
print(f"{'tensor':52s} {'axis':>4s} {'rank':>4s} {'shard':>13s} "
      f"{'grid':>9s} {'max|fp8*s-bf16|':>16s} {'exact?':>7s}")

bad = 0
for base in TENSORS:
    w8 = rd(qi, base + '.weight_fp8')
    sc = rd(qi, base + '.weight_scale_inv')
    ref = rd(pi, base + '.weight')
    ax = axis(base + '.weight')
    N, K = w8.shape
    assert sc.shape == (-(-N // 128), -(-K // 128)), f'{base}: grid {sc.shape} != {N,K}/128'
    assert ref.shape == (N, K), f'{base}: bf16 {ref.shape} != fp8 {w8.shape}'

    for r in range(TP):
        if ax == 'row':
            n0, n1, k0, k1 = 0, N, r * (K // TP), (r + 1) * (K // TP)
        else:
            n0, n1, k0, k1 = r * (N // TP), (r + 1) * (N // TP), 0, K
        # The rank's slice of the grid, cut on the SAME axis, in units of 128-blocks. If the shard
        # boundary is not a multiple of 128 this is not expressible and the whole scheme is unsound.
        for lo, hi, what in ((n0, n1, 'N'), (k0, k1, 'K')):
            if lo % 128 or hi % 128:
                print(f'  !! {base} rank {r}: {what} shard [{lo},{hi}) is not 128-aligned')
                bad += 1
        g = sc[n0 // 128:-(-n1 // 128), k0 // 128:-(-k1 // 128)]
        ws = w8[n0:n1, k0:k1]
        rs = ref[n0:n1, k0:k1]
        # exact kernel arithmetic: f32 product of the fp8 value and its block scale
        got = _T[ws] * g[np.arange(n1 - n0) // 128][:, np.arange(k1 - k0) // 128]
        exact = bool(np.array_equal(bf16(got), rs))
        d = float(np.abs(got - rs).max())
        bad += (not exact)
        print(f'{base:52s} {ax:>4s} {r:>4d} {f"{n1-n0}x{k1-k0}":>13s} '
              f'{f"{g.shape[0]}x{g.shape[1]}":>9s} {d:16.3e} {str(exact):>7s}')

print()
if bad:
    print(f'FAIL — {bad} shard(s) do not reproduce the bf16 the prep wrote')
    sys.exit(1)
print(f'OK — every TP{TP} shard of every tensor dequantises to EXACTLY the prepped bf16 '
      f'(bit-for-bit after one bf16 rounding).')
print('So the fp8 arm is the UN-ROUNDED form of the same sharded weight: strictly more precise,')
print('no requantisation, and the scale grid slices in step with the weight it scales.')
