#!/usr/bin/env python3
# mx_act_even_ref.py gen|check DIR: numpy reference for mx_act_even_gfx950_test.hip.
#
# The reference is AMD Quark's dynamic MXFP4 activation quant for scale_calculation_mode "even",
# round_method "half_even" (amd-quark 0.12):
#   scale: quark/torch/quantization/utils.py even_round(): bits(amax) + (1 << 21), keep sign and
#          exponent; 0 -> 2^-126; floor(log2) - emax(fp4) = 2; clamp [-127, 127].
#   codes: hw_emulation/csrc/mxfp4/fake_kernels.cuh fp16_to_fp4_simulate(): x / scale rounded
#          half-to-even onto e2m1, saturating at 6, sign kept.
# `gen` writes blocks.f32 (adversarial 32-element blocks); `check` compares device.u8 bit-exactly
# (staging codes+scale and the decode fake-quant values on the bf16-exact blocks, bridge
# codes+scale on all) and reports how
# gfx950's v_cvt_scalef32_pk_fp4_f32 rounds at the same scale, and how often AITER's scale rules
# (RCEIL; FlyDSL's fused 1.5-round) pick another scale.
import os, sys
import numpy as np

GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def f32(bits):
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def bf16_rne(x):
    u = bits(x).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    return f32(u.astype(np.uint32))


def quark_byte(amax):
    r = f32(((bits(amax).astype(np.uint64) + (1 << 21)) & 0xFF800000).astype(np.uint32)).astype(np.float64)
    r = np.where(amax == 0, 2.0 ** -126, r)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(r)) - 2
    return (np.clip(e, -127, 127) + 127).astype(np.int64)


def rne_codes(x, byte):
    y = x.astype(np.float64) / np.exp2(byte.astype(np.float64) - 127)[..., None]
    a = np.minimum(np.abs(y), 6.0)
    d = np.abs(a[..., None] - GRID)
    lo = d.argmin(-1)
    tie = np.isclose(d, d.min(-1, keepdims=True), rtol=0, atol=0).sum(-1) > 1
    idx = np.where(tie & (lo % 2 == 1), lo + 1, lo)
    return idx | (np.signbit(x).astype(np.int64) << 3)


def rceil_byte(amax):  # AITER mx_quant_utils.h RoundUp: ceil_pow2(amax * fl(1/6))
    u = bits((amax * np.float32(1.0 / 6.0)).astype(np.float32)).astype(np.int64)
    e = (u >> 23) & 0xFF
    return np.where((e < 0xFF) & ((u & 0x7FFFFF) != 0), e + 1, e)


def r15_byte(amax):  # AITER FlyDSL fused stage-1 quant: E(bits + 2^22) - 2
    e = ((bits(amax).astype(np.int64) + 0x400000) & 0xFF800000) >> 23
    return np.maximum(e - 2, 0)


def gen(d):
    rng = np.random.default_rng(7)
    B = []
    ties = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    for k in range(-124, 125, 3):  # every e2m1 midpoint, both signs, +-1 ulp, at scale 2^k
        s = np.float32(2.0 ** k)
        v = np.concatenate([[4.0], ties, -ties, np.nextafter(ties.astype(np.float32), 0),
                            np.nextafter(ties.astype(np.float32), 9), [-4.0, 6.0, -0.0]])[:32]
        B.append((v * s).astype(np.float32))
    mants = [1.0, 1.25, 1.5, 1.75, 1.875]
    for k in range(-126, 127, 5):  # amax at the scale-rounding edges (f32 and bf16 ulps)
        for m in mants:
            c = np.float32(m * 2.0 ** k)
            for a in (c, np.nextafter(c, np.float32(0)), np.nextafter(c, np.float32(np.inf)),
                      f32(bits(c) - 0x10000), f32(bits(c) + 0x10000)):
                v = rng.uniform(-1, 1, 32).astype(np.float32) * a
                v[rng.integers(32)] = a if rng.integers(2) else -a
                B.append(v.astype(np.float32))
    B.append(np.zeros(32, np.float32))
    B.append(-np.zeros(32, np.float32))
    B.append(np.where(rng.integers(2, size=32) == 1, 0.0, -0.0).astype(np.float32))
    for e in range(-149, -118):  # subnormal and bottom-binade amax
        a = np.float32(2.0 ** e)
        for m in (1.0, 1.5, 1.75, 1.9):
            v = (rng.uniform(-1, 1, 32) * m * a).astype(np.float32)
            v[0] = np.float32(m * a)
            B.append(v)
    for b in (1, 2, 3, 0x7F, 0x10000, 0x3F0000, 0x600000, 0x7F0000, 0x7FFFFF):
        v = f32(rng.integers(0, b + 1, 32).astype(np.uint32))
        B.append(np.where(rng.integers(2, size=32) == 1, v, -v).astype(np.float32))
    for k in range(-60, 61, 4):  # saturation: amax mantissa in [1.5, 1.75) -> amax/scale in [6, 7)
        s = 2.0 ** k
        for m in (1.5, 1.53125, 1.625, 1.6875, 1.71875):
            v = rng.uniform(-6, 6, 32) * s / 4
            v[:6] = np.array([m * 4, -m * 4, 6.5, -6.5, 6.25, 5.0]) * s
            B.append(v.astype(np.float32))
    for _ in range(3000):  # natural blocks over a wide range, heavy tails
        v = rng.standard_t(3, 32) * 2.0 ** rng.integers(-40, 40)
        B.append(v.astype(np.float32))
    for e in (120, 125, 126, 127):
        for m in (1.0, 1.5, 1.7):
            v = rng.uniform(-1, 1, 32) * m * 2.0 ** e
            v[3] = m * 2.0 ** e
            B.append(v.astype(np.float32))
    X = np.stack(B).astype(np.float32)
    half = np.arange(len(X)) % 2 == 0  # half the blocks bf16-exact (the staging path's input)
    X[half] = bf16_rne(X[half])
    os.makedirs(d, exist_ok=True)
    X.tofile(os.path.join(d, "blocks.f32"))
    print(f"{len(X)} blocks -> {d}/blocks.f32")


def unpack(b):
    b = b.astype(np.int64)
    return np.stack([b & 15, b >> 4], -1).reshape(b.shape[0], 32)


def check(d):
    X = np.fromfile(os.path.join(d, "blocks.f32"), dtype=np.float32).reshape(-1, 32)
    D = np.fromfile(os.path.join(d, "device.u8"), dtype=np.uint8).reshape(-1, 128)
    amax = np.abs(X).max(-1)
    sb = quark_byte(amax)
    codes = rne_codes(X, sb)
    exact = (bits(X) & 0xFFFF == 0).all(-1)
    ok = True
    for name, c0, s0, sel in (("staging (bf16)", 0, 16, exact), ("bridge (f32)", 17, 33, np.ones_like(exact))):
        cb, sc = unpack(D[:, c0 : c0 + 16]), D[:, s0].astype(np.int64)
        bad_s = (sc != sb) & sel
        bad_c = (cb != codes).any(-1) & sel
        print(f"{name:15s} blocks={sel.sum():5d} scale mismatches={bad_s.sum()} code mismatches={bad_c.sum()}")
        for i in np.nonzero(bad_s | bad_c)[0][:4]:
            print(f"   block {i}: amax={amax[i]:.6g} ref byte {sb[i]} dev {sc[i]}; "
                  f"ref {codes[i][:8].tolist()} dev {cb[i][:8].tolist()}")
        ok &= not (bad_s.any() or bad_c.any())
    deq = np.where(codes & 8, -1.0, 1.0) * GRID[codes & 7] * np.exp2(sb - 127.0)[:, None]
    fq = D[:, 64:128].copy().view(np.uint16).astype(np.uint32) << 16
    bad_f = (fq != bits(bf16_rne(deq.astype(np.float32)))).any(-1) & exact
    print(f"{'fake-quant bf16':15s} blocks={exact.sum():5d} value mismatches={bad_f.sum()}")
    ok &= not bad_f.any()
    hw = unpack(D[:, 34:50])
    diff = hw != codes
    y = np.abs(X.astype(np.float64) / np.exp2(sb - 127.0)[:, None])
    tie = np.isin(np.minimum(y, 7), [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    print(f"hw v_cvt_scalef32_pk_fp4_f32 vs half-even (same scale): {diff.sum()} of {diff.size} codes differ "
          f"({(diff & tie).sum()} at exact ties of {tie.sum()}, {(diff & ~tie).sum()} elsewhere)")
    nat = slice(len(X) - 3012, len(X) - 12)
    for nm, f in (("AITER RCEIL", rceil_byte), ("AITER FlyDSL 1.5-round", r15_byte)):
        print(f"{nm:23s} scale != Quark even on {(f(amax[nat]) != sb[nat]).mean() * 100:.1f}% of natural blocks")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(gen(sys.argv[2]) if sys.argv[1] == "gen" else check(sys.argv[2]))
