#!/usr/bin/env python3
"""SplitZip exponent-histogram feasibility study on real weights.

bf16: exponent = bits 7..14 of the uint16 (8 bits, 256 values)
fp8 e4m3: exponent = bits 3..6 of the uint8 (4 bits, 16 values)

Format arithmetic (paper's shape, chunk = 1024 elements):
  bits/elem = CODE_BITS + PAYLOAD_BITS + esc_rate * (POS_BITS + EXP_BITS)
where PAYLOAD = sign+mantissa (bf16: 8, e4m3: 4), POS_BITS = 10 (1024-elem chunk),
EXP_BITS = raw exponent width (bf16: 8, e4m3: 4).
"""
import json, os, sys, glob
import numpy as np
from safetensors import safe_open

BN, BK = 128, 64
TILE = BN * BK
CHUNK = 1024
POS_BITS = 10


def tensor_iter(model_dir, dtype):
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    for f in files:
        import torch
        with safe_open(f, framework="pt") as sf:
            for name in sorted(sf.keys()):
                t = sf.get_slice(name)
                if len(t.get_shape()) != 2:
                    continue
                if "embed" in name or "lm_head" in name or "scale" in name:
                    continue
                tt = sf.get_tensor(name)
                nbits = tt.element_size() * 8
                u = tt.view(torch.int16 if nbits == 16 else torch.int8).numpy()
                yield name, u


def exponents(t, dtype):
    if dtype == "bf16":
        u = t.view(np.uint16)
        return ((u >> 7) & 0xFF).astype(np.uint8), 256, 8, 8
    else:  # e4m3
        u = t.view(np.uint8)
        return ((u >> 3) & 0x0F).astype(np.uint8), 16, 4, 4


def ratio_from_esc(esc, code_bits, payload_bits, exp_bits):
    return 16.0 / (code_bits + payload_bits + esc * (POS_BITS + exp_bits)) \
        if payload_bits == 8 else \
        8.0 / (code_bits + payload_bits + esc * (POS_BITS + exp_bits))


def analyze(model_dir, dtype, label, tile_limit_tensors=None):
    print(f"\n{'='*78}\n{label}  ({model_dir})\n{'='*78}")
    nbins_total = 256 if dtype == "bf16" else 16
    payload_bits = 8 if dtype == "bf16" else 4
    exp_bits = 8 if dtype == "bf16" else 4
    raw_bits = 16.0 if dtype == "bf16" else 8.0

    global_hist = np.zeros(nbins_total, dtype=np.int64)
    per_tensor = []
    tile_ratios_all = []

    for name, t in tensor_iter(model_dir, dtype):
        e, nb, pb, eb = exponents(t, dtype)
        flat = e.ravel()
        h = np.bincount(flat, minlength=nbins_total).astype(np.int64)
        global_hist += h
        n = flat.size
        top16 = np.sort(h)[::-1][:16].sum()
        top8 = np.sort(h)[::-1][:8].sum()
        top32 = np.sort(h)[::-1][:32].sum()
        per_tensor.append((name, n, 1 - top16 / n, 1 - top8 / n, 1 - top32 / n,
                           int((h > 0).sum())))

        # ---- per-tile (BN,BK) analysis, tile-local code table ----
        N, K = e.shape
        if N % BN == 0 and K % BK == 0:
            tv = e.reshape(N // BN, BN, K // BK, BK).transpose(0, 2, 1, 3)
            tv = np.ascontiguousarray(tv).reshape(-1, TILE)
            ntile = tv.shape[0]
            # histogram per tile via offset bincount, in batches
            B = 4096
            for s in range(0, ntile, B):
                blk = tv[s:s + B]
                nb_ = blk.shape[0]
                off = (np.arange(nb_, dtype=np.int64)[:, None] * nbins_total
                       + blk.astype(np.int64))
                hh = np.bincount(off.ravel(),
                                 minlength=nb_ * nbins_total).reshape(nb_, nbins_total)
                hs = np.sort(hh, axis=1)[:, ::-1]
                cols = []
                for kk in (4, 8, 16, 32):
                    if kk <= nbins_total:
                        cols.append(1.0 - hs[:, :kk].sum(1) / TILE)
                tile_ratios_all.append(np.stack(cols, 1).astype(np.float32))
        del e, t

    # ---------- global ----------
    n = global_hist.sum()
    order = np.argsort(global_hist)[::-1]
    print(f"\ntotal 2-D GEMM elements: {n:,}   distinct exponents used: {(global_hist>0).sum()}")
    print("\ntop-20 exponent values (global):")
    print(f"{'rank':>4} {'expfield':>8} {'2^e':>10} {'count':>15} {'frac':>9} {'cum':>9}")
    cum = 0.0
    for i in range(min(20, (global_hist > 0).sum())):
        b = order[i]; c = global_hist[b]; f = c / n; cum += f
        print(f"{i:>4} {b:>8} {2.0**(int(b)-127) if dtype=='bf16' else 2.0**(int(b)-7):>10.3e} {c:>15,} {f:>9.5f} {cum:>9.5f}")

    res = {}
    for k, cb in ((8, 3), (16, 4), (32, 5)):
        if k > nbins_total:
            continue
        cov = global_hist[order[:k]].sum() / n
        esc = 1 - cov
        bpe = cb + payload_bits + esc * (POS_BITS + exp_bits)
        res[k] = (cov, esc, bpe, raw_bits / bpe)
        print(f"\ntop-{k:<3} in {cb} bits: coverage={cov:.6f}  escape={esc:.3e}  "
              f"bits/elem={bpe:.4f}  ratio={raw_bits/bpe:.4f}x")

    print(f"\nbreak-even escape rate for top-16/4-bit: "
          f"{(raw_bits - 4 - payload_bits)/(POS_BITS+exp_bits):.4f}")

    # ---------- per tensor ----------
    print(f"\nper-tensor escape rates (top-16), worst 12 of {len(per_tensor)}:")
    per_tensor.sort(key=lambda r: -r[2])
    print(f"{'tensor':<52}{'esc16':>10}{'esc8':>10}{'esc32':>10}{'#exp':>6}")
    for r in per_tensor[:12]:
        print(f"{r[0]:<52}{r[2]:>10.3e}{r[3]:>10.3e}{r[4]:>10.3e}{r[5]:>6}")
    print("best 5:")
    for r in per_tensor[-5:]:
        print(f"{r[0]:<52}{r[2]:>10.3e}{r[3]:>10.3e}{r[4]:>10.3e}{r[5]:>6}")

    # by projection type
    print("\nby projection type (element-weighted escape rate, top-16):")
    kinds = {}
    for name, cnt, e16, e8, e32, nx in per_tensor:
        k = name.split(".")[-2] if "." in name else name
        a = kinds.setdefault(k, [0, 0.0, 0.0, 0.0])
        a[0] += cnt; a[1] += e16 * cnt; a[2] += e8 * cnt; a[3] += e32 * cnt
    print(f"{'proj':<14}{'elems':>16}{'esc16':>12}{'esc8':>12}{'esc32':>12}{'ratio16':>10}")
    for k, a in sorted(kinds.items(), key=lambda x: -x[1][1] / max(x[1][0], 1)):
        e16 = a[1] / a[0]
        bpe = 4 + payload_bits + e16 * (POS_BITS + exp_bits)
        print(f"{k:<14}{a[0]:>16,}{e16:>12.3e}{a[2]/a[0]:>12.3e}{a[3]/a[0]:>12.3e}{raw_bits/bpe:>10.4f}")

    # by layer
    print("\nby layer index (element-weighted esc16), first/last 6:")
    lay = {}
    for name, cnt, e16, e8, e32, nx in per_tensor:
        p = name.split(".")
        li = None
        for i, tok in enumerate(p):
            if tok == "layers" and i + 1 < len(p):
                li = int(p[i + 1])
        if li is None:
            continue
        a = lay.setdefault(li, [0, 0.0])
        a[0] += cnt; a[1] += e16 * cnt
    ks = sorted(lay)
    for li in ks[:6] + (["..."] if len(ks) > 12 else []) + ks[-6:]:
        if li == "...":
            print("   ...")
            continue
        a = lay[li]
        e16 = a[1] / a[0]
        print(f"  layer {li:<3} esc16={e16:.4e}  ratio={raw_bits/(4+payload_bits+e16*(POS_BITS+exp_bits)):.4f}x")

    # ---------- per-tile distribution ----------
    if tile_ratios_all:
        allesc = np.concatenate(tile_ratios_all, 0)
        ks = [k for k in (4, 8, 16, 32) if k <= nbins_total]
        # per-tile code table must be shipped: k entries x exp_bits, + 16b esc count
        print(f"\nper-tile (BN,BK)=({BN},{BK}) = {TILE} elems, TILE-LOCAL code table")
        print(f"  tiles: {allesc.shape[0]:,}")
        qs = [50, 90, 99, 99.9, 100]
        for j, k in enumerate(ks):
            cb = {4: 2, 8: 3, 16: 4, 32: 5}[k]
            esc = allesc[:, j].astype(np.float64)
            meta = (k * exp_bits + 16) / TILE   # table + escape count, bits/elem
            bpe = cb + payload_bits + esc * (POS_BITS + exp_bits) + meta
            rat = raw_bits / bpe
            be = (raw_bits - cb - payload_bits - meta) / (POS_BITS + exp_bits)
            loss = int((esc > be).sum())
            print(f"\n  --- top-{k} in {cb} bits (meta {meta:.4f} b/elem) ---")
            print("   esc pct: " + " ".join(f"p{q}={np.percentile(esc,q):.3e}" for q in qs))
            print("   ratio  : " + " ".join(f"p{100-q if q<100 else 0}={np.percentile(rat,100-q):.4f}" for q in qs))
            print(f"   AGGREGATE ratio (raw/mean_bits) = {raw_bits/bpe.mean():.4f}x"
                  f"   worst tile = {rat.min():.4f}x")
            print(f"   mean escapes per tile = {esc.mean()*TILE:.2f}"
                  f"   p99.9 escapes/tile = {np.percentile(esc,99.9)*TILE:.1f}")
            print(f"   break-even esc = {be:.4f}; tiles that LOSE: {loss} "
                  f"({100.0*loss/esc.size:.5f}%)")
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("bf16", "both"):
        analyze("/root/models/Qwen3-4B", "bf16", "Qwen3-4B  bf16")
    if which in ("fp8", "both"):
        analyze("/workspace/models/gemma-4-31B-it-fp8", "e4m3", "gemma-4-31B-it  fp8 e4m3")
