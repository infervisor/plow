#!/usr/bin/env python3
"""Last fp8 lossless candidate: sz12-style pair-coded exponents (P10 KV-0).

fp8-e4m3 = s1 e4 m3. The 3-bit window (sz7) was killed (2.3-2.8% escapes).
Remaining fixed-length shape: keep sign+mant raw (4 b/elem), code adjacent
exponent PAIRS with a 7-bit top-128 LUT -> 15 b/pair = 7.5 b/elem = 1.067x
layout ceiling, minus escape slots. This measures the pair escape rate and
joint entropy on the kernel-exact fp8 twin (per-row amax/448 RNE) of the
bf16 dumps. Gate context: fp8 track needs >=1.12.

Usage: fp8_pair_check.py <dump-dir> [--sample N-per-class]
"""
import argparse, json, os, time
import torch

torch.set_num_threads(max(8, os.cpu_count() // 4))
E4M3_MAX = 448.0


def entropy(hist):
    n = int(hist.sum())
    if n == 0:
        return 0.0
    p = hist.to(torch.float64) / n
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--sample", type=int, default=6)
    ap.add_argument("--full-geom", default="1,32768,512")
    ap.add_argument("--slide-geom", default="8,16384,256")
    args = ap.parse_args()

    fk, fr, fh = map(int, args.full_geom.split(","))
    sk, sr, sh = map(int, args.slide_geom.split(","))
    sizes = {fk * fr * fh * 2: ("full", fk, fr, fh),
             sk * sr * sh * 2: ("sliding", sk, sr, sh)}

    man, ctx = {}, None
    for line in open(os.path.join(args.dump, "manifest.txt")):
        p = line.split()
        if p[0] == "ctx":
            ctx = int(p[1])
        else:
            man[p[0]] = int(p[1])

    counts, acc = {}, {}
    t0 = time.time()
    for name, nbytes in sorted(man.items()):
        if nbytes not in sizes:
            continue
        ltype, kvh, ring, hd = sizes[nbytes]
        if counts.get(ltype, 0) >= args.sample:
            continue
        counts[ltype] = counts.get(ltype, 0) + 1
        rows = min(ctx, ring)
        x = torch.from_file(os.path.join(args.dump, name + ".raw"),
                            dtype=torch.uint16, size=kvh * ring * hd)
        v = x.reshape(kvh, ring, hd)[:, :rows, :].view(torch.bfloat16).to(torch.float32)
        amax = v.abs().amax(dim=2, keepdim=True)
        scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
        b8 = (v / scale).to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int64)
        e = (b8 >> 3) & 0xF
        a = acc.setdefault(ltype, {"pair": torch.zeros(256, dtype=torch.int64),
                                   "marg": torch.zeros(16, dtype=torch.int64)})
        a["marg"] += torch.bincount(e.reshape(-1), minlength=16)
        pair = (e[..., 0::2] * 16 + e[..., 1::2]).reshape(-1)
        a["pair"] += torch.bincount(pair, minlength=256)
        print(f"  {name} ({ltype}) t={time.time()-t0:.0f}s", flush=True)

    out = {}
    for ltype, a in acc.items():
        n = int(a["pair"].sum())
        top = torch.sort(a["pair"], descending=True).values
        hm, hj = entropy(a["marg"]), entropy(a["pair"])
        out[ltype] = {
            "H_exp_marg": hm, "H_pair": hj,
            "pair_saving_bits_per_elem": hm - hj / 2,
            "top128_escape_pct": float(100.0 * (1 - top[:128].sum().item() / n)),
            "top64_escape_pct": float(100.0 * (1 - top[:64].sum().item() / n)),
            "layout_ratio_7b_pair": 16.0 / 15.0,
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
