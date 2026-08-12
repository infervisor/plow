#!/usr/bin/env python3
"""Pair-exponent joint-coding audit (P10 follow-up: PolarQuant/TurboQuant).

PolarQuant (arXiv:2502.00527) quantizes RoPE'd keys as 2D (radius, angle)
sub-vectors — lossy, but its structural claim has a lossless shadow: a RoPE
pair (x, y) sits on a circle of radius r, so max(|x|,|y|) >= r/sqrt(2) and the
two EXPONENTS are jointly constrained. If H(e_x, e_y) << H(e_x) + H(e_y), a
joint pair code (k-bit code -> LUT -> exponent pair) beats sz12's independent
4-bit codes losslessly.

plow pairing: half-split (i, i+hd/2) — op_norm.cuh d_headnorm_rope.

Measures per (class, K|V): marginal H, joint H over RoPE pairs, joint H over
adjacent dims (control — V has no RoPE, K adjacent-pair separates RoPE
structure from generic correlation), and escape rate of top-{64,128,256}
most-frequent-pair LUT codes (6/7/8-bit pair codes = 11/11.5/12 b per elem
with the 8-bit lo plane).

Usage: kv_pair_audit.py <dump-dir> [--out PREFIX] [--sample N-per-class]
"""
import argparse, json, os, time
import torch

torch.set_num_threads(max(8, os.cpu_count() // 4))


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
    ap.add_argument("--out", default="perf-data/kv0-pair-audit")
    ap.add_argument("--sample", type=int, default=16)
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

    counts = {}
    acc = {}  # (class, kv) -> dict of hists
    t0 = time.time()
    for name, nbytes in sorted(man.items()):
        if nbytes not in sizes:
            continue
        ltype, kvh, ring, hd = sizes[nbytes]
        kv = name.rsplit(".", 1)[1]
        key = (ltype, kv)
        if counts.get(key, 0) >= args.sample:
            continue
        counts[key] = counts.get(key, 0) + 1
        rows = min(ctx, ring)
        x = torch.from_file(os.path.join(args.dump, name + ".raw"),
                            dtype=torch.uint16, size=kvh * ring * hd)
        x = x.reshape(kvh, ring, hd)[:, :rows, :]
        exp = ((x.to(torch.int32) >> 7) & 0xFF).to(torch.int64)

        a = acc.setdefault(key, {
            "marg": torch.zeros(256, dtype=torch.int64),
            "rope": torch.zeros(65536, dtype=torch.int64),
            "adj": torch.zeros(65536, dtype=torch.int64),
        })
        a["marg"] += torch.bincount(exp.reshape(-1), minlength=256)
        h2 = hd // 2
        jr = (exp[..., :h2] * 256 + exp[..., h2:]).reshape(-1)
        a["rope"] += torch.bincount(jr, minlength=65536)
        ja = (exp[..., 0::2] * 256 + exp[..., 1::2]).reshape(-1)
        a["adj"] += torch.bincount(ja, minlength=65536)
        print(f"  {name} ({ltype}.{kv}) t={time.time()-t0:.0f}s", flush=True)

    res = {"ctx": ctx, "dump": args.dump, "sample_per_class": args.sample,
           "classes": {}}
    for (ltype, kv), a in sorted(acc.items()):
        hm = entropy(a["marg"])
        cls = {"H_marginal_per_elem": hm}
        for pk in ("rope", "adj"):
            h = a[pk]
            n = int(h.sum())
            hj = entropy(h)
            top = torch.sort(h, descending=True).values
            cov = {f"top{k}_escape_pct":
                   float(100.0 * (1 - top[:k].sum().item() / n))
                   for k in (64, 128, 256)}
            cls[pk] = {"H_joint_per_pair": hj,
                       "saving_bits_per_elem": hm - hj / 2, **cov}
        res["classes"][f"{ltype}.{kv.upper()}"] = cls

    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
