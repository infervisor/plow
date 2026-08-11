#!/usr/bin/env python3
"""XOR-delta + per-block bit-packing audit on the KV-0 dumps (P10 follow-up).

Tests the Gorilla/FastLanes-style alternative to the sz12 plane split: treat
the head-major slab as a u16 stream, XOR each element with a neighbor, pack
each B-element block at the block's max significant bit width (+4-bit width
header). Fixed row stride would then require provisioning rows at a fixed
width anyway, but first: does the XOR even shrink the stream?

Variants per tensor:
  raw       : no delta (baseline for the packer itself)
  xor-mem   : XOR with previous element in memory order (adjacent dims)
  xor-seq   : XOR with same dim of previous ring row (sequence axis)
each also in a 'rot' form (sign bit rotated to bit 0 so a sign flip alone
does not force width 16). Reports packed bits/elem at B in {128,256,512},
plus Shannon entropy of the xor-seq hi/lo bytes (the ideal-coder bound).

Usage: kv_xor_audit.py <dump-dir> [--out PREFIX] [--sample N-per-class]
"""
import argparse, json, os, time
import torch

torch.set_num_threads(max(8, os.cpu_count() // 4))
BLOCKS = (128, 256, 512)


def entropy(hist):
    n = int(hist.sum())
    if n == 0:
        return 0.0
    p = hist.to(torch.float64) / n
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def bitlen(x_i32):
    bl = torch.zeros_like(x_i32)
    for k in range(16):
        bl = torch.where(x_i32 >= (1 << k), k + 1, bl)
    return bl


def packed_bits(stream_i32, B):
    n = (stream_i32.numel() // B) * B
    m = stream_i32[:n].reshape(-1, B).amax(dim=1)
    return float(bitlen(m).to(torch.float64).mean()) + 4.0 / B


def rot(x_i32):
    return ((x_i32 << 1) | (x_i32 >> 15)) & 0xFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default="perf-data/kv0-xor-audit")
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
    acc = {}  # class -> variant -> [sum_bits per B] + entropy hists
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
        x = x.reshape(kvh, ring, hd)[:, :rows, :].contiguous().to(torch.int32)

        flat = x.reshape(-1)
        streams = {
            "raw": flat,
            "xor-mem": flat[1:] ^ flat[:-1],
            "xor-seq": (x[:, 1:, :] ^ x[:, :-1, :]).reshape(-1),
        }
        a = acc.setdefault(ltype, {})
        for vname, s in streams.items():
            for form, ss in (("", s), ("+rot", rot(s))):
                key = vname + form
                d = a.setdefault(key, {B: [0.0, 0] for B in BLOCKS})
                for B in BLOCKS:
                    d[B][0] += packed_bits(ss, B)
                    d[B][1] += 1
        # ideal-coder bound on the seq-xor stream
        h = a.setdefault("_hists", [torch.zeros(256, dtype=torch.int64),
                                    torch.zeros(256, dtype=torch.int64)])
        sq = streams["xor-seq"]
        h[0] += torch.bincount((sq >> 8).to(torch.int64), minlength=256)
        h[1] += torch.bincount((sq & 0xFF).to(torch.int64), minlength=256)
        print(f"  {name} ({ltype}) t={time.time()-t0:.0f}s", flush=True)

    res = {"ctx": ctx, "dump": args.dump, "sample_per_class": args.sample,
           "classes": {}}
    for ltype, a in acc.items():
        cls = {}
        for key, d in a.items():
            if key == "_hists":
                cls["xor_seq_H_hi"] = entropy(a["_hists"][0])
                cls["xor_seq_H_lo"] = entropy(a["_hists"][1])
                continue
            cls[key] = {f"B{B}": {"bits_per_elem": round(v[0] / v[1], 3),
                                  "ratio": round(16 / (v[0] / v[1]), 4)}
                        for B, v in d.items()}
        res["classes"][ltype] = cls

    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
