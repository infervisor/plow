#!/usr/bin/env python3
"""Opaque-byte compressibility of KV streams (P10 follow-up).

No float semantics: derive the kernel-exact fp8-e4m3 and packed-nvfp4 twins
(plus raw bf16 for reference) from dump tensors, write raw byte streams, and
run general-purpose compressors (zstd -3/-19, gzip -6, xz -6) on them. This
bounds what ANY byte-oriented codec (entropy + LZ matching) finds — relevant
to the KV-4 transfer track where variable-length output is acceptable;
resident-KV reads remain fixed-stride-only regardless.

Usage: kv_bytes_compress.py <dump-dir> [--tmp /dev/shm/kv0/bytes] [--sample N]
"""
import argparse, json, os, subprocess, time
import torch

torch.set_num_threads(max(8, os.cpu_count() // 4))
E4M3_MAX = 448.0
FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def twins(x_u16, hd):
    v = x_u16.view(torch.bfloat16).to(torch.float32)
    amax = v.abs().amax(dim=2, keepdim=True)
    scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
    f8 = (v / scale).to(torch.float8_e4m3fn).view(torch.uint8)

    kvh, rows, _ = v.shape
    vb = v.reshape(kvh, rows, hd // 16, 16)
    bmax = vb.abs().amax(dim=3, keepdim=True)
    bscale = torch.where(bmax > 0, bmax / 6.0, torch.ones_like(bmax))
    vq = vb / bscale
    idx = (vq.abs().unsqueeze(-1) - FP4_VALS).abs().argmin(dim=-1)
    code = (idx + torch.where(vq < 0, 8, 0)).to(torch.uint8).reshape(kvh, rows, hd)
    f4 = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()  # packed nibbles
    return f8.contiguous(), f4


def run(cmd, path):
    t0 = time.time()
    out = subprocess.run(cmd + [path, "-c"], capture_output=True).stdout
    return len(out), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--tmp", default="/dev/shm/kv0/bytes")
    ap.add_argument("--sample", type=int, default=2)  # per class
    ap.add_argument("--full-geom", default="1,32768,512")
    ap.add_argument("--slide-geom", default="8,16384,256")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)

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

    codecs = [("zstd-3", ["zstd", "-3", "-q"]), ("zstd-19", ["zstd", "-19", "-q"]),
              ("gzip-6", ["gzip", "-6"]), ("xz-6", ["xz", "-6", "-T", "4"])]
    counts, res = {}, {}
    for name, nbytes in sorted(man.items()):
        if nbytes not in sizes:
            continue
        ltype, kvh, ring, hd = sizes[nbytes]
        kv = name.rsplit(".", 1)[1]
        key = f"{ltype}.{kv.upper()}"
        if counts.get(key, 0) >= args.sample:
            continue
        counts[key] = counts.get(key, 0) + 1
        rows = min(ctx, ring)
        x = torch.from_file(os.path.join(args.dump, name + ".raw"),
                            dtype=torch.uint16, size=kvh * ring * hd)
        x = x.reshape(kvh, ring, hd)[:, :rows, :]
        f8, f4 = twins(x, hd)
        streams = {"bf16": x.contiguous().view(torch.uint8).numpy().tobytes(),
                   "fp8": f8.numpy().tobytes(),
                   "fp4packed": f4.numpy().tobytes()}
        for sname, raw in streams.items():
            path = os.path.join(args.tmp, "probe.bin")
            with open(path, "wb") as f:
                f.write(raw)
            d = res.setdefault(key, {}).setdefault(sname, {"raw": 0})
            d["raw"] += len(raw)
            for cname, cmd in codecs:
                sz, dt = run(cmd, path)
                e = d.setdefault(cname, {"comp": 0, "sec": 0.0})
                e["comp"] += sz
                e["sec"] += dt
        print(f"  {name} ({key}) done", flush=True)

    table = {}
    for key, d in sorted(res.items()):
        table[key] = {}
        for sname, dd in d.items():
            table[key][sname] = {c: round(dd["raw"] / v["comp"], 4)
                                 for c, v in dd.items() if c != "raw"}
    print(json.dumps(table, indent=1))
    with open("perf-data/kv0-bytes-compress.json", "w") as f:
        json.dump(table, f, indent=1)
    os.remove(os.path.join(args.tmp, "probe.bin"))


if __name__ == "__main__":
    main()
