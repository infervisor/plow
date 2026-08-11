#!/usr/bin/env python3
"""Regenerate g12_sample.bin — a representative slab of real gemma-4-12B bf16 decode
weight bytes (uint16) for the sz microbench harnesses (sz_batch_sm120.cu tiles it to
fill each shape). Prints the best affine exponent base to confirm the harness's
hard-coded EXP_BASE=109 matches the real distribution.

Usage: make_sample.py <model-dir> <out.bin> [--mb N]
"""
import sys, os, json, struct, mmap
import torch

def read_header(path):
    with open(path, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hn))
    return hdr, 8 + hn

def best_base(rawbytes):
    u16 = torch.frombuffer(rawbytes, dtype=torch.uint16).to(torch.int32)
    exh = torch.bincount((u16 >> 7) & 0xFF, minlength=256)
    c = torch.cumsum(exh, 0)
    best_s, best_v = 0, -1
    for s in range(0, 241):
        v = int(c[s + 15] - (c[s - 1] if s else 0))
        if v > best_v:
            best_v, best_s = v, s
    return best_s, best_v / int(exh.sum())

def main():
    md, out = sys.argv[1], sys.argv[2]
    mb = int(sys.argv[sys.argv.index("--mb") + 1]) if "--mb" in sys.argv else 96
    want = mb * 1024 * 1024  # bytes
    path = os.path.join(md, "model.safetensors")
    hdr, base = read_header(path)
    f = open(path, "rb"); mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    picks = ["gate_proj", "up_proj", "down_proj", "q_proj", "o_proj"]
    names = []
    for p in picks:
        for k in hdr:
            if k.endswith("weight") and ".layers.0." in k and p in k:
                names.append(k); break
    buf = bytearray()
    for name in names:
        info = hdr[name]; off0, off1 = info["data_offsets"]
        nb = off1 - off0
        take = min(nb, want - len(buf))
        buf += mm[base + off0: base + off0 + take]
        if len(buf) >= want:
            break
    b, cov = best_base(bytes(buf))
    print(f"sample: {len(buf)} bytes ({len(buf)/1e6:.1f} MB), best_base={b} cov={cov*100:.3f}%")
    with open(out, "wb") as fo:
        fo.write(buf)
    print("wrote", out)

if __name__ == "__main__":
    main()
