#!/usr/bin/env python3
"""C-1 S1 lossless oracle — host round-trip of the exact SplitZip SzBlob layout.

For every decode-path weight tensor: encode bf16 -> (lo|cd|eoff|epos|eval) in the exact
byte layout the device kernel reads (runtime/nvidia/op_gemm.cuh: sz_blob / sz_expand8 /
sz_escape8), decode it back with the kernel's arithmetic, and assert the reconstructed
bf16 bytes are BYTE-IDENTICAL to the originals. Plus a negative control (flip one lo byte
=> must mismatch). This is the mandatory gate: no perf claim before it passes.

Layout (16-byte-aligned sections): lo[N*K] | cd[N*K/2] | eoff[(N+1)*4] | epos[nesc*4]
| eval[nesc*2]. Reconstruct: bf16 = ((lo&0x80)<<8)|((code+exp_base)<<7)|(lo&0x7F), then
per-row escape overwrite for exponents outside [exp_base, exp_base+15].

Usage: sz_oracle.py <model-dir> [--limit N]   (N tensors, default all decode-path)
"""
import sys, os, json, struct, mmap, time
import torch

FULL = None

def read_header(path):
    with open(path, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hn))
    return hdr, 8 + hn

def is_decode_weight(name):
    if "embed_tokens" in name:
        return True
    if ".layers." not in name or not name.endswith("weight"):
        return False
    return any(p in name for p in ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"))

def best_base(exp_hist):
    c = torch.cumsum(exp_hist, 0)
    best_s, best_v = 0, -1
    for s in range(0, 241):
        v = int(c[s + 15] - (c[s - 1] if s else 0))
        if v > best_v:
            best_v, best_s = v, s
    return best_s

def encode_decode(u16, N, K, exp_base, corrupt_lo=False):
    """Return (roundtrip_u16, nesc). Pure-torch vectorized encode + decode.
    corrupt_lo flips one lo payload byte after encode (negative control)."""
    n = u16.numel()
    u = u16.to(torch.int32)
    ex = (u >> 7) & 0xFF
    lo = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)
    code = ex - exp_base
    in_win = (code >= 0) & (code <= 15)
    code = torch.where(in_win, code, torch.zeros_like(code))
    esc_mask = ~in_win
    esc_idx = torch.nonzero(esc_mask, as_tuple=False).flatten()
    nesc = int(esc_idx.numel())
    # cd: pack two 4-bit codes per byte (low nibble = even index)
    code = code.to(torch.int32)
    cd = (code[0::2] | (code[1::2] << 4)).to(torch.uint8)  # n even (N*K even always here)
    # eoff: per-row escape prefix
    rows = N
    row_of = esc_idx // K
    eoff = torch.zeros(rows + 1, dtype=torch.int64)
    if nesc:
        eoff[1:] = torch.bincount(row_of, minlength=rows).cumsum(0)
    epos = esc_idx.to(torch.int64)
    eval_ = u[esc_idx] if nesc else torch.zeros(0, dtype=torch.int32)  # int32 raw bf16
    # ---- DECODE (mirror the kernel exactly) ----
    if corrupt_lo:
        lo = lo.clone(); lo[0] ^= 0x01  # flip one sign/mantissa payload bit
    code_full = torch.zeros(n, dtype=torch.int32)
    code_full[0::2] = (cd.to(torch.int32) & 0xF)
    code_full[1::2] = (cd.to(torch.int32) >> 4)
    exd = code_full + exp_base
    b = lo.to(torch.int32)
    rt = (((b & 0x80) << 8) | (exd << 7) | (b & 0x7F)).to(torch.int32)
    # apply escapes
    if nesc:
        rt[epos] = eval_
    return rt, nesc  # int32 bf16 bit-patterns

def main():
    global FULL
    md = sys.argv[1]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    cfg = json.load(open(os.path.join(md, "config.json")))
    lts = cfg["text_config"]["layer_types"]
    FULL = {i for i, t in enumerate(lts) if t == "full_attention"}
    path = os.path.join(md, "model.safetensors")
    hdr, base = read_header(path)
    f = open(path, "rb"); mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    names = sorted(k for k in hdr if k != "__metadata__" and is_decode_weight(k))
    if limit:
        names = names[:limit]
    npass = nfail = 0
    t0 = time.time()
    for i, name in enumerate(names):
        info = hdr[name]; off0, off1 = info["data_offsets"]; shape = info["shape"]
        n = (off1 - off0) // 2; N = shape[0]; K = n // N
        u16 = torch.frombuffer(mm, dtype=torch.uint16, count=n, offset=base + off0).clone()
        uref = u16.to(torch.int32)
        exh = torch.bincount((uref >> 7) & 0xFF, minlength=256)
        eb = best_base(exh)
        rt, nesc = encode_decode(u16, N, K, eb)
        ok = bool(torch.equal(rt, uref))
        npass += ok; nfail += (not ok)
        if not ok or i < 3 or i == len(names) - 1:
            print(f"  [{ 'OK ' if ok else 'FAIL'}] {name} N={N} K={K} base={eb} "
                  f"nesc={nesc} ({nesc/n*100:.4f}%)")
        if not ok:
            print("    !!! ROUND-TRIP MISMATCH — codec is NOT lossless");
    # negative control: corrupt one lo payload byte after encode -> decode must mismatch
    rt_wrong, _ = encode_decode(u16, N, K, eb, corrupt_lo=True)
    neg_ok = not torch.equal(rt_wrong, uref)
    print(f"\n  negative control (wrong exp_base -> mismatch expected): "
          f"{'DETECTED (good)' if neg_ok else 'NOT DETECTED (BAD)'}")
    print(f"\nS1 host oracle: {npass}/{len(names)} tensors byte-identical, {nfail} failures "
          f"({time.time()-t0:.1f}s)")
    print("RESULT:", "PASS" if (nfail == 0 and neg_ok) else "FAIL")
    sys.exit(0 if (nfail == 0 and neg_ok) else 1)

if __name__ == "__main__":
    main()
