#!/usr/bin/env python3
"""C-0 compressibility audit for the P9 lossless-compression plan (v2).

CPU-only. Reads the real gemma-4-12B checkpoint (+ fp8 twins when present) and
answers, per tensor class x {sliding,full} layer type:
  1. bf16 hi-byte / lo-byte plane Shannon entropy (bits/byte) + exponent-field
     entropy; top-16 contiguous exponent-window coverage, escape rate, and the
     per-tensor EXP_BASE (window start).
  2. splitzip-12b effective ratio -- EXACT emulation of the in-tree layout
     (runtime/nvidia/experiments/splitzip_gemv.cu charged footprint):
       bytes = n*1.5 (lo 1B + cd 4b) + (rows+1)*4 (eoff) + escapes*6 (epos u32
       + eval u16) + 16
  3. 3-bit top-8 window variant: n*11/8 + (rows+1)*4 + escapes*6 + 16.
  4. Reference upper bounds: zstd-19 whole-tensor and byte-plane-split zstd
     (hi[] and lo[] compressed separately) per class -- the "near Shannon?"
     sanity column.
  5. fp8 twins: e4m3 byte entropy, top-8 exponent-field window coverage, and
     the 7 b/elem C-3fp8 ratio emulation (n*7/8 + (rows+1)*4 + escapes*5 + 16).

Decode-path tensors only (the 23.8 GB/token stream): embed_tokens (tied
lm_head), q/k/v/o projections, gate/up/down MLP. Norms/scalars and
vision/audio towers are not part of the decode weight stream.

Go thresholds (plan section C-0): C-1 GO if bytes-weighted mean splitzip ratio
>= 1.25 AND escape rate <= 0.5% on every major class; C-3fp8 GO if fp8 ratio
>= 1.12. KV rows deferred (needs a GPU-side dump; weights alone gate C-1).

Usage: compress_audit.py <model-dir> [--out PREFIX] [--no-zstd]
  writes PREFIX.json and PREFIX.md (default perf-data/c0-compress-audit)
"""
import sys, os, json, struct, mmap, math, ctypes, subprocess, time
import torch

CHUNK = 1 << 26          # elements per processing chunk (64 Mi)
FULL_LAYERS = None       # set from config.json layer_types


def raw_bytes(t):
    t = t.contiguous()
    return ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())


def read_header(path):
    with open(path, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hn))
    return hdr, 8 + hn


def layer_of(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return int(parts[i + 1])
    return None


def classify(name):
    """-> (class, layer_type) or None if not a decode-path weight."""
    if "embed_tokens" in name:
        return "embed/lm_head", "shared"
    li = layer_of(name)
    if li is None:
        return None
    lt = "full" if li in FULL_LAYERS else "sliding"
    for pat, cls in (("q_proj", "qkv"), ("k_proj", "qkv"), ("v_proj", "qkv"),
                     ("o_proj", "o"), ("gate_proj", "gate/up"),
                     ("up_proj", "gate/up"), ("down_proj", "down")):
        if pat in name and name.endswith("weight"):
            return cls, lt
    return None


def entropy(hist):
    n = int(hist.sum())
    if n == 0:
        return 0.0
    p = hist.to(torch.float64) / n
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def best_window(hist, width):
    """(start, covered) best contiguous window of `width` bins."""
    c = torch.cumsum(hist, 0)
    best_s, best_v = 0, -1
    for s in range(0, hist.numel() - width + 1):
        v = int(c[s + width - 1] - (c[s - 1] if s else 0))
        if v > best_v:
            best_v, best_s = v, s
    return best_s, best_v


class Agg:
    """Per (class, ltype) accumulator."""
    def __init__(self):
        self.n = 0                 # elements
        self.rows = 0
        self.tensors = 0
        self.h_hi = torch.zeros(256, dtype=torch.int64)
        self.h_lo = torch.zeros(256, dtype=torch.int64)
        self.h_exp = torch.zeros(256, dtype=torch.int64)
        self.esc16 = 0             # per-tensor-EXP_BASE escapes, summed
        self.esc8 = 0
        self.sz12_bytes = 0.0
        self.sz11_bytes = 0.0
        self.exp_bases = []        # per-tensor (base16, cov%)
        self.zstd_whole = None
        self.zstd_hi = None
        self.zstd_lo = None


def audit_bf16(path, base_off, hdr, names, out_rows, aggs, do_zstd):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    for name in names:
        cls_lt = classify(name)
        if cls_lt is None:
            continue
        info = hdr[name]
        assert info["dtype"] == "BF16", name
        off0, off1 = info["data_offsets"]
        shape = info["shape"]
        n = 1
        for s in shape:
            n *= s
        rows = shape[0]
        t0 = time.time()
        h_hi = torch.zeros(256, dtype=torch.int64)
        h_lo = torch.zeros(256, dtype=torch.int64)
        h_exp = torch.zeros(256, dtype=torch.int64)
        for cs in range(0, n, CHUNK):
            ce = min(n, cs + CHUNK)
            u16 = torch.frombuffer(mm, dtype=torch.uint16,
                                   count=ce - cs, offset=base_off + off0 + cs * 2)
            u = u16.to(torch.int32)
            h_hi += torch.bincount((u >> 8) & 0xFF, minlength=256)
            h_lo += torch.bincount(u & 0xFF, minlength=256)
            h_exp += torch.bincount((u >> 7) & 0xFF, minlength=256)
        b16, cov16 = best_window(h_exp, 16)
        b8, cov8 = best_window(h_exp, 8)
        esc16, esc8 = n - cov16, n - cov8
        sz12 = n * 1.5 + (rows + 1) * 4 + esc16 * 6 + 16
        sz11 = n * 11 / 8 + (rows + 1) * 4 + esc8 * 6 + 16
        row = {
            "name": name, "class": cls_lt[0], "ltype": cls_lt[1],
            "shape": shape, "elems": n, "raw_bytes": n * 2,
            "H_hi": round(entropy(h_hi), 4), "H_lo": round(entropy(h_lo), 4),
            "H_exp": round(entropy(h_exp), 4),
            "exp_base16": b16, "win16_cov": round(cov16 / n, 6),
            "esc16_rate": round(esc16 / n, 6),
            "ratio_sz12": round(2 * n / sz12, 4),
            "exp_base8": b8, "esc8_rate": round(esc8 / n, 6),
            "ratio_sz11": round(2 * n / sz11, 4),
        }
        out_rows.append(row)
        a = aggs.setdefault(cls_lt, Agg())
        a.n += n; a.rows += rows; a.tensors += 1
        a.h_hi += h_hi; a.h_lo += h_lo; a.h_exp += h_exp
        a.esc16 += esc16; a.esc8 += esc8
        a.sz12_bytes += sz12; a.sz11_bytes += sz11
        a.exp_bases.append((b16, round(cov16 / n, 6)))
        print(f"  {name}  {shape}  sz12={row['ratio_sz12']}x esc={row['esc16_rate']*100:.4f}% "
              f"base={b16} ({time.time()-t0:.1f}s)", flush=True)
    # zstd reference bounds, per class (streams re-read from page cache)
    if do_zstd:
        for cls_lt, a in aggs.items():
            members = [r for r in out_rows if (r["class"], r["ltype"]) == cls_lt]
            procs = {}
            for k in ("whole", "hi", "lo"):
                procs[k] = subprocess.Popen(
                    ["zstd", "-19", "-T0", "-c"], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            import threading
            sizes = {}
            readers = []
            for k, p in procs.items():
                def rd(k=k, p=p):
                    tot = 0
                    while True:
                        b = p.stdout.read(1 << 20)
                        if not b:
                            break
                        tot += len(b)
                    sizes[k] = tot
                th = threading.Thread(target=rd)
                th.start(); readers.append(th)
            t0 = time.time()
            for r in members:
                info = hdr[r["name"]]
                off0, _ = info["data_offsets"]
                n = r["elems"]
                for cs in range(0, n, CHUNK):
                    ce = min(n, cs + CHUNK)
                    u16 = torch.frombuffer(mm, dtype=torch.uint16,
                                           count=ce - cs, offset=base_off + off0 + cs * 2)
                    u = u16.to(torch.int32)
                    procs["whole"].stdin.write(raw_bytes(u16))
                    procs["hi"].stdin.write(raw_bytes(((u >> 8) & 0xFF).to(torch.uint8)))
                    procs["lo"].stdin.write(raw_bytes((u & 0xFF).to(torch.uint8)))
            for p in procs.values():
                p.stdin.close()
            for th in readers:
                th.join()
            for p in procs.values():
                p.wait()
            a.zstd_whole, a.zstd_hi, a.zstd_lo = sizes["whole"], sizes["hi"], sizes["lo"]
            print(f"  zstd {cls_lt}: whole={2*a.n/sizes['whole']:.4f}x "
                  f"planes={2*a.n/(sizes['hi']+sizes['lo']):.4f}x ({time.time()-t0:.1f}s)",
                  flush=True)
    mm.close(); f.close()


def audit_fp8(path, out_rows, aggs, do_zstd):
    hdr, base_off = read_header(path)
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    for name in sorted(hdr):
        if name == "__metadata__" or name.endswith("_scale"):
            continue
        cls_lt = classify(name)
        if cls_lt is None:
            continue
        info = hdr[name]
        assert info["dtype"] == "F8_E4M3", name
        off0, _ = info["data_offsets"]
        shape = info["shape"]
        n = 1
        for s in shape:
            n *= s
        rows = shape[0]
        h_b = torch.zeros(256, dtype=torch.int64)
        h_exp = torch.zeros(16, dtype=torch.int64)
        for cs in range(0, n, CHUNK):
            ce = min(n, cs + CHUNK)
            u8 = torch.frombuffer(mm, dtype=torch.uint8, count=ce - cs,
                                  offset=base_off + off0 + cs)
            u = u8.to(torch.int32)
            h_b += torch.bincount(u, minlength=256)
            h_exp += torch.bincount((u >> 3) & 0xF, minlength=16)
        b8, cov8 = best_window(h_exp, 8)
        esc8 = n - cov8
        sz7 = n * 7 / 8 + (rows + 1) * 4 + esc8 * 5 + 16
        row = {
            "name": name, "class": cls_lt[0], "ltype": cls_lt[1], "shape": shape,
            "elems": n, "raw_bytes": n, "H_byte": round(entropy(h_b), 4),
            "H_exp4": round(entropy(h_exp), 4), "exp_base8": b8,
            "esc8_rate": round(esc8 / n, 6), "ratio_sz7": round(n / sz7, 4),
        }
        out_rows.append(row)
        a = aggs.setdefault(cls_lt, Agg())
        a.n += n; a.rows += rows; a.tensors += 1
        a.h_hi += h_b
        a.h_exp[:16] += h_exp
        a.esc8 += esc8
        a.sz12_bytes += sz7      # reuse field: fp8 7b footprint
        a.exp_bases.append((b8, round(cov8 / n, 6)))
    if do_zstd:
        for cls_lt, a in aggs.items():
            members = [r for r in out_rows if (r["class"], r["ltype"]) == cls_lt]
            p = subprocess.Popen(["zstd", "-19", "-T0", "-c"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            import threading
            sizes = {}
            def rd():
                tot = 0
                while True:
                    b = p.stdout.read(1 << 20)
                    if not b:
                        break
                    tot += len(b)
                sizes["w"] = tot
            th = threading.Thread(target=rd); th.start()
            for r in members:
                info = hdr[r["name"]]
                off0, _ = info["data_offsets"]
                n = r["elems"]
                for cs in range(0, n, CHUNK):
                    ce = min(n, cs + CHUNK)
                    u8 = torch.frombuffer(mm, dtype=torch.uint8, count=ce - cs,
                                          offset=base_off + off0 + cs)
                    p.stdin.write(raw_bytes(u8))
            p.stdin.close(); th.join(); p.wait()
            a.zstd_whole = sizes["w"]
    mm.close(); f.close()


def agg_row(cls_lt, a, fp8=False):
    n = a.n
    d = {
        "class": cls_lt[0], "ltype": cls_lt[1], "tensors": a.tensors,
        "elems": n, "raw_gb": round(n * (1 if fp8 else 2) / 1e9, 3),
        "H_hi" if not fp8 else "H_byte": round(entropy(a.h_hi), 4),
        "H_exp": round(entropy(a.h_exp), 4),
    }
    if not fp8:
        d["H_lo"] = round(entropy(a.h_lo), 4)
        d["esc16_rate"] = round(a.esc16 / n, 6)
        d["ratio_sz12"] = round(2 * n / a.sz12_bytes, 4)
        d["esc8_rate"] = round(a.esc8 / n, 6)
        d["ratio_sz11"] = round(2 * n / a.sz11_bytes, 4)
        d["exp_base16_range"] = [min(b for b, _ in a.exp_bases),
                                 max(b for b, _ in a.exp_bases)]
        if a.zstd_whole:
            d["zstd19_whole"] = round(2 * n / a.zstd_whole, 4)
            d["zstd19_planes"] = round(2 * n / (a.zstd_hi + a.zstd_lo), 4)
            d["zstd19_hi_bits"] = round(8 * a.zstd_hi / n, 3)
            d["zstd19_lo_bits"] = round(8 * a.zstd_lo / n, 3)
    else:
        d["esc8_rate"] = round(a.esc8 / n, 6)
        d["ratio_sz7"] = round(n / a.sz12_bytes, 4)
        d["exp_base8_range"] = [min(b for b, _ in a.exp_bases),
                                max(b for b, _ in a.exp_bases)]
        if a.zstd_whole:
            d["zstd19_whole"] = round(n / a.zstd_whole, 4)
    return d


def main():
    global FULL_LAYERS
    model_dir = sys.argv[1]
    out = "perf-data/c0-compress-audit"
    do_zstd = "--no-zstd" not in sys.argv
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    lts = cfg["text_config"]["layer_types"]
    FULL_LAYERS = {i for i, t in enumerate(lts) if t == "full_attention"}
    print(f"full layers: {sorted(FULL_LAYERS)}")

    path = os.path.join(model_dir, "model.safetensors")
    hdr, base_off = read_header(path)
    names = sorted(k for k in hdr if k != "__metadata__")
    bf_rows, bf_aggs = [], {}
    audit_bf16(path, base_off, hdr, names, bf_rows, bf_aggs, do_zstd)

    fp8_path = os.path.join(model_dir, "fp8", "model.safetensors")
    fp_rows, fp_aggs = [], {}
    if os.path.exists(fp8_path):
        audit_fp8(fp8_path, fp_rows, fp_aggs, do_zstd)

    # ---- verdicts ----
    tot_n = sum(a.n for a in bf_aggs.values())
    tot_sz12 = sum(a.sz12_bytes for a in bf_aggs.values())
    wmean_ratio = 2 * tot_n / tot_sz12
    worst_esc = max(a.esc16 / a.n for a in bf_aggs.values())
    c1_go = wmean_ratio >= 1.25 and worst_esc <= 0.005
    fp8_ratio = None
    c3_go = None
    if fp_aggs:
        fn = sum(a.n for a in fp_aggs.values())
        fp8_ratio = fn / sum(a.sz12_bytes for a in fp_aggs.values())
        c3_go = fp8_ratio >= 1.12

    classes = [agg_row(k, a) for k, a in sorted(bf_aggs.items())]
    fclasses = [agg_row(k, a, fp8=True) for k, a in sorted(fp_aggs.items())]
    res = {
        "date": time.strftime("%Y-%m-%d"), "model": model_dir,
        "plan": "the design notes C-0",
        "decode_stream_gb": round(tot_n * 2 / 1e9, 3),
        "bf16_classes": classes, "bf16_tensors": bf_rows,
        "fp8_classes": fclasses, "fp8_tensors": fp_rows,
        "verdict": {
            "splitzip12_bytes_weighted_ratio": round(wmean_ratio, 4),
            "worst_class_escape_rate": round(worst_esc, 6),
            "C1_go": c1_go,
            "fp8_ratio_sz7": round(fp8_ratio, 4) if fp8_ratio else None,
            "C3fp8_go": c3_go,
            "kv_audit": "deferred (needs GPU-side KV dump; weights alone gate C-1)",
        },
    }
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=1)

    # ---- markdown ----
    L = []
    L.append("# C-0 compressibility audit — gemma-4-12B (P9 v2 step 0)\n")
    L.append(f"Date {res['date']}. CPU-only, real checkpoint `{model_dir}`. "
             f"Decode-path weight stream audited: {res['decode_stream_gb']} GB bf16 "
             "(328 projections + tied embed/lm_head).\n")
    L.append("## bf16 classes (per class x layer type; bytes-weighted within class)\n")
    hdr_cols = ("class|ltype|GB|H_hi|H_lo|H_exp|EXP_BASE16|esc16 %|**sz12 ratio**|"
                "sz11(3b) ratio|esc8 %|zstd19 whole|zstd19 planes")
    L.append("| " + " | ".join(hdr_cols.split("|")) + " |")
    L.append("|" + "---|" * len(hdr_cols.split("|")))
    for c in classes:
        L.append("| {class} | {ltype} | {raw_gb} | {H_hi} | {H_lo} | {H_exp} | "
                 "{eb} | {e16:.4f} | **{r12}x** | {r11}x | {e8:.4f} | {zw} | {zp} |".format(
                     eb="{}..{}".format(*c["exp_base16_range"]),
                     e16=c["esc16_rate"] * 100, r12=c["ratio_sz12"],
                     r11=c["ratio_sz11"], e8=c["esc8_rate"] * 100,
                     zw=c.get("zstd19_whole", "-"), zp=c.get("zstd19_planes", "-"), **c))
    if fclasses:
        L.append("\n## fp8 (e4m3) twins\n")
        L.append("| class | ltype | GB | H_byte | H_exp4 | EXP_BASE8 | esc8 % | "
                 "**sz7 ratio** | zstd19 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in fclasses:
            L.append("| {class} | {ltype} | {raw_gb} | {H_byte} | {H_exp} | {eb} | "
                     "{e8:.4f} | **{r7}x** | {zw} |".format(
                         eb="{}..{}".format(*c["exp_base8_range"]),
                         e8=c["esc8_rate"] * 100, r7=c["ratio_sz7"],
                         zw=c.get("zstd19_whole", "-"), **c))
    v = res["verdict"]
    L.append("\n## Verdict (plan C-0 thresholds)\n")
    L.append(f"- splitzip-12b bytes-weighted mean ratio: **{v['splitzip12_bytes_weighted_ratio']}x** "
             "(threshold >= 1.25)")
    L.append(f"- worst class escape rate: **{v['worst_class_escape_rate']*100:.4f}%** "
             "(threshold <= 0.5%)")
    L.append(f"- **C-1 {'GO' if v['C1_go'] else 'NO-GO'}**")
    if v["fp8_ratio_sz7"]:
        L.append(f"- fp8 sz7 bytes-weighted ratio: **{v['fp8_ratio_sz7']}x** (threshold >= 1.12) "
                 f"=> **C-3fp8 {'GO' if v['C3fp8_go'] else 'NO-GO'}**")
    L.append(f"- KV: {v['kv_audit']}")
    with open(out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[-8:]))
    print(f"\nwrote {out}.json / {out}.md")


if __name__ == "__main__":
    main()
