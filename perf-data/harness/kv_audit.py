#!/usr/bin/env python3
"""KV-0 compressibility audit for the P10 kv-zip plan (plans/p10-kv-zip.md).

CPU-only. Reads a PLOW_DUMP_KV dump (raw head-major ring bytes exactly as
d_headnorm_rope wrote them) and answers, per (layer-type, K|V):

  1. bf16 exponent entropy H_exp; conditional H(exp|head), H(exp|dim) — decides
     the EXP_BASE granularity (per-tensor vs per-head vs per-dim).
  2. top-16 (4-bit sz12) and top-8 (3-bit sz11) contiguous-window escape rates
     at each base granularity.
  3. per-ROW escape-count tail under the per-head top-16 base — the ring
     constraint means escapes must live in FIXED slots/row; p(esc>E) sizes E.
  4. net sz12 ratio including slot provisioning.
  5. fp8-e4m3 twin, derived with the exact d_headnorm_rope_fp8 scheme (per-row
     scale = amax/448, RNE): byte entropy, 3-bit exponent-field window escape,
     net sz7 ratio vs the fp8 stream. Gate >= 1.12 (same bar as C-3fp8).
  6. fp4 (e2m1, block-16 amax/6 scales) twin: 4-bit code entropy -> lossless
     ceiling 4/H4. Informational (plow has no fp4-KV path).
  7. positional stability (first vs last quarter of the full-layer ring) and
     lo-plane (sign|mant7) entropy — expected ~8 b (incompressible).

Geometry is inferred from tensor byte sizes (defaults = gemma-4-12B @ b1:
full kvh1/hd512, sliding kvh8/ring16384/hd256; --full-geom sets the ring).

Usage: kv_audit.py <dump-dir> [--out PREFIX] [--full-geom 1,32768,512]
"""
import argparse, json, math, os, sys, time
import torch

torch.set_num_threads(max(8, os.cpu_count() // 4))

E4M3_MAX = 448.0
FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def entropy(hist):
    n = int(hist.sum())
    if n == 0:
        return 0.0
    p = hist.to(torch.float64) / n
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def best_window(hist, width):
    """(start, covered_count) of the best contiguous window of `width` bins."""
    c = torch.cumsum(hist, 0)
    best_s, best_v = 0, -1
    for s in range(hist.numel() - width + 1):
        v = int(c[s + width - 1] - (c[s - 1] if s else 0))
        if v > best_v:
            best_v, best_s = v, s
    return best_s, best_v


def cond_entropy(hist2d):
    """H(exp | group) from [ngroup, nbins] hists, weighted by group mass."""
    n = hist2d.sum()
    if n == 0:
        return 0.0
    h = 0.0
    for g in range(hist2d.shape[0]):
        ng = int(hist2d[g].sum())
        if ng:
            h += ng / n * entropy(hist2d[g])
    return float(h)


def group_hists(vals, ngroup, group_idx, nbins):
    flat = (group_idx.to(torch.int64) * nbins + vals.to(torch.int64)).reshape(-1)
    return torch.bincount(flat, minlength=ngroup * nbins).reshape(ngroup, nbins)


def esc_from_bases(exp, bases, width, axis_idx):
    lo = bases[axis_idx]
    return (exp < lo) | (exp >= lo + width)


class Agg:
    def __init__(self):
        self.n = 0
        self.rows = 0
        self.tensors = 0
        self.h_exp = torch.zeros(256, dtype=torch.int64)
        self.h_lo = torch.zeros(256, dtype=torch.int64)
        self.esc16_t = 0        # per-tensor base
        self.esc16_h = 0        # per-head base
        self.esc16_d = 0        # per-dim base
        self.esc8_h = 0
        self.hexp_list = []     # per-tensor H_exp
        self.hexp_head = []     # per-tensor H(exp|head)
        self.hexp_dim = []      # per-tensor H(exp|dim)
        self.bases = []         # per-tensor (base16, cov)
        self.row_esc_hist = torch.zeros(4096, dtype=torch.int64)  # per-row esc16_h counts
        self.pos_hexp = []      # (first-quarter, last-quarter) H_exp, full layers
        # fp8 twin
        self.f8_h_byte = torch.zeros(256, dtype=torch.int64)
        self.f8_h_exp = torch.zeros(16, dtype=torch.int64)
        self.f8_esc8_h = 0      # 3-bit window, per-head base
        self.f8_row_esc_hist = torch.zeros(4096, dtype=torch.int64)
        # fp4 twin
        self.f4_h_code = torch.zeros(16, dtype=torch.int64)


def audit_tensor(x_u16, hd, kvh, agg, ltype):
    """x_u16: [kvh, rows, hd] uint16 bf16 bits of the VALID ring rows."""
    x_u16 = x_u16.contiguous()
    kvh_, rows, hd_ = x_u16.shape
    exp = ((x_u16.to(torch.int32) >> 7) & 0xFF).to(torch.uint8)
    lo = ((x_u16.to(torch.int32) >> 8) & 0x80) | (x_u16.to(torch.int32) & 0x7F)

    agg.n += exp.numel()
    agg.rows += kvh_ * rows
    agg.tensors += 1
    h_exp = torch.bincount(exp.reshape(-1).to(torch.int64), minlength=256)
    agg.h_exp += h_exp
    agg.h_lo += torch.bincount(lo.reshape(-1).to(torch.int64), minlength=256)
    agg.hexp_list.append(entropy(h_exp))

    # per-tensor base
    b16, cov16 = best_window(h_exp, 16)
    agg.esc16_t += exp.numel() - cov16
    agg.bases.append((b16, cov16 / exp.numel()))

    # per-head base
    head_idx = torch.arange(kvh_).view(kvh_, 1, 1).expand_as(exp)
    hh = group_hists(exp, kvh_, head_idx, 256)
    agg.hexp_head.append(cond_entropy(hh))
    hbases = torch.zeros(kvh_, dtype=torch.int64)
    for h in range(kvh_):
        bs, cov = best_window(hh[h], 16)
        hbases[h] = bs
        agg.esc16_h += int(hh[h].sum()) - cov
        bs8, cov8 = best_window(hh[h], 8)
        agg.esc8_h += int(hh[h].sum()) - cov8

    # per-dim base
    dim_idx = torch.arange(hd_).view(1, 1, hd_).expand_as(exp)
    dh = group_hists(exp, hd_, dim_idx, 256)
    agg.hexp_dim.append(cond_entropy(dh))
    for d in range(hd_):
        bs, cov = best_window(dh[d], 16)
        agg.esc16_d += int(dh[d].sum()) - cov

    # per-row escape tail under per-head base (the slot-provisioning datum)
    esc = esc_from_bases(exp, hbases, 16, head_idx)
    per_row = esc.sum(dim=2).reshape(-1).clamp(max=4095)
    agg.row_esc_hist += torch.bincount(per_row.to(torch.int64), minlength=4096)

    # positional stability (full layers carry the growing ring)
    if ltype == "full" and rows >= 4096:
        q = rows // 4
        for sl in (slice(0, q), slice(rows - q, rows)):
            hq = torch.bincount(exp[:, sl, :].reshape(-1).to(torch.int64), minlength=256)
            agg.pos_hexp.append(entropy(hq))

    # ---- fp8 e4m3 twin: exact d_headnorm_rope_fp8 scheme (per-row amax/448, RNE)
    v = x_u16.view(torch.bfloat16).to(torch.float32)
    amax = v.abs().amax(dim=2, keepdim=True)
    scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
    q8 = (v / scale).to(torch.float8_e4m3fn)
    b8 = q8.view(torch.uint8).to(torch.int32)
    agg.f8_h_byte += torch.bincount(b8.reshape(-1).to(torch.int64), minlength=256)
    e8 = ((b8 >> 3) & 0xF).to(torch.uint8)
    f8h = group_hists(e8, kvh_, head_idx, 16)
    agg.f8_h_exp += f8h.sum(dim=0)
    f8bases = torch.zeros(kvh_, dtype=torch.int64)
    for h in range(kvh_):
        bs, cov = best_window(f8h[h], 8)
        f8bases[h] = bs
        agg.f8_esc8_h += int(f8h[h].sum()) - cov
    esc8 = esc_from_bases(e8, f8bases, 8, head_idx)
    per_row8 = esc8.sum(dim=2).reshape(-1).clamp(max=4095)
    agg.f8_row_esc_hist += torch.bincount(per_row8.to(torch.int64), minlength=4096)

    # ---- fp4 e2m1 twin: block-16 amax/6 scales, nearest-value quantize
    vb = v.reshape(kvh_, rows, hd_ // 16, 16)
    bmax = vb.abs().amax(dim=3, keepdim=True)
    bscale = torch.where(bmax > 0, bmax / 6.0, torch.ones_like(bmax))
    vq = vb / bscale
    idx = (vq.abs().unsqueeze(-1) - FP4_VALS).abs().argmin(dim=-1)
    code = (idx + torch.where(vq < 0, 8, 0)).to(torch.uint8)  # s<<3 | mag-idx
    agg.f4_h_code += torch.bincount(code.reshape(-1).to(torch.int64), minlength=16)


def tail(hist, E):
    n = int(hist.sum())
    return float(hist[E + 1 :].sum()) / n if n else 0.0


def net_ratio(hd, bits_per_elem, E, slot_bytes):
    return (hd * 2.0) / (hd * bits_per_elem / 8.0 + E * slot_bytes)


def pick_E(hist, budget=1e-6, cap=64):
    for E in range(cap + 1):
        if tail(hist, E) <= budget:
            return E
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default="perf-data/kv0-kv-audit")
    ap.add_argument("--full-geom", default="1,32768,512")   # kvh,ring,hd
    ap.add_argument("--slide-geom", default="8,16384,256")
    args = ap.parse_args()

    fk, fr, fh = map(int, args.full_geom.split(","))
    sk, sr, sh = map(int, args.slide_geom.split(","))
    full_bytes = fk * fr * fh * 2
    slide_bytes = sk * sr * sh * 2

    man = {}
    ctx = None
    for line in open(os.path.join(args.dump, "manifest.txt")):
        p = line.split()
        if p[0] == "ctx":
            ctx = int(p[1])
        else:
            man[p[0]] = int(p[1])
    assert ctx, "manifest has no ctx"
    print(f"dump {args.dump}: ctx={ctx}, {len(man)} kv tensors", flush=True)

    aggs = {}  # (ltype, k|v) -> Agg
    t0 = time.time()
    for name, nbytes in sorted(man.items()):
        if nbytes == full_bytes:
            ltype, kvh, ring, hd = "full", fk, fr, fh
        elif nbytes == slide_bytes:
            ltype, kvh, ring, hd = "sliding", sk, sr, sh
        else:
            print(f"  SKIP {name}: unrecognized size {nbytes}", flush=True)
            continue
        rows = min(ctx, ring)
        kv = name.rsplit(".", 1)[1]  # 'k' or 'v'
        x = torch.from_file(os.path.join(args.dump, name + ".raw"),
                            dtype=torch.uint16, size=kvh * ring * hd)
        x = x.reshape(kvh, ring, hd)[:, :rows, :]
        agg = aggs.setdefault((ltype, kv), Agg())
        audit_tensor(x, hd, kvh, agg, ltype)
        print(f"  {name} done ({ltype}.{kv}) t={time.time()-t0:.0f}s", flush=True)
    print(f"audit pass done in {time.time()-t0:.0f}s", flush=True)

    hd_of = {"full": fh, "sliding": sh}
    res = {"ctx": ctx, "dump": args.dump, "classes": {}}
    for (ltype, kv), a in sorted(aggs.items()):
        hd = hd_of[ltype]
        E12 = pick_E(a.row_esc_hist, 1e-6)
        E12_p99 = pick_E(a.row_esc_hist, 1e-2)
        E7 = pick_E(a.f8_row_esc_hist, 1e-6)
        h4 = entropy(a.f4_h_code)
        cls = {
            "GB": a.n * 2 / 1e9, "tensors": a.tensors,
            "H_exp": entropy(a.h_exp), "H_lo": entropy(a.h_lo),
            "H_exp_tensor_mean": sum(a.hexp_list) / len(a.hexp_list),
            "H_exp_given_head": sum(a.hexp_head) / len(a.hexp_head),
            "H_exp_given_dim": sum(a.hexp_dim) / len(a.hexp_dim),
            "esc16_tensor_pct": 100.0 * a.esc16_t / a.n,
            "esc16_head_pct": 100.0 * a.esc16_h / a.n,
            "esc16_dim_pct": 100.0 * a.esc16_d / a.n,
            "esc8_head_pct": 100.0 * a.esc8_h / a.n,
            "row_esc_p0": tail(a.row_esc_hist, 0),
            "row_esc_p_gt4": tail(a.row_esc_hist, 4),
            "row_esc_p_gt8": tail(a.row_esc_hist, 8),
            "row_esc_max": int(torch.nonzero(a.row_esc_hist).max()),
            "E_slots_1e-6": E12, "E_slots_1e-2": E12_p99,
            "sz12_net_ratio": net_ratio(hd, 12, E12, 4) if E12 is not None else None,
            "sz11_layout_ratio": 16 / 11.0,  # slots not provisioned; see esc8_head_pct
            "pos_hexp": a.pos_hexp[:8],
            "fp8_H_byte": entropy(a.f8_h_byte),
            "fp8_H_exp4": entropy(a.f8_h_exp),
            "fp8_esc8_head_pct": 100.0 * a.f8_esc8_h / a.n,
            "fp8_E_slots_1e-6": E7,
            "fp4_H_code": h4,
            "fp4_ceiling": 4.0 / h4 if h4 > 0 else None,
        }
        # fp8 ratio is vs the fp8 stream (1 B/elem), not vs bf16:
        cls["fp8_sz7_net_ratio"] = ((hd * 1.0) / (hd * 7 / 8.0 + E7 * 3)
                                    if E7 is not None else None)
        res["classes"][f"{ltype}.{kv.upper()}"] = cls

    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)

    # markdown
    lines = [f"# KV-0 audit — {args.dump} (ctx {ctx})", ""]
    cols = ["GB", "H_exp", "H_exp_given_head", "H_exp_given_dim",
            "esc16_tensor_pct", "esc16_head_pct", "esc16_dim_pct", "esc8_head_pct",
            "E_slots_1e-6", "sz12_net_ratio", "fp8_H_exp4", "fp8_esc8_head_pct",
            "fp8_sz7_net_ratio", "fp4_H_code", "fp4_ceiling", "H_lo"]
    lines.append("| class | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for k, c in res["classes"].items():
        row = [k] + [f"{c[x]:.4g}" if isinstance(c[x], float) else str(c[x]) for x in cols]
        lines.append("| " + " | ".join(row) + " |")
    open(args.out + ".md", "w").write("\n".join(lines) + "\n")
    print(json.dumps(res, indent=1))
    print(f"wrote {args.out}.json / .md")


if __name__ == "__main__":
    main()
