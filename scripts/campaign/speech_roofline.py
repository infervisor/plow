#!/usr/bin/env python3
"""Per-op roofline of a packet program: analytic bytes/FLOPs per instruction (from
`plowrt disasm --format json`) against measured marginal times (instruction-cap sweeps from
`step_bench --sweep` or `packet_bench --sweep`, or per-program times from `packet_bench --each`).

  speech_roofline.py DISASM.json PROGRAM SWEEP.jsonl [--ctx N] [--label L] [--json OUT]

Ceilings are the measured H100 SXM numbers (docs/bringup: measure yours): HBM read 3.15 TB/s,
FP32 FFMA 47.6 TFLOP/s, TF32 tensor 389, BF16 tensor 803. Each op is priced at the ceiling of the
arithmetic it actually runs (a DenseGemmF32 without its tensor-core flag is priced at FP32 FFMA),
and its roofline time is max(bytes / BW, flops / peak).
"""
import argparse, collections, json, math

BW = 3.15e12
PEAK = {"fp32": 47.6e12, "tf32": 389e12, "bf16": 803e12}


def load(path):
    s = open(path).read()
    return json.loads(s[s.index("{\n"):])


def ints(inst):
    return {x["name"].rstrip("?"): x["value"] for x in inst.get("ints", []) if "value" in x}


def present(inst, name):
    return any(t.get("name") == name and t.get("present") for t in inst.get("tensors", []))


def cost(inst, ctx):
    """(bytes, flops, precision) of one instruction."""
    op, i = inst["op_name"], ints(inst)
    g = lambda k, d=0: int(i.get(k, d) or 0)
    if op == "Gemv":
        m, n, k = g("M", 1), g("N"), g("K")
        return n * k * 2 + m * (k + n) * 2, 2 * m * n * k, "bf16"
    if op == "GemvGlu":
        m, n, k = g("M", 1), g("N"), g("K")
        return 2 * n * k * 2 + m * (k + n) * 2, 4 * m * n * k, "bf16"
    if op == "GemvQkv":
        m, k = g("M", 1), g("K")
        n = g("Nq") + g("Nk") + g("Nv")
        return n * k * 2 + m * (k + n) * 2, 2 * m * n * k, "bf16"
    if op == "FlashDecode":
        b, h, kvh, hd = g("n_batch", 1), g("n_head"), g("n_kv_head"), g("hd")
        return max(b, 1) * ctx * kvh * hd * 2 * 2, 4 * max(b, 1) * h * ctx * hd, "fp32"
    if op in ("FlashMerge",):
        return g("n_batch", 1) * g("n_head") * g("nsplit") * g("hd") * 4 * 2, 0, "fp32"
    if op in ("RmsNorm", "AddNorm"):
        return g("rows", 1) * g("feat") * 2 * (4 if op == "AddNorm" else 2), 0, "fp32"
    if op == "HeadNormRope":
        return g("ntok", 1) * g("nhead") * g("hd") * 2 * 2, 0, "fp32"
    if op in ("Embed",):
        return g("ntok", 1) * g("hidden") * 2 * 2, 0, "fp32"
    if op in ("EmbedPosBf16", "EmbedOverlayBf16"):
        return g("rows", 1) * g("width") * 2 * 3, 0, "fp32"
    if op == "Argmax":
        return g("n") * max(g("n_batch"), 1) * 2, 0, "fp32"
    if op == "ArgmaxFin":
        return 1024, 0, "fp32"
    if op == "Gemm":
        m, n, k = g("M"), g("N"), g("K")
        return (n * k + m * k + m * n) * 2, 2 * m * n * k, "bf16"
    if op == "FlashPrefill":
        nq, nkv, h, hd = g("n_q"), g("n_kv"), g("n_head"), g("hd")
        return (nq * h * hd * 2 + nkv * g("n_kv_head") * hd * 2 * 2) * 2, 2 * nq * nkv * h * hd, "bf16"
    if op == "DenseGemmF32":
        m, n, k, f = g("M"), g("N"), g("K"), g("flags")
        wbytes = 2 if f & 4 else 4
        prec = "bf16" if f & 8 else ("tf32" if f & 16 else "fp32")
        return n * k * wbytes + m * k * 4 + m * n * 4, 2 * m * n * k, prec
    if op in ("Conv1dF32", "ConvTranspose1dF32"):
        b, tin, cin, cout, kk, s, groups = g("batch", 1), g("in_rows"), g("in_channels"), g("out_channels"), g("kernel"), g("stride", 1), max(g("groups", 1), 1)
        if op == "Conv1dF32":
            pads = g("pads"); pb, pa = pads & 0xFFFF, pads >> 16
            dil = max(g("dilation", 1), 1)
            tout = (tin + pb + pa - dil * (kk - 1) - 1) // max(s, 1) + 1
            flops = 2 * b * tout * cout * (cin // groups) * kk
        else:
            crops = g("crops"); cb, ca = crops & 0xFFFF, crops >> 16
            tout = (tin - 1) * s + kk + g("output_padding") - cb - ca
            flops = 2 * b * tin * cin * (cout // groups) * kk
        by = cout * (cin // groups) * kk * 4 + b * tin * cin * 4 + b * tout * cout * 4 * (2 if present(inst, "residual") else 1)
        return by, flops, ("tf32" if g("flags") & (1 << 13) else "fp32")
    if op == "Conv2dF32":
        b, f, w, cin, cout, kk, s = g("batch", 1), g("in_frames"), g("in_width"), g("in_channels"), g("out_channels"), g("kernel"), max(g("stride", 1), 1)
        of, ow = (f + g("pad_before") + g("pad_after") - kk) // s + 1, (w + g("pad_before") + g("pad_after") - kk) // s + 1
        depthwise = g("flags") & 1
        flops = 2 * max(b, 1) * of * ow * cout * (1 if depthwise else cin) * kk * kk
        return cout * cin * kk * kk * 4 + max(b, 1) * (f * w * cin + of * ow * cout) * 4, flops, "fp32"
    if op in ("LayerNormF32",):
        return g("rows") * g("feat") * 4 * 2, 0, "fp32"
    if op == "AttentionF32":
        b, q, kv, h, hw = g("batch", 1), g("q_rows"), g("kv_rows"), g("heads"), g("head_width")
        return b * (q * 2 + kv * 2) * h * hw * 4, 4 * b * h * q * kv * hw, "fp32"
    if op == "GroupedAttentionF32":
        rows, width, grp = g("rows"), g("width"), g("group_rows")
        return rows * width * 4 * 4, 4 * rows * grp * width, "fp32"
    if op == "UnaryF32":
        return g("rows") * g("width") * 4 * 2, 0, "fp32"
    if op == "BinaryF32":
        return g("items", 1) * g("rows") * g("width") * 4 * 3, 0, "fp32"
    if op == "CopyColsF32":
        return g("items", 1) * g("rows") * g("cols") * 4 * 2, 0, "fp32"
    if op == "GatherRowsF32":
        return g("rows") * g("width") * 4 * (3 if g("flags") & 2 else 2), 0, "fp32"
    if op == "RandF32":
        return g("items", 1) * g("rows") * g("width") * 4, 0, "fp32"
    if op == "CumSumF64":
        return g("items", 1) * g("rows") * g("width") * 4 * 2, 0, "fp32"
    if op in ("ScaledAddF32", "Residual"):
        return g("n") * 4 * 3, 0, "fp32"
    if op == "PackNcfwRowsF32":
        return g("rows") * g("channels") * g("frames") * 4 * 2, 0, "fp32"
    if op == "Glu":
        return g("n") * 2 * 3, 0, "fp32"
    return 0, 0, "fp32"


def roof_us(by, fl, prec):
    return max(by / BW, fl / PEAK[prec]) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("disasm")
    ap.add_argument("program", type=int)
    ap.add_argument("measure")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = load(a.disasm)
    rows = [json.loads(l) for l in open(a.measure) if l.startswith("{")]
    if rows and "role" in rows[0]:
        # --each: one program per op; programs indexed by the measured program numbers.
        insts = [(d["programs"][r["program"]]["insts"][0], r["us"]) for r in rows]
        skeleton, total = 0.0, sum(u for _, u in insts)
        ops = [(inst, us) for inst, us in insts]
    else:
        prog = d["programs"][a.program]["insts"]
        key = "us" if "us" in rows[0] else "ms"
        scale = 1.0 if key == "us" else 1000.0
        t = {r["cap"]: r[key] * scale for r in rows}
        skeleton = t.get(0, 0.0)
        total = t.get(-1, t.get(len(prog), max(t.values())))
        ops = [(inst, max(t.get(n + 1, 0) - t.get(n, 0), 0.0)) for n, inst in enumerate(prog) if n + 1 in t and n in t]
    per = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    out = []
    for inst, us in ops:
        by, fl, prec = cost(inst, a.ctx)
        r = roof_us(by, fl, prec)
        kind = inst["op_name"] + ("" if prec == "fp32" or fl == 0 else f"[{prec}]")
        p = per[kind]
        p[0] += 1; p[1] += us; p[2] += r; p[3] += by; p[4] += fl
        out.append(dict(idx=inst["idx"], op=inst["op_name"], us=us, roof_us=r, bytes=by, flops=fl, prec=prec))
    roof_total = sum(o["roof_us"] for o in out)
    print(f"## {a.label}: measured {total:.0f} us (skeleton {skeleton:.0f} us), roofline {roof_total:.0f} us "
          f"-> {100 * roof_total / max(total, 1e-9):.1f}% of roofline")
    print(f"{'op':28s} {'n':>4s} {'us':>9s} {'roof_us':>9s} {'%roof':>6s} {'GB/s':>7s} {'TFLOP/s':>8s} {'%time':>6s}")
    for k, (n, us, r, by, fl) in sorted(per.items(), key=lambda x: -x[1][1]):
        print(f"{k:28s} {n:4d} {us:9.1f} {r:9.1f} {100 * r / max(us, 1e-9):6.1f} {by / max(us, 1e-9) / 1e3:7.0f} "
              f"{fl / max(us, 1e-9) / 1e6:8.2f} {100 * us / max(total, 1e-9):6.1f}")
    if a.json:
        json.dump(dict(label=a.label, total_us=total, skeleton_us=skeleton, roof_us=roof_total, ops=out), open(a.json, "w"))


if __name__ == "__main__":
    main()
