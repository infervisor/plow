#!/usr/bin/env python3
# dsa_midctx_report.py — consolidate the DSA mid-ctx experiment sweeps into a markdown + JSON
# report and COMPUTE the dense/gather crossover + suggested emit-gate per TP degree.
#
# Reads TSV rows on stdin, one measurement per line:
#     <tp>\t<variant>\t<ctx>\t<tpot_ms>
# where variant is one of: dense | gather_after | gather_before  (ctx an integer, tpot ms/tok).
#
# Emits:
#   - a per-TP markdown table (dense vs gather-before vs gather-after, with dense/gather speedup),
#   - the CROSSOVER ctx (where dense tpot first meets the ~flat gather tpot; linear-interpolated,
#     extrapolated past the sweep with the dense linear fit when dense never catches up in range),
#   - a suggested emit gate `ctx > <floor-to-8k(crossover)>` per TP,
#   - the same as JSON (--json PATH).
#
#   ... | python3 scripts/dsa_midctx_report.py --md results.md --json results.json
import sys, json, argparse
from statistics import median

def crossover(dense, gather_flat):
    """ctx where dense(ctx) == gather_flat. dense: sorted [(ctx,tpot)]. Linear interp; extrapolate
    with the end slope if dense never reaches gather_flat within the sampled range."""
    if not dense:
        return None, "no-dense"
    d = sorted(dense)
    # bracketing pair where dense crosses gather_flat
    for (x0, y0), (x1, y1) in zip(d, d[1:]):
        if (y0 - gather_flat) <= 0 <= (y1 - gather_flat) or (y0 - gather_flat) >= 0 >= (y1 - gather_flat):
            if y1 == y0:
                return x0, "interp"
            xc = x0 + (gather_flat - y0) * (x1 - x0) / (y1 - y0)
            return int(round(xc)), "interp"
    # no bracket: extrapolate from the last segment slope
    (x0, y0), (x1, y1) = d[-2], d[-1] if len(d) >= 2 else (d[0], d[0])
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
    if slope <= 0:
        return None, "flat-or-falling"
    xc = x1 + (gather_flat - y1) / slope
    tag = "extrap>range" if xc > d[-1][0] else "extrap<range"
    return int(round(xc)), tag

def k(x):
    return f"{x/1024:.0f}k" if x >= 1024 else str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--title", default="DSA mid-ctx experiments")
    args = ap.parse_args()

    # data[tp][variant] = {ctx: tpot}
    data = {}
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        tp, var, ctx, tpot = parts[0], parts[1], parts[2], parts[3]
        try:
            tp, ctx, tpot = int(tp), int(ctx), float(tpot)
        except ValueError:
            continue
        data.setdefault(tp, {}).setdefault(var, {})[ctx] = tpot

    out = {"title": args.title, "by_tp": {}}
    md = [f"# {args.title}", ""]
    for tp in sorted(data):
        vs = data[tp]
        ctxs = sorted({c for v in vs.values() for c in v})
        md += [f"## TP{tp}", "", "| ctx | dense | gather-before | gather-after | dense/gather-after |",
               "|-----|-------|---------------|--------------|--------------------|"]
        for c in ctxs:
            de = vs.get("dense", {}).get(c)
            gb = vs.get("gather_before", {}).get(c)
            ga = vs.get("gather_after", {}).get(c)
            spd = f"{de/ga:.2f}x" if de and ga else "-"
            md.append(f"| {k(c)} | {de or '-'} | {gb or '-'} | {ga or '-'} | {spd} |")
        md.append("")

        rec = {"rows": {k(c): {v: vs.get(v, {}).get(c) for v in
               ("dense", "gather_before", "gather_after")} for c in ctxs}}
        dense = list(vs.get("dense", {}).items())
        for gv in ("gather_after", "gather_before"):
            g = vs.get(gv, {})
            if dense and g:
                flat = median(g.values())
                xc, tag = crossover(dense, flat)
                rec[gv + "_flat_ms"] = round(flat, 2)
                rec[gv + "_crossover_ctx"] = xc
                rec[gv + "_crossover_tag"] = tag
                if gv == "gather_after" and xc:
                    gate = (xc // 8192) * 8192  # floor to an 8k boundary; emit gather above it
                    rec["suggested_gate"] = gate
                    md.append(f"- **{gv} crossover ≈ {k(xc)}** ({tag}); gather flat ~{flat:.1f} ms.")
                    md.append(f"  Suggested emit gate: `ctx > {gate}`  ({k(gate)}).")
                elif xc:
                    md.append(f"- {gv} crossover ≈ {k(xc)} ({tag}); gather flat ~{flat:.1f} ms.")
        md.append("")
        out["by_tp"][tp] = rec

    text = "\n".join(md)
    if args.md:
        open(args.md, "w").write(text + "\n")
    if args.json:
        open(args.json, "w").write(json.dumps(out, indent=2) + "\n")
    print(text)

if __name__ == "__main__":
    main()
