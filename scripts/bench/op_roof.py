#!/usr/bin/env python3
"""op_roof.py <disasm.txt> --ctx N [--sweep B=sweep.jsonl ...] [--segtime server.log --segtime-rows T]

Per-op roofline for a dense packet (Llama / Gemma / Qwen shapes; GLM/MLA/MoE packets: see
scripts/campaign/op_roofline.py), as markdown: bytes and FLOPs per op, the floor max(bytes/BW,
FLOPs/peak), the measured time and the floor as % of it.

  disasm      `plowrt disasm <model.pkt>` output (every program; prefill rungs first, then decode)
  --sweep     decode rung B measured with `step_bench <assets> B <ctx> 10 --warmup 4 --sweep 0..<n>`
              (lines {"cap":k,"ms":x}; cap k runs instructions 0..k-1, so inst i costs ms[i+1]-ms[i];
              cap 0 is the interpreter skeleton, cap -1 the full step). Raw step_bench logs work.
  --segtime   prefill measured with PLOW_PF_SEG_TIME=1 (segment-site lines, per chunk); the chunk
              rows are --segtime-rows. SEG_TIME drains per segment: shares, not latency.
  no measurement: the floor of every program (one row per prefill/decode rung).

Weights and activations are priced at --wbytes (2 = bf16), KV at --kvbytes; attention reads
min(ctx, window) keys per row. Ceilings: --gpu (scripts/campaign/roofline.py registry), --bw /
--tflops override (use a measured BW: ~3.0 TB/s streams on H100, not the 3.35 datasheet).
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "campaign"))
from roofline import lookup_gpu  # noqa: E402

LINE = re.compile(r"^#(\d+)\s+(\w+)\s+b=(\d+)\s*(.*)$")
PROGRAM = re.compile(r"^===== program T=(\d+)")
WEIGHT = re.compile(r"(?:W|B|W_gate|W_q|table)<-(\S+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SITE = re.compile(r'sites="([^"]+)" count=(\d+) elapsed_ms="([0-9.]+)"')


def parse(text):
    """[(phase, rows, [(idx, op, name, params)])] in disasm order."""
    progs, cur, phase = [], None, "prefill"
    for line in text.splitlines():
        m = PROGRAM.match(line)
        if m:
            rows = int(m.group(1))
            if cur is not None and rows < cur[1]:
                phase = "decode"  # plowc orders prefill rungs, then decode rungs, each ascending
            cur = (phase, rows, [])
            progs.append(cur)
            continue
        m = LINE.match(line.strip())
        if cur is None or not m:
            continue
        head, _, tail = m.group(4).partition(" | ")
        p = {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+)(?=\s|$)", tail)}
        w = WEIGHT.search(head)
        name = re.sub(r"layers\.\d+\.", "", w.group(1)).replace("model.language_model.", "").replace("model.", "") if w else ""
        cur[2].append((int(m.group(1)), m.group(2), name, p))
    return progs


def cost(op, p, rows, ctx, wb, kvb):
    """(bytes, flops) of one instruction, or None when the op is not priced."""
    g = p.get
    m = g("M", rows)
    if op in ("Gemm", "Gemv", "GemmMed", "GemmSmall", "GemmWide"):
        n, k = g("N"), g("K")
        return wb * n * k + 2 * m * k + 2 * m * n, 2 * m * n * k
    if op == "GemvQkv":
        n, k = g("Nq") + g("Nk") + g("Nv"), g("K")
        return wb * n * k + 2 * m * k + 2 * m * n, 2 * m * n * k
    if op in ("GemvGlu", "GemmGlu"):
        n, k = g("N"), g("K")
        return 2 * wb * n * k + 2 * m * k + 2 * m * n, 4 * m * n * k
    if op == "FlashDecode":
        nb, hq, hk, hd, win = g("n_batch", rows), g("n_head"), g("n_kv_head"), g("hd"), g("window", 0)
        keys = min(ctx, win) if win else ctx
        return nb * keys * hk * hd * 2 * kvb + 4 * nb * hq * hd, 4 * nb * hq * keys * hd
    if op == "FlashPrefill":
        nq, nkv, hq, hk, hd, win = g("n_q"), g("n_kv"), g("n_head"), g("n_kv_head"), g("hd"), g("window", 0)
        keys = g("q_pos0", 0) + nq / 2  # causal: mean keys per query
        keys = min(keys, win) if win else keys
        return 4 * nq * hq * hd + nkv * hk * hd * 2 * kvb, 4 * nq * hq * keys * hd
    if op == "FlashMerge":
        n, hh, ns, hd = g("n_batch", rows), g("n_head"), g("nsplit", 1), g("hd")
        return 4 * n * hh * hd * ns + 2 * n * hh * hd, 0
    if op in ("RmsNorm", "NormResidual", "NormResidualNorm"):
        tensors = {"RmsNorm": 2, "NormResidual": 3, "NormResidualNorm": 4}[op]
        return 2 * tensors * g("rows", rows) * g("feat"), 0
    if op == "GluStrided":
        return 6 * g("rows", rows) * g("width"), 0
    if op in ("Glu", "Residual"):
        return 6 * g("n"), 0
    if op == "HeadNormRope":
        return 4 * g("ntok") * g("nhead") * g("hd"), 0
    if op == "Embed":
        return 4 * g("ntok") * g("hidden"), 0
    if op == "SoftCap":
        return 8 * g("n"), 0
    if op in ("Argmax", "ArgmaxFin"):
        return 4 * g("n", 0), 0
    return None


def price(prog, ctx, wb, kvb, bw, tflops):
    """Per instruction: key, bytes, flops, floor ms, bound."""
    out = {}
    for idx, op, name, p in prog[2]:
        c = cost(op, p, prog[1], ctx, wb, kvb)
        if c is None:
            out[idx] = (f"{op}", None, None, 0.0, "unpriced")
            continue
        mem, mat = c[0] / (bw * 1e6), c[1] / (tflops * 1e9)
        key = f"{op}:{name}" if name else op
        out[idx] = (key, c[0], c[1], max(mem, mat), "matrix" if mat > mem else "hbm")
    return out


def sweep_deltas(path):
    caps = {}
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith('{"cap"'):
            d = json.loads(ln)
            caps[d["cap"]] = d["ms"]
    full = caps.pop(-1, None)
    ks = sorted(caps)
    return caps[ks[0]], {k - 1: caps[k] - caps[k - 1] for k in ks[1:]}, full


def table(title, rows, measured_total, extra_lines, top):
    """rows: key -> [n, bytes, flops, floor, measured|None, bound]"""
    out = [f"### {title}", ""]
    floor = sum(r[3] for r in rows.values())
    have = any(r[4] is not None for r in rows.values())
    out.append(f"floor {floor:.3f} ms" + (f", measured {measured_total:.3f} ms = {100 * floor / measured_total:.0f}% of roofline"
                                          if measured_total else ""))
    out += extra_lines + [""]
    out.append("| op | n | MB | GFLOP | bound | floor ms |" + (" measured ms | % roof |" if have else ""))
    out.append("|---|---:|---:|---:|---|---:|" + ("---:|---:|" if have else ""))
    order = sorted(rows.items(), key=lambda kv: -(kv[1][4] if have and kv[1][4] is not None else kv[1][3]))
    for key, (n, by, fl, fm, ms, bound) in order[:top]:
        line = (f"| `{key}` | {n} | {'-' if by is None else f'{by / 1e6:.1f}'} | "
                f"{'-' if fl is None else f'{fl / 1e9:.2f}'} | {bound} | {fm:.3f} |")
        if have:
            line += f" {'-' if ms is None else f'{ms:.3f}'} | {f'{100 * fm / ms:.0f}%' if ms and ms > 0 and fm else '-'} |"
        out.append(line)
    if len(order) > top:
        rest = order[top:]
        out.append(f"| (other {len(rest)} ops) | {sum(r[0] for _, r in rest)} | | | | {sum(r[3] for _, r in rest):.3f} |"
                   + (f" {sum(r[4] or 0 for _, r in rest):.3f} | |" if have else ""))
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("disasm")
    ap.add_argument("--ctx", type=int, required=True, help="decode live context per row")
    ap.add_argument("--sweep", action="append", default=[], metavar="B=FILE")
    ap.add_argument("--segtime")
    ap.add_argument("--segtime-rows", type=int)
    ap.add_argument("--gpu", default="h100")
    ap.add_argument("--bw", type=float, help="GB/s (default: the --gpu registry)")
    ap.add_argument("--tflops", type=float, help="dense matrix TFLOP/s at the weight dtype")
    ap.add_argument("--wbytes", type=float, default=2.0)
    ap.add_argument("--kvbytes", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    hw = lookup_gpu(a.gpu)
    bw, tf = a.bw or hw.bandwidth_for_bound_gbps, a.tflops or hw.bf16_tflops_dense
    progs = parse(open(a.disasm).read())
    if not progs:
        sys.exit(f"{a.disasm}: no programs (expected `plowrt disasm` output)")
    print(f"# Op roofline: {os.path.basename(a.disasm)}, ctx {a.ctx}, {hw.name} {bw:.0f} GB/s, {tf:.0f} TFLOP/s\n")
    find = lambda phase, rows: next((p for p in progs if p[0] == phase and p[1] == rows), None)

    for spec in a.sweep:
        b, path = spec.split("=", 1)
        prog = find("decode", int(b))
        if prog is None:
            sys.exit(f"no decode program T={b} in {a.disasm}")
        pr = price(prog, a.ctx, a.wbytes, a.kvbytes, bw, tf)
        skel, delta, full = sweep_deltas(path)
        rows = collections.OrderedDict()
        for idx, (key, by, fl, fm, bound) in pr.items():
            r = rows.setdefault(key, [0, 0, 0, 0.0, 0.0, bound])
            r[0] += 1
            r[1] = None if by is None else r[1] + by
            r[2] = None if fl is None else r[2] + fl
            r[3] += fm
            r[4] += delta.get(idx, 0.0)
        cls = collections.Counter()
        for key, r in rows.items():
            c = ("lm_head" if "embed_tokens" in key and key.startswith(("Gemv", "Gemm")) else
                 "projections" if key.startswith(("Gemv", "Gemm")) else
                 "attention" if key.startswith("Flash") else "other")
            cls[c] += r[4]
        extra = [f"skeleton (cap 0) {skel:.3f} ms; by class: " +
                 ", ".join(f"{k} {v:.3f}" for k, v in sorted(cls.items(), key=lambda kv: -kv[1]))]
        print(table(f"decode B={b} ctx {a.ctx} (step_bench sweep)", rows, full, extra, a.top))

    if a.segtime:
        if not a.segtime_rows:
            sys.exit("--segtime needs --segtime-rows (the chunk's prefill rows)")
        prog = find("prefill", a.segtime_rows)
        if prog is None:
            sys.exit(f"no prefill program T={a.segtime_rows} in {a.disasm}")
        pr = price(prog, a.ctx, a.wbytes, a.kvbytes, bw, tf)
        rows, chunks = collections.OrderedDict(), 0
        for ln in open(a.segtime, errors="replace"):
            ln = ANSI.sub("", ln)
            if "seg-class wall time (chunk)" in ln:
                chunks += 1
            m = SITE.search(ln)
            if not m:
                continue
            pcs = [int(x.split(":")[0][2:]) for x in m.group(1).split("+")]
            key = "+".join(sorted({pr[i][0] for i in pcs if i in pr}))
            r = rows.setdefault(key, [0, 0, 0, 0.0, 0.0, "hbm"])
            r[4] += float(m.group(3))
            if chunks <= 1:
                r[0] += len(pcs)
                for i in pcs:
                    _, by, fl, fm, bound = pr.get(i, ("", None, None, 0.0, "unpriced"))
                    r[1] = None if by is None or r[1] is None else r[1] + by
                    r[2] = None if fl is None or r[2] is None else r[2] + fl
                    r[3] += fm
                    r[5] = bound if fm else r[5]
        for r in rows.values():
            r[4] /= max(chunks, 1)
        tot = sum(r[4] for r in rows.values())
        print(table(f"prefill T={a.segtime_rows} (PLOW_PF_SEG_TIME, mean of {chunks} chunk(s); SEG_TIME drains per "
                    "segment: attribution, not latency)", rows, tot, [], a.top))

    if not a.sweep and not a.segtime:
        print("| program | rows | floor ms | floor ms/row | top op (floor) |")
        print("|---|---:|---:|---:|---|")
        for prog in progs:
            pr = price(prog, a.ctx, a.wbytes, a.kvbytes, bw, tf)
            by = collections.Counter()
            for key, _, _, fm, _ in pr.values():
                by[key] += fm
            f = sum(by.values())
            top = by.most_common(1)[0]
            print(f"| {prog[0]} | {prog[1]} | {f:.3f} | {f / prog[1]:.4f} | `{top[0]}` {top[1]:.3f} |")


if __name__ == "__main__":
    main()
