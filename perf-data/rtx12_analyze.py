#!/usr/bin/env python3
"""rtx12_analyze.py — turn the RTX-12 packing bench artifacts into a combined
JSON + markdown. Per workload cell it slices the server.log by the line-count
brackets recorded in brackets.tsv, builds the R-per-launch histogram from the
PACKLOG lines, diffs the cumulative PACKLOG WALL counters for the prefill/decode
wall-time split, and joins the inference-benchmarker results JSON (matched by
run_id RTX12-<label>) for tok/s, ITL p99, TTFT p99.

Usage: rtx12_analyze.py <mode-dir> [<mode-dir> ...]
Writes: perf-data/rtx12-packing-baseline.{json,md}
"""
import json, os, re, sys, glob, statistics

PACK_RE = re.compile(r"PACKLOG R=(\d+) rows=(\d+) bucket=(\d+) chunks=\[([0-9,]*)\]")
WALL_RE = re.compile(
    r"PACKLOG WALL prefill_ns=(\d+) decode_ns=(\d+) prefill_ticks=(\d+) decode_ticks=(\d+) ticks=(\d+)")


def parse_log(path):
    """Return (packs, walls). packs: list of (lineno,R,rows,bucket,[chunks]).
    walls: list of (lineno, prefill_ns, decode_ns, pf_ticks, dec_ticks, ticks)."""
    packs, walls = [], []
    with open(path, errors="replace") as f:
        for i, line in enumerate(f, 1):
            m = PACK_RE.search(line)
            if m:
                chunks = [int(x) for x in m.group(4).split(",") if x != ""]
                packs.append((i, int(m.group(1)), int(m.group(2)), int(m.group(3)), chunks))
                continue
            w = WALL_RE.search(line)
            if w:
                walls.append((i,) + tuple(int(w.group(k)) for k in range(1, 6)))
    return packs, walls


def wall_at(walls, lineno):
    """Cumulative wall counters at the last WALL line <= lineno (or zeros)."""
    prev = (0, 0, 0, 0, 0, 0)
    for w in walls:
        if w[0] <= lineno:
            prev = w
        else:
            break
    return prev  # (lineno, pf_ns, dec_ns, pf_ticks, dec_ticks, ticks)


def load_ib_results(mode_dir):
    """Index ib result files by run_id -> the non-warmup result dict."""
    idx = {}
    for fp in glob.glob(os.path.join(mode_dir, "results", "*.json")):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        rid = d.get("config", {}).get("run_id")
        if not rid:
            continue
        # pick the measured (non-warmup) result; keep the last if several.
        for r in d.get("results", []):
            if r.get("id") != "warmup":
                idx[rid] = {"file": os.path.basename(fp), "cfg": d.get("config", {}), "r": r}
    return idx


def summarize_cell(label, l0, l1, kind, popts, packs, walls, ib):
    sub = [p for p in packs if l0 < p[0] <= l1]
    hist = {}
    rows_by_R = {}
    all_chunks = []
    total_rows = 0
    for (_, R, rows, bucket, chunks) in sub:
        hist[R] = hist.get(R, 0) + 1
        rows_by_R[R] = rows_by_R.get(R, 0) + rows
        all_chunks.extend(chunks)
        total_rows += rows
    launches = len(sub)
    r1 = hist.get(1, 0)
    rge2 = launches - r1
    # wall split over the cell
    w0 = wall_at(walls, l0)
    w1 = wall_at(walls, l1)
    d_pf_ns = w1[1] - w0[1]
    d_dec_ns = w1[2] - w0[2]
    d_pf_ticks = w1[3] - w0[3]
    d_dec_ticks = w1[4] - w0[4]
    d_ticks = w1[5] - w0[5]
    wall_tot = d_pf_ns + d_dec_ns
    pf_frac = (d_pf_ns / wall_tot) if wall_tot else None

    cell = {
        "label": label, "kind": kind, "prompt_options": popts,
        "log_lines": [l0, l1],
        "packing": {
            "launches": launches,
            "R_histogram": {str(k): hist[k] for k in sorted(hist)},
            "frac_R1": (r1 / launches) if launches else None,
            "frac_Rge2": (rge2 / launches) if launches else None,
            "rows_by_R": {str(k): rows_by_R[k] for k in sorted(rows_by_R)},
            "total_packed_rows": total_rows,
            "mean_R": (sum(k * v for k, v in hist.items()) / launches) if launches else None,
            "max_R": max(hist) if hist else 0,
            "chunk_rows_stats": chunk_stats(all_chunks),
        },
        "walltime": {
            "prefill_ns": d_pf_ns, "decode_ns": d_dec_ns,
            "prefill_ticks": d_pf_ticks, "decode_ticks": d_dec_ticks,
            "total_ticks": d_ticks,
            "prefill_frac": pf_frac,
        },
    }
    if ib:
        r = ib["r"]
        cell["serving"] = {
            "tok_s": r.get("token_throughput_secs"),
            "itl_p99_ms": r.get("inter_token_latency_ms", {}).get("p99"),
            "itl_p50_ms": r.get("inter_token_latency_ms", {}).get("p50"),
            "ttft_p99_ms": r.get("time_to_first_token_ms", {}).get("p99"),
            "ttft_p50_ms": r.get("time_to_first_token_ms", {}).get("p50"),
            "e2e_p99_ms": r.get("e2e_latency_ms", {}).get("p99"),
            "total_requests": r.get("total_requests"),
            "successful": r.get("successful_requests"),
            "failed": r.get("failed_requests"),
            "req_rate": r.get("request_rate"),
            "result_file": ib["file"],
        }
    else:
        cell["serving"] = None
    return cell


def chunk_stats(ch):
    if not ch:
        return None
    ch = sorted(ch)
    n = len(ch)
    def pct(p):
        return ch[min(n - 1, int(p * n))]
    return {
        "n": n, "min": ch[0], "max": ch[-1],
        "median": statistics.median(ch),
        "mean": round(statistics.fmean(ch), 1),
        "p10": pct(0.10), "p90": pct(0.90),
        "frac_le512": sum(1 for x in ch if x <= 512) / n,
        "frac_ge2048": sum(1 for x in ch if x >= 2048) / n,
    }


def main(mode_dirs):
    cells = []
    for md in mode_dirs:
        srvlog = os.path.join(md, "server.log")
        brackets = os.path.join(md, "brackets.tsv")
        if not (os.path.exists(srvlog) and os.path.exists(brackets)):
            print(f"skip {md}: missing artifacts", file=sys.stderr)
            continue
        packs, walls = parse_log(srvlog)
        ibidx = load_ib_results(md)
        for line in open(brackets):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            label, l0, l1, kind, popts = parts[0], int(parts[1]), int(parts[2]), parts[3], parts[4]
            ib = ibidx.get(f"RTX12-{label}")
            cells.append(summarize_cell(label, l0, l1, kind, popts, packs, walls, ib))

    out = {"cells": cells, "note": "R = requests packed per batched-prefill launch; "
           "chunk_rows = per-request rows that launch; walltime split from cumulative "
           "PACKLOG WALL counters diffed over each cell's log-line bracket."}
    root = os.path.dirname(os.path.dirname(os.path.abspath(mode_dirs[0])))  # perf-data/harness/.. -> perf-data
    outjson = os.path.join(root, "..", "rtx12-packing-baseline.json")
    # write json next to perf-data; OUTBASE env overrides the file stem
    # (RTX-12 chunked-packing campaign writes rtx12-chunked-packing.{json,md}).
    perf = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    base = os.environ.get("OUTBASE", "rtx12-packing-baseline")
    outjson = os.path.join(perf, base + ".json")
    json.dump(out, open(outjson, "w"), indent=2)
    print("wrote", outjson)
    write_md(cells, os.path.join(perf, base + ".md"))


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def write_md(cells, path):
    lines = []
    lines.append("| workload | kind | R-hist (R=1 / R≥2) | %R≥2 | meanR | maxR | tok/s | ITL p99 ms | TTFT p99 ms | pf wall% | reqs ok/fail |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        p = c["packing"]; w = c["walltime"]; s = c["serving"] or {}
        hist = p["R_histogram"]
        r1 = hist.get("1", 0)
        rge2 = p["launches"] - r1
        histstr = f"{r1} / {rge2}"
        pctge2 = fmt((p["frac_Rge2"] or 0) * 100, 0)
        pfw = fmt((w["prefill_frac"] or 0) * 100, 0) if w["prefill_frac"] is not None else "-"
        okfail = f"{s.get('successful','-')}/{s.get('failed','-')}" if s else "-"
        lines.append("| {lb} | {kd} | {h} | {g} | {mr} | {mx} | {ts} | {itl} | {ttft} | {pf} | {of} |".format(
            lb=c["label"], kd=c["kind"], h=histstr, g=pctge2,
            mr=fmt(p["mean_R"], 2), mx=p["max_R"],
            ts=fmt(s.get("tok_s"), 1), itl=fmt(s.get("itl_p99_ms"), 1),
            ttft=fmt(s.get("ttft_p99_ms"), 0), pf=pfw, of=okfail))
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:])
