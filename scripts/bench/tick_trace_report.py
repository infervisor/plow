#!/usr/bin/env python3
"""Accounting table from a PLOW_TICK_TRACE file (one JSON line per AMD serve tick).

usage: tick_trace_report.py TRACE [--csv OUT] [--skip-ms MS] [--ticks N]

Times in the trace are us on one monotonic clock. A launch's device interval is enqueue ->
last drain return (`enq`..`dr1`; `dr0` when no enqueue was seen); everything else on the
engine thread is host time: plan (tick start -> first launch), pre (launch start -> enqueue:
prepare, rearm, xctr, enqueue), post (drain end -> launch return: audit, readback, sampling),
emit (launch return -> next launch / tick end: token handling, frontier updates) and the idle
gap between ticks. Requests are keyed by arrival; TTFT here is server-side (arrival -> first
token), decode cadence is the gap between a request's consecutive tokens.
"""
import argparse
import collections
import json
import statistics
import sys


def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    k = min(len(v) - 1, max(0, int(round(p / 100.0 * (len(v) - 1)))))
    return v[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--csv", help="per-tick CSV output")
    ap.add_argument("--skip-ms", type=float, default=0.0, help="drop ticks in the first MS of the trace")
    ap.add_argument("--ticks", type=int, default=0, help="print the first N ticks after the skip")
    ap.add_argument("--cells", help="llm_grid cells.log: keep only ticks inside --cell")
    ap.add_argument("--cell", help="cell tag in --cells (e.g. g128.r1)")
    a = ap.parse_args()

    ticks = []
    with open(a.trace) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ticks.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line
    if not ticks:
        sys.exit("no ticks")
    if a.cells and a.cell:
        win = {}
        for line in open(a.cells):
            p = line.split()
            if len(p) == 3 and p[1] == a.cell:
                win[p[0]] = float(p[2])
        lo, hi = win.get("CELL_BEGIN", 0.0), win.get("CELL_END", float("inf"))
        ticks = [t for t in ticks if lo <= t.get("unix", 0.0) <= hi]
        if not ticks:
            sys.exit(f"no ticks inside {a.cell}")
    ticks.sort(key=lambda t: (t["th"], t["t0"]))
    t_first = min(t["t0"] for t in ticks)
    ticks = [t for t in ticks if t["t0"] - t_first >= a.skip_ms * 1e3]

    tot = collections.Counter()
    shapes = collections.defaultdict(lambda: collections.Counter())
    kinds = collections.defaultdict(lambda: collections.defaultdict(list))
    rows_csv = []
    prev_end = {}
    for t in ticks:
        wall = t["t1"] - t["t0"]
        idle = t["t0"] - prev_end[t["th"]] if t["th"] in prev_end else 0
        prev_end[t["th"]] = t["t1"]
        dev = pre = post = emit = 0
        ls = t["launches"]
        plan = (ls[0]["t0"] - t["t0"]) if ls else wall
        for i, l in enumerate(ls):
            start = l["enq"] or l["dr0"] or l["t0"]
            end = l["dr1"] or l["t1"]
            d = max(0, end - start)
            dev += d
            pre += max(0, start - l["t0"])
            post += max(0, (l["t1"] or end) - end)
            nxt = ls[i + 1]["t0"] if i + 1 < len(ls) else t["t1"]
            emit += max(0, nxt - (l["t1"] or end))
            k = kinds[l["kind"]]
            k["dev"].append(d / 1e3)
            k["pre"].append(max(0, start - l["t0"]) / 1e3)
            k["post"].append(max(0, (l["t1"] or end) - end) / 1e3)
            k["rows"].append(l["rows"])
            k["rung"].append(l["rung"])
            k["dec"].append(l["dec"])
            k["spans"].append(l["spans"])
            k["skew"].append(l["skew"] / 1e3)
            k["slow"].append(l["slow_rank"])
        names = [l["kind"] for l in ls]
        shape = "+".join(names) if names else "empty"
        if t.get("held"):
            shape = "HELD:" + shape
        if len(shape) > 48:
            c = collections.Counter(names)
            shape = ("HELD:" if t.get("held") else "") + "+".join(f"{k}x{n}" for k, n in sorted(c.items()))
        s = shapes[shape]
        s["n"] += 1
        s["wall"] += wall
        s["dev"] += dev
        s["idle"] += idle
        s["dec_tok"] += len(t["dec"])
        s["first"] += len(t["first"])
        s["pf_rows"] += sum(l["rows"] for l in ls)
        s["pad_rows"] += sum(max(0, l["rung"] - l["rows"] - l["dec"]) for l in ls if l["kind"] != "decode")
        for k, v in (("wall", wall), ("idle", idle), ("dev", dev), ("plan", plan), ("pre", pre),
                     ("post", post), ("emit", emit)):
            tot[k] += v
        tot["ticks"] += 1
        tot["dec_tok"] += len(t["dec"])
        tot["first"] += len(t["first"])
        tot["hits"] += t.get("hits", 0)
        tot["held"] += 1 if t.get("held") else 0
        rows_csv.append([t["n"], (t["t0"] - t_first) / 1e3, wall / 1e3, idle / 1e3, dev / 1e3,
                         plan / 1e3, pre / 1e3, post / 1e3, emit / 1e3, t["ready_dec"], t["waiting"],
                         t["waiting_rows"], int(bool(t.get("held"))), t.get("hits", 0), shape,
                         sum(l["rows"] for l in ls), "/".join(str(l["rung"]) for l in ls),
                         len(t["dec"]), len(t["first"])])

    span = (max(t["t1"] for t in ticks) - min(t["t0"] for t in ticks)) / 1e3
    ms = lambda us: us / 1e3
    print(f"# {a.trace}: {tot['ticks']} ticks over {span:.0f} ms, {tot['dec_tok']} decode tokens, "
          f"{tot['first']} first tokens, prefix-hit rows {tot['hits']}, held ticks {tot['held']}")
    print()
    print("## Engine-thread time (ms, % of span)")
    acct = [("device (enqueue -> drain end)", tot["dev"]), ("host plan (tick start -> first launch)", tot["plan"]),
            ("host pre (launch -> enqueue)", tot["pre"]), ("host post (drain end -> return)", tot["post"]),
            ("host emit (return -> next launch/tick end)", tot["emit"]), ("idle between ticks", tot["idle"])]
    for name, us in acct:
        print(f"| {name:44s} | {ms(us):10.1f} | {100 * ms(us) / span:5.1f}% |")
    resid = span * 1e3 - sum(us for _, us in acct)
    print(f"| {'residual':44s} | {ms(resid):10.1f} | {100 * ms(resid) / span:5.1f}% |")
    print()
    print("## Launches by kind")
    print("| kind | n | rows | rung | dec | spans | dev ms p50 | dev ms mean | us/row (rows+dec) | pre ms | post ms | skew ms p50/max | slow rank mode |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for kind, k in sorted(kinds.items()):
        n = len(k["dev"])
        rows = statistics.mean(k["rows"])
        dec = statistics.mean(k["dec"])
        devm = statistics.mean(k["dev"])
        per = 1e3 * devm / max(1e-9, rows + dec)
        slow = collections.Counter(k["slow"]).most_common(1)[0][0]
        print(f"| {kind} | {n} | {rows:.0f} | {statistics.mean(k['rung']):.0f} | {dec:.1f} | "
              f"{statistics.mean(k['spans']):.1f} | {pct(k['dev'], 50):.2f} | {devm:.2f} | {per:.1f} | "
              f"{statistics.mean(k['pre']):.2f} | {statistics.mean(k['post']):.2f} | "
              f"{pct(k['skew'], 50):.2f}/{max(k['skew']):.2f} | {slow} |")
    print()
    print("## Tick shapes")
    print("| shape | ticks | share | wall ms | device ms | host ms | idle before ms | decode tok/tick | prefill rows/tick | pad rows/tick |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for shape, s in sorted(shapes.items(), key=lambda kv: -kv[1]["wall"]):
        n = s["n"]
        print(f"| {shape} | {n} | {100 * s['wall'] / max(1, tot['wall']):.1f}% | {ms(s['wall']) / n:.1f} | "
              f"{ms(s['dev']) / n:.1f} | {ms(s['wall'] - s['dev']) / n:.1f} | {ms(s['idle']) / n:.2f} | "
              f"{s['dec_tok'] / n:.1f} | {s['pf_rows'] / n:.0f} | {s['pad_rows'] / n:.0f} |")
    print()

    # Per-request cadence.
    toks = collections.defaultdict(list)
    for t in ticks:
        for key, ts in t["first"]:
            toks[key].append((ts, True))
        for key, ts in t["dec"]:
            toks[key].append((ts, False))
    gaps, ttft, ntok = [], [], []
    for key, v in toks.items():
        v.sort()
        if v and v[0][1]:
            ttft.append(v[0][0] / 1e3 - key / 1e6)
        ntok.append(len(v))
        for (a_t, _), (b_t, first) in zip(v, v[1:]):
            if not first:
                gaps.append((b_t - a_t) / 1e3)
    print("## Per-request cadence (server side)")
    print(f"requests {len(toks)}, tokens/request mean {statistics.mean(ntok) if ntok else 0:.1f}")
    if gaps:
        print(f"inter-token gap ms: p50 {pct(gaps, 50):.1f} p90 {pct(gaps, 90):.1f} p99 {pct(gaps, 99):.1f} "
              f"mean {statistics.mean(gaps):.1f}")
    ttft = [x for x in ttft if x >= 0]
    if ttft:
        print(f"arrival -> first token ms: p50 {pct(ttft, 50):.0f} p90 {pct(ttft, 90):.0f} p99 {pct(ttft, 99):.0f}")
    if a.ticks:
        print()
        print("## First ticks")
        print("n t_ms wall idle dev plan pre post emit ready_dec waiting waiting_rows held hits shape rows rungs dec first")
        for r in rows_csv[: a.ticks]:
            print(" ".join(f"{x:.1f}" if isinstance(x, float) else str(x) for x in r))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "t_ms", "wall_ms", "idle_ms", "dev_ms", "plan_ms", "pre_ms", "post_ms", "emit_ms",
                        "ready_dec", "waiting", "waiting_rows", "held", "hits", "shape", "pf_rows", "rungs",
                        "dec_tok", "first_tok"])
            w.writerows(rows_csv)


if __name__ == "__main__":
    main()
