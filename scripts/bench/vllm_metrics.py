#!/usr/bin/env python3
"""vLLM engine accounting from its Prometheus /metrics, per bench cell.

  vllm_metrics.py poll <port> <out.tsv>     poll /metrics every 50 ms until killed (pb_metrics_start)
  vllm_metrics.py cells <resdir> [--max-prefix-hit 0.05] [--json]
      per cell of <resdir>/cells.log (pb_cell markers) over <resdir>/metrics.tsv: active wall, engine
      steps, ms/step, tokens/step, prompt/generated tokens, prefix-cache hit rate, running/waiting,
      engine time in poll intervals that scheduled prompt tokens (mixed) vs not (decode), and the
      decode-only phase (after the last prompt token was scheduled) ms/step.
      Exits 1 when any cell's prefix-cache hit rate exceeds --max-prefix-hit: the reference then
      skipped prefill the other side computed (reused prompts; see pb_bench's per-cell seed).
"""
import argparse
import json
import re
import sys
import time
import urllib.request

KEYS = ["num_requests_running", "num_requests_waiting", "iteration_tokens_total_count",
        "iteration_tokens_total_sum", "prompt_tokens_total", "generation_tokens_total",
        "prefix_cache_hits_total", "prefix_cache_queries_total"]


def poll(port, out):
    with open(out, "w") as f:
        f.write("t\t" + "\t".join(KEYS) + "\n")
        while True:
            try:
                txt = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2).read().decode()
            except Exception:  # noqa: BLE001 - server starting or gone; keep polling until killed
                time.sleep(0.2)
                continue
            vals = [sum(float(m.group(2)) for m in re.finditer(
                r"^vllm:" + k + r"(\{[^}]*\})? ([0-9.eE+-]+)$", txt, re.M)) for k in KEYS]
            f.write(f"{time.time():.3f}\t" + "\t".join(f"{v:g}" for v in vals) + "\n")
            f.flush()
            time.sleep(0.05)


def load(resdir):
    with open(f"{resdir}/metrics.tsv") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, map(float, ln.split("\t")))) for ln in f if ln.strip()]
    cells = {}
    for ln in open(f"{resdir}/cells.log"):
        k, tag, t = ln.split()
        cells.setdefault(tag, {})[k] = float(t)
    return rows, cells


def cell_stats(rows, c):
    w = [r for r in rows if c["CELL_BEGIN"] <= r["t"] <= c.get("CELL_END", float("inf"))]
    act = [i for i in range(1, len(w)) if w[i]["iteration_tokens_total_count"] > w[i - 1]["iteration_tokens_total_count"]]
    if not act:
        return None
    a, b = w[act[0] - 1], w[act[-1]]
    d = {k: b[k] - a[k] for k in KEYS[2:]}
    steps = d["iteration_tokens_total_count"]
    live = w[act[0]:act[-1] + 1]
    s = dict(active_s=b["t"] - a["t"], steps=steps, ms_per_step=(b["t"] - a["t"]) / steps * 1e3,
             tok_per_step=d["iteration_tokens_total_sum"] / steps, prompt_tokens=d["prompt_tokens_total"],
             gen_tokens=d["generation_tokens_total"],
             prefix_hit=d["prefix_cache_hits_total"] / d["prefix_cache_queries_total"] if d["prefix_cache_queries_total"] else 0.0,
             running=sum(r["num_requests_running"] for r in live) / len(live),
             waiting=sum(r["num_requests_waiting"] for r in live) / len(live))
    # Engine time by poll interval: "mixed" when prompt tokens were scheduled in it, else "decode".
    # 50 ms granularity; an attribution, not a kernel split.
    s["mixed_s"] = s["decode_s"] = 0.0
    for i in range(act[0], act[-1] + 1):
        dt = w[i]["t"] - w[i - 1]["t"]
        s["mixed_s" if w[i]["prompt_tokens_total"] > w[i - 1]["prompt_tokens_total"] else "decode_s"] += dt
    # Decode-only phase: from the poll after the last prompt-token increase to the last active poll.
    grow = [i for i in range(1, len(w)) if w[i]["prompt_tokens_total"] > w[i - 1]["prompt_tokens_total"]]
    j = (grow[-1] + 1) if grow else act[0]
    if j < act[-1]:
        p, q = w[j], w[act[-1] - 1] if act[-1] - 1 > j else w[act[-1]]
        dsteps = q["iteration_tokens_total_count"] - p["iteration_tokens_total_count"]
        if dsteps > 20:
            s["decode_ms_per_step"] = (q["t"] - p["t"]) / dsteps * 1e3
            s["decode_running"] = max(r["num_requests_running"] for r in w[j:act[-1]])
    return s


def cells(resdir):
    rows, cs = load(resdir)
    return {tag: st for tag, c in cs.items() if (st := cell_stats(rows, c))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("poll")
    p.add_argument("port")
    p.add_argument("out")
    c = sp.add_parser("cells")
    c.add_argument("resdir")
    c.add_argument("--max-prefix-hit", type=float, default=0.05)
    c.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.cmd == "poll":
        poll(a.port, a.out)
        return
    res = cells(a.resdir)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"{'cell':10s} {'active s':>8} {'steps':>6} {'ms/step':>8} {'tok/step':>8} {'prompt':>8} {'gen':>7} "
              f"{'pfx hit':>7} {'run':>6} {'wait':>6} {'dec ms/step':>11}")
        for tag, s in res.items():
            dec = s.get("decode_ms_per_step")
            print(f"{tag:10s} {s['active_s']:8.2f} {s['steps']:6.0f} {s['ms_per_step']:8.2f} {s['tok_per_step']:8.1f} "
                  f"{s['prompt_tokens']:8.0f} {s['gen_tokens']:7.0f} {100 * s['prefix_hit']:6.1f}% {s['running']:6.1f} "
                  f"{s['waiting']:6.1f} {'-' if dec is None else f'{dec:11.2f}'}")
    bad = [t for t, s in res.items() if s["prefix_hit"] > a.max_prefix_hit]
    if bad:
        print(f"PREFIX-CACHE HITS > {100 * a.max_prefix_hit:.0f}% in {bad}: the reference reused prompts; "
              "those cells are not a matched comparison", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
