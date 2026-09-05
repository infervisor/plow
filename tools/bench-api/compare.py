#!/usr/bin/env python3
"""Side-by-side markdown comparison of two bench.py results.json files.

    compare.py A.json B.json [--labels plowrt,vllm] [--stat p50|mean|p90] [--samples N]

Ratios are B/A: <1 means B has lower latency (TTFT/TPOT) or lower throughput.
"""
import argparse
import json


def load(path):
    with open(path) as f:
        d = json.load(f)
    runs = {}
    for r in d["runs"]:
        runs[(r["workload"], str(r["concurrency"]))] = r
    return d, runs


def fmt_ms(x):
    return "-" if x is None else "%.0f" % (x * 1000)


def fmt(x, nd=1):
    return "-" if x is None else ("%%.%df" % nd) % x


def ratio(a, b):
    if a is None or b is None or a == 0:
        return "-"
    return "%.2fx" % (b / a)


def conc_key(c):
    try:
        return (0, int(c))
    except ValueError:
        return (1, c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", default="A,B", help="comma-separated names for the two files")
    ap.add_argument("--stat", default="p50", choices=["mean", "p50", "p90"], help="latency statistic to compare")
    ap.add_argument("--samples", type=int, default=0, help="print N sample outputs per cell side by side")
    args = ap.parse_args()
    la, lb = [s.strip() for s in args.labels.split(",")]
    da, ra = load(args.a)
    db, rb = load(args.b)
    st = args.stat

    print("# %s vs %s" % (la, lb))
    print()
    for lab, d in ((la, da), (lb, db)):
        m = d["meta"]
        print("- **%s**: `%s` model `%s` (%s, max_tokens=%d, seed=%d, prompt tokens via %s)" % (
            lab, m["base_url"], m["model"], m["timestamp"], m["max_tokens"], m["seed"], m["prompt_token_source"]))
    if da["meta"]["seed"] != db["meta"]["seed"] or da["meta"]["max_tokens"] != db["meta"]["max_tokens"]:
        print("- **warning**: seed or max_tokens differ; requests are not identical")
    print()
    print("Latency stat: **%s**. Ratio = %s / %s (latency: <1 means %s faster; throughput: >1 means %s faster)." % (
        st, lb, la, lb, lb))
    print()
    print("| workload | conc | TTFT %s ms %s | %s | ratio | TPOT %s ms %s | %s | ratio | out tok/s %s | %s | ratio | req/s %s | %s | ratio | err %s/%s |" % (
        st, la, lb, st, la, lb, la, lb, la, lb, la, lb))
    print("|" + "---|" * 15)
    keys = sorted(set(ra) | set(rb), key=lambda k: (k[0], conc_key(k[1])))
    for k in keys:
        A = ra.get(k, {}).get("agg")
        B = rb.get(k, {}).get("agg")
        ga = lambda path: _get(A, path)  # noqa: E731
        gb = lambda path: _get(B, path)  # noqa: E731
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s/%s |" % (
            k[0], k[1],
            fmt_ms(ga(("ttft_s", st))), fmt_ms(gb(("ttft_s", st))), ratio(ga(("ttft_s", st)), gb(("ttft_s", st))),
            fmt_ms(ga(("tpot_s", st))), fmt_ms(gb(("tpot_s", st))), ratio(ga(("tpot_s", st)), gb(("tpot_s", st))),
            fmt(ga(("output_tok_per_s",))), fmt(gb(("output_tok_per_s",))),
            ratio(ga(("output_tok_per_s",)), gb(("output_tok_per_s",))),
            fmt(ga(("req_per_s",)), 2), fmt(gb(("req_per_s",)), 2), ratio(ga(("req_per_s",)), gb(("req_per_s",))),
            fmt(ga(("errors",)), 0), fmt(gb(("errors",)), 0)))
    print()

    # Sanity: identical prompts (server-side token counts) and output agreement.
    mism, ident, total = 0, 0, 0
    for k in keys:
        if k not in ra or k not in rb:
            continue
        bi = {r["index"]: r for r in rb[k]["requests"]}
        for r in ra[k]["requests"]:
            s = bi.get(r["index"])
            if not s:
                continue
            total += 1
            if r["prompt_chars"] != s["prompt_chars"]:
                mism += 1
            if r["text_head"] and r["text_head"] == s["text_head"]:
                ident += 1
    if total:
        print("Paired requests: %d; prompt mismatches: %d; identical first-%d-char outputs: %d (%.0f%%)." % (
            total, mism, 80, ident, 100.0 * ident / total))
        print()

    if args.samples:
        print("## Samples")
        print()
        for k in keys:
            if k not in ra or k not in rb:
                continue
            print("**%s @ %s**" % k)
            print()
            bi = {r["index"]: r for r in rb[k]["requests"]}
            for r in ra[k]["requests"][: args.samples]:
                s = bi.get(r["index"])
                print("- #%d %s" % (r["index"], r["kind"]))
                print("  - %s: `%s`" % (la, r["text_head"].replace("`", "'").replace("\n", " ")))
                if s:
                    print("  - %s: `%s`" % (lb, s["text_head"].replace("`", "'").replace("\n", " ")))
            print()


def _get(d, path):
    for p in path:
        if d is None:
            return None
        d = d.get(p)
    return d


if __name__ == "__main__":
    main()
