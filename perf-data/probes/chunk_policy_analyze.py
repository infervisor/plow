#!/usr/bin/env python3
"""Reduce the chunk-policy battery to the tables the decision needs.

TTFT: per-cell means, control spread, and the arm deltas -- reported ONLY at the
lengths where the plan actually differs, since a cell whose plan is unchanged is
a null by construction and its delta is a noise estimate.

Identity: first differing CHARACTER index between every arm pair, per question
and length. That index is the instrument -- "the texts differ" is not a finding,
"they diverge at character 139 of a 900-character answer" is.
"""
import argparse, json, os, sys

LAD = [128, 512, 1024, 2048, 4096, 8192]


def plan(bkt, n, lr, ragged, maxc=8192):
    b = sorted({x for x in bkt if 0 < x <= maxc})
    if n == 0:
        return []
    if ragged:
        mb = b[-1]; out = []; rem = n
        while rem > mb:
            out.append(mb); rem -= mb
        if rem:
            out.append(next(x for x in b if x >= rem))
        return out
    q = b[0]; rows = -(-n // q); INF = float("inf")
    cost = [INF] * (rows + 1); pick = [0] * (rows + 1); cost[0] = 0
    for r in range(1, rows + 1):
        for x in b:
            st = max(x // q, 1); prev = max(r - st, 0)
            if cost[prev] == INF:
                continue
            c = cost[prev] + x + lr
            if c < cost[r]:
                cost[r] = c; pick[r] = x
    out = []; r = rows
    while r > 0:
        x = pick[r]; out.append(x); r = max(r - max(x // q, 1), 0)
    return sorted(out, reverse=True)


ARM_PLAN = {
    "ctrl":    lambda n, lr: plan(LAD, n, 416, False),
    "reprice": lambda n, lr: plan(LAD, n, lr, False),
    "ragged":  lambda n, lr: plan(LAD, n, 416, True),
}


def load(d, mode, arm):
    p = os.path.join(d, f"{mode}_{arm}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--lr", type=int, default=1780)
    ap.add_argument("--arms", default="ctrl,reprice,ragged")
    a = ap.parse_args()
    arms = a.arms.split(",")

    t = {arm: load(a.dir, "ttft", arm) for arm in arms}
    if t.get(arms[0]):
        print("## TTFT (3 interleaved reps, exact token counts)\n")
        hdr = "| tokens | ctrl plan | " + " | ".join(arms) + " | spread(ctrl) | "
        hdr += " | ".join(f"d({x})" for x in arms[1:]) + " |"
        print(hdr); print("|" + "---|" * (2 + len(arms) + len(arms)))
        base = {c["tokens"]: c for c in t[arms[0]]["cells"]}
        for n in sorted(base):
            row = [str(n), str(ARM_PLAN["ctrl"](n, a.lr))]
            means = {}
            for arm in arms:
                cells = {c["tokens"]: c for c in t[arm]["cells"]} if t.get(arm) else {}
                m = cells.get(n, {}).get("mean_ms")
                means[arm] = m
                row.append("%.1f" % m if m else "-")
            row.append("%.2f%%" % base[n]["spread_pct"])
            for arm in arms[1:]:
                if means.get(arm) and means[arms[0]]:
                    d = means[arm] - means[arms[0]]
                    same = ARM_PLAN[arm](n, a.lr) == ARM_PLAN["ctrl"](n, a.lr)
                    row.append(("%+.1f" % d) + (" (same plan)" if same else ""))
                else:
                    row.append("-")
            print("| " + " | ".join(row) + " |")
        print()

    for mode in ("ident", "facts"):
        recs = {arm: load(a.dir, mode, arm) for arm in arms}
        if not recs.get(arms[0]):
            continue
        print(f"## {mode}: cross-arm character identity\n")
        idx = {arm: {(c["q"], c["tokens"]): c for c in recs[arm]["cells"]}
               for arm in arms if recs.get(arm)}
        keys = sorted(idx[arms[0]])
        pairs = [(arms[i], arms[j]) for i in range(len(arms)) for j in range(i + 1, len(arms))]
        print("| q | tokens | plans differ | " + " | ".join(f"{x}~{y}" for x, y in pairs) + " | len |")
        print("|" + "---|" * (4 + len(pairs)))
        tally = {p: [0, 0] for p in pairs}
        depths = {p: [] for p in pairs}
        for q, n in keys:
            cells = {arm: idx[arm].get((q, n)) for arm in arms if arm in idx}
            row = [q, str(n)]
            plans = {arm: ARM_PLAN[arm](n, a.lr) for arm in arms}
            row.append("yes" if len(set(map(str, plans.values()))) > 1 else "no")
            L = len(cells[arms[0]]["text"]) if cells.get(arms[0]) else 0
            for p in pairs:
                x, y = p
                if not (cells.get(x) and cells.get(y)):
                    row.append("-"); continue
                d = first_diff(cells[x]["text"], cells[y]["text"])
                if d is None:
                    row.append("IDENTICAL"); tally[p][0] += 1
                else:
                    row.append("char %d" % d); tally[p][1] += 1; depths[p].append((d, L))
            row.append(str(L))
            print("| " + " | ".join(row) + " |")
        print()
        for p in pairs:
            ident, diff = tally[p]
            line = "  %s vs %s: %d/%d identical" % (p[0], p[1], ident, ident + diff)
            if depths[p]:
                dd = [d for d, L in depths[p]]
                ff = [100.0 * d / L for d, L in depths[p] if L]
                line += "; divergence at char median %d (min %d, max %d) = %.0f%% into the answer" % (
                    sorted(dd)[len(dd) // 2], min(dd), max(dd), sum(ff) / len(ff))
            print(line)
        if mode == "facts":
            print()
            for arm in arms:
                if not recs.get(arm):
                    continue
                ok = sum(1 for c in recs[arm]["cells"] if c.get("needle_present"))
                print("  %s: needle present in %d/%d answers" % (arm, ok, len(recs[arm]["cells"])))
        print()


if __name__ == "__main__":
    main()
