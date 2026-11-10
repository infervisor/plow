#!/usr/bin/env python3
"""Which launch capacities a packet is missing, from a serving plowrt's /metrics.

    bucket_demand.py http://host:8001/metrics            # since process start
    bucket_demand.py before.prom after.prom              # the window between two snapshots
    bucket_demand.py after.prom --model maya2 --family depth --add 3

Reads the `plowrt_bucket_*` series (docs/runtime/host-path.md "Bucket demand"): per model and
launch family the capacities the packet declares, how many frames asked for each live row count,
and what each capacity launched. A frame whose row count has no capacity runs the next larger one
and pays for the padding. For K = 1..--add it prints the K capacities whose addition minimises the
launched rows over the observed demand (exact dynamic program; every frame runs the smallest
capacity holding it), and where neighbouring capacities carry a measured launch time, the time
that saves (cost interpolated linearly between measured capacities).

It only recommends: capacities are added to the packet's ladder at the next emit.
"""
import argparse
import re
import sys
import urllib.request
from collections import defaultdict

PREFIX = "plowrt_bucket_"
COUNTERS = ("demand_total", "launches_total", "padded_rows_total", "launch_seconds_sum",
            "launch_seconds_count", "overflow_total")
LINE = re.compile(r"^plowrt_bucket_(\w+)\{(.*)\}\s+(\S+)$")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def read(source):
    if re.match(r"https?://", source):
        with urllib.request.urlopen(source, timeout=10) as r:
            return r.read().decode()
    with open(source) as f:
        return f.read()


def parse(text, per_engine=False):
    """{(model, family): {series: {key: value}}}; engines of one model sum unless per_engine."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        series, labels, value = m.group(1), dict(LABEL.findall(m.group(2))), float(m.group(3))
        model = labels.get("model_name", "")
        if per_engine:
            model += "#" + labels.get("engine", "0")
        key = labels.get("rows") or labels.get("capacity")
        cell = out[(model, labels.get("family", ""))][series]
        key = int(key) if key is not None else None
        if series in ("capacity_info", "demand_max"):
            cell[key] = max(cell[key], value)
        else:
            cell[key] += value
    return out


def window(first, second):
    """`second - first` on the counters; a counter that went backwards means a restart."""
    for group, series in first.items():
        for name in COUNTERS:
            for key, before in series.get(name, {}).items():
                after = second.get(group, {}).get(name, {})
                if after.get(key, 0.0) < before:
                    raise SystemExit(f"{group}: {name} went backwards between the snapshots "
                                     "(restart?); pass the later snapshot alone")
                if before:
                    after[key] -= before
    return second


def cover(caps, rows):
    return next((c for c in caps if c >= rows), caps[-1])


def launched(caps, demand):
    return sum(n * cover(caps, r) for r, n in demand.items())


def plan(caps, demand, k_max):
    """[(added capacities, launched rows)] for K = 1..k_max: the K capacities to add to `caps`
    that minimise sum(frames * smallest capacity holding the frame). Exact: a best capacity is
    always an observed row count, and between two capacities the cost depends on the upper one
    only, so a shortest path over the sorted candidates with K free stops is optimal."""
    declared = set(caps)
    points = sorted(declared | set(demand))
    # count[i]: frames whose rows are in (points[i-1], points[i]]
    prefix = [0.0]
    for p in points:
        prefix.append(prefix[-1] + demand.get(p, 0.0))
    inf = float("inf")
    # best[i][k]: least launched rows for demand <= points[i] with points[i] a capacity, k added
    best = [[inf] * (k_max + 1) for _ in points]
    back = [[None] * (k_max + 1) for _ in points]
    for i, p in enumerate(points):
        add = 0 if p in declared else 1
        for j in range(i - 1, -2, -1):
            seg = (prefix[i + 1] - prefix[j + 1]) * p
            for k in range(add, k_max + 1):
                prev = (0.0 if k == add else inf) if j < 0 else best[j][k - add]
                if prev + seg < best[i][k]:
                    best[i][k], back[i][k] = prev + seg, (j, k - add)
            if j >= 0 and points[j] in declared:
                break
    out, last = [], len(points) - 1
    for k in range(1, k_max + 1):
        if best[last][k] == inf:
            break
        added, at = [], (last, k)
        while at[0] >= 0:
            if points[at[0]] not in declared:
                added.append(points[at[0]])
            at = back[at[0]][at[1]]
        out.append((sorted(added), best[last][k]))
    return out


def cost_model(series):
    """capacity -> seconds per launch, measured or interpolated between measured neighbours."""
    counts = series.get("launch_seconds_count", {})
    mean = {c: s / counts[c] for c, s in series.get("launch_seconds_sum", {}).items()
            if counts.get(c, 0) > 0}
    known = sorted(mean)

    def seconds(cap):
        if cap in mean:
            return mean[cap]
        lo = [c for c in known if c < cap]
        hi = [c for c in known if c > cap]
        if not lo or not hi:
            return None
        a, b = lo[-1], hi[0]
        return mean[a] + (mean[b] - mean[a]) * (cap - a) / (b - a)
    return seconds


def launch_seconds(caps, demand, seconds):
    total = 0.0
    for r, n in demand.items():
        t = seconds(cover(caps, r))
        if t is None:
            return None
        total += n * t
    return total


def pct(part, whole):
    return f"{100.0 * part / whole:.1f} %" if whole else "n/a"


def report(group, series, k_max, top):
    model, family = group
    caps = sorted(c for c in series.get("capacity_info", {}) if c)
    if not caps:
        return []
    demand = {r: n for r, n in series.get("demand_total", {}).items() if n > 0 and 0 < r <= caps[-1]}
    frames = sum(demand.values())
    launches = series.get("launches_total", {})
    ran = sum(n * c for c, n in launches.items())
    padded = sum(series.get("padded_rows_total", {}).values())
    overflow = sum(series.get("overflow_total", {}).values())
    lines = [f"model={model} family={family}",
             "  declared: " + " ".join(map(str, caps)),
             f"  frames: {frames:.0f}   launches: {sum(launches.values()):.0f}   padded rows: "
             f"{padded:.0f} of {ran:.0f} launched ({pct(padded, ran)})   overflow: {overflow:.0f}"
             f"   max rows: {max(series.get('demand_max', {None: 0}).values()):.0f}"]
    if overflow:
        lines.append(f"  {overflow:.0f} frames had more rows than the widest capacity {caps[-1]}: "
                     "a wider top capacity is needed (not part of the plans below)")
    if not frames:
        return lines + ["  no demand in the window", ""]
    base = launched(caps, demand)
    live = sum(n * r for r, n in demand.items())
    lines.append("  demand, largest padding first:")
    lines.append(f"    {'rows':>6} {'frames':>10} {'share':>8}  {'runs at':>7} {'padding':>8}")
    rows = sorted(demand, key=lambda r: (-demand[r] * (cover(caps, r) - r), r))
    for r in rows[:top]:
        c = cover(caps, r)
        lines.append(f"    {r:>6} {demand[r]:>10.0f} {pct(demand[r], frames):>8}  {c:>7} "
                     f"{pct(demand[r] * (c - r), base):>8}")
    if len(rows) > top:
        lines.append(f"    ... {len(rows) - top} more row counts")
    seconds = cost_model(series)
    base_s = launch_seconds(caps, demand, seconds)
    lines.append(f"  smallest-capacity model: {base:.0f} launched rows, {pct(base - live, base)} padding")
    for added, rows_k in plan(caps, demand, k_max):
        line = (f"  add {len(added)}: {added}  launched rows -{pct(base - rows_k, base)}, "
                f"padding {pct(rows_k - live, rows_k)}")
        with_s = base_s and launch_seconds(sorted(caps + added), demand, seconds)
        if with_s:
            line += f", launch time -{pct(base_s - with_s, base_s)} ({base_s - with_s:.1f} of {base_s:.1f} s)"
        lines.append(line)
    if base_s is None:
        lines.append("  launch time: n/a (a capacity in use has no measured neighbours)")
    return lines + [""]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+", help="/metrics URL or saved text; two = the window between them")
    ap.add_argument("--model", help="only this model_name")
    ap.add_argument("--family", help="only this family (decode, prefill, depth, codec)")
    ap.add_argument("--add", type=int, default=3, help="largest number of capacities to add (default 3)")
    ap.add_argument("--top", type=int, default=12, help="demand rows to print (default 12)")
    ap.add_argument("--per-engine", action="store_true", help="do not sum a model's engines")
    args = ap.parse_args(argv)
    if len(args.metrics) > 2:
        ap.error("one snapshot, or two for a window")
    snaps = [parse(read(s), args.per_engine) for s in args.metrics]
    data = window(*snaps) if len(snaps) == 2 else snaps[0]
    lines = []
    for group in sorted(data):
        if args.model and group[0].split("#")[0] != args.model:
            continue
        if args.family and group[1] != args.family:
            continue
        lines += report(group, data[group], args.add, args.top)
    if not lines:
        print("no plowrt_bucket_* series (plowrt too old, or no model with a declared ladder)", file=sys.stderr)
        return 1
    print("\n".join(lines).rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
