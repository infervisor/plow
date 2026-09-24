#!/usr/bin/env python3
"""Prefill/decode wall split and the decode cost-vs-rows curve, from a PLOW_PF_PACKLOG=1 log.

    packlog_split.py <server.log>

`packlog_gap.py` gives the per-tick residual; this gives the totals that answer "where does the
wall go" and the marginal cost of a decode row.

Two traps this encodes, both of which produced wrong conclusions before:

  * `t_ms` is elapsed-since-start, a TIMESTAMP. Summing it yields nonsense (once: "wall in ticks
    15018 s"). Only prefill_ms and decode_ms are durations.
  * `decode_ms` is 0.00 on every tick with did_prefill=1. That is NOT decode starvation: the
    unified token batch decodes `feeds` inside the prefill pass and clears them (mux.rs:2101)
    before `packlog::tick` reads `feeds.len()`, so riding decode is billed to prefill_ms. Do not
    read a zero there as "decode never rides prefill".

Because of the second trap, sum(prefill_ms) is prefill rows PLUS the decode rows that rode with
them, and this script cannot separate the two. Splitting that term needs the unified pass to time
decode rows separately.
"""
import collections
import re
import statistics as st
import sys

TICK = re.compile(
    r"PACKLOG TICK t_ms=([\d.]+) prefill_ms=([\d.]+) decode_ms=([\d.]+) "
    r"did_prefill=(\d+) decode_rows=(\d+)"
)


def main(path):
    ticks = []
    for line in open(path, errors="replace"):
        m = TICK.search(line)
        if m:
            ticks.append(
                (float(m[1]), float(m[2]), float(m[3]), int(m[4]), int(m[5]))
            )
    if not ticks:
        raise SystemExit(f"no PACKLOG TICK lines in {path} (was PLOW_PF_PACKLOG=1 set?)")

    span = (ticks[-1][0] - ticks[0][0]) / 1000
    pf = sum(t[1] for t in ticks) / 1000
    dc = sum(t[2] for t in ticks) / 1000
    print(f"ticks={len(ticks)}  span={span:.2f} s")
    print(f"  sum prefill_ms = {pf:6.2f} s   (includes decode rows that rode the prefill pass)")
    print(f"  sum decode_ms  = {dc:6.2f} s   (decode-only ticks)")
    print(f"  accounted      = {pf + dc:6.2f} s   residual vs span {span - pf - dc:.2f} s")
    print("  compare 'accounted' to the cell's wall; residual vs span is server idle, not overhead")

    pf_ticks = [t for t in ticks if t[3] == 1]
    dc_ticks = [t for t in ticks if t[3] == 0]
    print(f"\nprefill ticks={len(pf_ticks)}  decode-only ticks={len(dc_ticks)}")

    print("\ndecode-only steps, cost by row count:")
    by = collections.defaultdict(list)
    for t in dc_ticks:
        by[t[4]].append(t[2])
    for rows in sorted(by):
        v = by[rows]
        med = st.median(v)
        print(
            f"  rows={rows:3d} n={len(v):4d} median={med:7.3f} ms "
            f"total={sum(v) / 1000:6.2f} s ms/row={med / max(rows, 1):6.3f}"
        )
    lo = min(by) if by else None
    hi = max(by) if by else None
    if lo is not None and hi is not None and hi > lo:
        a, b = st.median(by[lo]), st.median(by[hi])
        print(
            f"\n  marginal cost per row: ({b:.3f} - {a:.3f}) / {hi - lo} "
            f"= {(b - a) / (hi - lo):.3f} ms/row over a {a:.3f} ms fixed pass at rows={lo}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
