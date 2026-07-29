#!/usr/bin/env python3
"""Summarise the prefill cross-GPU rendezvous ceiling A/B.

Reads `PF CHUNK T=<bucket> ... drain=<ms> ms` lines — the DEVICE WALL of one prefill
launch — out of each arm's plowrt log, and reports median/mean per arm against the
INTERLEAVED control (the control runs at positions 1/3/6, so control drift is visible
rather than assumed away).
"""
import glob
import os
import re
import statistics as st
import sys

PAT = re.compile(r"PF CHUNK T=(\d+) c0=(\d+) clen=(\d+) seg=(\d+) drain=([0-9.]+) ms")
D = sys.argv[1] if len(sys.argv) > 1 else "/home/lava/models/glm52_xrpf/rt_logs"
WARM = int(os.environ.get("WARM", "2"))  # drop the first N launches of each arm


def load(path):
    out = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            out.append((int(m.group(1)), int(m.group(3)), float(m.group(5))))
    return out


runs = sorted(glob.glob(os.path.join(D, "*.*.log")), key=os.path.getmtime)
print(f"{'#':>2} {'arm':<10} {'bucket':>7} {'n':>4} {'median ms':>10} {'mean':>9} {'sd':>7} {'min':>9} {'max':>9}")
by_arm = {}
for i, path in enumerate(runs, 1):
    arm = os.path.basename(path).split(".")[0]
    rows = load(path)
    if not rows:
        print(f"{i:>2} {arm:<10} {'-':>7} {0:>4}   NO PF CHUNK LINES")
        continue
    buckets = sorted({r[0] for r in rows})
    for b in buckets:
        ms = [r[2] for r in rows if r[0] == b][WARM:]
        if not ms:
            continue
        sd = st.stdev(ms) if len(ms) > 1 else 0.0
        # The noise on this instrument is ADDITIVE (occasional +100..+200 ms excursions
        # from box interference, never a negative one), so the MINIMUM is the least biased
        # estimator of the true device wall and the median still carries the excursions.
        # Both are reported; the verdict below is taken on the min.
        print(f"{i:>2} {arm:<10} {b:>7} {len(ms):>4} {st.median(ms):>10.3f} "
              f"{st.mean(ms):>9.3f} {sd:>7.3f} {min(ms):>9.3f} {max(ms):>9.3f}")
        by_arm.setdefault((arm, b), []).append((i, st.median(ms), min(ms)))

print()
for b in sorted({k[1] for k in by_arm}):
    ctl = [(i, m, mn) for (a, bb), v in by_arm.items() if a == "base" and bb == b for i, m, mn in v]
    if not ctl:
        continue
    cmed = st.median([m for _, m, _ in ctl])
    cmin = st.median([mn for _, _, mn in ctl])
    print(f"bucket T={b}: control medians {[f'{i}:{m:.1f}' for i, m, _ in sorted(ctl)]}"
          f"  mins {[f'{i}:{mn:.1f}' for i, _, mn in sorted(ctl)]}")
    print(f"   {'CONTROL':<10} median {cmed:>9.3f}   min {cmin:>9.3f}")
    for (a, bb), v in sorted(by_arm.items()):
        if bb != b or a == "base":
            continue
        m = st.median([x for _, x, _ in v])
        mn = st.median([x for _, _, x in v])
        print(f"   {a:<10} median {m:>9.3f}   min {mn:>9.3f}   "
              f"delta(min) {mn - cmin:+8.3f} ms = {100*(mn-cmin)/cmin:+.2f}% of the launch")
