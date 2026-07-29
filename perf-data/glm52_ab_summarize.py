#!/usr/bin/env python3
"""Summarise a run log: per-blob ms medians, and blob-to-blob token-stream identity."""
import re, sys, collections

t = open(sys.argv[1]).read()
blocks = re.split(r'##########\s+(\S+)', t)[1:]
ms = collections.defaultdict(list)
ms4 = collections.defaultdict(list)
seq, mech = {}, {}
for i in range(0, len(blocks), 2):
    name, body = blocks[i], blocks[i + 1]
    for c, v in re.findall(r'^  (1024|4096)\s+([0-9.]+)', body, re.M):
        (ms if c == '1024' else ms4)[name].append(float(v))
    m = re.search(r'  ids:([0-9 ]+)', body)
    if m:
        seq.setdefault(name, []).append(m.group(1).split())
    m = re.search(r'max id (\d+), cross-rank disagreements (\d+)', body)
    if m:
        mech.setdefault(name, []).append((int(m.group(1)), int(m.group(2))))

print(f"{'blob':16s} {'runs @1k':38s} {'mean':>8s} {'@4k':>8s}  max-id / disagree")
for k in ms:
    r = ms[k]
    mm = f"{sum(r)/len(r):8.3f}"
    m4 = f"{sum(ms4[k])/len(ms4[k]):8.3f}" if ms4[k] else "       -"
    mk = " ".join(f"{x:.3f}" for x in r)
    print(f"{k:16s} {mk:38s} {mm} {m4}  {mech.get(k, [])}")

base = 'c1.pkt'
if base in ms:
    b = sum(ms[base]) / len(ms[base])
    print(f"\ndelta vs {base} mean {b:.3f}:")
    for k in ms:
        if k != base:
            print(f"  {k:16s} {sum(ms[k])/len(ms[k]) - b:+7.3f}")

print("\ntoken streams vs c1.pkt:")
if base in seq:
    ref = seq[base][0]
    for k, runs in seq.items():
        for j, s in enumerate(runs):
            if s == ref:
                hi = sum(1 for x in s if int(x) >= 38720)
                print(f"  {k:16s} run{j}: IDENTICAL ({len(s)} ids, {hi} outside rank-0 vocab shard)")
            else:
                d = next((i for i, (x, y) in enumerate(zip(s, ref)) if x != y), None)
                print(f"  {k:16s} run{j}: differs at index {d}")
