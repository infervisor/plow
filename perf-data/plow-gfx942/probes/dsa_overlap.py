#!/usr/bin/env python3
"""B3 decisive probe: real overlap structure of GLM-5.2 DSA prefill selections.

Reads the dumped act.iidx_pf ([t, topk] i32, -1 pad) and act.iuni (union table)
from the LAST prefill chunk and reports:
  - adjacent-query selection overlap vs the random baseline
  - true union-of-8 size distribution vs the kernel's cnt[qt]
  - implied flash-walk ratio vs causal
Usage: overlap.py <iidx.bin> <iuni.bin> <t> <topk> <q_pos0> <cap>
"""
import sys
import numpy as np

iidx_p, iuni_p, t, topk, q_pos0, cap = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), int(sys.argv[6]))
idx = np.fromfile(iidx_p, dtype=np.int32)[: t * topk].reshape(t, topk)
P = 8
n_qt = (t + P - 1) // P
uni = np.fromfile(iuni_p, dtype=np.uint8)
cnt = uni[: ((n_qt * 4 + 255) // 256) * 256].view(np.uint32)[:n_qt]

rows = [set(r[r >= 0].tolist()) for r in idx]
row_len = np.array([q_pos0 + i + 1 for i in range(t)])

# adjacent overlap on selecting rows only (row_len > topk)
sel = [i for i in range(t - 1) if row_len[i] > topk and row_len[i + 1] > topk]
ov = np.array([len(rows[i] & rows[i + 1]) for i in sel[:2000]])
rnd = np.array([topk * topk / row_len[i] for i in sel[:2000]])
print(f"adjacent overlap: mean {ov.mean():.0f} / {topk}  (random baseline {rnd.mean():.0f})")
print(f"  p10/p50/p90: {np.percentile(ov,10):.0f}/{np.percentile(ov,50):.0f}/{np.percentile(ov,90):.0f}")

# true union per pack vs kernel cnt
tu, ku, causal = [], [], []
for q in range(0, min(t, 4096), P):
    u = set()
    for j in range(q, min(q + P, t)):
        u |= rows[j]
    tu.append(len(u))
    ku.append(int(cnt[q // P]))
    causal.append(int(row_len[min(q + P - 1, t - 1)]))
tu, ku, causal = map(np.array, (tu, ku, causal))
print(f"union-of-8 TRUE: mean {tu.mean():.0f}  kernel cnt: mean {ku.mean():.0f}  causal: mean {causal.mean():.0f}")
print(f"  kernel==true fraction: {(tu == ku).mean():.2f}   max|diff| {np.abs(tu-ku).max()}")
print(f"implied walk ratio (true union / causal): {tu.sum()/causal.sum():.3f}")
print(f"implied walk ratio (kernel cnt / causal): {ku.sum()/causal.sum():.3f}")
# structure: how much of each selection is 'local tail' (last 512) + 'sink head' (first 64)?
q0 = sel[0] if sel else 0
r0 = rows[q0]
pos = np.array(sorted(r0))
tail = (pos >= row_len[q0] - 512).sum()
head = (pos < 64).sum()
print(f"sample row {q0}: {len(r0)} sel, {head} in first-64, {tail} in last-512")
