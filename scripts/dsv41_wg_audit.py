#!/usr/bin/env python3
"""How many of a packet's workgroups actually do work, per op.

The megakernel launches all 304 workgroups for every packet. An op that confines itself to a
subset serializes: its wall time is the subset's, while the rest of the machine idles. XREDUCE2
was caught this way (24 of 304 busy, the 24 perfectly balanced). This asks every op the same
question, so the answer is a table rather than one anecdote.

busy      = workgroups whose (t_end - t_ready) exceeds 5% of that packet's max
aggregate = sum of all workgroups' busy time
ideal304  = aggregate / 304, i.e. the wall time if the SAME work filled the machine evenly
serial    = max / ideal304, the factor by which the op is narrower than the machine
"""
import os
import re
import struct
import sys
from collections import defaultdict

REC = struct.Struct("<IIIHHQQQ")
TPUS = 100.0
NCU = 304


def dop_values():
    """`PLOW_DOP_*` name -> opcode, read from runtime/common/dev_isa.h beside this script."""
    isa = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime", "common", "dev_isa.h")
    return {m[1]: int(m[2]) for m in re.finditer(r"PLOW_DOP_(\w+)\s*=\s*(\d+)", open(isa).read())}


OPS = {v: k for k, v in dop_values().items()}

blob = open(sys.argv[1], "rb").read()
n = len(blob) // REC.size
per = defaultdict(lambda: defaultdict(list))  # op -> inst -> [busy_us]
for i in range(n):
    cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(blob, i * REC.size)
    if not te:
        continue
    per[op][inst].append((te - tr) / TPUS)

print(f"{'op':<24}{'pkts':>6}{'WGs':>6}{'busy':>7}{'max us':>9}"
      f"{'aggreg':>10}{'ideal304':>10}{'serial':>8}")
rows = []
for op, insts in per.items():
    tot_max = tot_agg = 0.0
    busy_n = wg_n = 0
    for inst, dur in insts.items():
        mx = max(dur)
        tot_max += mx
        tot_agg += sum(dur)
        busy_n += sum(1 for d in dur if d > 0.05 * mx)
        wg_n += len(dur)
    k = len(insts)
    ideal = tot_agg / NCU
    rows.append((tot_max, OPS.get(op, f"op{op}"), k, wg_n // k, busy_n / k, tot_max,
                 tot_agg, ideal, tot_max / ideal if ideal else 0))
for _, nm, k, wgs, busy, mx, agg, ideal, ser in sorted(rows, reverse=True):
    print(f"{nm:<24}{k:>6}{wgs:>6}{busy:>7.0f}{mx/k:>9.1f}"
          f"{agg/1000:>9.1f}k{ideal/1000:>9.1f}k{ser:>8.2f}")
