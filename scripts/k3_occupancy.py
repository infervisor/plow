"""Per-PACKET workgroup occupancy for one opcode, from a PLOW_TRACE_RAW dump.

k3_trace_report.py aggregates by opcode and reports a straggler as max-min across
workgroups. That column cannot distinguish a slow workgroup from an ABSENT one, and it
double-counts idle CUs -- a call whose tile count is under the 304-CU grid reads as a
large straggler when nothing is straggling at all. This prints, per packet, how many
workgroups did real work and what perfect spread would have cost, which is the question
a narrow GEMM actually poses.

Usage: k3_occupancy.py <trace.bin> [opcode]   (default 184, GemmFp8Mx)
"""

import struct
import sys
from collections import defaultdict

REC = struct.Struct("<IIIHHQQQ")
TPUS = 100.0
OP = int(sys.argv[2]) if len(sys.argv) > 2 else 184  # default GemmFp8Mx

blob = open(sys.argv[1], "rb").read()
n = len(blob) // REC.size

# per packet -> list of (cu, body_us); and the packet envelope
work = defaultdict(list)
env = {}
for i in range(n):
    cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(blob, i * REC.size)
    if not te or op != OP:
        continue
    work[inst].append((cu, (te - tr) / TPUS))
    e = env.get(inst)
    if e is None:
        env[inst] = [tr, te]
    else:
        e[0] = min(e[0], tr)
        e[1] = max(e[1], te)

print(f"op {OP}: {len(work)} packets")
print(f"{'inst':>6} {'wgs':>4} {'span_us':>8} {'busy':>5} {'idle':>5} "
      f"{'maxbody':>8} {'meanbody':>9} {'wasted_us':>9}")
tot_waste = 0.0
for inst in sorted(work):
    rows = work[inst]
    bodies = sorted(b for _, b in rows)
    span = (env[inst][1] - env[inst][0]) / TPUS
    mx = bodies[-1]
    # "idle" = a workgroup that did essentially nothing: under 10% of the busiest one.
    idle = sum(1 for b in bodies if b < 0.10 * mx)
    busy = len(bodies) - idle
    mean = sum(bodies) / len(bodies)
    # If the work in this packet were spread over ALL workgroups, the packet would take
    # mean instead of max. That difference, times the span, is what the narrow shape costs.
    waste = mx - mean
    tot_waste += waste
    print(f"{inst:6d} {len(rows):4d} {span:8.1f} {busy:5d} {idle:5d} "
          f"{mx:8.1f} {mean:9.1f} {waste:9.1f}")
print(f"\nper-layer cost of imperfect spread across op {OP}: {tot_waste:.1f} us")
