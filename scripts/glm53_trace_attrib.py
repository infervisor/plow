#!/usr/bin/env python3
"""Summarize per-packet timing envelopes from PLOW_TRACE_RAW.

PlowTraceRec (dev_isa.h, 40 B): u32 cu, u32 pc, u32 inst, u16 op, u16 slice,
u64 t_arrive, u64 t_ready, u64 t_end, on a CONSTANT ~100 MHz clock (not the shader
clock, which moves with DVFS).

Per PACKET, over the workgroups that ran it:
    ready = max(t_ready) - min(t_arrive)   arrival and dependency-wait envelope
    tail  = max(t_end)   - max(t_ready)    completion after the last WG is ready
    span  = max(t_end)   - min(t_arrive)

Packets overlap. Their summed spans are not a serialized chain or a wall-time
breakdown. The ready envelope includes staggered workgroup arrival as well as
dependency waits; earlier workgroups may already be executing during it. The tail
is not the full kernel execution duration. These envelopes alone cannot attribute
time to interpreter overhead or establish a critical path.

    python3 scripts/glm53_trace_attrib.py trace.bin --opcodes opcodes.csv --ghz 0.1
"""
import argparse, collections, struct, sys

AP = argparse.ArgumentParser()
AP.add_argument("trace")
AP.add_argument("--opcodes", required=True, help="CSV of NAME,opcode from dev_isa.h")
AP.add_argument("--ghz", type=float, default=0.1, help="trace clock in GHz (~100 MHz)")
AP.add_argument("--top", type=int, default=16)
AP.add_argument("--drain-ms", type=float, default=None, help="measured GPU drain for this step")
A = AP.parse_args()

names = {}
for line in open(A.opcodes):
    line = line.strip()
    if not line or "," not in line:
        continue
    n, c = line.rsplit(",", 1)
    names[int(c)] = n

data = open(A.trace, "rb").read()
n = len(data) // 40
packets = {}
for i in range(n):
    cu, pc, inst, op, slc, ta, tr, te = struct.unpack_from("<IIIHHQQQ", data, i * 40)
    if te == 0 and ta == 0:
        continue  # slot never executed
    p = packets.get(inst)
    if p is None:
        packets[inst] = [op, ta, tr, te, 1]
    else:
        p[1] = min(p[1], ta)
        p[2] = max(p[2], tr)
        p[3] = max(p[3], te)
        p[4] += 1

if not packets:
    sys.exit("trace holds no executed records")

us = lambda ticks: ticks / (A.ghz * 1e3)
agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])  # op -> [count, stall_us, body_us, wgs]
for inst, (op, ta, tr, te, wgs) in packets.items():
    a = agg[op]
    a[0] += 1
    a[1] += us(max(0, tr - ta))
    a[2] += us(max(0, te - tr))
    a[3] += wgs

tot_stall = sum(v[1] for v in agg.values())
tot_body = sum(v[2] for v in agg.values())
tot = tot_stall + tot_body
wall = us(max(p[3] for p in packets.values()) - min(p[1] for p in packets.values()))

print(f"=== {A.trace}")
print(f"    {len(packets)} packets, {n} trace slots, clock {A.ghz*1e3:.0f} MHz")
print()
print(f"{'op':<24}{'n':>5}{'wg/pkt':>8}{'ready ms':>10}{'tail ms':>10}{'span ms':>10}{'%sum':>8}")
for op, (c, st, bd, wgs) in sorted(agg.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[: A.top]:
    sp = st + bd
    print(f"{names.get(op, f'op{op}'):<24}{c:>5}{wgs/c:>8.0f}"
          f"{st/1e3:>10.3f}{bd/1e3:>10.3f}{sp/1e3:>10.3f}{100*sp/tot:>8.1f}")
print()
print(f"{'SUM (overlapping)':<24}{len(packets):>5}{'':>8}"
      f"{tot_stall/1e3:>10.3f}{tot_body/1e3:>10.3f}{tot/1e3:>10.3f}{100:>8.1f}")
print(f"    ready share of span sum : {100*tot_stall/tot:.1f}%")
print("    packet spans overlap; sums and shares are not wall-time attribution")
print(f"    wall span of the trace   : {wall/1e3:.3f} ms")
if A.drain_ms:
    print(f"    measured GPU drain       : {A.drain_ms:.3f} ms"
          f"   (trace covers {100*wall/1e3/A.drain_ms:.0f}% of it)")
