#!/usr/bin/env python3
"""k3_trace_report.py — decompose a decode step's per-packet protocol cost.

Reads the raw `PlowTraceRec[n_stream]` buffer that `PLOW_TRACE_RAW=<path>` makes
`plowrt amd-bench` dump (see `AmdEngine::trace_write`).

    PlowTraceRec = { u32 cu, u32 pc, u32 inst, u16 op, u16 slice,
                     u64 t_arrive, u64 t_ready, u64 t_end }   (40 B)

Clocks are `s_memrealtime` ticks, 100 MHz on gfx950 (the `TPUS` constant in
runtime/bench/interp_dispatch_floor.hip, calibrated against the GLM autopsy).

# The timeline this builds, and why it is not just a sum

One PACKET is one instruction dispatched over `b` workgroups, each of which
writes its own record. The packet as a whole:

    arrive = min t_arrive   the first workgroup to reach it
    ready  = max t_ready    the last workgroup through the gate = when the body
                            really starts, since a wide op is only as started as
                            its slowest slice
    end    = max t_end      the last workgroup to signal

Decode is a serial chain, so the packets are walked in INSTRUCTION order (which
the compiler emits topologically) and charged against a running clock:

    start_i = max(end_{i-1}, arrive_i)
    gate_i  = max(0, ready_i - start_i)
    body_i  = max(0, end_i - max(ready_i, start_i))

    and advances `end_{i-1}` monotonically. This partitions the critical envelope
    without double counting when independent packets overlap.

WHAT `gate` MEANS depends on the run. On a normal run it is protocol latency PLUS
genuine dependency waiting (a consumer whose producer is still computing). Under
`PLOW_K3_ABLATE=<all opcodes>` the bodies are `Nop`, so the dependency component
is gone and what is left is the protocol: cross-XCD convergence, the acquire
fence, and the release signal. Compare the two runs to separate them.

Usage:  k3_trace_report.py <trace.bin> [--top N] [--opnames <dev_isa.h>]
"""
import re
import struct
import sys
from collections import defaultdict

REC = struct.Struct("<IIIHHQQQ")
TPUS = 100.0  # s_memrealtime ticks per microsecond


def op_names(path):
    names = {}
    try:
        with open(path) as f:
            for m in re.finditer(r"PLOW_DOP_(\w+)\s*=\s*(\d+)", f.read()):
                names.setdefault(int(m.group(2)), m.group(1))
    except OSError:
        pass
    return names


def main():
    argv = sys.argv[1:]
    path = argv[0]
    top = 26
    isa = "runtime/common/dev_isa.h"
    for i, a in enumerate(argv):
        if a == "--top":
            top = int(argv[i + 1])
        if a == "--opnames":
            isa = argv[i + 1]
    names = op_names(isa)

    blob = open(path, "rb").read()
    n = len(blob) // REC.size

    # Fold the per-workgroup records into per-PACKET envelopes.
    pk = {}  # inst -> [op, b, arrive, ready_max, end_max, end_min, ready_min]
    for i in range(n):
        cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(blob, i * REC.size)
        if not te:
            continue  # slot no workgroup reached
        e = pk.get(inst)
        if e is None:
            pk[inst] = [op, 1, ta, tr, te, te, tr]
        else:
            e[1] += 1
            e[2] = min(e[2], ta)
            e[3] = max(e[3], tr)
            e[4] = max(e[4], te)
            e[5] = min(e[5], te)
            e[6] = min(e[6], tr)
    order = sorted(pk)
    print(f"{path}: {n} slots, {len(order)} packets, "
          f"{sum(v[1] for v in pk.values())} (workgroup,packet) records")

    # Walk the chain.
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # (op,b) -> n, gate, body, strag
    trans = defaultdict(lambda: [0, 0.0])
    prev_end = None
    prev_b = None
    tot_g = tot_b = 0.0
    gs, bs = [], []
    for inst in order:
        op, b, arr, rdy, end, end0, rdy0 = pk[inst]
        start = arr if prev_end is None else max(prev_end, arr)
        g = max(0.0, (rdy - start) / TPUS)
        bd = max(0.0, (end - max(rdy, start)) / TPUS)
        a = agg[(op, b)]
        a[0] += 1
        a[1] += g
        a[2] += bd
        a[3] += (end - end0) / TPUS  # this packet's own straggler spread
        if prev_b is not None:
            t = trans[(prev_b, b)]
            t[0] += 1
            t[1] += g
        tot_g += g
        tot_b += bd
        gs.append(g)
        bs.append(bd)
        prev_end, prev_b = end if prev_end is None else max(prev_end, end), b

    span = (prev_end - pk[order[0]][2]) / TPUS
    print(f"\nchain: {len(order)} packets, span {span:.1f} us = gate {tot_g:.1f} "
          f"+ body {tot_b:.1f}  (residual {span - tot_g - tot_b:.1f})")
    print(f"  per packet: gate {tot_g / len(order):.3f} us  body "
          f"{tot_b / len(order):.3f} us  total {(tot_g + tot_b) / len(order):.3f} us")

    print(f"\n{'op':<32}{'b':>5}{'n':>6}{'gate us':>10}{'body us':>10}"
          f"{'tot us':>9}{'gate/pk':>9}{'body/pk':>9}{'strag/pk':>9}")
    for (op, b), (cnt, g, bd, st) in sorted(agg.items(),
                                            key=lambda kv: -(kv[1][1] + kv[1][2]))[:top]:
        nm = names.get(op, f"op{op}")
        print(f"{nm:<32}{b:>5}{cnt:>6}{g:>10.2f}{bd:>10.2f}{g + bd:>9.2f}"
              f"{g / cnt:>9.3f}{bd / cnt:>9.3f}{st / cnt:>9.3f}")
    print(f"{'TOTAL':<32}{'':>5}{len(order):>6}{tot_g:>10.2f}{tot_b:>10.2f}"
          f"{tot_g + tot_b:>9.2f}")

    print(f"\n{'prev_b -> b':<16}{'n':>7}{'gate us':>10}{'gate/pk':>10}")
    for (pb, b), (cnt, g) in sorted(trans.items(), key=lambda kv: -kv[1][1])[:16]:
        print(f"{f'{pb} -> {b}':<16}{cnt:>7}{g:>10.2f}{g / cnt:>10.3f}")

    gs.sort()
    bs.sort()
    q = lambda v, p: v[min(len(v) - 1, int(len(v) * p))]
    print(f"\ngate us  p10 {q(gs,.1):.2f} p50 {q(gs,.5):.2f} p90 {q(gs,.9):.2f} "
          f"p99 {q(gs,.99):.2f} max {gs[-1]:.2f}")
    print(f"body us  p10 {q(bs,.1):.2f} p50 {q(bs,.5):.2f} p90 {q(bs,.9):.2f} "
          f"p99 {q(bs,.99):.2f} max {bs[-1]:.2f}")

    # STRAGGLER vs CONVERGENCE, for the wide producers only. `straggler` is the
    # spread of the producer's own workgroups' t_end; `convergence` is what the
    # NEXT packet still waited after the producer's LAST workgroup had signalled,
    # i.e. pure fabric + counter-line contention.
    strag = conv = 0.0
    cnt = 0
    for i in range(1, len(order)):
        p = pk[order[i - 1]]
        if p[1] < 64:
            continue
        c = pk[order[i]]
        strag += (p[4] - p[5]) / TPUS
        conv += max(0.0, (c[3] - p[4]) / TPUS)
        cnt += 1
    if cnt:
        print(f"\nwide (b>=64) producer -> next packet, n={cnt}")
        print(f"  straggler spread  max-min t_end of the producer's own WGs : "
              f"{strag:>9.1f} us  {strag / cnt:.3f}/pk")
        print(f"  convergence       t_ready(next) - max t_end(producer)     : "
              f"{conv:>9.1f} us  {conv / cnt:.3f}/pk")


main()
