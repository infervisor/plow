#!/usr/bin/env python3
"""k3_trace_wg.py — per-WORKGROUP anatomy of a wide packet's straggler tail.

Sibling of k3_trace_report.py, same raw `PlowTraceRec` input. For every packet with b >= 64 it
looks INSIDE the packet: where its workgroups arrived, when they cleared the gate, how long each
body ran, and which of those the last-ending workgroup is late on.

Per packet, with WG `L` = argmax t_end:
    end_sp      = max t_end - min t_end                        (the "straggler spread")
    L:late_st   = t_ready[L] - min t_ready                     (L cleared the gate late)
    L:long_b    = (t_end-t_ready)[L] - median(t_end-t_ready)   (L's body ran long)
    L:late_ar   = t_arrive[L] - min t_arrive
    L:prevov    = max(0, end of L's PREVIOUS packet - min t_arrive of this one)
    bod_*       = min / median / max of the per-WG body (t_end - t_ready)
    multi       = workgroups holding >1 record of the packet (claimed two slices)

Usage:  k3_trace_wg.py <trace.bin> [--opnames <dev_isa.h>] [--top N] [--op NAME] [--dump inst]
"""
import re
import struct
import sys
from collections import defaultdict

REC = struct.Struct("<IIIHHQQQ")
TPUS = 100.0
NXCD = 8


def op_names(path):
    names = {}
    try:
        with open(path) as f:
            for m in re.finditer(r"PLOW_DOP_(\w+)\s*=\s*(\d+)", f.read()):
                names.setdefault(int(m.group(2)), m.group(1))
    except OSError:
        pass
    return names


def median(v):
    s = sorted(v)
    return s[len(s) // 2]


def main():
    argv = sys.argv[1:]
    path = argv[0]
    isa = "runtime/common/dev_isa.h"
    top = 12
    only = None
    dump = None
    for i, a in enumerate(argv):
        if a == "--opnames":
            isa = argv[i + 1]
        if a == "--top":
            top = int(argv[i + 1])
        if a == "--op":
            only = argv[i + 1]
        if a == "--dump":
            dump = int(argv[i + 1])
    names = op_names(isa)

    blob = open(path, "rb").read()
    n = len(blob) // REC.size
    recs = defaultdict(list)  # inst -> [(cu, pc, slice, ta, tr, te, op)]
    by_cu = defaultdict(list)  # cu -> [(pc, inst, ta, tr, te)]
    for i in range(n):
        cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(blob, i * REC.size)
        if not te:
            continue
        recs[inst].append((cu, pc, sl, ta, tr, te, op))
        by_cu[cu].append((pc, inst, ta, tr, te, op))
    prev_end = {}
    prev_op = {}
    for cu, lst in by_cu.items():
        lst.sort()
        for j in range(1, len(lst)):
            prev_end[(cu, lst[j][0])] = lst[j - 1][4]
            prev_op[(cu, lst[j][0])] = (lst[j - 1][5], lst[j - 1][1])

    agg = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    xcd_off = defaultdict(lambda: [0.0] * NXCD)
    slice_half = defaultdict(lambda: [0.0, 0.0])
    multi = defaultdict(int)
    dump_list = []
    for inst, rs in recs.items():
        b = len(rs)
        if b < 64:
            continue
        op = rs[0][6]
        key = (names.get(op, f"op{op}"), b)
        if only and key[0] != only:
            continue
        cnt[key] += 1
        ta0 = min(r[3] for r in rs)
        tr0 = min(r[4] for r in rs)
        te0 = min(r[5] for r in rs)
        bodies = [r[5] - r[4] for r in rs]
        L = max(rs, key=lambda r: r[5])
        a = agg[key]
        a["arrive_spread"] += (max(r[3] for r in rs) - ta0) / TPUS
        a["ready_spread"] += (max(r[4] for r in rs) - tr0) / TPUS
        a["end_spread"] += (L[5] - te0) / TPUS
        a["body_spread"] += (max(bodies) - min(bodies)) / TPUS
        a["body_med"] += median(bodies) / TPUS
        a["body_min"] += min(bodies) / TPUS
        a["body_max"] += max(bodies) / TPUS
        a["late_start"] += (L[4] - tr0) / TPUS
        a["long_body"] += ((L[5] - L[4]) - median(bodies)) / TPUS
        a["late_arrive"] += (L[3] - ta0) / TPUS
        pe = prev_end.get((L[0], L[1]))
        if pe is not None:
            a["prev_ov"] += max(0.0, (pe - ta0) / TPUS)
        a["L_arr_after_open"] += max(0.0, (L[3] - tr0) / TPUS)
        a["L_is_last_ready"] += 1.0 if L[4] == max(r[4] for r in rs) else 0.0
        a["L_is_last_arrive"] += 1.0 if L[3] == max(r[3] for r in rs) else 0.0
        if only:
            dump_list.append(inst)
        for r in rs:
            xcd_off[key][r[0] % NXCD] += (r[5] - te0) / TPUS / (b / NXCD)
            slice_half[key][0 if r[2] < b // 2 else 1] += (r[5] - te0) / TPUS / (b / 2)
        seen = defaultdict(int)
        for r in rs:
            seen[r[0]] += 1
        multi[key] += sum(1 for v in seen.values() if v > 1)

    print(f"{path}: {n} slots; wide packets (b>=64): {sum(cnt.values())}")
    keys = sorted(cnt, key=lambda k: -agg[k]["end_spread"])[:top]
    print(f"\n{'op':<26}{'b':>4}{'n':>5} | {'arr_sp':>7}{'rdy_sp':>7}{'end_sp':>7} | "
          f"{'bod_min':>8}{'bod_med':>8}{'bod_max':>8}{'bod_sp':>7} | "
          f"{'L:late_st':>9}{'L:long_b':>9}{'L:late_ar':>9}{'L:prevov':>9}{'multi':>6}"
          f"{'L:ar>op':>8}{'L=lstR':>7}{'L=lstA':>7}")
    for k in keys:
        a = agg[k]
        c = cnt[k]
        print(f"{k[0]:<26}{k[1]:>4}{c:>5} | {a['arrive_spread']/c:>7.2f}{a['ready_spread']/c:>7.2f}"
              f"{a['end_spread']/c:>7.2f} | {a['body_min']/c:>8.2f}{a['body_med']/c:>8.2f}"
              f"{a['body_max']/c:>8.2f}{a['body_spread']/c:>7.2f} | {a['late_start']/c:>9.2f}"
              f"{a['long_body']/c:>9.2f}{a['late_arrive']/c:>9.2f}{a['prev_ov']/c:>9.2f}"
              f"{multi[k]/c:>6.2f}{a['L_arr_after_open']/c:>8.2f}{a['L_is_last_ready']/c:>7.2f}"
              f"{a['L_is_last_arrive']/c:>7.2f}")
    print("\nper-XCD (cu%8) mean t_end offset from the packet's first-ending WG, us; slice halves lo/hi")
    for k in keys:
        c = cnt[k]
        xs = " ".join(f"{v / c:5.2f}" for v in xcd_off[k])
        sc = slice_half[k]
        print(f"{k[0]:<26}{k[1]:>4}  {xs}   slice<b/2 {sc[0]/c:5.2f}  >= {sc[1]/c:5.2f}")

    if dump is not None:
        if dump < 0:  # --dump -k with --op: the k-th packet (1-based) of that op
            dump_list.sort()
            dump = dump_list[-dump - 1]
        rs = sorted(recs[dump], key=lambda r: r[5])
        ta0 = min(r[3] for r in rs)
        print(f"\ninst {dump} op {names.get(rs[0][6])} b={len(rs)}  (us rel. first arrival)")
        print(f"{'cu':>4}{'xcd':>4}{'slice':>6}{'arrive':>8}{'ready':>8}{'end':>8}{'body':>7}"
              f"{'prev_end':>9}  prev")
        for r in rs:
            pe = prev_end.get((r[0], r[1]))
            po = prev_op.get((r[0], r[1]))
            ps = "" if pe is None else f"{(pe - ta0) / TPUS:9.2f}"
            pn = "" if po is None else f"{names.get(po[0], po[0])}#{po[1]}"
            print(f"{r[0]:>4}{r[0] % 8:>4}{r[2]:>6}{(r[3] - ta0) / TPUS:>8.2f}{(r[4] - ta0) / TPUS:>8.2f}"
                  f"{(r[5] - ta0) / TPUS:>8.2f}{(r[5] - r[4]) / TPUS:>7.2f}{ps:>9}  {pn}")


main()
