#!/usr/bin/env python3
"""k3_xr_decode_report.py — attribute the decode one-shot XREDUCE body across ranks.

Reads one raw `PlowTraceRec` dump per rank (`PLOW_TRACE_RAW=<path> PLOW_TRACE_ALLRANKS=1`,
files `<path>.rk0..7`; a single rank-0 file also works). Same 40-byte record and 100 MHz
`s_memrealtime` clock as scripts/k3_trace_report.py. Clocks on different GPUs share no
epoch, so nothing here subtracts timestamps across ranks.

Per XREDUCE packet and rank, body = max t_end - max t_ready over that rank's workgroups.
The rank that arrives LAST at a collective never waits for a peer, so

    floor(k)  = min over ranks of body(k)       the protocol + data cost itself
    wait_r(k) = body_r(k) - floor(k)            how long rank r waited for the last rank

With `PLOW_XR_TRACE_PHASES=1` objects the one-shot writes the phase schema (slice bits
15:14 = 0b10): t_arrive = body entry, cu = signal loop issued (slice 0), pc = arrival gate
cleared, t_ready = system acquire done, t_end = reduce done, all deltas entry-relative.
That splits the body into signal / wait / acquire / reduce+publish per workgroup.

Usage:  k3_xr_decode_report.py <trace.rk0> [<trace.rk1> ...] [--opnames <dev_isa.h>]
"""
import re
import statistics
import struct
import sys
from collections import defaultdict

REC = struct.Struct("<IIIHHQQQ")
TPUS = 100.0


def op_ids(path):
    ids = {}
    try:
        with open(path) as f:
            for m in re.finditer(r"PLOW_DOP_(\w+)\s*=\s*(\d+)", f.read()):
                ids.setdefault(m.group(1), int(m.group(2)))
    except OSError:
        pass
    return ids


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))] if v else 0.0


def load(path, xr):
    """inst -> list of (slice, phase, cu, pc, t_arrive, t_ready, t_end); phase = schema flag."""
    blob = open(path, "rb").read()
    out = defaultdict(list)
    for i in range(len(blob) // REC.size):
        cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(blob, i * REC.size)
        if not te or op != xr:
            continue
        phase = (sl & 0xC000) == 0x8000
        out[inst].append((sl & 0x3FFF, phase, cu, pc, ta, tr, te))
    return out


def main():
    argv = sys.argv[1:]
    isa = "runtime/common/dev_isa.h"
    paths = []
    i = 0
    while i < len(argv):
        if argv[i] == "--opnames":
            isa = argv[i + 1]
            i += 2
        else:
            paths.append(argv[i])
            i += 1
    xr = op_ids(isa).get("XREDUCE")
    if xr is None:
        sys.exit(f"PLOW_DOP_XREDUCE not found in {isa}")
    ranks = [load(p, xr) for p in paths]
    insts = sorted(set().union(*[r.keys() for r in ranks]))
    if not insts:
        sys.exit("no XREDUCE records")
    print(f"{len(paths)} rank(s), {len(insts)} XREDUCE packets")

    # Per (inst, rank): b, body, phase split.
    body = {}  # (inst, r) -> body us
    cls = {}  # inst -> b
    phases = defaultdict(list)  # (b, 'slice0'|'other') -> [(sig, wait, acq, red)]
    phases_r = defaultdict(list)  # (b, r) -> [(sig, wait, acq, red)]
    for inst in insts:
        for r, recs in enumerate(ranks):
            rs = recs.get(inst)
            if not rs:
                continue
            cls[inst] = len(rs)
            rdy = max(x[5] for x in rs)
            end = max(x[6] for x in rs)
            body[(inst, r)] = (end - rdy) / TPUS
            for sl, phase, cu, pc, ta, tr, te in rs:
                if not phase:
                    continue
                sig = cu / TPUS if sl == 0 else 0.0
                wait = (pc - cu) / TPUS
                acq = (tr - ta - pc) / TPUS
                red = (te - tr) / TPUS
                phases[(len(rs), "slice0" if sl == 0 else "other")].append((sig, wait, acq, red))
                phases_r[(len(rs), r)].append((sig, wait, acq, red))

    any_phase = any(x[1] for recs in ranks for rs in recs.values() for x in rs)
    if any_phase:
        # Under the phase schema the record's t_ready is the acquire point, so `body`
        # above is reduce+publish only; report the full entry-to-end envelope instead.
        for inst in insts:
            for r, recs in enumerate(ranks):
                rs = recs.get(inst)
                if rs:
                    body[(inst, r)] = (max(x[6] for x in rs) - max(x[4] for x in rs)) / TPUS
        print("phase schema: body = entry-to-end envelope (signal + wait + acquire + reduce)")

    for b in sorted(set(cls.values())):
        ks = [k for k in insts if cls[k] == b]
        print(f"\n== XREDUCE b={b}  n={len(ks)}")
        for r in range(len(ranks)):
            v = [body[(k, r)] for k in ks if (k, r) in body]
            if v:
                print(f"  rank {r} body us  mean {statistics.mean(v):6.2f}  p10 {q(v,.1):6.2f}  "
                      f"p50 {q(v,.5):6.2f}  p90 {q(v,.9):6.2f}  max {max(v):6.2f}")
        if len(ranks) > 1:
            floor, wait0, last = [], [], defaultdict(int)
            for k in ks:
                v = [(body[(k, r)], r) for r in range(len(ranks)) if (k, r) in body]
                if len(v) < 2:
                    continue
                f, lr = min(v)
                floor.append(f)
                wait0.append(body[(k, 0)] - f)
                last[lr] += 1
            if floor:
                print(f"  protocol floor (min over ranks)  mean {statistics.mean(floor):6.2f}  "
                      f"p10 {q(floor,.1):6.2f}  p50 {q(floor,.5):6.2f}  p90 {q(floor,.9):6.2f}")
                print(f"  rank-0 wait for last rank        mean {statistics.mean(wait0):6.2f}  "
                      f"p50 {q(wait0,.5):6.2f}  p90 {q(wait0,.9):6.2f}  max {max(wait0):6.2f}")
                print("  last-arriving rank histogram    ",
                      {r: last[r] for r in sorted(last)})
        for key in (("slice0",), ("other",)):
            v = phases.get((b, key[0]))
            if not v:
                continue
            names = ("signal", "wait", "acquire", "reduce+publish")
            print(f"  phases {key[0]:>6}:", "  ".join(
                f"{nm} p50 {q([x[j] for x in v],.5):5.2f} p90 {q([x[j] for x in v],.9):5.2f}"
                for j, nm in enumerate(names)))
        for r in range(len(ranks)):
            v = phases_r.get((b, r))
            if v:
                print(f"  rank {r} slice-0 signal p50 {q([x[0] for x in v if x[0] > 0],.5):5.2f}"
                      f"  wait p50 {q([x[1] for x in v],.5):5.2f}"
                      f"  reduce p50 {q([x[3] for x in v],.5):5.2f}")


main()
