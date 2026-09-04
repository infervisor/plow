#!/usr/bin/env python3
"""k3_trace_critpath.py — the TRUE critical path of a decode trace, from the blob's counter edges.

k3_trace_report.py walks packets in instruction order with a monotone clock, which is an
upper-bound envelope: it charges an independent sibling as if it were serial. This one uses the
producer -> consumer edges the blob carries, so a packet that ran late off the path (K3's routed
`xe` GEMV finishing under the router, say) is not charged, and a packet that IS on the path
because of head-of-line blocking (the shared-expert GemvGlu claimed 60 us after it was ready)
shows up as exactly that: a huge `gate` against a producer that finished long before.

    plowrt disasm --program 1 --counters --format json <assets>/model.pkt > dec.json
    k3_trace_critpath.py <trace.bin> dec.json [--opnames runtime/common/dev_isa.h]

Per packet: rdy_max = last workgroup through the gate (body start), end = last workgroup's end,
gating producer = the producer with the latest end; a packet with no listed producer (a segment
head — cross-segment edges are dropped from raw-segmented programs) falls back to the latest-
ending earlier packet, i.e. the host launch barrier. `tail>bal` is what a wide packet on the
path would save if its workgroups' work were perfectly balanced from each one's own arrival
(makespan T with sum_i max(0, T - t_ready_i) = sum_i body_i, plus one row-block of slack);
`tail>med` is end - start - median body, the raw straggler tail.
"""
import json
import re
import struct
import sys
from collections import defaultdict

REC = struct.Struct("<IIIHHQQQ")
TPUS = 100.0


def op_names(path):
    names = {}
    try:
        with open(path) as f:
            for m in re.finditer(r"PLOW_DOP_(\w+)\s*=\s*(\d+)", f.read()):
                names.setdefault(int(m.group(2)), m.group(1))
    except OSError:
        pass
    return names


def load_json(path):
    s = open(path).read()
    return json.loads(s[s.index("{"):])  # plowrt logs before the document


def balanced_end(rs, item):
    work = sum(te - tr for tr, te in rs)
    lo, hi = min(tr for tr, _ in rs), max(te for _, te in rs)
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(max(0.0, mid - tr) for tr, _ in rs) < work:
            lo = mid
        else:
            hi = mid
    return hi + item


def main():
    argv = sys.argv[1:]
    trace, decj = argv[0], argv[1]
    isa = "runtime/common/dev_isa.h"
    for i, a in enumerate(argv):
        if a == "--opnames":
            isa = argv[i + 1]
    names = op_names(isa)

    blob = open(trace, "rb").read()
    recs = defaultdict(list)
    ops = {}
    for i in range(len(blob) // REC.size):
        cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(blob, i * REC.size)
        if te:
            recs[inst].append((tr, te))
            ops[inst] = op
    pk = {}
    for inst, rs in recs.items():
        pk[inst] = dict(op=names.get(ops[inst], f"op{ops[inst]}"), b=len(rs),
                        rdy_max=max(r[0] for r in rs), rdy_min=min(r[0] for r in rs),
                        end=max(r[1] for r in rs), rs=rs)

    prod = defaultdict(list)
    prog = load_json(decj)["programs"][0]
    for c in prog["counters"]["per_counter"]:
        for cons in c["consumers"]:
            prod[cons].append(c["producer"])

    order = sorted(pk)
    pref, best = {}, None
    for inst in order:
        pref[inst] = best
        if best is None or pk[inst]["end"] > pk[best]["end"]:
            best = inst
    last = max(order, key=lambda i: pk[i]["end"])
    path, cur = [], last
    while cur is not None:
        cands = [q for q in prod.get(cur, []) if q in pk]
        pred = max(cands, key=lambda q: pk[q]["end"]) if cands else pref[cur]
        path.append((cur, pred))
        cur = pred
    path.reverse()

    span = (pk[last]["end"] - pk[order[0]]["rdy_min"]) / TPUS
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    tot_g = tot_b = 0.0
    for inst, pred in path:
        p = pk[inst]
        start = pk[pred]["end"] if pred is not None else p["rdy_min"]
        g = max(0.0, (p["rdy_max"] - start) / TPUS)
        bd = max(0.0, (p["end"] - max(p["rdy_max"], start)) / TPUS)
        a = agg[(p["op"], p["b"])]
        a[0] += 1
        a[1] += g
        a[2] += bd
        tot_g += g
        tot_b += bd
        if p["b"] >= 64:
            bodies = sorted(te - tr for tr, te in p["rs"])
            med = bodies[len(bodies) // 2]
            a[3] += max(0.0, (p["end"] - balanced_end(p["rs"], med / 7.0)) / TPUS)
            a[4] += (p["end"] - max(p["rdy_max"], start) - med) / TPUS
    print(f"critical path: {len(path)} of {len(order)} packets, span {span:.1f} us = "
          f"gate {tot_g:.1f} + body {tot_b:.1f}")
    print(f"{'op':<26}{'b':>4}{'n':>5}{'gate':>9}{'body':>9}{'gate/pk':>8}{'body/pk':>8}"
          f"{'tail>bal':>9}{'tail>med':>9}")
    for (op, b), (n, g, bd, tb, tm) in sorted(agg.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
        print(f"{op:<26}{b:>4}{n:>5}{g:>9.1f}{bd:>9.1f}{g / n:>8.2f}{bd / n:>8.2f}{tb:>9.1f}"
              f"{tm:>9.1f}")

    # Who gates each packet class on the path, by (producer op, b, instruction distance).
    gated = defaultdict(lambda: defaultdict(int))
    for inst, pred in path:
        if pred is not None:
            gated[(pk[inst]["op"], pk[inst]["b"])][(pk[pred]["op"], pk[pred]["b"], inst - pred)] += 1
    print("\ngating producer of each path packet class (producer op, b, inst distance): count")
    for k, v in sorted(gated.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
        top = sorted(v.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {k[0]:<24}{k[1]:>4}: " + ", ".join(f"{t[0]}/{t[1]}/{t[2]}: {c}" for t, c in top))


main()
