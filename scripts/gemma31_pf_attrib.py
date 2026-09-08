#!/usr/bin/env python3
"""Attribute one prefill chunk's GPU envelope to ops, using the raw PlowTraceRec dump
plus `plowrt disasm --format json` for op/tensor identity.

  PlowTraceRec = { u32 cu, u32 pc, u32 inst, u16 op, u16 slice, u64 t_arrive, u64 t_ready, u64 t_end }

Ticks are s_memrealtime, 100 MHz (10 ns).  Packets are walked in INSTRUCTION order against a
monotone clock, exactly as scripts/k3_trace_report.py does, so overlapping independent packets
are not double counted:

    start_i = max(end_{i-1}, arrive_i)
    gate_i  = max(0, ready_i - start_i)      # dependency wait + protocol
    body_i  = max(0, end_i - max(ready_i, start_i))
"""
import json, struct, sys, collections

REC = struct.Struct("<IIIHHQQQ")
TPMS = 100_000.0  # ticks per millisecond


def load_disasm(path):
    txt = open(path).read()
    d = json.loads(txt[txt.index('{\n  "blob"'):])
    return d['programs'][0]['insts']


def main():
    trace, dis = sys.argv[1], sys.argv[2]
    insts = load_disasm(dis)
    buf = open(trace, 'rb').read()
    n = len(buf) // REC.size
    arrive = {}
    ready = {}
    end = {}
    wg = collections.Counter()
    for i in range(n):
        cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(buf, i * REC.size)
        if te == 0:
            continue
        wg[inst] += 1
        if inst not in arrive or ta < arrive[inst]:
            arrive[inst] = ta
        if inst not in ready or tr > ready[inst]:
            ready[inst] = tr
        if inst not in end or te > end[inst]:
            end[inst] = te

    def label(i):
        e = insts[i]
        nm = e['op_name']
        ints = {x['name']: x['value'] for x in e['ints']}
        w = [t['tensor'] for t in e['tensors']
             if t['present'] and not t['tensor'].startswith(('act.', 'in.', 'kv.'))]
        role = ''
        if w:
            t0 = w[0]
            for k in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj',
                      'down_proj', 'embed_tokens'):
                if k in t0:
                    role = k
                    break
        shape = ''
        if 'N' in ints and 'K' in ints:
            shape = f" N={ints['N']} K={ints['K']}"
        return f"{nm}{'/' + role if role else ''}{shape}"

    clock = 0
    rows = collections.defaultdict(lambda: [0, 0, 0, 0])  # n, gate, body, blocks
    tot_gate = tot_body = 0
    lo = min(arrive.values())
    for i in range(len(insts)):
        if i not in end:
            continue
        s = max(clock, arrive[i])
        g = max(0, ready[i] - s)
        b = max(0, end[i] - max(ready[i], s))
        clock = max(clock, end[i])
        k = label(i)
        r = rows[k]
        r[0] += 1
        r[1] += g
        r[2] += b
        r[3] += wg[i]
        tot_gate += g
        tot_body += b
    span = (clock - lo) / TPMS
    print(f"packets={len(rows)} groups, records={n}, chain span={span:.2f} ms, "
          f"gate={tot_gate/TPMS:.2f} ms ({100*tot_gate/(tot_gate+tot_body):.1f}%), "
          f"body={tot_body/TPMS:.2f} ms")
    print()
    h = f"{'op / role / shape':<46}{'n':>5}{'wg/pkt':>8}{'gate_ms':>9}{'body_ms':>9}{'tot_ms':>9}{'%':>7}{'us/pkt':>8}"
    print(h)
    print('-' * len(h))
    tot = tot_gate + tot_body
    for k in sorted(rows, key=lambda k: -(rows[k][1] + rows[k][2])):
        c, g, b, blk = rows[k]
        t = g + b
        print(f"{k:<46}{c:>5}{blk // c:>8}{g / TPMS:>9.2f}{b / TPMS:>9.2f}{t / TPMS:>9.2f}"
              f"{100 * t / tot:>6.1f}%{t / c / 100.0:>8.1f}")
    print('-' * len(h))
    print(f"{'TOTAL':<46}{'':>5}{'':>8}{tot_gate / TPMS:>9.2f}{tot_body / TPMS:>9.2f}{tot / TPMS:>9.2f}")


main()
