#!/usr/bin/env python3
"""Per-op packet census + gate-stall attribution for a GLM decode trace.

    plowrt disasm <blob> --program 1 | grep '^#' > dec.txt
    PLOW_TRACE_RAW=tr.bin plowrt amd-bench --blob <blob> ... --tp 8 --ctx 1024 --steps 24
    python3 scripts/glm52_decode_census.py dec.txt tr.bin [ghz]

Reports, averaged over the steady-state MoE layers: packets/layer, per-op span/busy/
gate-wait, the gate-wait share of in-packet CU time, and one layer's full packet timeline
with the gap after each packet (the serial packet-boundary dead time).

GROUPED BY `inst`, NOT `pc`. `pc` is the per-workgroup stream slot, so grouping by it
shreds one packet across up to 304 rows and every per-packet number comes out ~1/304 of
the truth. This is the same trap `trace_block.py` records; it is repeated here because
this reducer is the one the decode-fold census was taken with
(perf-data/plow-gfx942/glm52-decode-packet-folds.md).

Record: 40B <IIIHHQQQ> cu,pc,inst,op,slice,arrive,ready,end.
"""
import sys, re, struct
from collections import defaultdict

disf, tracef = sys.argv[1], sys.argv[2]
ghz = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

insts = {}
layer_of = {}
cur = -1
for line in open(disf):
    m = re.match(r'#(\d+)\s+(\S+)\s+b=(\d+)\s*(.*)', line)
    if not m: continue
    i = int(m.group(1)); op = m.group(2); b = int(m.group(3)); rest = m.group(4)
    lm = re.search(r'layers\.(\d+)\.input_layernorm', rest)
    if lm: cur = int(lm.group(1))
    insts[i] = (op, b, rest.strip()[:110])
    layer_of[i] = cur

data = open(tracef, 'rb').read()
n = len(data)//40
per = defaultdict(lambda: dict(busy=0, wait=0, cnt=0, t0=None, t1=0, arr=None))
nrec = 0
for k in range(n):
    cu, pc, inst, op, slc, ta, tr, te = struct.unpack_from('<IIIHHQQQ', data, k*40)
    if te == 0 and tr == 0: continue
    nrec += 1
    r = per[inst]
    r['busy'] += te - tr
    r['wait'] += tr - ta
    r['cnt'] += 1
    r['t0'] = tr if r['t0'] is None else min(r['t0'], tr)
    r['t1'] = max(r['t1'], te)
    r['arr'] = ta if r['arr'] is None else min(r['arr'], ta)

tick = 1e-3/ghz  # cycles -> us
# steady-state MoE layers
moe = sorted(L for L in set(layer_of.values()) if isinstance(L,int) and 6 <= L <= 74)
sel = [i for i in per if layer_of.get(i,-2) in moe]

print(f"records={nrec}  distinct packets executed={len(per)}  program insts={len(insts)}")
print(f"MoE layers sampled: {len(moe)} (L{moe[0]}..L{moe[-1]})\n")

# ---- per-layer packet census
lay = defaultdict(list)
for i in sel: lay[layer_of[i]].append(i)
sizes = sorted(len(v) for v in lay.values())
print(f"packets per MoE layer: {sizes[len(sizes)//2]} (min {sizes[0]}, max {sizes[-1]})")

spans = [(max(per[i]['t1'] for i in v) - min(per[i]['t0'] for i in v))*tick for v in lay.values()]
spans.sort()
print(f"MoE layer span: median {spans[len(spans)//2]:.1f} us  min {spans[0]:.1f}  max {spans[-1]:.1f}\n")

# ---- aggregate by op name
agg = defaultdict(lambda: dict(busy=0, wait=0, cnt=0, pk=0, wg=0, span=0))
for i in sel:
    op, b, _ = insts[i]
    a = agg[op]; r = per[i]
    a['busy'] += r['busy']; a['wait'] += r['wait']; a['cnt'] += r['cnt']
    a['pk'] += 1; a['wg'] += b
    a['span'] += (r['t1'] - r['t0'])

nl = len(moe)
tot_busy = sum(a['busy'] for a in agg.values())
tot_wait = sum(a['wait'] for a in agg.values())
print(f"{'op':<22}{'pkts/lay':>9}{'wg/pkt':>8}{'span us/lay':>13}{'busy CU-us/lay':>16}{'gate-wait CU-us/lay':>21}")
rows = sorted(agg.items(), key=lambda kv: -kv[1]['span'])
for op, a in rows:
    print(f"{op:<22}{a['pk']/nl:>9.2f}{a['wg']/a['pk']:>8.1f}"
          f"{a['span']*tick/nl:>13.2f}{a['busy']*tick/nl:>16.1f}{a['wait']*tick/nl:>21.1f}")
print(f"{'TOTAL':<22}{sum(a['pk'] for a in agg.values())/nl:>9.2f}{'':>8}"
      f"{sum(a['span'] for a in agg.values())*tick/nl:>13.2f}"
      f"{tot_busy*tick/nl:>16.1f}{tot_wait*tick/nl:>21.1f}")

print(f"\ngate-wait share of (busy+wait): {tot_wait/(tot_busy+tot_wait)*100:.1f}%")

# ---- per-packet detail for ONE median layer
Lmid = moe[len(moe)//2]
print(f"\n--- layer {Lmid} packet timeline (span us | wg | busy CU-us | gate-wait CU-us | gap from prev end) ---")
ids = sorted(lay[Lmid])
prev_end = None
tot_gap = 0
for i in ids:
    op, b, det = insts[i]
    r = per[i]
    gap = (r['t0'] - prev_end)*tick if prev_end is not None else 0.0
    tot_gap += max(gap, 0)
    prev_end = r['t1']
    print(f"#{i:<6}{op:<22}{(r['t1']-r['t0'])*tick:>8.2f}{b:>6}"
          f"{r['busy']*tick:>11.1f}{r['wait']*tick:>12.1f}{gap:>10.2f}")
print(f"serial gap total (packet-boundary dead time) = {tot_gap:.2f} us/layer")
