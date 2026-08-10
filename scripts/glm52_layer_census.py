#!/usr/bin/env python3
"""Per-op packet census for ONE representative layer of a GLM trace (prefill or decode).

Generalises scripts/glm52_decode_census.py to (a) prefill programs and (b) an
arbitrary layer window, so the 3 DENSE layers (0..2) and the 75 MoE layers
(3..77) can be decomposed separately.

    plowrt disasm <blob> --program <P> | grep '^#' > prog.txt
    PLOW_TRACE_RAW=tr.bin plowrt amd-bench --blob ... --tp 8 --prompt "<T tokens>"
    python3 scripts/glm52_layer_census.py prog.txt tr.bin.prefill --layers 8:70

GROUPED BY `inst`, NOT `pc`.  `pc` is the per-workgroup stream slot: grouping by
it shreds one packet across up to 304 rows.  Record = 40B <IIIHHQQQ>
cu,pc,inst,op,slice,arrive,ready,end.

Per packet we report
    span      = max(end) - min(ready)            wallclock the packet occupied
    busy      = sum(end - ready)   over CUs      CU-us actually inside the kernel
    gatewait  = sum(ready - arrive) over CUs     CU-us a claimant sat on the gate
    gap       = min(ready) - max(prev end)       serial packet-boundary dead time

`busy` here is "in-packet CU time"; the campaign's gate-wait share is
gatewait / (busy + gatewait).
"""
import sys, re, struct, json, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("disasm")
ap.add_argument("trace")
ap.add_argument("--layers", default="6:74", help="inclusive layer window a:b")
ap.add_argument("--ghz", type=float, default=0.1)
ap.add_argument("--json", default=None)
ap.add_argument("--timeline", type=int, default=None, help="layer to print packet-by-packet")
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--last-dispatch", action="store_true",
                help="keep only records after the largest time gap. REQUIRED for a decode "
                     "trace taken in a --prompt run: the buffer is indexed per "
                     "(workgroup, packet) and the decode program touches far fewer slots "
                     "than prefill did, so the tail of the file is stale PREFILL records "
                     "mapped onto decode `inst` indices.")
a = ap.parse_args()

LO, HI = (int(x) for x in a.layers.split(":"))

insts, layer_of = {}, {}
cur = -1
for line in open(a.disasm):
    m = re.match(r'#(\d+)\s+(\S+)\s+b=(\d+)\s*(.*)', line)
    if not m:
        continue
    i, op, b, rest = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    lm = re.search(r'layers\.(\d+)\.input_layernorm', rest)
    if lm:
        cur = int(lm.group(1))
    insts[i] = (op, b, rest.strip()[:110])
    layer_of[i] = cur

data = open(a.trace, 'rb').read()
n = len(data) // 40

t_min = 0
if a.last_dispatch:
    ready = sorted(struct.unpack_from('<Q', data, k * 40 + 24)[0] for k in range(n)
                   if struct.unpack_from('<Q', data, k * 40 + 32)[0] or
                      struct.unpack_from('<Q', data, k * 40 + 24)[0])
    gap, at = max(((ready[i + 1] - ready[i], i) for i in range(len(ready) - 1)),
                  default=(0, 0))
    t_min = ready[at + 1]
    print(f"--last-dispatch: {gap/1e2:.0f} us idle gap splits the file; keeping the "
          f"{len(ready)-at-1} records after it (dropped {at+1} stale)")

per = defaultdict(lambda: dict(busy=0, wait=0, cnt=0, t0=None, t1=0))
nrec = 0
for k in range(n):
    cu, pc, inst, op, slc, ta, tr, te = struct.unpack_from('<IIIHHQQQ', data, k * 40)
    if te == 0 and tr == 0:
        continue
    if tr < t_min:
        continue
    nrec += 1
    r = per[inst]
    r['busy'] += te - tr
    r['wait'] += tr - ta
    r['cnt'] += 1
    r['t0'] = tr if r['t0'] is None else min(r['t0'], tr)
    r['t1'] = max(r['t1'], te)

tick = 1e-3 / a.ghz                       # cycles -> us
lay = defaultdict(list)
for i in per:
    L = layer_of.get(i, -2)
    if isinstance(L, int) and LO <= L <= HI:
        lay[L].append(i)
sel_layers = sorted(lay)
if not sel_layers:
    print(f"NO LAYERS IN WINDOW {LO}:{HI} (layers present: "
          f"{sorted(set(layer_of[i] for i in per))[:20]} ...)")
    sys.exit(1)
nl = len(sel_layers)

spans = sorted((max(per[i]['t1'] for i in v) - min(per[i]['t0'] for i in v)) * tick
               for v in lay.values())
lay_span = spans[len(spans) // 2]

agg = defaultdict(lambda: dict(busy=0, wait=0, span=0, pk=0, wg=0, cnt=0))
for L in sel_layers:
    for i in lay[L]:
        op, b, _ = insts[i]
        A, r = agg[op], per[i]
        A['busy'] += r['busy']; A['wait'] += r['wait']; A['cnt'] += r['cnt']
        A['span'] += (r['t1'] - r['t0']); A['pk'] += 1; A['wg'] += b

gaps = []
gap_by_op = defaultdict(float)
for L in sel_layers:
    ids = sorted(lay[L])
    prev, tot = None, 0.0
    for i in ids:
        r = per[i]
        if prev is not None:
            g = (r['t0'] - prev) * tick
            if g > 0:
                tot += g
                gap_by_op[insts[i][0]] += g
        prev = r['t1']
    gaps.append(tot)
gaps.sort()

tot_busy = sum(A['busy'] for A in agg.values()) * tick / nl
tot_wait = sum(A['wait'] for A in agg.values()) * tick / nl
tot_span = sum(A['span'] for A in agg.values()) * tick / nl

if not a.quiet:
    print(f"{a.trace}  records={nrec}  packets={len(per)}  layers {sel_layers[0]}..{sel_layers[-1]} (n={nl})")
    print(f"layer span (median)          {lay_span:10.1f} us")
    print(f"sum of packet spans          {tot_span:10.1f} us   (serial chain)")
    print(f"packet-boundary gap (median) {gaps[len(gaps)//2]:10.1f} us")
    print(f"packets / layer              {sum(A['pk'] for A in agg.values())/nl:10.2f}\n")
    print(f"{'op':<24}{'pkt/L':>7}{'wg':>6}{'span us':>10}{'%span':>7}"
          f"{'busy CUus':>11}{'wait CUus':>11}{'wait%':>7}{'gap us':>8}")
    for op, A in sorted(agg.items(), key=lambda kv: -kv[1]['span']):
        sp = A['span'] * tick / nl
        bu = A['busy'] * tick / nl
        wa = A['wait'] * tick / nl
        print(f"{op:<24}{A['pk']/nl:>7.2f}{A['wg']/A['pk']:>6.0f}{sp:>10.1f}"
              f"{100*sp/tot_span:>7.1f}{bu:>11.0f}{wa:>11.0f}"
              f"{100*wa/max(bu+wa,1e-9):>7.1f}{gap_by_op[op]/nl:>8.2f}")
    print(f"{'TOTAL':<24}{sum(A['pk'] for A in agg.values())/nl:>7.2f}{'':>6}{tot_span:>10.1f}"
          f"{100.0:>7.1f}{tot_busy:>11.0f}{tot_wait:>11.0f}"
          f"{100*tot_wait/max(tot_busy+tot_wait,1e-9):>7.1f}"
          f"{sum(gaps)/nl:>8.2f}")
    print(f"\ngate-wait share of in-packet CU time: "
          f"{100*tot_wait/max(tot_busy+tot_wait,1e-9):.1f}%")
    # ---- CU-time budget of the WHOLE layer: how much of the machine did real work?
    NCU = 304
    avail = lay_span * NCU
    other = avail - tot_busy - tot_wait
    print(f"\nlayer CU-time budget ({NCU} CU x {lay_span:.0f} us = {avail:.0f} CU-us):")
    print(f"  real work (busy)          {tot_busy:12.0f} CU-us  {100*tot_busy/avail:5.1f}%")
    print(f"  in-packet gate wait       {tot_wait:12.0f} CU-us  {100*tot_wait/avail:5.1f}%")
    print(f"  outside any packet        {other:12.0f} CU-us  {100*other/avail:5.1f}%")
    print(f"  perfect-pack floor        {tot_busy/NCU:12.0f} us     "
          f"(packing efficiency {100*tot_busy/NCU/lay_span:.1f}%)")

if a.timeline is not None and a.timeline in lay:
    print(f"\n--- layer {a.timeline} timeline ---")
    print(f"{'#inst':<8}{'op':<24}{'span':>9}{'wg':>6}{'busy':>10}{'wait':>10}{'gap':>8}  detail")
    prev = None
    for i in sorted(lay[a.timeline]):
        op, b, det = insts[i]
        r = per[i]
        g = (r['t0'] - prev) * tick if prev is not None else 0.0
        prev = r['t1']
        print(f"#{i:<7}{op:<24}{(r['t1']-r['t0'])*tick:>9.2f}{b:>6}"
              f"{r['busy']*tick:>10.1f}{r['wait']*tick:>10.1f}{g:>8.2f}  {det[:60]}")

if a.json:
    out = dict(trace=a.trace, layers=[sel_layers[0], sel_layers[-1]], n_layers=nl,
               layer_span_us=lay_span, sum_span_us=tot_span,
               gap_us=gaps[len(gaps) // 2],
               pkts_per_layer=sum(A['pk'] for A in agg.values()) / nl,
               busy_cuus=tot_busy, wait_cuus=tot_wait,
               ops={op: dict(pk=A['pk'] / nl, span=A['span'] * tick / nl,
                             busy=A['busy'] * tick / nl, wait=A['wait'] * tick / nl,
                             gap=gap_by_op[op] / nl)
                    for op, A in agg.items()})
    json.dump(out, open(a.json, 'w'), indent=1)
