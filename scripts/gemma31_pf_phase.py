#!/usr/bin/env python3
"""Five-phase packet decomposition from a PLOW_TRACE_PHASE=1 prefill trace.

  claim+gate      = pc & 0xffff
  acquire+rdv     = (t_ready - t_arrive) - (pc & 0xffff)
  op body         = pc >> 16
  publish+release = (t_end - t_ready) - (pc >> 16)
"""
import json, struct, sys, collections

REC = struct.Struct("<IIIHHQQQ")
T = 100.0  # ticks/us

trace, dis = sys.argv[1], sys.argv[2]
txt = open(dis).read()
insts = json.loads(txt[txt.index('{\n  "blob"'):])['programs'][0]['insts']
buf = open(trace, 'rb').read()

# Per (op, inst) accumulate the MAX over workgroups of each phase, then average per packet.
per = collections.defaultdict(lambda: collections.defaultdict(list))
for i in range(len(buf) // REC.size):
    cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(buf, i * REC.size)
    if te == 0:
        continue
    gate = pc & 0xFFFF
    acq = max(0, (tr - ta) - gate)
    body = pc >> 16
    pub = max(0, (te - tr) - body)
    per[insts[inst]['op_name']][inst].append((gate, acq, body, pub))

rows = []
for name, byinst in per.items():
    n = len(byinst)
    tot = [0.0] * 4
    for inst, recs in byinst.items():
        for k in range(4):
            tot[k] += max(r[k] for r in recs)
    rows.append((name, n, [t / n / T for t in tot], sum(tot) / n / T))
rows.sort(key=lambda r: -r[3] * r[1])

h = f"{'op':<16}{'pkts':>6}{'claim+gate':>12}{'acquire':>10}{'body':>10}{'publish':>10}{'us/pkt':>9}{'ms tot':>9}"
print(h)
print('-' * len(h))
tt = [0.0] * 4
for name, n, ph, s in rows:
    for k in range(4):
        tt[k] += ph[k] * n
    print(f"{name:<16}{n:>6}{ph[0]:>12.1f}{ph[1]:>10.1f}{ph[2]:>10.1f}{ph[3]:>10.1f}{s:>9.1f}{s*n/1000:>9.2f}")
print('-' * len(h))
print(f"{'TOTAL ms':<16}{'':>6}{tt[0]/1000:>12.2f}{tt[1]/1000:>10.2f}{tt[2]/1000:>10.2f}{tt[3]/1000:>10.2f}"
      f"{'':>9}{sum(tt)/1000:>9.2f}")
print("\n(max over workgroups per packet, summed over packets; overlapping packets are NOT"
      " deconflicted, so the total exceeds the chain envelope.)")
