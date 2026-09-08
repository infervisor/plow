#!/usr/bin/env python3
"""Decode-step residue attribution from a PLOW_TRACE_PHASE=1 trace.

    plowrt disasm <blob> --no-analysis > dis.txt
    PLOW_TRACE_RAW=tr.bin plowrt amd-bench --blob <blob> --hsaco <tracedir> \
        --checkpoint <ckpt> --batched --steps 24 --ctx 1024
    python3 scripts/gemma31_dec_phase.py dis.txt tr.bin [program-label]

The trace buffer holds the LAST launch only, so this is ONE steady-state decode step.

Per (workgroup, packet) the instrument records four phases (interp.hip PLOW_TRACE_PHASE):

    claim+gate      = pc & 0xffff          dependency wait -- belongs to the producer
    acquire+rdv     = (t_ready-t_arrive) - claim+gate
    op body         = pc >> 16
    publish+release = (t_end-t_ready) - body

TWO TABLES, and the second is the one that adds up.

RAW is max-over-workgroups per packet, summed over packets. It over-counts: decode packets
OVERLAP (q/k/v run concurrently on disjoint CU sets, and a packet's gate wait runs under its
producer's body), so the raw total exceeds the step envelope.

CRITICAL PATH walks the packets in program order keeping a frontier F = max end so far.
A packet contributes `end - max(F, first_arrival)`; the sum is exactly the covered envelope.
Inside that contribution the phases of the packet's SLOWEST workgroup (the one that closes it)
are charged innermost-first: publish, then body, then whatever is left is gate/dependency.
That is the split the residue question asks for -- what a millisecond of the step was spent on
when nothing else could proceed.
"""
import sys, re, struct, collections

REC = struct.Struct("<IIIHHQQQ")
T = 100.0  # ticks per microsecond (100 MHz s_memrealtime)

dis, tracef = sys.argv[1], sys.argv[2]
want = sys.argv[3] if len(sys.argv) > 3 else "T=4"

insts, cur = {}, None
for line in open(dis):
    m = re.match(r"===== program (\S+)", line)
    if m:
        cur = m.group(1)
        continue
    if cur != want:
        continue
    m = re.match(r"#(\d+)\s+(\S+)\s+b=(\d+)", line)
    if m:
        insts[int(m.group(1))] = (m.group(2), int(m.group(3)))
if not insts:
    sys.exit(f"no program {want!r} in {dis}")

buf = open(tracef, "rb").read()
per = collections.defaultdict(list)
for i in range(len(buf) // REC.size):
    cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(buf, i * REC.size)
    if te == 0:
        continue
    gate = pc & 0xFFFF
    acq = max(0, (tr - ta) - gate)
    body = pc >> 16
    pub = max(0, (te - tr) - body)
    per[inst].append((gate, acq, body, pub, ta, te))
if not per:
    sys.exit("trace has no completed records")

# ---------------- raw: max over workgroups, mean over packets
byop = collections.defaultdict(lambda: dict(n=0, ph=[0.0] * 4, wg=0))
for inst, recs in per.items():
    name, _ = insts.get(inst, ("?%d" % inst, 0))
    e = byop[name]
    e["n"] += 1
    e["wg"] += len(recs)
    for k in range(4):
        e["ph"][k] += max(r[k] for r in recs)

rows = sorted(
    ((n, e["n"], e["wg"] // e["n"], [x / e["n"] / T for x in e["ph"]]) for n, e in byop.items()),
    key=lambda r: -sum(r[3]) * r[1],
)
h = (f"{'op':<18}{'pkts':>5}{'wg':>5}{'claim+gate':>11}{'acquire':>9}"
     f"{'body':>8}{'publish':>9}{'us/pkt':>8}{'ms raw':>9}")
print("RAW  (max over workgroups per packet; packets overlap, so this over-counts)")
print(h)
print("-" * len(h))
tot = [0.0] * 4
for name, n, wg, ph in rows:
    for k in range(4):
        tot[k] += ph[k] * n
    s = sum(ph)
    print(f"{name:<18}{n:>5}{wg:>5}{ph[0]:>11.1f}{ph[1]:>9.1f}{ph[2]:>8.1f}"
          f"{ph[3]:>9.1f}{s:>8.1f}{s*n/1000:>9.3f}")
print("-" * len(h))
npk = sum(r[1] for r in rows)
print(f"{'TOTAL ms':<18}{npk:>5}{'':>5}{tot[0]/1000:>11.3f}{tot[1]/1000:>9.3f}"
      f"{tot[2]/1000:>8.3f}{tot[3]/1000:>9.3f}{'':>8}{sum(tot)/1000:>9.3f}")

# ---------------- critical path
order = sorted(per.keys())
crit = collections.defaultdict(lambda: dict(n=0, gate=0.0, body=0.0, pub=0.0))
F = min(min(r[4] for r in recs) for recs in per.values())
lo, hi = F, 0
for inst in order:
    recs = per[inst]
    a = min(r[4] for r in recs)
    slow = max(recs, key=lambda r: r[5])  # the workgroup that closes the packet
    e = slow[5]
    hi = max(hi, e)
    start = max(F, a)
    c = max(0, e - start)
    F = max(F, e)
    pub = min(c, slow[3])
    body = min(c - pub, slow[2])
    gate = c - pub - body
    name, _ = insts.get(inst, ("?%d" % inst, 0))
    d = crit[name]
    d["n"] += 1
    d["gate"] += gate
    d["body"] += body
    d["pub"] += pub

print()
print("CRITICAL PATH  (program order, frontier-clipped; the columns sum to the step)")
h2 = (f"{'op':<18}{'pkts':>5}{'gate ms':>10}{'body ms':>10}{'publish ms':>12}"
      f"{'total ms':>10}{'share':>8}{'us/pkt':>8}")
print(h2)
print("-" * len(h2))
env = hi - lo
crows = sorted(crit.items(), key=lambda kv: -(kv[1]["gate"] + kv[1]["body"] + kv[1]["pub"]))
tg = tb = tp = 0.0
for name, d in crows:
    s = d["gate"] + d["body"] + d["pub"]
    tg, tb, tp = tg + d["gate"], tb + d["body"], tp + d["pub"]
    print(f"{name:<18}{d['n']:>5}{d['gate']/T/1000:>10.3f}{d['body']/T/1000:>10.3f}"
          f"{d['pub']/T/1000:>12.3f}{s/T/1000:>10.3f}{100*s/env:>7.1f}%{s/d['n']/T:>8.1f}")
print("-" * len(h2))
tt = tg + tb + tp
print(f"{'TOTAL':<18}{npk:>5}{tg/T/1000:>10.3f}{tb/T/1000:>10.3f}{tp/T/1000:>12.3f}"
      f"{tt/T/1000:>10.3f}{100*tt/env:>7.1f}%")
print(f"{'step envelope':<18}{'':>5}{'':>10}{'':>10}{'':>12}{env/T/1000:>10.3f}")
print(f"\nworkgroup-records {sum(len(v) for v in per.values())}, packets {npk}")
print(f"protocol on the critical path: gate {100*tg/env:.1f}% + publish {100*tp/env:.1f}% "
      f"= {100*(tg+tp)/env:.1f}% of the step ({(tg+tp)/T/1000:.3f} ms)")
