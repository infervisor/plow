#!/usr/bin/env python3
"""Critical-path contribution per PROJECTION, not per opcode.

Same frontier walk as `gemma31_dec_phase.py`, but packets are keyed by the weight tensor
suffix (q_proj / k_proj / ... / lm_head) so the concurrent q|k|v and gate|up pairs can be
told apart -- which is what decides whether fusing them can save anything.

    python3 scripts/gemma31_dec_phase_proj.py dis.txt tr.bin [program-label]
"""
import sys, re, struct, collections

REC = struct.Struct("<IIIHHQQQ")
T = 100.0

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
    m = re.match(r"#(\d+)\s+(\S+)\s+b=(\d+)\s*(.*)", line)
    if not m:
        continue
    op, b, rest = m.group(2), int(m.group(3)), m.group(4)
    w = re.search(r"(?:W|B|table)<-\S*?\.([a-z_0-9]+)\.weight", rest)
    key = f"{op}:{w.group(1)}" if w else op
    insts[int(m.group(1))] = (key, b)

buf = open(tracef, "rb").read()
per = collections.defaultdict(list)
for i in range(len(buf) // REC.size):
    cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(buf, i * REC.size)
    if te:
        per[inst].append((pc & 0xFFFF, max(0, (tr - ta) - (pc & 0xFFFF)), pc >> 16,
                          max(0, (te - tr) - (pc >> 16)), ta, te))

crit = collections.defaultdict(lambda: dict(n=0, gate=0.0, body=0.0, pub=0.0, b=0))
F = min(min(r[4] for r in v) for v in per.values())
lo, hi = F, 0
for inst in sorted(per):
    recs = per[inst]
    a = min(r[4] for r in recs)
    slow = max(recs, key=lambda r: r[5])
    e = slow[5]
    hi = max(hi, e)
    c = max(0, e - max(F, a))
    F = max(F, e)
    pub = min(c, slow[3])
    body = min(c - pub, slow[2])
    key, b = insts.get(inst, ("?%d" % inst, 0))
    d = crit[key]
    d["n"] += 1
    d["b"] = b
    d["gate"] += c - pub - body
    d["body"] += body
    d["pub"] += pub

env = hi - lo
h = (f"{'packet'    :<28}{'pkts':>5}{'wg':>5}{'gate ms':>9}{'body ms':>9}"
     f"{'pub ms':>8}{'total':>8}{'share':>8}{'us/pkt':>8}")
print(h)
print("-" * len(h))
for key, d in sorted(crit.items(), key=lambda kv: -(kv[1]["gate"] + kv[1]["body"] + kv[1]["pub"])):
    s = d["gate"] + d["body"] + d["pub"]
    print(f"{key:<28}{d['n']:>5}{d['b']:>5}{d['gate']/T/1000:>9.3f}{d['body']/T/1000:>9.3f}"
          f"{d['pub']/T/1000:>8.3f}{s/T/1000:>8.3f}{100*s/env:>7.1f}%{s/d['n']/T:>8.1f}")
print("-" * len(h))
print(f"{'step envelope':<28}{'':>5}{'':>5}{'':>9}{'':>9}{'':>8}{env/T/1000:>8.3f}")
