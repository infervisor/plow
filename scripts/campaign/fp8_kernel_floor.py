"""Roofline FLOOR totals per phase, fp8 vs bf16, from the same dispatch_audit rows.

Per (op,N,K) take the widest rung's row, compute the per-instruction floor
  max(bytes/BW, 2*m*n*k/peak(prec))
and multiply by `insts` (layer multiplicity). Sum. lm_head (N>=200k) is split out because it is
BF16 in BOTH packets and is the single largest audited decode kernel.

This is a FLOOR, not a prediction: it covers only the audited linear ops. The MoE expert GEMMs are
NOT audited at all (dispatch_audit's MoE blind spot), nor are attention/norms/rope.
"""
import collections
import json

BW = 3352e9
PEAK = {"bf16": 989e12, "fp8": 1979e12}
Q = ("Fp8", "W8a8", "W8a16")

PKTS = [
    ("26B fp8", "/opt/dlami/nvme/tmp/agent-geom/p26fp8/assets/build.json", "fp8"),
    ("26B bf16", "/opt/dlami/nvme/tmp/agent-geom/p26lad2/assets/build.json", "bf16"),
    ("12B fp8", "/opt/dlami/nvme/tmp/agent-geom/p12fp8c/assets/build.json", "fp8"),
]

out = {}
for tag, path, pprec in PKTS:
    d = json.load(open(path))
    ops = (d.get("dispatch_audit") or {}).get("ops") or []
    for kind in ("prefill", "decode"):
        g = collections.defaultdict(list)
        for o in ops:
            if o.get("kind") == kind:
                g[(o["op"], o.get("n"), o.get("k"))].append(o)
        lm = rest = 0.0
        for key, rs in g.items():
            op, n, k = key
            big = max(rs, key=lambda x: x.get("bytes", 0))
            by = big.get("bytes") or 0
            m = big.get("m") or 0
            fl = 2 * m * (n or 0) * (k or 0)
            pr = "fp8" if any(q in op for q in Q) else "bf16"
            f = max(by / BW, fl / PEAK[pr]) * 1e3 * big.get("insts", 1)
            if (n or 0) >= 200000:
                lm += f
            else:
                rest += f
        out[(tag, kind)] = (lm, rest)

print("Audited-linear roofline FLOOR (ms). lm_head is bf16 in every packet here.")
print(f"{'packet':<10} {'phase':<8} {'lm_head':>9} {'other lin':>10} {'total':>9}")
for (tag, kind), (lm, rest) in out.items():
    print(f"{tag:<10} {kind:<8} {lm:>9.3f} {rest:>10.3f} {lm+rest:>9.3f}")

print()
for kind in ("prefill", "decode"):
    a = out[("26B bf16", kind)]
    b = out[("26B fp8", kind)]
    print(f"26B {kind:<8}: other-linear floor bf16 {a[1]:.3f} -> fp8 {b[1]:.3f} ms "
          f"= {a[1]/b[1] if b[1] else 0:.2f}x better; lm_head unchanged at {a[0]:.3f} ms "
          f"({100*a[0]/(a[0]+a[1]):.0f}% of the bf16 total, {100*b[0]/(b[0]+b[1]):.0f}% of the fp8 one)")

print()
print("#93 (quantise lm_head) is worth HALF its floor, since at AI ~1-16 it is memory bound:")
for tag in ("26B fp8", "12B fp8"):
    d_lm = out[(tag, "decode")][0]
    p_lm = out[(tag, "prefill")][0]
    print(f"  {tag:<9} decode {d_lm:.3f} -> ~{d_lm/2:.3f} ms per STEP (saves {d_lm/2:.3f}); "
          f"prefill {p_lm:.3f} -> ~{p_lm/2:.3f} ms per REQUEST")
print()
print("Measured 26B C1 TPOT gap vs vLLM = 0.42 ms/step of context-independent work.")
print(f"  lm_head at fp8 would save ~{out[('26B fp8','decode')][0]/2:.3f} ms/step = "
      f"{100*(out[('26B fp8','decode')][0]/2)/0.42:.0f}% of that gap, from ONE change.")
