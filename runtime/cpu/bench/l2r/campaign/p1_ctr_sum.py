import json, glob, os
O = "/tmp/g4c/l2r/results/p1_ctr"
def csv(f):
    d = {}
    for l in open(f):
        p = l.strip().split(",")
        if len(p) > 2 and p[0].replace(".", "").isdigit():
            d[p[2]] = float(p[0])
    return d
print("| scen | lock | steps/core | L2 lines in / step / core | weight lines | KV lines/step | silent / non-silent out per step | CHA SF evictions / s | GB/s/core (min) | held after |")
print("|---|---:|---:|---:|---:|---:|---|---:|---|---:|")
for sc in "ACD":
    for lk in "01":
        t = f"{O}/s1310720.l{lk}.{sc}"
        c, u = csv(t + ".core.csv"), csv(t + ".cha.csv")
        rows = [json.loads(l) for l in open(t + ".jsonl")]
        s = [r for r in rows if r.get("summary")][0]
        steps = sum(r["steps"] for r in rows if "cpu" in r)
        lin = c["r1f25"] / steps
        sil, ns = c["r0126"] / steps, c["r0226"] / steps
        kv = 4096 if sc in "CD" else 0
        sf = sum(v for k, v in u.items() if k.startswith("uncore_cha")) / 10
        print(f"| {sc} | {lk} | {steps / 90:,.0f} | {lin:,.0f} | 20,480 | {kv} | {sil:,.0f} / {ns:,.0f} | {sf:,.0f} | {s['per_core_mean']:.1f} ({s['min_gbs']:.1f}) | {s.get('held_l2_after', float('nan')):.4f} |")
