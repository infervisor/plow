import json, statistics as st
O = "/tmp/g4c/l2r/results/p1_long"
out = ["| run | secs | GB/s/core mean (min core) | worst p99 µs | max step µs (worst core) | step µs first 10% → last 10% (median core) | worst core drift | held L2 before → after |",
       "|---|---:|---|---:|---:|---|---:|---|"]
for tag, name in (("l0.600", "plain"), ("l0.1800", "plain"), ("l1.600", "locked")):
    rows = [json.loads(l) for l in open(f"{O}/{tag}.jsonl")]
    s = [r for r in rows if r.get("summary")][0]
    c = [r for r in rows if "cpu" in r]
    drift = [(r["last_us"] - r["first_us"]) / r["first_us"] * 100 for r in c]
    held = "n/a" if s["held_l2_before"] < 0 else f"{s['held_l2_before']:.4f} → {s['held_l2_after']:.4f}"
    out.append(f"| {name} | {s['secs']} | {s['per_core_mean']:.2f} ({s['min_gbs']:.2f}, cpu {s['min_cpu']}) | {s['worst_p99_us']:.2f} | "
               f"{max(r['max_us'] for r in c):.0f} | {st.median(r['first_us'] for r in c):.2f} → {st.median(r['last_us'] for r in c):.2f} | "
               f"{max(drift, key=abs):+.1f}% | {held} |")
open(f"{O}/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
