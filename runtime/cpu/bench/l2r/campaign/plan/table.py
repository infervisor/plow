"""table.py: markdown plan table from run.sh outputs (<model>.b<B>.c<ctx>.txt summary lines)."""
import re
print("| model | B | ctx | stages | layers / stage | bottleneck stage µs | token latency ms | tok/s (sequences in flight) |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for m in ("E2B", "E4B"):
    for b in (1, 16):
        for c in (2048, 16384):
            s = open(f"/tmp/g4c/l2r/plan/{m}.b{b}.c{c}.txt").read().strip().splitlines()[-1]
            g = re.match(r"(\d+) stages for (\d+) layers \(([\d.]+) layers/stage\), bottleneck ([\d.]+) us, token latency (\d+) us, "
                         r"(\d+) tok/s at (\d+) sequences", s)
            n, _, lps, bn, lat, tps, seq = g.groups()
            print(f"| {m} | {b} | {c // 1024}K | {n} | {lps} | {float(bn):.1f} | {int(lat) / 1000:.1f} | {int(tps):,} ({seq}) |")
