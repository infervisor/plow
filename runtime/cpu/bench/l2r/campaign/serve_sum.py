"""serve_sum.py <root>: p0_serve cells -> markdown (TTFT/TPOT p50/p95/p99 pooled over reps; rep spread of
per-rep median TPOT and output tok/s; failures)."""
import json, sys, glob, os, re, statistics as st


def pct(v, q):
    v = sorted(v)
    if not v:
        return float('nan')
    k = (len(v) - 1) * q / 100
    i = int(k)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (k - i)


root = sys.argv[1]
print("| model | ISL | conc | reps | TTFT p50 / p95 / p99 ms | TPOT p50 / p95 / p99 ms | out tok/s (spread) | median TPOT spread | failed |")
print("|---|---:|---:|---:|---|---|---|---:|---:|")
for m in sorted(os.listdir(root)):
    if not os.path.isdir(f'{root}/{m}'):
        continue
    for isl_dir in sorted(glob.glob(f'{root}/{m}/isl*'), key=lambda p: int(re.sub(r'\D', '', os.path.basename(p)))):
        if not os.path.isdir(isl_dir):
            continue
        isl = os.path.basename(isl_dir)[3:]
        cells = {}
        for f in glob.glob(f'{isl_dir}/g*.r*/bench.json'):
            c = int(re.search(r'/g(\d+)\.r', f).group(1))
            cells.setdefault(c, []).append(json.load(open(f)))
        for c in sorted(cells):
            reps = cells[c]
            ttft, tpot, med, otps, fail = [], [], [], [], 0
            for d in reps:
                rt = []
                for t, lat, o in zip(d['ttfts'], d['latencies'], d['output_lens']):
                    if t is None or not o:
                        continue
                    ttft.append(t * 1e3)
                    if o > 1:
                        rt.append((lat - t) * 1e3 / (o - 1))
                tpot += rt
                med.append(st.median(rt) if rt else float('nan'))
                otps.append(d['output_throughput'])
                fail += d['failed']
            sp = lambda v: 100 * (max(v) - min(v)) / st.mean(v) if len(v) > 1 else 0
            print(f"| {m.replace('gemma-4-', '').replace('-it', '')} | {isl} | {c} | {len(reps)} | {pct(ttft, 50):,.0f} / {pct(ttft, 95):,.0f} / {pct(ttft, 99):,.0f} "
                  f"| {pct(tpot, 50):.2f} / {pct(tpot, 95):.2f} / {pct(tpot, 99):.2f} | {st.mean(otps):.0f} ({sp(otps):.1f}%) | {sp(med):.1f}% | {fail} |")
