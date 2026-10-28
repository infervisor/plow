"""p0_bw_sum.py <dir>: tiers.jsonl -> markdown table (mean of reps, spread); imc.csv; turbostat summaries."""
import json, sys, collections, re, os, statistics as st
d = sys.argv[1]
rows = [json.loads(l) for l in open(f'{d}/tiers.jsonl') if l.startswith('{')]
g = collections.defaultdict(list)
for r in rows:
    g[(r['huge'], r['bytes_per_thread'], r['mode'])].append(r)
print("| pages | KiB/core | mode | total GB/s (mean of reps) | rep spread | per-core mean | p5 | min (cpu) |")
print("|---|---:|---|---:|---:|---:|---:|---|")
for (h, sz, mode), rs in sorted(g.items()):
    t = [r['total'] for r in rs]
    m = st.mean(t)
    print(f"| {'THP' if h else '4K'} | {sz >> 10} | {mode} | {m:,.0f} | {100 * (max(t) - min(t)) / m:.1f}% | {st.mean(r['per_core_mean'] for r in rs):.1f} "
          f"| {min(r['p5'] for r in rs):.1f} | {min(r['min'] for r in rs):.1f} ({min(rs, key=lambda r: r['min'])['min_cpu']}) |")
print()
try:
    run = json.loads(open(f'{d}/imc_run.jsonl').readline())
    mib = {}
    for l in open(f'{d}/imc.csv'):
        f = l.strip().split(',')
        if len(f) > 3 and 'cas_count' in f[2]:
            mib[f[2]] = float(f[0])
    secs = None
    for l in open(f'{d}/imc.csv'):
        if 'seconds time elapsed' in l:
            secs = float(l.split()[0])
    rd = sum(v for k, v in mib.items() if 'read' in k) * 1048576 / 1e9
    wr = sum(v for k, v in mib.items() if 'write' in k) * 1048576 / 1e9
    print(f"IMC during 64 MiB/core read: client {run['total']:.0f} GB/s; IMC read {rd:.1f} GB, write {wr:.1f} GB total over the run {mib}")
except Exception as e:
    print("imc parse:", e)
print()
for f in sorted(os.listdir(d)):
    if not f.startswith('turbostat'):
        continue
    mhz = collections.defaultdict(list)
    pkgw = []
    tmp = []
    hdr = None
    for l in open(f'{d}/{f}'):
        p = l.split()
        if not p:
            continue
        if p[0] == 'CPU':
            hdr = p
            continue
        if hdr is None or len(p) != len(hdr):
            continue
        r = dict(zip(hdr, p))
        if r['CPU'] == '-':
            if 'PkgWatt' in r and r['PkgWatt'] != '-':
                pkgw.append(float(r['PkgWatt']))
            continue
        c = int(r['CPU'])
        if c < 96 and c not in (0, 1, 32, 33, 64, 65) and float(r['Busy%']) > 90:
            mhz[c].append(float(r['Bzy_MHz']))
            if r.get('CoreTmp', '-') not in ('-', ''):
                tmp.append(float(r['CoreTmp']))
    allv = [v for vs in mhz.values() for v in vs]
    if allv:
        print(f"{f}: busy workers {len(mhz)}, Bzy_MHz mean {st.mean(allv):.0f} min {min(allv):.0f} max {max(allv):.0f}; "
              f"PkgWatt mean {st.mean(pkgw) if pkgw else float('nan'):.0f}; CoreTmp max {max(tmp) if tmp else float('nan'):.0f}")
    else:
        print(f"{f}: no busy samples")
for l in open(f'{d}/freq_runs.jsonl'):
    r = json.loads(l)
    print(f"  {r['mode']}: total {r['total']:.0f} {r['unit']}, per core {r['per_core_mean']:.1f}, min {r['min']:.1f}")
