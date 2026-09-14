import csv, sys, collections
rows = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if l.count(',') >= 9)]
# per (rows, ctx): arm -> us
cells = collections.defaultdict(dict)
for r in rows:
    key = (int(r['rows']), int(r['ctx']))
    arm = r['arm'] + (f"@ns{r['ns']}" if r['arm'] in ('split4_v2', 'aiter_qh8_kernel', 'aiter_qh8_pack+kernel+reduce') else '')
    cells[key][arm] = float(r['median_us'])
    if 'FAIL' in r['note']:
        print('FAIL', r)
arms = sorted({a for d in cells.values() for a in d}, key=lambda a: (a.split('@')[0], int(a.split('@ns')[1]) if '@ns' in a else 0))
print('rows,ctx,' + ','.join(arms))
for key in sorted(cells):
    print(f"{key[0]},{key[1]}," + ','.join(f"{cells[key].get(a, float('nan')):.0f}" if a in cells[key] else '' for a in arms))
# roofline columns for dense: causal FLOPs = rows * (ctx - rows/2) * 8 heads * (576+512) * 2
print()
print('dense efficiency (TFLOP/s) per arm; peak bf16 MFMA 1307 TF/s')
for key in sorted(cells):
    R, C = key
    flop = R * (C - R / 2) * 8 * (576 + 512) * 2
    line = []
    for a in arms:
        if a.startswith(('broad8', 'small8', 'flash4_v2', 'split4')) and a in cells[key]:
            line.append(f"{a}={flop / cells[key][a] / 1e6:.0f}")
    print(R, C, ' '.join(line))
