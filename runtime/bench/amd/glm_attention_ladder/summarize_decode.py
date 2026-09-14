import csv, sys, collections
rows = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if l.count(',') >= 10)]
fails = [r for r in rows if 'FAIL' in r['note']]
print('cells', len(rows), 'FAIL', len(fails))
for r in fails: print('FAIL', r)
def tbl(op, key_fn, filt=lambda r: True):
    cells = collections.defaultdict(dict)
    for r in rows:
        if r['op'] != op or not filt(r): continue
        cells[(int(r['rows']), int(r['ctx']))][key_fn(r)] = float(r['median_us'])
    keys = sorted({k for d in cells.values() for k in d}, key=lambda k: (k.split('@')[0], int(k.split('@ns')[1].split('/')[0]) if '@ns' in k else 0))
    print('\n#', op, '(median us per packet, one rank)')
    print('rows,ctx,' + ','.join(keys))
    for c in sorted(cells):
        print(f"{c[0]},{c[1]}," + ','.join(f"{cells[c][k]:.1f}" if k in cells[c] else '' for k in keys))
tbl('flash_sparse', lambda r: f"{r['arm']}@ns{r['ns']}")
tbl('flash_dense', lambda r: f"{r['arm']}@ns{r['ns']}/gf{r['gf']}")
tbl('merge_fold', lambda r: f"{r['arm']}@ns{r['ns']}")
tbl('index_score', lambda r: r['arm'])
tbl('index_select', lambda r: r['arm'])
tbl('kv_write_fp8', lambda r: f"{r['arm']}@blk{r['blocks']}")
