#!/usr/bin/env python3
"""px11_sass.py <sass.txt> <name-substr> — instruction census of the matching device functions.
Counts global/shared load WIDTHS, which is the thing the coordinator asked to check before
theorising about why the fp8 KV read is slow."""
import re, sys, collections
t = open(sys.argv[1]).read()
want = sys.argv[2]
for f in re.split(r'\n\s*Function : ', t):
    name = f.split('\n')[0].strip()
    if want not in name:
        continue
    c = collections.Counter()
    for m in re.finditer(r'^\s+/\*[0-9a-f]+\*/\s+@?!?P?\d?\s*([A-Z][A-Z0-9_.]*)', f, re.M):
        op = m.group(1)
        base = op.split('.')[0]
        if base in ('LDG', 'LDS', 'STS', 'STG', 'LDSM', 'BAR', 'BSSY', 'BSYNC'):
            c[op] += 1
        else:
            c['~' + base] += 1
    tot = sum(v for k, v in c.items())
    print(f'{name}  [{tot} instrs]')
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        if not k.startswith('~') or v >= 20:
            print(f'    {k:28s} {v}')
    print()
