#!/usr/bin/env python3
"""splitpatch.py SRC OUT FILE... : copy unified diff SRC to OUT without the per-file sections for FILE..."""
import sys

src, out, drop = sys.argv[1], sys.argv[2], set(sys.argv[3:])
lines = open(src).read().split('\n')
blocks, cur = [], []
for i, l in enumerate(lines):
    starts = l.startswith('diff ') or (
        l.startswith('--- ') and i + 1 < len(lines) and lines[i + 1].startswith('+++ ')
        and not (cur and cur[-1].startswith('diff ')))
    if starts and cur:
        blocks.append(cur)
        cur = []
    cur.append(l)
if cur:
    blocks.append(cur)
kept, names = [], []
for b in blocks:
    name = next((l[4:].split('\t')[0] for l in b if l.startswith('+++ ')), None)
    if name and name.startswith('b/'):
        name = name[2:]
    if name in drop:
        continue
    if name:
        names.append(name)
    kept += b
open(out, 'w').write('\n'.join(kept) + '\n')
print('\n'.join(names))
