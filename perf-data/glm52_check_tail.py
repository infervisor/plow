#!/usr/bin/env python3
"""Dump the lm_head tail of EVERY program in a blob: the Gemv (N, a_row0), the Argmax (N), and
whichever of ArgmaxFin (18) / XArgmaxFin (its opcode) closes it, with the xctr ids. Verifies the
sharded head and the fold move together in the PREFILL buckets, not just in decode."""
import struct, sys

GEMV, ARGMAX, ARGMAX_FIN = 10, 17, 18

def _repo_root():
    """The checkout this script lives in.

    Was an absolute path into `.claude/worktrees/agent-a9bf5b5581423ca9f/` — a directory that is
    gitignored, belongs to a since-reaped agent worktree, and never existed for anyone else. It
    was opened at import time with no guard, so this file raised FileNotFoundError on `import`
    for every reader. Derived from __file__ instead: perf-data/ sits at the repo root.
    """
    import os
    env = os.environ.get('PLOW_REPO')
    if env:
        return env
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def opcode_of(name):
    """Parse the opcode number for `name` out of runtime/common/dev_isa.h."""
    import os, re
    with open(os.path.join(_repo_root(), 'runtime/common/dev_isa.h')) as f:
        src = f.read()
    m = re.search(r'PLOW_DOP_' + name + r'\s*=\s*(\d+)', src)
    return int(m.group(1)) if m else None

XAF = opcode_of('XARGMAX_FIN')
print(f"PLOW_DOP_XARGMAX_FIN = {XAF}")

raw = open(sys.argv[1], 'rb').read()
assert raw[:7] == b'PLOWDEV'
# header: magic(8) then u32 fields; locate programs by scanning for the inst arrays is fragile, so
# use the blob header layout from runtime/common/dev_blob.h via the tensor-decl stride trick.
# Simpler and sufficient: DevInst is a fixed 64-byte record; find them by the known tail signature.
# Instead of parsing the container, scan for Argmax->{Argmax,X}Fin adjacency in the inst stream.
INST = 64
best = []
for off in range(0, len(raw) - INST * 3, 4):
    op0 = struct.unpack_from('<H', raw, off)[0]
    if op0 != ARGMAX:
        continue
    op1 = struct.unpack_from('<H', raw, off + INST)[0]
    if op1 not in (ARGMAX_FIN, XAF):
        continue
    gem = struct.unpack_from('<H', raw, off - INST)[0]
    if gem != GEMV:
        continue
    # i[] array offset inside DevInst: after op(u16) pad(u16) t[8](u32) -> 4 + 32 = 36
    gi = struct.unpack_from('<8I', raw, off - INST + 32)
    ai = struct.unpack_from('<8I', raw, off + 32)
    fi = struct.unpack_from('<8I', raw, off + INST + 32)
    best.append((op1, gi[0], gi[1], gi[2], gi[4], ai[0], fi[0], fi[1], fi[2], fi[3], fi[4]))

print(f"{'closer':12s} {'gemvM':>6s} {'gemvN':>8s} {'K':>6s} {'a_row0':>7s} {'amaxN':>8s} "
      f"{'nparts':>7s} {'nbatch':>7s} {'vocab_l':>8s} {'gate':>5s} {'val':>5s}")
for op1, m, n, k, ar, an, fp, fb, fv, fg, fvid in best:
    nm = 'ArgmaxFin' if op1 == ARGMAX_FIN else 'XArgmaxFin'
    print(f"{nm:12s} {m:6d} {n:8d} {k:6d} {ar:7d} {an:8d} {fp:7d} {fb:7d} {fv:8d} {fg:5d} {fvid:5d}")
print(f"\n{len(best)} program tails found")
