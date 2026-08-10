#!/usr/bin/env python3
"""Find loops (backward conditional branches) in an amdgcn disassembly and print
instruction-mix tables for the largest ones.

Usage: loops.py <file.asm> [symbol-substring] [--top N] [--body ADDR]
"""
import sys, re
from collections import Counter

path = sys.argv[1]
symfilt = None
top = 6
dump_body = None
args = sys.argv[2:]
i = 0
while i < len(args):
    if args[i] == '--top': top = int(args[i+1]); i += 2
    elif args[i] == '--body': dump_body = int(args[i+1], 16); i += 2
    else: symfilt = args[i]; i += 1

# parse: lines like "  addr: <bytes>  \tinstr operands"
insts = []  # (addr, opcode, full)
cursym = None
sym_of = {}
for line in open(path):
    m = re.match(r'^([0-9a-f]{6,16}) <(.+)>:', line)
    if m:
        cursym = m.group(2); continue
    m = re.match(r'\s+([a-z0-9_]+)\s*(.*?)\s*//\s*([0-9A-F]{6,16}):\s*\S+\s*(.*)', line)
    if m:
        op, rest, addr = m.group(1), m.group(2) + ' ' + (m.group(4) or ''), int(m.group(3), 16)
        insts.append((addr, op, rest))
        sym_of[addr] = cursym
        continue
    # objdump alt format: "\taddr:\tencoding\tinstr"
    m = re.match(r'^\s*([0-9a-f]+):\s+(?:[0-9a-f ]+\t)?\s*([a-z0-9_.]+)\s*(.*)', line)
    if m and not line.strip().startswith('.'):
        addr = int(m.group(1), 16)
        insts.append((addr, m.group(2), m.group(3)))
        sym_of[addr] = cursym

addr_ix = {a: i for i, (a, _, _) in enumerate(insts)}
sym_base = {}
for a, _, _ in insts:
    sym = sym_of.get(a)
    if sym is not None and sym not in sym_base:
        sym_base[sym] = a

def classify(op, rest):
    if op.startswith('v_mfma'): return 'MFMA'
    if op.startswith('ds_read'): return 'ds_read'
    if op.startswith('ds_write'): return 'ds_write'
    if op.startswith(('global_load', 'buffer_load', 'flat_load')): return 'gload'
    if op.startswith(('global_store', 'buffer_store', 'flat_store')): return 'gstore'
    if op == 's_waitcnt': return 'waitcnt'
    if op == 's_setprio': return 'setprio'
    if op == 's_barrier': return 'barrier'
    if op.startswith(('s_nop', 's_sleep')): return 'snop'
    if op.startswith('v_'): return 'valu'
    if op.startswith('s_'): return 'salu'
    return 'other'

# find backward branches
loops = []
for i, (a, op, rest) in enumerate(insts):
    if op.startswith('s_cbranch') or op == 's_branch':
        tgt = None
        m2 = re.search(r'label_([0-9A-Fa-f]+)', rest)
        m3 = re.search(r'<[^>]*\+0x([0-9a-f]+)>', rest)
        m4 = re.search(r'<([^+>]+)>\s*$', rest)
        if m3:
            base = sym_base.get(sym_of.get(a))
            if base is not None:
                tgt = base + int(m3.group(1), 16)
        elif m2 and insts:
            tgt = insts[0][0] + int(m2.group(1), 16) * 4
        else:
            mx = re.search(r'0x([0-9a-f]+)', rest)
            if mx: tgt = int(mx.group(1), 16)
        if tgt is None:
            continue
        if tgt < a and tgt in addr_ix:
            body = insts[addr_ix[tgt]:i + 1]
            if symfilt and (sym_of.get(a) is None or symfilt not in sym_of[a]): continue
            loops.append((tgt, a, len(body), body, sym_of.get(a)))

loops.sort(key=lambda l: -l[2])
seen = set()
shown = 0
for tgt, a, n, body, sym in loops:
    if shown >= top: break
    key = (tgt >> 4)
    if key in seen: continue
    seen.add(key)
    shown += 1
    c = Counter(classify(op, rest) for _, op, rest in body)
    # waitcnt detail
    wc = Counter()
    for _, op, rest in body:
        if op == 's_waitcnt':
            wc[rest.strip()[:40]] += 1
    print(f"== loop {tgt:#x}..{a:#x}  {n} insts  sym={sym}")
    print("   mix:", dict(c.most_common()))
    if wc: print("   waitcnts:", dict(wc.most_common(8)))
    # pipeline shape: position of first gload vs first MFMA vs waitcnts
    seqs = [(classify(op, rest), i) for i, (_, op, rest) in enumerate(body)]
    firsts = {}
    for cls, i in seqs:
        firsts.setdefault(cls, i)
    print("   first-occurrence:", {k: v for k, v in sorted(firsts.items(), key=lambda kv: kv[1]) if k in ('gload','ds_write','ds_read','MFMA','waitcnt','setprio','barrier')})

if dump_body is not None and dump_body in addr_ix:
    print("=== body dump ===")
    for a, op, rest in insts[addr_ix[dump_body]:addr_ix[dump_body]+400]:
        print(f"{a:#x}: {op} {rest[:80]}")
