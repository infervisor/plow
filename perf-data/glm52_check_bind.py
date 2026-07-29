#!/usr/bin/env python3
"""Offline pre-flight for the harness bind loop: for every model.*/lm_head tensor the packet
declares, find it in the weight dir and check the col/row/replicated size rule the C harness
applies (glm_col / glm_row / else). Catches a declare-vs-disk mismatch in seconds instead of
after a 4-minute weight load."""
import json, os, struct, sys, re

pkt, wdir, N = sys.argv[1], sys.argv[2], int(sys.argv[3])

def st_index(d):
    idx = {}
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith('model-') and fn.endswith('.safetensors')):
            continue
        with open(os.path.join(d, fn), 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k == '__metadata__' or k in idx:
                continue
            a, b = v['data_offsets']
            idx[k] = (v['dtype'], v['shape'], b - a)
    return idx

# --- packet tensor table (PLOWDEV v9) ---------------------------------------------------------
raw = open(pkt, 'rb').read()
assert raw[:7] == b'PLOWDEV', raw[:8]
# scan for the tensor table by brute-force: names are NUL-terminated ASCII in a fixed-size record.
# Simpler + robust: pull (name, bytes) pairs out of the decl array by locating the ASCII names and
# the 8-byte size that follows the 64-byte name field.
NAME = 80
recs = []
# PlowTensorDecl layout is discovered from the blob: find the first "in.ids" and walk backwards.
i = raw.find(b'in.ids\x00')
assert i > 0
# record stride: distance to the next known name
j = raw.find(b'in.pos\x00', i)
stride = j - i
n_rec = 0
p = i
names = []
while p + stride <= len(raw):
    nm = raw[p:p + NAME].split(b'\x00')[0].decode('ascii', 'replace')
    if not re.match(r'^[A-Za-z0-9_.\-]+$', nm):
        break
    # the byte count is the first u64 after the name field
    nb = struct.unpack_from('<Q', raw, p + NAME)[0]
    names.append((nm, nb))
    p += stride
print(f"packet: {len(names)} tensor decls, stride {stride}")

def glm_col(n, nb=0, got=0, N=1):
    if n == 'lm_head.weight':
        return N > 1 and got == nb * N
    return any(s in n for s in ('derived.q_absorb', 'derived.q_rope', 'derived.v_absorb',
                                'shared_experts.gate_proj', 'shared_experts.up_proj',
                                'mlp.gate_proj.', 'mlp.up_proj.'))
def glm_row(n):
    return any(s in n for s in ('o_proj.weight', 'shared_experts.down_proj', 'mlp.down_proj.'))

idx = st_index(wdir)
bad = miss = ok = 0
for nm, nb in names:
    if 'mlp.experts.' in nm:
        continue
    if not (nm.startswith('model.') or nm.startswith('lm_head')):
        continue
    if 'expert_weight_table' in nm or 'expert_scale_table' in nm:
        continue
    rec = idx.get(nm)
    if rec is None:
        print(f"  MISSING {nm}")
        miss += 1
        continue
    dt, shape, got = rec
    if N > 1 and glm_col(nm, nb, got, N):
        want = nb * N
        tag = 'col'
    elif N > 1 and glm_row(nm):
        want = nb * N
        tag = 'row'
    else:
        want = nb
        tag = 'rep'
    if got != want:
        print(f"  SIZE {tag} {nm}: pkt {nb} x{N if tag!='rep' else 1} = {want}, disk {got} {dt} {shape}")
        bad += 1
    else:
        ok += 1
print(f"ok {ok}, size-mismatch {bad}, missing {miss}")
sys.exit(1 if (bad or miss) else 0)
