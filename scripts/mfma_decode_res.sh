#!/usr/bin/env bash
# Per-kernel resources for the MFMA decode bench object: metadata VGPR/AGPR/LDS/spill AND the
# ISA truth (`scratch_load_*` / `scratch_store_*` counts), because earlier candidates in this
# tree reported `.vgpr_spill_count: 0` and spilled anyway.
#   nix develop /app/plow --command scripts/mfma_decode_res.sh <co> [<disasm-out>]
set -euo pipefail
CO="$1"
DIS="${2:-/tmp/mf31.s}"
BUN="$(ls -1 "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
        "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)"
if head -c4 "$CO" | grep -q ELF; then cp "$CO" "$DIS.elf"; else
"$BUN" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$CO" --output="$DIS.elf"
fi
"$ROCM_PATH/llvm/bin/llvm-objdump" -d --triple=amdgcn-amd-amdhsa --mcpu=gfx942 "$DIS.elf" > "$DIS"
"$ROCM_PATH/llvm/bin/llvm-readelf" --notes "$DIS.elf" > "$DIS.notes"

python3 - "$DIS" <<'PY'
import re, sys, collections
dis = open(sys.argv[1]).read()
notes = open(sys.argv[1] + ".notes").read()

# metadata: msgpack dump lists the fields of one record alphabetically, so `.agpr_count` and
# `.group_segment_fixed_size` arrive BEFORE the `.name` they belong to. Buffer, then commit.
meta = {}
pend = {}
KEYS = ('.vgpr_count', '.agpr_count', '.sgpr_count', '.group_segment_fixed_size',
        '.private_segment_fixed_size', '.vgpr_spill_count', '.sgpr_spill_count')
for line in notes.splitlines():
    s = line.strip().lstrip('- ').strip()
    if s.startswith('.agpr_count:'):
        pend = {}
    for key in KEYS:
        m2 = re.match(re.escape(key) + r':\s+(\S+)', s)
        if m2:
            pend[key] = m2.group(1)
    m = re.match(r'\.name:\s+(\S+)$', s)
    if m:
        meta[m.group(1)] = pend

# ISA: split the disassembly by kernel symbol
bodies = {}
cur = None
for line in dis.splitlines():
    m = re.match(r'^[0-9a-f]{16} <(\S+)>:', line)
    if m:
        cur = m.group(1); bodies[cur] = []
    elif cur:
        bodies[cur].append(line)

print(f"{'kernel':<16} {'vgpr':>5} {'agpr':>5} {'sgpr':>5} {'lds':>7} {'scratch':>8} "
      f"{'mdspill':>8} {'scr_ld':>7} {'scr_st':>7} {'mfma':>6} {'valu':>7} {'bufld':>6} {'dsrd':>5}")
for k in sorted(bodies):
    if k.endswith('.kd'):
        continue
    b = "\n".join(bodies[k])
    md = meta.get(k, {})
    scr_ld = len(re.findall(r'\bscratch_load_', b))
    scr_st = len(re.findall(r'\bscratch_store_', b))
    mfma = len(re.findall(r'\bv_mfma_', b))
    valu = len(re.findall(r'\tv_[a-z]', b)) - mfma
    bufld = len(re.findall(r'\bbuffer_load_', b))
    dsrd = len(re.findall(r'\bds_read_', b))
    print(f"{k:<16} {md.get('.vgpr_count','?'):>5} {md.get('.agpr_count','?'):>5} "
          f"{md.get('.sgpr_count','?'):>5} {md.get('.group_segment_fixed_size','?'):>7} "
          f"{md.get('.private_segment_fixed_size','?'):>8} {md.get('.vgpr_spill_count','?'):>8} "
          f"{scr_ld:>7} {scr_st:>7} {mfma:>6} {valu:>7} {bufld:>6} {dsrd:>5}")
PY
