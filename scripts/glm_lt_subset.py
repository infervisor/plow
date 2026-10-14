#!/usr/bin/env python3
"""Cut a small gfx950 code object holding only the named Tensile kernels out of the unbundled
hipBLASLt library image: per kernel the code bytes (function start to the next function), the
64-byte descriptor (entry offset re-linked) and its own NT_AMDGPU_METADATA entry.

glm_lt_subset.py IMAGE NOTES_TXT NAMES_FILE OUT_DIR  ->  OUT_DIR/subset.s (+ code blobs); the
caller assembles with llvm-mc and links with ld.lld -shared (scripts/build_glm_lt.sh --gfx950).
NOTES_TXT is `llvm-readelf --notes IMAGE`. The image has no relocations and no data besides the
descriptors, which this relies on.
"""
import os
import struct
import sys

image, notes, names_file, out = sys.argv[1:5]
data = open(image, 'rb').read()
shoff, = struct.unpack_from('<Q', data, 0x28)
shentsize, shnum = struct.unpack_from('<HH', data, 0x3A)
secs = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize) for i in range(shnum)]
want = [n.strip() for n in open(names_file) if n.strip()]
funcs, kds = {}, {}
starts = []
for name_off, typ, flags, addr, off, size, link, info, align, entsize in secs:
    if typ != 2:  # SYMTAB
        continue
    stroff = secs[link][4]
    for j in range(size // 24):
        st_name, st_info, _, st_shndx, st_value, st_size = struct.unpack_from('<IBBHQQ', data, off + j * 24)
        if st_shndx == 0 or st_shndx >= len(secs):
            continue
        sec_addr, sec_off = secs[st_shndx][3], secs[st_shndx][4]
        fo = st_value - sec_addr + sec_off
        if st_info & 0xF == 2:  # FUNC
            starts.append(fo)
        e = data.index(b'\0', stroff + st_name)
        nm = data[stroff + st_name:e].decode()
        if nm.endswith('.kd') and nm[:-3] in want:
            kds[nm[:-3]] = fo
        elif nm in want and st_info & 0xF == 2:
            funcs[nm] = fo
text_end = max(s[4] + s[5] for s in secs if s[2] & 0x4)  # SHF_EXECINSTR
starts = sorted(set(starts)) + [text_end]

meta = {}
blocks = open(notes).read().split('amdhsa.kernels:\n')[1:]
for blk in blocks:
    body = blk.split('\namdhsa.version:')[0]
    for line in body.splitlines():
        if line.strip().startswith('.name:') and line.startswith('    .name:'):
            meta[line.split(':', 1)[1].strip()] = body
os.makedirs(out, exist_ok=True)
asm = ['.amdgcn_target "amdgcn-amd-amdhsa--gfx950"', '.amdhsa_code_object_version 4', '.text']
for i, nm in enumerate(want):
    fo = funcs[nm]
    end = starts[starts.index(fo) + 1]
    open(f'{out}/k{i}.bin', 'wb').write(data[fo:end])
    asm += ['.p2align 8', f'.globl {nm}', f'.protected {nm}', f'.type {nm},@function', f'{nm}:', f'.incbin "{out}/k{i}.bin"']
asm += ['.rodata']
for i, nm in enumerate(want):
    d = data[kds[nm]:kds[nm] + 64]
    asm += ['.p2align 6', f'.globl {nm}.kd', f'.protected {nm}.kd', f'.type {nm}.kd,@object', f'.size {nm}.kd, 64', f'{nm}.kd:',
            '.byte ' + ','.join(str(b) for b in d[:16]),
            f'.quad {nm} - {nm}.kd',
            '.byte ' + ','.join(str(b) for b in d[24:])]
asm += ['.amdgpu_metadata', '---', 'amdhsa.kernels:']
for nm in want:
    asm.append(meta[nm].rstrip('\n'))
asm += ['amdhsa.version:', '  - 1', '  - 1', '...', '.end_amdgpu_metadata']
open(f'{out}/subset.s', 'w').write('\n'.join(asm) + '\n')
print(f'{len(want)} kernels, {sum(os.path.getsize(f"{out}/k{i}.bin") for i in range(len(want)))} code bytes')
