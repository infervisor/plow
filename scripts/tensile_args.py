"""Print one kernel's kernarg layout from an extracted Tensile image's AMDGPU metadata note.

    scripts/tensile_args.py <glm_lt_gfx942.elf> [kernel-name-substring]

`amd_gemm_lt.rs` builds that 160-byte block by hand as `Args`, so this is how to check a field's
offset against the body rather than infer it -- in particular the fourteen `epilogue` words, of
which only `dstD` is currently set."""
import json
import struct
import sys

if len(sys.argv) < 2:
    sys.exit('usage: tensile_args.py <glm_lt_gfx942.elf> [kernel-name-substring]')
PATH = sys.argv[1]
WANT = sys.argv[2] if len(sys.argv) > 2 else None


def unpack(b, i):
    """Minimal msgpack reader: returns (value, next_index)."""
    c = b[i]
    i += 1
    if c <= 0x7F:
        return c, i
    if c >= 0xE0:
        return c - 256, i
    if 0x80 <= c <= 0x8F:
        return mapn(b, i, c & 0xF)
    if 0x90 <= c <= 0x9F:
        return arrn(b, i, c & 0xF)
    if 0xA0 <= c <= 0xBF:
        n = c & 0x1F
        return b[i:i + n].decode('utf-8', 'replace'), i + n
    if c == 0xC0:
        return None, i
    if c == 0xC2:
        return False, i
    if c == 0xC3:
        return True, i
    if c == 0xCA:
        return struct.unpack_from('>f', b, i)[0], i + 4
    if c == 0xCB:
        return struct.unpack_from('>d', b, i)[0], i + 8
    if c in (0xCC, 0xCD, 0xCE, 0xCF):
        n = 1 << (c - 0xCC)
        return int.from_bytes(b[i:i + n], 'big'), i + n
    if c in (0xD0, 0xD1, 0xD2, 0xD3):
        n = 1 << (c - 0xD0)
        return int.from_bytes(b[i:i + n], 'big', signed=True), i + n
    if c in (0xD9, 0xDA, 0xDB):
        w = 1 << (c - 0xD9)
        n = int.from_bytes(b[i:i + w], 'big')
        i += w
        return b[i:i + n].decode('utf-8', 'replace'), i + n
    if c in (0xC4, 0xC5, 0xC6):
        w = 1 << (c - 0xC4)
        n = int.from_bytes(b[i:i + w], 'big')
        i += w
        return b[i:i + n], i + n
    if c in (0xDC, 0xDD):
        w = 2 if c == 0xDC else 4
        n = int.from_bytes(b[i:i + w], 'big')
        return arrn(b, i + w, n)
    if c in (0xDE, 0xDF):
        w = 2 if c == 0xDE else 4
        n = int.from_bytes(b[i:i + w], 'big')
        return mapn(b, i + w, n)
    raise ValueError('msgpack byte 0x%02x at %d' % (c, i - 1))


def arrn(b, i, n):
    out = []
    for _ in range(n):
        v, i = unpack(b, i)
        out.append(v)
    return out, i


def mapn(b, i, n):
    out = {}
    for _ in range(n):
        k, i = unpack(b, i)
        v, i = unpack(b, i)
        out[k] = v
    return out, i


d = open(PATH, 'rb').read()

# Find the NT_AMDGPU_METADATA note payload. The note name is "AMDGPU\0\0"; the msgpack
# map that follows starts with a fixmap/map16/map32 whose first key is "amdhsa.kernels"
# or "amdhsa.version".
# The image carries one metadata note per kernel, so scan every one rather than the first.
def note_at(anchor):
    for back in range(1, 12):
        try:
            v, _ = unpack(d, anchor - back)
        except Exception:
            continue
        if isinstance(v, dict) and 'amdhsa.kernels' in v:
            return v
    return None


anchors = []
at = d.find(b'amdhsa.kernels')
while at >= 0:
    anchors.append(at)
    at = d.find(b'amdhsa.kernels', at + 1)
if not anchors:
    sys.exit('no amdhsa.kernels in image')

pick, total = None, 0
for a in anchors:
    meta = note_at(a)
    if meta is None:
        continue
    for k in meta['amdhsa.kernels']:
        total += 1
        nm = k.get('.name') or k.get('.symbol', '')
        if pick is None and (WANT is None or WANT in nm):
            pick = k
print('kernels in image: %d' % total)
if pick is None:
    sys.exit('no kernel matched %r' % WANT)

print('kernel: %s' % (pick.get('.name') or pick.get('.symbol'))[:110])
print('kernarg_segment_size = %s   group_segment = %s   private = %s' % (
    pick.get('.kernarg_segment_size'), pick.get('.group_segment_fixed_size'),
    pick.get('.private_segment_fixed_size')))
print()
print('%-6s %-6s %s' % ('off', 'size', 'name'))
for a in pick.get('.args', []):
    print('%-6s %-6s %s' % (a.get('.offset'), a.get('.size'), a.get('.name')))
