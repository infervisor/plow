#!/usr/bin/env python3
"""Section-aware digest of a v6 packet: the directory at header[40..48] (magic + count, 48-byte
entries: kind u32, pad, off u64, size u64, name[24]) names the metadata sections; the device part
is everything that is neither a section payload nor the directory (the header's directory offset
field excluded). Two packets that differ only in serve.json have the same device digest and the
same digests for every other section."""
import hashlib
import sys


def parse(path):
    b = open(path, "rb").read()
    d = int.from_bytes(b[40:48], "little")
    if d == 0:
        return None, {"<whole>": hashlib.sha256(b).hexdigest()[:16]}
    n = int.from_bytes(b[d + 4:d + 8], "little")
    sections = {}
    ranges = [(d, len(b))]
    for i in range(n):
        e = b[d + 8 + 48 * i:d + 8 + 48 * (i + 1)]
        off = int.from_bytes(e[8:16], "little")
        size = int.from_bytes(e[16:24], "little")
        name = e[24:48].split(b"\0", 1)[0].decode()
        sections[name] = hashlib.sha256(b[off:off + size]).hexdigest()[:16] + f" ({size} B)"
        ranges.append((off, off + size))
    ranges.sort()
    h = hashlib.sha256()
    h.update(b[:40])
    h.update(b[48:64])
    pos = 64
    for a, z in ranges:
        if a > pos:
            h.update(b[pos:a])
        pos = max(pos, z)
    if pos < len(b):
        h.update(b[pos:])
    return h.hexdigest()[:16], sections


for path in sys.argv[1:]:
    dev, sections = parse(path)
    print(f"{path}: device {dev}")
    for k in sorted(sections):
        print(f"    {k}: {sections[k]}")
