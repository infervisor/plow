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



def read_section(path, name):
    """Read one metadata section without loading weights or scanning arbitrary tensor bytes."""
    import os
    import struct

    with open(path, "rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:7] != b"PLOWDEV" or header[7] not in (8, 9, 10, 11):
            raise ValueError(f"{path}: expected a sectioned devblob")
        size = os.fstat(stream.fileno()).st_size
        directory = struct.unpack_from("<Q", header, 40)[0]
        if not 64 <= directory <= size - 8:
            raise ValueError(f"{path}: invalid section directory")
        stream.seek(directory)
        magic, count = struct.unpack("<4sI", stream.read(8))
        if magic != b"SECT" or count > 4096 or directory + 8 + 48 * count > size:
            raise ValueError(f"{path}: malformed section directory")
        entries = stream.read(48 * count)
        found = None
        for i in range(count):
            kind, flags, offset, length, raw_name = struct.unpack_from("<IIQQ24s", entries, 48 * i)
            section_name = raw_name.split(b"\0", 1)[0].decode("utf-8")
            if offset < 64 or offset + length > directory:
                raise ValueError(f"{path}: invalid section range for {section_name}")
            if section_name == name:
                if kind != 4:
                    raise ValueError(f"{path}: {name} is not metadata")
                if found is not None or length > 64 * 1024 * 1024:
                    raise ValueError(f"{path}: duplicate or oversized {name}")
                found = offset, length
        if found is None:
            return None
        stream.seek(found[0])
        return stream.read(found[1])


if __name__ == "__main__":
    for path in sys.argv[1:]:
        dev, sections = parse(path)
        print(f"{path}: device {dev}")
        for k in sorted(sections):
            print(f"    {k}: {sections[k]}")
