#!/usr/bin/env python3
"""Read the CROSS-GPU COLLECTIVES straight out of a .pkt. No GPU, no runtime.

Answers, per prefill bucket and for decode: how many elements does the all-reduce
actually move (`i[0]`), across how many workgroups (`blocks`), into what slot
(`i[2]`) — i.e. the concrete `n` and `slot_bytes` the tile/watermark question turns on.

Layout mirrors runtime/common/dev_blob.h + dev_isa.h (which are the authority).
"""
import struct
import sys

XREDUCE, XREDUCE2 = 24, 29
NAMES = {XREDUCE: "XReduce(one-shot)", XREDUCE2: "XReduceTwoShot"}


def main(path):
    b = open(path, "rb").read()
    magic = b[:8]
    n_cu, n_tensor, n_prog, n_kvrow, flags, target = struct.unpack_from("<6I", b, 8)
    (init_bytes,) = struct.unpack_from("<Q", b, 32)
    (dir_off,) = struct.unpack_from("<Q", b, 40)
    print(f"{path}\n  magic={magic!r} n_cu={n_cu} n_tensor={n_tensor} n_prog={n_prog} flags={flags}")

    off = 64 + 96 * n_tensor + init_bytes + 4 * n_kvrow
    if magic == b"PLOWDEV\x09" and dir_off:
        assert b[dir_off:dir_off + 4] == b"SECT", "no section directory"
        (nsec,) = struct.unpack_from("<I", b, dir_off + 4)
        for i in range(nsec):
            base = dir_off + 8 + i * (4 + 4 + 8 + 8 + 24)
            kind, _pad, soff, ssz = struct.unpack_from("<IIQQ", b, base)
            if kind == 0:  # SECT_PROGRAMS
                off = soff
                break

    hidden = None
    rows = []
    for p in range(n_prog):
        n_inst, n_stream, n_wait, n_succ, n_counter, T = struct.unpack_from("<6I", b, off)
        off += 24
        insts = off
        for k in range(n_inst):
            o = insts + 64 * k
            op, blocks = struct.unpack_from("<HH", b, o)
            if op in (XREDUCE, XREDUCE2):
                i = struct.unpack_from("<8I", b, o + 32)
                rows.append((T, NAMES[op], blocks, i[0], i[1], i[2]))
        off = insts + 64 * n_inst + 24 * n_stream + 4 * n_cu * 2 + 8 * n_wait + 4 * n_succ

    # collapse: every collective in a program has the same shape
    seen = {}
    for T, name, blocks, n, tp, slot in rows:
        key = (T, name, blocks, n, tp, slot)
        seen[key] = seen.get(key, 0) + 1
    print(f"\n  {'T':>6} {'op':<18} {'count':>5} {'blocks':>6} {'n(elems)':>12} "
          f"{'MB(bf16)':>9} {'tp':>3} {'slot_bytes':>12}")
    for (T, name, blocks, n, tp, slot), c in sorted(seen.items()):
        print(f"  {T:>6} {name:<18} {c:>5} {blocks:>6} {n:>12} "
              f"{n * 2 / 2**20:>9.2f} {tp:>3} {slot:>12}")
        if name.startswith("XReduce(") and T == 1:
            hidden = n
    if hidden:
        print(f"\n  hidden = {hidden}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
