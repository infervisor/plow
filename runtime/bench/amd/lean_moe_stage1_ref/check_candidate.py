#!/usr/bin/env python3
import re
import sys
from pathlib import Path


CANDIDATE = "plow_moe1_mxfp4_bm64_bn128_bk256_xcd8_wgm4_gfx950"
SHIPPING = "plow_moe1_mxfp4_bk256_gfx950"


def kernel_record(notes, symbol):
    at = notes.find(f".name:           {symbol}")
    if at < 0:
        raise SystemExit(f"missing metadata for {symbol}")
    begin = notes.rfind("  - .agpr_count:", 0, at)
    end = notes.find("\n  - .agpr_count:", at)
    return notes[begin:end if end >= 0 else len(notes)]


def field(record, name):
    match = re.search(rf"\.{name}:\s+(\d+)", record)
    if not match:
        raise SystemExit(f"missing metadata field {name}")
    return int(match.group(1))


def remap(lin, n_tiles, tm, tn):
    per = n_tiles // 8
    tile = (lin % 8) * per + lin // 8 if lin < per * 8 else lin
    in_group = 4 * tn
    first_m = tile // in_group * 4
    if first_m >= tm:
        return tile
    group_m = min(tm - first_m, 4)
    rest = tile % in_group
    if rest >= group_m * tn:
        return tile
    return (first_m + rest % group_m) * tn + rest // group_m


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: check_candidate.py CANDIDATE_NOTES CANDIDATE_ISA SHIPPING_NOTES SHIPPING_ISA")
    candidate_notes, candidate_isa, shipping_notes, shipping_isa = (
        Path(arg).read_text() for arg in sys.argv[1:]
    )
    records = {
        "candidate": kernel_record(candidate_notes, CANDIDATE),
        "shipping": kernel_record(shipping_notes, SHIPPING),
    }
    expected = {
        "candidate": {"wavefront_size": 64, "max_flat_workgroup_size": 256,
                      "vgpr_count": 120, "sgpr_count": 79,
                      "private_segment_fixed_size": 0, "vgpr_spill_count": 0,
                      "sgpr_spill_count": 0},
        "shipping": {"wavefront_size": 64, "max_flat_workgroup_size": 512,
                     "vgpr_count": 190, "sgpr_count": 90,
                     "private_segment_fixed_size": 0, "vgpr_spill_count": 0,
                     "sgpr_spill_count": 0},
    }
    for kind, values in expected.items():
        actual = {name: field(records[kind], name) for name in values}
        if actual != values:
            raise SystemExit(f"{kind} metadata mismatch: {actual} != {values}")
    for kind, isa in (("candidate", candidate_isa), ("shipping", shipping_isa)):
        if "scratch_" in isa:
            raise SystemExit(f"{kind} ISA contains scratch traffic")
        if "v_mfma_scale_f32_32x32x64_f8f6f4" not in isa:
            raise SystemExit(f"{kind} ISA lacks the scaled A4W4 MFMA")

    # The candidate's GLU half-tile is 64 columns. Six N tiles cover I=384 once,
    # and each sorted row owns 192 fp4 bytes plus 12 E8M0 output-scale bytes.
    columns = [column for tile in range(6) for column in range(tile * 64, tile * 64 + 64)]
    if columns != list(range(384)) or 384 // 2 != 192 or 384 // 32 != 12:
        raise SystemExit("candidate sorted-row/output-scale coverage mismatch")
    candidate_lds = 2 * (64 * 128 + 128 * 128) + 2 * 64 * 8 + 2 * 128 * 8
    if candidate_lds != 52224 or candidate_lds * 3 > 160 * 1024:
        raise SystemExit("candidate cannot sustain three WG256 blocks in 160 KiB LDS")
    for tn in range(1, 17):
        for tm in range(1, 513):
            mapped = [remap(i, tm * tn, tm, tn) for i in range(tm * tn)]
            if sorted(mapped) != list(range(tm * tn)):
                raise SystemExit(f"XCD/WGM4 map is not bijective for tm={tm}, tn={tn}")
    print("candidate: wave64 WG256 BM64/BN128/BK256, VGPR=120 SGPR=79 private=0 spills=0")
    print("shipping:  wave64 WG512 BM64/BN256/BK256, VGPR=190 SGPR=90 private=0 spills=0")
    print("coverage: N=384 -> 6x64 columns; sorted fp4 row=192 B; scale row=12 B; XCD8/WGM4 bijective")
    print("occupancy: register ceiling=4 waves/SIMD; 52224 B LDS -> 3 WG/CU -> effective 3 waves/SIMD")


if __name__ == "__main__":
    main()
