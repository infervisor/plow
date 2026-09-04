#!/usr/bin/env python3
import re
import subprocess
import sys


def record(notes, symbol):
    at = notes.find(f".name:           {symbol}")
    if at < 0:
        raise SystemExit(f"missing metadata for {symbol}")
    begin = notes.rfind("  - .agpr_count:", 0, at)
    end = notes.find("\n  - .agpr_count:", at)
    return notes[begin : end if end >= 0 else len(notes)]


def field(kernel, name):
    match = re.search(rf"\.{name}:\s+(\d+)", kernel)
    if not match:
        raise SystemExit(f"missing metadata field {name}")
    return int(match.group(1))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_reuse.py LLVM_READELF REUSE.elf")
    notes = subprocess.check_output([sys.argv[1], "-n", sys.argv[2]], text=True)
    expected = {
        "plow_moe1_quant_sort_a4_gfx950": (256, 64),
        "plow_moe1_a4_reuse_16x16x128_gfx950": (256, 192),
    }
    for symbol, (workgroup, max_vgpr) in expected.items():
        kernel = record(notes, symbol)
        values = {
            name: field(kernel, name)
            for name in (
                "wavefront_size",
                "max_flat_workgroup_size",
                "vgpr_count",
                "sgpr_count",
                "private_segment_fixed_size",
                "vgpr_spill_count",
                "sgpr_spill_count",
            )
        }
        if values["wavefront_size"] != 64 or values["max_flat_workgroup_size"] != workgroup:
            raise SystemExit(f"{symbol} wave/workgroup mismatch: {values}")
        if values["vgpr_count"] > max_vgpr or any(
            values[name]
            for name in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count")
        ):
            raise SystemExit(f"{symbol} resource gate failed: {values}")
        occupancy = 512 // values["vgpr_count"]
        if occupancy < 2:
            raise SystemExit(f"{symbol} occupancy {occupancy} < 2: {values}")
        print(f"{symbol}: {values}, register_occupancy={occupancy}")


if __name__ == "__main__":
    main()
