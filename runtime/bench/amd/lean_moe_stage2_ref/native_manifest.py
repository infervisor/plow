#!/usr/bin/env python3
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_TEXT_SHA256 = "0c697ba09401e11c0f10fa7bf47e3eaf7d289f689ee98329867dbe1737016644"
SYMBOL = "plow_moe2_mxfp4_16x16x128_gfx950"
MARKERS = (
    "plow_moe2_mxfp4_stage2_abi_3",
    "plow_moe2_mxfp4_stage2_layout_shuffled_1",
    "plow_moe2_mxfp4_stage2_no_spill_1",
    "plow_moe2_mxfp4_stage2_f32_scatter_1",
    "plow_moe2_mxfp4_stage2_dynamic_lds_4352",
    "plow_moe2_mxfp4_stage2_vgpr_le_100",
)


def field(notes, name):
    match = re.search(rf"\.{re.escape(name)}:\s+(\d+)", notes)
    if not match:
        raise SystemExit(f"missing AMDGPU metadata field {name}")
    return int(match.group(1))


def kernel_notes(notes, symbol):
    pos = notes.find(f".name:           {symbol}")
    if pos < 0:
        raise SystemExit(f"missing AMDGPU metadata for {symbol}")
    begin = notes.rfind("  - .agpr_count:", 0, pos)
    end = notes.find("\n  - .agpr_count:", pos)
    return notes[begin if begin >= 0 else 0:end if end >= 0 else len(notes)]


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: native_manifest.py OBJECT MANIFEST LLVM_READELF LLVM_OBJCOPY")
    obj, output, readelf, objcopy = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    ep_full_i = os.environ.get("PLOW_MOE2_EP_FULL_I") == "1"
    symbol = "plow_moe2_ep_full_i_16x16x128_gfx950" if ep_full_i else SYMBOL
    digest = hashlib.sha256(obj.read_bytes()).hexdigest()
    with tempfile.NamedTemporaryFile() as text, tempfile.NamedTemporaryFile() as copied:
        subprocess.check_call([
            objcopy, f"--dump-section=.text={text.name}", str(obj), copied.name,
        ])
        text_digest = hashlib.sha256(Path(text.name).read_bytes()).hexdigest()
    if not ep_full_i and text_digest != EXPECTED_TEXT_SHA256:
        raise SystemExit(
            f"native executable reproducibility failure: {text_digest} != {EXPECTED_TEXT_SHA256}"
        )
    notes = subprocess.check_output([readelf, "-n", str(obj)], text=True)
    if f".name:           {symbol}" not in notes or "amdgcn-amd-amdhsa--gfx950" not in notes:
        raise SystemExit("native object symbol/target gate failed")
    symbols = subprocess.check_output([readelf, "-sW", str(obj)], text=True)
    markers = (MARKERS[:-1] + ("plow_moe2_ep_full_i_3072", "plow_moe2_ep_full_i_vgpr_le_128")) if ep_full_i else MARKERS
    for marker in markers:
        if not re.search(rf"OBJECT .* {re.escape(marker)}$", symbols, re.MULTILINE):
            raise SystemExit(f"native object lacks required marker {marker}")
    knotes = kernel_notes(notes, symbol)
    resources = {
        "vgpr": field(knotes, "vgpr_count"),
        "sgpr": field(knotes, "sgpr_count"),
        "fixed_lds_bytes": field(knotes, "group_segment_fixed_size"),
        "dynamic_lds_bytes": 4352,
        "lds_bytes": 4352,
        "private_bytes": field(knotes, "private_segment_fixed_size"),
        "vgpr_spills": field(knotes, "vgpr_spill_count"),
        "sgpr_spills": field(knotes, "sgpr_spill_count"),
    }
    expected = {
        "vgpr": 120 if ep_full_i else 98, "sgpr": 40, "fixed_lds_bytes": 0, "dynamic_lds_bytes": 4352,
        "lds_bytes": 4352, "private_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0,
    }
    if resources != expected:
        raise SystemExit(f"native resource gate failed: {resources}")
    manifest = {
        "schema": 1,
        "implementation": "native-hip",
        "status": "production-capability-routed",
        "generator": {
            "source": "Plow native HIP, schedule derived from MIT AITER",
            "toolchain": "nix ROCm 7.14",
        },
        "object": {
            "file": "kernel.co", "sha256": digest, "text_sha256": text_digest, "symbol": symbol,
        },
        "capability": {"arch": "gfx950", "wavefront": 64, "workgroup": 256},
        "geometry": {
            "tokens": 1024, "topk": 2 if ep_full_i else 16, "model_dim": 3584,
            "inter_dim": 3072 if ep_full_i else 384,
            "experts": 112 if ep_full_i else 896, "tile_m": 32, "tile_n": 256, "tile_k": 128,
            "sort_block_m": 64,
        },
        "encoding": {
            "activation": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
            "weight": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
            "output": "f32-fixed-part-scatter",
            "weight_layout": "expert-table[E*3+2]-N/16,Kbytes/32,2,16,16",
            "scale_layout": "expert-scale-table[E*3+2]-pad256x8-shuffled",
        },
        "abi": {
            "kernarg_bytes": 80,
            "arguments": [
                "part*", "activation*", "weight_table*", "activation_scale*",
                "weight_scale_table*", "meta*", "row_partidx*", "row_gate*",
                "model_dim:i32", "inter_dim:i32", "experts:i32", "reserved:i32",
            ],
        },
        "resources": resources,
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "object": str(obj), "sha256": digest, "text_sha256": text_digest,
        "resources": resources,
    }))


if __name__ == "__main__":
    main()
