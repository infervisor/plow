#!/usr/bin/env python3
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_TEXT_SHA256 = "b72d36947ea11fec74cdd51bf5bbb7354571c4f8714d1dd9c244dd39699c6ad1"
SYMBOL = "plow_moe2_mxfp4_16x16x128_gfx950"
MARKERS = (
    "plow_moe2_mxfp4_stage2_abi_2",
    "plow_moe2_mxfp4_stage2_layout_shuffled_1",
    "plow_moe2_mxfp4_stage2_no_spill_1",
    "plow_moe2_mxfp4_stage2_dynamic_lds_16640",
    "plow_moe2_mxfp4_stage2_vgpr_le_144",
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
    digest = hashlib.sha256(obj.read_bytes()).hexdigest()
    with tempfile.NamedTemporaryFile() as text, tempfile.NamedTemporaryFile() as copied:
        subprocess.check_call([
            objcopy, f"--dump-section=.text={text.name}", str(obj), copied.name,
        ])
        text_digest = hashlib.sha256(Path(text.name).read_bytes()).hexdigest()
    if text_digest != EXPECTED_TEXT_SHA256:
        raise SystemExit(
            f"native executable reproducibility failure: {text_digest} != {EXPECTED_TEXT_SHA256}"
        )
    notes = subprocess.check_output([readelf, "-n", str(obj)], text=True)
    if f".name:           {SYMBOL}" not in notes or "amdgcn-amd-amdhsa--gfx950" not in notes:
        raise SystemExit("native object symbol/target gate failed")
    symbols = subprocess.check_output([readelf, "-sW", str(obj)], text=True)
    for marker in MARKERS:
        if not re.search(rf"OBJECT .* {re.escape(marker)}$", symbols, re.MULTILINE):
            raise SystemExit(f"native object lacks required marker {marker}")
    knotes = kernel_notes(notes, SYMBOL)
    resources = {
        "vgpr": field(knotes, "vgpr_count"),
        "sgpr": field(knotes, "sgpr_count"),
        "fixed_lds_bytes": field(knotes, "group_segment_fixed_size"),
        "dynamic_lds_bytes": 16640,
        "lds_bytes": 16640,
        "private_bytes": field(knotes, "private_segment_fixed_size"),
        "vgpr_spills": field(knotes, "vgpr_spill_count"),
        "sgpr_spills": field(knotes, "sgpr_spill_count"),
    }
    expected = {
        "vgpr": 100, "sgpr": 42, "fixed_lds_bytes": 0, "dynamic_lds_bytes": 16640,
        "lds_bytes": 16640, "private_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0,
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
            "file": "kernel.co", "sha256": digest, "text_sha256": text_digest, "symbol": SYMBOL,
        },
        "capability": {"arch": "gfx950", "wavefront": 64, "workgroup": 256},
        "geometry": {
            "tokens": 1024, "topk": 16, "model_dim": 3584, "inter_dim": 384,
            "experts": 896, "tile_m": 32, "tile_n": 256, "tile_k": 128,
            "sort_block_m": 64,
        },
        "encoding": {
            "activation": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
            "weight": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
            "output": "bf16-atomic-accumulate",
            "weight_layout": "expert-table[E*3+2]-N/16,Kbytes/32,2,16,16",
            "scale_layout": "expert-scale-table[E*3+2]-pad256x8-shuffled",
        },
        "abi": {
            "kernarg_bytes": 88,
            "arguments": [
                "out*", "activation*", "weight_table*", "activation_scale*",
                "weight_scale_table*", "meta*", "row_partidx*", "sorted_weights*",
                "tokens:i32", "model_dim:i32", "inter_dim:i32", "experts:i32", "topk:i32",
                "reserved:i32",
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
