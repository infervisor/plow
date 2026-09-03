#!/usr/bin/env python3
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_TEXT_SHA256 = "374a485d18af2f762718ddfff762909210af357004704ef746e6864afbd94282"
SYMBOL = "plow_moe2_mxfp4_16x16x128_gfx950"


def field(notes, name):
    match = re.search(rf"\.{re.escape(name)}:\s+(\d+)", notes)
    if not match:
        raise SystemExit(f"missing AMDGPU metadata field {name}")
    return int(match.group(1))


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
    resources = {
        "vgpr": field(notes, "vgpr_count"),
        "sgpr": field(notes, "sgpr_count"),
        "fixed_lds_bytes": field(notes, "group_segment_fixed_size"),
        "dynamic_lds_bytes": 16640,
        "lds_bytes": 16640,
        "private_bytes": field(notes, "private_segment_fixed_size"),
        "vgpr_spills": field(notes, "vgpr_spill_count"),
        "sgpr_spills": field(notes, "sgpr_spill_count"),
    }
    expected = {
        "vgpr": 94, "sgpr": 46, "fixed_lds_bytes": 0, "dynamic_lds_bytes": 16640,
        "lds_bytes": 16640, "private_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0,
    }
    if resources != expected:
        raise SystemExit(f"native resource gate failed: {resources}")
    manifest = {
        "schema": 1,
        "implementation": "native-hip",
        "status": "gate-only-not-routed",
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
            "sort_block_m": 32,
        },
        "encoding": {
            "activation": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
            "weight": "mxfp4-e2m1-paired-nibbles-e8m0-block32",
            "output": "bf16-atomic-accumulate",
            "weight_layout": "E,N/16,Kbytes/32,2,16,16-permute-0,1,3,4,2,5",
            "scale_layout": "pad256x8-view-sm/32,2,16,sn/8,2,4-permute-0,3,5,2,4,1",
        },
        "abi": {
            "kernarg_bytes": 96,
            "arguments": [
                "out*", "activation*", "weight*", "activation_scale*", "weight_scale*",
                "sorted_token_ids*", "expert_ids*", "sorted_weights*", "num_valid_ids*",
                "bias*", "tokens:i32", "model_dim:i32", "inter_dim:i32", "expert_blocks:i32",
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
