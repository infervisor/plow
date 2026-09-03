#!/usr/bin/env python3
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SYMBOL = "plow_moe_combine_fixed_order_gfx950"
EXPECTED_TEXT_SHA256 = "20852a9f9e4a47779c97668c9877d7df0d0c83666987b3fb5ff744593faa4da8"
MARKERS = (
    "plow_moe_combine_fixed_order_abi_1",
    "plow_moe_combine_fixed_order_slots16_1",
    "plow_moe_combine_materialized_f32_1",
    "plow_moe_combine_wave64_1",
    "plow_moe_combine_no_spill_1",
)


def field(notes, name):
    match = re.search(rf"\.{re.escape(name)}:\s+(\d+)", notes)
    if not match:
        raise SystemExit(f"missing AMDGPU metadata field {name}")
    return int(match.group(1))


def kernel_notes(notes):
    position = notes.find(f".name:           {SYMBOL}")
    if position < 0:
        raise SystemExit(f"missing AMDGPU metadata for {SYMBOL}")
    begin = notes.rfind("  - .agpr_count:", 0, position)
    end = notes.find("\n  - .agpr_count:", position)
    return notes[begin if begin >= 0 else 0:end if end >= 0 else len(notes)]


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: manifest.py OBJECT CONTROL_OBJECT MANIFEST")
    obj, control, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    readelf = shutil.which("llvm-readelf") or "/opt/rocm/llvm/bin/llvm-readelf"
    objcopy = shutil.which("llvm-objcopy") or "/opt/rocm/llvm/bin/llvm-objcopy"
    with tempfile.NamedTemporaryFile() as text, tempfile.NamedTemporaryFile() as copied:
        subprocess.check_call([
            objcopy, f"--dump-section=.text={text.name}", str(obj), copied.name,
        ])
        text_digest = hashlib.sha256(Path(text.name).read_bytes()).hexdigest()
    if text_digest != EXPECTED_TEXT_SHA256:
        raise SystemExit(
            f"executable reproducibility failure: {text_digest} != {EXPECTED_TEXT_SHA256}"
        )
    notes = subprocess.check_output([readelf, "-n", str(obj)], text=True)
    symbols = subprocess.check_output([readelf, "-sW", str(obj)], text=True)
    for marker in MARKERS:
        if not re.search(rf"OBJECT .* {re.escape(marker)}$", symbols, re.MULTILINE):
            raise SystemExit(f"object lacks required marker {marker}")
    knotes = kernel_notes(notes)
    resources = {
        "vgpr": field(knotes, "vgpr_count"),
        "sgpr": field(knotes, "sgpr_count"),
        "fixed_lds_bytes": field(knotes, "group_segment_fixed_size"),
        "private_bytes": field(knotes, "private_segment_fixed_size"),
        "vgpr_spills": field(knotes, "vgpr_spill_count"),
        "sgpr_spills": field(knotes, "sgpr_spill_count"),
        "wavefront": field(knotes, "wavefront_size"),
    }
    if resources["wavefront"] != 64:
        raise SystemExit(f"wave64 required: {resources}")
    if resources["private_bytes"] or resources["vgpr_spills"] or resources["sgpr_spills"]:
        raise SystemExit(f"zero private/spill required: {resources}")
    if resources["fixed_lds_bytes"] or resources["vgpr"] > 64 or resources["sgpr"] > 64:
        raise SystemExit(f"lean resource budget exceeded: {resources}")
    manifest = {
        "schema": 1,
        "status": "isolated-candidate",
        "object": {
            "file": "kernel.co",
            "sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
            "text_sha256": text_digest,
            "symbol": SYMBOL,
        },
        "control_object": {
            "file": "control.co",
            "sha256": hashlib.sha256(control.read_bytes()).hexdigest(),
            "symbol": "plow_moe_combine_interpreter_order_gfx950",
        },
        "capability": {"arch": "gfx950", "wavefront": 64, "workgroup": 256},
        "contract": {
            "hidden": "runtime-nonzero",
            "topk": 16,
            "tokens": "runtime-nonzero",
            "reference_shape": {"hidden": 3584, "tokens": 8192},
            "part": "f32[T,topk,H]",
            "output": "bf16[T,H]",
            "association": "residual,shared,part-slot-0-through-15",
        },
        "resources": resources,
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
