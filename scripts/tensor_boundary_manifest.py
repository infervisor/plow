#!/usr/bin/env python3
"""Build a semantic tensor manifest from raw Plow or hook captures."""

import argparse
import hashlib
import json
from pathlib import Path

DTYPE_BYTES = {"bf16": 2, "float32": 4, "f32": 4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--prompt-sha256-u32le", required=True)
    p.add_argument(
        "--tensor",
        action="append",
        default=[],
        metavar="SEMANTIC,LAYER,DTYPE,FILE[,LAST_ELEMENTS]",
        help="add a raw tensor; may be repeated",
    )
    p.add_argument("--hook-dir", type=Path)
    p.add_argument(
        "--hook-sample-offset",
        type=int,
        default=0,
        help="0 selects the latest hook sample per boundary, 1 the previous sample",
    )
    args = p.parse_args()
    rows = []
    for value in args.tensor:
        parts = value.split(",")
        if len(parts) not in (4, 5):
            raise ValueError(f"invalid --tensor specification: {value}")
        semantic, layer, dtype, filename = parts[:4]
        last_elements = int(parts[4]) if len(parts) == 5 else None
        path = Path(filename).resolve()
        raw = path.read_bytes()
        if dtype not in DTYPE_BYTES or len(raw) % DTYPE_BYTES[dtype]:
            raise ValueError(f"invalid {dtype} tensor file: {path}")
        item = {
            "semantic": semantic,
            "layer": int(layer),
            "rank": 0,
            "dtype": "bf16" if dtype == "bf16" else "float32",
            "source_dtype": "bf16" if dtype == "bf16" else "float32",
            "source_shape": [len(raw) // DTYPE_BYTES[dtype]],
            "source_stride": [1],
            "shape": [len(raw) // DTYPE_BYTES[dtype]],
            "file": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "prompt_sha256_u32le": args.prompt_sha256_u32le,
        }
        if last_elements is not None:
            if not 0 < last_elements <= item["shape"][0]:
                raise ValueError(f"invalid last-elements selection: {last_elements}")
            item["selection"] = {"last_elements": last_elements}
            item["shape"] = [last_elements]
        rows.append(item)
    if args.hook_dir:
        captured = {}
        for path in sorted(args.hook_dir.glob("*.json")):
            item = json.loads(path.read_text())
            if item.get("prompt_sha256_u32le") != args.prompt_sha256_u32le:
                raise ValueError(f"history mismatch in {path}")
            key = (item["semantic"], item["layer"], item["rank"])
            captured.setdefault(key, []).append((item.get("call_sequence", 0), path, item))
        for samples in captured.values():
            samples.sort(key=lambda value: value[0], reverse=True)
            if args.hook_sample_offset >= len(samples):
                raise ValueError(
                    f"hook sample offset {args.hook_sample_offset} needs {len(samples) + 1} samples"
                )
            _, path, item = samples[args.hook_sample_offset]
            rows.append(
                {
                    "semantic": item["semantic"],
                    "layer": item["layer"],
                    "rank": item["rank"],
                    "dtype": item["stored_dtype"],
                    "source_dtype": item["source_dtype"],
                    "source_shape": item["source_shape"],
                    "source_stride": item["source_stride"],
                    "shape": item["stored_shape"],
                    "file": str((path.parent / item["file"]).resolve()),
                    "sha256": item["sha256"],
                    "prompt_sha256_u32le": item["prompt_sha256_u32le"],
                }
            )
    if not rows:
        raise ValueError("no tensors supplied")
    keys = [(x["semantic"], x["layer"], x["rank"]) for x in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate semantic/layer/rank key")
    result = {
        "schema": 1,
        "producer": "semantic-tensor-boundaries",
        "name": args.name,
        "prompt_sha256_u32le": args.prompt_sha256_u32le,
        "tensors": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
