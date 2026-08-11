#!/usr/bin/env python3
"""Create deterministic finite-bf16 inputs for an AMD decode-block replay."""

import argparse
import array
import json
import os
import re
import subprocess
from pathlib import Path


def tensor_table(plowrt: Path, blob: Path) -> list[dict]:
    env = dict(os.environ)
    env["RUST_LOG"] = "error"
    raw = subprocess.check_output(
        [
            str(plowrt),
            "disasm",
            str(blob),
            "--program",
            "1",
            "--tensors",
            "--no-analysis",
            "--format",
            "json",
        ],
        env=env,
    )
    return json.loads(raw)["tensors"]


def write_bf16(path: Path, nbytes: int, seed: int) -> None:
    if nbytes % 2:
        raise SystemExit(f"{path}: bf16 fixture has odd byte count {nbytes}")
    values = (0x3D80, 0xBD00, 0x3E00, 0xBE40, 0x3C80, 0x3E80, 0xBD80, 0x3D00)
    remaining = nbytes // 2
    index = seed % len(values)
    with path.open("wb") as out:
        while remaining:
            count = min(remaining, 1 << 20)
            chunk = array.array("H", (values[(index + i) % len(values)] for i in range(count)))
            out.write(chunk.tobytes())
            index = (index + count) % len(values)
            remaining -= count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plowrt", type=Path, default=Path("target/release/plowrt"))
    parser.add_argument(
        "--include",
        default=r"^(act\.x|kv\.\d+\.(ckv|krot))$",
        help="full-match regex over tensor names",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    include = re.compile(args.include)
    args.out.mkdir(parents=True, exist_ok=True)
    selected = [t for t in tensor_table(args.plowrt, args.blob) if include.fullmatch(t["name"])]
    if not selected:
        raise SystemExit("fixture regex selected no tensors")
    for ordinal, tensor in enumerate(selected):
        path = args.out / f"{tensor['name']}.bin"
        write_bf16(path, int(tensor["bytes"]), args.seed + ordinal)
        print(f"{tensor['name']}\t{tensor['bytes']}\t{path}")


if __name__ == "__main__":
    main()
