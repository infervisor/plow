#!/usr/bin/env python3
"""Capture rank-zero GLM TP8 grouped-decode inputs and intermediates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "blob", "objects", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=[2, 4, 8, 16, 20], default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = {"act.xn2": "x", "act.tab": "tab", "act.moe_fug": "fu",
             "act.moe_meta": "meta", "act.moe_rowtok": "rt", "act.moe_rowpart": "rp",
             "act.moe_rowgate": "rg", "act.part": "part"}
    lengths = [512] * (args.rows - 1) + [32768]
    prompts = [",".join([str(i + 1)] * length) for i, length in enumerate(lengths)]
    options = {
        "PLOW_DUMP_ACT": ",".join(f"{name}:{args.out / file}" for name, file in names.items()),
        "PLOW_MLA_PF_V2": "1", "PLOW_MLA_PF_AITER": "1",
        "PLOW_PF_CHUNK": "8192", "PLOW_PF_INTERLEAVE": "0",
    }
    cmd = [str(args.runtime), "amd-bench", "--blob", str(args.blob), "--hsaco",
           str(args.objects), "--checkpoint", str(args.checkpoint), "--prompt",
           ";".join(prompts), "--batched", "--tp", "8", "--steps", "2"]
    with (args.out / "capture.log").open("w") as log:
        subprocess.run(cmd, env=os.environ | options, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    for step in range(2):
        directory = args.out / f"step{step}"
        directory.mkdir()
        for file in names.values():
            (directory / f"{file}.bin").symlink_to(
                (args.out / f"{file}.b{step:03d}.bin").resolve())
    record = {
        "runtime_sha256": sha(args.runtime), "packet_sha256": sha(args.blob),
        "object_sha256": {p.name: sha(p) for p in sorted(args.objects.iterdir())
                          if p.is_file() and p.suffix in (".elf", ".co")},
        "environment": options, "prompt_lengths": lengths,
        "prompt_tokens": list(range(1, args.rows + 1)), "rows": args.rows,
        "steps": 2, "tp": 8, "rank": 0,
        "checkpoint": str(args.checkpoint),
        "captures_sha256": {p.name: sha(p) for p in sorted(args.out.glob("*.bin"))},
    }
    (args.out / "capture.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
