#!/usr/bin/env python3
"""Build a decode GEMV workgroup-cap plan from `plowrt disasm --format json`.

The input may contain plowrt's logging preamble. Only plain `Gemv` packets are
listed because those are the packets `PLOW_GEMV_WG_TUNING` currently controls.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_CAPS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 76, 96, 112, 128, 152, 176, 200, 224, 256)


def load_report(path: Path) -> dict:
    text = path.read_text()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"{path}: no JSON object")
    return json.loads(text[start:])


def int_value(inst: dict, name: str) -> int:
    for item in inst.get("ints", ()):
        if item.get("name") == name:
            return int(item["value"])
    raise ValueError(f"instruction {inst.get('idx')} has no {name} operand")


def effective_blocks(n: int, cap: int) -> int:
    cap = min(n, cap)
    per = (n + cap - 1) // cap
    return (n + per - 1) // per


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("disasm_json", type=Path)
    parser.add_argument("--program", type=int, default=1)
    parser.add_argument(
        "--caps",
        default=",".join(map(str, DEFAULT_CAPS)),
        help="comma-separated candidate caps",
    )
    args = parser.parse_args()

    report = load_report(args.disasm_json)
    program = next((p for p in report["programs"] if int(p["t"]) == args.program), None)
    if program is None:
        raise SystemExit(f"program T={args.program} not found")
    caps = sorted({int(v) for v in args.caps.split(",") if int(v) > 0})
    counts = Counter()
    for inst in program["insts"]:
        if inst.get("op_name") != "Gemv":
            continue
        counts[(int_value(inst, "N"), int_value(inst, "K"), int(inst["blocks"]))] += 1

    print("N\tK\tblocks\tpackets\tcandidate cap:effective-blocks")
    for (n, k, blocks), count in sorted(counts.items()):
        candidates = []
        seen = set()
        for cap in caps:
            effective = effective_blocks(n, cap)
            if effective in seen:
                continue
            seen.add(effective)
            candidates.append(f"{cap}:{effective}")
        print(f"{n}\t{k}\t{blocks}\t{count}\t{','.join(candidates)}")

    print("\nOne-shape A/B overrides (keep the packet's global cap on other shapes):")
    for n, k, blocks in sorted(counts):
        for cap in caps:
            if effective_blocks(n, cap) == blocks:
                continue
            print(f"PLOW_GEMV_WG_TUNING={n}x{k}={cap}")


if __name__ == "__main__":
    main()
