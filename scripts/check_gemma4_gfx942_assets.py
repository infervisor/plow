#!/usr/bin/env python3
"""Fail unless a Gemma-4 MI300X asset manifest is production-qualified."""

import argparse
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_json", type=Path)
    parser.add_argument("precision", choices=("bf16", "fp8"))
    parser.add_argument("profile", choices=("c128", "wide"))
    args = parser.parse_args()
    build = json.loads(args.build_json.read_text())

    require(build.get("arch") == "gfx942", "asset architecture is not gfx942")
    require(build.get("n_cu") == 304, "asset CU count is not MI300X's 304")
    precision = build.get("precision", {})
    expected = (
        {"weight_enc": "bf16", "act_enc": "bf16", "kv_enc": "bf16"}
        if args.precision == "bf16"
        else {"weight_enc": "fp8", "act_enc": "mixed", "kv_enc": "bf16"}
    )
    for axis, value in expected.items():
        require(precision.get(axis) == value, f"{axis} is not {value}")

    shapes = build.get("shapes", {})
    expected_prefill = (
        [128, 512, 1024]
        if args.profile == "c128"
        else [128, 512, 1024, 2048, 4096, 8192]
    )
    require(shapes.get("prefill_buckets") == expected_prefill, "wrong prefill ladder")
    programs = build.get("programs", [])
    decode = sorted({p.get("batch") for p in programs if p.get("kind") == "decode"})
    expected_decode = (
        [1, 2, 4, 8, 16, 32, 64, 128]
        if args.profile == "c128"
        else [1, 2, 4, 8]
    )
    require(decode == expected_decode, "wrong decode ladder")
    packed = build.get("objects", {}).get("packed_prefill", {})
    require(packed.get("required") is True, "packed-prefill sibling topology is absent")
    topologies = {p.get("topology") for p in programs if p.get("kind") == "prefill"}
    require(
        {"ordinary", "packed"}.issubset(topologies),
        "ordinary and packed-only prefill topologies are not both present",
    )

    tuning = build.get("tuning", {})
    lookups = tuning.get("tile_lookups", 0)
    measured = tuning.get("tile_measured", 0)
    require(lookups > 0, "manifest records no GEMM tuning lookups")
    require(
        tuning.get("tile_source") == "measured" and measured == lookups,
        f"GEMM tuning is not complete ({measured}/{lookups} measured)",
    )
    lean = build.get("lean", {})
    require(lean.get("verified") is True, "Lean program verification did not complete")
    require(lean.get("oracle") is True, "Lean scheduling oracle did not complete")

    replay = build.get("emit_config", {}).get("replay", {})
    required_replay = {
        "PLOW_DECODE_BATCH": str(expected_decode[-1]),
        "PLOW_DECODE_BATCH_LADDER": ",".join(map(str, expected_decode)),
        "PLOW_DENSE_PF_NS": "1",
        "PLOW_EMIT_PACKED_PREFILL": "1",
        "PLOW_L2_PLACE_PREFILL": "0",
        "PLOW_MAX_CHUNK": str(expected_prefill[-1]),
    }
    for name, value in required_replay.items():
        require(str(replay.get(name)) == value, f"emit replay does not pin {name}={value}")

    print(
        f"qualified {args.precision} manifest: {len(shapes['prefill_buckets'])} prefill rungs, "
        f"{len(decode)} decode rungs, {measured}/{lookups} measured GEMM choices"
    )


if __name__ == "__main__":
    main()
