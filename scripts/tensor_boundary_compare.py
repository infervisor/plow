#!/usr/bin/env python3
"""Compare layer/op tensors and localize the first excess over a self floor."""

import argparse
from array import array
import json
import math
from pathlib import Path
import sys


def manifest(path):
    data = json.loads(path.read_text())
    rows = {}
    for item in data["tensors"]:
        key = (item["semantic"], item["layer"], item.get("rank", 0))
        rows[key] = item
    return data, rows


def tensor(item):
    path = Path(item["file"])
    raw = path.read_bytes()
    if item["dtype"] == "bf16":
        halves = array("H")
        halves.frombytes(raw)
        if sys.byteorder != "little":
            halves.byteswap()
        words = array("I", (value << 16 for value in halves))
        value = array("f")
        value.frombytes(words.tobytes())
    elif item["dtype"] == "float32":
        value = array("f")
        value.frombytes(raw)
        if sys.byteorder != "little":
            value.byteswap()
    else:
        raise ValueError(f"unsupported dtype {item['dtype']}")
    if "selection" in item:
        value = value[-int(item["selection"]["last_elements"]) :]
    return value


def metrics(candidate, reference):
    if len(candidate) != len(reference):
        raise ValueError(f"element mismatch: {len(candidate)} vs {len(reference)}")
    delta2 = math.fsum((a - b) * (a - b) for a, b in zip(candidate, reference))
    ref2 = math.fsum(b * b for b in reference)
    cand2 = math.fsum(a * a for a in candidate)
    dot = math.fsum(a * b for a, b in zip(candidate, reference))
    max_abs = max(abs(a - b) for a, b in zip(candidate, reference))
    scale = max(max(abs(b) for b in reference), 1e-30)
    return {
        "rel_l2": math.sqrt(delta2) / max(math.sqrt(ref2), 1e-30),
        "normalized_max_abs": max_abs / scale,
        "max_abs": max_abs,
        "cosine": dot / max(math.sqrt(cand2 * ref2), 1e-30),
        "candidate_rms": math.sqrt(cand2 / len(candidate)),
        "reference_rms": math.sqrt(ref2 / len(reference)),
        "candidate_max_abs": max(abs(a) for a in candidate),
        "reference_max_abs": scale,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--reference-repeat", type=Path)
    p.add_argument("--floor-multiplier", type=float, default=2.0)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    ref_meta, refs = manifest(args.reference)
    cand_meta, candidates = manifest(args.candidate)
    if ref_meta["prompt_sha256_u32le"] != cand_meta["prompt_sha256_u32le"]:
        raise ValueError("candidate and reference token-history hashes differ")
    repeats = None
    if args.reference_repeat:
        repeat_meta, repeats = manifest(args.reference_repeat)
        if repeat_meta["prompt_sha256_u32le"] != ref_meta["prompt_sha256_u32le"]:
            raise ValueError("reference repeat token-history hash differs")
    rows = []
    for key in sorted(set(refs) & set(candidates), key=lambda x: (x[1], x[0], x[2])):
        row = {"semantic": key[0], "layer": key[1], "rank": key[2]}
        row.update(metrics(tensor(candidates[key]), tensor(refs[key])))
        row["candidate_dtype"] = candidates[key]["dtype"]
        row["reference_dtype"] = refs[key]["dtype"]
        row["candidate_source_dtype"] = candidates[key].get("source_dtype")
        row["reference_source_dtype"] = refs[key].get("source_dtype")
        row["candidate_shape"] = candidates[key]["shape"]
        row["reference_shape"] = refs[key]["shape"]
        row["candidate_source_shape"] = candidates[key].get("source_shape")
        row["reference_source_shape"] = refs[key].get("source_shape")
        row["candidate_source_stride"] = candidates[key].get("source_stride")
        row["reference_source_stride"] = refs[key].get("source_stride")
        if repeats is not None and key in repeats:
            floor = metrics(tensor(repeats[key]), tensor(refs[key]))
            row["reference_repeat_floor"] = floor
            row["floor_ratio"] = max(
                row["rel_l2"] / max(floor["rel_l2"], 1e-30),
                row["normalized_max_abs"] / max(floor["normalized_max_abs"], 1e-30),
            )
            row["outside_repeat_floor"] = row["floor_ratio"] > args.floor_multiplier
        rows.append(row)
    if not rows:
        raise RuntimeError("no common semantic/layer/rank tensors")
    excess = [x for x in rows if x.get("outside_repeat_floor")]
    result = {
        "schema": 1,
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "prompt_sha256_u32le": ref_meta["prompt_sha256_u32le"],
        "floor_multiplier": args.floor_multiplier,
        "rows": rows,
        "first_excess": excess[0] if excess else None,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["first_excess"], indent=2))


if __name__ == "__main__":
    main()
