#!/usr/bin/env python3
"""Qualify an attention implementation at a captured tensor boundary.

The gate is intentionally model agnostic.  Each manifest must contain the same
input semantics (normally Q, K, and V) and one or more output semantics.  Input
payload hashes must match exactly, preventing an upstream numerical difference
from being charged to the attention implementation.
"""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys


ABI_SOURCE_SEMANTICS = {
    "latent.q",
    "latent.kv",
    "rope.k",
    "weight.q_projection",
    "weight.kv_projection",
    "weight.output_projection",
}


def load_manifest(path):
    data = json.loads(path.read_text())
    rows = {}
    for item in data["tensors"]:
        key = (item["semantic"], item.get("layer", 0), item.get("rank", 0))
        if key in rows:
            raise ValueError(f"{path}: duplicate tensor {key}")
        rows[key] = item
    return data, rows


def payload(item):
    path = Path(item["file"])
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != item["sha256"]:
        raise ValueError(f"{path}: payload hash is {digest}, manifest says {item['sha256']}")
    if "selection" in item:
        n = int(item["selection"]["last_elements"])
        width = {"bf16": 2, "float32": 4}[item["dtype"]]
        raw = raw[-n * width :]
    return raw


def values(item):
    raw = payload(item)
    if item["dtype"] == "bf16":
        halves = array("H")
        halves.frombytes(raw)
        if sys.byteorder != "little":
            halves.byteswap()
        words = array("I", (x << 16 for x in halves))
        out = array("f")
        out.frombytes(words.tobytes())
        return out
    if item["dtype"] == "float32":
        out = array("f")
        out.frombytes(raw)
        if sys.byteorder != "little":
            out.byteswap()
        return out
    raise ValueError(f"unsupported dtype {item['dtype']}")


def metrics(candidate, reference):
    if len(candidate) != len(reference) or not candidate:
        raise ValueError(f"element mismatch: {len(candidate)} vs {len(reference)}")
    if not all(math.isfinite(x) for x in candidate) or not all(
        math.isfinite(x) for x in reference
    ):
        raise ValueError("non-finite boundary tensor")
    delta2 = math.fsum((a - b) ** 2 for a, b in zip(candidate, reference))
    ref2 = math.fsum(b * b for b in reference)
    cand2 = math.fsum(a * a for a in candidate)
    dot = math.fsum(a * b for a, b in zip(candidate, reference))
    max_abs = max(abs(a - b) for a, b in zip(candidate, reference))
    if cand2 == 0.0 and ref2 == 0.0:
        cosine = 1.0
    elif cand2 == 0.0 or ref2 == 0.0:
        cosine = 0.0
    else:
        cosine = dot / math.sqrt(cand2 * ref2)
    return {
        "rel_l2": math.sqrt(delta2) / max(math.sqrt(ref2), 1e-30),
        "max_abs": max_abs,
        "cosine": cosine,
    }


def parse_semantics(values_):
    return set(x for value in values_ for x in value.split(",") if x)


def canonical_dtype(item):
    value = item.get("source_dtype", item["dtype"]).removeprefix("torch.")
    return "bf16" if value == "bfloat16" else value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", action="append", required=True, type=Path)
    p.add_argument("--absorbed", required=True, type=Path)
    p.add_argument("--materialized", required=True, type=Path)
    p.add_argument("--input-semantic", action="append", required=True)
    p.add_argument("--output-semantic", action="append", required=True)
    p.add_argument("--require-output-source-dtype", default="bf16")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    ref_loaded = [load_manifest(x) for x in args.reference]
    if len(ref_loaded) < 2:
        raise ValueError("at least two adjacent reference captures are required")
    absorbed_meta, absorbed = load_manifest(args.absorbed)
    materialized_meta, materialized = load_manifest(args.materialized)
    metas = [x[0] for x in ref_loaded] + [absorbed_meta, materialized_meta]
    histories = {x["prompt_sha256_u32le"] for x in metas}
    if len(histories) != 1:
        raise ValueError("token-history hashes differ")
    contracts = [x.get("contract") for x in metas]
    if any(x is not None for x in contracts):
        if any(x is None for x in contracts):
            raise ValueError("boundary contract is missing from one or more manifests")
        encoded_contracts = {json.dumps(x, sort_keys=True, separators=(",", ":")) for x in contracts}
        if len(encoded_contracts) != 1:
            raise ValueError("boundary contracts differ")
        contract_sha256 = hashlib.sha256(encoded_contracts.pop().encode()).hexdigest()
    else:
        contract_sha256 = None

    inputs = parse_semantics(args.input_semantic)
    if all(x.get("schema") == "plow.mla-boundary.v1" for x in metas):
        inputs |= ABI_SOURCE_SEMANTICS
    outputs = parse_semantics(args.output_semantic)
    all_rows = [x[1] for x in ref_loaded] + [absorbed, materialized]
    input_hashes = {}
    for key in sorted(set.intersection(*(set(x) for x in all_rows))):
        if key[0] not in inputs:
            continue
        hashes = [hashlib.sha256(payload(rows[key])).hexdigest() for rows in all_rows]
        input_hashes["/".join(map(str, key))] = hashes[0]
        if len(set(hashes)) != 1:
            raise ValueError(f"upstream input payloads differ for {key}: {hashes}")
    found_inputs = {key.split("/", 1)[0] for key in input_hashes}
    if found_inputs != inputs:
        raise ValueError(f"input semantics found {sorted(found_inputs)}, expected {sorted(inputs)}")

    references = [x[1] for x in ref_loaded]
    common = set(absorbed) & set(materialized)
    common &= set.intersection(*(set(x) for x in references))
    keys = sorted((x for x in common if x[0] in outputs), key=lambda x: (x[1], x[0], x[2]))
    if {x[0] for x in keys} != outputs:
        raise ValueError("not every requested output semantic is common to all manifests")

    report_rows = []
    passed = True
    for key in keys:
        output_items = [x[key] for x in references] + [absorbed[key], materialized[key]]
        output_dtypes = [canonical_dtype(x) for x in output_items]
        if args.require_output_source_dtype and any(
            x != args.require_output_source_dtype for x in output_dtypes
        ):
            raise ValueError(
                f"{key}: output source dtypes {output_dtypes}, expected "
                f"{args.require_output_source_dtype}"
            )
        ref_values = [values(x[key]) for x in references]
        floors = [metrics(ref_values[i + 1], ref_values[i]) for i in range(len(ref_values) - 1)]
        floor = {
            "rel_l2": max((x["rel_l2"] for x in floors), default=0.0),
            "max_abs": max((x["max_abs"] for x in floors), default=0.0),
            "cosine_loss": max((1.0 - x["cosine"] for x in floors), default=0.0),
        }
        absorbed_error = metrics(values(absorbed[key]), ref_values[0])
        materialized_error = metrics(values(materialized[key]), ref_values[0])
        limits = {
            "rel_l2": absorbed_error["rel_l2"] + floor["rel_l2"],
            "max_abs": absorbed_error["max_abs"] + floor["max_abs"],
            "cosine_loss": (1.0 - absorbed_error["cosine"]) + floor["cosine_loss"],
        }
        row_passed = (
            materialized_error["rel_l2"] <= limits["rel_l2"]
            and materialized_error["max_abs"] <= limits["max_abs"]
            and 1.0 - materialized_error["cosine"] <= limits["cosine_loss"]
        )
        passed &= row_passed
        report_rows.append(
            {
                "semantic": key[0],
                "layer": key[1],
                "rank": key[2],
                "reference_repeat_floor": floor,
                "absorbed_error": absorbed_error,
                "materialized_error": materialized_error,
                "materialized_limits": limits,
                "passed": row_passed,
            }
        )
    result = {
        "schema": 1,
        "gate": "same-input-attention-boundary",
        "prompt_sha256_u32le": histories.pop(),
        "contract_sha256": contract_sha256,
        "references": [str(x) for x in args.reference],
        "absorbed": str(args.absorbed),
        "materialized": str(args.materialized),
        "input_payload_sha256": input_hashes,
        "rows": report_rows,
        "passed": passed,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": passed, "rows": report_rows}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
