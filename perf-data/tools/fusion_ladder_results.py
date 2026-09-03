#!/usr/bin/env python3
"""Validate and aggregate model-independent prefill-fusion ladder rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


SCHEMA = "plow.fusion-ladder.row.v1"
SCOPES = ("kernel", "operator-chain", "block", "composed-block", "network")
SHA256_LEN = 64
COMMON_GATES = {
    "measured",
    "artifact_verified",
    "configuration_verified",
    "correctness_passed",
    "route_verified",
    "coverage_complete",
    "decisive",
    "regression_free",
    "paired_measurement",
}
SCOPE_GATES = {
    "kernel": {"exact_unfused", "intermediate_boundaries_verified", "resource_audit_complete"},
    "operator-chain": {"delta_reconciled"},
    "block": {"architecture_classes_complete"},
    "composed-block": {"qualified_inputs_only", "incremental_composition_checked"},
    "network": {
        "production_path",
        "cold_prefill_measured",
        "warm_prefill_measured",
        "decode_regression_checked",
        "collective_time_measured",
        "memory_measured",
    },
}


class RowError(ValueError):
    pass


def fail(where: str, message: str) -> None:
    raise RowError(f"{where}: {message}")


def object_field(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(where, "must be an object")
    return value


def nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        fail(where, "must be a non-empty string")
    return value


def sha256(value: Any, where: str) -> str:
    value = nonempty(value, where)
    if len(value) != SHA256_LEN or any(c not in "0123456789abcdef" for c in value):
        fail(where, "must be a lowercase SHA-256 digest")
    return value


def configuration_sha256(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def samples(value: Any, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 3:
        fail(where, "must contain at least three samples")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            fail(f"{where}[{index}]", "must be numeric")
        item = float(item)
        if not math.isfinite(item) or item <= 0:
            fail(f"{where}[{index}]", "must be finite and positive")
        result.append(item)
    return result


def validate_row(row: Any, where: str) -> dict[str, Any]:
    row = object_field(row, where)
    required = {
        "schema", "campaign_id", "candidate_id", "scope", "shape_id", "shape", "target",
        "artifacts", "configuration", "correctness", "latency", "gates", "decision",
    }
    missing = sorted(required - row.keys())
    if missing:
        fail(where, f"missing fields: {', '.join(missing)}")
    allowed = required | {"notes"}
    extra = sorted(row.keys() - allowed)
    if extra:
        fail(where, f"unknown fields: {', '.join(extra)}")
    if "notes" in row and not isinstance(row["notes"], str):
        fail(f"{where}.notes", "must be a string")
    if row["schema"] != SCHEMA:
        fail(f"{where}.schema", f"must equal {SCHEMA!r}")
    for name in ("campaign_id", "candidate_id", "shape_id"):
        nonempty(row[name], f"{where}.{name}")
    shape = object_field(row["shape"], f"{where}.shape")
    if not shape:
        fail(f"{where}.shape", "must not be empty")
    scope = row["scope"]
    if scope not in SCOPES:
        fail(f"{where}.scope", f"must be one of {', '.join(SCOPES)}")

    target = object_field(row["target"], f"{where}.target")
    if set(target) != {"vendor", "arch", "device", "tp"}:
        fail(f"{where}.target", "must contain exactly vendor, arch, device, tp")
    for name in ("vendor", "arch", "device"):
        nonempty(target[name], f"{where}.target.{name}")
    if isinstance(target["tp"], bool) or not isinstance(target["tp"], int) or target["tp"] < 1:
        fail(f"{where}.target.tp", "must be an integer >= 1")

    artifacts = object_field(row["artifacts"], f"{where}.artifacts")
    artifact_fields = {"candidate_sha256", "baseline_sha256", "config_sha256"}
    if set(artifacts) != artifact_fields:
        fail(f"{where}.artifacts", f"must contain exactly {', '.join(sorted(artifact_fields))}")
    for name in artifact_fields:
        sha256(artifacts[name], f"{where}.artifacts.{name}")
    configuration = object_field(row["configuration"], f"{where}.configuration")
    if not configuration:
        fail(f"{where}.configuration", "must not be empty")
    try:
        actual_config_sha256 = configuration_sha256(configuration)
    except (TypeError, ValueError) as error:
        fail(f"{where}.configuration", f"is not canonical JSON: {error}")
    if actual_config_sha256 != artifacts["config_sha256"]:
        fail(f"{where}.artifacts.config_sha256", "does not match canonical configuration JSON")

    correctness = object_field(row["correctness"], f"{where}.correctness")
    correctness_required = {
        "passed", "method", "checksum_algorithm", "candidate_checksum", "baseline_checksum",
    }
    if not correctness_required <= correctness.keys():
        fail(f"{where}.correctness", "missing pass, method, or checksum evidence")
    if correctness.keys() - (correctness_required | {"tolerance"}):
        fail(f"{where}.correctness", "contains unknown fields")
    if not isinstance(correctness["passed"], bool):
        fail(f"{where}.correctness.passed", "must be boolean")
    if "tolerance" in correctness and correctness["tolerance"] is not None:
        object_field(correctness["tolerance"], f"{where}.correctness.tolerance")
    for name in ("method", "checksum_algorithm", "candidate_checksum", "baseline_checksum"):
        nonempty(correctness[name], f"{where}.correctness.{name}")

    latency = object_field(row["latency"], f"{where}.latency")
    if set(latency) != {"metric", "unit", "candidate_samples", "baseline_samples"}:
        fail(f"{where}.latency", "must contain exactly metric, unit, and both sample arrays")
    nonempty(latency["metric"], f"{where}.latency.metric")
    if latency["unit"] not in ("ns", "us", "ms"):
        fail(f"{where}.latency.unit", "must be ns, us, or ms")
    candidate = samples(latency["candidate_samples"], f"{where}.latency.candidate_samples")
    baseline = samples(latency["baseline_samples"], f"{where}.latency.baseline_samples")
    if len(candidate) != len(baseline):
        fail(f"{where}.latency", "candidate and baseline sample counts must match")

    gates = object_field(row["gates"], f"{where}.gates")
    required_gates = COMMON_GATES | SCOPE_GATES[scope]
    missing_gates = sorted(required_gates - gates.keys())
    if missing_gates:
        fail(f"{where}.gates", f"missing scope gates: {', '.join(missing_gates)}")
    for name, value in gates.items():
        if not isinstance(value, bool):
            fail(f"{where}.gates.{name}", "must be boolean")
    if correctness["passed"] != gates["correctness_passed"]:
        fail(f"{where}.gates.correctness_passed", "must match correctness.passed")
    gates_pass = all(gates[name] for name in required_gates)
    correctness_pass = correctness["passed"] and gates["correctness_passed"]
    measured_win = statistics.median(candidate) < statistics.median(baseline)
    expected_decision = "pass" if gates_pass and correctness_pass and measured_win else "fail"
    if row["decision"] != expected_decision:
        fail(
            f"{where}.decision",
            f"must be {expected_decision!r} from correctness, required gates, and median latency",
        )
    return row


def load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text()
    except OSError as error:
        raise RowError(f"{path}: {error}") from error
    rows = []
    if path.suffix == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RowError(f"{path}:{line_number}: {error}") from error
    else:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise RowError(f"{path}: {error}") from error
        rows = document if isinstance(document, list) else [document]
    if not rows:
        raise RowError(f"{path}: no result rows")
    return [validate_row(row, f"{path}:row[{index}]") for index, row in enumerate(rows)]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    campaigns: dict[tuple[Any, ...], dict[str, Any]] = {}
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        target = row["target"]
        key = (
            row["campaign_id"], row["candidate_id"], target["vendor"], target["arch"],
            target["device"], target["tp"],
        )
        row_key = key + (row["scope"], row["shape_id"])
        if row_key in seen:
            raise RowError(
                f"duplicate result for campaign={row['campaign_id']!r}, "
                f"candidate={row['candidate_id']!r}, scope={row['scope']!r}, "
                f"shape={row['shape_id']!r}"
            )
        seen.add(row_key)
        campaign = campaigns.setdefault(
            key,
            {
                "campaign_id": row["campaign_id"],
                "candidate_id": row["candidate_id"],
                "target": target,
                "scopes": {scope: [] for scope in SCOPES},
            },
        )
        latency = row["latency"]
        campaign["scopes"][row["scope"]].append(
            {
                "shape_id": row["shape_id"],
                "decision": row["decision"],
                "metric": latency["metric"],
                "unit": latency["unit"],
                "config_sha256": row["artifacts"]["config_sha256"],
                "candidate_sha256": row["artifacts"]["candidate_sha256"],
                "baseline_sha256": row["artifacts"]["baseline_sha256"],
                "candidate_median": statistics.median(latency["candidate_samples"]),
                "baseline_median": statistics.median(latency["baseline_samples"]),
            }
        )

    result = []
    for campaign in campaigns.values():
        for scope in SCOPES:
            campaign["scopes"][scope].sort(key=lambda item: item["shape_id"])
        missing = [scope for scope in SCOPES if not campaign["scopes"][scope]]
        failed = [
            f"{scope}:{item['shape_id']}"
            for scope in SCOPES
            for item in campaign["scopes"][scope]
            if item["decision"] != "pass"
        ]
        campaign["qualified"] = not missing and not failed
        campaign["missing_scopes"] = missing
        campaign["failed_rows"] = failed
        result.append(campaign)
    result.sort(key=lambda item: (item["campaign_id"], item["candidate_id"]))
    return {"schema": "plow.fusion-ladder.aggregate.v1", "campaigns": result}


def self_test() -> None:
    configuration = {"dtype": "bf16"}
    base = {
        "schema": SCHEMA,
        "campaign_id": "self-test",
        "candidate_id": "generic-fusion",
        "shape_id": "m128-n256-k512",
        "shape": {"m": 128, "n": 256, "k": 512, "dtype": "bf16"},
        "target": {"vendor": "amd", "arch": "gfx950", "device": "test", "tp": 1},
        "artifacts": {
            "candidate_sha256": "a" * 64,
            "baseline_sha256": "b" * 64,
            "config_sha256": configuration_sha256(configuration),
        },
        "configuration": configuration,
        "correctness": {
            "passed": True,
            "method": "exact-unfused-oracle",
            "checksum_algorithm": "sha256",
            "candidate_checksum": "candidate-output",
            "baseline_checksum": "baseline-output",
        },
        "latency": {
            "metric": "elapsed",
            "unit": "us",
            "candidate_samples": [9.0, 9.1, 9.2],
            "baseline_samples": [10.0, 10.1, 10.2],
        },
        "decision": "pass",
    }
    rows = []
    for scope in SCOPES:
        row = json.loads(json.dumps(base))
        row["scope"] = scope
        row["shape_id"] = f"{scope}-shape"
        row["configuration"] = {"dtype": "bf16", "scope": scope}
        row["artifacts"]["config_sha256"] = configuration_sha256(row["configuration"])
        row["gates"] = {name: True for name in COMMON_GATES | SCOPE_GATES[scope]}
        rows.append(validate_row(row, f"self-test:{scope}"))
    assert aggregate(rows)["campaigns"][0]["qualified"] is True
    rows[0]["decision"] = "fail"
    try:
        validate_row(rows[0], "self-test:bad-decision")
    except RowError:
        pass
    else:
        raise AssertionError("validator accepted a false failure decision")
    incomplete = aggregate(rows[1:])["campaigns"][0]
    assert incomplete["qualified"] is False and incomplete["missing_scopes"] == ["kernel"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("input", type=Path)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("input", type=Path)
    aggregate_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("PASS: fusion ladder result validator self-test")
            return 0
        if args.command is None:
            parser.error("a command or --self-test is required")
        rows = load_rows(args.input)
        if args.command == "validate":
            print(f"PASS: {len(rows)} fusion ladder result row(s)")
            return 0
        document = aggregate(rows)
        output = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(output)
        else:
            sys.stdout.write(output)
        return 0 if all(item["qualified"] for item in document["campaigns"]) else 2
    except (OSError, RowError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
