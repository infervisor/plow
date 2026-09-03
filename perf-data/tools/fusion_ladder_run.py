#!/usr/bin/env python3
"""Run an evidence-bound, alternating prefill-fusion promotion ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import fusion_ladder_results as results


MANIFEST_SCHEMA = "plow.fusion-ladder.run.v1"
EVIDENCE_SCHEMA = "plow.fusion-ladder.evidence.v1"
MAX_STDOUT_BYTES = 16 * 1024 * 1024
MAX_STDERR_BYTES = 16 * 1024 * 1024


class RunError(ValueError):
    pass


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def command_digest(argv: list[str]) -> str:
    return canonical_digest(argv)


def hash_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for root_index, root in enumerate(paths):
        if not root.exists() and not root.is_symlink():
            raise RunError(f"artifact does not exist: {root}")
        digest.update(b"root\0")
        digest.update(root_index.to_bytes(8, "big"))
        if root.is_dir():
            entries = [
                (str(child.relative_to(root)), child)
                for child in sorted(root.rglob("*"))
                if child.is_file() or child.is_symlink()
            ]
        else:
            entries = [(".", root)]
        if not entries:
            raise RunError(f"artifact root contains no files: {root}")
        for name, path in entries:
            encoded_name = name.encode()
            digest.update(b"entry\0")
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            if path.is_symlink():
                target = os.readlink(path).encode()
                digest.update(b"link\0")
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
            else:
                digest.update(b"file\0")
                digest.update(path.stat().st_size.to_bytes(8, "big"))
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
    return digest.hexdigest()


def artifact_snapshot(scope_spec: dict[str, Any]) -> dict[str, str]:
    return {
        arm: hash_paths([Path(item) for item in scope_spec[arm]["artifacts"]])
        for arm in ("baseline", "candidate")
    }


def require_artifact_snapshot(
    scope_spec: dict[str, Any], expected: dict[str, str], where: str
) -> None:
    actual = artifact_snapshot(scope_spec)
    changed = [arm for arm in ("baseline", "candidate") if actual[arm] != expected[arm]]
    if changed:
        raise RunError(f"{where}: artifact arm(s) changed during measurement: {', '.join(changed)}")


def require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunError(f"{where} must be an object")
    return value


def require_argv(value: Any, where: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise RunError(f"{where} must be a non-empty string argv array")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = require_object(json.loads(path.read_text()), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise RunError(f"{path}: {error}") from error
    required = {"schema", "campaign_id", "candidate_id", "rounds", "target", "scopes"}
    if set(manifest) != required:
        raise RunError(f"manifest fields must be exactly {', '.join(sorted(required))}")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise RunError(f"manifest schema must be {MANIFEST_SCHEMA}")
    for name in ("campaign_id", "candidate_id"):
        if not isinstance(manifest[name], str) or not manifest[name]:
            raise RunError(f"manifest {name} must be non-empty")
    rounds = manifest["rounds"]
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 3:
        raise RunError("manifest rounds must be an integer >= 3")
    target = require_object(manifest["target"], "manifest target")
    if set(target) != {"vendor", "arch", "device", "tp"}:
        raise RunError("manifest target must contain exactly vendor, arch, device, tp")
    for name in ("vendor", "arch", "device"):
        if not isinstance(target[name], str) or not target[name]:
            raise RunError(f"manifest target {name} must be non-empty")
    if isinstance(target["tp"], bool) or not isinstance(target["tp"], int) or target["tp"] < 1:
        raise RunError("manifest target tp must be an integer >= 1")
    scopes = manifest["scopes"]
    if not isinstance(scopes, list):
        raise RunError("manifest scopes must be an array")
    names = [require_object(scope, "scope").get("scope") for scope in scopes]
    if len(names) != len(results.SCOPES) or set(names) != set(results.SCOPES):
        raise RunError(f"manifest must contain exactly one of every scope: {', '.join(results.SCOPES)}")
    for scope in scopes:
        name = scope["scope"]
        required_scope = {
            "scope", "shape_id", "shape", "configuration", "baseline", "candidate"
        }
        if set(scope) != required_scope:
            raise RunError(f"scope {name} fields must be exactly {', '.join(sorted(required_scope))}")
        if not isinstance(scope["shape_id"], str) or not scope["shape_id"]:
            raise RunError(f"scope {name} shape_id must be non-empty")
        if not require_object(scope["shape"], f"scope {name} shape"):
            raise RunError(f"scope {name} shape must not be empty")
        configuration = require_object(scope["configuration"], f"scope {name} configuration")
        if not configuration:
            raise RunError(f"scope {name} configuration must not be empty")
        try:
            results.configuration_sha256(configuration)
        except (TypeError, ValueError) as error:
            raise RunError(f"scope {name} configuration is not canonical JSON: {error}") from error
        for arm in ("baseline", "candidate"):
            command = require_object(scope[arm], f"scope {name} {arm}")
            if set(command) != {"argv", "artifacts"}:
                raise RunError(f"scope {name} {arm} must contain exactly argv and artifacts")
            require_argv(command["argv"], f"scope {name} {arm} argv")
            artifacts = command["artifacts"]
            if not isinstance(artifacts, list) or not artifacts or any(
                not isinstance(item, str) or not item for item in artifacts
            ):
                raise RunError(f"scope {name} {arm} artifacts must be non-empty paths")
    return manifest


def parse_evidence(stdout: bytes, scope: str, shape_id: str, where: str) -> dict[str, Any]:
    if len(stdout) > MAX_STDOUT_BYTES:
        raise RunError(f"{where}: stdout exceeds {MAX_STDOUT_BYTES} bytes")
    lines = stdout.decode("utf-8", errors="strict").splitlines()
    if not lines:
        raise RunError(f"{where}: missing stdout evidence")
    try:
        evidence = require_object(json.loads(lines[-1]), f"{where} final stdout line")
    except json.JSONDecodeError as error:
        raise RunError(f"{where}: final stdout line is not evidence JSON: {error}") from error
    required = {
        "schema", "scope", "shape_id", "latency", "correctness",
        "route_verified", "coverage_complete", "regression_free", "gates",
    }
    if set(evidence) != required:
        raise RunError(f"{where}: evidence fields must be exactly {', '.join(sorted(required))}")
    if evidence["schema"] != EVIDENCE_SCHEMA or evidence["scope"] != scope or evidence["shape_id"] != shape_id:
        raise RunError(f"{where}: evidence identity does not match the manifest")
    latency = require_object(evidence["latency"], f"{where} latency")
    if set(latency) != {"metric", "unit", "value"}:
        raise RunError(f"{where}: latency must contain metric, unit, value")
    if not isinstance(latency["metric"], str) or not latency["metric"]:
        raise RunError(f"{where}: latency metric must be non-empty")
    if latency["unit"] not in ("ns", "us", "ms"):
        raise RunError(f"{where}: latency unit must be ns, us, or ms")
    value = latency["value"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RunError(f"{where}: latency value must be finite and positive")
    correctness = require_object(evidence["correctness"], f"{where} correctness")
    required_correctness = {"passed", "method", "checksum_algorithm", "checksum"}
    if set(correctness) - (required_correctness | {"tolerance"}) or not required_correctness <= set(correctness):
        raise RunError(f"{where}: invalid correctness evidence fields")
    if correctness["passed"] is not True:
        raise RunError(f"{where}: command reported failed correctness")
    for field in ("method", "checksum_algorithm", "checksum"):
        if not isinstance(correctness[field], str) or not correctness[field]:
            raise RunError(f"{where}: correctness {field} must be non-empty")
    if "tolerance" in correctness and (
        not isinstance(correctness["tolerance"], dict) or not correctness["tolerance"]
    ):
        raise RunError(f"{where}: correctness tolerance must be a non-empty object")
    for gate in ("route_verified", "coverage_complete", "regression_free"):
        if evidence[gate] is not True:
            raise RunError(f"{where}: command did not prove {gate}")
    gates = require_object(evidence["gates"], f"{where} gates")
    expected_gates = results.SCOPE_GATES[scope]
    if set(gates) != expected_gates or any(value is not True for value in gates.values()):
        raise RunError(f"{where}: all scope gates must be explicitly true: {', '.join(sorted(expected_gates))}")
    return evidence


def run_command(argv: list[str], log_dir: Path, scope: str, shape_id: str) -> dict[str, Any]:
    log_dir.mkdir(parents=True)
    (log_dir / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (log_dir / "stdout.log").write_bytes(completed.stdout)
    (log_dir / "stderr.log").write_bytes(completed.stderr)
    if len(completed.stderr) > MAX_STDERR_BYTES:
        raise RunError(f"{log_dir}: stderr exceeds {MAX_STDERR_BYTES} bytes")
    if completed.returncode != 0:
        raise RunError(f"{log_dir}: command exited {completed.returncode}")
    return parse_evidence(completed.stdout, scope, shape_id, str(log_dir))


def validate_paired_correctness(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], scope: str
) -> tuple[str, str, dict[str, Any] | None]:
    if len(baseline) != len(candidate) or not baseline:
        raise RunError(f"scope {scope}: correctness evidence is not paired")
    if any(item["correctness"]["passed"] is not True for item in baseline + candidate):
        raise RunError(f"scope {scope}: paired correctness evidence did not pass")
    contracts = {
        (item["correctness"]["method"], item["correctness"]["checksum_algorithm"])
        for item in baseline + candidate
    }
    if len(contracts) != 1:
        raise RunError(f"scope {scope}: commands reported inconsistent correctness contracts")
    tolerances = [item["correctness"].get("tolerance") for item in baseline + candidate]
    has_tolerance = any(tolerance is not None for tolerance in tolerances)
    if has_tolerance:
        if any(not isinstance(tolerance, dict) for tolerance in tolerances):
            raise RunError(f"scope {scope}: tolerance must be explicit on both arms in every round")
        encoded = {json.dumps(item, sort_keys=True, separators=(",", ":")) for item in tolerances}
        if len(encoded) != 1:
            raise RunError(f"scope {scope}: baseline/candidate tolerance contracts differ")
        tolerance = tolerances[0]
    else:
        tolerance = None
        for round_index, (base, arm) in enumerate(zip(baseline, candidate), 1):
            if base["correctness"]["checksum"] != arm["correctness"]["checksum"]:
                raise RunError(
                    f"scope {scope} round {round_index}: baseline/candidate checksums differ without tolerance"
                )
    method, checksum_algorithm = next(iter(contracts))
    return method, checksum_algorithm, tolerance


def run_ladder(manifest_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output_dir / "manifest.json")
    rows = []
    for scope_spec in manifest["scopes"]:
        scope = scope_spec["scope"]
        shape_id = scope_spec["shape_id"]
        arm_digests = artifact_snapshot(scope_spec)
        command_digests = {}
        for arm in ("baseline", "candidate"):
            command = scope_spec[arm]
            command_digests[arm] = command_digest(command["argv"])
        observations: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
        for round_index in range(manifest["rounds"]):
            order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
            for order_index, arm in enumerate(order):
                where = output_dir / scope / f"round-{round_index + 1:02d}-{order_index + 1}-{arm}"
                observations[arm].append(
                    run_command(scope_spec[arm]["argv"], where, scope, shape_id)
                )
                require_artifact_snapshot(scope_spec, arm_digests, str(where))
        require_artifact_snapshot(scope_spec, arm_digests, f"scope {scope} final snapshot")
        all_evidence = observations["baseline"] + observations["candidate"]
        metrics = {(item["latency"]["metric"], item["latency"]["unit"]) for item in all_evidence}
        if len(metrics) != 1:
            raise RunError(f"scope {scope}: commands reported inconsistent latency contracts")
        method, checksum_algorithm, tolerance = validate_paired_correctness(
            observations["baseline"], observations["candidate"], scope
        )
        baseline_samples = [float(item["latency"]["value"]) for item in observations["baseline"]]
        candidate_samples = [float(item["latency"]["value"]) for item in observations["candidate"]]
        decisive = all(candidate < baseline for candidate, baseline in zip(candidate_samples, baseline_samples))
        correctness_passed = all(item["correctness"]["passed"] for item in all_evidence)
        configuration = scope_spec["configuration"]
        gates = {name: True for name in results.COMMON_GATES | results.SCOPE_GATES[scope]}
        gates["decisive"] = decisive
        gates["correctness_passed"] = correctness_passed
        candidate_checksums = [item["correctness"]["checksum"] for item in observations["candidate"]]
        baseline_checksums = [item["correctness"]["checksum"] for item in observations["baseline"]]
        row = {
            "schema": results.SCHEMA,
            "campaign_id": manifest["campaign_id"],
            "candidate_id": manifest["candidate_id"],
            "scope": scope,
            "shape_id": shape_id,
            "shape": scope_spec["shape"],
            "target": manifest["target"],
            "artifacts": {
                "candidate_sha256": arm_digests["candidate"],
                "baseline_sha256": arm_digests["baseline"],
                "config_sha256": results.configuration_sha256(configuration),
                "candidate_command_sha256": command_digests["candidate"],
                "baseline_command_sha256": command_digests["baseline"],
            },
            "configuration": configuration,
            "correctness": {
                "passed": correctness_passed,
                "method": method,
                "checksum_algorithm": checksum_algorithm,
                "candidate_checksum": canonical_digest(candidate_checksums),
                "baseline_checksum": canonical_digest(baseline_checksums),
            },
            "latency": {
                "metric": next(iter(metrics))[0],
                "unit": next(iter(metrics))[1],
                "candidate_samples": candidate_samples,
                "baseline_samples": baseline_samples,
            },
            "gates": gates,
            "decision": "pass" if decisive and correctness_passed else "fail",
        }
        if tolerance is not None:
            row["correctness"]["tolerance"] = tolerance
        rows.append(results.validate_row(row, f"generated:{scope}"))
    rows_path = output_dir / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    aggregate = results.aggregate(rows)
    (output_dir / "promotion.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    return rows


def evidence(scope: str, shape_id: str, latency: float) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_SCHEMA,
        "scope": scope,
        "shape_id": shape_id,
        "latency": {"metric": "elapsed", "unit": "us", "value": latency},
        "correctness": {
            "passed": True,
            "method": "oracle",
            "checksum_algorithm": "sha256",
            "checksum": "output",
        },
        "route_verified": True,
        "coverage_complete": True,
        "regression_free": True,
        "gates": {name: True for name in results.SCOPE_GATES[scope]},
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="fusion-ladder-run-") as directory:
        root = Path(directory)
        artifact = root / "artifact"
        artifact.write_text("stable")
        scopes = []
        for scope in results.SCOPES:
            shape_id = f"{scope}-shape"
            base = json.dumps(evidence(scope, shape_id, 10.0), separators=(",", ":"))
            candidate = json.dumps(evidence(scope, shape_id, 9.0), separators=(",", ":"))
            scopes.append({
                "scope": scope,
                "shape_id": shape_id,
                "shape": {"rows": 128},
                "configuration": {"dtype": "bf16", "scope": scope},
                "baseline": {"argv": [sys.executable, "-c", f"print({base!r})"], "artifacts": [str(artifact)]},
                "candidate": {"argv": [sys.executable, "-c", f"print({candidate!r})"], "artifacts": [str(artifact)]},
            })
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "campaign_id": "self-test",
            "candidate_id": "fusion",
            "rounds": 3,
            "target": {"vendor": "amd", "arch": "gfx950", "device": "test", "tp": 1},
            "scopes": scopes,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        rows = run_ladder(manifest_path, root / "run")
        assert len(rows) == 5 and all(row["decision"] == "pass" for row in rows)
        assert len(list((root / "run").glob("*/round-*/stdout.log"))) == 30
        assert (root / "run/kernel/round-01-1-baseline/stdout.log").is_file()
        assert (root / "run/kernel/round-02-1-candidate/stdout.log").is_file()
        bad = json.loads(json.dumps(manifest))
        bad["scopes"].pop()
        bad_path = root / "bad.json"
        bad_path.write_text(json.dumps(bad))
        try:
            load_manifest(bad_path)
        except RunError:
            pass
        else:
            raise AssertionError("runner accepted an incomplete ladder")
        invalid = evidence("kernel", "kernel-shape", 1.0)
        invalid["route_verified"] = False
        try:
            parse_evidence(
                (json.dumps(invalid) + "\n").encode(),
                "kernel",
                "kernel-shape",
                "self-test-invalid-evidence",
            )
        except RunError:
            pass
        else:
            raise AssertionError("runner accepted a failed command-evidenced gate")

        baseline_dir = root / "baseline-artifacts"
        candidate_dir = root / "candidate-artifacts"
        baseline_dir.mkdir()
        candidate_dir.mkdir()
        (baseline_dir / "same.bin").write_text("baseline")
        (candidate_dir / "same.bin").write_text("candidate")
        assert hash_paths([baseline_dir, candidate_dir]) != hash_paths(
            [candidate_dir, baseline_dir]
        )
        mutation_spec = {
            "baseline": {"artifacts": [str(baseline_dir)]},
            "candidate": {"artifacts": [str(candidate_dir)]},
        }
        snapshot = artifact_snapshot(mutation_spec)
        (baseline_dir / "same.bin").write_text("mutated-by-candidate")
        try:
            require_artifact_snapshot(mutation_spec, snapshot, "self-test-cross-arm-mutation")
        except RunError:
            pass
        else:
            raise AssertionError("runner missed a mutation of the other artifact arm")

        base_evidence = evidence("kernel", "kernel-shape", 10.0)
        candidate_evidence = evidence("kernel", "kernel-shape", 9.0)
        candidate_evidence["correctness"]["checksum"] = "different"
        try:
            validate_paired_correctness([base_evidence], [candidate_evidence], "kernel")
        except RunError:
            pass
        else:
            raise AssertionError("runner accepted unmatched checksums without tolerance")
        base_evidence["correctness"]["tolerance"] = {"atol": 0.001}
        candidate_evidence["correctness"]["tolerance"] = {"atol": 0.002}
        try:
            validate_paired_correctness([base_evidence], [candidate_evidence], "kernel")
        except RunError:
            pass
        else:
            raise AssertionError("runner accepted mismatched tolerance contracts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("PASS: fusion ladder runner self-test")
            return 0
        if args.manifest is None or args.output is None:
            parser.error("manifest and --output are required")
        rows = run_ladder(args.manifest, args.output)
        qualified = all(row["decision"] == "pass" for row in rows)
        print(f"{'PASS' if qualified else 'FAIL'}: wrote {len(rows)} rows to {args.output}")
        return 0 if qualified else 2
    except (OSError, UnicodeError, RunError, results.RowError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
