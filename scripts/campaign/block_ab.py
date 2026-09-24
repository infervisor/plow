"""Fail-closed, same-clock scoring of one frozen four-arm modular-block session."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ARMS = ("ctl", "treat", "ctl2", "treat2")


def score(measurements):
    base = measurements["ctl"]
    medians, mads = {}, {}
    for name in ARMS:
        row = measurements[name]
        for field in ("scope", "batch", "ctx", "tp", "clock", "cache_policy", "warmup", "checkpoint"):
            if row[field] != base[field]:
                raise ValueError(f"{name}: mismatched {field}")
        if (row["scope"] != "single-block-decode" or row["trace_instrumented"]
                or row.get("host_timing_instrumented", False)):
            raise ValueError(f"{name}: expected uninstrumented block timings")
        if not all(row[field] is True for field in ("oracle_verified", "rank_identity", "finiteness")):
            raise ValueError(f"{name}: numerical gate failed")
        samples = row["samples_us"]
        if len(samples) < 30 or len(samples) != len(base["samples_us"]):
            raise ValueError(f"{name}: insufficient or unequal sample counts")
        if not all(math.isfinite(v) and v > 0 for v in samples):
            raise ValueError(f"{name}: invalid timing sample")
        medians[name] = sorted(samples)[len(samples) // 2]
        mads[name] = statistics.median(abs(v - medians[name]) for v in samples)
        if not math.isclose(medians[name], row["latency_us_median"], rel_tol=1e-9):
            raise ValueError(f"{name}: inconsistent reported median")
    drift = abs(medians["ctl"] - medians["ctl2"])
    spread = abs(medians["treat"] - medians["treat2"])
    noise = max(drift, spread) + 2 * max(mads.values())
    control = (medians["ctl"] + medians["ctl2"]) / 2
    treatment = (medians["treat"] + medians["treat2"]) / 2
    delta = control - treatment
    stable = spread <= 3 * drift
    return dict(medians_us=medians, mad_us=mads, control_drift_us=drift,
                treatment_spread_us=spread, noise_floor_us=noise, saving_us=delta,
                reduction_percent=100 * delta / control, stable=stable,
                gate_pass=stable and delta > noise, scope="single-block only; no default promotion")


def output_hashes(path):
    files = sorted(path.glob("*.bin"))
    if not files:
        raise ValueError(f"{path}: missing numerical outputs")
    hashes = {}
    for file in files:
        with file.open("rb") as stream:
            hashes[file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def compare(root, require_bitwise=False, routed_repeat_audit=None):
    records = {name: json.loads((root / name / "run-record.json").read_text()) for name in ARMS}
    for name in ARMS:
        for field in ("inputs", "runtime_sha256", "env", "cell"):
            if records[name][field] != records["ctl"][field]:
                raise ValueError(f"{name}: mismatched frozen {field}")
    for first, second in (("ctl", "ctl2"), ("treat", "treat2")):
        for field in ("packet_sha256", "objects"):
            if records[first][field] != records[second][field]:
                raise ValueError(f"{first}/{second}: changed {field}")
    outputs = {name: output_hashes(root / name / "outputs") for name in ARMS}
    if any(outputs[name].keys() != outputs["ctl"].keys() for name in ARMS):
        raise ValueError("arms have different numerical output sets")
    allowed = set()
    if routed_repeat_audit is not None:
        audit = json.loads(routed_repeat_audit.read_text())
        tp = json.loads((root / "ctl/measurement.json").read_text())["tp"]
        expected_checks = {(arm, rank) for arm in ARMS for rank in range(tp)}
        if (require_bitwise or audit.get("scope") != "routed-BF16-atomic-repeat-v1"
                or audit.get("passed") is not True or audit.get("tp") != tp
                or audit.get("output_sha256") != outputs
                or len(audit.get("checks", [])) != len(expected_checks)
                or {(c["arm"], c["rank"]) for c in audit["checks"]} != expected_checks
                or not all(c["stable_boundaries_bitwise"] is True and c["reduction_in_bounds"] is True
                           for c in audit["checks"])):
            raise ValueError("missing, stale, incomplete or incompatible routed repeat audit")
        allowed = {f"rank{rank}.act.{name}.bin" for rank in range(tp) for name in
                   ("moe_fug", "moe_rowtok", "moe_rowpart", "moe_rowgate", "part", "attn", "xnext")}
        if not allowed <= outputs["ctl"].keys():
            raise ValueError("routed repeat audit requires all boundary outputs")
        for arm in ARMS:
            if any(digest != outputs["ctl"][name] for name, digest in outputs[arm].items() if name not in allowed):
                raise ValueError("routed A/B changed an unaudited output")
    for first, second in (("ctl", "ctl2"), ("treat", "treat2")):
        if any(value != outputs[second][name] for name, value in outputs[first].items() if name not in allowed):
            raise ValueError(f"{first}/{second}: nonrepeatable numerical outputs")
    result = score({name: json.loads((root / name / "measurement.json").read_text()) for name in ARMS})
    result["changed_output_files"] = sum(outputs["ctl"][name] != value for name, value in outputs["treat"].items())
    result["output_files_per_arm"] = len(outputs["ctl"])
    result["require_bitwise"] = require_bitwise
    if routed_repeat_audit is not None:
        result["routed_repeat_audit_sha256"] = hashlib.sha256(routed_repeat_audit.read_bytes()).hexdigest()
        result["repeat_contract"] = "bitwise stable boundaries plus BF16 atomic order bounds and shared/TP/residual rounding"
    if require_bitwise and result["changed_output_files"]:
        result.update(gate_pass=False, error="bitwise lever changed numerical outputs")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--routed-repeat-audit", type=Path)
    args = parser.parse_args()
    root = args.root
    (root / "PASS").unlink(missing_ok=True)
    try:
        result = compare(root, args.require_bitwise, args.routed_repeat_audit)
    except (ValueError, KeyError, OSError, TypeError) as error:
        result = dict(gate_pass=False, error=str(error))
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["gate_pass"]:
        (root / "PASS").write_text("single-block candidate only; full-model qualification still required\n")
    sys.exit(0 if result["gate_pass"] else 1)
