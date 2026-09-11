#!/usr/bin/env python3
"""Select exact-profile Gemma-4 H100 kernels with a correctness-first GPU search.

The tuner consumes a packet audit through ``gemma4_ladder_campaign.py`` and a
JSON specification containing profile selectors and benchmark variants. Every
variant supplies separate verification and benchmark commands. Commands receive
the exact profile fields plus ``{variant}``, ``{seed}``, ``{trial}``,
``{warmups}``, and ``{iterations}``.

Verification runs five deterministic seeds for every selected profile and
variant before any timing starts. Benchmark commands then run as three separate
processes and must report exactly 50 positive samples after 10 warmups. Raw
stdout and stderr exist only in a temporary directory under /tmp. The final
summary path must also be outside the repository.

Each command prints one JSON object with this common identity:

    {"profile_key": "...", "variant": "...", "compiled_profile": {...}}

Verification additionally reports ``seed``, ``correct``, ``all_finite``,
``input_sha256``, and ``reference_sha256``. Benchmarking reports ``trial``,
``isolated``, ``correct``, ``warmups``, ``iterations``, and ``samples_us``.
The compiled profile is declared in the spec and includes the cubin hash,
symbol, launch resources, tile, pipeline, memory path, and execution mode. The
tuner rejects any identity mismatch rather than attributing a result to the
wrong object.

The spec reuses the ladder campaign inventory and command placeholders:

    {
      "audit": "/tmp/packet-audit.json",
      "arch": "sm90a",
      "dtype": "bf16",
      "packed_topology": "packed-varlen",
      "max_concurrency": 16,
      "profile_selectors": [
        {"phase": "prefill", "family": "gemm",
         "rung": [128, 256, 512], "n": 3840, "k": [8192, 15360]}
      ],
      "variant_groups": {
        "prefill_gemm": [{
          "name": "native-split-k", "reference": false,
          "profile_selector": {"rung": 128},
          "binary": "/tmp/plow-build/native-gemm-bench",
          "object": "/tmp/plow-build/native-gemm.cubin",
          "verify_command": ["{binary}", "verify", "--profile", "{profile_json}",
                             "--variant", "{variant}", "--seed", "{seed}"],
          "benchmark_command": ["{binary}", "bench", "--profile", "{profile_json}",
                                "--variant", "{variant}", "--warmups", "{warmups}",
                                "--iterations", "{iterations}", "--trial", "{trial}"],
          "compiled_profile": {
            "object_sha256": "...", "kernel_symbol": "...",
            "threads": 384, "warps": 12, "registers": 160,
            "smem_bytes": 196608, "tile": [128, 256, 64], "stages": 4,
            "tma": true, "swizzle": "128b", "spills": 0,
            "segment_mode": "split"
          }
        }]
      }
    }
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = ROOT / "scripts" / "gemma4_ladder_campaign.py"
_CAMPAIGN_SPEC = importlib.util.spec_from_file_location(
    "gemma4_ladder_campaign", CAMPAIGN_PATH
)
campaign = importlib.util.module_from_spec(_CAMPAIGN_SPEC)
_CAMPAIGN_SPEC.loader.exec_module(campaign)

WARMUPS = 10
ITERATIONS = 50
TRIALS = 3
VERIFY_SEEDS = (0, 1, 2, 3, 4)
SHA256 = re.compile(r"[0-9a-f]{64}")
EXECUTION_MODES = {"direct", "persistent", "split"}
RESOURCE_FIELDS = (
    "object_sha256",
    "kernel_symbol",
    "threads",
    "warps",
    "registers",
    "smem_bytes",
    "tile",
    "stages",
    "tma",
    "swizzle",
    "spills",
    "segment_mode",
)


class TunerError(ValueError):
    pass


def read_json(path):
    with open(path) as source:
        return json.load(source)


def external_output(path):
    output = Path(path).resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        return output
    raise TunerError("tuner summaries must be written outside the repository")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_resource(resource):
    if not isinstance(resource, dict) or any(name not in resource for name in RESOURCE_FIELDS):
        raise TunerError(f"compiled_profile requires {', '.join(RESOURCE_FIELDS)}")
    if not SHA256.fullmatch(str(resource["object_sha256"])):
        raise TunerError("compiled_profile has an invalid object_sha256")
    for name in ("threads", "warps", "registers", "stages"):
        if type(resource[name]) is not int or resource[name] <= 0:
            raise TunerError(f"compiled_profile has invalid {name}")
    for name in ("smem_bytes", "spills"):
        if type(resource[name]) is not int or resource[name] < 0:
            raise TunerError(f"compiled_profile has invalid {name}")
    if (
        not isinstance(resource["tile"], list)
        or len(resource["tile"]) != 3
        or any(type(value) is not int or value <= 0 for value in resource["tile"])
    ):
        raise TunerError("compiled_profile tile must contain three positive integers")
    if type(resource["tma"]) is not bool or not isinstance(resource["swizzle"], str):
        raise TunerError("compiled_profile has invalid TMA/swizzle identity")
    if resource["segment_mode"] not in EXECUTION_MODES:
        raise TunerError("compiled_profile segment_mode must be direct, persistent, or split")


def _resolve_file(path, cwd, label):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(cwd) / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise TunerError(f"{label} does not exist: {resolved}")
    return resolved


def validate_variant(variant, cwd):
    required = ("name", "reference", "binary", "verify_command", "benchmark_command", "compiled_profile")
    if any(name not in variant for name in required):
        raise TunerError(f"variant requires {', '.join(required)}")
    if not isinstance(variant["name"], str) or not variant["name"]:
        raise TunerError("variant name must be nonempty")
    if type(variant["reference"]) is not bool:
        raise TunerError(f"{variant['name']}: reference must be boolean")
    validate_resource(variant["compiled_profile"])
    binary = _resolve_file(variant["binary"], cwd, f"{variant['name']} binary")
    object_path = variant.get("object")
    if object_path is not None:
        object_file = _resolve_file(object_path, cwd, f"{variant['name']} object")
        if file_sha256(object_file) != variant["compiled_profile"]["object_sha256"]:
            raise TunerError(f"{variant['name']}: object SHA256 differs from compiled_profile")
    for name in ("verify_command", "benchmark_command"):
        command = variant[name]
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise TunerError(f"{variant['name']}: {name} must be a nonempty string array")
        if not any("{binary}" in part for part in command):
            raise TunerError(f"{variant['name']}: {name} must invoke the declared {{binary}}")
    return {**variant, "binary": str(binary), "binary_sha256": file_sha256(binary)}


def selector_matches(profile, selector):
    if not isinstance(selector, dict) or not selector:
        raise TunerError("profile selectors must be nonempty objects")
    for name, expected in selector.items():
        if name not in profile:
            return False
        values = expected if isinstance(expected, list) else [expected]
        if profile[name] not in values:
            return False
    return True


def select_profiles(profiles, selectors):
    if not isinstance(selectors, list) or not selectors:
        raise TunerError("spec requires explicit profile_selectors")
    selected = [
        profile
        for profile in profiles
        if any(selector_matches(profile, selector) for selector in selectors)
    ]
    if not selected:
        raise TunerError("profile_selectors matched no packet profiles")
    keys = [profile["profile_key"] for profile in selected]
    if len(keys) != len(set(keys)):
        raise TunerError("selected packet profiles are not unique")
    return selected


def variants_for_profile(spec, profile, cwd):
    groups = spec.get("variant_groups") or {}
    group = groups.get(f"{profile['phase']}_{profile['family']}")
    if group is None:
        group = groups.get(profile["family"])
    if not isinstance(group, list) or not group:
        raise TunerError(
            f"missing variants for {profile['phase']}_{profile['family']}"
        )
    selected = [
        variant
        for variant in group
        if "profile_selector" not in variant
        or selector_matches(profile, variant["profile_selector"])
    ]
    if not selected:
        raise TunerError(f"{profile['profile_key']}: no variant matches the exact profile")
    variants = [validate_variant(variant, cwd) for variant in selected]
    names = [variant["name"] for variant in variants]
    if len(names) != len(set(names)):
        raise TunerError(f"{profile['profile_key']}: duplicate variant names")
    references = [variant for variant in variants if variant["reference"]]
    if len(references) != 1:
        raise TunerError(f"{profile['profile_key']}: exactly one reference variant is required")
    return variants


def load_plan(spec):
    for name in ("audit", "arch", "dtype", "packed_topology", "max_concurrency"):
        if name not in spec:
            raise TunerError(f"spec is missing {name}")
    campaign.validate_spec(spec)
    audit = read_json(spec["audit"])
    profiles, _, _ = campaign.audit_inventory(
        audit,
        spec["arch"],
        spec["dtype"],
        spec["packed_topology"],
        spec["max_concurrency"],
    )
    selected = select_profiles(profiles, spec.get("profile_selectors"))
    cwd = str(Path(spec.get("cwd", ROOT)).resolve())
    entries = []
    for profile in selected:
        entries.append(
            {"profile": profile, "variants": variants_for_profile(spec, profile, cwd)}
        )
    return cwd, entries


def _safe_label(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _record(stdout, label):
    records = campaign.parse_json_lines(stdout, label)
    if len(records) != 1:
        raise TunerError(f"{label}: command must print exactly one JSON object")
    return records[0]


def invoke(command, cwd, env, timeout, raw_dir, label):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    stem = _safe_label(label)
    (raw_dir / f"{stem}.stdout").write_text(result.stdout)
    (raw_dir / f"{stem}.stderr").write_text(result.stderr)
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1:] or result.stdout.strip().splitlines()[-1:]
        raise TunerError(f"{label}: exit {result.returncode}: {' '.join(detail)}")
    return _record(result.stdout, label)


def command_fields(profile, variant, **extra):
    return {
        **profile,
        "variant": variant["name"],
        "binary": variant["binary"],
        "binary_sha256": variant["binary_sha256"],
        "profile_json": json.dumps(profile, sort_keys=True, separators=(",", ":")),
        "compiled_profile_json": json.dumps(
            variant["compiled_profile"], sort_keys=True, separators=(",", ":")
        ),
        "warmups": WARMUPS,
        "iterations": ITERATIONS,
        **extra,
    }


def command_env(spec, variant, **extra):
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})
    env.update({str(k): str(v) for k, v in (variant.get("env") or {}).items()})
    env.update(
        {
            "PLOW_TUNER_WARMUPS": str(WARMUPS),
            "PLOW_TUNER_ITERATIONS": str(ITERATIONS),
            **{str(k): str(v) for k, v in extra.items()},
        }
    )
    return env


def validate_identity(record, profile, variant, label):
    if record.get("profile_key") != profile["profile_key"]:
        raise TunerError(f"{label}: profile_key differs from scheduled profile")
    if record.get("variant") != variant["name"]:
        raise TunerError(f"{label}: variant differs from scheduled variant")
    if record.get("compiled_profile") != variant["compiled_profile"]:
        raise TunerError(f"{label}: compiled_profile differs from declared resource identity")


def validate_verification(record, profile, variant, seed, label):
    validate_identity(record, profile, variant, label)
    if record.get("seed") != seed:
        raise TunerError(f"{label}: seed differs from scheduled seed")
    if record.get("correct") is not True or record.get("all_finite") is not True:
        raise TunerError(f"{label}: correctness failed")
    for field in ("input_sha256", "reference_sha256"):
        if not SHA256.fullmatch(str(record.get(field, ""))):
            raise TunerError(f"{label}: invalid {field}")


def validate_benchmark(record, profile, variant, trial, label):
    validate_identity(record, profile, variant, label)
    if record.get("trial") != trial:
        raise TunerError(f"{label}: trial differs from scheduled trial")
    if record.get("isolated") is not True or record.get("correct") is not True:
        raise TunerError(f"{label}: benchmark is not isolated and correct")
    if record.get("warmups") != WARMUPS or record.get("iterations") != ITERATIONS:
        raise TunerError(f"{label}: benchmark protocol must be {WARMUPS}/{ITERATIONS}")
    samples = record.get("samples_us")
    if (
        not isinstance(samples, list)
        or len(samples) != ITERATIONS
        or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in samples)
    ):
        raise TunerError(f"{label}: expected exactly {ITERATIONS} positive timing samples")
    return [float(value) for value in samples]


def verify_all(spec, entries, cwd, raw_dir):
    timeout = int(spec.get("verify_timeout_s", 900))
    identities = collections.defaultdict(dict)
    for entry in entries:
        profile = entry["profile"]
        for variant in entry["variants"]:
            for seed in VERIFY_SEEDS:
                label = f"verify/{profile['profile_key']}/{variant['name']}/seed{seed}"
                fields = command_fields(profile, variant, seed=seed)
                command = campaign.expand_command(variant["verify_command"], fields)
                record = invoke(
                    command,
                    cwd,
                    command_env(spec, variant, PLOW_TUNER_SEED=seed),
                    timeout,
                    raw_dir,
                    label,
                )
                validate_verification(record, profile, variant, seed, label)
                pair = (record["input_sha256"], record["reference_sha256"])
                previous = identities[(profile["profile_key"], seed)]
                previous[variant["name"]] = pair
    for (profile_key, seed), variants in identities.items():
        if len(set(variants.values())) != 1:
            raise TunerError(
                f"{profile_key}/seed{seed}: variants did not verify the same input/reference"
            )


def benchmark_all(spec, entries, cwd, raw_dir):
    timeout = int(spec.get("benchmark_timeout_s", 1800))
    timings = collections.defaultdict(list)
    for trial in range(TRIALS):
        for entry in entries:
            profile = entry["profile"]
            variants = entry["variants"]
            offset = trial % len(variants)
            ordered = variants[offset:] + variants[:offset]
            for variant in ordered:
                label = f"bench/{profile['profile_key']}/{variant['name']}/trial{trial}"
                fields = command_fields(profile, variant, trial=trial, seed=trial)
                command = campaign.expand_command(variant["benchmark_command"], fields)
                record = invoke(
                    command,
                    cwd,
                    command_env(
                        spec,
                        variant,
                        PLOW_TUNER_TRIAL=trial,
                        PLOW_TUNER_SEED=trial,
                    ),
                    timeout,
                    raw_dir,
                    label,
                )
                timings[(profile["profile_key"], variant["name"])].append(
                    validate_benchmark(record, profile, variant, trial, label)
                )
    return timings


def _relative_mad(values):
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values) / center


def select_winners(spec, entries, timings):
    gates = spec.get("gates") or {}
    minimum_speedup = float(gates.get("minimum_speedup", 1.01))
    maximum_trial_slowdown = float(gates.get("maximum_trial_slowdown", 1.02))
    maximum_relative_mad = float(gates.get("maximum_relative_mad", 0.10))
    if minimum_speedup < 1 or maximum_trial_slowdown < 1 or maximum_relative_mad <= 0:
        raise TunerError("invalid selection gates")
    results = []
    for entry in entries:
        profile = entry["profile"]
        variants = entry["variants"]
        reference = next(variant for variant in variants if variant["reference"])
        ref_trials = [
            statistics.median(samples)
            for samples in timings[(profile["profile_key"], reference["name"])]
        ]
        alternatives = []
        for variant in variants:
            samples_by_trial = timings[(profile["profile_key"], variant["name"])]
            if len(samples_by_trial) != TRIALS:
                raise TunerError(f"{profile['profile_key']}/{variant['name']}: missing trials")
            trial_medians = [statistics.median(samples) for samples in samples_by_trial]
            all_samples = [sample for trial_samples in samples_by_trial for sample in trial_samples]
            median_us = statistics.median(trial_medians)
            speedup = statistics.median(
                ref / candidate for ref, candidate in zip(ref_trials, trial_medians)
            )
            no_slow_trial = all(
                candidate <= ref * maximum_trial_slowdown
                for ref, candidate in zip(ref_trials, trial_medians)
            )
            stable = _relative_mad(all_samples) <= maximum_relative_mad
            qualified = variant["reference"] or (
                speedup >= minimum_speedup and no_slow_trial and stable
            )
            alternatives.append(
                {
                    "variant": variant["name"],
                    "reference": variant["reference"],
                    "execution_mode": variant["compiled_profile"]["segment_mode"],
                    "median_us": median_us,
                    "trial_medians_us": trial_medians,
                    "speedup_vs_reference": speedup,
                    "relative_mad": _relative_mad(all_samples),
                    "qualified": qualified,
                    "binary_sha256": variant["binary_sha256"],
                    "compiled_profile": variant["compiled_profile"],
                }
            )
        winner = min((row for row in alternatives if row["qualified"]), key=lambda row: row["median_us"])
        results.append(
            {
                "profile_key": profile["profile_key"],
                "profile": profile,
                "selected_variant": winner["variant"],
                "selected_execution_mode": winner["execution_mode"],
                "selected_binary_sha256": winner["binary_sha256"],
                "selected_compiled_profile": winner["compiled_profile"],
                "alternatives": alternatives,
            }
        )
    return results


def run_tuner(spec, output):
    cwd, entries = load_plan(spec)
    with tempfile.TemporaryDirectory(prefix="plow-gemma4-kernel-tuner-", dir="/tmp") as directory:
        raw_dir = Path(directory)
        verify_all(spec, entries, cwd, raw_dir)
        timings = benchmark_all(spec, entries, cwd, raw_dir)
    winners = select_winners(spec, entries, timings)
    summary = {
        "schema_version": 1,
        "model": "google/gemma-4-12B-it",
        "arch": spec["arch"],
        "dtype": spec["dtype"],
        "protocol": {
            "verify_seeds": list(VERIFY_SEEDS),
            "warmups": WARMUPS,
            "iterations": ITERATIONS,
            "isolated_trials": TRIALS,
        },
        "spec_sha256": hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest(),
        "profiles": winners,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def lease_command(spec_path, summary_path):
    lease = ROOT / "perf-data" / "tools" / "gpulease"
    if not lease.is_file():
        raise TunerError(f"gpulease is missing: {lease}")
    return [
        str(lease),
        "-n",
        "1",
        "gemma4-h100-kernel-tuner",
        "env",
        "PLOW_KERNEL_TUNER_LEASED=1",
        sys.executable,
        str(Path(__file__).resolve()),
        "_run",
        "--spec",
        str(Path(spec_path).resolve()),
        "--summary",
        str(Path(summary_path).resolve()),
    ]


def plan_json(spec):
    _, entries = load_plan(spec)
    families = collections.Counter(
        f"{entry['profile']['phase']}_{entry['profile']['family']}" for entry in entries
    )
    return {
        "model": "google/gemma-4-12B-it",
        "protocol": {
            "verify_seeds": list(VERIFY_SEEDS),
            "warmups": WARMUPS,
            "iterations": ITERATIONS,
            "isolated_trials": TRIALS,
        },
        "profile_count": len(entries),
        "families": dict(sorted(families.items())),
        "profiles": [
            {
                **entry["profile"],
                "variants": [
                    {
                        "name": variant["name"],
                        "reference": variant["reference"],
                        "execution_mode": variant["compiled_profile"]["segment_mode"],
                        "binary_sha256": variant["binary_sha256"],
                        "compiled_profile": variant["compiled_profile"],
                    }
                    for variant in entry["variants"]
                ],
            }
            for entry in entries
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--spec", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--summary", required=True)
    inner = subparsers.add_parser("_run", help=argparse.SUPPRESS)
    inner.add_argument("--spec", required=True)
    inner.add_argument("--summary", required=True)
    args = parser.parse_args(argv)
    try:
        spec_path = Path(args.spec).resolve()
        spec = read_json(spec_path)
        if args.action == "plan":
            json.dump(plan_json(spec), sys.stdout, indent=2, sort_keys=True)
            print()
            return 0
        summary_path = external_output(args.summary)
        if args.action == "run" and os.environ.get("PLOW_KERNEL_TUNER_LEASED") != "1":
            return subprocess.run(lease_command(spec_path, summary_path), check=False).returncode
        if os.environ.get("PLOW_KERNEL_TUNER_LEASED") != "1":
            raise TunerError("internal tuner execution requires gpulease")
        summary = run_tuner(spec, summary_path)
        print(f"profiles={len(summary['profiles'])} summary={summary_path}")
        return 0
    except (
        TunerError,
        campaign.CampaignError,
        KeyError,
        OSError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
