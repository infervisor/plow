#!/usr/bin/env python3
"""Select exact-profile Gemma-4 H100 kernels with a correctness-first GPU search.

The tuner consumes a packet audit through ``gemma4_ladder_campaign.py`` and a
JSON specification containing profile selectors and benchmark variants. Every
variant supplies separate verification and benchmark commands. Commands receive
the exact profile fields plus ``{variant}``, ``{seed}``, ``{trial}``,
``{warmups}``, and ``{iterations}``.

Verification runs five deterministic seeds for every selected profile and
variant before any timing starts. Each of four timing trials runs the reference,
the rotated candidate arms, then the reference again. Benchmark commands run as
separate processes and must report exactly 50 positive samples after 10 warmups.
Raw stdout and stderr exist only in a temporary directory under /tmp. The final
summary path must also be outside the repository.

Each command prints one JSON object with this common identity:

    {"profile_key": "...", "packet_sha256": "...", "variant": "...",
     "compiled_profile": {...}}

Verification additionally reports ``seed``, ``correct``, ``all_finite``,
``input_sha256``, ``reference_sha256``, and ``output_sha256``. Benchmarking
reports ``trial``, ``isolated``, ``correct``, ``cache_state``, clock/power
``telemetry``, ``counters``, ``warmups``, ``iterations``, and ``samples_us``.
The compiled profile is declared in the spec and includes the cubin hash,
symbol, launch resources, tile, pipeline, memory path, and execution mode. The
tuner rejects any identity mismatch rather than attributing a result to the
wrong object.

The spec reuses the ladder campaign inventory and command placeholders:

    {
      "audit": "/tmp/packet-audit.json",
      "arch": "sm90a",
      "gpu": "H100 SXM5",
      "toolchain": "nvcc 13.1",
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
          "hypothesis": "split K fills idle SMs at M128",
          "lever": "split-k count",
          "predicted_savings_us": 3.0,
          "expected_counter_changes": {"sm_active": "increase"},
          "profile_selector": {"rung": 128},
          "binary": "/tmp/plow-build/native-gemm-bench",
          "object": "/tmp/plow-build/native-gemm.cubin",
          "verify_command": ["{binary}", "verify", "--profile", "{profile_json}",
                             "--variant", "{variant}", "--seed", "{seed}"],
          "benchmark_command": ["{binary}", "bench", "--profile", "{profile_json}",
                                "--variant", "{variant}", "--warmups", "{warmups}",
                                "--iterations", "{iterations}", "--trial", "{trial}",
                                "--arm", "{arm}"],
          "compiled_profile": {
            "object_sha256": "...", "kernel_symbol": "...",
            "threads": 384, "warps": 12, "registers": 160,
            "smem_bytes": 196608, "tile": [128, 256, 64], "stages": 4,
            "tma": true, "swizzle": "128b", "spills": 0,
            "stack_bytes": 0, "spill_store_bytes": 0,
            "spill_load_bytes": 0,
            "segment_mode": "split",
            "sm_count": 132, "launch_blocks": 132,
            "blocks_per_sm": 1, "cluster": [1, 1, 1]
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
TRIALS = 4
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
    "stack_bytes",
    "spill_store_bytes",
    "spill_load_bytes",
    "segment_mode",
    "sm_count",
    "launch_blocks",
    "blocks_per_sm",
    "cluster",
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
    for name in (
        "threads", "warps", "registers", "stages", "sm_count",
        "launch_blocks", "blocks_per_sm",
    ):
        if type(resource[name]) is not int or resource[name] <= 0:
            raise TunerError(f"compiled_profile has invalid {name}")
    if resource["threads"] % 32 or resource["warps"] != resource["threads"] // 32:
        raise TunerError("compiled_profile warps must match the SM90 thread count")
    for name in (
        "smem_bytes", "spills", "stack_bytes", "spill_store_bytes",
        "spill_load_bytes",
    ):
        if type(resource[name]) is not int or resource[name] < 0:
            raise TunerError(f"compiled_profile has invalid {name}")
    if (
        not isinstance(resource["tile"], list)
        or len(resource["tile"]) != 3
        or any(type(value) is not int or value <= 0 for value in resource["tile"])
    ):
        raise TunerError("compiled_profile tile must contain three positive integers")
    if (
        not isinstance(resource["cluster"], list)
        or len(resource["cluster"]) != 3
        or any(type(value) is not int or value <= 0 for value in resource["cluster"])
    ):
        raise TunerError("compiled_profile cluster must contain three positive integers")
    if resource["smem_bytes"] > 227328:
        raise TunerError("compiled_profile exceeds the H100 dynamic shared-memory limit")
    if resource["smem_bytes"] * resource["blocks_per_sm"] > 227328:
        raise TunerError("compiled_profile occupancy exceeds H100 shared memory per SM")
    if (
        resource["registers"] * resource["threads"] * resource["blocks_per_sm"]
        > 65536
    ):
        raise TunerError("compiled_profile occupancy exceeds the H100 register file")
    cluster_blocks = math.prod(resource["cluster"])
    if resource["launch_blocks"] % cluster_blocks:
        raise TunerError("compiled_profile launch grid is not divisible by its cluster")
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
    if not variant["reference"]:
        for name in ("hypothesis", "lever"):
            if not isinstance(variant.get(name), str) or not variant[name].strip():
                raise TunerError(f"{variant['name']}: candidate requires nonempty {name}")
        predicted = variant.get("predicted_savings_us")
        if type(predicted) not in (int, float) or not math.isfinite(predicted) or predicted <= 0:
            raise TunerError(
                f"{variant['name']}: candidate requires positive predicted_savings_us"
            )
        counters = variant.get("expected_counter_changes")
        if not isinstance(counters, dict) or not counters or any(
            not isinstance(name, str) or not name
            or change not in ("increase", "decrease")
            for name, change in counters.items()
        ):
            raise TunerError(
                f"{variant['name']}: expected_counter_changes must map counters to increase/decrease"
            )
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
    if len(variants) < 2:
        raise TunerError(f"{profile['profile_key']}: at least one candidate is required")
    return variants


def load_plan(spec):
    for name in (
        "audit", "arch", "gpu", "toolchain", "dtype", "packed_topology",
        "max_concurrency",
    ):
        if name not in spec:
            raise TunerError(f"spec is missing {name}")
    for name in ("gpu", "toolchain"):
        if not isinstance(spec[name], str) or not spec[name].strip():
            raise TunerError(f"spec {name} must be nonempty")
    campaign.validate_spec(spec)
    audit = read_json(spec["audit"])
    packet_sha256 = audit.get("packet_sha256")
    if not SHA256.fullmatch(str(packet_sha256 or "")):
        raise TunerError("packet audit has no valid packet SHA256")
    profiles, _, _ = campaign.audit_inventory(
        audit,
        spec["arch"],
        spec["dtype"],
        spec["packed_topology"],
        spec["max_concurrency"],
    )
    selected = select_profiles(profiles, spec.get("profile_selectors"))
    for profile in selected:
        profile["packet_sha256"] = packet_sha256
    cwd = str(Path(spec.get("cwd", ROOT)).resolve())
    entries = []
    for profile in selected:
        entries.append(
            {"profile": profile, "variants": variants_for_profile(spec, profile, cwd)}
        )
    return cwd, packet_sha256, entries


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
    packet_sha256 = profile.get("packet_sha256")
    if packet_sha256 is not None and record.get("packet_sha256") != packet_sha256:
        raise TunerError(f"{label}: packet SHA256 differs from scheduled packet")


def validate_verification(record, profile, variant, seed, label):
    validate_identity(record, profile, variant, label)
    if record.get("seed") != seed:
        raise TunerError(f"{label}: seed differs from scheduled seed")
    if record.get("correct") is not True or record.get("all_finite") is not True:
        raise TunerError(f"{label}: correctness failed")
    for field in ("input_sha256", "reference_sha256", "output_sha256"):
        if not SHA256.fullmatch(str(record.get(field, ""))):
            raise TunerError(f"{label}: invalid {field}")


def validate_benchmark(record, profile, variant, trial, arm, label):
    validate_identity(record, profile, variant, label)
    if record.get("trial") != trial:
        raise TunerError(f"{label}: trial differs from scheduled trial")
    if record.get("arm") != arm:
        raise TunerError(f"{label}: arm differs from scheduled arm")
    if record.get("isolated") is not True or record.get("correct") is not True:
        raise TunerError(f"{label}: benchmark is not isolated and correct")
    if record.get("cache_state") not in ("hot", "cold"):
        raise TunerError(f"{label}: cache_state must be hot or cold")
    telemetry = record.get("telemetry")
    if not isinstance(telemetry, dict):
        raise TunerError(f"{label}: missing clock/power telemetry")
    for name in ("sm_clock_mhz", "memory_clock_mhz", "power_w"):
        value = telemetry.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise TunerError(f"{label}: invalid telemetry {name}")
    counters = record.get("counters")
    if not isinstance(counters, dict) or any(
        not isinstance(name, str) or not name
        or type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for name, value in counters.items()
    ):
        raise TunerError(f"{label}: counters must be nonnegative finite numbers")
    if record.get("warmups") != WARMUPS or record.get("iterations") != ITERATIONS:
        raise TunerError(f"{label}: benchmark protocol must be {WARMUPS}/{ITERATIONS}")
    samples = record.get("samples_us")
    if (
        not isinstance(samples, list)
        or len(samples) != ITERATIONS
        or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in samples)
    ):
        raise TunerError(f"{label}: expected exactly {ITERATIONS} positive timing samples")
    return {
        "trial": trial,
        "arm": arm,
        "samples_us": [float(value) for value in samples],
        "cache_state": record["cache_state"],
        "telemetry": telemetry,
        "counters": {name: float(value) for name, value in counters.items()},
    }


def verify_all(spec, entries, cwd, raw_dir):
    timeout = int(spec.get("verify_timeout_s", 900))
    identities = collections.defaultdict(dict)
    evidence = []
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
                evidence.append({
                    "profile_key": profile["profile_key"],
                    "variant": variant["name"],
                    "seed": seed,
                    "input_sha256": record["input_sha256"],
                    "reference_sha256": record["reference_sha256"],
                    "output_sha256": record["output_sha256"],
                    "correct": True,
                    "all_finite": True,
                })
    for (profile_key, seed), variants in identities.items():
        if len(set(variants.values())) != 1:
            raise TunerError(
                f"{profile_key}/seed{seed}: variants did not verify the same input/reference"
            )
    return evidence


def benchmark_all(spec, entries, cwd, raw_dir):
    timeout = int(spec.get("benchmark_timeout_s", 1800))
    timings = collections.defaultdict(list)
    for trial in range(TRIALS):
        for entry in entries:
            profile = entry["profile"]
            variants = entry["variants"]
            reference = next(variant for variant in variants if variant["reference"])
            candidates = [variant for variant in variants if not variant["reference"]]
            offset = trial % len(candidates)
            ordered = candidates[offset:] + candidates[:offset]
            arms = [(reference, "control_before")]
            arms.extend((variant, "candidate") for variant in ordered)
            arms.append((reference, "control_after"))
            for variant, arm in arms:
                label = (
                    f"bench/{profile['profile_key']}/{variant['name']}/"
                    f"trial{trial}/{arm}"
                )
                fields = command_fields(
                    profile, variant, trial=trial, seed=trial, arm=arm
                )
                command = campaign.expand_command(variant["benchmark_command"], fields)
                record = invoke(
                    command,
                    cwd,
                    command_env(
                        spec,
                        variant,
                        PLOW_TUNER_TRIAL=trial,
                        PLOW_TUNER_SEED=trial,
                        PLOW_TUNER_ARM=arm,
                    ),
                    timeout,
                    raw_dir,
                    label,
                )
                timings[(profile["profile_key"], variant["name"])].append(
                    validate_benchmark(record, profile, variant, trial, arm, label)
                )
    return timings


def _relative_mad(values):
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values) / center


def _absolute_mad(values):
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _trial_median(record):
    return statistics.median(record["samples_us"])


def _reference_trials(records, profile_key):
    if len(records) != 2 * TRIALS:
        raise TunerError(f"{profile_key}: missing control anchors")
    pairs = []
    for trial in range(TRIALS):
        current = [record for record in records if record["trial"] == trial]
        before = [record for record in current if record["arm"] == "control_before"]
        after = [record for record in current if record["arm"] == "control_after"]
        if len(before) != 1 or len(after) != 1:
            raise TunerError(f"{profile_key}/trial{trial}: invalid control anchors")
        pairs.append((before[0], after[0]))
    return pairs


def _counter_evidence(expected, reference_pairs, candidate_trials, minimum_change):
    evidence = {}
    passed = True
    for name, direction in expected.items():
        if any(
            name not in record["counters"]
            for pair in reference_pairs for record in pair
        ) or any(name not in record["counters"] for record in candidate_trials):
            evidence[name] = {"direction": direction, "passed": False, "reason": "missing"}
            passed = False
            continue
        references = [
            (before["counters"][name] + after["counters"][name]) / 2
            for before, after in reference_pairs
        ]
        candidates = [record["counters"][name] for record in candidate_trials]
        reference = statistics.median(references)
        candidate = statistics.median(candidates)
        relative_change = (
            (candidate - reference) / reference if reference != 0 else None
        )
        changes = [
            candidate_value - reference_value
            for reference_value, candidate_value in zip(references, candidates)
        ]
        trial_passes = sum(
            change > 0 if direction == "increase" else change < 0
            for change in changes
        )
        if relative_change is None:
            large_enough = direction == "increase" and candidate > 0
        else:
            direction_change = relative_change if direction == "increase" else -relative_change
            large_enough = direction_change >= minimum_change
        counter_passed = trial_passes >= 3 and large_enough
        evidence[name] = {
            "direction": direction,
            "reference_median": reference,
            "candidate_median": candidate,
            "relative_change": relative_change,
            "passing_trials": trial_passes,
            "passed": counter_passed,
        }
        passed = passed and counter_passed
    return passed, evidence


def select_winners(spec, entries, timings):
    gates = spec.get("gates") or {}
    minimum_speedup = float(gates.get("minimum_speedup", 1.01))
    maximum_trial_slowdown = float(gates.get("maximum_trial_slowdown", 1.02))
    maximum_relative_mad = float(gates.get("maximum_relative_mad", 0.10))
    maximum_clock_drift = float(gates.get("maximum_clock_drift", 0.05))
    minimum_counter_change = float(gates.get("minimum_counter_change", 0.01))
    allow_resource_spills = gates.get("allow_resource_spills", False)
    if (
        minimum_speedup < 1 or maximum_trial_slowdown < 1
        or maximum_relative_mad <= 0 or not 0 < maximum_clock_drift < 1
        or not 0 <= minimum_counter_change < 1
        or type(allow_resource_spills) is not bool
    ):
        raise TunerError("invalid selection gates")
    results = []
    for entry in entries:
        profile = entry["profile"]
        variants = entry["variants"]
        reference = next(variant for variant in variants if variant["reference"])
        reference_records = timings[(profile["profile_key"], reference["name"])]
        reference_pairs = _reference_trials(reference_records, profile["profile_key"])
        ref_trials = [
            (_trial_median(before) + _trial_median(after)) / 2
            for before, after in reference_pairs
        ]
        ref_samples = [
            sample
            for trial in reference_records
            for sample in trial["samples_us"]
        ]
        noise_by_trial = [
            abs(_trial_median(before) - _trial_median(after))
            + 2 * max(
                _absolute_mad(before["samples_us"]),
                _absolute_mad(after["samples_us"]),
            )
            for before, after in reference_pairs
        ]
        noise_floor_us = statistics.median(noise_by_trial)
        reference_stable = _relative_mad(ref_samples) <= maximum_relative_mad
        alternatives = []
        for variant in variants:
            samples_by_trial = timings[(profile["profile_key"], variant["name"])]
            if variant["reference"]:
                trial_medians = ref_trials
                all_samples = ref_samples
                cache_matched = True
                clocks_matched = True
                counter_matched = True
                counter_evidence = {}
            elif len(samples_by_trial) != TRIALS or any(
                trial["trial"] != index or trial["arm"] != "candidate"
                for index, trial in enumerate(samples_by_trial)
            ):
                raise TunerError(f"{profile['profile_key']}/{variant['name']}: missing trials")
            else:
                trial_medians = [_trial_median(trial) for trial in samples_by_trial]
                all_samples = [
                    sample for trial in samples_by_trial for sample in trial["samples_us"]
                ]
                cache_matched = all(
                    candidate["cache_state"]
                    == before["cache_state"]
                    == after["cache_state"]
                    for candidate, (before, after) in zip(
                        samples_by_trial, reference_pairs
                    )
                )
                clocks_matched = all(
                    abs(candidate["telemetry"][name] / anchor["telemetry"][name] - 1)
                    <= maximum_clock_drift
                    for candidate, anchors in zip(samples_by_trial, reference_pairs)
                    for anchor in anchors
                    for name in ("sm_clock_mhz", "memory_clock_mhz")
                )
                counter_matched, counter_evidence = _counter_evidence(
                    variant["expected_counter_changes"], reference_pairs,
                    samples_by_trial, minimum_counter_change,
                )
            median_us = statistics.median(trial_medians)
            speedup = statistics.median(
                ref / candidate for ref, candidate in zip(ref_trials, trial_medians)
            )
            observed_savings_us = statistics.median(
                ref - candidate for ref, candidate in zip(ref_trials, trial_medians)
            )
            above_noise = variant["reference"] or observed_savings_us > noise_floor_us
            no_slow_trial = all(
                candidate <= ref * maximum_trial_slowdown
                for ref, candidate in zip(ref_trials, trial_medians)
            )
            stable = _relative_mad(all_samples) <= maximum_relative_mad
            resource = variant["compiled_profile"]
            resource_clean = all(
                resource[name] == 0
                for name in (
                    "spills", "stack_bytes", "spill_store_bytes", "spill_load_bytes"
                )
            )
            qualified = variant["reference"] or (
                speedup >= minimum_speedup and above_noise and no_slow_trial and stable
                and reference_stable and cache_matched and clocks_matched
                and counter_matched
                and (allow_resource_spills or resource_clean)
            )
            rejection_reasons = []
            if not variant["reference"]:
                if speedup < minimum_speedup:
                    rejection_reasons.append("speedup")
                if not above_noise:
                    rejection_reasons.append("noise_floor")
                if not no_slow_trial:
                    rejection_reasons.append("trial_regression")
                if not stable or not reference_stable:
                    rejection_reasons.append("unstable")
                if not cache_matched:
                    rejection_reasons.append("cache_state")
                if not clocks_matched:
                    rejection_reasons.append("clock_drift")
                if not counter_matched:
                    rejection_reasons.append("counter_hypothesis")
                if not allow_resource_spills and not resource_clean:
                    rejection_reasons.append("local_memory")
            alternatives.append(
                {
                    "variant": variant["name"],
                    "reference": variant["reference"],
                    "execution_mode": variant["compiled_profile"]["segment_mode"],
                    "median_us": median_us,
                    "trial_medians_us": trial_medians,
                    "trial_samples_us": [
                        trial["samples_us"] for trial in samples_by_trial
                    ],
                    "speedup_vs_reference": speedup,
                    "observed_savings_us": observed_savings_us,
                    "noise_floor_us": noise_floor_us,
                    "above_noise_floor": above_noise,
                    "relative_mad": _relative_mad(all_samples),
                    "reference_relative_mad": _relative_mad(ref_samples),
                    "cache_states": [trial["cache_state"] for trial in samples_by_trial],
                    "telemetry": [trial["telemetry"] for trial in samples_by_trial],
                    "resource_clean": resource_clean,
                    "cache_matched": cache_matched,
                    "clocks_matched": clocks_matched,
                    "counter_hypothesis_matched": counter_matched,
                    "counter_evidence": counter_evidence,
                    "qualified": qualified,
                    "rejection_reasons": rejection_reasons,
                    "hypothesis": variant.get("hypothesis"),
                    "lever": variant.get("lever"),
                    "expected_counter_changes": variant.get("expected_counter_changes"),
                    "predicted_savings_us": variant.get("predicted_savings_us"),
                    "prediction_fraction": (
                        observed_savings_us / variant["predicted_savings_us"]
                        if not variant["reference"] else None
                    ),
                    "binary_sha256": variant["binary_sha256"],
                    "compiled_profile": variant["compiled_profile"],
                    "grid_waves": (
                        variant["compiled_profile"]["launch_blocks"]
                        / (
                            variant["compiled_profile"]["sm_count"]
                            * variant["compiled_profile"]["blocks_per_sm"]
                        )
                    ),
                }
            )
        winner = min(
            (row for row in alternatives if row["qualified"]),
            key=lambda row: row["median_us"],
        )
        results.append(
            {
                "profile_key": profile["profile_key"],
                "profile": profile,
                "selected_variant": winner["variant"],
                "selected_execution_mode": winner["execution_mode"],
                "selected_binary_sha256": winner["binary_sha256"],
                "selected_compiled_profile": winner["compiled_profile"],
                "weighted_reference_us": next(
                    row["median_us"] for row in alternatives if row["reference"]
                ) * profile["occurrences"],
                "weighted_selected_us": winner["median_us"] * profile["occurrences"],
                "weighted_noise_floor_us": noise_floor_us * profile["occurrences"],
                "weighted_predicted_savings_us": (
                    winner["predicted_savings_us"] * profile["occurrences"]
                    if not winner["reference"] else 0.0
                ),
                "promotion_decision": (
                    "T2-kernel-qualified-candidate"
                    if not winner["reference"] else "keep-reference"
                ),
                "next_gate": "T3-packet-role-block" if not winner["reference"] else None,
                "control_anchor_medians_us": [
                    [_trial_median(before), _trial_median(after)]
                    for before, after in reference_pairs
                ],
                "control_noise_by_trial_us": noise_by_trial,
                "alternatives": alternatives,
            }
        )
    return results


def rung_rollup(results):
    groups = collections.defaultdict(list)
    for result in results:
        profile = result["profile"]
        key = (
            profile["phase"], profile["rung"], profile["request_topology"],
            profile["family"], profile.get("kv_length"),
        )
        groups[key].append(result)
    rows = []
    for key, group in sorted(groups.items()):
        reference = sum(row["weighted_reference_us"] for row in group)
        selected = sum(row["weighted_selected_us"] for row in group)
        rows.append({
            "phase": key[0], "rung": key[1], "request_topology": key[2],
            "family": key[3], "kv_length": key[4],
            "profile_count": len(group), "weighted_reference_us": reference,
            "weighted_selected_us": selected, "weighted_speedup": reference / selected,
            "weighted_savings_us": reference - selected,
            "weighted_noise_floor_us": sum(
                row["weighted_noise_floor_us"] for row in group
            ),
            "weighted_predicted_savings_us": sum(
                row["weighted_predicted_savings_us"] for row in group
            ),
            "qualified_candidate_profiles": sum(
                row["promotion_decision"] == "T2-kernel-qualified-candidate"
                for row in group
            ),
        })
    return sorted(rows, key=lambda row: row["weighted_savings_us"], reverse=True)


def run_tuner(spec, output):
    cwd, packet_sha256, entries = load_plan(spec)
    with tempfile.TemporaryDirectory(prefix="plow-gemma4-kernel-tuner-", dir="/tmp") as directory:
        raw_dir = Path(directory)
        verification = verify_all(spec, entries, cwd, raw_dir)
        timings = benchmark_all(spec, entries, cwd, raw_dir)
    winners = select_winners(spec, entries, timings)
    summary = {
        "schema_version": 3,
        "model": "google/gemma-4-12B-it",
        "arch": spec["arch"],
        "gpu": spec["gpu"],
        "toolchain": spec["toolchain"],
        "dtype": spec["dtype"],
        "packet_sha256": packet_sha256,
        "protocol": {
            "verify_seeds": list(VERIFY_SEEDS),
            "warmups": WARMUPS,
            "iterations": ITERATIONS,
            "isolated_trials": TRIALS,
            "control_anchors_per_trial": 2,
        },
        "spec_sha256": hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest(),
        "profiles": winners,
        "rung_rollup": rung_rollup(winners),
        "verification": verification,
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
    _, packet_sha256, entries = load_plan(spec)
    families = collections.Counter(
        f"{entry['profile']['phase']}_{entry['profile']['family']}" for entry in entries
    )
    return {
        "model": "google/gemma-4-12B-it",
        "gpu": spec["gpu"],
        "toolchain": spec["toolchain"],
        "packet_sha256": packet_sha256,
        "protocol": {
            "verify_seeds": list(VERIFY_SEEDS),
            "warmups": WARMUPS,
            "iterations": ITERATIONS,
            "isolated_trials": TRIALS,
            "control_anchors_per_trial": 2,
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
                        "hypothesis": variant.get("hypothesis"),
                        "lever": variant.get("lever"),
                        "expected_counter_changes": variant.get("expected_counter_changes"),
                        "predicted_savings_us": variant.get("predicted_savings_us"),
                        "predicted_weighted_savings_us": (
                            variant.get("predicted_savings_us", 0)
                            * entry["profile"]["occurrences"]
                        ),
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
