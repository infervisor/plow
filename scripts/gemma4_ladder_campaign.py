#!/usr/bin/env python3
"""Drive and gate a shape-complete Gemma-4-12B H100 tuning campaign.

The JSON spec names a packet audit, target axes, paired commands for the
``gemm`` and ``attention`` families, and control/candidate/vLLM serving
commands. Kernel commands print one JSON object with ``profile_key``,
``correct``, ``samples_us`` and, for candidates, ``compiled_profile``.
Serving commands write bench_packed_serve.py JSONL to ``{output}``;
``{contexts}`` and ``{concurrencies}`` expand to individual arguments.
Full-logit commands run once per phase and write one JSON record per rung to
``{output}``. Records must identify the packet and reference artifacts, assert
isolated execution, and report bitwise equality for full-vocabulary snapshots.
Raw command output stays in a temporary directory under ``/tmp``.

Attention profiles expand the packet rung over live runtime KV lengths, including
boundary-adjacent global histories through 16K and local-window/ring histories.
Packed homogeneous and ragged request tables are distinct profiles. Compiled
profiles identify the cubin, symbol, launch geometry, tile, resource
budgets, pipeline, TMA/swizzle choices, spills, and segment mode. Evidence for
one cubin therefore cannot be silently applied to another execution profile.
"""

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile


RUNGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DECODE_RUNGS = (1, 2, 4, 8, 16, 32)
CONTEXTS = RUNGS + (16384,)
GLOBAL_KV_LENGTHS = tuple(sorted({
    boundary + offset
    for boundary in CONTEXTS
    for offset in (-1, 0, 1)
    if 1 <= boundary + offset <= 16384
}))
LOCAL_KV_LENGTHS = tuple(sorted({
    boundary + offset
    for boundary in (1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    for offset in (-1, 0, 1)
    if 1 <= boundary + offset <= 16384
}))
GEMM_COUNTS = {
    (512, 3840): 8,
    (2048, 3840): 80,
    (4096, 3840): 40,
    (15360, 3840): 96,
    (8192, 3840): 8,
    (3840, 4096): 40,
    (3840, 8192): 8,
    (3840, 15360): 48,
    (262144, 3840): 1,
}
ATTENTION_COUNTS = {
    (16, 8, 256, 1024, 1): 40,
    (16, 1, 512, 0, 1): 8,
}
DECODE_GEMM_COUNTS = {
    ("Gemv", 512, 3840, 0, 0): 8,
    ("Gemv", 8192, 3840, 0, 0): 8,
    ("Gemv", 3840, 4096, 0, 0): 40,
    ("Gemv", 3840, 8192, 0, 0): 8,
    ("Gemv", 3840, 15360, 0, 0): 48,
    ("Gemv", 262144, 3840, 0, 0): 1,
    ("GemvGlu", 15360, 3840, 0, 0): 48,
    ("GemvQkv", 4096, 3840, 2048, 2048): 40,
}
DECODE_ATTENTION_COUNTS = {
    ("bf16_kv", 16, 8, 16384, 1024, 17, 256, 16383): 40,
    ("bf16_kv", 16, 1, 20480, 0, 17, 512, 0xffffffff): 8,
}
SHA256 = re.compile(r"[0-9a-f]{64}")
GEMMA4_VOCAB = 262144


class CampaignError(ValueError):
    pass


def read_json(path):
    with open(path) as source:
        return json.load(source)


def canonical_key(profile):
    common = (
        f"{profile['arch']}/{profile['dtype']}/{profile['phase']}/{profile['family']}/"
        f"r{profile['rung']}/c{profile['concurrency']}/{profile['request_topology']}/"
        f"{profile['packed_topology']}"
    )
    if profile["family"] == "gemm":
        return (
            f"{common}/{profile['dispatch_arm']}/"
            f"m{profile['m']}n{profile['n']}k{profile['k']}"
            f"x{profile.get('fused_n0', 0)}x{profile.get('fused_n1', 0)}"
        )
    return (
        f"{common}/{profile['kv_dtype']}/q{profile['query_rows']}kv{profile['kv_length']}"
        f"h{profile['q_heads']}x{profile['kv_heads']}d{profile['head_dim']}"
        f"w{profile['window']}s{profile['splits']}/hist-{profile['history_layout']}"
    )


def full_logit_key(phase, rung):
    return f"full-logit/{phase}/r{rung}"


def full_logit_plan():
    return [
        {
            "profile_key": full_logit_key(phase, rung),
            "phase": phase,
            "rung": rung,
            "status": "pending",
        }
        for phase, rungs in (("prefill", RUNGS), ("decode", DECODE_RUNGS))
        for rung in rungs
    ]


def _kernel_counts(program):
    gemm = collections.Counter()
    attention = collections.Counter()
    for case in program.get("kernel_cases", []):
        op = case.get("op", "")
        dims = case.get("i")
        pcs = case.get("pcs")
        if not isinstance(dims, list) or len(dims) < 8 or not isinstance(pcs, list):
            raise CampaignError(f"rung {program.get('rows')}: malformed kernel case")
        count = len(pcs)
        if op.startswith("Gemm"):
            gemm[(op, dims[0], dims[1], dims[2])] += count
        elif op.startswith("FlashPrefill"):
            kv_dtype = "fp8_kv" if op.endswith("Fp8") else "bf16_kv"
            fj = case.get("fj_bits")
            if not isinstance(fj, list) or len(fj) < 3:
                raise CampaignError(f"rung {program.get('rows')}: attention case has no KV layout")
            attention[(kv_dtype, dims[0], dims[1], dims[2], dims[3], dims[6],
                       dims[5], dims[7], fj[1], fj[2])] += count
    return gemm, attention


def _decode_kernel_counts(program):
    gemm = collections.Counter()
    attention = collections.Counter()
    for case in program.get("kernel_cases", []):
        op = case.get("op", "")
        dims = case.get("i")
        pcs = case.get("pcs")
        if not isinstance(dims, list) or len(dims) < 8 or not isinstance(pcs, list):
            raise CampaignError(f"decode rung {program.get('rows')}: malformed kernel case")
        count = len(pcs)
        if op.startswith("Gemv"):
            gemm[(op, dims[1], dims[2], dims[3], dims[4])] += count
        elif op.startswith("FlashDecode"):
            kv_dtype = "fp8_kv" if op.endswith("Fp8") else "bf16_kv"
            attention[(kv_dtype, dims[1], dims[2], dims[3], dims[4], dims[5],
                       dims[6], dims[7])] += count
    return gemm, attention


def _prefill_topologies(rung, topology, max_concurrency):
    active = min(rung, max_concurrency)
    rows_lo, extra = divmod(rung, active)
    result = [
        {"request_topology": "single", "packed_topology": "single",
         "concurrency": 1, "active_requests": 1,
         "rows_per_request_min": rung, "rows_per_request_max": rung,
         "history_layout": "homogeneous"},
        {"request_topology": "packed_homogeneous", "packed_topology": topology,
         "concurrency": active, "target_concurrency": max_concurrency,
         "active_requests": active, "rows_per_request_min": rows_lo,
         "rows_per_request_max": rows_lo + bool(extra),
         "history_layout": "homogeneous"},
    ]
    if active > 1:
        result.append({
            **result[-1], "request_topology": "packed_ragged",
            "history_layout": "ragged",
        })
    return tuple(result)


def _decode_topologies(rung, topology, max_concurrency):
    result = [{
        "request_topology": "decode_homogeneous", "packed_topology": topology,
        "concurrency": rung, "target_concurrency": max_concurrency,
        "active_requests": rung, "rows_per_request_min": 1,
        "rows_per_request_max": 1, "history_layout": "homogeneous",
    }]
    if rung > 1:
        result.append({
            **result[0], "request_topology": "decode_ragged",
            "history_layout": "ragged",
        })
    return tuple(result)


def _kv_profiles(minimum, window, history_layout):
    lengths = LOCAL_KV_LENGTHS if window else GLOBAL_KV_LENGTHS
    eligible = [value for value in lengths if value >= minimum]
    for index, value in enumerate(eligible):
        if history_layout == "ragged" and index == 0:
            continue
        previous = eligible[max(0, index - 1)]
        yield {
            "kv_length": value,
            "kv_length_min": previous if history_layout == "ragged" else value,
            "effective_kv_rows": min(value, window) if window else value,
            "q_pos0": value - minimum,
            "q_pos0_min": previous - minimum if history_layout == "ragged" else value - minimum,
        }


def audit_inventory(audit, arch, dtype, topology, max_concurrency):
    prefill = [p for p in audit.get("programs", []) if p.get("phase") == "prefill"]
    programs = {p.get("rows"): p for p in prefill}
    if len(prefill) != len(programs):
        raise CampaignError("packet audit contains duplicate prefill rungs")
    if tuple(sorted(programs)) != RUNGS:
        raise CampaignError(
            f"prefill rungs must be {list(RUNGS)}, got {sorted(programs)}"
        )
    decode = [p for p in audit.get("programs", []) if p.get("phase") == "decode"]
    decode_programs = {p.get("rows"): p for p in decode}
    if len(decode) != len(decode_programs):
        raise CampaignError("packet audit contains duplicate decode rungs")
    if tuple(sorted(decode_programs)) != DECODE_RUNGS:
        raise CampaignError(
            f"decode rungs must be {list(DECODE_RUNGS)}, got {sorted(decode_programs)}"
        )
    profiles = []
    rung_summary = []
    for rung in RUNGS:
        gemm, attention = _kernel_counts(programs[rung])
        gemm_shapes = collections.Counter()
        for (_, m, n, k), count in gemm.items():
            gemm_shapes[(m, n, k)] += count
        attention_shapes = collections.Counter()
        for (_, *dims), count in attention.items():
            attention_shapes[tuple(dims[:-2])] += count
        expected_gemm = {
            (1 if nk == (262144, 3840) else rung, *nk): count
            for nk, count in GEMM_COUNTS.items()
        }
        expected_attention = {
            (rung, rung, qh, kvh, hd, window, splits): count
            for (qh, kvh, hd, window, splits), count in ATTENTION_COUNTS.items()
        }
        if gemm_shapes != expected_gemm:
            raise CampaignError(f"rung {rung}: GEMM shapes/counts differ from Gemma-4-12B")
        if attention_shapes != expected_attention:
            raise CampaignError(
                f"rung {rung}: attention shapes/counts differ from Gemma-4-12B"
            )
        for (_, _, _, _, _, _, window, _, stride, mask) in attention:
            valid_layout = (
                stride >= max(1024, window)
                and (mask == stride - 1 if window else mask == 0xffffffff)
            )
            if not valid_layout:
                raise CampaignError(f"rung {rung}: invalid Gemma KV ring layout")
        before = len(profiles)
        for request in _prefill_topologies(rung, topology, max_concurrency):
            for (dispatch_arm, m, n, k), occurrences in sorted(gemm.items()):
                profile = {
                    "arch": arch,
                    "dtype": dtype,
                    "phase": "prefill",
                    "family": "gemm",
                    "dispatch_arm": dispatch_arm,
                    "rung": rung,
                    "m": m,
                    "n": n,
                    "k": k,
                    "fused_n0": 0,
                    "fused_n1": 0,
                    "occurrences": occurrences,
                    **request,
                }
                profile["profile_key"] = canonical_key(profile)
                profiles.append(profile)
            for dims, occurrences in sorted(attention.items()):
                (kv_dtype, packet_q, packet_kv, q_heads, kv_heads, head_dim,
                 window, splits, kv_stride, kv_mask) = dims
                query_rows = request["rows_per_request_max"]
                for history in _kv_profiles(query_rows, window, request["history_layout"]):
                    profile = {
                        "arch": arch, "dtype": dtype, "kv_dtype": kv_dtype,
                        "phase": "prefill", "family": "attention", "rung": rung,
                        "m": rung, "packet_query_rows": packet_q,
                        "packet_kv_rows": packet_kv, "query_rows": query_rows,
                        "q_heads": q_heads, "kv_heads": kv_heads,
                        "gqa": q_heads // kv_heads, "head_dim": head_dim,
                        "window": window, "splits": splits,
                        "kv_stride": kv_stride, "kv_mask": kv_mask,
                        "occurrences": occurrences, **request, **history,
                    }
                    profile["profile_key"] = canonical_key(profile)
                    profiles.append(profile)
        rung_summary.append(
            {
                "rung": rung,
                "gemm_shapes": len(gemm_shapes),
                "gemm_calls": sum(gemm_shapes.values()),
                "attention_shapes": len(attention_shapes),
                "attention_calls": sum(attention_shapes.values()),
                "execution_profiles": len(profiles) - before,
            }
        )
    decode_summary = []
    for rung in DECODE_RUNGS:
        gemm, attention = _decode_kernel_counts(decode_programs[rung])
        if gemm != DECODE_GEMM_COUNTS:
            raise CampaignError(f"decode rung {rung}: linear shapes/counts differ from Gemma-4-12B")
        expected_attention = DECODE_ATTENTION_COUNTS
        if attention != expected_attention:
            raise CampaignError(f"decode rung {rung}: attention shapes/counts differ from Gemma-4-12B")
        before = len(profiles)
        for request in _decode_topologies(rung, topology, max_concurrency):
            for (dispatch_arm, n, k, fused_n0, fused_n1), occurrences in sorted(gemm.items()):
                profile = {
                    "arch": arch, "dtype": dtype, "phase": "decode",
                    "family": "gemm", "dispatch_arm": dispatch_arm,
                    "rung": rung, "m": rung, "n": n, "k": k,
                    "fused_n0": fused_n0, "fused_n1": fused_n1,
                    "occurrences": occurrences, **request,
                }
                profile["profile_key"] = canonical_key(profile)
                profiles.append(profile)
            for dims, occurrences in sorted(attention.items()):
                (kv_dtype, q_heads, kv_heads, kv_stride, window,
                 split_cap, head_dim, kv_mask) = dims
                for history in _kv_profiles(1, window, request["history_layout"]):
                    profile = {
                        "arch": arch, "dtype": dtype, "kv_dtype": kv_dtype,
                        "phase": "decode", "family": "attention", "rung": rung,
                        "m": rung, "packet_query_rows": rung, "query_rows": 1,
                        "q_heads": q_heads, "kv_heads": kv_heads,
                        "gqa": q_heads // kv_heads, "head_dim": head_dim,
                        "window": window, "splits": split_cap,
                        "kv_stride": kv_stride, "kv_mask": kv_mask,
                        "occurrences": occurrences, **request, **history,
                    }
                    profile["profile_key"] = canonical_key(profile)
                    profiles.append(profile)
        decode_summary.append({
            "rung": rung, "gemm_shapes": len(gemm),
            "gemm_calls": sum(gemm.values()), "attention_shapes": len(attention),
            "attention_calls": sum(attention.values()),
            "execution_profiles": len(profiles) - before,
        })
    return profiles, rung_summary, decode_summary


class Fields(dict):
    def __missing__(self, key):
        raise CampaignError(f"command template references unknown field {key!r}")


def expand_command(template, fields):
    if not isinstance(template, list) or not template or not all(isinstance(x, str) for x in template):
        raise CampaignError("commands must be nonempty string arrays")
    command = []
    for part in template:
        if part.startswith("{") and part.endswith("}"):
            value = fields.get(part[1:-1])
            if isinstance(value, (list, tuple)):
                command.extend(map(str, value))
                continue
        command.append(part.format_map(Fields(fields)))
    return command


def parse_json_lines(text, label):
    records = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise CampaignError(f"{label}: stdout line {number} is not JSON") from error
    if not records:
        raise CampaignError(f"{label}: command produced no JSON")
    return records


def invoke(command, cwd, env, timeout, label):
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
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1:] or result.stdout.strip().splitlines()[-1:]
        raise CampaignError(f"{label}: exit {result.returncode}: {' '.join(detail)}")
    return result.stdout


def validate_kernel_record(record, profile, arm, minimum_samples):
    if record.get("profile_key") != profile["profile_key"]:
        raise CampaignError(f"{arm}: result profile key does not match scheduled profile")
    samples = record.get("samples_us")
    if not isinstance(samples, list) or len(samples) < minimum_samples or any(
        type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples
    ):
        raise CampaignError(f"{profile['profile_key']}: invalid timing samples")
    if record.get("correct") is not True:
        raise CampaignError(f"{profile['profile_key']}: {arm} failed correctness")
    normalized = {
        "profile_key": profile["profile_key"],
        "arm": arm,
        "median_us": statistics.median(samples),
        "samples": len(samples),
    }
    if arm == "candidate":
        compiled = record.get("compiled_profile") or {}
        required = (
            "object_sha256", "kernel_symbol", "threads", "warps", "registers",
            "smem_bytes", "tile", "stages", "tma", "swizzle", "spills",
            "segment_mode",
        )
        if any(name not in compiled for name in required):
            raise CampaignError(f"{profile['profile_key']}: incomplete compiled profile")
        if not SHA256.fullmatch(str(compiled["object_sha256"])):
            raise CampaignError(f"{profile['profile_key']}: invalid object hash")
        for name in ("threads", "warps", "registers", "stages"):
            if type(compiled[name]) is not int or compiled[name] <= 0:
                raise CampaignError(f"{profile['profile_key']}: invalid compiled {name}")
        if type(compiled["smem_bytes"]) is not int or compiled["smem_bytes"] < 0:
            raise CampaignError(f"{profile['profile_key']}: invalid compiled smem_bytes")
        if not isinstance(compiled["tile"], list) or len(compiled["tile"]) != 3 or any(
            type(value) is not int or value <= 0 for value in compiled["tile"]
        ):
            raise CampaignError(f"{profile['profile_key']}: invalid compiled tile")
        if type(compiled["tma"]) is not bool or not isinstance(compiled["swizzle"], str):
            raise CampaignError(f"{profile['profile_key']}: invalid memory profile")
        if type(compiled.get("spills")) is not int or compiled["spills"] < 0:
            raise CampaignError(f"{profile['profile_key']}: missing compiled spill count")
        if compiled["segment_mode"] not in ("direct", "persistent"):
            raise CampaignError(f"{profile['profile_key']}: invalid segment mode")
        normalized["compiled_profile"] = compiled
    return normalized


def run_kernels(spec, profiles, cwd, env):
    commands = spec.get("kernel_commands") or {}
    minimum_samples = int((spec.get("gates") or {}).get("minimum_samples", 5))
    timeout = int(spec.get("timeout_s", 900))
    rows = []
    for profile in profiles:
        family = commands.get(f"{profile['phase']}_{profile['family']}")
        if family is None:
            family = commands.get(profile["family"])
        if not isinstance(family, dict):
            raise CampaignError(f"missing {profile['family']} kernel commands")
        measurements = {}
        for arm in ("control", "candidate"):
            fields = {**profile, "arm": arm}
            command = expand_command(family.get(arm), fields)
            stdout = invoke(command, cwd, env, timeout, f"{arm} {profile['profile_key']}")
            record = parse_json_lines(stdout, profile["profile_key"])[-1]
            measurements[arm] = validate_kernel_record(record, profile, arm, minimum_samples)
        control = measurements["control"]["median_us"]
        candidate = measurements["candidate"]["median_us"]
        rows.append(
            {
                **profile,
                "control_us": control,
                "candidate_us": candidate,
                "speedup": control / candidate,
                "weighted_control_us": control * profile["occurrences"],
                "weighted_candidate_us": candidate * profile["occurrences"],
                "compiled_profile": measurements["candidate"]["compiled_profile"],
            }
        )
    return rows


def _metric(record, name):
    value = record.get(name)
    if isinstance(value, dict):
        value = value.get("p50")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise CampaignError(f"invalid serving metric {name}")
    return value


def reduce_serving(records, arm, max_concurrency, contexts):
    groups = collections.defaultdict(list)
    for record in records:
        key = (record.get("input"), record.get("concurrency"))
        groups[key].append(record)
    expected = {(context, concurrency) for context in contexts for concurrency in (1, max_concurrency)}
    if set(groups) != expected:
        raise CampaignError(f"{arm}: serving cells differ: missing={sorted(expected-set(groups))}, extra={sorted(set(groups)-expected)}")
    cells = []
    for (context, concurrency), group in sorted(groups.items()):
        workloads = {(x.get("output"), x.get("requested_cached_prefix_tokens", 0)) for x in group}
        if len(workloads) != 1:
            raise CampaignError(f"{arm}: mixed output lengths or cache targets in one serving cell")
        output, cached_prefix = workloads.pop()
        if type(output) is not int or output <= 1:
            raise CampaignError(f"{arm}: serving output length must exceed one for TPOT")
        if type(cached_prefix) is not int or not 0 <= cached_prefix < context:
            raise CampaignError(f"{arm}: invalid cached prefix target")
        cells.append(
            {
                "arm": arm,
                "context": context,
                "concurrency": concurrency,
                "repeats": len(group),
                "output": output,
                "requested_cached_prefix_tokens": cached_prefix,
                "output_tok_s": statistics.median(_metric(x, "output_tok_s") for x in group),
                "ttft_ms": statistics.median(_metric(x, "ttft_ms") for x in group),
                "latency_ms": statistics.median(_metric(x, "latency_ms") for x in group),
                "tpot_ms": statistics.median(_metric(x, "tpot_ms") for x in group),
            }
        )
    return cells


def run_serving(spec, cwd, env, temporary):
    commands = spec.get("serving_commands") or {}
    contexts = tuple(spec.get("serving_contexts", CONTEXTS))
    if not contexts or any(type(x) is not int or x < 1 or x > 16384 for x in contexts):
        raise CampaignError("serving contexts must be positive integers through 16384")
    if 16384 not in contexts:
        raise CampaignError("serving campaign must include 16384 context")
    maximum = spec["max_concurrency"]
    timeout = int(spec.get("serving_timeout_s", 7200))
    cells = []
    for arm in ("control", "candidate", "vllm"):
        output = temporary / f"{arm}.jsonl"
        fields = {
            "arm": arm,
            "output": str(output),
            "contexts": contexts,
            "concurrencies": (1, maximum),
            "max_concurrency": maximum,
        }
        command = expand_command(commands.get(arm), fields)
        stdout = invoke(command, cwd, env, timeout, f"serving {arm}")
        if output.exists():
            records = parse_json_lines(output.read_text(), f"serving {arm}")
            output.unlink()
        else:
            records = parse_json_lines(stdout, f"serving {arm}")
        cells.extend(reduce_serving(records, arm, maximum, contexts))
    return cells


def validate_full_logit_record(record, cell, packet_sha256):
    for name in ("profile_key", "phase", "rung"):
        if record.get(name) != cell[name]:
            raise CampaignError(
                f"{cell['profile_key']}: full-logit result has wrong {name}"
            )
    if record.get("packet_sha256") != packet_sha256:
        raise CampaignError(f"{cell['profile_key']}: full-logit packet hash differs")
    if not SHA256.fullmatch(str(record.get("reference_sha256", ""))):
        raise CampaignError(f"{cell['profile_key']}: invalid full-logit reference hash")
    if record.get("isolated") is not True:
        raise CampaignError(f"{cell['profile_key']}: full-logit run was not isolated")
    expected_snapshots = 1 if cell["phase"] == "prefill" else cell["rung"]
    if record.get("snapshots") != expected_snapshots:
        raise CampaignError(
            f"{cell['profile_key']}: expected {expected_snapshots} full-logit "
            f"snapshots, got {record.get('snapshots')}"
        )
    if record.get("vocab") != GEMMA4_VOCAB:
        raise CampaignError(f"{cell['profile_key']}: full vocabulary was not checked")
    passed = (
        record.get("correct") is True
        and record.get("bitwise_equal") is True
        and record.get("all_finite") is True
    )
    return {
        **cell,
        "status": "pass" if passed else "fail",
        "packet_sha256": packet_sha256,
        "reference_sha256": record["reference_sha256"],
        "snapshots": record["snapshots"],
        "vocab": record["vocab"],
        "bitwise_equal": record.get("bitwise_equal") is True,
        "all_finite": record.get("all_finite") is True,
    }


def run_full_logits(spec, cells, packet_sha256, cwd, env, temporary):
    commands = spec.get("full_logit_commands") or {}
    timeout = int(spec.get("full_logit_timeout_s", 7200))
    results = []
    for phase, phase_cells in _group(cells, lambda cell: cell["phase"]).items():
        template = commands.get(phase)
        if template is None:
            raise CampaignError(f"missing {phase} full-logit command")
        output = temporary / f"full-logit-{phase}.jsonl"
        fields = {
            "phase": phase,
            "rungs": [cell["rung"] for cell in phase_cells],
            "profile_keys": [cell["profile_key"] for cell in phase_cells],
            "packet_sha256": packet_sha256,
            "output": str(output),
        }
        stdout = invoke(
            expand_command(template, fields), cwd, env, timeout, f"full-logit/{phase}"
        )
        records = parse_json_lines(
            output.read_text() if output.exists() else stdout,
            f"full-logit/{phase}",
        )
        expected = {cell["profile_key"]: cell for cell in phase_cells}
        actual = [record.get("profile_key") for record in records]
        if len(actual) != len(set(actual)):
            raise CampaignError(f"full-logit/{phase}: duplicate result profile key")
        if set(actual) != set(expected):
            raise CampaignError(
                f"full-logit/{phase}: cells differ: "
                f"missing={sorted(set(expected) - set(actual))}, "
                f"extra={sorted(set(actual) - set(expected))}"
            )
        results.extend(
            validate_full_logit_record(record, expected[record["profile_key"]], packet_sha256)
            for record in records
        )
    failures = [x["profile_key"] for x in results if x["status"] != "pass"]
    return {"pass": not failures, "failures": failures, "cells": results}


def gate_kernels(rows, gates):
    regression = float(gates.get("kernel_regression_tolerance", 1.02))
    minimum = float(gates.get("minimum_weighted_speedup", 1.01))
    rung_rows = []
    failures = []
    group_key = lambda x: (
        x["phase"], x["rung"], x["request_topology"], x["family"], x.get("kv_length")
    )
    for key, group in sorted(_group(rows, group_key).items()):
        control = sum(x["weighted_control_us"] for x in group)
        candidate = sum(x["weighted_candidate_us"] for x in group)
        speedup = control / candidate
        rung_rows.append({
            "phase": key[0], "rung": key[1], "request_topology": key[2],
            "family": key[3], "kv_length": key[4], "control_us": control,
            "candidate_us": candidate, "speedup": speedup,
        })
        if speedup < minimum:
            failures.append(
                f"{key[0]} rung {key[1]} {key[2]} {key[3]} KV={key[4]} "
                f"weighted speedup {speedup:.4f} < {minimum:.4f}"
            )
    for row in rows:
        if row["candidate_us"] > row["control_us"] * regression:
            failures.append(f"{row['profile_key']} regressed by {row['candidate_us']/row['control_us']:.4f}x")
    ranked = sorted(rows, key=lambda x: x["weighted_control_us"], reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["cost_rank"] = rank
        row["weighted_savings_us"] = row["weighted_control_us"] - row["weighted_candidate_us"]
    return {"pass": not failures, "failures": failures, "rungs": rung_rows}, ranked


def _group(rows, key):
    result = collections.defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return result


def gate_serving(cells, max_concurrency, gates, baseline):
    if baseline == "vllm":
        tolerance = float(gates.get("vllm_latency_tolerance", 1.0))
        throughput_min = float(gates.get("vllm_minimum_throughput_speedup", 1.0))
    else:
        tolerance = float(gates.get("serving_regression_tolerance", 1.02))
        throughput_min = float(gates.get("minimum_throughput_speedup", 1.0))
    indexed = {(x["arm"], x["context"], x["concurrency"]): x for x in cells}
    failures = []
    comparisons = []
    for context in sorted({x["context"] for x in cells}):
        for concurrency in (1, max_concurrency):
            candidate = indexed[("candidate", context, concurrency)]
            reference = indexed[(baseline, context, concurrency)]
            for axis in ("output", "requested_cached_prefix_tokens", "repeats"):
                if candidate.get(axis) != reference.get(axis):
                    raise CampaignError(f"{context} C{concurrency}: {axis} differs from {baseline}")
            row = {"context": context, "concurrency": concurrency, "baseline": baseline}
            for metric in ("output_tok_s", "ttft_ms", "latency_ms", "tpot_ms"):
                row[metric + "_ratio"] = candidate[metric] / reference[metric]
            comparisons.append(row)
            for metric in ("ttft_ms", "latency_ms", "tpot_ms"):
                if candidate[metric] > reference[metric] * tolerance:
                    failures.append(f"{context} C{concurrency} {metric} is {candidate[metric]/reference[metric]:.4f}x {baseline}")
            if concurrency == max_concurrency and candidate["output_tok_s"] < reference["output_tok_s"] * throughput_min:
                failures.append(f"{context} C{concurrency} throughput is {candidate['output_tok_s']/reference['output_tok_s']:.4f}x {baseline}")
    return {"pass": not failures, "baseline": baseline, "failures": failures, "comparisons": comparisons}


def external_output(path):
    root = Path(__file__).resolve().parents[1]
    output = Path(path).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        return output
    raise CampaignError("campaign summaries must be written outside the repository")


def campaign_plan(spec):
    audit = read_json(spec["audit"])
    if not SHA256.fullmatch(str(audit.get("packet_sha256", ""))):
        raise CampaignError("packet audit has no valid packet SHA256")
    profiles, rungs, decode_rungs = audit_inventory(
        audit, spec["arch"], spec["dtype"], spec["packed_topology"], spec["max_concurrency"]
    )
    return {
        "schema_version": 1,
        "model": "google/gemma-4-12B-it",
        "packet_sha256": audit.get("packet_sha256"),
        "rungs": rungs,
        "decode_rungs": decode_rungs,
        "full_logits": full_logit_plan(),
        "kv_lengths": {
            "global": list(GLOBAL_KV_LENGTHS),
            "local": list(LOCAL_KV_LENGTHS),
        },
        "profile_count": len(profiles),
        "profiles": profiles,
    }


def run_campaign(spec, output):
    plan = campaign_plan(spec)
    cwd = str(Path(spec.get("cwd", ".")).resolve())
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})
    with tempfile.TemporaryDirectory(prefix="plow-gemma4-campaign-", dir="/tmp") as directory:
        kernel_rows = run_kernels(spec, plan["profiles"], cwd, env)
        temporary = Path(directory)
        full_logits = run_full_logits(
            spec, plan["full_logits"], plan["packet_sha256"], cwd, env, temporary
        )
        serving_cells = run_serving(spec, cwd, env, temporary)
    gates = spec.get("gates") or {}
    kernel_gate, ranked = gate_kernels(kernel_rows, gates)
    control_gate = gate_serving(serving_cells, spec["max_concurrency"], gates, "control")
    vllm_gate = gate_serving(serving_cells, spec["max_concurrency"], gates, "vllm")
    summary = {
        **{k: v for k, v in plan.items() if k != "profiles"},
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "kernel_profiles": ranked,
        "full_logits": full_logits,
        "serving_cells": serving_cells,
        "promotion": {
            "pass": kernel_gate["pass"] and full_logits["pass"] and control_gate["pass"],
            "kernel": kernel_gate,
            "full_logits": full_logits,
            "serving_vs_control": control_gate,
        },
        "vllm_goal": vllm_gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def validate_spec(spec):
    required = ("audit", "arch", "dtype", "packed_topology", "max_concurrency")
    if any(name not in spec for name in required):
        raise CampaignError(f"spec requires {', '.join(required)}")
    if type(spec["max_concurrency"]) is not int or spec["max_concurrency"] <= 1:
        raise CampaignError("max_concurrency must be an integer greater than one")
    if spec["arch"] != "sm90a":
        raise CampaignError("Gemma-4-12B H100 campaign requires arch=sm90a")
    if spec["dtype"] not in ("bf16", "fp8", "w8a8", "w8a16"):
        raise CampaignError("dtype must be bf16, fp8, w8a8, or w8a16")
    if not isinstance(spec["packed_topology"], str) or not spec["packed_topology"]:
        raise CampaignError("packed_topology must be a nonempty string")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan_parser = sub.add_parser("plan", help="validate the packet audit and print exact profiles")
    plan_parser.add_argument("--spec", required=True)
    run_parser = sub.add_parser("run", help="run paired kernels, serving arms and gates sequentially")
    run_parser.add_argument("--spec", required=True)
    run_parser.add_argument("--summary", required=True)
    run_parser.add_argument("--require", choices=("none", "promotion", "goal"), default="promotion")
    args = parser.parse_args(argv)
    try:
        spec = read_json(args.spec)
        validate_spec(spec)
        if args.action == "plan":
            json.dump(campaign_plan(spec), sys.stdout, indent=2, sort_keys=True)
            print()
            return 0
        summary = run_campaign(spec, external_output(args.summary))
        print(
            f"profiles={len(summary['kernel_profiles'])} "
            f"promotion={'PASS' if summary['promotion']['pass'] else 'FAIL'} "
            f"vllm_goal={'PASS' if summary['vllm_goal']['pass'] else 'FAIL'}"
        )
        if args.require == "promotion" and not summary["promotion"]["pass"]:
            return 2
        if args.require == "goal" and not (
            summary["promotion"]["pass"] and summary["vllm_goal"]["pass"]
        ):
            return 2
        return 0
    except (CampaignError, KeyError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
