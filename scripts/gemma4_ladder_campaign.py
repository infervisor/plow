#!/usr/bin/env python3
"""Drive and gate a rung-complete Gemma-4-12B accelerator campaign.

The JSON spec names a packet audit, target axes, anchored commands for the
``gemm`` and ``attention`` families, and control/candidate/vLLM serving
commands. Kernel commands print one JSON object with ``profile_key``,
``correct``, ``samples_us`` and, for candidates, ``compiled_profile``. Every
kernel candidate is bracketed by control anchors and must clear their drift
plus two median absolute deviations. Serving promotion interleaves two
control/candidate pairs and requires the per-request TTFT and TPOT bootstrap
95% confidence-interval upper bound to remain below zero in every selected
cell.
Serving commands write bench_packed_serve.py JSONL to ``{output}``;
``{contexts}`` and ``{concurrencies}`` expand to individual arguments. Runs are
paired one concurrency rung at a time so a long C128/16K matrix remains
attributable and can be resumed by selecting only its missing rungs.
Full-logit commands run once per phase and write one JSON record per rung to
``{output}``. Records must identify the packet and reference artifacts, assert
isolated execution, and report bitwise equality or same-session control-floor
bounded error for full-vocabulary snapshots.
Raw command output stays in a temporary directory under ``/tmp``.

Attention profiles expand the packet rung over explicit live-KV buckets and
boundary-adjacent global histories through 16K plus local-window/ring histories.
Packed homogeneous and ragged request tables are distinct profiles. Compiled
profiles identify the GPU object, symbol, launch geometry, tile, resource
budgets, BQ/BKV, split count, occupancy, pipeline, memory path, spills, and
segment mode. Evidence for one object therefore cannot be silently applied to
another execution profile.

The spec may select ``prefill_rungs``, ``decode_rungs``, ``context_rungs``,
``concurrency_rungs`` and ``kernel_families``. CLI rung flags override those
arrays for focused/resume runs. A full run defaults to contexts
128/1K/4K/8K/16K and concurrency 1/2/4/8/16/32/64/128. The production gate
command can use scripts/gemma4_prefix_copack_gate.py and must prove the exact
prefix-hit suffix co-pack route before promotion.
"""

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile


RUNGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DECODE_RUNGS = (1, 2, 4, 8, 16, 32)
CONTEXTS = RUNGS + (16384,)
DEFAULT_CONTEXT_RUNGS = (128, 1024, 4096, 8192, 16384)
DEFAULT_CONCURRENCY_RUNGS = (1, 2, 4, 8, 16, 32, 64, 128)
GLOBAL_KV_LENGTHS = tuple(sorted({
    boundary + offset for boundary in CONTEXTS for offset in (-1, 0, 1)
    if 1 <= boundary + offset <= 16384
}))
LOCAL_KV_LENGTHS = tuple(sorted({
    boundary + offset
    for boundary in (1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    for offset in (-1, 0, 1) if 1 <= boundary + offset <= 16384
}))
SERVING_HIGHER_IS_BETTER = ("request_s", "output_tok_s")
SERVING_LOWER_IS_BETTER = tuple(
    f"{family}_{stat}_ms"
    for family in ("ttft", "tpot", "itl", "e2el")
    for stat in ("mean", "median", "p99")
)
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
        f"{common}/{profile['kv_dtype']}/bucket{profile['live_kv_bucket']}"
        f"/q{profile['query_rows']}kv{profile['kv_length']}"
        f"h{profile['q_heads']}x{profile['kv_heads']}d{profile['head_dim']}"
        f"w{profile['window']}s{profile['splits']}/hist-{profile['history_layout']}"
    )


def full_logit_key(phase, rung):
    return f"full-logit/{phase}/r{rung}"


def full_logit_plan(prefill_rungs=RUNGS, decode_rungs=DECODE_RUNGS):
    return [
        {
            "profile_key": full_logit_key(phase, rung),
            "phase": phase,
            "rung": rung,
            "status": "pending",
        }
        for phase, rungs in (("prefill", prefill_rungs), ("decode", decode_rungs))
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
        elif op.startswith("Gemv"):
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


def _validate_decode_gemma_shapes(rung, gemm, attention):
    linear = collections.Counter()
    qkv = collections.Counter()
    glu = collections.Counter()
    for (op, n, k, _fused_n0, _fused_n1), count in gemm.items():
        if "Qkv" in op:
            qkv[(n, k)] += count
        elif "Glu" in op:
            glu[(n, k)] += count
        else:
            linear[(n, k)] += count
    expected_base = collections.Counter({
        (512, 3840): 8,
        (8192, 3840): 8,
        (3840, 4096): 40,
        (3840, 8192): 8,
        (3840, 15360): 48,
        (262144, 3840): 1,
    })
    if qkv:
        if qkv != collections.Counter({(4096, 3840): 40}):
            raise CampaignError(f"decode rung {rung}: invalid fused QKV inventory")
    else:
        expected_base.update({(4096, 3840): 40, (2048, 3840): 80})
    if glu:
        if glu != collections.Counter({(15360, 3840): 48}):
            raise CampaignError(f"decode rung {rung}: invalid fused GLU inventory")
    else:
        expected_base[(15360, 3840)] += 96
    if linear != expected_base:
        raise CampaignError(f"decode rung {rung}: linear shapes/counts differ from Gemma-4-12B")

    logical_attention = collections.Counter()
    for (kv_dtype, q_heads, kv_heads, kv_stride, window, splits, head_dim, kv_mask), count in attention.items():
        valid_layout = (
            kv_stride >= max(1024, window)
            and (kv_mask == kv_stride - 1 if window else kv_mask == 0xffffffff)
            and splits > 0
        )
        if not valid_layout:
            raise CampaignError(f"decode rung {rung}: invalid Gemma KV ring layout")
        logical_attention[(kv_dtype, q_heads, kv_heads, window, head_dim)] += count
    expected_attention = collections.Counter({
        ("bf16_kv", 16, 8, 1024, 256): 40,
        ("bf16_kv", 16, 1, 0, 512): 8,
    })
    if logical_attention != expected_attention:
        raise CampaignError(f"decode rung {rung}: attention shapes/counts differ from Gemma-4-12B")


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


def _boundary_lengths(boundaries, maximum):
    return tuple(sorted({
        boundary + offset
        for boundary in boundaries
        for offset in (-1, 0, 1)
        if 1 <= boundary + offset <= maximum
    }))


def _kv_profiles(
    minimum, window, history_layout, global_lengths, local_lengths, live_kv_buckets
):
    lengths = local_lengths if window else global_lengths
    eligible = [value for value in lengths if value >= minimum]
    for index, value in enumerate(eligible):
        if history_layout == "ragged" and index == 0:
            continue
        previous = eligible[max(0, index - 1)]
        yield {
            "kv_length": value,
            "live_kv_bucket": next(
                (bucket for bucket in live_kv_buckets if value <= bucket),
                live_kv_buckets[-1],
            ),
            "kv_length_min": previous if history_layout == "ragged" else value,
            "effective_kv_rows": min(value, window) if window else value,
            "q_pos0": value - minimum,
            "q_pos0_min": previous - minimum if history_layout == "ragged" else value - minimum,
        }


def audit_inventory(
    audit,
    arch,
    dtype,
    topology,
    max_concurrency,
    prefill_rungs=None,
    decode_rungs=None,
    context_rungs=None,
):
    use_legacy_kv_grid = context_rungs is None
    prefill = [p for p in audit.get("programs", []) if p.get("phase") == "prefill"]
    programs = {p.get("rows"): p for p in prefill}
    if len(prefill) != len(programs):
        raise CampaignError("packet audit contains duplicate prefill rungs")
    prefill_rungs = tuple(sorted(programs)) if prefill_rungs is None else tuple(prefill_rungs)
    missing = set(prefill_rungs) - set(programs)
    if missing:
        raise CampaignError(f"packet audit is missing prefill rungs {sorted(missing)}")
    decode = [p for p in audit.get("programs", []) if p.get("phase") == "decode"]
    decode_programs = {p.get("rows"): p for p in decode}
    if len(decode) != len(decode_programs):
        raise CampaignError("packet audit contains duplicate decode rungs")
    decode_rungs = tuple(sorted(decode_programs)) if decode_rungs is None else tuple(decode_rungs)
    missing = set(decode_rungs) - set(decode_programs)
    if missing:
        raise CampaignError(f"packet audit is missing decode rungs {sorted(missing)}")
    context_rungs = tuple(context_rungs or CONTEXTS)
    maximum_context = max(context_rungs)
    if use_legacy_kv_grid:
        global_lengths = GLOBAL_KV_LENGTHS
        local_lengths = LOCAL_KV_LENGTHS
    else:
        global_lengths = _boundary_lengths(context_rungs, maximum_context)
        local_boundaries = tuple(sorted(
            set(context_rungs) | {1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
        ))
        local_lengths = _boundary_lengths(local_boundaries, maximum_context)
    profiles = []
    rung_summary = []
    for rung in prefill_rungs:
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
                for history in _kv_profiles(
                    query_rows, window, request["history_layout"], global_lengths,
                    local_lengths, context_rungs,
                ):
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
    for rung in decode_rungs:
        gemm, attention = _decode_kernel_counts(decode_programs[rung])
        _validate_decode_gemma_shapes(rung, gemm, attention)
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
                for history in _kv_profiles(
                    1, window, request["history_layout"], global_lengths,
                    local_lengths, context_rungs,
                ):
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
        "mad_us": statistics.median(
            abs(value - statistics.median(samples)) for value in samples
        ),
        "samples": len(samples),
    }
    if arm == "candidate":
        compiled = record.get("compiled_profile") or {}
        required = (
            "object_sha256", "kernel_symbol", "threads", "tile", "stages",
            "spills", "segment_mode",
        )
        if profile["arch"] == "gfx942":
            required += (
                "wavefronts", "vgprs", "agprs", "lds_bytes", "memory_path", "mfma",
            )
        else:
            required += ("warps", "registers", "smem_bytes", "tma", "swizzle")
        if profile["family"] == "attention":
            required += ("program_digest", "bq", "bkv", "nsplit", "occupancy")
        if any(name not in compiled for name in required):
            raise CampaignError(f"{profile['profile_key']}: incomplete compiled profile")
        if not SHA256.fullmatch(str(compiled["object_sha256"])):
            raise CampaignError(f"{profile['profile_key']}: invalid object hash")
        if profile["family"] == "attention" and not SHA256.fullmatch(
            str(compiled["program_digest"])
        ):
            raise CampaignError(f"{profile['profile_key']}: invalid program digest")
        positive = ["threads", "stages"]
        positive += ["wavefronts", "vgprs"] if profile["arch"] == "gfx942" else ["warps", "registers"]
        for name in positive:
            if type(compiled[name]) is not int or compiled[name] <= 0:
                raise CampaignError(f"{profile['profile_key']}: invalid compiled {name}")
        if profile["family"] == "attention":
            for name in ("bq", "bkv", "nsplit"):
                if type(compiled[name]) is not int or compiled[name] <= 0:
                    raise CampaignError(f"{profile['profile_key']}: invalid compiled {name}")
            occupancy = compiled["occupancy"]
            if (
                type(occupancy) not in (int, float)
                or not math.isfinite(occupancy)
                or occupancy <= 0
            ):
                raise CampaignError(f"{profile['profile_key']}: invalid compiled occupancy")
        nonnegative = ["spills"]
        nonnegative += ["agprs", "lds_bytes"] if profile["arch"] == "gfx942" else ["smem_bytes"]
        for name in nonnegative:
            if type(compiled[name]) is not int or compiled[name] < 0:
                raise CampaignError(f"{profile['profile_key']}: invalid compiled {name}")
        if not isinstance(compiled["tile"], list) or len(compiled["tile"]) != 3 or any(
            type(value) is not int or value <= 0 for value in compiled["tile"]
        ):
            raise CampaignError(f"{profile['profile_key']}: invalid compiled tile")
        if profile["arch"] == "gfx942":
            if not isinstance(compiled["memory_path"], str) or not compiled["memory_path"]:
                raise CampaignError(f"{profile['profile_key']}: invalid memory path")
            if not isinstance(compiled["mfma"], str) or not compiled["mfma"]:
                raise CampaignError(f"{profile['profile_key']}: invalid MFMA profile")
        elif type(compiled["tma"]) is not bool or not isinstance(compiled["swizzle"], str):
            raise CampaignError(f"{profile['profile_key']}: invalid memory profile")
        if compiled["segment_mode"] not in ("direct", "persistent", "split", "segmented"):
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
        for arm in ("control_before", "candidate", "control_after"):
            fields = {**profile, "arm": arm}
            template = family.get("candidate" if arm == "candidate" else "control")
            command = expand_command(template, fields)
            stdout = invoke(command, cwd, env, timeout, f"{arm} {profile['profile_key']}")
            record = parse_json_lines(stdout, profile["profile_key"])[-1]
            measurements[arm] = validate_kernel_record(record, profile, arm, minimum_samples)
        before = measurements["control_before"]
        after = measurements["control_after"]
        control = (before["median_us"] + after["median_us"]) / 2
        candidate = measurements["candidate"]["median_us"]
        noise = abs(before["median_us"] - after["median_us"]) + 2 * max(
            before["mad_us"], after["mad_us"]
        )
        rows.append(
            {
                **profile,
                "control_before_us": before["median_us"],
                "control_after_us": after["median_us"],
                "control_us": control,
                "candidate_us": candidate,
                "control_noise_floor_us": noise,
                "above_control_noise": control - candidate > noise,
                "speedup": control / candidate,
                "weighted_control_us": control * profile["occurrences"],
                "weighted_candidate_us": candidate * profile["occurrences"],
                "weighted_noise_floor_us": noise * profile["occurrences"],
                "compiled_profile": measurements["candidate"]["compiled_profile"],
            }
        )
    return rows


def _metric(record, name):
    value = record.get(name)
    if value is None and name == "request_s":
        output = record.get("output")
        output_tok_s = record.get("output_tok_s")
        if type(output) is int and output > 0 and type(output_tok_s) in (int, float):
            value = output_tok_s / output
    if isinstance(value, dict):
        value = value.get("p50")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise CampaignError(f"invalid serving metric {name}")
    return value


def _summary_metric(record, family, stat):
    metrics = record.get("metrics") or {}
    summary = metrics.get(f"{family}_ms") or {}
    key = "p50" if stat == "median" else stat
    value = summary.get(key)
    if value is None:
        value = record.get(f"{family}_{stat}_ms")
    if value is None:
        legacy = {"ttft": "ttft_ms", "tpot": "tpot_ms", "e2el": "latency_ms"}.get(family)
        if legacy:
            value = record.get(legacy)
            if isinstance(value, dict):
                value = value.get(key if key in value else "p50")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise CampaignError(f"invalid serving metric {family}_{stat}_ms")
    return value


def _rung_list(value):
    if type(value) is int:
        return (1, value)
    return tuple(value)


def _prefill_execution(context, prefill_rungs):
    eligible = [rung for rung in prefill_rungs if rung <= context]
    bucket = max(eligible) if eligible else min(prefill_rungs)
    return bucket, math.ceil(context / bucket)


def _decode_execution(concurrency, decode_rungs):
    eligible = [rung for rung in decode_rungs if rung >= concurrency]
    if not eligible:
        raise CampaignError(
            f"decode ladder ends at {max(decode_rungs)}, below concurrency {concurrency}"
        )
    return min(eligible)


def _serving_identity(group, arm, concurrency, required):
    if not required and not all(isinstance(record.get("requests"), list) for record in group):
        return {}
    prompt_rows = []
    output_rows = []
    repeats = set()
    for record in sorted(group, key=lambda row: row.get("repeat", -1)):
        repeat = record.get("repeat")
        requests = record.get("requests")
        if type(repeat) is not int or repeat in repeats:
            raise CampaignError(f"{arm}: serving repeats must be unique integers")
        if not isinstance(requests, list) or len(requests) != concurrency:
            raise CampaignError(f"{arm}: serving cell has incomplete request identities")
        prompts = [request.get("prompt_sha256") for request in requests]
        outputs = [request.get("text") for request in requests]
        if any(not SHA256.fullmatch(str(value)) for value in prompts):
            raise CampaignError(f"{arm}: serving cell has invalid prompt identity")
        if any(not isinstance(value, str) for value in outputs):
            raise CampaignError(f"{arm}: serving cell has invalid completion identity")
        repeats.add(repeat)
        prompt_rows.append((repeat, prompts))
        output_rows.append((repeat, [hashlib.sha256(value.encode()).hexdigest() for value in outputs]))
    encode = lambda value: hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        "prompt_set_sha256": encode(prompt_rows),
        "completion_set_sha256": encode(output_rows),
    }


def _request_metrics(group, arm, concurrency, required):
    if not required and not all(isinstance(record.get("requests"), list) for record in group):
        return {}
    result = {"ttft_ms": [], "tpot_ms": [], "e2el_ms": []}
    for record in sorted(group, key=lambda row: row.get("repeat", -1)):
        requests = record.get("requests")
        if not isinstance(requests, list) or len(requests) != concurrency:
            raise CampaignError(f"{arm}: serving cell has incomplete request timings")
        for request in requests:
            values = {
                "ttft_ms": request.get("ttft_ms"),
                "tpot_ms": request.get("tpot_ms"),
                "e2el_ms": request.get("latency_ms"),
            }
            for name, value in values.items():
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise CampaignError(f"{arm}: invalid per-request {name}")
                result[name].append(float(value))
    return {"request_metrics": result}


def reduce_serving(
    records,
    arm,
    concurrency_rungs,
    contexts,
    prefill_rungs=RUNGS,
    decode_rungs=None,
    require_identity=False,
):
    concurrency_rungs = _rung_list(concurrency_rungs)
    decode_rungs = concurrency_rungs if decode_rungs is None else tuple(decode_rungs)
    groups = collections.defaultdict(list)
    for record in records:
        key = (record.get("input"), record.get("concurrency"))
        groups[key].append(record)
    expected = {(context, concurrency) for context in contexts for concurrency in concurrency_rungs}
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
        bucket, chunks = _prefill_execution(context, prefill_rungs)
        decode_bucket = _decode_execution(concurrency, decode_rungs)
        cell = {
            "arm": arm,
            "context": context,
            "concurrency": concurrency,
            "rung_key": f"ctx{context}/c{concurrency}",
            "prefill_bucket": bucket,
            "prefill_chunks": chunks,
            "decode_bucket": decode_bucket,
            "repeats": len(group),
            "output": output,
            "requested_cached_prefix_tokens": cached_prefix,
            **_serving_identity(group, arm, concurrency, require_identity),
            **_request_metrics(group, arm, concurrency, require_identity),
        }
        for metric in SERVING_HIGHER_IS_BETTER:
            cell[metric] = statistics.median(_metric(x, metric) for x in group)
        for family in ("ttft", "tpot", "itl", "e2el"):
            for stat in ("mean", "median", "p99"):
                name = f"{family}_{stat}_ms"
                cell[name] = statistics.median(
                    _summary_metric(x, family, stat) for x in group
                )
        # Compatibility names for existing summary consumers.
        cell["ttft_ms"] = cell["ttft_median_ms"]
        cell["tpot_ms"] = cell["tpot_median_ms"]
        cell["latency_ms"] = cell["e2el_median_ms"]
        cells.append(cell)
    return cells


def run_serving(
    spec,
    cwd,
    env,
    temporary,
    context_rungs,
    concurrency_rungs,
    prefill_rungs,
    decode_rungs,
):
    commands = spec.get("serving_commands") or {}
    maximum = spec["max_concurrency"]
    timeout = int(spec.get("serving_timeout_s", 7200))
    cells = []
    arms = tuple(spec.get(
        "serving_arms", ("control", "candidate", "control2", "candidate2", "vllm")
    ))
    if "candidate" not in arms:
        raise CampaignError("serving_arms must include candidate")
    for concurrency in concurrency_rungs:
        for arm in arms:
            output = temporary / f"{arm}-c{concurrency}.jsonl"
            fields = {
                "arm": arm,
                "output": str(output),
                "contexts": context_rungs,
                "context_rungs": context_rungs,
                "concurrencies": (concurrency,),
                "concurrency_rungs": (concurrency,),
                "concurrency": concurrency,
                "max_concurrency": maximum,
            }
            template = commands.get(arm)
            if template is None and arm in ("control2", "candidate2"):
                template = commands.get(arm[:-1])
            command = expand_command(template, fields)
            stdout = invoke(command, cwd, env, timeout, f"serving {arm} C{concurrency}")
            if output.exists():
                records = parse_json_lines(output.read_text(), f"serving {arm} C{concurrency}")
                output.unlink()
            else:
                records = parse_json_lines(stdout, f"serving {arm} C{concurrency}")
            cells.extend(reduce_serving(
                records,
                arm,
                (concurrency,),
                context_rungs,
                prefill_rungs,
                decode_rungs,
                True,
            ))
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
    bitwise = record.get("bitwise_equal") is True
    floor = {}
    if not bitwise:
        for metric in ("max_abs", "rel_l2"):
            error = record.get(f"{metric}_error")
            control = record.get(f"control_{metric}_floor")
            if (
                type(error) not in (int, float) or not math.isfinite(error) or error < 0
                or type(control) not in (int, float) or not math.isfinite(control) or control < 0
            ):
                raise CampaignError(
                    f"{cell['profile_key']}: invalid full-logit {metric} error/floor"
                )
            floor[f"{metric}_error"] = float(error)
            floor[f"control_{metric}_floor"] = float(control)
    floor_bounded = bitwise or all(
        floor[f"{metric}_error"] <= floor[f"control_{metric}_floor"]
        for metric in ("max_abs", "rel_l2")
    )
    passed = (
        record.get("correct") is True
        and floor_bounded
        and record.get("all_finite") is True
    )
    return {
        **cell,
        "status": "pass" if passed else "fail",
        "packet_sha256": packet_sha256,
        "reference_sha256": record["reference_sha256"],
        "snapshots": record["snapshots"],
        "vocab": record["vocab"],
        "bitwise_equal": bitwise,
        "floor_bounded": floor_bounded,
        "all_finite": record.get("all_finite") is True,
        **floor,
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


def validate_production_gate_record(record, plan):
    if record.get("schema") != "plowrt.production-gate.v1":
        raise CampaignError("production gate has the wrong schema")
    if record.get("packet_sha256") != plan["packet_sha256"]:
        raise CampaignError("production gate packet hash differs")
    if record.get("features") != plan["production_features"]:
        raise CampaignError("production gate feature set differs from the campaign")
    expected = {
        "completed": 2,
        "cached_prefix_rows_per_request": 1024,
        "suffix_rows_per_request": 512,
        "copacked_rows": 1024,
        "decode_rows": 0,
        "prefill_requests": 2,
        "restore_calls": 2,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise CampaignError(
                f"production gate expected {name}={value}, got {record.get(name)!r}"
            )
    if record.get("correct") is not True:
        raise CampaignError("production scheduler gate failed correctness")
    return {**record, "pass": True}


def run_production_gate(spec, plan, cwd, env, temporary):
    template = spec.get("production_gate_command")
    if template is None:
        raise CampaignError("missing production_gate_command")
    output = temporary / "production-gate.jsonl"
    fields = {
        "output": str(output),
        "packet_sha256": plan["packet_sha256"],
        "prefill_rungs": plan["axes"]["prefill"],
        "decode_rungs": plan["axes"]["decode"],
        "max_concurrency": max(plan["axes"]["concurrency"]),
    }
    stdout = invoke(
        expand_command(template, fields),
        cwd,
        env,
        int(spec.get("production_gate_timeout_s", 7200)),
        "production scheduler gate",
    )
    records = parse_json_lines(
        output.read_text() if output.exists() else stdout,
        "production scheduler gate",
    )
    if len(records) != 1:
        raise CampaignError("production scheduler gate must emit exactly one record")
    return validate_production_gate_record(records[0], plan)


def gate_kernels(rows, gates):
    regression = float(gates.get("kernel_regression_tolerance", 1.02))
    minimum = float(gates.get("minimum_weighted_speedup", 1.01))
    rung_rows = []
    failures = []
    group_key = lambda x: (
        x["phase"], x["rung"], x["request_topology"], x["family"]
    )
    for key, group in sorted(_group(rows, group_key).items()):
        control = sum(x["weighted_control_us"] for x in group)
        candidate = sum(x["weighted_candidate_us"] for x in group)
        noise = sum(x.get("weighted_noise_floor_us", 0.0) for x in group)
        savings = control - candidate
        speedup = control / candidate
        rung_rows.append({
            "phase": key[0], "rung": key[1], "request_topology": key[2],
            "family": key[3], "control_us": control,
            "candidate_us": candidate, "speedup": speedup,
            "savings_us": savings, "noise_floor_us": noise,
            "above_noise_floor": savings > noise,
        })
        if speedup < minimum:
            failures.append(
                f"{key[0]} rung {key[1]} {key[2]} {key[3]} "
                f"weighted speedup {speedup:.4f} < {minimum:.4f}"
            )
        if savings <= noise:
            failures.append(
                f"{key[0]} rung {key[1]} {key[2]} {key[3]} "
                f"savings {savings:.3f} us <= noise floor {noise:.3f} us"
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


def gate_serving(cells, concurrency_rungs, gates, baseline, candidate_arm="candidate"):
    concurrency_rungs = _rung_list(concurrency_rungs)
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
        for concurrency in concurrency_rungs:
            try:
                candidate = indexed[(candidate_arm, context, concurrency)]
                reference = indexed[(baseline, context, concurrency)]
            except KeyError as error:
                raise CampaignError(
                    f"{context} C{concurrency}: missing candidate or {baseline} serving cell"
                ) from error
            for axis in (
                "output", "requested_cached_prefix_tokens", "repeats",
                "prefill_bucket", "prefill_chunks", "decode_bucket",
                "prompt_set_sha256", "completion_set_sha256",
            ):
                if candidate.get(axis) != reference.get(axis):
                    raise CampaignError(f"{context} C{concurrency}: {axis} differs from {baseline}")
            row = {"context": context, "concurrency": concurrency, "baseline": baseline}
            for metric in SERVING_HIGHER_IS_BETTER + SERVING_LOWER_IS_BETTER:
                if metric not in candidate or metric not in reference:
                    raise CampaignError(f"{context} C{concurrency}: missing serving metric {metric}")
                row[metric + "_ratio"] = candidate[metric] / reference[metric]
            comparisons.append(row)
            for metric in SERVING_LOWER_IS_BETTER:
                regressed = candidate[metric] > reference[metric] * tolerance
                if baseline == "vllm":
                    regressed = candidate[metric] >= reference[metric] * tolerance
                if regressed:
                    failures.append(f"{context} C{concurrency} {metric} is {candidate[metric]/reference[metric]:.4f}x {baseline}")
            for metric in SERVING_HIGHER_IS_BETTER:
                regressed = candidate[metric] < reference[metric] * throughput_min
                if baseline == "vllm":
                    regressed = candidate[metric] <= reference[metric] * throughput_min
                if regressed:
                    failures.append(
                        f"{context} C{concurrency} {metric} is "
                        f"{candidate[metric]/reference[metric]:.4f}x {baseline}"
                    )
    return {"pass": not failures, "baseline": baseline, "failures": failures, "comparisons": comparisons}


def _bootstrap_upper(differences, samples, seed):
    if not differences:
        raise CampaignError("bootstrap requires paired request differences")
    rng = random.Random(seed)
    count = len(differences)
    means = []
    for _ in range(samples):
        means.append(sum(differences[rng.randrange(count)] for _ in range(count)) / count)
    means.sort()
    return means[min(len(means) - 1, math.ceil(0.975 * len(means)) - 1)]


def gate_t4_serving(cells, concurrency_rungs, gates):
    indexed = {(x["arm"], x["context"], x["concurrency"]): x for x in cells}
    bootstrap_samples = int(gates.get("bootstrap_samples", 2000))
    minimum_c1 = int(gates.get("minimum_t4_prompts_c1", 32))
    minimum_c16 = int(gates.get("minimum_t4_prompts_c16", 160))
    if bootstrap_samples < 200 or minimum_c1 < 1 or minimum_c16 < 1:
        raise CampaignError("invalid T4 serving gates")
    failures = []
    comparisons = []
    for context in sorted({x["context"] for x in cells}):
        for concurrency in _rung_list(concurrency_rungs):
            try:
                arms = {
                    arm: indexed[(arm, context, concurrency)]
                    for arm in ("control", "candidate", "control2", "candidate2")
                }
            except KeyError as error:
                raise CampaignError(
                    f"{context} C{concurrency}: T4 requires control/candidate/control2/candidate2"
                ) from error
            for arm in ("candidate", "control2", "candidate2"):
                for axis in (
                    "output", "requested_cached_prefix_tokens", "repeats",
                    "prefill_bucket", "prefill_chunks", "decode_bucket", "prompt_set_sha256",
                    "completion_set_sha256",
                ):
                    if arms[arm].get(axis) != arms["control"].get(axis):
                        raise CampaignError(
                            f"{context} C{concurrency}: T4 {axis} differs in {arm}"
                        )
            prompt_count = sum(
                len(arms[name].get("request_metrics", {}).get("ttft_ms", []))
                for name in ("candidate", "candidate2")
            )
            minimum = minimum_c1 if concurrency < 16 else minimum_c16
            if prompt_count < minimum:
                failures.append(
                    f"{context} C{concurrency}: T4 has {prompt_count} candidate prompts < {minimum}"
                )
            row = {
                "context": context, "concurrency": concurrency,
                "candidate_prompts": prompt_count, "minimum_prompts": minimum,
            }
            for metric in ("ttft_ms", "tpot_ms"):
                differences = []
                for candidate_arm, control_arm in (
                    ("candidate", "control"), ("candidate2", "control2")
                ):
                    candidate = arms[candidate_arm].get("request_metrics", {}).get(metric, [])
                    control = arms[control_arm].get("request_metrics", {}).get(metric, [])
                    if len(candidate) != len(control) or not candidate:
                        raise CampaignError(
                            f"{context} C{concurrency}: unpaired T4 {metric} samples"
                        )
                    differences.extend(a - b for a, b in zip(candidate, control))
                seed = int.from_bytes(
                    hashlib.sha256(f"{context}/{concurrency}/{metric}".encode()).digest()[:8],
                    "little",
                )
                upper = _bootstrap_upper(differences, bootstrap_samples, seed)
                mean = statistics.mean(differences)
                row[f"{metric}_mean_delta"] = mean
                row[f"{metric}_ci95_upper"] = upper
                if upper >= 0:
                    failures.append(
                        f"{context} C{concurrency}: {metric} CI upper {upper:.4f} ms >= 0"
                    )
            comparisons.append(row)
    return {
        "pass": not failures,
        "bootstrap_samples": bootstrap_samples,
        "failures": failures,
        "comparisons": comparisons,
    }


def external_output(path):
    root = Path(__file__).resolve().parents[1]
    output = Path(path).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        return output
    raise CampaignError("campaign summaries must be written outside the repository")


def _validated_axis(spec, name, default, maximum=None):
    values = spec.get(name, default)
    if (
        not isinstance(values, list) and not isinstance(values, tuple)
    ) or not values or any(type(x) is not int or x <= 0 for x in values):
        raise CampaignError(f"{name} must be a nonempty integer array")
    values = tuple(values)
    if values != tuple(sorted(set(values))):
        raise CampaignError(f"{name} must be strictly increasing and unique")
    if maximum is not None and values[-1] > maximum:
        raise CampaignError(f"{name} exceeds {maximum}")
    return values


def campaign_axes(spec, audit):
    audit_prefill = sorted(
        p["rows"] for p in audit.get("programs", []) if p.get("phase") == "prefill"
    )
    audit_decode = sorted(
        p["rows"] for p in audit.get("programs", []) if p.get("phase") == "decode"
    )
    contexts = _validated_axis(
        spec,
        "context_rungs",
        spec.get("serving_contexts", DEFAULT_CONTEXT_RUNGS),
        16384,
    )
    concurrencies = _validated_axis(
        spec,
        "concurrency_rungs",
        tuple(x for x in DEFAULT_CONCURRENCY_RUNGS if x <= spec["max_concurrency"]),
        spec["max_concurrency"],
    )
    prefill = _validated_axis(spec, "prefill_rungs", audit_prefill)
    decode = _validated_axis(spec, "decode_rungs", audit_decode, spec["max_concurrency"])
    if spec.get("require_full_matrix", True):
        if contexts[-1] != 16384:
            raise CampaignError("full campaign must include the 16384 context rung")
        required_concurrency = tuple(
            x for x in DEFAULT_CONCURRENCY_RUNGS if x <= spec["max_concurrency"]
        )
        if concurrencies != required_concurrency:
            raise CampaignError(
                f"full campaign concurrency_rungs must be {list(required_concurrency)}"
            )
        if spec["max_concurrency"] != 128:
            raise CampaignError("full Gemma-4-12B campaign must run through concurrency 128")
    return {
        "prefill": prefill,
        "decode": decode,
        "context": contexts,
        "concurrency": concurrencies,
    }


def campaign_plan(spec):
    audit = read_json(spec["audit"])
    if not SHA256.fullmatch(str(audit.get("packet_sha256", ""))):
        raise CampaignError("packet audit has no valid packet SHA256")
    axes = campaign_axes(spec, audit)
    profiles, rungs, decode_rungs = audit_inventory(
        audit,
        spec["arch"],
        spec["dtype"],
        spec["packed_topology"],
        spec["max_concurrency"],
        axes["prefill"],
        axes["decode"],
        axes["context"],
    )
    kernel_families = tuple(spec.get("kernel_families", ("gemm", "attention")))
    profiles = [profile for profile in profiles if profile["family"] in kernel_families]
    global_lengths = _boundary_lengths(axes["context"], max(axes["context"]))
    local_boundaries = tuple(sorted(set(axes["context"]) | {1, 32, 64, 256, 512, 2048}))
    return {
        "schema_version": 4,
        "model": "google/gemma-4-12B-it",
        "packet_sha256": audit.get("packet_sha256"),
        "architecture": spec["arch"],
        "axes": {name: list(values) for name, values in axes.items()},
        "production_features": spec.get("production_features", {}),
        "kernel_families": list(kernel_families),
        "rungs": rungs,
        "decode_rungs": decode_rungs,
        "full_logits": full_logit_plan(axes["prefill"], axes["decode"]),
        "kv_lengths": {
            "global": list(global_lengths),
            "local": list(_boundary_lengths(local_boundaries, max(axes["context"]))),
        },
        "live_kv_buckets": list(axes["context"]),
        "serving_rungs": [
            {
                "rung_key": f"ctx{context}/c{concurrency}",
                "context": context,
                "concurrency": concurrency,
                "prefill_bucket": _prefill_execution(context, axes["prefill"])[0],
                "prefill_chunks": _prefill_execution(context, axes["prefill"])[1],
                "decode_bucket": _decode_execution(concurrency, axes["decode"]),
            }
            for context in axes["context"]
            for concurrency in axes["concurrency"]
        ],
        "profile_count": len(profiles),
        "profiles": profiles,
    }


def run_campaign(spec, output):
    plan = campaign_plan(spec)
    axes = plan["axes"]
    cwd = str(Path(spec.get("cwd", ".")).resolve())
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})
    with tempfile.TemporaryDirectory(prefix="plow-gemma4-campaign-", dir="/tmp") as directory:
        kernel_rows = run_kernels(spec, plan["profiles"], cwd, env)
        temporary = Path(directory)
        full_logits = run_full_logits(
            spec, plan["full_logits"], plan["packet_sha256"], cwd, env, temporary
        )
        production_gate = run_production_gate(spec, plan, cwd, env, temporary)
        serving_cells = run_serving(
            spec,
            cwd,
            env,
            temporary,
            tuple(axes["context"]),
            tuple(axes["concurrency"]),
            tuple(axes["prefill"]),
            tuple(axes["decode"]),
        )
    gates = spec.get("gates") or {}
    kernel_gate, ranked = gate_kernels(kernel_rows, gates)
    control_gate = gate_serving(serving_cells, axes["concurrency"], gates, "control")
    control2_gate = gate_serving(
        serving_cells, axes["concurrency"], gates, "control2", "candidate2"
    )
    t4_gate = gate_t4_serving(serving_cells, axes["concurrency"], gates)
    vllm_gate = gate_serving(serving_cells, axes["concurrency"], gates, "vllm")
    vllm2_gate = gate_serving(
        serving_cells, axes["concurrency"], gates, "vllm", "candidate2"
    )
    vllm_gate["pass"] = vllm_gate["pass"] and vllm2_gate["pass"]
    vllm_gate["replicate"] = vllm2_gate
    summary = {
        **{k: v for k, v in plan.items() if k != "profiles"},
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "kernel_profiles": ranked,
        "full_logits": full_logits,
        "production_gate": production_gate,
        "serving_cells": serving_cells,
        "promotion": {
            "pass": (
                kernel_gate["pass"] and full_logits["pass"]
                and production_gate["pass"] and control_gate["pass"]
                and control2_gate["pass"] and t4_gate["pass"]
            ),
            "kernel": kernel_gate,
            "full_logits": full_logits,
            "production_scheduler": production_gate,
            "serving_vs_control": control_gate,
            "serving_vs_control2": control2_gate,
            "t4_bootstrap": t4_gate,
        },
        "vllm_goal": vllm_gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def validate_spec(spec, production=False):
    required = ("audit", "arch", "dtype", "packed_topology", "max_concurrency")
    if any(name not in spec for name in required):
        raise CampaignError(f"spec requires {', '.join(required)}")
    if type(spec["max_concurrency"]) is not int or spec["max_concurrency"] <= 1:
        raise CampaignError("max_concurrency must be an integer greater than one")
    if spec["arch"] not in ("sm90a", "gfx942"):
        raise CampaignError("Gemma-4-12B campaign requires arch=sm90a or gfx942")
    if spec["dtype"] not in ("bf16", "fp8", "w8a8", "w8a16"):
        raise CampaignError("dtype must be bf16, fp8, w8a8, or w8a16")
    if not isinstance(spec["packed_topology"], str) or not spec["packed_topology"]:
        raise CampaignError("packed_topology must be a nonempty string")
    families = spec.get("kernel_families", ("gemm", "attention"))
    if not isinstance(families, (list, tuple)) or not families or any(
        family not in ("gemm", "attention") for family in families
    ) or len(set(families)) != len(families):
        raise CampaignError("kernel_families must select gemm and/or attention once")
    if production:
        features = spec.get("production_features")
        required_features = (
            "prefix_cache", "continuous_batching", "unified_token_batch", "suffix_copack"
        )
        if not isinstance(features, dict) or any(
            features.get(name) is not True for name in required_features
        ):
            raise CampaignError(
                "production_features must enable prefix_cache, continuous_batching, "
                "unified_token_batch, and suffix_copack"
            )


def _add_rung_arguments(parser):
    parser.add_argument("--prefill-rung", action="append", type=int)
    parser.add_argument("--decode-rung", action="append", type=int)
    parser.add_argument("--context-rung", action="append", type=int)
    parser.add_argument("--concurrency-rung", action="append", type=int)
    parser.add_argument("--kernel-family", action="append", choices=("gemm", "attention"))


def _apply_rung_arguments(spec, args):
    overrides = {
        "prefill_rungs": args.prefill_rung,
        "decode_rungs": args.decode_rung,
        "context_rungs": args.context_rung,
        "concurrency_rungs": args.concurrency_rung,
        "kernel_families": args.kernel_family,
    }
    if any(value is not None for value in overrides.values()):
        spec = dict(spec)
        spec.update({name: sorted(set(value)) for name, value in overrides.items() if value})
        spec["require_full_matrix"] = False
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan_parser = sub.add_parser("plan", help="validate the packet audit and print exact profiles")
    plan_parser.add_argument("--spec", required=True)
    _add_rung_arguments(plan_parser)
    run_parser = sub.add_parser("run", help="run paired kernels, serving arms and gates sequentially")
    run_parser.add_argument("--spec", required=True)
    run_parser.add_argument("--summary", required=True)
    run_parser.add_argument("--require", choices=("none", "promotion", "goal"), default="promotion")
    _add_rung_arguments(run_parser)
    args = parser.parse_args(argv)
    try:
        spec = _apply_rung_arguments(read_json(args.spec), args)
        validate_spec(spec, production=True)
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
