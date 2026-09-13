#!/usr/bin/env python3
"""Run the Gemma prefix-hit suffix co-pack production gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def token_stream(seed, length):
    return [100 + ((seed * 7919 + index * 104729) % 4096) for index in range(length)]


def prompt_rows():
    warm_a = token_stream(1, 1025)
    warm_b = token_stream(2, 1025)
    return (warm_a, warm_b, warm_a + token_stream(3, 511), warm_b + token_stream(4, 511))


def bench_command(plowrt, assets, row_file, checkpoint=None, fp8_dir=None):
    command = [plowrt]
    if checkpoint:
        command.extend(("--rt-checkpoint", checkpoint))
    if fp8_dir:
        command.extend(("--fp8-dir", fp8_dir))
    command.extend([
        "bench", "--assets", str(assets), "--prompt-rows", str(row_file),
        "--concurrency", "2", "--warmup-requests", "2", "--requests", "2",
        "--output-len", "8", "--max-hold-ms", "8", "--slo-ms", "60000",
        "--engine-diagnostics", "--token-audit",
    ])
    return command


def validate(report, log, rows):
    if report.get("schema") != "plowrt.bench.v1" or report.get("vendor") != "Some(Amd)":
        raise ValueError("not an AMD production bench report")
    if report.get("num_gpus") != 1:
        raise ValueError("prefix co-pack gate requires one GPU")
    if (report.get("concurrency"), report.get("warmup_requests"), report.get("requests")) != (2, 2, 2):
        raise ValueError("unexpected request layout")
    if (report.get("completed"), report.get("failed")) != (2, 0):
        raise ValueError("measured requests did not complete")
    scheduler = report.get("scheduler") or {}
    if scheduler.get("rejected") != 0 or scheduler.get("admit_shed") != 0:
        raise ValueError("scheduler rejected or shed work")
    if (report.get("engine") or {}).get("batch_capacity") != 128:
        raise ValueError("production token-batch capacity is not 128")
    audit = report.get("token_audit") or {}
    if audit.get("prompt_token_ids") != list(rows[2:]):
        raise ValueError("token audit does not contain the exact hit prompts")
    outputs = audit.get("output_token_ids") or []
    if len(outputs) != 2 or any(len(output) != 8 for output in outputs):
        raise ValueError("token audit does not contain two eight-token outputs")
    copacks = [
        line for line in log.splitlines()
        if "AMD token batch" in line
        and all(re.search(rf"\b{name}=\"?{value}\"?\b", line) for name, value in (
            ("rows", 1024), ("decode", 0), ("prefill", 2), ("completed", 2),
        ))
        and re.search(r"\bfires=\"?true\"?\b", line)
    ]
    if len(copacks) != 1:
        raise ValueError(f"expected one 1024-row two-suffix co-pack, found {len(copacks)}")
    restores = [line for line in log.splitlines() if "PFX" in line and "prefix restore" in line]
    if len(restores) != 1 or not re.search(r"\bcalls=2\b", restores[0]):
        raise ValueError("prefix observer did not report exactly two restores")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plowrt", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--fp8-dir")
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    assets = Path(args.assets).resolve()
    packet = assets / "model.pkt"
    packet_sha256 = hashlib.sha256(packet.read_bytes()).hexdigest()
    if packet_sha256 != args.packet_sha256:
        raise SystemExit("packet hash differs from the campaign audit")
    rows = prompt_rows()
    with tempfile.TemporaryDirectory(prefix="plow-gemma4-prefix-copack-", dir="/tmp") as directory:
        row_file = Path(directory) / "rows.csv"
        row_file.write_text("".join(",".join(map(str, row)) + "\n" for row in rows))
        env = os.environ.copy()
        env.update({
            "RUST_LOG": "plowrt::serve::mux=debug,plowrt::obs::pfx=info,plowrt=info",
            "PLOW_PFX_LOG": "1",
            "PLOW_PREFIX_CACHE": "1",
            "PLOW_TOKEN_BATCH": "1",
            "PLOW_PF_BATCH": "1",
            "PLOW_PF_NO_INTERLEAVE": "0",
            "PLOW_PF_DEFER_DECODE": "0",
            "PLOW_AMD_SHARED_PREFIX": "0",
            "PLOW_VMM_KV": "0",
            "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
        })
        result = subprocess.run(
            bench_command(
                args.plowrt, assets, row_file, args.checkpoint, args.fp8_dir
            ),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise SystemExit(result.stderr.strip().splitlines()[-1])
        report = json.loads(result.stdout)
        validate(report, result.stderr, rows)

    record = {
        "schema": "plowrt.production-gate.v1",
        "packet_sha256": packet_sha256,
        "features": {
            "prefix_cache": True,
            "continuous_batching": True,
            "unified_token_batch": True,
            "suffix_copack": True,
        },
        "correct": True,
        "completed": 2,
        "cached_prefix_rows_per_request": 1024,
        "suffix_rows_per_request": 512,
        "copacked_rows": 1024,
        "decode_rows": 0,
        "prefill_requests": 2,
        "restore_calls": 2,
    }
    Path(args.output).write_text(json.dumps(record, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
