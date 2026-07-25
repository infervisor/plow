#!/usr/bin/env python3
"""Run the reproducible Gemma-4-26B standalone global-queue decode sweep.

The sweep is deliberately decode-only: ``PLOW_PREFILL=0``, 16 discarded
warmup steps, 120 measured steps, and three exact vLLM RandomDataset seed-0
prompts at each of the seven B1 contexts.  Generate the prompt files first:

  /workspace/venvs/vllm/bin/python \
    perf-data/harness/make_vllm_random_ids.py MODEL_DIR PROMPT_DIR

One invocation covers one explicitly labelled precision/packet pair and takes
one gpulease around all 21 processes.  Full-FP8 is intentionally not inferred:
the caller must provide both its actual label and its FP8 weight directory.

Example BF16 run:

  python3 perf-data/harness/run_gemma4_26b_decode_sweep.py \
    --precision-label bf16 --binary /tmp/gemma4_sm120_chat \
    --packet /tmp/gemma26-bf16.pkt \
    --model-dir /workspace/models/gemma-4-26B-A4B-it \
    --prompt-dir /tmp/gemma26-seed0 --out-dir /tmp/gemma26-bf16-sweep

Example for a real FP8 asset (name the path honestly, for example
``fp8-dense-only`` while experts/router remain BF16):

  python3 perf-data/harness/run_gemma4_26b_decode_sweep.py \
    --precision-label fp8-full --vllm-config fp8 \
    --fp8-dir /workspace/models/gemma-4-26B-A4B-it-fp8 \
    --binary /tmp/gemma4_sm120_chat_fp8 --packet /tmp/gemma26-fp8.pkt \
    --model-dir /workspace/models/gemma-4-26B-A4B-it \
    --prompt-dir /tmp/gemma26-seed0 --out-dir /tmp/gemma26-fp8-sweep

Use ``--self-test`` to exercise parsing and aggregation without a GPU, or
``--dry-run`` to validate a concrete campaign and print all commands.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VLLM_JSON = ROOT / "perf-data/gemma4-26b-a4b-vllm-sm120.json"
CONTEXTS = (1024, 4096, 16384, 32768, 65536, 98304, 131072)
PROMPTS_PER_CONTEXT = 3
WARMUP = 16
STEPS = 136
TIMED_STEPS = STEPS - WARMUP

RESULT_RE = re.compile(
    r"^PLOW_RESULT\s+ctx=(?P<context>\d+)\s+"
    r"mean_ms=(?P<mean>[0-9]+(?:\.[0-9]+)?)\s+"
    r"median_ms=(?P<median>[0-9]+(?:\.[0-9]+)?)\s+"
    r"sd_ms=(?P<sd>[0-9]+(?:\.[0-9]+)?)\s+"
    r"n=(?P<n>\d+)\s+"
    r"host_ms=(?P<host>[0-9]+(?:\.[0-9]+)?)\s+"
    r"kernel_ms=(?P<kernel>[0-9]+(?:\.[0-9]+)?)$",
    re.MULTILINE,
)
FAIL_RE = re.compile(r"\b(?:DISAGREE|FATAL|FAILED)\b", re.IGNORECASE)
SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SweepError(RuntimeError):
    pass


@functools.lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(text: str, expected_context: int) -> dict[str, Any]:
    failure = FAIL_RE.search(text)
    if failure:
        line = text.count("\n", 0, failure.start()) + 1
        raise SweepError(f"failure marker {failure.group(0)!r} at log line {line}")
    if "scheduler: GLOBAL QUEUE" not in text:
        raise SweepError("run did not report the GLOBAL QUEUE scheduler")
    if "SKELETON BUILD" in text:
        raise SweepError("skeleton interpreter output is not a valid measurement")
    if not re.search(r"argmax check:.*\bAGREE\b", text):
        raise SweepError("missing successful device/host argmax agreement")
    if "PLOW_IDS" not in text:
        raise SweepError("missing PLOW_IDS correctness artifact")
    matches = list(RESULT_RE.finditer(text))
    if len(matches) != 1:
        raise SweepError(f"expected exactly one PLOW_RESULT, found {len(matches)}")
    fields = matches[0].groupdict()
    context = int(fields["context"])
    samples = int(fields["n"])
    if context != expected_context:
        raise SweepError(f"PLOW_RESULT ctx={context}, expected {expected_context}")
    if samples != TIMED_STEPS:
        raise SweepError(f"PLOW_RESULT n={samples}, expected {TIMED_STEPS}")
    return {
        "context": context,
        "mean_ms": float(fields["mean"]),
        "median_ms": float(fields["median"]),
        "reported_sd_ms": float(fields["sd"]),
        "timed_steps": samples,
        "host_ms": float(fields["host"]),
        "kernel_ms": float(fields["kernel"]),
    }


def load_vllm(path: Path, config_name: str) -> dict[int, float]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SweepError(f"cannot read vLLM baseline {path}: {error}") from error
    configs = [item for item in document.get("configs", []) if item.get("config") == config_name]
    if len(configs) != 1:
        names = [item.get("config") for item in document.get("configs", [])]
        raise SweepError(
            f"vLLM config {config_name!r} not found exactly once in {path}; available={names}"
        )
    result = {int(row["input_len"]): float(row["tpot_ms"]) for row in configs[0]["results"]}
    missing = set(CONTEXTS) - result.keys()
    if missing:
        raise SweepError(f"vLLM config {config_name!r} is missing contexts {sorted(missing)}")
    return result


def prompt_path(prompt_dir: Path, context: int, prompt: int) -> Path:
    return prompt_dir / f"ids_{context}_p{prompt}.bin"


def validate_args(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[int, float]]:
    if not SAFE_LABEL_RE.fullmatch(args.precision_label):
        raise SweepError("--precision-label must contain only letters, digits, '.', '_' or '-'")
    if args.precision_label == "bf16":
        if args.fp8_dir is not None:
            raise SweepError("bf16 must not set --fp8-dir")
        if args.vllm_config is None:
            args.vllm_config = "bf16"
    else:
        if args.fp8_dir is None:
            raise SweepError(
                "non-bf16 precision requires --fp8-dir; label hybrid assets explicitly"
            )
        if args.vllm_config is None:
            raise SweepError("non-bf16 precision requires explicit --vllm-config")

    required_files = [args.binary, args.packet, args.vllm_json]
    for path in required_files:
        if not path.is_file():
            raise SweepError(f"required file does not exist: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SweepError(f"binary is not executable: {args.binary}")
    for path in (args.model_dir, args.prompt_dir):
        if not path.is_dir():
            raise SweepError(f"required directory does not exist: {path}")
    if args.fp8_dir is not None and not args.fp8_dir.is_dir():
        raise SweepError(f"FP8 directory does not exist: {args.fp8_dir}")

    prompts = []
    for context in CONTEXTS:
        for prompt in range(PROMPTS_PER_CONTEXT):
            path = prompt_path(args.prompt_dir, context, prompt)
            if not path.is_file():
                raise SweepError(
                    f"missing exact seed-0 prompt {path}; generate it with "
                    "make_vllm_random_ids.py"
                )
            expected_bytes = context * 4
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise SweepError(
                    f"prompt {path} is {actual_bytes} bytes, expected {expected_bytes} "
                    f"({context} little-endian int32 IDs)"
                )
            prompts.append(
                {
                    "context": context,
                    "prompt": prompt,
                    "path": path,
                    "sha256": sha256(path),
                }
            )
    return prompts, load_vllm(args.vllm_json, args.vllm_config)


def run_command(args: argparse.Namespace, prompt: dict[str, Any], log_path: Path) -> str:
    command = [
        str(args.binary),
        str(args.packet),
        str(args.model_dir),
        str(prompt["path"]),
        str(STEPS),
    ]
    environment = os.environ.copy()
    environment["PLOW_PREFILL"] = "0"
    environment["PLOW_WARMUP"] = str(WARMUP)
    if args.fp8_dir is None:
        environment.pop("PLOW_FP8_DIR", None)
    else:
        environment["PLOW_FP8_DIR"] = str(args.fp8_dir)
    if args.dump_steps:
        environment["PLOW_DUMP_STEPS"] = "1"
    else:
        environment.pop("PLOW_DUMP_STEPS", None)

    print("+", " ".join(command), flush=True)
    output: list[str] = []
    with log_path.open("w") as log:
        log.write("# command: " + " ".join(command) + "\n")
        log.write(
            f"# precision={args.precision_label} PLOW_PREFILL=0 "
            f"PLOW_WARMUP={WARMUP} steps={STEPS}\n"
        )
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            output.append(line)
        return_code = process.wait()
    text = "".join(output)
    if return_code != 0:
        raise SweepError(f"benchmark exited {return_code}; raw log: {log_path}")
    return text


def git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip()
        )
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def aggregate(rows: list[dict[str, Any]], vllm: dict[int, float]) -> list[dict[str, Any]]:
    aggregates = []
    for context in CONTEXTS:
        selected = [row for row in rows if row["context"] == context]
        if not selected:
            continue
        # Incremental checkpoint files intentionally omit an aggregate until all
        # three equal-weight prompt measurements for this context are present.
        if len(selected) != PROMPTS_PER_CONTEXT:
            continue
        means = [row["mean_ms"] for row in selected]
        mean_ms = statistics.fmean(means)
        aggregates.append(
            {
                "context": context,
                "prompt_count": len(selected),
                "aggregation": "equal weight per prompt",
                "mean_ms": mean_ms,
                "median_ms_mean": statistics.fmean(row["median_ms"] for row in selected),
                "within_run_sd_ms_mean": statistics.fmean(
                    row["reported_sd_ms"] for row in selected
                ),
                "across_prompt_mean_sd_ms": statistics.stdev(means),
                "host_ms_mean": statistics.fmean(row["host_ms"] for row in selected),
                "kernel_ms_mean": statistics.fmean(row["kernel_ms"] for row in selected),
                "vllm_tpot_ms": vllm[context],
                "plow_over_vllm": mean_ms / vllm[context],
                "vllm_over_plow": vllm[context] / mean_ms,
            }
        )
    return aggregates


def write_outputs(
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    vllm: dict[int, float],
    status: str,
) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate(rows, vllm) if rows else []
    prompt_hashes = {
        f"{item['context']}/p{item['prompt']}": item["sha256"] for item in prompts
    }
    document = {
        "schema_version": 1,
        "status": status,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "campaign": "gemma4-26b-global-queue-decode",
        "model": "gemma-4-26B-A4B-it",
        "precision_label": args.precision_label,
        "paths": {
            "binary": str(args.binary.resolve()),
            "packet": str(args.packet.resolve()),
            "model_dir": str(args.model_dir.resolve()),
            "fp8_dir": str(args.fp8_dir.resolve()) if args.fp8_dir else None,
            "prompt_dir": str(args.prompt_dir.resolve()),
        },
        "sha256": {
            "binary": sha256(args.binary),
            "packet": sha256(args.packet),
            "prompts": prompt_hashes,
        },
        "git": git_revision(),
        "method": {
            "scheduler_required": "GLOBAL QUEUE",
            "prefill": False,
            "warmup_steps": WARMUP,
            "total_steps": STEPS,
            "timed_steps": TIMED_STEPS,
            "contexts": list(CONTEXTS),
            "prompts_per_context": PROMPTS_PER_CONTEXT,
            "prompt_source": "vLLM RandomDataset seed 0, reset per context",
            "aggregate": "arithmetic mean of the three per-prompt means (equal weight)",
        },
        "vllm": {
            "source": str(args.vllm_json.resolve()),
            "config": args.vllm_config,
            "tpot_ms": {str(key): value for key, value in vllm.items()},
            "ratio_convention": "plow_over_vllm < 1.0 means Plow is faster",
        },
        "results": rows,
        "aggregates": aggregates,
    }
    json_path = args.out_dir / f"{args.precision_label}.json"
    json_path.write_text(json.dumps(document, indent=2) + "\n")

    csv_path = args.out_dir / f"{args.precision_label}.csv"
    columns = [
        "precision_label",
        "context",
        "prompt",
        "kind",
        "mean_ms",
        "median_ms",
        "reported_sd_ms",
        "across_prompt_mean_sd_ms",
        "timed_steps",
        "host_ms",
        "kernel_ms",
        "vllm_tpot_ms",
        "plow_over_vllm",
        "vllm_over_plow",
        "prompt_sha256",
        "log_path",
    ]
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "precision_label": args.precision_label,
                    "context": row["context"],
                    "prompt": row["prompt"],
                    "kind": "prompt",
                    "mean_ms": row["mean_ms"],
                    "median_ms": row["median_ms"],
                    "reported_sd_ms": row["reported_sd_ms"],
                    "timed_steps": row["timed_steps"],
                    "host_ms": row["host_ms"],
                    "kernel_ms": row["kernel_ms"],
                    "vllm_tpot_ms": row["vllm_tpot_ms"],
                    "plow_over_vllm": row["plow_over_vllm"],
                    "vllm_over_plow": row["vllm_over_plow"],
                    "prompt_sha256": row["prompt_sha256"],
                    "log_path": row["log_path"],
                }
            )
        for row in aggregates:
            writer.writerow(
                {
                    "precision_label": args.precision_label,
                    "context": row["context"],
                    "prompt": "all",
                    "kind": "equal_weight_aggregate",
                    "mean_ms": row["mean_ms"],
                    "median_ms": row["median_ms_mean"],
                    "reported_sd_ms": row["within_run_sd_ms_mean"],
                    "across_prompt_mean_sd_ms": row["across_prompt_mean_sd_ms"],
                    "timed_steps": TIMED_STEPS,
                    "host_ms": row["host_ms_mean"],
                    "kernel_ms": row["kernel_ms_mean"],
                    "vllm_tpot_ms": row["vllm_tpot_ms"],
                    "plow_over_vllm": row["plow_over_vllm"],
                    "vllm_over_plow": row["vllm_over_plow"],
                }
            )
    print(f"wrote {json_path} and {csv_path} ({len(rows)} prompt rows)", flush=True)


def run_sweep(
    args: argparse.Namespace, prompts: list[dict[str, Any]], vllm: dict[int, float]
) -> None:
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        context = prompt["context"]
        index = prompt["prompt"]
        log_path = raw_dir / f"ctx{context}_p{index}.log"
        if args.resume and log_path.is_file():
            print(f"resume: parsing {log_path}", flush=True)
            text = log_path.read_text(errors="replace")
        else:
            text = run_command(args, prompt, log_path)
        try:
            row = parse_log(text, context)
        except SweepError as error:
            raise SweepError(f"ctx={context} prompt={index}: {error}; raw log: {log_path}") from error
        row.update(
            {
                "prompt": index,
                "prompt_sha256": prompt["sha256"],
                "vllm_tpot_ms": vllm[context],
                "plow_over_vllm": row["mean_ms"] / vllm[context],
                "vllm_over_plow": vllm[context] / row["mean_ms"],
                "log_path": str(log_path.resolve()),
            }
        )
        rows.append(row)
        write_outputs(args, prompts, rows, vllm, "running")
    write_outputs(args, prompts, rows, vllm, "complete")


def self_test() -> None:
    good = """scheduler: GLOBAL QUEUE (one atomic cursor)
argmax check: device=42 host=42  AGREE
PLOW_IDS 42 17 9
PLOW_RESULT ctx=1024 mean_ms=8.0000 median_ms=7.9000 sd_ms=0.1000 n=120 host_ms=0.0400 kernel_ms=7.9600
"""
    parsed = parse_log(good, 1024)
    assert parsed["mean_ms"] == 8.0
    rows = []
    for prompt, mean in enumerate((7.0, 8.0, 9.0)):
        row = dict(parsed)
        row.update({"prompt": prompt, "mean_ms": mean})
        rows.append(row)
    aggregates = aggregate(rows, {context: 10.0 for context in CONTEXTS})
    assert len(aggregates) == 1
    assert aggregates[0]["mean_ms"] == 8.0
    assert math.isclose(aggregates[0]["plow_over_vllm"], 0.8)
    try:
        parse_log(good.replace("AGREE", "*** DISAGREE ***"), 1024)
    except SweepError:
        pass
    else:
        raise AssertionError("DISAGREE marker was accepted")
    try:
        parse_log(good.replace("GLOBAL QUEUE", "STATIC per-block stream"), 1024)
    except SweepError:
        pass
    else:
        raise AssertionError("static scheduler was accepted")
    print("self-test PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision-label", help="honest asset label: bf16, fp8-full, ...")
    parser.add_argument("--binary", type=Path, help="global-queue gemma4_sm120_chat binary")
    parser.add_argument("--packet", type=Path, help="packet matching this precision path")
    parser.add_argument("--model-dir", type=Path, help="base BF16 checkpoint directory")
    parser.add_argument("--fp8-dir", type=Path, help="FP8 twin safetensors directory")
    parser.add_argument("--prompt-dir", type=Path, help="seed-0 prompt .bin directory")
    parser.add_argument("--out-dir", type=Path, help="raw logs plus CSV/JSON output")
    parser.add_argument("--vllm-json", type=Path, default=DEFAULT_VLLM_JSON)
    parser.add_argument("--vllm-config", help="comparison config in the vLLM JSON")
    parser.add_argument("--lease-label", default=None)
    parser.add_argument("--lease-timeout", type=int, default=43200)
    parser.add_argument("--gpulease", type=Path, default=Path("/usr/local/bin/gpulease"))
    parser.add_argument("--resume", action="store_true", help="reuse valid existing raw logs")
    parser.add_argument("--dump-steps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = ("precision_label", "binary", "packet", "model_dir", "prompt_dir", "out_dir")
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        parser.error("required for a sweep: " + ", ".join(missing))

    try:
        prompts, vllm = validate_args(args)
        if args.dry_run:
            print(
                f"validated {args.precision_label}: {len(prompts)} runs, one lease, "
                f"contexts={','.join(map(str, CONTEXTS))}"
            )
            for prompt in prompts:
                print(
                    args.binary,
                    args.packet,
                    args.model_dir,
                    prompt["path"],
                    STEPS,
                )
            return 0

        if os.environ.get("PLOW_26B_SWEEP_IN_LEASE") != "1":
            if not args.gpulease.is_file():
                raise SweepError(f"gpulease does not exist: {args.gpulease}")
            label = args.lease_label or f"gemma26-{args.precision_label}"
            command = [
                str(args.gpulease),
                label,
                "env",
                "PLOW_26B_SWEEP_IN_LEASE=1",
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ]
            environment = os.environ.copy()
            environment["GPU_LEASE_TIMEOUT"] = str(args.lease_timeout)
            print(
                f"acquiring one gpulease for all {len(prompts)} {args.precision_label} runs",
                flush=True,
            )
            return subprocess.call(command, env=environment)

        run_sweep(args, prompts, vllm)
        return 0
    except SweepError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
