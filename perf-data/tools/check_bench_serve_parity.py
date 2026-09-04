#!/usr/bin/env python3
"""Fail-closed exact token and AMD dispatch parity check for bench vs serve."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


class ParityError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ParityError(message)


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ParityError(f"cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: root must be an object")
    return value


def token_rows(value: Any, name: str) -> list[list[int]]:
    require(isinstance(value, list) and len(value) == 1, f"{name} must contain exactly one row")
    row = value[0]
    require(isinstance(row, list) and row, f"{name}[0] must be a non-empty token array")
    require(
        all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in row),
        f"{name}[0] contains an invalid token id",
    )
    return value


def validate_diagnostics(diagnostics: Any, prompt_len: int, output_len: int, num_gpus: int) -> None:
    require(isinstance(diagnostics, dict), "bench diagnostics are missing")
    require(diagnostics.get("supported") is True, "bench diagnostics are unsupported")
    require(diagnostics.get("complete") is True, "bench diagnostics are incomplete")
    require(diagnostics.get("overflowed") is False, "bench diagnostics overflowed")

    prefill = diagnostics.get("prefill_selections")
    require(isinstance(prefill, list) and prefill, "prefill selections are missing")
    cursor = 0
    for index, selection in enumerate(prefill):
        require(isinstance(selection, dict), f"prefill selection {index} is not an object")
        slot = selection.get("slot")
        row_start = selection.get("row_start")
        rows = selection.get("rows")
        bucket = selection.get("bucket")
        require(slot == 0, f"prefill selection {index} used slot {slot}, expected 0")
        require(row_start == cursor, f"prefill selection {index} starts at {row_start}, expected {cursor}")
        require(isinstance(rows, int) and rows > 0, f"prefill selection {index} has invalid rows")
        require(isinstance(bucket, int) and bucket >= rows, f"prefill selection {index} bucket is too small")
        cursor += rows
    require(cursor == prompt_len, f"prefill selections cover {cursor}/{prompt_len} prompt rows")

    decode = diagnostics.get("decode_selections")
    require(isinstance(decode, list), "decode selections are missing")
    decode_steps = 0
    for index, selection in enumerate(decode):
        require(isinstance(selection, dict), f"decode selection {index} is not an object")
        require(selection.get("occupied_rows") == 1, f"decode selection {index} is not B1")
        require(
            isinstance(selection.get("bucket"), int) and selection["bucket"] >= 1,
            f"decode selection {index} has invalid bucket",
        )
        require(
            isinstance(selection.get("steps"), int) and selection["steps"] > 0,
            f"decode selection {index} has invalid steps",
        )
        decode_steps += selection["steps"]
    expected_decode_steps = max(output_len - 1, 0)
    require(
        decode_steps == expected_decode_steps,
        f"decode selections cover {decode_steps}/{expected_decode_steps} post-prefill tokens",
    )

    agreement = diagnostics.get("rank_agreement")
    if num_gpus == 1 and agreement is None:
        return
    require(isinstance(agreement, dict), "TP rank agreement evidence is missing")
    require(agreement.get("ranks") == num_gpus, "TP rank count does not match num_gpus")
    require(
        isinstance(agreement.get("sampled_token_every"), int)
        and agreement["sampled_token_every"] > 0,
        "TP sampled-token cadence is invalid",
    )
    require(agreement.get("counter_audit_every_dispatch") is True, "TP counter audit is not fail-closed")
    require(agreement.get("prefill_completion_all_ranks") is True, "TP prefill completion was not checked")


def check(bench: dict[str, Any], endpoint: dict[str, Any]) -> None:
    require(bench.get("schema") == "plowrt.bench.v1", "unsupported bench schema")
    require(bench.get("concurrency") == 1, "bench concurrency must be 1")
    require(bench.get("requests") == 1, "bench requests must be 1")
    require(bench.get("completed") == 1, "bench did not complete exactly one request")
    require(bench.get("failed") == 0, "bench reports failed requests")
    require(bench.get("warmup_requests") == 0, "bench parity run must not include warmup")
    require(isinstance(bench.get("num_gpus"), int) and bench["num_gpus"] >= 1, "invalid num_gpus")
    require(isinstance(bench.get("model"), str) and bench["model"], "bench model is missing")
    parity = bench.get("parity")
    require(isinstance(parity, dict), "bench parity report is missing")
    prompt_rows = token_rows(parity.get("prompt_token_ids"), "prompt_token_ids")
    output_rows = token_rows(parity.get("output_token_ids"), "output_token_ids")
    prompt = prompt_rows[0]
    output = output_rows[0]
    require(bench.get("prompt_tokens") == len(prompt), "bench prompt token count mismatch")
    require(bench.get("output_tokens") == len(output), "bench output token count mismatch")
    input_report = bench.get("input")
    require(isinstance(input_report, dict), "bench input report is missing")
    require(input_report.get("mode") == "token_ids", "bench input was not exact token IDs")
    require(input_report.get("tokens_per_request") == len(prompt), "bench input length mismatch")
    validate_diagnostics(bench.get("diagnostics"), len(prompt), len(output), bench["num_gpus"])

    require(endpoint.get("object") == "text_completion", "endpoint response is not a completion")
    require(endpoint.get("model") == bench["model"], "bench/serve model identity differs")
    choices = endpoint.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "endpoint must contain one choice")
    require(isinstance(choices[0], dict), "endpoint choice is not an object")
    require(choices[0].get("finish_reason") == "length", "endpoint did not finish at exact length")
    endpoint_ids = endpoint.get("token_ids")
    require(isinstance(endpoint_ids, dict), "endpoint token_ids are missing")
    require(endpoint_ids.get("prompt") == prompt, "bench/serve prompt token IDs differ")
    require(endpoint_ids.get("completion") == output, "bench/serve output token IDs differ")
    usage = endpoint.get("usage")
    require(isinstance(usage, dict), "endpoint usage is missing")
    require(usage.get("prompt_tokens") == len(prompt), "endpoint prompt usage mismatch")
    require(usage.get("completion_tokens") == len(output), "endpoint completion usage mismatch")
    require(usage.get("total_tokens") == len(prompt) + len(output), "endpoint total usage mismatch")


def fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = [1, 2, 3]
    output = [4, 5, 6]
    bench = {
        "schema": "plowrt.bench.v1",
        "concurrency": 1,
        "requests": 1,
        "completed": 1,
        "failed": 0,
        "warmup_requests": 0,
        "num_gpus": 2,
        "model": "model",
        "prompt_tokens": 3,
        "output_tokens": 3,
        "input": {"mode": "token_ids", "tokens_per_request": 3, "seed": None},
        "parity": {"prompt_token_ids": [prompt], "output_token_ids": [output]},
        "diagnostics": {
            "supported": True,
            "complete": True,
            "overflowed": False,
            "scope": "warmup_and_measured",
            "prefill_selections": [{"slot": 0, "row_start": 0, "rows": 3, "bucket": 128}],
            "decode_selections": [{"occupied_rows": 1, "bucket": 1, "steps": 2}],
            "rank_agreement": {
                "ranks": 2,
                "sampled_token_every": 1,
                "counter_audit_every_dispatch": True,
                "prefill_completion_all_ranks": True,
            },
        },
    }
    endpoint = {
        "object": "text_completion",
        "model": "model",
        "choices": [{"finish_reason": "length", "text": "x"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
        "token_ids": {"prompt": prompt, "completion": output},
    }
    return bench, endpoint


def self_test() -> None:
    bench, endpoint = fixture()
    check(bench, endpoint)
    failures = []
    bad_endpoint = copy.deepcopy(endpoint)
    bad_endpoint["token_ids"]["completion"][1] = 99
    failures.append((bench, bad_endpoint))
    bad_chunk = copy.deepcopy(bench)
    bad_chunk["diagnostics"]["prefill_selections"][0]["row_start"] = 1
    failures.append((bad_chunk, endpoint))
    bad_tp = copy.deepcopy(bench)
    bad_tp["diagnostics"]["rank_agreement"]["ranks"] = 1
    failures.append((bad_tp, endpoint))
    for index, pair in enumerate(failures):
        try:
            check(*pair)
        except ParityError:
            continue
        raise AssertionError(f"negative parity fixture {index} passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench", nargs="?", type=Path)
    parser.add_argument("endpoint", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("PASS: bench/serve parity checker self-test")
            return 0
        require(args.bench is not None and args.endpoint is not None, "bench and endpoint JSON are required")
        check(load(args.bench), load(args.endpoint))
        print("PASS: bench and serve token streams and dispatch evidence match")
        return 0
    except ParityError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
