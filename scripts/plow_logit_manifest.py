#!/usr/bin/env python3
"""Describe an ``amd-bench --dump-logits`` run and emit vLLM oracle cases."""

import argparse
import ast
import array
import hashlib
import json
import re
import sys
from pathlib import Path


def read_ids(path):
    text = path.read_text().strip()
    if text.startswith("[") or text.startswith("{"):
        value = json.loads(text)
        if isinstance(value, dict):
            value = value["prompt_token_ids"]
        return [int(x) for x in value]
    return [int(x) for x in text.split(",") if x.strip()]


def digest(ids):
    values = array.array("I", ids)
    if values.itemsize != 4:
        raise RuntimeError("host unsigned int is not 32 bits")
    if sys.byteorder != "little":
        values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--prompt", required=True, type=Path)
    p.add_argument("--stdout", required=True, type=Path)
    p.add_argument("--logits-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise ValueError("unsafe name")

    prompt = read_ids(args.prompt)
    text = args.stdout.read_text()
    first_match = re.search(r"prefill:.*?->\s*(\d+)", text)
    list_matches = re.findall(r"^\s*\[(?:\s*\d+\s*,?)+\]\s*$", text, re.M)
    if not first_match or not list_matches:
        raise ValueError("stdout lacks the prefill token or greedy decode list")
    first = int(first_match.group(1))
    decoded = [int(x) for x in ast.literal_eval(list_matches[-1])]

    rows = []
    history = list(prompt)
    files = [("prefill", first)]
    history.append(first)
    for step, token in enumerate(decoded):
        files.append((f"{step:03}", token))
    for index, (tag, sampled) in enumerate(files):
        cid = f"{args.name}-{tag}"
        path = args.logits_dir / f"logits_{tag}.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
        row_history = prompt if index == 0 else history[: len(prompt) + index]
        rows.append(
            {
                "id": cid,
                "file": str(path.resolve()),
                "dtype": "bf16",
                "prompt_token_ids": row_history,
                "prompt_len": len(row_history),
                "prompt_sha256_u32le": digest(row_history),
                "sampled_token_id": sampled,
            }
        )
        if index > 0:
            history.append(decoded[index - 1])

    result = {"schema": 1, "producer": "plow-amd-bench", "name": args.name, "cases": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
