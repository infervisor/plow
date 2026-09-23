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


def assemble_sharded_row(directory, tag, shards, vocab):
    if shards < 2 or vocab < 1 or vocab % shards:
        raise ValueError("vocabulary must divide into at least two equal shards")
    row_bytes = 2 * (vocab // shards)
    parts = []
    for rank in range(shards):
        path = directory / f"logits.rk{rank}.{tag}.bin"
        part = path.read_bytes()
        if len(part) != row_bytes:
            raise ValueError(f"{path}: expected exactly one BF16 vocabulary-shard row ({row_bytes} bytes)")
        bits = array.array("H")
        bits.frombytes(part)
        if sys.byteorder != "little":
            bits.byteswap()
        if any(value & 0x7F80 == 0x7F80 for value in bits):
            raise ValueError(f"{path}: nonfinite BF16 logits")
        parts.append(part)
    path = directory / f"full_logits_{tag}.bin"
    path.write_bytes(b"".join(parts))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--prompt", required=True, type=Path)
    p.add_argument("--stdout", required=True, type=Path)
    p.add_argument("--logits-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tp-shards", type=int, default=1,
                   help="assemble logits.rkR.TAG.bin captured with PLOW_DUMP_ACT and PLOW_TRACE_ALLRANKS")
    p.add_argument("--vocab", type=int, help="full vocabulary size; required with --tp-shards")
    args = p.parse_args()
    if args.tp_shards < 1 or (args.tp_shards > 1 and (not args.vocab or args.vocab % args.tp_shards)):
        p.error("--tp-shards requires a positive divisible --vocab")
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
        path = (assemble_sharded_row(args.logits_dir, tag, args.tp_shards, args.vocab)
                if args.tp_shards > 1 else args.logits_dir / f"logits_{tag}.bin")
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
                "generation_step": index,
                "execution_phase": "prefill_output" if index == 0 else "decode_output",
            }
        )
        if index > 0:
            history.append(decoded[index - 1])

    result = {"schema": 1, "producer": "plow-amd-bench", "name": args.name, "cases": rows}
    if args.tp_shards > 1:
        result.update(vocab_size=args.vocab, vocabulary_shards=args.tp_shards, sequence_row=0)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
