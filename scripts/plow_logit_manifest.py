#!/usr/bin/env python3
"""Describe amd-bench or step_bench full-logit dumps on their exact token histories."""

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


def assemble_sharded_row(directory, tag, shards, vocab, batch_size=1, row=0):
    if shards < 2 or vocab < 1 or vocab % shards:
        raise ValueError("vocabulary must divide into at least two equal shards")
    if batch_size < 1 or not 0 <= row < batch_size:
        raise ValueError("invalid batch row")
    row_bytes = 2 * (vocab // shards)
    parts = []
    for rank in range(shards):
        path = directory / f"logits.rk{rank}.{tag}.bin"
        part = path.read_bytes()
        if len(part) != row_bytes * batch_size:
            raise ValueError(f"{path}: expected {batch_size} BF16 vocabulary-shard rows ({row_bytes * batch_size} bytes)")
        bits = array.array("H")
        bits.frombytes(part)
        if sys.byteorder != "little":
            bits.byteswap()
        if any(value & 0x7F80 == 0x7F80 for value in bits):
            raise ValueError(f"{path}: nonfinite BF16 logits")
        parts.append(part[row * row_bytes : (row + 1) * row_bytes])
    suffix = f"_s{row}" if batch_size > 1 else ""
    path = directory / f"full_logits_{tag}{suffix}.bin"
    path.write_bytes(b"".join(parts))
    return path


def assemble_replicated_row(directory, tag, vocab, batch_size=1, row=0):
    if vocab < 1 or batch_size < 1 or not 0 <= row < batch_size:
        raise ValueError("invalid replicated vocabulary row")
    row_bytes = 2 * vocab
    source = directory / f"logits.{tag}.bin"
    data = source.read_bytes()
    if len(data) != row_bytes * batch_size:
        raise ValueError(f"{source}: expected {batch_size} full BF16 vocabulary rows ({row_bytes * batch_size} bytes)")
    values = array.array("H")
    values.frombytes(data[row * row_bytes : (row + 1) * row_bytes])
    if sys.byteorder != "little":
        values.byteswap()
    if any(value & 0x7F80 == 0x7F80 for value in values):
        raise ValueError(f"{source}: nonfinite BF16 logits")
    suffix = f"_s{row}" if batch_size > 1 else ""
    path = directory / f"full_logits_{tag}{suffix}.bin"
    path.write_bytes(data[row * row_bytes : (row + 1) * row_bytes])
    return path


def batched_cases(args, text):
    batch = args.batch_size
    prompts = [[int(token.strip()) for token in one.split(",")]
               for one in args.prompt.read_text().strip().split(";")]
    if not prompts or len(prompts) > batch or any(not ids for ids in prompts):
        raise ValueError("invalid batched prompt count")
    declared = re.findall(r"batched TP decode: (\d+) active of (\d+) slots per dispatch", text)
    legacy = re.findall(r"batched TP decode: (\d+) sequences per dispatch", text)
    if declared:
        valid_batch = len(declared) == 1 and int(declared[0][0]) == batch and int(declared[0][1]) >= batch
    else:
        valid_batch = legacy == [str(batch)]
    if not valid_batch:
        raise ValueError("stdout batch size differs from requested batch")
    prefills = re.findall(r"^\s*slot (\d+): prefill (\d+) tokens -> sampled (\d+)\s*$", text, re.M)
    if [int(slot) for slot, _, _ in prefills] != list(range(batch)):
        raise ValueError("missing or duplicate slot prefill records")
    chains = [ast.literal_eval(line) for line in
              re.findall(r"^\s*\[(?:\s*\d+\s*,?)+\]\s*$", text, re.M)]
    if len(chains) != batch or not chains[0] or any(len(c) != len(chains[0]) for c in chains):
        raise ValueError("missing or inconsistent per-slot decode chains")
    histories = []
    for slot, (_, length, first) in enumerate(prefills):
        ids = prompts[slot % len(prompts)]
        if len(ids) != int(length):
            raise ValueError("prefill length differs from slot prompt")
        histories.append([*ids, int(first)])
    rows = []
    for step in range(len(chains[0])):
        tag = f"b{step:03}"
        for slot in range(batch):
            if args.replicated_vocab:
                path = assemble_replicated_row(args.logits_dir, tag, args.vocab, batch, slot)
            else:
                path = assemble_sharded_row(args.logits_dir, tag, args.tp_shards,
                                            args.vocab, batch, slot)
            history = list(histories[slot])
            sampled = chains[slot][step]
            rows.append(dict(id=f"{args.name}-s{slot}-{tag}", file=str(path.resolve()),
                             dtype="bf16", prompt_token_ids=history, prompt_len=len(history),
                             prompt_sha256_u32le=digest(history), sampled_token_id=sampled,
                             generation_step=step + 1, execution_phase="decode_output",
                             sequence_row=slot, batch_size=batch))
            histories[slot].append(sampled)
    return rows


def step_cases(path, name, vocab):
    data = json.loads(path.read_text())
    if data.get("max_inst") is not None or data.get("max_segments") is not None:
        raise ValueError("partial decode dumps are not complete decode logits")
    batch = data["slots"]
    histories, sampled = data["token_histories"], data["sampled_token_ids"]
    if vocab < 1 or batch < 1 or len(histories) != batch or len(sampled) != batch:
        raise ValueError("invalid step dump vocabulary, batch or history count")
    if data["steps"] < 1 or data["warmup"] < 0:
        raise ValueError("step dump needs at least one measured decode step")
    for ids, token in zip(histories, sampled):
        if len(ids) != data["ctx"] + data["warmup"] + data["steps"]:
            raise ValueError("step dump history length disagrees with execution")
        if any(type(t) is not int or not 0 <= t < vocab for t in [*ids, token]):
            raise ValueError("step dump token outside vocabulary")
    tensors = [t for t in data["tensors"] if t["name"] == "act.logits"]
    if len(tensors) != 1:
        raise ValueError("step dump must contain exactly one act.logits tensor")
    tensor = tensors[0]
    source = path.parent / tensor["file"]
    raw = source.read_bytes()
    row_bytes = 2 * vocab
    if len(raw) != tensor["bytes"] or len(raw) < batch * row_bytes or len(raw) % row_bytes:
        raise ValueError("step dump is not a complete BF16 vocabulary-row allocation")
    rows = []
    for slot, (history, token) in enumerate(zip(histories, sampled)):
        part = raw[slot * row_bytes:(slot + 1) * row_bytes]
        bits = array.array("H")
        bits.frombytes(part)
        if sys.byteorder != "little":
            bits.byteswap()
        if any(value & 0x7F80 == 0x7F80 for value in bits):
            raise ValueError("nonfinite active step logits")
        output = path.parent / f"full_logits_{name}_s{slot:03}.bin"
        output.write_bytes(part)
        rows.append(dict(id=f"{name}-s{slot:03}", file=str(output.resolve()), dtype="bf16",
                         prompt_token_ids=history, prompt_len=len(history),
                         prompt_sha256_u32le=digest(history), sampled_token_id=token,
                         generation_step=data["warmup"] + data["steps"], execution_phase="decode_output",
                         sequence_row=slot, batch_size=batch))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--prompt", type=Path)
    p.add_argument("--step-manifest", type=Path, help="step_bench --dump-tensors act.logits manifest")
    p.add_argument("--stdout", type=Path)
    p.add_argument("--logits-dir", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tp-shards", type=int, default=1,
                   help="assemble logits.rkR.TAG.bin captured with PLOW_DUMP_ACT and PLOW_TRACE_ALLRANKS")
    p.add_argument("--replicated-vocab", action="store_true",
                   help="read full-vocabulary rank-0 logits.TAG.bin instead of assembling TP shards")
    p.add_argument("--vocab", type=int, help="full vocabulary size; required with --tp-shards")
    p.add_argument("--batch-size", type=int, default=1,
                   help="assemble amd-bench --batched decode dumps; prompts use semicolon-separated token lists")
    args = p.parse_args()
    if args.step_manifest:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
            p.error("unsafe name")
        if not args.vocab or args.prompt or args.stdout or args.logits_dir or args.tp_shards != 1 or args.replicated_vocab or args.batch_size != 1:
            p.error("--step-manifest requires --vocab and cannot be combined with amd-bench capture options")
        rows = step_cases(args.step_manifest, args.name, args.vocab)
        args.output.write_text(json.dumps(dict(schema=1, producer="plow-step-bench", name=args.name,
            vocab_size=args.vocab, cases=rows, source_manifest=str(args.step_manifest.resolve()),
            source_manifest_sha256=hashlib.sha256(args.step_manifest.read_bytes()).hexdigest()), indent=2) + "\n")
        return
    if not all((args.prompt, args.stdout, args.logits_dir)):
        p.error("amd-bench capture requires --prompt, --stdout and --logits-dir")
    if args.tp_shards < 1 or (args.tp_shards > 1 and not args.replicated_vocab and (not args.vocab or args.vocab % args.tp_shards)):
        p.error("--tp-shards requires a positive divisible --vocab")
    if args.replicated_vocab and (not args.vocab or args.vocab < 1):
        p.error("--replicated-vocab requires a positive --vocab")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise ValueError("unsafe name")
    if args.batch_size < 1 or (args.batch_size > 1 and args.tp_shards < 2 and not args.replicated_vocab):
        p.error("batched capture requires --tp-shards and a positive --batch-size")

    text = args.stdout.read_text()
    if args.batch_size > 1:
        result = dict(schema=1, producer="plow-amd-bench", name=args.name,
                      cases=batched_cases(args, text), vocab_size=args.vocab,
                      batch_size=args.batch_size)
        result["vocabulary_layout"] = "replicated" if args.replicated_vocab else "sharded"
        result["tensor_parallel_ranks" if args.replicated_vocab else "vocabulary_shards"] = args.tp_shards
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return

    prompt = read_ids(args.prompt)
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
        if args.replicated_vocab:
            path = assemble_replicated_row(args.logits_dir, tag, args.vocab)
        elif args.tp_shards > 1:
            path = assemble_sharded_row(args.logits_dir, tag, args.tp_shards, args.vocab)
        else:
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
                "generation_step": index,
                "execution_phase": "prefill_output" if index == 0 else "decode_output",
            }
        )
        if index > 0:
            history.append(decoded[index - 1])

    result = {"schema": 1, "producer": "plow-amd-bench", "name": args.name, "cases": rows}
    if args.replicated_vocab:
        result.update(vocab_size=args.vocab, tensor_parallel_ranks=args.tp_shards,
                      vocabulary_layout="replicated", sequence_row=0)
    elif args.tp_shards > 1:
        result.update(vocab_size=args.vocab, vocabulary_shards=args.tp_shards, sequence_row=0)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
