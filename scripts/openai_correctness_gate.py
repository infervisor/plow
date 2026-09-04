#!/usr/bin/env python3
"""Deterministic token-level correctness gate for two OpenAI-compatible servers.

The script never starts a server. Network operations require ``--execute``.
Capture one artifact after each independent cold server start, then compare three
artifacts from each endpoint.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import urllib.error
import urllib.request


SCHEMA = "plow.openai-correctness.v1"
CORPUS_SCHEMA = "plow.openai-correctness-corpus.v1"
OUTPUT_LENGTHS = (1, 256)


class GateError(ValueError):
    pass


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_json(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def write_new(path, value):
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def post_json(base_url, route, body, timeout):
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{route}",
        canonical_json(body),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise GateError(f"POST {route}: HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise GateError(f"POST {route}: {error}") from error


def tokenize(base_url, model, prompt, timeout):
    response = post_json(
        base_url,
        "/tokenize",
        {"model": model, "prompt": prompt, "add_special_tokens": False},
        timeout,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) for token in tokens
    ):
        raise GateError("/tokenize response has no integer tokens array")
    return tokens


def detokenize(base_url, model, tokens, timeout):
    response = post_json(
        base_url, "/detokenize", {"model": model, "tokens": tokens}, timeout
    )
    prompt = response.get("prompt")
    if not isinstance(prompt, str):
        raise GateError("/detokenize response has no prompt string")
    return prompt


def completion_token_ids(response):
    """Normalize Plow's root extension and vLLM's per-choice extension."""
    root_ids = response.get("token_ids")
    if isinstance(root_ids, dict):
        prompt = root_ids.get("prompt")
        completion = root_ids.get("completion")
    else:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise GateError("completion response must contain exactly one choice")
        prompt = choices[0].get("prompt_token_ids")
        completion = choices[0].get("token_ids")
    if not isinstance(prompt, list) or not all(isinstance(token, int) for token in prompt):
        raise GateError("completion response has no integer prompt token IDs")
    if not isinstance(completion, list) or not all(
        isinstance(token, int) for token in completion
    ):
        raise GateError("completion response has no integer output token IDs")
    return prompt, completion


def read_corpus(path):
    corpus = load_json(path)
    if corpus.get("schema") != CORPUS_SCHEMA:
        raise GateError(f"wrong corpus schema in {path}")
    prompts = corpus.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise GateError("corpus must contain a non-empty prompts array")
    names = set()
    for entry in prompts:
        if not isinstance(entry, dict):
            raise GateError("each corpus prompt must be an object")
        name, prompt, expected = (
            entry.get("name"),
            entry.get("prompt"),
            entry.get("expected_tokens"),
        )
        if not isinstance(name, str) or not name or name in names:
            raise GateError(f"invalid or duplicate prompt name: {name!r}")
        if not isinstance(prompt, str) or not prompt:
            raise GateError(f"prompt {name!r} is empty")
        if not isinstance(expected, int) or expected <= 0:
            raise GateError(f"prompt {name!r} has invalid expected_tokens")
        names.add(name)
    return corpus


def prepare_corpus(args):
    plan = {
        "action": "prepare-corpus",
        "endpoint": args.base_url,
        "model": args.model,
        "output": os.path.abspath(args.output),
        "target_tokens": args.target_tokens,
        "network": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    targets = args.target_tokens
    if len(set(targets)) != len(targets) or any(target <= 0 for target in targets):
        raise GateError("target token counts must be distinct positive integers")
    seed = "The quick brown fox jumps over the lazy dog. "
    repeated = seed
    needed = max(targets)
    while len(tokenize(args.base_url, args.model, repeated, args.timeout)) < needed:
        repeated += repeated
    source_ids = tokenize(args.base_url, args.model, repeated, args.timeout)
    prompts = []
    for target in targets:
        ids = source_ids[:target]
        prompt = detokenize(args.base_url, args.model, ids, args.timeout)
        roundtrip = tokenize(args.base_url, args.model, prompt, args.timeout)
        if roundtrip != ids:
            first = first_difference(ids, roundtrip)
            raise GateError(
                f"target {target} does not detokenize round-trip at token {first}"
            )
        prompts.append(
            {"name": f"tokens-{target}", "prompt": prompt, "expected_tokens": target}
        )
    corpus = {"schema": CORPUS_SCHEMA, "prompts": prompts}
    write_new(args.output, corpus)
    print(f"PASS wrote {args.output} sha256={sha256_json(corpus)}")
    return 0


def capture(args):
    corpus = read_corpus(args.corpus)
    plan = {
        "action": "capture",
        "arm": args.arm,
        "run_id": args.run_id,
        "endpoint": args.base_url,
        "model": args.model,
        "corpus": os.path.abspath(args.corpus),
        "corpus_sha256": sha256_json(corpus),
        "output_lengths": list(OUTPUT_LENGTHS),
        "output": os.path.abspath(args.output),
        "network": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    records = []
    for entry in corpus["prompts"]:
        prompt_ids = tokenize(args.base_url, args.model, entry["prompt"], args.timeout)
        if len(prompt_ids) != entry["expected_tokens"]:
            raise GateError(
                f"{entry['name']}: expected {entry['expected_tokens']} prompt tokens, "
                f"got {len(prompt_ids)}"
            )
        outputs = {}
        for output_len in OUTPUT_LENGTHS:
            response = post_json(
                args.base_url,
                "/v1/completions",
                {
                    "model": args.model,
                    "prompt": entry["prompt"],
                    "max_tokens": output_len,
                    "temperature": 0,
                    "top_p": 1,
                    "ignore_eos": True,
                    "stream": False,
                    "return_token_ids": True,
                },
                args.timeout,
            )
            response_prompt, output_ids = completion_token_ids(response)
            if response_prompt != prompt_ids:
                first = first_difference(prompt_ids, response_prompt)
                raise GateError(
                    f"{entry['name']} len={output_len}: completion retokenized prompt "
                    f"differently at token {first}"
                )
            if len(output_ids) != output_len:
                raise GateError(
                    f"{entry['name']} len={output_len}: got {len(output_ids)} output tokens"
                )
            usage = response.get("usage")
            if not isinstance(usage, dict) or (
                usage.get("prompt_tokens") != len(prompt_ids)
                or usage.get("completion_tokens") != output_len
            ):
                raise GateError(
                    f"{entry['name']} len={output_len}: usage token counts disagree"
                )
            outputs[str(output_len)] = output_ids
        if outputs["1"] != outputs["256"][:1]:
            raise GateError(
                f"{entry['name']}: one-token output is not the 256-token output prefix"
            )
        records.append(
            {"name": entry["name"], "prompt_token_ids": prompt_ids, "outputs": outputs}
        )
    artifact = {
        "schema": SCHEMA,
        "arm": args.arm,
        "run_id": args.run_id,
        "endpoint": args.base_url.rstrip("/"),
        "model": args.model,
        "corpus_sha256": sha256_json(corpus),
        "output_lengths": list(OUTPUT_LENGTHS),
        "records": records,
    }
    write_new(args.output, artifact)
    print(f"PASS wrote cold run {args.run_id}: {args.output}")
    return 0


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def validate_artifacts(paths, side):
    artifacts = [load_json(path) for path in paths]
    if len(artifacts) != 3:
        raise GateError(f"{side}: exactly three cold-run files are required")
    for path, artifact in zip(paths, artifacts):
        if artifact.get("schema") != SCHEMA:
            raise GateError(f"{side}: wrong schema in {path}")
        if artifact.get("output_lengths") != list(OUTPUT_LENGTHS):
            raise GateError(f"{side}: wrong output lengths in {path}")
        records = artifact.get("records")
        if not isinstance(records, list) or not records:
            raise GateError(f"{side}: missing records in {path}")
        for record in records:
            prompt = record.get("prompt_token_ids") if isinstance(record, dict) else None
            outputs = record.get("outputs") if isinstance(record, dict) else None
            name = record.get("name") if isinstance(record, dict) else None
            if not isinstance(name, str) or not isinstance(prompt, list):
                raise GateError(f"{side}: malformed record in {path}")
            if not all(isinstance(token, int) for token in prompt):
                raise GateError(f"{side}: non-integer prompt IDs in {path}")
            for output_len in OUTPUT_LENGTHS:
                output = outputs.get(str(output_len)) if isinstance(outputs, dict) else None
                if not isinstance(output, list) or len(output) != output_len or not all(
                    isinstance(token, int) for token in output
                ):
                    raise GateError(f"{side}: malformed out{output_len} in {path}")
            if outputs["1"] != outputs["256"][:1]:
                raise GateError(f"{side}: out1 is not an out256 prefix in {path}")
    run_ids = [artifact.get("run_id") for artifact in artifacts]
    if not all(isinstance(run_id, int) for run_id in run_ids) or sorted(run_ids) != [
        1,
        2,
        3,
    ]:
        raise GateError(f"{side}: run IDs must be exactly 1, 2, 3")
    identity = [
        (artifact.get("arm"), artifact.get("model"), artifact.get("corpus_sha256"))
        for artifact in artifacts
    ]
    if len(set(identity)) != 1:
        raise GateError(f"{side}: arm/model/corpus identity changed across cold runs")
    return sorted(artifacts, key=lambda artifact: artifact["run_id"])


def compare_record_sets(reference, candidate, context):
    left_records, right_records = reference.get("records"), candidate.get("records")
    if not isinstance(left_records, list) or not isinstance(right_records, list):
        raise GateError(f"{context}: missing records")
    if len(left_records) != len(right_records):
        raise GateError(
            f"{context}: prompt count differs: {len(left_records)} vs {len(right_records)}"
        )
    for left, right in zip(left_records, right_records):
        if left.get("name") != right.get("name"):
            raise GateError(
                f"{context}: prompt order differs: {left.get('name')!r} vs {right.get('name')!r}"
            )
        name = left["name"]
        first = first_difference(
            left.get("prompt_token_ids", []), right.get("prompt_token_ids", [])
        )
        if first is not None:
            raise GateError(f"{context}/{name}: prompt token IDs first differ at {first}")
        for output_len in OUTPUT_LENGTHS:
            key = str(output_len)
            first = first_difference(
                (left.get("outputs") or {}).get(key, []),
                (right.get("outputs") or {}).get(key, []),
            )
            if first is not None:
                a = (left.get("outputs") or {}).get(key, [])
                b = (right.get("outputs") or {}).get(key, [])
                av = a[first] if first < len(a) else "<end>"
                bv = b[first] if first < len(b) else "<end>"
                raise GateError(
                    f"{context}/{name}/out{output_len}: first divergence at output "
                    f"{first}: {av} vs {bv}"
                )


def compare(args):
    left = validate_artifacts(args.left, "left")
    right = validate_artifacts(args.right, "right")
    if left[0]["arm"] == right[0]["arm"]:
        raise GateError("left and right arm labels must differ")
    if left[0]["corpus_sha256"] != right[0]["corpus_sha256"]:
        raise GateError("left and right corpus hashes differ")
    for run in left[1:]:
        compare_record_sets(left[0], run, f"left cold run 1 vs {run['run_id']}")
    for run in right[1:]:
        compare_record_sets(right[0], run, f"right cold run 1 vs {run['run_id']}")
    for left_run, right_run in zip(left, right):
        compare_record_sets(
            left_run,
            right_run,
            f"cross-arm cold run {left_run['run_id']}",
        )
    summary = {
        "schema": SCHEMA,
        "result": "pass",
        "left_arm": left[0]["arm"],
        "right_arm": right[0]["arm"],
        "cold_runs": 3,
        "prompts": len(left[0]["records"]),
        "output_lengths": list(OUTPUT_LENGTHS),
        "corpus_sha256": left[0]["corpus_sha256"],
    }
    if args.output:
        write_new(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def selftest(_args):
    if completion_token_ids({"token_ids": {"prompt": [1], "completion": [2]}}) != (
        [1],
        [2],
    ):
        raise GateError("self-test failed to normalize Plow token IDs")
    if completion_token_ids(
        {"choices": [{"prompt_token_ids": [1], "token_ids": [2]}]}
    ) != ([1], [2]):
        raise GateError("self-test failed to normalize vLLM token IDs")
    corpus_hash = "0" * 64
    paths = {"left": [], "right": []}
    with tempfile.TemporaryDirectory() as directory:
        for arm in paths:
            for run_id in (1, 2, 3):
                artifact = {
                    "schema": SCHEMA,
                    "arm": arm,
                    "run_id": run_id,
                    "endpoint": f"http://{arm}",
                    "model": "fixture",
                    "corpus_sha256": corpus_hash,
                    "output_lengths": list(OUTPUT_LENGTHS),
                    "records": [
                        {
                            "name": "fixture",
                            "prompt_token_ids": [10, 11],
                            "outputs": {"1": [0], "256": list(range(256))},
                        }
                    ],
                }
                path = os.path.join(directory, f"{arm}-{run_id}.json")
                write_new(path, artifact)
                paths[arm].append(path)
        args = argparse.Namespace(left=paths["left"], right=paths["right"], output=None)
        compare(args)
        broken = load_json(paths["right"][2])
        broken["records"][0]["outputs"]["256"][17] = 999
        os.unlink(paths["right"][2])
        write_new(paths["right"][2], broken)
        try:
            compare(args)
        except GateError as error:
            if "first divergence at output 17" not in str(error):
                raise
        else:
            raise GateError("self-test failed to detect an output divergence")
    print("PASS self-test")
    return 0


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-corpus")
    prepare.add_argument("--base-url", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--target-tokens", nargs="+", type=int, default=[32, 1024, 1025, 8192]
    )
    prepare.add_argument("--timeout", type=float, default=1800)
    prepare.add_argument("--execute", action="store_true")
    prepare.set_defaults(function=prepare_corpus)

    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--base-url", required=True)
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument("--arm", required=True)
    capture_parser.add_argument("--run-id", required=True, choices=(1, 2, 3), type=int)
    capture_parser.add_argument("--corpus", required=True)
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--timeout", type=float, default=1800)
    capture_parser.add_argument("--execute", action="store_true")
    capture_parser.set_defaults(function=capture)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--left", nargs=3, required=True)
    compare_parser.add_argument("--right", nargs=3, required=True)
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(function=compare)

    test = commands.add_parser("self-test")
    test.set_defaults(function=selftest)
    return top


def main():
    args = parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, OSError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
