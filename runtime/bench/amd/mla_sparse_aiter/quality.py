#!/usr/bin/env python3
"""Concurrent GLM retrieval screen. Arguments: server URL, arm label, output JSON.

`--suite tail` (or `all`) adds the short-tail cases: the question sits in a request's final chunk
of 271, 698 or 1464 rows after 8 x 8192 = 65,536 prior rows, the rows `PLOW_AMD_TAIL_SPARSE_CTX`
moves from dense attention over every prior key into the sparse bucket's top-2048 selection. The
18 base cases never produce such a tail: their ~68.8k-token prompts end in a 3,264-row chunk that is
already sparse. Tail prompts are token ids of an exact length built with the server's `/tokenize`,
and a cell whose served prompt length differs fails, so every tail cell provably hits that path.
"""

import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import time
import urllib.request

spec = importlib.util.spec_from_file_location(
    "needle", Path(__file__).resolve().parents[4] / "scripts/glm53_needle_probe.py"
)
needle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(needle)

TAIL_PRIOR = 8 * 8192
# (tail rows, needle depths): the 512 bucket holds the needle inside the tail itself, the 2048
# bucket makes the question rows find it at 10/50/90 % of the prior.
TAIL_SHAPES = ((271, ("tail",)), (698, (.1, .5, .9)), (1464, (.1, .5, .9)))


def base_cases():
    cases = []
    for length in (8000, 100000):
        reps = max((length - 128) // needle.FILLER_TOK, 1)
        for depth in (.1, .5, .9):
            before = int(reps * depth)
            for nid, sentence, question, expected in needle.NEEDLES:
                prompt = (needle.FILLER * before + " " + sentence + " "
                          + needle.FILLER * (reps - before) + "\n\n" + question)
                cases.append(dict(suite="base", length_label=length, depth=depth, needle=nid,
                                  prompt=prompt, expected=expected))
    return cases


def tail_prompt(filler, needle_ids, question_ids, tail, depth):
    total = TAIL_PRIOR + tail
    body = total - len(needle_ids) - len(question_ids)
    stream = (filler * (body // len(filler) + 1))[:body]
    at = body - tail // 4 if depth == "tail" else int(TAIL_PRIOR * depth)
    assert depth != "tail" or at >= TAIL_PRIOR, "tail too short to hold the needle"
    ids = stream[:at] + needle_ids + stream[at:] + question_ids
    assert len(ids) == total
    return ids


def tail_cases(url, model):
    def tokens(text):
        return needle.post(url + "/tokenize", dict(model=model, prompt=text, add_special_tokens=False))["tokens"]

    filler = tokens(needle.FILLER)
    cases = []
    for tail, depths in TAIL_SHAPES:
        for depth in depths:
            for nid, sentence, question, expected in needle.NEEDLES:
                ids = tail_prompt(filler, tokens(" " + sentence + " "), tokens("\n\n" + question), tail, depth)
                cases.append(dict(suite="tail", length_label=len(ids), depth=depth, needle=nid,
                                  prompt=ids, expected=expected, tail_rows=tail))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("arm")
    parser.add_argument("path")
    parser.add_argument("--concurrency", type=int, choices=range(1, 33), default=4)
    parser.add_argument("--suite", choices=("base", "tail", "all"), default="base",
                        help="base = the 18 cases (default); tail = the 21 short-tail cases; all = both")
    args = parser.parse_args()
    url, arm, path = args.url, args.arm, args.path
    model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
    cases = base_cases() if args.suite in ("base", "all") else []
    if args.suite in ("tail", "all"):
        cases += tail_cases(url, model)

    def run(case):
        start = time.perf_counter()
        body = dict(model=model, prompt=case["prompt"], max_tokens=24, temperature=0)
        if case["suite"] == "base":
            body["add_special_tokens"] = False
        result = needle.post(url + "/v1/completions", body)
        output = result["choices"][0]["text"]
        usage = result.get("usage") or {}
        passed = case["expected"].lower() in output.lower()
        record = dict(
            suite=case["suite"], length_label=case["length_label"], depth=case["depth"],
            needle=case["needle"], expected=case["expected"], text=output, passed=passed,
            usage=usage, seconds=time.perf_counter() - start,
        )
        if case["suite"] == "tail":
            record["tail_rows"] = case["tail_rows"]
            record["length_ok"] = usage.get("prompt_tokens") == case["length_label"]
            record["passed"] = passed and record["length_ok"]
        return record

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, case) for case in cases]):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
            Path(path).write_text(json.dumps(dict(arm=arm, concurrency=args.concurrency, suite=args.suite,
                                                  cells=records), indent=2) + "\n")
    assert all(record["passed"] for record in records), "retrieval failures"


if __name__ == "__main__":
    main()
