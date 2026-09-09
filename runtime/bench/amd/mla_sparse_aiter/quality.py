#!/usr/bin/env python3
"""Concurrent GLM retrieval screen. Arguments: server URL, arm label, output JSON."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("arm")
    parser.add_argument("path")
    parser.add_argument("--concurrency", type=int, choices=range(1, 33), default=4)
    args = parser.parse_args()
    url, arm, path = args.url, args.arm, args.path
    model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
    cases = []
    for length in (8000, 100000):
        reps = max((length - 128) // needle.FILLER_TOK, 1)
        for depth in (.1, .5, .9):
            before = int(reps * depth)
            for nid, sentence, question, expected in needle.NEEDLES:
                prompt = (needle.FILLER * before + " " + sentence + " "
                          + needle.FILLER * (reps - before) + "\n\n" + question)
                cases.append((length, depth, nid, prompt, expected))

    def run(case):
        length, depth, nid, prompt, expected = case
        start = time.perf_counter()
        result = needle.post(url + "/v1/completions", dict(
            model=model, prompt=prompt, add_special_tokens=False, max_tokens=24, temperature=0,
        ))
        output = result["choices"][0]["text"]
        return dict(
            length_label=length, depth=depth, needle=nid, expected=expected, text=output,
            passed=expected.lower() in output.lower(), usage=result.get("usage"),
            seconds=time.perf_counter() - start,
        )

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, case) for case in cases]):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
            Path(path).write_text(json.dumps(dict(arm=arm, concurrency=args.concurrency, cells=records), indent=2) + "\n")
    assert all(record["passed"] for record in records), "retrieval failures"


if __name__ == "__main__":
    main()
