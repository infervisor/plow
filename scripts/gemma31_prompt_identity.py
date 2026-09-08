#!/usr/bin/env python3
"""Greedy character-identity corpus with SEVERAL prompts per length at EVERY concurrency.

`bench_packed_serve.py` derives its prompt from the request index, so its concurrency-1 cells
carry exactly one prompt per length — enough for a throughput median, not enough for an identity
gate that has to hold "3 prompts x 64 tokens at 128 / 2048 / 8192, concurrency 1 and 4".

Writes the same record shape `gemma31_bench_identity.py` reads, so the two corpora go through
one comparator. Prompts are the same function of the seed as in `bench_packed_serve.py`, so a
seed means the same text in both.

    python3 scripts/gemma31_prompt_identity.py --url http://127.0.0.1:8000 \\
        --out ident.json --inputs 128 2048 8192 --concurrency 1 4 --prompts 4
"""
import argparse
import concurrent.futures
import json
import time
import urllib.request

WORDS = ["hello", "world", "test", "data"]

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--out", required=True)
p.add_argument("--label", default="identity")
p.add_argument("--inputs", type=int, nargs="+", default=[128, 2048, 8192])
p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
p.add_argument("--prompts", type=int, default=4, help="distinct prompts per (length, concurrency)")
p.add_argument("--tokens", type=int, default=64)
p.add_argument("--repeats", type=int, default=2)
a = p.parse_args()
if a.prompts < 1 or a.tokens < 2 or a.repeats < 1:
    p.error("--prompts/--repeats must be >= 1 and --tokens >= 2")

model = json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]


def request(length, seed):
    prompt = (" " + WORDS[seed % len(WORDS)]) * length
    body = {"model": model, "prompt": prompt, "add_special_tokens": False, "temperature": 0,
            "max_tokens": a.tokens, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(a.url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    usage = r["usage"]
    assert usage["prompt_tokens"] == length, usage
    assert usage["completion_tokens"] == a.tokens, usage
    return {"text": r["choices"][0]["text"], "latency_ms": (time.perf_counter() - start) * 1000,
            "seed": seed}


with concurrent.futures.ThreadPoolExecutor(max_workers=max(a.concurrency)) as pool, \
        open(a.out, "w") as log:
    for length in a.inputs:
        for conc in a.concurrency:
            for repeat in range(a.repeats):
                results = []
                # Issued in waves of `conc`, so a concurrency-4 wave really is four requests
                # scheduled together — the property the identity gate is about.
                for lo in range(0, a.prompts, conc):
                    seeds = range(lo, min(lo + conc, a.prompts))
                    results += [f.result()
                                for f in [pool.submit(request, length, s) for s in seeds]]
                rec = {"schema_version": 2, "label": a.label, "model": model, "manifest": None,
                       "warmups_per_case": 0, "input": length, "output": a.tokens,
                       "concurrency": conc, "repeat": repeat, "elapsed_s": 0.0,
                       "output_tok_s": 0.0, "request_s": 0.0, "ttft_ms": 0.0, "tpot_ms": None,
                       "requests": results}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"{length:>6} tok / conc {conc} / repeat {repeat}: "
                      f"{len(results)} completions", flush=True)
