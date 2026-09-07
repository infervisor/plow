#!/usr/bin/env python3
"""Gemma 4 text-only serving comparison; verify_packed_serve.py checks correctness."""
import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--inputs", type=int, nargs="+", default=[128, 1024, 4096])
parser.add_argument("--outputs", type=int, nargs="+", default=[64])
parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--warmups", type=int, default=1)
parser.add_argument("--manifest", help="JSON recording checkpoint, precision, GPU and server settings")
args = parser.parse_args()
if min(args.inputs + args.outputs + args.concurrency + [args.repeats]) < 1 or args.warmups < 0:
    parser.error("workload sizes and repeats must be positive; warmups must be nonnegative")
model = json.load(urllib.request.urlopen(args.url + "/v1/models"))["data"][0]["id"]
manifest = json.load(open(args.manifest)) if args.manifest else None


def quantiles(values):
    values = sorted(values)
    def percentile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99)}

def request(length, output, seed):
    prompt = (" " + ["hello", "world", "test", "data"][seed % 4]) * length
    body = {"model": model, "prompt": prompt, "add_special_tokens": False, "temperature": 0,
            "max_tokens": output, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(args.url + "/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    usage = None
    text = []
    arrivals = []
    done = False
    finish_reason = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if line.strip() == b"data: [DONE]":
                done = True
                break
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("text", "")
                if delta:
                    now = time.perf_counter()
                    arrivals.append((now - start) * 1000)
                    if first is None:
                        first = now
                finish_reason = choice.get("finish_reason") or finish_reason
                text.append(delta)
    end = time.perf_counter()
    assert usage and usage["prompt_tokens"] == length, usage
    assert usage["completion_tokens"] == output, usage
    assert done and first is not None and finish_reason == "length", (done, finish_reason)
    gaps = [b-a for a, b in zip(arrivals, arrivals[1:])]
    return {"ttft_ms": (first-start)*1000, "latency_ms": (end-start)*1000,
            "tpot_ms": (end-first)*1000/(output-1) if output > 1 else None,
            "text_chunk_arrival_ms": arrivals,
            "text_chunk_gap_ms": quantiles(gaps) if gaps else None,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "usage": usage, "text": "".join(text)}

with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.concurrency)) as pool, open(args.out, "w") as log:
    for length in args.inputs:
        for output in args.outputs:
            for concurrency in args.concurrency:
                for repeat in range(-args.warmups, args.repeats):
                    start = time.perf_counter()
                    futures = [pool.submit(request, length, output, i) for i in range(concurrency)]
                    results = [f.result() for f in futures]
                    elapsed = time.perf_counter()-start
                    if repeat < 0:
                        continue
                    result = {"schema_version": 2, "label": args.label, "model": model,
                        "manifest": manifest, "warmups_per_case": args.warmups, "input": length,
                        "output": output, "concurrency": concurrency, "repeat": repeat,
                        "elapsed_s": elapsed, "output_tok_s": output*concurrency/elapsed,
                        "request_s": concurrency/elapsed,
                        "ttft_ms": statistics.median(r["ttft_ms"] for r in results),
                        "latency_ms": quantiles([r["latency_ms"] for r in results]),
                        "tpot_ms": quantiles([r["tpot_ms"] for r in results]) if output > 1 else None,
                        "requests": results}
                    log.write(json.dumps(result)+"\n")
                    log.flush()
                    print({k:v for k,v in result.items() if k not in ("requests", "manifest")}, flush=True)
