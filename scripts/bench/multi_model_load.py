#!/usr/bin/env python3
"""Parallel multi-model load test for plowrt.

Sends concurrent chat-completion requests spread across several models served by ONE plowrt
process (multi-model registry), and reports per-model + overall throughput, latency (TTFT/TPOT),
and success rate. Exercises plow's co-tenant scheduling: many streams, multiple models, one device.

Usage:
  python3 scripts/bench/multi_model_load.py --url http://127.0.0.1:8200 \
      --models qwen3-0.6b smollm2-360m-instruct --concurrency 8 --requests 48 --max-tokens 96

Pure stdlib (no deps), streaming SSE so TTFT/TPOT are real.
"""
import argparse, json, time, threading, urllib.request, statistics as st
from collections import defaultdict

PROMPTS = [
    "Explain how a bicycle stays upright, briefly.",
    "Write two sentences about the ocean.",
    "List three uses for a paperclip.",
    "What is the capital of Japan? One short sentence.",
    "Summarize why sleep matters in one sentence.",
    "Give a one-line tip for learning to code.",
]

def stream(url, model, prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; ntok = 0; ok = True
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    d = json.loads(data)["choices"][0]["delta"].get("content")
                except Exception:
                    continue
                if d:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
    except Exception:
        ok = False
    return {"model": model, "ok": ok, "ttft": ttft, "ntok": ntok, "wall": time.perf_counter() - t0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=96)
    a = ap.parse_args()

    # Round-robin the request list across models so both are hit concurrently.
    jobs = [(a.models[i % len(a.models)], PROMPTS[i % len(PROMPTS)]) for i in range(a.requests)]
    results = []; lock = threading.Lock(); sem = threading.Semaphore(a.concurrency)

    def work(model, prompt):
        with sem:
            r = stream(a.url, model, prompt, a.max_tokens)
        with lock:
            results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=work, args=j) for j in jobs]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0

    by = defaultdict(list)
    for r in results: by[r["model"]].append(r)
    total_tok = sum(r["ntok"] for r in results)
    fails = sum(1 for r in results if not r["ok"])
    print(f"\n=== plow multi-model load: {a.requests} reqs, concurrency {a.concurrency}, "
          f"{len(a.models)} models ===")
    print(f"wall={wall:.2f}s  aggregate={total_tok/wall:.1f} tok/s  "
          f"success={a.requests-fails}/{a.requests}")
    for m in a.models:
        rs = [r for r in by[m] if r["ok"] and r["ttft"]]
        if not rs:
            print(f"  {m}: no successful responses"); continue
        ttfts = sorted(r["ttft"] for r in rs)
        decs = sorted((r["ntok"] - 1) / (r["wall"] - r["ttft"]) for r in rs
                      if r["wall"] > r["ttft"] and r["ntok"] > 1)
        toks = sum(r["ntok"] for r in rs)
        print(f"  {m:26s}: reqs={len(rs):3d}  toks={toks:5d}  "
              f"TTFT p50={ttfts[len(ttfts)//2]*1000:6.0f}ms p95={ttfts[int(len(ttfts)*0.95)]*1000:6.0f}ms  "
              f"per-stream decode med={st.median(decs) if decs else 0:5.1f} tok/s")

if __name__ == "__main__":
    main()
