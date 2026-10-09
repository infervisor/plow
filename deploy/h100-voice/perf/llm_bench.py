#!/usr/bin/env python3
"""Closed-loop load on /v1/completions (Gemma-4 E4B): TTFT, TPOT and throughput per concurrency.

  python perf/llm_bench.py --conc 1,16,64 --n 256 --isl 1000 --osl 128 [--out results/llm.json]

Each request is a unique random token-id prompt of exactly --isl tokens (starting with <bos>), so
the prefix cache cannot skip prefill, generating exactly --osl tokens (ignore_eos), streamed.
This is the shape of the BASELINE.md E4B rows (vllm bench serve, random ISL 1000 / OSL 128,
c1 = 16 prompts, c64 = 256 prompts); numbers from the two clients are close but not identical.
TPOT per request = (end - first token) / (output tokens - 1).
"""
import argparse
import asyncio
import json
import os
import random
import time

import aiohttp

BOS = 2


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


async def one(s, a, prompt):
    body = {"model": a.model, "prompt": prompt, "max_tokens": a.osl, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    first, chunks, usage = None, 0, None
    async with s.post(a.url + "/v1/completions", json=body) as r:
        if r.status != 200:
            return dict(error=f"{r.status} {(await r.text())[:200]}")
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if "error" in ev:
                return dict(error=str(ev["error"])[:200])
            if ev.get("usage"):
                usage = ev["usage"]
            if any(c.get("text") for c in ev.get("choices", [])):
                first = first or time.perf_counter()
                chunks += 1
    end = time.perf_counter()
    out = (usage or {}).get("completion_tokens") or chunks
    inp = (usage or {}).get("prompt_tokens") or len(prompt)
    return dict(ttft=(first or end) - t0, e2e=end - t0, out=out, inp=inp,
                tpot=(end - first) / (out - 1) if first and out > 1 else None)


async def level(a, conc, n, seed):
    rng = random.Random(seed)
    prompts = [[BOS] + [rng.randrange(1000, 250000) for _ in range(a.isl - 1)] for _ in range(n)]
    auth = {"Authorization": f"Bearer {os.environ['PLOW_API_KEY']}"} if os.environ.get("PLOW_API_KEY") else {}
    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession(headers=auth, timeout=aiohttp.ClientTimeout(total=None, sock_read=600),
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        async def run(p):
            async with sem:
                return await one(s, a, p)
        await one(s, a, [BOS] + [rng.randrange(1000, 250000) for _ in range(31)])  # warm-up
        t0 = time.perf_counter()
        res = await asyncio.gather(*(run(p) for p in prompts))
        wall = time.perf_counter() - t0
    ok = [r for r in res if "error" not in r]
    errs = [r["error"] for r in res if "error" in r]
    ms = lambda v: None if v is None else round(1000 * v, 2)  # noqa: E731
    tt = [r["ttft"] for r in ok]
    tp = [r["tpot"] for r in ok if r["tpot"] is not None]
    return dict(model=a.model, conc=conc, requests=n, errors=len(errs), first_error=errs[0] if errs else None,
                isl=a.isl, osl=a.osl, wall_s=round(wall, 2),
                output_tok_s=round(sum(r["out"] for r in ok) / wall, 1),
                total_tok_s=round(sum(r["out"] + r["inp"] for r in ok) / wall, 1),
                ttft_p50_ms=ms(pct(tt, .5)), ttft_p90_ms=ms(pct(tt, .9)), ttft_p99_ms=ms(pct(tt, .99)),
                tpot_p50_ms=ms(pct(tp, .5)), tpot_p90_ms=ms(pct(tp, .9)), tpot_p99_ms=ms(pct(tp, .99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("PLOW_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default="gemma-4-e4b")
    ap.add_argument("--conc", default="1,16,64")
    ap.add_argument("--n", type=int, default=0, help="requests per level (default: max(16, 4 x conc))")
    ap.add_argument("--isl", type=int, default=1000)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="append one JSON line per level")
    a = ap.parse_args()
    rows = []
    for c in map(int, a.conc.split(",")):
        r = asyncio.run(level(a, c, a.n or max(16, 4 * c), a.seed * 7919 + c))
        rows.append(r)
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    raise SystemExit(1 if any(r["errors"] for r in rows) else 0)


if __name__ == "__main__":
    main()
