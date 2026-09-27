"""Concurrent streaming probe: when does each SSE chunk reach the client?

  python stream_probe.py <port> [concurrency] [max_tokens] [prompt_tokens]

Prints, per request, the client-side time of the first and last chunk and the chunk count, so
server-side generation timing (the server log) can be compared with delivery timing.
"""
import asyncio
import json
import sys
import time

import aiohttp

port = int(sys.argv[1])
conc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32
n_prompt = int(sys.argv[4]) if len(sys.argv) > 4 else 256
text = len(sys.argv) > 5 and sys.argv[5] == "text"  # tokenized by the server, as vllm bench does


async def one(session, i, t0):
    body = {
        "model": "m",
        "prompt": " ".join(f"w{(i * 7919 + k * 31) % 997}" for k in range(n_prompt)) if text
        else [100 + (i * 7919 + k * 31) % 50000 for k in range(n_prompt)],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    times = []
    async with session.post(f"http://127.0.0.1:{port}/v1/completions", json=body) as r:
        buf = b""
        async for chunk in r.content.iter_any():
            now = time.perf_counter() - t0
            buf += chunk
            while b"\n\n" in buf:
                msg, buf = buf.split(b"\n\n", 1)
                msg = msg.strip()
                if not msg.startswith(b"data: "):
                    continue
                payload = msg[6:]
                if payload == b"[DONE]":
                    continue
                d = json.loads(payload)
                if d.get("choices"):
                    times.append(now)
    return i, times


async def main():
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        res = await asyncio.gather(*[one(s, i, t0) for i in range(conc)])
    for i, t in sorted(res):
        gaps = [b - a for a, b in zip(t, t[1:])]
        print(f"req {i}: chunks={len(t)} first={t[0]:.2f}s last={t[-1]:.2f}s "
              f"max_gap={max(gaps) if gaps else 0:.2f}s median_gap={sorted(gaps)[len(gaps)//2] if gaps else 0:.3f}s")


asyncio.run(main())
