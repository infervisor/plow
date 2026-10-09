#!/usr/bin/env python3
"""Gemma-4 E4B text: /v1/chat/completions (streamed or not, logprobs) and raw /v1/completions.

  python clients/chat.py "What is the capital of France?" [--stream] [--logprobs 3] [--max-tokens 64]
  python clients/chat.py --raw "<bos>The capital of France is" [--max-tokens 8]

Raw completions take the prompt verbatim: Gemma prompts must start with "<bos>" (the server adds
no BOS). Chat requests get the checkpoint's chat template, BOS included.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import request, sse_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default="gemma-4-e4b")
    ap.add_argument("--system", default=None)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--raw", action="store_true", help="POST /v1/completions with the prompt verbatim")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--logprobs", type=int, default=0, help="top-N logprobs per token (0 = off)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.raw:
        body = {"model": a.model, "prompt": a.prompt, "max_tokens": a.max_tokens, "temperature": a.temperature}
        if a.logprobs:
            body["logprobs"] = a.logprobs
        print(json.dumps(json.loads(request("POST", "/v1/completions", body).read()), indent=1, ensure_ascii=False))
        return
    messages = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
    body = {"model": a.model, "messages": messages, "max_tokens": a.max_tokens, "temperature": a.temperature,
            "stream": a.stream}
    if a.logprobs:
        body.update(logprobs=True, top_logprobs=a.logprobs)
    resp = request("POST", "/v1/chat/completions", body)
    if not a.stream:
        reply = json.loads(resp.read())
        print(json.dumps(reply, indent=1, ensure_ascii=False))
        print(f"latency {1000 * (time.perf_counter() - t0):.0f} ms", file=sys.stderr)
        return
    first, n = None, 0
    for ev in sse_events(resp):
        if "error" in ev:
            raise SystemExit(f"error: {ev['error']}")
        for ch in ev.get("choices", []):
            delta = ch.get("delta", {}).get("content")
            if delta:
                first = first or time.perf_counter() - t0
                n += 1
                print(delta, end="", flush=True)
            if ch.get("finish_reason"):
                print(f"\n[finish_reason={ch['finish_reason']}]")
        if ev.get("usage"):
            print(f"usage: {ev['usage']}")
    total = time.perf_counter() - t0
    tpot = (total - first) / max(n - 1, 1) if first else 0
    print(f"TTFT {1000 * (first or total):.0f} ms, {n} chunks, ~{1000 * tpot:.1f} ms per chunk, total {total:.2f} s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
