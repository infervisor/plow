#!/usr/bin/env python3
"""Chunk-policy battery: TTFT and character-identity at EXACT prompt token counts.

One arm per invocation (the arm is whatever env the server on --port was started
with). Three modes:

  ttft   streamed TTFT (first SSE content delta) at each --lens cell, --reps
         INTERLEAVED passes over the whole cell list (never rep-major: that
         measures DVFS drift, not the arm).
  ident  greedy LONG free-form answers for cross-arm character identity. Long on
         purpose -- a short answer diverges within a few tokens and proves
         nothing about the acceptance class.
  facts  long answers that carry a CHECKABLE needle, so a plan change that
         alters wording can be separated from one that alters correctness.

Prompts are built to land on an EXACT token count, verified per cell against the
server's own `usage.prompt_tokens`, so a cell that misses its target fails loudly
instead of being timed at the wrong length.
"""
import argparse, json, time, urllib.request

FILLER = " the"  # one token under the GLM tokenizer


def post(port, body, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def model_id(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def prompt_tokens(port, model, text):
    """Server-reported prompt token count for `text` (1-token completion)."""
    with post(port, {"model": model, "messages": [{"role": "user", "content": text}],
                     "max_tokens": 1, "temperature": 0}) as r:
        return json.load(r)["usage"]["prompt_tokens"]


_CACHE = {}


def exact_prompt(port, model, question, target):
    """`question` padded with FILLER to EXACTLY `target` prompt tokens.

    The template overhead is MEASURED, not assumed, and the result is VERIFIED,
    so a tokenizer that does not treat FILLER as one token fails here instead of
    leaving a cell at the wrong length.
    """
    key = (question, target)
    if key in _CACHE:
        return _CACHE[key]
    n_fill = target - prompt_tokens(port, model, question)
    if n_fill < 0:
        raise SystemExit(f"question is over the {target}-token target")
    got = -1
    for _ in range(8):
        text = question + FILLER * n_fill
        got = prompt_tokens(port, model, text)
        if got == target:
            _CACHE[key] = text
            return text
        n_fill += target - got
    raise SystemExit(f"cannot hit {target} tokens exactly (last {got})")


def ttft_once(port, model, text, max_tokens=4):
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True}
    t0 = time.perf_counter()
    with post(port, body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                return (time.perf_counter() - t0) * 1e3
    raise SystemExit("no content delta in stream")


def answer(port, model, text, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": 0}
    with post(port, body) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d["usage"]


QUESTIONS = {
    "essay":    "Write a detailed six-sentence explanation of how a modern CPU cache hierarchy works.",
    "gold":     "Explain in at least six sentences why gold is chemically unreactive compared with iron.",
    "sky":      "Explain in at least six sentences why the sky is blue at noon and red at sunset.",
    "tides":    "Explain in at least six sentences how the Moon and the Sun together produce ocean tides.",
    "vaccine":  "Explain in at least six sentences how an mRNA vaccine trains the immune system.",
    "compiler": "Explain in at least six sentences what an optimizing compiler does between parsing and code generation.",
}

FACTS = {
    "f_prime":   ("Is 391 a prime number? Explain your reasoning in full, then state its factorization.", "17"),
    "f_capital": ("Name the capital of Australia, then explain in five sentences why it is not Sydney.", "Canberra"),
    "f_planet":  ("Which planet in the Solar System has the strongest surface gravity? Answer, then justify in five sentences.", "Jupiter"),
    "f_sum":     ("Compute 17 times 23 and show the working, then explain in four sentences why the method is valid.", "391"),
    "f_element": ("What is the chemical symbol for tungsten? Answer, then explain in five sentences where the symbol comes from.", "W"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", choices=["ttft", "ident", "facts"], required=True)
    ap.add_argument("--lens", required=True, help="comma-separated EXACT prompt token counts")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--questions", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = model_id(a.port)
    lens = [int(x) for x in a.lens.split(",")]
    rec = {"arm": a.arm, "mode": a.mode, "model": model, "lens": lens, "cells": []}

    if a.mode == "ttft":
        # One neutral question at every length: TTFT must not depend on which
        # words the filler surrounds, only on how many tokens there are.
        base_q = "Summarize the following text in one word."
        texts = {n: exact_prompt(a.port, model, base_q, n) for n in lens}
        ttft_once(a.port, model, texts[lens[0]])  # warm
        samples = {n: [] for n in lens}
        for _ in range(a.reps):        # rep-major OUTER, cell inner = interleaved
            for n in lens:
                samples[n].append(ttft_once(a.port, model, texts[n]))
                print(f"  {a.arm} {n}: {samples[n][-1]:.1f} ms", flush=True)
        for n in lens:
            v = samples[n]
            m = sum(v) / len(v)
            rec["cells"].append({"tokens": n, "ttft_ms": v, "mean_ms": m,
                                 "spread_pct": 100 * (max(v) - min(v)) / m})
    else:
        bank = QUESTIONS if a.mode == "ident" else {k: v[0] for k, v in FACTS.items()}
        keys = [q for q in a.questions.split(",") if q] or list(bank)
        for q in keys:
            for n in lens:
                text = exact_prompt(a.port, model, bank[q], n)
                txt, usage = answer(a.port, model, text, a.max_tokens)
                cell = {"q": q, "tokens": n, "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"], "text": txt}
                if a.mode == "facts":
                    cell["needle"] = FACTS[q][1]
                    cell["needle_present"] = FACTS[q][1] in txt
                rec["cells"].append(cell)
                print(f"  {a.arm} {q}@{n}: {usage['completion_tokens']} tok"
                      + ("" if a.mode == "ident" else f" needle={cell['needle_present']}"), flush=True)

    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
