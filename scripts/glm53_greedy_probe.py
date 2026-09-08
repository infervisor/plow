#!/usr/bin/env python3
"""Greedy token streams at fixed contexts, for comparing two decode arms.

`perf-data/probes/facts_gate.py` is the QUALITY instrument (machine-checkable
answers, paired McNemar) and stays the right tool for "is the candidate worse".
It cannot serve as the IDENTITY instrument for GLM-5.3, because that checkpoint
is a reasoning model: its `message.content` is empty until the reasoning budget
is spent, so a 288-token chat cell records an empty string and every arm
"agrees" trivially.

This probe therefore drives `/v1/completions` — raw text continuation, one
stream, no reasoning/content split — at temperature 0 with `ignore_eos`, so both
arms produce exactly `--max-tokens` tokens from exactly the same prompt and the
comparison is over the whole stream. Prompts are padded with a one-token filler
to hit a target context, and the ACHIEVED `prompt_tokens` is recorded rather
than asserted: what matters is that both arms see the same bytes, not that the
length is round.
"""
import argparse, json, sys, time, urllib.request

# Padding is PROSE, not a repeated single token. A 16k-token run of " the" drives
# GLM-5.3 into degenerate output on every arm (measured: both arms emit the same
# ' const class1.题目' loop), which agrees trivially and tests nothing. A repeated
# natural paragraph keeps the continuation coherent, so a real divergence is
# visible as a real divergence. `--filler-token` keeps the old behaviour for the
# rare case where an exact token count matters more than coherence.
FILLER_PROSE = (
    "The engine keeps one persistent dispatch per rank and advances every sequence "
    "through the same instruction stream, so a token costs one launch and nothing "
    "else. Weights are carved from a single allocation, the key/value cache is a "
    "ring addressed by position, and the collective seams are folded into the "
    "packets that produce their operands. "
)
FILLER_TOKEN = " the"  # one token under the GLM tokenizer

# Four openings with different subject matter, so a divergence in one is not a
# property of a single prompt. Each is continued, not answered — the completions
# endpoint has no chat template, which is exactly what makes the stream clean.
SEEDS = [
    "The following is a technical note on memory bandwidth in GPU inference.\n\n"
    "Decode-time attention reads the key/value cache once per token, so",
    "A short history of the printing press.\n\n"
    "Movable type reached Europe in the fifteenth century, and the",
    "Recipe notes: sourdough starter maintenance.\n\n"
    "A mature starter doubles within six hours at room temperature, which",
    "Notes on the Peloponnesian War.\n\n"
    "Thucydides argues that the truest cause of the conflict was",
]


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", default="1024,4096,8192,16384",
                    help="target prompt token counts")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--filler-token", action="store_true",
                    help='pad with the one-token " the" instead of prose')
    a = ap.parse_args()

    model = json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
    rec = {"arm": a.arm, "model": model, "max_tokens": a.max_tokens, "cells": []}
    t0 = time.perf_counter()
    for n in [int(x) for x in a.lens.split(",")]:
        for i, seed in enumerate(SEEDS):
            # The filler goes FIRST so the seed stays adjacent to the generated
            # text: a continuation that has to reach back 16k tokens for its
            # topic is a retrieval test, not a numerics test.
            pad = max(n - 64, 0)
            if a.filler_token:
                prompt = FILLER_TOKEN * pad + "\n\n" + seed
            else:
                # ~90 tokens per repeat under the GLM tokenizer; the achieved
                # `prompt_tokens` is what gets recorded, so the estimate only has
                # to be close.
                prompt = FILLER_PROSE * max(pad // 90, 1) + "\n\n" + seed
            d = post(a.url + "/v1/completions", {
                "model": model, "prompt": prompt, "add_special_tokens": False,
                "temperature": 0, "max_tokens": a.max_tokens, "ignore_eos": True})
            ch = d["choices"][0]
            rec["cells"].append({
                "item": f"seed{i}", "tokens": n,
                "prompt_tokens": d["usage"]["prompt_tokens"],
                "completion_tokens": d["usage"]["completion_tokens"],
                "finish_reason": ch.get("finish_reason"),
                "answer_char": 0, "correct": True,
                "text": ch["text"]})
            print(f"  {a.arm} seed{i}@{n}: prompt_tokens="
                  f"{d['usage']['prompt_tokens']} {ch['text'][:56]!r}", flush=True)
    rec["elapsed_s"] = time.perf_counter() - t0
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(rec['cells'])} cells in {rec['elapsed_s']:.0f}s")


if __name__ == "__main__":
    sys.exit(main() or 0)
