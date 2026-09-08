#!/usr/bin/env python3
"""Needle retrieval at depth — the quality question greedy agreement cannot answer.

`glm53_greedy_probe.py` measures how far two arms stay token-identical. That is
the right instrument for "did the arithmetic change" and the wrong one for "did
the answer get worse": fp8 KV diverges by construction, so a divergence number on
its own says nothing about whether the model still knows what it read.

This is the other half, and it is aimed at fp8 KV's DOCUMENTED failure mode —
retrieval degrading with context (`flags-reference.md`: at 7.8k every arm finds
the needle, at 66.9k only bf16 does). A unique fact is planted at several depths
inside prose filler, and the model is asked for it through `/v1/completions` (raw
continuation, so GLM-5.3's reasoning channel cannot eat the answer budget the way
it eats a 288-token chat cell). Grading is substring containment of the planted
token, which is crude and deliberately so: a wrong answer here is a wrong answer,
not a rewording.

The verdict is PAIRED. A cell both arms miss is a model limit at that depth, not
a regression of the candidate.
"""
import argparse, json, sys, time, urllib.request

FILLER = (
    "The engine keeps one persistent dispatch per rank and advances every sequence "
    "through the same instruction stream, so a token costs one launch and nothing "
    "else. Weights are carved from a single allocation, the key/value cache is a "
    "ring addressed by position, and the collective seams are folded into the "
    "packets that produce their operands. "
)
FILLER_TOK = 90  # approximate, under the GLM tokenizer; the achieved count is recorded

# (id, planted sentence, the question, the string the answer must contain)
NEEDLES = [
    ("code", "The maintenance access code for the Kestrel substation is 7429-BLUE.",
     "Q: What is the maintenance access code for the Kestrel substation?\nA: The code is",
     "7429"),
    ("city", "Dr. Imelda Vasquez relocated the entire seed archive to Trondheim in 1994.",
     "Q: To which city did Dr. Imelda Vasquez relocate the seed archive?\nA: She relocated it to",
     "Trondheim"),
    ("mass", "The calibration weight used by the Halloran team masses exactly 312 grams.",
     "Q: What is the mass of the calibration weight used by the Halloran team?\nA: It masses",
     "312"),
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
    ap.add_argument("--lens", default="4096,16384,30000")
    ap.add_argument("--depths", default="0.1,0.5,0.9",
                    help="fraction of the filler BEFORE the needle")
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args()

    model = json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
    rec = {"arm": a.arm, "model": model, "cells": []}
    t0 = time.perf_counter()
    for n in [int(x) for x in a.lens.split(",")]:
        reps = max((n - 128) // FILLER_TOK, 1)
        for d in [float(x) for x in a.depths.split(",")]:
            head = int(reps * d)
            for nid, sentence, question, expect in NEEDLES:
                prompt = (FILLER * head) + " " + sentence + " " + (FILLER * (reps - head)) \
                    + "\n\n" + question
                r = post(a.url + "/v1/completions", {
                    "model": model, "prompt": prompt, "add_special_tokens": False,
                    "temperature": 0, "max_tokens": a.max_tokens, "ignore_eos": False})
                txt = r["choices"][0]["text"]
                ok = expect in txt
                rec["cells"].append({
                    "item": "%s@%.1f" % (nid, d), "tokens": n, "depth": d,
                    "prompt_tokens": r["usage"]["prompt_tokens"],
                    "correct": ok, "expect": expect, "text": txt, "answer_char": 0})
                print("  %s %s@%.1f n=%d(%d): %s %r" % (
                    a.arm, nid, d, n, r["usage"]["prompt_tokens"],
                    "OK " if ok else "MISS", txt[:48]), flush=True)
    rec["elapsed_s"] = time.perf_counter() - t0
    json.dump(rec, open(a.out, "w"), indent=1)
    n_ok = sum(c["correct"] for c in rec["cells"])
    print("wrote %s: %d/%d retrieved in %.0fs" % (a.out, n_ok, len(rec["cells"]), rec["elapsed_s"]))


if __name__ == "__main__":
    sys.exit(main() or 0)
