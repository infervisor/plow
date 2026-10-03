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
import argparse, hashlib, json, struct, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

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


def exact_prompt(tokenize, length, depth, sentence, question, prefix="", ending=""):
    prefix_ids = tokenize(prefix) if prefix else []
    filler = tokenize(FILLER)
    needle = tokenize(" " + sentence + "\n\n")
    suffix = tokenize(question + ending)
    padding = length - len(prefix_ids) - len(needle) - len(suffix)
    if not filler or padding < 0 or not 0 <= depth <= 1:
        raise ValueError("invalid exact-length needle geometry")
    repeated = (filler * ((padding + len(filler) - 1) // len(filler)))[:padding]
    split = int(padding * depth)
    return prefix_ids + repeated[:split] + needle + repeated[split:] + suffix


def chat_frame(hf_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True)
    marker = "PLOW_NEEDLE_CONTENT"
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": marker}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    if rendered.count(marker) != 1:
        raise ValueError("chat template must retain one user content marker")
    return rendered.split(marker)


def compare_captures(reference, candidate):
    if reference.get("prompt_format", "raw") != candidate.get("prompt_format", "raw"):
        raise ValueError("capture prompt formats differ")
    for field in ("concurrency", "repeats", "max_tokens", "ignore_eos", "lengths", "depths"):
        if field not in reference or reference[field] != candidate.get(field):
            raise ValueError(f"capture settings differ or are missing: {field}")

    def index(capture):
        cells = {}
        for c in capture["cells"]:
            key = (c["tokens"], c["depth"], c["item"], c["repeat"])
            digest = c.get("prompt_sha256_u32le")
            if key in cells or not digest or c["prompt_tokens"] != c["tokens"]:
                raise ValueError("duplicate cell or missing exact prompt identity")
            if not c.get("expect") or not isinstance(c.get("text"), str):
                raise ValueError("missing retrieval target or response")
            cells[key] = c
        if not cells:
            raise ValueError("empty retrieval capture")
        return cells

    ref, cand = index(reference), index(candidate)
    expected = {(n, d, "%s@%.1f" % (nid, d), repeat)
                for n in reference["lengths"] for d in reference["depths"]
                for nid, *_ in NEEDLES for repeat in range(reference["repeats"])}
    if ref.keys() != cand.keys() or ref.keys() != expected:
        raise ValueError("retrieval cell sets differ or are incomplete")
    regressions, improvements = [], []
    reference_correct = candidate_correct = 0
    for key, a in ref.items():
        b = cand[key]
        if a["prompt_sha256_u32le"] != b["prompt_sha256_u32le"] or a["expect"] != b["expect"]:
            raise ValueError(f"retrieval prompts or targets differ: {key}")
        left, right = a["expect"] in a["text"], b["expect"] in b["text"]
        reference_correct += left
        candidate_correct += right
        if left and not right:
            regressions.append(key)
        elif right and not left:
            improvements.append(key)
    return dict(cells=len(ref), unique_prompts=len({c["prompt_sha256_u32le"] for c in ref.values()}),
                reference_correct=reference_correct, candidate_correct=candidate_correct,
                regressions=regressions, improvements=improvements,
                verdict="fail" if regressions else ("pass" if reference_correct else "inconclusive"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--arm")
    ap.add_argument("--compare", nargs=2, metavar=("REFERENCE", "CANDIDATE"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", default="4096,16384,30000")
    ap.add_argument("--depths", default="0.1,0.5,0.9",
                    help="fraction of the filler BEFORE the needle")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--exact-lengths", action="store_true", help="use /tokenize and explicit token-ID prompts")
    ap.add_argument("--chat-template-hf", help="local checkpoint chat template; requires --exact-lengths")
    a = ap.parse_args()
    if a.compare:
        captures = []
        for path in a.compare:
            with open(path) as source:
                captures.append(json.load(source))
        result = compare_captures(*captures)
        with open(a.out, "w") as out:
            json.dump(result, out, indent=2)
        print(json.dumps(result))
        return {"pass": 0, "fail": 1, "inconclusive": 2}[result["verdict"]]
    if not a.url or not a.arm:
        ap.error("capture requires --url and --arm")
    if a.concurrency < 1 or a.repeats < 1:
        ap.error("concurrency and repeats must be positive")
    if a.chat_template_hf and not a.exact_lengths:
        ap.error("--chat-template-hf requires --exact-lengths")
    prefix, ending = chat_frame(a.chat_template_hf) if a.chat_template_hf else ("", "")

    model = json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
    @lru_cache(maxsize=None)
    def tokenize(text):
        return post(a.url + "/tokenize", {"model": model, "prompt": text,
                                         "add_special_tokens": False})["tokens"]
    rec = {"arm": a.arm, "model": model, "cells": [],
           "prompt_format": ("chat:" + hashlib.sha256((prefix + ending).encode()).hexdigest()
                             if a.chat_template_hf else "raw"),
           "concurrency": a.concurrency, "repeats": a.repeats,
           "max_tokens": a.max_tokens, "ignore_eos": a.ignore_eos,
           "lengths": [int(x) for x in a.lens.split(",")],
           "depths": [float(x) for x in a.depths.split(",")]}
    t0 = time.perf_counter()
    jobs = []
    for n in rec["lengths"]:
        reps = max((n - 128) // FILLER_TOK, 1)
        for d in rec["depths"]:
            head = int(reps * d)
            for nid, sentence, question, expect in NEEDLES:
                if a.chat_template_hf:
                    question = (question.split("\nA:")[0].removeprefix("Q: ")
                                + "\nAnswer with only the requested value.")
                prompt = (FILLER * head) + " " + sentence + " " + (FILLER * (reps - head)) \
                    + "\n\n" + question
                prompt_hash = None
                if a.exact_lengths:
                    prompt = exact_prompt(tokenize, n, d, sentence, question, prefix, ending)
                    prompt_hash = hashlib.sha256(struct.pack(f"<{len(prompt)}I", *prompt)).hexdigest()
                for repeat in range(a.repeats):
                    jobs.append((n, d, nid, expect, prompt, prompt_hash, repeat))

    def capture(job):
        n, d, nid, expect, prompt, prompt_hash, repeat = job
        r = post(a.url + "/v1/completions", {
            "model": model, "prompt": prompt, "add_special_tokens": False,
            "temperature": 0, "max_tokens": a.max_tokens, "ignore_eos": a.ignore_eos})
        if a.exact_lengths and r["usage"]["prompt_tokens"] != n:
            raise ValueError(f"prompt length changed: expected {n}, got {r['usage']['prompt_tokens']}")
        txt = r["choices"][0]["text"]
        ok = expect in txt
        print("  %s %s@%.1f r%d n=%d(%d): %s %r" % (
            a.arm, nid, d, repeat, n, r["usage"]["prompt_tokens"],
            "OK " if ok else "MISS", txt[:48]), flush=True)
        return {"item": "%s@%.1f" % (nid, d), "tokens": n, "depth": d,
                "repeat": repeat, "prompt_tokens": r["usage"]["prompt_tokens"],
                "prompt_sha256_u32le": prompt_hash,
                "correct": ok, "expect": expect, "text": txt, "answer_char": 0}

    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        rec["cells"] = list(pool.map(capture, jobs))
    rec["elapsed_s"] = time.perf_counter() - t0
    with open(a.out, "w") as out:
        json.dump(rec, out, indent=1)
    n_ok = sum(c["correct"] for c in rec["cells"])
    print("wrote %s: %d/%d retrieved in %.0fs" % (a.out, n_ok, len(rec["cells"]), rec["elapsed_s"]))


if __name__ == "__main__":
    sys.exit(main() or 0)
