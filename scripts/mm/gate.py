#!/usr/bin/env python3
"""E2E multimodal gate: replay `scripts/mm/hf_ref.py all` chat cases against a plowrt server.

Per case: usage.prompt_tokens must equal HF's, and the greedy completion must match HF's for at
least --min-prefix tokens of the shorter text (bf16 drift may fork late). Each case runs twice; the
second (prefix-cache hit) must equal the first. Cases sharing text but not media must differ from
each other exactly as HF's do, which catches media-blind prefix-cache keys.
"""
import argparse, json, sys, urllib.request


def chat(base, model, messages, max_new):
    body = {"model": model, "messages": messages, "max_tokens": max_new, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        o = json.load(r)
    return o["choices"][0]["message"]["content"] or "", o["usage"]["prompt_tokens"]


def common_words(a, b):
    a, b = a.split(), b.split()
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n, min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--min-prefix", type=int, default=8, help="leading words that must match HF")
    args = ap.parse_args()
    ref = json.load(open(args.ref))
    ok, got = True, {}
    for c in ref["cases"]:
        text, ptok = chat(args.base, args.model, c["messages"], args.max_new)
        again, _ = chat(args.base, args.model, c["messages"], args.max_new)
        n, m = common_words(text, c["text"])
        exact = text.strip() == c["text"].strip()
        passed = ptok == c["prompt_tokens"] and again == text and (exact or n >= min(args.min_prefix, m))
        ok &= passed
        got[c["name"]] = text
        print(json.dumps({"case": c["name"], "pass": passed, "exact": exact, "prompt_tokens": ptok,
                          "hf_prompt_tokens": c["prompt_tokens"], "common_words": n, "repeat_equal": again == text,
                          "plow": text[:160], "hf": c["text"][:160]}), flush=True)
    by_text = {}
    for c in ref["cases"]:
        by_text.setdefault(c["messages"][0]["content"][-1]["text"], []).append(c)
    for group in by_text.values():
        for a in group:
            for b in group:
                if a["name"] < b["name"] and a["media"] != b["media"] and (a["text"] != b["text"]) and got[a["name"]] == got[b["name"]]:
                    ok = False
                    print(json.dumps({"media_collision": [a["name"], b["name"]]}), flush=True)
    print(json.dumps({"gate": "pass" if ok else "fail"}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
