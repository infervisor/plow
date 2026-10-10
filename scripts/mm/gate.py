#!/usr/bin/env python3
"""E2E multimodal gate: replay `scripts/mm/hf_ref.py all` chat cases against a plowrt server.

Per case: usage.prompt_tokens must equal HF's, and the greedy completion must match HF's for at
least --min-prefix words of the shorter text, or fork from it only at a near tie (plow's top-2
logprob margin <= --tie-margin there, and plow's runner-up is HF's token).

Each case runs --repeats times. The prefix cache publishes a prompt from its second sighting, so
the last run is a cache hit while the first prefilled cold. Runs with the same prefill split are
bit-identical; a different split (cold vs cached) changes bf16 rounding (|dlogprob| ~0.1), so runs
must agree except at such a near tie. Cases sharing text but not media must differ from each
other exactly as HF's do, which catches media-blind prefix-cache keys.
"""
import argparse, json, sys, urllib.request


def chat(base, model, messages, max_new):
    body = {"model": model, "messages": messages, "max_tokens": max_new, "temperature": 0,
            "logprobs": True, "top_logprobs": 2}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        o = json.load(r)
    ch = o["choices"][0]
    lps = (ch.get("logprobs") or {}).get("content") or []
    toks = [(e["token"], [(t["token"], t["logprob"]) for t in e.get("top_logprobs", [])]) for e in lps]
    cached = (o["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return ch["message"]["content"] or "", o["usage"]["prompt_tokens"], toks, cached


def margin(top):
    return top[0][1] - top[1][1] if len(top) > 1 else float("inf")


def run_fork(a, b, tie):
    """(first differing token index or None, whether the fork is a near tie in either run)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x[0] != y[0]:
            return i, min(margin(x[1]), margin(y[1])) <= tie
    return (None, True) if len(a) == len(b) else (min(len(a), len(b)), False)


def hf_tie_fork(toks, hf, tie):
    """Whether plow's text leaves HF's at a near tie where plow's runner-up continues HF's text."""
    acc = ""
    for tok, top in toks:
        if hf.startswith(acc + tok):
            acc += tok
            continue
        alts = [t for t, _ in top[1:]]
        return margin(top) <= tie and any(hf.startswith(acc + t) or (acc + t).startswith(hf) for t in alts)
    return True


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
    ap.add_argument("--tie-margin", type=float, default=0.25, help="top-2 logprob gap counted as a near tie")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    ref = json.load(open(args.ref))
    ok, got = True, {}
    for c in ref["cases"]:
        runs = [chat(args.base, args.model, c["messages"], args.max_new) for _ in range(max(1, args.repeats))]
        text, ptok, toks, _ = runs[0]
        forks = [run_fork(toks, r[2], args.tie_margin) for r in runs[1:]]
        stable = all(tie for _, tie in forks)
        n, m = common_words(text, c["output"])
        exact = text.strip() == c["output"].strip()
        hf_tie = not exact and n < min(args.min_prefix, m) and hf_tie_fork(toks, c["output"], args.tie_margin)
        passed = ptok == c["prompt_tokens"] and stable and (exact or n >= min(args.min_prefix, m) or hf_tie)
        ok &= passed
        got[c["name"]] = text
        print(json.dumps({"case": c["name"], "pass": passed, "exact": exact, "prompt_tokens": ptok,
                          "hf_prompt_tokens": c["prompt_tokens"], "common_words": n, "hf_tie_fork": hf_tie,
                          "repeat_equal": all(r[0] == text for r in runs[1:]), "repeat_forks": [f[0] for f in forks],
                          "repeat_stable": stable, "cached_tokens": [r[3] for r in runs],
                          "plow": text[:160], "hf": c["output"][:160]}), flush=True)
    by_text = {}
    for c in ref["cases"]:
        by_text.setdefault(c["messages"][0]["content"][-1]["text"], []).append(c)
    for group in by_text.values():
        for a in group:
            for b in group:
                if a["name"] < b["name"] and a["media"] != b["media"] and a["output"] != b["output"] and got[a["name"]] == got[b["name"]]:
                    ok = False
                    print(json.dumps({"media_collision": [a["name"], b["name"]]}), flush=True)
    print(json.dumps({"gate": "pass" if ok else "fail"}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
