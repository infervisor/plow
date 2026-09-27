#!/usr/bin/env python3
"""Second-turn TTFT with and without `X-Session-Id` on /v1/completions (greedy).

session_turns.py --url http://127.0.0.1:PORT --model M [--prompt-tokens 1200] [--trials 8]

Each trial: turn 1 = a long prompt, turn 2 = turn 1's prompt + its answer + a follow-up, streamed.
Arm `session` sends both turns in one session (turn 2 resumes the retained rows), arm `plain` sends
none. Prints TTFT medians, the turn-2 X-Session-Cache headers, and whether every turn-2 text
matches between the arms.
"""
import argparse, json, os, statistics, time, urllib.request, uuid

WORDS = ("the quick brown fox jumps over the lazy dog while a gentle breeze moves through the tall "
         "grass near the river and the children laugh as they play in the warm afternoon sun").split()


def call(url, body, headers):
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    t0 = time.perf_counter()
    ttft, text = None, []
    with urllib.request.urlopen(req, timeout=300) as r:
        cache = r.headers.get("X-Session-Cache"), r.headers.get("X-Session-Cached-Tokens")
        if not body.get("stream"):
            return None, json.loads(r.read())["choices"][0]["text"], cache
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices") and d["choices"][0].get("text"):
                ttft = ttft or time.perf_counter() - t0
                text.append(d["choices"][0]["text"])
    return ttft, "".join(text), cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-words", type=int, default=900)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--first-tokens", type=int, default=None, help="turn 1's max_tokens (default --max-tokens)")
    args = ap.parse_args()
    out = {}
    for arm in ("plain", "session", "plain2"):
        ttfts, texts, caches = [], [], []
        for t in range(args.trials):
            prompt = " ".join(WORDS[(i * 7 + t) % len(WORDS)] for i in range(args.prompt_words))
            h = {"X-Session-Id": f"turns-{uuid.uuid4()}"} if arm == "session" else {}
            base = dict(model=args.model, max_tokens=args.max_tokens, temperature=0)
            _, a1, _ = call(args.url, dict(base, prompt=prompt, max_tokens=args.first_tokens or args.max_tokens), h)
            ttft, a2, cache = call(args.url, dict(base, prompt=prompt + a1 + " and then", stream=True), h)
            ttfts.append(ttft)
            texts.append(a2)
            caches.append(cache)
        out[arm] = texts
        print(json.dumps(dict(arm=arm, ttft_med_ms=round(1e3 * statistics.median(ttfts), 2),
                              ttft_ms=[round(1e3 * x, 2) for x in ttfts], cache=caches)), flush=True)
    same = [a == b for a, b in zip(out["plain"], out["session"])]
    shared = [len(os.path.commonprefix([a, b])) for a, b in zip(out["plain"], out["session"])]
    print(json.dumps(dict(identical=sum(same), trials=len(same), shared_chars=shared,
                          text_chars=[len(a) for a in out["plain"]],
                          plain_repeatable=sum(a == b for a, b in zip(out["plain"], out["plain2"])))), flush=True)


if __name__ == "__main__":
    main()
