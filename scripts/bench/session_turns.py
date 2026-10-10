#!/usr/bin/env python3
"""Second-turn TTFT with and without `X-Session-Id` on /v1/completions (greedy).

session_turns.py --url http://127.0.0.1:PORT --model M [--prompt-tokens 1200] [--trials 8]

Each trial: turn 1 = a long prompt, turn 2 = turn 1's prompt + its answer + a follow-up, streamed.
Arm `session` sends both turns in one session (turn 2 resumes the retained rows), arm `plain` sends
none. Prints TTFT medians, the turn-2 X-Session-Cache headers, and whether every turn-2 text
matches between the arms.

--turns N: agentic sessions instead. --conversations C chats of N streamed turns, --concurrency K
at a time; each turn appends the previous answer and a unique ~--obs-words observation, so turn n
resends the whole history. Prints per-turn prompt tokens, TTFT and usage cached_tokens.
--dump writes every turn's tokens and logprobs; --compare REF diffs them against a dump from a
cache-off (cold) server: cached and cold greedy logprobs must be identical.
"""
import argparse, json, os, random, statistics, time, urllib.error, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor

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


def stream_turn(url, body, headers):
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    t0 = time.perf_counter()
    ttft, text, tokens, lps, usage = None, [], [], [], None
    try:
        r = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r} (prompt chars {len(body['prompt'])})") from None
    with r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices") or []:
                ttft = ttft or time.perf_counter() - t0
                text.append(c.get("text") or "")
                lp = c.get("logprobs") or {}
                tokens += lp.get("tokens") or []
                lps += lp.get("token_logprobs") or []
    # Prefix-cache sessions (`X-Session-Cache: prefix-cache`) report reuse in usage, not the header.
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return dict(ttft_ms=1e3 * (ttft or 0), text="".join(text), tokens=tokens, logprobs=lps,
                cached=cached, usage=usage)


def conversation(args, c):
    rng = random.Random(args.seed * 1000 + c)
    vocab = WORDS + [f"item{n}" for n in range(200)] + ["error:", "ok", "path", "/src", "{", "}"]
    words = lambda n: " ".join(rng.choice(vocab) for _ in range(n))
    hist = (f"<bos><start_of_turn>system\nYou are a coding agent. Session {c}. {words(args.prompt_words)}"
            f"<end_of_turn>\n<start_of_turn>user\nFix the failing test. {words(40)}<end_of_turn>\n"
            "<start_of_turn>model\n")
    h = {"X-Session-Id": f"agent-{args.seed}-{c}-{uuid.uuid4()}"} if args.session else {}
    out = []
    for t in range(args.turns):
        body = dict(model=args.model, prompt=hist, max_tokens=args.max_tokens, temperature=0,
                    stream=True, logprobs=args.logprobs, stream_options={"include_usage": True})
        r = stream_turn(args.url, body, h)
        r.update(conv=c, turn=t + 1, prompt_chars=len(hist))
        out.append(r)
        hist += (r["text"] + f"<end_of_turn>\n<start_of_turn>user\nTool result {t}: {words(args.obs_words)}"
                 "<end_of_turn>\n<start_of_turn>model\n")
    return out


def multi_turn(args):
    with ThreadPoolExecutor(args.concurrency) as pool:
        runs = [x for conv in pool.map(lambda c: conversation(args, c), range(args.conversations)) for x in conv]
    for t in range(1, args.turns + 1):
        rows = [r for r in runs if r["turn"] == t]
        ptoks = [(r["usage"] or {}).get("prompt_tokens") for r in rows]
        cached = [r["cached"] for r in rows if r["cached"] is not None]
        ttfts = sorted(r["ttft_ms"] for r in rows)
        print(json.dumps(dict(turn=t, prompt_tokens_med=statistics.median([p for p in ptoks if p] or [0]),
                              ttft_med_ms=round(statistics.median(ttfts), 2),
                              ttft_p90_ms=round(ttfts[int(0.9 * (len(ttfts) - 1))], 2),
                              cached_med=statistics.median(cached) if cached else None)), flush=True)
    if args.dump:
        json.dump(runs, open(args.dump, "w"))
    if args.compare:
        ref = {(r["conv"], r["turn"]): r for r in json.load(open(args.compare))}
        same_tok = same_lp = 0
        worst = 0.0
        for r in runs:
            q = ref[(r["conv"], r["turn"])]
            same_tok += r["tokens"] == q["tokens"]
            same_lp += r["logprobs"] == q["logprobs"]
            if r["tokens"] == q["tokens"]:
                worst = max([worst] + [abs(a - b) for a, b in zip(r["logprobs"], q["logprobs"])])
        print(json.dumps(dict(compared=len(runs), tokens_identical=same_tok, logprobs_identical=same_lp,
                              max_abs_dlogprob=worst)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-words", type=int, default=900)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--first-tokens", type=int, default=None, help="turn 1's max_tokens (default --max-tokens)")
    ap.add_argument("--turns", type=int, default=None, help="agentic multi-turn mode (module doc)")
    ap.add_argument("--conversations", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--obs-words", type=int, default=300)
    ap.add_argument("--logprobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--session", type=int, default=1, help="1 = one X-Session-Id per conversation")
    ap.add_argument("--dump")
    ap.add_argument("--compare")
    args = ap.parse_args()
    if args.turns:
        return multi_turn(args)
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
