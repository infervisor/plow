#!/usr/bin/env python3
"""Agentic multi-turn serving benchmark: N concurrent sessions x T turns, same client for any
OpenAI-compatible server (plowrt, vLLM).

  agentic_turns.py --url http://127.0.0.1:PORT --sessions 32 --out res.json
      [--model M] [--tokenizer DIR] [--turns 10] [--system-tokens 1536] [--target-tokens 15600]
      [--max-tokens 128] [--api chat|completions] [--temperature 0 | --temperature 1 --top-p 0.95]
      [--seed 1] [--sessions-per-worker 1] [--no-session-header]

Every session shares one system prompt (per seed) and appends, per turn, a tool output plus a short
user line sized so the prompt grows linearly to --target-tokens at the last turn. Each turn sends
the full history with the model's real previous replies (fixed length: ignore_eos + max_tokens, so
both servers see the same prompt lengths). `X-Session-Id` is sent to every server (plowrt resumes
the session's rows; vLLM ignores it and relies on automatic prefix caching). Content is a pure
function of (--seed, session, turn): a different seed per cell and repeat keeps a prefix cache from
replaying an earlier cell.

Reports per turn and overall: TTFT/TPOT p50/p99/max, output tok/s, requests/s, errors, cached-token
fraction (client: usage.prompt_tokens_details.cached_tokens, else plowrt's X-Session-Cached-Tokens;
server: the /metrics vllm:prefix_cache_hits/queries delta, which both servers export).
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_metrics import KEYS, metric_values  # noqa: E402

WORDS = ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega vector buffer index "
         "kernel tensor layer cache token batch queue slot stream frame packet route shard block "
         "config parse build check merge patch commit branch deploy fetch write read open close "
         "error warn info debug trace value result status count total offset length width height "
         "user model tool file path line func class struct method field param return async await").split()
TOOLS = ("grep", "read_file", "run_tests", "list_dir", "git_diff", "search", "http_get", "sql")


def rng_for(seed, *parts):
    h = hashlib.sha256(repr((seed,) + parts).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "little"))


class Sizer:
    """Token counts from the model's tokenizer.json; ~4 chars/token when none is given."""

    def __init__(self, tokenizer_dir):
        self.tok = None
        if tokenizer_dir:
            from tokenizers import Tokenizer
            path = tokenizer_dir if tokenizer_dir.endswith(".json") else os.path.join(tokenizer_dir, "tokenizer.json")
            self.tok = Tokenizer.from_file(path)

    def count(self, text):
        if self.tok is None:
            return max(1, len(text) // 4)
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def fill(self, r, tokens, line):
        """Lines from `line(r)` until the text holds `tokens` tokens, cut at a line or word."""
        lines, n = [], 0
        while n < tokens:
            s = line(r)
            lines.append(s)
            n += self.count(s + "\n")
        text = "\n".join(lines)
        while self.count(text) > tokens and " " in text:
            text = text.rsplit(" ", 1)[0]
        return text


def tool_line(r):
    words = " ".join(r.choice(WORDS) for _ in range(r.randint(4, 12)))
    return f"src/{r.choice(WORDS)}_{r.randint(0, 999)}.rs:{r.randint(1, 4000)}: {words} = {r.randint(0, 1 << 20)}"


def prose_line(r):
    return " ".join(r.choice(WORDS) for _ in range(r.randint(8, 20))).capitalize() + "."


def build_sessions(args, sizer):
    sys_r = rng_for(args.seed, "system")
    system = ("You are a coding agent. Use the tools, read their output, and answer concisely.\n"
              + sizer.fill(sys_r, args.system_tokens, prose_line))
    # Prompt at turn k = S + k*U + (k-1)*R (+ template); U solves turn T = target.
    per_turn = max(64, (args.target_tokens - args.system_tokens - (args.turns - 1) * args.max_tokens) // args.turns)
    sessions = []
    for s in range(args.sessions * args.sessions_per_worker):
        turns = []
        for t in range(args.turns):
            r = rng_for(args.seed, "session", s, "turn", t)
            tool = r.choice(TOOLS)
            ask = f"[session {args.seed}-{s} turn {t + 1}] " + prose_line(r)
            body = sizer.fill(r, max(16, per_turn - sizer.count(ask) - 16), tool_line)
            turns.append(f"Tool `{tool}` returned:\n{body}\n\n{ask}")
        sessions.append(turns)
    return system, sessions, per_turn


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))
    return xs[i]


def summarize(rows, wall_s):
    ok = [r for r in rows if r.get("error") is None]
    ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    out_tok = sum(r["completion_tokens"] for r in ok)
    prompt_tok = sum(r["prompt_tokens"] for r in ok)
    cached = [r["cached_tokens"] for r in ok if r["cached_tokens"] is not None]
    d = dict(requests=len(rows), errors=len(rows) - len(ok),
             ttft_p50_ms=pct(ttft, 50), ttft_p99_ms=pct(ttft, 99), ttft_max_ms=max(ttft, default=None),
             tpot_p50_ms=pct(tpot, 50), tpot_p99_ms=pct(tpot, 99),
             prompt_tokens_mean=prompt_tok / len(ok) if ok else None, output_tokens=out_tok,
             cached_fraction=(sum(cached) / prompt_tok) if cached and prompt_tok else None,
             cached_reported=len(cached))
    if wall_s:
        d.update(wall_s=wall_s, output_tok_s=out_tok / wall_s, request_s=len(ok) / wall_s)
    return d


def metrics_snapshot(url):
    try:
        txt = urllib.request.urlopen(url + "/metrics", timeout=5).read().decode()
    except Exception:  # noqa: BLE001 - a server without /metrics still benches
        return None
    return dict(zip(KEYS, metric_values(txt)))


async def one_turn(http, args, sid, history, system, user):
    import aiohttp
    if args.api == "chat":
        msgs = [{"role": "system", "content": system}]
        for u, a in history:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})
        body = dict(messages=msgs)
        path = "/v1/chat/completions"
    else:
        parts = [f"System: {system}"]
        for u, a in history:
            parts += [f"User: {u}", f"Assistant: {a}"]
        parts += [f"User: {user}", "Assistant:"]
        body = dict(prompt="\n\n".join(parts))
        path = "/v1/completions"
    body.update(model=args.model, max_tokens=args.max_tokens, temperature=args.temperature, stream=True,
                ignore_eos=True, stream_options={"include_usage": True})
    if args.top_p is not None:
        body["top_p"] = args.top_p
    headers = {"Content-Type": "application/json"}
    if args.session_header:
        headers["X-Session-Id"] = sid
    t0 = time.perf_counter()
    ttft, text, usage, header_cached = None, [], None, None
    async with http.post(args.url + path, json=body, headers=headers,
                         timeout=aiohttp.ClientTimeout(total=args.timeout)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        hc = resp.headers.get("X-Session-Cached-Tokens")
        header_cached = int(hc) if hc and hc.isdigit() else None
        buf = b""
        async for chunk in resp.content.iter_any():
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("error"):
                    raise RuntimeError(f"stream error: {json.dumps(d['error'])[:200]}")
                if d.get("usage"):
                    usage = d["usage"]
                for c in d.get("choices") or []:
                    delta = c.get("delta") or {}
                    piece = c.get("text") or delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        ttft = ttft or time.perf_counter() - t0
                        text.append(piece)
    e2e = time.perf_counter() - t0
    if not usage or usage.get("completion_tokens") is None or usage.get("prompt_tokens") is None:
        raise RuntimeError(f"stream carried no token usage: {usage}")
    n_out = usage["completion_tokens"]
    # ignore_eos: anything short of max_tokens was cut, not finished.
    if n_out < args.max_tokens:
        raise RuntimeError(f"reply cut at {n_out} of {args.max_tokens} tokens")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = header_cached
    return dict(ttft_ms=None if ttft is None else ttft * 1e3, e2e_ms=e2e * 1e3,
                tpot_ms=(e2e - ttft) / (n_out - 1) * 1e3 if ttft is not None and n_out > 1 else None,
                prompt_tokens=usage["prompt_tokens"], completion_tokens=n_out,
                cached_tokens=cached), "".join(text)


async def run(args, system, sessions):
    import aiohttp
    rows = []
    queue = list(range(len(sessions)))
    tag = f"agentic-{args.seed}"

    async def worker(http, w):
        for s in queue[w::args.sessions]:
            history = []
            for t, user in enumerate(sessions[s]):
                t_start = time.time()
                try:
                    row, reply = await one_turn(http, args, f"{tag}-{s}", history, system, user)
                    row["error"] = None
                except Exception as e:  # noqa: BLE001 - recorded, the session stops
                    row, reply = dict(ttft_ms=None, tpot_ms=None, prompt_tokens=0, completion_tokens=0,
                                      cached_tokens=None, error=f"{type(e).__name__}: {e}"[:300]), None
                row.update(session=s, turn=t + 1, start=t_start)
                rows.append(row)
                if reply is None:
                    break
                history.append((user, reply))
                if args.think_ms:
                    await asyncio.sleep(args.think_ms / 1e3)

    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as http:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(http, w) for w in range(args.sessions)))
        wall = time.perf_counter() - t0
    return rows, wall


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model")
    ap.add_argument("--tokenizer", help="dir holding tokenizer.json (sizes the content)")
    ap.add_argument("--sessions", type=int, required=True, help="concurrent sessions")
    ap.add_argument("--sessions-per-worker", type=int, default=1)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--system-tokens", type=int, default=1536)
    ap.add_argument("--target-tokens", type=int, default=15600, help="prompt tokens at the last turn")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--api", choices=("chat", "completions"), default="chat")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--think-ms", type=float, default=0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--no-session-header", dest="session_header", action="store_false")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    args.url = args.url.rstrip("/")
    if not args.model:
        with urllib.request.urlopen(args.url + "/v1/models", timeout=30) as r:
            args.model = json.load(r)["data"][0]["id"]
    sizer = Sizer(args.tokenizer)
    system, sessions, per_turn = build_sessions(args, sizer)
    before = metrics_snapshot(args.url)
    rows, wall = asyncio.run(run(args, system, sessions))
    after = metrics_snapshot(args.url)
    server = None
    if before and after:
        hits = after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
        queries = after["prefix_cache_queries_total"] - before["prefix_cache_queries_total"]
        server = dict(prefix_hits=hits, prefix_queries=queries,
                      prefix_token_hit=hits / queries if queries > 0 else None,
                      prompt_tokens=after["prompt_tokens_total"] - before["prompt_tokens_total"])
    per_turn_stats = [dict(turn=t, **summarize([r for r in rows if r["turn"] == t], None))
                      for t in range(1, args.turns + 1)]
    overall = summarize(rows, wall)
    cfg = {k: v for k, v in vars(args).items()}
    cfg["user_tokens_per_turn"] = per_turn
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=cfg, overall=overall, per_turn=per_turn_stats, server=server, requests=rows), f, indent=1)

    def fmt(v, p=1):
        return "-" if v is None else f"{v:.{p}f}"
    print(f"{'turn':>5} {'reqs':>5} {'err':>4} {'prompt':>7} {'cached':>7} {'ttft p50':>9} {'ttft p99':>9} "
          f"{'tpot p50':>9} {'tpot p99':>9}")
    for t in per_turn_stats:
        print(f"{t['turn']:5d} {t['requests']:5d} {t['errors']:4d} {fmt(t['prompt_tokens_mean'], 0):>7} "
              f"{fmt(None if t['cached_fraction'] is None else 100 * t['cached_fraction']):>6}% "
              f"{fmt(t['ttft_p50_ms']):>9} {fmt(t['ttft_p99_ms']):>9} {fmt(t['tpot_p50_ms'], 2):>9} "
              f"{fmt(t['tpot_p99_ms'], 2):>9}")
    o = overall
    print(f"ALL   reqs {o['requests']} err {o['errors']} wall {o['wall_s']:.1f}s  out tok/s {o['output_tok_s']:.1f}  "
          f"req/s {o['request_s']:.3f}  ttft p50/p99/max {fmt(o['ttft_p50_ms'])}/{fmt(o['ttft_p99_ms'])}/"
          f"{fmt(o['ttft_max_ms'])} ms  tpot p50/p99 {fmt(o['tpot_p50_ms'], 2)}/{fmt(o['tpot_p99_ms'], 2)} ms  "
          f"cached {fmt(None if o['cached_fraction'] is None else 100 * o['cached_fraction'])}%  "
          f"server prefix hit {fmt(None if not server or server['prefix_token_hit'] is None else 100 * server['prefix_token_hit'])}%")
    return 1 if o["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
