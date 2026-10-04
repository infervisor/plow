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

--open-loop: production-like mix instead of fixed closed-loop sessions. Sessions arrive as a Poisson
process (--rate per s, or --target-concurrency in-flight requests via Little's law) for --duration s;
each draws an app (one of --apps shared system prompts), a geometric turn count, lognormal think
times, first-message / tool-output / reply lengths (reply: ignore_eos + per-turn max_tokens), capped
at --max-model-len. The plan is a pure function of the seed and flags (identical for every server).
Requests started in [--warmup, --duration - --cooldown] are measured: TTFT/TPOT/E2E, cached tokens,
throughput, mean in-flight requests and live sessions, goodput (requests meeting --slo-ttft-ms and
--slo-tpot-ms per s). No request starts after --duration.
"""
import argparse
import asyncio
import hashlib
import json
import math
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


async def one_turn(http, args, sid, history, system, user, max_tokens=None):
    import aiohttp
    max_tokens = max_tokens or args.max_tokens
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
    body.update(model=args.model, max_tokens=max_tokens, temperature=args.temperature, stream=True,
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
    if n_out < max_tokens:
        raise RuntimeError(f"reply cut at {n_out} of {max_tokens} tokens")
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


# ---------------------------------------------------------------- open loop (--open-loop)
def lognormal(r, median, sigma, lo, hi):
    return min(hi, max(lo, median * math.exp(sigma * r.gauss(0.0, 1.0))))


def geometric(r, mean, hi):
    """Turns in 1..hi, P(n) ~ (1-p)^(n-1) p with p = 1/mean (truncated at hi)."""
    n = 1
    while n < hi and r.random() >= 1.0 / mean:
        n += 1
    return n


def build_plan(args):
    """Seeded open-loop schedule, token sizes only: (system sizes, sessions). A pure function of the
    args, independent of the server, so both stacks get the same arrivals, turns, think times and
    lengths. Prompt at turn k = system + users[:k+1] + replies[:k] (+ chat template, held in
    --template-margin); a session ends before the turn that would exceed --max-model-len."""
    apps = [int(lognormal(rng_for(args.seed, "app", a), args.system_median, args.system_sigma, 256, 4096))
            for a in range(args.apps)]
    arr = rng_for(args.seed, "arrivals")
    sessions, t = [], 0.0
    while True:
        t += arr.expovariate(args.rate)
        if t >= args.duration:
            break
        r = rng_for(args.seed, "plan", len(sessions))
        app = r.randrange(args.apps)
        n = geometric(r, args.turns_mean, args.turns_max)
        ctx, turns = apps[app], []
        for k in range(n):
            first = k == 0
            user = int(lognormal(r, args.first_median if first else args.tool_median,
                                 args.first_sigma if first else args.tool_sigma, 64, args.max_model_len))
            out = int(lognormal(r, args.out_median, args.out_sigma, args.out_min, args.out_max))
            think = 0.0 if first else lognormal(r, args.think_median_s, args.think_sigma, 0.0, args.think_max_s)
            room = args.max_model_len - args.template_margin - ctx - out
            if first:
                user = max(64, min(user, room))
            elif user > room:
                break
            turns.append(dict(user_tokens=user, max_tokens=out, think_s=think, prompt_tokens=ctx + user))
            ctx += user + out
        sessions.append(dict(arrival_s=t, app=app, turns=turns))
    return apps, sessions


def build_open_content(args, sizer, apps, plan):
    systems = []
    for a, n in enumerate(apps):
        r = rng_for(args.seed, "system", a)
        head = f"You are assistant app {a} (seed {args.seed}). Use the tools, read their output, answer concisely.\n"
        systems.append(head + sizer.fill(r, max(16, n - sizer.count(head)), prose_line))
    users = []
    for s, sess in enumerate(plan):
        turns = []
        for k, turn in enumerate(sess["turns"]):
            r = rng_for(args.seed, "session", s, "turn", k)
            ask = f"[session {args.seed}-{s} turn {k + 1}] " + prose_line(r)
            head = "Context:\n" if k == 0 else f"Tool `{r.choice(TOOLS)}` returned:\n"
            body = sizer.fill(r, max(8, turn["user_tokens"] - sizer.count(ask + head) - 4), tool_line)
            turns.append(f"{head}{body}\n\n{ask}")
        users.append(turns)
    return systems, users


def rate_for_concurrency(args, concurrency, e2e_s):
    """Little's law: in-flight requests L = lambda * turns/session * E2E -> lambda."""
    plan = build_plan(argparse.Namespace(**dict(vars(args), rate=1.0, duration=2000.0)))[1]
    turns = sum(len(p["turns"]) for p in plan) / len(plan)
    return concurrency / (turns * e2e_s)


def window_metrics(rows, args, timeline):
    """Requests started in [warmup, duration - cooldown]; goodput = requests meeting both SLOs / s."""
    w0, w1 = args.warmup, args.duration - args.cooldown
    win = w1 - w0
    sel = [r for r in rows if w0 <= r["t_start"] < w1]
    d = summarize(sel, win)
    ok = [r for r in sel if r.get("error") is None]
    good = [r for r in ok if r["ttft_ms"] is not None and r["ttft_ms"] <= args.slo_ttft_ms
            and (r["tpot_ms"] is None or r["tpot_ms"] <= args.slo_tpot_ms)]
    e2e = [r["e2e_ms"] for r in ok]
    busy = sum(max(0.0, min(r["t_end"], w1) - max(r["t_start"], w0)) for r in rows)
    samples = [x for x in timeline if w0 <= x[0] < w1]
    d.update(window_start_s=w0, window_end_s=w1,
             total_tok_s=sum(r["prompt_tokens"] + r["completion_tokens"] for r in ok) / win,
             goodput_req_s=len(good) / win, slo_attainment=len(good) / len(sel) if sel else None,
             e2e_p50_ms=pct(e2e, 50), e2e_p99_ms=pct(e2e, 99),
             mean_inflight=busy / win, max_inflight=max((x[1] for x in samples), default=None),
             mean_sessions=sum(x[2] for x in samples) / len(samples) if samples else None,
             slo_ttft_ms=args.slo_ttft_ms, slo_tpot_ms=args.slo_tpot_ms)
    return d


async def run_open(args, systems, plan, users):
    import aiohttp
    rows, timeline = [], []
    live = dict(inflight=0, sessions=0)
    tag = f"prod-{args.seed}"

    async def session(http, s, t0):
        sess = plan[s]
        await asyncio.sleep(max(0.0, sess["arrival_s"] - (time.perf_counter() - t0)))
        live["sessions"] += 1
        history = []
        for k, turn in enumerate(sess["turns"]):
            if turn["think_s"]:
                await asyncio.sleep(turn["think_s"])
            start = time.perf_counter() - t0
            if start >= args.duration:  # hard stop: no request starts after the arrival window
                break
            live["inflight"] += 1
            try:
                row, reply = await one_turn(http, args, f"{tag}-{s}", history, systems[sess["app"]],
                                            users[s][k], turn["max_tokens"])
                row["error"] = None
            except Exception as e:  # noqa: BLE001 - recorded, the session stops
                row, reply = dict(ttft_ms=None, tpot_ms=None, e2e_ms=None, prompt_tokens=0, completion_tokens=0,
                                  cached_tokens=None, error=f"{type(e).__name__}: {e}"[:300]), None
            live["inflight"] -= 1
            row.update(session=s, turn=k + 1, app=sess["app"], t_start=start, t_end=time.perf_counter() - t0,
                       planned_prompt_tokens=turn["prompt_tokens"], max_tokens=turn["max_tokens"])
            rows.append(row)
            if reply is None:
                break
            history.append((users[s][k], reply))
        live["sessions"] -= 1

    async def sampler(t0, done):
        while not done.is_set():
            timeline.append((round(time.perf_counter() - t0, 3), live["inflight"], live["sessions"]))
            try:
                await asyncio.wait_for(done.wait(), 1.0)
            except asyncio.TimeoutError:
                pass

    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as http:
        done = asyncio.Event()
        t0 = time.perf_counter()
        samp = asyncio.ensure_future(sampler(t0, done))
        await asyncio.gather(*(session(http, s, t0) for s in range(len(plan))))
        wall = time.perf_counter() - t0
        done.set()
        await samp
    return rows, timeline, wall


def plan_stats(apps, plan):
    def q(xs):
        return dict({f"p{k}": pct(xs, k) for k in (1, 50, 90, 99)}, mean=sum(xs) / len(xs)) if xs else None
    turns = [len(p["turns"]) for p in plan]
    return dict(system_tokens=apps, sessions=len(plan), requests=sum(turns), turns=q(turns),
                prompt_tokens=q([t["prompt_tokens"] for p in plan for t in p["turns"]]),
                output_tokens=q([t["max_tokens"] for p in plan for t in p["turns"]]),
                think_s=q([t["think_s"] for p in plan for t in p["turns"] if t["think_s"]]))


def main_open(args, sizer):
    if args.target_concurrency:
        args.rate = rate_for_concurrency(args, args.target_concurrency, args.est_e2e_s)
    if not args.rate:
        raise SystemExit("--open-loop needs --rate or --target-concurrency")
    if args.warmup + args.cooldown >= args.duration:
        raise SystemExit("--warmup + --cooldown must be shorter than --duration")
    apps, plan = build_plan(args)
    systems, users = build_open_content(args, sizer, apps, plan)
    before = metrics_snapshot(args.url)
    rows, timeline, wall = asyncio.run(run_open(args, systems, plan, users))
    after = metrics_snapshot(args.url)
    server = None
    if before and after:
        hits = after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
        queries = after["prefix_cache_queries_total"] - before["prefix_cache_queries_total"]
        server = dict(prefix_hits=hits, prefix_queries=queries,
                      prefix_token_hit=hits / queries if queries > 0 else None)
    overall = window_metrics(rows, args, timeline)
    overall.update(run_wall_s=wall, errors_total=sum(1 for r in rows if r.get("error") is not None))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), plan=plan_stats(apps, plan), overall=overall, server=server,
                       timeline=timeline, requests=rows), f, indent=1)

    def fmt(v, p=1, scale=1):
        return "-" if v is None else f"{v * scale:.{p}f}"
    o = overall
    print(f"PROD  rate {args.rate:.3f}/s reqs {o['requests']} err {o['errors']} (run {o['errors_total']})  "
          f"goodput {fmt(o['goodput_req_s'], 3)} req/s ({fmt(o['slo_attainment'], 1, 100)}%)  "
          f"req/s {o['request_s']:.3f}  tok/s {o['total_tok_s']:.0f} (out {o['output_tok_s']:.0f})  "
          f"ttft p50/p99 {fmt(o['ttft_p50_ms'])}/{fmt(o['ttft_p99_ms'])} ms  "
          f"tpot p50/p99 {fmt(o['tpot_p50_ms'], 2)}/{fmt(o['tpot_p99_ms'], 2)} ms  "
          f"cached {fmt(o['cached_fraction'], 1, 100)}%  inflight {fmt(o['mean_inflight'])} "
          f"sessions {fmt(o['mean_sessions'])}")
    return 1 if o["errors_total"] else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model")
    ap.add_argument("--tokenizer", help="dir holding tokenizer.json (sizes the content)")
    ap.add_argument("--sessions", type=int, help="concurrent sessions (closed loop)")
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
    o = ap.add_argument_group("open loop (--open-loop): Poisson session arrivals, sampled turns/think/lengths")
    o.add_argument("--open-loop", action="store_true")
    o.add_argument("--rate", type=float, help="session arrivals per second")
    o.add_argument("--target-concurrency", type=float, help="in-flight requests; sets --rate by Little's law")
    o.add_argument("--est-e2e-s", type=float, default=8.0, help="mean request E2E assumed by --target-concurrency")
    o.add_argument("--duration", type=float, default=360, help="arrival window, s; no request starts after it")
    o.add_argument("--warmup", type=float, default=90, help="requests started earlier are not measured")
    o.add_argument("--cooldown", type=float, default=30, help="requests started in the last N s are not measured")
    o.add_argument("--apps", type=int, default=4, help="shared system prompts")
    o.add_argument("--system-median", type=float, default=1536)
    o.add_argument("--system-sigma", type=float, default=0.6)
    o.add_argument("--turns-mean", type=float, default=6)
    o.add_argument("--turns-max", type=int, default=20)
    o.add_argument("--first-median", type=float, default=1500, help="first user message tokens")
    o.add_argument("--first-sigma", type=float, default=1.0)
    o.add_argument("--tool-median", type=float, default=700, help="later turns' tool output tokens")
    o.add_argument("--tool-sigma", type=float, default=1.0)
    o.add_argument("--out-median", type=float, default=160)
    o.add_argument("--out-sigma", type=float, default=0.7)
    o.add_argument("--out-min", type=int, default=16)
    o.add_argument("--out-max", type=int, default=1024)
    o.add_argument("--think-median-s", type=float, default=5.0)
    o.add_argument("--think-sigma", type=float, default=0.8)
    o.add_argument("--think-max-s", type=float, default=60.0)
    o.add_argument("--max-model-len", type=int, default=16384)
    o.add_argument("--template-margin", type=int, default=512, help="tokens held back for the chat template")
    o.add_argument("--slo-ttft-ms", type=float, default=2000)
    o.add_argument("--slo-tpot-ms", type=float, default=100)
    args = ap.parse_args(argv)
    if not args.open_loop and not args.sessions:
        ap.error("--sessions is required without --open-loop")
    args.url = args.url.rstrip("/")
    if not args.model:
        with urllib.request.urlopen(args.url + "/v1/models", timeout=30) as r:
            args.model = json.load(r)["data"][0]["id"]
    sizer = Sizer(args.tokenizer)
    if args.open_loop:
        return main_open(args, sizer)
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
