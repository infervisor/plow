#!/usr/bin/env python3
"""The `bench_speed.sh` client, extracted standalone so BOTH engines are driven by the
SAME code and the SAME metric definitions.

`scripts/bench_speed.sh` carries a warning that its numbers must not be tabled next to a
vLLM number, because it is a different client from `vllm bench serve`. That warning is
about the CLIENT, not the metrics: the fix is not to avoid the comparison, it is to point
the same client at both servers. This file is that client, with the same metric arithmetic
as the heredoc in bench_speed.sh:

    TTFT  first SSE chunk carrying a content delta
    TPOT  (last_token_time - first_token_time) / (n_deltas - 1)
    ITL   inter-delta latencies, so a p99 exists
    out tok/s  sum(n_deltas) / wall of the cell
    req/s      n_requests / wall of the cell

Both servers speak /v1/chat/completions with `stream: true`, so the same reader works on
both. `stream_options.include_usage` is requested but tolerated absent: when the server
returns it, the tokenizer's own prompt/completion counts are recorded, which is the only
way to state what "in_len 1024" actually was on each engine's tokenizer.

Usage (env-driven, like bench_speed.sh):
    BASE_URL=http://127.0.0.1:8477 MODEL=auto TAG=vllm-fp8 \
    IN_LENS=1024 CONCS="1 2 4 8 16" NPROMPT=16 OUTLEN=128 REPS=3 \
    CSV=/tmp/out.csv python3 scripts/sweep_client.py
"""
import json, os, statistics as st, sys, threading, time, urllib.request, queue

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "auto")
IN_LENS = [int(x) for x in os.environ.get("IN_LENS", "1024").split()]
CONCS = [int(x) for x in os.environ.get("CONCS", "1 2 4 8 16").split()]
NPROMPT = int(os.environ.get("NPROMPT", "16"))
# NPROMPT_SCALE>0 sets NPROMPT = SCALE*conc per cell instead of a fixed count. With a FIXED
# count the cell runs ceil(NPROMPT/conc) waves and the last one is short whenever conc does not
# divide NPROMPT, which depresses aggregate throughput at 3/6/12 for a reason that has nothing
# to do with the engine. The ladder sweep only ever ran divisor concurrencies (1,2,4,8,16) so it
# never met this; an off-rung sweep does, and must control for it.
NPROMPT_SCALE = int(os.environ.get("NPROMPT_SCALE", "0"))
OUTLEN = int(os.environ.get("OUTLEN", "128"))
REPS = int(os.environ.get("REPS", "3"))
TAG = os.environ.get("TAG", "arm")
CSV = os.environ.get("CSV", "")
GATE = os.environ.get("GATE", "1") == "1"
URL = f"{BASE}/v1/chat/completions"


def models():
    return json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=30).read())


if MODEL == "auto":
    MODEL = models()["data"][0]["id"]
# plow publishes the packet's decode batch on /v1/models; vLLM does not. Absent => 0,
# reported as unknown rather than guessed (same rule as bench_speed.sh).
try:
    BATCH = int(next((d.get("batch") for d in models().get("data", []) if d.get("batch")), 0)) or 0
except Exception:
    BATCH = 0


def prompt_of(ntok):
    # IDENTICAL to bench_speed.sh: ~1 token per word for this filler. Every arm gets the
    # same prompt text, and the server's own usage.prompt_tokens is recorded below.
    return " ".join(["the quick brown fox jumps over the lazy dog"] * max(1, ntok // 9))


def one(ptext):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": ptext}],
                       "max_tokens": OUTLEN, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; times = []; usage = None; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            d = raw[6:].strip()
            if d == b"[DONE]":
                break
            try:
                j = json.loads(d)
            except Exception:
                continue
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or [{}]
            if not ch:
                continue
            delta = (ch[0].get("delta") or {}).get("content")
            if delta:
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                times.append(now)
                text.append(delta)
    return ttft, times, usage, "".join(text)


def nostream(msg, maxtok=32):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": msg}],
                       "max_tokens": maxtok, "temperature": 0}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    j = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return j["choices"][0]["message"]["content"]


if GATE:
    # A fast wrong server is not a result. Same gate on both engines.
    a = nostream("What is the capital of France?")
    print(f"# coherence gate [{TAG}] -> {a!r}", file=sys.stderr)
    if "paris" not in a.lower():
        print(">>> coherence gate FAIL", file=sys.stderr)
        sys.exit(1)
    print(">>> coherence gate: PASS", file=sys.stderr)

rows = []
hdr = ("tag,rep,in_tok,conc,n,ttft_mean_ms,ttft_med_ms,tpot_mean_ms,tpot_med_ms,"
       "itl_p99_ms,out_tok_s,req_s,prompt_tokens,completion_tokens,deltas")
print(hdr)
sys.stdout.flush()

# One warm-up request per input length, discarded: the first request on either engine pays
# JIT / graph capture.
for il in IN_LENS:
    one(prompt_of(il))

for rep in range(1, REPS + 1):
    for il in IN_LENS:
        p = prompt_of(il)
        for c in CONCS:
            npr = NPROMPT_SCALE * c if NPROMPT_SCALE else NPROMPT
            q = queue.Queue(); res = []
            for _ in range(npr):
                q.put(p)
            lock = threading.Lock(); t_start = time.time()

            def worker():
                while True:
                    try:
                        pt = q.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        r = one(pt)
                    except Exception as e:
                        r = (None, [], None, ""); print("  ERR", e, file=sys.stderr)
                    with lock:
                        res.append(r)

            ths = [threading.Thread(target=worker) for _ in range(c)]
            [t.start() for t in ths]; [t.join() for t in ths]
            wall = time.time() - t_start
            ttfts = [r[0] * 1000 for r in res if r[0] is not None]
            tpots = []; itls = []
            for _, ts, _u, _t in res:
                if len(ts) > 1:
                    tpots.append((ts[-1] - ts[0]) / (len(ts) - 1) * 1000)
                    itls += [(b - a) * 1000 for a, b in zip(ts, ts[1:])]
            ndelta = sum(len(ts) for _, ts, _u, _t in res)
            us = [r[2] for r in res if r[2]]
            ptok = st.median([u.get("prompt_tokens", 0) for u in us]) if us else 0
            ctok = st.median([u.get("completion_tokens", 0) for u in us]) if us else 0
            itls.sort()
            p99 = itls[int(len(itls) * 0.99)] if itls else 0.0
            row = (f"{TAG},{rep},{il},{c},{len(res)},"
                   f"{st.mean(ttfts) if ttfts else 0:.1f},{st.median(ttfts) if ttfts else 0:.1f},"
                   f"{st.mean(tpots) if tpots else 0:.2f},{st.median(tpots) if tpots else 0:.2f},"
                   f"{p99:.2f},{ndelta / wall:.1f},{len(res) / wall:.2f},"
                   f"{ptok:.0f},{ctok:.0f},{ndelta}")
            print(row); sys.stdout.flush()
            rows.append(row)

if CSV:
    with open(CSV, "w") as f:
        f.write(hdr + "\n" + "\n".join(rows) + "\n")
print(f"# model={MODEL} batch_reported={BATCH or 'n/a'}", file=sys.stderr)
