#!/usr/bin/env python3
"""Campaign R2 SPEED bench -- throughput ladder + long-context ladder, streaming.

ONE client for BOTH engines, same metric definitions as scripts/bench_speed.sh:
    TTFT       time to the first SSE delta that carries content
    ITL        inter-token latencies (so a p99 exists)
    TPOT       (last_tok_t - first_tok_t) / (out_tok - 1)
    out tok/s  aggregate completion tokens / wall

bench_speed.sh says in its own header that its numbers "MUST NOT BE TABLED NEXT TO A vLLM
NUMBER" because it is a different client from vLLM's reference harness. This file avoids that
the only way it can be avoided: it is the SAME client for both engines. Metric definitions are
copied from it so the numbers stay comparable to the project's own history.

EVERY PROMPT IS > 2048 TOKENS, DELIBERATELY, and this is a correctness constraint not a taste:
vLLM routes MLA prefill with `prefill_max_seq_len <= topk_tokens` (2048) to the DENSE MHA path
(mla_attention.py:756), and on ROCm the selected ROCM_AITER_MLA_SPARSE backend does not
implement `forward_mha` -- a short prompt kills the engine outright. Forcing the sparse path on
short prompts instead (sparse_mla_force_mqa) runs it out of spec and MEASURED 0.175 on GSM8K
against plow's 0.970. Above 2048 both engines run their intended path and both are correct, so
that is the only region where a speed comparison means anything.

PROMPTS ARE DISTINCT PER REQUEST. A shared prefix would be served from vLLM's prefix cache if it
were ever enabled, and would flatter whichever engine caches; plow has no prefix cache at all.

env: PORT MODEL LABEL OUT CONCS INLEN OUTLEN NPROMPT CTXS LC_OUTLEN
"""
import json, os, queue, random, threading, time, urllib.request

PORT = os.environ["PORT"]; LABEL = os.environ["LABEL"]; OUT = os.environ["OUT"]
CONCS = [int(x) for x in os.environ.get("CONCS", "1 4 8 16 32").split()]
INLEN = int(os.environ.get("INLEN", "4096"))
OUTLEN = int(os.environ.get("OUTLEN", "128"))
NMULT = int(os.environ.get("NMULT", "4"))          # requests per cell = NMULT * conc
CTXS = [int(x) for x in os.environ.get("CTXS", "4096 8192 16384 32768").split()]
LC_OUTLEN = int(os.environ.get("LC_OUTLEN", "32"))
BASE = f"http://127.0.0.1:{PORT}"
URL = f"{BASE}/v1/chat/completions"

MODEL = os.environ.get("MODEL", "auto")
if MODEL == "auto":
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        MODEL = json.load(r)["data"][0]["id"]
print(f"  model id: {MODEL}", flush=True)
res = {"label": LABEL, "model": MODEL, "inlen": INLEN, "outlen": OUTLEN}

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee "
         "zulu apple bridge candle dragon ember forest garden harbor island jungle").split()


def make_prompt(ntok, seed):
    """~1 token per word. Distinct per request via the seed, so no two requests share a prefix."""
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(ntok))


def stream_once(prompt, max_tokens, timeout=3600):
    """Returns (ttft_s, tpot_ms, n_out, itls). Streaming, so TTFT is the real first token."""
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    last = None
    itls = []
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if not delta.get("content"):
                continue           # role frame carries no token -- not TTFT
            now = time.time()
            if first is None:
                first = now
            else:
                itls.append(now - last)
            last = now
            n += 1
    if first is None:
        return None
    ttft = first - t0
    tpot = ((last - first) / (n - 1) * 1000.0) if n > 1 else 0.0
    return ttft, tpot, n, itls


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


# --------------------------------------------------------------- throughput ladder
print(f"=== THROUGHPUT ladder  in={INLEN} out={OUTLEN} ===", flush=True)
thr = {}
for conc in CONCS:
    nreq = NMULT * conc
    work = queue.Queue()
    for i in range(nreq):
        work.put(i)
    rows = []
    lock = threading.Lock()

    def run():
        while True:
            try:
                i = work.get_nowait()
            except queue.Empty:
                return
            try:
                r = stream_once(make_prompt(INLEN, (conc << 20) + i), OUTLEN)
            except Exception as e:
                with lock:
                    print(f"    req {i} ERROR {e}", flush=True)
                continue
            if r:
                with lock:
                    rows.append(r)

    t0 = time.time()
    th = [threading.Thread(target=run) for _ in range(conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    if rows:
        tot = sum(r[2] for r in rows)
        allitl = [x for r in rows for x in r[3]]
        thr[conc] = {"requests": len(rows), "wall_s": round(wall, 2),
                     "out_tok": tot, "out_tok_s": round(tot / wall, 2),
                     "req_s": round(len(rows) / wall, 4),
                     "ttft_p50_ms": round(pct([r[0] for r in rows], .5) * 1000, 1),
                     "tpot_p50_ms": round(pct([r[1] for r in rows], .5), 3),
                     "itl_p99_ms": round(pct(allitl, .99) * 1000, 2)}
        t = thr[conc]
        print(f"  conc {conc:3d}  out {t['out_tok_s']:8.2f} tok/s  req {t['req_s']:.4f}/s  "
              f"TTFT p50 {t['ttft_p50_ms']:8.1f} ms  TPOT p50 {t['tpot_p50_ms']:7.3f} ms  "
              f"ITL p99 {t['itl_p99_ms']:7.2f} ms", flush=True)
res["throughput"] = thr

# ------------------------------------------------------------- long-context ladder
print(f"=== LONG CONTEXT ladder  out={LC_OUTLEN}, 3 reps, median ===", flush=True)
lc = {}
for ctx in CTXS:
    reps = []
    for r in range(3):
        try:
            o = stream_once(make_prompt(ctx, 7000 + ctx * 10 + r), LC_OUTLEN)
            if o:
                reps.append(o[0] * 1000.0)      # TTFT ms
        except Exception as e:
            print(f"  ctx {ctx} rep {r} ERROR {e}", flush=True)
    if reps:
        s = sorted(reps)
        med = s[len(s) // 2]
        spread = 100 * (s[-1] - s[0]) / med if med else 0
        lc[ctx] = {"ttft_median_ms": round(med, 1), "reps": [round(x, 1) for x in s],
                   "spread_pct": round(spread, 1)}
        print(f"  ctx {ctx:6d}  TTFT median {med:9.1f} ms  reps {[round(x,1) for x in s]}"
              f"  spread {spread:.1f}%", flush=True)
res["long_context"] = lc

os.makedirs(OUT, exist_ok=True)
p = f"{OUT}/{LABEL}_speed.json"
json.dump(res, open(p, "w"), indent=2)
print(f"=== wrote {p} ===", flush=True)
