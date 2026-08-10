#!/usr/bin/env python3
"""Campaign R2 measurement client -- TTFT ladder + TPOT + GSM8K against ANY OpenAI endpoint.

ONE file, used for BOTH plow and vLLM. That is the point: the campaign's headline is a RATIO
between two engines, and if each engine is measured by its own harness the ratio partly measures
the harnesses. Same prompts, same reps, same estimator, same exact-match rule.

env: PORT MODEL LABEL OUT N SHOTS MAXTOK CONC CTXS GSM
"""
import json, os, queue, re, statistics, threading, time, urllib.request

PORT = os.environ["PORT"]; LABEL = os.environ["LABEL"]; OUT = os.environ["OUT"]
N = int(os.environ.get("N", "100")); SHOTS = int(os.environ.get("SHOTS", "8"))
MAXTOK = int(os.environ.get("MAXTOK", "320")); CONC = int(os.environ.get("CONC", "1"))
CTXS = [int(x) for x in os.environ.get("CTXS", "1024 4096 8192 16384").split()]
GSM = os.environ.get("GSM", os.environ.get("GSM8K_DIR", os.path.expanduser("~/.cache/gsm8k")))
BASE = f"http://127.0.0.1:{PORT}"
URL = f"{BASE}/v1/chat/completions"

MODEL = os.environ.get("MODEL", "auto")
if MODEL == "auto":
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        MODEL = json.load(r)["data"][0]["id"]
print(f"  model id: {MODEL}", flush=True)
res = {"label": LABEL, "model": MODEL}


def post(body, timeout=1800):
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ---------------------------------------------------------------- TTFT ladder
# max_tokens=1 so this times PREFILL, not prefill+decode. Reps=3 and the MEDIAN is reported
# with the round-to-round spread beside it, because this box's own DVFS noise on plow has been
# measured at 17.9% where vLLM's was <=1.0% -- a single rep would be a coin flip, and a mean
# would be dragged by the slow one. State the spread or the number is not interpretable.
print("=== TTFT ladder (max_tokens=1, 3 reps, median) ===", flush=True)
lad = {}
for ctx in CTXS:
    prompt = " ".join(["apple"] * ctx)   # ~1 token/word; only self-consistency across arms matters
    reps = []
    for r in range(3):
        t = time.time()
        try:
            post({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 1, "temperature": 0})
            reps.append((time.time() - t) * 1000)
        except Exception as e:
            print(f"  ctx {ctx} rep {r} ERROR {e}", flush=True)
    if reps:
        s = sorted(reps); med = s[len(s) // 2]
        spread = 100 * (s[-1] - s[0]) / med if med else 0
        lad[ctx] = {"median_ms": round(med, 1), "reps": [round(x, 1) for x in s],
                    "spread_pct": round(spread, 1)}
        print(f"  ctx {ctx:6d}  median {med:8.1f} ms  reps {[round(x,1) for x in s]}"
              f"  spread {spread:.1f}%", flush=True)
res["ttft"] = lad

# ---------------------------------------------------------------------- TPOT
print("=== TPOT (256 tok, 3 reps) ===", flush=True)
tp = []
for r in range(3):
    t = time.time()
    try:
        o = post({"model": MODEL,
                  "messages": [{"role": "user", "content": "Count from 1 to 200, one per line."}],
                  "max_tokens": 256, "temperature": 0})
        dt = (time.time() - t) * 1000
        n = (o.get("usage") or {}).get("completion_tokens") or 256
        tp.append(dt / max(n, 1))
    except Exception as e:
        print(f"  rep {r} ERROR {e}", flush=True)
if tp:
    s = sorted(tp); res["tpot_ms"] = round(s[len(s) // 2], 3)
    print(f"  TPOT median {res['tpot_ms']} ms/tok  reps {[round(x,3) for x in s]}", flush=True)

# --------------------------------------------------------------------- GSM8K
# 8-shot CoT, greedy, exact match on the LAST number with commas/trailing period stripped --
# the lm-eval-harness `gsm8k` convention. Deliberately NOT a "contains" match, which would
# score a model that emits the right digits anywhere inside a wrong derivation.
#
# GREEDY IS NOT A SIMPLIFICATION HERE: plow's AMD backend samples argmax ON DEVICE and never
# shows the host a logit row, so temperature>0 would report a number the backend cannot produce.
def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


test = load(f"{GSM}/test.jsonl")[:N]
shots = load(f"{GSM}/train.jsonl")[:SHOTS] if SHOTS else []
pre = []
for s in shots:
    pre.append({"role": "user", "content": s["question"]})
    pre.append({"role": "assistant", "content": s["answer"].replace("####", "The answer is")})

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def final_number(s):
    m = NUM.findall(s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def gold(a):
    return a.split("####")[-1].strip().replace(",", "")


print(f"=== GSM8K {SHOTS}-shot greedy n={len(test)} conc={CONC} ===", flush=True)
ok = bad = err = 0
lat = []
lock = threading.Lock()
work = queue.Queue()
for i, q in enumerate(test):
    work.put((i, q))
t0 = time.time()


def run_one():
    global ok, bad, err
    while True:
        try:
            i, q = work.get_nowait()
        except queue.Empty:
            return
        ts = time.time()
        try:
            out = post({"model": MODEL,
                        "messages": pre + [{"role": "user", "content": q["question"]}],
                        "max_tokens": MAXTOK, "temperature": 0})["choices"][0]["message"]["content"]
        except Exception as e:
            with lock:
                err += 1
                print(f"  [{i}] ERROR {e}", flush=True)
            continue
        dt = time.time() - ts
        got, want = final_number(out), gold(q["answer"])
        try:
            hit = got is not None and abs(float(got) - float(want)) < 1e-4
        except ValueError:
            hit = (got == want)
        with lock:
            lat.append(dt)
            ok += hit
            bad += (not hit)
            done = ok + bad
            if done % 20 == 0 or done == 1:
                print(f"  [{done}/{len(test)}] acc={ok/done:.3f}", flush=True)


th = [threading.Thread(target=run_one) for _ in range(CONC)]
for t in th:
    t.start()
for t in th:
    t.join()

n = ok + bad
if n:
    wall = time.time() - t0
    lat.sort()
    res["gsm8k"] = {"n": n, "errors": err, "exact_match": round(ok / n, 4),
                    "median_s": round(lat[len(lat) // 2], 2),
                    "mean_s": round(sum(lat) / len(lat), 2),
                    "wall_s": round(wall, 1), "throughput_qps": round(n / wall, 3),
                    "conc": CONC}
    print(f"  GSM8K exact_match = {ok}/{n} = {ok/n:.4f}  errors={err}", flush=True)
    print(f"  latency/q median {lat[len(lat)//2]:.2f}s  wall {wall:.0f}s"
          f"  {n/wall:.3f} q/s", flush=True)

os.makedirs(OUT, exist_ok=True)
p = f"{OUT}/{LABEL}.json"
json.dump(res, open(p, "w"), indent=2)
print(f"=== wrote {p} ===", flush=True)
