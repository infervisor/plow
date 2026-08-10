#!/usr/bin/env python3
"""Full-set GSM8K with PER-QUESTION records, so two arms can be compared PAIRED.

Why this exists. `run_plow.sh` reports GSM8K as a single accuracy number, and at its default
n=100 that number cannot settle the question it is usually asked. Measured 2026-08-09 on the
MoE deterministic-writer arm: 0.970 vs 0.950 is 0.72 sigma unpaired, and McNemar on the
implied discordant count gives p ~= 0.50. An aggregate at n=100 can detect GROSS damage and
nothing else.

Two changes make it decisive:

  * the FULL test set (1319), which at a true 2 pp effect puts McNemar around z = 3;
  * per-question 0/1 records, so the comparison can be PAIRED. Both arms see the same
    questions in the same order under greedy decoding, and both arms are deterministic
    run-to-run, so each arm's vector IS its answer -- no repeats needed. Pairing is what
    buys the power here: it conditions away question difficulty, which is the dominant
    variance term.

Emits `<OUT>/<LABEL>_gsmfull.json` with a `correct` list of 0/1 aligned to test order.
Analyse with `mcnemar.py a.json b.json`.

env: PORT MODEL N SHOTS MAXTOK CONC GSM8K_DIR LABEL OUT
"""
import json, os, queue, re, threading, time, urllib.request

PORT = os.environ["PORT"]
N = int(os.environ.get("N", "1319"))
SHOTS = int(os.environ.get("SHOTS", "8"))
MAXTOK = int(os.environ.get("MAXTOK", "320"))
CONC = int(os.environ.get("CONC", "1"))
LABEL = os.environ.get("LABEL", "x")
OUT = os.environ.get("OUT", "/tmp/twoengine")
GSM = os.environ.get("GSM8K_DIR", os.path.expanduser("~/.cache/gsm8k"))
BASE = f"http://127.0.0.1:{PORT}"
MODEL = os.environ.get("MODEL", "auto")
if MODEL == "auto":
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        MODEL = json.load(r)["data"][0]["id"]


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


# -1 = never answered (an error), so an arm that silently drops requests cannot look correct.
correct = [-1] * len(test)
got_num = [None] * len(test)
lat = []
err = 0
lock = threading.Lock()
work = queue.Queue()
for i, q in enumerate(test):
    work.put((i, q))
t0 = time.time()


def run_one():
    global err
    while True:
        try:
            i, q = work.get_nowait()
        except queue.Empty:
            return
        body = json.dumps({
            "model": MODEL,
            "messages": pre + [{"role": "user", "content": q["question"]}],
            "max_tokens": MAXTOK,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(f"{BASE}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        ts = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.load(r)["choices"][0]["message"]["content"] or ""
        except Exception as e:
            with lock:
                err += 1
                print(f"  [{i}] ERROR {e}", flush=True)
            continue
        dt = time.time() - ts
        g, want = final_number(out), gold(q["answer"])
        try:
            hit = g is not None and abs(float(g) - float(want)) < 1e-4
        except ValueError:
            hit = (g == want)
        with lock:
            lat.append(dt)
            correct[i] = int(hit)
            got_num[i] = g
            done = sum(1 for c in correct if c >= 0)
            if done % 100 == 0:
                acc = sum(c for c in correct if c > 0) / done
                el = time.time() - t0
                print(f"  [{done}/{len(test)}] acc={acc:.4f}  {el/60:.1f} min elapsed, "
                      f"~{el / done * (len(test) - done) / 60:.0f} min left", flush=True)


th = [threading.Thread(target=run_one) for _ in range(CONC)]
for t in th:
    t.start()
for t in th:
    t.join()

n_ans = sum(1 for c in correct if c >= 0)
n_ok = sum(1 for c in correct if c == 1)
wall = time.time() - t0
lat.sort()
res = {
    "label": LABEL, "model": MODEL, "shots": SHOTS, "n_requested": len(test),
    "n_answered": n_ans, "errors": err,
    "exact_match": round(n_ok / n_ans, 4) if n_ans else None,
    "median_s": round(lat[len(lat) // 2], 2) if lat else None,
    "wall_s": round(wall, 1),
    "correct": correct, "predicted": got_num,
}
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/{LABEL}_gsmfull.json", "w"))
print(f"\nGSM8K {SHOTS}-shot greedy: {n_ok}/{n_ans} = {n_ok / max(n_ans,1):.4f}  errors={err}  "
      f"wall {wall/60:.1f} min", flush=True)
print(f"wrote {OUT}/{LABEL}_gsmfull.json", flush=True)
