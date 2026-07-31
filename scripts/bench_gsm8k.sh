#!/usr/bin/env bash
# GSM8K accuracy against a plowrt OpenAI endpoint.                       [ACCURACY-GATE]
#
# THIS IS THE FIRST ACCURACY HARNESS IN THE TREE. Everything that existed before it is a
# TOKEN-IDENTITY gate — `k3_tp_equivalence.sh` (tp1 vs tp8 on one asset), the Paris continuation
# in `kimi-k3-README.md` §4, the coherence gate inside `bench_plowrt_serve.sh`. Those prove the
# runtime is self-consistent; NONE of them proves the model is right. A throughput number without
# an accuracy number is not publishable against vLLM, which is what this closes.
#
#   $1 assets  $2 port  $3 model-slug  [$4 ready-timeout]
#
#   N          questions (default 200; the full test split is 1319)
#   SHOTS      few-shot exemplars (default 8 — the standard GSM8K setting)
#   MAXTOK     generation cap per question (default 320)
#   CONC       in-flight requests (default 1); >1 is what tests a BATCHED packet end to end
#   TEMP       sampling temperature (default 0 — greedy, so the run is reproducible)
#   GSM8K      path to a local `test.jsonl`; else it is fetched once to $CACHE
#   CACHE      dataset cache dir (default /home/lava/models/gsm8k)
#   PLOWRT_BIN pre-copied binary, same reason as bench_plowrt_serve.sh
#
# METHOD, and the two things that make an accuracy number honest here:
#
#   * GREEDY (temperature 0). plow's gfx950 backend samples argmax ON DEVICE and ignores
#     top_p/top_k/penalties entirely (`mux.rs`: "the gfx950 engine samples on device and the host
#     never sees the logit row"). Asking for temperature > 0 would therefore report a number the
#     backend cannot actually produce. Greedy is not a simplification here, it is the only
#     faithful setting.
#   * EXACT MATCH on the final number, extracted as the LAST number in the completion, with commas
#     and a trailing period stripped. GSM8K's reference answer is the token after `####`. This is
#     the lm-eval-harness `exact_match` convention for `gsm8k` in its 8-shot CoT form; it is
#     deliberately NOT a "contains" match, which scores a model that emits the right digits
#     anywhere in a wrong derivation.
#
# The server is started and torn down exactly as `bench_plowrt_serve.sh` does it — `setsid`, kill
# by PROCESS GROUP — because `nix develop -c` execs a shell that forks plowrt, so killing the pid
# we waited on leaves the real server holding the cards.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:?assets dir}"; PORT="${2:?port}"; MODEL="${3:?model slug}"; READY="${4:-1800}"
N="${N:-200}"; SHOTS="${SHOTS:-8}"; MAXTOK="${MAXTOK:-320}"; TEMP="${TEMP:-0}"
CONC="${CONC:-1}"
CACHE="${CACHE:-/home/lava/models/gsm8k}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"

mkdir -p "$CACHE"
DATA="${GSM8K:-$CACHE/test.jsonl}"
if [ ! -s "$DATA" ]; then
  echo "fetching GSM8K test split -> $DATA"
  curl -sL -o "$DATA" \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    || { echo "FAIL: could not fetch GSM8K; set GSM8K=<path to test.jsonl>"; exit 2; }
fi
TRAIN="$CACHE/train.jsonl"
if [ "$SHOTS" -gt 0 ] && [ ! -s "$TRAIN" ]; then
  curl -sL -o "$TRAIN" \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl \
    || { echo "FAIL: could not fetch GSM8K train split for few-shot"; exit 2; }
fi
wc -l "$DATA" | awk '{print "  test split:", $1, "questions"}'

echo "starting plowrt serve on :$PORT  (assets $ASSETS)"
setsid nix develop "$WT" --command "$BIN" serve --assets "$ASSETS" --port "$PORT" \
  > /tmp/gsm8k_serve_$PORT.log 2>&1 &
SPID=$!
SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"
cleanup() { [ -n "${SPGID:-}" ] && kill -TERM "-$SPGID" 2>/dev/null; sleep 3;
            [ -n "${SPGID:-}" ] && kill -KILL "-$SPGID" 2>/dev/null; }
trap cleanup EXIT INT TERM

for i in $(seq 1 "$READY"); do
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SPID" 2>/dev/null || { echo "FAIL: server died during load"; tail -25 /tmp/gsm8k_serve_$PORT.log; exit 1; }
  sleep 1
done
curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "FAIL: server never became ready"; exit 1; }
echo "  ready; served models: $(curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 300)"

# `MODEL=auto` resolves the slug from /v1/models. The served id comes from the BLOB's network
# name, not from a flag (kimi-k3-README.md §4: "query /v1/models, don't guess"), and guessing it
# wrong fails the coherence gate below in a way that looks exactly like a bad model.
if [ "$MODEL" = "auto" ]; then
  MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
          | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  [ -n "$MODEL" ] || { echo "FAIL: could not resolve model id from /v1/models"; exit 1; }
  echo "  resolved model id: $MODEL"
fi

# The same coherence gate bench_plowrt_serve.sh uses: a fast wrong server is not a result, and an
# accuracy harness that scores 0% because the model id was wrong looks exactly like a bad model.
GATE=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0}")
echo "$GATE" | grep -qi paris || { echo ">>> coherence gate FAIL — accuracy below would be meaningless"; echo "$GATE" | head -c 500; exit 1; }
echo ">>> coherence gate: PASS"

N="$N" SHOTS="$SHOTS" MAXTOK="$MAXTOK" TEMP="$TEMP" MODEL="$MODEL" PORT="$PORT" CONC="$CONC" \
DATA="$DATA" TRAIN="$TRAIN" python3 - <<'PY'
import json, threading, queue, os, re, sys, time, urllib.request

N=int(os.environ["N"]); SHOTS=int(os.environ["SHOTS"]); MAXTOK=int(os.environ["MAXTOK"])
TEMP=float(os.environ["TEMP"]); MODEL=os.environ["MODEL"]; PORT=os.environ["PORT"]
CONC=int(os.environ.get("CONC","1"))
URL=f"http://127.0.0.1:{PORT}/v1/chat/completions"

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

test = load(os.environ["DATA"])[:N]
shots = load(os.environ["TRAIN"])[:SHOTS] if SHOTS else []

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def final_number(s):
    """Last number in the string, commas and trailing period stripped. lm-eval's convention."""
    m = NUM.findall(s.replace("$", ""))
    if not m: return None
    v = m[-1].replace(",", "").rstrip(".")
    return v

def gold(a):
    return a.split("####")[-1].strip().replace(",", "")

# 8-shot CoT prompt, exemplars carried as prior turns so the chat template is exercised the way
# a real client would use it (plowrt serves /v1/chat/completions only).
preamble = []
for s in shots:
    preamble.append({"role": "user", "content": s["question"]})
    preamble.append({"role": "assistant", "content": s["answer"].replace("####", "The answer is")})

# CONC IN-FLIGHT REQUESTS. At CONC=1 this is the original sequential harness, question for
# question. Above 1 it is also the ONLY way this file tests a BATCHED packet: a sequential client
# keeps ONE slot live and the other B-1 idle, so it would exercise the batched decode PROGRAM
# without ever exercising batching. Accuracy must not move with CONC — if it does, sequences are
# contaminating each other, which is what scripts/k3_batch_gate.sh guards at the token level and
# what this catches end to end.
#
# Greedy (TEMP=0) keeps each answer independent of the others' timing, so only the ORDER of
# completion varies with CONC, never a completion.
ok = bad = err = 0
t0 = time.time()
lat = []
lock = threading.Lock()
work = queue.Queue()
for i, q in enumerate(test):
    work.put((i, q))

def run_one():
    global ok, bad, err
    while True:
        try:
            i, q = work.get_nowait()
        except queue.Empty:
            return
        msgs = preamble + [{"role": "user", "content": q["question"]}]
        body = json.dumps({"model": MODEL, "messages": msgs,
                           "max_tokens": MAXTOK, "temperature": TEMP}).encode()
        req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
        ts = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.load(r)["choices"][0]["message"]["content"]
        except Exception as e:
            with lock:
                err += 1
                print(f"  [{i}] REQUEST ERROR {e}", flush=True)
            continue
        dt = time.time() - ts
        got, want = final_number(out), gold(q["answer"])
        try:
            hit = got is not None and abs(float(got) - float(want)) < 1e-4
        except ValueError:
            hit = (got == want)
        with lock:
            lat.append(dt)
            ok += hit; bad += (not hit)
            done = ok + bad
            if done % 10 == 0 or done == 1:
                print(f"  [{done}/{len(test)}] acc={ok/done:.3f}  last: got={got} want={want}",
                      flush=True)

threads = [threading.Thread(target=run_one) for _ in range(CONC)]
for t in threads: t.start()
for t in threads: t.join()

n = ok + bad
print()
print(f"GSM8K  {SHOTS}-shot  greedy(temp={TEMP})  n={n}  errors={err}")
print(f"  exact_match = {ok}/{n} = {ok/n:.4f}" if n else "  no completions")
if lat:
    lat.sort()
    print(f"  latency/question: median {lat[len(lat)//2]:.2f}s  mean {sum(lat)/len(lat):.2f}s"
          f"  total {time.time()-t0:.0f}s")
sys.exit(0 if n else 3)
PY
