#!/usr/bin/env bash
# Campaign R2 runner: ONE server load -> HSA gate -> coherence gate -> TTFT ladder -> GSM8K.
#
# One load, because loading GLM-5.2 TP8 costs 70-250 s and every arm otherwise pays it twice.
#
# THREE GATES BEFORE ANY NUMBER, each of which has produced a confident wrong answer here:
#   1. GPU lock + no sibling plowrt   (a sibling on the same port corrupts an arm silently)
#   2. HSA backend                    (the CPU reference backend SERVES PERFECTLY -- correct
#                                      answers, meaningless timings)
#   3. coherence                      (a fast wrong server is not a result)
#
# $1 assets  $2 port  $3 label
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="${PLOW_REPO:-$(cd "$HERE/../.." && pwd)}"
ASSETS="${1:?assets}"; PORT="${2:?port}"; LABEL="${3:?label}"
OUT="${OUT:-${TMPDIR:-/tmp}/twoengine}"; mkdir -p "$OUT"
N="${N:-100}"; SHOTS="${SHOTS:-8}"; MAXTOK="${MAXTOK:-320}"; CONC="${CONC:-1}"
CTXS="${CTXS:-1024 4096 8192 16384}"
GSM="${GSM:-${GSM8K_DIR:-$HOME/.cache/gsm8k}}"
LOG="$OUT/serve_$LABEL.log"
LOCK=/tmp/plow_gpu.lock

# GATE 0 -- the binary itself. `cargo build`/`cargo test` WITHOUT `--features hsa` silently
# replaces target/release/plowrt with a CPU-only binary, and that binary SERVES PERFECTLY:
# correct answers, coherence gate green, every timing fiction. Measured 2026-08-09: four
# interleaved A/B arms were destroyed this way by an unrelated `cargo test --workspace` running
# in the same session. Gate 2 below catches it after a 75 s model load; this catches it in a
# millisecond, before the GPU lock is even taken.
PLOWRT_BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
[ -x "$PLOWRT_BIN" ] || { echo "FAIL: no plowrt at $PLOWRT_BIN"; exit 1; }
grep -aq "libhsa-runtime64" "$PLOWRT_BIN" || {
  echo "FAIL: $PLOWRT_BIN was built WITHOUT --features hsa (no libhsa-runtime64 reference)."
  echo "      It would serve correct answers at fictional speed. Rebuild:"
  echo "      nix develop . -c cargo build --release -p plowrt --features hsa"
  exit 1; }

HAVE_LOCK=0

# THE TRAP MUST `exit`. `trap 'release' EXIT INT TERM` releases the lock and then lets the script
# keep running UNLOCKED, and its later EXIT deletes a lock that by then belongs to someone else.
release() {
  [ -n "${SPGID:-}" ] && kill -TERM "-$SPGID" 2>/dev/null
  sleep 3
  [ -n "${SPGID:-}" ] && kill -KILL "-$SPGID" 2>/dev/null
  # HAVE_LOCK is set ONLY after mkdir succeeds, so a script that never got the lock cannot
  # delete the holder's.
  #
  # `rm -rf`, NOT `rmdir`: we drop an `owner` file inside the lock dir for diagnosis, and
  # `rmdir` refuses a non-empty directory *silently* under `2>/dev/null`. The lock then LEAKS
  # and the next run blocks for the full acquire timeout on a lock whose owner is long dead --
  # which looks exactly like a legitimately busy box.
  [ "$HAVE_LOCK" = 1 ] && rm -rf "$LOCK"
  return 0
}
trap 'release; exit 143' INT TERM
trap 'release' EXIT

for i in $(seq 1 600); do
  mkdir "$LOCK" 2>/dev/null && { HAVE_LOCK=1; break; }
  sleep 5
done
[ "$HAVE_LOCK" = 1 ] || { echo "FAIL: could not take GPU lock"; exit 1; }
echo "$$ $LABEL" > "$LOCK/owner" 2>/dev/null

# `pgrep -x` is comm-exact and MISSES a sibling running a renamed binary (plowrt_stock).
# `pgrep -f "plowrt serve"` is worse -- it self-matches this launcher.
if pgrep '^plowrt' >/dev/null 2>&1; then
  echo "FAIL: a plowrt is already running:"; pgrep -a '^plowrt'; exit 1
fi

echo "=== [$LABEL] serving $ASSETS on :$PORT ==="
cd "$WT"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
# FULL PATH to nix: a detached/nohup context does not inherit the login PATH, and `setsid nix`
# then fails with "failed to execute nix" -- which the readiness loop reports as "server died
# during load", i.e. a toolchain problem wearing a model problem's clothes.
NIX="${PLOW_NIX:-/nix/var/nix/profiles/default/bin/nix}"
[ -x "$NIX" ] || { echo "FAIL: nix not found at '$NIX' (set PLOW_NIX)"; exit 1; }
# LD_LIBRARY_PATH MUST BE SET *INSIDE* THE NIX SHELL, and this is the single most expensive
# trap on this box. The flake does not carry /opt/rocm-*/lib, so `dlopen libhsa-runtime64.so.1`
# fails, plowrt falls back to the CPU reference interpreter, and it SERVES PERFECTLY -- correct
# answers, coherence gate green, every timing fiction. Setting it outside does not survive
# `nix develop`, which is why this goes through an inner `bash -c`.
ROCM_LIB="${PLOW_ROCM_LIB:-/opt/rocm-7.2.4/lib}"
[ -e "$ROCM_LIB/libhsa-runtime64.so.1" ] || {
  echo "FAIL: no libhsa-runtime64.so.1 under '$ROCM_LIB' (set PLOW_ROCM_LIB)"; exit 1; }
# SERVE_ENV: runtime flags the BLOB requires. GLM's blob carries the causal KV-split (ns=2),
# which only the V2 flash arm honours, so serving without PLOW_MLA_PF_V2=1 is REFUSED at load
# ("would write nsplit=1 partials under an ns-wide merge"). The refusal is the arm-refusal chain
# working -- it is a loud error, not wrong output, and it is why build.json must travel with
# model.pkt.
SERVE_ENV="${SERVE_ENV:-PLOW_MLA_PF_V2=1}"
echo "  serve env: $SERVE_ENV"
setsid "$NIX" develop "$WT" -c bash -c \
  "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:$ROCM_LIB\"; export $SERVE_ENV; \
   exec ./target/release/plowrt serve --assets '$ASSETS' --port '$PORT'" \
  > "$LOG" 2>&1 &
SPID=$!
SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"

for i in $(seq 1 1800); do
  kill -0 "$SPID" 2>/dev/null || { echo "FAIL: server died during load"; tail -30 "$LOG"; exit 1; }
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "FAIL: never became ready"; tail -30 "$LOG"; exit 1; }

# GATE 2 -- the CPU reference backend serves CORRECT ANSWERS at meaningless speed, so no
# output-based gate can catch it. This is the only thing that can.
if grep -q "CPU reference backend active" "$LOG"; then
  echo "FAIL: plowrt selected the CPU REFERENCE BACKEND -- every number below would be fiction."
  grep -E "HSA probe failed|hsa_init" "$LOG" | head -3; exit 1
fi
grep -qE "HSA backend selected|hsa=true" "$LOG" && echo "  HSA backend: OK" \
  || echo "  >>> WARN: no HSA banner (check $LOG)"

MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
        | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
[ -n "$MODEL" ] || { echo "FAIL: no model id"; exit 1; }
echo "  model: $MODEL"

# GATE 3
GATE=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":200,\"temperature\":0}")
if ! echo "$GATE" | grep -qi paris; then
  echo ">>> COHERENCE GATE FAIL -- refusing to report timings for a wrong server."
  echo "$GATE" | head -c 600; exit 1
fi
echo "  coherence gate: PASS"

MODEL="$MODEL" PORT="$PORT" LABEL="$LABEL" OUT="$OUT" N="$N" SHOTS="$SHOTS" \
MAXTOK="$MAXTOK" CONC="$CONC" CTXS="$CTXS" GSM="$GSM" python3 - <<'PY'
import json, os, re, statistics, threading, queue, time, urllib.request

PORT=os.environ["PORT"]; MODEL=os.environ["MODEL"]; LABEL=os.environ["LABEL"]
OUT=os.environ["OUT"]; N=int(os.environ["N"]); SHOTS=int(os.environ["SHOTS"])
MAXTOK=int(os.environ["MAXTOK"]); CONC=int(os.environ["CONC"])
CTXS=[int(x) for x in os.environ["CTXS"].split()]
URL=f"http://127.0.0.1:{PORT}/v1/chat/completions"
res={"label":LABEL,"model":MODEL}

def post(body, timeout=1800):
    req=urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

# ---- TTFT ladder. max_tokens=1 so the measurement is PREFILL, not prefill+decode.
# Off-rung by construction is NOT wanted here: we want the shipped ladder cells.
print("=== TTFT ladder (max_tokens=1, 3 reps, median) ===", flush=True)
lad={}
for ctx in CTXS:
    # A repeated single word tokenises ~1:1, so ctx words ~ ctx tokens. Exactness does not
    # matter for a ladder that is compared against ITSELF across arms.
    prompt=" ".join(["apple"]*ctx)
    reps=[]
    for r in range(3):
        t=time.time()
        try:
            post({"model":MODEL,"messages":[{"role":"user","content":prompt}],
                  "max_tokens":1,"temperature":0})
            reps.append((time.time()-t)*1000)
        except Exception as e:
            print(f"  ctx {ctx} rep {r} ERROR {e}", flush=True)
    if reps:
        reps.sort(); med=reps[len(reps)//2]
        spread=100*(max(reps)-min(reps))/med if med else 0
        lad[ctx]={"median_ms":round(med,1),"reps":[round(x,1) for x in reps],
                  "spread_pct":round(spread,1)}
        print(f"  ctx {ctx:6d}  median {med:8.1f} ms   reps {[round(x,1) for x in reps]}"
              f"   spread {spread:.1f}%", flush=True)
res["ttft"]=lad

# ---- TPOT: long generation at short ctx, so the decode steps dominate.
print("=== TPOT (256 tokens @ short ctx, 3 reps) ===", flush=True)
tp=[]
for r in range(3):
    t=time.time()
    try:
        o=post({"model":MODEL,"messages":[{"role":"user","content":"Count from 1 to 200, one number per line."}],
                "max_tokens":256,"temperature":0})
        dt=(time.time()-t)*1000
        n=o.get("usage",{}).get("completion_tokens") or 256
        tp.append(dt/max(n,1))
    except Exception as e:
        print(f"  rep {r} ERROR {e}", flush=True)
if tp:
    tp.sort(); res["tpot_ms"]=round(tp[len(tp)//2],3)
    print(f"  TPOT median {res['tpot_ms']} ms/token   reps {[round(x,3) for x in tp]}", flush=True)

# ---- GSM8K: accuracy AND speed, 8-shot CoT, greedy, exact-match on the last number.
GSM=os.environ["GSM"]
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
test=load(f"{GSM}/test.jsonl")[:N]
shots=load(f"{GSM}/train.jsonl")[:SHOTS] if SHOTS else []
pre=[]
for s in shots:
    pre.append({"role":"user","content":s["question"]})
    pre.append({"role":"assistant","content":s["answer"].replace("####","The answer is")})
NUM=re.compile(r"-?\d[\d,]*\.?\d*")
def final_number(s):
    m=NUM.findall(s.replace("$",""))
    return m[-1].replace(",","").rstrip(".") if m else None
def gold(a): return a.split("####")[-1].strip().replace(",","")

print(f"=== GSM8K {SHOTS}-shot greedy n={N} conc={CONC} ===", flush=True)
ok=bad=err=0; lat=[]; lock=threading.Lock(); work=queue.Queue()
for i,q in enumerate(test): work.put((i,q))
t0=time.time()
def run_one():
    global ok,bad,err
    while True:
        try: i,q=work.get_nowait()
        except queue.Empty: return
        ts=time.time()
        try:
            out=post({"model":MODEL,"messages":pre+[{"role":"user","content":q["question"]}],
                      "max_tokens":MAXTOK,"temperature":0})["choices"][0]["message"]["content"]
        except Exception as e:
            with lock: err+=1; print(f"  [{i}] ERROR {e}", flush=True)
            continue
        dt=time.time()-ts
        got,want=final_number(out),gold(q["answer"])
        try: hit=got is not None and abs(float(got)-float(want))<1e-4
        except ValueError: hit=(got==want)
        with lock:
            lat.append(dt); ok+=hit; bad+=(not hit); done=ok+bad
            if done%20==0 or done==1:
                print(f"  [{done}/{len(test)}] acc={ok/done:.3f}", flush=True)
th=[threading.Thread(target=run_one) for _ in range(CONC)]
for t in th: t.start()
for t in th: t.join()
n=ok+bad
if n:
    lat.sort()
    res["gsm8k"]={"n":n,"errors":err,"exact_match":round(ok/n,4),
                  "median_s":round(lat[len(lat)//2],2),
                  "mean_s":round(sum(lat)/len(lat),2),
                  "wall_s":round(time.time()-t0,1),
                  "throughput_qps":round(n/(time.time()-t0),3)}
    print(f"  GSM8K exact_match = {ok}/{n} = {ok/n:.4f}   errors={err}", flush=True)
    print(f"  latency/q median {lat[len(lat)//2]:.2f}s  wall {time.time()-t0:.0f}s", flush=True)

p=f"{OUT}/{LABEL}.json"
json.dump(res, open(p,"w"), indent=2)
print(f"=== wrote {p} ===", flush=True)
PY
echo "=== [$LABEL] done ==="
