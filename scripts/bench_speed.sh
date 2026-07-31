#!/usr/bin/env bash
# Self-contained serving speed bench against a plowrt OpenAI endpoint.        [SPEED-BENCH]
#
# `bench_plowrt_serve.sh` is the reference harness and should be preferred WHEN IT CAN RUN: it
# drives `vllm bench serve` so plow and vLLM are measured by the same client binary with the same
# metric definitions, which is the only way the two are comparable. It needs the
# `rocm/vllm:...` image, and on a box without it the script dies at the docker step.
#
# This is the fallback: same metric DEFINITIONS, no docker, streaming so TTFT is real.
#   TTFT  first SSE content delta
#   TPOT  (last_token_time - first_token_time) / (out_tokens - 1)
#   ITL   inter-token latencies, so a p99 exists
#   out tok/s  aggregate completion tokens / wall
#
# NUMBERS FROM HERE MUST NOT BE TABLED NEXT TO A vLLM NUMBER. Different client, unvalidated
# against the reference implementation. It measures plow against ITSELF across arms, which is what
# a regression gate needs.
#
#   $1 assets  $2 port  $3 model (or `auto`)  [$4 ready-timeout]
#
#   IN_LENS  prompt lengths in tokens, space separated (default "128 1024 4096")
#   CONCS    concurrencies (default 1) — K3 decode is structurally batch-1 (the KDA recurrent
#            state has no batch axis), so anything above 1 measures QUEUEING, not batching.
#            The harness prints that warning itself rather than letting a reader assume otherwise.
#   NPROMPT  requests per cell (default 8)
#   OUTLEN   completion tokens (default 128)
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${1:?assets}"; PORT="${2:?port}"; MODEL="${3:?model|auto}"; READY="${4:-1800}"
IN_LENS="${IN_LENS:-128 1024 4096}"; CONCS="${CONCS:-1}"
NPROMPT="${NPROMPT:-8}"; OUTLEN="${OUTLEN:-128}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"

setsid nix develop "$WT" --command "$BIN" serve --assets "$ASSETS" --port "$PORT" \
  > /tmp/speed_serve_$PORT.log 2>&1 &
SPID=$!
SPGID="$(ps -o pgid= "$SPID" 2>/dev/null | tr -d ' ')"
cleanup() { [ -n "${SPGID:-}" ] && kill -TERM "-$SPGID" 2>/dev/null; sleep 3;
            [ -n "${SPGID:-}" ] && kill -KILL "-$SPGID" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "starting plowrt serve on :$PORT"
for i in $(seq 1 "$READY"); do
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SPID" 2>/dev/null || { echo "FAIL: server died"; tail -25 /tmp/speed_serve_$PORT.log; exit 1; }
  sleep 1
done
[ "$MODEL" = auto ] && MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
  | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
echo "  model: $MODEL"

# A fast wrong server is not a result — same gate bench_plowrt_serve.sh applies.
curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0}" \
  | grep -qi paris || { echo ">>> coherence gate FAIL"; exit 1; }
echo ">>> coherence gate: PASS"

MODEL="$MODEL" PORT="$PORT" IN_LENS="$IN_LENS" CONCS="$CONCS" NPROMPT="$NPROMPT" OUTLEN="$OUTLEN" \
python3 - <<'PY'
import json, os, statistics as st, threading, time, urllib.request, queue

MODEL=os.environ["MODEL"]; PORT=os.environ["PORT"]
IN_LENS=[int(x) for x in os.environ["IN_LENS"].split()]
CONCS=[int(x) for x in os.environ["CONCS"].split()]
NPROMPT=int(os.environ["NPROMPT"]); OUTLEN=int(os.environ["OUTLEN"])
URL=f"http://127.0.0.1:{PORT}/v1/chat/completions"

def prompt_of(ntok):
    # ~1 token per word for this filler; the exact count is not load-bearing because every arm
    # gets the SAME prompt, and the tokenizer's own count is reported by the server anyway.
    return " ".join(["the quick brown fox jumps over the lazy dog"] * max(1, ntok // 9))

def one(ptext):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":ptext}],
                     "max_tokens":OUTLEN,"temperature":0,"stream":True}).encode()
    req=urllib.request.Request(URL, body, {"Content-Type":"application/json"})
    t0=time.time(); ttft=None; times=[]
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.startswith(b"data: "): continue
            d=raw[6:].strip()
            if d==b"[DONE]": break
            try: j=json.loads(d)
            except Exception: continue
            ch=j.get("choices") or [{}]
            delta=(ch[0].get("delta") or {}).get("content")
            if delta:
                now=time.time()
                if ttft is None: ttft=now-t0
                times.append(now)
    return ttft, times

print(f"\n{'in_tok':>7}{'conc':>6}{'n':>4}{'ttft_ms':>10}{'ttft_med':>10}"
      f"{'tpot_ms':>9}{'itl_p99':>9}{'out_tok_s':>11}{'req_s':>8}")
for il in IN_LENS:
    p=prompt_of(il)
    for c in CONCS:
        q=queue.Queue(); res=[]
        for _ in range(NPROMPT): q.put(p)
        lock=threading.Lock(); t_start=time.time()
        def worker():
            while True:
                try: pt=q.get_nowait()
                except queue.Empty: return
                try: r=one(pt)
                except Exception as e: r=(None,[]); print("  ERR",e)
                with lock: res.append(r)
        ths=[threading.Thread(target=worker) for _ in range(c)]
        [t.start() for t in ths]; [t.join() for t in ths]
        wall=time.time()-t_start
        ttfts=[r[0]*1000 for r in res if r[0] is not None]
        tpots=[]; itls=[]
        for _,ts in res:
            if len(ts)>1:
                tpots.append((ts[-1]-ts[0])/(len(ts)-1)*1000)
                itls += [(b-a)*1000 for a,b in zip(ts, ts[1:])]
        ntok=sum(len(ts) for _,ts in res)
        itls.sort()
        p99=itls[int(len(itls)*0.99)] if itls else 0.0
        print(f"{il:>7}{c:>6}{len(res):>4}{st.mean(ttfts) if ttfts else 0:>10.1f}"
              f"{st.median(ttfts) if ttfts else 0:>10.1f}{st.mean(tpots) if tpots else 0:>9.2f}"
              f"{p99:>9.2f}{ntok/wall:>11.1f}{len(res)/wall:>8.2f}")
        if c > 1:
            print(f"{'':>7}  ^ conc>1 on a batch-1 engine measures QUEUEING, not batching")
PY
