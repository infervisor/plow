#!/usr/bin/env bash
# glm52_dsa_l4_oracle.sh — the DSA correctness question on a TRUNCATED model, in seconds.
#
#   perf-data/tools/gpulease -n 4 dsa-l4 sg render -c './scripts/glm52_dsa_l4_oracle.sh 4'
#
# WHY THIS EXISTS. The DSA arm was falsified on the full 78-layer model (perf-data/
# glm52-dsa-correctness.md §8), and every iteration on it paid the 167-255 s / 183 GiB-per-rank
# weight load first — `runtime/tests/glm52_decode.c:224` calls that load "the whole cost of a run".
# `GLM_NLAYERS=N` (mla.rs:3569) truncates the emit to the first N layers while KEEPING the full
# serving structure and the TP degree, so at N=4 the load is 4/78 of that. Use GLM_FULL=1 with it:
# the single-layer gate (GLM_FULL unset) asserts `tp == 1` at mla.rs:3860 and cannot do TP4/TP8.
#
# WHY THE GATE HERE IS AN ORACLE AND NOT COHERENCE. A 4-layer truncation of a 78-layer model emits
# gibberish BY CONSTRUCTION, so "does it say Paris" is meaningless. What IS meaningful, and is the
# whole point:
#
#   AT `kv_len <= top_k` A CORRECT DSA STEP IS ARITHMETICALLY DENSE.
#   `d_index_select_coop` clamps `top_k = min(index_topk, kv_len)`, so the selector emits EVERY
#   live row; `d_flash_mla_decode<GATHER=true>` then attends to all of them. Softmax is
#   permutation-invariant, so the order they come back in cannot matter. The DSA arm and the dense
#   arm MUST therefore produce the SAME TOKENS, gibberish or not — for any prompt shorter than
#   index_topk (2048). A divergence is a bug, full stop, and needs no reference implementation.
#
# That makes this a real oracle on 4 layers and a few seconds of load, and it is the vehicle the
# next person should iterate on. Only `longcoherence` and the published TTFT/TPOT table need the
# full 78 layers, because those are dominated by prefill over the whole stack.
#
# $1 = tp (bundles are emitted at TP4 by default; see the header of glm52_tpctx_sweep.sh)
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${GLM_SWEEP_DIR:-/home/lava/models/glm52_ctxsweep}"
TP="${1:-4}"
PORT="${PORT:-8201}"
cd "$WT" || exit 1
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

# Prompts stay well under index_topk=2048 so the clamp argument above applies to every one.
PROMPTS=${PROMPTS:-"The capital of France is|Once upon a time in a distant land|17 * 23 ="}

serve_and_ask () { # <assets> <tag>  -> writes /tmp/l4_<tag>.txt
  # SEPARATE statements, not one `local a=$1 b=$2 c=$b`: bash expands every word of the command
  # line before the `local` builtin runs, so `c=$b` sees the OLD (here: unset) b and dies under -u.
  local assets="$1" tag="$2"
  local log="/tmp/l4_srv_$tag.log"
  setsid nix develop -c ./target/release/plowrt serve --assets "$assets" --port "$PORT" \
    >"$log" 2>&1 &
  local srv=$!
  local ok=0 i
  for i in $(seq 1 300); do
    kill -0 $srv 2>/dev/null || break
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
    sleep 1
  done
  if [ "$ok" != 1 ]; then echo "!! $tag never became ready"; tail -20 "$log"; fi
  # The CPU-reference backend answers fluent-looking garbage through a byte-fallback tokenizer and
  # would make the two arms "differ" for a reason that has nothing to do with DSA. Refuse.
  if grep -q "CPU reference backend active" "$log"; then
    echo "!! $tag ran on the CPU REFERENCE BACKEND — this oracle measures NOTHING"
    grep -E "hsa_init|HSA probe failed" "$log" | head -2
    echo "   (this account is not in the 'render' group: GPU runs need 'sg render -c')"
    ok=0
  fi
  echo "== $tag ready after ${i}s (load is the whole cost of a run; 4/78 of it here)"
  : > "/tmp/l4_$tag.txt"
  local IFS='|' p
  for p in $PROMPTS; do
    curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys;print(json.dumps({"model":"glm-5.2","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":32,"temperature":0}))' "$p")" \
      | python3 -c 'import json,sys
try:
    r=json.load(sys.stdin); c=r["choices"][0]["message"]["content"]
    print(repr(c))
except Exception as e:
    print("REQUEST FAILED:", e)' >> "/tmp/l4_$tag.txt"
  done
  kill -TERM -"$srv" 2>/dev/null || kill -TERM "$srv" 2>/dev/null
  sleep 2; kill -KILL -"$srv" 2>/dev/null; sleep 3
  [ "$ok" = 1 ]
}

# BUNDLE_PREFIX lets the SAME oracle run against the FULL 78-layer A/B pair (`BUNDLE_PREFIX=`),
# not only the 4-layer truncation. The invariant is identical and depth-independent — below
# `index_topk` the two arms must emit the same tokens — so the only thing depth changes is the
# load time. Use the truncation to iterate and the full pair to sign off.
PFX="${BUNDLE_PREFIX-l4-}"
serve_and_ask "$OUT/${PFX}dense-tp$TP" dense || exit 1
serve_and_ask "$OUT/${PFX}dsa-tp$TP"   dsa   || exit 1

echo
echo "===== dense (4 layers, PLOW_GLM_DSA=0)"; cat /tmp/l4_dense.txt
echo "===== dsa   (4 layers, gate armed)";     cat /tmp/l4_dsa.txt
echo
if diff -q /tmp/l4_dense.txt /tmp/l4_dsa.txt >/dev/null; then
  echo ">>> DSA-vs-dense token oracle (kv_len < top_k, so they MUST agree): PASS"
  exit 0
else
  echo ">>> DSA-vs-dense token oracle: FAIL — the arms diverge where they are arithmetically equal"
  diff /tmp/l4_dense.txt /tmp/l4_dsa.txt | head -20
  exit 1
fi
