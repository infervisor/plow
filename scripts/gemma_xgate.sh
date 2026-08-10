#!/usr/bin/env bash
# Gemma-4-12B fp8/w8a8 gfx942 CROSS-GATE — ON A FRESHLY EMITTED BLOB.       [GEMMA-XGATE]
#
# WHY THIS SCRIPT EXISTS. The standing cross-gate was a hand-run procedure that re-used the
# STORED asset `/workspace/assets/gfx942/g12b-fp8` and rebuilt only the OBJECTS. That asset was
# emitted 2026-08-04, before the fused activation quant landed, so for weeks the gate
# re-certified a blob that did not contain the regression under test. `PLOW_FUSE_QUANT`
# (default ON) shipped WRONG OUTPUT on gfx942 the whole time — a freshly emitted blob answers
# "capital of France" with `,1___....1.111111111111` — and the gate said PASS on every merge.
#
#   A GATE THAT NEVER RE-EMITS CANNOT CATCH AN EMITTER REGRESSION.
#
# So this gate emits the blob from the CURRENT CHECKOUT on every run, and the objects too
# unless a prebuilt set is handed in. The stored asset stays available as an explicit control
# arm (`PLOW_XGATE_STORED`) — as a bracket, never as the subject.
#
# WHY GEMMA IS THE SUBJECT. `qnorm_fuse` is local to `emit_phase`, which only the dense-GQA
# family reaches, and it needs `w8a8` — GLM/Kimi/DeepSeek are not exposed. Gemma-4-12B fp8 is
# the smallest model on this box that exercises that path, and it prefills, which is where the
# fold lives and where a corrupted KV cache gets written. `amd-bench`'s `last id` is NOT a
# correctness signal (it never prefills); that is the second reason this gate serves a real
# prompt instead of benching.
#
# WHAT IT ASSERTS. Coherence, not speed. Each prompt below must be answered with a match for
# its regex. The failure this was built to catch is FLUENT AND CONFIDENT, so "the server came
# up and returned 32 tokens" would not see it — only the content does.
#
# PROVEN TO FAIL. A gate never shown to fail on a known-bad input is not evidence of anything.
# The known-bad input is `PLOW_QNORM_FUSE=1`, which still reaches the broken fold on gfx942
# (75fb82f changed only the DEFAULT). Both transcripts are in
# perf-data/plow-gfx942/gemma-xgate-fresh-blob.md.
#
# USAGE
#   scripts/gemma_xgate.sh [name]
#
#   name              label for the job dir (default: a timestamp)
#
#   PLOW_XGATE_EMIT_ENV   extra `env` assignments for the EMIT, space separated. This is the
#                         knob that makes the gate falsifiable:
#                         `PLOW_XGATE_EMIT_ENV=PLOW_QNORM_FUSE=1` re-enables the broken fold on
#                         gfx942 and the gate MUST go red.
#   PLOW_XGATE_HSACO      reuse an existing objects dir instead of building one from this tree.
#   PLOW_XGATE_STORED     serve this asset dir INSTEAD of emitting (the old gate's behaviour,
#                         kept only so the two can be run side by side).
#   PLOW_XGATE_PORT       default 8196. NEVER 8199 — a concurrent run's server has silently
#                         answered a whole A/B battery there, twice.
#   PLOW_XGATE_JOBS       hipcc parallelism for the object build (default 16).
#   PLOW_XGATE_NO_LOCK=1  skip the GPU lock (only when the caller already holds it).
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-$(date +%Y%m%d-%H%M%S)}"
PORT="${PLOW_XGATE_PORT:-8196}"
JOBS="${PLOW_XGATE_JOBS:-16}"
LOCK="${PLOW_GPU_LOCK:-/tmp/plow_gpu.lock}"
JOBDIR="${PLOW_XGATE_DIR:-/workspace/assets/gfx942/xgate-$NAME}"

HF_BF16=/workspace/models/gemma-4-12B-it
HF_FP8=/workspace/models/gemma-4-12B-it-fp8

PLOWC="$REPO/target/release/plowc"
PLOWRT="$REPO/target/release/plowrt"

# The prompts, and what a WORKING model says. Deliberately short and factual: the job is to
# separate 'Paris' from ',1___....1.111111111111', not to score the model.
PROMPTS=(
  "What is the capital of France?|(?i)paris"
  "What is the capital of Japan?|(?i)tokyo"
  "What is 17 times 23? Answer with the number only.|391"
)

say() { printf '%s\n' "$*"; }
die() { printf 'XGATE ABORT: %s\n' "$*" >&2; exit 2; }

mkdir -p "$JOBDIR" || die "cannot create $JOBDIR"

# ---------------------------------------------------------------- preflight
[ -x "$PLOWC" ]  || die "no plowc at $PLOWC — cargo build --release -p plowc"
[ -x "$PLOWRT" ] || die "no plowrt at $PLOWRT — cargo build --release -p plowrt --features hsa"
# Without `--features hsa` plowrt serves the CPU reference and decodes garbage, which would
# fail this gate for a reason that has nothing to do with the emitter. Refuse rather than lie.
# NOTE what this does and does not prove: it proves the FEATURE is compiled in, not that the
# library is loadable. The runtime half of the same question is the CPU-fallback check after
# the server starts, and that one is the load-bearing one.
if ! strings "$PLOWRT" 2>/dev/null | grep -q libhsa-runtime64; then
  die "plowrt was not built --features hsa (no libhsa-runtime64 string in the binary). It would
     serve the CPU reference and decode garbage, and this gate would go red for the wrong
     reason. Rebuild: cargo build --release -p plowrt --features hsa"
fi
[ -d "$HF_BF16" ] || die "missing $HF_BF16"
[ -d "$HF_FP8" ]  || die "missing $HF_FP8"

# ---------------------------------------------------------------- GPU lock
# The handler MUST exit. A trap that only cleans up releases the lock and then keeps running
# UNLOCKED, and later deletes whatever sibling took the lock next.
SPID=""; HELD=0
release() {
  if [ -n "$SPID" ]; then
    # SIGTERM, never SIGKILL. `kill -9` leaves the persistent megakernel RESIDENT and corrupts
    # every later run on this box.
    kill -TERM "$SPID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$SPID" 2>/dev/null; then
      say "WARNING: plowrt $SPID still up 60 s after SIGTERM. NOT escalating to SIGKILL — a"
      say "         resident megakernel poisons every later run. Investigate by hand."
    fi
    SPID=""
  fi
  if [ "$HELD" = 1 ]; then rmdir "$LOCK" 2>/dev/null; HELD=0; fi
}
on_signal() { release; exit 130; }
trap on_signal INT TERM
trap release EXIT

if [ "${PLOW_XGATE_NO_LOCK:-0}" != 1 ]; then
  say "waiting for $LOCK"
  # POLL FAST. The campaign's 20 s interval was written when two agents shared this box; with
  # five or more contending, a 20 s poll STARVES — a sibling that releases and immediately
  # re-takes the lock wins every round, and this script sat through several such cycles
  # unable to get a turn. A few seconds of jitter breaks the lockstep between pollers.
  until mkdir "$LOCK" 2>/dev/null; do sleep $(( 2 + RANDOM % 4 )); done
  HELD=1
  # Holding the lock is NOT sufficient: a sibling's server can outlive its lease. Spin on the
  # PROCESS NAME. `pgrep -f "plowrt serve"` matches this script's own launcher and spins
  # forever WHILE HOLDING THE LOCK — that has already cost one agent a session.
  #
  # `^plowrt`, NOT `-x plowrt`. Observed on this box 2026-08-08: a sibling was serving as
  # `/tmp/plowrt_stock` at 93% GPU and `pgrep -x plowrt` reported NOTHING, because -x demands an
  # exact comm match and the binary had been copied under another name. A prefix match catches
  # plowrt_stock / plowrt.old / plowrt-ab while still matching on the NAME, so it cannot
  # self-match this script the way -f does. The mkdir lock covered us that time; the check
  # should not have needed covering.
  while pgrep '^plowrt' >/dev/null 2>&1; do
    say "  foreign plowrt resident ($(pgrep -a '^plowrt' | head -1 | cut -c1-90)), waiting"
    sleep 10
  done
  USE=$(rocm-smi --showuse 2>/dev/null | grep -oE 'GPU use \(%\): *[0-9]+' \
        | grep -oE '[0-9]+$' | sort -rn | head -1)
  say "GPU lock held; peak rocm-smi use ${USE:-?}%"
  if [ "${USE:-0}" -gt 5 ] 2>/dev/null; then
    die "GPU reads ${USE}% busy with the lock held — something is resident (a SIGKILLed
     megakernel does exactly this). Do not trust anything measured from here."
  fi
fi

# ---------------------------------------------------------------- objects
if [ -n "${PLOW_XGATE_STORED:-}" ] && [ -z "${PLOW_XGATE_HSACO:-}" ]; then
  # A stored asset ships its own objects; building a set we then refuse to link (see the
  # relink guard below) would burn ~20 min for nothing.
  HSACO=""
  say "objects: none built — the stored asset brings its own"
elif [ -n "${PLOW_XGATE_HSACO:-}" ]; then
  HSACO="$PLOW_XGATE_HSACO"
  say "objects: REUSED $HSACO — NOT built from this tree. Say so in the transcript."
else
  HSACO="$JOBDIR/hsaco"
  say "objects: building from this tree -> $HSACO"
  # Outside `nix develop`: hipcc needs the system glibc.
  #
  # PLOW_HIPCC / PLOW_BUNDLER are resolved HERE rather than left to the build script. Its own
  # `ls -1 <three candidate paths> | head -1` runs under `set -o pipefail`, so on a box where
  # any candidate is missing — this one: /opt/rocm/lib/llvm/bin/clang-offload-bundler does not
  # exist — `ls` exits 2, the pipeline fails, and the build dies BEFORE printing a single line.
  # An empty log and exit 2 is what that looks like; do not go hunting for a compile error.
  HIPCC="${PLOW_HIPCC:-$(command -v hipcc || echo /opt/rocm/bin/hipcc)}"
  BUNDLER="${PLOW_BUNDLER:-}"
  if [ -z "$BUNDLER" ]; then
    for c in "${ROCM_PATH:-/opt/rocm}"/lib/llvm/bin/clang-offload-bundler \
             "${ROCM_PATH:-/opt/rocm}"/llvm/bin/clang-offload-bundler \
             /opt/rocm-7.2.4/lib/llvm/bin/clang-offload-bundler; do
      [ -x "$c" ] && { BUNDLER="$c"; break; }
    done
  fi
  [ -x "$HIPCC" ]   || die "no hipcc (set PLOW_HIPCC)"
  [ -x "$BUNDLER" ] || die "no clang-offload-bundler (set PLOW_BUNDLER)"
  say "  hipcc $HIPCC"
  say "  bundler $BUNDLER"
  env PLOW_HIPCC="$HIPCC" PLOW_BUNDLER="$BUNDLER" PLOW_OCC4=1 JOBS="$JOBS" \
    bash "$REPO/scripts/build_gfx942.sh" "$HSACO" \
    > "$JOBDIR/objects.log" 2>&1 \
    || { tail -20 "$JOBDIR/objects.log"; die "object build failed, see $JOBDIR/objects.log"; }
fi
[ -n "$HSACO" ] && { [ -d "$HSACO" ] || die "no objects at $HSACO"; }

# ---------------------------------------------------------------- the asset
if [ -n "${PLOW_XGATE_STORED:-}" ]; then
  ASSET="$PLOW_XGATE_STORED"
  say "asset: STORED $ASSET"
  say "  ^ this is the OLD gate's behaviour. It CANNOT see an emitter regression. Control only."
else
  ASSET="$JOBDIR/asset"
  mkdir -p "$ASSET/checkpoint" || die "cannot create $ASSET"
  say "asset: emitting FRESH at $(git -C "$REPO" rev-parse --short HEAD) -> $ASSET"
  say "  emit env: PLOW_FP8=1 PLOW_W8A8=1 ${PLOW_XGATE_EMIT_ENV:-<no extra>}"
  # shellcheck disable=SC2086
  env PLOW_FP8=1 PLOW_W8A8=1 PLOW_FP8_DIR="$HF_FP8" ${PLOW_XGATE_EMIT_ENV:-} \
    "$PLOWC" --emit devblob --hf-dir "$HF_BF16" \
    --gpu MI300X --arch gfx942 --num-gpus 1 --seq 128,512,1024 \
    --out "$ASSET/model.pkt" > "$JOBDIR/emit.log" 2>&1 \
    || { tail -20 "$JOBDIR/emit.log"; die "emit failed, see $JOBDIR/emit.log"; }
  # build.json is written next to model.pkt by the emit and MUST STAY THERE: it carries the
  # `backends.<arch>.requires` set that plowrt's arm-refusal chain reads. Several assets in
  # this campaign were assembled by copying model.pkt alone and had an INERT arm-refusal chain.
  [ -f "$ASSET/build.json" ] || die "emit produced no build.json next to model.pkt — the
     arm-refusal chain would be silently disabled"
  # `plowc --emit devblob` does NOT write weights.json, and `plowrt serve` opens it
  # unconditionally (bare `Io { path: .../weights.json, NotFound }` during load without it).
  # `network` is the model slug clients pass.
  cat > "$ASSET/weights.json" <<'JSON'
{
  "network": "gemma-4-12b-it",
  "gpu": "MI300X",
  "num_gpus": 1,
  "parallel": "tp",
  "weight_shared": false,
  "weight": null,
  "kv": null,
  "fusion": null,
  "buckets": [],
  "static_tensors": [],
  "static_tensors_file_emitted": false,
  "weight_tiling": null
}
JSON
  ln -sfn "$HF_BF16/tokenizer.json" "$ASSET/tokenizer.json"
  # fp8 weights are the fp8 twin shadowing the bf16 checkpoint in ONE directory.
  ln -sfn "$HF_BF16/model.safetensors" "$ASSET/checkpoint/bf16-model.safetensors"
  ln -sfn "$HF_FP8/model.safetensors"  "$ASSET/checkpoint/fp8-model.safetensors"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json; do
    ln -sfn "$HF_BF16/$f" "$ASSET/checkpoint/$f"
  done
fi
# NEVER relink a STORED asset's objects. `PLOW_XGATE_STORED` points at a shared directory that
# other agents' runs depend on; repointing its `hsaco` symlink would silently change what THEY
# serve. Set PLOW_XGATE_STORED_RELINK=1 only if you own that asset.
if [ -n "$HSACO" ] && { [ -z "${PLOW_XGATE_STORED:-}" ] || [ "${PLOW_XGATE_STORED_RELINK:-0}" = 1 ]; }; then
  ln -sfn "$HSACO" "$ASSET/hsaco" 2>/dev/null || true
else
  say "objects: using the stored asset's OWN hsaco ($(readlink -f "$ASSET/hsaco" 2>/dev/null))"
  say "         (not relinking a shared asset; PLOW_XGATE_HSACO ignored here)"
fi
BLOB_SHA=$(sha256sum "$ASSET/model.pkt" | cut -c1-16)
say "blob: sha256 $BLOB_SHA…  $(du -h "$ASSET/model.pkt" | cut -f1)"

# ---------------------------------------------------------------- serve
# PRE-FLIGHT: the port must be DEAD before we bind it. If anything answers here it is a
# sibling's server, and every answer this gate collects would be its answers, not ours. That
# co-tenancy failure has already corrupted two agents' arms in this campaign.
if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  die "something is ALREADY serving on :$PORT. Pick another port (never 8199) or wait it out.
     Never kill a sibling's server."
fi
LOG="$JOBDIR/serve.log"
# LAUNCH UNDER `nix develop`. Building plowrt `--features hsa` only compiles the dlopen IN; the
# library still has to be FOUND at runtime, and libhsa-runtime64.so.1 is reachable only via the
# LD_LIBRARY_PATH the dev shell exports. Launched bare, plowrt logs one WARN and serves the CPU
# reference — which answers this gate's prompts with `<unused87>` tokens. That is a green-looking
# harness bug wearing the costume of the emitter bug this gate exists to catch, and it cost a
# full pair of arms here before the serve log was read. §"the CPU-fallback check" below is the
# belt to this braces: even if the launch changes, the gate refuses to score a CPU run.
# BOTH halves are needed and NEITHER is sufficient. `nix develop` supplies ROCr's own
# dependencies (libstdc++, libelf, libdrm, numactl, zlib) — without it the dlopen gets as far as
# libhsa and then dies on `libstdc++.so.6: cannot open shared object file`. The dev shell then
# tries to add the ROCm lib dir itself, but it looks in /opt/rocm/lib, WHICH DOES NOT EXIST ON
# THIS BOX (/opt/rocm is a directory of `alternatives` symlinks; the libraries live under
# /opt/rocm/core-7.14/lib). So the dir is resolved here and appended inside the shell.
# Verified: `plowrt devices` reports 8 agents x 304 CU with both halves, and falls back to the
# CPU backend with either one missing.
ROCM_LIB="${PLOW_ROCM_LIB:-}"
if [ -z "$ROCM_LIB" ]; then
  for c in /opt/rocm/lib /opt/rocm/core-7.14/lib /opt/rocm-7.2.4/lib; do
    [ -e "$c/libhsa-runtime64.so.1" ] && { ROCM_LIB="$c"; break; }
  done
fi
[ -n "$ROCM_LIB" ] || die "no libhsa-runtime64.so.1 found (set PLOW_ROCM_LIB)"
say "  rocm lib: $ROCM_LIB"
NIX="${PLOW_NIX:-$(command -v nix || echo /root/.nix-profile/bin/nix)}"
if [ -x "$NIX" ]; then
  setsid env PLOW_FP8_DIR="$HF_FP8" ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
    "$NIX" develop "$REPO" --command bash -c \
    'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:'"$ROCM_LIB"'"; exec "$@"' _ \
    "$PLOWRT" serve --assets "$ASSET" --port "$PORT" \
    > "$LOG" 2>&1 &
else
  say "  WARNING: no nix at $NIX — launching plowrt bare. If libhsa is not on LD_LIBRARY_PATH"
  say "           this will serve the CPU reference; the check below will catch it."
  setsid env PLOW_FP8_DIR="$HF_FP8" ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
    "$PLOWRT" serve --assets "$ASSET" --port "$PORT" > "$LOG" 2>&1 &
fi
SPID=$!
say "serve: plowrt pid $SPID on :$PORT  (log $LOG)"
READY=0
for _ in $(seq 1 1800); do
  curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { READY=1; break; }
  kill -0 "$SPID" 2>/dev/null || { tail -30 "$LOG"; die "server died, see $LOG"; }
  sleep 1
done
[ "$READY" = 1 ] || { tail -30 "$LOG"; die "server never became ready"; }
# THE CPU-FALLBACK CHECK. `--features hsa` compiles the dlopen in; it does not make
# libhsa-runtime64.so.1 loadable. When the dlopen fails plowrt does not die — it prints a WARN
# and serves the CPU reference interpreter, which produces garbage on these blobs. A gate that
# scores that run reports a RED that has nothing to do with the emitter, which is precisely as
# useless as the stored-asset GREEN this script was written to replace.
if grep -qE "No GPU backend available|CPU reference backend active|running on the CPU reference" \
     "$LOG" 2>/dev/null; then
  say ""
  grep -E "HSA probe failed|No GPU backend|CPU reference" "$LOG" | head -4
  die "plowrt fell back to the CPU REFERENCE backend — it never touched the GPU. Any verdict
     from this run would be about the harness, not the emitter. Usual cause: launched outside
     \`nix develop\`, so libhsa-runtime64.so.1 is not on LD_LIBRARY_PATH."
fi
MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" \
        | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
[ -n "$MODEL" ] || die "no model id on :$PORT"
say "  model: $MODEL"
# CHECK THE model: LINE. A sibling's server on a shared port has already answered a whole A/B
# battery for one agent this campaign. The id alone does not prove the process is ours, so
# confirm the listener belongs to the group `setsid` just created.
# `ss` / `lsof` / `netstat` are all absent on this box, so this walks /proc directly. It must
# NOT degrade silently into "no check" — an unconfirmable owner is reported, loudly, as such.
OWNER=$(python3 "$REPO/scripts/gemma_xgate_portowner.py" "$PORT")
case "$OWNER" in
  NOSOCK|UNKNOWN|"")
    say "  WARNING: could not identify the pid listening on :$PORT ($OWNER). The port was"
    say "           verified free before launch and the GPU lock is held, so co-tenancy is"
    say "           unlikely — but this run is NOT owner-verified. Say so if you quote it." ;;
  *)
    PGID_OWNER=$(ps -o pgid= -p "$OWNER" 2>/dev/null | tr -d ' ')
    [ "$PGID_OWNER" = "$SPID" ] || die ":$PORT is held by pid $OWNER (pgid $PGID_OWNER), not by
     the server this gate started (pgid $SPID). A foreign server would have answered for us."
    say "  port owner: pid $OWNER, pgid $PGID_OWNER = ours" ;;
esac
case "$MODEL" in
  *gemma*) ;;
  *) die "served model is '$MODEL', not a gemma — wrong asset or wrong server" ;;
esac

# ---------------------------------------------------------------- the gate
FAILED=0
say ""
printf '%-46s %-10s %s\n' "prompt" "expect" "answer"
say "--------------------------------------------------------------------------------"
for entry in "${PROMPTS[@]}"; do
  Q="${entry%%|*}"; RE="${entry##*|}"
  BODY=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],
           "messages":[{"role":"user","content":sys.argv[2]}],
           "max_tokens":32,"temperature":0}))' "$MODEL" "$Q")
  A=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$BODY" \
      | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    print(d["choices"][0]["message"]["content"].strip().replace("\n", " "))
except Exception as e:
    print("<<no answer: %s>>" % e)')
  if printf '%s' "$A" | grep -Pq "$RE"; then V=ok; else V=FAIL; FAILED=1; fi
  printf '%-46.46s %-10.10s %-46.46s [%s]\n' "$Q" "$RE" "$A" "$V"
done
say "--------------------------------------------------------------------------------"

release
STAMP="blob $BLOB_SHA…  head $(git -C "$REPO" rev-parse --short HEAD)  emit-env '${PLOW_XGATE_EMIT_ENV:-<default>}'"
if [ "$FAILED" = 0 ]; then say "XGATE PASS  $STAMP"; exit 0; fi
say "XGATE FAIL  $STAMP"
exit 1
