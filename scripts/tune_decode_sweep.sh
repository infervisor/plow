#!/usr/bin/env bash
# tune_decode_sweep.sh — walk the DECODE knob grid and score every point by
# END-TO-END step_bench TPOT. Implements perf-data/tuner-decode-sweep-design.md.
#
#   scripts/tune_decode_sweep.sh [options]
#
# WHY BASH AND NOT `plowc tune`. The job is cross-toolchain subprocess
# orchestration and nothing else: nvcc must run under `env -i` (nix's CPATH
# points its host pass at glibc headers that fight the CUDA math headers),
# plowc/cargo must run INSIDE `nix develop`, and step_bench must run under
# `gpulease` with its own LD_LIBRARY_PATH. A Rust subcommand would be a
# `Command` wrapper over those three, would gain no types, and would have to
# invoke the very `plowc` binary it lives in to emit packets. `plowc tune`'s own
# module doc says it deliberately does not run benchmarks. The typed half — the
# record — is in Rust where types pay: `tunedb` (`tunedb-decode ingest|best`).
#
# WHAT IT SWEEPS. Two families of knob, which is why they must be swept jointly:
#
#   OBJECT knobs      -> one nvcc build each (~40 s)
#     PLOW_NV_FORCE_MINBLK  blocks/SM the object is compiled for
#     GV_UNROLL             dense GEMV weight streams per thread
#     GV_UNROLL_GLU         same, GLU arm
#     GV_MOE_UN             MoE expert arm streams
#     PLOW_MOE_DOWN_SG      MoE-down lane-split sub-groups
#     GV_MM_MAX             widest batched GEMV rung  <- ONLY meaningful vs --batch
#     PLOW_NV_FA_*          flash decode arm (wpr / gf / gf_full / kun)
#     --extra-defines       anything else, as NAME=VAL sets (fp8 arms live here)
#
#   PACKET knobs      -> one `plowc --emit devblob` each (~60 s)
#     --n-cu                grid width; MUST match FORCE_MINBLK (132 <-> 1, 264 <-> 2)
#     PLOW_NS_ABS           flash decode split count
#     PLOW_NS_FULL_ABS      same, FULL-attention layers only
#     PLOW_DECODE_BATCH     decode slots  <- see below, this is NOT a run-time argument
#
# `(FORCE_MINBLK, --n-cu)` is a PAIR — the engine refuses a packet emitted for
# 132 blocks against an object that reaches 264 — so it is swept as one axis
# (--occ "1:132 2:264"). Everything else crosses freely.
#
# WHY --batch IS AN AXIS AND NOT A CONSTANT. It was a hardcoded `1` in the
# step_bench invocation until px15, which made GV_MM_MAX unrepresentable: the
# batched GEMV loads a weight row once and dots it against MM activation rows,
# so B costs ceil(B/GV_MM_MAX) weight passes and the knob's optimum INVERTS with
# batch. op_gemm.cuh's own ladder: at B=8, =8 gives 355 tok/s and =16 gives 294;
# at B=16 it is 387 vs 520. A batch-blind sweep answers one of those and calls
# it the answer, which is how an asset shipped =16 while serving B=8
# (perf-data/px10-batched-decode.md: -19.4% at 131k, -33.8% at 1k).
#
# BATCH IS A PACKET AXIS, NOT A RUN ARGUMENT. `plowc --batch` is the PREFILL
# bucket list; the decode batch is `PLOW_DECODE_BATCH` (devgen sizes the KV
# cache, activations, GEMV M, flash n_batch and per-sequence argmax from it and
# bakes the result into the blob). So each batch needs its OWN packet, and
# step_bench's slot argument only selects among slots the packet already has —
# `slots = want.min(engine.batch())` silently clamps otherwise. Getting this
# wrong files a batch-1 timing under B=8, which is worse than not measuring.
#
# SCORING IS END-TO-END, DELIBERATELY. The campaign's central methodological
# finding (perf-data/gemma26b-h100-gemv-mlp.md, round 2) is that the isolated
# microbench DISAGREES with the megakernel: `gemv_lab_h100.cu` says row-blocking
# wins 1.4x on every decode shape, and in context it loses. So `gemv_lab` is a
# PRUNER and step_bench TPOT is the SCORER. This script only ever runs the
# scorer; nothing here reads a microbench number.
#
# CONTENTION. Every GPU command runs under `gpulease`, and before each one the
# script waits for `memory.used` to fall under --mem-idle. gpulease's own audit
# cannot see processes outside our PID namespace, and a ~52 GB foreign holder
# has already invalidated numbers on this box, so the VRAM check is the real
# gate. gpulease rc=76 ("completed but contended") discards the sample outright.
#
# Every row records the VRAM its WORST rep started at (`vram_before_mib`) and
# whether that cleared --mem-idle (`uncontended`). Those are MEASURED, never
# assumed: a card that is never idle is a fact about the run and belongs in the
# artifact, not in a footnote someone has to remember. `--wait-max` +
# `--mem-run` opt into taking caveated samples when a holder simply never
# leaves; by default the two thresholds coincide and no caveated row is ever
# written.
#
# OUTPUT. One JSON object per (config, ctx) appended to --results, with every
# rep's mean_ms, the median, the object's register count, and the cubin/packet
# sha256. Re-running skips rows already present, so an interrupted sweep
# resumes. Feed the file to `tunedb-decode ingest`.
#
# EXAMPLE — the full grid from the design doc:
#   scripts/tune_decode_sweep.sh \
#     --model /workspace/models/gemma-4-26B-A4B-it \
#     --work /dev/shm/plowtune/sweep \
#     --results perf-data/tune-decode-h100-26b-bf16.jsonl \
#     --occ "1:132 2:264" --ns-abs "8 16 32 48" \
#     --gv-unroll "4 8" --gv-moe-un "2 4" --moe-down-sg "4 8" \
#     --ctx "1024 8192 32768" --reps 3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL=/workspace/models/gemma-4-26B-A4B-it
# What the asset dir's `checkpoint` points at. Usually the model dir, but an
# fp8-weight packet needs the checkpoint carrying the fp8 tensor file while the
# record should still name the model. Both default to --model.
CHECKPOINT=""
MODEL_NAME=""
WORK=/dev/shm/plowtune/sweep
RESULTS="$ROOT/perf-data/tune-decode-h100-26b-bf16.jsonl"
PLOWC=""
STEP_BENCH=""
OCC="1:132 2:264"
NS_ABS="8 16 32 48"
GV_UNROLL="8"
GV_UNROLL_GLU="0"          # 0 = leave the source default (4)
GV_MOE_UN="2"
MOE_DOWN_SG="4"
# Widest batched GEMV rung. "" = leave the source default (8) and emit no -D, so
# a sweep that ignores this axis builds objects byte-identical to a pre-px15 one.
GV_MM_MAX=""
# Decode slots per step. 1 reproduces every pre-px15 row exactly.
BATCHES="1"
# Extra -D sets, one per arm, '+'-joined so a shell word is one arm:
#   --extra-defines "none PLOW_FP8_LD16=1 PLOW_FP8_LD16=1+PLOW_FP8_FAST=1"
# The literal `none` arm adds nothing. This is the README's "a whole new op
# family rides extra_defines" path: no schema growth, old rows still load.
EXTRA_DEFINES_ARMS="none"
# ---- flash decode arm ----------------------------------------------------
# 0 = leave the source default, EXCEPT where the source default and the shipped
# value differ (FA_WPR ships 1, source defaults 0), which is why these are
# explicit lists rather than a sentinel.
FA_WPR=""                  # PLOW_NV_FA_WPR   {0,1}   warp-per-row score phase
FA_GF=""                   # PLOW_NV_FA_GF    {2,4}   GQA fusion, sliding layers
FA_GF_FULL=""              # PLOW_NV_FA_GF_FULL {4,8} GQA fusion, FULL-attention layers
FA_KUN=""                  # PLOW_NV_FA_KUN   {1,2,4} K-stream pre-issue depth
NS_FULL_ABS="0"            # packet; nsplit for the FULL layers only. 0 = emitter default.
# Defines every object in this sweep carries. The knob axes are deltas on top;
# a later -D wins, so this can also override a value the build script hardcodes.
BASE_DEFINES=""
# Opcode-body ablation mask. Non-zero builds a TWIN object per config with the
# op compiled out, so each row carries TPOT(full), TPOT(ablated), and the
# difference — that op's true wall-clock contribution at the shipped grid.
# FLASH_DECODE is opcode 12 (1<<12 = 4096), FLASH_MERGE 13 (8192).
ABLATE_LO=0
ABLATE_HI=0
CTXS="1024 8192 32768"
# 3 for the screening grid — the campaign's own rule, and what the raw
# perf-data artifact keeps. Note that `tunedb` will NOT accept a 3-rep row:
# `Stats::from_samples` refuses fewer than 5 samples at construction and there
# is deliberately no flag that lowers the floor. So the winners get re-run with
# --reps 5 and only those become records. Screening wide and confirming narrow
# is the same stage-A/stage-B split the design doc draws, applied to reps.
REPS=3
STEPS=128
# The packet's max_ctx must exceed the largest measured ctx PLUS the timed
# steps: step_bench reserves ctx+steps+1 for the slot, so a 32768-max-ctx packet
# cannot run ctx=32768. Emitting wider is measurement-neutral (round 11 measured
# ctx=1024 identical on a 131072-max-ctx packet) and costs only KV reservation.
MAXCTX=65536
DTYPE=bf16
# Board identity written on every row. EMPTY means "derive from --arch", which is
# the only safe default: these used to be hardcoded to the H100 with no flag to
# change them, so `--arch sm120a` built and measured an sm_120a object and then
# filed it under `nvidia/sm_90a/h100-nvl`. That is not a mislabelled row, it is a
# 5090 measurement ranked against H100 measurements inside one cell.
GPU=""
HARDWARE=""
# Which interpreter object to build and load. The knob VALUES are portable; the
# build script, the cubin filenames the engine looks for, and the mangled kernel
# symbol `cuobjdump` reports registers against are NOT — so this is one switch
# rather than three chances to half-port a sweep to a second GPU.
ARCH=sm90a
CAMPAIGN="tuner-decode-sweep"
MEM_IDLE=2000              # MiB; at or below this the card is verifiably ours
MEM_RUN=""                 # MiB; hard ceiling to attempt a run at all (default = MEM_IDLE)
WAIT_MAX=0                 # s to wait for MEM_IDLE before falling back to MEM_RUN; 0 = forever
WAIT_S=60
SETTLE_S=6                 # s to let our own VRAM drain between reps
# Max (max-min)/median a row may show before it is called unstable. Reps on a
# quiet card land inside 0.001-0.004 on this step; an order of magnitude past
# that is a disturbed run, not a slow configuration. A configuration cannot make
# its own timing erratic -- a neighbour can. Recorded on the row, and
# `tunedb-decode ingest` refuses to qualify it.
MAX_SPREAD=0.01
LABEL_PREFIX=tds
DRY=0

usage() { sed -n '2,60p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --model-name) MODEL_NAME="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --results) RESULTS="$2"; shift 2;;
    --plowc) PLOWC="$2"; shift 2;;
    --step-bench) STEP_BENCH="$2"; shift 2;;
    --occ) OCC="$2"; shift 2;;
    --ns-abs) NS_ABS="$2"; shift 2;;
    --gv-unroll) GV_UNROLL="$2"; shift 2;;
    --gv-unroll-glu) GV_UNROLL_GLU="$2"; shift 2;;
    --gv-moe-un) GV_MOE_UN="$2"; shift 2;;
    --moe-down-sg) MOE_DOWN_SG="$2"; shift 2;;
    --gv-mm-max) GV_MM_MAX="$2"; shift 2;;
    --batch) BATCHES="$2"; shift 2;;
    --extra-defines) EXTRA_DEFINES_ARMS="$2"; shift 2;;
    --arch) ARCH="$2"; shift 2;;
    --hardware) HARDWARE="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --fa-wpr) FA_WPR="$2"; shift 2;;
    --fa-gf) FA_GF="$2"; shift 2;;
    --fa-gf-full) FA_GF_FULL="$2"; shift 2;;
    --fa-kun) FA_KUN="$2"; shift 2;;
    --ns-full-abs) NS_FULL_ABS="$2"; shift 2;;
    --base-defines) BASE_DEFINES="$2"; shift 2;;
    --ablate-lo) ABLATE_LO="$2"; shift 2;;
    --ablate-hi) ABLATE_HI="$2"; shift 2;;
    --ctx) CTXS="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --max-ctx) MAXCTX="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --campaign) CAMPAIGN="$2"; shift 2;;
    --mem-idle) MEM_IDLE="$2"; shift 2;;
    --mem-run) MEM_RUN="$2"; shift 2;;
    --wait-max) WAIT_MAX="$2"; shift 2;;
    --settle) SETTLE_S="$2"; shift 2;;
    --max-spread) MAX_SPREAD="$2"; shift 2;;
    --label) LABEL_PREFIX="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) usage 0;;
    *) echo "unknown option $1" >&2; usage 2;;
  esac
done

# Everything arch-specific, resolved once. `KSYM` is the mangled decode-kernel
# name cuobjdump prints registers for; getting it wrong yields an empty register
# column rather than an error, so it is derived here and not guessed per-call.
case "$ARCH" in
  sm90a)  BUILD_SH=build_sm90a_cubin.sh; CUBIN=interp_sm90a.cubin
          CUBIN_PF=interp_sm90a_pf.cubin; KSYM='interp_sm90a11PlowProgram'
          DEF_HW="nvidia/sm_90a/h100-nvl"; DEF_GPU="H100 NVL";;
  sm120a) BUILD_SH=build_sm120_cubin.sh; CUBIN=interp_sm120.cubin
          CUBIN_PF=interp_sm120_pf.cubin; KSYM='interp_sm12011PlowProgram'
          DEF_HW="nvidia/sm_120a/rtx-5090"; DEF_GPU="RTX 5090";;
  *) echo "unknown --arch $ARCH (want sm90a|sm120a)" >&2; exit 2;;
esac
[ -n "$HARDWARE" ] || HARDWARE="$DEF_HW"
[ -n "$GPU" ] || GPU="$DEF_GPU"

# The board must match the object. `DecodeCell.hardware` is the top of the key,
# so a wrong value does not mislabel a row — it ranks a 5090 measurement against
# H100 measurements inside one cell and reports the faster BOARD as the better
# knob set. Cheap to check, so check.
SMS="$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null >/dev/null && \
       nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
case "$HARDWARE:$SMS" in
  *sm_120a*:*5090*|*sm_90a*:*H100*) ;;
  *:"") echo "warn: cannot read the GPU name; --hardware $HARDWARE unverified" >&2;;
  *) echo "FATAL: --hardware $HARDWARE does not match the installed GPU ($SMS)." >&2
     echo "       Pass the right --hardware, or the rows will be ranked against another board." >&2
     exit 2;;
esac

[ -n "$PLOWC" ] || PLOWC="$ROOT/target/release/plowc"
[ -n "$STEP_BENCH" ] || STEP_BENCH="$ROOT/target/release/examples/step_bench"
[ -n "$MEM_RUN" ] || MEM_RUN="$MEM_IDLE"
[ -n "$CHECKPOINT" ] || CHECKPOINT="$MODEL"
[ -n "$MODEL_NAME" ] || MODEL_NAME="$(basename "$MODEL")"
[ -x "$PLOWC" ] || { echo "no plowc at $PLOWC (--plowc)" >&2; exit 2; }
[ -x "$STEP_BENCH" ] || { echo "no step_bench at $STEP_BENCH (--step-bench)" >&2; exit 2; }
[ -d "$MODEL" ] || { echo "no model dir $MODEL" >&2; exit 2; }

# The lease helper lives in the repo. Calling it bare assumed it was on PATH,
# which it is on the H100 box and is NOT here — and a bare `gpulease` that is
# missing does not fail the sweep, it fails every RUN, so the sweep completes
# having recorded nothing and looks like a grid with no trustworthy samples.
GPULEASE="$(command -v gpulease || echo "$ROOT/perf-data/harness/gpulease")"
[ -x "$GPULEASE" ] || { echo "no gpulease at $GPULEASE" >&2; exit 2; }

# WHICH libcuda. `device::cuda` tries /usr/local/cuda/compat BEFORE the distro
# path — right on a box whose toolkit outruns its kernel driver, wrong on one
# where it is the other way round. On the RTX 5090 box compat ships 580.167.08
# against a 580.159.03 kernel driver and every run dies with
# CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE before it measures anything.
# LD_LIBRARY_PATH cannot fix it: /usr/lib/x86_64-linux-gnu on the path shadows
# nix's glibc and the binary dies in the loader instead.
if [ -z "${PLOW_LIBCUDA:-}" ] && [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d ' ')"
  [ -e "/usr/local/cuda/compat/libcuda.so.$drv" ] || {
    export PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1
    echo "libcuda: pinned $PLOW_LIBCUDA (driver $drv; compat build differs)" >&2
  }
fi

mkdir -p "$WORK/cubin" "$WORK/pkt" "$WORK/assets" "$WORK/log" "$(dirname "$RESULTS")"
touch "$RESULTS"

TOOLCHAIN="$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/cuda-\1/p' | head -1)"
IMPL="$(cat "$ROOT"/runtime/nvidia/interp_sm120.cu "$ROOT"/runtime/nvidia/interp_sm90a.cu \
        "$ROOT"/runtime/nvidia/op_gemm.cuh "$ROOT"/runtime/nvidia/op_moe.cuh \
        "$ROOT"/runtime/nvidia/op_attention.cuh 2>/dev/null | sha256sum | cut -c1-16)"

echo "root       : $ROOT"
echo "model      : $MODEL"
echo "work       : $WORK"
echo "results    : $RESULTS"
echo "toolchain  : $TOOLCHAIN   impl $IMPL"
echo "arch       : $ARCH -> $BUILD_SH / $CUBIN"
echo "grid       : occ[$OCC] ns_abs[$NS_ABS] unroll[$GV_UNROLL] glu[$GV_UNROLL_GLU]"
echo "             moe_un[$GV_MOE_UN] sg[$MOE_DOWN_SG] mm_max[${GV_MM_MAX:-default}]"
echo "             extra[$EXTRA_DEFINES_ARMS] batch[$BATCHES] ctx[$CTXS] reps=$REPS"
echo

# ---------------------------------------------------------------- utilities

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }

# Block until the card is actually idle. gpulease serialises US; it cannot evict
# a holder outside our PID namespace, and a run started against 52 GB of
# somebody else's resident weights is either an OOM or a contended number.
#
# `--wait-max` bounds the patience. Past it, a run is still ATTEMPTED if VRAM is
# under `--mem-run`, but the row it produces is marked `"uncontended": false`
# and carries the VRAM it started at. That is the honest handling of a holder
# that never drains: a caveated number beats no number, and an uncaveated one is
# worse than both. `--mem-run` defaults to `--mem-idle`, so the default build
# never takes a caveated sample.
#
# Echoes the VRAM the run is about to start at; returns 1 if that is above
# MEM_RUN, meaning do not run at all.
wait_for_idle() {
  local used waited=0
  while :; do
    used="$(vram_used)"
    if [ "${used:-999999}" -le "$MEM_IDLE" ]; then echo "${used}"; return 0; fi
    if [ "$WAIT_MAX" != "0" ] && [ "$waited" -ge "$WAIT_MAX" ]; then
      if [ "${used:-999999}" -le "$MEM_RUN" ]; then
        echo "  [wait] gave up after ${waited}s at ${used} MiB — running CAVEATED" >&2
        echo "${used}"; return 0
      fi
      echo "${used}"; return 1
    fi
    # Never sleep past the deadline: polling every WAIT_S but only checking the
    # give-up condition afterwards made --wait-max 10 cost a full WAIT_S per run.
    local step="$WAIT_S"
    if [ "$WAIT_MAX" != "0" ] && [ "$((WAIT_MAX - waited))" -lt "$step" ]; then
      step="$((WAIT_MAX - waited))"
    fi
    [ "$step" -gt 0 ] || step=1
    echo "  [wait] memory.used=${used} MiB > ${MEM_IDLE} — GPU is not ours; sleeping ${step}s" >&2
    sleep "$step"
    waited=$((waited + step))
  done
}

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# --------------------------------------------------------------- cubin build
# Keyed by the SHA of the full define string, not by a hand-built name. Adding a
# knob family (this is the second: GEMM, then flash) then costs nothing here and
# cannot collide with an existing cache entry the way a name scheme silently can.
defines_for() {  # gemm knobs, flash knobs, extra arm -> echoes the -D string
  local mb="$1" un="$2" glu="$3" mun="$4" sg="$5" wpr="$6" gf="$7" gff="$8" kun="$9"
  local mm="${10}" xarm="${11}"
  local d="$BASE_DEFINES -DPLOW_NV_FORCE_MINBLK=$mb -DGV_UNROLL=$un -DGV_MOE_UN=$mun -DPLOW_MOE_DOWN_SG=${sg}u"
  [ "$glu" = "0" ] || d="$d -DGV_UNROLL_GLU=$glu"
  # GV_MM_MAX=0 is not "the default", it is a gemv_walk with no instantiated
  # rung. Absence is the only spelling of "leave it alone".
  [ -z "$mm" ] || d="$d -DGV_MM_MAX=$mm"
  # Flash knobs are emitted only when asked for, so a sweep that does not name
  # them builds byte-identical objects to one from before they existed.
  [ -z "$wpr" ] || d="$d -DPLOW_NV_FA_WPR=$wpr"
  [ -z "$gf"  ] || d="$d -DPLOW_NV_FA_GF=$gf"
  [ -z "$gff" ] || d="$d -DPLOW_NV_FA_GF_FULL=$gff"
  [ -z "$kun" ] || d="$d -DPLOW_NV_FA_KUN=$kun"
  if [ -n "$xarm" ] && [ "$xarm" != "none" ]; then
    local one; for one in ${xarm//+/ }; do d="$d -D$one"; done
  fi
  echo "$d"
}

# The extra arm as a JSON object, for the row's `extra_defines` map. Empty arm
# -> `{}`, which `tunedb` skips serialising, so an unnamed axis leaves no trace.
extra_json() {
  local xarm="$1" one first=1
  if [ -z "$xarm" ] || [ "$xarm" = "none" ]; then echo '{}'; return; fi
  printf '{'
  for one in ${xarm//+/ }; do
    [ "$first" = 1 ] || printf ','
    first=0
    printf '"%s":"%s"' "${one%%=*}" "${one#*=}"
  done
  printf '}'
}

# $1 defines, $2 "" | "abl"  -> echoes dir
build_cubin() {
  local defs="$1" kind="${2:-}"
  local key; key="$(printf '%s|%s' "$defs" "$kind" | sha256sum | cut -c1-16)"
  local dir="$WORK/cubin/$key"
  if [ -f "$dir/$CUBIN" ] && [ -f "$dir/$CUBIN_PF" ]; then
    echo "$dir"; return 0
  fi
  if [ "$kind" = "abl" ]; then
    defs="$defs -DPLOW_NV_ABLATE_LO=${ABLATE_LO}ull -DPLOW_NV_ABLATE_HI=${ABLATE_HI}ull"
  fi
  mkdir -p "$dir"
  printf '%s\n' "$defs" > "$dir/defines"
  echo "  [build] ${key} ${kind}  ($defs)" >&2
  if [ "$DRY" = "1" ]; then echo "$dir"; return 0; fi
  PLOW_ROOT="$ROOT" PLOW_EXTRA_DEFINES="$defs" \
    "$ROOT/scripts/$BUILD_SH" "$dir/$CUBIN" >"$WORK/log/build_$key.log" 2>&1 \
    || { echo "FATAL: cubin build failed for $key - see $WORK/log/build_$key.log" >&2; exit 1; }
  registers_of "$dir/$CUBIN" > "$dir/registers" || true
  echo "$dir"
}

# Registers of the decode megakernel. Recorded next to every timing because a
# knob that buys ILP by raising pressure for every other arm is not obviously a
# win - and because the campaign's "255 registers explains the regression"
# story was later retracted, so this is evidence to keep, not a verdict.
registers_of() {
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin cuobjdump -res-usage "$1" 2>/dev/null \
    | awk -v sym="$KSYM" '$0 ~ sym {f=1}
           f && /REG:/ { for(i=1;i<=NF;i++) if($i ~ /^REG:/){ sub("REG:","",$i); print $i; exit } }'
}

# -------------------------------------------------------------- packet emit
# Key: every PACKET knob. `--n-cu` and PLOW_NS_ABS are the only two the design
# leaves on the packet; both already have ctx in scope at emit time.
emit_packet() {  # $1 n_cu  $2 ns_abs  $3 ns_full_abs  $4 batch  $5 gf_full -> echoes dir
  local ncu="$1" ns="$2" nsf="$3" bsz="$4" gff="$5"
  # Batch is BAKED IN, so it keys the packet. Reusing a B=1 packet for a B=8
  # point does not measure B=8, it measures B=1 under a B=8 label.
  # GF_FULL likewise: the emitter sizes nsplit from `n_grp = heads/GF_FULL`, so
  # a packet built for one GF against an object built for another measures the
  # mismatch. It is a PAIR, like (FORCE_MINBLK, --n-cu).
  local key="ncu${ncu}_ns${ns}_nsf${nsf}_b${bsz}_gff${gff:-d}"
  local dir="$WORK/pkt/$key"
  if [ -f "$dir/model.pkt" ]; then echo "$dir"; return 0; fi
  mkdir -p "$dir"
  echo "  [emit ] $key" >&2
  if [ "$DRY" = "1" ]; then echo "$dir"; return 0; fi
  local env_kv=(PLOW_UNISEG=1 PLOW_DECODE_BATCH="$bsz")
  [ -z "$gff" ] || env_kv+=(PLOW_FA_GF_FULL="$gff")
  [ "$DTYPE" = "fp8" ] && env_kv+=(PLOW_FP8=1)
  [ "$ns" = "0" ] || env_kv+=(PLOW_NS_ABS="$ns")
  # NS_FULL_ABS touches ONLY the 5 full-attention layers. Those are the layers
  # that read the whole context while the 25 sliding ones are window-capped, so
  # this is the ctx-sensitive half of the split and is keyed separately.
  [ "$nsf" = "0" ] || env_kv+=(PLOW_NS_FULL_ABS="$nsf")
  local dtflag=()
  [ "$DTYPE" = "fp8" ] && dtflag=(--weight-dtype fp8)
  env "${env_kv[@]}" "$PLOWC" \
      --hf-dir "$CHECKPOINT" --emit devblob --max-ctx "$MAXCTX" --n-cu "$ncu" \
      "${dtflag[@]}" --out "$dir" \
      >"$WORK/log/emit_$key.log" 2>&1 \
    || { echo "FATAL: packet emit failed for $key - see $WORK/log/emit_$key.log" >&2; exit 1; }
  echo "$dir"
}

# ------------------------------------------------------------- asset assembly
# All symlinks: the checkpoint is 52 GB and the tokenizer 32 MB, and this box
# has no disk to spare for either.
assets_for() {  # $1 cubin dir  $2 pkt dir  $3 tag -> echoes dir
  local cdir="$1" pdir="$2" tag="$3"
  local dir="$WORK/assets/$tag"
  mkdir -p "$dir"
  ln -sfn "$CHECKPOINT" "$dir/checkpoint"
  ln -sfn "$MODEL/tokenizer.json" "$dir/tokenizer.json"
  ln -sfn "$pdir/model.pkt" "$dir/model.pkt"
  ln -sfn "$cdir/$CUBIN" "$dir/$CUBIN"
  ln -sfn "$cdir/$CUBIN_PF" "$dir/$CUBIN_PF"
  echo "$dir"
}

# ------------------------------------------------------------------ one point
# Returns the mean_ms of one step_bench run, or empty if the run was contended
# or failed. Never returns a number it does not trust.
# Echoes "<mean_ms> <vram_before_mib>", or just "<vram_before_mib>" when there
# is no trustworthy sample. Both travel on stdout because the caller reads this
# through a command substitution, and a subshell cannot set a variable in its
# parent — which silently zeroed the provenance field the first time.
run_once() {  # $1 assets  $2 ctx  $3 label  $4 batch -> echoes "ms vram" | "vram"
  local adir="$1" ctx="$2" label="$3" bsz="${4:-1}" rc pre
  local log="$WORK/log/run_${label}.log"
  local vramf="$WORK/log/vram_${label}"

  # A light pre-check so we do not queue behind a holder that never leaves; the
  # AUTHORITATIVE reading is taken after the lease is held, below.
  set +e
  wait_for_idle >/dev/null
  set -e

  # Read memory.used INSIDE the lease. Reading it before means another branch's
  # in-flight run looks like a permanent holder and the point is skipped, when
  # what should happen is that gpulease serialises us and we measure once the
  # card is actually ours. It also means the number recorded as provenance is
  # the one that was true while WE ran, not one from before we had the card.
  set +e
  "$GPULEASE" "$label" bash -c '
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " " > "$1"
      shift; exec "$@"
    ' _ "$vramf" "$STEP_BENCH" "$adir" "$bsz" "$ctx" "$STEPS" >"$log" 2>&1
  rc=$?
  set -e
  pre="$(tr -cd '0-9' < "$vramf" 2>/dev/null || true)"
  pre="${pre:-999999}"

  # rc=76 is gpulease's "completed but contended" — a detected foreign process
  # during the run. That is a failed measurement, not a caveated one.
  if [ "$rc" -ne 0 ]; then
    echo "  [run  ] $label rc=$rc DISCARDED (see $log)" >&2
    echo "$pre"; return 0
  fi
  if [ "$pre" -gt "$MEM_RUN" ]; then
    echo "  [run  ] $label ran at ${pre} MiB, over --mem-run ${MEM_RUN} — DISCARDED" >&2
    echo "$pre"; return 0
  fi
  # Let our OWN allocation drain before the next rep reads memory.used, or the
  # driver mistakes its previous process for a foreign holder.
  sleep "$SETTLE_S"
  local ms; ms="$(sed -n 's/.*RAW_STEP .*mean_ms=\([0-9.]*\).*/\1/p' "$log" | head -1)"
  # The batch step_bench ACTUALLY ran, read back off its own line rather than
  # assumed from the argument. `slots = want.min(engine.batch())` clamps
  # SILENTLY, so a packet emitted for 1 slot answers a --batch 8 request with a
  # batch-1 number and nothing in the log says so. Recording the request instead
  # of the result is precisely the provenance loss the batch axis exists to fix.
  local got; got="$(sed -n 's/.*RAW_STEP slots=\([0-9]*\).*/\1/p' "$log" | head -1)"
  if [ -n "$ms" ]; then echo "$ms $pre ${got:-0}"; else echo "$pre"; fi
}

median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END{ if(NR==0) exit 1; print a[int((NR+1)/2)] }'; }

# ---------------------------------------------------------------------- sweep
# An unset flash axis contributes a single empty value, which defines_for() then
# omits entirely -- so naming no flash knob reproduces the pre-flash sweep byte
# for byte rather than pinning the source defaults explicitly.
list_or_blank() { if [ -z "$1" ]; then echo ""; else echo "$1"; fi; }
WPRS="$(list_or_blank "$FA_WPR")"; GFS="$(list_or_blank "$FA_GF")"
GFFS="$(list_or_blank "$FA_GF_FULL")"; KUNS="$(list_or_blank "$FA_KUN")"

# `for x in $EMPTY` iterates zero times, which would drop the whole grid; a
# single-element list holding the empty string is what "leave it alone" needs.
[ -n "$WPRS" ] || WPRS='""'
[ -n "$GFS" ]  || GFS='""'
[ -n "$GFFS" ] || GFFS='""'
[ -n "$KUNS" ] || KUNS='""'
MMS="$(list_or_blank "$GV_MM_MAX")"; [ -n "$MMS" ] || MMS='""'
unquote() { [ "$1" = '""' ] && echo "" || echo "$1"; }

total=0
for occ in $OCC; do for ns in $NS_ABS; do for nsf in $NS_FULL_ABS; do
for un in $GV_UNROLL; do for glu in $GV_UNROLL_GLU; do for mun in $GV_MOE_UN; do
for sg in $MOE_DOWN_SG; do for w in $WPRS; do for g in $GFS; do for gf in $GFFS; do
for k in $KUNS; do for mm in $MMS; do for xa in $EXTRA_DEFINES_ARMS; do
for b in $BATCHES; do for ctx in $CTXS; do
  total=$((total+1))
done; done; done; done; done; done; done; done; done; done; done; done; done; done; done
echo "grid points: $total"
[ "$ABLATE_LO" != "0" ] || [ "$ABLATE_HI" != "0" ] && \
  echo "ablation: LO=$ABLATE_LO HI=$ABLATE_HI (each point also runs an ablated twin)"

done_n=0
for occ in $OCC; do
  mb="${occ%%:*}"; ncu="${occ##*:}"
  for un in $GV_UNROLL; do for glu in $GV_UNROLL_GLU; do for mun in $GV_MOE_UN; do
  for sg in $MOE_DOWN_SG; do
  for wq in $WPRS; do for gq in $GFS; do for gfq in $GFFS; do for kq in $KUNS; do
  for mmq in $MMS; do for xa in $EXTRA_DEFINES_ARMS; do
    w="$(unquote "$wq")"; g="$(unquote "$gq")"; gf="$(unquote "$gfq")"; k="$(unquote "$kq")"
    mm="$(unquote "$mmq")"
    defs="$(defines_for "$mb" "$un" "$glu" "$mun" "$sg" "$w" "$g" "$gf" "$k" "$mm" "$xa")"
    xjson="$(extra_json "$xa")"
    cdir="$(build_cubin "$defs")"
    [ -s "$cdir/registers" ] || registers_of "$cdir/$CUBIN" > "$cdir/registers"
    regs="$(tr -cd '0-9' < "$cdir/registers" 2>/dev/null || true)"
    [ -n "$regs" ] || regs=null
    csha="$(sha256sum "$cdir/$CUBIN" 2>/dev/null | cut -c1-16 || true)"
    abldir=""
    if [ "$ABLATE_LO" != "0" ] || [ "$ABLATE_HI" != "0" ]; then
      abldir="$(build_cubin "$defs" abl)"
    fi
    for ns in $NS_ABS; do for nsf in $NS_FULL_ABS; do
    for bsz in $BATCHES; do
      # One packet PER BATCH: PLOW_DECODE_BATCH is baked in at emit.
      pdir="$(emit_packet "$ncu" "$ns" "$nsf" "$bsz" "$gf")"
      psha="$(sha256sum "$pdir/model.pkt" 2>/dev/null | cut -c1-16 || true)"
      cfg="mb${mb}_ncu${ncu}_un${un}_glu${glu}_mun${mun}_sg${sg}_mm${mm:-d}_ns${ns}_nsf${nsf}"
      cfg="${cfg}_wpr${w:-d}_gf${g:-d}_gff${gf:-d}_kun${k:-d}_x${xa}"
      adir="$(assets_for "$cdir" "$pdir" "${cfg}_b${bsz}")"
      for ctx in $CTXS; do
        done_n=$((done_n+1))
        if grep -qF "\"config\":\"$cfg\",\"ctx\":$ctx,\"batch\":$bsz," "$RESULTS" 2>/dev/null; then
          echo "[$done_n/$total] $cfg ctx=$ctx B=$bsz - already recorded, skipping"
          continue
        fi
        echo "[$done_n/$total] $cfg ctx=$ctx B=$bsz"
        [ "$DRY" = "1" ] && continue
        samples=(); worst_vram=0; got_b=0
        for r in $(seq 1 "$REPS"); do
          read -r ms pre got <<<"$(run_once "$adir" "$ctx" "${LABEL_PREFIX}-${cfg}-c${ctx}-b${bsz}-r${r}" "$bsz")"
          if [ -z "${pre:-}" ]; then pre="$ms"; ms=""; fi
          [ "${pre:-0}" -gt "$worst_vram" ] && worst_vram="${pre:-0}"
          [ -n "$ms" ] && { samples+=("$ms"); got_b="${got:-0}"; }
        done
        if [ "${#samples[@]}" -eq 0 ]; then
          echo "  -> no trustworthy sample; not recorded" >&2
          continue
        fi
        # A packet emitted for fewer slots than requested makes step_bench run a
        # SMALLER batch without saying so. Recording the request would file a
        # batch-1 timing under B=8 — a wrong record, which is worse than none.
        if [ "$got_b" != "$bsz" ]; then
          echo "  !! asked for B=$bsz, engine ran B=$got_b (packet batch too small)" >&2
          echo "  -> not recorded; re-emit the packet with plowc --batch covering $bsz" >&2
          continue
        fi
        med="$(median "${samples[@]}")"
        list="$(printf '%s,' "${samples[@]}")"; list="[${list%,}]"
        spread="$(printf '%s\n' "${samples[@]}" | sort -g \
          | awk -v m="$med" 'NR==1{lo=$1} END{printf "%.6f", ($1-lo)/m}')"
        if awk -v s="$spread" -v m="$MAX_SPREAD" 'BEGIN{exit !(s>m)}'; then
          stable=false
          echo "  !! rep spread ${spread} exceeds ${MAX_SPREAD} - row marked UNSTABLE" >&2
        else
          stable=true
        fi

        # The ablated twin, same packet and ctx. TPOT(full) - TPOT(ablated) is
        # the op's real contribution AT THIS GRID, imbalance included, which is
        # a far better signal than total TPOT for a sub-millisecond op.
        abl_med=null; abl_cost=null
        if [ -n "$abldir" ]; then
          aadir="$(assets_for "$abldir" "$pdir" "${cfg}_b${bsz}__abl")"
          asamples=()
          for r in $(seq 1 "$REPS"); do
            read -r ams apre agot <<<"$(run_once "$aadir" "$ctx" "${LABEL_PREFIX}-${cfg}-c${ctx}-b${bsz}-a${r}" "$bsz")"
            if [ -z "${apre:-}" ]; then apre="$ams"; ams=""; fi
            [ "${apre:-0}" -gt "$worst_vram" ] && worst_vram="${apre:-0}"
            [ -n "$ams" ] && asamples+=("$ams")
          done
          if [ "${#asamples[@]}" -gt 0 ]; then
            abl_med="$(median "${asamples[@]}")"
            abl_cost="$(awk -v a="$med" -v b="$abl_med" 'BEGIN{printf "%.3f", a-b}')"
          fi
        fi

        # `batch` sits next to `ctx`: both are run-axis provenance, and the
        # resume check greps for the pair.
        printf '{"config":"%s","ctx":%s,"batch":%s,"dtype":"%s","gpu":"%s","hardware":"%s","model":"%s",' \
          "$cfg" "$ctx" "$bsz" "$DTYPE" "$(json_escape "$GPU")" "$HARDWARE" "$MODEL_NAME" >>"$RESULTS"
        printf '"minblk":%s,"n_cu":%s,"gv_unroll":%s,"gv_unroll_glu":%s,"gv_moe_un":%s,"moe_down_sg":%s,' \
          "$mb" "$ncu" "$un" "$glu" "$mun" "$sg" >>"$RESULTS"
        printf '"gv_mm_max":%s,"extra_defines":%s,' "${mm:-null}" "$xjson" >>"$RESULTS"
        printf '"ns_abs":%s,"ns_full_abs":%s,"fa_wpr":%s,"fa_gf":%s,"fa_gf_full":%s,"fa_kun":%s,' \
          "$ns" "$nsf" "${w:-null}" "${g:-null}" "${gf:-null}" "${k:-null}" >>"$RESULTS"
        printf '"samples_ms":%s,"median_ms":%s,"ablated_ms":%s,"op_cost_ms":%s,' \
          "$list" "$med" "$abl_med" "$abl_cost" >>"$RESULTS"
        printf '"ablate_lo":%s,"ablate_hi":%s,"rel_spread":%s,"stable":%s,' \
          "$ABLATE_LO" "$ABLATE_HI" "$spread" "$stable" >>"$RESULTS"
        printf '"registers":%s,"toolchain":"%s","implementation":"%s",' \
          "$regs" "$TOOLCHAIN" "$IMPL" >>"$RESULTS"
        if [ "$worst_vram" -le "$MEM_IDLE" ]; then unc=true; else unc=false; fi
        printf '"cubin_sha":"%s","pkt_sha":"%s","vram_before_mib":%s,"uncontended":%s,"campaign":"%s","ts":"%s"}\n' \
          "$csha" "$psha" "$worst_vram" "$unc" "$CAMPAIGN" "$(date -Is)" >>"$RESULTS"
        echo "  -> B=${bsz} median ${med} ms of ${#samples[@]} (${samples[*]})  abl=${abl_med} op_cost=${abl_cost}  spread=${spread} stable=${stable} vram=${worst_vram} unc=${unc}"
      done; done
    done; done
  done; done; done; done; done; done; done; done; done; done
done

echo
echo "done. $RESULTS"
echo "ingest with: tunedb-decode ingest --db tuning --results $RESULTS"
