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
#
#   PACKET knobs      -> one `plowc --emit devblob` each (~60 s)
#     --n-cu                grid width; MUST match FORCE_MINBLK (132 <-> 1, 264 <-> 2)
#     PLOW_NS_ABS           flash decode split count
#
# `(FORCE_MINBLK, --n-cu)` is a PAIR — the engine refuses a packet emitted for
# 132 blocks against an object that reaches 264 — so it is swept as one axis
# (--occ "1:132 2:264"). Everything else crosses freely.
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
GPU="H100 NVL"
HARDWARE="nvidia/sm_90a/h100-nvl"
CAMPAIGN="tuner-decode-sweep"
MEM_IDLE=2000              # MiB; at or below this the card is verifiably ours
MEM_RUN=""                 # MiB; hard ceiling to attempt a run at all (default = MEM_IDLE)
WAIT_MAX=0                 # s to wait for MEM_IDLE before falling back to MEM_RUN; 0 = forever
WAIT_S=60
SETTLE_S=6                 # s to let our own VRAM drain between reps
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
    --label) LABEL_PREFIX="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) usage 0;;
    *) echo "unknown option $1" >&2; usage 2;;
  esac
done

[ -n "$PLOWC" ] || PLOWC="$ROOT/target/release/plowc"
[ -n "$STEP_BENCH" ] || STEP_BENCH="$ROOT/target/release/examples/step_bench"
[ -n "$MEM_RUN" ] || MEM_RUN="$MEM_IDLE"
[ -n "$CHECKPOINT" ] || CHECKPOINT="$MODEL"
[ -n "$MODEL_NAME" ] || MODEL_NAME="$(basename "$MODEL")"
[ -x "$PLOWC" ] || { echo "no plowc at $PLOWC (--plowc)" >&2; exit 2; }
[ -x "$STEP_BENCH" ] || { echo "no step_bench at $STEP_BENCH (--step-bench)" >&2; exit 2; }
[ -d "$MODEL" ] || { echo "no model dir $MODEL" >&2; exit 2; }

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
echo "grid       : occ[$OCC] ns_abs[$NS_ABS] unroll[$GV_UNROLL] glu[$GV_UNROLL_GLU]"
echo "             moe_un[$GV_MOE_UN] sg[$MOE_DOWN_SG] ctx[$CTXS] reps=$REPS"
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
# Key: every OBJECT knob. Built once, reused by every packet and ctx that wants it.
build_cubin() {  # $1 minblk  $2 unroll  $3 glu  $4 moe_un  $5 sg   -> echoes dir
  local mb="$1" un="$2" glu="$3" mun="$4" sg="$5"
  local key="mb${mb}_un${un}_glu${glu}_mun${mun}_sg${sg}"
  local dir="$WORK/cubin/$key"
  if [ -f "$dir/interp_sm90a.cubin" ] && [ -f "$dir/interp_sm90a_pf.cubin" ]; then
    echo "$dir"; return 0
  fi
  local defs="-DPLOW_NV_FORCE_MINBLK=$mb -DGV_UNROLL=$un -DGV_MOE_UN=$mun -DPLOW_MOE_DOWN_SG=${sg}u"
  [ "$glu" = "0" ] || defs="$defs -DGV_UNROLL_GLU=$glu"
  mkdir -p "$dir"
  echo "  [build] $key  ($defs)" >&2
  if [ "$DRY" = "1" ]; then echo "$dir"; return 0; fi
  PLOW_ROOT="$ROOT" PLOW_EXTRA_DEFINES="$defs" \
    "$ROOT/scripts/build_sm90a_cubin.sh" "$dir/interp_sm90a.cubin" >"$WORK/log/build_$key.log" 2>&1 \
    || { echo "FATAL: cubin build failed for $key — see $WORK/log/build_$key.log" >&2; exit 1; }
  registers_of "$dir/interp_sm90a.cubin" > "$dir/registers" || true
  echo "$dir"
}

# Registers of the decode megakernel. Recorded next to every timing because a
# knob that buys ILP by raising pressure for every other arm is not obviously a
# win — and because the campaign's "255 registers explains the regression"
# story was later retracted, so this is evidence to keep, not a verdict.
registers_of() {
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin cuobjdump -res-usage "$1" 2>/dev/null \
    | awk '/interp_sm90a11PlowProgram/{f=1}
           f && /REG:/ { for(i=1;i<=NF;i++) if($i ~ /^REG:/){ sub("REG:","",$i); print $i; exit } }'
}

# -------------------------------------------------------------- packet emit
# Key: every PACKET knob. `--n-cu` and PLOW_NS_ABS are the only two the design
# leaves on the packet; both already have ctx in scope at emit time.
emit_packet() {  # $1 n_cu  $2 ns_abs  -> echoes dir
  local ncu="$1" ns="$2"
  local key="ncu${ncu}_ns${ns}"
  local dir="$WORK/pkt/$key"
  if [ -f "$dir/model.pkt" ]; then echo "$dir"; return 0; fi
  mkdir -p "$dir"
  echo "  [emit ] $key" >&2
  if [ "$DRY" = "1" ]; then echo "$dir"; return 0; fi
  local nsenv=()
  [ "$ns" = "0" ] || nsenv=(PLOW_NS_ABS="$ns")
  env PLOW_UNISEG=1 "${nsenv[@]}" "$PLOWC" \
      --hf-dir "$MODEL" --emit devblob --max-ctx "$MAXCTX" --n-cu "$ncu" --out "$dir" \
      >"$WORK/log/emit_$key.log" 2>&1 \
    || { echo "FATAL: packet emit failed for $key — see $WORK/log/emit_$key.log" >&2; exit 1; }
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
  ln -sfn "$cdir/interp_sm90a.cubin" "$dir/interp_sm90a.cubin"
  ln -sfn "$cdir/interp_sm90a_pf.cubin" "$dir/interp_sm90a_pf.cubin"
  echo "$dir"
}

# ------------------------------------------------------------------ one point
# Returns the mean_ms of one step_bench run, or empty if the run was contended
# or failed. Never returns a number it does not trust.
# Echoes "<mean_ms> <vram_before_mib>", or just "<vram_before_mib>" when there
# is no trustworthy sample. Both travel on stdout because the caller reads this
# through a command substitution, and a subshell cannot set a variable in its
# parent — which silently zeroed the provenance field the first time.
run_once() {  # $1 assets  $2 ctx  $3 label -> echoes "ms vram" | "vram"
  local adir="$1" ctx="$2" label="$3" rc pre
  local log="$WORK/log/run_${label}.log"
  local vramf="$WORK/log/vram_${label}"

  # A light pre-check so we do not queue behind a holder that never leaves; the
  # AUTHORITATIVE reading is taken after the lease is held, below.
  set +e
  wait_for_idle >/dev/null
  set -e

  # Read memory.used INSIDE the lease. Reading it before means a sibling agent's
  # in-flight run looks like a permanent holder and the point is skipped, when
  # what should happen is that gpulease serialises us and we measure once the
  # card is actually ours. It also means the number recorded as provenance is
  # the one that was true while WE ran, not one from before we had the card.
  set +e
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat \
    gpulease "$label" bash -c '
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " " > "$1"
      shift; exec "$@"
    ' _ "$vramf" "$STEP_BENCH" "$adir" 1 "$ctx" "$STEPS" >"$log" 2>&1
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
  if [ -n "$ms" ]; then echo "$ms $pre"; else echo "$pre"; fi
}

median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END{ if(NR==0) exit 1; print a[int((NR+1)/2)] }'; }

# ---------------------------------------------------------------------- sweep
total=0; done_n=0
for occ in $OCC; do for ns in $NS_ABS; do for un in $GV_UNROLL; do for glu in $GV_UNROLL_GLU; do
for mun in $GV_MOE_UN; do for sg in $MOE_DOWN_SG; do for ctx in $CTXS; do
  total=$((total+1))
done; done; done; done; done; done; done
echo "grid points: $total"

for occ in $OCC; do
  mb="${occ%%:*}"; ncu="${occ##*:}"
  for un in $GV_UNROLL; do for glu in $GV_UNROLL_GLU; do for mun in $GV_MOE_UN; do for sg in $MOE_DOWN_SG; do
    cdir="$(build_cubin "$mb" "$un" "$glu" "$mun" "$sg")"
    # A pre-built object (see the grid warm-up in the header) may not carry the
    # sidecar yet; derive it rather than emitting a null into the record.
    [ -s "$cdir/registers" ] || registers_of "$cdir/interp_sm90a.cubin" > "$cdir/registers"
    regs="$(tr -cd '0-9' < "$cdir/registers" 2>/dev/null || true)"
    [ -n "$regs" ] || regs=null
    csha="$(sha256sum "$cdir/interp_sm90a.cubin" 2>/dev/null | cut -c1-16 || true)"
    for ns in $NS_ABS; do
      pdir="$(emit_packet "$ncu" "$ns")"
      psha="$(sha256sum "$pdir/model.pkt" 2>/dev/null | cut -c1-16 || true)"
      cfg="mb${mb}_ncu${ncu}_un${un}_glu${glu}_mun${mun}_sg${sg}_ns${ns}"
      adir="$(assets_for "$cdir" "$pdir" "$cfg")"
      for ctx in $CTXS; do
        done_n=$((done_n+1))
        if grep -qF "\"config\":\"$cfg\",\"ctx\":$ctx," "$RESULTS" 2>/dev/null; then
          echo "[$done_n/$total] $cfg ctx=$ctx — already recorded, skipping"
          continue
        fi
        echo "[$done_n/$total] $cfg ctx=$ctx"
        [ "$DRY" = "1" ] && continue
        samples=(); worst_vram=0
        for r in $(seq 1 "$REPS"); do
          read -r ms pre <<<"$(run_once "$adir" "$ctx" "${LABEL_PREFIX}-${cfg}-c${ctx}-r${r}")"
          # One field means "no sample, here is the VRAM"; two means "sample, VRAM".
          if [ -z "${pre:-}" ]; then pre="$ms"; ms=""; fi
          # The row is only as trustworthy as its WORST rep, so keep the max.
          [ "${pre:-0}" -gt "$worst_vram" ] && worst_vram="${pre:-0}"
          [ -n "$ms" ] && samples+=("$ms")
        done
        if [ "${#samples[@]}" -eq 0 ]; then
          echo "  -> no trustworthy sample; not recorded" >&2
          continue
        fi
        med="$(median "${samples[@]}")"
        list="$(printf '%s,' "${samples[@]}")"; list="[${list%,}]"
        printf '{"config":"%s","ctx":%s,"dtype":"%s","gpu":"%s","hardware":"%s","model":"%s",' \
          "$cfg" "$ctx" "$DTYPE" "$(json_escape "$GPU")" "$HARDWARE" "$MODEL_NAME" >>"$RESULTS"
        printf '"minblk":%s,"n_cu":%s,"gv_unroll":%s,"gv_unroll_glu":%s,"gv_moe_un":%s,"moe_down_sg":%s,"ns_abs":%s,' \
          "$mb" "$ncu" "$un" "$glu" "$mun" "$sg" "$ns" >>"$RESULTS"
        printf '"samples_ms":%s,"median_ms":%s,"registers":%s,"toolchain":"%s","implementation":"%s",' \
          "$list" "$med" "$regs" "$TOOLCHAIN" "$IMPL" >>"$RESULTS"
        if [ "$worst_vram" -le "$MEM_IDLE" ]; then unc=true; else unc=false; fi
        printf '"cubin_sha":"%s","pkt_sha":"%s","vram_before_mib":%s,"uncontended":%s,"campaign":"%s","ts":"%s"}\n' \
          "$csha" "$psha" "$worst_vram" "$unc" "$CAMPAIGN" "$(date -Is)" >>"$RESULTS"
        echo "  -> median ${med} ms of ${#samples[@]} (${samples[*]})  vram_before=${worst_vram} MiB uncontended=${unc}"
      done
    done
  done; done; done; done
done

echo
echo "done. $RESULTS"
echo "ingest with: tunedb-decode ingest --db tuning --results $RESULTS"
