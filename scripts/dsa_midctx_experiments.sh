#!/usr/bin/env bash
# dsa_midctx_experiments.sh — PIPELINE for the GLM-5.2-FP8 DSA mid-ctx perf experiments.
#
# Runs, end-to-end, the experiments that follow from wiring the MFMA indexer + 32-WG select into the
# persistent interp (branch glm-dsa-midctx) and turns them into a consolidated crossover report:
#
#   E1 selwg     single-block dsa bench; sweep DSA_SELWG {32,64,128,256} x ctx -> the indexer-floor
#                (idx-fast vs idx-mfma) and the select-contention curve. Single GPU, cheap. Also the
#                on-device exactness gate (idx relmax 0.0000, select set-EXACT) at every point.
#   E2 build     compile the AFTER (this branch) decode interp+host and emit its dense + gather pkts;
#                add a BASE_REF (consolidated) worktree and compile the BEFORE (fast+256WG) gather pkt.
#   E3 crossover full 78-layer decode sweep over a wide ctx grid: dense vs gather-AFTER vs gather-BEFORE;
#                the flat-gather / linear-dense curves pin the dense/gather CROSSOVER per TP.
#   E4 report    consolidate -> <out>/results.md + results.json, with the crossover + suggested emit gate.
#   tp8          E2+E3+E4 at --tp 8 (needs all 8 GPUs; the design-doc target). Auto-parked if <8 devs.
#
# HSA IGNORES HIP_VISIBLE_DEVICES — GPUs are pinned with ROCR_VISIBLE_DEVICES (--devs). Default 4,5,6,7
# (a sibling owns 0-3). Weights: /home/lava/models/GLM-5.2-FP8-plow.
#
#   nix develop -c ./scripts/dsa_midctx_experiments.sh [--stage all|selwg|build|crossover|report|tp8]
#       [--devs 4,5,6,7] [--tp N] [--ctx 16k,24k,32k,48k,64k,96k,128k] [--ctxmax 131072]
#       [--steps 11] [--out DIR] [--prep DIR] [--base-ref origin/glm-dsa-consolidated] [--force]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=all
DEVS="4,5,6,7"
TP=""
CTX_SWEEP="16k,24k,32k,48k,64k,96k,128k"
CTXMAX=131072
STEPS=11
OUT="${PLOW_BUILD_DIR:-/home/lava/models/dsa_midctx}"
PREP="/home/lava/models/GLM-5.2-FP8-plow"
BASE_REF="origin/glm-dsa-consolidated"
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2;;
    --devs) DEVS="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --ctx) CTX_SWEEP="$2"; shift 2;;
    --ctxmax) CTXMAX="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --prep) PREP="$2"; shift 2;;
    --base-ref) BASE_REF="$2"; shift 2;;
    --force) FORCE=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

NDEV=$(awk -F, '{print NF}' <<<"$DEVS")
[ -z "$TP" ] && TP="$NDEV"
[ "$TP" -gt "$NDEV" ] && { echo "FATAL: --tp $TP > $NDEV devices ($DEVS)"; exit 1; }
mkdir -p "$OUT"
AFTER="$OUT/after"; BEFORE="$OUT/before"; BENCH="$OUT/bench"
PIN="/usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME LD_LIBRARY_PATH=/opt/rocm/lib ROCR_VISIBLE_DEVICES=$DEVS"
echo "== DSA mid-ctx pipeline == stage=$STAGE tp=$TP devs=$DEVS ctx=$CTX_SWEEP steps=$STEPS out=$OUT"

# ---- E1: single-block SELWG / indexer-floor sweep (single GPU, exactness gate) --------------------
stage_selwg() {
  echo "### E1 selwg — dsa single-block bench (idx-fast/idx-mfma + select-contention curve)"
  [ "$FORCE" = 1 -o ! -f "$BENCH/test_kernels.elf" ] && "$REPO/scripts/build_dsa_bench.sh" "$BENCH" >/dev/null
  local dev1="${DEVS%%,*}"
  : > "$OUT/selwg.txt"
  printf 'selwg |  ctx  | idx-fast idx-mfma  sel-us | gates\n' | tee -a "$OUT/selwg.txt"
  for w in 32 64 128 256; do
    # bench row: ctx | dense gather | scalar fast mfma sel | dense/gat | score <relmax> sel <EXACT> gat <PASS> ...
    sg render -c "cd $BENCH && $PIN ROCR_VISIBLE_DEVICES=$dev1 DSA_SELWG=$w ./dsa_bench test_kernels.elf" 2>&1 \
      | awk -v w=$w '/^[0-9]/{printf "%5d | %6d | %7s %7s %8s | %s sel %s gat %s\n", w,$1,$7,$8,$9,$14,$16,$18}' \
      | tee -a "$OUT/selwg.txt"
  done
  echo "   -> $OUT/selwg.txt"
}

# ---- build one variant's harness (interp+host) via the repo's own build_glm52_decode.sh ----------
build_variant() { # <repo-dir> <out-dir>
  local rd="$1" od="$2"
  [ "$FORCE" = 0 ] && [ -f "$od/interp_decode.elf" ] && [ -f "$od/glm52_decode" ] && { echo "   (cached $od)"; return; }
  env PLOW_PREP_DIR="$PREP" "$rd/scripts/build_glm52_decode.sh" "$od" "$TP" "" "$CTXMAX" >/dev/null
}

stage_build() {
  echo "### E2 build — AFTER (this branch) + BEFORE ($BASE_REF) harnesses & pkts"
  build_variant "$REPO" "$AFTER"
  # dense pkt (PLOW_GLM_DSA=0) — dense is variant-independent, emitted once alongside AFTER.
  if [ "$FORCE" = 1 ] || [ ! -f "$AFTER/dense.pkt" ]; then
    ( cd "$REPO" && cargo build --quiet -p plowc --bin plowc )
    env GLM_FULL=1 PLOW_FP8=1 PLOW_GLM_DSA=0 "$REPO/target/debug/plowc" --hf-dir "$PREP" --emit devblob \
        --max-ctx "$CTXMAX" --n-cu 256 --num-gpus "$TP" --out "$AFTER/dense.pkt" >/dev/null
  fi
  # BEFORE: a throwaway worktree at the consolidated base (non-destructive; removed on exit).
  local wt="$OUT/wt_before"
  if [ ! -e "$wt/scripts/build_glm52_decode.sh" ]; then
    git -C "$REPO" worktree add --detach "$wt" "$BASE_REF" >/dev/null 2>&1 || git -C "$REPO" worktree add "$wt" "$BASE_REF"
  fi
  build_variant "$wt" "$BEFORE"
  echo "   AFTER gather=$AFTER/glm52_tp${TP}.pkt dense=$AFTER/dense.pkt | BEFORE gather=$BEFORE/glm52_tp${TP}.pkt"
}

# ---- run one decode sweep, parse '  ctx  ms/tok  tok/s' rows into the crossover TSV --------------
run_sweep() { # <dir> <pkt> <variant>
  local dir="$1" pkt="$2" var="$3"
  echo "   sweep $var ($pkt)" >&2
  sg render -c "cd $dir && $PIN ./glm52_decode $pkt $PREP --tp $TP --sweep $CTX_SWEEP --steps $STEPS" 2>&1 \
    | awk -v tp=$TP -v v=$var '/^  *[0-9]+ +[0-9.]+ +[0-9.]+/ {print tp"\t"v"\t"$1"\t"$2}' \
    | tee -a "$OUT/crossover.tsv"
}

stage_crossover() {
  echo "### E3 crossover — dense vs gather-AFTER vs gather-BEFORE, tp=$TP over $CTX_SWEEP"
  : > "$OUT/crossover.tsv"
  run_sweep "$AFTER"  "dense.pkt"            "dense"
  run_sweep "$AFTER"  "glm52_tp${TP}.pkt"    "gather_after"
  run_sweep "$BEFORE" "glm52_tp${TP}.pkt"    "gather_before"
  echo "   -> $OUT/crossover.tsv"
}

stage_report() {
  echo "### E4 report"
  [ -s "$OUT/crossover.tsv" ] || { echo "   no crossover.tsv — run --stage crossover first"; return 1; }
  python3 "$REPO/scripts/dsa_midctx_report.py" --md "$OUT/results.md" --json "$OUT/results.json" \
    --title "DSA mid-ctx experiments (tp$TP, devs $DEVS)" < "$OUT/crossover.tsv"
  echo "   -> $OUT/results.md  $OUT/results.json"
}

case "$STAGE" in
  selwg)     stage_selwg;;
  build)     stage_build;;
  crossover) stage_crossover;;
  report)    stage_report;;
  tp8)
    [ "$NDEV" -lt 8 ] && { echo "PARKED: tp8 needs 8 GPUs; got $NDEV ($DEVS). Re-run with --devs 0,1,..,7 when the node is free."; exit 0; }
    TP=8; AFTER="$OUT/after_tp8"; BEFORE="$OUT/before_tp8"
    stage_build; stage_crossover; stage_report;;
  all)       stage_selwg; stage_build; stage_crossover; stage_report;;
  *) echo "unknown --stage $STAGE"; exit 2;;
esac
echo "== done: $STAGE =="
