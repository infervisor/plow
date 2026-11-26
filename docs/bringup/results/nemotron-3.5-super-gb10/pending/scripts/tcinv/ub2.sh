#!/usr/bin/env bash
# ub2.sh (gpuq, jg02): single-SM ubench (Nemotron and Qwen3.8 packets) for each frozen tree given.
#   ub2.sh <out> <reps> <tree:tag[:appendflags]>... -- <case>...
# Case prefixes pick the set: none = nemotron-dec, pf/ = nemotron-pf, q/ = qwen-dec. Each tree builds
# its driver and the ops cubin of every listed set (BUILD_DIR=<out>/build-<tag>); flags (NVCC_APPEND
# defines) make a variant of the same tree. Extra trees / cases: <out>.extra / <out>.cases.
set -uo pipefail
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
Q=/home/ifv/.claude/jobs/72c86dac/tmp/nemotron/mtp/quiet.sh
U=/home/ifv/plow-scratch/ubench
O=$1; REPS=$2; shift 2
TREES=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do TREES+=("$1"); shift; done
shift
CASES=("$@")
[ -f "$O.extra" ] && while read -r l; do [ -n "$l" ] && TREES+=("$l"); done < "$O.extra"
[ -f "$O.cases" ] && while read -r l; do [ -n "$l" ] && CASES+=("$l"); done < "$O.cases"
# set <prefix> -> "set assets buildlog object"
setof() { case $1 in
  pf) echo "nemotron-pf $U/nemotron/assets $U/nemotron/build.log interp_sm120_pf" ;;
  q) echo "qwen-dec $U/qwen/assets $U/qwen/build.log interp_sm120_rs" ;;
  *) echo "nemotron-dec $U/nemotron/assets $U/nemotron/build.log interp_sm120_rs" ;;
esac; }
pre() { [[ $1 == */* ]] && echo "${1%%/*}" || echo dec; }
PRES=$(for c in "${CASES[@]}"; do pre "$c"; done | sort -u)
mkdir -p "$O"
# (no NP_RUNTIME: userns restricted after reboot)
date -u +%T
R() { echo "bash $1/runtime/ubench/run_ubench_cu.sh --nv"; }
for spec in "${TREES[@]}"; do
  IFS=: read -r T TAG XF <<<"$spec"
  [ -d "$T/runtime" ] || { echo "skip $TAG: no tree $T"; continue; }
  B=$O/build-$TAG
  mkdir -p "$B"
  CMD="$(R $T) --build-only"
  for p in $PRES; do read -r S A L OBJ <<<"$(setof $p)"; CMD="$CMD && $(R $T) --build-only --set $S --assets $A --build-log $L --object $OBJ"; done
  (cd "$W" && /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh env BUILD_DIR="$B" NVCC_APPEND_FLAGS_X="$XF" bash -c \
     'export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} $NVCC_APPEND_FLAGS_X"; '"$CMD") > "$O/build-$TAG.log" 2>&1
  echo "$TAG build rc=$? $(date -u +%T)"; grep -E "ops defines| error|FATAL|built" "$O/build-$TAG.log" | cut -c1-200 | head -8
done
$Q wait 900
$Q watch "$O/compilers.log" & QW=$!
trap 'kill $QW 2>/dev/null' EXIT
t0=$(date +%s)
for c0 in "${CASES[@]}"; do
  p=$(pre "$c0"); c=${c0#*/}; read -r S A L OBJ <<<"$(setof $p)"
  for spec in "${TREES[@]}"; do
    IFS=: read -r T TAG XF <<<"$spec"
    B=$O/build-$TAG
    [ -f "$B/ubench_${S}_sm_121a.cubin" ] || continue
    timeout 1200 bash -c "cd $W && /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh env BUILD_DIR=$B $(R $T) --set $S --assets $A --build-log $L --object $OBJ --case $c --reps $REPS --json $O/$TAG.$p.$c.json" > "$O/$TAG.$p.$c.log" 2>&1
    rc=$?
    grep "^| [a-z]" "$O/$TAG.$p.$c.log" | grep -v "^| case" | sed "s/^/$TAG /"
    echo "  $TAG $p/$c $(grep -o '"digest": "[0-9a-f]*"' "$O/$TAG.$p.$c.json" 2>/dev/null | tr '\n' ' ')"
    [ $rc -ne 0 ] && { echo "  $TAG $c rc=$rc"; grep -E "FAIL|HANG|alloc|error|MISMATCH" "$O/$TAG.$p.$c.log" | head -3; }
  done
done
echo "ub done $($Q report $O/compilers.log $t0) $(date -u +%T)"
