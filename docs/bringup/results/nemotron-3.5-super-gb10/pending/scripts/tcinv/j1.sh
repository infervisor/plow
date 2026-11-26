#!/usr/bin/env bash
# j1.sh (gpuq, jg02): FP8 TC GEMV row-invariance, job 1.
#   build: op test variants (nvcc), dev-tree cargo (knob tests, plowc, step_bench)
#   optest: sm120_interp_op_test per variant (row-invariance lines + full suite verdict)
#   ub: single-SM ubench, Qwen arms ri0/ri1/ri2 (TC rung 1), Nemotron arms n0/n1 (rung 4 packet)
set -uo pipefail
PH=${1:-build optest ub j2}
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
T=/home/ifv/plow-tcinv
O=/home/ifv/plow-scratch/tcinv/j1
J=/home/ifv/.claude/jobs/72c86dac/tmp/tcinv
DJ=/home/ifv/.claude/jobs/72c86dac/tmp/declat
Q=/home/ifv/.claude/jobs/72c86dac/tmp/nemotron/mtp/quiet.sh
export CARGO_TARGET_DIR=/home/ifv/plow-target-tcinv
mkdir -p $O
date -u +%T
on() { [[ " $PH " == *" $1 "* ]]; }
DEV() { (cd $W && $J/nx.sh "$@"); }
TC1="-DPLOW_NV_FP8_DECODE_TC_ON=1 -DPLOW_NV_FP8_DECODE_TC=1"
if on build; then
  ot() { local tag=$1; shift
    DEV nvcc -std=c++17 -O2 -arch=sm_121a -I $T/runtime/common -I $T/runtime/nvidia "$@" $J/rows_test.cu -o $O/optest-$tag > $O/optest-$tag.build.log 2>&1
    echo "optest-$tag build rc=$? $(grep -m2 ' error' $O/optest-$tag.build.log)"; }
  python3 $J/mkrows.py $T/runtime/tests/sm120_interp_op_test.cu $J/rows_test.cu
  ot r4 &
  ot r1ri0 $TC1 &
  ot r1ri1 $TC1 -DPLOW_NV_FP8TC_ROW_INVARIANT=1 &
  ot r1ri2 $TC1 -DPLOW_NV_FP8TC_ROW_INVARIANT=2 &
  M=$T/Cargo.toml
  DEV bash -c "cargo test --manifest-path $M -p devgen --lib knob && cargo test --manifest-path $M -p devgen --lib fp8_decode_tc && cargo test --manifest-path $M -p plowrt --features cuda,hsa --lib knob" > $O/knob-test.log 2>&1
  echo "knob tests rc=$? $(grep -h 'test result' $O/knob-test.log | tr '\n' ' ')"
  DEV bash -c "cargo build --manifest-path $M --release -p plowc && cargo build --manifest-path $M --release -p plowrt --features cuda --example step_bench" > $O/cargo-build.log 2>&1
  echo "plowc+step_bench build rc=$? $(grep -m3 '^error' $O/cargo-build.log)"
  wait
  echo "build done $(date -u +%T)"
fi
if on optest; then
  $Q wait 1800
  for tag in r4 r1ri0 r1ri1 r1ri2; do
    [ -x $O/optest-$tag ] || { echo "optest-$tag missing"; continue; }
    t0=$(date +%s)
    timeout 1800 $O/optest-$tag > $O/optest-$tag.log 2>&1
    echo "optest-$tag rc=$? $(( $(date +%s) - t0 ))s FAIL-lines=$(grep -c -- '-> FAIL\|MISMATCH' $O/optest-$tag.log)"
    grep "^  rows " $O/optest-$tag.log
  done
fi
if on ub; then
  bash $J/ub2.sh $O/ubq 5 "$T:ri0:$TC1" "$T:ri1:$TC1 -DPLOW_NV_FP8TC_ROW_INVARIANT=1" "$T:ri2:$TC1 -DPLOW_NV_FP8TC_ROW_INVARIANT=2" -- \
    q/gemv_fp8_out_proj q/gemv_fp8_o_proj q/gemv_fp8_k_proj q/gemv_fp8_in_proj_a q/gemv_fp8_in_proj_qkv q/gemv_fp8_mlp_down
  bash $J/ub2.sh $O/ubn 5 "$T:n0:" "$T:n1:-DPLOW_NV_FP8TC_ROW_INVARIANT=1" -- gemv_fp8_attn_q gemv_fp8_attn_o
fi
echo "j1 done $(date -u +%T)"
if on j2; then RIV=1 bash $J/j2.sh; fi
