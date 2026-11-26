#!/usr/bin/env bash
# j2.sh [phases] (gpuq, jg02): Qwen3.8-27B FP8 gates for the row-invariant TC GEMV route.
# Arms (decode objects rebuilt by qvar.sh on the ladder-16 packet $QA):
#   C = FFMA (base runtime), T = current TC rung 1 (base runtime), R = TC rung 1 + ROW_INVARIANT=$RIV (dev)
# Ladder-32 packet (B=8 spec, verify 32 rows): emitted from the dev tree with the recipe +
#   PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 PLOW_FP8_DECODE_TC_ROW_INVARIANT=1 (E-R, objects as emitted)
#   and E-T = its decode objects rebuilt from the base runtime (current TC).
# phases: build emit time lg spec spec8
set -uo pipefail
PH=${1:-build emit optest time lg spec spec8}
RIV=${RIV:-1}
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
T=/home/ifv/plow-tcinv
BASE=/home/ifv/plow-scratch/tcinv/base-runtime
O=/home/ifv/plow-scratch/tcinv/j2
J=/home/ifv/.claude/jobs/72c86dac/tmp/tcinv
DJ=/home/ifv/.claude/jobs/72c86dac/tmp/declat
Q=/home/ifv/.claude/jobs/72c86dac/tmp/nemotron/mtp/quiet.sh
QA=/home/ifv/plow-scratch/ubench/qwen/assets
QL=/home/ifv/plow-scratch/ubench/qwen/build.log
E=$O/emit32
PY=/home/ifv/venvs/vllm-gb10/bin/python
P=/home/ifv/plow-scratch/declat/qwen_prompts1k.json
export CARGO_TARGET_DIR=/home/ifv/plow-target-tcinv
SB=$O/step_bench
TC1="-DPLOW_NV_FP8_DECODE_TC_ON=1 -DPLOW_NV_FP8_DECODE_TC=1"
mkdir -p $O
date -u +%T
on() { [[ " $PH " == *" $1 "* ]]; }
DEV() { (cd $W && $J/nx.sh "$@"); }
if on build; then
  cp $CARGO_TARGET_DIR/release/examples/step_bench $SB || echo "no step_bench"
  DEV bash $DJ/qvar.sh $QA $QL $O/q-C $BASE > $O/v-C.log 2>&1 & P1=$!
  DEV bash $DJ/qvar.sh $QA $QL $O/q-T $BASE $TC1 > $O/v-T.log 2>&1 & P2=$!
  DEV bash $DJ/qvar.sh $QA $QL $O/q-R $T/runtime $TC1 -DPLOW_NV_FP8TC_ROW_INVARIANT=$RIV > $O/v-R.log 2>&1 & P3=$!
  for p in $P1 $P2 $P3; do wait $p; echo "qvar pid rc=$?"; done
  for a in C T R; do echo "$a: $(grep -cE 'error' $O/v-$a.log) error lines; $(ls -la $O/q-$a/interp_sm120_rs.cubin 2>&1 | awk '{print $5}')B rs"; done
  # Nemotron recipe packet (knob off): decode objects from base vs dev runtime must be SASS-identical
  V3=/home/ifv/.claude/jobs/72c86dac/tmp/nemotron/dec7/variant3.sh
  NF=/home/ifv/plow-scratch/declat/full/assets
  for x in base:$BASE dev:$T/runtime; do
    IFS=: read -r n rt <<<"$x"; rm -rf $O/nemo-$n
    DEV bash $V3 $NF $O/nemo-$n $rt interp_sm120.cubin interp_sm120_rs.cubin > $O/v-nemo-$n.log 2>&1 &
  done
  wait
  for c in interp_sm120.cubin interp_sm120_rs.cubin; do
    for n in base dev; do $J/nx.sh bash -c "\$(dirname \$PLOW_NVCC)/cuobjdump -sass $O/nemo-$n/$c" > $O/nemo-$n-$c.sass 2>&1; done
    echo "nemotron $c: bytes $(cmp -s $O/nemo-base/$c $O/nemo-dev/$c && echo same || echo differ), sass $(cmp -s $O/nemo-base-$c.sass $O/nemo-dev-$c.sass && echo IDENTICAL || echo DIFFER) ($(wc -l < $O/nemo-dev-$c.sass) lines)"
  done
fi
if on emit; then
  rm -rf $E
  (cd $T && PLOW_CAMPAIGN_NO_NIX=1 QWEN38_CHECKPOINT=/home/ifv/models/Qwen/Qwen3.8-27B $J/nx.sh python3 scripts/campaign/campaign.py build \
     scripts/campaign/recipes/qwen38-27b.gb10.fp8.toml --out $E --no-probe \
     --env PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 --env PLOW_FP8_DECODE_TC_ROW_INVARIANT=1) > $O/emit32.log 2>&1
  echo "emit32 rc=$? $(tail -n 2 $O/emit32.log | tr '\n' ' ' | cut -c1-200)"
  for f in model.pkt interp_sm120.cubin interp_sm120_rs.cubin plow_config.h; do
    [ -s $E/assets/$f ] || { echo "EMIT32 FAILED: $E/assets/$f missing"; PH=${PH/spec8/}; }
  done
  grep -h "FP8TC_ROW_INVARIANT\|FP8_DECODE_TC " $E/assets/plow_config.h
  DEV bash $DJ/qvar.sh $E/assets $E/build.log $O/e-T $BASE > $O/v-eT.log 2>&1
  echo "e-T qvar rc=$? $(grep -cE 'error' $O/v-eT.log) error lines"
fi
export LD_LIBRARY_PATH="$CUBLAS_LIB"  # cuBLAS lib dir from the flake env
export PLOW_CHECKPOINT=/home/ifv/plow-scratch/qwen38-checkpoint-mixed
export PLOW_VMM_LIVE=1 PLOW_HOST_KVCACHE_SIZE=0
$Q wait 1800
$Q watch $O/compilers.log & QW=$!
trap 'kill $QW 2>/dev/null' EXIT
if on optest; then
  # the nix-linked driver's cudart dlopens libcuda.so.1: give it the host driver only
  for tag in r4 r1ri0 r1ri1 r1ri2; do
    t0=$(date +%s)
    LD_LIBRARY_PATH=/home/ifv/plow-scratch/tcinv/libcuda timeout 1800 /home/ifv/plow-scratch/tcinv/j1/optest-$tag > $O/optest-$tag.log 2>&1
    echo "optest-$tag rc=$? $(( $(date +%s) - t0 ))s FAIL-lines=$(grep -c -- '-> FAIL\|MISMATCH' $O/optest-$tag.log) $(tail -n 1 $O/optest-$tag.log)"
    grep "^  rows " $O/optest-$tag.log
  done
fi
run() { local l=$1; shift; local t0; t0=$(date +%s)
  timeout 1800 $SB "$@" > $O/$l.log 2>&1
  echo "$l rc=$? $(grep -h 'SPEC_IDENTITY' $O/$l.log | grep -o 'SPEC_IDENTITY [A-Za-z]*\|spec_fnv=[0-9a-f]*\|mean_emit=[0-9.]*\|plain_ms=[0-9.]*\|verify_ms=[0-9.]*\|tick_ms=[0-9.]*' | tr '\n' ' ')$(grep -h 'RAW_STEP' $O/$l.log | grep -o 'mean_ms=[0-9.]*\|p50_ms=[0-9.]*' | tr '\n' ' ')$(grep -h 'TOK_STREAM' $O/$l.log | grep -o 'fnv=[0-9a-f]*' | head -1) $(grep -h 'rror\|divergence' $O/$l.log | tail -n 2 | cut -c1-120 | tr '\n' ' ') $($Q report $O/compilers.log $t0) $(( $(date +%s) - t0 ))s"; }
if on time; then
  for r in 1 2; do
    for b in 1 2 4 8 16; do
      for arm in T R; do
        run p$b-$arm-r$r $O/q-$arm $b 1024 48 --warmup 8 --prompt-tokens $P --force-prompt
      done
    done
  done
fi
if on lg; then
  for b in 1 8 16; do
    for arm in C T R C2; do
      a=${arm:0:1}
      run lg$b-$arm $O/q-$a $b 1024 64 --warmup 0 --prompt-tokens $P --force-prompt --dump-step-logits $O/lg$b-$arm.bin
    done
    echo "lg B=$b C vs C2 (floor): $($PY $DJ/lgcmp.py $O/lg$b-C.bin $O/lg$b-C2.bin $b 248320)"
    echo "lg B=$b C vs T:          $($PY $DJ/lgcmp.py $O/lg$b-C.bin $O/lg$b-T.bin $b 248320)"
    echo "lg B=$b C vs R:          $($PY $DJ/lgcmp.py $O/lg$b-C.bin $O/lg$b-R.bin $b 248320)"
    echo "lg B=$b T vs R:          $($PY $DJ/lgcmp.py $O/lg$b-T.bin $O/lg$b-R.bin $b 248320)"
    rm -f $O/lg$b-*.bin
  done
fi
if on spec; then
  for b in 1 4; do
    for arm in T R; do
      run s$b-$arm $O/q-$arm $b 1024 64 --warmup 1 --prompt-tokens $P --spec mtp --spec-k 3
    done
    run g$b-R $O/q-R $b 1024 64 --warmup 0 --prompt-tokens $P
  done
  run c4-R $O/q-R 4 1024 64 --warmup 1 --prompt-tokens $P --spec cycle --spec-k 3
  run c4p-R $O/q-R 4 1024 64 --warmup 1 --prompt-tokens $P --spec cycle --spec-k 3 --spec-poison
  run c1-R $O/q-R 1 1024 64 --warmup 1 --prompt-tokens $P --spec cycle --spec-k 3
fi
if on spec8; then
  for arm in e-T emit32/assets; do
    l=$( [ $arm = e-T ] && echo eT || echo eR )
    run s8-$l $O/$arm 8 1024 64 --warmup 1 --prompt-tokens $P --spec mtp --spec-k 3
    run p8-$l $O/$arm 8 1024 48 --warmup 8 --prompt-tokens $P --force-prompt
  done
  run c8-eR $O/emit32/assets 8 1024 64 --warmup 1 --prompt-tokens $P --spec cycle --spec-k 3
  run s4-eR $O/emit32/assets 4 1024 64 --warmup 1 --prompt-tokens $P --spec mtp --spec-k 3
fi
echo "j2 done $(date -u +%T)"
