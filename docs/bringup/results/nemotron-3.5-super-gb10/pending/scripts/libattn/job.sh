#!/usr/bin/env bash
# job.sh (gpuq, jg02): GB10 attention, plow native bodies vs libraries (bench_lib_attn.py gb10).
#   job.sh [phases]   phases: build warm nemo-pf nemo-dec qwen-pf qwen-dec (default all)
set -uo pipefail
PH=${1:-build warm nemo-pf nemo-dec qwen-pf qwen-dec}
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
J=/home/ifv/.claude/jobs/72c86dac/tmp/libattn
NX=/home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh
Q=/home/ifv/.claude/jobs/72c86dac/tmp/nemotron/mtp/quiet.sh
O=/home/ifv/plow-scratch/libattn
NA=/home/ifv/plow-scratch/scoreboard/full
QA=/home/ifv/plow-scratch/hd256/q-pf
T=$W/scripts/gen_kernels/bench_lib_attn.py
mkdir -p $O
on() { [[ " $PH " == *" $1 "* ]]; }
date -u +%T
if on build; then
  b() { (cd $W && $NX python3 $T gb10-build "$@") > $O/build-$(basename "$4" .cubin).log 2>&1
        echo "build $(basename "$4") rc=$? $(grep -m3 ' error' $O/build-$(basename "$4" .cubin).log)"; }
  b $NA/assets $NA/build.log interp_sm120_pfseg $O/nemo-pf.cubin --hd 128 --gf 16
  b $NA/assets $NA/build.log interp_sm120_rs $O/nemo-dec.cubin --hd 128 --gf 16
  b $QA/assets $QA/build.log interp_sm120_pfseg $O/qwen-pf.cubin --hd 256 --gf 2
  b $QA/assets $QA/build.log interp_sm120_rs $O/qwen-dec.cubin --hd 256 --gf 2
  echo "build done $(date -u +%T)"
fi
R=$J/r.sh
NEMO="--fp8 --baked 1:24,2:12,4:6,8:3,16:3,32:3 --merge-blocks 32"
QWEN="--baked 1:4,2:4,4:4,8:4,16:4 --merge-blocks 24"
PFL="fi,triton,fa2,fa4,cudnn"
DCL="fi,fid,fidtc,xqa,xqac,triton,fa2,fa4,cudnn"
if on warm; then
  # JIT (FlashInfer ninja, Triton, CuTe DSL, driver PTX JIT) outside the timed runs.
  timeout 3600 $R $T gb10 $O/nemo-pf.cubin $NEMO --libs $PFL --classes 32,2,128,1,256,1280 > $O/warm-nemo-pf.log 2>&1; echo "warm nemo-pf rc=$?"
  timeout 3600 $R $T gb10 $O/nemo-dec.cubin $NEMO --libs $DCL --classes 32,2,128,1,1,1024 32,2,128,2,4,1024 > $O/warm-nemo-dec.log 2>&1; echo "warm nemo-dec rc=$?"
  timeout 3600 $R $T gb10 $O/qwen-pf.cubin $QWEN --libs $PFL --classes 24,4,256,1,256,1280 > $O/warm-qwen-pf.log 2>&1; echo "warm qwen-pf rc=$?"
  timeout 3600 $R $T gb10 $O/qwen-dec.cubin $QWEN --libs $DCL --classes 24,4,256,2,1,1024 > $O/warm-qwen-dec.log 2>&1; echo "warm qwen-dec rc=$?"
  cat $O/warm-*.log | grep -v "^h[0-9]" | grep -iE "error|Traceback" | sort | uniq -c | head -20
  grep -h "^h[0-9]" $O/warm-*.log | cut -c1-2000
fi
cold() { local i=0; until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1)" -le 58 ] || [ $i -ge 120 ]; do sleep 10; i=$((i+1)); done; }
timed() {  # timed <tag> <cubin> <opts> <libs> <classes...>
  local tag=$1 cub=$2 opts=$3 libs=$4; shift 4
  [ -f "$cub" ] || { echo "MISSING native cubin $cub: $tag runs libraries only (error)"; cub=-; }
  $Q wait 1800
  cold
  : > $O/compilers-$tag.log
  $Q watch $O/compilers-$tag.log & local qw=$!
  timeout 7200 $R $T gb10 $cub $opts --libs $libs --json $O/$tag.json --classes "$@" > $O/$tag.log 2>&1
  echo "$tag rc=$? $(date -u +%T) temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1)"
  kill $qw 2>/dev/null; wait $qw 2>/dev/null
  python3 $J/overlap.py $O/$tag.json $O/compilers-$tag.log
}
C=()
if on nemo-pf; then
  C=(); for r in 128 1024 8192; do for p in 0 8192 32768 131072; do C+=("32,2,128,1,$r,$((p + r))"); done; done
  timed nemo-pf $O/nemo-pf.cubin "$NEMO" $PFL "${C[@]}"
fi
if on nemo-dec; then
  C=(); for b in 1 4 8; do for t in 1 4; do for c in 1024 32768 131072; do C+=("32,2,128,$b,$t,$c"); done; done; done
  timed nemo-dec $O/nemo-dec.cubin "$NEMO" $DCL "${C[@]}"
fi
if on qwen-pf; then
  C=(); for r in 1024 8192; do for p in 0 8192 32768; do C+=("24,4,256,1,$r,$((p + r))"); done; done
  timed qwen-pf $O/qwen-pf.cubin "$QWEN" $PFL "${C[@]}"
fi
if on qwen-dec; then
  C=(); for b in 1 4 16; do for c in 1024 32768; do C+=("24,4,256,$b,1,$c"); done; done
  timed qwen-dec $O/qwen-dec.cubin "$QWEN" $DCL "${C[@]}"
fi
echo "job done $(date -u +%T)"
