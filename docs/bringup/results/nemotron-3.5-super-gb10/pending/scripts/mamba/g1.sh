#!/usr/bin/env bash
# g1.sh [phases] : (gpuq, jg02) gates for the grouped CHUNK_OUT body (PLOW_NV_MAMBA_OUT_GRP) from a source tree.
# T = recipe emit of the tree; C = the same assets with the five prefill objects rebuilt $CDEF
#   (default -DPLOW_NV_MAMBA_OUT_GRP=0); NOTE = the nv-block-ab note.
#   phases (default "build knob ab oracle pc ttft"): build = cargo + emits (full, L6 mamba, L6..8) + C variants;
#   knob = devgen / plowrt knob tests; ab = four-arm nv-block-ab L6 prefill 1024/4096/8192/16384 + certify;
#   oracle = block oracle L6 and L6..8 --kv fp8 --route-check (T); pc = packed-check 5 cases (T, L6..8);
#   ttft = full-model TTFT ABBA (TTFT_L, default 8K / 32K; GPU <= 58 C before each arm).
set -uo pipefail
PH=${1:-build knob ab oracle pc ttft}
CDEF=${CDEF:--DPLOW_NV_MAMBA_OUT_GRP=0}
NOTE=${NOTE:-Mamba-2 CHUNK_OUT grouped body (C B^T once per group, cp.async head stages)}
R=${ROOT:-/home/ifv/plow-mamba}
S=/home/ifv/plow-scratch/mamba
N=/home/ifv/.claude/jobs/72c86dac/tmp/nemotron
O=${O:-$S/g1}
B=$O/bin
PY=/home/ifv/venvs/vllm-gb10/bin/python
CK=/home/ifv/models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
export NEMOTRON35_CHECKPOINT=$CK PLOW_CHECKPOINT=$CK
export PLOW_VERIFY_BIN=/home/ifv/.claude/jobs/72c86dac/tmp/plow-gb10-build/lean-plow/.lake/build/bin/plow_verify
export CARGO_TARGET_DIR=$S/target
export PLOW_CAMPAIGN_NO_NIX=1  # nx.sh already provides the flake env; campaign.py must not nest nix develop
RECIPE=scripts/campaign/recipes/nemotron-3.5-super.gb10.nvfp4.toml
PFC="interp_sm120_pf.cubin interp_sm120_pfseg.cubin interp_sm120_pfpackedseg.cubin interp_sm120_pfgemm.cubin interp_sm120_pfpackedgemm.cubin"
Q=$N/mtp/quiet.sh
mkdir -p $O $B
t0=$(date +%s)
ts() { echo "[$(( $(date +%s) - t0 ))s] $*"; }
on() { [[ " $PH " == *" $1 "* ]]; }
cd $R || exit 1
if on build; then
  ts "build: cargo"
  bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh bash -c "
  cargo build --release -p plowc && cargo build --release -p plowrt --features cuda --bin plowrt --example block_run --example step_bench
  echo cargo rc=\$?
  for b in plowc plowrt examples/block_run examples/step_bench; do cp \$CARGO_TARGET_DIR/release/\$b $B/; done
  " > $O/cargo.log 2>&1
  grep -E "cargo rc|^error" $O/cargo.log | head
  for b in plowc plowrt block_run step_bench; do [ -x $B/$b ] || { echo "FATAL: missing $B/$b"; tail -n 20 $O/cargo.log; exit 2; }; done
  ts "build: emits"
  rm -rf $O/full $O/b6 $O/b69
  bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh bash -c "
  python3 scripts/campaign/campaign.py build $RECIPE --out $O/full --no-probe > $O/full.log 2>&1 &
  python3 scripts/campaign/campaign.py build $RECIPE --block 6..7 --out $O/b6 --no-probe > $O/b6.log 2>&1 &
  python3 scripts/campaign/campaign.py build $RECIPE --block 6..9 --out $O/b69 --no-probe > $O/b69.log 2>&1 &
  wait
  python3 scripts/check_recipe.py $RECIPE --bundle $O/full 2>&1 | tail -n 4
  " 2>&1 | grep -v "Git tree\|dev shell\|toolchain\|sccache"
  for a in full b6 b69; do echo "emit $a: $(ls $O/$a/assets/model.pkt 2>/dev/null) $(grep -h -E 'panicked|Error' $O/$a.log | head -3)"; done
  for a in full b6 b69; do
    for f in model.pkt $PFC; do
      [ -s $O/$a/assets/$f ] || { echo "FATAL: emit $a missing $f"; tail -n 20 $O/$a.log; exit 2; }
    done
  done
  $B/plowrt disasm $O/b6/assets/model.pkt 2>/dev/null | grep -E "Mamba2ChunkOut" | head -3
  ts "build: C variants"
  for a in full b6 b69; do
    rm -rf $O/$a-C
    bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh env EXTRA="$CDEF" bash $N/dec7/variant3.sh $O/$a/assets $O/$a-C $R/runtime $PFC \
      > $O/v-$a-C.log 2>&1 &
  done
  wait
  for a in full b6 b69; do echo "variant $a-C: $(ls $O/$a-C/*.cubin 2>/dev/null | wc -l) cubins $(grep -m1 -i ' error' $O/v-$a-C.log)"; done
  for a in full b6 b69; do
    for f in $PFC; do
      [ -s $O/$a-C/$f ] && [ ! -L $O/$a-C/$f ] || { echo "FATAL: variant $a-C missing rebuilt $f"; tail -n 20 $O/v-$a-C.log; exit 2; }
    done
  done
fi
if on knob; then
  for spec in "-p devgen --lib knob" "-p plowrt --features cuda,hsa --lib knob"; do
    # shellcheck disable=SC2086
    bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh cargo test -q $spec > $O/knob.log 2>&1
    echo "cargo test $spec rc=$? $(grep -E '^test result' $O/knob.log | tail -n 1)"
    grep -E "FAILED|panicked|^error" $O/knob.log | head -5
  done
fi
source /home/ifv/plow-scratch/nemotron/moepf/env.sh
$Q wait 1800
$Q watch "$O/compilers.log" & QW=$!
trap 'kill $QW 2>/dev/null' EXIT
BR=$B/block_run
if on ab; then
  AB=$O/ab-b6
  rm -rf $AB
  bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh python3 scripts/campaign/campaign.py nv-block-ab --control $O/b6-C --treatment $O/b6/assets \
    --verb prefill-bench --args "--query-rows ${AB_ROWS:-1024,4096,8192,16384} --prior 0 --iters ${ABIT:-7} --warmup 2" \
    --env PLOW_CHECKPOINT=$CK --block-run $BR --plowrt $B/plowrt \
    --ceilings /home/ifv/plow-scratch/res/ceilings-gb10.json --out $AB --dry-run \
    --note "$NOTE" 2>&1 | grep "wrote\|rror"
  t1=$(date +%s)
  bash $AB/run.sh > $AB.out 2>&1; echo "ab-b6 rc=$? $($Q report $O/compilers.log $t1)"
  grep -h "| L\|| prefill" $AB/comparison.md | head -12
  bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh python3 scripts/campaign/certify_kernel.py --input $AB/comparison.json 2>&1 | tail -n 4
fi
if on oracle; then
  o() { local name=$1 a=$2; shift 2; local t1; t1=$(date +%s)
    timeout 2400 $PY $R/runtime/tests/nemotron_block_oracle.py --asset $a --block-run $BR --ckpt $CK --scratch $O/or-$name "$@" \
      > $O/or-$name.log 2>&1
    echo "or-$name rc=$? ok=$(grep -c ' ok$' $O/or-$name.log) $($Q report $O/compilers.log $t1)"
    grep -v " ok$" $O/or-$name.log | tail -n 6; rm -rf $O/or-$name/*.npy; }
  o b6 $O/b6/assets --layers 6
  o b69 $O/b69/assets --layers 6,7,8 --kv fp8 --route-check --prefill 7,128,1000,1500,1537,2048,3000,6000,8192,12000,16384
fi
if on pc; then
  pc() { local n=$1; shift
    timeout 600 $BR $O/b69/assets packed-check "$@" > $O/pc-$n.log 2>&1
    echo "pc $n rc=$? | $(grep -h -E 'packed KV bytes|sparse-slot KV:|parity=PASS|Error' $O/pc-$n.log | sed 's/^ *//' | tr '\n' '|')"; }
  pc default
  pc mix8 --lens 1,2,130,127,129,256,500,3 --starts 0,64,0,1000,5,0,128,77
  pc big3 --lens 2000,1500,700
  pc q4 --requests 4 --query-rows 128 --kv-base 1024
  pc mix4 --lens 33,300,1000,7
fi
if on ttft; then
  export PLOW_VMM_LIVE=1 PLOW_HOST_KVCACHE_SIZE=0
  PR=/home/ifv/plow-scratch/nemotron/dec4/prompts256k.json
  temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader; }
  cool() { local n=0; while [ "$(temp)" -gt 58 ] && [ $n -lt 120 ]; do sleep 10; n=$((n+1)); done; }
  run() { # assets arm L suffix
    cool; local t1; t1=$(date +%s)
    ts "ttft $2 L=$3 start temp $(temp)"
    timeout 1800 $B/step_bench $1 2 $3 0 --prompt-tokens $PR --warmup 0 > $O/ttft-$2-L$3$4.log 2>&1 || echo "rc=$?"
    grep -E "prompt consumed" $O/ttft-$2-L$3$4.log | tr '\n' ' '; echo "end temp $(temp) $($Q report $O/compilers.log $t1)"; }
  for L in ${TTFT_L:-8192 32768}; do
    run $O/full-C C $L a; run $O/full/assets T $L a; run $O/full/assets T $L b; run $O/full-C C $L b
  done
fi
ts "g1 done"
