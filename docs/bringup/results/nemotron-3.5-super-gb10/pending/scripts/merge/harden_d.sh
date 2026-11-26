#!/usr/bin/env bash
# harden_d.sh [phases] : compile-only regression check, a7b444bad vs HEAD+merge-harden.patch.
# Phases: plowc (release plowc per tree), emit (CPU emit devblob+cubin per recipe per tree, ptxas -v),
# res (per-cubin resource usage). Both trees build with the HEAD flake's CUDA 12.9 env (devenv.sh,
# captured by `nix print-dev-env`; the a7b444bad flake has no aarch64 CUDA shell, and userns is
# restricted on this boot), so only the sources differ.
PH=${1:-plowc emit res}
M=/home/ifv/.claude/jobs/72c86dac/tmp/merge
O=/home/ifv/plow-scratch/merge/objs
J=/home/ifv/.claude/jobs/72c86dac/tmp/refactor-r1
source $M/devenv.sh > /dev/null 2>&1
[ -n "${NIX_BUILD_TOP:-}" ] && rmdir "$NIX_BUILD_TOP" 2>/dev/null
unset TMPDIR TMP TEMP TEMPDIR NIX_BUILD_TOP
# campaign.py would wrap each step in `nix develop`, which cannot nest under this job's proot; the
# flake env is already loaded from devenv.sh, so its steps run directly.
export CUDA_VISIBLE_DEVICES= PLOW_CAMPAIGN_NO_NIX=1
export PLOW_VERIFY_BIN=/home/ifv/plow/.claude/worktrees/gb10-sm121a/lean-plow/.lake/build/bin/plow_verify
export QWEN38_CHECKPOINT=/home/ifv/models/Qwen/Qwen3.8-27B
export HF_HUB_CACHE=/home/ifv/plow-scratch/hfhub
export NVCC_APPEND_FLAGS="-Xptxas -v"
declare -A TREE=([base]=/home/ifv/plow-scratch/merge/base [hard]=/home/ifv/plow-scratch/merge/hard)
declare -A TGT=([base]=/home/ifv/plow-target-mergebase [hard]=/home/ifv/plow-target-harden)
declare -A R=(
  [p-asr06-l4]=recipes/infervisor/qwen3-asr-0.6b/sm89-l4-tp1.toml
  [p-qwen38-h100-bf16]=recipes/infervisor/qwen3.8-27b/sm90a-h100-tp1.toml
  [p-asr06-5090]=recipes/infervisor/qwen3-asr-0.6b/sm120a-rtx5090-tp1.toml
)
mkdir -p "$O/recipes"
on() { [[ " $PH " == *" $1 "* ]]; }
date -u +%T
if on plowc; then
  for t in base hard; do
    s=$(date +%s)
    ( cd "${TREE[$t]}" && CARGO_TARGET_DIR=${TGT[$t]} cargo build --release -p plowc > "$O/plowc-$t.log" 2>&1 )
    echo "plowc $t rc=$? secs=$(( $(date +%s) - s ))"
  done
fi
if on emit; then
  for n in p-asr06-l4 p-asr06-5090 p-qwen38-h100-bf16; do
    for t in base hard; do
      python3 "$J/prep_recipe.py" "${TREE[$t]}/${R[$n]}" "$O/recipes/$n.$t.toml"
      rm -rf "$O/$t/$n"; mkdir -p "$O/$t"
      s=$(date +%s)
      ( cd "${TREE[$t]}" && CARGO_TARGET_DIR=${TGT[$t]} python3 scripts/campaign/campaign.py build "$O/recipes/$n.$t.toml" --out "$O/$t/$n" --no-probe ) > "$O/$t/$n.log" 2>&1
      rc=$?
      nc=$(ls "$O/$t/$n/assets"/*.cubin 2>/dev/null | wc -l)
      echo "emit $t $n rc=$rc secs=$(( $(date +%s) - s )) cubins=$nc mem_avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)G"
      if [ "$rc" != 0 ] || [ ! -s "$O/$t/$n/assets/model.pkt" ] || [ "$nc" = 0 ]; then
        echo "EMIT FAILED $t $n: rc=$rc model.pkt=$([ -s "$O/$t/$n/assets/model.pkt" ] && echo ok || echo MISSING) cubins=$nc"
        grep -m5 -E "error|Operation not permitted|Traceback" "$O/$t/$n.log"
      fi
    done
  done
fi
if on res; then
  CUOBJ="$(dirname "$PLOW_NVCC")/cuobjdump"
  for t in base hard; do
    for c in "$O/$t"/*/assets/*.cubin; do
      "$CUOBJ" --dump-resource-usage "$c" > "$c.res" 2>&1
    done
  done
  python3 -I "$M/res_diff.py" "$O" p-asr06-l4 p-asr06-5090 p-qwen38-h100-bf16
fi
date -u +%T
