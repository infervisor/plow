#!/usr/bin/env bash
# p0_ab2.sh: reproduce the source c1 cell (E2B ISL 1000 c1, 10.65 ms TPOT p50) under the current host state.
#   d: source assets (4fea6d8c3932) + source plowrt 9f1cef09, no taskset (as the final report)
#   e: source assets + l2r plowrt (dc497fb0), no taskset
#   f: pk90-2k assets + source plowrt, isolated 90 cores
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p0_ab; mkdir -p $R
M=gemma-4-E2B-it
SRC=/tmp/g4c/build/$M/assets
cat > /tmp/g4c/l2r/bin/plowrt-src-iso <<'EOF'
#!/usr/bin/env bash
exec taskset -c 2-31,34-63,66-95 env PLOW_CPU_THREADS=90 /tmp/plow-target/release/plowrt "$@"
EOF
chmod +x /tmp/g4c/l2r/bin/plowrt-src-iso
cd $WT
SHA=$(git rev-parse --short=12 HEAD)
run() {
  local tag=$1 assets=$2 rt=$3
  PLOW_CPU_WEIGHT_AFFINE=1 ASSETS=$assets PLOWRT=$rt PLOWRT_GIT_SHA=$SHA CONCS=1 ISL=1000 OSL=128 REPS=2 \
    /tmp/g4c/grid.sh plow $M $R/$M.$tag > $R/$M.$tag.log 2>&1
  echo "$tag rc=$?"
}
run d $SRC /tmp/plow-target/release/plowrt
run e $SRC /tmp/g4c/l2r/bin/plowrt
run f /tmp/g4c/l2r/pk90-2k/$M /tmp/g4c/l2r/bin/plowrt-src-iso
echo AB2_DONE
