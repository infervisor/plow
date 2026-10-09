#!/usr/bin/env bash
# p0_ab5.sh: host/tokio threads sharing a worker core?
#   k: rtx90-2k, workers on 2-31,34-63,66-95, their SMT siblings allowed (host threads can float there)
#   l: rtx96-2k, taskset 0-95 (no siblings)
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p0_ab
M=gemma-4-E2B-it
cat > /tmp/g4c/l2r/bin/plowrt-iso-smt <<'EOF'
#!/usr/bin/env bash
exec taskset -c 2-31,34-63,66-95,98-127,130-159,162-191 env PLOW_CPU_THREADS=90 /tmp/g4c/l2r/bin/plowrt "$@"
EOF
chmod +x /tmp/g4c/l2r/bin/plowrt-iso-smt
cd $WT
SHA=$(git rev-parse --short=12 HEAD)
run() {
  local tag=$1 assets=$2 rt=$3
  PLOW_CPU_WEIGHT_AFFINE=1 ASSETS=$assets PLOWRT=$rt PLOWRT_GIT_SHA=$SHA CONCS=1 ISL=1000 OSL=128 REPS=2 \
    /tmp/g4c/grid.sh plow $M $R/$M.$tag > $R/$M.$tag.log 2>&1
  echo "$tag rc=$?"
}
run k /tmp/g4c/l2r/rtx90-2k/$M /tmp/g4c/l2r/bin/plowrt-iso-smt
run l /tmp/g4c/l2r/rtx96-2k/$M /tmp/g4c/l2r/bin/plowrt-all
echo AB5_DONE
