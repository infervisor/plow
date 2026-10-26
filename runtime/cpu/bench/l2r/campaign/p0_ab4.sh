#!/usr/bin/env bash
# p0_ab4.sh: n_cu 90 packet vs isolation. j: rtx90-2k on cpus 0-29,32-61,64-93 (30/node, housekeeping cores used).
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p0_ab
M=gemma-4-E2B-it
cat > /tmp/g4c/l2r/bin/plowrt-90lo <<'EOF'
#!/usr/bin/env bash
exec taskset -c 0-29,32-61,64-93 env PLOW_CPU_THREADS=90 /tmp/g4c/l2r/bin/plowrt "$@"
EOF
chmod +x /tmp/g4c/l2r/bin/plowrt-90lo
cd $WT
PLOW_CPU_WEIGHT_AFFINE=1 ASSETS=/tmp/g4c/l2r/rtx90-2k/$M PLOWRT=/tmp/g4c/l2r/bin/plowrt-90lo PLOWRT_GIT_SHA=$(git rev-parse --short=12 HEAD) \
  CONCS=1 ISL=1000 OSL=128 REPS=2 /tmp/g4c/grid.sh plow $M $R/$M.j > $R/$M.j.log 2>&1
echo "j rc=$?"
