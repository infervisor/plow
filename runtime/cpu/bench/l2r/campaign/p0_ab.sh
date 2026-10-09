#!/usr/bin/env bash
# p0_ab.sh: explain the P0 c1 TPOT gap against the source baseline (E2B, ISL 1000, OSL 128, c1, 2 reps).
#   a: source bundle (n_cu 96, max_ctx 2048), 96 threads on cpus 0-95
#   b: pk90 (n_cu 90, max_ctx 16384), isolated 90 cores
#   c: pk90-2k (n_cu 90, max_ctx 2048), isolated 90 cores
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p0_ab; mkdir -p $R
M=${M:-gemma-4-E2B-it}
d=/tmp/g4c/l2r/pk90-2k/$M
if [ ! -s $d/model.pkt ]; then
  rm -rf $d; mkdir -p $d
  extra=""; [ $M = gemma-4-E4B-it ] && extra="PLOW_DENSE_PF_NS_MIN=2"
  env $extra PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 PLOW_MAX_CHUNK=2048 /tmp/g4c/l2r/bin/plowc --hf-dir /tmp/models/google/$M \
    --gpu xeon6975p --arch amx --n-cu 90 --emit devblob --max-ctx 2048 --out $d > $d.emit.log 2>&1 || { echo "emit failed"; exit 1; }
  ln -sfn /tmp/models/google/$M $d/checkpoint
  cp /tmp/models/google/$M/tokenizer.json $d/
fi
cat > /tmp/g4c/l2r/bin/plowrt-all <<'EOF'
#!/usr/bin/env bash
exec taskset -c 0-95 env PLOW_CPU_THREADS=96 /tmp/g4c/l2r/bin/plowrt "$@"
EOF
chmod +x /tmp/g4c/l2r/bin/plowrt-all
cd $WT
SHA=$(git rev-parse --short=12 HEAD)
run() {
  local tag=$1 assets=$2 rt=$3
  PLOW_CPU_WEIGHT_AFFINE=1 ASSETS=$assets PLOWRT=$rt PLOWRT_GIT_SHA=$SHA CONCS=1 ISL=1000 OSL=128 REPS=2 \
    /tmp/g4c/grid.sh plow $M $R/$M.$tag > $R/$M.$tag.log 2>&1
  echo "$tag rc=$?"
}
run a /tmp/g4c/bundles/$M /tmp/g4c/l2r/bin/plowrt-all
run b /tmp/g4c/l2r/pk90/$M /tmp/g4c/l2r/bin/plowrt-iso
run c $d /tmp/g4c/l2r/bin/plowrt-iso
echo AB_DONE
