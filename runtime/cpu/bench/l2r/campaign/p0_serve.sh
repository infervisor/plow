#!/usr/bin/env bash
# p0_serve.sh: P0.3 serving baseline on the isolated worker set (90 inference cores, n_cu 90 packets).
#   E2B, E4B: concurrency 1/4/16 at ISL 1900 and 15900 (OSL 128), REPS 3. Raw under results/p0_serve/.
set -u
WT=/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16
R=/tmp/g4c/l2r/results/p0_serve; mkdir -p $R
T=/tmp/plow-target-l2r; export CARGO_TARGET_DIR=$T
cd $WT
[ -x /tmp/g4c/l2r/bin/plowrt ] || nix develop -c cargo build --release -p plowrt --no-default-features --features cpu < /dev/null > $R/build_rt.log 2>&1 || { echo "plowrt build failed"; exit 1; }
[ -x /tmp/g4c/l2r/bin/plowc ] || nix develop -c cargo build --release -p plowc < /dev/null > $R/build_plowc.log 2>&1 || { echo "plowc build failed"; exit 1; }
mkdir -p /tmp/g4c/l2r/bin && [ -x /tmp/g4c/l2r/bin/plowrt ] || cp $T/release/plowrt /tmp/g4c/l2r/bin/plowrt && cp $T/release/plowc /tmp/g4c/l2r/bin/plowc
cat > /tmp/g4c/l2r/bin/plowrt-iso <<'EOF'
#!/usr/bin/env bash
exec taskset -c 2-31,34-63,66-95,98-127,130-159,162-191 env PLOW_CPU_THREADS=90 /tmp/g4c/l2r/bin/plowrt "$@"
EOF
chmod +x /tmp/g4c/l2r/bin/plowrt-iso
for M in gemma-4-E2B-it gemma-4-E4B-it; do
  d=/tmp/g4c/l2r/pk90/$M; [ -s $d/model.pkt ] && continue; rm -rf $d; mkdir -p $d
  extra=""; [ $M = gemma-4-E4B-it ] && extra="PLOW_DENSE_PF_NS_MIN=2"
  env $extra PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32 PLOW_MAX_CHUNK=2048 /tmp/g4c/l2r/bin/plowc --hf-dir /tmp/models/google/$M \
    --gpu xeon6975p --arch amx --n-cu 90 --emit devblob --max-ctx 16384 --out $d > $d.emit.log 2>&1 || { echo "emit $M failed"; exit 1; }
  ln -sfn /tmp/models/google/$M $d/checkpoint
  [ -e $d/tokenizer.json ] || cp /tmp/models/google/$M/tokenizer.json $d/
done
for M in gemma-4-E2B-it gemma-4-E4B-it; do
  for isl in 1900 15900; do
    out=$R/$M/isl$isl; mkdir -p $R/$M
    PLOW_CPU_WEIGHT_AFFINE=1 PLOW_SESSION_SLACK=32 ASSETS=/tmp/g4c/l2r/pk90/$M PLOWRT=/tmp/g4c/l2r/bin/plowrt-iso \
      PLOWRT_GIT_SHA=$(git rev-parse --short=12 HEAD) CONCS="1 4 16" ISL=$isl OSL=128 REPS=3 \
      /tmp/g4c/grid.sh plow $M $out > $out.log 2>&1
    echo "$M isl $isl rc=$?"
  done
done
echo P0_SERVE_DONE
