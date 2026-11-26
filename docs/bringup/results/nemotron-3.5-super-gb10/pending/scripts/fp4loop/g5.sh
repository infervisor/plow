#!/usr/bin/env bash
# g5.sh : (gpuq, jg02) cand3 = cand2 (maps only for prefill buckets >= 2048) + the TMA walks as their own kernel
# (`plow_moe_w4a4_pf_tma`, plowrt launches it for i7 != 0), so the cp.async entry keeps HEAD's codegen.
# Cargo tests; g1 build; SASS check of `plow_moe_w4a4_pf_pfseg`: T (TMA=1) vs C (TMA=0) vs H (HEAD runtime),
# stop unless T == C; then L1 A/B 128..16384, oracle T and C, packed-check, TTFT ABBA 8K / 32K.
set -uo pipefail
F=/home/ifv/.claude/jobs/72c86dac/tmp/fp4loop
O=/home/ifv/plow-scratch/fp4loop/g5
W=/home/ifv/plow/.claude/worktrees/gb10-sm121a
C=/home/ifv/plow-fp4loop/cand3
N=/home/ifv/.claude/jobs/72c86dac/tmp/nemotron
export CARGO_TARGET_DIR=/home/ifv/plow-scratch/fp4loop/target
mkdir -p $O
cd $C || exit 1
bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh bash -c "
cargo test -p packet --lib > $O/t-packet.log 2>&1; echo packet rc=\$?
cargo test -p plow-asset --lib > $O/t-asset.log 2>&1; echo plow-asset rc=\$?
cargo test -p devgen --lib knob > $O/t-devgen-knob.log 2>&1; echo devgen-knob rc=\$?
cargo test -p devgen --lib nemotron > $O/t-devgen-nemo.log 2>&1; echo devgen-nemotron rc=\$?
cargo test -p plowrt --features cuda,hsa --lib knob > $O/t-plowrt-knob.log 2>&1; echo plowrt-knob rc=\$?
" 2>&1 | grep -E "rc=" | tee $O/t-rc.txt
for f in $O/t-*.log; do echo "== $f"; grep -E "^test result|FAILED|^error" $f | head -6; done
fails=$(grep -vc " rc=0$" $O/t-rc.txt)
echo "test logs failing=$fails"
[ "$fails" = 0 ] && [ "$(wc -l < $O/t-rc.txt)" = 5 ] || exit 1
ROOT=$C O=$O bash $F/g1.sh build || exit 1
[ -s $O/b1/assets/interp_sm120_pfseg.cubin ] || { echo "FATAL: no T pfseg cubin"; exit 1; }
rm -rf $O/h $O/b1-H && mkdir -p $O/h && git -C $W archive HEAD runtime | tar -x -C $O/h
bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh bash -c "
bash $N/dec7/variant3.sh $O/b1/assets $O/b1-H $O/h/runtime interp_sm120_pfseg.cubin > $O/v-b1-H.log 2>&1
CU=\$(dirname \$(command -v nvcc))/cuobjdump
for a in b1/assets b1-C b1-H; do
  n=\$(echo \$a | tr / -)
  \$CU -sass -fun plow_moe_w4a4_pf_pfseg $O/\$a/interp_sm120_pfseg.cubin > $O/sass-w4-\$n.txt 2>&1
  \$CU -res-usage $O/\$a/interp_sm120_pfseg.cubin 2>&1 | grep -A1 -E 'Function (plow_moe_w4a4|_Z18interp)' > $O/res-\$n.txt
  \$CU -sass -fun plow_moe_w4a4_pf_tma_pfseg $O/\$a/interp_sm120_pfseg.cubin > $O/sass-w4tma-\$n.txt 2>&1
done
" 2>&1 | grep -iE " error" | head
for n in b1-assets b1-C b1-H; do
  echo "$n: $(grep -c . $O/sass-w4-$n.txt) lines, md5 $(md5sum < $O/sass-w4-$n.txt | cut -c1-12), tma kernel lines $(grep -c UTMALDG $O/sass-w4tma-$n.txt) UTMALDG | $(tr '\n' ' ' < $O/res-$n.txt | tr -s ' ')"
done
python3 $F/sasseg.py $O/sass-w4tma-b1-assets.txt
[ -s $O/b1-H/interp_sm120_pfseg.cubin ] && [ "$(grep -c OMMA $O/sass-w4-b1-C.txt)" -gt 0 ] || { echo "FATAL: empty SASS dump"; exit 1; }
cmp -s $O/sass-w4-b1-assets.txt $O/sass-w4-b1-C.txt || { echo "cp.async kernel SASS T != C: stop"; exit 1; }
echo "cp.async kernel SASS T == C"
ROOT=$C O=$O ABIT=10 ORC=1 AB_ROWS=128,1024,2048,2176,4096,8192 TTFT_L="8192 32768" bash $F/g1.sh "ab oracle pc ttft"
