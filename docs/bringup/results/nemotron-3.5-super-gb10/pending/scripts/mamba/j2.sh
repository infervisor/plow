#!/usr/bin/env bash
# j2.sh (gpuq, jg02): CHUNK_OUT grouped body (dev tree /home/ifv/plow-mamba) vs HEAD body.
#   build : mamba2_ops cubins old (PLOW_NV_MAMBA_OUT_GRP=0) / new, ptxas -v, SASS counts
#   dev   : dev_out.py bitwise + full-GPU / 1-SM timing; mamba2_oracle.py --pf tc on the new cubin
#   ub    : run_ubench_cu.sh nemotron-pf mamba2_chunk_out (+ scan_state), base tree vs dev tree
# Args: [phases: build dev ub]
set -uo pipefail
PH=${1:-build dev ub}
D=/home/ifv/plow-mamba
B=/home/ifv/plow-mamba-base
S=/home/ifv/plow-scratch/mamba
M=/home/ifv/.claude/jobs/72c86dac/tmp/mamba
Q=/home/ifv/.claude/jobs/72c86dac/tmp/nemotron/mtp/quiet.sh
PY=/home/ifv/venvs/vllm-gb10/bin/python
CK=/home/ifv/models/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
O=$S/${JOB:-j2}
mkdir -p "$O"

on() { [[ " $PH " == *" $1 "* ]]; }
date -u +%T
if on build; then
  for v in 0 1 2 3 4 5 6; do
    case $v in 0) DF=-DPLOW_NV_MAMBA_OUT_GRP=0 ;; 1) DF="" ;; 2) DF=-DMPF_OG_RBMAP=0 ;; 3) DF=-DMPF_OG_NV=1 ;;
      4) DF="-DMPF_OG_NV=1 -DMPF_OG_SPLITP=1" ;; 5) DF=-DMPF_OG_DIAG=1 ;; 6) DF=-DMPF_OG_DIAG=2 ;; esac
    (cd $D && bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh bash -c 'bash runtime/cmake/nvcc_cubin.sh "$PLOW_NVCC" '$O'/ops'$v'.cubin mamba2_chunk_out_bf_tc_k required -arch=sm_121a -O3 -cubin -Xptxas -v '$DF' -I runtime/common -I runtime/nvidia runtime/tests/mamba2_ops_kernels.cu > '$O'/nvcc'$v'.log 2>&1; echo "nvcc grp='$v' rc=$?"
cuobjdump -sass '$O'/ops'$v'.cubin > '$O'/ops'$v'.sass')
    grep -E "error" $O/nvcc$v.log | head -5
    grep -A2 -E "chunk_out" $O/nvcc$v.log | grep -vE "^--" | head -24
  done
  for v in 0 1 2 3 4 5 6; do [ -s $O/ops$v.cubin ] || { echo "FATAL: ops$v.cubin missing"; tail -n 20 $O/nvcc$v.log; exit 2; }; done
  $PY -I - "$O/ops1.sass" <<'EOF'
import re, sys
txt = open(sys.argv[1]).read()
for blk in re.split(r"\n\s*Function : ", txt)[1:]:
    name = blk.split("\n", 1)[0].strip()
    if "chunk_out" not in name: continue
    c = {k: len(re.findall(k, blk)) for k in ("HMMA", "LDSM", "LDGSTS", "MUFU.EX2", "BAR.SYNC", "LDL", "STL")}
    print("sass", name[:60], c)
EOF
fi
if on dev; then
  $Q wait 1800
  $Q watch "$O/compilers-dev.log" & QW=$!; t0=$(date +%s)
  timeout 1800 $PY $M/dev_out.py $D $O/ops0.cubin $O/ops1.cubin $O/ops2.cubin $O/ops3.cubin $O/ops4.cubin $O/ops5.cubin $O/ops6.cubin > $O/dev_out.log 2>&1
  echo "dev_out rc=$?"; grep -v "^$" $O/dev_out.log | tail -n 80
  timeout 1800 $PY $D/runtime/tests/mamba2_oracle.py --cubin $O/ops1.cubin --ckpt $CK --pf tc --json $O/oracle.json > $O/oracle.log 2>&1
  echo "oracle rc=$? $(grep -c FAIL $O/oracle.log) FAIL rows"; grep -E "FAIL|scan fused|packed" $O/oracle.log | head -20
  kill $QW 2>/dev/null
  echo "dev: $($Q report $O/compilers-dev.log $t0) $(date -u +%T)"
fi
if on ub; then
  U=/home/ifv/plow-scratch/ubench
  NA=$U/nemotron/assets; NL=$U/nemotron/build.log
  for tr in base dev; do
    T=$D; [ $tr = base ] && T=$B
    (cd $T && bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh env BUILD_DIR=$S/ub-$tr bash runtime/ubench/run_ubench_cu.sh --nv --build-only --set nemotron-pf --assets $NA --build-log $NL --object interp_sm120_pf) > $O/ub-build-$tr.log 2>&1
    echo "ub build $tr rc=$?"; grep -E " error|built|refuse" $O/ub-build-$tr.log | head -3
  done
  $Q wait 600
  $Q watch "$O/compilers-ub.log" & QW=$!; t0=$(date +%s)
  for c in mamba2_chunk_out mamba2_chunk_scan_state; do
    for tr in base dev; do
      T=$D; [ $tr = base ] && T=$B
      timeout 900 bash -c "cd $T && bash /home/ifv/.claude/jobs/72c86dac/tmp/tcinv/nx.sh env BUILD_DIR=$S/ub-$tr bash runtime/ubench/run_ubench_cu.sh --nv --set nemotron-pf --assets $NA --build-log $NL --object interp_sm120_pf --case $c --reps 7 --json $O/ub-$tr.$c.json" > $O/ub-$tr.$c.log 2>&1
      echo "$tr $c rc=$? $(grep "^| [a-z]" $O/ub-$tr.$c.log | grep -v '^| case')"
      $PY -I -c "import json;d=json.load(open('$O/ub-$tr.$c.json'))['cases'][0];print('  digest',d.get('digest'),'verify',d['verify'])" 2>/dev/null
    done
  done
  kill $QW 2>/dev/null
  echo "ub: $($Q report $O/compilers-ub.log $t0) $(date -u +%T)"
fi
echo "j2 done $(date -u +%T)"
