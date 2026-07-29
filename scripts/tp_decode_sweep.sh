#!/usr/bin/env bash
# tp_decode_sweep.sh — drive the P1-D decode sweep across TP x ctx and assemble
# the table. One tp_decode process per TP degree (binds that many GPUs once, then
# sweeps every ctx), so the TP x ctx grid comes out as a set of rows.
#
#   scripts/tp_decode_sweep.sh <build-dir> <model.pkt> <model-dir> [tp-list] [ctx-list]
#
# defaults: tp-list "1 2 4 8"   ctx-list "1k,4k,8k,16k,32k,64k"
# (ctx values >= the pkt's compiled max_ctx are reported as skipped by the harness.)
#
# TP=1 is the honest single-GPU baseline. TP>1 rows are DP-replica timings (each
# GPU runs the FULL model) until tp-core's sharded packets land — they validate
# the N-device launch + concurrent step, not the weight-sharding speedup. Flip to
# real sharded packets (per-rank pkt) for the N x scaling numbers.
set -euo pipefail
DIR="${1:?build dir with tp_decode + interp_decode.elf}"
PKT="${2:?model.pkt}"
MODEL="${3:?model dir}"
TPS="${4:-1 2 4 8}"
CTXS="${5:-1k,4k,8k,16k,32k,64k}"
STEPS="${STEPS:-21}"
NDEV=$(ls /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l || echo 8)

echo "TP decode sweep — ctx list: $CTXS  (median of $STEPS)"
for tp in $TPS; do
  echo "============================================================"
  echo "TP=$tp"
  # ROCR_VISIBLE_DEVICES must be FORWARDED through `env -i`, not dropped.
  #
  # gpulease correctly exports it for the cards it leased, but `env -i` clears the
  # environment, so tp_decode never saw it and mapped rank r -> HSA agent r. A job that
  # leased [1 2 3 4] therefore ran on GPUs 0-3: off its own lease, on top of whatever else
  # held those cards, and the timing silently invalid. This is what produced a 28.0 ms
  # TP4 reading that re-ran at 11.74 once pinned correctly.
  #
  # Only ROCR_VISIBLE_DEVICES is forwarded. Deliberately NOT HIP_VISIBLE_DEVICES: setting
  # both breaks hipcc ("no ROCm-capable device is detected"), and this is the ROCr/HSA
  # path, so ROCR alone is both necessary and sufficient.
  sg render -c "cd $DIR && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME \
    ${ROCR_VISIBLE_DEVICES:+ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES} \
    LD_LIBRARY_PATH=/opt/rocm/lib ./tp_decode $PKT $MODEL --tp $tp \
    --sweep $CTXS --steps $STEPS" 2>&1 | sed -n '/SWEEP/,/^$/p'
done
