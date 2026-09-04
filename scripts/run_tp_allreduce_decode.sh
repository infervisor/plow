#!/usr/bin/env bash
# run_tp_allreduce_decode.sh <bindir> <mode> [oracle] [probe] [iters] [hidden] [nwg] [gpus...]
#
# Runs scripts/build_tp_allreduce.sh's tp_allreduce_bench in one decode arm under the clean
# host environment the harness needs, keeping the visible-device variables a
# `perf-data/tools/gpulease -n <k>` wrapper exports. Modes: oneshot (hot control), cold
# (control with a per-iteration producer + local gate), tagged, tagged_cold. Oracle: benign
# (default) or order (strict rank-0..N-1 f32 order). Default grid is the emitted decode
# collective: 14 workgroups at hidden 7168, 7 at 3584.
#
#   perf-data/tools/gpulease -n 8 xr-tagged \
#     scripts/run_tp_allreduce_decode.sh /tmp/xr-tag-bench tagged_cold order 1 4000 7168 14
set -euo pipefail
BIN="${1:?bindir}"; MODE="${2:?mode}"; ORACLE="${3:-benign}"; PROBE="${4:-0}"
ITERS="${5:-4000}"; HIDDEN="${6:-7168}"; NWG="${7:-}"
shift $(( $# < 7 ? $# : 7 ))
[ -n "$NWG" ] || { NWG=14; [ "$HIDDEN" = 3584 ] && NWG=7; }
if [ $# -gt 0 ]; then GPUS="$*"; else GPUS="0 1 2 3 4 5 6 7"; fi
cd "$BIN"
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" LD_LIBRARY_PATH=/opt/rocm/lib \
  ${ROCR_VISIBLE_DEVICES:+ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"} \
  ${HIP_VISIBLE_DEVICES:+HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"} \
  TP_MODE="$MODE" TP_ORACLE="$ORACLE" TP_PROBE="$PROBE" TP_ITERS="$ITERS" \
  TP_HIDDEN="$HIDDEN" TP_NWG="$NWG" ./tp_allreduce_bench $GPUS
