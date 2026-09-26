#!/usr/bin/env bash
# Build single-define object variants of veena-v5m16 for the B=1 GEMV sweep.
set -e
B=/root/tts-work/assets/veena-v5m16
mk() { bash /root/tts-work/mk_trace_asset.sh $B /root/tts-work/assets/sw-$1 "${@:2}" | grep -c built | sed "s/^/sw-$1 built /"; }
mk ctl -DGV_MM_MAX=16
mk rb8 -DGV_MM_MAX=16 -DGV_RB=8 -DGV_UNROLL_RB=8
mk un16 -DGV_MM_MAX=16 -DGV_UNROLL=16
mk noxreg -DGV_MM_MAX=16 -DPLOW_NV_GEMV_XREG=0
