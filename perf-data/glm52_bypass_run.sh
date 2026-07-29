#!/usr/bin/env bash
# Chain-bypass CEILING sweep for tile-granular partial-completion signalling.
# Every arm keeps 2756 packets and 259,505 workgroup-packets (graphstat verified);
# only chain depth / gate structure changes. Control interleaved at 4 positions so the
# deltas do not rest on drift (§6b-STALE).
set -u
cd /home/lava/models/glm52_bypass
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
W=/home/lava/models/GLM-5.2-plow
run () {
  echo "########## $1"
  PLOW_INTERP=i_base.elf ./glm52_decode "$1.pkt" "$W" --tp 4 --sweep 1024 --steps 65 --gen 24 2>&1
  echo "########## end $1 rc=$?"
}
run ctl
run b_spine3
run b_resid
run b_rms
run ctl
run b_all4
run b_comb
run b_xr
run ctl
run b_rope
run b_spine
run ctl
echo ALLDONE
