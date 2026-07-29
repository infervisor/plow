#!/usr/bin/env bash
# Fourth lease. Narrow-op sizing round 2, on the emitter GLM actually uses.
#
#   n_rope  GLM_ROPE_FIT=1     - 156 HeadNormRope packets 256 -> 2 (q, nh_l=16 heads / 8 waves)
#                                and 256 -> 1 (k, shared single head), on DISJOINT slices because
#                                all 78 pairs overlap in the trace.
#   n_cmb   GLM_COMBINE_FIT=1  - 75 MoeCombine packets 256 -> ceil(6144/512) = 12.
#   n_both                     - do they compose?
#
# Both are emit_xreduce's bug shape on mla.rs: a packet handed 256 workgroups whose body can
# use a handful, so the rest poll the arrival counter, take the acquire fence and exit. Measured
# on the SHIPPING blob's own trace: rope burns 0.968 ms/CU/token in workgroups that do zero
# arithmetic, combine 0.439 -- against the 1.779 that the collective burned before xrfit, which
# was worth -1.82 ms of token.
#
# n_ctl.pkt is md5-identical to /home/lava/models/glm52_tp/glm52_tp4_64k.pkt (e818c91b), so both
# knobs are provably inert when unset and the control IS the shipping program.
# Control interleaved at positions 1 / 4 / 6 (§6b-STALE: re-derive your own baseline).
set -u
cd /home/lava/models/glm52_skew
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
W=/home/lava/models/GLM-5.2-plow
run () {
  echo "########## $1"
  PLOW_INTERP=i_base.elf ./glm52_decode "$1.pkt" "$W" --tp 4 --sweep 1024 --steps 65 --gen 24 2>&1
  echo "########## end $1 rc=$?"
}
run n_ctl
run n_rope
run n_cmb
run n_ctl
run n_both
run n_ctl
# AFTER-picture, deliberately after the timing A/B (the trace store costs ~1.6%).
echo "########## n_both-traced"
PLOW_INTERP=i_base.elf PLOW_TRACE_RAW=tr/nboth ./glm52_decode n_both.pkt "$W" \
    --tp 4 --sweep 1024 --steps 65 2>&1
echo "########## end n_both-traced rc=$?"
echo ALLDONE5
