#!/bin/sh
# B4 single-block real-weight oracle gate for the MLA merge+fold rewrite.
#
# Arms: {separate epilogue, fused epilogue} x {baseline object, rewritten object}. The separate
# arm is the regression control (it never reaches d_mla_merge_fold, so it must not move); the fused
# arm is the one that covers the rewrite, and `MLA attn_out` is the stage directly downstream of it.
# Tolerance 1.5e-2 bf16 (glm52_real_block_gfx950_test.c:346).
#
# TWO OPERATIONAL RULES, both learned the hard way:
#  - Each arm is its OWN invocation, and each is killed once its bf16 verdict prints. The harness's
#    SECOND pass (fp8) hangs at CU occupancy 2 with the host asleep — a pre-existing fault in this
#    instrument, unrelated to the fold (it reproduces on the untouched baseline object with the
#    separate epilogue). Every stage merge+fold feeds is reported by the bf16 pass, so that pass is
#    the gate, and one hung arm must not cost the arms behind it.
#  - Nothing is piped through a filter. The harness sets `setbuf(stdout,NULL)` precisely so its
#    progress is visible live; a `grep` without --line-buffered in the pipeline re-buffers it and
#    makes a working run look byte-for-byte identical to a hung one.
# Kill discipline: the chain is gpulease -> sg render -> binary, so `rocm-smi --showpids` is the
# only reliable way to confirm the GPU was actually released.
set -e
D=/home/lava/models/glm52b4
cd "$D"
LD_LIBRARY_PATH=/opt/rocm/lib
export LD_LIBRARY_PATH
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

arm() { # <elf> <fused> <label>
    out=/tmp/b4arm_$3.txt
    echo "############ $3: elf=$1 GLM_FUSED_FOLD=$2"
    GLM_FUSED_FOLD=$2 ./glm52_real_test "$1" glm52_real_fixture.bin > "$out" 2>&1 &
    p=$!
    i=0
    while [ $i -lt 300 ]; do
        grep -q "=> " "$out" 2>/dev/null && break
        kill -0 $p 2>/dev/null || break
        i=$((i + 1))
        sleep 10
    done
    sleep 2
    kill -9 $p 2>/dev/null || true
    wait $p 2>/dev/null || true
    sed -n '/latent epilogue/,/=> /p' "$out"
    echo
}

arm interp_b4_mfold.elf 1 mfold_fused
arm interp_b4_base.elf  1 base_fused
arm interp_b4_mfold.elf 0 mfold_separate
