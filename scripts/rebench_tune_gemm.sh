#!/usr/bin/env bash
# Re-run the gfx950 prefill-GEMM tile campaign (knob-contract step 1).
#
# SUPERSEDED BY `plowc tune gemm`, which does this AND derives the shape list instead of carrying
# it by hand, AND ingests (so "measured but never published" cannot happen by omission):
#
#   sg render -c 'nix develop --command ./target/release/plowc \
#       --hf-dir <ckpt> --max-ctx <c> --n-cu 256 --num-gpus <g> \
#       tune gemm --gpu mi350 --root . --obj <objdir> --samples <out.jsonl> --lease'
#
# The `sg render -c` must be OUTSIDE `nix develop`: nix runs in a user namespace where root maps
# to `nobody`, so /usr/bin/newgrp's setuid bit is inert inside it and `sg` dies with
# `setgroups: Operation not permitted`.
#
# Validated against this script on 2026-07-29, same object and same 3-shape grid: 30 rows each,
# identical schema, row order, tile set and correctness verdicts — only the timing samples differ,
# which is what two runs of a measurement are. `--shapes auto` at `--max-ctx 131072` reproduces
# the 32-entry GLM block below EXACTLY, and at `--max-ctx 4096` correctly drops the M=8192 rung
# that ladder never instantiates. This script is kept until the campaign has been re-run through
# the new path on an UNCONTENDED box; it is not the recommended entry point.
#
# Two things `--shapes auto` establishes about the list below, which nobody could see before:
#   * `128 576 6144  glm52-kv-a-proj` is NOT in GLM's demand at any bucket on the shipping ladder
#     — the header's own suspicion that "N=576 is an N no ladder ever requests" is confirmed.
#     It is measured and never consulted.
#   * The list is a function of BOTH the ladder AND `--max-ctx`. The 32 GLM entries were derived
#     at ctx 131072 while rebench_emit_glm.sh's default emit is ctx 4096, where only 24 are asked
#     for. A superset is harmless; a subset is the bug this whole file's header is about.
#
# Every `interp.hip` / `op_gemm.h` / `op_moe.h` edit changes the PREPROCESSED build digest, so
# every record in `tuning/` goes stale and `pick_tile` silently reverts to the analytical model.
# `cargo test -p devgen --test tuned_tile_selection` is the signal; this is the fix.
#
# Two halves, deliberately: the C harness writes SAMPLES (it cannot know the build identity),
# `plowc tune ingest` attaches the probed digests and applies the store's gates.
#
#   $1  object dir holding the freshly built test_kernels.elf
#   $2  output jsonl
#
# Must run OUTSIDE nix (system ROCm) and UNDER a gpulease (it is a timing run).
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="${1:?object dir}"
JSONL="${2:?output jsonl}"

# 50 shapes, bf16. The trailing 18 are the ORIGINAL hand-authored set (Gemma-26B/12B/31B, Qwen,
# Kimi) — kept verbatim so this campaign stays comparable to the one it replaces. The leading 32
# are GLM-5.2's DERIVED demand; see below.
#
# ---------------------------------------------------------------------------------------------
# GLM-5.2 PREFILL WAS ENTIRELY UNMEASURED UNTIL 2026-07-29, and the reason is that this list is
# AUTHORED BY HAND while the compiler's actual demand was never read back.
#
# `PLOW_TUNE_DUMP=1` (crates/devgen/src/lib.rs) prints every shape the compiler asks about and
# whether the store answered. When first run on a GLM-5.2 TP4 emit it printed 32 distinct shapes —
# 8 distinct (N,K) at 4 bucket M values — and **ALL 32 WERE MISS**. Not one hit. Every M>=256
# record in the store was a Gemma-31B or Qwen shape (K in {2560,4096,5376,8192,21504}); GLM's K
# is 6144 and its only two records were M=128 — and N=576, an N no ladder ever requests.
#
# LADDER CORRECTION, and the shape list moved with it: the top rung was briefly 32768, which
# CANNOT RUN. `MAX_CHUNK = 8192` (crates/plowrt/src/exec/amd.rs:718) and `plan_chunks` (:735)
# filters `b <= MAX_CHUNK`, so a 32768 bucket is dropped at runtime while still costing `act.part`
# device memory at emit (T*top_k*hidden f32, ~6.4 GiB at 32768). glm52_tpctx_sweep.sh's header
# records a previous harness making the same mistake at 16384. The ladder is now
# full:512,2048,4096,8192 and this list is the demand AT THAT LADDER: the M=32768 rung was
# dropped and M=4096 added. Against the merged store that reads 24 HIT / 8 MISS — the 8 misses
# are exactly the new M=4096 rung, which the next campaign run fills.
#
# The 8 (N,K) pairs, so a reader can tell what they are: (6144,512) kv_b up-projection,
# (64,6144) k_rope, (512,6144) kv_a, (256,6144) router, (2048,6144) q_a, (6144,4096) o_proj
# (K = nh_l 16 * v_head 256 at TP4), (8192,2048) q_b, (1024,2048).
#
# So every GLM prefill GEMM above the smallest bucket picked its tile from the ANALYTICAL MODEL,
# while `tuned_tile_selection` kept passing because SOME qualified record existed (a Gemma one).
# That is invisible from the outside: the tier reads "measured".
#
# This matters more than a tuning detail. Prefill is where GLM loses to vLLM by 7.0x at 4k rising
# to 132.6x at 128k (perf-data/glm52-ctx-sweep.md), and prefill is exactly the GEMM-bound phase
# these tiles govern — decode has ZERO Gemm ops.
#
# The GLM block below is the dumped demand, verbatim. DERIVE IT, DO NOT HAND-EDIT IT: re-run
# `PLOW_TUNE_DUMP=1 ... plowc --emit devblob` after any emitter change and re-sync.
# ---------------------------------------------------------------------------------------------
SHAPES="
512 6144 512    glm52-kvb-up-M512
2048 6144 512    glm52-kvb-up-M2048
4096 6144 512    glm52-kvb-up-M4096
8192 6144 512    glm52-kvb-up-M8192
512 64 6144    glm52-krope-M512
2048 64 6144    glm52-krope-M2048
4096 64 6144    glm52-krope-M4096
8192 64 6144    glm52-krope-M8192
512 512 6144    glm52-kva-M512
2048 512 6144    glm52-kva-M2048
4096 512 6144    glm52-kva-M4096
8192 512 6144    glm52-kva-M8192
512 256 6144    glm52-router-M512
2048 256 6144    glm52-router-M2048
4096 256 6144    glm52-router-M4096
8192 256 6144    glm52-router-M8192
512 2048 6144    glm52-qa-M512
2048 2048 6144    glm52-qa-M2048
4096 2048 6144    glm52-qa-M4096
8192 2048 6144    glm52-qa-M8192
512 6144 4096    glm52-oproj-M512
2048 6144 4096    glm52-oproj-M2048
4096 6144 4096    glm52-oproj-M4096
8192 6144 4096    glm52-oproj-M8192
512 8192 2048    glm52-qb-M512
2048 8192 2048    glm52-qb-M2048
4096 8192 2048    glm52-qb-M4096
8192 8192 2048    glm52-qb-M8192
512 1024 2048    glm52-narrow-M512
2048 1024 2048    glm52-narrow-M2048
4096 1024 2048    glm52-narrow-M4096
8192 1024 2048    glm52-narrow-M8192
128 128 2816   gemma26b-router
128 256 6144   glm52-router
128 512 3840   gemma12b-k-global
128 576 6144   glm52-kv-a-proj
128 576 7168   kimi-kv-a-proj
128 2112 2816  gemma26b-dense-gateup
256 8192 5376  gemma31b-q-M256
512 8192 5376  gemma31b-q-M512
1024 8192 5376 gemma31b-q-M1024
2048 8192 5376 gemma31b-q-M2048
2048 21504 5376 gemma31b-gateup-M2048
2048 5376 21504 gemma31b-down-M2048
4096 2048 5376 gemma31b-o-M4096
4096 2560 4096 qwen-o-M4096
4096 5376 8192 gemma31b-down8k
4096 9728 2560 qwen-gateup-M4096
8192 1024 4096 qwen-kv-M8192
8192 8192 5376 gemma31b-q-M8192
"

cd "$OBJ"
[ -f test_kernels.elf ] || { echo "no test_kernels.elf in $OBJ" >&2; exit 1; }

# Host binary, SYSTEM gcc, clean env — same rule as build_gfx950.sh's `chat`.
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 \
    -o gemm_tile_sweep "$WT/runtime/ubench/gemm_tile_sweep.c" "$WT/runtime/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm

rm -f "$JSONL"
export PLOW_GEMM_JSONL="$JSONL"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
echo "$SHAPES" | while read -r M N K LABEL; do
  [ -z "${M:-}" ] && continue
  echo "=== $LABEL  ${M}x${N}x${K}"
  ./gemm_tile_sweep "$M" "$N" "$K" "$LABEL"
done
echo "wrote $(wc -l < "$JSONL") rows -> $JSONL"
