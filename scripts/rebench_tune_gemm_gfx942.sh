#!/usr/bin/env bash
# RE-RUN THE gfx942 PREFILL-GEMM TILE CAMPAIGN AND PUBLISH IT.
#
# The gfx942 twin of `rebench_tune_gemm_all.sh`, which is hardcoded to gfx950/mi350 in four
# places. Its absence is why the gfx942 cell went stale and stayed stale:
#
#   1. `build_gfx942.sh` produced no `test_kernels.elf` (fixed 2026-08-09), so
#   2. `plowc tune gemm --obj` had nothing to time, so
#   3. no runner could exist for this arch, so the cell -- seeded once by hand -- could never be
#      REFRESHED and went stale on the first `runtime/amd/` edit after seeding, while
#   4. `tuned_tile_selection.rs` named gfx950 eleven times and gfx942 zero, so nothing turned red.
#
# Net effect for days: every gfx942 compile chose GEMM tiles from the ANALYTICAL MODEL and
# reported tier `portable`, which is byte-identical to what it reports when nothing was ever
# measured. `plowc tune status --gpu MI300X` is the check; THIS SCRIPT IS THE FIX.
#
#     scripts/rebench_tune_gemm_gfx942.sh
#
# WHEN TO RUN IT: after ANY edit reachable from `interp.hip` (`runtime/amd/op_*.h`,
# `runtime/common/dev_isa.h`, `interp.hip` itself) AND after any change to the object recipe's
# DEFINES -- both participate in the digest that keys the store (`kernelcaps::BuildId::label`).
# Flipping a default in `build_gfx942.sh` re-stales the store exactly as a source edit does.
#
# ------------------------------------------------------------------------------------------
# THE ENVIRONMENT RULES, all of them learned the expensive way on this box:
#
#   * `build_gfx942.sh` runs OUTSIDE nix. hipcc is the system ROCm one and nix's
#     CPATH/LIBRARY_PATH shadow the glibc it was built against. The digest the store keys on
#     comes from THESE sources, so the campaign must build the objects it then measures.
#   * `PLOW_HIPCC=/opt/rocm-7.2.4/bin/hipcc` and the matching ROCM_PATH/HIP_PATH:
#     `/opt/rocm/bin/hipcc` on this box is broken (its internal clang++ is missing).
#   * plowc runs INSIDE nix (it needs the cargo toolchain), and needs `/opt/rocm-*/lib` on
#     LD_LIBRARY_PATH from INSIDE that shell -- the flake does not carry it, and without it the
#     HSA dlopen fails.
#   * `ROCR_VISIBLE_DEVICES`, NOT `HIP_VISIBLE_DEVICES`. They COMPOSE, so setting both makes a
#     correctly targeted card report "no ROCm-capable device is detected".
#   * `ROCM_PATH` must be exported for PLOWC ITSELF, not just for hipcc. `plowc tune gemm` builds
#     the `gemm_tile_sweep` host harness with /usr/bin/gcc under `.env_clear()` and computes its
#     `-I<rocm>/include` from the PARENT's ROCM_PATH, defaulting to `/opt/rocm`
#     (crates/plowc/src/tune/gemm.rs:424). On this box `/opt/rocm/include/hsa/hsa.h` DOES NOT
#     EXIST -- only /opt/rocm-7.2.4 has it -- so without the export the campaign dies at step
#     4/5 with `fatal error: hsa/hsa.h`, AFTER the whole object build has been paid for.
#     The RUN of that harness had the same bug one layer down: its `LD_LIBRARY_PATH` was the
#     literal "/opt/rocm/lib", which does not exist here either, so a harness that compiled would
#     have died on libhsa-runtime64.so.1 at exec. Both are ROCM_PATH-relative now (`rocm_root`).
#   * Take the GPU lock. This measures kernel timings; a sibling plowrt makes them fiction.
set -euo pipefail

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="${PLOW_TUNE_OBJ:-/tmp/tunecamp942-objs}"
OUT="${PLOW_TUNE_OUT:-/tmp/tunecamp942}"
ROCR="${ROCR_VISIBLE_DEVICES:-0}"
HF="${PLOW_HF_DIR:-/workspace/models/GLM-5.2-FP8}"
MAXCTX="${PLOW_MAX_CTX:-73728}"
NCU="${PLOW_N_CU:-304}"
NGPU="${PLOW_NUM_GPUS:-8}"
ROCM="${PLOW_ROCM:-/opt/rocm-7.2.4}"
NIX="${PLOW_NIX:-/nix/var/nix/profiles/default/bin/nix}"
LOCK=/tmp/plow_gpu.lock
HAVE_LOCK=0

release() { [ "$HAVE_LOCK" = 1 ] && rm -rf "$LOCK"; return 0; }
trap 'release; exit 143' INT TERM
trap 'release' EXIT

mkdir -p "$OUT"
cd "$WT"

echo "=== 0/5  GPU lock"
for i in $(seq 1 720); do mkdir "$LOCK" 2>/dev/null && { HAVE_LOCK=1; break; }; sleep 5; done
[ "$HAVE_LOCK" = 1 ] || { echo "FAIL: could not take the GPU lock"; exit 1; }
echo "$$ tune-gfx942" > "$LOCK/owner" 2>/dev/null
if pgrep '^plowrt' >/dev/null 2>&1; then
  echo "FAIL: a plowrt is already running — kernel timings would be fiction:"; pgrep -a '^plowrt'; exit 1
fi

echo "=== 1/5  building the gfx942 objects + test_kernels.elf into $OBJ (outside nix)"
# The SHIPPING recipe, not a special one: the defines participate in the store's digest, so a
# campaign measured against a different -D set is stale the moment it lands.
PLOW_HIPCC="$ROCM/bin/hipcc" HIP_PATH="$ROCM" ROCM_PATH="$ROCM" ROCM_HOME="$ROCM" \
  PLOW_OCC4=1 PLOW_L2HIER=1 JOBS="${JOBS:-14}" ./scripts/build_gfx942.sh "$OBJ"
[ -f "$OBJ/test_kernels.elf" ] || {
  echo "FAIL: $OBJ/test_kernels.elf missing — build_gfx942.sh must build it (see its own note)"; exit 1; }

echo
echo "=== 2/5  building plowc"
"$NIX" develop "$WT" -c cargo build --release -p plowc

echo
echo "=== 3/5  digest BEFORE (expect: every record stale, if this is a refresh)"
"$NIX" develop "$WT" -c bash -c \
  "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:$ROCM/lib\"; \
   ./target/release/plowc tune status --gpu MI300X --root . || true"

echo
echo "=== 4/5  measuring and publishing (ROCR_VISIBLE_DEVICES=$ROCR)"
# `--shapes auto` DERIVES the list from the compiler's own demand rather than a hand-written
# file. That matters here specifically: lib.rs records that the hand list is how "GLM-5.2 came
# to have exactly two measured shapes ... while every M>=256 record in the store was a
# Gemma-31B or Qwen shape with K in {2560,4096,5376,8192,21504} — never GLM's K=6144."
#
# The SHIPPING emit env is exported below, and two of those vars are load-bearing:
#   * without GLM_FULL=1 the --num-gpus 8 demand emit PANICS — `mla.rs` asserts tp==1 on the
#     single-layer bring-up path ("GLM TP sharding is milestone-3; use --tp 1");
#   * with GLM_FULL=1 but no PLOW_MLA_PREFILL=full the emit declares NO prefill buckets, so
#     `pick_tile` is never reached and the command aborts with "the emit asked the tuning store
#     about no dense GEMM at all" — the demand recorder correctly refusing an empty campaign.
# With both set the demand is 48 distinct shapes. The remaining three are the recipe's shipping
# knobs; they do not change the shape set (checked), but the campaign should observe the compile
# that ships rather than a neighbour of it.
"$NIX" develop "$WT" -c bash -c \
  "set -euo pipefail; cd '$WT'; \
   export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:$ROCM/lib\"; \
   export ROCM_PATH=$ROCM HIP_PATH=$ROCM; \
   export ROCR_VISIBLE_DEVICES=$ROCR; unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; \
   export GLM_FULL=1 PLOW_MLA_PREFILL=full GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1; \
   ./target/release/plowc --hf-dir '$HF' --max-ctx $MAXCTX --n-cu $NCU --num-gpus $NGPU \
       --gpu MI300X --arch gfx942 \
       tune gemm --gpu MI300X --root . --obj '$OBJ' \
       --samples '$OUT/bf16.jsonl' --campaign gfx942-mi300x-gemm-tile"

echo
echo "=== 5/5  verify — the store must now REACH the compiler, not merely contain records"
"$NIX" develop "$WT" -c bash -c \
  "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:$ROCM/lib\"; \
   ./target/release/plowc tune status --gpu MI300X --root ."
# The guard is the point: a campaign that measures and publishes nothing reports success at
# every boundary it owns, which is how this failed before.
"$NIX" develop "$WT" -c cargo test --release -p devgen --test tuned_tile_selection \
    gfx942_measurements_reach_the_compiler
echo
echo "DONE. Objects: $OBJ   samples: $OUT/bf16.jsonl"
