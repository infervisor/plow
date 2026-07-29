#!/usr/bin/env bash
# The DECODE-GEMV row campaign — the half of the network the tuner has never measured.
#
# `tunedb` is GEMM-ONLY (`tunedb::gemm_op_case` is its sole shape lookup), and the DECODE
# program contains ZERO `Gemm` ops: every decode matmul is `Gemv`/`GemvQkv`/`GemvGlu`/
# `GemvFp8Blk`, which `build.json`'s per-program `arms` list shows directly. So the prefill tile
# campaign cannot move ms/token by construction, and until this script existed nothing measured
# the ops that can.
#
# ---------------------------------------------------------------------------------------------
# THE SHAPE LIST IS DERIVED, NOT AUTHORED. Do not hand-edit `SHAPES`.
#
# `scripts/rebench_tune_gemm.sh`'s list WAS hand-authored, and that is exactly how GLM-5.2
# prefill came to be 100% unmeasured for the tuner's entire life — every lookup missed while the
# differential test kept passing, because some qualified record existed for some other model.
# The fix there was `PLOW_TUNE_DUMP`. The GEMV path now has the same instrument
# (`crates/packet/src/devbuild.rs`, hooked at `Builder::emit_dep` — the single choke point all
# thirty-odd emit sites in `devgen` funnel through, so a site added later cannot escape it):
#
#     PLOW_TUNE_DUMP=1 nix develop -c bash scripts/rebench_emit_glm.sh /tmp/x.pkt 2>&1 \
#       | grep TUNEDUMP_GEMV | awk '{print $2, $3, $4}' | sort -u
#
# Re-derive after ANY emitter change and re-sync the block below from that output.
# ---------------------------------------------------------------------------------------------
#
# TWO HALVES, deliberately, same discipline as the GEMM campaign: the C harness writes SAMPLES
# (it cannot know the build identity), and `tunedb-gemv ingest` attaches the probed digests and
# applies the store's gates. A missing ingest leaves the store untouched with every gate green —
# it silently killed a benchmark run at 00:08 on 2026-07-29 — so BOTH halves run here.
#
# Must run OUTSIDE nix (system ROCm) and UNDER a gpulease (it is a timing run). The ingest half
# needs cargo and therefore nix, so it is invoked separately at the end.
#
#   $1  object dir holding the freshly built test_kernels.elf
#   $2  output jsonl
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="${1:?object dir}"
JSONL="${2:?output jsonl}"

# DUMPED 2026-07-29 by `scripts/gemv_census.sh`, verbatim. The M axis is NOT here: the sweep
# runs all five `PLOW_GEMV_MM` rungs itself, because the rung IS the axis being measured. Only
# (N,K) varies per invocation.
#
# GLM-5.2 TP4, ctx 32768, shipping stacked knobs: 467 GEMV emits over SIX distinct shapes, all
# at M=1 — GLM decode is batch-1 because `AmdServe::load` refuses batched+TP. `GemvFp8Blk` does
# NOT appear at this configuration and its absence is a fact about the config, not the census:
# `GLM_LINEAR_FP8` is off (o_proj and the shared-expert projections stay bf16), the routed
# experts are `MoeExpertGluFp8Blk` rather than a GEMV, and `PLOW_GLM_DSA` arms only above
# ctx 65536. Re-derive if any of those change.
#
# Gemma-4-31B, PLOW_DECODE_BATCH=16 PLOW_GEMV_MM=8 PLOW_GEMV_WALK=1: 251 emits over SEVEN
# shapes at M=16. The unfused B=16 control emits 411 for the same network — 160 more packets,
# which is `fuse_qkv` + `glu_fused` turning off and is exactly what §6g-WALK's companion change
# restores.
#
# Every one of these is a MISS today — there has never been a GEMV record of any kind.
#
# NAMED RESIDUAL, not an omission: GLM also asks for `GemvFp8Blk 1 x 6144 x 3072` (W8A8).
# `gemv_row_sweep.c` has bf16 arms only — a block-fp8 arm needs the `[N/128][K/128]` f32 scale
# grid built and checked, which is a different oracle, not a different shape. Until it exists
# that shape reads MISS in the census, which is the correct reading and is why the census
# prints HIT/MISS per shape instead of a single coverage percentage.
SHAPES="
256 6144      glm52-router
512 6144      glm52-shared-glu
2624 6144     glm52-qkv-a
6144 512      glm52-kvb-up
6144 4096     glm52-oproj
9216 2048     glm52-qkv-g
38720 6144    glm52-lmhead-tp4
2048 5376     gemma31b-narrow
5376 8192     gemma31b-down8k
5376 16384    gemma31b-down16k
5376 21504    gemma31b-down
16384 5376    gemma31b-qkv
21504 5376    gemma31b-gateup
262144 5376   gemma31b-lmhead
"

cd "$OBJ"
[ -f test_kernels.elf ] || { echo "no test_kernels.elf in $OBJ" >&2; exit 1; }

# Host binary, SYSTEM gcc, clean env — same rule as build_gfx950.sh's `chat`.
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 \
    -o gemv_row_sweep "$WT/runtime/ubench/gemv_row_sweep.c" "$WT/runtime/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm

rm -f "$JSONL"
export PLOW_GEMV_JSONL="$JSONL"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
echo "$SHAPES" | while read -r N K LABEL; do
  [ -z "${N:-}" ] && continue
  echo "=== $LABEL  N=${N} K=${K}"
  ./gemv_row_sweep "$N" "$K" "$LABEL"
done
echo "wrote $(wc -l < "$JSONL") rows -> $JSONL"
echo
echo "NOW INGEST (needs nix for cargo; the sweep half above must NOT run under nix):"
echo "  nix develop -c cargo run --release -p tunedb --bin tunedb-gemv -- \\"
echo "      ingest --db tuning --samples $JSONL --campaign gemv-row-inventory"
