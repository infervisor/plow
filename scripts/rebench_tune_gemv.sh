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
#
# ---------------------------------------------------------------------------------------------
# RE-RUN, VERBATIM. The K3 MXFP4 campaign was published against build `gfx950-3d15138c0b7b8e4e`
# on rocm-smi device 5 (KFD node 9 = ROCR index 7 on this box — see the pinning note below).
# The digest is over the PREPROCESSED translation unit, so it covers every `#include`: ANY edit
# to `runtime/amd/op_gemm.h` (or anything it pulls in) re-stales every record here, and the
# store will report them stale rather than serve them. When that happens, from the repo root:
#
#   # 1. rebuild the golden object (test_kernels.hip takes NO -DPLOW_GEMV_MM, so it compiles at
#   #    op_gemm.h's default of 1 — which is the decode bucket the mxfp4 arms need)
#   mkdir -p obj && cd obj
#   hipcc --offload-arch=gfx950 -O3 -w --genco ../runtime/amd/test_kernels.hip -o tk.co \
#         -I../runtime/amd -I../runtime/common
#   /opt/rocm/lib/llvm/bin/clang-offload-bundler --unbundle --type=o \
#         --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --input=tk.co --output=test_kernels.elf
#   cd ..
#
#   # 2. sweep (OUTSIDE nix, under the render group, pinned to ONE idle card)
#   flock /tmp/gpulease/gpu.5.lease flock /tmp/gpulease/gpu.7.lease \
#     sg render -c 'ROCR_VISIBLE_DEVICES=7 PLOW_GEMV_ONLY=k3- \
#                   bash scripts/rebench_tune_gemv.sh "$PWD/obj" /tmp/k3_gemv.jsonl'
#
#   # 3. ingest (needs nix for cargo; must NOT run under the lease)
#   nix develop -c cargo run --release -p tunedb --bin tunedb-gemv -- \
#       ingest --db tuning --samples /tmp/k3_gemv.jsonl --campaign k3-mxfp4-decode-gemv
#
# Drop `PLOW_GEMV_ONLY` to sweep the GLM/Gemma rows too. `scripts/gemv_campaign_lease.sh` wraps
# steps 2+3 under `gpulease -n 1`, but that helper takes the LOWEST free card, not a chosen one.
# ---------------------------------------------------------------------------------------------
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

# KIMI-K3, TP8, DECODE (M=1). The MXFP4 campaign's shapes.
#
# K3 is a NATIVE MXFP4 checkpoint: its whole decode path runs `GemvMxfp4`/`GemvGluMxfp4`, and
# until `tunedb::gemv::SYMBOLS` gained the quant axis those op cases could not even be spelled,
# so nothing had ever been measured on them. The sweep now runs a w4a16 arm at every shape below
# beside the bf16 one, from the SAME object and the same hW, so the pair differs in the ENCODING
# and nothing else.
#
# GEOMETRY: hidden 7168, q_lora 1536, kv_lora 512, qk_rope 64, v_head 128, 96 heads,
# KDA proj 96*128 = 12288, latent 3584, moe_inter 3072, dense inter 18432, vocab 163840. Under
# TP8 the HEAD axis and the expert intermediate shard by 8 (nh_l = 12, local KDA proj 1536, local
# moe_inter 384); the LATENT projections (q_lora, kv_lora, the 3584 routed-expert latent) do NOT
# shard.
#
# Rows, in the order below: KDA q/k/v/g one stream (69 layers); the same unsharded, for the
# sharding delta; f_a; f_b (K is the SHORT axis, the one shape here that is not HBM-bound);
# b_proj; o_proj. Then MLA (24 layers): q_a, kv_a, kv_b off the 512 latent. Then MoE (92
# layers): router, latent down, latent up, shared-expert glu and its down. Then the lm_head
# tail, the widest single weight in the network.
#
# NOT DERIVED FROM `PLOW_TUNE_DUMP` — a stated exception, not a lapse. The census instrument
# dumps what an emitter RESOLVES, and a K3 decode packet cannot be emitted on this branch yet.
# These come from the checkpoint's own config, the only source that exists today. Re-derive from
# the dump the moment K3 decode emits, and expect the four KDA projection rows to collapse into
# ONE `gemvqkvg` case when it does (`DevOp::GemvQkvg`, four output streams in one packet).
SHAPES="$SHAPES
1536 7168     k3-kda-qkvg-one
12288 7168    k3-kda-proj-unsharded
128 7168      k3-kda-f-a
1536 128      k3-kda-f-b
96 7168       k3-kda-b-proj
7168 1536     k3-kda-o-proj
1536 512      k3-mla-kv-b
512 7168      k3-mla-kv-a
896 7168      k3-moe-router
3584 7168     k3-moe-latent-down
7168 3584     k3-moe-latent-up
384 7168      k3-moe-shared-glu
7168 384      k3-moe-shared-down
163840 7168   k3-lmhead
"

# Run a SUBSET by label: `PLOW_GEMV_ONLY=k3-` sweeps only the K3 rows. The full list is ~28
# shapes and the two widest (`k3-lmhead`, `gemma31b-lmhead`) each build a 1.17e9-element mxfp4
# fixture on the host before the first launch, so an unfiltered run is a long one. The filter is
# a plain `grep -E` over the LABEL, and an empty setting keeps every row.
ONLY="${PLOW_GEMV_ONLY:-}"

cd "$OBJ"
[ -f test_kernels.elf ] || { echo "no test_kernels.elf in $OBJ" >&2; exit 1; }

# Host binary, SYSTEM gcc, clean env — same rule as build_gfx950.sh's `chat`.
/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 \
    -o gemv_row_sweep "$WT/runtime/ubench/gemv_row_sweep.c" "$WT/runtime/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm

rm -f "$JSONL"
export PLOW_GEMV_JSONL="$JSONL"

# DEVICE PINNING IS `ROCR_VISIBLE_DEVICES`, AND ITS INDEX IS NOT rocm-smi's.
#
# This harness is a bare ROCr/HSA binary — it never links HIP — so `HIP_VISIBLE_DEVICES` does
# NOTHING here and unsetting it (as this line did, alone) leaves the sweep on whatever ROCr calls
# agent 0. On a shared 8-GPU box with other agents working, that is how a timing run lands on a
# contended card and reports numbers nobody can use.
#
# ROCr enumerates GPU agents in KFD NODE order, while rocm-smi has its own device order, and on
# this box the two disagree: `ROCR_VISIBLE_DEVICES=5` drives the card rocm-smi calls device 4
# (node 7), and rocm-smi's device 5 (node 9) is ROCr index 7. VERIFY, do not assume — run the
# sweep and watch which row in `rocm-smi` goes to 100%. Leaving ROCR_VISIBLE_DEVICES unset here
# is deliberate: the caller (`scripts/gemv_campaign_lease.sh` via `gpulease -n 1`) exports it to
# the leased card, and overriding it here would break the lease's guarantee.
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
echo "$SHAPES" | while read -r N K LABEL; do
  [ -z "${N:-}" ] && continue
  [ -n "$ONLY" ] && { echo "$LABEL" | grep -Eq "$ONLY" || continue; }
  echo "=== $LABEL  N=${N} K=${K}"
  ./gemv_row_sweep "$N" "$K" "$LABEL"
done
echo "wrote $(wc -l < "$JSONL") rows -> $JSONL"
echo
echo "NOW INGEST (needs nix for cargo; the sweep half above must NOT run under nix):"
echo "  nix develop -c cargo run --release -p tunedb --bin tunedb-gemv -- \\"
echo "      ingest --db tuning --samples $JSONL --campaign gemv-row-inventory"
