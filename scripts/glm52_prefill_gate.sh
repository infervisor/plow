#!/usr/bin/env bash
# glm52_prefill_gate.sh — the GLM-5.2 PREFILL-BUCKET correctness gate.        [GLM52-PF-GATE]
#
# Emits ONE block with a whole-layer prefill bucket, builds the T-row HF oracle fixture for the
# same layer, and diffs every stage of the emitted bucket program against it. Run it for a SPARSE
# layer (>= first_k_dense_replace, the grouped MoE arms) AND a DENSE one (< it, the same arms under
# degenerate 1-expert routing — the path that had no oracle at all):
#
#   ./scripts/glm52_prefill_gate.sh 3      # MoE
#   ./scripts/glm52_prefill_gate.sh 0      # dense
#
# The verdict is the SAME per-stage residual table the decode gate prints (glm52_run.c, GLM6
# fixture) at the SAME tolerances, so the two are directly comparable. Prefill-vs-decode TOKEN
# identity is explicitly NOT the bar: the phases run different kernels with different bf16
# accumulation orders, and no production engine meets it.
#
# ENVIRONMENT — the two halves build in DIFFERENT shells and mixing them wastes hours:
#   * plowc is a nix binary        -> `nix develop -c`
#   * the host harness + hipcc     -> SYSTEM /usr/bin/gcc in a clean env (nix shadows glibc and
#                                     every ROCm binary then dies with GLIBC_2.38 not found)
# `--max-ctx 4096` is load-bearing: the default 131072 arms the DSA indexer (`GlmCfg::dsa` gates it
# at ctx > 65536), which is unvalidated past ctx 2048 and produces garbage on the decode path too.
#
# The oracle needs `q_b_proj`/`kv_b_proj`, which weight-prep ABSORBS away (a product is not
# invertible). GLM_EXTRA_DIRS points at a small dir holding just those, per layer, from the
# original fp8 checkpoint; see glm52_real_oracle.py's header.
set -euo pipefail

LAYER="${1:-3}"
T="${PLOW_PF_T:-128}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${GLM_CKPT:-/home/lava/models/GLM-5.2-plow}"
EXTRA="${GLM_EXTRA_DIRS:-/home/lava/models/glm52_hf_extra}"
OBJS="${PLOW_HSACO:-/home/lava/models/glm52_objs_pf}"
PY="${GLM_PYTHON:-/home/lava/models/oracle_venv/bin/python}"
OUT="${PLOW_PF_OUT:-/home/lava/models/glm52_pf}"
mkdir -p "$OUT"

cd "$REPO"   # `nix develop` resolves the flake from the CWD
echo "== emit block $LAYER with a T=$T prefill bucket =="
# --no-rope-gen: the C harness has no RoPE-table generator, so the tables must be BAKED into the
# init section (a v7 blob would load with cos=sin=0 and no error).
PLOW_MLA_PREFILL="full:$T" nix develop -c "$REPO/target/release/plowc" \
    --hf-dir "$CKPT" --emit devblob --out "$OUT/emit$LAYER" --gpu mi350 \
    --num-gpus 1 --parallel tp --max-ctx 4096 --block "$LAYER" --no-rope-gen
cp "$OUT/emit$LAYER/model.pkt" "$OUT/blk$LAYER.pkt"

echo "== host harness (system gcc, clean env) =="
env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o "$OUT/glm52_run" \
    "$REPO/runtime/tests/glm52_run.c" "$REPO/runtime/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d "$OUT/glm52_run" | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }
cp "$OBJS/interp_prefill_mla_moe.elf" "$OUT/"

echo "== T=$T oracle fixture for layer $LAYER =="
GLM_MODEL_DIR="$CKPT" GLM_EXTRA_DIRS="$EXTRA" GLM_T="$T" GLM_LAYER="$LAYER" \
    "$PY" "$REPO/runtime/tests/glm52_real_oracle.py" "$OUT/pf_l${LAYER}_T${T}.bin"

echo "== run (ONE gpu; ~10 GB for a MoE layer's 256 fp8 experts) =="
cat >"$OUT/.run.sh" <<EOF
cd "$OUT"
exec env -i PATH=/usr/bin:/bin HOME="\$HOME" LD_LIBRARY_PATH=/opt/rocm/lib \\
  ROCR_VISIBLE_DEVICES="\${ROCR_VISIBLE_DEVICES:-0}" HIP_VISIBLE_DEVICES="\${HIP_VISIBLE_DEVICES:-0}" \\
  PLOW_INTERP=interp_prefill_mla_moe.elf \\
  ./glm52_run blk$LAYER.pkt "$CKPT" pf_l${LAYER}_T${T}.bin $LAYER
EOF
exec "$REPO/perf-data/tools/gpulease" -n 1 "glm52-pf-l$LAYER" \
     sg render -c "bash $OUT/.run.sh"
