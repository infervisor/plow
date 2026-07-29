#!/usr/bin/env bash
# glm52_stall_ab.sh — the whole gate-stall attribution + A/B campaign, one file.
# Companion to perf-data/glm52-gate-stall-attribution.md.
#
#   ./glm52_stall_ab.sh objects            # hipcc + gcc, OUTSIDE nix
#   nix develop -c ./glm52_stall_ab.sh blobs
#   gpulease -n 4 glm-stall sg render -c '<this> trace'   # attribution, all 4 ranks traced
#   gpulease -n 4 glm-stall sg render -c '<this> ab'      # timing A/B, controls interleaved
#   ./glm52_stall_ab.sh report
#
# knob-contract §0a: ROCm tooling must run OUTSIDE `nix develop` (the nix glibc shadows the
# system one and every ROCm binary dies with GLIBC_2.38); cargo/plowc must run INSIDE it.
set -u
REPO="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${PLOW_STALL_DIR:-/home/lava/models/glm52_skew}"
W="${PLOW_GLM_WEIGHTS:-/home/lava/models/GLM-5.2-plow}"
R="$REPO/runtime"
mkdir -p "$OUT/tr"

case "${1:-}" in
objects)
  BUN="$(ls -1 /opt/rocm/lib/llvm/bin/clang-offload-bundler /opt/rocm/llvm/bin/clang-offload-bundler 2>/dev/null | head -1)"
  cd "$OUT"
  for arm in "base:" "nowait:-DPLOW_XR_NOWAIT=1"; do
    tag=${arm%%:*}; def=${arm#*:}
    hipcc --offload-arch=gfx950 -O3 -w -DPLOW_BUCKET_DECODE=1 $def --genco "$R/amd/interp.hip" \
        -o "i_$tag.co" -I"$R/amd" -I"$R/common"
    "$BUN" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
        --input="i_$tag.co" --output="i_$tag.elf"
  done
  # host harness: SYSTEM gcc, clean env — a nix-gcc RUNPATH aborts at HSA load
  /usr/bin/gcc -O2 -std=gnu11 -o glm52_decode "$R/tests/glm52_decode.c" "$R/amd/hsa_backend.c" \
      -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
  readelf -d glm52_decode | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }
  ls -l --time-style=+%H:%M:%S i_base.elf i_nowait.elf glm52_decode | awk '{print "   ",$NF,$5"B",$6}'
  ;;
blobs)
  cd "$REPO"
  cargo build --release -p plowc --bin plowc
  emit () { local tag="$1"; shift
    unset PLOW_XR_CUS GLM_SPINE_CUS PLOW_GLM_FUSE_B1 || true
    for kv in "$@"; do export "$kv"; done
    GLM_FULL=1 PLOW_FP8=1 "$REPO/target/release/plowc" --hf-dir "$W" --emit devblob \
        --max-ctx 65536 --n-cu 256 --num-gpus 4 --no-rope-gen --out "$OUT/$tag.pkt" 2>/dev/null
    ls -l --time-style=+%H:%M:%S "$OUT/$tag.pkt" | awk '{print "   ",$NF,$5"B",$6}'; }
  emit xrfit                                 # the shipping default (collective sized to its work)
  emit xr32     PLOW_XR_CUS=32               # the dense emitter's old flat default
  emit res32    GLM_SPINE_CUS=32             # ceiling instrument: Residual 1 -> 32 workgroups
  emit xrfitb1  PLOW_GLM_FUSE_B1=1           # (Residual, RmsNorm) -> AddNorm. NOT token-identical.
  # The CONTROL is the pre-fix blob and cannot be re-emitted from a fixed tree — keep the
  # published one, whose md5 this campaign verified: aba55146073f19c238ea349a263ae87d.
  echo "control: cp /home/lava/models/glm52_tp/glm52_tp4_64k.pkt $OUT/xr_base.pkt"
  ;;
trace)
  cd "$OUT"; unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  # ALL FOUR RANKS. A rank-0-only trace makes rank 0 the systematic straggler (it carries the
  # trace store) and drives its measured cross-rank peer-wait to ~0 BY CONSTRUCTION.
  PLOW_INTERP=i_base.elf PLOW_TRACE_RAW=tr/base1 PLOW_TRACE_ALLRANKS=1 \
    ./glm52_decode xr_base.pkt "$W" --tp 4 --sweep 1024 --steps 65 --gen 24
  PLOW_INTERP=i_nowait.elf PLOW_TRACE_RAW=tr/nowait \
    ./glm52_decode xr_base.pkt "$W" --tp 4 --sweep 1024 --steps 65      # ceiling: no rendezvous
  PLOW_INTERP=i_base.elf PLOW_TRACE_RAW=tr/base2 \
    ./glm52_decode xr_base.pkt "$W" --tp 4 --sweep 1024 --steps 65      # nowait's own control
  ;;
ab)
  cd "$OUT"; unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  for a in xr_base xrfit xr32 res32 xr_base xrfitb1 xr_base; do
    echo "########## $a"
    PLOW_INTERP=i_base.elf ./glm52_decode "$a.pkt" "$W" --tp 4 --sweep 1024 --steps 65 --gen 24
    echo "########## end $a rc=$?"
  done
  ;;
report)
  cd "$OUT"
  MS=$(grep -A2 'raw trace -> tr/base1.rk0' -m1 . 2>/dev/null; true)
  python3 "$REPO/scripts/glm52_stall_attrib.py" tr/base1.insts.txt \
      tr/base1.rk0.tp4.ctx1024.bin tr/base1.rk1.tp4.ctx1024.bin \
      tr/base1.rk2.tp4.ctx1024.bin tr/base1.rk3.tp4.ctx1024.bin --traced-ms "${2:-28.764}"
  python3 "$REPO/scripts/glm52_token_attrib.py" tr/base1.insts.txt \
      tr/base1.rk0.tp4.ctx1024.bin --tp 4 --traced-ms "${2:-28.764}"
  ;;
*) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
