#!/usr/bin/env bash
# Build the flash-attention golden harness (attention_gfx950_test.c) against a given
# object dir. The SAME source, twice: once next to the pre-change test_kernels.elf and
# once next to the post-change one, so the single-kernel gate is a like-for-like run.
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for D in "$@"; do
  /usr/bin/gcc -O2 -std=gnu11 -o "$D/t_attn" \
      "$W/runtime/tests/attention_gfx950_test.c" "$W/runtime/amd/hsa_backend.c" \
      -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
  readelf -d "$D/t_attn" | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }
  echo "built $D/t_attn"
done
