#!/usr/bin/env bash
# Build the multi-GPU TP decode orchestration harness (P1-C + P1-D).
#
# Produces, into the output dir (default /tmp/tpd, or $1):
#   tp_decode            the N-device host harness (links hsa_backend.c)
#
# It reuses the DEVICE code objects that scripts/build_gfx950.sh already builds
# (interp_decode.elf — the static decode interpreter) and, for the cross-GPU
# handshake probe, scripts/build_tp_p2p.sh's tp_p2p_kernels.elf. This script
# only builds the HOST binary; copy those .elf files next to tp_decode (or run
# it from the dir that has them).
#
# Same toolchain contract as build_tp_p2p.sh: the SYSTEM gcc in a CLEAN env for
# the host (a nix RUNPATH into glibc 2.42 vs the system 2.35 ELF interpreter
# aborts with a bogus "stack smashing").
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$REPO/runtime"
OUT="${1:-${PLOW_BUILD_DIR:-/tmp/tpd}}"
mkdir -p "$OUT"; cd "$OUT"

rm -f tp_decode

/usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o tp_decode \
    "$R/tests/tp_decode.c" "$R/amd/hsa_backend.c" \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
readelf -d tp_decode | grep -qi runpath && { echo "FAIL: RUNPATH leaked"; exit 1; }

ls -l --time-style=+%H:%M:%S tp_decode | awk '{print "   ", $NF, $5"B", $6}'
echo "run (needs interp_decode.elf [+ tp_p2p_kernels.elf] in cwd):"
echo "  sg render -c 'cd $OUT && /usr/bin/env -i PATH=/usr/bin:/bin HOME=\$HOME \\"
echo "    LD_LIBRARY_PATH=/opt/rocm/lib ./tp_decode model.pkt <model-dir> --tp 2 --verify prompt.ids 8'"
