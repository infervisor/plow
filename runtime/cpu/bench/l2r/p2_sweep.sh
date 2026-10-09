#!/usr/bin/env bash
# p2_sweep.sh <refroot> <outdir> [steps]: l2r_layer over every ref dump in <refroot>, AVX-512 and AMX GEMV, with and
# without the activation all-gather (L2R_NOBCAST), 3 reps each, then the numerical gate over all outputs.
set -u
R=${1:?refroot} O=${2:?outdir} N=${3:-5000}
B=${L2R_LAYER_BIN:-/tmp/g4c/l2r/l2r_layer}
mkdir -p "$O"
for d in "$R"/*/; do
  d=${d%/}; t=$(basename "$d")
  [ -s "$d/meta.json" ] || continue
  for gv in avx amx; do
    for nb in 0 1; do
      for rep in 1 2 3; do
        L2R_GEMV=$gv L2R_NOBCAST=$nb "$B" "$d" "$N" > "$O/$t.$gv.nb$nb.r$rep.json" 2> "$O/$t.$gv.nb$nb.r$rep.err"
      done
    done
  done
done
python3 "$(dirname "$0")/p2_gate.py" "$O"/*.json > "$O/gate.txt"; echo "gate rc=$?"
