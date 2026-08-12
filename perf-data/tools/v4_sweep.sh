#!/usr/bin/env bash
# v4_sweep.sh — fan a V4 decode-step A/B across every GPU on the box.
#
# One variant per card, all at once, each under its OWN `gpulease -n 1` so the
# arms cannot contend. Contended timings are worthless and the whole point of
# leasing per card is that a 4-way sweep on an 8-GPU box is 4 clean rooms, not
# one noisy one.
#
# Builds are CPU-bound and run concurrently first; only the timed runs take a
# lease. A variant that fails to build is reported and skipped rather than
# silently missing from the table.
#
#   perf-data/tools/v4_sweep.sh 'label:-Dflag=x' 'label2:-Dflag=y' ...
#   perf-data/tools/v4_sweep.sh            # the default KT sweep
#
# Env: OUT (work dir), SRC (test to build).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && cd .. && pwd)"
OUT="${OUT:-${TMPDIR:-/tmp}/v4sweep-$USER}"
SRC="${SRC:-$REPO/runtime/tests/v4_decode_step_gfx942.hip}"
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$REPO"

if [ "$#" -gt 0 ]; then
  VARIANTS=("$@")
else
  VARIANTS=(
    "kt16:-DPLOW_V4_ATTN_KT=16u"
    "kt32:-DPLOW_V4_ATTN_KT=32u"
    "kt64:-DPLOW_V4_ATTN_KT=64u"
    "base:"
  )
fi

echo "building ${#VARIANTS[@]} variant(s)..."
for v in "${VARIANTS[@]}"; do
  lab="${v%%:*}"; flags="${v#*:}"
  (
    # shellcheck disable=SC2086
    if hipcc --offload-arch=gfx942 -O3 -w $flags \
         -I runtime/amd -I runtime/common "$SRC" -o "$OUT/$lab" >"$OUT/$lab.build" 2>&1; then
      echo "  built $lab"
    else
      echo "  BUILD FAIL $lab"; tail -6 "$OUT/$lab.build" | sed 's/^/    /'
    fi
  ) &
done
wait

# EACH ARM IS PINNED TO ITS OWN CARD EXPLICITLY. `gpulease -n 1` picks a free
# card and exports the visible-device vars itself, but several of those running
# CONCURRENTLY handed only the first arm a usable device here and the rest died
# with "no ROCm-capable device is detected". Assigning the index ourselves and
# still taking a lease per card keeps both properties: distinct cards, and the
# advisory lock that stops another agent landing on one mid-measurement.
NDEV="${GPU_LEASE_NGPU:-8}"
echo "running, one arm per card (0..$((NDEV-1)))..."
i=0
for v in "${VARIANTS[@]}"; do
  lab="${v%%:*}"
  [ -x "$OUT/$lab" ] || continue
  dev=$((i % NDEV)); i=$((i+1))
  (
    # EXACTLY ONE of these, never both: ROCR filters the agent list first and
    # HIP indexes into what survives, so setting both to `dev` makes ROCR narrow
    # to a single agent and HIP then ask for index `dev` of a 1-element list —
    # "no ROCm-capable device is detected" for every arm but card 0.
    export HIP_VISIBLE_DEVICES="$dev"
    GPU_LEASE_NGPU="$NDEV" perf-data/tools/gpulease "v4-$lab-gpu$dev" \
      "$OUT/$lab" >"$OUT/$lab.out" 2>&1
  ) &
done
wait

printf '\n%-8s %14s %14s %12s\n' variant as-measured floor-removed attention
for v in "${VARIANTS[@]}"; do
  lab="${v%%:*}"
  f="$OUT/$lab.out"
  [ -f "$f" ] || { printf '%-8s %14s\n' "$lab" "(no run)"; continue; }
  am=$(grep -oE 'as measured *[0-9.]+ ms' "$f" | grep -oE '[0-9.]+' | head -1)
  fr=$(grep -oE 'floor removed *[0-9.]+ ms' "$f" | grep -oE '[0-9.]+' | head -1)
  at=$(awk '/sparse attention/ {print $4; exit}' "$f")   # $3 is the packet COUNT
  printf '%-8s %11s ms %11s ms %9s us\n' "$lab" "${am:--}" "${fr:--}" "${at:--}"
done
echo
echo "logs: $OUT"
