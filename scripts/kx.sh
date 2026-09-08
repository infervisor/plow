#!/usr/bin/env bash
# kx — run one single-block kernel experiment. See runtime/bench/amd/kx/README.md.
#
#   scripts/kx.sh <experiment> [--grid=1|model|both] [--it=N] [--shape=NAME] [--isa] [--rebuild]
#
# ONE COMMAND, SECONDS. Everything below exists so that a result is either trustworthy or refused.
#
# THE ENVIRONMENT RULES, all of them learned expensively on this box:
#   * The DEVICE objects are built with the SAME toolchain the shipped objects use -- the flake's
#     ROCm 7.14 hipcc, INSIDE `nix develop`, exactly as scripts/build_gfx942.sh does. The system
#     /opt/rocm/bin/hipcc on this box is a different compiler and its objects are not comparable.
#   * The DEFINES come from the experiment itself (`--plan`), copied from the shipped recipe. They
#     gate which arms of op_gemm.h / op_attention.h instantiate and they participate in the digest
#     the tuning store keys on; a bench built without them measures a different kernel.
#   * ROCR_VISIBLE_DEVICES, NOT HIP_VISIBLE_DEVICES. They COMPOSE, and setting both makes a
#     correctly targeted card report "no ROCm-capable device is detected". gpulease -n 1 sets
#     ROCR and unsets the other two; the driver REFUSES to run if that is not the state it sees.
#   * A GPU LEASE, always. This measures kernel timings to single-digit microseconds; a sibling
#     plowrt makes them fiction. An unpinned run on a contended box has read A/A 0.27..4.57 here.
#   * STALE BINARIES FAIL LOUDLY. The cache key is a hash of the source, EVERY header the compile
#     actually pulled in (from hipcc's own -MD depfile), the defines, the arch and the toolchain
#     label. A whole result on this branch was a FALSE NULL because a candidate was measured
#     against a stale binary; a `-nt` mtime check would not have caught it.
set -euo pipefail

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KXD="$WT/runtime/bench/amd/kx"
NIX="${PLOW_NIX:-/nix/var/nix/profiles/default/bin/nix}"
CACHE="${KX_CACHE:-/tmp/kx-cache}"

EXP="${1:-}"
if [ -z "$EXP" ] || [ "$EXP" = "-h" ] || [ "$EXP" = "--help" ]; then
  echo "usage: scripts/kx.sh <experiment> [--grid=1|model|both] [--it=N] [--shape=NAME] [--isa] [--rebuild]"
  echo "experiments:"
  for f in "$KXD"/exp_*.hip; do echo "  $(basename "$f" .hip | sed 's/^exp_//')"; done
  exit 0
fi
shift
SRC="$KXD/exp_${EXP}.hip"
[ -f "$SRC" ] || { echo "kx: no experiment '$EXP' ($SRC)" >&2; exit 2; }

ISA=0; REBUILD=0; RUNARGS=()
for a in "$@"; do
  case "$a" in
    --isa) ISA=1 ;;
    --rebuild) REBUILD=1 ;;
    *) RUNARGS+=("$a") ;;
  esac
done

OUT="$CACHE/$EXP"
mkdir -p "$OUT"

# ---------------------------------------------------------------- 1. the host driver + the plan
# The descriptor is compiled from the SAME .hip as the arms (-DKX_HOST), so a variant list and the
# arms it names cannot drift apart.
DRV="$OUT/drv"
if [ "$REBUILD" = 1 ] || [ ! -x "$DRV" ] || [ "$SRC" -nt "$DRV" ] || [ "$KXD/kx_main.cpp" -nt "$DRV" ] \
   || [ "$KXD/kx.h" -nt "$DRV" ]; then
  g++ -O2 -std=c++17 -DKX_HOST=1 -x c++ "$SRC" "$KXD/kx_main.cpp" -o "$DRV" \
      -I"$KXD" -I"$WT/runtime/amd" -I"$WT/runtime/common" -I/opt/rocm-7.2.4/include \
      -D__HIP_PLATFORM_AMD__ -L/opt/rocm-7.2.4/lib -lamdhip64
fi
PLAN="$("$DRV" --plan)"
ARCH="$(awk -F'\t' '$1=="arch"{print $2}' <<<"$PLAN")"
BASEDEF="$(awk -F'\t' '$1=="defines"{print $2}' <<<"$PLAN")"
mapfile -t VARLINES < <(awk -F'\t' '$1=="variant"{print $2"\t"$3}' <<<"$PLAN")

# ---------------------------------------------------------------- 2. the device objects
cat > "$OUT/build.sh" <<'BUILDEOF'
#!/usr/bin/env bash
# Runs INSIDE nix develop. $1 src  $2 out.co  $3 arch  $4.. defines
set -euo pipefail
: "${PLOW_HIPCC:?nix develop did not set PLOW_HIPCC}"
[ "${PLOW_TOOLCHAIN_LABEL:-}" = "rocm-7.14.0-nix" ] || {
  echo "kx: FAIL — expected the flake's rocm-7.14.0-nix toolchain, got '${PLOW_TOOLCHAIN_LABEL:-unset}'." >&2
  echo "  The shipped gfx942 objects are built with it; another hipcc is not comparable." >&2
  exit 2; }
SRC="$1"; CO="$2"; ARCH="$3"; shift 3
exec "$PLOW_HIPCC" --offload-arch="$ARCH" -O3 -w -std=c++17 --genco \
  -H -Rpass-analysis=kernel-resource-usage \
  "$@" -I"$KX_INC_AMD" -I"$KX_INC_COMMON" -I"$KX_INC_KX" "$SRC" -o "$CO"
BUILDEOF
chmod +x "$OUT/build.sh"

# THE STALENESS KEY. Not mtimes: a whole result on this branch was a FALSE NULL because a
# candidate was measured against a stale binary, and an `-nt` test would not catch a header
# restored from git with an older timestamp. This hashes the CONTENT of the source and of every
# header the previous compile actually opened -- read out of hipcc's own `-H` include trace in the
# build log, so the list cannot go stale the way a hand-maintained one does.
recipe_hash() { # $1 = this variant's extra defines
  { echo "$BASEDEF"; echo "$ARCH"; printf '%s\n' "$1"; } | sha256sum | cut -d' ' -f1
}
deps_hash() { # $1 = build log
  { echo "$SRC"; sed -n 's/^\.\+ \(\/.*\)$/\1/p' "$1" 2>/dev/null || true; } | sort -u \
    | xargs -r sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1
}

COS=(); NEEDBUILD=(); LABELS=()
for line in "${VARLINES[@]}"; do
  lbl="${line%%$'\t'*}"; def="${line#*$'\t'}"
  co="$OUT/$lbl.co"
  LABELS+=("$lbl"); COS+=("$co")
  printf '%s' "$def" > "$co.defines.new"
  want="$(cat "$co.defines.new")"
  have="$(cat "$co.defines" 2>/dev/null || true)"
  stamp_new="$(recipe_hash "$want")"
  stamp_old="$(sed -n 1p "$co.stamp" 2>/dev/null || echo x)"
  dep_new="$(deps_hash "$OUT/$lbl.log")"
  dep_old="$(sed -n 2p "$co.stamp" 2>/dev/null || echo y)"
  if [ "$REBUILD" = 1 ] || [ ! -f "$co" ] || [ "$stamp_new" != "$stamp_old" ] || [ "$dep_new" != "$dep_old" ]; then
    NEEDBUILD+=("$lbl")
  fi
  mv "$co.defines.new" "$co.defines"
  [ "$want" = "$have" ] || true
done

if [ "${#NEEDBUILD[@]}" -gt 0 ]; then
  echo "kx: building ${#NEEDBUILD[@]}/${#LABELS[@]} code object(s) for '$EXP' [$ARCH] ..." >&2
  cat > "$OUT/buildall.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export KX_INC_AMD="$WT/runtime/amd" KX_INC_COMMON="$WT/runtime/common" KX_INC_KX="$KXD"
pids=()
EOF
  for lbl in "${NEEDBUILD[@]}"; do
    def="$(cat "$OUT/$lbl.co.defines")"
    printf '%s "%s" "%s" "%s" %s %s > "%s" 2>&1 & pids+=($!)\n' \
      "$OUT/build.sh" "$SRC" "$OUT/$lbl.co" "$ARCH" "$BASEDEF" "$def" "$OUT/$lbl.log" \
      >> "$OUT/buildall.sh"
  done
  echo 'rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc' >> "$OUT/buildall.sh"
  chmod +x "$OUT/buildall.sh"
  if ! "$NIX" develop "$WT" --command "$OUT/buildall.sh"; then
    echo "kx: BUILD FAILED. logs:" >&2
    for lbl in "${NEEDBUILD[@]}"; do echo "  --- $lbl ---" >&2; tail -25 "$OUT/$lbl.log" >&2; done
    exit 3
  fi
  for lbl in "${NEEDBUILD[@]}"; do
    co="$OUT/$lbl.co"
    recipe_hash "$(cat "$co.defines")" > "$co.stamp"
    deps_hash "$OUT/$lbl.log" >> "$co.stamp"
  done
fi

# ---------------------------------------------------------------- 3. ISA facts (optional)
if [ "$ISA" = 1 ]; then
  for i in "${!LABELS[@]}"; do
    "$NIX" develop "$WT" --command python3 "$WT/scripts/kx_isa.py" \
      --co "${COS[$i]}" --arch "$ARCH" --label "${LABELS[$i]}" --res "$OUT/${LABELS[$i]}.log"
  done
fi

# ---------------------------------------------------------------- 4. lease, pin, run
exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-1800}" \
  "$WT/perf-data/tools/gpulease" -n 1 "kx-$EXP" \
  "$DRV" "${COS[@]}" "${RUNARGS[@]+"${RUNARGS[@]}"}"
