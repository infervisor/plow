#!/usr/bin/env bash
# hipcc_hsaco.sh — compile ONE gfx9xx code object, unbundle it, and gate it.
#
#   hipcc_hsaco.sh <hipcc> <bundler> <arch> <out.elf> <kernel-symbol> \
#                  <max-total-regs> <min-occ> <hipcc args...>
#
# The AMD counterpart of cmake/nvcc_cubin.sh. Same reason for existing: the
# artifacts the AMD path actually loads were built by a SECOND, independent
# definition of the compile (scripts/build_gfx950.sh), and a build defined twice
# drifts. This is the one place the flags become an object.
#
# Two gates, both of which have already cost a debugging session on this branch:
#
#   REGISTER CLIFF. An 8-wave interpreter over 256 total regs (VGPR+AGPR) drops
#   to 1 wave/SIMD and, past the hardware budget, fails to launch at all with
#   HSA_STATUS_ERROR_INVALID_ISA — a RUNTIME error for a compile-time fact. It is
#   a build error here. Measured live: under ROCm 7.0.2/clang-20 the fp8 and MLA
#   prefill objects land at 262/occ-1 and this gate fires; under the 7.2.4/clang-22
#   toolchain the branch targets they land at 256/occ-2 and pass.
#
#   KERNEL SYMBOL. runtime/tests/gemma4_chat.c resolves the entry point by NAME
#   (`plow_interp_gfx950`, `plow_interp_dec_gfx950`, `plow_interp_flash_gfx950`,
#   `_gq`-suffixed under the global queue), so an object that compiled fine but
#   whose symbol moved is a runtime failure, not a build one.
#
# Unlike nvcc_cubin.sh there is no `env -i`: hipcc resolves its own toolchain out
# of ROCM_PATH and does not pick up the nix CPATH the CUDA host pass trips over.
set -euo pipefail

HIPCC="${1:?hipcc path}"
BUNDLER="${2:?clang-offload-bundler path}"
ARCH="${3:?offload arch}"
OUT="${4:?out.elf}"
SYM="${5:?kernel symbol}"
MAXREG="${6:?max total regs}"
MINOCC="${7:?min occupancy}"
shift 7

CO="${OUT%.elf}.co"
READELF="$(dirname "$BUNDLER")/llvm-readelf"

fail() {
    # A build that dies must leave NOTHING behind to run by mistake: a stale
    # artifact that outlives its failed compile is how a test prints CORRECT
    # against a binary that never built.
    rm -f "$OUT" "$CO"
    echo "FATAL: $*" >&2
    exit 1
}

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT" "$CO"

# ONE compile. The resource-usage remarks ride the REAL build, so the numbers
# gated on below are the numbers of the object that ships — build_gfx950.sh
# compiles a second time to measure, and a second compile is a second chance to
# measure something other than what shipped.
LOG="$("$HIPCC" --offload-arch="$ARCH" -O3 -w "$@" --genco \
       -Rpass-analysis=kernel-resource-usage -o "$CO" 2>&1)" ||
    fail "hipcc failed for $OUT
$LOG"

# Pull the block for THIS kernel rather than the first one in the file: an object
# carrying more than one __global__ would otherwise be gated on whichever the
# backend happened to emit first. The trailing space in the Function Name match
# matters — without it `plow_interp_gfx950` also matches the `_gq` twin's block.
# Every remark line ends in ` [-Rpass-analysis=kernel-resource-usage]`, so the
# value is always the second-to-last whitespace field.
read -r V A O S <<<"$(awk -v s="$SYM" '
    index($0, "Function Name: " s " ") { f = 1; next }
    f && index($0, "Function Name: ")  { exit }
    f && index($0, "VGPRs: ")       && v  == "" { v  = $(NF - 1) }
    f && index($0, "AGPRs: ")       && a  == "" { a  = $(NF - 1) }
    f && index($0, "Occupancy ")    && o  == "" { o  = $(NF - 1) }
    f && index($0, "VGPRs Spill: ") && sp == "" { sp = $(NF - 1) }
    END { print v, a, o, sp }' <<<"$LOG")"
[ -n "$V" ] && [ -n "$O" ] ||
    fail "no kernel-resource-usage remark for $SYM in $OUT — the kernel did not
       compile under these defines, or its name moved."
: "${A:=0}" "${S:=0}"

if [ "$((V + A))" -gt "$MAXREG" ] || [ "$O" -lt "$MINOCC" ]; then
    fail "$(basename "$OUT") over the register cliff: total $((V + A)) > $MAXREG, or occ $O < $MINOCC.
       This object would not launch (HSA_STATUS_ERROR_INVALID_ISA) or would run at
       half occupancy. Check the ROCm version — the branch targets 7.2.4/clang-22."
fi

"$BUNDLER" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
    --input="$CO" --output="$OUT" || fail "unbundle failed for $OUT"

# Read the symbol table into a variable rather than piping it into `grep -q`:
# grep exits at the FIRST match, closing the pipe while readelf is still
# writing; readelf dies of SIGPIPE and `set -o pipefail` promotes that to a
# pipeline failure. The gate then rejects an object whose symbol grep had just
# matched. Measured with the piped form under `-j16`: 4-5 of 17 objects failed
# per run, every one with PIPESTATUS `141 0` — writer killed by SIGPIPE, grep
# exit 0. It needs the reader to exit between two of the writer's write() calls,
# so it vanishes when run standalone (80/80 clean) and only shows under a
# loaded parallel build. cmake/nvcc_cubin.sh had the same construct.
SYMS="$("$READELF" -sW "$OUT" 2>/dev/null || true)"
grep -qE "FUNC .* $SYM\$" <<<"$SYMS" ||
    fail "$SYM not found in $OUT — kernel name/signature changed; update the
       symbol constants in runtime/tests/gemma4_chat.c."

grep -qE "OBJECT .* plow_packed_prefill_abi_1\$" <<<"$SYMS" ||
    fail "$OUT does not advertise plow_packed_prefill_abi_1"
PACKED_MLA=0; PACKED_KDA=0; K3=0; MLA=0
for arg in "$@"; do
    case "$arg" in
        -DPLOW_PACKED_PREFILL_CONSUMERS=1) PACKED_MLA=1; PACKED_KDA=1 ;;
        -DPLOW_PACKED_PREFILL_MLA_CONSUMERS=1) PACKED_MLA=1 ;;
        -DPLOW_PACKED_PREFILL_KDA_CONSUMERS=1) PACKED_KDA=1 ;;
        -DPLOW_K3=1) K3=1 ;;
        -DPLOW_MLA_PREFILL=1|-DPLOW_MLA_PF_V2_ARM=1) MLA=1 ;;
    esac
done
for cap in mla kda; do
    marker="plow_packed_prefill_${cap}_consumers_1"
    required=0
    [ "$cap" = mla ] && [ "$PACKED_MLA" = 1 ] && required=$MLA
    [ "$cap" = kda ] && [ "$PACKED_KDA" = 1 ] && required=$K3
    if [ "$required" = 1 ]; then
        grep -qE "OBJECT .* ${marker}\$" <<<"$SYMS" || fail "$OUT is missing $marker"
    elif grep -qE "OBJECT .* ${marker}\$" <<<"$SYMS"; then
        fail "default $OUT unexpectedly advertises $marker"
    fi
done

rm -f "$CO"
printf "built %s (%s B), %s VGPR=%s AGPR=%s total=%s occ=%s spill=%s\n" \
    "$OUT" "$(stat -c%s "$OUT")" "$SYM" "$V" "$A" "$((V + A))" "$O" "$S"
