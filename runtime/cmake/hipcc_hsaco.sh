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

# Resource remarks omit final SGPR/private-segment facts. Read the record for
# the exact shipping entry point, not the first helper kernel in the ELF.
META="$($READELF -n "$OUT" 2>/dev/null || true)"
read -r MV MA MSG MSP MP MW MWG MLDS <<<"$(awk -v want="$SYM" '
    function emit() {
        if (name == want) {
            print vgpr+0, agpr+0, sgpr+0, sgspill+0, private+0, wave+0, wg+0, lds+0
            found = 1
        }
    }
    $1 == "-" && $2 == ".agpr_count:" {
        if (started) emit()
        started = 1; name = ""; vgpr = 0; agpr = $3; sgpr = 0; sgspill = 0
        private = 0; wave = 0; wg = 0; lds = 0
        next
    }
    started && $1 == ".name:"                       { name = $2 }
    started && $1 == ".vgpr_count:"                 { vgpr = $2 }
    started && $1 == ".sgpr_count:"                 { sgpr = $2 }
    started && $1 == ".sgpr_spill_count:"           { sgspill = $2 }
    started && $1 == ".private_segment_fixed_size:" { private = $2 }
    started && $1 == ".wavefront_size:"             { wave = $2 }
    started && $1 == ".max_flat_workgroup_size:"    { wg = $2 }
    started && $1 == ".group_segment_fixed_size:"   { lds = $2 }
    END { if (!found) emit() }
' <<<"$META")"
[ -n "${MV:-}" ] || fail "$SYM has no AMDGPU metadata record in $OUT"
[ "$MW" = 64 ] || fail "$SYM advertises wavefront_size=$MW; gfx9xx requires wave64"
[ "$MWG" -ge 64 ] && [ "$MWG" -le 1024 ] && [ $((MWG % MW)) -eq 0 ] ||
    fail "$SYM has invalid max workgroup/wave geometry: wg=$MWG wave=$MW"

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

LEAN=0; NOSPILL=0; REQUIRED_MARKER=""
for arg in "$@"; do
    case "$arg" in
        -DPLOW_LEAN_OBJECT=1) LEAN=1 ;;
        -DPLOW_NO_SPILL=1) NOSPILL=1 ;;
        -DPLOW_REQUIRED_MARKER=*) REQUIRED_MARKER=${arg#*=} ;;
    esac
done
if [ "$LEAN" != 1 ]; then
    grep -qE "OBJECT .* plow_packed_prefill_abi_1\$" <<<"$SYMS" ||
        fail "$OUT does not advertise plow_packed_prefill_abi_1"
fi
[ "$NOSPILL" != 1 ] || [ "$S" = 0 ] ||
    fail "$(basename "$OUT") spills $S VGPRs"
[ "$NOSPILL" != 1 ] || [ "$MP" = 0 ] ||
    fail "$(basename "$OUT") has a ${MP}-byte private segment"
if [ -n "$REQUIRED_MARKER" ]; then
    grep -qE "OBJECT .* ${REQUIRED_MARKER}\$" <<<"$SYMS" ||
        fail "$OUT does not advertise $REQUIRED_MARKER"
fi
PACKED_MLA=0; PACKED_MLA_NORM=0; PACKED_MLA_FLASH=0; PACKED_KDA=0; KDA_CHUNK=0; KDA_QPRE=0
K3=0; MLA=0; HIER=0; L2=0; GQ=0; DECODE=0
for arg in "$@"; do
    case "$arg" in
        -DPLOW_PACKED_PREFILL_CONSUMERS=1) PACKED_MLA=1; PACKED_KDA=1 ;;
        -DPLOW_PACKED_PREFILL_MLA_CONSUMERS=1) PACKED_MLA=1 ;;
        -DPLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS=1) PACKED_MLA_NORM=1 ;;
        -DPLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS=1) PACKED_MLA_FLASH=1 ;;
        -DPLOW_PACKED_PREFILL_KDA_CONSUMERS=1) PACKED_KDA=1 ;;
        -DPLOW_KDA_CHUNK=1) KDA_CHUNK=1 ;;
        -DPLOW_KDA_CHUNK_QPRE=1) KDA_QPRE=1 ;;
        -DPLOW_K3=1) K3=1 ;;
        -DPLOW_MLA_PREFILL=1|-DPLOW_MLA_PF_V2_ARM=1) MLA=1 ;;
        -DPLOW_GATE_HIER=1) HIER=1 ;;
        -DPLOW_L2_PLACE_DISPATCH=1) L2=1 ;;
        -DPLOW_GLOBAL_QUEUE=1) GQ=1 ;;
        -DPLOW_BUCKET_DECODE=1) DECODE=1 ;;
    esac
done
if [ "$HIER" = 1 ] && { [ "$L2" != 1 ] || [ "$GQ" != 1 ] || [ "$DECODE" != 1 ]; }; then
    fail "$OUT enables PLOW_GATE_HIER outside a decode GQ object with L2-domain dispatch"
fi
if [ "$HIER" = 1 ]; then
    grep -qE "OBJECT .* plow_gate_hier_1\$" <<<"$SYMS" ||
        fail "$OUT enables PLOW_GATE_HIER but is missing plow_gate_hier_1"
elif grep -qE "OBJECT .* plow_gate_hier_1\$" <<<"$SYMS"; then
    fail "$OUT unexpectedly advertises plow_gate_hier_1"
fi
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

for spec in \
    "PLOW_BUCKET_PACKED_MLA_NORM=plow_packed_prefill_mla_norm_segments_1" \
    "PLOW_BUCKET_FLASH=plow_packed_prefill_mla_flash_segments_1" \
    "PLOW_BUCKET_PACKED_KDA=plow_packed_prefill_kda_serial_segments_1" \
    "PLOW_BUCKET_PACKED_KDA=plow_kda_family_segments_1"; do
    flag=${spec%%=*}
    marker=${spec#*=}
    required=0
    for arg in "$@"; do
        case "$arg" in
            -D${flag}|-D${flag}=1) required=1 ;;
        esac
    done
    # Ordinary flash objects do not consume packed descriptors.
    if [ "$flag" = PLOW_BUCKET_FLASH ] && [ "$PACKED_MLA_FLASH" != 1 ]; then
        required=0
    fi
    if [ "$required" = 1 ]; then
        grep -qE "OBJECT .* ${marker}\$" <<<"$SYMS" || fail "$OUT is missing $marker"
    fi
done

if [ "$PACKED_KDA" = 1 ] && [ "$KDA_CHUNK" = 1 ]; then
    for marker in plow_kda_family_segments_1 plow_packed_prefill_kda_chunk_segments_1 \
                  plow_kda_chunk_bt64_arm_1; do
        grep -qE "OBJECT .* ${marker}\$" <<<"$SYMS" || fail "$OUT is missing $marker"
    done
    if [ "$KDA_QPRE" = 1 ]; then
        grep -qE "OBJECT .* plow_kda_chunk_qpre_arm_1\$" <<<"$SYMS" ||
            fail "$OUT enables chunk-KDA qpre but is missing plow_kda_chunk_qpre_arm_1"
    fi
    [ "$S" = 0 ] || fail "$(basename "$OUT") packed chunk-KDA object spills $S VGPRs"
elif grep -qE "OBJECT .* plow_packed_prefill_kda_chunk_segments_1\$" <<<"$SYMS"; then
    fail "default $OUT unexpectedly advertises plow_packed_prefill_kda_chunk_segments_1"
fi

rm -f "$CO"
printf "built %s (%s B), %s VGPR=%s AGPR=%s total=%s occ=%s vgpr_spill=%s metadata=[vgpr=%s agpr=%s sgpr=%s sgpr_spill=%s private=%sB lds=%sB wave=%s wgmax=%s waves=%s]\n" \
    "$OUT" "$(stat -c%s "$OUT")" "$SYM" "$V" "$A" "$((V + A))" "$O" "$S" \
    "$MV" "$MA" "$MSG" "$MSP" "$MP" "$MLDS" "$MW" "$MWG" "$((MWG / MW))"
