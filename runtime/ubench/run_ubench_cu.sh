#!/usr/bin/env bash
# run_ubench_cu.sh — Plow μBench: single-CU kernel profiler for MI350X.
#
# Compiles the benchmark code object (which #includes the production op headers),
# builds the host driver, and runs under rocprof for hardware counter profiling.
#
# Usage:
#   ./run_ubench_cu.sh                        # full 3-pass rocprof profiling
#   ./run_ubench_cu.sh --timing-only          # wall-clock only (fast)
#   ./run_ubench_cu.sh --kernel d_gemm        # single kernel deep-dive
#   ./run_ubench_cu.sh --decode               # decode (M=1) configs
#   ./run_ubench_cu.sh --json results.json    # structured output
#
# Prerequisites:
#   - ROCm toolchain (hipcc, rocprof) in PATH
#   - gfx950 device (MI350X / MI355X)
#   - User must be in the 'render' group: sudo usermod -aG render $USER
#   - On multi-GPU nodes, set ROCR_VISIBLE_DEVICES=N to select a GPU

set -euo pipefail
cd "$(dirname "$0")"

# ============================================================================
# NVIDIA sm_12x (GB10 = sm_121a): `--nv` first. Runs inside `nix develop` (PLOW_NVCC, CUDA_PATH,
# PLOW_NVCC_PATH from the flake); every timed run goes through scripts/bench/gpuq.py.
#   ./run_ubench_cu.sh --nv --ceilings [--json out.json]
#   ./run_ubench_cu.sh --nv --fp4-tile [--nk 1024] [--json out.json]   (NVFP4 MMA tile loops)
#   ./run_ubench_cu.sh --nv --set nemotron-dec --assets <dir> --build-log <plowc build.log> \
#       --object interp_sm120_rs [--case name] [--reps N] [--json out.json] [--build-only]
# The ops cubin is bench_cu_sm120.cu (which includes interp_sm120.cu) built with the packet's
# plow_config.h and the define list plowc used for <object> (its "nvcc <object>.cubin (...)" log
# line plus the manifest's PLOW_EXTRA_DEFINES), checked against the production nvcc command in
# the emit's .cubin-build, so the op bodies compile exactly as served.
# ============================================================================
if [[ "${1:-}" == "--nv" ]]; then
    shift
    ARCH=sm_121a; BUILD_DIR="${BUILD_DIR:-./build}"; SET=""; ASSETS=""; BLOG=""; OBJECT=""
    BUILD_ONLY=0; CEIL=0; FP4=0; ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --arch)       ARCH="$2"; shift 2 ;;
            --ceilings)   CEIL=1; shift ;;
            --fp4-tile)   FP4=1; shift ;;
            --set)        SET="$2"; shift 2 ;;
            --assets)     ASSETS="$2"; shift 2 ;;
            --build-log)  BLOG="$2"; shift 2 ;;
            --object)     OBJECT="$2"; shift 2 ;;
            --build-only) BUILD_ONLY=1; shift ;;
            --case|--reps|--json|--kv-fp8|--list|--nk) ARGS+=("$1" "$2"); shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
    : "${PLOW_NVCC:?run inside nix develop (PLOW_NVCC unset)}"
    mkdir -p "$BUILD_DIR"
    NVCC_CUBIN=(bash ../cmake/nvcc_cubin.sh "$PLOW_NVCC")
    CEIL_CUBIN="$BUILD_DIR/ceiling_${ARCH}.cubin"
    DRIVER="$BUILD_DIR/ubench_cu_roofline_nv"
    if [[ ! -f "$CEIL_CUBIN" || ceiling_sm120.cu -nt "$CEIL_CUBIN" ]]; then
        "${NVCC_CUBIN[@]}" "$CEIL_CUBIN" ceil_mma required -arch="$ARCH" -O3 -cubin ceiling_sm120.cu
    fi
    FP4_CUBIN="$BUILD_DIR/fp4_tile_${ARCH}.cubin"
    if [[ "$FP4" -eq 1 && ( ! -f "$FP4_CUBIN" || fp4_tile_sm120.cu -nt "$FP4_CUBIN" ) ]]; then
        "${NVCC_CUBIN[@]}" "$FP4_CUBIN" ft_chain2 required -arch="$ARCH" -O3 -cubin fp4_tile_sm120.cu
    fi
    if [[ ! -f "$DRIVER" || bench_cu_roofline.c -nt "$DRIVER" || ../nvidia/cuda_backend.c -nt "$DRIVER" ]]; then
        # Link the driver's own libcuda (never the toolkit stub, which nix's ld would put on the rpath).
        LIBCUDA=""
        for d in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64 /run/opengl-driver/lib; do
            [[ -e "$d/libcuda.so.1" ]] && { LIBCUDA="$d/libcuda.so.1"; break; }
        done
        [[ -n "$LIBCUDA" ]] || { echo "no libcuda.so.1 (NVIDIA driver) found"; exit 1; }
        mkdir -p "$BUILD_DIR/libcuda"
        ln -sf "$LIBCUDA" "$BUILD_DIR/libcuda/libcuda.so.1"
        ln -sf "$LIBCUDA" "$BUILD_DIR/libcuda/libcuda.so"
        LCD="$(realpath "$BUILD_DIR/libcuda")"
        cc -O2 -std=c11 -DPLOW_CUDA -I . -I ../common -I ../amd -I ../../include \
            -I "$(dirname "$(dirname "$PLOW_NVCC")")/include" \
            bench_cu_roofline.c ../nvidia/cuda_backend.c \
            -L "$LCD" -lcuda -lm -Wl,-rpath,"$LCD" -o "$DRIVER"
    fi
    OPS=()
    if [[ -n "$SET" ]]; then
        [[ -n "$ASSETS" && -n "$BLOG" && -n "$OBJECT" ]] || { echo "--set needs --assets, --build-log, --object"; exit 1; }
        DEFS=()
        line="$(grep -m1 "nvcc ${OBJECT}.cubin (" "$BLOG" || true)"
        [[ -n "$line" ]] || { echo "no 'nvcc ${OBJECT}.cubin (...)' line in $BLOG"; exit 1; }
        IFS=';' read -r -a DEFS <<<"$(sed -E 's/.*\.cubin \((.*)\)$/\1/' <<<"$line")"
        # plowc appends the manifest's requires/recommends to every object as PLOW_EXTRA_DEFINES
        # (crates/plowc/src/main.rs, build_cubin_from_packet): the requires it filters by prefix
        # below, then every recommend except PLOW_NV_FA_GF_FULL (a CMake cache variable, already
        # in the object's line). The check after this compares against the production command.
        man="$(grep -m1 "building interpreter object from manifest" "$BLOG" | sed 's/\x1b\[[0-9;]*m//g' || true)"
        [[ -n "$man" ]] || { echo "no 'building interpreter object from manifest' line in $BLOG"; exit 1; }
        for d in $(grep -o 'requires=\[[^]]*\]' <<<"$man" | grep -o '"[^"]*"' | tr -d '"'); do
            case "$d" in
                PLOW_NV_FA_GF=*|PLOW_NV_QWEN_GDN=*|PLOW_NV_FP8_M1=*|PLOW_NV_FP8_M1_TMA=*|\
                PLOW_NV_FP8_M1_XCACHE=*|PLOW_NV_FP8_M1_FAST_ACCUM=*|PLOW_NV_FP8_M1_BK1024=*|\
                PLOW_NV_QUANT_FP8_VLLM=*|PLOW_NV_TMA_GEMM=*|PLOW_NV_FP8_PF_SCALE_WFIRST=*|\
                PLOW_NV_GEMMA3=*|PLOW_NV_FA_PIPE=*|PLOW_NV_KV_E4M3=*) DEFS+=("-D$d") ;;
            esac
        done
        for d in $(grep -o 'recommends=\[[^]]*\]' <<<"$man" | grep -o '"[^"]*"' | tr -d '"'); do
            [[ "$d" == PLOW_NV_FA_GF_FULL=* ]] || DEFS+=("-D$d")
        done
        # Fail closed: the define list must equal the production object's nvcc command, in order
        # (a repeated -D takes the last value), as CMake recorded it in the emit's .cubin-build.
        built="$(grep -m1 -o "built [^ ]*/${OBJECT}\.cubin " "$BLOG" | sed -E 's/^built //; s/ $//' || true)"
        mk="$(dirname "$built")/.cubin-build/CMakeFiles/sm120_cubins.dir/build.make"
        [[ -n "$built" && -f "$mk" ]] || { echo "no production build rule for ${OBJECT} ($mk): cannot check defines"; exit 1; }
        prod="$(grep -F "nvcc_cubin.sh" "$mk" | grep -F " $built " | head -n1 | grep -o -- ' -D[^ ]*' | sed 's/^ //; s/\\"/"/g' || true)"
        mine="$(printf '%s\n' "${DEFS[@]}")"
        if [[ -z "$prod" || "$prod" != "$mine" ]]; then
            echo "ubench define list differs from the production ${OBJECT} command:"
            diff <(echo "$prod") <(echo "$mine") || true
            exit 1
        fi
        OPS_CUBIN="$BUILD_DIR/ubench_${SET}_${ARCH}.cubin"
        if [[ ! -f "$OPS_CUBIN" || "$mine" != "$(cat "$OPS_CUBIN.defs" 2>/dev/null)" ||
              -n "$(find bench_cu_sm120.cu ../nvidia ../common "$ASSETS/plow_config.h" \
                  -newer "$OPS_CUBIN" \( -name '*.cu' -o -name '*.cuh' -o -name '*.h' \) -print -quit)" ]]; then
            "${NVCC_CUBIN[@]}" "$OPS_CUBIN" ubench_exec required -arch="$ARCH" -O3 -cubin \
                -I ../common -I ../nvidia -I "$(realpath "$ASSETS")" "${DEFS[@]}" bench_cu_sm120.cu
            echo "$mine" > "$OPS_CUBIN.defs"
        fi
        OPS=(--pkt "$ASSETS/model.pkt" --cubin "$OPS_CUBIN" --set "$SET")
    fi
    [[ "$BUILD_ONLY" -eq 1 ]] && exit 0
    if [[ "$CEIL" -eq 1 ]]; then
        exec "$DRIVER" --base "$CEIL_CUBIN" --ceilings "${ARGS[@]}"
    fi
    if [[ "$FP4" -eq 1 ]]; then
        exec "$DRIVER" --base "$CEIL_CUBIN" --fp4-tile "$FP4_CUBIN" "${ARGS[@]}"
    fi
    exec "$DRIVER" --base "$CEIL_CUBIN" "${OPS[@]}" "${ARGS[@]}"
fi

# ============================================================================
# Configuration
# ============================================================================
ARCH="${PLOW_HIP_ARCH:-gfx950}"
BUILD_DIR="${BUILD_DIR:-./build}"
ELF="${BUILD_DIR}/ubench_cu_${ARCH}.elf"
DRIVER="${BUILD_DIR}/ubench_cu_roofline"
RESULTS_DIR="${BUILD_DIR}/results"

KERNEL_FILTER=""
TIMING_ONLY=0
DECODE=""
JSON_OUT=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel)      KERNEL_FILTER="$2"; shift 2 ;;
        --timing-only) TIMING_ONLY=1; shift ;;
        --decode)      DECODE="--decode"; shift ;;
        --json)        JSON_OUT="$2"; shift 2 ;;
        --iters)       EXTRA_ARGS="$EXTRA_ARGS --iters $2"; shift 2 ;;
        --arch)        ARCH="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--kernel name] [--timing-only] [--decode] [--json path] [--iters N] [--arch gfxXXX]"
            echo ""
            echo "Kernels: d_gemm d_gemm_norm d_gemv d_gemv_norm d_rmsnorm d_rowrms"
            echo "         d_headnorm_rope d_residual d_glu d_softcap d_embed interp_overhead"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$BUILD_DIR" "$RESULTS_DIR"

# ============================================================================
# Step 1: Compile the benchmark code object
# ============================================================================
echo "═══════════════════════════════════════════════════════════════"
echo "  Plow μBench — Single CU Kernel Profiler (Gemma 4 on $ARCH)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [[ ! -f "$ELF" ]] || [[ "bench_cu_gfx950.hip" -nt "$ELF" ]] \
   || [[ "../amd/op_gemm.h" -nt "$ELF" ]] || [[ "../amd/op_norm.h" -nt "$ELF" ]] \
   || [[ "../amd/op_elementwise.h" -nt "$ELF" ]] || [[ "../amd/amd_common.h" -nt "$ELF" ]]; then
    echo "[1/3] Compiling benchmark kernels (includes production op_*.h) → $ELF"
    ROCM_PATH="${ROCM_PATH:-$(hipconfig --rocmpath 2>/dev/null || echo /opt/rocm)}"
    BUNDLER="${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler"
    BUNDLE="${BUILD_DIR}/ubench_cu_${ARCH}.bundle"
    hipcc --genco \
        --offload-arch="$ARCH" \
        -O3 -std=c++17 \
        -I ../amd -I ../common -I ../../include \
        bench_cu_gfx950.hip \
        -o "$BUNDLE"
    # hipcc --genco produces a clang offload bundle; plow_hsa_load_code_object
    # expects a raw AMDGPU ELF. Unbundle it.
    "$BUNDLER" --type=o \
        --input="$BUNDLE" \
        --output="$ELF" \
        --unbundle \
        --targets="hipv4-amdgcn-amd-amdhsa--${ARCH}"
    echo "      ✓ Code object: $(du -h "$ELF" | cut -f1) (unbundled from $(du -h "$BUNDLE" | cut -f1))"
else
    echo "[1/3] Code object up to date: $ELF"
fi

# ============================================================================
# Step 2: Compile the host driver
# ============================================================================
if [[ ! -f "$DRIVER" ]] || [[ "bench_cu_roofline.c" -nt "$DRIVER" ]]; then
    echo "[2/3] Compiling roofline driver → $DRIVER"
    ROCM_PATH="${ROCM_PATH:-$(hipconfig --rocmpath 2>/dev/null || echo /opt/rocm)}"
    # Use hipcc for the host driver too — it knows where HSA headers and libs live.
    hipcc -O2 -std=c11 -x c -D_POSIX_C_SOURCE=199309L \
        -I . -I ../common -I ../amd -I ../../include \
        -I "${ROCM_PATH}/include" \
        bench_cu_roofline.c \
        ../amd/hsa_backend.c \
        -L "${ROCM_PATH}/lib" -lhsa-runtime64 -lm \
        -Wl,-rpath,"${ROCM_PATH}/lib" \
        -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
        -o "$DRIVER"
    echo "      ✓ Driver binary: $(du -h "$DRIVER" | cut -f1)"
else
    echo "[2/3] Driver binary up to date: $DRIVER"
fi

# Symlink ELF for driver
ln -sf "$(realpath "$ELF")" "${BUILD_DIR}/ubench_cu_gfx950.elf" 2>/dev/null || true
ln -sf "$(realpath "$ELF")" "${BUILD_DIR}/bench_cu_gfx950.elf" 2>/dev/null || true

# ============================================================================
# Step 3: Run
# ============================================================================
DRIVER_ARGS="$DECODE"
[[ -n "$KERNEL_FILTER" ]] && DRIVER_ARGS="$DRIVER_ARGS --kernel $KERNEL_FILTER"
[[ -n "$JSON_OUT" ]]      && DRIVER_ARGS="$DRIVER_ARGS --json $JSON_OUT"
DRIVER_ARGS="$DRIVER_ARGS $EXTRA_ARGS"

pushd "$BUILD_DIR" > /dev/null

if [[ "$TIMING_ONLY" -eq 1 ]]; then
    echo "[3/3] Running timing-only (no hardware counters)"
    echo ""
    ./ubench_cu_roofline --timing-only $DRIVER_ARGS
else
    echo "[3/3] Running 3-pass hardware counter profiling via rocprof"
    echo ""

    PASS1="${RESULTS_DIR}/pass1.csv"
    PASS2="${RESULTS_DIR}/pass2.csv"
    PASS3="${RESULTS_DIR}/pass3.csv"

    COUNTER_DIR="$(realpath ..)"

    # Pass 1: ALU / MFMA utilization
    echo "  Pass 1/3: ALU + MFMA utilization counters..."
    rocprof --counters "${COUNTER_DIR}/bench_cu_counters_pass1.txt" \
            -o "$PASS1" \
            ./ubench_cu_roofline --timing-only $DRIVER_ARGS \
            2>&1 | grep -v "^$" || true
    echo "      ✓ $(wc -l < "$PASS1") rows"

    # Pass 2: Memory subsystem (TCP = L1, TCC = L2)
    echo "  Pass 2/3: Memory subsystem (TCP/TCC) counters..."
    rocprof --counters "${COUNTER_DIR}/bench_cu_counters_pass2.txt" \
            -o "$PASS2" \
            ./ubench_cu_roofline --timing-only $DRIVER_ARGS \
            2>&1 | grep -v "^$" || true
    echo "      ✓ $(wc -l < "$PASS2") rows"

    # Pass 3: LDS bank conflicts + occupancy
    echo "  Pass 3/3: LDS conflicts + occupancy counters..."
    rocprof --counters "${COUNTER_DIR}/bench_cu_counters_pass3.txt" \
            -o "$PASS3" \
            ./ubench_cu_roofline --timing-only $DRIVER_ARGS \
            2>&1 | grep -v "^$" || true
    echo "      ✓ $(wc -l < "$PASS3") rows"

    echo ""
    echo "  ──────────────────────────────────────────────────────────"
    echo "  Counter data collected. Re-running analysis..."
    echo ""

    # Final analysis pass with counter data
    ./ubench_cu_roofline --csv "$PASS1,$PASS2,$PASS3" $DRIVER_ARGS
fi

popd > /dev/null

echo ""
echo "Done. Results in ${RESULTS_DIR}/"
[[ -n "$JSON_OUT" ]] && echo "JSON: $JSON_OUT"
