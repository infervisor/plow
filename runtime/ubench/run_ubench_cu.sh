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
