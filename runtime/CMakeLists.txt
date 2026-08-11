cmake_minimum_required(VERSION 3.18)
project(plow_runtime C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(PLOW_CUDA "Build the CUDA (Hopper/Blackwell) kernel backend" OFF)
# --- kernel-arm options shared by BOTH nvidia build paths ---------------------
# Declared here, once, because two consumers read them: the STATIC LIBS the
# tests link (under `if(PLOW_CUDA)` below) and the SERVED .cubin artifacts
# (`if(PLOW_SM120_CUBIN)`, further down). They used to be declared only inside
# the CUDA block while scripts/build_sm120_cubin.sh carried its own hand-written
# copy of every define set — and the two drifted. See the cubin section's header
# comment for what that cost.
#
# rtx-11 PX-2: PLOW_NV_W8A8 selects the true w8a8 fp8 mma.sync.m16n8k32 prefill
# path (both operands e4m3 + per-row activation quant) on ALL gemma interp
# objects, so a w8a8 pkt traces/runs consistently. OFF keeps the w8a16 dequant
# path.
option(PLOW_NV_W8A8 "Build the gemma interp objects with the true w8a8 fp8 mma path" OFF)
# rtx-19 E3: fp8 (e4m3) KV cache. Compiles the HEADNORM_ROPE_FP8 /
# FLASH_DECODE_FP8 arms into every gemma object so the shared dispatch is
# consistent decode/prefill/seg. OFF => bf16 KV, byte-identical to HEAD.
option(PLOW_FP8_KV "Build the gemma interp objects with the fp8 e4m3 KV-cache arms" OFF)
# beat-fp8-mma: keep the fp8-KV prefill objects on the FAST PIPE=1 build — the
# FLASH_PREFILL_FP8 op then runs an fp8-mma arm instead of the synchronous PIPE=0
# staging path. OFF preserves the legacy all-layer fp8-KV PIPE=0 prefill.
#
# NO LONGER hd512-only. Until PX-23 the only fast arm was px4 (hd512 FULL layers),
# so an ALL-LAYER e4m3 packet emitted hd256 FLASH_PREFILL_FP8, trapped, and the
# whole packet fell to PIPE=0 at ~176 s of prefill per 127k request — the whole of
# PX-20's measured 7.61x all-fp8 gap. d_flash_prefill_px23<256,64,32> covers hd256
# now, so BOTH head dims run the fast path and the mixed-packet workaround
# (PLOW_FP8_KV_FULL=1) is no longer required to avoid the trap.
option(PLOW_FP8_KV_FASTPF "fp8-KV prefill via the PIPE=1 fp8-mma arms (hd256 px23 + hd512 px4)" OFF)
option(PLOW_ROCM "Build the ROCm (MI300/MI350) kernel backend" OFF)

# CDNA3 (MI300X/MI325X) and CDNA4 (MI350X/MI355X). Both are compiled into the
# one plugin; the loader picks the per-arch table at runtime. Override to a
# single arch to halve device-compile time during kernel iteration, e.g.
#   -DPLOW_HIP_ARCH=gfx950
set(PLOW_HIP_ARCH "gfx942;gfx950" CACHE STRING "HIP offload architectures")

set(PLOW_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu)

# --- host library: ABI decode + dispatch + interpreter + CPU golden kernels ---
add_library(plow_runtime STATIC
    common/dispatch.c
    common/decode.c
    common/interp.c
    common/memmap.c
    cpu/gemm.c
    cpu/flash.c
    cpu/row.c
    cpu/layout.c
    cpu/dma.c
    cpu/rdma.c
    cpu/control.c
    cpu/register.c)
target_include_directories(plow_runtime PUBLIC ${PLOW_INCLUDE})
target_compile_options(plow_runtime PRIVATE -Wall -Wextra)
# PIC: the GPU backends below are SHARED plugins that link this core statically,
# and a shared object can only absorb PIC objects.
set_target_properties(plow_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT WIN32)
    target_link_libraries(plow_runtime PUBLIC m)
endif()

# --- tests --------------------------------------------------------------------
enable_testing()

add_executable(abi_layout tests/abi_layout.c)
target_link_libraries(abi_layout PRIVATE plow_runtime)
add_test(NAME abi_layout COMMAND abi_layout)

add_executable(golden_program tests/golden_program.c)
target_link_libraries(golden_program PRIVATE plow_runtime)
add_test(NAME golden_program COMMAND golden_program)

add_executable(memmap_test tests/memmap_test.c)
target_link_libraries(memmap_test PRIVATE plow_runtime)
add_test(NAME memmap_test COMMAND memmap_test)

# --- GPU backends (build-gated; unrunnable without a toolchain) ---------------
#
# These are SHARED plugins, not static libs: nothing in the host links them, and
# nothing should. `plowrt` is a static binary that `dlopen`s a vendor backend on
# the machine that actually has the hardware, then `dlsym`s one `plow_register_*`
# to populate its `dispatch_table` (the table is a plain function-pointer array,
# so it crosses the .so boundary as data). That keeps the NVIDIA and AMD
# toolchains out of the host build entirely, and lets one process carry both.
#
# Each plugin absorbs the (PIC) core statically, so it is self-contained apart
# from its vendor runtime.
if(PLOW_CUDA)
    enable_language(CUDA)
    add_library(plow_runtime_cuda SHARED
        nvidia/gemm.cu
        nvidia/flash.cu
        nvidia/row.cu
        nvidia/layout.cu
        nvidia/dma.cu
        nvidia/gemma_sm120.cu
        nvidia/register_hopper.cu
        nvidia/register_blackwell.cu
        nvidia/register_rtx6000.cu)
    target_include_directories(plow_runtime_cuda PUBLIC ${PLOW_INCLUDE})
    # Hopper sm_90a / datacenter Blackwell sm_100a / consumer Blackwell (RTX 6000
    # Pro, RTX 5090) sm_120a. All compiled; the loader picks the per-arch table.
    set_target_properties(plow_runtime_cuda PROPERTIES
        CUDA_ARCHITECTURES "90a;100a;120a"
        OUTPUT_NAME plow_rt_cuda)
    # PRIVATE: the core is absorbed, not re-exported to whoever dlopens us.
    target_link_libraries(plow_runtime_cuda PRIVATE plow_runtime)

    # Numeric validation of the sm_120 kernels against the CPU f32 golden oracle.
    # GUARDED: tests/sm120_numeric_test.cu is authored on a concurrent branch (the
    # gemma_sm120.cu B-tile fix). Configuring without it is a hard CMake error, which
    # blocks every OTHER CUDA target in this file — including the interpreter and the
    # end-to-end chat harness, neither of which depends on it. Skip rather than fail;
    # the block below is unchanged and comes back the moment the file lands.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/sm120_numeric_test.cu)
    add_executable(sm120_numeric_test tests/sm120_numeric_test.cu)
    set_target_properties(sm120_numeric_test PROPERTIES CUDA_ARCHITECTURES "120a")
    target_link_libraries(sm120_numeric_test PRIVATE plow_runtime_cuda plow_runtime)

    # ONE CTest entry PER CASE, not one run of the whole binary: a CUDA fault is
    # sticky, so a single-process run aborts at the first failing case and every
    # case after it is silently never executed. Separate processes also make the
    # failing case obvious from the ctest summary.
    set(SM120_CASES
        gemm_exact_1tile gemm_exact_multi gemm_exact_ragged
        gemm_exact_ragged_k gemm_exact_ragged_k40 gemm_real
        gemm_norm gemm_norm_small gemm_splitk gemm_norm_splitk
        flash_causal_small flash_causal_big flash_decode
        flash_sliding flash_sliding_small
        row_rmsnorm row_residual row_swiglu
        row_normrope row_normropescale row_normrope_pos
        row_normrope_hd64 row_normrope_hd32
        layout_gather
        negctl_gemm negctl_rmsnorm negctl_flash negctl_gemm_ragged_k)
    foreach(case ${SM120_CASES})
        add_test(NAME sm120_${case} COMMAND sm120_numeric_test ${case})
    endforeach()
    # Guard the guard: an unmatched case name must fail, or a typo above would
    # read as a pass having run nothing.
    add_test(NAME sm120_unknown_case_fails COMMAND sm120_numeric_test __no_such_case__)
    set_tests_properties(sm120_unknown_case_fails PROPERTIES WILL_FAIL TRUE)
    endif()

    # ---- the persistent packet interpreter -------------------------------------------
    # Built as its own object library rather than folded into plow_runtime_cuda: it is a
    # COOPERATIVE-launch megakernel with a 154-register footprint, and it must be compiled
    # for sm_120a alone. plow_runtime_cuda is built for "90a;100a;120a", and this kernel
    # has no Hopper story (its occupancy and grid sizing are derived from GB202's 170 SMs).
    add_library(plow_interp_sm120 STATIC nvidia/interp_sm120.cu)
    target_include_directories(plow_interp_sm120 PUBLIC ${PLOW_INCLUDE} nvidia)
    set_target_properties(plow_interp_sm120 PROPERTIES
        CUDA_ARCHITECTURES "120a"
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON)

    # Hopper persistent interpreter. H100 and H200 are both sm_90a; runtime SM
    # count and the occupancy-derived cooperative grid distinguish SKUs/MIG.
    add_library(plow_interp_sm90a STATIC nvidia/interp_sm90a.cu)
    target_include_directories(plow_interp_sm90a PUBLIC ${PLOW_INCLUDE} nvidia)
    set_target_properties(plow_interp_sm90a PROPERTIES
        CUDA_ARCHITECTURES "90a" CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON)

    # Per-op numeric validation of the interpreter's dispatch arms against the CPU golden.
    add_executable(sm120_interp_op_test tests/sm120_interp_op_test.cu)
    set_target_properties(sm120_interp_op_test PROPERTIES CUDA_ARCHITECTURES "120a")
    target_include_directories(sm120_interp_op_test PRIVATE nvidia common)
    target_link_libraries(sm120_interp_op_test PRIVATE plow_interp_sm120 plow_runtime)
    add_test(NAME sm120_interp_ops COMMAND sm120_interp_op_test)

    # NEGATIVE CONTROL AS A BUILD TARGET. The same test compiled with the naively
    # transliterated wave64 reduction offsets (__shfl_xor starting at 32 on a 32-lane warp).
    # It MUST fail: measured, it breaks 7 of 10 ops (HEADNORM_ROPE relL2 0.29, FLASH 0.50).
    # This is what makes the passing run above mean something — it proves the suite detects
    # exactly the wave64->warp32 porting bug the whole exercise is exposed to.
    # It compiles interp_sm120.cu into ITSELF rather than linking plow_interp_sm120: the
    # define has to reach the reductions inside the INTERPRETER's op bodies, and the library
    # above is (correctly) built without it. Linking the good library here would produce a
    # control that silently passes, which is the exact failure mode this target exists to
    # rule out elsewhere.
    add_executable(sm120_interp_op_test_w64
        tests/sm120_interp_op_test.cu nvidia/interp_sm120.cu)
    set_target_properties(sm120_interp_op_test_w64 PROPERTIES CUDA_ARCHITECTURES "120a")
    target_include_directories(sm120_interp_op_test_w64 PRIVATE nvidia common)
    target_compile_definitions(sm120_interp_op_test_w64 PRIVATE FA_NV_WAVE64_NEGCTRL)
    target_link_libraries(sm120_interp_op_test_w64 PRIVATE plow_runtime)
    add_test(NAME sm120_interp_wave64_negctrl COMMAND sm120_interp_op_test_w64)
    set_tests_properties(sm120_interp_wave64_negctrl PROPERTIES WILL_FAIL TRUE)

    # End-to-end: a real Qwen3-4B token from real weights, greedy-decoded against an HF
    # reference. NOT registered as a ctest — it needs a 7.5 GiB checkpoint on disk, a
    # compiled .pkt and a tokenized prompt, none of which the build tree provides.
    add_executable(qwen3_sm120_chat tests/qwen3_sm120_chat.cu)
    set_target_properties(qwen3_sm120_chat PROPERTIES CUDA_ARCHITECTURES "120a")
    target_include_directories(qwen3_sm120_chat PRIVATE nvidia common ${PLOW_INCLUDE})
    target_link_libraries(qwen3_sm120_chat PRIVATE plow_interp_sm120)

    # ---- Gemma-4 objects: DECODE (hd 256/512, GF=2) and PREFILL (tiled GEMM + FLASH_PREFILL) ----
    # Two objects from the ONE interp_sm120.cu: the prefill arms (mma.sync GEMM, hd 256/512 flash
    # prefill) are register/smem-hungry (217 regs, ~68 KiB smem) so they build SEPARATELY from the
    # 150-reg decode object rather than stacking (rtx-04 §3 / rtx-06 register warning). The prefill
    # object's exported symbols are suffixed `_pf` (PLOW_SYM), so both link into one harness.
    # rtx-11 PX-2: -DPLOW_NV_W8A8=ON (declared at the top of this file, shared with the served
    # cubins) selects the true w8a8 fp8 mma.sync.m16n8k32 prefill path (both operands e4m3 +
    # per-row activation quant) on ALL gemma interp objects, so a w8a8 pkt traces/runs
    # consistently. OFF keeps the w8a16 dequant path. Applied to every gemma object so the shared
    # PLOW_NV_W8A8 dispatch + d_quant_fp8 compile identically across decode/prefill/seg.
    set(PLOW_NV_W8A8_DEFS "")
    if(PLOW_NV_W8A8)
        set(PLOW_NV_W8A8_DEFS PLOW_NV_W8A8=1)
    endif()

    # fp8 (e4m3) KV cache (rtx-19 E3; PLOW_FP8_KV is declared at the top of this file, shared with
    # the served cubins). Compiles the HEADNORM_ROPE_FP8 / FLASH_DECODE_FP8 arms into every gemma
    # object so the shared dispatch is consistent decode/prefill/seg. OFF => bf16 KV,
    # byte-identical to HEAD. Applied via ${PLOW_FP8_KV_DEFS} alongside the w8a8 defs.
    set(PLOW_FP8_KV_DEFS "")
    # The prefill-family objects (_pf/_seg/_gemm) additionally need PLOW_NV_FA_PIPE=0 when fp8-KV is
    # on: the FLASH_PREFILL_FP8 arm dequants at the smem stage (cp.async cannot convert fp8 inline),
    # so it only exists on the synchronous-staging (PIPE=0) build and traps under PIPE=1. Decode is
    # PIPE-agnostic. bf16-KV runs use the default (PIPE=1) harness, so this does not affect them.
    set(PLOW_FP8_KV_PF_DEFS "")
    # beat-fp8-mma: PLOW_FP8_KV_FASTPF (declared at the top of this file, shared with the served
    # cubins) keeps the prefill objects on the FAST PIPE=1 build — the FLASH_PREFILL_FP8 op then
    # runs the px4 fp8-mma arm (hd512 only; pair with mixed-KV packets, PLOW_FP8_KV_FULL=1). OFF
    # preserves the legacy all-layer fp8-KV PIPE=0 prefill.
    # PX-8: the fp8-mma prefill arm with an e4m3 P.V at BKV=32 (perf-data/px8-flash-fp8-pv.md).
    # Only meaningful ON TOP OF PLOW_FP8_KV_FASTPF, since it replaces that arm's P.V; it is an
    # sm_120a-only opcode (the 8-bit transposing ldmatrix) and it changes numerics, so it is OFF
    # by default and only the prefill-family objects ever see it.
    option(PLOW_NV_FA_FP8PV "fp8-KV prefill P.V as mma.m16n8k32.e4m3 (needs PLOW_FP8_KV_FASTPF)" OFF)
    if(PLOW_FP8_KV)
        set(PLOW_FP8_KV_DEFS PLOW_FP8_KV=1)
        if(PLOW_FP8_KV_FASTPF)
            set(PLOW_FP8_KV_PF_DEFS PLOW_FP8_KV=1)
            if(PLOW_NV_FA_FP8PV)
                list(APPEND PLOW_FP8_KV_PF_DEFS PLOW_NV_FA_FP8PV=1)
            endif()
        else()
            set(PLOW_FP8_KV_PF_DEFS PLOW_FP8_KV=1 PLOW_NV_FA_PIPE=0)
            if(PLOW_NV_FA_FP8PV)
                message(FATAL_ERROR "PLOW_NV_FA_FP8PV needs PLOW_FP8_KV_FASTPF=ON (it replaces "
                                    "the px4 fp8-mma arm's P.V, and FASTPF=OFF has no such arm)")
            endif()
            # Not an error — all-layer fp8-KV packets REQUIRE this path — but it is the slow one
            # and the cost is invisible at runtime, so say it at configure time. FASTPF cannot be
            # defaulted ON: legality depends on the PACKET (emit-time PLOW_FP8_KV_FULL=1 keeps
            # e4m3 to the hd512 layers), which this build cannot see. Measured: 13843 -> 10947 ms,
            # -21% total prefill at 67k (perf-data/px8-flash-fp8-pv.md).
            message(WARNING
                "PLOW_FP8_KV=ON with PLOW_FP8_KV_FASTPF=OFF: the prefill objects fall back to "
                "PLOW_NV_FA_PIPE=0, losing the cp.async flash pipeline for prefill (-21% at 67k). "
                "If you emit MIXED fp8-KV packets (PLOW_FP8_KV_FULL=1), set PLOW_FP8_KV_FASTPF=ON. "
                "Keep it OFF only for legacy all-layer fp8-KV packets, which trap under PIPE=1.")
        endif()
    endif()

    add_library(plow_interp_sm120_gemma STATIC nvidia/interp_sm120.cu)
    target_include_directories(plow_interp_sm120_gemma PUBLIC ${PLOW_INCLUDE} nvidia)
    target_compile_definitions(plow_interp_sm120_gemma PRIVATE PLOW_NV_GEMMA=1 PLOW_NV_FA_GF=2 ${PLOW_NV_W8A8_DEFS} ${PLOW_FP8_KV_DEFS})
    # Opt-in per-op cycle attribution for the DECODE program. -DPLOW_NV_TRACE_DECODE=ON compiles
    # ONLY the decode object with PLOW_NV_TRACE=1; the launch helper self-dumps block-0's per-packet
    # gate/body/sig for the first decode step (PLOW_TRACE lines on stdout; PLOW_NV_TRACE_SKIP env
    # picks a warm launch index). Mutually exclusive with PLOW_NV_TRACE_PF so the g_tr_* device
    # globals resolve to exactly one definition. PLOW_NV_TRACE_DEC is the T9a-era alias.
    option(PLOW_NV_TRACE_DECODE "Instrument the decode object with per-op cycle trace" OFF)
    option(PLOW_NV_TRACE_DEC "Alias of PLOW_NV_TRACE_DECODE" OFF)
    if(PLOW_NV_TRACE_DECODE OR PLOW_NV_TRACE_DEC)
        target_compile_definitions(plow_interp_sm120_gemma PRIVATE PLOW_NV_TRACE=1)
    endif()
    # Measurement-only: run the decode gate/signal skeleton with NO op bodies (garbage logits) to
    # size the interpreter's pure scheduling floor. -DPLOW_NV_SKELETON_DECODE=ON.
    option(PLOW_NV_SKELETON_DECODE "Build decode object with op bodies removed (dispatch floor)" OFF)
    if(PLOW_NV_SKELETON_DECODE)
        target_compile_definitions(plow_interp_sm120_gemma PRIVATE PLOW_NV_SKELETON=1)
    endif()
    set_target_properties(plow_interp_sm120_gemma PROPERTIES
        CUDA_ARCHITECTURES "120a" CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

    add_library(plow_interp_sm90a_gemma STATIC nvidia/interp_sm90a.cu)
    target_include_directories(plow_interp_sm90a_gemma PUBLIC ${PLOW_INCLUDE} nvidia)
    target_compile_definitions(plow_interp_sm90a_gemma PRIVATE
        PLOW_NV_GEMMA=1 PLOW_NV_FA_GF=2 ${PLOW_FP8_KV_DEFS})
    set_target_properties(plow_interp_sm90a_gemma PROPERTIES
        CUDA_ARCHITECTURES "90a" CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

    add_library(plow_interp_sm90a_gemma_pf STATIC nvidia/interp_sm90a.cu)
    target_include_directories(plow_interp_sm90a_gemma_pf PUBLIC ${PLOW_INCLUDE} nvidia)
    target_compile_definitions(plow_interp_sm90a_gemma_pf PRIVATE
        PLOW_NV_GEMMA=1 PLOW_NV_PREFILL=1 PLOW_NV_FA_GF=2 ${PLOW_FP8_KV_PF_DEFS})
    set_target_properties(plow_interp_sm90a_gemma_pf PROPERTIES
        CUDA_ARCHITECTURES "90a" CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

    add_library(plow_interp_sm120_gemma_pf STATIC nvidia/interp_sm120.cu)
    target_include_directories(plow_interp_sm120_gemma_pf PUBLIC ${PLOW_INCLUDE} nvidia)
    target_compile_definitions(plow_interp_sm120_gemma_pf PRIVATE
        PLOW_NV_GEMMA=1 PLOW_NV_PREFILL=1 PLOW_NV_FA_GF=2 ${PLOW_NV_W8A8_DEFS} ${PLOW_FP8_KV_PF_DEFS})
    # Opt-in per-op cycle attribution for the PREFILL bucket programs. -DPLOW_NV_TRACE_PF=ON
    # compiles ONLY the _pf object with PLOW_NV_TRACE=1; the launch helper self-dumps block-0's
    # per-packet gate/body/sig for the first prefill chunk (PLOW_TRACE lines on stdout). The decode
    # object stays trace-free, so the g_tr_* device globals resolve to exactly one definition.
    option(PLOW_NV_TRACE_PF "Instrument the prefill (_pf) object with per-op cycle trace" OFF)
    if(PLOW_NV_TRACE_PF)
        target_compile_definitions(plow_interp_sm120_gemma_pf PRIVATE PLOW_NV_TRACE=1)
    endif()
    # PX-6 recommendation A: compile the M=1 GEMV arm into the PREFILL object so a blob emitted
    # with PLOW_PF_GEMV_HEAD=1 can run lm_head on it instead of the tiled arm (which computes
    # BM=128 rows to keep 1). Measured 1.991 -> 1.213 ms on the op; +2 registers (236 -> 238),
    # 0 spill, occupancy unchanged. Only gemv_rows<1> is instantiated, NOT gemv_walk.
    # With this ON the object carries BOTH arms, so a base-vs-swapped A/B is one binary and two
    # blobs differing in exactly 5 bytes (the lm_head opcode, one per prefill bucket) — no
    # register or codegen delta between the two arms of the comparison.
    option(PLOW_NV_PF_GEMV_HEAD "PX-6 A: M=1 GEMV arm in the prefill object (lm_head)" OFF)
    if(PLOW_NV_PF_GEMV_HEAD)
        target_compile_definitions(plow_interp_sm120_gemma_pf PRIVATE PLOW_NV_PF_GEMV_HEAD=1)
    endif()
    set_target_properties(plow_interp_sm120_gemma_pf PROPERTIES
        CUDA_ARCHITECTURES "120a" CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

    # T9c SEGMENTED prefill object: same prefill arms as _pf, but built PLOW_NV_SEGMENTS=1 so the
    # interpreter bounds each cooperative launch to prog.cur_seg's window. Exported symbols are
    # suffixed `_pfseg` (PLOW_SYM), so it co-links with the byte-identical single-segment _pf object.
    # The harness selects it only when PLOW_NV_SEGMENTS=1 is set at runtime, so defaults are unchanged.
    add_library(plow_interp_sm120_gemma_seg STATIC nvidia/interp_sm120.cu)
    target_include_directories(plow_interp_sm120_gemma_seg PUBLIC ${PLOW_INCLUDE} nvidia)
    target_compile_definitions(plow_interp_sm120_gemma_seg PRIVATE
        PLOW_NV_GEMMA=1 PLOW_NV_PREFILL=1 PLOW_NV_FA_GF=2 PLOW_NV_SEGMENTS=1 ${PLOW_NV_W8A8_DEFS} ${PLOW_FP8_KV_PF_DEFS})
    set_target_properties(plow_interp_sm120_gemma_seg PROPERTIES
        CUDA_ARCHITECTURES "120a" CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

    # T9c Stage 2 — LEAN GEMM segment object (PLOW_NV_SEG_GEMM=1): the class-8 (GEMM/tier-A) segments
    # get an occupancy-2 object. FLASH_PREFILL + the dead 26B-MoE prefill arms compile out, the arena
    # shrinks to the GEMM tile (~40 KiB), and __launch_bounds__(256,2) caps at 128 regs. ptxas: 128
    # regs / occ-2 reachable (2x40 KiB < 100 KiB; 128x256x2 = 64K regs/SM exactly), ~600 B residual
    # spill from the shared switch. Built here so the register gate is CI-visible; the harness does NOT
    # yet drive it — occ-2 needs the program re-sliced to 2*n_cu blocks (the emitter slices for n_cu),
    # which is the Stage-3 prerequisite. Symbols `_pfgemm`.
    add_library(plow_interp_sm120_gemma_gemm STATIC nvidia/interp_sm120.cu)
    target_include_directories(plow_interp_sm120_gemma_gemm PUBLIC ${PLOW_INCLUDE} nvidia)
    target_compile_definitions(plow_interp_sm120_gemma_gemm PRIVATE
        PLOW_NV_GEMMA=1 PLOW_NV_PREFILL=1 PLOW_NV_FA_GF=2 PLOW_NV_SEGMENTS=1 PLOW_NV_SEG_GEMM=1 ${PLOW_NV_W8A8_DEFS} ${PLOW_FP8_KV_PF_DEFS})
    set_target_properties(plow_interp_sm120_gemma_gemm PROPERTIES
        CUDA_ARCHITECTURES "120a" CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

    # rtx-11 PX-3 — LEAN GEMM object with a BN=64 N-tile that reaches occ-2 WITHOUT halving the
    # pipeline (PLOW_NV_SEG_GEMM_BN64=1). Unlike the T10 object above (BN=128 + halved 2/1 stages,
    # 620/780 B spill), this keeps the full 3-stage plain / 2-stage GLU ring: 45 KiB arena, 128 regs,
    # 28/40 B spill, occ-2 VERIFIED (perf-data/px3-bn64-occ2.md). Built here so the register/occupancy
    # gate is CI-visible; NOT linked into the harness (same `_pfgemm` symbols as the object above —
    # they cannot co-link) and the e2e occ-2 win is blocked on the emitter 2*n_cu re-slice (Stage-3),
    # so this is the correctness-/occupancy-verified building block, not a driven object.
    add_library(plow_interp_sm120_gemma_gemm_bn64 STATIC nvidia/interp_sm120.cu)
    target_include_directories(plow_interp_sm120_gemma_gemm_bn64 PUBLIC ${PLOW_INCLUDE} nvidia)
    target_compile_definitions(plow_interp_sm120_gemma_gemm_bn64 PRIVATE
        PLOW_NV_GEMMA=1 PLOW_NV_PREFILL=1 PLOW_NV_FA_GF=2 PLOW_NV_SEGMENTS=1 PLOW_NV_SEG_GEMM=1
        PLOW_NV_SEG_GEMM_BN64=1)
    set_target_properties(plow_interp_sm120_gemma_gemm_bn64 PROPERTIES
        CUDA_ARCHITECTURES "120a" CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

    # DECODE GF8 TWIN (beat12b-ctx-switch): a second decode object at full-attn GQA-fusion 8.
    # Identical decode op family as plow_interp_sm120_gemma; ONLY the hd512 full-attn flash arm is
    # instantiated at GF=8 (2x KV re-read vs GF2's 8x) and its regs/arena grow (234 regs, 16448 B).
    # Symbols suffixed `_gf8` (PLOW_NV_GF8_TWIN) so it co-links with the GF2 decode object; the chat
    # harness picks the launcher per decode step by kvlen (ctx-regime switching, PLOW_GF_SWITCH).
    add_library(plow_interp_sm120_gemma_gf8 STATIC nvidia/interp_sm120.cu)
    target_include_directories(plow_interp_sm120_gemma_gf8 PUBLIC ${PLOW_INCLUDE} nvidia)
    target_compile_definitions(plow_interp_sm120_gemma_gf8 PRIVATE
        PLOW_NV_GEMMA=1 PLOW_NV_GF8_TWIN=1 PLOW_NV_FA_GF=2 PLOW_NV_FA_GF_FULL=8
        ${PLOW_NV_W8A8_DEFS} ${PLOW_FP8_KV_DEFS})
    set_target_properties(plow_interp_sm120_gemma_gf8 PROPERTIES
        CUDA_ARCHITECTURES "120a" CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

    # End-to-end Gemma-4: chunked PREFILL (buckets) then greedy DECODE. Not a ctest (needs a
    # checkpoint + compiled .pkt). PLOW_PREFILL=0 falls back to decode-only (the Phase-0 reference).
    add_executable(gemma4_sm120_chat tests/gemma4_sm120_chat.cu)
    set_target_properties(gemma4_sm120_chat PROPERTIES CUDA_ARCHITECTURES "120a")
    target_include_directories(gemma4_sm120_chat PRIVATE nvidia common ${PLOW_INCLUDE})
    target_link_libraries(gemma4_sm120_chat PRIVATE
        plow_interp_sm120_gemma plow_interp_sm120_gemma_pf plow_interp_sm120_gemma_seg
        plow_interp_sm120_gemma_gemm plow_interp_sm120_gemma_gf8)
endif()

# --- the SERVED sm_120a cubins ------------------------------------------------
#
# `plowrt serve` does not load anything built above: the targets under
# `if(PLOW_CUDA)` are STATIC LIBS for the C test harnesses. The artifacts serve
# actually loads are .cubin files, and they were built by a SECOND, independent
# definition of the same compile — scripts/build_sm120_cubin.sh, which wrote out
# every object's define set longhand. The two drifted, and the drift was the bug:
#
#   * -DPLOW_NV_FA_GF_FULL=4 reached the PLAIN decode object but not its _fp8kv
#     sibling — the object an fp8-KV asset actually loads — so the served object
#     silently fell back to GF_FULL=2 and left a measured 1.48x on the floor;
#   * the fp8-KV PREFILL object hardcoded -DPLOW_NV_FA_PIPE=0, so the
#     PLOW_FP8_KV_FASTPF option could not reach it at all (a measured 5.4x on a
#     127k prefill for anyone who rebuilt).
#
# Both are the same structural fault: four hand-written define sets, no way for a
# flag to be checked against its sibling. Here every object's set is DERIVED from
# ONE base plus two axes (prefill?, fp8-KV?), so a flag physically cannot reach
# one object and miss the other. scripts/build_sm120_cubin.sh is now a thin
# wrapper over this target and keeps its old CLI.
#
# NOT under enable_language(CUDA), deliberately. Under `nix develop`, CPATH
# points nvcc's host pass at nix glibc headers that conflict with the CUDA math
# headers; CMake's CUDA compiler detection fails the same way the compile does.
# cmake/nvcc_cubin.sh runs nvcc under the same clean `env -i` the script always
# used, so this configures with no CUDA language support at all. One maintained
# definition of the build beats CMake purity.
option(PLOW_SM120_CUBIN "Emit the sm_120a .cubin artifacts `plowrt serve` loads" OFF)
# The Hopper twin. Same base + axes as sm_120a; what differs is the TU
# (nvidia/interp_sm90a.cu), the -arch, the mangled symbols, and the per-arch
# define set scripts/build_sm90a_cubin.sh has always applied (dead-arm gating +
# the H100 decode-arm fixes — see _nv_var_sm90a below, which cites the script).
# That script remains the way to get the experimental segmented/lean objects
# (PLOW_BUILD_SEG etc.); the two DEFAULT served objects come from here.
option(PLOW_SM90A_CUBIN "Emit the sm_90a .cubin artifacts `plowrt serve` loads" OFF)
option(PLOW_SM120_CUBIN_FP8KV "Also emit the _fp8kv cubin variants (all enabled arches)" OFF)

# GQA fusion factor for the FULL-attention (hd512) flash-decode instantiation.
# Measured: GF_FULL should equal `gqa` when the full layers have kv_heads == 1
# (perf-data/px11-flash-decode.md) — Gemma-4 is gqa=8, and the historical 4 is
# what the shipped assets were built at. This is ONE variable applied to every
# decode-family object, which is what bug #2 above needed and did not have.
set(PLOW_NV_FA_GF_FULL "4" CACHE STRING "full-attn (hd512) flash-decode GQA fusion: 1|2|4|8")
set(PLOW_CUBIN_NVCC "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "nvcc used for the served cubins")
set(PLOW_CUBIN_ARCH "sm_120a" CACHE STRING "-arch for the served cubins")
set(PLOW_CUBIN_DIR "${CMAKE_BINARY_DIR}/cubin" CACHE PATH "output directory for the served cubins")
# Passed through verbatim after the derived set — the decode tuner's sweep knob
# (scripts/tune_decode_sweep.sh, tuning/README-decode-tuner.md) and the escape
# hatch for a one-off A/B. Semicolon- or space-separated `-D...` tokens.
set(PLOW_EXTRA_DEFINES "" CACHE STRING "extra raw nvcc flags appended to every served cubin")
# Phase D: a generated per-packet `plow_config.h` (devgen::manifest::config_header).
# Set it and every object below compiles only the arms that packet dispatches to, and
# stamps the packet's pairing hash so `plowrt` cannot pair it with a different packet.
# UNSET (the default) => every arm compiles and the objects are byte-identical to what
# they have always been. See the header block in nvidia/interp_sm120.cu.
set(PLOW_CUBIN_CONFIG "" CACHE FILEPATH "generated plow_config.h to specialise the cubins for one packet")

if(PLOW_SM120_CUBIN OR PLOW_SM90A_CUBIN)
    # An fp8-KV build without FASTPF puts the PREFILL objects on the synchronous
    # PIPE=0 staging arm. That is correct, and it is slow: measured 175.90 s vs
    # 32.39 s on a 127k prefill. Say so at configure time rather than let it be
    # rediscovered as a benchmark regression.
    if(PLOW_FP8_KV AND NOT PLOW_FP8_KV_FASTPF)
        message(WARNING
            "PLOW_FP8_KV=ON without PLOW_FP8_KV_FASTPF: the fp8-KV prefill object is built "
            "PLOW_NV_FA_PIPE=0 (synchronous smem staging). Measured 5.4x slower on a 127k "
            "prefill than the FASTPF PIPE=1 fp8-mma arms. Since PX-23 both head dims have a "
            "fast arm (hd256 px23, hd512 px4/px8), so this no longer needs hd512-only packets "
            "-- an ALL-LAYER e4m3 packet is served by the fast path end to end.")
    endif()

    string(REPLACE " " ";" _cubin_extra "${PLOW_EXTRA_DEFINES}")
    set(_cubin_inc -I "${CMAKE_CURRENT_SOURCE_DIR}/common" -I "${CMAKE_CURRENT_SOURCE_DIR}/nvidia")
    set(_cubin_run "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcc_cubin.sh")

    # THE ONE BASE. Every interp object below starts from exactly this.
    set(_cubin_base -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1)
    if(PLOW_NV_W8A8)
        list(APPEND _cubin_base -DPLOW_NV_W8A8=1)
    endif()
    # The generated header rides the BASE, so a specialised build cannot reach one
    # object and miss its sibling — the same property the axes below give the flags.
    if(PLOW_CUBIN_CONFIG)
        get_filename_component(_cfg_dir "${PLOW_CUBIN_CONFIG}" DIRECTORY)
        get_filename_component(_cfg_name "${PLOW_CUBIN_CONFIG}" NAME)
        list(APPEND _cubin_base "-DPLOW_CONFIG=\"${_cfg_name}\"")
        list(APPEND _cubin_inc -I "${_cfg_dir}")
    endif()

    # AXIS 1 — decode vs prefill. Decode carries the full-attn GF instantiation;
    # prefill carries the tiled-GEMM + FLASH_PREFILL arms.
    set(_ax_decode -DPLOW_NV_FA_GF_FULL=${PLOW_NV_FA_GF_FULL})
    set(_ax_prefill -DPLOW_NV_PREFILL=1)
    # AXIS 2 — bf16 KV vs fp8 (e4m3) KV. On the PREFILL side fp8-KV also fixes the
    # staging arm, and it comes from the OPTION, never a literal: FLASH_PREFILL_FP8
    # dequants at the smem stage (cp.async cannot convert fp8 inline), so PIPE=0 is
    # the legacy all-layer arm and PIPE=1 is the px4 fp8-mma arm FASTPF selects.
    set(_ax_fp8kv -DPLOW_FP8_KV=1)
    if(PLOW_FP8_KV_FASTPF)
        set(_ax_fp8kv_pf -DPLOW_FP8_KV=1 -DPLOW_NV_FA_PIPE=1)
    else()
        set(_ax_fp8kv_pf -DPLOW_FP8_KV=1 -DPLOW_NV_FA_PIPE=0)
    endif()

    # ARCH VARIANTS: <stem>|<-arch>|<TU>|<decode symbol>|<prefill symbol>|<per-arch defs var>.
    # Rows are still base + axes; a variant adds only what its arch has always
    # carried. sm_120a adds nothing. sm_90a adds the two define sets
    # scripts/build_sm90a_cubin.sh applies unconditionally:
    #   * dead-arm gating (h100-interp arm-ablation): MLA/MAMBA/DSA are op
    #     families a Gemma model never emits; compiling them out shrinks the
    #     decode cubin ~43% and its stack to 0. The flags default ON in the
    #     source, so the sm_120a objects stay byte-identical.
    #   * the H100 decode-arm fixes (perf-data/gemma26b-h100-gemv-mlp.md):
    #     GEMV row-blocking + warp router, MoE-down lane split, flash
    #     warp-per-row, fp8 RB=4. All default 0 in the sources.
    set(_nv_var_sm120a "")
    set(_nv_var_sm90a
        -DPLOW_NV_MLA=0 -DPLOW_NV_MAMBA=0 -DPLOW_NV_DSA=0
        -DPLOW_NV_GEMV_RB=1 -DPLOW_MOE_DOWN_LANESPLIT=1 -DPLOW_NV_FA_WPR=1 -DPLOW_NV_FP8_RB=4)
    set(_cubin_variants "")
    if(PLOW_SM120_CUBIN)
        list(APPEND _cubin_variants
            "interp_sm120|${PLOW_CUBIN_ARCH}|interp_sm120.cu|_Z12interp_sm12011PlowProgram|_Z15interp_sm120_pf11PlowProgram|_nv_var_sm120a")
    endif()
    if(PLOW_SM90A_CUBIN)
        list(APPEND _cubin_variants
            "interp_sm90a|sm_90a|interp_sm90a.cu|_Z12interp_sm90a11PlowProgram|_Z15interp_sm90a_pf11PlowProgram|_nv_var_sm90a")
    endif()

    set(_cubin_outs "")
    foreach(_variant ${_cubin_variants})
        string(REPLACE "|" ";" _vf "${_variant}")
        list(GET _vf 0 _vstem)
        list(GET _vf 1 _varch)
        list(GET _vf 2 _vtu)
        list(GET _vf 3 _ksym_dec)
        list(GET _vf 4 _ksym_pf)
        list(GET _vf 5 _vdefs_var)
        set(_vsrc "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/${_vtu}")

        # THE TABLE: <stem> <kernel symbol> <axis-1 var> [<axis-2 var>]. Nothing below
        # spells out a define set; every row is base + variant + axes.
        set(_cubin_rows
            "${_vstem}|${_ksym_dec}|_ax_decode|"
            "${_vstem}_pf|${_ksym_pf}|_ax_prefill|")
        if(PLOW_SM120_CUBIN_FP8KV)
            list(APPEND _cubin_rows
                "${_vstem}_fp8kv|${_ksym_dec}|_ax_decode|_ax_fp8kv"
                "${_vstem}_pf_fp8kv|${_ksym_pf}|_ax_prefill|_ax_fp8kv_pf")
        endif()

        foreach(_row ${_cubin_rows})
            string(REPLACE "|" ";" _f "${_row}")
            list(GET _f 0 _stem)
            list(GET _f 1 _sym)
            list(GET _f 2 _a1)
            list(GET _f 3 _a2)
            set(_defs ${_cubin_base} ${${_vdefs_var}} ${${_a1}})
            if(_a2)
                list(APPEND _defs ${${_a2}})
            endif()
            set(_out "${PLOW_CUBIN_DIR}/${_stem}.cubin")
            add_custom_command(OUTPUT "${_out}"
                COMMAND bash "${_cubin_run}" "${PLOW_CUBIN_NVCC}" "${_out}" "${_sym}" required
                        -arch=${_varch} -O3 -cubin ${_cubin_inc}
                        ${_defs} ${_cubin_extra} "${_vsrc}"
                DEPENDS "${_vsrc}" "${_cubin_run}"
                COMMENT "nvcc ${_stem}.cubin (${_defs})"
                VERBATIM)
            list(APPEND _cubin_outs "${_out}")
        endforeach()
    endforeach()

    # The device stochastic sampler is NOT the megakernel and shares none of its
    # defines, so it is not a row in the table above. A failure here is non-fatal:
    # serving falls back to host sampling (hence `optional`).
    set(_smp_src "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/sample_sm120.cu")
    if(PLOW_SM120_CUBIN AND EXISTS "${_smp_src}")
        set(_smp_out "${PLOW_CUBIN_DIR}/sample_sm120.cubin")
        add_custom_command(OUTPUT "${_smp_out}"
            COMMAND bash "${_cubin_run}" "${PLOW_CUBIN_NVCC}" "${_smp_out}" plow_sample optional
                    -arch=${PLOW_CUBIN_ARCH} -O3 -cubin "${_smp_src}"
            DEPENDS "${_smp_src}" "${_cubin_run}"
            COMMENT "nvcc sample_sm120.cubin"
            VERBATIM)
        list(APPEND _cubin_outs "${_smp_out}")
    endif()

    # `nv_cubins` builds every enabled arch variant; `sm120_cubins` keeps the
    # name scripts/build_sm120_cubin.sh has always driven.
    add_custom_target(nv_cubins ALL DEPENDS ${_cubin_outs})
    add_custom_target(sm120_cubins DEPENDS ${_cubin_outs})
endif()

# --- the SERVED gfx950 code objects (.hsaco) ----------------------------------
#
# The AMD mirror of the cubin section above, and it exists for the same reason.
# `if(PLOW_ROCM)` below builds the LEGACY plugin (amd/{mfma,flash,row,golden}.hip
# — dead code, see its own note); it does NOT build the production megakernel.
# The objects the AMD path actually loads (runtime/tests/gemma4_chat.c dlopens
# them through ROCr) were defined ONLY by scripts/build_gfx950.sh, so the AMD
# side had exactly the second-independent-definition problem the NVIDIA side was
# just consolidated out of. Both toolchains now emit their served artifacts from
# one cmake build:
#
#   cmake -S runtime -B build -DPLOW_SM120_CUBIN=ON -DPLOW_GFX950_HSACO=ON
#
# Naming: the artifacts keep the `.elf` extension the script has always written,
# because gemma4_chat.c opens them by literal filename (`interp_prefill.elf`,
# `interp_decode_gq.elf`, ...). They are hsaco code objects; only the suffix is
# historical.
#
# NOT ported from the script: the host `chat` harness (a gcc link, not a kernel
# artifact) and the scripts/asm_audit.py instruction-selection pass, which wants
# the whole object set at once. build_gfx950.sh remains the way to get those.
option(PLOW_GFX950_HSACO "Emit the gfx950 .hsaco code objects the AMD path loads" OFF)
# The variant axes. Unlike build_gfx950.sh (where each is an opt-in env var, so
# the common case builds the bf16 pair only), these default ON: the point of the
# cmake path is that ONE configure emits every object, so a packet cannot reach
# for a variant the build quietly skipped.
option(PLOW_HSACO_GQ "Also emit the global-queue (_gq) twins" ON)
option(PLOW_HSACO_FP8 "Also emit the fp8 w8a16 decode + w8a8 prefill objects" ON)
# DECODE + PREFILL + the class-4 flash twin, not just decode — the description said "decode
# object" long after the other two rows landed, which is the same doc-vs-table drift the block
# below exists to prevent. What is still NOT composed here is MLA x fp8-KV: every MLA row pairs
# with fp8(weight)/moe/a4w4/k3 and none with this axis, because there is no `FlashMlaPrefillFp8`
# to put in one yet. `FlashMlaDecodeFp8` will need no row of its own — the MLA DECODE arm is
# unconditional, so it already rides `interp_decode_fp8kv`.
option(PLOW_HSACO_FP8KV "Also emit the fp8 e4m3 KV-cache objects (decode + prefill + flash)" ON)
option(PLOW_HSACO_MXFP4 "Also emit the MX-FP4 (e2m1 + E8M0) decode object" ON)
option(PLOW_HSACO_MLA "Also emit the MLA prefill object (Kimi/DeepSeek/GLM)" ON)
# WAS THE ONE AXIS THAT DEFAULTED OFF. It now follows the "one configure emits every object" rule
# with the rest, because the argument for the exception was WRONG — not out of date, wrong when it
# was written. It read: "defaulting this ON would put seven unreachable arms in the instruction
# stream of every object those models DO load", citing the +32% decode regression that the mere
# PRESENCE of the GF=8 flash arm cost (runtime/amd/op_attention.h:1282).
#
# This option does not do that and never did. Every `PLOW_HSACO_*` here is `list(APPEND _hs_rows)`
# — it adds OBJECTS, it does not add defines to the rows already in the list. `interp_decode` and
# `interp_prefill` carry no `-DPLOW_K3` at any setting of this flag, so a Gemma-4, Qwen or GLM-5.2
# object is BYTE-IDENTICAL either way. The op_attention precedent is about an arm entering a
# SHARED object; the K3 arms only ever enter `interp_*_k3`, which only a K3 packet loads.
#
# THE MEASUREMENT THE ROWS BELOW SAY THIS AXIS OWED, taken on ROCm 7.2.4/clang-22, gfx950.
# `interp_prefill_k3_moe*` are the rows this branch adds for the K3 MoE prefill:
#
#     object                        bytes    VGPR  occ  spill
#     interp_decode                287,328    248   2     0
#     interp_decode_k3             330,008    248   2     0     <- +14.9% size, +0 registers
#     interp_prefill               216,752    256   2     2
#     interp_prefill_mla           278,000    256   2     2
#     interp_prefill_k3            320,168    256   2     2
#     interp_prefill_k3_moe        385,032    256   2     2
#     interp_prefill_k3_moe_a4w4   524,296    256   2     6
#
# The nine arms cost ZERO registers and zero occupancy — decode holds 248/256 at 0 spill with and
# without them — so the register-cliff half of the question is settled. The size half is real
# (+14.9% on decode) and is exactly why these stay SEPARATE OBJECTS rather than becoming defines
# on the rows above. Cost of the default: ~31 s of build, doubled by the `_gq` twins, and ~3 MB.
option(PLOW_HSACO_K3 "Also emit the Kimi-K3 / KDA objects (decode + prefill)" ON)
option(PLOW_K3_DECODE_MXFP4_PROJ
       "Include standalone MXFP4 projection ops in K3 decode objects" ON)
option(PLOW_K3_DECODE_GROUPED
       "Include grouped A4W4 expert ops in a K3 B=1 decode object" OFF)
option(PLOW_MXFP4_DECODE_NT
       "Use non-temporal loads for one-use decode MXFP4 projection weights" ON)
option(PLOW_MOE_PF_A4W4_WEIGHT_NT
       "Use non-temporal loads for grouped A4W4 expert weights" OFF)
option(PLOW_K3_DECODE_TILE_BINSEARCH
       "Use binary expert-tile lookup in batched K3 decode" ON)
option(PLOW_K3_DECODE_ALIGN_PAR_PREFIX
       "Use a workgroup-parallel expert prefix in batched K3 decode" ON)
option(PLOW_KDA_PF_STATE_RESIDENT
       "Keep K3 TP8 prefill KDA state in registers across serial tokens" OFF)
option(PLOW_MLA_PF_SV
       "Use the V2 MLA-prefill LDS swizzle and fragment staging" ON)

set(PLOW_HSACO_ARCH "gfx950" CACHE STRING "offload arch for the served code objects")
set(PLOW_HSACO_HIPCC "/opt/rocm/bin/hipcc" CACHE FILEPATH "hipcc used for the served code objects")
# Discovered from the INSTALLED ROCm rather than pinned: this path was pinned to
# 7.0.2 once and the build died on the first box that had 7.2.4.
set(PLOW_HSACO_BUNDLER "/opt/rocm/lib/llvm/bin/clang-offload-bundler"
    CACHE FILEPATH "clang-offload-bundler that unbundles the code object")
set(PLOW_HSACO_DIR "${CMAKE_BINARY_DIR}/hsaco" CACHE PATH "output directory for the code objects")
# Raising this starves parallelism (measured 6-8x slower); it is a knob, not a tuning target.
set(PLOW_GQ_BATCH "1" CACHE STRING "global-queue claim batch")
# DECODE BATCH BUCKET. Must match the PLOW_DECODE_BATCH devgen emitted the program with:
# devgen sizes the decode program, the KV cache and the argmax for B sequences, and this
# sizes the GEMV that has to produce B logits ROWS. Left unset the kernels compiled at
# op_gemm.h's PLOW_GEMV_MM default of 1 and wrote row 0 only, so every sequence but the
# first sampled token 0 off an all-zero logits row -- with a correct B-wide program around
# it. See scripts/build_gfx950.sh for the full note.
set(PLOW_DECODE_BATCH "1" CACHE STRING "decode batch bucket (-> PLOW_GEMV_MM)")
option(PLOW_GEMV_WALK "Walk a <=16-row GEMV object across wider decode batches" OFF)
set(PLOW_HSACO_EXTRA_DEFINES "" CACHE STRING "extra raw hipcc flags appended to every code object")

if(PLOW_GFX950_HSACO)
    if(NOT PLOW_DECODE_BATCH MATCHES "^[0-9]+$")
        message(FATAL_ERROR "PLOW_DECODE_BATCH must be an integer in 1..32")
    endif()
    if(PLOW_DECODE_BATCH LESS 1 OR PLOW_DECODE_BATCH GREATER 32)
        message(FATAL_ERROR "PLOW_DECODE_BATCH must be in 1..32")
    endif()
    if(PLOW_DECODE_BATCH GREATER 16 AND NOT PLOW_GEMV_WALK)
        message(FATAL_ERROR "PLOW_DECODE_BATCH above 16 requires PLOW_GEMV_WALK=ON")
    endif()
    string(REPLACE " " ";" _hs_extra "${PLOW_HSACO_EXTRA_DEFINES}")
    set(_hs_src "${CMAKE_CURRENT_SOURCE_DIR}/amd/interp.hip")
    set(_hs_tk_src "${CMAKE_CURRENT_SOURCE_DIR}/amd/test_kernels.hip")
    set(_hs_inc -I "${CMAKE_CURRENT_SOURCE_DIR}/amd" -I "${CMAKE_CURRENT_SOURCE_DIR}/common")
    set(_hs_run "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipcc_hsaco.sh")
    # The op bodies live entirely in headers, so without these a kernel edit
    # leaves every .elf stale — the exact failure every guard in
    # build_gfx950.sh was written to prevent.
    file(GLOB _hs_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/amd/*.h"
                       "${CMAKE_CURRENT_SOURCE_DIR}/common/*.h")

    # THE AXES. No row below spells out a define set; every row is axes only.
    set(_hs_ax_none "")
    set(_hs_ax_prefill -DPLOW_BUCKET_DECODE=0)
    # next_pow2(PLOW_DECODE_BATCH), clamped to PLOW_GEMV_MAXM (16). One instantiation with a
    # runtime M and an `m < M` predicate serves every M <= MM, so the bucket is a ceiling, not
    # an exact width. Do NOT make this a runtime ladder -- that was built, measured at 276
    # registers, and removed (op_gemm.h:2163). Measured cost, ROCm 7.2.4/gfx950 decode object:
    # MM=1 248/occ2/spill 0, MM=2 248/occ2/spill 0, MM=4 252/occ2/spill 0, MM=8 256/occ2/spill 19.
    set(_hs_gvmm 1)
    while(_hs_gvmm LESS PLOW_DECODE_BATCH AND _hs_gvmm LESS 16)
        math(EXPR _hs_gvmm "${_hs_gvmm} * 2")
    endwhile()
    set(_hs_ax_decode -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=${_hs_gvmm})
    if(PLOW_GEMV_WALK)
        list(APPEND _hs_ax_decode -DPLOW_GEMV_WALK=1)
    endif()
    # 4 waves => 1 wave/SIMD => a 512-reg budget, and ITS spill is intentional
    # (the Q-hoist spills Q to scratch, cheaper than the L2 re-read it replaces).
    set(_hs_ax_flash -DPLOW_BUCKET_DECODE=0 -DPLOW_BUCKET_FLASH -DPLOW_WG_WAVES=4
                     -DFA_DC=256 -DFA_DBUF=1)
    if(PLOW_HSACO_ARCH STREQUAL "gfx942")
        list(APPEND _hs_ax_flash -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_L2_PLACE_DISPATCH=1)
        if(PLOW_MLA_PF_SV)
            list(APPEND _hs_ax_flash -DPLOW_MLA_PF_SV=1)
        endif()
    endif()
    set(_hs_ax_gq -DPLOW_GLOBAL_QUEUE=1 -DPLOW_GQ_BATCH=${PLOW_GQ_BATCH})
    # PRECISION AXES, ONE PER INDEPENDENT CHOICE. They were not independent: _hs_ax_fp8kv used
    # to expand to `-DPLOW_FP8=1 -DPLOW_FP8_KV=1`, so the KV-encoding axis silently dragged the
    # WEIGHT-encoding axis with it and a bf16-weight/fp8-KV packet — a combination the NVIDIA
    # side has always been able to build (PLOW_FP8_KV is independent of PLOW_NV_W8A8 there) —
    # had no gfx950 object at all. It would have been handed interp_decode_fp8kv, whose fp8
    # arms REPLACED the bf16 GEMV_GLU/GEMV_QKV it needs. Now the axes compose instead.
    set(_hs_ax_fp8 -DPLOW_FP8=1)          # WEIGHT encoding: fp8 e4m3 (prefill w8a8, decode w8a16)
    set(_hs_ax_fp8kv -DPLOW_FP8_KV=1)     # KV encoding: fp8 e4m3 cache + per-row scales
    if(PLOW_HSACO_ARCH STREQUAL "gfx942")
        set(_hs_ax_fp8kv_k3_flash ${_hs_ax_fp8kv} -DPLOW_K3=1)
    else()
        set(_hs_ax_fp8kv_k3_flash ${_hs_ax_fp8kv})
    endif()
    set(_hs_ax_fp8_fp8kv ${_hs_ax_fp8} ${_hs_ax_fp8kv})   # both, the shipped fp8-KV profile
    set(_hs_ax_mxfp4 -DPLOW_MXFP4=1)      # WEIGHT encoding: MXFP4 e2m1+E8M0 (w4a16 projections)
    if(NOT PLOW_MXFP4_DECODE_NT)
        list(APPEND _hs_ax_mxfp4 -DPLOW_MXFP4_DEC_NT=0)
    endif()
    set(_hs_ax_mla -DPLOW_MLA_PREFILL=1)
    # Composed, not spelled out: an all-fp8 Kimi K2.7 / GLM-5.2 / DeepSeek prefill
    # needs the MLA attention arms AND the w8a8 GEMM arms in the same object, and a
    # showcase run must be all-fp8 or all-bf16 — never one bucket of each.
    set(_hs_ax_mla_fp8 ${_hs_ax_mla} ${_hs_ax_fp8})
    # The FFN half of an MLA-family prefill. An MLA prefill object with no MoE arm is
    # attention-complete and FFN-incomplete — the expert packets fall through to `default:`
    # and write nothing — so these are the rows that actually make a Kimi/GLM/DeepSeek
    # prompt runnable. Measured: the grouped-expert GEMM costs occ NOTHING (see the rows).
    set(_hs_ax_mla_moe ${_hs_ax_mla} -DPLOW_MOE_PREFILL=1)
    set(_hs_ax_mla_moe_fp8 ${_hs_ax_mla_fp8} -DPLOW_MOE_PREFILL=1)
    # A4W4: MXFP4 on BOTH operands through the matrix core, the encoding ATOM runs. Compile-time
    # like every other precision axis -- a runtime encoding switch would make the register
    # allocator budget for the worst body and then every body spills (interp.hip states the trap).
    set(_hs_ax_a4w4_weight "")
    if(PLOW_MOE_PF_A4W4_WEIGHT_NT)
        set(_hs_ax_a4w4_weight -DPLOW_MOE_PF_A4W4_WEIGHT_NT=1)
    endif()
    set(_hs_ax_mla_moe_a4w4
        ${_hs_ax_mla_moe} -DPLOW_MOE_PF_A4W4=1 ${_hs_ax_a4w4_weight})
    # THE ALL-MXFP4 MLA PREFILL OBJECT. A Kimi/GLM/DeepSeek all-fp4 packet needs the MXFP4
    # PROJECTIONS (GemmMxfp4, op 93 — gated by PLOW_MXFP4) and the A4W4 grouped EXPERTS (ops
    # 85/86) in the SAME object; they were previously on separate rows, so no single object
    # could run the packet even though every arm existed.
    set(_hs_ax_mla_moe_a4w4_full ${_hs_ax_mla_moe_a4w4} -DPLOW_MXFP4=1)
    # KIMI-K3 / KDA. One axis, both buckets: a K3 layer runs the same graph at T=1 and T>1, and an
    # AttnRes present only in decode would make prefill and decode compute different models. The
    # MLA prefill axis comes with it because K3's full-attention layers are MLA.
    # GV_UNROLL=14 on the K3 rows ONLY. The global default is 11 because that is `nchunk` for
    # Gemma-31B's K=5376; K3's dominant decode GEMV is K=7168, whose nchunk is exactly 14, so 11
    # issues two groups with 8 dead slots and starves the in-flight-work the op_gemm.h sweep names
    # as the sub-ceiling cause. MEASURED on the real TP8 asset, ctx 8000, interleaved repeats:
    # 39.543 ms/token at 11 vs 39.120 at 14 (-0.42 ms, -1.07%), within-arm spread 0.03 ms. SWEPT, so
    # 14 is a measured optimum and not a guess from nchunk: UN 7 -> 39.334, 11 -> 39.543,
    # 14 -> 39.135, 16 -> 39.256 ms/token. The minimum sits exactly at nchunk, which is what the
    # in-flight-work explanation predicts, and every arm reads VGPR=248 / occ=2 / spill=0. Bit-exact
    # (unroll changes only how many loads issue before they are consumed, never the accumulation
    # order) and register-free: interp_decode_fp8kv_k3 stays VGPR=248 / occ=2 / spill=0, because
    # REPLACING the unroll does not grow the register union the way adding a second arm does.
    set(_hs_ax_k3 -DPLOW_K3=1 -DGV_UNROLL=14)
    set(_hs_ax_mla_k3 ${_hs_ax_mla} -DPLOW_K3=1)
    # K3's optional compressed-latent KV uses the same model arms with the fp8-KV swap. These are
    # separate objects: PLOW_FP8_KV removes the bf16 FlashMLA arm, so composing it into the plain
    # K3 row would make bf16 packets silently miss their dispatch case.
    set(_hs_ax_k3_fp8kv ${_hs_ax_k3} ${_hs_ax_fp8kv})
    set(_hs_ax_mla_k3_fp8kv ${_hs_ax_mla_k3} ${_hs_ax_fp8kv})
    # A BATCHED DECODE OBJECT ALSO NEEDS THE GROUPED EXPERT ARMS. At PLOW_DECODE_BATCH > 1 the
    # decode program carries B independent sequences, and devgen emits the SAME grouped chain a
    # prefill bucket uses (ops 83-87) — correctly, because MoE has no cross-row state and grouping
    # by expert is what makes an expert's weights cross HBM once for every row that chose it,
    # which is the entire memory-bound win of batching.
    #
    # Those five packets are behind PLOW_MOE_PREFILL. Without it they fall through the dispatch
    # `default:` and write NOTHING — silently, on all 92 MoE layers, leaving output that is finite
    # and plausible and wrong. This is the same trap the comment below spells out for
    # `interp_prefill_k3`, one object over.
    #
    # Gated on the batch so a B=1 object is byte-identical: at one row grouping buys nothing, and
    # adding arms grows the register union (interp_decode_fp8kv_k3 is VGPR=248 / occ=2 / spill=0
    # and that headroom is not spare).
    set(_hs_ax_k3_dec ${_hs_ax_k3})
    set(_hs_ax_k3_dec_fp8kv ${_hs_ax_k3_fp8kv})
    if(PLOW_DECODE_BATCH GREATER 1 OR PLOW_K3_DECODE_GROUPED)
        set(_hs_ax_k3_tile_search "")
        if(PLOW_K3_DECODE_TILE_BINSEARCH)
            set(_hs_ax_k3_tile_search -DPLOW_MOE_TILE_BINSEARCH=1)
        endif()
        set(_hs_ax_k3_align_prefix "")
        if(PLOW_K3_DECODE_ALIGN_PAR_PREFIX)
            set(_hs_ax_k3_align_prefix -DPLOW_MOE_ALIGN_PAR_PREFIX=1)
        endif()
        set(_hs_ax_k3_dec_mxfp4 "")
        if(PLOW_K3_DECODE_MXFP4_PROJ)
            set(_hs_ax_k3_dec_mxfp4 ${_hs_ax_mxfp4})
        endif()
        set(_hs_ax_k3_dec ${_hs_ax_k3_dec} -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_PF_A4W4=1
            ${_hs_ax_a4w4_weight} ${_hs_ax_k3_tile_search} ${_hs_ax_k3_align_prefix}
            ${_hs_ax_k3_dec_mxfp4})
        set(_hs_ax_k3_dec_fp8kv
            ${_hs_ax_k3_dec_fp8kv} -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_PF_A4W4=1
            ${_hs_ax_a4w4_weight} ${_hs_ax_k3_tile_search} ${_hs_ax_k3_align_prefix}
            ${_hs_ax_k3_dec_mxfp4})
    endif()
    # ...and the FFN half. A K3 prefill bucket runs the GROUPED expert chain (ops 83-87), not the
    # per-slot decode ops: at T rows an expert's weights cross HBM once for every token that chose
    # it. Without PLOW_MOE_PREFILL those five packets fall through the dispatch `default:` and
    # write NOTHING, which on this interpreter is silent. So `interp_prefill_k3` above is an
    # ATTENTION-ONLY object and a whole-layer K3 prompt needs one of these two.
    set(_hs_ax_k3_moe ${_hs_ax_mla_k3} -DPLOW_MOE_PREFILL=1)
    set(_hs_ax_k3_moe_fp8kv ${_hs_ax_mla_k3_fp8kv} -DPLOW_MOE_PREFILL=1)
    set(_hs_ax_k3_pf_state "")
    if(PLOW_KDA_PF_STATE_RESIDENT)
        set(_hs_ax_k3_pf_state -DPLOW_KDA_PF_STATE_RESIDENT=1)
    endif()
    # THE ROW A K3 PACKET ACTUALLY LOADS. K3 ships MXFP4 experts, and `enc = PLOW_MOE_ENC_MXFP4`
    # on ops 85/86 selects the A4W4 body — which lives behind PLOW_MOE_PF_A4W4. Without it the arm
    # takes `moe_pf_refuse` (loud, by design) rather than computing anything. PLOW_MXFP4 rides
    # along for the same reason `_mla_moe_a4w4_full` carries it: an all-fp4 packet needs the fp4
    # PROJECTIONS (op 93) in the same object as the fp4 experts, and they were on separate rows.
    set(_hs_ax_k3_moe_a4w4
        ${_hs_ax_k3_moe} -DPLOW_MOE_PF_A4W4=1 ${_hs_ax_a4w4_weight}
        ${_hs_ax_k3_pf_state} -DPLOW_MXFP4=1)
    set(_hs_ax_k3_moe_a4w4_fp8kv
        ${_hs_ax_k3_moe_fp8kv} -DPLOW_MOE_PF_A4W4=1 ${_hs_ax_a4w4_weight}
        ${_hs_ax_k3_pf_state} -DPLOW_MXFP4=1)

    # Symbols carry the arch: the loader builds the kernel name from the LIVE
    # HSA agent name (exec/amd.rs symbol_name), so a gfx942 object must export
    # plow_interp*_gfx942. -DPLOW_ARCH_SUFFIX is what renames them; interp.hip
    # defaults it to gfx950, so passing it explicitly keeps the gfx950 objects
    # byte-identical. The CDNA3 codegen divergences (64 KiB LDS tile, single
    # buffering, MFMA halving) are NOT flags here: they are defaults in
    # op_gemm.h / amd_arch.h keyed on PLOW_CDNA4, exactly so a build that
    # forgets them still gets an arena that fits (see scripts/build_gfx942.sh).
    set(_hs_arch_def -DPLOW_ARCH_SUFFIX=${PLOW_HSACO_ARCH})
    set(_hs_kpf "plow_interp_${PLOW_HSACO_ARCH}")
    set(_hs_kdec "plow_interp_dec_${PLOW_HSACO_ARCH}")
    set(_hs_kfl "plow_interp_flash_${PLOW_HSACO_ARCH}")

    # THE TABLE: <stem> <kernel symbol> <axis-1> <axis-2> <max total regs> <min occ>.
    # The 8-wave interpreters must stay <= 256 total (2 waves/SIMD); the 4-wave
    # flash object <= 512 (1 wave/SIMD). Rows describe the STATIC object; the
    # global-queue twin is generated from the same row below, because it differs
    # only by the scheduling loop (that is what makes an A/B on it meaningful).
    set(_hs_rows
        "interp_prefill|${_hs_kpf}|_hs_ax_prefill|_hs_ax_none|256|2"
        "interp_decode|${_hs_kdec}|_hs_ax_decode|_hs_ax_none|256|2"
        "interp_flash|${_hs_kfl}|_hs_ax_flash|_hs_ax_none|512|1")
    if(PLOW_HSACO_FP8)
        # MIXED-PRECISION OBJECTS, and the names do not say so. `interp_*_fp8` carries the fp8
        # arms IN ADDITION TO the bf16 ones, and the fp8 arms themselves are not one precision:
        # the prefill GEMMs are w8a8 (both operands e4m3 + QUANT_FP8) while the decode GEMVs are
        # w8a16 (bf16 activation, dequant on load). Both deliberate, neither visible in the stem.
        #
        # They used to SWAP for the bf16 arms. That dropped the one bf16 GEMM an fp8 prefill
        # program still emits — the lm_head, `pick_tile(1, vocab, hidden)`, unquantized unless
        # PLOW_FP8_HEAD=1 — so interp_prefill_fp8 had no `case PLOW_DOP_GEMM_SMALL`, the lm_head
        # hit `default:`, and every fp8 run sampled token id 0 from all-zero logits. Additive
        # costs NOTHING here: measured 2026-07-27 on ROCm 7.2.4, 256 VGPR / 0 AGPR / occ 2 /
        # spill 20 with and without the bf16 arms (the swap's register argument was measured on
        # 7.0.2/clang-20, where these objects landed at 262/occ-1).
        list(APPEND _hs_rows
            "interp_decode_fp8|${_hs_kdec}|_hs_ax_decode|_hs_ax_fp8|256|2"
            "interp_prefill_fp8|${_hs_kpf}|_hs_ax_prefill|_hs_ax_fp8|256|2")
    endif()
    if(PLOW_HSACO_FP8KV)
        # BOTH buckets. Only the decode object was ever built, so an fp8-KV run had
        # no prefill object at all: PLOW_DOP_FLASH_PREFILL_FP8 (39) is implemented
        # for hd 256 AND 512 and compiled into ZERO of the shipped objects, and the
        # fp8-weight prefill object (`_fp8`, PLOW_FP8 only) carries the BF16 flash.
        # A packet must not be able to ask for an arm the build quietly skipped —
        # that is the whole reason this table exists. Measured: 256 VGPR / occ 2.
        # `_fp8_fp8kv`, not `_fp8kv`: these are the fp8-WEIGHT + fp8-KV objects, which is what
        # every fp8-KV packet emitted today actually is. The stems keep their historical names
        # because gemma4_chat.c and plowrt resolve them by literal filename.
        list(APPEND _hs_rows
            "interp_decode_fp8kv|${_hs_kdec}|_hs_ax_decode|_hs_ax_fp8_fp8kv|256|2"
            "interp_prefill_fp8kv|${_hs_kpf}|_hs_ax_prefill|_hs_ax_fp8_fp8kv|256|2"
            # ...and the class-4 twin. Gemma's flash prefill runs on the 4-wave object
            # via segmented dispatch, so an fp8-KV prefill without this row had to run
            # its hd512 flash on the 8-wave interpreter — correct, but without the
            # 512-register budget the Q-hoist needs. Same cliff as the bf16 twin
            # (measured 256+256 = 512 / occ 1 / spill 228, the intentional Q-hoist).
            "interp_flash_fp8kv|${_hs_kfl}|_hs_ax_flash|_hs_ax_fp8kv_k3_flash|512|1")
    endif()
    if(PLOW_HSACO_MXFP4)
        # Same asymmetry: PLOW_DOP_GEMM_MXFP4 (93) is the w4a16 PREFILL twin of the
        # two decode mxfp4 arms and had no object to live in. (Its dispatch arm was
        # also mis-nested into the decode branch — see interp.hip; fixed with this.)
        # Measured after the fix: 256 VGPR / occ 2, same as the bf16 prefill.
        list(APPEND _hs_rows
            "interp_decode_mxfp4|${_hs_kdec}|_hs_ax_decode|_hs_ax_mxfp4|256|2"
            "interp_prefill_mxfp4|${_hs_kpf}|_hs_ax_prefill|_hs_ax_mxfp4|256|2")
    endif()
    if(PLOW_HSACO_MLA)
        list(APPEND _hs_rows
            "interp_prefill_mla|${_hs_kpf}|_hs_ax_prefill|_hs_ax_mla|256|2")
        if(PLOW_HSACO_FP8)
            # Kimi K2.7 / GLM-5.2 / DeepSeek ship fp8, so the bf16-GEMM MLA prefill
            # object above cannot serve them without mixing dtypes across buckets.
            # Measured: 256 VGPR / 0 AGPR / occ 2 — the MLA prefill arms and the w8a8
            # GEMM arms coexist inside the cliff because the fp8 GEMMs SWAP for the
            # bf16 ones rather than adding to them.
            list(APPEND _hs_rows
                "interp_prefill_mla_fp8|${_hs_kpf}|_hs_ax_prefill|_hs_ax_mla_fp8|256|2")
        endif()
        # ...and the same two objects WITH the grouped-expert FFN. Separate rows rather than
        # folding PLOW_MOE_PREFILL into the rows above: the MLA-only objects are what agent B's
        # emitter work is currently building against, and a dense-FFN MLA model (there are none
        # today, but the axis is real) should not pay for the grouped GEMM.
        # Measured, prefill_mla: 256 VGPR / 0 AGPR / occ 2, spill 1 -> 2 with the MoE arms.
        # Measured, prefill_mla_fp8: 256 / 0 / occ 2, spill 20 -> 20 (unchanged).
        # The grouped GEMM's 81920 B double-buffered tile fits INSIDE the interpreter's existing
        # 147464 B arena, so it does not move the LDS high-water mark either.
        list(APPEND _hs_rows
            "interp_prefill_mla_moe|${_hs_kpf}|_hs_ax_prefill|_hs_ax_mla_moe|256|2")
        if(PLOW_HSACO_FP8)
            list(APPEND _hs_rows
                "interp_prefill_mla_moe_fp8|${_hs_kpf}|_hs_ax_prefill|_hs_ax_mla_moe_fp8|256|2")
        endif()
        if(PLOW_HSACO_MXFP4)
            # Measured: 256 VGPR / 0 AGPR / occ 2 / spill 2 -- IDENTICAL to the w8a16 twin, because
            # the hardware applies both E8M0 scales and the software promotion accumulator the
            # block-fp8 body needs disappears. A4W4 is CHEAPER in registers, not dearer.
            list(APPEND _hs_rows
                "interp_prefill_mla_moe_a4w4|${_hs_kpf}|_hs_ax_prefill|_hs_ax_mla_moe_a4w4|256|2"
                # ...and the object an all-MXFP4 packet actually loads: projections AND experts.
                "interp_prefill_mla_moe_a4w4_full|${_hs_kpf}|_hs_ax_prefill|_hs_ax_mla_moe_a4w4_full|256|2")
        endif()
    endif()
    # KIMI-K3 / KDA objects. Both buckets, for the reason the axis note gives. The register
    # budgets are the ordinary 256/occ-2 the other decode and prefill rows use, and the arms fit
    # inside them with room: the table at the option above records 248/occ-2/spill-0 for decode,
    # IDENTICAL to the object without them. The gate stays because a gate that only fires on
    # regression is doing its job — it turns HSA_STATUS_ERROR_INVALID_ISA at launch into a build
    # error — and because op_attention.h:1282 is why "it fits" was never the whole question.
    if(PLOW_HSACO_K3)
        list(APPEND _hs_rows
            "interp_decode_k3|${_hs_kdec}|_hs_ax_decode|_hs_ax_k3_dec|256|2"
            "interp_prefill_k3|${_hs_kpf}|_hs_ax_prefill|_hs_ax_mla_k3|256|2"
            "interp_prefill_k3_moe|${_hs_kpf}|_hs_ax_prefill|_hs_ax_k3_moe|256|2"
            "interp_prefill_k3_moe_a4w4|${_hs_kpf}|_hs_ax_prefill|_hs_ax_k3_moe_a4w4|256|2")
        if(PLOW_HSACO_FP8KV)
            # Variant precedes the model arm in object_name(): interp_*_fp8kv_k3{,_moe...}.
            list(APPEND _hs_rows
                "interp_decode_fp8kv_k3|${_hs_kdec}|_hs_ax_decode|_hs_ax_k3_dec_fp8kv|256|2"
                "interp_prefill_fp8kv_k3|${_hs_kpf}|_hs_ax_prefill|_hs_ax_mla_k3_fp8kv|256|2"
                "interp_prefill_fp8kv_k3_moe|${_hs_kpf}|_hs_ax_prefill|_hs_ax_k3_moe_fp8kv|256|2"
                "interp_prefill_fp8kv_k3_moe_a4w4|${_hs_kpf}|_hs_ax_prefill|_hs_ax_k3_moe_a4w4_fp8kv|256|2")
        endif()
    endif()

    set(_hs_kinds static)
    if(PLOW_HSACO_GQ)
        list(APPEND _hs_kinds gq)
    endif()

    set(_hs_outs "")
    foreach(_row ${_hs_rows})
        string(REPLACE "|" ";" _f "${_row}")
        list(GET _f 0 _stem)
        list(GET _f 1 _sym)
        list(GET _f 2 _a1)
        list(GET _f 3 _a2)
        list(GET _f 4 _maxreg)
        list(GET _f 5 _minocc)
        foreach(_kind ${_hs_kinds})
            set(_sfx "")
            set(_gq "")
            if(_kind STREQUAL "gq")
                # Both the filename and the kernel symbol take the suffix, and
                # gemma4_chat.c resolves both by name.
                set(_sfx "_gq")
                set(_gq ${_hs_ax_gq})
            endif()
            set(_out "${PLOW_HSACO_DIR}/${_stem}${_sfx}.elf")
            add_custom_command(OUTPUT "${_out}"
                COMMAND bash "${_hs_run}" "${PLOW_HSACO_HIPCC}" "${PLOW_HSACO_BUNDLER}"
                        "${PLOW_HSACO_ARCH}" "${_out}" "${_sym}${_sfx}" ${_maxreg} ${_minocc}
                        ${_hs_inc} ${_hs_arch_def} ${${_a1}} ${${_a2}} ${_gq} ${_hs_extra} "${_hs_src}"
                DEPENDS "${_hs_src}" "${_hs_run}" ${_hs_hdrs}
                COMMENT "hipcc ${_stem}${_sfx}.elf"
                VERBATIM)
            list(APPEND _hs_outs "${_out}")
        endforeach()
    endforeach()

    # Golden __device__ wrappers. They call the SAME op_*.h bodies the
    # interpreter runs, so they are rebuilt WITH it — building the interpreter
    # and not these is how `t_attn` once passed three times against a stale
    # kernel. Not a table row: many entry points, no single symbol to gate, and
    # no register cliff (nothing dispatches them from the interpreter).
    set(_hs_tk_out "${PLOW_HSACO_DIR}/test_kernels.elf")
    add_custom_command(OUTPUT "${_hs_tk_out}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${PLOW_HSACO_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E rm -f "${_hs_tk_out}" "${_hs_tk_out}.co"
        COMMAND "${PLOW_HSACO_HIPCC}" --offload-arch=${PLOW_HSACO_ARCH} -O3 -w --genco
                ${_hs_inc} "${_hs_tk_src}" -o "${_hs_tk_out}.co"
        # No quotes around the paths: VERBATIM already quotes each argument, and
        # a second pair ends up as literal characters in the filename.
        COMMAND "${PLOW_HSACO_BUNDLER}" --unbundle --type=o
                --targets=hipv4-amdgcn-amd-amdhsa--${PLOW_HSACO_ARCH}
                --input=${_hs_tk_out}.co --output=${_hs_tk_out}
        COMMAND "${CMAKE_COMMAND}" -E rm -f "${_hs_tk_out}.co"
        DEPENDS "${_hs_tk_src}" ${_hs_hdrs}
        COMMENT "hipcc test_kernels.elf"
        VERBATIM)
    list(APPEND _hs_outs "${_hs_tk_out}")

    add_custom_target(gfx950_hsaco ALL DEPENDS ${_hs_outs})
endif()

if(PLOW_ROCM)
    enable_language(HIP)
    # The gfx950 device code is compiled to a RAW code object and loaded through
    # ROCr by amd/hsa_backend.c — it is not linked into this plugin. hipcc is used
    # purely as a compiler here.
    #
    # The AMD MEGAKERNEL (amd/interp.hip + amd/op_*.h -- the AMD production path)
    # has NO CMake target. It is built only by scripts/build_gfx950.sh and
    # build_gfx950_qwen.sh via `hipcc --genco` + `clang-offload-bundler --unbundle`,
    # and loaded through ROCr by amd/hsa_backend.c. This comment used to point at a
    # `plow_gfx950_code_object` target "below"; no such target exists in this file.
    #
    # NOTE (dead code): amd/{mfma,flash,row,golden}.hip and amd/register_mi300.hip
    # are the pre-existing CDNA3 golden/stub tier. mfma.hip is 14 empty TODO
    # bodies and register_mi300.hip leaves every performant slot unregistered.
    # They are superseded by amd/op_*.h + amd/interp.hip but left in place rather
    # than deleted.
    add_library(plow_runtime_rocm SHARED
        amd/golden.hip
        amd/flash.hip
        amd/row.hip
        amd/mfma.hip
        amd/register_mi300.hip)
    target_include_directories(plow_runtime_rocm PUBLIC ${PLOW_INCLUDE})
    set_target_properties(plow_runtime_rocm PROPERTIES
        HIP_ARCHITECTURES "${PLOW_HIP_ARCH}"
        OUTPUT_NAME plow_rt_rocm)
    target_link_libraries(plow_runtime_rocm PRIVATE plow_runtime)
endif()

# --- micro-benchmarks (μBench) -----------------------------------------------
add_subdirectory(ubench)

# --- system-level benchmarks --------------------------------------------------
add_subdirectory(bench)

# --- install ------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS plow_runtime ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/packet.h
    common/dispatch.h
    common/kernel.h
    common/decode.h
    common/interp.h
    common/memmap.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/plow)
foreach(gpu_target plow_runtime_cuda plow_runtime_rocm)
    if(TARGET ${gpu_target})
        install(TARGETS ${gpu_target} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
endforeach()
