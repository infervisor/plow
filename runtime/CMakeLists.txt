cmake_minimum_required(VERSION 3.18)
project(plow_runtime C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(PLOW_CUDA "Build the CUDA (Hopper/Blackwell) kernel backend" OFF)
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
    # rtx-11 PX-2: -DPLOW_NV_W8A8=ON selects the true w8a8 fp8 mma.sync.m16n8k32 prefill path
    # (both operands e4m3 + per-row activation quant) on ALL gemma interp objects, so a w8a8 pkt
    # traces/runs consistently. OFF keeps the w8a16 dequant path. Applied to every gemma object so
    # the shared PLOW_NV_W8A8 dispatch + d_quant_fp8 compile identically across decode/prefill/seg.
    option(PLOW_NV_W8A8 "Build the gemma interp objects with the true w8a8 fp8 mma path" OFF)
    set(PLOW_NV_W8A8_DEFS "")
    if(PLOW_NV_W8A8)
        set(PLOW_NV_W8A8_DEFS PLOW_NV_W8A8=1)
    endif()

    # fp8 (e4m3) KV cache (rtx-19 E3). Compiles the HEADNORM_ROPE_FP8 / FLASH_DECODE_FP8 arms into
    # every gemma object so the shared dispatch is consistent decode/prefill/seg. OFF => bf16 KV,
    # byte-identical to HEAD. Applied via ${PLOW_FP8_KV_DEFS} alongside the w8a8 defs.
    option(PLOW_FP8_KV "Build the gemma interp objects with the fp8 e4m3 KV-cache arms" OFF)
    set(PLOW_FP8_KV_DEFS "")
    # The prefill-family objects (_pf/_seg/_gemm) additionally need PLOW_NV_FA_PIPE=0 when fp8-KV is
    # on: the FLASH_PREFILL_FP8 arm dequants at the smem stage (cp.async cannot convert fp8 inline),
    # so it only exists on the synchronous-staging (PIPE=0) build and traps under PIPE=1. Decode is
    # PIPE-agnostic. bf16-KV runs use the default (PIPE=1) harness, so this does not affect them.
    set(PLOW_FP8_KV_PF_DEFS "")
    # beat-fp8-mma: PLOW_FP8_KV_FASTPF keeps the prefill objects on the FAST PIPE=1 build — the
    # FLASH_PREFILL_FP8 op then runs the px4 fp8-mma arm (hd512 only; pair with mixed-KV packets,
    # PLOW_FP8_KV_FULL=1). OFF preserves the legacy all-layer fp8-KV PIPE=0 prefill.
    option(PLOW_FP8_KV_FASTPF "fp8-KV prefill via the PIPE=1 px4 fp8-mma arm (hd512-only packets)" OFF)
    if(PLOW_FP8_KV)
        set(PLOW_FP8_KV_DEFS PLOW_FP8_KV=1)
        if(PLOW_FP8_KV_FASTPF)
            set(PLOW_FP8_KV_PF_DEFS PLOW_FP8_KV=1)
        else()
            set(PLOW_FP8_KV_PF_DEFS PLOW_FP8_KV=1 PLOW_NV_FA_PIPE=0)
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

if(PLOW_ROCM)
    enable_language(HIP)
    # The gfx950 device code is compiled to a RAW code object and loaded through
    # ROCr by amd/hsa_backend.c — it is not linked into this plugin. hipcc is used
    # purely as a compiler here; see the plow_gfx950_code_object target below.
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
