# runtime/bench/CMakeLists.txt — Plow system-level benchmarks.
#
# NOT part of the default build. Enable with: cmake -DPLOW_BENCH=ON
# These are multi-CU / interpreter-through / library-comparison benchmarks.
# For single-CU micro-benchmarks, see ../ubench/CMakeLists.txt.

cmake_minimum_required(VERSION 3.18)

option(PLOW_BENCH "Build the system-level benchmark suite" OFF)

if(NOT PLOW_BENCH)
    return()
endif()

# ============================================================================
# AMD / ROCm interpreter benchmarks (dispatch/ and interp/)
# ============================================================================
if(PLOW_ROCM)
    enable_language(CXX)

    # --- dispatch family ---
    # aql_launch_floor: raw HSA dispatch-chain overhead measurement
    add_executable(aql_launch_floor
        dispatch/aql_launch_floor.c
        ../amd/hsa_backend.c)
    target_include_directories(aql_launch_floor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(aql_launch_floor PRIVATE _POSIX_C_SOURCE=199309L)
    target_link_libraries(aql_launch_floor PRIVATE hsa-runtime64 m)

    # --- interp family ---
    # interp_gemm_bench: GEMM throughput through the persistent interpreter
    add_executable(interp_gemm_bench
        interp/interp_gemm_bench.c
        ../amd/hsa_backend.c)
    target_include_directories(interp_gemm_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(interp_gemm_bench PRIVATE _POSIX_C_SOURCE=199309L)
    target_link_libraries(interp_gemm_bench PRIVATE hsa-runtime64 m)

    # interp_mxfp4_bench: w4a16 GEMM ladder through the production interpreter
    add_executable(interp_mxfp4_bench
        interp/interp_mxfp4_bench.c
        ../amd/hsa_backend.c)
    target_include_directories(interp_mxfp4_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(interp_mxfp4_bench PRIVATE _POSIX_C_SOURCE=199309L)
    target_compile_options(interp_mxfp4_bench PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(interp_mxfp4_bench PRIVATE hsa-runtime64 m)

    # Grouped simulated-A4W4 GLU+DOWN oracle/benchmark. MPA4C3_INTERP=1 sends the
    # packets through the production gfx942 persistent interpreter object.
    set(MOE_A4W4_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../tests/moe_prefill_a4w4_cdna3_test.hip)
    set_source_files_properties(${MOE_A4W4_TEST_SRC} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
    find_library(PLOW_AMDHIP64 amdhip64 PATHS "$ENV{ROCM_PATH}/lib" NO_DEFAULT_PATH)
    if(NOT PLOW_AMDHIP64)
        message(FATAL_ERROR "amdhip64 from the active Nix ROCm shell was not found")
    endif()
    if(NOT PLOW_AMDHIP64 MATCHES "^/nix/store/")
        message(FATAL_ERROR "amdhip64 must resolve from the Nix store: ${PLOW_AMDHIP64}")
    endif()
    set(MOE_A4W4_W2_TOUCH_ELF
        ${CMAKE_CURRENT_BINARY_DIR}/moe_prefill_a4w4_w2_touch_gfx942.elf)
    add_custom_command(
        OUTPUT ${MOE_A4W4_W2_TOUCH_ELF}
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
                "$ENV{PLOW_HIPCC}" "$ENV{PLOW_BUNDLER}" gfx942
                ${MOE_A4W4_W2_TOUCH_ELF} k_glu_w2_touch 256 2
                -DMPA4C3_DEVICE=1 -DPLOW_MOE_PF_A4W4=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${MOE_A4W4_TEST_SRC}
        DEPENDS ${MOE_A4W4_TEST_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_moe.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
        VERBATIM)
    add_custom_target(moe_prefill_a4w4_w2_touch_gfx942_elf
        DEPENDS ${MOE_A4W4_W2_TOUCH_ELF})
    add_executable(moe_prefill_a4w4_cdna3_test ${MOE_A4W4_TEST_SRC})
    target_include_directories(moe_prefill_a4w4_cdna3_test PRIVATE
        "$ENV{ROCM_PATH}/include"
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common)
    target_compile_definitions(moe_prefill_a4w4_cdna3_test PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(moe_prefill_a4w4_cdna3_test PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(moe_prefill_a4w4_cdna3_test PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(moe_prefill_a4w4_cdna3_test moe_prefill_a4w4_w2_touch_gfx942_elf)

    # Focused BV8 oracle for the opt-in K3 prefill state-residency path.
    set(KDA_STATE_RESIDENT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../tests/kda_step_cdna3_test.hip)
    set(KDA_STATE_RESIDENT_ELF ${CMAKE_CURRENT_BINARY_DIR}/kda_state_resident_gfx942.elf)
    add_custom_command(
        OUTPUT ${KDA_STATE_RESIDENT_ELF}
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
                "$ENV{PLOW_HIPCC}" "$ENV{PLOW_BUNDLER}" gfx942
                ${KDA_STATE_RESIDENT_ELF} k_step 256 2
                -DKDA3_DEVICE=1 -DPLOW_KDA_PF_STATE_RESIDENT=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${KDA_STATE_RESIDENT_SRC}
        DEPENDS ${KDA_STATE_RESIDENT_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_kda.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_elementwise.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
        VERBATIM)
    add_custom_target(kda_state_resident_gfx942_elf DEPENDS ${KDA_STATE_RESIDENT_ELF})
    set_source_files_properties(${KDA_STATE_RESIDENT_SRC}
        PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
    add_executable(kda_state_resident_gfx942_test ${KDA_STATE_RESIDENT_SRC})
    target_compile_features(kda_state_resident_gfx942_test PRIVATE cxx_std_17)
    target_include_directories(kda_state_resident_gfx942_test PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(kda_state_resident_gfx942_test PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(kda_state_resident_gfx942_test PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(kda_state_resident_gfx942_test PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(kda_state_resident_gfx942_test kda_state_resident_gfx942_elf)

    # Standalone gfx942 grouped A8W4 experiment. It is intentionally a raw code object plus a
    # module-loading host oracle: no production packet or interpreter arm is changed.
    set(PLOW_A8W4_HIPCC "$ENV{PLOW_HIPCC}")
    set(PLOW_A8W4_BUNDLER "$ENV{PLOW_BUNDLER}")
    if(NOT PLOW_A8W4_HIPCC MATCHES "^/nix/store/")
        message(FATAL_ERROR "A8W4 experiment hipcc must resolve from the Nix store")
    endif()
    if(NOT PLOW_A8W4_BUNDLER MATCHES "^/nix/store/")
        message(FATAL_ERROR "A8W4 experiment bundler must resolve from the Nix store")
    endif()
    set(MOE_A8W4_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../tests/moe_prefill_a8w4_gfx942_test.hip)
    set(MOE_A8W4_ELF ${CMAKE_CURRENT_BINARY_DIR}/moe_prefill_a8w4_gfx942.elf)
    add_custom_command(
        OUTPUT ${MOE_A8W4_ELF}
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
                ${PLOW_A8W4_HIPCC} ${PLOW_A8W4_BUNDLER} gfx942
                ${MOE_A8W4_ELF} a8w4_group_glu 256 2
                -std=c++17 -DMPA8W4_DEVICE=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                ${MOE_A8W4_SRC}
        DEPENDS ${MOE_A8W4_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_arch.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
        VERBATIM)
    add_custom_target(moe_prefill_a8w4_gfx942_elf DEPENDS ${MOE_A8W4_ELF})
    find_program(PLOW_BENCH_PYTHON python3 REQUIRED)
    add_custom_target(moe_prefill_a8w4_gfx942_audit
        COMMAND ${PLOW_BENCH_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/asm_audit.py
                --expect ${CMAKE_CURRENT_SOURCE_DIR}/../tests/moe_prefill_a8w4_gfx942_expect.json
                ${MOE_A8W4_ELF}
        DEPENDS moe_prefill_a8w4_gfx942_elf
        VERBATIM)
    set_source_files_properties(${MOE_A8W4_SRC} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
    add_executable(moe_prefill_a8w4_gfx942_test ${MOE_A8W4_SRC})
    target_compile_features(moe_prefill_a8w4_gfx942_test PRIVATE cxx_std_17)
    target_include_directories(moe_prefill_a8w4_gfx942_test PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(moe_prefill_a8w4_gfx942_test PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(moe_prefill_a8w4_gfx942_test PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(moe_prefill_a8w4_gfx942_test PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(moe_prefill_a8w4_gfx942_test moe_prefill_a8w4_gfx942_elf)

    # Standalone K3 grouped W4A16 bridge control. It is deliberately not an interpreter arm:
    # CMake emits a private gfx942 code object from the device half of the source, while the host
    # half supplies routing/padding sentinels and the f64 oracle.
    set(PLOW_W4A16_HIPCC "$ENV{PLOW_HIPCC}")
    set(PLOW_W4A16_BUNDLER "$ENV{PLOW_BUNDLER}")
    if(NOT PLOW_W4A16_HIPCC MATCHES "^/nix/store/")
        message(FATAL_ERROR "W4A16 control hipcc must resolve from the Nix store")
    endif()
    if(NOT PLOW_W4A16_BUNDLER MATCHES "^/nix/store/")
        message(FATAL_ERROR "W4A16 control bundler must resolve from the Nix store")
    endif()
    set(MOE_W4A16_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/moe_prefill_w4a16_bridge_gfx942_test.hip)
    set(MOE_W4A16_ELF
        ${CMAKE_CURRENT_BINARY_DIR}/moe_prefill_w4a16_bridge_gfx942.elf)
    add_custom_command(
        OUTPUT ${MOE_W4A16_ELF}
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
            ${PLOW_W4A16_HIPCC} ${PLOW_W4A16_BUNDLER} gfx942
            ${MOE_W4A16_ELF} k_w4a16 256 1
            -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
            -DMPW4A16_DEVICE=1 ${MOE_W4A16_SRC}
        DEPENDS ${MOE_W4A16_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_arch.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
        VERBATIM)
    add_custom_target(moe_prefill_w4a16_bridge_gfx942_elf DEPENDS ${MOE_W4A16_ELF})
    set_source_files_properties(${MOE_W4A16_SRC}
        PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
    add_executable(moe_prefill_w4a16_bridge_gfx942_test ${MOE_W4A16_SRC})
    target_compile_features(moe_prefill_w4a16_bridge_gfx942_test PRIVATE cxx_std_17)
    target_include_directories(moe_prefill_w4a16_bridge_gfx942_test PRIVATE
        "$ENV{ROCM_PATH}/include")
    target_compile_definitions(moe_prefill_w4a16_bridge_gfx942_test
        PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(moe_prefill_w4a16_bridge_gfx942_test
        PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(moe_prefill_w4a16_bridge_gfx942_test PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(moe_prefill_w4a16_bridge_gfx942_test
        moe_prefill_w4a16_bridge_gfx942_elf)

    # qwen_interp_bench: same as above but Qwen3 shapes
    add_executable(qwen_interp_bench
        interp/qwen_interp_bench.c
        ../amd/hsa_backend.c)
    target_include_directories(qwen_interp_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(qwen_interp_bench PRIVATE _POSIX_C_SOURCE=199309L)
    target_link_libraries(qwen_interp_bench PRIVATE hsa-runtime64 m)

    # dsa_gather_bench: DSA gather throughput
    add_executable(dsa_gather_bench
        interp/dsa_gather_bench.c
        ../amd/hsa_backend.c)
    target_include_directories(dsa_gather_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(dsa_gather_bench PRIVATE _POSIX_C_SOURCE=199309L)
    target_link_libraries(dsa_gather_bench PRIVATE hsa-runtime64 m)

    # --- gemm family (multi-CU sweeps) ---
    add_executable(gemm_bench_8k
        gemm/gemm_bench_8k.c
        ../amd/hsa_backend.c)
    target_include_directories(gemm_bench_8k PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(gemm_bench_8k PRIVATE _POSIX_C_SOURCE=199309L)
    target_link_libraries(gemm_bench_8k PRIVATE hsa-runtime64 m)

    add_executable(gemm_tile_sweep
        gemm/gemm_tile_sweep.c
        ../amd/hsa_backend.c)
    target_include_directories(gemm_tile_sweep PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(gemm_tile_sweep PRIVATE _POSIX_C_SOURCE=199309L)
    target_link_libraries(gemm_tile_sweep PRIVATE hsa-runtime64 m)

    # Same-tile fused-vs-unfused MXFP4 GLU through the gfx942 interpreter.
    add_executable(interp_mxfp4_glu_gfx942
        gemm/interp_mxfp4_glu_gfx942.c
        ../amd/hsa_backend.c)
    target_include_directories(interp_mxfp4_glu_gfx942 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(interp_mxfp4_glu_gfx942 PRIVATE _POSIX_C_SOURCE=200809L)
    target_compile_options(interp_mxfp4_glu_gfx942 PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(interp_mxfp4_glu_gfx942 PRIVATE hsa-runtime64 m)

    add_executable(gemm_tensile_ext
        gemm/gemm_tensile_ext.c
        ../amd/hsa_backend.c)
    target_include_directories(gemm_tensile_ext PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(gemm_tensile_ext PRIVATE _POSIX_C_SOURCE=199309L)
    target_link_libraries(gemm_tensile_ext PRIVATE hsa-runtime64 m)

    add_executable(cu_gemm_bench
        gemm/cu_gemm_bench.c
        ../amd/hsa_backend.c)
    target_include_directories(cu_gemm_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(cu_gemm_bench PRIVATE _POSIX_C_SOURCE=199309L)
    target_link_libraries(cu_gemm_bench PRIVATE hsa-runtime64 m)
endif()
