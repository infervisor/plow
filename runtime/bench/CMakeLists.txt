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
