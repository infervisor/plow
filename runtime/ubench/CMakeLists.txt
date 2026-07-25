# runtime/bench/CMakeLists.txt — Plow μBench: single-CU/SM kernel profiler.
#
# Builds:
#   - ubench_cu_gfx950.elf    : HIP code object (benchmarks #include production op_*.h)
#   - ubench_cu_roofline      : Host driver (launch + roofline analysis + print)
#
# Enable: cmake -B build -DPLOW_ROCM=ON -DPLOW_UBENCH=ON -DPLOW_HIP_ARCH=gfx950

cmake_minimum_required(VERSION 3.18)

option(PLOW_UBENCH "Build the single-CU/SM micro-benchmark suite" OFF)

if(NOT PLOW_UBENCH)
    return()
endif()

# ============================================================================
# AMD / ROCm path
# ============================================================================
if(PLOW_ROCM)
    # The benchmark code object includes the production op headers directly.
    # It compiles the EXACT same device code the interpreter runs.
    set(UBENCH_HIP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/bench_cu_gfx950.hip)
    set(UBENCH_ELF ${CMAKE_CURRENT_BINARY_DIR}/ubench_cu_gfx950.elf)

    # Dependencies: the production headers
    set(UBENCH_DEPS
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_norm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_elementwise.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_attention.h)

    add_custom_command(
        OUTPUT ${UBENCH_ELF}
        COMMAND hipcc --genco
                --offload-arch=${PLOW_HIP_ARCH}
                -O3 -std=c++17
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                ${UBENCH_HIP_SRC}
                -o ${UBENCH_ELF}
        DEPENDS ${UBENCH_HIP_SRC} ${UBENCH_DEPS}
        COMMENT "μBench: compiling benchmark kernels (includes production op_*.h)"
        VERBATIM
    )
    add_custom_target(ubench_elf ALL DEPENDS ${UBENCH_ELF})

    # Host driver — links against HSA runtime
    add_executable(ubench_cu_roofline
        bench_cu_roofline.c
        ../amd/hsa_backend.c)
    target_include_directories(ubench_cu_roofline PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(ubench_cu_roofline PRIVATE _POSIX_C_SOURCE=199309L)
    target_link_libraries(ubench_cu_roofline PRIVATE hsa-runtime64 m)
    set_target_properties(ubench_cu_roofline PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    # Symlink ELF for runtime discovery
    add_custom_command(TARGET ubench_cu_roofline POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                ubench_cu_gfx950.elf bench_cu_gfx950.elf
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    # Copy counter config files
    foreach(pass 1 2 3)
        configure_file(bench_cu_counters_pass${pass}.txt
                       ${CMAKE_CURRENT_BINARY_DIR}/bench_cu_counters_pass${pass}.txt COPYONLY)
    endforeach()

    # Convenience targets
    add_custom_target(ubench
        COMMAND $<TARGET_FILE:ubench_cu_roofline> --timing-only
        DEPENDS ubench_cu_roofline ubench_elf
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "μBench: timing-only single-CU profiling (Gemma 4 prefill)")

    add_custom_target(ubench_decode
        COMMAND $<TARGET_FILE:ubench_cu_roofline> --timing-only --decode
        DEPENDS ubench_cu_roofline ubench_elf
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "μBench: timing-only single-CU profiling (Gemma 4 decode)")

    add_custom_target(ubench_full
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_ubench_cu.sh
        DEPENDS ubench_cu_roofline ubench_elf
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "μBench: full 3-pass rocprof hardware counter profiling")
endif()

# ============================================================================
# NVIDIA / CUDA path (future)
# ============================================================================
if(PLOW_CUDA)
    message(STATUS "μBench: NVIDIA path uses the same abstraction — create bench_cu_sm120.cu")
endif()
