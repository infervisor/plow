# runtime/ubench/CMakeLists.txt — Plow μBench: single-CU/SM kernel profiler.
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
    function(plow_rocm_code_object output source)
        set(bundle "${output}.co")
        add_custom_command(
            OUTPUT ${output}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/ceiling
            COMMAND ${PLOW_HSACO_HIPCC} --genco
                    --offload-arch=${PLOW_HIP_ARCH} -O3 -w
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    ${source} -o ${bundle}
            COMMAND ${PLOW_HSACO_BUNDLER} --unbundle --type=o
                    --targets=hipv4-amdgcn-amd-amdhsa--${PLOW_HIP_ARCH}
                    --input=${bundle} --output=${output}
            COMMAND ${CMAKE_COMMAND} -E rm -f ${bundle}
            DEPENDS ${source}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_arch.h
            VERBATIM)
    endfunction()

    set(MFMA_RATE_ELF ${CMAKE_CURRENT_BINARY_DIR}/ceiling/mfma_rate.elf)
    set(DECODE_BW_ELF ${CMAKE_CURRENT_BINARY_DIR}/ceiling/bw_probe.elf)
    plow_rocm_code_object(${MFMA_RATE_ELF}
                          ${CMAKE_CURRENT_SOURCE_DIR}/mfma_rate_gfx942.hip)
    plow_rocm_code_object(${DECODE_BW_ELF}
                          ${CMAKE_CURRENT_SOURCE_DIR}/../tests/decode_bw_probe.hip)
    add_custom_target(ceiling_probe_objects DEPENDS ${MFMA_RATE_ELF} ${DECODE_BW_ELF})

    add_executable(mfma_rate_test mfma_rate_test.c ../amd/hsa_backend.c)
    add_executable(decode_bw_probe ../tests/decode_bw_probe.c ../amd/hsa_backend.c)
    foreach(probe mfma_rate_test decode_bw_probe)
        target_include_directories(${probe} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../common
            ${CMAKE_CURRENT_SOURCE_DIR}/../amd
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
        target_compile_definitions(${probe} PRIVATE _POSIX_C_SOURCE=199309L)
        target_compile_options(${probe} PRIVATE -Wall -Wextra -Werror)
        target_link_libraries(${probe} PRIVATE hsa-runtime64 m)
        set_target_properties(${probe} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ceiling)
        add_dependencies(${probe} ceiling_probe_objects)
    endforeach()
    add_custom_target(ceiling_probes DEPENDS mfma_rate_test decode_bw_probe ceiling_probe_objects)

    # The benchmark code object includes the production op headers directly.
    # It compiles the EXACT same device code the interpreter runs.
    set(UBENCH_HIP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/bench_cu_gfx950.hip)
    set(UBENCH_ELF ${CMAKE_CURRENT_BINARY_DIR}/ubench_cu_gfx950.elf)

    # Dependencies: the production headers
    set(UBENCH_DEPS
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm_common.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm_gfx942.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm_gfx950.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_norm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_elementwise.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_attention.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_attention_common.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_attention_gfx942.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_attention_gfx950.h)

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
# NVIDIA / CUDA path (sm_12x; GB10 = sm_121a via -DPLOW_CUBIN_ARCH=sm_121a)
#   ceiling_<arch>.cubin   : single-SM ceilings + harness utilities (fill, flush, digest)
#   fp4_tile_<arch>.cubin  : single-SM NVFP4 MMA tile loops (--fp4-tile)
#   ubench_cu_roofline_nv  : host driver (bench_cu_roofline.c -DPLOW_CUDA, driver API backend)
#   ubench_ops_<arch>.cubin: bench_cu_sm120.cu (= interp_sm120.cu + ubench_exec), only with
#                            -DPLOW_CUBIN_CONFIG=<assets>/plow_config.h and
#                            -DPLOW_UBENCH_OBJECT_DEFS="<the object's plowc define list>"
# run_ubench_cu.sh --nv does the same from a plowc build log.
# ============================================================================
if(PLOW_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(_ub_nvcc bash ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/nvcc_cubin.sh ${PLOW_CUBIN_NVCC})
    set(UBENCH_CEIL_CUBIN ${CMAKE_CURRENT_BINARY_DIR}/ceiling_${PLOW_CUBIN_ARCH}.cubin)
    add_custom_command(
        OUTPUT ${UBENCH_CEIL_CUBIN}
        COMMAND ${_ub_nvcc} ${UBENCH_CEIL_CUBIN} ceil_mma required
                -arch=${PLOW_CUBIN_ARCH} -O3 -cubin ${CMAKE_CURRENT_SOURCE_DIR}/ceiling_sm120.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ceiling_sm120.cu
        COMMENT "μBench: single-SM ceiling cubin"
        VERBATIM)
    add_custom_target(ubench_nv_ceiling_cubin ALL DEPENDS ${UBENCH_CEIL_CUBIN})
    set(UBENCH_FP4_CUBIN ${CMAKE_CURRENT_BINARY_DIR}/fp4_tile_${PLOW_CUBIN_ARCH}.cubin)
    add_custom_command(
        OUTPUT ${UBENCH_FP4_CUBIN}
        COMMAND ${_ub_nvcc} ${UBENCH_FP4_CUBIN} ft_chain2 required
                -arch=${PLOW_CUBIN_ARCH} -O3 -cubin ${CMAKE_CURRENT_SOURCE_DIR}/fp4_tile_sm120.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fp4_tile_sm120.cu
        COMMENT "μBench: single-SM NVFP4 tile-loop cubin"
        VERBATIM)
    add_custom_target(ubench_nv_fp4_tile_cubin ALL DEPENDS ${UBENCH_FP4_CUBIN})

    add_executable(ubench_cu_roofline_nv bench_cu_roofline.c ../nvidia/cuda_backend.c)
    target_include_directories(ubench_cu_roofline_nv PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(ubench_cu_roofline_nv PRIVATE PLOW_CUDA=1)
    target_link_libraries(ubench_cu_roofline_nv PRIVATE CUDA::cuda_driver m)
    add_dependencies(ubench_cu_roofline_nv ubench_nv_ceiling_cubin)

    set(PLOW_UBENCH_OBJECT_DEFS "" CACHE STRING
        "define list plowc used for the measured interpreter object (its build.log 'nvcc <object>.cubin (...)' line)")
    if(PLOW_CUBIN_CONFIG AND PLOW_UBENCH_OBJECT_DEFS)
        get_filename_component(_ub_cfg_dir "${PLOW_CUBIN_CONFIG}" DIRECTORY)
        set(UBENCH_OPS_CUBIN ${CMAKE_CURRENT_BINARY_DIR}/ubench_ops_${PLOW_CUBIN_ARCH}.cubin)
        add_custom_command(
            OUTPUT ${UBENCH_OPS_CUBIN}
            COMMAND ${_ub_nvcc} ${UBENCH_OPS_CUBIN} ubench_exec required
                    -arch=${PLOW_CUBIN_ARCH} -O3 -cubin
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/../common -I ${CMAKE_CURRENT_SOURCE_DIR}/../nvidia
                    -I ${_ub_cfg_dir} ${PLOW_UBENCH_OBJECT_DEFS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/bench_cu_sm120.cu
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bench_cu_sm120.cu ${PLOW_CUBIN_CONFIG} ${_cubin_headers}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../nvidia/interp_sm120.cu
            COMMENT "μBench: ops cubin (interp_sm120.cu + ubench_exec)"
            VERBATIM)
        add_custom_target(ubench_nv_ops_cubin ALL DEPENDS ${UBENCH_OPS_CUBIN})
    endif()

    add_custom_target(ubench_nv_ceilings
        COMMAND $<TARGET_FILE:ubench_cu_roofline_nv> --base ${UBENCH_CEIL_CUBIN} --ceilings
        DEPENDS ubench_cu_roofline_nv
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "μBench: GB10 / sm_12x single-SM ceilings")
    add_custom_target(ubench_nv_fp4_tile
        COMMAND $<TARGET_FILE:ubench_cu_roofline_nv> --base ${UBENCH_CEIL_CUBIN} --fp4-tile ${UBENCH_FP4_CUBIN}
        DEPENDS ubench_cu_roofline_nv ubench_nv_fp4_tile_cubin
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "μBench: single-SM NVFP4 tile loops")
endif()
