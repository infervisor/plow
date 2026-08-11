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

add_executable(control_queue_probe_test dispatch/control_queue_probe_test.c)
target_compile_options(control_queue_probe_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME control_queue_probe_abi COMMAND control_queue_probe_test)

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

    set(PLOW_CONTROL_QUEUE_HIPCC "$ENV{PLOW_HIPCC}")
    if(NOT PLOW_CONTROL_QUEUE_HIPCC MATCHES "^/nix/store/")
        message(FATAL_ERROR "control-queue gfx942 probe requires Nix PLOW_HIPCC")
    endif()
    set(CONTROL_QUEUE_PROBE_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/control_queue_dispatch_floor.hip)
    set(CONTROL_QUEUE_PROBE_BIN
        ${CMAKE_CURRENT_BINARY_DIR}/control_queue_dispatch_floor_gfx942.bin)
    add_custom_command(
        OUTPUT ${CONTROL_QUEUE_PROBE_BIN}
        COMMAND "${PLOW_CONTROL_QUEUE_HIPCC}" --offload-arch=gfx942 -O3 -std=c++17
                -Wall -Wextra -Werror ${CONTROL_QUEUE_PROBE_SRC}
                -o ${CONTROL_QUEUE_PROBE_BIN}
        DEPENDS ${CONTROL_QUEUE_PROBE_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/../common/control_queue_probe.h
        VERBATIM)
    add_custom_target(control_queue_dispatch_floor_gfx942
        DEPENDS ${CONTROL_QUEUE_PROBE_BIN})

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

    set(PLOW_MLA_HIPCC "$ENV{PLOW_HIPCC}")
    set(PLOW_MLA_BUNDLER "$ENV{PLOW_BUNDLER}")
    if(NOT PLOW_MLA_HIPCC MATCHES "^/nix/store/")
        message(FATAL_ERROR "MLA gfx942 benches require Nix PLOW_HIPCC")
    endif()
    if(NOT PLOW_MLA_BUNDLER MATCHES "^/nix/store/")
        message(FATAL_ERROR "MLA gfx942 benches require Nix PLOW_BUNDLER")
    endif()

    set(MLA_PREFILL_BENCH_ELF ${CMAKE_CURRENT_BINARY_DIR}/mla_prefill_bench_gfx942.elf)
    add_custom_command(
        OUTPUT ${MLA_PREFILL_BENCH_ELF}
        COMMAND "${PLOW_MLA_HIPCC}" --offload-arch=gfx942 -O3 -w --genco
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/test_kernels.hip
                -o ${MLA_PREFILL_BENCH_ELF}.co
        COMMAND "${PLOW_MLA_BUNDLER}" --unbundle --type=o
                --targets=hipv4-amdgcn-amd-amdhsa--gfx942
                --input=${MLA_PREFILL_BENCH_ELF}.co --output=${MLA_PREFILL_BENCH_ELF}
        COMMAND ${CMAKE_COMMAND} -E rm -f ${MLA_PREFILL_BENCH_ELF}.co
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../amd/test_kernels.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_attention.h
        VERBATIM)
    add_custom_target(mla_prefill_bench_gfx942_elf DEPENDS ${MLA_PREFILL_BENCH_ELF})
    add_executable(mla_prefill_bench_gfx942
        ../tests/mla_prefill_bench_gfx950.c
        ../amd/hsa_backend.c)
    target_include_directories(mla_prefill_bench_gfx942 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(mla_prefill_bench_gfx942 PRIVATE _POSIX_C_SOURCE=199309L)
    target_compile_options(mla_prefill_bench_gfx942 PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(mla_prefill_bench_gfx942 PRIVATE hsa-runtime64 m)
    add_dependencies(mla_prefill_bench_gfx942 mla_prefill_bench_gfx942_elf)

    set(MLA_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/mla_fixture.bin)
    set(MLA_REF ${CMAKE_CURRENT_BINARY_DIR}/mla_ref)
    find_program(PLOW_BENCH_RUSTC rustc REQUIRED)
    add_custom_command(
        OUTPUT ${MLA_FIXTURE}
        COMMAND ${PLOW_BENCH_RUSTC} -O ${CMAKE_CURRENT_SOURCE_DIR}/../tests/mla_ref.rs
                -o ${MLA_REF}
        COMMAND ${MLA_REF} ${MLA_FIXTURE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../tests/mla_ref.rs
        VERBATIM)
    add_custom_target(mla_fixture DEPENDS ${MLA_FIXTURE})
    add_executable(mla_gfx942_test
        ../tests/mla_gfx950_test.c
        ../amd/hsa_backend.c)
    target_include_directories(mla_gfx942_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(mla_gfx942_test PRIVATE _POSIX_C_SOURCE=199309L)
    target_compile_options(mla_gfx942_test PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(mla_gfx942_test PRIVATE hsa-runtime64 m)
    add_dependencies(mla_gfx942_test mla_fixture mla_prefill_bench_gfx942_elf)

    # Exact TP8 K3 latent-MoE seam: current two-packet combine -> XReduce
    # protocol versus per-slice fused publication, both at H=3584/top-k=16/b=7.
    set(K3_CXR_ELF ${CMAKE_CURRENT_BINARY_DIR}/tp_moe_combine_xreduce_gfx942.elf)
    add_custom_command(
        OUTPUT ${K3_CXR_ELF}
        COMMAND "${PLOW_MLA_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                -DPLOW_BUCKET_DECODE=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${CMAKE_CURRENT_SOURCE_DIR}/../tests/tp_moe_combine_xreduce_kernels.hip
                -o ${K3_CXR_ELF}.co
        COMMAND "${PLOW_MLA_BUNDLER}" --unbundle --type=o
                --targets=hipv4-amdgcn-amd-amdhsa--gfx942
                --input=${K3_CXR_ELF}.co --output=${K3_CXR_ELF}
        COMMAND ${CMAKE_COMMAND} -E rm -f ${K3_CXR_ELF}.co
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../tests/tp_moe_combine_xreduce_kernels.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_moe.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_collective.h
        VERBATIM)
    add_custom_target(tp_moe_combine_xreduce_gfx942_elf DEPENDS ${K3_CXR_ELF})
    add_executable(tp_moe_combine_xreduce_bench
        ../tests/tp_moe_combine_xreduce_bench.c
        ../amd/hsa_backend.c)
    target_include_directories(tp_moe_combine_xreduce_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../amd
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_compile_definitions(tp_moe_combine_xreduce_bench PRIVATE _POSIX_C_SOURCE=199309L)
    target_compile_options(tp_moe_combine_xreduce_bench PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(tp_moe_combine_xreduce_bench PRIVATE hsa-runtime64 m)
    add_dependencies(tp_moe_combine_xreduce_bench tp_moe_combine_xreduce_gfx942_elf)

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

    # Current-shape K3 B1 GEMV grid sweep. The host validates every winning grid
    # byte-for-byte before reporting it as a packet candidate.
    set(K3_GEMV_GRID_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_gemv_grid_sweep_gfx942.co)
    add_custom_command(
        OUTPUT ${K3_GEMV_GRID_CO}
        COMMAND "$ENV{PLOW_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_K3=1
                -DGV_UNROLL=14 -DPLOW_FP8_KV=1 -DPLOW_GEMV_LG=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_gemvbf16_bench.hip
                -o ${K3_GEMV_GRID_CO}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_gemvbf16_bench.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
        VERBATIM)
    add_custom_target(k3_gemv_grid_sweep_gfx942_object DEPENDS ${K3_GEMV_GRID_CO})
    add_executable(k3_gemv_grid_sweep amd/k3_gemv_grid_sweep.cpp)
    target_compile_features(k3_gemv_grid_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_gemv_grid_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_gemv_grid_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_gemv_grid_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_gemv_grid_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_gemv_grid_sweep k3_gemv_grid_sweep_gfx942_object)

    add_executable(k3_gemv_rsplit_sweep amd/k3_gemvbf16_bench.cpp)
    target_compile_features(k3_gemv_rsplit_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_gemv_rsplit_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_gemv_rsplit_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_gemv_rsplit_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_gemv_rsplit_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_gemv_rsplit_sweep k3_gemv_grid_sweep_gfx942_object)

    set(K3_GEMV_COHORT_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_gemv_cohort_sweep_gfx942.co)
    add_custom_command(
        OUTPUT ${K3_GEMV_COHORT_CO}
        COMMAND "$ENV{PLOW_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_K3=1
                -DGV_UNROLL=14 -DPLOW_FP8_KV=1 -DPLOW_GEMV_LG=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_gemv_cohort_sweep.hip
                -o ${K3_GEMV_COHORT_CO}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_gemv_cohort_sweep.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
        VERBATIM)
    add_custom_target(k3_gemv_cohort_sweep_gfx942_object DEPENDS ${K3_GEMV_COHORT_CO})
    add_executable(k3_gemv_cohort_sweep amd/k3_gemv_cohort_sweep.cpp)
    target_compile_features(k3_gemv_cohort_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_gemv_cohort_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_gemv_cohort_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_gemv_cohort_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_gemv_cohort_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_gemv_cohort_sweep k3_gemv_cohort_sweep_gfx942_object)

    set(K3_MLA_KDU4_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_mla_kdu4_gfx942.co)
    set(K3_MLA_KDU1_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_mla_kdu1_gfx942.co)
    set(K3_MLA_KDU2_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_mla_kdu2_gfx942.co)
    set(K3_MLA_KDU8_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_mla_kdu8_gfx942.co)
    foreach(KDU IN ITEMS 1 2 4 8)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/k3_mla_kdu${KDU}_gfx942.co
            COMMAND "$ENV{PLOW_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                    -DPLOW_BUCKET_DECODE=1 -DPLOW_FP8_KV=1 -DPLOW_K3=1
                    -DPLOW_WG_WAVES=8 -DFA_MLA_KDU=${KDU}
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                    ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_mla_kdu_sweep.hip
                    -o ${CMAKE_CURRENT_BINARY_DIR}/k3_mla_kdu${KDU}_gfx942.co
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_mla_kdu_sweep.hip
                    ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_attention.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
            VERBATIM)
    endforeach()
    add_custom_target(k3_mla_kdu_sweep_gfx942_objects
        DEPENDS ${K3_MLA_KDU1_CO} ${K3_MLA_KDU2_CO} ${K3_MLA_KDU4_CO} ${K3_MLA_KDU8_CO})
    add_executable(k3_mla_kdu_sweep amd/k3_mla_kdu_sweep.cpp)
    target_compile_features(k3_mla_kdu_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_mla_kdu_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_mla_kdu_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_mla_kdu_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_mla_kdu_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_mla_kdu_sweep k3_mla_kdu_sweep_gfx942_objects)

    # Model-free full-grid screen for replacing K3 B1 dense BF16 weights with
    # block-scaled E4M3 W8A16 or checkpoint-native-layout MXFP4. Only candidates
    # projecting >=5 ms/token proceed to checkpoint conversion and TP8 serving.
    set(K3_GEMV_W8_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_gemv_w8a16_sweep_gfx942.co)
    add_custom_command(
        OUTPUT ${K3_GEMV_W8_CO}
        COMMAND "$ENV{PLOW_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_K3=1 -DPLOW_MXFP4=1
                -DGV_UNROLL=14 -DPLOW_FP8_KV=1 -DPLOW_GEMV_LG=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_gemv_w8a16_sweep.hip
                -o ${K3_GEMV_W8_CO}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_gemv_w8a16_sweep.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
        VERBATIM)
    add_custom_target(k3_gemv_w8a16_sweep_gfx942_object DEPENDS ${K3_GEMV_W8_CO})
    add_executable(k3_gemv_w8a16_sweep amd/k3_gemv_w8a16_sweep.cpp)
    target_compile_features(k3_gemv_w8a16_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_gemv_w8a16_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_gemv_w8a16_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_gemv_w8a16_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_gemv_w8a16_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_gemv_w8a16_sweep k3_gemv_w8a16_sweep_gfx942_object)

    # Exact K3 B1 FP8 MLA single-op sweep. GF4 is the shipped control; GF6/GF12
    # test latent-row reuse before any interpreter or packet change.
    set(K3_MLA_GF_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_mla_gf_sweep_gfx942.co)
    add_custom_command(
        OUTPUT ${K3_MLA_GF_CO}
        COMMAND "$ENV{PLOW_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_mla_gf_sweep.hip
                -o ${K3_MLA_GF_CO}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_mla_gf_sweep.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_attention.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
        VERBATIM)
    add_custom_target(k3_mla_gf_sweep_gfx942_object DEPENDS ${K3_MLA_GF_CO})
    add_executable(k3_mla_gf_sweep amd/k3_mla_gf_sweep.cpp)
    target_compile_features(k3_mla_gf_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_mla_gf_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_mla_gf_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_mla_gf_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_mla_gf_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_mla_gf_sweep k3_mla_gf_sweep_gfx942_object)

    # Exact K3 B1 grouped-MXFP4 cold-weight block/grid screen. A one-block row
    # is diagnostic; only full-grid results are valid promotion evidence.
    set(K3_MOE_GRID_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_moe_grid_sweep_gfx942.co)
    add_custom_command(
        OUTPUT ${K3_MOE_GRID_CO}
        COMMAND "$ENV{PLOW_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                -DPLOW_BUCKET_DECODE=1 -DPLOW_MXFP4=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_moe_grid_sweep.hip
                -o ${K3_MOE_GRID_CO}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_moe_grid_sweep.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_moe.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
        VERBATIM)
    add_custom_target(k3_moe_grid_sweep_gfx942_object DEPENDS ${K3_MOE_GRID_CO})
    add_executable(k3_moe_grid_sweep amd/k3_moe_grid_sweep.cpp)
    target_compile_features(k3_moe_grid_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_moe_grid_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_moe_grid_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_moe_grid_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_moe_grid_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_moe_grid_sweep k3_moe_grid_sweep_gfx942_object)

    # Exact one-workgroup K3 B1 router tail. Two objects differ only in whether
    # each lane caches its packed expert keys across the 16 block reductions.
    set(K3_ROUTER_TOPK_SRC ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_router_topk_sweep.hip)
    set(K3_ROUTER_TOPK_CTL ${CMAKE_CURRENT_BINARY_DIR}/k3_router_topk_control_gfx942.elf)
    set(K3_ROUTER_TOPK_LOCAL ${CMAKE_CURRENT_BINARY_DIR}/k3_router_topk_local_gfx942.elf)
    add_custom_command(
        OUTPUT ${K3_ROUTER_TOPK_CTL}
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
                "$ENV{PLOW_HIPCC}" "$ENV{PLOW_BUNDLER}" gfx942
                ${K3_ROUTER_TOPK_CTL} k3_router_topk 256 2
                -std=c++17 -DPLOW_BUCKET_DECODE=1 -DPLOW_K3=1
                -DPLOW_MOE_ROUTER_SELECT=1 -DPLOW_MOE_ROUTER_SELECT_LOCAL=0
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${K3_ROUTER_TOPK_SRC}
        DEPENDS ${K3_ROUTER_TOPK_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_moe.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
        VERBATIM)
    add_custom_command(
        OUTPUT ${K3_ROUTER_TOPK_LOCAL}
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
                "$ENV{PLOW_HIPCC}" "$ENV{PLOW_BUNDLER}" gfx942
                ${K3_ROUTER_TOPK_LOCAL} k3_router_topk 256 2
                -std=c++17 -DPLOW_BUCKET_DECODE=1 -DPLOW_K3=1
                -DPLOW_MOE_ROUTER_SELECT=1 -DPLOW_MOE_ROUTER_SELECT_LOCAL=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${K3_ROUTER_TOPK_SRC}
        DEPENDS ${K3_ROUTER_TOPK_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_moe.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
        VERBATIM)
    add_custom_target(k3_router_topk_sweep_gfx942_objects
        DEPENDS ${K3_ROUTER_TOPK_CTL} ${K3_ROUTER_TOPK_LOCAL})
    add_executable(k3_router_topk_sweep amd/k3_router_topk_sweep.cpp)
    target_compile_features(k3_router_topk_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_router_topk_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_router_topk_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_router_topk_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_router_topk_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_router_topk_sweep k3_router_topk_sweep_gfx942_objects)

    set(K3_KDA_DB_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_kda_conv_step_db_gfx942.co)
    add_custom_command(
        OUTPUT ${K3_KDA_DB_CO}
        COMMAND "$ENV{PLOW_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                -DPLOW_BUCKET_DECODE=1 -DPLOW_K3=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_kda_conv_step_db.hip
                -o ${K3_KDA_DB_CO}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_kda_conv_step_db.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_kda.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/amd_common.h
        VERBATIM)
    add_custom_target(k3_kda_conv_step_db_gfx942_object DEPENDS ${K3_KDA_DB_CO})
    add_executable(k3_kda_conv_step_db amd/k3_kda_conv_step_db.cpp)
    target_compile_features(k3_kda_conv_step_db PRIVATE cxx_std_17)
    target_include_directories(k3_kda_conv_step_db PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_kda_conv_step_db PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_kda_conv_step_db PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_kda_conv_step_db PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_kda_conv_step_db k3_kda_conv_step_db_gfx942_object)

    set(K3_QKVG_CONV_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_qkvg_conv_sweep_gfx942.co)
    add_custom_command(
        OUTPUT ${K3_QKVG_CONV_CO}
        COMMAND "$ENV{PLOW_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_K3=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_qkvg_conv_sweep.hip
                -o ${K3_QKVG_CONV_CO}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_qkvg_conv_sweep.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_kda.h
        VERBATIM)
    add_custom_target(k3_qkvg_conv_sweep_gfx942_object DEPENDS ${K3_QKVG_CONV_CO})
    add_executable(k3_qkvg_conv_sweep amd/k3_qkvg_conv_sweep.cpp)
    target_compile_features(k3_qkvg_conv_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_qkvg_conv_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_qkvg_conv_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_qkvg_conv_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_qkvg_conv_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_qkvg_conv_sweep k3_qkvg_conv_sweep_gfx942_object)

    set(K3_KDA_NORM_GEMV_CO ${CMAKE_CURRENT_BINARY_DIR}/k3_kda_norm_gemv_sweep_gfx942.co)
    add_custom_command(
        OUTPUT ${K3_KDA_NORM_GEMV_CO}
        COMMAND "$ENV{PLOW_HIPCC}" --offload-arch=gfx942 -O3 -w -std=c++17 --genco
                -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 -DPLOW_K3=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_kda_norm_gemv_sweep.hip
                -o ${K3_KDA_NORM_GEMV_CO}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/amd/k3_kda_norm_gemv_sweep.hip
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_gemm.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_kda.h
        VERBATIM)
    add_custom_target(k3_kda_norm_gemv_sweep_gfx942_object DEPENDS ${K3_KDA_NORM_GEMV_CO})
    add_executable(k3_kda_norm_gemv_sweep amd/k3_kda_norm_gemv_sweep.cpp)
    target_compile_features(k3_kda_norm_gemv_sweep PRIVATE cxx_std_17)
    target_include_directories(k3_kda_norm_gemv_sweep PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_kda_norm_gemv_sweep PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_kda_norm_gemv_sweep PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_kda_norm_gemv_sweep PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_kda_norm_gemv_sweep k3_kda_norm_gemv_sweep_gfx942_object)

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

    set(KDA_JOURNAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../tests/kda_journal_gfx942_test.hip)
    set(KDA_JOURNAL_ELF ${CMAKE_CURRENT_BINARY_DIR}/kda_journal_gfx942.elf)
    add_custom_command(
        OUTPUT ${KDA_JOURNAL_ELF}
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
                "$ENV{PLOW_HIPCC}" "$ENV{PLOW_BUNDLER}" gfx942
                ${KDA_JOURNAL_ELF} k_journal_state 256 2
                -DKDAJ_DEVICE=1 -DPLOW_K3_SPEC_VERIFY=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${KDA_JOURNAL_SRC}
        DEPENDS ${KDA_JOURNAL_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_kda.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_elementwise.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
        VERBATIM)
    add_custom_target(kda_journal_gfx942_elf DEPENDS ${KDA_JOURNAL_ELF})
    set_source_files_properties(${KDA_JOURNAL_SRC}
        PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
    add_executable(kda_journal_gfx942_test ${KDA_JOURNAL_SRC})
    target_compile_features(kda_journal_gfx942_test PRIVATE cxx_std_17)
    target_include_directories(kda_journal_gfx942_test PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(kda_journal_gfx942_test PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(kda_journal_gfx942_test PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(kda_journal_gfx942_test PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(kda_journal_gfx942_test kda_journal_gfx942_elf)

    # Exact seven-query DSpark MLA semantic oracle. The device object contains both the
    # non-causal specialization and the ordinary causal control; the host requires the former to
    # match an independent f64 reference and the latter to diverge on the same committed prefix.
    set(K3_DSPARK_MLA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../tests/k3_dspark_mla_gfx942_test.hip)
    set(K3_DSPARK_MLA_ELF ${CMAKE_CURRENT_BINARY_DIR}/k3_dspark_mla_gfx942.elf)
    add_custom_command(
        OUTPUT ${K3_DSPARK_MLA_ELF}
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
                "$ENV{PLOW_HIPCC}" "$ENV{PLOW_BUNDLER}" gfx942
                ${K3_DSPARK_MLA_ELF} k_dspark_noncausal 256 1
                -DK3DSPARK_MLA_DEVICE=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../amd
                -I ${CMAKE_CURRENT_SOURCE_DIR}/../common
                ${K3_DSPARK_MLA_SRC}
        DEPENDS ${K3_DSPARK_MLA_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/../amd/op_attention.h
                ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipcc_hsaco.sh
        VERBATIM)
    add_custom_target(k3_dspark_mla_gfx942_elf DEPENDS ${K3_DSPARK_MLA_ELF})
    set_source_files_properties(${K3_DSPARK_MLA_SRC}
        PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
    add_executable(k3_dspark_mla_gfx942_test ${K3_DSPARK_MLA_SRC})
    target_compile_features(k3_dspark_mla_gfx942_test PRIVATE cxx_std_17)
    target_include_directories(k3_dspark_mla_gfx942_test PRIVATE "$ENV{ROCM_PATH}/include")
    target_compile_definitions(k3_dspark_mla_gfx942_test PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(k3_dspark_mla_gfx942_test PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(k3_dspark_mla_gfx942_test PRIVATE ${PLOW_AMDHIP64} m)
    add_dependencies(k3_dspark_mla_gfx942_test k3_dspark_mla_gfx942_elf)

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
