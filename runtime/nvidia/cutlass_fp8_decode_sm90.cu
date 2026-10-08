// Small-M FP8 W8A8 decode projections on CUTLASS sm90 (swap-AB, as vLLM's small-M
// cutlass_scaled_mm): D[m][n] bf16 = sx[m] * sw[n] * sum_k X[m][k] W[n][k], X/W e4m3 row-major,
// sx per token, sw per output channel. Built twice from this file:
//   cubin:                  extern "C" plow_cutlass_fp8_<cfg>(__grid_constant__ Params)
//   host .so (-DPLOW_CUTLASS_HOST): plow_cutlass_fp8_prepare marshals that Params blob (TMA
//   descriptors through the driver's cuTensorMapEncodeTiled) and the launch geometry.
// Both halves must come from one build: plowrt checks each cfg's Params size against the cubin.
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>

namespace plow_cutlass {
using namespace cute;

// Swapped problem: M' = n (weight rows), N' = m (tokens); D is written column-major [n][m],
// i.e. row-major [m][n].
template <class TileShape, class KernelSchedule>
struct SwapGemm {
  using ElementA = cutlass::float_e4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;
  using Cluster = Shape<_1, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using ScaleW = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float,
                                                             Stride<Int<1>, Int<0>, Int<0>>>;
  using ScaleX = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float, float,
                                                             Stride<Int<0>, Int<1>, Int<0>>>;
  using Mul = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float,
                                                     cutlass::FloatRoundStyle::round_to_nearest>;
  using MulOut = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, float,
                                                        cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<
      MulOut, ScaleX,
      cutlass::epilogue::fusion::Sm90EVT<Mul, ScaleW, cutlass::epilogue::fusion::Sm90AccFetch>>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD, 8, ElementD,
      LayoutD, 8, EpilogueSchedule, EVT>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, 16, ElementA,
      LayoutB, 16, float, TileShape, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue>;
  using Params = typename GemmKernel::Params;

#ifdef PLOW_CUTLASS_HOST
  static typename GemmKernel::Arguments arguments(int m, int n, int k, const void* x,
                                                  const void* w, void* d, const float* sx,
                                                  const float* sw) {
    typename GemmKernel::StrideA sa{};
    typename GemmKernel::StrideB sb{};
    typename GemmKernel::StrideD sd{};
    cute::get<0>(sa) = k;
    cute::get<2>(sa) = int64_t(n) * k;
    cute::get<0>(sb) = k;
    cute::get<2>(sb) = int64_t(m) * k;
    cute::get<1>(sd) = n;
    cute::get<2>(sd) = int64_t(m) * n;
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {n, m, k, 1},
        {reinterpret_cast<const ElementA*>(w), sa, reinterpret_cast<const ElementA*>(x), sb},
        {{}, nullptr, sd, reinterpret_cast<ElementD*>(d), sd}};
    args.epilogue.thread = {{sx}, {{sw}, {}, {}}, {}};
    args.hw_info.sm_count = 132;
    return args;
  }
#endif
};

using FastWs = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
using FastPp = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
using PromWs = cutlass::gemm::KernelTmaWarpSpecialized;
using PromPp = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
// cfg = 3 * promoted + (m <= 32 ? 0 : m <= 64 ? 1 : 2); H100 graph-timed picks per M class.
using Cfg0 = SwapGemm<Shape<_64, _32, _256>, FastWs>;
using Cfg1 = SwapGemm<Shape<_64, _64, _256>, FastWs>;
using Cfg2 = SwapGemm<Shape<_64, _64, _256>, FastPp>;
using Cfg3 = SwapGemm<Shape<_64, _32, _128>, PromWs>;
using Cfg4 = SwapGemm<Shape<_64, _64, _128>, PromWs>;
using Cfg5 = SwapGemm<Shape<_64, _64, _128>, PromPp>;
}  // namespace plow_cutlass

#define PLOW_CUTLASS_CFGS(X) \
  X(0, Cfg0) X(1, Cfg1) X(2, Cfg2) X(3, Cfg3) X(4, Cfg4) X(5, Cfg5)

#ifndef PLOW_CUTLASS_HOST
#define PLOW_KERNEL(id, C)                                                                   \
  extern "C" __global__ void __launch_bounds__(                                              \
      plow_cutlass::C::GemmKernel::MaxThreadsPerBlock,                                       \
      plow_cutlass::C::GemmKernel::MinBlocksPerMultiprocessor)                               \
      plow_cutlass_fp8_##id(CUTLASS_GRID_CONSTANT plow_cutlass::C::Params const params) {    \
    extern __shared__ char smem[];                                                           \
    plow_cutlass::C::GemmKernel op;                                                          \
    op(params, smem);                                                                        \
  }                                                                                          \
  extern "C" __device__ unsigned plow_cutlass_fp8_params_bytes_##id =                        \
      sizeof(plow_cutlass::C::Params);
PLOW_CUTLASS_CFGS(PLOW_KERNEL)
extern "C" __device__ unsigned plow_cutlass_fp8_abi = 1;
#else
template <class C>
static int prepare(int m, int n, int k, const void* x, const void* w, void* d, const float* sx,
                   const float* sw, void* out, unsigned cap, unsigned* geom) {
  auto args = C::arguments(m, n, k, x, w, d, sx, sw);
  if (!C::GemmKernel::can_implement(args)) return -1;
  if (C::GemmKernel::get_workspace_size(args) != 0) return -2;
  auto params = C::GemmKernel::to_underlying_arguments(args, nullptr);
  if (sizeof(params) > cap) return -3;
  memcpy(out, &params, sizeof(params));
  dim3 grid = C::GemmKernel::get_grid_shape(params);
  dim3 block = C::GemmKernel::get_block_shape();
  geom[0] = grid.x;
  geom[1] = grid.y;
  geom[2] = grid.z;
  geom[3] = block.x * block.y * block.z;
  geom[4] = C::GemmKernel::SharedStorageSize;
  return int(sizeof(params));
}

// geom: grid x, y, z, block threads, dynamic smem bytes. Returns the Params size, < 0 = reject.
extern "C" int plow_cutlass_fp8_prepare(int cfg, int m, int n, int k, const void* x,
                                        const void* w, void* d, const float* sx,
                                        const float* sw, void* out, unsigned cap,
                                        unsigned* geom) {
#define PLOW_CASE(id, C) \
  case id:               \
    return prepare<plow_cutlass::C>(m, n, k, x, w, d, sx, sw, out, cap, geom);
  switch (cfg) {
    PLOW_CUTLASS_CFGS(PLOW_CASE)
    default:
      return -4;
  }
}
extern "C" unsigned plow_cutlass_fp8_host_abi = 1;
#endif
