# Native SM90 BF16 decode projections

`gemv_sm90_transposed.cu` exports the body shared with the Gemma decode probe
through `op_gemv_transposed.cuh`. It does not link cuBLASLt or CUTLASS. Runtime
packet selection is not yet integrated.

Build from the repository root, under `nix develop`:

```sh
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -std=c++17 -O3 -arch=sm_90a -cubin -Xptxas=-v \
  -I runtime/common -I runtime/nvidia \
  runtime/nvidia/gemv_sm90_transposed.cu -o /tmp/gemv_sm90_transposed.cubin
```

`PLOW_GEMV_TRANSPOSE_SWIZZLE=1` selects an experimental XOR layout. The default
uses padded rows. Select by measured shape; the XOR layout is not uniformly
faster or free of measured shared-load bank conflicts.

ABI marker `plow_gemv_transposed_abi=1`, projection block size marker 128.
Entry names are `plow_gemv_bf16_m{8,16}_bk{128,256}_s{3,2}`: BK128 has three
stages; BK256 has two. Parameters:

```
(bf16* output, float* partials, const bf16* input, const bf16* weight,
 int M, int N, int K, int splits)
```

Contract:

- H100/sm_90a, BF16 row-major input `[M,K]`, weight `[N,K]`, output `[M,N]`.
- `1 <= M <=` the entry's M capacity; positive N and K; K divisible by 8.
  Input and weight bases must be aligned to 16 bytes. N may have a tail.
- Splits 1/2/4/8. Grid `(ceil(N/64), splits, 1)`, block `(128,1,1)`.
- Dynamic shared memory at least `stages*(64+capacity)*(BK+8)*2` bytes works
  for both layouts. Set the function's dynamic-shared-memory limit before
  launch. Output/inputs/partials must not overlap.
- Split 1 writes BF16 output directly; the partial pointer is unused.
  Larger splits overwrite `[splits,M,N]` FP32 partials without zeroing.
  Then launch `plow_gemv_bf16_reduce(output, partials, M*N, splits)` on the
  same stream with 256 threads and `ceil(M*N/256)` blocks.
- Caller validates allocation extents and integer products. Completion of
  the reduction precedes consumers. These kernels do not publish packet
  counters; an ordered graph/segment route must establish that dependency.
- Object hashes, scratch ownership and graph lifetime must be bound by the
  runtime before enabling a packet route. The standalone probe is not a
  substitute for that validation.

Pass a cubin as the fifth probe argument after `gemma4` to test actual driver
entry points, e.g. `decode_tc_driver_probe 11 0 -1 gemma4 OBJECT.cubin`.
The probe uses cuBLASLt only as a comparison and checks sampled FP64 dots
independently. Evidence is recorded in the Gemma H100 native tuning report.
