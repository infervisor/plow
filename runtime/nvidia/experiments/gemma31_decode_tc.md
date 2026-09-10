# Gemma 31B BF16 decode tensor-core experiment

`gemma31_decode_tc.cu` is a standalone H100 experiment. It does not change
production dispatch, packet formats, or defaults.

The candidate puts weight rows in MMA's `m16` dimension and requests in its
`n8` dimension. At batch 4/8 this halves the number of MMA instructions versus
the existing activation-first `m16` layout. Batch 16 uses two request fragments.
Weights stay in their original row-major BF16 layout. It uses asynchronous
shared-memory staging, FP32 accumulation, direct BF16 output at split 1, and
ordered FP32 partial reduction at larger splits.

Build and run from the repository root:

```sh
nix develop -c /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_90a \
  -I runtime/common -I runtime/nvidia -Xptxas=-v \
  runtime/nvidia/experiments/gemma31_decode_tc.cu -lcublasLt -lcuda \
  -o /tmp/gemma31-decode-tc
nix develop -c /tmp/gemma31-decode-tc 31 8
```

Arguments are repetitions, optional batch (`0`, `1`, `2`, `4`, `8`, `16`), optional
shape index (`-1` = all), optional `gemma4`, and optional native cubin path.
Gemma 31B uses shape indices 0–8 and defaults to batches 4/8/16. `gemma4`
uses indices 0–9 and defaults to batches 1/2/4/8/16. The last shape is the
ragged `N83/K136` guard.
The CUDA loader must resolve the real driver, without toolkit `stubs` entries
in `LD_LIBRARY_PATH`.

## Controls and verification

- Production `d_gemv` is included directly, with `GV_MM_MAX=8`, at 132 and
  528 blocks. Its isolated wrapper uses 111 registers without spills; this
  does not model the interpreter's register pressure or scheduling cost.
- Existing `d_gemm_splitk<16,64,128,4,3>` runs with 528 physical blocks,
  including zeroing and BF16 conversion in its measured time.
- cuBLASLt selects the fastest of up to 16 heuristic algorithms using seven
  separate cold repetitions, then runs the same measured 31 repetitions.
- Each measured sample first reads an unrelated 192 MiB eviction buffer.
  CUDA events exclude eviction and include all projection/reduction launches.
- Inputs are deterministic synthetic BF16 values. Every output is checked
  for finiteness and compared with cuBLASLt. For every shape, 128 sampled
  outputs are checked against an FP64 CPU dot product with the predeclared
  bound `abs_error <= 0.0001 + abs(reference)/128`.
- Two consecutive executions must produce identical candidate BF16 outputs.
  The ragged guard covers K and N tails and empty split slices. Compute
  Sanitizer memcheck reported zero errors at batches 4, 8, and 16.

## H100 screening result, 2026-09-09

All 24 dense shape/batch cells passed the sampled numerical bound and the
candidate repeatability check. The fastest candidate beat native GEMV and
the existing split-K control in every cell. cuBLASLt was faster in every
dense cell. These are selected configurations from one screening sweep,
not independent tuning/holdout evidence or an end-to-end serving result.

Batch 8 medians, microseconds:

| N | K | Native GEMV, 132 blocks | Existing TC, best split | Candidate, best tile/split | cuBLASLt |
|---:|---:|---:|---:|---:|---:|
| 8192 | 5376 | 97.664 | 45.824 | 35.360 | 32.704 |
| 4096 | 5376 | 56.256 | 28.384 | 25.024 | 21.344 |
| 16384 | 5376 | 182.368 | 80.768 | 65.760 | 61.312 |
| 2048 | 5376 | 34.720 | 20.192 | 15.808 | 14.304 |
| 21504 | 5376 | 232.416 | 96.896 | 87.328 | 79.040 |
| 5376 | 8192 | 112.864 | 42.944 | 38.208 | 33.920 |
| 5376 | 16384 | 291.040 | 74.560 | 69.952 | 60.512 |
| 5376 | 21504 | 429.984 | 93.664 | 88.928 | 81.376 |

The candidate uses a standalone grid of `ceil(N/64) * splits`, unlike the
interpreter's persistent grid. Integrating it requires measuring that execution
form and its dependency/reduction costs. Arithmetic is not bit-exact to native
GEMV or cuBLASLt for the dense shapes. Full-model quality, rung transitions,
and serving qualification remain necessary before production use.

Raw logs, build output, binary, and hashed summary are under
`/opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/`:

- `gemma31-decode-tc-{b4,b8,b16,tail,memcheck}.log`
- `gemma31-decode-tc-build3.log`
- `gemma31-decode-tc-results.json`
- `bin/gemma31-decode-tc`
