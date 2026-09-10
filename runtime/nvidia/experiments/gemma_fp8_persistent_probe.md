# Opt-in FP8-weight / BF16-activation decode MMA

`PLOW_NV_FP8_DECODE_MMA=1` selects the register-only helper in
`op_gemv_fp8_mma.cuh` through the actual `d_gemv_fp8` and `d_gemv_glu_fp8`
dispatchers. Default is zero. Eligibility is Hopper, 256 threads, M≥8 for GEMV
or M≥16 for GLU, and a positive K divisible by 256. Other shapes retain the
existing FFMA path. The GLU threshold preserves the faster FFMA route at M=8.

Each instruction slice retains `[slice*ceil(N/nblk), min((slice+1)*ceil(N/nblk),N))`
output ownership. Warps tile that range in eight-column groups and requests in
16-row groups. It needs no allocation, additional launch, barrier, shared-memory
arena, or cross-CTA reduction. Sparse and padded rows retain their existing
runtime layout. FP8 Q/K/V projections already use separate GEMV instructions.

E4M3 weights convert exactly to BF16; activations remain BF16. FP32 channel
scales stay after the reduction. Fused GLU keeps gate and up in FP32 through the
existing SiLU/GELU helper, rounding only the final output. Accumulation order
changes, so model qualification is required despite unchanged operand precision.

This preserves the interpreter's scheduling contract and therefore omits the
standalone experiment's cross-CTA split-K optimization. Its speedups must be
measured separately.

Build the dispatcher probe:

```sh
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -arch=sm_90a -O3 -std=c++17 -I runtime/common -I runtime/nvidia \
  runtime/nvidia/experiments/gemma_fp8_persistent_probe.cu -o /tmp/gemma_fp8_persistent_probe
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
  /tmp/gemma_fp8_persistent_probe
```

Optional `M N K blocks` selects one timed shape; the ownership/fallback tests
always run first. The control calls the same production FFMA templates directly
with `GV_MM_MAX=16`. Cases check untouched output guards and exact instruction
slice boundaries, including empty slices, M=1/4 fallbacks, M=8/17/32, K=264
fallback, and both GLU activations. Timing uses a 256MiB cache flush, four
warmups and 15 CUDA-event repetitions.

The initial actual Gemma FP8 B16 decode cubin builds successfully. Interpreter
register count changes from 188 to 189; stack and local memory remain zero,
static shared memory remains 1040 bytes. Rebuilding with the flag absent is
byte-identical to the qualified default cubin (SHA-256
`85e9f293f60d4d1228844fb52377441412fbe27bc27de8eccb21bf59d03a95c4`).

The first candidate (GLU also enabled at M=8) passed 90 GPU checks. Its maximum
full-output relative L2 difference was 0.000433. Compute Sanitizer reported zero
errors for the ownership/fallback cases and M=16/N=8192/K=5376. An isolated M=8
GELU GLU regressed from 364.416 to 413.696µs, motivating the final threshold.

Final-candidate model decode-step screening used the same frozen `step-bench-profile`
runtime for both assets, synthetic 1K prompts, 16 warmups and 64 timed steps:

| Active slots | Baseline mean ms | Final candidate mean ms |
|---:|---:|---:|
| 1 | 20.983 | 20.954 |
| 8 | 92.263 | 85.761 |
| 16 | 109.139 | 107.570 |

No CPU builds or other GPU jobs overlapped. These are 7.0% and 1.4% latency
reductions at 8 and 16 slots; standalone kernel gains did not transfer proportionally.
The first candidate took 89.674ms at 8 slots, so retaining FFMA for M=8 GLU
improved the final result. The final dispatcher probe again passed all 90 checks,
and its memcheck run reported zero errors.
This frozen helper predates the latest packed-serving changes, and these are
decode-only measurements, not serving measurements. The packet's Q/K/V
instructions use 66/33/33 slices, unlike the initial 132-CTA kernel screen.

The final threshold also passes 256 full-vocabulary frames on two natural
prompts (1K/16K), teacher-forced independently at physical slots 7 and 15.
All four initial prefill frames are bit-exact. Top-1 agrees for all 256 frames,
but logits differ: maximum absolute difference 1.078125, minimum cosine
0.9994801, maximum relative L2 0.0347104. This is numerical screening, not a broad
model-quality or serving qualification. The final cubin SHA-256 is
`fb73ac0e1f83363e838344fd799a37b3b9983674d14362c0b4559988f11af203`.

Serving, concurrency and broad model-quality qualification remain pending.
Build command and logs
are under campaign directory
`/opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/` as
`gemma-fp8-persistent-{cubin-command.json,cubin-build.log,default-build.log,probe-build2.log}`.
First-candidate assets are `fp8-b16-c1024-persistent-mma-h100`; final-threshold
assets are `fp8-b16-c1024-persistent-mma2-h100`. Only the decode cubin differs
from the qualified FP8 packed-prefill OCC1 assets. Final-candidate step logs and
results are `gemma-fp8-persistent2-step-{baseline,candidate}-b{1,8,16}.log` and
`gemma-fp8-persistent2-step-results.json`. Final probe and memcheck logs are
`gemma-fp8-persistent-{probe2,memcheck2}.log`; model comparison is
`fp8-mma2-model-logits-comparison.json`.
