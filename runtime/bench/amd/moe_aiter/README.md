# GLM TP8 grouped MoE: plow and AITER

This compares the emitted per-rank shape: H=6144, expert I=256, E=256, top-k=8.
It calls plow's actual grouped align/GLU/down/combine bodies and AITER's fused
MoE entry point on the same synthetic inputs. It is a kernel screening tool,
not a serving implementation.

Inside `nix develop`, with ROCm PyTorch and `amd-aiter==0.1.19` available:

```sh
bench=runtime/bench/amd/moe_aiter
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC \
  -Iruntime/amd -Iruntime/common -c "$bench/kernels.hip" -o /tmp/moe-compare.o
c++ -shared /tmp/moe-compare.o -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o /tmp/moe-compare.so
for arm in plow aiter ck; do
  perf-data/tools/gpulease -n 1 moe-compare env AITER_FLYDSL_FORCE=0 \
    python "$bench/compare.py" --arm "$arm" --library /tmp/moe-compare.so \
    --out "/tmp/moe-$arm.json" --rows 128 2048 8192
done
```

The measured environment is MI300X gfx942, ROCm 7.14 compiler/runtime, Torch
`2.12.0+git6bbd260` (HIP build `7.2.53211`). The local Python command was
`/app/plow/build-gemma31/vllm-python` with
`VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib`.

## Matched inputs and timing boundary

- Seed 31; BF16 normal activations; exactly representable FP8 integer weights
  in [-8,8], with independently varying scales per 128x128 block. Routing has
  eight unique experts per token and normalized positive weights.
- AITER receives FNUZ FP8; plow receives numerically converted OCP FP8. Weight
  values and block scales match. This is not a reinterpretation of FP8 bytes.
- Plow uses eight waves, epilogue hoisting and deterministic FP64 accumulation,
  matching the measured production object's flags. Its adapter includes the
  accumulator clear, four align phases, GLU, down and combine.
- AITER includes activation quantization, sorting, expert computation and
  reduction. Weight conversion/preshuffling and router top-k are outside timing
  in both arms. Shared expert, residual, TP communication and packet protocol
  are excluded in both arms.
- GPU graph/event timing uses seven samples after warmup and reports the
  median. Allocations, JIT and oracle evaluation are outside timing.
- The independent FP32 oracle dequantizes each selected expert with every block
  scale, applies SiLU and routing weights, and checks first/middle/last tokens.
  All outputs must be finite. Plow's screening threshold is 1% relative L2;
  the activation-quantized AITER arms use 10%. These thresholds are explicitly
  **not** a model-quality acceptance gate.

## Measured results

The installed heuristic chose one-stage assembly at all three row counts.
`--arm ck` disables that candidate before the first dispatch lookup to measure
the two-stage CK path. This changes only the benchmark process.

| Query rows | Plow ms | AITER assembly ms | AITER CK ms |
|---:|---:|---:|---:|
| 128 | 0.910 | 0.317 | 0.485 |
| 2048 | 1.574 | 0.771 | 0.847 |
| 8192 | 4.732 | 2.640 | 2.439 |

At 8192 rows, CK is **1.94x faster** than the plow adapter. Assembly wins at
the smaller row counts. Sampled relative L2 error is 0.23–0.24% for plow and
4.06–4.37% for AITER. The latter includes FP8 activation quantization absent
from plow's W8A16 path; no full-model quality conclusion follows.

The measured assembly object is
`fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co`; the CK implementation
loaded `module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2.so`.
Recorded environment, library hashes and all nine cells are in
[mi300x-results.json](mi300x-results.json). No third-party binary is vendored.

Adapting these kernels requires activation block quantization, scale/weight
layout conversion, native launch boundaries and actual-model quality checks.
The result supports that work; it does not establish that rewriting plow's
existing inner loop in assembly would yield the same gain.
