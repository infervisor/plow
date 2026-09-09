# GLM sparse MLA with FP8 KV on MI300X

This qualifies existing device templates, not FP8 serving. GLM's emitter still
refuses batched FP8 KV and sparse FP8 prefill. Packet operands, object capability
checks and the native AITER packing adapter need integration before those
refusals can be removed. No speculative decoding is involved.

## Correctness finding

The V2 `GATHER=true, FP8=true` body gathered latent/rope rows through the union
table but loaded scales by the union entry number. Fix both softmax variants
and the non-deferred PV scale load to use the selected cache row. Inactive
union entries use a safe scale address. The BF16 and dense FP8 expressions
are unchanged after template specialization.

On identical synthetic T129 inputs, sampled FP8 relative L2 error against
FP32 attention over the **quantized** cache falls from **1.543 to 0.001557**.
Tests cover T1/129/4464, nonidentity selections, varying row scales, ragged
query/union tails, poisoned inactive scales, and deferred/non-deferred softmax.

Decode passes 24 cells: B1/2/4/8/16/20, one or 16 splits, uniform 70k contexts
and mixed live lengths including 0/1/129/2047/2048/65537/70000/79800.
FP8 kernel relative L2 error is at most **8.10e-7**. Index tails are poisoned;
logical work slices are permuted. Cache-writer tests verify slot positions,
ring wrapping, rung 1 in an eight-slot allocation, zero rows and three repeated
writes. Every untouched cache byte/scale retains its sentinel. One quantized
element differs at an FP8 rounding midpoint because of FP32 reduction order;
the test checks proximity to that midpoint instead of allowing general error.

Plow uses OCP `float8_e4m3fn` (448 maximum), including on gfx942. Do not substitute
the hardware's `float8_e4m3fnuz` encoding. The CDNA3 writer canonicalizes -0 to +0.

## Measurements

The [record](mi300x-results.json) includes source/library hashes and all cells.
Final runs held all eight GPU leases and used one GPU. Times are medians of
11 HIP graph replays and cover attention only: no quantization, union building,
merge, interpreter dispatch, HSA adapter, TP communication or serving scheduler.

| Decode, all slots at 70k, 16 splits | BF16, ms | FP8, ms |
|---|---:|---:|
| B1 | 0.03970 | 0.04242 |
| B8 | 0.04110 | 0.04358 |
| B16 | 0.09907 | 0.09094 |
| B20 | 0.14590 | 0.13428 |

FP8 is slightly slower at small batches and about 8% faster at B16/20 in these
cells. Synthetic quantization differences are roughly 4–5% relative L2.

The actual GLM capture is layer 77 using layer 74's selection, TP8, 70k prompt,
T8192 bucket, chunk start 65536 and 4464 live rows. Capture hashes match the
[previous sparse MLA record](../mla_sparse_aiter/mi300x-model-results.json).
Five sampled queries, all eight heads, give **0.000718** kernel relative L2
against FP32 with quantized KV, and **0.01321** against the original BF16 KV.
Attention takes **1.901 ms BF16 vs 2.414 ms FP8**. This is one captured layer,
not a model quality score or a serving speedup.

At ctx81920, 78 latent/rope caches plus 21 BF16 indexer caches require:

| Slots | BF16 KV, GiB/rank | FP8 latent + BF16 rope, GiB/rank |
|---|---:|---:|
| 8 | 58.13 | 33.94 |
| 16 | 116.25 | 67.88 |
| 20 | 145.31 | 84.85 |

These are allocation formulas, excluding weights, activations, workspaces and
allocator overhead. They do not establish that B20 serving fits or is faster.
The next integration should preserve the qualified native AITER prefill path
by packing/dequantizing FP8 latent rows once, then measure full-model quality
and serving. Simply enabling the interpreter FP8 path regresses prefill here.

## Reproduce

Build inside `nix develop`, once for each softmax variant:

```sh
DEFER=1 # also test 0
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 -fPIC \
  -DFA_MLA_PF2_DEFER="$DEFER" -Iruntime/amd -Iruntime/common \
  -c runtime/bench/amd/mla_fp8_kv/kernels.hip -o /tmp/mla-fp8.o
c++ -shared /tmp/mla-fp8.o -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o /tmp/mla-fp8.so
```

Run with a ROCm-enabled PyTorch installation under a GPU lease:

```sh
perf-data/tools/gpulease -n 8 mla-fp8 python \
  runtime/bench/amd/mla_fp8_kv/compare.py \
  --library /tmp/mla-fp8.so --out /tmp/mla-fp8-decode.json
perf-data/tools/gpulease -n 8 mla-fp8 python \
  runtime/bench/amd/mla_fp8_kv/prefill.py \
  --library /tmp/mla-fp8.so --out /tmp/mla-fp8-prefill.json
```

For the recorded GLM capture, add `--rows 4464 --capture "$CAPTURE" --scale 0.0625`
to the prefill command. The capture must contain the six files described by
the existing sparse MLA benchmark. Reproducing the old defect requires building
against `op_attention.h` at `f5f15dd2` and adding `--record-failure`.

## External compatibility references

[vLLM's KV quantization documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/quantized_kvcache.md)
describes FP8 types and scale policies. They are not a drop-in ABI for plow's
split latent/rope layout and per-row scales.
[AITER's MLA dispatch](https://github.com/ROCm/aiter/blob/main/aiter/mla.py)
selects kernels by architecture, shape and format; retain the already qualified
gfx942 QH8 object rather than inferring compatibility from another FP8 variant.
The broader AITER/hipBLASLt/ROCm review and pinned assembly integration remain in
[the branch report](../../../../docs/amd/tp-bringup-mi300x.md#11-aiter-hipblaslt-rocm-and-vllm-kernel-review-2026-09-09).
