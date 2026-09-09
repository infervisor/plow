# GLM sparse MLA with FP8 KV on MI300X

GLM sparse FP8 KV now has an opt-in MI300X runtime path, including batched
decode and native AITER prefill. TP8 batch 16 passes the retrieval screen.
The serving screen gains 4.7% throughput but regresses mean TPOT by 49.6%;
this is a capacity option, not a new default. No speculative decoding is involved.

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
The runtime integration below preserves native AITER prefill by dequantizing
FP8 latent rows once into its BF16 workspace. Simply enabling the interpreter
FP8 path regresses prefill in the captured attention measurement above.

## Runtime qualification

The [runtime record](mi300x-runtime-results.json) contains the serving metrics,
per-request lengths, retrieval responses and artifact hashes. The earlier
[template record](mi300x-results.json) remains a separate kernel experiment.

Opcodes 109/110 retain scales in `t7`; `j0 = selected_handle + 1` selects sparse
indices/union, and zero keeps dense semantics. The AMD loader validates handles,
capacities, qualified gfx942 QH8 geometry and new object markers, including
smaller decode rungs. Sparse prefill must route through a pure V2 segment.
The final release emitter reproduces the benchmarked B16 packet byte-for-byte.
CUDA, packed prefill and unqualified batched dense GLM FP8 remain refused.

The native adapter adds an FP8 pack entry while preserving the BF16 ABI and
pinned AITER assembly. It dequantizes each live latent row into BF16, copies
rope and queries, then uses the existing two-split attention and FP32 reduction.
Scratch remains 438,961,152 bytes/rank. The HSA test checks BF16/FP8, two slots,
rows 1/129, varying row scales, exact packed FP8 values and attention output.

TP8 batch 16 passes **18/18** retrieval cases at concurrency 16, prompt lengths
5433–5438 and 68797–68802, depths 0.1/0.5/0.9 and three facts. Both the earlier
BF16 B8 screen and this screen pass all cases; 14/18 responses are text-identical.
Concurrency changes too, so this is not an isolated quantization comparison or
broad model accuracy evaluation.

| 20-request C20 serving screen | BF16 B8 | FP8 B16 |
|---|---:|---:|
| Successful / failed | 20 / 0 | 20 / 0 |
| Duration, s | 489.35 | 467.39 |
| Output tok/s | 28.19 | 29.51 |
| Mean TTFT, s | 195.14 | 158.19 |
| Mean TPOT, ms | 218.68 | 327.23 |
| Median ITL, ms | 116.60 | 161.28 |
| P99 ITL, ms | 1685.00 | 1778.82 |

Both screens use identical per-request input/output lengths: 1,414,538 input
and 13,795 output tokens. FP8 B16 gives **+4.70% throughput**, **−18.94% mean
TTFT**, and **+49.64% mean TPOT**. This single before/after changes KV format
and decode capacity together; it is not a repeated A/B or isolated kernel gain.
It uses 20 requests, not the H200 reference's 100. The reference's 273.67 output
tok/s remains unmet. Its aggregate includes speculative decoding and its server
sampling/topology are unspecified; plow uses device argmax and no speculation.

### Build and serve

Run inside `nix develop`. Start from the qualified BF16 sparse-serving object
set, then build the FP8 families in **separate directories** so each audit keeps
its own `build_defines.json`. Overlay their ELF files into `OBJECT_DIR`:

```sh
PLOW_DECODE_BATCH=16 PLOW_DSA_PF=1 PLOW_ROWS_ONLY==interp_decode_fp8kv \
  scripts/build_gfx942.sh "$BUILD/decode"
PLOW_DECODE_BATCH=16 PLOW_DSA_PF=1 PLOW_ROWS_ONLY==interp_prefill_fp8kv_mla_moe \
  scripts/build_gfx942.sh "$BUILD/prefill"
PLOW_DECODE_BATCH=16 PLOW_DSA_PF=1 PLOW_ROWS_ONLY==interp_flash_fp8kv \
  scripts/build_gfx942.sh "$BUILD/flash"
for rung in 1 2 4 8; do
  PLOW_DECODE_BATCH=16 PLOW_DECODE_TIER="$rung" PLOW_DSA_PF=1 \
  PLOW_ROWS_ONLY==interp_decode_fp8kv \
    scripts/build_gfx942.sh "$OBJECT_DIR/lowrung$rung"
done
scripts/build_mla_sparse_aiter.sh "$OBJECT_DIR" "$AITER_QH8_OBJECT"
```

`PLOW_DECODE_TIERS` currently rebuilds BF16 decode families; use the explicit
FP8 loop above. The loader enforces `plow_mla_sparse_fp8_decode_arm`,
`plow_mla_sparse_fp8_prefill_arm` and the FP8 adapter ABI marker. The manifest
names this capability `PLOW_MLA_SPARSE_FP8=1`; it is not a separate build switch.
The decode/flash axes differ from the stored static baseline, so their audit
reports do not establish unchanged baseline resource budgets.

Use the previous all-layer sparse TP8 `build.json` as `REPLAY`:

```sh
PLOW_VERIFY_BIN=lean-plow/.lake/build/bin/plow_verify GLM_FULL=1 \
PLOW_MLA_PF_AITER=1 PLOW_MLA_PF_V2=1 PLOW_UNISEG=0 PLOW_GLM_DSA_PF_SPAN=3 \
  target/release/plowc --hf-dir /workspace/models/GLM-5.3-plow-lite \
    --emit devblob --gpu MI300X --arch gfx942 --max-ctx 81920 --num-gpus 8 \
    --replay-knobs "$REPLAY" --glm-dsa 1 --glm-fuse-rope=false \
    --glm-fp8-kv=true --emit-decode-batch-ladder 1,2,4,8,16 --out "$ASSETS"
```

Under an eight-GPU lease, serve with:

```sh
PLOW_HSACO="$OBJECT_DIR" PLOW_MLA_PF_AITER=1 PLOW_MLA_PF_V2=1 \
PLOW_PF_CHUNK=8192 PLOW_PF_INTERLEAVE=0 \
  target/release/plowrt serve --assets "$ASSETS" --port 8080
```

Use the [existing retrieval client](../mla_sparse_aiter/quality.py) with
`--concurrency 16`. The serving client is the supplied vLLM command with
`--num-prompts 20`, this server address and model `glm-5.3-plow-lite`.
Assert 20 completed, zero failed and nonzero output in the saved JSON; the
client can exit successfully even when requests fail.

## Reproduce template tests

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
