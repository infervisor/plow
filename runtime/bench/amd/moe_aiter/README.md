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


## Actual GLM capture and reusable weight packing

`captured.py` reads the checkpoint's TP8 expert slices and actual layer-40
activations/routing from a 70k-token prefill. Two captures cover the full
8192-token chunk at c0=57344 and the final 4464 live rows at c0=65536.
The baseline GLU is **bit-identical across all 16,777,216 / 9,142,272 captured
values**, after matching the runtime's OCP negative-zero canonicalization.
This checks the capture, gate/up order, expert routing and TP weight slices.

OCP-to-FNUZ conversion must preserve values near the format limit. Reinterpret
canonicalized OCP bytes as FNUZ and double the weight block scales. The runner
checks exact value equality across every weight. Casting OCP values directly
to FNUZ would overflow values above FNUZ's range. The checkpoint slice contains
4523 negative-zero bytes; matching the runtime's upload scrub is necessary for
plow's optimized CDNA3 weight staging too.

| Captured live rows | Plow, ms | AITER assembly, ms | AITER CK, ms | Assembly + reusable packing, ms |
|---:|---:|---:|---:|---:|
| 8192 | 4.833 | 1.971 | 2.371 | 2.647 |
| 4464 | 2.825 | 1.245 | 1.485 | 1.918 |

The full-chunk path is **45.2% faster including packing**, or 1.83x the baseline
throughput at this kernel boundary. Sampled FP32-oracle relative L2 is
**0.237% plow vs 3.80–3.85% AITER** for the full chunk and **0.237% vs
3.55–3.56%** for the tail. These are three sampled output rows, not model-level
quality or serving measurements. The standalone baseline reproduces the model's
GLU exactly, but its downstream combine excludes shared expert and TP work.
Across every output element, AITER differs from plow by 3.54–3.60% relative L2.
Resident and repacked assembly runs differ by 0.20–0.28% despite identical
packed weights. Do not assume the fused output preserves plow's deterministic
FP64 combine contract; that needs an explicit integration decision and test.

Keeping a shuffled copy of every layer would duplicate about 87.75 GiB/rank
for these expert weights, before scale storage. Instead, `plow_moe_pack`
accepts plow's per-expert weight/scale pointer tables and packs one layer into
**1,208,254,464 reusable bytes** (about 1.125 GiB). Weight shuffling and doubled
scales are included on every dispatch in the `aiter-repack` arm. Input
quantization, sorting, expert computation and reduction are included too.

The first pack loop took 1.285 ms. Assigning eight workgroups per expert and
hoisting the expert pointers outside each copy loop reduced this to about
0.674 ms. Both versions match every packed weight and scale byte against
AITER's shuffle. The final test also reverses the expert pointer tables and
restores them, checking exact packed bytes after each reuse. No additional
assembly was needed for this packing loop. The attention-independent MoE
assembly object remains the one named above.

The [capture result record](mi300x-captured-results.json) includes artifact,
checkpoint-slice and source hashes. All measurements held eight GPU leases and
used one GPU for the isolated MoE kernels. Captures used TP8 BF16 KV, native
AITER sparse attention and the existing B1 packet; the capture request uses
seed 0. They do not use the vLLM serving benchmark's random-token generator.

### Reproduce the capture

Use the B1 native sparse-MLA bundle from the
[MLA capture instructions](../mla_sparse_aiter/README.md), whose layer-40
post-attention/MoE work completes in segment 82. Set `CAPTURE` to a fresh
existing directory; snapshots use create-new semantics. Inside `nix develop`:

```sh
# C0=57344 for all 8192 rows; C0=65536 for the 4464-row tail.
export PLOW_PF_CAPTURE="8192@$C0:82:act.xn2=$CAPTURE/x.bin,act.tab=$CAPTURE/tab.bin,act.moe_fug=$CAPTURE/fu.bin,act.moe_meta=$CAPTURE/meta.bin,act.moe_rowtok=$CAPTURE/rt.bin,act.moe_rowpart=$CAPTURE/rp.bin,act.moe_rowgate=$CAPTURE/rg.bin,in.kvlen=$CAPTURE/len.bin"
PLOW_HSACO="$OBJECT_DIR" PLOW_MLA_PF_AITER=1 PLOW_MLA_PF_V2=1 \
PLOW_PF_CHUNK=8192 PLOW_PF_INTERLEAVE=0 \
  perf-data/tools/gpulease -n 8 moe-capture target/release/plowrt bench \
    --assets "$B1_ASSETS" --prefill-sweep --prefill-lengths 70000 \
    --prefill-reps 1 --prefill-warmups 0
unset PLOW_PF_CAPTURE
```

Build the library with the command at the top of this document. Then run each
arm with ROCm PyTorch/AITER, using `ROWS=8192` or `ROWS=4464` for its capture:

```sh
for arm in plow aiter ck aiter-repack; do
  perf-data/tools/gpulease -n 8 moe-captured env AITER_FLYDSL_FORCE=0 \
    python runtime/bench/amd/moe_aiter/captured.py --arm "$arm" \
      --library /tmp/moe-compare.so --capture "$CAPTURE" \
      --checkpoint /workspace/models/GLM-5.3-plow-lite --layer 40 --rank 0 \
      --rows "$ROWS" --out "/tmp/moe-captured-$arm.json"
done
```

Next integration gate: native HSA dispatch from an ordered, pure MoE segment,
with a reusable packed-weight workspace and correct shared-expert/TP combine.
The activation quantization change then needs full-model quality and serving
A/B measurements. This commit does not enable AITER MoE in production.
