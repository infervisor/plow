# Gemma 4 12B prefill review — evidence and open questions

This review separates measured Gemma3 history from the new Gemma4 target. The
Gemma4 checkpoint is pinned to707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; its
23,919,549,408-byte safetensors file matches the repository's SHA256
5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d.
Initial Gemma4 serving screens are complete through16K/C16 for native Plow and
through16K/C128 for vLLM BF16. They show a substantial remaining gap; these are
one-repeat screens, not final repeated measurements.

## Findings by layer

| Layer | Evidence | Interpretation |
|---|---|---|
| Compiler | Earlier Gemma3 slicing duplicated decode work; fixed and regression-tested. Current Gemma4 mixed packet cannot combine B128 decode with PF128; maximum prefill chunk is8192. cuBLASLt decode conversion accepts B<=32. | Real correctness/coverage constraints exist. They are not measured shares of the remaining latency. |
| Runtime | Earlier Gemma3 `token_batch=true` still reported ready=false/fires=false. New Gemma4 BF16 run reports ready=true and fires=true, and435-node prefill graph capture. | Configured flags are insufficient evidence of active fast paths. The terminal contract extension now permits this model's route. |
| Interpreter/object | Gemma3 W8A16 native-body speedup1.42–1.55x initially regressed inside FAT; dedicated object helped. Register cap alone failed until shared-memory reservation was reduced132160→99376bytes. | Object compilation and resource coupling can defeat a fast isolated kernel. This is a demonstrated mechanism, not a quantified Gemma4 loss. |
| Kernels | Dedicated attention and GEMM occupancy changes materially reduced Gemma3 TTFT. Gemma4 shape-matched cuBLASLt/attention measurements remain pending. | Kernel selection matters; we cannot label every custom kernel efficient from its name or the old model's results. |
| Serving policy | Previous comparisons used physical B16 with C128 queued and smaller prefill token budgets than vLLM. | Queueing, packing, memory capacity and scheduling can dominate concurrency results. Both latency and throughput policies need separate tuning. |

## Actual Gemma4 packets and runtime

The new packet audit example parses the same DevBlob representation as the
runtime and dumps all instructions, tensor names, immediates and segment windows.
The native audit packet has PF128/512/1024/2048/4096/8192 and
decode B1/2/4/8/16/32/64. Its declared KV extent is340GiB; it is an inspection
artifact, not a configuration qualified to run on H10080GB.

Each prefill rung has329 GEMMs (328 projections plus head),48 attention ops,
144 HeadNormRope,96 NormResidual,97 RMSNorm,48 GLU and the final sampling ops.
Native decode has113 GEMV,40 fused QKV GEMV,48 fused GLU GEMV per rung, plus
attention/norm/sampling. Emitted opcodes alone do not identify the executed body.

The B16 serving packet instead has329 GEMV opcodes per decode rung. Runtime
confirms328 projection routes through cuBLASLt, with eight distinct plans per
rung. Calling these328 operations slow custom GEMV solely from the packet
opcode would be incorrect. Prefill uses the dedicated384-thread GEMM object,
197696bytes shared memory, grid132, and dedicated attention. GPU memory loaded
is43.53GiB; physical capacity16, ingress capacity128. The5913-token retrieval
smoke returns the expected code, but reference logit parity is still pending.

## Comparator and architecture assessment

Installed vLLM0.29 source implements Gemma4 with fused QKV and merged gate/up
linears. Its global K=V implementation loads K into both K/V slots. Consequently,
comparisons must include equivalent fused groups and total layer cost, not just
identically named individual matrices. Its BF16 linear implementation dispatches
through `dispatch_unquantized_gemm`; the actual selected CUDA kernels must be
captured from the running model rather than assumed to be cuBLAS.

vLLM's documented chunked-prefill policy prioritizes decode and fills the token
budget with prefill; token budget changes the latency/throughput tradeoff:
https://docs.vllm.ai/en/stable/configuration/optimization/

NVIDIA documents GEMM tiling, software pipelining and the register/occupancy
tradeoff, including Hopper warp specialization:
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html

The evidence supports specific implementation and integration weaknesses, not
the conclusion that Plow's entire architecture cannot compete. A persistent
interpreter that couples unrelated kernels' resource requirements is a credible
architectural risk; the dedicated-object results demonstrate a way to reduce
that coupling within the existing design. Shared CPU/AMD/NVIDIA orchestration
does not require identical hardware kernel schedules.

We have also spent too much effort tuning individual flags before completing
the end-to-end attribution and verifying feature activation. This review changes
the sequence: correctness and dispatch proof, then attribution, then optimization.

## Measurements required before assigning blame

1. Establish same-checkpoint Gemma4 reference correctness and vLLM baselines.
2. Measure queue/CPU preparation/host submission/GPU critical path separately.
   Do not add overlapping GPU event intervals and label the sum wall time.
3. For every unique emitted GEMM shape and fused equivalent: compare native body,
   production object, cuBLASLt heuristic candidates and actual vLLM dispatch.
   Record precision, cache state, workspace, launch shape, registers, spills,
   SMEM, TFLOP/s and bytes/s. Distinguish tensor-core occupancy from tile coverage.
4. Compare local/global attention at every context rung through16K, with the same
   head geometry, window, dtype and batch layout. Include normalization and KV
   movement in the layer comparison.
5. Compare unified batching, packed prefill, token budgets and physical batch
   capacity on equal workloads. Sweep C1/4/16/32/64/128 and push higher only when
   capacity permits; report p50/p95/p99 TTFT, TPOT, E2E and aggregate throughput.
6. Promote a change only after its full-model correctness and serving advantage
   are measured. Existing cuBLASLt routing makes library substitution possible
   without replacing the shared compiler/runtime architecture.

## Gemma4 measurements, 2026-09-10

H10080GB, same checkpoint, BF16 activations/KV, uncached inputs, output128,
one warmup and one measured repeat. Native Plow uses physicalB16 and a2048-row
prefill budget; vLLM uses maximum128 sequences and8192 tokens per iteration.
Thus concurrency results compare current serving configurations, not matched
scheduler policies. All298 vLLM and34 requests per completed Plow screen have
the exact requested input/output counts and zero cached tokens.

| Variant | 1K/C1 TTFT ms | 16K/C1 TTFT ms | 16K/C16 tok/s |
|---|---:|---:|---:|
| vLLM BF16 |43.25|665.27|176.84|
| Plow native BF16 |93.20|1899.57|34.60|
| Plow native W8A16 |273.93|4520.97|21.24|
| Plow BF16, HD512 WGMMA |90.38|1456.49|35.17|
| Plow BF16, packed WGMMA work items |92.64|1437.21|59.72|
| Plow BF16, packed WGMMA, 4K budget/B16 |—|1368.31|65.27|
| Plow BF16, packed WGMMA, 8K budget/B8 |—|1383.09|66.72|

W8A16 is compared here as a Plow configuration, not as a precision-matched vLLM
result. A vLLM W8A16 checkpoint now exists but GPU qualification is pending.

Event-instrumented per-segment diagnostics identify different priorities:

- BF16 first1K chunk: GEMM37.4ms, light/FAT11.8ms, attention12.4ms.
  Last chunk of16K: GEMM37.3ms, light10.4ms, attention119.0ms. The eight
  global attention segments each take about13.65ms there.
- W8A16 first1K chunk: GEMM204.8ms, light11.7ms, attention12.0ms.
  GEMM remains approximately203ms per chunk. Segment6 and its per-layer
  equivalents contain separate gate/up projections, M1024/N15360/K3840.
- These diagnostics add event/launch overhead and do not equal serving wall
  time. They identify kernel classes; they do not assign every microsecond of
  the end-to-end gap to a component.

The packed attention object serializes its per-request calls inside each CTA.
It selects each slot's TMA descriptor and calls the ordinary optimized body
with req=nullptr. Therefore it is incorrect to blame a generic varlen fallback
for this build. The HD512 WGMMA screen improves C1 long prefill, but barely
changes C16 throughput; serialized small spans remain a plausible bottleneck.

An opt-in PLOW_NV_PACKED_FA_WGMMA experiment now sends eligible fused-output,
single-split packed requests to the existing WGMMA work-item enumerator. It
uses cp.async loads because packed mapkv is a table of per-slot descriptor
pointers, whereas that body accepts one descriptor pair. Other configurations
retain their previous dispatch. Packed serving and long retrieval pass; the
four-cell screen completes with34/34 output texts identical to the serial
WGMMA control. At16K/C16, TTFT improves50466→26623ms and throughput35.17→59.72
tokens/s. C1/1K TPOT varies despite an unchanged decode object, so repeated
interleaved runs are required before attributing small timing differences.
No production default has changed. WGMMA-vs-original generation has some
differences, so independent numerical/quality qualification remains required.

Raw evidence is in plans/gemma4-12b-roofline: *-screen.json JSONL,
profile-{bf16,fp8}-native.log, *-ladder-audit.json and screen-audit.json.
The audit example now maps queue windows to instruction PCs so a timed segment
can be associated with its actual operations and tensors.

The larger-budget screens pass packed serving checks and exact token/cache
audits. B16/PF8192 would allocate80GiB of sliding KV alone; B16/PF4096 loads63.6GiB.
The8K budget therefore uses B8 and queues the remaining requests. Compiler KV
sizing currently couples aggregate packed capacity to every request's ring size.
Separating these safely needs an enforced per-request span contract; simply
shrinking the ring would overwrite history during long chunks.

## Independent packed attention check

`runtime/tests/packed_flash_sm90_correct.cu` calls the production packed attention
body with the opt-in enabled. It checks Gemma localHD256/KV8/window1024 and
globalHD512/KV1, Q16, ragged65/33-row requests, noncontiguous reversed slots,
16K history, local ring wrapping and30 padded rows. The sampled FP64 oracle
checks4608 local and9216 global output values. All output values must be finite
and every padded output must be zero.

Measured worst relative L2: local0.00267912, global0.00252794 (gate0.004).
Maximum absolute error: local0.00110828, global0.00117296 (gate0.01).
Compute Sanitizer memcheck: zero errors. This validates the non-TMA packed
body used by the new branch; it is not a full-model quality evaluation.

Build and run from the repository root:

```sh
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -std=c++17 -gencode arch=compute_90a,code=sm_90a -O3 \
  -I runtime/common -I runtime/nvidia runtime/tests/packed_flash_sm90_correct.cu \
  -o /tmp/packed_flash_sm90_correct
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
  /tmp/packed_flash_sm90_correct
```
