# H100 serving workload sweep

The active performance screen uses `vllm bench serve` and the existing
`scripts/bench_vllm26_native.sh` / `scripts/bench_plowrt_serve.sh` harnesses.
128-token inputs remain quick correctness checks. Kernel and single-block rankings
must use shapes from the longer serving cells before full-model confirmation.

## Workload contract

| Parameter | Value |
|---|---|
| Models | Gemma4 12B, Gemma4 31B, Qwen3.8 27B |
| Prompt tokens | 1024,4096,8192,16384,32768,65536,131072,261632 |
| Generated tokens | 512 per request, EOS ignored |
| Client concurrency | 1,4,16 |
| Samples | 32 measured requests +16 warmups per cell, two repeats |
| Client | vLLM0.28.0, `openai`, `/v1/completions` |
| Dataset | Random fixed-length tokens, range ratio0, seed42, temperature0 |
| Arrival rate | Infinite; concurrency-limited saturation screen |
| Baseline precision | BF16 weights and KV, single H10080GB |
| vLLM scheduler | max-num-seqs16, max-num-batched-tokens8192 |
| vLLM runtime | Default compilation/graphs/attention selection, prefix cache off, memory utilization0.90 |

These are controlled workload dimensions, not a production traffic trace. The
benchmark exposes input/output lengths, arrival rate and concurrency as separate
controls; concurrency limits requests in flight and does not prove the server ran
that many requests together. See the [vLLM benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/).

Configured maximum context is65536 for the1K–32K grid,131072 for64K, and262144
for131072/261632. The final prompt reserves512 positions for output. Plow comparisons
must match these capacities and record compiled resident batch separately from
client concurrency. Startup failures, capacity limits, rejected requests and
incorrect output counts are not valid performance cells.

The first reference sweep contains144 planned cells. It was launched on2026-09-05
and is **in progress**, starting with Gemma12 input8192/output512/C1. No completed
Plow comparison or aggregate win is claimed here. The launch manifest records exact
model revisions, harness hash, all environments and output directories:

`/opt/dlami/nvme/tmp/plow-h100-campaign/realworld-vllm-bf16-20260905/manifest.json`

`current.json` identifies the active server group; `runs.jsonl` records completed
groups, including failures. Each group retains server logs, client logs, detailed
per-cell JSON, actual token counts and scheduler running/waiting traces. The harness
now rejects failed clients, incomplete request/output counts and non-finite metrics
instead of emitting a successful-looking NaN row. Existing JSON results cannot be
silently overwritten.

Report TTFT, TPOT, p99 ITL, end-to-end latency, output tokens/s and request throughput
from the same client on both endpoints. Keep repeats separate before aggregation.
Long-context Plow numerical qualification and resident-batch support remain open;
reference results do not close those gates. FP8 comparisons require a separate
matched-precision matrix.

## First completed reference cells

Gemma12, input8192/output512, first repeat, max context65536. Each cell completed
32/32 requests with zero failures,262144 input tokens and16384 output tokens;
every individual request had the specified input/output lengths.

| Concurrency | TTFT ms | TPOT ms | p99 ITL ms | E2E ms | Output tok/s |
|---:|---:|---:|---:|---:|---:|
|1|351.644|10.533478|11.431093|5734.251|89.284154|
|4|996.197|11.632726|11.979480|6940.520|294.970504|
|16|2304.410|18.585268|326.336441|11801.482|691.562909|

Raw: campaign `gemma12b-1k-32k-r1/in8192_c{1,4,16}.json` under the manifest directory.
These are individual vLLM reference cells, not an aggregate or a Plow comparison.

The complete first Gemma12 grid (1K/4K/8K/16K/32K × C1/4/16) subsequently finished:
480/480 requests, zero failures,245760 output tokens, with every input/output
length checked. [All15 cells](vllm-028-h100-realworld-gemma12-r1.csv) retain separate
mean/median metrics. At32K/C16, TTFT is9671.675ms, TPOT46.535611ms, p99ITL490.444421ms
and output243.538065tok/s. The larger concurrent workloads expose queueing and
prefill interference that the128-token screen does not describe. Qwen's first
reference group is now running; repeats and the other groups remain pending.

## Fast experiment preparation

`scripts/tune_decode_sweep.sh --block L --block-bucket ROWS --block-run PATH` now
reuses the architecture-aware packet/cubin grid with the existing `block_run`
runner. Qwen layers0/3 dry-run successfully at8K/32K, B1, PF1024 and context65536;
dry-runs emit no timing rows. The compiler selects the real GDN/full-attention
layer, its weights/state and BF16 `act.x` input/output. Cubin cache keys include
the native source hash, and each block result has a separate output directory.

Six Gemma12 full-attention layer5 packets are emitted and verified offline for
GF4/8 × NS16/33/64, context65536, prefill ladder128/512/1024. Actual layer geometry
is D512/NH16/KV1/scale1. GF is an explicit cubin constant; the generated manifest's
recommendation8 does not replace the required GF4/GF8 build flags. Artifact recipes
and instruction checks: `/tmp/plow-model-support-checks/gemma12-realworld-block-grid/`.

Qwen FP8 prefill requires independent decoder and prefill capability markers.
The runtime rejects an unpaired prefill cubin before launch; the new marked object
has byte-identical CUDA text sections to its preceding candidate. These builds and
offline checks do not replace GPU model qualification. Block benchmarks use
synthetic inputs for ranking; numeric checks require captured decode inputs or one
exact prefill bucket, with inputs explicitly refreshed between chunks.

## First larger-shape kernel check

The existing FP8 comparator ran ten M1024 projection/tail cases on the H100. All
outputs were bit-exact against installed vLLM CUTLASS, activation bytes/scales were
exact and output canaries passed. This is a primitive arithmetic gate, not model
parity or TTFT.

Qwen QKV, M1024/N10240/K5120: native210.560µs vs CUTLASS79.760µs, or2.64× slower.
The timed loop excludes quantization, descriptor construction and allocations,
uses five warmups and median30 CUDA-event samples, with700MiB eviction outside each
sample. The result makes larger-M scheduling/pipelining a concrete target; a
successful small-M kernel is insufficient.

Raw evidence:
`/tmp/plow-model-support-checks/qwen-fp8-m128/m1024-candidate.json`.
Frozen sources, builds, SASS, resources and the implemented/missing CUTLASS technique
table are in the same directory's `HANDOFF.md` and `provenance.json`.

## Larger-workload candidates prepared after the first cells

The existing uniform FP8 TMA kernel has frozen stage2/stage3 pairs for M1024,
4096 and8192. Both primitives use137 registers without spills. The full stage2
prefill interpreter reports1432 stack bytes,2320 spill-store bytes and5792 spill-load
bytes, versus1856/4248/10600 for stage3. Its advertised shared arena remains103424
bytes. Stage2 also excludes an unused generic TMA GLU arm through an existing
capacity guard, so the whole-object resource change is not solely the loop's
code generation. Qwen emits a separate GLU. No GPU result or occupancy gain is
claimed. Recipes and six existing-comparator commands:
`/tmp/plow-model-support-checks/qwen-fp8-tma-stages/`.

A separate default-off compiler experiment isolates mapped FP8 GEMMs into native
interpreter segments. `PLOW_QWEN_FP8_PF_ISOLATE=1` changes the frontend's segmentation
while preserving instructions, operands, waits, successors and decode. Kernel-role
selection belongs in the compiled packet's generic `segment_roles.json` descriptor;
the runtime executes that descriptor and validates capabilities and cooperative
residency. The initially proposed model-specific runtime option was removed before
consolidation. `PLOW_FP8_PF_GEMM_ROLE=1` is a compiler experiment switch.
Quantization, BF16 head, attention, norms and the existing GDN path stay unchanged.
The specialized object uses180 registers and zero stack/spills; the broad object
uses255 registers with spills. The arithmetic headers are identical.

The cost is substantial: full-model native launches increase from97 to929 per
chunk, with496 GEMMs routed and48 GDN calls unchanged. First compare the same
isolated packet with role routing off/on; then compare against the original
97-launch packet. Neither spill removal nor CPU ordering tests establish a speed
gain. Source/build/SASS evidence:
`/tmp/plow-model-support-checks/fp8-prefill-gemm-role/`. The role advertises generic
ABI `plow_fp8_gemm_tma128_abi=1` and compiles without a Qwen flag. Its four kernel
text sections are byte-identical to the earlier role object.

### Existing Gemma chunk-size tradeoff

Ten full-model packets were emitted with the existing `PLOW_MAX_CHUNK` setting,
holding the other compiler flags fixed. Active Lean ordering checks passed for
every program. The1024 setting gives a2048-row sliding KV ring;4096 gives8192.
Both satisfy `ring >= window + chunk - 1`. Smaller chunks require more prefill
launches, so this is a memory/performance experiment, not an assumed improvement.

| Model | Context capacity | Resident batch | Total declared GiB, chunk4096 | Chunk1024 |
|---|---:|---:|---:|---:|
|Gemma12|65536|1|26.681|24.259|
|Gemma12|65536|16|79.188|48.641|
|Gemma31|65536|1|69.962|64.328|
|Gemma31|65536|2|81.212|70.891|
|Gemma31|262144|1|85.525|79.892|

These totals sum the actual packet tensor table, including weights/KV/other
buffers. They exclude allocator alignment, programs, counters, graphs and workspace;
they are not measured high-water usage or successful startup. The device reports
81559MiB (79.647GiB), so even the smaller Gemma31/262144 packet exceeds capacity
before overhead. The smaller-ring assets materially change the capacity experiment
for Gemma12/B16 and Gemma31/B2. GPU startup, chunk-boundary correctness and serving
remain pending. Exact packets, commands, hashes, tensor disassembly and totals:
`/tmp/plow-model-support-checks/gemma-realworld-chunk-capacity/`.
