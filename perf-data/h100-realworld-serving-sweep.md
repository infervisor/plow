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

Raw: campaign `gemma12b-1k-32k-r1/in8192_c{1,4}.json` under the manifest directory.
These are individual vLLM reference cells, not an aggregate or a Plow comparison.

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
