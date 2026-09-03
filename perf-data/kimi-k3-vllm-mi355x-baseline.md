# Kimi-K3 vLLM baseline on MI355X

The current C1 baseline was measured 2026-09-03 on one node with 8×AMD
Instinct MI355X. It is the predeclared median-of-folds reading from three exact cells in
campaign `k3-showdown-c1-7a54489-retry2`. The 2026-09-02 single-run C1
values are superseded. The older C128 cell is unchanged and was not rerun.

## Stack and server

- Image: `vllm/vllm-openai-rocm@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032`.
- vLLM `0.28.0+rocm723`; PyTorch `2.12.0+git6bbd260`; TP8.
- Local 96-shard Kimi-K3 MXFP4 checkpoint, BF16 activations, text-only mode.
- AITER, SiTU-v2 A8W4 layout, CUDA graphs, and chunked prefill with
  `max_num_batched_tokens=4096` enabled.
- MTP/speculative decoding, prefix caching, and FP8 KV disabled.
- Host kernel `6.8.0-71-generic`; NUMA balancing enabled.

## Benchmark contract

vLLM 0.28 `bench serve` used the completions endpoint, random exact
8192-token inputs and exact 1024-token outputs, greedy sampling,
`--ignore-eos`, infinite request rate, and no prefix reuse. Each C1 fold used
one discarded warmup and ten measured requests; all 30 measured requests
completed. The older C128 cell used 1280 measured requests.

## Results

| metric | C1, 3-fold mean | C128, N=1280 (unchanged) |
|---|---:|---:|
| successful / failed | 10 / 0 per fold | 1280 / 0 |
| duration | 219.12 s | 1155.90 s |
| output throughput | 46.73 tok/s | 1133.93 tok/s |
| total token throughput | 420.60 tok/s | 10205.41 tok/s |
| mean / median TTFT | 567.86 / 568.35 ms | 5441.21 / 1436.79 ms |
| P90 / P99 TTFT | 570.28 / 572.01 ms | 2594.22 / 70942.77 ms |
| mean / median TPOT | 20.86 / 20.86 ms | 107.232 / 109.630 ms |
| median / P90 / P99 ITL | 20.86 / 20.96 / 21.06 ms | 49.633 / 287.441 / 290.985 ms |
| mean / median E2E | 21.911 / 21.910 s | 115.139 / 113.738 s |

C1 implies a 14,414.8 input-token/s TTFT proxy and 47.94 decode tok/s from
median TPOT. These are served-workload ratios, not isolated-kernel measurements.

## Exact C1 fold provenance

| fold | mean / median / P99 TTFT (ms) | mean / median / P99 TPOT (ms) | output tok/s | mean / median / P99 E2E (ms) |
|---|---:|---:|---:|---:|
| showdown-1 | 568.60 / 568.77 / 573.42 | 20.86 / 20.86 / 20.87 | 46.73 | 21911.41 / 21910.17 / 21921.70 |
| showdown-2 | 567.86 / 568.35 / 572.01 | 20.78 / 20.78 / 20.79 | 46.91 | 21829.56 / 21830.47 / 21841.47 |
| showdown-3 | 567.06 / 567.31 / 570.38 | 20.93 / 20.93 / 20.95 | 46.59 | 21979.50 / 21979.00 / 21991.35 |

Artifact-set digest:
`f76f9e25be770d36f397f446f285b14061dca8a8f918012d977e8ff5bf18a2b6`.
The JSON preserves every `cells.tsv` field, source log basename, and
config/tokenizer identity. Headline values are medians of the three fold
statistics; P90 and duration values come from the
three raw client logs.

## Superseded result

The 2026-09-02 C1 single run (218.13 s, 46.94 output tok/s, 567.52 ms mean
TTFT, 20.768 ms mean TPOT) remains in the JSON `supersedes` record but is
not the current C1 baseline.

Raw C1 data: `perf-data/kimi-k3-vllm-mi355x-c1.json`.
Unchanged C128 data: `perf-data/kimi-k3-vllm-mi355x-c128.json`.

## Comparison rules

- Compare Plow with the same 8192→1024 shape, exact output count, greedy
  sampling, prefix-cache state, concurrency, and NUMA setting.
- C128 is a scheduler/system target, not a single-kernel target.
- TTFT includes scheduling and first-token work. `input_tokens / TTFT` is
  only a prefill proxy; kernel-only prefill must be reported separately.
