# Kimi-K3 vLLM baseline on MI355X

The current C1 baseline was measured 2026-09-04 on one node with 8×AMD
Instinct MI355X. It is the predeclared median-of-folds reading from three fresh,
alternating cells in campaign `k3-showdown-c1-fe871e6-final`. Older C1 rows
are superseded. The older C128 cell is unchanged and was not rerun.

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
| duration | 218.53 s | 1155.90 s |
| output throughput | 46.86 tok/s | 1133.93 tok/s |
| total token throughput | 421.72 tok/s | 10205.41 tok/s |
| mean / median TTFT | 567.87 / 567.74 ms | 5441.21 / 1436.79 ms |
| P90 / P99 TTFT | 570.08 / 570.58 ms | 2594.22 / 70942.77 ms |
| mean / median TPOT | 20.81 / 20.81 ms | 107.232 / 109.630 ms |
| median / P90 / P99 ITL | 20.81 / 20.88 / 20.97 ms | 49.633 / 287.441 / 290.985 ms |
| mean / median E2E | 21.853 / 21.852 s | 115.139 / 113.738 s |

C1 implies a 14,429.1 input-token/s TTFT proxy and 48.05 decode tok/s from
median TPOT. These are served-workload ratios, not isolated-kernel measurements.

## Exact C1 fold provenance

| fold | mean / median / P99 TTFT (ms) | mean / median / P99 TPOT (ms) | output tok/s | mean / median / P99 E2E (ms) |
|---|---:|---:|---:|---:|
| showdown-1 | 568.31 / 568.20 / 570.58 | 20.70 / 20.70 / 20.71 | 47.09 | 21745.91 / 21746.12 / 21752.54 |
| showdown-2 | 567.87 / 567.74 / 570.27 | 20.81 / 20.81 / 20.81 | 46.86 | 21852.87 / 21852.24 / 21859.58 |
| showdown-3 | 566.74 / 566.43 / 570.78 | 20.81 / 20.81 / 20.82 | 46.85 | 21855.13 / 21854.52 / 21867.26 |

Artifact-set digest:
`f76f9e25be770d36f397f446f285b14061dca8a8f918012d977e8ff5bf18a2b6`.
The JSON preserves every `cells.tsv` field, source log basename, and
config/tokenizer identity. Headline values are medians of the three fold
statistics; P90 and duration values come from the
three raw client logs.

## Superseded result

The 2026-09-03 three-fold row (568.35 ms median TTFT, 20.86 ms median TPOT,
46.73 output tok/s) is replaced by this fresh same-client campaign and remains
only in the JSON `supersedes` summary.

Raw C1 data: `perf-data/kimi-k3-vllm-mi355x-c1.json`.
Unchanged C128 data: `perf-data/kimi-k3-vllm-mi355x-c128.json`.

## Comparison rules

- Compare Plow with the same 8192→1024 shape, exact output count, greedy
  sampling, prefix-cache state, concurrency, and NUMA setting.
- C128 is a scheduler/system target, not a single-kernel target.
- TTFT includes scheduling and first-token work. `input_tokens / TTFT` is
  only a prefill proxy; kernel-only prefill must be reported separately.
