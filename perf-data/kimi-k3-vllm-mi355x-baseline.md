# Kimi-K3 vLLM baseline on MI355X

Measured 2026-09-02 on one node with 8×AMD Instinct MI355X (`gfx950`,
309,220,868,096 bytes VRAM/card). The local checkpoint was 96 safetensors
shards, 1453.74 GiB. This supersedes the failed MI325X vLLM bring-up attempt.

## Stack and server

- Image: `vllm/vllm-openai-rocm:v0.28.0@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032`
- vLLM: `0.28.0+rocm723`; PyTorch: `2.12.0+git6bbd260`; TP=8.
- Model: local Kimi-K3 MXFP4 checkpoint, BF16 activations, text-only mode.
- Enabled: AITER, SiTU-v2 A8W4 layout, CUDA graphs, chunked prefill with
  `max_num_batched_tokens=4096`.
- Disabled for this baseline: MTP/speculative decoding, prefix caching, FP8 KV.
- Host kernel: `6.8.0-71-generic`. NUMA balancing was enabled; AITER warned that
  this is suboptimal, so comparisons must preserve or explicitly change that condition.

Startup selected the fused KDA decode kernel, AITER MLA prefill, AITER MXFP4-BF16
MoE, AITER custom all-reduce, and the `norm_quant`, `act_quant`,
`allreduce_rms`, and `mla_dual_rms_norm` fusions.

The image entrypoint is `vllm serve`; its material arguments and environment were:

```text
VLLM_ROCM_USE_AITER=1
SAFETENSORS_FAST_GPU=1
VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
AITER_SITUV2_A8W4=1
AITER_BF16_FP8_MOE_BOUND=0
VLLM_USE_BREAKABLE_CUDAGRAPH=0

/model_weights --served-model-name kimi-k3 --dtype auto
--tensor-parallel-size 8 --trust-remote-code --no-enable-prefix-caching
--load-format auto --gpu-memory-utilization 0.95 --moe-backend auto
--mm-encoder-tp-mode data --max-num-seqs 128
--max-num-batched-tokens 4096 --reasoning-parser kimi_k3
--language-model-only --disable-uvicorn-access-log
```

## Benchmark contract

Both cells used vLLM 0.28 `bench serve`, the completions endpoint, random exact
8192-token inputs and exact 1024-token outputs, greedy sampling, `--ignore-eos`,
infinite request rate, and no prefix reuse. C1 used one discarded warmup and 10
measured requests. C128 used 1280 measured requests (ten saturation waves).

```text
vllm bench serve --backend openai --endpoint /v1/completions \
  --model kimi-k3 --tokenizer /model_weights --trust-remote-code \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 \
  --random-range-ratio 0 --request-rate inf --ignore-eos --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
```

## Results

| metric | C1, N=10 | C128, N=1280 |
|---|---:|---:|
| successful / failed | 10 / 0 | 1280 / 0 |
| duration | 218.13 s | 1155.90 s |
| request throughput | 0.0458 req/s | 1.1074 req/s |
| output throughput | 46.94 tok/s | 1133.93 tok/s |
| peak output throughput | 49 tok/s | 2688 tok/s |
| total token throughput | 422.50 tok/s | 10205.41 tok/s |
| mean / median TTFT | 567.52 / 567.03 ms | 5441.21 / 1436.79 ms |
| P90 / P99 TTFT | 569.22 / 570.50 ms | 2594.22 / 70942.77 ms |
| mean / median TPOT | 20.768 / 20.768 ms | 107.232 / 109.630 ms |
| median / P90 / P99 ITL | 20.755 / 20.878 / 21.008 ms | 49.633 / 287.441 / 290.985 ms |
| mean / median E2E | 21.813 / 21.813 s | 115.139 / 113.738 s |

For attribution, C1 TTFT implies a 14,434.84 input-token/s prefill proxy
(`8192 / TTFT`), and C1 mean TPOT implies 48.15 decode tok/s excluding the
first token. C128 processed 9071.47 accounted input tok/s while decode was
active. These are served-workload ratios, not isolated-kernel measurements.

Raw benchmark output:

- `perf-data/kimi-k3-vllm-mi355x-c1.json`
- `perf-data/kimi-k3-vllm-mi355x-c128.json`

## Comparison rules

- Compare Plow with the same 8192→1024 shape, exact output count, greedy
  sampling, prefix-cache state, concurrency, and NUMA setting.
- C128 is a scheduler/system target, not a single-kernel target. Plow now has a
  B128 rung, but its current recurrence-safe prefill is not cross-request
  co-packed; decode-width coverage alone is not a throughput comparison.
- TTFT includes scheduling and first-token work. `input_tokens / TTFT` is only
  a prefill proxy; kernel-only prefill must be reported separately.
