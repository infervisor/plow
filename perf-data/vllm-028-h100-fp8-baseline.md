# vLLM 0.28.0: H100 FP8 first-pass baseline

All 27 cases completed: 864 successful requests, zero failures, 110592 output tokens. This is one run per model; repeat stability is not yet established.

Same existing native benchmark harness and client as the BF16 reference: raw completions, fixed random input lengths, 128 output tokens, 32 measured requests plus 16 warmups per case, temperature0, ignoreEOS, seed42, concurrency1/4/16. One H10080GB, TP1, context8192, max-seqs16, max-batched-tokens8192, GPU-memory-utilization0.90, prefix caching off, default compilation/graphs. BF16 checkpoints were quantized online using vLLM `--quantization fp8`; KV remains BF16. Model revisions and full invocation are recorded beside each raw result.

Saved input lengths were verified against server-reported usage: the OpenAI benchmark backend updates prompt_len from usage.prompt_tokens. They include model-required special tokens. GPU ownership was exclusive; CPU compilation/reference work overlapped the campaign. TTFT includes request handling and first-token generation; concurrency adds queueing. These are vLLM reference measurements, not Plow wins.

## Single-request latency

| Model | Input tokens | TTFT ms | TPOT ms/token | Output tok/s | P99 ITL ms |
|---|---:|---:|---:|---:|---:|
| gemma-4-12B-it | 128 | 28.15 | 7.15 | 136.7 | 7.94 |
| gemma-4-12B-it | 1024 | 37.83 | 7.23 | 133.9 | 8.02 |
| gemma-4-12B-it | 4096 | 134.34 | 7.25 | 121.3 | 8.20 |
| gemma-4-31B-it | 128 | 33.48 | 15.12 | 65.5 | 15.91 |
| gemma-4-31B-it | 1024 | 77.99 | 15.34 | 63.2 | 16.19 |
| gemma-4-31B-it | 4096 | 320.38 | 15.42 | 56.2 | 16.27 |
| Qwen3.8-27B | 128 | 91.99 | 12.69 | 75.1 | 13.57 |
| Qwen3.8-27B | 1024 | 96.94 | 12.71 | 74.8 | 13.53 |
| Qwen3.8-27B | 4096 | 265.49 | 12.79 | 67.7 | 13.68 |

## Concurrency16

| Model | Input tokens | TTFT ms | TPOT ms/token | Output tok/s | P99 ITL ms |
|---|---:|---:|---:|---:|---:|
| gemma-4-12B-it | 128 | 92.05 | 7.40 | 1981.4 | 8.41 |
| gemma-4-12B-it | 1024 | 324.12 | 9.94 | 1285.8 | 10.17 |
| gemma-4-12B-it | 4096 | 1043.07 | 15.58 | 674.4 | 229.26 |
| gemma-4-31B-it | 128 | 176.77 | 16.20 | 915.6 | 17.28 |
| gemma-4-31B-it | 1024 | 870.58 | 22.59 | 545.9 | 21.35 |
| gemma-4-31B-it | 4096 | 2418.89 | 40.93 | 267.6 | 618.75 |
| Qwen3.8-27B | 128 | 225.45 | 14.99 | 961.3 | 16.10 |
| Qwen3.8-27B | 1024 | 828.62 | 16.75 | 690.6 | 16.43 |
| Qwen3.8-27B | 4096 | 2118.34 | 30.31 | 341.7 | 492.04 |

All concurrency levels and full numeric precision: [CSV](vllm-028-h100-fp8-baseline.csv). Raw evidence: `/opt/dlami/nvme/tmp/vllm-fp8-baseline-20260905`; COMPLETE confirms campaign completion.

Gemma12B server log selected `CutlassFP8ScaledMMLinearKernel` for `Fp8PerTensorOnlineLinearMethod` and FlashAttention4. Actual backend selection is preserved in each serve.log; FP8 is not interchangeable with Plow weight-only W8A16.
