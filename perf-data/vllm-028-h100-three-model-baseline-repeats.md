# vLLM 0.28.0: H100 BF16 repeated baseline

Both runs completed for all three models on September 4–5, 2026. All 54 cases passed: 1,728 successful requests, zero failures and 221,184 generated tokens. Each case has 32 requests with exact input lengths and 128 output tokens.

Same native `vllm bench serve` client and existing `scripts/bench_vllm26_native.sh`; raw `/v1/completions`, fixed random lengths, ignore EOS, temperature 0, seeds 42/43, 16 warmups per case. One H100 80 GB HBM3, driver 595.91.07, vLLM 0.28.0, PyTorch 2.13.0, BF16 weights/KV, TP1, context8192, max-seqs16, max-batched-tokens8192, memory0.90, prefix cache off, default graphs. Exact commands and pinned model revisions are in each raw directory’s invocation.json and provenance.txt.

Ranges below are the minimum and maximum of the two run means, not confidence intervals. TTFT includes request handling and first-token generation; concurrency adds queueing. TPOT measures subsequent decode latency. CPU compilation overlapped parts of the campaign; GPU ownership was exclusive. These are reference measurements, not Plow performance claims.

| Model | Input | C | TTFT ms | TPOT ms/token | Output tok/s | P99 ITL ms |
|---|---:|---:|---:|---:|---:|---:|
| gemma-4-12B-it | 128 | 1 | 28.19–28.69 | 10.54–10.55 | 93.54–93.63 | 11.47–11.50 |
| gemma-4-12B-it | 128 | 4 | 51.20–52.20 | 10.58–10.61 | 365.92–366.70 | 11.62–11.66 |
| gemma-4-12B-it | 128 | 16 | 99.06–101.12 | 11.00–11.01 | 1364.42–1367.06 | 12.17–12.19 |
| gemma-4-12B-it | 1024 | 1 | 46.71–46.76 | 10.61–10.62 | 91.67–91.76 | 11.50–11.55 |
| gemma-4-12B-it | 1024 | 4 | 128.44–128.67 | 11.16–11.19 | 330.19–331.04 | 12.00–12.07 |
| gemma-4-12B-it | 1024 | 16 | 422.85–423.91 | 13.74–13.76 | 939.56–941.15 | 14.01–14.56 |
| gemma-4-12B-it | 4096 | 1 | 169.27–170.17 | 10.62–10.64 | 84.12–84.28 | 11.58–11.60 |
| gemma-4-12B-it | 4096 | 4 | 467.60–467.78 | 12.16–12.20 | 253.64–254.30 | 11.82–12.28 |
| gemma-4-12B-it | 4096 | 16 | 1299.24–1300.55 | 21.90–21.94 | 498.51–499.16 | 324.43–331.16 |
| gemma-4-31B-it | 128 | 1 | 33.92–37.05 | 23.46–23.47 | 42.43–42.45 | 24.28–24.37 |
| gemma-4-31B-it | 128 | 4 | 73.85–74.13 | 23.77–23.77 | 165.50–165.53 | 24.64–24.72 |
| gemma-4-31B-it | 128 | 16 | 200.56–207.76 | 24.93–24.94 | 606.49–607.83 | 26.04–26.04 |
| gemma-4-31B-it | 1024 | 1 | 105.56–105.79 | 23.69–23.69 | 41.09–41.10 | 24.45–24.60 |
| gemma-4-31B-it | 1024 | 4 | 330.02–331.84 | 25.36–25.36 | 144.06–144.13 | 25.80–25.83 |
| gemma-4-31B-it | 1024 | 16 | 1732.46–2224.15 | 30.10–30.33 | 297.38–304.00 | 32.32–96.69 |
| gemma-4-31B-it | 4096 | 1 | 416.89–418.62 | 23.76–23.76 | 37.25–37.27 | 24.54–24.69 |
| gemma-4-31B-it | 4096 | 4 | 958.70–1012.12 | 30.04–30.39 | 105.97–106.15 | 402.07–402.83 |
| gemma-4-31B-it | 4096 | 16 | 6407.21–8181.68 | 38.96–43.25 | 126.55–154.21 | 422.49–426.91 |
| Qwen3.8-27B | 128 | 1 | 94.95–110.70 | 19.98–19.98 | 48.32–48.62 | 20.42–20.83 |
| Qwen3.8-27B | 128 | 4 | 171.43–185.14 | 20.76–20.76 | 181.37–182.28 | 21.21–21.46 |
| Qwen3.8-27B | 128 | 16 | 253.74–255.51 | 22.42–22.42 | 659.67–660.02 | 23.36–23.37 |
| Qwen3.8-27B | 1024 | 1 | 100.48–100.62 | 20.04–20.05 | 48.37–48.37 | 20.58–20.60 |
| Qwen3.8-27B | 1024 | 4 | 307.20–309.87 | 21.11–21.11 | 171.16–171.32 | 21.84–21.88 |
| Qwen3.8-27B | 1024 | 16 | 1126.69–1128.15 | 24.27–24.28 | 484.55–484.62 | 23.81–23.85 |
| Qwen3.8-27B | 4096 | 1 | 346.88–352.67 | 20.11–20.12 | 44.03–44.11 | 21.01–21.64 |
| Qwen3.8-27B | 4096 | 4 | 878.52–880.98 | 24.67–24.69 | 127.36–127.49 | 162.33–163.47 |
| Qwen3.8-27B | 4096 | 16 | 2766.11–2770.54 | 43.73–43.79 | 244.46–244.83 | 672.87–674.82 |

Gemma31B has roughly10GiB available for KV cache in this vLLM configuration; its long-input concurrency results reflect that capacity constraint. FP8 and Plow comparisons remain pending.

Raw evidence: `/opt/dlami/nvme/tmp/vllm-baseline-20260904`; campaign.log and COMPLETE confirm all six runs. Full numeric results: [CSV](vllm-028-h100-three-model-baseline-repeats.csv).
