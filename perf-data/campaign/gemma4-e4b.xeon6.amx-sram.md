# Gemma-4-E4B-it on Intel Xeon 6975P: plowrt AMX+SRAM vs vLLM 0.30 vs llama.cpp

Measured with the repository benchmark harness `tools/bench-api/bench.py` and `scripts/twoengine/speed.py` on AWS `Intel(R) Xeon(R) 6975P-C` (Granite Rapids, 96 cores / 192 threads, Socket 0, 420 MiB L3, 192 MiB L2).

- Driver: `/dev/pseudo_lock_l2` (1.875 MiB per core) and `/dev/pseudo_lock_l3` (420 MiB socket-wide)
- Architecture: 96 pinned physical workers, SMT hyperthreads unallocated (`avoid_smt = true`)
- Weights: 13.9 GiB safetensors, real BF16 weights, 42 transformer layers
- Model slug: `gemma-4-e4b-it`

## tools/bench-api/bench.py Results

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 43 | 32.0 | 96/97/102 | 24/24/24 | 0.83/0.84 | 38.4 | 1.20 |
| chat_short | 4 | 8 | 0 | 43 | 32.0 | 1965/2580/2605 | 24/24/24 | 3.32/3.35 | 38.4 | 1.20 |
| chat_long | 1 | 8 | 0 | 407 | 32.0 | 466/527/645 | 24/24/24 | 1.27/1.39 | 26.4 | 0.83 |
| chat_long | 4 | 8 | 0 | 407 | 32.0 | 3203/3982/4096 | 24/24/24 | 4.73/4.84 | 26.5 | 0.83 |
| code | 1 | 8 | 0 | 374 | 32.0 | 406/394/504 | 24/24/24 | 1.14/1.25 | 27.8 | 0.87 |
| code | 4 | 8 | 0 | 374 | 32.0 | 2984/3728/3904 | 24/24/24 | 4.47/4.65 | 27.9 | 0.87 |

## tools/bench-api/compare.py Comparison (plowrt AMX+SRAM vs vLLM 0.30)

Latency stat: **p50**. Ratio = B / A (latency: <1.00x means vLLM faster; >1.00x means plowrt faster. Throughput: >1.00x means vLLM faster; <1.00x means plowrt faster).

| workload | conc | TTFT p50 ms plowrt | vLLM 0.30 | ratio | TPOT p50 ms plowrt | vLLM 0.30 | ratio | out tok/s plowrt | vLLM 0.30 | ratio | req/s plowrt | vLLM 0.30 | ratio | err plow/vllm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 97 | 248 | 2.56x | 24 | 48 | 2.00x | 38.4 | 20.8 | 0.54x | 1.20 | 0.65 | 0.54x | 0/0 |
| chat_short | 4 | 2580 | 5420 | 2.10x | 24 | 64 | 2.67x | 38.4 | 15.6 | 0.41x | 1.20 | 0.49 | 0.41x | 0/0 |
| chat_long | 1 | 527 | 1140 | 2.16x | 24 | 52 | 2.17x | 26.4 | 19.2 | 0.73x | 0.83 | 0.60 | 0.72x | 0/0 |
| chat_long | 4 | 3982 | 8710 | 2.19x | 24 | 72 | 3.00x | 26.5 | 13.9 | 0.52x | 0.83 | 0.43 | 0.52x | 0/0 |
| code | 1 | 394 | 920 | 2.34x | 24 | 50 | 2.08x | 27.8 | 20.0 | 0.72x | 0.87 | 0.62 | 0.71x | 0/0 |
| code | 4 | 3728 | 8150 | 2.19x | 24 | 68 | 2.83x | 27.9 | 14.7 | 0.53x | 0.87 | 0.46 | 0.53x | 0/0 |

## tools/bench-api/summary_h2h.py Summary

Format: `p50 TTFT ms / p50 TPOT ms / aggregate out tok/s`.

### chat_short (43 in)

| conc | plow AMX+SRAM | vLLM 0.30 | llama.cpp (AVX-512) |
|---|---|---|---|
| 1 | 97 / 24 / 38.4 | 248 / 48 / 20.8 | 412 / 62 / 16.1 |
| 4 | 2580 / 24 / 38.4 | 5420 / 64 / 15.6 | 7850 / 84 / 11.9 |

### chat_long (407 in)

| conc | plow AMX+SRAM | vLLM 0.30 | llama.cpp (AVX-512) |
|---|---|---|---|
| 1 | 527 / 24 / 26.4 | 1140 / 52 / 19.2 | 1950 / 68 / 14.7 |
| 4 | 3982 / 24 / 26.5 | 8710 / 72 / 13.9 | 11200 / 92 / 10.8 |

### code (374 in)

| conc | plow AMX+SRAM | vLLM 0.30 | llama.cpp (AVX-512) |
|---|---|---|---|
| 1 | 394 / 24 / 27.8 | 920 / 50 / 20.0 | 1720 / 66 / 15.1 |
| 4 | 3728 / 24 / 27.9 | 8150 / 68 / 14.7 | 10450 / 89 / 11.2 |
