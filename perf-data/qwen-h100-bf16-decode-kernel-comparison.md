# Qwen BF16 decode kernel comparison — H100

One isolated run, M=1, H100 80GB, CUDA 13.2. This is a kernel selection experiment, not a serving win or a measurement of vLLM’s selected CUDA algorithm. Installed vLLM BF16 linear dispatch reaches torch.nn.functional.linear.

Existing `runtime/bench/nvidia/bf16_gemm_vs_cublas_bench.cu`, built with `PLOW_BENCH_QWEN_GEMV`, compares Plow GV_UNROLL=8 with the best measured cuBLASLt heuristic and classical cuBLAS. Five warmups, 30 timed iterations, nonconstant inputs, rotated weights and L2 eviction. All 13 shape correctness gates passed (relative L2 <= 0.006). Times include standalone kernel launch effects; interpreter integration needs separate measurement.

| Projection | cuBLASLt µs | Plow µs | cuBLAS µs | Lt speedup vs Plow |
|---|---:|---:|---:|---:|
| a_or_b | 7.20 | 9.30 | 7.26 | 1.29× |
| qkv | 38.80 | 51.10 | 38.98 | 1.32× |
| z | 25.10 | 35.70 | 25.24 | 1.42× |
| gdn_out | 25.10 | 35.50 | 25.23 | 1.41× |
| q_full | 45.40 | 59.20 | 47.63 | 1.30× |
| k_or_v | 9.60 | 11.60 | 9.65 | 1.21× |
| gate_or_up | 61.80 | 78.00 | 61.98 | 1.26× |
| down | 63.90 | 94.30 | 63.86 | 1.48× |
| lm_head | 804.00 | 923.30 | 842.21 | 1.15× |
| fused_ba | 7.40 | 9.60 | 7.20 | 1.30× |
| fused_qkvz | 57.70 | 74.60 | 58.13 | 1.29× |
| fused_qkv | 51.20 | 66.60 | 51.46 | 1.30× |
| fused_gtup | 119.20 | 140.70 | 127.09 | 1.18× |

Raw log: `/tmp/plow-model-support-checks/qwen-m1-vs-cublas-result.log`. Executable: `/tmp/plow-model-support-checks/qwen-m1-vs-cublas 1`.

## Native activation register cache

The opt-in `PLOW_NV_GEMV_XREG=1` candidate keeps BF16 activation fragments in registers for M1, N >= 1024, K5120/6144. Other shapes use the original kernel. It preserves packet scheduling and reduction order. The persistent decoder compiles at 205 registers with no stack or local memory; its shared-memory requirement is unchanged.

| Projection | Original Plow µs | Register-cache Plow µs |
|---|---:|---:|
| qkv | 51.0 | 38.5 |
| z | 35.8 | 25.0 |
| gdn_out | 35.3 | 25.6 |
| full Q | 58.9 | 45.6 |
| K/V | 11.7 | 7.6 |
| gate/up | 78.3 | 61.7 |
| down (fallback) | 94.1 | 94.0 |
| output head | 923.7 | 806.7 |

Both isolated variants passed all 13 shape checks. Full-model logits matched byte-for-byte on five teacher-forced prefixes and two request resets per variant. In one serving run with unchanged TMA prefill, input/output 128, C1, 32 measured requests, 16 warmups and seed42, the candidate completed32/32 with TTFT88.72ms, TPOT32.46ms and output30.40tok/s. Previous native repeats measured TPOT35.36–35.37ms; the cuBLASLt dispatch smoke measured33.84ms. The native improvement is about8%, not an overall win: contemporaryvLLM measured19.99ms TPOT and48.61tok/s. A fresh same-binary control subsequently completed, as recorded below.

Raw kernel logs: `/tmp/plow-model-support-checks/qwen-bf16-xreg/result{0,1}.log`. Quality: `/tmp/plow-model-support-checks/qwen-xreg-quality.json`. Serving: `/opt/dlami/nvme/tmp/plow-h100-campaign/qwen-xreg-128-c1-r1/in128_c1.json`.

The fresh control uses the identical frozen runtime binary, asset, prefill objects, client arguments, seed and request counts; only the decoder cubin differs. Control completed32/32 with TTFT87.5605ms, TPOT35.3790ms, p99ITL35.4995ms, meanE2E4580.69ms and output27.9419tok/s. Candidate completed32/32 with TTFT88.7236ms, TPOT32.4555ms, p99ITL34.2987ms, meanE2E4210.58ms and output30.3975tok/s. Thus TPOT improved8.26% and outputthroughput8.79%, while meanTTFT increased1.16ms in this pair. This is a decode gain, not an all-metrics win. Controlraw: `/opt/dlami/nvme/tmp/plow-h100-campaign/qwen-xreg-control-128-c1-r1/in128_c1.json`.

## Native down-projection activation panels

The opt-in `PLOW_NV_GEMV_KPANEL=1` candidate covers only M1/N5120/K17408 with256 threads and at most40 owned rows per packet slice. Each warp caches2048 activation elements per panel and retains up to five row accumulators across panels. It preserves ascending FMA order, the final warp reduction, output ownership and shared-memory usage. A4096-element panel was rejected after compilation produced spills; the2048 candidate uses140 primitive registers and no spills.

The fresh primitive pair passes all13 shape checks. Down projection measures86.2µs control versus66.4µs candidate, with cuBLASLt63.9µs. Both variants report the same error against Lt (relativeL2 .000178396, max_abs .125). The historical94.3µs control above is a different, M1-only harness binary; the fresh generalized harness takes M as an argument. Both use GV_UNROLL8, so86.2→66.4 is the relevant matched comparison.

The full interpreter pair enables the existing XREG optimization in both arms and varies only KPANEL. Control uses205 registers, candidate241; both use the12352-byte arena and no stack/local memory. All five teacher-forced full-model logit rows are bit-exact between the variants, and two exact reset repetitions pass for each. Whole-serving measurement remains necessary because increased register allocation applies to the persistent interpreter. Artifacts and frozen build recipes: `/tmp/plow-qwen-kpanel/`, including `full-quality.json`, `control-result.log` and `candidate-k2048-result.log`.

The fresh serving pair subsequently completed32/32 requests per arm with zero failures. Both use `plowrt-qwen-w8a8-candidate1`, identical native TMA prefill, input/output128, C1, 16 warmups, seed42 and detailed latency output. Only the frozen XREG/KPANEL decoder cubin differs:

| Metric | XREG control | XREG + KPANEL |
|---|---:|---:|
| TTFT ms | 87.190 | 87.156 |
| TPOT ms | 32.182 | 31.334 |
| p99 ITL ms | 32.289 | 31.444 |
| Mean E2E ms | 4174.30 | 4066.57 |
| Output tok/s | 30.662 | 31.474 |

KPANEL reduces TPOT2.64% and increases throughput2.65% in this pair. Native decode still trails the contemporary vLLM19.99ms reference; repetitions and broader contexts remain open. Raw results: `qwen-kpanel{,-control}-128-c1-r1/in128_c1.json` under the campaign directory.
