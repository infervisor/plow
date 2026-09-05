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
