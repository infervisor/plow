# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl32768`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v18/isl32768`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v18-int/gates.json` (sha256 360cd843bdbd, packet 360814fc607b, PASS)
- Values are means over repeats; ratios are Infervisor / Baseline. Total throughput = (input + output) tok/s. `*` = repeat spread > 10%.

## g1

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 32768 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 16 prompts, request rate inf / concurrency 1 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 34944 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,141.8 GiB | 1,485.4 GiB (0.69x) |
| Total throughput | 20,317 tok/s | 14,948 tok/s (0.74x) |
| Throughput / GPU | 2,540 tok/s/GPU | 1,869 tok/s/GPU (0.74x) |
| TTFT P99 | 1,124.1 ms * | 822.9 ms (0.73x) |
| TPOT P99 | 6.96 ms | 11.06 ms (1.59x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.9%; Total throughput 3.8% / 0.2%; Throughput / GPU 3.8% / 0.2%; TTFT P99 73.1% / 2.8%; TPOT P99 0.6% / 0.1%.

**FLAGGED: spread > 10%: Baseline TTFT P99 73.1%.**

## g8

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 32768 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 24 prompts, request rate inf / concurrency 8 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 34944 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt ca11e3f0822b, packet 360814fc607b) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,141.8 GiB | 1,492.5 GiB (0.70x) |
| Total throughput | 38,833 tok/s * | 28,795 tok/s (0.74x) |
| Throughput / GPU | 4,854 tok/s/GPU * | 3,599 tok/s/GPU (0.74x) |
| TTFT P99 | 5,771.3 ms * | 6,666.5 ms (1.16x) |
| TPOT P99 | 47.84 ms * | 63.90 ms (1.34x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 12.1% / 0.3%; Throughput / GPU 12.1% / 0.3%; TTFT P99 19.3% / 1.4%; TPOT P99 17.6% / 0.6%.

**FLAGGED: spread > 10%: Baseline Total throughput 12.1%; Baseline Throughput / GPU 12.1%; Baseline TTFT P99 19.3%; Baseline TPOT P99 17.6%.**
