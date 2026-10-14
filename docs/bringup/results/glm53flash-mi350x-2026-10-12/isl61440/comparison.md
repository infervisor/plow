# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl61440`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v15b/isl61440`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v15b/gates.json` (sha256 a72e1c18daec, packet 0abfd01688cc, PASS)
- Values are means over repeats; ratios are Infervisor / Baseline. Total throughput = (input + output) tok/s. `*` = repeat spread > 10%.

## g1

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 61440 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 16 prompts, request rate inf / concurrency 1 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 63616 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 442940c6f553, packet 0abfd01688cc) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.6 GiB | 1,203.9 GiB (0.56x) |
| Total throughput | 26,947 tok/s | 16,900 tok/s (0.63x) |
| Throughput / GPU | 3,368 tok/s/GPU | 2,112 tok/s/GPU (0.63x) |
| TTFT P99 | 2,039.3 ms * | 2,013.5 ms (0.99x) |
| TPOT P99 | 8.83 ms * | 13.09 ms (1.48x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.2% / 1.2%; Total throughput 6.0% / 0.6%; Throughput / GPU 6.0% / 0.6%; TTFT P99 65.5% / 0.8%; TPOT P99 45.6% / 1.4%.

**FLAGGED: spread > 10%: Baseline TTFT P99 65.5%; Baseline TPOT P99 45.6%.**

## g8

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | GLM-5.3-Flash (config+weights ea5874f56d39) | Same as baseline |
| Precision / quantization | weights fp8 (fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128, 128]); KV cache bfloat16 | Same as baseline |
| Input / output length | 61440 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 24 prompts, request rate inf / concurrency 8 | Same as baseline |
| GPU type & count | 8 x AMD Instinct MI350X | Same as baseline |
| Serving stack | vLLM ok    nix dev shell (ROCM_PATH=/nix/store/xbd1f4xn2j6cry6np5wij7caz0ycrykz-rocm-therock-gfx94X-dcgpu-7.14.1 CUDA_PATH=/nix/store/p49i1vrhcaw5nf2r3bwgmwfz5x8zgb14-cuda-merged-12.9)
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 63616 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 442940c6f553, packet 0abfd01688cc) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,147.7 GiB | 1,211.0 GiB (0.56x) |
| Total throughput | 43,576 tok/s | 24,442 tok/s (0.56x) |
| Throughput / GPU | 5,447 tok/s/GPU | 3,055 tok/s/GPU (0.56x) |
| TTFT P99 | 10,716.7 ms * | 16,804.4 ms (1.57x) |
| TPOT P99 | 80.48 ms * | 138.02 ms (1.71x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 5.7% / 0.8%; Throughput / GPU 5.7% / 0.8%; TTFT P99 18.0% / 2.6%; TPOT P99 14.5% / 1.5%.

**FLAGGED: spread > 10%: Baseline TTFT P99 18.0%; Baseline TPOT P99 14.5%.**
