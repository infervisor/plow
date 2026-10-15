# Serving comparison: Infervisor vs baseline

**Status: MATCHED, EQUIVALENT**

- Baseline results: `/home/lava/bench/glm53flash/vllm-tp8/isl61440`
- Infervisor results: `/home/lava/bench/glm53flash/plow-v20/isl61440`
- FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v19-r3-g/gates.json` (sha256 d22020e1f1c6, packet 49b696e4e77c, PASS)
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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 63616 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 49b696e4e77c) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,145.6 GiB | 1,472.6 GiB (0.69x) |
| Total throughput | 26,947 tok/s | 20,512 tok/s (0.76x) |
| Throughput / GPU | 3,368 tok/s/GPU | 2,564 tok/s/GPU (0.76x) |
| TTFT P99 | 2,039.3 ms * | 1,817.2 ms (0.89x) |
| TPOT P99 | 8.83 ms * | 10.80 ms (1.22x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.2% / 1.0%; Total throughput 6.0% / 1.7%; Throughput / GPU 6.0% / 1.7%; TTFT P99 65.5% / 1.4%; TPOT P99 45.6% / 21.7%.

**FLAGGED: spread > 10%: Baseline TTFT P99 65.5%; Baseline TPOT P99 45.6%; Infervisor TPOT P99 21.7%.**

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
0.31.1rc1.dev296+gfe4df3df0.rocm100 (--gpu-memory-utilization 0.90 --max-num-seqs 256 --tensor-parallel-size 8 --trust-remote-code --max-model-len 63616 --max-num-seqs 256 --no-enable-prefix-caching --tool-call-parser glm47 --reasoning-parser glm47) | Infervisor (plowrt 29af24ff94c4, packet 49b696e4e77c) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.03677, top1_decisive 0.9868, needle_acc 1) | Equivalent |
| Peak GPU memory | 2,147.7 GiB | 1,479.6 GiB (0.69x) |
| Total throughput | 43,576 tok/s | 29,657 tok/s (0.68x) |
| Throughput / GPU | 5,447 tok/s/GPU | 3,707 tok/s/GPU (0.68x) |
| TTFT P99 | 10,716.7 ms * | 14,410.6 ms (1.34x) |
| TPOT P99 | 80.48 ms * | 113.53 ms (1.41x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 5.7% / 0.0%; Throughput / GPU 5.7% / 0.0%; TTFT P99 18.0% / 0.4%; TPOT P99 14.5% / 0.2%.

**FLAGGED: spread > 10%: Baseline TTFT P99 18.0%; Baseline TPOT P99 14.5%.**
