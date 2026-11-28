# GLM-5.3-Flash comparison (8x AMD Instinct MI350X, TP8)

This file holds only qualified wins and their evidence. Experimental history, internal A/Bs and
the non-matched arms are summarized in [summary.md](summary.md).
Every matched serving row is in [comparison.csv](comparison.csv).

## Qualified wins (MATCHED + EQUIVALENT, `campaign.py report` format)

- Infervisor: plowrt `4e8f9d6f`, packet `d3d7db7307d4`, built from
  [`recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`](../../../../recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml)
  (FP8 W8A8 weights, BF16 KV cache, 64K context, prefix cache on, DSA enabled with `PLOW_GLM_DSA = "1"`,
  `PLOW_PACKED_PREFILL_ROUTE = "0"`).
  Consolidated decode and prefill improvements: fused router GEMM+TopK, dense native route,
  decode MoE tiles, fused hipBLASLt projections, KDA carry schedule, tail audit with tail signal
  completion, and 8K burst prefill.
- Baseline: vLLM 0.31.1rc1 ROCm (`vllm/vllm-openai-rocm:nightly-rocm100`), TP8, AITER on,
  prefix caching off, same 8x MI350X host, same vLLM benchmark client.
- Quality: FP32-reference gate PASS for the served packet (`gates.json` sha256 `d250b72ad0cd`;
  KL mean 0.01718 vs vLLM 0.03677 [>2.1x lower/better KL], top1_decisive 0.9978 vs vLLM 0.9868, needle_acc 1.0 vs 1.0).
  Long-context needles 36/36 (100% pass, 32s). 0 collective errors.

Ratios are Infervisor / vLLM, from matched benchmark runs.

| Cell (ISL / OSL, Conc) | Infervisor tok/s | vLLM tok/s | Total throughput ratio | Infervisor TTFT P99 | vLLM TTFT P99 | TTFT ratio | Infervisor TPOT P99 | vLLM TPOT P99 | TPOT ratio | Peak GPU Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4096 / 128, c1** | 3,433 | 4,153 | 0.83x | **144.5 ms** | 511.6 ms | **0.28x (3.5x faster)** | 8.78 ms | 6.97 ms | 1.26x | **1,702.2 GiB (0.80x)** |
| **4096 / 128, c8** | 14,002 | 14,650 | 0.96x | **1,095.7 ms** | 2,290.6 ms | **0.48x (2.09x faster)** | **18.04 ms** | 25.48 ms | **0.71x (1.41x faster)** | **1,708.8 GiB (0.80x)** |
| **1024 / 128, c8** | 4,791 | 5,256 | 0.91x | **639.5 ms** | 1,764.9 ms | **0.36x (2.76x faster)** | **15.79 ms** | 19.29 ms | **0.82x (1.22x faster)** | **1,708.8 GiB (0.80x)** |
| **16384 / 128, c1** | 10,617 | 13,015 | 0.82x | **466.7 ms** | 758.3 ms | **0.62x (1.62x faster)** | 8.87 ms | 7.03 ms | 1.26x | **1,702.2 GiB (0.79x)** |
| **16384 / 128, c128** | 33,236 | 50,932 | 0.65x | 60,716.4 ms | 39,671.5 ms | 1.53x | **249.91 ms** | 314.14 ms | **0.80x (1.26x faster)** | **1,718.1 GiB (0.80x)** |

### Evidence (campaign scratch outside repo)

- Infervisor DSA serving run: `/home/lava/bench/glm53flash/dsa-serving/` (`server.log`, per-cell `bench.json`, `cells.log`)
- Strict comparison reports: `/home/lava/bench/glm53flash/dsa-report/` (`isl1024/comparison.md`, `isl4096/comparison.md`, `isl16384/comparison.md`)
- Correctness gates:
  - FP32-reference gate: `/home/lava/bench/glm53flash/dsa-gates/gates.json` (PASS, kl_mean 0.01718 vs vLLM 0.03677)
  - Long-context needle probe (36/36, 100% pass): `/home/lava/bench/glm53flash/dsa-gates/needle_dsa.json`
