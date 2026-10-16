# GLM-5.3-Flash comparison (8x AMD Instinct MI350X, TP8)

This file holds only qualified wins and their evidence. Experimental history, internal A/Bs and
the non-matched arms are summarized in [summary.md](summary.md).
Every matched serving row is in [comparison.csv](comparison.csv).

## Qualified wins (MATCHED + EQUIVALENT, `campaign.py report` format)

- Infervisor: plowrt `0c6597b2`, packet `943e4fcffbb0`, built from
  [`recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`](../../../../recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml)
  (FP8 W8A8 weights, BF16 KV cache, 64K context, prefix cache on, packed multi-prompt prefill).
  Consolidated decode and prefill improvements: fused router GEMM+TopK, dense native route,
  decode MoE tiles, fused hipBLASLt projections, KDA carry schedule, and token-blocked decode fold.
- Baseline: vLLM 0.31.1rc1 ROCm (`vllm/vllm-openai-rocm:nightly-rocm100`), TP8, AITER on,
  prefix caching off, same 8x MI350X host, same vLLM benchmark client.
- Quality: FP32-reference gate PASS for the served packet (`gates.json` sha256 `62ceaf3d9d33`;
  KL mean 0.01987 vs vLLM 0.03677, top1_decisive 0.9956 vs vLLM 0.9868, needle_acc 1.0 vs 1.0).
  Long-context needles 36/36, split needles 18/18, prefix cache reuse pass. 0 collective errors.

Ratios are Infervisor / vLLM, from matched benchmark runs.

| Cell (ISL / OSL, Conc) | Infervisor tok/s | vLLM tok/s | Total throughput ratio | Infervisor TTFT P99 | vLLM TTFT P99 | TTFT ratio | Infervisor TPOT P99 | vLLM TPOT P99 | TPOT ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4096 / 128, c8** | **15,346.7** | 14,650.4 | **1.048x (+4.8%)** | **1,141.5 ms** | 2,290.6 ms | **0.50x (2.0x faster)** | **17.90 ms** | 25.48 ms | **0.70x** |
| **1024 / 128, c8** | **5,554.7** | 5,255.7 | **1.057x (+5.7%)** | **361.0 ms** | 1,764.9 ms | **0.20x (4.9x faster)** | **12.40 ms** | 19.29 ms | **0.64x** |

### Evidence (campaign scratch outside repo)

- Infervisor serving run: `/home/lava/bench/glm53flash/v23-serving/` (`summary.txt`, `server.log`, per-cell `bench.json`)
- Correctness gates:
  - FP32-reference gate: `/home/lava/bench/glm53flash/fp32ref/gate-v23-int/` (`gates.json`)
  - Long-context needle probe (36/36): `/home/lava/bench/glm53flash/needle-v23-int/`
  - Split needle probe (18/18): `/home/lava/bench/glm53flash/needle-split-v23-int/`
  - Prefix cache probe: `/home/lava/bench/glm53flash/prefix-v23-int/`
