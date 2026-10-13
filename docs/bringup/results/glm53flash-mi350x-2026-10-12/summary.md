# GLM-5.3-Flash on 8x MI350X (gfx950), TP8: plowrt vs vLLM

Final recipe: [`recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`](../../../../recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml)
(status `validated`; plowrt 84c775e9, packet sha256 980625b3…, max-ctx 65536, prefix cache on).
Campaign log: [`docs/amd/glm53-flash-mi350x.md`](../../../amd/glm53-flash-mi350x.md).

Baseline: vLLM 0.31.1rc1 ROCm (Docker `vllm/vllm-openai-rocm:nightly-rocm100`), TP8, AITER on,
prefix caching off, same box, same client (`scripts/bench/llm_grid.sh`, greedy, OSL 128, two
repeats). Strict reports: `campaign.py report`, every cell MATCHED and EQUIVALENT.

## Qualified wins

None. plowrt's total throughput is 0.08-0.31x vLLM's in every cell
([comparison.csv](comparison.csv), 19 cells over isl 1024/4096/16384 at c1..c128 and
32768/61440 at c1/c8). Two isolated metrics favour plowrt but are not wins under the strict
format: TTFT p99 at isl 4096 c1 (420 vs 512 ms) and TPOT p99 at isl 16384 c128 (244 vs 314 ms).

## Accuracy

FP32-reference gate (46 cases incl. 18 needles to 15872 tokens, prefix cache on): PASS, plowrt
kl_mean 0.0142 vs vLLM 0.0368, top1_decisive 0.9956 vs 0.9868. Long-context needles to 61440
tokens: plowrt 36/36, vLLM 36/36. Cross-slot prefix reuse at c8: cached streams equal cold.

## Where the time goes

- c1 decode: 26-30 ms/token vs vLLM 7 ms: ~1850 interpreter packets per step at ~12-15 us each.
- Prefill: 198 ms at 4k (vLLM TTFT ~105 ms), 2.1 s TTFT at 32k (1.1 s); grouped MoE and the KDA
  chunk path remain 2-10x off their roofline.
- c32+: requests prefill one at a time between decode steps, so TTFT p99 reaches 9-267 s and
  pulls throughput down; vLLM batches chunked prefill into its decode steps.

Raw evidence (outside the repo): `/home/lava/bench/glm53flash/` (grids `plow-v8`, `vllm-tp8`,
`fp32ref/gate-v8`, `needle-v8`, `prefix-check2-v8`; v7 in `plow-v7`, `gate-v7`, `needle-v7`).
