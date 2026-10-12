# GLM-5.3-Flash on 8x MI350X (gfx950), TP8: plowrt vs vLLM

Final recipe: [`recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`](../../../../recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml)
(status `validated`; packet sha256 451b1b07…, max-ctx 65536). Campaign log:
[`docs/amd/glm53-flash-mi350x.md`](../../../amd/glm53-flash-mi350x.md).

Baseline: vLLM 0.31.1rc1 ROCm (Docker `vllm/vllm-openai-rocm:nightly-rocm100`), TP8, AITER on,
prefix caching off, same box, same client (`scripts/bench/llm_grid.sh`, greedy, OSL 128, two
repeats). Strict reports: `campaign.py report`, every cell MATCHED and EQUIVALENT.

## Qualified wins

None. plowrt's total throughput is 0.07-0.26x vLLM's in every cell
([comparison.csv](comparison.csv), 19 cells over isl 1024/4096/16384 at c1..c128 and
32768/61440 at c1/c8). Two isolated metrics favour plowrt but are not wins under the strict
format: TTFT p99 at isl 4096 c1 (443 vs 512 ms) and TPOT p99 at isl 16384 c128 (301 vs 314 ms).

## Accuracy

FP32-reference gate (46 cases incl. 18 needles to 15872 tokens): PASS, plowrt kl_mean 0.0140
vs vLLM 0.0368, top1_decisive 0.9967 vs 0.9868. Long-context needles to 61440 tokens: plowrt
36/36, vLLM 36/36.

## Where the time goes

- c1 decode: 28 ms/token vs vLLM 7 ms: ~1850 interpreter packets per step at ~12-15 us each.
- Prefill: 335 ms at 4k (vLLM TTFT ~105 ms), 2.75 s at 32k (0.7 s); the hyper-connection mix,
  Sinkhorn pre, KDA chunk carry and grouped MoE are each 2-30x off their roofline.
- c32+: requests prefill one at a time between decode steps, so TTFT p99 reaches 10-95 s and
  pulls throughput down; vLLM batches chunked prefill into its decode steps.

Raw evidence (outside the repo): `/home/lava/bench/glm53flash/` (grids `plow-v7`, `vllm-tp8`,
`fp32ref/gate-v7`, `needle-v7`, `validate-v7`).
