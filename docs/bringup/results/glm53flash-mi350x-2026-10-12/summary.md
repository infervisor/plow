# GLM-5.3-Flash on 8x MI350X (gfx950), TP8: plowrt vs vLLM

Final recipe: [`recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`](../../../../recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml)
(status `validated`; v10: plowrt 927f76eb, packet sha256 e4786a2c…, max-ctx 65536, prefix cache on,
back-to-back prefill, end-of-prompt prefix arm, packed multi-prompt prefill with decode rows riding it). Campaign log:
[`docs/amd/glm53-flash-mi350x.md`](../../../amd/glm53-flash-mi350x.md).

Baseline: vLLM 0.31.1rc1 ROCm (Docker `vllm/vllm-openai-rocm:nightly-rocm100`), TP8, AITER on,
prefix caching off, same box, same client (`scripts/bench/llm_grid.sh`, greedy, OSL 128, two
repeats). Strict reports: `campaign.py report`, every cell MATCHED and EQUIVALENT.

## Qualified wins

None. plowrt's total throughput is 0.25-0.62x vLLM's ([comparison.csv](comparison.csv), 19 cells
over isl 1024/4096/16384 at c1..c128 and 32768/61440 at c1/c8): 0.60-0.62x at c1 up to 16k,
0.26-0.46x at c8, 0.25-0.33x at c32-c128. Isolated metrics favouring plowrt, not wins under the
strict format: TTFT p99 at isl 1024 c1 (111 vs 141 ms), isl 1024 c8 (729 vs 1765 ms), isl 4096 c1
(184 vs 512 ms) and isl 4096 c8 (1973 vs 2291 ms).

## Accuracy

FP32-reference gate (46 cases incl. 18 needles to 15872 tokens, prefix cache on): PASS, plowrt
kl_mean 0.0134 vs vLLM 0.0368, top1_decisive 0.9989 vs 0.9868 (DP2xTP4: 0.0130). Long-context needles to 61440
tokens: plowrt 36/36, vLLM 36/36; with 512-row prefill splits 18/18. Prefix reuse at c8: cached
streams equal cold.

## Where the time goes

- c1 decode: 11.5 ms/token vs vLLM 7 ms; about 800 interpreter hops per step at ~5 us each.
- Batched decode step: B=8 22.9, B=32 30.2, B=128 40.3 ms (v9 31.7/52.4/84.7).
- Prefill: 167 ms at 4k (vLLM TTFT ~105 ms); grouped MoE 4-5x and attention prefill ~10x off their roofline.
- c32+: TPOT p99 49-287 ms vs vLLM 15-85 ms; prefill rows per step plus scheduler overhead dominate
  (c128 isl1024: 40 ms decode step vs 117 ms TPOT p50).

Raw evidence (outside the repo): `/home/lava/bench/glm53flash/` (grids `plow-v10`, `vllm-tp8`,
`fp32ref/gate-v10`, `needle-v10`, `needle-split-v10`, `prefix-check2-v10`, `validate-v10`; earlier
versions in `plow-v7`..`plow-v9`).
