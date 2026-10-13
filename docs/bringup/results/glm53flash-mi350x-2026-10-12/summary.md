# GLM-5.3-Flash on 8x MI350X (gfx950), TP8: plowrt vs vLLM

Final recipe: [`recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`](../../../../recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml)
(status `validated`; v11: plowrt ac229648, packet sha256 9328c621…, max-ctx 65536, prefix cache on,
back-to-back prefill with an 8192-row tick budget, end-of-prompt prefix arm, packed multi-prompt
prefill with decode rows riding it). Campaign log:
[`docs/amd/glm53-flash-mi350x.md`](../../../amd/glm53-flash-mi350x.md).

Baseline: vLLM 0.31.1rc1 ROCm (Docker `vllm/vllm-openai-rocm:nightly-rocm100`), TP8, AITER on,
prefix caching off, same box, same client (`scripts/bench/llm_grid.sh`, greedy, OSL 128, two
repeats). Strict reports: `campaign.py report`, every cell MATCHED and EQUIVALENT.

## Qualified wins

None. plowrt's total throughput is 0.28-0.61x vLLM's ([comparison.csv](comparison.csv), 19 cells
over isl 1024/4096/16384 at c1..c128 and 32768/61440 at c1/c8): 0.60-0.61x at c1 up to 16k,
0.40-0.50x at c8, 0.28-0.49x at c32-c128 (v10: 0.25-0.62x). Isolated metrics favouring plowrt,
not wins under the strict format: TTFT p99 at isl 1024 c8 (542 vs 1765 ms), isl 4096 c1 (182 vs
512 ms), isl 4096 c8 (1556 vs 2291 ms) and isl 16384 c1 (624 vs 758 ms).

## DP2xTP4 (two TP4 groups behind one endpoint) vs vLLM DP2xTP4

`PLOW_DP=2` on the v11 TP4 packet (92ae9bd5, same knobs as the TP8 recipe, PF_INTERLEAVE 8192)
against vLLM `--tensor-parallel-size 4 --data-parallel-size 2`, same client and cells (isl
1024/4096/16384 x c1..c128, two repeats), strict reports in [dp2tp4/](dp2tp4/), rows `dp2tp4-*` in
[comparison.csv](comparison.csv). All 15 cells MATCHED/EQUIVALENT; no qualified win: plowrt
0.72-0.76x at c1, 0.64-0.73x at c8, 0.44-0.73x at c32-c128 (v10: 0.42-0.75x). FP32 gate on DP2xTP4:
PASS, kl_mean 0.0125. plowrt DP2xTP4 beats plowrt TP8 at c32+ (isl16384 c128 30416 vs 22196 tok/s);
vLLM's own DP2xTP4 is 0.66-0.88x its TP8.

## Accuracy

FP32-reference gate (46 cases incl. 18 needles to 15872 tokens, prefix cache on): PASS, plowrt
kl_mean 0.0148 vs vLLM 0.0368, top1_decisive 0.9978 vs 0.9868 (v10 0.0134; DP2xTP4 v10 0.0130). Long-context needles to 61440
tokens: plowrt 36/36, vLLM 36/36; with 512-row prefill splits 18/18. Prefix reuse at c8: cached
streams equal cold.

## Where the time goes

- c1 decode: 11.5 ms/token vs vLLM 7 ms; about 800 interpreter hops per step at ~5 us each.
- Batched decode step: B=8 22.9, B=32 30.2, B=128 40.3 ms (v9 31.7/52.4/84.7).
- Prefill: 167 ms at 4k (vLLM TTFT ~105 ms); grouped MoE 4-5x and attention prefill ~10x off their roofline.
- c32+: TPOT p99 49-287 ms vs vLLM 15-85 ms; prefill rows per step plus scheduler overhead dominate
  (c128 isl1024: 40 ms decode step vs 117 ms TPOT p50).

Raw evidence (outside the repo): `/home/lava/bench/glm53flash/` (grids `plow-v11`, `vllm-tp8`,
`fp32ref/gate-v11c8`, `needle-v11c8`, `needle-split-v11c8`, `prefix-check2-v11c8`, `validate-v11c8`;
earlier versions in `plow-v7`..`plow-v10`).
