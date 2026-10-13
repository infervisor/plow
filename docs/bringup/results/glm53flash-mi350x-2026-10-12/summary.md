# GLM-5.3-Flash on 8x MI350X (gfx950), TP8: plowrt vs vLLM

Final recipe: [`recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`](../../../../recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml)
(status `validated`; v9: plowrt 70fdf9eb, packet sha256 1533860b…, max-ctx 65536, prefix cache on,
back-to-back prefill, end-of-prompt prefix arm). Campaign log:
[`docs/amd/glm53-flash-mi350x.md`](../../../amd/glm53-flash-mi350x.md).

Baseline: vLLM 0.31.1rc1 ROCm (Docker `vllm/vllm-openai-rocm:nightly-rocm100`), TP8, AITER on,
prefix caching off, same box, same client (`scripts/bench/llm_grid.sh`, greedy, OSL 128, two
repeats). Strict reports: `campaign.py report`, every cell MATCHED and EQUIVALENT.

## Qualified wins

None. plowrt's total throughput is 0.14-0.56x vLLM's ([comparison.csv](comparison.csv), 19 cells
over isl 1024/4096/16384 at c1..c128 and 32768/61440 at c1/c8): 0.52-0.56x at c1 up to 16k,
0.25-0.34x at c8, 0.14-0.24x at c32-c128. Isolated metrics favouring plowrt, not wins under the
strict format: TTFT p99 at isl 4096 c1 (211 vs 512 ms) and isl 1024 c8 (1191 vs 1765 ms).

## Accuracy

FP32-reference gate (46 cases incl. 18 needles to 15872 tokens, prefix cache on): PASS, plowrt
kl_mean 0.0138 vs vLLM 0.0368, top1_decisive 0.9989 vs 0.9868. Long-context needles to 61440
tokens: plowrt 36/36, vLLM 36/36; with 512-row prefill splits 18/18. Prefix reuse at c8: cached
streams equal cold.

## Where the time goes

- c1 decode: 12.7 ms/token vs vLLM 7 ms (was 26-28 ms before the inlined decode dispatch).
- Prefill: ~200 ms at 4k (vLLM TTFT ~105 ms), 2.0 s TTFT at 32k (1.1 s).
- c32+: prefill and decode are separate launches; a tick prefills up to 4096 rows, then decodes,
  so TPOT p99 reaches 130-410 ms; vLLM mixes chunked prefill into its decode batches. The B=128
  decode step is ~85 ms.

Raw evidence (outside the repo): `/home/lava/bench/glm53flash/` (grids `plow-v9`, `vllm-tp8`,
`fp32ref/gate-v9`, `needle-v9`, `needle-split-v9`, `prefix-check2-v9`, `validate-v9`; earlier
versions in `plow-v7`, `plow-v8`).
