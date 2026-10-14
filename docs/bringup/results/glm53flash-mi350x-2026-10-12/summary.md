# GLM-5.3-Flash on 8x MI350X (gfx950), TP8: plowrt vs vLLM

Final recipe: [`recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`](../../../../recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml)
(status `validated`; v17: plow_git 8485e951, plowrt 001c991d…, packet sha256 6629af02…, max-ctx 65536, prefix
cache on, packed multi-prompt prefill with decode rows riding it). v16c routes the MoE (decode rungs 8-128 and
prefill) through vLLM's own AITER gfx950 asm kernels behind a plow adapter, at the same W8A8 precision; v17 adds
AITER's fused mHC decode kernels (compiled from AITER source). v15 added dedicated decode W8A8 MoE kernels, the
KDA prefill rework, the dedicated MoE prefill kernels actually running (v13/v14 fell back to the interpreter) and
the fused MoE combine with the shared expert folded in as an expert, on top of v13, which added dedicated kernels outside the
interpreter (MLA flash prefill, block-FP8 grouped MoE prefill, batched KDA decode, grouped KDA prefill GEMM),
fused decode mHC and grouped split-K projections, lean KDA in packed serving prefill, and vLLM's precision for
the MoE (W8A8 block FP8 with 1x128 activation groups, prefill and decode). Campaign log:
[`docs/amd/glm53-flash-mi350x.md`](../../../amd/glm53-flash-mi350x.md).

Baseline: vLLM 0.31.1rc1 ROCm (Docker `vllm/vllm-openai-rocm:nightly-rocm100`), TP8, AITER on,
prefix caching off, same box, same client (`scripts/bench/llm_grid.sh`, greedy, OSL 128, two
repeats). Strict reports: `campaign.py report`, every cell MATCHED and EQUIVALENT, run on the v17 build
(8485e951; v16c reports were 0.44-0.75x).

Known algorithm difference: vLLM runs sparse MLA attention (DSA indexer, top-2048 tokens,
`ROCM_AITER_MLA_SPARSE`); plowrt runs dense MLA (`PLOW_GLM_DSA=0`). plowrt's DSA path matches vLLM's
selection and passes the FP32 gate (kl 0.0129) but has no prefix-cache hits or packed prefill yet.

## Qualified wins

None. plowrt's total throughput is 0.47-0.76x vLLM's ([comparison.csv](comparison.csv), 19 cells
over isl 1024/4096/16384 at c1..c128 and 32768/61440 at c1/c8; v16c 0.44-0.75x, v15b 0.37-0.66x, v13 0.34-0.61x, v11 0.28-0.61x):

| isl | c1 | c8 | c32 | c64 | c128 |
|---|---|---|---|---|---|
| 1024 | 0.61 | 0.61 | 0.54 | 0.47 | 0.52 |
| 4096 | 0.62 | 0.71 | 0.64 | 0.60 | 0.67 |
| 16384 | 0.66 | 0.71 | 0.76 | 0.67 | 0.66 |
| 32768 | 0.66 | 0.71 | | | |
| 61440 | 0.66 | 0.64 | | | |

Decode MoE in W8A16 instead of W8A8 (screen, isl 1024/4096): equal at c1/c128, +4-6% at c32.

## DP2xTP4 (two TP4 groups behind one endpoint) vs vLLM DP2xTP4 (v11, not re-run on v13)

`PLOW_DP=2` on the v11 TP4 packet (92ae9bd5) against vLLM `--tensor-parallel-size 4
--data-parallel-size 2`, same client and cells (isl 1024/4096/16384 x c1..c128, two repeats), strict
reports in [dp2tp4/](dp2tp4/), rows `dp2tp4-*` in [comparison.csv](comparison.csv). All 15 cells
MATCHED/EQUIVALENT; no qualified win: 0.44-0.76x. FP32 gate on DP2xTP4: PASS, kl_mean 0.0125.

## Accuracy

FP32-reference gate (46 cases incl. 18 needles to 15872 tokens, prefix cache on): PASS, plowrt
kl_mean 0.0224 vs vLLM 0.0368, top1_decisive 0.9923 vs 0.9868 (v13 0.0164; the AITER MoE kernel accumulates with
atomics: runs of the AITER builds measured 0.018-0.026, all PASS).
Long-context needles to 61440 tokens: plowrt 36/36, vLLM 36/36; with 512-row prefill splits 18/18.
Prefix reuse at c8: 57344 hit tokens.

## Where the time goes (v17, measured against a vLLM per-kernel profile)

- Prefill 1k/4k/8k/16k/32k: 66/109/174/365/798 ms (v13 87/143/229/475/1019); vLLM 8k ~170 ms.
  Remaining gap: dense GEMMs (hipBLASLt 18.5 ms per 8k vs ~38), KDA chunk (22 vs ~34), dense vs sparse attention.
- Decode (amd-bench, same prompt in every slot) r1/B8/B32/B128 11.8/19.7/20.2/26.7 ms; vLLM ~6.9 / — / — / ~14-17 ms.
  Same-prompt batches understate serve decode (all rows route to the same experts): served c128 steps 42-46 ms.
- Serving c128: prefill and decode never overlap (prefill 47% / 68% of wall at isl 1k / 4k); each extra span in
  a packed prefill costs ~6.5 ms.
- c1: op bodies (10 ms) dominate interpreter hops (~1 ms); vLLM runs ~1600 graph-captured kernels at ~4.5 us.
- Collectives are at the XGMI floor (rank-arrival skew, not transfer, explains the rest).

Raw evidence (outside the repo): `/home/lava/bench/glm53flash/` (grid `plow-v17`, `fp32ref/gate-v17`, `needle-v17`, `prefix-v17`; v16c grid `plow-v16c`; v15b grid `plow-v15b`, screen `plow-v15sm`; v15a grid `plow-v15a`, screens `plow-v15as`, `plow-v14fs`; v13 grid `plow-v13`, screens `plow-v13fs`,
`plow-v13w16s`, `vllm-tp8`, `fp32ref/gate-v13f`, `needle-v13f`, `needle-split-v13f`, `prefix-v13f`,
`validate-v13f`; earlier versions `plow-v7`..`plow-v12s`).
