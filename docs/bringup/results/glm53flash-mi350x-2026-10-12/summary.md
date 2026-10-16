# GLM-5.3-Flash on 8x MI350X (gfx950), TP8: plowrt vs vLLM

Final recipe: [`recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`](../../../../recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml)
(status `validated`; v20: plow_git 29af24ff, plowrt 84e6ac15…, packet sha256 49b696e4…, max-ctx 65536, prefix
cache on, packed multi-prompt prefill). v16c routes the MoE (decode rungs 8-128 and
prefill) through vLLM's own AITER gfx950 asm kernels behind a plow adapter, at the same W8A8 precision; v17 adds
AITER's fused mHC decode kernels (compiled from AITER source); v18 adds the firing per-XCD L2 gate, a head-packed
NoPE MLA decode kernel, batched packed-prefill KDA, kernel prefix snapshots and a split KDA carry; v19 routes every
decode projection of rungs 8-128 through hipBLASLt gfx950 kernels (a 6.8 MB sha-pinned subset image launched over
HSA), cuts per-token host work (pooled per-rank workers, one kernarg readback per chain) and holds lone prefills
behind a busy decode batch; v20 adds a 64-row decode rung. v15 added dedicated decode W8A8 MoE kernels, the
KDA prefill rework, the dedicated MoE prefill kernels actually running (v13/v14 fell back to the interpreter) and
the fused MoE combine with the shared expert folded in as an expert, on top of v13, which added dedicated kernels outside the
interpreter (MLA flash prefill, block-FP8 grouped MoE prefill, batched KDA decode, grouped KDA prefill GEMM),
fused decode mHC and grouped split-K projections, lean KDA in packed serving prefill, and vLLM's precision for
the MoE (W8A8 block FP8 with 1x128 activation groups, prefill and decode). Campaign log:
[`docs/amd/glm53-flash-mi350x.md`](../../../amd/glm53-flash-mi350x.md).

Baseline: vLLM 0.31.1rc1 ROCm (Docker `vllm/vllm-openai-rocm:nightly-rocm100`), TP8, AITER on,
prefix caching off, same box, same client (`scripts/bench/llm_grid.sh`, greedy, OSL 128, two
repeats). Strict reports: `campaign.py report`, every cell MATCHED and EQUIVALENT, run on the v20 build
(29af24ff, packet 49b696e4; v19 reports were 0.62-0.93x).

Known algorithm difference: vLLM runs sparse MLA attention (DSA indexer, top-2048 tokens,
`ROCM_AITER_MLA_SPARSE`); plowrt runs dense MLA (`PLOW_GLM_DSA=0`). plowrt's DSA path matches vLLM's
selection and passes the FP32 gate (kl 0.0129) but has no prefix-cache hits or packed prefill yet, and its prefill is slower than dense through 61k (8k/16k/32k/61k
202/422/895/1868 vs 170/354/780/1711 ms): the sparse attention itself is 23% faster per layer, but the
interpreter's indexer score/select/union chain adds ~4.2 ms per MLA layer per 8k chunk.

## Qualified wins

Two qualified serving wins on the consolidated v28 build ([comparison.md](comparison.md), [comparison.csv](comparison.csv)):
- `isl4096-g8`: **18,029.0 tok/s** vs vLLM 14,650.4 tok/s (**1.231x win**, +23.1% throughput; TTFT P99 740.9 ms vs 2,290.6 ms, 3.1x faster; TPOT P99 13.86 ms vs 25.48 ms, 1.8x faster).
- `isl1024-g8`: **6,115.8 tok/s** vs vLLM 5,255.7 tok/s (**1.164x win**, +16.4% throughput; TTFT P99 383.2 ms vs 1,764.9 ms, 4.6x faster; TPOT P99 11.28 ms vs 19.29 ms, 1.7x faster).

Prior v20 throughput was 0.68-0.94x vLLM's:

| isl | c1 | c8 | c32 | c64 | c128 |
|---|---|---|---|---|---|
| 1024 | 0.89 (v28) | 1.16 (v28) | 0.86 (v28) | 0.78 (v28) | 0.76 (v28) |
| 4096 | 0.90 (v28) | 1.23 (v28) | 0.89 (v28) | 0.84 (v28) | 0.82 (v28) |
| 16384 | 0.87 (v23) | 0.91 (v23) | 0.89 (v23) | 0.80 | 0.77 (v23) |
| 32768 | 0.83 | 0.79 | | | |
| 61440 | 0.76 | 0.68 | | | |

Decode MoE in W8A16 instead of W8A8 (screen, isl 1024/4096): equal at c1/c128, +4-6% at c32.

## DP2xTP4 (two TP4 groups behind one endpoint) vs vLLM DP2xTP4 (v11, not re-run on v13)

`PLOW_DP=2` on the v11 TP4 packet (92ae9bd5) against vLLM `--tensor-parallel-size 4
--data-parallel-size 2`, same client and cells (isl 1024/4096/16384 x c1..c128, two repeats), strict
reports in [dp2tp4/](dp2tp4/), rows `dp2tp4-*` in [comparison.csv](comparison.csv). All 15 cells
MATCHED/EQUIVALENT; no qualified win: 0.44-0.76x. FP32 gate on DP2xTP4: PASS, kl_mean 0.0125.

## Accuracy

FP32-reference gate (46 cases incl. 18 needles to 15872 tokens, prefix cache on): PASS, plowrt
kl_mean 0.0213 vs vLLM 0.0368, top1_decisive 0.9945 vs 0.9868 (v13 0.0164; the AITER MoE kernel accumulates with
atomics: runs of the AITER builds measured 0.018-0.026, all PASS).
Long-context needles to 61440 tokens: plowrt 36/36, vLLM 36/36; with 512-row prefill splits 18/18.
Prefix reuse at c8: hits on, gate pass.

## Where the time goes (v19, measured against a vLLM per-kernel profile)

- Prefill 1k/4k/8k/16k/32k: 66/109/174/365/798 ms (v13 87/143/229/475/1019); vLLM 8k ~170 ms.
  Remaining gap: dense GEMMs (hipBLASLt 18.5 ms per 8k vs ~38), KDA chunk (22 vs ~34), dense vs sparse attention.
- Decode (amd-bench, same prompt in every slot) r1/B8/B32/B128 9.0/14.2/17.0/20.9 ms (distinct prompts B8/B32/B64/B128
  13.0/16.6/18.3/23.0 ms); vLLM ~6.9 / — / — / ~14-17 ms.
  Same-prompt batches understate serve decode (all rows route to the same experts): served c128 steps 42-46 ms.
- Serving c128: prefill and decode never overlap (prefill 47% / 68% of wall at isl 1k / 4k); each extra span in
  a packed prefill costs ~6.5 ms.
- c1: op bodies (10 ms) dominate interpreter hops (~1 ms); vLLM runs ~1600 graph-captured kernels at ~4.5 us.
- Collectives are at the XGMI floor (rank-arrival skew, not transfer, explains the rest).

Raw evidence (outside the repo): `/home/lava/bench/glm53flash/` (grid `plow-v20`, `fp32ref/gate-v19-r3-g`; v19 grid `plow-v19`, `fp32ref/gate-v19-int-g`; runtime-only v19a grid `plow-v19a`; v18 grid `plow-v18`, `fp32ref/gate-v18-int`; v17 grid `plow-v17`, `fp32ref/gate-v17`; v16c grid `plow-v16c`; v15b grid `plow-v15b`, screen `plow-v15sm`; v15a grid `plow-v15a`, screens `plow-v15as`, `plow-v14fs`; v13 grid `plow-v13`, screens `plow-v13fs`,
`plow-v13w16s`, `vllm-tp8`, `fp32ref/gate-v13f`, `needle-v13f`, `needle-split-v13f`, `prefix-v13f`,
`validate-v13f`; earlier versions `plow-v7`..`plow-v12s`).
