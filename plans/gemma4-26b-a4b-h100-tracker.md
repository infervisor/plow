# Gemma-4-26B-A4B vs vLLM on H100 — campaign tracker

Started: 2026-09-19. Branch: `worktree-gemma4-26b-beat-vllm` (base `origin/main` 91f03b9c).
Checkpoint: `/opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it` (BF16, 2 shards, 51.6 GB).
Goal: beat vLLM 0.28 apples-to-apples (same client, same precision) on every metric,
rung by rung to 8K, then serving throughput at high concurrency.

## Geometry (verified against the safetensors index, not the config)

30 layers; full attention at 5/11/17/23/29 (5 full, 25 sliding); sliding window 1024.
Every layer is a HYBRID dense-MLP + MoE block. Tied embeddings. No v_proj on full layers.

| | 26B-A4B | 12B (for contrast) |
|---|---|---|
| hidden H | 2816 | 3840 |
| n_head | 16 | 16 |
| sliding head_dim / kv heads / GQA | 256 / 8 / **2** | 256 / 8 / 2 |
| full head_dim / kv heads / GQA | 512 / 2 / **8** | 512 / 1 / **16** |
| dense MLP inter | 2112 | 15360 |
| MoE | 128 experts, top-8, I_moe 704 | none |
| layers | 30 | 48 |
| vocab | 262144 | 262144 |

## Kernel dims that must be supported

Weight shapes, all confirmed from the checkpoint header:

| site | shape (N, K) | count | status |
|---|---|---:|---|
| q_proj sliding | 4096 x 2816 | 25 | ok |
| k_proj sliding | 2048 x 2816 | 25 | ok |
| v_proj sliding | 2048 x 2816 | 25 | ok |
| o_proj sliding | 2816 x 4096 | 25 | ok |
| q_proj full | 8192 x 2816 | 5 | ok |
| k_proj full | 1024 x 2816 | 5 | ok (V = raw k_proj, k_eq_v) |
| o_proj full | 2816 x 8192 | 5 | ok |
| dense gate/up | 2112 x 2816 | 60 | ok |
| dense down | 2816 x 2112 | 30 | ok |
| router | 128 x 2816 | 30 | ok |
| experts gate_up | [128, 1408, 2816] | 30 | see MoE below |
| experts down | [128, 2816, 704] | 30 | see MoE below |
| lm_head (tied) | 262144 x 2816 | 1 | ok |

### Checks that PASS

* **GEMV MMA contract** (`op_gemv_mma.cuh`): needs `K % 32 == 0` and `N % 8 == 0`.
  Every K here (2816, 4096, 8192, 2112, 704) is a multiple of 32; every N is a
  multiple of 8. The tensor-core decode walk applies unchanged.
* **Sliding attention** is GQA **2** — identical to 12B, so
  `interp_sm90a_pfattn_hd256_gqa2_bkv32.cu` (which hardcodes
  `n_head=16, n_kv_head=8`) matches 26B exactly. The GQA2-pair role is reusable as is.
* **Decode attention dispatch** (`interp_sm120.cu` `PLOW_DOP_FLASH_DECODE`) requires
  `gqa % PLOW_NV_FA_GF_FULL == 0` at hd512. The sm_90a decode object is built with
  `PLOW_NV_FA_GF_FULL=4`; 26B's full-attn gqa is 8, and `8 % 4 == 0` — passes.
  hd256 takes the `gqa % 2 == 0` arm, gqa 2 — passes.
* **MoE N-tiling** (`op_moe_sm90.cuh`): `tiles_n = ceil(I_moe/128)` with a `cc < I_moe`
  guard, so I_moe 704 is correct (6 tiles, last one half-used — a perf tax, not a bug).
  H 2816 = 22 x 128 exactly.
* **MoE decode bodies** (`d_moe_router_gemma*`, `d_moe_expert_glu_gemma`,
  `d_moe_expert_down_gemma`) were authored for this model — the source comments cite
  H=2816 and 8-of-128 routing directly.

### Checks that FAIL, and what they cost

1. **hd512 px4 BQ64 role is 12B-only.** `interp_sm90a_pfattn_hd512.cu` hardcodes
   `n_kv_head = 1` and `n_head = 16` under `PLOW_NV_FA512_PX4_BQ64` (lines 144-160).
   26B has 2 global KV heads. The role must be emitted OFF for 26B
   (`PLOW_GEMMA4_SM90_HD512_PX4_BQ64_ROLE=0`); the generic hd512 body reads runtime
   `n_kv_head` and is correct. Cost: 26B starts without the best 12B prefill
   attention role. A GQA-8 px4 variant is the obvious follow-up.
2. **The fused GLU prefill role is 12B-only.** `interp_sm90a_pfgemm_glu_gemma4.cu`
   pins `plow_pfgemm_glu_gemma4_n = 15360`, `_k = 3840` and refuses any other shape
   at dispatch. 26B is N=2112, K=2816. Not a blocker: the 12B production default is
   already `PLOW_NO_GLU_FUSE=1 + PLOW_EMIT_PREFILL_CUBLASLT=1` (GLU role OFF), which
   is shape-agnostic. 26B should start there.
3. **MoE decode GLU is a dot8 CUDA-core walk**, not tensor-core — the same shape of
   bug that made 12B dense decode compute-bound at B>=4 before `PLOW_NV_GEMV_MMA`.
   Expect MoE decode TPOT to scale with batch. This is the predicted #1 lever for
   high concurrency, to be confirmed by measurement, not assumed.

### Levers this geometry opens that 12B did not have

* `PLOW_NV_FA_GF_FULL=8` exactly matches 26B's full-attn GQA (12B's 16 needs
  `PLOW_NV_FA_GF16_BENCH`). One grouped pass instead of two.
* `PLOW_NV_FA_TC_GQA8_HD512` — an opt-in tensor-core hd512 decode candidate that is
  `static_assert`ed to `D == 512 && GF == 8`. It is unusable on 12B and native on 26B.

## Roofline (H100 SXM5, 3.35 TB/s datasheet, 989 BF16 TFLOP/s)

Active params per decoded token: 3.08B in the layers + 0.74B lm_head = 3.82B
=> 7.64 GB read per token at BF16 => **2.28 ms/token memory floor at B=1**.
(12B reads all 12B params = 24 GB => 7.2 ms floor; its measured TPOT is 10.5-12.7 ms.)

The MoE twist for serving: expert selection decorrelates across a batch, so by
B~16 nearly all 128 experts are touched and the per-step expert traffic saturates at
30 x 128 x 2112 x 2816 x 2 B = **45.6 GB/step ~ 13.6 ms**, independent of batch.
Decode TPOT therefore floors out while throughput keeps scaling — the opposite shape
of the 12B curve, and it sets where the high-concurrency work has to aim.

## What the emit actually hit (2026-09-19)

Three blockers, none of them visible by reading — each found by running the emit.
All three are fixed in `045a38e3`.

1. **cuBLASLt prefill admitted nothing.** `CUBLASLT_PREFILL_GEMMA4_SHAPES` is a
   hardcoded 8-entry list of 12B projections, every one keyed on hidden 3840.
   26B is 2816-keyed, so zero segments qualified and the emit tripped
   `assert!(selected > 0, "no eligible dense prefill projections")`. Added the
   parallel 26B list; the emit now admits **1025 projection segments**.
2. **Packed prefill was silently dropped**, which left
   `interp_sm90a_pfpackedseg.cubin` unbuilt and failed the object step with no
   error text (the script's `test -f` precondition under `set -e`). Cause: the
   live-KV manifest's direct-operand audit did not name the Gemma MoE opcodes.
   They already existed and were already implemented — they had simply never
   been emitted through this path. Audited and added; see the commit.
3. **The emit default router is not bit-identical.** `gemma_moe_router_exact`
   defaults false, so `MoeRouterGemmaScoreFast` is emitted, although its own
   opcode doc says the changed reduction association is not bit-identical and
   that packets should opt in only for experiments. The recipe now pins
   `PLOW_GEMMA_MOE_ROUTER_EXACT=1`; the whole router is ~0.5 ms of a ~7.9 ms
   decode step, so this costs nothing worth defending against.

Not a blocker but worth recording: the rewrite/egglog path refuses this model
("Gemma 4 MoE routing and exhaustive expert checkpoint binding are not
implemented; refusing dense or representative-expert fallback"), so the emit
falls back to the hand fusions. That refusal is the right behaviour — a
representative-expert fallback would produce a fast, wrong packet.

## Corrected roofline, and why the old one could not be trusted

`roofline.py` keyed off a hand-maintained table that listed 26B-A4B as 26B
dense / 46 layers / hidden 5120. Against an actual 3.82B active / 30 / 2816
that is a 6.8x error in the decode ceiling — every "% of roofline" computed
from it was fiction. It now derives the spec from the checkpoint's config.json,
counts MoE active params at top-k, caps sliding-layer KV at the window rather
than the context, and bands the sliding attention FLOPs.

Cross-check on the 12B, whose measurements are already recorded: derived 11.91B
active; decode floor 7.12 ms vs vLLM's measured 10.55 ms TPOT (1.48x); 4K
prefill floor 103.6 ms vs its 170.2 ms TTFT (1.64x). Both ratios are plausible,
and the old table (40 layers, 30 heads, 10 kv) could not have produced them.

| ctx | 26B-A4B decode floor | 26B-A4B prefill floor | 12B decode | 12B prefill |
|---:|---:|---:|---:|---:|
| 128 | 2.29 ms | 2.29 ms | 7.12 | 7.13 |
| 1024 | 2.35 | 8.44 | 7.21 | 25.5 |
| 4096 | 2.37 | 34.8 | 7.22 | 103.6 |
| 8192 | 2.39 | 72.4 | 7.24 | 211.7 |

### The MoE serving curve

Routing decorrelates across a batch, so a decode step streams the UNION of the
batch's expert choices: expected `n*(1-(1-k/n)^B)` experts.

| B | experts touched | weight bytes | step floor | per-token |
|---:|---:|---:|---:|---:|
| 1 | 8.0 / 128 | 7.64 GB | 2.28 ms | 2.28 ms |
| 4 | 29.1 | 15.18 | 4.53 | 1.13 |
| 16 | 82.4 | 34.20 | 10.20 | 0.64 |
| 32 | 111.8 | 44.67 | 13.33 | 0.42 |
| 64 | 125.9 | 49.73 | 14.84 | 0.23 |

TPOT floors out near 13-15 ms while per-token cost keeps falling — the opposite
shape from 12B, and it is where the throughput work has to aim.

## Status

| step | state |
|---|---|
| Kernel-dims audit | **done** |
| plowc/plowrt release build | done |
| Emit + objects + role emit | **done** (`045a38e3`) |
| Roofline model corrected + validated on 12B | **done** |
| cuBLASLt probe (leased) | pending plowrt rebuild |
| vLLM 0.28 26B reference (gpulease) | script written, not run |
| Rung ladder 128/1024/4096/8192 | not started |
| Roofline compare + kernel push | not started |
| High concurrency / throughput | not started |

## Protocol

Same as the 12B campaign: `scripts/campaign/campaign.py` (build/bench/compare/ledger),
one TOML recipe per cell, every GPU run under `perf-data/tools/gpulease -n 1`,
`--profile realtime|throughput` always passed, same precision on both sides,
`PLOW_PREFIX_CACHE=0` for any vLLM-matched cell.
