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

## Is plow's architecture actually used for MoE routing / expert selection?

Partly. Measured, per layer at B=1 (`runtime/tests/moe_decode_gemma26b_bench.cu`):

| | ms/layer | share of MoE | achieved BW |
|---|---:|---:|---:|
| router score | 0.0461 | 45 % | **15.6 GB/s (0.47 % of peak)** |
| router topk | 0.0082 | 8 % | — (1 CTA) |
| expert GLU | 0.0336 | 33 % | 1888 GB/s (56 %) |
| expert DOWN | 0.0154 | 15 % | ~2059 GB/s |

**The ROUTER is 53 % of MoE decode time and the expert GEMMs are 47 %.** The GEMMs are
in good shape; the SELECTION is where the architecture is left on the table.

USED:
* Split router is default-on — the score GEMV runs on 16 CTAs instead of serializing on
  one. (`PLOW_GEMMA_MOE_ROUTER_FUSED` is the escape hatch BACK to one CTA, not an
  improvement — read the knob name carefully.)
* Fusion is selected where it pays: `MoeExpertGluNormGemma` folds the pre-FFN norm into
  the expert GLU, removing a packet boundary.
* Counter-gated overlap is real: decode is 519 ops / 608 edges, critical path 364, so
  ~30 % of ops are off the critical path (p50 = 2 concurrent, peak 3).

NOT USED:
* **No compile-time specialisation of MoE geometry.** The packet bakes 208
  `PLOW_PACKET_*` constants including `PLOW_PACKET_GQA 8` for attention, and NOT ONE
  MoE constant: `n_exp=128`, `top_k=8`, `I_moe=704` arrive as runtime `unsigned` args,
  so the top-k loop, the score reduction and the expert-table indexing cannot be
  unrolled or specialised. This is exactly the "compile a packet for this model"
  advantage, unused on the most shape-dependent op family in the model.
* **The router cannot fill the grid at low batch.** `max_useful = (nrow*n_exp)/8`
  capped at `n_cu` (devgen `lib.rs` `gemma_moe_router_split_plan`) gives 16 CTAs at
  B=1 and only reaches 132 from B ~ 12; `PLOW_GEMMA_MOE_ROUTER_BLOCKS` is clamped to
  it. NOT an oversight — going wider means splitting each expert's H=2816 reduction
  across CTAs, which breaks the "exact fmaf association" the body deliberately keeps.

CALIBRATION, so nobody over-invests: MoE is only ~6 % of the decode step (3.10 ms of
48.8 ms). A PERFECT router saves ~1.3 ms, i.e. **~2.7 % of TPOT**. The unattributed
~43 ms dominates everything. Instrument the CUDA decode tick first.

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

## RESOLVED — 26B serves on H100 (2026-09-19)

Gemma-4-26B-A4B now answers correctly through the OpenAI endpoint and passes the
bench client's coherence gate. 47.0 GiB weights + 27.5 GiB KV, `grid_flash=132`,
`fa512=true`, `fa256_gqa2=true`, `bytes_per_token=225280` (the correct hybrid KV
figure: 5 full x 2 x 512 + 25 sliding x 8 x 256, x2 for K/V, x2 bytes).

TWO bugs, both the same shape — **an opcode the packet uses is compiled out of the
object that must run it**, and neither reachable from a dense checkpoint, which is
why 12B was never affected:

1. **Prefill (recipe error).** The five MoE prefill opcodes are guarded by
   `#if !PLOW_NV_SEG_GEMM && !PLOW_NV_FA_ONLY && !PLOW_NV_FATLITE`. The build makes
   exactly three object flavours and each sets one of those flags: the GEMM object
   (`SEG_GEMM=1`), the attention object (`FA_ONLY=1`) and — with
   `PLOW_BUILD_FATLITE=1`, which `build_sm90a_gemma4_segments.sh:75` feeds into
   `-DPLOW_NV_FATLITE` — the fat packed-seg object too. No object implemented MoE
   prefill. Recipe now sets `PLOW_BUILD_FATLITE=0`.
2. **Decode (upstream bug, `interp_sm120.cu`).** `PLOW_HAS_MOE_GEMMA` ORed only the
   two BASE opcodes, with a comment asserting the family is all-or-nothing. The
   emitter actually picks VARIANTS — the split `MoeRouterGemmaScore`+`Topk` router
   and the norm-fused `MoeExpertGluNormGemma`. Evaluated against the real config with
   the decode object's defines (`-DPLOW_BUCKET_DECODE=1`): `..._SCORE 1`,
   `..._TOPK 1`, `..._EXPERT_GLU_NORM_GEMMA 1`, old gate `(0 || 0) = 0`. Fixed by
   naming every member; verified 1 for the decode object and still 0 for prefill
   objects, so no prefill object grows.

Diagnostic note worth keeping: comparing compiled object SIZE to test the decode gate
was misleading — the probe was built with `PLOW_BUCKET_DECODE=0`, which is a PREFILL
object, where the gate is correctly 0 either way. Evaluate `PLOW_HAS_*` macros with
the target object's own defines (`cpp -I <assets> -DPLOW_BUCKET_DECODE=1`).

## First measured 26B rung ladder vs vLLM

Same client, same flags, prefix caching off both sides. vLLM column from
`perf-data/campaign/gemma4-26b-a4b.h100.reference-vllm028-bf16.csv`.

| cell | TTFT plow / vLLM | TPOT plow / vLLM | tok/s plow / vLLM |
|---|---:|---:|---:|
| 128 / C1 | 42.21 / 39.52 (**1.07x**) | 48.81 / 5.03 (9.7x) | 20.5 / 188.5 |
| 128 / C4 | 84.75 / 73.31 (**1.16x**) | 50.66 / 7.24 (7.0x) | 78.5 / 515.6 |

**TTFT is already at 1.07-1.16x of vLLM on an unoptimised packet** — with no tuned
Lt table, no GQA-8 hd512 role, and fatlite surrendered. That is the encouraging half.

**Decode is the whole gap**, and the shape of it names the cause:

* TPOT is essentially flat from C1 to C4 (48.81 -> 50.66, +3.8%) while throughput
  scales 3.8x. A per-step cost that does not grow with the batch is compute-bound,
  not bandwidth-bound.
* 48.8 ms/token against 7.64 GB of weight traffic is ~156 GB/s effective on a
  3352 GB/s card — 4.6 % of bandwidth. A memory-bound GEMV would be near 1-2 TB/s.
* `runtime/nvidia/op_moe.cuh` confirms it: the Gemma MoE expert GLU and DOWN decode
  bodies have ONLY the dot8 CUDA-core walk (`GV_MOE_RB` row-blocking, one warp per
  output channel). There is no tensor-core path. `op_gemv_mma.cuh` exists and is
  wired into the three DENSE walks (`gemv_rows`, `gemv_glu_rows`, `gemv_qkv_rows`)
  behind `PLOW_NV_GEMV_MMA`, but not into the MoE expert walks.

This is the same bug shape the 12B campaign already paid for once: the dense dot8
walk was compute-bound at B>=4 until `PLOW_NV_GEMV_MMA` replaced it with the
`mma.sync m16n8k16` row-block walk (C16 TPOT 44.6 -> 16.2 ms). The MoE expert
GLU/DOWN need the same treatment, and that is the single largest item on the board.

Caveat worth stating: at B=1 a tensor-core walk is not automatically the answer —
the dense hook only engages at `MM >= 4`. At C1 the expert GEMV is one row per
expert, so the win has to come from either the walk's load efficiency or from the
~150 extra serialized MoE stages per decode step (5 MoE ops x 30 layers) on top of
the dense ones. ATTRIBUTE FIRST (the 12B lesson: three knob A/Bs came back null
before one attribution run found the real cost) — use the decode step timing
(`PLOW_DSTEP_LOG` / `step_time`), not a guess.

## First light notes

Coherence gate PASS. First light is SLOW: in128/C1 warmup ran 6.24 s per 128-token
request, i.e. **~48.8 ms/token against vLLM's 5.03 ms** — roughly 10x behind, and
~21x above the 2.29 ms roofline floor. That is an unoptimised packet and the reasons
are known and enumerable:

* `PLOW_BUILD_FATLITE=0` is now forced, which surrenders the occupancy win fatlite
  exists for (12B measured -2.8 ms @1024, -4.8 ms @4096). Recovering it for 26B means
  widening the MoE prefill guard so the ops survive fatlite, not flipping the knob.
* No tuned cuBLASLt algorithm table for the 26B shapes (the 12B assets ship one;
  26B picks algorithms at load). `campaign.py probe` fills this.
* No hd512 px4 role — that object hardcodes `n_kv_head=1` and 26B has 2, so 26B runs
  the generic hd512 body.
* The decode MoE GLU is the dot8 CUDA-core walk, the exact shape of bug that made 12B
  decode compute-bound before `PLOW_NV_GEMV_MMA`.

## Superseded: the prefill fault investigation

A packet emits cleanly and loads (47.0 GiB weights + 8.75 GiB KV, 54.7 GiB total,
24.8 GiB free), then the FIRST prefill chunk faults:

```
prefill chunk (seg graph): stream sync failed  slot=0 bucket=0 bucket_t=128
  chunk_start=0 prompt_len=39
CUDA_ERROR_ILLEGAL_ADDRESS (700)   [CUDA_ERROR_LAUNCH_FAILED (719) with roles on]
```

**CORRECTION (do not repeat the mistake below).** The first version of this section
argued "a vanilla packet faults too, therefore the bug is pre-existing and not the
campaign's fault". That reasoning was WRONG and cost hours. The controls that settle
it, run afterwards:

| packet | 12B | 26B |
|---|---|---|
| bare `plowc --emit devblob+cubin` | faults 700 | faults 700 |
| bare + `--segmented` | faults 700 | — |
| full recipe (role objects + production knobs + matching serve env) | **WORKS** | faults 719 |

A bare emit is not a serving configuration for ANY Gemma-4 model — it faults on 12B
too, and 12B is the model this repo already beats vLLM with. So the vanilla-packet
evidence proved nothing, and time spent hunting a 26B/MoE-specific cause for it was
wasted. There is NO regression on main: 12B through the full recipe serves correctly
("The capital of France is Paris.").

The only meaningful comparison is recipe-vs-recipe, and on that basis the 26B fault
IS specific to the 26B configuration.

### Recipe-level bisect against the working 12B configuration

With 12B-full-recipe as a known-good control, the 26B recipe was bisected toward it.
Each row is a full rebuild (~2 h each for 26B) plus a leased smoke test:

| 26B variant | result |
|---|---|
| recipe as first written (`PLOW_BUILD_PFATTN_WG=0`) | faults 719 |
| `PLOW_BUILD_PFATTN_WG=1` (matches 12B; the role ABI is `..._wg32_v1`, so this was a real recipe bug worth fixing regardless) | faults 719 |
| `PLOW_EMIT_PREFILL_CUBLASLT=0` (removes the campaign's own shape-list change and all 1025 Lt segments) | faults 719 |

So neither the hd512 role geometry nor the cuBLASLt admission is the cause, and the
campaign's own changes are cleared: the fault survives with the shape list unused.

### Where that leaves it

12B works through the same pipeline, the same binaries, the same serve env. The MoE
opcodes are the only 26B-unique ops, and they are clean in isolation across 16
input shapes. The fault therefore lives in something the isolated harness cannot
see — the most likely candidates, in order:

1. **MoE ops inside the cooperative megakernel**, not standalone: the shared smem
   arena (`op_moe_sm90.cuh` raises `PGM_ARENA_BF16` to 99328 B and other ops in the
   same launch share that allocation), or counter ordering between the 1-block align
   and the 132-block consumers. The repro sidesteps both by launching separately.
2. **`act.moe.*` scratch aliasing** with other activations in the packet's tensor
   arena — sizes were checked and are individually correct, but overlap was not.
3. The hybrid structure itself: every 26B layer runs the dense MLP AND the MoE,
   which no other model in the tree does.

Hypothesis 2 was checked and is CLEAN — `plowrt disasm --tensors` gives every MoE
scratch extent exactly as the emitter intends: `act.moe.part` 369098752 B
(4096·8·2816·4), `act.moe.fug` 69206016 (49152·704·2), `act.moe.table` 262144
(4096·8·8), `act.moe.meta` 1544 ((3·128+2)·4), `moe.ewt.<l>` 2048 (128·2·8).
No sizing or aliasing bug in the packet.

That leaves hypothesis 1 — the MoE ops *inside* the cooperative megakernel — as the
live one, and it is the one the isolated repro structurally cannot test. The way to
settle it is to extend the repro to launch the chain COOPERATIVELY with the packet's
own counter semantics and a shared arena, rather than as separate kernels with
`cudaDeviceSynchronize` between them.

### Ruled OUT (each checked, do not re-suspect)

| hypothesis | how it was ruled out |
|---|---|
| Segment roles / hd512 / GQA2 | fault persists with every role off |
| cuBLASLt admission | the vanilla emit does not use it and still faults |
| Opcode wiring | all five grouped ops dispatch in `interp_sm120.cu` |
| `I_moe=704` N-tile remainder | `moe90_stage_b` guards `nn < n`; repro passes |
| smem arena undersize | `PGM_ARENA_BF16` raise propagates (op_moe.cuh included at interp_sm120.cu:191, after op_gemm.cuh:177, before PLOW_NV_PRE_B at :729); host gave 103424 B vs 99328 B needed |
| `PLOW_MOE_MAXE` too small | 256 >= 128 experts |
| Compile-time GQA mismatch | prefill/decode attention take `n_head`/`n_kv_head` at runtime; 26B's 2 global KV heads are not static |
| Wrong packet geometry | `plowrt disasm` confirms sliding nhead=8 hd=256 n_kv_head=8 kv_stride=2048, full nhead=2 hd=512 window=0, V from `act.kg` (k_eq_v). All correct |
| Memory pressure | 24.8 GiB free after load |
| Grouped MoE bodies themselves | `runtime/tests/moe_group_gemma26b_repro.cu` runs align+GLU+DOWN clean on sm_90a at exact geometry |
| Wrong MoE arm compiled | `interp_sm90a.cu:16` defines `PLOW_NV_HOPPER`, so the wgmma fork IS used — same arm the repro forced |
| Missing dependency edge | `devgen lib.rs:6105+` declares router -> align -> glu; `--counters` reports 840 edges, 0 redundant, 0 removable polls |

Also ruled out after the first write-up:

* **Counter pattern.** The 1-block-producer -> N-block-consumer shape is not novel:
  gpt-oss emits `MoeAlignPf` with `vec![0]` feeding 132-block consumers through the
  same `Dep::Coarse` edge, and it works. So the shape itself is proven; only a
  Gemma-specific mismatch could still be at fault.
* **MoE scratch sizing.** `moe_part` is `moe_rows * top_k * hidden * f32` and
  `moe_fug` is `total_pad * moe_inter * bf16`, with
  `total_pad = moe_rows*top_k + n_exp*128` — the exact bound the repro used and
  passed. (`moe_mfu` is sized by `dbatch`, but that is decode-only scratch and the
  prefill path does not touch it.)
* **Decode-only bisect is not available.** `PLOW_MOE_PREFILL=0` produces a
  decode-only blob and the emit then panics at `lib.rs:9482 "prefill buckets"`, so
  there is no way to serve this model without the grouped prefill path.

### The live lead

The repro passes because it `cudaDeviceSynchronize()`s between align and GLU. In the
real megakernel `MoeAlignGemmaPf` runs on **one** block (`b=1`) and
`MoeGroupGluGemmaPf` on **132**, ordered only by a counter. If GLU reads `meta`
before align has published it, `total_tiles` is garbage, `ntiles` explodes, and
`rowbase = rowoff[e] + (mtile - tilep[e])*128` indexes `fu` far out of bounds —
which is exactly an illegal address. The expert index itself is clamped by
`pgm_moe_expert_of_mtile`, so a bad `e` is NOT the mechanism; `rowbase` is.

`disasm --stream` confirms the SHAPE is right — inst 19 has 1 slice (wait_len 1,
succ_len 1) and inst 20 has 132 slices (wait_len 2, succ_len 1) — but the dump does
not expose the wait THRESHOLDS, which is the one number that would settle it. Read
those from the packet bytes rather than the disasm.

**The whole MoE prefill chain is now exonerated, not just the GEMMs.** The repro was
extended to cover ops 73 (router) and 77 (combine+norm) as well as 74/75/76, and to
sweep routing distributions — including running the REAL router over NaN-filled
padding rows, which is what a 39-token prompt in a 128-row bucket actually presents.
16 of 16 cases pass:

| T | even | all-to-one (127 empty) | concentrated (65 empty) | router over NaN padding |
|---:|---|---|---|---|
| 128 | ok | ok | ok | ok (11 empty, max_cnt 93) |
| 512 | ok | ok | ok | ok |
| 1024 | ok | ok | ok | ok |
| 4096 | ok | ok | ok | ok (363 tiles, 46464 of 49152 rows) |

So the fault is NOT in ops 73-77 at any bucket width or routing shape. It is either
in a non-MoE op of the prefill program, or in an interaction the isolated harness
cannot see — the cooperative launch, the shared arena, or counter ordering. Attention
and the dense MLP are the untested remainder, and the 26B-specific thing about them
is that every layer runs the dense MLP AND the MoE (12B runs only the dense MLP).

Tooling note: compute-sanitizer cannot attach to plowrt (consistent with
[[ncu-on-nix-plowrt]] — profilers do not work against the cooperative megakernel),
so attribution has to come from the packet and from isolated harnesses.

## vLLM 0.28 reference, Gemma-4-26B-A4B BF16, this H100

Same client and flags as the plow side, prefix caching off, greedy, 32 prompts,
128 output tokens. `perf-data/campaign/gemma4-26b-a4b.h100.reference-vllm028-bf16.csv`.

| cell | TTFT ms | TPOT ms | out tok/s | TTFT vs roofline | TPOT vs roofline |
|---|---:|---:|---:|---:|---:|
| 128 / C1 | 39.52 | 5.030 | 188.5 | 17.3x | 2.20x |
| 1024 / C1 | 44.11 | 5.080 | 185.7 | 5.2x | 2.16x |
| 4096 / C1 | 93.71 | 5.090 | 172.8 | 2.7x | 2.15x |
| 8192 / C1 | 181.10 | 5.090 | 154.7 | 2.5x | 2.13x |
| 128 / C4 | 73.31 | 7.240 | 515.6 | — | 1.60x |
| 1024 / C4 | 93.46 | 7.570 | 485.1 | — | 1.58x |
| 4096 / C4 | 227.59 | 7.920 | 414.8 | — | 1.62x |
| 8192 / C4 | 458.74 | 8.530 | 331.7 | — | 1.74x |

Three things this table says about where a win is available:

1. **TPOT is flat at ~5.08 ms across every context length at C1.** The 1024 sliding
   window caps decode KV growth, exactly as the corrected roofline predicts (floor
   moves only 2.29 -> 2.39 ms from 128 to 8192). Decode sits ~2.2x above its floor
   at C1 — this is the largest single pool of headroom.
2. **Long-context prefill is already efficient for vLLM** — 2.5x off roofline at
   8192. TTFT wins there have to come from real kernel work, and the dims audit
   already names the candidate (an hd512 px4 GQA-8 role, which 26B currently cannot
   use at all).
3. **Short-context prefill is soft**: 17.3x off floor at in128, i.e. launch and
   host overhead, not math. The 12B campaign found the same shape and the lever
   there was launch COUNT, not backend.

vLLM's 26B TPOT (5.03 ms) is about half its 12B TPOT (10.55 ms), which is what the
active-parameter difference predicts (3.82B vs 11.91B) — independent evidence that
the corrected roofline model is right.

## Levers, in priority order (evidence, not guesses)

1. **~~MoE decode tensor-core walk~~ — RETRACTED, MEASURED WRONG.** This was the
   stated priority-1 item, inferred from end-to-end effective bandwidth (48.8 ms for
   7.64 GB = ~156 GB/s, 4.6 % of roof) plus the fact that `op_moe.cuh`'s expert walks
   have no tensor-core path. `runtime/tests/moe_decode_gemma26b_bench.cu` measured the
   ops directly and the inference was wrong:

   | B | score ms | topk ms | glu ms | down ms | MoE/token | GB/s glu |
   |---:|---:|---:|---:|---:|---:|---:|
   | 1 | 0.0461 | 0.0082 | 0.0336 | 0.0154 | **3.10 ms** | **1890** |
   | 4 | 0.0558 | 0.0241 | 0.1162 | 0.1952 | 2.93 ms | 2183 |
   | 16 | 0.0969 | 0.0865 | 0.4818 | 0.7444 | 2.64 ms | 2107 |

   The expert GLU runs at 1890-2239 GB/s — 56-67 % of the card's 3352. It is not
   bandwidth-starved, a tensor-core rewrite buys little, and the ENTIRE MoE share of a
   decode token is 3.10 ms of 48.8 (6 %). Adding dense traffic (~4.8 GB at ~2 TB/s
   ~ 2.4 ms) the kernels total ~5.5 ms — and vLLM's measured TPOT is 5.03 ms. **plow's
   decode kernels are already competitive; ~43 ms (88 %) is overhead on top of them.**

   Caveat on the bench, stated so nobody over-trusts it: it allocates ONE layer's fused
   expert tensors (1.5 GB working set); the served model has 30 layers (~45 GB), so
   TLB/page behaviour at full scale could be worse than measured. The conclusion
   "kernels are not the bottleneck" is strongly indicated, not proven at scale.

1b. **THE OPEN QUESTION: where the other ~43 ms goes.** Still not attributed, and the
   attempt taught something worth recording so the next session starts ahead.

   `PLOW_STEP_TIME=1` reports gap/submit/sync/upload/kernel/download but its
   `StepTiming::log_every(128)` (gpu.rs:6228) sits inside `step_slots_sampled`.
   Instrumentation was added to the GPU engine's `token_batch_step`
   (`exec/gpu/token_batch.rs`, same `PLOW_STEP_TIME` knob, splitting host-side
   `enqueue` from `terminal` which carries the device wait). It is correct and
   committed — but it did not answer the question, because:

   * `token_batch_step` fires **once per request**, at prefill (`rows=16` for a
     39-token prompt), not per decode step: a 64-token generation produced exactly
     ONE `unified token batch committed`.
   * `step_slots_sampled`'s own timer also produced nothing across a 300-token
     generation with `PLOW_MULTISTEP=0`, so C1 decode is not going through that
     function either.

   So the C1 decode step traverses a THIRD path. The added instrumentation is still
   worth keeping — it reports as soon as the GPU token-batch route drives decode, which
   is what happens at higher concurrency, exactly where the serving cells live.

   **ROOT REASON THE 43 ms IS UNATTRIBUTABLE — the tool does not exist for this path.**
   `crates/plowrt/src/obs/dstep.rs` is exactly the right instrument: its own header is
   "§DSTEP — where a DECODE token's wall clock goes, host phase by host phase", it
   splits each phase into `pre ` / `GPU ` / `post` by whether pipelining could hide it,
   and it dumps a per-phase table every 64 tokens ending in
   "HOST TOTAL (everything but the drain)" — the exact number this campaign needs.

   It does not report on the CUDA serve decode path. Measured, not assumed:
   `PLOW_DSTEP_LOG=1 PLOW_DSTEP_EVERY=4` over a 40-token generation produced **zero**
   dumps (`dstep_log: true` confirmed in the serve config line), and a 300-token run
   produced none either.

   Be precise about WHY, because the first write-up of this said "dstep is AMD-only"
   and that is not accurate: `serve/mux.rs` brackets its decode tick with
   `dstep::begin_token()` (3281) and `dstep::finish_token()` (3386), unconditionally
   and engine-agnostically, and already times `STREAM` around `handle_produced_token`.
   That bracket is fine. The finding is that **the CUDA serve decode does not traverse
   that mux tick** — some other decode loop drives it. The phase set is separately
   AMD-shaped (`seed_ids x ranks`, `zero_xctr` cross-GPU, `AQL launch`, `TP safety
   audit`, `agree` cross-rank), so a CUDA port also needs phases that mean something
   on one GPU: submit, drain, sample D2H, detok/stream, idle.

   **So the highest-value next task is: find the decode loop the CUDA serve actually
   uses and bracket it with `dstep`** — the mux tick at 3281/3386 is the model to copy,
   not the site to fix. Until that instrument reports, every statement about where 26B
   decode time goes is a guess, and this campaign has already had four guesses refuted
   by measurement. Do not author a kernel before it reports.

   What is already excluded as the explanation, each by measurement against the 12B
   control or a standalone bench:
   * not weight bytes — 26B streams 7.64 GB/token vs 12B's 23.8 and is 3.8x slower per stage;
   * not stage count — 519 decode stages vs 12B's 540;
   * not low-block starvation — 12B has the same distribution (55 % vs 59 % of stages on <=16 of 132 blocks);
   * not the MoE kernels — measured above at 3.10 ms of 48.8.
2. **cuBLASLt algorithm table — DONE, and it is a NULL. Do not re-run it.** The store
   had 48 entries for 12B's 3840-keyed shapes and 0 for 26B's 2816-keyed ones;
   `campaign.py probe` wrote 40 measured entries and the server confirms it loaded
   (`cuBLASLt algorithm table loaded ... shapes=40`, then `stored algorithm pinned`
   per shape). Paired A/B on identical assets and cells:

   | cell | baseline TTFT | with Lt table |
   |---|---:|---:|
   | 128 / C1 | 42.21 | 42.24 |
   | 128 / C4 | 84.75 | 84.83 |
   | 1024 / C1 | 125.02 | 125.02 |

   Flat at every rung, including the one where prefill dominates. The GEMM backend is
   NOT the prefill lever for 26B — the same result the 12B campaign recorded
   ("cuBLASLt at every rung — NULL ... the lever for 128 tokens is launch COUNT
   (fused QKV / gate|up) or a cheaper launch, not the backend"). Keep the table (it
   pins shapes and is not slower), but spend effort elsewhere.

2b. **Launch count / per-launch floor — the real prefill lever.** The 1024 prefill
   program is **691 launches**: 206 Gemm, 121 RmsNorm, 90 HeadNormRope, 60
   NormResidual, **150 MoE** (5 ops x 30 layers), 30 Glu, 30 FlashPrefill, 1 SoftCap.
   TTFT 125 ms over 691 launches is ~181 us average against the 12B campaign's measured
   ~52 us projection-GEMM floor, so both the count AND the per-launch cost are in play.
   MoE adds 22 % more launches than a dense model of the same depth.

   **Do NOT reach for `PLOW_GEMMA_MOE_TAIL_FUSE` (op72 `MoeCombineResidNormGemma`).**
   It looks like the obvious launch-count win — it is the one fusion the packet
   reports as absent (`PLOW_PACKET_HAS_MOE_COMBINE_RESID_NORM_GEMMA 0`) — but it is
   default-OFF because it was MEASURED NEGATIVE (devgen `lib.rs:6033`, P9 2026-07-20):
   +0.18 ms/token on both bf16 and fp8 at 40 ctx, because the 1-block 4-pass SCALAR
   body costs more than the packet boundary it removes, and its reduction order
   differs from the vectorized `NormResidualNorm` (last-ulp bf16 flips). It is also
   B=1 only and the emitter asserts loudly if combined with t > 1. The source names
   the only version worth building: a register-cached VECTORIZED body that replicates
   NRN's summation order.

   So the launch-count lever has to come from somewhere else — the router
   score/topk pair, or fusing across the dense-MLP/MoE seam that only this hybrid
   architecture has. Attribute per-op first (`PLOW_PF_SEG_TIME=1`, attribution only,
   never a number) rather than guessing which of the 691 launches is the cost.
3. **fatlite, recovered properly.** `PLOW_BUILD_FATLITE=0` is currently forced to make
   MoE prefill exist at all, surrendering the occupancy win (12B: -2.8 ms @1024,
   -4.8 ms @4096). The fix is to widen the MoE prefill guard so the ops survive
   fatlite, not to flip the knob back.
4. **hd512 GQA-8 role.** 26B runs the generic hd512 body because the px4 BQ64 object
   hardcodes `n_kv_head=1`. A GQA-8 variant is the long-context prefill lever.

## Per-rung kernel pass (2026-09-20): the prefill router was the top kernel

Standalone harness `runtime/tests/moe_group_gemma26b_repro.cu` now times all five MoE prefill
ops (73-77). Per layer, ms:

| T | router 73 | align | glu | down | combine |
|---|---|---|---|---|---|
| 128 | 0.112 | 0.013 | 0.517 | 0.372 | 0.009 |
| 1024 | 0.881 | 0.042 | 0.519 | 0.414 | 0.110 |
| 4096 | 3.410 | 0.146 | 1.016 | 0.867 | 0.419 |

* Op 73 ran the [T,E,H] score GEMM as scalar fmaf dots: 0.09% of peak, linear in T,
  larger than both grouped GEMMs from T=1024 up. It was missing from the first timing pass.
* FIX `moe90_router_gemma_pf` (op_moe_sm90.cuh): one m128n128 wgmma tile per 128 tokens,
  h2 computed by the threads straight into the swizzled A stage as bf16 (what vLLM feeds
  its GateLinear), f32 accumulators = the fp32 logits vLLM asks for, logits parked in the
  dead stage ring, thread-per-token softmax/top-k. Flat 0.31 ms at every T (10.5x at 4096).
  Scalar body kept for T < MOE90_RT_MIN_T (512); crossover ~T=400.
* Selection vs scalar on synthetic near-uniform logits (worst case): 98.5-99.2% identical
  top-k sets, gates within 2e-4. Served: 1456-token needle prompt answered identically.

Served TTFT ms, C1 / C4 (before -> after | vLLM):
  128   42.15 -> 42.32 | 39.52      84.97 -> 63.26 | 73.31  (first metric ahead of vLLM)
  1024  124.96 -> 71.85 | 44.11     392.60 -> 190.06 | 93.46
  4096  408.93 -> 181.91 | 93.71    1046.59 -> 463.20 | 227.59
  8192  828.78 -> 373.66 | 181.10   3869.51 -> 2861.86 | 458.74
Served gain (-227 ms at 4096/C1) is 2.5x the standalone estimate (-92 ms). TPOT unchanged.

NULLS, do not retry:
* warp-per-token router (bit-exact, no barriers): 1.25x at 4096, slower at <=512. The scalar
  body was work-bound, not barrier-bound — all 8 warps were already busy.
* warp-per-token combine (bit-exact): slower at every T (0.419 -> 0.500 at 4096). Reverted.

LOW-RUNG ROOF CORRECTION: at T=128 all 128 experts are hit, so glu/down must stream every
expert's weights (1.02 GB + 0.51 GB per layer). The roof is BANDWIDTH (0.30 + 0.15 ms), not
FLOPs; glu is at 59% and down at 41% of it. Tile padding is not the waste — the lever is
faster B staging (TMA load engine, tma_ws_moe_group.cu), most of all for down (22 n-tiles).

## DECODE ROOT CAUSE (2026-09-20): every step ran the B=16 program

The ~43 ms of "unattributed decode overhead" was not overhead. Server log, every run:
`WARN decode ladder retains widest execution: narrower addressing is not qualified`.
`validate_decode_ladder` (plowrt exec/gpu/decode_rung.rs) normalises each op's row field
against an allowlist; any other op hits `_ => compatible = false` and the engine keeps ONLY
the widest rung. None of the Gemma MoE decode ops were listed, so C1 decode executed the
16-row program: 16 rows x top-8 -> up to ~80 experts streamed per layer instead of 8.
That is also why TPOT was flat from C1 to C16, and why the c32 packet's TPOT looked
"saturated". Same class of gap as the live-KV allowlist that dropped packed prefill.

FIX: arms for MoeRouterGemmaScore/ScoreFast (i2), MoeRouterGemmaTopk (i3),
MoeExpertGluNormGemma / MoeExpertDownGemma (i5), MoeCombineNormGemma (i2) — devgen's `nb`
immediate, 0 at B=1 else B. Greedy output byte-identical before/after; 19/19 decode_rung tests.

TPOT ms (before -> after | vLLM), tok/s after | vLLM:
  128/C1   48.76 ->  8.15 | 5.03    118.7 | 188.5      128/C4   50.65 -> 19.70 | 7.24   197.4 | 515.6
  1024/C1  49.69 ->  8.29 | 5.08    113.8 | 185.7      1024/C4  52.00 -> 21.15 | 7.57   178.2 | 485.1
  4096/C1  50.69 ->  8.35 | 5.09    103.1 | 172.8      4096/C4  53.51 -> 21.81 | 7.92   158.6 | 414.8
  8192/C1  51.97 ->  8.43 | 5.09     88.7 | 154.7      8192/C4  56.37 -> 24.21 | 8.53   106.6 | 331.7
C1 gap 9.7x -> 1.6x. Next decode target is B=4: 19.7 ms is 2.4x the B=1 step while vLLM's
B=4 costs 1.44x its B=1 — the MoE decode kernels' batch scaling, not dispatch.

## Decode batch scaling + post-router prefill attribution (2026-09-20)

DECODE-ONLY LADDER (in128/out512, PLOW_DECODE_MAX_RUNG=16, so TPOT = the rung's step):
  B      1      2      4      8      16
  ms     8.20   12.26  18.00  28.08  50.69     ~ 5.4 + 2.8*B
  tok/s  121    162    221    284    315
Batching buys almost nothing: every added row streams ~8 more experts and the MoE decode ops
run at ~2000 GB/s (60% of peak). vLLM's B=16 step is ~10 ms. Standalone MoE at B=16 is
~26 ms/step (glu 0.48 + down 0.26 + score + topk per layer), so ~25 ms of the 50.7 is NOT in
the four MoE ops — unattributed, the CUDA decode path still has no per-op timing.

Fixes landed from the standalone decode bench:
* expert DOWN lane-split arm was gated `nrow == 1` -> B>=2 fell to the unblocked body
  (651 vs 2058 GB/s). Generalised: 2.5-2.8x at B>=2. Served C4 TPOT 19.7 -> 18.2,
  1024/C16 218.9 -> 253.4 tok/s.
* exact scorer 0.0460 ms/layer vs ScoreFast 0.0085 at B=1 = 1.1 ms of the 8.15 ms C1 step.
  Recipes switched to ScoreFast (f32 end to end; vLLM's router input is bf16).
* scalar prefill router (T<512) now uses the decode path's bit-exact warp-parallel top-k:
  0.112 -> 0.087 ms/layer at T=128, table byte-identical.

PREFILL ATTRIBUTION, b26j packet, T=4096 chunk (157.6 ms attributed vs 181 ms real TTFT —
SEG_TIME inflation is now small): MoE segment 75.2% (3.95 ms/layer), dense GEMMs 11.2%,
FlashPrefill 7.6%, HeadNormRope 3.4%. Standalone MoE ops sum to 2.77 ms/layer
(router 0.32, align 0.15, glu 1.02, down 0.87, combine 0.42). The 206 dense GEMMs do the
same FLOPs as the routed experts in 0.59 ms/layer vs ~1.9 for glu+down: the grouped GEMM's
cp.async staging (51% of its k-step) is the remaining prefill target -> TMA load engine.

## Fast scorer served + the grouped GEMM's real waste is partial tiles (2026-09-20)

SERVED b26k (ScoreFast + warp top-k in the scalar prefill router), before -> after | vLLM:
  C1 TPOT  8.15 -> 6.50 | 5.03   (1024: 6.68, 4096: 6.76, 8192: 6.83)   C1 tok/s 118.7 -> 147.5 | 188.5
  C4 TPOT 18.18 -> 16.87 | 7.24   1024/C16 253.4 -> 266.5 tok/s | 1379.7
  TTFT unchanged (the 0.75 ms top-k saving at T=128 is inside noise).

A cuBLAS ROUTE IS A NULL BY PROXY. torch.bmm over the same fixed-count 128-row expert tiles
(the only sync-free shape): gate_up+down 0.646 / 0.924 / 1.815 ms at T=128/1024/4096 vs plow
wgmma 0.889 / 0.933 / 1.883. The kernel is not slow for the tiles it is given.

THE WASTE IS THE TILE COUNT. Real routing (harness dist 3) ends every expert's segment in a
partial tile: 364 tiles where 256 would do at T=4096, 175 for 64 at T=1024. The earlier
even-routing (dist 0) timings hid it — dist 0 has zero partial tiles at 4096 — and it is why
the served MoE segment (3.95 ms/layer) exceeded the standalone sum (2.77).

FIX: MOE90_HALF — align pads to 64 rows; a block's two warpgroups each take one half-tile,
of different experts when they differ (then B is staged once per warpgroup; arena 99 -> 165
KB, smem_optin is 232 KB so still 1 block/SM). `part` is BYTE-IDENTICAL on 12/12 cases
(harness now has expert-distinct random weights; part is (token,slot)-indexed = layout-free).
  real routing, ms/layer    glu             down
  T=128                     0.521 -> 0.381  0.362 -> 0.284
  T=1024                    0.679 -> 0.627  0.508 -> 0.421
  T=4096                    1.395 -> 1.333  1.023 -> 0.921
  even routing T=4096       1.017 -> 1.049  0.868 -> 1.045   <- and SERVED it regressed:
                            TTFT 182 -> 197 at 4096 (b26m). Not the B-ring layout, not the arena
                            (128-row kernel with the 165 KB claim times identically). CAUSE: the
                            per-tile expert lookup, a linear scan of ~64 GLOBAL loads per thread
                            (~3 us); half mode did it twice. A DOWN tile is only 11 k-steps
                            (~20 us), so that is +15-20% there, +3% on GLU. Bisected (7 loads) and
                            the second expert stepped forward from the first:
  128-row path, bisect only 1.021 -> 0.910  0.869 -> 0.739   (-11% / -15%, free)
  half + bisect, T=4096 even         0.960           0.713
  half + bisect, real T=128   0.518 -> 0.362  0.363 -> 0.226
                       T=1024 0.682 -> 0.605  0.511 -> 0.377
                       T=4096 1.394 -> 1.304  1.026 -> 0.883   all 12 cases BYTE-IDENTICAL
SERVED b26n vs b26k, TTFT ms: 1024/C1 71.96 -> 67.12, 4096/C1 182.4 -> 176.1, 8192/C1
374.7 -> 363.1, 128/C4 61.4 -> 57.6, 1024/C4 181 -> 172, 4096/C4 452 -> 438.
Off when PLOW_NV_W8A8 is compiled in: the e4m3 twins still walk 128-row tiles.

## The bench "128" cell is a 512-bucket measurement; rung crossings cost 17-29 ms (2026-09-20)

Request wall (max_tokens=1) vs server-side prompt_tokens, b26m:
  8..128 tok  24.2-25.2 ms  | 129..510   41.8-44.2 | 1020..1023  62.2 | 1025..1100   91.1-91.5
  4090..4095  177-178       | 4097..4200 202-203   | 8190        368  | 8193         394
vLLM's 128 cell is 39.5 ms; plow at <=128 rows is 24 ms. The client sends exactly N tokens but
the server re-tokenises and adds BOS, so the 128 cell is 129 rows -> the 512 bucket, where 383
of 512 rows are padding that the MoE router still routes (~75% of that bucket's MoE work).
This is why 128/C1 TTFT sat at 42.2-42.6 through EVERY kernel change this session.
Levers: PLOW_PF_LADDER_APPEND rungs that swallow the +1 (256 / 1152 / 4224), and the ragged-M
row shrink (kvrow.rs PREFILL_ROW_FIELDS — already lists the Gemma MoE pf ops, but only the
AMD and Apple engines apply it; the CUDA engine runs the padded bucket).

## Tune store: this board has NO kernel measurements (2026-09-20)

`plowc tune status --gpu h100`: cell nvidia/sm_90a/h100-sxm5 is empty (store holds AMD cells
only) -> every emit here used the analytical model. On NVIDIA the tile is object-wide, so the
measured flow is scripts/tune_decode_sweep.sh (object x packet knobs, scored by step_bench
TPOT, `--ablate-lo` twins for per-op decode cost, `--block L` for single-layer packets) ->
`tunedb-decode ingest|best`. Prior fp8 records under h100-nvl show occupancy is the big decode
lever: 26B B=1 5.52 ms at 132 blocks vs 4.77 at 264 (2 blocks/SM). Stage A on this board:
occ {1:132, 2:264} x batch {1,4,16}, bf16. Needs ripgrep on PATH (nix shell nixpkgs#ripgrep).

## Measured decode tune on THIS board (tune_decode_sweep.sh -> tunedb-decode), 2026-09-20

Setup that was missing here: ripgrep (nix shell nixpkgs#ripgrep), LD_LIBRARY_PATH=
/usr/local/cuda/lib64 for step_bench's libcublasLt, `--base-defines` takes the full -D form.
step_bench TPOT, bf16, ctx 1024, 5 reps, <=0.4% spread, vram_before 0 MiB:

  occ (FORCE_MINBLK:n_cu)   knobs            B=1     B=4      B=16
  1:132                     un8              5.922   14.069   34.193
  2:264 (MOE90_HALF=0)      un8              6.866   16.129   34.602
  2:264 (MOE90_HALF=0)      un4 glu2         6.720   -        36.306
OCC-2 LOSES on this board in bf16 (it won 14% in the fp8/h100-nvl records) -> 132 stays. Note
occ-2 cannot even load with MOE90_HALF: the pf object must share the decode grid and its arena
(165 KB) no longer fits 2/SM of the 232 KB opt-in; irrelevant while occ-1 wins.
Served vs pure: C1 6.47 vs 5.92 (0.55 ms serving loop); C16 50.5 vs 34.2 (16 ms is serving
interference — prefill interleave / admission — not kernels).
Rows are in tuning/nvidia/sm_90a/h100-sxm5 as PROVISIONAL ("correctness not checked");
qualify with gpu_lifecycle on the winning asset dir, then ingest --correctness pass.

TUNER BUG FIXED: build_cubin hashed the cache key BEFORE appending the ablation mask, so every
mask shared one twin — 13 ops all "cost" 0.044 ms (HeadNormRope's). Mask is now in the key.

PER-OP DECODE COST, ablation twins, B=1 (clean; sums to 87% of 5.917 ms):
  Gemv (o_proj + dense down, 60 ops) 1.346 22.7% | MoeExpertGluNorm 1.002 16.9%
  MoeExpertDown 0.657 11.1% | GemvQkv 0.585 9.9% | FlashDecode 0.476 8.0%
  NormResidualNorm 0.323 5.5% | MoeCombineNorm 0.263 4.4% | GemvGlu 0.190 3.2%
  FlashMerge 0.135 | MoeScoreFast 0.122 | HeadNormRope 0.039 | MoeTopk 0.028 | GemvArgmax ~0
  -> the plain dense GEMV arm is the top cost: ~22 us/op for 12-23 MB while the fused GemvGlu
     arm moves 24 MB in ~6 us (3.5x). MoE total is 2.07 ms (35%).
B=16 rows are CONFOUNDED except for ops whose output no router reads before their cost is
paid: skipping a body feeds garbage downstream and moves the expert union (negative "costs").
Clean: MoeExpertGluNorm = 19.7 ms of the 34.2 ms step (58%) — the batch-serving target;
FlashDecode 3.4 ms. CSV: perf-data/campaign/gemma4-26b-a4b.h100.bf16-decode-ablation.csv

## MoE ragged tail on the CUDA engine + decode knob result (2026-09-20)

KNOBS ARE FLAT (stage C, step_bench, occ 1:132; B=1 / B=16 ms): un4 5.906/34.72, un8 5.919/
34.19 (shipped), un12 5.948/33.94, un16 5.959/34.30, glu2 6.067/34.91, glu8 6.089/35.06, mun4
5.971/34.21. Everything within ~1%; every object sits at the 255-register cap. Knob tuning
cannot close C1 5.92 vs vLLM 5.03 — that takes kernel work.
CORRECTION to the ablation read: the sweep's packets are emitted WITHOUT the recipe's
PLOW_FUSE_ARGMAX=1, so the lm_head is a plain Gemv there (61 Gemv, no GemvArgmax). "Gemv 1.35
ms" includes the lm_head (~0.5-0.8); o_proj/dense-down are ~10 us each, not 22.

MoE RAGGED TAIL (user direction: fine-grained rungs / ragged tail). kvrow::rebase_chunk_rows
existed but only the AMD/Apple engines applied it. The CUDA engine now rewrites the three Gemma
MoE pf row operands (router i3, align i0, combine i2 — located via PREFILL_ROW_FIELDS, same
"field == bucket width" guard) to the launch's REAL rows, in both the chunk path and the packed
path, under the existing PLOW_RAGGED_CHUNK (=0 is the control). Dense ops keep the bucket
width: their cuBLASLt plans are shape-static. MoE rows are independent, so real rows compute
exactly what they did — greedy text identical 10/10 across the A/B.
  request wall ms, padded -> ragged: 173 rows 44.7 -> 36.8 | 1093 rows 88.7 -> 70.1 |
  893 rows 61.8 -> 58.1 | 8893 rows 405.8 -> 398.6 | 11 rows 25.6 -> 24.0
  served: 1024/C4 TTFT 172.4 -> 137.8, 1024/C16 268.8 -> 280.4 tok/s, 1024/C32 285.3 -> 302.8
  C1 cells barely move (128/C1 42.24): the bench prompt is RANDOM tokens, and 129 random rows
  already route across all 128 experts, so every expert's weights are streamed with or without
  the padding. Natural text touches fewer experts, hence the 8 ms the probe saw.

## MoE decode batch GLU + sg8 + fine-grained rungs (2026-09-20)

MoE DECODE GLU (user direction). The SERVED op is the fused norm+GLU (op 71); its B>1 path was
the scalar body (2 B loads, no unroll, xn recomputed per output channel) — the standalone bench
had been timing the NON-fused op and under-reported it. B rows of xn do not fit the decode
arena, so the op now stages them in the packet's moe.xn2 tensor (t5, B*H bf16) and runs the
existing vector body. t5 rides EVERY rung: validate_decode_ladder compares operands across
rungs and a B>1-only operand would put decode back on widest-only execution. The batch body
has its own unroll (GV_MOE_GLU_UN_B=4): the dense GLU's 10 from the build flags was slower.
  per layer ms, scalar -> staged: B=2 0.145 -> 0.073, B=4 0.292 -> 0.139, B=8 0.549 -> 0.269,
  B=16 1.091 -> 0.647. relL2 ~3e-3 (bf16 xn, as the dense arms and vLLM round it).
MoE DOWN: PLOW_MOE_DOWN_SG=8 beats 4 by 11-15% standalone; sg2 silently leaves the lane-split
arm at I_moe=704 (704 % 128 != 0).
END TO END (tune_decode_sweep.sh, step_bench, occ 1:132, packets re-emitted with t5), ms:
                        B=1     B=4      B=16
  before                5.922   14.069   34.193
  batch GLU, sg4        5.905   11.034   26.786
  batch GLU, sg8        5.833   10.648   26.387     -> sg8 into the recipe via PLOW_EXTRA_DEFINES

FINE-GRAINED RUNGS (user direction): PLOW_PF_LADDER_APPEND=256,1152,4224. Two emitter issues:
* MAX_CHUNK must be a power of two and 8192 doubles every slot's sliding KV. The real
  constraint is the ring: next_pow2(1024+4096-1) = 8192 rows >= 1024+4224-1. appended_rungs()
  admits a rung on `window + rung - 1 <= ring`; the ring assert now checks the WIDEST rung.
* chunk activations (main, TP partial slot, flash partials) were sized ctx.min(MAX_CHUNK), so a
  4224-row launch would have overrun 4096-row tensors. Caught by packed prefill's "attention
  tensor extent" check, which silently dropped packed prefill (the segments script then failed
  on the missing pfpackedseg cubin). chunk_rows() sizes all three from the widest rung.
Ladder: [128, 256, 512, 1024, 1152, 2048, 4096, 4224]. Natural-text request wall, before ->
after: 129 rows 41.8 -> 24.5 | 1025 91.1 -> 61.9 | 4097 202.4 -> 169.1 | 8193 394.1 -> 340.5.
Greedy answers identical solo vs 4-concurrent (4/4), same text as every earlier packet.

## The 128/C1 TTFT was a socket, not a kernel: TCP_NODELAY (2026-09-20) — FIRST WINS vs vLLM

CORRECTION: the bench's N-token cells do NOT cross their rungs. For vllm-bench random prompts
the server sees exactly N rows on /v1/completions (no BOS row); measured via PLOW_STEP_TIME
rows_total. The 42-vs-24 ms split seen earlier was prompt CONTENT (random tokens touch every
expert; natural text touches few). The fine-grained rungs still remove the 17-29 ms cliff for
arbitrary-length traffic; they just do not move these cells.

128/C1 TTFT sat at 42.2-42.6 ms through every kernel change of the day. Device prefill for those
128 rows is 28.7 ms; a fresh connection sees the first chunk at 32.5 ms; vllm bench reports 42.3.
Ruled out one variable at a time with the real client: --max-hold-ms 0 (42.21), PLOW_MULTISTEP=1
(42.55), and every bench payload field (repetition_penalty / logprobs / include_usage /
ignore_eos: all 32 ms). CAUSE: plowrt never set TCP_NODELAY. A streamed response is headers
then the first token as a second small write; on the bench's pooled keep-alive connection Nagle
holds that write until the client's ~40 ms delayed ACK, so any TTFT under 40 ms reads as ~42.
(At 1024 rows the first chunk lands after the ACK timer, hence no stall there.) Fix:
axum::serve(..).tcp_nodelay(true) on the OpenAI listener.

SERVED, b26q + nodelay (plow | vLLM 0.28):
  TTFT ms   128/C1 31.85 | 39.52 *WIN*   128/C4 56.74 | 73.31 *WIN*   1024/C1 67.39 | 44.11
            1024/C4 134.4 | 93.46        4096/C1 175.7 | 93.71        4096/C4 339.0 | 227.6
            8192/C1 371.6 | 181.1        8192/C4 1228 | 458.7
  TPOT ms   C1 6.58-6.85 | 5.03-5.09     C4 11.75 / 13.40 / 15.33 / 17.41 | 7.24 / 7.57 / 7.92 / 8.53
  tok/s     1024/C16 371 | 1380   1024/C32 405 | 1736   4096/C16 222 | 885   4096/C32 222 | 1081

## Grouped GEMM at 1024/4096: it is at the HBM roof; the fix is a vendor batched route (2026-09-20)

User direction: fix the prefill MoE grouped GEMM for the 1024 / 4096 rungs.

GLU k-step anatomy, real routing (harness switches MOE90_DBG_NOSTAGE_B / NOSTAGE / NOMMA), ms/layer:
             full    B-staging off   all staging off   tensor core off
  T=1024     0.601   0.347           0.245             0.508
  T=4096     1.304   0.844           0.595             1.008
B-tile staging is 35-42% of GLU. WHY A NEW LOAD ENGINE CANNOT HELP: bytes. Each expert's 7.9 MB
gate_up is re-streamed for EVERY row-tile pair of that expert and each 128-row A tile is re-read
for all 6 column tiles: ~2.5 GB of B + 1.3 GB of A per layer in 1.3 ms = ~3.0 TB/s against a
3.35 TB/s HBM peak. The kernel is BANDWIDTH-bound at the roof (minimum would be 1.0 + 0.22 GB).
L2 is 64 MB device-wide and one round's B working set is ~185 MB, so reordering for reuse does
not work; holding two row-tiles' accumulators hits the 255-register cap.

Measured, in order:
* TMA load engine for DOWN (d_moe_group_down_gemma_pf_tma, mbarrier ring, 2-D maps over the
  fused down tensor + gathered fu): BYTE-IDENTICAL first try, and a WASH — 0.843 vs 0.789 ms at
  T=4096, +-4% elsewhere. Both engines sit at ~1.4 us per k-step. Removed (nothing dispatched it).
  tma_ws_moe_group.cu's 1.6-1.75x was K=3840 / E=8, where the body was issue-bound, not roof-bound.
* Pipelined DOWN k-loop (prefetch one stage, wait<1> on the previous mma group; safe on the 3-deep
  ring): bit-exact, down 0.377 -> 0.340 (T=1024), 0.880 -> 0.783 (T=4096). ON by default. GLU cannot
  take it: a 3-deep ring with two B sets exceeds the 232 KB smem budget.
* PROXY for a vendor route, per-EXPERT batches (torch.bmm [E][C][*], C = fixed capacity), ms/layer
  gate_up+down vs plow wgmma:  T=1024 C=128 0.62 vs 0.945 (1.5x) | T=4096 C=256 0.70, C=384 0.90 vs
  2.087 (2.3-3x).  (The earlier 128-row-TILE bmm proxy was no faster — shape matters.)

PLAN — sync-free batched expert route (the dense projections already escape via cuBLASLt):
  1. align: capacity layout rows[e*C + i] for the first C rows of each expert; rows past C go to the
     existing tile meta as OVERFLOW, served by ops 75/76 unchanged (zero tiles when none).
  2. device ops: gather xn2 -> [E*C, H]; existing Glu over the two batched outputs; gate-scale +
     scatter into part. 3. a batched GEMM instruction (batch E, stride; gate and up are strided views
     of the fused gate_up) routed to cuBLASLt strided-batched plans in plowrt (segment = one op).
  4. every gate this campaign tripped: slots + dev.rs/dev_isa.h docs, opclass, rowclass, opaudit,
     live_kv direct_operands, kvrow PREFILL_ROW_FIELDS (ragged), segment_roles Lt shapes, packed
     prefill audit. Expected: -36 ms TTFT at 4096, -10 ms at 1024, -70 ms at 8192.

DECODE, from the fusion-knob sweep (stage E, step_bench B=1): PLOW_FUSE_ARGMAX=1 — which the
recipe set — is SLOWER, 6.501 vs 5.837 ms; recipe now 0. FUSE_MERGE / FUSE_HNR: no effect.
PLOW_GEMMA_MOE_ROUTER_FUSED: 142.7 ms/step (legacy one-block router) — never.

## Single-block roofline pass at 1k/4k/8k/16k + latency-bound MoE stages (2026-09-20)

User direction: check the kernels at the 1k/4k/8k/16k rungs on single modular blocks, compare to
roofline, improve the kernels.

TOOLING. `scripts/campaign/block_roofline.py <hf> --kind sliding|full --plow sweep.json --vllm
layer.json`: FLOP/byte roof per layer vs plow `block_run` and vLLM's own `Gemma4DecoderLayer`
(`scripts/block_layer_bench.py`, fixed for vLLM 0.28). Block recipes = the realtime recipe +
`--block L` (L0 sliding, L5 full; the full block needs PLOW_GEMMA4_SM90_HD256_GQA2_ROLE=0),
max_ctx 18432; emit <1 min, objects ~6 min. block_run needs `--pf-chunk 4096` past 4k.

BLOCK TABLE, prefill ms/layer B=1 (plow | % of roof | vLLM layer harness):
  sliding  1k 1.92 | 29.7% | 3.10   4k 5.01 | 19.0% | 4.58   8k 10.51 | 17.1% | 8.52   16k 20.56 | 17.5% | 16.37
  full     1k 2.00 | 30.5% | 3.17   4k 6.80 | 19.0% | 5.76   8k 17.17 | 17.6% | 12.54  16k 46.99 | 17.6% | 31.08
The vLLM layer harness UNDER-represents served vLLM (30 x 0.34 ms decode = 10 ms vs served 5.03), so
the target stays served TTFT/30 = 1.47 / 3.12 / 6.04 ms at 1k / 4k / 8k. Short rungs, sliding block:
0.96 / 1.16 / 1.48 / 1.91 ms at T=128/256/512/1024 -> a ~0.9 ms LATENCY FLOOR per layer, then
~1.0 us/token/layer marginal; vLLM's ladder is the opposite shape (~39 ms fixed, ~0.5 us/token/layer).
Served TTFT minus 30 x block = 2.9 / 9.5 / ~15 ms at 128 / 1k / 4k: non-layer time that GROWS with T
and is not the lm_head (last row only) — unattributed, next.

OP ATTRIBUTION (PLOW_PF_SEG_TIME on the block): the MoE segment (2 norms + ops 73..77 + NormResidual)
is 1.47 of 1.92 ms at 1k and 3.68 of 5.01 at 4k (it also absorbs the wait on the async Lt GEMMs
queued before it). Full layer: hd512 FlashPrefill 1.76 ms at 4k and 10.76 ms for the chunk that
attends 16k keys (70.8% of that chunk, ~18% of its roof) = ~25 ms/layer over a 16k prefill.
hd512 KERNEL CHOICE IS ALREADY RIGHT: the 26B runs the WGMMA BQ64/BKV32 role (1.76 ms at 4k); the
px4/BQ64 role that hard-codes n_kv_head=1 measured SLOWER on the 12B (2.45-2.65 vs 1.73-1.95), so a
2-KV-head px4 variant is not worth building. Long-KV hd512 still needs a redesigned kernel.

THE LENS THAT PAID: latency-bound vs work-bound, per stage (harness, T=1024 / T=4096, ms/layer):
  router   0.309 / 0.309  FLAT in T -> latency-bound. Split: top-k 0.125, k-sweep 0.18 of which the
           mma is ~0: the step time was the threads' scalar h2 staging (128 rows/block) while 124
           of 132 blocks idled at T=1024, and the top-k was a 1400-iteration serial scan per token.
           FIX (MOE90_RT_SPREAD, default on): a tile holds ceil(T/nblk) rows (8 at 1k, 32 at 4k),
           rows past that are never staged; with <=32 rows/block the bit-exact warp-parallel top-k
           takes R/8 rounds. Routing tables BYTE-IDENTICAL at T=512/1024/1152/4096/4224.
           0.309 -> 0.064 (1k), 0.309 -> 0.101 (4k). It also beats the scalar block-per-token router
           at the short rungs (0.086 / 0.170 / 0.253 at T=128/256/384 vs 0.063 flat), so
           MOE90_RT_MIN_T 512 -> 128.
  combine  0.112 / 0.421  13.6 us/token/block at both rungs. NOT barrier-bound (dropping all three
           barriers + the warp reduce: -4%); it is the read of `part`: 8 f32 slot rows per token,
           369 MB at 4k at ~2.4 TB/s. float4 reads (PLOW_MOE_COMBINE_PF_V4, as the decode twin):
           0.100 / 0.374. Structural fix left: bf16 partials (halves DOWN's write and this read,
           and is what vLLM sums) — a numerics-policy change, not done.
  glu/down all 132 blocks busy, ~3.5 GB/layer staged at ~3.1 TB/s: at the design's bandwidth roof
           (max intensity with m64x2 / n128-n256 tiles is ~85 FLOP/B = ~26% of peak; we are at 20-23%).

REAL ROUTING (harness `dist 4` = replay of act.moe.table dumped by `block_run check --in
<embeddings.npy> --dump-tensors act.moe.table`, layer 0, random token ids like the bench):
HEAVILY SKEWED — T=4096: ~30 hot experts with 1000-2300 rows, median expert 25 rows, 33 empty;
T=1024: max 556, median 7, 36 empty. `dist 3` is the NaN-padded worst case (8 hot experts), NOT real
routing — earlier "real routing" labels in this file mean dist 3. On dist 4:
  * n256 DOWN confirmed: 0.267 -> 0.244 (1k), 0.719 -> 0.652 (4k), bit-exact. Default on.
  * HALF tiles: GLU 0.501 vs 0.458 full-tile at 1k (mixed pairs stage 4 B tiles), equal at 4k; DOWN
    better with half. Net equal (0.876 vs 0.897 total at 1k) — left on.
  * THE FIXED-CAPACITY BATCHED VENDOR PLAN ABOVE IS DEAD: C=256 covers 37% of rows at 4k.
  * Vendor-class ceiling on this routing (torch._grouped_mm, device offsets, gate_up+act+down):
    0.608 ms (1k), 1.065 ms (4k) + gather 0.02 / 0.11, vs plow 0.745 / 1.781. So a grouped vendor
    route is worth ~-3 ms/chunk at 1k and ~-18 ms at 4k, and needs either a host sync per layer
    (cublasGemmGroupedBatchedEx takes host size arrays; plowrt binds neither libcublas nor the Lt
    batch attributes) or a CUTLASS dependency. NOT the 1k gap.

## HARNESS WINS DID NOT CARRY: the fat prefill object was the problem (2026-09-20)

Served packet with router spread + float4 combine + n256 DOWN REGRESSED (TTFT 128/C1 31.67 -> 33.79,
1024/C1 66.80 -> 68.71, 4096/C1 174.71 -> 192.26) although every change won in the harness.

HOW IT WAS FOUND (keep this method):
  * Blocks are a faithful in-situ proxy: `block_run check --in <real embeddings>.npy` with
    PLOW_PF_SEG_TIME=1 reproduced the served MoE segment EXACTLY (3.37 ms/layer, hot-expert 4k).
    Inputs: rtin/embed_rand_{1024,4096}.npy (random ids x sqrt(H)), embed_same_4096.npy (one row
    repeated = the " hello"*N prompt: 8 experts x 4096 rows).
  * New `PLOW_BUILD_SEG_EXTRA_DEFINES` (segments script) = A/B arms of a kernel default per block
    recipe; five blocks build in parallel in ~10 min. `PLOW_BUILD_SEG_EXTRA_DEFINES=-DPLOW_NV_TRACE=1`
    + PLOW_PF_TRACE_LOG=1 gives block 0's per-opcode gate/body cycles IN SITU.
  * PLOW_SEG_PER_OP is NOT usable for this: it also segments the decode programs and the CUDA
    runtime rejects them.
BISECT, MoE segment ms/layer (rand-1k / rand-4k / hot-4k):
  fat object, all old                1.660 / 3.698 / 3.356
  fat, all new                       1.731 / 4.286 / 3.801
  fat, router reverted               2.026 / 4.661 / 4.153   -> router spread is a real win (-0.3..-0.4)
  fat, n256 DOWN reverted            1.276 / 3.281 / 2.927   -> n256 is a big in-situ LOSS
  fat, float4 combine reverted       1.731 / 4.311 / 3.841   -> float4 is a small win
TRACE: with n256 compiled in, GLU's body on block 0 DOUBLED (1851 -> 3614 kcyc) while DOWN's own
body improved (1674 -> 1340). GLU runs before DOWN ever executes, so it is CODEGEN: the fat
`pfpackedseg` object sits at the 255-register cap with 1.9 KB of spill stack, and a body added to it
degrades its neighbours. (The C7519 "wgmma.mma_async serialized" warning is the ROUTER's alone —
present before this work, GLU-only / DOWN-only objects assemble clean; __noinline__ changes nothing.)
The harness kernels are lean (163-194 registers, no spills), which is why the harness lied: in situ
the MoE stages ran 0.6-0.7 ms/layer above their harness sum at 4k.

FIX — PLOW_NV_FATLITE_MOE (recipe: PLOW_BUILD_FATLITE=1 + PLOW_BUILD_FATLITE_MOE=1): FATLITE's
stripping (native GEMM / GLU / flash arms the fat object never runs once Lt, the GEMM object and
the FA objects own them) but the Gemma MoE prefill opcodes stay in, at occ-1 with the full
register budget. 0 bytes of spill stack. Block outputs BIT-IDENTICAL to the unstripped object.
  stripped, n128 DOWN                1.004 / 2.541 / 2.253
  stripped, n256 DOWN                0.964 / 2.381 / 2.133   -> n256 wins again; in situ == harness sum
= -0.70 ms/layer at 1k and -1.3 ms/layer at 4k against the original packet.
RULE: a harness win is a hypothesis. Confirm on a block with real inputs before a served build.

## Op-fusion audit (2026-09-20) — user direction: "make sure we fuse ops properly"

METHOD: `plowrt disasm <assets> --program 1|<T> --format json` -> per-layer op chain
($T/opseq.py); fusion knobs from emit_config.rs vs the recipe (12B and 26B recipes agree on every
fusion knob); unverified knobs A/B'd on stripped-fat sliding blocks; decode gate/body profile from
a -DPLOW_NV_TRACE=1 decode object in step_bench ($T/cc_dec.sh, $T/dec_trace.sh).

DECODE, 17 ops/layer (521/step): FlashDecode, FlashMerge(16 blk), Gemv(o), NormResidualNorm(1 blk),
ScoreFast(16), Topk(1), GemvGlu, Gemv(down), RmsNorm h1 (1), MoeExpertGluNorm, MoeExpertDown,
MoeCombineNorm(1), NormResidualNorm(1), GemvQkv, HeadNormRope x3 (2 blk each).
  fused already: sandwich NRN (x2), QKV (25 sliding layers), dense gate|up + GLU, pre-norm + MoE GLU.
  NOT fused, and why:
    fused lm_head+argmax (PLOW_FUSE_ARGMAX)      measured SLOWER here, 6.501 vs 5.837 ms
    MoE tail op72 (PLOW_GEMMA_MOE_TAIL_FUSE)     documented negative (+0.18 ms/token), B=1 only
    fused router (PLOW_GEMMA_MOE_ROUTER_FUSED)   142 ms/step
    PLOW_FUSE_HNR, PLOW_FUSE_MERGE               null in decode AND prefill (re-measured today)
    q|k on the 5 k_eq_v full layers              no fused arm exists; bounded by the NO_FUSE_QKV A/B below
  IN-SITU PROFILE (block 0, B=1): gate 24% / body 73% / signal 3%. The gate time sits in GemvQkv
  (gate ~= body: it waits on the single-block MoeCombineNorm -> NormResidualNorm tail, ~14 us/layer),
  Gemv(o) (FlashDecode stragglers + FlashMerge) and GemvGlu (the post-attention NRN). A packet
  boundary itself is ~1.4 us (0.75 ms over 521 packets), and the norm bodies are already
  register-cached with the 2-barrier block_sum — so a fused tail is worth ~1%, which is why op72
  lost. The tail's real cost is CombineNorm's 90 KB f32 `part` read on ONE block (bf16 partials
  would halve it). At B=4 the profile is MoE GLU 25% + DOWN 17% body, gate 19.5%.

PREFILL, 23 ops / 13 segments per layer: FlashPrefill | Gemm o | NormResidual+RmsNorm | Gemm gate |
Gemm up | Glu | Gemm down | RmsNorm h1 + RmsNorm xn2 + 73 74 75 76 77 + NormResidual | RmsNorm |
Gemm q | Gemm k | Gemm v | HeadNormRope x3.
  PLOW_PF_GFUSE=1 (sandwich fusion in prefill)   MEASURED NEGATIVE: 1.24 -> 1.27 ms/layer at 1k,
                                                  3.64 -> 3.72 at 4k, equal at 128. Stays off.
  segment floor at T=128: Lt GEMM ~11 us, tiny native segments 10-17 us -> the 12 non-MoE segments
  are 0.19 ms/layer; the MoE segment is 0.49 ms (72%) and is weight-streaming-bound (every touched
  expert streams once), not launch-bound. Segment-level fusion has ~nothing left.
  Structural candidates NOT built (value estimated from the segment table): one Lt GEMM for q|k|v
  and one for gate|up (weights are 4096-multiples, so they can be carved contiguously; HNR / Glu
  would need packed-input variants): -2 / -1 Lt calls per layer = ~-0.7 ms/chunk at T=128 (3%),
  ~1% at 1k, nothing at 4k.

BUG FOUND BY THE AUDIT — a recipe knob that never reached its object: `[objects.env]
PLOW_EXTRA_DEFINES="-DPLOW_MOE_DOWN_SG=8u"` only reaches the segments script; the decode object is
built by plowc from the manifest, so every served packet ran sg4 (served C4 TPOT 10.99 == the
sweep's sg4 11.03, not sg8's 10.65). Now a manifest rule: shapes.moe_down_inter (from the decode
MoeExpertDownGemma sites) -> tuning.moe_down_sg=8 when I_moe % 32 == 0 -> `#define
PLOW_MOE_DOWN_SG 8u` in plow_config.h on sm_90a. Block decode us/layer: B=4 413.6 -> 400.4,
B=16 982.7 -> 958.3, B=1 flat.

FUSIONS PROTOTYPED / A-B'D IN THIS AUDIT (all measured, none shipped):
  * decode QKV fusion is worth NOTHING here: PLOW_NO_FUSE_QKV=1 on a sliding block 216.9 / 398.3 /
    955.5 us vs fused 216.3 / 400.4 / 959.6 (B=1/4/16). So a q|k arm for the k_eq_v layers is moot.
  * prefill DOWN -> combine fusion (the AMD PLOW_MOE_PF_ATOMIC shape: DOWN atomicAdds gate*y into
    ONE [T,H] f32 row per token, combine runs unchanged with k=1), harness, replayed real routing:
      T=1024  down 0.242 -> 0.277, combine 0.098 -> 0.030, clear +0.005  = -0.03 ms/layer
      T=4096  down 0.654 -> 0.795, combine 0.372 -> 0.226, clear +0.017  = wash
    The atomic RMW in DOWN eats the combine saving, and the k-way sum becomes arrival-order
    (99.995% of outputs equal, relL2 1e-5, run-to-run nondeterministic). Dropped. (AMD measured
    bf16 partials at ~0% and top-1 flips; its deterministic f64 form costs more than the atomic.)
  * decode DOWN -> combine: not viable deterministically — DOWN spreads 22528 (slot, channel) dots
    over ~8.4k sub-groups at ~2.7 rounds per block; one owner summing all 8 slots of a channel is
    8 rounds (3x DOWN's latency).
VERDICT: every fusion that pays is already on; the only structural candidates left are the two
Lt GEMM merges above (~3% at T=128, ~1% at 1k).

ALSO MEASURED TODAY (null): MoE half vs full tiles on the stripped object, in situ — 0.965 / 2.386
/ 2.132 vs 0.980 / 2.442 / 2.121 ms (rand-1k / rand-4k / hot-4k). Half stays.

## STANDING vs vLLM 0.28 after this round (packet b26u, commit 6a3b1cfa + CSVs), 2026-09-20

Realtime ladder, mean TTFT ms (start of day -> now | vLLM):
  128/C1     31.67 ->   23.29 |  39.52  WIN        128/C4     56.59 ->   40.54 |  73.31  WIN
  1024/C1    66.80 ->   46.76 |  44.11  1.06x      1024/C4   133.63 ->   95.77 |  93.46  1.02x
  4096/C1   174.71 ->  134.48 |  93.71  1.44x      4096/C4   323.86 ->  265.81 | 227.59  1.17x
  8192/C1   370.37 ->  288.73 | 181.10  1.59x      8192/C4  1194.36 -> 1010.26 | 458.74  2.20x
TPOT ms:  C1 5.83 / 5.99 / 6.06 / 6.13 (was 5.91-6.19) | vLLM 5.03-5.09   1.16-1.20x
          C4 10.72 / 11.99 / 13.49 / 15.11 (was 10.99-16.67) | vLLM 7.24 / 7.57 / 7.92 / 8.53
Serving tok/s: 1024/C16 421 -> 456 | 1380, 1024/C32 454 -> 499 | 1736, 4096/C16 237 -> 263 | 885,
               4096/C32 237 -> 263 | 1081.   Still 3.0-4.1x behind: decode batch scaling.
What moved it: row-spread router + warp top-k, float4 prefill combine, n256 DOWN, and above all
the stripped fat prefill object (FATLITE_MOE); sg8 finally reaching the decode object.

NEXT, by value:
  1. Serving/decode batch scaling (3-4x gap): MoE GLU+DOWN are 42% of the B=4 step body and more at
     B=16; the per-slot walk streams B*8 expert weight sets per layer. Parked grouped-decode patch
     (plans/gemma4-26b-decode-grouped-moe.wip.patch) needs an IN-SITU block verdict, not the
     harness one (0.557 vs 0.868 ms/layer at B=16 was harness-vs-harness).
  2. 1024/C1 and 1024/C4 are 2-6% from flipping: attribute the ~9 ms between served TTFT and
     30 x block with BENCH-LIKE prompts ($T/mk_rand_prompts.py; " hello"*N routes to 8 experts and
     is ~11 ms cheaper at 1k than random-token prompts), then the two Lt GEMM merges (~1%).
  3. 4k/8k: MoE segment is still 65% of a layer and at the tile design's bandwidth roof; a vendor
     grouped GEMM is worth ~-18 ms at 4k (torch._grouped_mm ceiling) and needs a host sync or
     CUTLASS. hd512 long-KV attention (GQA-8 KV re-staged per head) is the 8k/16k item.

## Status

| step | state |
|---|---|
| Kernel-dims audit | **done** |
| 26B serves on H100 | **done** — both root causes fixed |
| vLLM 26B reference ladder | **done** (8 cells) |
| Plow 26B rung ladder | **done** (6 cells; 8K rung blocked by max_ctx, now fixed to 8704, needs rebuild) |
| cuBLASLt probe + A/B | probe done (40 entries); A/B in flight |
| Beat vLLM | **PARTIAL** — TTFT wins at 128/C1 (31.9 vs 39.5) and 128/C4 (56.7 vs 73.3); C1 TPOT 1.3x, TTFT >=1024 1.5-2.1x, C4 TPOT 1.6-2.0x, serving tok/s 3.7-4.9x behind |
| plowc/plowrt release build | done |
| Emit + objects + role emit | **done** (`045a38e3`) |
| Roofline model corrected + validated on 12B | **done** |
| vLLM 26B reference harness | **done** (`91fd8c51`) |
| vLLM 26B reference ladder | **partly measured** (128, 1024; 4096/8192 running) |
| Isolated grouped-MoE repro | **done, passes** (`91fd8c51`) |
| **Plow 26B serving** | **BLOCKED — prefill faults, see above** |
| Rung ladder 128/1024/4096/8192 | blocked on the fault |
| Roofline compare + kernel push | blocked |
| High concurrency / throughput | blocked |

## Next actions, in order

1. Root-cause the prefill fault: `disasm --stream` wait/bump counts for inst 19/20
   of program T=128. This is the whole critical path — no plow number exists until
   it is fixed.
2. Finish the vLLM reference ladder at 4096 and 8192, and add C16/C32 for the
   serving cells.
3. Only then: rung ladder, roofline attribution (`PLOW_PF_SEG_TIME=1`, attribution
   only), and the kernel work the dims audit already identified
   (hd512 px4 GQA-8 variant, `PLOW_NV_FA_GF_FULL=8`, `PLOW_NV_FA_TC_GQA8_HD512`,
   tensor-core MoE decode GLU for high concurrency).

## Protocol

Same as the 12B campaign: `scripts/campaign/campaign.py` (build/bench/compare/ledger),
one TOML recipe per cell, every GPU run under `perf-data/tools/gpulease -n 1`,
`--profile realtime|throughput` always passed, same precision on both sides,
`PLOW_PREFIX_CACHE=0` for any vLLM-matched cell.
