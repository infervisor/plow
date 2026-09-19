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

1. **MoE decode tensor-core walk — the big one.** `op_moe.cuh`'s Gemma expert GLU and
   DOWN decode bodies carry ONLY the dot8 CUDA-core walk. `op_gemv_mma.cuh`'s
   `mma.sync m16n8k16` row-block walk is wired into the three DENSE walks behind
   `PLOW_NV_GEMV_MMA` but never into the MoE ones. Decode runs at 4.6 % of the memory
   roof with TPOT flat across batch — compute-bound, the exact condition the dense
   conversion fixed on 12B (C16 44.6 -> 16.2 ms). Attribute with decode step timing
   before authoring: the dense hook only engages at `MM >= 4`, so at C1 the win may
   instead be in the ~150 extra serialized MoE stages per step (5 MoE ops x 30 layers).
2. **cuBLASLt algorithm table — DONE, and it was a null at in128.** The store had 48
   entries for 12B's 3840-keyed shapes and 0 for 26B's 2816-keyed ones; `campaign.py
   probe` wrote 40 measured entries. First A/B cell (128/C1) came back 42.24 vs 42.21 ms
   — noise, as expected where prefill is negligible. The 1024/4096 cells are the real
   test. Record the verdict here either way; a null at every rung would mean the
   prefill inefficiency is in the native GEMM path or the attention role, not the
   backend — which is exactly what the 12B campaign found ("the lever for 128 tokens is
   launch COUNT, not the backend").
3. **fatlite, recovered properly.** `PLOW_BUILD_FATLITE=0` is currently forced to make
   MoE prefill exist at all, surrendering the occupancy win (12B: -2.8 ms @1024,
   -4.8 ms @4096). The fix is to widen the MoE prefill guard so the ops survive
   fatlite, not to flip the knob back.
4. **hd512 GQA-8 role.** 26B runs the generic hd512 body because the px4 BQ64 object
   hardcodes `n_kv_head=1`. A GQA-8 variant is the long-context prefill lever.

## Status

| step | state |
|---|---|
| Kernel-dims audit | **done** |
| 26B serves on H100 | **done** — both root causes fixed |
| vLLM 26B reference ladder | **done** (8 cells) |
| Plow 26B rung ladder | **done** (6 cells; 8K rung blocked by max_ctx, now fixed to 8704, needs rebuild) |
| cuBLASLt probe + A/B | probe done (40 entries); A/B in flight |
| Beat vLLM | **NOT achieved** — decode 7-10x behind, prefill 2.8-4.6x |
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
