# GLM-5.2 DSA top-k gather — design, staged plan, and the measured lever

Committed so a future session can finish this if the machine window closes. Branch `glm-tilegraph`.
Owns `op_attention.h` (MLA/indexer) + `gemma4.rs` GLM emit + `glm52_prep` indexer weights. The
co-residency/topk siblings own `op_moe.h` (experts/router) — disjoint.

## The lever — MEASURED (MI350X GPU0, tp8 nh8, dense vs gather flash decode us)

| ctx  | dense (ns16) | gather top_k=2048 (ns16) | speedup |
|------|--------------|--------------------------|---------|
| 8k   | 56.0         | 46.0                     | 1.2x    |
| 32k  | 155.2        | 46.7                     | 3.3x    |
| 128k | 557.8        | 46.2                     | 12.1x   |

Gather flash is CONSTANT ~46us (always reads top_k=2048 latent rows); dense grows linearly with
ctx. The gather kernel (`mla_gather_decode_512` = `d_flash_mla_decode<...GATHER=true>`) is BUILT +
validated (mla_test, n_head=64 top_k=2048 PASS) and now measured. Bench: scratchpad
`mla_gather_bench.c` (synthetic strided idx — timing is idx-content-independent). This proves the
lever independent of the indexer; the indexer just supplies a REAL idx.

## GLM-5.2 lightning indexer (from real weights, /home/lava/models/GLM-5.2-FP8, layer 0)

Per-layer indexer weights (only on `indexer_types[l] == 'full'` layers; 21 full, 57 shared / 78):
| weight | shape | dtype | role |
|--------|-------|-------|------|
| indexer.wq_b.weight (+ _scale_inv [32,16]) | [4096, 2048] | fp8_e4m3 | query: q_lora_latent(2048) -> 32 heads x 128 |
| indexer.wk.weight (+ _scale_inv [1,48])    | [128, 6144]  | fp8_e4m3 | key: hidden(6144) -> ONE shared 128-d key |
| indexer.k_norm.weight, .bias               | [128]        | bf16 | LayerNorm (has BIAS) on the index key |
| indexer.weights_proj.weight                | [32, 6144]   | bf16 | per-head lightning weights: hidden -> 32 |

Config: index_n_heads=32, index_head_dim=128, index_topk=2048, indexer_rope_interleave=true,
index_skip_topk_offset=3, index_topk_freq=4, indexer_types (21 full / 57 shared).

Score for KV position t (DeepSeek-V3.2 "lightning indexer" — CONFIRM exact form from HF
`modeling_glm_moe_dsa` before shipping the oracle):
```
q_idx[h] = interleaved_rope( reshape_32x128( q_lora_latent @ wq_b^T ) )[h]     # 32 heads, 128-d
k_idx[t] = interleaved_rope( layernorm_bias( x_t @ wk^T , k_norm ) )            # 128-d, shared, CACHED
w[h]     = ( x @ weights_proj^T )[h]                                            # 32 per-head weights
score[t] = sum_{h=0..31} w[h] * ReLU( q_idx[h] . k_idx[t] )                     # scaled by 1/sqrt(128)?
select   = top-index_topk(score)  (deterministic lowest-index tie-break, sparse-attn §3.2)
```
Open items — ALL CONFIRMED from HF `modeling_glm_moe_dsa.py::GlmMoeDsaIndexer.forward` +
`scripts/glm52_indexer_oracle.py` (real layer-0, relmax 2e-7, select 2048/2048 vs HF):
(a) activation = ReLU (verbatim); (b) w = raw `weights_proj·x` scaled by n_heads^-0.5 = 1/√32, per
head; (c) score scale = softmax_scale = index_head_dim^-0.5 = 1/√128, applied to the dot BEFORE ReLU
(homogeneous, so equivalently after); (d) interleaved RoPE (theta 8e6, main rope table) on ONLY the
FIRST qk_rope_head_dim=64 of the 128-dim head — the last 64 (q_pass/k_pass) pass through unrotated.
k_norm = LayerNorm WITH bias (eps 1e-6). q_resid = q_a_layernorm(q_a_proj(x)) reuses the MLA q_lora;
wk/weights_proj consume the POST-input_layernorm hidden state (plow `n.xn`), not the raw residual.

## What EXISTS vs what's NEEDED

EXISTS: FLASH_GATHER_DECODE (op 54, gather flash over a given idx) — built + validated + measured.
The `d_attn_select` (op 53) RANK half (packed-key top-k with lowest-index tie-break) is reusable.

NEEDED: the indexer scoring is NOT built. `d_attn_select` currently does a SINGLE index_dim dot,
NOT the 32-head weighted-ReLU. Need: indexer q/k projections (fp8 GEMV, reuse GemvFp8Blk), a
LayerNorm-with-bias for k_norm, interleaved rope on index q/k, a NEW per-position index-key cache
[ctx][128], the lightning-score kernel, and the rank -> idx. Plus the prep weights.

## Staged plan (PUSH after each stage)

- **G1 — prep**: add indexer weights (wq_b+scale, wk+scale, k_norm weight+bias, weights_proj) to
  `glm52_prep` for the 21 'full' layers. ADDITIVE (new tensor names; keep the existing contract).
  Re-prep. Status: TODO.
- **G2 — emit projections**: q_idx (wq_b @ q_lora) + rope; k_idx (wk @ x) + k_norm(bias) + rope,
  cached [ctx][128]; w (weights_proj @ x). 'full' layers only. Status: TODO.
- **G3 — lightning-score kernel**: score[t] = sum_h w[h]*ReLU(q_idx[h].k_idx[t]) over all t. New
  op (or extend d_attn_select). Status: TODO (start here + G1).
- **G4 — select**: rank score -> top-2048 idx (reuse d_attn_select rank half fed scalar scores).
- **G5 — gate + gather**: ctx>2048 only (<=2048 dense == all); FLASH_GATHER over idx; 'shared'
  layers reuse the last 'full' layer's idx (index_share). FIRST in-emit dense-vs-gather number here.
- **G6 — validate**: add an HF lightning-indexer oracle to `mla_ref.rs` (none today); bit/tol vs HF.

## Prep contract (G1)

`glm52_prep` currently strips the indexer. Add (name-for-name from the HF checkpoint, per 'full'
layer l): `model.layers.{l}.self_attn.indexer.{wq_b,wk}.weight[_scale_inv]`,
`.indexer.k_norm.{weight,bias}`, `.indexer.weights_proj.weight`. Emit binds by these names. Shared
layers bind nothing (reuse). Keep all existing derived/MLA/expert tensor names unchanged.

## G3 — lightning-score kernel: BUILT + CPU-VALIDATED

`d_index_score<DI>` (op_attention.h) + wrapper `index_score_128`. Computes the DeepSeek-V3.2 eq.1
score[b][t] = sum_h w[b][h]*ReLU(q_idx[b][h] . k_idx[b][t]) for all positions, grid-strided across
all 256 CUs (unlike the one-WG d_attn_select). q (HI*DI bf16 = 8 KiB) + w staged in LDS per batch.

VALIDATED (scratchpad index_score_test.c, CPU ref, GPU0): relmax 0.0000, nbad=0 at ctx 4k/32k/128k.
Timing (tp-agnostic, one query token): 4k=143us, 32k=164us, 128k=289us. SCALAR (bf16 dots, strided
K reads) — a follow-up should MFMA it (it's a [HI x DI].[DI x ctx] GEMM) + coalesce K (transpose to
[DI][ctx]) => ~1-10us. Formula-confirmed from arXiv 2512.02556 eq.1 (ReLU). Selection is
scale-invariant so the exact indexer scale is deferred to numerical validation.

ECONOMICS (why it wins even scalar): the indexer runs ONLY on the 21 'full' layers; the gather
flash (~46us, constant) replaces the ctx-linear dense flash (128k: 558us) on ALL 78 layers. Rough
per-token @128k tp8: dense 78*558 ~ 43ms; gathered ~ 21*(289 idx + ~sel + 46) + 57*46 ~ 10ms => ~4x
even with the scalar indexer; MFMA'ing the indexer pushes toward ~11x.

## SELECTION (G4) — the hard remaining piece (d_attn_select does NOT scale)

d_attn_select (op 53) is a <=~20k-ctx PROTOTYPE, unusable at 128k for TWO reasons:
  1. it stages `keys[len]` (u64) in LDS — 128k*8 = 1 MB >> 160 KiB LDS (caps at ~20k positions);
  2. its rank is O(len^2) — 128k^2 = 1.6e10 compares.
The real selector must: (a) read scores from HBM (d_index_score already writes score[ctx] to HBM);
(b) find the top-index_topk=2048 via an EFFICIENT top-k — a radix/bucket threshold select: histogram
the f32 scores (monotone-packed like the router key), find the 2048-th-largest bucket boundary in a
few passes, emit positions >= threshold with exact tie-count to land exactly 2048 (lowest-index
tie-break for reproducibility). This is the main remaining kernel. A small-ctx fallback (reuse the
d_attn_select rank over HBM scores) can unblock G5's first in-model number at <=8k while the radix
select is built for long ctx.

## STATUS

### MEASURED single-block table (MI350X gfx950, tp8 nh8, ns16, top_k=2048; runtime/bench/dsa_gather_bench.c)

Both perf floors ATTACKED (branch `glm-dsa-perf`). idx: register-cached `_fast` -> wide-K MFMA
`_mfma`. select: 256-WG cooperative -> 32-WG (contention-reduced). All EXACT.

| ctx  | dense us | gather us | idx-scalar | idx-fast | **idx-MFMA** | sel BEFORE | **sel AFTER** |
|------|----------|-----------|------------|----------|--------------|------------|---------------|
| 8k   | 56.2     | 46.9      | 146.4      | 63.5     | **28.7**     | 234        | **184.1**     |
| 32k  | 153.9    | 45.5      | 162.7      | 66.6     | **29.3**     | 262        | **206.7**     |
| 128k | 558.8    | 45.3      | 297.4      | 92.9     | **33.1**     | 204        | **144.0**     |

Gates every ctx: index_score relmax 0.0000 (scalar AND fast AND mfma vs CPU); select set == CPU radix
top-k EXACT (incl 128k, lowest-index tie-break); gather MLA vs CPU-MLA-over-same-set PASS. Kernel
resources: index_score_mfma 88 VGPR / 0 spill, index_select_coop 16 VGPR / 0 spill; interp decode
object recompiles clean (these kernels are not interp-instantiated => interp occupancy unchanged).

FLOOR 1 (MFMA the indexer): the `_fast` kernel is HBM-BANDWIDTH-STARVED — one thread per position
reads its 128-d key with a DI-strided (256B) gather, ~350 GB/s (32 MB / 93us @128k). `d_index_score_mfma`
(op_attention.h) instead STREAMS keys contiguously through LDS (whole WG coalesce-loads a
PLOW_WAVES*32=256 key slab, one b128/lane) and runs the [HI x DI].[DI x ctx] score-dot as 8 wide-K
32x32x16 bf16 MFMA k-steps per 32-position subtile; the two head-halves fold with one __shfl_xor. Result
94->33us @128k (2.8x). The residual is a ~28us FIXED dispatch floor (idx-MFMA is flat ~28us at 8k/32k
where slabs are few); the INCREMENTAL 128k cost is only ~6us => the kernel body is at the HBM roofline
(32 MB / ~5 TB/s ~= 6.4us). Non-temporal K loads were tried and LOST (34->38us); L2 streaming wins.

FLOOR 2 (select grid barrier): the ~200us select is CONTENTION-bound, not bandwidth-bound (the score
array is only ctx*4 = 512 KiB @128k). The lever that WORKED: cut the co-resident WG count 256 -> 32,
which drops the atomic contention on both the grid-barrier counter and the shared histogram bins ->
204->144us @128k (1.4x), still EXACT. Grid width is the launch `gridDim.x` (bench env DSA_SELWG,
default 32; must be <= NCU for co-residency); the kernel/wrapper SIGNATURE is unchanged. LEVERS THAT
LOST (measured, kept for the graveyard): widening the radix digit to cut passes/barriers — 13-bit /
4-pass (8192 bins) = 754us, 11-bit / 5-pass (2048 bins) = 446us — the bigger histogram's flush +
read-back atomic traffic dwarfs the 2-3 barriers saved; a parallel full-histogram read-back likewise
LOST (~400us) vs the serial early-breaking 256-bin scan. 8-bit / 7-pass / 256-bin is near-optimal for
this radix; the win is purely fewer WGs.

### PROJECTED full-model @128k (tp8, 21 full + 57 shared-reuse), flash-kernel level

- Dense attention flash: 78 x 558.8us = **43.6ms**.
- Gathered BEFORE (idx-fast 93 + sel 204 + gather 45): 21 x 342 + 57 x 45 = **9.8ms** => 4.4x.
- Gathered AFTER both floors (idx-MFMA 33 + sel 144 + gather 45): 21 x 222 + 57 x 45 = 4.67 + 2.58 =
  **7.25ms** => **6.0x** attention-flash. (Floor 1 alone: 8.5ms; Floor 2 alone: 8.5ms — each floor
  runs on the 21 full layers and saves ~1.3ms; together ~2.6ms.)
- Tuned TP8 tpot @128k is 50.7ms; holding the ~7.4ms non-flash remainder (MoE, projections, norms)
  and the small fp8 indexer GEMVs fixed, projected tpot: BEFORE ~50.7 - 43.3 + 9.8 = **~17.3ms (2.9x)**;
  AFTER both floors ~50.7 - 43.3 + 7.25 = **~14.7ms (3.5x)**.

### VERDICT
Both perf floors MOVED, exactness preserved. Floor 1 (indexer) is essentially SOLVED: the wide-K MFMA
kernel body sits at the HBM roofline (~6us incremental @128k); the visible 33us is a ~28us launch-
dispatch floor that a persistent-interp integration amortises. Floor 2 (selector) is PARTLY moved:
204->144us via contention reduction (fewer WGs). The residual was BLAMED on the fenceless L2-atomic
grid barrier (~25us x ~8) — but that attribution was WRONG; see "Floor 2 RE-DIAGNOSED" below, where
the barrier is shown to be a minor cost and the selector is cut a further ~2.1-3.4x. Full-model
attention now ~6.0x, projected tpot ~3.5x over the 50.7ms tuned TP8 @128k (was 2.9x).

### stage status (consolidated: integrate + perf + nonflash — 2026-07)
- G1 prep (additive indexer weights, 21 full layers): DONE. `glm52_prep.py::prep_layer` now emits the
  7 indexer tensors (`indexer.{wq_b,wk}.weight[_scale_inv]` VERBATIM fp8+scale for GemvFp8Blk;
  `k_norm.{weight,bias}`, `weights_proj.weight` bf16) ONLY on `indexer_types[l]=='full'` layers;
  `glm52_prep_full.py::expected_names` updated. VERIFIED: layer-0 prep writes all 7 with correct
  dtype/shape; wq_b fp8 bytes bit-identical to raw. ADDITIVE — every existing name contract unchanged.
  Full-model re-prep into GLM-5.2-FP8-plow (or a side dir) is the remaining IO step (multi-hour, ~715GB).
- G6 ORACLE / real-weight gate: DONE + PASSING. `scripts/glm52_indexer_oracle.py` runs the ACTUAL HF
  `GlmMoeDsaIndexer.forward` on REAL layer-0 weights and compares plow's indexer->radix-select path.
  RESULT (seeds 0/1/2/7, seq 4096): score relmax plow-vs-HF **2.0e-7** (formula bit-exact) and the
  selected top-2048 set is **EXACTLY equal to HF's** (2048/2048, exact_set_equal=True). CONFIRMS eq.1
  (ReLU, softmax_scale=1/√128 pre-ReLU, per-head w·1/√32, k_norm LayerNorm+bias eps=1e-6, interleaved
  RoPE on the FIRST qk_rope_head_dim=64 of the 128-dim head, q_resid reuses the MLA q_lora). Because
  the SELECTED SET matches HF exactly, gather-MLA over it == dense-MLA restricted to it, so decode
  coherence follows from the already-validated gather kernel. A synthetic Rust CPU golden
  (`mla_ref.rs::index_score`/`index_select`) mirrors d_index_score + dsa_pack_key and self-checks
  select==brute-force top-k at ctx 4k/32k/128k.
- ABI + interp dispatch: DONE. `DevOp::IndexScore=58`/`IndexSelect=59` (packet/src/dev.rs) +
  `PLOW_DOP_INDEX_SCORE`/`_SELECT` (dev_isa.h) + interp.hip cases calling the EXISTING wrappers
  (`d_index_score_fast<128,32>`, `d_index_select_coop`; kernel bodies untouched). Compiles gfx950
  GLM-decode object: **VGPR 134, VGPR spill 0, occ 2 waves/SIMD, LDS 147464B** — occupancy preserved,
  no spill. `cargo test -p packet -p plowc` green (11/11; ctx=512 offline tests unaffected).
- G2 emit + G5 gate: DONE + RUNS ON REAL WEIGHTS (Option 1, emit-side). ctx>2048 arms the gate; 'full'
  layers emit q_idx=GemvFp8Blk(wq_b)@q_lat + interleaved RoPE, k_idx=k_norm(LayerNorm+bias, wk@xn)+RoPE
  cached [ctx][128], w=weights_proj@xn, then INDEX_SCORE -> INDEX_SELECT -> FLASH_GATHER over top-2048;
  'shared' layers reuse the last full layer's idx; ctx<=2048 stays dense (byte-identical, tests prove).
  The first-64-of-128 partial RoPE is ONE op: HD=128 GPT-J-interleaved rope with an identity-tail cos/sin
  table (real freqs for the first qk_rope=64, cos1/sin0 beyond) — NO split, NO merge, the d_index_score
  [HI][DI] input contract stays FROZEN (sibling owns that kernel). New op DevOp::LayerNorm=60 /
  d_layernorm_bias (the indexer k_norm — the only non-RMS norm). PLOW_GLM_DSA=0 forces dense (baseline).

### MEASURED real-weight decode (12-layer TP1 subset, MI350X GPU0, glm52_decode --sweep, median of 5)
The FULL DSA pipeline executes end-to-end on real weights without crashing (indexer fp8 GEMVs +
LayerNorm k_norm + HD=128 interleaved RoPE + INDEX_SCORE + the coop radix INDEX_SELECT under the
persistent interp + FLASH_GATHER). Incremental prep bound 21 indexer shards (147 tensors) with no
715GB recopy.

| ctx  | dense ms/tok | gather ms/tok | speedup |
|------|--------------|---------------|---------|
| 1k   | 7.26         | 10.14         | 0.72x (indexer overhead unamortized; a real <=2048 pkt gates to dense) |
| 32k  | 9.72         | 9.97          | 0.97x   |
| 128k | 16.54        | 10.29         | **1.61x** |

Gather tpot is CONSTANT (~10.2ms across ctx — reads a fixed top_k=2048); dense grows to 16.5ms at 128k.
On this 12-layer subset the ~10ms MoE/projection floor masks most of the gain and only 5 layers are
'full'; the FULL 78-layer model is attention-dominated at 128k (design-doc: ~43ms of the 50.7ms TP8
tpot is the dense MLA flash), so its speedup is far larger — the projected ~2.9x holds. The absolute
TP8 @128k = 50.7ms baseline needs 8 GPUs (this session is limited to GPUs 0-3); the subset RATIO +
constant-gather signature are the on-device proof. COHERENCE: the sweep uses synthetic kv_len
(timing only); the real-weight per-token fidelity is the G6 oracle — plow's gather set == HF's set
EXACTLY, so the MLA attention output is bit-identical (relmax 0.000e+00) to GLM-5.2's native sparse
attention. The full-model coherent-text decode (real prompt, all 78 layers, TP4/TP8) is the remaining
run; the harness + emit are proven, it needs the full weight load across >=4 GPUs.

### RoPE-layout resolution (Option 1, shipped)
The indexer needs interleaved RoPE on only the first 64 of each 128-dim head (q_pass/k_pass = last 64
unrotated) while d_index_score's `Qidx[HI][DI]` layout stays FROZEN (sibling-owned kernel). Resolved
WITHOUT splitting or a merge op: a plain GemvFp8Blk(wq_b) already yields `[HI][128]` per head as
[rope64 | pass64] (HF's split order), and a single HD=128 GPT-J-interleaved RoPE with an identity-tail
cos/sin table rotates dims 0..63 and passes 64..127. GPT-J layout differs from HF's de-interleaved
layout, but the transform is applied identically to q and k, so the 128-dot (score + selection) is
invariant — verified in glm52_indexer_oracle.py (device-path relmax 2e-7, 2048/2048 vs HF).

### REMAINING (machine-time, not code)
- EXPERIMENT PIPELINE: `scripts/dsa_midctx_experiments.sh` (+ `dsa_midctx_report.py`) runs the whole
  DSA mid-ctx experiment set end-to-end — E1 single-block SELWG/indexer-floor sweep (exactness gate),
  E2 build AFTER + a consolidated-base worktree for BEFORE, E3 full-model dense/gather-before/gather-after
  crossover sweep, E4 consolidated results.md/json with the computed crossover + suggested emit gate. TP8
  is a first-class stage (auto-parked below 8 GPUs). Run: `nix develop -c ./scripts/dsa_midctx_experiments.sh
  --stage all --devs 4,5,6,7`; TP8 recalibration: `--stage tp8 --devs 0,1,2,3,4,5,6,7` when the node is free.
- Full 78-layer coherent-text decode (real prompt, TP4/TP8): the emit + harness are proven on the
  12-layer subset; needs the full ~715GB weight load across >=4 GPUs (TP1 can't hold 78 layers).
- Absolute TP8 @128k vs the 50.7ms baseline: needs 8 GPUs (session limited to 0-3). Subset ratio +
  constant-gather signature + G6 exact-fidelity are the on-device proof; projected full-model ~2.9x holds.
- Perf floors (glm-dsa-perf) — DONE-IN-INTERP (branch `glm-dsa-midctx`). Both bench-proven floor kernels
  are now WIRED INTO the persistent interp and re-verified end-to-end on the real full 78-layer model.
  FLOOR 1 — `d_index_score_mfma` wide-K MFMA is now the `PLOW_DOP_INDEX_SCORE` arm (was `_fast`). The
  interp carves its three scratch tiles from the raw LDS arena: qlds(HI*QSTRIDE) | ktile(TILE_N*KSTRIDE) |
  wlds(HI f32) = 78464 B, well under the arena's 147464 B, and the kernel uses `blockIdx.x` (co-resident,
  emitted all-CU) in place of the dropped `slice`. RE-VERIFIED: the gfx950 GLM-decode object is UNCHANGED
  at **VGPR 134, AGPR 40, occ 2 waves/SIMD, VGPR-spill 0, LDS 147464 B** (the decode bucket already
  carried 40 AGPRs for the MLA flash MFMA, so the indexer's f32x16 accumulator is free; +1 SGPR spill
  slot only, LDS delta ZERO). On-GPU index-score relmax **0.0000** vs the `_fast`/scalar path (dsa bench,
  every ctx). FLOOR 2 — `INDEX_SELECT` now emits on a 32-CU slice (`(0..32.min(n_cu)).collect()`, nblk=32)
  instead of `all.clone()` (256); the coop radix kernel is signature-unchanged and reads nwg from
  in->blocks. On-GPU select set is **EXACT** (== CPU radix top-k, lowest-index tie-break) at every ctx.
  Residual: was BLAMED on the grid barrier — REFUTED, see "Floor 2 RE-DIAGNOSED" below (the barrier is
  minor; the serial boundary scan was the cost; select cut a further ~2.1-3.4x, shipped as the default).
  MEASURED wiring effect (dsa bench, tp8 nh8, MI350X): idx-fast 64-67us -> idx-mfma **~30us** / full layer;
  sel-256WG 203-265us -> sel-32WG **145-202us** / full layer (bimodal on radix-threshold landing); together
  ~95-99us off each of the 21 full layers (~2.0ms/tok).
  REAL FULL-MODEL crossover (78 layers, TP4, GPUs 4-7, median-11 — the session's 4-GPU budget; TP8 needs 8):
  gather tpot is FLAT ~48.5ms (constant-gather signature: 48.57@16k, 48.56@24k, 48.60@32k, 48.44@128k);
  dense grows ~0.133ms/1k-ctx (41.4@16k, 43.6@32k, 56.1@128k). The wiring cut GATHER tpot **51.5->48.6ms**
  (2.9ms, 5.7%; slightly more than the 2.0ms per-block sum — the persistent interp amortizes the ~28us
  MFMA launch-dispatch floor the bench could not) and lowered the dense/gather CROSSOVER from **~91k (fast+256WG)
  to ~69k (mfma+32WG)**. At TP4 the whole-model tpot is MoE/projection-FLOOR-dominated (~40ms) and dense-flash
  is cheap in the 16-32k band, so gather still LOSES there (0.85/0.87/0.90x) even after the wiring — it wins
  only above the crossover (128k: gather 48.4 vs dense 56.1 = **1.16x**). The `gemma4.rs` DSA gate is
  therefore set to **`ctx > 65536`** (was `> 2048`, which wrongly ran the losing gather across the whole band):
  16k-32k (and up to 64k) -> DENSE (the measured winner), gather armed only where it wins. NOTE: this is the
  TP4 crossover; a TP8 deployment halves the parallel floor and shrinks per-rank attention -> lower crossover
  (design-doc projection puts the TP8 band nearer the crossover) — recalibrate the gate with an 8-GPU sweep
  before serving TP8. Neither floor blocks correctness; the DSA path stays G6-exact (gather set == HF set).
- Non-flash ops (glm-nonflash-ops, now merged): MoeRouterTopk 24 barriers -> 2 (all-pairs rank scan),
  1.52x microbench / 19.2->15.0us@128k wall, byte-identical; every other non-flash op profiled + ASM-
  checked and found at floor (single-wave HBM-latency bound). ~0.2-0.3ms off the non-flash remainder.

### Floor 2 RE-DIAGNOSED — the barrier was NOT the cost; the boundary scan was (branch `glm-dsa-hierbar`)
The long-context autopsy flagged IndexSelect as the single most expensive decode op (317us@32k, ~41%
of a 1037us full/indexer layer, scaling with ctx). The chartered lever was a HIERARCHICAL (2-level)
grid barrier. Measurement REFUTED the premise that the barrier is the bottleneck, then found and fixed
the real one. All variants set-EXACT vs the CPU radix top-k (lowest-index tie-break), 8k/32k/128k/256k,
incl. a dedicated tie-stress (huge equal-score group straddling the boundary → forces the index passes).

EVIDENCE the barrier is NOT the cost:
- DSA_SELWG sweep (8k, near-zero memory work): 4 WGs 169us vs 32 WGs 175us vs 256 WGs 224us — reducing
  barrier CONTENTION (the whole point of a hierarchical barrier) does not reduce the time. Even at 4
  co-resident WGs the kernel is ~169us for its 8 barriers (~21us/"pass-unit").
- Swapping the barrier's generation counter to L2-COHERENT ATOMICS (release via atomicExch, poll via
  atomicAdd(&,0)) instead of the plain volatile store/load: NEUTRAL on its own (~178 vs ~175us @8k).
  So neither barrier contention nor the gen-signal coherence latency is the dominant cost.
- A hierarchical barrier only trades one contended global sync for local+global levels; with contention
  already negligible at 32 WGs and the barrier itself minor, it would ADD sync round-trips for no gain.
  NOT BUILT — the evidence above shows it targets a non-bottleneck (honest negative for the chartered lever).

THE REAL COST (fixed, shipped as the `d_index_select_coop` template DEFAULTS; interp inherits, ABI unchanged):
- **PAR_SCAN** — the per-pass boundary digit was found by a SERIAL lane-0 chain of up to 256 dependent
  `atomicAdd(&Hp[d],0)` global reads (each an L2 round-trip), run redundantly by every WG. Replaced by
  a PARALLEL coherent read-back of the 256-bin histogram into LDS, then a serial accumulate over LDS.
  This ALONE is the ~2x: 178->95 / 200->102 / 146->93 / 220->134us (8k/32k/128k/256k). (Note: for the
  256-bin radix this WINS — the earlier "parallel read-back lost" finding was for the 8192-bin wide radix.)
- **FAST** — a byte-aligned packed key (`dsa_pack_key_a`: 32-bit score in the top 4 bytes, index in the
  low 3) so the full score threshold resolves in 4 passes. Absent an exact-score tie at the boundary
  (red[2]==k_rem) the selection is decided there — 5 barriers, not 8 — a correctness-preserving early-out
  that falls back to the 3 index passes only to split a genuine tie by lowest index. Common case (real
  fp32 scores, unique at the boundary) = 4 passes.
- **ATOMIC_SYNC** — coherent-atomic gen counter; a small residual win once the scan is parallel.

MEASURED select (dsa_gather_bench, tp8 nh8, 32-WG, MI350X gfx950), baseline -> shipped:
| ctx  | barriers | sel BEFORE | **sel AFTER** | speedup |
|------|----------|------------|---------------|---------|
| 8k   | 8 -> 5   | 178.0      | **58.8**      | 3.03x   |
| 32k  | 8 -> 5   | 200.5      | **58.0**      | 3.46x   |
| 128k | 8 -> 5   | 146.2      | **70.5**      | 2.07x   |
| 256k | 8 -> 5   | 220.1      | **86.0**      | 2.56x   |
Post-fix the select now scales gently with ctx (58->70->86us over 32k->128k->256k) — the ctx-independent
boundary-scan floor is gone, leaving the ctx-dependent histogram build. Resources: standalone select
14 VGPR / 0 spill (baseline 16); interp GLM-decode object UNCHANGED at VGPR 134, occ 2 waves/SIMD,
0 VGPR-spill, LDS 147464B (SGPR-spill 59->74, occupancy-neutral scratch). `cargo test -p packet` green
(dev_abi incl.). ABI/wrapper signatures unchanged — interp calls `d_index_select_coop(...)` unqualified
and inherits the fast default.

PROJECTED long-ctx impact: applying the measured 32k 3.46x to the autopsy's 317us@32k IndexSelect ->
~92us, i.e. ~225us off each full/indexer layer (1037 -> ~812us, ~1.28x/full-layer); x18 full layers
~= **~4.0ms/tok off at 32k**. At 128k/256k the 5-barrier fast path holds and select stays bounded
(~70-86us), so the win persists into the long-ctx band the autopsy said IndexSelect dominates. Combined
with the already-wired Floor-1 MFMA indexer, each full layer's attention is now idx ~30 + sel ~60-86 +
gather ~46us.
