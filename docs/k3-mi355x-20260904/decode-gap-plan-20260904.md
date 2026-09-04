# Kimi-K3 decode gap plan (2026-09-04): 25.25 → ≤ 18.8 ms/token on 8×MI355X TP8

Position: Plow 25.25 ms served TPOT (`perf-data/kimi-k3-plowrt-mi355x-baseline.md`, stack-2,
two exact folds) vs vLLM 0.28 20.88. Target ≤ 18.8 (10% margin) → **−6.45 ms**.

Evidence base
- Final-stack all-rank raw trace `/tmp/k3-xr-phase-gate/trace-stack3.raw` (8192→64, rank 0,
  `/tmp/k3-stack3` bundle = the showdown packet, 2165 decode insts, 233 ordered segments). The
  raw dump carries 37,490 stale records from an earlier process in the unzeroed trace buffer
  (the b=71/39/65/33 rows and the 1e9 µs stragglers in `trace-stack3.report`); they were dropped
  by epoch (`/tmp/decode-gap/trace_filtered.py`, cleaned copy `trace-stack3.clean.raw`).
  Cleaned step span **24.81 ms** device (served 25.25 → host/HTTP ≈ 0.44 ms/token).
- Counter-edge critical path: `scripts/k3_trace_critpath.py` + per-layer split
  (`/tmp/decode-gap/critpath_layers.py`) against `plowrt disasm --program 1 --counters`
  (`/tmp/decode-gap/dec-stack3.clean.json`). Per-WG anatomy: `scripts/k3_trace_wg.py`.
- Before/after: control trace `trace-ctl.raw` (pre-tagged, pre-standalone-MoE) 28.18 ms span.
- vLLM side from `/tmp/vllm-v0.28.0/vllm/models/kimi_k3/amd/*`, pinned AITER, and
  `/tmp/k3-showdown-c1-stack2-20260904/serve-vllm-r1.log` (subagent report, §2).

## 1. Attribution: Plow 24.81 ms device (critical path) vs vLLM ≈ 20.9 ms

Critical path = 1,377 of 1,981 traced packets; span 24.81 = gate 8.77 + body 16.04 ms
(the 184 untraced standalone GLU/DOWN launches appear as the MOE_COMBINE gate).
KDA+MoE layer **246 µs** (gate 87 + body 159; 69 layers = 16.98 ms); MLA+MoE layer **326 µs**
(gate 114 + body 212; 24 layers = 7.82 ms). vLLM ≈ 224 µs/layer average.

| family | Plow ms (path) | KDA-layer µs | MLA-layer µs | packets/layer | vLLM est. µs/layer | vLLM ms | note |
|---|---:|---:|---:|---:|---:|---:|---|
| dense GEMV (bf16, M=1) | 7.86 (body 5.35 / gate 2.51) | 69 | 129 | 10 / 12 | 49 / 42+69 fp4 bmm | ~6.2 | per-WG body 9.1 µs for 86 KB = ~4.5 µs fixed + stream at ~23 GB/s/CU; MLA q-chain serial (28.6 µs gate on #66) |
| MoE (router top-k, GLU+DOWN standalone, combine) | 6.98 | 75 | 75 | 4 (+2 raw launches) | ~32 | ~2.9 | top-k **23 µs** single-WG on path (2.48 ms); standalone segment gap **40 µs** = 17 body + 3 AQL boundaries (2.1 ms) |
| TP reductions (3/layer, tagged one-shot) | 2.44 | 26 | 27 | 3 | 55.7 (AITER custom AR 20.5/14.7/20.5) | ~5.2 | Plow **ahead by 2.7 ms** after tagging (ctl 4.71 → 2.44) |
| AttnRes (×2/layer, 1 WG) | 3.30 | 35 | 37 | 2 | ~6 (Triton attn_res) | ~0.6 | 17.1 µs body/site: ~250 KB through ONE CU |
| KDA (conv3, state step, gated norm) | 2.82 | 41 | — | 3 | 10.4 (fused_kda_decode 8.4 M + f_b) | 0.72 | 18.8 µs of it is CONV3 waiting on the serial f_a→f_b GEMV chain |
| MLA (flash, merge-fold, out-gate, headnorm) | 1.37 | — | 57 | 4 | 124 (asm 43 M + 2×34.6 fp4 bmm M) | ~3.0 | Plow ahead; 2 specialist-segment boundaries ≈ 18 µs/layer |
| other (embed, lm_head 2×16 µs, argmax) | 0.04 | — | — | — | ~0.15 | ~0.15 | |
| host per token | 0.44 | — | — | 233 launches | 1–3 (V1 scheduler + graph replay) | 1–3 | vLLM's device time is ≈ 18–19.5 ms |
| **total** | **24.81 + 0.44** | 246 | 326 | ~21 | ~156 (KDA) / ~265 (MLA) | **20.88** | |

Where the 8.77 ms of path gate goes: standalone-MoE segment (3 boundaries + 17 µs body) 3.73;
head-of-line waits in the in-order global queue 2.24 (GEMV #59 behind up-proj #57 after every
MoE, 12.7 µs × 92; MLA q-chain #66 28.6 µs × 24); KDA CONV3 behind f_a→f_b 1.30; MLA specialist
boundaries 0.19 + 0.25; per-packet protocol gate (HIER on) ≈ 1.4 µs × 1,377 ≈ 1.9.

Bandwidth floor for reference: 23.79 GB touched/token/rank (emitter oracle, `emit.log`;
216 MB dense + 33 MB experts per layer) → 3.0–3.8 ms. Both engines are latency chains
(Plow 21 packets/layer × ~11.5 µs; vLLM ~23–35 kernels/layer inside one FULL graph). MALL
residency across layers is irrelevant at 216 MB/layer with zero cross-layer reuse.

## 2. What vLLM does per decode layer (from source; log `serve-vllm-r1.log`)

- One FULL CUDA graph replay per step (`CUDAGraphMode.FULL_AND_PIECEWISE`, sizes [1..256]),
  no torch.compile fusion passes ("model does not support it" → no AR+RMSNorm fusion).
- Dense bf16 M=1: `ops.wvSplitK` (rocm skinny GEMM) for every projection; router
  `GateLinear` hipBLASLt fp32 + `biased_grouped_topk`; MoE `AITER_MXFP4_BF16` Situv2 A8W4,
  FlyDSL stage1/stage2 (tuned csv: 10.66 + 9.46 = 20.1 µs at M=1) + sort/quant kernels;
  latent-MoE tail = AR(3584) → RMSNorm → addmm up_proj → AR(7168): **3 all-reduces/layer**,
  AITER `CustomAllreduce` one-stage (7 KiB 14.7 µs, 14 KiB 20.5 µs measured, 11.5 µs floor).
- KDA: `_C.fused_kda_decode` (conv + delta rule + gated norm, grid 12 heads) 8.4 µs M.
- MLA: padded asm persistent decode (12 heads → 16) 43 µs M at ctx 8K, W_UK/W_UV as MXFP4
  batched GEMM 34.6 µs each M; `mla_use_nope` → no rotary.
- Per step ≈ 2.4k kernels, 277 all-reduces; est. in-layer 17.1 ms + lm_head/sampler ≈ 0.15 +
  host 1–3 ms ≈ 20.9 measured.

## 3. Structural findings

1. **Plow already wins the collective (2.44 vs 5.2 ms) and MLA (1.37 vs 3.0).** The 4.4 ms
   gap is dense GEMV latency shape (+1.7), MoE routing + segment boundaries (+4.1), AttnRes
   (+2.7), KDA chain (+2.1), minus Plow's wins (−4.3), minus vLLM's host tax (−1..−2).
2. The decode is a **serial packet chain at 1 WG/CU** (147 KB LDS arena, `amdgpu_waves_per_eu(2,2)`,
   `interp.hip:4192`): wide packets never overlap; only b ≤ ~40 packets hide under a
   neighbour. Every packet removed from the chain saves its whole body + ~1.4 µs gate.
3. Per-packet fixed latency dominates the bodies: GEMV b=256 per-WG body min 8.3 / median
   9.1 µs for 86 KB (`k3_trace_wg`); end spread 7.4 µs of which ~4.1 µs is the XCD ramp
   (per-XCD t_end offsets 4.1–4.3 µs, `k3_trace_wg` per-XCD table).
4. Segment boundaries are AQL barrier-bit hand-offs on the CP (`amd.rs:11658`, decode all
   segments enqueued then one drain, `amd_tp.rs:927`): ~7.7 µs each. The grouped-MoE route
   is **two raw launches** (`enqueue_decode_segment`, `amd.rs:11423`: GLU then DOWN) → 3
   boundaries/layer; the MLA specialist adds 2 per MLA layer. 324 boundaries ≈ 2.5 ms/token.
5. The tunedb handoff constant (10.3 µs, `crates/tunedb/src/moe_decode.rs:22`) undercounts:
   the in-network gap is 38–41 µs vs a 16.8 µs isolated pair → ~23 µs of hand-off per layer.

## 4. Ranked levers (ms saved per engineering day, bit-exact first)

| # | lever | mechanism | evidence | Δ ms/token | days | ms/day | exactness | prereq | gate |
|---|---|---|---|---:|---:|---:|---|---|---|
| L1 | Router top-k body | replace k=16 rounds of `block_max_u64` with one LDS radix/bitonic select over 896 keys (same key encoding, same winners, same gate arithmetic) | `op_moe.h:378-470`; path body 22.9–23.1 µs × 92 = 2.48 ms | −1.4..−1.6 | 1.5 | ~1.0 | bit-exact | none | 3 alternating 8192→256 folds, checksum identical, TPOT ≤ −1.2 |
| L2 | Hoist independent GEMVs out of serial chains (emit order) | place f_a/b_proj before GemvQkvg, kv_a/k_rope_down before q_a, and g_proj (#72) into the pre-MLA segment; ASAP ties currently keep instruction order | `critpath`: KDA_CONV3 gate 18.8 µs/layer (gated by QKVG @4), MLA #66 gate 28.6, #72 gate 10.5 | −0.9..−1.2 | 1 | ~1.0 | bit-exact by construction | `packet::devbuild` ASAP tie-break by fan-in depth | exact folds; per-layer critpath shows CONV3 gate ≤ 4 µs |
| L3 | Fold f_b_proj (N=1536, K=128, 393 KB) into KdaStateStepG prologue | each of 192 step WGs computes its head's 128 gate values from f_a with the GEMV's per-column reduction routine | disasm #41→#43→#45 chain; f_b body 9.3 + gate | −0.6..−0.7 | 2 | 0.3 | bit-exact if `gemv_rows` column routine is reused | L2 (f_a first) | exact folds, KDA family −10 µs/layer |
| L4 | MoeCombine folded into the b=7 XReduce publish | the 7 XReduce WGs sum the 16 f32 slot partials per element in MoeCombine's fixed order before publishing; removes 1 packet/layer (8.4 µs body + 2 µs gate) | disasm #55/#56; `op_collective.h:527` publish-by-copy already reads a partial | −0.5..−0.8 | 2.5 | 0.25 | bit-exact (fixed slot order kept) — the C2 tree is NOT needed | none | exact folds; XREDUCE b=7 body ≤ 11 µs |
| L5 | AttnRes on 8–16 WGs with an in-packet rendezvous, then gang the preceding b=14 XReduce into the same packet | column bands per WG, partial (Σx², Σx·w) per row to global, last-arriver election (HIER pattern) → probs broadcast → banded mix + norm; phase 2 = D6 gang (XReduce 14 WGs feed AttnRes bands directly) | `op_k3.h:26-64` "only lever at T=1 is the body of one WG"; body 17.1 µs × 187 = 3.20 ms; 250 KB/site through one CU | −2.0..−2.4 (+−0.3 gang) | 4 (+3) | 0.55 | **C3 contract** (reduction order changes; same contract as the promoted f32-mix object; seam relL2 ≤ 2e-7 vs CPU port) | f32-mix object as the reference | GSM8K n=200 parity, seam oracle, TPOT ≤ −1.8 |
| L6 | One GLU+DOWN kernel, dataflow-gated (no grid barrier) | DOWN tiles poll per-expert-tile arrival counters; in-order claim, grid ≤ resident capacity (the closed cooperative variant used a grid barrier at >1 WG/CU) | 3 boundaries/layer × 7.7 µs; `amd.rs:11423`; `moe_decode.rs:22` | −0.6..−0.8 | 4 | 0.18 | bit-exact | none | exact folds; MOE_COMBINE gate ≤ 32 µs |
| L7 | Fused KDA decode arm inside the interpreter (conv3+step+norm in one packet) | port `kda_decode_fused.hip` (6.9 µs B1) as an interpreter arm; removes 2 packets + 2 gates/layer | ctl: conv3 6.8 + step 8.3 + norm 4.7 + 3 gates ≈ 24 µs vs 6.9 | −0.9..−1.1 | 6 | 0.17 | needs proof: fused kernel uses DPP row sums — exact only if it reuses the interpreter's step/norm reduction order; else C2-style contract | VGPR budget (ordinary object 248) | exact folds or oracle; KDA family ≤ 12 µs/layer |
| L8 | Cross-packet weight prefetch (claim-ahead + LDS ring) | thread 0 claims the next slice at body start; first UN weight chunks of the next GEMV issued via `GV_DMA` ring before polling its gate; hides ~2 µs DRAM latency/packet | GEMV per-WG fixed ≈ 4.5 µs of 9.1 (`k3_trace_wg`); 730 path GEMVs | −1.0..−1.5 | 7 | 0.18 | bit-exact (data-only) | `GV_DMA` ring on gfx950 (`op_gemm.h:3931` default 0), LDS budget under the 147 KB arena | exact folds; GEMV b=256 body/pk ≤ 8.5 µs |
| L9 | `sc1` scoped producer stores (D4 second half) | drop the per-packet L2 writeback in the release; HIER leader keeps the acquire | plan v3 D4; path gate p50 1.37 µs × 1,377 | −0.5..−0.8 | 5 | 0.13 | bit-exact; stale-word oracle over 1000 tokens | none | exact folds + oracle |
| L10 | Tagged publish from producer epilogue | GEMV/Combine epilogue writes the 8-B tagged words directly; removes the copy pass | `op_collective.h:527` note (~0.9 µs/collective) | −0.2..−0.3 | 2 | 0.12 | bit-exact | L4 (Combine epilogue) | exact folds |

Not levers (measured or structural): MALL residency (216 MB/layer, no reuse); same-input
two-output GEMV fusion (closed, GemvQkv Nv=0 +0.097); GEMV occupancy 3 (closed, LDS-pinned);
sharding routed_expert_down_proj (all-gather costs more than the 6 µs it saves); porting AITER
AR (Plow already 2.1× faster at 14 KiB); phase-chain AQL replay (host-only); fewer decode
segments by merging (the boundaries are the raw MoE/MLA objects, not the interpreter).

## 5. Bottom line

| stack | Δ ms | TPOT (served, host 0.44 kept) | days |
|---|---:|---:|---:|
| now | — | 25.25 | — |
| exact, cheap: L1 + L2 + L3 + L4 | −3.4..−4.3 | 21.0–21.9 | 7 |
| + L6 + L10 (exact) | −4.2..−5.4 | 19.9–21.1 | 13 |
| + L7 + L8 + L9 (exact, harder, 70% yield assumed) | −6.0..−7.8 | **17.5–19.3** | 31 |
| L1–L4 + **L5** (C3 contract) | −5.4..−7.0 | **18.3–19.9** | 14 |
| L1–L6 + L5 + L10 | −6.5..−8.4 | **16.9–18.8** | 20 |

- **≤ 18.8 with margin needs L5** (AttnRes) plus the exact set L1–L4/L6/L10: −6.5..−8.4 ms
  in ~20 engineering days; L7–L9 are the safety margin (another −2.4..−3.4 at full yield).
- **Exact-only** (no contract change) realistically lands at 19–20 ms after L1–L4/L6/L10 and
  reaches 17.5–19.3 only if L7/L8/L9 all yield; the C3-contract AttnRes rewrite is the single
  largest and cheapest item and is the same contract the promoted f32-mix object already uses.
- Order: week 1 L2 → L1 → L4 (all exact, 5 days, ~−3.5 ms, each a 3-fold gate); week 2 L5
  phase 1 + L3; week 3 L6 + L10 + L5 gang; then L8/L7/L9 as margin.
- Every promotion: 3 order-alternated 8192→256 folds vs same-source control, checksum identity
  (or the C3 seam oracle + GSM8K for L5), cleaned all-rank trace re-attributed with
  `critpath_layers.py`, rollback flag, dated `perf-data/` row in the campaign summary.
