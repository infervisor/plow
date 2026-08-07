# Gemma-4-12B prefill vs vLLM on GH200 — campaign round 1 (2026-08-06)

Goal (user): beat vLLM prefill for the Gemma-4 family, bf16 AND fp8, apples-to-apples.

Box: GH200 480GB (132 SM, 900 W, 1980 MHz). Method: `vllm bench serve`
(vLLM 0.26.0) as the ONE client for every arm — `--backend openai-chat`,
random dataset, `--random-output-len 8`, `--num-prompts 6 --max-concurrency 1
--seed 42`; TTFT median of the cell. plow arms served by this branch's plowrt
on :8090, one at a time; vLLM bf16 is the user's server on :8081
(`--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.50`);
vLLM fp8 a dedicated `--quantization fp8` server on :8082, mem-util 0.30.

CAVEAT: the vLLM bf16 server (52 GiB) stayed RESIDENT (idle) through every
plow cell and gpulease flagged all runs rc=76. Its own cells ran while plow
was not serving. Compute contention ~none (0% util), but re-run clean before
publishing absolutes; ratios plow-vs-plow are solid.

## TTFT (median ms), 2026-08-06 evening, single round per cell

| in_tok | vLLM bf16 | vLLM fp8 | plow bf16 stock | plow bf16 +TMA | plow fp8 shipped (w8a16) | plow fp8 w8a8 | plow fp8 w8a8+GLU | plow fp8 w8a8+GLU+TMA |
|---|---|---|---|---|---|---|---|---|
| 128  | 27.2 | 27.4 | 83.0 | **73.6** | 136.0 | 95.8 | 72.3 | **65.1** |
| 1024 | 63.9 | 57.9 | 208.2 | **185.0** | 348.7 | 228.4 | 174.0 | **155.3** |
| 4096 | 200.0 | 175.5 | 750.4 | **706.9** | 1192.7* | 841.7 | 643.0 | **588.9** |

## Rounds 2-4 (same method; all arms correctness-gated)

| lever | @128 | @1024 | @4096 | verdict |
|---|---|---|---|---|
| chunk 4096 (`PLOW_MAX_CHUNK=4096`, single-chunk 4k) | = | = | bf16 707->569, fp8 589->472 | **-20% @4k** (4x fewer per-chunk packet+launch costs; KV +1.9 GiB) |
| GLU TMA ring, bf16 | 65.8 | 167.0 | **512.7** | **-10%** — rescues the GLU the cp.async fork lost 3% on; token-identical |
| GLU TMA ring, w8a8 | 61.2 | 145.1 | **445.9** | -5.5%; 3/4 prompts identical, 1 coherent alternative (fp8 tie jitter) |
| L2-domain placement (PLOW_NV_PLACE, 8x18 SM) | 61.4 | 145.7 | 446.6 | **NULL** (validated on HW for the first time; 3 loader fixes) |
| counter V3 (acquire-hoist) | 61.1 | 146.0 | 448.0 | **NULL** |
| GATE_SLEEP=0 | 61.6 | 146.0 | 447.0 | **NULL** |
| TMA ring depth NS=5 / NS=6 | — | — | 449.5 / 457.5 | **worse** — mainloop is math-bound at occ-1, not latency-bound |
| fp8 promote-128 off | — | 146.4 | 447.9 | **NULL** — promotion is FREE; stays on for accuracy |
| flash-prefill TMA K/V (hd256) | 61.1 | 144.9 | bf16 507.1, fp8 **442.6** | **~-1%** — the arm was already wgmma; staging was not its bottleneck |

Ablation attribution (w8a8 base, chunk 4096, @1k/@4k): fp8 GEMM family
93/299 ms (**67%**, ~288 TF/s effective), flash prefill 22/107 ms (24%),
everything else ~30/40 ms.

Cumulative: fp8 prefill **2.7x** (1193 -> 442.6 @4k), bf16 **1.48x**
(750 -> 507.1). Gap to vLLM: fp8 2.2x/2.5x/2.5x, bf16 2.4x/2.6x/2.5x.

## Per-op trace (PLOW_NV_TRACE prefill build + PLOW_PF_TRACE_LOG=1, now wired on CUDA)

One 4k chunk, block-0 cycles (read the SHAPE, not absolutes): gate 10.8% /
body 88.6% / signal 0.5%. Body by op: GEMM_GLU_FP8 25.5%, FlashPrefill 21%,
GEMM_FP8 18%, **QuantFp8 14.4%**, HeadNormRope+NormResidual+RmsNorm ~9%.

FUSION AUDIT this exposes (missed until the trace):
- **QuantFp8 is a whole extra activation read+write pass, 4 packets/layer.**
  The 2 norm-fed sites (qkv-in after RmsNorm, mlp-in after NormResidual) can
  emit e4m3+row-scale from the norm's existing row pass (absmax rides the rms
  reduction). The 2 tiled-producer sites (attn-out, glu-out) need a cross-tile
  row absmax and stay separate for now. Worth ~7% of TTFT.
- **Prefill runs the sandwich norms UNFUSED** (NormResidual + RmsNorm as two
  packets/passes; decode has NORM_RESIDUAL_NORM). ~2-3%.
- q/k/v are three GEMM packets (decode fuses GEMV_QKV); at M=4096 the win is
  packet count + A-reuse, minor.
- Occ-2 whole-object REFUTED (1250 vs 442.6 — capped-register spills demolish
  the wgmma arms); occ-2 decode side-finding: ~8% TPOT win.

## Segmented multi-module prefill: launcher LANDED, occ-2 locked behind one bug

The T9c host launcher now works on CUDA (PLOW_PF_SEG_DIR + PLOW_BUILD_SEG=1;
one cooperative launch per wave-class segment, alternating the fat _pfseg and
lean _pfgemm objects; stream order satisfies cross-segment gates). Measured
chain @4k (w8a8, coherent-output gates):
  monolithic best 442.6 | seg fat-only 450.5 (+1.8% = the 97 launches are
  CHEAP) | lean uniform maxnreg(128) NS=3 469.2 (128-reg spills) | lean
  uniform launch_bounds NS=2 574.9 | lean WARP-SPEC setmaxnreg: DEADLOCK
  (in-model only; the standalone probe shape passes; gated PLOW_NV_SEG_WS).
RESOLVED BY ANALYSIS (SASS audit of the failing object):
- H1 (likely hang): the PER-OP dec/inc register-restore cycle — the probe and
  CUTLASS split ONCE per launch and never restore; the PTX spec is silent on
  repeated cycles and the producer's restore TRY_ALLOC spin is exactly a
  100%-util park. Standalone repro of every OTHER ingredient passes.
- H2 (decisive regardless): inside the megakernel TU, ptxas grants the
  consumer 224 regs but still spills all 128 accumulators (263 STL/LDL,
  C7515 per-QGMMA serialization) — the probe's zero-spill compilation does
  NOT reproduce with the interp arms in the same function. Even hang-fixed,
  this path cannot reach 391 TF/s.
=> ARCHITECTURE CONCLUSION: the lean GEMM segment must be a DEDICATED KERNEL
   (probe-shaped TU: the ws mainloop + a GQ-lite claim loop over GEMM packets
   only, split once at kernel entry), with norms/quant/rope reclassified to
   the fat object's segments at emit. That is the buildable form of the occ-2
   2x on the 43-67% GEMM share.

## Why the next 2.5x needs an architecture step, not another knob

Every probe-validated lever is now integrated and measured. The GEMM mainloop
runs at its occ-1 ceiling (in-model ~288 TF/s ~= the probe's occ-1 rows; the
probe's 460-500 TF/s rows all needed 2 blocks/SM via the entry-register clamp).
The remaining plan, in order:
1. **Lean GEMM prefill segment at occ-2** (T10 machinery: PLOW_NV_SEG_GEMM,
   segment objects, 100 KiB arena cap -> NS=2 TMA ring fits): 2x math
   warpgroups/SM on the 67% GEMM share.
2. BN=256 tiles on the occ-2 object (B-traffic halves; probe 128x256 rows).
3. hd512 (full-layer) flash: still px4 mma.sync + cp.async — the last
   non-wgmma prefill arm.
4. Small-T floor (~60 ms @128 = many tiny bodies): op fusion, not protocol
   (V3/GATE_SLEEP nulls prove the gate is not it).

*shipped-fp8 4k from the earlier self-harness run (bench_speed.sh, not this
client) — direction/scale only.

Correctness: every plow arm gated on 4 greedy prompts x 32 tokens.
bf16+TMA is TOKEN-IDENTICAL to stock bf16. w8a8 and w8a8+GLU are
TOKEN-IDENTICAL to the shipped w8a16 arm.

## What moved and why

- **fp8 w8a8 (PLOW_BUILD_W8A8=1 cubin + PLOW_W8A8=1 emit)**: the shipped fp8
  prefill was w8a16 dequant into Hopper's EMULATED mma.sync.e4m3 (12xF2FP+2xHMMA).
  QGMMA + on-device activation quant + promote-128: **1.42-1.53x**.
- **+ GLU wgmma fork (PGM90_FORK_GLU=1, folded into PLOW_BUILD_W8A8)**: the FFN
  GLU was still emulated-fp8; the fork takes the fp8 arm to **1.86x total**
  (1193 -> 643 @4k) and past plow bf16. NOTE the bf16 GLU fork measured a ~3%
  LOSS (773 vs 750) — the spill-pollution the op_gemm.cuh gate note predicted —
  so it stays default-0; only the w8a8 build turns it on.
- **bf16 TMA GEMM (PLOW_BUILD_TMA_GEMM=1 + PLOW_TMA_GEMM=1 emit)**: -6..-11%.
  Two integration findings recorded in the kernel commit: warp-spec halves the
  math warpgroups at 256 threads (1.8x LOSS; uniform-TMA is the right shape),
  and __noinline__ serializes wgmma across the ABI boundary (uniform 1.8x LOSS
  on every packet). In-tree standalone A/B (`tma_uni_gemm_ab.cu`): TMA 0.83x
  time at every 12B chunk shape (274 -> 328 TF/s q_proj chunk).

## Where the remaining gap lives (bf16 707 vs 200 = 3.5x)

Effective E2E rate: plow ~122 TF/s vs vLLM ~450 on ~86 TFLOP of prefill.
Rough decomposition at 4k (from deltas + probe rates): plain GEMM ~260 ms at
~330 TF/s (near TMA-kernel rate — limited headroom), GLU ~230 ms on mma.sync
(bf16 wgmma fork loses to spills; a TMA GLU ring is the fix that changes the
tradeoff), flash prefill ~150 ms on the 2.8x-slower-than-TMA-probe path,
fixed/protocol ~70 ms (TTFT@128 floor vs vLLM's 27 TOTAL).
Chunking is NOT the lever: chunk 8192-vs-1024 measured 2% on sm120 (devgen
`default_chunk` note) and weight re-reads at 4 chunks cost ~32 ms of BW here.

Next levers, in expected-value order:
1. TMA flash prefill (hd256 arm): probe 2.6-2.8x on the attention share.
2. w8a8 TMA twin (plain + GLU rings): fp8 GEMM 342 -> ~500 TF/s class.
3. bf16 GLU via TMA ring (2-stage (A,Bg,Bu)): revisits the spill tradeoff with
   ~2048 fewer cp.async per stage.
4. Prefill trace on CUDA serve (the a82d2e3 fix was AMD-only) for exact
   per-op attribution of the ~70 ms fixed floor.

## Repro

Cubins: `PLOW_BUILD_W8A8=1` and/or `PLOW_BUILD_TMA_GEMM=1
scripts/build_sm90a_cubin.sh <out.cubin>` (PLOW_ROOT=<tree>).
Packets: `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33 [PLOW_W8A8=1|PLOW_TMA_GEMM=1]
plowc --hf-dir <snapshot> --emit devblob --arch sm_90a --gpu h100
--max-ctx 8192 --out <dir>`. fp8 bundles need the quantize_fp8.py twin
linked as `checkpoint/fp8-model.safetensors`.

## 32k-context round (2026-08-07)

Bundles re-emitted at `--max-ctx 32768`, chunk 4096 (KV 3.0 GiB). vLLM 32k
baselines are DEDICATED servers on :8083 squeezed beside the resident 8k
server: `--gpu-memory-utilization 0.34 --kv-cache-memory 5.8e9
--max-num-batched-tokens 4096 --enforce-eager` + expandable_segments (bf16
OOM'd without) — i.e. a chunked, eager, memory-tight vLLM; an unconstrained
vLLM would be somewhat faster. TTFT median ms, NPROMPT=3:

| in_tok | vLLM bf16 | vLLM fp8 | plow bf16 | plow w8a8 | gap (fp8) |
|---|---|---|---|---|---|
| 8192  | 617  | 553  | 1112 | 973  | 1.76x |
| 16384 | 1350 | 807* | 2606 | 2363 | 2.9x* |
| 32000 | 3289 | 3009 | 6923 | 6408 | 2.13x |

(*fp8@16k median noisy: mean 939.) plow @32k matches the context-scaling
memo's calibrated prediction (~6.5 s), confirming the hd512 px4 arm runs
~40 TF/s here (not the H100-NVL 15-20) and is 55-73% of long-ctx TTFT.

**hd512 wgmma arm <512,64,16> (PLOW_NV_FA512_WG): REFUTED as instantiated.**
After an arena-sizing fix (PLOW_NV_PRE_A must track the dispatched triple —
first run died ILLEGAL_ADDRESS writing 131 KiB into the old claim), it runs
coherently but 772.6 @4k / 15763 @32k — 1.75-2.5x WORSE than px4: n16 score
tiles x 32 k-steps are issue-bound and the 128-reg O accumulator hits the
same megakernel-TU spill regime the ws-GEMM SASS audit found (263 spills,
per-QGMMA serialization). Flag stays 0; the code is the skeleton for the
dedicated-kernel version.

CONVERGED ARCHITECTURE VERDICT (two independent refutations now agree):
heavyweight wgmma bodies — the occ-2 warp-spec GEMM AND an hd512 flash —
cannot be compiled inside the megakernel translation unit without losing
their probe-grade register allocation. The path to the remaining 1.8-2.5x
is DEDICATED per-class kernels (probe-shaped TUs with a GQ-lite claim loop)
driven by the now-landed segmented launcher.

## T11-T20: the dedicated-kernel round (2026-08-07, second session)

The converged verdict executed. Everything below is gpulease-measured TTFT
median @4096 tok on the 12B, same 4-prompt greedy gate per step.

### The chain (fp8 w8a8)

| step | @4k ms | what landed |
|---|---|---|
| single-launch champion (prev) | 446 | GLU-TMA ring, uniform w8a8 |
| T11 ws-entry | 436 | ONCE-per-launch setmaxnreg split + fully DIVERGENT role loops (the per-op dec/inc cycle was the deadlock; a reconverging entry split is dropped by ptxas C7507). Pure-fp8 classing (PLOW_SEG_PURE_GEMM=fp8 / PLOW_PF_SEG_PURE=fp8). |
| T11 quant-into-norm | 411 | RmsNorm t3/t4 = fused xq/ascale (PLOW_QNORM_FUSE=1); token-identical (quantizes the bf16-rounded value). |
| T11 GLU-into-quant + v8 quant | 361 | QuantFp8 t3/t4/i2 = fused GLU producer; d_quant_fp8 vectorized (scalar loops, not bandwidth, were the 29% share). |
| T12 FA object | 323 | hd512 <512,64,16> wgmma arm — REFUTED in the fat TU — WINS in a flash-only TU (*_pffa, class-2 segments, 251 regs / 0 spills). Same TU-isolation law as the GEMM. |
| T12b 'all' | 318 | hd256 sliding flash also on the FA object (PLOW_SEG_FA512=all). |
| T13 band raster | 307 | sm90_tile_remap (BAND=16): concurrent blocks share B tiles via L2 (the ws GEMM measured exactly the 128 FLOP/B no-reuse roofline). |
| T14 FATLITE | 301.7 | fat object arm-stripped + 128-reg cap + occ-2 (mostly neutral; PGM90_TMA_STAGES=3 arena fix mattered). |
| T15 uni256 | 284.1 | UNIFORM m128n256 occ-1 body (wgmma_m64n256k32, 128 acc/thread, NS=4 x 48 KiB ring): both n128 bodies were SMEM-BW bound at ~118 B/cyc; m128n256 needs ~88. GEMM class 156.7 -> 136.8 ms. |
| T18 uniseg tail | 274.7 | small buckets emit ONE segment (PLOW_UNISEG_MAX_T=512): the ~50-token tail chunk paid ~480 launches (~40 ms) for ~5 ms of work. |

vLLM fp8 @4k = 175.5 → gap 1.57x (was 2.5x at the last report, 6.8x at session start).

bf16 @4k: 507.1 → 388.1 on the same architecture with zero new bf16 kernels
(T20: occ-1 lean object PLOW_BUILD_GEMM_OCC1, PLOW_SEG_PURE_GEMM=1 classing).
vLLM bf16 200 → gap 1.94x.

### Steady-state attribution @4k fp8 (PLOW_PF_SEG_TIME per-class CUDA events)

GEMM class 136.8 ms (192 launches, ~640 TF/s vs 1979 peak) | fat 35.1 (241
launches, light rows) | FA 54.0 (48 launches, ~8 TF/s effective — KV re-read
bound) | tail chunk ~10 | client ~13. First chunk after server start pays a
one-time ~57 ms warmup (seg0).

### Refuted this round (all hardware-measured)

- BN256 in the ws-entry object: ptxas C7602 — a 128-acc wgmma cannot compile
  under the 128-reg entry cap (kept as skeleton).
- classing v2 (rope/merge→FA, quant→GEMM): merged launches < occ-1 light-body
  loss (296 vs 284).
- PLOW_PF_SEG_NONCOOP: identical — the ~60-90us launch floor is kernel
  entry/drain, not the cooperative API.
- PLOW_PF_SEG_EQSMEM (uniform smem request to dodge carveout reconfig): 299.
- hd512 BKV=32 (PLOW_NV_FA512_BKV): neutral — FA time is KV-locality, not
  score-tile shape.
- flash head-major work order: 289 — head-fastest wins on causal load balance.
- NS=3 / BAND=32 sweeps on uni256: within noise.

### Remaining path (in expected-yield order)

1. GEMM 137→~70: cluster-pair TMA multicast (halves B traffic; needs
   cuLaunchKernelEx cluster plumbing) or bf16/fp8 smem-side double-issue;
   at 640 TF/s the body still leaves 2x on the table vs CUTLASS.
2. FA 54→~25: GQA-aware KV sharing (one KV window read feeds all q-heads —
   needs an FA3-style restructure of the wgmma arm; the enumeration-order
   probe showed L2 alone cannot absorb the re-read).
3. bf16 GEMM: m128n256k16 uniform twin of T15 (same smem-wall math).
4. fat 35: ~60% is per-launch floor at 241 launches; per-layer packet fusion
   (NR+norm+quant riding one packet) is the only lever the classing allows.

## T21-T24 continuation (same session)

| step | fp8 @4k | bf16 @4k | what |
|---|---|---|---|
| T21 sliding BKV=64 | 269.7 | 345.9 | n64 score wgmma halves per-KV-tile barrier+drain count (two-box KV TMA; shared mma.sync body constexpr-guarded out of wgmma shapes) |
| T21b hd512 BKV=32 | 260.5 | 336.4 | stacks with T21 (canonical: -DPLOW_NV_FA256_BKV=64 -DPLOW_NV_FA512_BKV=32) |
| T22 vectorized epilogue | 240.3 | 316.0 | uni256 did 128 scalar 2B stores/thread; bf16x2 pairs lift the standalone probe to 947/978/1174 TF/s (q/gate/down). New probe: experiments/uni256_probe.cu |
| T24 lm_head reclass | 239.7 | 316.2 | mapped bf16 GEMMs class 8 in fp8 mode (lm_head off the spilled fat body) |

REFUTED: sm90a M=1 lm_head GEMV arm (corrupt first token, zero delta);
kv-proj n128 fallback (probe said 2.7x win solo, -8ms in-model — kv shares its
segment with q on disjoint CU sets; standalone probes mislead on co-scheduled ops).

**FINAL this round: fp8 239.7 ms (vLLM 175.5, gap 1.37x) | bf16 316 ms
(vLLM 200, gap 1.58x) | 1024-tok medians 101 / 121 ms.**
Session total: fp8 446→240 (1.86x), bf16 507→316 (1.60x).

Remaining program (unchanged priorities): FA warp-specialized rewrite (the two
per-tile wgmma drains + redundant-S are ~55% overhead at ~830 TC-cycles/tile);
GEMM smem-staged epilogue + cluster multicast; fat per-layer packet fusion.

## T25-T27 addendum + closing budget (2026-08-07)

- PLOW_SEG_CLASS_SLICE=light: neutral (per-entry overhead is not a factor).
- sudo ncu now works on this box (perf counters via sudo): bf16 uni256 = 67%
  tensor-pipe, 0 spills, no clock throttling (1980 MHz flat, 470 W peak).
- REFUTED: 2-deep wgmma window (issuer deadlock at lead NS-1 — the elected
  issuer in wg0 blocks on an arrival only wg0's own next stage can make; -4%
  at lead NS-2); split dual-n128 acc chains (-7% — HW pipelines the single
  n256 chain better).
- CLOSING BUDGET @4k fp8 (240 ms): GEMM 120 (matches the probe blend exactly —
  no hidden in-model loss), fat ~30, FA 38, tail ~10, client 13. The last 65 ms
  to vLLM 175.5 = FA3-lite flash (-14) + GEMM TC 67→85% (-20) + fat fusion (-8)
  + launch amortization via cuGraph (-8) + margin. Designs in
  plans/gh200-kernel-review.md "T28 design notes".

**FINAL: fp8 240.1 ms (1.37x) | bf16 316.3 ms (1.58x) @4k.**

## T31-T36: the 384-thread round + UNCONTENDED SHOWDOWN (2026-08-07, cont.)

The foreign servers exited mid-session — every number below is from an
otherwise-idle GH200, all four arms benched sequentially with the same
vllm-bench-serve client (NPROMPT=9, median TTFT ms):

| arm | 1024 tok | 4096 tok |
|---|---|---|
| plow fp8 (w8a8) | 69.9 | **175.9** |
| vLLM fp8 | 63.5 | 173.9 |
| plow bf16 | 82.6 | 214.2 |
| vLLM bf16 | 68.8 | 198.7 |

fp8 @4k = 1.011x — STATISTICAL PARITY (repeat runs 176-179 vs 174). bf16 1.08x.
Session start was 6.8x / 2.5x.

### What landed
- T31 (the decisive step): the segmented launcher reads a per-object BLOCK SIZE
  (plow_block_pfgemm global) and the lean GEMM object builds at 384 THREADS —
  wg0 dedicated TMA producer (entry setmaxnreg 32), wg1/wg2 224-reg consumers,
  one m64n256 slab each (the cuBLAS shape; cuBLASLt measures 1324-1468 TF/s fp8
  / 804-861 bf16 at these shapes). fp8 239->207, bf16 316->251. GEMM class
  120->89.5 ms (~1145 TF/s in-model, at probe parity).
- T32: overhang ladder rungs (PLOW_PF_LADDER_APPEND=640,1152,2176,4224) — the
  chat template's +14 rows forced a second full-model tail pass (~30-36 ms,
  found via the new CUDA-arm §TTFT breakdown: TTFT = tokenize 10.8 + prefill,
  nothing else). fp8 207->177, bf16 251->215.
- T35: CUDA-graph segment chains (cuGraph* in the dlopen table; 483-node serial
  graphs cached per bucket x slot-table) — ~1 ms.
- T33: hd512 TMA KV staging (mint gate widened; neutral, kept).

### Refuted this round (all correct, all measured)
- cluster-pair TMA multicast (0 mismatches; -16..-37% — rank-0 issuer serializes
  B for two CTAs; the L2-service-wall theory died with it)
- smem-staged epilogue (+15% standalone probe, -3ms in-model: probe C-residency
  artifact — PROBES OVERSTATE; in-model is the only ground truth)
- 2-deep wgmma window (issuer deadlock at lead NS-1, root-caused; -4% fixed)
- split dual-n128 acc chains (-7%)
- wgitem flash (wg-per-item, token-identical, neutral at equal BKV — the BKV
  wins were staging granularity, not barriers)
- T34 bf16 256B k-stages at NS=2 (-44ms: ring starvation)
- T36 quant on ws384 consumers (saves 2 launches/layer, halves quant's block
  parallelism: +8.4ms)
- flash head-major order, NS/band sweeps, eq-smem, noncoop, GEMV lm_head (bug).

### Remaining to WIN both
bf16 is the blocker (1.08x): its GEMM runs ~700 TF/s vs cuBLAS 830 — the next
lever is a bf16-specific mainloop (RS-form A-in-registers or acc double-buffer,
CUTLASS-grade), worth ~15-18 ms. fp8 @1k (1.10x) and bf16 @1k (1.20x) are
launch-floor + tokenize bound. Canonical config: cubin-ws384 + t32b/bf16-t32
bundles, serve PLOW_PF_SEG_DIR + PURE=(fp8|1) + FA512=all + SEG_GRAPH=1.

## T37: generality audit + 31B proof + tuner verdict (2026-08-07)

**The stack is model-agnostic.** Audit fixes (all landed):
- pure-GEMM classing (both modes) now requires the TMA maps — an unmapped GEMM
  from any other emitter/model falls to the fat object's cp.async fallback
  instead of trapping in the lean object.
- PLOW_SEG_FA512=all claims only head dims the *_pffa object instantiates
  (256/512); hd128 models (Qwen/Llama) keep flash on the fat object.
- odd-N trap guards on the paired-store n256 epilogues (fail loud, bad_k style).
- seg_classes cap 512 -> 2048 (603 segments on 60-layer models).

**Proof — Gemma-4-31B (hidden 5376, inter 21504, 60 layers, 32 heads) on the
UNMODIFIED canonical stack** (same cubins, same knobs): gate coherent, TTFT
medians 148.2 ms @1k / 526.7 ms @4k bf16. Every dim divides the new tiles
(N%256, K%128) — as do Qwen3/Llama dims by inspection; hd128 attention falls
back safely. 12B fp8 regression after the fixes: 176.7 (unchanged).

**plowc tuning: does NOT currently help the sm_90a stack.** `pick_tile`
short-circuits on NVIDIA (`nvidia_prefill_gemm_op` returns one canonical opcode;
tile geometry is fixed by cubin -D macros), and `plowc tune gemm` is a
gfx950-only campaign (builds via hipcc, measures the AMD object). The tunedb /
CompilerOracle machinery only prices AMD tiles. If per-shape choices ever matter
on NVIDIA (n128-vs-n256 by (M,N,K), BKV per hd, ws384-vs-uniform), the oracle is
the right home — but today the knobs are build/env-level and nothing on the
NVIDIA path consults measurements.

Bring-up methodology codified for future networks: perf-data/harness/BRINGUP.md
(+ bringup_gate/bench/showdown/ceiling scripts).

## Main-merge consolidation (2026-08-07)

Merged origin/main (VMM weight-slab, NRN2 fold, gfx942 kernels) into the
branch; conflicts hand-resolved (devgen keeps qnorm_fuse + PLOW_PF_GFUSE beside
NRN2/fold_proj; gpu.rs takes main's slab upload with our SegPf/graph work
intact). Workspace tests green (slots table synced to the fused RmsNorm/QuantFp8
doc comments; tunedb gemv sweep-path fixed runtime/ubench -> runtime/bench/gemm,
broken on main too). Branch is strictly ahead of origin/main — merge to main is
a clean fast-forward.

Post-merge GPU regression: fp8 gate coherent, bf16 gate coherent; TTFT @4k
179.2 ms median vs 179.05 on the pre-merge binary rebenched the same day —
merge is perf-neutral (the 175.9 record was day-to-day drift). VMM slab vs
Flat vs PerTensor: identical TTFT (upload path does not touch steady-state
prefill).

Footgun for the record: any `cargo build`/`cargo test` without
`--features cuda` clobbers target/release/plowrt with a CUDA-less binary that
silently serves via the CPU reference backend + byte-fallback tokenizer —
deterministic garbage that masquerades as a kernel bug. Check the serve log
header (`cuda=true`) before believing a correctness regression.

