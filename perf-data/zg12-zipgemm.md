# ZG-1 + ZG-2 — fused register-direct ZipGEMM (`k_zipgemm`)

Date 2026-07-22. Branch `zg12-zipgemm` (based on `zg0-tc-baseline`, 7bcc7cc).
Model gemma-4-12B, RTX PRO 6000 Blackwell (188 SM, HBM 1535 GB/s achievable).
System nvcc 13.0, `sm_120a`. Harness `runtime/tests/zg12_zipgemm_sm120.cu` (standalone;
`op_gemm.cuh`/interp UNCHANGED — default build byte-identical; ZG-3 wiring not done).

**Verdict: all three correctness gates PASS; PERF is a decisive NEGATIVE.** The fused
register-direct sz12 decompress does NOT hide under the weight loads at decode batch
sizes. `k_zipgemm` runs **0.31×–0.95×** of the ZG-0 tuned TC-bf16 baseline at every
shape × M (bar was ≥1.15×). It is **recon/issue-bound, not BW-bound** — the −25% weight
bytes are more than eaten by the exposed decompress. This is the ZG-2 KILL condition.
The C-1T structural finding stands even with (a) a correct-by-construction TCA
fragment-lane layout (coalesced per-lane loads) and (b) register-direct recon into the
mma B-fragment (no bf16 smem tile, one sync) — the two things C-1T got wrong.

## What was built

**ZG-1 — offline TCA packer + oracles.**
- Ported the validated sz12 codec (byte-plane split: `lo` = sign@7|mant7, `cd` = 4-bit
  affine exponent code `code = exp − 109`, per-tensor escape list) + V3 `recon_pair`
  from `origin/c1t-tensorcore-bsweep:runtime/tests/sz_decomp_sm120.cu`. Kept sz12 (NOT
  ZipServ's 3-bit bitmap), per the plan.
- **Offline reorder into tensor-core FRAGMENT-LANE order.** One shared derivation
  (`zg_lo_byte`/`zg_cd_byte`, `__host__ __device__`) used by BOTH the host packer AND
  the kernel → correct-by-construction. Derivation taken from plow's validated path
  (`op_gemm.cuh pgm_load_bfrags` :504-511): `n = lane>>2`, `b0 = {k=2·tig, +1}`,
  `b1 = {k=2·tig+8, +9}` (`tig = lane&3`). Per lane: **4 contiguous lo bytes** = one
  coalesced u32 (b0.lo|b1.lo), **2 contiguous cd bytes** = the two code-pairs. Within a
  `[BN][BK]` k-block, TensorCoreTiles (16k×8n) are ordered `tct = (kt/16)·(BN/8)+(nt/8)`;
  a lane's u32 sits at `tct·128 + lane·4` → consecutive lanes read consecutive words
  (128-byte coalesced transaction, no bank conflict).
- **Escapes:** per-row prefix `eoff[n]` + absolute `epos`/`eval`; patched register→register
  after recon (rare branch, hoisted eoff loads). Measured escape rate **0.015%** on
  realistic Gaussian(σ=0.02) weights (matches the c0 audit 0.018%), ratio **1.33×**.

**ZG-2 — fused `k_zipgemm`.** ZG-0 tuned TC skeleton (split-K, BK64, STAGES4, BM16/BN128
+ BM32/64/BN64), B path replaced: cp.async the compressed tile (lo+cd, 1.5 B/elem) into
an smem ring → each lane loads its own coalesced u32+u16 → `recon_pair` (V3) register→
register directly into the mma B-fragment `bf[nj][0..1]`. **No intermediate bf16 smem
tile, one `__syncthreads`.** A-operand via `ldmatrix.x4` as ZG-0.

## Gate results (all PASS — correctness is solid)

| gate | result |
|------|--------|
| **1. Host lossless byte-exact** | **PASS** — pack→unpack `memcmp==0` over all 4 shapes × {BN128,BN64}; neg-control (1 corrupted lo byte) → mismatch as required. escape 0.0147–0.0154%, ratio 1.3325–1.3326×. |
| **2. Fragment oracle** (recon(baked TCA) == `ldmatrix`(uncompressed), per lane) | **PASS** — 0 lane-frag mismatch over 4 random tiles (BN128/BK64, all warps/nj/kf/lane/reg incl. escape patch). Confirms the shared derivation matches actual `ldmatrix` output — the C-1T 35% misroute is structurally impossible here. |
| **3. Device bit-exact** (`k_zipgemm`(ksplit=1) == TC-bf16(ksplit=1)) | **PASS** — 0 byte mismatches at **every** shape × M ∈ {1,2,4,8,16,32,64} (28/28 cells). Lossless recon → identical bf16 B → identical f32 accumulation → byte-identical output. |

**ptxas (sm_120a, 0 spill everywhere, no 255 cliff):** `k_zipgemm` BM16 **40** regs /
BM32 **64** / BM64 **96**; `k_tc` baseline BM16 48 / BM32 70 / BM64 80.

## Perf — `k_zipgemm` vs ZG-0 tuned TC-bf16 (GB/s = logical bf16 bytes ÷ time; cold, L2-flushed)

| shape | M=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|-------|----:|--:|--:|--:|---:|---:|---:|
| qkv     | 0.822× | 0.864× | 0.861× | 0.864× | 0.853× | 0.854× | 0.563× |
| o_proj  | 0.805× | 0.832× | 0.827× | 0.871× | 0.863× | 0.925× | 0.659× |
| gate/up | 0.903× | 0.905× | 0.912× | 0.907× | 0.901× | 0.951× | 0.557× |
| down    | 0.479× | 0.479× | 0.479× | 0.482× | 0.481× | 0.517× | 0.314× |

**Best cell 0.95× (o_proj/gate M=32); worst 0.31× (down M=64). Never ≥1.0×, let alone
1.15×.** `down` (deepest K=15360) is worst by far. M=64 (BM64 config) collapses on every
shape. `zip_pct1535` peaks at 59% (gate/up) — well under the wall.

## Root cause — RECON-BOUND, not BW-bound (DIAG, cold L2-flushed)

The decisive diagnostic: zip's **actual** DRAM throughput (logical GB/s × 0.751
compressed fraction) vs the cold-read ceiling of the **same compressed footprint**
(`k_stream_reduce` over lo+cd bytes).

| shape | M | zip logical | zip actual-DRAM | bf16 cold-ceil | **comp cold-ceil** | zip_actual / comp_ceil |
|-------|--:|------------:|----------------:|---------------:|-------------------:|-----------------------:|
| qkv     | 8  | 765 | 574 | 959 | 922 | **62%** |
| qkv     | 64 | 482 | 362 | 959 | 922 | **39%** |
| o_proj  | 8  | 580 | 435 | 821 | 795 | **55%** |
| o_proj  | 64 | 414 | 310 | 821 | 795 | **39%** |
| gate/up | 8  | 898 | 674 | 1029 | 1023 | **66%** |
| gate/up | 64 | 521 | 391 | 1029 | 1023 | **38%** |
| down    | 8  | 470 | 353 | 1029 | 1024 | **34%** |
| down    | 64 | 289 | 217 | 1029 | 1024 | **21%** |

Two conclusions:
1. **The footprint-ramp penalty is negligible.** `comp cold-ceil ≈ bf16 cold-ceil`
   (within ~4%): the smaller 0.75× compressed footprint sits essentially at the same
   point on the cold latency ramp. So the slowdown is **not** "smaller footprint = lower
   BW" — the byte savings are genuinely available.
2. **The recon is on the critical path.** zip moves compressed bytes at only **21–66%**
   of the achievable rate for that footprint. The kernel is **recon/issue-bound**: the
   V3 decompress (~5.5 ops/elem) + per-fragment smem gathers + escape checks are executed
   by the same warps that must issue cp.async, so the weight stream is starved and DRAM
   is never saturated. Worse with K depth (more k-steps → more exposed recon per output:
   `down` 240 k-steps vs `qkv` 60) and with BM64 (recon a larger serial fraction).

`ncu` hardware counters were unavailable on this shared GPU (`ERR_NVGPUCTRPERM` — no perf-
counter permission), so the limiter is established from this timing decomposition rather
than SM-side counters; it is unambiguous (actual DRAM ≪ achievable DRAM ⇒ not memory-bound).

## Why ZG-0's "BW-bound ⇒ compression pays" did not carry over

ZG-0 proved the *uncompressed* TC GEMM tracks the cold-read ceiling (94–96%): the bf16
weight load saturates what DRAM can deliver at that cold footprint, and the tiny mma hides
under it. But there is **no spare compute shadow** to also hide a 5.5-op/elem decompress:
adding recon to the same warps steals exactly the issue slots that were feeding cp.async.
This is C-1T's structural verdict ("the wall is structural, not the recon ALU"; V3→V0 was
<1%) reproduced at the register-direct + coalesced-TCA level — the two fixes the plan
prescribed were implemented and correct (gates 1–3 prove it) but do not change the regime.

## Honest caveats

- **Cold single-tensor microbench.** These are L2-flushed cold reads (the latency-ramp
  regime, 34–66% of 1535). ZG-0 argued the *sustained* megakernel regime (~1522 GB/s,
  328 back-to-back tensors, no L2 flush) is where DRAM is truly saturated. The DIAG shows
  that even against the *matched cold ceiling* zip only reaches 21–66% — the recon deficit
  is a same-regime comparison, so it is not a cold-vs-sustained artifact. But a full e2e
  sustained-decode measurement (ZG-3) was out of scope; the definitive sustained number is
  not measured here.
- **Explicit slice-wise interleave not implemented.** The kernel has the tile-level
  double-buffer (STAGES4) but recon is issued in-place in the mma loop, relying on the
  compiler's ILP scheduler across NFRAG/warps to overlap recon (int pipe) with mma (tensor
  pipe) — not an explicit software pipeline of recon k-slice `s+1` under mma of `s`. Given
  the DIAG shows the bottleneck is cp.async *issue starvation* (recon stealing issue slots),
  not mma latency, reordering recon vs mma cannot add DRAM throughput it isn't feeding — and
  C-1T's V3-vs-V0 A/B already showed recon-op-count is not the lever. So this is recorded as
  the one residual degree of freedom, but the analysis predicts it will not flip <1.0× to
  ≥1.15×.

## Bottom line (plan ZG-2 kill)

Correctness is fully proven (offline TCA packer + shared lane derivation + all 3 gates).
The fused inline decompress does **not** realize the 1.33× byte ratio — it lands **≤0.95×**
and degrades toward high batch and deep K. Per the plan's ZG-2 KILL bar (<1.15× at every
B≤64), **ZipGEMM does not pay on the decode TC path.** Ship the tuned uncompressed TC-bf16
(ZG-0) at B≥16; shelve fused compression. `PLOW_NV_ZG`/emit wiring (ZG-3) not started.
