# PX-11 — the flash DECODE kernel is at ~93% of its own access map. The map and the GQA re-read are the levers, not the body.

RTX 5090 (sm_120a, **170 SMs**, 96 MiB L2) · 2026-07-26
bench `perf-data/px11_flash_decode_bench.cu` (calls the SHIPPED `d_flash_decode` directly — no
cubin, no megakernel dispatch, px8 arm-Bp style) · raw `perf-data/px11-flash-decode-raw.txt`
build `perf-data/px11_build.sh` · runners `px11_run.sh` / `px11_knobs.sh` / `px11_run2.sh` /
`px11_run3.sh` · SASS `px11_sass.sh` + `px11_sass.py` · ptxas `px11_regs.sh`
Every GPU run under `perf-data/harness/gpulease`.

Companion to **PX-10**, which owns end-to-end attribution and is not re-derived here. PX-10's
numbers are used as the scaling base and are **not edited**.

## Question

PX-10 measured the hd512 full-layer `FlashDecodeFp8` at **527 GB/s** at B=8/131k while plow's own
M=1 GEMV streams fp8 weights at **1495 GB/s** on the same card — a 2.8× gap on what is supposedly
a pure streaming read. Prefill was compute-bound (PX-8/PX-9); decode at qlen=1 is GEMV-shaped and
bandwidth-bound, so the instrument here is **achieved GB/s against a ceiling this binary measures
itself**, not TFLOP/s.

The ladder is PX-9's method in a bandwidth unit: measure the ceiling first, then add ONE thing per
rung until the real kernel is reached, and attribute the gap by elimination.

## Result 0 — the ceiling, and the two ways to mis-measure it

2.15 GB working set, `__ldcs` (what the KV path uses), 170 blocks × 256 threads, one block/SM
(`__launch_bounds__(256,1)`, matching the megakernel). `U` = independent loads in flight per thread.

| rung | ms | GB/s |
|---|---|---|
| 0  linear 16 B/lane, U=1, occ 1 | 1.3778 | 1558.6 |
| **0  linear 16 B/lane, U=4, occ 1** | 1.2646 | **1698.1** |
| 0  linear 16 B/lane, U=8, occ 1 | 1.2628 | 1700.6 |
| 0b linear **8 B/lane**, U=1, occ 1 | 2.0789 | 1033.0 |
| **0b linear 8 B/lane, U=8, occ 1** | 1.2656 | **1696.8** |
| 0b linear 8 B/lane, U=16, occ 1 | 1.2617 | 1702.1 |
| 0c linear 16 B/lane, U=4, **occ 4** | 1.2629 | 1700.5 |
| 0c linear 8 B/lane, U=8, **occ 4** | 1.2613 | 1702.5 |

**1700 GB/s. This reproduces the in-tree 1695.6 to 0.3%**, so the denominator used everywhere
below is validated by this binary rather than assumed. (Against the 1792 GB/s spec pin that is
95%.)

Two eliminations fall out immediately, and both correct load-bearing beliefs in this campaign:

* **Occupancy is not a limiter.** 1 block/SM and 4 blocks/SM land within 0.3%. The megakernel's
  `__launch_bounds__(256,1)` costs the KV stream nothing.
* **8-byte loads are NOT slower than 16-byte loads.** At U=1 they look 35% slower (1033 vs 1559)
  — that is **memory-level parallelism, not request width**. With 8 loads in flight an 8 B/lane
  stream reaches the identical 1697 GB/s.

  > **This corrects `rtx19-e4-tc-fp8-decode.md`**, which reports "achievable fp8-byte bandwidth is
  > ~55–62% of the bf16 wall" and treats it as a hardware property that gates every fp8 decode
  > number. It is not a wall; it is an un-unrolled loop. `rtx19-e4` is left untouched — this is a
  > cross-reference, and its TC-vs-FFMA crossover conclusion does not depend on the number.

  The first version of this very ladder had the same bug (rungs 0b and 2 measured U=1 and read
  ~35% low). It was caught by rung 0b disagreeing with rung 1 in the wrong direction — a scattered
  map cannot beat a coalesced one — and fixed by parameterising every probe by `U`. Recorded in
  Gates.

## Result 1 — the ACCESS MAP is the wall, and it is a property of the row stride

Same 2.15 GB, same `__ldcs`, only the address map changes. Rung 1 is the score phase's map (one
whole KV row per THREAD, so 32 lanes of a warp are `ROWB` bytes apart); rung 2 is the V phase's map
(`NDT = ROWB/W` lanes cover one row, fully coalesced); rung 3 is what `PLOW_NV_FA_WPR` turns the
score phase into (a warp owns a row).

| rung | map | ms | GB/s | % of 1700 |
|---|---|---|---|---|
| 1 | row/thread, **1024 B** row, 16 B/ln, U=1 | 3.5979 | 596.9 | **35.1%** |
| 1 | row/thread, 1024 B row, 16 B/ln, U=4 | 3.5950 | 597.4 | 35.1% |
| 1 | row/thread, 1024 B row, 16 B/ln, U=8 | 3.5743 | 600.8 | 35.3% |
| 1 | row/thread, **512 B** row, 16 B/ln, U=8 | 1.7274 | 1243.2 | 73.1% |
| 1 | row/thread, **512 B** row, 8 B/ln, U=8 | 1.7006 | 1262.8 | 74.3% |
| 1 | row/thread, **256 B** row, 8 B/ln, U=8 | 1.5274 | 1406.0 | 82.7% |
| 2 | rowgrp, 1024 B row, 16 B/ln, U=1 / 2 / 4 / 8 | 1.3784 / 1.2691 / 1.2637 / 1.2636 | 1557.9 / 1692.1 / 1699.3 / **1699.6** | 100% |
| 2 | rowgrp, 512 B row, 8 B/ln, U=1 / **2** / **4** / 8 | 2.0908 / 1.3669 / 1.2679 / 1.2653 | 1027.1 / **1571.0** / **1693.7** / 1697.2 | 92–100% |
| 3 | **warp/row**, 1024 B row, 16 B/ln, U=4 | 1.2649 | **1697.7** | **99.9%** |

Read that as three facts:

1. **The score phase's row-per-thread map costs 65% of the machine on a 1024-byte row** (bf16
   hd512) and 26% on a 512-byte row (fp8 hd512 / bf16 hd256). It is **not** an MLP problem —
   U=1, 4 and 8 agree to 0.7%. The stride itself is the wall.
2. **The V phase's map is perfect** — but only with enough rows in flight. The kernel's
   `FA_DEC_VU(GF)` is 8 at GF=2, **4 at GF=4, and 2 at GF=8**, and rung 2 says U=2 gives
   **1571** where U=4 gives 1694. So GF=8 buys a 2× cut in demand and pays 7% on the V map.
3. **Warp-per-row recovers the whole thing** — 1697.7 GB/s, 2.83× the row-per-thread map at the
   same 1024 B row. That is exactly what `PLOW_NV_FA_WPR` does, and it defaults **OFF**.

## Result 2 — SASS: the fp8 score phase issues 8-byte loads, and pays 4 ALU ops per byte

The coordinator asked for the load width **before** theorising. `cuobjdump -sass` on the real
`d_flash_decode<512,4,true>` (fp8 KV), instruction census over the whole body:

| | shipped fp8 arm | `+PLOW_FP8_LD16 +PLOW_FP8_FAST` |
|---|---|---|
| K score-phase global loads | **64 × `LDG.E.64`** (8 B) | **32 × `LDG.E.128`** (16 B) |
| V-phase global loads | 5 × `LDG.E.64.CONSTANT` (8 B, VU-unrolled) | unchanged |
| Q from smem | 258 × `LDS.128` | 258 × `LDS.128` |
| `PRMT` (e4m3 byte extract/pack) | 1852 | 1852 |
| `F2FP.F16.E4M3.UNPACK_B` | 276 | 276 |
| `HADD2.F32` (half → f32) | 552 | 552 |
| **`F2F.BF16.F32` (the bf16 round-trip)** | **552** | **0** |
| `FFMA` | 2212 | 2212 |
| `BAR.SYNC` | 16 | 16 |
| **total instructions** | **9176** | **8048 (−12.3%)** |

So the dequant chain per e4m3 pair is `2×PRMT → 1×F2FP → 2×HADD2 → 2×F2F` and the last pair is
pure waste: e4m3 has 3 mantissa bits, which fit exactly in bf16 *and* in f32, so
`e4m3 → half → f32` and `e4m3 → half → f32 → bf16 → f32` produce **identical f32 bits**. The
shipped arm rounds through bf16 for nothing.

Neither `PLOW_FP8_FAST` nor `PLOW_FP8_LD16` is set by any build in the tree —
`scripts/build_sm120_cubin.sh` and `runtime/CMakeLists.txt` both leave them undefined.

## Result 3 — the real kernel, both layer classes, B=1 and B=8

`bytes_phys` = `B · NKV · span · D · elem · 2` (distinct HBM bytes);
`bytes_issued` = `B · (NH/GF) · span · D · elem · 2` (demand including the GQA re-read).
`reread = gqa/GF`. A **phys** rate above 1700 would mean a wrong denominator; an **issued** rate
above 1700 is legitimate and means L2 is absorbing re-reads.

### 3a. FULL layers (D=512, NH=16, NKV=1, gqa=16), B=8, ctx 131072, fp8 KV

| GF | nsplit | n_work | ms | GB/s phys | GB/s issued | % of 1700 |
|---|---|---|---|---|---|---|
| 2 | 21 | 1344 | 3.0805 | 348.6 | 2788.4 | 20.6% |
| 4 | 16 | 512 | 2.9372 | 365.6 | 1462.2 | 21.6% |
| **4** | **21** | 672 | **2.3183** | **463.2** | 1852.6 | 27.3% |
| 4 | **32** *(what PX-10 ran)* | 1024 | 2.6918 | 398.9 | 1595.6 | 23.5% |
| 4 | 85 | 2720 | 2.6926 | 398.8 | 1595.1 | 23.5% |
| 8 | 16 | 256 | 2.2693 | 473.2 | 946.3 | 27.9% |
| **8** | **21** | 336 | **1.7656** | **608.1** | 1216.3 | **35.9%** |
| 8 | 43 | 688 | 2.1182 | 506.9 | 1013.8 | 29.9% |
| 8 | 85 | 1360 | 1.8924 | 567.4 | 1134.8 | 33.5% |

My isolated 463–608 GB/s brackets PX-10's in-model **527 GB/s**; the isolated kernel gets no L2
warmth from neighbouring layers, so its absolute ms run ~30% high. **Every claim below is a
ratio measured within this binary**, never an absolute transplanted into PX-10's budget.

### 3b. SLIDING layers (D=256, NH=16, NKV=8, gqa=2, window 1024, ring 16384), B=8

The sliding working set is `B·NKV·1024·256` = **16.8 MB at fp8** — smaller than the 96 MiB L2, so
a warm loop measures L2, not HBM. In a real step those rows were last written up to 1024 steps and
25 GiB of traffic ago. Both protocols, `PX11_FLUSH=1` evicting 192 MB before each timed launch:

| dt | nsplit | warm ms | warm GB/s | **cold ms** | **cold GB/s** | cold % of 1700 |
|---|---|---|---|---|---|---|
| fp8 | **4** | 0.0170 | 1978.4 *(116% — L2)* | **0.0389** | **863.0** | 50.9% |
| fp8 | 8 | 0.0199 | 1688.5 | 0.0451 | 744.7 | 43.9% |
| fp8 | 16 *(shipped cap)* | 0.0278 | 1206.6 | 0.0512 | 655.8 | 38.7% |
| fp8 | 22 *(PX-10's trace)* | 0.0382 | 877.5 | 0.0737 | 455.3 | 26.9% |
| bf16 | **4** | 0.0427 | 1572.1 | **0.0553** | **1213.6** | 71.6% |
| bf16 | 16 | 0.0427 | 1572.1 | 0.0635 | 1057.6 | 62.4% |

**The 116.7%-of-ceiling cell is the check working.** A phys rate above the wall means the
denominator or the protocol is wrong, and here it was the protocol; the flush control resolves it.
PX-10's in-model 1161 GB/s sits between my warm and cold numbers, which is what a partially
L2-resident window should do — **PX-10's "the sliding flash is at bandwidth" verdict stands**, and
this note does not reopen it.

One free thing does fall out: **at B=8 the sliding nsplit wants to be ~4, not 16/22** (0.0389 vs
0.0512/0.0737 cold = 1.32×/1.89×), while at B=1 it wants 16 (0.0123 vs 0.0164). `devgen`'s
`ns.min(win/64)` cap is **batch-blind**. On a 5%-of-step component this is worth ~0.5 ms/step at
B=8 — small, free, and it belongs to the emitter, not this kernel.

## Result 4 — the kernel is at 92–94% of the ceiling its own map implies

Combine rung 1 (K half) and rung 2 at that GF's `FA_DEC_VU` (V half) as a harmonic mean, since the
two phases move equal bytes:

| arm | K map (rung 1) | V map (rung 2 @ VU) | implied issued ceiling | **measured issued** | **% of implied** |
|---|---|---|---|---|---|
| fp8, GF=8, shipped 8 B K loads (VU=2) | 1262.8 | 1571.0 | 1400.1 | 1216.3 | **86.9%** |
| fp8, GF=8, `+LD16` 16 B K loads (VU=2) | 1243.2 | 1571.0 | 1388.0 | 1285.1 | **92.6%** |
| fp8, GF=4, `+LD16` (VU=4) | 1243.2 | 1693.7 | 1433.9 | 2346.2 | *n/a — L2-amplified at reread 4* |

**At GF=8 — the arm closest to a single pass over the KV — the flash decode mainloop runs at
87–93% of the bandwidth its own access map allows.** That is the PX-9 result in a bandwidth unit:
the body (dequant, softmax, barriers, the 5 `__syncthreads` per tile, the online rescale) is worth
7–13%, and **the 2.8× gap against the GEMV's 1495 GB/s is not in the body at all**. It is:

* **1.35×** from the score phase's row-per-thread map (1263 vs 1700 on a 512 B fp8 row);
* **the rest** from the GQA re-read, which the shipped `GF_FULL=4` sets to **4×** on a model whose
  full layers have **one** KV head serving all 16 query heads.

The re-read is not free — L2 serves it, but at a price. Marginal rate on the extra issued bytes,
shipped arm, ns=21, B=8: GF 8→4 costs 0.553 ms for +1.07 GB (**1943 GB/s**), GF 4→2 costs 0.762 ms
for +2.15 GB (**2817 GB/s**). Both are L2-class rates, both are far from free, and both scale the
whole op.

## Result 5 — three build flags, measured. Two are bit-exact and one is register-negative.

FULL class, B=8, ctx 131072, fp8 KV. `base` is HEAD. The unaffected dtype in each arm is its own
**null control** and moved by <1% everywhere, which is how a "the `-D` never reached the code" bug
would have shown up.

| arm | GF=4 ns=21 | GF=8 ns=21 | GF=4 ns=32 *(PX-10's config)* |
|---|---|---|---|
| **base (HEAD)** | 2.3183 | 1.7656 | **2.6918** |
| `+PLOW_FP8_FAST` | 2.1972 (0.948×) | 1.6447 (0.932×) | — |
| `+PLOW_FP8_LD16 +FAST` | **1.8306 (0.790×)** | **1.6710 (0.946×)** | 2.1352 |
| `+LD16 +FAST +REDBOUND` | **1.7606 (0.759×)** | **1.6507 (0.935×)** | 2.0741 |
| `+PLOW_NV_FA_KUN=4 / =8` | 2.3019 / 2.3124 | 1.7556 / 1.7534 | — |
| `+PLOW_NV_FA_WPR=1` | 2.3105 | 1.7555 | — |
| `+WPR=1 +WPR_RB=4` | 2.3022 | 1.7554 | — |

`FA_KUN` and `FA_WPR` are **inert on the fp8 arm by construction** (`KUN` guards the bf16 branch;
`WPR` is `if constexpr (!SZKV && !FP8KV)`), and the table confirms it to 0.6%. On the **bf16** arm
they are not inert at all:

| arm (bf16 KV, FULL, B=8) | GF=4 ns=21 | GF=8 ns=21 |
|---|---|---|
| base | 3.0032 | 2.5069 |
| `+FA_KUN=8` | 2.5273 (0.842×) | 2.4625 (0.982×) |
| `+FA_REDBOUND=1` | 2.5781 (0.859×) | 2.5185 (1.005×) |
| `+FA_WPR=1` | 2.8468 (0.948×) | 2.1444 (0.855×) |
| **`+FA_WPR=1 +WPR_RB=4`** | **2.4262 (0.808×)** | **1.9488 (0.777×)** |
| `+WPR +RB=4 +QGLOB=1` | 2.4252 | 1.9579 |

**`FA_WPR=1 FA_WPR_RB=4` is worth 1.29× on the bf16 hd512 full layer** and lands it at 1102 GB/s
(65% of ceiling) — exactly the rung-1→rung-3 map change. `FA_QGLOB` adds nothing on top.

**But it does not reach the shipped configuration.** The 12B ships fp8 KV on the full layers, where
`WPR` is compiled out, and bf16 KV on the sliding layers, where the row is 512 B and the map costs
only 26% — measured on the sliding class L2-cold at B=8, `WPR_RB=4` is 0.0594 vs base 0.0553 ms at
ns=4, i.e. **slightly worse**. So `FA_WPR` stays off; see Result 6 for what would make it pay.

### The recommended pair, and its resource gate

`-DPLOW_NV_FA_GF_FULL=8 -DPLOW_FP8_LD16 -DPLOW_FP8_FAST`, against what the deployment cubin builds
today (`-DPLOW_NV_FA_GF_FULL=4`, `NS_FULL_ABS=32`):

| | ms | GB/s phys | vs deployed |
|---|---|---|---|
| deployed: GF=4, ns=32, HEAD flags | 2.6918 | 398.9 | 1.00× |
| GF=8, ns=21, HEAD flags | 1.7656 | 608.1 | **1.52×** |
| **GF=8, ns=21, +LD16 +FAST** | **1.6710** | **642.6** | **1.61×** |
| GF=8, ns=21, +LD16 +FAST +REDBOUND | 1.6507 | 650.5 | 1.63× |

Holds at B=1 too (base GF=4 ns=85 0.3415 → GF=8 +LD16 ns=85 **0.2262**, 1.51×).

`-Xptxas -v` on the **real** `interp_sm120.cu` decode object
(`PLOW_NV_GEMMA=1 PLOW_NV_FA_GF=2 PLOW_NV_EMBED_SMEM=1 PLOW_FP8_KV=1 PLOW_NV_W8A8=1`) — the
"megakernel resources are global" hazard, checked explicitly as in PX-8/PX-9:

| object | registers | spills | stack | static smem |
|---|---|---|---|---|
| `GF_FULL=2` | 241 | 0 | 1024 B | 2192 B |
| `GF_FULL=4` *(deployed)* | 241 | 0 | 1024 B | 2192 B |
| `GF_FULL=8` | **245** | 0 | 1024 B | 2192 B |
| **`GF_FULL=8` + `LD16` + `FAST`** | **241** | **0** | 1024 B | 2192 B |

**The recommended pair is register-identical to today.** `GF_FULL=8` alone costs 4 registers
(`oacc[8][8]`); `LD16` gives them back by halving the score-phase address chain. Occupancy is
1 block/SM at every point (241 > 128), so nothing moves there either. The dynamic arena grows
`FA_DEC_SMEM_FLOATS(512,4)` = 16448 B → `(512,8)` = **24640 B**, which the object publishes via
`plow_arena_bytes` and the host reads back, so it is self-consistent. (`interp_sm120.cu`'s
GF8-twin comment quotes "16448 vs 12352 B arena" for GF8-vs-GF2; 16448 is the **GF4** arena — the
GF8 arena is 24640 B. Noted, not edited.)

### Correctness arguments for each

* **`GF_FULL=8` is bit-exact at fixed `nsplit`.** `GF` only selects which query heads share a work
  item; each head keeps its own `m_st`/`l_st`/`oacc` and the loops over `d` and over `kv` are
  unchanged, so every accumulator sees the identical operand sequence. **Measured: `maxdiff` between
  GF=2, GF=4 and GF=8 at the same `nsplit` is `0.000e+00` in every one of the 100+ cells in the raw
  log**, both dtypes. The binding invariant `gqa % GF == 0` holds (16 % 8 = 0) and is already
  trapped at dispatch. Changing `nsplit` is **not** bit-exact (different split boundaries → a
  different merge), so `ns=21` needs the usual greedy gate, not a bit-exactness claim.
* **`PLOW_FP8_LD16` + `PLOW_FP8_FAST` are bit-exact.** e4m3 carries 3 mantissa bits, so
  `e4m3 → half → f32` is exact and the extra `→ bf16 → f32` hop the shipped arm takes changes no
  bit. `LD16` regroups the `d` loop 8→16 but each `dot[g]` still accumulates its own terms in
  `d`-increasing order. **Measured: 65536/65536 outputs exactly equal**, `maxabs` 0.000e+00, at both
  GF=4 and GF=8; the bf16 null control is byte-identical.
* **`PLOW_NV_FA_REDBOUND=1` is bit-exact.** Rows past `rmax_t` hold `NEG_INF`, so they contribute
  `fmaxf(x, NEG_INF) = x` to the max and `FA_EXP(NEG_INF − mnew) = 0.0f` to the sum, and `x + 0.0f`
  is exact in f32. Value is config-dependent (−14% on bf16/GF=4, −4% on fp8/GF=4, ~0 at GF=8) —
  recommended as neutral-to-positive, not as a headline.

### Scaled to PX-10's budget

Applying the **measured ratio** 1.6710/2.6918 = 0.621 to PX-10's 16.31 ms full-layer flash at
B=8/131k: **16.31 → ~10.1 ms**, so the 42.60 ms deployed step → **~36.4 ms**, and stacked on
PX-10's §5a `GV_MM_MAX` fix (34.35 ms) → **~28.2 ms**, i.e. **2.50× → ~1.65×** off the 17.06 ms
floor. Derived, not measured end-to-end — see Gates.

## Result 6 — is a kernel-body change warranted?

**Not urgently, and the honest answer is "the body is at 93%; the map is the lever."** Ranked by
measured ceiling, not plausibility:

1. **Flags first (Result 5): 1.61×, zero code, bit-exact, register-neutral.** Nothing in the body
   competes with this and it should land before any kernel work.
2. **Extend the warp-per-row score phase to the fp8 arm.** `FA_WPR` exists, is measured at 1.29× on
   the bf16 hd512 layer, and is excluded from fp8 only by `if constexpr (!SZKV && !FP8KV)`. The
   ladder sizes the prize exactly: the fp8 K map goes 1243 → 1698, so the GF=8 implied ceiling goes
   1388 → 1632 (**+17.6%**), and the kernel is at 93% of it. So ~**1.16× on the op** after the flags,
   i.e. ~1.4 ms of a 36 ms step. Real, bounded, and the only body change with a measured ceiling.
   Not bit-exact (the K reduction becomes a warp tree), so it needs a greedy gate.
3. **`FA_DEC_VU(8) = 2 → 4.**` rung 2 says the V map goes 1571 → 1694 at U=4, worth **+3.3%** of the
   GF=8 ceiling. `FA_DEC_VU` is a bare `#define` with no `#ifndef`, so it cannot even be A/B'd
   without editing the header. Small; listed for completeness, not recommended.
4. **Dead ends, measured:** `FA_KUN` (0.6% on fp8 — it guards a branch the fp8 arm does not take;
   1.16× on bf16/GF=4 only), `FA_QGLOB` (0.0%), higher occupancy (Result 0), wider loads as such
   (Result 0 — 8 B/lane already reaches the wall), and `FA_VDBUF` (not re-run; already measured
   negative).

**What is definitively NOT the problem:** the tensor-core lever PX-8 found for prefill does not
transfer — there is no mma in this kernel and the hypothesis that it would help is refuted by the
kernel already sitting at 93% of a *bandwidth* ceiling. Occupancy is not it. Request width is not
it. `nsplit` is not it in the sense PX-10 tested (count), though **`nsplit` alignment interacts
with `GF`**: at GF=8 `n_grp` drops to 2, so ns=21 (336 items ≈ 2/block on 170) is the aligned
choice and ns=32/85 are 20–7% worse. A GF change without a matching ns change leaves a third of
its value on the table.

## Gates

| gate | result |
|---|---|
| ceiling measured by this binary, not assumed | **PASS** — 1698–1702 GB/s across 4 independent rungs and 2 occupancies; reproduces the in-tree 1695.6 to 0.3% |
| every phys GB/s below the ceiling | **PASS after a FAILURE** — the L2-warm sliding arm read **116.7%** of 1700. Not a denominator bug: the 16.8 MB sliding window is L2-resident. Caught by this check, resolved by the `PX11_FLUSH` control, both columns reported (Result 3b) |
| shipped kernel measured, not a bench copy | **PASS** — `k_fd_bf16`/`k_fd_fp8` call `d_flash_decode<D,GF,FP8>` from the unmodified header |
| `runtime/` source unchanged | **PASS** — this note adds only `perf-data/` files; `git diff` touches no kernel |
| GF bit-exactness at fixed nsplit | **PASS** — `maxdiff` GF=2/4/8 = `0.000e+00` in every cell, both dtypes, 5 nsplits, B=1 and B=8 |
| `LD16`+`FAST` bit-exactness vs HEAD | **PASS** — 65536/65536 outputs exactly equal at GF=4 and GF=8; bf16 null control byte-identical |
| per-arm null controls (fp8 knobs must not move bf16, and vice versa) | **PASS** — every unaffected cell within 1% |
| megakernel registers / spills from `-Xptxas -v` | **PASS** — 241 → 241, 0 spills, on the REAL `interp_sm120.cu` decode object; `GF_FULL=8` alone is 245 |
| SASS load width read before theorising | **PASS** — 64 × `LDG.E.64` → 32 × `LDG.E.128`; 552 `F2F.BF16.F32` → 0 |
| L2-cold protocol on the sliding class | **ENFORCED** — 192 MB eviction before each timed launch, outside the timed window |
| L2-cold protocol on the FULL class | **NOT NEEDED** — 1.07 GB (fp8) / 2.15 GB (bf16) working set, 11–22× the L2 |
| reproducibility across leases | **PASS** — `GF=4/ns=21` measured 2.3068 / 2.3181 / 2.3183 / 2.3182 (0.5% spread); `GF=8/ns=21` 1.7507 / 1.7656 / 1.7673 / 1.7875 (2.1%). Treat <2% as noise |
| GPU exclusive | **ENFORCED** — `gpulease`, worktree copy (the fixed `foreign()`); rc=0 on every run of record |
| `ncu` counter attribution | **NOT RUN** — `ERR_NVGPUCTRPERM` in this container, as in PX-9. Every claim here is differential timing + SASS census + an independently measured map ladder, never a hardware counter |
| **end-to-end decode step A/B with the recommended flags** | **NOT RUN** — no checkpoint or block asset in this worktree; the 1.61× is an isolated-kernel ratio and the 42.6 → 36.4 ms figure is *derived* from PX-10's 16.31 ms, not measured. **This is the gate to run next.** |
| greedy-token parity for `ns=21` | **NOT RUN** — `nsplit` changes the merge order and is not bit-exact. The GF and flag changes ARE bit-exact and need no such gate; `ns` does |
| B=1 sliding numbers | **LOW CONFIDENCE** — 0.012–0.020 ms is at the CUDA-event quantum (values snap to ~0.002 ms steps). Directionally reported only |
| PX-10's fp8-KV hd512 prefill crash at bucket ≥ 4096 | **NOT REPRODUCED** — this bench never runs a prefill arm. Nothing learned; still open |

### Bugs found mid-run, recorded

1. **`PLOW_NV_FA_WPR=1` together with `PLOW_NV_FA_QGLOB=1` silently corrupts the fp8 and SZ KV
   arms.** `op_attention.cuh:446` guards the Q→smem staging with
   `#if !(PLOW_NV_FA_WPR && PLOW_NV_FA_QGLOB)`, which is a **preprocessor** condition and therefore
   removes the staging from *every* instantiation — but the `QGLOB` read-from-global path lives
   inside `if constexpr (!SZKV && !FP8KV)`, so the fp8 and SZ arms fall through to the default body
   and read a **never-written `qsm`**. Measured: fp8 output diverges by 5.03e-01 on a peak-2.9e-03
   tensor, i.e. total garbage, while the bf16 arm is fine. No shipped build sets both flags, so
   nothing in the tree is affected today — but the guard is wrong and a future `QGLOB` sweep on an
   fp8 packet would produce fluent wrong text with no crash. Not fixed here (out of scope: this
   note changes no `runtime/` source); the one-line fix is to make the guard
   `#if !(PLOW_NV_FA_WPR && PLOW_NV_FA_QGLOB)` *and* keep the staging whenever `SZKV || FP8KV`,
   which needs it to move inside the template as an `if constexpr`.
2. **This ladder's own first version measured the wrong ceiling.** Rungs 0b and 2 used a
   non-unrolled grid-stride loop, so they were latency-bound at ~1 load in flight and reported
   1033 GB/s for an 8 B/lane stream and 1027 for the V map — 39% low. It looked like a real
   "narrow loads are slower" finding and it agrees with a number already in the tree
   (`rtx19-e4`'s 55–62%), which is precisely what made it dangerous. Caught because rung 1 (a
   *scattered* map) beat rung 0b (a *coalesced* one), which is impossible. Fixed by giving every
   probe an explicit `U` and reporting the whole `U` sweep rather than one point.
3. **`gpulease` run `px11-r2` exited rc=141.** That is SIGPIPE from my own `| head -60`, not the
   harness and not contention — the run was silently truncated after one arm. Re-run without the
   pipe. Noted because rc=141 on a leased run reads like a lease failure and is not one.

## Reproduce

    bash perf-data/px11_build.sh /tmp/px11_base
    bash perf-data/px11_build.sh /tmp/px11_fp8ld16 -DPLOW_FP8_LD16 -DPLOW_FP8_FAST
    GPU_LEASE_TIMEOUT=3000 perf-data/harness/gpulease px11 bash perf-data/px11_run2.sh 20
    GPU_LEASE_TIMEOUT=3000 perf-data/harness/gpulease px11 bash perf-data/px11_run3.sh 20
    bash perf-data/px11_regs.sh          # ptxas on the real megakernel object

## Recommended order

1. **`-DPLOW_NV_FA_GF_FULL=8` in `scripts/build_sm120_cubin.sh`** (it currently says 4;
   `runtime/CMakeLists.txt` says 2). Bit-exact at fixed `nsplit`, +4 registers alone. **1.52×** on
   the full-layer flash op at B=8.
2. **`-DPLOW_FP8_LD16 -DPLOW_FP8_FAST`** on the fp8-KV objects. Bit-exact, gives the 4 registers
   back. **1.61×** combined with (1).
3. Teach `devgen` that the full-layer `nsplit` target depends on `FA_GF_FULL` (it hard-codes
   `const FA_GF_FULL: u32 = 2` at `crates/devgen/src/lib.rs:65` while the deployment cubin builds
   4). At `GF=8`, `ns=21` is the aligned choice and is 7–20% better than 85/32. Needs a greedy gate.
4. Make the sliding `nsplit` cap batch-aware (~4 at B=8, 16 at B=1). ~0.5 ms/step, emitter-side.
5. Only then: extend `FA_WPR` to the fp8 arm (Result 6.2), ceiling-sized at ~1.16× on the op.
