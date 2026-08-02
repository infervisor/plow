# PX-16 — occupancy 2 is NOT worth pursuing for DECODE. It measures **1.07×**, and the premise it rests on is wrong in both directions.

RTX 5090 (sm_120a, **170 SMs**, 96 MiB L2) · 2026-07-26
bench `runtime/bench/nvidia/px16_occ_bench.cu` (calls the SHIPPED `d_flash_decode<512,GF,true>` directly,
px11 style) · build `px16_build.sh` · runner `px16_run.sh` · ptxas `px16_regs.sh` /
`px16_minblk.sh` · raw `perf-data/px16-decode-occupancy-raw.txt`
Every GPU run under `perf-data/harness/gpulease` (labels `px16`, `px16b`, `px16c`; rc=0).

Companion to **PX-11**, whose ladder, denominators and 1700 GB/s measured wall are reused, not
re-derived. PX-11's numbers are not edited.

## VERDICT (one line)

**No.** Best occ-1 config **1.789 ms** → best occ-2 config **1.679 ms** = **1.066×** on the
full-layer flash decode op, against a **1.2×** bar. At the *deployed* GF=4 config occupancy 2 is
**negative** (0.88–0.99×). The whole per-cell "occ 2 wins 1.1–1.4×" signal is **wave
quantisation**, not residency — a matched-grid control kills it.

---

## 0. The arithmetic — verified in structure, wrong in two inputs, and **moot**

Device numbers re-queried on the card (`cudaGetDeviceProperties`, printed in the raw log), not
assumed:

| quantity | claimed | **measured** |
|---|---|---|
| shared mem / SM | 102,400 B | **102,400 B** ✓ |
| registers / SM | 65,536 | **65,536** ✓ |
| max dynamic smem / block | 101,376 B | **101,376 B** ✓ |
| SMs | 170 | **170** ✓ |
| spec HBM pin | 1792 GB/s | **1792.1 GB/s** ✓ (derived from 14001 MHz × 512 bit) |

The chain is right: occ 2 at 256 threads needs `65536/(2·256)` = **128 regs/thread**, and
occupancy is a step function so partial relocation buys exactly zero. Two inputs are off:

* **The decode object is 241 registers, not 236.** `-Xptxas -v` on the object
  `scripts/build_sm120_cubin.sh` actually deploys (`PLOW_NV_GEMMA=1 PLOW_NV_FA_GF=2
  PLOW_NV_FA_GF_FULL=4 PLOW_NV_EMBED_SMEM=1`): **241 registers, 0 spills, 1024 B stack**.
  Identical at `+PLOW_FP8_KV`, identical at `+PLOW_FP8_KV +PLOW_NV_W8A8`. (Prefill is 238.)
  This reproduces PX-11's 241 exactly. So the cut is **113** registers, not 108.
* **The arena already in use is 16,448 B, not 12,352 B.** 12352 is the `FA_GF=2` value and is
  explicitly the *legacy default* in `crates/plowrt/src/exec/gpu.rs:639` for pre-metadata
  cubins. The deployed object's `plow_arena_bytes` is
  `FA_DEC_SMEM_FLOATS(512, PLOW_NV_FA_GF_FULL=4)` = **16,448 B**
  (`runtime/nvidia/interp_sm120.cu:469`).

Redone with the measured inputs: 113 × 4 B × 256 = **115,712 B** to relocate; at occ 2 a block
gets 51,200 B and 16,448 is spoken for, leaving **34,752 B ≈ 33.9 regs/thread**. Short by
**3.33×**, not 2.85×. The conclusion "relocation cannot reach occ 2" **stands and is stronger**.

### …and none of it matters, because nobody has to relocate anything

**`ptxas` already does the cut.** `interp_sm120.cu:433` carries a `PLOW_NV_FORCE_MINBLK` knob
whose comment describes this exact experiment. Compiled at `PLOW_NV_FORCE_MINBLK=2`
(`perf-data/px16_minblk.sh`), the **real decode megakernel**:

| object | registers | spill st / ld | stack | static smem |
|---|---|---|---|---|
| `FORCE_MINBLK=1` (deployed) | 241 | 0 / 0 | 1024 B | 2192 B |
| **`FORCE_MINBLK=2`** | **128** | **484 B / 640 B** | 1216 B | 2192 B |

Occupancy 2 costs **one `-D`** and about half a kilobyte of spill. The proposed
110-KB-of-registers-into-smem project is not merely infeasible — it is **unnecessary**. That
reframes the question completely: this was never a "can we afford the register cut" problem, it
is a "**is occupancy 2 worth anything at all**" problem. It is not.

> Aside worth knowing: **`-maxrregcount` does not cap this object.** Measured — `-maxrregcount=128`
> and `=96` both still report 241 registers, because a source `__launch_bounds__` takes precedence
> over the flag. Anyone probing megakernel occupancy with `-maxrregcount` is measuring nothing.

---

## 1. The register count everyone is quoting does not belong to flash decode

The 241 is the megakernel's **worst case over every inlined op**. Compiled on its own, the
decode kernel is far below it (`cudaFuncGetAttributes`, printed per arm in the raw log):

| kernel | registers | spill | blocks/SM at its own arena |
|---|---|---|---|
| `d_flash_decode<512,4,true>` (deployed GF_FULL=4) | **128** | 0 | **2** |
| `d_flash_decode<512,8,true>` (PX-11's recommended GF) | **168** | 0 | 1 |
| `d_flash_decode<512,8,true>` @ `__maxnreg__(128)` | 128 | 24 B | 2 |
| `d_flash_decode<512,8,true>` @ `__maxnreg__(96)` | 96 | 80 B | 2 |

**At the deployed `GF_FULL=4`, flash decode is already inside the occ-2 budget with zero
registers to spare and none to remove.** The 113-register overhang belongs to some *other* op
in the interpreter's switch, not to the kernel this proposal wanted to rewrite. Any register
campaign aimed at `d_flash_decode` would have cut the wrong function.

---

## 1b. Dynamic register allocation (`setmaxnreg`) does not change any of this

Raised mid-campaign: NVIDIA has dynamic register allocation, and the interpreter's kernels are
normal kernels. Both halves checked (`px16_setmaxnreg_probe.cu` / `px16_setmaxnreg.sh`), because
this is exactly the kind of thing that sounds like it rescues the proposal.

**It is supported on sm_120a — with a caveat.** `setmaxnreg.dec/inc.sync.aligned.u32` assembles,
and a kernel using it launches clean on the 5090 (`launch k_smn: no error`). But it is **not in
the base `sm_120` target**: an executable built plain `-arch=sm_120a` also embeds `compute_120`
PTX for forward compat, and that JIT path is a hard error —
`Instruction 'setmaxnreg.dec' not supported on .target 'sm_120'`. Any object using it must be
pinned `-gencode arch=compute_120a,code=sm_120a` and forfeits its PTX fallback.

**It is not used anywhere in the tree.** `grep -rn setmaxnreg runtime/ crates/ include/` is empty.
The `wgmma`/`warpgroup` hits are all sm_90 paths or TODOs. The interpreter is not warp-specialized:
all 8 warps run the same op body out of the same switch, which is the shape `setmaxnreg` has
nothing to offer.

**And it cannot raise occupancy, by construction.** Two independent confirmations:

* **Measured.** The probe kernel decs one warpgroup to 24 and incs the other to 232. It still
  reports `numRegs = 128` and `blocks/SM = 2` — identical to its `__launch_bounds__(256,2)`
  declaration. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` reads the *launch-time* allocation;
  a runtime `dec` is invisible to it. A control kernel with no `setmaxnreg` at 14 registers gets 6.
* **The SASS says so in the mnemonic.** `cuobjdump` on both sm_120a and sm_90a:

      USETMAXREG.DEALLOC.CTAPOOL     0x18        // dec to 24
      USETMAXREG.TRY_ALLOC.CTAPOOL   UP0, 0xe8   // inc to 232

  **`.CTAPOOL`.** Registers are released to, and acquired from, the **CTA's own pool** — and the
  probe is exactly zero-sum: `dec` gives up 104/thread across 128 threads, `inc` takes back
  104/thread across 128 threads. The CTA's aggregate footprint never changes, so the SM's
  CTA-slot accounting never sees it. That is why `TRY_ALLOC` can fail (`UP0`) — it is competing
  for its own CTA's freed registers, not the SM's.

So `setmaxnreg` is a **warp-specialization** tool: it buys producer/consumer *asymmetry* inside a
CTA, not more CTAs per SM. Using it here would mean warp-specializing the interpreter — rewriting
every op body — to arrive at the same blocks/SM.

**It is moot regardless.** At the deployed `GF_FULL=4`, `d_flash_decode` is *already* 128 registers
and already occ-2-capable with zero register work of any kind (§1) — and §3a measures occupancy 2
there as **negative, 0.88–0.99×**. There is no register mechanism, static or dynamic, that changes
a verdict whose bottleneck is the access map.

## 2. Probe A — what occupancy is worth to these access maps when it is **free**

Same 2.15 GB working set and `__ldcs` as PX-11's ladder, but each map compiled at
`__launch_bounds__(256, MINB)` for MINB = 1..4 and launched at MINB × 170 blocks. These probes
use 14–160 registers, so **occupancy 1→4 costs nothing** — no cap, no spill. This is the
absolute ceiling on what occ 2 could ever buy on each map.

| map (what it is in the kernel) | occ 1 | occ 2 | occ 3 | occ 4 |
|---|---|---|---|---|
| linear stream (PX-11's wall pin) | **1700.3** | 1701.5 (1.001×) | — | 1699.5 (1.000×) |
| **row/thread 512 B, U=8** — fp8 hd512 **SCORE** phase | **1261.3** | 1264.9 (**1.003×**) | 1247.5 (0.989×) | 1224.0 (0.970×) |
| row/thread 512 B, **U=1** — the latency-starved variant | **1254.7** | 1264.3 (**1.008×**) | — | 1227.4 (0.978×) |
| row/thread 1024 B, U=4 — bf16 hd512 score phase | **599.0** | 626.1 (1.045×) | — | 599.5 (1.001×) |
| **rowgrp 512 B, U=2** — **V** phase at `FA_DEC_VU(8)` | **1534.3** | 1691.6 (**1.103×**) | — | 1694.5 (1.104×) |
| rowgrp 512 B, **U=4** — V phase at `FA_DEC_VU(4)`, **occ 1** | **1693.2 (1.104×)** | — | — | — |
| rowgrp 512 B, **U=8** — V phase at `FA_DEC_VU(2)`, **occ 1** | **1697.0 (1.106×)** | — | — | — |

Read that as three facts, and they are the whole story:

1. **Occupancy is worth 0.3% to the score phase.** The row-per-thread map is at 74% of the wall
   and *stays* there at 2, 3 and 4 blocks/SM. It gets no better with more warps because the
   stride, not the warp count, is the wall — and PX-11 proved that independently by showing U=1
   and U=8 agree. Note rung 2: even the *deliberately latency-starved* U=1 variant gains only
   0.8% from doubling residency. If extra warps cannot rescue a map that has one load in flight
   per thread, they will not rescue anything.
2. **Occupancy is worth 10.3% to the V phase — and only because `FA_DEC_VU(8)` is 2.** The last
   three rows are the control: **occ 2 at U=2 (1691.6) and occ 1 at U=4 (1693.2) land within
   0.1% of each other.** Extra resident warps and a wider unroll are buying the *same* memory-level
   parallelism. `FA_DEC_VU` is a one-line `#define` (`op_attention.cuh:98`).
3. **Occupancy 3 and 4 are negative** on every real map (0.97–0.99×).

So before touching the real kernel we already know the shape of the answer: occ 2 can only pay
where the V-phase unroll is short, and where it pays, a `#define` pays the same for free.

---

## 3. Probe B — the real kernel, with occupancy **isolated** from grid size

This is where the naive experiment goes wrong, so it is worth being explicit. Comparing
grid = 170 against grid = 340 conflates **residency** with **wave quantisation**: `d_flash_decode`
is a grid-stride loop over `n_work = B·(NH/GF)·nsplit` items, so changing the grid changes how
evenly the items divide. The control is to **fix the grid at 340 and change only the occupancy**,
by padding the dynamic arena past `102400/2`. The kernel only ever indexes the first
`FA_DEC_SMEM_FLOATS` floats, so the pad is inert; both pads exceed 48 KB, so both opt into the
same max-shared carveout and leave the same L1 behind. The only difference is that 50,176 B lets
two blocks fit and 52,224 B lets one.

### 3a. GF=4 — occupancy 2 is **free** here (128 regs) and it **loses**

B=8, ctx 131072, fp8 KV, grid fixed at 340, `maxdiff` = 0.000e+00 in every cell.

| nsplit | occ 1 (pad 52224) ms | occ 2 (pad 50176) ms | occ 2 (natural arena) ms | **occ2 / occ1** |
|---|---|---|---|---|
| 21 | **2.0217** | 2.2988 | 2.3883 | **0.879×** |
| 32 *(deployed)* | **2.4335** | 2.4629 | 2.4506 | **0.988×** |
| 43 | **2.3367** | 2.6345 | 2.6362 | **0.887×** |
| 64 | **2.3542** | 2.4396 | 2.4640 | **0.965×** |
| 85 | **2.3516** | 2.5280 | 2.5181 | **0.930×** |

**Five nsplits, five losses, 1.2% to 12.1%.** Two blocks per SM means two independent KV streams
per SM and 340 concurrent streams instead of 170; the score map gains nothing (Probe A) and the
V map is already at `FA_DEC_VU(4)` = the knee (Probe A row 6), so there is nothing to win and a
real cost to pay. The padded and natural-arena occ-2 columns agree to ~1%, which is the check
that the pad is inert.

### 3b. GF=8 — occupancy 2 wins **per cell**, and the win is wave quantisation

At GF=8 the kernel needs 168 registers, so occ 2 requires the `__maxnreg__(128)` cap and its
24 B spill. Grid fixed, occupancy set by the pad:

| nsplit | cap128 occ 1 ms | cap128 occ 2 ms | ratio |
|---|---|---|---|
| 16 | 2.4024 | 1.8063 | 1.330× |
| 21 | 1.8542 | 1.6909 | 1.097× |
| 32 | 2.4516 | 1.7375 | 1.411× |
| 43 | 2.2428 | 1.9399 | 1.156× |

1.10–1.41× looks decisive, and it is a trap. At GF=8, `n_grp = 2`, so `n_work = 16·nsplit`; at
ns=32 that is 512 items over 340 blocks — **1.5 items per block**, i.e. half the blocks do 2 and
half do 1, and at occ 1 the second wave runs half-empty. Occupancy 2 is *recovering a
quantisation loss that occ 1 recovers for free by choosing a better nsplit*. Which is exactly
PX-11 recommendation #3, and PX-6's result that the bucket ladder is a function of `n_cu`.

---

## 4. Best-vs-best — the only comparison that decides anything

Ratios of single cells are wave-quantisation noise. Ratios of the **minimum over nsplit of each
family** are not. GF=8, B=8, ctx 131072, fp8 KV, full 8-point nsplit sweep (raw log has all 32
cells):

| nsplit | occ 1 nat-168 grid 1× | occ 1 nat-168 grid 2× | occ 2 cap-128 | occ 2 cap-96 |
|---|---|---|---|---|
| 11 | 3.3324 | 3.3346 | 2.3253 | 2.4316 |
| 16 | 2.2777 | 2.2785 | 1.7784 | 1.7787 |
| **21** | **1.7889** | 1.8085 | 1.6908 | **1.6785** |
| 26 | 2.1740 | 2.1550 | 2.0216 | 1.9090 |
| 32 | 2.3268 | 2.2994 | 1.7129 | 1.7048 |
| 43 | 2.1429 | 2.1367 | 1.9259 | 1.8392 |
| 64 | 2.0321 | 2.0298 | 1.7672 | 1.6987 |
| 85 | 1.8958 | 1.8790 | 1.7189 | 1.7204 |
| **min** | **1.7889** | 1.8085 | 1.6908 | **1.6785** |

> **min occ 1 = 1.7889 ms · min occ 2 = 1.6785 ms · ratio = 1.066×**

Run-to-run spread across three leases is 1.9% on the occ-1 minimum (1.7707 / 1.7889 / 1.8040;
PX-11 independently measured 1.7656) and 2.6% on the occ-2 minimum (1.6495 / 1.6785 / 1.6932).
So the honest statement is **1.05–1.09×**, and the point estimate is **1.066×**.

Note also that grid oversubscription **at occ 1** (column 2) buys nothing at the aligned nsplit
(1.8085 vs 1.7889) — confirming again that the grid is not a free lever once nsplit is right.

### The same conclusion stated against the deployed configuration

| configuration | ms | vs deployed |
|---|---|---|
| deployed today: GF=4, ns=32, grid 170 | 2.7243 | 1.00× |
| GF=4, ns=**21**, grid 170 — *emitter-side, free, occ 1* | 2.4484 | 1.113× |
| GF=4, ns=32, grid 340, **occ 2** | 2.4506 | 1.112× |
| **GF=8, ns=21, grid 170 — PX-11 recs #1+#3, occ 1** | **1.7889** | **1.523×** |
| GF=8, ns=21, grid 336, occ 2 (cap 96, spilling) | 1.6785 | 1.624× |

The third row is the one that has been fooling people: at the deployed config occupancy 2 does
appear to give 1.112×. **The second row gives the same 1.113× at occupancy 1, for free, by
changing one integer in the emitter.** And `GF_FULL=8` + the aligned nsplit — PX-11 recs #1 and
#3, both bit-exact or greedy-gated, neither touching a register — gives 1.52× before occupancy
is discussed at all. (This bench does not compile the `LD16`/`FAST` arms; PX-11 measures those
as a further 1.06× on top, and nothing here revisits them.)

---

## 5. The registers-vs-GB/s curve

GF=4, ns=32, grid fixed at 340, `__maxnreg__` swept. Columns are what ptxas actually produced
and what the driver actually computed — a cap that ptxas ignored and an occupancy that smem
blocked look identical in ms, so both are read back, never assumed.

| `__maxnreg__` | regs achieved | spill | blocks/SM | ms | GB/s phys | vs natural |
|---|---|---|---|---|---|---|
| — (natural) | 128 | 0 | 2 | 2.4466 | 438.9 | 1.000× |
| 224 | 117 | 0 | 2 | 2.4375 | 440.5 | 1.004× |
| 192 | 117 | 0 | 2 | 2.4477 | 438.7 | 1.000× |
| 168 | 117 | 0 | 2 | 2.4479 | 438.6 | 0.999× |
| 152 | 117 | 0 | 2 | 2.4444 | 439.3 | 1.001× |
| 136 | 117 | 0 | 2 | 2.4565 | 437.1 | 0.996× |
| **128** | 117 | 0 | 2 | 2.4530 | 437.7 | 0.997× |
| 120 | 101 | 0 | 2 | 2.4574 | 436.9 | 0.996× |
| 104 | 94 | 0 | 2 | 2.4543 | 437.5 | 0.997× |
| 96 | 85 | 0 | 2 | 2.4533 | 437.7 | 0.997× |
| 88 | 80 | 0 | **3** | 2.4486 | 438.5 | 0.999× |
| 80 | 76 | 0 | **3** | 2.4609 | 436.3 | 0.994× |
| 72 | 72 | 0 | 3 | 2.6223 | 409.5 | **0.933×** |
| 64 | 64 | **32 B** | **4** | 2.8460 | 377.3 | **0.860×** |

**The curve is flat from 224 registers all the way down to 80, and then it falls off a cliff.**
Across a 2.8× swing in register count and occupancy 2 → 3, throughput moves by 0.6% — under the
2% noise floor. There is no knee to find, because occupancy is not the variable this kernel
responds to. Below 80 registers ptxas starts spilling into the address chain and the kernel
loses 7–14%.

The same sweep read as blocks/SM: occ 2 = 1.000×, occ 3 = 0.999×, occ 4 = 0.860×.

## 6. Where the kernel actually stands (denominator check)

Every `GB/s phys` above is under the 1700 GB/s wall PX-11 measured and this binary re-measured
(1699.5–1701.5 across three occupancies, reproducing PX-11 to 0.1%). The best decode cell,
GF=8/ns=21/occ 2, is **639.7 GB/s phys = 37.6% of the wall**. `GB/s issued` legitimately exceeds
the wall (max 2145.7 at GF=4, where `reread = gqa/GF` = 4 and L2 absorbs it), which is PX-11's
stated rule, not a denominator bug.

The gap to the wall is where PX-11 left it: the score phase's row-per-thread map (1261 of 1700)
and the GQA re-read. Occupancy addresses neither.

## 7. B=1

| arm | ms | vs |
|---|---|---|
| GF=4 ns=85 grid 170 | 0.3434 | 1.000× |
| GF=4 ns=85 grid 340 (occ 2, free) | 0.3030 | 1.133× |
| GF=4 ns=85 grid 340 cap96 | 0.2990 | 1.148× |

B=1 does show 1.13–1.15×, but `n_work` = 1·4·85 = 340, i.e. **exactly one item per block at grid
340 and two at grid 170** — this is the purest wave-quantisation cell in the whole sweep, not an
occupancy result. The GF=4 §3a control (which holds the grid fixed) is the number to trust, and
it says occupancy 2 loses.

---

## 8. If someone still wants occupancy 2, this is what it costs

Not a recommendation — a scoping note, because the kernel side is the *small* half.

* **`grid = occ × sm_count` must equal the packet's `n_cu`**, and it is a fatal load-time gate:
  `crates/plowrt/src/exec/gpu.rs:645-656` computes `occ = occupancy_blocks_per_sm(f, 256, smem)`
  **from the cubin** and refuses to launch if `occ × 170 != blob.n_cu`. So a `FORCE_MINBLK=2`
  cubin *automatically* demands **every packet re-emitted at `n_cu = 340`**. There is no
  kernel-side-only version of this change.
* **PX-6 showed the wave-quantisation bucket ladder is a function of `n_cu`**, so the re-emission
  is not a constant substitution — the prefill bucket set moves with it. §3b and §7 above are the
  same effect showing up inside a single op.
* The prefill object would follow: `gpu.rs:2263-2272` additionally requires the prefill grid to
  equal the decode grid.
* And PX-7's 1.05× for occ 2 on the prefill GEMM is the benchmark this would have to beat
  *twice over* to be worth the re-emission.

## 9. What to do instead — all measured, all cheaper

1. **`-DPLOW_NV_FA_GF_FULL=8` with the aligned `nsplit`** (PX-11 recs #1+#3). **1.52×** on this op
   measured here at occupancy 1 (2.7243 → 1.7889 ms), register-neutral, no packet change —
   against occupancy 2's **1.066×** for a whole-fleet packet re-emission. Adding PX-11's
   `-DPLOW_FP8_LD16 -DPLOW_FP8_FAST` on top is a further 1.06× by PX-11's measurement.
2. **`nsplit` alignment (PX-11 rec #3).** ns=21 vs ns=32 is 1.11× at GF=4 and 1.30× at GF=8, from
   one integer in the emitter. This is the lever occupancy 2 was accidentally proxying for.
3. **`FA_DEC_VU(8): 2 → 4`** (`op_attention.cuh:98`). Probe A prices it at +10.3% of the V map,
   which is *the same thing occupancy 2 buys at GF=8* (1691.6 vs 1693.2 GB/s, 0.1% apart) — for
   one `#define` and no registers, no spill, no packet re-emission. It is currently a bare
   `#define` with no `#ifndef`, so it cannot be A/B'd without editing the header; adding the
   guard is the prerequisite and is the single highest-value follow-up in this note.
4. Extend `FA_WPR` to the fp8 arm (PX-11 rec #5), ceiling-sized at ~1.16× on the op — still the
   only body change with a measured ceiling.

**Do not scope the register cut.** It is not needed (§0: ptxas reaches 128 regs by itself),
and it is not worth it (§4: 1.066× against a 1.2× bar).

---

## Gates

| gate | result |
|---|---|
| coordinator's arithmetic checked before anything was built | **PASS with 2 corrections** — 241 regs not 236, 16448 B arena not 12352 B; shortfall is 3.33× not 2.85×. Structure and step-function conclusion both stand |
| device constants queried, not assumed | **PASS** — 102400 B/SM, 65536 regs/SM, 101376 B optin, 170 SMs, all printed in the raw log |
| ceiling re-measured by this binary | **PASS** — 1699.5 / 1700.3 / 1701.5 GB/s at occ 1/2/4, reproduces PX-11's 1700 to 0.1% |
| every `phys` GB/s below the 1700 wall | **PASS** — max 639.7. `issued` exceeds it only where `reread`>1 (PX-11's stated rule) |
| sanity vs the 1792 GB/s spec pin | **PASS** — computed from the device's own 14001 MHz × 512 bit = 1792.1, and nothing measured exceeds 1701 |
| occupancy MEASURED, not inferred from `__launch_bounds__` | **PASS** — every arm reports `cudaFuncGetAttributes.numRegs` / `.localSizeBytes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| occupancy isolated from grid size / wave quantisation | **PASS, and it inverted the answer** — the naive grid-170-vs-340 comparison reads +11%; the matched-grid smem-pad control reads −1% (§3a). This is the load-bearing control in the note |
| carveout matched between the occ-1 and occ-2 controls | **PASS** — both pads (50176 / 52224 B) exceed 48 KB and opt into the same max-shared carveout; padded and natural-arena occ-2 columns agree to ~1% |
| register cap did not change arithmetic | **PASS** — `maxdiff` = 0.000e+00 in all 80+ flash-decode cells; a cap adds spills, never operations |
| shipped kernel measured, not a bench copy | **PASS** — `k_fd_nat` / `k_fd_reg` call `d_flash_decode<D,GF,true>` from the unmodified header |
| `runtime/` source unchanged | **PASS** — this note adds only `perf-data/` files; `git diff` touches no kernel. `PLOW_NV_FORCE_MINBLK` is a pre-existing knob |
| best-vs-best, not cell-vs-cell | **PASS** — 8-point nsplit sweep per family; the headline 1.066× is min/min. Single cells range 1.06×–1.43× and are quantisation |
| reproducibility across leases | **PASS** — occ-1 min 1.7707/1.7889/1.8040 (1.9%); occ-2 min 1.6495/1.6785/1.6932 (2.6%). Treat <2.6% as noise; 1.066× is stated as 1.05–1.09× |
| no isolated ratio scaled through an assumed budget | **ENFORCED** — every number here is a ratio measured inside one binary. **No end-to-end step time is claimed or derived.** The 1.52× for PX-11's flags is likewise this bench's own ratio, not PX-11's transplanted |
| `setmaxnreg` availability checked, not assumed | **PASS** — assembles for sm_120a and launches clean; fatal on the base `sm_120` PTX target. sm_90a used as the known-good control so a failure could not be my syntax |
| `setmaxnreg` occupancy claim verified two ways | **PASS** — driver reports `numRegs=128 / blocks-per-SM=2` unchanged by a dec-24/inc-232 pair, and the SASS scope is literally `USETMAXREG.*.CTAPOOL` |
| `ncu` counter attribution | **NOT RUN** — `ERR_NVGPUCTRPERM` in this container, as in PX-9/PX-11. Every claim is differential timing + driver-reported resources |
| L2-cold protocol | **NOT NEEDED** — 1.07 GB fp8 working set at B=8, 11× the L2. Only the FULL class is measured here |
| SLIDING layer class | **NOT MEASURED** — PX-11 has it at 5% of the step; occupancy was never proposed for it |
| **megakernel-wide `FORCE_MINBLK=2` end-to-end** | **NOT RUN** — no checkpoint or block asset in this worktree, and it is a different question: MINBLK=2 changes *every* op, not just flash decode. See below |

### The one thing this note does NOT close

`FORCE_MINBLK=2` is a whole-object flag. It makes flash decode **1.07× faster at best and 0.88×
at the deployed GF**, but the decode object also contains the **GEMV** arms, and there is prior
in-tree evidence that *those* like occupancy: `decode-cpasync-occ-sweep-26b-h100.md` measures the
H100 GEMV-family aggregate at 172.1 → 145.9 µs going occ 1 → 2 (**1.18×**), and
`c1r-decode-occupancy.md` measures 1.09–1.12× at 2 blocks/SM on sm_120-class silicon. **That is a
live question and this note does not answer it.** What it does answer is the question that was
asked: *the flash-decode kernel does not want occupancy 2, and no register-relocation work is
justified by it.* If someone re-opens occupancy for the decode object, the case must be built on
the GEMV arms, at n_cu=340, end-to-end — and flash decode will be paying 0.88–1.07× into it.

## Reproduce

    bash perf-data/px16_build.sh /tmp/px16_base
    GPU_LEASE_TIMEOUT=3600 perf-data/harness/gpulease px16 bash perf-data/px16_run.sh 20
    bash perf-data/px16_regs.sh        # ptxas on the objects the deployment builds
    bash perf-data/px16_minblk.sh      # the real decode object at __launch_bounds__(256,1) vs (256,2)
