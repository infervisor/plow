# C-1T — tensor-core B-sweep + decompressor instruction tuning

Date 2026-07-22. Branch `c1t-tensorcore-bsweep` (based on `c1r-decode-occupancy`).
Model gemma-4-12B, RTX PRO 6000 Blackwell (188 SM, 1535 GB/s achievable, L2 96 MiB).
Plan `plans/p9-lossless-compression.md` §C-1T. GPU RTX 13.0 toolchain, system nvcc.
Sample: real gemma-4-12B layer-0 `down_proj` bf16 (112 MB), `EXP_BASE=109`, ratio 1.331×.
Harnesses: `runtime/tests/sz_decomp_sm120.cu` (Thread 2), `runtime/tests/sz_tc_sm120.cu`
(Thread 1). **All kernels bit-exact** (tc-sz output == tc-bf16 output, byte-identical; every
decompressor variant == bf16; 0 mismatches everywhere). `op_gemm.cuh` UNCHANGED — both
harnesses are standalone in `runtime/tests/`, so the default bf16 build is byte-identical.

## TL;DR

- **Thread 2 (decompressor): WIN.** `V3` (fully-packed 32-bit, 2 bf16/lane, `__byte_perm` +
  packed exp add) is **1.47× faster** than the current scalar `sz_expand8`, at **~5.5 SASS
  ops/elem vs ~11.8**, bit-exact, 0 spill. Clear drop-in for the B=1 cp.async decode kernel.
- **Thread 1 (tensor-core B-sweep): HONEST NEGATIVE.** A small-M (BM=16) tensor-core decode
  GEMM is **not weight-BW-bound on sm_120** — TC-bf16 tops out at ~1000 GB/s, below the 1535
  wall and below FFMA's 1464 GB/s at B=1. So SplitZip's fewer bytes buy nothing on the TC
  path: **TC-sz is 0.44–0.66× of TC-bf16 at every B and every grid.** The decompress does not
  overlap the mma — the expand→smem roundtrip + 2 `__syncthreads`/k-step is serial overhead.
- **There is NO B at which TC-sz beats TC-bf16.** The compression win does **not** extend up
  the B axis via tensor cores. Compression stays a **B=1** (C-1R FFMA cp.async) story.
- **What DOES win up the B axis: TC-bf16 (uncompressed).** It beats FFMA-bf16 at **B≥16** for
  all shapes (B≥8 for `down`). Right-sizing: switch decode to the tensor-core bf16 path at
  B≥16 — but leave the weights uncompressed there.
- **The leaner decompressor did NOT move the crossover** (proven by a V0-vs-V3 A/B *inside*
  the TC kernel: identical TC-sz). The bottleneck is structural, not the decompress ALU.
- **Register-staging (expand straight into `mma` fragments, no smem roundtrip) is WORSE**, not
  better — 0.19–0.52× of TC-bf16 (vs the tile-expand's 0.51–0.60×) — because the per-lane
  narrow byte-gathers are uncoalesced. See the follow-up probe below.

## Thread 2 — decompressor variant table (V0..V4)

ALU-bound microbench, compressed operands held in registers across 4096 iterations (no HBM
traffic), XOR-sink defeats DCE, inputs perturbed by the loop index (no serial feedback →
throughput, not latency). 58.98M real bf16 elems. ops/elem = SASS ALU-insn count ÷ 64
(8-way unroll × 8 elems); approximate. All bit-exact vs bf16 (in-window), 0 spill, 39 regs.

| variant | reconstruct | SASS ops/elem | elems/ns | expanded GB/s | vs V0 | bit-exact | ratio |
|---------|-------------|--------------:|---------:|--------------:|------:|:---------:|------:|
| V0 | scalar per-elem (current `sz_expand8`) | 11.81 | 4610 |  9221 | 1.000× | OK | 1.3310× |
| V1 | `lop3.b32` 3-way sign\|exp\|mant merge | 7.44 | 4661 |  9322 | 1.011× | OK | 1.3310× |
| V2 | `__byte_perm` base16 spread + scalar exp | 6.16 | 6071 | 12141 | 1.317× | OK | 1.3310× |
| **V3** | **packed 32-bit, 2 bf16/lane (WINNER)** | **5.53** | **6758** | **13517** | **1.466×** | **OK** | 1.3310× |
| V4 | pre-split byte planes, even base 108 | 6.36 | 5656 | 11313 | 1.227× | OK | 1.3321× |

- **V1 (`lop3`) is a nothing-burger** (1.011×): ptxas already fuses the two ORs; the scalar
  per-element loop structure is what costs, not the merge.
- **V3 is the winner.** The lever is *vectorization*, not any single PTX op: process 2 bf16 per
  32-bit lane, `__byte_perm` to spread the lo bytes into the sign/mantissa base, then a single
  packed `+base` and `<<7` builds both exponents at once. `mant | ((in<<8)&mask)` and
  `(e+baseK)<<7` are the whole kernel.
- **The `exp<<7` cross-byte shift (the "crux") is handled analytically**, not by a layout
  change: place `code+base` at bits[7:0]/[23:16] of the lane and one `<<7` lands both exponent
  fields correctly (low slot 14:7, high slot 30:23). No pre-split needed.
- **V4 (pre-split layout) — tradeoff quantified, NOT adopted.** Moving `exp[0]` into the low
  byte and coding `sign+exp[7:1]` over an *even* base makes the assembly byte-level (`prmt`),
  but the high byte still needs a base-add, so V4 (6.36 ops/elem, 1.23×) is **slower than V3**.
  Its ratio on this tensor (1.3321×) is *slightly better*, but that is **down_proj-specific**:
  window `[108,123]` happens to escape less here. The C-0 audit chose base=109 as the global
  best across all tensors; the even-base shift *risks* ratio wherever 109 is optimal. Net: V4
  costs ratio-generality for no speed → rejected. **V3 keeps the exact 1.331× ratio.**

**Winner fed to Thread 1's stageB: V3.**

## Thread 1 — per-shape B-crossover (GRID=188, 1 blk/SM)

GB/s = logical (uncompressed) weight bytes ÷ time. TC-sz uses V3 in the ring stager. FFbf/FFsz
= `gemv_rows` / `gemv_rows_sz` (C-1R FFMA; FFsz is the *naive* per-lane sz kernel). All
BITEXACT (tc-sz == tc-bf16, 0 mismatches).

| shape | B | TC-bf16 | TC-sz | FF-bf16 | FF-sz | TC-sz / TC-bf16 | TC-bf16 / FF-bf16 |
|-------|--:|--------:|------:|--------:|------:|:---------------:|:-----------------:|
| qkv K3840 | 1 | 957 | 542 | 1464 | 1084 | 0.566× | 0.654× |
| qkv K3840 | 8 | 944 | 539 | 758 | 394 | 0.571× | 1.245× |
| qkv K3840 | 16 | 934 | 537 | 232 | 199 | 0.575× | **4.026×** |
| qkv K3840 | 32 | 492 | 299 | 111 | 106 | 0.607× | 4.428× |
| o_proj K4096 | 1 | 1042 | 540 | 1428 | 1050 | 0.518× | 0.730× |
| o_proj K4096 | 8 | 1038 | 536 | 662 | 380 | 0.517× | 1.569× |
| o_proj K4096 | 16 | 1043 | 532 | 227 | 196 | 0.510× | **4.598×** |
| o_proj K4096 | 32 | 570 | 370 | 110 | 104 | 0.649× | 5.201× |
| gate/up K3840 | 1 | 1015 | 611 | 1481 | 1063 | 0.602× | 0.685× |
| gate/up K3840 | 8 | 1007 | 608 | 742 | 398 | 0.604× | 1.357× |
| gate/up K3840 | 16 | 997 | 605 | 235 | 202 | 0.607× | **4.238×** |
| gate/up K3840 | 32 | 491 | 303 | 112 | 107 | 0.617× | 4.399× |
| down K15360 | 1 | 943 | 497 | 1475 | 844 | 0.527× | 0.639× |
| down K15360 | 8 | 938 | 496 | 424 | 318 | 0.529× | **2.212×** |
| down K15360 | 16 | 935 | 496 | 218 | 183 | 0.530× | 4.280× |
| down K15360 | 32 | 620 | 271 | 107 | 100 | 0.437× | 5.782× |

(B=2,4 omitted here for space — in `perf-data/c1t-tensorcore-bsweep.json`; they track B=1/B=8.)

### Reading the table

- **TC-sz vs TC-bf16 (the compression question): sz loses everywhere, ~0.44–0.66×, flat in B.**
  No crossover exists — compression never wins on the tensor-core path.
- **TC-bf16 vs FFMA-bf16 (the right-sizing question):** FFMA is the BW-bound winner at B=1–4
  (1400+ GB/s); it collapses as B grows (compute-bound: 758→232→111 GB/s for qkv at B=8/16/32).
  TC-bf16 stays flat ~1000 GB/s (weight-BW-bound at the padded M=16 tile) until B=32 (two
  m-tiles). **Crossover: TC-bf16 overtakes FFMA-bf16 at B≥16 (B≥8 for `down`).**

## Why Thread 1 is negative (root cause)

1. **The TC path is not at the BW wall.** TC-bf16 ≈ 1000 GB/s = 65% of the 1535 ceiling and
   below FFMA's 1464. A scheme that only removes *bytes* cannot help a kernel that is not
   byte-bound. This mirrors the T6 prefill result (fp8 dequant-in-smem, the structural analog,
   +0.4..3.5% *slower* than bf16).
2. **Occupancy is not the lever here.** Sweeping GRID 188→940 (1→5 blk/SM) leaves TC-bf16 flat
   (~900–1020 GB/s) and TC-sz flat (0.53–0.66×). Unlike C-1R's naive FFMA-sz (which *was*
   latency/occupancy-bound), the TC pipeline is limited by its own expand→sync→ldmatrix→mma
   structure, not memory parallelism.
3. **The leaner decompressor cannot rescue it.** Swapping V3→V0 *inside* the TC kernel moves
   TC-sz by <1% (qkv B=8: 539→534 GB/s). The wall is the expand→smem roundtrip (write 10 KiB
   bf16/k-step) plus the 2 `__syncthreads` that bracket it — pure serial overhead the small-M
   mma is too short to hide. Decompress ALU throughput is irrelevant on this path.

## Follow-up probe — register-staged decompress (avoid the SRAM roundtrip)

Hypothesis (from the root cause): the wall is the expand→bf16-smem-tile→`ldmatrix` roundtrip
plus 2 syncs, not the decompress ALU. So expand **directly into the `mma` B-fragment
registers** — each lane loads only its own fragment's lo/cd bytes smem→reg and reconstructs
register→register (V3 `recon_pair`) — dropping the bf16 convert tile, the `ldmatrix`, and one
`__syncthreads`. smem falls 32.5 → 22.3 KiB; ptxas 96 reg, 0 spill. `k_tc_sz_reg`.

**Result: WORSE, not better.** TC-szREG is slower than the tile-expand TC-sz at every shape:

| shape (B=8) | TC-bf16 | TC-sz (tile) | TC-szREG (reg) |
|---|--:|--:|--:|
| qkv K3840 | 946 | 540 (0.57×) | 408 (0.43×) |
| o_proj K4096 | 1040 | 535 (0.51×) | 349 (0.34×) |
| gate/up K3840 | 1006 | 602 (0.60×) | 481 (0.48×) |
| down K15360 | 939 | 509 (0.54×) | **182 (0.19×)** |

Why it loses: the per-lane compressed operands are **narrow, uncoalesced smem loads** (2 B lo +
1 B cd per element) issued *inside* the `mma` k-loop, with bank conflicts and no vectorization,
whereas the tile-expand does one **coalesced `uint4`** smem write + HW `ldmatrix`. `down`
(K15360 → 480 k-steps) amplifies the per-k-step scatter to 0.19×. The elegant
"skip-the-roundtrip" idea loses to the fact that the roundtrip's coalesced path is
HW-accelerated and the scattered byte-gather is not.

Correctness caveat: `szr=BAD` — the hand-built m16n8k16 B-fragment lane-mapping has a layout
bug (~35% misrouted values). Not chased: the timing is **layout-independent** (identical loads,
recon, and `mma` count regardless of which lane owns which value), so the GB/s faithfully
measures the approach's cost, and that cost is already a decisive loss. This does not affect any
bit-exact claim — the shipped/primary TC-sz path (`k_tc_sz`, tile-expand) is bit-exact (0
mismatches); only this exploratory probe is not.

**This strengthens the Thread-1 negative:** *both* ways of feeding the tensor cores compressed
weights — the coalesced tile-expand (0.51–0.60×) and the register-staged gather (0.19–0.52×) —
are slower than uncompressed TC-bf16. The TC path is not weight-BW-bound, so no decompress
placement helps; the register-stage is additionally handicapped by uncoalesced access.

## Verdicts

- **Thread 2: KEEP V3.** 1.47× faster decompress (~5.5 ops/elem), bit-exact, 0 spill. Drop it
  into the C-1R B=1 cp.async kernel's `sz_expand8_s`. (Standalone-microbench win; e2e wiring is
  C-1R's unbuilt S3.)
- **Thread 1: KILL the tensor-core-sz path.** The SplitZip win does **not** extend up the B
  axis via tensor cores — the small-M decode GEMM is not BW-bound, so compression is invisible
  and its inline expand is strictly negative (0.44–0.66×). No crossover.
- **Right-sizing answer (how far up B the *compression* win extends): B=1 only.** Beyond B=1,
  no path (FFMA-sz or TC-sz) makes compression pay: FFMA-sz goes compute-bound, TC-sz is
  non-BW-bound. The *throughput* win at B≥16 belongs to **TC-bf16 (uncompressed)**.
- **Did the leaner decompress move the crossover? NO** — structural, proven by A/B.
- **`down` (K15360): not fixed.** TC-sz is the *worst* shape (0.53× at B≤16, 0.44× at B=32).
  TC-bf16 does beat FFMA at `down` B≥8, but with no compression.

## ptxas (sm_120a, 0 spill everywhere)

| kernel | regs | spill | barriers |
|--------|-----:|------:|---------:|
| `k_tc_bf16` | 52 | 0 | 1 |
| `k_tc_sz` (V3 inline) | 114 | 0 | 1 |
| decomp `k_thru` V0..V4 | 39 | 0 | 0 |

No 255-cliff risk. `op_gemm.cuh` untouched (new kernels live in `runtime/tests/`), so the
default bf16 cubin is byte-identical.

## Honest negatives / not done

- **No e2e / TPOT / serving numbers** — per-op microbench only, faithful to the megakernel
  geometry (GRID=188, HBM-resident real weights). e2e needs the sz weight emitter + on-load
  encoder (C-1R's unbuilt S3).
- **The tensor-core-sz thesis is refuted on sm_120** for the small-M decode geometry. Not
  retried: warp-specialized producer/consumer (C-1R H2 already KILLED it), or double-buffering
  the expand (would remove one sync, maybe +10–20% — cannot close a 2× gap when the path is not
  BW-bound to begin with).
- **qkv shape uses N=6144** (the C-1R harness value, for apples-to-apples vs the FFMA table);
  the true fused qkv on gemma-4-12B is N=8192 (q4096+k2048+v2048). Aspect ratio, not exact N,
  drives the BW-vs-compute regime, so the verdict is unaffected.
