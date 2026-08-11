# PX-8 — the fp8 P.V for the hd512 flash prefill: what actually blocks it, and what does not

RTX 5090 (sm_120a, **170 SMs**, 96 MiB L2, 101,376 B dynamic-smem cap) · 2026-07-26
bench `runtime/bench/nvidia/px8_flash_fp8pv_bench.cu` · raw `perf-data/px8-flash-fp8-pv-raw.txt`
run under `perf-data/tools/gpulease`
Follows PX-7 Result 4/5. `px7_w8a8_ceiling_bench.cu` untouched.

## Question

PX-7 Result 4 attributes **16.48 s of plow's 30.85 s 127k prefill (53%)** to the hd512
full-attention flash prefill — 128 TFLOP/s against vLLM's implied 230. PX-7 Result 5 proposed ONE
change to close it: fp8-only staging, **BQ=64 / BKV=32**, and e4m3 on BOTH mmas, arguing the smem
budget makes the bigger tiling free (83,968 B claimed, against 85,248 B allegedly used today and a
101,376 B cap).

This note verifies that budget from the source, finds the two real blockers (neither is smem), and
measures whether an e4m3 P.V is worth building.

## Provenance — exactly which code every number below came from

This matters more than usual here, because Result 1 shows the shipped fp8-KV prefill cubin does
not contain the px4 fp8mma arm at all. Stating it up front so no table can be misread:

| what | how it was built / run |
|---|---|
| every timing and numerics row | **`runtime/bench/nvidia/px8_flash_fp8pv_bench.cu`**, a standalone binary. It does **not** load a cubin and does **not** go through the megakernel dispatch. |
| the **px4 column** | `k_armA` calls **`d_flash_prefill_px4<512,32,16,true>` directly** — the real shipped source, no copy. |
| the px4 arm's compile flags | `-gencode arch=compute_120a,code=sm_120a -DPLOW_FP8_KV=1`, with `PLOW_NV_FA_PIPE=1` and `PLOW_NV_FA_FP8MMA=1` at their defaults. **That is byte-for-byte the arm `PLOW_FP8_KV_FASTPF=ON` selects.** |
| the **px8 column** | `k_armBp` calls **`d_flash_prefill_px8<512,32,32,true>` directly**, same flags plus `-DPLOW_NV_FA_FP8PV=1`. |
| arm B (bench-local copy) | only carries the ablation bits and the LDS.8 oracle; it agrees with `k_armBp` at every point, and the shipped-arm row is the one quoted. |
| the smem / register / SASS facts | the real `interp_sm120.cu` object built four ways (see Result 1), `-Xptxas -v` and `cuobjdump -sass`. |

**So the 0.716 ratio compares px8 against the correct incumbent.** `PLOW_FP8_KV_FASTPF` is a CMake
option that only decides whether `-DPLOW_NV_FA_PIPE=0` is added to the *cubin* build; the bench
bypasses that by instantiating the template itself at `PIPE=1`, which is the FASTPF=ON
configuration. The px4 baseline here is the px4 fp8mma arm, not the PIPE=0 fallback.

**What is NOT settled by that:** whether the *end-to-end* long-context numbers earlier in this
campaign ever exercised px4's fp8 arm. See the warning at the end of Result 1.

## Result 1 — two explicit corrections to PX-7 Result 5

`px7-w8a8-ceiling.md` is **not edited**; these are cross-references, and PX-7's Results 1-4 are
untouched by any of this. Two of Result 5's load-bearing numbers are wrong.

**Correction A — the smem figure for "today" is the wrong object.** PX-7 Result 5 says the fp8
arm "already uses 85,248 B today" and builds its whole argument on 83,968 < 85,248. 85,248 B is
the **bf16** prefill object's arena. The fp8 arm claims **89,104 B**. The comparison was between
two different objects.

**Correction B — "occupancy is unchanged; the tiling is free" is true but vacuous.** It is offered
as if occupancy were the thing at stake. Occupancy is 1 block/SM and cannot move: the real
`interp_sm120_pf` object is **238 registers with 0 spills**, and 2 blocks/SM needs <=128. Registers
pin it independently of smem, so no smem reduction can buy occupancy. The relevant question was
never "does the tiling fit" — it was "what does the mma need", which is Result 2.

`FA_PRE_SMEM_FLOATS` compiled straight from the real headers, both arms:

| arm                                        | bytes       | PX-7 said |
|--------------------------------------------|-------------|-----------|
| px4 bf16, BQ32/BKV16                       | 70,672      | 70,656    |
| **px4 fp8mma, BQ32/BKV16 — what ships today** | **89,104**  | "measured 85,248" |
| generic hd256, BQ64/BKV32 (the OTHER arena claim) | 81,664 | 19,840 floats (stale comment) |
| px4 fp8mma, BQ64/BKV32 as written          | 186,384     | 183,808   |
| `sharedMemPerBlockOptin`                   | 101,376     | 101,376   |

PX-7's "85,248 B the arm already uses today" is the **bf16** prefill object's arena
(`max(hd256 81,664 ; hd512 70,672)` plus the GEMM claim), not the fp8 object's. The fp8 arm is at
**89,104 B**.

`-Xptxas -v` on the real `_Z15interp_sm120_pf11PlowProgram`, all three build variants (bf16 /
fp8 PIPE=0 / fp8 PIPE=1): **238 registers, 0 spills, 1024 B stack**. So occupancy is 1 block/SM
twice over — smem (89 KiB > 50 KiB) *and* registers (238 > 128). Shrinking smem cannot raise
occupancy. PX-7's "the tiling is free in occupancy terms" is true but vacuous.

### Build-flag finding — possibly bigger than this whole note

The shipped fp8-KV prefill cubin is built **`-DPLOW_NV_FA_PIPE=0`** — `scripts/build_sm120_cubin.sh`
does it unconditionally under `PLOW_BUILD_FP8KV=1`, and `runtime/CMakeLists.txt` does it whenever
`PLOW_FP8_KV_FASTPF` is `OFF`, which is its default:

```cmake
option(PLOW_FP8_KV_FASTPF "fp8-KV prefill via the PIPE=1 px4 fp8-mma arm (hd512 only)" OFF)
if(PLOW_FP8_KV)
    if(PLOW_FP8_KV_FASTPF) set(PLOW_FP8_KV_PF_DEFS PLOW_FP8_KV=1)
    else()                 set(PLOW_FP8_KV_PF_DEFS PLOW_FP8_KV=1 PLOW_NV_FA_PIPE=0)
```

`PLOW_NV_FA_PIPE=0` makes `FA_PX4_ELIGIBLE(HD)` false, so **the default fp8-KV prefill object
contains no px4 fp8mma arm and no fp8 mma anywhere**. It runs the generic PIPE=0
`d_flash_prefill<512,32,16,true>`, which dequants K and V inline from gmem to bf16 and does both
mmas in bf16.

> **This casts doubt on a PX-7 conclusion, and it is worth checking before anyone builds on it.**
> PX-7 Result 4 states that "enabling the px4 fp8 arm recovered only 10% of the quadratic term
> (b 1.131e-9 -> 1.022e-9)". If that A/B was run on default-built fp8-KV cubins, it never enabled
> the fp8 arm at all — it toggled `PLOW_FP8_KV` while the prefill object stayed on the bf16-mma
> PIPE=0 path, and the 10% would be the fp8 *cache* (halved KV bytes) rather than the fp8 *mma*.
> That number should be re-derived with `PLOW_FP8_KV_FASTPF=ON` explicitly recorded. **Not
> resolved here** — this note's own measurements bypass the cubin entirely (see Provenance), so
> they are unaffected either way, but the end-to-end campaign numbers may not be.

## Result 2 — the real blocker: `mma.m16n8k32` needs V TRANSPOSED

`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` wants BOTH operands with the CONTRACTION
dimension contiguous per lane (validated by the shipped QK code: `B` lane L reads 8 consecutive
k-bytes at `Ks8[n*(HD+PAD8) + k]`). For P.V the contraction is **kv**. P is register-resident so its
layout is ours to choose. V is staged natural `Vs8[kv][hd]` — hd contiguous, kv strided by 544 B.

So an fp8 P.V needs V transposed. That is precisely what FlashAttention-3's fp8 path pays for, and
it is the thing PX-7 Result 5 never mentions. It is also why "BKV=32 fills k32" is necessary but
nowhere near sufficient.

**sm_120a has an 8-bit transposing ldmatrix.** Measured fragment map (`px8 ... layout` mode; this
instruction is undocumented in the in-tree ISA notes, which only record an open forum thread):

    ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {r0,r1,r2,r3}, [addr];
    lane L supplies the address of source row L (rows 0..15 = matrix 0, 16..31 = matrix 1)
      r0 = T[n = L>>2    ][srcrow 4*(L&3) .. +3]   (matrix 0)
      r1 = T[n = (L>>2)+8][srcrow 4*(L&3) .. +3]   (matrix 0)
      r2, r3 = the same two columns, matrix 1

The mma's B operand instead wants lane L to hold `B[n = L>>2][k = 8*(L&3) .. +7]`. Those differ by
a **quad permutation of k** — 4 SHFLs per ldmatrix if fixed in registers.

They do not have to be fixed in registers. Nothing forces smem row order to equal kv order, and
cp.async copies whole 16 B lines, so **only the destination row index changes**:

    smem row s  holds  kv = 8*((s>>2)&3) + (s&3) + (s>=16 ? 4 : 0)
    kv -> s = ((kv & 4) ? 16 : 0) + 4*(kv>>3) + (kv & 3)

With V staged in that order, `{r0,r2}` and `{r1,r3}` **are** the two B operands. The transpose costs
zero instructions, zero extra smem, and zero extra HBM traffic.

## Result 3 — the second blocker: BQ=64 is register-infeasible, not smem-infeasible

`oacc` is `BQ*HD / (PLOW_NV_WARPS*32)` f32 per lane — **64 at BQ=32, 128 at BQ=64**, because
`PLOW_NV_THREADS` is hard-wired to 256. Measured on an isolated P.V accumulator loop
(`/tmp/px8/bq64_regs.cu`): 86 registers at NJ=16, **150 at NJ=32**, 0 spills either way — exactly
+64.

The new arm below measures 192 registers at BQ=32 (arm A, the shipped px4 fp8mma, is 180). BQ=64
therefore lands at ~256 against a 255 cap: ptxas spills, and the spilled values are *accumulators
live across the whole KV loop*, so they reload every tile. BQ=64 at HD=512 needs 512 threads per
block, which the megakernel's fixed 256 forbids.

**PX-7's "three wins arrive as one change" is wrong: BQ 32->64 is a different, harder change with a
register wall, and it is not a precondition for anything else.**

## Result 4 — the third blocker: per-row `v_scale` cannot fold into a raw e4m3 V

plow's KV cache carries a **per-kv-row** f32 scale. The shipped fp8mma arm folds `v_scale` into V
during its e4m3->fp16 dequant, which is exactly the pass an fp8 P.V deletes. With V left raw, the
scale must fold into P instead — but `P * v_scale ~ 1e-2` sits **below e4m3's smallest normal**
(2^-6 = 0.0156), so a naive fold flushes the entire P tile to subnormals.

Fix used here: normalise P by the tile's max `v_scale` (8 broadcast `LDS.128` + 31 `FMAX`, all lanes
reading the same addresses) and carry `vmax/256` in the accumulator's **units**, folding the unit
change into the online-softmax `corr` multiply that already exists. Cost in the P.V loop: zero.

## Result 5 — static instruction counts (SASS, `cuobjdump -sass`), per warp per 32 kv rows

| | arm A x2 (shipped px4 fp8mma, BKV=16) | arm B (new, BKV=32) |
|---|---|---|
| QK tensor core   | 16 `QMMA.16832.F32.E4M3.E4M3` | 16 `QMMA.16832` |
| P.V tensor core  | 32 `HMMA.16816.F32`           | **16 `QMMA.16832`** |
| B-operand loads  | 32 `LDSM.16.MT88.2`           | **8 `LDSM.8.MT1616.2`** |
| V dequant        | the `F2FP`/`STS` pass         | **gone** |
| `BAR.SYNC` / tile| 14                            | **7** |

`HMMA.16816` does half the MACs of `QMMA.16832` at the same issue cost, so arm A's P.V is 4x the
tensor-core time of arm B's for identical work.

## Result 6 — the P.V phase in isolation, per 32 kv rows, 170 blocks, 1 block/SM

Everything the two arms pay between "V[t] has landed as raw e4m3 in smem" and "O accumulated":

| P.V arm | ns / 32-kv tile | vs px4 |
|---|---|---|
| px4: dequant e4m3->fp16 (v_scale folded) + 32 `ldmatrix.b16` + 32 f16 mma | 1065.6 | 1.00x |
| **px8: 8 `ldmatrix.m16n16.x2.trans.b8` + 16 e4m3 mma** | **368.5** | **2.89x** |
| px8 with an 8x`LDS.8` gather instead of the ldmatrix | 1453.5 | **0.73x** |
| px8 with the Vs8 row pad at 16 B instead of 32 B | 363.2 | 2.93x |

**The 8-bit transposing ldmatrix is not a convenience, it is the whole lever.** Gathering the
transposed B operand by hand is 35% *slower* than the fp16 P.V it replaces — an fp8 P.V built the
obvious way is a NEGATIVE. Everything here rests on that one instruction plus the free row
permutation.

The Vs8 row pad is worth 1.9% (32 B gives a 4-way bank conflict over the 16 lane-supplied row
addresses, 16 B gives 2-way). Left at 32 B in the shipped arm: the same tile stride is read as
`uint2` by the QK path, where 32 is the conflict-free choice, and 1.9% does not justify a second
stride.

## Result 7 — the full kernel, trailing 8k chunk, nh=16 nkv=1, grid 170

`ns/tile` is per block per KV tile of that arm's own BKV, so it is directly comparable within a
column and 2x apart between them.

| seq_kv | px4 ms | px4 ns/tile | px4 TFLOP/s | **px8 ms** | px8 ns/tile | **px8 TFLOP/s** | px8/px4 |
|--------|--------|-------------|-------------|--------|-------------|-------------|---------|
| 8k     | 8.164  | 1318.4      | 134.7       | **6.032** | 1948.1   | **182.3**   | **0.739** |
| 32k    | 56.357 | 1304.5      | 136.6       | **40.521**| 1876.0   | **189.9**   | **0.719** |
| 128k   | 250.682| 1310.9      | 136.0       | **179.542**| 1877.7  | **189.8**   | **0.716** |

The shipped `d_flash_prefill_px8` (arm Bp, called directly) reproduces the bench copy at every
point: 6.062 / 40.500 / 180.147 ms.

**1.40x on the op**, flat in context. Per kv row the arms are 81.9 ns (px4) vs 58.7 ns (px8).

`ns/tile` is **flat across 8k -> 128k in both arms** (px4 spread 1.0%). That reproduces PX-4's
finding: **the hd512 flash is not KV-traffic bound**, so BQ=64's halving of the KV re-read — the
second of PX-7's "three wins" — buys nothing even before the register wall. Only one of the three
was real.

### What that does to the PX-7 budget

Rescaling PX-7 Result 4's 16.48 s quadratic term by the measured 0.716:

| | now | with px8 |
|---|---|---|
| flash prefill (quadratic) | 16.48 s | **11.80 s** |
| total 127k prefill        | 30.85 s | **26.17 s** |
| vLLM                      | 14.2 s  | 14.2 s |

Implied flash TFLOP/s goes 128 -> 179 against vLLM's 230. **A real 15% off total prefill, and not
parity.** PX-7 sized this lever at ~2x (16.48 -> 8.2 s); the measured value is 1.40x. The gap is
that PX-7 costed only the tensor-core work and assumed the traffic halved too — Result 7 shows the
traffic was never the wall, so only one of PX-7's "three wins" was real.

## Result 8 — numerics against an f32 reference

Same bf16 Q, same e4m3 K/V, same per-row scales; seq_q 256, seq_kv 1024, 4 heads, causal.
"max rel" is taken only where the reference exceeds 5% of its peak magnitude.

| arm | max abs | max rel | rms / rms_ref |
|---|---|---|---|
| px4 (fp16 P.V, the shipped baseline) | 4.382e-03 | 4.260e-03 | 1.888e-03 |
| px8 via the LDS.8 gather (oracle)    | 7.443e-03 | 6.739e-03 | 2.292e-03 |
| px8 via `ldmatrix.trans.b8`          | 7.443e-03 | 6.739e-03 | 2.292e-03 |
| **shipped `d_flash_prefill_px8`**    | **7.443e-03** | **6.739e-03** | **2.292e-03** |

Two things fall out.

1. **The ldmatrix path and the hand-gather oracle agree to every printed digit.** The gather is
   correct by construction, so this validates the measured fragment map AND the `FA_PX8_VROW`
   permutation. The shipped kernel matches both.
2. **Quantising P to e4m3 costs 1.7x on max error and 1.21x on RMS, not the ~100x a naive
   mantissa-bit count predicts.** Two reasons, and both are structural rather than lucky: the
   softmax denominator `l` is accumulated from the **unquantised** p, so the quantisation error
   lives only in the numerator and cannot compound through the online rescale; and the P.V
   contracts 32 independent kv terms, so the per-element error averages down. The remaining error
   is dominated by the e4m3 K/V that the baseline already carries.

## Result 9 — the bug this run found

The first ablation table came back with **every** cell at −0.7%, including "everything skipped".
Cause: the ablation bits were a template parameter tested with `#if !(ABL & 1)`. The preprocessor
cannot see a template argument — it evaluated `ABL` as an undefined identifier, i.e. 0, so no arm
was ever ablated and all six binaries were the same code. The uniform −0.7% was pure run-to-run
noise, which is itself the useful reading: **noise on this harness is ~0.7%, so the 28% headline is
40x noise.** Fixed by `if constexpr`; the corrected variants have different register counts
(139/174/184/188/192), which is the cheap proof that the ablation is now real.

## Result 10 — where the px8 arm's time actually goes (corrected ablation, seq_kv 32k)

Full = 40.374 ms. Each row nulls ONE phase while every barrier, commit and wait stays in place.

| phase nulled | ms | exposed cost | share |
|---|---|---|---|
| — (control)                       | 40.374 | —      | —     |
| QK mma                            | 35.922 | 4.452  | 11.0% |
| softmax (incl. P quant + rescale) | 34.459 | 5.915  | 14.7% |
| P.V mma                           | 35.225 | 5.149  | 12.8% |
| **cp.async issue (no gmem)**      | 30.478 | **9.897** | **24.5%** |
| everything (loop + barrier floor) | 7.252  | 33.123 | 82.0% |

The three compute phases now sit at 11–15% each — **balanced**, which is what an fp8 P.V was
supposed to achieve and did. The largest single item is the **cp.async staging exposure at 24.5%**,
and the loop+barrier floor is 18% of the total. Neither is a tensor-core problem, so the next
lever on this kernel is streaming/pipelining (a real K/V double buffer now that fp8-only staging
freed the bytes), not more mma work. Individual deltas sum to 63% against 82% for all four
together, i.e. the phases overlap by about a fifth.

Two runs, independently queued: the headline is 0.716–0.719 (128k) and 0.719–0.720 (32k) — the
0.7% noise floor Result 9 exposed.

Related: the isolated `NJ_PV=32` P.V loop (the BQ=64 accumulator shape) runs at **0.985x of 2x the
BQ=32 loop**, i.e. no per-unit penalty *in isolation* at 159 registers. The BQ=64 wall is not the
P.V loop on its own; it is 192 + 64 inside the whole kernel.

## Result 11 — the two end-to-end gates (the assets on disk could not run them)

### The harness had to be rebuilt from scratch, and that is itself a finding

Nothing on disk could run this gate. `/root/plow-out/lc-b2` — the obvious long-context fp8-KV
asset — fails three ways at once:

1. its `model.pkt` is a **v7** blob and `gemma4_sm120_chat` only accepts **v5** (emit with
   `--no-rope-gen`);
2. its decode program is **T=2**, and the harness force-DISABLES prefill when `DB > 1`, so a
   prefill A/B on it silently measures nothing;
3. it is a **pure bf16-KV packet** — 0 `kv.*_scale` tensors, `op39` count 0 in every prefill
   bucket. It never emits `FLASH_PREFILL_FP8` at all.

`gemma4_sm120_chat` **statically links** the interp objects, so the flash arm is fixed at CMake
time and there is no cubin env var to get wrong — but the *packet* must match the arm, because
the px4/px8 fp8-mma arm is hd512-only and `__trap()`s on an hd256 fp8 packet. Two packets were
emitted for this gate, and the pairing was verified before running:

| packet | emit env | `k_scale` tensors | pairs with |
|---|---|---|---|
| `px8-pkt-bf16kv` | `PLOW_FP8=1` | **0** (bf16 KV) | `ref` |
| `px8-pkt-fp8kvfull` | `+ PLOW_FP8_KV=1 PLOW_FP8_KV_FULL=1` | **8** (the 8 hd512 full layers only) | `px0`/`px4`/`px8` |

Both at `--max-ctx 73728 --n-cu 170`, `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_DECODE_BATCH=1`,
fp8 weights on every arm identically so weight format is not a variable.

**Four binaries**, differing only in the prefill flash arm:

| arm | CMake | prefill flash arm |
|---|---|---|
| `ref` | `-DPLOW_CUDA=ON` | bf16 KV, generic |
| `px0` | `+ -DPLOW_FP8_KV=ON` | fp8 KV, **PIPE=0 generic — no fp8 mma anywhere** |
| `px4` | `+ -DPLOW_FP8_KV_FASTPF=ON` | fp8 KV, PIPE=1 px4 fp8-mma (fp16 P.V) |
| `px8` | `+ -DPLOW_NV_FA_FP8PV=ON` | fp8 KV, px4 arm + **e4m3 P.V** |

`px0` exists because without it the needle result below is unattributable.

### Gate 1 — greedy-token parity, 8192-token prompt, 64 greedy tokens — **PASS**

Non-repetitive repo prose (unique-8gram share 1.000, so the softmax is not artificially peaked).

| pair | first divergence |
|---|---|
| px4 vs bf16 (the incumbent's own divergence) | token **15** |
| **px8 vs bf16** | token **15** |
| px8 vs px4 | token 20 |

**px8 diverges from bf16 at exactly the same index as the incumbent**, and px8 tracks px4 for
*longer* (20) than either tracks bf16 (15) — i.e. the e4m3 P.V perturbs the stream less than the
fp8 KV cache that px4 already carries. That is the bar the coordinator set, and it is met.

Prefill for the same 8192 tokens: ref 1311.2 ms, px4 1303.1 ms, **px8 1283.3 ms** (1.5%).

### Gate 2 — needle in a haystack, 66,901 tokens, needle at 50% depth — **FAIL, and not because of px8**

First attempt was **invalid and is recorded as such**: a haystack of repo source + markdown with
the question appended as raw tokens made the **bf16 reference itself** degenerate into repeating
the question. A gate whose reference fails says nothing about any kernel. Fixed the way the
earlier server-side needle test in this campaign did it — benign repetitive filler so the needle
is the only salient fact, and the model's **chat template** (this is an instruct model; a raw
continuation prompt just continues).

| arm | 66.9k needle | prefill ms | generated stream |
|---|---|---|---|
| `ref` bf16 KV | **RETRIEVED** `PELICAN-7734` | 15643.5 | — |
| `px0` fp8 KV, no fp8 mma | **MISS** | 13843.3 | `818 496 215646 138 561 138 ...` |
| `px4` fp8 KV, fp8-mma | **MISS** | 10947.3 | *byte-identical to px0* |
| **`px8`** fp8 KV, e4m3 P.V | **MISS** | **10781.6** | *byte-identical to px0* |

**All three fp8-KV arms produce the identical 96-token stream, and all three miss.** `px0` has no
fp8 mma in it at all, so the failure cannot be the fp8 QK, cannot be the e4m3 P.V, and cannot be
px8. It is upstream of the prefill flash arm, in the fp8-KV path itself (cache format and/or the
fp8 decode arms, which this experiment does not separate).

So: **px8 fails the 69k needle gate, exactly as the shipped incumbent fails it, for a reason px8
did not introduce.** Recorded as a FAIL rather than dressed up — but the correct target of the
failure is plow's mixed fp8-KV at long context, not this PR. That is a pre-existing defect this
gate happened to surface, and it deserves its own investigation.

### Gate 2b — the same needle at 7,826 tokens — **all four arms RETRIEVE**

Without this control the 66.9k failure could have been a broken fp8-KV build rather than a
long-context degradation. It is the latter:

| arm | 7.8k needle | 66.9k needle | prefill ms @ 7.8k |
|---|---|---|---|
| `ref` bf16 KV | **RETRIEVED** | RETRIEVED | 1315.6 |
| `px0` fp8 KV, no fp8 mma | **RETRIEVED** | MISS | 1670.3 |
| `px4` fp8 KV, fp8-mma | **RETRIEVED** | MISS | 1305.3 |
| `px8` fp8 KV, e4m3 P.V | **RETRIEVED** | MISS | **1285.4** |

**Every fp8-KV arm retrieves at 7.8k and loses it by 66.9k.** So the defect is a context-scaling
property of the fp8 KV cache, not a wiring or build error, and it sits somewhere between those two
lengths. Nothing about it is specific to any prefill flash arm.

This run also gives a second, independent greedy-divergence point on a different prompt, and px8
does BETTER than the incumbent on it:

| pair | first divergence (of 96) |
|---|---|
| px0 vs bf16 | token 10 |
| px4 vs bf16 | token 10 |
| **px8 vs bf16** | token **27** |

Two prompts, two results: px8 ties the incumbent at 8k prose (15/15) and beats it 27 vs 10 on the
8k needle. Nothing suggests the e4m3 P is the weak link.

Note `px0` is *slower than bf16* at 7.8k (1670.3 vs 1315.6 ms): the PIPE=0 synchronous-staging fp8
arm loses more to its inline dequant than it gains from halved KV bytes at short context. The
default fp8-KV build is therefore a short-context regression as well as a long-context one.

### Result 12 — the PX-7 doubt from Result 1, now measured

Result 1 warned that PX-7's "the px4 fp8 arm recovered only 10% of the quadratic term" may have
been measured on a cubin containing no fp8 arm. `px0` vs `px4` is exactly that A/B — same packet,
same fp8 KV cache, differing **only** in the prefill flash arm:

| | prefill ms @ 66.9k | vs px0 |
|---|---|---|
| px0 (PIPE=0, what the default fp8-KV cubin ships) | 13843.3 | 1.00x |
| px4 (PIPE=1 fp8-mma, `PLOW_FP8_KV_FASTPF=ON`) | 10947.3 | **0.791x** |
| px8 (+ e4m3 P.V) | 10781.6 | 0.779x |

**Turning on an arm that already exists in the tree takes 21% off total prefill at 67k**, against
the 1.5% px8 adds on top of it. Derived (not directly measured): attributing the px4->px8 delta
to the flash op using this note's measured 0.716 op ratio puts the flash op at ~583 ms inside
px4's 10947 ms (5.3%), hence ~3479 ms inside px0's 13843 ms (25%). Scaled quadratically to 127k
that is ~12.5 s, which is the same order as PX-7's 16.48 s — **consistent with PX-7 having
measured the px0-class arm all along.**

**The actionable ordering is therefore the reverse of what this PR assumed:** flipping
`PLOW_FP8_KV_FASTPF` to ON is a one-line build change worth ~21%, and px8 is worth ~1.5% on top
of that. px8's 1.40x is real but it applies to an op that the FASTPF arm has already shrunk to
~5% of prefill. Neither is blocked by the other; the cheap one should go first.

## Gates

| gate | result |
|---|---|
| smem budget verified from source, not from PX-7's table | **PASS** — and PX-7's table was wrong (89,104 B, not 85,248) |
| registers / spills from `-Xptxas -v`, not guessed | **PASS** — 238 shipped, 240 with px8, 0 spills |
| 8-bit ldmatrix fragment map measured, not assumed | **PASS** — dumped per lane, `layout` mode |
| ldmatrix path validated against an independent oracle | **PASS** — LDS.8 gather agrees to every printed digit |
| shipped kernel measured, not just a bench copy | **PASS** — arm Bp calls `d_flash_prefill_px8` |
| numerics vs f32 reference | **PASS** — rms 2.29e-03 vs the baseline's 1.89e-03 (1.21x) |
| bf16 prefill cubin unchanged | **PASS** — byte-identical to the HEAD build |
| GPU exclusive | **ENFORCED** — `gpulease`. The `foreign-during` WARN and the resulting `rc=76` on these runs are the known `foreign()`-compares-against-`$$` bug (fixed in `a669e52`): the wrapper flags its own CUDA child. The GPU read 2 MiB / 0% immediately before this run acquired, so there was no real contention. |
| per-tile time flat 8k->128k (the BQ=64 premise) | **PASS/REFUTES** — flat to 0.9%, so BQ=64 buys nothing |
| ablation bits actually ablate | **FAILED then FIXED** — `#if` on a template arg; see Result 9 |
| result reproduces on an independent run | **PASS** — 0.716/0.719 vs 0.718/0.720, inside the 0.7% noise floor |
| **greedy-token parity at >= 8k against a bf16-KV run** | **PASS** — px8 and the px4 incumbent both first diverge from bf16 at token **15** of 64; px8 tracks px4 to token 20. Result 11. |
| **needle test at 69k** | **FAIL** — but `px0` (fp8 KV with NO fp8 mma anywhere) misses with a byte-identical stream, so the failure is upstream of the prefill flash arm and px8 did not introduce it. Result 11. |
| needle gate has a working reference | **FAILED then FIXED** — the first prompt made the bf16 REFERENCE degenerate; a gate whose reference fails is not a gate. Rebuilt with benign filler + the chat template. Result 11. |
| the needle failure is attributable | **PASS** — the `px0` control is what makes it attributable; without it "px8 fails the needle" would have been a false accusation. |
| needle failure is context-dependent, not a broken build | **PASS** — all four arms RETRIEVE the same needle at 7.8k; only bf16 still does at 66.9k. Gate 2b. |
| second greedy-divergence point, different prompt | **PASS** — px8 holds **27** tokens vs bf16 where px4 and px0 both hold 10. Gate 2b. |

`PLOW_NV_FA_FP8PV` stays **default 0**. Not because of the greedy gate, which it passes, but
because the arm it builds on (`PLOW_FP8_KV_FASTPF`) is itself off by default, and because the
fp8-KV path loses 67k needle retrieval for reasons nobody has attributed yet. Enabling px8 while
that is open would be enabling the small win on top of an unfixed defect.

**Recommended order, from these measurements:**
1. investigate why mixed fp8-KV loses the needle at 67k (`px0` reproduces it with no fp8 mma —
   this is not a kernel-arm question);
2. flip `PLOW_FP8_KV_FASTPF` ON — **21% off total prefill at 67k**, one line, arm already in tree;
3. then px8, worth a further ~1.5% end-to-end (1.40x on an op the FASTPF arm has already shrunk
   to ~5% of prefill).
