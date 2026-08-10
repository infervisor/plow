# GLM-5.2 decode: where the token goes, and two aiter-shape arms for the expert GEMVs

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **CDNA3-SPECIFIC (and see the VOID banner)** — sections 5/5b/5c are void for an object-routing error unrelated to arch.

> **>>> CORRECTION 2026-08-08 — EVERY MEASURED TABLE BELOW IS VOID (sections 5, 5b, 5c). <<<**
>
> All three arms in this file were built with `PLOW_ROWS_ONLY=interp_decode_fp8`, i.e. into
> `interp_decode_fp8{,kv}{,_gq}.elf`. **A GLM-5.2 packet never loads those objects.**
> `Variant::detect` (exec/amd.rs) selects the object by scanning opcodes for `GemvFp8` and the
> fp8-KV flash ops; GLM-5.2's fp8 is BLOCK-scaled (`GemvFp8Blk`, `MoeExpertDownFp8Blk`, ...) and
> its MLA is bf16, so the blob contains ZERO of those opcodes, the variant detects as `Bf16`, and
> decode runs on **`interp_decode_gq.elf`**. The arms and the ablation were compiled into objects
> that were never opened, which is why the arm read NULL, why the packet busy time "did not move
> AT ALL", and why "deleting one hundred percent of the work" cost nothing — nothing was deleted.
>
> Rebuilt into `interp_decode_gq.elf` (`PLOW_ROWS_ONLY=interp_decode`):
> **`lgx2` is -8.2% and token-identical** (28.905 -> 26.534 ms, three interleaved rounds), and
> **`ABL=2` is -11.8%** (25.494 ms) — so the DOWN packet's 47.4 us is ~37.6 us of REAL KERNEL and
> ~1.7 us of packet protocol, not "entirely packet protocol". §5c's conclusion ("the MoE-decode
> kernel-rate route is CLOSED") is exactly backwards: it is the largest kernel lever in decode.
>
> **Section 4's ISA audit, section 3's bit-identity argument, the off-device reduction and
> row-coverage checks, and section 6's serve gate all STAND.** They are what make the arm
> shippable. Only the numbers in 5 / 5b / 5c and the verdict drawn from them are wrong.
>
> Full record, per-unit protocol costs, and the traced decomposition:
> `perf-data/plow-gfx942/glm52-packet-protocol-xcd.md`.

2026-08-08, branch `decode-gemv` (base `worktree-glm52-bringup` @ 9468fdb), box
[[plow-devbox-is-gfx942]] (8x MI300X, 304 CU, ROCm 7.2.4). Instruments:
`PLOW_TRACE_RAW` + a per-op reducer over the raw `PlowTraceRec` dump (grouped by
`inst`, NOT `pc`), `llvm-objdump --mcpu=gfx942`,
`perf-data/plow-gfx942/probes/asm_loops.py`, `scripts/bench_speed.sh`.

## 1. The span census — ONE decode step, measured, not guessed

`plowrt amd-bench --tp 8 --ctx 1024 --steps 24` on the best asset
(`/workspace/assets/gfx942/glm52-tp8-best`, objects `hsaco_glm16`, bound
checkpoint, `PLOW_MLA_PF_V2=1`): **29.798 ms/token, all 8 ranks token-identical**
— i.e. the campaign's 29.7 ms reproduced. The last launch's trace, 2678 packets:

| op | npkt | span us | % | us/pkt | busy CU-us | wait CU-us | WG/pkt |
|---|--:|--:|--:|--:|--:|--:|--:|
| **MoeExpertDownFp8Blk** | 600 | 28500 | 44.3 | **47.5** | 834557 | 639661 | 32 |
| **MoeExpertGluFp8Blk** | 600 | 14518 | 22.6 | 24.2 | 333361 | 210434 | 32 |
| GemvQkv | 156 | 3112 | 4.8 | 20.0 | 380799 | 1721839 | 147.5 |
| FlashMlaDecode | 78 | 2616 | 4.1 | 33.5 | 304373 | 469774 | 128 |
| Gemv (shared-expert down, N=6144 K=256) | 75 | 2603 | 4.0 | **34.7** | 111730 | 60561 | 48 |
| XReduce | 156 | 2259 | 3.5 | 14.5 | 24140 | 65199 | 12 |
| Gemv (router, N=256 K=6144) | 75 | 1649 | 2.6 | 22.0 | 382313 | 651584 | 304 |
| Gemv (o_proj, N=6144 K=2048) | 78 | 1610 | 2.5 | 20.6 | 389169 | 884259 | 304 |
| GemvGlu (shared g+u) | 75 | 1506 | 2.3 | 20.1 | 39746 | 18341 | 48 |
| MlaMergeFold | 78 | 1420 | 2.2 | 18.2 | 81773 | 315829 | 64 |
| RmsNorm | 235 | 1199 | 1.9 | 5.1 | 1199 | 12170 | 1 |
| MoeRouterTopk | 75 | 1107 | 1.7 | 14.8 | 1107 | 900 | 1 |
| HeadNormRope | 156 | 667 | 1.0 | 4.3 | 667 | 3784 | 1 |
| MoeCombine | 75 | 508 | 0.8 | 6.8 | 5017 | 54631 | 12 |
| AddNorm / Residual | 156 | 768 | 1.2 | 4.9 | 768 | 6876 | 1 |
| GemvFp8Blk + DenseGluFp8Blk (3 dense layers) | 6 | 184 | 0.3 | 30.7 | 44576 | 45126 | 304 |
| lm_head Gemv (N=19360 K=6144) | 1 | 78 | 0.1 | 78.0 | 22619 | 17263 | 304 |

Span sums to 64.4 ms against a 29.8 ms step because the 8 routed experts run
CONCURRENTLY on disjoint 32-CU slices (`GLM_MOE_CORESIDENT=2`, `GLM_SHARED_CUS=48`
→ `routed_w = (304-48)/8 = 32`). The wall picture is the per-layer timeline
(layer 40, 358.3 us end to end; 29.798 ms / 78 layers = 382 us, so this layer is
representative):

```
#1345 GemvQkv       b=146  start   6.6  end  30.2  span 23.5
#1347 GemvQkv       b=149          42.1       59.2       17.1
#1351 FlashMlaDecode b=128          71.6      106.3       34.7
#1352 MlaMergeFold  b= 64          112.6      131.0       18.4
#1353 Gemv o_proj   b=304          133.4      156.7       23.3
#1354 XReduce       b= 12          161.8      175.2       13.4
#1356 Gemv router   b=304          182.5      204.7       22.1
#1357 MoeRouterTopk b=  1          206.3      220.2       13.8
#1358 GemvGlu shared b=48          193.9      213.9       20.0   (co-resident)
#1359 Gemv  shared down b=48       216.0      250.2       34.2   (co-resident)
#1360..#1375  8 x (MoeExpertGlu, MoeExpertDown) on 32-CU slices:
      glu   starts 221.6 .. 270.6      spans 21.8 .. 26.7
      down  starts 247.1 .. 294.0      spans 38.0 .. 55.0
#1376 MoeCombine    b= 12          332.8      339.8        7.1
#1377 XReduce       b= 12          340.9      353.4       12.5
#1378 Residual      b=  1          354.2      358.3        4.1
```

**The routed-MoE section is 221.6 → 332.0 = 110 us of a 358 us layer, and DOWN is
38–55 us of every expert's chain — the largest single packet in the layer.** The
staggered expert starts (221.6 for slot 0, 270.6 for slot 7) are claim-path
latency, not the kernels, and are NOT what this work attacks.

## 2. Why DOWN is the top consumer — a shape defect, not a rate one

`wave_dot_fp8_blk` gives ONE output row to a whole 64-lane wave and hands lane `L`
the 16 fp8 at `k = 16*L`. GLM-5.2 at TP8 routes DOWN with **K = I_moe = 256**:

* only lanes 0..15 are in range. 48 of 64 lanes convert and dot bytes the buffer
  descriptor returned as zero;
* the wave's `buffer_load_dwordx4` covers **256 useful bytes of a 1024-byte
  request**;
* `wave_sum`'s first two butterfly offsets (32, 16) add exact `+0.0`;
* `nchunk = ceil(256/1024) = 1`, so there is no chunk loop to prefetch — the whole
  row is ONE dependent `load → wait → dequant → dot → reduce` chain with ONE load
  in flight, repeated `H/(nblk*PLOW_WAVES) = 6144/256 = 24` times per wave.

Traced cost: 1613 CU-us per 32-CU expert slice for 49 KB of weights per workgroup
≈ **1 GB/s per CU**. For scale, o_proj (N=6144, K=2048) moves EIGHT TIMES the
bytes in 23.3 us against DOWN's 47.5.

The header of `wave_dot_fp8_blk` records that a same-stream UN-deep prefetch was
tried here and regressed; that is consistent — at `nchunk = 1` there is nothing to
prefetch. The lever is the LANE MAP and cross-ROW depth, which is exactly what the
MXFP4 twin already ships (`wave_dot_mxfp4_lg2`, 1.80x) and block-fp8 was excluded
from only because GLM-5.2 had to stay byte-identical.

## 3. What changed (commit 4d69fca) — two opt-in arms, default OFF

**`PLOW_MOE_DEC_LG=1`** — narrow-K LANE-GROUP DOWN (`moe_down_lg_fp8_blk`,
op_moe.h). Split the wave into `RG=4` row-groups of `LPG=16` lanes; group `g` owns
its own output row and lane `sub = lane%16` owns `k = 16*sub`, so at K=256 every
lane is live and ONE load fetches `RG` whole CONSECUTIVE rows fully coalesced.
`UNR=6` row-batches are issued before any is consumed → **24 rows in flight**, and
at GLM's geometry `ng = 6144/24 = 256` equals the wave count exactly, so every wave
runs ONE balanced pass. Guarded to `K <= LPG*16` and a 16-multiple K; anything
wider falls through to the shipped walk. Weight rows are plain 16-byte global
vectors, not buffer loads, because the row index is now lane-varying and a
`buf_rsrc` on a divergent base compiles to a readfirstlane waterfall.

**`PLOW_MOE_DEC_X2=1`** — gate|up as ONE loop with both weight streams in flight
and the activation fragments read once (`wave_dot_fp8_blk_x2`), the fp8 twin of the
shipped `wave_dot_mxfp4_x2`.

**Numerics.** Both are bit-identical by construction, by the same argument
`wave_dot_mxfp4_lg2` records: the set of `(lane → k)` fragments per row is
unchanged, each row's partials are summed by the SAME xor-butterfly (the 64-lane
`wave_sum`'s leading offsets 32 and 16 were adding lanes that held exact `+0.0`),
and each accumulator in the paired GLU sees the same terms in the same order. The
only representable difference is the SIGN of a zero result (`-0.0 + 0.0 = +0.0`),
which the f32 `MoeCombine` that consumes `part` cannot distinguish. Which wave owns
which row also moves, and that is unobservable — `part_slot[h]` is written by
exactly one lane either way.

Both halves of that were checked off-device rather than asserted:

* **reduction**: 200,000 random 16-value f32 vectors (magnitudes spanning 1e-6 to
  1e6), 64-lane `wave_sum` over `[a, 48 zeros]` vs the 16-lane butterfly over `a`
  — **0 differing bit patterns out of 200,000**;
* **row coverage**: the `(slice, wave, f, u, grp)` walk simulated for
  `(H, nblk)` in {(6144,32), (6144,304), (6143,32), (100,32), (6144,1), (24,32)}
  — every output row written **exactly once**, including the ragged tails.

The grouped opcodes 48/49 inherit both arms for free: `d_moe_group_{glu,down}_fp8_blk`
call the per-slot bodies in their default (slot-outer) branch. The
`PLOW_MOE_GROUP_FLAT` branch has its own inlined body and does NOT get them — it is
a different non-default axis and GLM does not emit it (`GLM_GROUP` unset → per-slot
ops 45/46, which is what the trace shows).

**Default-OFF is verified, not asserted.** With both unset the control object
reproduces the canonical `hsaco_glm16/interp_decode_fp8_gq.elf` to **one
instruction in 138,288** (`v_lshlrev_b32_e32` 9150 vs 9151 — hipcc is not byte
reproducible, so instruction-mix equality is the available test), and
VGPR/AGPR/LDS/spill are **108 / 0 / 30,776 / 0** in EVERY arm.

## 4. ISA audit — the shape reached the machine code

`llvm-objdump -d --mcpu=gfx942` on the **shipped** `interp_decode_fp8_gq.elf`
(not a standalone TU), inner loop extracted with `probes/asm_loops.py`.
DOWN is normalised per OUTPUT ROW (the shipped loop makes 1 row/iteration, the LG
loop makes `RG*UNR = 24`); GLU per chunk-PAIR (the shipped body runs two separate
6-chunk loops, one per stream).

| | insts | VALU | gload | full `vmcnt(0)` | partial waits |
|---|--:|--:|--:|--:|---|
| DOWN shipped, per row | 142 | 122 | 4.00 | 2.00 | — |
| **DOWN `PLOW_MOE_DEC_LG`, per row** | **32.7** | **26.8** | **0.58** | **0.375** | vmcnt(3) x3, vmcnt(2), vmcnt(1) x3 per 24 rows |
| GLU shipped, per chunk-pair | 282 | 244 | 8 | 2.00 | — |
| **GLU `PLOW_MOE_DEC_X2`, per chunk-pair** | **241** | **213** | **6** | **1.00** | vmcnt(1) x3, vmcnt(2) |

DOWN: 4.3x fewer instructions per row, 4.6x less VALU per row, and the memory pipe
now carries 24 rows across three partial waits instead of draining twice per row —
the one discipline `glm52-asm-innerloop-diff.md` identifies as the real gap to
aiter ("13–39 sustained outstanding loads with partial waits", §2).

Both depths were SWEPT on the ISA, not guessed (tables at the two `#define`s):

* `LG_UNR` 2/3/4/6 → 0.63/0.50/0.44/**0.375** full drains per row, monotone,
  VGPR flat at 108. 6 is also the value that makes `ng` exactly 256 at GLM's
  geometry.
* `X2_UN` 1/2/3 → **1**/4/3 full drains per chunk-pair.

**A load-bearing negative result on method.** The X2 depth does NOT transfer from
a standalone TU. Compiled alone (`/tmp/dg_isa_probe.hip`, same flags), `UN=3`
produces the textbook aiter body — **18 loads issued, ZERO `vmcnt(0)`, every wait
partial (`vmcnt(11)`, `(10)`, `(9)`, `(8)`, `(7)`, `(6)` x2, `(5)`, `(3)`)**. The
SAME source inside the 108-VGPR megakernel, whose register budget is the union
over every arm it carries, cannot hold `2*UN` fp8v16 fragments live and
re-serializes — the drains come straight back. **An inner-loop shape measured in a
probe TU is not evidence about the megakernel; audit the shipped object.**

## 5. A/B — interleaved, on the box

THREE ARMS, all built by the SAME `scripts/build_gfx942.sh` invocation differing
only in the two `-D` flags, all 28 objects otherwise the canonical `hsaco_glm16`
set, all pointed at the SAME `model.pkt` (this is a kernel-only axis — the blob
does not change):

* `ctl`   — both axes unset (the shipped bodies)
* `lg`    — `PLOW_MOE_DEC_LG=1`
* `lgx2`  — `PLOW_MOE_DEC_LG=1 PLOW_MOE_DEC_X2=1`

Rounds are A/B/C/A/B/C/A/B/C so the control's own round-to-round spread is
measured in the same session as the treatment (the box has ±20% DVFS drift on
absolute walls; only the interleaved spread is a usable noise floor).
`scripts/bench_speed.sh`, `IN_LENS="1024 4096" CONCS=1 NPROMPT=8 OUTLEN=128`,
lock held throughout, `rocm-smi` 0% on acquire, no foreign `plowrt` resident.

**TPOT, ms/token:**

| in_tok | arm | r1 | r2 | r3 | mean | ctl spread | vs ctl |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1024 | ctl | 29.93 | 29.67 | 29.72 | 29.77 | **0.26 (0.9%)** | — |
| 1024 | lg | 29.72 | 29.66 | 29.87 | 29.75 | | −0.08% |
| 1024 | lgx2 | 29.63 | 29.66 | 29.72 | 29.67 | | −0.35% |
| 4096 | ctl | 31.99 | 31.99 | 31.93 | 31.97 | **0.06 (0.2%)** | — |
| 4096 | lg | 31.94 | 31.92 | 31.90 | 31.92 | | −0.16% |
| 4096 | lgx2 | 31.90 | 31.92 | 31.95 | 31.92 | | −0.15% |

`amd-bench --tp 8 --ctx 1024 --steps 24`, interleaved ctl/lgx2/ctl/lgx2, all ranks
token-identical every step: **29.782 / 29.767 / 29.736 / 29.765**.

**Verdict: NULL.** At 1024 the whole treatment effect (−0.35%) is a third of the
control's own round-to-round spread (0.9%). At 4096 the spread is tighter (0.2%)
and the effect (−0.15%) is smaller than it. Nothing here beats its own noise
floor, and `amd-bench` — a different instrument on the same objects — reads the
two arms as indistinguishable. **A 4.3x cut in the DOWN inner loop bought
approximately nothing.**

### 5b. Why — the packet busy time did not move AT ALL

The wall being flat could be scheduling. It is not: the trace says the KERNEL did
not get faster either. Same census, ctl vs lgx2, two reps each, `busy` = summed
per-workgroup `t_end - t_ready` over the whole step:

| op | ctl rep0 | ctl rep1 | lgx2 rep0 | lgx2 rep1 |
|---|--:|--:|--:|--:|
| MoeExpertDownFp8Blk busy (CU-us) | 830208 | 832418 | **833188** | **831065** |
| MoeExpertDownFp8Blk us/pkt | 47.2 | 47.4 | **47.4** | **47.3** |
| MoeExpertGluFp8Blk busy (CU-us) | 333475 | 333540 | **332893** | **332837** |
| MoeExpertGluFp8Blk us/pkt | 24.2 | 24.2 | **24.1** | **24.1** |

**0.1% on DOWN and 0.2% on GLU — both inside the ctl-vs-ctl rep spread (0.27%).**

The arm IS being taken. The dispatch is
`d_moe_expert_down_fp8_blk(..., in->i[1]=H=6144, in->i[2]=I_moe=256, ..., in->i[6]=enc=1)`
(interp.hip:2183), `PLOW_MOE_ENC_FP8BLK` is `1u`, and the guard is
`enc == 1 && I_moe <= 256 && (I_moe & 15) == 0` — all true. The loop is in the
object at 0x1a1cc and issues **ten `global_load`s back to back before its first
`s_waitcnt vmcnt(0)`**.

So between the two arms, ALL FOUR of the things the aiter diff nominates changed
by 4x or more, and the packet did not move:

| | shipped | LG arm |
|---|---|---|
| live lanes | 16 of 64 | 64 of 64 |
| VALU per output row | 122 | 26.8 |
| loads outstanding | 1 | 10 issued before the first drain, 24 rows/pass |
| weight locality per wave | 24 rows at 64 KB stride | 24 CONSECUTIVE rows (6 KB block) |
| f32 result stores per wave | 24, each its own cache line | 24 covering 2 cache lines |

Byte counts, request counts and addresses are identical between the arms (same
1.5 MB per expert, same 96 cache-line requests per wave). What is left is the
memory system's service rate for this packet, and the two DOWN implementations ask
it for exactly the same thing.

For scale, the same trace prices the two MoE ops against each other on the same
machine at the same instant: GLU moves 96 KB per workgroup in 17.4 us (5.5 GB/s
per CU); DOWN moves 49 KB per workgroup in 43.2 us (1.1 GB/s per CU) — **half the
bytes, 2.5x the time, 5x worse per byte** — and none of that ratio is explained by
anything visible in the instruction stream.

### 5c. The ablation — the DOWN packet costs the same when it does NOTHING

`PLOW_MOE_DEC_ABL` (wrong output by construction, instrument only): **1** keeps the
row walk and the `part` store and deletes every load and the dot; **2** retires the
op entirely — the kernel resolves the expert, then returns. The objects are real,
not a silent fallback: `d_moe_expert_down_fp8_blk` is **437 → 198 → 172**
instructions and its 8 weight/scale `global_load`s are gone (the 3 that remain are
the routing-table indirection in the header). Same `model.pkt`, same 24 other
objects, `amd-bench --tp 8 --ctx 1024 --steps 24`:

| arm | DOWN busy (CU-us) | DOWN us/pkt | GLU busy | step summed span | ms/token |
|---|--:|--:|--:|--:|--:|
| ctl (full kernel) | 830208 / 832418 | 47.2 / 47.4 | 333475 | 64258 / 64212 | 29.782 / 29.736 |
| lgx2 (arms on) | 833188 / 831065 | 47.4 / 47.3 | 332893 | 64224 / 64126 | 29.767 / 29.765 |
| **ABL=1 (no loads, no dot)** | **832498** | **47.3** | 334050 | 64209 | **29.742** |
| **ABL=2 (op retired)** | **832654** | **47.4** | 332608 | 64154 | **29.742** |

**Deleting one hundred percent of the work in the single largest op of the decode
program — 44.3% of the step's summed packet span, 1.5 MB of fp8 per expert per
layer — moves its own packet by 0.3% and the token by 0.0%.**

That is the whole explanation for section 5's null, and it is much stronger than
"the arm did not help". The DOWN packet's 47.4 us is **entirely packet protocol**:
gate poll, per-XCD L2 maintenance, the exit rendezvous, the counter release, and
above all the claim latency that staggers the 8 co-resident expert slots (the
layer-40 timeline shows slot 0's GLU starting at 221.6 us and slot 7's at 270.6 —
49 us of pure dispatch skew, which is itself larger than the DOWN kernel it
precedes). `busy` = `t_end - t_ready` per workgroup therefore counts the wait at
the packet's exit rendezvous for its slowest CLAIMED sibling, which is why making
every wave 4x faster is invisible: the waves just wait longer.

**Conclusion — the MoE-decode kernel-rate route is CLOSED at audit grade.** No
instruction-level work on `d_moe_expert_{glu,down}_fp8_blk` can pay while the
packet floor stands, and this is now measured rather than inferred: the floor was
priced by deleting the kernel, not by modelling it. This independently confirms and
sharpens the campaign's standing thesis (`glm52-dsa-sparse-b3.md`: "29.7 − 12.8
traced floor = 16.9 ms is gate/dispatch stall") — the claim-path rebuild is not one
of several options for decode, it is the ONLY one.

## 6. Serve gate

`plowrt serve` on each arm's asset (same `model.pkt`, only the 4 decode-fp8
objects differ), real prefill, temp 0. **All three arms PASS, and the three
answers are character-identical across arms:**

```
--------- GATE arm=ctl | lg | lgx2      (identical output, all three)
model: glm-5.2-fp8
Q: What is the capital of France?
A: The capital of France is Paris.
Q: Compute 17*23
A: 17 * 23 = 391
Q: Name the chemical symbol for gold and one common use
A: * **Chemical Symbol:** Au
   * **Common Use:** Jewelry making (though it is also heavily used in
     electronics and dentistry).
```

`bench_speed.sh`'s own 'Paris' gate also passed on all 9 battery runs, and
`amd-bench` reported all 8 ranks token-identical on every step of all 4 runs. The
bit-identity claim of section 3 is therefore supported end to end as well as by
construction.

## 7. Exact recipe

```bash
export PATH=/nix/var/nix/profiles/default/bin:/root/.nix-profile/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib
export ROCM_PATH=/opt/rocm-7.2.4 HIP_PATH=/opt/rocm-7.2.4 ROCM_HOME=/opt/rocm-7.2.4
export PLOW_HIPCC=/opt/rocm-7.2.4/bin/hipcc

# objects (OUTSIDE nix). Only the decode-fp8 rows change; copy them over a full set.
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_ROWS_ONLY=interp_decode_fp8 \
    PLOW_MOE_DEC_LG=1 PLOW_MOE_DEC_X2=1 bash scripts/build_gfx942.sh <outdir>

# blob: UNCHANGED — this is a kernel-only axis, the best-config emit still applies
#   GLM_FULL=1 PLOW_MLA_PREFILL=full GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 \
#   GLM_SHARD_HEAD=1 PLOW_MLA_PF_V2=1 plowc --emit devblob \
#     --hf-dir /workspace/models/GLM-5.2-FP8 --gpu MI300X --arch gfx942 \
#     --num-gpus 8 --max-ctx 73728

# census
PLOW_MLA_PF_V2=1 PLOW_TRACE_RAW=/tmp/t.bin nix develop . --command \
  ./target/release/plowrt amd-bench --blob <asset>/model.pkt --hsaco <asset>/hsaco \
    --checkpoint /workspace/models/GLM-5.2-plow-lite --tp 8 --ctx 1024 --steps 24

# serve / bench (needs PLOW_MLA_PF_V2=1: the blob carries the causal KV-split)
PLOW_MLA_PF_V2=1 IN_LENS="1024 4096" CONCS=1 NPROMPT=8 OUTLEN=128 \
  PLOWRT_BIN=$PWD/target/release/plowrt bash scripts/bench_speed.sh <asset> 8195 auto
```

`plowrt` must be built `--features hsa`. `amd-bench --checkpoint` needed the clap
id fix in 34eaf78 to run at all.

## 8. What is kept, and what to do next

**Kept, default OFF, correct**: `PLOW_MOE_DEC_LG` and `PLOW_MOE_DEC_X2` are
bit-identical, ISA-verified, resource-neutral (108 VGPR / 0 spill), serve-gated on
three canonical prompts, and are the RIGHT shape for these two kernels — 4.3x
fewer instructions per DOWN output row, full lane occupancy, 10 loads outstanding
where there was 1, and consecutive-row locality. They are worth nothing today
because the thing in front of them costs 47 us no matter what. Turn them on the
day the claim path stops setting the packet cost; do not re-derive them.

**Do NOT re-try** (this file is the record): deeper unrolls of either arm — swept
on the ISA, `X2_UN` past 1 and `LG_UNR` past 6 are register-bound in the
megakernel; and any further instruction-level work on the decode expert bodies at
all, for the reason in 5c.

**Next, in order:**

1. **The claim path.** 49 us of dispatch skew across 8 co-resident expert slots
   and a 47 us floor on an EMPTY packet. Per-CU precomputed schedules + stealing,
   as `glm52-dsa-sparse-b3.md` proposes. Everything else in decode is downstream
   of this number.
2. **Packet COUNT, not packet rate.** If a packet costs ~47 us regardless of its
   body, the lever is emitting fewer of them. GLM decode runs 2678 packets/step,
   16 of them per layer just for the routed experts. `GLM_GROUP=1` already collapses
   the 2*tk per-slot packets into 2 grouped ones (ops 48/49) and is NOT in the best
   config — it should be re-priced against this measurement, and it inherits both
   arms for free through the slot-outer branch. That is the first thing to try.
3. The shared-expert down `Gemv` (b=48, N=6144, K=256, 34.7 us/pkt) has the SAME
   narrow-K lane defect in the bf16 `gemv_rows` body, and the same verdict should
   be assumed to apply until the floor moves. It is fully hidden under the routed
   experts today.

## COROLLARY TESTED AND REFUTED (coordinator, 2026-08-08): GLM_GROUP=1 loses by 15-16%

This report closed by naming `GLM_GROUP=1` — collapsing the 16 routed-expert packets per
layer into 2 (ops 48/49) — as "the first thing to try against this measurement", on the
reasoning that if every packet costs ~47 us of protocol regardless of body, then deleting 14
packets per layer must pay. It does not. Two corrections:

**(1) The knob was not untried.** `crates/devgen/src/kda.rs:312-317` cites
the design notes §6g-KNOBS measuring `GLM_GROUP=1` at **38% fewer ops for +2.88 ms**,
and both the KDA and K3 emitters explicitly justify their own merges as *not* being this
shape. (The design notes are local-only, so the primary record is the code comment.)

**(2) Re-tested anyway on the current config — it loses harder.** 3 interleaved rounds on
`hsaco_glm17`, blob = round-2 final recipe + `GLM_GROUP=1`, port 8196:

| ctx | TPOT final r1/r2/r3 | TPOT group r1/r2/r3 | Δ | TTFT final | TTFT group |
|---|---|---|---|---|---|
| 1024 | 28.83 / 28.84 / 28.85 | 33.51 / 33.49 / 33.55 | **+16.2%** | ~349 | ~350 |
| 4096 | 31.15 / 31.16 / 31.17 | 36.00 / 35.81 / 35.87 | **+15.1%** | ~1002 | ~1000 |

Both arms' spreads are ≤0.2 ms; the gap is 100× that. Prefill is untouched, as expected —
the knob only restructures the decode expert loop.

### Why this matters more than a failed knob

The re-test rules out the simplest reading of this report's own ablation. If the ~47 us were
a fixed per-packet toll, removing 14 of 16 packets per layer would have to pay. It cost 15%
instead. **Therefore the 47 us is not a serial toll — it is already largely OVERLAPPED across
the 8 co-resident expert slots** (`GLM_MOE_CORESIDENT=2` plus the slot-outer branch), and
collapsing the packets into one loop *serializes what was concurrent*, exposing the protocol
cost that the concurrency had been hiding. That is the same failure mode the knob contract
recorded: merging along a LOOP dimension surrenders CU-slice concurrency, whereas merging
along an OUTPUT dimension (what the KDA P1-P4 fuse does) makes the op wider without
serializing anything.

**Consequence for the claim-path work:** the target is not "fewer packets". It is the 49 us
of *dispatch skew* across the co-resident slots — the ramp during which some slots are
already working and others have not yet been claimed. Packet COUNT is not the lever; packet
START ALIGNMENT is. A claim-path rebuild should be judged on whether it compresses that
ramp, not on how many packets it eliminates.

## RETRACTION (coordinator, 2026-08-08): THE ABLATION IN THIS REPORT MEASURED AN OBJECT THE RUN NEVER LOADS

Everything in this report that rests on the ablation table is **withdrawn**. The mechanism,
found by the `protocol-xcd` agent and verified independently at `crates/plowrt/src/exec/amd.rs:276`:

`Variant::detect` chooses the interpreter object by scanning opcodes for `DevOp::GemvFp8` and
the three fp8-KV flash ops. **GLM-5.2's fp8 is BLOCK-SCALED** (`GemvFp8Blk`,
`MoeExpertDownFp8Blk`, …) **and its MLA is bf16**, so the blob contains **zero** matching
opcodes, `detect` falls through to `Variant::Bf16`, and decode runs on
**`interp_decode_gq.elf`** — not `interp_decode_fp8_gq.elf`.

Both the optimization arms and the ablation were built into
`PLOW_ROWS_ONLY=interp_decode_fp8`. **"Deleting 100% of the work moves the token 0.0%" was
nothing being deleted.** The arms measured null for the same reason.

Rebuilt into the object that is actually loaded:

| arm | ms/token |
|---|---|
| control | 28.90 |
| `ABL=2`, DOWN retired | **25.49 (−11.8%)** |

So the 47.4 µs packet is **37.6 µs of real kernel** plus ~1.7 µs of protocol, not 47 µs of
protocol. And `PLOW_MOE_DEC_LG` — the lane-group DOWN rewrite this report built and measured as
a null — is worth **−7.6% @1k / −7.3% @4k**, 50× the control spread, serve-gated
character-identical including a ~14.7k-token long-context prompt. It is now default-on.

### What is retracted, precisely

1. **"Decode MoE is packet-protocol-bound; its kernel body is irrelevant at M=1."** False.
2. **"The claim path is not one of several options for decode — it is the only one."** False as
   stated. The claim path is still a real cost (the serial packet boundary is 2.13 µs = 5.5 ms/token
   = 19%, and it is fully exposed), but kernel work pays too, and it pays now.
3. The corollary about `GLM_GROUP=1` remains **correct but for a corrected reason** — it lost 15-16%
   because collapsing the packets serialises 8 concurrent slots that were each doing *real kernel
   work*, not because a fixed protocol toll failed to disappear.

### The methodological lesson, which is the expensive part

An ablation that shows "removing the work changes nothing" has **two** explanations — the work is
free, or *the work was never removed*. This report took the first without excluding the second.
The check that would have caught it costs one command: confirm which object the run actually
opens (`Variant::detect` on the real blob) **before** attributing meaning to a null. The campaign's
existing discipline — serve-gate every claim — cannot catch this class of error, because a change
that is never loaded serves perfectly.

**Any future ablation must first prove the ablated code is in the loaded object.**
