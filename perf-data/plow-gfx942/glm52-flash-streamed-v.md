# Streamed-V for `d_flash_mla_prefill_v2`: the idea is REFUTED, the V-stage arm that
# survives it (`PLOW_MLA_PF_SV`) is landed opt-in

> **Scope:** compile/ISA only -- llvm-objdump --mcpu=gfx942; benching on 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **CDNA3-SPECIFIC** — LDS bank algebra at 32 banks, the 64 KiB arena and the gfx942 register file drive every conclusion. One exception is broader: D16 writing the full VGPR is GFX940+, so it covers gfx950 too.

2026-08-08, branch `flash-streamv` off `worktree-glm52-bringup` @ 9468fdb. Compile- and
ISA-verified only — **no GPU was touched**; every number below is read out of the ELF or
out of `llvm-objdump --mcpu=gfx942`. Benching is the coordinator's.

Question: can the MoonMath/aiter "stream the second operand global→VGPR, skip the LDS
round-trip" discipline (perf-data/plow-gfx942/glm52-asm-innerloop-diff.md §1) be applied
to the P·V stage of the V2 MLA prefill flash kernel?

**Answer: no, and the reason is structural, not a tuning failure.** The salvage — the LDS
bank-conflict that the transpose has been paying since the kernel was written — is landed
behind `PLOW_MLA_PF_SV`, default OFF, bit-identical, at zero register cost, and it turns
96 full LDS drains per inner iteration into 12.

---

## 1. The staging map (what actually goes through LDS today)

Measured against `interp_flash.elf` built `PLOW_OCC4=1 PLOW_L2HIER=1` at 9468fdb.
`d_flash_mla_prefill_v2<512,64,false>` is a standalone symbol at 0x1b00; the KV-tile inner
loop is the backward branch 0x3b78..0x65a0, **1564 instructions**.

LDS arena, 41,472 B of the flash object's 58,368 B `fa`:

| region | shape | bytes | who writes | who reads |
|---|---|---|---|---|
| `Ksm` | `[BKV=32][KSTR=584]` bf16, kv-major, `latent(512) \| rope(64) \| pad(8)` | 37,376 | all 4 waves, cooperatively | all 4 waves — **twice** |
| `Pw0` | `[4][RW=16][BKV=32]` bf16, one private strip per wave | 4,096 | own wave | own wave |
| `Msm` | `[32]` u64 membership words (GATHER arm only) | 256 | all | all |

**MLA's K IS its V**: the PV pass re-reads the SAME `Ksm` rows it just used for QK. There
is no second V stream and no V re-stage — that was already the "V from L1" trick, taken
one step further (V from LDS, already paid for). Q never touches LDS: it is 18 bf16x8
A-fragments (72 registers) hoisted once per work item.

Per inner iteration, from the disassembly:

| | count | notes |
|---|---|---|
| `s_barrier` | **2** | one before the commit (previous PV done), one after (slab visible) |
| `global_load_dwordx4` | 9 | the `PLOW_MLA_PF2_DBUF` register prefetch of tile *n+1*; lands in **AGPRs** `a[200:231]` + `v[252:255]` |
| `ds_write_b128` | 9 | the commit of tile *n*'s prefetch (8 latent + 1 rope) |
| `ds_read_b128` | **37** | 36 = QK K-fragments, 1 = the P A-fragment |
| `ds_write_b16` | 8 | P through the wave's private strip |
| `ds_read_u16` | **256** | **the PV V-fragment transpose** — 32 output tiles × 8 |
| `v_perm_b32` | 128 | packing those 256 u16 into 128 bf16x2 MFMA operand halves |
| `ds_bpermute_b32` | 32 | the quarter-wave softmax max/sum reductions |
| `v_mfma_f32_16x16x16_bf16` | 136 | 72 QK (2 n-tiles × 18 k-tiles × 2) + 64 PV (32 tiles × 2) |
| `s_waitcnt` | 131 | **96 of them full `lgkmcnt(0)`**, 34 partial, 1 `vmcnt(0)` |
| outstanding VMEM, steady state | **9** | issued right after barrier #2, waited at the next iteration's `vmcnt(0)` |

Where the barriers sit: `__syncthreads(); [GATHER mask stage] commit 9×ds_write_b128;
__syncthreads(); issue 9 global loads for n+1; QK; softmax; PV`. The global-load latency
window is the entire QK+softmax+PV body — DBUF already spends it correctly, and depth 1 is
all the register file affords (see §3).

The two hot LDS shapes:

```
QK   ds_read_b128 v[240:243], v202 offset:0/64/.../1088, 18688.../19776
     -> s_waitcnt lgkmcnt(0) -> 2× v_mfma      × 36, FULLY SERIALIZED (one read in flight)
PV   ds_read_u16 v2,   v173 offset:128
     ds_read_u16 v240, v173 offset:1296        (+1168 B = KSTR×2, i.e. the next kv ROW)
     ... ×8 -> lgkmcnt(4) -> perm,perm,mfma -> lgkmcnt(0) -> perm,perm,mfma   × 32
```

## 2. Why the PV operand cannot be streamed — the fragment-map argument

`plow_mfma_bf16_16x16` maps **both** source fragments the same way: `A[m][k]` and
`B[n][k]` with `m`/`n` = `lane%16` and `k = 8*(lane/16)+j`. So the **contraction axis must
be the lane-contiguous axis of both operands.**

* QK contracts over `d`. `d` is the fast axis of Q (registers) and of `Ksm` (kv-major).
  Everything is a 16-byte `ds_read_b128`. No transpose.
* PV contracts over `kv`. `kv` is the **minor** axis of the latent — in LDS *and* in HBM
  (`Ckv` is `[kv][DK=512]` row-major, 1024 B row stride). **The PV B-operand is V^T and a
  transpose is mandatory.**

gfx950/CDNA4 has `ds_read_b64_tr_b16` for exactly this. **gfx942 does not.** So the
transpose must be paid in one of four places:

| placement | cost per lane per KV tile | verdict |
|---|---|---|
| **T1 LDS read side** (shipped) | 256 `ds_read_u16` + 128 `v_perm_b32` | cheapest — but 4-way bank-conflicted (§4) |
| T2 LDS write side (store V dim-major) | see below | bank-catastrophic **and** LDS-infeasible |
| T3 global read side (**task option (a)**) | 256 `global_load_ushort` | 28× VMEM issue inflation |
| T4 cross-lane in registers (**task option (b)**) | ~256 `ds_bpermute_b32` + 144 VGPR | strictly worse than T1, and no registers |

### T3 — "stream the latent global→VGPR a second time" (option (a)): REFUTED

Lane `(fr,kg)` needs `V[kv0+kg*8+j][t*16+fr]` for `j∈[0,8)`, `t∈[0,32)` — 256 halves,
each from a *different* 1024-B-strided HBM row-offset pair. For a fixed `(j,t)` a lane
needs exactly **one** 2-byte element; the 16 `fr` lanes of a k-group cover 16 consecutive
halves (32 B) and the 4 k-groups sit on 4 different rows. The widest legal load is
therefore `global_load_ushort`:

* **256 VMEM instructions per lane per KV tile**, against 9 `global_load_dwordx4` today —
  **28× the VMEM issue rate**, on a unit that retires roughly one address-quarter-wave per
  cycle. 4 waves × 256 = 1024 VMEM instructions per CU per KV tile, versus 136 MFMA per
  wave. The kernel becomes address-throughput-bound before it is anything else.
* Each instruction moves 4 rows × 32 B useful out of 4 × 64 B lines fetched: **50% line
  efficiency**, and the whole 32×512 tile is fetched **once per wave** instead of once per
  workgroup — 4× amplification on top.
* Register cost of the "deep prefetch" that would make it worthwhile: one destination VGPR
  per in-flight ushort. Reaching aiter's 13–39 outstanding needs 13–39 free VGPRs. There
  are **zero** (§3).

The L2-hit argument ("it was just fetched") is real and irrelevant: the limit is not where
the bytes come from, it is how many instructions it takes to ask for them.

### T4 — "keep K fragments live in registers across softmax" (option (b)): REFUTED TWICE

**(i) Register arithmetic.** The QK K-fragments are `nt∈{0,1} × kt∈[0,18)` = 36 `bf16x8` =
**144 registers per lane**. The measured budget (§3) has 0 free and 215 already spilled.

**(ii) Even with infinite registers, they are the wrong permutation.** Lane `(fr,kg)`
holds, after QK, `K[kv = nt*16+fr][d = kt*32+kg*8+j]` — i.e. `kv ≡ fr (mod 16)`, `d`
spread over all 576. For PV the same lane needs `V[kv ∈ [8kg, 8kg+8)][d ≡ fr (mod 16)]`.
Intersection: at most 8 of the 256 halves it needs — **97% of the PV operand has to move
between lanes.** Wave-wide the two sets are exact covers (64 lanes × 36 × 8 = 18,432 =
32×576; 64 × 32 × 8 = 16,384 = 32×512), so this is a full 32×512 transpose across the
wave. `ds_bpermute_b32` moves one dword per lane, and the two halves of a source dword
(`d`, `d+1`) have *different* destination lanes (`fr = d%16` vs `(d+1)%16`), so they
cannot be co-routed: **≈256 `ds_bpermute_b32` per lane per KV tile**, on the same LDS
crossbar as the 256 `ds_read_u16` they would replace, plus 144 registers. Strictly worse.

### T2 — store V dim-major so the PV read is wide: REFUTED

Attractive (32 `ds_read_b128` instead of 256 `ds_read_u16`) and it fails on two counts:

1. **QK loses its operand.** A dim-major tile cannot serve the QK B-fragment (which needs
   `d` contiguous). Keeping both layouts is 36,864 + 37,376 + 4,096 = **78,336 B against a
   58,368 B arena**. The alternative is to stream K for QK from global — 36
   `global_load_dwordx4` per lane per KV tile, i.e. the *whole* 36,864 B tile re-read per
   wave, 4× amplified, with 144 registers wanted for the prefetch depth that would justify
   it. Same wall as T3.
2. **The transposed store is bank-catastrophic.** A 16-byte PV read forces the dim-major
   row stride `P ≡ 0 (mod 8)` halves. The store has all 64 lanes of a wave writing the
   *same* kv offset in *different* `d` rows, so the lane stride is `8·P` halves = `4P`
   dwords ≡ 0 (mod 32) for every such `P` → **all 64 lanes land in one bank**. Relaxing
   the read to `ds_read_b64` allows `P ≡ 0 (mod 4)` and `P = 36` does come out
   conflict-free on both sides — but blocker (1) stands regardless.

**Conclusion: T1 is the cheapest of the four placements, and the shipped kernel already
picked it.** What was still on the table inside T1 is the bank conflict.

## 3. Register budget — the measurement, not an estimate

`llvm-readelf --notes`, `interp_flash.elf`, base and arm:

| | vgpr_count (unified) | agpr_count | group_segment | vgpr_spill_count |
|---|---|---|---|---|
| base 9468fdb | 512 | 256 | 58,368 | **215** |
| `PLOW_MLA_PF_SV=1` | 512 | 256 | 58,368 | **215** |

`vgpr_count` is the *unified* arch+acc total on gfx90a+ (`agpr_count` is the accumulator
subset, not an addition — the build script documents this). So: **256 arch VGPR + 256
AGPR, occupancy 1, and 215 registers already spilled to scratch. Free registers: zero.**

Named live state per lane inside the KV loop, from the disassembly:

| | registers | observed home |
|---|---|---|
| `oacc[32]` f32x4 (16 q rows × 512 cols) | 128 | AGPR |
| `qa[18]` bf16x8 (Q A-fragments) | 72 | AGPR `a[128:199]` |
| `rl[9]` bf16v8 (DBUF prefetch) | 36 | AGPR `a[200:231]` + `v[252:255]` |
| `sacc[2]` f32x4 | 8 | AGPR `a[0:7]` |
| `m_st`,`l_st`,`pe[2][4]` | 24 | VGPR |
| address bases / masks / f2bf scratch | ~40 live | VGPR |
| **total named** | **~308 of 512** | rest is scheduler pressure → 215 spills |

Against that: option (b) asks for **+144**; a T3/T2 prefetch deep enough to matter asks for
**+36 to +144**. There is no headroom, and the 215-register spill says the allocator is
already past the cliff. **Said plainly: the streamed-V variant needs more registers than
exist on this kernel at occupancy 1.**

## 4. What is landed: `PLOW_MLA_PF_SV` (opt-in, default OFF, bit-identical)

Three pieces, all inside T1. Nothing streams from global.

### (1) The kv-block LDS swizzle — the transpose read was 4-way bank-conflicted

For lane `(fr,kg)` reading `Ksm[(kg*8+j)*KSTR + t*16 + fr]` the LDS bank is

```
bank = ( (KSTR/2)·(8·kg + j) + 8·t + floor(fr/2) )  mod 32
```

and `8·kg·(KSTR/2) ≡ 0 (mod 32)` **for every `KSTR` that is a multiple of 8** — which is
forced, because QK reads 16-byte `ds_read_b128` fragments out of the same rows. So `kg`
drops out of the bank index entirely. `fr` and `fr+1` share a dword and broadcast (reads
to one address are free), but the four k-groups land four *distinct* addresses in one
bank: **every one of the 256 reads is a 4-way bank conflict**, i.e. ~1024 LDS cycles per
wave per KV tile where 256 would do, ×4 waves on one LDS unit against 136 MFMA
(≈2176 matrix cycles) per SIMD. This is not a tuning oversight in the pad value — it is
true for *all* legal pads: `KSTR = 8m ⇒ (KSTR/2)·8·kg = 32·m·kg ≡ 0`.

The fix shifts each 8-row kv **block** by +16 halves:

```
krow(kv) = kv·KSTR + 16·(kv >> 3)
```

16 halves = 8 dwords, so the bank index gains an `8·kg` term: `kg` spreads over
`{0,8,16,24}`, `floor(fr/2)` over `{0..7}`, and the 64 lanes cover all 32 banks with one
address each — **conflict-free**. 16 is also a multiple of 8 halves, so QK's `b128`
alignment survives. Cost: `3×16` halves = **96 B of LDS** (41,472 → 41,568 B, arena is
58,368).

**It is free in the ISA**, which the disassembly confirms:

* PV base `v186 = (kg<<5) | (fr<<1) | (kg·0x1240)<<1` — the `kg<<5` *is* the swizzle,
  folded into the lane's base register; all 256 offsets stay immediates (`160, 1328,
  2496, …`, stride 1168).
* QK base `v184 = (kg<<4) + fr·0x490 + ((fr<<2) & 32)` — the last term is `16·(fr>>3)`
  halves, loop-invariant per lane. `nt=1` gets `v185` with `((fr|16)<<2) & 0x60` = 64 or
  96 B, i.e. `16·((16+fr)>>3)` halves. Correct on both halves.
* Store side: `r>>3` is the compile-time constant `it/2` (DBUF path, since
  `r = wave + 4·it` with `wave < 4`) and the per-thread constant `wave` (rope path) — it
  folds into the immediate/base.
* The P-strip base moved `37376 → 37472` (exactly +96 B), which is the independent check
  that `KSLAB` grew by the 48 halves of slack and nothing else moved.

Inner-loop VALU: 828 → 831 (+3, from (2)/(3)'s index math). The swizzle itself adds two
VALU to the per-work-item *prologue*, outside the KV loop.

### (2) QK K-fragment double buffer

The shipped form reloads `kf` into the register the next MFMA pair consumes, so the
compiler emits `ds_read_b128; s_waitcnt lgkmcnt(0); mfma; mfma` **36 times** — one LDS
read in flight, its full latency exposed every fragment. Flattening `(nt,kt)` to a single
`st` and issuing `st+1`'s read before `st`'s MFMAs gives the compiler a dependence graph it
can pipeline. It responds by allocating **two alternating AGPR groups** (`a[4:7]`/
`a[8:11]`) for the staging — which also takes the QK operand off the arch-VGPR file.

### (3) PV V-fragment double buffer

The next output tile's 8 `ds_read_u16` issue before the current tile's MFMAs, taking the
lgkm pipeline from 8 outstanding to 16.

Depth 3 was built and measured: identical waitcnt profile (`lgkmcnt(0)` still 12, max
outstanding 14 vs 13) and `interp_flash_fp8kv` spill 12 → 14. **Depth 2 is the stopping
point** — beyond it the LLVM scheduler, not the source structure, is binding.

## 5. Before / after: the annotated ISA delta

Inner loop = the KV-tile backward branch. Base `0x3b78..0x65a0`; arm `0x3bb4..0x65e8`.

| metric, per inner iteration | base | `PLOW_MLA_PF_SV=1` | Δ |
|---|---|---|---|
| instructions | 1564 | 1562 | −2 |
| `v_mfma_f32_16x16x16_bf16` | 136 | 136 | — |
| `s_barrier` | **2** | **2** | — |
| `ds_read_u16` (V stage) | 256 | 256 | — (count structural, §2) |
| — their LDS bank conflict | **4-way** | **conflict-free** | **−4×** |
| `ds_read_b128` (QK + P) | 37 | 37 | — |
| `ds_write_b128` / `ds_write_b16` | 9 / 8 | 9 / 8 | — |
| `v_perm_b32` | 128 | 128 | — |
| `s_waitcnt` total | 131 | 129 | −2 |
| — **full `lgkmcnt(0)` drains** | **96** | **12** | **−87.5%** |
| — partial `lgkmcnt(1..13)` | 34 | **116** | +82 |
| — deepest LDS pipeline waited on | `lgkmcnt(7)` | **`lgkmcnt(13)`** | ~2× |
| `s_waitcnt vmcnt(0)` | 1 | 1 | — |
| steady-state outstanding VMEM | **9** | **9** | **unchanged** |
| VALU | 828 | 831 | +3 |
| vgpr / agpr / spill | 512 / 256 / **215** | 512 / 256 / **215** | **0** |

Waitcnt histograms, verbatim:

```
base : lgkmcnt(0)×96  lgkmcnt(4)×32  lgkmcnt(7)×1  lgkmcnt(1)×1  vmcnt(0)×1
arm  : lgkmcnt(1)×51  lgkmcnt(4)×30  lgkmcnt(8)×28 lgkmcnt(0)×12 lgkmcnt(2)×3
       lgkmcnt(7)×2   lgkmcnt(11)×1  lgkmcnt(13)×1 vmcnt(0)×1
```

The QK subloop, before and after:

```
BASE                                        ARM
ds_read_b128 v[240:243], v202               ds_read_b128 a[4:7],  v184
s_waitcnt lgkmcnt(0)          <- full        ds_read_b128 a[8:11], v184 offset:64
v_mfma a[0:3], a[128:129], v[240:241]        s_waitcnt lgkmcnt(1)   <- partial
v_mfma a[0:3], a[130:131], v[242:243]        v_mfma ... a[4:5] ...
ds_read_b128 v[240:243], v202 offset:64      ds_read_b128 a[4:7], v184 offset:128
s_waitcnt lgkmcnt(0)          <- full        s_waitcnt lgkmcnt(1)   <- partial
                                             ...
```

The PV subloop, after (8 reads for tile *t+1* issued between tile *t*'s two MFMAs):

```
ds_read_u16 v242..v45, v186 offset:160,1328,2496,3664,4832,6000,7168,8336
v_mfma a[16:19], v[120:121], v[240:241], a[16:19]
s_waitcnt lgkmcnt(8)          <- partial, 16 outstanding
v_perm / v_perm
v_mfma a[16:19], v[122:123], v[228:229], a[16:19]
```

**Honest reading of this table:** the arm does *not* deliver what the brief asked for on
two of its three axes. `ds_read` count is unchanged (§2 says it cannot change on gfx942),
barriers are unchanged at 2 (they already were — V2's whole point), and outstanding VMEM
is unchanged at 9 (the DBUF depth is register-bound). What it does deliver is the LDS
half: 96 → 12 full pipeline drains, ~2× deeper LDS pipeline, and a 4× reduction in the
bank cycles of the single largest LDS consumer in the loop — for 96 bytes of LDS and zero
registers.

## 6. Verification (no GPU)

```
export PATH=/nix/var/nix/profiles/default/bin:/root/.nix-profile/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib
export ROCM_PATH=/opt/rocm-7.2.4 HIP_PATH=/opt/rocm-7.2.4 ROCM_HOME=/opt/rocm-7.2.4
export PLOW_HIPCC=/opt/rocm-7.2.4/bin/hipcc

# arm on — 28 objects, every row inside the cliff
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 bash scripts/build_gfx942.sh <out>/hsaco_sv
# flash-only iteration (~2 min): add PLOW_ROWS_ONLY=interp_flash
```

* **Arm ON, full build: PASS**, 28/28 objects, cliff check clean. `interp_flash` 512/256/
  58,368/spill 215 — identical to base. Only `interp_flash_fp8kv{,_gq}` move (spill
  10→12, 6→11), and only because they carry the same V2 body.
* **Arm OFF is byte-identical to the base commit.** Built from the *same* worktree path at
  9468fdb and with the patch applied, flag unset: `interp_flash`, `interp_flash_gq`,
  `interp_flash_fp8kv`, `interp_flash_fp8kv_gq` all `cmp`-identical.
* Full 28-object off-vs-base compare across *different* worktree directories shows a
  90-byte delta in every object, confined to the `__hip_cuid_<hash>` symbol (hipcc hashes
  the compilation unit's absolute path). **Disassembly text is identical for all 28
  objects** (`llvm-objdump -d` diff = the filename banner line only), which is the
  codegen-level proof; the same-path `cmp` above is the byte-level one.

## 7. Emit + bench recipe

`PLOW_MLA_PF_SV` is **object-level and bit-identical** — no blob re-emit, no manifest
`requires`, no host routing, no marker symbol. It is wired exactly like
`PLOW_MLA_PF2_DBUF`. So the A/B is objects-only, against the shipped best blob:

```
# 1. two object sets, same tree
env PLOW_OCC4=1 PLOW_L2HIER=1                    bash scripts/build_gfx942.sh <tmp>/hsaco_ctl
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1   bash scripts/build_gfx942.sh <tmp>/hsaco_sv

# 2. reuse the CURRENT BEST blob unchanged (asset glm52-tp8-best, emitted with
#    PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=2). Point the asset's hsaco symlink at each set.

# 3. serve env is unchanged for both arms:
#    PLOW_MLA_PF_V2=1  (+ the blob's PLOW_MLA_PF_NS requires)
#    interleaved 2 rounds @ 1k/4k/8k/16k, port 8196/8195, GPU-lock protocol,
#    'Paris' coherence gate per arm.
```

Expected signal: the arm touches only the MLA-prefill flash packet, so read the **traced
flash span per MoE layer @8k** first (control ≈ 4.29 ms/layer at v2+ns2) — TTFT will move
by roughly the flash share of it. If the flash span does not move, the loop was
MFMA-issue-bound rather than LDS-bound and the arm should stay off.

Correctness: bit-identical by construction (LDS addresses and load issue order only; the
MFMA sequence, operands and accumulation order are untouched). The `st`-flattened QK loop
walks `nt = st/NKT`, `kt = st%NKT` in exactly the original order. Still gate it — a logit
compare against the control arm should be **byte-identical**, not merely close; anything
else means the swizzle is wrong somewhere and the arm must not ship.

## 8. What this closes, and what is left

**CLOSED — do not re-try on gfx942:** streaming the PV operand global→VGPR (T3), and
holding K fragments across softmax (T4). Both are refuted above by fragment-map algebra
and by a register file with zero free registers and 215 spills. The memory note "PV
transpose blocked by the 58 KiB arena" should be read as: *blocked by the MFMA fragment
contract and the LDS bank algebra*; the arena is only the third wall (T2).

Left on this loop, in order of expected value:

1. ~~**`ds_read_u16_d16_hi`.**~~ **REFUTED IN HARDWARE, 2026-08-09 — DO NOT RE-TRY.**
   The claim below was wrong about *why* LLVM does not form the instruction, and the
   remedy it proposed (gfx942-guarded inline asm) cannot work.

   > The 256 transpose reads are followed by 128 `v_perm_b32` to pack half-pairs. Pairing
   > each read with a `_d16_hi` sibling writes both halves of one VGPR directly and deletes
   > all 128 perms (8% of the loop) at no register cost. LLVM does not form it from the
   > current source; it needs gfx942-guarded inline asm, which is why it is not in this arm.

   **On GFX940+ a D16 instruction writes the FULL 32-bit VGPR.** It does not preserve the
   other half, so `ds_read_u16_d16_hi` cannot merge into a low half that a sibling
   `ds_read_u16` loaded — it zeroes it. Measured directly by
   `probes/d16sem.hip`, which seeds the destination and issues ONE d16_hi:

       pool[4] = 0xab04, destination register seeded 0x00005a5a
         ds_read_u16_d16_hi -> 0xab040000     (preserve would be 0xab045a5a)
         ds_read_u16_d16    -> 0x0000ab04

   That is the `d16-write-vgpr32` subtarget property ("D16 instructions potentially have
   32-bit data dependencies", visible in `llc -mattr=help`), and it is exactly why the
   backend never selects the pattern — the absence was CORRECT, not a missed optimisation.

   `probes/d16probe.hip` runs eight arms of the real PV staging shape (BKV 32, KSTR 584,
   NT 32, `Ksm` at a nonzero LDS offset so the generic->LDS address cast is exercised) and
   gates each bit-for-bit against the shipped form:

   | arm | idiom | gate | loop body | perm/or | waits |
   |---|---|---|---:|---:|---:|
   | 0 | shipped (per-element insert, SV(3) dbuf) | ref | 546 | 128 | 62 (29x lgkm(8), 29x lgkm(12)) |
   | 1 | `bf16x2` pair build | BIT-IDENTICAL | 546 | 128 | 62 |
   | 2 | `u16x2` pair build | BIT-IDENTICAL | 546 | 128 | 62 |
   | 3 | asm d16_hi, drained per tile | **MISMATCH 8192/8192** | 386 | 0 | 33 |
   | 4 | asm d16_hi, keeps the dbuf (`lgkmcnt(8)`) | **MISMATCH 8192/8192** | 547 | 0 | 33 |
   | 5 | asm d16_hi, full drain BETWEEN lo and hi | **MISMATCH 8192/8192** | 418 | 0 | 65 |
   | 6 | asm: 8 reads, ONE wait, keep the perms | BIT-IDENTICAL | 546 | 128 | 33 |
   | 7 | arm 6 + cross-tile dbuf | BIT-IDENTICAL | 549 | 128 | 33 |

   Arm 5 is the one that kills the idea outright: even a **full `lgkmcnt(0)` drain between
   the low and high loads** does not make the merge work, so this is not a
   two-loads-in-flight hazard that better scheduling could fix — the instruction simply has
   no merge semantics here. Arms 1 and 2 confirm the source-idiom half: two plausible
   rewrites that should have handed the backend a v2i16 insert produce **byte-identical**
   code to the shipped form.

   The two arms that ARE correct buy nothing. Arm 6 trades 29 partial waits for 33 full
   drains at the same body size; arm 7 keeps the pipelining but blows the register file
   (**vgpr 260**), which is fatal in a kernel already at occupancy 1 with 215 spills.

   Two smaller corrections this leaves behind: the perms are **128 of a 546-instruction PV
   body (23%), not 8%**, and the shipped double buffer is scheduling *well* — 29 x
   `lgkmcnt(8)` plus 29 x `lgkmcnt(12)`, not the full drains an earlier reading of a
   less faithful probe suggested. The transpose pack is not a soft target.
2. **A deeper DBUF.** Depth 2 would want +36 registers. Not available today; it becomes
   available if the fp8-latent arm (which halves `rl[]`) is ever revisited for footprint.
3. ~~**The softmax VALU.**~~ **ATTACKED AND REFUTED ON HARDWARE, 2026-08-09.** 828 VALU
   against 136 MFMA is 6:1; aiter's fmha runs 6.7 VALU/MFMA *including* softmax, so plow was
   already the leaner of the two and there was less here than the entry implied. The one
   concrete sub-item — "the f2bf conversion of P alone costs ~50 instructions of
   exec-mask-branched rounding per tile" — was real, and removing it made things WORSE.

   Making `f2bf` branchless is unambiguously better on every static measure: over a
   64-conversion tile 1881 -> 838 instructions, 1098 -> 643 VALU, 151 -> 19 exec-mask ops, and
   on the real prefill megakernel **-5.0% instructions, -13.3% SALU, -30% exec-mask ops**.
   Served, over 4 interleaved arms x 2 rounds:

   | ctx | branchless | branched | Δ |
   |---:|---:|---:|---:|
   | 1024 | 325.6 | 323.1 | +0.77% |
   | 4096 | 751.4 | 718.9 | **+4.51%** |
   | 8192 | 1708.6 | 1622.2 | **+5.33%** |
   | 16384 | 3783.6 | 3535.8 | **+7.01%** |
   | TPOT | 26.485 | 26.271 | +0.81% |

   Ordering is excluded: the branchless arm reproduces to within 0.1% whether it runs first or
   third. The select form computes BOTH the RNE and the quiet-NaN on every conversion where the
   branch computed one, and `f2bf` is on every store path in the runtime — so the exec-mask
   overhead that was deleted was cheaper than the arithmetic that replaced it.

   It also was **not output-identical in situ** despite being value-identical over all 2^32
   float bit patterns: GSM8K 0.960 vs 0.970, reproducible per arm across both rounds. Inlining
   a function this widely used perturbs surrounding codegen and, at -O3, which fp contractions
   form. Kept as `PLOW_F2BF_SELECT=1`, default off. **This closes item 3 as a NO-GO** — the
   softmax VALU is not a soft target either.

Artifacts: `<tmp>/{base,sv}_flash.asm`, `<tmp>/hsaco_{base,off,sv,sv_flash,sv3}`.
Loop extractor: `perf-data/plow-gfx942/probes/asm_loops.py`.

## ADOPTION GATE (coordinator, 2026-08-08): SV ADOPTED — win at 6× the noise floor

Objects-only A/B (same blob both arms), 3 interleaved rounds, port 8196, control =
`hsaco_fulloff` (flag-off build, byte-identical to base), arm = `hsaco_sv`. TTFT ms:

| ctx | ghctl r1/r2/r3 | mean | sv r1/r2/r3 | mean | Δ |
|---|---|---|---|---|---|
| 4096 | 1009.6 / 1018.4 / 1010.9 | 1013.0 | 1006.3 / 998.9 / 998.0 | 1001.1 | **−1.2%** |
| 8192 | 1752.2 / 1759.2 / 1756.4 | 1755.9 | 1717.6 / 1711.8 / 1707.3 | 1712.2 | **−2.5%** |
| 16384 | 3801.4 / 3815.7 / 3809.2 | 3808.8 | 3732.2 / 3704.2 / 3701.0 | 3712.5 | **−2.5%** |

Control round-to-round spread is 7.0 ms @8k (0.40%) and 14.3 ms @16k (0.38%); the arm's
deltas are 43.7 and 96.3 ms — ~6× the noise floor, and every individual sv round beats
every individual control round at 8k and 16k (no overlap between the two distributions).

**Bit-identity gate PASSED at the strict setting.** Because the arm claims bit-identity by
construction, the gate was character-identity on BOTH arms rather than the usual
sanity check: 4/4 answers (Paris / 391 / Au+jewelry / Rayleigh-scattering) are identical
strings between control and arm, including the two long free-form generations where any
numeric drift would diverge within a few tokens. This is the evidence that the bank
swizzle algebra is right — a wrong swizzle reads the wrong halves and cannot produce
identical text.

**The bank-conflict model is confirmed by the shape of the win.** The gain scales with
KV-tile count (−1.2% @4k → −2.5% @8k/16k, flattening once the flash dominates), which is
what an LDS-side fix predicts and what an MFMA-issue-bound loop would NOT produce. Risk (2)
in the original report — "if the loop is MFMA-issue-bound the arm is a wash" — is
discharged by measurement.

**Status: part of the canonical object recipe** (`PLOW_MLA_PF_SV=1`), flash object only.
Objects-only: no blob re-emit, no manifest `requires`, serve env unchanged — an object
built without the flag remains fully correct, just slower, so there is no arm-check hazard.

Next lever named in this report and NOT yet taken: ~~`ds_read_u16_d16_hi`~~ — **CLOSED
2026-08-09, refuted in hardware** (see the struck item 1 above: D16 writes the full VGPR
on GFX940+, so there is no merge to exploit). The remaining levers on this loop are the
deeper DBUF (blocked on fp8-latent KV for the register room) and the 6:1 softmax VALU.
