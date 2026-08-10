# GLM-5.2 MoE grouped prefill: fusing ops 86 -> 87, and deleting the f32 `part` round trip

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **AMD-GENERAL mechanism / CDNA3 constants** — atomic accumulation is aiter's own mechanism on both arches; the 357 G/s atomic ceiling and the -64.6% are MI300X.

2026-08-08, branch `moe-fuse` (base `968fc3a`, worktree `.claude/worktrees/moe-fuse`).
Target: the largest addressable component in
`glm52-current-cost-decomposition.md` — the MoE grouped path, 515 ms of TTFT at 8k.

Everything below is either (a) disassembled from a built object, (b) measured on this box
this session, or (c) arithmetic over (a)+(b) with the derivation shown.

## VERDICT, first

**BUILT AND IT WORKS, BUT IT IS SMALLER THAN THE COST DECOMPOSITION PREDICTED, AND IT DOES NOT
PASS A CHARACTER-IDENTICAL GATE.**

* `PLOW_MOE_PF_ATOMIC` (opt-in, default off, default object BYTE-IDENTICAL) fuses ops 86→87 by
  atomic accumulation — aiter's own mechanism, confirmed by disassembling its shipped gfx942
  `g1u1` object.
* **DRAM removed: 2.416 GB per MoE layer per rank at T=8192** (181 GB per 8k chunk over 75
  layers). plow's MoE pair goes from **2.29× to 1.36×** aiter's bytes.
* **Served TTFT: −33.9 / −66.8 / −128.8 ms = −3.5% / −4.0% / −3.5%** at 4k/8k/16k, 3 interleaved
  rounds, against a control spread of 1.34% / 0.38% / 0.61%. All 9 fused points below all 9
  control points. TPOT unchanged.
* **Traced, independent of the wall:** device prefill wall 1402.9 -> 1329.9 ms (−5.20%); the MoE
  combine packet **1,516.6 -> 536.4 µs/layer (−64.6%)**, the DOWN packet +1.6% for its atomics,
  the router +5.3% for the zero-fill, and **nothing else in the layer moves by more than 0.9%**.
  Perfect-pack prediction from the trace: −68.4 ms. Served: −66.8 ms. They agree to 2.4%.
* **Not 286 ms.** That figure was an aiter-ratio aspiration attached to this exact mechanism; the
  mechanism is worth 66.8 ms, because the pair was never bandwidth-bound. What paid was op 87's
  SHAPE (k=8 strided streams → 1 contiguous, −86.9% on a probe), not the bytes. §10 corrects the
  record.
* **Numerics: not bit-identical AND not run-to-run deterministic** (atomic arrival order).
  Character-identical on 2 of 6 gate prompts, and unstable against itself on 3 of 6. The
  divergences are wording, not correctness (both fused runs answer 17×23 = 391 with correct
  working). **Do not default it on as it stands**; §11 item 1 is the deterministic variant that
  would keep most of the win.
* **The op-graph decomposition is now CLOSED as a lever.** plow's MoE bytes are within 1.36× of
  aiter's while its time is still ~2.2×, which puts the entire remaining gap on tile-structure
  round-trip serialization.

---

## 1. ANATOMY — the three-packet dataflow, exactly

Per MoE layer per rank, TP8: `T` tokens, `H = 6144`, `I_moe = 256`, `E = 256`, `k = 8`.
`R = MPF_MAX_ROWS(T,k,E) = T*k + E*(MPF_BM-1)` padded gathered rows (73,728 at T=8192).

| op | reads | writes | bytes at T=8192 |
|---|---|---|---:|
| 83 `MoeRouterTopkPf` | `rlogit[T,E]` bf16 | `tab[T*k]` | 4 MB |
| 84 `MoeAlignPf` (1 WG) | `tab` | `meta`, `row_token[R]`, `row_partidx[R]`, `row_gate[R]` | ~1 MB |
| 85 `MoeGroupGluPf` | gate+up fp8 `E*2*I*H`, `xn2` GATHERED by `row_token` | `fu_g[R, I]` bf16 | 805.3 + 100.7 -> **37.7 W** |
| 86 `MoeGroupDownPf` | down fp8 `E*H*I`, `fu_g` | **`part[row_partidx[row]][H]` f32** | 402.7 + 37.7 -> **1611.0 W** |
| 87 `MoeCombinePf` | **`part[T*k, H]` f32**, `zero_h[T,H]`, `shared[T,H]` | `dg_tp[T,H]` bf16 | **1611.0 R** + 100.7 + 100.7 -> 100.7 W |

`part` is the only term in the layer that scales as `T*k*H`. At T=8192 it is
**1.611 GB written by op 86 and 1.611 GB read by op 87, per layer, per rank** — and there
are 75 MoE layers.

### 1.1 Why it is f32

Not for cross-expert reduction precision *inside* op 86 — op 86 writes each slot exactly
once. It is f32 because op 87 sums the `k` slots in f32 and the campaign's shipped numerics
contract is "same expression, same order as the decode `d_moe_combine`". `PLOW_MOE_PF_PART16`
already tried bf16 and it **flipped top-1** (`glm52-moepf-activation-arms.md`).

### 1.2 Why it must currently round-trip DRAM — and it is NOT the grouping or a barrier

`d_moe_align_pf` writes `row_partidx[pos] = s` where `s = token*k + slot`, i.e. `part` is
indexed **by routing slot**, and the combine reads `part[tok*k + j]` for `j < k`.

The `k` slots of one token are chosen by `k` DIFFERENT experts. In the expert-sorted row order
that op 85/86 walk, those `k` rows are in `k` different m-tiles, and the m-tiles are handed to
workgroups by `for (lin = slice; lin < n_tiles; lin += nblk)` — so in general they are on
**`k` different CUs, and (with `GLM_MOE_CORESIDENT`/XCD placement) different XCDs**.

**The k-way reduction is therefore inherently CROSS-WORKGROUP.** It is not a barrier that can
be moved, not a row-scatter that can be re-indexed, and not an artifact of the grouping: any
expert-grouped GEMM has this property. There are exactly two implementations:

1. a second pass over a materialised intermediate — which is what op 87 **is**; or
2. a global atomic accumulate.

### 1.3 What the reference actually does — settled by disassembly, not by inference

`glm52-experiments.md (consolidated: MoE k-loop, six arms null)` says aiter's fused `g1u1` "never materialises an intermediate".
Confirmed, and the *mechanism* is now on the record:

```
$ llvm-objdump --disassemble --mcpu=gfx942 \
    aiter_meta/hsa/gfx942/fmoe_fp8_blockscale_g1u1_subGU_256.co | grep -o global_atomic.*
96 x  global_atomic_pk_add_bf16
```

96 packed-bf16 atomic adds and **no scatter store at all**. Its `out` is `[token_cnt, dim]`
(`csrc/py_itfs_cu/asm_fmoe.cu`), and the buffer is zeroed by `moe_sorting_fwd`, not by
`torch.zeros` (`fused_moe_bf16_asm.py:38` allocates it with `torch.empty`). aiter takes
route 2. So the fusion is an ATOMIC ACCUMULATE, and that is what this branch builds.

---

## 2. DESIGN — with the LDS budget as the hard constraint

### 2.1 The LDS arithmetic, first, because it kills two of the three candidate fusions

`plow_smem` on gfx942 is **64,512 B**; the grouped-prefill arena is already at **64,560 B**
(measured `.group_segment_fixed_size`), and `MPF_DBUF` is already forced to 1 because
double-buffering would need `2 * MPF_TILE * 2 = 81,920 B`. There is no headroom.

**(A) Full g1u1-style single pass (85+86 in one kernel) — REJECTED, misses by 1,024 B.**

* GEMM staging tile, single-buffered: `MPF_TILE * 2 = (64 + 256) * 64 * 2 = 40,960 B`.
* gate/up -> down bridge: `MPF_BM x I_moe` bf16 = `64 * 256 * 2 = 32,768 B`.
  (Dimensionally exact — this is the observation the audit's §5 makes.)
* Naive sum `40,960 + 32,768 = 73,728 B` — **9,216 B over**.
* With perfect aliasing (the GLU phase's staging is dead once the bridge is written), the
  binding constraint is the DOWN phase, where the bridge IS the A operand and must be live
  simultaneously with DOWN's own B tile:
  `bridge 32,768 + DOWN B tile (256 x 64 x 2) 32,768 = 65,536 B` vs a **64,512 B** arena —
  **1,024 B over**, before any of the 48 B of scalars the body already carries.
* The only way in is `BN = 128` for the DOWN phase (`32,768 + 16,384 = 49,152 B`, fits) — which
  doubles DOWN's n-tiles from 24 to 48 and therefore **doubles its epilogue count**, on a kernel
  whose k-loop is 4 trips deep and which is already epilogue-dominated (`k-loop = 32%` of the
  instruction stream). Rejected on cost, not only on space.
* And the prize is small regardless: 85->86 fusion removes only the `fu` round trip,
  `37.7 MB W + 37.7 MB R = 75.4 MB/layer` = **2.6% of the pair's 2.848 GB**. It does not touch
  either 1.611 GB `part` traversal.

**(B) 86->87 through LDS or registers — IMPOSSIBLE IN PRINCIPLE, not by budget.** §1.2.

**(C) 86->87 by global atomic — CHOSEN. LDS cost: ZERO.**

That is the whole reason this design is buildable and (A) is not. Verified, not asserted:

```
object                               vgpr   agpr       lds   spill
interp_prefill_fp8_mla_moe (control)  256      0     64560       2
interp_prefill_fp8_mla_moe (fused)    256      0     64560       2
```

Register arithmetic: the DOWN epilogue's live set is unchanged. It already holds
`accf[SM][SN][16]` (2 x 16 f32) plus the two `PLOW_MOE_PF_EPI` hoist registers
(`epi_pidx`, `epi_gate`). The fusion adds **one** value per element — `pidx >> ksh` — which the
compiler materialises into the same VGPR it was going to use for the `part` row index. Measured
VGPR count is identical (256, which is the cap, so a real increase would have shown as spill:
spill is 2 in both).

### 2.2 The decomposition that ships

| packet | before | after |
|---|---|---|
| 83 `MoeRouterTopkPf` | router top-k | + grid-strided **zero of `acc[T,H]` f32** (`t2`/`i0`) |
| 85 `MoeGroupGluPf` | unchanged | unchanged |
| 86 `MoeGroupDownPf` | `nt_store(gate*v, &part[pidx*H + nn])` | `atomicAdd(&acc[(pidx>>log2 k)*H + nn], gate*v)` (`i4 = log2(k)+1`) |
| 87 `MoeCombinePf` | `k = 8` strided streams | **same code**, emitted with `k = 1`: one contiguous stream |

Three properties that make this cheap:

1. `row_partidx[row] == token*k + slot` **by construction** in `d_moe_align_pf`, so the token
   index is `pidx >> log2(k)` — one `v_lshrrev_b32`. No extra load, no division, no new tensor.
   (The arm therefore requires a power-of-two `k`; the emit asserts it.)
2. The zero lands on op 83, which is **already** the first packet of the MoE chain
   (83 -> 84 -> 85 -> 86). The existing dependency edges order it. **No new packet, no new gate**
   — which matters because `emit_dep_work` is dead code in every shipped program and only coarse
   gates exist.
3. **op 87's kernel is not modified at all.** `k = 1` makes its existing loop read the
   accumulator. That is what keeps the numerics story to one sentence.

### 2.3 Why this is not `PLOW_MOE_PF_PART16` a second time

This is the objection that has to be answered before spending a battery, because part16 halved
*the same stream* and measured a wash (DownPf −1.4%, **combine 0.0%**, TTFT −0.3%).

part16 changed the **bytes** and left the **shape** alone: op 87 still issued `k` strided loads
per output element (just narrower), op 86 still issued one store per element. This changes the
shape:

* op 87 goes from **k = 8 streams at `H*4 = 24 KB` stride** to **one contiguous stream** — an
  instruction-count change (8 loads/element -> 1), not a width change. That is precisely the
  axis part16 could not move, and it is why the combine measured a dead wash under part16.
* op 86's store disappears entirely rather than getting narrower.

Stated as a falsifiable prediction before measurement: if the combine is byte-bound this buys
nothing (part16 already showed it is not); if it is *stream-count* bound this buys most of op
87's 1,543 µs/layer.

---

## 3. DRAM BYTES MOVED, per MoE layer per rank, T=8192 — the number this build is about

`T=8192, H=6144, I_moe=256, E=256, k=8`, so `R = 73,728` padded rows. Decimal GB, matching
`glm52-experiments.md (consolidated: MoE k-loop, six arms null)` §2.1 (whose 2.996 GB pair total this table reproduces).

| stream | shipped | fused | delta |
|---|---:|---:|---:|
| gate+up fp8 (op 85 B) | 0.8053 | 0.8053 | — |
| A gather, distinct (op 85) | 0.1007 | 0.1007 | — |
| `fu_g` write / read (85 -> 86) | 0.0377 / 0.0377 | 0.0377 / 0.0377 | — |
| down fp8 (op 86 B) | 0.4027 | 0.4027 | — |
| **`part` f32 scatter (op 86 W)** | **1.6106** | — | **−1.6106** |
| **accumulator RMW (op 86, atomics)** | — | **0.4027** | **+0.4027** |
| **`part` read-back (op 87 R)** | **1.6106** | — | **−1.6106** |
| accumulator read (op 87 R) | — | 0.2013 | +0.2013 |
| `zero_h` + `shared` read (op 87) | 0.2013 | 0.2013 | — |
| `dg_tp` write (op 87) | 0.1007 | 0.1007 | — |
| **accumulator zero (op 83 W)** | — | **0.2013** | **+0.2013** |
| **pair (85+86) — the audit's basis** | **2.9947** | **1.7868** | **−1.208** |
| **whole chain (83+85+86+87)** | **4.9074** | **2.4915** | **−2.416** |

**2.416 GB of DRAM traffic removed per MoE layer per rank; 181 GB per 8k prefill chunk over
75 layers.** The pair lands at **1.787 GB against aiter's 1.309 GB** — from 2.29× aiter's bytes
to 1.36× — and the *chain* (which is the honest comparison, since aiter's single kernel has no
combine) goes from 3.75× to 1.90×.

**The one number in that table that is an upper-bound estimate, flagged as such:** the
accumulator RMW row. `402.65 M` f32 atomics land on a `0.2013 GB` footprint, i.e. 1.573 M
distinct 128 B lines each touched 8 times. If a line survives its 8 touches at the coherence
point (MI300X's 256 MB memory-side Infinity Cache holds the whole 201 MB accumulator, but it is
competing with a 1.208 GB weight stream) the DRAM cost is one read + one write = 0.4027 GB, as
tabled. The pessimal case is 8× that = 3.22 GB, which would be WORSE than the scatter it
replaces. §6 measures which it is rather than arguing.

---

## 4. ISA — in the SHIPPED megakernel object, not a probe TU

`/tmp/mf/obj_atom/interp_prefill_fp8_mla_moe.elf`, symbol `_Z19d_moe_group_down_pf...`,
built by the canonical recipe + `PLOW_MOE_PF_ATOMIC=1`. The out-of-line device function in the
megakernel, so the `X2_UN` lesson (unroll shapes do not transfer from standalone TUs) does not
apply.

### 4.1 Resources — unchanged, which is the load-bearing claim of §2.1

| | control | fused |
|---|---:|---:|
| `.vgpr_count` | 256 | 256 |
| `.agpr_count` | 0 | 0 |
| `.group_segment_fixed_size` | 64,560 | 64,560 |
| `.vgpr_spill_count` | 2 | 2 |
| occupancy | 2 waves/SIMD | 2 waves/SIMD |

VGPR count is at the cap in both, so a real register increase would have shown as spill; spill
is 2 either way.

### 4.2 The fused epilogue, per output element

```
ds_bpermute_b32 v70, v192, v225        ; row_partidx  (existing PLOW_MOE_PF_EPI hoist)
ds_bpermute_b32 v65, v192, v226        ; row_gate     (existing)
s_waitcnt      lgkmcnt(1)
v_cmp_ne_u32_e64 s[4:5], -1, v70       ; pad-row test  (existing)
s_and_saveexec_b64 ...
v_lshrrev_b32_e32 v69, v22, v70        ; <-- pidx >> log2(k).  THE ONLY ADDED INSTRUCTION
v_mad_u64_u32  v[70:71], s[8:9], v69, v14, 0
v_lshl_add_u64 v[70:71], v[70:71], 2, v[66:67]
s_waitcnt      lgkmcnt(0)
v_mul_f32_e32  v65, v40, v65           ; gate * value  (existing)
global_atomic_add_f32 v[70:71], v65, off
```

Three things to read off it:

* **`global_atomic_add_f32` with no return and no `s_waitcnt vmcnt` anywhere in the chain.** The
  atomics are fire-and-forget; nothing drains on them. This is why the epilogue does not gain a
  serialization defect of the kind `PLOW_MOE_PF_EPI` was built to remove.
* **`sc0`/`sc1`/`nt` are all clear** (encoding `DD348000 007F4146`). That is LLVM's gfx942
  memory model for `__HIP_MEMORY_SCOPE_AGENT` — agent scope needs no cache bits because the
  atomic is performed at a device coherence point. Since MI300X's L2 is PER-XCD and the k slots
  of a token are in general on different XCDs, this is a correctness precondition, not a
  performance detail; §5 checks it on hardware instead of trusting the model.
* One extra VALU per element against the shipped non-temporal store. Nothing else moves.

Static instruction census of `d_moe_group_down_pf` (both branches are compiled in; exactly one
runs, selected by `atom_ksh`):

| | control | fused |
|---|---:|---:|
| total instructions | 3,757 | 4,744 |
| `global_store_dword` (the f32 `part` scatter) | 64 | 64 (runtime-dead) |
| `global_store_short` (the part16 scatter) | 64 | 64 (runtime-dead) |
| `global_atomic_add_f32` | 0 | **64** |
| `ds_bpermute_b32` | 256 | 384 |
| `ds_read_b128` / `ds_write_b128` | 24 / 20 | 24 / 20 |

64 atomics = 2 template instantiations (fp8, bf16) x `SM*SN*16 = 32` elements — one per output
element, the same count as the store it replaces. The k-loop is byte-identical between the two
objects.

`d_moe_router_topk_pf` gains 34 instructions and exactly one non-temporal `global_store_dword`
(the zero-fill loop).

### 4.3 Default object byte-identity — verified, not asserted

Built the base commit and this commit **at the same path** (hipcc embeds the source path, so any
other comparison is meaningless), canonical axes, `PLOW_MOE_PF_ATOMIC` unset:

```
BYTE-IDENTICAL  interp_prefill_fp8_mla_moe.elf
BYTE-IDENTICAL  interp_prefill_fp8_mla_moe_gq.elf
```

This needed one deliberate move: the new parameters on `d_moe_group_pf_t`,
`d_moe_group_down_pf` and `d_moe_router_topk_pf` are INSIDE the `#if`. With them outside, `.text`
was still byte-identical but the two mangled names grew and shifted 1,055 bytes of `.strtab` —
recorded because "the instruction stream is identical, only `.strtab` moved" is a weaker claim
than the one this campaign's discipline asks for, and the fix cost three lines.

The control BLOB is likewise **byte-identical to the shipped `glm52-tp8-final2/model.pkt`**.

---

## 5. REJECTED ALTERNATIVES, and why

| candidate | verdict | reason |
|---|---|---|
| **Full g1u1 single pass (85+86 fused, LDS bridge)** | REJECTED | LDS: `bridge 32,768 + DOWN B tile 32,768 = 65,536 B` against a **64,512 B** arena — misses by 1,024 B with perfect aliasing, by 9,216 B without. The only fit (`BN=128` on the DOWN phase) doubles DOWN's n-tiles from 24 to 48 and therefore doubles its epilogue count on an epilogue-dominated kernel. And the prize is 75.4 MB/layer (2.6% of the pair) — it does not touch either 1.611 GB `part` traversal. |
| **86->87 through LDS / registers / a wave-group role split** | IMPOSSIBLE | The k contributions to one token come from k different experts, hence k different m-tiles, hence in general k different workgroups on k different CUs (§1.2). Cross-workgroup by construction; no LDS is shared. |
| **`global_atomic_pk_add_bf16` (aiter's exact instruction), accumulating in bf16 into `dg_tp` pre-seeded with `shared`** | NOT BUILT — it is the FASTER arm and the WORSE numerics | It halves the atomic instruction count and deletes op 87 outright. But the k-way sum then rounds to bf16 after every one of the 8 adds, which is a strictly WORSE error class than `PLOW_MOE_PF_PART16` (one bf16 rounding per slot, before a f32 sum) — and part16 already flipped top-1. It also needs a cross-lane swap: a lane's 16 accumulator elements are 16 different ROWS at one column (`mfma_acc_m` varies, `mfma_acc_n` does not), so packing two adjacent columns into one atomic requires a permute the f32 arm does not. Recorded as the known faster/worse point on the curve, deliberately not taken. |
| **Zeroing the accumulator in op 87 (previous layer) instead of op 83** | REJECTED | Saves nothing (same 201 MB) and needs a program-start zero for the first MoE layer, because layers 0..2 are DENSE and write the same region classically. op 83 is self-contained and needs no cross-layer invariant. |
| **A dedicated zero-fill packet** | REJECTED | A new opcode + dispatch + a gate, for a store that fits inside a packet that already runs first. Fine-grained deps are dead code in every shipped program (`emit_dep_work` has only `#[cfg(test)]` callers), so a new packet is a new COARSE gate on the critical path. |
| **Recovering `tok` with `pidx / k`** | REJECTED | `k` is a runtime uniform, so that is a ~10-instruction reciprocal division per output ELEMENT — +320 instructions on a ~1,800-instruction output tile (+18%). `k` power-of-two + one `v_lshrrev` costs 1. The emit refuses the arm for non-power-of-two `k`. |
| **Shrinking `act.part` from `T*k*H*4` to `T*H*4`** | NOT TAKEN (available) | The fused arm uses only the first `T*H` f32 of the allocation, so 1.41 GB of VRAM per rank is free for the taking. Left alone because an emit-side size change that disagreed with the kernel arm for any reason would be a silent k-fold heap overrun, and VRAM is not the measured constraint. Follow-up, gated on the same predicate as `atom`. |

---

## 6. PRE-REGISTERED PREDICTION (written before the battery ran; §10 grades it)

Recorded up front so the result cannot be read backwards. Two micro-probes on this box, this
session, GPU lock held, at the exact shipped shape (`perf-data/plow-gfx942/probes/`):

**`atomic_scatter_probe.hip`** — `N = 402,653,184` elements (one layer at T=8192), wave64 runs of
64 consecutive columns at random row starts:

| arm | best of 4 | rate |
|---|---:|---:|
| `nontemporal f32 store` -> `part[T*k,H]` 1.611 GB (**the shipped path**) | 0.469 ms | 858 G elem/s, 3,434 GB/s |
| `global_atomic_add_f32` -> `acc[T,H]` 201 MB (**the fused path**) | 1.127 ms | 357 G elem/s |
| `global_atomic_pk_add_bf16` -> 100 MB (aiter's instruction, N/2 ops) | 1.101 ms | 366 G elem/s |
| plain f32 store -> 201 MB (control, cache-resident) | 0.166 ms | 2,431 G elem/s |

**Atomics are 2.4x slower than the store per element in ISOLATION — but that is not the number
that matters.** op 86's packet span at T=8192 is 2,822 µs, so the kernel needs
`402.65 M / 2822 µs = 143 G atomics/s` against a measured ceiling of **357 G/s** — 2.5x headroom.
The atomics are fire-and-forget (§4.2: no `vmcnt` wait), so they cost op 86 only what
backpressure they cause, not their saturated latency.

So the prediction, per MoE layer at T=8192, against the traced baseline
(`glm52-current-cost-decomposition.md` A.1: DOWN 2,822 µs / combine 1,544 µs / router 334 µs):

* op 83 **+59 µs** (201 MB of non-temporal stores at stream rate),
* op 86 **+0 to +658 µs** (0 if the 2.5x atomic headroom absorbs it; 658 if it does not),
* op 87 **−1,140 to −1,400 µs** (1.913 GB in 8 strided streams -> 0.503 GB in one contiguous one;
  at stream rate 144 µs, at its *current* measured 1.24 TB/s 406 µs),
* **net −0.42 to −1.34 ms/layer => −32 to −100 ms of TTFT at 8k (−1.9% to −6.0%).**

That range is the honest expectation. It is NOT the 286 ms the cost decomposition's
"addressable" column carries for this component: that figure is
`measured x (1 - 1/aiter_ratio)`, an aspiration bounded by a rate observed on other code, not a
mechanism-linked estimate. The mechanism here is worth what the table above says it is worth.

### 5.1 One latent correctness improvement, found while reading the shipped path

Under EXPERT PARALLELISM the grouped GEMM skips tiles whose expert is not local
(`if (wb0 == 0ull) continue;`, op_moe.h:2197 — "a null base is the EP 'not my expert' sentinel").
Those slots' rows in `part` are therefore **never written**, and op 87 reads whatever the
previous layer left there. The fused path cannot have that bug: an unwritten slot contributes
exactly `0.0f`, because the accumulator was zeroed.

This does not affect the configuration measured here — GLM-5.2 at TP8 is tensor-parallel, every
expert is local on every rank, and both paths agree. It is recorded because it is a real
difference in EP behaviour and the fused side is the correct one.

---

## 7. MEASURED — micro-probes, GPU lock held, 2026-08-08

### 7.1 Cross-XCD atomic coherence: **COHERENT** (correctness precondition, checked not assumed)

MI300X's L2 is PER-XCD. The fused arm has op 83 non-temporally zero an accumulator from all
304 CUs and op 86 atomically add into it from all 304 CUs, with the k slots of one token in
general on different XCDs. If `__HIP_MEMORY_SCOPE_AGENT` f32 atomics were performed in the
issuing XCD's L2, the result would be silently wrong — and silently wrong in an LLM reads as
fluent text, which is exactly how this campaign's MoE-PF LDS OOB bug survived its gates.

`probes/atomic_coherence_probe.hip`, the exact fused pattern (`T*k = 65,536` rows each adding
1.0f into `acc[row>>3][h]`, `H = 6144`, checked against 8.0f over all 50.3 M elements):

```
rep 0..4: elements != 8.0 -> 0   ok
cross-XCD agent-scope f32 atomic accumulate: COHERENT
```

### 7.2 The combine shape probe — this is the result the design was built on

`probes/combine_shape_probe.hip`, T=8192 H=6144 k=8, best of 4:

| arm | time | vs current |
|---|---:|---:|
| **current** — `k=8` f32 streams at `H*4 = 24 KB` stride, 1.61 GB read | **1.296 ms** | — |
| `part16` — `k=8` bf16 streams, 0.81 GB read (**half the bytes**) | 1.130 ms | **−12.8%** |
| **FUSED** — one contiguous f32 stream, 0.20 GB read | **0.170 ms** | **−86.9%** |
| FUSED + zero the accumulator for the next layer | 0.225 ms | −82.6% |
| zero-only (the op-83 prologue's cost) | **0.047 ms** | |

**This is the §2.3 prediction, measured, and it is the whole argument of this branch.** Halving
the BYTES bought 12.8%. Changing the SHAPE bought 86.9%. `PLOW_MOE_PF_PART16` measured a wash on
this op for a reason that had nothing to do with how much traffic it removed, and that reason
does not apply here.

The probe's `current` arm (1.296 ms) sits close to op 87's traced packet span (1.544 ms/layer at
T=8192, `glm52-current-cost-decomposition.md` A.1), so the model transfers. Combine side, per
layer: **1.296 −> 0.170 ms, minus 0.047 ms of new zero-fill = −1.079 ms/layer.**

### 7.3 Atomic scatter throughput (reproduced; §6's table is the first run of the same probe)

| arm | time | rate |
|---|---:|---:|
| `nontemporal f32 store` -> 1.611 GB `part` (shipped) | 0.468 ms | 861 G elem/s |
| `global_atomic_add_f32` -> 201 MB accumulator (fused) | 1.126 ms | 358 G elem/s |
| `global_atomic_pk_add_bf16` (aiter's instruction, N/2 ops) | 1.095 ms | 368 G elem/s |
| plain store -> 201 MB (control) | 0.166 ms | 2,424 G elem/s |

Two runs of this probe, ~50 minutes apart, agree to 0.2%.

### 7.4 The object under test is the object that RUNS — verified, because this campaign has been burned

`Variant::detect` matches on `GemvFp8`, and GLM-5.2's fp8 is BLOCK-scaled (`GemvFp8Blk`), so the
run loads `interp_prefill_mla_moe_gq.elf` (`variant=Bf16`) — **not** the
`interp_prefill_fp8_mla_moe.elf` that §4's census was first taken on. Re-taken on the object the
serve log shows being opened:

| | control | fused |
|---|---:|---:|
| `d_moe_group_down_pf`, `global_atomic_add_f32` | **0** | **64** |
| `plow_moe_pf_atomic_arm` in the symbol table | absent | present |
| `d_moe_router_topk_pf`, non-temporal `global_store_dword` | 0 | 1 |

Counts are identical to the fp8 row, so §4's ISA reading transfers unchanged — but it is now
established on the loaded object rather than inherited from a sibling one.

---

## 8. MEASURED — served, interleaved A/B

Harness `scripts/bench_speed.sh`, `IN_LENS="4096 8192 16384"`, conc 1, 8 prompts/cell,
`OUTLEN=32`, serve env `PLOW_MLA_PF_V2=1`, ONE server at a time on port 8196, arms strictly
alternating within each round, GPU lock held for the whole battery. Both arms carry the
built-in coherence gate (`PASS` on every arm).

Assets: `glm52-moefuse-{ctl,atom}`, each with its OWN `build.json` beside the packet, so the
`requires` -> marker refusal chain is live. The **control blob is byte-identical to the shipped
`glm52-tp8-final2/model.pkt`** and its round-1 numbers reproduce the campaign's canonical
973 / 1677 / 3627 to within 0.9%.

### All three rounds, TTFT ms

| ctx | ctl r1 | ctl r2 | ctl r3 | **ctl mean** | fused r1 | fused r2 | fused r3 | **fused mean** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 978.1 | 970.6 | 965.1 | **971.3** | 937.8 | 936.5 | 937.8 | **937.4** |
| 8192 | 1680.9 | 1679.8 | 1674.6 | **1678.4** | 1612.8 | 1609.4 | 1612.6 | **1611.6** |
| 16384 | 3657.8 | 3645.9 | 3635.5 | **3646.4** | 3521.1 | 3515.7 | 3515.9 | **3517.6** |

### The result, against the control's own round-to-round spread

| ctx | delta | | **control's own spread** | delta / spread | every fused round < every control round? |
|---:|---:|---:|---:|---:|:--:|
| 4096 | **−33.9 ms** | **−3.49%** | 13.0 ms (1.34%) | **2.6x** | **YES** (937.8 < 965.1) |
| 8192 | **−66.8 ms** | **−3.98%** | 6.3 ms (0.38%) | **10.6x** | **YES** (1612.8 < 1674.6) |
| 16384 | **−128.8 ms** | **−3.53%** | 22.3 ms (0.61%) | **5.8x** | **YES** (3521.1 < 3635.5) |

The fused arm's own spread is 0.14–0.21%, TIGHTER than the control's — which is worth noting
because atomic accumulation is run-to-run nondeterministic and one might have expected the
opposite. **The distributions do not overlap at any context: all 9 fused points are below all 9
control points.** The control also drifts monotonically downward across rounds
(978.1 -> 970.6 -> 965.1 at 4k) — a warm-up trend, and it is the reason the 4k spread is 1.34%
while the fused arm's is 0.14%; the interleaving is what keeps that from biasing the delta.

The result lands inside the pre-registered −32 to −100 ms band at 8k (§6), at −66.8.

TPOT is unchanged (ctl 29.05–29.14 / 29.32–29.37 / 29.76–29.80; fused 29.06–29.17 /
29.34–29.46 / 29.79–29.89). That is the expected result — the arm touches only prefill packets
— and it is a useful negative control: a change that moved TPOT would mean the arm had leaked
somewhere it should not have.

---

## 9. NUMERICS — the gate, and it does NOT pass a character-identical bar

Stated up front, before the transcript: **this arm is not bit-identical and not run-to-run
deterministic, and the gate shows both.** That was predicted in §2.2/§6 from the mechanism (the
k-way f32 sum happens in atomic-arrival order instead of fixed slot order), and it is a finding
to report, not something to tune away.

Method: one server per arm, six prompts at `temperature=0` — five canonical short/medium ones
plus a long free-form essay (~3,050 characters of generation) — and then **the identical set
asked a SECOND time against the SAME server**, which is what separates "the arm changed the
answer" from "the arm cannot repeat itself".

| # | prompt | ctl == ctl(rep) | ctl == fused | fused == fused(rep) |
|---|---|:--:|:--:|:--:|
| 0 | capital of France | **YES** | **YES** | **YES** |
| 1 | name three prime numbers | **YES** | **YES** | **YES** |
| 2 | how a GPU does a matmul, 2 sentences | **YES** | no | **YES** |
| 3 | reverse a singly linked list (code) | **YES** | no | **no** |
| 4 | 17 * 23, show your work | **YES** | no | **no** |
| 5 | MoE routing essay, ~3,050 chars | **YES** | no | **no** |

**The control is byte-deterministic on all six, both passes.** The fused arm is
character-identical on the two short factual prompts, and diverges on the four longer ones —
including from ITSELF on three of them. The pattern is exactly what the mechanism predicts:
longer generations give more opportunities for a near-tied logit to land on the other side of
an f32 rounding difference.

### The divergences are WORDING, not correctness — checked, not assumed

| # | control | fused |
|---|---|---|
| 4 | `17 * 23 = 391` + long multiplication + a difference-of-squares check | `17 * 23 = 391` + long multiplication (`51 = 3x17`, `340 = 20x17`) + the same difference-of-squares trick |
| 2 | "...partitioning the large operation into thousands of smaller, independent dot-products" | "...partitioning the large matrices into smaller blocks and assigning the computation..." |
| 3 | correct `ListNode` + iterative O(n)/O(1) reversal | correct `ListNode` + iterative O(n)/O(1) reversal, different prose |
| 5 | correct MoE essay | correct MoE essay |

Both fused runs of prompt 4 give **391** with correct working. The first divergence in prompt 4
is at character 28, inside `"Here are two ways"` vs `"Here are two different ways"` — i.e. in the
scaffolding, after the answer.

### Verdict on shippability

**This arm should NOT be defaulted on as it stands**, on two grounds that are independent of its
speed:

1. It fails a character-identical gate, which is the bar this campaign has used for every landed
   lever.
2. It is **run-to-run nondeterministic on the same binary and the same server**, which the
   shipped path is not. That is a reproducibility property, not a quality one, and it is the
   more serious of the two: it makes every future A/B on this configuration noisier and makes
   bug reports harder to reproduce.

It is a genuinely GENTLER numerics class than `PLOW_MOE_PF_PART16` (which rounded every slot to
bf16 before summing and flipped top-1 on the very first token of a fixed prompt): here the sum
stays f32 and the two shortest prompts are character-identical. But "gentler" is not "identical".

**What would make it shippable, and it is a bounded piece of work:** the nondeterminism is
entirely in the ARRIVAL ORDER of the k atomic adds. A deterministic variant exists — have op 86
atomically accumulate into `k` FIXED sub-accumulators... no; the cheap deterministic route is to
keep the fused SHAPE for op 87 (which §7.2 shows is where 86.9% of the win lives) while giving
op 86 a deterministic destination. That is the `slot -> token` reduction done by an ordered
tree rather than by atomics, and it is the obvious follow-up. Recorded in §11.

### Negative case — the refusal chain fires

Running the FUSED blob against the CONTROL objects:

```
Error: Device("packet/object MISMATCH: this packet requires PLOW_MOE_PF_ATOMIC=1 but
  /tmp/mf/obj_ctl/interp_prefill_mla_moe_gq.elf was built WITHOUT it — none of
  [\"plow_moe_pf_atomic_arm\"] is in its symbol table. The AMD dispatch's `default:` does not
  trap, so those ops would write nothing and the prefill would complete with garbage instead of
  failing. Rebuild the prefill object with -DPLOW_MOE_PF_ATOMIC=1 ...")
```

That is the PASS for this case. Without it, op 86 would take the `part` scatter branch with
`Cout` pointing at a `[T,H]`-sized accumulator and scatter k-fold past its end.

---

## 10. MEASURED — the in-situ traced census, which is the number this build is about

`PLOW_TRACE_RAW`, `amd-bench --tp 8 --steps 4 --prompt <8192 ids>`, reducer
`scripts/glm52_layer_census.py <prog> <trace>.prefill --layers 6:74`, 590,207 records / 2,021
packets per arm, GPU lock held, arms back to back. Median of the 69 MoE layers L6..L74.

**Device prefill wall at T=8192, independent of the serve path: 1402.9 -> 1329.9 ms = −73.0 ms
(−5.20%).** The control's 1402.9 reproduces the campaign's recorded 1390.0 to 0.9%.

**Both arms emit the SAME first token (373) on this 8,192-token prompt, and all 8 ranks agree in
both.**

### Per MoE layer, the three packets the arm touches

| op | span ctl | span fused | Δ span | busy ctl | busy fused | Δ as perfect-pack TTFT (×75/304) |
|---|---:|---:|---:|---:|---:|---:|
| **`MoeCombinePf`** | 1,516.6 µs | **536.4 µs** | **−980.2 (−64.6%)** | 451,421 | **155,504** | **−73.0 ms** |
| `MoeGroupDownPf` | 2,861.6 | 2,906.4 | **+44.8 (+1.6%)** | 860,390 | 873,879 | **+3.3 ms** |
| `MoeRouterTopkPf` | 338.0 | 356.0 | **+18.0 (+5.3%)** | 97,810 | 103,264 | **+1.3 ms** |
| **net** | | | **−917.4 µs/layer** | | | **−68.4 ms** |

Measured served delta at 8k: **−66.8 ms**. The traced perfect-pack prediction is **−68.4 ms**.
Those agree to 2.4%.

### The negative control: everything else is untouched

| op | span ctl | span fused | Δ |
|---|---:|---:|---:|
| `FlashMlaPrefill` | 3,147.1 | 3,143.0 | −0.1% |
| `MoeGroupGluPf` | 2,415.8 | 2,403.7 | −0.5% |
| `XReduceTwoShot` ×2 | 2,290.9 | 2,284.3 | −0.3% |
| `MlaMergeFold` | 1,587.6 | 1,579.6 | −0.5% |
| `Gemm` ×3 / `GemmMed` / `GemmWide` / `GemmSmall` / `GemmGlu` | | | ≤0.9% |

Nothing outside the three touched packets moves by more than 0.9%, which is what a correctly
scoped change looks like. Layer span **18,034.7 -> 17,008.0 µs (−5.7%)**; total busy
**4,995,235 -> 4,697,384 CU-µs (−6.0%)**; packing efficiency 91.1% -> 90.9% (unchanged, as it
should be — this removes work, it does not repack).

### Three corrections the trace forces on the §6 model — all in the same direction

| quantity | predicted (§6, from standalone probes) | **traced in situ** |
|---|---:|---:|
| op 87 fused span | 203 µs (probe ratio 0.170/1.296) | **536.4 µs** — the probe OVER-predicted the speedup, −64.6% not −86.9% |
| op 86 atomic cost | +0 to +658 µs, inferred residual +254 | **+44.8 µs** — the atomics are nearly free in situ |
| op 83 zero cost | +47 µs (standalone stream rate) | **+18.0 µs** — it overlaps inside the packet |

The op-86 row is the one worth dwelling on: the atomic-scatter probe said atomics are **2.4×
slower per element than the non-temporal store** it replaces, and in the real kernel that costs
**1.6%** of the packet. The reason is the one written down in §6 before the measurement: op 86
needs 143 G atomics/s against a measured 357 G/s ceiling, so the 2.4× shows up only as
backpressure, and the epilogue issues them with no `vmcnt` wait to expose it.

**So the entire win is op 87, and the entire win is its SHAPE.** −980 µs of the −917 µs net.

---


## 11. WHAT REMAINS

1. **A DETERMINISTIC variant of this fusion.** §9 is the only thing standing between this arm
   and a default. §10 shows the win is **entirely** op 87 (−980 µs/layer) and that op 86's
   atomics COST 45 µs — so the obvious construction keeps op 87 fused and reading one contiguous
   stream, and gives op 86 a *deterministic* destination: an ordered slot->token reduction rather
   than an atomic. Since the atomic buys nothing on its own, a deterministic writer that lands
   the same `[T,H]` layout would recover essentially all of the 66.8 ms with a bit-identical
   numerics story. **This is the highest value follow-up in this file, and §10 is what makes it
   look cheap.**
2. **`act.part` can shrink from `T*k*H*4` to `T*H*4`** under the arm: 1.611 GB -> 0.201 GB, i.e.
   **1.41 GB of VRAM per rank returned**. Not taken here (§5) because a size change that
   disagreed with the kernel arm would be a silent k-fold heap overrun; gate it on the same
   predicate as `atom`.
3. **op 87 reads `zero_h`, a `[T,H]` buffer of literal zeros, every MoE layer** — 100.7 MB/layer,
   7.5 GB per 8k chunk, purely to satisfy `d.t[1]` under TP. The kernel already handles
   `residual == nullptr` (`residual ? bf2f(residual[i]) : 0.0f`), so emitting `TENSOR_NONE`
   instead is **bit-identical** and removes one of the fused combine's four streams — which, on
   the §7.2 evidence that this op is stream-COUNT bound, is worth more than its 5% share of the
   bytes suggests. One line, plus the `mla.rs` test that asserts `t[1] == zero`.
4. **`global_atomic_pk_add_bf16`** (aiter's actual instruction) is measured here at 368 G elem/s
   vs the f32 atomic's 358 — i.e. essentially free relative to f32, for half the instructions.
   It is the faster arm and the worse numerics (§5); it only becomes interesting if item 1
   proves impossible.
5. The 85->86 bridge remains closed by LDS arithmetic (§2.1) and is not worth reopening at 75 MB.
