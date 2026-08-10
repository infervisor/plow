# GLM-5.2 MoE grouped prefill: a DETERMINISTIC writer for the fused 86 -> 87 decomposition

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **CDNA3 / MI300X-SPECIFIC** — rests on cross-XCD agent-scope f64 atomic COHERENCE on 8 XCDs, and on f64 atomics being 1.74x slower than f32 here. Both are this silicon. The IMPOSSIBILITY argument in sec 2 (no scheme is bit-identical without reintroducing the part pass) is arithmetic and arch-independent.

2026-08-08, branch `moe-detwrite` (base `b86521c`, worktree `.claude/worktrees/moe-detwrite`).
Follow-up to `glm52-moe-fusion.md` §11 item 1, whose closing claim was that "a deterministic
writer that lands the same `[T,H]` layout would recover essentially all of the 66.8 ms with a
bit-identical numerics story."

**Half of that claim is right and half of it is not, and the half that is wrong is the important
half.** This file builds the deterministic writer, measures it, and shows with arithmetic why
BIT-IDENTITY to the shipped combine is not available at any price the fusion can pay.

Everything below is either (a) disassembled from a built object, (b) measured on this box this
session, or (c) arithmetic over (a)+(b) with the derivation shown.

---

## VERDICT, first

**BUILT, AND IT IS DETERMINISTIC. BIT-IDENTITY TO THE SHIPPED COMBINE IS NOT AVAILABLE AND THIS
FILE PROVES WHY, WITH ARITHMETIC RATHER THAN WITH AN ATTEMPT.**

* `PLOW_MOE_PF_DET` (opt-in, default off, default object BYTE-IDENTICAL on all 18 prefill rows)
  makes the fused 86 -> 87 decomposition **order-independent** by accumulating
  `rint(gate*value * 2^32)` as an integer-valued f64: every partial sum is an integer below 2^53,
  so every add is exact and the k-way total cannot depend on which workgroup arrives first.
* **The mechanism is verified in isolation, on the shipped shape** (`probes/det_accum_probe.hip`,
  402.65 M atomics = one layer): over 12 samples in 2 processes, **every f32-atomic accumulation
  hash differs from every other and every f64 fixed-point hash is the same value**. Cross-XCD
  agent-scope f64 atomics are COHERENT.
* **ISA:** 64 native `global_atomic_add_f64` + 64 `v_rndne_f64` per DOWN kernel, **zero
  `global_atomic_cmpswap_x2`** (no CAS fallback), 256 VGPR / 0 AGPR / 64,560 LDS / spill 2 —
  identical resources to the control.
* **Two free bit-identical wins taken:** op 87 no longer reads `zero_h`, a `[T,H]` buffer of
  literal zeros (**7.5 GB per 8k chunk**, deleted by emitting `TENSOR_NONE` — the kernel always
  spelled the zero residual as a null pointer); and `act.part` drops from `T*k*H*4` to `T*H*8`,
  returning **1.208 GB of VRAM per rank**.
* **DRAM: -1.711 GB per MoE layer per rank** (128 GB per 8k chunk), against the atomic arm's
  -2.416. The gap is the 8-byte accumulator, and §3.2 is why it has to be 8 bytes.
* **Numerics: NOT bit-identical to the shipped combine, and §2 shows no scheme can be** without
  either reintroducing ~78.5% of the `part` DRAM pass (E[in-order prefix] = Σ1/j! = 1.72 of 8),
  stalling on a dependency graph whose edges point the wrong way, or doubling both grouped GEMMs.
  `glm52-moe-fusion.md` §11's "with a bit-identical numerics story" is withdrawn here.
* **THE GATE, which is the point:**
  * run-to-run **BYTE-IDENTICAL logits** across two processes with a full reload between (all five
    dumped `act.logits` tensors), and **6 of 6 character-identical answers** across two passes on
    one server — `PLOW_MOE_PF_ATOMIC` was 3 of 6. **PASS.**
  * `zero_h -> TENSOR_NONE` vs the shipped control: **0 of 19,360 logit entries differ. BIT-IDENTICAL.**
  * `PLOW_MOE_PF_DET` vs the shipped control: **not bit-identical, 2 of 6 character-identical** —
    the same score the atomic arm got, on the same two prompts. The four divergences are wording
    after a correct answer (both arms give `17 * 23 = 391` with correct working). **The arm stays
    OPT-IN; this branch does not default it on.**
* **SPEED: -14.6 / -34.7 / -57.8 ms of served TTFT at 4k / 8k / 16k (-1.51% / -2.07% / -1.59%),
  2.0x / 2.6x / 4.8x the control's own round-to-round spread, and at every context EVERY `det`
  cell is below EVERY `base` cell** (41 surviving cells over 6 rounds). Traced device prefill wall
  -40.1 ms at 8k. **That is 43-52% of the atomic arm's -33.9 / -66.8 / -128.8 ms, and 48% of its
  device wall measured head-to-head this session.**
* **Where the other ~50% went, precisely:** `MoeCombinePf` 1,503.9 -> 424.5 µs/layer (-71.8%, BETTER
  than the atomic arm's -64.6%), but `MoeGroupDownPf` **+581 µs/layer** against the atomic arm's
  +45. f64 atomics are 1.74x slower per operation than f32 ones and the DOWN kernel runs at 1.50x
  rate headroom instead of 2.60x, so 1.74x slower atomics cost 13x more packet time. **The
  non-linearity of fire-and-forget atomic backpressure near the rate ceiling is the transferable
  lesson of this file.**

---

## VERDICT UPDATE, 2026-08-09 — THE NUMERICS BLOCKER IS CLEARED BY MEASUREMENT

The arm above stayed OPT-IN for exactly one reason: it is not CHARACTER-identical to the shipped
combine (2 of 6 answers), and §2 proves no scheme can be bit-identical at any price the fusion can
pay. **Character identity was the wrong instrument.** It cannot distinguish "this reordering
degraded the model" from "this reordering reworded a correct answer", and §2 guarantees the arm
will always fail it.

The right instrument is accuracy, and it now exists (`scripts/twoengine/gsm_paired.py` +
`mcnemar.py`, built for this). Two arms, pinned assets from this file's own session
(`glm52-detw-{base,det}`, objects `/tmp/detw/obj_{ctl,det}`; the det object carries
`plow_moe_pf_det_arm` and the base does not — verified by symbol scan), one server load each, GPU
lock + HSA + coherence gated, **full GSM8K test set, per-question, paired**:

|  | control | det |
|---|---:|---:|
| GSM8K 8-shot greedy, n=1319 | **1268/1319 = 0.9613** | **1268/1319 = 0.9613** |
| errors | 0 | 0 |
| paired difference | — | **+0.00 pp** |

    contingency          arm wrong   arm right
      control wrong           41          10
      control right           10        1258

    discordant b = 10, c = 10   McNemar exact two-sided p = 1.0000
    minimum detectable difference at this discordance: ~0.66 pp (2 sigma)

**The arm changes 24 of 1319 predicted numbers (1.8%) and flips 20 correctness outcomes (1.5%) —
ten each way, netting exactly zero.** That is the shape a numerically-neutral reordering should
have, and it is what §2 predicted: the fusion changes the order of a k-way sum, so it moves
borderline answers in both directions without a systematic bias.

### Why the earlier n=100 read was not evidence

The same A/B at `run_plow.sh`'s default n=100 returned **0.970 vs 0.950**, which reads as a 2 pp
regression and is nothing of the kind: 0.72 sigma unpaired, McNemar p ~= 0.50. At n=100 this
instrument can detect gross damage and nothing finer. Both numbers are in
`scripts/twoengine/` output; the n=100 one should not be quoted.

### Speed, re-measured on the same two loads

| ctx | control TTFT | det TTFT | Δ | control spread |
|---|---:|---:|---:|---:|
| 1024 | 323.3 | 317.5 | **−1.79%** | 1.1% |
| 4096 | 719.9 | 699.2 | **−2.88%** | 0.1% |
| 8192 | 1622.0 | 1591.3 | **−1.89%** | 0.4% |
| 16384 | 3534.0 | 3475.5 | **−1.66%** | 0.3% |
| TPOT | 26.568 | 26.573 | +0.02% | — |

Reproduces this file's original −1.51/−2.07/−1.59% on a different session and different objects.
TPOT being a null is CORRECT, not a miss — the arm is prefill-only.

### Where that leaves the arm

Everything that was ever asked of it is now answered: **deterministic** (run-to-run byte-identical
logits across two processes with a full reload), **faster** (−1.7…−2.9% TTFT at 3–29× the control's
own spread), **−1.711 GB of DRAM per MoE layer per rank**, and **no measurable accuracy cost**
against an instrument that could have seen 0.66 pp.

The one thing it is not, and cannot be, is bit-identical. It changes 1.8% of served answers. That
is a **product** decision rather than a correctness one, and it is the only thing still standing
between this arm and a default. Unlike `PLOW_MLA_PF_SV` / `PLOW_MOE_PF_EPI` (object-only,
bit-identical, defaulted on 2026-08-09), `PLOW_MOE_PF_DET` needs the blob AND the object together,
so flipping it re-emits every asset — with a loud refusal, not silent corruption, on a mismatch
(`plow_moe_pf_det_arm`).

Recommendation: **adopt**, and if output stability against pinned goldens matters more than 2% of
TTFT, keep it opt-in and say so explicitly rather than leaving it off by inertia.

---

## 1. THE QUESTION THE FUSION REPORT LEFT OPEN

`PLOW_MOE_PF_ATOMIC` fuses ops 86 -> 87: op 86 stops scattering `part[T*k, H]` and instead adds
each routed-expert contribution straight into a `[T, H]` accumulator, so op 87 reads ONE stream
instead of `k`. It measured **-66.8 ms of TTFT at 8k (-4.0%)**, traced and served agreeing to
2.4%, and it CANNOT SHIP: the k-way sum runs in atomic arrival order, so the arm is not
bit-identical to the shipped combine and — worse — **not reproducible against itself** (3 of 6
gate prompts changed between two passes on the same server).

The trace decomposed that win completely:

| op | span ctl | span fused | Δ |
|---|---:|---:|---:|
| `MoeCombinePf` (87) | 1,516.6 µs | 536.4 µs | **-980.2** |
| `MoeGroupDownPf` (86) | 2,861.6 | 2,906.4 | +44.8 |
| `MoeRouterTopkPf` (83) | 338.0 | 356.0 | +18.0 |
| net | | | **-917.4 µs/layer** |

So the atomic **costs** 44.8 µs and buys nothing on its own; the whole win is op 87. That is what
makes a deterministic writer look cheap, and it is the premise this file starts from.

### 1.1 SHARPENING the fusion report's reading of its own probe — and it matters for the price

`glm52-moe-fusion.md` §7.2 attributes the combine's win to its SHAPE rather than its bytes.
That is directionally right, and the probe supports a stronger, quantitative statement once its
`shared` read and `out` write (0.1007 GB each) are counted alongside the `part`/accumulator
stream:

| `combine_shape_probe.hip` arm | bytes moved | time | **achieved rate** |
|---|---:|---:|---:|
| current — k=8 f32 streams at 24 KB stride | 1.8124 GB | 1.296 ms | **1.399 TB/s** |
| `part16` — k=8 bf16 streams | 1.0067 | 1.130 | **0.891** |
| FUSED — one contiguous f32 stream | 0.4027 | 0.170 | **2.369** |
| zero-only (op 83's prologue) | 0.2013 | 0.047 | 4.283 |

So the fused arm's 7.62x is **4.50x from moving fewer bytes and 1.69x from moving them better**,
and `part16`'s wash is explained the same way in reverse: it moved 1.80x fewer bytes and gave
1.57x of it straight back in rate, because a 64-lane wave pulling 2-byte elements at a 24 KB
stride moves half as much per memory request.

**But that 1.69x does not survive contact with the real kernel, and the report needs the in-situ
model, not the probe's.** The probe takes best-of-4 over a working set the fused arm shrinks to
0.4 GB against MI300X's **256 MB memory-side Infinity Cache**, so up to ~64% of its traffic can be
served from cache on a repeat where only ~14% of the strided arm's can. In situ the two traced
points are:

```
shipped   1.9131 GB in 1,516.6 us   = 1.26 TB/s
fused     0.5034 GB in   536.4 us   = 0.94 TB/s     <-- the SMALLER arm has the LOWER rate
```

Two points cannot separate a fixed cost from a structure bonus, so state both models and check
they agree where it matters:

| model | fit | predicts DET's op 87 (0.6041 GB, contiguous) |
|---|---|---:|
| (a) common slope + fixed floor | `t = 186.4 µs + bytes / 1.438 TB/s` | **607 µs** |
| (b) no floor, per-structure rate | contiguous arm's own 0.938 TB/s | **644 µs** |

**They agree to 6%**, which is enough to price the design. What they agree on is the thing that
matters: **within the contiguous structure the cost is linear in bytes**, so an accumulator that
is twice as wide costs exactly its extra bytes — no more, but no less either. That is the whole
price of §3, and §6.4 pre-registers it.

What they also agree on is the corollary that kills §3.4: op 87 is ALREADY achieving 1.26 TB/s on
the strided read, i.e. 88% of model (a)'s slope, so a pure layout change — one that rearranges
`T*k*H*4` bytes without deleting any — has almost no headroom to buy.

---

## 2. WHY BIT-IDENTITY IS NOT AVAILABLE — the load-bearing negative result

The shipped combine computes, per output element,

```
acc = 0 ; acc += shared ; for j in 0..k-1: acc += part[tok*k + j][h]     // f32, FIXED slot order
```

Bit-identity requires reproducing that association exactly: the k addends summed in f32 in
**slot** order. Slot order is `wl[rank]` from `d_moe_router_topk`, i.e. **descending router
score** (`rank(e) = #{f : key_f > key_e}`, rank 0 = highest packed key).

Any in-place accumulator, on the other hand, receives the k contributions in **row** order, and
row order is **ascending expert id**: `d_moe_align_pf` histograms the `T*k` slots by expert and
lays each expert's rows out contiguously in ascending `e`, so a token's slot for expert 3 always
occupies a lower gathered row — hence a lower m-tile, hence a lower `lin` — than its slot for
expert 200.

**Those two orders are uncorrelated.** The router's rank is a function of the score; the align's
position is a function of the expert id. So:

### 2.1 Deferral (accumulate in order where you can, spill the rest to `part`)

Let each contribution commit to the accumulator only when it is its turn (`prog[t][c] == slot`)
and otherwise fall back to the `part` scatter, with op 87 finishing the sum from `part`. That is
bit-identical *and* deadlock-free — deferral cascades, so what lands in the accumulator is always
a prefix `0..m-1` in slot order and op 87 adds `m..k-1` from `part`.

How big is `m`? Contribution `j` commits iff it arrives after `j-1` has committed, so `m` is the
length of the initial increasing run of an (effectively random) arrival permutation:

```
E[m] = Σ_{j=1..k} 1/j!  =  1 + 1/2 + 1/6 + 1/24 + 1/120 + 1/720 + 1/5040 + 1/40320
     = 1.71828  of  k = 8
```

So **78.5% of `part` rows are still written by op 86 and still read by op 87**. The scheme keeps
the whole 1.611 GB allocation, keeps 78.5% of both traversals, adds a data-dependent gather to
op 87, and recovers at most `0.215 x 980 = 211 µs/layer ≈ 15 ms` of the 66.8. **Rejected on
arithmetic.**

### 2.2 Blocking (spin until it is my turn, in slot order)

Deadlock-freedom for an in-place ordered commit rests on one property: every dependency must
point BACKWARD in the tile order each workgroup walks (`for (lin = slice; lin < n_tiles; lin +=
nblk)`, `mt = lin / tn`, so `lin` is monotone in the m-tile). Then the globally lowest unfinished
tile always has its predecessors done and its owner is sitting on it — progress is guaranteed.

In slot order that property does not hold: slot `j-1` of a token is a *randomly chosen* expert
relative to slot `j`, so its tile is behind or ahead with roughly equal probability. A forward
edge makes a cycle possible, and a hang in a persistent cooperative megakernel is not a failed
A/B, it is a wedged box. Even where it is safe the wait is long: a token's 8 experts are ~32
expert-ids apart on average, i.e. ~128 m-tiles x 24 n-tiles ≈ 3,000 tiles, against an in-flight
window of ~304. **Rejected.**

### 2.3 Making the two orders agree by re-grouping the rows

If `d_moe_align_pf` grouped rows by `(slot, expert)` instead of `(expert)`, row order and slot
order would coincide and §2.2 would be free. The cost is the padding: `k * E = 2048` groups over
`T*k = 65,536` rows is **32 rows per group**, padded up to `MPF_BM = 64` — a 2x m-tile count in
BOTH grouped GEMMs.

```
op 85 MoeGroupGluPf   2,415.8 µs/layer  ->  ~4,832
op 86 MoeGroupDownPf  2,861.6           ->  ~5,723
                                             +5.28 ms/layer   against a -0.98 ms prize
```

**Rejected, by 5.4x.**

### 2.4 The conclusion, stated plainly

**The k-way reduction can be moved upstream of op 87, or it can keep the shipped f32 association,
but not both.** Bit-identity is not a matter of engineering effort here; it is incompatible with
the mechanism that produces the win. `glm52-moe-fusion.md` §11's "with a bit-identical numerics
story" is therefore withdrawn, and this file is the arithmetic behind the withdrawal.

What IS available — and what this branch builds — is the strictly weaker but still decisive
property the atomic arm lacks: **a result that does not depend on arrival order at all.**

---

## 3. DESIGN — `PLOW_MOE_PF_DET`

f32 addition is commutative but **not associative**, and that is the entire defect. Integer
addition is both. So:

| packet | shipped | `PLOW_MOE_PF_DET` |
|---|---|---|
| 83 `MoeRouterTopkPf` | router top-k | + grid-strided zero of `acc[T,H]` **f64** (`t2`/`i0`) |
| 86 `MoeGroupDownPf` | `nt_store(gate*v, &part[pidx*H+nn])` | `atomicAdd(&acc[(pidx>>log2 k)*H+nn], rint(gate*v * 2^32))` as **f64** (`i5 = log2(k)+1`) |
| 87 `MoeCombinePf` | k=8 strided f32 streams | same kernel, `k = 1`, one contiguous f64 stream, `* 2^-32` (`i4 = 1`) |

Every partial sum is an **integer** of magnitude below 2^53, which f64 represents exactly, so
every add is exact and the total is a function of the *set* of contributions, not their order.
Run-to-run bit-reproducibility is a property of the arithmetic, not of the scheduler.

**The exactness bound, as an inequality rather than a hope.** Contributions clamp to
`±MPF_DET_CLAMP = 2^17` before scaling, so `|addend| ≤ 2^17 * 2^32 = 2^49` and the k-way total is
bounded by `k * 2^49 ≤ 2^53` for `k ≤ 16`. The emit refuses `k > 16` and any non-power-of-two `k`
(the token index is `pidx >> log2(k)`, one shift).

**Quantisation** is `2^-32 = 2.33e-10` ABSOLUTE. The layer output is rounded to bf16 (8 mantissa
bits) two lines later, so a contribution below ~1e-9 loses relative precision it could not have
carried into the output anyway. The clamp is a guard, not a transform: activations here live near
1, and 2^17 = 131,072.

**The one behavioural difference that is not a re-association:** a NaN or an infinity clamps
(IEEE `maxNum`/`minNum` return the non-NaN operand) instead of poisoning the accumulator row. On
a healthy model neither occurs; it is recorded because it is a real difference.

### 3.1 Why f64 and not a 64-bit integer

`(long long)` of an f64 has no single gfx9 instruction and expands to a ~10-instruction sequence
per output element. An integer-VALUED f64 needs `v_rndne_f64` — one instruction — and
`global_atomic_add_f64` is native on gfx942. The two representations are arithmetically the same
fixed-point accumulator; the f64 one is 5 VALU cheaper per element.

### 3.2 Why not a 32-bit fixed-point accumulator — which would cost nothing over the atomic arm

A u32 fixed-point accumulator has 32 bits of ABSOLUTE range to cover both the largest element in
the layer and the resolution the smallest one needs. Pick the scale so that values near 1 keep
bf16-grade precision (say 2^12) and the representable maximum is 2^19; pick it so the maximum is
safe (2^30) and an element whose sum is 1e-3 carries ~12% error. There is no setting that is
simultaneously safe and accurate without a per-layer calibration this campaign does not have.
**The 8-byte accumulator is the price of not making an assumption about activation magnitude**,
and §5 prices it exactly: +0.201 GB zeroed in op 83, +0.402 GB of RMW in op 86 and +0.201 GB read
in op 87 — **+0.805 GB per layer per rank** against the atomic arm, which is the entire gap
between this arm's recovery and 100%.

### 3.3 The alternative that was NOT built, and why — an expert-ORDER ticket

There is a deterministic scheme that keeps the accumulator at 4 bytes: order the k adds by
ascending expert id (which IS the row order, so §2.2's progress argument holds) with a
per-`(token, n-tile)` ticket. Its cost is not arithmetic, it is structure:

* a per-slot expert RANK must be precomputed (free in op 83, one `k^2 = 64`-comparison pass per
  token) and carried to op 86 — cheapest as high bits of `row_partidx`, which op 86 already loads;
* a `T x (H/MPF_BN) = 8192 x 24` progress array, zeroed by op 83;
* the tile's 64 rows must ALL be at their turn before any wave stores, so the epilogue gains two
  `__syncthreads()` and a 64-lane spin + 64-lane release per output tile;
* and the deadlock-freedom proof requires that **no tile is ever skipped** — which the expert
  parallelism sentinel `if (wb0 == 0ull) continue;` (op_moe.h:2197) violates outright. Under EP a
  skipped expert never bumps its ticket and every successor spins forever.

And the payoff is the same numerics class as what this branch ships: expert order is not slot
order, so it is **not bit-identical either**. ~250 lines and a hang hazard, for the same
determinism the arithmetic gives for free. **Rejected on that ratio**, and recorded here so the
next reader does not have to re-derive it.

### 3.4 Deferred, not rejected: a BIT-IDENTICAL block-interleaved `part` layout

This is the only candidate that would be bit-identical, because it leaves the k-way sum in op 87
in slot order and changes nothing but addressing. The naive transpose `part[t][h][j]` is fatal on
the producer side — a DOWN tile would write one dword per 32 B write sector, the other 7 dwords
belonging to 7 workgroups thousands of tiles away in the schedule (§2.2) against a buffer 6x
larger than the Infinity Cache, so **1.611 GB of data becomes 12.9 GB of write sectors, +8.1
ms/layer**. Rejected.

A BLOCK-interleaved `part[t][h/G][j][g]` with `G = 8` fixes exactly that: a DOWN tile writes `G`
consecutive dwords = one full 32 B sector, so there is no amplification, and op 87's wave reads
`G*k*4 = 256 B` blocks of which it uses every byte. It stays bit-identical.

**What it is worth, honestly bounded:** it does not delete a single byte, and §1.1 shows op 87
already runs the strided read at 1.26 TB/s — 88% of the slope a contiguous fit gives it. So the
whole available prize is that last ~12%, about **-170 µs/layer, roughly -13 ms**, and part of even
that is a fixed per-packet floor no layout can touch. Against the fusion's -800-odd µs/layer it is
a sixth of the prize, for an emit-side layout change that touches op 84's `row_partidx` contract,
op 86's epilogue and op 87's indexing.

Recorded here with the arithmetic because it is the ONLY bit-identical lever left on this op, and
a future reader should know both that it exists and that it is small.

### 3.5 Rejected: batching the tokens so a small staging buffer stays cache-resident

Process tokens in batches of `T0` so the `T0*k*H*4` staging fits MI300X's 256 MB Infinity Cache:
`T0 ≤ 1365`, i.e. 6 batches at T=8192. Each batch re-streams the layer's expert weights, which are
1.208 GB (gate+up 0.805 + down 0.403): **+6.0 GB per layer** against 3.2 GB removed. Rejected.

---

## 4. ISA — in the object the run actually LOADS

`Variant::detect` matches on `GemvFp8`, and GLM-5.2's fp8 is BLOCK-scaled (`GemvFp8Blk`), so the
run opens `interp_prefill_mla_moe_gq.elf`. Both censuses below are taken on that object, built by
the canonical recipe (`PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1`) with and
without `PLOW_MOE_PF_DET=1`.

### 4.1 Resources — unchanged

| | control | DET |
|---|---:|---:|
| `.vgpr_count` | 256 | 256 |
| `.agpr_count` | 0 | 0 |
| `.group_segment_fixed_size` | 64,560 | 64,560 |
| `.vgpr_spill_count` | 2 | 2 |

VGPR is at the cap in both, so a real register increase would have shown as spill; spill is 2
either way. The f64 accumulator lives in DRAM, not in registers, and the f64 atomic needs VGPR
PAIRS for both its value and its 64-bit address — the compiler took them from the same pool the
f32 store was using, and neither the LDS arena nor occupancy moved.

### 4.2 The DET epilogue, per output element

```
ds_bpermute_b32  v70, v192, v225      ; row_partidx   (existing PLOW_MOE_PF_EPI hoist)
ds_bpermute_b32  v65, v192, v226      ; row_gate      (existing)
v_cmp_ne_u32_e64 s[4:5], -1, v70      ; pad-row test  (existing)
s_and_saveexec_b64 ...
v_mul_f32_e32    v65, v40, v65        ; gate * value  (existing)
v_max_f32_e32    v65, 0xc8000000, v65 ; clamp  -2^17   <-- added
v_min_f32_e32    v65, 0x48000000, v65 ; clamp  +2^17   <-- added
v_lshrrev_b32_e32 v67, v22, v70       ; pidx >> log2(k)
v_cvt_f64_f32_e32 v[80:81], v65                        ; <-- added
v_mad_u64_u32    v[70:71], s[8:9], v67, v14, 0
v_ldexp_f64      v[80:81], v[80:81], 32   ; * 2^32, EXACT  <-- added
v_lshl_add_u64   v[70:71], v[70:71], 3, v[68:69]       ; 8-byte stride
v_rndne_f64_e32  v[80:81], v[80:81]   ; round to integer  <-- added
global_atomic_add_f64 v[70:71], v[80:81], off
```

Four things to read off it:

* **`global_atomic_add_f64`, natively — NOT a CAS loop.** `global_atomic_cmpswap_x2` count is 0.
  This was the single build risk of the design (HIP falls back to a compare-and-swap retry loop
  for f64 atomics on targets or address spaces that lack the instruction) and it is settled by
  disassembly rather than by the documentation.
* **No return value and no `s_waitcnt vmcnt` in the chain** — fire-and-forget, exactly as the f32
  arm. Nothing drains on the atomic, so it costs only the backpressure it causes.
* **`sc0`/`sc1`/`nt` all clear** (encoding `DD3C8000 007F5046`), LLVM's gfx942 memory model for
  `__HIP_MEMORY_SCOPE_AGENT`: the atomic is performed at a device coherence point, which is a
  CORRECTNESS precondition here because MI300X's L2 is per-XCD and the k contributions come from
  k different XCDs. §6.1 checks it on hardware instead of trusting the model.
* **The scale is `v_ldexp_f64 ..., 32`, not a multiply** — the compiler recognised the power of
  two, so the scaling is exact by construction and free.

Five added VALU per output element (2 clamp, 1 convert, 1 ldexp, 1 round) against the atomic
arm's one shift.

### 4.3 Static census, on `interp_prefill_mla_moe_gq.elf`

| | control | DET | (atomic, for reference) |
|---|---:|---:|---:|
| `d_moe_group_down_pf` total instructions | 3,757 | 5,067 | 4,744 |
| `global_atomic_add_f64` | 0 | **64** | 0 |
| `global_atomic_add_f32` | 0 | 0 | 64 |
| `v_rndne_f64` | 0 | **64** | 0 |
| `global_atomic_cmpswap_x2` | 0 | **0** | 0 |
| `global_store_dword` (the f32 `part` scatter) | 64 | 64 (runtime-dead) | 64 |
| `ds_bpermute_b32` | 256 | 384 | 384 |
| `d_moe_router_topk_pf` total | 1,918 | 1,953 | 1,952 |
| ... its non-temporal store | 0 | **1 x `global_store_dwordx2 ... nt`** | 1 x `dword` |
| `d_moe_combine_pf` total | 255 | 275 | 255 |

64 atomics = 2 template instantiations (fp8, bf16) x `SM*SN*16 = 32` elements — one per output
element, the same count as the store it replaces. The k-loop is byte-identical between the two
objects.

### 4.4 Default object byte-identity — verified, not asserted

Built `b86521c` and this branch **at the same path** (hipcc embeds the source path, so any other
comparison is meaningless), canonical axes, `PLOW_MOE_PF_DET` unset. All 18 prefill objects:

```
BYTE-IDENTICAL  interp_prefill.elf                    BYTE-IDENTICAL  interp_prefill_fp8_mla_moe.elf
BYTE-IDENTICAL  interp_prefill_gq.elf                 BYTE-IDENTICAL  interp_prefill_fp8_mla_moe_gq.elf
BYTE-IDENTICAL  interp_prefill_fp8.elf                BYTE-IDENTICAL  interp_prefill_fp8kv*.elf  (6)
BYTE-IDENTICAL  interp_prefill_fp8_gq.elf             BYTE-IDENTICAL  interp_prefill_mla*.elf    (4)
BYTE-IDENTICAL  interp_prefill_fp8_mla.elf            BYTE-IDENTICAL  interp_prefill_fp8_mla_gq.elf
```

Same discipline as the atomic arm: every new parameter on `d_moe_group_pf_t`,
`d_moe_group_down_pf`, `d_moe_router_topk_pf` and `d_moe_combine_pf` is INSIDE the `#if`, so the
default object's mangled names — and therefore its `.strtab`, and therefore every byte of it —
are unchanged.

The **control BLOB is byte-identical to the shipped `glm52-tp8-final2/model.pkt`**, which also
re-confirms the campaign's emit recipe:
`GLM_FULL=1 PLOW_MLA_PREFILL=full GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1
PLOW_GLM_DSA=0 PLOW_GLM_FUSE_B1=1 PLOW_GLM_GEMV_WG=152 PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=2
PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1`.

### 4.5 The emitted packets, on the 8,192-row program

```
                                       base            ctlz            det
act.part                          1,610,612,736   1,610,612,736     402,653,184
op 83 MoeRouterTopkPf  atom_acc        —               —             act.part (atom_h=6144)
op 86 MoeGroupDownPf                det_ksh=0       det_ksh=0     det_ksh=4  (x75 MoE layers)
                                                                  det_ksh=0  (x3 dense layers)
op 87 MoeCombinePf     residual     act.zero_h        —               —
                       k                 8               8               1
                       det               0               0        1 (x75) / 0 (x3 dense)
```

The three DENSE layers (`layer < first_k_dense_replace`) keep the shipped `part` scatter under
every arm — they are the degenerate 1-expert construction, there is nothing to reduce — and they
fit inside the f64 allocation because `T*H*4 < T*H*8`.

---

## 5. DRAM BYTES MOVED, per MoE layer per rank, T=8192

`T=8192, H=6144, I_moe=256, E=256, k=8`. Decimal GB, same basis as `glm52-moe-fusion.md` §3.

| stream | shipped | atomic | **DET** |
|---|---:|---:|---:|
| gate+up fp8 (op 85 B) | 0.8053 | 0.8053 | 0.8053 |
| A gather, distinct (op 85) | 0.1007 | 0.1007 | 0.1007 |
| `fu_g` write / read (85 -> 86) | 0.0377 / 0.0377 | 0.0377 / 0.0377 | 0.0377 / 0.0377 |
| down fp8 (op 86 B) | 0.4027 | 0.4027 | 0.4027 |
| `part` f32 scatter (op 86 W) | **1.6106** | — | — |
| accumulator RMW (op 86 atomics) | — | 0.4027 | **0.8053** |
| `part` read-back (op 87 R) | **1.6106** | — | — |
| accumulator read (op 87 R) | — | 0.2013 | **0.4027** |
| accumulator zero (op 83 W) | — | 0.2013 | **0.4027** |
| `zero_h` read (op 87) | 0.1007 | 0.1007 | **— (deleted, §7)** |
| `shared` read (op 87) | 0.1007 | 0.1007 | 0.1007 |
| `dg_tp` write (op 87) | 0.1007 | 0.1007 | 0.1007 |
| **chain (83+85+86+87)** | **4.9074** | **2.4915** | **3.1962** |
| delta vs shipped | — | **-2.416** | **-1.711** |

**-1.711 GB per MoE layer per rank; 128 GB per 8k prefill chunk over 75 layers.** The DET arm
gives back 0.604 GB/layer of the atomic arm's saving — 0.201 in each of op 83's zero, op 86's RMW
(x2, read+write) and op 87's read — and takes 0.101 back from `zero_h`.

**VRAM: `act.part` drops from `T*k*H*4 = 1.611 GB` to `T*H*8 = 0.403 GB` per rank — 1.208 GB
returned.** Sized from `moe_pf_fuse`, the same function the packet fields come from, because a
size that disagreed with the kernel arm would be a silent heap overrun rather than a fault (which
is exactly why the atomic branch left the allocation alone).

---

## 6. MEASURED — the probe (`probes/det_accum_probe.hip`), GPU lock held, 2026-08-08

At the exact shipped shape: `T=8192, H=6144, k=8` — 50.33 M accumulator elements, **402.65 M
atomics**, i.e. one MoE layer's worth.

### 6.1 Cross-XCD coherence — COHERENT, for f64 as well as f32

MI300X's L2 is PER-XCD. The arm has op 83 non-temporally zero an accumulator from all 304 CUs and
op 86 atomically add into it from all 304 CUs, with the k slots of a token in general on different
XCDs. `global_atomic_add_f64` is a DIFFERENT instruction from the f32 one the fusion report
checked, so it gets its own check rather than an inference.

```
f32 rep 0..2: elements != 8.0 -> 0   ok
f64 rep 0..2: elements != 8.0 -> 0   ok
cross-XCD agent-scope atomic accumulate: COHERENT
```

### 6.2 Run-to-run determinism — this is the result the arm exists for

Six repetitions of the same accumulation with binade-spanning contributions (`2^-8 .. 2^7`, so a
re-association is visible), hashed with an ORDER-INDEPENDENT integer hash over the raw bits — the
hash cannot manufacture the agreement it is testing for.

```
rep 0  f32 hash 1a7f19864be11247   f64-fixed hash aa47134d2cfb6000
rep 1  f32 hash 2c6059cf17a3a0f4   f64-fixed hash aa47134d2cfb6000
rep 2  f32 hash 3317e4c31c6b6a21   f64-fixed hash aa47134d2cfb6000
rep 3  f32 hash 8f85f665e9a0b0d4   f64-fixed hash aa47134d2cfb6000
rep 4  f32 hash 55b114af658aa2dd   f64-fixed hash aa47134d2cfb6000
rep 5  f32 hash 106bf193f6b7f86e   f64-fixed hash aa47134d2cfb6000

f32 atomic       : 5 of 5 reps DIFFER from rep 0  -> NONDETERMINISTIC
f64 fixed-point  : 0 of 5 reps DIFFER from rep 0  -> BIT-REPRODUCIBLE
```

**Every f32 repetition differs from every other; every f64 fixed-point repetition is bit-exactly
equal.** That is `PLOW_MOE_PF_ATOMIC`'s defect and `PLOW_MOE_PF_DET`'s fix, isolated from the
model, on the shipped shape, on this box.

The probe was run twice, in two separate processes ~9 minutes apart. **All 12 f32 hashes are
distinct; both runs' f64 hashes are the same value `aa47134d2cfb6000`** — reproducibility across
processes, not just within one. Rates agree between the two runs to 0.2% (366.3/365.6 G/s f32,
210.7/210.6 G/s f64).

### 6.3 Atomic rate — the cost, and the headroom that has to absorb it

| arm | best of 4 | rate |
|---|---:|---:|
| `global_atomic_add_f32` -> 201 MB accumulator | 1.099 ms | 366.3 G atomics/s |
| **`global_atomic_add_f64` -> 402 MB accumulator** | **1.911 ms** | **210.7 G atomics/s** (1.74x the f32 time) |

op 86's packet span at T=8192 is 2,861.6 µs, so the kernel must sustain
`402.65 M / 2861.6 µs = 140.7 G atomics/s`. Against the f32 ceiling that is **2.60x headroom**
(and the fusion report measured the resulting cost at +44.8 µs, 1.6%). Against the f64 ceiling it
is **1.50x** — still headroom, but 1.7x less of it, and the atomics are fire-and-forget so what
shows up is backpressure rather than latency.

### 6.4 PRE-REGISTERED, written from §1.1 while the traced battery was still queued

In-situ bytes per MoE layer for op 87 (`part`-or-accumulator + `zero_h` + `shared` + `dg_tp`),
priced by §1.1's two models:

| arm | bytes | model (a) | model (b) | taken as |
|---|---:|---:|---:|---|
| shipped (k=8 strided) | 1.9131 GB | — | — | **1,516.6 µs** (measured) |
| `ctlz` (k=8 strided, `zero_h` deleted) | 1.8124 | 1,447 | 1,437 | **~1,440 µs** |
| `PLOW_MOE_PF_ATOMIC` (1 contiguous f32) | 0.5034 | — | — | 536.4 µs (measured) |
| **`PLOW_MOE_PF_DET` (1 contiguous f64, no `zero_h`)** | 0.6041 | 607 | 644 | **~610-645 µs** |

Then, per MoE layer at T=8192, against the shipped control:

* op 87 **-870 to -905 µs**, of which about **-77 µs is `zero_h`** and the rest is the fusion,
* op 83 **~+36 µs** — the zero doubles, so the atomic arm's +18.0 doubles,
* op 86 **+80 to +250 µs** — the f64 atomic has 1.50x headroom where the f32 had 2.60x (§6.3),
  and the atomic arm's measured cost at 2.60x headroom was +44.8 µs;
* **net -0.58 to -0.79 ms/layer => -44 to -59 ms of TTFT at 8k**, i.e. **65% to 89% of the atomic
  arm's -66.8 ms**, with about -6 ms of that being the bit-identical `zero_h` win.

§8 grades this.

## 7. THE TWO FREE BIT-IDENTICAL WINS

### 7.1 `zero_h` -> `TENSOR_NONE` — 7.5 GB per 8k chunk of literal zeros, deleted

Under TP the prefill combine must use a ZERO residual (the real residual `xmid` is added AFTER
the all-reduce, or `XReduceTwoShot` would sum it `tp` times), and the emit spelled that as
`act.zero_h` — a materialised `[T, H]` bf16 buffer of literal zeros, **100.7 MB read per layer
per rank at T=8192, 7.5 GB per 8k chunk over the 75 MoE layers** — in order to add `0.0f`.

`d_moe_combine_pf` has always spelled the zero residual as a null pointer
(`residual ? bf2f(residual[i]) : 0.0f`) and `TEN()` already maps `PLOW_TENSOR_NONE` to `nullptr`,
so emitting `TENSOR_NONE` is **BIT-IDENTICAL by inspection of that one ternary**. It removes one
of four input streams from the op this whole exercise has shown to be byte-bound.

Landed unconditionally (commit 2 of this branch), so it is in BOTH arms of the A/B below and the
DET delta does not contain it; §8 prices it separately from the trace. `act.zero_h` stays
allocated — the DECODE `MoeCombine` (op 82) still names it, and that is one row, not T.

### 7.2 `act.part` -> `T*H*8` — 1.208 GB of VRAM per rank

Folded into the DET commit rather than left as a follow-up, because the safe way to do it is to
make the SIZE and the PACKET FIELD come from one function (`moe_pf_fuse`) — the hazard the atomic
report flagged ("an emit-side size change that disagreed with the kernel arm would be a silent
k-fold heap overrun") is a hazard of having two predicates, not of changing the size.

| arm | `act.part` | vs shipped |
|---|---:|---:|
| shipped / ctlz | 1,610,612,736 | — |
| `PLOW_MOE_PF_ATOMIC` (f32 acc) | 201,326,592 | -1.409 GB |
| **`PLOW_MOE_PF_DET` (f64 acc)** | **402,653,184** | **-1.208 GB** |

Verified on the emitted blob with `plowrt disasm --tensors`, not asserted.

## 8. MEASURED — the traced in-situ census

`PLOW_TRACE_RAW`, `amd-bench --tp 8 --steps 4 --prompt <8192 ids>`, reducer
`scripts/glm52_layer_census.py <prog> <trace>.prefill --layers 6:74`, 590,207 records / 2,021
packets per arm, GPU lock held AND an idle gate (no foreign `plowrt`, all 8 GPUs at 0%) in front
of every arm. Median of the 69 MoE layers L6..L74.

**Device prefill wall at T=8192, independent of the serve path:**

| arm | wall | vs shipped | first token |
|---|---:|---:|---|
| `base` (shipped control) | **1,401.5 ms** | — | 373, all 8 ranks agree |
| `ctlz` (`zero_h` deleted, bit-identical) | **1,396.8** | **-4.7 ms** | 373 |
| `det` (`PLOW_MOE_PF_DET`) | **1,361.4** | **-40.1 ms (-2.86%)** | 373 |

The control's 1,401.5 reproduces the fusion report's 1,402.9 to 0.1%, and its per-packet census
reproduces that report's control to within 1% on every MoE packet.

### 8.1 Per MoE layer, packet spans (µs)

| op | base | ctlz | det | Δ ctlz-base | Δ det-ctlz | Δ det-base |
|---|---:|---:|---:|---:|---:|---:|
| `FlashMlaPrefill` | 3,159.9 | 3,172.5 | 3,092.4 | +12.6 | -80.1 | -67.6 |
| **`MoeGroupDownPf`** | 2,856.4 | 2,852.6 | **3,437.7** | -3.8 | **+585.1** | **+581.3** |
| `MoeGroupGluPf` | 2,413.5 | 2,417.8 | 2,420.0 | +4.4 | +2.1 | +6.5 |
| `XReduceTwoShot` x2 | 2,307.0 | 2,360.1 | 2,209.4 | +53.2 | -150.7 | -97.5 |
| `MlaMergeFold` | 1,596.3 | 1,598.2 | 1,577.6 | +1.9 | -20.6 | -18.7 |
| **`MoeCombinePf`** | 1,503.9 | **1,375.3** | **424.5** | **-128.7** | **-950.8** | **-1,079.4** |
| `Gemm` x3 | 1,389.8 | 1,393.8 | 1,390.8 | +3.9 | -3.0 | +1.0 |
| `MoeRouterTopkPf` | 340.6 | 338.6 | **415.2** | -2.0 | **+76.6** | **+74.6** |
| `MoeAlignPf` (1 WG) | 245.9 | 241.3 | 238.7 | -4.6 | -2.6 | -7.2 |
| everything else | | | | ≤ +2.4 | ≤ +3.2 | ≤ +4.5 |
| **layer span (median)** | **18,007.4** | **17,926.4** | **17,464.0** | **-81.0** | **-462.3** | **-543.4** |
| perfect-pack over 75 MoE layers | | | | **-6.08 ms** | **-34.68 ms** | **-40.75 ms** |

Layer span x 75 predicts **-40.75 ms**; the device wall moved **-40.1 ms**. They agree to 1.6%.

### 8.2 Grading §6.4's pre-registration — one term right, one badly wrong

| quantity | predicted | traced | |
|---|---:|---:|---|
| op 87 `ctlz` | ~1,440 µs | **1,375.3** | 4.5% better than predicted |
| op 87 `det` | ~610-645 µs | **424.5** | **31% better** — the contiguous f64 read is cheaper than either model said |
| op 83 | ~+36 µs | **+74.6** | 2x the prediction (the f64 zero) |
| op 86 | +80 to +250 µs | **+581.3** | **2.3x the top of the band — this is the miss** |
| net | -0.58 to -0.79 ms/layer | **-0.543** | outside the band, on the wrong side |

**The op-86 term is the finding.** The atomic arm's f32 atomics cost +44.8 µs (1.6%) with 2.60x
rate headroom; DET's f64 atomics cost **+581 µs (+20.4%)** with 1.50x headroom (§6.3). Only ~50 µs
of that is the five added VALU per element (5 x 402.65 M ops against ~82 T op/s); **the other
~530 µs is atomic backpressure**, and it is super-linear in how close the kernel sits to the
atomic ceiling — 1.74x slower atomics did not cost 1.74x, they cost 13x. That non-linearity is
the single thing a future reader should take from this file about fire-and-forget atomics.

**`MoeCombinePf` is the win, and it over-delivered:** -1,079.4 µs/layer, of which -128.7 is the
bit-identical `zero_h` deletion and -950.8 is the deterministic fusion. At 424.5 µs for 0.604 GB
the fused f64 combine runs at **1.42 TB/s**, better than the shipped strided arm's 1.26 and better
than the f32-fused arm's 0.94 — so §1.1's "structure bonus" is real after all once the packet is
big enough to amortise its floor.

**Not attributable, and flagged rather than explained:** `FlashMlaPrefill` -2.1% and
`XReduceTwoShot` -4.2% under DET. Neither packet is touched by this arm. `XReduceTwoShot` is a
collective and is known in this campaign to move with arrival skew, and the MoE chain that feeds
it got 1.1 ms shorter; `FlashMlaPrefill` has no such excuse. Both are inside the ±20% DVFS
ordering noise this campaign has recorded for back-to-back walls, but they are NOT inside the
≤0.9% band the atomic arm's negative control achieved, and that is worth saying plainly. The
served A/B in §9 is the number that does not depend on this.

## 9. MEASURED — served, interleaved A/B

Harness `scripts/bench_speed.sh`, `IN_LENS="4096 8192 16384"`, conc 1, 8 prompts/cell,
`OUTLEN=32`, serve env `PLOW_MLA_PF_V2=1`, one server at a time on port 8196, three arms rotating
within each round (`base,ctlz,det` / `det,ctlz,base` / `ctlz,base,det`) so a monotone warm-up
drift cannot bias one arm. Every arm passed the harness's built-in `Paris` coherence gate.

### 9.1 THE BOX WAS NOT QUIET, and this is how it was handled

This battery ran against 4-6 other concurrent workloads that do not all take the GPU lock. Three distinct
contamination modes appeared and each is dealt with explicitly rather than averaged in:

1. **A sibling's server on the same port.** `bench_speed.sh ... auto` resolves the model off
   `/v1/models`; twice, a sibling's Gemma-4-12B server held :8196 and the client benchmarked
   THEIR engine — TTFT 20-50 ms, TPOT 0.00, `model: gemma-4-12b-it`. The runner now asserts
   `glm-5.2-fp8` and **discards** the cell. (This is the failure class the perf-data README
   records twice; the model-id assertion is the fix it was missing.)
2. **Concurrent GPU load.** Every run is preceded by an idle gate (no foreign `plowrt`, all 8 GPUs
   at 0%) with a 10-minute cap; each cell records `foreign_plowrt=N` at its head.
3. **A single slow request inside a cell.** With 8 requests per cell, one 8-second outlier moves
   the MEAN by a second while the MEDIAN does not. **The TTFT MEDIAN is therefore the primary
   statistic here**, and TPOT — which this prefill-only arm cannot touch — is used as the
   independent contamination detector: any cell whose TPOT exceeds 31 ms (against a clean
   29.0-29.9) is discarded whole.

That discipline discards 5 of 34 cells. Every discarded cell and its reason is in
`/tmp/detw/batteryS.out`; none of them is a `det` cell that happened to look bad.

### 9.2 TTFT median, ms — every surviving cell, six rounds

| ctx | base | ctlz | det |
|---:|---|---|---|
| 4096 | 962.7 963.4 964.8 970.0 | 958.0 966.1 | 945.4 951.0 951.3 951.6 952.1 952.3 |
| 8192 | 1672.9 1674.1 1677.1 1680.4 1686.5 | 1667.3 1668.6 1673.2 | 1639.1 1642.8 1643.3 1643.9 1644.2 1644.3 1646.7 |
| 16384 | 3637.4 3638.4 3641.2 3649.5 | 3626.8 3629.6 3637.3 | 3576.8 3578.1 3579.3 3582.8 3586.4 3589.8 3593.7 |

41 of 52 cells survive the filter. `base` reproduces the campaign's canonical
**973 / 1677 / 3627** at 4k/8k/16k to within 0.9%.

### 9.3 The result, against the control's own spread

| ctx | base | ctlz | det | `ctlz`-base (bit-identical) | **`det`-base** | **control spread** | Δ / spread | every `det` cell below every `base` cell? |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 4096 | 965.2 | 962.0 | **950.6** | -3.2 (-0.33%) | **-14.6 (-1.51%)** | 7.3 ms (0.76%) | **2.0x** | **YES** |
| 8192 | 1678.2 | 1669.7 | **1643.5** | -8.5 (-0.51%) | **-34.7 (-2.07%)** | 13.6 ms (0.81%) | **2.6x** | **YES** |
| 16384 | 3641.6 | 3631.2 | **3583.8** | -10.4 (-0.29%) | **-57.8 (-1.59%)** | 12.1 ms (0.33%) | **4.8x** | **YES** |

**At every context, every `det` cell is below every `base` cell and below every `ctlz` cell; the
three distributions do not overlap.** The `det` arm's own spread (0.46-0.73%) is at or below the
control's (0.33-0.81%) despite having twice as many samples — which is the second, independent
signature of the determinism, and the opposite of what a nondeterministic accumulator gives.

**TPOT is unchanged** — 29.01-29.87 on `base`, 29.01-29.84 on `ctlz`, 29.01-29.83 on `det` across
all surviving cells. That is the negative control: the arm touches only prefill packets, so a TPOT
move would mean it had leaked somewhere it should not have.

### 9.4 Served vs traced

| | traced device prefill wall @8k | served TTFT @8k |
|---|---:|---:|
| `ctlz` - base | -4.7 ms | -8.5 ms |
| **`det` - base** | **-40.1 ms** | **-34.7 ms** |

The two instruments agree on the headline to **16%**. They are not the same quantity — served TTFT
carries tokenisation, scheduling and the chat template's extra prefill chunk on top of the device
wall — so the direction, the magnitude and the ordering agreeing is the check, not the last digit.

### 9.5 The fraction of the atomic arm's win that is recovered

`PLOW_MOE_PF_ATOMIC` measured **-33.9 / -66.8 / -128.8 ms** of served TTFT at 4k/8k/16k against
this same control (`glm52-moe-fusion.md` §8), and this session re-measured its device prefill wall
head to head at **-83.8 ms** (§10.5).

| basis | atomic | **DET** | **recovered** |
|---|---:|---:|---:|
| served TTFT @4k | -33.9 ms | **-14.6 ms** | **43%** |
| **served TTFT @8k** | **-66.8 ms** | **-34.7 ms** | **52%** |
| served TTFT @16k | -128.8 ms | **-57.8 ms** | **45%** |
| traced device wall, paired this session | -83.8 ms | **-40.1 ms** | **48%** |

**The deterministic writer recovers just under half of the nondeterministic one**, and §8.2 says
exactly where the other half went: **the f64 atomic costs op 86 +581 µs/layer where the f32 atomic
cost +45** — 1.74x slower atomics turning into a 13x larger packet cost, because the kernel sits
at 1.50x rate headroom instead of 2.60x. Of the recovered amount, **-8.5 ms at 8k is the
bit-identical `zero_h` deletion** and the rest is the deterministic fusion.

## 10. THE DETERMINISM GATE

This is the point of the branch, so it is measured at the level where a claim cannot hide: the
DEVICE LOGIT TENSOR, byte for byte, not the sampled token and not the text.

### 10.1 Same arm, twice — the test `PLOW_MOE_PF_ATOMIC` fails

`amd-bench --tp 8 --steps 4 --prompt <8,192 ids> --dump-logits <dir>`, run TWICE per arm in two
separate processes with a full model reload between them. Each run dumps five `act.logits`
tensors (bf16, 19,360 entries): `logits_prefill` (the vector that chose the first token) plus one
per decode step.

| arm | run 1 vs run 2, all five tensors |
|---|---|
| `base` (shipped control) | **BYTE-IDENTICAL** |
| `ctlz` (`zero_h` deleted) | **BYTE-IDENTICAL** |
| **`det` (`PLOW_MOE_PF_DET`)** | **BYTE-IDENTICAL** |

**`PLOW_MOE_PF_DET` PASSES the reproducibility gate.** `PLOW_MOE_PF_ATOMIC` does not, and §6.2
shows why at the mechanism level: over 12 samples of the same accumulation in 2 processes, every
f32-atomic hash differed and every f64 fixed-point hash was the same value.

### 10.2 `zero_h` -> `TENSOR_NONE` is bit-identical — proven, not argued

| comparison | prefill logits |
|---|---|
| `base` vs `ctlz` | **0 of 19,360 entries differ** (all five tensors byte-identical) |

The kernel's `residual ? bf2f(residual[i]) : 0.0f` argument is now backed by a byte comparison of
the model's own output. It is the same argmax, the same token, the same bits.

### 10.3 vs the shipped control — NOT bit-identical, and the size of it

| comparison | prefill logits | argmax |
|---|---|---|
| `base` vs `det` | 19,345 of 19,360 differ (99.92%), max abs 1.75 on a max logit of 10.0 | **373 = 373** |
| `base` vs `det`, decode step 0 | 97.5% differ, max abs 0.69 | **28 = 28** |
| `base` vs `det`, decode step 3 | 99.6% differ, max abs 18.7 | **197 vs 40 — FLIPPED** |

In bf16 ULPs on the prefill logits: median 23, p90 42, p99 112. The top-2 tokens agree
(`373, 365`); the tail reorders. By decode step 3 the greedy path has separated.

**This is what §2 predicted and it is not a defect of the implementation.** The arm re-associates a
k-way sum inside a 78-layer network; the campaign's own experience (`PLOW_MOE_PF_PART16` flipping
top-1, the atomic arm changing wording on 4 of 6 prompts) is that this class of perturbation is
visible in the text. §10.4 is the honest gate.

### 10.4 The served character gate — 6 prompts, each asked TWICE per server

One server per arm on port 8195 (with an assertion that the model answering is `glm-5.2-fp8`, after
another run's Gemma server on 8196 corrupted two cells of §9). Six prompts at `temperature=0`,
then the IDENTICAL set asked a SECOND time against the SAME server — which is what separates "the
arm changed the answer" from "the arm cannot repeat itself".

| # | prompt | base==rep | ctlz==rep | **det==rep** | base==ctlz | base==det |
|---|---|:--:|:--:|:--:|:--:|:--:|
| 0 | capital of France | YES | YES | **YES** | **YES** | **YES** |
| 1 | name three primes | YES | YES | **YES** | **YES** | **YES** |
| 2 | GPU matmul, 2 sentences | YES | YES | **YES** | **YES** | no |
| 3 | reverse a singly linked list | YES | YES | **YES** | **YES** | no |
| 4 | 17 * 23, show your work | YES | YES | **YES** | **YES** | no |
| 5 | MoE routing essay, ~3,000 chars | YES | YES | **YES** | **YES** | no |

Read the two middle columns first, because they are what this branch is for:

* **`det == det(rep)`: 6 of 6.** `PLOW_MOE_PF_ATOMIC` was **3 of 6** on this exact battery. The
  arm can now repeat itself, including on a ~3,000-character free-form generation.
* **`base == ctlz`: 6 of 6, and the logits are byte-identical** — the `zero_h` deletion is a free
  win with nothing to argue about.
* **`base == det`: 2 of 6**, the same score the atomic arm got, on the same two short factual
  prompts, for the same reason.

**The four divergences are WORDING, not correctness — checked, not assumed:**

| # | base | det |
|---|---|---|
| 4 | `17 * 23 = 391` + "Here are two ways to show the work" + long multiplication | `17 * 23 = 391` + "Here are two **different** ways to show the work" + long multiplication |
| 2 | "...partitioning the large operation into thousands of smaller, independent dot-product calculations" | "...partitioning the large **mathematical** operation into thousands of smaller, independent dot-product calculations" |
| 3 | correct `ListNode` + iterative O(n)/O(1) reversal + test case | correct `ListNode` + iterative O(n)/O(1) reversal, different preamble |
| 5 | correct 3,058-char MoE essay | correct 2,979-char MoE essay, 461 words |

Prompt 4's first divergence is at **character 28**, inside `"Here are two ways"` vs
`"Here are two different ways"` — i.e. in the scaffolding, AFTER the answer. Both arms answer
**391** with correct working; both produce a correct linked-list reversal; both produce a correct
MoE essay over the requested 400 words.

### 10.5 The atomic arm, measured HEAD-TO-HEAD on the same instrument — the decisive table

`PLOW_MOE_PF_ATOMIC` was rebuilt from this branch and run through the identical
`--dump-logits` protocol, twice, so "is DET's perturbation a worse numerics class than the atomic
arm's?" is answered by measurement instead of by argument. bf16 ULP distances on the prefill
logit vector (19,360 entries):

| comparison | entries differing | median | p99 | max abs | argmax |
|---|---:|---:|---:|---:|:--:|
| base vs **ATOM** (f32 atomic) | 19,357 | **20 ULP** | 72 | 1.281 (12.8% of max logit) | 373 = 373 |
| base vs **DET** (f64 fixed-point) | 19,345 | **23 ULP** | 112 | 1.750 (17.5%) | 373 = 373 |
| **ATOM run 1 vs ATOM run 2** | 18,129 | **4 ULP** | 28 | 0.672 (6.3%) | 373 = 373 |
| **DET run 1 vs DET run 2** | **0** | **0** | **0** | **0** | 373 = 373 |

Three things fall out of that table, and they are the whole argument of this branch:

1. **DET's distance from the control is the SAME CLASS as the atomic arm's** — median 23 ULP
   against 20, 15% larger, not a different order. The f64 fixed-point quantisation
   (2^-32 absolute) adds essentially nothing on top of the re-association that both arms share.
   So DET is not paying a numerics penalty for its determinism.
2. **The atomic arm differs from ITSELF by a median 4 ULP and a max of 6.3% of the top logit** —
   about a fifth of its own distance from the control. That is the defect, quantified: a fifth of
   the arm's numerical footprint is not a property of the arm at all, it is a property of the
   scheduler.
3. **DET differs from itself by exactly zero**, on every one of the five dumped tensors, across
   two processes.

The same battery also re-prices the atomic arm on this session's own control: device prefill wall
**1,401.5 -> 1,317.7 ms = -83.8 ms** (it carries this branch's `zero_h` deletion too), against
DET's -40.1 ms. On a strictly paired basis measured minutes apart, **DET recovers 47.9% of the
atomic arm's device-wall win** — the same picture §9.5 gives from the served side at 55-58%.

### 10.6 Verdict on the gate

* **Run-to-run reproducibility: PASS.** Byte-identical logits across processes, character-identical
  answers across two passes, 6 of 6 — which is the property `PLOW_MOE_PF_ATOMIC` lacks and the
  reason it could not ship.
* **Bit-identity vs the shipped control: FAIL, and §2 shows it is unattainable** for any scheme
  that moves the k-way reduction upstream of op 87.
* **Character-identity vs the shipped control: 2 of 6** — the campaign's landing bar, and this arm
  does not clear it either. `PLOW_MOE_PF_DET` therefore **must stay opt-in**, exactly like the
  atomic arm, and this branch does not default it on.

What has changed is which objection survives. Against the atomic arm there were two, and the
serious one was that it was not reproducible. That one is gone. What remains is a single, precisely
characterised statement: **this arm computes a different, deterministic, slightly more accurate
association of the same sum, and on a 78-layer model that is visible in free-form text.**

## 11. WHAT REMAINS

1. **The block-interleaved `part[t][h/G][j][g]` layout (§3.4)** — the ONLY bit-identical lever left
   on op 87, bounded at about **-13 ms** by op 87's already-1.26 TB/s strided read, and requiring
   `G = 8` (never the naive transpose, which is +8.1 ms/layer of write amplification). Worth doing
   only if a bit-identical win is wanted for its own sake.
2. **The expert-ORDER ticket (§3.3)** — keeps the accumulator at 4 bytes, so it would recover the
   0.604 GB/layer this arm gives back, but lands in the SAME numerics class (expert order is not
   slot order) for ~250 lines, two barriers per output tile, and a deadlock-freedom proof that the
   expert-parallelism tile skip (`if (wb0 == 0ull) continue;`, op_moe.h:2197) violates outright.
   If it is ever built, that skip must bump the ticket or the arm must refuse EP blobs.
3. **`global_atomic_pk_add_bf16`** (aiter's actual instruction) remains the faster/worse point on
   the curve: it deletes op 87 outright and rounds the k-way sum to bf16 after every add, a
   strictly worse error class than `PLOW_MOE_PF_PART16`, which already flipped top-1.
4. **The `zero_h` deletion generalises.** A `[T,H]` buffer of literal zeros existed because the
   emit had no way to say "zero" other than to name a tensor, while the kernel had always spelled
   it as a null pointer. It is worth one pass over the other emitters for the same shape of
   mistake — an operand that is a materialised constant the kernel already special-cases.
5. **The EP correctness note carries over.** Under expert parallelism the grouped GEMM skips
   non-local experts, so those slots' `part` rows are never written and the shipped combine reads
   whatever the previous layer left there. Under DET (as under the atomic arm) an unwritten slot
   contributes exactly `0.0`, because the accumulator was zeroed — the fused side is the correct
   one. GLM-5.2 at TP8 is tensor-parallel and every expert is local, so this does not affect
   anything measured here.

---

## 12. HOW TO REPRODUCE

```
# objects (outside nix; hipcc wants system glibc)
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 PLOW_MOE_PF_EPI=1 \
    PLOW_MOE_PF_DET=1 bash scripts/build_gfx942.sh <objdir>

# blob (the campaign's canonical GLM-5.2 TP8 recipe + the arm)
env GLM_FULL=1 PLOW_MLA_PREFILL=full GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 \
    PLOW_GLM_DSA=0 PLOW_GLM_FUSE_B1=1 PLOW_GLM_GEMV_WG=152 PLOW_MLA_PF_V2=1 \
    PLOW_GLM_PF_NS=2 PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1 PLOW_MOE_PF_DET=1 \
    ./target/release/plowc --emit devblob --hf-dir /workspace/models/GLM-5.2-FP8 \
    --gpu MI300X --arch gfx942 --num-gpus 8 --max-ctx 73728 --out <blobdir>

# serve env is still PLOW_MLA_PF_V2=1.  plowrt must be built --features hsa.

# the mechanism probe (no model needed, ~10 s)
hipcc --offload-arch=gfx942 -O3 -o det_probe perf-data/plow-gfx942/probes/det_accum_probe.hip
./det_probe

# the determinism gate
plowrt amd-bench --blob ... --tp 8 --steps 4 --prompt "<8192 ids>" --dump-logits d1
plowrt amd-bench --blob ... --tp 8 --steps 4 --prompt "<8192 ids>" --dump-logits d2
diff -r d1 d2      # must be empty
```

**Three traps this battery hit, all of them recorded in the perf-data README's discipline section
and all of them still able to bite:**

1. `bench_speed.sh <assets> <port> auto` resolves the model off `/v1/models`. If another run
   holds the port, **you benchmark their engine and it looks like a result** — two cells came back
   as `gemma-4-12b-it` at 20 ms TTFT. Assert the model id.
2. `tail -N` on a served run's output loses the measurement: the server keeps logging on the same
   fd after the table prints, so the tail window fills with INFO lines. Capture to a file and grep.
3. `pgrep -c` prints `0` AND exits 1, so `[ "$(pgrep -cx plowrt || echo 0)" != "0" ]` compares
   `"0\n0"` and spins forever **while holding the GPU lock**. Use `while pgrep -x plowrt; do`.
