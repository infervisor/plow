# Band-pipelined TP collectives: CU-subset scheduling — the overlap is real, and it is structurally capped

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **PLOW-ARCHITECTURAL** — 'overlap on a saturated grid is a static partition of the same CUs' is a property of the persistent-kernel execution model, not of CDNA3, and has replicated across architectures.

2026-08-08. Branch `band-pipeline`, base `worktree-glm52-bringup` @ **a5b0423**.
8× MI300X gfx942, GLM-5.2-FP8 TP8, objects `hsaco_glm17`, blob = the canonical
`glm52-tp8-final` recipe. **No `runtime/` file is touched** — this is an emit-side
branch, so the ISA is bit-identical by construction.

Closes the follow-up left open by `glm52-experiments.md (consolidated: band overlap, null)`. That report landed the
banding machinery (`PLOW_GLM_XR_BAND=K`, logits byte-identical) but measured it NET
NEGATIVE (+3–8% TTFT), with the fix recorded as: *"the overlap requires band XRs on a
CU SUBSET so the remaining CUs claim the next band's producer."* This is that build.

## 0. Verdict

**The diagnosis was right, the fix works, and the lever is still negative.**

| arm | TTFT vs control, interleaved, control's own spread 0.2–0.5% |
|---|---|
| banding, no subset (`b4` — what shipped default-off) | **+7.9% @4k, +4.6% @8k** |
| banding + CU subset (`PLOW_GLM_XR_BAND_CUS=112`) | **+1.2% @4k, +0.7% @8k, +1.0% @16k** |
| banding + smaller subset (`…_CUS=76`) | +1.7 / +1.2 / +1.5% — monotone the wrong way |
| collective width alone (`PLOW_XR_CUS=152`, no banding) | −0.9% @4k, +0.2% @8k, +0.5% @16k — **a null** |

**The CU subset removed 85% of the banding penalty** (+7.9 → +1.2 @4k, +4.6 → +0.7 @8k)
and still did not cross zero. The
overlap it enables is REAL — the trace shows band *i+1*'s producer running 500+ µs
inside band *i*'s collective — but it is paid for out of the producer, one CU at a
time, and the two sides cancel. **Recommend: stop paying for this lever.** Everything
below is why, with the arithmetic to check it against.

## 1. The arithmetic ceiling — stated before optimising

Control trace @8k (`PLOW_TRACE_RAW` + `trace_block.py`, 73 MoE layers, ctx bucket
T=8192, n = T·hidden = 50,331,648 elements, **median layer span 18,504 µs**):

| packet | span/layer | busy CU-µs/layer | nCU |
|---|---|---|---|
| o_proj `Gemm` (attn-seam producer) | 897 µs | 203,167 | 304 |
| `MoeCombinePf` (MoE-seam producer) | 1400 µs | 416,341 | 304 |
| `XReduceTwoShot` × 2 | **2259 µs** | 634,275 | 304 |

**XR is 12.2% of the layer**; TTFT @8k is 1713 ms, so the collectives are ~209 ms of
it. The earlier "XR sync = 7–9% of TTFT" figure is the `PLOW_XR_NOWAIT`-priced *wait*
subset of this; 12.2% is the whole packet, wait plus transfer, and it is the honest
upper bound.

**Deleting the collectives entirely — no fabric, no barrier, wrong answers — cannot buy
more than 12.2% at 8k.** The overlap lever's own ceiling is strictly smaller and can be
written down exactly, because the producers are 897 and 1400 µs against collectives of
1063 and 1167 µs:

* Overlap can hide at most `min(P, X)` per seam = 897 + 1167 = 2064 µs of the 2259 µs.
  **On the attention seam the collective is LARGER than the producer it must hide
  behind** (1063 > 897), so even a free, contention-free overlap leaves 166 µs exposed.
* That 2064 µs is a *gross* ceiling that assumes the CUs donated to the collective are
  free. They are not — §4 measures the price.
* **At 1k–2k context the ceiling is below the noise floor.** TTFT @1k is 350 ms, the
  layer is ~4× smaller, but the collective's rendezvous cost is fixed, and the whole XR
  term lands under the ±0.2–0.5% round-to-round spread of the serve harness. The
  measurement cannot resolve it there, which is why the A/B runs 4k/8k/16k only.

## 2. Why a full-grid rendezvous per band is the cost — from the code

Three facts compose into the +3–8%.

**(a) A packet's workgroup count IS its emitted `cus` list length.** `Builder::emit(op,
cus, …)` writes `blocks: cus.len()` (`crates/packet/src/devbuild.rs:504`); the
interpreter passes `in->blocks` straight in as `nblk`
(`runtime/amd/interp.hip:3303`). There is no other width knob.

**(b) The global queue is one monotonic cursor with `PLOW_GQ_BATCH=1`.** Workgroups
claim stream entries with a single `fetch_add` in op-major order
(`runtime/amd/interp.hip:3073-3080`). A packet emitted on all 304 CUs contributes 304
consecutive entries, so **all 304 workgroups are consumed by it** — nobody is left to
walk past and start the next band's producer. GLM's prefill program is not L2-placed
(`emit_glm` places decode only; prefill objects are built without
`-DPLOW_L2_PLACE_DISPATCH`), so this is literally one queue and one cursor.

**(c) `gate_ag` is a cross-GPU barrier whose threshold scales with the width.** Every
workgroup signals every peer and waits `nranks*nblk` arrivals (`op_collective.h`,
PHASE-2 note — the asymmetry with `gate_rs` is deliberate and fixed a live race). At
`nblk=304, nranks=8` that is **2,432 remote system-scope RMWs per rank per collective**.

K bands on the full grid = K× (b) × (c): K full-grid cross-GPU barriers, strictly
serialised by the cursor, with no overlap available. That is the whole +3–8%.

### What one rendezvous costs

`perf-data/plow-gfx942/probes/xrwg.hip` §(B) runs the exact protocol (relaxed poll, one
system acquire, release RMW, 512 threads/WG) with **no payload**, 8 devices concurrent,
200 reps:

| nblk | 304 | 152 | 76 | 64 | 38 | 32 | 19 | 16 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| µs / rendezvous PAIR | **82.7** | 39.0 | 22.6 | 20.0 | 18.0 | 17.6 | 16.6 | 16.1 | 14.1 |
| remote RMWs/rank/pair | 2440 | 1224 | 616 | 520 | 312 | 264 | 160 | 136 | 72 |

Fits `13.6 + 0.0227·nblk` µs to within 2%: a ~14 µs fabric-latency floor plus **22.7 ns
per participating workgroup** — the announcement traffic, exactly as the
`PLOW_XR_NOSIG` note predicted ("the announcement is the dominant term by count").

**Cost model, and it is the whole §0 story in one line.** K bands at width c cost
`K·(13.6 + 0.0227c)` µs of rendezvous where the unbanded seam costs 82.5. At the
shipped c=304, K=4 spends 331 µs per seam — ×156 seams = **52 ms per 8k prefill chunk
of pure added barrier**, ~3% of TTFT, before anything else happens. At c=112 the same
K=4 spends 65 µs, i.e. *less* than the unbanded barrier. **The subset does not merely
enable the overlap; it is what makes K barriers affordable at all**, and it is why the
lever moved from +3–8% to +0.7–1.5%.

## 3. The width question, and reconciling it with `coll-tune`

Building §2(a) meant finding where the width comes from, and the GLM **prefill** emitter
hardcoded it — `let pxr: Vec<u32> = pall.clone();`. `PLOW_XR_CUS`, the knob whose entire
documented job is "Cap XReduce participant CUs", was silently **decode-only** on the GLM
path, and decode saturates at `ceil(hidden/512) = 12` workgroups anyway, so the knob had
never done anything for GLM at all. (The sibling `coll-tune` agent found the same line
independently and left it; it is fixed here.)

That made the width worth pricing. `perf-data/plow-gfx942/probes/xrts.hip` calls the
shipped `d_xreduce_twoshot_mega` itself — both rendezvous, RS reduce, staggered AG,
512 threads/WG — 8 devices concurrent, at the @8k shape:

| nblk | 304 | 256 | 224 | 192 | 176 | **152** | 128 | 112 | 96 | 76 | 64 | 38 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| µs / collective | 1279 | 974 | 918 | 873 | 873 | **863** | 894 | 922 | 992 | 1133 | 1263 | 2028 |

Reproducible to 1.5%. **In isolation the shipped width is 48% past a broad optimum at
152–192.** And §(C) of `xrwg.hip` says why — the controlling variable is threads in
flight, not workgroups:

| threads | configs | GB/s |
|---|---|---|
| 155,648 | 304×512, 608×256 | 177, 181 |
| **77,824** | 304×256, 152×512, 608×128 | **254, 258, 263** |
| 38,912 | 304×128, 152×256, 76×512 | 174.5, 174.5, 174.6 |
| 19,456 | 304×64 | 89.5 |

The collapse is exact — three groupings at 77,824 threads all give 254–263 GB/s, three
at 38,912 all give 174.5–174.6. The uncached peer path peaks near **78k threads
(1,216 waves)** and falls off on BOTH sides.

**Reconciliation with `glm52-collective-tuning-mi300x.md`.** That report measures the
AG rate "dead linear from 19 to 304 workgroups … never saturates", and concludes
`PLOW_XR_CUS` narrowing is not free. Its sweep is at **304 WG × 256 thr** (stated in its
own ceilings table). 304×256 = 77,824 threads = *exactly the peak*. Its top data point
sits on the maximum, so within its range the law is linear and correct — it simply never
enters the falling regime. The shipped prefill object is built with `PLOW_WG_WAVES=8`
(`scripts/build_gfx942.sh:55`, `CDNA3_TILE` → `AX_PREFILL`), i.e. **512 threads/WG**, so
at `nblk=304` the real kernel runs 155,648 threads and is on the far side. Both
measurements are right; they are about different points on the same curve. Its
consequence #1 ("narrowing costs fabric time roughly 1:1") holds below ~78k threads and
is inverted between 152 and 304 workgroups at 512 threads.

**But the isolated probe overstates it, and the in-situ trace is the arbiter.** Same
layer, same objects, `nblk` 304 → 152:

| | attn XR span | MoE XR span | busy CU-µs/layer |
|---|---|---|---|
| `ctl` (nblk=304) | 1063 µs | 1167 µs | 634,275 |
| `xr152` (nblk=152) | 1115 µs | 1120 µs | **362,762 (−43%)** |

**In situ the collective's wall time is flat between 152 and 304 CUs** (2230 vs 2235 µs
per layer) while its CU-time halves. The −33% the isolated probe promised does not
appear inside the megakernel. So the honest in-situ law is neither "linear" nor
"peaked": between 152 and 304 workgroups the prefill two-shot is **already saturated**,
and the extra 152 CUs buy nothing and cost nothing. That is why `xr152` benches as a
null (§5) — it frees half the machine, and the freed half has nothing to do, because the
only packet after the collective is the `Residual` that depends on it.

**Which is exactly the opening banding was supposed to exploit.**

*(Correction of record: the first commit on this branch, `ce39322`, headlines the
isolated-probe −33% as if it were the win. It is not — the in-situ table above and the
served A/B in §5 supersede it. The probe number is still the right description of the
fabric; it simply is not what the megakernel's critical path sees.)*

## 4. CU-subset scheduling: the overlap is real, and it is paid for out of the producer

`PLOW_GLM_XR_BAND_CUS=c` emits each band's two-shot on a `c`-prefix of `xr_cus`
(`crates/devgen/src/mla.rs`, `xr_band_cus`), leaving `304 − c` workgroups free to walk
past it on the monotonic cursor and claim the next band's producer. The producer bands
keep `all`, so whichever workgroups are free drain their 304 slices — it self-balances,
and no new dependency machinery is involved. (Note for the record: this does **not** use
`Dep::Fine`. The MoE design review is right that fine edges are dead code; the original
band commit already argued the round-robin tile claim collapses that map, which is why
banding splits the PACKET. What this adds is coarse gates scoped to CU subsets.)

### Which `c`, and why — the sweep and the arithmetic

`c` was not picked; it was swept twice.

1. **On the collective alone**, 12 widths with the shipped kernel (§3 table, 304 → 38).
2. **Balance argument**, then two served arms. Write `C = 304`. The pipeline steady
   state is `max(producer chain, collective chain)`; the producer chain runs on `C − c`
   and the collective chain is `K` bands at width `c`. With the control trace's
   `P_attn = 897 µs`, the two chains balance near `c ≈ 100–130` — below that the
   collective chain dominates (at `c = 38` one band alone costs 507 µs and four bands
   exceed the whole seam), above it the producer donation dominates (at `c = 152` the
   producer runs at half speed). So **`c = 112` is the model's optimum and `c = 76` its
   nearest sanity check on the other side** — and the measured ordering
   `c=76 > c=112 > unbanded` (worse to better, §5) confirms the model's shape while
   refuting its conclusion: the optimum of a function that is everywhere above zero is
   still above zero.

**It works. The packets are demonstrably concurrent.** Layer 40, T=8192, K=4, c=112:

```
 pc   op                start   end   span   nCU
1503  Gemm  (band 0)     6848   7073   224   304
1504  XReduceTwoShot     7075   7598   523   112   <- collective on 112 CUs
1505  Gemm  (band 1)     7055   7373   318   304   <- STARTS INSIDE band 0's collective
1506  XReduceTwoShot     7375   7925   550   112
1507  Gemm  (band 2)     7259   7739   480   304   <- overlaps again
1508  XReduceTwoShot     7740   8240   500   112
1509  Gemm  (band 3)     7710   8095   385   304   <- and again
1510  XReduceTwoShot     8097   8520   423   112
1511  Residual           8521   8645   124   304
```

The `gap` column in `trace_block.py` is negative (−543, −665, −530 µs) — that IS the
overlap. This is precisely the premise: TP data movement starting as soon as a band of
the producing tensor is ready, running while the rest is still being computed.

**And it is still a wash, because the overlap is paid for out of the producer.**

| seam @ layer 40 | control (unbanded, 304) | banded K=4, c=112 | Δ |
|---|---|---|---|
| attn: o_proj + XR + Residual | 6859 → 8690 = **1831 µs** | 6848 → 8645 = **1797 µs** | −34 µs (−1.9%) |
| MoE: combine + XR + Residual | 15631 → 18325 = **2694 µs** | 15676 → 18883 = **3207 µs** | **+513 µs (+19%)** |
| MoE layer span (median, 73 layers) | **18,504 µs** | **18,848 µs** | **+344 µs (+1.9%)** |

The attention seam is the honest best case and it is a dead heat. The reason is in the
same trace: the o_proj GEMM chain went 643 µs (one packet, 304 CUs) → 224+318+480+385 =
**1407 µs**, 2.2× slower, because it now shares the machine with the collective. The
collective chain went 1063 → 4×~500 = **1996 µs**, 1.9× slower, for the mirror reason.
Each side gave up almost exactly what the other gained.

That is the break-even identity, and it is structural, not a tuning miss:

> In a persistent megakernel with a **saturated grid**, "overlap" is not free
> concurrency — it is a static partition of the same CUs. Donating `c` of `C` CUs to the
> collective slows the producer by `c/(C−c)` and speeds the collective by at most what
> the fabric will still take. It wins only if the two partitions are limited by
> DIFFERENT resources. Here they are not: §3 shows the collective is already saturated
> at 152 CUs, so the CUs it gives back were never its bottleneck, while the producer
> loses them 1:1.

The MoE seam is worse than a wash (+19%) for the sharper version of the same reason:
`MoeCombinePf` streams ~1.6 GB of `part[]` per layer from HBM, and the two-shot's RS
phase and `out` write are HBM traffic too. Overlapping them puts two bandwidth-bound
packets on the same memory system, and both slow down. **The subset choice is monotone
in the wrong direction** — `c=76` is worse than `c=112` at every context (§5), exactly
as the model predicts, because a smaller subset slows the collective more than it
speeds the producer.

## 5. A/B: interleaved, 3 rounds, TTFT median (ms)

`scripts/bench_speed.sh`, one server per arm per round, arms interleaved within each
round, `IN_LENS="4096 8192 16384" CONCS=1 NPROMPT=2 OUTLEN=32`, port 8196, all coherence
gates PASS. `(±x%)` is that arm's own round-to-round spread — **the noise floor of this
measurement, and it is small**: the control moves 0.2% @4k, 0.2% @8k, 0.5% @16k, far
below the ±20% DVFS spread that back-to-back `amd-bench` walls show on this box.

| ctx | `ctl` | `xr152` (width only) | `x152b4c112` (band+subset) | `x152b4c76` (band+subset) |
|---|---|---|---|---|
| 4096 | **1000** (±0.2%) | 991 (±0.8%) **−0.9%** | 1016 (±0.6%) **+1.5%** | 1018 (±0.6%) **+1.7%** |
| 8192 | **1714** (±0.2%) | 1718 (±0.4%) +0.2% | 1726 (±0.2%) **+0.7%** | 1735 (±0.3%) **+1.2%** |
| 16384 | **3709** (±0.5%) | 3728 (±0.5%) +0.5% | 3745 (±0.2%) **+1.0%** | 3766 (±0.4%) **+1.5%** |

Per-round detail (ms):

```
ctx=4096   ctl 1000 1000 1002 | xr152  990  998  991 | c112 1016 1016 1011 | c76 1018 1018 1012
ctx=8192   ctl 1712 1714 1714 | xr152 1717 1724 1718 | c112 1726 1727 1723 | c76 1737 1735 1731
ctx=16384  ctl 3698 3709 3715 | xr152 3717 3728 3734 | c112 3746 3745 3739 | c76 3772 3766 3757
```

Readings:

* **Band + CU subset is negative at every context, on both subset widths, in every
  round** — +0.7…+1.7%, cleanly outside the 0.2–0.5% control spread. Not noise.
* **The ordering `c=76` worse than `c=112` worse than unbanded is monotone and holds at
  all three contexts** — the CU-donation prediction of §4, confirmed three ways.
* **`xr152` is a null.** −0.9% @4k is 4.5× the control spread and is probably real but
  worth ~9 ms; @8k and @16k it is +0.2/+0.5%, i.e. at or just past the spread in the
  WRONG direction. Consistent with §3's in-situ finding that the collective is already
  saturated at 152 CUs and the freed CUs have nothing to do. **The isolated probe's
  −33% does not survive contact with the megakernel** — recorded so nobody re-derives
  the width lever from the microbenchmark alone.
* Nothing improves with context. The lever was expected to matter most at long context
  where the collectives are largest; the measured penalty is instead flat-to-shrinking,
  because the seams grow together with everything else.

## 6. Serve gate — CHARACTER-IDENTICAL, as the byte-identical logits require

`plowrt serve` on port 8195, `temperature=0`, one server per arm, three questions.
Because the emit is logits-byte-identical, the bar here is not "sensible" — it is
**exact string equality with the control**. Anything else would mean the banding broke.

```
=== GATE arm ctl ===          model: glm-5.2-fp8
--- Q1 capital of France ---
The capital of France is Paris.
--- Q2 17x23 ---
391
--- Q3 Au ---
The chemical element with the symbol **Au** is **gold**.

One common use of gold in jewelry is for making **wedding bands and engagement rings**.
Because pure gold is naturally soft, it is usually alloyed (mixed) with other metals like
copper, silver, or palladium to increase its durability and strength for everyday wear.
```

| arm | result |
|---|---|
| `ctl` | reference, 420 chars over the three answers |
| `xr152` | **CHARACTER-IDENTICAL** |
| `x152b4c112` | **CHARACTER-IDENTICAL** |
| `x152b4c76` | **CHARACTER-IDENTICAL** |

All `bench_speed.sh` in-line coherence gates PASS on every arm in every round as well
(12 + 6 server starts).

### 6b. Does the CU subset actually fix what the previous report blamed? Yes — 85% of it

Same session, same objects, interleaved, 2 rounds, to put the no-subset arm and the
subset arm on one ruler:

| ctx | `ctl` | `b4` — banding, NO subset | `x152b4c112` — banding + CU subset |
|---|---|---|---|
| 4096 | **1001** (±0.2%) | 1079 (±0.7%) **+7.9%** | 1013 (±0.7%) **+1.2%** |
| 8192 | **1713** (±0.4%) | 1791 (±0.3%) **+4.6%** | 1725 (±0.4%) **+0.7%** |

`b4` reproduces the previously recorded +3–8% on today's tree, and the CU subset removes
**85% of it** (+7.9 → +1.2 @4k, +4.6 → +0.7 @8k). The prior report's diagnosis — "all GQ
claimants pile into each rendezvous" — was correct, and the prescribed fix does what it
was supposed to do. It just lands ~1% short of the control instead of ~6% short.

## 7. Emit recipe

Objects — **unchanged**, `hsaco_glm17` serves every arm (no `runtime/` file is touched):

```
env PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_MLA_PF_SV=1 bash scripts/build_gfx942.sh <objdir>
```

Blob — the canonical `glm52-tp8-final` recipe, plus the arm's knob:

```
env GLM_FULL=1 PLOW_MLA_PREFILL=full GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 \
    GLM_SHARD_HEAD=1 PLOW_GLM_DSA=0 PLOW_GLM_FUSE_B1=1 PLOW_GLM_GEMV_WG=152 \
    PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=2 PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1 \
    <ARM KNOBS> \
  plowc --emit devblob --hf-dir /workspace/models/GLM-5.2-FP8 --gpu MI300X \
    --arch gfx942 --num-gpus 8 --max-ctx 73728 --out <assets>/model.pkt
```

| arm | knobs | XR packets @ T=8192 | blob md5 |
|---|---|---|---|
| `ctl` | (none) | 156 × `b=304`, n=50,331,648 | `48d7d77c…` = canonical asset |
| `xr152` | `PLOW_XR_CUS=152` | 156 × `b=152` | `1780b1ec…` |
| `b4` | `PLOW_GLM_XR_BAND=4` | 615 × `b=304`, n=12,582,912 | `939afc6c…` |
| `x152b4c112` | `PLOW_XR_CUS=152 PLOW_GLM_XR_BAND=4 PLOW_GLM_XR_BAND_CUS=112` | 615 × `b=112` | `44bdc6d5…` |
| `x152b4c76` | `… PLOW_GLM_XR_BAND_CUS=76` | 615 × `b=76` | `a8c96671…` |

Serve env still needs `PLOW_MLA_PF_V2=1`. Banding engages only where
`t % K == 0 && t/K >= 512`, so at K=4 the 2048/4096/8192 buckets band and 128/512/1024
do not.

Probes (both new, both in `perf-data/plow-gfx942/probes/`):
```
hipcc --offload-arch=gfx942 -O3 -o xrwg xrwg.hip                       # rate vs WGs, vs threads, rendezvous cost
hipcc --offload-arch=gfx942 -O3 -I<repo>/runtime -o xrts xrts.hip      # the SHIPPED two-shot at varying nblk
```

## 8. Emit / correctness audit

* **Knobs unset ⇒ byte-identical.** `ctl` md5 `48d7d77cfdf827cf75597e6ba598b0a2` ==
  `/workspace/assets/gfx942/glm52-tp8-final/model.pkt`. Re-verified *after* the
  `xr_cus_capped` refactor (arm `ctl2`, same md5) — the refactor is byte-neutral.
* **Prefill logits BYTE-IDENTICAL on every arm.** `amd-bench --dump-logits`, T=4096
  prompt, "all 8 ranks agree": `xr152`, `b4`, `x152b4c76`, `x152b4c112` all `cmp`-equal
  to `ctl`. The strong correctness property the banding shipped with is intact, and the
  width change inherits it for the same structural reason — `d_xreduce_twoshot_mega`
  gives each element to exactly one thread and sums `r = 0..nranks-1` in order, so only
  the element→workgroup partition moves.
* **The `slot_bytes` loader trap is not tripped.** The loader infers the window slot size
  as `max(XR i[2])` (`crates/plowrt/src/asset/devblob.rs`); every arm reports
  `slot_bytes=100663296` at load, identical to control. `disasm` confirms the band
  packets' `slot=` values are exactly the two region bases {0, 100663296} the unbanded
  emit uses — band offsets ride in `i[5]` as required.
* **Decode is untouched by `PLOW_XR_CUS=152`.** `disasm --program 1` on `ctl` vs
  `x152b4c112` differ only in the blob-path line; the decode XReduce stays `b=12`
  (`ceil(hidden/512)`), because `emit_xreduce_gather` already saturated it there.
* **No ISA delta to audit.** `git diff a5b0423..HEAD --name-only` = `crates/devgen/src/mla.rs`
  plus two probe files. Zero `runtime/` files, so there is no kernel object that could
  regress the way 6748e5b did.

## 9. What this closes, and what it leaves

**CLOSED — do not re-commission band-pipelined TP collectives on this architecture.**
The construction is correct (byte-identical logits), the overlap is real (concurrent
packets in the trace), the recorded fix worked (the CU subset removed most of the
penalty), and it still does not cross zero at any context or any subset width. The
reason is structural and will not yield to tuning: a persistent megakernel on a
saturated grid has no spare CUs, so overlap is a static partition, and here both
partitions are limited by the same memory system. The knobs stay in the tree,
default OFF, as the record.

**Ceiling to judge future proposals against:** the collectives are 12.2% of TTFT @8k
(2259 µs of an 18,504 µs layer), 7–9% of it is the `NOWAIT`-priced wait. Anything that
only reschedules the collective is bounded by that, and the overlap sub-lever is bounded
by `min(P, X)` per seam, which on the attention seam is *less than the collective*.
Below ~4k the whole term is under the harness's own noise floor.

**Left open, in order of expected value:**
1. **Move fewer bytes.** The one direction the arithmetic still allows: the two-shot
   moves `2(N−1)/N · [T,hidden]` bf16 per seam. Quantized all-reduce (QuickReduce-style)
   halves that at a numerics cost, and unlike scheduling it attacks the 12.2% directly.
2. **`PLOW_XR_CUS` for K3.** `crates/devgen/src/k3.rs:2626` has the same
   `(0..n_cu).collect()` shape the GLM prefill path had. Not touched here (out of lane),
   but it means the K3 prefill collectives are also pinned at full width. Worth a look
   for footprint reasons even though §5 says the perf effect is a null on GLM.
3. **The fabric peak itself** (~78k threads, `xrwg.hip` §C) is a property nobody had
   measured before. It is a null on the megakernel's critical path today because the
   collective is already saturated at 152 CUs, but it is the right constant to know for
   any future collective that is NOT on the critical path.
