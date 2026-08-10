# DSA sparse, the last lever: TP-sharding the indexer query axis — SCOPED, and it does not reach the goal

> **Scope:** no GPU -- arithmetic over measured inputs · **CDNA3 inputs / AMD-GENERAL conclusion** — already states its gfx950 case explicitly: CDNA4's 2x fp8 MFMA halves a term that sharding has already made small, so the decision does not change.

2026-08-09. Directive: *"plow has advantage for sparse, check gfx950 for dsa sparse, take
plow system design advantage for it, use double buffer for this if needed or copy on write
if index being changed."* This is the scoping `glm52-dsa-indexer-rebuild.md` §5 asked for
before anything is built. Script: `probes/dsa_tp_shard_net.py` (arithmetic only, no GPU).

## Where this picks up

Three closures already stand, and none of them is re-litigated here:

* `glm52-dsa-sparse-prefill.md` — the union-per-tile gather is net-negative.
* `glm52-dsa-sparse-b3.md` — the union floor is a **data property** (adjacent queries share
  89% of their top-2048), not a tunable.
* `glm52-dsa-indexer-rebuild.md` — the indexer was rebuilt exactly (−62% chain) and **even a
  perfect indexer leaves sparse under the 15% bar at 16k**, because selection is
  intrinsically 47% of the arithmetic of the attention it avoids.

That last note named exactly one surviving lever: the indexer is **replicated on all 8 TP
ranks** while the flash is **sharded 8 ways**, and that replication is the whole reason the
per-rank ratio is 47% rather than 6%. Shard the indexer's query axis, all-gather the
selected indices, and the ratio collapses.

## What the directive's two mechanisms turn out to be worth

Both were priced, and the answer is not the one the directive assumed.

| lever | 16k NET | Δ vs the row before it |
|---|---:|---|
| replicated indexer (ledger's closure) | 198 ms | — |
| + TP-shard the query axis | 607 ms gross of gather | **+409 ms — this is the whole prize** |
| − all-gather, u32 positions | 569 ms | −38 |
| − all-gather, **u16 positions** | 588 ms | **+19 vs u32** |
| − all-gather, fully hidden by double-buffering | 607 ms | **+19 vs u16** |

**The double buffer the directive asks for is not the thing that matters; the u16 index
width is.** Positions index a context that fits in 16 bits at every length in this table, so
the gather bytes halve for free, and that single change recovers half of what a full
overlap pipeline would — the overlap is then worth a further 19 ms out of a 3245 ms TTFT
(0.6 pp). Priced at plow's own **measured** fabric rate (240.5 GB/s at 304 workgroups,
`op_collective.h` §PLOW_XR_MLP [E]), not a spec number.

The overlap is also worth less than it looks for a reason the same measurement supplies:
that probe found the all-gather rate is **dead linear in workgroup count** (18.4 / 36.5 /
72.6 / 141.8 / 240.5 GB/s at 19/38/76/152/304 WGs) — the links never saturate. A gather
overlapped with scoring therefore *gets a fraction of the CUs and slows in proportion*, so
the 607 ms "overlap" column is a ceiling that a real pipeline would not reach. Copy-on-write
on the index buffer answers a different question again (a decode-side concern: the selection
changes every step) and does not appear on the prefill critical path at all.

## The full table

Whole prompt, ×78 layers. Positive NET = sparse is cheaper than dense.

|    T | flash dense | ideal sparse | gross | idx ×1 | NET | idx ÷8 | gath u32 | NET | gath u16 | **NET** | NET overlap |
|-----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  4096 |   77 |   63 |    14 |   29 |  −15 |   4 |  10 |    1 |  5 |    **6** |   10 |
|  8192 |  240 |  118 |   122 |  117 |    5 |  15 |  19 |   88 | 10 |   **98** |  107 |
| 16384 |  915 |  249 |   666 |  468 |  198 |  59 |  38 |  569 | 19 |  **588** |  607 |
| 32768 | 3570 |  511 |  3059 | 1873 | 1186 | 234 |  76 | 2749 | 38 | **2787** | 2825 |

As a fraction of TTFT — reported against **both** bases, because this directory holds two
and mixing them is how the campaign has previously fooled itself:

|    T | R3 TTFT | shard+u16 | R2 TTFT (current) | shard+u16 |
|-----:|---:|---:|---:|---:|
|  4096 |  973 |  0.6% |  712 |  0.8% |
|  8192 | 1677 |  5.8% | 1372 |  7.1% |
| 16384 | 3627 | **16.2%** | 3245 | **18.1%** |

**So the ledger's §5 row is confirmed and slightly improved: sharding does clear plow's own
15% bar at 16k, at 16.2–18.1% rather than the 15.6% that row estimated** — the improvement
being entirely the u16 index width, which that row flagged but did not take.

## And it still does not reach the campaign's goal

The goal is not "clear a 15% internal bar", it is *beat vLLM at long context by a margin*.
Against the R2 baseline and vLLM 0.26 on the same box:

|    T | plow | vLLM | gap | best sparse NET | plow+sparse | still behind | gap closed |
|-----:|---:|---:|---:|---:|---:|---:|---:|
|  4096 |  712 |  566 |  146 |  10 |  702 |  136 |  7.1% |
|  8192 | 1372 |  672 |  700 | 107 | 1265 |  593 | 15.3% |
| 16384 | 3245 | 1631 | 1614 | 607 | 2638 | 1007 | **37.6%** |

At 16k the whole lever closes **37.6% of the deficit** and leaves plow 1.6× behind. There is
no length in this table where sparse turns the long-context comparison around.

32k is deliberately absent from that table: plow has no measured 32k TTFT on the R2 basis,
and extrapolating one would be exactly the kind of borrowed number this directory keeps
having to retract. The 32k NET ceiling is 2787 ms against a vLLM 32k TTFT of 3493 ms, which
is the strongest cell in the analysis and the one worth measuring first if this is revived.

## The contingency that governs all of it

Every "gross" figure above is `dense flash − IDEAL sparse flash`, where *ideal* means a walk
that costs exactly its own top-k. **No such kernel exists.** `glm52-dsa-sparse-b3.md`
measured the head-batched kernel that does exist delivering **−11% flash busy against a −68%
walk ratio**, because it recomputes every union KV row against all 64 pack rows. Banking any
of this table needs the per-query membership-skipping MQA flash that b3 costed at *weeks*
and explicitly advised against building on the evidence available.

So the honest dependency order is: **the flash rebuild is the expensive prerequisite, and
the TP-shard is the cheap multiplier that makes it worth having** — not the other way round.
Building the shard first would produce a correct, gated, exactly-zero-value change, because
the sparse path it accelerates is default-OFF and stays that way until the flash lands.

## gfx950, as the directive asks

The one CDNA4 difference that bears on DSA is real but lands in the wrong place. On gfx942
fp8 MFMA runs at **bf16 rate**, so vLLM's fp8 indexer buys bandwidth here and zero
arithmetic; on gfx950 fp8 runs at **2× bf16**, so the same indexer would roughly halve
again. But TP-sharding has already cut that term to 59 ms of a 666 ms gross at 16k — halving
it again is worth ~30 ms, or 0.9 pp of TTFT. **CDNA4 does not change this decision**, and the
`mfma_scale_f32_32x32x64_f8f6f4` / `cvt_scalef32` family that gfx950 adds does not touch the
selection arithmetic at all.

## Verdict

1. The lever is **real and confirmed**: TP-sharding the indexer query axis is worth +409 ms
   at 16k, and it is an emit + collective change, not a kernel one.
2. **u16 index positions, not double-buffering, are the second-order win** (+19 ms vs +19 ms
   for a whole overlap pipeline, at a fraction of the complexity). If this is ever built, do
   u16 and skip the pipeline.
3. It **clears plow's internal 15% bar at 16k (16.2–18.1%)** and **does not close the vLLM
   gap** (37.6% of the deficit at 16k, still 1.6× behind).
4. It is **strictly downstream of the MQA flash rebuild** and is worth exactly zero until
   that exists.
5. Recommended: **do not build it now.** The sparse axis stays default-OFF on both knobs.
   Revisit only if the MQA flash is undertaken, and measure a 32k TTFT first — 32k is the
   only cell where the arithmetic is strong enough to be worth the build.

One thing this scoping does NOT need and should not acquire: `XAllGather` (op 26) is
declared in `crates/packet` and has **no AMD arm**. The gather this design wants is a
put-into-peer-slice plus a gate, which is what `d_xreduce_twoshot_mega`'s phase 2 already
is — reuse that, do not implement a general collective for it.
