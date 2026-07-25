# Tuner, second knob family — FLASH DECODE (H100 NVL, 26B fp8)

Extends `scripts/tune_decode_sweep.sh` from the GEMM knobs
(`perf-data/tuner-decode-sweep-h100.md`) to the flash-attention decode arm.
Raw rows: `tune-flash-h100-26b-fp8.jsonl`.

Swept around round 15's best fp8 configuration — occ-2, `PLOW_NS_ABS=32`,
`FP8_RB=2`, `GV_MOE_RB=1`, `GV_UNROLL=4`, `GV_UNROLL_GLU=2`, `GV_MOE_UN=2`,
`PLOW_MOE_DOWN_SG=4` — which the harness reproduces at **4.601 ms** against its
recorded 4.606.

## What is new in the harness

**Per-op ablation, not just total TPOT.** `--ablate-lo/--ablate-hi` build a
*twin* object per configuration with one opcode's body compiled out, keeping
every gate and signal. Each row carries `TPOT(full)`, `TPOT(ablated)` and the
difference — that op's true wall-clock cost at the shipped grid, imbalance
included. FLASH_DECODE is opcode 12. For a ~0.4 ms op inside a ~4.6 ms step this
is the difference between signal and noise, and it turned out to matter for more
than precision (see GF_FULL below).

**Objects keyed by the SHA of their define string** rather than a hand-built
name, so a knob family can be added without a naming scheme that can silently
collide. An unnamed flash axis emits no `-D`, so a sweep that ignores the family
builds byte-identical objects to one from before it existed.

## The contention caveat, unchanged

No number here was taken on a verified-idle GPU: the box's ~52.5 GB foreign
holder never drained, and a second agent measured concurrently throughout. Every
row records `vram_before_mib` and `uncontended: false`, and `tunedb-decode
ingest` refuses to qualify them — they are stored `provisional`. Rep spread is
0.002–0.015 ms and all cells were measured minutes apart on the same binaries,
so *comparisons within the table* are sound; the absolute numbers are not
certified.

## Result 1 — WPR still wins, and the ablation proves it is flash that moved

| config | TPOT | ablated (flash body out) | **flash** |
|---|---|---|---|
| `FA_WPR=1` (shipped) | 4.604 | 4.243 | **0.361** |
| `FA_WPR=0` (pre-round-9 body) | 5.167 | 4.253 | **0.914** |

Warp-per-row cuts flash **0.914 → 0.361 ms, 2.53×**, and the ablated remainder
is unchanged (4.253 vs 4.243) — the knob touches only the arm it names. Round 9
measured 1.079 → 0.686 on bf16 at occ-1; it wins harder at occ-2 on fp8. The
shipped default is correct and is now correct *for a measured reason*.

## Result 2 — the context growth is ENTIRELY the flash arm

| ctx | TPOT | ablated (non-flash) | flash | flash share of step |
|---|---|---|---|---|
| 1024 | 4.604 | 4.243 | 0.361 | 7.8 % |
| 8192 | 4.811 | **4.238** | 0.573 | 11.9 % |
| 32768 | 5.550 | **4.236** | **1.314** | 23.7 % |

**The non-flash remainder is flat to within 0.007 ms across a 32× context
range** — 4.243, 4.238, 4.236 — while flash grows 3.6×, 0.361 → 1.314 ms.
Total TPOT grows +0.946 ms from 1k to 32k and flash's own cost grows +0.953 ms:
**100.7 % of the context growth is the flash arm**, i.e. all of it, to within
the measurement's own spread.

The campaign asserted this ("the growth is entirely the 5 full-attention
layers"); this measures it directly, at three contexts, with the rest of the
engine held still as a control. It is also the whole argument for why the flash
knobs are worth conditioning on ctx when the GEMM knobs were not: at ctx=1024
flash is 7.8 % of the step and tuning it is nearly pointless, while at 32k it is
23.7 % and it is the only thing worth tuning.

## Result 3 — `GF_FULL=8` loses, and a third of the loss is NOT flash

`GF_FULL=8` reads each full-attention KV head once instead of twice, so the
standing theory was that it should pay *more* as ctx grows. It does the
opposite, and the ablation says why.

| ctx | config | TPOT | ablated | flash | dyn smem | occ/SM |
|---|---|---|---|---|---|---|
| 1024 | `GF_FULL=4` | 4.604 | 4.243 | 0.361 | 16448 B | 2 |
| 1024 | `GF_FULL=8` | 4.846 (+0.242) | 4.324 (**+0.081**) | 0.522 (+0.161) | 24640 B | 2 |
| 8192 | `GF_FULL=4` | 4.811 | 4.238 | 0.573 | 16448 B | 2 |
| 8192 | `GF_FULL=8` | 5.328 (**+0.517**) | 4.318 (**+0.080**) | 1.010 (+0.437) | 24640 B | 2 |

Two separable costs, and the split is only visible because of the ablation:

- **A constant arena tax of ~0.080 ms, independent of ctx.** The ablated
  remainder rises by +0.081 at 1k and +0.080 at 8k — with flash's body compiled
  out. The knob widens the dynamic smem arena 16448 → 24640 B while
  `occ_per_sm` stays 2, so it is *not* an occupancy loss. The arena is a
  **union sized by the largest claim**, and the largest claim is flash's, so
  widening it charges every other op for space only flash uses.
- **A flash penalty that GROWS with ctx**: +0.161 ms at 1k, +0.437 ms at 8k.
  Whatever GF_FULL=8 saves in KV re-reads, it loses more of at longer context —
  the regression is worst exactly where the theory predicted a win.

**`GF_FULL=8` is refuted again, now with a mechanism.** Round 11 measured it
worse at every ctx on bf16 pre-WPR and could only report that; the flash body
has since been rewritten and it is still worse, more so as ctx grows.

Scoring on total TPOT alone would have booked the whole regression against flash
and left the wrong mechanism in the record. This is the same
isolation-vs-in-context coupling the campaign has hit three times, seen from the
other side: here the *in-context* number is inflated by a cost the op itself does
not pay.

## Result 4 — `NS_FULL_ABS`: an optimum that DOES move with ctx

The packet-side split for the **full-attention layers only** — the 5 layers that
read the whole context, where the other 25 are window-capped at 1024.

| `NS_FULL_ABS` | ctx 1024 | ctx 8192 |
|---|---|---|
| default (emitter CU-fill) | **4.604** (0.361) | 4.811 (0.573) |
| 33 | 4.605 (0.363) | 4.802 (0.557) |
| 66 | 4.612 (0.343) | **4.715** (**0.445**) |

**The optimum moves: the emitter default at ctx=1024, `66` at ctx=8192**, where
it is worth **−0.096 ms, about 2 % of the step** — an order of magnitude more
than `33`'s −0.009. This is the first knob in either family whose *winner*,
not merely its effect size, changes with context.

`66` is also the principled value at this occupancy: `n_cu/gcd(n_grp,n_cu)` =
264/4 = 66, where the build script's recorded 33 is the `n_cu=132` figure. So
the knob is coupled to occupancy **and** to ctx — the design doc's joint-sweep
claim, now demonstrated on the flash family.

### Why it moves — the ablation gives the mechanism

Splitting the full layers trades two costs against each other, and only the
ablated twin separates them:

| ctx | Δ flash (body) | Δ ablated (gate/protocol) | Δ TPOT |
|---|---|---|---|
| 1024 | **−0.018** | **+0.026** | +0.008 |
| 8192 | **−0.128** | **+0.032** | **−0.096** |

More splits make flash's **body** cheaper but add **gate and signal** time for
the extra work items — which still execute with flash's body compiled out, which
is exactly why the twin can see them. **The gate cost is near-constant (+0.026 →
+0.032) while the body saving grows 7× (−0.018 → −0.128).** So splits pay only
once the body term is large enough to outrun a fixed protocol cost, and that is
a context condition.

This was a *prediction* before it was a measurement: the ctx=1024 decomposition
alone implied `66` should improve with context, and it did, by 0.104 ms of
swing. An earlier revision of this card — written on the 1024 data — concluded
"the effect size is ctx-dependent, the optimum is not". The 8192 point refutes
that, and it is corrected here rather than left standing.

## Result 5 — `FA_GF` is a null at short ctx, and that is informative

| `FA_GF` (sliding-layer GQA fusion) | ctx 1024 | ctx 8192 |
|---|---|---|
| 2 (shipped) | 4.604 (0.361) | 4.811 (0.573) |
| 4 | 4.604 (0.362) | 4.812 (0.573) |

Identical at **both** contexts — flash's own cost matches to 0.001 ms and 0.000
ms. Not "small", *nothing*. That is the expected shape: `FA_GF` governs the
**sliding** layers, which are window-capped at 1024, so they read the same
window at ctx=8192 as at ctx=1024 and regrouping cannot change how many bytes
move. It is the exact counterpart to `FA_GF_FULL`, which governs the 5 full
layers and moves the number hard.

Recorded because a null on the sliding-layer knob is the control that makes the
full-layer results credible. The two knobs are structurally identical — same
fusion, different layer set — and only the one attached to the layers that grow
with ctx has any effect, at either context. A sweep that found *everything*
mattered would be measuring its own noise; this one finds the thing that cannot
matter, doesn't.

## What this puts in `tunedb` — nothing new, and that is the result

The flash grid is a **3-rep screening pass**, and `Stats` refuses fewer than 5
samples at construction, so none of these rows can become a record. That is the
same screen-wide / confirm-narrow split the GEMM round used, and here it lands
on a convenient answer: **every flash knob's winner is the value already
shipped** — `FA_WPR=1`, `FA_GF_FULL=4`, `FA_GF=2`, `FA_KUN=1`. The winning
configuration is therefore the one the store *already holds* at 5 reps from the
GEMM round, `mb2_ncu264_un4_glu2_mun2_sg4_ns32` at 4.772 ms.

So the flash family's contribution to the record is a confirmation, not a new
selection. That is worth stating plainly rather than manufacturing a record to
show for the work: the tuner's job includes reporting that the current defaults
survive, and this family's defaults did.

## Reproduce

```
scripts/tune_decode_sweep.sh \
  --model /workspace/models/gemma-4-26B-A4B-it --checkpoint <fp8-canonical-shards> \
  --dtype fp8 --base-defines "-DPLOW_NV_FP8_RB=2 -DGV_MOE_RB=1" \
  --occ "2:264" --ns-abs 32 --gv-unroll 4 --gv-unroll-glu 2 \
  --fa-wpr "0 1" --fa-gf-full "4 8" --fa-kun "1 2 4" --ns-full-abs "0 33 66" \
  --ctx "1024 8192 32768" --reps 3 --ablate-lo 4096 \
  --results perf-data/tune-flash-h100-26b-fp8.jsonl
```

## Cost note, and what it forced

The fp8 packets carry **no prefill program** (`prefill_buckets=0`), so the
prompt is consumed one decode step at a time: a ctx=32768 run costs ~160 s
against ~20 s at ctx=1024, and with an ablated twin at 3 reps that is ~16 min
per long-ctx point. A full cross of 4 flash axes × 3 ctx × a twin is days of
GPU on a shared card. The sweep is therefore **one axis at a time** around the
round-15 base, with the expensive 32768 leg spent on `GF_FULL` and
`NS_FULL_ABS` — the two knobs that touch only the 5 full-attention layers, and
so the only two with a ctx argument. Emitting packets with prefill buckets would
remove this constraint and is the cheapest available speedup for any future
long-ctx sweep.
