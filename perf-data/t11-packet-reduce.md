# T11-packet-reduce — does REDUCING decode packet count improve TPOT?

Hypothesis (user): "chunk design reduces packets, could improve perf" — fewer decode
packets → less counter-gate overhead → lower TPOT. This campaign MEASURES it (does not
assume the doc's <4% dispatch-floor ceiling).

Method of record: `gemma4_sm120_chat`, `PLOW_PREFILL=0`, `PLOW_WARMUP=16`, 136 steps →
120 timed, RTX PRO 6000 Blackwell (sm_120, 188 SMs), TP1 B1, GF_FULL=4, UNISEG,
`PLOW_NS_FULL_ABS=48`. Build `build-t11` (plain env, not nix). Parity = greedy token
stream (`PLOW_IDS`) byte-identical to the fused HEAD baseline (md5 of the id line).

## Baseline decode packet counts (prog 6, T=1, 188 CU)

| model | layers (full) | decode packets | wg-packets | pkts/layer |
|---|---|--:|--:|--:|
| gemma-4-12B    | 48 (8)  | 542 | 46 591 | ~11.3 |
| gemma-4-31B    | 60 (10) | 676 | 59 479 | ~11.3 |
| gemma-4-26B-A4B| 30 (5)  | 521 | 41 089 | ~17.4 (MoE) |

## Ceiling (from T9b skeleton, `-DPLOW_NV_SKELETON_DECODE=ON`)

Full 59 479-entry decode stream, every op body compiled out → pure dispatch/gate/sync:
**0.659 ms/step on 31B, ctx-invariant.** That is the absolute cap on ANY packet/gate
lever. 0.66 ms / 676 packets ≈ **~1 µs per packet** of pure dispatch, and that 1 µs is
largely overlapped by bodies at real ctx. Removing k packets can save at most ~k µs.

## Lever 3 — coarsen non-load-bearing fine deps: VERIFIED NO-OP

`select_granularity` (devbuild.rs) union-finds fine-edge regions and downgrades every
fine edge in a homogeneous-work region to coarse. Emitter report for the decode program:

    counter granularity: 0 fine edges kept, 270 downgraded to coarse   (31B decode)

**Zero fine edges survive in decode** on all three models — the declared `hn_dep`
(gemv→headnorm) and `fa_dep` (headnorm→flash) maps are all downgraded. There is no
surviving fine-dep counter to coarsen. Lever 3 has nothing to do. CONFIRMED, no GPU.

## Lever 2 — fuse/widen the small norm into its consumer

- **Fuse (norm mode 2, GEMV recomputes the row RMS):** pre-existing in-code result,
  `gemma4.rs:1862-1867`: MEASURED **22.4 → 24.4 ms/token, SLOWER**. Five consumers
  (q,k,v,gate,up) each redo the reduction; the two gates saved do not pay for it. The op
  still supports mode 2 but the compiler does not use it here. REFUTED (documented).
- **Widen:** the input RMS already runs on `rows=[0]` (1 CU) and for l>0 is fused into
  the prior layer's `NormResidualNorm` (so it is not even a separate packet per layer).
  Widening a single T=1 hidden-vector RMS across CUs needs a cross-CU split+merge (adds a
  gate) or redundant full recompute (no wall-time gain) — architecturally weak; the norm
  is ~1-2 µs, far below the gate-wait it is blamed for. Not pursued past the analysis.

## Lever 1 proxy — QKV fusion A/B (the clean gate-count probe)

Direct new-code-free measurement of counter-gate marginal cost. `PLOW_NO_FUSE_QKV=1`
reverts the fused `GemvQkv` (1 packet, all-CU) to the historical `split3` path (q/k/v =
**3 separate bf16 Gemv packets on disjoint CU sets** = +2 packets/layer). **Total
wg-packets identical** (46 591 / 59 479) — isolates GATE count, not work. Tokens
bit-identical (each output column is the same per-column dot).

Packet counts: 12B 542 → 622 (+80 = +2/layer × 40 non-full). 31B 676 → 776 (+100).

| model @ctx | base (fewer pkts) | +2 gates/layer (nofuse) | Δ (ms) | Δ% | sd | parity |
|---|--:|--:|--:|--:|--:|:--:|
| 12B @1k | 18.890 | 18.861 | **−0.029** | −0.15% | 0.012 | ✓ identical |
| 31B @1k | 46.371 | 46.329 | **−0.042** | −0.09% | 0.026 | ✓ identical |
| 12B @4k | 18.925 | 18.900 | **−0.025** | −0.13% | 0.011 | ✓ identical |
| 31B @4k | 46.528 | 46.486 | **−0.042** | −0.09% | 0.025-0.036 | ✓ identical |

**Every case: ADDING packets is neutral-to-slightly-FASTER.** The counter gate's marginal
TPOT cost is below the 0.01-0.03 ms noise floor (and the concurrent split3 overlap edges
out uniform fill). Fusing (fewer packets) is the shipped default yet is not a TPOT win —
it was adopted for uniform CU fill, not measured latency.

## Why packet reduction cannot help here (architecture)

The persistent interpreter runs one packet to completion per block, `__syncthreads()`,
then claims the next from one atomic cursor. The dynamic smem **arena is a UNION across op
bodies — each fully consumes its arena before the next instruction's gate**, and the
ACQUIRE `__threadfence()` at every gate forbids reads across it. So there is **no cross-op
prefetch**: merging two packets removes only a counter gate, it cannot lengthen any
prefetch pipeline. In-kernel producer/consumer double-buffering IS exploited *within* each
op (GEMV `PGM_STAGES=3` cp.async ring, flash `PLOW_NV_FA_PIPE` K-prefetch), but that is
intra-op and unaffected by packet count. The only cross-op overlap is inter-block via the
scheduler counters — which the A/B shows already saturates the benefit.

## Verdict

**Packet reduction is NOT a decode TPOT lever, and the measured result is BELOW the doc's
<4% prediction — it is ≈0 (indistinguishable from noise, marginally negative).**

- The 0.66 ms dispatch floor is the hard cap; the real gate-WAIT the traces show
  (GEMV_QKV/GEMV 24-34%, flash-merge) is producer→consumer *latency* and *split
  imbalance*, NOT counter-mechanism overhead. The one producer-latency gate lever the
  traces localised (flash full-layer split imbalance) was already harvested (ns47/ns48,
  −3.4% @128k) and the MoE gates were already fused out (T9a: 7-11%).
- Lever 1 (coalesce 3 HeadNormRope → 1) would remove ~2 gates/layer — the same class this
  A/B shows is worth ≤0.04 ms and possibly negative. Not worth a new fused device kernel
  and its parity risk; the ceiling is confirmed empirically, so it was not built.
- Lever 2-fuse refuted (slower), lever 2-widen architecturally weak, lever 3 a no-op.

**Nothing to fold to default.** 12B (where plow wins) shows no regression in any arm; the
honest answer to "does fewer packets help" is **no**.
