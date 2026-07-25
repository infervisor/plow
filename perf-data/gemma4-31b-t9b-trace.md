# Gemma-4-31B decode — single-block per-op trace (T9b-31b-tune)

Trace-first decomposition of ONE decode step, block 0, thread 0, via `PLOW_NV_TRACE=1`
(interp_sm120.cu launch helper self-dump). Built through a new CMake knob
`-DPLOW_NV_TRACE_DECODE=ON` (decode object only; the `_pf` trace stays on `PLOW_NV_TRACE_PF`).
Reducer: `scripts/trace_reduce.py`. 31B is compared against 12B (where plow decode WINS
vs vLLM) at the SAME contexts — the delta profile is the tuning map.

## Method

- Build: `cmake -S runtime -B build-trace -DPLOW_CUDA=ON -DPLOW_NV_TRACE_DECODE=ON` (plain
  system env, NOT nix — glibc RUNPATH). Harness `gemma4_sm120_chat`, PLOW_UNISEG pkt, prefill
  primed to ctx, trace dumps the FIRST decode step (KV already grown to ctx).
- **READ THE SHAPE, NOT THE ABSOLUTE TOTAL.** `clock64()` serializes on the recording thread
  and the ACQUIRE fence lands inside "body", so the traced step over-reports (31B@1k traces
  46.15 ms vs the untraced 45.30 ms). The gate/body/sig SPLIT and the per-op cyc/op are the
  finding, not the sum. Block 0 is one of 188; its own gate is idle-wait, so it UNDER-reports
  the cross-block serialization that sets end-to-end latency.
- Block 0 executes 308 of the 59 479 wg-packets (31B) / 243–244 of 46 591 (12B). Op counts
  below are block-0's share, one packet per layer per op-slice.

## Block-0 gate/body/sig split

| trace | pkts | total_cyc | gate% | body% | sig% | untraced TPOT |
|---|--:|--:|--:|--:|--:|--:|
| 31B @ 4k   | 308 | 111.3 M | 6.6  | 92.2 | 1.2 | 45.30 ms (loses vLLM +1.4%) |
| 31B @ 128k | 308 | 141.2 M | 10.2 | 88.7 | 1.0 | 58.45 ms (loses vLLM +5.4%) |
| 12B @ 4k   | 244 |  45.0 M | 9.3  | 88.5 | 2.2 | ~18.3 ms (WINS vLLM) |
| 12B @ 128k | 243 |  58.3 M | 17.0 | 81.4 | 1.7 | ~24.2 ms (WINS vLLM) |

Body dominates block 0 everywhere (81–92%). Gate is small and — counter to a "fixed dispatch
intercept" story — 31B's block-0 gate FRACTION is LOWER than 12B's, not higher.

## Per-op body cyc/op — 31B vs 12B (the delta map)

| op | 31B b/op | 12B b/op | ratio | ∝ |
|---|--:|--:|--:|---|
| GEMV_GLU (gate\|up) | 740 k | 383 k | 1.93× | N·K = inter·hidden (21504·5376 / 15360·3840 = 1.96) |
| GEMV (o / down)     | 314–325 k | 148–151 k | 2.1× | hidden² and inter·hidden (1.96×) |
| GEMV_QKV            | 281 k | 108 k | 2.6× | (heads·hd + 2·kvh·hd)·hidden |
| FLASH_DECODE @128k  | 401 k | 185 k | 2.17× | heads · hd · KV |

**Every GEMV body scales EXACTLY with its FLOP/byte count** — 31B's GEMVs are 1.9–2.1× the
12B's because 31B is 1.9–2.1× the weights, not because they are less efficient. Decode GEMV is
HBM-bandwidth-bound and already matches vLLM byte-for-byte; there is no GEMV-body win to take
(consistent with H1's unroll refutation). **The o_proj/down_proj/GLU GEMVs are not the lever.**

## What is 31B-SPECIFIC (the actual gap)

The 4k→128k gate growth, decomposed per op (block-0 cyc):

| op | 31B Δgate 4k→128k | 12B Δgate 4k→128k |
|---|--:|--:|
| GEMV (all)   | +4.9 M | +5.9 M |
| **FLASH_MERGE** | **+3.8 M (658 k cyc/op gate @128k, 2.9% of step)** | **0 — not on 12B block 0** |

**FLASH_MERGE gate is the one 31B-specific long-ctx penalty in the trace.** Root cause is the
full-layer (hd512, kv4) flash split imbalance: `n_work = n_grp·nsplit = 16·16 = 256` work items
over 188 resident blocks → 68 blocks do 2 items (2× time), 120 do 1. FLASH_MERGE must wait for
the 2× blocks, so its gate balloons at long ctx. 12B (kvh_full=1 → n_grp differs, fewer full
layers) does not surface this on block 0. `n_grp=16` shares only gcd 4 with 188, so ceil() is
always ragged unless `nsplit` is a multiple of 47 (16·47 = 752 = 4·188, perfectly balanced).

## Dispatch floor (SKELETON build — bodies removed, `-DPLOW_NV_SKELETON_DECODE=ON`)

The gate/sync/schedule machinery run on the REAL 59 479-entry decode stream with every op body
compiled out (garbage logits, timing only) — the pure interpreter dispatch floor:

| build | ctx | TPOT |
|---|--:|--:|
| 31B skeleton | 4k   | **0.659 ms** |
| 31B skeleton | 128k | **0.659 ms** (identical — no bodies, so ctx-invariant) |

**The ENTIRE dispatch/gate/sync cost is 0.66 ms** — the absolute ceiling on any gate-coalescing
or scheduler lever. So H3's "+2.0 ms fixed intercept" is NOT mostly dispatch: 0.66 ms is
dispatch, the other ~1.4 ms is real fixed math (T=1 GEMVs at tiny KV, embed, argmax, norms) a
faster schedule cannot remove. Gate coalescing can chase at most a fraction of 0.66 ms — relevant
only at 1k where the whole gap is 0.6 ms, and NOT a long-ctx lever.

## Tuning implications (validated separately in campaign T9b-31b-tune)

1. GEMV bodies: HBM-bound, = vLLM. No lever (matches H1).
2. **Flash full-layer split imbalance (lever c): the clearest trace signal.** Emitter already
   exposes `PLOW_NS_FULL_ABS` (full-layers-only nsplit, leaves the hd256 sliding layers at 16).
   Grid-aligned nsplit (multiple of 47) should collapse the FLASH_MERGE gate at long ctx.
3. Short-ctx gap (+1.4%@1k, 0.6 ms) is gate-light and body-bound — little room; PTXSYNC
   (shipped) already took −0.6%.

## Confirmation — FLASH_MERGE gate collapse (ns47 vs base, block-0 @128k)

Re-tracing the shipped ns47 (grid-aligned) build at 128k, block 0:

| metric @128k | base ns16 | ns47 (aligned) |
|---|--:|--:|
| FLASH_MERGE gate cyc/op | **658 390** | **47 278** (14× lower) |
| block-0 total gate cyc | 14.41 M (10.2%) | 6.65 M (4.9%) |
| TPOT (method of record) | 58.576 ms | 56.588 ms (−3.4%) |

The predicted mechanism is exactly what moved: balancing full-layer flash work to 4 items/block
(752 = 4·188) removes the wait for the 2×-loaded blocks, so the FLASH_MERGE gate — the one
31B-specific long-ctx penalty this trace localised — nearly vanishes. See
`perf-data/gemma4-31b-t9b-tune.md` for the full ladder and gates.
