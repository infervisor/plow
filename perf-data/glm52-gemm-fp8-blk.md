# `GemmFp8Blk` — the dense prefill block-fp8 GEMM, and `GLM_LINEAR_FP8` measured on a STACKED blob

**§0-BENCH.** Nothing in this file may be placed next to a vLLM number. The end-to-end numbers are
plow-internal, device-side, `plowrt amd-bench`, an A/B **of plow against itself**. §6b-WIDTH already
established why: a served A/B at a lease-sized sample cannot resolve a decode-internal change of
1–2% (its two interleaved *controls* differed by 1.77 ms, four times the effect being measured).

This is the follow-on to `perf-data/glm52-linear-fp8-reeval.md`, whose §4 verdict was:

> The knob is worth about −0.3 ms on the shipping decode configuration, **and it CANNOT BE
> SHIPPED** … Turning it on therefore requires a dense prefill block-fp8 GEMM (`GemmFp8Blk`), which
> does not exist. The −0.31 ms is the size of the prize for building one — not a switch to flip.

`GemmFp8Blk` exists now. It is opcode **107**.

---

## 1. What was missing, precisely

`GLM_LINEAR_FP8=1` re-declares four tensors per layer from the checkpoint's block-fp8 bytes — 1 B/elt
instead of 2 — plus the checkpoint's `[N/128][K/128]` f32 `weight_scale_inv` grid. The DECODE
emitters route all four to `GemvFp8Blk` (44) / `DenseGluFp8Blk` (47). The PREFILL emitters put a
bf16 `Gemm`/`GemmGlu` on the same handles, so a **stacked** blob would read fp8 bytes as bf16 and
run off the end of all four tensors, on all 78 layers, with no fault. `declare_glm_rows` refused
that combination rather than emit it.

The refusal was right, because there was nothing to route to:

| candidate | why it cannot serve `o_proj` or a dense GLU |
|---|---|
| `GemmFp8` (33) / `GemmGluFp8` (36) | the **w8a8** rung: one f32 per output CHANNEL plus a per-row activation scale from `QuantFp8`. Cannot address a `[128,128]` grid; needs an fp8 A operand this path does not produce |
| ops 85/86 `MoeGroupGluPf` / `MoeGroupDownPf` | a **genuine** block-fp8 prefill body — but reachable only under the grouped-MoE contract: expert weight/scale TABLES, `MoeAlignPf`'s `meta`, `row_token` gather indices, `row_partidx`/`row_gate` scatter+scale maps, f32 `part[T*k,H]` output. A plain `o_proj` has none of it |

So it was a new kernel, not a re-route.

## 2. The kernel

`d_gemm_fp8_blk` (`runtime/amd/op_gemm.h`) is `d_gemm_t` with one extra template flag, `WFP8BLK`:
the B-fetch decodes 8 e4m3 bytes to bf16 **exactly** on the way to LDS, and the f32 accumulator is
multiplied by the block scale every 128 K and promoted into a second accumulator. The A operand, LDS
swizzle, XOR swizzle, ping-pong schedule, bf16 MFMA and epilogue are the bf16 path byte for byte —
the same relationship `d_gemm_mxfp4` (`WFP4`) already has to it.

**One scale convention, three kernels.** It indexes `S[(n >> 7) * ceil(K/128) + (k >> 7)]`, which is
what `gemv_rows_fp8_blk` (44) and `d_moe_group_pf_t<FP8=true>` (85/86) already read. A second
convention here is exactly the silent-corruption class this campaign keeps producing, so §4's oracle
checks the two against each other and not only against a reference.

**PROMOTED, NOT FOLDED**, copied deliberately from `op_moe.h`'s grouped arm so prefill and decode
stay in one numeric family. The scale is an arbitrary f32, so it cannot ride the cvt's `scalef32`
operand — that is E8M0/exponent-only and discards the mantissa (`amd_common.h` records a measured
~22% error on a real GLM scale, probed 2026-07-17) — and folding it in software before the bf16
store would round an exact fp8 value AFTER scaling and lose the precision fp8 had. e4m3 has 3
mantissa bits and bf16 has 7, so the LDS decode is lossless.

### 2.1 ONE tile rung, and it is register arithmetic

The promotion accumulator **doubles** a tile's accumulator cost. Against the prefill dispatch's
existing worst case (the 256x256 bf16 arm's 128 accumulator registers):

| tile | acc | + promotion | verdict |
|---|--:|--:|---|
| 64x128 | 16 | 32 | fits |
| **128x128** | **32** | **64** | **this rung** |
| 128x256 | 64 | 128 | ties the worst case |
| 192x256 | 96 | 192 | over |
| 256x256 | 128 | 256 | the whole AGPR file — cannot run 8 waves at all |

So a five-rung block-fp8 family is not merely expensive: its **top two rungs cannot be built**. That
is why `GemmFp8Blk` is emitted DIRECTLY and is not in `gfx950_gemm_inventory` — a
`QuantScheme::BlockFp8` row there must name five opcodes, and adding rungs re-stales every tunedb
record for every shape. 128x128 is also the better of the two feasible fills at GLM's TP4 shapes:
`o_proj` (N=6144) is 48 column-tiles here against 24 at 128x256, and the shared gate/up (N=512) is 4
against 2.

### 2.2 Prefill bucket only

`#if PLOW_BUCKET_PREFILL`, **not** `#if PLOW_FP8`. That flag selects the w8a8 rung, and the object
GLM's stacked blob loads (`interp_prefill_mla_moe`) is built with neither `PLOW_FP8` nor
`PLOW_MXFP4` — yet it already runs block-fp8 through ops 85/86, because the encoding there is a
runtime FIELD and not a compile flag (knob-contract §3). Block-fp8 is additive w8a16 and a
`GLM_LINEAR_FP8` prefill bucket genuinely mixes bf16 GEMMs (q/kv/router/lm_head) with block-fp8
ones, which is the corollary in knob-contract §4.

Decode is untouched **by construction**, and that is not tidiness: a GF=8 arm that fit the register
budget (+6 VGPR, occupancy 2 held, zero spill) was a **+32% decode regression** purely by growing
the decode object 15.6% inside the persistent megakernel. Decode's block-fp8 is ops 44/47 and has no
use for a tiled GEMM.

## 3. Emitter routing

* `emit_glm_mla_prefill` — `o_proj`, on **every** layer including the dense ones, because
  `emit_glm_dense_block_prefill` calls the same MLA emitter.
* `emit_glm_block_prefill` — the shared expert. The fused gate|up **unfuses** into two
  `GemmFp8Blk` + a `Glu`, because there is no `GemmGluFp8Blk`. That is the same shape, for the same
  missing-fusion reason, that the MXFP4 prefill arm already uses; same operand slots, same
  `n.shfu_up`.
* The dense FFN prefill deliberately does **not** switch to op 107. The grouped 1-expert arms cost
  that object nothing (measured) and a dense FFN genuinely *is* an expert. Op 107 exists for the
  case the grouped arms cannot serve.

`declare_glm_rows`'s `rows == 1` refusal is gone. Emitted op counts confirm the routing exactly: the
prefill bucket goes **2021 → 2171 ops**, i.e. +150 = 2 extra packets on each of the 75 sparse layers
— the gate|up unfusing, and nothing else.

## 4. NUMERICS FIRST — the oracle, on hardware, before any timing

`runtime/amd/test_kernels.hip::d_gemm_fp8_blk_k` + `runtime/tests/block_fp8_gfx950_test.c::
run_gemm_blk`. Weights are random valid e4m3 with exp field ≤ 7 (so a K-long dot stays
well-conditioned and a real layout bug reads as ~100% error, not as legitimate f32-vs-f64
cancellation); block scales carry the dynamic range. Tolerance **3e-2**, the same the decode GEMV
uses.

Two legs per shape:

* **vs an f64 CPU reference** over `Σ_k A[m,k] · e4m3(W[n,k]) · S[n>>7][k>>7]`, 512 sampled `(m,n)`;
* **vs `gemv_fp8_blk` (op 44)** on the SAME weights and scale grid, row 0, every column.

The second leg is the point, not a bonus: the two kernels index the grid completely differently —
the GEMV folds the scale into a per-lane chunk partial, the GEMM promotes a whole MFMA accumulator
at a k-tile boundary — so agreeing proves they read **one** convention, which is what the emitter
relies on when it hands both phases the same handle.

| shape | M | N | K | vs f64 ref | vs op 44 |
|---|--:|--:|--:|--:|--:|
| `o_proj` TP4 | 512 | 6144 | 4096 | 0.0032 | 0.0041 |
| `o_proj` TP4, M=2048 | 2048 | 6144 | 4096 | 0.0033 | 0.0024 |
| shared gate TP4 | 512 | 512 | 6144 | 0.0035 | 0.0000 |
| shared down TP4 | 512 | 6144 | 512 | 0.0033 | 0.0012 |
| `o_proj` TP8 | 512 | 6144 | 2048 | 0.0033 | 0.0029 |
| shared gate TP8 | 512 | 256 | 6144 | 0.0037 | 0.0000 |
| shared down TP8 | 512 | 6144 | 256 | 0.0032 | 0.0057 |
| ragged M,N tails | 100 | 130 | 256 | 0.0028 | 0.0012 |

**ALL PASS**, worst 0.0037 against the reference and 0.0057 against the decode kernel. The whole
pre-existing block-fp8 suite (10 GEMV shapes, the MoE expert path, the dense GLU) passes unchanged
in the same run.

`K % 64 == 0` is REQUIRED and the emitter asserts it: the kernel is the `KEXACT` instantiation, so a
ragged K-tile reads 8 halves past the row end and the MFMA silently accumulates the NEXT output
channel's weights. A ragged N and a ragged M are both fine (guarded per element / per row); only K
is unforgiving. Every real block-fp8 K is a 128-multiple by construction — the scale grid is
`[128,128]` — so the assert can only fire on a shape the checkpoint could not have quantised.

## 5. Cost, per bucket

Register budget (`scripts/build_gfx950.sh`'s cliff gate) — **unchanged**:

```
prefill          VGPR=256 AGPR=0 total=256 occ=2 spill=2
prefill_mla_moe  VGPR=256 AGPR=0 total=256 occ=2 spill=2
decode           VGPR=256 AGPR=0 total=256 occ=2 spill=2
```

Object size is the thing the register gate does **not** check, and the GF=8 finding is why it has to
be reported separately (`scripts/objsize_probe.sh`):

| bucket | before | after | delta |
|---|--:|--:|--:|
| `i_decode.co` | 313 328 | 313 328 | **0** |
| `i_flash.co` | 93 496 | 93 496 | **0** |
| `i_prefill.co` | 232 272 | 239 184 | +6 912 (+2.98%) |
| `i_prefill_mla_moe.co` | 347 888 | 354 544 | +6 656 (+1.91%) |

**The decode object is byte-identical.** That is the property that matters: the +32% regression the
GF=8 arm caused was a decode-object growth effect, and this change cannot reproduce it.

## 6. Build digest — the tunedb is re-staled, deliberately

Touching `runtime/amd/*` moves the probed build digest, and every gfx950 GEMM tunedb record is keyed
on it:

```
before   gfx950-d2bd91a05c62b8b0
after    gfx950-b2f50de835dd9495
```

`devgen::tuned_tile_selection::published_measurements_reach_the_compiler_and_change_its_answer`
FAILS until the campaign is re-run — 1350 records go stale and tile selection falls back to the
analytical model. That test is **not** weakened; the failure is the instrument working. The digest
is printable now: `cargo run -p devgen --example probe_digest`, added because
`tuned_tile_selection` reports that records went stale without saying what to re-key them to.

## 7. Coherence FIRST — the gate the MoE confound makes non-optional

**A timing A/B cannot detect wrong numerics on this model, because wrong numerics make the token
FASTER.** Routing is data-dependent: garbage activations collapse the router's top-k, the expert ops
do less work, and the arm over-reports. `PLOW_XR_SHUFFLE` captured 45% of a measured "ceiling" while
deleting nothing. So this ran before any ms was believed.

600 tokens through the **whole 78-layer prefill program** — 78 `GemmFp8Blk` `o_proj` packets, 75×2
shared gate|up, 75 shared down — then 48 greedy steps. TP4, real weights.

| arm | first token after prefill | 48 generated tokens | cross-rank |
|---|--:|---|---|
| `stk_base` (bf16) | **389** | fluent English continuation | all 4 ranks identical, every step |
| `stk_lfp8` (block-fp8) | **389** | **byte-identical to the control** | all 4 ranks identical, every step |

The two arms agree on the argmax of a 154 880-wide vocabulary after 78 layers of block-fp8 prefill,
and then on all 48 following tokens. Across the six timing folds and the two agreement runs, **all 4
ranks were token-identical on every step of every run** — the check a skipped or mis-shaped
collective fails.

**Read this as corroboration, not as the primary gate.** The prompt is a repeated paragraph, which
makes the continuation low-entropy and the argmax confident, so identity here is a weaker
discriminator than a diverse prompt would be. The primary gate is §4's f64 oracle; this is the
end-to-end evidence that the *emitter routing* — which handle goes to which opcode, on which layers,
in which order — is right, which no kernel-level test can show.

## 8. Measured end to end — STACKED blob, current interpreter

GLM-5.2 TP4, 4× gfx950, real weights (`/home/lava/models/GLM-5.2-plow-q`), `plowrt amd-bench --tp 4`,
ctx 1024, 65 steps. Objects built fresh from this branch (`build-amd/lfp8-stk-objs`). Arms
INTERLEAVED inside each fold; **two leases**, folds 1–3 and 4–6, because n=3 could not separate the
effect from the fold noise — the same reason `glm52-linear-fp8-reeval.md` §3.2 needed n=6.

**This is the first time the knob has been measured on a stacked blob at all.** Every prior number
in the record — −0.05 / +0.39 / −0.44 / −0.31 — is from a DECODE-ONLY blob, because a stacked one
was refused.

| fold | `stk_base` (bf16) | `stk_lfp8` | delta |
|---|--:|--:|--:|
| 1 | 25.523 | 25.191 | −0.332 |
| 2 | 25.357 | 25.526 | **+0.169** |
| 3 | 25.877 | 25.042 | −0.835 |
| 4 | 25.475 | 25.192 | −0.283 |
| 5 | 25.497 | 25.273 | −0.224 |
| 6 | 25.964 | 24.968 | −0.996 |
| **median** | **25.510** | **25.191** | |

**mean −0.417 ms, sd 0.428, se 0.175, 5/6 folds negative.**
95% CI **−0.866 … +0.032** (t, df=5) — it grazes zero on a two-sided test and clears it one-sided
(t = −2.38, p ≈ 0.03). Median-of-medians −0.319 ms.

### 8.1 Measured vs projected

| | ms |
|---|--:|
| **projected** — 2547 MB/rank/token removed ÷ 6200 GB/s measured HBM | **−0.431** |
| **measured** — n=6, paired per-fold delta, stacked blob, this interpreter | **−0.417 ± 0.175** |

**101% of the predicted floor.** That is a change of regime, not a re-run: on the previous
interpreter the same knob returned **72%** of the same floor (−0.31 of −0.431,
`glm52-linear-fp8-reeval.md` §3.2), and the shortfall was attributed to `GemvFp8Blk` running at
966 GB/s where bf16 `Gemv` runs at 1728, i.e. half the bytes through a kernel 1.8× slower per byte.
Reaching the full floor means the surrounding interpreter no longer leaves that penalty exposed —
which is §6b-STALE's mechanism running in the *favourable* direction for once. **Do not read it as
the kernel gap having closed.** It has not; the token around it moved (25.5 ms here against 28.1 ms
in the §3.2 campaign, ~9% shorter), and the same knob will move again.

The recorded verdict, now five entries:

| object | verdict |
|---|--:|
| pre-`MLA_MERGE_FOLD`-rewrite | −0.05 (noise) |
| post-fold-rewrite | **+0.39** (a regression) |
| 2026-07-28 12:19, `CORESIDENT=1` | **−0.44** |
| `glm52-linear-fp8-reeval.md`, `CORESIDENT=1` / SHIPPING, decode-only | −0.13 (noise) / **−0.31** |
| **this branch, SHIPPING, STACKED** | **−0.417 ± 0.175** |

### 8.2 What it cost on the prefill side — a real TTFT regression

The same change makes `o_proj` and the shared expert *prefill* GEMMs block-fp8, and that is not
free. From the coherence run (600-token prompt, n=1 per arm — indicative, not a campaign):

| arm | prefill, 600 tokens |
|---|--:|
| `stk_base` | 1257.9 ms (477 tok/s) |
| `stk_lfp8` | 1299.4 ms (462 tok/s) |
| | **+41.5 ms, +3.3%** |

Two plausible causes, neither measured apart: `o_proj` moves from `pick_tile`'s selected bf16 rung
to the single 128x128 block-fp8 rung, and the shared gate|up unfuses into two GEMMs plus a `Glu`,
which materialises `[T, imoe_l]` to HBM and reads it back — the one place MXFP4 prefill pays the
same tax, and for the same missing-fusion reason. **A `GemmGluFp8Blk` would remove the second, and a
second tile rung (64x128 for narrow M) may remove part of the first.** Both are follow-ups; neither
blocks the decode win.

Note this cuts the other way from the usual TTFT argument: prefill is a per-REQUEST cost and decode
is per-TOKEN, so at 128 output tokens the decode saving (128 × 0.417 = 53 ms) already exceeds the
41.5 ms of TTFT. It is a genuine trade, not a free win, and it is context- and output-length
dependent.

### 8.3 Honesty about the instrument

* **The box was contended for the whole campaign.** Another agent's `plowrt serve` ran on the other
  four cards throughout both leases (`rocm-smi --showpids`, checked at four points). A lease
  serialises the four cards it holds; it does not partition the shared power budget. The control arm
  drifted 25.36 → 25.96 ms across the six folds — **more than the effect being measured.**
  Interleaving the arms inside each fold is what makes that survivable, because the drift lands on
  the arm and its control alike, and it is why the statistic reported is the paired per-fold delta
  and never the difference of two medians. Fold 2's `+0.169` is the one sign flip and it lands
  exactly where the foreign process arrived.
* **§0-BENCH.** These are device-side `amd-bench` numbers, an A/B of plow against itself. They must
  not be placed next to a vLLM number. §6b-WIDTH already established that the served endpoint at a
  lease-sized sample cannot resolve a change this size.

## 9. How much of the decode weight stream moved

Per rank per token, TP4, ctx 1k, 78 layers, from `perf-data/glm52-weight-stream-split.md` §1:

| | before | after | |
|---|--:|--:|---|
| total weight stream | 18 214.5 MB | **15 667.5 MB** | **−2547 MB, −14.0%** |
| carried as bf16 | 12 652.5 MB (69.5%) | **7 558.5 MB (48.2%)** | −5094 MB |
| carried as fp8 | 5 562.0 MB (30.5%) | **8 109.0 MB (51.8%)** | +2547 MB |

The four tensors are `o_proj` (3744 MB/token), shared gate+up (900) and shared down (450) — 5094 MB
of bf16 becoming 2547 MB of fp8. **fp8 crosses 50% of the stream for the first time.**

What is still bf16 and why (unchanged by this work, listed so the next lever is obvious):

| behind | MB | blocker |
|---|--:|---|
| `GemvQkv` fusions A and G | 5206.5 | no block-fp8 arm; un-fusing is explicitly refused (fusion A exists to stop N=512/N=64 starving CUs and measures 83% of ceiling fused) |
| `lm_head` + router | 2040 | **bf16 in the checkpoint** — a SHARDING lever, not a precision one |
| `q_absorb` | 2496 | a genuine PRODUCT of two fp8 tensors; needs requantisation, the only part with an accuracy question |
| `v_absorb` / `q_rope` | 624 | fp8 values survive, the `[128,128]` grid does not survive the slice/transpose |

## 10. Follow-ups this opens, in order

1. **`GemmGluFp8Blk`** — removes §8.2's extra `[T, imoe_l]` HBM round-trip and half the TTFT
   regression. Cheap: it is `d_gemm_fp8_blk` with `GLU=true`, and the epilogue already exists.
2. **A second tile rung (64x128) for `GemmFp8Blk`** — the narrow-M case. At that point both rungs
   belong in `gfx950_gemm_inventory` as a `QuantScheme::BlockFp8` row, and the five-opcode shape of
   that table has to be relaxed first.
3. **`gemv_rows_fp8_blk`'s memory-level parallelism** — still the item `glm52-linear-fp8-reeval.md`
   §4 named, and still the thing that decides whether block-fp8 is worth its floor or twice it.
4. **Re-run the GEMM tuning campaign** against `gfx950-b2f50de835dd9495` (§6).

