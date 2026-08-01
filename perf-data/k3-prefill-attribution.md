# K3 prefill, attributed — and the 179 ms floor (which is EXPERT BANDWIDTH, not padding)

**Measured 2026-07-31**, K3 TP8 on 8x MI355X, B=1 packet, `amd-bench --steps 1 --prompt <T ids>`.
Every number here is from `PLOW_TRACE_RAW` + `scripts/k3_trace_report.py`.

## 0. Prefill had never been traced

Three independent gaps, each of which produced NOTHING rather than complaining:

1. The kernarg handed the trace pointer **only to the decode program**
   (`trace: if p == self.decode { ... } else { 0 }`), so every prefill dispatch ran with a null
   trace.
2. The buffer was sized for the decode program's stream. K3's prefill buckets carry 2942 stream
   entries against decode's 2459, so handing the pointer over without resizing would have
   overflowed it.
3. The trace slot is unique per (workgroup, packet) — *"no atomic and no ring"* — so every dispatch
   OVERWRITES the previous one. A run that prefills and then decodes wrote a DECODE trace no matter
   how large the prompt.

All three are fixed; `amd-bench` now writes `<PLOW_TRACE_RAW>.prefill` before the decode loop.

## 1. Where a 1024-token prefill goes (792.8 ms, 2942 packets, 712,797 records)

| subsystem | ms | % |
|---|--:|--:|
| **MoE** (DOWN_PF + GLU_PF + COMBINE + ROUTER + ALIGN) | **283.7** | **35.6%** |
| **KDA** (state step + conv3) | **219.9** | **27.6%** |
| GEMM (small / wide / med) | 158.5 | 19.9% |
| collectives (XREDUCE + XREDUCE2) | 93.8 | 11.8% |
| MLA (flash prefill + merge) | 12.7 | 1.6% |

**Prefill is MoE- and KDA-bound, not GEMM-bound.**

## 2. The shape sweep

| tokens | chunks | wall | tok/s |
|---|--:|--:|--:|
| 256 | 1 (512 bucket) | 434.5 ms | 589 |
| 512 | 1 | 487.8 ms | 1050 |
| 1024 | 1 | 775.9 ms | 1320 |
| 2048 | 1 | 1386.7 ms | 1477 |
| 4096 | 1 | 2382.9 ms | 1719 |
| **6144** | **2** (4096+2048) | 3791.1 ms | **1621** |
| 8192 | 1 | 4587.2 ms | 1786 |

Fitting the single-chunk points: **prefill(T) ~= 179 ms + 0.538 ms/token.** A large FIXED cost per
chunk — 23% of a 1024-token prefill, 41% of a 256-token one.

6144 dips because it plans as two chunks and pays the fixed cost twice. That is the DP making the
RIGHT call, not a bug: 4096+2048 measured 3791 ms against 4587 ms for a single 8192 chunk.
`LAUNCH_ROWS = 416` implies ~224 ms per launch against the measured ~179 ms — the right order,
slightly conservative.

## 3. Per-op scaling, T=1024 -> T=8192 (8x tokens)

| op | T=1k ms | T=8k ms | ratio | %@8k |
|---|--:|--:|--:|--:|
| `KDA_STATE_STEP_G` | 158.4 | 1332.5 | 8.41x | **29.2%** |
| `MOE_GROUP_DOWN_PF` | 187.3 | 779.9 | **4.16x** | 17.1% |
| `KDA_CONV3` | 61.7 | 636.3 | **10.31x** | 13.9% |
| `MOE_GROUP_GLU_PF` | 88.7 | 393.8 | **4.44x** | 8.6% |
| `XREDUCE` | 42.5 | 352.7 | 8.30x | 7.7% |
| `FLASH_MLA_PREFILL_FP8` | 8.0 | 230.1 | **28.6x** | 5.0% |

Three shapes, three different causes:

* **MoE is SUBLINEAR (4.2x)** — it gets more efficient as T rises, because expert tiles fill up.
* **`KDA_CONV3` is SUPERLINEAR (10.3x)** and its straggler explodes from 442 us to **4384 us**
  (48% of its own body). Load imbalance that grows with T — a separate defect from the fill problem.
* **`FLASH_MLA_PREFILL_FP8` at 28.6x** is attention being O(T^2). Only 5% at 8k, but it is the term
  that bites at 32k+.

**KDA is 43.1% of an 8192-token prefill** and runs on **192 of 256 workgroups**
(`state_step_blocks = proj/bv = 1536/8`).

## 4. THE FLOOR, and why it is exactly 179 ms

The decisive measurement. T=256 and T=512 both plan to the **512 bucket**, so they run the same
program — and within it:

| op | T=256 | T=512 | |
|---|--:|--:|---|
| `KDA_STATE_STEP_G` | 40.7 ms | **80.0 ms** | exactly 2x — scales with REAL tokens |
| `MOE_GROUP_DOWN_PF` | 115.1 ms | **113.9 ms** | **FLAT** |
| `MOE_GROUP_GLU_PF` | 65.3 ms | **65.0 ms** | **FLAT** |

KDA scales because `rebase_chunk` patches every KDA op's row count to `clen`. **The grouped MoE
GEMMs do not scale at all**, and the reason is not the chunk padding — it is a FLOOR:

```
n_exp = 896, top_k = 16, MPF_BM = 64

Once T*top_k is more than a few thousand, essentially EVERY expert is hit at least once, so the
tile count is pinned at n_exp and the padded row count is pinned at 896 * 64 = 57,344 — whatever
the prompt length is.
```

| T | real slots | padded rows | fill |
|---|--:|--:|--:|
| 256 | 4,096 | 57,344 | **7.1%** |
| 512 | 8,192 | 57,344 | 14.3% |
| 1024 | 16,384 | 57,344 | 28.6% |
| 8192 | 131,072 | 172,032 (3 tiles/expert) | 76.2% |

**DOWN_PF 114 ms + GLU_PF 65 ms = 179 ms**, which is precisely the fixed-per-chunk constant fitted
independently in §2. The fit and the mechanism agree without being tuned to each other.

## 5. What to do about it

`MPF_BM = 64` is the knob, and it must be chosen per bucket rather than fixed, because the trade
reverses with T:

* At **T <= 3584** (rows/expert < 64) every expert needs exactly ONE tile at either BM, so halving
  BM halves the M-work and changes the weight traffic NOT AT ALL.
* At **T = 8192** (146 rows/expert) BM=64 gives 3 tiles per expert and BM=32 gives 5 — so a smaller
  BM would *increase* weight re-reads. BM=64 is right there.

Predicted, from the measured floor:

| MPF_BM | floor | saving on a 1024-token prefill (776 ms) |
|---|--:|---|
| 64 (today) | 179 ms | — |
| **32** | **90 ms** | **-90 ms = -12%** |
| 16 | 45 ms | impossible: `SM = BM/WM/MFMA_M` < 1 at MFMA_M=32 |

So **32 is the only step down**, and it needs the wave grid re-derived — both are consistent:

```
BM=64  WMc=2 WNc=4  -> SM=1, SN=2, 8 waves     (today)
BM=32  WMc=1 WNc=8  -> SM=1, SN=1, 8 waves     (the change)
```

The body to template is **`d_moe_group_pf_a4w4`**, not `d_moe_group_pf_t`: K3 ships MXFP4 experts,
so `enc == PLOW_MOE_ENC_MXFP4` selects the A4W4 arm, and the object that serves it is
`interp_prefill_fp8kv_k3_moe_a4w4_gq.elf`. The align op pads to `MPF_BM` and both must move
together — the header says so: *"the align op and both GEMMs must agree."*

## 6. Ranked, all measured

1. **`MPF_BM` per bucket** — the 179 ms floor, worth ~12% of a 1024-token prefill and ~20% of a
   256-token one. Short prompts are the serving case.
2. **`BV=4` for KDA prefill** — 43.1% of long prefill running on 75% of the machine
   (`state_step_blocks` gives 192 of 256; BV=4 gives 384 capped to 256).
3. **`KDA_CONV3` straggler** — 48% of its own body at T=8192.
4. `FLASH_MLA_PREFILL_FP8` — O(T^2), only 5% at 8k, the 32k+ problem.

## 7. Reproduce

```bash
PLOW_TRACE_RAW=/tmp/tr.bin plowrt amd-bench --blob <b1>/model.pkt --hsaco <b1>/hsaco \
  --checkpoint <ckpt> --tp 8 --steps 1 --ctx <T> --prompt "<T comma-separated ids>"
python3 scripts/k3_trace_report.py /tmp/tr.bin.prefill --top 20
```

Note `.prefill` — the plain path holds the DECODE trace, because the decode dispatch overwrites the
buffer.


---

# 8. A 32K PREFILL BUCKET — built, measured, and NOT worth it

Asked for directly, on the reasoning that a wider bucket spreads better across 8 ranks and trades
against collective count. Built it, measured it, and the answer is no.

## 8.1 What it took to build

Three caps, all of which had to move together:

| cap | was | why it exists |
|---|--:|---|
| `MAX_CHUNK_MAX` (`devgen/lib.rs`) | 8192 | caps the emitted ladder |
| `MAX_CHUNK` (`plowrt/exec/amd.rs`) | 8192 | filters `plan_chunks`, so an emitted 32k rung is never selected |
| `PLOW_MAX_CHUNK` (`dev_isa.h`) | 8192 | asserted against `PLOW_KV_RING >= window + MAX_CHUNK - 1` |

`K3_PREFILL` already takes a list, so the ladder itself needed no code change. The binding
constraint is the last row: `PLOW_KV_RING` is **16384**, so a 16384 chunk needs the ring at 32768
and a 32768 chunk needs 65536.

**That ring is only for SLIDING layers** — *"a full-attention layer keeps a linear cache of `ctx`
rows and passes mask = 0xFFFFFFFF"*. K3 has none (MLA is full attention, KDA is recurrent), so K3
pays nothing for the ring; a WINDOWED model on the same build would pay 4x its ring VRAM. That
alone makes the global constant the wrong place for this.

## 8.2 The measurement

Emitted `[128, 512, 1024, 2048, 4096, 8192, 16384, 32768]` (9 programs, blob 211 -> 278 MB) and ran
a real 32,016-token prompt through `plowrt serve`, INTERLEAVED and order-reversed:

| arm | median | min | max | n |
|---|--:|--:|--:|--:|
| 8192 buckets (4 chunks) | **20.70 s** | 20.30 | 22.50 | 3 |
| 32768 bucket (1 chunk) | **21.65 s** | 21.50 | 21.80 | 2 |

**+4.6%, i.e. the 32k bucket is slightly SLOWER. VERDICT: NOT RESOLVED — the ranges overlap.**

There is no measurable benefit.

## 8.3 Two single-run results that were both wrong, in opposite directions

Recorded because this is the third time in this campaign a single run has produced the wrong SIGN:

* `amd-bench` at 16384: **1243 tok/s against 1739 at 8192** — apparently 47% worse than the linear
  fit, which would have said "bigger buckets are catastrophic".
* `serve`, one pass each: **24.1 s (4 chunks) vs 21.9 s (1 chunk)** — apparently 9% FASTER, which
  would have said "ship it".

The interleaved repeats say neither: +4.6%, unresolved. Any single-run number on this box is worth
nothing at this effect size (§8.6a of `k3-batched-decode-design.md`).

## 8.4 Why it does not help, mechanically

* **The fixed cost is already amortised at 8192.** It is 179 ms of a 4.7 s chunk — 3.9%. Going to
  one 32k chunk saves 3 x 179 = 537 ms of a ~21 s prefill, i.e. 2.5% GROSS, before any cost.
* **MoE tile fill saturates by 16384.** 57.1% at 4k, 76.2% at 8k, **91.4% at 16k, and 91.4% at
  32k** — the axis that made bigger chunks attractive stops paying before 32k.
* **Attention is unchanged.** Chunk `i` attends over `[0, c0+clen)`, so the causal triangle is the
  same total work whether it is one 32k chunk or four 8k ones. Chunking does not cost attention
  anything, so merging cannot save any.
* **The collective argument does not apply either.** Fewer chunks means fewer collective
  INVOCATIONS, but the rendezvous was MEASURED at 0.39% of a step (§10 of
  `k3-batched-decode-design.md`) — quartering it saves single-digit ms.

## 8.5 The cost, for completeness

* `act.pf.moe.part` alone goes **1.9 -> 7.6 GiB** (every activation is declared for the WIDEST
  bucket, so the whole prefill working set scales with the top rung, not with the prompt).
* Blob 211 -> 278 MB.
* `PLOW_KV_RING` 4x globally, which windowed models would pay for and K3 would not.

**Reverted.** The 8192 cap is well chosen and the chunk DP is already making the right decisions
(§2: 6144 planning as 4096+2048 measured 3791 ms against 4587 ms for one 8192 chunk).

The prefill lever is still §5's `MPF_BM` — the 179 ms floor is 23% of a 1024-token prefill and 41%
of a 256-token one, and short prompts are what serving actually sees.


---

# 9. MPF_BM: PREDICTED -12%, MEASURED +4.6%. The hypothesis was wrong.

§5 argued the 179 ms floor was padded-M — 896 experts x `MPF_BM=64` rows for a handful of real rows
each — and predicted that halving BM would halve the M-work and buy ~12% of a 1024-token prefill.

**Built it and measured it. It does not.**

## 9.1 The change

`MPF_BM` and `MPF4_BM` 64 -> 32, with the wave grid re-derived (`WMc` 2 -> 1, `WNc` 4 -> 8, keeping
`SM=1` and 8 waves). `pf_t` was split onto its own `MPFT_*` constants so it kept compiling at 64 —
its array initialiser is hardcoded for `SN=2` — which is sound because K3 (MXFP4) consumes the
align op's `meta` with the **A4W4** body, never `pf_t`.

## 9.2 The measurement — interleaved, order-reversed, T=1024

| arm | runs (ms) | min | median |
|---|---|--:|--:|
| BM=64 | 1608.9, 2147.9, **774.0** | **774.0** | 1608.9 |
| BM=32 | 809.5, 857.6, 810.6 | **809.5** | 810.6 |

BM=32 is strikingly CONSISTENT (809-858) while BM=64 carries two contention outliers. Comparing the
least-contaminated sample from each arm — the only fair estimator on a contended box —
**BM=64: 774 ms vs BM=32: 810 ms, i.e. BM=32 is 4.6% SLOWER.**

The median says "-49.6%", which is entirely BM=64's outliers, and the harness refused to call it.
**Do not read the median here.**

## 9.3 What the floor actually is

MEASURED at load: `routed experts packed ... gib=168.39` per rank across 92 layers = 1874 MiB per
layer per rank. At T=1024 with top_k=16 there are 16,384 routing slots over 896 experts, so
**every expert is hit and the WHOLE 168 GiB is streamed for one chunk.**

```
168.4 GiB at 6 TB/s  =   30 ms
MEASURED DOWN+GLU    =  179 ms      ->  5.9x off peak
```

`MPF_BM` changes the M dimension. It does not change one weight byte. That is why halving it did
nothing, and why the small regression is explained by the narrower `SN=1` wave grid rather than by
the tile size.

It also re-explains §4's T=256-vs-T=512 flatness: not padding, but the same 896 experts' weights
being streamed either way.

## 9.4 The lever this reveals instead

**The MoE floor is expert-weight bandwidth running at 5.9x off peak**, on 35.6% of prefill. If the
grouped GEMM reached even 50% of peak the floor would fall from 179 ms to ~60 ms. That is a much
larger prize than padding ever was, and the likely mechanism is the access pattern: 896 separate
~2 MB weight buffers per layer reached through `expert_weight_table` pointers, gathered per tile —
many small streams and a pointer chase, not one contiguous read.

## 9.5 Ranked, updated

1. **MoE expert-weight streaming efficiency** — 5.9x off peak on 35.6% of prefill. NEW, and the
   biggest headroom in the model.
2. **`BV=4` for KDA prefill** — 43.1% of an 8k prefill running on 192 of 256 workgroups.
3. **`KDA_CONV3` straggler** — 4384 us of a 9222 us body at T=8192.
4. **Close the tunedb for K3** — `0 HIT / 96 MISS`, so every prefill GEMM tile is chosen by the
   ANALYTICAL model. This is what makes any of the above durable rather than another constant.
5. `xr_cus` — collectives are 11.8% of prefill and K3 never reads `PLOW_XR_CUS`.

~~`MPF_BM`~~ — falsified here. Reverted; the tree carries no BM change.


---

# 10. LEVER 2 (`BV`) — structurally impossible, not merely unhelpful

§5 ranked `BV=4` second: `state_step_blocks = proj/bv = 1536/8 = 192`, so KDA prefill runs on 192
of 256 workgroups, and `BV=4` would give 384 items capped to 256.

**It cannot be done at the current workgroup shape.** `op_kda.h`:

```c
const unsigned cols_per_wave = BV / PLOW_WAVES;   /* 2 at BV=16, PLOW_WAVES=8 */
```

With 8 waves, `BV=4` gives `cols_per_wave = 0` — every wave would process zero value columns and
the kernel would compute nothing.

| BV | items | workgroups | cols_per_wave | |
|---|--:|--:|--:|---|
| 16 | 96 | 96 | 2 | |
| **8 (today)** | **192** | **192** | **1** | the legal minimum |
| 4 | 384 | 256 | **0** | INVALID |

Reaching 256 items needs `BV = H*D/256 = 6` — below the 8-wave floor and not a power of two. So
**192 of 256 is the structural maximum at TP8** with this item map (`items = H * D/BV`, and TP8
gives H=12).

More parallelism needs a different decomposition, not a knob: splitting the state's key dimension
would need a cross-workgroup reduction per token, and dropping to 4 waves is an interpreter-wide
change (the flash object already builds at `PLOW_WG_WAVES=4`, the prefill object at 8).

# 11. LEVER 4 (tunedb coverage) — CLOSED. Zero perf, and that is the finding.

`tune status` reported **0 HIT / 96 MISS**: every prefill GEMM tile K3 emits was chosen by the
ANALYTICAL model, with 1600 of 2600 records stale against the current build digest.

Ran the campaign — and note two things it needed that are not obvious:

* **`--obj` must be ABSOLUTE.** `argv[0]` is built as `obj/gemm_tile_sweep` and then
  `Command::current_dir(obj)` re-roots it, so a relative `--obj` looks for
  `<obj>/<obj>/gemm_tile_sweep` and dies with a bare `No such file or directory (os error 2)`.
* `--shapes auto` DOES reach K3 now. The flag's own doc says *"Kimi-K3 has no full-model emit, so
  its demand cannot be observed"* — stale; `K3_FULL=1` makes the demand observable and `auto`
  derived all 96 shapes.

Result:

```
published   : 480 qualified record(s) into tuning/amd/gfx950/mi350x
selectable  : 50 -> 146 op cases
coverage    : 0 HIT / 96 MISS  ->  96 HIT / 0 MISS
```

**The compiler now emits a different blob** — `8efdda25d70d721066dcc967391df245` against the
long-standing `7db2fbb34230050f0508a4e706523a98` — because it is choosing measured tiles rather
than modelled ones.

MEASURED, interleaved, T=1024:

| arm | median | min | max |
|---|--:|--:|--:|
| analytical tiles | 778.1 ms | **772.1** | 852.6 |
| MEASURED tiles | 801.8 ms | **773.6** | 841.5 |

**+0.2% on the least-contended sample. NOT RESOLVED.**

**The analytical model was already choosing near-optimally for K3's shapes.** That is worth knowing
rather than assuming, and it is why the campaign is still worth keeping: the value is DURABILITY,
not speed. The choice is now backed by measurement with an oracle-passed correctness gate, and a
future kernel change that shifts the optimum will be caught by a stale-digest report instead of
silently mis-picking.

# 12. Levers, after working four of them

| # | lever | share | outcome |
|---|---|--:|---|
| 1 | `MPF_BM` 64 -> 32 | 35.6% (MoE) | **FALSIFIED** — predicted -12%, measured +4.6%. The floor is expert-weight bandwidth, not padding. |
| 2 | `BV` 8 -> 4 | 27.6% (KDA) | **IMPOSSIBLE** — `cols_per_wave` would be 0. 192/256 is the structural max at TP8. |
| 4 | tunedb coverage | 19.9% (GEMM) | **CLOSED, 0% perf** — the analytical model was already right. Kept for durability. |
| — | 32k bucket | — | **REVERTED** (§8) — +4.6%, and 4x the activation memory. |

**Still open, and now the ranking is measured rather than guessed:**

1. **MoE expert-weight streaming at 5.9x off peak** on 35.6% of prefill (§9.4). The only lever left
   with a large, mechanically-understood headroom.
2. **`KDA_CONV3` straggler** — 4384 us of a 9222 us body at T=8192.
3. **`xr_cus`** — collectives are 11.8% of prefill and K3 hardcodes `(0..n_cu)`; GLM measured -5.3%
   from `PLOW_XR_CUS=32`, bit-identical.

Four levers worked, three of them negative. The negatives are the useful part: they moved the
target from padding and knobs to **expert-weight bandwidth**, which is where the time actually is.
