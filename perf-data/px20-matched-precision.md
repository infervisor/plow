# PX-20 — is the headline comparison apples-to-apples? Matched KV precision, both engines

RTX 5090 (sm_120a, 170 SM, 32 GiB / 30.86 GiB free at startup, driver 580.159.03) · 2026-07-26 ·
Gemma-4-12B-it, **fp8 weights on both engines in every row**. Every GPU run under
`perf-data/tools/gpulease`. Companion to `px12-consolidated-baseline.md` (the 127k cell),
`gemma4-12b-longctx-5090.md` §9a (the needle claim) and `px17-mixed-batching.md` (the prefill
patch-site fix this note depends on). Those documents are **not edited**; corrections are stated
here.

**The question.** `px12-consolidated-baseline.md` / `gemma4-12b-longctx-5090.md` §13 report plow
**29.91** out tok/s against vLLM **42.49** on 8 × 126,976-token requests, 1024 out, concurrency 8.
The offered load was verified identical, but the **KV cache was not**: vLLM ran e4m3 on all 48
layers, plow ran a *mixed* packet (e4m3 on the 8 hd512 full layers, bf16 on the 40 sliding rings).
So plow carried higher precision on 40 of 48 layers, and neither side's KV quality had been
validated against the other.

**TTFT and TPOT are NOT reported anywhere in this note.** `vllm bench serve --backend openai-chat`
stamps TTFT on the first SSE chunk carrying a `choices` array without checking `delta.content`, and
plowrt emits a role-only chunk before generation — so plow's whole prefill lands in its first ITL
sample while vLLM's does not. Aggregate output tok/s, benchmark duration and **median** ITL are the
only client metrics used.

---

## 0. Answers, up front

1. **The 67k needle is retrieved by every engine × KV-dtype cell.** vLLM fp8, vLLM bf16, plow
   all-layer e4m3, plow bf16 — four for four, same 66,901-token prompt. The quality confound the
   brief was written to settle **does not exist at this context**.
2. **`gemma4-12b-longctx-5090.md` §9a is WRONG, and the cause is a second copy of the PX-17 bug.**
   The miss it reports is an artifact of the PX-8 gate harness
   (`runtime/tests/gemma4_sm120_chat.cu`), which patches only the **bf16** prefill opcodes and
   therefore corrupts every fp8-KV prefill chunk after the first. PX-8's own raw timings carry the
   fingerprint. §2 proves it from the source and from PX-8's numbers.
3. **8 × 127k at bf16 KV does not fit on EITHER engine.** Measured: vLLM admits **4** of the 8,
   plow **6** (its packet ceiling). The matched-precision cell the brief asked for is not runnable
   at 127k on this card by either engine — §3.
4. **The headline was NOT apples-to-apples on KV dtype, and matching it does not give one number —
   it gives two, pointing opposite ways.** At matched **bf16** the gap collapses to **1.10×**
   (vLLM 25.01 vs plow 22.66 at 127k); at matched **all-layer e4m3** it explodes to **7.61×**
   (42.55 vs 5.59), because an all-layer fp8-KV packet has no fast prefill arm in plow at all.
   §4–§5.
5. **plow's reported 29.91 is not beaten by vLLM at matched precision at 127k** — vLLM's own bf16
   number is 25.01. The mixed packet is not a way of flattering plow; it is the only configuration
   in which plow's long-context prefill fast path is reachable at all.

---

## 1. The needle — four cells, one prompt

Same haystack construction as PX-8 (benign repetitive filler, needle at depth 0.50, the model's
chat template), rebuilt as an OpenAI chat body so the identical prompt goes to both engines
(`perf-data/px20_make_needle.py`). Both servers apply the Gemma-4 template themselves and **both
report `prompt_tokens = 66,901`** — the same count PX-8's raw file records, so the haystack is
byte-comparable with the run being re-tested. `temperature 0`, 96 max completion tokens.

| engine | KV dtype | weights | 66,901-token needle | answer | prefill wall |
|---|---|---|---|---|---|
| vLLM 0.26.0 | **e4m3, all 48 layers** (`--kv-cache-dtype fp8`) | fp8 | **RETRIEVED** | `PELICAN-7734` | 9.3 s |
| vLLM 0.26.0 | **bf16** (dtype auto) | fp8 | **RETRIEVED** | `PELICAN-7734` | 13.4 s |
| plow | **e4m3, all 48 layers** | fp8 (W8A8) | **RETRIEVED** | `PELICAN-7734` | 37.2 s |
| plow | **bf16** | fp8 (W8A8) | **RETRIEVED** | `PELICAN-7734` | 13.7 s |

Every cell answered with the bare code and stopped (8 completion tokens), so there is no
degenerate-reference problem of the kind PX-8's first attempt hit.

**Note the prefill walls.** plow's all-fp8 arm is the *slowest* of the four at 37.2 s against its
own bf16 arm's 13.7 s. An fp8-KV arm that were silently discarding all but the first prefill chunk
would be *faster*, not 2.7× slower — see §2.

### 1a. Multi-launch fp8-KV prefill correctness (a separate question)

PX-17 found that the per-chunk patch-site collector matched only the bf16 opcodes, so on an fp8-KV
packet every prefill launch after the first wrote its KV at row 0 with `q_pos0 = 0`. That fix had
been verified by code path and by a wall-time discrepancy, never by retrieval. At the emitted
`max_chunk` of 1024 a 7,826-token prompt is **8 prefill launches**, so it exercises the fixed path
hard:

| packet | prompt | prefill launches | result |
|---|---|---|---|
| all-fp8 KV | 7,826 | **8** | **RETRIEVED** — `PELICAN-7734`, 1.7 s |
| all-fp8 KV | 66,901 | **66** | **RETRIEVED** — `PELICAN-7734`, 37.2 s |
| mixed fp8 KV | 7,826 | 8 | **RETRIEVED** — 1.2 s |
| mixed fp8 KV | 66,901 | 66 | **RETRIEVED** — 13.0 s |

**Multi-launch fp8-KV prefill is correct on this tree**, on both fp8 KV layouts, at both the
single-digit-launch and 66-launch scales. Gate PASSED.

The packets' KV dtype is read from the emitted build manifest, **not** from a doc claim — that
mislabelling is what caused this whole exercise:

| packet | `shapes.kv_dtype` | KV / seq |
|---|---|---|
| `needle-fp8`, `cell127-fp8`, `cell16-fp8` | `{hd256: e4m3, hd512: e4m3}` | 1.333 GiB @132k |
| `needle-bf16`, `cell127-bf16-b6`, `cell16-bf16` | `{hd256: bf16, hd512: bf16}` | 2.640 GiB @132k |
| `needle-mixed` (cross-check only, §2) | `{hd256: bf16, hd512: e4m3}` | 1.641 GiB @132k |

The all-layer packet re-emitted here reports **KV 10.66 GiB at B=8**, identical to
`/root/plow-out/lc-b8`'s `kv_gib` 10.66455 — independently confirming PX-12 §0a's finding that
lc-b8 is an all-layer, not a mixed, packet.

---

## 2. §9a is an artifact of the PX-8 harness, not a property of e4m3

`gemma4-12b-longctx-5090.md` §9a concludes "bf16 KV retrieves; every fp8 KV arm misses … a
context-scaling property of the e4m3 cache, with the failure between 7.8k and 66.9k". That
conclusion does not survive.

**The mechanism, from the source.** PX-8's gates ran on the standalone
`runtime/tests/gemma4_sm120_chat.cu`, not on plowrt. Its per-chunk patch loop (lines 786–795, and
again at 878–887) is:

    if (in->op == PLOW_DOP_HEADNORM_ROPE && in->fj[1].u != 0)  in->i[3] = c0;
    else if (in->op == PLOW_DOP_FLASH_PREFILL) { in->i[1] = c0 + real; in->i[4] = c0; }

An fp8-KV packet does not emit those opcodes. It emits the twins —
`PLOW_DOP_HEADNORM_ROPE_FP8 = 37` and `PLOW_DOP_FLASH_PREFILL_FP8 = 39`
(`runtime/common/dev_isa.h:299,301`) — which carry the same operands at the same indices and are
**never matched**. So on every fp8-KV prompt long enough to need a second prefill launch, each
later chunk writes its KV at row 0 and reads with `q_pos0 = 0`. This is exactly the bug PX-17 found
and fixed in `crates/plowrt/src/exec/gpu.rs`; the copy in the test harness was never fixed and is
still there.

**The fingerprint is in PX-8's own raw data.** PX-17 records that the bug makes a prefill *look*
faster, because the flash never grows past the first chunk's keys. `px8-gates-raw.txt`:

| prompt | bf16 ref | fp8 (px4) | fp8 vs bf16 | PX-8 verdict |
|---|---|---|---|---|
| 7,826 tokens | 1315.6 ms | 1670.3 ms | **1.27× SLOWER** | all arms RETRIEVE |
| 66,901 tokens | 15,643.5 ms | 10,947.3 ms | **1.43× FASTER** | every fp8 arm MISSES |

The same fp8 arm is 1.27× slower than bf16 on the short prompt and 1.43× faster on the long one — a
1.8× swing in the same direction as the retrieval failure. A dequant arm does not become 1.8× more
efficient with context; a flash that stops growing past the first chunk does exactly that.

*(Inference, stated as such: PX-8's load line prints `7 programs` against this note's 4, so its
packet carried 6 prefill buckets rather than 3 and a top bucket well above 1024. A 7,826-token
prompt on that ladder is one or two launches, which is why its "7.8k control" saw nothing. I could
not read the bucket ladder out of that v5 packet to confirm the exact count, and **nothing below
depends on it** — the direct re-runs are the evidence.)*

**Cross-check.** PX-8's exact KV layout — the mixed packet, e4m3 on the 8 hd512 full layers and
bf16 on the 40 sliding rings, on the FASTPF (PIPE=1 px4 fp8-mma) prefill arm PX-8's `px4` column
used — re-run through the fixed plowrt path on the same 66,901-token needle:

| | PX-8 (`gemma4_sm120_chat`, 2026-07-26 morning) | PX-20 (plowrt, fixed patch loop) |
|---|---|---|
| 66,901-token needle, mixed fp8 KV | ***** MISS *** | **RETRIEVED** — `PELICAN-7734` |
| 7,826-token needle, mixed fp8 KV | RETRIEVED | RETRIEVED |

Same packet layout, same needle, same GPU, same day. The only thing that changed is which process
patched the per-chunk prefill sites. **That is the whole of §9a.**

**Correction to `gemma4-12b-longctx-5090.md` §9a:** the sentence "the capacity win is partly bought
with retrieval quality" is withdrawn. It was bought with nothing; the arm that appeared to lose the
needle was a harness that had silently discarded all but the first prefill chunk.

---

## 3. VRAM — why the brief's primary ask cannot be run at 127k

**vLLM**, from its own startup accounting (`--gpu-memory-utilization 0.90`, fp8 weights):
30.86 GiB free → 28.22 GiB budget → 12.89 GiB weights + 0.94 peak activation + 0.16 non-torch +
0.54 CUDAGraph, leaving **14.23 GiB of KV cache** in every configuration below.

| vLLM KV dtype | `--max-model-len` | GPU KV cache | vLLM's own "maximum concurrency" | measured resident |
|---|---|---|---|---|
| e4m3 | 70,000 | 312,018 tok | 4.46× | — (needle, conc 1) |
| bf16 | 70,000 | 156,003 tok | 2.23× | — (needle, conc 1) |
| e4m3 | 18,432 | 93,709 tok | 5.08× | **8, 0 preemptions** |
| bf16 | 18,432 | 46,853 tok | 2.54× | **8, 0 preemptions** |
| e4m3 | 131,200 | 510,778 tok | 3.89× | **8, 0 preemptions** |
| bf16 | 131,200 | 255,100 tok | 1.94× | **4, 0 preemptions** |

The last two rows are the load-bearing ones: **turning vLLM's KV cache from e4m3 to bf16 halves its
residency at 127k, 8 sequences → 4**, with no preemption in either case (it simply admits fewer).

**vLLM's "maximum concurrency" line understates this model badly** — it divides total blocks by
`max_model_len` uniformly, while the 40 sliding layers only ever hold ~1 window of blocks. At
18,432 it predicts 2.54 bf16 sequences and the engine then held **8 with zero preemptions**. It is
reported here because it is the engine's own accounting, not because it is the residency; every
residency figure in this note is measured.

**plow**, from the emitted packets (weights 12.04 GiB + activations ~0.30 GiB). plow rings the 40
sliding layers at `next_pow2(window + chunk − 1)` = 2048 rows, so only the 8 full layers grow with
context: KV/seq = 0.625 GiB + ctx × 16,384 B at bf16, 0.3125 GiB + ctx × 8,192 B at e4m3.

plowrt's planner reports ~0.41 GiB of measured load overhead beyond the packet's plan, so the
budget for KV is 30.86 − 12.04 − 0.30 − 0.41 = **18.11 GiB**.

| plow KV dtype | ctx | KV/seq | total at B | fits 30.86 GiB? |
|---|---|---|---|---|
| e4m3 all-layer | 132,096 | 1.333 GiB | B=8 → 10.66 + 12.75 = **23.41 GiB** | **yes** |
| bf16 | 132,096 | 2.640 GiB | B=8 → 21.12 + 12.75 = **33.87 GiB** | **no** |
| bf16 | 132,096 | 2.640 GiB | B=7 → 18.48 + 12.75 = **31.23 GiB** | **no** |
| bf16 | 132,096 | 2.640 GiB | B=6 → 15.84 + 12.75 = **28.59 GiB** | **yes** — the row run below |

So **neither engine can serve 8 × 127k at bf16 KV**: plow's ceiling there is **6** sequences, not
8. Turning the arithmetic around (`ctx = (KV/seq − 0.625 GiB) × 65,536`), the longest context at
which plow holds **8 resident** at bf16 is **≈ 107,000 tokens**. The brief's suggested fallbacks
(64k, then 32k) are comfortably inside plow's bf16 8-resident range. For vLLM the picture is only
partly known: its own estimator says ~19,500 tokens, but it then held 8 at 17,408 where the
estimator predicted 2.54, so the estimator is not the ceiling. **16,384 was chosen as a context
both engines would certainly hold 8 at, and both did (0 preemptions); the true maximum was not
searched for** — that would have cost cells this note spent on the conc-1 sweep instead. Row B is
therefore a valid matched-residency row, not necessarily the longest one.

---

## 4. The matrix

Same client, same tokenizer, same offered load on every row: `vllm bench serve --backend
openai-chat --dataset-name random --random-range-ratio 0 --num-prompts 8 --max-concurrency 8
--ignore-eos --seed 0`. plow runs with `PLOW_PF_DEFER_DECODE` **OFF**; no throughput-mode row is
reported. plow's decode object is the **deployed** `-DPLOW_NV_FA_GF_FULL=4` (see §4f).

### Row A — the campaign's cell: 8 × 126,976 in / 1024 out, concurrency 8

Every row reports `Total input tokens = 1,015,914` and `Total generated tokens = 8,192` — the same
counts PX-12 §3 reports, so this is like-for-like with the recorded cell.

| engine | KV dtype | out tok/s | wall (s) | median ITL (ms) | resident seqs |
|---|---|---|---|---|---|
| **vLLM 0.26.0** | **e4m3, all layers** | **42.55** | 192.53 | 18.83 | **8**, 0 preemptions |
| **vLLM 0.26.0** | **bf16** | **25.01** | 327.53 | 19.83 | **4**, 0 preemptions |
| **plow** | **e4m3, all layers** | **5.59** | 1465.75 | 38.76 | **8** (packet B=8) |
| **plow** | **bf16** | **22.66** | 361.49 | 49.20 | **6** (packet B=6, its ceiling) |
| *plow, MIXED fp8 KV — px12 arm E, for reference only* | *e4m3 on 8 of 48 layers* | *29.91* | *273.9* | *40.39* | *8* |

**The vLLM column reproduces the campaign to 0.15%** — 42.55 / 192.53 s / 18.83 ms against PX-12's
recorded 42.49 / 192.8 / 18.87. Nothing about the machine or the harness moved between sessions,
so every delta below is the configuration.

### Row B — matched residency: 8 × 16,384 in / 1024 out, concurrency 8

16,384 is the longest context tried at which **both engines hold all 8 sequences resident in both
KV dtypes** (measured, 0 preemptions everywhere). Every row reports `Total input tokens = 131,178`
and `Total generated tokens = 8,192`.

| engine | KV dtype | out tok/s | wall (s) | median ITL (ms) | resident seqs |
|---|---|---|---|---|---|
| **vLLM 0.26.0** | e4m3 | **324.08** | 25.28 | 13.95 | 8, 0 preemptions |
| **vLLM 0.26.0** | bf16 | **280.76** | 29.18 | 15.09 | 8, 0 preemptions |
| **plow** | e4m3 | **146.94** | 55.75 | 21.83 | 8 |
| **plow** | bf16 | **189.38** | 43.26 | 22.47 | 8 |

### 4a. The gap, per matched cell

| context | at matched **bf16** KV | at matched **all-fp8** KV |
|---|---|---|
| 126,976 | **vLLM 1.10× ahead** (25.01 vs 22.66) | **vLLM 7.61× ahead** (42.55 vs 5.59) |
| 16,384 | **vLLM 1.48× ahead** (280.76 vs 189.38) | **vLLM 2.21× ahead** (324.08 vs 146.94) |
| *the unmatched headline* | — | *vLLM 1.42× ahead (42.49 vs plow's MIXED 29.91)* |

At **concurrency 1** (Row C), the same two columns read very differently:

| context, conc 1 | at matched **bf16** KV | at matched **all-fp8** KV |
|---|---|---|
| 8,192 | **plow 1.07× ahead** | tie (77.94 vs 77.80) |
| 32,768 | **plow 1.09× ahead** | vLLM 1.42× ahead |
| 126,976 | **plow 1.13× ahead** | vLLM 5.06× ahead |

### 4b. Why the two matched columns disagree so violently

Both engines lose throughput when KV goes from e4m3 to bf16, for different reasons, and plow gains:

* **vLLM at 127k: −41%** (42.55 → 25.01). Pure capacity — bf16 halves its KV cache, residency goes
  8 → 4, and a prefill-dominated cell with half the parallelism takes 1.70× the wall. Its median
  ITL barely moves (18.83 → 19.83), which is the signature of a residency loss rather than a
  per-token one.
* **plow at 16k: +29% going TO bf16** (146.94 → 189.38). plow's all-layer e4m3 packet emits hd256
  `FLASH_PREFILL_FP8`, and the PIPE=1 px4 fp8-mma arm `__trap()`s on hd256
  (`interp_sm120.cu:846`), so **an all-layer fp8-KV packet has no fast prefill arm at all** — it is
  forced onto the synchronous PIPE=0 staging path. The two arms' median ITL is within 3%
  (21.83 vs 22.47), so the entire 12.5 s difference in wall is prefill.

So "match the KV dtype" is not one experiment. Matching **down** to bf16 costs vLLM its capacity
advantage and costs plow nothing it had; matching **up** to all-layer e4m3 costs plow its entire
prefill fast path and costs vLLM nothing. Neither engine's best long-context configuration is the
matched one.

### Row C — concurrency 1, context sweep: the scheduler removed as a variable

`--num-prompts 4 --max-concurrency 1`, 1024 output tokens, one server per (engine, KV dtype) so
all three contexts run against one load. At conc 1 there is no mux tick, no batch formation and no
prefill/decode interleaving on either side, so everything that differs here is kernels.

**all-layer e4m3 KV**

| ctx | vLLM tok/s | vLLM wall (s) | vLLM ITL (ms) | plow tok/s | plow wall (s) | plow ITL (ms) |
|---|---|---|---|---|---|---|
| 8,192 | 77.80 | 52.65 | 12.21 | **77.94** | 52.56 | **11.21** |
| 32,768 | **62.23** | 65.82 | 13.62 | 43.93 | 93.24 | **11.73** |
| 126,976 | **27.23** | 150.41 | 16.07 | 5.38 | 761.64 | **13.69** |

**all-bf16 KV**

| ctx | vLLM tok/s | vLLM wall (s) | vLLM ITL (ms) | plow tok/s | plow wall (s) | plow ITL (ms) |
|---|---|---|---|---|---|---|
| 8,192 | 75.60 | 54.18 | 12.47 | **81.06** | 50.53 | **11.21** |
| 32,768 | 53.88 | 76.03 | 14.65 | **58.74** | 69.73 | **11.80** |
| 126,976 | 18.45 | 222.06 | 19.26 | **20.80** | 196.90 | **13.98** |

Every row reports matching token counts (32,821 / 131,125 / 507,957 in, 4,096 out) on both engines.

**At matched bf16 KV and concurrency 1, plow wins every context** — 1.07× at 8k, 1.09× at 32k,
1.13× at 127k. And **plow's median ITL is lower in all six cells of both tables**, by 1.09× to
1.38×. The conc-1 decode anchor holds: vLLM's 127k ITL comes out at **16.07 ms**, the value already
on record.

### 4d. The prefill/decode split the conc-1 rows imply

Per request, `prefill ≈ wall/4 − 1023 × median ITL`. This is arithmetic on the two metrics that are
valid for both engines, not a measurement, and it does not use TTFT.

| arm, ctx 126,976 | implied prefill (s) | decode (1023 × ITL, s) |
|---|---|---|
| vLLM, e4m3 | **21.2** | 16.4 |
| vLLM, bf16 | **35.8** | 19.7 |
| plow, bf16 | **34.9** | 14.3 |
| plow, e4m3 | **176.4** | 14.0 |

Two things fall out, and both correct earlier campaign statements:

* **plow's 127k bf16 prefill (34.9 s) is within 3% of vLLM's 127k bf16 prefill (35.8 s).** It
  reproduces the campaign's own conc-1 anchor for that arm (33.09 s, `gemma4-12b-longctx-5090.md`
  §2) to 5%. `gemma4-12b-longctx-5090.md` §2's "prefill is 2.2–2.6× behind" compared **plow's
  bf16-KV prefill against vLLM's fp8-KV prefill** — vLLM's own prefill slows by **1.69×** (21.2 →
  35.8 s) when its KV cache goes to bf16. At matched KV dtype at 127k, prefill is a tie.
* **plow's all-layer e4m3 prefill is 176.4 s**, reproducing PX-12 §2's 177.50 s for the same PIPE=0
  arm on `/root/plow-out/lc-b8` to 0.6%. That single number is the whole of the 7.61× conc-8 gap in
  the e4m3 column.

### 4e. conc-1 vs conc-8 at the same context — sizing the runtime overhead

Same context (126,976), same KV dtype, aggregate out tok/s. plow's conc-8 rows carry 8 sequences
against conc-1's 1, so a perfectly scaling runtime would multiply.

| arm | conc 1 | conc 8 | scaling | resident at conc 8 |
|---|---|---|---|---|
| vLLM, e4m3 | 27.23 | 42.55 | **1.56×** | 8 |
| vLLM, bf16 | 18.45 | 25.01 | **1.36×** | 4 |
| plow, bf16 | 20.80 | 22.66 | **1.09×** | 6 |
| plow, e4m3 | 5.38 | 5.59 | **1.04×** | 8 |

plow converts almost none of its concurrency into throughput at 127k: **1.09× from 6 resident
sequences**, against vLLM's 1.36× from 4. Its median ITL over the same step is 13.98 → 49.20 ms
(3.52×) while vLLM's is 19.26 → 19.83 ms (1.03×). vLLM's decode step is essentially free per extra
sequence; plow's is not. **That is the deficit the matched-precision comparison actually exposes**,
and it is a batched-decode/scheduling property, not a KV-precision one — consistent with PX-17's
finding of host-side per-tick turnaround that device-side instrumentation cannot see. It is not
sized further here; conc-1 vs conc-8 at one context is the measurement, not the attribution.

### 4f. `GF_FULL` — checked, and it does not move this comparison

The campaign tip marks `PLOW_NV_FA_GF_FULL=8` as **CONTESTED** (PX-11 measured 1.52× on the flash
op; PX-15 measured `=4` winning every end-to-end cell at ctx ≥ 8k). Every plow row above therefore
runs the **deployed `=4`**. The A/B, same packet, same everything else:

| 16k bf16 cell | out tok/s | wall (s) | median ITL (ms) |
|---|---|---|---|
| `GF_FULL=8` | 188.65 | 43.42 | 22.28 |
| `GF_FULL=4` (deployed, reported) | **189.38** | 43.26 | 22.47 |

**0.4% — noise.** Same at 16k fp8 (147.29 at `=8` vs 146.94 at `=4`). The knob is real elsewhere;
it is not what decides any row in this note.

One thing the campaign tip asserts that could NOT be applied here: `PLOW_FA_GF_FULL`, the emitter
half of the pair, aborts on a whole-model Gemma-4 packet — it is applied to every layer and the 40
sliding layers have `gqa = 16/8 = 2`, so `GF 4 must divide GQA 2` panics
(`devgen/src/lib.rs:1702`). Its grid-aligned `nsplit` rule is gated on `kvh_full >= 4` and this
model's full layers have `kvh_full = 1`, so it would not have fired regardless. Both arms are
emitted at the emitter's constant GF=2 with `PLOW_NS_FULL_ABS=32` pinning `nsplit`, exactly as the
campaign did. The prefill objects are **md5-identical** across the `=4` and `=8` builds
(`797ac464…` bf16, `4928b8ce…` fp8), which is the proof that `GF_FULL` reaches only decode.

---

## 5. Verdict

**Was the headline apples-to-apples? No — the KV dtypes differed.** But the confound it was
suspected of hiding is not there, and the correction does not run the way the brief anticipated.

**1. The quality confound does not exist.** All four engine × KV-dtype cells retrieve the 66,901-
token needle, and both fp8 arms also retrieve it at 7,826. `gemma4-12b-longctx-5090.md` §9a's
"fp8 KV loses the 67k needle" is an artifact of the PX-8 gate harness, which still contains the
PX-17 patch-site bug (§2). vLLM's fp8 KV retrieves; so does plow's. **Nobody's capacity is being
bought with retrieval quality**, on either engine.

**2. Matching the precision does not produce one gap. It produces two, and they point opposite
ways**, because each engine's preferred configuration is the other's worst:

| cell (8 × 126,976 in / 1024 out, conc 8) | vLLM | plow | gap |
|---|---|---|---|
| matched **all-layer e4m3** | 42.55 | 5.59 | **vLLM 7.61× ahead** |
| matched **bf16** | 25.01 | 22.66 | **vLLM 1.10× ahead** |
| *the unmatched headline (plow MIXED)* | *42.49* | *29.91* | *vLLM 1.42× ahead* |

**Is plow's deficit larger than we thought? At all-fp8 KV, catastrophically yes — 7.61×, not
1.42×.** And the reason is structural, not a tuning gap: an all-layer e4m3 packet emits hd256
`FLASH_PREFILL_FP8`, the PIPE=1 px4 fp8-mma arm `__trap()`s on hd256, so plow falls to the
synchronous PIPE=0 staging path at **176 s of prefill per 127k request** against the mixed
packet's 32 s. plow does not have a fast prefill arm for the configuration vLLM ships by default.
That is the single most actionable finding in this note.

**At bf16 KV, no — the deficit is smaller than we thought, and at concurrency 1 it is not a
deficit at all.** plow wins all three conc-1 bf16 contexts (1.07× / 1.09× / 1.13×) and has the
lower median ITL in **all six** conc-1 cells. Two campaign claims fall out of that:

* `gemma4-12b-longctx-5090.md` §2's "**prefill is 2.2–2.6× behind**" compared plow's bf16-KV
  prefill against vLLM's **fp8-KV** prefill. vLLM's own prefill slows 1.69× when its cache goes to
  bf16. At matched dtype and 127k, prefill is **34.9 s vs 35.8 s — a tie** (§4d).
* What is left is **batched decode**. From conc 1 to conc 8 at 127k bf16, vLLM's median ITL moves
  1.03× while plow's moves **3.52×**; plow converts 6 resident sequences into 1.09× of throughput
  where vLLM converts 4 into 1.36×. The gap that survives matched precision is a
  scheduling/batched-decode gap, and it is the same one PX-10, PX-11 and PX-17 have been circling.

**3. The mixed packet was never the thumb on the scale.** It is the only plow configuration in
which the fast prefill arm and a B=8-sized KV cache coexist. plow's 29.91 there is *higher* than
vLLM's own matched-bf16 25.01. If a single headline is wanted, the honest framings are either
"each engine at its best long-context configuration: 42.49 vs 29.91, vLLM 1.42× ahead" or
"matched bf16 KV: 25.01 vs 22.66, vLLM 1.10× ahead" — and both should carry the residency
(vLLM 4, plow 6 or 8) and the fact that neither engine fits 8 × 127k at bf16.

**What NOT to conclude.** Nothing here says plow's e4m3 decode is bad — it is the fastest decode in
the whole note (13.69 ms at 127k conc 1, against vLLM's 16.07). The e4m3 column loses on one
kernel arm that does not exist yet.

---

## 6. Gates

| gate | result |
|---|---|
| same offered load on every row | **PASS** — every 16k row reports `Total input tokens` 131,178 / `Total generated` 8,192; every 127k row reports 1,015,914 / 8,192. Verified per row, not assumed |
| fp8 weights held constant across all 8 rows | **PASS** — vLLM `--quantization fp8` everywhere; plow `PLOW_FP8=1 PLOW_W8A8=1` at emit and `-DPLOW_NV_W8A8=1` in every cubin |
| KV dtype is the ONLY packet variable | **PASS** — read from the emitted `build.json`: `{hd256: e4m3, hd512: e4m3}` vs `{hd256: bf16, hd512: bf16}`, everything else in `shapes`/`tuning` identical (chunk 1024, buckets [128,512,1024], `gf_full` 8) |
| KV dtype is the ONLY cubin variable | **PASS** — both sets come from one CMake base + two axes; the fp8 set adds exactly `-DPLOW_FP8_KV=1` (decode) and `-DPLOW_FP8_KV=1 -DPLOW_NV_FA_PIPE=0` (prefill) |
| plow's all-layer packet is really all-layer | **PASS** — manifest says e4m3 on both hd classes, and the re-emitted packet's `kv_gib` at B=8 is **10.66**, matching `/root/plow-out/lc-b8`'s 10.66455 exactly (PX-12 §0a confirmed independently) |
| needle prompt identical across engines | **PASS** — both engines report `prompt_tokens = 66,901`, the same count `px8-gates-raw.txt` records |
| needle has a working (non-degenerate) reference | **PASS** — both bf16 arms answer with the bare code and stop at 8 tokens |
| **8 × 127k at bf16 KV on both engines** | **FAIL — and that is the finding.** plow's bf16 ceiling at 132k ctx is **B=6**; vLLM admitted **4** of the 8. The brief's primary ask is not runnable on this card by either engine (§3) |
| matched residency at the fallback context | **PASS** — at 16,384 both engines held **8 resident with 0 preemptions**, in both KV dtypes |
| **multi-launch fp8-KV prefill correctness, 7.8k** | **PASS** — 7,826-token needle, all-layer e4m3 packet, `max_chunk` 1024 = 8 prefill launches: **RETRIEVED**. Also RETRIEVED at 66,901 (66 launches) and on the mixed packet at both sizes. PX-17's patch-site fix is now verified by retrieval, not only by code path (§1a) |
| conc-1 anchors reproduce | **PASS** — vLLM 127k conc-1 median ITL **16.07 ms** against the recorded 16.07; plow 127k bf16 implied prefill **34.9 s** against the recorded 33.09 s (5%); plow all-layer e4m3 implied prefill **176.4 s** against PX-12's 177.50 s (0.6%) |
| vLLM column reproduces the campaign | **PASS** — 42.55 tok/s / 192.53 s / 18.83 ms against PX-12's 42.49 / 192.8 / 18.87 (0.15%) |
| conc-1 rows have matched offered load | **PASS** — 32,821 / 131,125 / 507,957 input and 4,096 output tokens on both engines at each context |
| plow all-layer fp8-KV on the FASTPF prefill arm | **STRUCTURALLY IMPOSSIBLE** — an all-layer packet emits hd256 `FLASH_PREFILL_FP8`, and the PIPE=1 px4 fp8-mma arm `__trap()`s on it (`interp_sm120.cu:846`). plow's fp8-KV row is PIPE=0 by construction, and that is most of its fp8 deficit (§4) |
| `PLOW_PF_DEFER_DECODE` off in every reported row | **PASS** — never set by `px20_cell.sh`; no throughput-mode row is reported |
| TTFT / TPOT | **NOT REPORTED, deliberately** — invalid for plow (role-only SSE chunk poisons the first ITL sample) |
| GPU exclusive | **ENFORCED** — 17 `perf-data/tools/gpulease` leases, **all rc=0**, zero `WARN foreign` lines for any `px20-*` label |
| sanity vs physical bounds | **PASS** — the fastest decode step in the note (plow, 127k bf16, conc 1, 13.98 ms) implies 2.64 GiB of KV read per step = **202 GB/s**; the busiest (plow, 127k bf16, conc 8, 6 × 2.64 GiB / 49.20 ms) = **345 GB/s**. Both far under the 1792 GB/s ceiling, as expected for a latency-bound decode — no row needs a bandwidth denominator to be believed |
| greedy parity between the two plow KV arms | **NOT RUN, and not meaningful** — changing the KV cache dtype changes the numerics by construction; there is no bit-exactness to gate |
| plow arm tuned per KV dtype | **NOT RUN** — both arms use `GF_FULL=8` and stock `GV_MM_MAX`. `FP8PV` (the largest lever in PX-12/13) is unreachable here because it needs FASTPF, which needs a mixed packet |
| longest matched-residency context located | **NOT RUN** — 16,384 was chosen as safely inside both engines' 8-resident range and both held 8; 32k bf16 on vLLM was not tried, so row B is a valid matched row but not provably the longest one (§3) |
| attribution of plow's 3.52× conc-1 → conc-8 ITL growth | **NOT RUN** — §4e sizes it, it does not explain it. Needs the per-tick split PX-17's `PLOW_PF_PACKLOG` provides |
| a fast prefill arm for all-layer fp8-KV packets | **DOES NOT EXIST** — the actionable finding of this note (§5) |

### Bugs found

1. **`runtime/tests/gemma4_sm120_chat.cu` corrupts every multi-chunk fp8-KV prefill** (§2). Its
   per-chunk patch loop matches `PLOW_DOP_HEADNORM_ROPE` / `PLOW_DOP_FLASH_PREFILL` but not the
   `_FP8` twins (opcodes 37/39), so each chunk after the first writes KV at row 0 and reads with
   `q_pos0 = 0`. This is the PX-17 bug in a second location, it is **still present**, and it is
   what produced `gemma4-12b-longctx-5090.md` §9a's retraction-worthy conclusion. Fix: add
   `|| in->op == PLOW_DOP_HEADNORM_ROPE_FP8` and `|| in->op == PLOW_DOP_FLASH_PREFILL_FP8` to the
   two loops (lines 788/790 and 880/882). **Not fixed here** — this is a measurement task, and
   changing the harness would invalidate the binaries PX-8's numbers came from.
2. **The completion-tokens branch does not compile with `--features cuda`.**
   `crates/plowrt/src/exec/gpu.rs` carries committed conflict markers (`<<<<<<< HEAD` … at line
   2506) from an earlier merge, inside a comment block. Only the `cuda` feature
   compiles that path, so a default `cargo build` passes and the served binary does not exist.
   Resolved in this worktree (comment text merged, no semantic change).

---

## 7. Reproduce

    W=$PWD
    perf-data/px20_build.sh                      # plowrt (--features cuda) + plowc
    perf-data/px20_build_cubins.sh               # the two served cubin sets
    perf-data/px20_emit.sh <name> <ctx> <B> <bf16|fp8kv>
    perf-data/px20_make_needle.py --tokens 66901 --out /tmp/px20/needle67k.json
    perf-data/tools/gpulease px20-vllm-needle perf-data/px20_vllm_needle.sh 70000
    perf-data/tools/gpulease px20-plow-needle perf-data/px20_plow_needle.sh
    perf-data/px20_queue.sh                      # conc-8 matrix, one lease per cell
    perf-data/px20_retool_gf4.sh                 # deployed GF_FULL=4 cubins
    perf-data/px20_mk_gf4_assets.sh              # same packets, GF_FULL=4 decode object
    perf-data/px20_queue2.sh                     # the plow conc-8 rows, GF_FULL=4
    perf-data/px20_gates2.sh                     # GF A/B + 7.8k multi-launch gate + mixed cross-check
    perf-data/px20_emit_c1.sh                    # B=1 full-context packets
    perf-data/px20_queue3.sh                     # the concurrency-1 context sweep

Raw logs: `/tmp/px20/{vllm,plow,cells,g2,c1}` (bench + serve logs per cell). Assets:
`/root/px20/{cubins,cubins-gf4,pkt,pkt-gf4,pkt-c1}`.
