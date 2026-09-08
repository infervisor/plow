# The unified token batch, serving: gfx942 / MI300X, Gemma-4 31B

Status 2026-09-08. **The route serves.** It produces tokens end to end on a real checkpoint,
its packed output is character-identical to the same prompts run one at a time, and it has been
measured against mixed step v1 (`--fusion`) and against neither, on one blob, with the arms
interleaved.

This is the serving half of `plans/unified-token-batch.md`. The contract, the ABI, the opcode
and the operator classification are [17-unified-token-batch.md](../arch/17-unified-token-batch.md);
the AMD device arms are [17-unified-token-batch-dense-gqa.md](../arch/17-unified-token-batch-dense-gqa.md);
the windowed/recurrent axes are [token-batch-recurrent-and-window.md](token-batch-recurrent-and-window.md).
Those three describe a route that was **armed and could not fire**. This one describes what it
took to make it fire, and what it is worth.

---

## 1. Which model, and which phase

**Gemma-4 31B, BF16, dense GQA with per-layer sliding-window attention (1024 on 50 of its 60
layers) and a logit softcap of 30.** That makes this a **Phase 3 enablement — the plan's
"windowed attention + softcap tail" item — not a Phase 2 one.** The distinction is the plan's
own (§8), and it is worth stating precisely rather than claiming the cheaper phase:

| Phase 2 asks for | This model |
|---|---|
| dense causal GQA, one class-C conversion | dense GQA, **windowed** on 50 of 60 layers |
| no softcap | `SoftCap` (cap 30) in the tail |
| Llama 3 / Qwen 3 | Gemma-4 |

Phase 2's own record says a dense-GQA checkpoint does not exist on this host and that Gemma "was
**not** substituted — it is Phase 3 by the plan's own table". That was right when it was written.
What changed is that the landed arms turn out to cover the Phase 3 windowed axis already:

* `d_flash_prefill`'s `TB` arm rebinds `q_pos0` per span and derives the window bound from the
  rebound base at all four sites that can disagree — the workgroup-uniform `win_lo`, the
  `kv_lo = (win_lo/BKV)*BKV` tile carve, the `kv0 + BKV <= win_lo` tile skip and the per-element
  `(qg - kg) < window` mask. The KV ring is preserved: K/V loads use `(kv & kv_mask)`.
* `d_flash_decode`'s `TB` arm takes `len` from `spans[b].kv_len`, so `first = len - window` is
  per request.
* `d_headnorm_rope` writes at `(position & kv_mask)` with `position` from the descriptor, so the
  ring wrap on a windowed layer is the row's own.
* `SoftCap` is dispatched under `PLOW_RUNTIME_ROWS`, which on this axis is the descriptor's live
  row count.
* Tied and untied heads are a tensor-handle choice with no second code path.

So the answer to "do the landed arms cover Gemma" is **yes, for the body**. What they do *not*
cover is the other half of that Phase 3 bullet: **"SoftCap stage in the terminal segment"**.
There is no terminal segment on this route (§4 below). The `SoftCap` packet is correct over a
runtime row count, and this route samples rows of the body rather than of a compact tail, so the
softcap it applies is the body's — right answer, different mechanism from the one the plan
describes.

**Coverage note.** `runtime/tests/token_batch_dense_gfx942_test.hip` — the 31 Phase-2 gates —
runs at `HD=128`, `window = 0` and `PLOW_KV_MASK_NONE`. The window and ring code in the `TB` path
was correct by construction and **had no test**. This model exercises both on every step: 50
windowed layers at `hd=256` with a 16384-entry KV ring. The identity result in §3 is that
coverage.

---

## 2. What it took

Three things stood between an armed route and a token, and none of them was the device.

### 2.1 The span table had to cover `[0, M)`

`plow_asset::mixed_step::plan_into` put decode rows in a **band** ahead of the spans.
`runtime/common/mixed_step.h` requires that band (it traps when `spans[0].row0 == 0`);
`runtime/amd/token_batch.h` requires its absence. Both are right; they are different contracts
over one array, which is exactly the drift §4.3 exists to stop.

The planner now takes an explicit `SpanCover`. `DecodeBand` is v1's shape, kept verbatim because
v1 is the only route that served before this and the comparison depends on it. `PrefixFree` is
§4.3's: every row belongs to a span, decode spans included, tiling `[0, M)` from zero.

### 2.2 A completing prompt had to be able to sample without a second pass

This is the part that is not bookkeeping. Under `PrefixFree`, a prefill span that **completes its
prompt** is split: its final token leads the batch as a length-one span, its body follows as an
ordinary span, **both on the same physical slot in the same step**.

That is legal because of the order inside one launch. The step's single RoPE/cache stage
(`HeadNormRope`) writes every row's K/V — body rows at positions `f..p-1`, the terminal row at
`p` — before any attention instruction runs, because the synthesized program is a strict linear
chain. The terminal row then attends over `[0, p+1)`: its own complete prompt, written by the
same launch that is reading it.

The leading run of length-one spans is what `plow_tb_decode_spans` reads as the attention
partition and what `PLOW_SAMPLE_ROWS` reads as the selection stage's row count, so putting the
terminal rows there is what makes them sampled. `validate_token_batch_rows` refuses any plan
where the device's count of that run would differ from the host's count of the rows it intends
to deliver.

The consequence is the one the plan asks for in §1 and §6.5: **this route does not arm
`split_terminal_prefill`, and `finish_prefill_batch` never runs on it.** Fusion peels the
prompt's last row into its own one-row chunk and replays it through `step_batch` — an ordinary
decode-shaped transformer pass, on the TTFT critical path, **once per request, whether or not
the fusion ever fires**. Measured over the identity corpus: fusion ran **33** of those passes
and the token batch ran **0**; over a campaign round, 100 against 0.

### 2.3 `converted_c` had to name the conversions that exist

It was empty, so `Capabilities::refuse_program` refused every real program. It now names **two**
class-C conversions, not the one §8's Phase 2 bullet lists:

| opcode | why it is class C | converted where |
|---|---|---|
| `FlashPrefill` | `i4 = q_pos0` plus the causal and window bounds built on it | `d_flash_prefill`'s `TB` arm resolves each work item to its span |
| `HeadNormRope` | `i3 = out_row0`, host-patched per chunk; every row's write address is `out_row0 + t` | `runtime/amd/op_norm.h` takes slot and absolute KV position from the descriptor |

`HeadNormRope` **is** the dense path's KV cache write — there is no separate write opcode. A
route that converted only `q_pos0` would put one request's K/V into another request's cache and
would not trap. The audit runs at load, per bucket, and refuses by capability name.

`RowGather` is deliberately **absent** from `descriptor_aware`, so `can_run_output()` keeps
saying no: the object has the arm, this route emits no terminal segment, and "armed" and "can
fire" are different claims.

### 2.4 Serving

`MixedAmdStep` gained a `StepRoute` rather than a copy. The two routes share the synthesized
program, the staging layout and the launch; they differ in the object
(`interp_tokbatch_gq.elf`), the kernel symbol (`plow_interp_tokbatch_<arch>_gq`, deliberately not
the mixed one so a stale object cannot answer the lookup), the span contract, and which buckets
they may execute. `SeqEngine` gained a token-batch surface whose defaults decline, so no other
backend changes, and the mux has an arm ahead of the mixed one.

The arm does **not** require a decode row — a step of prefill spans alone is legal, which is what
lets a prompt's first generated token come out of the launch that consumed the prompt. It does
require **two participants** (`feeds + pack >= 2`) on top of the correctness floor of at least one
sampled row; §5.3 is the measurement that put that rule there, and `--amd-token-batch-solo` turns
it off so it stays falsifiable.

---

## 3. Identity: is the output row for row the same?

Corpus: 8 **distinct** prompts per (length, concurrency), each a different pseudo-random sequence
of one-token words, at 128/512/2048/8192 input tokens and concurrency 1/4/8, greedy, 32 output
tokens. Distinctness is the point — `bench_packed_serve.py` repeats one word, so every row of a
packed step carries the same token and a row that picked up its neighbour's state is invisible.

**Two comparisons, and they answer different questions.**

**(1) Packed against isolated — the mapping gate.** Concurrency 4 and 8 against concurrency 1, on
the same arm, over the same 32 prompts. Both cells run the same kernels on the same bucket, so
any difference is a row that took another request's KV, position or hidden state.

| arm | packed == isolated | fails on |
|---|---|---|
| ordinary (unfused) | **32/32** | — |
| **unified token batch** | **32/32** | — |
| mixed step v1 (`--fusion`) | 27/32 | 128/6, 512/2, 2048/{2,3,6} |

The unified route is the **more faithful** of the two packed routes, not the less: v1 fails this
gate on five prompts and on two of them emits a degenerate repeated token where the reference
emits varied text. Those five are what a packed step is supposed to be tested for, and nothing
in tree was testing for them, because `bench_packed_serve.py`'s corpus repeats one word.

**(2) Against the ordinary (unfused) route.** This one also moves the prefill bucket, so a
difference here is not necessarily a mapping error.

| arm | c1 | c4 | c8 | total |
|---|---|---|---|---|
| **unified token batch** | 30/32 | 30/32 | 30/32 | **90/96** |
| mixed step v1 | **32/32** | 28/32 | 29/32 | 89/96 |

Reported per concurrency because the two arms fail differently and an aggregate hides it. **v1
is exact at concurrency 1 and the unified route is not**, which is the honest form of the
comparison: v1's divergences are concurrency-dependent (it only diverges when it packs), the
unified route's are the *same two prompts with the same text at every concurrency* — a fixed
structural difference, not a packing one.

Both of the unified route's misses begin at generated token 0 or 1, i.e. at the prefill's
sampled row. They are token flips, not low-order bits.

### Why the two prompts diverge, attributed

Two things differ from the ordinary route at 2048 input, and the c1 cell cannot separate them on
its own: the terminal-span split, and the prefill BUCKET (the route's `nsplit == 1` restriction
puts a 2048-token prompt on the 4096 rung where the ordinary route uses the 2048 rung).
`--amd-token-batch-rows` pins the rung, which holds the split fixed and moves only the reduction
order:

```
tb @ rung 4096  vs  tb @ rung 8192   IDENTICAL          <- the bucket is not the cause
tb @ rung 4096  vs  ordinary         2048/1/{2,3}
tb @ rung 8192  vs  ordinary         2048/1/{2,3}       <- same two, on either rung
ordinary @ chunk 8192 vs @ chunk 1024   2048/1/{2,3,6}  <- no packing anywhere
```

So it is the **terminal-span split**: the route runs a completing prompt as a 2047-row body span
plus a 1-row terminal span where the ordinary route runs one 2048-row chunk. That is a different
row partition of the same tokens, and the last line says the ordinary route's own answer moves
the same way — on a superset of the same prompts — when its chunk changes.

Cross-tabulated over four row partitions of the same eight prompts:

| prompt | 1×2048 rows | 2×1024 rows | 2047+1 on rung 4096 | 2047+1 on rung 8192 |
|---|---|---|---|---|
| seeds 0,1,4,5,7 | A | A | A | A |
| seed 2 | A | B | B | B |
| seed 3 | A | B | C | C |
| seed 6 | A | B | A | A |

**5 of 8 are identical under every partition.** On seed 2 the token batch lands on the answer
the ordinary route itself produces at a different chunk; on seed 6 the token batch agrees with
the unchunked reference where re-chunking does not. These are random one-token-word sequences
with no signal — exactly the input on which a near-tie flips — and the plan anticipates this
("Changing M may select different kernels and reduction orders; record that rather than
requiring unjustified universal bit identity"). The honest form of that record is this table,
not the aggregate count.

### The pass that is not run

| arm | token-batch launches | terminal-prefill decode passes |
|---|---|---|
| ordinary | 0 | 0 |
| mixed step v1 | 0 | **33** (identity corpus), **100** per campaign round |
| unified token batch | 99 / 219 / 227 | **0** |

`split_terminal_prefill` + `finish_prefill_batch` is one extra decode-shaped model pass per
request, on the TTFT critical path, whenever fusion is armed. The unified route runs none: it
consumes the prompt's last token in the same step and samples it there. That is the commit's
central claim, and it lands.

---

## 4. What this route still does not have

* **No terminal segment.** `RowGather` (opcode 154) has an arm in this object and is not used.
  The route samples the body's leading rows, which works because the planner puts every sampled
  row there. The plan's §6.2 compact tail — gather, then the model's own final norm, head and
  selection at `M = S` — is not emitted, and until it is, `S` is bounded by the synthesized
  program's `dcap` (`min(batch - 1, T - 1)`), not by the sample capacity.
* **Only buckets with `nsplit == 1` and a fused flash epilogue.** The device arm traps otherwise,
  and correctly: a split KV partition moves with the span boundary, and
  `exec_mixed_prefill_merge` still resolves rows through the decode prefix this contract removes.
  On this blob that is buckets 4096 and 8192. Every rung from 32 to 2048 splits its attention
  (`nsplit` 10, 8, 10, 5, 3, 2 going up), so **the token batch cannot use any of them**, and the
  sub-128 rungs in particular are invisible to it. See §6.
* **No TP.** `AmdServe::token_batch_rows` returns `None` for `Ranks::Tp`, as v1 does.
* **No prefix cache, no `decode_only`, no non-chunked prefill.** Same gates as v1.
* **One route or the other.** The engine refuses to load the token batch beside fusion; arming
  both would double the resident executables to measure neither.

---

## 5. Measured

### 5.1 Provenance

Everything below is **one blob and one object set**, with the three arms interleaved and their
order rotated between rounds, because a sibling measured 4% between-arm drift from server-process
drift alone. Per-cell figures are the median over rounds of the median over 3 repeats after 1
warmup; `±` is the round-to-round range as a percentage of the median, so a delta inside it
should be read as noise.

| | |
|---|---|
| model | Gemma-4 31B, BF16, dense windowed GQA + softcap, TP1 |
| blob | `build-utb/assets`, `model.pkt` sha256 `6b0cfabc34c061bf…`, max_ctx 32768, `PLOW_MAX_CHUNK=8192` |
| prefill rungs | 32, 64, 128, 512, 1024, 2048, 4096, 8192 |
| decode ladder | 1, 2, 4, 8 (8 KV slots — which is what makes concurrency 8 a cell at all) |
| tiles | **2220 of 3160 dense-GEMM tiles chosen by measurement; 940 fell back to the analytical model.** The 940 are exactly the shapes the new 32/64 rungs introduce — the same emit without the sub-128 floor reports "all 2220 chosen BY MEASUREMENT". Compare arms against each other, not against this document's absolutes. |
| objects | `build-utb/hsaco`, 47 objects, `PLOW_DECODE_BATCH=8 PLOW_DECODE_TIERS=1,2,4` |
| serving | `PLOW_PF_CHUNK=8192 PLOW_MULTISTEP=4`, one leased MI300X, 64 output tokens |

`interp_mixed_gq.elf` was verified **byte-identical** before and after the token-batch device
change in this branch, so the fusion arm is the object it always was.

### 5.2 The three arms

Deltas are signed so that **positive is better** in every table.

**Output tokens/s**

| in | c | off | fusion | token batch | fusion vs off | tb vs off | **tb vs fusion** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 40.80 ±1.1 | 40.77 ±0.8 | 39.95 ±1.5 | −0.1% | −2.1% | **−2.0%** |
| 128 | 4 | 88.21 ±1.8 | 92.20 ±0.5 | 96.00 ±1.5 | +4.5% | +8.8% | **+4.1%** |
| 128 | 8 | 93.48 ±0.9 | 107.67 ±0.1 | 110.05 ±0.1 | +15.2% | +17.7% | **+2.2%** |
| 512 | 1 | 38.30 ±0.1 | 38.28 ±0.4 | 36.88 ±0.5 | −0.1% | −3.7% | **−3.6%** |
| 512 | 4 | 79.50 ±1.4 | 79.36 ±1.2 | 82.27 ±3.9 | −0.2% | +3.5% | **+3.7%** |
| 512 | 8 | 83.31 ±0.1 | 86.80 ±0.0 | 90.28 ±0.8 | +4.2% | +8.4% | **+4.0%** |
| 2048 | 1 | 32.39 ±0.3 | 32.32 ±0.3 | 29.01 ±0.2 | −0.2% | −10.4% | **−10.2%** |
| 2048 | 4 | 58.02 ±1.0 | 49.79 ±0.1 | 49.57 ±0.2 | −14.2% | −14.6% | **−0.4%** |
| 2048 | 8 | 60.01 ±0.3 | 50.14 ±0.4 | 52.87 ±0.0 | −16.5% | −11.9% | **+5.4%** |
| 8192 | 1 | 19.66 ±0.0 | 19.63 ±0.2 | 19.63 ±0.2 | −0.1% | −0.1% | **+0.0%** |
| 8192 | 4 | 23.03 ±0.0 | 18.91 ±0.1 | 18.73 ±0.5 | −17.9% | −18.7% | **−1.0%** |
| 8192 | 8 | 22.08 ±0.0 | 18.03 ±0.2 | 18.52 ±0.6 | −18.3% | −16.1% | **+2.7%** |

**TTFT (ms)**

| in | c | off | fusion | token batch | fusion vs off | tb vs off | **tb vs fusion** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 106.7 ±14.2 | 104.2 ±10.0 | 140.5 ±17.6 | +2.4% | −31.7% | **−34.9%** |
| 128 | 4 | 327.5 ±14.7 | 425.9 ±4.0 | 261.5 ±28.8 | −30.0% | +20.2% | **+38.6%** |
| 128 | 8 | 631.7 ±8.1 | 609.6 ±0.2 | 430.4 ±0.6 | +3.5% | +31.9% | **+29.4%** |
| 512 | 1 | 174.4 ±0.4 | 174.5 ±0.5 | 240.4 ±3.0 | −0.1% | −37.9% | **−37.7%** |
| 512 | 4 | 513.3 ±7.2 | 854.4 ±5.0 | 739.3 ±15.7 | −66.5% | −44.0% | **+13.5%** |
| 512 | 8 | 983.7 ±0.1 | 1019.1 ±0.0 | 1113.7 ±30.4 | −3.6% | −13.2% | **−9.3%** |
| 2048 | 1 | 464.0 ±0.1 | 468.2 ±2.0 | 694.3 ±1.1 | −0.9% | −49.6% | **−48.3%** |
| 2048 | 4 | 1233.8 ±3.9 | 1606.6 ±0.1 | 1715.5 ±6.2 | −30.2% | −39.0% | **−6.8%** |
| 2048 | 8 | 2276.6 ±1.1 | 3143.2 ±1.3 | 3170.1 ±3.3 | −38.1% | −39.2% | **−0.9%** |
| 8192 | 1 | 1718.4 ±0.1 | 1719.4 ±0.1 | 1721.7 ±0.2 | −0.1% | −0.2% | **−0.1%** |
| 8192 | 4 | 5166.3 ±0.0 | 6387.0 ±0.2 | 6451.9 ±0.4 | −23.6% | −24.9% | **−1.0%** |
| 8192 | 8 | 9910.4 ±0.0 | 12648.7 ±0.4 | 12803.1 ±0.9 | −27.6% | −29.2% | **−1.2%** |

TPOT p50 is within ±2.5% of fusion everywhere except 512/8 (+4.7%), 2048/8 (+3.1%) and 8192/8
(+5.5%), all in the token batch's favour; against `off` it tracks fusion.

**Read it as three regimes.**

1. **Short prompts, concurrency ≥ 4 — the token batch wins outright.** 128 and 512 tokens: it
   beats fusion by +2.2 to +4.1% throughput and beats *both* other arms on TTFT at 128
   (+20.2% against off, +38.6% against fusion at c4). This is the cell fusion was adopted for,
   and the token batch is better in it.
2. **Long prompts — both packed routes lose to running neither**, by 12–19%, and the token
   batch's margin over fusion is small but consistently positive at concurrency 8 (+5.4% at
   2048, +2.7% at 8192). Whatever is wrong at 2048+ is wrong for packing in general, not for
   this route in particular.
3. **Concurrency 1 — the token batch loses and fusion does not.** §5.3.

**Why the long-prompt cells lose, mechanically.** The route only loads buckets whose
`FlashPrefill` has `nsplit == 1` — on this blob, 4096 and 8192; every rung from 32 to 2048 splits
its attention (`nsplit` 10, 10, 10, 5, 3, 2 going up). With decode in flight the interleave cap
is 2048 rows, so both arms take 2048-row chunks — but the ordinary route runs them on the 2048
rung and the token batch is forced onto the 4096 rung. The grid is sized at `T`, not at the live
row count: `rebase_chunk_rows` says the padded workgroups "still run the interpreter and still
signal their successor counters". So the token batch drains twice the workgroups per op per
layer, and gets half the KV-split parallelism on top. **This is the same tax the sub-128 prefill
floor was added to remove at the bottom of the ladder, paid at the top.**

### 5.3 Concurrency 1: what the route was doing wrong

At concurrency 1 the route fired on a **solo** prompt: no decode rows to fuse with, nothing to
pack, and the `nsplit == 1` restriction putting a 512-token prompt on the 4096 rung instead of
the 512 one. Cost: −2.1 / −3.7 / −10.4% throughput and up to −49.6% TTFT, for no packing.

There is nothing on the other side of that trade. The **ordinary** route already samples a
completing prompt's last row inside its own prefill program with no extra pass; the
`split_terminal_prefill` tax is fusion's alone. So admission now requires **two participants**
(`feeds + pack >= 2`) on top of the correctness floor of one sampled row.

Measured, concurrency 1, three servers simultaneously on separate GPUs (solo-firing, gated, and
the ordinary route as the anchor). The gated arm ran **0** token-batch launches, which is the
rule working: with one request there is one participant.

| input | solo tok/s | gated | ordinary | **gated vs solo** | gated vs ordinary |
|---:|---:|---:|---:|---:|---:|
| 128 | 40.02 | 40.64 | 41.03 | **+1.5%** | −0.9% |
| 512 | 37.35 | 38.85 | 38.93 | **+4.0%** | −0.2% |
| 2048 | 29.39 | 32.90 | 32.92 | **+11.9%** | −0.1% |
| 4096 | 22.59 | 27.33 | 27.37 | **+21.0%** | −0.2% |
| 7168 | 15.76 | 20.90 | 20.86 | **+32.6%** | +0.2% |

| input | solo TTFT ms | gated | ordinary | **gated vs solo** | gated vs ordinary |
|---:|---:|---:|---:|---:|---:|
| 128 | 136.8 | 122.5 | 107.2 | **+10.5%** | −14.2% |
| 512 | 233.2 | 178.3 | 171.6 | **+23.6%** | −3.9% |
| 2048 | 683.0 | 461.6 | 458.8 | **+32.4%** | −0.6% |
| 4096 | 1325.7 | 847.6 | 841.6 | **+36.1%** | −0.7% |
| 7168 | 2541.6 | 1556.2 | 1559.3 | **+38.8%** | +0.2% |

The rule recovers the whole loss: gated is the ordinary route to within ±0.9% throughput at every
length. The one residue is TTFT at 128 tokens, −14.2% — 15 ms absolute, on an arm that fires
zero times, so it is either the standing cost of a second HSA executable resident on the agent or
noise at that scale; it is not the route running.

**A correction to the audit's expectation.** `docs/arch/17-operator-row-identity-classes.md` §7.2
predicted fusion would cost 2.9–5.5% at concurrency 1 from the extra decode-shaped pass. On this
blob it costs **0.1–0.2% throughput and 0.1–2.4% TTFT** — the pass is real (100 of them per
campaign round) and it is not measurable here. The mechanism was right; the magnitude was from a
different configuration. That removes concurrency 1 as a place where this route can beat fusion:
there is nothing left to win there, only something to avoid losing.


---

## 6. `nsplit == 1`: the restriction, and what removing it costs

### 6.1 Why it binds

`dense_flash_split` gives `nsplit = ceil(n_cu / (ceil(t/256) * heads))`. On gfx942 with
`n_cu = 304` and 32 heads that is:

| bucket | 32 | 64 | 128 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|
| `nsplit` | 10 | 10 | 10 | 5 | 3 | 2 | **1** | **1** |

The route is qualified only at `nsplit == 1`, so it **loads only the 4096 and 8192 rungs**. It
does not fall back for shorter prompts — it runs them on the 4096 rung. Measured directly:
2048-token prompts at concurrency 1 produced 8 launches, all `rows=4096 decode=0 prefill=1
completed=1`, and **zero** declines; even 40-token prompts run `rows=4096`.

That has two consequences, and they are the two biggest facts in this document.

* **The sub-128 prefill floor (rungs 32 and 64) is invisible to this route.** It cannot select
  any rung below 4096, so the floor can only reach it through the ordinary path it falls back
  to. Whatever the floor is worth, it is not worth it *here*.
* **Every sub-4096 cell pays a rung-width tax.** The grid is sized at `T`, not at the live row
  count — `rebase_chunk_rows` says the padded workgroups "still run the interpreter and still
  signal their successor counters" — so a 2048-row chunk on the 4096 rung drains twice the
  workgroups per op per layer, and gets half the KV-split parallelism on top. This is the same
  tax the sub-128 floor was added to remove at the bottom of the ladder, paid at the top.

### 6.2 Removing it: `PLOW_DENSE_PF_NS=1`

The knob caps the heuristic (it can only remove splits, never over-split past the
`Opart`/`mlpart` capacity `max_splits` sizes from the same formula). At `=1` every bucket
qualifies, and `ns == 1` additionally switches the flash to its own bf16 epilogue and drops
`FlashMerge` entirely — verified on this blob, bucket 128: **60 `FlashMerge` packets stock, 0
capped**.

That is not free, and the emitted-packet count does not show why. The split exists to **fill the
machine**: at `t = 128`, one q-tile × 32 heads is 32 work items against 304 CUs, and `ns = 10`
takes that to 320. Capping at 1 gives that up for an *isolated* prefill. The falsifiable claim is
that a **token batch does not need it**, because the step is filled by the other spans and the
decode rows sharing it rather than by splitting one prompt's attention.

Six arms — `{stock, capped}` blob × `{ordinary, fusion, token batch}` — with the three arms of a
wave served **simultaneously, one per leased GPU**, so any host disturbance lands on all three in
the same wall-clock window rather than on whichever arm happened to be running. `capped.ordinary`
against `stock.ordinary` is printed beside the route's own delta precisely so that a cap that is
a straight win on its own does not get attributed to the route.

### 6.3 The answer: the cap does not help the route

**It does what it was supposed to do mechanically, and it buys nothing.**

The route does start selecting the rungs it could not reach: on the capped blob its launches are
`rows=512` and `rows=1024` where on the stock blob every launch — including for 40-token prompts
— was `rows=4096`. So the restriction is real, the cap removes it, and the route uses what the
cap gives it.

Two rounds, blob order rotated between them, three arms per wave served simultaneously on
separate GPUs. 64 output tokens, 3 repeats after 1 warmup, medians.

**Output tokens/s**

| in | c | stock .off | stock .fusion | stock .tb | capped .off | capped .fusion | capped .tb | **capped.tb vs stock.tb** | capped.tb vs stock.fusion | *capped.off vs stock.off* |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 40.52 | 40.87 | 40.31 | 40.50 | 41.13 | 40.24 | **−0.2%** | −1.5% | *−0.1%* |
| 128 | 4 | 87.03 | 94.86 | 97.22 | 88.32 | 94.78 | 97.20 | **−0.0%** | +2.5% | *+1.5%* |
| 128 | 8 | 92.83 | 108.42 | 110.08 | 93.17 | 108.62 | 109.93 | **−0.1%** | +1.4% | *+0.4%* |
| 512 | 1 | 37.82 | 38.37 | 36.94 | 37.86 | 38.47 | 36.78 | **−0.4%** | −4.1% | *+0.1%* |
| 512 | 4 | 78.92 | 79.95 | 82.25 | 80.14 | 79.14 | 81.32 | **−1.1%** | +1.7% | *+1.5%* |
| 512 | 8 | 82.74 | 87.29 | 89.81 | 83.36 | 87.43 | 90.31 | **+0.6%** | +3.5% | *+0.8%* |
| 2048 | 1 | 31.95 | 32.46 | 28.96 | 32.45 | 32.77 | 28.99 | **+0.1%** | −10.7% | *+1.6%* |
| 2048 | 4 | 57.58 | 49.85 | 49.76 | 59.13 | 50.37 | 49.84 | **+0.1%** | −0.0% | *+2.7%* |
| 2048 | 8 | 59.44 | 50.34 | 52.67 | 60.88 | 50.60 | 52.84 | **+0.3%** | +5.0% | *+2.4%* |
| 4096 | 1 | 26.54 | 26.95 | 22.42 | 26.64 | 26.99 | 22.42 | **−0.0%** | −16.8% | *+0.4%* |
| 4096 | 4 | 39.03 | 32.97 | 30.66 | 39.97 | 33.06 | 30.64 | **−0.1%** | −7.1% | *+2.4%* |
| 4096 | 8 | 38.74 | 32.28 | 32.53 | 39.77 | 32.34 | 32.47 | **−0.2%** | +0.6% | *+2.7%* |
| 7168 | 1 | 20.31 | 20.58 | 15.69 | 20.33 | 20.56 | 15.69 | **+0.0%** | −23.8% | *+0.1%* |
| 7168 | 4 | 24.71 | 21.13 | 18.85 | 25.40 | 21.14 | 18.84 | **−0.1%** | −10.9% | *+2.8%* |
| 7168 | 8 | 23.71 | 20.36 | 19.74 | 24.48 | 20.37 | 19.80 | **+0.3%** | −2.8% | *+3.2%* |

**TTFT (ms)** — the cap's effect on the route is again nothing, and on the ordinary route it is
consistently positive:

| in | c | stock .off | stock .tb | capped .off | capped .tb | **capped.tb vs stock.tb** | *capped.off vs stock.off* |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 360.4 | 255.9 | 320.6 | 224.2 | **+12.4%** | *+11.1%* |
| 512 | 4 | 522.5 | 628.9 | 478.2 | 683.2 | **−8.6%** | *+8.5%* |
| 2048 | 4 | 1237.3 | 1821.0 | 1162.3 | 1870.0 | **−2.7%** | *+6.1%* |
| 4096 | 4 | 2473.3 | 3660.2 | 2395.0 | 3663.8 | **−0.1%** | *+3.2%* |
| 7168 | 4 | 4721.5 | 7198.7 | 4579.9 | 7204.9 | **−0.1%** | *+3.0%* |
| 7168 | 8 | 9059.7 | 12657.5 | 8704.0 | 12667.1 | **−0.1%** | *+3.9%* |

**TPOT p50** moves by less than 1.2% for the route in every cell, and by −0.3% to +1.9% for the
ordinary route. No arm buys throughput with per-token latency here.

Round-to-round spread on the capped blob is **≤ 0.9%** in every cell (the three arms of a wave
were served simultaneously on separate GPUs, which is why it is that tight), so a delta inside
±1% is nothing.

**Three readings, in order of how much they change what to do next.**

1. **`capped.token-batch` vs `stock.token-batch` is flat: −2.4% to +1.5% throughput.** Giving the
   route the right rung does not pay. The falsifiable claim — that a token batch does not need
   the KV split because the step is filled by the other spans — is **not supported**: the route
   neither gains from getting the narrower rung nor loses from giving up the split. Whatever
   binds it is not the rung.
2. **It refutes my own §6.1 explanation.** I attributed the long-prompt regression to the
   rung-width tax. It is not that: with the cap the route runs the natural rung and still loses
   — `capped.token-batch` 18.83 tok/s against `capped.ordinary` 25.39 at 7168/4, −26%. The
   long-prompt regression is unexplained and is the open question this campaign leaves.
3. **`capped.ordinary` vs `stock.ordinary` is a small straight win: +0.1 to +3.1% throughput,
   +0.5 to +18.1% TTFT.** That is the cap on its own, with no packing anywhere, and it is
   *larger* than anything the cap does for the route. It is a separate and possibly more
   interesting finding than the one this campaign was run for, and it must not be attributed to
   the token batch — which is exactly why the fifth arm was measured.

---

## 7. Recommendation

**Do not default it globally. Default it in place of `--fusion` for the (gfx942, dense windowed
GQA BF16, TP1) pair, with the two-participant admission rule on, and leave both packed routes off
by default as fusion already is.**

Scoped to the cells actually measured — 128/512/2048/4096/7168 input, concurrency 1/4/8, one blob,
one object set, two rounds:

| workload | best arm | token batch vs the next best |
|---|---|---|
| ≤512 input, concurrency ≥ 4 | **unified token batch** | +4.2 to +18.6% tokens/s over the ordinary route; +1.4 to +3.5% over fusion; +28 to +40% TTFT over fusion at 128 |
| 2048 input, concurrency 8 | **ordinary** | token batch is +5.0% over fusion but −11.4% under ordinary |
| ≥2048 input, concurrency 4; ≥4096 anywhere | **ordinary** | both packed routes lose 11–24%; the token batch loses more than fusion above 2048 |
| concurrency 1 | **ordinary** | with the two-participant rule the route does not fire and is the ordinary route to within ±0.9% |

Three claims support "in place of `--fusion`" rather than "as well as", in the regime where a
packed route is worth arming at all:

1. **It beats fusion in every cell where either is worth arming.** +1.4 to +3.5% throughput and
   +28 to +40% TTFT at 128/512 with concurrency ≥ 4, and it does not have fusion's concurrency-1
   tax to carry.
2. **It is more faithful.** Packed output equals isolated output 32/32; fusion is 27/32, and two
   of its five misses replace varied reference text with a degenerate repeated token.
3. **It deletes a per-request cost rather than inheriting it.** 140 extra decode-shaped passes
   per campaign round on fusion, 0 on this route.

What it is **not** yet, and why the default stays scoped:

* **One (backend, family) pair, one arch, one blob.** gfx942, Gemma-4 31B, BF16, TP1, batch 8. TP
  returns `None`; no MoE, MLA, DSA, recurrent or FP8 family has been near this route. FP8 was
  asked for and is not measured here.
* **The long-prompt regression is unexplained.** It is not the rung width — §6.3's cap experiment
  refutes that directly — and it is not the split-K, since capping `nsplit` moves the route by
  less than 1.1%. Both packed routes lose at ≥2048, so it is a property of packing prefill with
  decode at long context rather than of this route; the token batch simply loses more of it above
  2048. Finding the cause is the next work item and it is worth more than anything else on this
  list, because it is the difference between a short-prompt feature and a general one.
* **The terminal segment is still not emitted.** `S` is capped by the synthesized program's
  `dcap = min(batch - 1, T - 1)` = 7, and the campaign shows the route running *at* that ceiling
  (`decode=6 prefill=1`, `decode=3 prefill=4 completed=4`). A compact `RowGather` tail with its
  own sample capacity would raise it, and is what the plan's §6.2 asks for.
* **940 of 3160 dense-GEMM tiles on this blob are analytical, not measured** — exactly the shapes
  the 32/64 rungs add. Every arm shares that, so the comparison holds; the absolutes do not
  transfer.

One finding that belongs to nobody on this list: **capping `nsplit` to 1 is a small straight win
for the ORDINARY route** — +0.1 to +3.2% throughput and +0.2 to +11.1% TTFT, rising with input
length — while doing nothing measurable for the token batch. That is a separate question from
this route and should be pursued as one.

---

## 8. Reproducing

```bash
# Blob. plowc probes the checkout it is RUN in for the tile store, so run it from the worktree.
PLOW_L2_PLACE=0 PLOW_AMD=1 PLOW_MAX_CHUNK=8192 \
  target-utb/release/plowc --hf-dir build-gemma31/checkpoint \
  --gpu MI300X --arch gfx942 --num-gpus 1 --max-ctx 32768 --out build-utb/assets

# Objects. PLOW_DECODE_BATCH must match the blob's decode ladder or the mixed/token-batch
# objects advertise a GEMV bucket the route can exceed.
PLOW_DECODE_BATCH=8 PLOW_DECODE_TIERS=1,2,4 scripts/build_gfx942.sh build-utb/hsaco

# Serve. INSIDE nix develop, or plowrt silently uses the CPU reference backend — check the
# serve log's "backend ready — GPU accelerated" line before believing any number.
PLOW_TOKEN_BATCH=1 PLOW_PF_CHUNK=8192 PLOW_MULTISTEP=4 \
  nix develop /app/plow --command scripts/glm53_serve_inner.sh \
  build-utb/assets 21708 build-utb/hsaco target-utb/release
```

Knobs this work added, all opt-in and all through `RuntimeConfig`:

| flag | env | what |
|---|---|---|
| `--token-batch` | `PLOW_TOKEN_BATCH` | arm the route (mutually exclusive with `--fusion`) |
| `--amd-token-batch-solo` | `PLOW_TOKEN_BATCH_SOLO` | admit a step with one participant; §5.3 is why it is off |
| `--amd-token-batch-rows` | `PLOW_TOKEN_BATCH_ROWS` | pin the prefill rung, for attribution (§3) |

The identity and attribution corpora are `ident.py` / `attrib.sh` in this campaign's scratch;
they differ from `bench_packed_serve.py` in one way that matters — **every request gets a
different pseudo-random word sequence**, so a row that picked up its neighbour's state produces
different text. With one repeated word, as the shipped bench corpus uses, it does not.

The route logs one line at load with `armed` and `fires` as separate fields:

```
route="unified-token-batch/dense-gqa" object=.../interp_tokbatch_gq.elf
kernel=plow_interp_tokbatch_gfx942_gq armed=true fires=true reason="-"
```

`armed=true fires=false` names the capability that is missing. A route that reports one
`enabled` flag is how three campaigns on this branch measured "no effect" from something that
never fired.
