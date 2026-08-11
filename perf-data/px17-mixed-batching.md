# PX-17 — prefill⊕decode fusion is worth 5%, and `main` has been serving corrupt fp8-KV prefills

RTX 5090 (sm_120a, 170 SM, 31.36 GiB usable) · 2026-07-26 · Gemma-4-12B-it, fp8 W8A8, MIXED
fp8-KV packet (e4m3 on the 8 hd512 FULL layers, bf16 on the 40 sliding rings). Every GPU run under
`perf-data/tools/gpulease`. Companion to `gemma4-12b-longctx-5090.md` §12–§14,
`px12-consolidated-baseline.md` (§3 is the cell) and `px14-batched-prefill-fp8.md`.

**Question.** plow loses the matched 127k cell 29.91 vs vLLM 42.49 (1.42×), and ~81% of its wall is
serial prefill that does not overlap decode. Today a tick is `prefill chunk launch` **+** a separate
`decode launch`, two full 12 GiB weight reads. **What does folding the decode rows into the prefill
launch actually buy on this cell?**

**Answer: ~7%, and a six-line scheduler flag already gets it.** The addressable wall is
**22.46 s of 282.55 s** — every decode launch that shares a tick with a prefill chunk — plus ~10 ms
of per-tick host turnaround that `packlog` does not see. Fusion projects to **−17.5 s ⇒ ~30.9
tok/s**. A scheduler-only control that attacks the same fixed cost (`PLOW_PF_DEFER_DECODE=1`, §4)
**measures −18.73 s ⇒ 31.05 tok/s** with no cubin, no packet change and no kernel work. vLLM stays
**1.37×** ahead either way. **Recommendation: do not build fusion for this cell.** Sizing was posted
before implementing, per the brief; the fusion feature is NOT implemented.

**The run that was supposed to be a control found a correctness bug instead** (§1). `main` never
received the fp8-KV prefill patch-site fix that `gemma4-12b-longctx-5090.md` §4 documents. On `main`
every fp8-KV prefill chunk after the first writes its KV at row 0 and reads with `q_pos0 = 0`. It
does not crash — it makes prefill look **1.76× faster** and the whole cell look **1.56× faster**
(45.21 tok/s, i.e. "plow beats vLLM"), which is exactly how it survives a benchmark. Fixed here.

---

## 1. BUG: `main` collects prefill patch sites by bf16 opcode only

`exec/gpu.rs` builds each prefill bucket's per-chunk patch list by matching opcodes:

```rust
if inst.op == DevOp::HeadNormRope as u16 && inst.fj[1] != 0 { rope.push(ix) }
else if inst.op == DevOp::FlashPrefill as u16 { flash.push(ix) }
```

A **mixed fp8-KV packet emits the Fp8 twins** (`HeadNormRopeFp8` / `FlashPrefillFp8`, dev.rs 37/39),
which carry the same operands at the same indices. Neither matches, so `rope_sites` and
`flash_sites` come back **empty** and `run_one_prefill_chunk`'s patch loop is a no-op: `i[3]`
(out_row0), `i[1]` (seq_kv) and `i[4]` (q_pos0) keep their emitted values on every chunk.

This is byte-for-byte the bug `gemma4-12b-longctx-5090.md` §4 found and fixed in the campaign
worktree. **It was never merged.** `main` (`ddd8d01`) has the bf16-only match; the px12 worktree has
the twin-matching version. The 127k campaign numbers were all produced by the fixed build, so no
published figure is affected — but anything measured on `main` since is.

### 1a. Why it reads as a speedup, not a failure

The flash never sees `seq_kv` grow past the first chunk, so ~123 of 124 chunks do a fraction of
their attention work. Same asset (`/root/px12/mx/E`), same cubins, same packet, **only the host
binary differs**:

| arm | conc-1 127k prefill | cell wall | cell out tok/s | median ITL |
|---|---|---|---|---|
| `main`, bf16-only patch match (**corrupt**) | **15.73 s** | 181.19 s | **45.21** | 44.07 ms |
| + patch-site fix (this branch) | **27.65 s** | 282.55 s | **28.99** | 42.66 ms |
| *px12 arm E, recorded* | *27.59 s* | *273.9 s* | *29.91* | *40.39 ms* |

The fixed build reproduces px12's conc-1 prefill to **0.2%** and its cell to **3.1%**, so this
branch is a valid stand-in for the campaign baseline. The corrupt build would have been reported as
**plow beating vLLM by 1.06×**. It is 1.56× of fictitious throughput.

`crates/packet/src/devbuild.rs:434` already pairs the two FlashPrefill opcodes; only the runtime's
patch loop was missed. Fix: match the Fp8 twins, and mark the bucket `fp8_kv` so PX-1 batched
prefill stays off for it (the fp8 arm spends `t6`/`t7` on the k/v dequant scales, not on the
request table).

### 1b. A second `main` gap, same run

`main` also ignores the client's output cap: `ChatRequest` has no `max_completion_tokens` alias and
no `ignore_eos`. `vllm bench serve --backend openai-chat` sends only the new name, so the first run
of the cell generated **14,253** tokens instead of 8,192 and the "cell" was not the cell. Ported
from the campaign worktree (4 files, ~20 lines). Both fixes are in this branch.

---

## 2. The measurement: `PLOW_PF_PACKLOG` now splits mixed from pure-decode ticks

The tick loop (`serve/mux.rs`) is: gather `feeds` (slots past prefill) → run **one** capped prefill
chunk → **one** batched decode launch, and `if !feeds.is_empty() { break }` bounds the prefill chain
to one chunk per tick. So on this cell a tick is a prefill chunk *plus* a whole separate decode
launch. `packlog` already timed both halves; PX-17 adds the split that matters — decode wall on
ticks that **also** ran a prefill chunk, which is exactly what a fused launch could address, plus
the decode row counts.

Cell, arm B (fixed build), `PLOW_PF_PACKLOG=1`:

```
prefill_ns=244_780_142_608  decode_ns=46_902_077_399  prefill_ticks=877  decode_ticks=1748
mixed_decode_ns=22_463_338_393  mixed_ticks=875  mixed_rows=3500  decode_rows=8002
```

| quantity | measured |
|---|---|
| wall | 282.55 s |
| prefill (cell only, probe subtracted) | 217.3 s = 8 × 27.2 s |
| **mixed-tick decode launches** | **875 ticks, 22.46 s, 3500 rows (mean B = 4.00)** |
| pure-decode ticks | 873 ticks, 24.44 s, 4502 rows (mean B = 5.16) |

875 mixed ticks is the predicted 7 × 124: request 0's whole prefill runs in tick 1 (no decoders
live ⇒ `cap_rows = usize::MAX`, the chain runs uninterrupted), then every later chunk shares its
tick with a decode launch. 3500 = 124 × (1+2+…+7) exactly.

**Reproducibility:** `mixed_decode_ns` was 22.03 / 22.31 / 22.46 s across three cell runs (two of
them on the corrupt build, whose prefill phase differs but whose mixed-decode phase does not) —
±2%.

## 3. What fusion recovers: `a`, not `a + bB`

A decode launch costs `a + b·B`. `a` — the 12.036 GiB weight re-read plus launch/host turnaround —
is paid whatever `B` is; `b·B` is the per-row flash-decode + merge + lm_head, which **still runs
inside a fused launch**. Fusion removes `a` only. Two in-run points plus the conc-1 probe:

| point | B | ms/tick |
|---|---|---|
| mixed ticks | 4.00 | 25.67 |
| pure-decode ticks | 5.16 | 28.00 |
| conc-1 probe (median ITL) | 1 | 20.19 |

⇒ **`b` = 2.01 ms/row, `a` = 17.63 ms**; the fit predicts the B=1 probe at 19.6 ms vs 20.19
measured (3%). Weight bandwidth alone is 12.036 GiB / ~1.5 TB/s ≈ 8.6 ms, so a little over half of
`a` is the weight read and the rest is launch + host turnaround.

| | s | % of 282.55 s wall | cell out tok/s |
|---|---|---|---|
| baseline (arm B) | — | — | **28.99** |
| fusion, GPU-side only — recovers `a` × 875 | −15.43 | 5.46% | 30.66 |
| … minus bucket reservation (pack prefill to `1024 − B`, mean B=4 ⇒ +0.39% launches) | +0.85 | | 30.5 |
| **fusion, restated with the host cost §4 exposes (`a′` ≈ 28 ms)** | **−17.5** | **6.2%** | **~30.9** |
| fusion, absolute ceiling — the whole 22.46 s vanishes, host cost included | −25 | 8.8% | 31.7 |
| vLLM 0.26.0, same cell | | | **42.49** |

**vLLM stays 1.37× ahead at the restated number and 1.34× at the ceiling.** Applied instead to
px12's recorded 29.91, fusion gives ~31.9 — the same conclusion. The `a`-only row is kept because it
is what the GPU-side instrumentation alone supports; §4 is what corrects it, and the correction only
matters because it was measured rather than argued.

### 3a. Correcting the brief's own arithmetic

The brief sized fusion at "9–10 ms saved per tick on ~992 ticks". Two corrections, both measured:
mixed ticks are **875**, not 992 (request 0's prefill is one uninterrupted tick), and the saving per
tick is **17.63 ms**, not 9–10 — the weight read is only ~half of the launch's fixed cost. The two
errors run opposite ways and the product lands inside the brief's stated "single-digit to low-teens
percent" band, at the bottom of it.

An earlier design note (§1) sized this at **~0.6% at 127k** by assuming
the tick is a ~1390 ms chunk. The chunk is ~220 ms at the deployed chunk-1024 ladder, so that
figure is ~9× low. Neither estimate was measured; both are now superseded by 22.46 s.

## 4. Cross-check: the scheduler-only bound (`PLOW_PF_DEFER_DECODE=1`)

The `a`/`b` split is what the whole projection rests on, and this campaign has twice been wrong by
scaling an isolated ratio through an assumed budget. So it is checked end to end with a ~6-line
scheduler change that attacks the same `a` from the other side: while any slot is mid-prefill, drop
the decode feeds entirely, so the deferred rows are served later by **full-batch** ticks that pay
`a` once for 8 rows instead of once for 4.

Predicted from `a`/`b`: remove 22.46 s, add back 3500 tokens at B=8 (`a + 8b` = 33.7 ms per 8 rows)
= 14.7 s ⇒ **−7.7 s, −2.7%**.

| arm | wall | out tok/s | median ITL | mixed ticks | decode ticks |
|---|---|---|---|---|---|
| **B — baseline** | 282.55 s | **28.99** | 42.66 ms | 875 | 1748 |
| **C — `PLOW_PF_DEFER_DECODE=1`** | **263.82 s** | **31.05** | 44.38 ms | **0** | ~1030 |

**Measured −18.73 s, +7.1% — 2.4× more than the `a`/`b` model predicts.** The model was built from
`packlog`'s GPU-side timings, which is exactly what it should predict and exactly why the
cross-check was worth running: the miss is **per-tick host cost that `packlog` never sees**. Wall
minus (`prefill_ns` + `decode_ns`) is 18.5 s in arm B and 11.2 s in arm C, and arm C runs ~720 fewer
ticks — ≈ 10 ms of host turnaround per decode launch on top of the 17.63 ms `a`.

So the honest per-decode-launch fixed cost is **`a′` ≈ 28 ms**, not 17.63, and the fusion projection
must be restated with it:

| | s | cell out tok/s |
|---|---|---|
| baseline | — | 28.99 |
| **fusion, restated** — `a′` × 875 removed, `b·B` (7.0 s) retained inside the fused launch | **−17.5** | **~30.9** |
| **arm C, MEASURED** — scheduler only, 6 lines | **−18.73** | **31.05** |
| vLLM 0.26.0 | | **42.49** |

**A six-line scheduler flag already delivers everything fusion is projected to deliver on this
cell**, and it needs no cubin, no packet change and no kernel work. That is the §14-shaped result:
the lever is real, and the expensive way to pull it is not the one to build.

Arm C is **not** a shippable default — it holds every token until the last prompt is resident, so
TTFT collapses (plow's TTFT is unreportable here for the reason px12 gives, but the effect is
structural, not a measurement artifact). Median ITL also drifts 42.66 → 44.38 ms because every
decode tick now runs at the full batch. **Fusion's real advantage over arm C is that it buys the
same throughput without the latency cost** — which is an argument for building it on a
latency-sensitive profile, not on this throughput cell.

## 5. Design, validated against the code (for whoever revisits this at short context)

Fusion is the right architecture and pays on a *different* profile — the design notes size
it at 9–12% at 2k/8k with high concurrency, where the prefill chunk is small and the decode launch
is a comparable share of the tick. That case was not measured here. If it is built, the design below
is what survives contact with the tree.

**The GEMMs fuse, the attention does not.** `d_flash_prefill_mux` forces `ns = 1` for batched
requests; a decode row at `qlen = 1` has only split-KV parallelism (`NS_FULL_ABS = 32`), so it must
keep its own `FlashDecode` + `FlashMerge` instruction. Per layer: norm/HeadNormRope over `T` rows →
`FlashPrefill(Fp8)` over the prefill rows with the request table → `FlashDecode(Fp8)` +
`FlashMerge` over the decode rows → o_proj / gate|up / down over `T` rows. Both attention ops
present every layer, each gated to nothing in the interpreter when its row count is zero.

**Put the decode rows FIRST: `[0, B)` decode, `[B, T)` prefill.** This is a change from the brief and
it removes two problems at once:

* **No new `FlashDecode` operand.** The brief says `i7` is free so `q_row0` needs no wire change.
  **`i7` is not free** — `interp_sm120.cu` passes `in->i[7]` as `kv_mask` to `d_flash_decode`, and
  `fj[1].u` is `kv_cap`. With decode at rows `[0,B)` the op reads Q from row 0 with `n_batch = B`
  and needs no offset at all. PX-1's request table already carries a per-request `q0`, so prefill
  rows starting at `B` is expressible with no kernel change.
* **lm_head at M = B, no trap and no GEMV predication risk.** The prefill object emits lm_head at
  M=1 over the last prompt row, and `PLOW_NV_PF_GEMV_HEAD` routes it to an arm that traps on M≠1.
  With decode first, lm_head is `a_row0 = 0`, `M = B` on the existing **tiled** GEMM arm — a patch,
  not a new kernel. It costs ~1.3 ms/launch against the GEMV arm's ~0.78 ms, and px12 measured
  `PF_GEMV_HEAD` **inert** at this shape (32.51 vs 32.39 s), so the right move is to gate
  `PF_GEMV_HEAD` off for fused packets via the header bit rather than leave a trap reachable. The
  batched GEMV rung (`gv_mm<B>`) is deliberately **not** entered: it would re-open PX-10's
  remainder-arm predication bug and require `GV_MM_MAX == next_pow2(B)`, for 0.5 ms.

**Keep PX-1's contract:** pack prefill to `n−1` prompt rows so a finishing request becomes a decode
row on the NEXT tick. Then no prefill row ever needs logits.

**Positions and KV addressing are free**, as briefed: `t_pos` is per-row and staged host-side
(`gpu.rs:2660`), and the per-row slot map is PX-1's `d_slot`, already patched into `HeadNormRope`
`t[6]`. Fill `[0,B)` from each decode row's `pos[slot]` and `[B,T)` with `c0+k`.

**Declare fusion in the blob header.** Add `PLOW_BLOB_F_FUSED_PD = 4` beside `PLOW_BLOB_F_L2DOM`
(`devbuild.rs:790`), set it in the emitter, and verify it at load against a cubin global
(`plow_fused_pd`, the same pattern as `plow_arena_bytes` / `plow_dyn_kvrow`, `gpu.rs:637/954`),
refusing with a named `RuntimeError::Device` rather than a device trap. Bit clear ⇒ today's
two-launch path, byte-identical. Note: the `check_packet_pairing()` the brief points at exists only
in the campaign worktree, not on `main`; the equivalent load-time gate here is the
`grid != blob.n_cu` check at `gpu.rs:650`.

**The real build risk is the object, and it is cheap to retire first.** `PLOW_DOP_FLASH_DECODE` and
`FLASH_DECODE_FP8` sit inside `#if !PLOW_NV_PREFILL` (`interp_sm120.cu:899–1153`), so they are
compiled **out** of the prefill object — a fused packet needs a NEW cubin variant, and
`interp_sm120.cu:285-294` says the split is deliberate ("register- and smem-hungry … build as a
SEPARATE object rather than stacking onto the decode megakernel's 150-reg / hd512-flash-decode
footprint"). Registers take the max over arms and the prefill arms dominate, and the dynamic smem
arena is a union (a max, not a sum) — this asset's prefill object reports `smem_pf = 94096` of
101,376 reservable, against ~16–32 KiB for flash-decode — so neither should move. **Verify with
`-Xptxas -v` and a load test before writing any host code**: if occupancy moves, `grid = occ ×
sm_count` must still equal the packet's `n_cu` or the module refuses to load outright.

**The kernarg move** (request/slot tables out of `DevInst64.t` into `DevProgram`) is still the right
first step and is independently useful — it frees the fp8 `t6`/`t7` blocker and deletes the per-site
patch loops. px14 §6 prices it; it is unchanged by this note.

## 6. Gates

| gate | result |
|---|---|
| control reproduces px12 arm E, conc-1 prefill | **PASS** — 27.65 s vs 27.59 (0.2%) |
| control reproduces px12 arm E, full cell | **PASS** — 28.99 tok/s / 282.55 s vs 29.91 / 273.9 (3.1%) |
| same offered load as px12 | **PASS** — `Total input tokens` 1,015,914 and `Total generated` 8,192, identical to px12 §3 |
| **fp8-KV prefill patch-site bug on `main`** | **FOUND AND FIXED** — §1. `main` matched only the bf16 opcodes; every fp8-KV chunk after the first wrote KV at row 0. Presents as a 1.76× prefill speedup |
| `main` ignores `max_completion_tokens` / `ignore_eos` | **FOUND AND FIXED** — §1b. First cell run generated 14,253 tokens instead of 8,192 |
| fusion's addressable wall measured on the cell | **PASS** — 22.46 s of 282.55 s (7.95%), reproducible to ±2% over three runs |
| `a`/`b` decode-launch decomposition | **PASS** — three points, fit predicts the held-out B=1 probe to 3% |
| **scheduler-only cross-check (`PLOW_PF_DEFER_DECODE=1`)** | **PASS, and it invalidated the first projection** — measured −18.73 s / +7.1%, vs −7.7 s predicted from GPU-side timings alone. Exposed ~10 ms/tick of host turnaround `packlog` cannot see; §3's number is restated in §4 |
| **fusion implemented** | **NOT RUN — deliberately.** The brief says to post sizing first and stop if the arithmetic cannot pay. It projects ~6%, a six-line scheduler flag measures 7.1%, and vLLM stays 1.37× ahead |
| kernarg move landed bit-exact | **NOT RUN** — not implemented; §5 and px14 §6 price it |
| pure-prefill / pure-decode ticks bit-exact vs today's two objects | **NOT RUN** — nothing to compare; no fused object was built |
| greedy parity at ≥8k at FIXED chunk | **NOT RUN** — no fused arm to compare. Chunk was held constant (1024) across every arm here, per §14b |
| pure-decode tick must not regress | **NOT RUN** — no fused object. Measured baseline for it is recorded above: 28.00 ms at B=5.16, 20.19 ms at B=1 |
| occupancy of a merged object | **NOT RUN** — §5 gives the check (`-Xptxas -v` + load-time `grid == n_cu`) and the reason to expect no movement |
| long-context coherence after the fix | **PARTIAL** — the fixed build reproduces px12's wall to 0.2%/3.1%, which the corrupt build misses by 1.76×/1.56×. No needle-in-haystack was run on this branch |
| GPU exclusive | **ENFORCED** — every run under `perf-data/tools/gpulease`, rc=0 |

## 7. Reproduce

    perf-data/tools/gpulease px17-B perf-data/px17_run.sh B                          # baseline
    perf-data/tools/gpulease px17-C perf-data/px17_run.sh C /root/px12/mx/E PLOW_PF_DEFER_DECODE=1

`px17_run.sh` runs a conc-1 127k prefill probe and then the cell in ONE serve session with
`PLOW_PF_PACKLOG=1`; the probe's PACKLOG line is the prefix of the cell's. Asset
`/root/px12/mx/E` (px12 arm E: FASTPF + `PF_GEMV_HEAD` + `FP8PV` prefill object, tuned decode
object). Binaries from this worktree, `cargo build --release -p plowrt --features cuda`.

## 8. Verdict

**Fusion is real, correctly designed in the brief, and too small to matter here.** The wall it can
touch is 22.46 s of 282.55 s plus per-tick host cost; it projects to `28.99 → ~30.9 tok/s`, vLLM
still 1.37× ahead — and a **six-line scheduler flag measures 31.05** on the same cell. Every version
of the estimate made without measuring — 0.6% in the design notes, "~992 ticks × 9–10 ms"
in the brief, and this note's own first pass at 5.5% — was wrong in a different direction, which is
the same failure mode px12 §8 and px14 §9 already called out. The one that survived is the one that
was measured end to end.

What this cell is still bounded by has not changed since `gemma4-12b-longctx-5090.md` §6: the
prefill GEMM at 38% of fp8 peak, and 8 × 27.2 s of prefill that no decode-side change can touch.
Fusion is worth building for the short-context/high-QPS profile, where the design notes
size it at 9–12% — **that shape has not been measured and is the open question this note leaves.**

**The most valuable thing in this note is not the sizing.** It is that `main` has been silently
corrupting every multi-launch fp8-KV prefill, and that the corruption presents as a 1.56× throughput
*gain* on the exact benchmark the campaign uses to rank itself.
