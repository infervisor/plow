# Unified token batch: dense GQA on AMD

Status: the device route is implemented, built and gated on gfx942. It is **armed and does not
fire**: no dense-GQA checkpoint exists on this host and the shared planner still hands the device
a decode band ahead of the span table. Nothing here is a performance claim.

This is the AMD half of Phase 2 of `plans/unified-token-batch.md` — the first (family, backend)
pair, chosen because dense causal GQA is class-A work plus **one** class-C conversion, so it
isolates the descriptor's correctness from MoE, MLA, DSA, TP and recurrent state.

---

## 1. What the route is

One packed activation matrix per step holding every scheduled input token — decode rows and
prefill rows, from any number of requests — with per-row position, physical KV slot and validity
coming from a descriptor rather than from packet immediates.

Concretely, against the object it replaces:

| | `interp_mixed` (phase bands) | `interp_tokbatch` (this route) |
|---|---|---|
| Projections | GEMV over the decode band, then GEMM over the prefill band | **one** matmul over combined M; kernel chosen from the live row count |
| Fused GLU | same split | same, one call |
| `FlashPrefill` | a serial device-side loop over spans, whole grid per span, `__syncthreads()` between | one flat work list over `(span, q-tile, head, split)` |
| `FlashPrefill` bounds | per span, from the loop | per span, from the row's own span; the packet's `q_pos0`/`n_q`/`n_kv` are passed as literal `0` |
| `FlashDecode` slot | host-staged `t6 = in.decode_slot` tensor | `spans[b].slot` |
| `FlashDecode` length | host-staged `t5 = in.kvlen` tensor | `spans[b].kv_len` |
| KV write row | `HeadNormRope i3 = out_row0`, host-patched per chunk | the row's span position and slot |
| Row counts | compiled capacity, or the decode prefix `spans[0].row0` | the descriptor's live `real_rows` |
| Row `0` | must be a decode row — the resolver **traps** if `spans[0].row0 == 0` | spans cover exactly `[0, M)`; pure prefill and pure decode are both legal |

The axis is `PLOW_TOKEN_BATCH`, default `0`. Every shipped object is unchanged; see §6.

---

## 2. The descriptor this consumes

`runtime/amd/token_batch.h` is an **AMD-local adapter**, not the shared contract. The shared
`PlowTokenBatch` (§4.2 of the plan) belongs to the Phase 1a/1b/1c work and had not landed when
this was written. The header states the layout it assumes at the top of the file:

```c
typedef struct {
    uint32_t version;            /* 1 */
    uint32_t row_capacity;       /* compiled/padded allocation capacity */
    uint32_t real_rows;          /* M: scheduled input tokens this step */
    uint32_t sample_rows;        /* S */
    uint32_t n_spans;            /* R */
    uint32_t sample_capacity;
    const PlowPrefillSpan* spans;             /* [n_spans], covering EXACTLY [0, M) */
    const int32_t*         positions;         /* [M] */
    const int32_t*         active;            /* [row_capacity], 1 = live */
    const uint32_t*        sample_input_rows; /* [S] */
} PlowTokenBatch;
/* PlowProgram gains: const PlowTokenBatch* token_batch;   NULL = old semantics */
```

Until that lands, `PLOW_TOKEN_BATCH_DESC=0` synthesises the same view from the packed-prefill
tail `PlowProgram` already carries, which holds every field this family needs:

```
positions[row] = span->kv_row0 + (row - span->row0)     (§4.4's own invariant)
active[row]    = !prefill_parked[row]
real_rows      = the last span's end
```

Set `PLOW_TOKEN_BATCH_DESC=1` to switch to the real descriptor; the consumers do not change.

### Three conventions over one span table

This is worth stating plainly because it is exactly the drift §4.3 exists to prevent. The repo
now resolves the same `PlowPrefillSpan[]` three different ways:

| Header | Requires | Decode rows |
|---|---|---|
| `runtime/amd/packed_prefill.h` | `spans[0].row0 == 0` | none — prefill only |
| `runtime/common/mixed_step.h` | `spans[0].row0 != 0` (**traps** otherwise) | a band ahead of the spans, resolved from `decode_slots[]`/`positions[]` |
| `runtime/amd/token_batch.h` | `spans[0].row0 == 0`, dense cover of `[0, M)` | spans of length one, first in order |

The first two are mutually exclusive on the same table. The third is what the plan specifies and
is the only one of the three that can express a pure-prefill step. Collapsing all three onto the
shared resolver is Phase 1a's job; this header is written to be deleted.

### Defensive checks, on in release

Per §4.3, a mis-planned batch must trap rather than produce a plausible wrong answer.
`plow_tb_view` validates the whole table once per consumer: no zero-length spans, monotone
`row0`, dense cover from 0, `kv_row0 + n_rows == kv_len`, `M <= capacity`. `plow_tb_row`
additionally traps when a derived position disagrees with the descriptor's own `positions[]` —
that disagreement *is* the class-C hazard.

---

## 3. The class-C conversion

`plans/unified-token-batch.md` §3 classifies `FlashPrefill`'s `i4 = q_pos0` as class C: one
packet scalar defines the base position and each row's position is *derived* from its index.
Correct for one request's contiguous chunk; silently wrong for a packed batch, because rows of
an earlier span get a later request's sequence length, causal bound and KV slot — and on AMD
nothing traps.

`d_flash_prefill` gains a `TB` template arm. When `TB`:

* the work list is `n_work = Σ_spans ceil(n_rows/FA_QT) · n_head · nsplit`, and each work index
  resolves to a span by a scalar walk (workgroup-uniform, one entry per admitted request);
* `n_q`, `n_kv`, `q_pos0`, the `Q`/`O`/`Opart`/`mlpart` row base and the `K`/`V` slot base all
  come from that span;
* the parameters `n_q`, `n_kv`, `q_pos0` are **not read**.

It is a flat list rather than a loop over spans (§5.2: "Avoid a workgroup loop that serially
visits every request when independent tiles can be scheduled"). The same grid covers every
request's tiles at once, so a short span does not get a whole grid to itself and there is no
barrier between requests.

`TB` defaults false and every added branch is `if constexpr`, so a `TB=false` instantiation is
the body it always was.

### A second class-C site the plan's Phase 2 list does not name

§8's Phase 2 bullets name only `FlashPrefill`'s `q_pos0`. The dense family has another, and §5.2
states its rule without §8 counting it:

> Cache writers use the span's physical slot and absolute KV position through the established
> slot/ring/page helpers — never addresses derived from packed row indices.

`HeadNormRope` (op 30) *is* the dense path's KV cache write — there is no separate write opcode.
Its `i3 = out_row0` is a packet scalar, host-patched per prefill chunk in
`crates/plowrt/src/exec/kvrow.rs`, and every row's write address is `out_row0 + t`. Under packing
that puts one request's K/V rows into another request's cache. It is converted in
`runtime/amd/op_norm.h` under the same axis. **A dense-GQA route that converts only `q_pos0` is
a silent wrong answer.**

### `FlashDecode` is class B, and the descriptor frees an operand

`FlashDecode` already has the per-row indirection the plan asks for (`t6 = decode_slot`,
`t5 = kv_len`). The descriptor only has to fill it, which the `TB` arm does from
`spans[b].slot` / `spans[b].kv_len`. That is worth more than tidiness on this path: on the dense
emit path `t6` is the **fp8-KV K-scale handle**, so a host-staged `decode_slot` and fp8 KV cannot
coexist. Sourcing the slot from the descriptor frees the operand.

### Where the attention partition comes from

`plow_tb_decode_spans()` — the count of leading spans of length one — is defined exactly once and
called by both the prefill and decode arms, so they cannot disagree about who owns a span.

§4.1 forbids the **host plan** from calling a span "decode" because its length is one (a final
prefill chunk can also have length one). It explicitly permits the other thing: "Kernel selection
derives query geometry from span lengths." This is that, and only that. A one-row span at
frontier `p` attends over `[0, p+1)` with its query at `p` under either kernel, so a final chunk
of length one landing on `FlashDecode` is correct, not merely tolerated.

`ndec == 0` and `ndec == n_spans` are both legal; the other side simply has no work. That is how
this route serves an intermediate pure-prefill step, which the mixed-step resolver refuses.

---

## 4. Class-A work: combined M

`exec_gemm` and `exec_mixed_gemm_glu` no longer loop over phase bands. One logical matmul over
every live row, with the kernel family selected from the total live count (`d_gemv`'s internal
row walk below `PLOW_GEMV_MAXM`, `d_gemm` above it). That is Phase 2's exit condition "no
phase-band GEMV/GEMM calls in the new path".

This is not numerically free, and the plan anticipates it (§9): *"Changing M may select different
kernels and reduction orders; record that rather than requiring unjustified universal bit
identity."* Recorded here:

* **At fixed M the result is bit-identical row for row** — one combined-M `d_gemm` over 133 rows
  equals four per-span `d_gemm`s, every bit (gate 5 below). A row's dot product does not depend
  on M.
* **A decode row's arithmetic changes.** It now reduces over K in the GEMM's order rather than
  the GEMV's, so pure-decode output is not bit-identical to the legacy route.

---

## 5. Acceptance gates

`runtime/tests/token_batch_dense_gfx942_test.hip`, run by
`scripts/token_batch_dense_test.sh` (inside `nix develop`, under a GPU lease).
gfx942 / MI300X, ROCm 7.14 (flake), hd=128, 8 query heads, 2 KV heads, 4 slots.

**31 gates, 31 passing.** They answer §9's class-C row, which asks for two things:

> same tokens as one span vs split across spans, at every span boundary → Identical results; the
> unconverted form is provably unreachable on this route.

| Gate | What it does | Result |
|---|---|---|
| 1 | One request's chunk vs the SAME tokens split at **every** interior boundary. 21 shapes (T ∈ {1,5,32,33,64,65,129} × frontier ∈ {0,7,128}), up to 128 boundaries each | bit-identical, `worst_diff=0` in all 21 |
| 2 | Packed vs isolated-per-request, row for row. Includes §6.1's worked example (2 decode + a completing prompt + an intermediate chunk, M=132, 4 spans), a one-token prompt, tile-straddling cold chunks, and three spans sharing one slot at different frontiers | `diff=0` in all 5 |
| 3 | Gate 2 is run with `q_pos0`/`n_q`/`n_kv` **poisoned to `0xDEADBEEF`** and still matches. The same batch through the unconverted kernel is measured too | poisoned run matches; unconverted differs in **32744 of 65536** outputs |
| 4 | `FlashDecode` slot+length from spans vs the host slot map, 4 requests at 4 slots and 4 lengths, `nsplit=4` | `diff=0` |
| 5 | One combined-M `d_gemm` (M=133) vs one per span | `diff=0` |
| 6 | Rows past `real_rows` keep their sentinel | `touched_pad=0` |
| 7 | Packed output vs a CPU **f64** attention reference | `rel = 2.70e-3` into bf16, tol 8e-3 |

Gate 3's second half is the point: without it, gate 1 and gate 2 would pass just as well on a
kernel where the conversion did nothing.

### Why gate 1 is exact, and where it stops being exact

At `nsplit == 1`, moving a span boundary changes which q-tile a row belongs to, so its tile's
`kv_end` changes — but only *upward*, and the extra KV tiles are wholly masked. A wholly masked
tile is an **exact** no-op on the online softmax: every `p[i]` is `-inf`, so `rmax = -inf`,
`mnew = m_st`, `corr = exp(0) = 1.0f` exactly, `pe = 0.0f` exactly, `l_st = l_st·1 + 0` and
`oacc *= 1.0f`. The real KV tiles are still visited in ascending `kv0`. Hence bit-identity.

At `nsplit > 1` the KV *partition* moves with `kv_end`, so the partial sums differ and the merge
is not associative in floating point. **Bit-identity is not claimed there**, and the route refuses
it: a `FlashPrefill` packet with `i7 != 1` traps. The emitter already pins `ns = 1` when packing
(`crates/devgen/src/lib.rs`, `dense_flash_split`), for the same reason — and
`crates/devgen/src/lib_tests/token_batch_contract.rs` pins that, with a negative control, so a
change that lost it fails a test instead of trapping on a device.

---

## 6. Object contract and the A/B

`scripts/build_gfx942.sh` gains the row `interp_tokbatch`, the mixed object's exact shape plus
`-DPLOW_TOKEN_BATCH=1`. Its contract block fails a build that drops any of the four markers.

Measured, gfx942, `scripts/build_gfx942.sh`:

```
object             vgpr agpr    lds  occ  scratch(ISA)  unmet
interp_mixed        466  210  64544    1        43/40    1141
interp_tokbatch     444  188  64544    1        43/40     825
```

22 VGPR and 22 AGPR lighter than the object it replaces, same LDS, same scratch. The phase-band
loop and the per-span flash loop were not free.

`.vgpr_spill_count` is not the check and was not trusted — those are `scripts/asm_audit.py`'s
counts from the ISA. Both objects' entry kernels issue 43 (static) / 40 (`_gq`) scratch ops while
their metadata reports zero spill; the token batch adds none.

**Object A/B, axis OFF.** `interp_mixed`, `interp_prefill`, `interp_flash`, `interp_decode` built
from the pre-change tree and from this one, same defines, same toolchain:

```
interp_mixed    text=same  changed_insn_lines=0  symtab_delta=__hip_cuid only
interp_prefill  text=same  changed_insn_lines=0  symtab_delta=__hip_cuid only
interp_flash    text=same  changed_insn_lines=0  symtab_delta=__hip_cuid only
interp_decode   text=same  changed_insn_lines=0  symtab_delta=__hip_cuid only
```

Disassembled `.text` is byte-identical. The only symbol difference is `__hip_cuid_*`, a hash of
the source path, which differs only because the base tree was exported to `/tmp`. Nothing shipped
moved — which matters here, since adding an inline function has moved register allocation on this
branch before with an identical opcode histogram.

---

## 7. Refusals

Everything the route cannot execute is refused with the capability named. Nothing falls back.

**At build.** `scripts/build_gfx942.sh`'s `interp_tokbatch` contract fails on a missing marker.
`interp.hip` `#error`s on any non-BF16-dense encoding — placed **after** the op headers, because
`#if UNDEFINED` is `0` and a guard written before the axis macros resolve reads as satisfied and
proves nothing.

**At load** (`crates/plowrt/src/exec/amd_token_batch.rs`), from `.symtab`, before the object
reaches a device:

| Marker | Claim |
|---|---|
| `plow_token_batch_1` | the axis was compiled at all |
| `plow_token_batch_dense_gqa_1` | the dense-GQA operator set is present |
| `plow_token_batch_combined_m_1` | projections run at combined M, no phase band |
| `plow_token_batch_span_attn_1` | attention bounds come from spans |

The kernel symbol is `plow_interp_tokbatch_<arch>` — deliberately **not**
`plow_interp_mixed_<arch>`. The two objects have the same shape and different dispatch; a shared
name would let a stale `interp_mixed_gq.elf` answer a token-batch lookup and serve the phase-band
route against a descriptor with no decode prefix. Symbol resolution refuses it instead.

**At admission**, one capability name per reason:
`token_batch_prefix_free_spans`, `token_batch_unsplit_attention`, `token_batch_fused_epilogue`.

**Armed is not fires.** One log line at load carries both as separate fields:

```
route=unified-token-batch/dense-gqa object=… kernel=… armed=true fires=false
reason="capability `token_batch_prefix_free_spans`: the plan puts N decode row(s) in a band
ahead of the span table …"
```

On this build that is the honest line. A route that reports a single `enabled` flag is how three
campaigns on this branch measured "no effect" from something that never fired.

---

## 8. What is not done

* **Nothing serves end to end on this route.** Two independent reasons:
  1. **No dense-GQA checkpoint on this host.** `/workspace/models` holds GLM-5.3 (MLA + MoE +
     DSA), DeepSeek-V4-Flash and Kimi (MLA), and Gemma-4 (windowed + softcap → Phase 3).
     `dflash` is `model_type: qwen3` dense GQA but is a *draft* model with no `embed_tokens` and
     no `lm_head`; `eagle3` is `model_type: llama` but is a `midlayer` draft head. Neither is a
     standalone LM. Gemma-4 was **not** substituted — it is Phase 3 by the plan's own table.
     Everything above is synthetic shapes plus a CPU f64 reference, and is labelled as such.
  2. **The planner still emits a decode band.** `plow_asset::mixed_step::plan_into` produces
     spans starting at `decode_rows`; §4.4 wants spans covering `[0, M)`. That is Phase 1a and is
     refused by name rather than worked around.
* **The emitter still does not emit the body.** See §9.
* **The output segment (§6) is not here.** `RowGather`, the compact tail and the sample-row
  contract are Phase 1c.
* **No performance number.** The route has not fired, so there is nothing to measure. The object
  A/B and the register table are resource facts, not throughput claims.

---

## 9. The emitter seam, and why it is bigger than a Phase 2 bullet

Phase 2's third bullet is "Emit body variants directly from the emitter, preserving graph
dependencies and encodings." On AMD that is not a variant of an existing emitted body, because
**there is no emitted mixed body at all**:

* `crates/devgen` contains no mixed/token-batch emit path — zero hits for `mixed` or
  `PLOW_MIXED`, and `plowc` explicitly *rejects* `--mixed-rows` / `--mixed-object`
  (`crates/plowc/src/main.rs:2260`).
* AMD's mixed programs are **synthesized at model load** by
  `crates/plowrt/src/exec/mixed_program.rs`, which rewrites an ordinary prefill program:
  it folds gate/up/GLU into `GemmGlu`, splits `NormResidualNorm`, injects
  `FlashDecode`+`FlashMerge` ahead of a retargeted `FlashPrefill`, rewrites `Gemv` to `Gemm`, and
  rebuilds the result as a **strict linear chain** (`mixed_program.rs:474-491`).
* The compiler-emitted `mixed_step` packet section (`plow_asset::mixed_step::Manifest`) has no
  in-repo producer and is consumed only by the CUDA path.

So "emit the body variant" on AMD means writing AMD's *first* mixed-body emitter, and the
"preserving graph dependencies" clause is the substance of it: the load-time synthesis throws the
DAG away, which is precisely what the bullet objects to. That is a larger item than the class-C
conversion Phase 2 is scoped around, and it is independent of it.

The seam, for whoever picks it up:

1. `DenseGqaEmitter` (`crates/devgen/src/lib.rs:6181-6353`) gains a third `emit_*` method beside
   `emit_prefill` / `emit_decode`.
2. `emit_dense_gqa`'s prefill loop (`lib.rs:7715-7772`) emits a sibling program per bucket, the
   way `mla.rs:7025-7038` doubles its `pf_plan`.
3. `emit_phase` (`lib.rs:3277-5797`) needs a mode that emits, per layer, the decode attention
   pair **and** the prefill attention in one program — the one genuinely new thing, since
   `emit_phase` today emits either `Prefill` or `Decode`.
4. The four immediates that must stop being host-patched are exactly the ones
   `crates/plowrt/src/exec/kvrow.rs:334-339` patches today: `HeadNormRope.i[3]`,
   `FlashPrefill.i[4]`, `FlashPrefill.i[1]`, and `FlashDecode.t[6]`. All four are already
   descriptor-sourced on the device side by this work, so the emitter's job is to stop writing
   them, not to write them differently.
5. `ns = 1` and the fused epilogue are already the packed-emit behaviour (`dense_flash_split`),
   which is what the device arm requires and what
   `lib_tests/token_batch_contract.rs` now pins.

---

## 10. Files

| File | Role |
|---|---|
| `runtime/amd/token_batch.h` | descriptor adapter, row resolver, span-table validation, attention partition |
| `runtime/amd/op_attention.h` | `d_flash_prefill<…, TB>` flat per-span schedule; `d_flash_decode<…, TB>` span slot/length |
| `runtime/amd/op_norm.h` | `d_headnorm_rope` span-addressed KV write (the second class-C site) |
| `runtime/amd/interp.hip` | `PLOW_TOKEN_BATCH` dispatch, combined-M projections, markers, refusals, kernel symbol |
| `runtime/tests/token_batch_dense_gfx942_test.hip` | the seven gates |
| `scripts/token_batch_dense_test.sh` | build + lease + run |
| `scripts/build_gfx942.sh` | `interp_tokbatch` row and its object contract |
| `crates/plowrt/src/exec/amd_token_batch.rs` | load/admission capability gate, armed-vs-fires log |
