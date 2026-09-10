# The unified token batch

One packed activation matrix per step containing all scheduled input tokens — decode and
prefill, from any number of requests — and one compact terminal segment that produces exactly
the next-token distributions the step owes.

This document describes **what has landed** (the shared foundation: contract, planner, ABI,
opcode, CPU tail) and, where it matters, what has deliberately *not*. The design it implements
is `plans/unified-token-batch.md`; that plan is a design, this is a record of the parts that are
now code and tests. Where the two disagree, the disagreement is called out here.

## Contents

Runtime selection defaults on. `--token-batch=false` or `PLOW_TOKEN_BATCH=0` disables it;
explicit `--fusion` takes precedence. Selection still requires a gfx942 dense BF16 program,
matching object markers, multiple slots, unsplit attention and a fused attention epilogue.
Tensor parallelism, prefix-cache mode and unsupported programs use ordinary execution.
CUDA has no token-batch executor yet, so the default does not enable token batching on H100.

Startup logs distinguish object `armed` from executor `ready` and always report `fires=false`.
The first successful device dispatch reports `fires=true`. These capability and lifecycle
checks do not establish end-to-end performance or production qualification on an untested GPU.

Three documents, folded into one on 2026-09-08 because they were always read together and
their cross-references were the only thing tying them.

* **Part I** (below) — the shared foundation: contract, planner, ABI, `RowGather`, CPU tail.
* **Part II** — the gfx942 dense-GQA device route: what it is, its gates, its refusals, and
  the GEMM tile that decided its long-prompt behaviour.
* **Part III** — the row-identity classification of all 154 opcodes, and what it says about
  runtime fusion v1.

Each part keeps its own section numbering, so a reference like "Part II §4" resolves.

## 1. The three counts

Backend- and model-independent. A backend may **cap** them; nothing may redefine them.

| | meaning |
|---|---|
| `row_capacity` | compiled/padded allocation capacity — the launched `T` |
| `real_rows = M` | scheduled input tokens this step |
| `sample_rows = S` | hidden rows whose next-token distributions are needed |

`S = decode_requests + prompts_completed_this_step`. An intermediate prompt chunk contributes
tokens to `M` and **zero** to `S`.

**`S = 0` means no output segment.** It does not mean one row. The legacy
`n_batch == 0 means one row` convention is not reused here, and `plow_asset::token_batch` refuses
a zero-row selection stage rather than letting argmax read zero as one — which would deliver a
token to a request that asked for none.

## 2. What is new versus `mixed_step`

`crates/plow-asset/src/mixed_step.rs` already carries a backend-neutral mixed decode/prefill
plan. The token batch is a second, versioned contract beside it, not a replacement — the mixed
v1 route (`PLOW_FUSION`, `exec/amd_mixed_step.rs`, `exec/mixed_program.rs`) is untouched and
keeps its protocol until measurement replaces it per `(backend, family)` pair.

Two differences are load-bearing.

**No decode prefix.** `mixed_step`'s plan splits rows into a decode band and a prefill band, and
`runtime/common/mixed_step.h`'s resolver has a matching special case for rows below the first
span, resolved from a separate `decode_slots[]` image. Here **every row belongs to a span**,
decode spans included. One row source, one binary search, no second table to keep in step.

A decode request is a span of length one. **The converse is not true**: a final prefill chunk
can also have length one, so length never decides a span's phase. The planner carries an
explicit `Phase`, and `Phase::Prefill` with `end == prompt_len` is what makes a span sample.

**Padding belongs to nobody.** `mixed_step` extends the last owner's positions across the pad
and then has to bound that against `max_ctx` ("padding exceeds physical context"). Here padding
rows carry `active = 0`, id `0` and position `0`, belong to no span, and are inert. That removes
a check and the class of bug it guarded.

The mask polarity is the ISA's existing one: **`active[row] != 0` means the row is LIVE**, as
the GDN family's `active[B]` already means (ops 136–145). This is the *opposite* of
`PlowProgram::prefill_parked`, which is `1` for parked. The two are different fields of
different descriptors and are never both consulted for one row, but getting the polarity
backwards writes KV for padding and does not trap, so it is spelled out in `dev_isa.h` too.

## 3. The device descriptor and the ABI change

```c
typedef struct {
    uint32_t version;      /* PLOW_TOKEN_BATCH_VERSION */
    uint32_t row_capacity;
    uint32_t real_rows;    /* M */
    uint32_t sample_rows;  /* S; 0 = no output segment */
    uint32_t n_spans;      /* R */
    uint32_t flags;        /* reserved, must be 0 */
    uint32_t _pad0, _pad1;
    const PlowPrefillSpan* spans;           /* [n_spans], dense cover of [0, M) */
    const uint32_t*        input_ids;       /* [row_capacity] */
    const uint32_t*        positions;       /* [row_capacity] */
    const uint32_t*        active;          /* [row_capacity], 1 = live */
    const uint32_t*        sample_rows_idx; /* [sample_rows] */
} PlowTokenBatch;                            /* 72 bytes */
```

`PlowPrefillSpan` is **reused verbatim**, `state_slot` and `program` included — the D-class
families need the state slot, and a second span struct would need its own ABI lock and would
drift from this one. What changes is coverage, not layout.

`_pad0`/`_pad1` are explicit because the struct is memcpy'd to the device. An implicit gap ships
whatever the constructing stack frame held; that is exactly how a `TokenBody` field gap made
emitted packets non-reproducible in release for months. `dev_abi.rs` locks the size and every
offset, on both sides.

`PlowProgram` gains **one appended pointer**:

```c
    const PlowTokenBatch*  token_batch;   /* NULL retains every existing path */
```

**168 → 176 bytes.** Every existing field keeps its offset. The growth is the point:
`AmdEngine::load` derives its kernarg-segment check from `size_of::<DevProgram>()`, so an object
built before this field is **refused by name** rather than loading and then reading its grid
dimensions eight bytes off. Every loader that checks the kernarg segment inherits that refusal
for free; the C harnesses in `runtime/tests/` pass `sizeof(pr)` and zero the struct, so they
need no change.

## 4. The shared row resolver

`runtime/common/token_batch.h` resolves `row -> (span, local_row, slot, state_slot, position,
active)`. It is compiled:

* as **device** code by HIP and CUDA, under the same guards `mixed_step.h` uses, where a
  mis-planned batch **traps**;
* as **host** code through `runtime/cpu/dev/token_batch.c`, where the same predicates are
  returned as `PLOW_TB_E_*` codes so a host can refuse before launch and name what failed.

`plow_asset::token_batch` carries a **Rust twin** — the planner has to validate before any
device sees a plan — whose `Refusal::code()` mirrors the C codes exactly.
`crates/plowrt/tests/token_batch_resolver.rs` runs both over identical span tables, including
every malformed one, and compares acceptance, the exact code, and every resolved field. Two
resolvers that are never compared are two resolvers.

The defensive checks (dense cover, monotone `row0`, position agreement, park mask, `kv_len`
arithmetic, sample-row liveness) stay **on in release** on both sides. They are cheap next to
any packet that calls the resolver, and each one turns a mis-planned batch into a trap instead
of a plausible wrong answer — which on AMD, where the interpreter's dispatch `default:` neither
writes nor traps, is the actual failure mode rather than a hypothetical.

## 5. `RowGather` — opcode 154

```
t0=out(bf16 [S][H])  t1=x(bf16 [M][H])  t2=rows(u32 [S])
i0=S  i1=H  i2=M
out[s][h] = x[rows[s]][h]
```

**It is not `Embed`.** `Embed` gathers rows of the *embedding table* by token id and scales by
`sqrt(hidden)`; this gathers rows of a *hidden activation* by packed row index and copies them
verbatim, because the final RMSNorm that follows must see exactly the bytes the body wrote.
There was no existing generic hidden-row gather to repurpose.

`rows[s] >= M` **traps** on every backend and `abort()`s on the CPU golden tier. A clamped index
is not a degraded answer: it is another request's hidden row, and the token it produces is
fluent and wrong.

Coverage today: an arm in `runtime/amd/interp.hip`, an arm in `runtime/nvidia/interp_sm120.cu`
(which `interp_sm90a.cu` includes — see §9), and a golden kernel in the CPU op table registered
in `runtime/cpu/dev/golden/control.c`, so `plow_cpu_has` reports it and `KernelTable::resolve`
names it if it is ever absent.

Two object capability markers, deliberately **not** one symbol:

| symbol | claim |
|---|---|
| `plow_token_batch_abi_1` | the object was compiled against the `PlowProgram` carrying `token_batch` |
| `plow_row_gather_1` | `PLOW_DOP_ROW_GATHER` has a real dispatch arm |

The first says the route is **armed**; the second says a terminal segment **can fire**. Only the
second licenses a measurement of one. Neither claims any descriptor-aware *math* arm — those are
their own markers, because an object that parses the descriptor and then runs a packet-scalar
kernel is precisely the silent wrong answer this contract exists to prevent.

## 6. Operator row-identity classes

`crates/packet/src/rowclass.rs` is the design's §3 classification as code. Packing rows from
several requests into one matrix is safe for an operator exactly when the operator cannot
confuse one request's rows for another's.

| class | how a row's identity/position is learned | effect of packing |
|---|---|---|
| **A** | it is not — per row or per element, no position | none; live `M` is the only new input |
| **B** | an explicit per-row table already selects slot/state/validity | none, once the descriptor fills that table |
| **C** | one immediate or scalar defines a base; each row *derives* from its index | **silently wrong** |
| **D** | state is an operand with no request axis at all | cannot express more than one request per launch |

The match is **exhaustive**: adding an opcode is a compile error until somebody classifies it.
`class_of_op` reports `C` for any opcode value this build cannot name, because `C` is the class
that gets refused.

The class is per **opcode**, not per instruction. Several opcodes carry both a per-row form and
a packet-scalar form behind an immediate (`HeadNormRope`'s `n_batch_kv`, `QwenHeadNormRope`'s
`prefill`, the KDA family's `bstride`). Those take the conservative class, and "conservative"
differs by pair: C over B refuses a program (a false C costs a conversion, a false B costs a
wrong token), while D over B only *bounds* a step to one span per layer (a false D costs a
scheduling limit).

`token_batch::Capabilities::refuse_program` derives the load-time refusal from this table and
**names the capability**, e.g.

```
gfx942.PLOW_DOP_FLASH_PREFILL — class C: derives every row's position from a packet scalar,
and no converted form is declared for this target
```

## 7. The terminal segment

```text
body final residual X[M,H]
    -> RowGather(sample_input_rows[S])   -> selected_x[S,H]
    -> the model's existing final RMSNorm -> selected_norm[S,H]
    -> LM head with M = S                -> logits[S,V_local]
    -> [SoftCap]                          (Gemma family)
    -> selection stage                   -> sampled_ids[S]
```

Gathering *before* the final norm is valid because final RMSNorm is per token, and it avoids
normalizing rows nobody asked for. The head is selected by `S`, not `M`.

`crates/plowrt/tests/token_batch_tail.rs` builds this as a real `LoadedProgram` and runs it
through `exec::cpu::interp` on the persistent worker pool with the counter DAG — not by calling
kernels in a loop. Two comparisons:

1. **Isolated vs packed.** The same selected rows run one at a time at `S = 1` give the same ids
   as one compact `S`-row run. This is the mapping check.
2. **A host oracle.** An independent Rust implementation of the same chain with the same bf16
   rounding points, including the bf16 round on the logit before the argmax and the
   lowest-index tie rule.

Doing this on CPU first is what makes the row-selection contract testable with no hardware, and
it gives every later backend a reference.

## 8. Commit, delivery and failure

`TokenBatchStaging` (`crates/plowrt/src/exec/mixed_step_staging.rs`, beside the mixed-step
filler so there is exactly **one** host staging filler) reads the committed frontier *and* the
slot generation and changes neither. The commit re-checks **both, for every span, before
mutating anything**, so a slot recycled while the step was in flight is refused rather than
handed another request's output, and a refusal leaves host state exactly as it was.

Frontiers advance only for rows this step consumed as **input**. A sampled token advances
nothing until it is fed back.

Delivery is by **logical request**, in sample order, from `SampleOwner` — never by row number,
because packed rows, physical slots and compact output rows are three different numberings.

If the body succeeds but the tail fails, the caller discards and publishes nothing. Physical KV
may already have changed; that is a fault-path problem for the affected slots, and this type
does not pretend to roll it back.

## 9. Things the plan gets wrong or under-specifies

Recorded because the plan is a design, not a proof.

* **NVIDIA is one translation unit for opcode arms, not two.** The plan says an opcode needs
  "arms in **both** `interp_sm120.cu` and `interp_sm90a.cu`". `interp_sm90a.cu` is a 42-line
  wrapper that renames the public symbols and then `#include`s `interp_sm120.cu`. There is one
  switch. The plan's underlying concern is still right — the two *images* are separate by design
  so one cannot be mistaken for the other — but it does not translate into two arms.
* **`exec/cpu/ffi.rs` has no kernarg to validate.** The plan lists it among "every backend's
  kernel-argument validation" for the ABI change. The CPU interpreter never constructs a
  `PlowProgram`: it walks the stream in Rust and calls kernels with `(inst, slice, nblk,
  tensors, ctx)`. The CPU's equivalent duty is `plow_cpu_has` / `KernelTable::resolve`, which is
  where the refusal was put instead.
* **The `active` polarity inverts `prefill_parked`.** The plan says to adopt the ISA's existing
  `active[B]` convention (1 = live) and separately reuses `PrefillSpan` from the packed-prefill
  descriptor, whose companion mask is `prefill_parked` (1 = parked). Both are right in
  isolation; together they put two opposite conventions one struct apart. Kept as the plan says,
  documented loudly in `dev_isa.h`.
* **"Sample rows are the last scheduled row" needs a phase, not a length.** The plan's §4.4 says
  a sample index "is exactly its owner's last scheduled row", and §4.1 warns not to classify a
  span as decode because its length is one — but does not say what *does* decide. The planner
  needs `prompt_len` per request to answer it, which the plan's field table does not list.
* **`S = 0` and the sample-row pointer.** The plan says `S` may be zero and that the descriptor
  carries a selected-row pointer, without saying what that pointer is when `S = 0`. The
  descriptor builder here refuses a non-null `sample_rows_idx` at `S = 0` and vice versa: a
  stale pointer beside a zero count is how a zero-output step still reads somebody's row list.
* **Adding a dispatch arm is not free on AMD.** The plan treats opcode coverage as a pure
  correctness question. One extra `case` in `plow_exec` moved `interp_prefill`'s spill from
  1698 to 1770 scratch ops across the object — all 72 of them inside `plow_exec` itself, none in
  the outlined gather body. Register pressure in the per-packet dispatcher is a real cost of
  ISA growth and belongs in the plan's accounting.

## 10. Deliberately deferred

Everything past the shared foundation. In particular:

* **No descriptor-aware math arms anywhere in the SHARED contract.** No interpreter reads
  `PlowProgram::token_batch` — `runtime/amd/token_batch.h` still synthesises its view from the
  packed-prefill tail (`PLOW_TOKEN_BATCH_DESC` defaults to `0`), so the descriptor's
  `positions[]` cross-check is present and dead. The AMD dense-GQA arms themselves landed
  separately; see Part II below.
* **`Capabilities::converted_c` is no longer empty.** `Capabilities::amd_dense_gqa` declares the
  two conversions that family actually has — `FlashPrefill`'s `q_pos0` and `HeadNormRope`'s
  `i3 = out_row0`, which is the dense path's KV cache write — and `refuse_program` runs at load,
  per prefill bucket, naming the capability it refuses on.
* **The route serves on (gfx942, Gemma-4 31B).** `ServeEngine` carries a token-batch surface
  whose defaults decline, `serve/mux.rs` has an arm ahead of the mixed one, and
  `mixed_step::SpanCover::PrefixFree` gives it a span table covering `[0, M)`. Measurement,
  limits and the identity result are in
  [docs/amd/gemma4-31b-mi300x.md Appendix A](../amd/gemma4-31b-mi300x.md).
  Still absent there: the compact terminal segment. `RowGather` has an arm and is unused — the
  route samples rows of the BODY, which works only because the planner puts every sampled row at
  the front of the batch.
* **No blob/manifest section.** `ProgramRole` and the capability check exist; a `token_batch`
  asset section binding body and output payloads does not.
* **NVIDIA and CPU objects are not rebuilt** against the new `PlowProgram` here — no `nvcc` on
  this host, and the CPU dev library is rebuilt by cargo on every build so it needs no separate
  step. The gfx942 objects are rebuilt; gfx950 is not (no gfx950 hardware or lease in play, and
  the build script is a separate arch recipe).

---

# Part II: Unified token batch: dense GQA on AMD

*Folded in from `docs/arch/17-unified-token-batch.md (Part II)`. Section numbers below are this part's own.*
Status: the device route is implemented, built and gated on gfx942.

**Superseded on both counts as of 2026-09-08.** The planner no longer hands the device a decode
band (`mixed_step::SpanCover::PrefixFree`), the route is wired into serving, and it fires on
Gemma-4 31B — which is windowed + softcap dense GQA, so it is a **Phase 3** enablement of these
Phase 2 arms rather than the Phase 2 target this document was scoped around. What it took, what
it is measured to be worth, and the two limits it carries are in
[../amd/gemma4-31b-mi300x.md Appendix A](../amd/gemma4-31b-mi300x.md). §8 below is kept
as written, because what it records about the state of the tree at the time is still the reason
the work was shaped this way.

One correction this document owes its own §5: the 31 gates run at `HD=128`, `window = 0` and
`PLOW_KV_MASK_NONE`, so **the window and KV-ring code in the `TB` arms had no coverage**. They
were correct — Gemma-4 31B runs 50 of 60 layers windowed at 1024 with a 16384-entry ring and its
packed output is identical to its isolated output — but that was luck until it was measured.

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

---

# Part III: Operator row-identity classes, and the verdict on runtime fusion v1

*Folded in from `docs/arch/17-unified-token-batch.md (Part III)`. Section numbers below are this part's own.*
Packing rows from several requests into one activation matrix is safe for an
operator exactly when the operator cannot confuse one request's rows for
another's. This document is the answer to that question for all 154 opcodes of
the device ISA, reproduced from the tool rather than written by hand, plus what
the classification says about the fusion that ships today.

It implements Phase 1d of `plans/unified-token-batch.md` and carries that plan's
§3 table. Where this document and the plan disagree, the disagreements are listed
in [§5](#5-corrections-to-the-plans-3), with the code that settles each one.

## 1. Running it

```
plowrt op-audit --table                       # every opcode, static class
plowrt op-audit <blob-or-dir>                 # one blob's programs
plowrt op-audit <blob> --program 1 --format json
```

Static and offline, like `disasm`: the blob is a file, and reading it needs no
GPU, no driver and no features. The exit status is the verdict — `0` when every
program can be executed as one packed token batch, `1` when any opcode is
refused — so the audit can gate the enablement of a (family, backend) pair rather
than only being read.

The classification lives in `crates/plowrt/src/opaudit.rs`. **It belongs in
`crates/packet`, beside `slots.rs`**, so `devgen` can stamp the declared classes
into a blob's auxiliary metadata at emit time (plan §4.2) and a loader can check
them without re-deriving. It is in `plowrt` only because the audit had to exist
before the ISA could own it, and because `packet` is being changed concurrently.
Nothing in it reads anything `packet` does not already export, so the move is a
file move plus a re-export.

`classify` is an exhaustive `match` over `DevOp`. Adding an opcode to
`packet::dev` breaks that build until someone classifies it. That is the only
mechanism that keeps the plan's rule — *an opcode with no classification is
class C and is refused* — from decaying into *an opcode with no classification is
absent from the table*.

## 2. The classes

The four classes are the plan's §3, unchanged.

| Class | How the operator learns a row's identity/position | Effect of packing |
|---|---|---|
| **A — row-agnostic** | It does not. Per row or per element, no cross-row coupling, no position. | None. |
| **B — per-row indirection already present** | An explicit per-row table selects slot, state or validity. | None, once the descriptor fills it. |
| **C — packet-scalar position base** | One immediate or scalar tensor defines the base; each row's position is *derived* from its index. | **Silently wrong.** Rows of earlier spans get another request's sequence length. |
| **D — per-sequence carried state** | State is an operand whose shape has no request axis. | Cannot express more than one request per launch. |

The tool adds one axis §3 does not have, because §3's class A ("live `M` is the
only new input") assumes the operator *has* a live row count and 25 of them do
not:

| Extent | Meaning |
|---|---|
| `elementwise` | The count enters as a flat element count (`i0=n`); packing scales it. |
| `row-counted` | An explicit row / `M` / `T` / `n_batch` operand. |
| `single-row` | **No row-count operand at all.** The packet describes exactly one token. |

A `single-row` operator is genuinely class A — its arithmetic is row-agnostic —
but a packed batch cannot be expressed in one packet, and raising `M` is not
available because there is no `M`. The emitter must select the operator's `*Pf` /
T-row twin instead. The tool reports that as disposition `use-row-form`, and
refuses the packet. See [§5.2](#52-class-a-is-not-one-thing).

Dispositions: `ready` and `descriptor-fills` are packable; `use-row-form`,
`needs-conversion`, `per-span-launch` and `refuse` are not.

## 3. The ISA, as the tool reports it

`plowrt op-audit --table`, 154 opcodes:

| | count |
|---|---|
| class A | 101 |
| class B | 25 |
| class C | 17 |
| class D | 11 |

| disposition | count |
|---|---|
| `ready` | 74 |
| `descriptor-fills` | 25 |
| `use-row-form` | 25 |
| `needs-conversion` | 15 |
| `per-span-launch` | 11 |
| `refuse` | 4 |

**Class C (17).** `HeadNormRope` (3), `FlashPrefill` (11), `HeadNormRopeFp8` (37),
`FlashPrefillFp8` (39), `FlashMlaPrefill` (51), `AttnSelect` (53),
`FlashGatherPrefill` (55), `LayerNorm` (60), `Mamba2Scan` (90),
`FlashMlaPrefillFp8` (110), `IndexScorePf` (117), `IndexSelectPf` (118),
`IndexUnionPf` (119), `FlashMlaMaterializedPrefill` (127), `DsaPoolCompress`
(130), `DsaPoolExpand` (131), `QwenHeadNormRope` (142).

Four of those are class C only in one operand mode — see [§5.1](#51-four-opcodes-carry-two-addressing-modes-and-3-carries-three).

**Class D (11).** `KdaConv` (88), `KdaStateStep` (102), `KdaConv3` (111),
`KdaStateStepG` (112), `KdaChunkPrepare`/`Intra`/`Wu`/`Carry` (121-124),
`DsaPoolStash` (132), `QwenGdnConvPrefill` (143), `QwenGdnPrefill` (146).

**`use-row-form` (25).** Every decode-shaped MoE operator (`MoeRouter` 40,
`MoeExpertGlu`/`Down` 41-42, `MoeCombine` 43, the block-fp8 and Gemma twins
45-49 and 61-72, `MoeRouterTopk` 56), plus `DenseGluFp8Blk` (47), `IndexSelect`
(59), `GemvArgmax` (80) and `XReduceAddNorm` (116).

**`refuse` (4).** `XReduceScatter` (25) and `XFlashMerge` (27) have no built
body. `AttnSelect` (53) and `Mamba2Scan` (90) have no usable ISA operand spec and
are therefore class C and refused, per the plan's own rule. See
[§6](#6-corrections-to-the-isas-own-metadata).

## 4. The class table, per family

One emitted program per family, per the plan's Phase-1d exit condition. Cases
marked *(measured)* are the shipped blobs on this host; the rest are miniature
`config.json` fixtures driven through `devgen::run` by
`crates/plowrt/tests/op_audit_families.rs` — no checkpoint, no GPU.

### 4.1 Dense GQA — Llama 3 / Qwen 3

Prefill, `T=128`: `A=31 B=0 C=8 D=0` over 10 opcodes. Decode, `T=1`:
`A=19 B=2 C=6 D=0` over 11 opcodes.

| class | opcodes |
|---|---|
| A | `RmsNorm`, `Residual`, `Glu`, `Embed`, `GemmSmall`, `Gemv`, `GemvGlu`, `GemvQkv`, `AddNorm`, `FlashMerge`, `Argmax`, `ArgmaxFin` |
| B | `FlashDecode` (`t5=kv_len[b]`, `t6=decode_slot[b]`) |
| C | `FlashPrefill` (`i4=q_pos0`), `HeadNormRope` (`i3=out_row0`, `i6=0`) |
| D | — |

§2.2 calls this family "A + one C". It is A + **two** C: `FlashPrefill` as the
plan says, and the KV writer, which §3 does not name. Everything else is already
row-agnostic or per-row. It is still the cheapest bring-up vehicle.

### 4.2 Windowed + softcap — Gemma 4 dense *(measured: `build-gemma31/assets-final-plain`)*

Prefill, `T=128`: `A=776 B=0 C=240 D=0` over 13 opcodes. Decode, `T=1`:
`A=436 B=60 C=180 D=0` over 12 opcodes.

Identical class set to §4.1. The window is a static per-layer immediate
(`FlashPrefill i5`, `FlashDecode i4`), not a row property, so it adds no class;
`SoftCap` is elementwise class A. Refused for `FlashPrefill` x60 and
`HeadNormRope` x180 in prefill, and for `HeadNormRope` x180 alone in decode.

### 4.3 MLA + MoE — GLM-5.3 *(measured: `build-glm53/tp4-long`, TP4)*

Prefill, `T=2048`: `A=1397 B=459 C=234 D=0` over 19 opcodes. Decode, `T=1`:
`A=2211 B=234 C=0 D=0` over 20 opcodes.

| class | opcodes |
|---|---|
| A | `RmsNorm`, `Embed`, `Gemm`/`GemmSmall`/`GemmMed`/`GemmWide`/`GemmGlu`, `Gemv`, `GemvGlu`, `GemvQkv`, `GemvFp8Blk`, `AddNorm`, `Residual`, `MlaMergeFold`, `MoeRouterTopkPf`, `MoeCombinePf`, `XReduce`, `XReduceTwoShot`, `XArgmaxFin`, `Argmax` |
| B | `MoeAlignPf`, `MoeGroupGluPf`, `MoeGroupDownPf`, `FlashMlaDecode`, `HeadNormRope` *(decode: `i6 != 0`)* |
| C | `FlashMlaPrefill`, `HeadNormRope` *(prefill: `i6 == 0`)* |
| D | — |

Two results worth stating plainly.

**The grouped MoE prefill chain is class B, not class C.** `MoeAlignPf` builds
`row_token` / `row_partidx` / `row_gate`, and `MoeGroupGluPf` / `MoeGroupDownPf`
read them per gathered row. The descriptor fills a table that already exists,
which is exactly §5.3's claim, confirmed on the shipped blob.

**The decode program has zero class-C instructions and is still refused.** All
six blockers are class-A operators with no row-count operand: `MoeCombine` (75),
`MoeExpertGluFp8Blk` (600), `MoeExpertDownFp8Blk` (600), `DenseGluFp8Blk` (3),
`MoeRouterTopk` (75), `XReduceAddNorm` (78) — 1431 of 2445 instructions. Their
row-forms exist (`MoeCombinePf`, `MoeGroupGluPf`/`MoeGroupDownPf`,
`MoeRouterTopkPf`, `XReduceTwoShot`'s fused-consumer arm) and the prefill program
already uses them, but `DenseGluFp8Blk` has no 1:1 T-row twin — the shared expert
lowers to `GemmGlu` plus block-fp8 GEMMs at `T > 1`. Enabling MLA+MoE therefore
means *re-emitting the decode path against the prefill operator family*, which is
real work §3's "A needs nothing" does not describe.

Also: the control carries no `Index*` or `Dsa*` opcode at all, confirming §2.3's
"DSA disabled in the documented control". The DSA chain's four class-C operators
(117, 118, 119, 131) are unreached by this blob and are a separate axis.

### 4.4 MXFP4 + sinks — GPT-OSS

Prefill, `T=128`: `A=25 B=6 C=8 D=0` over 15 opcodes. Decode, `T=1`:
`A=25 B=2 C=6 D=0` over 14 opcodes.

| class | opcodes |
|---|---|
| A | `RmsNorm`, `Embed`, `Gemm`, `Gemv`, `GemvQkv`, `AddNorm`, `FlashMerge`, `MoeRouterTopkPf`, `MoeCombinePf`, `MoeGluMx`, `MoeDownMx`, `Argmax`, `ArgmaxFin` |
| B | `MoeAlignPf`, `MoeGluMxPf`, `MoeDownMxPf`, `FlashDecode` |
| C | `FlashPrefill`, `HeadNormRope` |
| D | — |

The sink fold stays class A: `FlashMerge`'s `t3=sinks` is one unscaled logit per
**head** with no value row, so it is row-independent and composes with packing
unchanged — the plan's §5.2 claim, confirmed on an emitted program. MXFP4 expert
grouping adds no class the block-fp8 family did not already have: the decode
forms carry `i6=n_batch`, the prefill forms carry `t5=row_token` /
`t6=row_partidx`.

### 4.5 Recurrent — Qwen 3.5 / GDN

Decode, `T=1`: `A=48 B=24 C=0 D=0` over 15 opcodes. **PACKABLE.**

| class | opcodes |
|---|---|
| A | `Residual`, `Glu`, `Embed`, `Gemv`, `FlashMerge`, `Argmax`, `ArgmaxFin` |
| B | `QwenGdnConv`, `QwenGdnStep`, `QwenGatedNorm`, `QwenQGateSplit`, `QwenSigmoidGate`, `QwenRmsNorm`, `QwenHeadNormRope`, `FlashDecode` |
| C | — |
| D | — |

This is the plan's §5.4 claim, confirmed: the GDN **decode** side is already
packed-friendly, entirely through the ISA's existing `active[B]` convention, and
this is the only family whose decode program the audit passes outright. The
class-D limit is on the *prefill* operators — `QwenGdnPrefill`'s
`state`/`outstate` `[1,HV,V,K]` and `QwenGdnConvPrefill`'s `history[1,C,W-1]` —
which this hybrid decode blob does not contain.

Note `QwenHeadNormRope` in decode mode (`i6=prefill == 0`) is class B, where the
dense families' `HeadNormRope` in legacy mode is class C. The GDN emitter got the
KV writer right; the dense one has not been asked to.

## 5. Corrections to the plan's §3

The plan is a design, not a proof. These are the places the ISA does not match
it, each with the code that settles it. Every one is pinned by a test in
`crates/plowrt/src/opaudit_tests.rs`.

### 5.1 Four opcodes carry two addressing modes, and §3 names none of them

§3 classifies opcodes. Four opcodes are class B in one operand mode and class C
in another, and the shipped blobs use *different modes for the same opcode*:

| opcode | class C when | class B when |
|---|---|---|
| `HeadNormRope` (3), `HeadNormRopeFp8` (37) | `i6=n_batch_kv == 0` — legacy `out_row0 + t`, one host-patched position per step | `i6 != 0` — row `t` is sequence `t`, written at `pos[t]` into its own batch-major ring |
| `QwenHeadNormRope` (142) | `i6=prefill == 1` — one selected KV slot for all rows | `i6 == 0` — one slot per row, `t5=pos`, `t6=active` |
| `LayerNorm` (60) | `i3=out_row0 != 0` — packet-scalar row base into the DSA index-key cache | `i3 == 0` — plain norm, no cache write |
| `KdaConv3` (111) | *(D)* `j0=bstride == 0` — serial single-sequence | `j0 != 0` — "the `T` rows are `B` separate sequences" (`op_kda.h:1041`), `j1=parked` masks writes |

This is not academic. **GLM-5.3's decode program emits `HeadNormRope` with
`i6 != 0` (class B); Gemma-4-31B's decode program emits the same opcode with
`i6 == 0` (class C), 180 times.** A per-opcode capability check cannot tell them
apart. The tool therefore has two entry points: `classify(op)` returns the
stricter reading, which is what a load-time check must use because it has no
instruction to read; `refine(op, inst)` reads the immediate, which is what an
audit of an emitted program uses. A program is only clear if *every* site of an
operand-conditional opcode is clear.

**`HeadNormRope` should be in §3's class-C example list.** It is the KV cache
writer for every dense family, it is class C in three of the five families'
shipped programs, and §5.2's "RoPE reads `positions[row]` and stays packed" is
true of the ISA's batched mode and false of what the dense emitter currently
emits.

### 5.2 Class A is not one thing

§3 says "A needs nothing. This is most of the FLOPs in every family, which is why
the packing is worth doing at all." The first sentence is false for 25 of the 101
class-A opcodes, and the false part is load-bearing for exactly the family the
plan cares most about: GLM-5.3's decode program is refused **entirely** by
class-A operators (§4.3).

A `single-row` class-A operator processes row 0 and leaves the rest unwritten.
That is the same failure shape as a class-C operator — plausible output, no trap
— reached by a different mechanism, so it deserves the same treatment. The tool
gives it disposition `use-row-form` and refuses.

The fix is per-operator and already largely exists in the ISA (`MoeCombinePf`,
`MoeRouterTopkPf`, `MoeGroupGluPf`/`MoeGroupDownPf`, `MoeGluMxPf`/`MoeDownMxPf`),
so this is scheduling work, not kernel work — but it is work, and §3 currently
says there is none.

### 5.3 §3 and §11 name the wrong opcode for the `s <= q_pos0 + t` derivation

§3's class-C row says "`IndexScore` (`s <= q_pos0 + t`)", and §11 repeats it.
`IndexScore` is op **58**, the decode indexer, and `d_index_score`
(`runtime/amd/op_attention.h:4558`) reads `kv_len[b]` — a per-row array indexed
by the batch row:

```c
for (unsigned b = 0; b < n_batch; b++) {
    const unsigned len = (unsigned)kv_len[b];
```

Op 58 is **class B**. The `s <= q_pos0 + t` derivation is in `IndexScorePf`, op
**117**, its prefill twin (`op_attention.h:5173-5174`):

```c
const unsigned len = (unsigned)as_glob(kv_len)[0];
const unsigned q_pos0 = len - n_tok;
```

`IndexScoreKpool` (134) is likewise per-row (`t6=kv_len(i32[n_batch])`) and class
B. The DSA prefill chain's actual class-C members are **117, 118, 119 and 131**.
§8's Phase-3 DSA item lists `IndexScore`/`DsaPoolExpand`/`IndexSelectPf`; it
should read `IndexScorePf`/`IndexSelectPf`/`IndexUnionPf`/`DsaPoolExpand`, and
`DsaPoolCompress` (130) belongs there too — it reads `pos[0]`.

### 5.4 MLA prefill's hazard is not a scalar, and that changes the fix

§3 and §5.2 describe MLA prefill as having "a per-packet query base". The
mechanism is different, and the difference matters:

```c
const unsigned len = (unsigned)kv_len[b];       // PER-ROW
const unsigned q_pos0 = len - n_tok;            // n_tok is a packet IMMEDIATE
```

(`op_attention.h:2877-2878` and `:3625-3626`.) `kv_len` is already a per-request
array. What is shared is `n_tok`, the chunk length. So MLA prefill *can* carry
several requests today — it just requires every one of them to contribute the
same number of query rows, ending at its own `kv_len[b]`. Under ragged packing it
is silently wrong, so it stays class C; but the conversion is "read the span's
row count", not "materialize a per-row position array", and a uniform-length
multi-span batch is already legal. That is a cheaper Phase-3 item than §5.2
implies, and it is worth measuring before writing the general form.

`FlashMlaMaterializedPrefill` (127) is a stricter case: it carries neither
`kv_len` nor a position operand (`i0=T i1=H i2=H_KV i3=D_QK i4=D_V i5=abi`), so
its causal bound *is* the row index inside the packet. It can only ever describe
one span starting at position 0.

### 5.5 The KDA D-class limit is narrower than §5.4 states

§5.4 says K3's KDA is class D because "`KdaStateStep` is serial-`T`, the chunked
scan is deliberately absent, and the chunk ops are documented 'dense
single-sequence'". True of `KdaStateStep` (102), `KdaConv` (88) and the chunk ops
(121-124).

Not true of the gated and fused variants. `op_kda.h:1041` already carries an
`INDEPENDENT-SEQUENCE PATH` selected by `bstride != 0`, with a per-row `parked`
mask whose comment is a restatement of the plan's own park-mask requirement:

> a row the server has parked -- a slot in the middle of a chunked prefill, or an
> idle slot -- would otherwise have its carried state advanced by a garbage
> token. An append-only KV cache tolerates that; a recurrence does not.

and `op_kda.h:1473` makes the row axis parallel on that path. So `KdaConv3` (111)
is class B when `j0=bstride != 0`, and `KdaConvStateStepG` (120) and
`KdaDecodeFused` (125) are class B unconditionally — their `t7=descriptor`
carries `in.pos` and an optional `parked`. The KDA **decode** side is in the same
position as GDN's: already packed-friendly through a per-row mask that exists.

`KdaStateStepG` (112) is the awkward one, and it is a finding in its own right:
`runtime/amd/interp.hip:2752` passes `bstride=0, parked=nullptr` on the ordinary
arm and sources both from **program metadata** on the `packed_kda` arm. The same
packet is therefore class D on one code object and class B on another, and
nothing in the instruction says which. The tool classifies it D (the strict
reading) and says so. A descriptor-carried class cannot be declared in blob
metadata (§4.2) unless the object variant is part of the declaration.

### 5.6 Small things

- **`AttnRes` (104) is class A.** Its softmax couples the block-residual ring
  rows *of one token*, not tokens, so it packs on `i0=T`. Worth stating because
  "the softmax couples the rows" in its own doc comment reads like a cross-row
  coupling and it is not.
- **`MoeCombinePf`'s `i3=t_row0` and the fp8 GEMMs' `i4=a_row0` are row bases,
  not position bases.** They are safe precisely because §4.1 requires spans to
  stay contiguous in X. If that invariant is ever relaxed for tile fitting, every
  `a_row0` becomes a class-C hazard. §4.1 calls contiguity "for reproducibility
  and simple metadata construction"; it is also load-bearing for correctness.
- **`XArgmaxFin` (28) caps `S` at 128**, as §6.4 says — `PLOW_XAMAX_MAX_BATCH`,
  `op_collective.h:884`. Its live count is `i1`, not the ISA's documented
  operand; see §6.

## 6. Corrections to the ISA's own metadata

These are in `crates/packet` and `runtime/common/dev_isa.h`, which a sibling
owns. Nothing here was edited; the tool classifies from the interpreter where the
spec is wrong and records that it did.

1. **`XArgmaxFin` (28)'s operand spec is wrong.** `dev.rs` and `slots.rs` say
   `i0=n_gpu i2=slot`. `runtime/amd/interp.hip:5421-5423` says, in its own
   comment, `i0=nparts i1=n_batch i2=vocab_l i3=gate i4=val_slot`, and takes
   `n_gpu` from the kernarg (`prog.n_gpu`). This matters directly for the plan:
   §6.4 builds the sharded-greedy tail on this opcode, and an implementer reading
   `slots.rs` would write the TP degree into the partial count.

2. **Three "reserved, body not built" opcodes are dispatched.** `slots.rs`'s
   `RESERVED` list is `Nop`, `FlashMlaPrefill` (51), `AttnSelect` (53),
   `FlashGatherPrefill` (55), with the comment "Keeping these explicit is what
   lets `Provenance::Undocumented` mean 'an oversight' rather than 'probably
   fine'." All three of the non-`Nop` entries have live AMD arms —
   `interp.hip:3745`, `:4021`, `:3748` — and `FlashMlaPrefill` is the operator
   the GLM-5.3 campaign's prefill programs are built from (78 instructions in the
   shipped TP4 blob). `AttnSelect` has a passing hardware test
   (`runtime/tests/attn_select_gfx950_test.c`). The audit currently refuses
   `AttnSelect` because the plan's rule leaves it no choice; giving op 53 an
   operand spec would make it class B and remove a refusal that is only there
   because the metadata is stale.

3. **`Mamba2Scan` (90) has no operand spec anywhere** — no doc comment in
   `packet::dev`, no `packet::slots` entry, `Provenance::Undocumented`. It is the
   only such opcode in the ISA. It is refused, correctly, and `dev.rs`'s own
   commentary already calls its packed-pointer-blob operand form "the cautionary
   tale, not the template".

4. **`KdaStateStepG` (112)'s `i7=parked`** is not read by
   `runtime/amd/interp.hip:2752`'s ordinary arm, which passes `nullptr`. Worth
   reconciling before anything relies on the slot.

## 7. The verdict on runtime fusion v1

"Mixed step v1" is the AMD runtime fusion behind `--fusion` / `PLOW_FUSION`
(default **false**, one consumer: `crates/plowrt/src/exec/amd.rs:11215`). It
synthesizes a packed program at model load by rewriting the blob's ordinary
prefill program (`exec/mixed_program.rs`), binds it to a separately-built code
object (`interp_mixed_gq.elf`), and runs one step that carries the decode batch
and one prefill span together.

Its measurements are in `docs/amd/gemma4-31b-mi300x.md`: **+5.5% throughput and
−35% TTFT at 128 tokens / concurrency 4, −17.6% at 8192/4, and −2.9 to −5.5% at
concurrency 1, where it cannot fire at all.**

### 7.1 Which of v1's assumptions are the class-C hazards the plan names?

Running the classifier over what v1 actually synthesizes from the measured
Gemma-4-31B blob (`opaudit_tests.rs::mixed_step_v1_packs_only_class_a_and_class_b_plus_two_hand_converted_c`):

| opcode | count | class | extent | disposition |
|---|---|---|---|---|
| `RmsNorm` | 121 | A | row-counted | ready |
| `HeadNormRope` | 180 | **C** | row-counted | needs-conversion |
| `Embed` | 1 | A | row-counted | ready |
| `SoftCap` | 1 | A | elementwise | ready |
| `Gemm` | 291 | A | row-counted | ready |
| `FlashPrefill` | 60 | **C** | row-counted | needs-conversion |
| `FlashDecode` | 60 | B | row-counted | descriptor-fills |
| `FlashMerge` | 120 | A | row-counted | ready |
| `NormResidual` | 120 | A | row-counted | ready |
| `Argmax` | 1 | A | row-counted | ready |
| `ArgmaxFin` | 1 | A | row-counted | ready |
| `GemmGlu` | 60 | A | row-counted | ready |

Six synthesized programs, one per prefill bucket, identical opcode sets; the two
smallest-bucket programs carry 60 `FlashMerge` rather than 120 (the split-K merge
is emitted only where `nsplit > 1`).

**Every operator v1 packs is class A or class B, except two — and those two are
exactly the ones the fusion converts by hand.** There is no unconverted C-class
operator and no D-class operator anywhere in a v1 program. The narrow gate makes
every other hazard unreachable, and it is reachable-by-construction rather than
by intent:

- **No `IndexScore`, `IndexScorePf`, `IndexSelectPf`, `IndexUnionPf`,
  `DsaPoolExpand`, `DsaPoolCompress`, `DsaPoolStash`.** These opcodes have no arm
  in `mixed_program.rs`'s match; the catch-all
  (`mixed_program.rs:451`) refuses the whole synthesis, which is caught once at
  `amd.rs:11223` and disarms fusion for the process. Upstream of that, a DSA
  model emits `FlashMlaPrefill`/`FlashMlaDecode`, so `mixed_program.rs:166-176`
  refuses at "no ordinary decode program" before the instruction walk starts.
- **No MoE opcode, no recurrent opcode.** Same catch-all. Additionally the mixed
  object is built without `$AX_MOE`/`$AX_MLA`/`$AX_K3`
  (`scripts/build_gfx942.sh:1045`), so those `case` arms are `#if`'d out — and
  the mixed object is the one place in the tree where `default:` is
  `__builtin_trap()` rather than a silent no-op (`interp.hip:4559-4562`).
- **No `single-row` class-A operator.** The rewrite's whole purpose is to give
  every packed operator a live row count: it normalizes five GEMM tile opcodes to
  `Gemm`, converts the lm_head `Gemv` to a `Gemm` with `i0=dcap`, folds
  `Gemm·Gemm·Glu` into `GemmGlu`, and rewrites `SoftCap`/`Argmax`/`ArgmaxFin`'s
  counts.
- **No `HeadNormRopeFp8`, `FlashDecodeFp8`, `FlashPrefillFp8`.** Refused by name
  (`plow-asset/src/mixed_step.rs:1011-1013`, "dense consumer requires BF16 direct
  KV"), and the object refuses to *compile* with `PLOW_FP8`/`PLOW_FP8_KV`/
  `PLOW_MXFP4` set (`interp.hip:19-21`).

So the answer is the one the task anticipated — the gate makes the hazards
unreachable — with two refinements worth recording.

**One claim in the gate is not enforced.** The gate is often described as "dense
GQA". Nothing in the tree compares `n_head` to `n_kv_head`; an MHA model passes
every check. What is enforced is *dense attention* — the decode program must
contain `FlashDecode` and no `FlashPrefill`, and a prefill program must contain
`FlashPrefill` — plus head dim ∈ {256, 512} (`mixed_program.rs:343`). MLA/MoE/KDA
models are excluded by opcode, not by a GQA test. That is a stronger exclusion,
not a weaker one, but the description should match the code.

**v1 already solved `HeadNormRope`'s class-C problem, and solved it the way the
plan should not.** v1 does not set `i6=n_batch_kv`. It binds a *new operand*
(`inst.t[6] = decode_slot`, `mixed_program.rs:332-334`) that only the mixed
object reads, and the mixed object's `d_head_norm_rope` takes the row's slot and
position from `plow_mixed_row` under `#if PLOW_MIXED_STEP`
(`runtime/amd/op_norm.h:456-499`). The class is promoted by a **compile-time
object variant**, not by anything visible in the packet. That is precisely the
pattern §4.2 replaces with a descriptor pointer in `PlowProgram` — and it is why
a declared-class field in blob metadata has to name the object variant it was
declared against (see §5.5's `KdaStateStepG`).

`FlashPrefill` is not converted at all: v1 keeps `i4=q_pos0` and is correct only
because it admits **exactly one prefill span** per step. Under the Phase-2 route,
`q_pos0` becomes per-span and that restriction goes away.

### 7.2 Is v1's concurrency-1 cost explained by the object swap?

**No. The object-swap mechanism is refuted by the code, and the real mechanism is
in the scheduler.**

The object-swap hypothesis — that arming fusion makes the server run a
differently-built decode object (`GM_BM=64 GM_BN=128`, 4 waves) on every tick —
does not survive reading the exec path:

- `RuntimeConfig::fusion` has exactly **one** consumer in the whole runtime,
  `exec/amd.rs:11215`.
- `GM_BM` / `GM_BN` / `PLOW_WG_WAVES` are `-D` defines of the asset build
  (`scripts/build_gfx942.sh`). No Rust code reads or sets them; the only mentions
  in `crates/` are `kernelcaps`' target descriptions and comments.
- Decode-object selection is `exec/amd.rs:3187`'s
  `object_name(phase, variant, arm, sched)`. Fusion is not one of its inputs.
- The mixed object is a separate HSA module with its own kernel symbol
  (`plow_interp_mixed_{arch}_gq`), launched from exactly one site,
  `amd_mixed_step.rs:433`, inside `AmdEngine::mixed_step` — which requires both a
  non-empty decode set and a non-empty prefill set (`amd_mixed_step.rs:356-357`)
  and therefore cannot run at concurrency 1.
- `interp_mixed_gq.elf` is built unconditionally and ships either way
  (`build_gfx942.sh:1045`); fusion only decides whether it is *loaded*.

What arming fusion does change on every request, at every concurrency, is the
**prefill schedule**. `serve/engine.rs:1318-1320`:

```rust
if self.mixed_step_rows(1, 1).is_some() {
    split_terminal_prefill(self.pf[slot].as_mut().expect("prepared cursor"));
}
```

`split_terminal_prefill` (`engine.rs:651-665`) peels the last row off the final
prefill chunk into its own one-row step. That peeled row is then not run as
prefill at all: `terminal_prefill_ready` becomes true, `finish_prefill_batch`
(`engine.rs:1337`) takes the prompt's last token and feeds it through
`step_batch` — **an ordinary decode-shaped transformer pass**. So arming fusion
adds one full extra model launch per request, on the TTFT critical path, whether
or not the fusion ever fires. At concurrency 1 it never fires and the extra pass
is pure cost.

That is not a new discovery so much as a rediscovery: it is the defect the plan's
§1 already names —

> Include a prompt's final input token in the body normally; sample its hidden row
> in the terminal segment. Do not replay that token through a decode transformer
> pass. This removes the same defect on both serving paths — AMD's
> `split_terminal_prefill`/`finish_prefill_batch` […]

— and it is a mechanism that is *visible in the code*, unlike the object swap.
Two honest caveats: (1) this accounts for one extra full pass per request, which
at 128-token prompts and typical output lengths is order 1%, not obviously all of
2.9-5.5%; (2) the standing costs of arming are real too — a second HSA executable
resident on the agent, and a private device copy of `in.ids`/`in.pos`/`in.kvlen`
plus widened `act.*` scratch and per-program buffers (`amd_mixed_step.rs:150-191`).
The cheap experiment that would settle the remainder is a one-line one: arm
fusion with the `split_terminal_prefill` call disabled and re-run the
concurrency-1 cell. Until someone does, "the object swap" should not be repeated
as the explanation.

### 7.3 What should happen to v1

**Keep it, deprecate it behind the Phase-2 route for (AMD, dense GQA) once that
pair is measured, and remove it only when a second pair has landed on the new
route.** Not removed now, and not kept indefinitely.

Reasons, in order of weight:

1. **v1 is the only working evidence that the packed step is worth building.**
   +5.5% throughput and −35% TTFT at 128/4 is the strongest existing measurement
   in favour of the plan's premise. Removing it before the replacement is
   measured deletes the control.
2. **Its correctness argument is now a proof rather than an assumption.** §7.1 is
   a machine-checked statement that v1 packs no unconverted hazard. That is
   exactly the standing v1 needs to keep shipping opt-in while the replacement is
   built.
3. **Migration cost is small and one-sided.** The Phase-2 route subsumes v1
   operator for operator: `FlashPrefill`'s per-span `q_pos0` generalizes v1's
   one-span restriction, and the descriptor generalizes v1's compile-time
   `PLOW_MIXED_STEP` row resolver. What does *not* carry over is
   `mixed_program.rs`'s 500-line load-time program rewriter and
   `amd_mixed_step.rs`'s ~30 structural checks — the Phase-2 route emits its
   variants from the emitter (plan §8 Phase 2) rather than synthesizing them from
   a blob, so that code is deleted rather than ported. The device side that stays
   is `runtime/common/mixed_step.h`'s row resolver, which §4.3 already plans to
   extend rather than replace.
4. **The plan keeps the new route opt-in per (backend, family) pair until
   measured**, so a period with both present is expected, not a failure. v1
   covers (AMD, dense GQA, single-GPU, BF16) and nothing else; the new route's
   first pair is the same one. Once the new route wins that cell on the same
   workload, v1 should stop being armable for it — leave `--fusion` accepted and
   have it select the new route, rather than keeping two paths a user can choose
   between.

One thing to carry across regardless of what happens to v1: `split_terminal_prefill`
and `finish_prefill_batch` are a fixed per-request tax that the new route must
delete rather than inherit, and §7.2 says the tax is measurable. That deletion is
in the plan (§6.5: "Remove `split_terminal_prefill`/`finish_prefill_batch`
behavior only for the new route") and it should be one of the first things the
Phase-2 measurement checks, because it is the one place where the new route can
beat v1 at concurrency 1 — where v1 can only lose.

## 8. What the audit does not do

- It classifies opcodes, not *programs*. It says an operator can address packed
  rows; it does not say the emitter has wired the descriptor to it.
- It reads AMD's interpreter where the ISA spec is absent or stale. The CPU and
  NVIDIA interpreters have their own arm coverage, and a per-backend disposition
  (does this object have an arm for this opcode?) is a separate axis this tool
  does not carry. It should: the plan's refusal duty is per (backend, family),
  and `crates/kernelcaps` already knows which kernels a target dispatches.
- It has no notion of the descriptor version or the object variant, so it cannot
  yet check a blob's *declared* classes against its actual ones (§4.2). That
  check is the natural next step once the table moves into `packet` and `devgen`
  stamps it.
