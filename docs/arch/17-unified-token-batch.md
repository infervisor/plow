# The unified token batch

One packed activation matrix per step containing all scheduled input tokens — decode and
prefill, from any number of requests — and one compact terminal segment that produces exactly
the next-token distributions the step owes.

This document describes **what has landed** (the shared foundation: contract, planner, ABI,
opcode, CPU tail) and, where it matters, what has deliberately *not*. The design it implements
is `plans/unified-token-batch.md`; that plan is a design, this is a record of the parts that are
now code and tests. Where the two disagree, the disagreement is called out here.

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

* **No descriptor-aware math arms anywhere.** No interpreter reads `PlowProgram::token_batch`
  yet. `FlashPrefill`'s `q_pos0` is still class C and unconverted on every target, which is why
  `Capabilities::converted_c` starts empty and every real program is refused.
* **No serving-path integration.** No `ServeEngine` capability query, no mux arm, no admission
  policy. `TokenBatchStaging` is reachable but nothing calls it in a serve loop.
* **No blob/manifest section.** `ProgramRole` and the capability check exist; a `token_batch`
  asset section binding body and output payloads does not.
* **NVIDIA and CPU objects are not rebuilt** against the new `PlowProgram` here — no `nvcc` on
  this host, and the CPU dev library is rebuilt by cargo on every build so it needs no separate
  step. The gfx942 objects are rebuilt; gfx950 is not (no gfx950 hardware or lease in play, and
  the build script is a separate arch recipe).
