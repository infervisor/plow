# 30 — Speculative decoding: drafters, attachment and exactness

Speculative decoding in plow has three parts:

- a **drafter** proposes `k` tokens per sequence;
- the target **verifies** `k+1` rows per sequence in one decode launch (span mode);
- the runtime **accepts** the longest prefix that plain decoding would have produced, then rolls
  back the rest.

This page covers four things: the drafter contracts, how a drafter attaches to a packet that was
emitted without it, what makes the verify token-exact, and what each model ships. The kernel-level
verify, accept and rollback machinery is in [29-techniques.md](29-techniques.md) §6.

## 1. Drafter kinds

| Kind | Draft KV | Reads from the target | Example |
|---|---|---|---|
| `chain_shared_kv` | none | its KV caches, embedding, final-norm hidden | Gemma-4 assistant MTP |
| `chain_own_kv` | own layers | hidden of the committed rows | EAGLE-class heads |
| `block_own_kv` | own layers | post-layer residuals at tap layers | DSpark |

There are two ways a drafter can ship:

- **Embedded:** the drafter is part of the target packet (`spec.json` and `spec.draft.programs`).
- **External:** the drafter is a directory that attaches to an already-emitted target packet. The
  target is not re-emitted or modified.

Both use the same draft programs, staging and verify path. The runtime attaches only
`chain_shared_kv` drafters externally today. It refuses the own-KV kinds by name until their KV
lifecycle (§5) exists.

## 2. External drafter ABI (`plow.spec.drafter.v1`)

A drafter directory holds four kinds of file:

- `drafter.json`: the manifest (`plow_asset::spec_drafter::Manifest`);
- `drafter.programs`: an aux-program section whose handles index the drafter's own tensor table;
- `drafter.init`: optional constant data;
- the drafter's weight shards, pinned by sha256.

Every tensor in the drafter's table has one source:

| Source | Meaning |
|---|---|
| `import{read}` | A target tensor, bound by name and byte size: KV caches, embedding, tables. Never written. |
| `import{read_write}` | A target activation that every launch rewrites. The draft may use it as stream-ordered scratch. |
| `weight` | Loaded from the drafter's pinned shards, exact length. |
| `init` | A slice of `drafter.init`. |
| `scratch` | Zeroed device memory owned by the drafter. |

The runtime builds a per-drafter pointer table from these sources: imports alias the target's
device memory, and everything else is the drafter's own allocation. The target's tensor table is
not changed.

### Target identity

A drafter states in `requires` what it needs from the target. The runtime checks it against facts
it derives itself from the loaded packet (`TargetView`), never against claims made by the packet:

- **Packet pin.** `target_packet_sha256` is the sha256 of the `model.pkt` bytes the engine actually
  loaded. The file is stamped around the read, and a file replaced since then is refused. A packet
  without `spec.target.json` can only be matched this way, so every packet emitted before this
  contract (all qualified packets) is matched by packet pin.
- **Identity pins.** A drafter without the packet pin must name the model, its 40-hex revision and
  the checkpoint content digest, plus the target's whole geometry key for key. All of these come
  from the target's `spec.target.json` (`plow.spec.target.v1`). Without the packet pin, the geometry
  is the only proof of KV layout and RoPE.
- **Tokenizer.** The two tokenizers must have the same sha256 of the canonical (key-sorted)
  `tokenizer.json["model"]`. Every added token the drafter uses must exist in the target with the
  same id; the target may add more.
- **Object symbols.** These are exports the decode object must carry with a given value, for
  example `plow_row_slot_1 = 1` for span-mode attention.
- **Imports.** Each import must exist in the target with the declared bytes. A writable import must
  be a target activation, and it must be at least its live size after context narrowing.

### Program safety

A drafter's programs are loaded only if every check below passes. All checks run before any
weight is uploaded.

- **Audited opcodes only.** Each instruction uses an opcode from the audited list, with the write
  slots that list records (`vetted_writes`). Forms that address a cache are refused: fused k+v
  HeadNormRope, and any KV row offset or ladder.
- **Allowed destinations.** Every write lands in drafter scratch or a target activation. It never
  lands in a weight, init data, a read-only import, or the inputs the runtime stages from the host
  (position, KV length, slot, hidden row). The draft tokens are outputs.
- **Bounded writes.** The bytes each write covers, derived from the instruction's own row and width
  fields (`write_extent`), must fit the allocated length of the tensor it names.
- **Remapped handles.** Integer-field tensor handles (`integer_tensor_handles`, for example the
  FP8 attention's row-slot map in `fj[2]`) are remapped with the `t[]` handles and validated the
  same way.
- **Hidden snapshot.** The hidden the runtime saves and restores across ticks must be a target
  activation import, not private memory the target never writes.

### Attaching

- `plowrt serve --assets DIR,drafter=PATH` attaches a drafter explicitly. A bundled
  `DIR/drafter/drafter.json` is used when no explicit drafter is given.
- An attachment is fixed for the engine's lifetime. A second, different drafter for the same assets
  is refused.
- An explicit drafter that does not fit fails the model's load, because it is a configuration
  error and must not fall back silently to plain decode. The same applies when spec is off for the
  model, and on a backend that runs no external drafters.
- The planner charges the drafter's own memory as side-car memory, never as slab-reusable memory:
  - its weights, init data and scratch;
  - the pointer table;
  - the program tables;
  - one hidden-snapshot row per slot.

### Building a drafter from an embedded emit

`plowrt drafter-extract --assets SRC --out DIR [--target TGT]` turns a packet's embedded drafter
into an external one:

1. It remaps the handles into a compact table.
2. It classifies each tensor. Only tensors the target binds become imports. A draft weight the
   target never loads is the drafter's own, even when it sits in one of the target's shards.
3. It hard-links the drafter's shards into `DIR` and pins them.
4. It records the tokenizer facts.

With `--target`, the drafter pins the served target packet instead of the source packet. That
packet is the same model emitted without the drafter, and every import is checked there by name
and bytes.

## 3. Target requirements for verify

- **Span-mode verify** needs decode objects built with the row-slot map (`PLOW_EMIT_SPAN_VERIFY`,
  `_rs` twins) and FP8 KV. A target emitted only with span verify keeps every target program
  instruction-identical to its plain emit, apart from the row-slot operands, which are identity on
  plain decode. Verify over BF16 KV is not implemented.
- **Verify width.** It is capped by the widest rung that span launch can run: 16 rows on the
  Gemma-4 H100 packets. Wider rungs are library-routed or sibling-object rungs. A verify of `B`
  sequences at depth `k` therefore needs `B·(k+1) ≤ 16`.
- **Batch shape.** Spec runs only on pure decode ticks.

## 4. Exactness: batch-invariant decode

Greedy spec-on output must be token-identical to spec-off output. A verify row runs at a wider
launch than the plain step it replaces, so every row must produce the same bits at every launch
width. Two things on a stock Gemma-4 packet make the result depend on the width:

| Source | Why it changes with the rung |
|---|---|
| Attention split count | Baked per rung: the 26B sliding layers use 16 splits at rung 1 and 4 at rung 4. A split's KV range is `ceil(span/nsplit)`, so the reduction order changes. |
| MoE expert path | Rows ≥ 4 take the grouped (dedup) FP8 tensor-core arm, rows < 4 the per-slot arm. The expert K split is sized from the row count. The router stages `h2` with a different association at B=1. |

`PLOW_DECODE_BATCH_INVARIANT` is unset by default and then follows `PLOW_SPEC`. When on:

- the narrow interpreter rungs take the narrowest rung's attention split counts, per attention
  site, from the narrowest up; pinning stops at the first library or sibling-object rung, or the
  first whose pinned splits overflow the partial buffers (the 12B FP8 128-row program at rung 1's
  132 splits). Spec verify never launches wider than the pinned rungs;
- every decode object that exports `plow_batch_invariant` uses the row-count-independent MoE arms
  (the grouped expert arm and a K split sized for one row at every width, the B=1 router
  association). A lone request therefore decodes with different (equally valid) MoE rounding
  than with the flag off; the reference for spec-on identity is the flag-on engine.

Enabling it is all or nothing:

- A packet that cannot honour it refuses an explicit request at load. Examples: generated decode
  attention, context-band rungs, or MoE objects that predate the symbol.
- When it is only implied by `PLOW_SPEC`, such a packet serves plain decode instead, and spec
  serving refuses an engine that is not invariant.

Scope:

- Invariance holds across the narrow gemv rungs (≤ 16 rows on the Gemma-4 H100 packets), which are
  the rungs span verify uses. Wider rungs switch to W8A8 GEMM and are never bit-equal to them.
- Objects export the flag only from the build configuration whose arms were audited (row-blocked
  router, f32 router staging, one dense GEMV arm from B=1). Sibling objects built otherwise (the
  wide-GEMV `_gw` object) lack it, and verify never launches on a sibling-object rung.
- Rows that ride a prefill or mixed launch use prefill numerics.
- The reference for the identity gate is the same engine with spec off and the invariant flag on.

Cost on Gemma-4 26B FP8, plain decode at ctx 2048 (step ms, flag off → on):

| B | Off | On | Cost |
|---|---|---|---|
| 1 | 4.594 | 4.870 | +6.0% |
| 2 | 5.270 | 5.668 | +7.6% |
| 4 | 6.288 | 6.511 | +3.5% |
| 8 | 7.604 | 8.042 | +5.8% |
| 16 | 9.426 | 10.712 | +13.6% |

On that packet a rung-diff of every instruction (rungs 1 vs 4, 2 vs 4, 8 vs 16) finds no differing
output byte, and spec-on greedy output equals the flag-on plain output at B=1/2/4/8, k=1/3/7, ctx
2048 and 8191.

Gemma-4 12B FP8 (dense; B=1 digest unchanged): B=1/2/4/8/16 +0 / +2.6 / +6.3 / +13.3 / +23.4%,
from rung 1's 132 attention splits on the wider rungs. The same rung-diff and identity matrix
passes.

This is why invariance is tied to spec, not on by default.

`step_bench --rung-diff A,B` runs one decode step at two rungs from the same state. It prints the
first instructions whose output rows differ, which is how new width-dependent arms are found.

## 5. Own-KV drafters (block and EAGLE-class)

A pluggable own-KV drafter needs three things from the target, and all three are additions to a
re-emitted target:

- **Taps.** The post-layer residuals at the declared layers. They are folded into the layer-ending
  residual op as an optional extra destination, so no instruction or counter gate is added.
- **A declared `spec.target.json`.** It lists the taps, the tap tensors and the geometry.
- **Span verify.**

The drafter brings its own work:

- an **ingest program** per target program, run after each target launch, which writes its KV at
  each row's slot and position;
- its **block or chain draft programs**.

The runtime contract for the drafter's own KV:

- The prefix namespace is seeded with the drafter digest.
- The drafter has its own full-layer high-water mark.
- Each slot carries a "drafter context complete" bit; a slot without it never drafts.
- Version 1 restricts drafter KV to the target's full-layer geometry.

The embedded DSpark path (context tail baked into every target program) is described in
[25-dspark-pending.md](25-dspark-pending.md).

## 6. Status (H100)

| Target | Drafter | State |
|---|---|---|
| Gemma-4 26B-A4B FP8 | MTP assistant, embedded or external | Re-emitted with invariant objects (programs identical to the qualified packet); greedy identity passes. Production mix (H100 screen): TPOT p50 −41% at 0.5 sessions/s and −43% at 1.8. |
| Gemma-4 12B FP8 | MTP assistant, external | Target re-emitted with span verify and invariant objects: no-drafter digests and step cost equal to the qualified packet. Drafter extracted against it; greedy identity passes. Serving screen: c1 TPOT 7.41 → 2.35 ms, production mix at 1.0 sessions/s −41% p50; c32 +22% (invariant splits on the 32-row rung). |
| Gemma-4 26B-A4B BF16 | MTP assistant | Blocked: span verify needs FP8 KV. |
| Gemma-4 12B FP8 | DSpark (official) | Embedded emit only. Pluggable form per §5 not built. |
| Qwen3.8-27B | MTP (1 layer) | Needs GDN recurrent-state rollback. |
