# Batched GLM-5.2 decode: where the blocker actually is

> **Scope:** no GPU -- source reading with file:line · **PLOW-ARCHITECTURAL** — emitter and slot-contract structure. Arch-independent: the missing token dimension on the MoE decode ops is the same on gfx950.

2026-08-09. Scoping for the throughput item. Everything below is read out of the tree with
file:line, not remembered — three of the five things this campaign "knew" about batched
decode turned out to be wrong, and all three were wrong in the direction of making the job
look bigger than it is.

## The measured symptom

Served GLM-5.2 throughput is **flat at 22.8–27.0 tok/s from concurrency 1 to 32**, with TTFT
rising linearly to 148.2 s. That is pure queueing: the engine is serving one sequence and
everyone else is waiting.

## Correction 1 — the muxer is NOT advancing slots one at a time

The task that owns this work has carried the title *"the muxer admits N slots then advances
them ONE AT A TIME"* since it was filed. That is false:

| | |
|---|---|
| `serve/mux.rs:1682` | `e.step_batch(&feeds)` — `feeds` is every live slot |
| `serve/engine.rs:586` | `step_batch` → `dispatch_all(&advance)` |
| `exec/amd.rs:5748` | `decode_step_batched` → **one** `self.run(dp, self.k_decode)` |

One tick is one dispatch that advances **all** rows. The batched path is built, documented,
and has a ragged `pos`/`kvlen` contract (`i[6] = n_batch_kv` on the decode `HeadNormRope`;
the kernel takes both the KV write row and the RoPE angle from `pos[t]`).

What actually caps concurrency at 1 is `engine.rs` `check_slot`, which rejects
`slot >= self.batch` — and `self.batch` is the **blob's compiled `PLOW_DECODE_BATCH`**.
GLM's blob is emitted at 1 (`devgen/src/mla.rs:5456`, `let n_batch = 1u32`), and
`PLOW_DECODE_BATCH` is silently dropped on the way (fixed to warn in `d6af634`).

So this is an **emitter** problem end to end. No runtime work is required.

## Correction 2 — the MLA attention chain is already batch-aware

From `crates/packet/src/slots.rs`, i.e. the slot contract itself rather than a kernel comment:

| op | batch field |
|---|---|
| `FlashMlaDecode` (:174) | `i[0] = n_batch` |
| `MlaMergeFold` (:178) | `i[0] = n_batch` |
| `OUvFold` (:175) | `i[0] = n_batch` |
| `GemvQkv` (:142) | `i[0] = M` |
| `HeadNormRope` (:123) | `i[6] = n_batch_kv` |

`SAMPLE_BATCH` (0x0704) is wired with a cap of 16, and activation sizing is already free:
`mla.rs:5192` takes `max_rows` over the prefill buckets because *"activations are sized for
the WIDEST bucket, so one tensor table serves every program"* — a B ≤ 16 decode fits inside
what the 8192-row prefill bucket already reserves.

## Correction 3 — and here is the blocker nobody had named

**The MoE DECODE op family carries no token dimension at all.**

| | ops | `i[]` |
|---|---|---|
| decode | `MoeRouterTopk` (:177), `MoeGroupGluFp8Blk` (:172), `MoeGroupDownFp8Blk` (:173), `MoeCombine` (:167) | `k, I_moe, H, n_exp` — **no T** |
| prefill | `MoeRouterTopkPf` (:206), `MoeAlignPf` (:207), `MoeGroupGluPf` (:208), `MoeGroupDownPf` (:209), `MoeCombinePf` (:210) | **all carry `T`** |

This is the reason the "2–4 days" estimate should not have been trusted, and it is also the
reason the Gemma precedent does not transfer: the Gemma ladder that proves batched decode at
1.84× does not exercise a block-fp8 grouped MoE. GLM's MoE seam at B>1 is genuinely
unproven, not merely unwritten.

## The route, and why it is cheaper than it looks

Do **not** extend four MoE decode kernels with a token axis. At `rows > 1`, emit the MoE
seam with the **prefill** op family at `T = rows`. Those ops exist, are AMD-dispatched, and
already handle small `T`.

There is direct precedent in the same file: `emit_glm_dense_block_prefill`
(`mla.rs:4179-4183`) already reuses `MoeGroupGluPf` / `MoeGroupDownPf` for the **dense** FFN
by emitting `n_exp = 1, top_k = 1` and letting `MoeAlignPf` do the bookkeeping — its own
comment opens *"The kernel that was 'missing' already exists."* Reusing the grouped prefill
ops for a shape they were not written for is a move this emitter has made before and gated.

It is also the right shape on the merits: at B=16 with top-8 of 256 experts, the grouped
form reads each touched expert's weights **once** instead of once per token that routes to
it, which is exactly the amortisation batching is for and exactly what the decode ops
cannot express.

## Plan

1. Thread a `rows` parameter through `emit_glm_mla` (:2213), `emit_glm_block` (:4584) and
   `emit_glm_dense_block` (:4990). None takes one today, and their shared GEMV helper
   hardcodes `d.i[0] = 1` (:2255) — its own comment says *"GEMV helper (M=1 decode, no norm
   fold)"*. This is the bulk of the work and it is mechanical.
2. At `rows > 1`, swap the MoE seam to the `*Pf` family at `T = rows`.
3. Tail to `SAMPLE_BATCH`.
4. Respect the `GEMV_MAXM` cap the loader already enforces (`exec/amd.rs:4268` refuses a blob
   whose `PLOW_DECODE_BATCH` exceeds the object's compiled `MM`).

## The gate, which is the part not to skip

**`rows = 1` must re-emit BYTE-IDENTICAL to today's shipped blob**, the way
`declare_glm_rows(rows = 1)` already does for prefill. Without that anchor there is no way
to tell a batching bug from a numerics bug, and this directory's history says that
distinction is where the weeks go. Then: coherence gate at B=1, then a concurrency ladder
against the flat 22.8–27.0 tok/s baseline.

Wants its own branch. Not started here — a half-threaded emitter would be worse than this
plan.

## The ladder is BUILT, end to end, and GLM is the only thing not calling it

Re-checked 2026-08-09 after the suggestion "support decode batch 1..16, or use the prefill
packets". Both halves are already true in the tree.

**The rung ladder needs no new machinery.** `PLOW_DECODE_BATCH_LADDER=1,2,4,8,16` is an emit
knob (`emit_config.rs:112`) whose accessor parses, clamps to `DECODE_RUNG_MAX`, sorts and dedupes
(`decode_rungs()`, :736). Unset it returns `[decode_batch]`, which is what makes an unset ladder
**byte-identical** — the emitter runs its one-program loop once at the same B. The runtime
consumes it fully: one program per rung (`engine.rs:107`), `decode_rungs()` reported at load
(:252), rung selection per tick (:702), and `decode_prog_for` / `decode_step_batched_at`
(`amd.rs:2867`) to fire a NAMED rung. Our serve banner prints `decode_rungs=[1]`.

**It has exactly one consumer, and it is not GLM.** `lib.rs:6566` is the Gemma-4 / 26B-A4B path,
whose own comment says: *"26B-A4B MoE decode is BATCHED (B in 1..=32): the router family, the
flat expert GLU/down and the combine all carry a batch row count and index [B][k] routing
slots."* That is the same fact from the other side: **Gemma's MoE decode ops carry a batch
dimension and GLM's do not.** GLM emits `MoeRouterTopk` / `MoeGroupGluFp8Blk` /
`MoeGroupDownFp8Blk` / `MoeCombine`, none of which has `T`.

**So "use the prefill packets" is the right move and the minimal one.** Per-slot resources are
already sized at the widest rung, and the design note at `lib.rs:6570` says why that works: slot
`s`'s KV offset is `s * (kv_head*ring*hd)`, INVARIANT in B, so *a sequence keeps its slot while
the program under it changes rung to rung*. Nothing about the KV layout has to move.

### What each op needs at rung B

| stage | at B > 1 | status |
|---|---|---|
| norms | `HeadNormRope i[6] = n_batch_kv` | exists |
| QKV / o_proj / router / shared | GEMV rungs at `M = B`; `gv_mm_max = next_pow2(B)` and the `GV_MM_MAX=16` arm | exists; MM instantiated at {1,2,4,8}, blocks of 8 above |
| attention | `FlashMlaDecode i[0] = n_batch`, `MlaMergeFold`, `OUvFold` all take `n_batch` | exists |
| **MoE seam** | **swap to the PREFILL family at `T = B`** — `MoeRouterTopkPf`, `MoeAlignPf`, `MoeGroupGluPf`, `MoeGroupDownPf`, `MoeCombinePf` | the only new emit work |
| tail | `SAMPLE_BATCH` (0x0704), cap 16 | exists |

The MoE swap has direct precedent in the same file: `emit_glm_dense_block_prefill`
(`mla.rs:4179`) already reuses `MoeGroupGluPf`/`DownPf` for the **dense** FFN by emitting
`n_exp = 1, top_k = 1` and letting `MoeAlignPf` do the bookkeeping — *"The kernel that was
'missing' already exists."*

Note B ≤ 16 is the natural cap for the seam, not an arbitrary one: `SAMPLE_BATCH` asserts 16,
and the GEMV `GV_MM_MAX` arm is the 16-row one. 1..16 is exactly the requested range.

### The order to build it

1. Thread `rows` through `emit_glm_mla` / `emit_glm_block` / `emit_glm_dense_block` (none takes
   one; the shared GEMV helper hardcodes `d.i[0] = 1`).
2. At `rows > 1` emit the MoE seam with the `*Pf` family at `T = rows`.
3. Call `ecfg.decode_rungs()` and emit one decode program per rung, as `lib.rs:6566` does.
4. **Gate: the B=1 rung must re-emit BYTE-IDENTICAL to today's blob.** The ladder accessor is
   built to make that true (`[decode_batch]` when unset), so this is a check, not a hope.

Expected payoff is the whole throughput gap: aggregate output is flat at 24.3-26.8 tok/s from
concurrency 1 to 32 while TTFT p50 grows 1.06 s -> 153.8 s, which is a single admitted slot and
nothing else.
