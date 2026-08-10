# The decode batch-size program ladder on gfx942

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4), Gemma-4-12B fp8 · **PLOW-ARCHITECTURAL** — a program-ladder design over compile-fixed `PLOW_DECODE_BATCH`. Arch-independent; the rung costs are gfx942.

`PLOW_DECODE_BATCH` is fixed at emit. A blob compiled `B=16` advances sixteen rows on
every dispatch whether or not they hold a request, so it is slow at concurrency 1; a blob
compiled `B=1` cannot batch at all. There was no runtime selection between the two.

This builds one: `PLOW_DECODE_BATCH_LADDER=1,2,4,8,16` emits **one decode program per
rung into a single blob**, and `AmdServe::dispatch_all` picks the narrowest rung that
covers the occupied slots, per step.

Branch `decode-ladder`, off `worktree-glm52-bringup` @968fc3a.

---

## 0. THE TARGET MODEL IS NOT GLM-5.2, AND THAT IS A FINDING

The commission named the GLM-5.2 TP8 asset. **GLM-5.2 has no batched decode to build a
ladder on.** `glm_emit_full` (crates/devgen/src/mla.rs:4964) emits its decode program at
one row, unconditionally and structurally:

* the decode `Embed` carries `d.i[0] = 1`;
* `emit_glm_tail(&mut b, &c, &tn, cur, &[dep], 1, &mut xgate)` — the literal `1` is the row
  count;
* `emit_glm_block` / `emit_glm_dense_block` (the decode layer emitters) take **no row
  parameter at all**, unlike their `_prefill` twins which take `t`;
* `prog_t.push(1)`.

`PLOW_DECODE_BATCH` never reaches that path. It is read in exactly two places
(`crates/devgen/src/lib.rs:6384`, the dense-GQA family, and `crates/devgen/src/mla.rs:7813`,
`k3_build_model` for Kimi-K3), and GLM is neither. Giving GLM a ladder means first giving
GLM batched decode — sequence-row carriers through MLA decode, the block-fp8 MoE decode
chain, every GEMV and the per-slot KV write — which is a build in its own right, not a
rung count.

So the ladder is built on the **dense-GQA family** and measured on **Gemma-4-12B fp8**,
which is the model on this box that already has working batched decode (the
`g12b-fp8-b4`/`-b16` assets in `/workspace/assets/gfx942` are its history). Everything
below — the emit path, the blob-format rule, the runtime selection — is family-agnostic
and lands on Kimi-K3 the moment `k3_build_model` is given the same loop; only the
measurements are Gemma's.

---

## 1. The per-slot B-invariance audit

The design rests on one property: **a sequence must be able to keep its slot while the
program under it changes rung**. That is true iff every per-slot resource's stride is
independent of `B`, so that allocating at `B_max` and running a narrower rung addresses
exactly the same bytes.

Enumerated from `declare()` (crates/devgen/src/lib.rs:1213-1620) — every tensor whose
SIZE is a function of `dbatch`:

| tensor | declared size | per-slot stride | stride depends on B? |
|---|---|---|:--:|
| `kv.{l}.k`, `kv.{l}.v` | `B·(ring·kvh_local·hd)·elt` | `ring·kvh_local·hd·elt` | **no** |
| `kv.{l}.k_scale`, `.v_scale` (fp8-KV) | `B·(ring·kvh_local)·4` | `ring·kvh_local·4` | **no** |
| `in.kvlen` | `B·4` | `4` | **no** |
| `logits` | `B·vocab_sh·2` | `vocab_sh·2` | **no** |
| `amax.part` | `B·parts·8` | `parts·8` | **no** |
| `mrgc` (merge-fold) | `B·heads·4` | `heads·4` | **no** |
| `moe.router_score` | `B·n_exp·4` | `n_exp·4` | **no** |
| `moe.mfu` | `B·top_k·I_moe·2` | `top_k·I_moe·2` | **no** |
| `moe.table`, `moe.xn2`, `moe.part` | `max(rows,B)·…` | `…` | **no** (`max` grows the size, not the stride) |
| `hn`, `qg`, `kg`, `dg`, `og_tp` | `rows·…`, `rows = ctx.min(max_chunk)` | `hidden` etc. | **no** — not a function of B at all |
| `in.ids`, `in.pos` | `ctx·4`, indexed `[slot]` | `4` | **no** |

Every one is `B ×` a quantity fixed by the model geometry, indexed `[slot]` with slot
outermost. The kernels agree: `d_headnorm_rope` computes
`obase = ((t·nhead + hh)·out_stride + (pos[t] & kv_mask))·hd` and `d_flash_decode` bases
K/V at `b·n_kv_head` and reads `kv_len[b]` — `B` appears as a *bound* on the work-item
range, never inside an address.

**Two resources are NOT `[B]`-shaped, and both are already handled:**

* **`kv.blkres`** (Kimi-K3's snapshot ring) matches `kv.` but is `[t][nb_cap][hidden]`
  sized at the widest PREFILL bucket. Dividing its bytes by `batch` invents a stride and
  walks off the end — a measured `Memory access fault by GPU node-7` at B=16. It is
  excluded from `kv_slot_stride` and from `carried_slot`, and it needs no rebasing at all
  (layer 0 resets it every pass). **Invariant in B for the ladder's purposes: it is not a
  per-slot resource.**
* **The KDA recurrent state** (Kimi-K3, ~0.44 GiB/slot) is `[heads, hd, hd]` f32 with *no*
  batch axis unless the emitter was told to give it one: `declare_kda_state(.., slots)`
  under `RowKind::Sequences`, with `PLOW_KDA_F_SEQ_ROWS` in the state step's flags as the
  carrier. `AmdEngine::load` refuses `batch > 1` against a state tensor whose decode
  program lacks the flag. Under a ladder the state is declared at `B_max` and the stride is
  `bytes / B_max` — **invariant**, provided every rung is emitted with the sequence-row
  carrier. That is a live constraint on extending the ladder to K3 and it is recorded as
  such: `k3_build_model` arms `RowKind::Sequences` on `dbatch > 1`, so a K3 ladder must
  arm it on the one-row rung too (`PLOW_K3_SEQ_ROWS` is exactly that switch, and it exists
  because at one row every carrier is a no-op by construction).

**Verdict: the design holds.** No per-slot stride moves with `B` on the dense-GQA path.

### The one place where B *did* leak into an address, and the fix

The one-row rung is not simply "the B=1 program". `d_headnorm_rope`'s legacy arm takes its
KV write row from `out_row0`, a **host-patched immediate** (`patch_kvrow`, `i[3]`) that
`decode_step_batched` deliberately does not write — above `batch == 1` the row comes from
`pos[]` instead. A laddered blob always has `batch = B_max > 1`, so a naively-emitted
one-row rung would rewrite KV row 0 on every step: fluent, wrong, no fault.

So under a ladder `i[6] = n_batch_kv` is armed at `t == 1` as well. At one row the armed
formula and the legacy formula compute the same address, which is what makes the arming
safe rather than merely necessary. Measured cost: **96 bytes** of blob difference between
`PLOW_DECODE_BATCH_LADDER=1` and `PLOW_DECODE_BATCH=1` — 48 layers × 2 head-norms × one
immediate, and nothing else.

## 2. What reads `decode_batch`, at emit and at serve

**Emit.** `EmitConfig::decode_batch` → `crates/devgen/src/lib.rs:6384` (dense GQA, clamped
1..32) and `crates/devgen/src/mla.rs:7813` (Kimi-K3, clamped 1..16). It sizes `declare()`,
becomes the decode program's `t`, and lands in `prog_t.last()`. Inside `emit_phase` it
reaches: the GEMV `M`, `FlashDecode.i[0] = n_batch` (+ `j[0]`, the KV row-capacity trap),
`HeadNormRope.i[6] = n_batch_kv`, the MoE decode row count `nb`, the lm_head's
`(M, a_row0) = (t, 0)`, and `Argmax.i[1] = nb_argmax`. `manifest.rs` re-derives it from
`FlashDecode.i[0]` in the last program and publishes `gv_mm_max = next_pow2(B)`.

**Serve.** `AmdEngine::load` derives `batch` from `in.kvlen`'s byte count and cross-checks
it against `progs[decode].t`; it caps at `PLOW_GEMV_MAXM = 16` and drives `kv_slot_stride`,
`carried_slot`, `read_sampled_batched`, `upload_parked`, and `check_slot`. `AmdServe`
sizes `pos`/`live`/`next_id`/`pf` to it and `mux.rs` sizes its slot table to `e.batch()`
(so mux slot *i* IS engine slot *i*), rejecting anything past it. `/v1/models` publishes it
so `bench_speed.sh` can say whether concurrency batched or queued.

Under a ladder every one of these keeps meaning `B_max` — the slot capacity — and only the
*program* moves. That is deliberate: `batch` is what the KV cache holds.

## 3. Blob-size cost of N decode programs — measured

Gemma-4-12B fp8, `--seq 128,512,1024 --max-ctx 4096`, gfx942, L2-placed:

| emit | decode programs | decode packets each | decode wg-packets | blob bytes | Δ vs B=1 |
|---|--:|--:|--:|--:|--:|
| `PLOW_DECODE_BATCH=1` | 1 | 527 | 74,452 | 32,808,100 | — |
| `PLOW_DECODE_BATCH=2` | 1 | 527 | 75,575 | 32,862,004 | +53,904 |
| `PLOW_DECODE_BATCH=4` | 1 | 527 | 77,821 | 32,969,812 | +161,712 |
| `PLOW_DECODE_BATCH=8` | 1 | 527 | 82,105 | 33,175,444 | +367,344 |
| `PLOW_DECODE_BATCH=16` | 1 | 527 | 90,577 | 33,582,100 | +774,000 |
| **`LADDER=1,2,4,8,16`** | **5** | 527 each | 400,530 total | **48,633,076** | **+15,824,976** |

Two things fall out of this table.

* **The instruction count does not move with B — 527 packets at every rung.** B changes
  work-item counts and immediates, not the op graph. The blob grows only in the
  *workgroup-packet stream*, at a flat **48 bytes per workgroup-packet**
  (774,000 / 16,125 = 48.0, and the ladder's +15.82 MB / +329,674 wg-packets = 48.0).
* **`set_tensor_dedup` does its job.** Five decode programs, one tensor table: the ladder's
  KV cache is **11.00 GiB — identical to the fixed B=16 blob**, not 5× anything. Weights
  12.0 GiB and activations 0.37 GiB are likewise unchanged. The entire cost of four extra
  rungs is 15.8 MB of packet stream against a 23 GiB resident footprint: **0.07%**.

Byte-identity of the unset path, verified rather than asserted:

    PLOW_DECODE_BATCH=1   vs pre-change emitter   IDENTICAL
    PLOW_DECODE_BATCH=16  vs pre-change emitter   IDENTICAL
    PLOW_DECODE_BATCH_LADDER=16 vs PLOW_DECODE_BATCH=16   IDENTICAL
    PLOW_DECODE_BATCH_LADDER=1  vs PLOW_DECODE_BATCH=1    96 bytes (the i[6] arming, §1)

## 4. Design

### How the runtime tells a rung from a prefill bucket

No new blob field. `packet::devbuild::decode_rung_lo(prog_t)` takes the maximal *trailing*
run of programs whose `t` is at most `DECODE_RUNG_MAX = 32` and strictly ascending. The two
ladders cannot collide by construction — a decode rung is a sequence count (≤32, the MoE
router's `inv[]` scratch bound) and a prefill bucket is a chunk width (128 and up) — and the
emitter carries an assert that refuses to write a blob where they would. On a blob with one
decode program the function returns `prog_t.len() - 1`, which is the `progs.len() - 1` every
caller used before, so nothing else had to change.

`AmdEngine` keeps both ends: `dec_lo` (narrowest rung) and `decode` (widest, still
`n_prog - 1`). `plan_for`/`chunk_steps` search `0..dec_lo` for prefill buckets;
`decode_prog()` still answers the widest rung, which is the only safe answer for a caller
that does not know the ladder exists (`amd-bench`, the TP gate audit, `patch_kvrow`).
`has_prefill()` replaces `n_programs() == 1` — a five-rung decode-only blob is five
programs and still decode-only.

### The selection rule: highest occupied SLOT, not live count

    rows = 1 + max{ s : live[s] or mid-prefill[s] }        rung = min{ w in ladder : w >= rows }

Choosing on the live *count* would be wrong. A rung of width `w` dispatches rows `[0, w)`
only; a sequence parked in slot 5 is simply not advanced by a width-4 rung, while the host
advances its `pos` regardless — so its next step would attend over a KV row nobody wrote.
Slots are handed out lowest-first (`mux.rs`: `slots.iter().position(|s| s.is_none())`),
which is what makes the count and the high-water mark agree in the common case.

A slot mid-chunked-prefill counts as occupied: its rows are real KV, and `dispatch_all`
already points it at its own frontier rather than row 0.

**No compaction on release.** Freeing slot 0 while slot 5 is live leaves the rung where it
is. Moving a sequence down a slot means copying its whole KV block — GiB — which costs
orders of magnitude more than the rows it would save. The ladder narrows again as the high
slots drain, and since admission reuses the lowest free slot, the high-water mark tracks the
live count closely under normal churn.

### Per-B specialisation: what is available, and what is NOT

The commission's premise is that a compiler-driven stack can make each rung a *different
program*, not the same program padded — which is what vLLM's captured CUDA graphs do. Two
levels have to be distinguished, because only one of them is free.

**PROGRAM level — specialised, automatically, at every rung.** Each rung is emitted by a
separate `Builder` at its own `t`, so it gets its own work-item decomposition (`FlashDecode`
runs `t·heads·ns` items, not `B_max·heads·ns`), its own CU slices and dependency sets from
`flash_mla_cus`/`blocked_gemv_cus`/`elem_cus`, its own L2 placement solved for its own
packet counts, its own MoE routing width, and its own fusion decisions where they are
`t`-gated (`PLOW_FUSE_ARGMAX`'s `GemvArgmax` epilogue is `t == 1` only, so the one-row rung
takes it and the wide rungs do not — inside one blob). That is the differentiator, and it
costs the 15.8 MB above.

**OBJECT level — NOT specialised, and this is the ladder's real tax.** The code object is
chosen per *phase*, not per program: `object_name(phase, variant, prefill_arm, sched)` in
`crates/plowrt/src/exec/amd.rs:436` composes a stem from `Phase::Decode` and knows nothing
about which decode program is running. So all five rungs run on ONE decode object, compiled
at ONE `PLOW_GEMV_MM`, which must cover the widest rung — `check_gemv_capacity` refuses a
blob whose batch exceeds the object's compiled bucket, and it is right to.

That is safe (`d_gemv_t<MM>` takes a runtime `M` and predicates on `m < M`, so one
instantiation serves every `M <= MM`) but it is not free, because `MM` sets register and
LDS pressure for the whole megakernel. Measured on this box, `scripts/build_gfx942.sh`'s own
cliff table, `interp_decode` / `interp_decode_fp8` rows (VGPR, AGPR, LDS, spill):

| objects | `PLOW_GEMV_MM` | VGPR | LDS | spill |
|---|--:|--:|--:|--:|
| `PLOW_OCC4=1` (shipped B=1 profile) | 1 | 108 | 30,768 | 0 |
| `PLOW_OCC4=1 PLOW_DECODE_BATCH=2` | 2 | 108 | 30,768 | 0 |
| `PLOW_OCC4=1 PLOW_DECODE_BATCH=4` | 4 | 108 | 30,768 | 0 |
| `PLOW_OCC4=1 PLOW_DECODE_BATCH=8` | 8 | 108 | 30,768 | 0 |
| `PLOW_DECODE_BATCH=16`, no occ4 | 16 | 256 | 64,560 | 26 |

The occupancy-4 decode profile — 104-VGPR cap, 30,736 B LDS, one workgroup at 4 waves/SIMD —
**survives `PLOW_GEMV_MM` up to 8 with zero spill**, which was not previously recorded (the
campaign note says only "occ4 is batch-1-only"). Without it the decode object sits at 256
VGPR, 64,560 B LDS and 26 spilled registers. So the ladder's `B_max` is not a free parameter:
it picks the object profile that *every* rung, including the one-row rung, then runs on.

### The rungs really are different programs — from the blob, not from intent

`plowrt disasm` on the laddered blob:

    programs  [128, 512, 1024, 1, 2, 4, 8, 16]

and per rung (layer 0 shown; the pattern repeats for all 48):

| rung `t` | `FlashDecode` | KV head-norm | `Argmax` | distinct workgroup widths in the program |
|--:|---|---|---|---|
| 1  | `n_batch=1  nsplit=16` | `ntok=1  n_batch_kv=1`  | `n_batch=0` | 1, 2, 16, 18, 64, 76, 152, 286, 304 |
| 2  | `n_batch=2  nsplit=16` | `ntok=2  n_batch_kv=2`  | `n_batch=2` | 1, 2, 4, 18, 32, 64, 76, 128, 152, 286, 304 |
| 4  | `n_batch=4  nsplit=16` | `ntok=4  n_batch_kv=4`  | `n_batch=4` | 1, 4, 8, 18, 64, 76, 152, 256, 286, 304 |
| 8  | `n_batch=8  nsplit=16` | `ntok=8  n_batch_kv=8`  | `n_batch=8` | 1, 8, 16, 18, 64, 76, 128, 152, 286, 304 |
| 16 | `n_batch=16 nsplit=16` | `ntok=16 n_batch_kv=16` | `n_batch=16` | 1, 16, 18, 32, 64, 76, 152, 256, 286, 304 |

Three things to read off it. The **head-norm dispatch width scales with the rung**
(2 → 32 workgroups) while the **GEMV width does not** (152 at every rung) — the emitter is
placing the row-parallel work per rung and leaving the weight-stream work alone, which is
the right answer and nobody had to write it down. The **argmax takes a genuinely different
code path** at the one-row rung (`n_batch=0`, the single-sequence fold) than at the wide
ones — two different programs inside one blob. And `n_batch_kv=1` on the one-row rung is the
§1 fix, visible: 96 instructions, 48 layers × k-norm and v-norm.

This is the part vLLM's captured-graph equivalent cannot do. A CUDA graph captured at
size 16 and replayed with 1 live sequence runs the size-16 kernel launches; here the size-1
rung is a size-1 *program*, with its own placement and its own epilogue.

## 5. How to reproduce

Objects (outside `nix`, hipcc needs the system glibc). `PLOW_DECODE_BATCH` here only sizes
`PLOW_GEMV_MM = next_pow2(B)` capped at 16 — it is a CEILING on the object, and the blob's
own rungs are what the ladder is made of:

    env PATH=/usr/bin:/bin:/opt/rocm-7.2.4/bin ROCM_PATH=/opt/rocm-7.2.4 HIP_PATH=/opt/rocm-7.2.4 \
        PLOW_HIPCC=/opt/rocm-7.2.4/bin/hipcc PLOW_OCC4=1 PLOW_L2HIER=1 PLOW_DECODE_BATCH=16 \
        bash scripts/build_gfx942.sh <objdir>

Blob:

    PLOW_FP8=1 PLOW_W8A8=1 PLOW_DECODE_BATCH_LADDER=1,2,4,8,16 \
      ./target/release/plowc --emit devblob --hf-dir /workspace/models/gemma-4-12B-it \
        --gpu MI300X --arch gfx942 --seq 128,512,1024 --batch 1 --max-ctx 4096 --out <assetdir>

`plowc` prints the ladder as it emits, and `plowrt disasm <blob>` prints
`programs [128, 512, 1024, 1, 2, 4, 8, 16]`. The server logs the rungs at load
(`decode_rungs=[1, 2, 4, 8, 16]`) and logs a line every time the rung CHANGES:

    INFO decode ladder rung rung=4 occupied=3

Serve and bench exactly as any other asset — the ladder needs no runtime flag.

Non-power-of-two rungs work and cost the same shape (`PLOW_DECODE_BATCH_LADDER=1,3,6,12,16`
emits `[…, 1, 3, 6, 12, 16]`), so the granularity question below is a measurement, not a
capability question.

`cargo build --release -p plowrt` **needs `--features hsa`** or serve silently runs the CPU
reference and decodes garbage.


---

## 6. THE MEASUREMENT WAS BLOCKED BY TWO PRE-EXISTING BUGS

Neither is a ladder bug. §6b is FIXED here (the arena constant was simply wrong). §6a is
**diagnosed and defaulted OFF on gfx942**, which stops the wrong output — the fused arm
itself is still broken on that part and fixing it is the top follow-up.

Neither is a ladder bug. Both were found because a decode ladder is the first thing on this
box that serve-gates a BATCHED gfx942 blob with a real prefill, and both would have silently
poisoned anyone else's batched work.

### 6a. `PLOW_FUSE_QUANT` (default ON) makes every Gemma-4 gfx942 blob emit WRONG output

A blob emitted from `worktree-glm52-bringup` @968fc3a with the documented recipe
(`PLOW_FP8=1 PLOW_W8A8=1`, gfx942, `--seq 128,512,1024`) answers "capital of France" with

    ',1___....1.111111111111'

on objects built from the SAME tree. `PLOW_FUSE_QUANT=0` on the identical command answers
**'Paris'**. Reproduced clean, with the shipped asset as a bracketing control in the same
session and the same server binary:

| blob | objects | answer |
|---|---|---|
| `/workspace/assets/gfx942/g12b-fp8` (shipped, 2026-08-04) | shipped `hsaco-occ4full` | **Paris** |
| shipped blob | objects built here from branch HEAD | **Paris** |
| branch-HEAD emit, default flags | objects built here | `,1___....1.111111111111` |
| branch-HEAD emit, **`PLOW_FUSE_QUANT=0`** | objects built here | **Paris** |
| branch-HEAD emit, `PLOW_L2_PLACE=0` | " | garbage (not the cause) |
| branch-HEAD emit, `--max-ctx 16384` | " | garbage (not the cause) |
| branch-HEAD emit, `PLOW_NO_FUSE_NRN=1` | " | garbage, byte-for-byte the SAME garbage |

The decode programs of the working and broken blobs are IDENTICAL instruction for
instruction (`plowrt disasm` diff: 622 vs 622, the only difference `nsplit` 38 vs 16 from the
different `--max-ctx`). The difference is in PREFILL: the branch fuses the activation quant
into the norm (`RmsNorm ... xq<-act.xqh ascale<-act.ash`, 862 insts) where the shipped blob
emits separate `QuantFp8` packets (1006 insts). A prefill that writes a wrong KV cache gives
exactly this signature — a fluent, confident, wrong stream.

**Why the campaign's Gemma cross-gate did not catch it:** the gate re-uses the STORED
`g12b-fp8` asset and rebuilds only the objects. That asset predates the fused quant, so the
gate has been re-certifying a blob that does not contain the regression. Any Gemma number
measured on a freshly-emitted branch blob since the fused-quant merge is measuring wrong
math.

**Blast radius, stated precisely.** `qnorm_fuse` is local to `emit_phase`, and `emit_phase`
is reached by the **dense-GQA family only** — `mla.rs` and `k3.rs` contain zero calls to it —
and it additionally requires `w8a8`, which those families never set (GLM's fp8 is
block-scaled, via `glm_linear_fp8`). So **GLM-5.2 is NOT exposed and no GLM number in this
campaign is affected**. The exposure is Gemma-4 / Llama / Qwen at `PLOW_W8A8` on AMD, on
FRESHLY EMITTED blobs only; any stored asset predating the fold is fine, which is exactly why
the standing gate did not see it.

**What is landed here.** The fold is now default-OFF on gfx942 — the part it is measurably
wrong on — and unchanged everywhere else. Verified: a default gfx942 emit is now
byte-identical to a `PLOW_FUSE_QUANT=0` emit; a default gfx950 emit is byte-identical to an
explicit `PLOW_QNORM_FUSE=1` emit (so CDNA4 still folds and still gets PR #56's win); and
`PLOW_QNORM_FUSE=1` still reaches the fold on gfx942 for whoever debugs the arm. **The arm
itself is NOT fixed** — the gfx942 `d_rmsnorm` t3/t4 path is the thing to repair, and until it
is, this default is a guard rather than a cure.

Every arm below is emitted without the fold, uniformly, so the comparison is internally fair.

### 6b. The emitter's LDS arena was the CDNA4 one, so batched decode on gfx942 was never safe

`gemv_qkv_rows` / `gemv_glu_rows` read `x` ONLY through LDS, so the emitter choosing those
fused opcodes is a promise that `M*K` fits the decode object's GEMM arena. `devgen`'s mirror
of that arena was the gfx950 value on every part:

| part | decode tile | buffers | arena, halves |
|---|---|--:|--:|
| gfx950 / sm_120 | 256x256x64 | 2 | 73,728 |
| gfx942, `build_gfx942.sh` default | 192x256x32 | 1 | 17,920 |
| gfx942, `PLOW_OCC4=1` — the SHIPPED decode profile | 128x256x32 | 1 | **15,360** |

At Gemma-4-12B's `hidden = 3840` the emitter therefore fused every batch up to **19** onto an
object holding **four rows**. Measured consequences, both reproduced:

* `PLOW_DECODE_BATCH=16` on the occ4 objects, three concurrent requests →
  `Queue ... inactivated due to async error: HSA_STATUS_ERROR_EXCEPTION`;
* the same blob on the wider default tile → no fault, fluent wrong text.

This is §6g-BATCH's bug (gfx950, `hidden = 5376`, slots 13/14/15 fluent-but-wrong) with the
gfx942 arena, and it means **no batched-decode blob has ever been correct on this box** —
the `g12b-fp8-b4` / `-b8` / `-b16` assets in `/workspace/assets/gfx942` included. It went
unnoticed because batched decode had only ever been serve-gated on gfx950, and because at
`PLOW_DECODE_BATCH=1` the bound is met on every part, so every shipped B=1 blob is unaffected
(verified byte-identical after the fix).

The fix makes the constant arch-aware and takes the OCC4 value on gfx942, because the object
profile is chosen when the objects are built, long after emit — the smaller of the two is the
only value right for both.

**And it is what turns the ladder from a shape into a specialisation.** With the true arena,
one ladder blob emits *structurally different* programs per rung:

| rung `t` | decode packets | NRN fold | GLU fusion |
|--:|--:|:--:|:--:|
| 1  | 527 | yes | yes |
| 2  | 527 | yes | yes |
| 4  | 622 | **no** (`4*3840+16 > 15,360`) | yes |
| 8  | 718 | no | **no** (`8*3840 > 15,360`) |
| 16 | 718 | no | no |

A captured-graph replay cannot express that: it is the same graph at every size. Here the
narrow rungs keep two fusions the wide rungs must give up, and the decision is made per rung
by an LDS capacity — in one blob.

---

## 7. The concurrency sweep

Gemma-4-12B fp8/w8a8, gfx942 (1 GPU, 304 CU), `scripts/bench_speed.sh`, in_len 1024,
out_len 128, **16 requests per cell**, concurrency 1/2/4/8/16, ROCm 7.2.4. All four arms in
ONE lock hold, same binary, same checkpoint, all emitted without the fused activation quant
(§6a) — which, since this branch defaults that fold OFF on gfx942, is now simply the DEFAULT
emit for this part. Every arm passed its own `Paris` coherence gate before its sweep.

| arm | blob | decode object |
|---|---|---|
| **a** | `PLOW_DECODE_BATCH=1` — today's shipped serving blob | `PLOW_OCC4=1`, `PLOW_GEMV_MM=1` |
| **c** | **`PLOW_DECODE_BATCH_LADDER=1,2,4,8,16`** | `PLOW_OCC4=1`, `PLOW_GEMV_MM=16` |
| **b** | `PLOW_DECODE_BATCH=16` | `PLOW_OCC4=1`, `PLOW_GEMV_MM=16` |
| **a16** | `PLOW_DECODE_BATCH=1` — the SAME blob as arm a | `PLOW_OCC4=1`, `PLOW_GEMV_MM=16` |

`a16` is the control that makes the rest readable: it is arm a's blob on arm c's object, so
the difference between `a` and `a16` is the OBJECT alone and the difference between `a16` and
`c` at concurrency 1 is the LADDER alone.

### TPOT, ms/token (lower is better)

| conc | a (B=1, MM=1) | **c (LADDER)** | b (B=16) | a16 (B=1, MM=16) |
|--:|--:|--:|--:|--:|
| 1  | **10.92** | 68.49 | 109.74 | 67.58 |
| 2  | 13.01 | **84.45** | 119.60 | 73.92 |
| 4  | 12.01 | **85.36** | 115.18 | 70.31 |
| 8  | 12.11 | **103.53** | 120.03 | 68.99 |
| 16 | 12.20 | **126.29** | 125.63 | 68.32 |

### Aggregate output throughput, tok/s (higher is better)

| conc | a (B=1, MM=1) | **c (LADDER)** | b (B=16) | a16 (B=1, MM=16) |
|--:|--:|--:|--:|--:|
| 1  | **80.8** | 14.4 | 9.1 | 14.5 |
| 2  | 69.2 | **23.0** | 16.4 | 13.3 |
| 4  | 74.3 | **45.1** | 33.7 | 14.0 |
| 8  | 73.8 | **72.4** | 62.7 | 14.3 |
| 16 | 73.3 | **106.4** | 106.1 | 14.4 |

### Mean TTFT, ms — the axis that shows batching is real

| conc | a (B=1) | c (LADDER) | b (B=16) |
|--:|--:|--:|--:|
| 1  | 196.8 | 197.7 | 198.3 |
| 2  | 1947.7 | 377.9 | 414.5 |
| 4  | 4713.5 | 472.9 | 510.5 |
| 8  | 9309.8 | 818.7 | 880.4 |
| 16 | 13318.0 | 2356.5 | 2512.0 |

## 8. What the sweep says

**The ladder does half of what it claimed, exactly, and the half it misses has a measured
single cause.**

1. **It matches a fixed B=16 blob at concurrency 16 — the throughput half of the claim.**
   106.4 vs 106.1 tok/s and 126.29 vs 125.63 ms TPOT: **+0.3% throughput, +0.5% TPOT**, i.e.
   the widest rung IS the B=16 program and carrying four extra rungs costs nothing measurable.

2. **It BEATS the fixed B=16 blob at every concurrency below 16**, which is the point of
   having rungs at all: −37.6% TPOT at concurrency 1 (68.49 vs 109.74), −25.9% at 4, −13.7%
   at 8, and +58% / +34% / +15% throughput at 1 / 4 / 8. A deployment that must be sized for
   16 concurrent users no longer pays B=16 prices when it has one.

3. **It does NOT reach B=1's latency at concurrency 1 — 68.49 vs 10.92 ms — and the cause is
   the OBJECT, not the ladder.** Arm a16 is arm a's own B=1 blob on the MM=16 object and
   measures **67.58 ms**. So:

       ladder rung 1                    68.49 ms
       the same B=1 program, same object 67.58 ms   <- the ladder costs +1.3%
       the same B=1 program, MM=1 object 10.92 ms   <- the object costs 6.2x

   The ladder's program selection is essentially free (+1.3%, and it must pick a rung, seed
   16 ids and read 16 slots' worth of `in.ids` to do it). The 6.2× is `PLOW_GEMV_MM=16` in
   the ONE decode object every rung shares — `object_name()` keys on `Phase::Decode`, not on
   which decode program is running, so `B_max` sets the register/issue profile for the
   one-row rung as much as for the sixteen-row one. That is the missing half of the design,
   it is named in §4 as the known object-level limit, and it is now priced.

4. **Concurrency really is batching here, and the harness's own caveat is answered three
   ways.** `bench_speed.sh` warns it cannot tell batching from queueing. For arm a it is
   queueing and the evidence is direct: throughput is FLAT at 73–81 tok/s across a 16×
   concurrency range while TTFT rises 68× (196.8 → 13318 ms) — requests waiting, not
   sharing. For arms b and c it is batching: throughput rises 7.4× / 11.7× over the same
   range, TTFT stays inside 2.5 s, and TPOT CLIMBS with concurrency (68.5 → 126.3), which is
   the signature of rows sharing one weight read. And the ladder's own log shows the rung
   moving with the load — `decode ladder rung rung=1 occupied=1`, `rung=2 occupied=2`,
   `rung=4 occupied=3` — which no amount of queueing produces.

## 9. Correctness

The ladder touches sequence/slot bookkeeping, where a bug corrupts one user's output with
another's. Gate, run on the ladder asset immediately before the sweep, in the same lock hold:

**Phase 1 — three DIFFERENT prompts issued simultaneously.** Each answer must match its own
question:

    [PASS] What is the capital of France? Answer with one word.   -> 'Paris'
    [PASS] Compute 17*23. Reply with just the number.             -> '391'
    [PASS] What is the chemical symbol for gold? One token only.  -> 'Au'

**Phase 2 — the rung TRANSITION.** One long request alone (rung 1), three more joining
three seconds into its generation (rung 2, then rung 4), all four checked:

    [PASS] Compute 17*23. Reply with just the number.             -> '391'
    [PASS] What is the chemical symbol for gold? One token only.  -> 'Au'
    [PASS] What is the capital of Japan? Answer with one word.    -> 'Tokyo'
    [PASS] the long solo request that spanned the transition
           -> '...that the dot products required for each cell in the output matrix are
               computed in parallel, which drastically reduces the time required compared
               to serial processing.  The capital of France is Paris.'

    CORRECTNESS: PASS

and the server's own rung log over that gate, which is what proves the transition was
exercised rather than assumed:

    decode ladder rung rung=1 occupied=1
    decode ladder rung rung=2 occupied=2
    decode ladder rung rung=4 occupied=3
    decode ladder rung rung=1 occupied=1     <- phase 1 drains
    decode ladder rung rung=2 occupied=2
    decode ladder rung rung=4 occupied=3     <- phase 2's three join mid-generation
    decode ladder rung rung=1 occupied=1

The long request's KV was written under rung 1, extended under rungs 2 and 4, and finished
under rung 1 again — and it still answers its own question correctly at the end. That is the
slot-invariance argument of §1 executed rather than argued.

**No cross-contamination in any run.** Every failing transcript recorded during the
investigation (§6) was WRONG-FOR-EVERYONE, never one user's answer appearing in another's
stream — consistent with the two causes found (a bad prefill and an LDS overrun), neither of
which mixes slots.

## 10. Verdict, and what to do next

**Does the ladder achieve "latency of B=1 at concurrency 1, throughput of B=16 at
concurrency 16"?** Half of it, exactly, and the other half is blocked by a named component
that is not the ladder:

*(§11 then supersedes the configuration, not the finding: a ladder capped at `B_max = 4`
beats the `1,2,4,8,16` one measured here on BOTH axes, because the object tax scales with
`B_max`. Read §8 as the anatomy and §11 as the recommendation.)*

* **throughput of B=16 at concurrency 16: YES** — 106.4 vs 106.1 tok/s, 126.29 vs 125.63 ms
  TPOT. Carrying four extra rungs costs +0.3% throughput and 15.8 MB of blob.
* **latency of B=1 at concurrency 1: NO** — 68.49 vs 10.92 ms. But the ladder's own
  contribution to that gap is **+1.3%** (68.49 against 67.58 for the identical B=1 blob on
  the identical object). The remaining **6.2×** is `PLOW_GEMV_MM=16` in the single decode
  object all rungs share.
* **against a fixed B=16 blob — the thing a throughput deployment ships today — the ladder is
  strictly better at every concurrency below 16 and equal at 16.** That is the result that
  stands on its own and needs no further work.

### The one thing that would finish it: a decode object per rung

`object_name(phase, variant, arm, sched)` (crates/plowrt/src/exec/amd.rs:436) keys the code
object on the PHASE. Every decode rung therefore runs on one object compiled at one
`PLOW_GEMV_MM`, which must cover the widest rung. Adding the GEMV bucket as an object axis —
`interp_decode_fp8_gq_mm1.elf`, `_mm4`, `_mm16` — and selecting per PROGRAM rather than per
phase would let rung 1 run the MM=1 object it wants. The measurement above says what that is
worth: **6.2× on the one-row rung**, which would take the ladder from 68.49 to ~10.9 ms at
concurrency 1 while leaving concurrency 16 where it is. The cost is build time (one decode
object per bucket) and resident code (a few hundred KB each), not design work — the loader
already opens three objects per model and refuses mismatches.

Until that exists, **the practical recommendation is a NARROW ladder**: `B_max` sets the
object profile for every rung, so a deployment should pick the smallest `B_max` that covers
its real concurrency rather than the largest the hardware allows. §11 measures this and it is
not a small effect — `1,2,4` on `PLOW_GEMV_MM=4` objects beats `1,2,4,8,16` on `MM=16` on
BOTH axes at once (+27% peak throughput AND 3.4x better concurrency-1 latency).

### Follow-ups, in the order they matter

1. **Repair the gfx942 fused-quant arm** (§6a). Defaulting it off here stops the wrong
   output but forfeits PR #56's Gemma <=4k TTFT win on this part; the `d_rmsnorm` t3/t4 path
   is where to look. Separately and more urgently: ~~**re-point the Gemma cross-gate at a
   FRESHLY EMITTED blob**~~ — **DONE 2026-08-08**, `scripts/gemma_xgate.sh`, transcripts in
   `gemma-xgate-fresh-blob.md`. It re-emits every run, and it was PROVEN TO FAIL on
   `PLOW_QNORM_FUSE=1` before being trusted to pass on the default. A gate that re-uses a
   stored asset and rebuilds only the objects cannot catch an emitter regression by
   construction, and that is the only reason this one survived a merge.
2. ~~**Re-check every batched-decode number ever recorded on gfx942**~~ — **DONE 2026-08-08**,
   branch `gate-fix`. Audit and verdicts: `gemma-xgate-fresh-blob.md` §3. Retracted in place:
   `g12b-b8_b8_tp1_general.csv`, `g12b-b8_b8ctx128_tp1_general.csv`,
   `gemv-mlp-and-tensile.md` §"Batched decode", `fusion-review-and-crossover-sweep.md` §3,
   `glm52-tp4-pp2-evaluation.md` §D.4(b)+§E.1, `glm52-experiments.md` item 2. Every DOCUMENT keeps
   its retraction in place; the two raw CSVs were later REMOVED from the tree before the merge
   to main, because a file that is a timing of wrong math should not be readable at all.
   **This document's own §7 and §11 tables are the exception and are SOUND**: `git log` puts
   every commit that wrote them (d9ce9f9, b807b32, d26270f, e55a41f) after the arena fix
   2130f04, and each arm carries its own served `Paris` gate (§9). Every B=1 number on this box
   is likewise safe — B=1 emits are byte-identical across the fix (§6b), so no single-row
   measurement was re-run.
3. **Per-rung decode objects** (above) — the 6.2×.
4. **A ladder on Kimi-K3** is a `for` loop away (`k3_build_model` already builds one program
   per bucket); the constraint is that every rung must carry `RowKind::Sequences` so the KDA
   state keeps its per-slot stride. The site now asserts rather than silently emitting one
   rung.
5. **A ladder on GLM-5.2 needs batched decode for GLM first** (§0) — that is a build, not a
   knob, and it is the only way this lever reaches the model the campaign cares about.

### Where this sits next to the campaign's other throughput numbers

These are **Gemma-4-12B** numbers, and they are not interchangeable with GLM's. A sibling
measured GLM-5.2 TP8 on the canonical `glm52-tp8-final2` asset over the same concurrency
range and found aggregate throughput **FLAT at 31.5 -> 28.9 tok/s while TTFT rose 50.6x** —
pure queueing, `batch=1`, no batching to speak of. That is the correct control for "what does
GLM do today", and it is why a decode-batch lever matters. But **this ladder cannot be
pointed at GLM yet**: §0 is not a scoping preference, it is that `glm_emit_full` emits a
one-row decode program and there is no batched decode to ladder. The honest statement is that
the ladder is proven on the dense-GQA family (1.84x aggregate throughput over the shipped B=1
blob at concurrency 16) and that reaching GLM's flat 31.5 tok/s with it requires building
GLM batched decode first — the emitter now refuses the combination loudly rather than
implying otherwise.

### Not measured here

* **vLLM at matched concurrency.** ~~Outstanding.~~ **DONE — see
  `glm52-decode-ladder-vs-vllm026.md`** (2026-08-08). `sweep_client.py` (the `bench_speed.sh`
  client, extracted standalone so BOTH engines are driven by the same code and the same metric
  definitions) and `bench_vllm26_native.sh` are in tree, the client is calibrated against
  `vllm bench serve` to ≤3.3% on TPOT, and the sweep ran at the same points. Result: **no
  crossover** — vLLM 0.26 + AITER is ahead on both axes at every concurrency, 1.44× on
  throughput at concurrency 1 widening to 6.81× at 16, and the ladder's contribution is to move
  the concurrency-16 deficit from 12.5× to 6.8×. Mechanism: plow's widest batched blob amortises
  the weight stream **1.39×** across 16 rows where vLLM gets **10.08×**, so §11's `B_max = 4`
  recommendation is a containment that fixes the aggregate ceiling at ~145 tok/s. The
  `gemv_rows<MM>` object tax is the whole gap.
* **Rung quantisation, priced from the other engine.** vLLM's decode is *also* rung-quantised —
  its `cudagraph_capture_sizes` below 16 are `[1, 2, 4, 8, 16]`, the same set — while its
  scheduler admits any batch size (`Running: 3/6/9/12/15 reqs` observed). Measured off-rung cost
  there: **+0.7% TPOT** at the worst point. That is independent support for §9's "finer rungs are
  not worth buying yet", from a stack where the tax is visible. plow's OWN off-rung cost is still
  unmeasured, and cannot be assumed equal — see below.
* **Rungs finer than powers of two on the GPU.** They emit and the blob is correct
  (`1,3,6,12,16` verified); no timing was taken, for the reason below.

### Granularity: are 3/6/12 worth it?

Not on this evidence, and the sweep says why without needing another run. The wasted-row cost
of a too-wide rung is bounded by the TPOT spread across rungs, and on the ladder arm that
spread is 68.49 → 126.29 ms over a 16× row range — a 1.8× cost for 16× the rows, because the
decode is weight-stream-bound and the GEMV re-reads the same weights whatever `M` is. So the
worst-case granularity loss at `1,2,4,8,16` (occupancy 9 running on rung 16) is a fraction of
that 1.8×, while an extra rung costs a fifth of the blob's packet stream. Finer rungs are
capability-complete (`PLOW_DECODE_BATCH_LADDER=1,3,6,12,16` emits and runs) and are not worth
spending on until the 6.2× object tax is gone — after which the per-rung spread, and so the
value of finer treads, will be a different and much larger number.

---

## 11. The object tax scales with `B_max` — so a NARROW ladder wins on BOTH axes

§8 attributes the concurrency-1 gap to the shared decode object. If that is right, the tax
should shrink with `B_max`, and a ladder capped lower should be better. It is, and it is:
same battery, same day, `PLOW_GEMV_MM=4` objects, three more arms.

**The object tax alone** — the SAME `PLOW_DECODE_BATCH=1` blob, three objects, concurrency 1:

| decode object | TPOT ms | vs MM=1 |
|---|--:|--:|
| `PLOW_OCC4=1 PLOW_GEMV_MM=1`  | 10.92 | 1.00x |
| `PLOW_OCC4=1 PLOW_GEMV_MM=4`  | 20.03 | **1.83x** |
| `PLOW_OCC4=1 PLOW_GEMV_MM=16` | 67.58 | **6.19x** |

One blob, one program, one packet stream; only the compiled GEMV row bucket differs. The
occupancy cliff table does not show it — all three build at 108 VGPR / 30,768 B LDS / zero
spill — so this is issue/latency cost inside `gemv_rows<MM>`, not an occupancy loss.

**TPOT, ms/token**

| conc | a (B=1, MM=1) | c (LADDER 1..16, MM=16) | **c4 (LADDER 1,2,4, MM=4)** | b4 (B=4, MM=4) | a4 (B=1, MM=4) |
|--:|--:|--:|--:|--:|--:|
| 1  | **10.92** | 68.49 | 20.12 | 22.27 | 20.03 |
| 2  | 13.01 | 84.45 | **27.49** | 28.28 | 23.38 |
| 4  | 12.01 | 85.36 | 30.68 | **26.90** | 21.43 |
| 8  | 12.11 | 103.53 | **27.66** | 34.27 | 21.59 |
| 16 | 12.20 | 126.29 | **27.65** | 70.07 | 21.37 |

**Aggregate output throughput, tok/s**

| conc | a (B=1, MM=1) | c (LADDER 1..16, MM=16) | **c4 (LADDER 1,2,4, MM=4)** | b4 (B=4, MM=4) | a4 (B=1, MM=4) |
|--:|--:|--:|--:|--:|--:|
| 1  | **80.8** | 14.4 | 46.5 | 42.3 | 46.7 |
| 2  | 69.2 | 23.0 | **67.1** | 65.3 | 40.4 |
| 4  | 74.3 | 45.1 | 118.5 | **133.5** | 43.9 |
| 8  | 73.8 | 72.4 | **134.7** | 107.4 | 43.6 |
| 16 | 73.3 | 106.4 | **134.8** | 54.2 | 44.0 |

**The narrow ladder dominates the wide one on BOTH axes at once**, which is not a trade-off
anyone had to make:

* **peak throughput 134.8 vs 106.4 tok/s (+27%)**, and
* **concurrency-1 TPOT 20.12 vs 68.49 ms (3.4x better)**.

`B_max = 16` is simply not worth buying on this hardware: the 6.19x object tax it imposes on
every rung costs more than the extra batch width returns. The best configuration measured is
**`PLOW_DECODE_BATCH_LADDER=1,2,4` on `PLOW_GEMV_MM=4` objects** — 134.8 tok/s at concurrency
16 against the shipped B=1 blob's 73.3 (**1.84x aggregate throughput**) for 20.12 ms TPOT at
concurrency 1 against its 10.92.

It also beats the fixed `B=4` blob it is built from at every concurrency at or above 8 —
134.8 vs 54.2 tok/s at concurrency 16, because `b4` has only four slots and the other twelve
requests QUEUE (its TTFT explodes to 23.4 s and its TPOT to 70 ms) while the ladder's slot
table is sized at its widest rung. That is the ladder earning its keep from the other
direction: it is not only cheaper than a wide blob at low load, it is wider than a narrow one
at high load.

**Corrected recommendation, now measured rather than inferred:** ship the ladder at
`B_max = 4`. Raising `B_max` past the object's comfortable GEMV bucket is negative on both
axes until decode objects are selected per rung (§10).
