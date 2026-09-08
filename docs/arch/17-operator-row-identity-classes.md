# Operator row-identity classes, and the verdict on runtime fusion v1

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
