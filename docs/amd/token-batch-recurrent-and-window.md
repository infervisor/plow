# Unified token batch, Phase 3: the recurrent axis and the windowed-attention axis

Status 2026-09-08, gfx942 / MI300X, ROCm 7.14.0 (nix toolchain). Two of the six Phase 3 items of
[`plans/unified-token-batch.md`](../../plans/unified-token-batch.md), taken independently as that
section says they may be. Everything below is a body-level or host-level result. **No serving
claim is made and none is implied**: no Qwen3.5 or Kimi-K3 checkpoint is on this host, and the
Gemma-4 blob was not re-emitted.

## The descriptor this work assumes

Phase 1a/1b (`PlowTokenBatch`, the `PlowProgram` pointer, `RowGather`, the shared row resolver) is
in flight elsewhere and is **not** on this branch. Rather than guess at its final field names, both
axes were built against the metadata the ISA already carries, which §4.1 of the plan explicitly
tells the token-batch contract to reuse:

| Plan field (§4.1) | What this work consumed |
|---|---|
| `spans[R]` | `PlowPrefillSpan` (`runtime/common/dev_isa.h`): `row0`, `n_rows`, `slot`, `flags`, `kv_row0`, `kv_len`, `state_slot`, `program`. 32 bytes, unchanged. |
| `positions[M]` | `span->kv_row0 + local_row`, exactly as `plow_mixed_row` and `plow_packed_prefill_position` already derive it. There is no separate position array on this path. |
| `active[M]` (park mask) | **Two existing conventions, not one** — see the polarity note below. `PlowProgram::prefill_parked` for the packed-prefill row axis; the per-op tensor operand (`active` for the GDN family, `parked` for KDA) for the decode slot axis. |
| `real_rows = M` | `PlowProgram::n_prefill_rows` via `mixed_rows()`. |
| `sample_rows = S` | Not consumed. The compact terminal segment does not exist yet; see "What is blocked" below. |

Nothing in `runtime/common/dev_isa.h`, `crates/packet/`, `runtime/common/mixed_step.h`,
`crates/plow-asset/`, the `PlowProgram` Rust mirror or the ABI-lock test was modified. When the
`PlowTokenBatch` descriptor lands, the consumers added here need it to expose the same five
quantities per row; the audit and the two device gates are written against the *quantities*, not
against the struct.

### The polarity note, and why it matters

§1 of the plan says the GDN family's `active[B]` is "the ISA's existing name for 'this row is
padding; write nothing'" and to "extend that convention rather than inventing a second one." A
second one already exists, and both are load-bearing:

| Family | Tensor | Sense | Filled by |
|---|---|---|---|
| Qwen3.5 GDN (ops 136-142) | `in.active`, `i32[batch]` | **1 = run** | `GpuEngine::upload_active_slots` (CUDA only) |
| Kimi-K3 KDA (ops 111, 112, 120, 125) | `in.parked`, `u32[T]` | **nonzero = skip** | `AmdEngine::upload_parked` |
| Packed-prefill rows (all families) | `PlowProgram::prefill_parked` | **nonzero = skip** | `stage_packed_prefill` |

The KDA sense was chosen deliberately (`crates/devgen/src/kda.rs`): an unwritten or zeroed tensor
means *every row participates*, so a caller that has never heard of the mask cannot silently drop
rows. That is a better default than `active`'s, and it is why the two device gates below run the
same schedule through both — a descriptor that fills only one of them will pass one gate and fail
the other.

## Axis A — recurrent

### A1. Packed-vs-isolated bit identity, GDN (`runtime/tests/qwen_gdn_gfx942_test.hip`)

The ten GDN arms already honour `active` and the existing f64 oracle already parks a slot. What
neither could see is a **mapping** property rather than an arithmetic one: whether a decode row
produces the same *bytes* alone as it does packed beside other requests' rows. §9 asks for
bit-identity; a 4e-3 tolerance would pass a body that leaked one request's row into another's.

Two rounds over six slots, mask changing between them:

```
slot     0  1  2  3  4  5
round 0  .  A  A  .  A  .
round 1  A  .  A  A  .  .
```

Slot 2 carries state through both; 1 and 4 have theirs frozen by a park after use; 0 and 3 resume
from a state a round stale (state-slot reuse); 5 is never live and must still hold the 0xa5 poison.
The isolated arm does not *submit* a parked row — that is what "isolated per-request execution"
means, and it is why a body that writes through a zero mask fails rather than being compared with
itself.

```
  gdn_conv out               packed==isolated YES   padding frozen YES
  gdn_conv history           packed==isolated YES   padding frozen YES
  gdn_step out               packed==isolated YES   padding frozen YES
  gdn_step state             packed==isolated YES   padding frozen YES
  gated_norm                 packed==isolated YES   padding frozen YES
  q_gate_split q             packed==isolated YES   padding frozen YES
  q_gate_split gate          packed==isolated YES   padding frozen YES
  sigmoid_gate               packed==isolated YES   padding frozen YES
  qwen_rmsnorm               packed==isolated YES   padding frozen YES
  headnorm_rope flat         packed==isolated YES   padding frozen YES
  headnorm_rope ring         packed==isolated YES   padding frozen YES
```

Identical at `PLOW_QWEN_GDN_VROWS` 1, 2, 4 and 8. VROWS is the knob that changes the row-to-wave
mapping, so it is the arm most likely to make a row's result depend on how many rows share the
launch; it does not.

**Negative control**: deleting the `active` guard from `d_qwen_rmsnorm` alone (one line, separate
include tree, same toolchain) turns exactly that row into `no / no`, every other row stays `YES`.

The file is now registered with `ctest` under the `gpu;gfx942` label. It built under CMake before
but had no `add_test`, unlike its KDA sibling.

### A2. Packed-vs-isolated bit identity, KDA (`runtime/tests/kda_step_cdna3_test.hip`)

The same schedule on the batched-decode axis of `d_kda_state_step_g`, at K3 geometry (H=96, D=128,
BV=16), through `parked` instead of `active`. Output and the whole f32 state are both
`packed==isolated YES`, padding frozen `YES`.

Before this, that file's `k_step` entry passed `parked = nullptr`, so **nothing in tree exercised
the per-row mask on the decode axis**. The `parked` fixtures further down belong to the packed-
*prefill* span path, which is a different axis: its rows are tokens of a span, not slots.

**Negative control**: passing `nullptr` for `parked` turns all three checks into `no` — and every
*other* line of that file still passes, including both f64 modes and both packed-span sections.
This gate is the only thing in tree that sees it.

### A3. The D-class span limit: refused, not truncated

`crates/plowrt/src/exec/amd_packed.rs::recurrent_span_limit` is §3's D class made executable,
restricted to the operator family where the plan says it binds. It runs once at load, into
`AmdProg::packed_recurrent_spans`.

| Opcodes | Class | Disposition |
|---|---|---|
| `KdaChunkPrepare/Intra/Wu/Carry`, `KdaConv3`, `KdaStateStep`, `KdaStateStepG` | D, per-span arm present (`d_kda_*_packed_bt64`, `d_kda_conv3_packed`, `d_kda_state_step_packed`) | unbounded |
| `KdaGate`, `KdaGatedNorm` | A over the packed row axis | unbounded |
| `KdaDecodeFused` | D, single-sequence, no per-span arm | **1 span** |
| `KdaConv`, `KdaConvStateStepG`, `Mamba2Scan`, `QwenGdnConvPrefill`, `QwenGdnQkvPrep`, `QwenGdnGatePrep`, `QwenGdnPrefill` | D, no per-span arm at all | **refused, named** |
| `QwenGdnConv/Step`, `QwenGatedNorm`, `QwenQGateSplit`, `QwenSigmoidGate`, `QwenRmsNorm`, `QwenHeadNormRope` | B on the *slot* axis | **refused, named** — a packed-prefill row is a token of a span, so binding these to a span table is a category error, not a missing arm |

**The hole this closes.** `check_packed_prefill_program` was a series of *"if this program needs
that family, check its objects and its shape"* — dense, MLA, KDA. A family none of the three
recognised fell straight through to `Ok`, and the route then staged a span table for a program with
no arm that reads one. On AMD that is not a slow path: the interpreter's dispatch `default:` is
`/* PLOW_DOP_NOP */`, which writes nothing and does not trap, so every span after the first would
run against the previous request's state and the run would complete fluently. §3's rule — "an
opcode with no classification is treated as C and refused" — is now applied here as a default-deny.

**Refused, not truncated.** `stage_packed_prefill` compares the plan's span count against the limit
and errors, naming both. Executing the first `limit` spans of a larger plan is a silently *short*
answer: the dropped requests keep their cursors and the caller commits their KV frontiers on the
strength of a launch that never covered them. `AmdEngine::packed_prefill_span_limit` exposes the
same number so a scheduler can pick a legal plan before touching cursors (§7) — the half that keeps
the refusal from becoming a liveness bug.

Kimi-K3 is the only recurrent family that reaches this route today and every KDA operator its
packed programs carry has a per-span arm, so its limit is `u32::MAX` and nothing about it moves.

### What Axis A does NOT deliver

* **No serving.** `plowc` still refuses a Qwen3.5 emit for an AMD target by name
  (`qwen35_amd_emit`); `PLOW_QWEN_GDN` is still 0 in every shipping object; there is no Qwen3.5
  checkpoint on this host. See [qwen35-gdn-mi300x.md](qwen35-gdn-mi300x.md).
* **No batched variable-length recurrent prefill.** That is the plan's §10 work item and it stays
  open. What landed is the *limit* being enforced and named instead of assumed.
* **`KdaConv` (88) and `KdaStateStep` (102) still carry no mask operand.** `d_kda_state_step_t`
  honours `parked`, but op 102's dispatch passes `nullptr` because all eight tensor slots are used
  and the opcode predates the mask; `KdaConv`'s dispatch hard-codes `bstride = 0` for the same
  reason. Neither is reachable from a K3 emit (devgen emits 111/112), and both are now refused on
  the packed route by name — but a *decode* packet built with either and `B > 1` would still be
  silently wrong. Closing that needs an `i[]`/`j[]` demotion like op 112's, in the shared operand
  contract, which this work does not own.

## Axis B — windowed attention and the softcap tail

### B1. The window bound, scored (`runtime/tests/mixed_flash_window_gfx942_test.hip`, new file)

`mixed_flash_prefill_gfx942_test.hip` fills V with a constant 1.0 and checks output *placement*.
It is blind to the window by construction — softmax over any subset of identical value rows is 1.0,
so its `window = 64` could be 0 or 3 and every assertion would still pass. **Nothing in tree scored
a windowed `d_flash_prefill` at a non-zero `q_pos0` against arithmetic**, and Gemma-4-31B runs 50 of
its 60 layers windowed at 1024.

`d_flash_prefill` derives every row's absolute position as `q_pos0 + row`, and the window bound
from it, at four places that can disagree: the workgroup-uniform `win_lo`, the
`kv_lo = (win_lo/BKV)*BKV` tile carve and its split partition, and the per-element
`(qg - kg) < window` mask. Under packing `q_pos0` is not the packet immediate — the emitter writes
`i[4] = 0` — but `span->kv_row0`, substituted per span by the loop in `interp.hip`. **So the
per-span window bound already exists**; the open question was whether it is right at the
boundaries.

21 checks, all passing, rms 1.8e-3..2.2e-3 against a 4e-3 bar (the bf16 quantum — the bound is
exact and what remains is the storage format):

* HD=256 single span: window 0; window inside the prefix; window straddling the chunk start;
  window wider than the whole cache (must degenerate to causal); `win_lo` not BKV-aligned; 1-row
  and 3-row final chunks; a 64-row ring the span wraps.
* `nsplit` 3 and 4 with `d_flash_merge`, windowed and unwindowed.
* HD=128, the Llama/Qwen shape.
* Three spans at different slots, prefixes (0, 96, 200) and lengths (24, 9, 31) — §9's "different
  live KV lengths" — scored, and byte-identical to the same three run one at a time.

**Two negative controls**, each a one-token edit to a copy:

| Injected fault | Result |
|---|---|
| pass `0u` for the window | 14 of 21 FAIL at rms 0.27..0.96. The seven that stay green are exactly the ones that should: the three window=0 cases, the window-wider-than-KV case, and the three identity checks, which do not depend on the window being *applied*, only on it being applied equally. |
| pass `spans[0].kv_row0` for every span's `q_pos0` — §3's class-C hazard written out literally | all six packed-span checks FAIL, including all three byte-identity ones. |

The span loop in the test kernel is a **transcription** of `PLOW_AMD_FLASH_PREFILL` (offsets
included) because the macro lives inside `exec_flash_prefill` and cannot be called from a test. If
the two ever disagree, the transcription is the bug — keep them together. **No file the Phase 2
dense-GQA lane owns was modified**: neither `runtime/amd/op_attention.h` nor `runtime/amd/interp.hip`.

### B2. The softcap tail and tied/untied heads — read, not changed

Traced end to end and found already correct on the route that exists, so nothing was written:

* **`DevOp::SoftCap` exists** (op 7, `t0=out t1=x`, `i0=n`, `f0=cap`, `cap*tanh(x/cap)`) and the
  AMD arm is already row-count aware:
  `d_softcap(..., (i0/i1) * PLOW_RUNTIME_ROWS(i1), ...)` under `PLOW_MIXED_STEP`.
* The emitter (`crates/devgen/src/lib.rs`) emits it only when `c.softcap > 0` — Gemma's cap 30 —
  because `d_softcap` divides by `cap` and would produce NaN at 0. Llama/Qwen skip the packet
  entirely, which is the "softcap absent" half of §9's Head row.
* `crates/plowrt/src/exec/mixed_program.rs` rewrites the packet for the mixed program
  (`i[0] *= dcap; i[1] = dcap`) and `plow-asset` validates that rewrite. Emitted alone, `i[1]` is 0
  and the mixed arm would trap — it is the *rewrite* that makes the packet legal, not the emit.
* **Tied and untied are covered by construction, not by an arm**: `head_w = if c.tied { n.emb }
  else { n.head }` picks a *tensor handle*, and the mixed rewrite only touches immediates
  (`i[0] = dcap`, `i[4] = 0` on the head GEMV). There is no second code path to gate.

### What Axis B is blocked on

**The compact terminal segment does not exist on this branch.** There is no `RowGather` opcode
anywhere in `crates/` or `runtime/` (Phase 1c). Consequently the mixed route's `S` is *the decode
rows*: `mixed_program.rs` rewrites the prefill head from `(M=1, a_row0 = t-1)` to
`(M=dcap, a_row0 = 0)`, and `PLOW_SAMPLE_ROWS` resolves to the decode-row count. A prompt that
finishes inside the step — request C of §6.1, the plan's stated acceptance case — gets **no sample
row**. That is the `split_terminal_prefill` / `finish_prefill_batch` defect §1 names, and it is
Phase 1c's to remove, not this axis's. "SoftCap stage in the terminal segment" cannot be
demonstrated until the segment exists; what is demonstrated is that the SoftCap packet is already
correct over a runtime row count.

## Reproducing

```
# Axis A, GDN. Device object inside nix develop, host half outside it (the nix ROCm links
# against a newer glibc than this host's libstdc++ provides).
nix develop /app/plow --command \
  $PLOW_HIPCC --offload-arch=gfx942 -O3 -w -DQGDN_DEVICE --genco \
    runtime/tests/qwen_gdn_gfx942_test.hip -o /tmp/qgdn.co -Iruntime/amd -Iruntime/common
g++ -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm-7.2.4/include \
    runtime/tests/qwen_gdn_gfx942_test.hip -o /tmp/qgdn -L/opt/rocm-7.2.4/lib -lamdhip64
perf-data/tools/gpulease -n 1 qgdn /tmp/qgdn /tmp/qgdn.co

# Axis A, KDA. Same two passes with -DKDA3_DEVICE on runtime/tests/kda_step_cdna3_test.hip.

# Axis B. One hipcc pass, inside nix develop.
$PLOW_HIPCC --offload-arch=gfx942 -O3 -Iruntime/amd -Iruntime/common -c \
    runtime/tests/mixed_flash_window_gfx942_test.hip -o /tmp/mixed_window.o
c++ /tmp/mixed_window.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/mixed_window
perf-data/tools/gpulease -n 1 mixed-flash-window /tmp/mixed_window

# Axis A3.
CARGO_TARGET_DIR=/app/plow/target-glm53 cargo test --release -p plowrt --features hsa \
    --lib exec::amd_packed
```

No shipping code object changes: the only files touched under `runtime/` are `tests/` and
`bench/CMakeLists.txt`, so `scripts/asm_audit.py --contract` sees the same 45 objects it did before.

## What the plan got wrong or under-specified

1. **"The ISA's existing `active[B]` convention" is two conventions with opposite polarity** (§1,
   §5.4). `active` (1 = run, GDN, CUDA host path) and `parked` (nonzero = skip, KDA, AMD host path)
   are both live, on disjoint host paths, and the KDA one has the better default. A descriptor that
   fills "the mask" has to know which family it is filling for.
2. **§5.2's class-C conversion of `FlashPrefill` is already done, by a different mechanism than the
   plan describes.** The plan says `i4=q_pos0` "must become a per-span base, or the packet must be
   launched per span". The second is what shipped, and the emitter has additionally made `i[4]`
   *dead*: it writes 0 unconditionally, and the span table is the sole carrier of position on the
   packed path. The plan's framing suggests the immediate is still meaningful; it is not.
3. **§9's Recurrent row asks for "the one-prefill-span limit enforced by admission"**, but on this
   branch the KDA family already has per-span arms, so its limit is unbounded and the interesting
   enforcement is not a *cap* but a *default-deny* for families with no per-span arm at all. The
   plan does not distinguish "D-class, so one span" from "D-class with a per-span arm, so
   unlimited" from "no arm at all, so refuse", and all three exist.
4. **§6.2's `RowGather` is a hard prerequisite for the softcap half of the windowed axis**, though
   Phase 3 is described as "independent of the others". The window bound is independent; "SoftCap
   stage in the terminal segment" is not, because there is no terminal segment.
5. **§3's operator table lists `QwenGdnPrefill` as the D-class example but not the seven
   decode-shaped GDN ops**, which need a different refusal for a different reason (wrong row axis,
   not missing state axis). The audit needs both categories.
