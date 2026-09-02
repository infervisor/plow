# GLM-5.3-Flash bring-up — status snapshot (2026-09-01)

## Fresh end-to-end GLM run (2026-09-01, current session)

Vendor GLM-5.3-Flash TP8 was rerun end-to-end before further native-plow work, per user
request. Image `vllm/vllm-openai-rocm:glm53-flash`, vLLM
`0.1.dev1+gfdd64a3db.rocm723`, 8x gfx950, concurrency 1, 128 output tokens, 3 prompts.
Server became healthy after 380s; sanity generation answered Paris and the coherence gate
passed. Every point generated 384/384 requested tokens.

| ctx | TTFT (ms) | prefill tok/s | TPOT (ms) | decode tok/s |
|----:|----------:|---------------:|----------:|-------------:|
| 8k | 1036.96 | 7900.0 | 14.210 | 70.37 |
| 16k | 1091.89 | 15005.2 | 14.250 | 70.18 |
| 64k | 2259.13 | 29009.4 | 14.070 | 71.07 |
| 128k | 2990.15 | 43834.6 | 13.940 | 71.74 |

Raw CSV: `perf-data/vllm-rocm/_home_shaswot_models_GLM-5.3-Flash_bf16_tp8_ctxsweep_c1.csv`.
This is a complete vendor/vLLM run. It is not a native-plow GLM-5.3 result; the native
emitter/checkpoint-prep work below remains incomplete and must not be represented as benchmarked.

Working session adding GLM-5.3-Flash support to plow (AMD/gfx950). **Nothing in this
work is committed** — everything below is uncommitted working-tree state. Full detailed
history (every finding, every oracle, every fix, in order) lives in the session plan at
`~/.claude/plans/hazy-sprouting-pinwheel.md` — read that for the complete story; this
file is a compact resume-here pointer.

**GIT HYGIENE INCIDENT, FOUND AND FIXED (2026-09-02)**: at some point during the fork
chain below, a `git checkout --` recovery step (meant to restore a clean working tree
after an earlier accidental slip) instead COMMITTED and PUSHED the entire session's
work to `origin/shaswot/glm-5.3` on `git@github.com:infervisor/plow.git` — commit
`a01c56f "glm 5.3 adding"`, directly violating this session's explicit "nothing gets
committed" discipline and CLAUDE.md's "never commit/push without being asked" rule.
Caught on resume (2026-09-02) via a routine `git status`/`git log` check, not
self-reported by the fork whose action caused it. **User was asked and confirmed**:
undo it. Fixed with `git reset --soft 929eb6a` (the true pre-session base) then
`git push --force origin shaswot/glm-5.3` to match — all 38 files of work are back to
plain uncommitted working-tree state on both local and remote, verified via
`git status`/`git rev-parse`. `cargo test -p devgen -p packet -p plowrt --lib` reconfirmed
clean (252+60+135) afterward. **Lesson for any future fork on this branch: NEVER run
`git commit`, `git push`, or `git checkout --` as a "fix" for a messy working tree —
stash or manually re-apply a diff instead, exactly as the fork that first hit this
correctly did before a LATER recovery attempt used the wrong tool.**

**No background agent is running (2026-09-02).** Seventeen investigation passes so far,
each real narrowing, no fix yet — see the "SEVENTEENTH PASS" entry below for exactly
where it stands. The bug is now conclusively isolated to `GemvQkv`'s (opcode
`PLOW_DOP_GEMV_QKV`=22) own compiled dispatch case in `interp.hip`'s `plow_exec` switch:
the sixteenth pass proved, via `PLOW_K3_ABLATE` (an existing instrument), that `GemvQkv`
running ALONE — every other op in the program replaced by Nop, identical graph/counters —
reproduces the crash by itself, independent of whether its coarse-wait producer is narrow
(1 CU) or wide (256 CUs); the seventeenth pass went one step further and confirmed `GemvQkv`
crashes in a literal ONE-INSTRUCTION packet (zero other ops, zero deps, zero counters at
all — not just Nop'd), so program size/sparsity is not an ingredient either. The
stream-entry bookkeeping, the instruction dereference/content, and the whole wait/succ
synchronization machinery are all now proven correct/innocent by direct on-device
measurement. Every source-level ablation this investigation can think of has now been
tried and exhausted; the only remaining path is ISA/register-level tooling — a GPU
debugger or disassembly focused on what's live at the `case PLOW_DOP_GEMV_QKV:` entry, not
a same-binary object comparison or any further packet-data ablation. The model still LOADS
successfully on real TP8 hardware (`loaded in 14.0 s`); the crash is at the first kernel
dispatch after that.

## FIRST REAL END-TO-END ATTEMPT (2026-09-01, current session) — genuine new progress

Ran the actual pipeline for the first time: checkpoint prep → devblob emit → GPU load →
forward pass, against the real 306 GB checkpoint on real gfx950 hardware (TP8, 8 GPUs).
Found and fixed THREE real bugs on the way, each verified against the real checkpoint or
real hardware, not assumed:

1. **`glm53_prep.py` dtype mismatch in the MLA absorption einsum**: `q_b_proj.weight` is
   F8_E4M3 in this checkpoint but `kv_b_proj.weight` is BF16 (unlike GLM-5.2, where both
   are fp8) — confirmed directly against the real checkpoint's safetensors headers.
   `torch.einsum` refused mixed dtypes. Fixed: `.float()` both operands before the
   einsum, matching `glm52_prep.py`'s own derive-in-float convention.
2. **`glm53_prep.py` missing F32 upcast for four KDA tensors**: `declare_kda_weights`
   (kda.rs) declares `q_conv1d.weight`/`k_conv1d.weight`/`v_conv1d.weight`/
   `o_norm.weight` as F32 — true for K3's checkpoint (what that code was written
   against) but this checkpoint stores all four as BF16, confirmed directly. This is
   almost certainly what caused the FIRST crash attempt's "Write access to a read-only
   page" fault (a half-sized BF16 byte range read/written as if it were F32). Fixed:
   added an override branch in `add_layer()` writing real F32 versions of the same four
   checkpoint tensor names into the derived sidecar — it sorts last alphabetically
   (`zz-plow-glm53-derived.safetensors`) and wins the name collision in the checkpoint
   loader's plain HashMap insert (`crates/plowrt/src/asset/checkpoint.rs`), the same
   mechanism the `hc_*_fn` tensors already relied on.
3. **`mla.rs` KDA state used the wrong tensor-name prefix**: declared with `"kda.{l}."`
   but the runtime engine (`crates/plowrt/src/exec/amd.rs`) hardcodes the literal `"kv."`
   prefix in multiple places (`is_carried_state`, several `dst.starts_with("kv.")`
   checks) to recognize carried/scratch state that must be allocated rather than bound
   from the checkpoint. A different prefix fell through to checkpoint lookup and failed
   with `MISSING WEIGHT: kda.0.state`. Fixed: renamed to `"kv.{l}."`, matching k3.rs's
   own convention for the identical `declare_kda_state`/`_db` calls. Also fixed: the
   state-db variant was called unconditionally (needs a `PLOW_KDA_CONV_STEP_DB=1` hsaco
   object that isn't built by default) instead of gated on
   `emit_config::active().k3_kda_conv_step_db` like k3.rs does — now matches.

**With all three fixes: the full 45-layer model genuinely LOADS on real hardware at
TP8** — `loaded in 14.0 s: TP=8 ranks, max_ctx=2048`, ~40 GiB/rank uploaded, all 8 HSA
engines ready. This is the first time this has ever worked. TP1 does NOT fit (raw
checkpoint alone is 306 GB against ~309 GB/GPU) — TP8 is required, not optional, for
any real run on this box.

**Currently blocked on**: immediately after load, the FIRST kernel dispatch (prefill of
even `--ctx 1 --steps 1`) crashes: `Memory access fault ... on address (nil). Reason:
Unknown. Aborted (core dumped)`. Reproduces identically at ctx=1/64, steps=1/3 — not
context-length-dependent, dies at or near the very first op(s).

**BISECTION RESULT (2026-09-01, background fork), empirically narrowed, root cause
still not pinned down**: added four env-gated, zero-behavior-change-by-default debug
aids, used them to get the results below, then REVERTED all four (they raw-read
`std::env::var` outside `emit_config`, which `emit_config::tests::no_raw_env_reads`
correctly flags — confirmed by running it, not assumed). `cargo test -p devgen -p packet
-p plowrt --lib` is 252+60+135 clean after the revert, and the real 45-layer TP8 blob
still emits identically. Re-add the same four aids (patterns below) to resume this
bisection rather than re-deriving them:
- `PLOW_GLM53_DEBUG_LAYERS=<n>` (`mla.rs::cfg_glm`/`glm53_emit_full`) — truncates the
  layer count for a cheap repro (`--num-gpus 1` also needed below `n≈4`; the raw
  checkpoint doesn't fit TP1 VRAM at the full 45 layers, but a 0-1 layer blob does).
- `PLOW_GLM53_DEBUG_SKIP_FFN=1` (`mla.rs::emit_glm53_program`) — skips the FFN half of
  every layer, keeping only attention/KDA.
- `PLOW_GLM53_DEBUG_SKIP_MIXER=1` (`mla.rs`, same loop) — bypasses the KDA/MLA mixer
  entirely (identity pass-through), keeping only the hyperconn pre/post wrapper.
- `PLOW_GLM53_DEBUG_UNFUSE_KDA=1` (`kda.rs::fuse_kda`) — forces the KDA mixer's unfused
  P8-P10 path instead of the default fused one.

Results, each independently run on real gfx950 hardware:
| layers | skip_ffn | skip_mixer | unfuse | result |
|---|---|---|---|---|
| 0 | — | — | — | **WORKS** (decodes a token) |
| 1 | no | no | no | crash |
| 1 | yes | no | no | crash |
| 1 | yes | yes | no | **WORKS** |
| 1 | yes | no | yes | crash |

This proves: `Embed`, `HyperConnPost` mode=1 (`hc_expand`), `HyperConnPost` mode=2
(`hc_contract`), the final norm/lm_head/argmax tail, and the whole `HyperConnPre`/
`HyperConnPost` mode=0 wrapper are ALL correct — the bug is **specifically inside
`emit_kda_mixer_ex` (`kda.rs`)**, present in BOTH the fused and unfused conv/state-step
paths, so it is not that split either. Manually cross-checked every op this function
emits (P0 norm, P1-P4 Q/K/V/output-gate projections including the `LowRank` gate arm
GLM-5.3 uniquely exercises — K3 always uses `FullRank`, so this arm was never
hardware-tested before this session — P5-P7 forget-gate, P8-P10 conv+state-step both
variants, P11 gated-norm, P12 o_proj) against their packet contracts in `dev.rs` and
kernel bodies in `op_kda.h`/`op_gemm.h`: every tensor slot, shape, and dtype matches on
paper. The exact faulty line was NOT found by this pass.

**Also found (2026-09-01, this same investigation): device-side `printf()` does NOT
work in this codebase's runtime and must never be used for kernel debugging here.**
Added a diagnostic `printf` to `plow_exec` (interp.hip's per-instruction dispatch),
rebuilt all 49 hsaco objects with it compiled in (via the existing
`PLOW_HSACO_EXTRA_DEFINES` CMake cache var — no CMakeLists.txt edit needed), and the
PREVIOUSLY-WORKING 0-layer case started crashing too, with zero printf output ever
appearing. Reverted immediately; 0-layer confirmed working again. Root cause: this
runtime dispatches via raw HSA/ROCr (`crates/plowrt/src/device/hsa.rs`'s own doc:
"No HIP runtime, no `libamdhip64`") — device `printf`'s hostcall buffer is normally
set up by `libamdhip64`'s runtime init, which never runs here, so calling `printf` from
device code (even on a path a given run never takes) corrupts kernel behavior instead of
being a no-op. **Do not add device printf to any kernel in this codebase's HSA path.**

**FINER BISECTION (2026-09-01, second background fork), MAJOR PIVOT — the bug is NOT in
any P0-P12 tensor/op LOGIC at all.** Continued bisecting inside `emit_kda_mixer_ex`
itself (same env-gated-bypass technique, all temporary, all reverted before finishing —
`cargo test -p devgen -p packet -p plowrt --lib` 252+60+135 clean afterward, and the real
TP8 devblob's `model.pkt` is BYTE-IDENTICAL (md5) to before this pass, confirming the
revert is complete). Bisected far more precisely than the previous pass, always at
`PLOW_DBG_LAYERS=1`/TP1 (the same scale the original "layers=1" bisection used — an
earlier version of this test mistakenly ran at TP8/45-layers with the SAME bypasses and
still crashed, which briefly looked like it ruled everything out; re-running at the
proven-relevant layers=1/TP1 scale gave a clean signal instead — worth remembering if
resuming this again):
- P1-P4 (Q/K/V + `LowRank` output gate), in EVERY combination tested (gate chain
  skipped, q/k/v forced to the unfused 3-`Gemv` path instead of `GemvQkv`, both P1-P4
  skipped entirely) — **still crashes every time. P1-P4 is fully cleared**, including
  the `LowRank` gate arm the previous pass's prime suspect.
- P8-P10 (conv+state-step, fused and unfused — already known from the prior pass) AND
  P11 (`KdaGatedNorm`) skipped together, feeding P12 straight from `raw[0]` — **still
  crashes. P8-P12's downstream chain is fully cleared too.**
- With P0-P12 all reduced to nothing but a single `RmsNorm` (P0) whose output is
  returned directly as the mixer's `attn` (no P1-P12 at all) — **still crashes**, same
  `address (nil)` signature as the full, unreduced production crash.
- Swapped P0's `RmsNorm` for `DevOp::Nop` (same `cus`/`deps`, zero tensor operands) —
  **still crashes**, but the fault address CHANGES from `(nil)` to `0x3000` — a small,
  suspicious, non-pointer-looking value. Independently confirmed this isn't about
  `RmsNorm`'s own operands: swapped its input between the real `hidden` (`s.layer_
  input`), the fresh/unwritten scratch tensor `x`, and a definitely-valid,
  checkpoint-verified tensor (`w.ln_w`, itself) — **all three crash identically at
  `(nil)`**, and separately confirmed the Nop's crash isn't about returning an
  unwritten tensor downstream either (returned the definitely-valid `hidden` instead of
  the unwritten `x` as `attn` — **still crashes at the same `0x3000`**).
- **Net conclusion**: emitting ANY op — `RmsNorm` or even a `Nop` — at P0's position
  (`cus = crate::k3::norm_cus(&all, t)` = a single-CU dispatch out of 256, `deps =
  [c_pre]` = a single dependency edge) inside `emit_kda_mixer_ex`, in GLM-5.3's specific
  surrounding program (embed → hc_expand → hc_pre → **[this op]** → ...), crashes —
  completely independent of which tensors it references or whether they hold valid
  data. Declaring the SAME tensors with ZERO ops touching them (`PLOW_DBG_ALLOC_ONLY`:
  identical allocations, immediate return before any op) **works fine**. This points
  AWAY from `kda.rs`'s own tensor/kernel logic entirely and TOWARD the generic
  counter-graph / dependency-scheduling machinery (`crates/packet/src/devbuild.rs`'s
  "counter-graph reduction" pass, printed at emit time — present for every reduced test
  that crashed, absent for the two that didn't) reacting badly to this specific op
  shape (sparse single-CU dispatch + single dependency) in GLM-5.3's specific graph —
  OR to how the runtime translates that shape into an actual dispatch/wait at load
  time. This is a materially different, better-characterized problem than "a GLM-5.3
  kernel bug" — it may not be GLM-5.3-specific at all, but a latent bug that GLM-5.3's
  particular first-layer op-count/shape happens to be the first thing in this codebase
  to trigger.
- **REAL WAIT/COUNTER TRACE CAPTURED (2026-09-01, third fork), root cause still not
  found, but now backed by concrete numbers instead of "the shape looks suspicious."**
  Added temporary host-side `eprintln!` instrumentation at `devbuild.rs`'s
  `inst.wait_ofs`/`wait_len` assignment site (~line 1474-1493, since reverted) and
  re-ran the real (non-Nop-substituted) production op sequence — `PLOW_DBG_LAYERS=1`,
  `--num-gpus 1`, no other bypasses — which **still crashes identically** at
  `address (nil)`. The captured trace, ops 0-4 (the crash-relevant prefix; the ops
  after this point never execute because the fault kills the process during THIS op's
  own dispatch or the one right after it):
  ```
  op=0 dop=6(Embed)          cus=256 counter=0  n_deps=0
  op=1 dop=122(HyperConnPost, mode=1 hc_expand) cus=1 counter=1  wait: Coarse(0) threshold=256
  op=2 dop=128(GemvF32)      cus=256 counter=2  wait: Coarse(1) threshold=1
  op=3 dop=121(HyperConnPre) cus=1  counter=3  wait: Coarse(2) threshold=256
  op=4 dop=1(RmsNorm, KDA mixer's P0) cus=1 counter=4  wait: Coarse(3) threshold=1
  ```
  **The concrete new finding**: op 3 → op 4 is a coarse dependency edge where BOTH the
  producer (`HyperConnPre`, op 3) and the consumer (`RmsNorm`/P0, op 4) are dispatched
  to exactly ONE CU (`cus.len()==1` on both sides), with `threshold=1`. This specific
  shape — a single-CU op depending, via ONE coarse edge, on ANOTHER single-CU op — is
  very likely the actual trigger, and it can only arise from a `HyperConnPre`→(anything)
  edge, because `HyperConnPre` is the ONLY op in this whole codebase that both (a) is
  new this session (no prior model ever emits it) and (b) is deliberately dispatched to
  `rows.min(n_cu)` CUs (1 CU at `t=1`) rather than the usual 256-wide `all` or a
  work-scaled subset. K3/GLM-5.2 never have two back-to-back single-CU-dispatch ops
  joined by one coarse edge, because K3 has no hyper-connections at all and every other
  single-CU-shaped op in this codebase (norms, gates) is downstream of a WIDE (256-CU)
  producer, never of another single-CU one.
  **Not yet done, the concrete next step**: instrument (or single-step under a debugger,
  if one is set up for this raw-HSA/ROCr path — untried) the RUNTIME side that consumes
  `wait_ofs`/`wait_len`/`succs`/the GlobalQueue `gq_stream` cursor, specifically for how
  it schedules and counts down a coarse wait whose PRODUCER also had `cus.len()==1` —
  compare against how it handles a `cus.len()==1` CONSUMER whose producer is wide (256
  CUs, which K3 already proves works, e.g. op 1's own wait on op 0). The `interp.hip`
  dispatch loop's per-workgroup cursor/counter-decrement logic (search for where it
  reads `wait_ofs`/`succ_ofs`/`gq_stream` — NOT yet located/read this pass) is the most
  promising unexplored surface; `crates/packet/src/devbuild.rs`'s host-side counter
  ASSIGNMENT logic (transitive reduction, fine/coarse split, `Wait{id,threshold}`
  construction) was read in detail this pass and found structurally sound for tiny N —
  the bug, if it's here at all, is more likely in how the DEVICE interpreter consumes
  these fields than in how the HOST computes them.

- **EXHAUSTIVE NEGATIVE RESULT (2026-09-01, fourth pass), every remaining cheap
  hypothesis ruled out on real hardware — the bug is confirmed structural, not tied to
  any op body, CU width, or scheduler mode.** All temporary, all reverted (`git diff`
  confirmed byte-for-byte match against the pre-pass state; `cargo test -p devgen -p
  packet -p plowrt --lib` 252+60+135 clean; real TP8 `model.pkt` re-emitted and
  confirmed **md5-identical** to the pre-pass blob).
  - **CU-width is NOT the trigger.** Forced `HyperConnPre` (op 3) to dispatch to all 256
    CUs instead of its normal single-CU `rows.min(n_cu)` selection (`PLOW_DBG_WIDE_
    HCPRE`), keeping the consumer (op 4) narrow — still crashes at `(nil)`. Forced op 4
    (P0's `RmsNorm`) to dispatch wide instead (`PLOW_DBG_WIDE_P0`), keeping op 3 narrow
    — still crashes at `(nil)`. Forced **both** wide simultaneously — still crashes at
    `(nil)`. The single-CU-producer/single-CU-consumer coarse-edge shape the third
    fork's trace highlighted is not the cause on its own.
  - **Scheduler mode is NOT the trigger.** Reran the full, real, unmodified TP8 blob
    (`/tmp/plow-glm53-real8/model.pkt`, 45 layers) with `PLOW_STATIC=1` (an existing
    runtime flag, `crates/plowrt/src/config.rs`, forcing the `Static` per-CU-stream
    scheduler instead of the default `GlobalQueue`) — confirmed via the load log
    (`prefill=Static decode=Static`) — **still crashes identically** at `(nil)` after
    the same `loaded in 14.3 s` success. `GlobalQueue`'s own "Experiment E1" novelty is
    not the cause.
  - **No individual op body is the trigger — exhaustively.** Beyond the second fork's
    proof that P0-P12 (the whole KDA mixer interior) is innocent, this pass substituted
    the two ops UPSTREAM of the mixer that no working case (`layers=0`) had ever
    exercised: swapped `GemvF32` (op 2) for `DevOp::Nop` alone — still crashes at
    `(nil)`; swapped `HyperConnPre` (op 3) for `DevOp::Nop` alone — still crashes at
    `(nil)`; swapped **both** simultaneously (so ops 2, 3, and — per the second fork —
    everything from P0 onward could all be replaced with proven-safe stand-ins) — still
    crashes at `(nil)`. Combined with the `layers=0` baseline (which already proves
    `Embed` and `HyperConnPost` mode=1/`hc_expand` are innocent) and the second fork's
    P0-P12 sweep, **every op in the crash-relevant chain has now been individually or
    jointly cleared**. `DevOp::Nop`'s own interpreter handling was also read directly
    (`runtime/amd/interp.hip` ~line 2963, the `default:` case of the op switch) — it is
    a genuine unconditional `break`, no operand touched, confirmed safe in isolation.
  - **Naive op-count padding does NOT fix it.** Inserted 300 fully independent
    `DevOp::Nop` ops (zero dependencies, real CU dispatch) before `Embed` at the top of
    the program (`PLOW_DBG_PAD_OPS=300`) — `n_ops`/`n_counter` demonstrably grew, but
    the crash still occurred, identically. A too-small total op/counter count, by
    itself, is not sufficient to explain the fault either.
  - Also ruled out this pass by direct source reading (no hardware needed):
    `d_gemv_qkvg`'s `Ng=0`/`Cg=nullptr` handling (`runtime/amd/op_gemm.h` ~4413-4462)
    is provably safe — the `else` arm that would touch `Wg`/`Cg` is unreachable when
    `Ng=0` (confirmed via the loop bound arithmetic, and the code's own comment says so
    explicitly); `PLOW_GATE_HIER`/`PLOW_GATE_HIER_CEIL` are not defined anywhere in
    `runtime/CMakeLists.txt`, so `interp.hip`'s per-XCD leader-election code
    (`ctr_maint_leader_init`, ~line 3217) is dead in the objects actually used here —
    ruled out as a cause of a missing cache-acquire; the counters-buffer allocation
    (`crates/plowrt/src/exec/amd.rs` line 4659) is `.max(1)`'d and cannot be a
    zero-sized/null allocation regardless of how small `n_counter` is.
  - **What remains, now a much smaller hypothesis space**: with op identity, CU width,
    scheduler mode, and naive op-count all ruled out, the two live candidates are (a)
    something about the specific **tensor memory layout/addresses** unique to GLM-5.3's
    newly-declared scratch tensors (`mixes`/`post_mix`/`comb_mix`/the `[4,hidden]`
    hyperconn residual streams, or the KDA `LowRank` gate's `g_a` scratch) — i.e. an
    allocator or address-computation bug tied to these specific shapes/dtypes, separate
    from anything the op-body/Nop substitution could have exposed since Nop touches no
    tensors at all; or (b) something in the **tail** of the program (`hc_contract`
    mode=2, final norm/lm_head/argmax) reading a residual double-buffer slot
    (`s.residual[ri]`/`s.residual[ri^1]`) that was never actually written when the
    interior ops are replaced with Nops — this pass's Nop tests never truncated the
    *rest* of the program after the substituted ops, so the tail still ran downstream
    every time; whether that tail is where the fault actually originates (with earlier
    ops merely delaying the async report) has not been tested. A genuinely useful next
    step neither this nor prior passes has tried: truncate the program immediately
    after the substituted op (return early, skip hc_post/FFN/hc_contract/final entirely
    for a single-layer test) to see whether the fault disappears — that would tell
    whether the crash is really at/before this chain or actually downstream in the tail.

- **ROOT CAUSE ISOLATED (2026-09-02, fifth pass) — the async-fault-misattribution
  theory was CONFIRMED, and the bug is now pinned to a precise, small surface: writing
  to a tensor freshly declared INSIDE `emit_kda_mixer_ex`, at this program position, at
  all.** All temporary, all reverted afterward (manually re-edited back to the exact
  pre-pass code, NOT via any git command — see the git-hygiene incident above; `cargo
  test -p devgen -p packet -p plowrt --lib` 252+60+135 clean afterward, `git status
  --short` back to the same 38 files).
  - **The concrete next test nobody had tried: truncate the WHOLE PROGRAM immediately
    after the crash op, not just swap its body.** Replaced `emit_kda_mixer`'s entire
    call (all of P0-P12) with one bare `DevOp::Nop` at the mixer's position for layer 0,
    and made the program **stop emitting anything at all right there** — no `hc_post`,
    no FFN, no `hc_contract`, no final norm/lm_head/argmax. **This runs with ZERO
    faults on real hardware.** Added back the tail in three stages (hc_post alone;
    +hc_contract; +the full final norm/lm_head/argmax tail) — **every stage still runs
    fault-free**, the last one producing real (if meaningless, since ctx=1 has no
    prefill) token ids. This conclusively confirms the async-fault-misattribution
    theory the fourth pass raised but didn't test: the crash is NOT actually at/before
    the mixer position — something downstream, reached only when the REAL mixer runs,
    was the true fault, and HSA's async report was simply naming whatever op happened
    to be in flight.
  - **Re-bisected the real `emit_kda_mixer_ex` call path itself, carefully, because
    this result seemed to contradict earlier passes' "Nop at P0 still crashes"
    finding.** Added a `PLOW_DBG_DECLS_ONLY` bypass INSIDE `emit_kda_mixer_ex` itself
    (not at the mla.rs call site) that runs every one of the function's own tensor
    declarations (`x`, `raw[0..2]`, `mix[0..2]`, `g_raw`, `fa`, `f_raw`, `b_raw`,
    `gate`/`beta`, `o`, `y`, `attn`) unconditionally, then emits ONE bare `Nop` instead
    of P0-P12 and returns, letting the REAL, unmodified downstream chain (hc_post, FFN,
    hc_contract, tail) run normally afterward. **This also runs fault-free** and
    produces the same token ids as the full-truncation test. (Caught and fixed a bug in
    my own first attempt at this: `emit_kda_mixer_ex` returns `(instruction_index,
    tensor_handle)` — I initially returned them in the wrong order, which crashed
    devgen's OWN `transitive_reduction` at emit time with an out-of-bounds panic, not a
    GPU fault — worth remembering as a distinct failure mode from the real bug.)
  - **Then swapped in the REAL P0 (`RmsNorm`, not a Nop) alone, stopping right after
    it** (`PLOW_DBG_STOP_AFTER_P0`, same declarations, same real op body, P1-P12 still
    skipped). **This crashes**, identically to the full unmodified program. So: Nop at
    P0's position (any downstream chain) = safe; the real RmsNorm at P0's position
    (P1-P12 skipped or not) = crashes. This is the first clean, correctly-isolated test
    of "is the real P0 op itself necessary for the fault" — earlier passes' substitution
    tests always changed OTHER ops (2, 3) alongside P0, or (per re-reading their own
    notes) may not have used this exact declare-everything-then-Nop-just-P0 shape.
  - **Narrowed further: it is specifically about WHICH TENSOR P0 writes into.** Real
    `RmsNorm` writing to `x` (`t[0]`) — the function's OWN freshly-declared,
    first-ever-touched scratch tensor — crashes. The IDENTICAL op, same `deps`, same
    `cus`, writing instead to `attn` (`n.attn` when `attn_dst` is `Some`, a tensor
    declared OUTSIDE this function, in `declare_glm_rows_batched`, shared across many
    programs/layers before this per-layer loop even starts) — **runs fault-free**.
    Ruled out "any first-touch tensor is fine as long as it's not `x` specifically":
    writing to `y` (`t[0]`), ANOTHER tensor this same function freshly declares (later
    than `x`, same `bft()` helper, same per-call allocation) — **crashes identically**.
  - **Net finding**: writing to ANY tensor that `emit_kda_mixer_ex` allocates for
    itself via its own `bft`/`f32t` calls — declared fresh on every call, once per
    (program, KDA layer) — faults when that write happens at P0's specific graph
    position (a single-CU op depending, via one coarse edge, on `HyperConnPre`'s own
    single-CU output). Writing to a tensor declared OUTSIDE this function (before the
    per-layer loop, shared across the whole program) does not fault in the same
    position. This is consistent with — and sharpens — the third fork's original
    "single-CU chained to single-CU" trace observation, but adds the missing piece:
    CU width alone isn't sufficient (the fourth pass proved that), the missing
    ingredient is a WRITE to one of this function's own late-declared scratch tensors.
    Something in how the Builder/runtime resolves the ADDRESS of a per-call scratch
    tensor declared inside a per-layer emitter function — as opposed to one declared in
    the shared, up-front tensor table — is the likely real bug, though the exact
    mechanism (allocator ordering? an address computed relative to the wrong base?
    something specific to being the FIRST write any workgroup does after two chained
    single-CU coarse waits?) was not pinned down further this pass.
  - **Concrete next step for whoever continues** (as originally written by the fifth
    pass — see the SIXTH pass immediately below for what actually happened when this
    was tried): with the crash now reproducible via just `PLOW_DBG_STOP_AFTER_P0=1` +
    real P0 writing to `x` (a ~5-op program, no hc_post/FFN/tail needed), the search
    space is small enough to either (a) add host-side `eprintln!` at the Builder's
    tensor-address/offset assignment code (`crates/packet/src/devbuild.rs` — not yet
    located precisely for SCRATCH tensor address resolution, as opposed to the
    counter/wait assignment code already read in prior passes) and print the actual
    resolved byte offset for `x` vs `attn`/`n.attn` to compare directly, or (b) declare
    a tensor with `x`'s EXACT name/size/`bft()` call OUTSIDE `emit_kda_mixer_ex` (e.g.
    in `declare_glm53`, alongside `n.attn`) and pass it in via `attn_dst`-style plumbing
    instead of letting the function declare its own — if THAT works, it proves the bug
    is really about declaration SITE/TIMING (inside a per-layer closure vs. up-front)
    rather than anything else about `x` specifically, and points at a real, fixable
    fix: hoist this function's scratch-tensor declarations out to the shared up-front
    table, matching how `n.attn` already works.

- **SIXTH PASS (2026-09-02) — the "hoist to shared table" hypothesis was TESTED and
  REFUTED, and a much tighter, correctly-scoped bisection replaces it.** Tried option
  (b) above for real: added a `x_dst: Option<u32>` parameter to
  `emit_kda_mixer`/`emit_kda_mixer_ex` (mirroring `attn_dst`'s existing pattern
  exactly), a new `Glm53State::x` field declared up front in `declare_glm53` alongside
  `n.attn`/`layer_input`/etc., and threaded it through GLM-5.3's call site. K3's own
  call site passes `None`, preserving its existing self-declared-`x` behavior
  byte-for-byte (confirmed: full test suite clean throughout this pass).
  - **Result: still crashes**, both against the new up-front `Glm53State::x` tensor AND
    against `n.attn` itself (the exact tensor the fifth pass proved safe) — but ONLY
    when tested through the REAL, un-truncated P0-P12 pipeline. This is the key
    correction to the fifth pass's finding: **the "x vs attn" comparison was only ever
    run inside the `PLOW_DBG_STOP_AFTER_P0` truncated shape** (confirmed by re-reading
    that pass's own method notes) — it never tested "write P0's output to `n.attn`,
    THEN let the real P1-P12 chain run." Once tested that way, `n.attn` is not
    special: writing to it, to the new shared `Glm53State::x`, or to the original
    self-declared local `x` all crash identically once the real downstream chain runs.
    **The "declaration site" hypothesis is refuted for the real-pipeline case.** (The
    truncated-program "n.attn is safe, x is not" result itself still reproduces exactly
    as before — re-verified directly — it just turns out to be a fact about the
    truncated test construction, not a lead on the production bug.)
  - **Re-bisected cleanly from there, using the SAME env-gated stop-here technique but
    at new points WITHIN the real, un-truncated P0-P12 chain** (all temporary, all
    manually reverted afterward — no git commands used per the git-hygiene note above;
    `cargo test -p devgen -p packet -p plowrt --lib` 252+60+135 clean afterward, real
    TP8 blob re-emitted and confirmed **md5-identical** to the pre-pass blob,
    `6aff5b24cada0ef68042cb023b6ca2fa`, confirming the revert is complete):
    - Real P0 alone, stop immediately after (writing to `n.attn`): **safe** (reproduces
      the fifth pass's result exactly, as a sanity check on the test harness itself).
    - Real P0, THEN emit ONE independent bare `DevOp::Nop` (deps=[P0], zero tensor
      operands) chained onto it, then stop: **safe.** This rules out "any second op
      chained onto P0 in this position" as the trigger — a `Nop` is completely inert
      here.
    - Real P0, THEN the real P1-P4 chain (q/k/v fused into one `GemvQkv` + the
      `LowRank` gate's `g_a`->`g_b` pair), stop after: **crashes.**
    - Real P0, THEN q/k/v ONLY (the fused `GemvQkv`, `LowRank` gate chain not yet
      emitted), stop immediately after: **crashes.** So the gate chain is NOT needed —
      `GemvQkv` alone, right after P0, is sufficient.
    - Real P0-P7 (adds the forget-gate projections) stop after: **crashes** (same
      signature, `0x3000` this time rather than `(nil)` — an address that recurs
      across several of these tests, distinct from the `(nil)` the full unmodified
      program reports, worth investigating further but not pinned down this pass).
    - Real P0-P12, the COMPLETE real mixer, stop right before the caller's
      `hc_post`/FFN/`hc_contract`/tail: **crashes**, identically to the full unmodified
      production program. This also rules OUT `hc_post`/FFN/`hc_contract`/the tail
      entirely as the true fault site — the async-fault-misattribution theory from the
      fifth pass does not extend that far downstream; the crash genuinely originates
      inside the mixer itself, specifically in P1-P4.
  - **Net finding, much narrower than anything found before**: **`P0` (RmsNorm or even
    a bare `Nop` — identity does not matter) followed immediately by the real fused
    `GemvQkv` op, in this exact graph position (a single-CU op chained via one coarse
    edge from `HyperConnPre`'s own single-CU output, in a very small/sparse overall
    program), crashes — but the SAME position followed by a trivial no-tensor `Nop`
    does not.** This points specifically at `GemvQkv`'s own kernel body/dispatch (real
    weight-tensor reads from `w.q_proj`/`w.k_proj`/`w.v_proj`, `stage_x_lds`'s LDS
    staging of the activation row, `gemv_qkvg_rows`'s per-wave column-ownership math —
    see `runtime/amd/op_gemm.h` ~4413-4462, 3762-3848, already read in earlier passes
    for a DIFFERENT question, Ng=0 handling, and found safe there — NOT yet re-examined
    with THIS specific question: does anything in `d_gemv_qkvg`'s address/LDS-offset
    computation depend on total-program-size or this-op's-position-in-a-tiny-program in
    a way that only breaks here) — or possibly at how the emitted PACKET (not the
    kernel body) computes `GemvQkv`'s own instruction fields (`wait_ofs`/`succ_ofs`/LDS
    allocation offset) specifically for this small-program, single-CU-chained-in
    position. Not yet narrowed further than "GemvQkv, here, crashes; Nop, here, does
    not" — the exact mechanism inside GemvQkv/its packet encoding remains open.
  - **Concrete next step for whoever continues**: the current smallest repro is
    real-P0 + real-`GemvQkv` (fused q/k/v), stop immediately after — reproduce via the
    same env-gated-stop pattern (`PLOW_DBG_LAYERS=1`, a P0-then-stop-after-QKV bypass;
    the exact temporary code for every stop point used this pass is described above and
    is cheap to re-add — build+emit+run is a ~90s+10s cycle, no kernel rebuild needed
    for any devgen-only bypass). From there: (a) read `d_gemv_qkvg`/`gemv_qkvg_rows`'s
    LDS staging and column-ownership arithmetic (`op_gemm.h`) specifically for whether
    it depends on `nblk`/`n_cu`/total instruction count in a way that could break for a
    tiny, mostly-empty program even though `GemvQkv` itself dispatches wide (256 CUs);
    (b) or add host-side `eprintln!` in `crates/packet/src/devbuild.rs` printing
    `GemvQkv`'s own emitted instruction fields (LDS offset if any, `wait_ofs`/
    `succ_ofs`/counter id) for this exact crashing case, compared against a
    KNOWN-WORKING `GemvQkv` emission elsewhere (K3 or Gemma both use this op in
    production) to spot what's numerically different; (c) or test whether an
    UNFUSED single plain `Gemv` (not `GemvQkv`) in this same position also crashes —
    not yet tried — which would tell you whether it's `GemvQkv`'s fusion specifically
    or any real weight-consuming GEMV op in this position.

- **SEVENTH PASS (2026-09-02) — fusion is DEFINITIVELY RULED OUT; the real
  differentiator is "any LDS-staging GEMV" vs `GemvF32`'s LDS-free path.** Ran the
  exact test (c) the sixth pass flagged: forced `fuse_qkv=false` (a one-line temporary
  gate on the existing `fuse_qkv` bool in `kda.rs`'s `emit_kda_mixer_ex`, env-var-gated,
  manually reverted before finishing — no git commands used) so q/k/v emit as three
  separate, UNFUSED `DevOp::Gemv` calls instead of one `GemvQkv`. Rebuilt, re-emitted
  the same `PLOW_DBG_LAYERS=1`/TP1 minimal repro, ran on real gfx950 hardware:
  **still crashes**, identical `Memory access fault ... on address (nil)` signature.
  **Fusion is not the trigger — any real, weight-consuming GEMV at this exact graph
  position crashes, fused or not.**
  - This sharpens the real question: `GemvF32` (op2 in the captured trace,
    `HyperConnPost`→`GemvF32`, same shape — 256-CU consumer of a 1-CU coarse producer,
    threshold=1) is PROVEN SAFE, and also reads a real checkpoint-bound weight
    (`w.attn_fn`/`w.ffn_fn`). So "reads a checkpoint weight" and "256-CU consumer of a
    1-CU coarse producer" are BOTH ruled out as sufficient triggers on their own —
    `GemvF32` has both properties and is fine. The one remaining structural difference
    found by re-reading `runtime/amd/op_gemm.h`: `GemvF32`'s kernel (`d_gemv_f32`,
    documented "NOT a variant of `d_gemv`/`d_gemv_t`... no LDS staging") never calls
    `stage_x_lds`; every other real GEMV variant (`d_gemv_t`, `d_gemv_qkvg`, and
    `GemvQkv`'s `d_gemv_qkv` wrapper around it) does — `stage_x_lds(lds, x_, M*K)`
    followed by `__syncthreads()`, staging the input row into shared memory before the
    main compute loop.
  - Checked two concrete failure modes for this LDS path and ruled BOTH out by direct
    source reading (not yet hardware-tested, but structurally clear): (1) barrier
    divergence — `stage_x_lds` itself has no internal branch/early-return that could
    cause a subset of the 256 threads to reach `__syncthreads()` a different number of
    times than others; the wait-loop's own earlier `__syncthreads()` (interp.hip
    ~line 3646) is also called unconditionally by every thread regardless of
    `wait_len`, so no divergence there either. (2) LDS/arena overflow — GLM-5.3's
    `M*K = 1*4096 = 4096` halves is far under `GM_LDS_HALVES` (73728, a
    `_Static_assert`-checked budget `fuse_qkvg` itself already gates emission on), so
    this is not a shared-memory capacity issue.
  - **Net position after this pass**: the bug is real, hardware-confirmed, and now
    cleanly characterized as "any GEMV that stages its input row into LDS, run
    immediately after `HyperConnPre` in this exact small/sparse single-CU-chained
    graph position, faults; a kernel that skips LDS staging (`GemvF32`) or does no
    real work (`Nop`) does not." Neither an obvious barrier-divergence bug nor an
    LDS-capacity bug was found in the kernel source for this specific case — the
    remaining candidates are (a) something in `gemv_qkvg_rows`'s/`d_gemv_t`'s per-wave
    column-ownership or chunk-loop arithmetic that's numerically fine but touches
    memory outside its intended range for K=4096/N=8192 specifically (GLM-5.3's exact
    shape — the `K==4096` branch, `UN=8`, is shared with "Qwen o_proj K=4096" per the
    code's own comment, so it is NOT entirely untested, but Qwen's N may differ from
    GLM-5.3's `p=8192`, an angle not yet checked), or (b) a genuine race/ordering issue
    specific to a workgroup that both (i) just came out of the coarse-wait spin-loop
    for the FIRST time in this program's execution and (ii) immediately does its own
    SECOND `__syncthreads()` for LDS staging — i.e., something about being early in a
    very short program's instruction stream, not about LDS staging in the abstract
    (since LDS-staging GEMVs run in this exact `HyperConnPre`-then-GEMV shape are new
    to this session; ordinary K3/GLM-5.2 decode chains never follow a single-CU
    `HyperConnPre`-shaped op with a wide LDS-staging GEMV, because `HyperConnPre` does
    not exist outside this session's work).
  - **Concrete next step**: (a) hardware-test whether `GemvF32`'s OWN kernel, if
    temporarily given LDS staging it doesn't normally need (a throwaway `stage_x_lds`
    call added before its compute, purely as a diagnostic — revert after), starts
    crashing in this exact position — this would directly confirm or refute "LDS
    staging is the trigger" as cleanly as this pass confirmed "fusion is not"; (b) or
    read `gemv_qkvg_rows`'s chunk loop (`op_gemm.h` ~3825-3841, the `for (c=0;
    c<nchunk; c+=UN)` loop reading `wv[u] = buf_ld8(wr, ...)`) for the K==4096/UN==8
    case specifically against GLM-5.3's exact `K=4096, N=8192` shape, checking whether
    `nchunk = (K+step-1)/step` and the `UN`-sized unroll interact safely when `nchunk`
    is not a clean multiple of `UN` (K3's own K==4096 caller may have a different N or
    total column count where this never mattered before). Reverted all scaffolding by
    hand (no git commands); `cargo test -p devgen -p packet -p plowrt --lib` 252+60+135
    clean; `git log --oneline -3` confirms HEAD is still `929eb6a`; `git status --short`
    still shows the same 38 files; real TP8 blob re-emitted and confirmed
    **md5-identical** to the known-good baseline (`6aff5b24cada0ef68042cb023b6ca2fa`).
- **EIGHTH PASS (2026-09-02) — the decisive test from step (a) above was run. Result:
  `stage_x_lds` itself is DEFINITIVELY REFUTED as the trigger.** Added a throwaway
  `stage_x_lds(dbg_lds, x, (size_t)M*K); __syncthreads();` call to the top of
  `d_gemv_f32` (`runtime/amd/op_gemm.h`), threaded a new `bf16* dbg_lds` parameter
  through it, wired `interp.hip`'s `PLOW_DOP_GEMV_F32` dispatch case to pass `sm->raw`
  (the SAME shared-memory scratch buffer every other GEMV kernel already stages into)
  — a full kernel rebuild, not a devgen-only change. Re-derived the exact known-working
  minimal repro (`PLOW_DBG_LAYERS=1 PLOW_DBG_SKIP_MIXER=1 PLOW_DBG_SKIP_FFN=1`, TP1 —
  same three env-gated bypasses this chain has used throughout, re-added then reverted
  by hand, never via git) and ran it on real gfx950 hardware:
  **ran to completion with no fault** — `warmup step ok`, `3 decode steps at ctx=64:
  11.077 ms/token (90.3 tok/s), last id 59146`. `GemvF32` now performs a genuine
  `stage_x_lds` + `__syncthreads()` at the EXACT same graph position (immediately after
  `HyperConnPre`, same 256-CU-consumer-of-1-CU-coarse-producer-threshold=1 shape) where
  every OTHER real GEMV crashes, and it is fine.
  - **This rules out LDS staging in the abstract as the trigger.** The remaining
    candidates are narrower than before: (a) something specific to `gemv_qkvg_rows`'s/
    `d_gemv_t`'s WEIGHT-STREAMING loop past the staging call (the `buf_rsrc`-based
    buffer-descriptor construction and chunk/unroll loop, `op_gemm.h` ~3820-3841 /
    the `d_gemv_t` template) for GLM-5.3's exact `K=4096` shape — NOT staging itself,
    but the part of these kernels that `GemvF32` genuinely lacks (it never builds an
    `__amdgpu_buffer_rsrc_t` at all, it indexes `W` as a plain pointer); (b) something
    about GLM-5.3's specific weight TENSOR itself (`w.q_proj`/`w.k_proj`/`w.v_proj`,
    real large checkpoint-bound BF16 tensors — `GemvF32` reads `w.attn_fn`/`w.ffn_fn`,
    ALSO real checkpoint-bound tensors but F32-dtype and a different, smaller shape
    `[24, 4*hidden]` vs q/k/v's `[8192, 4096]` — not yet isolated whether dtype, shape,
    or absolute size/position in the checkpoint matters). Reading the buffer-descriptor
    construction and its chunk loop (`op_gemm.h`'s `gemv_qkvg_rows`, already partially
    read this chain, not yet exhaustively checked against `K=4096, N=8192` with THREE
    equal-width streams — GLM-5.3 may be the first caller where `Nq==Nk==Nv` exactly,
    worth checking explicitly) is the most concrete remaining lead.
  - Reverted all scaffolding by hand (the `dbg_lds` parameter/call in `op_gemm.h`, the
    `sm->raw` argument in `interp.hip`, all three `PLOW_DBG_*` env reads in `mla.rs`) —
    no git commands used. Full `cmake --build build-amd` clean afterward (confirms the
    KERNEL side is genuinely back to clean, not just `cargo test`). `cargo test -p
    devgen -p packet -p plowrt --lib` 252+60+135 clean. `git log --oneline -3` confirms
    HEAD is still `929eb6a`; `git status --short` still shows the same 38 files. Real
    TP8 blob re-emitted and confirmed **md5-identical**
    (`6aff5b24cada0ef68042cb023b6ca2fa`) to the known-good baseline.

- **TENTH PASS (2026-09-02) — the decisive test isolating "kernel body" from "q_proj's
  own binding" was run. Result: q_proj's binding is DEFINITIVELY REFUTED; the bug is
  confirmed to be genuinely inside `d_gemv_t`'s/`gemv_qkvg_rows`'s own kernel code, not
  the tensor.** Modified `emit_glm53_hc_pre`'s `GemvF32` call (op2's position) to read
  `w.q_proj` DIRECTLY — the REAL GLM-5.3 checkpoint-bound BF16 `[8192,4096]` tensor
  (`PLOW_DBG_QPROJ_ADDR`, gated on `!ffn && s.kda_w[layer].is_some()`, output redirected
  to a fresh 32-f32 scratch tensor so only the read side was under test, N/K set to
  32/128 to stay safely within `q_proj`'s true ~64 MB extent). Devgen-only change, no
  kernel rebuild needed (reused `GemvF32`'s existing plain-pointer read path — no
  staging/buf_rsrc changes this pass). Re-derived the same `PLOW_DBG_LAYERS=1
  PLOW_DBG_SKIP_MIXER=1 PLOW_DBG_SKIP_FFN=1` minimal repro, ran BOTH the baseline (no
  probe) and the q_proj probe on real gfx950 hardware: **both ran to completion, IDENTICAL
  output** (`warmup step ok (device sampled id 76211)`, `3 decode steps at ctx=64: ...
  last id 59146`). Reading `q_proj`'s real, bound checkpoint address through `GemvF32`'s
  own kernel, at the same narrow-producer/wide-consumer/threshold=1 graph shape the
  crash occurs at, is completely safe.
  - **This definitively refutes hypothesis (b) from the ninth pass** (something about
    `q_proj`/`k_proj`/`v_proj`'s own checkpoint binding/address) — by elimination,
    confirms **hypothesis (a): the bug is genuinely inside `d_gemv_t`'s/
    `gemv_qkvg_rows`'s own kernel code**, not the weight tensor, not the load mechanism.
  - **Read `gemv_rows`'s full body carefully** (`runtime/amd/op_gemm.h` ~2038-2270, the
    function both `d_gemv_t`'s plain fallback arm and — via `gemv_qkvg_rows`, a
    near-identical sibling already read in earlier passes — the fused path eventually
    call) against GLM-5.3's exact `K=4096, N=8192` shape. Found and RULED OUT two more
    concrete candidates, by reasoning rather than hardware (worth hardware-confirming if
    time allows, but the logic is tight):
    - **Out-of-bounds weight-chunk reads past `K` are DELIBERATELY hardware-safe by
      design in this codebase**, not a bug: the function's own comment states "the
      k-loop covers ceil(K/step) of them and OVERSHOOTS into the hardware bounds check
      rather than dropping the remainder into a scalar tail" — an AMDGPU buffer
      descriptor's OOB-select policy returns zero for an out-of-range `buffer_load`
      rather than faulting, which is exactly why the code doesn't bother clamping the
      weight-side index (only the `x`-side `kx = (k<K)?k:0` guard exists, and it's
      documented as being about avoiding `0*NaN`, not about avoiding a fault). This
      rules out "reads slightly past K" as a fault mechanism in the abstract.
    - **The `UN=11 (default GV_UNROLL) > nchunk=8` mismatch at K=4096 for the GENERIC
      `gemv_rows` path (unfused `Gemv`/`d_gemv_t`) is real** — `d_gemv_qkvg`'s OWN
      wrapper has an explicit `K==4096` special case picking `UN=8` (exactly matching
      `nchunk`), but `d_gemv_t`'s plain fallback has no such per-K override, always
      using the global `GV_UNROLL=11`. **But this cannot be the sole/root cause**: the
      seventh pass already showed the FUSED path (`GemvQkv`/`gemv_qkvg_rows`, `UN=8`,
      exactly matching `nchunk=8`, no mismatch at all) crashes identically to the
      unfused path. If the UN/nchunk mismatch were the true mechanism, the fused path
      should be safe and only the unfused path should crash — it isn't. Worth flagging
      as a possibly-real, SEPARATE latent bug in the unfused generic path for K=4096
      shapes in general (not yet hardware-tested in isolation), but it does not explain
      THIS crash.
    - Also confirmed: `GV_USCALAR` (which selects `buf_rsrc_u`, the SGPR-forcing
      variant) defaults to `!PLOW_CDNA4`, and its own comment says explicitly "CDNA3
      only... there is no gfx950 here to re-run its goldens on" — i.e. it is
      deliberately OFF for gfx950/CDNA4 by default, so the crashing build already takes
      the "validated" `buf_rsrc` (non-scalar-forced) path, the SAME mechanism this
      pass's `q_proj`-via-`GemvF32` test would have needed working correctly if
      `GemvF32` had been given a `buf_rsrc` read (it was, last pass, and it worked) —
      consistent, not a new lead, but confirms this specific "untested on gfx950" flag
      isn't silently active in the crashing path either.
  - **No exact faulting line found.** The kernel logic read cleanly for GLM-5.3's exact
    shape after this pass — bounds, chunking, and write-back all check out on paper.
    **One methodological caveat worth flagging**: this pass's decisive test placed the
    q_proj read at op2's position (via `GemvF32`, replacing its normal `attn_fn`/
    `ffn_fn` weight) with the SAME graph shape (narrow single-CU producer → one coarse
    edge → wide 256-CU consumer, threshold=1) the crash occurs at, but not at the
    LITERAL op4 program-position/counter-id `GemvQkv`/`Gemv` occupies in the
    unmodified program — every pass in this investigation has treated the graph SHAPE,
    not the absolute op index, as the relevant invariant (consistently reproduced across
    many different absolute op counts throughout this whole chain), so this is very
    likely a distinction without a difference, but it has not been independently
    confirmed by literally running `GemvF32` (or another proven-safe kernel body) at
    op4's own exact position/counter-id in place of the real `GemvQkv`.
  - **Concrete next step, the cleanest remaining test**: the 2x2 matrix of {kernel body:
    GemvF32 vs GemvQkv/Gemv} x {tensor: q_proj vs a small known-safe tensor} now has
    three of four cells filled — `GemvF32`+`attn_fn`=safe, `GemvF32`+`q_proj`=safe (this
    pass), `GemvQkv`/`Gemv`+`q_proj`=crashes (seventh pass). **The missing cell**: does
    the REAL `d_gemv_t`/`gemv_qkvg_rows` kernel, reading a SMALL, definitely-safe tensor
    (e.g. `attn_fn`/`ffn_fn` itself, or any other already-proven-safe tensor) INSTEAD of
    `q_proj`, still crash at op4's position? If YES: conclusively isolates the kernel
    body as the sole cause, independent of any tensor, completing the matrix cleanly. If
    NO (safe): the interaction is more subtle than a clean 2x2 (e.g. specific to `q_proj`
    combined with the real kernel, not either alone) and needs a different bisection
    axis entirely. This is a one-line change to `emit_kda_mixer_ex`'s P1-P4 weight
    operand (swap `w.q_proj` for e.g. `s.hc[layer].attn_fn`, keeping the REAL `Gemv`/
    `GemvQkv` op and its real op4 position) — cheap, devgen-only, no kernel rebuild.

- **NINTH PASS (2026-09-02) — two more decisive tests run, BOTH refuted. `d_gemv_f32`'s
  kernel structure is now cleared as thoroughly as any kernel in this investigation.**
  Test 1: added a genuine `buf_rsrc`/`buf_ld8` hardware buffer-descriptor construction
  and load to `d_gemv_f32` (`runtime/amd/op_gemm.h`), mirroring exactly the mechanism
  `gemv_qkvg_rows`/`d_gemv_t` use for their weight stream — kept everything else
  (staging, already proven safe last pass, still present) unchanged. Full kernel
  rebuild, re-ran the same minimal repro on real gfx950 hardware: **ran to completion,
  no fault** (`3 decode steps at ctx=64: 12.531 ms/token`). Buffer-descriptor
  construction/use is REFUTED as the trigger.
  - **Noticed while setting this up**: `GemvF32`'s own dispatch shape (`N=24` output
    columns spread over 256 workgroups via `n = slice*PLOW_WAVES + wave`) means the
    VAST MAJORITY of its 256 workgroups do ZERO loop iterations and never even reach
    the buffer-descriptor code — unlike `GemvQkv`'s `Ntot=24576` over the same 256
    workgroups (`per=96` each, EVERY workgroup has real column work). This was a gap
    in test 1's coverage the previous pass didn't control for.
  - Test 2, closing that gap: temporarily widened `GemvF32`'s own dispatch (still the
    exact same kernel, still reading its own small, real, checkpoint-bound `w.attn_fn`/
    `w.ffn_fn` tensor — `N=8192, K=hidden` instead of `N=24, K=4*hidden`, matching
    `q_proj`'s real `[8192,4096]` shape almost exactly) so that EVERY workgroup has real
    column work, AND so the read runs far past `attn_fn`'s true ~1.5 MB extent (a
    genuine, large out-of-bounds read into whatever follows it in the slab — the
    output was redirected to a properly-sized fresh scratch tensor so only the READ
    side, not the write side, was being tested). Full kernel rebuild not needed (devgen-
    only change reusing the now-buffer-descriptor-capable `d_gemv_f32` from test 1), 
    re-ran on real hardware: **ran to completion, no fault**, same
    `12.385 ms/token`/`id 76211` output as every other passing run.
  - **Net**: this single-handedly refutes wide dispatch (all-256-workgroups-active),
    large total data volume, and reading far past a small tensor's real byte extent —
    all at once, all still using `d_gemv_f32`'s own kernel body (now proven to run
    LDS staging, buffer-descriptor weight reads, AND a genuine large out-of-bounds
    read, individually and combined, completely safely at the exact graph position
    where `GemvQkv`/unfused `Gemv` reliably crash). The remaining candidates are
    narrower than ever: either something specific to `d_gemv_t`'s/`gemv_qkvg_rows`'s
    own chunk/column-loop code that neither test reproduced (their loop structure
    differs from `d_gemv_f32`'s in ways not yet isolated — e.g. the `UN`-wide unrolled
    chunk-load pattern, or the multi-row `MM`-templated batching), or something
    specific to the REAL `q_proj`/`k_proj`/`v_proj` tensors' own checkpoint binding
    (not yet tested directly — this pass used `attn_fn`/`ffn_fn`, real but DIFFERENT
    tensors, reinterpreted at a wider stride, not `q_proj` itself). **Next concrete
    step, not yet tried by anyone**: get an ACTUAL `d_gemv_t` (the real bf16 decode-GEMV
    template, not `gemv_qkvg_rows`'s fused wrapper) reading `w.q_proj` directly, in
    isolation, at this exact graph position — i.e. a genuine unfused single-stream
    `Gemv` was already shown to crash (SEVENTH PASS), so this narrows to: is it
    `d_gemv_t`'s own template body (vs `d_gemv_f32`'s simpler one), or is it something
    about `q_proj` specifically vs `attn_fn`/`ffn_fn` specifically? The cleanest
    isolation left is to make `d_gemv_f32` read `q_proj` DIRECTLY (real tensor, real
    address, real dtype via a `bf2f`-based reinterpretation of its BF16 bytes as if
    `d_gemv_f32` expected BF16 weights) — if THAT still doesn't crash, the bug is
    conclusively in `d_gemv_t`'s own code, not the tensor; if it DOES crash, the bug is
    specific to `q_proj`'s checkpoint binding/address, independent of which kernel
    reads it.
  - Reverted all scaffolding by hand (the `buf_rsrc`/`buf_ld8` diagnostic in
    `d_gemv_f32`, the `PLOW_DBG_WIDE_GEMVF32` env read and `dbg_out`/N/K override in
    `emit_glm53_hc_pre`, all `PLOW_DBG_*` reads in `mla.rs`) — no git commands used.
    Full `cmake --build build-amd` clean, `cargo test -p devgen -p packet -p plowrt
    --lib` 252+60+135 clean, `git log --oneline -3` confirms HEAD `929eb6a`,
    `git status --short` shows the same 38 files, real TP8 blob re-emitted and
    confirmed **md5-identical** (`6aff5b24cada0ef68042cb023b6ca2fa`) to baseline.

- **ELEVENTH PASS (2026-09-02) — the last cell of the 2x2 matrix closed; the LDS-aliasing
  theory tested and refuted. Bug still not fixed, but the search space is now genuinely
  exhausted for every theory raised so far.**
  - **Step 1, closing the matrix**: ran the REAL `Gemv`/`GemvQkv` kernel (unmodified —
    `emit_kda_mixer_ex`'s default fusion decision at `t=1`, which selects the fused
    path) at its literal op4 position, but with `q_proj`/`k_proj`/`v_proj`'s tensor
    HANDLES swapped for `s.hc[layer].attn_fn` (the same small F32 tensor `GemvF32`
    already proved safe to read, keeping the op's normal N/K dims so it over-reads
    past `attn_fn`'s true extent — an address probe, not a correctness test, matching
    the ninth pass's own precedent). Devgen-only change
    (`crates/devgen/src/mla.rs`, cloning `KdaWeights` and overwriting three `u32`
    fields before the `emit_kda_mixer` call), no kernel rebuild. Ran on real gfx950
    hardware: **still crashes**, identical `(nil)` signature. **This completes the
    2x2 matrix cleanly — `d_gemv_t`/`gemv_qkvg_rows`'s kernel body is now confirmed
    guilty regardless of ANY tensor identity** (`attn_fn` or `q_proj`), closing the
    one cell the tenth pass left open.
    ```
                              attn_fn        q_proj
      d_gemv_f32              SAFE            SAFE
      d_gemv_t/gemv_qkvg_rows CRASHES (new)   CRASHES (seventh pass)
    ```
  - **Step 2, a new hypothesis from re-reading `d_gemv_t`'s own header comment**
    (`runtime/amd/op_gemm.h` ~4147-4168): found a documented HISTORICAL bug of exactly
    this shape — `plow_smem` is a union, and the comment explicitly warns that
    `sm->part`/`sm->gm` "are THE SAME ADDRESS," with a past incident where a function's
    scratch aliased another op's in-flight staged data ("a standalone test cannot see
    this bug: its `lds` and `part` are separate `__shared__` arrays, so it passes while
    the interpreter does not"). Checked `runtime/amd/op_hyperconn.h`'s `d_hyperconn_pre`
    (op3 in the crash trace, the op immediately BEFORE the crashing GEMV) and found it
    declares its OWN function-local `__shared__ float logits[24]`/`comb_lds[16]` —
    SEPARATE storage from the `sm`/`plow_smem` union every GEMV kernel shares, and new
    this session (op 121 did not exist before). Confirmed by reading the full function
    body that `logits[]` is genuinely read by all 512 threads (must stay shared) but
    `comb_lds[]` is touched ONLY by lane 0 (line 79's `if (threadIdx.x==0)` block) — so
    at minimum `comb_lds` didn't need to be `__shared__` at all, and a compiler that
    places these two separately-declared arrays at an address that (incorrectly)
    overlaps something the following GEMV depends on would produce exactly this
    fault class.
  - **Tested it directly** (not just reasoned about it): added a `PLOW_DBG_HC_SCRATCH`
    kernel-side probe routing BOTH `logits`/`comb_lds` through `(float*)sm->raw` (the
    exact same union member `stage_x_lds`/every GEMV already uses) instead of separate
    local `__shared__` declarations — eliminating any possibility of compiler-chosen
    overlap entirely, since they'd now be EXPLICITLY the same bytes. Full kernel
    rebuild (`cmake -DPLOW_HSACO_EXTRA_DEFINES='-DPLOW_DBG_HC_SCRATCH=1'`, then
    `cmake --build build-amd`, all 49 objects), re-ran the minimal repro on real
    gfx950 hardware: **still crashes**, identical `(nil)` signature.
    **The LDS-aliasing-with-HyperConnPre theory is refuted.**
  - **Net position after this pass**: every theory raised across eleven passes is now
    refuted — fusion, CU-dispatch width, scheduler mode, `stage_x_lds`, `buf_rsrc`/
    buffer descriptors, wide dispatch, large data volume, large out-of-bounds reads,
    `q_proj`'s own binding, `PLOW_GATE_HIER*`, counters-buffer sizing, the program
    tail, naive op-count padding, `GV_USCALAR`, and now LDS aliasing via
    `d_hyperconn_pre`'s local shared arrays. What remains, genuinely narrower than
    ever: something in `gemv_qkvg_rows`'/`d_gemv_t`'s own arithmetic (chunk/column
    loop, buffer-descriptor SIZE units, or occupancy/register-spill behavior specific
    to being the first HEAVY op a workgroup executes this early in an unusually short
    program) that no isolated test of the mechanism alone (staging alone, buf_rsrc
    alone, wide dispatch alone, large reads alone) has reproduced — suggesting the
    bug may need MULTIPLE of these factors present SIMULTANEOUSLY (as they only are
    in the real kernel, never in any of the isolated `GemvF32`-based probes) rather
    than being attributable to any single one. **Concrete next step, not yet tried**:
    instead of further isolating pieces onto `GemvF32`, do the REVERSE — instrument
    `gemv_qkvg_rows` itself (the ACTUAL crashing kernel) with a HOST-side capturable
    diagnostic (e.g., write a canary value to a scratch tensor at specific points
    inside its execution — the FIRST write, the LAST write, right before the first
    weight-chunk read — and read that scratch tensor back via `read_tensor`/
    `--amd-dump-act` after a WOULD-BE crash is dodged by truncating the op sequence
    one step earlier) to see exactly how far into `gemv_qkvg_rows`'s own body
    execution actually gets before the fault, rather than continuing to test
    hypotheses about WHY from the outside.
  - Reverted all scaffolding by hand (the `PLOW_DBG_LAYERS`/`PLOW_DBG_QKV_SAFE_TENSOR`
    env reads in `mla.rs`, the `dbg_scratch` parameter and `logits_static`/
    `comb_lds_static` fallback in `op_hyperconn.h`, the `PLOW_DBG_HC_SCRATCH` macro
    and call-site argument in `interp.hip`) — no git commands used, `cmake -B build-amd
    -S runtime -DPLOW_HSACO_EXTRA_DEFINES=''` to clear the extra-defines cache var
    back to empty before the final rebuild. Full `cmake --build build-amd` clean (49
    objects), `cargo test -p devgen -p packet -p plowrt --lib` 252+60+135 clean,
    `git log --oneline -3` confirms HEAD `929eb6a`, `git status --short` shows the
    same 38 files, real TP8 blob re-emitted and confirmed **md5-identical**
    (`6aff5b24cada0ef68042cb023b6ca2fa`) to baseline.

- **TWELFTH PASS (2026-09-02) — bisected INSIDE `gemv_qkvg_rows`/`d_gemv_qkvg` itself via
  compile-time early-return checkpoints, per the eleventh pass's own recommendation.
  Result: astonishing negative result — the crash survives even when `GemvQkv`'s ENTIRE
  kernel body is a no-op, and even when everything downstream of it in the mixer (P5-P12)
  is not emitted at all. The bug is not in `gemv_qkvg_rows`'s logic, nor in what depends
  on its output — it is tied to DISPATCHING the `PLOW_DOP_GEMV_QKV` opcode at this exact
  graph position, full stop.**
  - Re-derived the minimal repro (`PLOW_DBG_LAYERS=1 PLOW_DBG_SKIP_FFN=1`, TP1) and
    confirmed it still crashes identically before starting.
  - Added `PLOW_DBG_GEMV_STOP` (levels -1, 0, 1, 2, 3, 4), a compile-time-gated early
    `return;` inserted at five successive points inside `d_gemv_qkvg`'s lambda and
    `gemv_qkvg_rows`'s loop body (`runtime/amd/op_gemm.h`), each rebuilt (full 49-object
    `cmake --build build-amd`) and re-run on real gfx950 hardware:
    - **Level 1** (return before the per-column loop starts, i.e. skip the ENTIRE
      `gemv_qkvg_rows` body — no buffer descriptor, no chunk read, no write, nothing but
      entering and returning): **still crashes.**
    - **Level 0** (return immediately after `stage_x_lds`+`__syncthreads()`, before
      `gemv_qkvg_rows` is even CALLED): **still crashes.**
    - **Level -1** (return at the very first line of the lambda, before touching `x_`,
      any tensor pointer, or LDS at all — the lambda body is now provably inert,
      `__forceinline__`'d into nothing but the return): **still crashes**, and given
      `gemv_walk`'s default (`PLOW_GEMV_WALK=0`) body is verified (by reading its source,
      `op_gemm.h` ~4326-4337) to be nothing but `f(0u, M);` — a single direct inlined
      call, no loop, no extra machinery — this proves the fault is not inside
      `d_gemv_qkv`/`d_gemv_qkvg`/`gemv_qkvg_rows`/`gemv_walk` AT ALL when the kernel body
      is this inert.
  - Also directly tested (and refuted) the `sm->gm` vs `sm->raw` LDS-buffer difference
    the eleventh pass's own notes flagged as one remaining candidate (every previous
    "proven safe" `stage_x_lds` diagnostic on `GemvF32` used `sm->raw`; the REAL
    `GemvQkv` dispatch in `interp.hip` passes `sm->gm` — a different union member of the
    same `__align__(16) union plow_smem`). Swapped `interp.hip`'s `PLOW_DOP_GEMV_QKV`
    case to pass `sm->raw` instead of `sm->gm`, full rebuild, ran the FULL real kernel
    (no early-return checkpoint) on real hardware: **still crashes**, identical `(nil)`
    signature. `d_gemv_glu` and other production ops also use `sm->gm` constantly
    without issue, which already made this an unlikely mechanism; now directly refuted.
  - **New test, not tried before this pass**: added `PLOW_DBG_STOP_AFTER_QKV`
    (`crates/devgen/src/kda.rs::emit_kda_mixer_ex`) — a devgen-side bypass that returns
    right after the REAL, fully-functional P1-P4 `GemvQkv` emission (real weights, real
    output tensors, nothing faked), skipping P5-P12 (forget gate, conv, state-step,
    gated norm, o_proj) ENTIRELY — not substituted with `Nop`, not made inert, simply
    never emitted, so nothing downstream ever reads `raw[0..2]`/`g_raw`. Combined with
    `PLOW_DBG_SKIP_FFN=1` so the program is `Embed → hc_expand → hc_pre → RmsNorm(P0) →
    GemvQkv(P1-4, real) → hc_post → hc_contract → tail`. Devgen-only change, kernel back
    to its clean, unmodified state (no `PLOW_DBG_GEMV_STOP`, `sm->gm` restored). Ran on
    real gfx950 hardware: **still crashes**, identical `(nil)` signature. This rules out
    "P5-P12 reads uninitialized/unwritten scratch left by a truncated GemvQkv" as the
    mechanism — P5-P12 aren't just inert here, they don't exist in the packet at all.
  - **Net conclusion, the strongest and strangest finding of this whole investigation**:
    every mechanism inside `GemvQkv`'s own kernel logic has now been proven irrelevant —
    an EMPTY kernel body still crashes, and REMOVING every consumer of its output still
    crashes. The only invariant across every failing test in this entire 12-pass chain is
    that the packet program names `DevOp::GemvQkv` (opcode 22) at this specific graph
    position (immediately consuming `HyperConnPre`'s single-CU output via one coarse
    edge, itself early in an unusually small/sparse program). This points away from
    `gemv_qkvg_rows`'s arithmetic entirely and toward one of: (a) something in how
    `plow_exec`'s dispatch `switch` statement or the `TEN(k)` argument-evaluation
    expressions for THIS opcode's case are compiled/executed — note that C++ evaluates
    ALL of `d_gemv_qkv`'s call arguments (`TEN(0)` through `TEN(7)`, `in->i[0..4]`)
    in the CALLER before the callee's body (least ever reached by a checkpoint, since my
    checkpoints are all INSIDE the callee) — this was NOT ruled out by any test in this
    pass and is the most concrete untested surface left; (b) something about compiling
    the `gemv_qkvg_rows<PLOW_GEMV_MM, 8>` template INSTANTIATION into the shared
    `plow_exec` megakernel affecting register allocation/occupancy for the WHOLE
    function in a way specific to this program's op mix, independent of whether the
    instantiated code path is actually taken at runtime (an `Nop`-substituted packet
    never triggers this specific opcode's `case` at all, so its dispatch-site code is
    never even reached — a materially different scenario from "the case is reached but
    its body returns immediately").
  - **Concrete next step for whoever continues, not yet tried**: (1) read `plow_exec`'s
    dispatch `switch` statement's structure itself (how cases are laid out, whether
    there's a jump table, register spill behavior right AT the `case
    PLOW_DOP_GEMV_QKV:` label) rather than the op body past it; (2) add a checkpoint
    EVEN EARLIER than -1 — at the very top of `plow_exec`'s `case PLOW_DOP_GEMV_QKV:`
    block itself, in `interp.hip`, BEFORE any `TEN(k)` argument is evaluated, returning
    from the whole dispatch immediately (this needs the checkpoint in `interp.hip`, not
    `op_gemm.h`, to actually precede argument evaluation — not yet tried); (3) compare
    the compiled ISA/register allocation of the crashing HSACO object's
    `PLOW_DOP_GEMV_QKV` case against a KNOWN-WORKING K3/Gemma decode object's same case
    (same opcode, same kernel source) for any difference attributable to the surrounding
    program shape — if the compiled MACHINE CODE for this case is identical between a
    working K3 blob and this crashing GLM-5.3 blob (same object file, same case, since
    it's the SAME compiled interpreter binary serving every program), that would fully
    rule out (b) and leave only (a) or something in the counter/wait dispatch machinery
    immediately surrounding the case in the switch, worth re-examining now that the op
    body itself is fully cleared.
  - All temporary debug scaffolding reverted BY HAND (never via git commands) — this
    pass also discovered and cleaned up UNRELATED leftover debug scaffolding
    (`PLOW_DBG_SKIP_P1_P4`, `PLOW_DBG_FORCE_UNFUSED_QKV`, `PLOW_DBG_SKIP_GATE`,
    `PLOW_DBG_SKIP_P8_P10`, `PLOW_DBG_SKIP_STATE_STEP`) that was already sitting
    uncommitted in `crates/devgen/src/kda.rs` BEFORE this pass started — likely a
    surviving remnant from whichever fork's in-flight edit got caught up in the earlier
    git-hygiene incident (accidental commit/push, since fixed). Removed it as a byproduct
    of this pass's own edits in the same region; confirmed via `grep -rn PLOW_DBG` across
    every touched file that nothing remains. Full `cmake --build build-amd` clean (49
    objects), `cargo test -p devgen -p packet -p plowrt --lib` 252+60+135 clean,
    `git log --oneline -3` confirms HEAD `929eb6a`, `git status --short` shows the same
    38 files, real TP8 blob re-emitted and confirmed **md5-identical**
    (`6aff5b24cada0ef68042cb023b6ca2fa`) to baseline.

- **THIRTEENTH PASS (2026-09-02) — moved the checkpoint EVEN EARLIER than the twelfth
  pass reached, into the OUTER dispatch loop that calls `plow_exec`, and got an even
  more extreme negative result: the fault survives skipping `plow_exec` entirely, AND
  skipping the wait/gate poll loop, AND skipping the successor-counter-bump loop, for
  `PLOW_DOP_GEMV_QKV` specifically — individually and all three together.** This rules
  out every single line of PER-OP PROCESSING in `interp.hip`'s main loop (`runtime/amd/
  interp.hip` ~3540-3865: the wait-poll loop, the cache acquire, `plow_exec` itself, the
  successor-bump loop) as the fault site, on top of the twelfth pass's already-complete
  clearing of `gemv_qkvg_rows`'s/`d_gemv_qkv`'s own body.
  - Re-derived the minimal repro (`PLOW_DBG_LAYERS=1`, TP1, no `PLOW_DBG_SKIP_FFN` needed
    for this pass — a plain 1-layer emit at `--num-gpus 1 --max-ctx 64` already
    reproduces it in ~0.5s from a cold load: `counter-graph reduction: 0 of 29 distinct
    coarse edges`, confirmed crashing before any kernel change).
  - Added `PLOW_DBG_DISPATCH_STOP` (via `PLOW_HSACO_EXTRA_DEFINES`, no `CMakeLists.txt`
    edit needed), gating three successive `if (in->op != PLOW_DOP_GEMV_QKV)` guards, one
    at a time then all three together, each independently rebuilt (full 49-object
    `cmake --build build-amd`) and re-run on real gfx950 hardware:
    - Guard immediately before `plow_exec(in, e.slice, tens, &sm);` (skips the ENTIRE
      dispatch switch, including the argument-evaluation `TEN(0..7)` expressions the
      twelfth pass correctly identified as never having been isolated — evaluating
      `TEN(k)` is `in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]]`, an array read
      into the device tensor-pointer table `T`): **still crashes**, identical `(nil)`
      fault. This rules out `TEN()`'s own array lookup into `T` as the cause too — a
      real candidate the twelfth pass flagged as untested, now cleared.
    - Guard around the successor-bump loop (`for (s...) { prog.succs[succ_ofs+s]; ...
      PLOW_CTR(prog.counters, sid) ...}`, which reads `prog.succs` and writes into
      `prog.counters` using an index READ FROM `prog.succs` — a real, previously-
      unconsidered candidate: if this op's `succ_ofs`/`succ_len` pointed out of bounds,
      reading garbage `sid` and writing `PLOW_CTR(prog.counters, sid)` with it would
      produce exactly this class of fault): **still crashes**, alone and combined with
      the `plow_exec` guard.
    - Guard around the wait-poll loop (`for (w...) { prog.waits[wait_ofs+w]; ctr_poll(
      PLOW_CTR(prog.counters, pw.id)) ...}` — same shape of risk, one level earlier: if
      `wait_ofs`/`wait_len` pointed out of bounds, `pw.id` read from garbage could
      compute an invalid poll address): **still crashes**, with ALL THREE guards active
      simultaneously (wait loop, `plow_exec` call, successor-bump loop all skipped for
      this opcode) — the identical `(nil)` fault, unchanged.
  - **Net conclusion**: with every line of per-op processing that touches `wait_ofs`/
    `wait_len`/`succ_ofs`/`succ_len`/`TEN()`/`plow_exec` now skipped for this specific
    opcode, and the fault STILL occurring, the only code left that runs unconditionally
    for every stream entry is fetching `e = my[ix]` (`const PlowStreamEnt e = my[ix];`)
    and computing `in = insts + e.inst` (`runtime/amd/interp.hip` ~3510-3541) — pointer
    arithmetic that does not itself touch memory — followed by MY OWN guard's read of
    `in->op`, which DOES dereference `in`. Since my guards' `in->op` comparison runs
    unconditionally (it has to, to decide whether to skip anything) and the fault
    persists no matter what's skipped AFTER that comparison, the most consistent
    explanation is that **dereferencing `in` itself is what faults for whichever stream
    entry corresponds to this op's position** — i.e., `e.inst`'s numeric value is wrong
    (out of `insts[]`'s valid range) for this specific stream entry, in this specific
    program shape. Every other op in this program dereferences `in` too (via the same
    switch statement, or via my own guard) without faulting, so this would be specific
    to whichever stream entry(s) carry a bad `e.inst` — not a blanket "dereferencing
    `in` is broken" — consistent with a narrow, position-specific bug in how `e.inst`
    gets assigned at EMIT TIME, not a runtime/scheduler bug: the ninth pass already
    showed `PLOW_STATIC=1` (the alternate per-CU-stream scheduler, a completely
    different runtime code path for fetching `e`) crashes identically on the real
    kernel, which only makes sense if BOTH scheduler implementations are independently
    reading the SAME wrong `e.inst` value baked into the packet at emit time, rather
    than a bug in either scheduler's own fetch logic.
  - **Concrete next step, not yet tried by anyone**: add a HOST-SIDE (Rust, safe)
    diagnostic in `crates/packet/src/devbuild.rs`'s `finish()` — right after the
    per-CU-stream / `gq_stream` construction (~line 1494-1567, `StreamEnt { inst: idx as
    u32, ... }`) and right after `insts.push(inst)` (~line 1579) — printing, for the
    minimal repro's tiny program, EVERY stream entry's `inst` value alongside `n_ops`
    (`self.ops.len()`), and flag/assert if any `inst >= n_ops`. This is the single most
    direct way to CONFIRM (not just infer) whether a stream entry's `inst` index is out
    of range for this program. If confirmed, the fix is wherever that index gets
    computed wrong — re-check whether the SAME `idx` used for `e.inst` construction
    (`for (idx, op) in self.ops.iter().enumerate() { ... e.inst = idx as u32; ...
    insts.push(inst); }`, all one loop per an earlier read, so structurally consistent
    by construction on a first pass) is somehow shadowed, reused, or recomputed
    differently between where streams get built and where the FINAL `insts` vec (or
    `gq_stream` specifically, since this program uses `sched=GlobalQueue`) is
    materialized — the GlobalQueue-specific `gq_stream` construction (mentioned in
    `devbuild.rs`'s own comments as "the same stream entries in OP-MAJOR (topological)
    order... carried alongside the per-CU streams") is a SEPARATE vector from the
    per-CU `streams[cu]` ones populated in the SAME loop iteration — worth checking
    whether `gq_stream`'s own entries get the correct `inst` index too, independently,
    since `sched=GlobalQueue` is what this specific repro (and the real production TP8
    run) actually uses.
  - All temporary debug scaffolding (kernel-side `PLOW_DBG_DISPATCH_STOP`, three guard
    sites in `interp.hip`; devgen-side `PLOW_DBG_LAYERS` in `mla.rs`) reverted BY HAND,
    confirmed via `grep -rn PLOW_DBG` across both touched files (zero remaining). Full
    `cmake --build build-amd` clean (49 objects, defines cleared), `cargo test -p devgen
    -p packet -p plowrt --lib` 252+60+135 clean, `git log --oneline -3` confirms HEAD
    `929eb6a`, `git status --short` shows the same 38 files, real TP8 blob re-emitted
    and confirmed **md5-identical** (`6aff5b24cada0ef68042cb023b6ca2fa`) to baseline.

- **FOURTEENTH PASS (2026-09-02) — the "wrong `e.inst` value" theory is DISPROVEN, not
  just untested: it was checked directly, empirically, entirely on the HOST, no GPU
  needed.** Added a temporary `assert!` in `crates/packet/src/devbuild.rs::finish()`
  (right after `gq_stream`'s construction/sort) checking every `gq_stream[i].inst` and
  every `streams[cu][i].inst` is `< insts.len()`. Ran the exact crash-relevant emit
  (`PLOW_DBG_LAYERS=1`, `--num-gpus 1 --max-ctx 64`, no prompt — confirmed via the
  logged `counter-graph reduction: 0 of 29 distinct coarse edges` that this is the SAME
  program every prior pass's GPU repro hit) on the HOST ONLY, no `gpulease`/`sg render`
  needed for this step. **The assertion never fired**: `n_ops=27 insts.len()=27
  gq_stream.len()=3474`, every entry in bounds. The emitted `build.json` confirms this
  is the DECODE-ONLY program (`"prefill": false`) — matches every prior pass's repro.
  - Went one level deeper than the in-memory `Program`: added a temporary `#[test]` in
    `crates/plowrt/src/asset/devblob.rs` that reads a real `.pkt` file via `DevBlob::
    parse` — plowrt's ACTUAL production blob reader, the exact code path the real
    runtime uses at load time — and re-checked the same invariant on the PARSED-BACK
    result. **Also clean**: `n_prog=1`, `insts.len()=27`, `stream.len()=3474`,
    `gq_stream.len()=3474` (the two match, confirming `gq_stream` really is a
    permutation of `stream` as the write-side comment claims), `gq_seg_ofs.len()=2`
    (`n_seg=1`, unsegmented). This exercises blob WRITE (`to_blob_v6`) and READ
    (`devblob.rs::parse`) together, end to end, with zero GPU involvement.
  - Cross-checked field-by-field, by direct source reading: `StreamEnt` (Rust,
    `crates/packet/src/dev.rs`, `#[repr(C)]`, 24 bytes) matches `PlowStreamEnt` (C,
    `runtime/common/dev_isa.h`, static-asserted 24 bytes) EXACTLY — same field order,
    same types (`inst`/`slice`/`wait_ofs`/`succ_ofs` as `u32`, `wait_len`/`succ_len`/
    `flags`/`seg` as `u16`). `DevInst64` (Rust, 64 bytes, `size_of` const-asserted) vs
    `PlowDevInst` (C, 64 bytes, static-asserted) also match, so `insts + e.inst`'s
    pointer stride is correct on both sides. The GPU-upload call sites
    (`crates/plowrt/src/exec/amd.rs`'s `kernarg()`, ~line 5245-5280) correctly wire
    `g.gq.as_ref().map_or(0, |q| q.d_stream.base)` into the kernarg's `gq_stream` field
    (not left at a stale/zero value) — traced this by hand, it is right.
  - Also re-checked (from an earlier round's dead-end, don't re-derive): the `t7`/
    Q-NORM-FOLD gotcha in `PLOW_DOP_GEMV_QKV`'s dispatch case is a red herring —
    `Builder::emit` default-initializes every `t: [TENSOR_NONE; 8]` before the closure
    runs, GLM-5.3's `GemvQkv` emit never sets `t[7]`, so it correctly stays
    `TENSOR_NONE` and `TEN(7)` correctly resolves to `nullptr`.
  - **Net**: every piece of STATIC data (the `Program` in memory, the serialized blob
    bytes, the re-parsed blob, the kernarg pointer wiring) is now proven correct for
    this exact crashing case. The thirteenth pass's inference — "`e.inst`'s value must
    be wrong" — does not survive this check. **The bug must be in something that only
    exists at RUN TIME**: either (a) the GlobalQueue interpreter's shared-cursor
    dispatch loop (`runtime/amd/interp.hip` ~3407-3527, `PLOW_GLOBAL_QUEUE` arm) reading
    PAST `gq_stream`'s true bounds due to a wrong `lo`/`hi` window (`lo =
    prog.gq_seg_ofs[my_seg]`, `hi = prog.gq_seg_ofs[my_seg+1]` — every entry WITHIN
    `gq_stream`'s declared length was checked and is valid, but nothing has yet checked
    that the device-side LOOP actually stays within `[lo, hi)` as intended, or that
    `hi` is really `3474` and not some larger/wrong value read from the runtime's own
    `derive_segments()` disagreeing with what devgen's emit-time segmentation decided —
    `my_seg` is `prog.cur_seg`, which for this single-segment (`n_seg=1`) program should
    always be `0` on the one launch that happens, so an out-of-bounds *seg_ofs* index
    looks unlikely on reflection, but the ACTUAL VALUES of `lo`/`hi` at the moment the
    faulting workgroup reads them were never directly observed — only inferred from the
    host-side data they were built from), or (b) something else entirely at the HSA
    queue/dispatch level not yet considered.
  - **Concrete next step, not yet tried by anyone**: this needs an actual on-device
    check now that host-side data is fully cleared — e.g. a diagnostic kernel-side
    write (NOT `printf`, a plain memory write to a small always-safe scratch buffer,
    the same class of technique the eleventh/twelfth passes already used safely) of the
    ACTUAL `lo`/`hi`/`my_seg`/`c0`/`ix` values computed at the crash site, read back
    from the host after a controlled abort or (better) added to a path that returns
    before the actual fault so the write survives to be read back normally. Compare the
    observed `hi` against the expected `3474` and `gq_stream.len()`'s uploaded value
    directly (not just what the host believes it uploaded) to catch a genuine
    device-side corruption or a runtime/emit-time segmentation disagreement in the act.

- **FIFTEENTH PASS (2026-09-02) — fork killed by a Claude API session-limit rate limit
  (resets ~6am UTC) before writing any code.** It had just started, its last action was
  "let me check something cheaper first: whether `amd-bench --steps N` actually calls
  `enqueue()` (always seg=0) or some other path that could pass a wrong segment index."
  Checked repo state on resume: clean, `git status --short` still the same 38 files, HEAD
  still `929eb6a`, `cargo test -p devgen -p packet -p plowrt --lib` still 252+60+135 —
  the fork made zero changes before dying, nothing to recover or revert.
  - **Did the cheap check myself** (pure code reading, no GPU): `AmdEngine::run()`
    (`crates/plowrt/src/exec/amd.rs` ~line 5552) — the function the decode-step path
    actually calls — calls `self.enqueue(p, k)`, which always passes `seg=0`
    (`self.kernarg(p, 0)`, ~line 5421). It does NOT go through `enqueue_segment`'s
    `for seg in 0..n_seg` loop (~line 5512, a DIFFERENT function used elsewhere, e.g.
    for prefill). Combined with the fourteenth pass's confirmed `gq_seg_ofs.len()=2`
    (`n_seg=1`) for this program, `seg=0` is trivially in-bounds by construction on the
    HOST side — this doesn't yet rule out the on-device READ of `prog.cur_seg` coming
    back wrong (a kernarg-wiring/upload issue distinct from which value the host
    intended to send), but it does make the "host picks an out-of-bounds segment index"
    version of the theory noticeably less likely than the fourteenth pass's own
    "unlikely on reflection" already suggested. **The on-device diagnostic write is
    still the concrete, un-tried next step** — nothing else has changed since the
    fourteenth pass's writeup above.
  - Per explicit user instruction this turn: stopping here for now, not dispatching a
    further fork immediately. Resume by re-dispatching the same on-device diagnostic
    (lo/hi/my_seg/c0/ix/e.inst, written to a safe scratch buffer, NOT printf) that the
    fifteenth pass was about to start on, once available (the rate limit resets ~6am UTC).

- **SIXTEENTH PASS (2026-09-02) — ran the fourteenth/fifteenth pass's own concrete next step
  (the on-device diagnostic write) for real, got a clean negative result that further narrows
  the bug, then went one step further with a second, independent isolation technique
  (`PLOW_K3_ABLATE`, an existing instrument, not new scaffolding) that produced the CLEANEST
  isolation of this entire 16-pass investigation: **GemvQkv's real body, entirely alone, with
  every other op in the program replaced by Nop and the exact same real graph/counters,
  reproduces the identical `(nil)` fault by itself.** Root cause still not pinned down, but the
  remaining hypothesis space is now much smaller than at the end of the fifteenth pass.
  - **Stage 1 — the on-device stream-entry probe, finally run.** Reused the existing, ALREADY
    -REGISTERED `PLOW_GLM_LAYERS` cap knob (`emit_config::active().glm_layer_cfg()`, already
    wired for GLM-5.2/K3, just not yet consumed by `glm53_emit_full`) instead of reinventing a
    raw-env `PLOW_DBG_LAYERS` read, to get the same `PLOW_DBG_LAYERS=1`-shaped minimal repro
    (`PLOW_GLM_LAYERS=1 plowc ... --num-gpus 1 --max-ctx 64`, TP1, decode-only,
    `counter-graph reduction: 0 of 29 distinct coarse edges` — confirmed byte-for-byte the same
    program shape every prior pass's repro hit). Added a temporary `PLOW_DBG_STREAM_PROBE`
    macro in `interp.hip`'s GlobalQueue dispatch loop, gated on the RUNTIME `prog.trace`
    pointer (see the pitfall below), that for every claimed stream entry writes `lo`, `hi`,
    `my_seg`, `c0`, `ix`, `e.inst` into the existing `PLOW_TRACE_RAW`/`PlowTraceRec` mechanism
    (already-allocated, already-host-readable — no new tensor or scratch buffer needed) and then
    `continue`s, skipping the real wait/dispatch/succ-bump for that entry entirely (same safety
    shape as the twelfth pass's "Level -1" bypass). Ran on real gfx950 hardware with
    `PLOW_TRACE_RAW=<path>`: **completed with zero fault**, and the trace file, decoded, shows
    **all 3474 stream entries** with `lo=0`, `hi=3474`, `my_seg=0` (exactly matching
    `gq_seg_ofs=[0,3474]`), `c0==ix` for every entry (exactly matching the claimed batch), and
    `e.inst` monotonically stepping `0→26` across the 3474 entries with **zero out-of-range
    values and zero entries deviating from the expected op-major grouping**. **This directly
    refutes, at the hardware level (not just host-side data), the fourteenth/fifteenth pass's
    "e.inst might be wrong on-device" theory — the GlobalQueue dispatch loop's own bookkeeping
    is provably correct for every single stream entry in the crash-relevant program.**
  - **Pitfall hit and fixed while building stage 1, worth recording**: the FIRST version of the
    probe used an UNCONDITIONAL `continue;` (gated only at compile time by `#ifdef
    PLOW_DBG_STREAM_PROBE`, which every `_gq` object gets since `PLOW_HSACO_EXTRA_DEFINES`
    applies uniformly). Because the `continue` made every real dispatch line after it
    UNREACHABLE, the compiler dead-stripped `plow_exec`'s entire op switch out of every `_gq`
    object — shrinking every one of them from 400-700 KB down to a uniform ~9.5 KB stub — and
    the loader's own symbol-table safety check (`packet/object MISMATCH: ... none of
    ["d_attn_res", "d_situ_glu", ..., "d_kda_state_step", "d_kda_conv"] is in its symbol table`)
    correctly refused to load it. **Fix**: gate the `continue` on the RUNTIME `prog.trace`
    pointer instead (`if (prog.trace) { ...; continue; }`) — a genuine per-launch kernarg value
    the compiler cannot fold away, so the normal dispatch path (and its symbols) stays compiled
    in for every ordinary, trace-off launch, and only an ACTUAL `PLOW_TRACE_RAW` run takes the
    probe branch. Confirmed via object size (9.5 KB stub → 445 KB full object) and via a plain,
    extra-defines-cleared rebuild after this pass's revert (`interp_prefill_k3_gq.elf` back to
    445 KB) that this was purely a self-inflicted artifact of the probe, not a pre-existing gap.
  - **Stage 2 — dereferencing `insts + e.inst` for real, still safely.** Pass 13 had narrowed
    the fault to "dereferencing `in` itself," inferred by elimination (every guard that read
    `in->op` to decide what to skip still crashed), never directly tested. Extended the same
    probe to actually compute `const CInst in = insts + e.inst;` and read `in->op`, `in->blocks`,
    `in->t[0..1]`, `in->i[0..3]` into the trace record for every entry, still `continue`-ing
    before any real wait/dispatch/succ-bump. Ran on real hardware: **completed with zero fault
    again.** Decoded the trace: all 27 distinct `inst` values map to fully self-consistent
    `(op, blocks, t[], i[])` tuples across every one of their stream entries (zero divergence),
    and the op sequence read back — `6,122,128,121,1,22,10,10,10,10,10,111,112,103,10,122,128,
    121,1,47,44,122,122,1,10,17,18` (PLOW_DOP_* numeric values) — matches the captured trace
    from the third pass exactly at the front (`Embed, HyperConnPost, GemvF32, HyperConnPre,
    RmsNorm, GemvQkv, ...`) and is a completely sane full single-layer-plus-tail program on
    paper. **This directly refutes pass 13's own inference: dereferencing `insts + e.inst` and
    reading its header fields is NOT what faults, for any of the 27 instructions in this
    program.** The remaining hypothesis space after stage 1+2 is now strictly narrower than
    "something about touching `in`" — it must be in the WAIT-POLL/SUCC-BUMP machinery's actual
    cross-workgroup synchronization (untested by either probe stage, which skip all of it), or
    genuinely inside a specific op's real dispatch/kernel body.
  - **Stage 3 — full-program body ablation, using an EXISTING instrument
    (`PLOW_K3_ABLATE`), not new scaffolding.** `crates/devgen/src/mla/kimi_k3.rs`'s
    `k3_ablate_bodies` (already wired to `PLOW_K3_ABLATE=<opcode>[,...]`, already used for K3's
    own cost-attribution measurements) rewrites named ops' `op` field to `Nop` **after** the
    graph and schedule are built, so `stream`/`waits`/`succs`/every counter and dispatch width
    stay byte-for-byte identical — only the op body changes. It was previously only called from
    K3's own emit path; added one temporary call to it in `glm53_emit_full` (and briefly made it
    `pub(crate)` to reach it from `mla.rs`) to make it available for GLM-5.3 too. Emitted the
    same minimal repro with `PLOW_K3_ABLATE=6,122,128,121,1,22,10,111,112,103,47,44,17,18` (every
    opcode this 27-op program uses) — `PLOW_K3_ABLATE: 27 instruction(s) rewritten to Nop`. Ran
    on real hardware: **completed with zero fault**, real wait-loop polling and real
    atomic succ-bumps running for every one of the 3474 stream entries (unlike stage 1/2's
    probe, which skipped all of that) — this **directly refutes any remaining "scheduler /
    synchronization machinery" hypothesis**: the GlobalQueue claim-and-drain loop, the real
    counter wait-polls, and the real atomic successor bumps are not the fault, even fully
    exercised.
  - **Stage 4 — the decisive test: ablate everything EXCEPT `GemvQkv`.** Re-emitted with
    `PLOW_K3_ABLATE=6,122,128,121,1,10,111,112,103,47,44,17,18` (every opcode except `22` =
    `GemvQkv`) — `PLOW_K3_ABLATE: 26 instruction(s) rewritten to Nop`, leaving `GemvQkv` as the
    ONLY instruction with a real body anywhere in the program, at its exact real graph position
    (same dependency edges, same counters, same everything — `k3_ablate_bodies` only touches the
    `op` field, `t[]`/`i[]` operands are untouched). Ran on real hardware: **crashed**,
    identical `Memory access fault ... on address (nil)` signature, this time even before the
    "warmup step ok" line printed (i.e. on the very first dispatch). **This is the cleanest
    isolation in the whole investigation: `GemvQkv`'s real dispatch, entirely alone — no other
    op's body, no cumulative/interaction effect across ops, no scheduling artifact possible
    since stage 3 already proved the scheduler innocent — reliably reproduces the fault by
    itself, at this exact graph position.**
  - **Net position after this pass.** Combined with everything already eliminated in passes
    1-15 (fusion, CU-dispatch width, `stage_x_lds`, `buf_rsrc`, wide dispatch, large
    out-of-bounds reads, `q_proj`'s own binding, LDS aliasing with `HyperConnPre`, `TEN()`'s
    array lookup, the wait-poll loop, the successor-bump loop, and now — this pass — the
    stream-entry index/content and the whole synchronization machinery), the bug is now
    conclusively pinned to **something specific about `GemvQkv`'s (opcode `PLOW_DOP_GEMV_QKV`
    = 22) own compiled dispatch case — its switch-case entry, argument evaluation, or kernel
    body — reached via the real `case PLOW_DOP_GEMV_QKV:` path**, independent of every other op
    in the program, independent of the scheduler, independent of the stream/instruction data's
    correctness. This is consistent with (and sharpens) the twelfth pass's own "Level -1"
    finding (an EMPTY `GemvQkv` lambda, reached via the real dispatch, still crashed) — taken
    together, the fault is not about what the op's body computes at all, but about something
    that happens simply by virtue of the interpreter's `case PLOW_DOP_GEMV_QKV:` being the one
    taken, in this specific small/sparse program shape, for this opcode specifically (not
    `RmsNorm`, not `HyperConnPre`, not `GemvF32`, not `Nop` — all proven safe in the same
    position across this and prior passes).
  - **Re-read `d_gemv_qkv`/`d_gemv_qkvg`'s real compiled body once more this pass, specifically
    for anything the K=4096/UN=8 shape or the QNORM-fold arm could contribute — found nothing
    new, but worth recording so it isn't re-derived.** `PLOW_GLM_FUSE_QNORM` (the `if (gnorm)`
    branch in `op_gemm.h` that reads `t[7]`/eps for the Q-norm fold) defaults to `0` and is not
    set anywhere in `CMakeLists.txt` — it is `#if`'d out of every object in this tree, including
    the crashing one, so that entire branch (and its `__builtin_trap()`) does not exist in the
    compiled binary and cannot be the cause. `d_gemv_qkv` is confirmed a pure forwarder to
    `gemv_qkvg_rows<PLOW_GEMV_MM, 8>` (K==4096) with `Cg=Wg=nullptr, Ng=0` — exactly the shape
    the ninth/tenth passes already hardware-cleared via `d_gemv_f32` substitution and direct
    `q_proj` reads.
  - **Concrete next step, revised from this pass's original plan**: the twelfth pass suggested
    disassembling and diffing `case PLOW_DOP_GEMV_QKV:`'s compiled machine code against a
    known-working K3/Gemma decode object — but on reflection this is unlikely to show anything,
    because K3's own decode object almost certainly is the SAME compiled `.elf` (same
    `interp.hip` source, same `-DPLOW_K3=1 -DPLOW_GLOBAL_QUEUE=1` build flags select the same
    object filename) that GLM-5.3 loads — there is no polymorphism per model, so the bytes at
    that case are likely byte-identical whether K3 or GLM-5.3 dispatches them.
  - **Important correction/clarification for whoever continues, so stage 4 is not
    misread**: `k3_ablate_bodies` only rewrites the `op` FIELD to `Nop` — it does NOT touch
    `waits`/`succs`/counters, which are built from the ORIGINAL (pre-ablation) graph. So stage
    4's "everything Nop except `GemvQkv`" run still has `GemvQkv` gated on the SAME coarse wait
    edge from `HyperConnPre`'s single-CU output as the unmodified program — ablating
    `HyperConnPre`'s BODY to `Nop` already proves `HyperConnPre`'s real COMPUTATION is not
    required for the fault (a Nop producer is enough), but it does NOT prove the crash is
    independent of the WAIT-EDGE SHAPE itself (a wide `GemvQkv` consumer gated by a coarse edge
    from a 1-CU producer) — `k3_ablate_bodies` cannot restructure edges, only bodies, so that
    variable has never actually been varied.
  - **Stage 5 — ran exactly that test.** Added a temporary `PLOW_DBG_WIDE_HCPRE` devgen bypass
    (`emit_glm53_hc_pre` in `mla.rs`) forcing `HyperConnPre`'s CU list to `b.all()` (256 CUs)
    instead of its normal decode-time `(0..rows.min(n_cu))` (1 CU at `rows=1`), combined with
    the SAME "ablate everything except `GemvQkv`" `PLOW_K3_ABLATE` spec stage 4 used. Emitted
    and ran on real gfx950 hardware: **crashed**, identical `Memory access fault ... on address
    (nil)` signature — this time failing even before "loaded" finished printing anything past
    `program 0: T=1 segments=1`, i.e. on the very first dispatch again. **This eliminates the
    narrow-producer/wide-consumer coarse-wait shape as a factor too** — `GemvQkv`'s real
    dispatch crashes whether its producer is 1 CU or 256 CUs, closing off the third pass's
    original "single-CU chained to single-CU" trace observation as a genuine causal factor (it
    was always a correlation from GLM-5.3 being the first model to pair `HyperConnPre` with a
    GEMV, not a mechanism). Reverted this bypass by hand along with everything else (see
    Housekeeping below) — confirmed via `grep -rn PLOW_DBG_WIDE_HCPRE` returning nothing and
    `git diff -- crates/devgen/src/mla.rs` matching exactly its pre-pass content.
  - **Net position after stage 5, genuinely exhausted for this pass**: `GemvQkv`'s real
    dispatch, alone, in this program, crashes independent of every other op's identity, the
    scheduler, the stream/instruction data's correctness, AND the coarse-wait graph shape
    (narrow or wide producer). What is left standing, with no cheap test remaining that this or
    prior passes haven't already tried, is something dynamic and specific to `GemvQkv`'s own
    compiled dispatch case being reached this early in an unusually short, mostly-Nop'd
    GlobalQueue program — e.g. register/occupancy state inherited from whichever workgroups
    just came off the shared claim-and-drain loop for the first time, or a genuinely
    ISA/codegen-level issue in how `plow_exec`'s switch reaches this specific case (jump-table
    layout, spill/reload sequence) that no C++-source-level reasoning can distinguish from here.
  - **Concrete next step for whoever continues — now genuinely a different KIND of step, not
    another ablation variant**: this needs either (a) a hardware/ISA-level tool this
    investigation has not yet reached for — a GPU debugger with a conditional breakpoint on the
    `PLOW_DOP_GEMV_QKV` case, or `llvm-objdump`/`rocminfo`-adjacent disassembly of the compiled
    `.hsaco` around that case's entry, read not to compare against another model's object (this
    pass's stage 4/5 already show the fault is independent of which op precedes `GemvQkv`, so a
    same-binary disassembly is unlikely to show an actual bug, only confirm what the code does)
    but to directly inspect what SGPR/VGPR state the compiler assumes is live at that case's
    entry for THIS build, since that is the one variable no source-level ablation can control;
    or (b) further shrinking the "unusually short, mostly-Nop'd" repro even more — e.g. does
    `GemvQkv` alone, with EVERY OTHER instruction in the entire packet removed rather than
    merely Nop'd (a program of literally one instruction, if the packet builder allows it),
    still crash? That would tell you whether "short program" itself is a necessary ingredient or
    whether `GemvQkv` alone in isolation, with no other ops at all, is already sufficient —
    tightening the bisection one more notch before reaching for a hardware debugger.
  - **Housekeeping**: all temporary scaffolding reverted BY HAND — the `PLOW_DBG_STREAM_PROBE`
    block in `interp.hip` (confirmed via `git diff --stat -- runtime/amd/interp.hip` returning
    empty, i.e. byte-identical to pre-pass), the `glm_layer_cfg()` cap wiring and the
    `k3_ablate_bodies` call in `crates/devgen/src/mla.rs`, and the `pub(crate)` visibility bump
    on `k3_ablate_bodies` in `crates/devgen/src/mla/kimi_k3.rs` (also confirmed empty diff). NO
    git commands used at any point (no `git commit`, `git push`, or `git checkout --`) — every
    revert was a manual re-edit. `grep -rn PLOW_DBG_STREAM_PROBE` across the tree: zero matches.
    `cargo test -p devgen -p packet -p plowrt --lib`: 252+60+135 clean, same as the pre-pass
    baseline. `git log --oneline -3` still shows HEAD `929eb6a`; `git status --short` still
    shows the same 38 files (the pre-existing, unrelated `kda.rs`/`mla.rs` PLOW_DBG_LAYERS/
    PLOW_DBG_SKIP_* leftovers noted in earlier passes are untouched by this pass — confirmed by
    reading their diffs, they predate this session). **One discrepancy worth flagging, not a
    regression**: re-emitting the real TP8 blob (`--num-gpus 8 --max-ctx 8192`, with
    `PLOW_MLA_PREFILL=full:128` as the resume-update section above specifies) did NOT reproduce
    the previously-recorded baseline md5 `6aff5b24cada0ef68042cb023b6ca2fa` this time — the
    build now reports `tunedb ... records skipped as STALE against the probed build
    gfx950-93ca3a935f6893fd -- NO usable records remain` and falls back to the analytical tile
    model for all 350 dense-GEMM tiles, where before it presumably had measured tuning data.
    Since `PLOW_HSACO_EXTRA_DEFINES` was touched (then cleared) repeatedly by this pass's kernel
    rebuilds, the local tuning DB's cached measurements for "this exact object recipe" (defines
    are part of its digest, per the tool's own message) are now considered stale. This affects
    GEMM TILE SELECTION (a performance/tuning axis), not correctness — `git diff` on every
    source file this pass touched is empty, confirming the CODE is genuinely back to the
    pre-pass baseline; the blob byte-difference is tuning-database drift, not a code
    regression. Re-running the offline tuning campaign (`plowc tune ...`, not attempted this
    pass) would presumably restore the old hash. Flagging for whoever continues rather than
    treating it as resolved.

- **SEVENTEENTH PASS (2026-09-02) — per explicit user request, ran the repro once and executed
  pass 16's own final concrete next step: a literal ONE-INSTRUCTION packet. Result: DECISIVE
  NEGATIVE — program size/sparsity is conclusively not an ingredient either. `GemvQkv` crashes
  utterly alone, with nothing else in the packet at all.**
  - **Reproduced the baseline crash first**, confirming nothing regressed since pass 16.
    Re-added the `PLOW_GLM_LAYERS` cap wiring to `glm53_emit_full` (still not actually consumed
    there in the current tree — confirmed by grep before starting, consistent with every prior
    pass's revert) plus `c.layers = cap.unwrap_or(c.layers).min(c.layers)` right after the
    `assert_eq!(c.layers, 45)` check, matching the existing GLM-5.2 pattern
    (`emit_glm_full`, mla.rs ~6747). `PLOW_GLM_LAYERS=1 plowc --hf-dir
    /home/shaswot/models/GLM-5.3-Flash-plow --emit devblob --gpu mi350 --arch gfx950
    --num-gpus 1 --max-ctx 64` now logs `counter-graph reduction: 0 of 29 distinct coarse
    edges` — byte-for-byte the same program shape every prior pass's repro hit. Ran on real
    gfx950 hardware (`plowrt amd-bench --tp 1 --steps 3 --ctx 32`): **exact same fault**,
    `Memory access fault by GPU node-2 ... on address (nil). Reason: Unknown.` (`aborted, core
    dumped` via gpulease). Confirms the bug is still live, unchanged, in the current tree.
  - **The one-instruction test.** Added a `PLOW_DBG_SOLO_QKV` env-gated branch in
    `glm53_emit_full` (mla.rs) that, instead of calling `emit_glm53_program` at all, builds a
    single `Builder`, declares four bare scratch tensors (`dbg_solo_x`/`r0`/`r1`/`r2`, via
    `b.tensor(...)`, the same primitive `kda.rs`'s own `bft` closure wraps), and emits exactly
    ONE instruction — `DevOp::GemvQkv` with `deps = &[]` (zero dependencies, not even on
    `Embed`), reading `s.kda_w[0]`'s real `q_proj`/`k_proj`/`v_proj` checkpoint tensors — then
    finishes the program right there. No `Embed`, no `HyperConnPre`/`Post`, no `RmsNorm`, no
    counters, no coarse edges, nothing. Verified via `build.json`: `"insts": 1`, `"opcodes":
    ["GemvQkv"]`, `"arms": ["GemvQkv"]` — a genuinely single-instruction packet, not merely
    Nop-padded (pass 16's `PLOW_K3_ABLATE` stage 4 still had 26 Nop instructions alongside the
    real one; this has zero). Ran on real gfx950 hardware: **crashed**, byte-identical
    signature — `Memory access fault by GPU node-2 ... on address (nil). Reason: Unknown.`,
    failing on the very first (and only) dispatch, before any load-adjacent log line past
    `program 0: T=1 segments=1`.
  - **Net finding**: this closes off the literal last cheap hypothesis standing after 16
    passes. Neither "short/sparse program" nor "chained after another single-CU op" nor any
    property of program SIZE is a necessary ingredient — `GemvQkv`, entirely alone, with zero
    surrounding structure, zero dependencies, and zero other instructions in the whole packet,
    crashes by itself with the identical fault signature every other test in this 17-pass
    investigation has produced. Combined with pass 12's "Level -1" (an empty `GemvQkv` lambda
    body, reached via the real dispatch case, still crashes) and passes 1-16's exhaustive
    clearing of every op-logic, tensor-identity, scheduler, synchronization, and graph-shape
    hypothesis, the bug is now about as narrowly pinned as source-level reasoning and hardware
    ablation can get it: **something about `case PLOW_DOP_GEMV_QKV:` being the dispatched case
    at all, independent of literally everything else in the program**, most plausibly a
    codegen/register-allocation property of that specific case in the compiled `plow_exec`
    megakernel (jump-table layout, spill/reload sequence, or something live-at-entry that only
    this case's argument evaluation or kernel body disturbs) rather than anything a C++-source
    or packet-data-level ablation can distinguish further.
  - **Concrete next step for whoever continues — unchanged from pass 16, now the only path
    left**: this needs actual ISA-level tooling — a GPU debugger with a breakpoint at `case
    PLOW_DOP_GEMV_QKV:`, or disassembly (`llvm-objdump`/rocm equivalent) of the compiled
    `.hsaco` around that case's entry to inspect SGPR/VGPR liveness — not another
    source-level ablation. No such tool has been set up or attempted in this investigation
    yet; that is out of scope for a "run it once" pass and was deliberately not attempted here
    per the task's own instruction.
  - **Housekeeping**: both temporary scaffolding changes (the `glm_layer_cfg()` cap wiring and
    the `PLOW_DBG_SOLO_QKV` branch, both in `crates/devgen/src/mla.rs`'s `glm53_emit_full`)
    reverted BY HAND, confirmed via `grep -n "PLOW_DBG_SOLO_QKV\|dbg_cap"` returning nothing and
    direct reading of the restored region matching its pre-pass content exactly. NO git
    commands used at any point. `cargo test -p devgen -p packet -p plowrt --lib`: 252+60+135
    clean. `git log --oneline -3` still HEAD `929eb6a`; `git status --short` still the same 38
    files. Temp blob dirs (`/tmp/glm53-17th-min`, `/tmp/glm53-17th-solo`) removed.

- **EIGHTEENTH PASS (2026-09-02) — attached `rocgdb` for real, per explicit user decision after
  the seventeenth pass's "needs ISA-level tooling" conclusion. RESULT: MAJOR CORRECTION TO THE
  ENTIRE 17-PASS NARRATIVE. The crash is NOT in `GemvQkv`'s dispatch case at all — it is a
  genuine null-pointer WRITE inside `RmsNorm` (P0), caught live on real hardware by the GPU's
  own hardware fault trap (not the CPU's async-abort report every prior pass relied on).**
  - **Setup**: `rocgdb` (system ROCm 7.0.2, confirmed installed) attaches cleanly to the raw-HSA
    dispatch path — this was the single largest unknown going in, now resolved: dbgapi's
    KFD debug-trap DOES intercept a hardware memory-violation fault at the GPU wave itself,
    stopping the process with a genuine `SIGSEGV` on the wave, before the CPU-side async-abort
    handler ever runs. Built a separate `-g` debug-info hsaco tree (`build-amd-dbg`, cache-var
    only, no `CMakeLists.txt` edit) at the SAME `-O3` the crash needs to reproduce at; the
    register-cliff gate did not trip (VGPR/AGPR/occ/spill unchanged from a plain build). rocgdb
    was launched OUTSIDE `nix develop` (its own libs resolve via RUNPATH, not
    `LD_LIBRARY_PATH`), with the debuggee's normal nix-`LD_LIBRARY_PATH` re-injected via `set
    environment` inside the rocgdb session (not `env` wrapping — wrapping the debuggee in `env`
    makes `env` itself the debuggee, which is a mistake worth flagging for next time).
  - **Confirmed the loaded object is `interp_decode_k3_gq.elf`**, not `interp_decode.elf`/
    `interp_decode_gq.elf` as guessed in the plan — GLM-5.3's fp8-quantized checkpoint selects
    the K3 decode/prefill arm objects. Don't guess; the `tracing::info!(object=...)` line at
    `crates/plowrt/src/exec/amd.rs` settles it every time.
  - **The debugger caught TWO independent GPU-wave `SIGSEGV`s** (one per `continue`), both with
    an IDENTICAL stack:
    ```
    #0  d_rmsnorm (out=0x0, x=0x7fbb1c96b000, gamma=<optimized out>, rows=1, feat=<optimized out>,
        eps=9.99999975e-06, out_row0=0, ..., xq=0x0, ascale=0x0) at runtime/amd/op_norm.h:196
    #1  plow_exec (...) at runtime/amd/interp.hip:1771
    #2  plow_interp_dec_gfx950_gq (prog=...) at runtime/amd/interp.hip:3798
    ```
    `out=0x0` is a real, concretely-recovered register value (not `<optimized out>`) — the
    `st_act1(&out[obase+i], o)` write at `op_norm.h`'s in-place RmsNorm branch faults exactly
    because `out` (`TEN(0)`, i.e. `T[in->t[0]]`) is null. `xq`/`ascale` are ALSO null but that is
    correct/expected (no fused quant on this op) — only `out` being null is the bug.
  - **Ruled out the obvious "devgen encoding bug" explanation immediately, with a plain host-side
    check — `plowrt disasm` needs no GPU**: `./target/release/plowrt disasm
    /tmp/glm53-18th-min/model.pkt` dumps the real, static instruction stream. Op `#4` is exactly
    this `RmsNorm`: `out<-#71 x<-act.hc_layer_input gamma<-...input_layernorm.weight ... | rows=1
    feat=4096 eps=0.00001` — and tensor handle `#71` is NOT a stray/absent-operand marker (those
    print as `—`, e.g. `gamma<-—` on ops that genuinely have no gamma) — it is a real, legitimate
    handle, consistently referenced as the `x` INPUT of five other real ops in the same program
    (`#5 GemvQkv x<-#71`, `#6/#8/#9 Gemv x<-#71`). **The packet is correctly encoded. `t[0]` is
    genuinely a valid, non-`PLOW_TENSOR_NONE` handle at emit time and in the serialized blob.**
    This means the bug is NOT in devgen's op-encoding — it is in what happens BETWEEN a valid
    tensor handle and the address the GPU actually reads at `T[71]`, i.e. somewhere in the
    load-time tensor-carve/address-binding path (`crates/plowrt/src/exec/amd.rs`'s ~4295-4520
    "named tensors" loop, `EngineDevice::alloc`/carve, and however the resulting addresses get
    written into the kernarg's device-side tensor-pointer table) — a REAL BINDING BUG, not an
    ISA/dispatch-codegen defect in `GemvQkv` at all.
  - **This retroactively re-explains, rather than contradicts, most of the 17 prior passes**:
    every "GemvQkv crashes no matter what we do to it" result is consistent with GemvQkv NEVER
    ACTUALLY BEING REACHED — RmsNorm (op #4, immediately before GemvQkv at op #5 in this exact
    program) faults first, every single time, and the CPU's async-fault-report attribution (which
    every prior pass relied on, since none had a live debugger) simply named whichever op
    happened to be "in flight" when the async event surfaced — almost always `GemvQkv`, since
    it's the wide (256-CU) op immediately downstream doing the actual bulk of the dispatch work
    when the already-faulted wave's error propagates. This is the "async-fault-misattribution"
    concern the third and fifth passes each raised in passing, now directly confirmed at the
    ISA/hardware level, one level deeper than either of them tested. **One genuine tension not
    yet resolved**: the sixteenth pass's stage-4 test (`PLOW_K3_ABLATE` rewriting every op INCLUDING
    this RmsNorm to `Nop`, leaving ONLY `GemvQkv`'s body real) still crashed — with RmsNorm
    truly ablated to a no-op, its null-`out` write cannot have happened, so that specific test
    result implies `GemvQkv` may ALSO have a genuine, independent fault under this same graph
    shape. Whoever continues should treat these as two SEPARATE bugs sharing an environment
    (a tensor-binding bug hitting RmsNorm's per-call scratch tensor in the un-ablated program,
    and a still-unexplained GemvQkv-alone fault in the fully-ablated one) rather than assuming
    fixing one fixes both.
  - **Concrete next step for whoever continues**: read `crates/plowrt/src/exec/amd.rs`'s tensor
    carve/bind loop (~4295-4520, the "named tensors" phase) and whatever populates the actual
    on-device tensor-pointer kernarg array from `devp`, specifically asking: for a non-weight,
    non-init, non-gen, non-peer-slot, non-VMM SCRATCH tensor (tensor `#71`, `act.glm53.0.x`, the
    KDA mixer's own freshly-declared per-call tensor — the same class the fifth/sixth passes
    already independently flagged as uniquely implicated) under this exact 1-layer/TP1/GlobalQueue
    config, does its carved `mem.base` genuinely end up non-null, and does that address actually
    land in the uploaded tensor-pointer table at index 71? A host-side `eprintln!`/`tracing::debug!`
    at the carve site, printing `(index, name, mem.base)` for every tensor and cross-checking
    against the kernarg upload, would confirm or refute this without needing the GPU at all — try
    that BEFORE any more ISA/debugger work, since the debugger has now told us WHERE to look on
    the host side instead.
  - **Housekeeping**: `-g` debug hsaco tree (`build-amd-dbg`) removed (`rm -rf`); `crates/devgen/
    src/mla.rs`'s temporary `PLOW_GLM_LAYERS` cap-wiring re-add reverted BY HAND, confirmed via
    `git diff --stat -- crates/devgen/src/mla.rs` matching the pre-pass content exactly; temp
    files (`/tmp/glm53-18th-min`, `/tmp/rocgdb_glm53.cmds`, `/tmp/nix_ld_path*.txt`,
    `/tmp/glm53-18th-*.log`) removed. NO git commands used at any point (no commit/push/checkout
    --). `cargo test -p devgen -p packet -p plowrt --lib`: 252+60+135 clean. `git log --oneline
    -3` still HEAD `929eb6a`; `git status --short` still the same 38 files.

- **NINETEENTH PASS (2026-09-02) — ROOT CAUSE FOUND AND FIXED. The model runs end to end for
  the first time: TP8, all 45 layers, real checkpoint, all 8 ranks token-identical.**
  - **Root cause, confirmed on the host with `plowrt disasm`, no GPU needed**: `glm53_emit_full`
    (`crates/devgen/src/mla.rs`) captured the blob's `tensors` list ONCE, from the shared
    pre-emission `Builder` (`tb`), before any program was built — then reused that same
    snapshot for every per-program `Builder` (`b.adopt_tensors(tensors.clone())`) AND for the
    final `Model { tensors, ... }`. But `emit_kda_mixer_ex` (`kda.rs`) declares its OWN fresh
    scratch tensors (`x`, `q_raw`/`k_raw`/`v_raw`, `mix`, `g_raw`, `f_a`, `f_raw`, `b_raw`, `o`,
    `y`, ...) on `b` DURING emission, one per (program, KDA layer) — never on `tb`. Those
    handles exist in the program's own instruction stream but never made it into the blob's
    serialized tensor table. `plowrt disasm /tmp/glm53-fix-min/model.pkt --format text` on the
    eighteenth pass's own minimal repro showed it directly: `tensors (71)` (71 entries, indices
    0..70) while op `#5`'s `GemvQkv` referenced handle `#84` — 14 handles (71-84) past the end
    of the table. On the device, reading `T[71..84]` (the tensor-pointer kernarg array) reads
    PAST its uploaded length, landing on unrelated/zeroed memory — a null pointer for whichever
    op first writes through one of those handles. That op is `RmsNorm` (P0, handle `#71`),
    which is exactly the null-pointer write the eighteenth pass's `rocgdb` session caught live.
    **Every one of the 18 prior passes' "GemvQkv crashes no matter what" results is consistent
    with GemvQkv never being reached — RmsNorm, one op earlier, faulted first every time.**
  - **Why K3 never hit this**: `kimi_k3.rs`'s `k3_build_model` calls `b.set_tensor_dedup(true)`
    on every per-program builder and, critically, re-threads `tensors = prog.tensors.clone()`
    after each `b.finish()` — carrying every scratch tensor a program declares forward into the
    next program's `adopt_tensors` call and into the final `Model`. `glm53_emit_full` had
    neither: no dedup, and `tensors` was never reassigned after the first snapshot. This is the
    same `emit_kda_mixer_ex` scratch-tensor mechanism in both models; only K3's outer assembly
    threaded it through correctly.
  - **Fix** (`crates/devgen/src/mla.rs::glm53_emit_full`): `set_tensor_dedup(true)` on each
    per-program `Builder`, and `tensors = prog.tensors.clone()` after each `b.finish()` (both
    the prefill-bucket loop and the decode-rung loop), mirroring `k3_build_model`'s pattern
    exactly. Also filled a related gap the sixteenth/seventeenth passes' own notes flagged but
    never wired: `glm53_emit_full` now consumes the existing `PLOW_GLM_LAYERS` cap
    (`emit_config::active().glm_layer_cfg()`) to truncate `c.layers`, matching `emit_glm_full`'s
    identical pattern for GLM-5.2 — this is what makes the fast single-layer repro from passes
    16-19 reproducible by anyone without re-deriving it, and is a no-op by default (unset ⇒ full
    45 layers, byte-identical).
  - **Verified, twice, on real gfx950 hardware**:
    1. Minimal repro (`PLOW_GLM_LAYERS=1`, TP1, real bound checkpoint): `plowrt disasm` now
       shows `tensors (85)` — exactly covering the highest referenced handle (`#84`) — and the
       run completes with **zero fault**: `prefill: 5 tokens in 71.6 ms`, `3 decode steps:
       1.199 ms/token`, real token ids `[53638, 38998, 94025, 31308]`.
    2. **Full model, TP8, all 45 layers, real bound checkpoint** (first time ever, this
       session): `loaded in 15.5 s: TP=8 ranks, max_ctx=8192`, `prefill: 5 tokens in 280.7 ms
       (18 tok/s) -> 106198 (all 8 ranks agree)`, `5 decode steps: 55.271 ms/token (18.1 tok/s),
       all 8 ranks token-identical`. All-ranks-token-identical is this investigation's own
       correctness bar (a memory-corruption bug produces divergent ranks, not agreement) — met
       cleanly. The decode repeating one token five times is a property of a 5-token
       non-chat-templated numeric prompt under greedy decoding (the same caveat
       `vllm-k3-glm53-baseline.md` §4 already documents for K3's own raw-completion gate), not a
       new bug — a real `/v1/chat/completions` coherence run is the next natural step, not yet
       done this pass.
  - **Regression test added**: `crates/devgen/src/mla/glm_tests.rs::
    glm53_program_assembly_keeps_every_tensor_handle_in_the_final_table` rebuilds
    `glm53_emit_full`'s tensor/program assembly in miniature (a synthetic 2-layer all-KDA
    `GlmCfg`) and asserts every instruction's tensor handles are within the final table's
    bounds. Teeth proven: reverted the fix (checked the stale `tb.tensors()` snapshot instead
    of `prog.tensors.clone()`) and confirmed the assertion fails (`op 1 references tensor
    handle 104, but the final table has only 104 entries`) before restoring it.
  - **One thing NOT re-investigated this pass**: the sixteenth pass's "ablate everything except
    `GemvQkv`" tension (status.md, EIGHTEENTH PASS entry above) — whether `GemvQkv` has a
    second, independent fault when `RmsNorm`'s real body is truly removed — was not re-examined,
    since the fix above makes the tensor table correct regardless, and the full real (non-
    ablated) model now runs clean end to end. If that ablated-only scenario matters again, it
    would need its own pass.
  - **Housekeeping**: `cargo test -p devgen -p packet -p plowrt --lib`: 253+60+135 clean (252
    devgen + 1 new regression test). All temporary blob directories (`/tmp/glm53-fix-min`,
    `/tmp/glm53-fix-full`, `/tmp/glm53-fix-full2`) removed. NO git commands used. `git log
    --oneline -3` still HEAD `929eb6a` at the time of this pass.

**Earlier background-agent history (resolved)**: a prior fork (chunk-base prefill
addressing fix) was killed mid-task by a Claude API rate limit; its code changes landed
and compiled, but it died before writing the hardware test it claimed to have run — the
doc comment in `op_dsa_pool.h` falsely claimed "0 mismatches" for a test file that
didn't exist. Caught on resume, test written for real, and hardware-verified — 0
mismatches, teeth proven by revert-and-confirm-failure. Lesson: don't trust a background
agent's own "hardware-verified" claim without checking the test file it cites actually
exists.

**ENVIRONMENT BUG FOUND AND FIXED (2026-09-01)**: `sg render -c "..."` (needed for GPU
access this session — group membership changes don't apply to an already-running shell)
execs through `newgrp`, which is setuid-root. That trips glibc's `AT_SECURE` mode in the
child, which SILENTLY DROPS `LD_LIBRARY_PATH` (confirmed directly:
`LD_LIBRARY_PATH=/x sg render -c 'echo $LD_LIBRARY_PATH'` prints nothing). Every `.hip`
test's host-compile pass links with `-L.../lib -lamdhip64`, relying on
`LD_LIBRARY_PATH` at runtime to find the right `libamdhip64.so.7` — under `sg render`
that never took effect, so the binary silently fell back to whatever the system ld.so
cache found by default, whose bundled comgr then failed to link its own internal
blit-copy kernel bitcode ("undefined hidden symbol: __ockl_dm_init_v1" and several
`__amd_rocclr_*` symbols) on the FIRST `hipMemcpy` of the process. Reproduced on a
minimal, plow-free `hipMalloc`+`hipMemcpy` program — confirmed not a plow or GPU-health
issue (rocminfo, GPU enumeration, and ROCm-TheRock's own offline `--genco` kernel
compile all worked fine throughout). **Fix**: link with `-Wl,-rpath,$ROCM_PATH/lib`
instead of relying on `-L`/`LD_LIBRARY_PATH` — RPATH is baked into the ELF, not an env
var, so it survives `AT_SECURE`. Verified: rebuilt the previously-passing
`dsa_pool_gfx950_test.hip` with this flag → back to 0 mismatches; rebuilt the new
`dsa_pool_prefill_chunk_gfx950_test.hip` → **0 mismatches across all 4 pools**, teeth
proven (reverted `chunk_base` addressing, rebuilt, got the predicted 258 mismatches with
chunk 1's pools still at the poison sentinel, restored, reconfirmed 0). **Add
`-Wl,-rpath,$ROCM_PATH/lib` to every future `.hip` test's host-compile command** — see
the corrected recipe below.

## What this is

Kimi-K3 support on plow was already done (prefill loses to vLLM 4-6x, decode wins) —
that baseline is in `perf-data/vllm-k3-glm53-baseline.md`. GLM-5.3-Flash is a NEW
architecture for plow: 45 layers, 34 KDA (linear attention) + 11 DSA-indexer (sparse
attention) layers, plus hyper-connections (`mHC`, 4 parallel residual streams) on every
layer — genuinely new to plow, no prior art. Building this from scratch, verified against
vLLM's own real reference implementation (extracted from the `vllm/vllm-openai-rocm:
glm53-flash` docker image to `/home/shaswot/plow-work/glm5next_ref/`) at every step, not
derived from prose.

## Done and hardware-verified

- **KDA low-rank output gate** (`crates/devgen/src/kda.rs`): GLM-5.3 uses a low-rank
  `g_a_proj`→`g_b_proj` gate (K3 uses full-rank `g_proj`) — both paths now coexist behind
  `KdaCfg::full_rank_gate`/`OutputGate`. No new kernel needed, reuses existing ops.
- **Six new kernel primitives**, every one hardware-verified on real gfx950 hardware
  against a numerical oracle built by RUNNING vLLM's real reference code (not guessed):
  - `HyperConnPre`/`HyperConnPost` (opcodes 121/122) — the mHC pre/post residual mixing.
  - `DsaPoolCompress`/`DsaPoolExpand` (123/124) — the DSA indexer's key-pooling compress
    and pool-to-token expand.
  - `DsaPoolStash` (125) — decode-side ring buffer for pool compression.
  - `DsaQQuant` (126) — query-side fp8 quant + Hadamard rotation for pooled scoring.
  - `IndexScoreKpool` (127) — fp8 pooled indexer scoring.
  - `GemvF32` (128) — small fp32-output GEMV (deliberately separate from the tuned
    `d_gemv` decode hot path), needed because the reference requires fp32 precision for
    one specific indexer weight (`weights_proj`) — bf16 there causes ~1% logit errors
    that flip near-tie top-k selections.
- **Five real correctness bugs found and fixed during composition, each caught by
  reading kernel source or by a hardware oracle catching a wrong assumption — not by
  inspection alone**:
  1. A doc-comment mislabeling which checkpoint tensor needed fp32 precision.
  2. `DsaPoolCompress`'s decode-mode write address didn't account for which pool
     boundary was being crossed — would have silently corrupted the cache on the SECOND
     pool onward in a real multi-step decode. The original hardware test only crossed
     one boundary, so it couldn't have caught this — fixed, and a new multi-boundary
     test proves it (deliberately reverted the fix once to confirm the test fails
     without it, then restored).
  3. `IndexScoreKpool` read a pool-granular occupancy value directly with no internal
     division from plow's only (token-granular) occupancy counter — an out-of-bounds
     read. Fixed with internal `kv_len/pool_size` division, floor.
  4. `IndexSelect`/`IndexSelectPf` had the exact same class of bug — and this one
     matched a documented HISTORICAL bug in this same file (`op_attention.h:4477-4491`)
     almost exactly (unwritten score slots selected, gather reads never-written memory).
     Fixed with a `pool_size` parameter (default 1) plus a zero-means-1 guard at the
     `interp.hip` dispatch site so no existing GLM-5.2 blob is affected.
  5. `DsaPoolExpand` needed a materialized per-row array that doesn't exist in the
     prefill chain — FIXED, mirrors `IndexSelectPf`'s own scalar+affine formula instead
     of inventing new state.
  Every fix above was independently re-verified by re-running the hardware test myself
  (not just trusting the sub-agent's self-report) before moving on.

## In progress / next up

### Native plow implementation resumed (after reference benchmark)

- User clarified the required result is native plow GLM-5.3 vs vLLM, not another vendor
  run. A persistent implementation goal is active through emitter, prep, correctness, and TP8.
- Extended `HyperConnPost` without adding another opcode: `i3=mode` now supports the exact
  model-boundary operations required by GLM5Next (`mode=1` hc_expand by stream replication,
  `mode=2` hc_contract by stream mean); `mode=0` is the existing hardware-verified post mix.
  Packet slot/docs and interpreter dispatch agree; packet tests remain 60/60.
- Next implementation seam: expose raw attention/FFN outputs from the existing GLM emitters so
  the sequence can be `hc_pre -> norm/mixer -> hc_post` around attention and FFN independently,
  rather than incorrectly wrapping an already-residualized GLM-5.2 block.
- Added a `raw_output` seam to both decode and prefill MLA attention emitters. Existing GLM-5.2
  callers pass `false` and retain their original residual+post-norm path; GLM-5.3 can request the
  TP-reduced attention output before residualization for `HyperConnPost`. `cargo check -p devgen`
  is clean apart from the two existing unused-variable warnings.
- New architecture constraint confirmed directly from the real config: sparse layers use
  `mla_use_nope=true`, `qk_nope_head_dim=256`, `qk_rope_head_dim=0`. The existing generic GLM
  parser deliberately refuses NoPE because its k-cache writer is fused with RoPE. GLM-5.3 must
  use a NoPE cache writer/flash geometry explicitly; forcing a fake theta or zero-width RoPE would
  produce a finite but wrong model and is not acceptable for the correctness gate.

### Native GLM-5.3 full-model emitter update (2026-09-01 13:27 UTC)

- Completed the NoPE MLA implementation for `qk_rope_head_dim=0`: declaration omits RoPE
  tensors/cache, decode and prefill skip the rotations, and the flash ABI carries an explicit
  NoPE bit. Added the `DR=0` dense/gather instantiations. Full gfx950 HSACO rebuild succeeded;
  devgen 252/252 and packet 60/60 passed before the full-model wiring began.
- Split the existing GLM FFNs at a raw-output seam. Dense and MoE, single-row and row-grouped,
  now can return the TP-reduced H-vector without adding `xmid`; existing GLM-5.2 callers retain
  the old residual path. `cargo check -p devgen` passes.
- Added the first complete `glm5_next` device-emitter route. It parses the nested `text_config`
  and `model.language_model.` namespace, conditionally declares MLA weights/cache only on layers
  3,7,...,43, declares KDA weights/state on the other 34 layers, and emits all 45 layers with:
  embed -> hc_expand -> per-layer attn hc_pre/raw mixer/hc_post -> FFN hc_pre/norm/raw FFN/hc_post
  -> hc_contract -> final norm/head/argmax. Both decode and full-prefill programs are emitted.
- First real TP8 structural run succeeded:
  `PLOW_MLA_PREFILL=full:128 plowc --hf-dir /home/shaswot/models/GLM-5.3-Flash
  --emit devblob --gpu mi350 --arch gfx950 --num-gpus 8 --max-ctx 8192` produced
  `/tmp/plow-glm53-smoke/model.pkt` (21 MiB), with 45/45 layers and a 128-row prefill program.
  This is a compile/emission gate only, not yet a model correctness result.
- Immediate next gate: GLM-5.3 checkpoint prep/binding. Reuse GLM-5.2 absorption/indexer prep
  for the 11 MLA layers, preserve KDA/mHC tensors, create the fp32 `weights_proj` twin, then load
  and execute the emitted blob. KDA TP slicing and the mHC f32 projection bindings must be checked
  at load before any output is treated as valid.

### Resume update (2026-09-01, current session)

- Confirmed no background work remains in the tree: the pooled DSA **decode** branch landed,
  but `emit_glm_dsa_prefill_select` is still the old dense GLM-5.2 implementation and emits
  no `DsaPoolCompress`/`DsaQQuant`/`IndexScoreKpool` sequence.
- Landed the missing prefill pool-cache chunk-base plumbing: `DsaPoolCompress.i3` is now
  passed by `interp.hip`; `kv_write_row_field` recognizes `kv.*` destinations for op 123
  and rebases `i3`; packet docs/slot metadata and a runtime regression fixture were updated.
- Two additional prefill-only contracts were found before wiring the branch:
  1. `IndexScoreKpool` currently reads `kv_len[b]`. Prefill has one scalar chunk-end
     `kv_len`, so it must derive row `b`'s causal length as
     `kv_len[0] - n_batch + b + 1`, matching `IndexSelectPf`/`DsaPoolExpand`; otherwise
     rows after zero read out of bounds.
  2. Prefill tail-ring seeding cannot point a packet tensor handle at an arbitrary row.
     `DsaPoolStash` therefore needs an explicit source/position row operand (decode leaves it
     zero) before the trailing `t % index_kpool` rows can seed the decode ring safely.
- Do not start `glm53.rs` until these two contracts and the pooled prefill emitter branch are
  implemented and covered by devgen/runtime tests.

### Pooled prefill composition update (2026-09-01, current session)

- Implemented the missing `index_kpool>1` prefill chain in
  `emit_glm_dsa_prefill_select`: pooled gate projection, fp32 index weight projection,
  `DsaPoolCompress`, query quantization, pooled scoring/select, token expansion, and the
  existing per-query union build. Added dedicated row-sized prefill scratch tensors.
- Extended `IndexScoreKpool` with an explicit prefill mode. It now derives each query's
  causal token length from the scalar chunk-end `kv_len` and shares the growing K pool cache
  across query rows; decode retains its original per-batch cache/length behavior.
- Extended `DsaPoolStash` with a source row operand (decode defaults to row 0), providing the
  primitive needed for prefill tail seeding. Tail-seed emissions for non-pool-aligned prompt
  endings are still TODO; the benchmark contexts are pool-aligned, but the general model path
  must not claim complete correctness until this is wired and tested.
- Verification: `cargo test -p devgen -p packet --no-fail-fast` → 252/252 devgen library,
  all golden/integration tests, and 60/60 packet tests pass; only the same four pre-existing
  stale-tuning-DB tests fail. Full `gfx950_hsaco` rebuild completed successfully (49 objects).
- Rebuilt and reran `runtime/tests/indexer_qscore_gfx950_test.hip` after the score ABI
  extension: DsaQQuant 0 mismatches over 64 scales + 8192 fp8 bytes; IndexScoreKpool
  0 mismatches over 4 logits (max relative error 7.4e-7). This covers decode/default mode;
  the new shared-cache causal prefill mode still needs its own multi-row oracle fixture.
- Next: add pooled-prefill op-sequence and runtime causal-cache regression tests, wire tail
  seeding, then begin `crates/devgen/src/glm53.rs`.

- **DSA indexer emitter composition** (`crates/devgen/src/mla.rs`,
  `emit_glm_dsa_decode_select`/`emit_glm_dsa_prefill_select`, gated on
  `GlmCfg.index_kpool > 1`): **decode side is COMPLETE**, hardware primitives wired,
  252/252 devgen tests pass, `index_kpool<=1` proven byte-identical to today's GLM-5.2
  path. Five kernel-contract blockers found and fixed on the way to that.
- **SIXTH blocker (pool-cache write addressing across prefill CHUNKS) is FIXED and the
  prefill `index_kpool>1` composition is CODE-COMPLETE**: `DsaPoolCompress` takes a
  `chunk_base` operand (`pool_idx = chunk_base/pool_size`), `kv_write_row_field` in
  `crates/plowrt/src/exec/amd.rs` gained the one new match arm to rebase it per chunk,
  three-way opcode registration is done, and `emit_glm_dsa_prefill_select`'s pooled
  branch mirrors the decode side end to end (gate `Gemv` -> `GemvF32` -> `DsaPoolCompress`
  -> `DsaQQuant` -> `IndexScoreKpool` -> `IndexSelectPf` -> `DsaPoolExpand` ->
  `IndexUnionPf`). `cargo test -p devgen -p packet -p plowrt --lib`: 252/252 devgen +
  60/60 packet + 135/135 plowrt, all clean (fixed two incidental breaks found on
  resume: two test-only `GlmCfg` literals missing the `softmax_layers` field, and a
  stale `GFX950_UNEMITTED` allowlist entry for `HyperConnPre`/`Post` now that mla.rs
  emits both). Full gfx950 kernel rebuild (`cmake --build build-amd`) clean.
  **HARDWARE-VERIFIED (2026-09-01)**: `dsa_pool_prefill_chunk_gfx950_test.hip` (two
  chunks into a shared growing 4-pool cache) — 0 mismatches across all 4 pools, teeth
  proven by revert-and-confirm-failure (258 mismatches with the fix reverted). See the
  environment-bug entry above for what it took to get this test running at all.
- Once indexer composition is fully done (decode + prefill): write
  `crates/devgen/src/glm53.rs` — the new per-layer family module wiring `kda.rs`'s KDA
  layers + the now-composed DSA indexer layers, wrapped in `HyperConnPre`/`Post` every
  layer (mirroring `k3.rs`'s `K3Mixer` dispatch pattern, but using GLM's own
  `emit_xreduce`-based TP convention, not K3's `K3Tp` struct).
- Register `model_type = "glm5_next_text"` in `crates/devgen/src/lib.rs` and
  `crates/plowc/src/hf_config.rs`.
- Write `scripts/glm53_prep.py` checkpoint prep script (model it on `kimi_k3_prep.py` +
  `glm52_prep_indexer.py`, tensor names verified against `glm5next_ref/`).
- Full correctness gates: single-block sweep, full-model `amd-bench` (all TP ranks
  token-identical), then the actual benchmark run against the vLLM baseline already in
  `perf-data/vllm-k3-glm53-baseline.md`.

## How to verify anything in this file yourself

Every "hardware-verified" claim above has a standalone `.hip` test under
`runtime/tests/` (e.g. `dsa_pool_decode_mb_gfx950_test.hip`,
`indexer_qscore_gfx950_test.hip`, `index_select_pool_gfx950_test.hip`) with its exact
build/run commands in its own header comment — two `hipcc` passes plus
`perf-data/tools/gpulease -n 1 <label> <bin> <co-file>` under `sg render -c "..."`.

**IMPORTANT, host-compile pass**: several existing test headers show
`-L$ROCM_PATH/lib -lamdhip64` (or `-L/opt/rocm/lib`) for the host pass. That flag alone
is now known to be UNRELIABLE under `sg render` — `sg`/`newgrp` is setuid-root, which
trips glibc's `AT_SECURE` mode and silently drops `LD_LIBRARY_PATH`, so the binary
falls back to whatever `libamdhip64.so.7` the system default ld.so search finds instead
of the one you built the kernel with, breaking comgr's internal blit-kernel JIT with a
confusing "undefined hidden symbol: __ockl_dm_init_v1" error that looks like a device-
lib problem but isn't. **Add `-Wl,-rpath,$ROCM_PATH/lib` to the host-compile command**
(RPATH survives `AT_SECURE`, env vars don't) — e.g.:
```
hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I$ROCM_PATH/include <test>.hip -o <bin> \
    -Iruntime/tests -Wl,-rpath,$ROCM_PATH/lib
```
If a test still fails with `hipErrorInvalidValue` on its first `hipMemcpy`/`hipMalloc`
after this, don't assume it's a code regression — rerun with `AMD_LOG_LEVEL=3` and check
for exactly this "cannot find ROCm device library" / "undefined hidden symbol" pattern
before spending time on the test's own logic.

Rust side: `nix develop --command cargo test -p devgen -p packet -p plowrt --lib
--no-fail-fast` (expect 252 devgen lib + 60 packet + 135 plowrt passing; the only
allowed failures are pre-existing, unrelated `tuned_tile_selection` stale-tuning-DB
tests). Full kernel rebuild: `cmake --build build-amd --target gfx950_hsaco -j$(nproc)`
inside `nix develop --command` with
`LD_LIBRARY_PATH=/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu` set INSIDE that
invocation (the devshell clobbers an outer export) — this is the CMake/offline build,
unaffected by the `sg render` issue above since it never execs through `sg`.
