# Qwen3.5 Gated DeltaNet on gfx942 (MI300X)

Status as of 2026-09-08. Ten of the eleven Qwen3.5 opcodes now have AMD arms, oracle-checked on
gfx942; the eleventh is a host-side dispatch with no AMD adapter. `plowc` still refuses a Qwen3.5
emit for an AMD target — by name, listing exactly what is missing.

The refusal's most useful content is which gap is actually binding: **a decode-only Qwen3.5 packet's
opcodes are now all dispatched on AMD**, so decode is blocked on porting the *emitter*, not on
writing another kernel. That is asserted by a test, not by reading the code.

## The op inventory

Every opcode a Qwen3.5 emit declares, and where each is implemented. The "AMD before" column is the
state this work started from: **zero** cases in `runtime/amd/interp.hip` for the whole family.

| op  | `PLOW_DOP_*`               | `DevOp`             | NVIDIA (sm_90a/sm_120)              | AMD before | AMD now |
| --- | -------------------------- | ------------------- | ----------------------------------- | ---------- | ------- |
| 136 | `QWEN_GDN_CONV`            | `QwenGdnConv`       | `op_qwen_gdn.cuh:11`                | —          | arm     |
| 137 | `QWEN_GDN_STEP`            | `QwenGdnStep`       | `op_qwen_gdn.cuh:39`                | —          | arm     |
| 138 | `QWEN_GATED_NORM`          | `QwenGatedNorm`     | `op_qwen_gdn.cuh:115`               | —          | arm     |
| 139 | `QWEN_Q_GATE_SPLIT`        | `QwenQGateSplit`    | `op_qwen_gdn.cuh:135`               | —          | arm     |
| 140 | `QWEN_SIGMOID_GATE`        | `QwenSigmoidGate`   | `op_qwen_gdn.cuh:146`               | —          | arm     |
| 141 | `QWEN_RMSNORM`             | `QwenRmsNorm`       | `op_qwen_gdn.cuh:156`               | —          | arm     |
| 142 | `QWEN_HEADNORM_ROPE`       | `QwenHeadNormRope`  | `op_qwen_gdn.cuh:174`               | —          | arm     |
| 143 | `QWEN_GDN_CONV_PREFILL`    | `QwenGdnConvPrefill`| `op_qwen_gdn.cuh:213`               | —          | arm     |
| 144 | `QWEN_GDN_QKV_PREP`        | `QwenGdnQkvPrep`    | `op_qwen_gdn.cuh:238`               | —          | arm     |
| 145 | `QWEN_GDN_GATE_PREP`       | `QwenGdnGatePrep`   | `op_qwen_gdn.cuh:264`               | —          | arm     |
| 146 | `QWEN_GDN_PREFILL`         | `QwenGdnPrefill`    | **host-side**, `gdn_prefill.cpp`    | —          | **still refused** |

The rest of a Qwen3.5 program (`Gemm*`, `Gemv*`, `Glu`, `Embed`, `Residual`, `QuantFp8`,
`FlashPrefill`/`FlashDecode`/`FlashMerge` at HD256, `Argmax`/`ArgmaxFin`) is already dispatched on
gfx950 — those opcodes are in `GFX950_DISPATCHED` and reach the existing dense arms. The Qwen family
was the whole gap.

### Why op 146 is different in kind

It is not an interpreter arm on *either* backend. sm_90a dispatches it host-side into a generated
CuTe DSL kernel: `runtime/nvidia/gdn_prefill.cpp` (`plow_gdn_create` / `plow_gdn_run`), driven from
`crates/plowrt/src/exec/gpu.rs` (the `QwenGdnPrefill` adapter around line 6207). That kernel is
hard-wired to 48 heads and a 128x128 state and to sm_90a. An AMD prefill therefore needs a HIP
chunked body **and** a `plowrt` adapter, not a `case` label — clearing the ten interpreter arms does
not clear this one, which is why the refusal keeps naming it separately.

## The refusal

`plowc` refuses a Qwen3.5 emit for any AMD target before the emitter does any work.
`refuse_unimplemented_target` (`crates/devgen/src/qwen35.rs`) reports each gap as a named
capability. The three are not the same shape, and the order matters:

1. `qwen35_gdn_amd` — the interpreter arms. **CLOSED.** Derived live from `GFX950_DISPATCHED`, so
   it reopens by itself if an arm is ever dropped.
2. `qwen35_amd_emit` — **the binding constraint.** `qwen35::run` is NVIDIA-shaped throughout: it
   pins the NVIDIA tile inventory (`EmitAmdGuard::set(false)`), builds the manifest for `sm_90a`,
   and asserts an external paired CUDA interpreter. Nothing routes a packet to the new arms yet.
3. `qwen35_gdn_prefill_amd` — the host-side chunked prefill. **Does not block (2).**

That last point is the useful finding here, and it is machine-checked rather than asserted: only a
*prefill* program emits op 146 (the `self.prefill` branch of `Emitter::gdn`), and a decode-only
model — which is what `model()` builds whenever `PLOW_QWEN_PREFILL` is unset, and how Kimi-K3 spent
its whole AMD bring-up — emits `QwenGdnStep` instead.
`qwen35::tests::a_decode_only_model_emits_no_prefill_op` builds that model and asserts both that it
carries no op 146 **and** that every opcode it does carry is in `GFX950_DISPATCHED`. So a
decode-only AMD Qwen3.5 blob needs no new kernel at all — only the emitter port.

This replaced `assert_eq!(arch, "sm_90a", …)`, which stopped the same builds but named neither the
gap nor its size. Pinned by `qwen35::tests::amd_targets_are_refused_by_name`.

A second, independent gate is on the loader. The family is compiled OUT of every shipping object
(`PLOW_QWEN_GDN` defaults to 0, exactly as `PLOW_K3` does), so an object that does not advertise
`plow_qwen_gdn_arms_1` is refused a packet carrying any of the ten ops —
`check_qwen_gdn_arms` in `crates/plowrt/src/exec/amd.rs`. Without it the arms' absence would be
silent: AMD's dispatch `default:` is `/* PLOW_DOP_NOP */` and writes nothing.

## What was implemented, and how it maps onto KDA

`runtime/amd/op_qwen_gdn.h`. Ported body-for-body from `runtime/nvidia/op_qwen_gdn.cuh`, which is
the authoritative semantics — packet slots, rounding points and order of operations are that file's.

**Gated DeltaNet and KDA are the same recurrence with one difference: the gate's rank.**

| | KDA (`op_kda.h`) | Gated DeltaNet |
| --- | --- | --- |
| forget gate | per (head, key-channel) — a vector of length D, carried as a per-chunk log2-domain prefix | **scalar** per (token, head) |
| delta rule | `(I - beta k kᵀ)`, k L2-normalized | identical |
| q/k normalization | L2, eps inside the sqrt | identical |
| state | `[H, V, K]` f32 in HBM, V-first | identical, plus a slot axis |
| ownership | one workgroup per (head, V-tile), no cross-workgroup handoff | one **wave** per (slot, head, V-row) for decode |
| chunked prefill | prepare / intra / wu / carry, four packets | one packet, host-dispatched |

This is not an analogy: in FLA's own kernels (`chunk_delta_h.py`, `fused_recurrent.py`) the two are
literally one body under a `USE_GK` flag — `b_h *= exp(b_gk)[None, :]` for KDA against
`b_h *= exp(b_g)` for GDN. The consequence for the prefill work still outstanding is stated in the
header of `op_qwen_gdn.h`: the chunked arm should be `d_kda_chunk_carry_bt64` fed a **broadcast**
gate prefix, not a second recurrence framework.

The decode step is the one place the two genuinely diverge in shape. KDA's `d_kda_state_step_t`
owns a whole (head, D) state tile per wave because its gate is per-channel; GDN's gate is scalar, so
a wave can own a single V row of the V-first state and the state write needs no barrier at all. That
is the NVIDIA arm's decomposition too, and it is kept.

### What the 64-lane wave changed

Loop bounds, in every body but one: `PL = dim / PLOW_WAVE` is a template parameter with a rung
dispatch (64/128/192/256), so the bodies stay generic over geometry and an unsupported geometry
traps instead of running short. The NVIDIA arms hard-code `kdim == 128` and `dim == 256` because a
32-lane warp covers them in exactly 4 and 8 registers.

The exception is `d_qwen_headnorm_rope`. The NVIDIA body rotates the pair `(x[0], x[1])`, which at
warp 32 is elements `(d, d + 32)` of a 64-wide rotary section held in the **same lane**. At wave 64
register 0 already spans `[0, 64)`, so the partner of lane `l` is lane `l ^ 32` and the pair must be
exchanged across lanes with `__shfl_xor(_, half, 64)`. This is the transliteration
`amd_common.h:1075` warns is not available, and it is why both the flat and the ring-scatter forms
are covered separately by the oracle.

Transcendentals are the precise ones (`expf`/`log1pf`, not `__expf`): the bodies are memory bound,
and the gate feeds a multiplicative recurrence over a persistent state where relative error
compounds across steps.

## Oracle errors

`runtime/tests/qwen_gdn_gfx942_test.hip` — an f64 host oracle per body, implementing the spec rather
than mirroring the kernel (the state is indexed `[slot][head][v][k]` independently, because with
`vdim == kdim` a transposed state has exactly the right norm and no magnitude check finds it).
Geometry is Qwen3.5's: 16 key heads, 48 value heads, `dk = dv = 128`, conv width 4, HD256 with a
64-wide partial rotary. Grid 37 divides no extent, so a ragged `nblk` is exercised. Slot 1 is
`active = 0` in every decode-shaped case, so an arm that ignores the mask fails.

Measured on gfx942 (`gfx942:sramecc+:xnack-`), ROCm 7.14.0 nix toolchain:

```
  gdn_conv out           rms rel 1.643e-03 (bar 4e-03)   max rel 2.835e-03 (bar 4e-03)
  gdn_conv history       rms rel 0.000e+00 (bar 1e-12)   max rel 0.000e+00 (bar 1e-12)
  gdn_step out           rms rel 1.645e-03 (bar 6e-03)   max rel 2.441e-03 (bar 2e-02)
  gdn_step state         rms rel 3.335e-08 (bar 1e-05)   max rel 9.774e-08 (bar 1e-04)
  gated_norm             rms rel 1.636e-03 (bar 4e-03)   max rel 2.272e-03 (bar 8e-03)
  q_gate_split q         rms rel 0.000e+00 (bar 1e-12)   max rel 0.000e+00 (bar 1e-12)
  q_gate_split gate      rms rel 0.000e+00 (bar 1e-12)   max rel 0.000e+00 (bar 1e-12)
  sigmoid_gate           rms rel 1.664e-03 (bar 4e-03)   max rel 2.222e-03 (bar 8e-03)
  qwen_rmsnorm           rms rel 1.593e-03 (bar 4e-03)   max rel 3.511e-03 (bar 8e-03)
  headnorm_rope flat     rms rel 8.013e-04 (bar 4e-03)   max rel 3.057e-03 (bar 8e-03)
  headnorm_rope ring     rms rel 7.802e-04 (bar 4e-03)   max rel 3.217e-03 (bar 8e-03)
  gdn_conv_prefill out   rms rel 1.642e-03 (bar 4e-03)   max rel 2.966e-03 (bar 4e-03)
  gdn_conv_prefill hist  rms rel 0.000e+00 (bar 1e-12)   max rel 0.000e+00 (bar 1e-12)
  qkv_prep q             rms rel 1.739e-03 (bar 4e-03)   max rel 2.943e-03 (bar 8e-03)
  qkv_prep k             rms rel 1.734e-03 (bar 4e-03)   max rel 2.890e-03 (bar 8e-03)
  qkv_prep v             rms rel 0.000e+00 (bar 1e-12)   max rel 0.000e+00 (bar 1e-12)
  gate_prep alpha        rms rel 2.274e-08 (bar 1e-06)   max rel 5.875e-08 (bar 1e-05)
  gate_prep beta         rms rel 0.000e+00 (bar 1e-12)   max rel 0.000e+00 (bar 1e-12)
PASS (worst 0.741)
```

Reading these: 1.6e-3 rms on a bf16 output is the bf16 quantum (2⁻⁹·³), i.e. the arms are exact to
the storage format. The three exact-zero rows are the pure gathers and the bf16-rounded gate, which
must be bit-identical and are. `gdn_step state` at 3e-8 is the f32 state after a full decode step —
the recurrence itself, at f32 round-off. `gate_prep alpha` at 2e-8 is the transcendental's own error
against f64.

Build and run:

```
hipcc --offload-arch=gfx942 -O3 -w -DQGDN_DEVICE --genco \
    runtime/tests/qwen_gdn_gfx942_test.hip -o /tmp/qgdn.co -Iruntime/amd -Iruntime/common
hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm-7.2.4/include \
    runtime/tests/qwen_gdn_gfx942_test.hip -o /tmp/qgdn -L/opt/rocm-7.2.4/lib -lamdhip64
perf-data/tools/gpulease -n 1 qgdn '/tmp/qgdn /tmp/qgdn.co'
```

The device half is also a CMake target (`qwen_gdn_gfx942_test`, `runtime/bench/CMakeLists.txt`),
which runs it through `hipcc_hsaco.sh`'s register gate. `k_step` lands at **VGPR=68, occ 7, zero
spill, zero LDS** at `PL = 2` (128-wide key) — the decode recurrence is comfortably inside the
budget even at eight waves.

Note the host half must be compiled **outside** `nix develop`: the nix ROCm toolchain links against
a newer glibc than this host's `libstdc++` provides and the link fails on `arc4random`. The device
half must be compiled **inside** it.

## Provenance

Nothing was copied verbatim, so no licence header travelled. What was read and what it settled:

* `runtime/nvidia/op_qwen_gdn.cuh` — in-tree, the authoritative semantics. The AMD bodies match it
  slot for slot and rounding point for rounding point.
* vLLM 0.28+rocm723's FLA port,
  `vllm/third_party/flash_linear_attention/ops/{chunk,chunk_delta_h,wy_fast,solve_tril,cumsum,chunk_o,fused_recurrent}.py`
  (Apache-2.0 as redistributed by vLLM, over MIT upstream FLA, © 2023-2025 Songlin Yang, Yu Zhang).
  This is what actually runs Qwen3-Next on MI300X today — `_resolve_gdn_prefill_backend()` returns
  `"triton"` unconditionally off CUDA — and is the reference the chunked prefill must match. It also
  supplied the `USE_GK` observation that makes GDN and KDA one algorithm.
* AITER (`/workspace/aiter`, MIT, © AMD) —
  `aiter/ops/triton/_triton_kernels/gated_delta_rule/` has an AMD-tuned fork of the same pipeline
  with two extra fusions, and `csrc/kernels/chunk_gated_delta_rule_fwd_h.cu` is a hand-written
  `__gfx942__`-guarded MFMA state-carry kernel (`BT=64, K=V=128, BV=16, 256 threads`). vLLM 0.28
  wires AITER in for **decode only**; the AITER chunked prefill is present and unrouted. That file
  is the strongest starting point for the prefill work below.
* `runtime/amd/op_kda.h` and `op_kda_carry_regstate.h` — the in-tree chunked linear-attention
  family, and the structure the prefill arm should reuse.

## What still refuses, and what it would take

### Decode: the emitter port (`qwen35_amd_emit`)

The shortest route to a working AMD Qwen3.5, and it needs no new kernel — the decode-only packet's
opcodes are all dispatched on gfx950 today, which the test above asserts rather than assumes. What
`qwen35::run` needs:

* `EmitAmdGuard::set(target_is_amd(arch, gpu))` instead of the hard `false`, so `pick_tile` ranks
  the AMD inventory rather than the NVIDIA one.
* The `assert_eq!(arch, "sm_90a")` relaxed, and `manifest::build` given the gfx arch.
* An AMD interpreter object in place of the `embed_cubin`/`embed_hsaco` assertion in `lib.rs`
  (`"Qwen uses an external paired CUDA interpreter"`), plus a `PLOW_QWEN_GDN=1` object row in
  `scripts/build_gfx942.sh` and its symbol audit entry.
* `check_gfx950_opcode_coverage(&m, amd)` called on the built model, so the generic per-packet gate
  takes over from the static one.
* The production-default and decode-object gates in `emit_capabilities` /
  `apply_production_defaults`, which are `arch == "sm_90a"`-conditioned.

**This was not attempted** because there is no Qwen3.5 checkpoint on this host: `validate_coverage`
reads the checkpoint's tensor list, so `plowc` cannot be run end to end and the resulting blob could
not be shown to be right. Flipping `AMD_QWEN_EMIT` blind is exactly the silently-wrong blob the
refusal exists to prevent.

### Prefill: `PLOW_DOP_QWEN_GDN_PREFILL`. Two pieces:

1. A HIP chunked Gated DeltaNet body. The cheapest correct route is `d_kda_chunk_carry_bt64` with a
   broadcast (rank-1) gate prefix in place of KDA's per-channel one, plus the `prepare`/`intra`/`wu`
   stages it already has — the maths is identical once the gate's rank is fixed. The alternative is
   a port of AITER's `chunk_gated_delta_rule_fwd_h.cu`, which is already gfx942-native but is a
   state-carry kernel only and would still need the other four stages.
2. A `plowrt` adapter. The sm_90a path is a host-side `plow_gdn_run` call, so the AMD side either
   grows the same shape of adapter or the emit is restructured to lower op 146 into the four
   KDA-shaped packets. The second is more in this tree's idiom but is a devgen change, not a kernel
   one.

Until both are done the emit refuses. Decode is blocked only on the emitter port above; prefill is
blocked on both.

Also outstanding, and deliberately not attempted here:

* **No object row.** `PLOW_QWEN_GDN` defaults to 0 and no row in `scripts/build_gfx942.sh` or
  `build_gfx950.sh` sets it, because nothing routes to the arms yet and an arm merely *present* is
  paid for in the register budget (`interp.hip:118-150` records a +32% decode regression from
  exactly that). The family costs 41 KB of object when enabled (662 KB vs 621 KB on a decode-shaped
  build). Add the row and its symbol audit entry together with the prefill adapter.
* **No checkpoint.** There is no Qwen3.5 checkpoint under `/workspace/models` on this host, so no
  emit and no smoke serve were attempted. Everything above is validated against synthetic shapes
  derived from the geometry `crates/nn-graph/src/models/config/qwen3_5.rs` parses.
* **No performance number.** Every claim here is correctness. The bodies have not been benchmarked
  and no `kx` experiment has been added.
