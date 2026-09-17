# Gemma-4-12B dense realtime tracker

Updated: 2026-09-17
Branch: `gemma4-12b-mi300x-vllm-rungs-v2`
Checkpoint: `google/gemma-4-12B-it` rev `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
Plan: "Dense throughput — missing fused kernels + tile-specific GEMM"
Companion: `plans/gemma4-4k-8k-campaign-tracker.md` (prefill TTFT rung campaign)

## Goal

Beat vLLM 0.28 on Gemma-4-12B **realtime** workloads: C1–C4 TTFT *and* TPOT.
Sequencing (fixed): **B0 → A1 → A2+B3 → B1+B2**.

Realtime is dominated by decode, not prefill: at C1/in4096 vLLM spends 170 ms on
TTFT and 1363 ms generating 128 tokens at 10.64 ms/tok. A TTFT win that leaves
TPOT untouched moves ~11% of the request.

## Blocker: no target hardware

The work host is **1× H100 80GB HBM3** (driver 595.91.07, CUDA 13.2). There is no
MI300X, no ROCm runtime, and `perf-data/tools/gpulease` is local-only. The nix
flake can *build* gfx942 objects (TheRock ROCm 7.14) but nothing can execute
them here.

Consequence: **every gfx942 T2/T3/T4 gate in this plan is unrunnable on this
host**, B0 included. Per campaign rule "never transfer a performance result
between architectures", no gfx942 cell may be filled from an H100 run.

Open decision for the owner: retarget this campaign to H100 SM90a (hardware is
present, vLLM reference is measured on the same box), or keep authoring gfx942
code blind and gate it on an MI300X host later.

## Reference: vLLM 0.28, this H100, gemma-4-12B-it BF16 TP1

| Cell | TTFT ms | TPOT ms | Out tok/s |
|---|---:|---:|---:|
| C1 in128 | 28.19 | 10.55 | 94.8 |
| C1 in1024 | 46.71 | 10.62 | 94.1 |
| C1 in4096 | 170.17 | 10.64 | 94.0 |
| C4 in1024 | 128.44 | 11.16 | 331.0 |
| C16 in1024 | 423.91 | 13.76 | 939.6 |

Source: `perf-data/vllm-028-h100-three-model-baseline.md`.

## Plow recorded, same H100, same model

| Precision | 4K TTFT | 8K TTFT | TPOT |
|---|---:|---:|---:|
| BF16 | 221.6 ms | 467.0 ms | **no record** |
| W8A8/FP8 | 164.9 ms | 358.7 ms | **no record** |

Prefill p50, three-seed mean, protocol `g4-4k8k-v1`. At 4K, Plow BF16 is 1.30×
vLLM BF16; Plow FP8 is at rough parity but is a different precision and is not
an apples-to-apples win.

**There is no Plow TPOT or output-tok/s number for 12B anywhere on this branch.**
The realtime goal is currently unmeasured on the Plow side. Closing that is the
prerequisite for every ranking decision below — a fusion priced only against
prefill cannot be ranked for realtime.

## A0 audit — result

Measured, not estimated: `rewrite::fused_sites_for_config` on the exact
checkpoint config, B=1 S=8192.

| Fused kind | Sites | Lowered? | Device opcode | State |
|---|---:|---|---|---|
| `FusedNormResidualScaleNorm` | 144 | yes (`NormResidualNorm`) | `DevOp::NormResidualNorm` (23) | covered |
| `FusedNormResidualNorm` | 96 | yes (`NormResidualNorm`) | `DevOp::NormResidualNorm` (23) | covered |
| `SwiGLU` | 96 | **no** | `DevOp::GemmGlu` (20) / `GemvGlu` (19) + fp8 twins | **kernel exists, hand-fused by the emitter** |
| `FusedNormRope` | 80 | **no** | `DevOp::HeadNormRope` (3) / `HeadNormRopeFp8` (37) | **kernel exists, hand-fused by the emitter** |
| `FusedNormLinear` | 4 | **no** | fused `GemvQkv` path | kernel exists |
| `FusedEmbeddingScale` | 1 | **no** | — | unpriced, 1 site |

421 sites, 6 kinds. 240 (57%) lowered; 181 unlowered.

### What this changes about the plan

1. **The kernels already exist.** For Gemma-4-12B every fused kind the rewrite
   finds has a device opcode today. The "28 kinds discovered, 2 lowered" figure
   is a **lowering-table** gap, not a kernel gap, and it counts kinds that
   Gemma-4-12B never produces (Conv3d/GroupNorm/AdaLN/MLA/KDA belong to other
   architectures). No new fused-op kernel needs authoring for this model.

2. **The unlowered bulk is already hand-fused.** `SwiGLU` (96) and
   `FusedNormRope` (80) are 176 of the 181 unlowered sites, and the emitter
   already routes both by hand — `lib.rs:5496` "Prefill fuses too, via the GEMM
   epilogue (`DevOp::GemmGlu`)", and the `HeadNormRope`/NRF decode fold. So the
   unlowered count is **not** a measured loss; lowering them mostly moves an
   existing decision from the emitter into the rewrite table. Price before
   building, per A0's own rule.

3. **A2's named targets do not exist on this model.** `FusedRmsNormSiluGate`,
   `FusedResidualNorm` and `FusedResidual3Norm` produce **zero** sites on
   Gemma-4-12B. Of the two shipped lowerings only `NormResidualNorm` ever fires
   here; `AddNorm` is dead on this model. A2 should be re-scoped to `SwiGLU` +
   `FusedNormRope` or dropped for Gemma.

4. **A1 cannot be driven by the rewrite here.** `FusedLinearAct` /
   `FusedLinearBiasAct` produce zero sites on Gemma-4-12B, so A1 has to be a
   direct emitter decision, not a lowering.

5. **A1's activation is GeGLU, not SiLU.** `dev.rs:116` — "Gemma is GeGLU
   (gelu_tanh), not SwiGLU". The unverified Tensile `activationType` integer to
   settle is the **gelu_tanh** one for this model.

6. **On 12B the Tensile epilogue has no consumer.** The hipBLASLt route
   (`GemmLtPf`, `lib.rs:4153`) is gated on 31B geometry (hidden 5376, inter
   21504, N=5376 output projections). 12B (hidden 3840, inter 15360) routes its
   output projections through Plow-native `gemma4_gemm_glu_gfx942.hip`
   (`b9f85333`), whose epilogue already carries GeGLU. A1 as written is a
   **31B** item; for 12B, epilogue fusion means the native body's epilogue,
   and bias/scale there is a kernel edit, not a kernarg write.

## B0 — result

The "ALL dense-GEMM tile(s) chosen by the ANALYTICAL MODEL" state had two
causes, neither of them a missing campaign:

1. **The probe could not find `hipcc`.** `kernelcaps/src/probe.rs` cleared the
   environment and set `PATH=/usr/local/cuda/bin:/opt/rocm/bin:/usr/bin:/bin`,
   which predates the flake shipping ROCm from the nix store. On a nix-only
   host every AMD probe failed, no digest could be computed, and zero records
   loaded — reported as tier `portable`, byte-identical to "never measured".
   `plowc tune status --gpu MI300X` could not derive an inventory at all.
   Fixed: the compiler is resolved on the inherited PATH before the env is
   cleared; `HIP_DEVICE_LIB_PATH` is forwarded.
2. **Every one of the 8007 records is stale against the current source.**
   With the probe working, the store keys to `gfx942-437e258b24c15856` and
   all 18 measured digests differ in `implementation`+`interpreter` (14 of
   them also in `oracle`, 4 also in `toolchain`). The strict rule dropped all
   of them. `PLOW_TUNE_IGNORE_DIGEST` (default on) parks digest-mismatched
   records and fills only op cases with no current-digest record; exact
   records are never displaced and a parked case is taken whole.

Gemma-4-12B gfx942 emit, `wide` profile, same source, CPU-only:

| Knob | Tunedb verdict | Packet |
|---|---|---|
| `PLOW_TUNE_IGNORE_DIGEST=0` (strict) | ALL 3552 tile(s) ANALYTICAL | `9e1ce3b1…` |
| default (relaxed) | **275 op cases filled; all 3936 tile(s) BY MEASUREMENT** | `17b942dd…` |

Caveats, stated plainly: this is a structural result with **no hardware
certificate** — no MI300X was reachable. The relaxed path also ignores the
`oracle` digest, so 14 of 18 measured builds were certified by a weaker
correctness oracle than the current one; the T2 numerics gate on hardware is
not optional before any serving promotion. Re-running the campaign against
the current family (`scripts/rebench_tune_gemm_gfx942.sh`) retires both
caveats and is still the B0 deliverable.

## A1 — Tensile epilogue word map (static, no hardware)

Derived from the AMDGPU kernel metadata of the nix ROCm 7.14 hipBLASLt library
the pinned specs are extracted from (`clang-offload-bundler --unbundle` of
`TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_…_Alik_Bljk_…_gfx942.co`, then
`llvm-readelf --notes`). Validated against plow's existing `dstD` write at
byte 140 (`amd_gemm_lt.rs:356`).

The 5 bias specs (`…_BBS_BH_Bias_HA_S_SAV_UserArgs_…`, kernarg segment 160 B):

| Byte | `Args` field | Tensile kernarg |
|---:|---|---|
| 0 | `dims[0]` | `Gemm info` — bits 31:30 select the argument mode; plow's `1` → mode 0 = **inline** kernargs |
| 4..28 | `dims[1..7]` | `kernel info0/1`, `numWG`, `SizesFree0..2`, `SizesSum0` |
| 32..56 | `pointers[0..3]` | `D`, `C`, `A`, `B` |
| 64..92 | `strides[0..7]` | `strideD0/1`, `strideC0/1`, `strideA0/1`, `strideB0/1` |
| 96, 100 | `alpha`, `beta` | f32 |
| 104 | `epilogue[0..1]` | `AddressScaleAlphaVec` (f32 ptr, per-column alpha) |
| 112 | `epilogue[2..3]` | `bias` (ptr) |
| 120 | `epilogue[4]` | `biasType` (u32 enum) |
| 124 | `epilogue[5]` | `StrideBias` |
| 128 | `epilogue[6]` | `activationAlpha` (f32) |
| 132 | `epilogue[7]` | `activationBeta` (f32) |
| 136 | `epilogue[8]` | `activationType` (u32 enum) |
| 140 | `epilogue[9..10]` | `dstD` (bf16 ptr) — the one word plow writes |
| 148 | `epilogue[11..12]` | `Synchronizer` (ptr) |
| 156 | `epilogue[13]` | `GSUSync` |

The 3 no-bias specs (`…_BBS_BH_UserArgs_…`, `AFC0`) end at `beta`: kernarg
segment **104 B**, no epilogue words at all. Plow's trailing 56 bytes are
unread padding there, and the output goes through `D`. **A1 can attach only to
the five bias specs.** The bias kernels are `AFC1` — the activation is a
called function with its own symbol, not inlined in the kernel body.

**`activationType` is settled statically, from the kernel's own dispatch.**
`llvm-objdump` of the pinned `MT128x96x128` kernel: the inline-args path
advances the kernarg base by 16 (`s_add_u32 s0, s0, 16`), then at
`label_Load_Bias_End` compares the u32 at byte 136 with
`3 → Gelu, 5 → Relu, 6 → Sigmoid, 11 → Silu, 13 → Clamp`, else None. The Gelu
entry is the tanh approximation (`v_mul_f32 v8, 0x3d372713 (=0.044715), …`),
which is Gemma's `gelu_tanh`. So for A1: **`epilogue[8] = 3`** for GeGLU
(`11` for SiLU models). `hipblaslt.h` only carries the API-level
`HIPBLASLT_EPILOGUE_*` enum (`GELU = 32`), which is not this value.

Still unresolved, and not to be guessed: the `biasType` integer for a bf16
bias (Tensile `DataType` enum). The bias load path is visible
(`buffer_load_short_d16` + `v_lshlrev_b32 16`, i.e. bf16→f32), but which
`biasType` selects it needs either the same disassembly pass on the bias
cascade or a 1-GPU T2 (`scripts/tensile_args.py`). Not needed for an
activation-only or scale-only epilogue.

What A1 buys, stated plainly: Tensile has no *gated* epilogue. On the gate
projection it can apply `gelu_tanh(gate)` in-register, but `act(gate) * up`
still needs the multiply pass; the saving is the activation's read+write, not
the whole GLU. And on 12B nothing routes through Tensile (finding 6), so A1 is
a 31B item until 12B projections are re-routed to `GemmLtPf`.

## H100 realtime baseline — run notes

- Packet: BF16 sm90a, `--max-ctx 8192`, `PLOW_MAX_CHUNK=4096`, decode ladder
  `1` only, so it fits beside a co-tenant. Another user's idle `plowrt`
  (0% util) holds 52.9 GB outside the lease, so `gpulease` stamps every run
  "contended"; the full ladder packet (1..16 slots, 41 GiB KV) cannot load
  until it is gone. **C4/C16 cells need the co-tenant stopped.**
- The first runs failed the coherence gate with `111.111…` output. Not a
  packet fault: plowrt's `/v1/completions` does not prepend BOS (vLLM's
  does), and Gemma degenerates without it. Chat completions and an explicit
  `<bos>` prompt are correct. Fixed on the bench side with a chat-formatted
  `GATE_PROMPT`; the random-token workload differs from vLLM's by that one
  BOS token of input.
- `PLOW_UNISEG=1` (from the 31B FP8-KV recipe) is not needed here; the emit
  audit's "impure flash segment" warning is an AMD relaunch concern and does
  not affect the NVIDIA cooperative launch. Both packets serve correctly.

### C1 result — provisional (run `20260917T114127Z`, commit `99508b16`)

Plow BF16 sm90a, ladder 1, chunk 4096, qualified H100 roles **off**, single-object
prefill (`segmented=false`), co-tenant idle. vLLM 0.28 same box, same client
protocol. 32 requests + 16 warmups per cell, out 128.

| in | Plow TTFT | vLLM BF16 | vLLM FP8 | Plow TPOT | vLLM BF16 | vLLM FP8 | Plow tok/s | vLLM BF16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 46.98 | 28.19 | 28.15 | 12.07 | 10.55 | 7.15 | 81.0 | 94.8 |
| 1024 | 164.72 | 46.71 | 37.83 | 12.62 | 10.62 | 7.23 | 72.4 | 94.1 |
| 4096 | 654.74 | 170.17 | 134.34 | 13.19 | 10.64 | 7.25 | 54.9 | 94.0 |

Read plainly:

- **Decode is 1.14–1.24× behind vLLM BF16** (12.1–13.2 vs 10.6 ms/tok) and
  1.7–1.8× behind vLLM FP8. The BF16 weight stream is ~22 GiB/step, a ~7 ms
  floor at 3.35 TB/s; vLLM BF16 sits at ~67% of it, this packet at ~55%.
- **Prefill is the large gap**: ~6.3k tok/s vs vLLM's ~24k, and 655 ms at 4K is
  ~3× the branch's own recorded 221.6 ms. This packet lacks the H100 production
  recipe (segmented native prefill objects, `--tma-gemm`, the qualified
  `PLOW_GEMMA4_SM90_*_ROLE` roles). The recipe, not a kernel, is the first
  lever; re-run with it before any prefill kernel work.
- ITL median 0.0 / p99 ≈ 4 × TPOT: multistep delivers tokens in quanta of 4 at
  C1–C2. Throughput-neutral, but a 4-token stutter for streaming; price
  `--multistep` for the realtime profile.
- Provisional: gpulease stamped the run contended (idle co-tenant), ladder-1
  packet. C4/C16 need the co-tenant gone.

## Workstream status

| Item | State | Evidence / blocker |
|---|---|---|
| Merge `glm53-8k-ttft` | **done** | `044ee494`; 593 + 534 tests pass |
| A0 audit | **done** | table above |
| Probe PATH fix | **done** | `kernelcaps/src/probe.rs`; `tune status` sees 8007 records; 73 tests pass |
| B0 tile tuning | **structurally unblocked; hardware gate pending** | 275 cases / 3936 lookups by measurement; campaign re-run needs an MI300X |
| A1 GEMM epilogue | **ready to author, gate blocked** | premise verified: `amd_gemm_lt.rs:347` `epilogue: [u32; 14]`, only `[9]/[10]` (dstD) written; 12 words zero. Bias-capable specs already pinned (`gemma_lt_bias_gfx942.json` 5, `gemma_lt_nobias_gfx942.json` 3). `activationType` value needs a 1-GPU gfx942 T2 (`scripts/tensile_args.py`) |
| A2 glue fusions | **re-scope** | see finding 3 |
| B3 fused GLU/QKV | **partly already shipped** | `GemmGlu`/`GemvGlu` fuse gate+up+act today |
| B1 skinny GEMM | not started | last by sequencing |
| B2 coverage widening | not started | last by sequencing |
| Plow realtime baseline | **missing, highest priority** | no TPOT/tok-s record for 12B |

## Next actions

1. Resolve the hardware decision above. It gates the B0 campaign re-run and
   every gfx942 T2/T3/T4.
2. Independent of that: record a Plow C1/C4 TTFT+TPOT baseline for 12B against
   the vLLM table, so realtime has a control. Run under `gpulease -n 1`.
3. A1: author the epilogue fields behind `PLOW_GEMM_EPILOGUE_ACT`, defaulted
   off, with the gelu_tanh `activationType` left as the one unresolved constant;
   gate on an MI300X host.
4. On the first MI300X session: `plowc tune status --gpu MI300X`, then the B0
   campaign, then a T2 numerics pass on the relaxed-tile packet before anything
   is served from it.

## Mergeability

Checked against `origin/main` after every commit: `git merge-tree` is clean,
the branch is ahead only (fast-forward). The probe fix and the tuner knob are
each self-contained and mergeable on their own.

## Protocol

Unchanged from `plans/gemma4-4k-8k-campaign-tracker.md`: one variable per
candidate, `gpulease -n 1` on every GPU run (the box is shared by several
agents), T1 emit byte-identity + checkpoint S, T2 numerics, T3 rung, T4 served,
`perf-certs/<id>.json` before any default flip. No cross-architecture transfer.
