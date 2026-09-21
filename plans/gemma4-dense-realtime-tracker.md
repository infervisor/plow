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

### C1 result with the qualified H100 prefill roles — provisional

Same protocol, same co-tenant caveat. Emit: `PLOW_SEG_PURE_GEMM=1
PLOW_SEG_FA512=1 PLOW_SEG_FA256_GQA2=1 PLOW_TMA_GEMM=1` + the three
`PLOW_GEMMA4_SM90_*_ROLE=1` + `--segmented`; objects from
`build_sm90a_gemma4_segments.sh` with `PLOW_CUBIN_CONFIG`,
`PLOW_BUILD_FA_GQA2_PAIR=1`, `PLOW_BUILD_PFATTN_HD256_BKV32=1`,
`PLOW_BUILD_PFATTN_HD256_GQA2_BKV32=1`, `PLOW_BUILD_PFATTN_HD512_PX4_BQ64=1`
(px4 direct entry 128 registers, zero stack/spills); serve mirrors
`PLOW_PF_SEG_DIR`, `PLOW_PF_SEG_PURE=1`, `PLOW_PF_SEG_FA512=1`. The role
objects are looked up in the emit `--out` dir, so the order is base emit →
objects → role emit.

| in | Plow TTFT | vs plain packet | vLLM BF16 | vLLM FP8 | Plow TPOT | vLLM BF16 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 42.23 | 46.98 | 28.19 | 28.15 | 12.00 | 10.55 |
| 1024 | 89.75 | 164.72 | 46.71 | 37.83 | 12.62 | 10.62 |
| 4096 | 213.58 | 654.74 | 170.17 | 134.34 | 13.22 | 10.64 |

The recipe reproduces the campaign's recorded 221.6 ms at 4K. Prefill is now
1.25× vLLM BF16 at 4K and 1.5–1.9× at ≤1K (small-M under-occupancy: the emit
audit shows 15–60 GEMM tiles over 132 SMs at M≤1024). Decode is unchanged —
the roles are prefill-only — and remains the realtime gap.

### C1 result, FP8 PTPC weights — provisional, compared to vLLM FP8 only

Plow packet: `PLOW_FP8=1 PLOW_W8A8=1` with the same role recipe
(`PLOW_SEG_PURE_GEMM=fp8`, explicit `PLOW_EMIT_PACKED_PREFILL=1`, W8A8 GemmGlu
role), composed checkpoint = BF16 shards + the campaign's `fp8/`-keyed PTPC
twins (`plow-h100-campaign/gemma12b-w8a16/checkpoint-mixed`, 656 tensors).
Per the 31B bring-up doc this packet is **W8A8 prefill, W8A16 decode, BF16 KV,
BF16 lm_head**; vLLM's FP8 is W8A8 in both phases. Same weight bytes per decode
step, not the same arithmetic. Never table this row against a BF16 engine.

| in | Plow TTFT | vLLM FP8 | × | Plow TPOT | vLLM FP8 | × | Plow tok/s | vLLM FP8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 42.11 | 28.15 | 1.50 | 9.18 | 7.15 | 1.28 | 106.0 | 136.7 |
| 1024 | 70.38 | 37.83 | 1.86 | 9.81 | 7.23 | 1.36 | 97.2 | 133.9 |
| 4096 | 174.69 | 134.34 | 1.30 | 10.40 | 7.25 | 1.43 | 85.6 | 121.3 |

Read plainly: halving the weight stream took Plow's decode from 12.0 to 9.2
ms at in128, but vLLM's FP8 runs at 7.2, so **relative to its own reference the
FP8 packet is further behind (1.28–1.43×) than the BF16 packet is behind vLLM
BF16 (1.14–1.24×)**. The BF16-activation decode arm (W8A16) does more work per
step than a W8A8 arm for the same bytes, and TPOT grows with context (9.18 →
10.40 across 128 → 4096 tokens of KV) where vLLM FP8 is flat (7.15 → 7.25):
the decode attention over BF16 KV is a visible second term. The like-for-like
FP8 comparison needs Plow's W8A8 decode arm, which is not enabled.

Same-precision scoreboard (C1, TPOT):

| Pair | in128 | in1024 | in4096 |
|---|---:|---:|---:|
| Plow BF16 / vLLM BF16 | 1.14 | 1.19 | 1.24 |
| Plow FP8-weight / vLLM FP8 | 1.28 | 1.36 | 1.43 |

### D4 — streaming granularity (harness A/B, one variable)

`scripts/campaign/campaign.py bench … --env PLOW_MULTISTEP=0` vs default (8,
mux quantum 4 at C1). BF16 roles packet, C1, provisional (co-tenant).

| in | TPOT default | TPOT ms=0 | ITL med / p99 default | ITL med / p99 ms=0 |
|---:|---:|---:|---:|---:|
| 128 | 12.00 | 12.06 | 0.00 / 48.48 | 12.14 / 12.23 |
| 1024 | 12.63 | 12.69 | 0.00 / 50.57 | 12.69 / 12.76 |
| 4096 | 13.22 | 13.28 | 0.00 / 53.00 | 13.27 / 13.59 |

Per-token streaming for +0.06 ms/tok (0.5 %). For a realtime profile
`--multistep 0` is the right default; the throughput profile keeps 8. Ledger:
`perf-data/campaign/gemma4-12b.h100.bf16.csv`.

### D1 — occupancy-2 decode object (in progress)

Object: `build_sm90a_cubin.sh` with `PLOW_EXTRA_DEFINES="-DPLOW_NV_FORCE_MINBLK=2
-DGV_UNROLL=4"`, decode-only, bound to the packet config (`interp_sm90a.cu`
includes `interp_sm120.cu`, so the sm120 MINBLK hook applies). Result:
**128 registers, 176-byte stack frame** (bundled occ-1 object: 194 regs, no
stack) — the plan's spill risk is real and must be priced, not assumed away.
Loads at `grid=264, occ_per_sm=2`.

Runtime contracts met so far: `interpreter grid 132 != packet n_cu 264` (occ-1
object with a 264 packet — the documented pair), and `prefill grid 132 !=
decode grid 264 — ordinary prefill launches must share the decode grid`
(occ-2 object with a plain packet). `n_cu` is one blob-header field that sizes
every program's slice table, so a per-phase width is a packet-format change,
not a knob. The exemption is a fully segmented prefill chain; on a base segmented packet
at `n_cu=264` the server loads (decode object `grid=264 occ_per_sm=2`) but the
first prefill launch dies: `cuLaunchCooperativeKernel:
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`. A packet's `n_cu` streams are one per
block, so every launch — the occ-1 prefill objects included — needs 264
co-resident blocks. **Conclusion: D1 requires per-program stream counts in the
blob (prefill 132, decode 264) plus per-program launch grids and checks; it is
a packet-format change, not a knob.** Estimated cost: packet writer/reader
version bump, `gpu.rs` launch/grid checks keyed per program, AMD reader kept
on the header value; gain per the plan −1.0…−1.6 ms/step, against a 176-byte
stack frame at 128 registers that must be priced on hardware.

### P2 — cuBLASLt prefill for the small-M projections (harness A/B)

`build --env PLOW_EMIT_PREFILL_CUBLASLT=1` on the BF16-roles recipe (808
projection segments routed; serve needs `LD_LIBRARY_PATH=/usr/local/cuda/lib64`
for `libcublasLt.so.13`). Control: `bf16-roles-ms0`.

| in | TTFT control | TTFT Lt | Δ | TPOT control | TPOT Lt |
|---:|---:|---:|---:|---:|---:|
| 128 | 42.12 | 42.03 | −0.2 % | 12.06 | 12.03 |
| 1024 | 89.70 | 88.05 | −1.8 % | 12.69 | 12.69 |
| 4096 | 213.20 | 205.22 | −3.7 % | 13.28 | 13.29 |

This is the library ceiling for those shapes, and it moves ≤1K by almost
nothing: **the ≤1K TTFT gap (42 vs 28 ms at in128) is not GEMM math.** At
in128 the prefill is a few ms; the rest is fixed per-request cost. The serve
log shows `max_hold_ms=8.0` (muxer batch-formation hold); `PLOW_IDLE_DISPATCH`
skips it when nothing else is queued — next A/B. P1 (native split-K) can at
best match these Lt numbers at ≤512 and is deprioritized below the fixed-cost
work.

### Attribution — where ≤1K prefill time goes (measured, cache off)

`PLOW_PF_SEG_TIME=1` per-segment event timing on the BF16-roles packet,
`PLOW_PREFIX_CACHE=0`, distinct prompts, `--multistep 0`. Streaming first token
measured with a plain client: **49.5 ms @159 tokens, 111 ms @1055**. The two
earlier knob A/Bs (`PLOW_IDLE_DISPATCH`, `PLOW_PF_SEG_GRAPH`, both null) and
the "≈14 ms non-GPU" hypothesis are retired: the time is GPU time.

| request | chunks (bucket) | GPU ms | GemmGlu | Gemm ×5/layer | Flash | Norm | Rope |
|---|---|---:|---:|---:|---:|---:|---:|
| 159 tok | 1 × 512 | 47.1 | 28.0 | 13.6 | 2.0 | 1.9 | 1.7 |
| 1055 tok | 1024 + **31-token tail on the 128 bucket** | 81.5 + 26.6 | 52.8 + 9.2 | 16.7 + 12.6 | 7.3 + 1.9 | 2.1 + 1.8 | 2.6 + 1.1 |

Findings:

1. **Bucket padding.** 159 tokens run as 512 (47 ms; the 128 bucket runs in
   26.6 ms). 1055 tokens pay a second full 26.6 ms chunk for 31 tokens. vLLM
   has no such quantization. Finer buckets (256; a small tail bucket, or
   folding a ≤N-token tail into the main chunk) are an emit-only change.
2. **The fused GLU role is the dominant cost at every bucket** — 191 µs /
   583 µs / 1.10 ms per layer at 128/512/1024, i.e. ~16–22 % of tensor-core
   peak (M=1024: 241 GFLOP in 1.1 ms). cuBLASLt would do the gate/up GEMM at
   ~0.35 ms; even with a separate GeGLU pass the 1024 chunk drops from ~82 to
   ~50 ms, next to vLLM's 46.7 ms TTFT. This is the kernel-level target for
   ≤1K prefill: a T2 of `interp_sm90a_pfgemm_glu_gemma4.cu` at M∈{128,512,1024}
   against cuBLASLt on the same shape.
3. Projection GEMMs cost 52–57 µs per launch regardless of M (240 per chunk):
   launch/ramp-bound. P2 (cuBLASLt) barely moves them, so the fix is fewer
   launches (fused QKV: −96/chunk) or a persistent small-M path, not a faster
   tile.
4. Non-GPU overhead is ~2.5 ms per request. Decode at C1 is unaffected by any
   of this.

### Lt gate/up instead of the fused GLU role — first attribution-driven win

Emit A/B on the BF16-roles recipe: `PLOW_NO_GLU_FUSE=1` (gate/up as two
N=15360 GEMMs) + `PLOW_EMIT_PREFILL_CUBLASLT=1` + `PLOW_GEMMA4_SM90_GEMM_GLU_ROLE=0`
→ 1096 Lt segments (288 more than P2: 48 layers × gate/up × the 3 wide
buckets). Control: `bf16-roles-ms0`.

| in | TTFT control | TTFT Lt-GLU | Δ | × vLLM BF16 (was) | TPOT |
|---:|---:|---:|---:|---:|---:|
| 128 | 42.10 | 42.19 | 0 | 1.50 (1.50) | 12.03 |
| 1024 | 89.90 | **56.39** | **−37 %** | **1.21** (1.92) | 12.71 |
| 4096 | 213.63 | 204.48 | −4.3 % | 1.20 (1.26) | 13.32 |

The fused GLU role (≈20 % of tensor-core peak) is beaten by cuBLASLt plus a
separate GeGLU pass at every bucket Lt covers, including 4096 where the role
had been qualified at −3.3 %. in128 is unchanged only because
`cublaslt_prefill_bf16` admits just `(N,K) ∈ {(3840,15360),(3840,8192)}` at
M ∈ {128,256,512}; admitting `(15360,3840)` there is the next one-line A/B.
The native fused-GLU body is now a **T2 target against cuBLASLt on its own
shape**, not a promoted role.

### Pipeline integration of the Lt-GLU finding (done)

- **Production default**: `apply_production_defaults` sets `emit.prefill_cublaslt`
  and `emit.no_glu_fuse` for Gemma-4 BF16 on sm_90a TP1 (`LT_GLU_DEFAULT`,
  status `LT_GLU_QUALIFIED`); `=0` rolls either back; W8A8 untouched.
- **Certificates**: `perf-certs/emit.prefill_cublaslt.json` and
  `perf-certs/emit.no_glu_fuse.json`, checkpoint P over ctrl / ctrl2 / treat /
  treat2 (32 requests each): accepted — TTFT@1024 89.97 → 56.56 ms (floor
  0.54), @4096 accepted, in128 and TPOT neutral within floor. Generated by
  `campaign.py cert` from the runs' per-request samples.
- **Policy**: `cublaslt_prefill_bf16` admits `(15360, 3840)` at M ≤ 512; in128
  unchanged (42.15), so the small rows cost nothing and keep one policy.
- **Exact-shape algorithms in the pipeline**: `plowrt --lt-algos` pins a
  per-shape `cublasLtMatmulAlgo_t` table after `AlgoCheck`, `--lt-algos-write`
  produces it; `campaign.py probe` (auto in `build` when the GPU family is
  present) writes it and the tune store
  (`tuning/nvidia/sm90a/h100/cublaslt_algos.jsonl`); plowc packetizes from the
  store when no GPU is present. treat2 ran pinned (30 shapes) and reproduced
  treat within noise (42.16 / 56.16 / 203.67).
- Remaining for FP8 parity: the W8A8 GLU role has no Lt A/B; and the native
  fused-GLU body is now a T2 target against cuBLASLt on its own shape.
- `scripts/perf_gate_ci.sh origin/main` accepts both new certificates. Its one
  remaining flag, `emit.emit_packed_prefill`, is the DCP case the glm53 merge
  added to `PACKED_PREFILL_DEFAULT` — inherited, not part of this work, and
  needs its own certificate from the GLM owners before this branch merges.
- Tuner-digest relaxation is now decided per lookup (`GemmMeasurements::lookup`),
  not when the process-wide cache is built; the C8 tests pin the strict rule.

### Protocol correction — prefix cache was on in every harness bench

`PLOW_PF_SEG_TIME=1` under the real client showed `prefix attached … rows=96
prompt=128` and a 39-row token-batch prefill. vLLM-bench's random prompts share
a **96-token prefix**, plowrt's prefix cache is on by default, and the vLLM
references ran with prefix caching disabled. Hits per cell (32 requests):

| cell | hits | cached rows | share of prompt |
|---|---:|---:|---:|
| in128 | 16 | 96 | 75 % |
| in1024 | 24–27 | 96 | 9 % |
| in4096 | 24–27 | 96 | 2 % |

Consequences: same-packet A/Bs stand (both arms cached alike); the in128 cell
was a 32-row prefill (~12 ms GPU) that still took 42 ms — so the "in128 gap" is
prefix-attach/VMM/host work, not prefill; and every absolute row against vLLM
must be re-measured with `PLOW_PREFIX_CACHE=0`. All vLLM-matched recipes now
set it; cache-off control and Lt runs are in flight and the certificate is
regenerated from cache-off arms. A realtime-chat profile may turn the cache
on, but only against a reference that also has it on.

### The ≤1K gap, resolved: re-tokenized random prompts cross bucket boundaries

Cache-off, same Lt packet, plain client, first streamed chunk:

| prompt | client tokens | first chunk |
|---|---:|---:|
| English, 128 tokens | 128 | **22.0 ms** (vLLM BF16 in128: 28.2) |
| vLLM-style random ids "128" | 142–150 | 29.8–30.0 ms |
| English, 744 tokens | 744 | 48.6 ms |
| vLLM-style random ids "1024" | 1176–1199 | 82.5 ms |

vLLM-bench decodes random token ids to text; that text re-tokenizes 12–17 %
longer, so the bench's "128" runs Plow's 512 bucket and its "1024" runs
1024 + a 128-token tail chunk, while vLLM pads nothing. On exact-length
prompts Plow's in128 prefill is already faster than vLLM's. Cache-off Lt
cells: 42.27 / 54.63 / 201.83 ms (ledger `bf16-ltglu-ms0-nocache`).

`PLOW_PF_LADDER_APPEND=256,1280` was **null** (42.22 / 54.54 / 201.09; `4608`
with `PLOW_MAX_CHUNK=8192` is refused by the packed-prefill ring rule), so the
bench's prompts are not bucket-quantized either. A request-field probe (same
128-token prompt with the bench's `repetition_penalty: 1.0`, `logprobs: null`,
`stream_options.include_usage`) is also null: 22.1 ms first chunk on every
variant, and **server-side TPOT 10.7 ms** — at vLLM BF16's 10.55 — where the
bench reports 12.0.

What remains is the vLLM bench client itself: its random-text prompts (≈30 ms
with a plain client) and its own per-request/per-chunk overhead account for
the 42 ms it reports, and vLLM pays the same client overhead under the shared
protocol. The bench stays the yardstick; the server-side numbers are the ones
kernel work should be priced against: **in128 TTFT 22 ms and TPOT 10.7 ms
(BF16, exact-length, cache off) vs vLLM's client-measured 28.2 / 10.55.**

### Full attribution of the Lt packet (cache off, C1) — where the 1024/4096 gap is

Lt GEMM time from the pinned algorithm table (per layer q+k+v+o+gate+up+down),
non-Lt segments from `PLOW_PF_SEG_TIME=1`, gaps = TTFT − both.

| cell | Lt GEMM | attention | norm+resid | GeGLU | rope | gaps | TTFT | vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 26.5 | 7.2 | 4.2 | 4.0 | 2.8 | ~10 | 54.5 | 46.7 |
| 4096 | 105.0 | 34.8 | 14.7 | 13.8 | 8.9 | ~24 | 201 | 170 |

- GEMM: cuBLASLt at 850 TFLOP/s at M=1024 — at the ceiling. Done.
- Attention: ≈8 ms floor at 4096 (2.2 TFLOP global hd512 + 2.7 TFLOP sliding
  hd256 at ~600 TFLOP/s) vs 34.8 measured — **~25 % of FA3-class**. The
  "attn kernels pick flash attention" ask maps here: rent FA3/cuDNN SDPA for
  the 40 hd256 sliding layers (both support hd ≤ 256, GQA, causal, window);
  hd512 global layers stay native (no library supports hd512).
- Light stages (norm+residual, GeGLU, rope): ≈13 ms bandwidth floor at 4096
  vs 37.4 measured — they run on the fat occ-1 object (194 regs). A lean
  light-ops object (norm/rope/GeGLU arms only, high occupancy) is the
  prefill form of D2 and is a defines-only build if the arm gating follows
  the FA-only object's pattern.
- Gaps: 243 kernel segments + 336 Lt calls per chunk. Both launch-mode A/Bs on
  this packet are NULL (cache off, C1, provisional/contended):
  `PLOW_PF_SEG_GRAPH=1` 53.72 / 201.47, `PLOW_PF_SEG_NONCOOP=1` 54.54 / 201.95
  vs control 54.63 / 201.83. Launch mode is not the lever; ~12 ms of the gap is
  the vLLM bench client itself (plain client in128 TTFT 22 ms).

Exact per-family cost of one 4096 chunk (`segment-site wall time`, 243 segments,
73.4 ms non-Lt):

| family | launches | µs each | total | floor |
|---|---:|---:|---:|---:|
| FlashPrefill | 48 | 742 | 35.6 ms | ~8 ms compute |
| NormResidual+RmsNorm | 96 | 153 | 14.7 ms | ~4.7 ms HBM |
| Glu | 48 | 291 | 14.0 ms | ~5.7 ms HBM |
| HeadNormRope | 48 | 186 | 8.9 ms | ~2.0 ms HBM |

FlashPrefill split (same chunk): 40 hd256 sliding launches at 353–386 µs
(~15 ms, ~2.2× an FA3-class time) and 8 hd512 global launches at
2450–2650 µs (~20.7 ms, 275 GFLOP each → 106 TFLOP/s, ~11 % of peak). The
hd512 px4/BQ64 role is the single largest non-GEMM item at 4096; no library
(FA2/FA3/cuDNN all cap at hd256) covers it, so it stays native. cuDNN 9 is on
the box and is the stable-ABI library path for the hd256 sliding layers
(causal + sliding window + GQA), the cuBLASLt analogue for attention.

Root cause in the bodies (not occupancy): the warp-per-row norms walked one
16-B chunk per lane per iteration (a 15-deep load-latency chain per pass at
feat 3840), rope handled one (token, head) per warp iteration with 8-B lane
accesses, and the GeGLU loop was not unrolled. Fix in flight (`op_norm.cuh`,
`op_elementwise.cuh`): 8-chunk batched row walks (same accumulation order →
bit-identical), 4-item rope passes with gamma hoisted (2 items at hd512),
`#pragma unroll 4` GeGLU. Cell `campaign-bf16-light` (same packet + Lt table,
objects rebuilt).

**Refuted (2026-09-17, cell `campaign-bf16-light`, provisional).** The wide
batches (8 chunks, 4 rope items) made every object wider: decode 194 → 255
regs + 416 B stack, fat prefill stack 1176 → 2016 B, GEMM-only 1128 → 2296 B.
TTFT 42.2 / 59.5 / 214.0 vs control 42.3 / 54.6 / 201.8; TPOT +0.4–0.9 ms.
Seg-time at 4096: NormResidual+RmsNorm 153 → 227 µs, HeadNormRope 186 → 242,
Glu 291 → 293 (unroll changed nothing). So the per-lane load-latency chain is
NOT what bounds these bodies at 1 block/SM, and any extra register pressure in
a light arm is paid by the whole megakernel. ncu cannot profile the cooperative
megakernel at all (fails to prepare the kernel even with counter-only
sections; see memory `ncu-on-nix-plowrt`).

Second attempt, guard-free (experiments/h100_elementwise_ab.cu already recorded
that a per-chunk guard on the load loops stops ptxas register-promoting the
bf16v8 arrays): batches cover whole 256-element chunks only (4 per pass), the
remainder takes the original loop; rope issues every item's loads from a valid
row and masks only the stores; GeGLU's vector loop is branch-free with a
separate tail. Compile gate (ptxas, same defines as the cell build):

| object | baseline | guarded 8-chunk | guard-free 4-chunk |
|---|---|---|---|
| decode | 194 regs, 0 stack | 255 regs, 416 B | 217 regs, 0 stack, 0 STL/LDL |
| fat prefill | 255, 1176 B, 1570 STL/LDL | 255, 2016 B | 255, 1568 B, 1736 STL/LDL |
| pfpackedseg (runs the light ops here) | 255, 1872 B, 5716 B spill st, 3539 STL/LDL | 255, 2648 B, 6990 B | 255, 1856 B, 3760 B, 2266 STL/LDL |

Cell `campaign-bf16-ladder` (full decode ladder 1,2,4,8,16 so the same packet
serves the C4/C16 throughput profile) is the measurement.

**Landed (v3, cell `campaign-bf16-ladder3`, realtime profile, cache off, GPU
free).** Norm batching kept; rope restructure reverted (it measured 186 → 202
µs); GeGLU rewritten as load-first pipelining (the unroll alone was null because
the uint4-punned store keeps the next loads behind it).

| family @4096 | baseline µs | v3 µs |
|---|---:|---:|
| NormResidual+RmsNorm | 153 | 119 |
| Glu | 291 | 210 |
| HeadNormRope | 186 | 187 |
| non-Lt per chunk | 73.4 ms | 65.8 ms |

TTFT 42.20 / 53.64 / 196.19 vs control 42.27 / 54.63 / 201.83; TPOT unchanged.
Decode object 194 regs / 0 stack (unchanged), pfpackedseg spills 5716 → 3492 B.

Operational lessons (2026-09-17 evening): (a) a cell build reads the runtime
sources per object as it goes — never start a build while a kernel patch is
mid-flight (the masked-padding 16K cell picked up the unclamped split-K and its
server hung at the first rung-8 admission; rebuild one object with the cell's
own `plow_config.h` and swap it); (b) detached queue scripts must wait on
process names spelled so their own command line does not match (`serv[e]`),
and a launcher's name must not contain the pattern it kills; (c) the harness's
low-memory guard kills tracked benches while a server pages in weights — run
benches detached and monitor their results file.

Harness debt (not fixed today): a cell build compiles the interpreter object
set twice (once inside the base emit, again in `build_sm90a_gemma4_segments.sh`
with the segment/role defines) and the second pass runs one nvcc at a time;
ptxas at the 255-register ceiling takes 1.5–3 min per object × 16 objects, so
the objects stage alone is ~25 min on one core. Parallelising the script and
skipping the duplicate base set would cut a build to under 10 min.

### The 128-token cell is launch-count bound (seg-time, 128 rows, one chunk)

| family | launches | µs each | total |
|---|---:|---:|---:|
| Gemm (pre-Lt small rows; now Lt) | 69 | 54 | 3.7 ms |
| NormResidual+RmsNorm | 76 | 37 | 2.8 ms |
| FlashPrefill | 38 | 40 | 1.5 ms |
| HeadNormRope | 38 | 22 | 0.8 ms |
| Glu | 38 | 21 | 0.8 ms |

The work per launch at 128 rows is microseconds (a 128×3840 norm moves 1 MB); the
Argmax segment (no work) costs 13 µs, so ~13 µs is the fat object's per-launch
floor and the light segments sit at 1.6–2.8× it — latency-bound (one row per SM,
one warp active). Lt at M=128: 119 µs per layer (3 shapes, 300–330 TFLOP/s) =
5.7 ms per chunk. Server-side plain-client TTFT 22 ms ≈ Lt 5.7 + fat 5.7 + FA 1.9
+ ~8 ms host/launch gaps. vLLM's 28.2 ms client-measured is ≈16 ms server-side.
Conclusion: at 128 rows nothing is bandwidth- or compute-bound; the levers are
(1) fewer launches (fused QKV = −96, fused gate|up = −48 Lt calls per chunk;
folding light ops into GEMM epilogues needs Plow's own small-M GEMM), (2) a
cheaper per-launch floor (light-only object: small smem, low regs), (3) the
~8 ms of host gaps (graph capture INCLUDING the Lt calls — seg-graph alone was
null; check whether the capture spans them).

### First throughput cells (2026-09-17, cell `campaign-bf16-ladder`, GPU free)

Full decode ladder 1,2,4,8,16 (`GV_MM_MAX 16` in plow_config.h), cache off,
profile `throughput` (multistep 8), vs vLLM 0.28 BF16:

| in | C | TTFT | vLLM | TPOT | vLLM | tok/s | vLLM |
|---|---|---:|---:|---:|---:|---:|---:|
| 1024 | 4 | 224 | 128 | 19.8 | 11.2 | 187 | 331 |
| 1024 | 16 | 1089 | 424 | 49.7 | 13.8 | 273 | 940 |
| 4096 | 4 | 932 | 468 | 23.3 | 12.2 | 130 | 254 |
| 4096 | 16 | 4883 | 1301 | 57.9 | 21.9 | 164 | 499 |

Reading: TTFT at C16/1024 (1089 ms ≈ 16 × 55 ms) says prefill throughput is
flat at ~19–20K tok/s whatever the packing (a 4096-row packed chunk costs 200 ms
= the same per-token rate as a single 1024 request), while vLLM's batched
prefill reaches ~40K tok/s (424 ms for 16K tokens). Plow's 4096-row chunk is
105 ms of Lt GEMM at the ceiling plus ~97 ms of attention (36), light bodies
(37) and gaps (24) — the same items as the C1 gap, amplified. TPOT growing
~linearly with C is at least partly decode stalling behind other requests'
prefill chunks (no prefill/decode mixing beyond the unified token batch's
2048-row budget); the in128 C4/C16 probe (`tp128`) separates batched-decode
efficiency from prefill stalls.

### Batched decode: the dot8 GEMV walk is compute-bound at B≥8 (2026-09-17)

`gemv_rows<MM>` streams each weight vector once, then issues MM activation loads
and 8·MM FMAs per 16-byte chunk on the CUDA cores. Standalone at the
interpreter's geometry (132×256, `experiments/gemv_mma_batch_h100.cu`, f32 CPU
reference), M=16: o_proj 221 GB/s, down 100 GB/s, gate|up 366 GB/s, fused qkv
282 GB/s. The tensor-core row-block walk (`op_gemv_mma.cuh`: one warp = 8 output
rows × 16 activation rows, mma.sync m16n8k16 with a virtual K order so fragments
load straight from global as 16-byte vectors) runs the same shapes at
1.4–2.6 TB/s: 6.6–14.2× at M=16, 4.1–8.8× at M=8, relL2 unchanged (1.7e-3).
Per layer at B=16: 2.18 ms → 0.22 ms; per step ~105 → ~11 ms of GEMV. This is
the throughput lever: TPOT at C16 should fall from 44.6 to ~14 ms (vLLM 13.8),
i.e. ~1100 tok/s vs vLLM's 940. Gate: `PLOW_NV_GEMV_MMA=1` (hooks in
`gemv_rows`/`gemv_glu_rows`/`gemv_qkv_rows` for MM≥8, K%32==0). Integration:
the manifest now emits it into plow_config.h whenever gv_mm_max ≥ 8 on sm_90a
(devgen manifest.rs; registered `def.PLOW_NV_GEMV_MMA[_UNB]`). Cell
`campaign-bf16-ladder4`: decode object 216 regs / 0 stack (was 226), 570
HMMA.16816 in its SASS. A/B: in128 C4/C16 (decode-only) and the throughput
profile at 1024/4096 C4/C16 against `campaign-bf16-ladder`.

**Measured (GPU free, cache off, throughput profile).** Hooks now cover the
4/8/16-row rungs (commits 0c46ee46, 8dfbaacd); the 2-row rung stays on dot8.

| cell | TPOT before → after | tok/s before → after | vLLM TPOT / tok/s |
|---|---:|---:|---:|
| 128/4 (decode-only) | 18.1 → 13.8 | 218 → 284 | 10.6 / – |
| 128/16 (decode-only) | 44.6 → 16.2 | 350 → 935 | 11.0 / – |
| 1024/4 | 19.8 → 15.4 | 187 → 238 | 11.2 / 331 |
| 1024/16 | 49.7 → 21.2 → 19.7 (split-K) | 273 → 540 → 570 | 13.8 / 940 |
| 4096/4 | 23.3 → 19.1 | 130 → 154 | 12.2 / 254 |
| 4096/16 | 57.9 → 38.8 → 37.3 (split-K) | 164 → 249 → 255 | 21.9 / 499 |

Remaining throughput gap, in order: (1) prefill interleaving — decode-only C16
is 16.2 ms but 21.2 with prefill, and TTFT queues behind ~20K tok/s prefill
(the C1 prefill items: hd512 attention, light bodies, launch count); (2) the
`down` shape at 1.4 TB/s (480 of 1056 warps busy: split K across warps);
(3) long-context decode attention (4096/16: 38.8 ms); (4) the 2-row rung.
Harness debt: the low-memory guard kills tracked background benches while a
server pages in weights — launch benches detached (job tmp `launch_tp_c4.sh`).

### Fused prefill norm pair (`PLOW_PF_GFUSE=1`, 2026-09-17, late)

Prefill emitted every sandwich norm as NormResidual + RmsNorm (96 launches per
chunk at 97–119 µs each) while decode already uses the fused
`NormResidualNorm`, whose rounding reproduces the split pair exactly (op_norm.cuh
"fused N1"). The emitter gates prefill fusion behind `PLOW_PF_GFUSE` (registered,
documented, off; `seam_fused` defaults to true when the rewrite graph has no
opinion). With it, every prefill program carries 44 `NormResidualNorm` sites and
the instruction count drops 766 → 670. **REFUTED, both reduction orders:**

| cell | TTFT 128 / 1024 / 4096 | greedy vs control | non-Lt seg total |
|---|---|---|---|
| fat-lite control | 42.05 / 51.15 / 191.77 | 5/5 identical | 61.0 ms |
| gfuse, `NRN_WPR=0` | 42.09 / 64.12 / 207.07 | 4/5 | 65.0 ms |
| gfuse, `NRN_WPR=1` | 42.18 / 63.22 / 205.31 | 4/5 | 61.6 ms |

Two lessons, both worth more than the experiment. **(1) The premise was wrong:
the split pair was ALREADY one launch.** The seg-time site label
`NormResidual+RmsNorm` means both ops share one segment, so op-level fusion
removed instructions, not launches (96 segments before and after). Read the
site labels before counting launches. **(2) The cost is outside the timed
segments.** At `NRN_WPR=1` the fused op is even marginally cheaper per launch
(92 vs 97 µs) and the non-Lt total is within noise, yet TTFT is 12 ms worse at
1024 and 13 ms at 4096. So something in the Lt path or the gaps regressed —
most likely the fused op changes which tensor feeds the next projection
(`d.t[1]/t[2] = n.xr` instead of `n.x`), making some projections miss the
pinned cuBLASLt algorithm. Worth one `PLOW_LT_ALGOS` hit-rate check before any
future emitter change that re-points projection inputs.

### Fat-lite light object at two blocks per SM (2026-09-17, late)

The light ops ride the packed seg object at the 255-register ceiling set by
arms they never execute here. `PLOW_NV_FATLITE=1` (T14: flash arms out,
128-register cap) already exists; with the batched light bodies it compiles to
128 regs / 24 B stack / 10 STL+LDL, and the loader's occupancy query launches
it at grid 264. Seg-time at 4096: NormResidual+RmsNorm 119 → 97 µs, HeadNormRope
187 → 106 µs, Glu 213 → 217 (unchanged), Flash unchanged: ≈ −4.8 ms per chunk,
bit-identical bodies. The recipes never set `PLOW_BUILD_FATLITE`, so no cell
had it. **Measured (paired C1, realtime, cache off):** 42.05 / 51.15 / 191.77
vs control 42.43 / 53.96 / 196.55 (−2.8 ms @1024, −4.8 ms @4096); all five
greedy continuations byte-identical. `PLOW_BUILD_FATLITE = "1"` is now in the
recipes' objects env. Combined WG32 + fat-lite cell (`campaign-bf16-wg32fl`):
42.07 / 50.42 / 186.13 ms (−3.5 ms @1024, −10.4 ms @4096 vs the paired control;
greedy exactly as WG32 alone). Realtime standing with both: 1.49× / 1.08× /
1.09× of vLLM's 28.2 / 46.7 / 170.2 (from 1.50× / 1.16× / 1.16× this morning's
control), TPOT unchanged at 1.13–1.25×. `PLOW_SEG_SLICE_ALL` is only recorded in the manifest, not
implemented; the occupancy query makes it unnecessary.

### hd512 WGMMA BQ64/BKV32 role — real but smaller than the plan's number (2026-09-17)

The plan file recorded the WGMMA BQ64/BKV32 hd512 body at 1.40 ms per launch
vs px4's 2.5 ms, rejected only for a greedy-checksum change. Rebuilt it
(`PLOW_NV_FA512_WG=1`, KV16/KV64 off, QK unroll 4, masked padding; 240 regs,
no spills) and emitted with the px4 knob off (the WG32 role is the default
selection when `interp_sm90a_pfattn_hd512.cubin` is present): paired C1,
realtime profile, cache off: 42.22 / 52.88 / 190.81 vs 42.43 / 53.96 / 196.55
(−1.1 ms @1024, −5.7 ms @4096). Numerics: 4 of 5 64-token greedy continuations
(815–3239-token prompts) byte-identical, 1 diverges at ~token 12 —
accumulation-order class. Shipped as the opt-in recipe
`gemma4-12b.h100.bf16-hd512wg32.toml`; making it the default is a numerics-
policy decision. Seg-time at 4096: hd512 launches 1.73–1.95 ms (px4
2.45–2.65), FlashPrefill 35.6 → 29.9 ms per chunk, non-Lt total 65.8 → 60.7 ms
— consistent with the −5.7 ms TTFT. Expected ~5 ms per launch at 8K KV (px4
7.3), so the long-context cells gain proportionally more.

### cuBLASLt at every rung — NULL (2026-09-17, paired A/B)

Admitting all eight Gemma-4 shapes at 128/256/512 rows (1288 → 1640 Lt
segments; plowrt must be rebuilt after any admission change) measured
42.13 / 53.53 / 195.17 vs the paired control 42.43 / 53.96 / 196.55 ms: null.
So the ~176 native GEMM launches per 512-row chunk were not the 128-token
cost; what remains there is the per-launch floor × 528 launches and the
host gaps. The policy is kept (simpler, not slower); the lever for 128 tokens
is launch COUNT (fused QKV / gate|up) or a cheaper launch, not the backend.

### Split-K for the narrow decode shapes (2026-09-17)

`down` (480 row blocks over 132 blocks) idled half the warps: 1.4 TB/s. Warp
pairs now share a row block and split K, reducing through static smem
(`op_gemv_mma.cuh`, gated on ≤ WARPS/2 row blocks per block). Harness M=16:
down 1412 → 1959 GB/s, o_proj 2156 → 2439; decode object 225 regs, no spills,
3.6 KB smem. Measured (in128, decode-only): C16 TPOT 16.2 → 14.65 ms (935 →
1030 tok/s; vLLM's 1024/16 aggregate is 940), C4 13.76 → 13.65. Decode-only
at C16 is now 14.65 vs vLLM's 11.0 ms.
Lesson recorded in the commit: the unclamped partition underflowed for trailing
blocks and hung the walk — synccheck showed no barrier fault, so an infinite
loop, not divergence.

### Long context and higher concurrency (2026-09-17, first cells)

New cells and matched vLLM references (same client, max-model-len 16384 /
max-num-seqs 16 for ctx16k; 8192 / 32 for C32; prefix caching off):

| cell | packet | TTFT Plow / vLLM | TPOT Plow / vLLM | tok/s Plow / vLLM |
|---|---|---:|---:|---:|
| 8192/4 | ctx16k, chunk 4096, ladder 16 | 1924 / 994 | 24.9 / 13.9 | 100 / 186 |
| 8192/16 | same | 8460 / 2300 | 73.2 / 35.7 | 114 / 298 |
| 15000/4 | same | 3904 / 1653 | 35.2 / 18.5 | 61 / 128 |
| 15000/16 | same, masked padding (`-mp` cell) | 19624 / 3948 (all 32 complete; without masked padding 22–23 were rejected) | 141.5 / 61.8 | 53 / 172 |

The 15000/16 failure is an admission rule, not shedding (re-run with
`PLOW_QUEUE_TTL_MS=0` failed identically): a packed launch pads its bucket and
charges the pad rows to one request's KV positions, and at 15000 + 128 of
16384 no request can absorb a multi-thousand-row pad, so the planner rejects
the request outright (`plow-asset packed_prefill.rs`). Masked padding (pad
rows carry slot −1, objects built with `PLOW_NV_MASKED_PADDING=1`) avoids it:
emit with `PLOW_MAX_REQUEST_CHUNK=4096` → cell `campaign-bf16-ctx16k-mp`.
| 1024/32 | c32, chunk 2048, ladder 32, roles OFF | 1590 / 852 | 27.3 / 17.5 | 807 / 1322 |

Long context is ~2× behind on every axis: prefill throughput is flat (~20K
tok/s, the C1 items) and long-context decode attention scales with KV.
Concurrency above 16 is gated by packet memory, not kernels: the sliding ring
is `next_pow2(window + chunk − 1)` (chunk 4096 → 8192 rows → ~3 GiB/slot), and
the HD512/HD256 roles need the 4096/8192 prefill rungs, so a 32-slot packet
had to drop to chunk 2048 with both roles off. Unblocking C32+ with roles
means either roles qualified at the 2048 rung or a ring decoupled from the
chunk. The 32-row rung itself is fine: checkpoint G now takes rows = 1 for
the NVIDIA staged arms (they stage only at M=1), and the multi-tile walk
holds 1.2–2.5 TB/s at M=32.

### Long-context prefill is the hd512 global attention role (seg-time, 8192-token prompt)

C1, 16K cell with masked padding, realtime profile: TTFT 432 ms for two
4096-row chunks. Second chunk (KV 4K→8K): FlashPrefill 69.5 ms of 101 ms, of
which the 8 hd512 global launches are 7.1–7.4 ms each (57 ms) and the 40 hd256
sliding launches ~0.39 ms (16 ms); norms/GeGLU/rope unchanged at 119/212/204 µs.
The hd512 launch goes 2.6 ms (4K KV) → 7.3 ms (8K KV): 1.1 TFLOP in 7.3 ms
= 150 TFLOP/s, ~15 % of peak; extrapolated ~15 ms at 12K and ~30 ms at 16K, so
a 15000-token prompt spends most of its 3.9 s (C4) / 19.6 s (C16) TTFT in this
one role. The px4 BQ64 role is mma.sync; the earlier hd512 campaign (plans/
gemma4-4k-8k-native-block.md) got 2–6 % steps on it, and the wgmma variants
(`PLOW_NV_FA512_WG`, KV16/KV64) lost to px4 at the 255-register ceiling. This is
the single largest item left for both realtime 4096 (20.7 ms) and every
long-context cell, and it needs a redesigned kernel, not tuning.

### plowrt VMM prefix review (merged 2026-09-17, branch `gemma4-plowrt-vmm-fixes`)

Reviewed the four reported issues; merged as 50f89816.
- A confirmed: auto-activation keyed to the 12B geometry; `PLOW_PREFIX_CACHE=0`
  already disables every VMM side effect. Decision now logged (`mode=auto|explicit|off`).
- B confirmed (worse): every prefill completion / slot recycle published a snapshot
  (~160 pitched copies, cudaMalloc, stream sync, evictions) at any hit rate. New
  `PLOW_VMM_PUBLISH_SHARED` (default on) publishes only leads seen on a second
  sequence. Needs an H100 A/B with cache ON (unique prompts vs multi-turn).
- C refuted on CUDA (sleep loops were AMD-only); condvar replaces them; CUDA prefix
  pool now honours `PLOW_VMM_DEFERRED_RECLAIM` (default on) — needs a begin_slot A/B.
- D refuted for the default config: packed prefill stays on with token batching on
  and fusion off; the log now names the knobs.

### Next pipeline step — ctx-keyed attention selection at emit

The attention roles are one object across all live-KV histories (tracker:
"Live-KV attention object switching: missing"). Extend the harness `probe`
to sweep KV buckets per query rung and geometry over the candidate objects
and `nsplit`, record to tunedb (`attention_role` kind), packetize a
`kv_bucket → (object, nsplit, geometry)` table per site, and select at launch
by live KV (packed metadata carries `kvlen`; prefix hits change the effective
bucket). T2 sweep is the selection gate; served C1/C4 remains promotion.

### Common ladder to 16k / C32 vs vLLM 0.28, and what the merged tree needed (2026-09-20)

Run from `worktree-gemma4-26b-beat-vllm` (this branch merged in). Recipe
`gemma4-12b.h100.bf16-ladder16k.toml`, packet `p12q`, `vllm bench serve` random,
32 prompts (C1/C4) and 64 (C16/C32), prefix cache off, same client both sides.
Ledger: `perf-data/campaign/gemma4-12b.h100.bf16-ladder16k.csv`.

Three things stood between the merged tree and a number:

1. **The realtime recipe does not load on this card.** `MAX_CHUNK=16384` with the 32-slot
   ladder needs 109,929 MiB: a sliding ring is `next_pow2(window + chunk - 1)` rows per slot.
   `ladder16k` keeps the realtime settings at chunk 4096 / 16 slots. C32 cells queue on 16 slots.
2. **`c1f1ac38` made GLU fusion ignore `PLOW_NO_GLU_FUSE`** (the rewrite lowering was asked
   first): `GemmGlu` at every rung, no object implements it under FATLITE, first prefill launch
   traps (`CUDA_ERROR_LAUNCH_FAILED`). Fixed in devgen (`0fa9b7e4`): the knob outranks the rewrite.
3. **`c1f1ac38`'s BF16 NRN fold decodes garbage on the H100**, and not the same garbage twice
   (first token right, then noise; greedy output differs between identical requests). It is
   dense-only (`!c.moe`), so the 26B never saw it. `PLOW_NO_FUSE_NRN=1` in the recipe; with it
   the gate prompt answers "The capital of France is Paris." The fold removes 96 serial NRN
   packets per token (540 -> 525 ops with QKV unfused), so it is worth root-causing — not done.

| in | C | TTFT plow / vLLM | TPOT plow / vLLM | tok/s plow / vLLM |
|---:|---:|---:|---:|---:|
| 128 | 1 | **18.7** / 28.2 | 14.38 / 10.55 | 69 / 95 |
| 1024 | 1 | 49.7 / 46.7 | 14.61 / 10.62 | 67 / 94 |
| 4096 | 1 | 185 / 170 | 14.78 / 10.64 | 62 / 94 |
| 8192 | 1 | 406 / 350 | 14.98 / 10.61 | 55 / 75 |
| 15000 | 1 | 870 / 675 | 15.32 / 10.62 | 46 / 63 |
| 128 | 4 | **46.0** / 52.2 | 14.79 / 10.58 | 266 / 367 |
| 1024 | 4 | 168 / 128 | 16.17 / 11.16 | 230 / 331 |
| 4096 | 4 | **349** / 468 | 19.46 / 12.16 | 181 / 254 |
| 8192 | 4 | **773** / 994 | 23.82 / 13.85 | 135 / 186 |
| 15000 | 4 | 1661 / 1653 | 31.03 / 18.51 | 91 / 128 |
| 128 | 16 | 166 / 101 | 20.49 / 11.01 | 713 / 1364 |
| 1024 | 16 | **362** / 424 | 26.46 / 13.76 | 542 / 940 |
| 4096 | 16 | **702** / 1300 | 42.98 / 21.94 | 330 / 499 |
| 8192 | 16 | **1696** / 2300 | 64.79 / 35.67 | 203 / 298 |
| 15000 | 16 | **3556** / 3948 | 105.7 / 61.80 | 118 / 172 |
| 128..15000 | 32 | 2.1–16.6 s / 0.2–6.4 s (queued on 16 slots) | 20.7–111.8 / 11.6–123.1 | 120–746 / 184–2442 |

Ahead on 10 of 60 metric-cells, every one a TTFT (plus two C32 TPOTs that are the 16-slot
cap, not a win). TPOT trails 1.36–1.96x everywhere — and 14.38 at C1 is itself a regression
from this branch's own 12.72 (`b4fix-v2-realtime`), which the decode work below more than recovers.

### Decode object: what a traced step showed, and three fixes (2026-09-20)

`PLOW_NV_TRACE` on `p12q` (block 0): B=1 is 83% GEMV body; the fused lm_head `GemvArgmax` is
10% of a B=1 step and **31% of a B=16 step (7.3 of 23.5 ms)** — `d_gemv_argmax_batch_inner`
is a dot8 walk, compute-bound past one row, on a 2 GB head.

* **Batched GemvArgmax on the tensor-core walk.** Logits come from `d_gemv` (the walk the
  other B>=2 GEMVs already take); softcap + argmax fold over the columns the same block wrote
  (`gvmma_partition`), after its barrier. Served greedy consistency solo vs 4- vs 16-wide is
  identical to the production object (14/16, same two near-tie flips).
* **Rung 4 was one real function call from a fault.** The `MM=4` GEMV templates called
  THEMSELVES in their misaligned-K fallback, so they could not inline and rung 4 alone ran as
  a call (352 B frame). Entry frame + 352 > 1 KiB device stack -> `ILLEGAL_ADDRESS`: production
  ran at 592+352, any object whose entry frame grew past ~670 B faulted on rung 4 only (the
  traced 26B object, the first version of this patch, the grouped-MoE 26B object). A `TC`
  template flag breaks the recursion; the dead dot8 fallbacks are out of line (`*_dot4`).
* **The decode entry's size taxes every rung.** It is one function at the 255-register cap.
  Manifest rules now keep out what the packet cannot use: `xreg_k` (B=1 xreg kernels only for
  the packet's K sizes) and, for a dense packet, `gemv_mma_b1` (B=1 GEMVs on the tensor-core
  walk too, which drops the classic B=1 kernels altogether).

step_bench ms/step, ctx 1024 (production `p12q` object -> each step):

| object | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|
| production | 14.46 | 13.55 | 15.80 | 17.24 | 23.35 |
| + argmax walk, recursion fix (fallbacks out of line) | 13.12 | 12.96 | 13.55 | 14.77 | 16.59 |
| + xreg K set (classic B=1) | 12.60 | 13.01 | 13.59 | – | – |
| + dense B=1 on the walk (`gemv_mma_b1`) | **11.93** | **12.60** | **13.16** | **14.35** | **16.03** |

### 12B ladder on the final packet `p12s` (2026-09-20, late)

Same ladder, packet rebuilt with the decode-object fixes (manifest: `gemv_mma_b1`, `xreg_k`):

| cell | TPOT before -> after (vLLM) | tok/s before -> after (vLLM) | TTFT after (vLLM) |
|---|---:|---:|---:|
| 128/C1 | 14.38 -> **11.75** (10.55) | 69 -> 85 (95) | **18.6** (28.2) |
| 4096/C1 | 14.78 -> 12.10 (10.64) | 62 -> 74 (94) | 187 (170) |
| 15000/C1 | 15.32 -> 12.58 (10.62) | 46 -> 52 (63) | 865 (675) |
| 128/C4 | 14.79 -> 12.17 (10.58) | 266 -> 322 (367) | **40.2** (52.2) |
| 4096/C4 | 19.46 -> 16.66 (12.16) | 181 -> 207 (254) | **360** (468) |
| 128/C16 | 20.49 -> **13.46** (11.01) | 713 -> **1075** (1364) | 136 (101) |
| 1024/C16 | 26.46 -> 19.47 (13.76) | 542 -> 720 (940) | **347** (424) |
| 4096/C16 | 42.98 -> 36.09 (21.94) | 330 -> 384 (499) | **702** (1300) |
| 15000/C16 | 105.7 -> 100.0 (61.80) | 118 -> 124 (172) | **3528** (3948) |

Ahead on 12 of 60 metric-cells (was 10). TPOT is 1.11–1.18x vLLM at C1 (was 1.36–1.44x),
1.15–1.52x at C4, 1.22–1.64x at C16; tok/s 1.12–1.39x through C16 (was 1.4–1.9x). The long-input
C16 TPOT is prefill stall, not decode (26B tracker, tick timeline): a 4096-row chunk is ~186 ms here.

### GEMV walk depth 12, packet `p12u` (2026-09-20, night)

`PLOW_NV_GEMV_MMA_UNB` 8 -> 12 (now the default): step_bench 11.93/13.16/16.03 -> 11.18/12.58/15.25
ms at B=1/4/16 (16: 11.52/12.93/15.77; K=3840 is 120 k32 steps, 12 divides it). Ladder:

| cell | TPOT p12s -> p12u (vLLM) | tok/s (vLLM) |
|---|---:|---:|
| 128/C1 | 11.75 -> **10.96** (10.55) | 90.7 (94.8) |
| 1024/C1 | 11.96 -> 11.20 (10.62) | 86.9 (94.1) |
| 15000/C1 | 12.58 -> 11.82 (10.62) | 53.9 (63.3) |
| 128/C4 | 12.17 -> 11.61 (10.58) | 337 (367) |
| 128/C16 | 13.46 -> 12.68 (11.01) | 1138 (1364) |
| 1024/C16 | 19.47 -> 18.82 (13.76) | 741 (940) |

TPOT is 1.04–1.11x vLLM at C1 (night start: 1.36–1.44x), 1.10–1.50x at C4, 1.15–1.61x at C16.
By bytes the B=1 step is ~85% bandwidth-efficient now (GemvGlu at ~89% of its roof); what is
left at C1 is the serial chain (540 ops, gate 19%). The NRN fold would shorten it but its arm is
in the staged dot8 path the 12B no longer takes (`gemv_mma_b1`), so it needs a walk-compatible
form as well as the H100 garbage root-caused. Negatives: attention row batching
`PLOW_NV_FA_WPR_RB` 4 / 8 (B=16 -2% / -0.6%, B=1 +0.8% / +3.6%).

### 32 slots at a 16k context: `gemma4-12b.h100.bf16-c32-16k`, packet `p12c32` (2026-09-20, night)

ladder16k with chunk 1024 (2048-row sliding ring), decode ladder to 32, attention roles off.

| cell | 16-slot `p12u` TTFT / tok/s | 32-slot `p12c32` TTFT / tok/s | vLLM |
|---|---:|---:|---:|
| 128/C16 | 137 / 1138 | **67** / 1199 | 101 / 1364 |
| 128/C32 | 1384 / 1187 | **113** / **1887** | 202 / 2442 |
| 1024/C16 | 349 / 741 | **244** / 757 | 424 / 940 |
| 1024/C32 | 2322 / 762 | **571** / 947 | 852 / 1322 |
| 4096/C16 | **706** / 390 | 947 / 362 | 1300 / 499 |
| 4096/C32 | 4792 / 392 | 2540 / 363 | 2007 / 598 |

Three more TTFT cells flip (128/C16, 128/C32, 1024/C32). At 4096 the 16-slot chunk-4096 packet
keeps the better throughput (1024-row chunks, no attention roles). With the serving packet
chosen by input length (32-slot for <= 1024): **14 of 60 metric-cells ahead**.

Negatives on `p12u` (2026-09-20): walk depth 10 / 15 (B=1 11.58 / 11.70 vs 11.18 at 12);
`PLOW_MULTISTEP` 4 / 8 / 16 at C1 (TPOT 10.96 / 10.95 / 10.94, p99 ITL 44 / 88 / 175 ms) — the host
is already out of the C1 step.

### C1 pass: where the last few percent are (2026-09-21)

Body ablation on the B=1 step (10.94 ms; gates and signals intact, `PLOW_NV_ABLATE_LO`):
NormResidualNorm 0.57 ms (96 x 5.9 us), FlashDecode 0.71 (48 x 14.9 us), FlashMerge 0.28, RoPE
0.10; ALL bodies off = 0.74 ms, so the 25% gate share is waiting on narrow-op bodies and
stragglers, not packet machinery. By bytes GLU and QKV walk at ~3.2 TB/s (the HBM roof), the
single-stream GEMVs (down, o_proj) at ~2.4, the lm_head at 2.8.

Landed (`7cd6cd47`):
* **One activation load per k-step on the B=1 walk** (`ONE`): rows 0-7 and 8-15 of the mma both
  clamp to row 0 at B=1, so the second load was a duplicate. step_bench B=1 11.17 -> 11.04.
* **Recipe: threaded split tokenizer** (16 / 512, what the 26B recipes had): C1 TTFT 50.1 -> 47.3
  at 1024 in, 186.3 -> 176.7 at 4096, 863 -> 830 at 15000. Finer splits no better.
* **Recipe: BOS rungs** `256,1088,1152,4160,4224` + the Lt policy rows for them: 1088 is worth
  47.22 -> 46.82 at 1024 in (the 1152 bucket cost as much as 1024 + a 1-row tail launch); the
  others neutral. An appended rung NOT in `CUBLASLT_PREFILL_WIDE_ROWS` runs every projection on
  the native GEMM object — the 26B's 1152 rung had been doing that (neutral there).

Negatives: walk depth per stream count (NW=1 at 16/20/24: all worse than uniform 12); two row
blocks per k-step (B=1 -0.1 ms on the K=15360 walk, neutral-to-worse on the 26B); attention
`QGLOB` / `QREG` / `REDBOUND` (<= 0.4%); probing Lt algorithms for the appended rungs (default
heuristics were already as fast); `PLOW_MULTISTEP` 8 / 16.

Ladder on `p12y` (all of the above), vs vLLM 0.28, with the 32-slot packet for the <= 1024 C16/C32
cells: **16 of 60 metric-cells ahead**. New: 1024/C1 TTFT 46.70 vs 46.7 (level), 15000/C4 TTFT
1652 vs 1653. C1 TPOT 10.82 / 11.05 / 11.19 / 11.37 / 11.68 vs 10.55–10.62 (1.03–1.10x), C4
11.50–27.5 (1.09–1.49x), 128/C16 12.83 vs 11.01. NOTE three of the sixteen are the >= 4096 C32
TPOT cells, which "win" only because the 16-slot packet queues half the requests.

### Batched rungs: decode attention is the batch-scaling cost (2026-09-21)

Body ablation of FlashDecode on `p12y` (step_bench, distinct prompts, ctx 1024): B=4 12.58 ->
10.53 ms, B=16 15.22 -> 10.80 (body = 2.05 / 4.4 ms vs 0.71 at B=1). At ctx 192 the step is
10.82 / 11.50 / 12.40 (B=1/4/16) and 10.52 / 10.79 with attention free — BELOW vLLM's 10.58 /
11.01 TPOT, so the short-input C4/C16 gap is attention fixed latency, not the GEMV walk.

Landed (`9bf12059`): **`PLOW_SLIDING_NS_GRID`**. The sliding-layer nsplit was a ceil: t=2/4/8/16
-> ns 9/5/3/2 = 144/160/192/256 work items on 132 blocks, so some blocks ran two items and
FLASH_MERGE waited. Floored to 8/4/2/1: B=2/4/8 11.92/12.55/13.71 -> 11.45/11.98/13.04 (26B
7.90/9.94/13.13 -> 7.57/9.57/12.72); B=1, B=16 unchanged; ctx 192 B=4 11.50 -> 11.30, B=16
12.40 -> 12.24. Served greedy consistency unchanged.

Negatives: multi-row attention tiles (`FA_DEC_TILE_R` 2 / 4 KV rows per thread per tile, fewer
barriers): ~1%; tensor-core QK for the hd512 full layers (`PLOW_NV_FA_TC_GQA8_HD512`): ~1.5%.
So the cost is per ROW (score: one 5-round lane reduction per (row, head); P.V: one V load per
row-group), not per tile.

### FlashDecode by sub-phase, and the score partials in smem (2026-09-21)

Timing-only builds of the B=16 step (15.24 ms, attention 4.4): **score phase 2.71 ms** (K loads +
dots 2.15, the per-(row, head) shuffle reductions 0.56), **P.V 1.16**, everything else (barriers,
softmax reductions, Q staging, fold) ~0.5. At B=1 the split is 0.19 / 0.09 / 0.43 of 0.71: fixed
cost. At ctx 8192 / B=4 (13.34): score 1.47, P.V 0.92 — the hd512/GF8 layers. P.V moves the same
bytes 2.3x faster than score because it holds 8 V rows in flight and score only 2 (`WPR_RB`),
and the shuffles serialize the loop.

Landed: **`PLOW_NV_FA_SPART`** (+ `PLOW_NV_FA_WPR_RB256`), stamped by the manifest (`fa_spart`)
for DENSE sm_90a packets. Lanes park 64/GF lane-group partials per (row, head) in smem, threads
fold their own row after the barrier; hd256/GF2 then has no sync point in the score loop and
holds 8 rows in flight at the same 239 registers. step_bench B=1/2/4/8/16: 11.03/11.45/11.98/
13.04/15.24 -> **10.99/11.30/11.67/12.59/14.35**; ctx 8192 B=1/4 11.36/13.34 -> 11.30/12.99.
Numerics: f32 CPU oracle, 10 cases (GF 2/4/8, multi-tile, window, odd tail) relL2 0.0016-0.0017 =
baseline; served greedy consistency 14/16 + 16/16, no faults.
NOTE `runtime/tests/sm120_interp_op_test.cu` no longer compiles against the op headers (4 stale
call sites, none in the flash section); the oracle run used its helpers + flash section only, and
a launch asking > 48 KiB of dynamic smem needs `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize)`
— without it the kernel silently does not run (all-zero output).

Also stamped for dense packets: **`PLOW_NV_FA_TC_GQA8_HD512`** (the existing tensor-core QK + P.V
for hd512/GQA8). It was ~1.5% at ctx 1024 and is the long-context term: ctx 8192 B=1/4
11.30/12.99 -> 11.14/12.19 on top of the partials (control 11.36/13.34); oracle PASS. Full build
`p12j` (partials + TC), step_bench B=1/2/4/8/16: **11.00/11.32/11.65/12.53/14.30** (was
11.03/11.45/11.98/13.04/15.24); served greedy consistency 14/16 + 15/16, no faults.

**CORRECTION — what each part is actually worth (same day).** The `p12j` ladder showed 128/C1 TPOT
10.82 -> 10.93, which step_bench at ctx 1024 had hidden. Control objects with ONLY the arena floor
raised (`PLOW_NV_ARENA_MIN_BYTES`) are faster on this packet: a step, not a slope — any 40-128 KiB
claim gives B=1/4/16 10.72/11.11/12.13 at ctx 192 (25 KiB: 10.82/11.28/12.24) and
10.93/11.81/15.14 at ctx 1024 (11.03/11.98/15.24); 160 KiB starts losing, 208 KiB is a cliff
(11.79/12.78/18.0). The partials' 64 KiB scratch put the object in that band, so part of the
"kernel" gain above was the claim. Against the same claim, at ctx 1024 B=1/4/16:
hd256 depth 8 alone 11.03/11.82/14.57 (the B=16 win; B=1 +0.10), + partials -0.14 more at B=16,
TC alone 10.99/11.79/15.04 and 11.11/12.30 at ctx 8192 (11.25/13.18). The B=1 costs were wide
paths run on tiny items (a B=1 sliding item is <= 64 rows). Landed: the deep hd256 loop is chosen
PER TILE (`PLOW_NV_FA_WPR_RB256_MINROWS` = 128 live rows). With it, all-on is B=1/2/4/16
**10.98/11.35/11.70/14.26** at ctx 1024 and 10.85/11.19 at ctx 192 (lean = floor + gated depth
only: 10.97/11.37/11.80/14.54 and 10.76/11.16). One dense rule (partials + depth + TC); an
opt-in "serving" knob was written and dropped — all-on wins every cell direction except B=1 at
<= ~200 tokens of context (+0.09 ms). A row-count gate on the partials themselves: no gain.

Negatives: 8 rows in flight without the per-head-dim split (255 registers, spills, every rung
slower: 11.54/12.50/15.24); depth 16 (14.45 at B=16 vs 14.36); thread-per-row (`WPR=0`:
12.09/12.84/15.81); P.V numerators stored group-contiguous for vector loads (neutral);
out-of-line `d_flash_decode` (callees share the entry's register file: 233 registers but
15.47 at B=16); NRN's block-wide sums without their barriers / cross-warp fold (11.02-11.03 vs
11.03: barriers and the smem round trip are free, NRN's 5.9 us is its global loads and stores).
26B: every partials form costs its per-slot MoE rung (B=4 9.57 -> 9.93, the extra 64 KiB smem
claim) for -0.14 / -0.35 at B=8 / 16, so MoE packets keep the old score loop.

### The BF16 NRN fold (`c1f1ac38`) on H100: off, and why it should stay off here

With the fold on (emit default for non-AMD Gemma dense) the 12B serves fluent garbage on H100;
`PLOW_NO_FUSE_NRN=1` in the ladder recipes is the fix that was bisected. Reading the two halves:
NRN2 turns the fused `GemvQkv` into THREE `Gemv` packets with `i3` set -> `d_gemv_nrn`, and NRN1
rides `GemvGlu` `j1` -> `d_gemv_glu_nrn`. Both stage x + the norm into smem and then run the
STAGED DOT8 compute, one row at a time (`for m < M`) — not the tensor-core walk. On this host the
walk is what B=1 and every batched rung run on (12.60 -> 11.93 at B=1 when it replaced dot8), so
the fold trades 96 packets (~0.57 ms of NRN bodies + gates) for the slow GEMV arm on QKV and GLU
of all 48 layers. Even once its numerics are fixed it is a loss here; a fold worth having would
compute the NRN inside the walk ops. Garbage root cause not isolated (needs a fold-on build and a
per-layer diff); open upstream question for `gemma-12b-perf`.

### Sized, not started: vendor (cuBLASLt) attention for the hd512 global prefill layers

Why: the px4 role runs at ~15% of peak and grows with KV (2.6 / 7.3 / ~15 / ~25 ms per launch at
4K / 8K / 12K / 15K KV, x8 layers): ~400 ms of the 830 ms 15000/C1 TTFT, ~78 of 386 at 8192,
~21 of 177 at 4096. QK^T + P.V as batched GEMMs are 0.134 TFLOP per 1K KV per 4096-row chunk
(~2.4 ms at 15K at the measured 850 TFLOP/s) plus a masked softmax over the score tensor (2 GB
bf16 at 15K -> must be tiled by query rows). Estimate: 15000/C1 830 -> ~550 (vLLM 675), 8192/C1
386 -> ~340 (350), 4096/C1 177 -> ~168 (170).
What it takes (survey 2026-09-21): no batched GEMM FFI exists (`device/cuda/lt.rs` binds
`cublasLtMatmul` only, 2-D layouts, `StoredAlgo` keyed (m,n,k)); new role id + `MAX_ROLE` bump;
emit pass isolating the hd512 FlashPrefill (already isolated under packed prefill) and nopping it;
an `AttentionRoute` (2 Lt plans + softmax launch) next to `CublasLtDecodeRoute`; a masked-softmax
kernel as a role cubin (ABI/sha gate) or host .so; packed requests become per-request calls (px4 is
already serial per request); the fused epilogue into `at` must be replicated; not bit-identical to
px4, so new exactness evidence. Global-layer K/V are linear ([slot][kvh][row][hd], mask = ~0), Q is
[row][head][hd] — both usable as strided operands without a permute.

### Serving-tick and prefill-launch findings (2026-09-21, packet `p12m`)

* **Chunk planner bug (fixed, `f27e8d17`)**: `pick_prefill_bucket` planned `s * unit` rows at the goal
  state, so an appended rung that is not a unit multiple (1088 on a 128-row unit) never covered a
  1030-row prompt. The goal state now plans the true remainder. No bench cell moved: `vllm bench
  serve random` prompts launch exactly `rows=1024` / `rows=4096` (`PLOW_PF_PACKLOG`), so the tight
  BOS rungs 1026/4104 (packet `p12t`) are irrelevant to the ladder and were dropped.
* **vLLM reference rows re-measured (`e42e9c9e`)**: the 12B C1 rows at 128/1024/4096 were derived,
  now measured: TTFT 30.74 / 47.49 / 169.76, TPOT 10.57 / 10.64 / 10.65.
* **Prefill launch cost per rung (ms)**: 128 -> 16.8, 256 -> 19.4, 512 -> 26.6, 1024 -> 45.2,
  2048 -> 86.6, 4096 -> 172.3. Linear from 1024 up (~42 us/row), ~11 ms fixed at the small end.
  The 4096-row pass is 168-172 ms in-mux vs vLLM's 169.8 ms TTFT: the 4096+/C1 TTFT gap is prefill
  compute (non-GEMM segments ~52 ms of 172), not scheduling.
* **Static `PLOW_PF_INTERLEAVE` caps 2048/1024/512 — NULL**: splitting a >=4096-row prompt across
  launches loses on TTFT, TPOT and tok/s at 4096/8192-in, C4 and C16. Baseline (0 = unbounded):
  1024/C4 170.2/12.25/296.4, 1024/C16 377.7/17.99/746.7, 4096/C4 336.4/14.99/228.4,
  4096/C16 699/33.02/415.4, 8192/C4 762/18.84/162.1, 8192/C16 1516/55.29/237.4.
* **Fair-share adaptive interleave (launch rows by head count) — NULL**, same reason; removed.
* **Pair walk (`e7502569`, `PLOW_NV_GEMV_MMA_PAIR`, manifest stamps 1 on the dense B=1 walk)**:
  step_bench B=1/4/16 under the real packet flags 10.97/11.70/14.27 -> 10.92/11.60/13.94 (ctx 1024),
  11.10/12.22/16.32 -> 11.03/12.10/15.96 (ctx 8192). Off for the 26B. Needs a full packet build
  + ladder before the ledger moves.
* **Decode build-flag A/B — NULL**: `FA_GF_FULL=4` (hand `cc_dec.sh`) loses at ctx 8192 vs the
  packet's 8 + TC hd512; `GV_UNROLL_GLU=10` spills past the 255-register cap; KUN / PTXSYNC /
  NOSTAGE no effect. Packet flags stay.
* **Where the C16 TPOT gap is**: vLLM's B=16 step is ~11.4 ms (barely above its B=1); plow's is
  14.3 (ctx 1k) / ~16 (ctx 4k). Batch-step scaling, not the tick.
* **32-slot packet `p12c32b` (TTFT/TPOT/tok-s)**: 128/C16 67.8/12.93/1189, 128/C32 114.5/16.49/1821,
  1024/C16 244/18.30/788, 1024/C32 571/26.93/987, 4096/C16 947/34.69/374, 4096/C32 2541/61.63/372,
  8192/C16 2167/65.58/190, 8192/C32 10333/71.84/189, 15000/C16 10405/86.43/92.7,
  15000/C32 27065/88.66/90.5. TTFT ahead of vLLM at 128/C16, 128/C32, 1024/C32 (101/202/852).

### Serving fixes, measured with the memory column (2026-09-21, afternoon)

* **Peak GPU memory is a column** (`cb11570e`): both bench scripts sample the server's process
  tree with nvidia-smi; `peak_mem_mib` in results.csv / ledgers. 12B p12m 65.7-67.7 GiB by cell;
  vLLM 0.28 at 16384/32: 72.7-74.0 GiB (74442-75800 MiB). Sampler on/off/on at C1: no effect.
* **CPU-quiet lock**: concurrent compiles (even nice 19) inflated vLLM's 128-token TTFT 30.9 -> 45.1
  ms (spread 29-32 vs 34-55). Builds now `flock -s /tmp/plow-cpu-quiet.lock`, final sessions
  `flock -x`; the uniform vLLM re-measure under the lock matches the clean references.
* **Uniform vLLM 0.28 baseline** (16384 / 32, `--num-warmups 2 --seed 42`, 32 / 64 prompts, memory
  sampled): the old reference CSVs mixed five sessions (max-model-len 4736-16384, max-num-seqs
  4-32, 32 vs 64 requests at C16, 0 / 2 / 16 warm-ups) — the investor-report audit found four of
  the C16 "wins" were 64-vs-32-request comparisons. 12B C1/C4 rows re-measured: TTFT 30.90 / 47.46 /
  170.92 / 347.80 / 673.49 (C1), 55.05 / 130.57 / 467.37 / 996.38 / 1668.12 (C4).
* **Rung controller narrowed 16 -> 8 on a zero-seeded EWMA** (`7f9ec6a1`): `Ewma` started at 0.0,
  so rung 8 (samples only during ramps) read ~59% of its step time, won `throughput_seat` and
  admission dropped to 8 with 8 requests queued (0.1-0.4 s at MULTISTEP=0, 7.6-18 s at 8; seen in
  the p12m ladder run). Fixed; no Throughput narrowing left in the C16/C32 timelines; means unchanged
  at MULTISTEP=0 (rung fix is robustness, not speed).
* **Serving profile MULTISTEP 8 -> 0** (`ae86a1b8`): 128/C16 TTFT 112.8 -> 74.9 ms (vLLM 101.1),
  p99 ITL 98.5 -> 14.4, 1164 -> 1204 tok/s, TPOT 12.55 -> 12.71; 1024/4096 in level (1024/C16 is
  bimodal 255-378 ms run to run: arrivals sync into waves). MULTISTEP 2 = same means, p99 ITL 25.6.
* **Queue-driven prefill packing** (`7a6ff51b`, `PLOW_PF_INTERLEAVE_ADAPTIVE`, realtime profiles):
  v1 "never split" regressed 15000/C4 (a long tail ran as two padded launches); v2 "always fill"
  split short prompts at C16 (1024/C16 325 -> 380, 4096/C16 701 -> 769); v3 fills only with a
  prompt no launch holds whole. p12p C4 off -> on: 1024 in 167.0 -> 107.7 ms (vLLM 130.6), TPOT
  11.93 -> 12.32; 128 / 4096 / 15000 level. 26B 1024/C4 114.5 -> 94.4, TPOT 10.03 -> 10.32.
* **Packet `p12p`** = current tree with the pair walk stamped (`PLOW_NV_GEMV_MMA_PAIR 1`; decode object
  1.88 MB vs 1.34): step_bench ctx 1024 B=1/4/16 10.99 / 11.72 / 14.27 -> 10.90 / 11.58 / 13.95;
  served greedy consistency 15/16 + 15/16, needles 1025-8199 OK, 0 faults.
* **Prefix cache**: `--prefix-cache` defaults ON but `select_vmm_prefix_layout` auto-selection is off
  under packed prefill / live VMM (the serving configs). Explicit `PLOW_VMM_PREFIX=1` engages: 7230-
  token prompt 404 ms cold -> 33.6 ms on the third request, shared-prefix question 33.2; the second
  identical request still missed (insert after slot release). Default-mode check pending.

### Report day (2026-09-21): uniform baseline, p12mq, 32-slot nulls, prefix cache

* Uniform vLLM 0.28 baseline (`fa12adcc`): one server config for all 20 cells (16384 / 32,
  `--no-enable-prefix-caching`), client `--num-warmups 2 --seed 42`, 32 requests at C1/C4 and 64
  at C16/C32, each session alone on the host. 128/C1 TTFT 30.90 ms (spread 29-32); C16 at 64
  requests 101.7 / 415.7 / 1163 / 2024 / 3302 ms for 128 / 1024 / 4096 / 8192 / 15000 in.
* Quiet host: concurrent compiles (even `nice 19`) inflated vLLM's CPU-bound 128-token TTFT to
  45.1 ms in a discarded run. `campaign.py bench --quiet-lock FILE` (`753193f0`) holds the lock
  exclusively INSIDE the lease (lock-before-lease deadlocked against the baseline run);
  `scripts/bench/quietx.sh` / `quiets.sh` (`b3e79a2c`) add a writer-preferring gate and close the
  fd before exec: an sccache daemon had inherited the shared lock and idled a leased GPU 10 min,
  and flock lets new shared holders pass a waiting exclusive one.
* p12p FINAL (`ea5dfc5f`): TTFT ahead in 10 of 20 cells (all five C16 cells: 73 / 315 / 699 / 1366
  / 2775 ms vs 102 / 416 / 1163 / 2024 / 3302), parity at 1024/C1 and 15000/C4; C1 TPOT 1.02-1.05x.
* Tensor-core decode attention (agent/dec-batched-attn, cherry-picked `8cf90585..e672f22d`; recipe
  `a44776ae`: `PLOW_FA_MMAQK=3`, `PLOW_NS_FULL_ABS=66`). Scores as m16n8k16 mma with K streamed
  from global, 32-row TC P.V ring (claim 154 -> 91 KiB), full layers grid-filled. step_bench
  -0.30 / -0.51 / -1.02 ms at B=1/4/16 (ctx 1024), -0.36 / -0.75 / -1.99 (ctx 8192); the
  same-claim control proves the win is the kernel, not the claim. Served p12mq FINAL (`1d64894f`,
  0 faults): C1 TPOT 10.53-10.73 vs vLLM 10.56-10.66 (parity), 128/C1 94.4 vs 93.3 tok/s; C16 TPOT
  11.83 / 16.37 / 31.53 / 53.55 / 97.99 (p12p 12.34 / 17.79 / 32.86 / 55.51 / 99.95), tok/s 1292 /
  849 / 433 / 249 / 134; TTFT unchanged. Numerics: f32 oracle relL2 0.0016-0.0017, greedy digests
  identical, served consistency 15/16 + 16/16, needles OK. Agent nulls: TC P.V for hd256 (+0.06),
  L2 prefetch across barriers, GF=2 V rows in flight, depth 4, 16-row ring (worse at ctx 8192),
  GF16 full layers (+0.15 ms at B=1). NRN fold on the tensor-core walk (`PLOW_NV_NRN_MMA`, opt-in):
  -0.255 ms at B=1 on a fold-on packet, but the fold still serves garbage (c1f1ac38) -> blocked;
  next step is a per-layer act dump fold-on vs fold-off.
* 32-slot packet p12c32b re-measured on the report binary (`4af4c85f`, MULTISTEP 0): 128/C32
  113.7 ms and 1808 tok/s vs vLLM 160.8 / 2503 (16-slot: 1283 ms / 1301 tok/s); peak 45.5-48.5
  GiB. Chunk 2048 (p12c32c, `4f406e90`) is a NULL: long-prompt C32 TTFT halves (8192/C32 10314 ->
  4695 ms) but the doubled sliding ring slows every decode step (128/C16 TPOT 13.05 -> 17.97 ms,
  1179 -> 856 tok/s). Next: ring depth decoupled from the prefill chunk.
* GSM8K 8-shot greedy, N=200, CONC 8, same prompt bytes: vLLM 194/200, plow 193/200, same final
  answer 198/200, 147 byte-identical outputs (`$T/quality/12b-{plow,vllm}.jsonl`).
* Prefix cache-on scenario (`prefix_repetition`, 8 x 2048-token prefixes + 256-token suffixes,
  C4/C16, both stacks cache on): vLLM 12B C4 TTFT 120.8 ms; plow 291 ms with 1 of 35 lookups
  attached (70 published). Cause: `vmm_publish` records only the whole-prompt boundary unless
  `PLOW_AMD_PREFIX_FINE_ROWS` (the intermediate-checkpoint step, AMD-named) is set; the dataset is
  aligned (same-prefix prompts share exactly 2048 leading tokens incl. BOS, verified with the
  tokenizer). Re-run with 1024-row checkpoints (`cache2_12b_*`) pending; the default should change.

* Prefix cache-on follow-up (`cache2_12b_*`, `cache3_12b_*`): 1024-row checkpoints
  (`PLOW_AMD_PREFIX_FINE_ROWS=1024`) attach 15 of 35 / 29 of 67 lookups, C4 TTFT 291 -> 217 ms, C16
  541 -> 376. The rest miss on the caps, not the match: `PLOW_VMM_CACHE_MEMORY_UTILIZATION` defaults
  to 0.05 (~4 GiB = twelve 320 MiB sliding-window snapshots) and `PLOW_KV_POOL_MIB` to 512 (three of
  the eight 134 MB 2048-row prefix blocks), while every request publishes its own end-of-prompt AND
  end-of-generation boundary (2272 / 2400 rows, 320 MiB each) and evicts the shared checkpoints.
  `FINE_ROWS=2048 PLOW_VMM_CACHE_MIB=9216 PLOW_KV_POOL_MIB=2048`: 17 of 35 / 39 of 67 attached, C4
  TTFT 177 ms (vLLM 121), C16 343 (vLLM 283), TPOT 13.0 / 19.2 (vLLM 11.1 / 12.4), peak 75.4-76.3
  GiB, 0 faults. Remaining misses: first occurrence of each prefix, and same-prefix requests that
  arrive while the first one is still in prefill (publish happens at prefill end). Next: make the
  checkpoint publish and larger caps the defaults (knob defaults only), and consider skipping the
  end-of-generation boundary when the sequence will not be continued.

* Vendor-GEMM prefill attention ADOPTED (agent/pf-attn-batched-gemm, cherry-picked `e8227eed`
  `cb7992f0`; gate `4438f02a` + `16501159`; recipe serves `PLOW_PF_ATTN_GEMM=1`). cuBLASLt Q.K^T /
  causal softmax cubin / P.V per request/tile for the hd512 one-KV-head segments. Gating history:
  every bucket routed -> 128-token cells +0.8/1.6/4.3 ms (C1/C4/C16); bucket-rows gate -> 128/C16
  still +8% (16 x 129-row packs route per request and lose the segment graph); "every slice >=
  1024" -> route off in nearly every C4/C16 launch (a tail slice rides in most packs); shipped:
  "some slice >= 1024". Unexplained: 128/C16 78 vs 73 ms with the route loaded but nothing routed.
* END-TO-END set (`c446ea59`, `c0c824a1`; 14:22-15:50 UTC, one session after another, vLLM re-run
  in the same hour = the new reference CSVs): 12B p12r (recipe-built at HEAD, model.pkt identical
  to p12mq) C1 TTFT 18.35 / 46.65 / 170.0 / 360.3 / 750.0 (vLLM 30.7 / 47.3 / 171.4 / 348.4 /
  673.0); C4 39.5 / 102.9 / 331.9 / 718.2 / 1478 (vLLM 55.9 / 130.7 / 468.3 / 995.5 / 1667); C16
  77.8 / 322.7 / 678.2 / 1264.5 / 2449.5 (vLLM 100.9 / 408.2 / 1164.3 / 1955.4 / 3086.4), C16 TPOT
  11.85 / 16.94 / 30.84 / 50.39 / 86.47 (vLLM 10.96 / 13.79 / 22.79 / 38.04 / 67.79), tok/s 1286 /
  824 / 443 / 265 / 151. TTFT ahead 11 of 20, 15 of 60 comparisons ahead + 13 parity. GSM8K 193/200.
  Cache-on (2048-row checkpoints, 9 GiB snapshot cache, 2 GiB pool, on p12r): C4 187.1 ms / 12.35
  / 291 tok/s, C16 349.4 / 16.73 / 820, 17/35 and 42/67 attached (vLLM 120.8 / 282.6).
* Packet geometry (agent/packet-geometry `ed01b72f`, memo `plans/gemma4-packet-geometry.md`): ring
  depth costs 0.0 ms per decode step (step_bench B=16, ring 4096 vs 2048: 17.80 vs 17.79) — the
  chunk-2048 32-slot "null" was that build's decode object (FA_MMAQK unset, interp cubin 1.49 ->
  2.16 MB), not the ring. The runtime already caps request slices (`packed_prefill.rs
  Manifest::validate`); only devgen's bucket-ladder assert measured the launch chunk. New recipe
  `gemma4-12b.h100.bf16-c32-req1k-16k.toml`: 4096-row launches, 1024-row request slices, ring 2048
  (640 MiB/slot), 32 slots, roles on. MEASURED (17:01-17:38 UTC, both profiles, 0 faults, peak 45-53
  GiB): C32 TTFT 199/591/2142/4291/8271 (p12c32b 114/568/2420/10314/27062; vLLM 154/693/1991/3648/
  6441), 1996 tok/s at 128/C32; C16 long prompts +66-72% vs p12r (four 1024-row slices per launch
  finish together); C1 4096+ +3-11%, C4 mixed (4096 +26%, 15000 -10%). Verdict: replaces p12c32b as
  the 32-slot packet, not p12r; 8192/C32 gate vs vLLM (3648) not met. Dynamic slots exist (`PLOW_VMM_LIVE=1
  PLOW_VMM_LIVE_RINGS=1`, unmeasured). plowc-as-JIT not needed: a geometry change is a 4-9 s devblob
  re-emit against existing objects.

* Batched decode step ATTRIBUTED (agent/dense-batched-decode `f61390ba`, not cherry-picked): step_bench
  on the p12r decode object B=16 11.46 / 12.90 / 13.95 ms at ctx 192 / 1024 / 8192; attention bodies
  compiled out -> 10.48 flat, so attention = 0.97 / 2.42 / 3.46 ms and everything else grows only
  +0.58 ms from B=1 to B=16. Score walk + P.V at ctx 1024 (1.98 ms) sit at ~85% of the HBM byte floor;
  at ctx 192 attention is 3x its floor from per-item fixed latency (FlashDecode 12.6 us/item, FlashMerge
  5.2 us, x48 layers). Softmax reductions cost 0.02-0.04 ms (not a lever). Served vs step: 128/C16
  11.85 vs 11.46; 8192/C16 50.39 vs 13.95 -> the long-context C16 TPOT gap is prefill interference
  (serial prefill launches between decode steps), not the decode kernel. Implemented + oracle-clean
  (10/10, relL2 0.0017) `PLOW_NV_FA_WAUTO` warp-autonomous hd256 item: -0.07/-0.09/-0.10 ms at B=16,
  default OFF, not in any recipe — needs a decode-object rebuild, left for after the report. Next
  lever there: FlashMerge elision at ns=1 (~0.2 ms/step at 128/C16).
* Prefix cache ADOPTED (agent/prefix-cache `27080d9d` -> `bd317b28`; recipe `8b586ee1`): whole-block
  checkpoints at every 2048-row boundary, pressure eviction (the 26B OOM storm was a cuMemAlloc OOM
  mapped to `Device`, never evicting), `PLOW_PREFIX_INFLIGHT_WAIT` (packed admission holds a request
  whose blocks another slot is prefilling), `PLOW_PREFIX_CACHE_OUTPUT` (unmeasured). Serve with
  `PLOW_PREFIX_CACHE=1` alone. prefix_repetition 8 x 2048 + 256, defaults: 12B C4 113.0 ms (26/35
  hits) / C16 281.2 (57/67) vs vLLM 120.8 / 282.6; 26B C4 90.7 / C16 171.4 vs 98.3 / 204.7; 0 faults;
  peak 79-80 GiB (the cache fills free memory; `PLOW_VMM_CACHE_MIN_FREE_MIB` beside co-tenants).
  Caveat: every prompt still publishes on its first prefill (`PLOW_VMM_PUBLISH_SHARED=1` gates nothing
  on the packed path); numbers include it.

* Round 3 integrated for the e2e2 set (19:30-19:45 UTC):
  - Pack fairness (agent/pack-fairness `f5fa92ea` `c54a4382` -> `19b3f926` `dc676c99`; recipe `02daac0d`):
    P99 TTFT at C4/C16 is the cold first wave's makespan in every cell (later waves already beat
    vLLM's P99). Per-row first-wave cost 12B 43-52 us/row vs vLLM 36-40 (vLLM 8192-token steps, plow
    4224-row launch cap); no reorder policy closes it -> lever is 8192-row launches for packs of short
    slices (geometry re-emit). 128/C16 was scheduling: routed buckets built their seg graph on first
    use (23 ms) + the rung controller held 8 back for 4 ticks -> warm-up + `PLOW_RUNG_FAST_PROBE=1`
    (high_concurrency): P99 199 -> 150 ms, mean / tok/s level. COLD_DEMAND and adaptive interleave v3
    at C16 are nulls.
  - Prefill route (agent/pf-attn-packs `317be5e7` -> `3a1415f8`): the +5 ms at 128/C16 was the warmup
    skipping buckets >= 1024 rows (graphs built mid-cell); routed launches now run cached graph pieces
    around the library calls; C16 TTFT -0.7..-2% (single runs). Grouped route for packs =
    `PLOW_PF_ATTN_GEMM_GROUPED`, within 1% -> opt-in off. 15000/C1 747 ms: 63% projection GEMMs, the
    2712-row tail runs the 4096 bucket (next: 2816/3072 rung via PLOW_PF_LADDER_APPEND).
  - NRN fold (agent/nrn-fold-fix `830a8898` `1ed43611` -> `15521f54` `8057c9bc`): root cause =
    `__restrict__` arena on the fold arms let nvcc hoist the second RMS reduction above
    `__syncthreads` (lanes 1-31 reuse invb; ~8x row at layer 0; nondeterministic). Fixed: deterministic,
    4/4 and 16/16 slot agreement, layer-0 relL2 3.07e-3 (fold-off 3.17e-3), needles pass. Still off in
    every recipe: fold B=4/16 step 31/109 ms vs 10.7/11.3 (full weight pass per row), dot8 fold +1.2
    ms at B=1.

* E2E2 set (19:43-22:05 UTC, `339e6be8`; report regenerated + sent 22:10): 12B p12r on the round-3 binary
  (d0859681). TTFT ahead 12/20 (1 parity), 16/60 ahead + 12 parity. C1 TTFT 18.38/45.94/170.31/360.46/
  748.56 vs vLLM 30.31/47.55/170.51/349.21/672.28; C16 TTFT 73.3/312.0/678.0/1261.3/2453.2 vs 102.5/
  408.7/1164.4/2025.7/3084.4; C16 TPOT 11.86/16.54/30.83/50.27/86.40 vs 10.98/13.81/22.79/37.55/67.79.
  GSM8K 194/200 (vLLM 194). Cache-on (PLOW_PREFIX_CACHE=1 only): C4 122.3 ms (26/35 hits) vs vLLM 120.8,
  C16 314.7 (57/67) vs 282.6; peak 77 GiB. vLLM e2e2 within 1-2% of e2e (new reference CSVs).
  Serving front end (agent/serve-frontend `aadf88ed` `0b88ef14`, not yet integrated): host path <1-2% of
  every metric; `PLOW_MUX_INLINE_TICK=1` -1.0% TPOT C1/C16 (26B A/B); tokenizers 0.23 encode -24-36%.

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
