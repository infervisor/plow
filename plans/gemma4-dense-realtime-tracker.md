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
