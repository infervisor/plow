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

## Round 4 (2026-09-22): integrated set for e2e3 (`3a7a5a3b`, packet p12r4)

* Decode GEMV (agent/dense-dec-r4): ONE (one activation load per k-step, rungs 2-8) + claim-ahead L2
  prefetch of each walk's first 64 KiB (`PLOW_GEMV_PREFETCH`, now a GEMMA4_HOPPER dense default).
  step_bench p12r ctx 192 B=1/4/16 10.55/10.75/11.44 -> 10.44/10.57/11.41; 128/256 KiB budgets worse
  (compete with walks in flight). FlashMerge elision parked unmeasured (`d147d65d`). NS_LIVE: no effect.
* Wide GQA2 role on 4160/4224 (default): hc mean TTFT -2..-4%, realtime 4096/C4 -3%; 12B l8192
  launches NOT adopted (mean TTFT +19..44% at C16 from 8192-row packs, +0.9 GiB).
* Serve front end: tokenizers 0.23 (encode -25..-30%), inline mux tick (default on for CUDA).
* `PLOW_PF_DECODE_FIT` removed: served neutral (C16 1024/4096 359.7/980.1 -> 359.9/979.6 ms); the default
  mux trim (`ec1f4a76`) covers the spill case.
* Prefix cache v2 (defaults only in production): unique-prompt 12B peak 79.2 -> 73.7 GiB (vLLM 75.3),
  hits 26/35 and 57/67 unchanged; 4096/C16 still +2.7% TTFT cache-on (cause not isolated).
* Test hygiene: 3 plowrt lib tests fixed (2 also failed on main); AMD gfx942 full-layer nsplit drift from
  `639d1507` gated back to main's rule (`dce1fe05`).

### e2e3 result (p12r4, same session as vllmuni-e2e3, ledger `a3ea207e`, 0 faults)

* 60 comparisons: 16 ahead, 13 parity. TTFT ahead 12/20. C1 TPOT parity (10.42-10.62 vs 10.46-10.56).
* C1 TTFT 128..15000: 18.26/45.92/169.66/356.52/736.51 vs 30.04/47.24/170.08/348.92/671.77.
* C16 TPOT still behind: 11.68/16.18/30.27/49.62/84.77 vs 10.90/13.73/22.72/37.45/67.82.
* Peak memory below vLLM in all 20 cells (65.2-69.1 vs 72.7-74.0 GiB). GSM8K 193 vs 194.
* C32 on the request-sliced packet p12rq4: TTFT 146.75/597.10/2082.35/4215.96/8154.66 ms.
* Cache-on (prefix_repetition): C4 128.1 vs 120.8 ms, C16 297.6 vs 282.6 ms; peak 71.8-72.0 vs 73.5-73.9.
* `PLOW_FA_MMAQK=3` promoted to a GEMMA4_HOPPER emit default; recipes no longer name it.

### Scheduler anatomy vs vLLM 0.28, and the streaming quantum (2026-09-22)

vLLM's V1 scheduler (`v1/core/sched/scheduler.py`, the reference config's own log): one forward per
step over a flat token budget `max_num_batched_tokens=8192` (H100 default), `max_num_seqs=32`;
running requests first (a decode row costs 1 token, a partly prefilled prompt continues its chunk),
then waiting prompts FCFS, the last one cut to fill the budget (`long_prefill_token_threshold=0`, no
per-request cap). Async scheduling is on: step n+1's input ids are copied on the GPU from step n's
`prev_sampled_token_ids`, step n's ids go D2H on a side stream behind an event, and the CPU schedules
and detokenizes step n while the GPU runs n+1 — a one-step lookahead, every token streamed.

plow: one tick = decode feeds -> batched prefill pass -> decode launch. The unified token batch rides
decode rows inside the prefill launch (vLLM's mixed step). The launch budget is the packet's widest
prefill rung (4224 rows here; `PLOW_MAX_CHUNK=4096`), not 8192, because the sliding ring is
`next_pow2(window + chunk - 1)` rows per slot. Realtime packs oldest-first (`PF_INTERLEAVE_ADAPTIVE`),
high_concurrency fills FCFS.

Per-request ITL traces, e2e3 vs the vLLM reference (a stall = ITL > 60 ms):

| cell | stack | stalls/req | ms per stall | stall ms/req | ITL outside stalls | wave spread |
|---|---|---:|---:|---:|---:|---:|
| 4096/C4 | plow | 2.16 | 175 | 378 | 10.95 | 315 ms |
| | vLLM | 0.75 | 203 | 152 | 10.93 | 19 ms |
| 15000/C4 | plow | 7.69 | 191 | 1471 | 11.34 | 1271 ms |
| | vLLM | 3.00 | 304 | 912 | 11.15 | 72 ms |
| 4096/C16 | plow | 12.4 | 186 | 2304 | 13.32 | 1957 ms |
| | vLLM | 4.97 | 268 | 1333 | 12.72 | 1079 ms |
| 15000/C16 | plow | 46.7 | 205 | 9556 | 14.87 | 8221 ms |
| | vLLM | 21.1 | 340 | 7185 | 13.52 | 6027 ms |

Decode itself is level at C4 and 3-10 % behind at C16; the long-input TPOT gap is stall time. plow
takes 2-3x more stalls (4096-row cap vs an 8192-token budget) and oldest-first packing staggers each
wave, so prefill keeps landing on running decoders; vLLM's FCFS fill prefills a wave together, which
puts the same time in TTFT instead. **A scheduler only moves time between TTFT and TPOT**: mean E2E =
TTFT + 127 x TPOT exactly at 128 out, and plow's E2E ratio to vLLM is C1 0.99-1.04, C4 1.00-1.08, C16
1.05-1.12 — the same deficit its output tok/s shows. Winning both metrics at a cell needs E2E, i.e.
per-row prefill cost in mixed launches (45-50 vs 33-42 us/row at C16) and the batched decode step.

**Arms on p12r4, one session, same binary** (TTFT / TPOT / p99 ITL, ms):

| cell | ctl (MULTISTEP 4, adaptive) | MULTISTEP 0 | MULTISTEP 0 + FCFS | vLLM 0.28 |
|---|---|---|---|---|
| 128/C1 | 18.3 / 10.42 / 41.8 | 18.3 / 10.45 / 10.5 | 18.2 / 10.45 / 10.5 | 30.0 / 10.46 / 11.3 |
| 4096/C1 | 169.6 / 10.53 / 42.3 | 169.8 / 10.56 / 10.8 | 169.8 / 10.56 / 10.8 | 170.1 / 10.55 / 11.4 |
| 15000/C1 | 736.1 / 10.63 / 42.7 | 736.4 / 10.65 / 10.8 | 736.8 / 10.65 / 10.8 | 671.8 / 10.56 / 11.5 |
| 128/C4 | 34.4 / 10.58 / 42.4 | 39.1 / 10.66 / 13.9 | 34.2 / 10.60 / 10.9 | 55.3 / 10.49 / 11.3 |
| 1024/C4 | 101.4 / 11.61 / 53.7 | 101.6 / 11.63 / 53.5 | 123.3 / 11.46 / 53.2 | 129.6 / 11.07 / 11.8 |
| 8192/C4 | 701.3 / 17.18 / 190.0 | 566.2 / 18.03 / 187.9 | 567.5 / 18.04 / 187.3 | 995.8 / 13.53 / 13.6 |
| 15000/C4 | 1437.8 / 22.29 / 210.5 | 1041.6 / 26.53 / 211.7 | 1042.3 / 26.54 / 211.9 | 1665.4 / 18.06 / 326.0 |

* Per-token streaming costs 0.03 ms/token at C1 and cuts p99 ITL 4x, to **below vLLM on both TPOT and
  p99 ITL at every C1 rung**. The realtime profile should serve `PLOW_MULTISTEP=0`.
* At C4 the quantum was delaying prefill: TTFT -20 to -28 % at 8192/15000 in, TPOT +5 to +19 %, E2E
  4269 -> 4411 at 15000. A shift, not a win — the pipeline (below) is what should recover the TPOT.
* vLLM-style FCFS filling is a wash at a 4096-row budget: identical everywhere but 1024/C4 (adaptive
  keeps TTFT 101 vs 123 ms for +0.17 ms TPOT) and 128/C4 (FCFS 34.2 vs 39.1 ms TTFT).
* Next: `PLOW_DECODE_PIPELINE` (lookahead-1, per-token streaming, device-resident inputs) and an
  8192-row launch rung with `PLOW_MAX_REQUEST_CHUNK=4224` so the sliding ring stays at 8192 rows.

### Lookahead-1 decode pipeline, measured (PLOW_DECODE_PIPELINE, 2026-09-22)

p12r4, one session, one binary, `PLOW_MULTISTEP=0` in both arms, realtime profile, 128 out.
Arms repeated (`ms0`/`ms0b`, `pipe`/`pipe2`): within-arm spread is <=0.07 ms TPOT and <=3 ms p99 ITL,
so every delta below is outside the floor. Greedy equivalence: **12/12 byte-identical** completions
with the pipeline off vs on.

| cell | vLLM 0.28 | ms0 | pipe | E2E / vLLM |
|---|---|---|---|---|
| 128/C1 | 30.0 / 10.46 / 11.3 | 18.2 / 10.44 / 10.5 | 18.3 / 10.42 / 10.5 | 0.990 -> 0.987 |
| 1024/C1 | 47.2 / 10.54 / 11.4 | 46.1 / 10.52 / 10.6 | 45.9 / 10.50 / 10.6 | 0.998 -> 0.995 |
| 4096/C1 | 170.1 / 10.55 / 11.4 | 169.8 / 10.56 / 10.8 | 169.4 / 10.53 / 10.7 | 1.000 -> 0.998 |
| 15000/C1 | 671.8 / 10.56 / 11.5 | 736.1 / 10.65 / 10.8 | 737.0 / 10.62 / 10.8 | 1.038 -> 1.036 |
| 128/C4 | 55.3 / 10.49 / 11.3 | 32.0 / 10.57 / 10.7 | 36.0 / 10.60 / 12.2 | 0.990 -> 0.996 |
| 1024/C4 | 129.6 / 11.07 / 11.8 | 101.6 / 11.64 / 53.6 | 101.3 / 11.60 / 53.6 | 1.029 -> 1.026 |
| 4096/C4 | 465.8 / 12.07 / 11.9 | 324.8 / 13.85 / 175.0 | 324.5 / 13.81 / 175.0 | 1.043 -> 1.040 |
| 15000/C4 | 1665.4 / 18.06 / 326.0 | 1039.2 / 26.40 / 212.7 | 1184.8 / 24.21 / 212.2 | 1.109 -> 1.076 |

(TTFT / TPOT / p99 ITL in ms. E2E = TTFT + 127 x TPOT.)

* Every C1 rung gains 0.02-0.03 ms TPOT for no TTFT: the host turnaround was already small there,
  and what the pipeline removes is exactly that.
* **15000/C4 is the win**: TPOT -8.3 %, and E2E falls 1.109 -> 1.076 of vLLM, so it is not the
  TTFT/TPOT shuffle the MULTISTEP arms were.
* **128/C4 regresses** (TTFT 32 -> 36, p99 ITL 10.7 -> 12.2). Structural: with a 128-row prompt the
  in-flight lookahead step must be drained before a prefill launch, and at that size the drain is
  most of the tick. The prefill overlap below is what removes the drain.
* Against vLLM after this round: plow wins TTFT at 7 of 8 cells and p99 ITL across all of C1, and
  still trails on TPOT/E2E at C4 with long inputs and on p99 ITL at 1024-4096/C4 (53.6 / 175.0 vs
  ~11.8 — the cost of packing prefill into running decoders instead of prefilling a wave together).

### Prefill overlap (PLOW_PIPE_PREFILL) — implemented, measurement queued

The mixed prefill/decode launch is enqueued behind the step in flight instead of draining it:

* `d_last` `[batch]` device buffer holds every slot's newest sampled token. A decode step copies
  `in.ids` into it; a mixed step scatters its compact terminal block into it.
* `PackedTerminal` splits into `launch` (no sync) and `run`; the pipelined path submits body +
  terminal, records a D2H of the sample block and an event, and returns. Frontiers and `pos` commit
  at enqueue, which single-stream program order makes safe.
* Decode rows that ride the launch take their input from `d_last` (4-byte D2D after the staged
  upload) -- but only rows the pipe still owes a token; for any other row the host token is
  authoritative and `d_last` is stale.
* A prompt that completes in launch n joins launch n+1 as a decode row reading its first token off
  the device, so it does not idle a whole launch waiting for the host to read that token.
* `pipe_reap`/`pipe_full` hold the queue at lookahead-1 across consecutive prefill ticks; without
  them a run of prefill ticks overflows the two pinned readback buffers.

### Prefill overlap: first run faulted, and why (2026-09-22)

`PLOW_PIPE_PREFILL=1` on p12r4 died ~40 tokens into the greedy probe:

```
CUDA_ERROR_ILLEGAL_ADDRESS (700) at cuCtxSynchronize
gpu: batched prefill failed ... packed=1, fatal=true
```

Deterministic: both overlap arms of the ladder (`pf`, `pf2`) exited rc=28 with 0 cells, while every
pipeline-only arm ran all 8 cells.

**Cause.** `packed_token_body_inner`'s own SAFETY comments state the invariant: `pf_ids`, `pf_pos`,
`slot_buf`, `req_buf` "live on self past the stream_synchronize". Every packed launch drained the
stream before returning, so the host was free to rewrite them next tick. The pipelined path removed
that drain and kept the vectors, so the host restaged launch n+1 while launch n's async H2D copies
were still queued behind the running kernel; the copies then read the new contents. The terminal's
`host_rows` and patched instruction block have the same exposure. Corrupt row/program metadata is
exactly an illegal address, and the timing fits: correct until the host runs ahead of the device.

**Fix, per buffer's exposure window.**

* Body staging (`pf_ids`, `pf_pos`, `slot_buf`, `req_buf`, `kvlen_buf`): a `body_ev` event recorded
  right after those copies and waited on before the next staging. Those copies precede their own
  launch's kernel, so in the steady state the device is already past them and the wait returns at
  once; a host that runs ahead is throttled, which is the part that was missing.
* Terminal staging (`host_rows`, instructions): its copies sit AFTER the body kernel, so an event
  there would wait out the whole 170 ms launch and serialise what the pipeline exists to overlap.
  Those two small arrays are double-buffered instead (`stage_rows`, alternating per launch).

`PLOW_PIPE_PREFILL` became a level so the two new behaviours can be separated: `1` parks the mixed
launch and sources its decode rows from `d_last`, `2` additionally admits a just-prefilled row to
the next launch on its device-resident first token.

**What the overlap can be worth, measured.** `PLOW_PF_PACKLOG` over 659 ticks (idle outliers
trimmed): host gap is **1.67 ms against a 172.7 ms prefill tick (1.0 %)** and **0.052 ms against an
11.0 ms decode tick (0.5 %)**. So closing the gap is not where the value is — it has to come from
removing the drain (which costs 128/C4 its TTFT, 32.0 -> 36.0 ms under the pipeline) and from
decode rows not losing a launch after their prompt completes.

### 8192-row launch rung: null (2026-09-22)

p12r8 (`PLOW_MAX_CHUNK=8192`, `PLOW_MAX_REQUEST_CHUNK=4224`) against p12r4, same binary, arms
repeated: every cell within the repeat spread (E2E / vLLM 0.998 / 1.000 / 1.038 / 1.029 / 1.042 /
1.11x, identical either way). The rung is built but not chosen: a request is still capped at 4224
rows per launch, so filling 8192 needs two requests' chunks in one launch, which oldest-first
adaptive packing avoids and `pf_pack_budget`'s cost model rejects (a ~45 %-padded 8192 loses to
`[4096, tail]`). Testing the wider launch for real would need FCFS fill as well; not pursued.

### Checkpoint P gate: 20/20 accepted (2026-09-22)

`scripts/perf_gate_ci.sh 91f03b9c` -> rc=0, every flipped default certified. Two of the twenty
needed work beyond re-running the campaign:

**`emit.gemma4_sm90_hd256_gqa2_wide` — accepted after citing the right neutral metric.** TTFT is
better at all four cells of its group (8192/C1 359.6 -> 355.5, 8192/C4 763.8 -> 749.7, 15000/C1
748.0 -> 736.3, 15000/C4 1607.2 -> 1519.3 ms) but clears its floor only at 15000/C1. The first
cert cited only TPOT as neutral, so the three within-floor TTFT rungs had no accept path. The flip
widens the row range of an attention role that is already the default at <=4096 rows to the
4160/4224-row whole-tile launches -- same object, same math, no work added -- which is the
physical argument `rungVerdict` wants for "not worse"; with TTFT cited too it accepts.

**`emit.fa_mmaqk` — rejected on the merits, returned to opt-in.** 26B 128/C1 TPOT 5.400 -> 5.418 ms
against a 0.002 floor, reproducible across both treatment runs. That is a `servingVerdict`
rejection, and unlike `rungVerdict` the serving check has no neutral escape hatch by design: a
serving metric may not get worse, whatever the reason. The knob still wins where its evidence was
taken (26B 1024/C1 TPOT 5.551 -> 5.484 beyond floor, 128/C4 and 1024/C4 within it; 12B 128/C16
12.68 -> 11.80), so this is the `ccd36c59` situation exactly: the default returns to UNSET/OPT_IN,
`apply_production_defaults` drops its case, and the 16 BF16 Gemma-4 H100 recipes set
`PLOW_FA_MMAQK=3` themselves. Every campaign packet keeps the flag; only the uncertified default
goes away. `gemma4-12b.h100.bf16-plain` is deliberately left without it -- it is the
default-knobs control.

Floors are tight because `floorOf` is |median(ctrl) - median(ctrl2)| + k*max(MAD): when the two
control runs agree to 0.04% on a 5.4 ms TPOT, a 0.33% regression is outside the floor.

### Stage 1 after the frontier rework: re-measured, and the 128/C4 TTFT regression is gone

p12r4, `PLOW_MULTISTEP=0`, 32 prompts, 128 out. TTFT ms / TPOT ms / p99 ITL ms.

| cell | vLLM 0.28 | base | pipe (stage 1) | E2E / vLLM |
|---|---|---|---|---|
| 128/C1 | 30.0 / 10.46 / 11.3 | 18.3 / 10.44 / 10.5 | 18.3 / 10.42 / 10.5 | 0.990 -> 0.987 |
| 15000/C1 | 671.8 / 10.56 / 11.5 | 737.0 / 10.65 / 10.9 | 736.5 / 10.62 / 10.8 | 1.038 -> 1.036 |
| 128/C4 | 55.3 / 10.49 / 11.3 | 32.3 / 10.58 / 10.9 | 32.0 / 10.53 / 10.7 | 0.992 -> 0.987 |
| 15000/C4 | 1665.4 / 18.06 / 326.0 | 1039.8 / 26.43 / 212.4 | 1184.7 / 24.13 / 212.1 | 1.110 -> 1.073 |

The 15000/C4 win reproduces (TPOT 26.43 -> 24.13, -8.7%; E2E 1.110 -> 1.073) and it is a trade,
not a free win: TTFT goes 1039.8 -> 1184.7 there. No cell regresses on E2E.

**The regression that motivated the overlap is gone.** Before the rework, stage 1 cost 128/C4 its
TTFT (32.0 -> 36.0 ms); measured again on the reworked core it is 32.3 -> 32.0. The `ahead` lag
compensation was the cost, not the missing overlap. With the host gap already measured at 1.0 % of
a prefill tick and 0.5 % of a decode tick, stage 2's remaining upside is that gap alone.

### Prefill overlap, second attempt: replay at level 1, fault at level 2

The frontier rework removed the illegal address at level 1, and left a subtler bug: level 1 emitted
one token twice. `p1`'s completion is `p0`'s with a token duplicated and the tail one token short
("These are daily daily maritime...", "dereferferencing") -- the device stream was right and the
host fed a row twice.

Cause, in two parts:

* `pipe_step`'s internal drain re-enqueued from the mux's pre-drain `feeds`. The mux's own drain
  path re-gathers (`feeds = gpu_decode_feeds(...)`, right after it drains) and says so in its
  comment; `pipe_step` cannot re-gather from inside the engine, so a row the drain had just
  produced a token for was fed its previous token and resampled it.
* `carry` was set from `device_ids` -- whether the row's input came from the device -- rather than
  from its KIND. A mixed step whose decode rows the host staged therefore had no carry rows at
  all, which is what made `pipe_covers` fail and that drain reachable. It also dropped those rows
  from the prefix-cache history.

Both fixed. A third defect found by construction while reading: `gpu_decode_feeds` gathers only
`step > 0` rows and `use_pipe` requires a non-empty feed set, so a row whose first token is still
inside a parked step is invisible to the decode path; while it also held `did_prefill` true, the
mux's drain was suppressed and nothing could ever complete that step. `did_prefill` now excludes
rows the pipe owes a token, which are waiting on a readback, not on prefill.

Level 2 (`PLOW_PIPE_PREFILL=2`, a prompt decoding from its device-resident first token) still
faults: `CUDA_ERROR_ILLEGAL_ADDRESS` at `cuEventSynchronize`, batched prefill, `packed=1`. That is
a memory bug of its own and is not the liveness hole above.

### Prefill overlap: three bugs fixed, one left, and the verdict

The three defects found by reading, all fixed in the tree:

1. `pipe_step`'s internal drain re-enqueued from the mux's pre-drain `feeds`, so a row the drain
   had just produced a token for was fed its previous token and resampled it. The mux's own drain
   path re-gathers (`feeds = gpu_decode_feeds(...)`); `pipe_step` cannot, so it now refreshes the
   fed tokens from what the drain returned.
2. `carry` was set from `device_ids` (whether the input came from the device) instead of the row's
   KIND. A mixed step whose decode rows the host staged had no carry rows at all, which is what
   made `pipe_covers` fail and defect 1 reachable; it also dropped those rows from the
   prefix-cache history.
3. `did_prefill` counted a row whose prompt was consumed but whose first token was still inside a
   parked step. `gpu_decode_feeds` gathers only `step > 0` rows and `use_pipe` needs a non-empty
   feed set, so such a row is invisible to the decode path while it suppresses the mux's drain --
   nothing could complete the step holding its token. It now excludes rows the pipe owes.

Verified on plowrt_pipe5 (`pf_noise.sh`, `dupscan.py`): level 1 clean, 0 corruption hits, down
from 2. The noise floor matters here -- `off` vs `off2` is 8/10 identical at conc 4, so the
earlier "p0 8/10" reading of stage 1 was the floor, not a regression; `p0` vs `p0b` is 10/10.

**What is still broken.** Level 1 FAULTS when rows retire while a mixed launch is parked -- not a
deadlock, which is what the tick trace looked like before the fault lines were read:

```
PACKLOG TICK t_ms=1233.3 decode_ms=11.75  did_prefill=0 decode_rows=2   <- two requests retired
PACKLOG TICK t_ms=1408.1 decode_ms=172.21 did_prefill=0 decode_rows=2   <- parked-launch wait
decode pipeline: step failed ... fed=2 ... CUDA_ERROR_ILLEGAL_ADDRESS at cuEventSynchronize
```

Ticks stop there because the context is poisoned, not because the host is waiting; at 15000/C4 of
the ladder the same fault left the client waiting with the GPU pinned, which is why it first read
as a livelock. The repro is 4 x 14k-token prompts at 32 out.

The open lead is the retired-row frontier rollback added by the frontier rework:

```rust
if pipe.retire[b].is_some() {
    self.pos[b] = self.pos[b].saturating_sub(1);
    continue;
}
```

It is the only `pos` bookkeeping that fires exactly on retirement, which is exactly the trigger.
A mixed launch commits its rows' frontiers through the token-batch staging rather than through
`pipe_enqueue`, so a row retired while both a mixed step and a decode step hold it is rolled back
once for a frontier that moved twice -- and the next launch then maps and writes past what
`ensure_rows` reserved. Not yet confirmed.

Level 2 faults as well, in the batched prefill, and is a separate bug.

**Verdict: not shipped.** `PLOW_PIPE_PREFILL` stays opt-in at 0 and is documented as experimental
with both failure modes named. The case for finishing it is weak on the measurements: the host gap
it closes is 1.0 % of a prefill tick and 0.5 % of a decode tick, and the 128/C4 TTFT regression
that originally motivated it turned out to be the `ahead` lag compensation, which the frontier
rework already removed. Stage 1 (`PLOW_DECODE_PIPELINE`) is the part that pays, and it is correct
and measured.

## Full ladder at HEAD, both models, 20 cells each vs vLLM 0.28 (2026-09-23)

First pass: 34 of 40 cells. 12B 20/20, 26B 14/20. The gaps and the losses have three named
causes, none of them kernel speed.

### 1. The 6 missing 26B cells: the MoE memset-hoist aborts the prefill capture

`26B realtime` lost every cell at 4096 rows and above:

```
FAIL: incomplete cell: ok=0/32 failed=32 generated=0/4096
gpu: batched prefill failed error=device fault: graph edges: CUDA_ERROR_INVALID_VALUE
     (code 1) fatal=false packed=3
prefill seg graph warmup failed ... bucket=6 / bucket=7
```

`graph edges` is `hoist_memsets` (`device/cuda.rs`), the pass that lifts cuBLASLt's workspace
memsets above the MoE Lt glue kernel so they overlap instead of serialising. It runs only when
`untouched` is non-empty, and `untouched` is the MoE Lt glue list -- which is why the 26B fails and
the 12B, dense, never does. A driver refusal on a dependency *query* propagated out of
`graph_capture_hoisting` and failed the whole capture, so an optimization could kill the launch.

Fixed: query refusals skip that memset and leave the graph exactly as captured (the queries all run
before that node's first mutation, so a skip is always on an unmodified graph). The three edge
mutations stay fatal -- a half-moved edge is a race, not a lost optimization.

`PLOW_PF_INTERLEAVE_ADAPTIVE=0` was ruled out as the cause by a one-variable rerun: it failed
*earlier* (1024/C4, ok=8/32) with the same error.

Correction to an earlier note in this file: the `prefill seg graph warmup failed` warning is **not**
benign. On the 26B realtime profile it cost 6 of 20 ladder cells.

### 2. Every C32 cell: a 16-slot admission cap, not throughput

The fingerprint is exact. 12B: C32 tok/s 1318.9 vs C16 1310.7; 26B: 1371.5 vs 1371.1. Peak memory
is identical at both concurrencies. TPOT at C32 matches or beats vLLM (12B 11.41 vs 11.59). Only
TTFT explodes (12B 128/C32: 1267 ms vs vLLM 161).

The `ladder16k` / `ctx16k` packets serve 16 slots: chunk 4096 makes a slot's sliding ring
`next_pow2(window + 4095)` rows, and 32 slots would need ~110 GiB on an 80 GiB card. Half the
requests therefore wait out a whole 128-token generation before their first token. vLLM runs 32
slots on paged KV.

Worth recording: at C16/C32 plow peaks at 67.7-70.8 GiB against vLLM's 74.6-75.8 GiB. plow uses
*less* memory and still fits fewer sequences -- the cost is contiguous power-of-two rings per slot,
not total footprint.

`*-c32-16k.toml` is the packet shaped for this half (chunk 1024 -> 2048-row ring, 32 slots,
45-48 GiB peak). Being measured.

### 3. p99 ITL on the realtime profile: the MULTISTEP wave

p99 ITL at C1 is exactly 4x TPOT on both models -- 12B 41.8 = 4 x 10.45, 26B 21.6 = 4 x 5.38 --
and median ITL is literally `0.000`. That is `PLOW_MULTISTEP=4` emitting four tokens per wave.
vLLM streams one at a time (11.3 / 5.8 ms). TTFT and TPOT are unaffected; only the streaming
metric is. A/B running: 12B with `MULTISTEP=0 + PLOW_DECODE_PIPELINE=1` (per-token streaming is
precisely what stage 1 buys), 26B with `MULTISTEP=0` alone, since the pipeline is unavailable
whenever cuBLASLt is enabled (`gpu.rs:3539`).

### Hoist fix verified (007e864c)

26B realtime: 10/10 cells, zero `graph edges` faults. The new diagnostic is exact --
`refused=1 hoisted=12 nodes=154` on the big graphs, `refused=1 hoisted=2 nodes=22` on the small
ones. Exactly **one** node per capture refuses the query; every other memset still hoists. So the
optimization is ~92 % preserved and the launch no longer dies for it. Worth a follow-up: one node
per graph refusing a dependency query is a specific shape, not random driver flakiness.

### Complete ladder, 40/40 cells (plow wins-losses vs vLLM 0.28)

| model | TTFT | TPOT | p99 ITL | tok/s | E2E |
|---|---|---|---|---|---|
| Gemma-4-12B | 13-7 | 8-12 | 9-11 | 3-17 | 3-17 |
| Gemma-4-26B-A4B | 11-9 | 3-17 | 8-12 | 0-20 | 0-20 |

TTFT is the real strength and it is not close in places (26B 128/C1 21.3 vs 38.3 ms; 15000/C4
592 vs 804). TPOT and throughput are the weakness. Recovering the six 26B cells made the verdict
*worse*, not better -- they all land at E2E 1.08-1.25x.

The TPOT gap is context-dependent, which points at decode attention rather than the GEMMs.
26B at C4, 128 -> 15000 in: plow 7.85 -> 16.86 ms (+9.01), vLLM 7.25 -> 10.86 (+3.61). The
constant part is within 8 %; the per-KV-row part costs 2.5x. 12B at C16: plow +73.1 ms over the
same span, vLLM +56.9.

### Where the TPOT gap actually is: KV traversal, not the weight walk

Split TPOT into its context-free and per-KV parts using measurements only. (A least-squares
intercept is NOT safe here: vLLM's TPOT-vs-context curve is convex, so a straight line puts its
intercept ~1.5 ms under the measured 128-token cell and invents a constant-term gap that does not
exist. Use the 128-in cell as the constant and a two-point secant for the slope.)

```
                constant part (TPOT@128in)        per-KV slope (us per 1k ctx)
                plow   vLLM    x                  plow     vLLM     x
12B    C1      10.42  10.46  1.00                 13.5      6.7   2.00
       C4      10.63  10.49  1.01                  784      509   1.54
       C16     11.68  10.90  1.07                 4915     3827   1.28
26B    C1       5.38   5.04  1.07                 16.1      3.4   4.80
       C4       7.85   7.25  1.08                  606      243   2.50
       C16     11.17   8.84  1.26                 2963     1754   1.69
```

The context-free part is at parity on the dense 12B -- 1.00 / 1.01 / 1.07x, and the HBM efficiency
of the weight walk is the same as vLLM's (60.8-68.2 % of 3352 GB/s against 61.3-67.9 % over
23.81 GB of weights). The whole TPOT gap is the per-KV-row term.

So the decode target is decode attention, not the GEMMs: 1.28-2.0x on the 12B, 1.69-4.8x on the
26B. At C1 the absolute cost is negligible (13 us per 1k) so it never shows; at C16/15000 it is
73 ms of an 85 ms step.

(C32 rows read better than vLLM only because plow serves C32 on the same 16 slots as C16 -- that
column is the admission cap, not a decode result.)

### The 32-slot packet at C16/C32: a traffic-dependent trade, not a fix

`*-c32-16k.toml` (chunk 1024, 2048-row sliding ring, 32 slots) removes the admission cap and the
short-prompt C32 cells improve a lot. It also costs much more on long prompts, because chunk 1024
turns a 15000-token prompt into 15 launches. E2E vs vLLM at C32, 16-slot -> 32-slot:

```
in       12B                26B
128      1.663 -> 1.255     1.706 -> 1.302
1024     1.378 -> 1.253     1.630 -> 1.523
4096     1.204 -> 1.506     1.533 -> 2.032
8192     1.133 -> 1.581     1.514 -> 2.223
15000    1.101 -> 1.727     1.545 -> 2.437
```

The crossover is between 1024 and 4096 rows on both models. 12B 128/C32 TTFT goes 1267 -> 113 ms
(vLLM 161, so plow wins it) at 46-49 GiB peak instead of 67-70. But 15000/C32 TTFT goes
13405 -> 26936 ms. At C16 the 32-slot packet is worse everywhere (it pays chunk 1024 for slots it
does not need).

So **no single plow packet covers the C32 column the way vLLM's one config does.** The packet has
to be chosen for the traffic: c32-16k below ~1k input, the 16-slot ladder packet above. That is a
real limitation of the AOT-compiled packet model and is reported as one. The headline ladder below
uses the 16-slot packet for the whole C16/C32 half, which is the recipes' own choice and better on
8 of those 10 cells.

### Final ladder, 40/40 cells, one config per concurrency

C1/C4 from the realtime profile with MULTISTEP=0; C16/C32 from high_concurrency (16 slots).

| model | TTFT | TPOT | p99 ITL | tok/s | E2E |
|---|---|---|---|---|---|
| Gemma-4-12B | 13-7 | 8-12 | **15-5** | 4-16 | 4-16 |
| Gemma-4-26B-A4B | 11-9 | 3-17 | **13-7** | 0-20 | 0-20 |

Against the first pass, the MULTISTEP flip moved p99 ITL from 9-11 to 15-5 (12B) and 8-12 to 13-7
(26B) at no cost in TTFT or TPOT. Nothing moved TPOT or tok/s, because nothing in this round
touched decode attention, which is where the whole TPOT gap lives.

**The standing goal is not met.** plow wins TTFT and now p99 ITL on the majority of cells, ties the
context-free part of TPOT, and loses throughput and E2E almost everywhere. The three things
standing between here and it, in order of size:

1. Decode attention per KV row: 1.28-2.0x (12B) and 1.69-4.8x (26B) of vLLM. This is the whole
   TPOT and tok/s column.
2. The C32 admission cap: needs paged (or at least non-power-of-two) sliding rings so one packet
   can hold 32 slots at 16k without paying chunk 1024 on long prompts.
3. Prefill interference at C4+ with long prompts: p99 ITL 104-214 ms against vLLM's 8-13 at
   1024-8192 in, where the wave is already gone. That is the prefill chunk blocking decode.

## Correction: the decode step is at parity; the gap is not KV traversal (2026-09-23)

The "per-KV slope" above was derived from `tpot_ms`, which is the MEAN inter-token time and so
includes every decode step a prefill launch blocked. Splitting on `itl_med` -- the median step,
which prefill rarely lands on -- inverts the conclusion:

```
median ITL, plow / vLLM        128    1024   4096   8192  15000
12B  C1                       1.00    0.99   0.99   0.99   0.99
12B  C4                       1.00    0.99   1.00   1.01   1.02
12B  C16                      1.03    1.04   1.05   1.07   1.09
26B  C16                      1.22    1.20   1.22   1.26   1.25
```

The 12B decode step is within 9 % of vLLM everywhere and within 2 % at C1/C4. The 26B is ~22 %
behind at C16 and ~8 % at C1/C4. What `tpot_ms` was measuring is prefill blocking decode:

```
share of TPOT that is interference    plow    vLLM
12B  4096/C4                           20 %     9 %
12B  8192/C4                           35 %    18 %
12B 15000/C4                           53 %    38 %
26B  8192/C4                           33 %    11 %
26B 15000/C4                           50 %    28 %
```

**Do not use `tpot_ms` to reason about kernel speed on a continuously-batched server.** Use
`itl_med` for the step and `tpot_ms - itl_med` for the interference.

### Null result: capping the per-tick prefill budget does not fix it

`PLOW_PF_INTERLEAVE` is plow's `max_num_batched_tokens` (config.rs: unset -> 2048 on CUDA, `0` ->
`usize::MAX`, i.e. uncapped). Every campaign profile sets `0`, so once a slot is decoding a tick
may still admit unbounded prefill rows before running decode. A cold tick bypasses the cap, so
capping should be free for first-request TTFT.

Measured 0 -> 2048, both models, both profiles, 40 cells: **E2E better on 12, and every one of
those is inside the cell noise.** TPOT barely moves and TTFT gets worse at long inputs (26B
15000/C16 2417 -> 2702 ms, E2E 1.570 -> 1.730). The knob is not the lever; do not re-try it.

The reason it cannot be: at C16/C32 with long prompts the machine is saturated with prefill work,
and vLLM shows 80 % interference in the same cells. Interleaving differently moves latency between
TTFT and TPOT rather than creating throughput.

### What the wall-clock arithmetic says instead

12B 4096/C16, 64 prompts, 8192 output tokens:

```
                       plow        vLLM
wall (from tok/s)      18.2 s      16.2 s
decode  512 steps x    13.38 ms    12.78 ms   =  6.85 s  /  6.54 s
prefill 262k tokens                           = 10.8  s  / 10.9  s   (at each stack's C1 rate)
serial sum             17.7 s      17.4 s
```

plow's measured wall matches its serial sum; vLLM finishes 1.2 s BELOW its own serial sum. So vLLM
overlaps decode with prefill and plow largely does not -- even though `token_batch` is loaded and
its route fires on all four runs. That is the thing to measure next, with `PLOW_PF_PACKLOG=1`
(`PACKLOG WALL prefill_ns/decode_ns/ticks`), not to guess at.

## Runtime scheduling knob audit (2026-09-23, user ask)

Cross-referenced all 210 `rt.*` registry knobs against what every shipping Gemma-4 recipe sets
(`[serve.env]` plus each profile's `serve_env`). **A knob every recipe overrides is a wrong
default** -- the default is whatever nobody wants. Script: `knob_audit.py`.

### Already right

`rt.token_batch` ON/PROMOTED (mixed prefill+decode batching; serve log `token_batch: true`),
`rt.prefix_cache` ON/PROMOTED (recipes set `0` ONLY to match vLLM's `--no-enable-prefix-caching`
in the ladder -- production is cache-on), `rt.idle_dispatch`, `rt.block_packets`, `rt.pf_modular`.

### Always overridden -> wrong default

| knob | default | every recipe | evidence |
|---|---|---|---|
| `rt.pf_interleave_adaptive` | OFF | 1 | 12B 1024/C4 TTFT 167.0 -> 107.7 ms; 26B 114.5 -> 94.4 |
| `rt.rung_fast_probe` | OFF | 1 | 128/C16 P99 TTFT 170 -> 150 ms |
| `rt.multistep_adaptive` | OFF | 1 | 26B C1 TPOT 5.66 -> 5.59; 15000/C4 TTFT 700 vs 993 ms |
| `rt.queue_ttl_ms` | UNSET | 0 | |

### The two the cross-reference cannot see (no recipe touches them)

* **`rt.pf_cover` = ON, PROMOTED.** It selects the OLD covering chunk pick, so the cost-aware DP
  cover in `pick_prefill_bucket` never runs. `docs/flags-reference.md:959` documents the default as
  **off**; the registry says ON; the serve log settles it (`pf_cover: true` on every run in this
  campaign). This is the prefill padding bug: a 2331-row tail takes the whole 4096 rung.
* **`rt.multistep` = 8.** Pins p99 ITL at 8 x TPOT out of the box (median ITL `0.000`); both
  campaign profiles override to 0. Measured 4 -> 0 this session: p99 ITL 41.8 -> 10.5 ms at zero
  TTFT/TPOT cost. A global flip to 0 is NOT right -- on AMD/CPU the host gap multistep amortises is
  real. It should be engine-conditional like `rt.mux_inline_tick` ("unset = on for a CUDA engine;
  AMD and CPU keep the engine thread"), since the inline tick already cut the CUDA dispatcher
  handoff from 50-87 us to 0.3 us, which is the reason multistep existed there.

### Deliberately not flipped

`rt.decode_pipeline` -- only this session's 12B realtime profile sets it, and it is unavailable
whenever cuBLASLt is on, so the evidence does not support a global default.
`rt.pf_interleave` -- every recipe sets 0 (uncapped) against a 2048 default, but the measured A/B
of 0 vs 2048 is a wash on 40 cells, so neither value is clearly right.

Each flip needs a checkpoint-P certificate or the merge gate fails, and a knob the verifier rejects
goes back to opt-in. Certificates are measured with the arm value pinned explicitly, so they stay
valid whichever way the registry points; the registry is only flipped for knobs that pass.

## FP8 campaign: the dtype gate silently drops production defaults

Verified in code, 2026-09-23, while briefing the FP8 agents.

`apply_production_defaults` (`crates/devgen/src/lib.rs:7568`) gates the entire GEMMA4_HOPPER
block on `bf16 && capabilities.gemma && capabilities.full_attn_hd512 && arch == "sm_90a" && tp == 1`.
Any FP8/W8A8/W8A16/MXFP4 emit therefore ships WITHOUT `sliding_ns_grid`, `sliding_ns_cap`,
`gemv_prefetch` (dense) / `moe_pf_lt` + `moe_dec_lt` + `gemma_moe_dec_group=4` (MoE),
`attention_decode_balance_gf=4`, `seg_fa512`, `seg_fa256_gqa2` — with no assert and no warning.
The `prefill_cublaslt` / `no_glu_fuse` pair at :7557 is gated the same way, but those two have no
FP8 equivalent (`lib.rs:7956` hard-asserts cuBLASLt prefill off the BF16 path), so their absence is
correct; W8A8 uses the fused GLU role instead.

Three of the dropped knobs are genuinely dtype-irrelevant — attention runs BF16 either way because
`kv_cache_scheme` is null in both FP8 checkpoints. `gemv_prefetch` is NOT: it is the decode GEMV L2
prefetch, its comment records it as measured on the dense 12B in BF16, and under W8A16 decode the
GEMV reads FP8 weight bytes, i.e. half the footprint the prefetch distance was tuned against. It
must be A/B'd on an FP8 packet, not restored on the assumption that dtype does not reach it.

The same silent-gate shape appears in the decode emitter: the grouped MoE decode arm
(`gemma_moe_dec_group` / `moe_dec_lt`, `lib.rs:6083-6100`) exists only in the bf16 branch, so a 26B
FP8 packet falls back to per-slot `MoeExpertGluGemmaFp8` GEMV with no `MoeAlignGemmaPf` and no
diagnostic.

Fixed for the campaign at the RECIPE level, not in the gate: flipping a production default needs a
checkpoint-P certificate, and it would have to be certified on an FP8 cell that does not exist yet.
Whether the gate should key on family+arch rather than dtype for the attention-only knobs is a
follow-up for the user, with the above as evidence.

### FP8 emit limits found (26B-A4B)

* W8A8 emit succeeds: 2.5 min CPU, 24.8 MB packet, 610+610 `MoeGroupGluGemmaPfW8a8` /
  `MoeGroupDownGemmaPfW8a8` prefill ops. `perf-data/tools/quantize_fp8.py` already handles the
  fused `experts.gate_up_proj` / `down_proj`.
* It only emits with `PLOW_EMIT_PACKED_PREFILL` UNSET. With it on, emit panics two ways at
  `lib.rs:9774`: with `TMA_GEMM=1`, "FP8 GEMM tensor-map operands disagree with direct operands:
  GemmFp8"; with `TMA_GEMM=0`, "opcode has no audited direct-operand access contract:
  MoeGroupGluGemmaPfW8a8". So a 26B FP8 packet cannot carry packed prefill today.

### max_ctx is a ceiling, not a bind

`plowc --max-ctx` still sizes `in.pos` (which IS the runtime's bound, `gpu.rs:4955`), the
full-attention KV caches (`kv_ring` returns `(ctx, MASK_NONE)` only for `window == 0`; sliding
layers ring at `next_pow2(window + chunk - 1)`, ctx-independent — so on Gemma-4's 5:1 pattern ctx
sizes one layer in six), and clamps the rung ladder (`appended_rungs` caps at
`ctx.min(max_chunk(window))`).

But the runtime re-declares it at load (`crates/plowrt/src/exec/ctx_bound.rs`, entered at
`gpu.rs:3337-3392`). `PLOW_LIVE_CTX` narrows (mature; `devgen::mla::ctx_bound_tests` asserts a
narrowed 1M-ceiling GLM emit equals a native one) or widens (v1, NV-dense only, requires
`PLOW_VMM_LIVE=1` or `PLOW_VMM_PREFIX=1`; `crates/plowrt/tests/differential_widen.rs` widens 12B
8k->32k and asserts instruction-level equality with a native 32k emit). Widen refuses on DSA/indexer
ops, on `widest prefill bucket == ceiling` (the ladder was ctx-clamped), on
`ceiling < kv_ring_rows(window, chunk)`, on baked (non-recipe) RoPE, and on any cache with no
scaling rule.

`PLOW_RT_MAX_CTX` only LOWERS: `gpu.rs:3395` rejects `rt_ctx > packet_max_ctx` outright, and
`gpu.rs:4956` clamps with `.min(packet_max_ctx)`.

Neither `rt.live_ctx` nor `rt.vmm_live` is set by any recipe, and the widen test skips unless local
packets or the hf-cache are present — so this campaign has never exercised widening. For the FP8
arm we re-emit at `max_ctx = 16384` instead, because widening swaps the KV allocator to VMM and
would break apples-to-apples against a BF16 arm on contiguous rings.

## #51 finer prefill rungs: the padding fix works, the 2560 rung wedges — REVERTED

A/B on 2026-09-23, one variable: `PLOW_PF_LADDER_APPEND` gains 1536/2560/3072/3584 on the 12B
bf16-ladder16k recipe (packet `l12r`). Same binary, same everything else.

**The bucket arithmetic worked exactly as designed.** At 15000/C1 the picks went
`4224, 4224, 4224, 2328 -> 2560` against the old `4224 x3 + 2328 -> 4096`: useful 15000, launched
15232, so padding fell **10.6% -> 1.52%**.

**Then it wedged.** The 2560 pick is the last line in the packlog; plowrt then span at 99.6% CPU
(3535 s CPU in 3547 s wall) with the bench client blocked at 18 s CPU, holding the GPU lease and
the CPU-quiet lock for 59 minutes until killed. No fault, no diagnostic — a host-side spin.

It is NOT a missing program. `assets/build.json` shows 2560 fully emitted and wired:
`shapes.prefill_buckets = [128,256,512,1024,1088,1152,1536,2048,2560,3072,3584,4096,4160,4224]`,
`modular_pipeline.prefill_rungs` the same, and `dispatch_table` entries at index 8 for both
`prefill:dense_attention:2560` and `prefill:dense_ffn:2560`. So the bucket pick found a real rung
and the wedge is in EXECUTING it.

The first three 4224 chunks of the same request completed, and the old packet also runs 15000 as
four chunks, so chunk-chaining is not the variable. Geometry checks out too:
`kv_ring_rows(1024, 2560) = 3583 <= 8192` ring, and 15000 <= max_ctx 16384.

Pattern worth testing before anyone re-attempts this: every rung that has ever worked is a power of
two or `pow2 + {64,128}` (1088, 1152, 4160, 4224). All four NEW rungs are `pow2 + {512,1024,1536}`.
Cheap repro to isolate it, when the card is free: serve `l12r` and send ONE ~2500-token prompt, so
bucket 2560 runs as a single chunk with no chaining.

**Reverted** — the recipe is back to `256,1088,1152,4160,4224` and was never committed.

**This does not block the objective.** The cost-aware DP cover (`PLOW_PF_COVER=0`, #52) targets the
same 15000 padding with only EXISTING rungs: it should cut the 2331-row tail as `[2048, 512]` =
2560 launched rows — the identical 1.5% padding — without introducing a 2560 rung at all. That A/B
is running. If it wins, #51 is closed by #52 and the finer rungs are unnecessary.

### Root cause of the 2560 wedge: plowc silently truncates the Gemm segment set

Attributed from `l12r/assets/build.json` alone (CPU, no GPU needed —
`$CLAUDE_JOB_DIR/tmp/sched/rung_diff.py`).

Per-rung program counts split perfectly along rung shape:

| rung | segments | shape |
|------|----------|-------|
| 128, 256, 512, 1024, 2048, 4096 | 571 | pow2 |
| 1088, 1152, 4160, 4224 | 571 | pow2 + {64,128} |
| **1536, 2560, 3072, 3584** | **435** | **pow2 + {512,1024,1536}** |

The deficit is entirely ONE group — `kind=prefill topo=ordinary arms=('Gemm',)`, 329 programs at
2048 against 193 at 2560 — and the missing segments are the contiguous tail **435..570**, exactly
136 of them, identical for all four new rungs. Instruction count is unchanged at 766, so this is a
truncation of the emitted program set, not a different lowering.

`appended_rungs` (`lib.rs:3105-3117`) admits any appended rung that satisfies
`x <= cap || (window > 0 && x <= ctx && window + x - 1 <= ring)`. There is no pow2 or tile-shape
requirement, so the ladder accepts a rung whose Gemm segments the emitter then cannot fully cover,
and emits it anyway with no assert and no warning. The runtime's dispatch table gets an entry for
2560 (verified present for both `dense_attention` and `dense_ffn`), dispatches into the truncated
chain, and spins on a completion that never arrives.

`plowbench-doctor.sh` does not catch it: it verified the packet hash and "18 cubin(s), arch=sm_90a"
and reported `RESULT: clean, with 2 warning(s) — safe to lease`. It checks the OBJECT set, not
per-rung program completeness.

So there are two defects, and the second is the dangerous one:

1. Emit truncates the Gemm segment set for rungs that are not pow2 or pow2+{64,128}, silently.
2. Nothing between that and a served request validates rung completeness — not the emitter, not the
   doctor, not packet load. The failure mode is a 99.6%-CPU host spin holding a GPU lease, which is
   the worst possible shape for a leased-GPU campaign.

Cheapest guard, and it needs no kernel work: every prefill rung in a packet should carry the same
segment count, so assert that parity at emit (or in the doctor's artifact stage). That converts a
59-minute silent lease burn into a build-time refusal. NOT implemented here — recorded for the
user, because it is a production-emit change outside this campaign's scope.

## FP8 kernel selection: plow-native only, and it is the arm BF16 rejected

User question, 2026-09-23: "are we cublas or plow native ... based on actual run plow is picking
the kernels". Answered from the BUILT packet (`/opt/dlami/nvme/tmp/fp8-campaign/p12fp8a/assets/
build.json`, `$CLAUDE_JOB_DIR/tmp/sched/fp8_kernels.py`), not from the recipe.

plow FP8 emits **zero** cuBLAS/cuBLASLt. Arm inventory of the 12B W8A8 packet (2416 programs):
`QuantFp8` 960, `GemmFp8` 912, `RmsNorm` 486, `NormResidual` 480, `HeadNormRope/hd256` 201,
`FlashPrefill/hd256` 200, `Glu` 192, `GemmGluFp8` 48, `Gemm` (BF16) 5, plus one each of
`GemvFp8` / `GemvGluFp8` / `FlashDecode` / `FlashMerge` for the B=1 decode. No `lt_algos`, no Lt
glue.

It is structurally forced, not a selection. `lib.rs:7950` asserts `prefill_cublaslt` requires
`!any_fp8_weights() && !mxfp4` — "cuBLASLt prefill emission requires Gemma 4 BF16 on single-GPU
SM90" — and the BF16 `gemma4_sm90_gemm_glu_role` carries the same guard. FP8 routes to its own
`gemma4_sm90_w8a8_gemm_glu_role` instead.

**Consequence: the FP8 prefill path is the arm BF16 measured as SLOWER.** `LT_GLU_QUALIFIED`
records that gate/up as two cuBLASLt GEMMs plus a GeGLU pass beat the fused GLU role at every
bucket Lt covers, which is exactly why `prefill_cublaslt` and `no_glu_fuse` are production
defaults in BF16. FP8 cannot reach that route. On top of that it pays ~one `QuantFp8` per GEMM
(960 vs 912) for dynamic per-token activation quantization, which BF16 never pays. Add the four
decode knobs the `bf16` gate silently drops (recorded above) and the FP8 arms are functionally
complete but have had none of BF16's kernel-selection tuning — the measurement campaign only ever
ran BF16.

### The vLLM FP8 bar (measured, gate-passed)

12B C1, `--quantization compressed-tensors --kv-cache-dtype auto` (BF16 KV, matching plow):

| metric | vLLM BF16 | vLLM FP8 | FP8 gain |
|--------|-----------|----------|----------|
| TPOT | 10.56 ms | **7.38 ms** | 1.43x |
| TTFT @128 | 30.04 ms | 30.19 ms | — |
| TTFT @15000 | 671.8 ms | **548.1 ms** | 1.23x |
| tok/s @15000 | 63.6 | **86.0** | 1.35x |

vLLM's FP8 KV cache is 183,113 tokens (11.18x concurrency at 16384/request) because FP8 weights
free ~12 GB — a genuine FP8 benefit, but it means the C32/15000 cell measures admission, not decode.

plow BF16 decode was at PARITY with vLLM BF16 (`itl_med` 0.98-1.09x across all 20 cells), so vLLM
FP8 at 7.38 ms now sits well under plow's ~10.4 ms BF16. Beating vLLM on FP8 requires plow FP8
decode under 7.38 ms while running `GemvFp8` at B=1 only, without the dropped decode defaults, and
with no Lt fallback. That is the real gap, and it is a kernel-tuning gap, not a feature gap.

## peak_mem is NOT a footprint comparison — vLLM's column is its preallocation

Found 2026-09-23 while reviewing the FP8 cells. vLLM serves with `--gpu-memory-utilization 0.92`
on a 79.18 GiB card, and its own startup log states the target outright: "Desired GPU memory
utilization is (0.92, 72.85 GiB)". Every measured vLLM cell then reports 72.7-73.9 GiB BF16 and a
flat 74.4-75.9 GiB FP8 — i.e. the reservation, essentially exactly, in every cell of every model at
every concurrency. It is a configured ceiling, not demand: the same number would appear serving a
far smaller model.

So the report's claim at line 245, "Memory: the 12B peaks below vLLM in every matched cell", and the
`Peak GPU memory | 72.7 GiB | 65.2 GiB` rows in the 3.x scenario tables, compare plow's ACTUAL usage
against vLLM's CONFIGURED RESERVATION. The methodology line (sec. 2, "nvidia-smi
--query-compute-apps sampled every second ... the maximum is reported") is accurate, and the 0.92
setting is stated in the baseline-server cell, but nothing connects the two, so the memory rows read
as an efficiency win they do not establish. The 26B rows ("peaks 2.0-5.2 GiB above vLLM") are
confounded the same way and are, if anything, understated against plow.

Options, for the user to pick:
1. Drop the memory rows and the line-245 bullet. Cheapest, loses nothing measured.
2. Keep them with the caveat stated inline: vLLM's figure is its 0.92 reservation (72.85 GiB
   desired), so the column is a ceiling and the comparison is not like-for-like.
3. Replace bytes with the metric that is actually comparable at a fixed card: KV CAPACITY. vLLM
   publishes it directly ("GPU KV cache size: 183,113 tokens" on the FP8 12B); plow's is
   slots x context. That is a real efficiency comparison and it is the one a reader cares about,
   because it sets how many streams and how much context each stack can hold on one H100.

NOT changed here — the report is the user's and is deliberately uncommitted.

## gpulease has no FIFO: a releasing job re-acquires ahead of hour-long waiters

`gpulease` is a bare advisory flock. A driver that loops over cell groups takes a lease per group,
and on release re-acquires in the SAME SECOND, ahead of everything queued:

    05:08:47 vllm-fp8-12b ACQUIRED (waited 846s)
    05:34:41 vllm-fp8-12b RELEASED held=1554s
    05:34:41 vllm-fp8-12b ACQUIRED (waited 0s)     <- straight back in
    05:52:56 vllm-fp8-12b RELEASED held=1095s
    05:52:56 vllm-fp8-26b ACQUIRED (waited 0s)
    05:33:29 packlog-l12-cover0-15000c1 TIMEOUT after 1800s

Combined with the 1800 s default timeout this starves every other job silently (header-only CSV, see
above). Mitigation in force: `GPU_LEASE_TIMEOUT=43200` on every queued driver, so waiters survive
the whole loop rather than dying mid-queue. A real fix would be a ticket/FIFO in gpulease; not
attempted, since it is shared tooling outside this campaign.

## campaign.py cmd_build produces an unservable object set whenever a role rewrites the packet

Found 2026-09-23 on the card; every FP8 bench died at load with
"packet/interpreter MISMATCH: the loaded cubin was specialised for packet 0x836def099237d766, but
the packet in .../assets is 0xa82f36d7771fc903" (narrow realtime rc=2, wide realtime rc=2, wide
high_concurrency rc=2, GSM8K rc=1 — all one cause).

`cmd_build` runs base emit -> objects -> role emit, and hands the objects script
`PLOW_CUBIN_CONFIG=<base>/plow_config.h`. That is only sound if the ROLE emit leaves the packet
unchanged. Verified both ways from `plow_config.h`:

| packet | base | assets | |
|--------|------|--------|---|
| BF16 `l12` | `0x691e069b80c2fa23` | `0x691e069b80c2fa23` | match |
| FP8 `p12fp8a` | `0x836def099237d766` | `0xa82f36d7771fc903` | DIFFER |
| FP8 `p12fp8b` | `0x65a07f7e459c7474` | `0xb4c7405cf7c04bc3` | DIFFER |

The BF16 roles only BIND objects, so base == assets and
`objects/interp_sm90a_pf.cubin` is byte-identical to the assets copy. The W8A8 fused-GLU role
(`PLOW_GEMMA4_SM90_W8A8_GEMM_GLU_ROLE=1` ->
`gemma4_w8a8_gemm_glu_role::apply_output_object`) REWRITES instruction sites, so the role emit
yields a different packet and the base-config object set is stale. `campaign.py packet_env` then
points `PLOW_PF_SEG_DIR` at those stale objects and plowrt correctly refuses.

The BF16 12B recipe never trips this because it enables no GLU role at all — it uses
`NO_GLU_FUSE` + `PREFILL_CUBLASLT`. So the defect is reachable only on the FP8 path, which is why
it has never been seen: there is no `perf-data/campaign/gemma4-12b.h100.w8a8*.csv` in the tree, and
the committed `w8a8-roles.toml` has almost certainly never been served end to end.

Workaround in use (recipe/flow level, no code change): re-run
`scripts/build_sm90a_gemma4_segments.sh` with `gemma_base=<assets>` and
`PLOW_CUBIN_CONFIG=<assets>/plow_config.h` into an `objects2` dir, then serve with
`PLOW_PF_SEG_DIR=<objects2>`. The script's first act is `cp $base/*.cubin $out/`, so objects2 picks
up the role-emit cubins and recompiles the extra segment objects against the role-emit config.

Proposal, NOT implemented: `cmd_build` should either pass the ROLE emit's `plow_config.h` to the
objects step, or refuse when the base and assets packet hashes differ. Today it silently produces
an unservable set. Not changed here — shared tooling, three jobs using it live, and a guard there
needs its own GPU test pass.

### CORRECTION to the 2560 root cause: not a tail truncation, a constant 136-program Gemm deficit

The earlier entry said the missing segments were "the contiguous tail 435..570". That was CIRCULAR.
Segment ids are assigned per bucket, 0..n-1: every bucket in the packet is contiguous from 0, so a
bucket with 435 programs trivially has ids 0..434 and the "missing tail" is just the id-range
difference. Verified with `$CLAUDE_JOB_DIR/tmp/sched/seg_ids.py` — `contiguous_0..n-1=True` for all
fourteen buckets.

The corrected reading is cleaner and is stronger evidence of a real defect:

| bucket | non-Gemm programs | Gemm programs |
|--------|-------------------|---------------|
| 128, 256, 512, 1024, 1088, 1152, 2048, 4096, 4160, 4224 | 242 | 329 |
| 1536, 2560, 3072, 3584 | 242 | **193** |

The non-Gemm program set is IDENTICAL (242) at every rung. Only the Gemm set differs, and it differs
by a CONSTANT 136 — the same deficit at 1536 as at 3584, independent of rung width. So these shapes
do not "run out" of emission partway; they take a different Gemm segmentation path that produces 136
fewer programs, and the packet then wedges executing one.

What this changes: the guard I proposed (per-rung segment-count parity) still works as a detector,
because the parity is exact for every healthy rung. But the FIX is not "emit the rest" — it is
finding why the Gemm segmenter takes a different path for row counts that are not pow2 or
pow2+{64,128}. That mechanism is still unidentified; it needs the segment emitter source, not
arithmetic on build.json. 329 - 193 = 136, and 193 = 48*4 + 1 is suggestive of four Gemm per layer
plus lm_head, but 329 does not divide as cleanly, so do not build on that guess.

This matters more than it did this morning: `appended_rungs` admits any rung with
`window + x - 1 <= ring`, which at ring 8192 and window 1024 means rungs up to **7169** are free at
the CURRENT KV footprint. That is the lever for the remaining 15000 TTFT gap, since the isolated
PF_COVER A/B showed the leftover 4.29% is launch count rather than padding. But there is no
pow2-or-pow2+{64,128} value between 4224 and 7169 (the next power of two, 8192, needs
`next_pow2(1024+8192-1) = 16384` ring rows = 5.0 GiB/slot sliding KV = 80 GiB at 16 slots, which does
not fit alongside 23.8 GiB of weights). So every usable wide rung lands in the broken shape class,
and this defect is now what blocks the prefill lever — and makes pending task #45 (the 8192-row
launch rung A/B) unrunnable as specified.

### SECOND CORRECTION: the Gemm program deficit is INTENDED fallback, not the defect

The 329 -> 193 Gemm-program split is fully explained, and it is not a bug. `cublaslt_prefill_bf16`
(`crates/plow-asset/src/segment_roles.rs:71`) admits a shape only if its row count is in a hardcoded
whitelist:

    CUBLASLT_PREFILL_ROWS      = [128, 256, 512]
    CUBLASLT_PREFILL_WIDE_ROWS = [1024, 1088, 1152, 2048, 4096, 4160, 4224, 8192, 8320, 12288, 12416, 16384]

The union is EXACTLY the ten healthy rungs of the l12r packet; 1536/2560/3072/3584 are absent. A
non-whitelisted rung is not Lt-eligible, so `dense_cublaslt::isolate_segments` never splits its
projections into their own Lt segments — hence 193 rather than 329 — and it runs them on the native
GEMM object instead. The constant 136 is simply the Lt-eligible projection count, which is why it
does not vary with rung width.

**That path is supported and known to work.** The comment at `segment_roles.rs:37-40` documents
exactly this case: "1088 / 1152 / 4160 are fine-grained rungs (`PLOW_PF_LADDER_APPEND`) ... Left
out, such a rung ran every projection on the native GEMM object. Measured on h100-sxm5 2026-09-21:
12B C1 TTFT at 1024 in 47.22 -> 46.82 ms on the 1088 rung". So before 1088 was whitelisted it RAN,
at 47.22 ms. Non-whitelisted means slower, not hung.

So the segment-count difference is a red herring and the earlier entries over-claimed it twice
(first as a "contiguous tail truncation", then as "a different segmentation path" implying fault).
**The wedge mechanism remains unidentified.** What is established: an un-whitelisted rung falls back
to the native GEMM object, and something in that configuration at 2560 rows spins the host at 99.6%
CPU, where the same fallback at 1088 rows was fine.

Note also that the whitelist ALREADY anticipates wide rungs — 8192, 8320, 12288, 12416, 16384 are in
it. So Lt policy is not what blocks a wide rung; the KV ring is (a 8192 chunk needs a 16384-row ring
= 5.0 GiB/slot sliding KV = 80 GiB at 16 slots).

### The discriminating experiment, cheap and not yet run

Build one packet with a rung that is NOT in the whitelist but IS the "safe" shape class — 2112
(= 2048+64) or 4288 (= 4096+192, if the ring admits it). Then:

* 2112 runs -> "not whitelisted" is NOT sufficient to wedge, and the row count itself is what
  matters. The native GEMM object has a shape constraint and the fix is in that object.
* 2112 wedges -> "not whitelisted" IS the trigger, the native-GEMM fallback is broken generally,
  and the 1088 evidence above means it regressed since 2026-09-21.

Either answer is worth one build and one 15000/C1 cell, and it decides whether widening the ladder
(the lever for the remaining prefill gap, rungs up to 7169 being free at the current ring) needs a
whitelist entry, a tuner run, or a kernel fix.

## cuBLASLt FP8 IS reachable for plow's W8A8 scheme, as-is

Phase 0 of the kernel-route work, 2026-09-23, CPU only. This overturns the assumption the FP8 arm
was built on.

CUDA in this shell: nvcc 12.9.86, libcublasLt.so.12.9.1.4. `cublasLt.h:926` defines
`CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F = 3` — "vectors are expected to have M and N elements
respectively, and each (i,j)-th element of product of A and B is multiplied by i-th element of A
scale and j-th element of B scale" — selected via `CUBLASLT_MATMUL_DESC_A_SCALE_MODE` (31) /
`_B_SCALE_MODE` (32). The enum exists only from CUDA 12.8, and cuBLAS 12.9 enables outer-vector
(channel-wide) FP8 scaling **on Hopper**.

That is exactly plow's scheme. `packet/src/dev.rs:326` documents `GemmFp8` as
`t3=a_scale(f32[M])`, `t4=w_scale(f32[N])`, dequantised `acc*a_scale[m]*w_scale[n]` in the epilogue —
per-token activation scale, per-output-channel weight scale. And the existing BF16 Lt plan
(`plowrt/src/device/cuda/lt.rs:207-260`) already passes the WEIGHT as Lt's A with `TRANSA=OP_T` and
the ACTIVATION as Lt's B, so in Lt's terms `M_Lt = n` and `N_Lt = m`. OUTER_VEC therefore wants an
A-scale of n elements and a B-scale of m elements — `w_scale[N]` and `a_scale[M]`, bit for bit, in
the layout the packet already carries. **Zero re-quantization, no checkpoint change, no scale-layout
change.** FP8 on Hopper requires TN, which the Lt plan already sets, and every 12B/26B K is a
multiple of 16.

Per shape: qkv/o_proj and unfused gate/up/down are EXPRESSIBLE AS-IS. The fused `GemmGluFp8` is not
expressible as one Lt call (Lt has no GLU epilogue) but is expressible as two Lt calls plus a Glu
pass — which is precisely the arrangement `LT_GLU_QUALIFIED` already measured as the BF16 winner.
The 26B grouped expert GEMMs are NOT expressible today: the grouped path uses an optionally-loaded
`cublasLtGroupedMatrixLayoutCreate` and plow's expert scales are per-expert `[128,N,1]`. Decode at
M=1 is expressible in principle but Hopper FP8 Lt kernels are tile-shaped for N>=8, so the heuristic
may return no algo — that one must be measured.

### There is already an in-tree measurement, and it points the same way

`runtime/nvidia/op_gemm_sm90.cuh:1303`: "cuBLASLt fp8 measures **1324-1468 TF/s** at the 12B shapes
on this box vs the 256-thread uniform body's **950-1170**: the missing structure is a DEDICATED
producer warpgroup." `docs/bringup/07-perf-campaign.md:221` records fp8 1324-1468 vs bf16 804-861.
So Lt FP8 was benchmarked on these exact shapes on this box and beat plow's then-current W8A8 body
by ~35% — that measurement is what motivated building WS384. **Whether WS384 closed the gap is
unmeasured**, and that is now the pivotal Phase 1 question rather than a speculative one.

Stale comment worth fixing: `runtime/bench/nvidia/px9_gemm_body_bench.cu:499` says "Per-tensor
scales (cuBLASLt has no per-row scale)". True when written, false under 12.9, and probably why
nobody revisited this.

### Correction: the FP8 deficit is the missing Lt route, NOT fusion

I told both agents "FP8 necessarily runs the fused-GLU arm that LT_GLU_QUALIFIED measured as
slower". That is wrong except at one bucket. Verified independently
(`$CLAUDE_JOB_DIR/tmp/sched/glu_by_bucket.py`): `GemmGluFp8` = 48 at bucket **4096 only**, and 0 at
every other bucket, in BOTH p12fp8a and p12fp8b; every other bucket runs `GemmFp8` x192 plus a
separate `Glu` x48. The role object is `gemm_glu_w8a8_sm90_gemma4_4k8k_v2` — "4k8k" is literal.

So at almost every rung FP8 ALREADY runs the split structure BF16 prefers; it just runs it on
plow-native kernels instead of Lt. The accurate statement is that BF16 sends all 336 projection
GEMMs per chunk to Lt at every shipped rung and FP8 sends zero.

### A decode route gap nobody had written down

FP8 has no `GemvQkvFp8` arm. BF16 decode fuses q+k+v into one `GemvQkv` launch per sliding layer
(i=(1,4096,3840,2048) x40 on the 12B); FP8 issues three separate `GemvFp8` launches —
`(4096,3840)` x40 plus `(2048,3840)` x80. That is **+80 GEMV launches per token** on the 12B,
entirely independent of any Lt question, and it may be a large part of whatever FP8 decode deficit
gets measured. The 26B is the same (BF16 l26 has `GemvQkv` x77).

### Corrections: the serving cuBLASLt is 13.4.1.3, and the QuantFp8 tax is 4/layer not 1/GEMM

Two things recorded above are wrong and are fixed here.

**1. The gating CUDA version.** I recorded the nix `libcublasLt.so.12.9.1.4` as what decides whether
`OUTER_VEC_32F` is available. It is not the library plow serves against. `plowrt`'s Lt loader
(`crates/plowrt/src/device/cuda/lt.rs:39`) tries `libcublasLt.so.13` FIRST, and inside `nix develop`
that resolves to **13.4.1.3** from `/usr/local/cuda/lib64`; `libcublasLt.so.12` does not resolve
there at all. So the "is CUDA new enough" risk was never live — OUTER_VEC_32F is comfortably inside
13.4. The route-matrix bench is being built and run against that same `/usr/local/cuda`, so the
measurement uses the production library rather than the nix one.

**2. The QuantFp8 tax.** I told both agents FP8 pays "roughly one QuantFp8 per GEMM (960 vs 912)".
Those were per-PACKET program totals summed across five buckets, not launches, and the real
structure is different. Counted from `kernel_cases` for ONE 128-row prefill chunk of the 12B:

| | launches |
|---|---|
| BF16 (l12) | **766** — Gemm 329, HeadNormRope 144, RmsNorm 97, NormResidual 96, Glu 48, FlashPrefill 48, + 5 head ops |
| FP8 (p12fp8a) | **958** — GemmFp8 328, **QuantFp8 192**, HeadNormRope 144, RmsNorm 97, NormResidual 96, Glu 48, FlashPrefill 48, Gemm 1, + 4 head ops |

Exactly **+192 launches, +25.1%**, and the 192 is `4 per layer` at fixed sites — attention input,
MLP input, GLU output before down, attention output before o_proj — NOT one per GEMM. Against 328
GEMM launches that is 0.59 quantizes per GEMM, so my ratio overstated the per-GEMM tax while
understating how cleanly it attributes.

The same count states the Lt deficit exactly: of BF16's 329 `Gemm` launches, **328 are projections
whose (N,K) are all in `CUBLASLT_PREFILL_GEMMA4_SHAPES`**, so at every shipped rung all 328 go to
cuBLASLt and only `lm_head` stays native. Of FP8's 328 `GemmFp8`, **zero** can.

### Open, with evidence, not chased: the 26B fuses GLU at every rung and the 12B does not

On the 26B FP8 packet the dense-MLP gate/up is fused at EVERY rung (`GemmGluFp8` M=128 N=2112
K=2816 x30 at bucket 128); on the 12B FP8 packet it is split at 128/512/1024/2048 and fused only at
4096. Both packets have `no_glu_fuse=false` and `gemma4_sm90_w8a8_gemm_glu_role=1`. Both fused-GLU
role objects declare `min_rows=4096 max_rows=8192 n=15360 k=3840` — 12B geometry — so the 26B's
fusion cannot be the role object and must be the GENERIC `GemmGluFp8` interpreter arm. Why the 12B
does not also take that generic arm below 4096 is unanswered; an arena or tile constraint at
N=15360 K=3840 is the obvious suspect but is unverified. Consequence that matters now: "fused vs
split" means different things per model, so the two models' FP8 prefill results are not directly
comparable on that axis.

### CORRECTION: the 1324-1468 vs 950-1170 TF/s pair is from a GH200, not this H100

Recorded above as "already measured in-tree ... on this box", and repeated to the user twice. Wrong,
and the error is worth understanding because the source comment causes it.

`runtime/nvidia/op_gemm_sm90.cuh:1303` reads: "cuBLASLt fp8 measures 1324-1468 TF/s at the 12B
shapes **on this box** vs the 256-thread uniform body's 950-1170". The only other occurrence of that
pair, `docs/bringup/07-perf-campaign.md:221`, attributes it explicitly and warns against exactly the
use I made of it: "one recorded run on **GH200**/12B measured fp8 1324-1468 / bf16 804-861 TF/s,
`perf-data/gemma12b-gh200-prefill-campaign.md`; that is *that* box's ceiling, **not a target for
yours**." That perf-data file does not exist in this tree, and `nvidia-smi` here reports
`NVIDIA H100 80GB HBM3`.

So "did WS384 close the ~35% gap" is NOT answerable by comparing H100 numbers against 1324-1468.
GH200 and H100-SXM5 differ in clock and memory system and absolute TF/s does not transfer.

The answerable question, and the one P1a actually measures, is the RATIO on one box on one day:
plow-WS384-fp8 against cuBLASLt-fp8 at matched shapes. Ratio >= 1.0 kills the FP8-Lt thread on this
hardware whatever a GH200 once read; ratio ~0.7 (the shape of the historical gap) makes it live.
Absolute TF/s to be reported alongside, labelled H100 80GB HBM3.

**The comment at `op_gemm_sm90.cuh:1303` should say GH200, not "this box".** It is a one-word source
fix, it is not in this campaign's scope, and it will mislead the next reader the same way until
someone makes it. Added to the proposals list.

## A certificate arm was contaminated by an orphaned unlocked build — re-run queued

2026-09-23. The first `cert-rt_pf_interleave_adaptive` ctrl arm is not trustworthy and is being
re-measured. Recorded because a certificate gates the merge and its provenance has to be auditable.

Sequence, from the lease log and the arm's own `server.log` / `run.log`:

* 06:34:03 ctrl acquires the GPU lease, then sits at 0.0% CPU for ~11 min waiting on the CPU-quiet
  lock, which a concurrent packet rebuild held SHARED. Lease held, nothing running.
* 06:45:03 plowrt starts; 06:45:15 server ready; ~06:45:24 coherence gate PASS ("The capital of
  France is Paris.").
* ~06:45:25-06:46:30 the two timed cells run (1024/C1 then 1024/C4, 32 prompts each).
* ~06:45:45-06:46:30 an ORPHANED `nvcc` build runs with NO lock at all — a `campaign.py build` child
  that survived its parent being killed.

So the unlocked build overlaps BOTH timed cells, not merely model load. The arm's own numbers are
consistent with it: 1024/C4 reports `ttft 122.62` against `ttft_med 107.86`, a mean 14% ABOVE the
median, where the clean `cover0-rt` run at the same cell has the mean BELOW it (101.38 vs 107.55). A
mean pulled above the median by a few slow requests is what a transient compile produces. That is
corroboration, not proof, but a confirmed overlap plus a consistent signature is enough.

**All four arms are being re-run, not just ctrl.** The design is ABAB so the verifier can measure
control drift across the same span as the treatment; splicing a ctrl measured 40 minutes later
against the original treat would defeat that and would be WORSE than the contaminated certificate,
because the bias would be invisible instead of known. The re-run carries an explicit
`--fact integrity:` line naming the window, so the certificate records why it exists.

Two process lessons, both now applied by the agent that caused it:
1. No multi-build scripts under one lock hold — each build takes the lock separately, so a cert arm
   waits at most one build rather than a 27-minute stage.
2. Kill the `campaign.py` PID directly, not the wrapper: `campaign.py build` spawns compile children
   that survive the wrapper's death and then run unlocked, which is exactly how this happened.

This is the second time today that killing the wrong pid caused damage (the first burned 20 minutes
of lease on the rungs A/B driver). The general rule for this host: kill the process GROUP, and
verify with `pgrep -af` afterwards rather than assuming.

## 2026-09-23: rt.pf_cover certified; FP8 has never served; the C32 lever is cheaper than planned

### rt.pf_cover is the one metric that moved (certified, `2c3d0952`)

15000/C1 TTFT **737.256 -> 703.364 ms** (floor 7.169; the second treat arm gave 703.187), prefill
padding 10.59% -> 1.58%, tpot_ms 10.541 -> 10.538 inside a 0.007 floor. `perf-certs/rt.pf_cover.json`,
`perf_cert.py verify` rc=0.

The first certificate attempt was REJECTED and the request was at fault, not the knob: it claimed a
TTFT improvement on both touched rungs, but 8192 is ITSELF a prefill bucket, so the covering pick and
the cost-aware DP cover both emit one exact-fit launch and there is no padding at that rung for the
cover to remove. Re-declared 8192 neutral with that evidence via `--neutral ttft_ms@in8192`
(`campaign.py:509-511` supports per-input-length neutrality), rebuilt from the SAME four ABAB arms --
no re-measurement to obtain a better number.

Three other flips (`rt.pf_interleave_adaptive`, `rt.rung_fast_probe`, `rt.multistep_adaptive`) were
rejected on merit and need NO code change: all three were already `OFF, OPT_IN`, so the campaign was
attempting promotion, not certifying a flip. Deltas were inside noise and the fast probe was
directionally worse (128/C16 ttft 62.754 -> 67.006). Caveat worth keeping: `campaign.py cert` can only
claim ttft_ms/tpot_ms improvement, so `rt.multistep_adaptive`, whose documented benefit is the p99 ITL
wave, cannot express its claim with the current tooling.

### The FP8 campaign has never produced a data row -- and it is two bugs, not one

Every run under `/opt/dlami/nvme/tmp/fp8-campaign/` has `gate: false` and a header-only results.csv:

| packet | how it fails |
|---|---|
| p12fp8a, p12fp8b | `packet/interpreter MISMATCH` -- **never loaded** (stale objects vs assets) |
| p12fp8c | loads, then ILLEGAL_ADDRESS on the FIRST packed prefill |

Only the second is a kernel bug. The mismatch is hygiene: each `p12fp8*` has BOTH `objects/` and
`objects2/`, and `gemma4-12b.h100.fp8-ladder16k.toml` bakes `PLOW_PF_SEG_DIR=.../objects` into its
serve replay while the assets match `objects2/`.

Bisect of the real fault (39-row prompt, bucket 128, `CUDA_LAUNCH_BLOCKING=1`): `cuGraphLaunch` ->
`cuLaunchCooperativeKernel` (`PF_SEG_GRAPH=0`) -> `cuLaunchKernel` (`+ PF_SEG_NONCOOP=1`), and still
faults under `PF_SEG_FATONLY=1`. Only the reporting API moves, so it is the KERNEL BODY. Eliminated:
graph construction and memset nodes; grid cooperation; cuBLASLt (`gemm_launches=0`); all four role
objects; and every FP8 attention arm (`kv_dtype` is bf16 on both head dims). Since `GemmGluFp8` exists
only at bucket 4096, the remaining suspects are **QuantFp8 (1920 instances) or GemmFp8 (1872)**.
compute-sanitizer is unusable here -- it fails on ANY plowrt invocation in this nix env.

**No FP8 serving latency, throughput or quality number exists.** The static build.json analysis and the
route-matrix microbenchmark are unaffected; nothing else about FP8 is measured.

### Route matrix: the FP8 prize is 1.83x and it lives in cuBLASLt (`fe938458`)

Every dense projection shape of both models, both precisions, one protocol, reproduced across two
independent builds (1.83x identical both runs). `down` M=8192: Lt-BF16 1.179 -> Lt-FP8 0.645 ms;
`gate_or_up` M=8192: 1.207 -> 0.659. plow's native FP8 body captures only 1.29x of that. With zero of
the packet's 328 GemmFp8 launches able to reach Lt while 328 of BF16's 329 Gemm do, that is the whole
FP8 deficit. The OUTER_VEC (per-token x per-channel) scale mode costs 1.1-1.7x against per-tensor at
small M -- per-token scaling is not free.

### C32: item 3 needs NO kernel change, and the ring assertion is the whole ceiling

`plans/gemma4-packet-geometry.md` item 3 budgeted a row-offset field on HeadNormRope, a q-row window on
four role objects and a FlashMerge window. None is required:

* the flash body already derives rq0/qlen/slot/kvlen per request from `req[]` and computes
  `qp0 = kvlen - qlen` itself (`op_attention_sm90.cuh:356,717`), so a CLIPPED span table is enough --
  note `i[4]`/q_pos0 is overwritten on the packed path and cannot carry a stage offset;
* `d_headnorm_rope` already skips a masked row (`op_norm.cuh:781`, `:985`), so a per-stage SLOT MASK is
  enough for the K/V norm.

That keeps the change out of the fat `pfpackedseg` object at the 255-register cap. The 16-slot ceiling
is one line in `packed_prefill::Manifest::validate`:
`cache.stride >= cache.window + write_rows - 1` with `write_rows = max_request_rows.or(rows)` -- a
4096-row chunk demands an 8192-row ring (2.5 GiB/slot, 16 slots). `stage_rows` makes it the STAGE width:
2048 ring rows, 640 MiB/slot, 32 slots at the chunk-4096 TTFT. Within one cooperative launch the
protection is the WAR ordering (`HNR_{i+1}` after `FP_i`), not the launch boundary, which is why the
field is gated on stages actually being bound.

Landed and tested (`19932839`, `61d29498`, `247dae64`): `plan_stage`, `stage_slots`, `stages_needed`,
`Manifest::write_rows` with its gate, and a `bind_request` guard so load-time binding cannot repoint a
staged site at the whole-chunk tables (silent wrap, wrong tokens, no fault). 22 tests.

**Not done: the devgen emit loop (HNR_i -> FP_i with Dep::Coarse), the runtime per-stage table fills,
the packet build and the C32 cells.** Nothing is measured yet.

### Where the comparison actually stands

8192/C32 is **4291 ms vs vLLM 3648** -- unchanged this session. The 32-slot `c32-req1k-16k` packet is the
best plow packet at C32 for >=4096-token prompts (8192/C32 7552 -> 4291, 15000/C32 13646 -> 8271) but
still loses that cell, regresses C16 long prompts by 66-72%, and pays a constant +42 ms per lone
128-token arrival at C16 (still unattributed). vLLM is not beaten on all metrics, and FP8 cannot be
compared at all until the prefill fault is fixed.


## Decode rung 32: the occupancy route is closed by arithmetic

The B=32 decode knee (kernel-only, p12rq, ctx=128: 11.659 ms at B=16 -> **14.261** at B=32, marginal
0.069 -> 0.163 ms/row against vLLM's ~0.049) was attributed to warp occupancy. A megakernel unions every
arm's register demand, the decode entry sits at REG 255, and `threads/SM = 65536/R` puts decode at ~8
warps/SM -- 1.67 TB/s at B=32 against vLLM's 2.05 TB/s (50% vs 61% of the 3.35 TB/s peak). The remedy
would be a GEMV under 128 registers. Both halves are now measured, and both fail.

**Dropping arms does not lower registers.** `PLOW_NV_LEAN_DECODE=1` compiles the flash arms out:
SHARED 44048 -> 14480 (3x), REG 255 -> **255**. The MM=32 GEMV walk owns the ceiling, not
`d_flash_decode<512>`. The `PLOW_NV_LEAN_DECODE` contract at `interp_sm120.cu:594` promises "2-3
blocks/SM" against a 208-reg ceiling it attributes to the flash arms; that attribution does not hold
at MM=32.

**Forcing the cap works, and costs more than it buys.** `PLOW_NV_FORCE_MINBLK=2` is the
`__launch_bounds__(256, 2)` that makes ptxas target 2 blocks/SM. It lands exactly REG 128, and
STACK 384 -> 1216. Both arms are decode cubins from identical source differing only in that define,
run on p12rq via `PLOW_NV_CUBIN`; `PLOW_NV_MINBLK` appears only in `__launch_bounds__`
(`interp_sm120.cu:3046`) and never feeds the arena, so the A/B is single-variable.

| B | mm32 (255 reg, 8 warps/SM) | + MINBLK=2 (128 reg, 16 warps/SM) | speedup |
|---|---|---|---|
| 1 | 13.569 | 18.099 | 0.750x |
| 8 | 11.697 | 19.027 | 0.615x |
| 16 | 12.672 | 24.366 | 0.520x |
| 32 | 16.820 | 28.609 | 0.588x |

Doubling occupancy makes decode **1.3-1.9x slower at every batch**. The spill is +832 B/thread = 208
registers, so the walk's live state is **~463 registers**: fitting 128 natively means cutting live
state 3.6x, which is a different algorithm rather than a flag. And there is no intermediate --
`threads/SM = 65536/R` is invariant to block size, so at 256 threads 2 blocks/SM requires R <= 128 and
R=168 still yields 1 block/SM. The cliff is binary and the only available step loses.

(The generic-build bench reproduces across sessions to three decimals -- mm32 B=16 12.674 then 12.672,
B=32 16.825 then 16.820 -- so these deltas are real. Generic builds lack the packet-geometry defines
and so are slower than the stock packet-paired cubin in absolute terms; only the ordering is valid.)

### What role-segmenting decode would and would not buy

Prefill is already role-segmented: `PLOW_NV_SEG_GEMM` is documented at `interp_sm120.cu:526` as
"Design A: give the GEMM/tier-A segments their OWN kernel object, targeting occupancy 2 ...
__launch_bounds__ caps registers at 128 ... Flash segments keep the occ-1 `_pfseg` object." Decode is
the one path that never got the equivalent: one cubin carries gemv + flash-decode + norms + sampling
for every rung 1..32, and the `seg` field is a single id across all 540 instructions of prog[13].

The infrastructure is wired end to end. `DecodeProgramObject { index, rows, object }` in
`plow_asset::decode_objects` maps each decode program to its own cubin, each `DecodeObject` carries its
own `threads`/`arena_bytes`/`grid`, and `gpu.rs:3635-3723` selects and binds them. p12rq binds ONE
object for all rungs; devgen never emits more than one.

The launch cost is affordable. prog[13] (T=32, 540 insts) is 48 x (9-op GEMV run + 2-op flash run) =
**97 contiguous role runs**, so segmenting by role costs 97 launches = 0.29-0.78 ms at 3-8 us each =
2.0-5.4% of the 14.261 ms step, against a 23% gap.

But segmenting **alone buys nothing**, because it does not lower R -- that is exactly what the lean
build measured. It only pays combined with a GEMV that is natively low-register. The prefill precedent
survives its 128-reg cap because the GEMM tile was designed for occupancy 2; the decode MMA walk was
not, and squeezing it spills 208 registers.

**Status: seven levers refuted** (GV_MM_MAX=16, MMA_UNB=6, NOSTAGE, the cuBLASLt route, register
pipelining, arm-dropping, the MINBLK register cap). What remains is not a knob: either a decode GEMV
whose live state is ~3.6x smaller, or per-op launches where each kernel carries only its own register
budget. Both are projects to scope, and neither is measured.


## FP8 prefill fault: the masked-padding guard was compiled out (FIXED)

Every FP8 packet was `gate:false` with zero data rows, and p12fp8c faulted with
`CUDA_ERROR_ILLEGAL_ADDRESS` on its first packed prefill. That blocked the whole FP8 campaign.

`build_sm90a_gemma4_segments.sh` derived the masked-padding define from the precision:

```sh
gemma_bf16=$((1 - gemma_w8a8))
gemma_masked=${PLOW_BUILD_MASKED_PADDING:-$gemma_bf16}
```

so `PLOW_NV_MASKED_PADDING` went into every BF16 object and **out of every FP8 one**. It guards
the packed-prefill KV write:

```c
#if defined(PLOW_NV_MASKED_PADDING) && PLOW_NV_MASKED_PADDING
        if (out_stride && pfslot && pfslot[t] < 0) continue;            /* op_norm.cuh:780 */
#endif
        ...
        pfslot ? ((size_t)((unsigned)pfslot[t] * nhead + hh) * out_stride + ...)  /* :806 */
```

A negative slot casts to a huge `unsigned`, so `obase` runs off the KV cache. The guard is live
only when `out_stride != 0` -- exactly the bisect boundary, where instruction 6 (q, dense, stride
0) is clean and instruction 7 (the first KV write) faults.

**The guard has been load-bearing for bf16 all along.** p12fp8c has `max_request_rows` ABSENT
(verified by scanning `model.pkt`), and `packed_prefill.rs:585` asserts that on an unmasked plan
"the padding rows carry a real slot". They do not -- otherwise restoring the guard could not have
changed anything. `Manifest::validate_object` cannot catch it either: it demands the
masked-padding capability only when `max_request_rows.is_some()`. So every BF16 packet has been
protected by a define that FP8 happened to lack. Why unmasked plans carry negative padding slots
is still open, and the guard masks it.

Fix: `gemma_masked_def=${PLOW_BUILD_MASKED_PADDING:-1}` now carries the DEFINE at the two packed
object sites, precision-independent; `gemma_masked` still gates the hd512/hd256 ROLE OBJECTS,
which are a separate bf16-only default.

Proof, p12fp8c rebuilt with the default environment and no override:

```
input_len,concurrency,ttft_ms,...,out_tok_s,req_per_s,ok_reqs,gen_toks,peak_mem_mib
64,1,24.00,24.00,8.670,8.670,8.600,8.670,8.800,113.8,0.890,1,128,56648
coherence gate: PASS        "gate": true
```

First FP8 packet to serve on this host. `pfpackedseg` from the fixed default build is byte-identical
to the hand-forced one (`867ab982422711c6`) and differs from the faulting original
(`2f5fdf10df847d53`), while the FP8 object set keeps its bf16-only role objects off (13 cubins vs
14 when masked padding also turns the roles on).

Unblocks the FP8 campaign. Still missing for it:
`perf-data/campaign/gemma4-12b.h100.reference-vllm028-fp8.csv` does not exist, so `campaign.py
bench` on the FP8 recipe raises `FileNotFoundError` in `compare()` AFTER the bench itself
succeeds -- the vLLM FP8 baseline has to be measured first.

Method note: the bisect localized this to instruction 7, and masked padding was then dismissed BY
ARGUMENT ("-1 slots only appear with `max_request_rows`"). The argument was wrong, and operand
dumps could not have settled it either -- `pfslot` contents are runtime-filled and the host
patches `t6` at load (`op_norm.cuh:744`), so an encoded `t6=65535` does not mean nullptr at run
time. One rebuild with the define flipped was cheaper and decisive.


## Where the comparison actually stands, cell by cell (2026-09-23)

Scored from the committed ladders (`perf-data/campaign/*.csv`, plow cells at `453cf4c1`, vLLM
0.28 reference at `a3ea207e`) with the last run of each cell. A cell is a WIN only if plow is
better on ALL FOUR of TTFT, TPOT, p99 ITL and out_tok_s.

| config | cells winning all four |
|---|---|
| 12B ladder16k (16-slot packet) | **3 of 20** -- 128/C1, 1024/C1, 4096/C1 |
| 12B c32-16k (32-slot packet) | **0 of 20** |
| 26B ctx16k | **0 of 20** |
| 26B c32-16k | **0 of 10** |

**A correction to an earlier claim in this file.** "plow wins 8192 at C1/C4/C16" was TTFT-only.
Scored on all four metrics 8192/C4 and 8192/C16 LOSE: TPOT 17.2/49.6 against 13.5/37.5 and
out_tok_s 177.5/269.5 against 188.4/301.1. TTFT is plow's strength and it does win broadly at
C1-C16; it is not the whole scorecard.

### The three systematic deficits

1. **Request throughput at C>=4.** out_tok_s loses in essentially every C>=4 cell, even where
   TTFT wins by 1.5-2x.
2. **Decode TPOT under batch x context.** At C1 TPOT is at PARITY at every context
   (10.4/10.5 at 128 ... 10.6/10.6 at 15000). It degrades only with concurrency: at C16,
   +7% at 128, +18% at 1024, +33% at 4096, +32% at 8192, +25% at 15000. So this is batched KV
   traversal, not per-token weight streaming.
3. **p99 ITL at short context on the 32-slot packets** (43 ms against 11) -- the MULTISTEP wave.

### Effective concurrency is the throughput story, and the 32-slot packet already wins 3 of 5

Effective streams = out_tok_s x TPOT. It says how many sequences were really in flight, which is
what separates a real throughput win from under-subscription.

| in | C32, 16-slot pkt | C32, 32-slot pkt | vLLM |
|---|---|---|---|
| 128 | 15.0 | **29.9** | 29.0 |
| 1024 | 14.4 | **26.4** | 24.6 |
| 4096 | 14.1 | **23.3** | 22.7 |
| 8192 | 13.7 | 13.6 | 22.4 |
| 15000 | 13.3 | 8.0 | 22.7 |

The 16-slot packet is pinned near 14 at every length -- that is the slot cap, and its flattering
C32 TPOT (50.7 against vLLM's 67.5 at 8192) is purely under-subscription. The 32-slot packet
BEATS vLLM at 128/1024/4096 and then collapses at 8192 and 15000. **That collapse is the single
highest-value target left**: at 8192/C32, restoring eff 13.6 -> 23 takes out_tok_s from 269.6 to
roughly 455 against vLLM's 332, and TTFT falls with it because requests stop queueing.

### What the collapse is NOT

* **Not the KV admission budget.** The served log reports `per_token=16384 block_rows=2048
  max_rows=794890`, i.e. `kv_row_charge` took the full-attention-only branch (`gpu.rs:601`; the
  sliding rings are pre-allocated so they are already out of `free`). 32 x 15000 = 480000 rows
  fits inside 794890 with room to spare.
* **Not startup narrowing.** `mux capacity resolved ... capacity=32 ingress_capacity=128`, with
  `decode_rungs=[1,2,4,8,16,32] gemv_mm_cap=32 gemv_weight_passes=1`.
* **Not a harness artefact.** Both stacks run 64 requests per cell and the implied prefill rates
  are physical: at 15000/C32, plow 18150 tok/s = 436 TFLOP/s and vLLM 21600 = 518 TFLOP/s, i.e.
  44% and 52% of H100 dense BF16 peak. The comparison is sound.

So the collapse happens DURING the run, not at load. Next step is an instrumented 32-slot run at
8192/C32 that records admitted-slot count over time against KV pressure and prefill occupancy.


### Correction: C32 is NOT memory-gated (2026-09-23)

The line above -- "Concurrency above 16 is gated by packet memory, not kernels" -- and the ring
arithmetic that goes with it describe what a STATIC allocator would need. The runtime is not one:
KV is VMM-backed, `ensure_rows` maps pages on demand, and `kv_row_charge`'s own comment says
"with live rings every cache maps lazily". The `window + chunk - 1` ring is reserved ADDRESS
SPACE, not resident memory.

Measured peak memory per cell says so outright:

| in | 32-slot pkt peak | eff streams | 16-slot pkt peak | vLLM peak |
|---|---|---|---|---|
| 128 | 46.0 GiB | 29.9 | 66.1 | 73.1 |
| 1024 | 46.0 GiB | 26.4 | 66.1 | 74.0 |
| 4096 | 47.9 GiB | 23.3 | 67.1 | 74.0 |
| 8192 | 47.3 GiB | **13.6** | 68.1 | 74.0 |
| 15000 | 47.3 GiB | **8.0** | 69.1 | 74.0 |

Peak is FLAT at 46-48 GiB across every cell. It does not climb where effective concurrency
collapses, and it leaves ~33 GiB of an 80 GiB card unused while vLLM uses 74. Nothing here is
running out of memory.

**What the same data does show.** The 32-slot packet runs chunk 1024, so a 15000-token prompt
costs 15 chunk launches against 4 for the 16-slot chunk-4096 packet. At 128/1024/4096 that is
1-4 chunks and the 32-slot packet BEATS vLLM on effective streams (29.9/26.4/23.3 vs
29.0/24.6/22.7). At 8192/15000 it is 8-15 chunks and it collapses -- to the point that it is
WORSE than the 16-slot packet at 15000 (eff 8.0 vs 13.3, TTFT 26936 ms vs 13405) despite having
twice the slots. A memory constraint cannot produce that inversion; a per-chunk cost can.

So item 3 (staging) remains the right lever, but the justification changes: not "so 32 slots of
KV fit" -- they already fit -- but "so a long prompt stops paying 15 launches". The thing to
measure next is prefill launch cost against chunk count at fixed slots, not KV footprint.


### The chunk/ring trade is already measured, and item 3 is what splits it

`scripts/campaign/recipes/gemma4-12b.h100.bf16-c32-16k.toml` records the A/B in its own header:

> Chunk 2048 (ring 4096, p12c32c, 2026-09-21) fits too and halves long-prompt C32 TTFT
> (8192/C32 10314 -> 4695 ms) but the doubled ring slows every decode step (128/C16 TPOT
> 13.05 -> 17.97 ms, 1179 -> 856 tok/s), so 1024 stays. Both attention roles need the 4096 rung
> and are off.

So the collapse at 8192/15000 IS the chunk count, confirmed: doubling the chunk halves the
long-prompt C32 TTFT and brings 8192/C32 from 10314 to 4695 ms against vLLM's 3658. It is
rejected only because the same knob doubles the RING, and the ring stride costs every decode step
(+38% TPOT, -27% tok/s at 128/C16).

`PLOW_MAX_CHUNK` is one knob driving two independent things:

| | set by | wants |
|---|---|---|
| prefill launch count | chunk | LARGE (fewer launches on long prompts) |
| sliding ring stride | `window + write_rows - 1` | SMALL (every decode step traverses it) |

`stage_rows` is exactly the split: `Manifest::write_rows` (`packed_prefill.rs:174`) returns the
STAGE width, so the ring is sized `window + stage - 1` while the launch still covers the whole
chunk (tests at `:1065-1066`: unstaged `write_rows(4096) == 4096`, staged-at-1024 `== 1024`).
Chunk 4096 with `stage_rows` 1024 should give the 4695-ms-and-better long-prompt TTFT at the
ring-2048 decode cost, AND re-enable both attention role objects, which need the 4096 rung.

That is now the measured case for item 3, and it replaces both earlier rationales (the memory one
is wrong -- see the correction above -- and "fewer launches" alone did not explain why chunk 1024
was chosen).


### Two corrections to the sections above (2026-09-23, later)

**1. "33 GiB unused" was weak evidence; here is the right arithmetic.** Peak memory scales with
LIVE streams, so measuring 47.3 GiB at 15000/C32 while only 8 streams ran proves nothing about
headroom for 32. What 32 LIVE slots actually need, counting only rows a request touches (the
sliding ring is fully touched once a prompt exceeds it; full-attention KV is `inlen+128`):

| in | chunk 1024 (ring 2048) | chunk 2048 (ring 4096) | chunk 4096 (ring 8192) |
|---|---|---|---|
| 4096 | 50.4 GiB | 70.4 GiB | 71.7 GiB |
| 8192 | **58.4 GiB** | 78.4 GiB | 118.4 GiB |
| 15000 | **71.7 GiB** | 91.7 GiB | 131.7 GiB |

So at the shipped chunk 1024, 32 live slots fit at EVERY rung including 15000 (71.7 of 80 GiB).
The conclusion "C32 is not memory-gated" therefore survives, but for a sharper reason: the packet
could hold 32 streams at 15000 and instead ran 8, stopping ~24 GiB short. The margin is only
~8 GiB though, and chunk 2048/4096 genuinely do NOT fit 32 slots at long context -- which is
exactly why the c32-16k recipe picked chunk 1024, as its own header says.

**2. "The doubled ring slows every decode step" is an unverified cross-packet attribution.**
`kv_stride` (the ring) appears ONLY in the base-address computation --
`kbase = K + (slot * n_kv_head + hkv) * kv_stride * D` -- while the walk length is
`span = len - first` with `first = (window && len > window) ? len - window : 0`
(`op_attention.cuh:772-779`). A bigger ring cannot make the kernel read more rows; it only
spreads the slots further apart in the address space. The 128/C16 TPOT 13.05 -> 17.97 ms figure
was measured across TWO DIFFERENT PACKETS (p12c32c vs the chunk-1024 packet), the same
cross-packet attribution that made `GV_MM_MAX` look like a 0.23-0.37 ms tax when a
single-variable A/B put it at 0.7%. Re-test it single-variable before building `stage_rows` on
the premise that a small ring is worth protecting.

This does not retire item 3 -- if the TPOT cost is real, staging is still the only way to get
chunk-4096 prefill with a 2048-row ring, and the memory table above shows chunk 4096 needs it.
But the justification is now explicitly UNVERIFIED, and the cheaper experiment (single-variable
ring A/B on one packet) comes first.


## Glossary: row, slot, chunk, ring, window, ctx

These have been used loosely in this file. Each is pinned to the identifier that defines it.

| term | code identifier | what it counts | 12B value |
|---|---|---|---|
| **row** | `row_bytes` | one token x one layer, K AND V | 8 KiB (kv width 2048 x 2 B x 2) |
| **slot** | kernel `i0 = n_batch`, engine `batch`, mux `capacity` | one concurrent sequence's seat | 32 on the c32 packet |
| **window** | kernel `i4`, `cache.window` | how far back a query reads | 1024 sliding; **0** on the 8 full layers |
| **chunk** | `PLOW_MAX_CHUNK` | prompt tokens per PREFILL LAUNCH | 1024 |
| **ring** | kernel `i3 = kv_stride`, `cache.stride` | rows reserved per (slot, kv_head) on a sliding layer | `next_pow2(window+chunk-1)` = 2048 |
| `kv_mask` | kernel `i7` | ring - 1, for `(kv0 + r) & kv_mask` | 2047 |
| **write_rows** | `Manifest::write_rows` | rows one request may WRITE per launch | = chunk, or = `stage_rows` when staged |
| **ctx** | `max_ctx`; `kv_len` = kernel `t5` | whole-sequence capacity; sizes the FULL layers | 16384 capacity, `kv_len` live |

The ring rule (`packed_prefill.rs:222-229`) is a three-way OR, not the single clause quoted
earlier in this file:

```
cache.window == 0 || cache.stride >= max_ctx || cache.stride >= cache.window + write_rows - 1
```

i.e. legal if the cache is not sliding at all, OR its ring already spans the whole context, OR it
retains the window across everything one launch writes.

### The three quantities are independent, and chunk touches only two

Worked at 8192 in, chunk 1024, ring 2048, 32 slots:

* **decode work per step, one sliding layer** = `min(kv_len, window)` = min(8320, 1024) = **1024
  rows**. The ring does NOT enter -- `kv_stride` appears only in the base address
  (`op_attention.cuh:795`), while the walk is `span = len - first`, `first = len - window`
  (`:777-779`).
* **memory per slot** = `min(tokens touched, ring)` x 40 sliding + `kv_len` x 8 full
  = 0.62 + 0.51 = **1.13 GiB** -> x32 slots + 22.2 GiB weights = **58.4 GiB**.
* **prefill launches** = `ceil(prompt / chunk)` = **8**.

So `chunk` couples LAUNCH COUNT and RING MEMORY, and nothing else; `window` sets decode work and
the ring's floor; `ctx` sets full-layer memory only. There is no path from ring size to decode
work, which is why "the doubled ring slows every decode step" needs a single-variable re-test.


## Why C32 loses at long input: plow serializes prefill and decode, vLLM shares a pass

Prefill share of the timeline = (req/s x input_len) / that stack's own single-stream prefill rate,
where the single-stream rate is taken from its OWN C1 TTFT at the same input length. It says what
fraction of the achievable prefill throughput each stack is actually sustaining at C32.

| in | plow rate | plow used | plow share | vLLM rate | vLLM used | vLLM share |
|---|---|---|---|---|---|---|
| 1024 | 22012 | 8366 | 38.0% | 21677 | 10813 | 49.9% |
| 4096 | 21790 | 12083 | 55.5% | 24083 | 19128 | 79.4% |
| 8192 | 21158 | 12206 | **57.7%** | 23478 | 21299 | **90.7%** |
| 15000 | 19388 | 10650 | **54.9%** | 22329 | 21600 | **96.7%** |

**vLLM sustains 90-97% of its single-stream prefill rate WHILE ALSO decoding ~22 streams.** That
is impossible under serialization -- the only way is mixing prefill and decode tokens in the same
forward pass, where the decode rows are nearly free because the weights are already streamed for
the prefill rows.

plow sustains 55-58% and decodes 8-14 streams in the remainder. It is NOT slower at either job in
isolation: its single-stream prefill rate is within 10% of vLLM's at every length (21158 vs 23478
at 8192), and its C1 TPOT is at parity at every context. It loses because the two phases take
turns on the timeline instead of sharing it.

This is the gap `docs/arch/17-unified-token-batch.md:20` already names: "CUDA has no token-batch
executor yet, so the default does not enable token batching on H100." `exec::mixed_program` is
`#[cfg(feature = "hsa")]`, i.e. AMD host-side only; the NVIDIA kernel arms exist behind
`#if PLOW_MIXED_STEP`.

### What this reprioritises

The deficit that loses the most cells is out_tok_s at C>=4, and its cause is the missing mixed
step, not a kernel. Ranked by how much of the remaining gap each would close:

1. **CUDA token-batch executor (mixed prefill+decode step).** Directly attacks the 55% -> 90%+
   timeline-sharing gap. Largest and hardest.
2. **Fewer prefill launches at 32 slots** (item 3 / `stage_rows`): chunk 4096 with a 2048-row
   ring. Helps the prefill half only, and its "protect the small ring" premise is still
   UNVERIFIED (see the correction above).
3. **Decode batched-KV traversal** (TPOT +25-33% at C16, parity at C1). Real but smaller, and the
   occupancy route for it is closed by measurement.

Note the flattering C32 TPOT numbers (plow 50.7 vs vLLM 67.5 at 8192) are a SYMPTOM of this:
plow's decode runs uncontended in its own slice of the timeline, so each step is fast while
fewer steps happen. Judge throughput by out_tok_s and effective streams, never by TPOT alone.


### Correction: the CUDA token-batch route DOES exist and DOES fire

The section above cited `docs/arch/17-unified-token-batch.md:20` ("CUDA has no token-batch
executor yet") to conclude plow cannot mix prefill and decode on H100. **That doc line is stale.**
`crates/plowrt/src/exec/gpu/token_batch.rs` (572 lines) is a CUDA unified-token-batch route, and
the C32 serve log shows it loading and firing:

```
token-batch route status route="unified-token-batch" backend="cuda" ready=true fires=false
      reason="packed-prefill and compact-output capabilities loaded"
token-batch route fired  route="unified-token-batch" backend="cuda" ready=true fires=true
      requests=1 rows=39
```

`CudaTokenBatch::load` gates on `token_batch && !fusion && cc == (9,0) && packed_prefill &&
has_packed_terminal && recurrent.is_none() && mixed_step.is_none()`, and the c32-16k recipe sets
`PLOW_TOKEN_BATCH = "1"`. So the capability is present and engaged; it is `exec::mixed_program`
(the AMD v1 path) that is `#[cfg(feature = "hsa")]`, not the CUDA route.

**So the measured 55-58% vs 90-97% prefill share is NOT "plow cannot mix".** Both stacks batch
decode and both can carry prefill and decode in one pass. The open question is why plow's mix
degrades with input length while vLLM's tightens. What the data pins down:

* effective streams plateau at ~13 at 8192 whether C16 or C32 is requested, but reach 23.3 at
  4096 on the same packet;
* memory allows 32 live slots at 8192 (58.4 GiB of 80);
* the KV admission budget allows it (`max_rows=794890` against 262144 needed);
* `mux capacity resolved ... capacity=32`.

None of the static gates explain it, so it is a per-step selection effect. The route already
carries the instrumentation for exactly this: `PLOW_STEP_TIME=1` makes `token_batch.rs`'s
`steptime` module log `steps`, `rows_total`, `enqueue_ms` and `terminal_ms` as rolling means.
The "token-batch route fired" line is one-shot, so batch sizes cannot be recovered from existing
logs — this needs a fresh instrumented run at 8192/C32 on a 32-slot 16k packet.


### The sliding ring is FREE at decode. Item 3's premise is refuted.

The c32-16k recipe rejects a bigger chunk on decode grounds: "the doubled ring slows every decode
step (128/C16 TPOT 13.05 -> 17.97 ms, 1179 -> 856 tok/s)". The req1k recipe pays for the same
belief in the other currency, capping `PLOW_MAX_REQUEST_CHUNK` at 1024 so the ring stays 2048 —
its own header calls 8 launches for an 8192-token prompt "the price of the small ring".

Both numbers came from comparing two DIFFERENT packets. That is the same cross-packet attribution
that made `GV_MM_MAX` look like a 0.23-0.37 ms tax when a single-variable A/B put it at 0.7%.

`ab_r2048` and `ab_r4096` differ in exactly ONE emit setting — `max_chunk` 1024 vs 2048, hence
ring `next_pow2(1024 + chunk - 1)` 2048 vs 4096 — with `decode_batch 32` on both. `step_bench`,
48 layers, same binary, same lease:

```
  B   ctx   ring2048   ring4096     delta    ratio
 16   128     17.750     17.754    +0.004   1.000x
 16  1024     19.981     19.969    -0.012   0.999x
 32   128     22.771     22.665    -0.106   0.995x
 32  1024     27.141     26.989    -0.152   0.994x
```

Flat to 0.6%, and the two non-zero deltas favour the BIGGER ring. The claimed 1.38x is not there.

The kernel said so first and should have been believed before a packet was built around the
opposite. `op_attention.cuh:772-779` walks `span = len - first` with
`first = (window && len > window) ? len - window : 0`. That is `min(kv_len, window)` — live
sequence length and the compiled window, neither of which is the ring. `kv_stride` enters at
`:795` and only as the base address. **The ring cannot change decode work, because no loop bound
reads it.**

What the ring does cost is resident KV on the sliding layers: a prompt longer than the stride
wraps, so a bigger stride maps more pages for the same prompt. That is a memory question, and
peak memory on this packet is flat at 46-48 GiB of 80 with 33 GiB unused. It is not a decode
question.

Consequences, in order of cheapness:

* `PLOW_MAX_REQUEST_CHUNK = 1024` on the req1k recipe is buying nothing. Raising it toward
  `max_chunk` cuts an 8192-token prompt from 8 launches to 2 at the cost of ring residency —
  measurable on the existing packet family, no new kernel.
* **Item 3 (sub-chunk staging) loses its justification.** It exists to keep a small ring while
  raising the chunk. If the small ring is worth nothing, the staging machinery — emit loop,
  manifest `stage_rows`/`stages`, runtime per-stage fills — buys nothing either. The
  `emit.stage_rows` knob already landed (`e56c78b1`) is harmless and stays OPT_IN/UNSET, but the
  remaining work should not proceed on this rationale.
* The recipe headers that state the ring tax should be corrected rather than left to seed a third
  packet built around it.

This is the eighth refuted lever in this campaign and the third that was refuted by making the
A/B single-variable. The pattern is consistent enough to be a rule: **a number measured across
two packets is a hypothesis, not a result.**


### Decode rows already ride. The C32 deficit is prefill throughput, not packing.

`PLOW_PF_PACKLOG=1` on p12rq (32 slots, `max_chunk` 4096, `max_request_chunk` 1024), two cells on
one packet, one binary, one lease. 1024/C32 is a cell plow leads on effective streams; 8192/C32
is the one that was written up as a collapse.

```
                          1024/C32    8192/C32
  decode_feeds riding      18.56       19.95     mean per pack
    median / max           24 / 31     26 / 31
    packs with zero        3 / 27      17 / 177
  prefill reqs per launch   3.11        3.14
  prefill rows per launch  2504.6      3054.9    (max 4095)
  bucket padding            4.7%        6.9%
  cumulative prefill        3.18 s     27.40 s
  cumulative decode         5.86 s      5.01 s
  PREFILL SHARE            35.2%       84.5%
  decode batch width       32 median   12 median (mean 20.45 -> 15.19)
  ticks doing prefill      27 (6.5%)   163 (33.7%)
  mean prefill tick        117.9 ms    168.1 ms
```

`unified=true` on all 177 packs at 8192 and the route logs `token-batch route fired`, so these
decode rows are genuinely carried inside the prefill pass, not counted and dropped.

**Three things this settles.**

1. **Packing is at its ceiling, not below it.** ~20 decode rows ride every launch, and 109 of the
   177 launches at 8192 run a FULL 4096-row bucket (`rows->bucket 4096->4096 x109`). Padding is
   6.9%. There is no room to pack decode into, because the launches are already full. This also
   answers whether `sched::step::Backend::decode_rows_join_prefill` should default true: it must
   not. Its only effect is `budget -= decodes.len()` (`sched/step.rs:130-132`), and the CUDA arm
   already subtracts decode rows twice before calling the planner — `per_launch = ... .min(
   budget_max - decode_rows)` (`mux.rs:4234`) and the `trim` rule (`mux.rs:4333`). Flipping it
   would charge a third time and SHRINK the prefill slice. `gpu.rs:7726` says as much. It is
   false on AMD for the different reason that decode there is a separate dispatch.

2. **The mechanism is prefill work, not scheduling.** Decode wall time is flat between the two
   cells (5.86 -> 5.01 s) while prefill goes 3.18 -> 27.40 s. Slots mid-prefill contribute no
   decode row, so the decode batch halves (median 32 -> 12) as a CONSEQUENCE of prefill
   occupancy. The earlier framing — "effective streams plateau at ~13, so per-step row selection
   is at fault" — had the causality backwards.

3. **8192/C32 on this packet is not a collapse.** Effective streams (`out_tok_s * TPOT`) are
   22.1 for plow against 22.4 for vLLM — a tie. The 13.6 recorded earlier was the chunk-1024
   32-slot packet; p12rq keeps `max_chunk` 4096 and caps only the per-request slice, and that
   repairs most of it. What remains at 8192/C32:

   ```
                 plow      vLLM 0.28
     TTFT ms    4425.32    3657.96     -21%
     TPOT ms      82.06      67.49     -22%
     p99 ITL     206.28     353.90     +42%  (plow better)
     out_tok_s    269.1      332.2     -19%
     peak MiB     52334      75824
   ```

**The lever this points at.** 64 requests of 8192 tokens currently cost 584,768 bucket rows across
177 launches, because `max_request_chunk` 1024 caps each request to 1024 rows per launch and three
of them share a 4096-row bucket. At 4096 rows per request the same work is 524,288 rows in 128
launches: 10% fewer rows, 28% fewer launches, and each request leaves prefill after 2 launches
instead of 8, which returns it to the decode batch sooner. With prefill at 84.5% of the timeline,
10% off prefill is ~8% off the cell — roughly half the remaining throughput gap.

That cap existed only to hold the ring at 2048, and the ring A/B above shows the small ring is
worth 0.0-0.6% at decode. What the bigger ring does cost is residency, since a sliding layer
touches every ring row once the prompt passes the stride:

```
  req_chunk   ring rows   per slot   x32 slots   + 22 GiB weights
     1024        2048      640 MiB     20 GiB      measured peak 52.3 GiB at 8192/C32
     2048        4096      1.25 GiB    40 GiB      ~72 GiB of 80 -- tight, plausible
     4096        8192      2.5 GiB     80 GiB      rings alone exceed the card
```

So `rc2048` is the arm expected to land and `rc4096` is built to locate the wall rather than argue
about it. Both are one-variable overrides of the unchanged req1k recipe.


### The ring budget at 32 slots, from measured geometry

The previous two sections leave a contradiction: the ring is free at decode (so raise the
per-request chunk) but the recipes all cap it. Resolving it needs the memory arithmetic written
down once, because it has now been derived wrongly twice in one session — first as a static
allocator number that the runtime does not use, then as an estimate that ignored the
full-attention layers entirely.

Geometry, read from the checkpoint's `config.json`, not from recipe prose:

```
  num_hidden_layers   48      layer_types  40 sliding + 8 full
  num_key_value_heads  8      head_dim    256      sliding_window 1024
```

One KV row costs `2 (K+V) * 8 heads * 256 * 2 B = 8 KiB` per layer. So a row is **320 KiB across
the 40 sliding layers** and **64 KiB across the 8 full ones**.

`packed_prefill.rs:222-229` sizes a slot's sliding ring at `next_pow2(window + write_rows - 1)`,
and `Manifest::write_rows` is the PER-REQUEST chunk (`max_request_chunk`), not `max_chunk`:

```
  req_chunk  ring rows   per slot    x16 slots   x32 slots
     1024      2048       640 MiB      10 GiB      20 GiB    <- req1k today
     2048      4096      1.25 GiB      20 GiB      40 GiB
     4096      8192       2.5 GiB      40 GiB      80 GiB
     4224      8192       2.5 GiB      40 GiB      80 GiB    <- l8192 ships this at 16 slots
     8192     16384         5 GiB      80 GiB     160 GiB
```

The full-attention layers are NOT windowed, so they need `ctx` rows per slot regardless of chunk:
64 KiB/row * 8192 * 32 = **16 GiB at an 8k prompt**, **32 GiB at 16k**. Weights are 22.7 GiB.
Against an 80 GiB card:

```
  32 slots, 8192-token prompts:  22.7 weights + 16 full-KV = 38.7 GiB fixed -> ~41 GiB for rings
  32 slots, 15000-token prompts: 22.7 weights + 29 full-KV = 52   GiB fixed -> ~28 GiB for rings
```

Measured peak on p12rq (req 1024) at 8192/C32 is 52.3 GiB, consistent with 22.7 + 16 + 20 less the
slots that never filled.

**Consequences.**

* An 8k PER-REQUEST chunk at 32 slots wants 160 GiB of rings. It is not a tuning question; it is
  twice the card. At 16 slots it is 80 GiB, still the whole card before weights.
* `req_chunk` 4096 at 32 slots is 80 GiB of rings alone — also out.
* `req_chunk` 2048 at 32 slots is 40 GiB, which fits at 8k prompts (78.7 GiB total, tight) and
  does NOT fit at 15000 (~92 GiB). So the long end of the C32 ladder is where it breaks.
* Therefore the req1k cap is load-bearing after all — but for residency, which its header never
  states, and NOT for the decode cost its header does state, which the ring A/B refuted. Both
  things are true and they are about different resources.

**What an 8k chunk does mean, and already works.** `max_chunk` and `max_request_chunk` are
different knobs. `gemma4-12b.h100.bf16-l8192-16k.toml` ships `PLOW_MAX_CHUNK = 8192` — real
8192-row launches — with `PLOW_MAX_REQUEST_CHUNK = 4224`, so an 8k launch is assembled from
several requests' slices and the ring is sized by the slice. That is the reachable form of "8k
chunks", and PACKLOG shows the launch side is already saturated: 109 of 177 launches at 8192/C32
run a full 4096-row bucket at 6.9% padding.

**Where the static sizing is genuinely wasteful.** The ring is allocated per slot for the case
where every slot simultaneously runs a full chunk. PACKLOG measured `prefill reqs/pack` mean
**3.14**, max **5** — at most five slots are mid-prefill at any instant. A shared ring pool sized
for the concurrent-prefill count rather than the slot count would buy a 4x larger per-request
chunk at today's 20 GiB. That is a real design lever and the measured number to size it with is
3.14, but it is an allocator change, not a knob, and nothing in the current packet format
expresses it.


### Adaptive interleave loses at C32; serve-policy auto wins slightly; the ~13 plateau is undersized launches

Two runtime knobs measured on p12rq at C32, one variable each, `high_concurrency` profile
otherwise unchanged.

**`PLOW_PF_INTERLEAVE_ADAPTIVE=1`.** The `high_concurrency` profile has never set it (only
`realtime` does), so this is its first C32 measurement.

```
  in      TTFT ms          TPOT ms        p99 ITL          tok/s          eff
          adapt / off      adapt / off    adapt / off      adapt / off    adapt / off
  1024    590.8 /  587.8   25.92 / 24.27   76.7 / 179.0   1016.4 / 1093.3  26.3 / 26.5
  4096   2139.9 /    --    53.84 /  --    163.1 /  --      430.6 /  --     23.2 /  --
  8192   7913.7 / 4425.3   57.80 / 82.06  164.1 / 206.3    240.4 /  269.1  13.9 / 22.1
 15000  18356.0 /    --    66.18 /  --    178.6 /  --      131.7 /  --      8.7 /  --
```

It loses, and `policy.rs:91` says why in its own doc comment: "the oldest prompt runs whole and
later ones join only while that is cheaper. It wins when a few prompts arrive together and **loses
when the queue is deep enough that filling the launch matters more**." At C32 with 64 prompts the
queue is never shallow. TTFT nearly doubles at 8192 and effective streams fall 22.1 -> 13.9.
Under `PLOW_SERVE_POLICY=auto` the runtime already deselects it at this width
(`adaptive_packing` = `class() == Realtime`), which is the correct behaviour and is now measured
rather than assumed.

What it does buy is interval smoothness: p99 ITL 76.7 vs 179.0 at 1024 (2.3x better) and 164.1 vs
206.3 at 8192. So it is a real latency/throughput trade, not a bad knob — but the campaign goal is
scored on all four metrics, and it loses two of them badly.

**`PLOW_SERVE_POLICY=auto`** (default `pinned`) picks the class per tick from decode width and
queue depth. It beats the pinned baseline on ALL FOUR at 8192, narrowly:

```
            TTFT ms   TPOT ms   p99 ITL    tok/s   peak GiB
  pinned    4425.32    82.060    206.28    269.1     51.11
  auto      4362.25    81.060    203.87    272.4     51.11
```

and the log shows it working rather than sitting on a default:

```
  serve policy: profile switched from=HighConcurrency to=Realtime      width=1  queued=0
  serve policy: profile switched from=Realtime to=HighConcurrency      width=32 queued=0
```

Against vLLM 0.28 at C32 it stands:

```
  in       TTFT ms         TPOT ms        p99 ITL          tok/s         eff
           auto / vLLM     auto / vLLM    auto / vLLM      auto / vLLM
  8192    4362 / 3658     81.1 / 67.5    203.9 / 353.9   272.4 / 332.2  22.1 / 22.4
 15000    8741 / 6434    140.6 / 123.3   218.2 / 383.8   149.8 / 183.9  21.1 / 22.7
```

Still behind on TTFT/TPOT/throughput by 14-36% and ahead on p99 ITL by 43%.

**The unifying observation.** `base_ad` at 8192 lands at effective streams **13.9**. The
chunk-1024 32-slot packet landed at **13.6** at the same cell. Those are different causes —
adaptive shrinks a launch by heuristic, the chunk-1024 packet shrinks it by construction — with
the same effect: prefill launches too small for the standing queue. The ~13 plateau recorded
earlier as evidence of a per-step row-selection defect is nothing of the kind; it is the
signature of undersized prefill launches, and it reproduces on demand by shrinking them. That
retires the "per-step selection" hypothesis for good.

**Harness note.** The first rc2048 bench produced an empty results CSV: the packet had been built
with the job killed mid role-emit, so `assets/build.json` existed while `weights.json` did not and
the serve died with `Io { path: ".../weights.json", NotFound }`. `build.json` is written early and
is NOT sufficient evidence that an emit finished — check the file set. The campaign's coherence
gate caught it ("numbers above are not evidence"), so nothing false was recorded.


### req_chunk 2048 lands, and the residency prediction was too pessimistic

`rc2048` is the req1k recipe with one variable changed: `PLOW_MAX_REQUEST_CHUNK` 1024 -> 2048,
`max_chunk` 4096 on both, verified from `build.json` (`req_chunk = 2048`, `knobs K = verified`).
C32, `high_concurrency`, adaptive off.

```
   arm       in      TTFT ms   TPOT ms   p99 ITL    tok/s   peak GiB    eff
  req1024   1024      587.77    24.270    179.00   1093.3     47.11    26.5
  req2048   1024      586.36    24.680    180.59   1078.3     67.10    26.6
  req2048   4096     1947.53    48.620    193.19    493.9     69.10    24.0
  req1024   8192     4425.32    82.060    206.28    269.1     51.11    22.1
  req2048   8192     3822.43    84.120    205.22    276.0     71.10    23.2
  req2048  15000     7254.35   147.610    228.01    153.8     73.10    22.7
```

**It did not OOM at 15000.** The section above predicted ~92 GiB there and said rc2048 "does NOT
fit at 15000". Measured peak is **73.10 GiB**. That prediction was wrong, and the error is worth
naming because it is the same class of error the ring arithmetic was written to prevent: the
`weights + full-KV + rings` sum is an upper bound on RESERVED rows, not a measurement of RESIDENT
pages. Not every slot is live at full context simultaneously, requests retire and release, and the
VMM maps per block on demand. The bound is still useful as a ceiling — req_chunk 8192 at 160 GiB
of rings is impossible by any accounting — but a number within ~25% of the card must be measured,
not asserted.

Peak also grows a uniform +2.0 GiB per cell (67.1 / 69.1 / 71.1 / 73.1) across input lengths that
are not uniformly spaced, so it is not tracking context. Unexplained; not chased.

**What the chunk buys.** Against req1024 at the same cell, 8192/C32: TTFT **-13.6%** (4425.32 ->
3822.43), tok/s **+2.6%** (269.1 -> 276.0), p99 ITL -0.5%, TPOT +2.5% worse. Effective streams
22.1 -> 23.2. At 1024 the two are indistinguishable (TTFT -0.2%, tok/s -1.4%). So it is three of
four metrics better at 8192, neutral at short prompts, and it opens 15000 at eff 22.7 where
req1024 under `auto` managed 21.1 with TTFT 8741 (rc2048 is **17% faster to first token** there).
The cost is +20 GiB of peak, which the card has.

**Standing at C32 with req_chunk 2048, against vLLM 0.28:**

```
    in      TTFT ms          TPOT ms         p99 ITL           tok/s          eff
            plow / vLLM      plow / vLLM     plow / vLLM       plow / vLLM
   1024    586.4 /  704.4    24.68 / 18.19   180.6 / 250.6   1078.3 / 1351.9  26.6 / 24.6
   4096   1947.5 / 1990.4    48.62 / 37.89   193.2 / 325.4    493.9 /  597.8  24.0 / 22.7
   8192   3822.4 / 3658.0    84.12 / 67.49   205.2 / 353.9    276.0 /  332.2  23.2 / 22.4
  15000   7254.4 / 6433.8   147.61 /123.28   228.0 / 383.8    153.8 /  183.9  22.7 / 22.7
```

plow wins **TTFT at 1024 and 4096**, **p99 ITL at every length** (by 28-41%), and **effective
streams at every length**. It loses **TPOT and out_tok_s everywhere**, by 16-25%. No cell is a
clean four-metric win, so the honest C32 standing on the 12B is still 0/4 by the campaign rule,
but the shape of the remaining deficit is now unambiguous and singular: **decode step time**.
TTFT is within 4.5% at 8192 and ahead at two lengths; the throughput gap is TPOT and nothing else,
which is the KV-traversal gap already characterised in the metric-fingerprints note.

That also means the prefill levers are close to exhausted on this model: decode rows already ride
at ~20/pack, launches already fill their bucket 109/177, the ring is free, and the chunk has now
been raised as far as memory allows. What is left is the decode kernel.


### RETRACTION: the req_chunk 2048 result is a cross-packet delta, not a chunk effect

The section above reports "req_chunk 2048 lands" with 8192/C32 TTFT -13.6% and effective streams
22.1 -> 23.2, on the stated basis that rc2048 is "one variable off" req1024. **It is not.** The
baseline those deltas are taken against is packet `p12rq`, and a `build.json` env diff
(`replay` + `unrecorded_env`) puts the two **ten keys apart**:

```
  (unrec)PLOW_SEG_FA256_GQA2          1  ->  —
  (unrec)PLOW_SEG_FA512               1  ->  —
  (unrec)PLOW_SEG_PURE_GEMM           1  ->  —
  PLOW_ATTENTION_DECODE_BALANCE_GF    4  ->  —
  PLOW_EMIT_PREFILL_CUBLASLT          1  ->  —
  PLOW_NO_GLU_FUSE                    1  ->  —
  PLOW_SEG_PURE_GEMM                  1  ->  —
  PLOW_SLIDING_NS_CAP                 1  ->  —
  PLOW_SLIDING_NS_GRID                1  ->  —
  PLOW_MAX_REQUEST_CHUNK           1024  ->  2048
```

`p12rq` predates the current `c32-req1k-16k` recipe and carries six recorded knobs and three
unrecorded `PLOW_SEG_*` flags that rc2048 does not. `PLOW_SLIDING_NS_GRID` and
`PLOW_EMIT_PREFILL_CUBLASLT` are both known perf levers with their own certificates. So the
-13.6% cannot be assigned to the chunk.

**What survives:** rc2048's absolute cells are valid measurements of rc2048 (8192/C32 TTFT
3822.43, TPOT 84.120, p99 ITL 205.22, tok/s 276.0, peak 71.10 GiB; 15000/C32 7254.35 / 147.610 /
228.01 / 153.8 / 73.10 GiB). In particular **ring 4096 x 32 slots FITS at 15000** — 73.10 GiB of
80 — which refutes this document's own ~92 GiB estimate independently of any A/B, because it is a
single measured number, not a difference. The residency-bound correction stands.

**What does not:** every delta quoted against req1024, and the claim that the chunk is worth ~8%.

The control arm is now building: `rc1024`, the same recipe with NO override, so rc1024 vs rc2048
differs in exactly `PLOW_MAX_REQUEST_CHUNK`. The script asserts that before benching.

**This is the third time this session that a cross-packet delta was reported as a single-variable
result** — first `GV_MM_MAX` (0.23-0.37 ms "tax" that a real A/B put at 0.7%), then the ring tax
in two recipe headers, now this one, and this one is mine. The ring A/B itself was re-verified by
the same method and IS clean: `ab_r2048` vs `ab_r4096` differ in exactly one key
(`PLOW_MAX_CHUNK`), identical `tuning` block, identical `registry_digest` — so the ring refutation
above is unaffected.

**Process fix, not a resolution to be more careful.** `build.json` already carries everything
needed to catch this: `emit_config.replay` (the env that reproduces the build) plus
`emit_config.unrecorded_env` (env read outside `EmitConfig`). Diffing those two between the arms
takes a second and is now a preserved tool rather than an intention — see the packet preservation
record below. No A/B in this campaign should be quoted again without that diff printed alongside
it.


### RESOLVED: req_chunk 1024 vs 2048, one variable, and the verdict is DO NOT ADOPT

`rc1024` is the same recipe as `rc2048` with no override. `preserve_packet.py diff rc1024 rc2048
--expect PLOW_MAX_REQUEST_CHUNK` exits 0: **1 differing key of 12**, same registry digest, same
tuning. The script asserted that before spending the lease. Both ladders, C32, `high_concurrency`,
adaptive off:

```
    in    metric    req1024    req2048     delta       vLLM   verdict
  1024      TTFT     577.51     586.36     +1.5%     704.43       WIN
            TPOT      24.03      24.68     +2.7%      18.19      lose
          p99ITL     177.84     180.59     +1.5%     250.55       WIN
           tok/s    1104.80    1078.30     -2.4%    1351.90      lose
         peakGiB      47.10      67.10
  4096      TTFT    2138.73    1947.53     -8.9%    1990.36       WIN
            TPOT      48.32      48.62     +0.6%      37.89      lose
          p99ITL     193.90     193.19     -0.4%     325.35       WIN
           tok/s     484.40     493.90     +2.0%     597.80      lose
         peakGiB      49.10      69.10
  8192      TTFT    4274.20    3822.43    -10.6%    3657.96      lose
            TPOT      82.40      84.12     +2.1%      67.49      lose
          p99ITL     205.78     205.22     -0.3%     353.90       WIN
           tok/s     270.70     276.00     +2.0%     332.20      lose
         peakGiB      51.10      71.10
 15000      TTFT    8593.90    7254.35    -15.6%    6433.81      lose
            TPOT     140.31     147.61     +5.2%     123.28      lose
          p99ITL     227.09     228.01     +0.4%     383.84       WIN
           tok/s     150.90     153.80     +1.9%     183.90      lose
         peakGiB      53.10      73.10
```

**The chunk is a real TTFT lever at long prompts** — -8.9 / -10.6 / -15.6% at 4096 / 8192 / 15000,
with out_tok_s +2% — so the direction of the retracted claim was right.

**But the magnitude was inflated and the cost was hidden.** The confounded comparison reported
-13.6% at 8192; one variable gives -10.6%. The difference is that p12rq (4425 ms) is SLOWER than
rc1024 (4274 ms) at that cell, so the extra knobs p12rq carries were padding the baseline. And the
true A/B exposes a TPOT cost of +2.1 to +5.2% that the cross-packet delta had concealed entirely
(it showed +2.5% at 8192 against a baseline that was itself slow).

**Verdict: do not adopt.** The campaign is scored on all four metrics and the single remaining
deficit is TPOT. Spending 2-5% of TPOT to buy TTFT is trading the metric plow already leads into
the one it is stuck on, and it costs a flat +20 GiB of peak at every length. At 1024 it is worse
on all four. `rc2048` stays a measured arm, not a default; the req1k recipe keeps 1024.

What it does settle, independently of any A/B, is the residency question: **ring 4096 x 32 slots
fits** — 73.10 GiB of 80 at 15000 — so this document's earlier ~92 GiB estimate was wrong, and the
reserved-vs-resident correction stands.

**Standing at C32 on the clean rc1024 packet, against vLLM 0.28:** p99 ITL WINS at all four
lengths (177.8/193.9/205.8/227.1 vs 250.6/325.4/353.9/383.8, 29-41% better); TTFT WINS at 1024 and
4096; TPOT and out_tok_s LOSE at all four. 0 of 4 cells clean. The deficit remains singular and is
TPOT.


### The B=32 decode deficit is NOT attention. It is the per-stream cost of the GEMV walk.

Two entries in this campaign disagreed: the metric-fingerprints note concluded "the TPOT gap is
entirely KV traversal ... decode *attention* is the target, not the GEMMs" from a constant/slope
split of SERVED TPOT, while the decode-knee note found the same knee at ctx=1024 and called it
rung 32. Served TPOT cannot arbitrate — PACKLOG puts prefill at 84.5% of wall at 8192/C32, so
served TPOT carries interference. Kernel-only `step_bench` on p12rq does.

```
     ctx       B=1      B=32    B32-B1  per-stream
     128    10.811    14.247     3.436     0.1108
    1024    10.894    17.446     6.552     0.2114
    4096    10.909    18.354     7.445     0.2402
    8192    10.945    19.526     8.581     0.2768
   15000    11.021    21.464    10.443     0.3369
```

**Split at 8192/B=32, by measured secant (not a fit):** constant 14.247 ms, ctx-dependent
19.526 - 14.247 = 5.279 ms. So **73% of the step is ctx-independent and 27% is attention.**

**The deficit lives in the ctx-independent 73%.** At ctx=128, where KV traffic is ~1.6 GB against
23.8 GB of weights, plow already pays **0.1108 ms per added stream against vLLM's ~0.049** — a
2.26x gap with attention essentially out of the picture. B=1 is at parity or better (10.811 vs
vLLM 10.46-10.55 served at C1).

Stated as achieved bandwidth over the *identical* 23.8 GB weight stream:

```
  plow  B=1    23.8 GB / 10.811 ms  =  2.20 TB/s     (66% of 3.35 peak)
  plow  B=32   25.4 GB / 14.247 ms  =  1.78 TB/s     (53%)
  vLLM  B~29   23.8 GB / 11.59  ms  =  2.05 TB/s     (61%)
  cuBLASLt, projections only        =  1.54 TB/s
```

plow is the FASTEST of the four at B=1 and the slowest at B=32, on the same weight stream.
**Batching costs plow bandwidth efficiency and does not cost vLLM's.** That is the entire
remaining TPOT gap, and it is a weight-walk property, not an attention property.

This also closes the cuBLAS question for good. Decode carries no `Gemm` arm at all — p12rq's
decode programs are `Gemv`/`GemvQkv`/`GemvGlu`/`GemvArgmax` plus `FlashDecode`/`FlashMerge` — and
routing the projections to cuBLASLt is refuted by its own data (13.93 ms for the projections alone
vs 14.261 ms for plow's whole step). At M=32 there are no FLOPs for a better GEMM to win: the step
reads 23.8 GB of weights whatever the batch. The loaded Lt algorithm table is pinned at m=128,
i.e. prefill buckets; `cublaslt_decode` in the target caps routes 26B MoE grouped matmuls, and the
12B is dense.

**Two defects in this probe, recorded so they are not repeated.**

1. `PLOW_NV_LEAN_DECODE=1` was used as a runtime env to compile the flash arms out and measure the
   attention share directly. It is `Layer::ObjectDefine` (`devgen/src/knob_spec.rs:1721`) — a
   COMPILE-TIME define. Setting it in the environment of a prebuilt packet does nothing, and the
   arm duly returned lean == full to within 0.007 ms at every point. That arm is discarded. A real
   lean split needs a rebuilt decode object, and arm 1 already answers the question without it.
2. The byte model for the ctx-dependent part does not reconcile. 5.279 ms at 8192 against the
   27.9 GB the window arithmetic predicts (40 sliding x 1024 window + 8 full x 8192, x 8 KiB x 32)
   implies 5.3 TB/s, above the 3.35 TB/s HBM peak. Even the full-attention layers alone do not
   fit. So the KV byte estimate is wrong by roughly 3x somewhere — candidates are what `step_bench`
   actually populates per row, the depth-gated attention, or `PLOW_NS_FULL_ABS`. **No attention
   bandwidth figure should be quoted until that is checked.** The TIMES above are direct
   measurements and are unaffected.

**Where this leaves the target.** Not attention, not cuBLAS, not occupancy (closed by arithmetic:
the 128-register cap costs 1.3-1.9x in spill, live state is ~463 registers). What is left is the
MMA GEMV walk's per-stream cost, and the one direction never tested is DEEPER prefetch — only
`MMA_UNB=6`, the shallower arm, was tried, and it was worse at both rungs, which is what made the
recorded diagnosis "bandwidth latency hiding". Deeper costs registers on a walk already at 255
with spill, so it may lose too; it is one build and one `step_bench` sweep to find out, and
`preserve_packet.py diff --expect` can assert the arm is single-variable before the lease.

---

## Prefetch depth at MT=2: a real -3.64% on the decode step, and two wrong turns getting there

`op_gemv_mma.cuh` sets loads in flight from `UNB = PLOW_NV_GEMV_MMA_UNB / MT`. At MT=2 (the 32-row
rung) that halves the depth to 6. The change makes it per-MT — MT=1 keeps 12, MT=2 takes 2/3 (8),
MT>=4 untouched.

### The result that counts

`step_bench` ctx 128, each packet's OWN decode object, no cubin override. `rc1024` and
`rc1024permt` are built from the same recipe with the same emit env (`preserve_packet.py diff` = 0
differing keys of 12 — correct, the variable is the SOURCE) and have identical resources
(`REG:255 STACK:192 SHARED:40464`). 3 reps, spread <= 0.011 ms:

| MT1/MT2 |    B=1 |   B=16 |   B=32 |
|--------:|-------:|-------:|-------:|
|  12 / 6 | 10.724 | 11.582 | 14.239 |
|  12 / 8 | 10.766 | 11.586 | **13.720** |

**B=32 -3.64%, B=16 +0.03%, B=1 +0.39%.**

B=1 regresses although its MT=1 code is unchanged — the decode object is one kernel and the deeper
MT=2 unroll adds 41.6 KB to it. Same class as the fat prefill object: codegen for every rung
depends on what else is compiled in.

### It does reach the served ladder — at the size the arithmetic predicts

The GEMV step is a fixed ~14.2 ms of a TPOT that grows with context, so the served share falls:

|    in | step / TPOT | predicted | measured (3 ladders/arm) |
|------:|------------:|----------:|-------------------------:|
|  1024 | 14.2 / 24.2 = 59% |    -2.1% | **-2.19%** |
|  8192 | 14.2 / 82.1 = 17% |   -0.63% | **-0.63%** |

Both resolvable rungs land on the prediction. 4096 (-0.36%) and 15000 (+0.51%) sit inside their
own bands and carry no information either way.

### Wrong turn 1 — the ladder cannot adjudicate a change this size

TTFT moved across reps on **byte-identical prefill objects**, so its spread is a direct read of the
noise floor: **6.63% at 1024, 1.37% at 4096, 0.21% at 8192, 3.32% at 15000**. Re-running `rc1024`
against itself reproduced most of the apparent permt effect, sign flip at 15000 included (same-arm
TPOT -1.58 / -0.83 / -1.09 / +1.54 % versus permt-vs-baseline -2.50 / -0.70 / -0.79 / +1.89 %).

A pre-registered rule — a majority of rungs must beat their band — then returned "inside noise, do
not land". That rule was **too blunt**: the expected effect varies 6x across rungs and the band is
widest exactly where the effect is largest. The ladder is the right instrument for a scheduler or
geometry change and the wrong one for a 0.5 ms kernel delta. Adjudicate kernel changes on the
packet's own object and use the ladder only to confirm the sign and check nothing else moved.

### Wrong turn 2 — PAIR, a real harness defect that was not the explanation

The first A/B was built with `scripts/build_sm90a_cubin.sh` and reported B=32 -4.06%. That path
never defines `PLOW_NV_GEMV_MMA_PAIR` or `_B1`, so both fall to their header default of 0
(`SHARED:14480`), while `manifest.rs:2762` sets `PAIR 1` for every packet carrying `gemv_mma_pair`
(`SHARED:40464`). With PAIR=1 the plain GEMV path stops calling `gvmma_tile<NW=1>` and calls
`gvmma_tile<NW=2>` over two ROW BLOCKS (`W2[2] = {W, W + 8u*K}`); the prefetch buffer is
`wv[NW][UNB]`, so **loads in flight are NW*UNB and PAIR doubles them** — 12 in the packet at MT=2
against 6 in the generic object.

From that I predicted the shipped packet was already at the knee and the change was worth nothing
in situ. **That prediction was wrong** — measured -3.64% on the real object. The generic harness
overstated the gap (-4.06% vs -3.64%) and got B=1 backwards (-0.47% vs +0.39%), but it ranked the
two arms correctly. The defect is real and worth avoiding; it was not the reason the served ladder
showed little.

Also refuted on the way: register pressure. Both packets are `REG:255 STACK:192` — the edit costs
nothing in registers or spill.

**Rule for future UNB work:** build the A/B from the packet, or with
`PLOW_CUBIN_CONFIG=<packet>/assets/plow_config.h`, and gate on reproducing the shipped object's
`REG/STACK/SHARED` before believing a number. A `PLOW_CUBIN_CONFIG` rebuild alone is not enough —
it reproduced `SHARED:40464` but `STACK:336` against the shipped `192`, because the decode object
also carries tunedb defines (`PLOW_NV_FORCE_MINBLK`, `GV_UNROLL`, `GV_MOE_UN`, `PLOW_MOE_DOWN_SG`,
`GV_UNROLL_GLU`, `GV_MM_MAX`, the FA set) that the script does not know about. Rebuilding the
packet is the reliable route.

---

## FP8 runs on both Gemma-4 models (first 26B FP8 served row)

Both packets build and serve, coherence gate `true`, 128 in / C1 / 4 prompts:

| packet | TTFT ms | TPOT ms | tok/s | peak MiB |
|--------|--------:|--------:|------:|---------:|
| p12fp8 (12B FP8) | 24.52 | 8.630 | 114.2 | 56648 |
| p26fp8 (26B FP8) | 32.39 | 29.130 | 34.3 | 58740 |
| 26B BF16 reference (p26k, 2026-09-20) | 23.10 | 5.880 | 166.2 | 77-81 GiB |

The 12B reproduces the one previously recorded FP8 row (p12fp8c: 24.0 / 8.67). **The 26B has never
served FP8 before this.**

### The 26B FP8 decode is ~5x worse than BF16, as its recipe predicted

TPOT 29.130 vs 5.880 ms, tok/s 34.3 vs 166.2. This is note 2 of the recipe header, now confirmed
rather than predicted: `PLOW_GEMMA_MOE_DEC_GROUP` and the Lt decode route are read only inside the
bf16 branch of the decode emitter (`devgen/src/lib.rs:6083-6100`), so under any fp8 the 26B falls
back to per-slot `MoeExpertGluGemmaFp8` / `MoeExpertDownGemmaFp8` GEMV with no `MoeAlignGemmaPf` --
no assert, no warning. The BF16 grouped-decode win is simply gone.

What FP8 does deliver is the headroom it was built for: 58.7 GiB against a BF16 packet that peaks
at 77.4-80.8 GiB on an 80 GiB card. That is ~20 GiB, which is the constraint forcing 16 slots and
C32 queueing. **So 26B FP8 is today a capacity lever, not a latency one, and it cannot serve the
"beat vLLM on all metrics" goal until grouped MoE decode is reachable under fp8.**

### Two build defects fixed to get here (commit 9efbb116)

1. `build_sm90a_gemma4_segments.sh` could not build a packet without packed prefill. It required
   `interp_sm90a_pfpackedseg.cubin` unconditionally under `set -euo pipefail`, so a Gemma MoE FP8
   packet -- which legitimately has no packed-prefill topology -- aborted the script with **exit 1
   and no message at all**. Past that it compiled packed-request objects anyway and hit
   `interp_sm120.cu:203 "packed-request object requires packed-prefill packet topology"`. Now gated
   on the packet's own `PLOW_PACKET_HAS_PACKED_PREFILL_TOPOLOGY` in three places, and a genuinely
   missing base object names itself.

2. The 12B FP8 recipe's **role emit changes the packet hash** (base `0x65a07f7e459c7474` -> assets
   `0xb4c7405cf7c04bc3`), while `campaign.py:338` points `PLOW_PF_SEG_DIR` at `objects/`, which the
   segments script filled with `cp $base/*.cubin` -- objects specialised to the BASE packet. plowrt
   correctly refuses: `packet/interpreter MISMATCH`. A BF16 recipe never trips this because its
   role emit leaves the hash unchanged (rc1024: base == assets == `0x2ea414e0`). The 26B FP8 recipe
   also leaves it unchanged (`0x344208cf` both sides), which is why only the 12B was affected.
   Worked around by re-running the segments script against the FINAL assets config into `objects2/`
   and serving with `PLOW_PF_SEG_DIR=objects2`; `scripts/campaign/objenv.py` feeds it the recipe's
   `[objects.env]` so that rebuild cannot drift.

**Still open:** `campaign.py` does not do this itself, so a plain `campaign.py build` + `bench` on
the 12B FP8 recipe still cannot serve without the `--env` override. The fix belongs in `cmd_build`:
when the assets hash differs from the base hash, re-specialise and record the correct seg dir.

### A trap in preserve_packet.py, found the hard way

`p12fp8c`'s record carries `recipe_overrides` = the three role flags, and a `rebuild_command` that
passes them as `--env`. Replaying that **fails**: `--env` reaches the BASE emit, which panics at
`devgen/src/lib.rs:9826` looking for a role cubin the later objects step builds. The field is
derived by diffing the final packet's emit env against `[emit.env]`, so anything living in
`[emit_roles.env]` is misreported as an override. The recorded `rebuild_command` is not replayable
for any recipe that uses `[emit_roles.env]`.

---

## What served TPOT is made of, and why the interleave knob cannot flip a cell

Two questions that sound the same and have different answers. Keep them apart.

### 1. The TPOT gap sits outside the decode kernel

`step_bench` B=32 on rc1024permt against served C32 TPOT:

|    in | step(ctx) | served TPOT | "other" | other% | vLLM | gap |
|------:|----------:|------------:|--------:|-------:|-----:|----:|
|  1024 |    16.938 |       23.62 |    6.68 |    28% | 18.19 | 5.43 |
|  4096 |    17.852 |       47.89 |   30.04 |    63% | 37.89 | 10.00 |
|  8192 |    19.035 |       81.58 |   62.55 |    77% | 67.49 | 14.09 |
| 15000 |    20.989 |      142.65 |  121.66 |    85% | 123.28 | 19.37 |

The decode step is nearly context-flat (13.691 ms at ctx 128 -> 20.989 at 15000) while served TPOT
quadruples. KV traversal is **0.29 ms per 1k ctx** at B=32 — 4% of the TPOT rise from 1024 to 8192.
The whole vLLM TPOT gap fits inside "other" at every rung.

At **C1**, where nothing contends, plow is at parity or ahead on BOTH metrics up to 4096:
TTFT 18.33/46.02/169.55 vs vLLM 30.04/47.24/170.08; TPOT 10.42/10.50/10.53 vs 10.46/10.54/10.55.
So the kernels are competitive and the loss is concurrency-only.

PACKLOG at 8192/C32: 157 prefill ticks at a mean **174.10 ms**, 327 decode-only ticks at 15.39 ms,
27.33 s prefill vs 5.03 s decode (84.5% / 15.5%). Mean tick 66.9 ms against TPOT 82.04 — **TPOT is
tick cadence**, and a decode row riding a 4096-row launch waits the whole 174 ms.

> **Reading PACKLOG TICK:** `decode_rows` is 0 on every prefill tick and that does NOT mean decode
> was starved. `mux.rs:2101` — the unified token batch decodes `feeds` inside the prefill pass and
> CLEARS them; `packlog::tick` at :2498 then reads the emptied vector. Decode does ride prefill.
> This cost one wrong diagnosis; do not repeat it.

### 2. But granularity only redistributes time — it does not create any

`PLOW_PF_INTERLEAVE` (Layer::Runtime, rows per launch, 0 = uncapped) at 8192/C32, verified from
`server.log` as `pf_interleave: Some(N)` rather than assumed:

| rows/launch |     TTFT |   TPOT | p99ITL |  tok/s | beats vLLM |
|------------:|---------:|-------:|-------:|-------:|------------|
| vLLM 0.28   |  3657.96 |  67.49 | 353.90 |  332.2 | — |
| uncapped    |  4395.14 |  81.95 | 207.35 |  270.5 | 1/4 |
| 2048        |  6132.05 | 122.35 | 177.44 |  184.0 | 1/4 |
| **1024**    |  7874.86 |  **55.07** |  **64.65** |  246.5 | **2/4** |
| 512         | 16640.34 |  49.28 |  58.06 |  150.2 | 2/4 |

TPOT **is** reachable — 55.07 beats vLLM's 67.49, p99 ITL collapses to 64.65. But total per-request
latency, TTFT + 128 x TPOT, is **flat**: uncapped 14884 ms, 1024 14924 ms, and both 512 (22948) and
2048 (21793) are worse. vLLM is 12297 ms. The knob moves time between the prefill phase and the
decode phase and creates none.

The prediction that 2048 would halve the tick for ~0.4% was **wrong**, and the reason is worth
keeping: the cost curve at `mux.rs:3828` (launch ms 16.8/26.6/45.2/86.6/172.3 at
128/512/1024/2048/4096 rows) prices **prefill alone**. Every launch also carries a decode step,
so capping rows multiplies launches AND decode steps. That is why tok/s falls in every capped arm.
2048 is also non-monotonic (TPOT +50% against uncapped) and unexplained.

### Where the ~18% end-to-end deficit actually is

Wall for the cell: plow 30.3 s (8192 gen / 270.5 tok/s), vLLM 24.7 s. Against measured per-stack
rates — plow prefill at its own C1 rate 23.0k tok/s = 22.8 s, decode 256 steps x 19.0 ms = 4.9 s,
total 27.7 s predicted against 30.3 s measured:

* **decode step ~2.1 s (38%)** — plow 19.0 ms vs vLLM ~11 ms at B=32 on the same 23.8 GB.
* **unexplained overhead ~2.6 s (46%)** — the largest single unknown, and nothing yet measures it.
* **prefill kernel ~0.5 s (9%)** — plow is 2% off vLLM at C1/8192 (355.78 vs 348.92).

So the decode step is NOT irrelevant to throughput even though it is nearly irrelevant to TPOT;
those are different questions and the earlier "kernel work cannot reach it" was too strong. The
honest target is ~18-20% more end-to-end throughput, and the biggest unattributed piece is the
2.6 s overhead term.

## Where the 8192/C32 wall actually goes (measured, not residual)

Every prior split of this cell was `wall - estimated_prefill`, and residual-chaining produced two
wrong conclusions this campaign. This is the direct measurement: one served run of the ladder cell
(rc1024permt, 64 prompts, in 8192, C32, out 128) under `PLOW_PF_PACKLOG=1`, 491 ticks.
`out_tok_s` 270.0 reproduces the ladder cell exactly, so this is the cell, not a proxy.

    sum prefill_ms = 27.50 s
    sum decode_ms  =  4.89 s
    accounted      = 32.40 s     vs a 32.36 s wall  -- the split is complete
    (span 48.28 s; the difference is server idle outside the bench, not overhead)

    prefill ticks 171 / decode-only ticks 320
    decode_ms on prefill ticks = 0.00 s for all 171

The last line is not starvation: `packlog::tick` reads `feeds.len()` AFTER the unified pass has
cleared it (mux.rs:2101), so riding decode is billed inside `prefill_ms`. The 27.50 s therefore
contains both the prefill rows and the decode rows that rode with them.

### The decode kernel is not the problem

    decode-only, rows=32 : median 19.100 ms  (n=98, total 1.88 s)
    step_bench B=32 ctx8192 :     19.036 ms

Served == isolated, to 0.3%. There is no mixed-phase decode pathology to fix, and decode-only is
only 4.89 s of a 32.40 s wall. Tasks #59-#64 have been tuning 15% of the wall.

Marginal cost per decode row, from the same table: rows=1 is 10.785 ms and rows=32 is 19.100 ms,
so 0.268 ms/row over a 10.785 ms fixed weight pass. That is the per-row term task #64 names, and
it is real -- but it is worth at most 8.3 ms of a step that runs 98 times, i.e. ~0.8 s of 32.40 s.

### Prefill is 85% of the wall, and its GEMMs are already near the roofline

Per-site attribution (`PLOW_PF_SEG_TIME=1`, one decoder block, 8192 tokens, ctx 8192), against
H100 SXM5 BF16 dense peak 989 TFLOP/s. Gemma-4-12B is hybrid: 40 sliding layers (window 1024,
hd 256) + 8 full layers (hd 512), hidden 3840, ffn 15360, 16 Q / 8 KV heads.

    sliding block, 6.394 ms            full block, 9.172 ms
      Gemm q     0.326   791 TF/s        Gemm q     0.646   798 TF/s
      Gemm k     0.167   772 TF/s        Gemm o     0.635   811 TF/s
      Gemm v     0.164   786 TF/s        Gemm gate  1.223   790 TF/s
      Gemm o     0.331   779 TF/s        Gemm up    1.221   791 TF/s
      Gemm gate  1.232   784 TF/s        Gemm down  1.267   763 TF/s
      Gemm up    1.238   780 TF/s        FlashPref  3.130   351 TF/s
      Gemm down  1.274   758 TF/s
      FlashPref  0.701   196 TF/s

40 x 6.394 + 8 x 9.172 = 329.2 ms predicted vs 355.78 ms measured TTFT at 8192/C1 (-7.5%), so the
block model is validated and can be used to budget the whole model:

    GEMM (linear)        229.6 ms   70%    758-811 TF/s = 77-82% of peak
    FlashPrefill          53.0 ms   16%    196 / 351 TF/s = 20-36% of peak
    norms + Glu + rope    46.4 ms   14%    1.4-1.9 TB/s = 41-57% of HBM

70% of prefill already runs at ~80% of the roofline. vLLM is bounded by the same GEMM physics on
the same silicon, so prefill GEMM cannot be where the 7.6 s gap lives, and the earlier idea that
plow's prefill is 58% efficient was an artifact of counting only linear FLOPs against the wall.

### What is actually addressable

    4.73 s   prefill excess over the C1 rate (524288 tok / 23.0k tok/s = 22.77 s vs 27.50 s).
             Padding plus riding decode; NOT yet decomposed -- packlog cannot separate them
             because both are billed to prefill_ms. This is task #63 and it needs the unified
             pass to time decode rows separately from prefill rows.
    ~1.8 s   FlashPrefill at 20-36% of peak vs the GEMM path's 78%. 53.0 ms/355.78 ms of prefill;
             closing half of it is ~8% of prefill.
    ~2.0 s   the norm/Glu/rope tail, 14% of prefill, at 41-57% of HBM. Epilogue fusion.

    27.50 - 4.73 - 1.8 - 2.0 = 18.97 s prefill + 4.89 s decode = 23.9 s  vs vLLM 24.7 s

That is the first arithmetic in this campaign that reaches the goal, and none of its three terms
is the decode GEMV walk.

### Decomposing the 4.73 s: it is rung holes, not decode cost

Re-ran the same cell with `PLOW_PF_NO_INTERLEAVE=1`, which moves every decode row out of the
prefill pass, so `prefill_ms` becomes pure prefill. No code change needed.

                       unified     no-interleave
    out_tok_s           270.0         281.4        +4.2%
    TTFT ms            4424.14       4212.24
    TPOT ms              81.76         79.00
    p99 ITL ms          206.25        213.34
    sum prefill_ms      27.50 s       23.27 s
    sum decode_ms        4.89 s        7.82 s
    accounted           32.40 s       31.08 s      -1.32 s

So riding decode costs 4.23 s inside the prefill pass and saves only 2.93 s of decode-only time:
the unified pass is a NET LOSS of 1.30 s at this cell, which the 1.32 s wall delta confirms
independently. Pure prefill is 23.27 s against 22.77 s at the C1 rate, so rung/padding waste is
only 0.50 s -- far less than the 7.8% that had been assumed.

But the aggregate hides the real mechanism. Joining `PACKLOG PACK` (decode_feeds) to `PACKLOG R=`
(rows, bucket) to the tick's prefill_ms, over the 171 launches (168 of which carry decode):

    pf_rows=4068 + 28 feeds = 4096 -> bucket 4096   194.55 ms  n=45   fits the rung exactly
    pf_rows=4095 +  1 feed  = 4096 -> bucket 4096   182.25 ms  n=8
    pf_rows=2048 + 30 feeds = 2078 -> bucket 4096   163.68 ms  n=17   NO RUNG between 2048 and 4096
    pf_rows=1024 + 28 feeds = 1052 -> bucket 1088    68.35 ms  n=33   +64 rung, 6% pad, fine

The two 4096-total shapes price a riding decode row directly: 27 extra rows cost 12.30 ms, i.e.
**0.456 ms per decode row inside a prefill launch**, against 0.644 ms/row for a standalone batch
of 28 (10.785 ms fixed + 27 x 0.268). Riding decode is CHEAP when it fits the rung.

It is expensive only when it pushes across a rung hole. `2048 + 30 = 2078` has no bucket between
2048 and 4096, so 30 decode rows cost ~50 ms (163.68 vs ~114 for a 2048-rung launch) -- 1.67 ms
per decode row, 3.7x the in-rung price. That single shape, 17 launches, is ~1.1 s.

`mux.rs:4333` already trims for this and it is not the bug: the trim fires when the pack OVER-fills
a bucket, but here the pack UNDER-fills. Only 2 requests still had prefill work
(max_request_chunk=1024, so 2 x 1024 = 2048), which is queue drain, not over-packing. The fix is a
rung covering 2049-2112, not a scheduler change. Note 1088 = 1024+64 works in production, so the
"+64 rungs wedge" note from the finer-rung attempt (#51, the 2560 rung) does not generalise to all
appended rungs and the specific failure should be re-examined before ruling 2112 out.

### The whole remaining budget, priced

    ~1.1 s  rung hole at 2049-2112 (the 2048+30 shape)
    ~2.3 s  FlashPrefill at 196 (sliding) / 351 (full) TFLOP/s vs the GEMM path's ~780
    ~2.0 s  norms + Glu + rope, 14% of prefill at 41-57% of HBM (epilogue fusion)
    ~1.3 s  the unified pass's net loss (available today as PLOW_PF_NO_INTERLEAVE=1, but it
            trades p99 213.34 for 206.25, so it is a dial and not a free win)

    32.40 - 6.7 = ~25.7 s  =>  out_tok_s ~319 against vLLM's 332.2

Sliding attention is NOT doing wasted work: at ctx 4224 the second chunk (3968 queries) costs
0.389 ms against the first chunk's (4224 queries) 0.480 ms. Full causal would have made the
second chunk 2.76x MORE expensive, so the window is being exploited and the 196 TFLOP/s is real
inefficiency. There is no cheap 4x win there; it is a kernel project.

Honest read: these levers take 8192/C32 to roughly throughput parity while winning TTFT, TPOT and
p99 -- 3 of 4 metrics, not 4 of 4. Nothing measured so far closes the remaining ~4% of tok/s.

### Prefill is at its ceiling; the unified pass's per-row decode cost is the lever

Two more served runs at the same cell on the chunk-2048 packet (rc2048), with and without the
unified pass, settle two things the rc1024 data could not.

**There is no multi-span penalty.** With decode out of the launch, a 4096-row prefill launch costs
175.640 ms (n=120), i.e. 0.0429 ms/row -- identical to the C1 single-request rate of 0.0434 ms/row
(8192 rows in 355.78 ms). The earlier hypothesis that served launches pay 9.4% for carrying
several request spans is REFUTED; that 9.4% was entirely riding decode. plow's prefill floor is
524288 x 0.0429 = 22.5 s and the kernel is at its ceiling.

**Riding decode shares the weight pass only partly, and pays 1.9x per row.** Within one run, at
the same bucket, pricing prefill rows at 0.0429 ms/row:

    4095 prefill + 1 decode  = 182.25 ms  ->  1 decode row costs  6.55 ms
    4068 prefill + 28 decode = 194.55 ms  -> 28 decode rows cost 20.05 ms

    => riding decode   = 6.05 ms fixed + 0.50 ms/row
       standalone step = 10.785 ms fixed + 0.268 ms/row   (measured twice, 0.268 / 0.269)

So the fixed weight pass IS partly shared (6.05 vs 10.785), but the marginal per-row cost is 1.9x
worse. At 28 rows: riding 20.05 ms vs 18.02 ms standalone -- riding is 11% WORSE, which is exactly
why PLOW_PF_NO_INTERLEAVE=1 wins throughput. The interleave is not buying what it was built to buy.

The likely cause is that decode rows inside the unified pass go through the PREFILL attention path
rather than the decode attention kernel; a 1-row query is the worst case for a prefill flash tile.
If the marginal cost came down to the standalone 0.268 ms/row, riding 28 rows would cost
6.05 + 27 x 0.268 = 13.3 ms against 18.02 standalone -- a 4.7 ms saving per launch, ~0.80 s over
171 launches, AND it would make interleaving a net win so that more decode could ride.

Today only 42.2% of decode token-steps ride a prefill launch; 57.8% pay a private weight pass
(8393 token-steps counted against 8192 expected output tokens, so the accounting closes).

### Arm table at 8192/C32 (vLLM: 3657.96 / 67.490 / 353.90 / 332.2)

    arm                out_tok_s   TTFT    TTFT_med   TPOT    p99 ITL   wins
    rc1024 uncapped      270.0    4424.14  1971.44   81.76    206.25    p99
    rc1024 no-ilv        281.4    4212.24  1784.99   79.00    213.34    p99
    rc2048 uncapped      273.7    3874.06  1193.90   84.52    205.52    p99
    rc2048 no-ilv        278.7    3778.68   996.84   83.31    217.28    p99
    rc1024 ilv=1024      246.5    7875     -         55.07     64.65    TPOT + p99

rc2048 is the TTFT lever: 4424 -> 3874 uncapped, and its MEDIAN TTFT of 996.84 ms beats vLLM's
1662.94 outright. Mean TTFT 3778.68 is within 3% of vLLM's 3657.96. No arm yet clears more than
2 of 4, and tok/s remains the hard blocker on every one.

### The throughput blocker is the decode weight pass at 67% of HBM

The interleave cap was swept on rc2048 to see whether rc2048's TTFT headroom could buy the
TPOT/p99 wins without the throughput collapse. It cannot:

    arm              out_tok_s   TTFT      TPOT    p99 ITL   wins
    rc2048 uncapped    273.7    3874.06   84.52    205.52    p99
    rc2048 ilv=1024    245.6    7908.83   55.06     64.29    TPOT + p99
    rc2048 ilv=512     150.6   16597.25   49.18     58.12    TPOT + p99
    rc2048 ilv=256     120.5   22951.28   31.90     35.69    TPOT + p99
    vLLM               332.2    3657.96   67.490   353.90

Tightening the cap monotonically trades TTFT and throughput for TPOT and p99. This reconfirms
task #50 on a second packet: PLOW_PF_INTERLEAVE is a dial, not a throughput lever, and NO setting
clears more than 2 of 4.

That leaves total work, and plow's best arm does ~31.3 s of GPU work against vLLM's 24.66 s wall.
Prefill is already at its kernel ceiling (previous section), so the deficit is the decode step.

**The decode weight pass runs at 67% of HBM peak.** This needs no KV estimate:

    B=1, ctx 128, rc1024permt : 10.766 ms
    12B weights               : 40 sliding x 224.1M + 8 full x 271.3M + 1.007B embedding
                              = 12.14B params x 2 B = 24.3 GB
    => 24.3 GB / 10.766 ms    = 2.26 TB/s = 67.4% of the H100's 3.35 TB/s

At B=32 ctx 8192 the step is 19.65 ms. vLLM's median ITL at the same cell is 15.82 ms -- a 24%
faster step. Using a derived KV figure (32 seqs x ~705 MB with attention_k_eq_v=true) the two
land at plow 71% and vLLM ~88% of the roofline, but the ROBUST claim is the ratio 19.65 vs 15.82,
which needs no KV model at all.

This is the throughput gap, and it is the one thing that would flip out_tok_s. If the decode step
matched 15.82 ms, decode would fall from 7.98 s to ~6.4 s and the wall to ~29.8 s (tok/s ~293);
adding the priced prefill terms (~2.0 s norm/Glu fusion, ~2.3 s FlashPrefill) reaches ~25.5 s and
tok/s ~343, which would finally clear vLLM's 332.2.

The hard part: the usual route to a faster weight pass is occupancy, and that is CLOSED here by
arithmetic already done in this campaign -- decode live state is ~463 registers, so the 128-reg
cap needed for more resident waves costs 1.3-1.9x in spill. Prefetch depth has already been tuned
(per-MT depth, -3.6% at B=32). So reaching ~90% needs a different idea than occupancy or prefetch,
or it needs to read fewer bytes (FP8 weights), which a bf16-vs-bf16 ladder forbids.

Also note the closed-batch tail, which is NOT the gap but is worth knowing: 142 decode ticks at
<=2 rows burn 1.60 s producing 273 of 8393 token-steps (5.9 ms/token vs 0.614 at rows=32). vLLM
pays the same tail in a 64-request closed batch.

### Two refutations withdrawn: the probe mechanism measured nothing

`PLOW_EXTRA_DEFINES` is NOT plumbed into `campaign.py build`. Its knob is registered as

    KnobSpec::new("def.PLOW_EXTRA_DEFINES", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN)

and the second field -- the environment binding -- is None. Only `build_sm90a_cubin.sh`,
`build_sm120_cubin.sh` and the tune sweeps read it, and those build the GENERIC object, which is
a different kernel anyway. So every probe that gated a source edit behind `#if defined(FOO)` and
enabled it with `PLOW_EXTRA_DEFINES=-DFOO=1` compiled the ORIGINAL source and compared the
shipped kernel against itself.

Withdrawn on that basis:
  * "the B=32 step is NOT activation-bound" (commit e0476839, +0.01% / +0.03%)
  * "NOR IS IT TENSOR-CORE ISSUE" (commit 01983f92, +0.09% / 0.00%)

Those deltas are run-to-run noise. The REG:255 STACK:192 SHARED:40464 gate passed because the two
objects WERE the same object -- a resource-signature match is exactly what the broken path
produces, so it was never a gate at all. Caught only when the same mechanism was used to halve
the weight bytes the walk streams and returned 0.00% across three cells, which is physically
impossible. `grep -c GVMMA_PROBE_HALF_K <out>.build.log` returns 0.

Rule going forward: edit the source UNCONDITIONALLY and gate on the probe cubin's md5 DIFFERING
from the control's. A null that is too clean is evidence of a no-op, not of a refutation.

### What the decode step is actually made of

Re-measured with an unconditional half-K edit (`gvmma_tile` walks half its k-block range, so all
three walks stream half their weights), two real packets, md5-gated. step = F + W, so
W = 2*(step-half) and F = 2*half - step.

    B    ctx     step     half        W        F
    1    128    10.765   6.683    8.164    2.601
    32   128    13.690   8.932    9.516    4.174
    32   8192   19.036  14.250    9.572    9.464

Against ~22.3 GB of layer weights and 3.35 TB/s:

    B=1  : 22.3 / 8.164 = 2.73 TB/s = 81.5% of roofline
    B=32 : 22.3 / 9.516 = 2.34 TB/s = 70%

**The "decode weight pass runs at 67% of HBM" figure was an artifact** -- it divided total weight
bytes by the WHOLE step, which also contains attention, norms, sampling and entry overhead. The
walk alone is at 81.5% at B=1. Correct the earlier section accordingly.

Two things follow. First, batching costs this walk +1.35 ms on a BYTE-IDENTICAL weight stream
(gvmma_partition depends on N and nblk, not on rows), so the withdrawn "the batching cost is not
in this walk" conclusion is wrong in its own terms -- ~1.35 ms of it is exactly there. Second,
the headroom is smaller than advertised: taking the B=32 walk from 70% to 90% returns ~2.1 ms of
the 19.036 ms step, not the ~5 ms the 67% figure implied.

At B=32 ctx 8192 the step is almost exactly half walk (9.572) and half everything-else (9.464).
F grows +5.29 ms from ctx 128 to 8192 -- that is the KV traversal -- and carries 2.601 ms of
fixed cost already present at B=1 ctx 128 (entry, norms, sampling).

Against vLLM's 15.82 ms median ITL at the same cell, plow needs -3.2 ms. The walk can supply
~2.1 ms at best, so the remaining ~1.1 ms has to come out of F, where the 2.601 ms fixed term is
the obvious candidate. Neither is a knob.

### Re-run correctly: the walk IS activation-bound at B=32

Both withdrawn probes rebuilt with UNCONDITIONAL source edits, each packet gated on its cubin
md5 differing from the control's.

    probe                    B=32 ctx128        B=32 ctx8192       B=1 ctx128
    a1 = a0 (activations)    13.689 -> 12.548   19.032 -> 17.803   10.762 -> 10.489
                             -1.141 (-8.34%)    -1.229 (-6.46%)    -0.273 (-2.54%)
    half mma issue           13.689 -> 12.676   19.032 -> 17.966   10.762 -> 10.022
                             -1.013 (-7.40%)    -1.066 (-5.60%)    -0.740 (-6.88%)

The no-op versions of these same probes reported +0.01% and +0.09%. Subtracting the B=1 effect
isolates each term's share of the +1.35 ms that batching adds to the walk (8.164 -> 9.516 ms on a
byte-identical weight stream):

    activations  1.141 - 0.273 = 0.868 ms   <- dominant
    mma issue    1.013 - 0.740 = 0.273 ms

So PAIR=4 -- which the invalid probe told us not to build -- is the indicated change (task #70).
wgmma, worth 0.273 ms, is not worth its complexity yet.

Two caveats, both recorded in the header. Both probe objects came out REG:247 STACK:160 against
the control's REG:255 STACK:192, so part of the gain may be lower register/stack pressure rather
than the removed work; occupancy is unchanged (both fit one 256-thread block per SM) and LOCAL:0
in all three, so the confound should be small but is not zero. And a probe that DELETES work is
an upper bound on what a rearrangement recovers: PAIR=4 needs UNB halved 8 -> 4 to hold the same
prefetch register budget, trading activation traffic against the very prefetch depth the per-MT
fix was tuned for.

### Where the goal stands

0.868 ms of a 19.036 ms step is 4.6%, worth ~0.3 s of the 31.3 s wall at 8192/C32. Reaching
vLLM's 15.82 ms step needs -3.2 ms. So PAIR=4 alone does not flip out_tok_s, and the ranked
remainder is unchanged: ~2.1 ms available in the walk (70% -> 90% of roofline), ~1.1 ms that must
come from F (of which 2.601 ms is fixed entry/norm/sampling cost), plus the prefill terms
(#66 FlashPrefill ~2.3 s, #67 norm/Glu tail ~2.0 s) and the ~1.1 s rung hole (#65).

### PAIR=4 is blocked, and F's fixed term is the larger target anyway

The mandate from the corrected probes is real -- the walk IS activation-bound -- but the code
shape blocks the obvious fix. `gemv_rows_mma`'s PAIR block has two branches:

  * WIDE-N (`per >= 2u * PLOW_NV_WARPS`, op_gemv_mma.cuh:358) needs no shared memory, so PAIR=4
    is a trivial edit there. But for the 12B almost nothing reaches it. Row blocks per block,
    N/8/132: lm_head 262144 -> 248 YES; gate/up 15360 -> 14.5 (just under the 16 threshold);
    q 8192/4096 -> 7.8/3.9; down 3840 -> 3.6. It covers lm_head alone, ~2.0 GB of the 24.3 GB
    weight pass, worth ~0.07 ms of a 9.516 ms walk.
  * SPLIT-K PAIR, where gate/up/down/qkv/o actually go, needs `red2[2]` -> `red4[4]`.
    `gvmma_red_t<MT>` is float[WARPS-1][32][MT*4] = 7168 B at MT=2/WARPS=8, so 14336 -> 28672 B.
    On an object already at SHARED:40464 that lands near 54.8 KB, past the 48 KB static limit.

Making it work means the split-K reduction must stop scaling with NW (two passes through the
existing slots, or a register/shuffle reduction) -- a redesign, not a template parameter, for a
payoff capped at 0.868 ms that is itself an upper bound and is traded against halving UNB 8 -> 4.

Meanwhile F deserves the attention. At B=32 ctx 8192, F is 9.464 ms and splits as:

    +5.29 ms   KV traversal (the ctx 128 -> 8192 growth). Geometry says ~705 MB/seq x 32 =
               22.6 GB, which over 5.29 ms would be 4.27 TB/s -- ABOVE the 3.35 TB/s roofline.
               So the geometry overestimates (the sliding window is not full at every layer),
               and the honest reading is that this term is at or near roofline. Not a target.
     2.601 ms  FIXED, already present at B=1 ctx 128: megakernel entry, norms, sampling.
    ~1.57 ms   batch-dependent remainder (F goes 2.601 -> 4.174 from B=1 to B=32 at ctx 128).

**The 2.601 ms fixed term is larger than the walk's entire ~2.1 ms of headroom**, and the step
needs -3.2 ms to reach vLLM's 15.82 ms. Together they would more than cover it. The decode entry
cost has prior history (the entry function taxes every rung; arena bytes cost ms on the 26B), so
this is the next thing to size -- with a probe that strips layer work down to entry + sampling,
built unconditionally and md5-gated.

### The fixed per-step decode cost is ~1.3 ms (megakernel entry), measured without a rebuild

`PLOW_DEBUG_MAX_INST` caps the interpreter's instruction count through a module global
(gpu.rs:3714), so the decode step can be swept against instructions executed with no rebuild.
Every capped run was gated on the runtime's own `SET plow_debug_max_inst = <n>` line; all 16
passed, and cap=766 reproduces the uncapped step (13.688 vs 13.689 at B=32, 10.767 vs 10.775 at
B=1), so the mechanism is clean and does not itself distort.

    cap        B=32 ms    B=1 ms          marginal us/inst   B=32     B=1
      1          1.352     1.070            1 ->   32        18.00   13.84
     32          1.910     1.499           32 ->   64        21.62   17.28
     64          2.602     2.052           64 ->  128        21.66   17.33
    128          3.988     3.161          128 ->  256        21.29   16.81
    256          6.713     5.313          256 ->  512        20.83   16.56
    512         12.046     9.552          512 ->  540        58.6    42.6
    766/full    13.688    10.767

    linear fit over 1..512:  B=32  21.05 us/inst, intercept 1.285 ms
                             B=1   16.70 us/inst, intercept 1.010 ms

NOTE a correction: the DECODE program is 540 instructions, not 766 -- 766 is the PREFILL program
(build.json kernel_cases.programs[8..13] are kind=decode, instruction_count=540). So the range
512 -> 766 is not a cheap tail of 254 instructions; it is the decode program's last 28 real
instructions plus a no-op range, and those 28 cost 1.642 ms = 58.6 us/inst -- the MOST expensive
in the program, which is what lm_head (2.0 GB over a 262144 vocab) plus the final norm and
sampling should cost. An earlier reading of this as cheap gated arms was wrong.

**~1.285-1.352 ms of every B=32 decode step is spent before the first instruction does any work**
-- 9.4% of the 13.688 ms step. That is megakernel entry (grid sync, shared-memory claim, per-block
state init across 132 blocks), not launch latency, which is ~10 us. Note the arena/smem claim is
already known to cost ms on the 26B, so this has precedent; but 40464 B x 132 blocks is only
5.3 MB, which at 3.35 TB/s is ~1.6 us, so ZEROING is not the mechanism and the cause is still
unidentified.

So the B=32 ctx128 step decomposes as:

    1.285 ms   entry, before instruction 1
   10.76  ms   body, 511 instructions at 21.05 us
    1.642 ms   final 28 instructions (lm_head, final norm, sampling)
   -------
   13.688 ms   and at ctx 8192, +5.35 ms of KV traversal -> 19.036

Cross-checks against the half-K split: walk 9.516 + F 4.174 = 13.690 at ctx 128. Consistent.

### Does this close the gap?

The step needs -3.2 ms to reach vLLM's 15.82 ms. Available and now quantified:

    ~2.1 ms   walk headroom (70% -> 90% of roofline at B=32) -- but PAIR=4, the measured fix, is
              blocked by the split-K shared-memory budget, so this needs a reduction redesign
    ~1.3 ms   megakernel entry, cause not yet identified

Together 3.4 ms, which would cover the 3.2 ms. Neither is a knob and neither is close to free,
and realizing both in full is optimistic -- but for the first time the decode side of the goal
has a route whose terms are all measured rather than assumed.

### The interpreter's dispatch is free; ~1.25 ms per decode step is fixed overhead

`PLOW_NV_SKELETON` runs the gate/signal skeleton with no op bodies (interp_sm120.cu:2911). Its
knob `def.PLOW_NV_SKELETON` has env binding None, like PLOW_EXTRA_DEFINES, so it was enabled by
flipping the `#ifndef` default in the source and gated on the cubin md5 differing.

First attempt FAILED and produced no numbers: the built-in `PLOW_NV_SKEL_PAD` of 160 KB puts
static smem at 164868 B and static+dynamic then exceeds the device opt-in (232448 B), so every
run died on `cuFuncSetAttribute(max dynamic smem) -> CUDA_ERROR_INVALID_VALUE`. Base static is
164868 - 160*1024 = 1028 B, so PAD=38 gives 39940 B, matching the control object's 40464 and
therefore its launch profile. With that it runs:

    B    ctx     control   skeleton   op_work
    32   128      13.689     1.246     12.443
    1    128      10.770     0.983      9.787
    32   8192     19.034     1.273     17.761

**Dispatch over all 540 stream entries costs ~0.** The skeleton (entry + 540 gate/signal hops) is
1.246 ms, no more than entry alone measured by PLOW_DEBUG_MAX_INST=1 (1.285-1.352 ms on the
heavier REG:255 object; the skeleton is REG:28, which accounts for its being slightly lower). The
counter protocol, the atomic cursor and the 540 sequential dependency hops are not a tax. There is
no scheduling overhead to remove -- which also retires the idea that role-segmenting decode would
pay for itself through cheaper dispatch.

What remains is a **fixed ~1.25 ms per decode step**, and it is invariant where real work is not:
1.246 ms at B=32 vs 0.983 at B=1, and 1.246 at ctx 128 vs 1.273 at ctx 8192. That is 9.1% of the
13.689 ms step and 39% of the 3.2 ms needed to reach vLLM's 15.82 ms.

Now excluded as the cause: dispatch (this section), smem zeroing (40464 B x 132 blocks = 5.3 MB,
~1.6 us at 3.35 TB/s), and CUDA-graph launch overhead -- decode is ONE cooperative megakernel
launch, so a one-node graph would save ~5-10 us, not 1.25 ms, and the graph APIs are already
bound and used for prefill segments (rt.pf_seg_graph, ON/PROMOTED). A cooperative launch itself
is ~10-30 us. The cause is still unidentified; #71 stays open with these three ruled out.

Full decode step at B=32 ctx 128, every term now measured:

    1.25 ms   fixed entry/launch (cause open)
    9.52 ms   weight walk (2.34 TB/s = 70% of roofline; ~2.1 ms headroom, PAIR=4 blocked)
    2.92 ms   the rest of the op work (norms, attention, lm_head+sampling tail 1.642 ms)
   -------
   13.69 ms   + 5.35 ms KV traversal at ctx 8192 -> 19.03, vs vLLM's 15.82

## RUNG BY RUNG: plow already sweeps three cells 4/4

This should have been the first thing checked. The campaign has been judged on 8192/C32 -- the
hardest cell in the grid -- and reported as "0/4 clean cells". Comparing the whole ladder against
`perf-data/campaign/gemma4-12b.h100.reference-vllm028-bf16.csv` shows otherwise.

From ONE run, ONE packet (df11734dd142dc26, commit 06fdd7ca6e10, provisional=0, the realtime
profile with PLOW_MULTISTEP=0 + PLOW_DECODE_PIPELINE=1), request counts matched to vLLM at
32 reqs / 4096 gen tokens in every cell:

    cell        TTFT plow/vLLM    TPOT          p99 ITL        tok/s          wins
    128/C1       18.33/ 30.04*    10.42/10.46*  10.50/11.27*   95.40/94.20*   4/4
    1024/C1      46.02/ 47.24*    10.50/10.54*  10.59/11.37*   92.80/92.40*   4/4
    4096/C1     169.55/170.08*    10.53/10.55*  10.74/11.39*   84.90/84.80*   4/4
    128/C4       36.87/ 55.31*    10.59/10.49   11.26/11.34*  369.00/368.80*  3/4
    8192/C1     355.78/348.92     10.56/10.55   10.75/11.52*   75.40/75.80    1/4
    15000/C1    737.17/671.77     10.62/10.56   10.85/11.48*   61.30/63.60    1/4

The C1 column is three-fifths swept. 128/C4 misses only on TPOT, by 1% (10.59 vs 10.49).

### 8192/C1 is the closest unflipped cell, and it is very close

    TTFT   355.78 vs 348.92  -- lose by 6.86 ms (2.0%)
    TPOT    10.56 vs  10.55  -- lose by 0.01 ms (0.1%)
    p99     10.75 vs  11.52  -- WIN
    tok/s   75.40 vs  75.80  -- lose by 0.5%

tok/s at C1 is not independent: per-request wall is TTFT + 127*TPOT, so plow is
355.78 + 127*10.56 = 1696.9 ms against vLLM's 348.92 + 127*10.55 = 1688.8 ms. The whole cell
turns on **8.1 ms per request**. That can come from either end:

  * -8.1 ms of TTFT = 2.3% of prefill. FlashPrefill is 53.0 ms of 329.2 ms (16%) at 196 TF/s
    sliding / 351 TF/s full against the GEMM path's ~780, so a 15% improvement there covers it
    outright (#66).
  * or -0.064 ms of TPOT = 0.6% of the decode step. The step has 1.25 ms of fixed entry (#71)
    and ~2.1 ms of walk headroom, so 0.6% is well inside what either lever would return.

15000/C1 needs -73 ms per request (TTFT -9.7%), which the same FlashPrefill lever (~9% of
prefill) very nearly covers.

### What this changes

The decode work at B=32 that has occupied this session targets 8192/C32, which needs -3.2 ms on
a 19.03 ms step and is the hardest cell in the grid. The C1 column needs 2% on prefill and is
two cells from a clean sweep. Prefill FlashPrefill (#66) is now the highest-value item in the
campaign: it is the shared lever for 8192/C1, 15000/C1, and the ~2.3 s prefill term at C32.

### The 8192 prompt splits 4224 + 3968 and pads 128 rows for nothing

Measured composition on the shipped ladder packet (sha df11734dd142dc26, PLOW_PF_PACKLOG=1, C1):

    in=4096    rows=4096  -> [4096]                     bucket 4096            0 padded
    in=8192    rows=8192  -> [4224, 3968]               buckets 4224+4096    128 padded
    in=15000   rows=15000 -> [4224,4224,4224,1816,512]  buckets ...2048,512  232 padded, 5 launches

First, the N+1/BOS assumption in the recipe does NOT hold for this bench path: a 4096-token
prompt is 4096 rows, not 4097. The 1088/4160 rungs were added for a row count that does not occur
at C1, so they are dead weight there (they may still serve the C4/C16 packs).

Second, 8192 rows split as [4224, 3968] pay 128 padded rows where [4096, 4096] pays NONE, for the
same two launches. At the measured 352.71 ms / 8320 bucket-rows = 0.0424 ms per bucket-row that
is ~5.4 ms -- two thirds of the 8.1 ms that 8192/C1 needs.

Cause, `Backend::pick_prefill_bucket` (exec/gpu.rs:8833):

    // While the largest allowed rung still FILLS, it is optimal outright:
    // minimal padding and minimal launches at the same time.
    let top = n_allowed - 1;
    if rem >= self.prefill[top].t as usize { return top; }

The claim is false. It is only true when the top rung also TILES the remainder. Here top=4224 and
rem=8192, so the shortcut returns 4224 and strands 3968 in the 4096 bucket. The cost-aware DP
directly below it -- which minimizes exactly `sum(bucket_t) + chunk_cost*launches` -- would pick
4096 and pad nothing: [4096,4096] costs 8192+2*512=9216 against [4224,4096]'s 8320+1024=9344.

The DP is O(rem/unit * rungs) = ~128*10 for this ladder, so the shortcut buys nothing measurable
and is only reachable for the long prompts where it does the most damage.

Predicted, before measuring: capping chunks at 4096 (`PLOW_PF_CHUNK=4096`, a RUNTIME knob, no
rebuild) reproduces the fix at 8192 and should REGRESS 15000, because it forces 4096*3 + 2712 into
a 4096 bucket = 960 more padded rows (~+41 ms) and throws away the packer's 5-launch tail plan.
The regression is the real test of the 0.0424 ms/row price.

### Measured: the bucket-picker defect is worth 4.23 ms, and the 15000 prediction was wrong

`PLOW_PF_CHUNK=4096` (runtime, emulates the fix by capping the chunk below the 4224 rung),
p12rw, 32 prompts, CPU-quiet, PACKLOG on in both arms:

    cell        control    cap4096      delta   predicted
    8192/C1     355.81     351.58      -4.23      -5.4     composition [4224,3968] -> [4096,4096]
    15000/C1    702.89     699.03      -3.86      +5.4     WRONG SIGN

Control reproduces the ladder cell to 0.03 ms (355.81 vs 355.78), so the noise floor is far
below the effect. The 8192 composition changed exactly as the simulator predicted.

The 15000 prediction was wrong because it priced the rung swap and ignored that the TAIL also
changes: [1816->2048, 512] becomes [1688->2048, 1024]. Solving the two cells together:

    c(4224) - c(4096) = 4.23 ms      (8192: one 4224 launch replaced by a 4096 one)
    c(1024) - c(512)  = 8.83 ms      (15000: 3 x -4.23 plus the heavier tail = -3.86)

so both cells improve and the model is consistent -- the error was reading the simulator, not
the model. The DP fix is strictly better than this knob at 15000 (pad 40 vs 360) and identical
at 8192, so it is the version to land.

Note the defect only fires when the top rung does NOT tile the remainder. On rc1024permt (top
rung 4096) greedy and DP agree exactly at every ladder cell. It is the appended 4160/4224 rungs
-- added for a BOS+N row count that does NOT occur on this bench path (a 4096-token prompt is
4096 rows, verified by PACKLOG) -- that become the greedy top pick and strand the tail.

### The launch fixed cost is ~5.2 ms, not the 21.7 ms the cost model assumes

`pf_chunk_cost` charges a launch 512 rows, which at the measured 0.0424 ms/bucket-row prices a
launch at 21.7 ms. Fitting the two single-launch cells instead:

    F + 1024r =  46.02        r = 0.0398 ms/row
    F + 4096r = 168.38        F = 5.2 ms per launch

(128/C1 does not fit this line -- 10.3 predicted vs 18.33 measured -- because small-M GEMMs are
memory-bound; the fit is only valid over the compute-bound 1024-4096 range.)

The two-launch 8192 costs 351.58 against 2 x 168.38 = 336.76, and that 14.82 ms excess is the
second chunk's longer-KV attention, which one launch would also pay. So collapsing 8192 into a
SINGLE launch buys the launch overhead, ~5.2 ms, not the 21.7 the model implies.

### What 8192/C1 needs, priced

    TTFT   355.78 -> 351.6 (bucket fix) -> ~346.4 (single launch)   vs vLLM 348.92   WIN
    tok/s  128000/(346.4 + 127*10.56) = 75.85                       vs        75.80   WIN (thin)
    p99    10.75                                                    vs        11.52   WIN
    TPOT   10.56                                                    vs        10.55   LOSE by 0.01

TPOT is the sole remaining blocker and it is NOT noise: the A/B read 10.660 in both arms at
8192 and 10.720 in both at 15000, identical to three decimals. A real ~0.02 ms decode win is
needed, which is 2% of the ~1.0 ms megakernel entry (#71) -- by far the cheapest source.

### RETRACTED: the stranded 128 rows are NOT pick_prefill_bucket's greedy shortcut

The section above ("The 8192 prompt splits 4224 + 3968 and pads 128 rows for nothing") named
`pick_prefill_bucket`'s `if rem >= top { return top }` as the cause. That is WRONG, and the fix
built on it is a no-op. Retracted on the measurement:

  * `pf_cover: false` in the running server, so the cost-aware DP branch -- the one holding the
    shortcut -- is the branch that executes. The shortcut IS reachable.
  * The DP provably cannot choose 4224 at rem=8192: [4096,4096] costs 8192 + 2*chunk_cost and
    [4224,4096] costs 8320 + 2*chunk_cost, so 4096 wins for ANY chunk cost.
  * Gating the shortcut on `rem % top == 0`, rebuilt and re-run on the same packet, changed the
    composition NOT AT ALL (still `rows=4224 bucket=4224` + `rows=3968 bucket=4096`, read from
    PACKLOG) and the cell not at all: 8192/C1 TTFT 355.60 against the control's 355.81.

So the 4224 slice is not chosen by bucket selection. It is the per-request SLICE CAP --
`pf_request_max_rows()` / `pf_chunk_rows()`, i.e. `PLOW_MAX_REQUEST_CHUNK`=4224 -- applied to the
prompt before a bucket is picked; the bucket then merely covers the 4224-row slice exactly.
`PLOW_PF_CHUNK=4096` works because it lowers that cap, not because it changes the DP.

What survives, all measured:

    8192/C1   355.81 -> 351.58 with PLOW_PF_CHUNK=4096   (-4.23 ms)  composition [4096,4096]
    15000/C1  702.89 -> 699.03                           (-3.86 ms)

That is a serve-env / recipe lever with no code change, and it has NOT yet been run across the
full ladder, so it is not landed. PLOW_PF_CHUNK also feeds a per-tick row budget (mux.rs:2002),
so part of the -4.23 ms may not be the slice at all -- a full-ladder arm is required before any
claim.

### The binary drifted: decode TPOT is ~0.1 ms worse than the run that scored 4/4

Same packet (df11734dd142dc26, doctor-verified), clean box (no co-tenant, 100% GPU, 51 C, no
throttle, load 1.05), full realtime ladder on today's binary:

    cell        TPOT now   TPOT 02:24   vLLM
    128/C1        10.52       10.42      10.46     <- was a WIN, now a LOSS
    1024/C1       10.59       10.50      10.54     <- was a WIN, now a LOSS
    4096/C1       10.63       10.53      10.55     <- was a WIN, now a LOSS
    8192/C1       10.66       10.56      10.55
    15000/C1      10.72       10.62      10.56

0 cells at 4/4 on this binary, against 3 on the 02:24 one. Prefill did not regress (15000/C1 TTFT
703.25 is the rt.pf_cover cert's expected 737 -> 703). This is decode only, ~1%, uniform across
context, and no commit since 02:24 touches the decode path: e13277fd's `debug_max_inst` writes are
load-time and gated on the env var, 69083825/1659c37a are comment-only in op_gemv_mma.cuh, and the
packet's cubins predate all of them.

**Consequence for the campaign: the "3 cells at 4/4" result is tied to the 02:24 BINARY, not just
the packet.** Any 4/4 claim must name the binary and be re-measured against a control built from
the same commit. Finding this ~0.1 ms is worth more than the 0.01 ms gap at 8192/C1.

### The binary is exonerated: TPOT is bit-stable in a session and drifts BETWEEN sessions

ABAB of two plowrt binaries -- HEAD (md5 af11c9c3) and the 02:24 ladder's commit 06fdd7ca
(md5 d1850bcd), gate-checked as differing -- on the same packet, same recipe, 128/C1, 32 prompts,
interleaved old/head/old/head in ONE session:

    old1   TPOT 10.520     head1  TPOT 10.520
    old2   TPOT 10.520     head2  TPOT 10.520     (sm_clock 1830 MHz in all four)

All four identical to three decimals. So:

  * **No binary regression.** The 7 commits since 06fdd7ca that touch crates/ are exonerated --
    consistent with the fact that NONE of them touches the decode path (packed_prefill and
    rt.pf_cover are prefill, devgen is build-time, op_seq is an example, and e13277fd's gpu.rs
    write is load-time and env-gated). A bisect would have chased nothing.
  * **Within-session TPOT variance is 0.000 ms** across four full server starts.
  * The 10.42 -> 10.52 shift is therefore a BETWEEN-SESSION property of the box. SM clock reads
    1830 MHz against a 1980 max; 1980/1830 = 1.082, and a small compute component of an otherwise
    memory-bound step scaling with clock lands near the ~1% observed.

**This invalidates the way every cell in this campaign has been scored.** The 4/4 margins at
128/1024/4096 C1 were TPOT 0.04, 0.04 and 0.02 ms -- 0.2-0.4% -- while between-session drift is
~1%. The vLLM reference CSV was also taken in an earlier session, so today's plow is being scored
against a baseline measured under different conditions, in an unknown direction.

A cell is only decidable when BOTH stacks are measured in the SAME session. Within a session the
measurement is essentially exact (0.000 ms), so same-session paired runs are not merely better --
they are the only form in which these margins mean anything. Re-measuring vLLM now to pair it
against today's plow ladder.

## RESOLVED: the C1 wins were never lost — the baseline was stale, not the engine

vLLM 0.28 re-measured in the SAME session as today's plow ladder (gpulease, coherence gate PASS,
32 prompts / 128 out, prefix cache off both sides, `--num-warmups 2 --seed 42` both sides,
request counts matched 32/4096 in every cell):

    cell        TTFT plow/vLLM    TPOT          p99 ITL        tok/s          wins
    128/C1       18.32/ 31.18     10.52/10.56   10.59/11.46    94.5/93.2      4/4
    1024/C1      45.93/ 47.51     10.59/10.64   10.68/11.56    92.0/91.5      4/4
    4096/C1     170.51/171.48     10.63/10.65   10.83/11.57    84.2/84.0      4/4
    8192/C1     355.60/348.61     10.66/10.65   10.86/11.58    74.9/75.2      1/4
    15000/C1    703.25/675.86     10.72/10.65   10.85/11.60    62.0/63.1      1/4

**vLLM drifted by the same ~0.1 ms plow did.** Its TPOT over the ladder went 10.46 -> 10.56,
10.54 -> 10.64, 10.55 -> 10.65, 10.55 -> 10.65, 10.56 -> 10.65 between the stored reference and
today, and its TTFT moved with it (128: 30.04 -> 31.18; 15000: 671.77 -> 675.86). Both stacks
moved together, so the relative standing never changed.

So the earlier "0 cells at 4/4" was an ARTIFACT of scoring today's plow against a reference
measured in a different session. There was no engine regression, and the binary bisect that
finding implied would have chased nothing -- which the ABAB had already shown independently
(two different binaries, TPOT 10.520 in all four arms).

Paired reference stored at
`perf-data/campaign/gemma4-12b.h100.reference-vllm028-bf16.paired-2026-09-23T18.csv`; the
canonical multi-cell reference is left untouched.

### The rule this establishes

Within a session the measurement is essentially exact (plow TPOT 0.000 ms spread over four
server starts). Between sessions the whole box moves ~1%, which is 3-5x the margins that decide
these cells (the 4/4 TPOT margins here are 0.04, 0.05 and 0.02 ms). **A cell is only decidable
when both stacks are measured in the same session.** Cross-session scoring is not apples to
apples in either direction -- it can invent a regression, and it can equally invent a win.

Still 1/4 and genuinely behind, now on trustworthy numbers:

    8192/C1   needs -8.3 ms of per-request wall (TTFT -7.0 and TPOT -0.01)
    15000/C1  needs -36.3 ms (TTFT -27.4 and TPOT -0.07)

p99 ITL is won at every C1 cell by 0.7-0.9 ms, which is the MULTISTEP=0 + DECODE_PIPELINE=1
profile doing its job.

## Branch review: the rung-selection and knob defects, found by dedicated review passes

### Rung / slice selection — my earlier retraction was HALF wrong

I retracted the `pick_prefill_bucket` diagnosis after gating its greedy shortcut measured as a
no-op. The measurement was right; the inference went one step too far. The review shows why:

`serve/mux.rs:4292` pre-clamps the span to `chunk_cap` BEFORE the planner is called:

    let chunk_cap = pf_chunk_rows().min(e.pf_request_max_rows());      // 4224
    let remaining = (n - withheld - s.pf_pos).min(chunk_cap);          // 8192 -> 4224

so `pick_prefill_bucket` never sees 8192 and cannot tile it. The shortcut is NECESSARY but not
SUFFICIENT: "do (a) without (b) and nothing changes", and equally (b) without (a) -- which is
exactly what I measured. The complete fix is both:

    (a) mux.rs:4292  hand the planner the UNCAPPED length plus the cap
    (b) gpu.rs:8833  drop the greedy top-rung shortcut (or gate on rem % top == 0)

Hand-evaluated, that yields [4096,4096] at 8192 (zero padding, 9216 vs 9344) AND leaves 15000's
5-launch tail plan intact (17792 vs 17920). It is therefore STRICTLY BETTER than the
`PLOW_PF_CHUNK=4096` knob, which buys the same 8192 win but regresses 15000 padding 232 -> 360.

**Root cause chain, closed.** The "+1 row for BOS" premise is FALSE on `/v1/completions` (already
measured twice: gemma4-26b tracker:838-842 and this file:216). The rungs 1088/1152/4160/4224 were
appended to swallow a +1 that never occurs on the bench path. And because `ladder16k` sets no
`PLOW_MAX_REQUEST_CHUNK`, `pf_request_max_rows()` falls back to `pf_max_rows()` = the ladder top =
**4224 -- the appended rung itself**. A rung added for a phantom BOS row is what sets the slice.

**Second P0: the `trim` guard is dead code.** mux.rs:4333 calls `pf_pack_budget`, which returns the
FIRST RUNG OF A PLAN, where the COVERING bucket was meant (`token_batch.rs:135` /
`gpu.rs:9884` both inline `find(|b| b.t >= rows)`). The threshold needs a +512 jump that a
+1..32-row decode bump can never produce, so the branch never fires. On the l8192 packets
(`pf_max_rows` 8192 > `pf_request_max_rows` 4224) a 4224-row slice + 3 decode rows takes the 8192
rung: ~3965 padded rows, ~158 ms. Enabling it can only remove padding.

**`pf_chunk_cost` = 512: do NOT move the shared default.** My "512 is 4x too high" note was too
confident. The DP is insensitive here ([4096,4096] wins at 512 AND at 131), while
`queue_pack_rows` is linear in it and carries the measured PF_INTERLEAVE_ADAPTIVE certificate
(1024 in: TTFT 167.0 -> 107.7). The per-row price is also rung-dependent -- the two-cell solve
gives 0.033 ms/row at the 4096/4224 step but 0.017 at 512/1024 -- so one scalar cannot serve both
readers. Split the knob or leave it.

### Knobs — the "looks real, compiles nothing" class

    knob_gen.rs STALE, and NOT benign     the regen adds a Check::Load constraint
                                          (stage_rows_requires_max_request_chunk) that plowrt
                                          currently CANNOT enforce, plus 2 KNOBS entries that
                                          shift every later index in holds(). Blocks the merge
                                          gate. One command: KNOB_GEN_WRITE=1.
    PLOW_STAGE_ROWS                       parsed, then lib.rs:9754 hardcodes stage_rows: None
    rt.block_packets, rt.pf_modular       PROMOTED with "=false is the rollback"; neither field
                                          is read anywhere in plowrt
    PLOW_BLOCK_STAGE                      2 specs, 2 validated clap flags, 0 consumers
    PLOW_EXTRA_DEFINES                    confirmed: plowc builds -D only from the manifest's
                                          `recommends`, never from the environment
    27 PLOW_BUILD_* selectors             unregistered, so they change the served object set
                                          without entering build.json or moving registry_digest;
                                          docs list 11 of them with the PLOW_BUILD_ prefix
                                          STRIPPED, the one form the gate cannot match
    def.PLOW_BUILD_SEG                    wrong layer; survives the reverse check only via a
                                          comment in runtime/CMakeLists.txt:504
    PLOW_VERIFY_BIN                       unregistered, and selects the checkpoint-K verifier

Root cause of the coverage hole: no registry test scans `scripts/`, and the plowrt/devgen scans
cover 4 crates, not `lean_verify`.

## PLOW_PF_CHUNK=4096 measured clean, and the rung fix that replaces it (2026-09-23T19)

**The knob is real but insufficient, and it is the wrong shape of fix.** Re-measured without
PACKLOG (which itself costs ~0.1 ms/step), same session, scored against the same-session vLLM
reference:

| cell | control TTFT | PF_CHUNK=4096 | delta | vLLM |
|------|-------------|---------------|-------|------|
| 128/C1   | 18.32  | 18.25  | -0.07 | 31.18 |
| 1024/C1  | 45.93  | 46.33  | +0.40 | 47.51 |
| 4096/C1  | 170.51 | 170.19 | -0.32 | 171.48 |
| 8192/C1  | 355.60 | 351.14 | **-4.46** | 348.61 |
| 15000/C1 | 703.25 | 699.81 | **-3.44** | 675.86 |

Both arms score 3 cells at 4/4. The knob does NOT flip 8192/C1: it closes the per-request wall
gap from +8.3 to +3.8 ms, no further.

**Root cause, now actually confirmed from the packet rather than guessed.** `build.json` for
p12rw gives `shapes.prefill_buckets = [128,256,512,1024,1088,1152,2048,4096,4160,4224]`,
`max_chunk = 4224`. So `pf_request_max_rows()` = 4224 and an 8192-row prompt was sliced
`[4224, 3968]`; 3968 has no rung and pads up to 4096. **128 padded rows**, which at the measured
marginal r = 0.0398 ms/row is ~5.1 ms — matching the -4.46 ms the knob buys by forcing
`[4096, 4096]` instead.

This vindicates the earlier retraction: `pick_prefill_bucket` alone was never the bug. Its greedy
shortcut and the mux's pre-clamp are *jointly* responsible, and fixing either alone is a no-op —
which is exactly what the first (reverted) attempt measured.

**The fix (two edits, only meaningful together):**
* `exec/gpu.rs` — deleted the "largest allowed rung that still FILLS is optimal outright"
  shortcut. It is not optimal: `[4224, 3968->4096]` and `[4096, 4096]` are the same two launches,
  but the first strands 128 rows. The DP below already considers the top rung and picks it
  whenever it genuinely wins, so the shortcut only ever overrode the DP with a worse answer.
* `serve/mux.rs` — the per-request slice now goes through the new `pf_plan_slice(rem, cap)`,
  which plans the **uncapped** remainder. Previously `remaining` was clamped to 4224 *before*
  the planner ran, so the planner never saw that the request was 8192 long.

Hand-traced: `pick_prefill_bucket(8192, 4224)` now costs `[4096,4096]` at 9216 vs `[4224,4096]`
at 9344 and returns rung 4096. `pf_pack_budget` moves 4224 -> 4096 in step, so the slice and the
per-launch budget stay coherent. 128/1024/4096 are single-launch exact fills either way and serve
as the control.

**Honest ceiling on 8192/C1.** Even a perfect prefill fix does not make this cell 4/4. The four
metrics stand at TTFT 351.14/348.61, TPOT 10.66/10.65, p99 10.89/11.58, tok/s 75.1/75.2. TTFT,
p99 and tok/s all flip on a ~4 ms prefill win, but **TPOT is a decode metric and prefill cannot
touch it**. 10.66 vs 10.65 is a genuine 0.09% deficit in KV traversal at ctx 8192. So the
expected outcome of the rung fix is 8192/C1 at **3/4**, not 4/4, and the last metric needs a
decode-side win of ~0.01 ms/step.

## Branch review: dead code and knob hygiene (2026-09-23T19, four agents)

### Applied
* **knob_gen.rs regenerated** (`KNOB_GEN_WRITE=1`). It was stale and failed
  `generated_evaluator_is_current`, i.e. it blocked the merge gate. I had earlier waved this off
  as benign — wrong. The regen adds two Emit knobs that were registered but never emitted
  (`emit.max_request_chunk`, `emit.stage_rows`) plus the `stage_rows_requires_max_request_chunk`
  load constraint. 20/20 knob tests pass.
* **op_moe.cuh comment drift.** Two MEASURED results were parked above `PLOW_MOE_XN_BF16`, which
  **no build path sets**. The lane-split number (bf16 7.060 -> 6.766 ms) belongs to
  `PLOW_MOE_DOWN_LANESPLIT`, which both `build_sm90a_gemma4_segments.sh:49` and
  `build_sm90a_cubin.sh:132` do set; the staging-fu negative belongs to `PLOW_MOE_DOWN_STAGE_FU`.
* **Docs**: 10 files synced to the code. Load-bearing correction — `17-unified-token-batch.md`
  claimed "CUDA has no token-batch executor yet". It has one (`exec/gpu/token_batch.rs`,
  `CudaTokenBatch::load`, gated on compute capability exactly `(9,0)`), verified at
  `token_batch.rs:61` plus a `token_batch` default-on test, so the default DOES enable token
  batching on H100. Plus five inverted bool defaults and stale module paths.

### A grep flaw worth keeping
`grep -w MACRO` **cannot** match `-DMACRO=1` — the `D` is a word character, so the left word
boundary never occurs. The error direction is toward a false "never built". Any build-flag audit
must use plain substring grep. Verified live: `grep -cw PLOW_NV_PACKED_REQUEST
runtime/CMakeLists.txt` = 0, plain grep = 2.

Related: a bulk "registered but unreachable" sweep is invalid. `manifest.rs:2587` synthesizes
capability defines via `op.c_name().replace("PLOW_DOP_", "PLOW_HAS_")`, so ~120 `PLOW_HAS_*`
names appear in no builder yet are emitted on every build. The literal-name test only holds
OUTSIDE the `PLOW_HAS_*` / `PLOW_DOP_*` family. Hand-check per macro.

### Deferred deliberately (not applied)
* **CUDA dead code** — `op_dsa.cuh:455 d_gather_attn_decode` and `:516 d_gather_merge` (two
  `__global__` kernels, no callers, self-documented "NOT WIRED"), and `interp_sm120_poc.cu` (a
  whole file no build produces). Deleting these changes the cubin and would force a packet
  rebuild mid-campaign. Their comments are accurate, which is what makes them tolerable.
* **`trim` guard, mux.rs:4333** — uses `pf_pack_budget` (a *plan* rung) where a *covering*
  bucket was meant, so the branch is dead. Fixing it changes scheduling at C>1 only
  (`decode_rows > 0`), which is unmeasured; it needs its own A/B, not a drive-by.
* **~2,900 lines of staged dead code** (`het.rs`/`head.rs`/`kv_handoff.rs`, `orch/*`,
  `memory/*`) carry "Allowed dead until the head pool drives it" notes. That is an owner
  decision, not mine to take unilaterally.
* **Four re-classifications rejected as KEEP**: `PLOW_NV_ABLATE_LO/HI` is a live instrument
  (`tune_decode_sweep.sh:423` passes it unconditionally), `PLOW_NV_SKELETON` is a documented
  cmake switch, `PLOW_NV_FA_FP8ABL` is *specified* never to ship, `PLOW_NV_GEMV_LS` is a
  recorded negative. The useful test is not "is it built?" but "does a comment record why it
  is off?".
* Only `PLOW_NV_PLACE_DISPATCH` is a clean removal: genuinely unbuildable (its required
  `PLOW_NV_L2_SMS` exists nowhere in the repo), self-admittedly unsound, and it taxes 11
  unrelated `#if` guards.

### Rung fix MEASURED and committed (7c217beb)

| cell | control | PF_CHUNK=4096 | rung fix | fix vs control |
|------|---------|---------------|----------|----------------|
| 128/C1   | 18.32  | 18.25  | 18.31  | -0.01 |
| 1024/C1  | 45.93  | 46.33  | 46.30  | +0.37 |
| 4096/C1  | 170.51 | 170.19 | 170.07 | -0.44 |
| 8192/C1  | 355.60 | 351.14 | 351.04 | **-4.56** |
| 15000/C1 | 703.25 | 699.81 | 691.65 | **-11.60** |

The fix is **3.4x better than the knob at 15000** (-11.60 vs -3.44) and equal at 8192: the
planner optimizes the whole request, where the knob forces one slice size everywhere. So
`PLOW_PF_CHUNK` is not needed for this and no new knob was added.

All three arms score 3 cells at 4/4. **The fix flips nothing**, as predicted: 8192/C1 wall gap
narrows +8.3 -> +3.7 ms but TTFT is still 351.04 vs 348.61 and TPOT 10.66 vs 10.65. My
pre-registered prediction of "3/4 at 8192" was WRONG — I expected TTFT to flip on a ~4 ms win
and it did not; 2.43 ms of TTFT remain. Only p99 ITL is won there.

**Where 8192/C1 actually stands**: needs TTFT -2.43 AND TPOT -0.01 AND tok/s +0.1. The first is
prefill (reachable: FlashPrefill #66 runs at 196/351 TFLOP/s vs the GEMM path's ~780); the
second is decode KV traversal and no prefill work can touch it.

## 26B C1 ladder, paired (2026-09-23T20) — user ask

Packet p26l8 (bf16, ctx16k, 16 slots, rungs to 8192), vs a vLLM 0.28 26B reference measured in
the SAME session (coherence gate PASS, matched client/warmups/seed, 32 prompts / 128 out).

| cell | TTFT p/v | TPOT p/v | p99 ITL p/v | tok/s p/v | win |
|------|----------|----------|-------------|-----------|-----|
| 128/C1   | **21.34**/38.26  | 5.60/5.03 | **5.67**/5.68 | 174.6/189.0 | 2/4 |
| 1024/C1  | **38.24**/41.88  | 5.73/5.08 | **5.78**/5.86 | 167.2/186.4 | 2/4 |
| 4096/C1  | 103.39/93.06 | 5.79/5.09 | **5.86**/5.97 | 152.6/173.0 | 1/4 |
| 8192/C1  | 210.43/180.09| 5.87/5.09 | **5.95**/6.02 | 133.9/154.8 | 1/4 |
| 15000/C1 | 431.70/351.55| 5.99/5.09 | **6.04**/6.07 | 107.3/128.3 | 1/4 |

**0 cells at 4/4.** The 26B is well behind the 12B (3/5). But one defect found and fixed:

**The realtime profile was stale at `PLOW_MULTISTEP = "4"`** while the sibling
`bf16-ctx16k.toml` already carried `"0"` with its own certificate. MULTISTEP=4 emits four tokens
per wave, so `itl_med` is literally 0.000 and p99 ITL is exactly 4 x TPOT. Fixing it
(commit 6374d9db) took p99 ITL 22.38/22.88/23.17/23.43/23.93 -> 5.67/5.78/5.86/5.95/6.04 and
**won p99 ITL at every C1 cell**, with TTFT level and TPOT +0.02 — inside the recorded
certificate. Still stale, unmeasured, left alone: `bf16-realtime.toml`, `bf16-l8192-r3072-16k.toml`.

**What remains on the 26B C1 column, in order of size:**
1. **TPOT 5.6-6.0 vs 5.03-5.09 (+11-18%)** — lost at every cell, and it drags tok/s with it
   (at C1 out_tok_s is just 128000/(TTFT + 127*TPOT)). This is the single blocker for 128 and
   1024, which already win TTFT and p99.
2. **TTFT above 4096** — 103/93, 210/180, 431/351. The bench's own roofline puts prefill at
   32-34% of the compute roof, i.e. #66 (FlashPrefill at 196/351 TFLOP/s vs the GEMM path's
   ~780) is the lever, not scheduling.

### #71 narrowed: the fixed per-step cost is NOT host turnaround either

`sched/multistep.rs` states the mechanism plainly: MULTISTEP enqueues *k* decode iterations'
worth of packet streams at once "so the host isn't in the loop every token" — it removes
**host->device turnaround**, not per-launch device cost.

Today's 26B A/B is therefore a free experiment on #71. MULTISTEP 4 -> 0 removes three of every
four host round trips, and TPOT moved **5.58 -> 5.60 ms**, i.e. nothing. (MULTISTEP was
genuinely active, not silently disabled by `multistep_disabled_by_decode` at gpu.rs:3539 — p99
ITL fell 4x when it was turned off.)

So host turnaround joins dispatch, smem zeroing and CUDA-graph launch on the excluded list from
17cacd99. What is left for a ~1 ms cost invariant to batch, context AND register count is a
per-kernel-launch device cost.

### The 26B's decode problem is bandwidth efficiency, not scheduling

The bench's own roofline, from this run: 3.8B active params / 7.64 GB weights, decode at
**1375 GB/s = 41.0% of the memory roof**, flat across the whole ladder. Its diagnosis line reads
"low bandwidth efficiency; check dispatch overhead or wave tail quantization".

For contrast the 12B runs its weight walk at 2.34 TB/s = ~70% of roofline (17cacd99). vLLM's
5.03 ms on the same 7.64 GB implies ~1.52 TB/s = 45% — also poor, but 11% better than plow.
At B=1 an A4B MoE reads its 7.64 GB scattered across experts, so both stacks lose locality; the
gap is the addressable part.

This is why the 26B TPOT is lost at all five C1 cells and why tok/s follows it down. It is
tasks #35/#36 territory (MoE decode), not a scheduling knob, and it is the single blocker
keeping 128/C1 and 1024/C1 at 2/4 rather than 4/4.

## 12B C4 column, paired (2026-09-23T21) — first time scored against a same-session baseline

| cell | TTFT p/v | TPOT p/v | p99 ITL p/v | tok/s p/v | win |
|------|----------|----------|-------------|-----------|-----|
| 128/C4   | **38.63**/55.60   | 10.79/10.60 | 14.00/11.63  | 362.6/365.2 | 1/4 |
| 1024/C4  | **101.56**/130.81 | 11.81/11.16 | 53.60/12.03  | 319.3/330.5 | 1/4 |
| 4096/C4  | **324.20**/466.49 | 14.01/12.17 | 174.19/12.14 | 243.1/254.3 | 1/4 |
| 8192/C4  | **686.37**/994.78 | 17.29/13.63 | 189.88/13.70 | 177.3/187.7 | 1/4 |
| 15000/C4 | **1230.22**/1669.69 | 26.46/18.18 | **211.91**/324.48 | 111.2/128.6 | 2/4 |

**0 cells at 4/4, but the shape is a deliberate trade, not a uniform loss.** plow wins TTFT at
every C4 cell by **30-45%** and loses TPOT everywhere plus p99 ITL at four of five.

**Cause, from the config not a guess**: the realtime profile ships `PLOW_PF_INTERLEAVE = "0"`,
and `config.rs:986` maps 0 -> `usize::MAX`, i.e. **unbounded prefill rows per launch** (the unset
default is 2048). A 4096-row prefill therefore runs to completion while every decoder waits,
which is exactly the TTFT lead and exactly the ITL penalty.

**Decomposition** (pure batching vs prefill interference), using 128/C4 as the
near-zero-interference control:

|                          | plow  | vLLM  |
|--------------------------|-------|-------|
| B=4 batching cost (128/C4 minus C1) | +0.27 | +0.04 |
| 4096/C4 total over C1               | +3.38 | +1.52 |
| => prefill interference             | ~3.11 | ~1.48 |

plow's interference is ~2.1x vLLM's; its pure batching penalty is ~7x.

**Pre-registered ceiling for the interleave sweep.** p99 ITL >= TPOT always. At 4096/C4 plow's
TPOT (14.01) already exceeds vLLM's whole p99 (12.14), so p99 there **cannot** be won without
winning TPOT first. Even perfectly matching vLLM's interference leaves TPOT ~12.27 vs 12.17.
Expect big p99 gains and cells moving 1/4 -> 2/4 or 3/4, **not** 4/4 — consistent with the
earlier recorded null that PF_INTERLEAVE is a TPOT/TTFT dial that does not flip a cell. Running
it anyway because p99 ITL of 174 ms is the worst single number in the comparison, and the goal
names serving metrics explicitly.

### PLOW_PF_INTERLEAVE sweep at C4 — the pre-registered null holds, default unchanged

| arm | 4096 p99 | 8192 p99 | 15000 p99 | 4096 TTFT | 8192 TTFT | cells 4/4 |
|-----|----------|----------|-----------|-----------|-----------|-----------|
| unbounded (shipped) | 174.19 | 189.88 | 211.91 | 324.20 | 686.37 | 0 |
| 2048 | 155.82 | 164.73 | 173.40 | 557.08 | 934.90 | 0 |
| **1024** | **56.41** | **60.21** | **67.02** | 335.22 | **612.87** | 0 |
| 512 | 49.56 | 55.31 | 64.52 | 494.07 | 1046.50 | 0 |

**No arm flips a cell anywhere** — exactly the ceiling registered before the run, and it confirms
the earlier null (#50) that PF_INTERLEAVE is a TPOT/TTFT dial.

`IL=1024` is nonetheless the interesting point: it cuts p99 ITL ~3x AND improves TTFT at the two
largest cells (686.37 -> 612.87 at 8192, 1230.22 -> 1132.66 at 15000), paying in TPOT
(17.29 -> 18.97 at 8192) and tok/s. `IL=512` over-chunks: every metric degrades.

**Not claimed:** `IL=2048` scored 3/4 at 128/C4 (vs 1/4 control). Rejected as noise — the tok/s
margin is 365.5 vs 365.2 (0.08%), and the knob should not even bind on a 4x128 = 512-row
workload, so a mechanism is missing. Needs a repeat run before it means anything. This session
already lost a day to a cross-session artifact; see [[greedy-equivalence-needs-a-noise-floor]].

**Decision: leave the shipped `PLOW_PF_INTERLEAVE = "0"` alone.** It does not win a cell, and
moving it trades three metrics for one. Revisit only if TPOT at C4 comes down enough that p99
stops being gated by it.

## 12B C16 serving column, paired (2026-09-23T22) — and the unified verdict

high_concurrency profile (MULTISTEP 0, DECODE_MAX_RUNG 16, TOKEN_BATCH 1), 64 prompts, p12rw.

| cell | TTFT p/v | TPOT p/v | p99 ITL p/v | tok/s p/v | win |
|------|----------|----------|-------------|-----------|-----|
| 128/C16   | **73.39**/100.06    | 11.74/11.04 | 12.41/12.23 | 1304.9/1361.8 | 1/4 |
| 1024/C16  | **312.74**/412.86   | 16.26/13.84 | 166.21/28.69 | 857.3/941.5  | 1/4 |
| 4096/C16  | **656.23**/1165.22  | 30.27/22.84 | **194.03**/329.20 | 451.7/502.3 | 2/4 |
| 8192/C16  | **1268.96**/2025.04 | 49.73/37.59 | **204.52**/344.69 | 267.9/300.2 | 2/4 |
| 15000/C16 | **2919.36**/3095.20 | 83.24/68.06 | **225.52**/377.06 | 150.3/173.8 | 2/4 |

### THE SCOREBOARD across all 20 cells paired in one session

| metric | cells won |
|--------|-----------|
| TTFT   | **15 / 20** |
| p99 ITL| **14 / 20** |
| TPOT   | 3 / 20 |
| out_tok_s | 3 / 20 |
| **all four (4/4)** | **3 / 20** — 12B 128/C1, 1024/C1, 4096/C1 |

**One conclusion, and it is the same in every column: TTFT is won comprehensively and decode is
the entire remaining problem.** plow leads TTFT by 30-44% at C4 and C16 (656 vs 1165 ms at
4096/C16) while losing TPOT at 17 of 20 cells. And because at every concurrency
`out_tok_s ~ N*outlen*1000/(TTFT + (outlen-1)*TPOT)`, the 127x weight on TPOT means tok/s simply
follows TPOT down — tok/s is not an independent target, it is a TPOT readout.

So the campaign does not need more scheduling knobs or prefill work. It needs decode:
* 12B C1 8192/15000 — TPOT slope vs context ~13x vLLM's, crossing at ~6-7k.
* 12B C4/C16 — batched decode. plow's pure B=4 batching penalty is ~7x vLLM's (+0.27 vs +0.04).
* 26B everywhere — decode at 41% of the memory roof vs the 12B's ~70% (vLLM ~45%).

The prefill levers are now exhausted at the config level: PF_INTERLEAVE is a confirmed null,
the rung planner is fixed, and MULTISTEP staleness is fixed. What remains on prefill is kernel
work (#66, #67) which buys TTFT we already win.

## 8192/C1 FLIPPED TO 4/4 (2026-09-23T23) — first new cell of the campaign

Packet `p12rq8` from the new `gemma4-12b.h100.bf16-c1-rq8192.toml`. Two changes, each aimed at
one of the two blocked metrics:

* `PLOW_MAX_REQUEST_CHUNK` 4224 -> **8192** — an 8192-row prompt runs ONE `[8192]` launch
  instead of `[4096, 4096]`, saving a ~5.2 ms fixed launch cost. (TTFT lever.)
* `PLOW_DECODE_BATCH_LADDER` `1,2,4,8,16` -> **`1,2,4`** — the megakernel entry is the whole
  ~1 ms fixed per-step cost and it taxes every compiled rung, so fewer rungs = smaller entry.
  (TPOT lever.)

| 8192/C1 | plow | vLLM | |
|---------|------|------|---|
| TTFT    | **347.26** | 348.61 | won |
| TPOT    | **10.58**  | 10.65  | won |
| p99 ITL | **10.83**  | 11.58  | won |
| tok/s   | **75.7**   | 75.2   | won |

**The TPOT lever is general, not specific to 8192** — it moved every C1 cell:

    ctx      128    1024   4096   8192   15000
    5 rungs  10.52  10.59  10.63  10.66  10.72
    3 rungs  10.45  10.51  10.55  10.58  10.65

That is the fingerprint of a fixed per-step cost, and it is the first thing to actually MOVE
#71 rather than exclude a cause.

**C1 column: 3 cells at 4/4 -> 4.**

Margin discipline: TTFT's margin is 0.4%, so it rests on two runs of this packet (346.57 with
MULTISTEP=4, 347.26 with 0; MULTISTEP does not touch TTFT), and TPOT was bit-identical at 10.58
in both.

**A third stale recipe found.** `gemma4-12b.h100.bf16-l8192-16k.toml` carried the same
`PLOW_MULTISTEP="4"` as the 26B one, and the new recipe inherited it. Measured at exactly
4.02 x TPOT with `itl_med` = 0.000 (p99 41.87/42.15/42.35/42.56/42.87 vs TPOT 10.45..10.65).
Both fixed. **Always diff a derived recipe's `serve_env` against its sibling.**

**15000/C1 is out of reach by rung shape and stays 1/4** (TTFT 694.51 vs 675.86). 15001 rows
need 16384 padded rows under ANY split, because the ladder above 4224 admits only
pow2(+64/+128) shapes; ~1383 padded rows is ~55 ms and inherent. It needs #66.

## 128/C4 FLIPPED TO 4/4 — the entry lever is not C1-specific

The SAME `p12rq8` packet (4 slots, decode ladder 1,2,4) run at C4, vs the same-session vLLM C4
reference. It needs no C4-specific build: 4 slots IS C4's size.

| 128/C4 | control | rq8192 | vLLM | |
|--------|---------|--------|------|---|
| TTFT   | 38.63 | **35.83** | 55.60 | won |
| TPOT   | 10.79 | **10.50** | 10.60 | won |
| p99    | 14.00 | **10.96** | 11.63 | won |
| tok/s  | 362.6 | **373.5** | 365.2 | won |

TPOT fell **0.29 ms**, more than the 0.19 the cell needed. The lever is bigger at C4 than the
0.07 it gave at C1 — consistent with a fixed per-step cost being amortised over fewer tokens
per unit time as batch grows.

**The large-input C4 cells got WORSE, as predicted**: request_chunk 8192 makes a prefill launch
bigger, and C4's p99 is interference-bound — 8192/C4 TTFT 686 -> 927, p99 190 -> 343.

**`PLOW_PF_CHUNK=4224` does NOT cleanly undo that.** It restored the prefill behaviour at
8192/15000 but COST 128/C4 its p99 win (10.96 -> 14.89) and the cell dropped to 3/4. Cause: the
knob also feeds a per-tick row budget (mux.rs:2002), not just the per-request slice cap, so it
is not the single-variable dial I treated it as. Recorded as a second reason not to reach for
this knob.

**Best known C4 config = rq8192 uncapped: 1 cell at 4/4 (was 0).** One packet now serves both
the C1 and C4 columns.

| config | 4/4 cells at C4 | 15000/C4 |
|--------|-----------------|----------|
| control ladder16k | 0 | 2/4 |
| rq8192 uncapped   | **1** | 1/4 |
| rq8192 + PF_CHUNK=4224 | 0 | 2/4 |

## C16 lean ladder (1,4,16) — MEASURED NULL, and it costs TTFT. Recipe not shipped.

Single-variable vs the control: same max_chunk 4224, same 16 slots, same high_concurrency
profile, same 64 prompts. Only the compiled decode rung count differs.

| 128/C16 | control (1,2,4,8,16) | lean (1,4,16) | vLLM |
|---------|----------------------|---------------|------|
| TTFT    | **73.39** won | 108.73 lost | 100.06 |
| TPOT    | 11.74 | 11.31 | 11.04 |
| p99 ITL | 12.41 lost | **11.65** won | 12.23 |
| tok/s   | 1304.9 | 1324.1 | 1361.8 |

**The entry lever works but a coarse ladder costs more than it saves at C16.** TPOT did improve
(-0.43 ms) and p99 flipped to a win, but TTFT regressed 48%. Cause: with rungs 1,4,16 a batch of
2-3 rounds up to 4 and 5-15 to 16, so during the C16 ramp every decode step does extra work and
delays prefill. 4096/8192/15000 are unchanged at 2/4.

**128/C16 is unreachable with this lever even combining best-of-both**: TTFT 73.39 from the
control plus TPOT 11.31 from lean still loses TPOT (needs 11.04, i.e. -0.70; the lever gave
-0.43). C16 keeps ladder16k unchanged.

**So the entry lever has a shape constraint**: it pays when the dropped rungs are ones the
column does not use (C1 never needs 8 or 16; C4 never needs 8 or 16) and costs when they are
(C16 uses 2 and 8 constantly during ramp/tail).

## The 15104 rung WEDGES at run time — #45's blocker confirmed, and my checker was wrong

Attempt: a 15104 rung (= 118*128) so a 15000-token prompt (15001 rows with BOS) runs in ONE
launch instead of padding to 16384 across two. The arithmetic was sound — ~1280 fewer rows
(~51 ms at 0.0398 ms/row) plus one saved ~5.2 ms launch, against an 18.7 ms TTFT deficit — and
the ring was UNCHANGED at next_pow2(1024+15104-1) = 16384, so it cost no extra sliding KV.

Two build-time facts learned:
* `plowc` enforces `PLOW_MAX_REQUEST_CHUNK <= PLOW_MAX_CHUNK` (lib.rs:3078), and MAX_CHUNK must
  stay a power of two — so request_chunk 15104 forces MAX_CHUNK 8192 -> 16384. That grows chunk
  activations but NOT the KV ring.
* `campaign.py build` refuses a non-empty out dir ("reproducible only into a fresh dir"), so a
  failed attempt must be cleared before retrying.

**Result: the packet built, and then WEDGED on the 15000 cell.** 128/1024/4096/8192 completed
normally (and held their 4/4 — TTFT 18.15/45.96/170.03/347.44, TPOT 10.44/10.50/10.54/10.57),
then the 15000 cell produced no row for ~60 min while holding the GPU lease; a `gpulease` probe
queued behind it. That is exactly the #51 / #45 signature.

**My pre-bench check was the WRONG diagnostic.** I verified every rung had a complete
`dispatch_table` role set (all 13 rungs, 2/2 roles, including 15104) and concluded "SAFE to
bench". It wedged anyway. So `dispatch_table` completeness does NOT certify a rung shape —
whatever the Gemm segmentation defect is, it is not visible there. A real pre-flight check for
this still does not exist, which is the actual content of #45's "BLOCKED".

**Recipe deleted, not shipped.** 15000/C1 stays 1/4. Cost of the lesson: one wedged lease,
killed with TaskStop; GPU confirmed free afterwards (0% util, 0 MiB, no plowrt process).

Note the cell was never winnable in one shot anyway: TPOT at 15000 is 10.65 vs vLLM 10.65, an
exact tie, and the scorer requires strictly less.

## 26B C1: the lean ladder lands, and where the remaining TPOT gap actually is (2026-09-24)

### The lean decode ladder transfers from the 12B, and flips nothing

`PLOW_DECODE_BATCH_LADDER = "1,2,4"` (was `1,2,4,8,16`) on the 26B, paired against a FRESH
same-session vLLM 26B reference (the prior one was ~6 h old):

| cell | TPOT lean / 5-rung / vLLM | tok/s lean / 5-rung / vLLM | TTFT lean / vLLM | win |
|---|---|---|---|---|
| 128/C1 | 5.40 / 5.60 / **5.03** | 181.0 / 174.6 / **189.2** | **21.09** / 37.96 | 2/4 |
| 1024/C1 | 5.49 / 5.73 / **5.07** | 174.2 / 167.2 / **186.5** | **37.86** / 41.86 | 2/4 |
| 4096/C1 | 5.52 / 5.79 / **5.08** | 159.2 / 152.6 / **173.2** | 102.51 / **93.37** | 1/4 |
| 8192/C1 | 5.57 / 5.87 / **5.08** | 139.7 / 133.9 / **155.1** | 208.92 / **179.62** | 1/4 |
| 15000/C1 | 5.64 / 5.99 / **5.07** | 112.2 / 107.3 / **128.5** | 423.77 / **352.21** | 1/4 |

Strict improvement over the 5-rung packet at every cell (TPOT -0.20..-0.35, tok/s +5..+7, TTFT
and p99 also better), so it shipped (7bd45936). **0 cells at 4/4 before and after.** Committed as
an improvement, not a flip.

Two corrections to what this campaign had been assuming:

1. **26B C1 TTFT is NOT won across the board.** vLLM is faster at 4096/8192/15000
   (93.37 vs 102.51, 179.62 vs 208.92, 352.21 vs 423.77). The 26B long-context prefill is a
   second open deficit, separate from decode. p99 ITL is won at all five.
2. **This is not a bandwidth-starved regime.** At 7.64 GB read per step, plow runs 1415 GB/s and
   vLLM 1519 GB/s -- only 7% apart, and vLLM is itself at just 45% of the 3352 GB/s roof. The
   roofline's "43% of memory roof, low bandwidth efficiency" reads like a bandwidth problem and
   is really a latency/occupancy one that BOTH stacks have. Chasing the roof is the wrong frame;
   the 7% is the whole prize, and it is worth 0.37-0.57 ms of TPOT.

Also: the lean packet moved the roofline 41.0% -> 43.2%, which confirms megakernel entry overhead
was inflating the denominator rather than bandwidth being the only term.

### The one measured asymmetry inside the MoE step: expert-DOWN loads in flight

Read of the decode MoE arms as this packet actually compiles them (`plow_config.h` checked, not
assumed -- `PLOW_MOE_DOWN_SG 8u` is present, `GV_UNROLL_GLU` is NOT, so the GLU arm runs the
op_gemm.cuh source default of 4):

| arm | share of MoE weight traffic | loads in flight per lane |
|---|---|---|
| GLU (gate+up) | 2/3 | `GV_UNROLL_GLU=4` x 2 streams = **8** |
| expert DOWN | 1/3 | **2** ("2 chunks pre-issued", op_moe.cuh lane-split) |

The repo's own `runtime/nvidia/experiments/hbm_ceiling_h100.cu` measures a pure read at 1 block/SM
going **1222 -> 2490 GB/s as in-flight loads go 1 -> 8**, and the kernel comment above the
lane-split arm already recorded DOWN as the worst GEMV in the step (1163 GB/s against 3269
achievable). So 8 is not an arbitrary depth: it is what the other arm already has and what the
board's own curve asks for.

`PLOW_MOE_DOWN_PRE` (kernel, default 2) + `tuning.moe_down_pre` (manifest -> `plow_config.h`) +
`emit.PLOW_TUNE_MOE_DOWN_PRE`. With `PLOW_MOE_DOWN_SG=8`: `LCH=32`, `nch = 704/32 = 22` chunks, so
depth 8 covers 22 as 8+8+6. `acc` still walks `c` ascending at every depth, so the FMA order and
the output are **unchanged** -- depth 2 is bit-identical to the pairing it replaces.

**Plumbing trap avoided.** The define had to go through the MANIFEST, not
`PLOW_BUILD_SEG_EXTRA_DEFINES`: the recipe's own `[objects.env]` comment records that the decode
object is built from the manifest and that the segments env does not reach it. That is the same
shape as the recorded `PLOW_EXTRA_DEFINES is not plumbed to packets` finding, where a
`#if`-guarded probe silently measures the same object twice and the too-clean null looks like a
result. Guarded with two md5 gates on the built objects: `p26lean2` (no env) must be
byte-identical to the pre-edit `p26lean`, and `p26dp8` must differ from `p26lean2`. If either
gate fails the measurement is void.

Pre-registered target at 128/C1: TPOT **<= 5.163** takes tok/s, **<= 5.03** takes TPOT outright.
A null means loads-in-flight in the DOWN arm is not the binding term -- it does NOT mean try 16.

### RESULT: the expert-down pre-issue lever is CLOSED, and the rewrite to reach it was a regression

Measured 26B C1, H100, one session, paired against the fresh vLLM reference taken the same session.

| cell | TPOT lean2 (DP2) | TPOT dp8 | delta | TPOT shipped lean | vLLM |
|---|---|---|---|---|---|
| 128/C1 | 5.770 | 5.760 | -0.010 | **5.400** | 5.030 |
| 1024/C1 | 5.860 | 5.830 | -0.030 | **5.490** | 5.070 |
| 4096/C1 | 5.890 | 5.870 | -0.020 | **5.520** | 5.080 |
| 8192/C1 | 5.940 | 5.910 | -0.030 | **5.570** | 5.080 |
| 15000/C1 | 6.020 | 5.990 | -0.030 | **5.640** | 5.070 |

Two separate findings, and both are negative:

1. **Depth 8 buys 0.01-0.03 ms** -- about 0.5%, against the 0.237 ms needed for tok/s at 128/C1.
2. **The rewrite that made depth a knob cost +0.37 ms TPOT at every cell, at DP=2**, where it is
   arithmetically identical to the loop it replaced (same chunks ascending, same `acc` chain).
   TTFT moved <0.2 ms at every cell so it is not drift, and 18 of 19 objects were md5-identical so
   it is not the emit. That +0.37 is the whole remaining gap to vLLM and would have erased
   7bd45936's gain by itself. Reverted in 3afffd18.

**Why, from the SASS** (`nvdisasm -c`, addresses stripped before diffing): the array form added no
loads. `LD.E.128` 2156 -> 2162 (+6); the loads-in-flight run histogram did not move at all. The
+936 instructions were address arithmetic -- **LOP3.LUT +368, S2R +326**, MOV -186. Both objects
sit at **REG:255 LOCAL:0** -- the cap, with no spill. At the cap the compiler will not keep 8
`bf16v8` (32 registers) live; it rematerialises `kk[i]` and the lane ids every iteration. So
**written pre-issue depth never becomes loads in flight**, and the premise was wrong: the DOWN arm
is not starved because its loop says 2, it is starved because the megakernel is at its register
ceiling. Raising loads in flight there requires moving the arm OUT of the 255-register megakernel.
Same mechanism that already closed the occupancy route.

The measurement now lives in the kernel comment above the loop so it is not re-derived.

### A single-rung decode ladder cannot coexist with the grouped-Lt MoE decode emit

`PLOW_DECODE_BATCH_LADDER=1` aborts the emit: `devgen/src/lib.rs:9568` panics **"no grouped MoE
decode GLU/DOWN pair to isolate"**, because `PLOW_EMIT_MOE_DEC_LT=1` +
`PLOW_GEMMA_MOE_DEC_GROUP=4` need a rung >= 4 to isolate. So the "compile only what C1 runs" arm
has to drop the grouped route as well -- three knobs, one hypothesis (total compiled entry size),
and it must be reported as one combined arm rather than attributed to any single knob. Queued as
p26c1min; the two single-variable r1 arms in this round produced NO cells for this reason.

### Agentic turn-by-turn FP8, first attempt: one real dataset, two harness defects

Workload: 16 conversations x 8 turns, concurrency 4, 64 out tok/turn, 300-token observations, so a
conversation's prompt grows 1.7 -> 20.1 kchar inside one logical session. `/v1/completions` with
raw prompt text on both stacks, because plowrt refuses `tools` with a 400 by design
(`serve/chat.rs:119`) and vLLM's BFCL dataset therefore cannot be apples to apples.

**26B FP8 (p26fp8) ran clean, 128/128 turns:**

| turn | prompt kchar | TTFT ms | p99 TTFT | TPOT ms |
|---|---|---|---|---|
| 1 | 1.7 | 47.57 | 242.42 | 35.06 |
| 2 | 4.3 | 113.04 | 225.98 | 35.50 |
| 4 | 9.6 | 128.81 | 240.77 | 36.42 |
| 6 | 14.8 | 101.68 | 135.75 | 37.12 |
| 8 | 20.1 | 107.19 | 222.34 | 36.20 |

TTFT stays ~100-130 ms while the prompt grows 12x, which is the prefix cache doing its job (the
serve log confirms `prefix_cache: true`). TPOT ~36 ms against the same model's 5.4 ms in bf16 is
**~6.7x worse, not the ~5x previously recorded** -- grouped MoE decode is bf16-gated, so FP8 on the
26B remains a capacity story, not a latency one.

**12B FP8 (p12fp8) faulted on every request**, including the 8-token warmup:
`device fault: cuStreamSynchronize: CUDA_ERROR_ILLEGAL_ADDRESS`. The client only reported "no
tokens streamed" 128 times, so the fault existed only in the serve log -- the harness now smoke-
tests one completion first and prints the serve-log fault. The same client drove the 26B cleanly,
so this is the 12B FP8 path, not the workload. Both FP8 packets are from 09-23 14:2x and plowrt is
from 19:04, i.e. newer than both, so staleness alone does not explain why only the 12B faults; a
rebuild from the current tree discriminates packet/binary skew from a live defect, and
`pf_plan_slice` (added this session) is the first suspect if the rebuild still faults.

**Both vLLM arms died instantly**: `vllm: error: unrecognized arguments: --disable-log-requests`.
vLLM 0.28 removed the flag. My bug, fixed.

---

## 26B C1 column: the decode deficit is 100% inside the megakernel (2026-09-24)

Four measurements, all same-session paired, all on packet `p26lean` unless noted. Reference is the
same-session vLLM 0.28 C1 ladder (`vllm_26b2.csv`).

### 1. A runtime bug blocked every single-rung ladder (FIXED, commit 3fd3416a)

`packed_terminal.rs` validated sampled token ids with `self.ids(self.host_rows.len())`, but
`host_rows` is `[Vec<u32>; 2]` -- the double-buffered staging pair -- so `.len()` is the **array
arity 2**, never the staged row count. `output_bytes` three lines above derives it correctly as
`host_rows[i].len()`.

* `capacity >= 2`: reads one u32 that was never copied back (a stale word in the pinned slab).
  Latent spurious "compact terminal produced an invalid token".
* `capacity == 1`: `capacity = e.batch` = max decode rung, so `host_ids` is 4 bytes and the slice
  panics -- *range end index 8 out of range for slice of length 4* -- aborting the server on
  request 1.

It surfaced only as the campaign's generic "coherence gate did not pass", because the panic lands
in the per-run `server.log`. Fixed; the ladder-"1" packet now serves a full C1 ladder.

### 2. It is the WIDEST compiled rung that taxes the entry, not the rung count

`PLOW_DECODE_BATCH_LADDER` `1,2,4` -> `1,4` (one fewer rung) is a **clean null**:

    decode object  1,812,616 -> 1,820,552 B   (no shrink, marginally larger)
    STACK 544 -> 544,   SHARED 8848 -> 8848   (identical)
    TPOT  5.400/5.480/5.520/5.570/5.640       (identical to 3 dp, both arms)

The earlier 5->3 cut paid off because it dropped rungs **8 and 16**. Rung 2 is an interior narrow
rung; removing it changes no allocation. Do not count rungs when estimating this lever.

### 3. The entry-size route is CLOSED: a 28% smaller object buys 0.05 ms

Ladder `1` (+ `PLOW_EMIT_MOE_DEC_LT=0`, `PLOW_GEMMA_MOE_DEC_GROUP=0`, since `lib.rs:9568` needs a
rung >= 4 to isolate the grouped GLU/DOWN pair), serve `PLOW_MOE_DEC_LT=0 PLOW_DECODE_MAX_RUNG=1`:

    decode object 1,812,616 -> 1,296,776 B (-28%), STACK 544 -> 184, SHARED 8848 -> 3216

| cell | 1-rung | lean (1,2,4) | delta | vLLM | win |
|---|---|---|---|---|---|
| 128/C1 | 5.360 | 5.400 | -0.040 | 5.030 | 2/4 |
| 1024/C1 | 5.430 | 5.480 | -0.050 | 5.070 | 2/4 |
| 4096/C1 | 5.470 | 5.520 | -0.050 | 5.080 | 1/4 |
| 8192/C1 | 5.520 | 5.560 | -0.040 | 5.080 | 1/4 |
| 15000/C1 | 5.590 | 5.650 | -0.060 | 5.070 | 1/4 |

0.05 ms against the **0.33** needed at 128/C1. Closed. (3-variable arm, but immaterial at this
magnitude.)

### 3b. CORRECTION: the ladder-width lever is sign-unstable, and that closes it for good

Re-measured 2026-09-24 on this branch's tree, in `step_bench` (slots=1 ctx=128 n=64, 3 interleaved
order-reversed passes, one lease), with the one-rung packet now legal (#77):

         ladder   batch   megakernel     mean_ms      sd   vs ctl
    1,2,4 (ctl)       4   1,822,856 B     5.3687  0.0015       --
            1,2       2   1,816,456 B     5.3920  0.0020  +0.0233
              1       1   1,477,128 B     5.4923  0.0021  +0.1237

The noise floor for this harness is **0.0013 ms** -- in the previous run all three arms were
accidentally the *same* packet and spanned 5.3680-5.3693 -- so +0.1237 is ~95 sigma. It is a real,
clean regression: the object shrank 19% and the step got slower.

**It is also the opposite sign to section 3**, which measured the same nominal change (`1,2,4` ->
`1`) on the served ladder and got -0.040 to -0.060 ms at every C1 cell. The two are not the same
packet: section 3's arm also carried `PLOW_GEMMA_MOE_DEC_GROUP=0` and came out 1,296,776 B, against
1,477,128 B here. Different build, different harness, opposite sign, both under 0.13 ms.

Gates that make this a measurement rather than another silent no-op:

* the rung lists genuinely differ (`programs` printed per packet, with a HARD ABORT if they match --
  which is what caught the run before this one, where the recipe's own `[env]` silently beat the
  ambient env and all three "arms" were one packet);
* the **B=1 program is identical across arms** -- 551 ops, 551 counters, 640 edges, 31 deadctr,
  29303 ents, 37252 polls, 29272 bumps_live -- so the `PLOW_EMIT_MOE_DEC_LT=0` that the narrow arms
  require (`lib.rs:9616` asserts without a rung >= 4) is *confirmed* inert at B=1, not assumed inert;
* every arm loaded without `retains widest execution`, so no arm silently ran widest-only.

**Section 2's title is wrong as a general claim.** Dropping the *widest* rung (4) did not cut the
entry cost; it cost 0.124 ms. The occupancy explanation is refuted too: both arms log
`grid=132 smem=164864 occ_per_sm=1`, because the 164864 B dynamic arena sets occupancy, so the
static SHARED difference (8848 -> 5264) cannot move it.

What actually co-varies with ladder width is the packet's **max decode batch**, and with it the KV
allocation: ctl `batch=4 kv_gib=7.5`, `1,2` -> `batch=2`, `1` -> `batch=1 kv_gib=1.875`. Ladder width
and KV geometry are perfectly confounded in *both* directions, so neither this experiment nor
section 3 ever isolated the entry.

**Verdict: closed, and not worth another lease.** The lever's entire measured dynamic range across
ladder `1` -> `1,2,4` is 0.124 ms, its sign does not survive a change of build config or harness, and
128/C1 needs 0.339. A widening arm (`1,2,4,8`) was written and then abandoned before it took a lease
for exactly this reason -- best case ~-0.02 ms, and it would cost 15 GiB of KV against the C32
capacity story. Do not re-open this on object size or rung count.

Follow-up (not done here, to avoid perturbing the recipe digest and invalidating cached packets):
`gemma4-26b-a4b.h100.bf16-c1-lean.toml` still carries the comment "the megakernel ENTRY is
essentially the whole fixed per-step cost (#71) and it taxes EVERY compiled rung". That is the claim
refuted above and it will mislead the next campaign.

### 3c. The first SAME-SESSION paired 26B C1 ladder: 7/20, and the TPOT target is REAL

The 26B had never had a same-session paired vLLM reference (only the 12B did). Run 2026-09-24,
plow and vLLM 0.28 back to back, each under its own lease, matched client / backend / dataset /
NPROMPT 32 / OUTLEN 128 / `--num-warmups 2 --seed 42` / gate prompt, prefix caching OFF on both
sides, vLLM `--max-num-seqs 1`. Packet p26kvctl2 (this recipe, knobs `verified`), realtime profile.

    cell        TTFT plow/vLLM     TPOT           p99 ITL        tok/s         wins
    128/C1       20.97/ 37.76      5.390/5.030    5.45/5.82      181.4/189.2   2/4
    1024/C1      37.96/ 41.52      5.480/5.080    5.54/5.82      174.4/186.4   2/4
    4096/C1     102.72/ 92.28      5.520/5.090    5.58/5.94      159.3/173.1   1/4
    8192/C1     209.21/178.44      5.560/5.090    5.63/5.97      139.7/155.0   1/4
    15000/C1    424.56/350.58      5.640/5.090    5.70/6.03      112.2/128.3   1/4

**7 of 20 metric-cells.** p99 ITL is a clean 5/5 sweep. TPOT loses 5/5. tok/s loses 5/5 (it is a
TPOT readout). Peak memory: plow 59.3 GiB vs vLLM 74.6 -- 15 GiB less, not one of the four metrics.

**The stored cross-session reference is VALIDATED for the 26B.** Same-session vLLM minus the
stored `reference-vllm028-bf16.csv`, per cell: TPOT -0.010 / 0.000 / -0.010 / -0.010 / 0.000;
TTFT -0.56 / -0.36 / -0.24 / -0.08 / +0.20; p99 -0.02 / +0.03 / -0.01 / +0.05 / +0.05. Everything
is inside 0.01 ms on TPOT. So unlike the 12B -- where vLLM drifted ~0.1 ms between sessions and
faked a "0 cells at 4/4" (99d939ad) -- the 26B's stored baseline was never stale, and every past
26B scoreboard scored against it was decidable after all.

**This refutes the hypothesis that drove the run.** This recipe carried a note citing vLLM 0.28
"measured the same session" at 5.68 / 5.86 / 5.97 / 6.02 / 6.07, which would have meant plow's
5.58 already won 128/C1 and that the whole -0.34 ms hunt was chasing a stale number. It does not
reproduce: vLLM is 5.03-5.09 across C1. **The 26B C1 TPOT deficit is real** -- -0.36 ms at 128
widening to -0.55 at 15000 -- and the levers refuted above (#79 #80 #81 #82) were correctly
aimed, they simply did not work. The misleading note has been corrected in the recipe.

**A target this run exposes: TTFT above 1024.** plow wins TTFT at 128 (20.97 vs 37.76, 44% better)
and 1024 (37.96 vs 41.52), then LOSES it from 4096 up -- by 11% at 4096, 17% at 8192, 21% at
15000. The crossover sits between 1024 and 4096, i.e. it is prefill scaling, not a fixed cost.
That is 3 cells of 1 metric each against a KNOWN inefficiency (#66: FlashPrefill at 196/351
TFLOP/s while the GEMM path runs 77-82% of peak), where the TPOT column needs a route that is
currently closed. Cheaper cells than TPOT, and they are regressions from a column plow used to
lead.

Data: `perf-data/campaign/gemma4-26b-a4b.h100.reference-vllm028-bf16.paired-2026-09-24T16.csv`
and `gemma4-26b-a4b.h100.bf16-c1-lean.paired-2026-09-24T16.csv`.

### 3d. Why the 26B loses TTFT above 1024: FlashPrefill is the growing term

Fitting the paired C1 TTFT (3c) against input length separates a fixed cost from a per-token one:

    range          plow ms/tok   vLLM ms/tok   ratio
    1024 -> 4096      0.02108       0.01652     1.28x
    4096 -> 15000     0.02952       0.02369     1.25x
    intercept          9.6 ms        18.9 ms    plow 2x BETTER

plow's prefill is a uniform **~26% slower per token** and only wins 128/1024 on its much lower
fixed cost. So this is ONE defect, not three cells: 26% off per-token prefill flips 4096, 8192
and 15000 C1 TTFT together.

Attribution with `PLOW_PF_SEG_TIME=1` + `scripts/campaign/segtime_table.py`, 26B, 4 prompts at
4096 + 15000, C1 (SHARES, never latency -- SEG_TIME drains per segment, ~3.7x inflation). The
chunks below are one 15000-token request's successive chunks, so the change ACROSS them isolates
the context-dependent term:

    opcode                        ch33    ch34    ch35    ch36      behaviour
    MoeGroupDownPf+GroupGluPf    22.07   39.68   40.75   40.40   flat  (34-41%)
    FlashPrefill                  8.50   18.39   25.94   32.58   GROWS (16 -> 28%)
    Gemm                          9.65   17.94   18.47   18.56   flat  (16-18%)
    MoeAlign+MoeRouter+RmsNorm    5.20    8.46    8.58    8.60   flat
    MoeCombineNorm+NormResid      4.14    7.57    7.74    7.68   flat
    HeadNormRope                  2.95    5.70    5.88    5.81   flat
    NormResidual+RmsNorm          1.06    1.94    2.00    2.00   flat
    Glu                           0.79    1.17    1.23    1.24   flat

The MoE expert GEMMs are the single largest share, but they are FLAT -- per-token work, already
on the grouped cuBLASLt route. **FlashPrefill is the only term that grows**, 8.5 -> 32.6 ms inside
one request. That is the fingerprint the deficit has: plow trails vLLM by 10.4 ms at 4096, 30.8 at
8192 and 74.0 at 15000, i.e. ~N^1.5 -- faster than the token count, so a growing term dominates it.

The flat norm/rope/router/combine tail is ~22% of the chunk here (the earlier estimate was 14%;
SEG_TIME inflates small ops' shares because each pays a drain, so treat 22% as an upper bound).

**Neither lever is a knob, and neither closes 26% alone.**
* #66 FlashPrefill, 196 (sliding) / 351 (full) TFLOP/s against the GEMM path's ~780. Already
  established above that sliding attention is not doing wasted work -- the window IS being
  exploited -- so the inefficiency is real and, in this file's own words, "no cheap 4x win
  there; it is a kernel project". It is the growing term, so it is the one that decides these
  three cells.
* #67 norm/Glu/rope epilogue fusion, priced at ~2.0 s of the C32 wall and ~22% of a C1 chunk
  here. Flat, so it shifts the line down rather than changing its slope -- it helps every cell
  a little and does not by itself fix the crossover.

**Column verdict for the 26B C1 after 3c/3d:** p99 ITL 5/5 won. TTFT 2/5, needing a prefill
kernel project (#66, slope) plus epilogue fusion (#67, offset). TPOT 0/5 and tok/s 0/5 (tok/s is
a TPOT readout), needing the decode memory-parallelism route that #82 found gated off for MoE.
Three kernel projects, no remaining knobs -- that is the honest state of this column.

### 3e. REFUTED: path DEPTH is not the decode cost either — the spine fusion costs +0.29 ms

Section 3 of the decode-depth work made three predictions from the 3.01 us/level figure
(`GRAPHSTAT_CP=1 graphstat <assets> 1`, 366 of 551 B=1 ops on the critical path). Two were already
refuted (fusing 25 *parallel* ops: +0.0127, commit c7ddf1f0; widening the single-CTA combine:
+0.13, commit 10c08ddd). The third and strongest was **spine-consecutive** fusion: fuse ops that
are genuinely adjacent on the critical path and the depth must fall, so the time must fall with it.

It does fall, exactly as predicted, and the time goes UP.

`PLOW_GEMMA_MOE_TAIL_FUSE=1` folds the MoE combine (op70) and the following norm/residual into
one op72 per layer, 30 layers. Both CPU gates landed on the nose:

| gate | OFF (tf0) | ON (tf1) | predicted |
|---|---|---|---|
| B=1 decode ops | 551 | 521 | 521 |
| critical-path levels | 366 | 336 | 336 |
| megakernel bytes | 1 822 856 | 1 821 960 | — |

Measured, step_bench slots=1 ctx=128 n=64, three passes with the arm order reversed on pass 2,
one lease, `PLOW_MULTISTEP=0`:

| arm | mean_ms | sd |
|---|---|---|
| tail fuse OFF (ctl) | 5.3723 | 0.0032 |
| tail fuse ON | 5.6580 | 0.0000 |

**+0.2857 ms, a regression.** The noise floor for this harness is 0.0013 ms and the treatment sd is
0.0000 across three separated runs, so this is roughly 90 sigma. The prediction was -0.0900.
The sign is wrong, and the magnitude is 3x the predicted win in the other direction.

**What this closes.** Op *count*, op *width* and now path *depth* have each been measured and each
is a non-lever on this decode step. The 3.01 us/level figure describes a correlation across the
existing program, not a cost you can buy back by removing levels: removing 30 real levels bought
-0.0 and cost +0.29. Stop deriving decode work from the DAG shape. The remaining B=1 budget is a
bandwidth story (7.64 GB/step, 1425 GB/s = 45% of roof) — see the byte split in
`gemma4-26b-a4b-h100-campaign` memory, where attention q/k/v/o is 27%, lm_head 19%, experts 54%.

The knob stays `OFF`/`OPT_IN` (`devgen/src/knob_spec.rs:1002`), which is where it already was; what
changed is that it now *works* at t>1 and its ladder is accepted, so the number above is real
rather than a silently-skipped arm.

**A measurement defect worth more than the result.** The first two attempts at this A/B both
reported the treatment as unqualified and measured nothing. The rebuild line was

    nix develop --command cargo build --release -p plowrt --examples     # no --features

which exits rc=0, prints `Finished release profile`, and **does not relink**
`target/release/examples/step_bench` — the featureless `plowrt` is a different cargo unit, so the
GPU ran a binary built 3.5 h earlier, before the change existed. `docs/bringup/agent-tools.md:171`
documents this for the `plowrt` binary; the `--examples` form was not covered. Correct line:

    nix develop --command cargo build --release -p plowrt --features cuda --example step_bench

The cheap tell is `ls -la --time-style=+%H:%M:%S` on the binary before and after: mtime **and**
size must both move (cargo hardlinks its uplifted artifacts, so the mtime is the real link time).
Generalised: when a CPU harness and the runtime disagree about the same packet, suspect a stale
binary before suspecting the cubin or a runtime-only gate. Here the runtime logged
`decode ladder retains widest execution` while the validator qualified the same blob on CPU; I
spent the investigation on `plow_dyn_kvrow` and the three validator branches and both were
provably innocent (`plow_dyn_kvrow` is `= 1` unconditionally at `interp_sm120.cu:1022`, and both
arms return identical results from all three validators).

### 3f. Aiming #66: there is no unflipped switch in prefill attention

Before starting the FlashPrefill kernel project (#66, the growing term that decides 4096/8192/
15000 C1 TTFT), three cheap "it's just switched off" hypotheses were checked and all three are
dead. None cost a lease; all three would have.

1. **"The GQA2 kernel isn't being used."** The 26B is 16 heads / 8 KV heads, so GQA-2 pairing
   (load K/V once, run two query heads) is exactly its shape, and
   `PLOW_PF_SEG_FA256_GQA2` is an OPT_IN runtime knob sitting at OFF. But that is the *runtime*
   knob; the *emit* knob `PLOW_SEG_FA256_GQA2` is production-default **true**
   (`docs/flags-reference.md:299`), and `interp_sm90a_pfattn_hd256_gqa2_bkv32.cubin` is in the
   c1-lean recipe's `role_files` already. The shipped sliding kernel also carries
   `PLOW_NV_FA_GQA2_PAIR 1`, `PLOW_NV_FA_TMA 1`, `PLOW_NV_PACKED_FA_WGMMA 1`. Already on.

2. **"The full-attention layers aren't on the WGMMA arm."** This one looked strong:
   `FA_SM90_WG_ELIGIBLE(HD,BQ,BKV)` requires `BQ==64`, the hd512 object uses BQ=32 by default,
   and `interp_sm90a_pfattn_hd512.cu` reads `#define PLOW_NV_FA512_WG 0`. All true, and all
   irrelevant — that `#define` sits behind an `#ifndef`, and
   `build_sm90a_gemma4_segments.sh:278` passes `${PLOW_BUILD_PFATTN_WG:-1}`, i.e. **1**.
   Verified by rebuilding the object both ways with the script's exact argv: the shipped cubin is
   91 632 bytes, byte-size identical to the WG=1 build and 35 KB smaller than WG=0's 126 320.
   (The md5 differs only because the packet build adds `-Xptxas=-v`.) The full layers are on
   WGMMA at BQ=64. This is the third time a source-level `#define X 0` has been read as "off"
   when the build passes 1 — see [[plow-extra-defines-not-plumbed-to-packets]]: **read the nvcc
   argv, never the source default.**

3. **"Use BKV=64 for a wider QK^T."** BKV=32 does give the QK^T wgmma an N of only 32, which is
   a real tensor-core inefficiency, and `interp_sm90a_pfattn_hd256_bkv64.cu` exists. But
   `FA_SM90_WG_ELIGIBLE` accepts BKV=64 only at `BQ==64`, and the smem then goes 103 424 ->
   173 056 B, which drops the block from 2/SM to 1. Earlier notes called BKV64 "refuted from the
   record"; that was an armchair call, not a measurement. But it is **also not a lease-ready
   A/B**, which the first version of this section got wrong: `interp_sm90a_pfattn_hd256_bkv64.cu`
   carries neither `PLOW_NV_FA_GQA2_PAIR` nor `PLOW_NV_FA_WGITEM`, so swapping the sliding path
   onto it trades away GQA-2 K/V reuse (a 2x on K/V traffic) to buy the wider N — two variables
   at once, and probably a net loss. A clean test needs a `gqa2_bkv64` object that does not
   exist. Writing it is the smallest real unit of #66.

**So #66 is what the tracker already said it was: a kernel project, not a knob.** Both prefill
attention arms are already TMA + WGMMA, GQA2-paired where the shape allows, and 196 (sliding) /
351 (full) TFLOP/s is what this tiling achieves — against the GEMM path's ~780. The remaining
routes are genuine kernel work: widen the QK^T N (BKV=64 at 1 block/SM, untested), or a deeper
K/V pipeline. Do not spend another session looking for a switch.

### 4. PLOW_STEP_TIME: the step is device-bound, host cost is already hidden

Per-step means, stable over 512 steps (`log_every(128)`, gpu.rs:6829):

    dev_interp_ms   = 5.362      <- the megakernel
    sync_wait_ms    = 5.3637     <- host just blocks on it
    submit_us       = 35.7       = 0.036 ms
    dev_upload_us   = 20.7       = 0.021 ms
    dev_download_us =  7.4       = 0.0074 ms
    gap_us          = 705.6      <- NOT per-step serial

Measured TPOT 5.400 vs `dev_interp` 5.362 => **~0.04 ms of exposed non-kernel time**.

The 0.706 ms `gap` is a probe artifact, not a serving cost. Per `gpu.rs:6813-6826`, `gap_ns` is host
time *outside* `step_slots` between steps, and the probe pays one large inter-request interval
(curl + python + prefill) per 128 steps, so the mean is `(G + 127g)/128` -- constant in request
count, which is exactly why all four cumulative lines read 0.702-0.706. With G ~ 80 ms, per-step
g ~ 0.08 ms. Were all 0.706 serial, TPOT would be ~6.1 ms; it is 5.40.

**This refutes the CUDA-graph / launch-overhead route.** The megakernel alone (5.362) already
exceeds vLLM's entire TPOT (5.030); removing every host cost buys at most 0.04 of the 0.33 needed.

### Why the kernel is slow, and the one route left

7.64 GB/step at 5.362 ms = **1425 GB/s, ~45% of the 3352 roof** (vLLM 1519, also ~45%). A B=1 GEMV
streams each weight exactly once, so at half roof this is latency-bound on memory parallelism --
too few loads in flight -- not an HBM wall. That is the same constraint the reverted pre-issue
experiment hit, whose SASS verdict was: raising loads in flight needs the arm OUT of the
255-register megakernel, not a bigger DP ([[megakernel-array-loop-regression]]).

So the two prior nulls must be re-read: role-segmenting decode was measured **alone** (null), and
deeper pre-issue was measured **alone** (null, register-capped). The mechanism needs *both* --
segment the expert arm out so it has its own register budget, then deepen pre-issue inside it.

Not a knob flip: `emit.gemv_decode_role` (`PLOW_GEMV_DECODE_ROLE`) is gated on
`capabilities.dense_packet_contracts` (`lib.rs:7983`), and `emit.decode_grouped_moe_segments`
(`PLOW_SEG_DECODE_GROUPED_MOE`) only touches the grouped route, which is inactive at B=1
(`PLOW_MOE_DEC_LT=4` needs rung >= 4). Segmenting the B=1 expert arm is new emit work.

### Column verdict

| cell | TTFT | TPOT | p99 ITL | tok/s | still needs |
|---|---|---|---|---|---|
| 128/C1 | win 20.8/38.0 | lose 5.360/5.030 | win 5.42/5.75 | lose 182/189 | TPOT -0.33 |
| 1024/C1 | win 37.7/41.9 | lose 5.430/5.070 | win 5.48/5.83 | lose 176/187 | TPOT -0.36 |
| 4096/C1 | lose 102.3/93.4 | lose | win | lose | + TTFT -8.9 |
| 8192/C1 | lose 208.6/179.6 | lose | win | lose | + TTFT -29.0 |
| 15000/C1 | lose 423.4/352.2 | lose | win | lose | + TTFT -71.2 |

p99 ITL is won at all five cells -- plow's decode is steadier (5.360 mean -> 5.42 p99) than vLLM's
(5.030 -> 5.75). TPOT is the sole blocker at 128 and 1024; the other three additionally need the
prefill workstream (#66). TPOT at C1 is **not** KV traversal: 117x the KV (128 -> 15000) costs only
+0.23 ms, so it is a fixed per-step cost, and per (4) that cost is the kernel itself.

**The column does not reach 4/4 without a ~6.2% MoE decode kernel win.** Every in-kernel route
tried so far is closed by the 255-register cap; the segment-then-deepen combination is the one
untested mechanism.

---

## Interpreter gate + decode-kernel instruction audit (2026-09-24)

All timings `step_bench <assets> 1 128 64 --warmup 8` (kernel-only, no HTTP), 3 passes per arm,
interleaved and order-reversed inside ONE lease. Per-run sd 0.010-0.016 ms, per-arm sd of the
3 means 0.0006-0.0055 ms. `step_bench` slots=1 reads 5.377 against served TPOT 5.400 and
`dev_interp` 5.362, so it is a faithful proxy at ~0.2% noise and a kernel A/B no longer needs a
serving run. **Bar for 128/C1: TPOT must fall 0.330 ms** (5.360 -> under vLLM's 5.030).

### The gate is a busy spin, but it already backs off

Main decode gate, `interp_sm120.cu:3181` inside `PLOW_SYM(interp_sm120)`:

    while (ctr_poll(PLOW_CTR(prog.counters, pw.id)) < pw.threshold) { __nanosleep(64); }

`PLOW_NV_GATE_SLEEP` defaults to 64 ns ("0 spins flat out"). Contention is further limited by
"one thread per counter" (only `wait_len` threads per block poll, not all `PLOW_NV_THREADS`) and
by `CTR_STRIDE_U32 = 32`, one 128 B line per counter. The gates at 2711/2799/2853 that have NO
backoff are the *prefill* role loops (`plow_ws384_role_loop`, `plow_m128n128_direct_role`,
`plow_ws_role_loop`), not decode.

**Sweep (new `PLOW_TUNE_GATE_SLEEP` key, since reverted):**

| sleep ns | mean ms | sd | vs 64 |
|---|---|---|---|
| 0 | 5.3970 | 0.0017 | **+0.0190 (+0.35%)** |
| 1 | **5.3720** | 0.0017 | **-0.0060 (-0.11%)** |
| 16 | 5.3783 | 0.0046 | +0.0003 |
| 32 | 5.3733 | 0.0006 | -0.0047 |
| 64 | 5.3780 | 0.0010 | 0 |

**Spinning flat out is a real cost: +0.019 ms at ~11 sigma.** So the backoff earns its keep and
the 132-SM thrash concern is genuine. Among NONZERO values the dial is flat (16/32/64 span
0.005 ms), which reproduces AMD's verdict ("Swept 0/1/2/8 ... flat inside noise ... do not spend
anything tuning this", `interp.hip:5671`) except that AMD's sweep did not separate 0. `sleep=1`
is the best of the five by -0.006 ms (~3.5 sigma, consistent in all three passes) and the
mechanism is OVERSLEEP, not contention: at B=1 the 551-instruction program runs many gates on
the critical path, and a gate whose producer is already done still fails once and sleeps.

NOT SHIPPED: -0.006 ms does not justify a permanent knob. If wanted, change the source default
64 -> 1 (one line, no knob) after a 12B check.

### PTXSYNC V3 is structurally unavailable to packets

`interp_sm120.cu:2691` and `:2833` `#error` when `PLOW_NV_PTXSYNC != 1`, because
`plow_ws384_role_loop` and `plow_ws_role_loop` hand-copy the V1 protocol. The campaign builds
`--segmented`, so the decode object compiles them and any other value refuses to build.
`scripts/build_sm90a_cubin.sh:132` gets away with `-DPLOW_NV_PTXSYNC=3` because it builds a
non-segmented object. **V3 on a packet requires porting those role loops to the V3 gate** —
which is also where the missing backoff lives, so the two jobs are the same job.

V1's own measured win was -0.60% (12B decode, 18.398 -> 18.287, ~8 sigma); V2 is the control at
+0.16%, localising the effect to the seq_cst -> acq_rel fence downgrade.

### The four tuned flags the packet never receives: THREE MEASURED, ALL NEGATIVE

`build_sm90a_cubin.sh:132` sets a tuned group for the HAND-BUILT object. Nine flags reach the
packet by nvcc; four do not, and fall to source defaults. (`PLOW_NV_GEMV_MMA_UNB` was a false
alarm: the source default is already the hand-build's 12.)

| arm | flag | mean ms | sd | vs control | object | STACK |
|---|---|---|---|---|---|---|
| fctl | — | 5.3763 | 0.0015 | — | 1,820,552 | 544 |
| glu8 | `GV_UNROLL_GLU=8` | 5.5593 | 0.0015 | **+0.1830 (+3.40%)** | 1,862,664 | 576 |
| glu10 | `GV_UNROLL_GLU=10` | 5.6127 | 0.0015 | **+0.2363 (+4.40%)** | 1,906,184 | 672 |
| kun4 | `PLOW_NV_FA_KUN=4` | 5.3790 | 0.0010 | +0.0027 (null) | 1,820,552 | 544 |
| nost1 | `PLOW_NV_GEMV_NOSTAGE=1` | 5.4067 | 0.0055 | +0.0303 (+0.56%) | 1,828,744 | 544 |

`REG:255 LOCAL:0` on every arm — nothing spilled to local, but the GLU arms grew the object
(+42 KB, +85 KB) and the STACK FRAME (544 -> 576 -> 672), monotonically with unroll depth, and
the regression tracks it monotonically too.

**This is a clean instance of [[plow-insitu-vs-harness-fat-object]]: the hand-build's tuned
`GV_UNROLL_GLU=10` is 4.4% WORSE in the packet's 255-register megakernel.** Those values are
tuned for a different object. Do not import them.

It also **refutes** the hypothesis that motivated the arm (recorded here because it was wrong):
the B=1 walk spends 8.5% of instructions on software bf16 widening, so deeper unroll "should"
amortise the unpack over more loads. It does the opposite, monotonically. Register/stack
pressure at the cap costs more than the amortisation wins.

### SASS audit of the served decode object

111,024 instructions, `REG:255 STACK:544 SHARED:8848 LOCAL:0`. The kernel is dominated by
addressing and data movement, not math or loads:

    IMAD 18.2% (incl 8,342 IMAD.MOV/IMAD.IADD -- moves on the FMA pipe)
    ISETP/SEL/PRMT 15.6%     FP math 17.5%     PRMT alone 8.5% (9,466)
    SHF 7.3%                 branch 3.6%       shared LDS/STS 2.5%
    GLOBAL LOADS ~3.0% (LD.E.128 2,564 + LDG 696)
    HMMA 1.6%                MUFU 1.1% (576 EX2, 558 RCP, 44 RSQ)

* **Pure register shuffling is 12.3%** (`MOV` 5,064 + `IMAD.MOV` 7,706 + `UMOV` 908) — the
  255-register cap made visible, matching the reverted pre-issue arm's LOP3/S2R inflation.
* **`PRMT` is bf16->fp32 widening**, e.g. `PRMT R6, R6, 0x7732, RZ` (constant byte selector,
  RZ as the zero source), emitted in pairs. 48.2% sit within 24 instructions of a global load,
  **0.0% anywhere near an HMMA**, 42.5% within 8 of an FFMA. Consistent with the source: "sm_90a:
  BATCH>=2 decode rungs walk the weights on the tensor cores ... and the B=1 rung is untouched",
  so B=1 loads packed bf16 and widens in software for scalar FFMA. Only 3 `UPRMT`.
* **Reordering is already done.** LDG -> first-use of its destination: median 28 instructions,
  p25 12, p75 53; only 11.4% within 4. There is no pool of un-hidden loads to reclaim.
* **Bank conflicts are NOT determinable here.** `LDSM = 0`, shared traffic small (LDS 2,700,
  STS 327, SHARED 8,848 B), but static SASS cannot show conflict degree and ncu cannot attach to
  this cooperative megakernel ([[ncu-on-nix-plowrt]]). Recorded as unmeasured, not clean.
* Most 128-bit loads are GENERIC `LD.E.128` (2,564) rather than `LDG.E` (696), i.e. the compiler
  cannot prove global — expected for an interpreter reading a `void* const* T` table. The
  textbook fix is `__restrict__`, which is exactly what caused the NRN fold garbage
  ([[nvcc-restrict-barrier-hoist]]), so it is booby-trapped.

The design comment justifying the scalar path says "at M=1 it is weight-bandwidth bound". The
measurement disagrees: 7.64 GB/step at 5.362 ms = 1425 GB/s, **~45% of the 3352 roof**. The step
is not achieving the bound its design premise assumes. What that leaves is a *compute-issue*
hypothesis, and the unroll arms above are evidence AGAINST the obvious way to attack it.

### Round verdict

Net available from everything measured today: **-0.006 ms** (gate sleep 1), against the 0.330 ms
128/C1 needs. Closed this round: entry size (0.05), rung count (null), host/launch overhead
(0.04 exposed), PTXSYNC V3 (structurally blocked), GV_UNROLL_GLU (+3.4/+4.4%), FA_KUN (null),
GEMV_NOSTAGE (+0.56%), gate-sleep dial (0.006). The 26B C1 column remains blocked on a ~6.2%
MoE decode kernel win with no identified route.

All five tuning keys used for these measurements were REVERTED (measured losses; avoid knob
explosion). To recreate: add a `PLOW_TUNE_<X>` read in `manifest.rs::tuning()` with a STRING
LITERAL (a loop over names fails `every_emit_side_env_read_is_registered`, which scans for
`std::env::var("...")`), emit the define in `config_header` beside `moe_down_sg`, register
`env.PLOW_TUNE_<X>` in RAW_ENV, and **rebuild plowc** — `campaign.py` never does
([[plow-extra-defines-not-plumbed-to-packets]]).

---

## CORRECTION: B=1 already walks on the tensor cores, and the real scalar walk is the MoE expert arm (2026-09-24)

The SASS-audit section above concluded, from `op_gemm.cuh`'s `MM >= 2` gate and the comment
"BATCH>=2 decode rungs walk the weights on the tensor cores ... and the B=1 rung is untouched",
that the 8.5% `PRMT` was the B=1 dense walk widening bf16 in software for scalar FFMA.

**That is wrong.** The gate is `if constexpr (TC && (MM >= 2 || PLOW_NV_GEMV_MMA_B1))`
(`op_gemm.cuh:182`, and the same at 881/2669; the runtime twins at 499/1004/2796 read
`M > 1 || PLOW_NV_GEMV_MMA_B1`). `manifest.rs:2759` emits

    #ifndef PLOW_NV_GEMV_MMA_B1
    #define PLOW_NV_GEMV_MMA_B1 1

for every sm_90a packet whose tuning carries `gemv_mma_b1`, and a shipped packet's
`assets/plow_config.h` confirms `PLOW_NV_GEMV_MMA_B1 1` beside `PLOW_NV_GEMV_MMA 1` and
`PLOW_NV_GEMV_MMA_PAIR 1`. **The B=1 dense GEMV/QKV/GLU/lm_head walks already run on the tensor
cores.** The "route B=1 through the MMA walk" lever does not exist; it shipped in 8bf5b264/
b3b3b8d4's round.

Two further corrections to that section, both about method:

* **The instruction mix is STATIC, not a time attribution.** 111,024 instructions counted over
  `_Z12interp_sm90a11PlowProgram` (SASS lines 5-219,897) covers every rung and every op in the
  decode object at once. A percentage there says how much CODE exists, not where the B=1 step
  spends time. `HMMA 1.6%` in particular cannot be read as "the tensor cores are barely used" —
  one `m16n8k16` HMMA consumes 256 B of B-operand, so a bandwidth-bound walk needs few static
  HMMA sites and executes them enormously. The only other functions in the file are three
  `plow_moe_*_fp8_blk` kernels that a bf16 packet never launches.
* **`0.0% of PRMT within 24 instructions of an HMMA` was a real measurement of the wrong thing.**
  It is consistent with the widening living in a different arm entirely, which is what it does.

### Where the PRMT actually is: the MoE expert walk, and it is not compiled with its fix

`op_moe.cuh` contains **no `gvmma` call at all** — the tensor-core walk is `op_gemm.cuh`-only. The
expert arm dots weights scalar-wise:

    /* op_moe.cuh:478, inside dot8_fx */
    for (int j = 0; j < 8; j++) acc = fmaf(xs[j], __bfloat162float(w.x[j]), acc);

`__bfloat162float` on a packed pair lowers to exactly the audited idiom
(`PRMT Rd, Rs, 0x7732, RZ`, constant selector, `RZ` as the zero source, emitted in pairs). On the
26B-A4B the experts are essentially ALL of the 7.64 GB/step, so this — not the dense walk — is the
B=1 weight stream.

**And `dot8_fx` sits inside `#if PLOW_NV_GEMV_RB`, which is 0 in every packet.** Four sm_90a
decode arms are set by the hand-build scripts and reach no packet:

| flag | hand-build | packet | what it does |
|---|---|---|---|
| `PLOW_NV_GEMV_RB` | 1 | **0** | master gate for the row-blocked walks |
| `PLOW_MOE_DOWN_LANESPLIT` | 1 | **0** | MoE-down lane split |
| `PLOW_NV_GEMV_XREG` | 1 | **0** | dense GEMV activations in registers |
| `PLOW_NV_GEMV_KPANEL` | 1 | **0** | dense GEMV K-panel walk |

`build_sm90a_cubin.sh:132` sets all four; `build_sm90a_gemma4_segments.sh:49` sets
`PLOW_NV_GEMV_RB=1 -DPLOW_MOE_DOWN_LANESPLIT=1 -DPLOW_NV_FA_WPR=1 -DPLOW_NV_FP8_RB=4`. Verified
absent from packets four independent ways, because inferring it once already went wrong
([[plow-extra-defines-not-plumbed-to-packets]]):

1. No recipe under `scripts/campaign/recipes/` mentions `gemv_rb` in any spelling.
2. `manifest.rs::backend_nvcc()` builds `recommends` from exactly two keys — `gv_mm_max` and
   `gf_full` (line 1231-1238). There is no code path that could emit the others.
3. A real packet's `assets/build.json` records **2** `def.*` knobs total, both `null`.
4. `build_sm90a_gemma4_segments.sh` compiles only `pf*`/`pfattn*`/`pfgemm*`/`pfpacked*` objects —
   all PREFILL — and merely `cp`s `interp_sm90a.cubin`, which is the object decode runs. So the
   `-DPLOW_NV_GEMV_RB=1` on its line 49 never touches the decode megakernel.

### Why this is the best-evidenced remaining lever for the 26B C1 column

`op_moe.cuh:409-416`, on why the row-blocked arm exists:

> At the megakernel's 1 block/SM a warp that owns ONE output row keeps only UN weight loads in
> flight, and the H100 needs ~16 to reach its bandwidth (measured: a pure read at 1 blk/SM goes
> 1222 -> 2490 GB/s as loads-in-flight go 1 -> 8; `runtime/nvidia/experiments/hbm_ceiling_h100.cu`).

The B=1 step runs **1425 GB/s** (7.64 GB / 5.362 ms), ~45% of the 3352 roof — sitting at the
shallow end of precisely that curve, with `GV_MOE_RB=2 x GV_MOE_UN=2` worth of streams against the
~16 the part wants. `PLOW_MOE_DOWN_LANESPLIT`'s own note records **bf16 7.060 -> 6.766 ms at
1 block/SM** (-4.2%), and 1 block/SM IS the megakernel's occupancy. This also retires the standing
puzzle of the section above: the step is not achieving the bandwidth bound its design premise
assumes because the arm written to achieve it is compiled out.

**The honest risk.** Every hand-build flag imported into the packet so far has inverted at
REG:255 (`GV_UNROLL_GLU` +4.40%), and `GV_MOE_RB_DN=4` is on record as pushing the megakernel
REG 177 -> 229 in a leaner object. `LOCAL>0` on the arm object predicts a regression before any
timing is taken. Measured, not assumed.

**How the A/B reaches a packet.** `PLOW_EXTRA_DEFINES` IS a functional CMake cache var
(`runtime/CMakeLists.txt:523`, space-split at :545, appended to every served cubin at :707) — it
was only the *environment variable* that was never plumbed. But `plowc` overwrites it from the
manifest's `recommends` (`main.rs:1938-1946`), so an external env value cannot reach a packet.
Route: a `PLOW_TUNE_*` read in `tuning()` with a STRING LITERAL, a `rec.push` in
`backend_nvcc()`, `env.PLOW_TUNE_*` in RAW_ENV, and **rebuild plowc**. `recommends` deliberately
does not move the pairing hash (`pairing_hash_tracks_backend_requires_but_not_recommends`), so
control and arm packets differ in exactly one compile flag.

### RETRACTED, same day, by the measurement: the sm_90a arms DO reach every packet

The section immediately above claimed `PLOW_NV_GEMV_RB` and its group reach no packet. **That is
wrong.** `runtime/CMakeLists.txt:623-625` defines the sm_90a variant's define set as exactly "the
two define sets `scripts/build_sm90a_cubin.sh` applies unconditionally", and the second of those is
commented in place as

>     * the H100 decode-arm fixes (perf-data/gemma26b-h100-gemv-mlp.md):
>       GEMV row-blocking + warp router, MoE-down lane split, flash
>       warp-per-row, fp8 RB=4. All default 0 in the sources.

    set(_nv_var_sm90a
        -DPLOW_NV_MLA=0 -DPLOW_NV_MAMBA=0 -DPLOW_NV_DSA=0
        -DPLOW_NV_GEMV_RB=1 -DPLOW_NV_RB_GEMV=1 -DPLOW_NV_RB_QKV=1 -DPLOW_NV_RB_LMHEAD=1
        -DPLOW_NV_GEMV_XREG=1 -DPLOW_NV_GEMV_KPANEL=1 -DPLOW_MOE_DOWN_LANESPLIT=1
        -DPLOW_NV_FA_WPR=1 -DPLOW_NV_FP8_RB=4)

Every `_cubin_rows` object, decode included, gets it. `dot8_fx` and the row-blocked MoE arms are
compiled into the shipped decode megakernel and always have been.

**How four "independent" verifications all missed it.** Every one of them tested the
*recipe -> tuning -> recommends -> PLOW_EXTRA_DEFINES* route and the objects script. The flag does
not travel that route at all: it is hardcoded per-arch in CMakeLists, which no check looked at. The
`def.PLOW_NV_GEMV_RB` knob being `UNSET`/`OPT_IN` is a red herring — the knob system is not how this
define is delivered. Four checks of one route are still one check.

**THE ONLY VALID CHECK for "does define X reach object Y" is the generated nvcc command line:**

    grep -n "<object>.cubin" <packet>/assets/.cubin-build/CMakeFiles/nv_cubins.dir/build.make

which prints the full argv. That was available before the build and would have killed the
hypothesis in one command.

**AND md5 IS NOT A VALID A/B GATE.** All four arms had DIFFERENT cubin md5s and byte-identical
SASS (222,072 lines, PRMT 9,469, HMMA 1,768, identical `cmp`), with identical size 1,820,552 B and
identical `REG:255 STACK:544 SHARED:8848 LOCAL:0`. The md5 moved because the tuning table is
embedded in `plow_config.h`'s recipe-inputs text, which is compiled in as data. The guidance
"gate on cubin md5 differing" ([[plow-extra-defines-not-plumbed-to-packets]]) is too weak: md5
*equality* still proves a no-op, but md5 *inequality* proves nothing. **Gate on SASS.**

**What the arms measured, for the record** (step_bench slots=1, ctx 128, 3 interleaved
order-reversed passes, one lease). All three arms are redundant re-definitions of a flag already
set to the same value, so this is a null-by-construction and only confirms the harness's floor:

| arm | flag added | mean ms | sd | vs control |
|---|---|---|---|---|
| rctl | — | 5.3767 | 0.0025 | — |
| rb1 | `PLOW_NV_GEMV_RB=1` | 5.3780 | 0.0017 | +0.0013 (+0.02%) |
| ls1 | `PLOW_MOE_DOWN_LANESPLIT=1` | 5.3757 | 0.0006 | -0.0010 (-0.02%) |
| both | both | 5.3780 | 0.0000 | +0.0013 (+0.02%) |

Useful as a noise floor: three redundant rebuilds of the same kernel span 0.0023 ms, so the
step_bench slots=1 proxy resolves ~0.005 ms at n=3.

### What this closes, properly this time

**The MLP-depth route on the B=1 MoE walk is CLOSED.** The 1425 GB/s (~45% of the 3352 roof) is
achieved *with* `GEMV_RB`, `RB_GEMV`, `RB_QKV`, `RB_LMHEAD`, `GEMV_XREG`, `GEMV_KPANEL`,
`MOE_DOWN_LANESPLIT` and `FA_WPR` all on. So the shortfall against the pure-read curve
(1222 -> 2490 GB/s at 1 -> 8 loads in flight) is NOT un-enabled row blocking. `GV_MOE_RB=2` x
`GV_MOE_UN=2` is the *tuned* depth — `GV_MOE_UN=4` was measured worse (6.288 vs 6.194) and
`GV_MOE_RB_DN=4` costs REG 177 -> 229 — so the depth dial is at its measured optimum, not at a
default.

It also **reinstates last session's audit unchanged**: nine hand-build decode flags reach the
packet and four do not (`GV_UNROLL_GLU`, `PLOW_NV_FA_KUN`, `PLOW_NV_GEMV_NOSTAGE`, plus
structurally-blocked `PTXSYNC=3`), all four already measured. There is no fifth member. The
`PRMT`/`fmaf` scalar shape at `op_moe.cuh:478` is real and is what the tuned row-blocked arm
*already* compiles to; it is not evidence of a missing arm.

Net from this round: **0.000 ms**, and one route closed with a cheap, reusable test for the next
one.

---

## The B=1 op chain, measured with graphstat: the gate DAG has no slack, and the norms are the target (2026-09-24)

Reframing first, from today's own numbers. The B=1 step reads ~7.64 GB and runs 5.362 ms =
1425 GB/s. vLLM's 5.030 ms on the SAME byte count implies 1519 GB/s. **Both stacks sit at ~45% of
the 3352 GB/s roof**, and at bf16 with top-8 the byte count is irreducible (30 layers x 8 experts x
3 x 2816 x 704 x 2 B accounts for it). So the 6.2% deficit is NOT bandwidth and NOT bytes: it is the
fixed, non-walk term — the `F` of `op_gemv_mma.cuh`'s `step = F + W`, and task #71's ~1.3 ms.

`graphstat` (`crates/plowrt/examples/graphstat.rs`, no GPU touched) on the shipped C1 packet:

| T | ops | counters | edges | edges_tr | dead_ctr | ents | polls | bumps | bumps_live |
|---|---|---|---|---|---|---|---|---|---|
| 128..8192 | 691 | 691 | 840 | 840 | 1 | 75795 | 95463 | 75795 | 75794 |
| **1** | **551** | 551 | **640** | **640** | **31** | 29303 | 37252 | 29303 | 29272 |
| 2 | 551 | 551 | 640 | 640 | 31 | 30659 | 38637 | 30659 | 30628 |
| 4 | 551 | 551 | 640 | 640 | 1 | 33247 | 41283 | 33247 | 33246 |

### 1. CLOSED without a build: there are no redundant gates to remove

`edges_tr == edges` at every rung. **Transitive reduction removes nothing** — not one of the 640
producer->consumer edges is implied by a longer path. Any "prune the dependency graph" idea is dead
on arrival; gate count can only fall by *fusing ops*, which is what the op-71 GluNorm fusion did
(`lib.rs:6012`, "eliminating a separate RmsNorm op + counter gate").

### 2. The 31 dead counters are the MoE align ops, and their cost is <= 0.040 ms, NOT 0.27 ms

`dead_ctr = 31` at T=1/T=2 but `1` at T=4 and at every prefill program. The op histogram of T=1
(`op_seq <pkt> 9 0 551`) names them: **`PLOW_DOP_MOE_ALIGN_GEMMA_PF` x30** (+1 pre-existing), each
`blocks=1`, `i=[1, 128, 8, 4, ...]` = 1 row, 128 experts, top_k 8, group min 4. `lib.rs:6115` says:

> A rung below the threshold still carries the align op, but nothing waits on it: on the
> router -> GLU chain its 30 no-op packets cost **0.27 ms of a 5.9 ms B=1 step**.

**That 0.27 ms does not hold on this build.** At B=1 the grouped route never engages (`t=1 < min=4`),
so `PLOW_EMIT_MOE_DEC_LT=0` + `PLOW_GEMMA_MOE_DEC_GROUP=0` removes the align ops *and nothing else
that B=1 executes*. That is exactly the arm measured earlier today (section "the entry-size route is
CLOSED"), and its TOTAL was **-0.040 ms at 128/C1** — while also shrinking the object 28%, the stack
to 184 and shared to 3216, and dropping two rungs. So the align ops' own share is **at most 0.040 ms
and plausibly near zero**, an order of magnitude under the comment. The comment cites a 5.9 ms step
against today's 5.377, i.e. it is stale. Treat it as a recorded overestimate, not a budget.

(Still a real emit wart: 30 ops + 30 counters + 30 bumps per step that nothing consumes. Worth
removing for hygiene, not for the ladder.)

### 3. Where the fixed term actually is: 181 of 551 ops are norms

T=1 op histogram, 30 layers:

    90  PLOW_DOP_HEADNORM_ROPE          (3/layer)
    71  PLOW_DOP_GEMV
    60  PLOW_DOP_NORM_RESIDUAL_NORM     (2/layer)
    31  PLOW_DOP_RMSNORM                (~1/layer + final)
    30  each: MOE_ROUTER_GEMMA_TOPK, MOE_ROUTER_GEMMA_SCORE_FAST,
             MOE_EXPERT_GLU_NORM_GEMMA, MOE_EXPERT_DOWN_GEMMA,
             MOE_COMBINE_NORM_GEMMA, MOE_ALIGN_GEMMA_PF (dead),
             GEMV_GLU, FLASH_MERGE, FLASH_DECODE
    25  PLOW_DOP_GEMV_QKV
     1  each: SOFTCAP, EMBED, ARGMAX, ARGMAX_FIN

**181 ops (33%) are pure normalisation** — `HEADNORM_ROPE` 90 + `NORM_RESIDUAL_NORM` 60 +
`RMSNORM` 31 — each operating on ONE row at B=1 and each costing a stream entry, a counter, a gate
and a grid-wide ordering point. They move almost no bytes; their cost is entirely the per-op
overhead that constitutes `F`. This is the same shape as task #67 (fuse the prefill norm/Glu/rope
tail into GEMM epilogues) but on the DECODE program, where it has never been tried.

**The budget is the right size for once.** 0.330 ms over 181 ops is 1.8 us/op; over the 90
`HEADNORM_ROPE` alone it is 3.7 us/op. AMD's gate measurements are 3.46-13.16 us
(`interp.hip:5127-5200`), and the B=1 gate-sleep sweep showed spinning flat out costs +0.019 ms,
which is only consistent with gates being hit constantly. Fusing `HEADNORM_ROPE` into the
`GEMV_QKV` epilogue would delete up to 90 of 551 ops (16%) and their gates.

**Why it is not obviously free:** op 71's fusion is the precedent that it works, but every fusion
widens the surviving op's register footprint, and the decode megakernel is at `REG:255 LOCAL:0`
with a documented history of arithmetic-identical rewrites costing +0.37 ms
([[megakernel-array-loop-regression]]). Must be measured, single-variable, gated on SASS.

---

## MEASURED: the MoE expert walk is only 12% of the B=1 step (2026-09-24)

Half-K probe on the current build, the method `op_gemv_mma.cuh` validated: halve the k-range both
RB expert walks stream and `W = 2*(step-half)`, `F = 2*half-step`. Unconditional source edit at
`op_moe.cuh:2086/2120` (GLU over H) and `:1378/1415` (DOWN over I_moe); the router walk at `:911`
deliberately untouched (router weights are not part of the expert stream). Gated on **SASS
differing**, not md5. 3 interleaved order-reversed passes, one lease.

| arm | mean ms | sd |
|---|---|---|
| full walk | **5.3783** | 0.0006 |
| half walk | **5.0550** | 0.0010 |

    W (expert walk, k-proportional) = 0.6467 ms  = 12.0% of the step
    F (everything else)             = 4.7317 ms  = 88.0% of the step

**The expert weight walk is 12% of the B=1 step.** The 0.330 ms that 128/C1 needs is **51% of the
entire walk** — it cannot come from walk efficiency, and it could not come from eliminating the
walk outright in any realistic form.

### The byte model is contradicted, and the contradiction is the finding

From the op immediates (`i=[8, 704, 2816, 128]`: top_k 8, I_moe 704, H 2816, 128 experts) the
expert stream is exact: gate+up `5632 ch x 2 x 5632 B` = 63.4 MB/layer, down `22528 ch x 1408 B` =
31.7 MB/layer, **95.1 MB x 30 layers = 2.855 GB/step**. Each layer has its own table
(`moe.ewt.0..29`, 2 uses each), so nothing is shared. The probe removed ~1.58 GB of that and bought
0.323 ms, which implies **~4.9 TB/s — above the 3.35 TB/s roof, i.e. impossible from DRAM.**

The consistent explanation is `PLOW_GEMV_PREFETCH` (commit 32688ffb, "claim-ahead L2 weight
prefetch **before the gate**"): the expert weights are pulled toward L2 while earlier ops still
run, so by the time the walk's gate opens it reads largely resident data and the DRAM time is
hidden behind `F`. **The step is not bandwidth-bound; the prefetch already hid the bandwidth.**

### This retires the framing the last three rounds were built on

"7.64 GB/step at 5.362 ms = 1425 GB/s = ~45% of the 3352 roof" is **not a meaningful description of
this step** and should not be used to size levers again. It divides total weight bytes by the whole
step and reads the quotient as an efficiency, exactly the artifact `op_gemv_mma.cuh` already
retracted once for the B=32 case. The step is a sequence of 551 gated ops whose DRAM traffic is
prefetched; its time is not `bytes / bandwidth`.

Consequently CLOSED, with numbers rather than argument:

* **walk bandwidth / memory-level parallelism** — the walk is 12% of the step and reads from L2.
  Row blocking, deeper prefetch, PAIR, tensor-core walks: all bounded by 0.65 ms total, and all
  already on.
* **op count / gate count** — bounded at <= 1.3 us/op by the dead-align arm (30 `blocks=1` ops
  removed for <= 0.040 ms), so all 551 ops cap at ~0.72 ms and realistic merges buy ~0.04-0.08 ms.
  `edges_tr == edges` means none are removable without fusion.

### Where the 4.73 ms actually is, and the next probe

`F` is not gates (<= 0.72 ms) and not the expert walk (measured out). What remains inside it:
the **dense attention projections** — `GEMV_QKV` 25, `GEMV` 71, `GEMV_GLU` 30 — which carry their
own large weight stream through `op_gemm.cuh`'s gvmma path and were NOT touched by this probe;
`FLASH_DECODE`/`FLASH_MERGE` 30 each; 181 norm ops; and the megakernel entry.

**Next probe is the same instrument pointed at the dense walks:** halve the k-range in the gvmma
path and re-split. That isolates dense-weight time from the genuinely fixed remainder, and it is
the only remaining place a 0.330 ms can hide.

## MEASURED: the dense walk is FREE too — 88% of the B=1 step is per-op grid sync (2026-09-24)

Third half-K probe, same instrument, pointed at `gvmma_tile:200` (the k-span every dense
`GEMV_QKV`/`GEMV`/`GEMV_GLU` walks; `op_moe.cuh` has no `gvmma` call, so this isolates the dense
weights). Control reused, SASS-gated, 3 interleaved order-reversed passes, one lease.

| arm | mean ms | sd |
|---|---|---|
| full dense walk | 5.3770 | 0.0000 |
| half dense walk | **5.3927** | 0.0012 |

**Halving the dense walk made the step SLOWER**, by +0.0157 ms in all three passes. The object
shrank (1,820,552 -> 1,813,768 B) at identical `REG:255 STACK:544 SHARED:8848 LOCAL:0`, so the
penalty is the extra conditional the halving guard adds per tile call. `D = -0.031 ms`.

### The step decomposition is now complete

    dense weight walk    ~0.00 ms    0%     (measured: removing half the loads changes nothing)
    MoE expert walk       0.65 ms   12%     (probe 2)
    everything else       4.73 ms   88%

At ctx 128 the KV is ~1 MB, so `FLASH_DECODE` is not bytes either. **Almost none of the B=1 step
is memory traffic.** Both weight streams are hidden behind `PLOW_GEMV_PREFETCH`'s claim-ahead L2
prefetch, which is why neither responds to having its bytes halved.

### RETRACTION: the "<= 1.3 us/op" bound was invalid, and this reopens the op-count route

Earlier today I bounded per-op overhead at `<= 1.3 us` from the dead-align arm (30 `blocks=1` ops
removed for `<= 0.040 ms`) and used it to close the op-count route and cap task #79. **That
inference was wrong.** Those 30 ops are DEAD — `dead_ctr = 31`, nothing waits on their counters —
so the grid never blocks on them; only the single block that executes one is delayed. A dead op
costs no grid sync. It is not a proxy for a live op, and the bound never applied to the 551-op
chain.

The corrected model fits every measurement taken today:

    live op  ~ 4.73 ms / 551 = 8.6 us each   <- grid-wide ordering point, inside AMD's
                                                measured 3.46-13.16 us gate range
    dead op  ~ 1.3 us each                   <- no consumer waits, so no sync

**Both numbers are explained by the same mechanism, and the model is falsifiable: removing N LIVE
ops must save N x ~8.6 us.** It also finally explains task #71's ~1.3 ms "unidentified entry cost"
and why every bandwidth lever this campaign has tried returned a null — the step is a chain of 551
grid syncs with the memory traffic hidden underneath.

### The target, quantified: remove ~38 live ops

0.330 ms / 8.6 us = **38 ops of 551 (7%)**. Available, in order of cost-to-implement:

1. **The 3 `HEADNORM_ROPE` per layer -> 2 (30 live ops).** They are consecutive, independent, and
   each `blocks=2` of 132. Operand slots decide the split: q needs 6 (dst, src, q_norm, cos, sin,
   pos), k needs 6, v needs 3 (dst, src, pos) and takes NO norm weight and NO rope (`t2..t4` are
   all `65535:-` — it is a pure KV-cache store). **k+v = 8 slots exactly**, the op's full arity. So
   merge k and v; q stays. Predicted **-0.26 ms**.
2. The 30 dead `MOE_ALIGN_GEMMA_PF` ops: real, but ~1.3 us each = ~0.04 ms. Hygiene.
3. `MOE_ROUTER_GEMMA_SCORE_FAST` + `MOE_ROUTER_GEMMA_TOPK` are adjacent (30 pairs); `TOPK` is
   `blocks=1`. Fusing them is another 30 live ops = **-0.26 ms**.

1 and 3 together predict **-0.52 ms against the 0.330 needed** — the first credible surplus this
campaign has had. Both are op-71-shaped fusions (`lib.rs:6012` already did exactly this to remove
"a separate RmsNorm op + counter gate").

**Constraint on any such fusion:** the decode-ladder validator (`decode_rung.rs:179+`) compares
normalized instruction lists across rungs with `blocks` zeroed, so the merged op must be emitted
identically on EVERY compiled rung or the runtime silently falls back to widest-only execution.
And at `REG:255 LOCAL:0` a fusion that widens the surviving op can still lose
([[megakernel-array-loop-regression]]): measure single-variable, gate on SASS.

### The 8.6 us/live-op model is NOT yet validated — and there is prior evidence against it

Before building any fusion on it, two facts from the tree that bear directly on it:

**1. The three-HeadNormRope fold ALREADY EXISTS and measured a null.** `lib.rs:5016` (`PLOW_FUSE_HNR`,
off by default) folds all three hnr packets into `d_flash_decode`'s NRF arm. Its recorded result,
Gemma-4-12B fp8 occ4, 48 steps x 3 interleaved reps, token-identical serve:

    coarse deps onto q/k/v : 11.36 -> 11.99  (agent-scope fence per owner item)
                             11.36 -> 11.40  (the workgroup release actually required)
    FINE per-head deps     : 11.33 -> 11.37  (+0.3%)

with the stated reason: *"The deleted chain level was already almost fully OVERLAPPED: the hnr
packets' fine producer maps let them start before the slowest gemv workgroup, so their wall-clock
cost was ~the post-producer tail, **not a 10 us gate**."*

**That is direct evidence against ~8.6 us per live op — but only where FINE deps exist.** The arm
is gated `&& amd && fp8 && !fuse_qkv_fp8`, so it was never measured on this NVIDIA bf16 packet.

**2. This packet has NO fine deps at all.** `graphstat` reports `SE_FINE = 0` and
`counters == ops == 551` on every program. The cause is `hn_dep` (`lib.rs:4985`):

    if !gemv_family || fuse_qkv || fuse_qkv_fp8 { return vec![Dep::Coarse(gemv)]; }

and our decode program emits the FUSED `PLOW_DOP_GEMV_QKV` (25 of them), so every hnr dep falls
back to coarse. The source argues this is fine — *"the fused op is one uniform packet, so all
workgroups finish together and coarse costs ~nothing"* — which is an ASSUMPTION about the fused
GEMV's block skew, not a measurement on this packet.

So the two readings are not yet distinguishable:

* **A**: coarse gates on a uniform producer really are ~free, the remainder is genuine per-op
  execution, and op merging buys little (consistent with the AMD fold null).
* **B**: coarse gates on this packet cost ~8.6 us because nothing here is fine-grained, and the
  AMD null does not transfer (consistent with `F` = 4.73 ms over 551 coarse-gated ops).

**THE DECIDING EXPERIMENT (cheap, do this first, before any fusion):** inject N duplicate LIVE
coarse-gated ops into the B=1 program and measure the slope. Reading B predicts `+N x 8.6 us`;
reading A predicts ~0. A live injection needs a consumer that waits on the new counter — chain the
duplicate between an existing producer and its consumer so the op count rises with semantics
unchanged. 30 injected ops separate the two readings by 0.26 ms, i.e. ~50 sigma at the measured
0.005 ms floor.

Do NOT build the k+v hnr merge, the router fusion, or task #79 until that slope is known. Three
routes were closed today by measuring first; this one is the same shape.

## ANSWERED: a live coarse-gated op costs 3.01 us, and 110 of them are the 0.330 ms (2026-09-24)

`PLOW_TUNE_OPINJ=N` chained N duplicate LIVE `HeadNormRope` ops per layer between q's norm and its
consumer — dependency chain intact, same injection on every rung so the decode-ladder validator
still matches, correctness irrelevant to a timing probe. Gate was `graphstat`'s op count (this
edits the PACKET, not the kernel, so the cubin is identical by design and SASS/md5 are not the
gate). 30 layers, so N=1 -> +30 ops.

| inj | ops (T=1) | edges | mean ms | sd | vs control | us/op |
|---|---|---|---|---|---|---|
| 0 | 551 | 640 | 5.3763 | 0.0006 | — | — |
| 1 | 581 | 670 | 5.4693 | 0.0006 | +0.0930 | 3.10 |
| 3 | 641 | 730 | 5.6500 | 0.0020 | +0.2737 | 3.04 |

**Least-squares slope = 3.01 us per live coarse-gated op**, linear across both points, ~150 sigma
against the 0.0006 ms arm sd. The gate confirmed +30/+90 ops and +30/+90 edges exactly.

**BOTH earlier readings were wrong.** Reading A (coarse gates on a uniform producer are ~free) is
refuted: they cost 3.01 us each. Reading B (my 4.73/551 = 8.6 us) overestimated by 2.9x. The
correct accounting of the 4.73 ms remainder is therefore:

    per-op gate/dispatch   551 x 3.01 us = 1.66 ms   (35% of the remainder)
    work inside the ops                  ~3.07 ms    (65%)

And the AMD `PLOW_FUSE_HNR` null is now fully consistent rather than contradictory: there the hnr
ops carried FINE producer maps and really were nearly free; here every dep is coarse
(`SE_FINE = 0`) and each op costs 3.01 us. Same mechanism, different granularity. The dead-op
figure fits too — dead ops measured <= 1.3 us against live 3.01 us, because nothing waits on them.

### The op-fusion route is OPEN, and the arithmetic is exact

**0.330 ms / 3.01 us = 110 live ops of 551 (20%).** Candidates, all op-71-shaped, with what each is
worth at the measured slope:

| fusion | live ops removed | ms |
|---|---|---|
| 3 `HEADNORM_ROPE`/layer -> 2 (merge k+v; v takes no norm and no rope, so k+v fit the 8 operand slots exactly) | 30 | 0.090 |
| q `HEADNORM_ROPE` into the `GEMV_QKV` epilogue | 30 | 0.090 |
| `MOE_ROUTER_SCORE_FAST` + `MOE_ROUTER_GEMMA_TOPK` (adjacent; TOPK is `blocks=1`) | 30 | 0.090 |
| `MOE_COMBINE_NORM_GEMMA` + the following `NORM_RESIDUAL_NORM` (adjacent, both `blocks=1`) | 30 | 0.090 |
| drop the 30 dead `MOE_ALIGN_GEMMA_PF` | 30 (dead) | ~0.040 |
| **total** | **150** | **~0.40** |

That is **0.40 ms against the 0.330 needed** — the first route of this campaign with a measured
surplus rather than a hoped-for one. It does not require any of them individually to work: any four
of the five clear the bar.

**Order to build them in** (cheapest and least register-risk first): the k+v merge (both ops have
IDENTICAL immediates `i=[1,8,256,0,0,0,1,0]` and differ only in null `gamma`/`cos`/`sin`, so the
kernel already has the `skip_norm` parameter it needs); then the router pair; then combine+norm;
then the dead align; q-into-QKV last, since folding into a 132-block GEMV epilogue is the one that
can widen the survivor at `REG:255`.

**Constraints that apply to every one of them:** `decode_rung.rs:179+` compares normalized
instruction lists across rungs with `blocks` zeroed, so a merged op must be emitted identically on
EVERY compiled rung or the runtime silently falls back to widest-only execution. And at
`REG:255 LOCAL:0` a fusion that widens the surviving op can still lose outright
([[megakernel-array-loop-regression]]: +0.37 ms for an arithmetic-identical rewrite). Single
variable, gate on SASS, and check `LOCAL > 0` before believing any timing.

The probe knob is reverted.
