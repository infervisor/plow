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
