# Gemma 4 12B H100 prefill rung audit

2026-09-11. Scope: native Plow BF16, W8A16 and W8A8 GEMM, every requested
row count individually. Source/packet inspection, compiler tests and an
all-width W8A16 body experiment; their scopes are distinguished below.

## Baseline rung matrix before this change

P = current packet emits that prefill width. S = historical synthetic body and
loaded-interpreter GEMM screen. A = current async W8A16 body screen. Missing
widths use available bucket/chunk scheduling; this is not a specialized kernel
at that width. BF16 column uses b32-head-audit.json; FP8 columns use
w8a16-packed-audit.json and w8a8-b16-audit.json. The baseline inventories are
under `plans/gemma4-12b-roofline/` in the experiment workspace.

| Requested prefill M | BF16 | W8A16: BF16 A, E4M3 W | W8A8: E4M3 A/W | Baseline compiler restriction |
|---:|---|---|---|---|
| 1 | No PF; short PF uses128 | No PF; uses128 | No PF; uses128 | Append rejected with any positive decode width |
| 2 | No PF; uses128 | No PF; uses128 | No PF; uses128 | Append allowed only if max decode<2 |
| 4 | No PF; uses128 | No PF; uses128 | No PF; uses128 | Append allowed only if max decode<4 |
| 8 | No PF; uses128 | No PF; uses128 | No PF; uses128 | Append allowed only if max decode<8 |
| 16 | No PF; uses128 | No PF; uses128 | No PF; uses128 | Append allowed only if max decode<16 |
| 32 | No PF; uses128 | No PF; uses128 | No PF; uses128 | Append allowed with B16, not B32 |
| 64 | No PF; uses128 | No PF; uses128 | No PF; uses128 | Append allowed with B16/B32 |
| 128 | P, S | P, A | P, S | Shipped rung |
| 256 | No PF; other buckets | No PF; other buckets | No PF; other buckets | Explicit append supported |
| 512 | P, S | P, A | P, S | Shipped rung |
| 1024 | P, S | P, A | P, S | Shipped rung |
| 2048 | P, S | P, A | P, S | Shipped rung |
| 4096 | P, S | Absent in current packet | Absent packet; S only | PLOW_MAX_CHUNK>=4096 |
| 8192 | P, S | Absent in current packet | Absent packet; S only | PLOW_MAX_CHUNK=8192 |

The historical M1 synthetic screen was the N262144 head, not all eight layer
projection shapes. Decode M1/2/4/8/16 kernels are a different phase and do not
fill these prefill coverage gaps. The experiment completed140 W8A16 body cases across all14 requested widths: eight
model N/K shapes plus two tails per width. All exact sampled oracles and
1,400,877,564 complete output bytes match control. All140 cases also pass
Compute Sanitizer memcheck with zero errors. This uses the generic128
WGMMA tile, so it does not establish exact-M specialization or loaded
interpreter performance at newly emitted widths. See
[all-width W8A16 experiment](gemma4-12b-h100-data/w8a16-all-rungs-summary.json).

## Verified explicit rung matrix after this change

Each entry below is a separately emitted and strict-audited prefill program;
decode programs remain distinct. Appending widths is explicit opt-in.

| Prefill M | BF16 packet | W8A16 packet | W8A8 packet | New W8A16 body evidence |
|---:|---|---|---|---|
| 1 | PASS | PASS | PASS | Oracle + complete output match |
| 2 | PASS | PASS | PASS | Oracle + complete output match |
| 4 | PASS | PASS | PASS | Oracle + complete output match |
| 8 | PASS | PASS | PASS | Oracle + complete output match |
| 16 | PASS | PASS | PASS | Oracle + complete output match |
| 32 | PASS | PASS | PASS | Oracle + complete output match |
| 64 | PASS | PASS | PASS | Oracle + complete output match |
| 128 | PASS | PASS | PASS | Oracle + complete output match |
| 256 | PASS | PASS | PASS | Oracle + complete output match |
| 512 | PASS | PASS | PASS | Oracle + complete output match |
| 1024 | PASS | PASS | PASS | Oracle + complete output match |
| 2048 | PASS | PASS | PASS | Oracle + complete output match |
| 4096 | PASS | PASS | PASS | Oracle + complete output match |
| 8192 | PASS | PASS | PASS | Oracle + complete output match |

All three packets have19 programs:14 prefill plus5 decode. BF16:
13,434 instructions/5,753 exact cases; W8A16:13,834/1,278;
W8A8:16,522/5,814. BF16/W8A16 PF programs each have766 ops;
the400-instruction total difference comes from their five decode programs
(542 vs622 ops each). W8A8 PF programs have958 ops. Total57 programs,43,790 instructions,12,845 cases.
Packet/build hashes, exact GEMM PCs/block counts and all-op inventories are in
[the compiler audit evidence](gemma4-12b-h100-data/prefill-all-rungs-compile-summary.json).
These are inspection packets:85GiB KV each, exceeding H100 capacity.
No new all-width full-model serving claim. A B1 W8A16 recipe is prepared,
uncompiled, with unchanged20480 context and8192 request span; it is included
in the evidence JSON.

## Exact GEMM shapes required at every prefill rung

C[M,N] = A[M,K] W[N,K]^T. The same eight projection shapes occur in all three
precision packets. Counts sum328. All K satisfy current W8A16 async K%16 guard.

| Projection | N | K | Instructions per rung |
|---|---:|---:|---:|
| gate/up | 15360 | 3840 | 96 |
| local K/V | 2048 | 3840 | 80 |
| down | 3840 | 15360 | 48 |
| local Q | 4096 | 3840 | 40 |
| local O | 3840 | 4096 | 40 |
| global Q | 8192 | 3840 | 8 |
| global tied K/V | 512 | 3840 | 8 |
| global O | 3840 | 8192 | 8 |

There is additionally one BF16 head Gemm at M1,N262144,K3840 in each ordinary
prefill program, even for FP8 weights. Packed serving gathers terminal rows;
its head workload needs independent packed-terminal coverage. Do not replace
the head's M1 with the prefill bucket width in the inventory.

The complete requested layer-GEMM matrix is112 shapes per precision,336
precision/shape cells before tile candidates, head, tails or loaded variants.
Each BF16/W8A16 prefill program has766 ops:329 GEMMs,48 Glu,144 HeadNormRope,
96 NormResidual,97 RmsNorm,48 FlashPrefill, and one each Embed/SoftCap/Argmax/
ArgmaxFin. W8A8 has192 additional QuantFp8 ops,958 total. GEMM coverage does
not qualify these other operations. Attention is40 local HD256/KV8/window1024
and8 global HD512/KV1/Q16 layers at each rung.

## Native kernel paths

| Format | Dispatch and native body | TMA / specialization status |
|---|---|---|
| BF16 | interp_sm120.cu1130 Gemm/Med/Small; op_gemm_sm90.cuh d_gemm_sm90_tma, uni256, ws384, m64 roles | Maps in i6/i7 enable TMA; absent maps fall back to ordinary body. WS384 separates producer and two consumer groups. Existing M64 and head roles require matching metadata and launch resources. |
| W8A16 | interp_sm120.cu1275 GemmFp8; op_gemm.cuh1192 d_gemm_fp8; op_gemm_sm90.cuh d_gemm_sm90_impl<Weight=uint8_t> | BF16 tensor math after E4M3 conversion. Current async optimization uses cp.async raw weight copies, in-place expansion, proxy fence and CTA barriers; no weight TMA. Generic128x128x64 tile is not an exact-M specialization. |
| W8A8 | interp_sm120.cu1187; op_gemm_sm90.cuh d_gemm_w8a8_sm90_tma and uni256/WS roles | Real FP8 WGMMA; E4M3 maps in i6/i7, per-M activation/per-N weight scales. Missing maps use cp.async fallback except specialized entry which traps. |

Opcode GemmFp8 alone cannot establish activation precision; PLOW_NV_W8A8 and
packet tensor formats must agree. plowc tuned.rs214 distinguishes W8A16 and
W8A8 inventories. Three BF16 opcode names share the Hopper dispatch; a
Small opcode is not proof of a distinct small tile. Existing W8A8 M1 role
(interp_sm120.cu965) requires M1,K<=17408,K%16=0, weight map, scales, zero
row offset; existence does not establish routing for this Gemma packet.

Swizzled shared layouts exist in native GEMM bodies. No all-rung GEMM bank
conflict measurement is established here. Earlier zero LSU bank-conflict
counts were attention-specific. SASS containing TMA somewhere in a combined
object does not prove a particular packet arm executes it.

## Compiler change and qualification recipe

Shipped widths come from devgen/lib.rs8192. PLOW_PF_LADDER_APPEND already
accepts explicit positive widths below cap; defaults need not change.
The old guard at8455 required every prefill width exceed every decode width.
That is stronger than the canonical reader's requirement. decode_rung_lo in
packet/devbuild.rs3012 scans backward through the ascending decode suffix.
Full PF1..8192 followed by decode1..16 is unambiguous at8192->1.

Implemented minimal change: validate the canonical reader returns the actual
emitted prefill count before appending decode programs. Keep rejection of
ambiguous layouts. Existing consumers in manifest.rs, projection_rewrite.rs,
attention_prefill_role.rs and asset/devblob.rs use the canonical boundary.
Two packet boundary tests PASS. Release plowc build PASS. Regression covers all14 PF widths followed by1..8 decode widths, equal1/1
boundary and ambiguous[1,2]+[4,8]. No packet ABI change needed for this layout.
GPU execution of newly emitted small prefill programs remains a separate gate.

Prepared w8a16-all-prefill-rungs-recipe.json clones the existing precision and
segment recipe, sets PLOW_MAX_CHUNK=8192 and appends1,2,4,8,16,32,64,256.
This is an inspection/tuning packet: current W8A16 B16 sliding-KV sizing at8K
exceeds H100 capacity. Do not serve it as-is or shrink rings without the
request-span contract. Individual loaded-op probes can qualify its shapes
without loading its full KV allocation.

## CUTLASS and DeepGEMM techniques to adapt

1. First close14-width coverage with existing native bodies and real segment
   counters. For M1..64 compare SIMT, MMA and small WGMMA tiles with actual
   queue block counts; for M128..8192 compare existing64/128/256 tile widths.
   Rank complete model cost, not only standalone throughput. Preserve native
   fusion where register/shared-memory requirements remain compatible.
   Static tile-count inference: M<=128,N512 has only4 tiles with the current
   128x128 tile, leaving most of a132-CTA launch without a tile. Root measured
   some small-N async regressions around9–12%; no utilization-counter proof
   accompanies this inference. Prior mixed-MMA whole-main serving regression
   argues for testing lean small-M roles and split-K without inflating the
   main interpreter register budget.
2. W8A16 next candidate: separate raw-copy/dequant work from consumer WGMMA,
   using the existing lean role entry and explicit stage ownership. TMA can
   move raw E4M3 but cannot replace the conversion arithmetic. Initially
   preserve current BF16 operand/scale semantics and avoid adding quantization.
3. An alternative mixed-input design converts one operand into registers for
   register/shared WGMMA, avoiding shared expansion. CUTLASS's mixed-input
   collective uses this structure. Plow currently places weights on B; its
   applicability needs an operand/layout or output-transpose design, not a
   direct substitution. Treat as a separate prototype. [CUTLASS mixed input](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp).
4. Tune tile and ring size together. DeepGEMM's SM90 search includes small-M
   candidates, limits N to control spills, filters swizzle geometry, and
   requires at least3 stages, often4 for smaller tiles. These are candidate
   heuristics, not measured Plow defaults. Preserve Gemma's per-channel scale
   contract rather than importing block-scale assumptions. [DeepGEMM SM90 heuristics](https://github.com/deepseek-ai/DeepGEMM/blob/66081d4c9c7d7c44f13fea402e5b622aa0f409c2/csrc/jit_kernels/heuristics/sm90.hpp).
5. CUTLASS persistent cooperative and ping-pong schedules amortize prologues
   and overlap consumer epilogues. Plow already persists across queue work;
   adapt tile-stage ownership inside homogeneous GEMM segments before adding
   another global scheduler. Register-role transitions must remain outside
   incompatible interpreter arms. [CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).

Every staging variant needs correct generic/async proxy ordering, completed
WGMMA readers before reuse, numerical tails, loaded queue completion and
sanitizer coverage. Follow the [official PTX memory and WGMMA rules](https://docs.nvidia.com/cuda/parallel-thread-execution/).
None of these design proposals establishes a cuBLASLt/vLLM speed advantage.
