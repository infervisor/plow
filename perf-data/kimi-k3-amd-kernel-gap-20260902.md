# Kimi-K3 AMD kernel gap — 2026-09-02

This is the implementation and gate inventory for the current Plow AMD path against the pinned
[vLLM 0.28 MI355X baseline](kimi-k3-vllm-mi355x-baseline.md). Kimi-K3 is the
validation graph; the closure work below is runtime-, packet-, or shape-general.
MTP is excluded.

Status:

- **O/M** — optimized and measured on gfx950 with the stated scope.
- **I/U** — implemented, but not measured in the current matched gfx950 cell.
- **GF** — correct generic kernel or scheduling fallback.
- **MISS** — no equivalent optimized path.

## Reference boundary

The vLLM server selected fused KDA decode, AITER MLA prefill, AITER
MXFP4-BF16/SiTU-v2 MoE, AITER custom all-reduce, and norm/quant/reduction
fusions. Its matched 8192-to-1024 results are 20.768 ms C1 TPOT and 1133.93
output tok/s at C128. See the [baseline](kimi-k3-vllm-mi355x-baseline.md#results).

Plow does not yet have a current matched gfx950 run after the latest packet and
runtime changes. The historical gfx950 B1 result was 28.876 ms/token, versus
vLLM's current 20.768 ms TPOT, but that 1.39x gap is directional rather than a
release comparison. Current MI325X results are useful implementation evidence,
not substitutes for a gfx950 gate.

## Exact isolated gates — 2026-09-03

Measurements used clean one-GPU leases on the same MI355X host. The initial
KDA/reference measurements were separate leases; later arithmetic and cache
policy screens explicitly use interleaved or order-reversed same-lease A/B.
The reference was the pinned vLLM 0.28 image
sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032
running its ROCm fused_kda_decode benchmark. Plow used ROCm 7.14 and the
production op_kda.h bodies through
runtime/bench/amd/kda_decode_fused_exact.{hip,cpp}.

### Post-scan 8192-token prefill attribution

A fresh TP8 trace of the promoted BT64/BC16 scan captured all 2,971 packets
under one exclusive eight-GPU lease. Its 2,909.666 ms device span agrees with
the 2,952.023 ms endpoint TTFT within 1.44%; the trace-sweep median was
2,924.120 ms. Kernel bodies account for 2,901.893 ms (99.73%), while the
interpreter protocol gate accounts for only 8.933 ms (0.31%).

| current body category | time | device span |
|---|---:|---:|
| grouped MoE GLU + down | 1049.847 ms | 36.08% |
| MoE route + align + combine | 226.499 ms | 7.78% |
| TP reductions | 503.214 ms | 17.29% |
| KDA scan + conv + norm | 486.298 ms | 16.71% |
| MLA flash + merge + gate | 286.595 ms | 9.85% |
| dense GEMM/GEMV | 243.720 ms | 8.38% |
| AttnRes | 88.675 ms | 3.05% |

MoE is therefore the current first lever at 1,276.346 ms / 43.86%, not KDA.
Within MoE, down is 694.793 ms and GLU 355.054 ms. XReduce is 352.300 ms and
XReduce2 150.915 ms. The largest remaining KDA bodies are carry at 275.603 ms
and intra at 124.799 ms; MLA flash alone is 251.102 ms. The 69 KDA transformer
blocks span 2,060.098 ms with a 30.343 ms median, and the 24 MLA blocks span
849.505 ms with a 35.331 ms median. The pinned vLLM endpoint's 568.35 ms TTFT
implies an average whole-layer budget near 6.11 ms, but there is no matching
vLLM 8192-token per-kernel trace; only the isolated boundaries below are
claimed as apples-to-apples kernel comparisons.

| KDA decode boundary | B1 | B8 | Gate |
|---|---:|---:|---|
| vLLM fused conv + recurrence + gated RMSNorm | 8.40 us | 9.24 us | reference |
| Plow three-kernel control, same semantic boundary | 9.826 us | 21.294 us | loses 1.17x / 2.30x |
| Plow one-block-per-head fused candidate | 17.829 us | 18.434 us | rejected |
| Plow control with 1024 state-step workgroups | — | 14.409 us | diagnostic only; loses 1.56x |

The Plow oracle passed at B1 and B8: BF16 output and FP32 convolution state were
exact, and recurrent-state relative L2 was about 1.2e-8. The fused candidate
used 34 VGPR, 1,888 B LDS, and no spills. It is still slower, so none of it is
promoted to the production operator.

This is not strict dtype parity: vLLM stores convolution state as BF16 while
Plow stores it as FP32. The comparison is valid at the semantic boundary and
live geometry, but dtype parity remains a separate gate. The useful result is
the B8 scaling diagnosis: raising the state-step grid from the interpreter
ceiling of 256 workgroups to 1024 reduced that kernel from 12.659 to 7.241 us.
Multiple resident workgroups or an out-of-interpreter dispatch seam must be
tested before another fusion attempt.

An interleaved same-lease arithmetic A/B then replaced the state body's two
refined sqrt/div sequences with reciprocal square root, matching the reference
kernel's operation. At B8/grid256 the isolated state step moved from
12.620-12.647 us to 11.977-11.984 us (-5.1% to -5.3%). At the diagnostic
grid1024 it moved from 7.240 us to 7.065 us (-2.4%). BF16 output remained exact;
recurrent-state relative L2 was 1.46e-8. This candidate is promoted. It does
not close the boundary: the optimized three-kernel B8 chain is about 20.3 us
vs vLLM's 9.24 us fused kernel.

The next isolated state candidate keeps the FP32 V-first layout but maps each
BV8 value tile to a 128-thread workgroup containing eight 16-lane groups. Each
lane moves state through two vector loads and stores, and four DPP stages
replace the wave64 LDS-permute reduction. It uses 49 VGPR, 35 SGPR, 1,584 B
LDS, and no spills. A ROCm 7.14 confirmation measured B1 state at 2.884 us
and B8 state at 4.041 us, consistent with the initial 2.866/3.973-4.089 us
screen. The actual three-kernel chain improved from 10.346 to 9.632 us at B1
and 14.399 to 11.489 us at B8. The B1 final output
was exact; B8 final relative L2 was 4.21e-6. This is a decisive isolated Plow
kernel win, but remains benchmark-only: its chain still loses the pinned vLLM
fused boundary by 14.7% at B1 and 24.3% at B8. Production integration is gated
on a lean 128-thread dispatch path and a matched fused-block win.

A first 128-thread fused conv/state/norm block preserved the vector state
traffic and oracle but failed its promotion gate: 13.669 us at B1 and
15.022 us at B8, 40.5% and 30.9% slower than the corresponding three-kernel
vector chain. It used 65 VGPR, 45 SGPR, 1,840 B LDS, and no spills. One
workgroup per row/head exposes only 12/96 workgroups and serializes sixteen
BV8 state tiles without the reference kernel's preload schedule. The arm
remains benchmark-only and rejected; the next fused screen must use the
reference 256-thread, sixteen-group state pipeline.

That second fused screen initially appeared to win the exact semantic boundary,
but its fixture incorrectly reused the forget gate as the final output gate.
The corrected ABI-v2 kernel carries both gates independently. One 256-thread
workgroup owns each row/head; sixteen 16-lane groups preload all FP32 state
vectors before convolution, then use vector non-temporal state traffic and
DPP reductions. The corrected standalone object measured 6.902 us at B1 and
7.347 us at B8, 17.8% and 20.5% faster than the pinned vLLM 8.40/9.24 us.
Final outputs were exact at B1 and B8; recurrent-state relative L2 was 1.41e-8.
The kernel uses 128 VGPR, occupancy 4, and no spills.

The standalone gfx950 object uses the versioned
`plow_kda_decode_fused_256x16_2` marker and a 184-byte explicit kernarg. Packet,
devgen, and runtime route pure segments through it while leaving the ordinary
interpreter unchanged. A fresh emitted TP8 64-step stress passed exact per-token
cross-rank counter audit and all-rank token agreement. The matched one-layer
F-L-L-F gate measured fused 14.099 ms/token vs legacy 14.030 ms/token: fused is
0.49% slower, within noise, because its extra segment launches erase the kernel
win. The path therefore remains explicit opt-in and default-off; it does not
pass the fused-block promotion gate.

The exact BF16 MLA decode harness uses the emitted TP8 geometry rather than
the old FP8/gfx942 sweep: H=12, DK=512, DR=64, V=128, context=8192, ns=64,
scale=1/sqrt(192), and KV stride=32768. Plow's matched attention core
(flash plus standalone merge) measured 46.720 us at B1 and 113.397 us at B8.
The pinned vLLM/AITER core, including its H12-to-H16 pad/unpad, measured
43.232 us and 42.441 us. Plow is 8.1% slower at B1 and 2.67x slower at B8.
A grid256-to-grid96 screen was neutral and rejected.

The W_uv sub-boundary is not the cause. Plow's BF16 fold measured
10.756/10.832 us vs torch BF16 BMM at 13.895/13.949 us, about 1.29x faster.
The actual vLLM MXFP4 W_uv call measured 34.565/34.808 us at these small M
shapes. Plow's emitted fused merge-fold measured 30.936 us at B1 and
99.189 us at B8. Sums across separately timed calls are estimates, not a
serving claim: the actionable isolated miss is B8 flash attention. The fused
vs separate Plow path differed by relative L2 0.00101/0.00143; this is a
cross-path tolerance check, not yet an independent correctness oracle.
Changing GF4 to GF6 improved B8 flash from 81.381 to 75.681 us (-7.0%) but
regressed B1 from 14.488 to 19.032 us (+31.4%) and introduced four VGPR spills
and 20 B scratch. It remains benchmark-only. A stronger next screen is
batch/head-group-aware split selection: current B8 GF4/ns64 creates 1,536
logical units over 256 workgroups, while AITER selects 256 units.

The exact GF4 split sweep confirms that lever at the complete attention-core
boundary. B1 ns32/grid96 measured 38.532-38.612 us vs ns64
46.600-46.708 us (-17.3%) and now beats the matched vLLM/AITER 43.232 us
core. B8 ns32/grid256 measured 85.913-86.097 us vs ns64
113.281-113.341 us (-24.1%), but still trails AITER's 42.441 us by about 2x.
Both winning outputs were BF16-identical to ns64; GF4 resources remain
236 VGPR, 57 SGPR, 29,248 B LDS, and zero spills. This is still a cross-path
oracle. Production promotion requires per-rung split selection plus the
existing exact fallback.

The current grouped-MoE microbenchmark is not a valid AITER comparison. It
previously defaulted to unsharded I=3072; the emitted TP8 packet uses local
I=384. It also times only grouped GLU and down with routing metadata prepared,
while the selected vLLM AITER path uses A8W4 and its complete fused-MoE timing
includes sort, activation quantization and reduction. Router top-k remains
outside both boundaries. The benchmark now defaults to I=384 and exposes the
local width as an argument.

The final same-lease exact T=1024, H=3584, I=384, E=896, top-16, SiTU-v2
control measured Plow A4W4 stage 1 at 1.000-1.001 ms and stage 2 at
1.151-1.152 ms, or 2.151-2.153 ms for the pair. This supersedes an earlier
separate-lease 2.676 ms sample. The pinned AITER table's exact A4W4 peer is
205.373 us plus 144.527 us = 349.900 us; its actually selected A8W4 row is
199.569 us plus 139.605 us = 339.174 us. A8 intermediates explain only 3.2%
of that reference result. Plow's like-precision pair is still 6.15x slower
than AITER A4W4.

A live pinned-vLLM run measured the larger AITER fused-MoE boundary, including
sort, input quantization, both expert stages and reduction, at 575.825 us;
fused top-k plus that boundary was 640.586 us. The incomplete Plow expert pair
alone is therefore 3.74x slower than the larger expert boundary and 3.36x
slower than route plus experts. That is decisive even though the live
precision and boundary details differ.

The synthetic H=I=256, E=3 oracle passed before timing: stage 2 checked 14,208
values with no unwritten output and 5.43e-8 worst relative error; stage 1
checked 28,416 values with 0.4837 worst FP4 ULP. A production-size correctness
oracle remains required. The first target is stage 2: Plow padded 16,384 live
rows to 57,728 BM64 rows, then paid an FP32 scatter/partial round trip. The
AITER cell uses BM32 with an atomic/reduction stage. Stage 1 follows: compare
its A8W4 GUI-interleaved 32x128x256 schedule with the Plow A4W4
BM64/BN256/BK128 schedule under a matched precision oracle.

The follow-up used production's 512-thread workgroup at the same exact shape.
BM32/WNc8 reduced padded rows from 57,728 to 32,448, but stage 2 moved only from
1.153 ms to 1.134 ms (-1.65%); scatter plus reduction moved from 1.225 ms to
1.203 ms (-1.80%). Direct atomic accumulation includes the required 14 MiB
zero and lost to the corresponding scatter-plus-reduce boundary: 1.289 vs
1.225 ms at BM64 (+5.2%), and 1.212 vs 1.203 ms at BM32 (+0.7%). The
production-shape 3,670,016-value oracle passed with no mismatches. Both
candidates are rejected. The earlier 256-thread BM32 result was not a
production-representative gate.

The AITER-inspired cache-policy follow-up independently screened non-temporal
loads for the packed expert weights and their E8M0 weight scales. The exact
BM64 production mapping ran forward and reverse order under one uncontended
GPU lease; every cell used 31 timing iterations and passed the 3,670,016-value
stage-2 oracle. Times below are the two-run medians reported by each process:

| payload NT | weight-scale NT | stage 1 (ms) | stage 2 (ms) | pair (ms) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1.000 / 1.001 | 1.151 / 1.152 | 2.151 / 2.153 |
| 0 | 1 | 1.090 / 1.088 | 1.160 / 1.160 | 2.250 / 2.248 |
| 1 | 0 | 1.072 / 1.071 | 1.169 / 1.171 | 2.241 / 2.241 |
| 1 | 1 | 1.116 / 1.115 | 1.173 / 1.173 | 2.289 / 2.289 |

Scale NT with a temporal payload regressed stage 1 by 8.8% and stage 2 by
0.7%; the combined cell also lost. The scale-NT candidate was removed rather
than retained as a production/build knob; payload NT remains default-off.

An AITER-style packed-weight experiment then permuted MXFP4 payloads to
`[N/32][K/64][lane64][16 B]` and E8M0 scales to
`[N/32][K/64][lane64]`, letting each wave load its B fragments directly into
registers without the B-side LDS copy. A nonuniform-weight CPU oracle passed
both stages (down worst relative error 5.43e-8; GLU 0.4837 FP4 ULP).

At BM64/eight waves, direct B duplicates each fragment across the two M-wave
groups and lost repeatably: stage 1 moved from 1.002/1.001 ms to
1.163/1.163 ms, while stage 2 moved from 1.149/1.152 ms to 1.172/1.170 ms.
The experiment and its layout flag were removed. At the AITER-like BM32/four-
wave stage-2 geometry, direct B was a small win over an otherwise matched
staged-B control (0.912/0.911 vs 0.927/0.927 ms; scatter-plus-reduce
0.982/0.981 vs 0.996/0.996 ms), but the incremental 1.5-1.7% does not justify
a second weight slab or a production capability route by itself.

The useful result is the four-wave geometry: the existing staged-B body at
BM32/four waves is about 19% faster than BM64/eight waves at the complete
scatter-plus-reduce boundary. It remains benchmark evidence, not a promoted
interpreter setting: the current persistent interpreter is an eight-wave
object, so a safe promotion needs a separately capability-routed lean segment.
Expanding the stage-2 grid from 512 persistent workgroups to all 14,196 output
tiles moved 0.927 ms to only 0.916 ms, ruling work assignment out as the main
gap. The pinned like-precision AITER cell is still 144.527 us, 6.3x faster than
the best Plow kernel and 6.8x faster than its scatter-plus-reduce boundary.
Its remaining structural advantages are a two-stage direct-B/B-scale register
pipeline, a three-slot async A LDS ring, and 16x16x128 scaled MFMA issue groups;
Plow still stages both operands then issues 32x32x64 MFMAs.

A standalone four-wave 16x16x128 CDNA4 body then tested those issue groups
directly without changing the production fallback. The ROCm 7.14 builtin takes
the same `v8i32` operand carrier as the 32x32 form and an `f32x4`
accumulator. With ordinary row-major B loads, the independent nonuniform CPU
oracle passed all 14,208 down values with no unwritten output and 5.43e-8 worst
relative error. At the exact T=1024, H=3584, I=384, E=896, top-16 BM32/four-
wave geometry, 31 forward/reverse-interleaved iterations measured the existing
scatter-plus-reduce boundary at 0.996 ms and the 16x16x128 boundary at 1.070 ms
(+7.4%).

Coalescing the B loads through an offline-preshuffled payload did not recover
the loss: the matched boundary was 0.994 vs 1.096 ms (+10.2%). Keeping the
whole 384-K A tile in LDS to remove per-K publication barriers measured 0.996
vs 1.100 ms (+10.4%). A literal two-buffer register-carried B prefetch inflated
the live state and regressed to 1.813 ms against 0.998 ms (+81.6%). The exact
production-size constant-pattern cross-path check remained bit-identical, but
the preshuffled variants lost before a separate nonuniform layout oracle was
warranted. All experimental bodies, flags, and harness wiring were removed.
The negative result narrows the remaining gap: changing MFMA shape alone is
not sufficient in HIP C++; AITER depends on its compiler-controlled fragment
layout and low-live-range async pipeline. Stage 1 was not screened because the
stage-2 prerequisite failed its complete-boundary gate.

An ISA/resource comparison then extracted the exact stage-2 object produced by
the pinned vLLM ROCm image
`sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032`.
At T=1024, H=3584, I=384, E=896, top-16 it selected
`mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x256x128_vscale_fix3_fp4opt_v1_pm1`.
The matched Plow object was BM32/BN256/BK128 with four waves:

| exact stage-2 object | VGPR | SGPR | LDS | private | spills | static scaled MFMA | barriers |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plow BM32/four-wave | 150 | 92 | 55,552 B | 0 B | 0 | 4 x `32x32x64` in a three-trip K loop | 4 |
| pinned-image AITER | 98 | 46 | 16,640 B | 0 B | 0 | 24 x `16x16x128`, fully scheduled | 6 |

The MFMA counts represent equal work: one 32x32x64 issue has twice the MACs of
one 16x16x128 issue, and Plow executes its four-instruction body three times.
The material difference is scheduling and live state. Plow stages A, B, and
both scale tensors in LDS (`runtime/amd/op_moe.h:3337-3342`), then performs
LDS reads plus immediate `lgkmcnt` waits around each two-issue MFMA group
(`runtime/amd/op_moe.h:3548-3574`). AITER maps each wave to four 16-column N
subblocks and streams preshuffled B/B-scales directly into register fragments
(`/tmp/aiter-main/aiter/ops/flydsl/kernels/mega_moe/gemm2.py:353-424`). Its
two-stage carry launches the next B/B-scale and A-scale VMEM reads ahead of
the current MFMA cluster while A rotates through staged LDS
(`gemm2.py:517-652`). The extracted ISA confirms that overlap: vector loads
are interleaved through the 24-instruction MFMA sequence, rather than an LDS
publication/read/wait phase around a short MFMA group.

This evidence rules out spills as the existing Plow cause: both original
objects have zero spills. It also makes bytes alone insufficient. Plow reports
1.069 GB of mandatory stage-2 traffic and reaches 1.15 TB/s; its 234.9 MB FP32
partial scatter plus a separate reduction is an extra boundary absent from
AITER native BF16 packed atomics, but cannot explain 6.8x by itself. Removing
the unused GLU bridge from the down-only LDS allocation reduced Plow LDS to
39,168 B with no timing change (0.928 vs 0.928 ms). Compiling scatter-only code
reduced the object by 18% and SGPR 92 to 90, also with no timing change
(0.925/0.926 vs 0.924/0.927 ms). Thus LDS capacity and dormant epilogue branch
code are not the dominant stalls at the present 150-VGPR body.

The highest-confidence resource lever was tested only as a standalone object.
A two-block launch bound was non-binding and retained 150 VGPR. Combining lean
LDS with a four-block bound reached 128 VGPR only by adding 16 VGPR spills and
68 B of private storage. One-lease forward/reverse timing passed the exact
3,670,016-value oracle but regressed kernel time from 0.928/0.928 to
1.042/1.046 ms (+12.5%) and scatter-plus-reduce from 0.998/0.996 to
1.114/1.114 ms (+11.7%). The variant and all harness flags were removed. A
credible next body must reduce live ranges structurally to AITER-like resource
levels without spills and must remain a separate lean object routed only for
pure compatible segments; constraining the current mega body is rejected.

The exact installed source was then recovered from that image rather than
inferred from the newer checkout. It is amd-aiter 0.1.19 under the MIT license.
`compile_mixed_moe_gemm2` confirms that this object is four waves with
TM32/TN256/TK128, PM1/SBM32, A4W4 input and BF16 C-shuffle atomic output. Its
`b_nt` argument is explicitly ignored. The weight byte layout is
`[E][N/16][Kbytes/64][K-lane 4][N-lane 16][16 B]`; scale buffers are padded to
256 rows and eight K-scale columns and pack two M/N halves by two K halves into
each dword. Each wave owns N64 and eight `f32x4` accumulators (two M16 by four
N16). A ping/pong A tile lives in LDS while B stays in registers; the next A,
B and scale loads are issued before current compute. The installed scheduler
then repeats VMEM, four MFMAs, DSRD, four MFMAs. The 16 KiB BF16 C-shuffle
aliases the A LDS after compute and finishes with 16 packed-BF16 atomic
instructions per workgroup.

An independent source-backed gate constructed raw nonuniform FP4 payloads and
E8M0 scales, applied only the documented ABI permutations, and decoded the
reference with a separate FP4 lookup and FP32 matmul. It passed bit-for-bit for
32 routed rows x 3584 columns (114,688 values, zero max absolute/relative
error). A second exact cell used the same xorshift routing histogram as the
Plow harness: T1024/H3584/I384/E896/top-16, 32,448 padded rows and 1,014 M
blocks. Forward/reverse process order under one GPU lease measured AITER at
0.172842/0.173401 ms for the kernel and 0.173482/0.173522 ms including the
required output zero. The matched Plow BM32/four-wave process measured
0.927/0.927 ms for DOWN+scatter and 0.996/0.996 ms for scatter+reduce. On this
independently generated exact routing cell, the source-backed object is 5.35x
faster at the kernel boundary and 5.74x faster at the complete boundary. The
published AITER table remains 0.144527 ms; the local cell is deliberately
reported separately. A Plow port should preserve this layout and scheduler in
a generated or hand-scheduled lean object. A HIP-C++ body that merely holds
both B tiles live has already failed, so it must not be folded into the mega
interpreter.

### Standalone reference-derived object gate

The phase-1 Plow package under
`runtime/bench/amd/lean_moe_stage2_ref/` now reproduces the exact object from
the pinned vLLM ROCm image digest. The emitted 9,072-byte object has SHA-256
`3034c6cf087a0229cd723f226b74df4763f05f0c3cdf07b194bc03649a7899f5`.
Its manifest fixes the gfx950 capability, 96-byte ABI, shuffled MXFP4/E8M0
layouts, and the measured 98 VGPR/46 SGPR/16,640-byte LDS/no-spill resource
contract. The runtime gate does not import AITER or FlyDSL: it loads the raw
object through the HIP module API, applies independent host layout transforms,
and compares against an independent CPU FP4 decode/matmul oracle.

The independently loaded exact gate passed all 114,688 focused values
bit-for-bit. A repeated one-lease run at T1024/H3584/I384/E896/top-16,
32,448 padded rows and 1,014 M blocks measured 0.169241/0.169042 ms for
forward/reverse kernel order and 0.174121/0.173202 ms including output zero.
This is gate-only: no CMake target, production loader, model predicate, or mega
interpreter arm was added.

### Dependency-free native 16x16x128 object

`native_kernel.hip` in the same isolated package now implements the schedule
without an AITER or FlyDSL build/runtime dependency. Four wave64 waves own N64
each; B payload and paired scale dwords are preshuffled and held in a two-stage
register pipeline while the next A tile is loaded across the current eight-MFMA
group. Two A LDS slots ping-pong. Row IDs and gates occupy the 256-byte LDS tail
and are fetched once per workgroup. The selected pinned AITER configuration has
`use_async_copy=False`, so this object does not claim the optional three-slot
asynchronous-copy path.

The Nix ROCm 7.14 build emits executable-section SHA-256
`374a485d18af2f762718ddfff762909210af357004704ef746e6864afbd94282`.
The canonical full object from this run is
`64c385ee049239e8356365b0950c386b7643beb675b7077fc4fc7c2466065e1a`;
the compiler embeds a non-semantic per-output module identifier, so the build
gate uses the stable executable-section hash and records the full-object hash.
AMDGPU metadata reports 94 VGPR, 46 SGPR, 16,640 bytes of launch LDS, zero
private bytes, and zero spills. Static ISA has 24 scaled 16x16x128 MFMAs, six
barriers, 38 waits, and native packed-BF16 atomics. These closely match the
reference object at 98/46 registers, 16,640-byte LDS, 24 MFMAs, six barriers,
39 waits, and zero spills.

The independent nonuniform oracle again passed 114,688/114,688 values with
zero absolute and relative error. Two native forward/reverse runs measured
0.180522/0.180002 and 0.181041/0.179922 ms at the kernel boundary, and
0.186241/0.184641 and 0.184682/0.185321 ms including output zero. A
contemporaneous extracted-AITER run measured 0.170641/0.169482 and
0.175441/0.173161 ms respectively: native is about 6% behind AITER. Against the
previous exact Plow BM32/four-wave control (0.927 ms kernel, 0.996 ms complete),
native is 5.1x/5.4x faster. The available current eight-wave standalone control
measured 1.135/1.206 ms with its exact oracle passing.

Verdict: the kernel is a material, repeatable candidate for a separate lean
segment, but is not production-promotable yet. It still needs a production
object marker/ABI loader gate, exact stage-1-to-stage-2 layout parity, and
fused-block/full-network measurements. It must not enter the mega interpreter.

## Kernel and graph inventory

| Phase | Graph stage | Plow path | Status | Evidence and remaining gap |
|---|---|---|---|---|
| both | embedding, dense projections, `lm_head` | Generic tiled GEMM/GEMV carriers. Column-sharded `lm_head` is available. | **GF**, `lm_head` **O/M** | The gfx950 trace found most small GEMVs below the empty-packet time, so their poor bandwidth is protocol-shaped. Sharded `lm_head` was token-identical and improved 33.160 to 32.781 ms/token; see [the measured gate](archive/k3/k3-75tps-program.md#91-the-gate-the-readme-calls-impossible-is-the-wrong-gate). |
| both | AttnRes and following RMSNorm | `AttnRes` absorbs RMSNorm at all 186 layer sites; its existing body partitions batch rows across workgroups. | **O/M** fusion; B1 multi-CU **MISS** | The fusion is hard-on and bit-exact in `crates/devgen/src/k3.rs`. B1 multi-CU head splitting is unsafe without gang admission: raw AttnRes needs two grid barriers, and the fused RMS path needs two more. Static/GQ scheduling can currently let one resident workgroup claim multiple slices and deadlock. The historical graph attributed 115,758 polls to 187 AttnRes packets; see [counter graph](archive/k3/k3-decode-counter-graph.md#poll-cost-is-set-by-the-consumers-width-not-the-producers). |
| decode | KDA q/k/v/g projections | Four outputs collapse into decode-only `GemvQkvg`. | **O/M** | Removes three gates per KDA layer. The LDS guard deliberately routes all prefill rungs to separate GEMMs. |
| decode | KDA conv, gate, recurrence | `KdaConv3` plus `KdaStateStepG` reduces the six-packet chain to three. | **O/M** | The unfused 69-layer chain measured 5.03 ms on gfx950, with a 12.08 us/packet slope. Conv-to-recurrence fusion is correctly refused without double-buffered state because it races across workgroups. |
| prefill | KDA conv and recurrence | Model-independent BT64/BC16 scan for supported gfx950 geometry, with the serial-token recurrence retained as an explicit oracle/fallback. | **O/M** | A production TP8 8192-token parity gate produced identical prompts and all 32 output IDs while reducing TTFT 3714.31 to 2960.95 ms (-20.28%). The scan is now the gfx950 default; unsupported shapes and `PLOW_KDA_CHUNK=0` stay serial. See [the current baseline addendum](kimi-k3-plowrt-mi355x-baseline.md#post-baseline-gfx950-kda-scan-promotion-gate). |
| decode | KDA gated norm and output projection | Gated norm is separate; exact workgroup sizing and norm-to-GEMV folding exist. The B1 double-bank conv/state arm is opt-in. | **O/M** narrow fusions; double-bank **I/U** on gfx950 | A grid barrier prevents folding gated norm into the recurrence. The double-bank arm is B1-only and lacks a current matched gfx950 network gate. |
| decode | MLA A projections | Optional four-stream `GemvQkv`; default remains separate projections. | **I/U** | The fusion exists but is disabled. It needs a matched block and network A/B because merging independent producers can trade overlap for packet count. |
| both | MLA latent norms, RoPE, KV writes | Norm-to-GEMV folding is on; specialized head-norm/RoPE and KV writers remain separate. | **O/M** norm fusion; remaining ops **GF** | The live vLLM baseline uses BF16 KV. Plow's optional FP8-KV route must not be credited in that comparison. |
| prefill | MLA attention | `FlashMlaPrefill` plus fused softmax merge/V projection; large buckets can use the four-wave V2 object. | **O/M** isolated; current network **I/U** | V2 measured a 2.25-2.79x MLA-only improvement historically. Segment/object compatibility is load-checked. A current full-network gfx950 TTFT gate is missing. |
| decode | MLA attention | `FlashMlaDecode` plus `MlaMergeFold`, with fixed measured `nsplit`. | **I/U** current gfx950 | The old B16 trace put flash decode at only 1.2% of the step, so this is below MoE and batching work. The output gate deliberately remains separate: fusing it lost 1.19 ms at 32K by destroying overlap. |
| both | router and top-k | Router GEMV followed by phase-specific top-k. Wide decode objects have local-selection tuning. | **GF**, wide tuning **I/U** on gfx950 | There is no fused router-to-expert carrier comparable to a strided grouped selection. Keep it separate until an exact unfused-chain gate proves a generalized fusion. |
| decode B1 | routed SiTU/MXFP4 experts | One grouped GLU packet and one grouped down packet replace per-slot expert packets. | **O/M** | The real-weight TP8 gate improved 103.161 to 62.893 ms/token with bit-identical logits. |
| decode B2-B32 | routed SiTU/MXFP4 experts | Sorted/aligned grouped prefill-style GLU, down, and combine path; tile search, parallel prefix, and local router are compiled into wide objects. | **I/U** gfx950 | This is less fused than the selected AITER path and retains route, align, GLU, down, combine, and large intermediates. On the historical gfx950 B16 trace it consumed 20.3% of the step, dominated by low-fill stragglers; see [B16 attribution](archive/k3/k3-batched-decode-design.md#87a-the-moe-prefill-path-is-203-of-the-step-and-most-of-it-is-straggler). |
| prefill | routed SiTU/MXFP4 experts | Expert sort/align, grouped GLU, grouped down, combine; A4W4 bridge available. | **O/M** kernels; **GF** composition | Historical gfx950 prefill spent 35.6% in MoE and exposed a 179 ms small-rung floor from BM64 padding. Per-bucket expert tiling and a fused intermediate-free composition remain open. |
| both | shared expert | Gate/up/SiTU collapse into `GemvGlu`; down remains a GEMV. | **O/M** fusion; down **GF** | The fused SiTU epilogue preserves both transformed branches. The block residual and sharded routed-up gather are also folded into the existing tail collective. |
| both | TP reduction | Inline system-scope collectives inside the interpreter. Wide two-shot aggregation is available. | inline **O/M**; wide aggregation **I/U** gfx950 | Inline reduction measured 20.6x faster than RCCL at TP4. The wide aggregation reduces remote atomics from 2432 to 8 per rank/collective, but still needs the matched gfx950 gate against vLLM's AITER custom all-reduce. |
| decode | sampling and rank agreement | Per-row argmax through B128, row-parallel cross-rank folding above B1, every-token drain/counter audit, configurable full-rank token-read cadence, and a four-token device capture ring. | cadence **O/M**; row-parallel fold and multi-step **I/U** | The default remains every-token agreement. The wide-decode MM sweep is token-identical with the parallel fold, but does not isolate its speedup. Deferred readback preserves the per-token counter drain and bounds stop/cancel overshoot to four tokens. A matched long-output A/B remains required. |

## Packet, segment, and rung coverage

| Area | Current state | Status | Gap |
|---|---|---|---|
| Decode interpreter | One persistent interpreter launch per rank and token; graph ops are internal counter packets. Counter re-arm can overlap the in-flight launch. | **O/M** mechanism, current gfx950 network **I/U** | The historical graph remained 1739 levels deep with maximum liveness five. Fusion and batching, not launch-per-op library substitution, are the main levers. |
| Prefill packets | Whole-model compile-time ladder; large MLA prefill can be split into a pure four-wave segment, with other work on the eight-wave object. | **O/M** correctness, current gfx950 performance **I/U** | TP prefill needs per-segment/all-rank barriers for collective progress. Cross-request packing is absent. |
| Decode ladder | Independent-sequence rungs B1 through B128. Runtime chooses the narrowest rung covering the highest occupied slot; tiered B1/B32/B128 object inventories avoid wide-object dead lanes. | B128 object policy **O/M**; matched workload **I/U** | Real-weight TP8 B1 and B128 smoke gates pass. A production C128 A/B selected GEMV MM8 over MM16 by 10.0% output throughput with identical output; gfx950 B64/B128 objects now default to MM8 with the mandatory walk. Matched long-output B32/B64/B128 measurements remain. |
| Continuous batching | Fixed compiled slots with admission, parking, chunk cursors, and rung selection through B128. | **I/U** | A real-weight C128 smoke reached rung 128 with 128/128 exact completions and zero sheds. Prefill remained serialized, so this validates coverage rather than throughput. |
| Prefill/decode overlap | Chunked prefill, starvation-free round-robin chunk service, interleave, and throughput-defer policies exist. | **I/U** gfx950 | True co-packing is blocked on a packet ABI with per-row owner/KV base/position/length and recurrence boundaries. Prefill/decode scratch is still shared, so phases interleave but cannot overlap. P/D disaggregation stays behind this gate. |
| Device multi-step | Up to four greedy tokens feed device-to-device through a capture ring; each token still drains and audits counters, while host token reads follow the TP agreement cadence. | **I/U** | Short real-weight TP8 output is token-identical at K1/K4. The first A/B exposed and fixed an ignored `PLOW_MULTISTEP` cap; repeat after rebuild, then run long-output latency/throughput gates. |

## Prioritized closure gates

1. **Matched baseline gate:** run Plow on gfx950 at 8192-to-1024 C1, B32,
   B64, and B128. Require exact output count, zero failures, all-rank
   counter/token agreement, stable NUMA conditions, and an archived packet/op
   trace. This turns every **I/U** above into evidence or rejection.
2. **B128 runtime gate:** argmax storage/reduction and decode ladder coverage
   now reach B64/B128. Gate B1/B32/B64/B128 independently, then compare
   C128 with one wide cohort versus four B32 cohorts. Package the narrowest
   compatible interpreter object for each rung range.
3. **Wide grouped-MoE gate:** fuse or directly schedule sparse low-fill rows
   without the prefill-style padded path. Compare each kernel to its exact
   unfused chain, then the routed block, then the full network at B8/B32/B128.
   Preserve routing, SiTU, MXFP4 scaling, and deterministic combine semantics.
4. **Packed-prefill gate:** batch prompt chunks across requests without sharing
   recurrent state. Sweep per-bucket expert tiles, then evaluate a parallel KDA
   scan. Gate kernel, layer, full prompt, and mixed prefill/decode service in
   that order.
5. **Narrow-chain gate:** implement batched/multi-CU AttnRes and test only
   fusions whose producer has one semantic consumer. Require same-binary
   fused/unfused block and network A/Bs; packet-count reduction alone is not a
   pass.
6. **Collective and observation gate:** validate wide two-shot aggregation on
   gfx950, then device multi-step and deferred readback with bounded
   cancellation. Counter timeout detection remains per step even if redundant
   rank-token reads are sampled.
7. **Scheduling gate:** add a measured prefill/decode policy before considering
   P/D disaggregation. Report TTFT, TPOT, ITL tails, output throughput, occupied
   rung, and cohort count together.

## Current gfx950 bring-up findings

- The first full TP8 gate with independent ordered segments and eight device-side XCD windows
  per segment reduced the 8192-row `amd-bench` prefill from 2911.529 to 2238.263 ms
  (-673.266 ms, -23.12%). The matching production mux gate at 8191 input tokens completed
  3/3 requests with TTFT p50/p90 2400.863/2401.177 ms and 0.07% spread. This is 4.23x the
  pinned vLLM 8192-token TTFT of 568.35 ms; the one-token shape difference is explicit and a
  same-client 8192-to-1024 serve gate remains the publication authority. Packet checksum:
  `fnv1a64:ffefa4f7d413c3dc`; output checksum: `fnv1a64:ad600d375800bd1e`.
- The final identical-client prefill cell used `vllm bench serve` against Plow's
  `/v1/completions`: exact 8192 input tokens, C1, one output token, three measured requests
  after one warm-up. The endpoint passed the text coherence gate and completed 3/3 with mean
  TTFT 2274.58 ms and median 2276.64 ms. This is 4.01x the pinned vLLM 568.35 ms TTFT. TPOT
  is undefined for a one-token cell; the 8192-to-1024 decode/ITL publication cell remains
  pending and must use the same client and endpoint contract.
- A checkpoint-bound production-mux rerun at exactly 8192 input tokens and one output token,
  with the raw gfx950 MLA V2+SV segment enabled and packed KDA disabled, completed 3/3 after
  one warm-up at TTFT p50/p90 2257.594/2257.616 ms (0.05% spread). This is a prefill-boundary
  result, not the final same-client 8192-to-1024 publication cell; against the pinned vLLM
  568.35 ms TTFT it is 3.97x slower. Packet checksum: `fnv1a64:d1559238add38257`;
  output checksum: `fnv1a64:3ade942d4a3ee2fd`.
- The raw dense-BF16 MLA V2+SV object passed isolated production dispatch and improved the
  synthetic TP8 8192-token run from 1686.643 to 1552.287 ms. Its route is deliberately limited
  to pure dense-BF16 MLA segments at T>=2048. The standalone packed-KDA object passed its
  compile and capability gates but faulted under the real TP8 runtime route, so C1 remains on
  the primary interpreter until that fault has a correctness root cause.
- Kimi-K3's source checkpoint lacks 120 derived MLA tensors. The measured checkpoint farm was
  prepared with `scripts/kimi_k3_prep.py`; direct loading now fails closed on a missing derived
  tensor instead of producing a synthetic result. On ROCm 7.14, concurrent asynchronous copies
  into the shared 29 GB rank weight allocation produced GPU write-to-read-only faults. A single
  upload slot completed all eight rank loads, so one is now the safe default and values above
  one remain explicit experiments.
- The emitted program and B1/B32/B128 object families load with real Kimi-K3
  weights on eight MI355X GPUs. The C128 short gate completed 128/128 requests,
  reached rung 128, and reported zero rejects/sheds.
- Low-rung object packaging is material: in the same 64→2 smoke cell, using the
  B128 object at rung B1 cost 100.716 ms TPOT vs 54.283 ms with the B1 tier,
  with identical output checksums. These are smoke numbers, not a publishable
  8192→1024 baseline.
- The C128 smoke spent 101.25 s serving 64→2 and only 2.53 output tok/s because
  recurrence-safe prompt chunks are serialized. This is direct evidence that
  decode B128 alone does not close throughput without the packet-ABI work above.
- The gfx950 dense-GEMM TuneDB was refreshed under one exclusive 299-second
  lease with the same ROCm 7.14 toolchain used by the compiler fingerprint.
  BF16 produced 500 raw rows; the K3-derived BF16+MXFP4 ladder produced 1,440.
  The current digest `gfx950-8d25b6a4d36627e9` has 1,210 qualified records
  covering five production rungs for each of 242 selectable operator cells.
  The 3,080 historical rows remain correctly stale and unselectable. All four
  gfx950 TuneDB assertions pass; the two gfx942 stale-record assertions remain
  open and were not weakened by this campaign.
