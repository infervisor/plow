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

That second fused screen wins the exact semantic boundary. One 256-thread
workgroup owns each row/head; sixteen 16-lane groups preload all FP32 state
vectors before convolution, then use vector non-temporal state traffic and
DPP reductions. The ROCm 7.14 confirmation measured 5.974 us at B1 and
6.514 us at B8, 28.9% and 29.5% faster than pinned vLLM's 8.40/9.24 us.
Convolution state was exact; B1 final output was exact; B8 final relative L2
was 4.20e-6. The kernel uses 128 VGPR, 45 SGPR, 2,336 B LDS, and no spills.
This passes the isolated and fused-block gates. Production integration remains
separate and must route by opcode geometry/capability, with an exact fallback
for unsupported shapes.

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

## Kernel and graph inventory

| Phase | Graph stage | Plow path | Status | Evidence and remaining gap |
|---|---|---|---|---|
| both | embedding, dense projections, `lm_head` | Generic tiled GEMM/GEMV carriers. Column-sharded `lm_head` is available. | **GF**, `lm_head` **O/M** | The gfx950 trace found most small GEMVs below the empty-packet time, so their poor bandwidth is protocol-shaped. Sharded `lm_head` was token-identical and improved 33.160 to 32.781 ms/token; see [the measured gate](archive/k3/k3-75tps-program.md#91-the-gate-the-readme-calls-impossible-is-the-wrong-gate). |
| both | AttnRes and following RMSNorm | `AttnRes` absorbs RMSNorm at all 186 layer sites; its existing body partitions batch rows across workgroups. | **O/M** fusion; B1 multi-CU **MISS** | The fusion is hard-on and bit-exact in `crates/devgen/src/k3.rs`. B1 multi-CU head splitting is unsafe without gang admission: raw AttnRes needs two grid barriers, and the fused RMS path needs two more. Static/GQ scheduling can currently let one resident workgroup claim multiple slices and deadlock. The historical graph attributed 115,758 polls to 187 AttnRes packets; see [counter graph](archive/k3/k3-decode-counter-graph.md#poll-cost-is-set-by-the-consumers-width-not-the-producers). |
| decode | KDA q/k/v/g projections | Four outputs collapse into decode-only `GemvQkvg`. | **O/M** | Removes three gates per KDA layer. The LDS guard deliberately routes all prefill rungs to separate GEMMs. |
| decode | KDA conv, gate, recurrence | `KdaConv3` plus `KdaStateStepG` reduces the six-packet chain to three. | **O/M** | The unfused 69-layer chain measured 5.03 ms on gfx950, with a 12.08 us/packet slope. Conv-to-recurrence fusion is correctly refused without double-buffered state because it races across workgroups. |
| prefill | KDA conv and recurrence | Same correct serial-token recurrence, with parallel channel/head tiles. | **GF** | At T=8192 this was 43.1% of gfx950 prefill and used 192/256 workgroups. A parallel scan or equivalent fused prefill architecture is absent. See [prefill attribution](archive/k3/k3-prefill-attribution.md#3-per-op-scaling-t1024---t8192-8x-tokens). |
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
- The gfx950 tuning database's 3,080 records were stale against the current
  build fingerprint, so the emitter used analytical fallback choices. No
  current full-model asset should be described as fully tuned until those
  records are regenerated and the fused-kernel → block → network gates pass.
