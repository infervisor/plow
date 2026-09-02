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
