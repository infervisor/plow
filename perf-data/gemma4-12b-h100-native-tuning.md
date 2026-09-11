# Native Gemma 4 H100 tuning

The goal remains unmet. These changes improve native Plow kernels; they do not
establish vLLM parity, optimal kernels for every op, or maximum concurrency.
No CUTLASS or DeepGEMM kernel library is linked into Plow.

## Native changes

The SM90 GEMM producer can prefetch its tensor maps with
`PGM90_WS384_PREFETCH=1`. Combined with `PGM90_WS384_ISSUE_CURSOR=1`, tile-coordinate
mapping runs outside the producer's K loop. The shared BF16/W8A8 WGMMA body,
packet ABI, queues, dependency counters and segment execution remain in use.

`scripts/build_sm90a_gemma4_segments.sh BASE_DIR OUTPUT_DIR` builds the measured
candidate from an existing Gemma BF16 object set. It replaces ordinary/packed
GEMM and attention objects. It copies the existing decode and light-op objects,
preserving their fused execution. Serve with the output directory as
`--pf-seg-dir`, its decode/prefill cubins, and the matching packet configuration.
This is an experimental BF16 serving recipe, not a universal architecture default.

Attention selects the already implemented HD512/BKV32 body, enabling the existing
32-row TMA descriptors to be used. Packed requests resolve their own slot's map
pair. Partial KV tiles retain asynchronous-copy fallback. Q/K/V layout and
numerical math remain native Plow implementations.

The compiler's tuning inventory previously allowed W8A16 and W8A8 kernels to
match either activation format. Matching now requires the exact activation dtype;
a regression test covers both rejection directions.

## GEMM ladder evidence

The campaign derives 49 unique shapes from the emitted BF16 packet: eight
projection shapes at M128/512/1024/2048/4096/8192, plus the M1 vocabulary head.
It tests BF16 and synthetic W8A8 counterparts. The actual model's head remains
BF16; testing its FP8 counterpart does not deploy a quantized head.

Baseline and candidate each run the standalone native body and the real loaded
interpreter cubin: 196 executions, 392 measured cells. All pass sampled FP64
reference comparisons (relative L2 below 0.004), all-output finiteness, and
completion-counter checks. Measurements are warm-cache medians of 15 launches,
one campaign with alternating variant order. They are not full serving timings.

| M | BF16 segment speedup, geometric mean | W8A8 segment speedup, geometric mean |
|---:|---:|---:|
| 128 | 1.077x | 1.047x |
| 512 | 1.065x | 1.046x |
| 1024 | 1.055x | 1.047x |
| 2048 | 1.056x | 1.042x |
| 4096 | 1.033x | 1.051x |
| 8192 | 1.021x | 1.049x |

This compares two producer configurations, not every possible tile/pipeline.
W8A16 checkpoint performance is not represented by W8A8 measurements.

## Counter attribution

The probe additionally removes dependency edges from a synthetic program whose
GEMM output tiles are independent. It preserves the same interpreter kernel and
queue. At M1024/N4096/K3840, the initial measurement attributes about 1.2 us BF16
and 1.3 us W8A8 to the already-satisfied gate and completion signals. Production
dependencies are not removed. This does not measure blocked dependencies or an
entire mixed-op graph; body-versus-segment differences also include compilation
and resource differences.

## Attention hardware counters

Privileged Nsight Compute invocation now works without changing driver policy.
Six standalone cases cover HD256/BKV32, HD512/BKV16, HD512/BKV32, with and without
maps. All report zero
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` and zero measured
local spill requests. Those LSU metrics do not certify every tensor-core access,
other shapes, or all interpreter operations.

On the fixed ragged long-history case, profiled HD512 duration is 2.758 ms with
BKV16 versus 1.762 ms with BKV32/TMA. Both pass the independent attention oracle.
BKV32 reserves roughly 200 KiB of shared memory, so its effect inside a combined
object must be measured, not inferred from the standalone result.

## Serving and remaining coverage

The combined attention/GEMM candidate passes concurrent-versus-isolated output,
ragged requests, output limits, slot reuse, cancellation/recovery, and context
rejection checks. The 16K/C1 diagnostic remains about 1.26 s TTFT versus roughly
0.67 s for the earlier vLLM diagnostic. C16 measurements vary materially; no
throughput win is established. These diagnostics generate only two output tokens.

The all-op packet inventory includes floating parameter bits, tensor extents,
segment membership and counter counts. `packet-op-coverage.json` enumerates every
operator in the serving ladders. Performance qualification remains pending for
the complete set, including norms, RoPE, GLU, KV handling, decode and sampling.
Each architecture/dtype/shape needs measured selection; compatible operations
should retain fusion rather than acquire a separate launch merely for labeling.

Every new compile now emits `build.json.kernel_cases` directly from every
program's instructions. Cases retain opcode/arm, integer parameters, exact float
bits, extra stride words, operand presence, byte extents, workgroup count and
instruction PCs. This covers all opcodes without a Gemma-only allowlist. It is
architecture-neutral and accompanies the existing architecture, precision and
object inventory. It does not change dispatch or certify a case as tuned.
The Gemma BF16 recompile covers all 5,774 instruction PCs exactly once across
nine programs (1,883 parameter cases). Its packet bytes, generated header and
pairing hash remain identical. All 45 manifest tests pass.

The next HD256 candidate uses BKV64. Eight standalone cases pass the independent
FP64 oracle and CUDA memcheck (zero errors). In the mapped ragged case, BKV32
takes 81.581 us versus 60.778 us for BKV64. The combined interpreter increases
static spill loads from 192 to 532 bytes ordinary and 600 to 1,048 bytes packed.
These are compiler reports, not measured executed spill traffic.

The BKV64 serving candidate passes the same correctness checks. Three-repeat
diagnostic median TTFT is 80.00/931.80 ms at 1K C1/C16 and 1,239.10/21,151.79 ms
at 16K C1/C16. The standalone gain does not establish a clear serving win versus
the previous candidate. The checked-in recipe retains BKV32; isolating the
attention bodies deserves measurement before increasing resource coupling.

## Lean HD512 packet role

The existing HD512 executor now has an opt-in native WGMMA Q64/KV32 body with
packed-request TMA support. It reuses the attention implementation and the
existing packet-role loader, dependency counters and graph execution. The
compiler reads the object's tile metadata, pins its SHA256, and restricts this
variant to fused output with one KV split. The default Q32/KV16 object remains
byte-identical to the prior source's build.

All nine Gemma programs retain exactly the same instructions and segment
windows. Each prefill ladder still has 435 launches; eight HD512 segments select
the lean object. Runtime logs confirm those selections and the unified token
batch route firing. No runtime launch-path changes were needed.

The actual-interpreter oracle now accepts `--interpreter CUBIN`; add
`--lean-hd512` for the dedicated role. Its ragged, reversed-slot, 16K-history
cases compare against the same independent FP64 reference and check completion
counters. The combined interpreter takes 2,499.04 us with maps versus
1,951.71 us for the final lean candidate. Both pass numerical checks. The lean
candidate passes CUDA memcheck and compiles with 238 registers, zero spill
loads/stores, and a 16-byte stack frame. This is one synthetic packet workload,
not complete ladder tuning.

Hardware profiling nevertheless observes 530,944 local-load sectors and 808,832
local-store sectors in the mapped case. Dynamically indexed stage-state arrays
use local memory even though the compiler reports zero register spills. Both
cases have zero measured LSU shared-memory bank conflicts.

Serving parity within the candidate, cancellation, slot reuse and context
rejection checks pass. The output-128 screen has one measured repeat after one
warmup, prefix caching disabled, physical batch 16 and queue capacity 128:

| Input | Concurrency | TTFT ms | Output tokens/s | TPOT ms |
|---:|---:|---:|---:|---:|
| 1,024 | 1 | 96.94 | 69.08 | 13.82 |
| 1,024 | 16 | 973.75 | 263.49 | 53.48 |
| 16,384 | 1 | 1,202.50 | 41.84 | 14.62 |
| 16,384 | 16 | 20,858.33 | 72.07 | 59.22 |

These are screening results, not an interleaved serving A/B or a vLLM win.
Model-quality qualification against an independent full-model reference remains
separate. Batched decode and the aggregate-prefill/per-request KV-ring coupling
remain major work items.

Build the opt-in role through `scripts/build_sm90a_cubin.sh` with
`PLOW_BUILD_SEG=1 PLOW_BUILD_FA512=1 PLOW_BUILD_PFATTN_WGMMA=1` inside
`nix develop`. Place `interp_sm90a_pfattn_hd512.cubin` beside the intended output
packet **before** running `plowc`; object presence triggers validated selection.
Keep the measured ordinary/packed GEMM and light-object set from the earlier
recipe. Exact role build defines, object hash, unchanged-window checks and
runtime route evidence are in `gemma4-12b-h100-data/lean-attention-route-check.json`
and `lean-attention-route.log`. Six compiler role tests pass.

### TMA stage state in registers

The cooperative WGMMA body now uses two bitmasks for the double buffer's barrier
parities and pending-copy flags. This preserves phase continuity across work
items and avoids addressable local arrays. In the lean role, compilation reports
240 registers, zero stack bytes and zero spills. Both profiled cases report zero
local-memory sectors and zero LSU shared-memory bank conflicts. The mapped
packet's event time is 1,921.73 us; this small improvement over the preceding
lean role does not explain the entire model-level gap.

Eight attention cases pass the independent oracle at the production grid size
and at eight blocks, where each block processes multiple work items and requests.
The eight-block HD512 packet run also passes CUDA memcheck. This checks phase
reuse across items as well as full-TMA/partial-copy transitions.

The final serving candidate passes the same parity, cancellation, slot-reuse and
context checks. With 128 output tokens and the same one-repeat screen:

| Input | Concurrency | TTFT ms | Output tokens/s | TPOT ms |
|---:|---:|---:|---:|---:|
| 1,024 | 1 | 86.93 | 69.34 | 13.85 |
| 1,024 | 16 | 965.31 | 263.81 | 53.47 |
| 16,384 | 1 | 1,198.33 | 41.81 | 14.66 |
| 16,384 | 16 | 19,741.80 | 74.54 | 60.52 |

These results remain below the goal. The hardware traffic removal is verified;
the serving difference needs repeated, interleaved measurement before attributing
its full size to the bitmask change. Raw before/after counters and checks are in
`attention-stage-state-check.json` and the `attention-bits-*` files.

## References and reproduction

Reference implementation inspected at DeepGEMM revision
`66081d4c9c7d7c44f13fea402e5b622aa0f409c2`, with CUTLASS
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`. The producer/consumer structure,
descriptor prefetch, tile scheduling and epilogue staging are useful techniques;
their dtype/scale contracts must not be substituted for Plow's.
[DeepGEMM source](https://github.com/deepseek-ai/DeepGEMM/tree/66081d4c9c7d7c44f13fea402e5b622aa0f409c2).

The PTX review covers tensor-map addressing, asynchronous-copy completion byte
counts, barrier phases, and proxy ordering. In particular, partial copies must
not arm a barrier for bytes that will never arrive. The existing guards and
memory-ordering protocol remain intact.
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

The native probe is `runtime/nvidia/experiments/gemma4_segment_gemm.cu`; compile
with system NVCC, C++17, `-gencode arch=compute_90a,code=sm_90a`, includes
`runtime/common` and `runtime/nvidia`, and `-lcuda`. Arguments are
`M N K bf16|fp8 INTERPRETER_CUBIN`; `fp8` means W8A8 here. Build baseline without
the two producer flags and candidate with both set to one. Pair each with the
correspondingly built production cubin. Run through `nix develop` with system
CUDA libraries, not Nix's CUDA stubs.

## Exported native BF16 decode object (2026-09-10)

`runtime/nvidia/gemv_sm90_transposed.cu` now exports four shape/tile entry points
and deterministic FP32 split reduction. The probe and object share the body in
`op_gemv_transposed.cuh`. The object does not depend on cuBLASLt or CUTLASS.
Runtime packet routing is now available through the native opt-in below. Its launch and scratch contract
is documented in `runtime/nvidia/gemv_sm90_transposed.md`.

Two 900-case screens passed sampled FP64 numerical checks, including 300 actual
driver-loaded object cases each, across B1/B2/B4/B8/B16 and ten shapes (nine
Gemma shapes plus N83/K136 tails). Transposed variants passed repeatability.
Each screen uses 11 timed cold-weight repetitions after warmup; the best tile
and split are selected from that screen, without an independent holdout.

Weighted sum of standalone projections, milliseconds:

| Batch | Native vector | Native padded object | Native XOR object | cuBLASLt |
|---:|---:|---:|---:|---:|
| 1 | 11.867 | 10.268 | 9.944 | 9.777 |
| 2 | 13.635 | 10.152 | 9.853 | 9.780 |
| 4 | 16.547 | 10.124 | 9.914 | 9.760 |
| 8 | 22.299 | 10.367 | 9.946 | 9.760 |
| 16 | 41.932 | 10.949 | 10.142 | 9.774 |

These sums do not model fused projections, interpreter resource coupling,
attention, counters or serving. Arithmetic need not be bit-exact to vector
GEMV or cuBLASLt. Independent full-model quality remains unqualified.

Padded projection entry points use 40–58 registers; XOR entries use 48. All
report zero stack and spills. One actual M16/N15360/K3840/BK256/split1 launch
was profiled per layout: shared-load bank conflicts 7350 padded / 6121 XOR;
shared-store conflicts and local load/store sectors zero for both. This does
not certify every shape or attribute all conflicts to one instruction.
XOR is opt-in: across all candidate variants it slightly regresses B1–B8's
geometric mean, while its per-shape winners improve the weighted totals.

Compute Sanitizer reports zero errors for padded B16 and XOR B1/B16 tail
cases, including empty splits. Rebuilding the default object after adding
the optional layout produces a byte-identical cubin.

Evidence: [shape selections and object hashes](gemma4-12b-h100-data/decode-tc-driver-analysis.json),
[padded screen](gemma4-12b-h100-data/decode-tc-driver-screen.csv),
[XOR screen](gemma4-12b-h100-data/decode-tc-swizzle-screen.csv),
[padded counters](gemma4-12b-h100-data/decode-tc-driver-ncu.csv),
[XOR counters](gemma4-12b-h100-data/decode-tc-swizzle-ncu.csv).

## Native runtime integration (2026-09-10)

`--emit-decode-native-tc` selects native role 8 for Gemma 4 BF16 SM90 TP1,
B1/B2/B4/B8/B16. The packet pins the external object's SHA256. The runtime
rejects unknown shapes or mismatched hash/ABI and shares the existing ordered
segment/graph machinery with one 7.5 MiB scratch allocation. It does not load
cuBLASLt. Narrower rungs inherit the widest rung's BK and split count to
preserve summation order; the standalone best-per-rung table above is therefore
not the exact serving selection. The tested object uses the XOR layout.

Validation: 101 GPU-runtime unit tests, five role-schema tests and four compiler
tests passed. The native packet passed 122 bit-exact full-logit snapshots
across rung changes, holes, slot reuse and continuation. A corrupted cubin was
rejected for hash/ABI mismatch. Serving verification passed cancellation,
ragged concurrency, output limits, recovery and context rejection.

Physical B16, queue concurrency up to 128, PF chunk 1024/budget 2048, context
20480, prefix cache disabled, 128 output tokens per request:

| Input | Concurrency | Median TTFT (ms) | Output tokens/s |
|---:|---:|---:|---:|
| 1024 | 1 | 91.73 | 66.92 |
| 1024 | 16 | 967.91 | 540.20 |
| 16384 | 1 | 1191.33 | 42.28 |
| 16384 | 16 | 21526.86 | 80.60 |
| 1024 | 128 | 13808.98 | 531.36 |
| 16384 | 128 | 96670.34 | 82.50 |

All 290 measured requests completed with 128 outputs/cache zero. C1/C16 used
one warmup and one measured repeat per cell; C128 used one measured repeat
on the warmed server with no additional warmup. This is screening, not an
interleaved repeated A/B. Against the preceding Plow screen, short C16 rises
from 263.81 to 540.20 tokens/s and long C16 from 74.54 to 80.60. Long TTFT
regresses. Recorded vLLM BF16 reaches 969.38 and 176.84 tokens/s respectively;
the performance goal remains unmet.

Final generated text matches the preceding Plow screen on 33/34 synthetic
prompts. One 16K/C16 prompt differs in a repetitive hyphen sequence, also
compared with the initial native prototype. The cause has not been isolated.
The full-logit rung test establishes internal consistency, not agreement with
HF or vLLM; independent model-quality qualification remains open.

Evidence: [C1/C16](gemma4-12b-h100-data/native-tc-final-output128.json),
[C128](gemma4-12b-h100-data/native-tc-final-c128-output128.json),
[serving verification](gemma4-12b-h100-data/native-tc-final-verify.log),
[rung logits](gemma4-12b-h100-data/native-tc-rung-gpu.log),
[corrupted-object rejection](gemma4-12b-h100-data/native-tc-badhash.log).

## Larger prefill chunks (2026-09-10)

The previous 8K-budget experiment still used 1K per-request chunks. New native
BF16 packets test actual 4K chunks at B16 and 8K chunks at B8. Both retain the
compiled KV-ring contract, context 20480, identical kernel objects and disabled
prefix caching. Every emitted rung passed the packet coverage audit.

Median across measured repeats; 128 output tokens per request:

| Packet / chunk | Input | Concurrency | Repeats | TTFT ms | Output tokens/s |
|---|---:|---:|---:|---:|---:|
| B8 / 8192 | 1024 | 1 | 3 | 80.95 | 77.39 |
| B8 / 8192 | 4096 | 1 | 3 | 263.23 | 68.88 |
| B8 / 8192 | 16384 | 1 | 3 | 1147.09 | 45.07 |
| B16 / 4096 | 1024 | 1 | 2 | 92.97 | 77.36 |
| B16 / 4096 | 1024 | 16 | 2 | 924.87 | 541.46 |
| B16 / 4096 | 16384 | 1 | 2 | 1136.85 | 42.10 |
| B16 / 4096 | 16384 | 16 | 2 | 16063.93 | 91.63 |
| B16 / 4096 | 1024 | 128 | 1 | 13737.43 | 537.66 |
| B16 / 4096 | 16384 | 128 | 1 | 87884.65 | 92.81 |

C1/C16 use one warmup per cell. C128 follows on the warmed B16 server with no
additional warmup. All measured requests have the requested output count and
zero cached tokens. These are sequential screens, not interleaved comparisons.
The two long C16 TTFTs vary substantially (18.63 and 13.50 seconds). Against
the prior native 2K-budget/1K-chunk screen, long C16 throughput improves from
80.60 to 91.63, and C128 from 82.50 to 92.81 tokens/s. The recorded vLLM long
C16 result is still 176.84 tokens/s; no performance-goal completion is claimed.

The B16 candidate passes serving consistency, cancellation, slot reuse and
context rejection. One repeated 16K/C16 synthetic prompt produces two hyphen
sequences across slots/repeats, matching the previously observed variation.
This remains unresolved and prevents treating the screen as model-quality
qualification. Larger chunks are not promoted to defaults.

The GPU rung gate now accepts `TEST_DECODE_RUNG_PROMPT_ROWS`. With 16384 and
the matching B16/4K widest-only reference packet, all 122 full-logit snapshots
pass bit-exact comparison across B1/2/4/8/16, holes, slot reuse and continuation.
This tests sequentially populated KV histories and decode-rung transitions;
it does not compare different packed-prefill schedules or independent HF logits.
[Long-context gate](gemma4-12b-h100-data/native-4k-long-rung-gpu.log).

Diagnostic CUDA event timing on the B16/4K packed route attributes one 16K,
one-output request to 530.9 ms GEMM, 405.0 ms attention, and 144.6 ms in the
combined light-op object. Each chunk has 193/48/194 segments respectively.
The eight HD512 attention segments each reach about 13.8 ms in the last chunk.
Event instrumentation replaces graph replay, so these numbers locate costs
but are not serving latency measurements. Attention and light-op specialization
remain necessary alongside GEMM tuning; chunk launch overhead is insufficient
to explain the gap.

Reproduce compilation with `PLOW_MAX_CHUNK=4096`,
`PLOW_DECODE_BATCH_LADDER=1,2,4,8,16`, `PLOW_EMIT_DECODE_NATIVE_TC=true`,
`PLOW_TMA_GEMM=true`, `PLOW_NO_GLU_FUSE=true`, `PLOW_UNISEG=0`,
`PLOW_SEG_CLASS_SLICE=1`, `PLOW_SEG_FA512=all`, `PLOW_SEG_PURE_GEMM=1`,
and the native object recipe above. Serve with `--pf-chunk 4096
--pf-interleave 4096`. For B8/8K, use chunk/budget 8192 and ladder 1,2,4,8.

Evidence: [B8 latency](gemma4-12b-h100-data/native-8k-latency.json),
[B16 screen](gemma4-12b-h100-data/native-4k-screen.json),
[C128](gemma4-12b-h100-data/native-4k-c128.json),
[serving checks](gemma4-12b-h100-data/native-4k-verify.log),
[segment timings](gemma4-12b-h100-data/native-4k-segment-profile.log).

## Packed schedules and attention stall investigation (2026-09-10)

The ignored `gpu_packed_prefill_schedule_logits` diagnostic compares 128
teacher-forced decode steps after 16K prompts. It varies chunks (4096/1024/256),
sparse and reversed slots, and homogeneous versus mixed prompts. Across six
native-decode configurations, all 2,304 checked full-vocabulary frames are
bit-exact. This narrows the serving variation beyond those packing cases.

A seventh pass uses `token_batch_step`, which computes decode rows through the
prefill body and compact terminal. All 512 checked frames differ from native
decode: first-frame maximum absolute logit difference 0.46484375, maximum
across the run 9.0771484375. The selected greedy tokens remain identical in
this teacher-forced run. This establishes a route-dependent numerical
difference, not the cause of the exact serving hyphen variation or independent
model accuracy. Do not label the diagnostic's successful execution an accuracy
gate. Run it with `TEST_DECODE_RUNG_ASSETS` and `TEST_PACKED_LOGITS_OUT` plus
the same CUDA/packed runtime flags as the serving recipe.
[Summary and packet hash](gemma4-12b-h100-data/packed-schedule-logits-summary.json),
[execution log](gemma4-12b-h100-data/packed-schedule-logits.log).

The packet attention oracle now accepts compile-time `PLOW_TEST_FA_ROWS` for
a single full-context request; zero preserves the existing ragged cases.
Build with `-gencode arch=compute_90a,code=sm_90a -DPLOW_TEST_FA_ROWS=4096`
and run `--interpreter OBJECT --lean-hd512`. This reproduces the 4096-query,
16384-KV, HD512, Q16/KV1 shape seen in the final prefill chunk. Both mapped
and unmapped cases pass 4,608 sampled FP64 reference outputs, with worst
relative L2 0.00239514 and maximum absolute error 0.0000610031.

Two native attention experiments were rejected. Skipping unit output-rescale
factors showed no clear gain on the ragged workload. Computing scores and
softmax only in WG0, then sharing row corrections/normalizers with WG1,
regressed mapped actual-shape timing from 13.301 to 15.732 ms. The latter uses
243 registers/528 static shared bytes vs 240/16, with zero stack/spills for
both. It also regresses the ragged case. No production kernel change is
retained. The default rebuild was byte-identical to the deployed object.
The conditional WGMMA experiment used a condition uniform within each
warpgroup, and retained proxy ordering, consistent with the
[PTX WGMMA requirements](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
[Experiment hashes and decision](gemma4-12b-h100-data/attention-single-score-experiment.json),
[baseline oracle](gemma4-12b-h100-data/attention-4k-0-oracle.log),
[candidate oracle](gemma4-12b-h100-data/attention-4k-1-oracle.log).

For the deployed mapped kernel on this actual shape, Nsight Compute reports
13.599 ms, tensor-pipe active 22.65%, DRAM throughput 0.81%, and active-warp
stall proportions: fixed-latency wait 34.07%, GMMA 11.41%, barrier 11.11%,
math-pipe throttle 2.42%, long scoreboard 2.02%, short scoreboard 1.81%.
These are one profiled launch and distinct metric denominators; do not add
them or extrapolate them to every ladder. They favor investigating dependent
instruction chains and pipeline overlap over assuming HBM saturation.
[Hardware counters](gemma4-12b-h100-data/attention-4k-stalls.csv),
[profile correctness](gemma4-12b-h100-data/attention-4k-stalls.log).

## HD512 QK instruction unrolling

The cooperative native HD512 WGMMA body now accepts `PLOW_NV_FA_QK_UNROLL`.
The shared header defaults to 1; the dedicated SM90 HD512 WGMMA build selects
32, overridable with `PLOW_BUILD_PFATTN_QK_UNROLL`. This exposes constant
descriptor offsets across the 32 QK k16 steps. Tile geometry, arithmetic order,
barriers, shared-memory layout and interpreter integration are unchanged.
No external kernel library is linked. The selected object uses 224 registers
versus 240, with zero stack/spills in both variants.

Sequential warm screens at KV history ending at 16384, Q16/KV1, BF16, mapped
staging, direct interpreter body:

| Query rows | Unroll 1 (us) | Unroll 32 (us) |
|---:|---:|---:|
| 128 | 1943.648 | 1450.784 |
| 512 | 1947.648 | 1449.504 |
| 1024 | 3772.448 | 2837.568 |
| 2048 | 7248.352 | 5479.616 |
| 4096 | 13273.536 | 10119.775 |
| 8192 | 22707.359 | 16954.848 |

Every size passes sampled FP64 checks, and all output bytes match baseline
for both mapped and unmapped staging. Ragged multi-request checks also pass;
the one-block multi-item memory check reports zero errors. The probe's new
`--snapshot PREFIX` option was used with `--lean-hd512` for full-output
comparisons. These are kernel screens, not randomized serving A/B results or
qualification of every attention history.
[Unroll/resource evidence](gemma4-12b-h100-data/attention-qk-unroll.json),
[rung checks](gemma4-12b-h100-data/attention-qku32-rungs.json),
[memory check](gemma4-12b-h100-data/attention-qku32-memcheck.log).

The candidate packet retains the baseline 4K packet's instructions and queue
windows, with the new HD512 object hash. Serving verification passes. Two-repeat
16K/C16 throughput is 96.238 tokens/s; a warmed single-repeat C128 screen is
96.997 tokens/s. C128 remains queue concurrency with physical B16.
[Serving screen](gemma4-12b-h100-data/native-qku32-screen.json),
[C128](gemma4-12b-h100-data/native-qku32-c128.json),
[verification](gemma4-12b-h100-data/native-qku32-verify.log).

C1 results are confounded by an active-batch `Admit::Defer` sleep in the
scheduler: a baseline run after burst conditioning measured about 22.50 ms
TPOT, while baseline with `--max-hold-ms 0` measured about 13.38 ms. Do not
attribute that entire difference to this kernel. The scheduler fix and fresh
serving qualification remain open, as do all-op specialization and the vLLM
performance objective.

[Conditioned C1 screens](gemma4-12b-h100-data/active-hold-latency-screen.json)
retain the raw-file hashes and measured repeat summaries.

## Active-batch admission delay removed

The mux no longer sleeps on `Admit::Defer` after admitting live requests.
Batch formation still waits on idle ingress; overload shedding is unchanged.
Arrival-rate history can outlive a burst, so applying a formation hold to each
active decode tick inserted about 9 ms/token (8 ms plus timer scheduling).

With the default `max_hold_ms=8`, the baseline kernel now measures median
16K/C1 TPOT 13.395 ms, matching the earlier hold-disabled control (13.384 ms)
instead of the earlier conditioned 22.50 ms result. Fresh baseline and QK32
servers ran the same serving verification sequence, then three measured C1
repeats after one warmup per input:

| Kernel | Input | TTFT (ms) | TPOT (ms) | Output tokens/s |
|---|---:|---:|---:|---:|
| Baseline | 1024 | 85.111 | 12.427 | 76.970 |
| QK32 | 1024 | 80.727 | 12.359 | 77.547 |
| Baseline | 16384 | 1140.701 | 13.395 | 45.031 |
| QK32 | 16384 | 1073.533 | 13.263 | 46.422 |

The fixed QK32 throughput screen gives 566.935/562.404 tokens/s at 1K C16/C128
and 94.162/99.759 at 16K C16/C128. This is one repeat per cell after serving
verification, without additional per-case warmups; C128 uses physical B16.
All 300 measured requests across latency and throughput completed 128 tokens
with zero cached tokens. Serving cancellation, slot reuse, ragged prompts and
context rejection pass for all three fresh server runs. Mux20+scheduler37
unit tests pass; the CUDA release build succeeds.

[Latency evidence](gemma4-12b-h100-data/active-hold-fixed-latency.json),
[throughput evidence](gemma4-12b-h100-data/active-hold-fixed-throughput.json),
[unit tests](gemma4-12b-h100-data/active-hold-tests.log).
This removes a scheduler latency defect; it does not close the vLLM gap.
Recorded vLLM 16K C1/C128 throughput remains 64.692/196.035 tokens/s.

## Packed light-object occupancy experiment

The current packed light-op interpreter uses 255 registers and a 1952-byte
stack frame. Rebuilding its original flags from current source gives 255
registers and an 1880-byte frame; the binaries are not identical. The comparison
below therefore uses the fresh rebuild as its control. All other cubins,
including native projections and the selected HD512 object, remain fixed.

The existing `PLOW_NV_FATLITE=1` with `PGM90_TMA_STAGES=3` strips prefill
attention and allows two resident blocks/SM. It uses 128 registers, but its
stack frame increases to 2360 bytes. Only the packed light object is replaced;
ordinary prefill and decode objects are unchanged.

The paired packet uses existing `PLOW_SEG_SLICE_ALL=1`, doubling eligible
light-op slices from 132 to 264. The 128-row rung changes HeadNormRope and GLU;
512/1024/2048/4096 additionally change embedding, RMSNorm and NormResidual.
Other instruction fields and all decode instructions are unchanged. All queue
slices and dependencies pass the packet audit.

Sequential serving screens, two measured repeats after one warmup per cell,
BF16, physical B16, cache disabled, output128:

| Variant | 1K/C1 tok/s | 1K/C16 tok/s | 16K/C1 tok/s | 16K/C16 tok/s | 16K/C1 TTFT ms |
|---|---:|---:|---:|---:|---:|
| Fresh control | 77.212 | 566.621 | 46.166 | 96.105 | 1076.828 |
| Lean object only | 76.745 | 566.874 | 46.101 | 94.449 | 1080.080 |
| Lean object + doubled slices | 77.064 | 574.601 | 46.644 | 98.551 | 1047.097 |

All 204 measured requests complete128/cache0, and serving cancellation,
ragged prompts, slot reuse and context rejection checks pass. Both candidates
match 66/68 control outputs; the two differences occur on the previously
unstable repeated-hyphen prompt at 16K/C16. This is not independent model-quality
qualification. The object-only variant offers no clear gain.
[Screens and object hashes](gemma4-12b-h100-data/fatlite-screens.json).

For one diagnostic 16K request, summed light-segment time falls from
146.3 to 95.5 ms with doubled slices. GEMM is 530.5 vs 533.3 ms; attention is
342.2 vs 347.1 ms. Event profiling replaces graphs, so use these measurements
for attribution rather than serving latency. The reduction is real in this
diagnostic, but is only about a 2.5% C16 serving throughput gain in the screen.
[Control profile](gemma4-12b-h100-data/fatlite-profile-control.log),
[candidate profile](gemma4-12b-h100-data/fatlite-profile-slices.log).

The ignored packed-schedule diagnostic now records a SHA256 of every full
logit frame. This permits comparisons between separately loaded objects in
addition to its existing comparisons between schedules within one object.

The control and sliced candidate match all **2816 full-logit frame hashes**
across seven diagnostic passes, including the unified route. Each frame covers
262144 logits. This isolates the tested object/slice changes from the known
ordinary-vs-unified numerical difference; it does not explain every serving
interleaving or qualify the two text differences above.
[Full-logit comparison](gemma4-12b-h100-data/fatlite-logits-comparison.json).

Reproduce the packed object with
`PLOW_BUILD_FATLITE=1 scripts/build_sm90a_gemma4_segments.sh BASE OUTPUT` inside
`nix develop`; compile the paired packet with `PLOW_SEG_SLICE_ALL=1` alongside
the existing pure-GEMM/all-attention segmentation options. The new option
replaces only the packed light object in addition to the script's existing
GEMM/attention builds. Its default is off. The measurements here retained all
other qualified objects and the packet-pinned QK32 object. C128 has not yet
been screened with this candidate, and no all-op optimality claim is made.

## C128 follow-up and further light-object limits

The paired FATLITE/doubled-slice configuration now completes the C128 screen:
569.525 tokens/s at 1K input and 101.845 at 16K, versus the earlier unsliced
fixed-scheduler screens of 562.404/99.759. These are sequential single-repeat
screens, not a statistical A/B. The fresh server ran serving verification
first, with no additional per-case warmup. All 256 requests completed 128
tokens with zero cached tokens; physical batch remains 16.
[C128 evidence](gemma4-12b-h100-data/fatlite-c128.json),
[serving verification](gemma4-12b-h100-data/fatlite-c128-verify.log).

A further experimental packed light-only object retained the native op bodies
and queue/counter protocol, restricted supported opcodes, and reduced dynamic
scratch to 64 bytes. Its initial allow-list omitted `Nop` and trapped during
serving verification even with the original scratch size restored. The runtime
rewrites terminal instructions to `Nop`; adding that opcode restored serving
checks with the small scratch allocation. Static emitted-op coverage alone
therefore does not establish the complete runtime kernel contract.

Restricting the dispatch without explicitly removing heavy cases retained
128 registers and a 2360-byte stack. Explicitly compiling out unused GEMM,
MoE and extra normalization cases reduced this to 107 registers, zero stack
and 64 bytes of dynamic scratch. A build-only `RN_REG=16` variant instead used
113 registers and was not benchmarked. No larger occupancy was gained.

| Object | 16K/C1 tok/s | 16K/C1 TTFT ms | 16K/C16 tok/s |
|---|---:|---:|---:|
| Retained FATLITE + doubled slices | 46.644 | 1047.097 | 98.551 |
| Dispatch restriction + small scratch | 46.476 | 1055.910 | 96.445 |
| Explicitly stripped light object | 46.802 | 1037.441 | 98.598 |

Both working prototypes passed serving checks; their 136 measured requests
completed 128 tokens/cache0. These are two-repeat screens after one warmup per
cell, and full-logit equivalence was not qualified for the new prototypes.
The stripped object's C16 repeats varied from 99.954 to 97.243 tokens/s.
There is no clear C16 improvement over the retained configuration in this
screen, despite the smaller footprint. Experimental production-source changes
were removed, with the patch and cubins retained in ignored plans.
[Experiment evidence](gemma4-12b-h100-data/lightonly-experiment.json).

A fresh event-profile comparison gives FATLITE vs stripped summed light time
95.5 vs 79.1 ms, GEMM 532.7 vs 537.7 ms, and attention 348.8 vs 351.7 ms.
This demonstrates a smaller light-body cost, not an end-to-end throughput win.
GEMM and attention dominate the remaining prefill work; further optimization
should prioritize them and batching capacity. The vLLM objective remains unmet.
[Control profile](gemma4-12b-h100-data/lightonly-profile-control.log),
[stripped profile](gemma4-12b-h100-data/lightonly-profile-slices.log).

## B32 native projections and LM head

The opt-in native route now includes all 329 BF16 decode projections at
B1/2/4/8/16/32, including the tied vocabulary head. It keeps residual/norm
fusion and uses the existing ordered segment graph route. B32 adds native
RM32/BK128/stages3 and RM32/BK256/stages2 bodies with 66/64 registers and no
stack or spills. These use `cp.async` and `mma.sync`, not TMA/WGMMA. The main
decode interpreter still uses 198 registers with no stack/spills; projection
objects have independent resource footprints.

The B32 padded-layout screen passes 150 numerical cells, including 60 actual
driver-entry cases over eight layer shapes, the LM head and an N83/K136 tail.
Checks include 128 sampled FP64 dots per case, full-output finiteness and
repeat determinism. The padded tail passes Compute Sanitizer with zero errors.
Weighted cold-layer time at the selected cells is 11.550 ms native vs
9.175 ms cuBLASLt and 75.124 ms vector. B32 LM-head time is 739 us native vs
664 us cuBLASLt and 5,331 us vector. These are standalone screens, not actual
graph timings or held-out tuning measurements. The experimental XOR layout
reaches 9.757 ms at its best layer cells but is not integrated or tail-memcheck
qualified. [Padded screen](gemma4-12b-h100-data/b32-driver-screen.csv),
[XOR screen](gemma4-12b-h100-data/b32-xor-screen.csv),
[tail memcheck](gemma4-12b-h100-data/b32-tail-memcheck.log).

B32 exposed a normalization correctness issue: the row-count heuristic
switched decode to the prefill reduction order. Keeping block reduction in
the decode object fixes the observed full-logit rung mismatch. The runtime
requires the new marker and rejects old main objects for native B32. Both
layer-only and native-head variants pass 170 full-logit snapshots at each of
128 and 16,384 prompt rows, including resets and sparse high slots. These
compare narrow execution against the same native widest-rung implementation;
they do not establish vector-head or independent HF/vLLM equivalence.

Sequential two-repeat serving screens use physical B32, queue128,
context20480, aggregate prefill8192/request1024, cache disabled and 128 output
tokens, with one warmup per cell. Both variants pass serving verification.
All 264 measured requests complete 128 tokens with cache0.

| Input / concurrency | Vector head tok/s | Native head tok/s |
|---|---:|---:|
| 1K / C1 | 75.825 | 76.107 |
| 1K / C32 | 693.261 | 779.734 |
| 16K / C1 | 45.233 | 45.180 |
| 16K / C32 | 100.648 | 103.575 |

At 1K/C32 median TPOT falls from 32.506 to 27.558 ms. Paired output text
matches for 129/132 requests; three 16K/C32 hyphen continuations differ.
This screen does not isolate their cause. The configuration remains opt-in;
long-prefill throughput and complete all-op performance qualification remain
open. [Artifact hashes, checks and results](gemma4-12b-h100-data/b32-native-screen.json).

The native-head C128 follow-up completes 256/256 requests with 128 tokens and
cache0: **763.294 tokens/s at 1K and 101.940 at 16K**. Serving verification
passes before this single-repeat screen; no additional per-cell warmup is
used. The retained B16 screens were 569.525/101.845 tokens/s. This improves
short-context throughput, with no meaningful long-context gain established.
Historical vLLM BF16 C128 results are 2,131.578/196.035 tokens/s; the objective
remains unmet. These are historical sequential comparisons, not a fresh
interleaved vLLM A/B. [C128 raw results](gemma4-12b-h100-data/b32-screen-head-c128.json),
[serving checks](gemma4-12b-h100-data/b32-screen-head-c128-verify.log).

## Shared-memory GEMM epilogue screen

`PLOW_BUILD_GEMM_SMEPI=1` in `scripts/build_sm90a_gemma4_segments.sh` enables
the existing native `PGM90_WS384_SMEPI` epilogue in ordinary and packed GEMM
objects. Default remains off. The consumers stage BF16 results in the drained
shared-memory ring before coalesced 16-byte global stores; consumer barriers
and delayed empty-slot arrivals protect the reused storage. TMA loading,
WGMMA math, packet dependencies and segment routing remain the same. The
candidate uses 160 registers with zero stack/spills, like the control.

An actual-interpreter BF16 screen covers all 49 emitted GEMM M/N/K shapes:
eight projection shapes at M128/512/1024/2048/4096/8192 and the M1 LM head.
Each object/shape runs standalone-body, segment-with-dependencies and
queue-without-dependencies modes. All 294 mode checks pass 257 sampled FP64
dots, full-output finiteness and expected counter completion. The body
executable is unchanged control code; only segment/queue measurements compare
the two epilogues. This is one warm-cache campaign with alternating object
order, not held-out tuning or a serving measurement.

| Prefill rung | Weighted control GEMM ms | Staged epilogue ms | Speedup |
|---:|---:|---:|---:|
| 128 | 24.703 | 24.184 | 1.021× |
| 512 | 29.104 | 27.742 | 1.049× |
| 1024 | 39.791 | 36.295 | 1.096× |
| 2048 | 71.860 | 64.868 | 1.108× |
| 4096 | 139.275 | 122.460 | 1.137× |
| 8192 | 279.294 | 242.401 | 1.152× |

Weights are emitted instruction counts, not measured graph execution. Removing
dependency edges changes weighted large-rung GEMM timing by roughly 0–1% in
this isolated independent-tile test. It does not establish counter cost for
the complete interpreter or every op. M128/N3840/K15360 passes Compute
Sanitizer memcheck with zero errors. This does not qualify FP8 or arbitrary
tail shapes for the staged epilogue.

Fresh sequential serving screens keep the B32 native-head packet, attention,
light objects and scheduler fixed. Both variants pass serving verification;
all 264 measured requests return 128 tokens/cache0, and all 132 paired output
texts match. Two repeats follow one warmup per cell. Physical batch32,
queue128, context20480, request chunk1024 and aggregate budget8192.

| Input / concurrency | Control tok/s | Staged epilogue tok/s |
|---|---:|---:|
| 1K / C1 | 75.564 | 75.868 |
| 1K / C32 | 781.461 | 792.103 |
| 16K / C1 | 45.247 | 46.038 |
| 16K / C32 | 104.374 | 108.485 |

16K/C1 median TTFT improves from 1112.450 to 1070.736 ms. These are modest
serving gains; no fresh vLLM comparison was run for this candidate. The vLLM
goal and all-op performance qualification remain open.
[Screen, artifacts and limits](gemma4-12b-h100-data/gemm-smepi-summary.json),
[raw shape runs](gemma4-12b-h100-data/gemm-smepi-screen.jsonl),
[memcheck](gemma4-12b-h100-data/gemm-smepi-memcheck.log).

The probe's optional trailing `--exact-body` argument additionally compares
every output byte of segment and queue modes against its unchanged standalone
body. All 49 candidate shapes pass this full-output gate as well as sampled
FP64 checks. This covers the ordinary GEMM object; packed execution is covered
by the serving checks above. [Full-output evidence](gemma4-12b-h100-data/gemm-smepi-exact.jsonl).

C128 follow-up passes serving verification and completes all 256 requests
with 128 tokens/cache0: **784.967 tokens/s at 1K and 105.396 at 16K**, versus
the previous native-head screen's 763.294/101.940. This is one repeat after
verification with no extra per-case warmup, not an interleaved A/B. Historical
vLLM BF16 C128 results remain substantially ahead at 2,131.578/196.035 tokens/s.
[C128 raw results](gemma4-12b-h100-data/gemm-smepi-head-c128.json),
[serving checks](gemma4-12b-h100-data/gemm-smepi-head-c128-verify.log).

## HD512 KV64 single-stage screen

The native BF16 SM90 HD512 role now supports Q64/KV64 with one K/V staging
slot. Both warpgroups finish their asynchronous matrix reads before a CTA
barrier permits overwriting the slot. It retains TMA and 128-byte swizzles,
and unrolls the four P×V steps. This trades load/compute overlap for half as
many online-softmax tiles. The dedicated object uses 255 registers, no stack
or spills, and 205,824 bytes of dynamic shared memory. No CUTLASS or DeepGEMM
kernel is linked.

Build opt-in with `PLOW_BUILD_MASKED_PADDING=1 PLOW_BUILD_PFATTN_KV64=1`
using `scripts/build_sm90a_gemma4_segments.sh`. Copy the resulting HD512
object into the asset directory before compiling so the packet pins its
hash and geometry. `PLOW_BUILD_GEMM_SMEPI=1` selects the GEMM epilogue used
by both serving variants below. KV32 remains the default.

| Query rows | KV32 TMA µs | KV64 TMA µs |
|---:|---:|---:|
| 128 | 1438.464 | 1365.760 |
| 512 | 1439.360 | 1379.520 |
| 1024 | 2835.648 | 2645.984 |
| 2048 | 5432.288 | 5046.016 |
| 4096 | 9888.544 | 9154.145 |
| 8192 | 17166.113 | 15521.248 |

These load the actual dedicated role, at 16K history, 16 query heads and one
KV head. All 24 mapped/unmapped object/rung cases pass sampled FP64 checks.
Ragged role execution passes memcheck and racecheck with zero errors or
hazards. These checks do not measure bank conflicts. Full-model narrow vs
widest KV64 execution passes 170 logit snapshots at each of 128 and 16,384
prompt rows; this is rung consistency, not an independent model-quality test.

Fresh sequential serving uses physical B32, queue128, context20480,
request chunk1024, aggregate8192 and 128 outputs. C1/C32 have two repeats
after one warmup. Both variants pass serving verification, and all 264
measured requests return 128 tokens/cache0.

| Input / concurrency | KV32 tok/s | KV64 tok/s |
|---|---:|---:|
| 1K / C1 | 75.712 | 76.130 |
| 1K / C32 | 789.934 | 793.581 |
| 16K / C1 | 45.799 | 46.087 |
| 16K / C32 | 109.692 | 108.906 |

16K/C1 median TTFT is 1069.116→1056.092 ms. Only 129/132 paired output texts
match: three 16K/C32 requests differ in repeated digit or punctuation output.
The cause is not established; the changed softmax grouping can alter BF16
rounding. Keep KV64 opt-in pending broader numerical and serving qualification.

The C128 follow-up returns 128 tokens/cache0 for all 256 requests, at
782.164/109.078 tokens/s for 1K/16K. The preceding KV32 screen measured
784.967/105.396. These are single sequential screens, not an interleaved A/B.
Historical vLLM C128 remains ahead at 2131.578/196.035 tokens/s.

A separate 16K/C1 event profile attributes total attention time to
349.4→335.9 ms, GEMM to 527.1→528.0 ms and light ops to 137.1→137.3 ms.
Attention includes all 40 HD256 and eight HD512 layers per chunk. Event
profiling replaces graph execution, so these are diagnostic attribution
times rather than normal serving latency.
[Raw results, hashes, packet inventory and limits](gemma4-12b-h100-data/attention-kv64-summary.json),
[actual-role rung checks](gemma4-12b-h100-data/attention-kv64-rungs.jsonl).

## HD512 fixed head geometry

The dedicated role specializes 16 query heads, one KV head, no sliding window
and one split, passing those constants into the existing native body. Other
head geometries keep the generic path. Query length, history, request slots,
stride, mask and scale remain dynamic. This preserves the KV32 softmax tile
and its TMA pipeline. The object uses 224 registers and zero stack/spills.
Enable with `PLOW_BUILD_MASKED_PADDING=1 PLOW_BUILD_PFATTN_FIXED_HEADS=1` in
the segment build script; copy the role object into the asset directory
before compilation. Default remains off because serving results are mixed.
This screen uses KV32; combining fixed heads with KV64 is not qualified here.

| Query rows | Control TMA µs | Fixed geometry TMA µs |
|---:|---:|---:|
| 128 | 1437.824 | 1280.480 |
| 512 | 1437.952 | 1271.808 |
| 1024 | 2794.304 | 2527.520 |
| 2048 | 5447.680 | 4841.376 |
| 4096 | 10042.752 | 9035.937 |
| 8192 | 16972.863 | 15568.032 |

Actual-role runs alternate object order across rungs. All 24 mapped/unmapped
cases pass sampled FP64 checks at history ending at 16K; every output byte
matches the control at each rung. Ragged memcheck reports zero errors.
These are isolated warm-cache timings, not full-model throughput.

A preceding experiment computed QK/softmax only in warpgroup0 and shared row
statistics through the drained K slot. It passed the sampled oracle but
slowed the 1024-row mapped case to 4074.432 µs versus a fresh 2813.920 µs
control. The compiler warned of WGMMA serialization around the divergent
path. That implementation was rejected and is absent from production code.

Fresh sequential serving keeps the B32 native-head packet settings and SMEPI
GEMM objects fixed: context20480, request chunk1024, aggregate8192, output128,
two repeats after one warmup per cell. Both variants pass serving verification;
all 264 requests return 128 tokens/cache0 and all 132 paired texts match.

| Input / concurrency | Control tok/s | Fixed geometry tok/s |
|---|---:|---:|
| 1K / C1 | 75.123 | 75.184 |
| 1K / C32 | 795.742 | 797.126 |
| 16K / C1 | 45.844 | 46.086 |
| 16K / C32 | 113.832 | 110.117 |

16K/C1 median TTFT improves 1062.827→1053.988 ms. C32 throughput declines
in this screen despite faster isolated attention. This does not establish
the cause of the serving difference or a throughput win. No fresh C128 or
vLLM comparison was run for this candidate. The goal remains unmet.
[Evidence and artifact hashes](gemma4-12b-h100-data/attention-exactshape-summary.json),
[full-output rung checks](gemma4-12b-h100-data/attention-exactshape-rungs.jsonl).

## Rejected Hopper FP8 probability/value variants

Native Hopper adaptations of the FP8 P×V path were screened against the
current FP8-KV packed attention object. Single-term E4M3 probabilities failed
HD512/rows128 at relative L2 0.226483 (unchanged gate 0.015). A second E4M3
residual term recovered accuracy in all 12 HD256/HD512 rung cases. Both a
scalar shared-byte transpose and a 16-bit ldmatrix/register-byte transpose
passed, but neither improved HD512 latency.

| Rows | Current FP8 KV (µs) | Scalar residual (µs) | Transpose residual (µs) |
|---:|---:|---:|---:|
| 128 | 3527.392 | 4451.872 | 3928.896 |
| 512 | 3350.336 | 4368.960 | 3943.648 |
| 1024 | 6622.176 | 8630.976 | 7651.168 |
| 2048 | 12960.512 | 16749.088 | 14901.439 |
| 4096 | 12838.016 | 16763.328 | 15032.353 |
| 8192 | 12945.472 | 16729.695 | 14964.063 |

These are sequential warm-cache screens: median of nine CUDA-event samples
after an initial launch per case. The loaded packed interpreter executes the
ready dependency and successor counter path; queue/counter resets are outside
the timed region. Two ragged requests end at history 16384/8193, with at most
1024 real rows per request. The 4096/8192 rungs each contain only 2043 real
rows. These measurements are neither an interleaved tuning run nor serving
latency. No sanitizer or serving qualification was run for these rejected
variants.

Kernel edits were reverted. The probe retains optional `CUBIN --timing` mode
for subsequent native candidates. [Evidence and hashes](gemma4-12b-h100-data/fp8pv-hopper-summary.json)
include numerical/timing logs, compiler resource reports and the rejected
residual patches. Whole-entry register/spill reports include other dispatch
arms and must not be attributed entirely to the executed FP8 path.
