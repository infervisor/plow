# Gemma 4 12B packet ladder coverage

Target: `google/gemma-4-12B-it`, H100 `sm_90a`. Compilation coverage is not
performance qualification. Native bodies can remain in a shared interpreter;
a separate object per opcode is not a requirement.

## Per-case dispatch-arm validation

`packet_ladder_audit` now recomputes each case's dispatch arm from the actual
packet opcode and immediates, using the compiler's shared arm selector. With
`build.json` supplied, it rejects missing or mismatched arm declarations,
including a wrong head dimension. The audit output includes `dispatch_arm`
beside the exact parameters, PCs and declared segment roles.

The current BF16 B32, BF16-weight/FP8-KV B32, W8A8 B16 and fused-quant W8A8
B16 assets pass: **42 programs, 31,116 instructions, 9,276 cases**.
The [per-rung report](gemma4-12b-h100-data/ladder-dispatch-arm-audit.json)
records packet/build hashes, all arm names and opcode counts.

This closes an audit gap: previously an incorrect `arm` label could pass
even though the raw instruction fields were checked. It does not add kernel
specializations. Loaded-object selection, dtype/build flags, numerical checks
and per-case timing remain separate requirements. In particular, shared light
ops and FP8 decode still prevent an all-ops-specialized performance claim.

## Current all-op specialization review

The current BF16-weight B32 packets with BF16 KV and all-layer FP8 KV both
pass the strict packet/build/queue audit: **24 programs, 17,808 instructions,
5,472 exact cases** combined. Each has prefill rungs
128/512/1024/2048/4096/8192 and decode rungs 1/2/4/8/16/32.
The [source review and per-rung inventory](gemma4-12b-h100-data/current-all-op-specialization-review.json)
records packet/build hashes, declared precision, all 17 emitted families
across the two configurations, immediate shapes, role IDs, source hashes and
remaining work. Every emitted family has a review entry; none is omitted
because it is a light or fused op.

**All-op specialization is still incomplete.** RMSNorm, residual norms, GLU,
embedding, softcap and argmax use shared bodies with runtime dimensions.
Projection and attention templates do not establish that every exact case is
fast. Preserve useful fusion; qualify replacements using loaded segment
correctness, timing and serving results before promotion. This source review
does not resolve runtime overrides or qualify AMD/CPU implementations.

FP8 HD512 attention remains a measured performance gap. Two native Hopper
FP8 P×V residual variants passed the 12-case numerical probe but were slower
at every tested HD512 rung and were reverted. The single-term variant failed
the unchanged numerical gate. See the
[experiment evidence](gemma4-12b-h100-data/fp8pv-hopper-summary.json).

Latest B32 candidate: see [B32 all-op cases](#b32-all-op-cases) below. Earlier
sections describe retained historical packets, including their then-open gaps.

The BF16 8K aggregate-budget packet has prefill rungs 128, 512, 1024, 2048,
4096, 8192 and decode rungs 1, 2, 4, 8. Each prefill rung emits 766
instructions; each decode rung emits 542. Every compile now inventories the
exact emitted cases in `build.json.kernel_cases`. The older W8A16 artifact
predates that field and must be audited from its packet instead.

| Emitted family | Instructions per prefill / decode rung | Existing specialization; remaining qualification |
|---|---:|---|
| Gemm | 329 / 0 | Native SM90 WGMMA/TMA tile variants. Measured projection cases exist; all rung selections are not qualified. |
| Gemv, GemvQkv, GemvGlu | 0 / 113 + 40 + 48 | Fused decode bodies. Opt-in native tensor-core routing now covers 328 unfused layer projections at B1/2/4/8/16; LM head remains in the interpreter. |
| FlashPrefill | 48 / 0 | HD256 ×40, HD512 ×8. Packed descriptor support; dedicated HD512 Q64/KV32 role. Ragged correctness and selected hardware counters measured, not every rung/history. |
| FlashDecode, FlashMerge | 0 / 48 + 48 | HD256/512 templates; full per-batch/history tuning remains open. |
| HeadNormRope | 144 / 144 | HD256 ×120, HD512 ×24; dimension-specific templates fuse norm, rotary and cache writes. Per-rung timing and resource qualification remain open. |
| RmsNorm | 97 / 1 | Shared body with runtime row/width parameters. Exact-shape tuning remains open. |
| NormResidual, NormResidualNorm | 96 / 96 | Gemma sandwich residual and fused next normalization. Exact-shape tuning remains open. |
| Glu | 48 / 0 | Shared pointwise body; decode uses fused GemvGlu. Prefill tuning remains open. |
| Embed, SoftCap, Argmax, ArgmaxFin | 1 each / 1 each | Existing shared bodies. Per-rung tuning remains open. |

Source dispatch: `runtime/nvidia/interp_sm120.cu` (also included by SM90).
Existing measurements and their limits are in
[the native tuning report](gemma4-12b-h100-native-tuning.md).

Run `cargo run -p plowrt --example packet_ladder_audit -- model.pkt audit.json`
inside `nix develop`. The audit links every instruction to its queue windows
and each window to its declared role/object, including pinned hashes when
present. It rejects malformed role metadata or mismatched role counts.
`runtime_selected` means the packet did not bind a role; it does not mean
the runtime uses a generic kernel. `interpreter` can carry specialized and
fused bodies. Loaded object identity and runtime overrides still need launch
evidence.

Acceptance remains: exact arch/dtype/shape support, numerical correctness,
actual interpreter/segment timing, resource counters where relevant, and a
serving comparison. No blanket all-op tuning claim is supported yet. AMD and
CPU need independent backend qualification; H100 results do not qualify them.

## Verification

The audit built with `cargo build -p plowrt --example packet_ladder_audit`.
BF16 B16, BF16 B8/8K, and W8A16 packets passed bidirectional instruction/window
coverage checks: 28 programs, 18,712 instructions. Both newer BF16 manifests
also matched the packet's immediate fields, float bits and block counts.
A packet with a shortened role vector was rejected. The
[coverage snapshot](gemma4-12b-h100-data/packet-role-coverage.json) records packet
hashes, precision and per-rung counts. Each BF16 prefill rung declares eight
HD512 native-object windows and 427 interpreter windows; runtime routing can
further select GEMM/attention objects inside the latter.

## Aggregate-budget screening

Same validated BF16 kernel recipe; physical B8, queue limit 128, per-request
prefill chunk 1024, aggregate budget 8192, context 20480, cache disabled.
One warmup and one measured repeat per cell, 128 output tokens per request.
All 290 measured requests completed with the requested output count. This is
a screening run, not an interleaved A/B or independent model-quality test.

| Input | Concurrency | Median TTFT (ms) | Output tokens/s |
|---:|---:|---:|---:|
| 1024 | 1 | 78.80 | 69.20 |
| 1024 | 16 | 2526.43 | 244.78 |
| 1024 | 128 | 34633.01 | 225.41 |
| 16384 | 1 | 1204.57 | 41.55 |
| 16384 | 16 | 14548.36 | 65.52 |
| 16384 | 128 | 131456.43 | 62.28 |

Do not promote this configuration: the earlier B16/2K-budget screen reached
74.54 tokens/s at 16K/C16. C128 is queue concurrency, not a B128 kernel.
[Results](gemma4-12b-h100-data/bf16-lean-throughput8k-output128.json) include the
raw measurement file hash. Serving consistency, cancellation, slot reuse and
context rejection [passed](gemma4-12b-h100-data/bf16-lean-throughput8k-verify.log).

## Native projection packet and all-op cases

The final native BF16 packet has prefill rungs 128/512/1024/2048 and decode
rungs 1/2/4/8/16. Each decode rung contains 718 instructions: 328 layer GEMVs
bind native role 8, with an object hash and ABI checked at load. The LM head
remains an interpreter GEMV. Unfusing projections also exposes 48 GLU ops and
additional RMSNorm ops; residual/normalization fusion remains intact.

The audit now rejects unknown opcodes, invalid queue bounds, and instructions
without queue coverage. It groups every instruction into per-rung cases using
opcode, launch blocks, immediate bits, operand extents, and declared roles.
Cases retain all PCs, so repeated layer sites cannot disappear from the audit.
They identify specialization work; they do not certify a selected runtime
kernel or infer dtype from byte counts.

Four packet configurations passed: 36 programs and 25,256 instructions, with
exact case-to-instruction parameter/role coverage. The
[all-op snapshot](gemma4-12b-h100-data/all-op-ladder-coverage.json) records each
packet hash, rung, case count and op counts. The
[native route snapshot](gemma4-12b-h100-data/native-tc-final-coverage.json)
records 1,640 native projection sites across its five decode rungs.

Still unqualified: per-rung LM-head selection, RMSNorm, residual norms, GLU,
embedding, softcap, argmax, and the full attention/history matrix. BF16 native
projection evidence does not qualify FP8, AMD or CPU. All-op specialization
is incomplete; no blanket fast-path claim is made.

## Complete work-slice audit and HD512 rung checks

The audit now requires each instruction's slices `0..blocks` exactly once,
and compares queue entries against the scheduled stream, including wait,
successor, flags and segment fields. A present opcode with missing or duplicate
slices no longer passes. The output also retains the packet target fingerprint.
This is an offline check and adds no runtime dispatch overhead.

All 16 local Gemma 4 packet artifacts pass: **147 programs, 101,674
instructions**. The [snapshot](gemma4-12b-h100-data/all-packet-slice-coverage.json)
records each packet hash and every rung's op counts. Mutation tests reject
missing, duplicate and out-of-range slices and changed dependency metadata;
permuted valid work passes.

The native HD512 QK-unroll specialization passes direct interpreter-body
FP64 checks at query rows 128/512/1024/2048/4096/8192, with history ending at
16384, BF16, 16 query heads and one KV head. For both mapped and unmapped
staging, all output bytes match the unroll-1 baseline at every tested size.
[Rung evidence](gemma4-12b-h100-data/attention-qku32-rungs.json) contains object
hashes, oracle results and full-output hashes. This qualifies the tested
HD512 cases; it does not qualify HD256, decode attention or every history.

The current candidate uses prefill rungs 128/512/1024/2048/4096 and decode
rungs 1/2/4/8/16. Each prefill rung contains 766 instructions; each decode rung
contains 718. Every emitted family remains in the audit, including the shared
norm/pointwise/sampling bodies and fused HeadNormRope/NormResidualNorm bodies.
Their presence and successful execution do not establish exact-shape tuning.
The remaining all-op qualification list above is still open.

## Packet versus compiler case checks

Pass the matching `build.json` as the audit's third argument to require exact
all-op inventory coverage:

```sh
nix develop -c cargo run -p plowrt --example packet_ladder_audit -- model.pkt audit.json build.json
```

The check compares every rung, instruction count, PC, opcode, block count,
immediate, packed float/integer bits, operand presence and operand extent.
Missing, duplicate, out-of-range or mismatched cases fail. Float and integer
immediates are compared using the packet's shared wire fields. The report
records the packet SHA256 and the checked manifest path.

Twelve Gemma 4 artifacts with compiler case inventories pass:
**108 programs, 76,984 instructions**. The
[case coverage snapshot](gemma4-12b-h100-data/packet-build-case-coverage.json)
records packet and build-manifest hashes plus each rung's op counts. Older
artifacts without this inventory require the packet-only queue/slice audit.

This verifies what the compiler emitted. It does not certify that all ops have
qualified specializations: runtime-selected objects, attention history,
performance counters and end-to-end timing still need independent evidence.

## Request-limited aggregate ladder

The opt-in BF16 Gemma 4 SM90 TP1 contract separates `PLOW_MAX_CHUNK=8192`
aggregate rows from `PLOW_MAX_REQUEST_CHUNK=1024` real rows per request. The
full B16 packet has six prefill rungs (128/512/1024/2048/4096/8192), five decode
rungs (1/2/4/8/16), 766 instructions per prefill rung and 718 per decode rung.
At context 20480 it allocates 15 GiB KV, versus 45 GiB for the retained B16/4K
packet. Its activations occupy 1.63 GiB; weights remain 22.2 GiB.

Packed padding uses slot `-1`; fused BF16 HeadNormRope skips its KV writes but
still computes padded query rows. The compiler, planner, ordinary prefill and
mux enforce the per-request limit. Objects must export masked-padding ABI1;
the build manifest and generated header carry this requirement. Unsupported
architectures/precisions and limits without a matching rung are rejected.
Absent request limits preserve the existing contract.

Build matching segments with `PLOW_BUILD_MASKED_PADDING=1` and optionally
`PLOW_BUILD_FATLITE=1` using `scripts/build_sm90a_gemma4_segments.sh`. Copy its
HD512 role object into the asset directory before compiling the packet, so
the packet pins that object's hash. Larger physical decode batches require
separate native projection qualification; this change does not add B32.

`runtime/tests/packed_kv_padding_sm90.cu` verifies all six prefill rungs at
HD256/KV8 and HD512/KV1. All twelve cases pass with zero Compute Sanitizer
memcheck errors: wrapped real writes match query-body outputs byte-for-byte,
all other KV bytes stay untouched, and padded query rows are computed.
Full-packet serving checks pass ragged isolated/concurrent consistency,
cancellation, slot reuse, output counts and context rejection. These tests
do not replace independent model-quality or performance qualification.

The [screen](gemma4-12b-h100-data/request-limit-screen.json) ran one repeat
after one warmup per cell, 128 output tokens, cache disabled. Output throughput
at input 1K/16K was 77.29/45.54 tokens/s at C1 and 570.10/99.33 at C16.
All 34 measured requests returned 128 tokens. The retained configuration's
16K/C16 result is 98.55 tokens/s; this screen establishes no clear speedup.
The contract remains opt-in. [Artifact hashes and checks](gemma4-12b-h100-data/request-limit-contract.json),
[GPU memcheck](gemma4-12b-h100-data/request-limit-gpu-memcheck.log) and
[serving checks](gemma4-12b-h100-data/request-limit-serve-verify.log) record the
tested scope. No C128 screen or independent full-model logit comparison was
run for this candidate.

## B32 all-op cases

The BF16 native-head candidate emits six prefill rungs
128/512/1024/2048/4096/8192 and six decode rungs 1/2/4/8/16/32.
The strict packet/build audit passes **12 programs, 8,904 instructions and
2,736 exact cases**. The [case inventory](gemma4-12b-h100-data/b32-head-ladder-cases.json)
retains all PCs, shape immediates, operand extents, launch blocks and roles.
All **329 decode GEMVs per rung**, including the tied LM head, bind native
role 8: 1,974 projection sites across the ladder. Runtime selection covers
54 BF16 M/N/K cells; smaller rungs inherit the widest rung's BK/split choice
to preserve summation order. Unknown shapes and incompatible objects fail.

| Family | Prefill / decode instructions per rung | Current qualification |
|---|---:|---|
| Gemm / Gemv | 329 / 329 | Native SM90 prefill tiles; decode now includes B32 and LM-head tensor-core cells. All prefill cells still need launch-level performance qualification. |
| FlashPrefill / FlashDecode + FlashMerge | 48 / 48 + 48 | HD256/512 bodies; full rung/history timing and counter matrix remains open. |
| HeadNormRope | 144 / 144 | Fused dimension-specific bodies, masked-padding checks. Exact-case performance remains open. |
| RmsNorm | 97 / 1 | Decode block reduction now invariant through B32; full-logit rung checks pass. Exact-case performance remains open. |
| NormResidual / NormResidualNorm | 96 / 96 | Residual/normalization fusion retained. Exact-case performance remains open. |
| Glu | 48 / 48 | Native pointwise body, exposed by projection unfusing. Exact-case performance remains open. |
| Embed, SoftCap, Argmax, ArgmaxFin | 1 each / 1 each | Shared native bodies; exact-case performance remains open. |

B32 requires a projection capability marker and a decode normalization ABI
marker. An old main interpreter produced different full-model logits at a
rung transition; the fixed decode reduction passes 170 full-logit snapshots
at each of 128 and 16,384 prompt rows. Startup rejects the old main object.
These compare native narrow/widest execution, not independent model quality.

The [B32 evidence](gemma4-12b-h100-data/b32-native-screen.json) records tested
objects and numerical/serving limits. Per-op specialized high performance
for **every** emitted case is still incomplete. These H100 BF16 checks do not
qualify FP8, AMD or CPU.

The subsequent [GEMM epilogue screen](gemma4-12b-h100-native-tuning.md#shared-memory-gemm-epilogue-screen)
covers all 49 emitted BF16 GEMM shapes through the actual ordinary interpreter
object. Segment and queue execution match every output byte of the unchanged
standalone body, and all cases pass sampled FP64 checks. Per-rung weighted
GEMM speedups range from 1.021× to 1.152×; the packed serving screen improves
16K/C32 throughput from 104.374 to 108.485 tokens/s with all paired texts equal.
This narrows the GEMM qualification gap. It does not qualify all other ops,
all packed histories, FP8, or arbitrary shapes.

## Opt-in HD512 KV64 packet

The BF16 H100 KV64 candidate passes the strict packet/build audit at all six
prefill and six decode rungs: **8,904 instructions, 2,736 exact cases**.
Its [case inventory](gemma4-12b-h100-data/attention-kv64-ladder-cases.json)
retains every PC and declared role. All eight HD512 attention sites per
prefill rung bind the hash-pinned Q64/KV64 object. Compiler and runtime checks
reject mismatched geometry; the compiler also checks the 205,824-byte arena.
Other op routes and fusion remain as in the B32 native-head packet.

Actual-role sampled FP64 checks pass at rows128/512/1024/2048/4096/8192 with
16K history, with and without TMA descriptors. Ragged memcheck and racecheck
pass. The [tuning report](gemma4-12b-h100-native-tuning.md#hd512-kv64-single-stage-screen)
records the speedups and mixed serving results. This remains opt-in: changing
the softmax tile is not output-identical to KV32. Full-model rung consistency
passes, but three of 132 paired serving texts differ. This adds qualification
for one attention geometry; the all-op gaps above remain open.

## FP8 KV padding prerequisite

The FP8 HeadNormRope cache writer now skips negative slot IDs when
`PLOW_NV_MASKED_PADDING=1`, matching the BF16 writer. Previously, padding
slot `-1` became an unsigned cache offset: the regression reproduced 897
Compute Sanitizer errors before the fix.

The extended padding probe passes 24 cases with zero memcheck errors:
BF16 and FP8 at every prefill rung128/512/1024/2048/4096/8192, with
HD256/KV8 and HD512/KV1. FP8 real rows and scales match an unpadded invocation
byte-for-byte; all other cache bytes and scales retain their sentinels.
The cases exercise noncontiguous slots, ring wrapping and padded rows.
[Evidence](gemma4-12b-h100-data/packed-kv-fp8-padding-summary.json).

This fixes one prerequisite for larger batches. The compiler still rejects
FP8 KV with request chunk limits. Packed FP8 attention, object contracts,
full-model correctness and native B64 projections need qualification before
that restriction can be lifted. No throughput gain is claimed by this fix.

## Packed FP8 attention body checks

`runtime/tests/packed_flash_fp8_sm90_correct.cu` exercises the existing native
FP8 packed mux at all six prefill rungs, HD256/KV8 with a 1024-token local
window and HD512/KV1 with full attention. Two ragged requests use physical
slots2/0, histories ending at16384/8193, per-row K/V scales and up to1024
real rows per request. Every output is checked for finiteness, and every
padded output must be zero.

An FP64 reference reads the stored FP8 K/V values and row scales directly,
with BF16 queries, causal masking and the local ring mask. It checks first,
middle and last rows at three query heads per request. This separates
attention execution error from the upstream KV quantizer's approximation;
it is not an independent full-model quality check. All12 cases pass memcheck
with zero errors; worst sampled relative L2 is0.005879 and absolute error
is0.000283 (limits0.015/0.002).

Racecheck instruments the first two launches only: 128-row HD256/KV8 and
HD512/KV1. Both pass with zero hazards, errors or warnings. The full-ladder
racecheck attempt was deliberately stopped because of instrumentation cost;
it is not counted as a pass. [Raw results and scope](gemma4-12b-h100-data/packed-flash-fp8-summary.json).

This is body validation, not loaded-interpreter or serving qualification.
The compiler and packet validator retain their BF16-only request-limit
guards. Extending the contract must distinguish FP8 objects built after the
writer fix: the old general padding marker alone cannot establish that fix.

## FP8 padding object capability

Packed FP8 objects built with masked padding now export
`plow_pf_fp8_masked_padding_abi=1`. Object validation for an FP8 manifest with
request limits requires this marker in addition to general padding ABI1,
packed request ABI2 and FP8 request ABI1. Objects with only the old general
padding marker are rejected. BF16 and manifests without request limits keep
their existing requirements.

Nine contract tests pass. A newly built SM90 FP8 packed light object passes
actual ELF capability inspection; suppressing its new marker through the
reader callback makes validation reject it. The object compiles at128
registers with2368 stack bytes and8880/12292 spill-store/load bytes. These
are whole-entry compiler totals, not executed writer-path measurements.
[Checks and object hash](gemma4-12b-h100-data/fp8-masked-contract-summary.json).

Compiler and packet-level BF16-only request-limit guards remain in place.
This prepares stale-object rejection; it does not enable FP8 request limits,
qualify loaded GPU execution or demonstrate a throughput improvement.

## Loaded FP8 writer and attention checks

Both probes now accept an optional cubin path. Build with `-lcuda` to enable
driver loading. The writer probe targets the packed light interpreter; the
attention probe targets the packed FA interpreter. They read the exported
arena size and four request/padding capabilities, then execute a synthetic
single-op packet with 132 work slices, a satisfied dependency and a completion
counter. Each test requires all 132 completions.

The loaded FP8 writer passes 12 cases across six rungs and both head
geometries, matching every real byte and scale against the unpadded standalone
body and preserving all inactive sentinels. The loaded FP8 attention passes
all 12 numerical cases against the FP64 reference, exercising its tagged
request handle and separate scale operands. All outputs are finite and
padding is zero. Both loaded-object runs pass memcheck with zero errors.

The first FA build lacked exported arena metadata and was rejected before
launch; rebuilding with `PLOW_NV_EMBED_SMEM=1` fixed the setup. Its entry uses
255 registers, 256 stack bytes and 352/584 spill-store/load bytes. These are
compiler totals, not measured performance. The tests use synthetic single-op
packets; full-model FP8 request-limit compilation, serving and B64 remain
unqualified. [Logs, object hashes and scope](gemma4-12b-h100-data/fp8-loaded-ops-summary.json).

## FP8 request-limited compilation and first screen

Request limits now support FP8 KV with BF16 weights on packed Gemma 4 SM90
TP1. This supersedes the earlier compiler/packet restriction above. Existing
FP8 flags remain opt-in; FP8 weights, other backends and invalid request
limits remain rejected. Both all-layer FP8 KV and full-attention-only FP8 KV
pass structural compilation tests. The manifest records the new FP8 padding
capability and includes it in the pairing hash; stale objects remain rejected.

The real all-layer FP8-KV B32 model passes the strict packet/build audit:
12 programs, 8,904 instructions and 2,736 cases. KV allocation is
15.1953125 GiB versus 30 GiB for BF16 KV, at context20480 and request
chunk1024/aggregate8192. Weights remain BF16, with native decode projections.
Serving consistency, cancellation, slot reuse and context rejection pass.

One screening repeat after verification, without additional warmups, returns
128 tokens/cache0 for all 66 requests:

| Input / concurrency | Output tok/s | Median TTFT ms |
|---|---:|---:|
| 1K / C1 | 70.900 | 85.828 |
| 1K / C32 | 620.301 | 1481.448 |
| 16K / C1 | 39.330 | 1482.029 |
| 16K / C32 | 57.398 | 65382.882 |

This is slower than recent BF16-KV screens; it is a capacity option, not a
recommended throughput configuration. No fresh BF16/vLLM A/B or independent
full-model quality test was run. FP8 KV changes numerical precision.

A separate 16K/C1 event profile attributes 531.0 ms to GEMM, 138.8 ms to
light ops and 758.7 ms to attention. Earlier BF16-KV profiles were about
527/137/349 ms. All eight HD512 segment IDs appear in every chunk's top list;
their rounded times sum to 588.58 ms across 16 chunks. The packet audit maps
those IDs to HD512 FlashPrefillFp8. This makes global FP8 attention the main
next tuning target. Event profiling replaces graph execution and is not
normal serving latency. B64 and the vLLM objective remain unqualified.
[Evidence, objects and build recipe](gemma4-12b-h100-data/fp8-request-limit-summary.json).

## Audited W8A8 GEMM maps

Packed live-KV validation now accepts GEMM FP8 tensor maps only when both
E4M3 generators agree with the instruction source handles, K, row extent and
buffer sizes. It rejects duplicate generators, missing/invalid handles,
non-E4M3 descriptors, mismatched sources, truncated buffers and cache aliases.
The mapless path remains supported; indirect fused FP8 GLU maps remain
unqualified and rejected. This validation runs during packet compilation/load.

The actual H100 W8A8 B16 packet passes strict packet/build/queue checks at
prefill128/512/1024/2048 and decode1/2/4/8/16:9programs,6942instructions,
1904cases. Each prefill rung has328 mapped FP8 GEMMs and192 QuantFp8 ops.
The existing norm/GLU quantization fusion reduces each rung from958 to814
instructions. It also passes the audit, but its performance/quality results
below do not justify promotion.

[Builds, object hashes, all-op counts and tests](gemma4-12b-h100-data/w8a8-packed-summary.json)
record the tested pairing. Use explicitly matched W8A8 prefill objects and a
current decode object; the earlier reused decode object failed a cancellation
check. Full-model independent quality and maximum-concurrency qualification
remain open. This does not enable FP8-weight request chunk limits or native
BF16 projection routing for FP8 weights.

## FP8 decode and unified-path precision screen

The parameterized packed-logit diagnostic now accepts prompt length, repeated
token ID and decode steps. Its original defaults remain unchanged. A 901-token
prompt with three teacher-forced steps exercises seven schedules, including
noncontiguous slots, reversed admission and unified token batching.

All six native schedules are bit-identical for each tested configuration.
Unified execution uses prefill kernels even for decode rows; its numerical
policy differs from native decode. The baseline has a maximum logit difference
of 4.953125 without selected-token changes in the 12 unified snapshots.
This localizes the diagnostic difference to the execution paths, but does not
establish one sole cause or independent model accuracy.

Native FP8 WGMMA across all decode rungs, paired with uniformly tiled prefill
and FP32 accumulator promotion, lowers that difference to 3.0625. It passes
the serving consistency screen but regresses latency and long-context
throughput. Matching quantizer or normalization policies does not eliminate
divergence. BF16 rounding inside the fused GLU further lowers the short
diagnostic difference to 2.5, yet fails the broader serving consistency test.
No timing was run for that failed candidate.

The MMA/WGMMA all-rung, block-normalization and GLU-rounding experiments are
reverted. Their patches, object hashes, diagnostic outputs and serving logs
are retained in the [precision screen](gemma4-12b-h100-data/w8a8-decode-consistency-summary.json).
No candidate is promoted. Diagnostic execution passing is not a logit-equality
gate; these experiments have no independent quality or sanitizer qualification.

A separate native-head prototype leaves all 1,904 instruction cases unchanged
and assigns only the BF16 262144×3840 vocabulary projection to native role8
at B1/2/4/8/16. All 48 fused FP8 GLUs remain. Four routing tests and the strict
packet audit pass. Its 66-frame diagnostic matches the control's result:
native schedules are bit-identical; unified maximum difference is 4.953125
with no selected-token changes.

However, a fresh unchanged control fails cancellation consistency, and the
head candidate fails concurrent ragged consistency. Neither enters timing.
Earlier passing W8A8 serving screens therefore do not establish stable
scheduling invariance. The compiler enablement is reverted pending that
investigation; no native-head speedup or causal attribution of these failures
is claimed. [Prototype, exact-case audit and failures](gemma4-12b-h100-data/w8a8-native-head-summary.json).
