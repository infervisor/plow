# Gemma 4 12B packet ladder coverage

Target: `google/gemma-4-12B-it`, H100 `sm_90a`. Compilation coverage is not
performance qualification. Native bodies can remain in a shared interpreter;
a separate object per opcode is not a requirement.

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
