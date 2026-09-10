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
takes 81.581 us versus 60.778 us for BKV64. This candidate still needs serving
validation; the checked-in recipe retains the serving-tested BKV32 selection.

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
