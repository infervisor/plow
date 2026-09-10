# Gemma 3 12B W8A16 WGMMA experiment

The initial FAT interpreter builds regress. A subsequent dedicated GEMM object
plus prefill graph replay improves C1 TTFT by about 6% against a fresh control.
The vLLM goal remains unmet; wider performance qualification is pending.

`PLOW_NV_W8A16_WGMMA=1` opts into the existing BF16 WGMMA mainloop with an FP8
weight staging overload. E4M3 values expand exactly to BF16 in shared memory;
activations remain BF16, and FP32 per-channel scales apply after the reduction.
The shared-store path includes an async-proxy fence. No tensor maps, additional
launches, scratch allocations, or activation quantization are introduced. The
default remains the original kernel. K must be positive and divisible by eight
for the opt-in path; other K values retain the original dispatcher.

H100, 132 CTAs, 256 threads, synthetic BF16 activations and per-channel E4M3
weights. Fifteen CUDA-event samples, four warmups, 256 MiB cache flush before
each sample. No other GPU work or CPU builds overlapped timing.

| M | N | K | Original µs | WGMMA µs | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1024 | 4096 | 3840 | 306.400 | 216.032 | 1.42× |
| 1024 | 15360 | 3840 | 1197.024 | 815.488 | 1.47× |
| 1024 | 3840 | 15360 | 1176.800 | 774.592 | 1.52× |
| 128 | 2048 | 3840 | 164.864 | 107.232 | 1.54× |
| 128 | 3840 | 4096 | 176.000 | 114.240 | 1.54× |

All full outputs match the original kernel on these inputs. Each large shape
also passes 257 sampled FP64 reference checks; tiny integer tests check every
output, including ragged M/N. The integrated dispatcher passes Compute Sanitizer
memcheck with zero errors, including K=8 and K=264. Sanitizer timings are excluded
from this table. These checks do not establish full-model numerical parity.

The full FAT interpreter uses 255 registers with 1736 bytes of stack and reported
spill loads/stores. Its three-repeat C1 serving screen regresses from 320.43 to
341.77 ms TTFT at 1K and 4681.03 to 4920.35 ms at 16K. All six request texts match
the previous configuration, with exact counts and zero cache hits; the 5909-token
retrieval check also passes. FATLITE's 128-register cap increases spilling and
regresses further. A dedicated GEMM object needs explicit packet/object pairing
before mapless W8A16 projections can be moved out of FAT; the existing TMA-map
guard must not simply be removed.

FP8 packed serving qualification also completed using the prior attention-only
configuration: all serving checks and 32 measured request audits pass. At C16,
1K TTFT is 4684.38 ms, TPOT 43.55 ms, throughput 200.43 tok/s; 16K TTFT is
86356.83 ms, TPOT 55.78 ms, throughput 21.92 tok/s. These are one-repeat screens,
not final repeated comparisons.

Raw logs, probe sources, object build scripts, and request records are in
`gemma3-12b-h100/`, prefixed `w8a16`, `build-w8a16`, and `fp8-w8a16`.

## Dedicated object follow-up

`PLOW_SEG_PURE_GEMM=w8a16` and `--pf-seg-pure w8a16` select eligible mapless
W8A16 projections plus the mapped BF16 head. Compiler and runtime share the
eligibility predicate. The loader requires `plow_w8a16_gemm_abi_SUFFIX=1` from
a compatible 256-thread GEMM object; the old WS384 object is rejected on the GPU.
Modes `1` and `fp8` retain their map requirement. The opted-in GEMM object omits
the unused fused GLU arm. Its reported stack and spill loads/stores are 16 bytes
each, substantially below the FAT build.

One new packet eligibility test and all 326 exec tests pass (25 ignored).
Ordinary and packed objects load successfully. Long retrieval, isolated vs
concurrent ragged outputs, slot reuse, cancellation, and context rejection pass.
These are correctness gates, not a full quality evaluation.

C1, input 1K/16K, output 128, one warmup and three measured repeats:

| Configuration | 1K TTFT ms | 16K TTFT ms |
|---|---:|---:|
| FP8 attention-only, fresh control replay | 318.74 | 4686.52 |
| FP8 dedicated W8A16 GEMM | 302.47 | 4445.81 |
| FP8 dedicated GEMM + prefill graph | 298.96 | 4412.72 |
| BF16 attention + prefill graph | 104.09 | 1400.92 |

The prior BF16 attention screen was 121.52/1413.99 ms. All compared response texts
match on identical prompts, with exact token counts and zero cache hits.
Earlier serving screens used separate prefill segment launches; decode graphs
were separate. `--pf-seg-graph` now explicitly enables the prefill graph test.

Diagnostic profiling confirms the dedicated route is active: 193 GEMM, 242 FAT,
and 48 attention segments at 1K, taking 228.4/27.9/8.4 ms respectively. Graph
replay captures 483 nodes. GEMM remains the main cost; the isolated 1.5× speedup
has not transferred proportionally. Packed performance through C128 and the
16K/C128 candidate cell remain unqualified.


## Decode MMA and occupancy follow-up

- fp8-b16-lean-screen, input 1024: median TTFT 4063.34 ms, TPOT 43.325 ms
- fp8-b16-lean-screen, input 16384: median TTFT 76484.82 ms, TPOT 56.127 ms
- fp8-b16-decode-mma-screen, input 1024: median TTFT 4066.46 ms, TPOT 46.972 ms
- fp8-b16-decode-mma-screen, input 16384: median TTFT 76386.73 ms, TPOT 59.009 ms
- fp8-lean-occ2-screen, input 1024: median TTFT 328.44 ms, TPOT 7.922 ms
- fp8-lean-occ2-screen, input 16384: median TTFT 4716.99 ms, TPOT 8.400 ms

All measured requests have exact input/output counts and zero cache hits. C16
screens use one measured repeat; C1 occupancy screen uses three. Decode MMA
passes the 66-check native probe and 16-request isolated/concurrent response
comparison, but regresses TPOT and is not selected. The MINBLK=2 prefill build
still launches at one block/SM: TMA stages=4 requires 132160 bytes, despite its
128-register cap. It spills 760/1400 bytes stores/loads and regresses C1. Long
retrieval passes. Next experiment uses TMA stages=3 to make two-block occupancy
possible; build in progress. Full 16K/C128 qualification and vLLM win remain unmet.


## Three-stage TMA occupancy result

Input 1024: TTFT 298.96 → 212.45 ms (28.9% reduction).
Input 16384: TTFT 4412.72 → 2952.64 ms (33.1% reduction).

H100 loader confirms GEMM grid264, dynamic shared memory99376 bytes, versus
grid132/132160 bytes before. Three repeats per input, output128, C1, one warmup.
All six output texts and prompt hashes match the previous lean+graph control;
exact token counts and zero cache hits pass. The 5909-token retrieval check passes.
Both ordinary/packed objects build, but packed runtime qualification remains
pending. This is a promising C1 candidate, not a vLLM win or completed matrix.
Build recipe: build_w8a16_lean_occ2_s3.sh. No production defaults changed.


## Packed three-stage occupancy screen

Packed serving checks pass (ragged isolation/concurrency parity, slot reuse,
cancellation and context recovery). Runtime confirms GEMM grid264/smem99376.
C16 one-warmup/one-repeat screen, output128: 1K TTFT3843.55ms, TPOT43.430ms,
218.74tok/s; 16K TTFT72566.07ms, TPOT56.064ms,25.70tok/s. Only about5% TTFT
improvement over prior packed lean+graph, much less than C1 improvement.
All32 requests exact counts/cache0 and matching prompt hashes.

User asked whether unified token batching/stagger/segments were active.
Branch tp-bringup-mi300x. token_batch=true but route ready=false/fires=false.
Packed/chunked prefill and interleave2048 enabled; no separate stagger switch
found in this path. Packet class slicing/w8a16/FAall and runtime segmented=true
confirmed. Source investigation: CudaTokenBatch::load requires has_packed_terminal.
PackedTerminal::load requires VMM prefix plus an exact five-op terminal including
SoftCap and an unmapped head. Gemma3 packet has no SoftCap and uses mapped head.
This is a concrete compatibility limitation, not evidence that plowc omitted
segments. Next inspect/generalize terminal contracts safely for Gemma3, preserving
existing Gemma paths, and qualify unified batching before larger performance runs.
No production source edits this turn. C128 new candidate still pending; goal unmet.

Full response texts matching previous control: 32/32.
