# Gemma 3 12B: dedicated HD256 attention screen

H100, BF16 or per-channel FP8 W8A16; BF16 activations and KV. One warmup and three measured repeats per case, concurrency 1, output 128, prefix caching disabled. This is a candidate screen, not the full vLLM comparison.

| Precision | Input | Previous TTFT ms | HD256 TTFT ms | HD256 TPOT ms |
|---|---:|---:|---:|---:|
| bf16 | 1024 | 153.51 | 121.52 | 11.525 |
| bf16 | 16384 | 3613.52 | 1413.99 | 12.004 |
| fp8 | 1024 | 358.90 | 320.43 | 7.916 |
| fp8 | 16384 | 6750.36 | 4681.03 | 8.379 |

All 12 requests have exact input/output counts and zero cache hits. All output texts match the previous configuration on identical prompts. Both 5909-token retrieval checks return LEMON-482. These are limited correctness checks, not independent numerical parity or a quality evaluation.

Configuration: unchanged occ2 decode object; WS384 BF16 GEMM object; added dedicated attention object built by `gemma3-12b-h100/build_fa256.sh`. Packet adds `PLOW_SEG_FA512=all` to the previous replay knobs and slicing environment; runtime adds `--pf-seg-fa512 all`. Native code is reused without a kernel source edit.

Diagnostic event profile: GEMM stays near 37.7 ms/chunk while FAT grows from about 70 to 252 ms at 16K. The eight longest segments repeat at the global-attention layer spacing. Event timings are diagnostic and excluded from the serving screen.

Still behind vLLM. Remaining work includes independent attention numerical validation, repeated interleaved baseline/candidate measurements, packed attention routing, concurrency through 128, and the FP8 W8A16 TMA/WGMMA projection path.

## Packed follow-up

Added a distinct `_pfpackedfa` object and loader routing through existing packed ABI validation. An ordinary attention object renamed as packed is rejected at load. All 326 exec tests pass (25 ignored). Independent FP64 sampled oracle passes eight HD256 cases: 128/1024 query rows, 1024/16384 context, local window1024/global window0, 16 query heads/8 KV heads. Maximum relative L2 is 0.002714 across nine sampled vectors per case; all device outputs finite. This checks the native attention body, not full-model numerical parity.

B16, cuBLASLt decode, maximum prefill chunk2048, packing chunk1024, interleave2048: isolated vs concurrent serving parity, ragged prompts, slot reuse, cancellation and context rejection pass. One warmup/one measured repeat, concurrency16, output128:

| Input | Previous TTFT ms | Dedicated attention TTFT ms | Previous tok/s | Dedicated attention tok/s |
|---|---:|---:|---:|---:|
| 1024 | 2409.04 | 1307.65 | 391.14 | 492.45 |
| 16384 | 99656.60 | 33492.60 | 19.69 | 54.12 |

All32 requests have exact token counts and zero cache hits. Still behind vLLM; this single-repeat screen needs repeated qualification. FP8 packed assets are compiled but unqualified.

At input1024/concurrency128, one warmup/one measured repeat: TTFT15967.38ms, TPOT22.163ms, throughput491.45tok/s. All128 requests have exact counts and zero cache hits. This uses a B16 active batch with queuing up to128, not physical batch128. The 16K/concurrency128 cell remains unmeasured for this candidate.
