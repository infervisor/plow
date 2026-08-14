# DeepSeek V4 Flash TP1 on MI325X

Measured 2026-08-14 on one 256 GiB MI325X with the released 43-layer
DeepSeek-V4-Flash-0731 checkpoint. The checkpoint upload is 145.30 GiB; the
166.9 GB on-disk checkpoint and runtime state fit at TP1.

## Current gates

| Batch | Context | Step time | Aggregate rate | Correctness |
|---:|---:|---:|---:|---|
| 1 | 8K | 42.6 ms | 23.5 tok/s | real checkpoint |
| 1 | 16K | 42.6 ms | 23.5 tok/s | real checkpoint |
| 32 | 8K | 177.2 ms | 180.6 tok/s | 32 identical prompts agree |
| 32 | 16K | 177.2 ms | 180.6 tok/s | 32 identical prompts agree |

These are decode-step gates. V4 has no prefill bucket yet, so prompt processing
walks the decode program and the table is not a TTFT or complete serving claim.
The requested B1 >=50 tok/s and B32 >1000 aggregate tok/s targets are not met.

The pre-fp32-repair B32 progression was approximately 90 -> 119 -> 140 -> 149
-> 173 -> 176 tok/s. It came from per-row state/correctness repairs, parallel decode
tails, tiled hyper-connection projection, a token-wide hyper-connection expand,
dense block-FP8 attention GEMMs, and reducing sparse-attention split-K from four
chunks to one at B32.

## Roofline and trace

The local MI325X probe measured 4.164 TB/s HBM bandwidth. The analytic VALU
ceiling used by the kernel harness is 81.7 TFLOP/s, giving a 19.6 flop/byte
ridge point. One persistent-interpreter dispatch avoids a host launch for every
packet; the measured launch floor is 6.0 us for one kernel and 7.6 us for two.

The B32 attention block-FP8 family contains 193 packets and 213.3 GFLOP per
step. The trace attributes 6.67 GB to the family, for 32.0 flop/byte. Its
measured span is 22.5 ms = 9.48 TFLOP/s and 296 GB/s, far below either roof.
The remaining 129 shared-expert block-FP8 GEMVs contain 69.3 GFLOP and 2.16 GB
at 32.0 flop/byte, but take 24.0 ms = 2.89 TFLOP/s and 90 GB/s. Both are
latency/occupancy limited rather than HBM-roof limited.

The corrected fp32-compressor B32 trace attributes the largest families as
follows. Trace instrumentation raises the observed step time, so these spans
rank kernels but do not sum to the untraced 177.2 ms gate.

| Family | Step span |
|---|---:|
| V4 index top-k | 51.5 ms |
| Sparse attention | 47.3 ms |
| Shared-expert block-FP8 GEMV | 24.0 ms |
| Dense block-FP8 attention GEMM | 22.5 ms |
| V4 grouped output linear | 17.7 ms |
| FP32 compressor projection | 15.1 ms |
| Index score | 14.8 ms |
| Routed A4W4 MoE | 10.4 ms |
| Hyper-connection dot | 9.6 ms |
| Headnorm/RoPE | 7.5 ms |
| KV compression | 6.7 ms |
| Other bf16 GEMV | 6.1 ms |

An eight-GPU independent sweep of sparse-attention split counts
`1,2,3,4,6,8,12,16` found B32 aggregate rates of
`175.6,172.5,174.3,172.1,170.7,165.0,165.6,158.7` tok/s. All arms retained
same-prompt token-stream agreement. `SPLIT=1` is now the B32 default; B1 keeps four-way
split-K to fill the device.

The compressor projection must accumulate and store fp32 to match the reference
model. Vectorizing its bf16 inputs and weights in groups of eight reduced its
traced span from 74.4 ms to 15.1 ms and improved the corrected B32 gate from
157.0 to 180.6 tok/s. B1 retains the scalar path because the vector path lowers
its rate from 23.5 to 21.1 tok/s.

## Plow-specific advantages

- The counter-DAG persistent interpreter runs the 1,632-packet model in one
  host dispatch, including device-side argmax. Packet changes do not create a
  launch-per-op serving path.
- Native checkpoint layouts remain on device: block-FP8 dense projections and
  MXFP4 routed experts execute without expanding the 145.3 GiB upload.
- TP1 avoids collective latency and is possible because the complete weights,
  per-sequence KV rings, compressor state, and index state fit one MI325X.
- Per-op CU sets, GQ scheduling, and L2-domain placement make narrow reductions
  and wide projections coexist in the same packet graph.
- Raw per-packet device traces attribute wall time to model opcodes, which made
  the hyper-connection and projection wins measurable instead of inferred.

## Open gaps

- B1 needs 2.13x and B32 needs 5.54x more throughput to reach the requested
  gates. Kernel utilization, not the HBM byte roof, is the immediate limit.
- Converting the 129 shared-expert projections to dense block-FP8 GEMM is fast
  and exact by itself. Combining it with the 193 attention GEMMs exposes a
  same-row divergence in the routed-MoE/later residual path. A deterministic
  expert-slot order fixes one layer but not the full model, so this conversion
  is intentionally not shipped.
- The reference compressor casts its activation to fp32 before its bf16-weight
  projections. `V4LinearF32` now implements that contract and its 3-layer gate
  has exact fp32 projection and sparse-attention rows. An external reference-logit
  comparison is still required for model-level numerical parity.
- Prefill buckets are still required for meaningful 8K/16K TTFT and complete
  serving measurements.
