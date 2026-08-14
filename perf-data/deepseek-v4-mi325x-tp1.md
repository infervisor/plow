# DeepSeek V4 Flash TP1 on MI325X

Measured 2026-08-14 on one 256 GiB MI325X with the released 43-layer
DeepSeek-V4-Flash-0731 checkpoint. The checkpoint upload is 145.30 GiB; the
166.9 GB on-disk checkpoint and runtime state fit at TP1.

## Current gates

| Batch | Context | Step time | Aggregate rate | Evidence |
|---:|---:|---:|---:|---|
| 1 | 8K | 41.6 ms | 24.0 tok/s | bound checkpoint, requested position |
| 1 | 16K | 42.4 ms | 23.6 tok/s | bound checkpoint, requested position |
| 32 | 8K | 186.7 ms | 171.4 tok/s | bound checkpoint, requested positions |
| 32 | 16K | 191.4 ms | 167.2 tok/s | bound checkpoint, requested positions |

These are decode-step timing gates with the real checkpoint bound. V4 has no
prefill bucket yet, so the requested positions read KV rows that this run did
not populate: the schedule and timing are real, but the token ids are not a
model-correctness result. The separate full-checkpoint B32 gate with 32 copies
of one prompt has token-stream agreement, but its one-token decode-only prompt
puts the measured steps near position 5, not at 8K/16K. Earlier tables
incorrectly labeled those short-context prompt runs as 8K/16K because
`amd-bench --prompt` overrides `--ctx` with the prompt length. No complete
8K/16K serving claim is possible until a real prefill bucket populates the KV.
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

The corrected fp32-compressor B32 trace at a true 8K position attributes the
largest families as follows. Its 185.4 ms packet span closely reproduces the
untraced 186.7 ms gate.

| Family | Step span |
|---|---:|
| Sparse attention | 47.8 ms |
| Shared-expert block-FP8 GEMV | 24.0 ms |
| Dense block-FP8 attention GEMM | 22.6 ms |
| V4 grouped output linear | 17.7 ms |
| FP32 compressor projection | 15.1 ms |
| Index score | 14.8 ms |
| Routed A4W4 MoE | 10.4 ms |
| Hyper-connection dot | 9.6 ms |
| Headnorm/RoPE | 7.5 ms |
| KV compression | 6.7 ms |
| Other bf16 GEMV | 6.1 ms |
| V4 index top-k | 1.9 ms |

An eight-GPU independent sweep of sparse-attention split counts
`1,2,3,4,6,8,12,16` found B32 aggregate rates of
`175.6,172.5,174.3,172.1,170.7,165.0,165.6,158.7` tok/s. All arms retained
same-prompt token-stream agreement. `SPLIT=1` is now the B32 default.

An eight-GPU B1 sweep of `SPLIT=1,2,3,4,6,8,12,16`, followed by a same-GPU
paired gate of the leading arms, selected two-way split-K. At 8K it measured
41.6 vs 42.0 ms/token for the prior four-way default; at 16K, 42.4 vs 42.7 ms.
The full-checkpoint prompt stream remains unchanged. `SPLIT=2` is now the B1
default.

The compressor projection must accumulate and store fp32 to match the reference
model. Vectorizing its bf16 inputs and weights in groups of eight reduced its
traced span from 74.4 ms to 15.1 ms and improved the corrected B32 gate from
157.0 to 180.6 tok/s at short context. B1 retains the scalar path because the
vector path lowers its rate from 23.5 to 21.1 tok/s.

The selector originally serialized all 32 independent rows through one
workgroup. Mapping one workgroup per row reduced its true-8K traced span from
51.5 to 1.9 ms and the full-checkpoint timing gate from 236.2 ms / 135.5 tok/s
to 186.7 ms / 171.4 tok/s. The short-prompt correctness stream remains exact.

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

- At 8K, B1 needs 2.08x and B32 needs 5.83x more throughput. At 16K they need
  2.12x and 5.98x. Kernel utilization, not the HBM byte roof, is the immediate
  limit.
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
