# Kimi-K3 MI325X kernel audit

Date: 2026-08-11. Target: TP8 Kimi-K3 on MI325X (`gfx942`) with native
checkpoint MXFP4 weights and FP8 KV.

## Coverage

Every opcode in the current B8 prefill/decode packet resolves to a body in the
loaded gfx942 objects; no live packet falls through a default/NOP arm. The
critical decode object advertises K3, FP8-KV, grouped-A4W4, and L2-dispatch
capabilities. The flash-specialized object is currently rejected for lacking a
K3 marker, so those segments correctly fall back to the full interpreter.

The B8 grouped expert path is hybrid:

- gate/up: BF16 activation, native packed MXFP4 weight, software W4 decode,
  BF16 MFMA;
- SiTUv2 bridge: fused FP4 quantization of the 384-wide intermediate;
- down: FP4 activation and MXFP4 weight decode to BF16, BF16 MFMA;
- runtime weight requantization: none.

The bridge quantizes 49,152 values/layer, or 4,521,984 values over 92 MoE
layers per B8 step/rank. The shipped object contains no native FP4 or FP8 MFMA
in the active grouped wrappers.

## Critical path

The adopted B1 raw trace spans 51.468 ms over 2,459 packets. Major envelopes:

| family | time |
|---|---:|
| ordinary GEMV | 22.247 ms |
| grouped GLU + DOWN | 8.008 ms |
| KDA QKVG | 3.850 ms |
| TP collectives | 4.027 ms |
| KDA state/gated/conv | ~4.84 ms |
| router + combine | 4.534 ms |

Representative KDA+MoE and MLA+MoE layers each span about 0.54 ms. Host
enqueue is only about 4 us; interpreter depth, convergence, grouped expert
traffic, and per-layer dependencies dominate.

At T=8192, the historical 4.587 s prefill trace is dominated by KDA state
(1.333 s), Conv3 (0.636 s), routed GLU+DOWN (1.174 s), collectives (0.353 s),
and flash MLA (0.230 s). The TP host path also launched seven empty L2-domain
segments after the first launch had drained all domains; commit `1addc884`
removes those redundant launches while retaining barriers for true wave
segments.

## LDS and tuning

The gfx942 dense MXFP4 and grouped A4W4 bodies already use XOR-swizzled LDS
layouts. Prior BM32, BK32, and final-wave-culling experiments did not win;
another plain swizzle/tile retry is not justified without a new bank-conflict
trace. The live objects are at VGPR=256 and LDS about 64.56 KiB, so resource
cliffs are a first-order gate.

The MI325X tuning cell contains 686 records, but all are stale against the
current ROCm 7.14/source digest. It cannot select a production tile. New data
must be generated for `amd/gfx942/mi325x`, with interpreter-packet timing,
exact object/toolchain fingerprints, measured HBM/MFMA ceilings, and current
K3 shapes.

## Highest-value missing implementations

1. Grouped A8W4 FP8 MFMA. gfx942 has a measured ~1.89 PF/s FP8 MFMA ceiling vs
   ~0.94 PF/s BF16. A correct arm needs K32 E8M0 promotion, OCP/FNUZ
   correction, a per-row A8 scale contract, and two accumulator sets.
2. Weight-stream-aware grouped MoE. Decode discovers routed experts too late
   to prefetch during attention. W2 can be fetched during grouped GLU after
   routing; the per-XCD W1/W3 and W2 phase footprints fit separately in L2.
3. FP8 MLA flash specialization. Current K3 FP8 MLA prefill remains on the
   general interpreter; the existing flash class has no op110 body/routing.
4. Ragged expert work rather than padded BM changes. B8 has only 128 live
   routes across 896 experts; weight reads and padded expert tiles dominate.
5. Admission/pipeline work. Prefill chunks and decode are sequential inside a
   mux tick, ingress is unbounded, and fixed B8 computes parked rows. Local
   counter double buffers are allocated but the TP path always rearms bank 0;
   recurrent KDA/conv state and cross-rank counters are intentionally
   single-buffered.

## Lean and numeric gates

Lean currently checks packet ordering, address reuse, tile/LDS bounds, and
abstract framing. It does not prove K3 arithmetic, quantization, layouts,
collectives, cache indexing, or recurrent-state transitions. Shipping
hand-written devgen fusions are not derived from the egglog result.

Therefore every new precision/fusion arm still needs an independent hardware
oracle. Highest-value missing gates are one-shot vs chunked prefill state/logit
equivalence, parked-slot resume, long-context MLA position/FP8-scale history,
final output-AttnRes, real grouped routing, and TP1-vs-TP8/full-model logits.
