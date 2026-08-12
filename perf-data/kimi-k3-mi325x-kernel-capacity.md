# Kimi-K3 MI325X kernel and capacity rungs

Date: 2026-08-11. Hardware: 8x MI325X (`gfx942`, 304 CU/card). Toolchain:
flake-pinned ROCm 7.14.0. Client: flake-pinned vLLM 0.26.0
`bench serve`, OpenAI-chat. All serving runs enable compact exact TP audit.

## B8 dead projection arm

The live B8 packet has grouped A4W4 MoE packets but no standalone MXFP4
projection packet. `PLOW_K3_DECODE_MXFP4_PROJ=0` removes only those dead
projection bodies from the K3 decode object and retains the K3, FP8-KV,
grouped-A4W4, and L2-dispatch capability markers.

Static/GQ object changes:

- VGPR spills: 32 -> 2.
- Private segment: 2660 -> 2628 bytes.
- Object size: -17.5%.
- Static scratch instructions: -11.1%.
- VGPR=256 and LDS=64560/64568 bytes are unchanged.

Three C8/N32/out128 runs used the same seeds and eight warmups as the audited
control table. Every input length and generated text is exactly identical to
its matched control run.

| run | output tok/s | median TTFT (ms) | median TPOT (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|
| 1 | 51.012 | 826.99 | 148.33 | 480.05 |
| 2 | 50.837 | 827.75 | 148.08 | 480.61 |
| 3 | 50.862 | 824.51 | 147.57 | 479.70 |

Median = **50.862 tok/s**, +1.56% over the audited control median of
50.082 tok/s. Subsequent campaigns use one warmup.

## B16 capacity rung

The exact-width B16 packet has decode T=16 and matching `gemv_mm_cap_16`
objects. It fits easily in MI325X memory, but the combined decode object returns
to 32 VGPR spills. C16/N32/out128 with one warmup:

| run | output tok/s | median TTFT (ms) | median TPOT (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|
| 1 | 69.555 | 871.63 | 214.83 | 526.15 |
| 2 | 69.840 | 875.15 | 213.42 | 526.75 |

Both runs completed 32/32 requests and 4096/4096 output tokens with no client
or in-band error. Wider batching alone does not reach 100 tok/s; B16 needs a
substantial grouped-MoE/resource improvement.

## Context diagnostic

This diagnostic used B8 control, C8/N8/out32, no explicit warmup, exact input
range, and detailed error/output gates. It predates the one-warmup campaign
rule. Use `scripts/k3_context_sweep.sh` for the adopted three-seed protocol.

| requested context | output tok/s | median TTFT (ms) | median ITL (ms) | P99 ITL (ms) |
|---:|---:|---:|---:|---:|
| 128 | 30.073 | 2423 | 131.14 | 574.07 |
| 512 | 24.846 | 3428 | 130.90 | 803.94 |
| 1024 | 20.643 | 4600 | 132.43 | 1060.24 |
| 2048 | 15.582 | 6855 | 132.11 | 1560.70 |
| 4096 | 10.240 | 11673 | 132.82 | 2636.28 |
| 8192 | 5.640 | 23049 | 134.65 | 4767.02 |
| 16000 | 3.056 | 44514 | 141.01 | 5203.26 |
| 32000 | 1.375 | 101949 | 146.76 | 6346.73 |

Every cell completed 8/8 requests with exact 32-token outputs. Decode median
ITL stays nearly flat; TTFT growth is prefill/runtime scheduling. The mux runs
one lowest-slot prefill chunk per tick and only then a synchronous decode.
Cold C8 32k therefore serializes 32 chunks instead of packing pending rows.

Additional runtime findings:

- only the eight rank kernels overlap; staging, rearm, audit, and readback are
  hard host barriers;
- TP local counter double buffers are allocated but bank 0 is always rearmed;
  cross-rank counters and recurrent state are single-buffered;
- the TP prefill path redundantly submits seven empty L2-domain segments after
  the first dispatch already drained all eight domains (~0.35 ms/chunk).
