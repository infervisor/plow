# gfx950 grouped-MoE decode qualification (2026-09-04)

## Scope and inventory

The D10 inventory-pruned B=1 decode packet has one 2,165-instruction global-queue program.
It contains 92 adjacent `MoeGroupGluFp8Blk` + `MoeGroupDownFp8Blk` pairs, each emitted at
256 workgroups. The exact geometry is top-k 16, hidden 3584, intermediate 384, 896 experts,
MXFP4 weights, and SiTU with beta 4 / linear-beta 25. The surrounding order is GEMV, router,
GLU, DOWN, fixed-order combine, then XReduce.

The authoritative D10 trace attributes 2.29318 ms/token to grouped GLU and 2.28741 ms/token
to grouped DOWN: 4.58059 ms/token, or 49.789 us/layer. The inventory interpreter is wave64,
512 threads, 248 VGPR, 106 SGPR, 440 B private memory/thread, occupancy 2, and about 147.5 KiB
LDS/workgroup.

## Transfer and isolated gate

AITER's tuned B=1 H3584/I384/E896/top16 A8W4 entries use four-wave BM32/BK256 kernels
(10.664 us stage 1 and 9.459 us stage 2). Its activation quantization changes Plow's arithmetic,
so only the standalone-family ownership pattern was transferred. AITER's A16W4 stage 1 also
failed its own tolerance gate (34.6% error) and was not used.

The Plow single-pair harness uses the emitted tensor sizes and arithmetic. It gates full `fu` and
f32 partial buffers against the current device bodies before timing. Weight traffic per pair is
35,094,528 B. All tested variants had zero full-buffer difference.

| standalone body | grid | GLU+DOWN (us) | projected 92 layers (ms) |
|---|---:|---:|---:|
| 8 waves, linear | 256 | 25.172 | 2.316 |
| 4 waves, linear | 768 | 19.392 | 1.784 |
| 4 waves, XCD4 swizzle | 768 | 21.736 | 2.000 |
| 8 waves, linear | 768 | **16.776** | **1.543** |

The selected kernels compile without scratch or spills. Shipping-object metadata is:

| kernel | VGPR | SGPR | occupancy | LDS | private |
|---|---:|---:|---:|---:|---:|
| GLU | 79 | 77 | 6 | 0 | 0 |
| DOWN | 94 | 74 | 5 | 0 | 0 |

The isolated chain includes both ordered kernel launches. Against the authoritative body it saves
33.013 us/layer, or 3.037 ms/token before segment transitions. The pure-pair route adds 276 decode
segment transitions; at the measured 1.458 us ordered-AQL floor that is about 0.402 ms/token, leaving
about 2.63 ms/token projected margin. End-to-end serving remains an opt-in qualification gate.

## Route

`PLOW_MOE_DECODE_STANDALONE=1` makes only an exact adjacent grouped GLU+DOWN pair a pure decode
segment. The host validates MXFP4 geometry, tensor continuity, absence of interpreter-counter
obligations, unique segment ownership, object marker, kernarg sizes, and zero private memory. It
then launches the unchanged eight-wave device bodies in order at `3 * packet n_cu`. No model name
or model-specific predicate is used. The route and its CMake object are default-off.

Evidence files: `/tmp/inv-disasm-clean.json`,
`/tmp/k3-inventory-qualified-65fca83/fold1-candidate.trace.report`,
`/tmp/k3-moe-w8-v2.csv`, `/tmp/k3-moe-w4-v2.csv`, and `/tmp/k3-moe-w4-xcd4.csv`.
