# Hopper W8A8 decode screen

Standalone experiment; no runtime or default changes. It compares the existing
opt-in W8A16 dispatcher against native FP8 WGMMA at Gemma's emitted slice counts.
Two consumer warpgroups tile 128 output channels per CTA, with 8/16 requests as
the WGMMA N dimension. Double-buffered `cp.async` staging overlaps the next
128-element K panel with computation. Both warpgroups drain before a shared
stage can be overwritten.

This uses native `QGMMA.64x8/16x32.F32.E4M3.E4M3`, verified in generated SASS.
The previous FP8 `mma.sync` probe is unsuitable: Hopper emulates that instruction
using conversions and BF16 MMA. The previous native M1 WGMMA experiment exposed
serialized staging latency; this screen uses two warpgroups and pipelining.

Activation quantization calls existing `d_quant_fp8` with
`PLOW_NV_QUANT_FP8_VLLM=1`: dynamic per-token absolute maximum, true division,
scale floor `1/(448*512)`, and clamped E4M3 rounding. Byte and scale agreement are
checked against a host implementation. Quantization runs once per projection,
or once for the combined Q/K/V launch. Timings report both projection-only and
quantization-plus-projection latency.

Two accumulation variants separate activation loss from arithmetic effects:
fast WGMMA accumulation and promotion into an FP32 shadow after every 128 K
elements. A sampled FP64 oracle consumes the *quantized* activation bytes;
full-output comparisons against W8A16 measure the combined change. GEMV and
GELU GLU preserve FP32 scales/activation until final BF16 output. These comparisons
cannot establish model quality.

The screen includes isolated Q/K slices (66/33 CTAs), O/down/GLU (132 CTAs),
and one combined Q/K/V launch with 66/33/33 of the 132 CTAs. The latter preserves
simultaneous projection occupancy and reuses quantized inputs, but does not
simulate the persistent interpreter's complete schedule. The combined output
buffer concatenates separate Q, K and V matrices.

Build from the repository root:

```sh
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin /usr/local/cuda/bin/nvcc \
  -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 \
  -I runtime/common -I runtime/nvidia \
  runtime/nvidia/experiments/gemma_fp8_w8a8_decode.cu -o /tmp/gemma_fp8_w8a8_decode
nix develop -c env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
  /tmp/gemma_fp8_w8a8_decode
```

Optional arguments are `M N K blocks glu`, with M=1..16, K divisible by 16 and
glu=0/1. Exact integer/tail/ownership tests always run first. The CUDA13.2 build
needed explicit `compute_90a` generation: plain `-arch=sm_90a` also emitted a
generic sm90 target which rejected WGMMA. The failed compiler log is retained.

Dynamic shared memory including alignment is 35/37KiB for GEMV and 67/69KiB for
GLU at 8/16 rows. This exceeds the existing decode arena; any future integration
requires explicit resource and lifecycle qualification. The new helper is
intentionally limited to standalone testing until measured gains justify that
work and the activation precision change.

H100 80GB screening on 2026-09-09: all 78 checks pass, including exact integer
outputs on owned slices, output guards, empty slices, partial rows and K tails.
Quantized bytes and scale bits match the host vLLM-style implementation. All
large-shape outputs are finite. Sampled FP64-oracle relative L2 error is at most
0.001103 with promotion, versus 0.004138 with fast accumulation. Relative L2
change from BF16-input output is 2.55–2.59% for GEMV and 3.60–3.68% for GLU;
activation quantization dominates that difference. These are synthetic operands.

Median microseconds, 15 measurements with 256MiB cache flush and four warmups.
W8A8 below includes quantization and FP32 promotion. No other GPU jobs or CPU
builds overlapped timing.

| Projection | N × K | CTAs | B8 W8A16 → W8A8 | B16 W8A16 → W8A8 |
|---|---:|---:|---:|---:|
| Q | 8192 × 5376 | 66 | 246.912 → 52.672 | 263.392 → 63.616 |
| K or V | 4096 × 5376 | 33 | 252.000 → 45.088 | 268.704 → 57.088 |
| Combined Q/K/V | (8192+4096+4096) × 5376 | 66+33+33 | 248.832 → 61.312 | 262.848 → 73.184 |
| O | 5376 × 8192 | 132 | 180.000 → 64.064 | 190.656 → 83.744 |
| GELU GLU | 21504 × 5376 | 132 | 370.592 → 131.744 | 431.520 → 147.488 |
| Down | 5376 × 21504 | 132 | 491.968 → 159.200 | 526.144 → 211.744 |

Including quantization, promoted W8A8 is 2.28–5.59× faster in these isolated
cells; combined Q/K/V is 4.06×/3.59× faster at B8/B16. This does not establish
model or serving speedups. The existing one-CTA quantizer adds about 98µs for
B16 down, almost as much as its 114µs projection, so quantization placement
matters for integration. Memcheck and actual-model qualification are pending.

The next proposed experiment is CTA-local per-token scale reduction and panel
quantization, retaining instruction ownership and launch structure. It repeats
quantization across slices and GLU output tiles but avoids new packet tensors,
dependencies and global scratch. Ordinary shared-memory quantization stores
must be fenced into the asynchronous proxy before WGMMA reads them. Arena growth
would use the existing `plow_arena_bytes` contract. A separate parallel per-token
quantization stage would avoid repeated computation and share Q/K/V inputs, but
requires compiler dataflow/scratch integration; it cannot be injected as an
ordinary launch inside the persistent kernel. Neither is implemented here.

Campaign build command/logs:
`/opt/dlami/nvme/tmp/gemma31-glm53-h100-20260908/gemma-fp8-w8a8-decode-build-command.json`,
`gemma-fp8-w8a8-decode-build.log`, and `gemma-fp8-w8a8-decode-build2.log`.
Measured results are `gemma-fp8-w8a8-decode-{first.log,screen.log,screen.json}`.
