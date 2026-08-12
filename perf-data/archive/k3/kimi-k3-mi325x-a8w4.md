# Kimi-K3 MI325X grouped A8W4 probe

Date: 2026-08-11. Hardware: one leased MI325X (`gfx942`, 304 CU).
Toolchain: flake ROCm 7.14.0. This is a standalone kernel experiment; it does
not change production packets or interpreter dispatch.

## Implementation

The probe keeps checkpoint weights in packed MXFP4+E8M0 form, expands FP4
mantissas to FP8 in LDS, and uses per-row FP8 activation scales. Two K16 FP8
MFMAs form each K32 weight-scale block; the temporary accumulator is promoted
by E8M0 scale and the gfx942 OCP/FNUZ correction before entering the final FP32
accumulator.

- Tile: BM64 x BN256 x BK64.
- ISA: 8x `v_mfma_f32_32x32x16_fp8_fp8` per kernel loop.
- Resources: 117 VGPR, 20,480 B LDS, zero scratch, occupancy 4.
- Oracle: f64 dequantized OCP-E4M3 activation x MXFP4 weight, including OCP
  negative zero and K32 scale-boundary cases.

The exact K3 TP8 fixture uses H=3584, I=384, E=896, top-16 routing, 64 live
rows, 1024 BM64-padded rows, and grid 304.

## Result

Twelve timing samples, four launches/sample:

| arm | GLU | DOWN | total |
|---|---:|---:|---:|
| current A4 bridge / BF16 MFMA | 0.19754 ms | 0.12665 ms | 0.32419 ms |
| A8W4 / FP8 MFMA | 0.2609 ms | 0.1330 ms | 0.3939 ms |

A8W4 is **21.5% slower**: GLU +32.1%, DOWN +5.0%. The A8W4 GLU f64 RMS is
1.633e-3; DOWN RMS is 3.572e-8. Both correctness gates pass. The current A4
control also passes its full bridge/down oracle.

This comparison excludes the production activation-quant packet required to
create per-row A8 input, so it is favorable to A8W4. Adding that work cannot
reverse the result.

## Verdict

Reject production integration of this A8W4 arm. The grouped K3 shape is
limited by packed-W4 expansion, K32 scale promotion, expert padding, LDS
staging, and weight traffic rather than MFMA peak. The measured ~2x gfx942 FP8
MFMA ceiling does not transfer to this kernel.

The next grouped-MoE experiment should reduce selected-expert weight traffic or
padding without increasing weight rereads. W2 lookahead after routing is more
promising than another arithmetic-format substitution.
