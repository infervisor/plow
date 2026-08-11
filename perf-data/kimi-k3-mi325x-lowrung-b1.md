# Kimi-K3 MI325X: ladder-compatible low-rung objects

**Status: adopted for rungs 1, 2, 4, and 8.** The generic decode ladder selected
each packet correctly, but the canonical MM16+walk interpreter carried the
widest decode body. Dedicated exact-width objects keep the same packet, state
layout, counters, and model weights while reducing the compiled specialization.

The old single-rung B1 object was not compatible with the ladder packet: its
K3 decode program used per-expert opcodes and the ELF omitted
`plow_moe_pf_a4w4_arm`. Ladder B1 uses grouped ops 85/86 with MXFP4 encoding.
`PLOW_K3_DECODE_GROUPED=1` adds that exact grouped body to an MM1 object; the
loader and build script both require the capability marker.

## Build

All commands used the repository ROCm 7.14 Nix shell.

```bash
nix develop --command env \
  PLOW_DECODE_BATCH=1 \
  PLOW_K3_DECODE_GROUPED=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 \
  JOBS=2 \
  scripts/build_gfx942.sh \
  /home/lava/plow/build-amd/k3-b1-ladder-grouped
```

Both static and global-queue objects passed the gfx942 ISA/resource audit and
exported `plow_moe_pf_a4w4_arm`. The static object used 255 VGPR, 64,560 bytes
LDS, and one spill; GQ used 255 VGPR, 64,568 bytes LDS, and three spills.

The B2/B4/B8 objects use the same command without the B1-only grouped override:

```bash
for b in 2 4 8; do
  nix develop --command env \
    PLOW_DECODE_BATCH="$b" \
    PLOW_K3_DECODE_MXFP4_PROJ=0 \
    PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 \
    JOBS=2 \
    scripts/build_gfx942.sh \
    "/home/lava/plow/build-amd/k3-b${b}-ladder-grouped"
done
```

B2 used four spills in both objects; B4 and B8 used two. All remained at 256
VGPR and 64,560/64,568 bytes LDS and passed both the generic gfx942 and grouped
A4W4 audits.

## Served A/B

Both arms used the same 12-program ladder asset, checkpoint, prompt seed, and
runtime. The candidate added only the tiers covering the tested occupied extent:

```bash
PLOW_HSACO_LOWRUNG=/home/lava/plow/build-amd/k3-b1-ladder-grouped:1,/home/lava/plow/build-amd/k3-b2-ladder-grouped:2,/home/lava/plow/build-amd/k3-b4-ladder-grouped:4,/home/lava/plow/build-amd/k3-b8-ladder-grouped:8
```

The client was vLLM 0.27 `bench serve`, random input 32, output 512,
concurrency and request count equal to the tested width, one warmup,
temperature 0, ignore EOS. Compact TP counter audit and counter double
buffering were enabled.

| width | primary tok/s | tier tok/s | primary TPOT | tier TPOT |
|---:|---:|---:|---:|---:|
| 1 | 11.655 | 18.541 | 85.33 ms | 53.40 ms |
| 2 | 16.208 | 18.902 | 122.46 ms | 104.88 ms |
| 4 | 30.647 | 36.431 | 128.23 ms | 107.61 ms |
| 8 | 53.425 | 60.537 | 145.07 ms | 127.76 ms |

The tiers improve TPOT by 11.9--37.4% and output throughput by 13.3--59.1%.
Every arm completed its C1/C2/C4/C8 request set with 512 output tokens per
request, empty errors, and byte-identical generated-text arrays across A/B. Raw
detailed results:

- `/tmp/k3-primary-b1-c1-out512/seed0.json`
- `/tmp/k3-lowrung-b1-c1-out512/seed0.json`
- `/tmp/k3-primary-b2-c2-out512/seed0.json`
- `/tmp/k3-lowrung-b2-c2-out512/seed0.json`
- `/tmp/k3-primary-b4-c4-out512/seed0.json`
- `/tmp/k3-lowrung-b4-c4-out512/seed0.json`
- `/tmp/k3-primary-b8-c8-out512/seed0.json`
- `/tmp/k3-lowrung-b8-c8-out512/seed0.json`

This result changes only object selection through rung 8. B16 and aggregate
B32 remain on the canonical MM16+walk object and retain the 131.162 tok/s
control.
