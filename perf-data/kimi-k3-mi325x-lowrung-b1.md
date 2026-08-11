# Kimi-K3 MI325X: ladder-compatible B1 object

**Status: adopted for rung 1.** The generic decode ladder selected its B1
packet correctly, but the canonical MM16+walk interpreter made one pass through
the packet while carrying the widest decode body. A dedicated MM1 object keeps
the same packet, state layout, counters, and model weights while reducing the
compiled decode specialization.

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

## Served A/B

Both arms used the same 12-program ladder asset, checkpoint, prompt seed, and
runtime. The candidate added only:

```bash
PLOW_HSACO_LOWRUNG=/home/lava/plow/build-amd/k3-b1-ladder-grouped:1
```

The client was vLLM 0.27 `bench serve`, random input 32, output 512,
concurrency 1, one request, one warmup, temperature 0, ignore EOS. Compact TP
counter audit and counter double buffering were enabled.

| arm | output tok/s | median TPOT | steady GPU drain |
|---|---:|---:|---:|
| MM16+walk primary object | 11.655 | 85.33 ms | about 82.0 ms/token |
| dedicated MM1 grouped object | 18.541 | 53.40 ms | about 50.2 ms/token |

The dedicated object improves TPOT by **37.4%** and output throughput by
**59.1%**. Both runs completed 1/1 with 512 output tokens, empty errors, and
byte-identical generated text. Raw detailed results:

- `/tmp/k3-primary-b1-c1-out512/seed0.json`
- `/tmp/k3-lowrung-b1-c1-out512/seed0.json`

This result changes only rung-1 object selection. Aggregate B32 remains on the
canonical MM16+walk object and retains the 131.162 tok/s control.
