# Kimi-K3 MI325X batched-decode collective aggregation

Date: 2026-08-11. Hardware: 8x MI325X, TP8, gfx942. Toolchain: repository
Nix ROCm 7.14.0. Client: vLLM 0.27.0 `bench serve`, one warmup.

## Packet analysis

The B32 decode program has 2,942 packets and 73,550 local counters. Its 186
`XReduceTwoShot` packets comprise 94 packets at 304 workgroups and 92 at 224,
for 49,184 workgroups total. Every packet uses band offset zero, so the
collective begins only after its complete producer packet.

The packet graph already overlaps independent work around collectives. Router
score, latent projection, and shared gate/up start from the same post-attention
value. The routed reduction feeds both routed-up and shared-down's protected
partial-slot reuse. The final MoE collective folds the routed column gather
into the shared-down reduction, avoiding a separate packet and rendezvous.

The two-shot payload is 32,112,640 BF16 elements across the program. Its
reduce-scatter plus all-gather traffic is about 107.19 MiB/rank/step, or
857.50 MiB across TP8. This experiment does not change those bytes.

The missing decode build flag was `PLOW_XR_AGG`. Without it, every workgroup
issued one system-scope signal to every rank at the second rendezvous:

```text
unaggregated remote gate RMWs/rank/step  393,472
aggregated remote gate RMWs/rank/step      1,488
aggregated local arrivals/rank/step       49,184
remote RMW reduction                       99.622%
```

The closing local workgroup now issues one signal/rank carrying the unchanged
`nblk` count. Packet fields, counter targets, host audit expectations, tensor
addresses, and collective arithmetic remain unchanged. The candidate exports
`plow_xr_agg_1`; the control does not.

## Served A/B

Matched B32 objects differed only by `PLOW_K3_DECODE_XR_AGG`. Both used the
same packet, FP8 KV, native MXFP4 weights, router-local selection, parallel
ALIGN prefix, binary tile search, exact B1/B2/B4/B8 low-rung objects, compact
TP audit, and counter double buffering.

| seed | control tok/s | aggregate tok/s | change | control mean TPOT | aggregate mean TPOT | change |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 145.862 | 150.210 | +2.98% | 198.749 ms | 192.700 ms | -3.04% |
| 1 | 144.945 | 149.521 | +3.16% | 200.251 ms | 193.940 ms | -3.15% |

Each long cell was C32/N32, random input 32, output 512, greedy, and produced
16,384/16,384 output tokens with 32/32 completed requests, zero failures,
empty error strings, and no compact TP audit failure.

A separate 32-identical-prompt/output128 gate improved 109.323 to 111.345
tok/s (+1.85%). Each arm produced the exact same byte-level response multiset
with multiplicities `[30,1,1]`. The three variants were assigned to different
request positions, matching the existing admission/slot scheduling
nondeterminism recorded by the tile-search campaign; no new response appeared.

## Static gates

Control hashes:

```text
53562abacdd00ed9169aeeac4244278ed07b7ed2e2bbbf2a805a58d362b90115  interp_decode_fp8kv_k3.elf
838b8ebc885a1b9969bc9f9fca0383c859eb690c45050a76db2818f0cda61b18  interp_decode_fp8kv_k3_gq.elf
```

Aggregated hashes:

```text
d6a7bab70246301886113e3cb97c4fa6a71144be40a598a4b855b1189e45924c  interp_decode_fp8kv_k3.elf
eb2d55cd91223813029564c147dca00defc3ff1a10abc9c177a85cdbc1bb8b07  interp_decode_fp8kv_k3_gq.elf
```

Both arms remain VGPR=256, SGPR=108, private=6,364 B, and LDS=64,560/64,568
B. The generic gfx942 and grouped-A4W4 ISA audits pass. Script and CMake paths
use the same opt-out/default-on contract; `PLOW_K3_DECODE_XR_AGG=0` restores
the measured control.

Raw evidence:

```text
/tmp/k3-b32-xragg-ab/control/out512-seed{0,1}.json
/tmp/k3-b32-xragg-ab/candidate/out512-seed{0,1}.json
/tmp/k3-b32-xragg-ab/{control,candidate}/identical-out128.json
/tmp/k3-b32-xragg-ab/{control,candidate}-server.log
```

## Next pipeline experiment

All 186 collectives still use band offset zero. B32 ordinary GEMV already
walks two MM16 row groups inside one packet, but the first 16 rows cannot begin
their transfer while the second 16 compute. The packet ABI and interpreter
already support banded two-shot collectives. A K3-specific two-band emitter A/B
is the next bounded experiment, but it adds a rendezvous per seam and must be
measured after aggregation rather than assumed to win.
