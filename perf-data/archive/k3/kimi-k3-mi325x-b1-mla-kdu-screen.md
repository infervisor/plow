# Kimi-K3 B1 FP8 MLA KDU screen on MI325X

## Decision

Keep `FA_MLA_KDU=4`. Do not promote KDU1, KDU2, or KDU8 to a production interpreter A/B.
KDU8 is the fastest isolated body, but its maximum projected saving is only **0.238 ms/token**
at 128K and **0.036 ms/token** at short context. That cannot materially move the B1 target.

The screen reinforces the promotion pipeline: compile/resource bisect, exact full-grid single-op
measurement, then TP8 serving only when the isolated projection is multi-millisecond. A literal
one-workgroup timing would not preserve K3's 192 live MLA work items on a 304-CU device.

## Hypothesis

The exact B1 FP8-KV interpreter is 255 VGPR with 624 B private state and three VGPR spills.
A compile-only live-op bisect localized the only resource movement to op 109,
`FlashMlaDecodeFp8`: removing it gave 250 VGPR, 576 B private state, and zero VGPR spills.
Removing Gemv, GemvGlu, grouped GLU/DOWN, RouterTopk, AttnRes, GemvQkvg, or KdaStateStepG one at
a time left the resource report unchanged.

`FA_MLA_KDU` controls unrolling of the FP8 latent and rope dot loops. The shipped CDNA3 value is
4. Values 1/2 reduce register pressure; 8 raises memory-level parallelism at higher register cost.
The arithmetic loop order and packet are unchanged.

## Full-grid screen

The retained harness runs the production `d_flash_mla_decode<512,64,GF4,FP8>` body at:

- 12 TP-local heads, GF4, nsplit=64;
- 192 live work items over a 304-block/512-thread grid;
- contexts 149, 8192, 32768, and 128000;
- FP8 ckv, BF16 krot, per-row ckv scales;
- a 512 MiB device cache flush before every timed launch;
- 21 alternating samples and bit-identical full-output comparison.

Each row below is a matched KDU4 control vs candidate run. The last column projects one body
delta over all 24 MLA layers.

| KDU | ctx | KDU4 us | candidate us | speedup | projected save ms/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 149 | 15.960 | 13.720 | 1.1633x | 0.054 |
| 1 | 8192 | 43.399 | 46.519 | 0.9329x | -0.075 |
| 1 | 32768 | 71.879 | 77.079 | 0.9325x | -0.125 |
| 1 | 128000 | 252.397 | 279.436 | 0.9032x | -0.649 |
| 2 | 149 | 15.800 | 14.400 | 1.0972x | 0.034 |
| 2 | 8192 | 43.159 | 52.119 | 0.8281x | -0.215 |
| 2 | 32768 | 71.719 | 83.639 | 0.8575x | -0.286 |
| 2 | 128000 | 252.277 | 304.916 | 0.8274x | -1.263 |
| 8 | 149 | 15.839 | 14.320 | 1.1061x | 0.036 |
| 8 | 8192 | 43.199 | 39.479 | 1.0942x | 0.089 |
| 8 | 32768 | 71.919 | 69.199 | 1.0393x | 0.065 |
| 8 | 128000 | 252.597 | 242.677 | 1.0409x | 0.238 |

Every candidate output and LSE partial was bit-identical to KDU4 at every context.

Standalone resource reports for `k_mla_decode_fp8`:

| KDU | VGPR | SGPR | private bytes | VGPR spills |
|---:|---:|---:|---:|---:|
| 1 | 244 | 46 | 0 | 0 |
| 2 | 248 | 47 | 0 | 0 |
| 4 | 256 | 48 | 60 | 16 |
| 8 | 256 | 44 | 52 | 12 |

Lower register count is not the objective by itself: KDU1/2 lose the latency-hiding required by
the long FP8 KV stream. KDU8 pays more resources but wins slightly; the absolute body share is too
small to justify production integration.

## Reproduction

```sh
nix develop --command cmake -S runtime -B /tmp/plow-k3-mla-kdu \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-mla-kdu \
  --target k3_mla_kdu_sweep -j4

for kdu in 1 2 8; do
  nix develop --command env GPU_LEASE_DIR=/tmp/plow-gpulease-shared \
    perf-data/tools/gpulease -n 1 "k3-mla-kdu-$kdu" \
    /tmp/plow-k3-mla-kdu/bench/k3_mla_kdu_sweep \
    /tmp/plow-k3-mla-kdu/bench/k3_mla_kdu4_gfx942.co \
    /tmp/plow-k3-mla-kdu/bench/k3_mla_kdu${kdu}_gfx942.co 21
done
```

Host SHA256: `b8e2b79e792ba3f46f77cad210902bc588628beb81fa8aba49f617a44e01f156`.
Device bundle SHA256, KDU1/2/4/8:

- `5009a02eb8448bb27878ad57c71db72968e2132ea2a5b513796548294f69d78a`
- `125eb4d42577d46f49fffbc71a45c08885a43f74858972b1293c8bc80452db1e`
- `e20a690b69fc27032a10d1213362b273fa00fa5076dcedca0b372bcdf0afa7`
- `fef1b5e2246a8eaac63a93c9d7e3e9db0343a54908bf8a85e57d89d383244f84`

Toolchain: HIP 7.14.60850, AMD clang 23.0.0git. Post-run GPU audit was clean.
