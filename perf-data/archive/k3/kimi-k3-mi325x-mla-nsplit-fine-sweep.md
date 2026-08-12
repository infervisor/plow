# Kimi-K3 B1 MLA nsplit fine sweep (ns96 rejected)

Date: 2026-08-11. Hardware: 8x MI325X, TP8, gfx942. Toolchain: repository Nix
ROCm 7.14.0. Client: vLLM 0.27.0 `bench serve`, one warmup.

## Screen

`k3_mla_gf_sweep` runs the exact K3 B1 FP8 MLA decode body followed by its merge-fold
body. It uses 12 heads, latent width 512, rope width 64, value width 128, the production
304-block launch, and contexts through 128K. This is a model-free full-grid screen, not a
one-block timing: `ns` changes both live workgroups and merge work, so a one-block result
cannot represent its occupancy boundary.

The sweep covered GF4/GF6/GF12 and ns32/64/72/80/88/96/112/128/144/152/160/192. GF4
ns96 was the isolated 128K winner. Its 288 live work items fit on 304 CUs; ns112 creates
336 and crosses into a second wave.

| Context | GF4 ns64 | GF4 ns88 | GF4 ns96 | ns96 vs ns64 |
|---:|---:|---:|---:|---:|
| 8K | 0.0484 ms | 0.0536 ms | 0.0540 ms | -10.3% |
| 32K | 0.0768 ms | 0.0741 ms | 0.0736 ms | +4.3% |
| 64K | 0.1343 ms | 0.1142 ms | 0.1129 ms | +18.9% |
| 128K | 0.2498 ms | 0.1958 ms | 0.1894 ms | +31.9% |

GF6 was exact but slower and materially more resource-heavy. GF12 used 54,848 B LDS,
113 spills, and failed the output comparison. It is rejected.

Build and run:

```bash
nix develop --command cmake -S runtime -B /tmp/plow-k3-mla-gf-build \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-mla-gf-build \
  --target k3_mla_gf_sweep -j2
nix develop --command bash -lc \
  'perf-data/tools/gpulease -n 1 k3-mla-gf-fine \
   /tmp/plow-k3-mla-gf-build/bench/k3_mla_gf_sweep \
   /tmp/plow-k3-mla-gf-build/bench/k3_mla_gf_sweep_gfx942.co \
   > /tmp/k3-mla-gf-fine.csv'
```

The raw CSV SHA256 is `b4aa36b97e4ee2f3ccdda42bb1df32525309795603ca2c07069772471e4077e4`.

## Full-model gate

The candidate was a packet-only change: all 24 decode `FlashMlaDecodeFp8` packets use
GF4/ns96 and 288 blocks instead of GF4/ns64 and 192 blocks. The interpreter objects,
weights, tensor layouts, precision, and runtime flags were unchanged. Candidate packet
SHA256: `3bdde118010e4b900ba5e5af83e3b37027c22f8494d060ddbe49da9c3fda8407`.

| Served cell | Adopted ns64 | Candidate ns96 | Delta |
|---|---:|---:|---:|
| actual input 149, output 512 | 53.387 ms TPOT | 55.268 ms TPOT | +1.881 ms (+3.52%) |
| actual input 128,021, output 128 | 60.569 ms TPOT | 60.007 ms TPOT | -0.563 ms (-0.93%) |

Both ns96 cells completed 1/1, produced the requested output length, returned empty errors,
and passed compact exact TP counter/rank auditing. Detailed candidate JSON remains at
`/tmp/k3-ns96-result/{short,long}.json`; SHA256 is
`086b51f3373ac7834b18df1bb57021214935063a311aecea8d76a7f7fcece10e` and
`4960cd712abf81e8d8ff40b7af737d260dde1d1da78f0c502b513ec89056c2a1`.

## Decision

Keep ns64. Ns96 loses clearly at short context and saves only 0.56 ms at 128K. The
intermediate ns72/80/88 arms form the expected continuum and do not provide a universal
winner. A context-dependent packet selector is not justified by a sub-millisecond endpoint
gain while B1 still needs 33--41 ms removed to reach 20 ms/token. The harness remains as the
cheap first gate for future MLA-body changes; only full-grid winners advance to TP8 serving.
