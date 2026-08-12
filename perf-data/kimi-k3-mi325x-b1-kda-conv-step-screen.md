# Kimi-K3 B1 KDA Conv3/state-step screen

Date: 2026-08-11. Hardware: one MI325X (`gfx942`). Toolchain: repository Nix ROCm 7.14.0.

## Question

Can KDA's `KdaConv3 -> KdaStateStepG` pair become one packet without first implementing the
double-buffered recurrent convolution-state contract?

The shipping B1 graph spends this pair in 69 layers. Folding it naively is incorrect: each value
tile needs the old q/k convolution window, while an unordered workgroup could overwrite that
window before its siblings read it. A sound fusion reads one window bank, writes the other, and
swaps banks only after the packet completes.

Two screens bound the decision:

1. An interpreter program deletes the Conv3 packet but retains the exact shipping state-step and
   norm bodies. Its pre-populated convolution output is a protocol lower bound, not a correctness
   candidate.
2. A standalone exact-shape kernel implements the double-buffered transition over TP8
   `H=12,D=128,BV=8,W=4`, then compares all 69 layers against the shipping two-kernel spelling.

## Interpreter lower bound

| Chain | Packets/layer | 69-layer median |
|---|---:|---:|
| Original six-op spelling | 6 | 18.151 ms |
| Shipping Conv3 + StateStepG + norm | 3 | 9.829 ms |
| StateStepG + norm protocol floor | 2 | 6.305 ms |

Deleting the Conv3 packet has a maximum measured value of **3.525 ms/token**. This agrees with
the current full-model dependency trace: Conv3 carries 4.752 ms of observed-spine charge, of which
only 0.759 ms is its body.

## Double-buffered kernel screen

The standalone candidate assigns one workgroup to `(head, BV-value tile)`. Each workgroup reads
q/k/v convolution windows from the old bank. One tile writes the new q/k windows; every tile owns
and writes its disjoint v-window columns. The recurrence reads only locally computed convolution
values, so no workgroup can observe another workgroup's new bank.

| 69-layer chain | Median |
|---|---:|
| Conv3 control + StateStepG control | 0.512 ms |
| Double-buffered combined kernel | 0.375 ms |
| Standalone body/launch saving | 0.137 ms |

The isolated timing intentionally does not model interpreter counters. Its purpose is resource and
numeric feasibility; the interpreter lower bound prices the packet removal.

The candidate has 27 VGPR, 46 SGPR, 1,632 B dynamic LDS, zero private memory, and zero spills.
The standalone control state step has 45 VGPR. The combined body is therefore safe to compose at
the kernel level; the production megakernel remains a separate hard gate.

The double-buffered convolution state is byte-identical. Inlining the convolution changes four
BF16 values out of 317,952 through compiler reassociation; errors remain bounded:

| Output | Relative L2 | Max absolute |
|---|---:|---:|
| Convolved BF16 mix | 8.12e-6 | -- |
| Recurrent f32 state | 9.17e-7 | 2.26e-6 |
| BF16 state-step output | 1.45e-5 | -- |

There are no non-finite values. Production promotion therefore requires the existing real-weight
layer oracle and full-logit gate; byte identity cannot be claimed by construction.

## Reproduction

```bash
nix develop --command ./scripts/build_kda_fuse_bench.sh \
  /tmp/plow-kda-fuse-gfx942-final
nix develop --command bash -lc \
  'perf-data/tools/gpulease -n 1 kda-conv-step-floor-final \
   /tmp/plow-kda-fuse-gfx942-final/kda_fuse_bench \
   /tmp/plow-kda-fuse-gfx942-final/interp_decode_k3.elf 12 8 69 80 \
   | tee /tmp/kda-conv-step-floor-final.txt'

nix develop --command cmake -S runtime -B /tmp/plow-k3-kda-db-build \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-kda-db-build \
  --target k3_kda_conv_step_db -j2
nix develop --command bash -lc \
  'perf-data/tools/gpulease -n 1 kda-conv-step-db \
   /tmp/plow-k3-kda-db-build/bench/k3_kda_conv_step_db \
   /tmp/plow-k3-kda-db-build/bench/k3_kda_conv_step_db_gfx942.co \
   | tee /tmp/k3-kda-conv-step-db.json'
```

Evidence SHA256:

- interpreter: `34449174abcbcdd31250ffc405ba3ac62742b56fb06eb579d9a4d19b3375380c`
- floor output: `6f7f37d85bf940a1c88e8f766277dbb265b40ea7701f53c680fefc9f4d15824b`
- candidate bundle: `5c4010676b7eff7293fea9d50a83ca0b37b973d48d5bb7c5e79add8dee04fdf9`
- candidate JSON: `64e1f67841ba94c615917693b1d74927b9bd2092f61b643f34251fa3bff0e49e`

## Decision

Proceed to a default-off B1 production experiment. The isolated candidate clears resource and
numeric gates, while the interpreter establishes a 3.5 ms ceiling large enough to matter. The
production design must add an explicit two-bank convolution-state transition and refuse assets
without the matching capability marker. Accept only if the real TP8 packet removes 69 packets,
the megakernel does not increase spills, the real-weight/full-logit gates pass, and served TPOT
improves by at least 2 ms.
