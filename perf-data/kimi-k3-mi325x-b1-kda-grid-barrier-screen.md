# Kimi-K3 B1 KDA in-packet grid-barrier screen (rejected)

Date: 2026-08-11. Hardware: one MI325X (`gfx942`). Toolchain: repository Nix ROCm 7.14.0.

## Question

Can the numerically drifting double-buffered `Conv3 + StateStepG` candidate be replaced by a
bit-exact one-packet implementation? The candidate runs the shipping `d_kda_conv3` ownership and
arithmetic, performs a device-wide barrier, then runs the shipping `d_kda_state_step_g` body.
This preserves every BF16 materialization and f32 update exactly while deleting one host-visible
kernel boundary in the standalone model.

The grid barrier uses a reusable arrival/generation pair. Normal convolution stores require a
device visibility fence before publishing arrival and an acquire fence after observing the new
generation. Omitting those fences was caught immediately by the exact state comparison.

## Result

Final 69-layer, 12-sample medians:

| arm | time | delta from two-kernel control | exact state/output |
|---|---:|---:|---:|
| separate Conv3 then StateStepG | 0.542352 ms | -- | reference |
| existing double-buffer candidate | 0.373575 ms | -0.168777 ms | no |
| in-packet grid barrier | 1.886813 ms | +1.344461 ms | yes |

The barrier arm has zero differing convolution bytes, state bytes, and output bytes. Its kernel is
44 VGPR, 53 SGPR, zero private memory and zero spills. The two component kernels are 20/35 and
45/45 VGPR/SGPR; the fast double-buffer candidate is 27/46 with zero spills.

The existing fast candidate's drift is localized: eight convolution bytes differ across 69 layers,
then 32,438 f32 state bytes and six BF16 output bytes differ. Its state relative L2 remains
`9.172e-7`, but the prior full-model quality gate showed that this is sufficient to flip a near-tied
token and score 196/200 rather than the established 197/200.

## Decision

STOP. Do not add an internal grid barrier to the interpreter. Exact global visibility costs more
than three times the existing two-kernel chain. Keep the faster double-buffer arm explicit and
default-off; seek producer-side fusion that preserves ownership without a grid rendezvous.

Raw result: `/tmp/k3-kda-gridbar-final.jsonl`.

## Reproduction

```bash
nix develop --command cmake --build /tmp/plow-k3-kda-db-build \
  --target k3_kda_conv_step_db -j2
perf-data/tools/gpulease -n 1 k3-kda-gridbar-final \
  /tmp/plow-k3-kda-db-build/bench/k3_kda_conv_step_db \
  /tmp/plow-k3-kda-db-build/bench/k3_kda_conv_step_db_gfx942.co
```
