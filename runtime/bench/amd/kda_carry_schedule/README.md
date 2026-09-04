# KDA carry schedule screen

Isolated gfx950 screen for a dense BT64 gated-delta carry with runtime `T,H` and
`D=128,V=128`. Selection depends only on dimensions. It compares the current
q-precomputed V16/WG512 body with:

- V8/WG256: four waves and 192 independent `(head,V8)` items; the upper eight
  MFMA columns are masked.
- V16/WG256: four waves and 96 items, isolating workgroup size from V8 coverage.
- V32/WG512: all eight waves own one `(token16,value16)` MFMA tile.
- V32/WG256: four-wave occupancy probe, rejected before GPU use if it spills.
- V32 staged/WG512: padded state/value rows and one per-chunk LDS copy of W, Q,
  U, Aqk, and the BF16-high/residual key factor. This preserves the current two
  state-update MFMAs and their order.
- Tail overlap/WG512: removes the phase-2/phase-3 workgroup barrier and maps the
  state-update halves to the four otherwise-idle phase-2 waves.
- Key stage/WG512: computes the exact BF16-high/residual key factors and FP32
  decay once per chunk into LDS. The combined arm also applies tail overlap.

The oracle requires bit equality for the complete BF16 output and FP32 final
state, and verifies that Aqk is unchanged. `run.sh` rejects private memory or
register spills before acquiring one GPU.

Run inside `nix develop`:

```sh
runtime/bench/amd/kda_carry_schedule/run.sh
```

## MI355X result (2026-09-04)

Exact shape: `T8192,H12,D128,V128,BT64`, 21 order-rotated samples.

| arm | median | vs control | gfx950 resources |
| --- | ---: | ---: | --- |
| control V16/WG512 | 1.920898 ms | -- | VGPR 204, SGPR 68, occupancy 2, no spill |
| V8/WG256 | not run | rejected | VGPR 150, SGPR 106, occupancy 3, 23 SGPR spills |
| V16/WG256 | 2.829905 ms | +47.3% | VGPR 152, SGPR 86, occupancy 3, no spill |
| V32/WG512 | 2.674145 ms | +39.2% | VGPR 180, SGPR 82, occupancy 2, no spill |
| staged V32/WG512 | 2.977387 ms | +55.0% | VGPR 174, SGPR 83, occupancy 2, no spill |
| V32/WG256 | not run | rejected | VGPR 143, SGPR 106, occupancy 3, 36 SGPR spills |
| tail overlap/WG512 | 2.104419 ms | +9.6% | VGPR 204, SGPR 64, occupancy 2, no spill |
| key stage/WG512 | 2.071938 ms | +7.9% | VGPR 134, SGPR 68, occupancy 3, no spill |
| key stage + tail overlap/WG512 | 2.046818 ms | +6.6% | VGPR 134, SGPR 66, occupancy 3, no spill |

All three launched candidates matched all 12,582,912 BF16 output elements and
196,608 FP32 state elements bit-for-bit; Aqk remained unchanged. Halving the
independent block count from 96 to 48 costs more than activating the otherwise
idle waves recovers. WG256 also regresses at unchanged V16 coverage; its extra
occupancy does not repay splitting each state-update tile across loop iterations.
The 192-item V8 coverage arm cannot cross the zero-spill gate. Extra staging adds
barriers and LDS traffic, and its 120,320-byte dynamic allocation does not help
this sequential carry. Those five candidate schedules are rejected; none should
be routed into production or pay the 69 specialist-segment launches.

The three follow-up schedules are also bit-exact across 12,582,912 BF16 output
elements and 196,608 FP32 final-state elements. The control median in their
21-sample order-rotated run was 1.919937 ms. Removing the barrier alone exposes
less wave-level overlap than it removes synchronization, while the 47,616-byte
key stage trades duplicated exponentiation for LDS writes, reads, and another
barrier. The fastest follow-up remains 0.126881 ms slower than control. The
`>=10%` promotion gate therefore fails; there is no packet or network arm.

The remaining architectural candidate is AITER K5's register-resident state
layout, not another launch/grid tweak. It holds the `[V,K]` state as MFMA
accumulators across the serial chunk loop, writes a BF16 snapshot to padded LDS,
and uses gfx950 `ds_read_b64_tr_b16` for `W @ state`; it also pipelines the next
W tile while the current chunk runs. Adapting that mechanism while retaining
Plow's fused output phase requires a new four-wave lean object and a state-layout
oracle. It is a significant kernel rewrite, but it removes the current
LDS-resident-state/live-range floor without materializing the rejected 50.33 MB
global key-factor pair.
