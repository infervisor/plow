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

All three launched candidates matched all 12,582,912 BF16 output elements and
196,608 FP32 state elements bit-for-bit; Aqk remained unchanged. Halving the
independent block count from 96 to 48 costs more than activating the otherwise
idle waves recovers. WG256 also regresses at unchanged V16 coverage; its extra
occupancy does not repay splitting each state-update tile across loop iterations.
The 192-item V8 coverage arm cannot cross the zero-spill gate. Extra staging adds
barriers and LDS traffic, and its 120,320-byte dynamic allocation does not help
this sequential carry. All five candidate schedules are rejected; none should
be routed into production or pay the 69 specialist-segment launches.
