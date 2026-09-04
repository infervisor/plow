# KDA carry register-resident state screen

Isolated gfx950 screen for the dense BT64 gated-delta carry at `D=V=128` with the
q-precomputed operand. It compares the shipping V16/WG512 body
(`d_kda_chunk_carry_bt64<true>`) with the four-wave register-resident carry in
`runtime/amd/op_kda_carry_regstate.h`:

- `regstate_v16_wg256`: one (head, V16) item per 256-thread workgroup; the f32 state lives
  in MFMA accumulators (the update MFMA runs with swapped A/B roles so a lane holds four
  consecutive `d`), a bf16 `[v][d]` snapshot crosses waves through LDS, the scaled-key
  hi/lo factors are built one chunk ahead in row layout and staged in per-wave LDS tiles,
  and every chunk factor is prefetched from clamped addresses; two barriers per chunk.
- `regstate_hwcvt_v16_wg256`: the same body with gfx950 `v_cvt_pk_bf16_f32` for every
  bf16 rounding (bench-only arm: NaN payloads may differ from the software `f2bf`).
- `control_timed` / `regstate_timed`: `s_memtime` phase stamps accumulated by wave 0.

The oracle requires bit equality for the complete BF16 output and FP32 final state and
verifies that Aqk is unchanged; `run.sh` rejects private memory or register spills before
acquiring one GPU. `KDA_DEBUG=1` prints the state mismatch pattern by `d` and `v`.

Run inside `nix develop` (`T=8191` exercises the 63-row tail chunk):

```sh
runtime/bench/amd/kda_carry_regstate/run.sh            # T=8192 H=12, 21 samples, timers
T=8191 SAMPLES=3 TIMERS=0 runtime/bench/amd/kda_carry_regstate/run.sh
```

## MI355X result (2026-09-04)

Exact shape `T8192,H12,D128,V128,BT64`, 21 order-rotated samples; oracle also passed at
`T=64` and `T=8191`.

| arm | median | vs control | gfx950 resources |
| --- | ---: | ---: | --- |
| control V16/WG512 | 1.916 ms | -- | VGPR 204, SGPR 68, occupancy 2, no spill |
| regstate V16/WG256 | 0.726 ms | -62.1% (2.64x) | VGPR 235, SGPR 53, LDS 43,520 B, no spill |
| regstate hwcvt V16/WG256 (bench-only) | 0.572 ms | -70.2% (3.36x) | VGPR 238, SGPR 48, LDS 43,520 B, no spill |

Both candidate arms matched all 12,582,912 BF16 outputs and 196,608 FP32 state elements. The
hardware-convert arm is exact on every finite input but may differ in NaN payload, so the
shipped object uses the software RNE.

Per-chunk `s_memtime` attribution (wave 0, mean over 96 workgroups x 128 chunks):

| phase | control cycles | regstate cycles |
| --- | ---: | ---: |
| V' / from-state products | 8,700 | 567 |
| loads issue + barrier 1 | 114 | 2,366 |
| output product + store | 8,097 | 622 |
| barrier 2 | 162 | -- |
| state update (hi/lo MFMA + fma) | 17,899 | 1,575 |
| next-chunk key factors (exp2, split, tile) + prefetch | -- | 6,989 |
| barrier 3 / 2 | 213 | 568 |
| total | 35,327 | 12,949 |

The shipping body spends its time in serialized global-load round trips: every per-element
bounds predicate (`token < mv`, `s + j < valid`) becomes an exec-masked branch around one
`global_load_*` followed by `s_waitcnt vmcnt(0)`, ~40 dependent round trips per chunk; the
MFMA, LDS, and exp2 work is a small fraction. The register-resident body removes those and is
left VALU-bound in the once-per-chunk scaled-key computation (32 `exp2` and 64 RNE splits per
lane), which is duplicated across the eight V-tile workgroups of a head.
