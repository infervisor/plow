# Materialized MLA prefill screen on gfx950

## Contract

- Isolated one-layer attention, BF16, causal, batch 1, 12 local heads.
- Control: absorbed QK dimension 512 latent + 64 NoPE channels, latent V dimension 512,
  followed by the production 512-to-128 `MlaMergeFold`.
- Candidate: materialized Q/K dimension 192 and V dimension 128.
- Grid 256, 256 threads, wave64, 512 MiB cache flush before each arm, median of 7.
- Inputs use identity-structured K/V projections. This makes both forms compute the same
  algebra while retaining the production attention and merge kernels. It does not price the
  two materialization GEMMs.

## Result

| T | absorbed + fold | materialized attention | speedup | max abs | RMSE |
|---:|---:|---:|---:|---:|---:|
| 1024 | 504.883 us | 172.241 us | 2.931x | 0.0154724 | 0.0023064 |
| 8192 | 7570.535 us | 2718.140 us | 2.785x | 0.0124207 | 0.00123868 |

The candidate is numerically within the recorded BF16 attention tolerance, but misses the P6
8192 target of 1.5 ms. It stays experimental and default-off. The result still confirms the
structural FLOP reduction; the dense rectangular flash schedule, not materialization, is the
next isolated kernel lever.

## Resource gate

| kernel | VGPR | AGPR | SGPR | occupancy | LDS | scratch/spill |
|---|---:|---:|---:|---:|---:|---:|
| absorbed attention | 254 | 92 | 57 | 1 | 41472 B | 0 / 0 |
| absorbed merge/fold | 74 | 0 | 38 | 6 | 2560 B | 0 / 0 |
| materialized rectangular attention | 256 | 81 | 82 | 1 | 37888 B | 0 / 0 |

## BKV64 follow-up: rejected

AITER's gfx950 dispatch table records a 128x128 tile for BF16 DK192/DV128. The safe first step
was BKV32 to BKV64 in the existing generic body:

- Prefetched BKV64 fit with no scratch or spill (VGPR 256, AGPR 179, SGPR 92, occupancy 1,
  LDS 59392 B), but failed the T1024 oracle with infinities. No timing from this arm is valid.
- Synchronous BKV64 was sound and spill-free (VGPR 256, AGPR 108, SGPR 79, occupancy 1,
  LDS 67584 B). It improved its synchronous BKV32 control from 3896.986 to 3259.062 us at
  T8192, but regressed the prefetched BKV32 candidate at 2718.140 us and still missed 1.5 ms.

Increasing the generic body's KV tile is therefore closed. A 128-row KV schedule needs a new
four-wave body with a deeper pipeline and a smaller score/P residency scheme, not this template's
array expansion.

The 8-wave form is also closed for this body. Its default four-fragment Q chunk spilled two
VGPRs and used 12 B/lane private memory. Reducing the Q chunk to two removed all spill/private
(VGPR 256, SGPR 80, occupancy 2, LDS 37888 B), but measured 3044.183 us at T8192 and
283.642 us at T1024. That is slower than the 4-wave BKV32 arm. The useful reference pattern is
instead Q in LDS followed by Q/V aliasing, double-buffered K and V, and a multi-stage pipeline;
wave count alone does not reproduce it.

## Structural LDS pipeline follow-up: rejected

A harness-only kernel implemented that lifetime pattern with real, distinct K and V inputs:

- four wave64 waves, 16x16x32 BF16 MFMA, BQ64/BKV128;
- Q staged in LDS; the 34 KiB V tile aliases the dead 25 KiB Q slab and 9 KiB of the consumed
  K slot;
- two 50 KiB K slots ping-pong; P aliases the remainder of the consumed K slot;
- next-K loads are staged two vectors at a time around eight current-tile score MFMAs, rather
  than retaining a full prefetched tile in registers.

The layout is wave64, VGPR 256 / AGPR 4 / SGPR 98, occupancy 1, 128000 B LDS, and zero
scratch/spill/private. Median-of-7 results under the same 512 MiB flush contract:

| T | absorbed + fold | generic rectangular | structural LDS | structural vs absorbed |
|---:|---:|---:|---:|---:|
| 1024 | 506.484 us | 171.561 us | 103.321 us | 4.902x |
| 8192 | 7447.816 us | 2694.421 us | 4185.231 us | 1.780x |

At T8192 the structural arm agrees with absorbed at max abs 0.00334167 / RMSE 0.00325587.
At T1024 it is max abs 0.00424671 / RMSE 0.00397821: below the absolute-error ceiling but above
the existing 0.003 RMSE gate, so it is rejected numerically as well as on time. Direct comparison
to the accepted generic arm at T1024 is max abs 0.00384521 / RMSE 0.00205845, consistent with a
different BKV128 softmax association rather than memory corruption.

Two precursor layouts closed additional branches:

- BQ128/BKV64 with a full next-K register tile was spill-free (VGPR 256 / AGPR 246 / SGPR 86,
  102400 B LDS) and oracle-sound enough to time, but took 5309.599 us at T8192. Retaining the
  full prefetch and restoring the Q alias 64 times are both losses.
- BQ128/BKV128 with 32x32 MFMA compiled to 153600 B LDS but spilled 207 VGPRs and used
  700 B/lane scratch, so it was rejected before timing.

The target remains unmet. The reference-style 16x16 schedule wins decisively while T1024 is
underfilled, then loses to the generic 32x32 kernel at long sequence. A next attempt must keep
32x32 throughput while reducing BKV128 score/P residency; further expansion of either tested
template is closed. No graph or emitter integration was attempted.

## Promotion blockers

1. Beat 1.5 ms at T8192 with distinct K and V inputs. The identity-structured screen is an
   oracle construction, not permission to alias K and V in a real graph.
2. Add and price the K192/V128 materialization GEMMs in a fused one-layer block.
3. Select the path from MLA graph geometry (`qk_nope + qk_rope`, `v_head`, latent width), not a
   model name.
4. Pass a full-network exact A/B before changing the default.
