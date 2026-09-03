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

## Promotion blockers

1. Beat 1.5 ms at T8192 with distinct K and V inputs. The identity-structured screen is an
   oracle construction, not permission to alias K and V in a real graph.
2. Add and price the K192/V128 materialization GEMMs in a fused one-layer block.
3. Select the path from MLA graph geometry (`qk_nope + qk_rope`, `v_head`, latent width), not a
   model name.
4. Pass a full-network exact A/B before changing the default.
