# PX-3 — BN=64 occ-2 prefill GEMM (rtx-11 cheap check)

RTX PRO 6000 Blackwell (sm_120a, 188 SMs), gemma-4-12B bf16, TP1. Branch `px3-px5-cheap-checks`.

## Thesis

T10 found occ-2 needs a ≤50 KiB arena, but the BN=128 3-stage GEMM tile is 60 KiB → occ-1.
T10 reached occ-2 only by **halving** the pipeline (3/2 → 2/1 stages), which lost more than
occupancy bought. **PX-3:** a **BN=64** N-tile drops the arena to 45 KiB (plain) / 40 KiB (GLU)
while **keeping the full 3-stage plain / 2-stage GLU pipeline** — so occ-2 is reachable *without*
halving. T2 found BN=128 wins at occ-1; the question is whether BN=64/occ-2 beats BN=128/occ-1.

## Change (surgical)

- `runtime/nvidia/op_gemm.cuh`: `PGM_BN` made `#ifndef`-overridable (was a hard `#define 128`).
- `runtime/nvidia/interp_sm120.cu`: the lean GEMM segment object (`PLOW_NV_SEG_GEMM=1`) gains a
  `PLOW_NV_SEG_GEMM_BN64=1` variant → `PGM_BN=64` with `PGM_STAGES=3 / PGM_GLU_STAGES=2`
  (vs the default T10 lean object: BN=128 + halved 2/1).

## Gates — ALL PASS

| gate | BN=128 (occ-1 / T10 halved) | **BN=64 full-3-stage (PX-3)** |
|------|------------------------------|-------------------------------|
| **parity** vs f32 CPU ref | gemm relL2 2.979e-03, glu 3.782e-03 | **IDENTICAL** (2.979e-03 / 3.782e-03) → element-for-element = BN=128 → first-token agreement guaranteed |
| **ptxas** (standalone k_gemm) | 94 regs, 0 spill | **55 regs, 0 spill** |
| **ptxas** (megakernel `_pfgemm`) | 128 regs, **620/780 B spill** (T10 halved) | **128 regs, 28/40 B spill** (~20× less) |
| **occ-2** (driver-API, real cubin) | 40 KiB → 2 blk/SM | **45 KiB → 2 blk/SM** (≤50 KiB, ≤128 regs) ✓ |

The BN=64 object clears the T10 wall: occ-2 with the **full** pipeline, and far less spill
(BN=64 halves `PGM_NFRAG` 8→4 and the f32 accumulator 64→32/thread).

## Per-op A/B (standalone `k_gemm`, ms/op, 30-iter mean)

The e2e megakernel occ-2 A/B is **blocked** (see below), so the honest measurement is the
standalone tiled-GEMM kernel — its occupancy is set by the GEMM arena+regs **alone**, exactly
like the lean object. `grid=188` = 1 blk/SM, `grid=376` = 2 blk/SM.

| op (M=4096) | N | K | BN128/occ1 | BN64/occ1 | **BN64/occ2** | occ2 vs occ1(128) |
|-------------|-----|-----|-----------|-----------|---------------|-------------------|
| q_proj      | 4096 | 3840 | 0.7337 | 0.7685 | 0.7536 | +2.7% |
| o_proj      | 3840 | 4096 | 0.7654 | 0.7722 | 0.7336 | −4.2% |
| down_proj   | 3840 | 15360 | 2.7292 | 2.8996 | 2.7864 | +2.1% |
| **gateup_glu** | 15360 | 3840 | 5.0585 | 6.2241 | **4.5910** | **−9.2%** |
| **M=8192** | | | | | | |
| q_proj      | 4096 | 3840 | 1.3268 | 1.5164 | 1.4818 | +11.7% |
| o_proj      | 3840 | 4096 | 1.4109 | 1.4690 | 1.4246 | +1.0% |
| down_proj   | 3840 | 15360 | 4.9908 | 5.4923 | 5.2859 | +5.9% |
| **gateup_glu** | 15360 | 3840 | 8.9542 | 10.8063 | **8.4301** | **−5.9%** |

Per-layer GEMM sum (q+o+down+gateup_glu, representative): **−4.5% @ M=4096**, **−0.4% @ M=8192**.

## Reading

- **BN=64 at occ-1 is worse than BN=128 at occ-1 for every op** (confirms T2: the wider N-tile
  streams A once per N-tile, so it re-reads less activation — a direct bandwidth win).
- **occ-2 recovers BN=64 only where it matters**: `GEMM_GLU` (gate/up, two B-streams, the deepest
  pipeline and heaviest compute) — occ-2 turns BN=64's −23% occ-1 deficit into a −9% win over
  BN=128/occ-1. The plain projections are already SM-saturated at occ-1 with the bigger tile and
  stay wash-to-slightly-slower at BN=64/occ-2.

## Verdict — SMALL-WIN-OR-WASH (as the plan predicted)

occ-2 **is** reachable at BN=64 with the full pipeline — the "did we leave occupancy on the table"
question is answered: **a little, and only on `GEMM_GLU`.** occ-2 buys −5…−9% on the gate/up GEMM
and nothing (or a small loss) on the plain projections, which prefer the BN=128 occ-1 tile.
Net per-layer GEMM ≈ −4.5% @ 4k (GLU-driven), ≈ wash @ 8k. This confirms T2's arithmetic-intensity
finding: on this SM-saturated GEMM the tile shape matters more than occupancy. Lower priority than
PX-1/PX-2 stands.

## e2e status — blocked (honest negative)

The megakernel occ-2 A/B needs the emitter to re-slice the GEMM segments to `2*n_cu=376` blocks so
**both** resident blocks per SM get work; the emitter slices for `n_cu=188` (a block walks tiles by
the packet's `(slice, nblk)`, so at `nblk=188` blocks 188..375 idle). This is the *Stage-3
prerequisite* already documented in `runtime/CMakeLists.txt` and
`perf-data/gemma4-12b-t9c-segments-sm120.json`. Building that emitter path is **not a cheap check**,
so per the time-box this arm stops at the per-op standalone A/B — the faithful isolated measurement
of the same tile at the same occupancy the lean object would run. Given the per-op result (wash-or-
small-win, GLU-only), the emitter investment is **not** justified by this cheap check.

## Artifacts

- `perf-data/gemm_occ_bench.cu` — per-op timing bench (build `-DPGM_BN={64,128}`).
- `perf-data/gemm_parity.cu` — parity vs f32 CPU ref.
- `perf-data/occ_probe.cu` — driver-API occupancy on the real `_pfgemm` cubins.
- `perf-data/px3_build*.sh` — clean-env nvcc build helpers (nix CPATH conflicts).
