# L5 — multi-workgroup decode AttnRes: status (2026-09-04)

Branch `codex/decode-l5-attnres-mwg` (on `codex/amd-agent-harness`). Lever L5 of
`decode-gap-plan-20260904.md`: the K3 decode AttnRes (+ fused RMSNorm, 187 packets/token) runs on
ONE workgroup per site; this splits the 7168-wide row across `n` column-band workgroups inside the
same packet.

## Implemented

- `runtime/amd/op_k3.h` `d_attn_res_mwg` behind `-DPLOW_ATTNRES_DECODE_MWG=1` (default 0; flag-off
  objects untouched — the arm and its helpers are inside the `#if`). Band = `ceil(896/n)` 8-wide
  vectors, one per thread; every source row's band is loaded once into registers (9 x 4 VGPRs);
  per-row `(sum x^2, sum x*w)` partials wave-summed on a DPP tree, exchanged as tagged 8-byte
  words (32-bit round tag | f32) at agent scope — the op_collective.h tagged-XReduce scheme —
  polled one word per thread; a second one-word exchange folds the output norm's `sum mixed^2`.
  Round tag = `fetch_add(counter) / n + 1` per site (tokens are serial); norm words parity
  double-buffered so back-to-back rounds cannot race. Residual seam (`res_a/res_b/res_pre`) and the
  snapshot push are materialized per band inside the arm.
- `runtime/amd/interp.hip`: `PLOW_DOP_ATTN_RES` takes the banded arm when `i6 != 0` (the scratch
  handle), else the existing single-workgroup arm, byte for byte.
- Emit knob `PLOW_ATTNRES_DECODE_MWG=<n>` (`emit_config.attnres_decode_mwg`, 2..=16): `emit_attn_res`
  at `t == 1` with a fused norm emits `blocks = 0..n`, allocates `act.attnres.mwg.<id>` (2568 B, compiler-owned, loader-zeroed) per
  site, sets `i6` = scratch and `f1` = output-norm eps. Manifest feature `attnres_decode_mwg` ->
  `#define PLOW_ATTNRES_DECODE_MWG 1` in `plow_config.h` and `PLOW_ATTNRES_DECODE_MWG=1` in the
  object requirements. Slot table/doc: `i6=mwg_scratch`. Default off: packets byte-identical.
- Microbench `runtime/bench/amd/attnres_decode_mwg/` (`run.sh`): K3 shape (T=1, HID=7168, fused
  norm, nb 0..8, 186 cold weight sites, rows L2-hot and MALL-cold), in-kernel repeat loop, control
  = current single-workgroup arm.

## Numerics (which contract and why)

C3 f32-mix, the contract of the promoted prefill object `attn_res_f32mix_gfx950.hip`: the mix stays
f32 through the output RMSNorm (no bf16 seam). Reason: a banded split cannot reproduce the
single-workgroup arm's reduction order, so the bit-exact contract is unavailable regardless; the
f32-mix contract makes decode compute what prefill already computes (the plan's L5 row says the
same). Reduction order is fixed (lane FMAs -> DPP wave tree -> waves 0..7 -> bands 0..n-1 -> probs
then row-order mix), so output is deterministic for a given `n`. Bench check: every element within
1 bf16 ulp of a double-precision port of the contract, relL2 1.7e-3 (the bf16 rounding floor),
byte-stable across repeats, at every nb and n in {2,4,7,8,14,16}. The control (bf16-seam arm)
differs from the same reference by up to 2 ulp, as expected of a different contract.

## Microbench (MI355X, one GPU, us per site, K3 schedule-weighted mean over nb = ceil(layer/12))

| rows | 1 WG (current) | 2 WG | 4 WG | 7 WG | 8 WG | 14 WG | 16 WG |
|---|---:|---:|---:|---:|---:|---:|---:|
| L2-hot | 8.29 | 4.67 | **4.35** | 5.21 | 5.30 | 5.69 | 5.96 |
| MALL-cold | 8.42 | 4.83 | **4.55** | 5.29 | 5.36 | 5.74 | 6.07 |
| nb=8 hot | 11.18 | 5.73 | 4.93 | 5.92 | 6.04 | 6.44 | 6.60 |

x186 sites: 1.54 ms/token -> 0.81 ms/token at 4 bands (-0.73 ms in microbench terms; the in-trace
body was 17 us/site, which includes the materialize pass and the gate, both of which the banded arm
also absorbs — the TP8 gate decides the served delta). Recommend `PLOW_ATTNRES_DECODE_MWG=4`; more
bands pay more rendezvous latency than they save (floor ~3.6 us at nb=0 = 2 round trips + 6
barriers). A Gram-matrix variant folding the norm into the first exchange (`sum mixed^2 = p^T G p`)
was measured and rejected: 6.08 vs 4.35 us mean (45 wave reductions per band).

## Resources

- Bench candidate kernel (the arm alone): 98 VGPRs, 0 VGPR spill, 13 SGPR spill, 0 B private.
- `interp_decode_k3.elf` built flag-on from the `PLOW_ATTNRES_DECODE_MWG=8` emit (Lean verified,
  pairing 0x6892b68e52f0e447, 233 decode segments): 248 VGPRs / 0 VGPR spill / 108 SGPR spills /
  216 B private / 147496 B LDS, occupancy 2, accepted — the flag-off object is 248 / 108 / 288 B
  (the arm is far under the object's GEMM-set peak; the private segment moved 288 -> 216 B).

## Verified

- `cargo build --release -p plowc`, `cargo test -p devgen -p packet` (all pass except the two
  gfx942 `tuned_tile_selection` tests, which also fail on the base commit: the MI300X tunedb is
  stale against any runtime/amd edit), `cargo fmt --all --check` clean.
- Flag-on emit: `docs/k3-mi355x-20260904/scripts/showdown_bundle.sh <dir> PLOW_ATTNRES_DECODE_MWG=4`
  (needs the script's `wt=` pointed at this branch's checkout) — Lean verified, objects build.

## Remaining

1. TP8 gate (main session owns GPUs): flag-on bundle vs the served bundle, 8192->256, three
   order-alternated folds; C3 contract so expect a checksum change — run the seam oracle + GSM8K
   n=200 parity per the plan's L5 row; record TPOT delta in `perf-data/`.
2. Choose `n` on the served trace (4 from the microbench; 2 if the rendezvous costs more in-program).
3. Cross-XCD placement: bands are CUs 0..n-1; if the trace shows cross-XCD misses on the rows,
   place the bands on the producing XReduce's CUs (phase 2 / D6 gang in the plan).
4. Runtime hang guard: a rendezvous that never completes poisons after 2^20 polls (~1 s); wire the
   poison into the xctr status word if a served hang ever needs attribution.
