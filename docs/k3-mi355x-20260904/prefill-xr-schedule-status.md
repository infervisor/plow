# Prefill collective schedule: AITER-rate data movement under strict rank order (2026-09-04)

Branch `codex/prefill-xr-schedule` (from `codex/amd-agent-harness` e26daf3). Lever from
`k3-beat-vllm-0.28-v3.md` ("AITER-rate collective load schedule -30..-60 ms"): at the K3 prefill
sizes AITER's custom all-reduce moves 112 MiB in 500 us where Plow's two-shot takes 634-705 us
(0.71-0.79x); the campaign summary's erratum row fixes the 10x timestamp unit both numbers are
quoted in. Scope: the data-movement schedule only — accumulation order r = 0..7 in f32 and
every rounding point unchanged, so every output is bit-identical.

## What AITER does (vLLM 0.28 pin, `aiter_meta/csrc/include/custom_all_reduce.cuh`, gfx950 .so)

`cross_device_reduce_2stage<__bf16, 8>`: 80 workgroups x 512 threads, 48 VGPR, 8 KiB LDS.
- Stage 1 (reduce-scatter): each wave loads ONE 16-byte pack per lane (`flat_load_dwordx4`)
  from ITS peer — wave w reads peer `(rank + w) % 8` — into LDS; barrier; wave 0 sums the eight
  packs in that ROTATED order and stores 16 B; barrier. One remote load in flight per lane,
  eight links busy per workgroup, 640 KiB in flight per rank.
- Stage 2 (all-gather): wave w copies slice `(w + rank) % 8` with 16-byte loads/stores, grid
  stride 80 x 64 lanes.
- Per-block flags (`start/end[block][rank]`) instead of a global gate; plain (non-`sc`) stores.
- The rotated accumulate is why porting the kernel itself was rejected (campaign summary,
  "AITER custom-AR parity").

Plow's shipping two-shot (`d_xreduce_twoshot_mega`, RS-U2 on gfx950): 2-byte loads per lane,
peers walked serially with two elements in flight, all-gather 2-byte copies with a per-workgroup
peer stagger, 256 workgroups. The `xrbw` "2 B beats 16 B" rule that fixed these widths was
measured on MI300X at 304 workgroups — i.e. at an in-flight volume where 16 B loses (below).

## The arm: `-DPLOW_XR_SCHED=aiter` (`PLOW_XR_SCHED_AITER=1`, default OFF)

`runtime/amd/op_collective.h`: `xr_rs_sched` / `xr_ag_sched`, used by the two-shot's phases 1
and 2, op 25 (`d_xreduce_scatter_mega`, with the FFN-seam `gcols` fold and the band copy) and
op 26 (`d_xall_gather_mega`).
- Reduce-scatter: each lane loads one 16-byte pack from ALL eight peers before the first add
  (ISA: `global_load_dwordx4 x8`, then `v_add_f32 x64`), sums k = 0..7 in strict r = 0..7 f32
  order, one `f2bf`, 16-byte store. The eight links are as busy as under AITER's wave-per-peer
  form without the LDS round trip that idles seven waves during the sum (that form is
  `PLOW_XR_WAVE_RS`, measured +3% on 09-04).
- All-gather: wave w gathers slice `(w + rank) % 8` with 16-byte loads and stores, two packs in
  flight per lane (fused residual and `gcols` fold vectorised the same way, same per-element
  arithmetic).
- Misaligned ranges (slot, band, gather/residual base, packs straddling a slice) fall back to
  the scalar loop element by element; TP != 8 takes the scalar reduce-scatter and the
  workgroup-staggered 16-byte gather.
- Protocol untouched: gates, aggregation (`PLOW_XR_AGG`), signals, workgroup count, element
  ownership, `gate_expectations`.
- CMake `PLOW_XR_SCHED=off|aiter` appends the define to the PREFILL axes only; decode objects,
  the packet and plowrt are byte-identical either way. Marker `plow_xr_sched_aiter_1`.
- **The workgroup count is part of the schedule** (below): the arm must run on a packet emitted
  with `PLOW_XR_CUS=48` (only the collective packets' CU sets change; every other packet, tile
  and the TuneDB digest are untouched — the tuning build's defines are fixed in
  `kernelcaps::targets`). On the shipping 256-workgroup packet the arm is a +43% regression.

## Flag-off byte identity

`interp_prefill_mla_moe_a4w4_full` compiled from the parent commit e26daf3 and from this tree
with the served bundle's defines (`/tmp/k3-seqpar/assets/plow_config.h`, `-DPLOW_XR_AGG=1
-DPLOW_XR_RS_U=2 ...`): `.text` (377,792 B), `.rodata`, `.data` and `.note` sha256-identical;
the only differing bytes are the `__hip_cuid_*` symbol in `.dynstr`, a hash of the source PATH
(`/tmp/xrs-base-src` vs the worktree). The preprocessed translation unit
(`kernelcaps::preprocessed_digest` rules, `hipcc -E`) is byte-identical for gfx950 with the K3
defines and for gfx942 with the dense defaults, so no TuneDB record is re-staled by this change.
(The two `tuned_tile_selection` gfx942 failures in `cargo test -p devgen` fail identically on
the parent commit: that cell is stale for unrelated reasons.)

## Resources (`plow_interp_gfx950`, the K3 prefill object, contract 256 regs / occ 2)

| object | bytes | VGPR | AGPR | occ | VGPR spill | SGPR | SGPR spill | private |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flag off (= parent e26daf3) | 391,520 | 256 | 0 | 2 | 2 | 108 | 78 | 752 B |
| `-DPLOW_XR_SCHED=aiter` | 406,664 | 256 | 0 | 2 | 2 | 108 | 102 | 752 B |

Same VGPR/AGPR/occupancy/VGPR-spill/private envelope; +24 SGPR spills and +15 KB of code
(the eight-peer pack loops inlined at four sites). Accepted by the register-cliff gate.

Microbench kernels (`tp_allreduce_kernels.elf`, no interpreter envelope): two-shot 22 -> 108 VGPR,
op25+op26 24 -> 108 VGPR, 0 spill, 0 scratch either way.

## TP8 microbench (`tp_allreduce_prefill_bench`, 8x MI355X, 512 threads/WG, 10 ns tick)

Random-data strict-order oracle (`TP_RANDOM=1`: bf16 words with exponents 2^-8..2^7, host
reference in r = 0..7 f32 order with the device rounding), 5 iterations, `TP_PHASES=1` runs
op 25 then op 26 with a device-local barrier between them (each phase with its own rendezvous).
AITER reference: `bench_aiter_custom_ar.py` registered, `/tmp/xreduce-parity-results/aiter-order.json`.

Screen 1 — the arm at Plow's 256 workgroups (us; two numbers = two reps):

| arm | 112 MiB two-shot | 56 MiB | 28 MiB | 112 MiB + gather fold | op25 RS + op26 AG (112 MiB) |
|---|---:|---:|---:|---:|---:|
| off (2 B, RS-U2) | 700 / 703 | 414 / 422 | 271 / 278 | 944 / 939 | 317 + 313 / 312 + 318 |
| aiter (16 B, 8 peers hoisted, wave AG) | 1001 / 1011 | 566 / 544 | 291 / 288 | 1360 / 1407 | 545 + 436 / 489 + 491 |
| aiter, AG 2 packs/lane | 949 / 1002 | 539 | 292 | 1334 / 1358 | 529 + 455 |
| aiter, RS 2 packs/lane | 1048 | 556 | 289 | 1393 | 579 + 472 |
| aiter, WG-staggered 16 B AG | 1111 | 561 | 289 | 1350 | 617 + 462 |

Screen 2/3 — the workgroup count (112 MiB, op25 RS + op26 AG, and the two-shot):

| WGs | 16 | 24 | 32 | 40 | 48 | 64 | 80 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aiter RS + AG | 291+534 | 257+364 | 274+286 | 265+259 | 273+261 | 309+266 | 362+255 | 462+312 | 489+491 |
| aiter two-shot | 825 | 622 | 560 | 524 | 535 | 575 | 613 | 784 | 1001 |
| off RS + AG | | | | | | | 630+664 | 410+444 | 312+326 |
| off two-shot | | | | | | | 1296 | 871 | 705 |

Both loops want the same thing — roughly 256-512 KiB in flight per link — and reach it at
opposite workgroup counts: 2 B x 2 elements x 256 WG x 512 lanes = 512 KiB; 16 B x 48 x 512 =
384 KiB per peer. Past it the fabric degrades (16 B at 256 WGs has 16 MiB in flight), below it
the links idle. At the matched volume the 16-byte form is 12-22% faster per phase. Other forms
at 32-64 WGs: AITER's exact one-load-per-lane dependency-chained walk 612 / 561 us (32 / 64 WGs)
vs 560 / 575 hoisted; 8-byte packs 631 vs 617 at 80 WGs; RS 2 packs/lane 568 vs 560 at 32;
AG 2 packs/lane 516 vs 560 at 32 and 529 vs 535 at 48 (kept: `PLOW_XR_SCHED_AG_U=2`).

Screen 3 — the K3 shapes at the sweet spot vs the shipping loop (two-shot, us):

| shape | off @256 | aiter @32 | aiter @48 | AITER (80 WG, rotated order) |
|---|---:|---:|---:|---:|
| 8192 x 7168 (112 MiB) | 705 | 562 | 541 | 500 |
| 8192 x 7168 + gather fold | 944 | 941 | 800 | — |
| 4096 x 7168 (56 MiB) | 415 | 294 | 279 | 256 |
| 2048 x 7168 (28 MiB) | 268 | 168 | — | 145 |

Screen 4 — the promoted form (`AG_U=2`) at 40 / 48 / 56 workgroups, two reps each (values
within 1% between reps), 8 iterations, vs the shipping loop at 256 (us):

| shape | off @256 | aiter @40 | aiter @48 | aiter @56 |
|---|---:|---:|---:|---:|
| 112 MiB two-shot | 704 | 519-522 | 531-537 | 552-556 |
| 112 MiB op25 RS + op26 AG | 316 + 320 | 264 + 253 | 275 + 254 | 295 + 257 |
| 112 MiB + gather fold two-shot | 941 | 847 | 816 | 805 |
| 56 MiB two-shot | 423 | 279 | 281 | 299 |
| 56 MiB op25 RS + op26 AG | 199 + 158 | 141 + 131 | 146 + 134 | 165 + 133 |
| 28 MiB two-shot | 264 | 156-160 | 166-168 | 170 |

48 is the pick (the folded-gather seam, the one K3 collective with a third operand stream,
wants the extra workgroups; everything else is within 2-3% of 40). Note the two-shot's second
rendezvous: at 256 workgroups the two-shot costs 70 us more than op25+op26 (gate_ag counts
8 x 256 arrivals); at 48 the two-shot and the split phases cost the same.

Against AITER (500 / 256 / 145 us): 531 / 281 / 167 — 6-15% over AITER's rotated-order kernel,
exact, down from 41 / 65 / 82% over.

## Exactness

- Every run above: `parity=PASS bad=0` against the host strict-order oracle on all 8 ranks, and
  the per-rank FNV-1a checksums agree across ranks and across ARMS for each shape
  (112 MiB `0bfc954aec66d22c`, +gather `ab3b2ebfe897b44c`, 56 MiB `6249ba8cbd648476`, 28 MiB
  `05a4d68f24f24d1d`), i.e. flag-on output bytes == flag-off output bytes.
- `tp_seqpar_smoke` (op 25 with the `gcols` fold and the band copy, op 26 with two arrays):
  PASS on 8 ranks at rows 64 / 72 / 128 (48 WGs; the vector fold, band copy and wave gather
  paths at TP8) and on 1 rank (the scalar fallbacks), flag on and off, `bad_slot = bad_copy =
  bad_ag = bad_ag2 = 0`.
- The flag-off object is byte-identical to the parent (above).

## Projection

Per MoE layer under the seams (93 layers, T = 8192): attention-seam reduce-scatter 112 MiB,
FFN-seam reduce-scatter 112 MiB with the fold, all-gather of h2 112 MiB + xe 56 MiB + route
table, latent two-shot 56 MiB. Isolated deltas at 48 WGs vs the shipping loop at 256 (screen
4): RS -41 us each, AG -66 (112 MiB) and -25 (56 MiB), 56 MiB two-shot -142 (-77 against the
split-phase control) → about -250..-315 us per layer ≈ **-23..-29 ms TTFT** (of the ~140-200 ms
prefill collective envelope), at the low end of the lever's -30..-60 range. The 48-workgroup
packets also leave 208 CUs free during each
collective, so any global-queue interleaving of independent packets is upside the 256-workgroup
loop cannot offer (the P-C screen's +92 ms at `PLOW_XR_CUS=128` was the 2 B loop collapsing,
not overlap failing). Risk: in-network the collective's load competes with L2/fabric traffic of
neighbouring packets, which the isolated bench does not see; the gate decides.

## Gate

Main session, TP8, 8192->256, three alternating folds vs the flag-off control bundle from the
same source; both arms must print `fnv1a64:71a28c1449921c95`:

    docs/k3-mi355x-20260904/scripts/xr_sched_gate.sh

which builds `/tmp/k3-xrsched-ctl` (flag off, default packet) and `/tmp/k3-xrsched`
(`PLOW_BUNDLE_CMAKE_EXTRA="-DPLOW_XR_SCHED=aiter" showdown_bundle.sh /tmp/k3-xrsched PLOW_XR_CUS=48`)
from this worktree (`PLOW_BUNDLE_SRC`), then one audited candidate run and the folds with the
control's plowrt. If the in-network gain is short of the projection, sweep `PLOW_XR_CUS` 40/56
before concluding; if TPOT moves, `PLOW_XR_CUS` is binding on a decode collective and the cap
should move into the emitter as a prefill-only rule.

## Reproduce the microbench

    nix develop -c scripts/build_tp_allreduce.sh /tmp/xrs-off
    nix develop -c env PLOW_XR_SCHED=aiter scripts/build_tp_allreduce.sh /tmp/xrs-on
    GPU_LEASE_TIMEOUT=14400 perf-data/tools/gpulease -n 8 xr-sched bash -c \
      'cd /tmp/xrs-on && TP_RANDOM=1 TP_NWG=48 TP_ROWS=8192 TP_HIDDEN=7168 TP_ITERS=5 ./tp_allreduce_prefill_bench 0 1 2 3 4 5 6 7'
    # TP_PHASES=1 times op 25 and op 26 separately; TP_GATHER=1 is the folded-gather two-shot;
    # PLOW_XR_EXTRA_DEFS="-DPLOW_XR_SCHED_AG_U=1 ..." builds the screen knobs.
