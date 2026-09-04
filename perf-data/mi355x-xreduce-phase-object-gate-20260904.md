# gfx950 XReduce phase-object route: first exact TP8 network gate

Date 2026-09-04. Kimi-K3, 8x MI355X, TP8, BF16 KV, MXFP4 experts. Both arms emitted from
HEAD `937e41f` with the production hybrid emitter (`K3_FULL=1 --max-ctx 16384`); the
candidate adds `PLOW_PHASE_OBJECTS=1` at packet build, which isolates every
`XReduceTwoShot` into its own ordered segment and records the compiler-derived
`dispatch_chains`. Packet pairing hash `0x0a3297821329ae02` on both arms (segmentation is not
part of the hash); control packet SHA256 `f1bf783d…` is the published-baseline packet.
Objects: packet-paired gfx950 sets built with inventory pruning and decode MLA segments; the
candidate set also carries `interp_xreduce_gq.elf` (82 VGPR / 104 SGPR / 8,200 B LDS /
occupancy 5 / zero spill / zero private, markers `plow_phase_inventory_xreduce_only_1`,
`plow_phase_xreduce_wave64_occ2_nospill_1`). Runtime `--amd-phase-objects=true` on the
candidate only; `--amd-tp-no-audit`, segment-major TP dispatch, compact audit, one warm-up.
Artifacts: `/tmp/k3-xr-phase-gate/`.

## Two runtime defects found and fixed before the candidate could run

1. `AQL chain has 1157 packets but queue capacity is 1024`: the phase chain reserves one AQL
   packet per ordered segment before ringing, and the 1157-segment candidate exceeded the
   1024-entry HSA queue. Fixed by sizing the queue (and its kernarg ring) at 4096 in both the
   Rust and C backends (`97f3cba`).
2. `AQL chain emitted more than its reserved 1157 packets` followed by a GPU memory access
   fault: the reservation counted segments, but the A4-reuse stage-1 route launches twice
   per segment and the EP align route four times. Fixed by reserving the exact per-segment
   launch sum (`e834351`); the commit check still fails closed on any future disagreement.

## Result (8192→1, three alternating folds each; 8192→256 one pair)

| fold | control TTFT p50 | candidate TTFT p50 |
|---|---:|---:|
| 1 | 1262.392 ms | 1285.608 ms |
| 2 | 1261.639 ms | 1286.136 ms |
| 3 | 1262.828 ms | 1283.041 ms |
| **mean** | **1262.286 ms** | **1284.928 ms (+22.642 ms, +1.79%)** |

Every fold produced the identical one-token checksum `fnv1a64:337f0f290d5ae157`. The
8192→256 pair matched all 256 output IDs (`fnv1a64:b7682a38c151ac99`) with TPOT 28.694
(control) vs 28.646 ms (candidate): decode-neutral, as expected for a prefill-only route.

**Verdict: exact, but rejected as-is.** The route is correct end to end at TP8 and the
spill-free XReduce object executes every collective, yet the network loses 22.6 ms. Per-segment
timing (below) is required before deciding whether the collective body shrank and the +464
ordered boundaries absorbed more than it saved, or whether the isolated 10x gap between the
focused object (17.1 ms) and the in-network collective (177.6 ms) does not come from the
interpreter's register envelope at all.

## Per-segment attribution

`PLOW_PREFILL_SEG_TIMING=1` (per-segment all-rank drains, segment-major disabled), one 8192→1
request after warm-up, both arms. Sum of per-segment critical time vs endpoint TTFT: control
1269.9 ms vs 1277.0 ms; candidate 1292.3 vs 1301.0 ms.

| family | control segs | control ms | candidate segs | candidate ms |
|---|---:|---:|---:|---:|
| ordinary interpreter (incl. raw KDA family) | 324 | 891.4 | 510 | 710.2 |
| isolated XReduce phase segments (`interp_xreduce_gq.elf`) | — | — | 278 | **203.4 (731.6 µs each)** |
| lean MoE stage-1 (A4 reuse) | 92 | 136.3 | 92 | 136.6 |
| raw MLA V2 | 24 | 92.2 | 24 | 92.2 |
| lean MoE stage-2 | 92 | 70.6 | 92 | 70.6 |
| KDA intra wave-item | 69 | 40.3 | 69 | 40.3 |
| lean MoE combine | 92 | 39.2 | 92 | 39.1 |
| **total** | 693 | 1269.9 | 1157 | 1292.3 |

The interpreter family lost 181.2 ms when the 278 collectives left it, and the 278 isolated
XReduce segments cost 203.4 ms. So the spill-free phase object runs the collective at the
**same** cost as the mega-interpreter did (~0.73 ms per full 8192x7168 two-shot), and the extra
278 boundaries cost only ~22 ms (~80 µs each with segment-major restored, far below the
0.49 ms/boundary priced for the AttnRes fusion, which added convergence-heavy seams).

## Verdict

The 10.9x gap between the focused XReduce object (17.1 ms for the same calls) and the in-network
collective (~200 ms) is **not** the interpreter's register/spill envelope. At 0.73 ms per
collective the rank moves ~205 MB over xGMI (7/8 of 117 MB in reduce-scatter plus the same in
all-gather) at ~280 GB/s inbound, which is the fabric, not the kernel; the isolated harness
re-reads a 117 MB working set that fits the 256 MB Infinity Cache. Consequences for the plan:

- Reject P1 as a body lever. `PLOW_PHASE_OBJECTS` stays default-off. Keep the queue-size and
  chain-reservation fixes (correctness).
- The remaining prefill collective lever is overlap (v3 P-C): slice the producer GEMM and its
  collective over token halves so compute hides fabric time, and fold consumers into the
  collective's epilogue to remove round trips. Byte reduction is closed (bf16 already; folded
  gather already).
- Boundary cost at ~80 µs/segment with segment-major dispatch means phase objects are cheap to
  add where a body win exists (dense GEMM, AttnRes), and expensive only when they add
  convergence-heavy seams.

## Erratum (2026-09-04, later the same day)

The "focused object 17.1 ms / 10.9x" comparison this gate set out to explain was a units bug in
the isolated harnesses (`tp_allreduce_bench.c`, `tp_allreduce_prefill_bench.c` scaled
`s_memrealtime` by the 1 GHz `HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY`; the counter is the 100 MHz
REFCLK, as the trace calibration already used). Corrected isolated figures: 14 KiB one-shot
9.81 µs (not 0.98), 112 MiB two-shot 634.5 µs (not 63), i.e. the in-network 0.73 ms per full
collective was never 10x off its isolated cost. The verdict above stands on its own evidence
(the phase object ran the collective at the same in-network cost); the fabric-floor reading is
strengthened. The AITER parity report's 7-21x also needs the same correction: AITER is 1.5-2.1x
slower at decode sizes and 0.79-0.86x (faster) at 28-112 MiB. Fix and details:
`perf-data/kimi-k3-mi355x-xr-decode-tagged-20260904.md` (branch `worktree-agent-a3db1e5a3e4fae120`).
