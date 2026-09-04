# Decode one-shot XReduce: attribution, isolated-unit correction, tagged prototype (2026-09-04)

Source: `/tmp/k3-xr-phase-gate/trace-ctl.raw` (rank 0, one decode token, control objects,
HIER on), `scripts/k3_trace_report.py`, `scripts/k3_xr_decode_report.py`,
`runtime/tests/tp_allreduce_bench.c` on one leased gfx950. No TP8 run was made here; the
TP8 commands are at the end.

## 1. Where the 15.6 us goes (rank-0 trace, 278 collectives)

| class | n | gate/pk | body/pk | straggler/pk | producer | producer own WG spread |
|---|---:|---:|---:|---:|---|---:|
| XREDUCE b=14 (7168) | 186 | 1.49 us | 15.63 us | 1.95 us | GEMV b=256 | 3.15 us |
| XREDUCE b=7 (3584) | 92 | 0.96 us | 15.61 us | 1.82 us | MOE_COMBINE b=7 | 0.61 us |

Body = last workgroup t_end - last workgroup t_ready. Per workgroup (14 or 7 of them) the
post-ready time is 14.1-16.1 us with every slice equal within 0.3 us; slice 0, which issues
the eight peer signals, is only 0.27 us (b=14) / 1.2 us (b=7) above the others. Within-rank
ready spread is 0.8 / 0.3 us, so the local gate is not the term.

The body does not depend on the producer or on how long the ranks ran unsynchronised
before the collective:

| packets since previous XREDUCE | segment length | n | body mean | sd |
|---:|---:|---:|---:|---:|
| 4 | 36 us | 92 | 15.60 us | 1.38 |
| 8 | 101 us | 92 | 15.61 us | 1.10 |
| 10 | 97 us | 68 | 15.32 us | 1.85 |
| 15 | 168 us | 24 | 16.81 us | 1.96 |

corr(body, segment length) = 0.13; quartiles over the token 15.3 / 15.8 / 15.9 / 15.5 us.
Producer skew that accumulated over the segment would scale with it; it does not. The
15.6 us is a fixed protocol cost. The rank-0 trace alone cannot split it further: the
one-shot body has no internal timestamps and the last-arriving rank is not identifiable
from one rank.

## 2. The isolated "0.98 us" was 10x too small

`tp_allreduce_bench.c` and `tp_allreduce_prefill_bench.c` scaled `s_memrealtime` ticks by
`HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY` (1 GHz, `rocminfo`: "System Timestamp Freq.: 1000 MHz").
`s_memrealtime` is the 100 MHz REFCLK, which the interpreter trace calibrates against host
TTFT (`k3_trace_report.py`, 1400.8 ms device span vs 1415.5 ms host). Host wall clock now
printed per row confirms it on one GPU: 20000 hot iterations at 14 KiB, device 1.300 us vs
host 1.303 us per collective. Every number those two harnesses printed before this fix is
10x understated:

| cell | printed | actual |
|---|---:|---:|
| one-shot 14 KiB, TP8 hot, 14 WGs | 0.981 us | 9.81 us |
| one-shot 7 KiB, TP8 hot, 7 WGs | 0.964 us | 9.64 us |
| two-shot 8192 x 7168 (112 MiB) | 63.45 us | 634.5 us |
| RS-U2 isolated projection, 278 prefill calls | 17.1 ms | 171 ms |

Consequences: the AITER parity ratios in `kimi-k3-mi355x-aiter-xreduce-parity-20260904.md`
are 2.1x (14 KiB), 1.5x (7 KiB) and 0.79-0.86x (28-112 MiB, AITER faster), not 7-21x; the
prefill "integrated body is 10.9x the isolated projection" gap (`...-xreduce-phase-trace-...`)
was the units (171 vs 178 ms in-network); P1/D5's spill-free-phase expectation built on the
10x gap has no support. The decode collective's in-network body (15.6 us) is the isolated
hot protocol (9.8 us) plus about 6 us of cold-cache and last-rank effects. 634.5 us for the
112 MiB two-shot is 302 GB/s per rank, physically consistent; 63 us would have been 3 TB/s.
Other harnesses using the same scaling (`tp_p2p_bench.c`, `tp_tilegate_bench.c`,
`tp_coherence_bench.c`, `tp_moe_combine_xreduce_bench.c`) are not fixed here.

Single-GPU probe (peer = self, so no XGMI hop; ns per dependent op): system-scope 8 B load
157, returning system atomic 211, release fetch_add + `s_waitcnt vmcnt(0)` 352 clean / 819
with 64 KiB dirty L2, system-scope 4 B poll 84, poll + `buffer_inv sc0 sc1` 144.

## 3. Current protocol round trips (from the gfx950 ISA of `d_xreduce_mega`)

Signal loop, slice 0: per peer `global_load_dwordx2` (table) / `s_waitcnt` /
`buffer_wbl2 sc0 sc1` / `s_waitcnt vmcnt(0)` / `flat_atomic_add sc1`. The next iteration's
`s_waitcnt vmcnt(0)` waits for the previous non-returning atomic, so the eight signals are
eight serialised (writeback + remote atomic) round trips. Poll: `flat_load_dword sc0 sc1` +
`s_sleep 2`. Acquire: `buffer_inv sc0 sc1`. Reduce: per element per peer a table load,
`s_waitcnt`, a 2 B remote load, `s_waitcnt`: eight serialised remote reads per element (one
element per thread at 14 x 512 threads). Last-arriving rank to first reduced value: about
8 signal round trips + poll + acquire + 8 read round trips, 17 dependent fabric operations;
peers that arrived earlier see the last rank's signal after position r of its loop.

## 4. Tagged one-shot prototype (`d_xreduce_tagged_mega`, harness-only)

Each rank publishes its partial as 8-byte words: three bf16 values + a 16-bit sequence tag,
one aligned system-scope store per word (single-copy atomic on the fabric). Readers issue
all eight peer word loads at once at system scope (no stale L2, no acquire fence), reissue
only the words whose tag has not arrived, and accumulate in strict rank 0..7 f32 order as
`d_xreduce` does. No counter, no `buffer_wbl2`, no `buffer_inv`; the critical path after the
last producer's store lands is one remote read round trip plus at most one poll period.
Tag = low 16 bits of a per-token epoch x gates + gate id (harness: the iteration index);
double-buffered slots by parity as the emitter's partial_A/B, so a rank can only rewrite a
slot every peer has finished reading. Expected at TP8: 2-3 us hot including the prototype's
publish-by-copy (0.9 us on one GPU; a producer epilogue writing the layout directly removes
it), against 9.8 us hot / 15.6 us in-network, i.e. up to ~10 us x 278 = 2.8 ms/token if the
all-rank trace confirms the last-rank wait is small. Folded gather (gcols) is not covered.

Resources (gfx950, `-Rpass-analysis`): `tp_allreduce_tagged` 64 VGPR / 106 SGPR / 0 spills /
0 scratch / occupancy 7; control `tp_allreduce` 14 / 56 / 0 / 0 / 8; `tp_allreduce_cold`
18 / 70 / 0 / 0 / 8; `tp_peer_probe` 42 / 30 / 0 / 0 / 8. ISA: the eight
`global_load_dwordx2 ... sc0 sc1` issue back to back before the first tag check.

Single-GPU smoke (peer = self, 14 WGs, 14 KiB, us per collective): oneshot hot 1.30, cold
2.67, tagged hot 2.17, tagged_cold 3.97; benign and strict-order oracles bit-exact across
the 1..32 batch sweep in every arm. Single-rank numbers only show the local costs; the
protocol difference needs eight ranks.

## 5. Instruments added (default off, production `.text` byte-identical)

- `PLOW_XR_TRACE_PHASES=1` now also covers the one-shot: slice marker bits 15:14 = 0b10,
  t_arrive = entry, cu = signal loop issued (slice 0), pc = arrival gate cleared, t_ready =
  acquire done, t_end = reduce done. CMake passes the option to decode objects too. The
  diagnostic `interp_decode_k3_gq` builds at 248 VGPR / occ 2 / 0 VGPR spill / 105 SGPR
  spill (control 84) / 216 B private; flag-off `.text` identical to the pre-change build
  (`/tmp/xr-tag-hsaco/{before,after}/text.bin`).
- `scripts/k3_xr_decode_report.py`: per-rank body, min-over-ranks protocol floor, rank-0
  wait for the last rank, last-rank histogram, and the phase split when present.
- `tp_allreduce_bench.c`: `TP_MODE=oneshot|cold|tagged|tagged_cold`, `TP_ORACLE=order`,
  `TP_PROBE=1` fabric latency probe, host wall-clock column, corrected tick.

## 6. TP8 commands (parent session; never run here)

```sh
nix develop --command ./scripts/build_tp_allreduce.sh /tmp/xr-tag-bench
# 1. fabric latencies + corrected hot control, then the arms (14 KiB / 14 WGs and 7 KiB / 7 WGs)
perf-data/tools/gpulease -n 8 xr-probe scripts/run_tp_allreduce_decode.sh /tmp/xr-tag-bench oneshot benign 1 4000 7168
for m in cold tagged tagged_cold; do for hd in 7168 3584; do
  perf-data/tools/gpulease -n 8 xr-$m-$hd scripts/run_tp_allreduce_decode.sh /tmp/xr-tag-bench $m order 0 4000 $hd
done; done
# 2. all-rank decode trace, control objects: floor vs last-rank wait
PLOW_TRACE_RAW=/tmp/k3-xr-phase-gate/trace-xr PLOW_TRACE_ALLRANKS=1 <the trace-ctl plowrt bench invocation>
python3 scripts/k3_xr_decode_report.py /tmp/k3-xr-phase-gate/trace-xr.rk{0,1,2,3,4,5,6,7}
# 3. same with the phase object: cp -r hsaco-control hsaco-xrphase; replace interp_decode_k3_gq.elf
#    (+ .resources.json) with /tmp/xr-tag-hsaco/xrphase/, or rebuild with -DPLOW_XR_TRACE_PHASES=ON
```

Gate for a production route: TP8 strict-order oracle exact at 7168 and 3584, tagged_cold
< cold by >= 5 us at 14 KiB, and the all-rank trace floor >= 8 us (otherwise the body is
last-rank wait and no collective redesign reaches it).
