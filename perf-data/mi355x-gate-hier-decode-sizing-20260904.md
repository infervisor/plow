# gfx950 decode: PLOW_GATE_HIER on/off on the pruned, MLA-split object

Date 2026-09-04. Kimi-K3, 8x MI355X, TP8, BF16 KV. Same control packet (`f1bf783d…`, HEAD
`937e41f`), two packet-paired gfx950 object sets that differ only in the CMake option
`PLOW_GATE_HIER` (ON = production default, OFF = flat per-workgroup signal). Runtime
`--amd-tp-no-audit`, compact audit, segment-major TP prefill, one warm-up, three measured
8192→256 requests per fold, order-alternated on/off/off/on/on/off under an exclusive
`gpulease -n 8`. Artifacts: `/tmp/k3-xr-phase-gate/bench-hier-*.log`.

| fold | HIER on TPOT p50 | HIER off TPOT p50 |
|---|---:|---:|
| 1 | 28.752 ms | 34.607 ms |
| 2 | 28.682 ms | 34.776 ms |
| 3 | 28.747 ms | 34.595 ms |
| **mean** | **28.727 ms** | **34.659 ms** |

Delta −5.932 ms/token (−17.1%) with HIER on; every fold matched all 256 output IDs
(`fnv1a64:b7682a38c151ac99`). E2E moved by the same amount times 256 tokens (8,589 vs
10,101 ms), i.e. TTFT is unaffected, as expected for a decode-side signal change.

## Reading

- The measured value of the two-level signal on the current pruned + MLA-split decode object
  is larger than the pre-HIER joint protocol ceiling (`PLOW_GATE_NOINV + PLOW_GATE_RELAXSIG`,
  −5.04 ms/token, README §6). That ceiling was measured on the old 256-VGPR/2-spill object;
  it must not be used to size the remaining `sc1` opportunity.
- Sizing `sc1` therefore needs the per-packet gate cost with HIER on, from a decode trace of the
  current object (`PLOW_TRACE_RAW`, `scripts/k3_trace_report.py` gate/pk), not from the old
  13.20 → 3.84 µs signal measurement.
- Keep `PLOW_GATE_HIER` default-on for gfx950; no code change.
