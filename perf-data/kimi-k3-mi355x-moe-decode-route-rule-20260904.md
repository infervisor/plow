# gfx950 grouped-MoE decode route: measured profitability rule and promotion cell

Date 2026-09-04. Kimi-K3, 8x MI355X, TP8, BF16 KV, MXFP4 experts, HEAD `937e41f` + branch
`codex/d1-moe-decode-rule`. Two packets from the same source and emitter: control (interpreter
GLU+DOWN pair) and standalone (`PLOW_MOE_DECODE_STANDALONE=1`, adjacent grouped pairs isolated
into ordered raw launches of `moe_decode_grouped_mxfp4_gfx950.elf`). Both object sets
packet-paired (pairing `0x0a3297821329ae02`), inventory-pruned, decode-MLA-split, HIER on.
Runtime `--amd-tp-no-audit`, compact audit, one warm-up, three 8192→256 requests per fold,
order-alternated c/s/s/c/c/s under an exclusive `gpulease -n 8`. Artifacts
`/tmp/k3-xr-phase-gate/bench-d1-*.log`, decode trace `/tmp/k3-xr-phase-gate/trace-ctl.raw`.

## Why a rule

The standalone segment was qualified on 2026-09-04 (−0.716 ms/token) but left default-off:
only the k16/H3584/I384/E896 geometry had evidence and the repo forbids a model/shape predicate
(`perf-data/kimi-k3-mi355x-decode-grouped-moe-20260904.md`). This change adds a model-neutral
measured rule instead of a predicate: a `moe_decode_measurement.jsonl` record family keyed by
the exact geometry cell (hardware, n_cu, decode rung, top-k, hidden, local intermediate, expert
count, weight encoding), one record per route, and a selector
(`crates/tunedb/src/moe_decode.rs`) that reroutes only when qualified, current records for
BOTH routes show `standalone_body + handoff ≤ 0.9 x interpreter_body`. The handoff constant
(10.3 µs/layer, gfx950) is the segment transition cost measured in the qualification gate.
`--emit-decode-grouped-moe-segments` / `PLOW_SEG_DECODE_GROUPED_MOE` overrides the rule either
way; with no records the packet is byte-identical to before (`f1bf783d…` verified).

## Network folds (exact ids, `fnv1a64:b7682a38c151ac99` on every fold)

| fold | control TPOT p50 | standalone TPOT p50 |
|---|---:|---:|
| 1 | 28.660 ms | 27.933 ms |
| 2 | 28.620 ms | 27.934 ms |
| 3 | 28.541 ms | 27.934 ms |
| **mean** | **28.607 ms** | **27.934 ms (−0.673 ms, −2.35%)** |

TTFT was 1262.4-1262.9 ms on every fold (prefill unaffected). E2E 8192→256 8,557 → 8,386 ms.

## Route records

- Interpreter pair body: 92 per-layer samples from the one-token decode trace of the control
  (`MOE_GROUP_GLU_FP8_BLK` 17.944 + `MOE_GROUP_DOWN_FP8_BLK` 16.802 µs), median **34.70 µs**
  (p10 34.18, p90 35.31).
- Standalone pair body: network-derived, `interp_median − (control_TPOT − standalone_TPOT)/92 −
  handoff` over the nine per-request standalone TPOTs, median **17.09 µs** (16.99-17.15). The
  isolated harness had measured 16.78 µs for the same object, so the 10.3 µs handoff constant is
  consistent with the network to within 0.3 µs/layer.
- Selection: gain = 34.70 − (17.09 + 10.3) = 7.3 µs/layer = 21% of the interpreter body ≥ the
  10% floor → `Standalone`. Published under campaign `k3-moe-decode-network-derived-20260904`,
  digests `gfx950-e95f0a91a5a3c577` (emitter build label) / `rocm-7.14.0-nix`.

## Verification that the rule reproduces the measured packet

With the two records published against the emitter's build label (`gfx950-e95f0a91a5a3c577`;
the emit log now prints `grouped-MoE decode route: … -> Standalone (projected +7.31 us/layer after
handoff)`), a plain emit with no flags produces `model.pkt` SHA256 `a1f7f6f71f381951557bc43d…`,
byte-identical to the explicit `PLOW_SEG_DECODE_GROUPED_MOE=1` emit and to the standalone packet
that passed the folds above; with the records absent or stale the emit is byte-identical to the
control (`f1bf783d…`). Two pitfalls found on the way and now reported by the emitter: (1) the
records must carry the label the emitter computes for the shipping source (a kernel edit in the
tree changed it and staled every GEMM tile record too), and (2) the cell's weight encoding must
come from the emitter's routed-expert encoding, not the `PLOW_MXFP4` flag (this checkpoint's
experts are MXFP4 by quantization config with the flag unset).

## Decision

Promote by rule: with the two records present the K3 emit selects the standalone route
without any model predicate; any other geometry keeps the interpreter route until both of its
routes are measured and published with `scripts/tune_moe_decode_publish.py`. Rollback:
`PLOW_SEG_DECODE_GROUPED_MOE=0` (explicit) or removing the records.

## Stacked with the GLU UN=7 rung

Same control; candidate = the rule-selected standalone packet with objects built from the source
carrying the GLU rung (`perf-data/mi355x-gemv-glu-un7-20260904.md`), grouped object ON. Three
alternating exact 8192→256 folds: control 28.568 / 28.605 / 28.574 ms vs stacked
27.822 / 27.712 / 27.711 ms TPOT p50 → **−0.834 ms/token (−2.92%)**, checksum
`fnv1a64:b7682a38c151ac99` on every fold, TTFT unchanged. Artifacts `bench-stack-*.log`.
