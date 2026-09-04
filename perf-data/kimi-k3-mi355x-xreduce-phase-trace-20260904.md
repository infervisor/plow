# TP8 prefill XReduce phase diagnostic (2026-09-04)

## Why this trace is the next gate

The current T8192 packet trace attributes 209.477 ms of body time and 0.739 ms of outer
gate time to 278 `XReduceTwoShot` packets: 753.5 us of body per packet. The exact isolated
256-workgroup gates are 38.487 us (half), 72.976 us (full), and 96.434 us (folded gather),
or 19.272 ms weighted over 92/94/92 calls. The integrated body is therefore about 10.9x the
isolated projection. The measured 42.2 us packet straggler is also too small to explain the
684 us per-packet excess by itself.

The diagnostic must identify the rank that reaches each collective last before changing the
collective body. A rank-0-only trace cannot do that.

## Instrument

`PLOW_XR_TRACE_PHASES=1` is a compile-time, default-off instrument. For an
`XReduceTwoShot` record only, the existing 40-byte `PlowTraceRec` means:

- `t_arrive`: workgroup entered the two-shot body after its local graph dependency cleared.
- `t_arrive + pc`: that workgroup passed the local-partial publish code. Slice 0 is the rank's
  actual cross-rank publisher.
- `t_ready`: the initial all-rank partial-ready rendezvous cleared and its system acquire
  completed.
- `t_end`: all reduce-scatter, second rendezvous, and all-gather work completed.

The high bit of `slice` marks this alternate schema; the analyzer rejects an ordinary trace
instead of silently interpreting its stream index as a publish delta.

All other opcodes retain the normal trace schema. The instrument changes no values, counters,
wait thresholds, reduction order, or production object when disabled. Its focused gfx950
compile is spill-free: 76 VGPR, 104 SGPR, occupancy 6, 16 B LDS, zero scratch and zero spills.
The flag-off `.text` is byte-identical to the pre-change object. The trace arm adds 2 SGPR and
12 B LDS relative to the focused control, so timings are diagnostic attribution rather than a
production performance number.

`PLOW_TRACE_ALLRANKS=1` makes the TP benchmark paths dump one file per rank. It does not
change serving behavior. `plowrt bench` names them `trace.rkN`; `amd-bench` additionally
preserves the prefill dump as `trace.rkN.prefill`. `scripts/xreduce_phase_report.py` uses only
within-rank timestamp
differences; it does not assume clocks on separate GPUs share an epoch. For each collective it
reports rank-0 publish wait, inferred latest rank, cross-rank wait spread, remaining two-shot
time, within-rank ready spread, per-workgroup post-ready duration, envelope, immediate producer
opcode/dimensions, and message class. The inferred arrival skew is
`max(publish_to_ready) - min(publish_to_ready)`; system signal/observation variation remains in
that value and is the residual uncertainty.

## Run gate

Build a diagnostic prefill object with `PLOW_XR_TRACE_PHASES=1`, retain the packet-paired
production axes, and run one exact TP8 T8192 prefill after the active sizing lease releases:

```sh
PLOW_TRACE_RAW=/tmp/xr-phase PLOW_TRACE_ALLRANKS=1 \
  plowrt bench --assets <assets> --random-input-len 8192 --output-len 1 \
    --requests 1 --warmup-requests 0 --concurrency 1 --parity-report \
    --engine-diagnostics
RUST_LOG=error plowrt disasm <assets>/model.pkt --program 8192 --format json \
  > /tmp/xr-phase.disasm.json
python3 scripts/xreduce_phase_report.py /tmp/xr-phase.disasm.json \
  /tmp/xr-phase.rk{0,1,2,3,4,5,6,7}
```

Acceptance is complete coverage of all 278 collectives on all eight ranks, no xctr audit
failure, and unchanged final exact oracle. Rank producer families by summed inferred skew.
Only then choose a producer scheduling or producer-to-collective watermark experiment.

## Exact TP8 result

The audited T8192 run completed with one output token (`6896`), no failed request, all-rank
prefill completion, and counter audit on every dispatch. TTFT was 1415.478 ms. Every rank dump
is exactly 29,025,160 bytes and covers all 278 collectives. `output-len=1` is important here:
prefill is the last and only dispatched program, so the serve-bench dump cannot be overwritten
by decode.

Provenance:

- packet `f1bf783dac96791b7116ffb549862c8206ba33351310c7c113504916611e8921`
- config `9ac0ff4d022d0a5903794edf4a43e325e968d8378c2464c9c56ef46037cdf6ef`
- diagnostic source `d7cbf687fce1a56e75e6795dd018b02ee8f65a44`
- diagnostic runtime `2bcf2fb83695e70808c09a6d598b91ee049bbd44e9549a087500fdad22ae8791`
- prefill GQ object `f4cfba8886301bb568e55885559d1b659d45a93e1ac51bee9badd370cc96e914`

The lease wrapper returned 1 only because its postcondition looked for the newer
`tp_prefill_segment_major` debug field in the deliberately pinned pre-segment runtime. The model
run, exact/audit gates, eight trace files, and analyzer coverage all passed; no GPU retry is
needed.

The 100 ticks/us conversion is confirmed by this run itself: the eight within-rank trace spans
are 1400.820--1401.045 ms, versus 1415.478 ms host TTFT. A 1 GHz interpretation would imply an
impossible 140.1 ms device span.

| immediate producer | class | calls | cross-rank skew sum | max ready-spread sum | WG post-ready p50/p90/max | envelope sum |
|---|---:|---:|---:|---:|---:|---:|
| `MoeCombinePf` | half | 92 | 0.363 ms | 0.829 ms | 415 / 430 / 462 us | 41.801 ms |
| `GemmC5` | full | 94 | 1.316 ms | 0.904 ms | 750 / 777 / 847 us | 75.879 ms |
| `GemmC5` | gather | 92 | 2.709 ms | 0.749 ms | 990 / 1007 / 1039 us | 94.339 ms |

Cross-rank producer skew totals only 4.388 ms. The maximum within-rank spread between
workgroups clearing the first rendezvous totals only 2.482 ms. Neither explains the integrated
gap. Nearly every workgroup spends the long interval after the first rendezvous: the dominant
cost is inside reduce-scatter, the second rendezvous/cache-maintenance handoff, or all-gather,
not producer arrival or a late global-queue workgroup. The next gate should split this interval
at reduce-scatter completion and use the existing no-wait/no-signal controls to price the second
handoff before changing production routing.

## Pinned vLLM/AITER comparison

vLLM 0.28 pins AITER 0.1.19. Its default custom-all-reduce cutoff is 64 MiB even though the
registered pool is 128 MiB: 28 and 56 MiB select its custom two-stage kernel, while 117 MiB
normally falls through to another backend. The custom kernel uses 80 workgroups of 512 threads,
16-byte packs, one peer per wave staged through LDS, and per-block peer signal stores. Its peer
list is rotated by local rank, so its floating-point addition order is rank-dependent. That
ordering cannot replace Plow's strict global rank-0-through-rank-7 accumulation. Plow's prior
strict-order all-wave/LDS/16-byte experiment also regressed every isolated class, so this trace
does not reopen that rejected body port.
