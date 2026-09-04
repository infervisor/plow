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

`PLOW_TRACE_ALLRANKS=1` makes the `amd-bench` TP harness dump one file per rank. It does not
change serving behavior. `scripts/xreduce_phase_report.py` uses only within-rank timestamp
differences; it does not assume clocks on separate GPUs share an epoch. For each collective it
reports rank-0 publish wait, inferred latest rank, cross-rank wait spread, remaining two-shot
time, immediate producer opcode/dimensions, and message class. The inferred arrival skew is
`max(publish_to_ready) - min(publish_to_ready)`; system signal/observation variation remains in
that value and is the residual uncertainty.

## Run gate

Build a diagnostic prefill object with `PLOW_XR_TRACE_PHASES=1`, retain the packet-paired
production axes, and run one exact TP8 T8192 prefill after the active sizing lease releases:

```sh
PLOW_TRACE_RAW=/tmp/xr-phase PLOW_TRACE_ALLRANKS=1 \
  plowrt amd-bench ...
RUST_LOG=error plowrt disasm <assets>/model.pkt --program 8192 --format json \
  > /tmp/xr-phase.disasm.json
python3 scripts/xreduce_phase_report.py /tmp/xr-phase.disasm.json \
  /tmp/xr-phase.rk{0,1,2,3,4,5,6,7}.prefill
```

Acceptance is complete coverage of all 278 collectives on all eight ranks, no xctr audit
failure, and unchanged final exact oracle. Rank producer families by summed inferred skew.
Only then choose a producer scheduling or producer-to-collective watermark experiment.

## Pinned vLLM/AITER comparison

vLLM 0.28 pins AITER 0.1.19. Its default custom-all-reduce cutoff is 64 MiB even though the
registered pool is 128 MiB: 28 and 56 MiB select its custom two-stage kernel, while 117 MiB
normally falls through to another backend. The custom kernel uses 80 workgroups of 512 threads,
16-byte packs, one peer per wave staged through LDS, and per-block peer signal stores. Its peer
list is rotated by local rank, so its floating-point addition order is rank-dependent. That
ordering cannot replace Plow's strict global rank-0-through-rank-7 accumulation. Plow's prior
strict-order all-wave/LDS/16-byte experiment also regressed every isolated class, so this trace
does not reopen that rejected body port.
