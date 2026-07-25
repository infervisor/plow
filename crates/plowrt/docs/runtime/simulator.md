# §O Simulator / dry-run mode

Loads the compiled packets and walks each one **without a device**, honoring the
counter protocol and ordering, logging what every packet *would* do. Validates a
schedule (dependencies, deadlocks, timing) with no GPU — and shares its trace
format with live runs. Implemented in `src/sim/mod.rs`.

## How it reuses the interpreter

The CPU reference interpreter's counter-gated per-executor FIFO walk was
refactored into `device::cpu::run_streams<O: StepObserver>(program, pool, obs)`.
`interpret` passes a no-op observer; the simulator passes a recording one. There
is **no duplicated gating/ordering logic** — the only differences from a live run
are whether numerics execute and that every fired packet is recorded.

```
StepObserver::run_math() -> bool          // dry run = false, golden = true
StepObserver::on_fire(idx, inst, t0, t1)  // record a SimEvent / TaskSpan
```

## Math modes (`--math`)

- `dry` (default): decode + log every packet, run **no numerics**. Pure
  schedule/counter/timing validation.
- `golden`: also execute the CPU golden op bodies (`device::cpu::execute`) to
  check results against the reference.

## Outputs

**Per-packet log** — one clean line per packet:

```
#1     SM43  GEMM       m=1 n=4096 k=3072 tile=64x256x64 wait[c0] -> succ[c1] t=3572..4840cyc
```

`#seq  <RES><idx>  <NAME>  <body-summary>  wait[..] -> succ[..]  t=start..end cyc`.

**Chrome trace** — each `SimEvent` becomes a `TaskSpan` fed into the existing
`obs::trace::Timeline`; `--chrome out.json` writes chrome://tracing / Perfetto
JSON (`tid` = executor index, `name` = opcode).

**Summary** — packets fired/total + completion, simulated makespan next to the
compiler's `makespan`/`ideal_makespan`, packet counts by opcode family, and the
`CounterMonitor` unsatisfiable-counter report if the walk didn't complete.

## Timing model

Each executor stream carries a cycle clock; a fired packet runs
`[max(stream_clock, waited-producer-finish), + op_cost(body))` so the timeline
respects per-executor ordering *and* cross-executor dependencies. `op_cost` is a
coarse per-family estimate from the packet's own fields (GEMM ∝ tiles×k, Flash ∝
seq×head_dim, DMA ∝ bytes, + a fixed launch overhead). It is a **modelled**
number — reported next to, not as, the compiler's makespan; swap in `costmodel`
for fidelity.

## Dump the trace from a real run too

The same recording observer plugs into `exec::ExecutorSet::run_reference_traced`.
`plowrt serve --trace` records every request's spans into a shared `Timeline`,
dumpable at `GET /trace` in the identical Chrome format. So the simulator and a
live serve emit the same trace.

## CLI

```sh
# dry-run the first bucket, per-packet log to stdout
plowrt simulate --assets ./plow-out

# a specific bucket, golden math, log + Chrome trace to files
plowrt simulate --assets ./plow-out --bucket decode:1:128 \
  --math golden --log sim.log --chrome trace.json

# every compiled bucket, combined trace
plowrt simulate --assets ./plow-out --all-buckets --chrome trace.json
```

Exit status is non-zero if any bucket fails to complete (deadlock), so it doubles
as a CI schedule-validity check.

## Example (real `plowc` output, gemma4-12b decode, b1 s128)

```
=== bucket Decode b1 s128 (163 packets) ===
#0     SM18  ROW_REDUCE rows=1 feat=3072 br=1            wait[-]  -> succ[c0] t=0..3572cyc
#1     SM43  GEMM       m=1 n=4096 k=3072 tile=64x256x64 wait[c0] -> succ[c1] t=3572..4840cyc
...
packets: 163 fired / 163 total  (completed)
simulated makespan: 70580 cyc  (compiler makespan 17458, ideal 17458)
by family: GEMM=136 FLASH=12 TMA_STORE=12 ROW_REDUCE=2 ROW_PW=1
```
