# Graph-derived phase chains and AQL replay (MI355X, 2026-09-04)

## Decision

Keep phase-object partitioning as a compiler-derived experiment. Do not add a
model-name route. Do not promote AQL prebuild alone: it removes host packet
publication work but does not change the device dispatch boundary.

`build.json.dispatch_chains` now records, for every program and ordered segment:

- exact reachable arms and model-neutral opcode families;
- ordinary vs flash resource class;
- wave64, occupancy, register, spill and private-memory refusal contract;
- one agent-scope barrier chain, committed only after every TP rank is prepared.

The object builder must attach measured resources for each proposed phase
object. A candidate is rejected if occupancy falls below its class or VGPR
spill, SGPR spill, or private memory grows relative to the corresponding
ordinary object. Runtime selection is not enabled by this change.

## AQL boundary gate

Hardware: one idle MI355X/gfx950 leased through `gpulease`. Geometry matches the
prefill interpreter: 256 workgroups x 512 threads, 151,040 B group segment.
Chain length 624 matches the current ordinary 8192 prefill topology. Seven
repetitions, best sample.

| publication path | device period | host enqueue | 624-packet host total |
|---|---:|---:|---:|
| current, reserve/ring each packet | 1.464 us/packet | 0.192 us/packet | 119.8 us |
| contiguous reserve, one doorbell | 1.466 us/packet | 0.003 us/packet | 1.9 us |

Prebuild saves about 118 us of host work over 624 segments and changes device
time by +0.002 us/packet (noise). It cannot close the TTFT gap by itself. Its
value is enabling separately compiled, spill-isolated phase objects without
reintroducing a host drain between them.

The heterogeneous `pub -> check -> bump` chain was also run both ways over 624
packets. Both reached version 208, checked 54,525,952 words, and observed zero
stale words. Agent-scope barrier packets therefore preserve the tested cross-XCD
ordering with a single final doorbell.

All four test objects are wave64 with zero private segment: `d_pub` 6 VGPR/17
SGPR, `d_chk` 6/18, `d_bump` 3/14, and `d_nop` 0/6. No spill metadata is emitted.

## Architecture implication

The existing TP segment-major path already performs one final drain, so the old
6.538 us per-segment drain cost applies only to the diagnostic/fallback path.
Production phase replay should reserve each rank's full chain, fill all ranks,
then ring every rank. Reserving and ringing a complete chain on rank 0 before
preparing rank 1 repeats the recorded TP desynchronization failure and is not an
acceptable implementation.

Proceed only when a compiled phase object demonstrates device-body savings from
lower resource pressure. The first target should be the 8192 prefill collective
segment family because focused XReduce objects are spill-free while the ordinary
prefill interpreter still carries private memory. AQL replay is the transport,
not the performance lever.

## Reproduction

Build `runtime/bench/dispatch/aql_launch_floor.{hip,c}`, unbundle the gfx950 code
object, and run under a one-GPU lease:

```text
aqlfloor --iters 624 --reps 7 --arm chain
aqlfloor --iters 624 --reps 7 --arm prebuild
aqlfloor --iters 624 --arm verify
aqlfloor --iters 624 --arm verify-prebuild
```
