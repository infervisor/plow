# Chunk-KDA gfx950 pre-production gate

Date: 2026-09-03. Device: `gfx950:sramecc+:xnack-`, 256 CUs. Shape: TP8-local
`H=12, D=128, V=128`, BT64/BC16, bf16 operands and f32 V-first state. Run with:

```sh
nix develop --command scripts/bench_kda_chunk_gfx950.sh
```

The serial and chunk paths receive identical normalized q/k, v, raw gate/beta, initial state,
scale, and 512-thread/256-workgroup launches. Chunk time includes gate-prefix, full intra solve,
W/U transform, and ordered carry launches. State reset copies are outside event timing.

| T | serial ms | chunk ms | speedup | prefix ms | intra ms | W/U ms | carry ms | output RMS rel | state RMS rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.1554 | 0.3847 | 0.404x | 0.0081 | 0.3049 | 0.0130 | 0.0564 | 5.7516e-3 | 4.9735e-3 |
| 511 | 0.7495 | 0.5904 | 1.269x | 0.0154 | 0.3058 | 0.0369 | 0.2406 | 5.8195e-3 | 5.0670e-3 |
| 512 | 0.7528 | 0.5905 | 1.275x | 0.0155 | 0.3064 | 0.0383 | 0.2407 | 5.8191e-3 | 4.9762e-3 |
| 513 | 0.7605 | 0.6035 | 1.260x | 0.0165 | 0.3065 | 0.0379 | 0.2528 | 5.8163e-3 | 3.2255e-3 |
| 2048 | 2.9161 | 1.7094 | 1.706x | 0.0484 | 0.6103 | 0.1312 | 0.9338 | 5.8259e-3 | 4.6915e-3 |
| 8191 | 11.8100 | 6.2139 | 1.901x | 0.1920 | 1.8235 | 0.5078 | 3.7016 | 5.8285e-3 | 4.8748e-3 |
| 8192 | 11.8030 | 6.2225 | 1.897x | 0.1909 | 1.8238 | 0.5084 | 3.7080 | 5.8285e-3 | 4.8086e-3 |

The measured crossover is between 128 and 512 tokens; the first tested winning rung is 512.
Carry dominates at long context, while the fixed ~0.30 ms full-intra cost makes T=128 lose.

## Precision attribution

The f64 oracle isolates the gate prefix and triangular solve before the composed recurrence. Gate
prefix max-absolute error is `1.859e-5` (bounded gate) and `5.448e-5` (softplus); the full BT64
QK product is `2.838e-9` and inverse/triangular solve is `4.149e-8`. These stages are not the
source of the ~0.6% final-state drift.

The only precision change in the sweep above is the state-update operand
`K * exp2(g_last - g)`: it is represented as a bf16 high part plus bf16 residual and accumulated
by two native MFMA instructions into the same f32 accumulator. Against the prior single-bf16
staging run, state RMS falls from `5.9546/5.9462/5.7103/5.7394e-3` to
`4.9735/4.9762/4.6915/4.8086e-3`, removing 30-33% of squared state error. Output RMS is unchanged,
so scaled-K staging—not the f32 carry accumulator—is the dominant contributor to the failed
final-state bound. W/U and V' remain bf16 and are the remaining output-side error candidates; this
experiment does not distinguish them because neither changed, and neither needs widening for the
current 1% output and 0.5% state gates.

## Resources and object size

| kernel | VGPR | LDS | scratch |
|---|---:|---:|---:|
| serial recurrence | 62 | 2,112 B | 0 |
| gate prefix | 28 | 0 | 0 |
| full BT64 intra | 76 | 16,384 B | 0 |
| W/U transform | 84 | 0 | 0 |
| ordered carry | 163 | 14,336 B | 0 |

The standalone gfx950 test code object grows from 20,272 B at `HEAD` to 73,280 B with all chunk
wrappers: +53,008 B (+261.5%, 3.61x total). The separately paired production prefill interpreter
grows from 292,600 B to 335,824 B: +43,224 B (+14.77%). Its occupancy-setting metadata is unchanged
at 256 VGPR, 0 AGPR, 147,504 B LDS, 1,332 B private segment and four VGPR spills; SGPR count remains
108 while the metadata spill count rises from 76 to 78. Both gfx950 and gfx942 objects compile, and
only the opt-in object exports `plow_kda_chunk_bt64_arm_1`.

## Gate decision

The default remains the serial recurrence. An explicit compile flag now emits a four-op chunk
packet only for supported single-sequence programs compiled at T>=512, pairs it with an exact
capability marker, and leaves smaller/unsupported programs on serial. At dispatch, each op derives
`ceil(T/64)` and bounds its final chunk from the runtime-rebased token count; 511, 513, and 8191
exercise the 63-, 1-, and 63-row tails. The standalone numerical gate passes at production rungs:
output stays below 1%, final-state drift is 0.469-0.498%, and every tested T>=512 wins. The 511 result
is a safety oracle for a 512-row compiled program shortened by ragged execution, not an emitter
crossover decision.

The benchmark still calls the same production bodies through standalone wrappers rather than a
full persistent-interpreter packet stream. Required next work before making the flag default:

1. Validate the same bound against a trusted FLA/vLLM end-to-end oracle.
2. Run a full persistent-interpreter packet oracle and full-model prefill A/B.
3. Re-run packed multi-sequence parity before enabling mux dispatch; chunk emission currently
   refuses sequence-row programs.
