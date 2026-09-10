# Parallel GLM decode selection on MI300X

The previous B8 decoder emits eight serial `IndexSelect` instructions at each
of 21 full indexer layers. Each selection uses 32 cooperative workgroups and
shares its radix histogram/control with the next row.

A repeated-token diagnostic fills eight slots with 65,000 copies of token ID 1
using the real GLM-5.3 weights, then decodes three steps. The last instrumented
step spans 106.742 ms. Summed selection bodies account for 14.870 ms across
168 instructions; GEMV bodies account for 36.548 ms and MoE GLU 20.520 ms.
Per-instruction stalls overlap: their sum is not recoverable wall time. This
is a diagnostic input, not the random serving workload or a quality evaluation.

## Isolated qualification

`bench.hip` uses the existing `d_index_select_coop` algorithm in three forms:

- A serial chain of one kernel per row, with shared scratch.
- One kernel iterating over rows, with a completion barrier before scratch reuse.
- Independent rows in one grid, with separate histograms and control cache lines.

It compares complete selected sets against CPU score ordering with the same
lowest-index tie-break. All 111 synthetic cases and six captured cases pass.
Coverage includes batches 1/2/4/8/16/20/32, live lengths
0/1/129/2047/2048/8192/65537/79800, tied and unique scores, permuted logical
workgroup slices, twelve scratch reuses, untouched inactive output and a
512-byte output guard. Batch widths above 16 are isolated kernel experiments;
they do not establish runtime support.

The capture contains the last indexer's eight score rows after decode step 2
at 65,003 live KV positions. Its selected sets also match the CPU oracle.
Timing uses ten GPU-event samples, each covering ten graph-captured operations,
after two graph warmups. Allocation, copies and the CPU oracle are excluded.

| B8 selection | Captured scores, µs | Synthetic 70k scores, µs |
|---|---:|---:|
| Serial kernel chain, 32 WGs/row | 329.023 | 382.579 |
| Serial device loop, 32 WGs/row | 370.284 | 414.641 |
| Parallel, 8 WGs/row | 62.676 | 68.058 |
| Parallel, 16 WGs/row | 51.805 | 60.570 |
| Parallel, 32 WGs/row | 51.090 | 61.175 |

See [synthetic samples](mi300x-synthetic.json), [captured samples](mi300x-captured.json)
and [capture/trace provenance](qualification.json). The parallel implementation
changes concurrency and scratch ownership; it retains the radix algorithm.
These isolated timings do not establish a serving speedup.

Inside `nix develop`:

```sh
bench=runtime/bench/amd/dsa_select_parallel
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 \
  -Iruntime/amd -Iruntime/common -c "$bench/bench.hip" -o /tmp/select.o
c++ /tmp/select.o -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" \
  -lamdhip64 -o /tmp/select
perf-data/tools/gpulease -n 8 glm-select /tmp/select > /tmp/select-synthetic.json
perf-data/tools/gpulease -n 8 glm-select-capture /tmp/select \
  /path/to/score.b002.bin /path/to/idx.b002.bin > /tmp/select-captured.json
```

The qualification record contains the exact capture driver. It uses the prior
qualified packet/runtime hashes, all eight GPUs, `PLOW_TRACE_RAW`, and
`PLOW_DUMP_ACT=act.iscore:.../score,act.iidx:.../idx`. The capture argument form
expects eight rows at capacity 81,920 and live length 65,003.

## Interpreter integration: rejected

An experimental packet assigned 16 workgroups per row with private histograms
and control cache lines. It passed emitter, loader and packet tests, and Lean
verified all eight ranks. The option-off packet was byte-identical to the
previous qualified packet.

Both the global-queue and static decode serving runs stalled after entering
rung 4 with three occupied slots. Neither completed a quality case. Each was
stopped with SIGTERM. The same rebuilt objects and runtime with the unchanged
serial packet completed all 18 retrieval cases at concurrency 8. This isolates
the failure to the candidate integration; the exact cause is unresolved.

The experimental emitter, ISA and interpreter changes were removed. Only this
benchmark and its evidence are retained. No parallel-selection serving speedup
is claimed, and no assembly change is justified by these measurements.
