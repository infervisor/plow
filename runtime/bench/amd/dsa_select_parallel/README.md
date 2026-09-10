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
512-byte output guard. These isolated measurements do not establish runtime
support at batch widths above 16.

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

## Local workgroup selection

`--glm-select-local=true` (`PLOW_GLM_SELECT_LOCAL=1`) opts gfx942 TP8 decode
rows 2/4/8/16/20 into the existing prefill radix selector, rebased to one sequence
per workgroup. Row 1 and pooled selection retain the cooperative path. The
selector stays inside the interpreter, preserves per-XCD placement, and adds
no raw launches or cross-workgroup barriers. Every loaded decode tier must
advertise `plow_dsa_select_local_arm`; incompatible packets and objects are
rejected before execution.

Benchmark mode 3 passes all 132 synthetic and seven captured cases, including
short rows padded with -1, scratch reuse and permuted logical slices. Captured
B8 median time falls from 325.459 to 184.848 microseconds (43.2%). This compares
complete batches outside the interpreter; it does not establish a serving gain.
See [local selection evidence](mi300x-local.json).

The full model completed 18/18 retrieval cases at concurrency 8, including
partially occupied decode rungs. The matched on-then-off 20-request serving screen
completed without failures and with identical input/output lengths:

| Metric | Cooperative | Local workgroup | Change |
|---|---:|---:|---:|
| Output throughput, tok/s | 31.346 | 33.013 | +5.32% |
| Mean TPOT, ms | 195.677 | 180.850 | -7.58% |
| P99 TPOT, ms | 248.822 | 232.547 | -6.54% |
| Median ITL, ms | 117.715 | 107.200 | -8.93% |

Both arms passed all 18 retrieval cases. Only 3/20 random generated texts match
exactly; selected-index ordering can change downstream accumulation, so broader
quality equivalence remains unqualified. This is one matched pair, without a
repeat or confidence interval; it is not the 100-request H200 comparison. The
option remains off by default.

### FP8-KV batch 20

The local selector also supports rung 20. Its full-model packet replaces 420
serial selector instructions with 21 independent-row selectors. Prefill and
row 1 are unchanged; the disabled packet is byte-identical to the preceding
native-decode-GEMM packet. 179 AMD runtime tests, 36 GLM emitter tests, the release
build and all ten programs' Lean ordering/LDS checks pass.

The matched 20-request random70k/700/.14/C20 seed0 serving pair uses the same
FP8-KV B20 model and main MM16/WALK1 image in both arms, with native decode GEMM
enabled and native fold disabled. Narrow tiers are explicitly disabled in both
arms: the existing main image contains the local selector, but the older FP8
tier images do not. No kernel was rebuilt for this comparison. One exclusive
eight-GPU lease covers both arms, without concurrent builds or other GPU work.

| Metric | Cooperative | Local workgroup | Change |
|---|---:|---:|---:|
| Output throughput, tok/s | 38.523 | 42.604 | +10.59% |
| Mean TTFT, ms | 114802.69 | 112565.06 | -1.95% |
| Mean TPOT, ms | 315.860 | 281.661 | -10.83% |
| P99 TPOT, ms | 443.242 | 422.664 | -4.64% |
| Median ITL, ms | 181.884 | 147.288 | -19.02% |

Both arms complete 20 requests without failures and pass all 18 retrieval cases
at C20. Input/output length arrays match exactly: 1,414,538 input tokens and
13,795 output tokens. Four of 20 generated texts match exactly. This is one
pair, without a repeatability estimate or broad model-quality qualification;
the option remains false by default. It does not establish parity with the
100-request H200 reference. [Batch-20 evidence](mi300x-local-b20.json) includes
the source/object hashes, packet checks, quality outputs and reproduction recipes.

## Per-XCD defaults and prefill pairing

Per-XCD packet queues already default on for gfx942/gfx950 decode. The GLM TP8
run confirms eight domains and an active hierarchical gate on every rank.
These device-side queues are distinct from separate HSA submission queues.

GLM prefill placement remains opt-in (`--glm-place-pf=true`), paired with
`PLOW_L2HIER_PF=1` gfx942 prefill objects. Its emitter now preserves ordered
native-kernel segments when placement is enabled. Previously placement collapsed
those segments, making the native routes invalid. A full-emitter regression test
checks native MoE isolation; the full GLM packet preserves all 1,784 segment
descriptors and instruction operands, with decode unchanged.

The matched 20-request serving pair completed with zero failures and 18/18
retrieval checks in each arm. Placement increased output throughput from
30.861 to 31.546 tok/s (+2.22%) and reduced mean TPOT from 195.056 to
192.324 ms (-1.40%). P99 TPOT increased from 234.631 to 259.335 ms (+10.53%).
This mixed result leaves prefill placement opt-in. See the
[paired results and provenance](mi300x-prefill-xcd.json). One pair does not
establish a repeatable improvement or the 100-request H200 target.
