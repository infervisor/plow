# GLM prefill collective phases on MI300X

[mi300x-phases.json](mi300x-phases.json) records all-rank phase traces for the
156 two-shot collectives in the final 8191-row prefill chunk. Each collective
has all 304 workgroup records on all eight ranks. The parser excludes stale
records using the latest Embed completion and rejects missing slices,
saturated timestamps, and invalid phase order.

Reduce-scatter and all-gather account for 81.62% of the summed body phases of
the last-finishing workgroup in each collective/rank. The remaining phases
include gate signaling, waiting, acquire, and local barriers. Per-rank sums
of these selected workgroup bodies range from 144.44 to 164.10 ms. These are
collective diagnostics: overlapping envelopes and workgroup phase sums are
not an attribution of the full prefill wall time. GPU clocks are never
subtracted across ranks.

The diagnostic prefills 65,535 copies of token ID 1, in eight chunks, with
one exclusive eight-MI300X lease. Both arms agree across ranks on the sampled
token. Control prefill takes 7255.6 ms; the instrumented run takes 7268.9 ms.
Final-chunk drain times are 1000.347 and 1002.673 ms. A single repeated-token
pair is not serving performance or broad model-quality evidence.

The serving prefill image was built from `0a4ee6e3`. Its source hash matches
the original FP8 runtime evidence. Rebuilding that source reproduces `.text`,
`.rodata`, `.data`, and resource metadata exactly. The phase image differs
only by `PLOW_XR_TRACE_PHASES=1`; it raises LDS by 16 bytes, private storage by
16 bytes, and both SGPR/VGPR spill counts by two. Its timings therefore carry
instrumentation cost.

This check also found that the frozen serving prefill image predates the
ragged-fold fix in `9a3a7763`. A build from current source is not an identical
control. The refreshed asset is qualified separately below.
The record includes exact build/trace recipes, source and image hashes,
per-collective phase rows, and raw trace hashes. No production defaults change.

## Refreshed prefill image

[mi300x-refresh.json](mi300x-refresh.json) compares the current-source default
FP8 prefill GQ image against the original image. This includes the ragged-fold
fix and local-selector helper/marker; it does not isolate either change.
Phase instrumentation is disabled. Only this interpreter image differs;
both arms use the same runtime and 633-native-GEMM packet, with local selection
enabled, native fold disabled, and decode tiers disabled.

Each arm passes 18/18 retrieval checks at concurrency 20 and completes all 20
random serving requests without failures. Input/output length arrays match
exactly: 1,414,538 input tokens and 13,795 generated tokens, without speculation.

| Metric | Refreshed | Original | Change |
| --- | ---: | ---: | ---: |
| Output throughput (tok/s) | 44.003 | 42.747 | +2.94% |
| Mean TTFT (ms) | 111116.34 | 112266.90 | -1.02% |
| Mean TPOT (ms) | 271.57 | 274.23 | -0.97% |
| P99 TPOT (ms) | 405.41 | 406.63 | -0.30% |
| Median ITL (ms) | 134.39 | 138.67 | -3.08% |

One exclusive eight-MI300X lease covers both arms, with no concurrent builds
or other GPU work. This is one matched pair without a repeatability estimate.
Only 4/20 generated texts match exactly; retrieval checks do not establish
broad model-quality equivalence. The 100-request H200 target remains unmet.
The record includes build provenance, artifact hashes, and the campaign recipe.

## Current full-ladder trace and specialist sweep

The `5c26d90f` production runtime path was traced with the full 21-program
packet and default two-split sparse adapter. A cold 65,536-token request took
7489.89 ms without tracing and 7514.64 ms with tracing; prompt and output
checksums match. The final 8192-row chunk has complete opcode/workgroup
coverage for 1909 non-native instructions on each of eight ranks. Native
GEMM, MoE, index selection and sparse MLA do not write these trace records.

Rank 0's span after Embed is 962.79 ms. Summed completion tails are 161.21 ms
for XReduceTwoShot, 51.79 ms for GEMM, 45.45 ms for MLA merge/fold and 33.56 ms
for MoE combine. These envelopes overlap and are not wall-time percentages.
Raw traces, packet disassembly and parser checks are under
`/tmp/tp-glm53-prefill-instructions`.

`specialist_probe.hip` wraps the production `d_xreduce_twoshot_mega` body in
a dedicated kernel. The host probe runs all eight GPUs concurrently, varying
four/eight waves, the experimental wave-RS schedule, and 38/76/152/304
workgroups. Aggregation is enabled as in the serving prefill object. Each of
84 cells checks every output against an independent BF16/FP32 rank-order
oracle on all ranks, three changing input reuses, output/partial guards and
both rendezvous counts. All pass; all three kernels have zero private storage.

| Rows | Four waves, 304 WGs, us | Best measured choice | Best, us |
|---|---:|---|---:|
| 1 | 52.72 | Eight waves, 38 WGs | 43.43 |
| 128 | 75.47 | Eight waves, 76 WGs | 63.94 |
| 129 | 77.60 | Wave-RS object, 76 WGs | 62.93 |
| 512 | 128.63 | Eight waves, 76 WGs | 108.00 |
| 2048 | 283.26 | Eight waves, 152 WGs | 268.08 |
| 8191 | 908.48 | Wave-RS object, 152 WGs | 866.73 |
| 8192 | 902.95 | Eight waves, 152 WGs | 873.10 |

The wave-RS object falls back to its scalar body for the ragged 129/8191
cases. At 8192 rows, eight waves with 304 workgroups takes 1178.24 us,
30.5% slower than four waves. Do not enable eight waves globally from these
results. The large-rung gain is only 3–5% in isolation; no interpreter,
segment-selection or serving default changes follow from this sweep.

Times are medians of nine samples after five discarded samples, taking the
maximum within-rank GPU event duration. Input initialization and CPU checks
are outside timing. Host launches ranks sequentially; arm order is fixed.
The host baseline always uses 304 workgroups and therefore does not reproduce
the compiler's small-rung workgroup cap. Raw results and image/header hashes
are under `/tmp/tp-glm53-collective-specialist`.

Reproduce inside `nix develop`, with no other GPU work or compilation during
the timed probe:

```sh
out=$(mktemp -d /tmp/plow-collective.XXXXXX)
bench=runtime/bench/amd/glm_prefill_collective
for arm in four eight wave_rs; do
  flags=(-DPLOW_WG_WAVES=8)
  if [ "$arm" = four ]; then flags=(-DPLOW_WG_WAVES=4); fi
  if [ "$arm" = wave_rs ]; then flags+=(-DPLOW_XR_WAVE_RS=1); fi
  "$PLOW_HIPCC" --genco --offload-arch=gfx942 -O3 -w -std=c++17 \
    -DPLOW_XR_AGG=1 "${flags[@]}" -Iruntime/amd -Iruntime/common \
    "$bench/specialist_probe.hip" -o "$out/$arm.co"
  "$PLOW_BUNDLER" --unbundle --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
    --input="$out/$arm.co" --output="$out/$arm.elf"
done
c++ -std=c++17 -O3 -D__HIP_PLATFORM_AMD__ -I"$ROCM_PATH/include" \
  "$bench/specialist_probe.cpp" -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" -lamdhip64 -o "$out/probe"
GPU_LEASE_NGPU=8 perf-data/tools/gpulease -n 8 collective-specialist \
  "$out/probe" "$out/four.elf" "$out/eight.elf" "$out/wave_rs.elf"
```
