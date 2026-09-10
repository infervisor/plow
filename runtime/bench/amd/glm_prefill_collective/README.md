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
