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
control. Qualifying the refreshed FP8 prefill asset is a separate next step.
The record includes exact build/trace recipes, source and image hashes,
per-collective phase rows, and raw trace hashes. No production defaults change.
