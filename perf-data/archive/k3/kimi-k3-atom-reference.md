# Kimi-K3 on AMD's own ATOM stack — the reference plow is aiming at (2026-07-29)

**Why this exists.** plow has no K3 model-level emitter at all (`kimi_k3_emit` never returns,
`crates/devgen/src/lib.rs:4195`), so there is nothing to compare against yet. This is the *target*,
measured on the vendor's own image rather than quoted from a blog post. Per the standing rule, K3 is
never brought up against vLLM — the AMD reference is what we aim at.

## Provenance

| | |
|---|---|
| image | `rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3` |
| recipe | verbatim from `amd.com/.../2026/kimi-k3-on-amd-instinct-gpus.html` |
| server | `atom.entrypoints.openai_server --model moonshotai/Kimi-K3 --kv_cache_dtype fp8 -tp 8 --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 10240 --gpu-memory-utilization 0.93 --block-size 128 --no-enable_prefix_caching` |
| hardware | MI355X x8, all 8 cards, box otherwise idle (verified free before start) |
| client | direct OpenAI `/v1/chat/completions`, non-streaming |
| metrics | **server-reported** `usage.ttft_s` / `usage.tpot_s`, not client-side stamps |
| shape | 1488 prompt tokens / 128 completion tokens, **concurrency 1**, n=10, first 2 discarded |

`--no-enable_prefix_caching` is in the vendor recipe, and it matters here: it means the repeated
identical prompt is *not* being served from a cache, which is the defect that made the GLM-5.2
vLLM TTFT column illegal (`glm52-precision-symmetry.md` §5). The 10 requests are genuinely
re-prefilled.

## Result

| metric | mean | median |
|---|--:|--:|
| TTFT | 465.1 ms | **338.4 ms** |
| TPOT | 55.25 ms | **53.35 ms** |
| decode rate | | **18.7 tok/s** |

Per-request, in order (`ttft_s` / `tpot_s`):

```
8.8139/0.0522   0.2868/0.0645   0.8377/0.0506   0.3106/0.0499   0.5350/0.0543
0.7979/0.0662   0.3381/0.0557   0.3386/0.0524   0.2842/0.0477   0.2784/0.0652
```

Request 0 carries an 8.81 s TTFT — first-touch, discarded with request 1. TPOT is stable from the
first request (0.0522 on req 0 against a 0.0534 warm median), so the warm-up is a prefill-path
effect, not a decode one.

## The published 111 tok/s is NOT this number, and the difference is not a discrepancy

AMD's article quotes **111 tok/s at TP8**. This measures **18.7 tok/s**, 5.9x lower, on the vendor's
own image and recipe. Both can be true: this is **per-request decode rate at concurrency 1**, and
111 tok/s is almost certainly **aggregate output throughput with the server's batching engaged** —
the recipe ships `--max-num-seqs 64` and `--max-num-batched-tokens 10240`, neither of which does
anything at concurrency 1.

**Do not compare plow's conc-1 TPOT against 111 tok/s.** They measure different quantities. The
number on this page is the like-for-like target, because it is the shape plow's own GLM-5.2 numbers
are taken at. If an aggregate-throughput comparison is ever wanted, it has to be re-measured here at
matching concurrency — the box was released before that could be done.

## What was NOT measured, and why

* **No concurrency sweep.** The box was needed for the GLM-5.2 decode re-run, which is the active
  goal. The container holds ~265 GiB on all 8 cards and cannot share.
* **No coherence gate.** The single sanity completion was fluent and on-topic ("The user is asking
  about the capital of France..."), but the 10 timed requests used random-word prompts whose output
  was not checked. Per `glm52-ctx-sweep.md`, the recorded long-context failure mode on MoE models is
  degenerate text that runs *faster*, so these timings carry the usual caveat: **timed, not
  verified.** For a vendor stack serving its own model that risk is low, but it is not zero.
* **No TTFT breakdown.** 338 ms for 1488 tokens is ~4,400 tok/s of prefill. Recorded for scale
  against plow's GLM-5.2 ~750 tok/s (a different model — not a valid comparison, only a magnitude).

## Operational note — the harness defect this exposed

`$CLAUDE_JOB_DIR/tmp/k3_atom_baseline.sh` runs the container with `--rm -d` and **never tears it
down**. It waits for a fully idle box, starts an 8-card TP8 server, runs one sanity completion,
touches a `.done` file, and exits — leaving the container holding every GPU indefinitely. It sat
there ~1 h and would have blocked the GLM re-run forever.

Any future long-lived-server harness needs an explicit teardown, or a `--rm` container plus a trap.
A "low priority, only when the box is free" job that never releases the box is not low priority.
