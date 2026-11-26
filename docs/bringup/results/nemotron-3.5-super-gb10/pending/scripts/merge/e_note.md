## Cross-target impact (merge to main)

Behaviour this campaign changed for every target is gated so that only the GB10 recipes
(`scripts/campaign/recipes/nemotron-3.5-super.gb10.nvfp4.toml`, `qwen38-27b.gb10.fp8.toml`) keep it;
every other target (H100, L4/L40S, RTX 5090, AMD, CPU) keeps the a7b444bad behaviour by default.

| change | gate | GB10 recipes |
|---|---|---|
| cold-start burst hold (mux waits up to one SLO for requests still rendering; chat counts ingress before its template) | `PLOW_PF_COLD_BURST`, default 0 | `[serve.env]` = 1 |
| several serial prefill chunks (and Mamba packs) per tick within one widest launch's measured time | `PLOW_PF_TICK_BUDGET`, default 0 | `[serve.env]` = 1 |
| serial prefill planner prices a launch from the measured launch fit instead of 512 rows | `PLOW_PF_CHUNK_COST_FIT`, default 0 | `[serve.env]` = 1 |
| speculation after a prefill tick whose launch did not carry the decode rows | `PLOW_SPEC_AFTER_PREFILL`, default 0 | `[serve.env]` = 1 |
| recurrent prefix cache captures a prompt end only when a later request is expected to attach it | packet with an MTP drafter cache (`kv.mtp.k`) or `PLOW_VMM_PROMPT_END_EXPECTED` | both have the MTP cache; `[serve.env]` = 1 as well |
| FP8 TC GEMV L2 prefetch (`fp8tc_tile`) | object define `PLOW_NV_FP8TC_PF`, default on for sm_121a only | sm_121a object: on |

The cold-start hold no longer stalls a lone request on its own ingress guard: a request leaves the
counted "rendering" set before its job reaches the channel, when it leaves without submitting, and
before a media encode, and the hold wakes on that (`cold_burst_wait`, unit tests in `serve::mux`).

Still needs a non-GB10 GPU run:
- FlashMerge (`d_flash_merge`) runs two items per block when the head dim is below the block width
  (hd128 on 256 threads). The result is bitwise equal, but the merge pass changes on every arch:
  measure decode at long context with split KV on sm_89 (L4/L40S) and sm_90a (H100).
- FP8 TC GEMV L2 prefetch: measure `PLOW_NV_FP8TC_PF=1` on sm_90a (Gemma FP8 decode) and sm_120a, before it
  defaults on there.
- Each gated serving policy needs its own A/B on a non-GB10 GPU before it becomes a default there.
