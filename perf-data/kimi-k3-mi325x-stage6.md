# Kimi-K3 MI325X Stage 6: full-model floor

Date: 2026-08-10. Hardware: 8 leased MI325X GPUs (gfx942, 304 CUs each).
Toolchain: flake-pinned ROCm 7.14.0, HIP 7.14.60850, clang 23. The build and
both device runs used `nix develop`; timings held the repository `gpulease`.

## Artifact

The full 93-layer TP8 artifact uses the real `/home/lava/models/k3_farm`
checkpoint, native MXFP4 weights, FP8 KV, and the gfx942 global-queue
interpreters in `/home/lava/models/k3_mi325x/hsaco`.

K3 prefill programs are L2-placed. The K3 A4W4 prefill build rows now carry
`PLOW_L2_PLACE_DISPATCH=1`, matching the packet contract without enabling the
separate, unmeasured prefill hierarchy experiment.

## Correctness floor

Prompt ids: `1008,10484,318,15383,387`. Context: 5. Decode: 8 greedy steps.

- Prefill: 218.4 ms; all eight ranks selected token 17374.
- Decode: 79.690 ms/token; every token was identical on all eight ranks.
- Token ids: `17374,13,646,10484,318,16458,387,28202,13`.
- Decoded text: ` Paris. The capital of Germany is Berlin.`

The command exited zero. This is the required full-model correctness and
latency floor; it is not yet a vLLM comparison.

## Load profile

Each rank uploads 191.23 GiB, including 168.39 GiB of packed routed experts and
276 recurrent-state tensors (56 MiB per slot). Cold load was 186.3 s. A second
run with the checkpoint in the host page cache loaded in 59.1 s.

The optional four-wave flash object currently refuses the K3 `AttnRes` arm and
the runtime safely falls back to the eight-wave prefill interpreter. Correctness
is preserved; a K3-specific flash object remains a measured prefill-performance
experiment rather than a Stage-6 blocker.

## Remaining comparison gate

No same-box Kimi-K3 vLLM or SGLang environment is installed. A performance-win
claim requires interleaved runs on the same eight MI325X GPUs, identical prompts,
token counts, context/concurrency ladders, and prefix-cache policy.
