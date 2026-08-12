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

## Decode attribution and TP audit experiment

Raw interpreter tracing now writes only the completed program's live extent.
The previous widest-buffer readback retained prefill records after decode and
could report impossible workgroups above the MI325X limit.

The corrected one-token trace contains 2,459 decode packets and spans 67.576
ms inside an 80.022 ms wall step. Dense GEMV is the largest device component:
28.506 ms for 14.021 GB/rank/token, about 492 GB/s. The default TP safety audit
accounts for another 10.3--10.5 ms/token because it copies every cross-GPU
counter back through the copy engine.

`PLOW_TP_AUDIT_DIRECT=1` instead checks the same expected counter values through
host-mapped large-BAR memory after the device drain. It remains opt-in. A
64-token real-weight soak completed at 73.252 ms/token (13.7 tok/s), with every
token identical on all eight ranks. This is 8.2% faster than the 79.8 ms/token
audited control while retaining per-token gate validation.

The deliberately unaudited 64-token diagnostic floor is 69.091 ms/token. It is
not a serving recipe; it bounds the remaining host audit opportunity at 4.161
ms/token before any device-kernel change.

```bash
nix develop --command env \
  PLOW_L2_PLACE_DISPATCH=1 PLOW_TP_AUDIT_DIRECT=1 \
  perf-data/tools/gpulease -n 8 k3-mi325x-audit-direct \
  ./target/release/plowrt amd-bench \
  --blob /home/lava/models/k3_mi325x/model.pkt \
  --hsaco /home/lava/models/k3_mi325x/hsaco \
  --checkpoint /home/lava/models/k3_farm \
  --prompt 1008,10484,318,15383,387 --steps 64 --ctx 5 --tp 8
```

## Serve-path gate

The OpenAI-compatible serve path loaded the GPU backend and real K3 tokenizer,
then completed the four-prompt `bringup_gate.sh` battery under one TP8 lease.
All responses were coherent; the first prompt, `The capital of France is`,
returned exactly `Paris.`. This exercises tokenization, the complete graph,
carried KDA/MLA state, hidden-state flow, TP sampling, and detokenization rather
than only the `amd-bench` id path.

The optional flash object still refuses K3 capability and safely falls back to
the eight-wave interpreter. Current K3 FP8 MLA op 110 is excluded from class-4
routing, so merely adding the marker would remove the warning without changing
the measured K3 path.

## Adopted single-stream recipe

The gated asset now uses the 128-workgroup GEMV ownership packet described in
`perf-data/archive/k3/kimi-k3-mi325x-gemv-wg152.md` and FP8-KV decode objects containing the
compact exact-counter audit kernel. The composed 64-token median is 54.463
ms/token (18.4 tok/s), with all eight ranks token-identical. The OpenAI serve
gate also passed on this exact packet/object/runtime combination.

```bash
nix develop --command env \
  PLOW_L2_PLACE_DISPATCH=1 PLOW_TP_AUDIT_COMPACT=1 \
  perf-data/tools/gpulease -n 8 k3-mi325x-serve \
  ./target/release/plowrt serve \
  --assets /home/lava/models/k3_mi325x --port 8000
```

This AMD engine is single-sequence per rank. Concurrency above one measures
queueing rather than batched kernel capacity; it is not yet a Stage-7
concurrency result.

## Remaining comparison gate

No same-box Kimi-K3 vLLM or SGLang environment is installed. A performance-win
claim requires interleaved runs on the same eight MI325X GPUs, identical prompts,
token counts, context/concurrency ladders, and prefix-cache policy.
