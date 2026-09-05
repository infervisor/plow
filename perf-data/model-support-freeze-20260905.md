# model-support freeze — 2026-09-05

This freezes the consolidated model-support branch for main review. It does not
qualify every model, precision, context or concurrency, or claim a vLLM win.

## Consolidation

- The optional FP8 prefill GEMM role is selected by compiled
  `segment_roles.json` metadata. The runtime validates the declared ABI, queue
  ownership and cooperative residency. No model-specific runtime switch was
  added. Existing legacy runtime configuration remains outside this change.
- Compiler experiment switches use `EmitConfig`, including the existing Qwen
  switches that previously bypassed its configuration audit. Defaults are
  preserved. The new isolated role experiment remains off by default.
- Local nested Gemma checkpoint metadata retains architecture identification
  when the inner text configuration omits it. Missing and unexpected checkpoint
  tensors still fail compilation.
- Attention metadata lowering uses the sequence axis for `[1,S,H,D]`.
  Unsupported batch sizes fail explicitly. This limitation concerns the graph
  metadata lowering path; native devgen packet batching is separate.
- Broadcast scalar transfers respect the operand allocation. AMD NoPE packets
  cannot enter the RoPE-only V2 prefill kernel.

## Performance evidence and open gates

The [serving sweep report](h100-realworld-serving-sweep.md) records the completed
Gemma12 vLLM 0.28 reference grid and separates unmeasured kernel candidates from
measured results. The remaining reference sweep runs independently from this
source freeze. Matched Plow sweeps and general numerical parity remain open.

The isolated FP8 role has no GPU correctness or latency qualification. Its lower
register/spill footprint increases native launches from 97 to 929 per full-model
prefill chunk; it must beat an identically segmented control and the original
packet before promotion. Gemma chunk-capacity and attention candidates also
remain unqualified. Declared memory totals do not establish successful startup.

The full Gemma31 262K packet exceeds this H100's memory even with the smaller
prepared chunk setting. This freeze does not remove that capacity limit.

## Validation

Final validation results and the pinned commit are recorded with the freeze
checkout in `/tmp/plow-model-support-checks/freeze/`. Checks use `nix develop`.
GPU execution is not part of the freeze checks while the reference server owns
the H100. Compilation and CPU ordering tests do not establish GPU numerical parity.
