# Frozen private FP8 work — 2026-09-07

Status: **NOT QUALIFIED; do not promote or benchmark against vLLM.**
User requested freezing for later integration on another branch. No GPU jobs,
servers or leases remain owned by this agent. Production branch was untouched.

Private checkout: `/app/plow/build-gemma31/fp8-native-private`.
Base: `3c85c2720ef73d2139d857edcea94213d3a32ea1`.
Patch: `probe/private-native-ptpc.patch` (15 compiler/runtime/test files).
Exact source/artifact hashes: `probe/frozen-provenance.json` and
`probe/private-native-ptpc.sha256`. Do not rerun the old one-off integration
editing scripts. Apply/review the frozen patch against its stated base.

## Implemented privately

- Named PTPC W8A8 profile, four shared quant producers/layer, BF16 unscaled
  decode shadows, FP32 activation/weight scales in GEMV epilogue; BF16 head/KV.
- Actual emitter, full 60-layer Gemma4 assets, AmdEngine execution and strict
  teacher-forced full-logit comparison. PF shadows are NONE; B1-sized decode
  shadows cannot receive prefill rows. B1 restriction is temporary qualification
  scope; B4 remains an unresolved goal requirement.
- Profile/export/operand/extent/object-capability validation; unsupported
  backends and incompatible fusions fail closed; legacy W8A16 unchanged.
- Tagged native GELU BF16 activation rounding, bitwise against native custom
  op on saved inputs, but insufficient to fix full-model accuracy.

Verification: 12 legacy emitter bytegoldens, named emitter test, six runtime
contract tests pass; Nix ROCm 7.14 PF/decode FP8 objects build. Earlier minimal
two-layer B1/B2 tests pass; B4 chain still fails. Full-model original and
GELU-corrected runs both FAIL all eight unchanged L2<=1% / max-absolute<=0.5
gates, despite identical greedy tokens. Raw failures remain intact:
`plow-native-logits/comparison.json`, `plow-native-logits-gelu/comparison.json`.

## Capture validity: keep these separate

- `vllm-native-logits`: unmodified production compiled/graph vLLM 0.28 PTPC,
  native `+quant_fp8,+gelu_and_mul`, BF16 head/KV, original p16→p128 history.
  `production-repeat-control.json`: independent process repeats all eight
  full-vocabulary logit rows BITWISE.
- `vllm-layer0`, `vllm-compiled-layer0`, `vllm-quant-layer0`: **INVALID** as
  production references. Eager/module hooks/replacement quant op perturb
  compiler behavior. `quant-capture-control.json` explicitly rejects the
  replacement-op capture even though nominal settings match.
- `vllm-dispatch-layer0` and `vllm-fused-dispatch-layer0`: **VALID** observations
  of the original native ops after compilation. Corresponding
  `dispatch-capture-control.json` and `fused-dispatch-capture-control.json`
  show all eight full logit rows BITWISE identical to production, with matching
  configuration. Observer: `dispatch_capture/sitecustomize.py`, named worker
  RPC; no graph/operator/schema replacement or insecure serialization.

## Identified pending fix: native fused norm + quant

The valid observer detects 120 standalone quant calls and 120 fused norm-quant
calls per 60-layer prefill. The production baseline uses
`_C::rms_norm_dynamic_per_token_quant` before QKV and gate/up, with ordinary
per-token quant only before O/down. Captured original fused inputs, weights,
output FP8 bytes and FP32 scales are in `vllm-fused-dispatch-layer0`.

Native fused C++ retains a BF16 normalized-value rounding before multiplying
gamma, unlike native *standalone compiled* RMSNorm, where Inductor removes
that cast. Never infer fused arithmetic from the standalone probe.

`norm-round-fused-results.json` is the actual native fused-op same-input
primitive comparison on real embeddings from the matched p128 prompt:

| Plow norm arithmetic | B1 FP8-byte / scale differences | PF128 differences |
|---|---:|---:|
| Current private runtime (one final BF16 rounding) | 93 / 0 | 20,617 / 43 |
| Primitive candidate: BF16 normalized value before gamma | 0 / 0 | 0 / 0 |

This candidate uses existing `d_rmsnorm` without gamma, followed by a separate
BF16 gamma multiply, then actual native quant. It is a **primitive math proof**,
not an integrated or performance-qualified kernel. **No runtime norm correction
was implemented before freeze.** Direct comparison against the newly captured
production fused-output tensors remains pending; the captures already passed
their full-logit validity gate.

Official source snapshots: `vllm-fused-norm-quant-v0.28.0.cu` and
`vllm-layernorm-utils-v0.28.0.cuh`, from vLLM tag v0.28.0 under
`csrc/libtorch_stable/quantization/fused_kernels/`. FP8 normalization conversion
keeps BF16 rounding; FP8 quant division intentionally avoids reciprocal scale.

## Pending experiments, not launched

1. Directly compare the candidate against the valid captured fused bytes/scales
   for layer-0 QKV and gate/up, including captured input identity.
2. Implement only the matching norm→quant output paths in the named private
   profile; preserve standalone BF16 norm semantics, legacy profiles and
   phase-safe shadow extents. Add fail-closed operand/capability checks.
3. Rerun the actual full-model original eight-row gate without changing reference,
   settings or thresholds. Then localize subsequent mismatches using the valid
   observer. No assumption of monotonic error improvement.
4. Resolve B4 and broader batch/sequence/mixed-serving qualification, then make
   a genuinely matched FP8 serving comparison. No FP8 performance claim yet.

Parent's independent shipping BF16 control also fails seven of eight strict
native-logit gates and is bitwise repeatable across native engines. It is
context, not permission to subtract error norms or relax the FP8 gate.

## Execution notes

All commands used Nix ROCm 7.14 and `gpulease -n 1`. Current reliable terminal
form is `nix develop --command bash -c 'bash probe/<runner>.sh'` with tool
`login=false`. Long inline escaped shell commands intermittently hit a Nix
container mount-namespace error; an approval call then stalled and was aborted.
No such approval remains pending. Per-run GPU UUIDs were not persisted for
these private probes; do not infer UUIDs from later lease ownership.

No commands in this handoff should be run until the user resumes experiments.
