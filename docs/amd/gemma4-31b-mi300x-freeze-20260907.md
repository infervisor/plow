# Gemma MI300X freeze — 2026-09-07

Branch: `perf/gemma31-mi300x-consolidated`.
Status: **FROZEN at the user's request. Experiments stopped.**
No new integration branch was created. Resume only on explicit user instruction.

## Applied and qualified

- `39cb856e`: BF16 gfx942 MM1/MM4 direct GEMV dispatch and fused-norm packed-prefill compatibility, with loader capability checks and runtime mixed-program expansion. Runtime fusion remains opt-in; PF_GFUSE remains off by default.
- Verification: 496 combined CPU/CUDA/HSA library tests, eight API tests, exact real-asset mixed instruction/dependency parity, GPU numerical/lifecycle/HTTP gates and default Nix build pass.
- Latest fetched main `49f8c089` is included. PR22 is open; this freeze does not merge it.
- Fresh stock vLLM0.28 ROCm comparison completed72 runs,288 requests and36,864 output tokens with zero failures. vLLM still leads throughput/TTFT/TPOT in all six cells;101/120 measured paired completion texts match. This is not a completed performance goal. See [scorecard](gemma4-31b-mi300x-direct-bench-serve-20260907.json).

## Stopped and pending

| Experiment | State at freeze | Required before promotion |
|---|---|---|
| GFUSE off/on stock serving | Interrupted by user; fixed inputs and completed/partial outputs retained. Both servers and benchmark clients stopped. | Complete a new controlled crossover; do not present partial results as a complete comparison. |
| MM1 direct fused GLU/QKV dispatch |107 full-logit rows bitwise; additional model improvement1.45–2.72%. Patch saved, not applied. | Serving crossover and regression assessment. |
| MM4 direct FlashDecode dispatch |107 rows bitwise; model improvement2.0–2.8%, but projection primitives regress3–9%. Patch saved, not applied. | Serving comparison and resource tradeoff decision. Noinline alternative is rejected. |
| Native FP8 PTPC W8A8 | Private implementation fails full-model gates. Valid observers reproduce all eight production logit rows bitwise. Native fused norm-quant rounding mismatch identified; candidate correction is primitive-only. | Compare captured fused outputs, implement profile-specific correction, pass unchanged full-model gates, then qualify B4/mixed/serving. |
| Streaming host capture | Prior correctness/HTTP gates pass; serving reduces delivery gaps but regresses several throughput cells. Patch saved, not applied. | Controlled comparison with the qualified kernel configuration; investigate callback/HTTP cost. |
| Broader optimization studies | Expanded GFUSE capacities64/256/2048 and B8, attention grouping, combined tuning profiles, staggered arrivals and saturation studies remain pending. | Explicit resumption, defined matched workloads and numerical/performance gates. |

Saved patches and FP8 handoff: [pending artifacts](gemma4-31b-mi300x-pending-20260907/manifest.json).
They are review material, not active production changes. Their base revisions differ;
inspect context before applying them on a later integration branch. Large model,
ELF, raw logit and profiler artifacts remain under `/app/plow/build-gemma31/`,
with paths and hashes retained in the reports. Existing artifacts were not deleted.

## Background cleanup

The fresh Plow/vLLM backend study and both direct-GEMV crossovers completed and
released their leases. The remaining GFUSE orchestrator and its benchmark/server
descendants were interrupted at the user's request. Root verified their PIDs were
gone, no GFUSE lease wrapper remained, and ports18940–18949 had no listeners.
FP8 and kernel agents report no owned GPU processes or leases. Unrelated machine
workloads were left untouched.

The [GFUSE stop record](gemma4-31b-mi300x-pending-20260907/gfuse-interrupted.json) preserves44/72 completed runs,176 requests and22,528 output tokens, plus two interrupted run directories without completed accounting. No complete-crossover performance conclusion is drawn. GitHub branch checks are separate from stopped GPU experiments.
