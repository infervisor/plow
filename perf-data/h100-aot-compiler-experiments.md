# H100 AOT compiler experiments

These experiments preserve native packet execution and are disabled by default. They do not establish an all-model/all-context win over vLLM. GPU runs are serialized. Kernel microbenchmarks, direct decode diagnostics, and serving results are separate gates.

## Compiler candidates

| Experiment | Compile control | Implemented change | Current evidence |
|---|---|---|---|
| Fuse Qwen GDN a/b | `PLOW_QWEN_FUSE_AB=1`, `PLOW_QWEN_AB_BLOCKS=132` or `12` | Existing `GemvQkv(Nv=0)` writes separate BF16 outputs; 1077→1029 decode instructions | Both block counts preserve five full-model logit rows and two resets exactly |
| Projection dependencies | `PLOW_QWEN_PROJECTION_DAG=1` | Independent BF16 branches with explicit recurrence/gating and Q/K/V joins | Alone and combined with a/b fusion: five exact rows and two exact resets |
| Share activation quantization | `PLOW_QWEN_SHARE_QUANT=1` | Serialized W8A8 consumers reuse immutable quantized inputs; 496→256 quantization packets | Five full-model rows and two resets remain byte-exact against the native XCACHE control |
| Prefill norm fusion | Existing `PLOW_PF_GFUSE=1`, new cubin flag `PLOW_NV_NRN_WPR=1` | Fused norm/residual/norm retains the unfused warp reduction order and BF16 materializations | 40 GPU cells pass exact output/residual and in-place alias checks; model qualification pending |
| Prefill object selection | `PLOW_PF_SEG_GEMM_SMALL=<cubin>` | Compatible BF16 M≤128 segments can use a distinct native m128n128 TMA kernel instead of WS384 m128n256 | ABI/queue/slice tests pass; 192 of 483 Gemma12 M128 segments eligible; GPU qualification pending |

The Qwen changes are restricted to their supported native BF16 or W8A8 decode paths. Prefill remains unchanged by those three flags. Thirteen Qwen compiler tests cover preserved outputs/geometry, operand and alias dependencies for B1/B4, and full-model quantization reuse. The runtime selection checks reject unsupported precision, unmapped operands, fine-grained tile dependencies, and incomplete or duplicate packet slices. Graph and direct launches use the same selection computed at load time. The alternate object retains small spills and is not assumed faster.

## Direct decode timing

The existing `step_bench` runs the same release host, Qwen BF16 checkpoint, native TMA prefill, XREG+KPANEL decoder, B1, context128, 16 warmups and 128 measured steps. Only the compiled packet variant changes, except the explicitly labelled synchronization control. Each timing is one diagnostic run, not `vllm bench serve`.

| Packet/kernel variant | Mean step ms | Median step ms |
|---|---:|---:|
| Fresh AOT control | 31.289 | 31.285 |
| a/b fusion, 132 blocks | 30.875 | 30.877 |
| a/b fusion, 12 blocks | 30.890 | 30.889 |
| Projection DAG | 30.816 | 30.816 |
| DAG + a/b fusion, 12 blocks | 30.101 | 30.105 |
| Existing PTX synchronization mode3, stock packets | 31.313 | 31.315 |

The combined compiler candidate reduces this direct-step mean by3.80%. The combined candidate also preserves all four logit rows from native128-token prefill plus three decode steps byte for byte. A matched serving pair is running; broader numerical/context coverage remains necessary. The contemporary vLLM serving reference is19.99ms TPOT at input/output128/C1; Plow remains behind. Synchronization mode3 preserves five logit rows and two resets but shows no direct-step gain here.

Raw evidence: `/tmp/plow-model-support-checks/qwen-aot-*-step.log`, `qwen-aot-bf16-quality.json`, `qwen-aot-prefill-fp8-quality.json`, and `/tmp/plow-qwen-kpanel/ptxsync3-quality.json`. Assets are under `/opt/dlami/nvme/tmp/plow-h100-campaign/qwen27b-aot-*`; each carries the exact compiler invocation, flags and binary hash in `experiment.json`. Host binaries are frozen as `plowc-aot-candidates1`, `plowrt-aot-candidates1`, `decode-dump-aot-candidates1` and `step-bench-aot-candidates1`.

## What Lean and egg currently contribute

The built `lean-plow/.lake/build/bin/plow_verify` accepted the real Qwen baseline decode ordering with106439 GQ entries and rejected a deliberately introduced backward dependency. New BF16 candidate assets, the shared-quantization asset, and paired Gemma12/31 fusion assets were emitted with active Lean ordering checks. Their manifests record `lean.verified=true` for all programs.

This certificate checks encoded queue/counter ordering and selected staged-LDS requirements. The original operand dataflow and arena aliases are absent from the emitted ordering request; the compiler tests and GPU checks cover additional obligations separately. The scalable checker remains in the trusted base, and the lower-bound query is not a certified latency prediction. These checks do not prove CUDA floating-point equivalence.

Generic egglog extraction does not currently lower GPU packets. A bounded Kimi decision bridge exists, while these Qwen/Gemma candidates use the existing native emitter. Abstract rewrite proofs lack dtype/rounding semantics. Therefore the experiments preserve explicit BF16 outputs rather than assuming algebraic fusion is numerically valid.

Checker evidence: `/tmp/plow-model-support-checks/lean-qwen-baseline/`. Normalization artifacts and exact existing compiler recipes: `/tmp/plow-gemma-pf-gfuse/`. Alternate object, SASS and build commands: `/tmp/plow-model-support-checks/gemma-prefill-small-gemm/`.
