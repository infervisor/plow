# H100 AOT compiler experiments

These experiments preserve native packet execution and are disabled by default. They do not establish an all-model/all-context win over vLLM. GPU runs are serialized. Kernel microbenchmarks, direct decode diagnostics, and serving results are separate gates.

## Compiler candidates

| Experiment | Compile control | Implemented change | Current evidence |
|---|---|---|---|
| Fuse Qwen GDN a/b | `PLOW_QWEN_FUSE_AB=1`, `PLOW_QWEN_AB_BLOCKS=132` or `12` | Existing `GemvQkv(Nv=0)` writes separate BF16 outputs; 1077→1029 decode instructions | Both block counts preserve five full-model logit rows and two resets exactly |
| Projection dependencies | `PLOW_QWEN_PROJECTION_DAG=1` | Independent BF16 branches with explicit recurrence/gating and Q/K/V joins | Alone and combined with a/b fusion: five exact rows and two exact resets |
| Share activation quantization | `PLOW_QWEN_SHARE_QUANT=1` | Serialized W8A8 consumers reuse immutable quantized inputs; 496→256 quantization packets | Five full-model rows and two resets remain byte-exact against the native XCACHE control |
| Prefill norm fusion | Existing `PLOW_PF_GFUSE=1`, new cubin flag `PLOW_NV_NRN_WPR=1` | Fused norm/residual/norm retains the unfused warp reduction order and BF16 materializations | 40 exact GPU/alias cells; Gemma12/31 prefill128 plus two decode rows match controls exactly |
| Prefill object selection | `PLOW_PF_SEG_GEMM_SMALL=<cubin>` | Compatible BF16 M≤128 segments can use a distinct native m128n128 TMA kernel instead of WS384 m128n256 | Gemma12 prefill128 plus two decode rows match WS384 exactly with graph0 and graph1; 192 of 483 M128 segments selected |

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

The combined compiler candidate reduces this direct-step mean by3.80%. It also preserves all four logit rows from native128-token prefill plus three decode steps byte for byte. The contemporary vLLM serving reference is19.99ms TPOT at input/output128/C1; Plow remains behind. Synchronization mode3 preserves five logit rows and two resets but shows no direct-step gain here.

The subsequent matched `vllm bench serve` pair uses the same frozen `plowrt-aot-candidates1`, XREG+KPANEL cubin, checkpoint, native TMA prefill, input/output128, C1, 32 measured requests, 16 warmups, seed42 and detailed latency metrics. Only the compiled packet asset changes. Both complete32/32 with4096 output tokens and zero failures:

| Metric | AOT control | DAG + a/b fusion, 12 blocks |
|---|---:|---:|
| Mean TTFT ms | 87.150 | 87.207 |
| Mean TPOT ms | 31.333 | 30.143 |
| p99 ITL ms | 31.436 | 30.255 |
| Mean E2E ms | 4066.38 | 3915.43 |
| Output tok/s | 31.476 | 32.689 |

TPOT improves3.79% and output throughput3.86% in this pair; mean TTFT increases0.057ms. This is an AOT decode gain, not an all-metrics win. Repetition, B4/B16 serving and longer histories remain open. Raw: `qwen-aot-{control,dag-ab12}-128-c1-r1/in128_c1.json` under the campaign directory.

Raw evidence: `/tmp/plow-model-support-checks/qwen-aot-*-step.log`, `qwen-aot-bf16-quality.json`, `qwen-aot-prefill-fp8-quality.json`, and `/tmp/plow-qwen-kpanel/ptxsync3-quality.json`. Assets are under `/opt/dlami/nvme/tmp/plow-h100-campaign/qwen27b-aot-*`; each carries the exact compiler invocation, flags and binary hash in `experiment.json`. Host binaries are frozen as `plowc-aot-candidates1`, `plowrt-aot-candidates1`, `decode-dump-aot-candidates1` and `step-bench-aot-candidates1`.

## What Lean and egg currently contribute

The built `lean-plow/.lake/build/bin/plow_verify` accepted the real Qwen baseline decode ordering with106439 GQ entries and rejected a deliberately introduced backward dependency. New BF16 candidate assets, the shared-quantization asset, and paired Gemma12/31 fusion assets were emitted with active Lean ordering checks. Their manifests record `lean.verified=true` for all programs.

This certificate checks encoded queue/counter ordering and selected staged-LDS requirements. The original operand dataflow and arena aliases are absent from the emitted ordering request; the compiler tests and GPU checks cover additional obligations separately. The scalable checker remains in the trusted base, and the lower-bound query is not a certified latency prediction. These checks do not prove CUDA floating-point equivalence.

Generic egglog extraction does not currently lower GPU packets. A bounded Kimi decision bridge exists, while these Qwen/Gemma candidates use the existing native emitter. Abstract rewrite proofs lack dtype/rounding semantics. Therefore the experiments preserve explicit BF16 outputs rather than assuming algebraic fusion is numerically valid.

Checker evidence: `/tmp/plow-model-support-checks/lean-qwen-baseline/`. Normalization artifacts and exact existing compiler recipes: `/tmp/plow-gemma-pf-gfuse/`. Alternate object, SASS and build commands: `/tmp/plow-model-support-checks/gemma-prefill-small-gemm/`.

The norm-fusion model pair uses the same WPR1 TMA PFseg object in both arms, retaining stock GEMM/attention/decoder objects. Only the compiled fusion flag changes. Full-row comparisons are in `/tmp/plow-gemma-pf-gfuse/full-prefill128-quality.json`. Object-selection comparisons are in `/tmp/plow-model-support-checks/gemma-smallgemm-quality.json`; load logs confirm zero alternate selections in larger prefill buckets. These narrow checks do not qualify longer contexts or all batching configurations. The three-arm Gemma12 serving experiment completed with the same host/PFseg and matched input/output128/C1 controls; results follow.

## Follow-through measurements

The Gemma12 three-arm serving run uses frozen host `plowrt-aot-candidates1`, identical WPR1 PFseg/stock decoder, BF16 B1, context16384, input/output128, C1, 32 requests, 16 warmups and seed42. All arms completed32/32, zero failures,4096 input/output tokens each. The small-object arm is ABI1 m128n128, not the later m64n64 candidate.

| Arm | TTFT ms | TPOT ms | p99 ITL ms | E2E ms | Output tok/s |
|---|---:|---:|---:|---:|---:|
| Control | 42.090 | 12.084 | 12.242 | 1576.781 | 81.167 |
| Small GEMM ABI1 | 41.941 | 12.110 | 12.235 | 1579.898 | 81.007 |
| Norm fusion | 42.103 | 12.082 | 12.271 | 1576.459 | 81.183 |

Neither experiment establishes a material serving gain; both remain opt-in. Detailed raw results: `gemma12b-aot-{control,small,gfuse}-128-c1-r1/in128_c1.json` under the campaign directory. Runtime logs report prefix caching disabled. Reproduction of the installed benchmark's seed42 random dataset produces identical128-token IDs with both tokenizers for all32 prompts; combined ID SHA256 is `f071429f15f1a2605139b121f53a31c3dfa715b9c1eea5156ab347ddca079599`. This audit does not establish a general tokenizer guarantee.

Native W8A8 shared activation quantization reduces direct-step mean **37.572→34.978ms (6.90%)** with the same XCACHE cubin, frozen host,128-token history,16 warmups and128 measured steps. This asset has no native prefill: its prompt is consumed through decode. It remains slower than the BF16 DAG+a/b candidate, and these are diagnostic timings rather than serving results. Raw: `qwen-w8a8-{control,sharequant}-step.log` under the checks directory.

## Additional MLP packet fusion

`PLOW_QWEN_FUSE_MLP=1` uses the existing native `GemvQkv(Nv=0)` operation for gate/up projections while preserving distinct BF16 outputs and the subsequent SiLU/Glu operation. It removes64 decode packets:1077→1013 alone,1029→965 with DAG+a/b fusion. FP8 and cuBLASLt decode reject this combination; the default path and prefill remain unchanged.

Fourteen Qwen compiler tests cover B1/B4 packet outputs, geometry and operand dependencies. The emitted B1 asset passed active Lean ordering checks. Five full-model rows and two resets match the native control exactly; native128-token prefill plus three decode steps also produces four byte-exact rows. Evidence: `/tmp/plow-model-support-checks/qwen-aot-mlp-quality.json`.

The matched direct-step pair regresses **30.101→30.899ms (+2.65%)**, despite fewer packets. Both use identical XREG+KPANEL cubin and frozen host,128-token native prefill,16 warmups and128 measured steps. No serving promotion follows this result. Raw: `qwen-aot-mlp-{dag-ab12,dag-ab12-mlp}-step.log`. The dispatch audit found that generic GemvQkv bypasses the existing XREG path; the follow-up below also regresses.

A default-off MLP XREG dispatch follow-up preserves both BF16 output tensors but routes the fused gate/up packet through two existing XREG projection calls. Nine logit rows and two resets remain exact. Direct-step mean nevertheless regresses to **32.090ms**, versus30.899ms for generic MLP fusion and30.101ms for DAG+a/b without MLP fusion. This is a negative result; the path remains opt-in. Evidence: `/tmp/plow-model-support-checks/qwen-aot-mlp-xreg-quality.json` and `qwen-aot-mlp-xreg-step.log`.

The B4 control and DAG+a/b assets also passed the existing batch lifecycle check:20 full-logit rows each are exactly equal to the B1 control across slots0/3 with simultaneous and staggered requests. Idle recurrent state remains exact, and resetting slot3 preserves slot0. These are five-token decode checks; native B4 prefill, long histories and serving remain separate gates. Both assets passed active Lean ordering checks. Evidence: `/tmp/plow-model-support-checks/qwen-aot-b4-quality.json`.

## M64N64 prefill tile

The default-off ABI2 role object computes64x64 BF16 tiles using existing full128-row TMA maps, full ascending-K WGMMA and the native packet queue. SASS has128 entry registers and no stack/spills; the arena is99376 bytes. Nine M128 primitive cases match cuBLASLt exactly. Nine M65 cases pass numerical/canary gates; Gemma31 down has relative L2 .000177496, while the other eight are exact. Physical2GiB eviction is outside the timed CUDA events, and rotating weights exceed L2.

Gemma12 o-local/o-full/down primitive means improve50.2→36.9,89.7→76.9,158.2→151.6µs. Gemma31 and Qwen regress:168/160 output tiles exceed132 packet slices and force some slices through two serial full-K tiles. The original128-row TMA maps also overfetch for64-row computation. This makes selection model-specific; ABI2 is restricted to Gemma12. The pending M64N128 experiment targets one tile per active slice on Gemma31.

Full Gemma12 native128-token prefill plus two decode steps matches the WS384 control exactly with graph0 and graph1. Despite primitive gains, the subsequent matched serving pair shows **no improvement**:

| Metric | WS384 control | M64N64 candidate |
|---|---:|---:|
| Mean TTFT ms | 42.308 | 42.543 |
| Mean TPOT ms | 12.101 | 12.122 |
| p99 ITL ms | 12.295 | 12.314 |
| Mean E2E ms | 1579.157 | 1582.012 |
| Output tok/s | 81.043 | 80.896 |

Both arms use frozen `plowrt-aot-m64-v1`, identical GFUSE0 asset/stock PFseg/decoder, context16384 B1, input/output128 C1,32 requests,16 warmups and seed42. Both complete32/32 with4096 input and output tokens, zero failures. The candidate remains opt-in. Raw: `gemma12b-m64-{control,m64n64}-128-c1-r1/in128_c1.json`; quality: `/tmp/plow-model-support-checks/gemma-m64gemm-quality.json`; primitive audit: `gemma-prefill-m64n64/timing-audit.json` under the checks directory.

## Native FP8 M1 TMA staging

The default-off `PLOW_NV_FP8_M1_TMA=1` arm replaces synchronous weight staging with two BK512 TMA buffers inside the existing packet interpreter. It retains the XCACHE activation panels, ascending-K `m64n8k32` FP8 WGMMA accumulation and the exact nested FP32 scale epilogue. It requires FAST_ACCUM; no cuBLASLt callback or host dispatch is introduced. `PLOW_QWEN_FP8_M1_TMA=1` emits64-row E4M3 weight-map recipes in the existing packet field; the existing loader encodes them once. `PLOW_BUILD_FP8_M1_TMA=1` selects the kernel build. Unmapped packets retain the XCACHE fallback.

Fresh XCACHE and TMA primitive runs each pass15/15 bitexact comparisons against installed vLLM CUTLASS, including K528/640 tails and shapes requiring repeated output-row tiles. The existing comparator uses132 blocks of256 threads, preallocated outputs/maps, five warmups and30 CUDA-event samples, with700MiB eviction outside each timed sample. Values below are median cold-weight kernel latency, not serving latency:

| Shape N×K | Fresh XCACHE µs | TMA µs | CUTLASS µs, TMA run |
|---|---:|---:|---:|
| a/b48×5120 | 20.720 | 12.064 | 8.928 |
| qkv10240×5120 | 48.160 | 34.896 | 30.448 |
| gate/up17408×5120 | 70.352 | 50.880 | 43.296 |
| down5120×17408 | 71.936 | 49.952 | 44.640 |
| Synthetic FP8 head248320×5120 | 613.104 | 423.376 | 410.016 |

The synthetic head is a primitive stress case; the real Qwen output head remains BF16. These measurements establish a staging improvement for the tested primitives, not full-model or serving performance.

CPU checks passed15 Qwen compiler tests,28 barrier-phase/row-reuse simulations and64 alignment offsets. The arena reserves205840 bytes:65536 weight-ring bytes,139264 activation-cache bytes,16 barrier bytes and alignment slack. With2448 static shared bytes, the total208288 fits the232448-byte limit. Barrier initialization is published to the async proxy, consumption waits for TMA completion, buffers are reused only after WGMMA retirement and CTA synchronization, and barriers are invalidated before arena reuse. These checks complement the GPU cases; they are not a formal CUDA memory-model proof.

The matched full decoder pair is built from one frozen source snapshot with identical BF16 XREG/KPANEL, TMA-helper, FP8 M1/FAST_ACCUM/BK1024/XCACHE and vLLM-quantization flags; only the new TMA flag differs. Both use255 registers and2448 static shared bytes, with reported stack208 bytes for control vs232 for candidate and LOCAL0. The standalone TMA primitive uses160 registers and no stack/spills. The original older XCACHE decoder has different flags/resources and is not the matched runtime control.

The mapped shared-quantization asset `qwen27b-aot-sharequant-tma` was emitted with active Lean checks. Both matched full-model decoder arms pass five logit rows and two resets byte-exact against the prior XCACHE reference. Evidence: `full-quality.json` in the directory below. Matched direct-step timing with128-token history,16 warmups and128 measured steps improves **36.572→29.918ms (18.20%)**. Both arms use the same mapped shared-quantization asset and frozen host; the prompt is consumed through decode because native W8A8 prefill is absent. This is a decode diagnostic gain, not an all-metrics serving win. Raw: `step-{control,final}.log` in the evidence directory; no serving promotion is claimed. Compiler binary: campaign `plowc-aot-fp8-tma1`. Evidence directory: `/tmp/plow-model-support-checks/qwen-fp8-m1-tma/`, including `fresh-xcache-result.json`, `result.json`, `provenance.json`, frozen sources, SASS, `final-primitive.so`, `control-interp_sm90a.cubin` and `final-interp_sm90a.cubin`. Existing comparator invocation adds `--fp8-interpreter-only --fp8-tma` for the candidate; descriptor construction is outside the timed loop.

The initial M64N128 ABI3 candidate limits Gemma31 isolated o/down to84 output tiles across132 packet slices. Its nine M128 primitives matched cuBLASLt exactly; Gemma31 o-local/o-full/down measured74.8/146.4/195.2µs. This initial result is superseded for qualification: review found that the new M64 role needed an explicit async-proxy fence after barrier initialization. The fence is added to both M64 variants; corrected primitive and ABI3 CUDA builds pass with128 registers and no stack/spills, but final cubins require new GPU checks. Prior M64 timings and full-model results above describe the pre-fence binaries only. ABI3 remains disabled by default, with full-model/serving validation pending. Initial artifacts: `/tmp/plow-model-support-checks/gemma-prefill-m64n128/`; corrected builds: `gemma-prefill-m64n128-fence/` under the checks directory.
