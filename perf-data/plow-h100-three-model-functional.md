# H100 functional bringup — 2026-09-05

All three models serve in BF16 and Plow FP8 W8A16. These are functional smoke results, not performance wins or complete quality certification.

One H100 80GB, TP 1, context 8192. Existing `bench_plowrt_serve.sh` runs the vLLM 0.28.0 benchmark client against raw completions. Seed 42, concurrency 1, input 128. Gemma12B BF16 uses 32 measured / 16 warmup requests and 128 output tokens; other rows use 4 measured requests and 32 output tokens, except Gemma31B BF16 uses 128 output tokens. All passed the France/Paris coherence gate. Gemma coherence uses an explicit BOS; measured random requests are unchanged.

| Model | Precision | Successful requests | Input tokens | Output tokens | TTFT ms | TPOT ms |
|---|---|---:|---:|---:|---:|---:|
| Gemma4 12B | BF16 | 32 | 4096 | 4096 | 42.01 | 11.98 |
| Gemma4 12B | W8A16 | 4 | 512 | 128 | 80.38 | 9.25 |
| Gemma4 31B | BF16 | 4 | 512 | 512 | 56.35 | 28.12 |
| Gemma4 31B | W8A16 | 4 | 512 | 128 | 171.94 | 20.26 |
| Qwen3.8 27B | BF16 native prefill | 32 | 4096 | 4096 | 117.01 | 35.42 |
| Qwen3.8 27B | W8A16 | 4 | 512 | 128 | 3529.90 | 27.67 |

Plow W8A16 uses per-output-channel FP8 projection weights with BF16 activations, KV, embedding and output head. Qwen convolution and gate parameters remain BF16; recurrent state is FP32. This differs from the vLLM reference’s online per-tensor W8A8. Direct FP8 speed ratios would compare different arithmetic.

Qwen uses the checkpoint’s `qwen3_5` architecture: 48 gated-delta layers and 16 full-attention layers. Initial support is CUDA sm90a, TP 1, batch 1, with decode-only prompt consumption. Prefix reuse and multistep execution are disabled for recurrent assets. An opt-in BF16 native 128-token prefill path now supports exact chunks and decode remainders. The corrected 32-request / 16-warmup measurement is shown above (128 output tokens). It trails the matched vLLM BF16 reference: 110.70 ms TTFT and 19.98 ms TPOT. Earlier Qwen decode measurements preceded a KV-mask correction and are superseded; the initial native ~42 ms TTFT measurements are rejected because incomplete dispatch produced zero logits.

The Qwen2 tokenizer processor is matched to Transformers, including its Unicode splitting behavior. Both precision modes report exactly 512 input tokens across four 128-token requests after the fix. Earlier 494-token runs are superseded.

Verification: packet ABI, 108 native Qwen primitive GPU comparisons, API/tokenizer tests, quantizer tests, Hopper decode/prefill compilation, and CUDA/HSA/hub Rust checks passed. Full-model teacher-forced logit and request-reset diagnostics are retained in the campaign directory; numerical qualification remains in progress. Qwen corrected decode agrees with the checked vLLM histories at roughly 1% centered relative logit error. Native prefill plus continuation shows 0.95–1.31% against vLLM on four checked prefixes; one leading-token difference is a BF16 tie. A 130-token remainder path matches 128-token prefill plus decode bit-for-bit. Two 128-token chunks plus continuation match all four checked decode-only top tokens, with 1.37–1.81% centered relative logit error. These checks do not certify all prompts. The pre-existing three `plowc hf_dir_compile` workspace failures are recorded separately in the branch review.

vLLM reference matrices: [BF16 repeats](vllm-028-h100-three-model-baseline-repeats.md), [FP8](vllm-028-h100-fp8-baseline.md).

Raw evidence:

- `/opt/dlami/nvme/tmp/plow-h100-campaign/gemma12b-lt0-c1-r1/in128_c1.json`
- `/opt/dlami/nvme/tmp/plow-h100-campaign/gemma31b-stock-smoke/in128_c1.json`
- `/opt/dlami/nvme/tmp/plow-h100-campaign/gemma12b-w8a16-smoke-fixed/in128_c1.json`
- `/opt/dlami/nvme/tmp/plow-h100-campaign/gemma31b-w8a16-smoke-fixed/in128_c1.json`
- `/opt/dlami/nvme/tmp/plow-h100-campaign/qwen-w8a16-mask-fixed-smoke/in128_c1.json`

- `/opt/dlami/nvme/tmp/plow-h100-campaign/qwen-native-prefill-fixed-128-c1-r1/in128_c1.json`

## Qwen TMA prefill candidate

First matched seed42 run: 32 measured requests,16warmups,input128/output128,C1,BF16. TMA lowers TTFT from117.01ms to87.52ms; TPOT35.37ms. vLLM’s correspondingseed42 reference is110.70ms/19.98ms; seed43 reference94.95ms/19.98ms. A repeat and contemporaneous vLLM control remain pending; this is not an overall serving win. TMA128 logits match the corrected ordinary native128 path on the checked history. Long-context numerical qualification remains open.

Raw: `/opt/dlami/nvme/tmp/plow-h100-campaign/qwen-tma-128-c1-r1/in128_c1.json`.

TMA repeat (same seed 42/settings) completed 32/32: TTFT 87.28 ms, TPOT 35.36 ms, output 27.96 tok/s. Contemporary vLLM control completed 32/32: TTFT 94.12 ms, TPOT 19.99 ms, output 48.61 tok/s. Across two Plow repeats TTFT was 87.28–87.52 ms, p99 TTFT 87.84–87.98 ms; control p99 TTFT was 107.03 ms. This supports a roughly 7% TTFT improvement at 128/C1 only. Decode, output throughput, longer contexts and the overall objective remain unachieved. Raw repeat/control: `qwen-tma-128-c1-r2/in128_c1.json` and `vllm-qwen-128-control-r3/in128_c1.json` under the campaign directory.

## Experimental paths at this checkpoint

- Qwen native prefill with batch capacity 4 passed 42 exact comparisons against batch-1 native references at prefixes 128, 130 and 256, including inactive KV/state preservation and slot reset. Serving performance is pending.
- `PLOW_QWEN_DECODE_LT=1` routes body BF16 projections through cuBLASLt. Graph and non-graph outputs matched exactly on five checked prefixes. A four-request serving smoke measured TPOT 33.84 ms, versus 35.37 ms for the native decoder; a full repeated comparison is pending. `PLOW_QWEN_DECODE_LT_PRESEED=1` skips external NOP windows after validating counter ownership and ordering. Its five CPU tests passed; GPU correctness and timing remain pending.
- Packed-tensor FP8 weight export and opt-in `PLOW_NV_QUANT_FP8_VLLM=1` activation quantization prepare matching W8A8 arithmetic. Activation bytes and scale bits matched vLLM in all 45 checked GPU cases. Native full-model W8A8 batch-1 decode is now implemented behind the existing FP8/W8A8 emitter flags; numerical qualification and optimization remain in progress.
- Gemma B16 MM8/MM16 decoder variants produced identical logits on the checked three-token history; comparative serving timing is pending. The opt-in `PLOW_NV_GEMMA_HNR_BF16=1` norm/RoPE rounding change improves the checked Gemma12B prefix-6 relative logit error from 18.89% to 13.59%, but the leading token still differs from vLLM. It is not a complete quality fix, and segmented prefill needs a matching build before whole-serving evaluation.

These switches remain opt-in. Long-context Qwen and Gemma numerical qualification remains open. The existing baseline matrix does not cover all supported contexts or end-to-end latency percentiles.

The optimization direction is now restricted to kernels executed within Plow's packet interpreter. External cuBLASLt dispatch is parked as a comparison experiment; its isolated kernel measurements can guide native kernel work. The non-graph counter-preseed diagnostic subsequently matched all five previous Lt logit rows bit-for-bit and passed two request resets, but graph qualification and serving timing were not pursued after this decision. The external GDN prefill implementation also remains a reference for an interpreter implementation.

## Native Gemma12B batch-16 weight reuse experiment

Both variants execute inside the Plow interpreter and share the same segmented prefill objects. BF16, context 8192, input/output 128, concurrency 16, 32 measured requests, 16 warmups, seed 42. Three-token full-logit and request-reset diagnostics matched exactly between variants before timing.

| Decode weight reuse | Successful requests | TTFT ms | TPOT ms | p99 ITL ms | Mean E2E ms | p99 E2E ms | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| MM8 | 32 | 342.87 | 54.54 | 81.93 | 7269.72 | 7999.25 | 274.28 |
| MM16 | 32 | 321.41 | 48.22 | 74.47 | 6445.56 | 7200.21 | 309.51 |

MM16 reduces TPOT by 11.6% in this single comparison, but still trails the corresponding vLLM baseline (about 11.00 ms TPOT and 1364–1367 output tok/s). This supports investigating a small-batch tensor-core implementation; it does not qualify a default change or resolve Gemma numerical differences. Each process built 16 prefill graphs, one per slot. Raw results: `gemma12b-mm{8,16}-128-c16-r1/in128_c16.json` under the campaign directory, including end-to-end percentiles and detailed request data.

## Native kernel checkpoint

The cuBLASLt runtime flag remains available; optimization work prioritizes kernels inside the packet interpreter. Native BF16 activation caching passed isolated and full-model checks and improved the measured Qwen decode time; see [kernel comparison](qwen-h100-bf16-decode-kernel-comparison.md).

Qwen W8A8 batch-1 decode now uses explicit QuantFp8 packets and a native Hopper FP8 tensor-core projection arm. Weight scales come from the packed-tensor sidecar, activation scales are per token, output head/KV remain BF16 and recurrent state remains FP32. The loader rejects an unpaired interpreter or unsupported batch/prefill configuration. Ten emitter tests, the CUDA/HSA/hub workspace check, 213 runtime tests and release compilation passed. All 13 isolated FP8 shape gates passed, and the full model completed five teacher-forced tokens plus two exact request resets.

The initial FP8 kernel is slower than CUTLASS (QKV82.64 vs30.45µs). Its two-stage pipeline passed correctness but regressed QKV to86.45µs and remains disabled. Full-model leading tokens matched vLLM on all five prefixes, but centered logit error was2.8–4.5% and vLLM's first-prefix repeats themselves differed; qualification remains open. The initial kernel promotes accumulators every128K, whereas the installed CUTLASS M1 route uses fast accumulation; a separate matching-accumulation candidate is being checked. Raw evidence: `qwen-fp8-m1-native.json`, `qwen-fp8-m1-pipe.json`, `qwen-w8a8-native-quality.json` under `/tmp/plow-model-support-checks`, and `/tmp/plow-qwen-w8a8-vllm-short/manifest.json`.

A second Gemma rounding boundary was isolated by cutting the block0 packet after its post-attention normalization. Rounding that normalization's result to BF16 before residual addition reproduces the CPU BOS residual and following norm exactly. The opt-in `PLOW_NV_GEMMA_NRN_BF16=1` candidate, combined with HNR BF16 rounding, reduces the checked full-model prefix-6 error against vLLM from18.89% to4.73% and corrects its leading token496→9079. Prefix2 still differs (236747 vs575), so this is partial qualification only. Both block and full-model diagnostics completed, with exact request resets; decoder remains200registers with no stack/local memory. Evidence: `gemma12-nrn-replay.json` and `gemma12-hnr-nrn-quality.json` under `/tmp/plow-model-support-checks`.

The matching FP8 fast-accumulation primitive subsequently produced bit-exact outputs against installed CUTLASS on all 13 shapes, with exact activation bytes/scales. It remains slower (QKV 82.48 vs 30.43 µs). Full-model execution and two exact resets passed; centered logit differences against the first vLLM reference were 1.92%, 3.86%, 3.04%, 3.91%, 4.18%, with all five leading tokens matching. The unstable first-prefix vLLM repeat remains a reference limitation, and full-model qualification remains open. Evidence: `/tmp/plow-model-support-checks/qwen-fp8-m1-fastaccum.json` and `qwen-w8a8-fastaccum-quality.json`.

Increasing the single-stage FP8 K tile from 128 to 256 preserves accumulation order and passes all 13 bit-exact CUTLASS checks. Full-model logits match the fast-accumulation control byte for byte on five prefixes and two resets. QKV improves 82.48→68.67 µs, down 130.51→102.72 µs, and head 1047.74→841.62 µs; all measured shapes improve. The full interpreter uses 208 registers, no stack/local memory, and a 19456-byte arena. This is a kernel improvement, with no serving result yet. Evidence: `qwen-fp8-m1-bk256/result.json` and `qwen-w8a8-bk256-quality.json` under the checks directory.

The subsequent BK512 and BK1024 candidates also pass bit-exact CUTLASS checks on all13 model shapes plus K640/K528 tail cases. Each preserves all five full-model logit rows and two resets byte for byte against the fast-accumulation control. BK512 uses37888 arena bytes/236 interpreter registers; BK1024 uses74752 bytes/244 registers; both have zero stack/local memory. The primitive BK1024 harness explicitly opts into dynamic shared memory above48KiB, outside timed iterations.

| Native FP8 projection | BK128 µs | BK256 µs | BK512 µs | BK1024 µs | CUTLASS µs, latest run |
|---|---:|---:|---:|---:|---:|
| QKV | 82.48 | 68.67 | 53.55 | 50.72 | 30.43 |
| down | 130.51 | 102.72 | 82.53 | 75.39 | 44.62 |
| head stress shape | 1047.74 | 841.62 | 704.22 | 641.47 | 410.00 |

These are sequential kernel experiments with matched synthetic inputs and the same benchmark method, not serving measurements. BK1024 gains are smaller and not uniform: K640/K528 tails regress from7.62/7.74 to9.06/9.18µs. The model head remains BF16; the FP8 head row is a synthetic stress case. All variants remain opt-in. Evidence: `qwen-fp8-m1-bk{512,1024}/result.json`, provenance files, and `qwen-w8a8-bk{512,1024}-quality.json` under the checks directory.

Two subsequent native experiments preserve bit-exact CUTLASS results on all15 shapes. Contiguous output-row ownership yields mixed, small changes (QKV50.62µs, down73.46µs, head643.70µs) and has not advanced to a full interpreter build. Caching padded activations once per packet improves all measured shapes against BK1024: QKV48.19µs, down71.70µs, head611.98µs, tiny projection20.72µs. Cache setup is included in timing. The opt-in XCACHE full interpreter uses210 registers, no stack/spills/local memory, and205824 bytes of dynamic shared memory; total208272 bytes fits the queried232448-byte device limit. Five full-model rows and two resets remain bit-exact against the fast-accumulation control. Full-model vLLM FP8 qualification remains open. Evidence: `qwen-fp8-m1-{bk1024-blocked,xcache}/` and `qwen-w8a8-xcache-quality.json` under the checks directory.

Nsight Compute profiling of the fast-accumulation QKV control found 22% DRAM throughput, 8.19% compute throughput, 85.92% scheduler cycles with no eligible warp, and about 44.65% long-scoreboard stalls. This supports testing more outstanding memory work. Profiling uses a separate instrumented run; its timing is not substituted for benchmark timing. Evidence: `qwen-fp8-m1-qkv-profile.ncu-rep` and `qwen-fp8-m1-qkv-profile-details.txt`.

The additional opt-in `PLOW_NV_GEMMA_GLU_BF16=1` rounds gate/up projections and GELU activation to BF16 before multiplication, matching the installed vLLM activation primitive on six captured rows. Combined with HNR+NRN, Gemma12 now matches vLLM's leading token on all six short prefixes, with centered full-row errors 0.71%, 1.05%, 2.64%, 3.19%, 3.15%, 2.54%. Both exact reset repetitions passed. Prefix2 remains a zero-margin tie in Plow, and these short checks do not establish general model parity. Gemma31 also completed six tokens and two exact resets; its available CPU BOS reference has 0.85% centered error and the same leading token. No six-prefix vLLM31 reference was available for this check. Evidence: `gemma{12,31}-hnr-nrn-glu-quality.json` and `/tmp/plow-gemma{12,31}-hnr-nrn-glu/lifecycle.json`.

Native batch-16 tensor-core GEMV variants retain packet slice ownership and the existing12352-byte shared arena. Both passed16isolated shape checks. Their stronger control is native SIMD MM16, which already loads each weight once:

| Projection | SIMD MM16 µs | Tensor-core v1 µs | Warp-pipeline v2 µs |
|---|---:|---:|---:|
| gemma_down | 336.4 | 203.3 | 303.2 |
| gemma_q | 115.9 | 61.2 | 133.4 |
| gemma_o | 133.4 | 108.2 | 153.4 |
| qkv | 213.6 | 142.7 | 229.1 |
| k_or_v | 24.3 | 64.9 | 94.3 |
| down | 531.9 | 240.8 | 369.0 |
| lm_head | 4905.9 | 2253.8 | 3092.7 |

The warp-pipeline revision regresses every routed shape against v1 and remains disabled. No tensor-core variant is promoted for serving; full-model qualification and shape-specific selection remain necessary. Raw logs: `gemv-m16-mma/result1.log` and `gemv-m16-mma-pipeline/result{,-mm16-control}.log` under `/tmp/plow-model-support-checks`.

The v1 interpreter was rebuilt from its frozen source with `GV_MM_MAX=16`, removing the earlier MM8 fallback confound. Three-prefix full-model checks match the SIMD control's leading tokens, with centered full-row differences 0.52%, 1.04%, 2.71% and reference-head64 differences 0.27%, 0.29%, 0.62%. Two slot0 resets and a slot15 replay reproduce the candidate's own logits exactly. This does not establish general numerical equivalence.

A fresh serving pair uses the same frozen `plowrt-qwen-w8a8-candidate1` runtime, Gemma12 B16 asset, segmented prefill, input/output128, C16, 32 measured requests, 16 warmups and seed42; only the decoder cubin changes. Both complete32/32 with zero failures:

| Decoder | TTFT ms | TPOT ms | p99 ITL ms | Mean E2E ms | Output tok/s |
|---|---:|---:|---:|---:|---:|
| SIMD MM16 control | 321.72 | 48.15 | 74.48 | 6436.49 | 309.96 |
| Tensor-core v1, MM16 fallback | 300.44 | 42.12 | 68.47 | 5650.08 | 353.26 |

The measured TPOT reduction is12.51% and throughput increase13.97%. This single pair supports further correctness work and repetition, not promotion; vLLM remains about11ms TPOT at this cell. Raw serving files: `gemma12b-m16-mma{,-control}-128-c16-r1/in128_c16.json` under the campaign directory. Build and quality artifacts: `/tmp/plow-model-support-checks/gemv-m16-mma-mm16fallback/`.

## First 8192-token correctness cell

Gemma12 BF16 was compiled for context16384, batch1, with the existing WS384 segmented prefill. Native prefill of8192 tokens plus two teacher-forced continuations completed with finite logits. The matching vLLM0.28 oracle uses identical token IDs/context, BF16, eager execution, max-num-batched-tokens8192 and prefix caching off; all three prefix repeats are bit-exact. The oracle now records effective checkpoint suppression IDs258882/258883 and accepts negative infinity only at those declared IDs, preserving full raw vectors. Three CPU tests cover suppression and rejection behavior.

Both native stock and the decoder-only HNR+NRN+GLU variant match vLLM's leading token and top5 set at all three prefixes. Numerical qualification remains open: stock centered full-row errors are8.46%,6.93%,7.88%; decoder candidate errors are8.46%,7.34%,7.24%. Reference-head64 errors are2.40–4.41%. The first prefill row is bit-exact between native variants because their prefill objects are unchanged. These results require checking a matching prefill rounding build; they do not establish long-context parity. Evidence: `/tmp/plow-gemma-long-context/vllm-comparison.json`, native manifests and `vllm-bf16/manifest.json` in the same directory.

The corresponding vLLM BF16 serving baseline (context16384, input8192/output128, C1, max-num-seqs1, max-num-batched-tokens8192, default graphs, prefix cache off, 32 measured requests/16 warmups/seed42) completed32/32 with262144 input and4096 output tokens, zero failures. TTFT350.20ms, TPOT10.637ms, p99ITL11.547ms, meanE2E1701.10ms, output75.23tok/s. Raw: `vllm-gemma12b-bf16-ctx16384-i8192-o128-c1-r1/in8192_c1.json` under the campaign directory. Native timing is not yet a qualified comparison.

The corresponding stock native run also completed32/32 with262144 input and4096 output tokens, zero failures. The same context/input/output/concurrency/request/warmup/seed settings give TTFT636.80ms, TPOT12.865ms, p99ITL12.931ms, meanE2E2270.71ms and output56.36tok/s. It uses resident batch1, native decode, WS384 segmented prefill, and the frozen `plowrt-qwen-w8a8-candidate1` host. Plow is slower on this cell; the unresolved output discrepancy below makes this a provisional performance comparison. Raw: `gemma12b-bf16-native-ctx16384-b1-i8192-o128-c1-r1/in8192_c1.json` under the campaign directory. The initial launch used an incorrect served model ID and stopped at its404 correctness gate; these numbers come from the successful retry with the asset ID `checkpoint`.

The oracle now supports multiple output tokens from one request, recording the actual generated-prefix hash for every raw row. This distinguishes real decode from recomputing each prefix. Seven CPU tests cover suppression, history mapping, context bounds and preservation of distinct invalid-row diagnostics.

At this8192-token history, eager vLLM prefill plus two decode steps generates `[4284,9885,13512]`, matching all native variants; both requests repeat bit-exactly. Compiled/default-graph vLLM instead generates `[4284,9885,9885]`, also with bit-exact repeats. The third compiled raw row differs substantially from the native/reference rows. The installed FlatLogprobs export was checked for indexing/copy errors; none were found. A compile-on/graphs-off control is required before attributing this discrepancy or treating the default-graph timing as a qualified model comparison. Artifacts: `/tmp/plow-gemma-long-context/vllm-bf16-{compiled,eager}-decode/`.

The subsequent compile-on/graphs-off control completed successfully with effective mode `VLLM_COMPILE`, Inductor backend, custom ops `none`, and graph mode `NONE`. Both requests still generate `[4284,9885,9885]`; all six raw rows are byte-identical to the default-graph run. CUDA graph replay therefore does not explain this observed discrepancy. The remaining investigation concerns compiled execution and its selected operators/fusions; no vLLM bug is established. Evidence: `/tmp/plow-gemma-long-context/vllm-bf16-compiled-nographs-decode/manifest.json` and `/tmp/plow-model-support-checks/gemma-compiled-graph-control.json`.

Changing only the RMSNorm IR provider preference to `vllm_c` removes the third-token greedy divergence on this history: both requests generate `[4284,9885,13512]`, matching eager vLLM and native Plow. Effective execution remains `VLLM_COMPILE`/Inductor, `FULL_AND_PIECEWISE` graphs and custom ops `none`; RMSNorm priority becomes `[vllm_c,native]`, with other providers unchanged. All three repeated rows are byte-identical and all prefix hashes match the original comparison. The default compiled third row differs from eager by65.89% centered full-logit relative L2; this diagnostic control reduces that difference to7.91%.

| Prefix length | RMSNorm control vs eager, centered full relative L2 | RMSNorm control vs native, centered full relative L2 |
|---|---:|---:|
|8192|9.09%|8.73%|
|8193|9.18%|8.45%|
|8194|7.91%|8.43%|

This implicates the normalization implementation or its surrounding compiler fusion boundaries, without establishing a specific vLLM bug. Remaining logit differences leave numerical qualification open. The benchmark reference remains default compiled vLLM; this provider override is diagnostic. Exact file hashes, token margins, prefix checks and all pairwise metrics: `/tmp/plow-model-support-checks/gemma-compiled-rms-provider-result.json`; raw rows: `/tmp/plow-gemma-long-context/vllm-bf16-compiled-vllmc-norm-decode/`.

An isolated installed-forward probe also confirms a narrower distinction: eager execution rounds the residual sum to BF16 before applying the layer scalar, whereas default Inductor fusion matches Plow's FP32 sum followed by scaling. All six captured rows agree with those respective forms. No scalar-rounding production change was made. Evidence: `/tmp/plow-model-support-checks/vllm-gemma-layer-scalar-boundary.json`.

The standalone prefill GELU activation rounding was added under the same opt-in Gemma flag and tested with a fresh PFseg control/candidate pair, retaining identical GEMM, attention and decode objects. Its effect is mixed: against eager recomputed prefixes, centered errors change8.46/7.34/7.24%→10.72/7.92/6.92%, with all three leading tokens unchanged. It remains disabled by default and is not promoted. The fresh PFseg control reproduces the prior control logits; both builds retain the original255-register/1672-byte-stack resource profile. Evidence: `/tmp/plow-model-support-checks/gemma-pf-rounding/quality-eager.json`.

## Gemma31 short vLLM reference

The corrected native HNR+NRN+GLU decoder was compared against six identical teacher-forced prefixes in vLLM, repeated twice. All six vLLM repeats are bit-exact and all native leading tokens match. Centered full-row errors are0.85%,4.94%,2.80%,3.04%,2.77%,2.65%; head64 errors are0.33–1.49%. Unlike this Gemma12 checkpoint, Gemma31 declares no suppressed IDs and its full raw vocabulary is finite; no entries are masked in this comparison. Short-prefix agreement does not qualify all contexts. Evidence: `/tmp/plow-model-support-checks/gemma31-vllm-glu-quality.json` and `/tmp/plow-gemma31-vllm-short/manifest.json`.

## Query caching: kernel gain without a serving gain

The default-off `PLOW_NV_FA_QREG` caches invariant BF16 query fragments in registers. SASS confirms query shared loads move outside the KV-row loop, with unchanged arithmetic/reduction order. The final candidate is restricted to D512/GF≤4 because the Qwen D256/GF2 experiment regresses91.57→97.73µs. The full Gemma interpreter uses200→207 registers with zero stack/spills/local memory.

At context8192, GF4, nsplit33, the actual Gemma12 NH16/KV1/D512 shape improves mean cold decode+merge67.35→49.89µs across three trials; Gemma31 NH32/KV4/D512 improves128.00→91.94µs. Both pass the unchanged CPU softmax reference gate and exact A/B output/partial-state comparison. Earlier NH8 tests labelled Gemma12 were synthetic; their69.05→49.69µs result must not be substituted for the actual NH16 shape.

Full Gemma12 prefill8192 plus two decode steps produces identical logit rows for both full cubins, and the control also matches the prior stock decoder byte for byte. A fresh serving pair uses the same frozen host, context16384/B1 asset, WS384 prefill, input8192/output128/C1, 32 requests, 16 warmups, seed42 and detailed metrics. Both complete32/32 with zero failures:

| Decoder | TTFT ms | TPOT ms | p99 ITL ms | Mean E2E ms | Output tok/s |
|---|---:|---:|---:|---:|---:|
| QREG off | 637.57 | 12.878 | 13.001 | 2273.02 | 56.307 |
| QREG HD512 | 637.56 | 12.879 | 12.972 | 2273.24 | 56.301 |

There is no measured serving gain in this pair; the candidate remains experimental. An initial control launch omitted the existing BOS-prefixed coherence prompt and stopped at that gate without timing. The successful pair restores `GATE_PROMPT="<bos>The capital of France is"`, matching the earlier stock run. Raw results: `gemma12b-qreg{0,1}-ctx16384-i8192-o128-c1-r1/in8192_c1.json` under the campaign directory. Cubin/primitive provenance and full-logit comparisons: `/tmp/plow-attention-qreg/`.

The subsequent AOT packet-fusion, dependency, shared-quantization, prefill-norm and object-selection experiments are recorded in [the compiler experiment report](h100-aot-compiler-experiments.md).
