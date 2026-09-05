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
- Packed-tensor FP8 weight export and opt-in `PLOW_NV_QUANT_FP8_VLLM=1` activation quantization prepare matching W8A8 arithmetic. Activation bytes and scale bits matched vLLM in all 45 checked GPU cases. Full-model W8A8 execution is not implemented; the isolated FP8 M1 cuBLASLt prototype is compiled but awaits GPU qualification.
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
