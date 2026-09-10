# Gemma 3 12B: vLLM comparison baseline

Screened on H100 80GB, 2026-09-10. This is a baseline, not evidence that Plow beats vLLM.
One warmup and one measured batch per case. Final comparisons require repeated measurements.

Both engines use the same Gemma 3 checkpoint weights. FP8 uses the same 336 per-output-channel E4M3 projection matrices, with BF16 activations, KV, embeddings, and head. The FP8 data region was copied without requantization and its SHA256 verified. vLLM uses compressed-tensors W8A16.

vLLM 0.29.0 / torch 2.13.0+cu130; maximum model length 16640, maximum sequences 128, GPU memory utilization 0.90, chunked prefill 8192, language-model-only. Prefix caching disabled. Exactly 128 output tokens, temperature zero, ignore EOS. Each precision has 298 measured requests with verified exact input/output counts and zero cache hits.

| Precision | Input | Concurrency | TTFT median ms | TPOT median ms | E2E median ms | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1024 | 1 | 42.75 | 10.36 | 1358.34 | 94.20 |
| BF16 | 1024 | 4 | 142.88 | 10.80 | 1514.23 | 337.94 |
| BF16 | 1024 | 16 | 422.94 | 13.60 | 2150.30 | 948.99 |
| BF16 | 1024 | 128 | 2268.57 | 42.79 | 7702.97 | 1792.48 |
| BF16 | 16384 | 1 | 590.20 | 10.59 | 1935.39 | 66.10 |
| BF16 | 16384 | 4 | 1655.64 | 16.18 | 3710.49 | 136.71 |
| BF16 | 16384 | 16 | 4934.83 | 45.92 | 10766.28 | 186.91 |
| BF16 | 16384 | 128 | 39624.52 | 134.45 | 58205.20 | 202.42 |
| FP8 W8A16 | 1024 | 1 | 54.30 | 7.55 | 1012.53 | 126.34 |
| FP8 W8A16 | 1024 | 4 | 166.05 | 7.94 | 1173.82 | 435.79 |
| FP8 W8A16 | 1024 | 16 | 468.94 | 10.71 | 1829.54 | 1115.37 |
| FP8 W8A16 | 1024 | 128 | 2588.01 | 42.74 | 8014.38 | 2018.77 |
| FP8 W8A16 | 16384 | 1 | 645.37 | 7.79 | 1635.28 | 78.23 |
| FP8 W8A16 | 16384 | 4 | 1893.60 | 13.95 | 3664.94 | 138.75 |
| FP8 W8A16 | 16384 | 16 | 5631.77 | 47.50 | 11663.99 | 173.36 |
| FP8 W8A16 | 16384 | 128 | 43164.44 | 186.24 | 66834.71 | 183.51 |

Raw JSONL records (including per-request arrival times, tails, hashes, text, usage and manifests): [BF16](gemma3-12b-h100/vllm-bf16-screen.json), [FP8](gemma3-12b-h100/vllm-fp8-screen.json). FP8 [configuration](gemma3-12b-h100/vllm-fp8-config.json) and [provenance](gemma3-12b-h100/vllm-fp8-provenance.json).

Plow initial improvement measurements are in [the roofline report](gemma3-12b-h100-roofline.md). Subsequent packed/occupancy candidates remain under qualification; the full competitive goal remains unmet.

Nix environment notes: use Nix ninja on PATH; Humming nvrtc_compile needs the Nix ELF interpreter, and the CUDA wheel library directory must be on LD_LIBRARY_PATH so NVRTC finds its builtins. These are loader fixes; the vLLM kernel algorithms and quantization are unchanged.
