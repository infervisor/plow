# vLLM fp8 DECODE baseline (MI350X / gfx950) — the number plow-fp8 must beat

Measured **2026-07-17**, single-user (batch 1, `--max-concurrency 1`), TP-1, one gfx950 (MI350X-class)
GPU, chunked-prefill budget 8192, 128 output tok/req, 3 prompts/point, random dataset. Reproduce with
`perf-data/bench_vllm_docker.sh` (now image-agnostic + fp8-capable).

- **Image:** `vllm/vllm-openai-rocm:latest` → **vLLM `0.25.1+rocm723`** (ROCm 7.2.3).
- **Three configs per model, same image (apples-to-apples):**
  - `bf16`  — `--dtype bfloat16` (baseline).
  - `fp8`   — `--quantization fp8` (weight-only dynamic e4m3 from the bf16 checkpoint).
  - `fp8kv` — `--quantization fp8 --kv-cache-dtype fp8`.
- **Metric that matters:** `tpot_ms` = Mean TPOT = Mean ITL = **decode ms/token** (lower is better).
  TTFT (prefill) recorded too but decode is the campaign target.
- Data: `perf-data/{gemma4,llama8b,qwen3-4b}-vllm-fp8.json` (all three configs, every ctx).

## HEADLINE FINDING #1 — vLLM fp8 WORKS on gfx950 (prior "broken" verdict overturned)

The earlier note in this file said `--quantization fp8` crashes on gfx950 (`torch_channelwise_w8a8_scaled_mm`
"dimension size (1)"). **That was the older `rocm/vllm:latest` = vLLM `0.11.2.dev673`.** On the newer
`vllm/vllm-openai-rocm:latest` = **vLLM `0.25.1+rocm723`**, both `--quantization fp8` and
`--kv-cache-dtype fp8` start cleanly and produce **coherent** output (chat "capital of France?" → "Paris",
per config, all three models). Gemma-4-31B (`Gemma4ForConditionalGeneration`, `gemma4` arch) — which did
not even parse on the 0.11.2 image — **serves fine on 0.25.1** at TP-1, native ctx, no rope extension.

So there is now a real vLLM fp8 reference. But:

## HEADLINE FINDING #2 — vLLM fp8 does NOT help single-user decode on the SMALL models; it only helps the LARGE one

At batch 1, vLLM's fp8 `scaled_mm` has a fixed per-GEMM overhead that dominates when the weight matrices
are small. Result splits by model size:

| model | fp8 weight-only vs bf16 decode | why |
|---|---|---|
| Qwen3-4B  | **SLOWER** +22–25% at every ctx | 4B GEMMs too small; fp8 dequant/scaled_mm overhead > HBM saving |
| Llama-3.1-8B | ~neutral, **+2%** (marginally slower) | 8B still overhead-bound at batch 1 |
| Gemma-4-31B | **FASTER −10% to −19%** | 31B decode is weight-bandwidth-bound; fp8 halves the weight stream, overhead amortized |

`fp8kv` (fp8 KV cache) is the **universal long-context decode win** — it shrinks the KV read stream, so its
benefit grows with ctx and is independent of model size (helps Llama a lot at long ctx where weight-only fp8
did nothing).

### The vLLM number plow-fp8 must beat (BEST vLLM decode config per model), decode TPOT ms/tok

| ctx    | Qwen3-4B (best=bf16) | Llama-3.1-8B (best) | Gemma-4-31B (best=fp8kv) |
|--------|----------------------|---------------------|--------------------------|
| 1024   | **3.19** (bf16)      | **4.07** (fp8kv)    | **11.03** (fp8kv)        |
| 4096   | **3.30** (bf16)      | **4.11** (fp8kv)    | **11.98** (fp8kv)        |
| 8192   | **3.37** (bf16)      | **4.17** (fp8kv)    | **13.12** (fp8kv)        |
| 16384  | **3.55** (bf16)      | **4.28** (fp8kv)    | **13.92** (fp8kv)        |
| 32768  | **3.86** (bf16)      | **4.39** (fp8kv)    | **16.51** (fp8kv)        |
| 65536  | — (native cap)       | **4.67** (fp8kv)    | **18.39** (fp8kv)        |
| 128k   | — (native cap)       | **5.37** (fp8kv, @130816) | **22.18** (fp8kv, @131072) |

For **Qwen3-4B**, vLLM fp8 is a regression — plow-fp8 competes against vLLM **bf16** (3.19–3.86 ms).
For **Gemma-4-31B**, vLLM's own fp8 already beats its bf16, so plow-fp8 must beat vLLM-**fp8kv** (the harder bar).

---

## Full decode tables (TPOT ms/tok, all configs)

### Qwen3-4B (native 40960, no rope extension)
| ctx | bf16 | fp8 | fp8kv | fp8 Δ vs bf16 | fp8kv Δ vs bf16 |
|---|---|---|---|---|---|
| 1024  | 3.190 | 3.950 | 3.960 | +23.8% | +24.1% |
| 4096  | 3.300 | 4.040 | 4.090 | +22.4% | +23.9% |
| 8192  | 3.370 | 4.170 | 4.160 | +23.7% | +23.4% |
| 16384 | 3.550 | 4.380 | 4.360 | +23.4% | +22.8% |
| 32768 | 3.860 | 4.820 | 4.570 | +24.9% | +18.4% |

### Llama-3.1-8B-Instruct (native 131072; 128k point = input 130816 so +128 out fits)
| ctx | bf16 | fp8 | fp8kv | fp8 Δ vs bf16 | fp8kv Δ vs bf16 |
|---|---|---|---|---|---|
| 1024   | 4.080 | 4.190 | 4.070 | +2.7%  | −0.2% |
| 4096   | 4.250 | 4.320 | 4.110 | +1.6%  | −3.3% |
| 8192   | 4.310 | 4.380 | 4.170 | +1.6%  | −3.2% |
| 16384  | 4.460 | 4.570 | 4.280 | +2.5%  | −4.0% |
| 32768  | 4.790 | 4.840 | 4.390 | +1.0%  | −8.4% |
| 65536  | 5.560 | 5.650 | 4.670 | +1.6%  | −16.0% |
| 130816 | 7.200 | 7.280 | 5.370 | +1.1%  | −25.4% |

### Gemma-4-31B-it (served max-model-len 132096; text native 262144, no rope extension)
| ctx | bf16 | fp8 | fp8kv | fp8 Δ vs bf16 | fp8kv Δ vs bf16 |
|---|---|---|---|---|---|
| 1024   | 13.720 | 11.120 | 11.030 | −19.0% | −19.6% |
| 4096   | 14.590 | 12.160 | 11.980 | −16.7% | −17.9% |
| 8192   | 15.850 | 13.350 | 13.120 | −15.8% | −17.2% |
| 16384  | 16.670 | 14.220 | 13.920 | −14.7% | −16.5% |
| 32768  | 19.450 | 16.930 | 16.510 | −13.0% | −15.1% |
| 65536  | 21.550 | 19.020 | 18.390 | −11.7% | −14.7% |
| 131072 | 25.710 | 23.140 | 22.180 | −10.0% | −13.7% |

## Prefill (TTFT ms) — recorded, not the target

TTFT rows live in the JSONs. Note: **ctx=1024 TTFT is warm-up-contaminated** (first bench shape triggers
HIP-graph capture — e.g. Qwen bf16 1k=272 ms but 4k=65 ms); 4k+ is clean. fp8kv consistently lowers TTFT
at long ctx (smaller KV writes): e.g. Llama 128k TTFT 19367 ms bf16 → 15188 ms fp8kv; Gemma 128k prefill is
~41–46 s (31B, TP-1). Decode TPOT is unaffected by the 1k warm-up artifact.

## Cross-check vs the committed bf16 docker baseline (0.11.2)

The committed `perf-data/vllm-docker-baseline.md` bf16 (image 0.11.2) agrees with this run's 0.25.1 bf16
column within a few %: Qwen 32k 4.08 (0.11.2) vs 3.86 (0.25.1); Llama 64k 5.93 vs 5.56. The 0.25.1 image
is marginally faster and, unlike 0.11.2, serves Gemma and runs fp8 — so it is the correct apples-to-apples
base for the fp8 comparison here.

## Reproduce
```
IMAGE=vllm/vllm-openai-rocm:latest QUANT=fp8 KVFP8=0 GPU=<n> TP=1 \
  PORT=<p> CNAME=<c> OUTDIR=<dir> \
  MODELS_OVERRIDE="Qwen3-4B:40960:1024,4096,8192,16384,32768" \
  bash perf-data/bench_vllm_docker.sh
```
Set `QUANT=bf16` for baseline, `QUANT=fp8 KVFP8=1` for fp8+KV. The harness now uses `--entrypoint vllm`
(works for both the 0.25.1 `["vllm","serve"]` image and the 0.11.2 `/bin/bash` image) and a `TP` knob.

## Bottom line for the fp8-decode campaign
1. plow-fp8 competes against **working** vLLM fp8 now (not a capability gap) — EXCEPT that vLLM fp8 is a
   **decode regression on small models** (Qwen −0, actually slower). There, beating vLLM means beating vLLM **bf16**.
2. vLLM's fp8 win on Gemma-31B (−10…−19%) confirms the campaign's central thesis: at batch-1 the payoff
   scales with weight-matrix size (bandwidth-bound) and is eaten by dequant overhead when the GEMMs are small.
   plow-fp8 must keep the cvt/dequant off the critical path to beat vLLM-fp8 on Gemma and to beat vLLM-bf16 on Qwen/Llama.
3. `fp8kv` is the clean long-context decode lever for every model (Llama −25% @128k, Gemma −14% @128k) —
   plow's fp8-KV should target/exceed this.
