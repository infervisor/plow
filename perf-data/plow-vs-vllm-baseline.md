# plow vs vLLM — bf16 single-user baseline (MI350X / gfx950)

Refreshed 2026-07-15 with (a) the Qwen/Llama optimization campaign and (b) a fresh vLLM **docker**
sweep. bf16, batch 1, greedy, TP 1, one gfx950 GPU. Correctness: full-completion HF token match.

**Three data columns:**
- **plow main** (`a109dcf`) and **plow campaign** (verified wins on branches `qwen-prefill-perf`/
  `qwen-decode-perf`/`llama-decode-perf` — flash D=128 8-wave object, AddNorm fusion, exact-tiling
  GEMV; NOT yet merged to main). plow "prefill" is PURE prefill.
- **vLLM docker** = `rocm/vllm:latest` → **0.11.2**, served endpoint, `vllm bench serve` concurrency 1,
  HIP graphs ON. TTFT = prefill + 1st token; TPOT = per-token decode. Harness: `bench_vllm_docker.sh`
  (reproducible). Data: `{qwen3-4b,llama-3.1-8b}-vllm-docker.json`.

**⚠ vLLM version discrepancy — flagged, not resolved:** the committed `*-vllm-perf.json` (vLLM **0.25.1
native in-process**) reports prefill ~1.7× SLOWER than this **0.11.2 docker served** sweep (Qwen 8k:
147 ms in-process vs 88 ms served TTFT). Different version AND method (in-process `vllm.LLM` vs served
HTTP), so not a controlled tie. The served TTFT below is the more standard metric; using it, plow's
prefill gap is LARGER than the 0.25.1 numbers implied. A controlled same-version A/B is the fix.

## Prefill — plow main / campaign (pure prefill, ms) vs vLLM docker TTFT (ms), ratio = campaign/vLLM

| model | ctx | plow main | campaign | vLLM docker | x |
|---|---|---|---|---|---|
| **Qwen3-4B** | 4k | 222 | **148** | 51 | **2.9** |
| | 8k | 651 | **356** | 88 | **4.0** |
| | 16k | 2178 | — | 232 | — |
| **Llama-3.1-8B** | 4k | 237 | **173** | 59 | **2.9** |
| | 8k | 649 | **393** | 95 | **4.1** |
| | 16k | 2087 | — | 241 | — |

## Decode — plow main / campaign (ms/token) vs vLLM docker TPOT (ms), ratio = campaign/vLLM

| model | ctx | plow main | campaign | vLLM docker | x |
|---|---|---|---|---|---|
| **Qwen3-4B** | 4k | 4.8 | **4.7** (+~5% avail: nsplit) | 3.26 | **1.44** |
| | 8k | 5.2 | — | 3.39 | — |
| **Llama-3.1-8B** | 4k | 5.5 | **5.2** | 4.27 | **1.22** |
| | 8k | 5.9 | **5.6** | 4.34 | **1.29** |
| | 16k | 6.3 | **6.2** | 4.50 | **1.38** |

vLLM docker at long ctx (single-user, for reference): Qwen 32k TTFT 710 ms / TPOT 4.08; Llama 32k
706 / 4.93, **64k 2342 / 5.93**. (Qwen native cap 40960 — not swept past 32k. Gemma-31B not run in
this docker sweep.)

## The plow baseline, restated

- **Gemma-4-31B (design target):** prefill ~1.4×, decode ~parity (wins vLLM decode past ~16k via flat
  sliding-KV). The megakernel's sweet spot; unchanged by the campaign. (vs the committed 0.25.1 data.)
- **Small models — the campaign roughly HALVED the prefill gap** (Qwen 8k main 7.4× → campaign 4.0×
  vs docker vLLM; Llama similar). The lever: flash D=128 (a Gemma-tuned kernel ran at 2% of MFMA peak
  on head_dim 128 → a D=128-only 8-wave object). Decode closed modestly (AddNorm + GEMV tiling):
  Qwen ~1.5×→1.44×, Llama ~1.3–1.5×→1.22–1.38×.

**What's measured about the remaining gap:**
- Prefill is now **78% attention (flash), softmax-VALU-bound** (MFMA only 1.8% of issue at D=128) —
  algorithmic; fp8 barely helps it. The 22% GEMM has a real **+15% power-aware lever** (non-power-of-2
  192×256 arch-only tile, avoids AGPR moves in the power-limited 8-GPU regime).
- Decode is **overhead/occupancy-bound, NOT bandwidth-bound** (Qwen 27% of HBM roofline vs vLLM 42%).
  vLLM's edge is fewer/larger fused ops, not bandwidth or graphs — plow's megakernel is already
  **zero-launch (1 dispatch/decode-token; verified, a HIP graph reclaims nothing)**. QKV fusion is
  bs=1-neutral; the ~5% flash_decode nsplit reduction is the real decode win. KV storage (head-major)
  is jointly read+write optimal (verified).

Data files: `{gemma4,llama-3.1-8b,qwen3-4b,qwen3-1.7b}-vllm-perf.json` (0.25.1 native, prefill runs
slower — see discrepancy above); `{qwen3-4b,llama-3.1-8b}-vllm-docker.json` + `bench_vllm_docker.sh`
(0.11.2 docker, reproducible, THIS baseline's vLLM column).
</content>
