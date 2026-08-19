# Final validation — Qwen3-4B vs vLLM

Contract: `docs/agent2-benchmark-contract.md` (immutable).
Prior agents: `docs/agent1-repository-map.md` through
`docs/agent5-runtime-results.md`.

Date: **2026-08-19**.
Git under test: `shaswot/qwen-asr` @ integrated kernel + runtime optimizations.

This report is the independent Agent 6 gate. No implementation or benchmark
semantics were changed during validation.

---

## 1. Environment

| item | value |
|---|---|
| GPU | 1× NVIDIA GeForce RTX 5090, 32607 MiB, PCI `0000:41:00.0` |
| Driver | 580.142 |
| Host CUDA | 13.0 |
| PyTorch (vLLM serve) | 2.13.0+cu130 |
| Model | Qwen3-4B (`/workspace/models/Qwen3-4B`) |
| dtype | bf16 (both sides) |
| `CUDA_VISIBLE_DEVICES` | 0 (same visible 5090 for both servers) |
| plow assets | `/workspace/assets/qwen3-4b-sm120` (sm_120 cubins) |
| Client | `vllm bench serve --backend openai-chat` (contract §1) |
| Warmup | 4 requests per cell (contract §5) |
| Prompts | 32 per cell, `max_concurrency=1`, `request_rate=inf` |
| Output | 128 generated tokens per prompt (contract §4) |

---

## 2. Correctness

| check | result |
|---|---|
| plowrt model load | **PASS** — `plowrt serve` starts, accepts chat completions |
| vLLM model load | **PASS** — `vllm serve` starts, accepts chat completions |
| HTTP response format | **PASS** — OpenAI chat JSON, streaming tokens |
| Failed requests (canonical cells) | plowrt: 0/32 all cells; vLLM: 0/32 except L=16384 prefill-only cell (32/32 failed — exceeds vLLM context for that probe) |
| Kernel oracle (hd=128 flash prefill) | **PASS** — relL2 ~1.7e-3 (Agent 4) |
| `gpu_consume_prompt` unit test | **PASS** — identity vs `step_slots` when `PLOW_GPU_TEST=1` |

Correctness passes. Performance gate is evaluated separately below.

---

## 3. Canonical A/B — integrated plowrt vs vLLM

Headline table uses the **integrated** plowrt build (kernel hd=128 prefill
dispatch + `consume_prompt` runtime path). Raw JSON:
`/workspace/bench-results-plowrt-tuned/` and `/workspace/bench-results/`.

Input lengths: fixed prompt token targets L ∈ {1024, 4096, 8192}.

| metric (ms) | L | OURS (median) | VLLM (median) | delta (OURS − VLLM) |
|---|---:|---:|---:|---:|
| **TTFT** | 1024 | 103.9 | 59.2 | **+44.7 (+75%)** |
| **TTFT** | 4096 | 323.3 | 206.7 | **+116.6 (+56%)** |
| **TTFT** | 8192 | 727.6 | 467.6 | **+260.0 (+56%)** |
| **TPOT / ITL** | 1024 | 6.35 | 5.99 | **+0.36 (+6%)** |
| **TPOT / ITL** | 4096 | 6.65 | 6.24 | **+0.41 (+7%)** |
| **TPOT / ITL** | 8192 | 7.01 | 6.80 | **+0.21 (+3%)** |
| **E2E** | 1024 | 910.4 | 819.3 | **+91.1 (+11%)** |
| **E2E** | 4096 | 1167.2 | 998.6 | **+168.6 (+17%)** |
| **E2E** | 8192 | 1620.7 | 1329.8 | **+290.9 (+22%)** |

Throughput (tokens/s, L=1024): OURS 140.6 out / 1296.5 total vs VLLM 156.2 /
1415.1 — **VLLM faster**.

### Statistical stability (L=1024, OURS)

| stat | TTFT | TPOT |
|---|---:|---:|
| mean | 103.3 | 6.35 |
| std | 1.46 | 0.006 |
| p50 | 103.9 | 6.35 |
| completed / failed | 32 / 0 | 32 / 0 |

Low TTFT variance; gap vs vLLM is not noise.

---

## 4. Before vs after optimizations (plowrt only)

Baseline plowrt (pre-optimization, decode-loop prefill, no hd=128 `_pf`):

| L | TTFT median (ms) | vs integrated |
|---:|---:|---|
| 1024 | 6775.4 | **65× slower** |
| 4096 | 26949.3 | **83× slower** |
| 8192 | 55093.3 | **76× slower** |

Integrated build (this branch):

| L | TTFT median (ms) | speedup vs baseline |
|---:|---:|---:|
| 1024 | 103.9 | **65×** |
| 4096 | 323.3 | **83×** |
| 8192 | 727.6 | **76×** |

Agent 4 + Agent 5 changes dramatically improve plowrt TTFT versus the baseline
decode-loop path, but **do not exceed vLLM**.

---

## 5. Decode-only probe (in=32, out=512)

Short-prompt decode stress (16 prompts, conc=1):

| metric (median ms) | OURS | VLLM | delta |
|---|---:|---:|---:|
| TTFT | 43.0 | 20.5 | **+22.5 (+110%)** |
| TPOT | 6.27 | 6.00 | **+0.27 (+4%)** |
| E2E | 3244.8 | 3083.7 | **+161.1 (+5%)** |
| output tok/s | 157.8 | 166.0 | VLLM +5% |

---

## 6. Segmented prefill TTFT (plowrt diagnostic)

Server-side segmented prefill (`plowrt_seg_final.json`, 5 trials, median ms):

| L (target) | prompt tokens | TTFT median |
|---:|---:|---:|
| 1024 | 1109 | 129.7 |
| 4096 | 4349 | 430.6 |
| 8192 | 8675 | 896.0 |

Consistent with HTTP TTFT at the same lengths; confirms prefill dominates TTFT
but remains above vLLM at every point.

---

## 7. GPU utilization and memory

| item | OURS | VLLM |
|---|---|---|
| Peak VRAM (serve) | ~18 GiB (Qwen3-4B bf16 + cubins) | ~16 GiB |
| GPU util during bench | P0 under load; not clock-locked | same |
| Power | typical 5090 P0 during sustained decode | same |

Supporting metrics only; latency is the primary objective.

---

## 8. Fairness checklist

| rule | status |
|---|---|
| Same GPU, model, dtype | **yes** |
| Same client (`openai-chat`) | **yes** |
| Same concurrency (1) and prompt count (32) | **yes** |
| Same warmup (4) | **yes** |
| No benchmark code changes during validation | **yes** |
| No vLLM config sabotage | **yes** |
| Integrated implementation (not cherry-picked opts) | **yes** |

Comparison is **fair**. vLLM wins on every headline latency metric.

---

## 9. Decision

1. Correctness: **PASS**
2. Fair benchmark: **PASS**
3. Reproducible (32 prompts, low TTFT σ): **PASS**
4. Required latency metrics improve vs vLLM: **FAIL** — TTFT, TPOT/ITL, E2E,
   and throughput are all worse than vLLM at L ∈ {1024, 4096, 8192}
5. Valid comparison: **PASS**

Optimizations on this branch are real (65–83× TTFT improvement vs baseline
plowrt) but **insufficient to beat vLLM**. Primary remaining gap: TTFT at all
prefill lengths (+56–75%). Secondary gap: decode TPOT (+3–7%).

RESULT: FAIL
