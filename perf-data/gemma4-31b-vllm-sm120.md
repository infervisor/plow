# vLLM baseline — Gemma-4-31B-it on RTX PRO 6000 Blackwell (sm_120), single-user — campaign B1

Measured **2026-07-18**, single-user (batch 1, `--max-concurrency 1`), TP-1, one **NVIDIA RTX PRO 6000
Blackwell Server Edition** (sm_120 / cc 12.0, 188 SMs, 97887 MiB), chunked-prefill budget 8192,
128 output tok/req, 3 prompts/point, random dataset, served endpoint, CUDA graphs ON. These are the
**faithful vLLM numbers plow must beat** for campaign B1. Same methodology, harness, and box as the
12B run (`gemma4-12b-vllm-sm120.md`).

- **Model:** `google/gemma-4-31B-it`, revision `b9ea41a2887d8607f594846523f94c6cc75ac8a4`,
  arch **`Gemma4ForConditionalGeneration`** (`model_type: gemma4`; 60 layers, hidden 5376,
  head_dim 256 / global_head_dim 512, 32 Q / 16 KV heads, sliding_window 1024,
  final_logit_softcapping 30.0). Model max ctx 262144; served at `--max-model-len 132096`
  (128k + 128 out + slack). TP-1.
- **Exact versions:** vLLM **0.25.1** (PyPI, `/workspace/venvs/vllm`); torch **2.11.0+cu130**
  (CUDA **13.0**); driver **580.82.07**; host CUDA toolkit **13.0** (V13.0.88).
- **Attention backend: `TRITON_ATTN` in all three configs** — forced by vLLM because Gemma4 has
  heterogeneous head dims (`head_dim=256`, `global_head_dim=512`) so FA4 is unavailable. Verbatim log:
  `Gemma4 model has heterogeneous head dimensions ... FA4 not available, forcing TRITON_ATTN backend.`
- CUDA graphs **ON** (`cudagraph_mode FULL_AND_PIECEWISE`; `Graph capturing finished in 12–14 secs,
  took 0.77–0.84 GiB` per config). Not `--enforce-eager`.
- Data: `perf-data/gemma4-31b-vllm-sm120.json`. Raw logs: `perf-data/vllm_sm120_logs_31b/`.

**No OOM, no failed points: all 3 configs × 7 contexts (1k…128k) completed** — but bf16 is tight
(see VRAM table); its KV pool holds only 1.45× a 132k request, so batch-1 was the *only* way bf16
reaches 128k on this card.

## Decode — TPOT ms/token (HEADLINE, lower is better)

| ctx    | bf16   | fp8    | fp8kv  |
|--------|--------|--------|--------|
| 1024   | 44.670 | 25.620 | 25.740 |
| 4096   | 45.200 | 26.160 | 26.410 |
| 16384  | 46.930 | 27.800 | 27.310 |
| 32768  | 49.140 | 29.860 | 28.890 |
| 65536  | 51.220 | 31.990 | 29.680 |
| 98304  | 53.280 | 34.090 | 30.770 |
| 131072 | 55.460 | 36.130 | 38.630 |

### Δ vs bf16 (decode TPOT, negative = faster)

| ctx    | fp8     | fp8kv   |
|--------|---------|---------|
| 1024   | −42.6 % | −42.4 % |
| 4096   | −42.1 % | −41.6 % |
| 16384  | −40.8 % | −41.8 % |
| 32768  | −39.2 % | −41.2 % |
| 65536  | −37.5 % | −42.1 % |
| 98304  | −36.0 % | −42.2 % |
| 131072 | −34.9 % | −30.3 % |

fp8 is a flat ~−40 % decode win on this dense 31B (bigger than the 12B's ~−30 % — the 31B is more
bandwidth-bound). fp8kv holds ~−42 % through 98k, **but see the 128k anomaly below**.

### The one anomaly: fp8kv decode at 128k

fp8kv TPOT jumps 30.77 → **38.63 ms** between 98k and 128k — the only point on the ladder where
fp8kv decodes *slower than plain fp8* (36.13 ms). This is steady-state, not noise: median 38.41,
P99 39.39 across the point's 3 prompts, and the serve log shows **zero preemptions**. It looks like
the TRITON_ATTN fp8-KV decode kernel crosses a regime boundary above ~100k ctx. Recorded as measured;
a real vLLM behavior plow gets to exploit.

## Decode — output throughput tok/s (higher is better)

| ctx    | bf16  | fp8   | fp8kv |
|--------|-------|-------|-------|
| 1024   | 22.39 | 39.03 | 38.85 |
| 4096   | 22.12 | 38.23 | 37.86 |
| 16384  | 21.31 | 35.97 | 36.62 |
| 32768  | 20.35 | 33.49 | 34.61 |
| 65536  | 19.52 | 31.26 | 33.69 |
| 98304  | 18.77 | 29.33 | 32.50 |
| 131072 | 18.03 | 27.68 | 25.89 |

## Prefill — TTFT ms (recorded; TTFT = prefill + first decode token)

| ctx    | bf16     | fp8      | fp8kv    |
|--------|----------|----------|----------|
| 1024   | 222.36   | 159.50   | 138.70   |
| 4096   | 705.67   | 510.16   | 414.99   |
| 16384  | 3450.74  | 2709.35  | 1952.98  |
| 32768  | 10730.22 | 5490.77  | 3504.26  |
| 65536  | 31095.86 | 20197.22 | 10364.40 |
| 98304  | 61241.08 | 53191.82 | 15366.65 |
| 131072 | 99564.73 | 89034.38 | 20636.40 |

Long-ctx prefill collapses on this model in bf16/fp8: **128k TTFT is 99.6 s bf16 / 89.0 s fp8** —
the 512-dim global-attention layers dominate quadratic prefill across 60 layers. **fp8kv cuts 128k
TTFT to 20.6 s (−79 % vs bf16, −77 % vs fp8)** — by far its largest win, dwarfing the same effect on
the 12B (−50 %).

## Prefill — tok/s (DERIVED, convention note)

`prefill_tok_s = input_len / (Mean_TTFT_ms/1000)`. **This is not a true prefill throughput** —
TTFT includes the first decode token + scheduling, so it under-reports; kept only for consistency
with every other served vLLM file in `perf-data/`.

| ctx    | bf16   | fp8    | fp8kv  |
|--------|--------|--------|--------|
| 1024   | 4605.1 | 6420.1 | 7382.8 |
| 4096   | 5804.4 | 8028.9 | 9870.1 |
| 16384  | 4748.0 | 6047.2 | 8389.2 |
| 32768  | 3053.8 | 5967.8 | 9350.9 |
| 65536  | 2107.5 | 3244.8 | 6323.2 |
| 98304  | 1605.2 | 1848.1 | 6397.2 |
| 131072 | 1316.5 | 1472.2 | 6351.5 |

## VRAM / KV headroom per config (util 0.90, constant across configs)

| config | weights (ckpt) | available KV | KV tokens | max-conc @132k | VRAM used | max ctx reached |
|--------|---------------|--------------|-----------|----------------|-----------|-----------------|
| bf16   | 58.25 GiB     | 24.87 GiB    | 191,862   | 1.45×          | 93321 / 97887 MiB | 131072 (swept) |
| fp8    | ~29.5 GiB     | 52.22 GiB    | 402,842   | 3.05×          | 93475 / 97887 MiB | 131072 (swept) |
| fp8kv  | ~29.5 GiB     | 52.22 GiB    | 805,704   | 6.10×          | 93471 / 97887 MiB | 131072 (swept) |

bf16 **fits** 128k at batch 1 (no OOM anywhere), but with only 1.45× concurrency headroom — the 96 GB
card is the reason bf16-128k works at all for a 58.25 GiB checkpoint. fp8 frees ~27 GiB of KV pool;
fp8kv doubles tokens/GiB on top (805,704 = exactly 2× 402,842).

## Correctness gate (per config, greedy chat)

| config | generated text |
|--------|----------------|
| bf16  | `The capital of France is Paris.` |
| fp8   | `The capital of France is Paris.` |
| fp8kv | `The capital of France is Paris.` |

All three coherent (`final_logit_softcapping: 30.0`, mixed sliding/full attention layout survive
fp8 and fp8-KV).

## Caveats

- **`prefill_tok_s` is a derived proxy, not measured** (see convention note above).
- **ctx=1024 TTFT** carries first-shape warm-up (torch.compile specialization / cudagraph capture for
  the smallest bench shape); the chat warm-up request precedes each sweep but the 1024 point still
  sees some of it. Decode TPOT is unaffected.
- **vLLM is not bit-exact** (torch.compile inductor + TRITON_ATTN + cudagraphs).
- Single run per point, 3 prompts each — no repeat-run error bars.
- fp8kv logs `Checkpoint does not provide a q scaling factor. Setting it to k_scale` — harmless here
  (only matters for FP8 flash-attn/flashinfer backends; this run is TRITON_ATTN).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set for all configs.

## Reproduce

```
export GPU_LEASE_TIMEOUT=7200
gpulease b1-31b env \
  QUANT=bf16 KVFP8=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL_DIR=/workspace/models/gemma-4-31B-it \
  OUTDIR=/root/plow/perf-data/vllm_sm120_logs_31b \
  PORT=8131 CTXS=1024,4096,16384,32768,65536,98304,131072 MAXLEN=132096 GPU_UTIL=0.90 \
  HEALTH_TIMEOUT=1800 \
  bash <scratch copy of perf-data/bench_vllm_cuda.sh with line-58 PATH fixed>
```

`QUANT=fp8 KVFP8=0` for fp8; `QUANT=fp8 KVFP8=1` for fp8kv. Same CTXS/MAXLEN/GPU_UTIL for all three.
Venv: `/workspace/venvs/vllm` (`vllm==0.25.1`). The only harness edit is line 58's PATH — see the
JSON `notes`.

## Bottom line (what plow must beat)

1. **Decode TPOT floor at batch 1, TP1, sm_120:** bf16 44.7→55.5 ms (1k→128k); fp8 25.6→36.1 ms;
   fp8kv 25.7→30.8 ms through 98k but **38.6 ms at 128k** (regresses past fp8 — beatable).
2. **Prefill/TTFT floor:** fp8kv at 128k is 20.6 s; bf16/fp8 are 99.6 s / 89.0 s — vLLM's dense-31B
   long-ctx prefill on sm_120 is weak, the clearest target on the board.
3. All configs serve out of the box; backend is **TRITON_ATTN** throughout; bf16 fits 128k at
   batch 1 only (1.45× concurrency headroom).
