# fp8 DECODE — final sweep: plow-fp8 vs vLLM (MI350X / gfx950)

**Date:** 2026-07-17 · **branch:** `fp8-sweep` (base `fp8-merge`, the provisional 1.23× fp8 decode kernel) ·
**metric:** decode TPOT ms/tok, batch 1, greedy (lower is better).

plow numbers: `runtime/tests/tp_decode.c --sweep` (1-token decode, median of 15–21), fp8 weight-only
(w8a16 per-channel e4m3) and optional e4m3 KV, GLOBAL-QUEUE decode. **fp8 support was wired into the
`tp_decode` TP harness for this campaign** (weight+KV `fp8/` binding, sharded col/row, fp8/fp8kv decode
object selection) — see "Harness changes" below. vLLM: committed bf16 matched-TP grid
(`decode-only-sweep.md`) + fresh 0.25.1 `--quantization fp8 --kv-cache-dtype fp8` TP4/TP8 runs.

---

## TL;DR verdict

| Model | Short ctx (≤~8k) | Long ctx (≥32k) | Overall |
|-------|------------------|-----------------|---------|
| **Gemma-4-31B** | behind vLLM (fixed straggler+attn gap fp8 can't touch) | **WINS at matched TP** vs both vLLM bf16 **and** vLLM fp8kv | plow-fp8 differentiator: extends the bf16 long-ctx lead |
| **Llama-3.1-8B** | loses (~1.5×) | **loses worse (~3×)** at TP4 | vLLM wins everywhere |
| **Qwen3-4B** | loses (~1.9×) | n/a (TP1 only) | vLLM wins everywhere |

**One line:** fp8 decode is a **big-model long-context** lever. On **Gemma-31B** plow-fp8 crosses vLLM at
~8k and wins the long-ctx / high-TP regime outright (up to **−15 %** at 128k TP4), and its output is
**bit-identical to bf16**. On the **small models** (8B, 4B) plow-fp8 loses at every context — the weight
stream is too small to amortize plow's fixed per-token decode overhead, and TP sharding only makes it
worse. This matches the campaign premise: *short-ctx stays behind; fp8 wins long-ctx, chiefly on the big
model.*

---

## Gemma-4-31B — the win

### TP1 (ctx 1k–32k), plow-fp8 vs vLLM bf16
| ctx | plow fp8-weight | plow fp8-KV | vLLM bf16 | best-plow / vLLM |
|----:|----:|----:|----:|:--:|
| 1k  | 14.92 | 15.92 | 13.69 | 1.09 (lose) |
| 4k  | 15.02 | 16.20 | 14.63 | 1.03 (lose) |
| 8k  | 15.40 | 16.20 | 15.94 | **0.97 (win)** |
| 16k | 15.90 | 16.47 | 16.70 | **0.95 (win)** |
| 32k | 16.93 | 16.86 | 19.50 | **0.87 (win)** |

### Long context at matched TP — the headline
| ctx | TP | plow fp8-weight | vLLM bf16 | vLLM fp8kv | plow vs bf16 | plow vs fp8kv |
|----:|:--:|----:|----:|----:|:--:|:--:|
| 64k  | 4 | **14.14** | 15.61 | 15.86 | **−9 %** | **−11 %** |
| 128k | 4 | **16.93** | 19.84 | 19.70 | **−15 %** | **−14 %** |
| 64k  | 8 | **13.78** | 14.79 | 15.47 | **−7 %** | **−11 %** |
| 128k | 8 | **16.70** | 18.72 | 19.38 | **−11 %** | **−14 %** |

plow-fp8 beats vLLM at **matched TP** at every long-ctx point, against **both** vLLM configs. fp8
**extends plow's own bf16 long-ctx lead**: at 128k TP4 plow-bf16 was already 0.915× vLLM
(`decode-only-sweep.md`); fp8-weight takes it to **0.854×**.

### fp8-KV is NOT the Gemma lever (surprise, but physical)
Gemma is **sliding-window** attention — 50 of 60 layers are window-capped, so the decode KV stream barely
grows with ctx and there is little for fp8-KV to halve. fp8-KV **loses** to fp8-weight at short/mid ctx
(TP1 1k 15.92 vs 14.92), only **ties at ~32k** (16.86 vs 16.93), and at 64k/128k TP4 the heavier fp8-KV
decode object (180 vs 164 VGPR) stays **behind** fp8-weight (15.48 vs 14.14; 17.35 vs 16.93). **For Gemma
the winning config is fp8-weight + TP sharding.** (fp8-KV crossover exists but is beyond the useful range.)

### vLLM fp8 on Gemma/gfx950 is broken-ish
vLLM `--quantization fp8 --kv-cache-dtype fp8` (0.25.1) gives **no decode speedup** over bf16
(sometimes slightly slower) and is **numerically flaky**: the "capital of France" sanity returns
`our our our our` (garbage) at TP4, coherent at TP8. plow-fp8 by contrast is **bit-identical to bf16**.

---

## Llama-3.1-8B — plow loses everywhere

| ctx | TP | plow fp8-weight | plow fp8-KV | vLLM bf16 | best-plow / vLLM |
|----:|:--:|----:|----:|----:|:--:|
| 1k  | 1 | 6.05 | 6.35 | 3.95 | 1.53 |
| 8k  | 1 | 6.69 | 6.79 | 4.34 | 1.54 |
| 32k | 1 | 7.97 | 7.57 | 4.93 | 1.54 |
| 64k  | 4 | 9.01 | 8.67 | 3.14 | **2.76** |
| 128k | 4 | 11.74 | 10.71 | 3.33 | **3.22** |

TP1 ~1.5× slower; at TP4 the gap **widens to ~3×** because an 8B weight shard vanishes under TP while
plow's fixed per-token overhead (straggler + flash) does **not** shard, and vLLM's TP4 decode reaches
3.1–3.3 ms/tok. **fp8-KV does help plow here** (Llama is full-attention, big KV): 128k TP4 10.71 vs 11.74
(−9 %), the exact opposite of Gemma — but nowhere near enough to reach vLLM.

## Qwen3-4B — plow loses everywhere (TP1)

| ctx | plow fp8-weight | vLLM bf16 | ratio |
|----:|----:|----:|:--:|
| 1k  | 5.81 | 3.15 | 1.84 |
| 8k  | 6.49 | 3.39 | 1.91 |
| 32k | 7.98 | 4.08 | 1.96 |

A 4B model's ~11 GiB fp8 weight stream is far too small to amortize plow's fixed decode overhead; fp8
barely moves TPOT. vLLM's graph-captured decode is structurally leaner on small models.

---

## Why: the fp8-KV crossover is model-shaped

fp8-KV halves the decode KV **read**, so it pays only when the KV stream is a large share of the token and
the extra dequant/heavier object is worth it:

| model | attention | fp8-KV verdict |
|-------|-----------|----------------|
| Gemma-4-31B | sliding-window (KV capped) | **loses / marginal** — use fp8-weight |
| Llama-3.1-8B | full (KV ∝ ctx) | **wins long-ctx** (128k TP4 −9 %) but plow still behind vLLM |

This is consistent with P1's finding that plow's fp8-KV is a narrow long-ctx-only lever, and refines it:
it is a lever for **full-attention** models, a dead weight for **sliding-window** ones.

---

## Caveats / honesty notes
- **plow "full model" residency:** the decode-only pkt still declares & binds the bf16 prefill weights
  alongside the fp8 decode twins (84.5 GiB Gemma), so TP1 runs load both; timings are decode-only and
  unaffected, but this is why TP1 128k is routed to TP4/TP8.
- **plow TP>1 is real tensor-parallel** (sharded weights + XReduce all-reduce), not DP replicas.
  Value-correctness of the fp8 shards was gated at TP1 (full copy, bit-identical to bf16); TP>1 numbers are
  byte-correct-sharded **timing** (the sweep primes synthetic KV, so values don't affect TPOT).
- **vLLM fp8 baseline** used `vllm/vllm-openai-rocm:latest` (0.25.1); vLLM fp8 does **not** run on the
  older `rocm/vllm:latest` (0.11.2) docker (`vllm-fp8-baseline.md`), and its 0.25.1 output on Gemma is
  flaky (see above). The reproducible, coherent vLLM baseline is **bf16**.
- **ctx labels:** vLLM Llama "128k" = input 130048 (+128 out) to fit the 131072 window; plow "128k" =
  position 131071. Both are the 128k regime.
- No OOM hit in any run; GPUs 0–7 all idle when borrowed for TP8.

## Data
- `perf-data/fp8-decode-gemma.json`, `fp8-decode-llama.json`, `fp8-decode-qwen.json` (per-model, all points).
- Raw plow sweep logs: `/tmp/fp8sweep_gemma/out_*.txt`, `/tmp/fp8sweep_hd128/out_*.txt`.
- vLLM fp8 CSVs: `/tmp/vllm_gemma_fp8kv_tp{4,8}/`, `/tmp/vllm_llama_tp4/`; committed bf16 grid in
  `decode-only-sweep.md` + `vllm_longctx_logs/`.

## Harness changes (this campaign)
- `runtime/tests/tp_decode.c`: wired fp8 into the TP decode harness — open `PLOW_FP8_DIR`, bind `fp8/`
  weight+scale tensors (sharded: q/k/v/gate/up col, o/down row, scales replicated where output-full),
  select `interp_decode_fp8[kv]_gq.elf` from the pkt's `GEMV_FP8`/`FLASH_DECODE_FP8` ops.
- `scripts/build_gfx950.sh`: added the Gemma fp8-KV decode object build (`PLOW_FP8_KV=1`) + register-cliff
  check (180 VGPR, occ-2, 0 spill). (HD128 Llama/Qwen already had it in `build_gfx950_qwen.sh`.)
- `perf-data/bench_vllm_tp.sh`: added `QUANT=fp8` / `KVFP8=1` knobs for the matched-TP vLLM fp8kv runs.
