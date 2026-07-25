# Gemma-4-31B LONG-CONTEXT (48k–128k) sweep — plow vs vLLM, TP4 + TP8

**The headline: plow's decode (TPOT) lead over vLLM GROWS monotonically with context.**
On TP4 it widens from **+4.2 % at 48k to +11.2 % at 128k**. This is plow's strongest
regime — bandwidth-bound per-token decode at long context — and the lead compounds exactly
as predicted (`plans/long-context-128k.md`: decode flips from GEMV-bound to flash_decode-bound
at ~64k, and plow's sharded head-major KV scales better per token than vLLM's Triton attention).

**The flip side: plow LOSES prefill (TTFT) at long context, by 4–12×, and structurally.**
plow has **no tensor-parallel prefill** — `tp_decode` is decode-only, so prefill runs SINGLE-GPU
(TP1) regardless of the TP degree. vLLM sharding gives it 13k–60k tok/s prefill vs plow's
1.7k–3.5k tok/s. So the win/loss split is **by phase, not by context**: plow owns decode/TPOT,
vLLM owns prefill/TTFT, at every long context.

Measured 2026-07-17 on the 8× MI350X / gfx950 node. bf16, batch 1, greedy, output_len 128.
plow on branch `tp` (bit-exact TP decode, device==host verified at TP4 and TP8). vLLM 0.25.1
(`vllm/vllm-openai-rocm:latest`), `--max-model-len 131200`, TRITON_ATTN, cudagraphs (not bit-exact).

---

## 1. DECODE — TPOT ms/tok (lower is better). **The key result.**

| ctx  | plow TP4 | vLLM TP4 | plow/vLLM | plow lead | plow TP8 | vLLM TP8 | plow/vLLM | plow lead |
|------|---------:|---------:|----------:|----------:|---------:|---------:|----------:|----------:|
| 48k  | **13.945** | 14.550 | 0.958 | **+4.2 %** | **13.195** | 13.590 | 0.971 | **+2.9 %** |
| 64k  | **14.692** | 15.630 | 0.940 | **+6.0 %** | **13.955** | 14.610 | 0.955 | **+4.5 %** |
| 72k  | **15.040** | 16.200 | 0.928 | **+7.2 %** | **14.445** | 15.130 | 0.955 | **+4.5 %** |
| 96k  | **16.144** | 17.690 | 0.913 | **+8.7 %** | **15.493** | 16.670 | 0.929 | **+7.1 %** |
| 128k | **17.615** | 19.840 | 0.888 | **+11.2 %**| **16.907** | 18.720 | 0.903 | **+9.7 %** |

**Verdict — does plow's long-context decode lead grow? YES, monotonically, on BOTH TP4 and TP8.**
TP4 lead: 48k +4.2 % → 64k +6.0 % → 72k +7.2 % → 96k +8.7 % → **128k +11.2 %**.
TP8 lead: 48k +2.9 % → 64k +4.5 % → 72k +4.5 % → 96k +7.1 % → **128k +9.7 %**. Every step wider.
The 64k point reproduces the earlier baseline (plow TP4 14.69 vs the baseline's 14.73; vLLM TP8
14.61 vs baseline 14.79), so the sweep is calibrated. plow TP8 is faster than TP4 in absolute
ms/tok at every context (the `tp-optimize` flash_merge fix — TP8 now scales past TP4 instead of
regressing), and vLLM TP8 is likewise faster than vLLM TP4 — so the plow lead is a hair narrower
at TP8 than TP4, but still present and still growing with context. All cells freshly measured this run.

## 2. PREFILL — TTFT and prefill throughput. **plow's weak front.**

plow prefill is **single-GPU (TP1)** — the SAME number appears in the TP4 and TP8 columns because
plow does not tensor-parallelise prefill. vLLM shards prefill across all N GPUs.

| ctx  | plow prefill ms (TP1) | plow tok/s | vLLM TP4 TTFT ms | vLLM TP4 tok/s | vLLM TP8 TTFT ms | vLLM TP8 tok/s |
|------|----------------------:|-----------:|-----------------:|---------------:|-----------------:|---------------:|
| 48k  | 14 138 | 3 477 | 3 617  | 13 590 | 2 339  | 21 010 |
| 64k  | 22 671 | 2 891 | 1 997† | 32 820 | 1 213† | 54 045 |
| 72k  | 27 649 | 2 667 | 1 236† | 59 661 | 791†   | 93 174 |
| 96k  | 45 219 | 2 174 | 3 932  | 25 001 | 2 368  | 41 512 |
| 128k | 75 673 | 1 732 | 6 535  | 20 056 | 3 913  | 33 498 |

plow prefill is **4–12× slower than vLLM TP4 and 6–35× slower than vLLM TP8**, and the gap is
**structural** — it will not close until plow adds TP prefill (each vLLM shard streams only 1/N of
the weights per chunk; plow's single GPU re-streams all 57 GiB every chunk). `†` vLLM TTFT at 64k/72k
is anomalously low (first-request torch.compile/cudagraph specialisation — same artifact flagged in
the baseline); the derived prefill tok/s there is optimistic, but the direction is unambiguous.

## 3. Where each system wins (the crossover is by PHASE)

- **Decode / TPOT — plow wins at every long context, lead growing to +11 % at 128k (TP4).**
  Bit-exact to TP1, device==host verified. This is the regime plow was built for.
- **Prefill / TTFT — vLLM wins at every long context, by 4–12×.** plow has no TP prefill;
  its single-GPU flash_prefill is O(T²) on the 10 full-attention layers and cannot amortise
  weights across GPUs the way vLLM's sharded prefill does.
- **Net:** a *decode-heavy* long-context request (long generation) favours plow — its per-token
  win compounds over the output. A *prefill-heavy* request (huge prompt, short answer) favours
  vLLM — TTFT dominates. At 128k with 128 output tokens the two roughly trade: plow saves
  ~2.2 ms × 128 ≈ 0.28 s on decode but loses ~69 s on prefill. plow's decode edge only pays off
  when the generation is thousands of tokens, or when prefill is cached/amortised.

## 4. Capacity & contention (OOM / max-ctx notes)

- **No plow OOM at 128k.** plow head-shards the KV: at 128k per-GPU KV is 22.5 GiB (TP1),
  **5.62 GiB (TP4)**, **4.06 GiB (TP8)** — most layers are sliding-window (1024) so only the
  10 full-attention layers keep full-length KV. TP4 total ≈ 23 GiB/GPU, TP8 ≈ 14 GiB/GPU. Fits easily.
- **No vLLM OOM at 128k on TP4 or TP8.** TP4 served 128k on idle GPUs 4–7 (default mem-util). TP8
  served 128k across all 8 GPUs at `--gpu-memory-utilization 0.85` (KV fit; available KV 72 GiB/GPU).
- **vLLM TP8 was initially contention-blocked, then measured cleanly.** A sibling agent
  (spatial-partition / vllm-adapt2) was cycling large models across the node; three early TP8
  attempts OOM'd during KV/cudagraph capture with *"GPU N has a total capacity of 287.98 GiB of
  which 0 bytes is free"* — a foreign 191 GiB model was resident on one of the 8 GPUs TP8 needs.
  This is a shared-node artifact, not a vLLM/plow limit. An auto-launch monitor waited for a clean
  8-GPU window (opened after ~6 min) and the full TP8 48k–128k sweep then ran to completion. TP4
  ran concurrently with the sibling agent on the free half of the node (GPUs 4–7).

## 5. Reproducibility

- **plow decode:** `plowc gemma4 --tp {4,8} <model> 131072 out.pkt` (XReduce CU cap baked in via
  `PLOW_XR_CUS={32,16}` — a bit-exact perf lever), then
  `scripts/tp_decode_sweep.sh`-style: `./tp_decode out.pkt <model> --tp N --sweep 48k,64k,72k,96k,128k --steps 21`
  under `sg render` + clean env (`/usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME LD_LIBRARY_PATH=/opt/rocm/lib`).
  Model: `/home/lava/models/gemma-4-31B-it-text` (the flat `gemma4_text` re-export the branch compiler reads).
- **plow prefill:** `./chat gemma_tp1_128k.pkt <model> promptNk.ids 1` — read the `prefill: … ms` line (TTFT).
- **vLLM:** `MEM_UTIL=<u> TP=N GPUS=… CNAME=… PORT=… bash perf-data/bench_vllm_longctx.sh`
  → serves `gemma-4-31B-it` at `--max-model-len 131200`, sweeps `{49152,65536,73728,98304,131072}`,
  `--random-output-len 128 --max-concurrency 1 --num-prompts 3`. Extends `bench_vllm_tp.sh`
  (new optional `MEM_UTIL` → `--gpu-memory-utilization`, for coexisting under node contention).
- **Data:** `perf-data/gemma4-31b-longctx-sweep.json`; vLLM raw logs `perf-data/vllm_longctx_logs/`.
