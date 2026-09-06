# GPT-OSS-20B head-to-head via the OpenAI API (2026-09-06)

Same box (Sapphire Rapids 8c/16t, 58 GB), same bench (`bench.py --requests 8 --max-tokens 64 --fresh-prompts`),
one server at a time. plow: `plowrt serve --cpu-threads 16`, MXFP4 experts (checkpoint bytes verbatim), commit
897e117 (AMX grouped-expert prefill, ops 149/150). llama.cpp: `serve.sh gptoss` (gpt-oss-20b MXFP4 GGUF, 8 slots ×
8192 ctx). Raw: `plow/gptoss-amxpf-*.md`, `llamacpp/results-gptoss-fresh-*.md`. The earlier `llamacpp/results-gptoss.md`
and `plow/results-plow.md` were taken WITHOUT fresh prompts (llama-server's prefix cache made its c≥2 TTFT look
5–20× better than it is) and before the AMX prefill; keep them for history only.

TTFT / TPOT = mean ms. Lower is better. Bold = plow wins.

| workload | conc | in tok | plow TTFT | llama.cpp TTFT | plow TPOT | llama.cpp TPOT |
|---|---|---|---|---|---|---|
| chat_short | 1 | 32 | **576** | 748 | 45 | 41 |
| chat_short | 2 | 42 | **820** | 1625 | 73 | 65 |
| chat_short | 4 | 38 | **1358** | 3059 | 119 | 104 |
| chat_short | 8 | 34 | **2880** | 5260 | 200 | 177 |
| chat_long | 1 | 392 | **2035** | 4823 | **45** | 51 |
| chat_long | 2 | 439 | **2954** | 9520 | 89 | 81 |
| chat_long | 4 | 344 | **4001** | 14432 | 156 | 133 |
| chat_long | 8 | 423 | **9320** | 35666 | 326 | 215 |
| code | 1 | 344 | **1680** | 4212 | **47** | 50 |
| code | 2 | 357 | **2648** | 8780 | 81 | 73 |
| code | 4 | 383 | **3913** | 18762 | 156 | 123 |
| code | 8 | 337 | **7332** | 35120 | 268 | 207 |
| summarize | 1 | 1111 | **5374** | 10693 | **47** | 59 |
| summarize | 2 | 1124 | **6185** | 23777 | 142 | 113 |
| summarize | 4 | 1188 | **10883** | 57888 | 304 | 195 |
| summarize | 8 | 1136 | **23158** | 73138 | 507 | 430 |

Reading
* TTFT: plow wins every cell, 1.3× (32-token prompts) to 5× (long prompts at c≥2). Before the AMX grouped-expert
  prefill, plow's chat_long c=1 TTFT was 9067 ms (prefill ~40 tok/s); now 2035 (~200 tok/s).
* Decode: even at c=1 (45–47 vs 41–59; plow wins the long-context cells, loses the 32-token one by 4 ms); at
  c≥2 llama.cpp is 10–50 % better — plow's batched MoE decode runs every selected expert per row (rung 8 = up to
  64 expert passes/step) where llama.cpp batches the rows through each expert once.
* Next for decode: expert-deduplicated batched MoE decode (sort B·k slots by expert, one weight pass per expert,
  AMX x-as-B tiles as in the prefill kernels) — same fix the 26B needs.

## Batched decode grouped by expert (commit af83ce4, 09:1x)

Ops 147/148 at B >= 2 now sort the B·k slots by expert and dequantize each selected expert once for all
its rows (fused AVX-512 dequant-dot, M <= 8), with the slice partition weighted by per-expert work.
Fresh-prompt serve cells (`plow/gptoss-grpw-chat48.md`; the ungrouped build is `gptoss-amxpf-chat.md`):

| workload | conc | plow ungrouped TPOT | plow grouped TPOT | llama.cpp TPOT | plow grouped TTFT | llama.cpp TTFT |
|---|---|---|---|---|---|---|
| chat_short | 4 | 119 | 109 | 104 | 1305 | 3059 |
| chat_short | 8 | 200 | 178 | 177 | 3095 | 5260 |
| chat_long | 4 | 156 | 177 | 133 | 4273 | 14432 |
| chat_long | 8 | 326 | 294 | 215 | 9539 | 35666 |

Notes: an AMX-tile variant of the grouping (weights staged into A tiles) was 40 % slower (most experts
carry one row); an unweighted column partition was neutral (a slice owning a 4-row expert did ~2x the
work of its neighbours). Run-to-run TPOT spread at c >= 4 is ~±15 % (prefill of other slots interleaves
with decode), so chat_long c=4 is within noise. The remaining c >= 2 gap is prefill interleaving +
per-row dequant; the c=1 gap (45 vs 41 ms) is the bf16 dense weights (lm_head 1.16 GB + QKV/o ≈ 21 of
44 ms) — next lever: MXFP4 dense + head twin for GPT-OSS (~0.9 GB).

## MXFP4 dense projections + lm_head (commit 4df9a9b, 09:2x) — current plow numbers

`plowc --mxfp4` on GPT-OSS now reads q/k/v/o and lm_head from an MXFP4 twin
(`quantize_mxfp4.py <hf> /home/lava/models/gpt-oss-20b-mxfp4-dense model. --extra lm_head.weight`, 0.65 GB,
served with `PLOW_MXFP4_DIR=/home/lava/models/gpt-oss-20b-mxfp4-dense`); the bf16 dense weights were ~21 of
the 44 ms per decode token (lm_head alone 1.16 GB). Grouped batched MoE decode included. Raw: `plow/gptoss-mx4-*.md`.
llama.cpp = fresh-prompt re-run (`llamacpp/results-gptoss-fresh-*.md`). Bold = plow wins.

| workload | conc | plow TTFT | llama.cpp TTFT | plow TPOT | llama.cpp TPOT |
|---|---|---|---|---|---|
| chat_short | 1 | **602** | 748 | **33** | 41 |
| chat_short | 2 | **829** | 1625 | **61** | 65 |
| chat_short | 4 | **1369** | 3059 | 109 | 104 |
| chat_short | 8 | **2876** | 5260 | **174** | 177 |
| chat_long | 1 | **2070** | 4823 | **32** | 51 |
| chat_long | 2 | **2969** | 9520 | **75** | 81 |
| chat_long | 4 | **3837** | 14432 | 146 | 133 |
| chat_long | 8 | **10469** | 35666 | 285 | 215 |
| code | 1 | **1697** | 4212 | **33** | 50 |
| code | 2 | **2705** | 8780 | **68** | 73 |
| code | 4 | **3971** | 18762 | 148 | 123 |
| code | 8 | **7692** | 35120 | 243 | 207 |
| summarize | 1 | **5437** | 10693 | **33** | 59 |
| summarize | 2 | **6200** | 23777 | 128 | 113 |
| summarize | 4 | **11179** | 57888 | 283 | 195 |
| summarize | 8 | **24319** | 73138 | 466 | 430 |

Reading: plow wins TTFT everywhere (1.2–5×) and decode at c=1 (33 vs 41–59, 1.25–1.8×) and c=2; at c≥4 it is
at parity on short prompts and 10–45 % behind on long prompts, where other slots' whole-prompt prefills stall
the decode steps (TPOT p90 ≫ p50). Next: chunked prefill (PLOW_CPU_PF_CHUNK) by default at c≥2, then the
dense MXFP4 GEMV at batch ≥ 5 (AMX x_gemv_mxfp4 path) and per-row dequant cost in the grouped MoE decode.
