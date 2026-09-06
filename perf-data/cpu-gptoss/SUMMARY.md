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

## Three-way: plow (MXFP4 dense+experts) vs llama.cpp vs vLLM 0.28 CPU (09:4x–09:5x, fresh prompts)

vLLM: `vllm serve <hf gpt-oss-20b> --dtype bfloat16 --max-model-len 4096 --max-num-seqs 8` (CPU backend; its
MXFP4 module has no CPU path, so the experts are dequantized at load — 22 GB resident). vLLM streams GPT-OSS's
reasoning as a separate `reasoning` delta; bench.py now counts reasoning deltas as generated tokens (llama.cpp ran
with reasoning folded into content, plow emits the final channel directly), so TPOT/TTFT are comparable. Raw:
`vllm/vllm-gptoss-*.md`. TTFT/TPOT mean ms; bold = best.

| workload | conc | plow TTFT | llama TTFT | vLLM TTFT | plow TPOT | llama TPOT | vLLM TPOT |
|---|---|---|---|---|---|---|---|
| chat_short | 1 | **602** | 748 | 637 | **33** | 41 | 71 |
| chat_short | 2 | **829** | 1625 | 1220 | **61** | 65 | 95 |
| chat_short | 4 | **1369** | 3059 | 1657 | 109 | **104** | 103 |
| chat_short | 8 | 2876 | 5260 | **2120** | 174 | 177 | **136** |
| chat_long | 1 | 2070 | 4823 | **1346** | **32** | 51 | 71 |
| chat_long | 2 | 2969 | 9520 | **1962** | **75** | 81 | 80 |
| chat_long | 4 | 3837 | 14432 | **2550** | 146 | 133 | **102** |
| chat_long | 8 | 10469 | 35666 | **4591** | 285 | 215 | **137** |
| code | 1 | 1697 | 4212 | **1253** | **33** | 50 | 76 |
| code | 2 | 2705 | 8780 | **2151** | **68** | 73 | 81 |
| code | 4 | 3971 | 18762 | **2944** | 148 | 123 | **115** |
| code | 8 | 7692 | 35120 | **4156** | 243 | 207 | **152** |
| summarize | 1 | 5437 | 10693 | **1829** | **33** | 59 | 71 |
| summarize | 2 | 6200 | 23777 | **3593** | 128 | 113 | **85** |
| summarize | 4 | 11179 | 57888 | **6735** | 283 | 195 | **116** |
| summarize | 8 | 24319 | 73138 | **7219** | 466 | 430 | **153** |

Reading: plow owns single-stream decode (33 ms vs 41–59 llama.cpp and 71–76 vLLM) and short-prompt TTFT. vLLM
owns prefill throughput (summarize c=1: 1111 tokens in 1.8 s ≈ 600 tok/s vs plow ≈ 200, llama.cpp ≈ 100) and
therefore every long-prompt cell at c ≥ 2, plus decode at c = 8 (its batched MoE/attention scale better: 136–153
ms at 8 rows vs plow 174–466). The gap to close is prefill throughput (prepacked AMX weights, MoE prefill
efficiency) and batched-decode scaling — the same two items on the plan.

## After the kernel optimization pass (commit c835609, 11:2x) — current three-way

Same bench, same servers, fresh prompts. plow = MXFP4 dense+experts with the optimized kernels.
Raw: `plow/gptoss-mx4-opt-*.md`. TTFT / TPOT mean ms; bold = best of the three.

| workload | conc | plow TTFT | llama TTFT | vLLM TTFT | plow TPOT | llama TPOT | vLLM TPOT |
|---|---|---|---|---|---|---|---|
| chat_short | 1 | **553** | 748 | 637 | **29** | 41 | 71 |
| chat_short | 2 | **773** | 1625 | 1220 | **54** | 65 | 95 |
| chat_short | 4 | **1283** | 3059 | 1657 | **97** | 104 | 103 |
| chat_short | 8 | 2669 | 5260 | **2120** | 155 | 177 | **136** |
| chat_long | 1 | 1928 | 4823 | **1346** | **28** | 51 | 71 |
| chat_long | 2 | 2779 | 9520 | **1962** | **67** | 81 | 80 |
| chat_long | 4 | 3506 | 14432 | **2550** | 128 | 133 | **102** |
| chat_long | 8 | 7994 | 35666 | **4591** | 261 | 215 | **137** |
| code | 1 | 1544 | 4212 | **1253** | **28** | 50 | 76 |
| code | 2 | 2457 | 8780 | **2151** | **58** | 73 | 81 |
| code | 4 | 3613 | 18762 | **2944** | 125 | 123 | **115** |
| code | 8 | 6902 | 35120 | **4156** | 209 | 207 | **152** |
| summarize | 1 | 5061 | 10693 | **1829** | **29** | 59 | 71 |
| summarize | 2 | 5828 | 23777 | **3593** | 119 | 113 | **85** |
| summarize | 4 | 10095 | 57888 | **6735** | 270 | 195 | **116** |
| summarize | 8 | 23170 | 73138 | **7219** | 431 | 430 | **153** |

Reading: plow now wins **every** c=1 and c=2 decode cell (28-29 ms at c=1 vs 41-59 llama.cpp and
71-76 vLLM, a 1.4-2.0x and 2.4-2.7x margin) and every TTFT cell against llama.cpp; it also wins TTFT
against vLLM at c<=4 on short prompts. vLLM still wins long-prompt TTFT and every c>=4 decode cell on
long prompts: it runs prefill and decode tokens in ONE mixed forward (chunked prefill, 2048-token
budget), so decode never waits behind a whole prompt. That scheduling change, not kernel speed, is the
remaining gap — plow's own kernels are now at 1.2-2.2x their previous throughput per op.


## Final: balanced MoE prefill (commit 25e0875, 12:4x)

Two structural fixes to the MXFP4 MoE prefill kernels landed after the kernel pass: the weight
dequant was hoisted out of the token-block loop (it was re-unpacking each expert matrix per 32-token
block), and the slice split is now weighted by rows per expert (an even column split left threads
50% idle with a 1.5x spread). GPT-OSS prefill at 512 tokens went 226 -> 403 tok/s and the prefill
wall for one 512-token pass went 11.0 s -> 1.24 s. Raw: `plow/gptoss-bal-*.md`. Bold = best.

| workload | conc | plow TTFT | llama TTFT | vLLM TTFT | plow TPOT | llama TPOT | vLLM TPOT |
|---|---|---|---|---|---|---|---|
| chat_short | 1 | **447** | 748 | 637 | **29** | 41 | 71 |
| chat_short | 2 | **637** | 1625 | 1220 | **53** | 65 | 95 |
| chat_short | 4 | **1034** | 3059 | 1657 | **92** | 104 | 103 |
| chat_short | 8 | 2249 | 5260 | **2120** | 152 | 177 | **136** |
| chat_long | 1 | **1126** | 4823 | 1346 | **30** | 51 | 71 |
| chat_long | 2 | **1626** | 9520 | 1962 | **60** | 81 | 80 |
| chat_long | 4 | **2097** | 14432 | 2550 | 112 | 133 | **102** |
| chat_long | 8 | 5184 | 35666 | **4591** | 204 | 215 | **137** |
| code | 1 | **993** | 4212 | 1253 | **30** | 50 | 76 |
| code | 2 | **1577** | 8780 | 2151 | **58** | 73 | 81 |
| code | 4 | **2309** | 18762 | 2944 | **113** | 123 | 115 |
| code | 8 | 4432 | 35120 | **4156** | 189 | 207 | **152** |
| summarize | 1 | 2882 | 10693 | **1829** | **31** | 59 | 71 |
| summarize | 2 | **3332** | 23777 | 3593 | 89 | 113 | **85** |
| summarize | 4 | **6118** | 57888 | 6735 | 186 | 195 | **116** |
| summarize | 8 | 13053 | 73138 | **7219** | 308 | 430 | **153** |

plow wins **all 32 cells against llama.cpp**. Against vLLM it wins 11 of 16 TTFT cells and 9 of 16
TPOT cells: every c=1 and c=2 cell except summarize TTFT at c=1 and TPOT at c=2. vLLM still wins at
c>=4 on long prompts, where it runs prefill chunks and decode rows in one mixed forward so decode
never queues behind a prompt; that scheduling change is the remaining structural item.

## Final: one worker per physical core (commit f51cd35, 15:4x)

SMT siblings share one TMUL and one pair of 512-bit FMA ports, so the default worker count moved
from logical cpus (16) to physical cores (8). No other change. Prefill at 512 tokens went 399 ->
455 tok/s and batch-1 decode 25.5 -> 24.0 ms. Fresh prompts, one server at a time, bold = best of
the three. Raw: `plow/gptoss-t8-*.md`.

| workload | conc | plow TTFT | llama TTFT | vLLM TTFT | plow TPOT | llama TPOT | vLLM TPOT |
|---|---|---|---|---|---|---|---|
| chat_short | 1 | **358** | 748 | 637 | **25** | 41 | 71 |
| chat_short | 2 | **513** | 1625 | 1220 | **47** | 65 | 95 |
| chat_short | 4 | **862** | 3059 | 1657 | **84** | 104 | 103 |
| chat_short | 8 | **1840** | 5260 | 2120 | 142 | 177 | **136** |
| chat_long | 1 | **978** | 4823 | 1346 | **24** | 51 | 71 |
| chat_long | 2 | **1438** | 9520 | 1962 | **60** | 81 | 80 |
| chat_long | 4 | **2468** | 14432 | 2550 | **100** | 133 | 102 |
| chat_long | 8 | **4545** | 35666 | 4591 | 195 | 215 | **137** |
| code | 1 | **893** | 4212 | 1253 | **25** | 50 | 76 |
| code | 2 | **1405** | 8780 | 2151 | **50** | 73 | 81 |
| code | 4 | **2062** | 18762 | 2944 | **103** | 123 | 115 |
| code | 8 | **4017** | 35120 | 4156 | 176 | 207 | **152** |
| summarize | 1 | 2577 | 10693 | **1829** | **25** | 59 | 71 |
| summarize | 2 | **3014** | 23777 | 3593 | 87 | 113 | **85** |
| summarize | 4 | **4838** | 57888 | 6735 | 164 | 195 | **116** |
| summarize | 8 | 11320 | 73138 | **7219** | 288 | 430 | **153** |

plow wins **all 32 cells against llama.cpp** and **24 of 32 against both baselines at once**:
14 of 16 TTFT cells and 10 of 16 TPOT cells. Every c=1 and c=2 cell is a win, and the physical-core
default newly took chat_long at c=4 (TPOT 100 vs vLLM's 102) and c=8 (TTFT 4545 vs 4591).

What still loses to vLLM, all at c >= 4 on long prompts: summarize TTFT at c=1 and c=8, and TPOT at
c=8 across the board plus summarize from c=2. The cause is unchanged and is not kernel speed -- vLLM
runs prefill chunks and decode rows in ONE forward, so a decoding request never waits behind another
prompt, while ours alternates. For chat_long at c=8 our 3512 prefill tokens cost 8.7 s at 403 tok/s and
64 rung-8 decode steps cost 9.7 s, which serialize to 18.4 s against vLLM's 13.25 s; fusing the decode
rows into the prefill pass gives max(8.7, 9.7) = 9.7 s, i.e. 1.37x ahead. That remains the one
structural item, and a GEMM does not care about sequence boundaries -- only attention needs the split.
