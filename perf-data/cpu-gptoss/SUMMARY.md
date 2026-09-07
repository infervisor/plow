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


## Experimental int16 VNNI MXFP4 decode (commit 53d4f3d, opt-in)

`PLOW_MXFP4_INT16=1` converts MXFP4 blocks to int16 and uses VNNI for decode. It remains
off by default. Fresh-prompt results; TTFT / TPOT mean ms:

| workload | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| chat_short | 350 / 22 | 485 / 41 | 788 / 64 | 1710 / 110 |
| chat_long | 960 / 22 | 1375 / 53 | 2115 / 84 | 4329 / 171 |
| code | 811 / 23 | 1286 / 42 | 1872 / 81 | 3667 / 149 |
| summarize | 2333 / 23 | 2721 / 76 | 4464 / 139 | 10752 / 246 |

The opt-in path improves default MXFP4 decode from 24-25 ms to 22-23 ms at c=1 and
from 142-288 ms to 110-246 ms at c=8. Sampled outputs remained coherent in this campaign.

## After the MoE prefill epilogue and the flash-decode GQA fold (0d07dfa + 55f9e7f, 22:2x)

Two kernel changes since the physical-core table above, both bit-identical:

* `0d07dfa` transposes the 32x32 accumulator in both MXFP4 MoE prefill kernels so each token's 32
  outputs leave as one vector store instead of up to 1024 scattered 2-byte stores, and gates
  `dot_block`'s weight prefetch at the call site. The prefetch was walking 16 weight rows ahead into a
  dequantized strip already resident in L2, which was pure overhead on the MXFP4 path; that gating was
  worth more than the transpose. Prefill at 512 tokens 455 -> 470 tok/s.
* `55f9e7f` folds the GQA head groups onto one K/V pass in flash decode. With `gqa=8` the four head
  groups behind each kv head were separate work items re-reading the same K and V rows, so the kernel
  was loading 868 MB to consume 241 MB. It was never bandwidth-starved -- it ran at ~73 GB/s into the
  cores, essentially at this box's roofline -- it was moving 4x the bytes it needed. FLASH_DECODE went
  16.61 -> 7.91 ms/thread and the batch-8, 1100-context decode step 118.0 -> 105.3 ms.

Fresh prompts, one server at a time, 8 slots. TTFT / TPOT mean ms; bold = best of the three.

| workload | conc | plow TTFT | llama TTFT | vLLM TTFT | plow TPOT | llama TPOT | vLLM TPOT |
|---|---|---|---|---|---|---|---|
| chat_short | 1 | **325** | 748 | 637 | **23** | 41 | 71 |
| chat_short | 2 | **388** | 1625 | 1220 | **46** | 65 | 95 |
| chat_short | 4 | **727** | 3059 | 1657 | **76** | 104 | 103 |
| chat_short | 8 | **1782** | 5260 | 2120 | **130** | 177 | 136 |
| chat_long | 1 | **858** | 4823 | 1346 | **25** | 51 | 71 |
| chat_long | 2 | **1294** | 9520 | 1962 | **53** | 81 | 80 |
| chat_long | 4 | **1580** | 14432 | 2550 | **92** | 133 | 102 |
| chat_long | 8 | **3692** | 35666 | 4591 | 182 | 215 | **137** |
| code | 1 | **723** | 4212 | 1253 | **25** | 50 | 76 |
| code | 2 | **1369** | 8780 | 2151 | **51** | 73 | 81 |
| code | 4 | **1668** | 18762 | 2944 | **94** | 123 | 115 |
| code | 8 | **3773** | 35120 | 4156 | 163 | 207 | **152** |
| summarize | 1 | 2469 | 10693 | **1829** | **26** | 59 | 71 |
| summarize | 2 | **2427** | 23777 | 3593 | **78** | 113 | 85 |
| summarize | 4 | **3073** | 57888 | 6735 | 163 | 195 | **116** |
| summarize | 8 | 10037 | 73138 | **7219** | 279 | 430 | **153** |

plow wins **all 32 cells against llama.cpp** and **26 of 32 against vLLM**, up from 24: 14 of 16 TTFT
and 12 of 16 TPOT. The two newly won cells are chat_short TPOT at c=8 (130 vs 136) and summarize TPOT
at c=2 (78 vs 85). Every TTFT cell improved, several by a third (chat_long c=4 2468 -> 1580, summarize
c=4 4838 -> 3073).

The six remaining losses are the same structural item, not kernel speed. Decode is now at the memory
wall: at batch 8 the MoE ops move 4.31 GB and 2.15 GB per step at 100 and 98 GB/s against a measured
84-115 GB/s roofline, so there is no headroom left in them. `GEMV_MXFP4` is compute-bound rather than
bandwidth-bound -- its busy time scales with batch while its weight bytes do not, fitting to ~3.4 ms
fixed plus ~2.8 ms per batch row -- so the 25 GB/s figure it appears to run at is an artifact of
dividing mostly-MAC time by constant bytes. What is left is that vLLM runs prefill chunks and decode
rows in one forward, so a decoding request never queues behind another prompt.

Two notes for later. Reducing decode further needs fewer bytes, not faster kernels, since MXFP4 is
already 4-bit. And the `FA_GF=2` head pairing computes `hkv = h0 / gqa` for both heads of a pair,
which is wrong for MHA (`gqa == 1`); no local checkpoint is MHA (Gemma-4 is gqa 2 and 4, GPT-OSS is 8)
so it is latent, and the GQA fold guards itself to `ng=1` there.

## Long-prompt prefill: the AMX GEMM had an L2 capacity knee (commit 6ad987a)

Profiling at 512 vs 1024 tokens showed GEMM scaling 2.82x for 2x the tokens, where a dense GEMM
should be linear. That anomaly is invisible at 512 tokens, which is why earlier profiles missed it,
and it lands exactly at the prompt lengths where we lose cells to vLLM.

It was not a bucket-ladder artifact: the T=128/512/1024 programs are the same shape (413 insts,
6281 slices, 120 GEMM insts x 16 slices) and differ only in M. In an isolated harness the per-token
cost is flat to M=512 then steps: 3.24 / 3.26 / 4.11 / 4.47 / 4.91 us/tok at M=256/512/768/1024/1536.

**Cause.** `wm_run` sized its token chunk against the 8 MiB scratch only. The packed x panel is
2 KiB/token, so x plus the fp32 partials crossed the 2 MiB private L2 between M=512 (1.5 MiB) and
M=768 (2.25 MiB) -- exactly the knee. Decomposed with `PLOW_AMX_DEBUG`, the x-pack scaled linearly
(0.554 -> 1.055 ms) while the TDP loop went 4.0x for 2x tokens (0.756 -> 2.998 ms): both tile
operands had started coming from L3.

**Fix.** Bound the token chunk so x + partials + the slice's W panel stay under 3/4 of L2, and split
M evenly across chunks (a ragged tail otherwise paid a whole extra W pass for a few tokens). The
extra W pass per chunk is an L3 hit -- one op's W is 23.6 MB against a 260 MB L3 -- far cheaper than
the L2 miss it replaces. Budget swept on the real model: baseline / 2.0 / 1.5 / 1.25 MiB gave GEMM
294.9 / 223.7 / 217.7 / 225.5 ms/thr.

Verified here at 1024 tokens: GEMM 306.1 -> 245.1 ms/thr, prefill wall 1860.5 -> 1765.2 ms. Decode
is unchanged (the decode program contains no GEMM op) and output is bit-identical, since chunking
reorders no output tile's K accumulation. C suite 12/12, Rust suite green.

### It does not close the summarize cell, and the arithmetic says why

summarize TTFT at c=1 is the one lost cell with NO interference component, so it is pure prefill
speed. After this fix we run 1024 tokens in 1765 ms = 580 tok/s, against vLLM's 1111 tokens in
1829 ms = 607 tok/s. We are 4.7% slower on raw rate. Our prefill COMPUTE alone for that prompt
(the 1024 bucket plus the 128 bucket) is about 1986 ms, already more than vLLM's entire 1829 ms
TTFT, so no amount of serve-overhead removal can close this cell. It needs real prefill speed.

Where the remaining time is at 1024 tokens: MoE 928 ms/thr (57%), FLASH_PREFILL 264 (16%),
GEMM 245 (15%).

### Attention on AMX: measured and rejected

A roofline correction first: **VDPBF16PS is 1/cycle on this part, not 2.** Measured 105.3 GMAC/s per
core; calibrating against vfmadd132ps (known 2/cycle) shows the core turbos to ~3.56 GHz. Assuming
2/cycle at 2.3 GHz overstates the ceiling by 2x. Accounting for the shuffles PV actually issues
(4 dpbf16 plus 2 vpunpck, which cost 44% by contending for port 5) gives a realistic ceiling of
75.2 GMAC/s. FLASH_PREFILL achieves 42.6 GMAC/s at the real shape, i.e. **57% of that ceiling, not
the 21% a naive roofline suggests.**

* `FA_RB` sweep (2/4/6/8/12/16, bit-exact): 2 is 12% worse, 4 through 16 all within 2%. So it is not
  latency-bound on the 4-row softmax chain and widening the register-resident state buys nothing.
* `FA_BQ_TILE` sweep (128/64/32): 128 is best for causal layers. 64 helps the sliding-window layers
  by 9%, but those are only ~4 ms/thr of 232, inside noise, and it costs 3% on causal. The apparent
  2x window overcompute is not real: `v_pf_block` already skips fully-masked blocks, so true
  overcompute is ~1.25x.
* AMX is the only real lever (TDPBF16PS is 13.9x vdpbf16ps per core) but 43% of the kernel is
  already non-MAC work -- online softmax, K^T build, masking, loads -- so even a free MAC unit caps
  the win at ~2.3x on 15% of prefill, about 7% of wall, for a change that cannot be bit-exact (P
  must be bf16 for the PV tile op) and that needs the 4-row softmax restructured to feed 16- or
  32-row tiles. Not attempted.

### Identified, not taken

`pack_x_panel` runs once per SLICE, so all 16 slices pack the same full M x K activation. It is
25-38% of q_proj/o_proj time, and for kv_proj (N=512, one 32-column strip per slice) the pack is
~3x the strip's compute, which is why kv_proj gets 281 GFLOP/s/core against q_proj's 900. Splitting
the slice over M rather than N whenever N < M makes the pack 1x instead of 16x. Bit-exact, worth
about 1.2-1.5% of prefill wall.

### Negative result: a wider prefill bucket does not help (2026-09-07, 04:3x)

A 1111-token summarize prompt runs the 1024 bucket PLUS the 128 bucket. Fitting cost against
chunk width (128 rows at 265 tok/s, 1024 rows at 1686.6 ms) gives a fixed ~311 ms per chunk --
one full sweep of the weights -- plus 1.34 ms/row, so the second chunk appeared to be paying a
whole extra sweep to process 87 real tokens.

Partially-filled buckets do NOT compute their padding: `rebase_chunk_rows` rewrites the row-count
fields from `t` down to `clen` when `clen < t` (`kvrow.rs`), so a wider bucket costs nothing extra
in rows. That predicted ~311 ms back on the cell.

The ladder is capped by `default_chunk(window)`, which pins GPT-OSS to 1024 because it has
128-token sliding layers; `PLOW_MAX_CHUNK=2048` overrides it and emits a T=2048 bucket (verified
in the program list). Measured through serve, summarize c=1 TTFT:

| ladder | p50 | mean |
|---|---|---|
| 128/512/1024 (default) | 2122 | 2469 |
| 128/512/1024/2048 | 2184 | 2507 |

**No gain; if anything slightly worse.** The saved sweep is cancelled because the wider program is
less efficient per row: the T=1024 program runs 607 tok/s where T=2048 runs 554, about 9% worse,
and 1111 rows placed in the T=2048 program inherit that program's tiling. The two effects are the
same size, so they cancel.

Worth knowing for anyone tempted by the same idea: widening the ladder also doubles the sliding
layers' KV ring, since it is `next_pow2(window + chunk - 1)` = 2048 -> 4096. So it costs memory
for no throughput.
