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
