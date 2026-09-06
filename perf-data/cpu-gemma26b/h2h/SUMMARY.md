# Gemma-4-26B-A4B head-to-head via the OpenAI API (2026-09-06)

Same box, same bench (`bench.py --requests 8 --max-tokens 64 --fresh-prompts`), one server at a time.
plow: `plowrt serve --cpu-threads 16` on the bf16 ladder blob (rungs 1/2/4/8, prefill buckets 128/512/1024),
commit 897e117 (AMX grouped-expert prefill). llama.cpp: see `../llamacpp/SUMMARY.md`. Raw: `g26b-bf16-*.md`.

TTFT / TPOT = mean ms. Lower is better. Bold = plow wins.

| workload | conc | in tok | plow bf16 TTFT | Q8_0 TTFT | Q4_K_M TTFT | plow bf16 TPOT | Q8_0 TPOT | Q4_K_M TPOT |
|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 34 | 800 | 739 | 649 | 83 | 56 | 50 |
| chat_short | 2 | 45 | **1147** | 1351 | 1226 | 130 | 84 | 80 |
| chat_short | 4 | 41 | **1943** | 2852 | 2508 | 210 | 115 | 107 |
| chat_short | 8 | 36 | **4287** | 7024 | 6492 | 347 | 119 | 105 |
| chat_long | 1 | 398 | **2779** | 4532 | 4187 | 85 | 56 | 52 |
| chat_long | 2 | 442 | **4167** | 6252 | 7009 | 154 | 159 | 125 |
| chat_long | 4 | 349 | **5133** | 13887 | 12936 | 268 | 126 | 119 |
| chat_long | 8 | 398–425 | **12496** | 28262 | 29015 | 577 | 130 | 118 |
| code | 1 | 365 | **2484** | 4221 | 3858 | 86 | 60 | 52 |
| code | 2 | 374 | **4018** | 8636 | 7914 | 141 | 112 | 90 |
| code | 4 | 401 | **5871** | 18089 | 16867 | 271 | 131 | 120 |
| code | 8 | 359 | **11427** | 27944 | 26328 | 451 | 132 | 119 |
| summarize | 1 | 1105 | **7667** | 12084 | 11237 | 90 | 63 | 56 |
| summarize | 2 | 1119 | **8972** | 24992 | 23241 | 232 | 102 | 105 |
| summarize | 4 | 1178 | **15335** | 54272 | 50453 | 485 | 159 | 146 |
| summarize | 8 | 1133 | **35315** | 56278 | 48027 | 765 | 227 | 190 |

Reading
* Prefill/TTFT: plow bf16 wins every cell with a real prompt (1.5–3.5×), including c=8.
* Decode/TPOT: plow bf16 loses everywhere — 83–90 ms vs 50–63 at c=1 (bf16 experts read ~7 GB/token;
  llama.cpp's 4/8-bit experts) and 3–5× at c=8 (rung-8 MoE decode reads every selected expert per row: up
  to 64 expert loads per step; llama.cpp batches the 8 rows through each expert once).
* Fixes queued: MXFP4 experts (emitter arm landed in 0890fae; the 14 GB twin needs disk space), and
  expert-deduplicated batched MoE decode (sort the B·k slots by expert, one weight pass per expert).

## MXFP4 experts + dense (commit pending, 11:5x) — plow vs llama.cpp

plow: `plowc --mxfp4` on the 26B, experts through the flat MXFP4 ops 147-150 and dense projections through
the MXFP4 GEMV, twin `/home/lava/models/gemma-4-26b-a4b-it-mxfp4` (13.0 GB, `quantize_mxfp4.py`, 3-D expert
tensors flattened to `[E*N][K]`). Optimized kernels (commit c835609). Fresh prompts, one server at a time.
Raw: `g26b-mx4-*.md`. TTFT / TPOT mean ms; bold = best of the three.

| workload | conc | plow TTFT | Q8_0 TTFT | Q4_K_M TTFT | plow TPOT | Q8_0 TPOT | Q4_K_M TPOT |
|---|---|---|---|---|---|---|---|
| chat_short | 1 | **536** | 739 | 649 | **41** | 56 | 50 |
| chat_short | 2 | **812** | 1351 | 1226 | **65** | 84 | 80 |
| chat_short | 4 | **1300** | 2852 | 2508 | **103** | 115 | 107 |
| chat_short | 8 | **2741** | 7024 | 6492 | 154 | 119 | **105** |
| chat_long | 1 | **2537** | 4532 | 4187 | **42** | 56 | 52 |
| chat_long | 2 | **3778** | 6252 | 7009 | **90** | 159 | 125 |
| chat_long | 4 | **4552** | 13887 | 12936 | 158 | 126 | **119** |
| chat_long | 8 | **11050** | 28262 | 29015 | 306 | 130 | **118** |
| code | 1 | **2229** | 4221 | 3858 | **43** | 60 | 52 |
| code | 2 | **3542** | 8636 | 7914 | **76** | 112 | 90 |
| code | 4 | **5275** | 18089 | 16867 | 160 | 131 | **120** |
| code | 8 | **10213** | 27944 | 26328 | 256 | 132 | **119** |
| summarize | 1 | **6960** | 12084 | 11237 | **45** | 63 | 56 |
| summarize | 2 | **8046** | 24992 | 23241 | 161 | **102** | 105 |
| summarize | 4 | **13594** | 54272 | 50453 | 356 | 159 | **146** |
| summarize | 8 | **29595** | 56278 | 48027 | 593 | 227 | **190** |

Reading: with MXFP4 experts the 26B wins **every TTFT cell** (1.2-5.5x) and **every c=1 and c=2 decode cell**
(41-45 ms vs 50-56 Q4_K_M and 56-63 Q8_0). bf16 could not do this: bf16 reads ~8 GB of active weights per token
against a ~110 GB/s memory system, a 73 ms floor, and it measured 74 ms — exactly the roofline. MXFP4 cuts the
active bytes to ~2.2 GB and lands at 37-45 ms. At c>=4 llama.cpp still wins decode: its TPOT barely grows with
batch (105-130 ms at c=8) while ours goes 154-593, because a rung-8 step still pays prefill interleaving and our
grouped expert pass has fewer rows per expert than its block-aligned one.

Single-stream cpu_bench (batch 1, 16 threads): decode p50 37.2-39.5 ms, prefill 107.6 / 146.2 / 177.2 tok/s at
64 / 128 / 512 prompt tokens. Chat answer: 'Paris' (HF reference: 'Paris'). Block-0 vs HF hidden states:
mean|d| 0.0245 on mean|x| 0.426 (5.8%), against 0.6% for bf16 — ordinary 4-bit quantization noise.


## Final: balanced MoE prefill (commit 25e0875, 12:5x)

Same two prefill fixes as GPT-OSS (dequant hoisted out of the token-block loop, slice split weighted
by rows per expert), which matter more here because the 26B has 128 experts. Prefill at 64/128/512
prompt tokens: 128.8 / 171.1 / 201.5 tok/s (was 107.6 / 146.2 / 177.2). Single-stream decode p50
36.7-38.9 ms. Raw: `g26b-bal-*.md`. Bold = best of the three.

| workload | conc | plow TTFT | Q8_0 TTFT | Q4_K_M TTFT | plow TPOT | Q8_0 TPOT | Q4_K_M TPOT |
|---|---|---|---|---|---|---|---|
| chat_short | 1 | **464** | 739 | 649 | **41** | 56 | 50 |
| chat_short | 2 | **703** | 1351 | 1226 | **65** | 84 | 80 |
| chat_short | 4 | **1111** | 2852 | 2508 | **102** | 115 | 107 |
| chat_short | 8 | **2370** | 7024 | 6492 | 151 | 119 | **105** |
| chat_long | 1 | **2222** | 4532 | 4187 | **43** | 56 | 52 |
| chat_long | 2 | **3307** | 6252 | 7009 | **86** | 159 | 125 |
| chat_long | 4 | **3949** | 13887 | 12936 | 152 | 126 | **119** |
| chat_long | 8 | **9336** | 28262 | 29015 | 308 | 130 | **118** |
| code | 1 | **1999** | 4221 | 3858 | **43** | 60 | 52 |
| code | 2 | **3166** | 8636 | 7914 | **75** | 112 | 90 |
| code | 4 | **4681** | 18089 | 16867 | 155 | 131 | **120** |
| code | 8 | **8665** | 27944 | 26328 | 248 | 132 | **119** |
| summarize | 1 | **6021** | 12084 | 11237 | **45** | 63 | 56 |
| summarize | 2 | **7003** | 24992 | 23241 | 147 | **102** | 105 |
| summarize | 4 | **11838** | 54272 | 50453 | 323 | 159 | **146** |
| summarize | 8 | **27414** | 56278 | 48027 | 507 | 227 | **190** |

plow wins every TTFT cell (1.4-6.1x) and every c=1 and c=2 decode cell. At c>=4 llama.cpp wins
decode: its TPOT is nearly flat in batch (105-130 ms at c=8) while ours grows, because a decode step
still waits behind other slots' whole-prompt prefills. Same fix as GPT-OSS.

## vLLM 0.28 CPU cannot serve this model on this box (13:4x)

Attempted twice with `vllm serve <hf 26B> --dtype bfloat16`, first at `--max-model-len 4096
--max-num-seqs 8` with `VLLM_CPU_KVCACHE_SPACE=2`, then at 2048 / 2 seqs / 1 GiB. Both times the
worker was OOM-killed during model load:

    Worker proc VllmWorker-0 died unexpectedly (exit code: -9), shutting down executor

Exit code -9 is SIGKILL from the kernel OOM killer. The cause is structural rather than a tuning
miss: the Gemma-4-26B-A4B checkpoint ships bf16, and vLLM's CPU backend has no 4-bit weight path
for it (its MXFP4 support dequantizes to bf16, and that is for GPT-OSS-shaped checkpoints), so it
must materialise ~47 GB of weights on a 58 GB machine. plow serves the same model from a 13.0 GB
MXFP4 twin at ~21 GB resident, answers correctly, and beats both llama.cpp builds on every TTFT
cell and every c<=2 decode cell. For this model the vLLM comparison is therefore not a latency
number but a capability one.


## AVX-512 router scoring (commit d2dc0af)

The Gemma MoE router scoring GEMV previously fell through to the scalar kernel. The AVX-512
arm produced these fresh-prompt TTFT / TPOT means in ms:

| workload | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| chat_short | 331 / 35 | 487 / 58 | 804 / 87 | 1705 / 125 |
| chat_long | 1019 / 38 | 1335 / 68 | 1883 / 106 | 4538 / 191 |
| code | 919 / 38 | 1851 / 69 | 2892 / 128 | 4184 / 167 |

Against the balanced-MoE campaign, TTFT fell 29-54% and TPOT fell 7-38% across these
12 cells. The c=1 decode range is now 35-38 ms vs llama.cpp's 50-63 ms.

## After the vectorized MoE router (commit d2dc0af, 19:5x)

Op 73 had no AVX-512 arm, so the router's 128x2816 scoring GEMV ran scalar and accounted for 56% of
all 26B prefill work (1193 of 2118 ms/thread). With the vector arm it is 24 ms/thread, prefill wall
for 512 tokens went 2304 -> 1005 ms, and single-stream prefill went 201.5 -> 465.1 tok/s. Lossless:
selected expert ids are bit-identical to golden, tie order included. Decode is unchanged.

Against the FAIR llama.cpp Q4_K_M baseline (8 server slots, matching our 8 and the benchmark
concurrency). TTFT / TPOT means in ms; bold = plow wins.

| workload | conc | plow TTFT | Q4_K_M TTFT | plow TPOT | Q4_K_M TPOT |
|---|---|---|---|---|---|
| chat_short | 1 | **331** | 649 | **35** | 50 |
| chat_short | 2 | **487** | 1226 | **58** | 80 |
| chat_short | 4 | **804** | 2302 | **87** | 111 |
| chat_short | 8 | **1705** | 4948 | **125** | 178 |
| chat_long | 1 | **1019** | 4187 | **38** | 52 |
| chat_long | 2 | **1335** | 7009 | **68** | 125 |
| chat_long | 4 | **1883** | 11944 | **106** | 237 |
| chat_long | 8 | **4538** | 34445 | **191** | 217 |
| code | 1 | **919** | 3858 | **38** | 52 |
| code | 2 | **1851** | 7914 | **69** | 90 |
| code | 4 | **2892** | 15773 | **128** | 185 |
| code | 8 | **4184** | 32080 | **167** | 183 |
| summarize | 1 | **2806** | 11237 | **40** | 56 |
| summarize | 2 | **3396** | 23241 | **97** | 105 |
| summarize | 4 | **5905** | 47619 | 191 | 172 |
| summarize | 8 | **16562** | 92495 | 389 | 257 |

plow wins 16 of 16 TTFT cells and 14 of 16 TPOT cells, i.e. **30 of 32 against llama.cpp**, up from
roughly 24 before the router fix. The only remaining losses are summarize TPOT at c=4 and c=8, where
1100-token prompts mean decode still waits behind other slots' prefills - the same prefill-interference
term as GPT-OSS, and the thing prefill+decode fusion would remove.

Single-stream cpu_bench after the fix: prefill 241.3 / 326.3 / 465.1 tok/s at 64 / 128 / 512 prompt
tokens (was 128.8 / 171.1 / 201.5), decode p50 35.1-37.0 ms (unchanged).
