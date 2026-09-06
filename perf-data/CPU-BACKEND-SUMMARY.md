# plow CPU backend vs llama.cpp and vLLM — consolidated results

Box: Sapphire Rapids, 8 cores / 16 threads, 58 GB, AVX-512 + AMX. All figures are **through the
OpenAI API** (`plowrt serve`, `llama-server`, `vllm serve`), `tools/bench-api/bench.py` with
`--fresh-prompts`, 8 requests per cell, 64 max tokens, one server at a time on a quiet box, matched
server slots (8 everywhere). TTFT and TPOT are means in ms. Raw JSON and per-run Markdown
were removed after consolidation.

Read the per-model files for the full 16-cell tables: `cpu-gptoss/SUMMARY.md`,
`cpu-gemma26b/h2h/SUMMARY.md`, `cpu-gemma26b/llamacpp/SUMMARY.md`, `cpu-gemma/h2h/SUMMARY.md`.

## Scorecard

| model / data type | plow | llama.cpp | vLLM | verdict |
|---|---|---|---|---|
| GPT-OSS-20B MXFP4, decode c=1 | 24-25 | 41-59 | 71-76 | win both, 1.7-3.0x |
| GPT-OSS-20B MXFP4, TTFT | wins 14/16 vs both | | | win |
| GPT-OSS-20B MXFP4, decode c>=4 long | 142-195 | 177-430 | 137-153 | beats llama, loses vLLM |
| Gemma-4-26B-A4B MXFP4, decode c=1 | 35-38 | 50-63 | cannot load | win vs llama |
| Gemma-4-26B-A4B MXFP4, TTFT | wins all 16 vs llama | | cannot load | win |
| Gemma-4-12B MXFP4, decode c=1 | 88 | 121 | 460 | win both, 1.38x / 5.2x |
| Gemma-4-12B bf16, decode c=1 | 232 | 267 | 460 | win both, 1.15x / 2.0x |
| Gemma-4-12B fp8, decode c=1 | 133 | 133 | 460 | TIE vs llama, 3.5x vs vLLM |

vLLM cannot serve the 26B on this machine at all: the checkpoint is bf16 (47 GB), its CPU backend
has no 4-bit path for it, and the worker is OOM-killed at load even at 2048 context. plow serves the
same model from a 13 GB MXFP4 twin at ~21 GB resident.

## The two unmet items, and why

**fp8 ties llama.cpp Q8_0 and cannot do better here.** Our fp8 weights are 12.0 GB against Q8_0's
12.75 GB (8 bits plus an f16 scale per 32), and both run at the same ~96 GB/s wall rate. Profiling
shows workers busy 112 of 125 ms and moving 13 GB at ~116 GB/s while busy, which is this machine's
measured memory ceiling. There is no margin to win at equal bit width; MXFP4 is where the margin
lives on that axis, at 1.38x.

**The c>=4 cells are prefill interference, not kernel speed.** Our batched MoE decode at rung 8 runs
a step in 100 ms against vLLM's 137 ms measured TPOT — we are 1.37x *faster* per step. The served
number in those cells is 142-195 ms, so 42-95 ms per token is time spent waiting behind another
request's prompt. vLLM runs prefill chunks and decode rows in one forward; no plow backend does,
GPU included (program shape is an enum of prefill / decode / decode-tiled, and the "fused
prefill+decode tick" in the scheduler is a tick, not a step). Chunking the prefill instead does not
help: `--pf-interleave 512` moved chat_long c=4/c=8 TPOT from 100/195 to 100/198, because the work
is throughput-bound rather than stall-bound.

Not implemented at all: int8 (w8a8) weights, and an fp8 KV cache. Both have emitter flags and no CPU
kernels.

## What moved the numbers

| change | effect |
|---|---|
| MXFP4 experts for the 26B (quantizer + emitter ops 150-153) | made the model competitive at all: decode 74 -> 41 ms |
| MXFP4 dense weights + head for GPT-OSS (biased MXFP4 GEMV) | decode 44 -> 27 ms |
| MoE prefill: hoist the dequant out of the token-block loop | 512-token prefill 11.0 s -> 1.63 s |
| MoE prefill: weight the slice split by rows per expert | 1.63 s -> 1.24 s, worker idle 50% -> 31% |
| Worker width per model (physical for MoE, logical for dense) | GPT-OSS prefill 399 -> 455 tok/s |
| AMX pack-free prefill GEMM (weights as the A operand) | GEMM +21-33%, no weight pack at all |
| Kernel asm pass (spills, MXFP4 dequant, attention, Gemma MoE) | decode 1.2-2.2x per op |
| Gemma AVX-512 router scoring | TTFT -29-54%, TPOT -7-38% across 12 serve cells |
| Experimental int16 VNNI MXFP4 decode (opt-in) | GPT-OSS c=1 decode 24-25 -> 22-23 ms |

## Methodology notes that cost real time

* **Compare serve to serve, and use the mean.** `cpu_bench`'s median runs ~10 ms below its own mean
  and below every serve number; it made fp8 look like a 1.07x win twice. Its mean matches serve.
  A non-streaming curl confirms there is no client-side or serve-side per-token tax.
* **Match server slots to concurrency.** An early 26B llama.cpp baseline ran `-np 4` against
  concurrency 8, measuring llama.cpp on half the load; with matched slots four decode cells flipped.
* **Always `--fresh-prompts`.** Prefix caches inflate a server's c>=2 TTFT several-fold.
* **Single prefill measurements vary ~10% on this box.** Repeat three times.
* **Re-emit every blob after an opcode renumber.** A merge silently gave three opcodes two meanings.
