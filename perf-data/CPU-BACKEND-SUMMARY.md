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
| GPT-OSS-20B MXFP4, decode c=1 | 23-26 | 41-59 | 71-76 | win both, 1.6-3.1x |
| GPT-OSS-20B MXFP4, TTFT | wins 14/16 vs both | | | win |
| GPT-OSS-20B MXFP4, decode c>=4 long | 92-279 | 123-430 | 102-153 | beats llama 16/16, vLLM 12/16 |
| Gemma-4-26B-A4B MXFP4, decode c=1 | 35-38 | 50-63 | cannot load | win vs llama |
| Gemma-4-26B-A4B MXFP4, TTFT | wins all 16 vs llama | | cannot load | win |
| Gemma-4-12B MXFP4, decode c=1 | 81 | 121 | 460-544 | win both, 1.49x / 5.7x |
| Gemma-4-12B bf16, decode c=1 | 233 | 267 | 460-544 | win both, 1.15x / 2.0x |
| Gemma-4-12B fp8, decode c=1 | 127 | 133 | 460-544 | win both, 1.05x / 3.6x |

vLLM cannot serve the 26B on this machine at all: the checkpoint is bf16 (47 GB), its CPU backend
has no 4-bit path for it, and the worker is OOM-killed at load even at 2048 context. plow serves the
same model from a 13 GB MXFP4 twin at ~21 GB resident.

## The two unmet items, and why

**fp8 no longer ties llama.cpp Q8_0 — RESOLVED.** The earlier reading, that fp8 was pinned at the
memory ceiling with no margin available at equal bit width, was wrong about where the bytes were
going. Gemma-4 ties `lm_head` to `embed_tokens`, and that 2.01 GB bf16 tensor was in NEITHER
quantized twin, so every decode step streamed it for the output projection while every other weight
was quantized. It profiled at 17.06 ms/thread with a span of 17.53 ms, a serial tail worth 13% of the
fp8 step and 21% of MXFP4. Giving the final GEMV an MXFP4 copy while leaving the bf16 table bound for
the `EMBED` row lookup (`PLOW_MX4_HEAD`, commit c9d4033) took fp8 from 141 to 127 ms and MXFP4 from
93 to 81. All three data types now beat both baselines.

The lesson generalizes: the fp8-vs-Q8_0 byte comparison was sound for the *body* weights and led to
the conclusion that no margin existed, but the body was never the whole read. Check what the profile
says is actually being streamed before concluding a configuration is at its floor.

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
| MoE prefill: transposed epilogue + call-site-gated prefetch | prefill 455 -> 470 tok/s (GPT-OSS), +10% (26B) |
| Flash decode: fold GQA head groups onto one K/V pass | 4x fewer KV bytes read; batch-8 step 118 -> 105 ms |
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
