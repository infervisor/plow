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

vLLM cannot serve the 26B on this machine at all, and the reason is not the one recorded earlier.
Its CPU backend *does* have four quantized MoE expert paths for x86 (fp8, MXFP4, int4, int8, all
AMX-gated, and this box has AMX) — but every one of them requires a SILU-family activation, and
Gemma-4's MoE is GELU-tanh (`gemma4.py:368`). The only GELU-capable x86 expert path is the
unquantized one, so vLLM must hold the experts in bf16: 47.00 GiB of text weights (42.54 GiB of
that in experts) against a 58.85 GiB box, and the engine core dies during init at 180 s even at
2048 context with one sequence. A bigger box would only let the bf16 path fit; the quantized
kernels stay refused. plow serves the same model from a 13 GB MXFP4 twin at ~21 GB resident.
Full trace, with the byte accounting and the class-by-class activation gates, in
`cpu-gemma26b/vllm-baseline.md`.

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
* **When a baseline "cannot run", find the gate before recording why.** "vLLM has no 4-bit path for
  the 26B" was recorded here and was wrong: it has four, and they are refused on an activation
  check, not a memory or format one. Reading `CpuPlatform.supported_quantization == []`
  as "nothing supported" is the specific trap — vLLM treats an empty list as *no restriction*
  (`platforms/interface.py:966`). Static source tracing settled this in minutes with no model load.

## MXFP4 dense prefill: a 4-bit blob no longer carries bf16 too (commit 96ad1d5)

A quantized blob used to declare BOTH the bf16 originals and the quantized twins, because decode
went through `GEMV_MXFP4` while prefill used the plain bf16 `GEMM`. The `GEMM_MXFP4` family existed
in `dev_isa.h` with no CPU kernel at any tier, so the emitter had no choice. Consequence: the
Gemma-4-12B MXFP4 build was **larger than its own bf16 build**.

Implementing the family (golden + AMX, all six opcodes, riding the existing pack-free `wm_run`
driver with `dequant_strip` hoisted into `mxfp4_common.h`) lets the emitter drop the duplicates.

Verified here, independently of the implementing agent:

| Gemma-4-12B | resident set | 512-token prefill |
|---|---|---|
| plain bf16 | 28.46 GiB | - |
| MXFP4, bf16 prefill (before) | 34.69 GiB | 3457.8 ms, 148 tok/s |
| MXFP4, fp4 prefill (after) | **14.26 GiB** | **2413.1 ms, 212 tok/s** |

Resident set measured on the serving process (`ps -o rss=`), matching the agent's figure exactly.
Prefill is an interleaved A/B over two blobs emitted from the same command differing only in
`PLOW_MX4_PREFILL`, 3 pairs, all three consistent: 3438.7/3457.8/3467.4 against
2449.4/2405.4/2413.1. **-30% prefill time and -20.4 GiB.**

**My prior expectation was wrong, and the reason is worth keeping.** I predicted a memory win and
no speed win, reasoning that the GEMM reads ~1.27 GB per forward (~12 ms at this box's bandwidth)
against ~220 ms of busy time, so it is compute-bound and MXFP4 only adds dequant work. That
reasoning was right for GPT-OSS, where only q/k/v/o move and they are a small slice of a much
larger MoE read -- it measured flat there, -0.2%/-1.2%/+0.1%. It was wrong for a DENSE model: a
dense CPU prefill streams the entire weight set once per chunk, and the 12B's whole set is 22 GiB,
so quartering it dwarfs the unpack cost. Check whether the model is dense or sparse before
applying a bandwidth argument to its prefill.

GPT-OSS still gains the memory: 15.34 -> 12.94 GiB, and the -2.40 GiB matches its tensor table
exactly (97 tensors, 2.265 GiB: 24x4 projections plus lm_head).

Not bit-exact, since prefill now uses quantized weights. On the 12B, divergence from plain bf16
goes 49/160 greedy tokens (the existing fp4-decode error) to 67/160, and essentially all of the
increase is one prompt moving its first difference from index 30 to 13; two of five prompts stay
token-identical to the old fp4 blob. `PLOW_MX4_PREFILL=0` is a byte-identical opt-out at emit time.
Default is gated on the AMD/CPU target, off for sm_90a/sm_120a where the kernels do not exist.

C suite 12/12, Rust and devgen suites green.
