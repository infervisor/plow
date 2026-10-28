# 28 — Library attention kernels inside plowrt

**Status: experiment (2026-10-28). Kernel-level head-to-head done; one library route
(hd512 prefill) implemented and under in-model measurement; nothing adopted by a recipe yet.**

The question: where plowrt's native attention loses to what vLLM and SGLang run, can plowrt run
the library kernel itself, per rung, without changing the KV layout, the packet contract or the
one-binary rule? This document records what is expressible, what was measured, and the one
route built from it.

## 1. What the libraries can and cannot express

| | FlashAttention-3 (hopper) | FlashAttention-4 (CuTe DSL, SM90 path) | FlashInfer 0.6.16 |
|---|---|---|---|
| head dim | <= 256 (hd_v 512 only with hd_qk <= 64) | up to 512 | 64/128/256 (decode); fa3 backend <= 256 |
| FP8 KV | FP8 Q/K/V, descale per (sequence, KV head) | none for paged FP8 KV | scalar k/v scale folded into the softmax scale; block scales only on Blackwell (NVFP4) |
| GQA 6 decode (Qwen3.8) | yes | yes | tensor-core path only (CUDA-core path: "unsupported group size 6") |
| plowrt flat slot KV | yes, page table, no copy | yes, page table, no copy | yes, page table with explicit strides, no copy |
| sliding ring | yes: logical page j -> ring page (j*P & mask)/P, window_left = W-1 | same | same |

* plowrt's FP8 KV stores one f32 scale per (position, KV head) row. No library takes that
  granularity, so every FP8 route needs an adapter: gather the rows a launch reads, dequantize
  to bf16, attend. For decode that moves ~5x the native bytes (the rows are re-read every step),
  so FP8 decode attention stays native. For prefill the gathered rows are read once per launch
  against a compute-bound kernel, and the adapter is cheap (below).
* Layout mapping without copies: page size P divides 4096 (the live-KV commit block) and the slot
  stride. Row-interleaved slots `[slot][stride][kvh][hd]` are plain NHD pages, page id
  `(slot*stride + row)/P`. Head-major slots `[slot][kvh][stride][hd]` are pages of page stride
  `P*hd`, row stride `hd`, head stride `stride*hd`, page id `(slot*kvh*stride + row)/P` (ids are
  gapped; only referenced ids are read).
* Live-KV safety: every library path loads whole tiles or pages aligned to position 0 whose size
  divides 4096, so no load crosses a commit block. Measured with a VMM-backed cache mapping only
  `ceil(kv_len/4096)` blocks per slot at kv_len 4094/4096/8191/8192/4097 (B 4 and 16, mixed
  lengths): native, FA3 (with and without scheduler metadata) and FlashInfer pass.

### What vLLM 0.29 runs on H100

* Qwen3.8: FLASH_ATTN = FA3, prefill and decode.
* Gemma-4 with BF16 KV: FA4 for every layer ("heterogeneous head dimensions ... Using FA4 for all
  layers"), because FA3 has no hd512 kernel.
* Gemma-4 with FP8 KV (the 12B and 26B production arms): TRITON_ATTN for every layer; neither FA3
  nor FA4 accepts that cache there.

## 2. Kernel head-to-head at serving shapes

Tool: `scripts/gen_kernels/bench_lib_attn.py` (packet's own role objects through their direct
entries vs the libraries over the same memory; CUDA graph over rotating cold buffers; rel-L2 vs
FP32). Objects: Qwen3.8 BF16 packet cb2ea611 (`attn_dec_hd256_gqa6_ilv`,
`attn_pf_hd256_causal_ilv`), 12B/26B FP8 packets a0464475 / caf6c4a4 (identical
`attn_pf_hd256_sliding_fp8kv` and `attn_pf_hd512_fp8kv`), 26B BF16 (`attn_pf_hd256_sliding`).
Library time includes the FP8 adapter where one is needed. Ratios are library / native.

| rung | native | best library | ratio | note |
|---|---|---|---|---|
| Qwen decode B8-16, ctx 8-16K | 108-394 us | FA3 99-352 | 0.89-0.92 | B32: 0.99; B128: 0.93; ctx 1K: 1.0-1.06 |
| Qwen prefill 1024-2048 rows, past 0-14K | 40-1116 us | FA3 37-1054 | 0.90-0.95 | 512 rows: native wins (FA3 1.0-2.0) |
| Qwen prefill 128 rows, past 4-14K | 109-344 us | FA3 41-88 | 0.26-0.37 | native has no split-KV here |
| Gemma sliding hd256 BF16, 1024-2048 rows | 38-84 us | FA3 36-80 | 0.89-0.95 | 512 rows: native wins |
| Gemma sliding hd256 FP8, 1024-2048 rows | 64-134 us | FA3 + adapter 43-94 | 0.64-0.77 | adapter 8-17 us; riders (1-row requests) lose 1.2-2x to the adapter |
| Gemma global hd512 FP8, 512-2048 rows | 67-5250 us | FA4 + adapter | 0.48-0.60 | adapter 5-42 us |
| Gemma global hd512 FP8, 128 rows, past 4-14K | 391-1260 us | FA4 split-KV + adapter | 0.17-0.20 | without split-KV 0.51-0.60 |

Numerics (rel-L2 vs FP32): bf16 paths match native (2.0-2.3e-3). Through the bf16 adapter the FP8
paths roughly double it (hd512 1.7e-3 -> 3.4e-3, hd256 2.3e-3 -> 3.3e-3): the adapter rounds each
dequantized row to bf16, while the native kernels keep the scale in the MMA's f32 domain. An fp16
pool restores native error at a 5-50% speed cost. The full-model FP32 gate decides.

## 3. The hd512 prefill route

`crates/plowrt/src/exec/gpu/lib_attention.rs`, beside `attention_gemm.rs` and built the same way:
runtime-side, host-launched between the CUDA-graph pieces of a prefill launch, packet unchanged.

* Library: `libplow_fa4_hd512.so` (`scripts/build_sm90a_fa4_hd512.py`): vLLM's FA4 SM90 forward
  exported AOT with the CuTe DSL (`export_fa4_hd512_sm90.py`; the GDN-prefill precedent), causal,
  varlen Q, 64-row pages on the TMA path; a whole-KV and a split-KV entry per GQA ratio (16 over
  1 or 2 KV heads) plus the combine entry; vLLM's split rule. C ABI, libc-only dependencies.
* Adapter: `kv_dequant_sm90.cubin` gathers each request's rows `[0, kv_len)` per KV head into a
  bf16 page pool (64K rows per KV head), zero-filling the tail of the last page and never reading
  a slot row at or past kv_len.
* One table upload per launch (cu_seqlens, kv lengths, gather list, page table); every hd512
  site of the launch reuses it. A launch past the pool or 256 requests, or a windowed body, keeps
  the native role. Library launches do not signal the packet's counters: their consumers' waits
  are rewritten to stream order at load (`ordered_waits_for`), which also covers the fallback.
* Selection today: the two files beside the packet. Per-rung selection by the emitter follows once
  the per-rung gate says which rungs win.

### In-model prefill (12B FP8, one 16000-token prompt, step_bench packed prefill, 3 reps)

| chunk | native | hd512 library route | delta |
|---|---|---|---|
| 2048 | 0.607 s | 0.540 s | -11% (chunk at prefix 14K: 94 -> 78 ms) |
| 1024 | 0.634 s | 0.564 s | -11% (prefix 14K: 48 -> 40 ms) |
| 512 | 0.760 s | 0.664 s | -13% (prefix 14K: 28 -> 24 ms; 128-row tail 25 -> 16 ms) |

First chunks (no prefix) are equal; the gain grows with cached prefix, as the kernel table
predicts. Cost: 10x the prefill seg graphs (480 vs 48; warm-up 0.64 s vs 0.25 s), a 128 MiB
split workspace and 2 x 64 MiB per KV head of bf16 page pool.

## 4. Open

* FP32 gate for the hd512 route (12B, 26B FP8); 26B and chunk-128 in-model numbers.
* FA4 hd256 local (sliding) exports the same way (AOT compile verified); FA3 C++ would need a
  separate build. Measure FA4 hd256 before choosing.
* Qwen: FA3 decode/prefill wins of 5-11% at kernel level need an FA3 C++ AOT library and a route
  in `gen_decode.rs`; at 16 attention layers this is ~1-3% of a step.
