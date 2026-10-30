# 28 — Library attention kernels inside plowrt

**Status (2026-10-29): the hd512 prefill route (§3) is adopted by two production recipes:
Gemma-4 12B FP8 with the fp16 page pool (`PLOW_BUILD_FA4_HD512 = "f16"`) and 26B-A4B FP8 with the
bf16 pool (`= "1"`), both qualified 2026-10-29 (FP32 gate at c1/c8/c32 against same-concurrency
peers, production-mix replay; evidence in the recipe headers). A Qwen3.8 hd256 FA4 route exists
only on branch `lib-attn2`, not in prod-next (§4).**

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
* Selection: the packet carries the choice. The recipe's object-build knob
  `PLOW_BUILD_FA4_HD512` (`1` = bf16 pool, `f16` = fp16 pool; `scripts/build_sm90a_gemma4_segments.sh`)
  puts `libplow_fa4_hd512.so` and `kv_dequant_sm90.cubin` into the packet's objects directory, and
  plowrt routes every FP8 hd512 `FlashPrefill` segment through the library when both files are
  there (`lib_attention.rs` `LIBRARY` / `DEQUANT_OBJECT`), otherwise runs the native role. There is
  no runtime knob. Per-rung selection by the emitter is not built.

### In-model prefill (one 16000-token prompt, step_bench packed prefill, 2 x 3 reps ABAB)

| chunk | 12B native | 12B library | delta | 26B native | 26B library | delta |
|---|---|---|---|---|---|---|
| 2048 | 0.608 s | 0.540 s | -11% | 0.419 s | 0.373 s | -11% |
| 1024 | 0.634 s | 0.564 s | -11% | 0.505 s | 0.463 s | -8% |
| 512 | 0.759 s | 0.668 s | -12% | 0.739 s | 0.678 s | -8% |
| 128 | 2.381 s | 1.802 s | -24% | 2.325 s | 1.995 s | -14% |

First chunks (no prefix) are equal; the gain grows with cached prefix, as the kernel table
predicts. Both arms are run-to-run bit-identical; last-position KL native vs library is
1.4e-4..7.5e-4 (12B) and 2.4e-3..3.0e-2 (26B), top-1 unchanged. Cost: 10x the prefill seg graphs (480 vs 48; warm-up 0.64 s vs 0.25 s), a 128 MiB
split workspace and 2 x 64 MiB per KV head of bf16 page pool.

### FP32 gate, c16 screen (recipe env, production packets, one run each)

| | kl_trim | kl_p95 | top1_decisive | cont_decisive | needles | verdict |
|---|---|---|---|---|---|---|
| 12B library route | 0.0517 | 0.277 | 0.9852 | 0.891 | 1.0 | PASS |
| 12B vLLM 0.29 peer | 0.0653 | 0.367 | 0.9823 | 0.913 | 1.0 | |
| 26B library route | 0.1381 | 1.085 | 0.9672 | 0.848 | 1.0 | PASS |
| 26B vLLM 0.29 peer | 0.1242 | 0.892 | 0.9662 | 0.804 | 1.0 | |

The 26B run sits inside the recipe's measured native spread (kl_trim 0.124-0.153, p95 0.91-1.08).

### Promotion run (bin3, production packets 12B a0464475 / 26B caf6c4a4)

Production-mix replay, 2 reps, same client and vLLM 0.29 arm as the recipes:

| | TTFT P99 | vs vLLM | TPOT P99 | goodput | SLO met |
|---|---|---|---|---|---|
| 12B 2.7/s native (same binary) | 1338.5 ms | 1.19x | 105.4 ms | 9.05 req/s | 97.5% |
| 12B 2.7/s library route | 1187.5 ms | 1.05x | 103.9 ms | 9.16 req/s | 98.4% |
| 26B 1.8/s library route | 260.7 ms | 0.71x | 22.3 ms | 7.35 req/s | 100% |

FP32 gate at c1/c8/c32 against same-concurrency vLLM peers (captured with the c16 peers' flags):

* 26B: c1 and c8 pass; c32 passes 2 of 3 runs (kl_p95 1.216 / 0.972 / 0.939 vs limit 1.12;
  native 3 of 3, kl_p95 0.98-1.05, kl_trim 0.145-0.153 vs library 0.134-0.151).
* 12B: c8 and c32 pass; c1 fails: cont_decisive 40/46 vs the peer's 43/46 (limit 41). The c1
  capture is deterministic and native passes it at 44/46, so the 4 free-run divergences are the
  library route's: the bf16 page pool rounds each dequantized row (kernel rel-L2 2x native).

### 12B with an fp16 page pool (bin4, `PLOW_BUILD_FA4_HD512=f16`)

The fp16 library variant reads an fp16 pool; Q and O request rows are converted in place around
the call. Kernel level (runtime pieces, splits on): rel-L2 1.72-1.75e-3 = native (1.67-1.95e-3),
bf16 pool 3.3-3.4e-3, also with K scaled by the 12B global k_norm weight (0.0605); time +10-14%
over the bf16 pool at long prefix. In-model, 16000-token prompt: chunk 2048/1024/512/128 0.549/
0.572/0.677/1.813 s vs native 0.608/0.633/0.758/2.375 (-10/-10/-11/-24%).

FP32 gate vs the same peers: c1 PASS kl_trim 0.0678 (peer 0.0631, native 0.0496) top1 0.9764
(0.9803, native 0.9872) cont 41/46 (43, native 44; limit 41); c8 PASS 0.0680 (0.0678) cont 40 (42);
c32 PASS 0.0618 (0.0710) cont 39 (41). Natural-text KL sits above native and the bf16 pool at every
concurrency (c1 trimmed 0.152 vs 0.114 / 0.123), unexplained: the kernel-level error is native's.
Production mix 2.7/s, 2 reps: TTFT P99 1133.2 ms (1.01x vLLM; native 1338.5, 1.19x), TPOT P99
97.7 ms (native 105.4), goodput 9.13 req/s (native 9.05), report MATCHED + EQUIVALENT.

The natural-text gap is not a numerical defect of the route. Per-site dumps of every hd512 call of
the 15872-token natural case (Q before and after, FP8 K/V rows and scales, O) against an FP32
attention over the same inputs: fp16 pool rel-L2 1.69-1.73e-3 on all 56 sites (native kernel
1.7e-3), bf16 pool 1.7-4.5e-3; the Q round trip changes only subnormal elements (|q| < 8e-6, error
< 3e-8). A sharper softmax (logits x4..x22.6) keeps the fp16 pool at 1.8-2.2e-3 while the bf16
pool degrades to 0.7-2.0e-2. Per case, natural KL moves both ways against native (fp16 better on 9
of 20 cases, worse on 11; e.g. docs-128 0.484 vs 0.018, code-8192 0.210 vs 0.580): a different,
equally accurate rounding re-rolls the knife-edge positions, and the route is batch-invariant, so
the same re-roll repeats at c1/c8/c32.

On prod-next b36fcf18c (KV store copy fixes), same binary and packet, 2.7/s, 2 reps: native TTFT
P99 826.7 ms (0.73x vLLM), TPOT P99 71.1 ms; fp16 route 697.2 ms (0.62x, -15.7%), 62.7 ms (-11.9%),
goodput 9.41 vs 9.39 req/s. The c1 gate is unchanged (deterministic): cont 41/46 against a limit of
41, a pass with no margin; it also meets the gate-relaxation rule at c1 (kl_trim ≤ 1.10x peer,
top1 ≥ peer − 0.005, cont ≥ peer − 2), and the 12B recipe adopted the fp16 route on 2026-10-29.

## 4. Open

* Done: FP32 gate at c1/c8/c32 and production-mix replays for the hd512 route (§3; both recipes).
* FA4 hd256: a Qwen3.8 BF16 full-attention prefill route through an FA4 hd256 library is on branch
  `lib-attn2` (with a geometry check against the packet's declaration), not merged into prod-next.
  Gemma hd256 sliding layers stay native; FA3 C++ would need a separate build.
* Qwen: FA3 decode/prefill wins of 5-11% at kernel level need an FA3 C++ AOT library and a route
  in `gen_decode.rs`; at 16 attention layers this is ~1-3% of a step.

## 5. 2026-10-29: packet-declared per-rung selection

The file-presence selection of section 3 is gone. The packet declares the route; plowrt runs what
the packet declares and nothing else.

* Packet: object id 26 `LIBRARY_PREFILL_ATTENTION` (sha256 + ABI string) and, per prefill program,
  `ProgramRoles.library`: the segments the library serves. Each keeps its native role, which is the
  declared fallback. A decode program cannot declare one. Older plowrt rejects such a packet.
* ABI string: `lib_flash_prefill_v1:fa4:hd=<D>:heads=<H>:kv_heads=<K>[,<K>]:layout=<hm|ilv>:dtype=<bf16|f16>`
  (`plow_asset::LibraryAbi`). The library states its own geometry in the `plow_fa4_caps` marker; the
  emitter reads it without loading the library and plowrt checks it against the declaration at load.
* Emitter: `PLOW_EMIT_ATTN_LIBRARY` is a per-rung knob (`emit.attn_library`). The pass
  (`devgen/src/library_attention.rs`) runs last in both the Gemma and the Qwen pipelines and claims
  every causal global flash-prefill site of the library's head dim on the selected rungs (all or an
  error per program).
* Runtime: `PLOW_PF_ATTN_LIBRARY=0` runs the declared native roles (A/B, rollback). A launch outside
  the library envelope runs the native role, decided before anything is enqueued; fallbacks are
  counted.
* Direct mode (BF16 KV): the library reads the KV pages in place, with no gather and no copy.
  Head-major (Gemma hd512): page id = slot x (kv_heads x stride / 64) + j, head stride = stride x 512,
  rows [kv_len, roundup64) zero-filled. That is safe because a VMM block (pow2 >= 2 MiB, >= 2048 rows)
  is a multiple of the 64-row page. Row-interleaved (Qwen hd256): page id = slot x stride / 64 + j,
  head stride 256, no fill (the non-TMA paged path predicates rows at seqused_k).
* hd256 export (`scripts/export_fa4_hd256_sm90.py`, `build_sm90a_fa4_hd256.py OUT H KVH`): vLLM's own
  SM90 hd256 causal configuration (128 x 80 tiles, two stages, 384 threads, P in registers, intra-WG
  overlap, non-TMA paged KV) plus the hd256 combine; split rule tile_m 128 / tile_n 80.

### Qwen3.8-27B BF16 (hd256, 24 heads over 4 KV heads, row-interleaved)

One 16000-token prompt, chunk 4096/1024/512/128: 1.403/1.517/1.784/4.325 -> 1.386/1.491/1.757/3.996 s
(-1.2/-1.7/-1.5/-7.6%). ctx 4094/4095/8191, 8 x 2048 and 16 x 1000 are ties with no faults. The FP32
gate passes c1/c8/c32, with c1 kl_trim 0.000456 vs native 0.000448. Production mix at 0.45/s:
TTFT P99 1728 -> 1504 ms and TPOT P99 81.3 -> 75.7 ms, ahead in both repeats. Adopted on every
prefill rung.

At kernel level the library wins most at 128 rows with a long past (0.25-0.34x native) and
gives 0.88-0.97x from 512 rows up. Decode stays native. FA4/FA3 decode at B1 with ctx 4-16K runs at
0.35-0.57x native because the native split count is capped at 8. From B8 up it runs at 0.90-1.00x.
A decode route needs the FlashDecode + FlashMerge pair bound in `gen_decode.rs`, with device-side
metadata for graph capture.

### Gemma-4 26B BF16 (hd512, head-major, direct)

One 16000-token prompt, chunk 4096/2048/1024/512/128: 0.404/0.438/0.553/0.857/2.600 ->
0.345/0.381/0.490/0.821/2.273 s (-15/-13/-12/-4/-13%). The boundary sweep is 5-9% faster and
8 x 2048 is 3.5% faster.

The library is as accurate as native at the site. The last global layer was recomputed in FP32
from each arm's own q and cached K/V (read from the cache addresses):

| | c4096 | c1024 |
|---|---|---|
| library rel-L2 | 1.684e-3 | 1.678e-3 |
| native rel-L2 | 1.684e-3 | 1.677e-3 |
| library vs native output | 4.5% | 4.7% |

The library-vs-native output difference matches the drift of its inputs, which accumulates over
the earlier layers. So it is a rounding re-roll, not a wrong read or a dtype conversion; bf16 mode
converts neither Q nor O.

The library's own split count depends on the batch composition, so its logits move between
concurrent runs where native's do not. That is the spread seen in the c16 continuation metric:
0.870-0.891 vs native 0.935 in three runs. At c1, which is deterministic, the library continues
better than native (0.913 vs 0.891).

## 6. 2026-10-29: the library as packet cubins, no host library

The runtime loads no host library for attention. The only dynamically loaded code is the GPU
driver and cuBLASLt. The FA4 forward and combine kernels ship as cubins inside the packet object,
and plowrt launches them itself.

* Object: `plow_fa4_hd512.aot` / `plow_fa4_hd256.aot` (`scripts/fa4_aot/build.py`), a container:
  `PLOWAOT1`, a u32 manifest length, a JSON manifest, then the blobs. The manifest names the
  upstream (vLLM 0.28.0, the sha256 of its `vllm_flash_attn/cute` sources, CUTLASS DSL 4.6.2), the
  host stub it was recorded against (`fa4_sm90_fwd` v1) and the ABI string
  (`lib_flash_prefill_v2:fa4:...`). Each entry (whole, split, combine per KV-head count, and the
  `kv_adapters` cubin) carries its symbol, its parameter layout and its sha256.
* Cubins: extracted from the AOT export's embedded fatbin. They are byte-identical to the cubins
  the former host library loaded.
* Launch record: the export's own host code is linked against capture stubs and run on three
  exemplar shapes (split, long-context decode-like, whole). The record keeps grid, block, dynamic
  shared memory, launch and function attributes, and the parameter bytes. Bytes the host code
  never writes are masked: six captures in separate processes over differently filled stacks.
  The container is byte-reproducible.
* Runtime (`crates/plowrt/src/device/cuda/aot.rs`, `exec/gpu/fa4_sm90.rs`): `cuModuleLoadData`,
  `cuTensorMapEncodeTiled`, `cuLaunchKernelEx`. At load the driver's parameter layout
  (`cuFuncGetParamInfo`) must equal the recorded one. The stub rebuilds every exemplar and the
  object is refused on any difference: config, any built byte, the defined 64 bytes of each tensor
  map, or a masked byte the stub does not declare padding. A cubin whose host logic differs from
  the stub never runs.
* Removed: the host libraries and their build scripts, the `plow_fa4_caps` marker, the v1 ABI
  shorthand, the library's object-build host knobs.
* Two upstream details the runtime reproduces. (1) The upstream host code encodes tensor maps
  itself and sets two flags the driver encoder leaves clear: byte 8 bit 1 always, and byte 10
  bit 5 when the last element's byte offset is at least 128 KiB. (2) The varlen O view's base,
  2^30 rows before `o`, wraps below 2^44, and upstream stores it masked to the 57-bit address
  field. The driver refuses that base, so plowrt encodes at `o` and stores the wrapped base. The
  exemplar check refused the first runtime build until (1) was in place.
* Load and audit: `library prefill attention (FA4) loaded` at load; a
  `library prefill attention launches` line (served, fallback) every 60 s and at unload. The
  inventory item `library_prefill_attention` reports declared segments vs candidate sites. The
  doctor fails a packet whose recipe sets `PLOW_EMIT_ATTN_LIBRARY` but which declares no route.

### Proof: host library vs packet cubins

Same recipes, one emit per route; the cubin arm runs the host-library packet's objects with the
cubin route's packet and object (the two packets differ only in section 26). Declared segments:
12B 96, 26B FP8 60, Qwen3.8 BF16 80. Every run served all launches, 0 fallbacks.

Per block (one attention site: KV adapter, conversions, forward, combine), single-kernel timing
with clocks unlocked, last chunk, median over its sites. Roofline = max(FLOP / 989 TFLOPS,
bytes / 3.35 TB/s) per kernel:

| block | rows / past | host library | packet cubins | roofline | floor / measured |
|---|---|---|---|---|---|
| 12B hd512 FP8 (f16 pool) | 128 / 4096 | 98.2 us | 97.4 us | 30.9 us | 0.32 |
| 12B hd512 FP8 (f16 pool) | 4096 / 0 | 885.7 us | 886.4 us | 402.0 us | 0.45 |
| 26B hd512 FP8 (bf16 pool) | 128 / 4096 | 87.2 us | 87.8 us | 31.1 us | 0.35 |
| 26B hd512 FP8 (bf16 pool) | 4096 / 0 | 756.1 us | 760.3 us | 285.5 us | 0.38 |
| Qwen3.8 hd256 BF16 (direct) | 128 / 4096 | 41.6 us | 41.6 us | 18.4 us | 0.44 |
| Qwen3.8 hd256 BF16 (direct) | 4096 / 0 | 322.2 us | 322.5 us | 208.5 us | 0.65 |

The forward kernel dominates: 26-27% of its floor at 128 rows with split-KV (4-5 splits), 37%
at 4096 causal rows for hd512, 65% for hd256. The 12B f16 pool adds three 43 us conversions at
4096 rows, each at 93% of its bandwidth floor. Gemma's hd256 sliding layers run the native role
in both routes.

Bit identity: the last library site's output (`act.at`) is byte-equal in all six cases. So are
the last-position logits of a 16000-token prompt at chunks 128/1024/4096 (two repeats each) and
of ctx 4094/4095/8191, for all three models. Whole-prompt times agree within run-to-run noise.
Greedy 32-token decode streams after ctx 4094/4095/8191 (FNV digest of the tokens) are equal per
model as well.
