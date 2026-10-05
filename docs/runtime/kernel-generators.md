# Kernel generators in plow: headroom survey (H100, 2026-10-01)

Tools: TileLang 0.1.12, Triton 3.7.1, CuTe DSL 4.6.2. References: FlashInfer 0.6.16 and cuBLASLt
(torch 2.13 / cu130). One H100, every run under `gpulease -n 1`. AMD is paper analysis plus an
offline Triton compile. Base: main dd1be445. Citations to `plans/` point at the gitignored local
plan files.

## Policy

- Never inline generated PTX into the interpreter. It couples registers and stack across arms
  (measured 6-8% attention / +31% render regressions), and Hopper serializes wgmma inside a `.func`
  (C7510).
- Ship generated kernels as standalone role objects, the same path as `SegmentObject` cubins and
  cuBLASLt roles. Do this only where the in-model gain exceeds about 5% of the step.
- Toolchain precedent: Qwen3.5 GDN prefill on sm_90a is already a generated kernel.
  - Generator: FlashInfer CuTe-DSL `_FullyFusedDeltaRuleSm90`, compiled AOT with
    `cute.compile(...).export_to_c` (`scripts/export_gdn_prefill_sm90.py`).
  - Runtime: wrapped in `runtime/nvidia/gdn_prefill.cpp` and loaded as `libplow_gdn_prefill.so`
    (`exec/gpu.rs`, `device/cuda/qwen_gdn.rs`). No Python at runtime.

## Summary

- The headroom is in Gemma-4 12B/31B **prefill attention at 4K-8K**, not in decode.
  - hd512 causal roles run at about 9% of bf16 peak; TileLang reaches 487 TF/s.
  - hd256 sliding roles run at about 8.5%; FlashInfer fa3 reaches 409 TF/s.
  - Both are already separate role launches, so no launch is added.
- The blocker is **numerics policy**.
  - plow's own hd512 WGMMA body ran 2.25x faster and cut TTFT 17-26%.
  - It was rejected because greedy agreement against the bit-exact px4 body was 0/3 at 4K.
  - Generated kernels need a full-logit gate instead of bit-exactness.
- hd512 decode is already close: B=8 ctx 8192 takes 101.3 µs vs FlashInfer 97.5 µs
  (`attention-roofline.md`). Leave it.
- Standalone wins that do not pay off in-model:
  - Fused gate|up+GeGLU decode GEMM: 0-3% faster at M=64, 19% slower at M=128.
  - S3Gen 3-pass GEMMs: 12-17% faster standalone.
  - hd512 decode: 5-20% faster standalone.
  - Pilot 1, E4B decode attention: 1-3% of the step.
- FP8 W8A8 prefill GEMMs are 1.3-1.7x behind cuBLASLt fp8.

## Verdicts

Where an op runs:
- **interp**: an interpreter switch arm.
- **light**: a separate launch of the same body.
- **role**: a `SegmentObject` cubin.
- **Lt**: cuBLASLt.

Moving an op out of the interpreter costs about 2.5-7 µs per launch.

| family | where | gap | expected in-model gain | verdict |
|---|---|---|---|---|
| Flash prefill hd512 causal (12B/31B, E4B) | role 15 `PREFILL_ATTENTION_HD512_PX4_BQ64`; E4B `pffa` | 12B: 3.25 / 11.9 ms per site vs floor 0.28 / 1.11 ms | 12B TTFT −13% (4K), −21% (8K); E4B −1-2% | **pilot A** |
| Flash prefill hd256 sliding | role 14 `PREFILL_ATTENTION_HD256_GQA2_BKV32`; E4B `pffa` | 12B: 0.71 ms per site vs fa3 0.147 ms | 12B 4K −10-14%; E4B −4% | **pilot B** |
| FP8 W8A8 prefill GEMM | roles 13 / 1 | 767-1142 vs cuBLASLt fp8 1336-1473 TF/s | ~−30 ms at 4K (estimate) | **pilot C** |
| Batched decode GEMM, B≥48 | Lt + light glue | the gap is launch count | ≤2%, negative at B=128 | leave |
| Dense decode GEMV | interp | 81-93% of HBM bandwidth | ~0 | leave |
| bf16 prefill GEMM | Lt role 5 | 70-79% of roofline | ~0 | leave |
| MoE grouped prefill / decode | roles 16 / 17 | at the HBM roof / ≈6% of step | <3% | leave |
| Flash decode hd256/512, hd128/64 | interp / light | 5-20% standalone | <1% | leave |
| S3Gen CFM GEMMs | speech interp | Triton bf16×3 12-17% faster standalone | ≤0-3% | hand-port the schedule (BM128 BN64 BK64, 3 stages) |
| Norms / rope / sampling / SNAC / HiFT | interp / light | launch-bound | <3% | leave |
| MLA prefill (AMD GLM) | AMD interp | MI300X at 17% of peak, 50% of TTFT at 32k | large at long context | AMD Triton object (paper) |
| Qwen3.5 GDN prefill (AMD) | refused today | — | unblocks a refused path | AMD Triton (FLA) |

How the 12B gains are computed:
- Per-site times come from `plans/gemma4-4k-8k-native-block.md`, against TTFT 160.2 / 360.7 ms.
- Other schemas measure TTFT at 221.6 / 467 ms, so the percentages shift with the schema.
- No 8K gain is claimed for hd256, because there is no per-site timing at 8K.

## Probes (H100 sm_90a, standalone)

Method:
- Timing: CUDA graph over rotating cold buffers larger than L2.
- Error: rel-L2 vs an fp32 reference (attention) or an fp64 reference (f32 GEMM).

### hd512 causal prefill, one request

TileLang config: BM64 BN64, 1 stage, 256 threads. Times in µs.

| shape | rows | floor | plow v3 | 12B px4 per site | TileLang | FlashInfer fa2 |
|---|---|---|---|---|---|---|
| E4B 8/2 | 1000 | 8.3 | 94.1 | — | **48.1** | 104.6 |
| E4B 8/2 | 2000 | 33.1 | 187.5 | — | **136.6** | 306.8 |
| 12B 16/1 | 4096 | 278 | 1299 | ~3250 | **660-705** | 1738 |
| 12B 16/1 | 8192 | 1112 | 4961 | ~11900 | **2260** | 6755 |

- Generated rel-L2 is 0.9-2.3e-4. FA3 and FlashInfer fa3 have no hd512 kernel.
- plow's rejected WGMMA body ran 1.40 / 5.00 ms.

### hd256 sliding prefill

Times in µs.

| shape | rows | floor | plow | TileLang (no WS) | FlashInfer fa2 | FlashInfer fa3 |
|---|---|---|---|---|---|---|
| 12B 16/8, window 1024 | 4096 | 61 | ~708 per site | 287 | 237 | **147** |
| 12B | 8192 | 130 | — | 564 | 455 | **314** |
| E4B 8/2, window 512 | 1000 | 3.2 | 37.7 | 36.9 | 40.1 | **23.2** |
| E4B | 2000 | 7.4 | 40.4 | 55.8 | 36.6 | **25.0** |

### Fused gate|up + GeGLU decode GEMM

Shape: N = 2×10240, K = 2560. Floor 31.3 µs. Times in µs.

| M | cuBLAS | cuBLAS + GLU launch | Triton fused |
|---|---|---|---|
| 64 | 39.9 | 42.4 | 40.9 |
| 128 | 42.7 | 46.5 | 55.3 |

### FP8 GEMM at M=4096

Throughput in TF/s.

| shape | plow | cuBLASLt fp8 (per-tensor) | ratio |
|---|---|---|---|
| q | 877 | 1358 | 1.55 |
| kv512 | 589 | 1201 | 2.04 |
| kv2048 | 813 | 1292 | 1.59 |
| gate/up | 871 | 1422 | 1.63 |
| down | 1142 | 1473 | 1.29 |
| o | 767 | 1336 | 1.74 |

Scale granularity differs (plow vs per-tensor), so treat these ratios as an upper bound.

### hd512 decode

Times in µs.

| B / ctx | plow | TileLang | FlashInfer |
|---|---|---|---|
| 1 / 8192 | 26.5-31.6 | 21.4 | 24.3 |
| 8 / 8192 | 101.3 | 94.8 | 98.0 |
| 32 / 8192 | 376 | 347.6 | 351.7 |
| 8 / 1024 | 20.7 | 17.5 | 21.3 |

### S3Gen GEMMs, 8704 rows, f32

Times in µs.

| GEMM | plow wgmma 3×bf16 | Triton bf16×3 | cuBLAS f32 |
|---|---|---|---|
| qkv | 98 | 81.3 | 218 |
| ff1 | 91 | 75.4 | 156 |
| ff2 | 81 | 71.3 | 161 |
| out | 48 | 41.9 | 83 |

Single-pass is ruled out:
- 1-pass tf32 rel-L2 is 5e-4.
- A single bf16 pass moved mel by 3e-3 (`tts.md`).

## AMD

- **Generators:** CuTe DSL and FlashInfer are NVIDIA-only, and the TileLang ROCm backend is
  untested. Triton 3.7.1 is the realistic choice: it compiled an MFMA GEMM to gfx942 / gfx950
  `.hsaco` offline.
- **Loading:** plow already dispatches external code objects (Tensile, AITER). A Triton `.hsaco`
  would only need its kernarg layout pinned.
- **Candidates:**
  - MI300X GLM MLA prefill.
  - Qwen3.5 GDN prefill (Triton FLA `chunk_gated_delta_rule`).
- **Leave:**
  - gfx950 FP8 prefill GEMM: a tile-inventory fix, not a generator.
  - K3 routed DOWN: no FP4 MFMA on gfx942.
  - Decode: launch and protocol bound.

## Generated-kernel catalog (pilot A, 2026-10-02)

Generated kernels ship as packet role objects keyed by op signature, never by model name.

- **Table:** `tuning/nvidia/sm_90a/h100-sxm5/gen_kernels.json`. Each entry records its
  signature, the chosen config, the generated body's sha256, the generator version, and, per
  shape class, every config's µs and rel-L2 vs an fp32 reference.
- **Build:** `scripts/gen_kernels/build_catalog.py build OUT --entries a,b` regenerates the
  tuned config with TileLang and compiles it with nvcc into the wrapper
  `runtime/nvidia/gen_flash_prefill.cu`. It needs no GPU. It fails on generator drift, i.e.
  when the body digest differs from the table. `tune` (GPU, under `gpulease`) is the only
  command that rewrites the table; `bench` measures built objects.
  - The objects script runs `build` when `PLOW_BUILD_GEN_KERNELS` lists entries.
  - The recipe lists the cubin in `role_files`.
  - The role emit sets `PLOW_EMIT_GEN_KERNELS` to the same list.
- **Roles:** IDs 18..=25, one per entry. The ABI is
  `gen_flash_prefill_v1:<entry>:block=<n>:smem=<bytes>`, plus a sha256 pin and the attention
  capability. The grid is the packet grid: one persistent CTA per SM.
  - The wrapper owns the packed request table, the KV slot layout, padded-row zeroing and
    successor counters.
  - plowrt loads every entry through one path (`exec/gpu/gen_role.rs`), using the 128-byte
    direct ABI when requests are packed and the packet entry otherwise.
  - The vendor-GEMM attention route skips segments that carry a generated role.
- **Devgen:** `gen_kernels.rs` binds prefill ops matching an entry's head width and window on
  rungs of at least `min_rows`, before the hand-written roles 6/14/15 run. Those roles skip
  what a generated entry took.
- **Default off:** the 12B emit with the knob unset is byte-identical to HEAD (model.pkt sha256
  `b4ac7a1d…` from HEAD plowc, the worktree plowc and the campaign build alike).

### attn_pf_hd512 (TileLang BM64 BN64, 1 stage, 256 threads, cp.async; TMA/WS off)

Standalone, 12B heads 16/1, one request, packet-shaped buffers. Times in µs.

| rows | floor | px4 role 15 | generated | rel-L2 (gen / px4) |
|---|---|---|---|---|
| 1024 | 17.4 | 245.8 | **86.3** | 2.2e-4 / 2.2e-4 |
| 4096 | 278 | 2405 | **906** | 1.3e-4 / 1.3e-4 |
| 8192 | 1112 | 9210 | **3334** | 8.1e-5 / 8.1e-5 |

- A packed check covers three chunked requests over three slots with a zeroed tail: rel-L2
  2.6e-4.
- The sweep tried BN 16/32/64 and 1-3 stages. BN64 with 1 stage won every class. The
  table-selected object spills 84 B, a cost of the persistent loop.
- TileLang's own fast path (TMA + warp specialization, about 720 µs at 4K) needs per-request
  tensor maps (`__grid_constant__` descriptors). That is the next ABI step.

In-model, Gemma-4 12B, realtime profile, C1, ABAB, 2 reps, 32 prompts per cell. TTFT in ms.

| in | off (vendor-GEMM route) | gen | Δ | route off (WG32 role 6) |
|---|---|---|---|---|
| 1024 | 45.55 / 45.45 | 45.40 / 45.30 | −0.3% | 46.54 |
| 4096 | 169.41 / 169.18 | 169.67 / 169.03 | 0.0% | 176.11 |
| 8192 | 351.23 / 350.66 | 354.12 / 353.99 | +0.9% | 377.16 |

- The production baseline is the vendor-GEMM route (`PLOW_PF_ATTN_GEMM`, on when
  `attn_softmax_sm90a.cubin` ships), not px4.
- `PLOW_PF_SEG_TIME` per hd512 site:
  - First 4096-row chunk: route 0.94 ms vs generated 0.99 ms.
  - Second chunk (kv 8192): route 2.33 ms vs generated 2.85 ms.
- The generated object beats the hand-written WG32 role by 3-6% but not the route.
- Logit gate vs HF bf16 (`gemma_logit_parity.py`; standard prompts mostly below 1024 rows):
  - Generated: top1 0.9848, KL mean 9.2e-3.
  - Off: top1 0.9821, KL mean 9.2e-3.
  - Both meet top1 ≥ 0.98. Both miss the E4B KL bound of 2e-3, which the 12B baseline
    already misses.
- Long prompts (1.2K / 4.1K / 8.3K):
  - Top1 51/52, 64/64, 58/64 for generated vs 51/52, 64/64, 63/64 for off.
  - All cases: 0.9651 vs 0.9785.
- Gemma-4 26B-A4B (c1-lean recipe), same protocol. TTFT in ms, off vs gen:
  - 1024: 37.37 / 37.19 vs 36.97 / 37.00 (−0.8%).
  - 4096: 101.86 / 101.84 vs 101.63 / 101.62 (−0.2%).
  - 8192: 207.89 / 207.81 vs 209.08 / 209.16 (+0.6%).
  - No HF reference loads on the box (the checkpoint index names missing shards). Gen vs off
    served greedy outputs are identical on 9/9 gate prompts, including the 1933-token one.
- **Verdict:** the entry stays opt-in. It fails the >5% TTFT rule against the shipped route.
  The catalog infrastructure ships default off.

### Adding an entry (e.g. `attn_pf_hd256_sliding`)

1. Add `scripts/gen_kernels/catalog_<name>.py` exporting `ENTRIES` (TileLang body builder,
   signature, object name, sweep, shape classes). build_catalog.py imports every
   `catalog_*.py`. Body contract:
   - grid `(ceildiv(qlen, BM), heads)`;
   - params by name: `Q K V O heads qlen kvlen scale [window kv_mask]`;
   - K/V are one KV head's rows, with position p at row `p & kv_mask`;
   - no `blockIdx.z`, `gridDim` or TMA.
2. Add a row to `CATALOG` in `crates/devgen/src/gen_kernels.rs`: next role ID, object name,
   head width, window, ring KV, `min_rows`. The row for `attn_pf_hd256_sliding` (role 19,
   window 1024, ring KV) is already registered.
3. Run `build_catalog.py tune --entries <name>` under `gpulease -n 1`, then build a packet
   with the three switches above and gate it.

### attn_pf_hd256_sliding (hand CUDA FA3-style, 384 threads, TMA/cp.async + wgmma)

- Kernel: `runtime/nvidia/gen_attn_pf_hd256_sliding.cu` plus a generated wgmma include
  (`scripts/gen_kernels/attn_pf_hd256_sliding.py wgmma-inc`). WG0 produces (TMA from the op's
  GEN_TMAP_KV_PAIR, cp.async when absent). WG1/WG2 each own one head of a GQA pair over 64
  query rows: QK is SS m64n64k16, PV is RS m64n256k16 with P from registers, QK(t) overlaps
  PV(t-1). Window, ring mask, packed requests, padded-row zeroing and successor counters are
  in-kernel. The harness and catalog hooks are in `catalog_hd256.py`.
- Signature: hd256, any nonzero window (`ANY_SLIDING`), even GQA ratio (`pair_heads`), ring
  KV, rungs ≥ 1024 rows. One object serves 12B/26B (16/8, w1024) and E4B (8/2, w512).
  plowrt now passes the op's t7 KV tensor-map pair as `mapkv`; the hd512 object ignores it.
- Tuned: BN64, K ring 3, V ring 2, producer 56 / consumer 224 regs (sweep in the table).

Standalone, one request, cold buffers, TMA path, µs. rel-L2 vs fp32 is 2.0-2.2e-3 for both
the generated object and role 14 (bf16 output rounding).

| shape | floor | role 14 | FlashInfer fa3 | generated |
|---|---|---|---|---|
| 12B 1024, w1024 | 8.7 | 89.3 | 31.3 | 38.2 |
| 12B 4096, w1024 | 60.8 | 353 | 143 | 146.8 |
| 12B 8192, w1024 | 130.3 | 692 | 313 | 272.1 |
| E4B 1000, w512 | 3.2 | - | 23.2 | 26.7 |
| E4B 2000, w512 | 7.4 | - | 25.0 | 30.3 |

The harness checks 11 cases in both TMA and cp.async modes: packed with padding, ring wrap,
window longer than the prompt, linear KV, GQA 4/2, peaky logits. All pass. The packed table
check gives rel-L2 2.05e-3.

In-model, Gemma-4 12B, realtime profile, C1, ABAB, 2 reps, 32 prompts per cell. TTFT in ms.

| in | off | generated | Δ |
|---|---|---|---|
| 1024 | 45.77 / 45.46 | 42.18 / 42.12 | −7.6% |
| 4096 | 168.90 / 168.18 | 161.38 / 161.15 | −4.3% |
| 8192 | 351.02 / 349.05 | 334.84 / 334.98 | −4.3% |

- What actually runs in the shipped 12B packet: the vendor-GEMM route takes only the 8 hd512
  global sites. Sliding sites run role 14 on the 4096 / 8192 / wide rungs and the FA256
  segment body below that.
- `PLOW_PF_SEG_TIME` per sliding site (40 per chunk):
  - 4096-row chunk: role 14 0.395 ms vs generated 0.158 ms.
  - Second chunk (kv 8192): 0.435 ms vs 0.174 ms.
  - Sliding attention is about 16 of 168 ms at 4096, so even a free kernel caps the gain near
    9%.
- Logit gate vs HF bf16:
  - Standard prompts: generated top1 0.9821, KL mean 9.3e-3; off top1 0.9872, KL 9.2e-3.
  - Long prompts (1.2K / 4.1K / 8.3K): top1 51/52, 64/64, 61/64 generated vs 51/52, 64/64,
    63/64 off. All cases: 0.9785 vs 0.9839.
  - On the steps before greedy diverges from HF, the generated arm's max |Δlogprob| is equal
    or lower (long chat0 0.34 vs 0.40, chat2 step 0 0.25 vs 0.44; gate chat4 0.38 vs 0.45).
    The top1 gap comes from different continuations after divergence.
- Default-off emit is byte-identical (model.pkt `b4ac7a1d…`).
- **Verdict:** the entry stays opt-in. It is 2.4x faster than role 14 per site, but 12B TTFT
  at 4096/8192 improves 4.3%, short of the >5% rule.

### FP8-KV entries (`kv_dtype` in the signature)

The catalog signature carries the KV dtype (`KvDtype` in `gen_kernels.rs`): a bf16 entry binds
only `FlashPrefill`, an FP8-KV entry only `FlashPrefillFp8` (e4m3 K/V, f32 scale per
(position, KV head) row, `t6`/`t7`). FP8-KV objects export `plow_gen_flash_prefill_abi = 2`
(ABI family `gen_flash_prefill_fp8kv_v1`): the scales ride the direct ABI's opart/mlpart slots,
and the packed request table rides the op's `i[4]` handle (bit 31).

- `attn_pf_hd256_sliding_fp8kv` (role 20): the hd256 kernel with an FP8 producer (cp.async of
  raw e4m3 rows, exact e4m3 → bf16 into the swizzled stage); k_scale on S before masking,
  v_scale on P before PV, both fp32. BN64, K/V rings 2+2, producer 72 regs, 0 spills. The packet
  arg block lives in dynamic smem (static + dynamic would exceed 232448 B). rungs ≥ 128.
- `attn_pf_hd512_fp8kv` (role 21): TileLang body over e4m3 K/V (`flash_prefill_body_fp8kv`),
  wrapper `gen_flash_prefill.cu` with `PLOW_GEN_FP8_KV=1`. BM64 BN32, 1 stage, 0 spills.
  rungs ≥ 128.

Standalone vs FP32 over the dequantized cache (rel-L2), µs:

| entry | 1024 | 4096 | 1024 after 7K | 1024 after 14K | packed check |
|---|---|---|---|---|---|
| hd256 sliding FP8-KV | 64 (2.4e-3) | 240 | 71.7 | - | 2.3e-3 |
| hd512 global FP8-KV | 158.6 (1.4e-3) | 1962 (1.2e-3) | 1828 (2.4e-3) | 3444 (2.4e-3) | 1.9e-3 |

The interpreter FP8 arms it replaces: hd256 px23 178 µs at 1024, packed 16×128 363-1423 µs vs
generated 137-142 µs. TileLang in the nix shell needs a host g++ ≤ 14 (`-ccbin=/usr/bin/g++-14`
in `NVCC_APPEND_FLAGS`; the system g++ 15 fails in `cuda_fp16.h`).

## Gemma-4 12B FP8 per-rung routes (H100, FP8 KV, 16K, 128 slots)

Recipe `recipes/infervisor/gemma-4-12b/sm90a-h100-tp1.toml`; control = the same packet
without the generated roles, the decode defines and with the 64 rung. `step_bench`, 2 reps
(agree within 0.3%). Floors: `op_roof` at 3210 GB/s / dense peak, rows-linear for packs
(approximate).

Prefill, packed wall in s:

| B×ctx | control | hd256 gen | **hd256 + hd512 gen** | % floor (chosen / control) |
|---|---|---|---|---|
| 1×1024 | 0.0753 | 0.0489 | **0.0366** | 44 / 22 |
| 1×4096 | 0.4896 | 0.3349 | **0.1509** | 44 / 13 |
| 1×15000 | 3.838 | 3.227 | **0.719** | 34 / 6.4 |
| 4×1024 | 0.3474 | 0.2479 | **0.1970** | 33 / 19 |
| 32×128 | 0.2904 | 0.1929 | **0.1647** | 40 / 23 |
| 128×128 | 1.168 | 0.7718 | **0.6598** | 40 / 23 |
| 16×512 | 0.612 | 0.4524 | **0.3872** | 34 / 22 |
| 128×1024 | 11.12 | 7.918 | **6.294** | 34 / 19 |

Decode, ms per step (B=64 runs on the 128 rung once the ladder drops 64):

| ctx | B=1 | B=8 | B=32 | B=64 | B=128 (% floor) |
|---|---|---|---|---|---|
| 1024 | 8.54 / 8.84 | 9.73 / 10.34 | 15.71 / 17.36 | 20.17 / 139.6 | 27.21 / 40.5 (44%) |
| 4096 | 8.61 / 8.89 | 10.78 / 11.52 | 18.94 / 21.10 | 27.07 / 146.9 | 40.65 / 55.2 (32%) |
| 16000 | 8.83 / 9.15 | 13.79 / 14.87 | 32.12 / 35.82 | 52.57 / 175.0 | 91.98 / 111.1 (19%) |

(chosen / control.) Chosen routes:

- Prefill projections: unchanged cuBLASLt FP8 / native split (`segment_roles::cublaslt_prefill_fp8`
  route matrix).
- Prefill attention: generated FP8-KV hd256 + hd512 on every rung ≥ 128; runner-up the
  interpreter FP8 arms (above).
- Decode: light FP8 attention (16-byte loads, direct V, segmented FlashDecode grid), cuBLASLt
  head at B=128, ladder without 64 (its TC64 GEMV was 3-7x slower than the 128 rung).

Correctness:

- Packed `--same` runs agree on every slot (4×1024, 16×512, 2×4096) in every arm.
- The decode-only packet reproduces the control's prefill digests exactly.
- FP32-reference gate (`fp32_ref_gate.py` with `ignore_eos`, edbda0d4), served, prefix cache
  off, peer = the cached vLLM 0.28 captures: **PASS** against both vLLM repeats. All 1172
  positions scored. Candidate vs vLLM: kl_mean 0.109 vs 0.128, kl_p99 2.50 vs 3.19,
  top1_decisive 0.984 vs 0.982, cont_frac 0.650 vs 0.574, needle_acc 1.0 vs 1.0.

Served A/B (`vllm bench serve`, greedy, OSL 128, prefix cache off on both arms, ABAB). Output
tok/s, TTFT p50 s, TPOT p50 ms; candidate reps 1 / 2, control rep 1. Control rep 2 matched rep 1
within 0.1% at 4K c32; its other cells were cut to free the shared GPU.

| cell | candidate | control |
|---|---|---|
| 4K c32 | 381.3 / 381.5 tok/s, 0.87 s, 73.4 ms | 38.3, 12.1 s, 758 ms |
| 4K c128 | 385.4 / 385.5, 21.7 s, 160.2 ms | 18.7, 487 s, 3734 ms |
| 15K c32 | 94.6 / 94.6, 19.8 s, 156.6 ms | 7.3, 205 s, 2154 ms |
| 15K c128 | 95.7 / 95.7, 82.6 s, 172.9 ms | 6.2, 1256 s, 2904 ms |

The control's FP8-KV packed prefill (interpreter arms, serial over requests, decode rows riding
the pack) is what made this packet unservable at ≥ 4K. The generated roles deal every
request's tiles across all CTAs, riders included.

### Native FP8 GEMM object and request chunk 4096 (2026-10-04)

The native rows of the route matrix (down_proj at M = 1088 and ≥ 2048, the 2112/4160
exceptions) were measured on the ws384 body, but an FP8-KV packet loads
`interp_sm90a_pfpackedgemm_fp8kv.cubin`, which the segment script never rebuilt: those
projections ran the base emit's generic GEMM (128 regs, 1.8 KB stack). down_proj at M = 4096
took 1.76-2.34 ms there vs 0.37 ms on ws384 (cuBLASLt 0.40 ms; route bench, cold weights). That
was the whole per-row loss of the 2048/4096 rungs. `PLOW_BUILD_FP8KV_GEMM=1` builds the ws384
object for it (160 regs, no stack; same ABI symbols); the route matrix itself is unchanged.

`step_bench`, one request of M rows on the request-chunk-4096 packet, first chunk (prefix 0), ms
and µs/row, and `PLOW_PF_SEG_TIME` classes gemm / light / attention ms (generic → ws384). 4160 /
4224 are 4096 slices plus riding decode rows; a single request caps at 4096, so they share the
4096 kernels and were not timed separately.

| M | generic ms (µs/row) | ws384 ms (µs/row) | generic classes | ws384 classes |
|---|---|---|---|---|
| 1024 | 35.8 (34.9) | 35.6 (34.8) | 0.8 / 15.9 / 20.0 | 0.0 / 16.0 / 20.0 |
| 1088 | 67.4 (62.0) | 50.5 (46.4) | 26.8 / 21.6 / 19.9 | 8.8 / 21.6 / 20.1 |
| 1152 | 46.8 (40.6) | 46.8 (40.6) | 0.7 / 22.0 / 24.8 | 0.0 / 21.9 / 24.6 |
| 2048 | 102.7 (50.1) | 69.1 (33.7) | 43.4 / 32.1 / 27.9 | 9.3 / 32.1 / 27.8 |
| 2112 | 123.0 (58.2) | 78.5 (37.2) | 58.6 / 36.4 / 28.5 | 13.7 / 36.3 / 28.4 |
| 4096 | 212.6 (51.9) | 145.9 (35.6) | 84.9 / 73.0 / 54.6 | 18.1 / 73.0 / 54.3 |

One 15,872-token request (3 reps, warm): request chunk 1024 control 756 ms (with the ws384
object 758 ms: its only native rung, 1088, does not occur here), request chunk 4096 staged 745
ms (generic object 1012 ms). Every arm emits the same first token and token-stream digest. The
light ops (norm + quant, `interp_sm90a_pfpackedseg_fp8kv.cubin`, also the base generic body,
not FATLITE) run at ~20% of the HBM floor and are the next per-row cost; the projections run at
1.1-1.3 PFLOP/s.

FP32-reference gate, request chunk 4096: **PASS**, kl_mean 0.110 vs vLLM 0.128, kl_p99 3.0 vs
3.19, top1_decisive 0.980 vs 0.982, needle_acc 1.0.

Served (`llm_grid.sh`, REPS=2, one server per cell, same binary; recipe serve env, interleave
2048). Request chunk 1024 recipe as it was (generic GEMM object) → request chunk 4096 staged
with the ws384 object; output tok/s, TTFT p99 ms, TPOT p99 ms:

| cell | chunk 1024 | chunk 4096 |
|---|---|---|
| agentic16k c32 | 495.5, 5213, 76.5 | 581.4, 3708, 64.1 |
| 4K c32 | 404.0, 7966, 74.3 | 548.0, 5464, 54.6 |
| 4K c128 | 442.9, 34306, 139.2 | 618.1, 23921, 98.8 |
| 15K c32 | 107.9, 35165, 155.6 | 144.6, 25941, 114.7 |
| 15K c128 | 108.6, 144666, 156.4 | 146.1, 106476, 115.7 |

The recipe takes `PLOW_MAX_REQUEST_CHUNK = "4096"` and `PLOW_BUILD_FP8KV_GEMM = "1"`.

## Next pilots (ranked)

**A. hd512 causal prefill, generated role object, replacing role 15.**
- Add a new ABI string next to `PREFILL_ATTENTION_HD512_PX4_BQ64` in `segment_roles.rs`.
- Extend the role symbol table and `check_attention_hd512_role` in `exec/gpu.rs`.
- Build AOT: TileLang → cubin, or CuTe `export_to_c`.
- The kernel must add ring-indexed KV, the packed request table, and successor-counter publishing.
- Expected: 12B/31B TTFT −13% at 4K and −21% at 8K.
- Gate: a full-logit gate agreed before work starts.

**B. hd256 sliding prefill at FA3 class (role 14 plus the E4B `pffa` segment).**
- Use the FlashInfer fa3 template, or warp-specialized TileLang / CuTe.
- Expected: 12B 4K −10-14%; E4B −4%.
- Gate: same as A. The BKV64 promotion previously failed its checksums.

**C. FP8 W8A8 prefill GEMM (roles 13 / 1).**
- Use a CuTe-DSL persistent FP8 GEMM with a scale + GeGLU epilogue, or call cuBLASLt fp8 directly.
- Expected: GEMMs 1.3-1.7x faster. Re-measure on the current schema first.

**Alternate for an AMD window:** Triton GDN prefill behind an adapter shaped like `plow_gdn_run`.

The probe scripts are local, not committed, in `/opt/dlami/nvme/lava-tts/gen-survey/`:
- `pf512.py`, `pfgen.py`: prefill attention.
- `gemm_probe.py`, `fp8_probe.py`: GEMMs.
- `dec_grid.sh`: hd512 decode.
- `amd_compile.py`: the offline gfx942 / gfx950 compile.
