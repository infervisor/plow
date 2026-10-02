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
