# 29 — Techniques: what each op does, how a packet declares it, what runs it

**Status: reference (2026-10-29, checked against prod-next 8c61b93b8).** One section per technique
plowrt and devgen implement. Several models share a technique, so models appear only as examples.
Each section has the same fields:

- **What it computes**: the op, model-agnostic.
- **Examples**: networks that use it (devgen network ids).
- **Packet ABI**: device opcodes (`DevOp` in `crates/packet/src/dev.rs` = `PLOW_DOP_*` in
  `runtime/common/dev_isa.h`), packet sections (`crates/plow-asset/src`), tensor classes, and the
  emit knobs that switch it.
- **Kernels and objects**: device kernels per vendor and the object file that carries them.
- **Emitter**: the devgen module that lowers it.
- **KV / state class**: the per-request memory it owns, and how the runtime and the KV block
  store (`crates/plowrt/src/memory/kv_store.rs`, `exec/gpu/kv_blocks.rs`) manage it.
- **Status**: the backends that implement it, and their limits.
- **Deeper**: the chapters with the full design.


## 1. Attention

### Causal GQA flash attention (prefill, split-KV decode, FP8 KV)

What it computes: softmax(Q·Kᵀ·scale)·V per query row over the causal prefix, with `n_head` query
heads sharing `n_kv_head` K/V heads (GQA). Prefill tiles query rows FA-2 style; decode is one query
row per sequence, split over the KV length into `nsplit` partials (f32 `Opart` + running max/sum
`mlpart`) combined by a merge. FP8 KV stores e4m3 K/V with one f32 scale per (position, KV head)
row; the kernels dequantize at the staging step (`crates/packet/src/dev.rs` ops 38/39).
Examples: every dense LLM network (`llama`, `qwen3`, `gemma4`, `gpt_oss`, `nemotron`).
Packet ABI: `DevOp::FlashPrefill` (11), `FlashDecode` (12), `FlashMerge` (13),
`FlashDecodeFp8` (38), `FlashPrefillFp8` (39) (`PLOW_DOP_FLASH_*`, `crates/packet/src/dev.rs`;
`runtime/common/dev_isa.h`). Prefill `i0..i7 = n_q n_kv n_head n_kv_head q_pos0 window hd nsplit`,
`f0 = scale`, `j0 = kv_stride`, `j1 = kv_mask`; decode `i3 = kv_stride`, `i5 = nsplit`, `i6 = hd`,
`t5 = kv_len`, `t6 = decode_slot` (compact row -> physical slot). FP8 twins add `t6/t7 = k/v scale`
(`FlashPrefillFp8` `i7 = nsplit | span << 16`: bidirectional span for media items). KV geometry is
declared in `live_kv` (`Cache { heads, hd, stride, window, mask, scales, layout }`,
`crates/plow-asset/src/live_kv.rs`) and `serve.json` `kv` (`KvGeometry`). Emit knobs:
`PLOW_FP8_KV`, `PLOW_FP8_KV_FULL` (FP8 cache on all / full layers), `PLOW_EMIT_GEN_KERNELS`
(generated roles); runtime `PLOW_GEN_DECODE_STEP` (rows from which generated decode runs).
Kernels and objects: NVIDIA interpreter arms in `runtime/nvidia/op_attention.cuh`
(`d_flash_prefill*`, `d_flash_decode*`, `d_flash_merge`, packed `_mux` forms),
`op_attention_sm90.cuh` / `_sm90_v3.cuh` (wgmma prefill), compiled into `interp_sm90a.cubin`
(`interp_sm120.cu`, also included by `interp_sm89.cu`, for sm_120/sm_89) and the per-geometry
prefill segment objects (`interp_sm90a_pfattn_hd256_*.cu`, `_hd512.cu`). Generated roles
(`scripts/gen_kernels/build_catalog.py`; `gen_flash_prefill.cu`, `gen_flash_decode.cu`,
`gen_attn_pf_hd256_sliding*.cu`) ship as `gen_sm90a_attn_*.cubin`, declared in
`segment_roles.json` / `decode_objects.json` and launched by `exec/gpu/gen_role.rs`,
`gen_decode.rs`, `riders.rs`. AMD: `runtime/amd/flash.hip`, `op_attention_gfx942.h` /
`_gfx950.h`. CPU: `runtime/cpu/flash.c`, `runtime/cpu/dev/avx512/attention.c`.
Emitter: per-network attention lowering in `crates/devgen/src/lib.rs` (shared by qwen35, gptoss,
muse, mtp); generated-role selection `gen_kernels.rs`, `attention_prefill_role.rs`.
KV / state class: flat per-slot KV `[slot][kv_head][stride][hd]` (head-major) or
`[slot][stride][head][hd]` (`LAYOUT_ROW_INTERLEAVED`); full-attention layers grow with the live
frontier under live KV (4096-row commit blocks); FP8 scales are a parallel `[slot][head][stride]`
f32 tensor. In the block store a boundary's full-layer rows and scales are `Class::Tail` 32-row
prefix blocks (`exec/gpu/kv_blocks.rs`). Kernels must not load rows at or past `kv_len`.
Status: CUDA sm_90a (interpreter, segments, generated roles), sm_120/sm_89 (interpreter),
AMD gfx942/gfx950, CPU reference/AVX-512. Prefill `hd` must be 256 or 512 in the interpreter arm
(`dev.rs` op 11); other head dims go through generated or library roles (not verified per hd).
Deeper: `22-kv-cache.md` §1, §4, §5; `16-object-contract.md`; `17-unified-token-batch.md`;
`21-decode-context-parallelism.md` (merge across ranks, `XFlashMerge`).

### Sliding-window attention and hybrid sliding/global layouts

What it computes: a windowed layer attends only the last `window` positions (`window > 0` in the
flash ops; `0` = full causal). A hybrid model interleaves sliding layers with global (full)
layers, often with different head dims and KV head counts per class.
Examples: `gemma4` (sliding hd256 + global hd512, window 512/1024), `gpt_oss` (window 128 on even
layers, full on odd: `crates/devgen/src/gptoss.rs`).
Packet ABI: the same flash ops with `window` set (`FlashPrefill i5`, `FlashDecode i4`) and a ring
mask (`j1 = kv_mask` / `kv_mask` = ring − 1). Per-cache geometry in `live_kv` `Cache.window`,
`Cache.mask`, `Cache.stride`; `serve.json` `KvGeometry` lists `full_layers` / `slide_layers` with
their own `kv_heads_*`, `head_dim_*` and `window`. The `kv_ring.v1` Lean obligation
(`crates/plow-asset/src/kv_ring.rs`) checks that no packed launch writes two spans on one ring slot.
Kernels and objects: as causal flash; sliding-specific roles `gen_attn_pf_hd256_sliding.cu` /
`_fp8kv.cu` and the hd512 global prefill object (`interp_sm90a_pfattn_hd512.cu`) or the library
route below. Prefill segments are classed by head dim (`crates/plowrt/src/asset/devblob.rs`
segment classing of hd512 `FlashPrefill`).
Emitter: `kv_ring` / `kv_ring_rows` in `crates/devgen/src/lib.rs`: a sliding ring holds
`next_pow2(window + chunk − 1)` rows; full layers hold `ctx`. `appended_rungs` admits prefill
rungs that still fit the ring.
KV / state class: sliding ring per slot, sized by the prefill chunk, not the context (addressed
`row & kv_mask`); full layers as above. Block store: the last `window` ring rows and their FP8
scales are `Class::Window` 32-row blocks; full-layer rows are `Class::Tail`. The VMM prefix
cache auto-selects only the qualified hybrid geometry (hd_full 512, hd_slide 256, window 512/1024,
BF16/FP8 KV, no recurrent state: `exec/gpu/prefix.rs::select_vmm_prefix_layout`); others need
`PLOW_VMM_PREFIX=1`. Live rings (`PLOW_VMM_LIVE_RINGS`) commit ring backing on first use.
Status: CUDA sm_90a, AMD, CPU through the common flash ops.
Deeper: `22-kv-cache.md` §1–§2, §8; `26-kv-block-store.md`; `13-prefill-chunking.md`.

### KV-shared layers (KV-shared tail)

What it computes: the trailing `num_kv_shared_layers` layers write no KV and attend the KV cache
of an earlier layer of the same attention class. From the first shared layer on, every op is
row-local except attention, so only sampled rows need computing.
Examples: Gemma-3n / Gemma-4 E-series (`gemma4` E4B: `crates/devgen/src/config.rs` `kv_shared`).
Packet ABI: section `kv_shared_tail` (`crates/plow-asset/src/kv_shared_tail.rs`):
`boundaries[program]` = first instruction of the tail in each prefill program (must open a
segment) and `carried` = the activations the tail reads (`[rows][row_bytes]`). The shared layers'
flash ops name the source layer's K/V tensors.
Kernels and objects: no new kernel; the runtime gathers the sampled rows' `carried` activations into
rows `0..n` and runs the smallest bucket from its boundary (`crates/plowrt/src/exec/gpu/shared_tail.rs`,
CUDA token-batch route), decode rungs likewise (`DecodeTail`).
Emitter: `kv_shared_tail_section` in `crates/devgen/src/lib.rs` (packed-prefill packets whose every
prefill program marked the boundary).
KV / state class: none of its own; reads the source layers' slot KV / rings. The source rings must
hold what every sharer reads (block-store window blocks cover them).
Status: CUDA (token-batch route); other backends run the full program (not verified per backend).
Deeper: `17-unified-token-batch.md`, `26-kv-block-store.md` (E4B kv-shared notes).

### Attention sinks and logit softcap

What it computes: sinks add one learned, unscaled logit per head to the softmax denominator with
no value row: `gm' = max(gm, sink_h)`, `gl' = Σ l_s e^(m_s−gm') + e^(sink_h−gm')`. Softcap is
`cap · tanh(x / cap)` (`DevOp::SoftCap`, op 7, `f0 = cap`).
Examples: sinks: `gpt_oss` (`self_attn.sinks`). Softcap: final-logit softcap of `gemma4`
(`final_logit_softcapping`, cap 30), `muse_spark`, the DSpark drafter (`dspark.rs`). Attention-
logit softcap is refused (`config.rs`: "attention softcap unsupported").
Packet ABI: sinks are `FlashMerge t3` (bf16 `[n_head]`); with sinks the emitter emits the merge at
every `nsplit` including 1 and leaves `FlashPrefill t5` (`O_final`) unset (`dev.rs` op 13).
Final softcap is a `SoftCap` op on the logits tile, or folded into the fused lm_head + argmax
epilogue (`PLOW_FUSE_ARGMAX`, `d.f[0] = softcap`, `crates/devgen/src/lib.rs`).
Kernels and objects: `d_flash_merge` (interpreter objects), `d_softcap` in the elementwise arms.
Emitter: `crates/devgen/src/gptoss.rs` (sinks), `lib.rs` lm_head tail, `muse.rs`, `dspark.rs`.
KV / state class: none.
Status: wherever `FlashMerge` / `SoftCap` run (all interpreter backends).
Deeper: `17-unified-token-batch.md` §4.4 (MXFP4 + sinks).

### Library attention route

What it computes: the same causal/global attention as `FlashPrefill(Fp8)`, executed by a vendor
library kernel instead of the packet's native role. Today: FlashAttention-4's SM90 forward for
hd512 global prefill layers with FP8 KV.
Examples: `gemma4` 12B (fp16 pool) and 26B FP8 (bf16 pool) production recipes.
Packet ABI: object 26 `LIBRARY_PREFILL_ATTENTION` (`plow_fa4_hd512.aot`, ABI
`lib_flash_prefill_v2:fa4:...`) plus per-program `ProgramRoles.library` segments, declared by the
emitter (`PLOW_EMIT_ATTN_LIBRARY`, `crates/devgen/src/library_attention.rs`). The object holds the
forward/combine cubins with their recorded launch exemplars; plowrt launches them itself
(`crates/plowrt/src/exec/gpu/fa4_sm90.rs`), no host library.
Kernels and objects: the FA4 SM90 cubins (`scripts/fa4_aot/build.py`) plus the object's `kv_adapters` cubin
that copies each request's rows `[0, kv_len)` into a bf16/fp16 page pool (64-row pages, 64K rows
per KV head). One table upload per launch; a launch past the pool or 256 requests, or a windowed
body, keeps the native role. Consumers' counter waits are rewritten to stream order at load.
KV / state class: reads the slot KV through the gather into a transient pool; no persistent class.
Status: CUDA sm_90a only. hd512 (Gemma FP8 KV through the pool, BF16 KV direct) and hd256
(Qwen3.8 BF16, `plow_fa4_hd256.aot`, direct).
Deeper: `28-library-attention.md`.


## 2. Latent, sparse and memory attention; residual-stream variants

### Multi-head latent attention (MLA)

What it computes: K and V are not cached per head. Each token stores one head-shared low-rank
latent `c_kv` (`kv_lora_rank`, 512) plus one shared decoupled RoPE key `k_rope` (64). Decode uses
the absorbed form: `score = q_abs·c_kv + q_rope·k_rope` per head, with softmax and P·V accumulated
on the latent. A per-head `W_uv` fold then maps the latent accumulator to `v_head_dim`. Prefill
either runs the same latent attention or materializes per-head K/V (`MlaMaterializePack` + a
causal bf16 attention).

Examples: GLM-5.2 / GLM-5.3 (`glm_moe_dsa`, `glm5_next`), DeepSeek-V3 / Kimi K2 (`deepseek_v3`,
`kimi_k2`), Kimi K3's MLA layers (`kimi_k3`). DeepSeek-V4.1 is a fully absorbed variant: one
512-wide latent serves as both K and V, there is no `kv_b`, and it has a grouped output LoRA
(`crates/devgen/src/mla/dsv41.rs`).

Packet ABI:
- Read path:
  - `FlashMlaDecode` (50), with an optional in-op q-RoPE fold (`t7`, emit `PLOW_GLM_FUSE_ROPE`).
  - FP8-latent twins `FlashMlaDecodeFp8` (109) and `FlashMlaPrefillFp8` (110), with per-row
    scales in `t7`.
  - `OUvFold` (52), or the fused split-merge + fold `MlaMergeFold` (57). `i3=group` is the per-16-head
    fold of row-split attention (`PLOW_GLM_ROWSPLIT_ATTN`).
  - Materialized prefill `MlaMaterializePack` (126) + `FlashMlaMaterializedPrefill` (127), emit
    `PLOW_MLA_MATERIALIZED_PREFILL`.
  - `MlaOutGate` (106): K3's `out = a * sigmoid(g_proj(x))` before `o_proj`.
- Reserved, body not built: `FlashMlaPrefill` (51) (`slots.rs` `RESERVED`).
- Write path: ordinary ops. GEMV/GEMM for the down projections, `RmsNorm` (or `HeadNormRopeFp8` at
  `HD=512`) into `kv.L.ckv`, `HeadNormRope` into `kv.L.krot`.
- Emit knobs: `PLOW_MLA_PREFILL`, `PLOW_GLM_MLA_*` (W8A8, MHA, BF16_PS, NS_ROWS, DEC_AITER),
  `PLOW_SEG_DECODE_MLA` (promoted), `PLOW_MLA_GATHER_SPLIT`.

Kernels and objects:
- AMD: `runtime/amd/op_attention.h` (`d_flash_mla_decode<DK,DR,GF,GATHER>`, `d_mla_merge_fold`),
  dispatched in `interp.hip` and built into the gfx942/gfx950 interpreter objects.
- AMD segment and adapter sources: `mla_fold_pf_mfma.h`, `mla_bmm_fp8_pf.h`, `mla_mha_pf.h`,
  `mla_pf_dense_gfx950.hip`, `mla_materialize_pack.hip`, `mla_materialized_opus.hip` (the only
  production body for op 127: a capability-checked gfx950 raw object), and `mla_sparse_adapter.hip`.
- AMD pinned vendor `.co`: two MLA decode kernels (`agent-tools.md` §4).
- NVIDIA: `runtime/nvidia/op_mla.cuh`, which ports ops 50, 54 and 57 to sm_120a and is dispatched
  only in `interp_sm120.cu`. There is no sm_90a MLA arm.

Emitter: `crates/devgen/src/mla.rs` (shared MLA + MoE family; GLM, Kimi, DeepSeek-V3),
`mla/kimi_k3.rs`, and `mla/dsv41.rs` (V4.1 front end; see Status).

KV / state class:
- The latent cache `kv.L.ckv` (1024 B/row bf16, 512 B/row e4m3) and the rope cache `kv.L.krot`
  (128 B/row) are flat per slot. With FP8 there is also a per-row scale tensor `kv.L.scale`.
- `exec/kvrow.rs` patches the write rows each step.
- Prefix reuse is the AMD `SharedPrefix` (`exec/amd/shared_prefix.rs`). Its layout scan accepts
  exactly the roles `ckv` / `krot` / `kidx` / `scale` and the MLA / gather ops listed there. Any
  other `kv.*` kind refuses the layout.
- The CUDA KV block store does not serve MLA packets: no CUDA MLA serving path exists.
- Decode context parallelism shards the latent rows: docs/arch/21.

Status: AMD gfx942 / gfx950 serving (GLM TP4/TP8); sm_120a decode ops only; CPU none found.

Deeper: docs/arch/17 §4.3 and §5.4 (model-centric), docs/arch/21, docs/amd/glm53-*.md,
docs/amd/kimi-k3-mi325x.md.

### Sparse attention with a lightning indexer (DSA)

What it computes: an indexer scores every cached position,
`score[t] = Σ_h w[h]·ReLU(q_idx[h]·k_idx[t])`, with `q_idx` / `k_idx` projected,
LayerNorm'd and RoPE'd. An exact top-k select (lowest-index tie-break) keeps `top_k` positions.
Attention then reads only those rows, so its cost is constant in context.

Prefill builds a per-64-query-tile union of the selected rows. A pooled variant compresses each
group of `pool_size` index keys into one FP8 key: per-dim softmax pooling, then a Hadamard-128
rotation, then a power-of-two-scaled e4m3 quant. The query is quantized the same way, and the
current incomplete pool is always selected.

Examples: GLM-5.2 / GLM-5.3 (`glm_moe_dsa`, unpooled), GLM-5.3-Flash (`glm5_next`, pooled
`index_kpool`), DeepSeek-V3.2-style indexers; the DeepSeek-V4 / V4.1 indexer (not emitted).

Packet ABI:
- Decode: `IndexScore` (58), `IndexSelect` (59), `LayerNorm` (60) for the indexer key norm. The
  select's `t4=kv_len` is the live token count and is host-patched each step.
- Prefill: `IndexScorePf` (117), `IndexSelectPf` (118), `IndexUnionPf` (119).
- Pooled: `DsaPoolStash` (132; decode ring of `pool_size` slots), `DsaPoolCompress` (130; decode
  mode gated on `(pos+1) % pool_size == 0`), `DsaPoolExpand` (131), `DsaQQuant` (133),
  `IndexScoreKpool` (134). The selects take `pool_size` in `i2` / `i3`, default 1.
- Gathered attention: `FlashGatherDecode` (54), or the sparse extension of `FlashMlaDecodeFp8` /
  `FlashMlaPrefillFp8` (`j0` = idx or union handle + 1).
- AMD native segments: `IndexTpPf` (157; TP8 query-partitioned score, top-k and gather),
  `IndexFp8Decode` / `IndexFp8Prefill` (192/193; FP8 indexer with an owned block16 packed cache).
- Reserved, not built: `AttnSelect` (53), `FlashGatherPrefill` (55).
- Emit knobs: `PLOW_GLM_DSA`, `PLOW_GLM_DSA_PF` (promoted), `PLOW_GLM_DSA_*` (PF_SPAN, PF_B8,
  KPREP, KW_DUAL, PF_SHARD, PF_A2A, FUSE, SEL_UNION), `PLOW_GLM_INDEX_TP` / `PLOW_GLM_SELECT_LOCAL`
  (GLM recipe defaults), `PLOW_GLM_INDEXER_FP8`, `PLOW_GLM_INDEXER_WQ_W8A8`. Object define
  `PLOW_DSA_SELECT_SPLIT` and others (docs/flags-reference.md).

Kernels and objects:
- AMD: `op_attention.h` (`d_index_score[_fast|_kpool]`, `d_index_select_coop`, gathered
  `d_flash_mla_decode`), `op_dsa_pool.h`, `dsa_select_v3.h`, `dsa_score_fp8.h`, `dsa_prep.h`.
- AMD segment sources: `indexer_decode.hip`, `indexer_prefill[_append].hip`, `indexer_quant.hip`,
  `dsa_tp_adapter.hip` (`dsa_tp_adapter_gfx942.elf`).
- NVIDIA: `runtime/nvidia/op_dsa.cuh` (select + gather, sm_120a, `interp_sm120.cu`).

Emitter: `crates/devgen/src/mla.rs` (GLM family; `glm53_emit_full` for `glm5_next`).

KV / state class:
- Index-key cache `kv.L.kidx` (256 B/row bf16), flat per slot, written each step. The pooled
  variant keeps a pool cache plus a `pool_size` tail ring per layer, and the FP8 indexer an owned
  packed cache.
- AMD `SharedPrefix` reuses `kidx` beside the latent. A pooled cache (e.g. `kv.0.idx_pool`) is not
  one of the scan's roles, so a pooled packet gets no shared prefix (inferred from the scan, not
  run).

Status: AMD gfx942 / gfx950; sm_120a decode select and gather only. The `idx` table is per
request; selection is exact.

Deeper: `runtime/amd/dsa-gather-design.md`, docs/arch/21 (the DCP idx canonicalisation,
`IndexSelect i4=3`), docs/amd/glm53-*.md.

### Compressed KV attention (learned pooling, CSA)

What it computes: groups of `ratio` consecutive KV rows are pooled into one cache row, using a
per-channel softmax over the group with learned scores plus an additive position bias. The pooled
row is RMSNorm'd, its last `rd` channels RoPE-rotated, and the row quantized: FP8, or Hadamard
rotation then FP4. Attention reads the compressed cache.

Because each cached row was rotated at its own position, the attention output is de-rotated at
the query's position (`RopeInverseO`) before the fixed output projection.

V4.1 splits the compressor. Pooling + norm (`CompressPool i7=2`) feeds the indexer; RoPE +
blockwise FP4 quant (`CompressRopeQuant`) finishes the KV-cache and indexer-key rows.

Examples: DeepSeek-V4 / V4.1 (CSA2; the compressor runs on `kv_source` layers that other layers
read).

Packet ABI:
- `CompressPool` (212). `i7 = rotate` selects 0 = fp8, 1 = rotated fp4, 2 = V4.1 norm-only. A
  non-null `t7=pos` makes it a decode call gated on `(pos+1) % ratio == 0`.
- `RopeInverseO` (213).
- `CompressRopeQuant` (217). `i6 = qmode` selects the KV cache or the indexer keys; `i7 = n_head`
  covers the indexer queries.
- Interpreter defines `PLOW_DSV41_BLKFP8` / `PLOW_DSV41_ENGRAM` and manifest requirement
  `PLOW_DSV41_ENGRAM` (`devgen/src/manifest.rs`).

Kernels and objects: `runtime/amd/op_compress.h` (`d_compress_pool<ROTATE>`,
`d_rope_inverse_o`, `d_compress_rope_quant`), included by `interp.hip`. Only `RopeInverseO` has an
interpreter case. `d_compress_pool` is exercised by `runtime/tests/compress_pool_gfx942_test.hip`.

Emitter: `crates/devgen/src/mla/dsv41.rs` (`--block` multi-layer devblobs only). The full-model
emit refuses (`dsv41_refusal`, `devgen/src/lib.rs`).

KV / state class: a compressed cache per `kv_source` layer that grows by one row every `ratio`
tokens (docs/amd/deepseek-v41-flash-mi300x.md §6.3). No `plow-asset` section or runtime cache
class declares it yet.

Status: AMD kernels plus block emit only; not servable.

Deeper: docs/amd/deepseek-v4-mi300x.md, docs/amd/deepseek-v41-flash-mi300x.md §5–6.

### Engram conditional memory (hashed n-gram lookup)

What it computes: at a few layers, each position hashes itself together with the preceding
`max_ngram_size-1` tokens, once per (n-gram size, head) pair. Each hash selects a row of a large
FP8 table. The gathered rows (`n_hash_cols * head_dim` per token) are projected by `wkv` into
`hc_mult` keys plus one shared value.

A gate (a normalized dot of each residual stream copy against its key, a signed sqrt, then a
sigmoid) adds the value into every copy. A masked token (for example an image span) passes
through untouched.

Examples: DeepSeek-V4.1 (`engram_layer_ids` [1, 14]).

Packet ABI:
- The hash is host integer work over token ids. It arrives as an ids tensor like `pos`; there is
  no hash op.
- `EngramEmbed` (215): the row-sharded gathered table read, with ue8m0 block scales (`i3 = blk` =
  32) and `i4 = vocab_start` (the sentinel means `rank * part_rows`). It writes zeros outside the
  shard, and an `XReduce` sums the ranks.
- The `wkv` projection is an ordinary block-FP8 GEMM (`GemmFp8Mx`, 216).
- `EngramGate` (214): in place on the `[T][n][hidden]` streams, with a `token_mask`.
- Object define `PLOW_DSV41_ENGRAM`.

Kernels and objects: `runtime/amd/op_engram.h` (`d_engram_embed`, `d_engram_gate`). No interpreter
case on prod-next.

Emitter: `crates/devgen/src/mla/dsv41.rs` (block emit).

KV / state class: none per request. The table is a weight, sharded over rows. The host keeps a
lookback over token ids for the hash. That cache is still listed as emitter work in
`dsv41_refusal`.

Status: AMD kernel sources plus block emit; not dispatched, not servable.

Deeper: docs/amd/deepseek-v41-flash-mi300x.md.

### Hyper-connections (mHC: multi-stream residual)

What it computes: the residual is `n` (4) parallel streams per token instead of one.

- Before each sub-layer, `HyperConnPre` turns a projection of the streams (`mixes`) into three
  sets of weights: `pre` weights that combine the streams into the sub-layer input (emitted
  un-normed; the block's own RMSNorm follows), `post_mix` weights, and an `n×n` `comb_mix`
  matrix normalized by Sinkhorn iterations.
- After the sub-layer, `HyperConnPost` computes
  `new_residual[j] = Σ_i comb_mix[i][j]·residual[i] + post_mix[j]·x_out`.
- Entry and exit expand one stream to `n` and mean-contract `n` back to one (`HyperConnPost mode`
  1 and 2).

Examples: GLM-5.3-Flash (`glm5_next`), DeepSeek-V4 / V4.1 (`hc_mult`; mHC on every layer).

Packet ABI:
- `HyperConnPre` (128): `i3 = sinkhorn_repeat`, `f0/f1 = rms_eps/hc_eps`. `hc_post_mult_value` is
  a compile-time kernel constant the emitter must assert.
- `HyperConnPost` (129, `i3 = mode`).
- Emit knobs: `PLOW_GLM_HC_NORM_FOLD`, `PLOW_GLM_HC_POST_FOLD`, `PLOW_GLM_HC_PRE_TAIL`,
  `PLOW_GLM_HC_FUSE_DEC`, `PLOW_GLM_HC_FUSE_R1`, `PLOW_GLM53_PF_HC_AITER`.

Kernels and objects: `runtime/amd/op_hyperconn.h` (interpreter cases in `interp.hip`), plus the
segment sources `hc_mix_decode.hip` and `mhc_aiter.hip`.

Emitter: `crates/devgen/src/mla.rs` (`glm53_emit_full`), `mla/dsv41.rs`.

KV / state class: none. The `n` streams are activations, `n×` the hidden size per token in flight.

Status: AMD gfx942 / gfx950 (GLM-5.3-Flash); no NVIDIA or CPU arm.

Deeper: docs/amd/glm53-flash-mi350x.md, docs/amd/deepseek-v41-flash-mi300x.md §6.1.

### Attention residuals (AttnRes: softmax mix over block residuals)

What it computes: the plain residual add is replaced, twice per layer, by a softmax mix over the
running prefix sum and up to `nb ≤ 8` stored block residuals:

- `v = cat(block_residual, prefix_sum)`
- `k = RMS-normalize(v)` (RMSNorm variance, eps inside the rsqrt)
- `scores = k·score_w` (`score_w = norm.weight * proj.weight`, folded at prep)
- `out = softmax(scores) @ v`, mixing the raw rows

Every `attn_res_block_size` (12) layers a snapshot layer pushes its input onto the ring. The
final norm also reads an AttnRes mix.

Examples: Kimi K3 (`kimi_k3`, all 93 layers).

Packet ABI:
- `AttnRes` (104):
  - `nb` = rows live at this layer, `nb_cap` = ring stride.
  - Optional `push_src` / `push_row` for the snapshot layers, and `gamma` for a fused RMSNorm.
  - `res_a` / `res_b` / `res_pre` materialize the prefix sum, and `mwg_scratch` carries the banded
    multi-workgroup decode arm.
- Emit knobs: `PLOW_ATTNRES_DECODE_MWG` (also an object define), `PLOW_ATTNRES_F32MIX`
  (promoted), `PLOW_K3_FUSE_ARNORM` (promoted).
- The ring's capacity constant `PLOW_ATTNRES_MAXB` = 16 is an enum in `op_k3.h`, not a knob.
- K3's other block ops sit beside it: `SituGlu` (105) and `MlaOutGate` (106).

Kernels and objects: `runtime/amd/op_k3.h` (interpreter case in `interp.hip`),
`attn_res_f32mix_gfx950.hip`, `xreduce_attnres_fused.hip`. There is a CPU arm in `runtime/cpu`
(`gemv.c` / `control.c`).

Emitter: `crates/devgen/src/k3.rs` (block structure), `crates/devgen/src/mla/kimi_k3.rs`
(`kv.blkres`).

KV / state class: the ring `kv.blkres` holds `[T][nb_cap][H]` per token in flight and is rebuilt
every forward pass. It is not cross-token state, so `plow_asset::kv_contract::is_cache_tensor`
excludes `blkres` from the KV-geometry contract.

Status: AMD (MI325X K3 campaign) and CPU; no NVIDIA arm.

Deeper: docs/amd/kimi-k3-mi325x.md.


## 3. Recurrent mixers

### Recurrent state (shared contract for the mixers below)

What it computes: some layers carry a fixed-size state per sequence instead of (or as well as)
growing KV rows. Each new token reads that state and writes it back. The state's size does not
depend on context length, so its cost is paid once, at admission.

Examples: Gated DeltaNet (`qwen3_5`, Qwen3.8), KDA (`kimi_k3`), Mamba-2 (`nemotron_h`, block
emits only).

Packet ABI: the `recurrent_state` section (`crates/plow-asset/src/recurrent_state.rs`, version 1):
- `buffers`: slot-major `[slots][slot_bytes]` state tensors.
- `active`: an `i32[slots]` mask (tensor `in.active`) that the runtime fills before every decode
  step and prompt launch.
- `prefill_initstate`: an optional gathered init-state scratch for a packed prompt scan
  (`act.gdn.initstate`).
- `prefill_updates_state`: true when every prefill program advances every buffer.

Every `state.` class tensor must be one of the declared buffers; otherwise the packet is refused
at load. The CUDA engine validates the manifest at load (`exec/gpu.rs` `RecurrentState`) and zeroes
a slot's buffers on admission.

KV / state class:
- CUDA: the KV block store's `State` class (`crates/plowrt/src/exec/gpu/kv_blocks.rs`) captures
  every declared buffer's per-slot bytes back to back, as one block per boundary. Because state
  exists only at the slot's current row, a boundary can be published only while the slot stands
  exactly at it (`captures_state`, `state_capture_cap`). The prefix cache serves these packets,
  but CUDA auto-selection excludes them, so the recipe sets `PLOW_VMM_PREFIX=1`
  (`exec/gpu/prefix.rs` `select_vmm_prefix_layout`).
- AMD: recurrent tensors live under `kv.{l}.state` / `kv.{l}.conv_state.*` and ride the slot-local
  prefix snapshot (`crates/plowrt/src/exec/amd.rs` ~7210, ~13867).

Deeper: docs/arch/26-kv-block-store.md (State class), 17-unified-token-batch.md §4.5.

### Short causal depthwise convolution (conv history)

What it computes: a per-channel causal convolution of width W over the current token and the
previous W-1 inputs, usually followed by SiLU. The previous W-1 inputs are the "conv history",
carried per sequence.

Examples: every GDN, KDA and Mamba-2 layer.

Packet ABI:
- GDN decode: `PLOW_DOP_QWEN_GDN_CONV` (136; history `[B,C,W-1]` bf16 oldest-first, masked by
  `in.active`).
- GDN prefill: `PLOW_DOP_QWEN_GDN_CONV_PREFILL` (143).
- KDA: `PLOW_DOP_KDA_CONV` / `KDA_CONV3` (q, k and v streams fused).
- Mamba-2: computed inside `MAMBA2_SCAN` (op 90, `t6=conv_state` f32).
- History tensors: `state.qwen.{l}.conv` (`crates/devgen/src/qwen35.rs` ~470), and for KDA
  `kv.{l}.conv_state.{q,k,v}` plus a `conv_state_alt` double buffer (`crates/devgen/src/kda.rs`
  ~440).

Kernels: in the GDN and KDA op files below.

KV / state class: a recurrent buffer (see Recurrent state).

### Gated DeltaNet (GDN)

What it computes: linear attention with a per-head state S of shape [V,K] in f32, per token:
- forget gate: `S <- exp(-exp(A_log)·softplus(a+dt_bias))·S`, a scalar per (token, head);
- delta-rule write: `S <- S + beta(v - S k) ⊗ k`, with L2-normalised q and k;
- read: `o = S q`.

It is followed by a gated RMSNorm with the z branch, then the output projection
(`runtime/amd/op_qwen_gdn.h` header). It is the same family as KDA, differing only in that the
gate is a scalar instead of a per-key-channel vector.

Examples: `qwen3_5` network (Qwen3.5 and Qwen3.8 hybrid chains: GDN layers interleaved with
full-attention layers).

Packet ABI: opcodes are `PLOW_DOP_QWEN_*` (136-146); devgen derives the list by prefix
(`qwen35.rs` `qwen_opcodes`).
- Decode chain: `QWEN_GDN_QKV_PREP`, `QWEN_GDN_GATE_PREP`, `QWEN_GDN_CONV`, then `QWEN_GDN_STEP`
  (137; t6 = state `[B,HV,V,K]` f32, t7 = active), then `QWEN_GATED_NORM`.
- Prefill: `QWEN_GDN_PREFILL` (146) is not an interpreter arm. It is a host-launched external
  segment whose state operand must be a declared recurrent buffer of the library's fixed stride
  (`exec/gpu.rs` `check_gdn_state_operands`).
- State tensors: `state.qwen.{l}.gdn` and `state.qwen.{l}.conv`, plus `act.gdn.initstate` for
  packed scans. The section is built by `qwen35.rs` `recurrent_state_section`.
- Not verified: the full list of emit knobs that switch this. `PLOW_QWEN_PREFILL` selects the
  prefill buckets, and `PLOW_QWEN_GDN_ISOLATE` gives the step its own segment from a rung width.

Kernels and objects:
- CUDA decode: `runtime/nvidia/op_qwen_gdn.cuh` inside the paired interpreter object; the
  manifest requires `PLOW_NV_QWEN_GDN=1`.
- CUDA prefill: a generated CuTe DSL chunked kernel shipped as a packet cubin
  (`plow_gdn_prefill.aot`) and launched by plowrt's host stub (`device/cuda/qwen_gdn.rs`).
- AMD: all ten interpreter arms in `runtime/amd/op_qwen_gdn.h`, oracle-checked. However, the
  devgen emitter is NVIDIA-shaped and no HIP prefill exists, so `refuse_unimplemented_target`
  refuses AMD emits by name.

Emitter: `crates/devgen/src/qwen35.rs` (hand-routed).

KV / state class: recurrent buffers (state f32 `HV·V·K·4` bytes per slot per layer, plus conv
history bf16). The full-attention layers of the same chain keep ordinary head-interleaved slot KV.
Decode rows ride packed prefill only when `act.gdn.initstate` is present
(`crates/plowrt/src/inventory.rs`).

Status: CUDA sm_90a serves it (BF16 and FP8 recipes). AMD emit is refused.

Deeper: 17-unified-token-batch.md §4.5; docs/amd/qwen35-gdn-mi300x.md (outside arch).

### Kimi Delta Attention (KDA)

What it computes: the delta-rule recurrence with a state S of shape [K,V]:
`S <- (I - beta k kᵀ)·diag(exp(g))·S + beta k vᵀ`, then `o = Sᵀ q`. The forget gate `g` is a
per-(head, key-channel) vector bounded by `gate_lower_bound`. The output passes through a gated
RMSNorm (`runtime/amd/op_kda.h`).

Examples: `kimi_k3` (the mixer in most layers; the rest are MLA).

Packet ABI:
- Decode: `KDA_CONV` / `KDA_CONV3`, then `KDA_GATE` / `KDA_STATE_STEP(_G)` (fused gate + step),
  then `KDA_GATED_NORM`. `KDA_CONV_STATE_STEP_G` is the fused conv + gate + step.
- Prefill chunk pipeline: `KDA_CHUNK_PREPARE`, then `KDA_CHUNK_INTRA`, `KDA_CHUNK_WU`, then
  `KDA_CHUNK_CARRY` (`crates/devgen/src/kda.rs` ~1346).
- Projections: q, k, v and the output gate use the four-stream `GemvQkvg`.
- State tensors: `kv.{l}.state` (f32) and `kv.{l}.conv_state.{q,k,v}`.
- Many emit knobs select chunk and carry variants (the `--kda-*` entries in
  `docs/flags-reference.md`).

Kernels and objects: AMD only. `runtime/amd/op_kda*.h` provides the interpreter arms, and
`runtime/amd/kda_*.hip` the chunk, carry and decode variants. No NVIDIA arm exists.

Emitter: `crates/devgen/src/kda.rs` (one KDA layer) inside `crates/devgen/src/k3.rs`.

KV / state class: recurrent, carried in `kv.*`-named tensors on AMD. The state is constant in
context; the `kda.rs` comment gives its size per (sequence, layer). It rides the AMD slot
prefix snapshot.

Status: gfx942/gfx950 only.

Deeper: docs/kimi-k3-kda.md (outside arch).

### Mamba-2 selective scan (SSD)

What it computes:
- a causal depthwise conv1d with SiLU over `xBC`;
- per head, `dt = softplus(dt_raw + dt_bias)`, `A = -exp(A_log)`;
- the state update `h <- exp(dt·A)·h + dt·x·B`, with output `y = C·h + D·x`;
- a gated RMSNorm with `silu(z)`.

Prefill (a full scan) and decode (a single step) use the same op (`runtime/common/dev_isa.h`
op 90 `MAMBA2_SCAN`).

Examples: `nemotron_h` / `nemotron3` (hybrid Mamba-2 + GQA + MoE).

Packet ABI: `PLOW_DOP_MAMBA2_SCAN` (90). Its operands are
`t6=conv_state f32[d_conv-1,conv_dim]` and `t7=ssm_state f32[n_head,head_dim,d_state]`, both
in/out, and `t5` packs `A_log|D|dt_bias|conv_b|norm_w`. No `recurrent_state` section is emitted
for it.

Kernels and objects: `runtime/nvidia/op_mamba.cuh`, wired into `runtime/nvidia/interp_sm120.cu`
only. The file states it is unverified on GPU (compile-only; its math mirrors the f32 golden
`mamba_ref`). AMD has no arm (`runtime/amd/interp.hip` ~4453).

Emitter: `crates/devgen/src/lib.rs` `nemotron_emit_block`, single-block extraction only
(`--block`); a full-model emit panics by design.

KV / state class: recurrent buffers, not integrated with serving, the prefix cache or the block
store.

Status: block-level experiment, sm_120.


## 4. Mixture of experts

### MoE routing (router + top-k)

What it computes: per token,
1. score each expert from `x·Wr` with softmax or sigmoid;
2. optionally add a selection bias and restrict selection to the top `topk_group` expert groups
   (group-limited, noaux_tc style);
3. take the top-k with a lowest-expert-id tie-break;
4. optionally renormalise the k gates, then scale by `route_scale`.

The result is written to a routing table `[k]{u32 expert_id, f32 gate}`, with unused slots marked
`PLOW_EXPERT_UNUSED` (`dev_isa.h` ops 40, 56).

Examples:
- sigmoid + bias + group-limited: DeepSeek-V3 / Kimi K2-family (`i6/i7`);
- flat softmax/sigmoid: GLM, Qwen, Mixtral-shaped;
- softmax with a weightless-RMS pre-transform and per-expert gate scale: Gemma-4 26B-A4B
  (`MOE_ROUTER_GEMMA`, 61);
- sigmoid with a logit scale and an f32 router: Muse Spark.

Packet ABI:
- Decode, one op: `MOE_ROUTER` (40; flags bit0 = sigmoid/softmax, bit1 = norm_topk).
- Decode, split: an ordinary GEMV for the logits, then `MOE_ROUTER_TOPK` (56; t3 = bias,
  i6 = n_group, i7 = topk_group, f1 = logit scale).
- Prefill tail: `MOE_ROUTER_TOPK_PF` (83, T tokens).
- Gemma-shaped family: `MOE_ROUTER_GEMMA[_PF|_SCORE|_SCORE_FAST|_TOPK]`.
- The routing table's completion counter is the data-dependent gate the expert slots wait on.
  The counter DAG stays static: every expert packet signals whether it computed or skipped.

Kernels: `runtime/nvidia/op_moe.cuh` and `op_moe_sm90.cuh`; `runtime/amd/op_moe.h`.

Emitter:
- `crates/devgen/src/lib.rs` `emit_gemma_moe_router` (Gemma);
- `crates/devgen/src/mla.rs` and `mla/*` (GLM, DeepSeek, Kimi, Nemotron `emit_nemotron_moe`);
- `crates/devgen/src/mla/dsv41.rs` `emit_dsv41_moe`;
- `crates/devgen/src/k3.rs` `emit_k3_latent_moe`;
- `crates/devgen/src/gptoss.rs`, `crates/devgen/src/muse.rs`.

KV / state class: none (per-launch activation tables).

### MoE experts: GLU / down, shared experts, combine

What it computes: for each routed (token, slot), `fu = act(gate_e·x) * (up_e·x)` (SiLU, gelu_tanh
for Gemma, `situ` for Kimi-K3), then `part = gate * W_down,e·fu`. The combine is
`out = residual + shared + Σ_j part[j]`, summed in fixed slot order so it is deterministic.
Shared experts are ordinary dense projections whose result enters the combine (`MOE_COMBINE`,
`t2 = shared`).

Packet ABI:
- Decode, per top-k slot: `MOE_EXPERT_GLU` (41) and `MOE_EXPERT_DOWN`. A sentinel expert skips
  and streams zero weight bytes. `MOE_COMBINE` (43) follows.
- Precision twins: block-FP8 `_FP8_BLK` ([128,128] scale grid, 44-49) and the Gemma `_GEMMA[_FP8]`
  twins.
- Prefill, grouped: `MOE_ALIGN_PF` / `MOE_ALIGN_GEMMA_PF` (84; sort rows by expert into tiles),
  then `MOE_GROUP_GLU_PF` (85), `MOE_GROUP_DOWN_PF` (86, scatter to `part[T*k,H]`) and
  `MOE_COMBINE_PF` (87).
- Weight encoding is the op field `i3`: 0 bf16, 1 block-fp8 (w8a16), 2 MXFP4 (A4W4 on gfx950).
  Further twins: `_W8A8` Gemma twins, `MOE_COMBINE_NORM_GEMMA(_PF)` and
  `MOE_COMBINE_RESID_NORM_GEMMA`.
- `PLOW_MOE_PF_DET` makes the prefill down + combine exact and order-independent (integer-valued
  f64 accumulation; object define `PLOW_MOE_PF_DET=1`).

Kernels and objects:
- NVIDIA: `op_moe.cuh` and `op_moe_sm90.cuh` in the interpreter object. The MXFP4 decode object
  `interp_sm90a_mxfp4_moe.cubin` (`crates/devgen/src/mxfp4_moe_role.rs`, sm_90a, decode rung 1
  only) carries it.
- cuBLASLt grouped route (`crates/plowrt/src/exec/gpu/moe_lt.rs`):
  - prefill: `PLOW_EMIT_MOE_PF_LT` isolates each layer's `MoeGroupGluGemmaPf` +
    `MoeGroupDownGemmaPf` in a `MOE_PREFILL_CUBLASLT` segment, served by two grouped matmuls with
    device-side row counts and pointers, plus glue kernels in `runtime/nvidia/moe_lt_sm90.cu`;
  - decode: `PLOW_EMIT_MOE_DEC_LT` does the same on the grouped decode rungs
    (`MOE_DECODE_CUBLASLT`);
  - `MoeAlignGemmaPf` still runs in the interpreter;
  - serving is on by default (`rt.moe_pf_lt` / `rt.moe_dec_lt`, `=0` rollback);
  - it needs cuBLASLt 13.4 (`asset/serve.rs` `runtime_requires`).
- AMD: `runtime/amd/op_moe.h`, `op_moe_fp8_block128.h`, `moe_fp8_pf.hip`, `moe_pf_w8a8.h`,
  `moe_w8a8_dec.hip` and `moe_decode_grouped_mxfp4.hip`, plus the AITER adapter
  (`moe_aiter_adapter.hip`, `MOE_AITER_FP8_PF`).

KV / state class: none.

Status:
- Gemma MoE (BF16/FP8/W8A8) on CUDA sm_90a;
- the MLA family's MoE on AMD gfx942/gfx950;
- MXFP4 on gfx950 (A4W4) and an sm_90a decode object.

### Expert parallelism

What it computes: experts are sharded across ranks, and tokens or heads are exchanged by
all-to-all, instead of replicating every expert per rank.

Packet ABI: emit knob `PLOW_MOE_PREFILL_EP` (opt-in; it produces an experiment asset for
`runtime/bench/amd/moe_ep_boundary`), the collective `XALLTOALL_HEADS` and the object define
`PLOW_HAS_XALLTOALL_HEADS`. Shipping TP packets replicate experts per rank and use the TP
collectives (docs/arch/09-multi-gpu.md).

Status: experimental, AMD only. Not verified beyond the knob doc.

### Muse-style gated residual and norms

What it computes:
- `MUSE_RMSNORM` (226): RMSNorm in three modes, plain `x·r·w`, centered `x·r·(1+w)` and
  input-scaled, with an optional output scale.
- `MUSE_RESIDUAL_GATE` (227): `out = alpha·res + beta·h` with `beta = sigmoid(g/T)` and
  `alpha = sqrt(max(sigmoid(-g/T)(1+beta), eps))`, a learned interpolation in place of a plain
  residual add.
- `MUSE_NORM_GATE_NORM` (229): norm, gate and norm fused.
- `MUSE_MOE_COMBINE_NORM` (228): MoE combine followed by a norm.

Examples: `muse_spark` / `muse_spark_text`. Its MoE uses a pre/post expert projection around
grouped MXFP4 experts, and the router is a sigmoid with a bias.

Kernels: `runtime/amd/op_muse.h`. Emitter: `crates/devgen/src/muse.rs` (gfx950, TP1 or TP).

KV / state class: none. Status: gfx950.


## 5. Precision and encoding

### FP8 weights, BF16 activations (W8A16)

What it computes: weights are stored as OCP e4m3 bytes with one f32 dequant scale per output
channel; activations stay bf16. Each fp8 weight is converted to bf16 on load and the per-channel
scale is applied once to the accumulated sum in the epilogue, so decode streams half the weight
bytes of bf16.
Examples: Gemma-4 12B/26B/31B FP8 checkpoints (decode projections), dense families under
`--w8a16` (sm_120).
Packet ABI: `GemvFp8` (30, `t5=w_scale f32[N]`), `GemvGluFp8` (31, gate/up with their own
scales), `GemvQkvFp8`; prefill one-row tail `GemmFp8` instructions eligible for the
`W8A16_PREFILL_M1` segment role (9, ABI `w8a16_prefill_m1_sm90_v1`,
`plow_asset::segment_roles`). Emit: `PLOW_FP8` (dense families: w8a16 on sm_120, block-fp8
expert arms on MLA+MoE families), `PLOW_W8A16`.
Kernels and objects: CUDA decode GEMVs in `runtime/nvidia/op_gemv_fp8_mma.cuh`,
`op_gemv_fp8_tc.cuh` (16-row mma tiles, e4m3->f16->bf16 exact), `op_gemv_fp8_wgmma.cuh` inside
`interp_sm90a.cubin`; the M=1 prefill role in `interp_sm90a_pfgemm_w8a16_m1.cubin`
(`crates/devgen/src/w8a16_prefill_role.rs`, eight qualified (N,K) shapes); the Qwen decode
`FP8_M1` role (segment role 4, ABI `fp8_gemm_m1_tma_v1`, entry `interp_sm90a_fp8m1`,
`crates/plow-asset/src/fp8_m1_role.rs`, `PLOW_QWEN_FP8_M1_TMA`). AMD: `runtime/amd/op_gemm.h`
(`gemv_nrn_lds` fold, `i3`).
KV / state class: none (weights only).
Status: CUDA sm_90a, sm_120; AMD gfx942/gfx950 decode GEMVs.

### FP8 weights and activations (W8A8)

What it computes: both GEMM operands are e4m3. Activations are quantized per row (token):
`a_scale[m] = max|x[m,:]| / 448`; weights carry one f32 per output channel; the f32 accumulator
is dequantized `acc * a_scale[m] * w_scale[n]` in the epilogue. One `QuantFp8` per activation is
reused by every GEMM that consumes it.
Examples: Gemma-4 12B/26B FP8 prefill, Qwen3.8 27B FP8, GLM-5.3 (`PLOW_GLM_*_W8A8`), MoE
experts (`PLOW_MOE_PF_W8A8`, `PLOW_MOE_DEC_W8A8`).
Packet ABI: `QuantFp8` (32; `t3/t4` fused GLU producer), `GemmFp8` (33) and its tile twins
(`GemmMedFp8`, `GemmSmallFp8`, `GemmWideFp8`, `GemmC5Fp8`), `GemmGluFp8`, MoE `*_W8A8` group
ops. Segment roles: `FP8_PREFILL_GEMM` (1), `W8A8_PREFILL_GEMM_GLU_GEMMA4` (13, ABI
`gemm_glu_w8a8_sm90_gemma4_4k8k_v2`), `CUBLASLT` (5) projection segments, `MOE_PREFILL_CUBLASLT`
(16) / `MOE_DECODE_CUBLASLT` (17). Emit: `PLOW_W8A8`, `PLOW_QWEN_W8A8_PREFILL`,
`PLOW_FP8_PF_GEMM_ROLE`, `PLOW_GEMMA4_SM90_W8A8_GEMM_GLU_ROLE`, `PLOW_EMIT_PREFILL_CUBLASLT`,
`PLOW_EMIT_DECODE_CUBLASLT`.
Kernels and objects: interpreter `op_gemm_sm90.cuh` in `interp_sm90a.cubin`; fused GLU role
`interp_sm90a_pfgemm_glu_w8a8_gemma4.cubin` (rows 4096-8192, N 15360, K 3840; globals checked by
`crates/devgen/src/gemma4_w8a8_gemm_glu_role.rs`); cuBLASLt with the outer-vector scale mode (`crates/plowrt/src/device/cuda/lt.rs`,
eligibility `segment_roles::cublaslt_prefill_fp8`, needs cuBLASLt >= 12.9 = `CUBLASLT_FP8_MIN`);
grouped MoE matmuls in cuBLASLt (`crates/devgen/src/dense_cublaslt.rs`, glue
`runtime/nvidia/moe_lt_sm90.cu`, >= 13.4 = `CUBLASLT_GROUPED_MIN`). AMD gfx950 wide-K MFMA
(`d_gemm_fp8_t`, `runtime/amd/op_gemm.h`).
KV / state class: none.
Status: CUDA sm_90a; AMD gfx950 (`--fp8` on gfx950 refuses and points at `--w8a8`). The packet
records the minimum cuBLASLt (`build.json` `runtime_requires.cublaslt`, `dense_cublaslt.rs`).

### Block-scaled FP8 (128x128 and 32x32 grids)

What it computes: e4m3 weights with a 2-D grid of dequant scales. The DeepSeek/GLM form is
`weight_block_size: [128,128]` with arbitrary f32 entries, `S[(n>>7)*ceil(K/128) + (k>>7)]`;
the scale is promoted into a second f32 accumulator every 128 K. The DeepSeek-V4.1 form blocks
32 on both axes with ue8m0 byte entries (`K % 32 == 0` required). Activations may be quantized in
dynamic groups of 128 along K (`QuantFp8Block128`).
Examples: GLM-5.3 linear layers and shared experts, DeepSeek-V3/V4.1 projections, Kimi.
Packet ABI: `GemmFp8Blk` (107, one 128x128x64 tile rung), `GemvFp8Blk`, MoE block-fp8 ops
(`MOE_EXPERT_GLU_FP8_BLK`, `MOE_GROUP_GLU_FP8_BLK`, ...), `GemmFp8Mx` (216, the 32x32 ue8m0 grid;
a separate opcode so a mismatched scale handle is a load refusal), `QuantFp8Block128` (184),
`MoeQuantFp8Block128` (189), `DenseGluFp8Blk`. Emit: `PLOW_FP8` on MLA+MoE families,
`PLOW_GLM_*` precision knobs.
Kernels and objects: AMD `d_gemm_fp8_blk`, `d_gemm_fp8_mx` (`runtime/amd/op_gemm.h`,
`op_gemm_common.h`) in the gfx942/gfx950 interpreter objects (`scripts/build_gfx942.sh`).
CUDA: not verified for these opcodes.
KV / state class: none.
Status: AMD gfx942/gfx950.

### MXFP4

What it computes: weights packed two fp4 values per byte (low nibble = even k) with one E8M0
exponent scale per 32-K block; the weight fetch dequantizes fp4 -> bf16 with the scale folded and
reuses the bf16 MMA path. Activations are bf16 in the GEMV/GEMM opcodes (w4a16).
Examples: GPT-OSS experts, DeepSeek-V4.1 routed experts (ops 85/86 grouped MoE), Kimi-K3
(`PLOW_K3_DECODE_MXFP4_PROJ`).
Packet ABI: `GemvMxfp4` (91), `GemvGluMxfp4` (92), `GemmMxfp4` (93) and its tile twins
(`GemmMedMxfp4`, ...), `GemvQkvMxfp4`; segment role `MXFP4_MOE` (7, ABI `mxfp4_moe_sm90_v1`).
Emit: `PLOW_MXFP4`; object define `def.PLOW_MXFP4`.
Kernels and objects: AMD `d_gemm_mxfp4`, `d_gemv_glu_mxfp4` (`runtime/amd/op_gemm.h`); CUDA
`runtime/nvidia/op_gemm.cuh` / `op_moe.cuh`, the decode MoE object
`interp_sm90a_mxfp4_moe.cubin` (entry `plow_sm90a_mxfp4_moe`, attached by
`crates/devgen/src/mxfp4_moe_role.rs` only for a sm_90a packet whose decode ladder is `[1]`), and
`interp_sm120.cu`.
KV / state class: none.
Status: AMD gfx942/gfx950; CUDA sm_90a (MoE decode role), sm_120.

### FP8 KV cache

What it computes: K and V are written as e4m3 with one f32 dequant scale per (position, KV head)
row, halving KV bytes and the decode KV stream. Prefill attention dequantizes at the shared-memory
stage (the MMA is unchanged); decode applies the per-row scale. Lossy versus a bf16 cache.
Examples: Gemma-4 12B/26B FP8 serving arms, GLM-5.3 (`PLOW_GLM_FP8_KV`).
Packet ABI: `HeadNormRopeFp8` (37, `t6=scale f32[kv_head][ctx]`), `FlashDecodeFp8` (38),
`FlashPrefillFp8` (39, `i7 = nsplit | span<<16`), MLA twins `FlashMlaDecodeFp8`,
`FlashMlaPrefillFp8`; generated role ABI `gen_flash_prefill_fp8kv_v1`. Emit: `PLOW_FP8_KV`,
`PLOW_FP8_KV_FULL`; object build `PLOW_BUILD_FP8KV_FA`, `PLOW_BUILD_FP8KV_GEMM`.
Kernels and objects: `runtime/nvidia/op_attention*.cuh` in `interp_sm90a.cubin`; generated
`gen_attn_pf_hd256_sliding_fp8kv.cu` / `attn_pf_hd512_fp8kv` roles; the library hd512 route
gathers rows `[0, kv_len)` into a bf16 or fp16 page pool through the object's `kv_adapters` cubin before
the FA4 cubins of `plow_fa4_hd512.aot` (`crates/plowrt/src/exec/gpu/lib_attention.rs`, docs/arch/28).
KV / state class: the scale tensors live beside each K/V track; in the block store the sliding
rings' scales ride the 32-row window blocks and the full layers' scales the 32-row prefix
(`Class::Tail`) blocks (`crates/plowrt/src/exec/gpu/kv_blocks.rs`).
Status: CUDA sm_90a; AMD (FlashPrefillFp8 on gfx942/gfx950).
Deeper: docs/arch/22-kv-cache.md, 26-kv-block-store.md, 28-library-attention.md.

### Library GEMM routes (cuBLASLt)

What it computes: a packet segment whose instructions are a dense projection (or a grouped MoE
GLU+DOWN pair) that plowrt runs through cuBLASLt instead of the interpreter; a runtime without
the route runs the packet's interpreter instructions unchanged.
Packet ABI: `segment_roles.json` roles `CUBLASLT` (5), `MOE_PREFILL_CUBLASLT` (16),
`MOE_DECODE_CUBLASLT` (17), written by `crates/devgen/src/dense_cublaslt.rs`; rung eligibility
`cublaslt_prefill_bf16` / `cublaslt_prefill_fp8` (prefill up to `CUBLASLT_PREFILL_MAX_ROWS`
16384), decode rungs up to `CUBLASLT_DECODE_MAX_ROWS` 128; `build.json` `runtime_requires.cublaslt`
(12.0 BF16, 12.9 FP8, 13.4 grouped).
Kernels and objects: `crates/plowrt/src/device/cuda/lt.rs` (library binding, load-time algorithm
selection, `PLOW_LT_ALGOS` table), `moe_lt_sm90.cu` glue.
Status: CUDA only.

### N-concatenated projections (derived weights)

What it computes: projections that read the same activation (gate|up, q|gate|k|v) as one GEMM
over their weights concatenated along N, block-scale grids concatenated the same way; consumers
read their columns of the one row-pitched output.
Examples: the block-scaled FP8 Qwen3.5-style dense MLP and full-attention projections.
Packet ABI: `derived_weights.json` (`crates/plow-asset/src/derived_weights.rs`) lists each `cat/`
tensor's checkpoint sources (axis 0, optional BF16 -> F32 widen); `GluPitched` (230); the row
pitch of `QwenQGateSplit` (`i3`) and `QwenHeadNormRope` (`j1` = pitch | column << 16). Emit:
`PLOW_FP8_BLOCK_CONCAT`. Runtime contract 4: a packet using any of these declares it and a
runtime that cannot serve them refuses the packet at load.
Kernels and objects: the loader assembles `cat/` tensors (`asset/checkpoint.rs`), the cuBLASLt
route runs the one GEMM, `d_glu_pitched` and the pitched split/KV-writer bodies run in the
sm_90a interpreter, light and light_quant entries.
Status: sm_90a (cuBLASLt decode and prefill); the HSA loader refuses contract-4 packets.


## 6. Multi-token drafting and speculative decode

### Chain drafter (MTP assistant)

What it computes: a small draft network proposes `k` tokens per sequence by chaining: step `j`
embeds `tokens[j]` with the target embedding table, projects it with the carried hidden (the
verify's final-norm row at step 0, then the previous step's projection), runs Q-only layers
against the target's KV at the committed length, and emits a greedy draft. The draft never
writes KV.
Examples: Gemma-4 assistant MTP (`gemma4_assistant`).
Packet ABI: `spec.json` (`plow_asset::spec_draft::SECTION`, version 1) and the draft programs in
the `spec.draft.programs` aux-program section, one program per `(rows, k)`. Emit:
`PLOW_EMIT_SPEC`, `PLOW_SPEC_KIND`, `PLOW_SPEC_ROWS`, `PLOW_SPEC_KMAX`, `PLOW_SPEC_DRAFT_DIR`,
`PLOW_SPEC_DRAFT_FP8`. Weights from `model-mtp.safetensors` (`scripts/gemma4_mtp_rekey.py`).
Kernels and objects: the ordinary decode interpreter (`interp_sm90a.cubin`) runs the draft
programs; plowrt stages inputs and reads back the tokens (`crates/plowrt/src/exec/gpu/spec_draft.rs`).
Emitter: `crates/devgen/src/mtp.rs`.
KV / state class: none of its own; reads the target's last sliding and last full caches.
Status: CUDA (serve `PLOW_SPEC=1`, opt-in).

### Block drafter (DSpark)

What it computes: a drafter that keeps its own KV: for each committed row, K/V of its
full-attention layers are projections of a fused sum of target post-layer residuals at fixed tap
layers. Every target program copies those residuals and ends with a context tail that writes the
drafter layers at each row's own position and slot, in the same launch as the target KV, so the
drafter context has no holes. The drafter then runs one `size`-row block per sequence.
Examples: Gemma-4 with a DSpark drafter (`PLOW_SPEC_KIND=gemma4_dspark`).
Packet ABI: `spec.json` (`BlockDraft`), `spec.context.json` (`plow_asset::spec_context`:
`taps`, `kv_layers`, `layout_sha256`), drafter layers as ordinary packet KV layers `kv.{L+i}.*`
listed with the full layers in `serve.json`.
Kernels and objects: decode interpreter; weights from `model-dspark.safetensors` (and the FP8
twins `model-dspark-fp8.safetensors`).
Emitter: `crates/devgen/src/dspark.rs`.
KV / state class: drafter KV layers share the target's full-layer layout, so live KV, the prefix
cache and the block store carry them; `layout_sha256` restricts a restored prefix to a packet
whose tail would have written the same rows.
Status: emitted only with `PLOW_SPEC_KIND=gemma4_dspark`; the CUDA serving path refuses block
drafters today (docs/arch/25).
Deeper: docs/arch/25-dspark-pending.md.

### Verify and acceptance (runtime)

What it computes: on pure decode ticks plowrt drafts greedy (and, with `PLOW_SPEC_SAMPLED`,
sampled) rows, verifies `k+1` rows per sequence in one launch and accepts the longest matching
prefix; sampled verify rows are sampled as plain decode samples them (lossless).
`PLOW_SPEC_RIDE` lets decode rows riding a prefill launch carry drafts. Depth per row comes from
measured step, draft and acceptance costs.
Packet ABI: the packet's draft programs plus its decode rungs; no verify-specific opcode.
Kernels and objects: runtime-side (`crates/plowrt/src/serve/mux.rs`, `exec/gpu/spec_draft.rs`).
KV / state class: rejected drafts' KV rows past the accepted length are overwritten by later
writes (not verified beyond the mux code path).
Status: CUDA; `PLOW_SPEC` opt-in, `PLOW_SPEC_SAMPLED` on, `PLOW_SPEC_SAMPLED_KMAX` 3.
Deeper: docs/flags-reference.md (`PLOW_SPEC*` rows).


## 7. Media and speech inputs

### Multimodal soft tokens (vision and audio encoders)

What it computes: a vision or audio tower and its projection run as separate sidecar packets
(`forward.v1`, with capacity rungs); their output rows replace placeholder tokens in the LM's
embedded rows. A prompt row whose id has bit 31 set gets the pad embedding from `Embed`, then
`MmRowsBf16` replaces its hidden row with the slab row named by the id. A modality with
bidirectional attention within each item also sets `SPAN_ID_BIT` (bit 30); prefill derives each
row's span extent (`MmSpanExtent`) and the marked attention sites let a query see the rest of its
item. Chunk ends and sliding stage boundaries never fall strictly inside a span
(`span_safe_rows`).
Examples: Gemma-4 vision and audio (E4B, 12B multimodal recipes).
Packet ABI: `plow.multimodal.v1` (`crates/plow-asset/src/multimodal.rs`: placeholder ids,
soft-token wrapping, processor parameters, sidecar names); `MmRowsBf16` (209), `MmSpanExtent`
(210), `FlashPrefillFp8` `i7` span half; tensors `in.mm_table`, slab rows; the
`media_geometry.v1` Lean obligation (`crates/plow-asset/src/media_geometry.rs`).
Kernels and objects: sidecar encoder packets run by `crates/plowrt/src/serve/mm/encoder.rs`;
LM ops in the interpreter object.
Emitter: `crates/devgen/src/mm.rs`.
KV / state class: none extra; media rows become ordinary KV rows. Media-derived ids make
prefix-cache keys distinguish different media with the same text.
Status: CUDA (gated by `mm_parity`, docs/bringup/agent-tools.md §5).

### Encoder overlay rows

What it computes: a packet input `in.encoder_overlay` (f32 `[overlay_rows][hidden]`) with an
index `in.encoder_overlay_index` (`i32[ctx]`) that overlays encoder or conditioning rows onto the
LM input at chosen positions.
Examples: Qwen3-ASR audio encoder output (`max_source_positions` rows), Chatterbox T3
conditioning (`chatterbox_t3.overlay_rows`) (`crates/devgen/src/config.rs`).
Packet ABI: the two tensors above (`crates/devgen/src/lib.rs` `Tn`).
KV / state class: none extra. Overlay rows are not keyed by prompt ids, so a packet with
`in.encoder_overlay` never selects the VMM prefix cache
(`crates/plowrt/src/exec/gpu/prefix.rs` `select_vmm_prefix_layout`).
Deeper: docs/arch/20-asr-pipelines.md, 24-tts-pipelines.md.

### Speech front-ends and codecs (pointer)

Conformer encoders (`crates/devgen/src/conformer.rs`), RNNT decoding (`rnnt.rs`), audio
subsampling (`asr_subsampling.rs`), VAD (`vad.rs`), codec LMs and audio codecs (`codec.rs`,
`mimi_encoder.rs`, `fish_codec.rs`, `fish_encoder.rs`, `fish_depth.rs`, `s3gen.rs`, `tts.rs`,
`voice_cond.rs`), with the contracts `speech_contract`, `speech_fusion`, `fish_depth` in
`crates/plow-asset/src`. Their ops, ABI and state are documented per pipeline in
docs/arch/20-asr-pipelines.md and docs/arch/24-tts-pipelines.md (and docs/runtime/asr.md,
tts.md, asr-packet-contract.md).

## 8. Backend parity

Checked against the code (prod-next 29f5f4f02 + b43fcdc5b), not the docs. Techniques move both
ways between backends; this section says which backend serves each technique today and what the
others need.

Cell values:
- **served**: devgen emits it for the target AND that backend dispatches it in a full-model
  serving path (interpreter `case PLOW_DOP_*`, objects, runtime). "No recipe" = the arm is
  compiled but no production recipe qualifies it there.
- **emitted-not-served**: devgen emits it (or only a `--block` extraction), but the backend does
  not serve it end to end (arm compiled out, no case, refused at load).
- **missing**: neither.

Build facts behind the NVIDIA columns:
- `interp_sm90a.cu` and `interp_sm89.cu` `#include "interp_sm120.cu"`. All three CUDA
  interpreters share one source and differ only in defines.
- `interp_sm120.cu` defaults `PLOW_NV_MLA`, `PLOW_NV_DSA` and `PLOW_NV_MAMBA` to 1.
- The sm_90a and sm_89 objects force all three to 0 to save cubin size and stack: `_nv_var_sm90a`
  / `_nv_var_sm89` in `runtime/CMakeLists.txt`, and `GEMMA_GATE` in `scripts/build_sm90a_cubin.sh`.
  A gated-out opcode traps, so on sm_90a/sm_89 those arms are compiled out, not unwritten.
- The CUDA runtime (`exec/gpu`) is one path for every `sm_` target.

Notes on "served" for sm_89: the CUDA runtime (`exec/gpu`) is one path for every `sm_` target, so
runtime features follow the interpreter arms; sm_89 recipes cover Gemma-4 E4B, Llama TTS, Qwen3 ASR,
fastconformer RNNT, fish-s2.

### Attention, latent/sparse/memory, recurrent

| Technique | sm_90a | sm_120 | gfx942 | CPU |
|---|---|---|---|---|
| Causal GQA flash + split-KV merge + FP8 KV | served | served (sm_89 too) | served | served (golden/AVX-512) |
| Sliding window / hybrid layouts | served | served | served | served |
| KV-shared tail | served | served (same CUDA engine) | missing | missing |
| Attention sinks / logit softcap | served | served | served | served |
| Library attention route (FA4 hd512) | served | missing (SM90 export only) | missing | n/a |
| MLA | emitted-not-served (arms gated out) | emitted-not-served (decode ops only) | served (also gfx950) | missing |
| DSA lightning indexer | emitted-not-served (arms gated out) | emitted-not-served (decode index only) | served (also gfx950) | missing |
| Compressed KV (CSA) | missing | missing | emitted-not-served (op 213 only) | missing |
| Engram | missing | missing | emitted-not-served (kernel, no case) | missing |
| Hyper-connections (mHC) | missing | missing | served (PLOW_K3 objects) | missing |
| AttnRes | missing | missing | served (K3 objects) | missing (golden arm, no emit) |
| Recurrent state contract | served (block store State) | n/a (no recurrent emit for sm_120) | served (slot snapshot) | missing |
| Conv history | served (GDN ops) | missing (no GDN emit) | served (KDA conv) | missing |
| Gated DeltaNet | served | missing (arms compile, no emit) | missing (arms gfx950, emit refused) | missing |
| KDA | missing | missing | served | missing |
| Mamba-2 | missing (arm gated out) | emitted-not-served (block only) | missing | missing |

### MoE, precision, drafting, media

| Technique | sm_90a | sm_120 | gfx942 | CPU |
|---|---|---|---|---|
| MoE routing, Gemma-style (softmax top-k, router score/topk split) | served (26B recipes) | served, no recipe | served (`PLOW_MOE_GEMMA` arms) | served (26B `amx` recipe) |
| MoE routing, sigmoid + bias + group-limited (GLM/DeepSeek/Kimi) | missing (prefill topk only) | missing | served (GLM-5.3, Kimi-K3 recipes) | missing (prefill topk only) |
| MoE experts, Gemma GLU/down/combine (+ cuBLASLt grouped route) | served (26B recipes; `moe_lt.rs`) | served, no recipe (Lt route sm_90a) | served | served |
| MoE experts, generic GLU/down, grouped prefill, FP8-block experts | missing | missing | served | missing |
| MoE experts, MXFP4 (GPT-OSS ops 150-153) | emitted-not-served (no recipe) | emitted-not-served (no recipe) | missing (refused) | dispatch only (no recipe) |
| Expert parallelism | missing | missing | emitted-not-served (opt-in) | missing |
| Muse gated residual / norms | missing | missing | missing (gfx950 only) | missing |
| W8A16 (FP8 weights, bf16 acts) | served (26B/31B FP8) | served (sm_89 FP8 recipes not verified) | served (GLM-5.3 FP8) | dispatch only (no recipe) |
| W8A8 (FP8 weights + acts) | served (`PLOW_NV_W8A8` arms) | served, no recipe | served | dispatch partial (no recipe) |
| Block-scaled FP8 128x128 (ops 107 + blk GEMV/GLU) | emitted-not-served (decode only) | same as sm_90a | served | missing |
| Block-scaled FP8 32x32 ue8m0 (op 216) | missing | missing | missing (gfx950 only) | missing |
| MXFP4 dense projections (ops 91-93 family) | emitted-not-served (decode GEMV only) | same | served (Kimi-K3 recipe) | dispatch only (no recipe) |
| FP8 KV cache | served | served (sm_89 recipes) | served | dispatch only (no recipe) |
| Library GEMM routes (cuBLASLt / hipBLASLt) | served (segment roles) | served, no recipe (Lt is CUDA-generic) | served (`amd_gemm_lt.rs`) | n/a |
| N-concatenated projections (derived weights, contract 4) | served (block-FP8 Lt, trial recipe) | missing (no FP8 block Lt emit) | missing (refused at load) | missing |
| MTP chain drafter + verify | served (`exec/gpu/spec*.rs`) | served, no recipe | missing | missing |
| DSpark block drafter | emitted-not-served (refused) | emitted-not-served | missing | missing |
| Multimodal soft tokens (ops 206-210) | served (trial recipes only, `mm_parity` gate) | served, no recipe | missing (refused) | missing (refused) |
| Encoder overlay rows (ASR audio, TTS conditioning) | served (ASR/TTS recipes) | served (sm_89 ASR/TTS recipes) | missing | missing |
| Speech front-ends (conformer, RNNT, codec LM, f32 speech ops) | served | served (sm_89 recipes) | missing | missing (4 of 7 f32 ops) |

### What each missing cell needs

- **N-concatenated projections on gfx942**: the HSA loader needs the `derived_weights.json` assembly (`Checkpoint::concatenated`), `GluPitched` and the two pitched sources in `runtime/amd/interp.hip` (`d_qwen_q_gate_split` / `d_qwen_headnorm_rope` addressing), and the hipBLASLt route over the concatenated shape; until then `asset::serve::refuse_concat_features` rejects such a packet.

- **KV-shared tail on gfx942**: needs a port of the runtime gather (`crates/plowrt/src/exec/gpu/shared_tail.rs` `shared_tail_enqueue`/`shared_tail_rows`) into `exec/amd.rs`, reading the same `kv_shared_tail` section (`crates/plow-asset/src/kv_shared_tail.rs`). No new device op: it reuses the flash ops over the source layer's cache.
- **KV-shared tail on CPU**: the same gather in `serve/cpu_serve.rs`'s prefill path. Not verified whether any CPU emit carries the section.
- **Library attention on sm_120**: needs an FA4 export for SM120 (`scripts/fa4_aot/build.py` exports the SM90 forward only) and a `kv_dequant` cubin built for sm_120 (`runtime/nvidia/kv_dequant_sm90.cu`). The host route `exec/gpu/lib_attention.rs` is arch-neutral apart from its object names.
- **Library attention on gfx942**: needs a ROCm library with the plow_fa4 C ABI (abi 2: create, run, dtype, destroy; `lib_attention.rs:283-330`), e.g. a CK or aiter FMHA export, plus a HIP port of the FP8 page gather. The host route must also move into `exec/amd.rs`: one table upload per launch, consumer waits rewritten to stream order (`ordered_waits_for`), and the fallback to the native role.
- **MLA on sm_90a**:
  - Decode: rebuild the sm_90a object with `-DPLOW_NV_MLA=1`. That brings in ops 50, 54 and 57 (`op_mla.cuh`). Their tuning (`d_flash_mla_decode<512,64,8>`) is sm_120's; sm_90a has no wgmma latent kernel.
  - Prefill (none exists on any CUDA arch): op 51 `FlashMlaPrefill`, op 52 `OUvFold`, and the FP8 latent ops 109/110. The templates are AMD `exec_flash_mla_prefill` (`interp.hip:2485`), `mla_fold_pf_mfma.h`, `mla_bmm_fp8_pf.h`, or the materialized path (ops 126/127, `mla_materialize_pack.hip`).
  - ABI: `live_kv` `Cache` (`crates/plow-asset/src/live_kv.rs:19`) describes K/V *pairs* of one shape. MLA needs a latent cache kind: `ckv` (kv_lora_rank wide, heads=1), `krot` (rope width) and optional fp8 `scale`. `serve.json` `KvGeometry` (`serve_manifest.rs:186`) needs the latent and rope widths.
  - Block store: `exec/gpu/kv_blocks.rs` `Class {Window, Tail, Scale, State}` sizes rows from `KvGeometry` heads x hd. It would need per-tensor row bytes for `ckv`/`krot` (Tail-like, 32-row prefix blocks) and a Scale entry for the fp8 latent.
  - Runtime: the CUDA engine has no MLA row patching. `exec/kvrow.rs` MLA rows and `mla_live_nsplit` are consumed only by `exec/amd.rs`; `exec/gpu` would need the same write-row/nsplit handling.
- **MLA on sm_120**:
  - Missing ops: 51/52 (prefill and fold), 109/110 (FP8 latent) and 106 `MlaOutGate` (K3); same templates as above.
  - Missing in devgen: an NVIDIA target path. The MLA prefill emitter always emits op 51 (`mla.rs:9443,9600,10457`), and no NVIDIA opcode-coverage gate exists (`check_gfx950_opcode_coverage` runs only for AMD), so a CUDA MLA packet would trap at the first prefill.
  - The KV/runtime gaps are the same as for sm_90a.
- **MLA on CPU**: needs golden C bodies for ops 50/51/52/57 (none in `runtime/cpu/dev/golden`). The template is the Rust oracle `runtime/tests/mla_ref.rs`.
- **DSA on sm_90a**: rebuild with `-DPLOW_NV_DSA=1` for decode `IndexScore`/`IndexSelect` and `FlashGatherDecode` (54, needs MLA=1 too). Plus everything listed under sm_120.
- **DSA on sm_120**:
  - Prefill indexer ops are missing: `IndexScorePf`, `IndexSelectPf`, `IndexUnionPf` (AMD `interp.hip:5645-5742`) and `FlashGatherPrefill` 55. Templates: `op_attention_common.h` (`d_index_*`, the gather instantiation of the MLA prefill).
  - K3 pooled-indexer ops are missing: `IndexScoreKpool`, `DsaPoolCompress/Expand/Stash`, `DsaQQuant` (AMD, `PLOW_K3`).
  - KV: an index-key cache `kv.L.kidx` (per-row index head dim) needs a block-store row class or per-tensor Tail rows. The K3 pooled cache (one row per `ratio` tokens) needs a pooled class whose block boundary is in pool rows, not token rows.
  - `ctx_bound.rs` already knows `IndexScore*` for context bounds.
- **DSA on CPU**: golden bodies for the index and gather ops; none exist.
- **CSA on all backends**:
  - AMD: `op_compress.h` kernels exist and are included by `interp.hip:328`, but only op 213 `RopeInverseO` has a case (`interp.hip:6695`). Ops 212 `CompressPool` and 217 `CompressRopeQuant` need `case` arms and `GFX950_DISPATCHED` entries.
  - devgen: the `deepseek_v41` full emit refuses (`lib.rs:9533`); only `--block` plans emit.
  - KV: a compressed cache written on the `kv_source` layers and read by every layer; a new row class, since its rows are per compressed block.
  - NVIDIA/CPU: no kernels; port from `op_compress.h`.
- **Engram on all backends**:
  - AMD: `op_engram.h` (ops 214/215) is not even included by `interp.hip`, so it needs the include, the case arms and `GFX950_DISPATCHED`.
  - Host side: the hashed n-gram ids (tokenizer-derived map and lookback cache, per `dsv41_refusal`) have no runtime path in `exec/amd.rs` or `exec/gpu`.
  - No KV class; the lookback is per-request host state.
  - NVIDIA/CPU: port both ops from `op_engram.h`.
- **Hyper-connections on sm_90a/sm_120**:
  - Needs ops 128/129 with their modes, ported from `runtime/amd/op_hyperconn.h` (Sinkhorn mix, `hc_mix_decode.hip` decode fold) and the `PLOW_GLM_HC_POST_FOLD` variant (op 135 with 129 folded).
  - The residual stream is `hc_mult` x hidden, so every row-op width in the CUDA arms must accept the widened stream.
  - `kvrow.rs:305-309` patches `T` for both ops on AMD only, so the CUDA packed-prefill metadata needs the same row field. No KV/state class.
- **Hyper-connections on CPU**: golden bodies for 128/129; none.
- **AttnRes on sm_90a/sm_120**:
  - Needs op 104 from `runtime/amd/op_k3.h`. The CPU golden arm (`runtime/cpu/dev/golden/control.c:83`, AVX-512 `gemv.c:371`) is a second, exact template.
  - The block-residual ring (`blkres`) is excluded from the KV contract, so a prefix attach or chunk split must not span a block boundary without carrying it. The CUDA token-batch/packed-prefill path would need the same carry rule as AMD (not verified in detail).
- **AttnRes on CPU**: the dispatch exists but no CPU emit targets K3. It needs the K3 emitter (`k3.rs`, AMD-shaped) to allow a CPU target.
- **Recurrent state contract on CPU**: needs the `recurrent_state` buffers wired into `serve/cpu_serve.rs` (zero on admit, active mask), plus golden bodies for the GDN/KDA ops. None are registered in `runtime/cpu`.
- **Conv history on sm_120**: follows GDN below.
- **Conv history on CPU**: needs golden conv-state bodies (ops 136/143).
- **GDN on sm_120**:
  - The arms compile under `PLOW_NV_QWEN_GDN=1`, which the manifest requires (`manifest.rs:1569`).
  - The emitter builds the manifest for sm_90a only (`qwen35.rs:1360`, "NVIDIA-shaped").
  - The prefill scan is `plow_gdn_prefill.aot`, a CuTe export for SM90 (`scripts/export_gdn_prefill_sm90.py`) carried as a cubin and launched by `device/cuda/qwen_gdn.rs`. sm_120 needs its own export or the interpreter `QwenGdnConvPrefill` path.
- **GDN on gfx942**:
  - The ten arms exist (`op_qwen_gdn.h`), but only gfx950 builds set `PLOW_QWEN_GDN=1` (`build_gfx950.sh:271,438`).
  - devgen refuses AMD (`qwen35.rs:1376`, `AMD_QWEN_EMIT = false`, `AMD_GDN_PREFILL = false`): it needs an AMD emitter path, a chunked HIP prefill body plus its adapter, and the arms in a gfx942 build.
  - The recipe `recipes/infervisor/qwen3.5-gdn/gfx942-mi300x-tp1.toml` is `status = "refused"`.
- **GDN on CPU**: golden bodies for the GDN ops.
- **KDA on sm_90a/sm_120**:
  - Needs the decode ops (`KdaConv`, `KdaGate`, `KdaStateStep`, `KdaGatedNorm`) and the four chunked-prefill ops, ported from `runtime/amd` (`PLOW_K3`, `PLOW_KDA_CHUNK`).
  - State: KDA keeps its state in `kv.*` tensors on AMD. On CUDA that state would have to be declared in `recurrent_state` so the block store's State class captures it.
  - `kda.rs` is AMD-shaped (gfx950 checks at `kda.rs:689`).
- **KDA on CPU**: golden bodies; none.
- **Mamba-2 on sm_90a**: needs a rebuild with `-DPLOW_NV_MAMBA=1` (`op_mamba.cuh`, op 90). On every backend it also needs a full-model Nemotron emit, since `lib.rs:9624` refuses all but `--block`, plus state declared in `recurrent_state`.
- **Mamba-2 on gfx942/CPU**: no kernel. Port `op_mamba.cuh`.
- **Sigmoid/bias/group MoE routing on sm_90a/sm_120**: decode `MoeRouterTopk` (op 56) and `MoeRouter`
  (op 40) arms in `interp_sm120.cu` (only `MOE_ROUTER_TOPK_PF` exists, under `PLOW_NV_MOE_COMMON`);
  template `runtime/amd/op_moe.h` router; the GLM/DeepSeek emitters (`mla.rs`) are AMD-gated
  (`emit_is_amd()` sites), so devgen also needs an NVIDIA target path.
- **Sigmoid/bias/group MoE routing on CPU**: decode router entries in `runtime/cpu/dev/*` tables (only
  the `_PF` top-k, align and combine are there).
- **Generic MoE experts on sm_90a/sm_120**: `MoeExpertGlu`/`MoeExpertDown` (decode), `MoeGroupGluPf`/
  `MoeGroupDownPf` (prefill) and the `_FP8_BLK` twins; the Gemma grouped route (`moe_lt.rs`,
  `runtime/nvidia/moe_lt_sm90.cu`) is the template for a cuBLASLt grouped version; AMD
  `op_moe.h` / `moe_glu_a4w4_kw.h` for the native one.
- **Generic MoE experts on CPU**: expert GLU/down entries in the CPU tables (Gemma ones exist).
- **MXFP4 MoE (ops 150-153) on sm_90a/sm_120**: arms exist (`interp_sm120.cu:2878`,
  `interp_sm90a_mxfp4_moe.cu`) and devgen emits for non-AMD (`gptoss.rs`), but there is no production
  recipe or gate; needs a qualified GPT-OSS recipe and the FP32 gate. (Not verified end to end.)
- **MXFP4 MoE on gfx942**: AMD has no 150-153 arms; `check_gfx950_opcode_coverage`
  (`lib.rs:10201`) refuses the emit; AMD serves MXFP4 experts through its own grouped ops with
  the weight encoding in `i3` instead, so the GPT-OSS emitter would need to lower to those.
- **MXFP4 MoE on CPU**: CPU tables have 150-153; needs a recipe/gate.
- **Expert parallelism on NVIDIA/CPU**: an all-to-all collective arm (`XAllToAllHeads`, AMD-only,
  `interp.hip:7602`, gated by `PLOW_HAS_XALLTOALL_HEADS`) and multi-GPU MoE dispatch in
  `exec/tp.rs`; AMD itself is opt-in (`emit.moe_prefill_ep`, `emit.glm_ep`, both OFF/OptIn).
- **Muse gates**: arms exist only under `PLOW_QWEN_GDN` in `interp.hip` and serve on gfx950
  (`muse-spark-open` recipe); gfx942 needs the coverage gate lifted after a gfx942 gate run; NVIDIA
  and CPU need the four ops (226-229) ported from the AMD arms.
- **W8A16 / W8A8 on CPU**: tables carry `GEMV_FP8`, `GEMV_QKV_FP8`, `GEMM_*_FP8` (golden/avx512/neon
  `fp8.c`) but no FP8 CPU recipe; needs a CPU FP8 recipe and gate. W8A8 `GEMM_FP8`/`QUANT_FP8` CPU
  entries not verified.
- **Block FP8 128x128 on sm_90a/sm_120**: decode GEMV/dense-GLU arms exist (`GEMV_FP8_BLK`,
  `DENSE_GLU_FP8_BLK` under `PLOW_NV_GEMMA`); missing the prefill `GemmFp8Blk` (op 107) and the MoE
  `_FP8_BLK` expert ops; template AMD `d_gemm_t<...>` in `op_gemm_common.h`, or a cuBLASLt
  block-scaled route (13.x block scales, not verified).
- **Block FP8 32x32 (op 216)**: only `interp.hip` under `PLOW_DSV41_BLKFP8` (gfx950); every other
  backend needs the `d_gemm_fp8_mx` instantiation; the DeepSeek-V4.1 emit is refused everywhere
  except the block path (`dsv41_refusal`).
- **MXFP4 dense on sm_90a/sm_120**: decode `GemvMxfp4` exists (`PLOW_NV_MXFP4_PROJ`); missing
  `GemvGluMxfp4`, `GemvQkvMxfp4` and every prefill `Gemm*Mxfp4` arm; template `d_gemm_mxfp4` (AMD).
- **FP8 KV on CPU**: `FLASH_*_FP8` tables exist; needs a CPU FP8-KV recipe and gate.
- **MTP + verify on AMD/CPU**: the speculative serve path is `exec/gpu/spec.rs`, `spec_serve.rs`,
  `spec_draft.rs` only; AMD needs a draft-program launch, verify rows in the packed launch and
  KV rollback in `exec/amd.rs`; CPU the same in `exec/cpu/engine.rs`.
- **DSpark on every backend**: `spec_serve.rs:65` rejects block drafters; needs the block-draft
  verify loop and the drafter-context tail consumer (`spec.context.json`) in the serve path.
- **Multimodal on AMD/CPU**: ops 206-210 arms (`RMSNORM_F32`, `ROPE_AXIAL_F32`, `CHUNK_ATTENTION_F32`,
  `MM_ROWS_BF16`, `MM_SPAN_EXTENT`) and the `Embed` pad immediate; `mm::check_target`
  (`crates/devgen/src/mm.rs:32`) refuses non-`sm_` targets.
- **Encoder overlay / speech on AMD**: the f32 speech ops (`op_speech_f32.cuh`: 7 ops) and
  `in.encoder_overlay` handling (`exec/gpu/token_batch.rs`, `ctx_bound.rs`) have no AMD arms.
- **Speech on CPU**: CPU has `LENGTH_AFFINE_U32`, `REDUCE_ROWS_F32`, `STREAM_TAIL_F32`,
  `TRIM_SILENCE_F32`; missing `CHUNK_ATTENTION_F32`, `RMSNORM_F32`, `ROPE_AXIAL_F32`.

### NVIDIA → AMD candidates

Runtime features that exist on CUDA and are absent or partial on AMD (`exec/amd*`).

| Feature | CUDA (prod-next) | AMD (exec/amd*) | What AMD would need |
|---|---|---|---|
| KV block store row classes (Window / Tail / Scale / State, content-addressed blocks, device + host slabs) | `memory/kv_store.rs`, `exec/gpu/kv_blocks.rs`; the only CUDA boundary storage | not used: `kv_store` users are `exec/gpu*`, `serve/*`, `memory/vmm.rs` only; AMD keeps `SharedPrefix` (`exec/amd/shared_prefix.rs`): VMM-shared full blocks + slot snapshots, roles `ckv`/`krot`/`kidx`/`scale` | a `SnapBlocks` implementation over HSA memory (`enable_block_store` in `memory/vmm.rs` is backend-generic), row-class extraction for AMD layouts (latent `ckv`/`krot` as Tail, index keys, recurrent `state.*` as State), gather/scatter kernels (template `runtime/nvidia/kv_*`), and the boundary tree wired in `exec/amd/prefix.rs` |
| F1 copy path (per-block gather/publish descriptors in launch params, event-sealed publish, host restores ahead of admission, state attach in the first chunk gather) | `kv_blocks.rs` `blocks_publish`/`blocks_gather`/`blocks_seal_after`/`state_init_gather`; merge e75dc6c75, 054022109 | none (depends on the block store) | the block store first; then HSA signals in place of CUDA events for the seal, and gather descriptors in the AMD launch kernargs |
| Library attention route (FA4 hd512 via `lib_attention.rs`, one table upload per launch; `attention_gemm.rs`) | sm_90a, FP8 KV hd512 prefill; selected by the objects the packet carries (`PLOW_BUILD_FA4_HD512`) | AMD has vendor routes of its own kind: pinned AITER `.co` MLA decode and fmoe kernels, `amd_moe_aiter.rs`, `amd_sparse_mla.rs`; no library flash prefill for dense GQA | an AMD library flash prefill (e.g. a CK/AITER FA export) with the same packet-declared object selection and table upload; the adapter pattern of `kv_dequant_sm90.cu` for FP8 KV |
| Per-rung route selection (`[emit.rung_env]`, `segment_roles` per bucket) | devgen per-rung knobs (`.per_rung`, `PLOW_NO_GLU_FUSE`), CUDA segment roles | devgen side is target-generic; AMD object selection per bucket via `plow_config.h` `PLOW_HAS_*` (`interp.hip:144`) | not verified whether AMD loaders honor per-rung segment roles; needs a mixed-rung AMD emit test |
| Unified token batch: decode riders in packed prefill | `exec/gpu/token_batch.rs`, `riders.rs` | `exec/amd/token_batch.rs`: load-time gate for dense-GQA phase 2; KDA riders (`PLOW_AMD_RIDER_KDA`, `exec/amd.rs:2940`) | rider arms for MLA/DSA and GDN chains (AMD serves those models); the dense-GQA phase-2 pair is gated, not served by a recipe |
| Split-KV riders (`PLOW_TB_SPLIT_ATTN`, `plow_<arch>_rider_*`) | sm_90a decode objects | none | rider split-KV flash kernels in the AMD decode object (template `rider_flash512/256` in the sm_90a decode object) |
| Spec ride (`PLOW_SPEC_RIDE`, drafts riding the prefill launch) | `exec/gpu/spec*.rs` | none (no spec decode on AMD) | spec decode on AMD first |
| Packed admission, in-flight prefix wait (`PLOW_PREFIX_INFLIGHT_WAIT`), chunk publish | `exec/gpu/prefix.rs`, CUDA packed prefill | none found in `exec/amd*` | the block store's in-flight publish; AMD packed admission in `exec/amd/packed.rs` |
| Stage fusion: decode router windows as light launches (`plow_<arch>_light_router`) | `exec/gpu/cublaslt.rs` (light router), `interp_sm120.cu:4119`; merge 1a1f1e06f | none (`light_router` absent from `runtime/amd`, `exec/amd*`) | an AMD light-launch kernel for the router window opcodes (`plow_light_router_ops` set) and host launch between hipBLASLt GEMMs in `amd_gemm_lt.rs` |
| Prefix-cache host tier | block store host slab, `host_tier_default` | present: `shared_prefix.rs:464-494` arms an AMD host tier from `vmm_host_cache_bytes()` (snapshot host tier, `0`/unset = off) | align the default (CUDA auto-sizes; AMD off unless set) after the block store lands |
| Live KV (frontier-committed KV, `PLOW_VMM_LIVE`) | `live_kv` section, 4096-row commit blocks | present: `SharedPrefix::ensure_rows` (`shared_prefix.rs:528`) and `PLOW_VMM_KV` (VMM KV on ROCr) | parity exists in substance; the packet `live_kv` section is CUDA-read (not verified on AMD) |
