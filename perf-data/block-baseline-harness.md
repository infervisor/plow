# Single-block PyTorch/vLLM baseline harness

The **baseline** side of the block harness. The plow side is
`block_run <asset> bench` (`crates/plowrt/examples/block_run.rs`), which times
ONE compiled transformer block on plow's runtime and writes `sweep.json`. This
harness times the **same single block** through a tuned reference path and writes
a `sweep.json` with the **identical row schema**, so the two diff directly.

- `scripts/block_baseline.py` — the harness. Two backends: `--backend torch`
  (SDPA + cuBLAS) and `--backend vllm` (vLLM's shipped fused kernels).
- `scripts/block_baseline.sh` — wraps it in `gpulease` (advisory flock) so
  concurrent agents on the one card never contend a timing run.

Driven by the plow-native block descriptors in `crates/plowc/examples/*.json`
(the same files `block_sim.sh` compiles), so block geometry cannot drift from
what plow compiles. Covers the two blocks in the ask:

| descriptor | block | geometry |
|---|---|---|
| `transformer_block_gemma4_12b.json` | Gemma-4-12B dense GeGLU | hidden 3072, 12Q/4KV heads, hd 256, inter 12288 |
| `moe_gemma4_26b_a4b.json` | Gemma-4-26B-A4B MoE | hidden 2560, 20Q/4KV, hd 256, inter 8192, 8 experts top-2 |

## Why eager PyTorch is the WRONG baseline

At decode (M=1) a single block is ~15-20 tiny ops. Eager mode launches each as a
separate kernel (~5-8 us), so **~100-150 us/step is pure launch overhead**, not
compute. vLLM erases this with CUDA graphs (`cudagraph_mode FULL_AND_PIECEWISE`
in the repo's vLLM baselines); plow erases it with its fused packet. An eager
baseline overstates block latency and flatters plow.

Measured here (12B dense, B1 T1024, H100 NVL, bf16):

| path | median/step | note |
|---|---|---|
| eager | **578 us** | launch-bound |
| CUDA graph | **232 us** | compute-bound — **2.5× lower** |

So the harness **captures the decode step into a CUDA graph by default** (the
same technique vLLM uses). `--no-cudagraph` reveals the eager number for
contrast. Using eager would have handed plow a fake 2.5× win.

## Making it "right" against vLLM

A single block cannot be *served* by vLLM in isolation, so the vLLM-grounded
per-block floor is derived from the measured full-model decode latency:

```
per_block_us  ~=  (TPOT_ms - fixed_overhead_ms) * 1000 / num_layers
```

`fixed_overhead` = embed + final norm + lm_head + sampling (layer- and
context-independent). Pass `--layers` and `--vllm-tpot-ms` and the harness prints
its per-block number next to that implied floor. If `harness * layers ~= TPOT`,
the isolated baseline is trustworthy.

The repo's vLLM TPOT numbers to anchor against live in
`perf-data/gemma4-12b-vllm-sm120.{md,json}` and
`perf-data/gemma4-26b-a4b-vllm-sm120.{md,json}`.

### Caveats the comparison depends on

1. **GPU must match.** The repo's vLLM baselines are on **RTX PRO 6000 Blackwell
   (sm_120)**; this harness reports on whatever CUDA device it runs (an H100 NVL
   in the current box). Cross-GPU block numbers are **not** comparable — the
   anchor is only valid when vLLM was measured on the same card.
2. **MoE geometry is approximate.** `moe_gemma4_26b_a4b.json` is a linear-chain
   approximation: **8 experts / top-2, hidden 2560**. The real Gemma-4-26B-A4B
   that vLLM serves is **128 experts / top-8, hidden 2816, 30 layers**. Expert
   count and top-k drive fused-MoE efficiency directly (see Finding 3), so the
   MoE numbers here describe the *descriptor's* block, not the shipped model.
   Fix the descriptor to the real geometry before comparing MoE against a served
   vLLM number.
3. **Attention backend.** `torch` uses `scaled_dot_product_attention`; `vllm`
   uses `flash_attn_varlen_func`. The repo's *served* vLLM baselines forced
   `TRITON_ATTN` (Gemma-4 heterogeneous head dims, FA4 unavailable), which is a
   third kernel again — same algorithm, different implementation.

## Measurement protocol (mirrors `block_run bench`)

**Both phases are measured, on both backends, over the same B x T grid:**

| phase | what runs | statistics |
|---|---|---|
| **prefill** | one full-block pass over `[B,T,H]` — norm, QKV, **causal** attention over T, o_proj, norm, FFN | warmup, then `--prefill-iters` timed passes → median / p95 ms + tok/s (`B*T/median`) |
| **decode** | one full-block step over `[B,1,H]` against the T-row KV cache | `--warmup`, then `--iters` timed steps → median / p95 us + tok/s (`1e6/median*B`) |

Both timed with CUDA events (pure GPU time). Weights are random (numerics are
irrelevant — the per-step KERNEL time is data-independent, exactly as
`block_run.rs` notes), so no checkpoint / HF download is needed.

> **Fixed (was a defect):** earlier revisions timed prefill as a *single
> un-warmed sample*, and the vLLM backend's `prefill()` ran only norm+QKV to fill
> the KV cache — no attention, no o_proj, no FFN. That made `prefill_ms`
> incomparable across backends (vLLM appeared ~2.4 ms vs torch ~38 ms at the same
> point purely because it did a fraction of the work). Both backends now run the
> identical full-block prefill with identical statistics. The decode tables were
> never affected. Row keys are now `prefill_ms_median` / `prefill_ms_p95` /
> `prefill_tok_s` (the old scalar `prefill_ms` is gone).

During the timed window the attended context is held fixed at T (the +iters
growth is immaterial and keeping shapes static is what makes the step
CUDA-graph capturable).

## Run

```bash
# 12B dense, anchored to the 12B vLLM TPOT (48 layers, 19.78 ms/token @ ctx 1k):
./scripts/block_baseline.sh crates/plowc/examples/transformer_block_gemma4_12b.json \
  -- --batch 1,4 --ctx 128,1024,4096 --layers 48 --vllm-tpot-ms 19.78 \
     --out /dev/shm/block-baseline/gemma12b.json

# 26B MoE:
./scripts/block_baseline.sh crates/plowc/examples/moe_gemma4_26b_a4b.json \
  -- --batch 1,4 --ctx 128,1024,4096 --out /dev/shm/block-baseline/gemma26b-moe.json

# same block through vLLM's fused kernels (needs a vLLM venv):
PLOW_PY=/workspace/venvs/vllm-blk/bin/python \
./scripts/block_baseline.sh crates/plowc/examples/moe_gemma4_26b_a4b.json \
  -- --backend vllm --batch 1,4 --ctx 128,1024,4096 --out /dev/shm/block-baseline/26b-vllm.json
```

## Backends: `--backend torch` vs `--backend vllm`

Rather than driving a vLLM *server*, `--backend vllm` calls vLLM's **shipped
kernels directly** on the isolated block — no HTTP, no scheduler, no model
loading:

| piece | `--backend torch` | `--backend vllm` |
|---|---|---|
| QKV / o / FFN GEMMs | `nn.Linear` → cuBLAS | same (vLLM's unquantized linear is also cuBLAS) |
| attention decode | `scaled_dot_product_attention` | `vllm.vllm_flash_attn.flash_attn_varlen_func` |
| MoE FFN | `top_k` dense expert GeGLUs (FLOP-equivalent proxy) | `fused_topk` + `fused_experts` (real routing, Triton grouped GEMM, `GELU_TANH`) |
| dense GeGLU act | `F.gelu(tanh) * up` | same |

The vLLM backend needs vLLM importable; point `PLOW_PY` at a vLLM venv:

```bash
python3 -m venv /workspace/venvs/vllm-blk
/workspace/venvs/vllm-blk/bin/pip install vllm==0.25.1   # brings its own torch
PLOW_PY=/workspace/venvs/vllm-blk/bin/python ./scripts/block_baseline.sh <desc> -- --backend vllm ...
```

Both backends CUDA-graph the decode step successfully on H100 (fused MoE
included). If a kernel ever refuses capture the harness falls back to eager for
that point and prints `[eager-fallback]` — never silently.

### Reference sweep (H100 NVL, bf16, CUDA graph, iters 100) — median us/step

All rows below are from **uncontended** runs (`gpulease` rc=0). An earlier pass
of the vLLM sweeps returned rc=76 (a foreign 666 MiB process appeared mid-run)
and was discarded and re-measured — contended timings are not reported.

**Gemma-4-12B dense block**

| ctx | torch B1 | vllm B1 | torch B4 | vllm B4 |
|---|---|---|---|---|
| 128 | **230.4** | 232.7 | **247.2** | 255.3 |
| 1024 | **232.5** | 237.9 | **252.3** | 264.7 |
| 4096 | **238.1** | 252.5 | **273.6** | 281.2 |

**Gemma-4-26B-A4B MoE block**

| ctx | torch B1 | vllm B1 | torch B4 | vllm B4 |
|---|---|---|---|---|
| 128 | **267.2** | 267.9 | **281.9** | 381.7 |
| 1024 | **268.9** | 271.7 | **288.1** | 424.4 |
| 4096 | **275.2** | 287.9 | **309.3** | 483.8 |

### Findings

1. **Decode is GEMM-bound.** Latency is nearly context-flat (128 → 4096 costs
   only ~3-8 %) and batching amortizes well. The block's cost is the projection
   and FFN GEMMs, not attention, at these contexts.
2. **On the dense block the two backends agree within 1-6 %** — expected, since
   both issue the same cuBLAS GEMMs and only the attention kernel differs
   (SDPA vs flash-attn varlen). torch edges ahead slightly at long ctx.
3. **vLLM's fused MoE is the WRONG kernel at single-block decode batch.** At
   B=4 it is 35-56 % *slower* than dense expert GEMMs (e.g. 423.7 vs 288.1 us
   at ctx 1024, 424.4 vs 288.1). With 4 tokens × top-2 spread over 8 experts, the Triton
   grouped GEMM degenerates into many tiny per-expert GEMMs plus sort/scatter
   overhead; `fused_experts` is built for large token counts. At B=1 the two
   converge. **Caveat:** the two MoE paths are not semantically identical — the
   torch path is a FLOP-equivalent proxy running a fixed `top_k` expert set with
   no gather/scatter, while vLLM's does real routing. Read torch as "what an
   ideal fused implementation could reach" and vLLM as "what the best available
   library actually achieves".

**The baseline plow must beat is the better of the two per point** (best-of-breed
reference), not whichever backend flatters plow. These are H100 numbers — re-anchor
against a same-GPU vLLM TPOT before claiming any plow-vs-vLLM ratio.


---

# Framework harness (`scripts/block_layer_bench.py`)

Supersedes the hand-written block above for anything vLLM implements. Drives
**vLLM's own decoder layer classes** standalone — no engine, no server, no
checkpoint. Adding a model is adding a JSON config in `perf-data/block-configs/`,
not code:

```
HF config .json -> architectures[0] -> vLLM model module -> *DecoderLayer
```

## Coverage (vLLM 0.25.1)

| architecture | module | layer class | runs |
|---|---|---|---|
| `Gemma4ForCausalLM` | `gemma4` | `Gemma4DecoderLayer` | **yes** |
| `Glm4MoeForCausalLM` | `glm4_moe` | `Glm4MoeDecoderLayer` | not yet |
| `GlmMoeDsaForCausalLM` (GLM-5.2) | `deepseek_v2` | `DeepseekV2DecoderLayer` | not yet |
| `DeepseekV3/V32ForCausalLM` | `deepseek_v2` | `DeepseekV2DecoderLayer` | not yet |
| `KimiLinearForCausalLM` | `kimi_linear` | `KimiDecoderLayer` | not yet |

**Kimi-K2 is NOT `kimi_linear`.** There is no KimiK2 class in vLLM's registry —
K2 ships `DeepseekV3ForCausalLM`, i.e. `DeepseekV2DecoderLayer`, the same shared
MLA+MoE layer as GLM-5.2 and DeepSeek-V3.2. (`crates/rewrite/src/kimi.rs` agrees:
"Kimi K2 / DeepSeek V2-V3 decode block".) `kimi_linear` is the separate Kimi
*Linear* model.

The MLA family resolves but does not yet run here — it needs a different
constructor signature (`(vllm_config, prefix, config=None, ...)`), a **real**
`ModelConfig` (it dereferences `model_config.use_mla` unconditionally), and MLA
attention plumbing (`self_attn.mla_attn.mla_attn`, `FLASH_ATTN_MLA`, head_size
576 / 1 KV head, a 3-D KV cache, builder-produced `FlashAttnMLAMetadata`).
`kv_lora_rank + qk_rope_head_dim` must be 320 or 576.

## Results — H100 NVL, bf16, CUDA graph, uncontended (`gpulease` rc=0)

decode = us/step median; prefill = ms median, one full-block pass over B x T.

| block | B | T | decode us | prefill ms |
|---|---|---|---|---|
| Gemma-4-12B dense | 1 | 128 | 271.4 | 1.12 |
| Gemma-4-12B dense | 4 | 128 | 295.3 | 1.28 |
| Gemma-4-12B dense | 1 | 1024 | 277.3 | 1.58 |
| Gemma-4-12B dense | 4 | 1024 | 322.1 | 4.41 |
| Gemma-4-31B dense | 1 | 128 | 340.7 | 1.20 |
| Gemma-4-31B dense | 4 | 128 | 358.7 | 1.53 |
| Gemma-4-31B dense | 1 | 1024 | 336.2 | 2.02 |
| Gemma-4-31B dense | 4 | 1024 | 390.1 | 6.91 |
| Gemma-4-26B-A4B MoE | 1 | 1024 | 398.7 | 2.52 |
| Gemma-4-26B-A4B MoE | 4 | 1024 | 529.1 | 7.31 |

MoE uses the **real** 128-expert / top-8 / hidden-2816 geometry, not the
8-expert/top-2/hidden-2560 approximation in `crates/plowc/examples/`.

## Why this beats the hand-written block

It measures ~17% **higher** than `block_baseline.py` on the same 12B block
(277 vs 231 us at B1/T1024) because vLLM's real layer includes RoPE, qk_norm and
the true sandwich-norm structure the hand-written block omitted. Higher, and
correct — the hand-written number was an understatement.

## Init steps a standalone layer must replicate

These are things `GPUModelRunner` does; skip them and layers fail, MoE loudly:

* `process_weights_after_loading()` on every `quant_method` — selects the fused
  MoE kernel. Without it: `assert self.moe_kernel is not None`.
* `init_workspace_manager(device)` — fused-MoE scratch. Without it:
  `WorkspaceManager not initialized`.
* Build under bf16 as the **default dtype** (it drives backend selection), then
  move device only — do **not** blanket-cast. `Glm4MoE.gate` is deliberately
  fp32; casting it gives `expected mat1 and mat2 to have the same dtype`.
* `set_forward_context(..., slot_mapping={layer_name: tensor})` — the kwarg is
  mandatory; without it the KV cache is **silently never written**.
* Enter `set_forward_context` ONCE, outside the timed region. vLLM enters it per
  model forward and amortizes over all N layers; timing it per layer call
  overstated decode 4.8x (1117 -> 271 us).


## Both sides now time prefill AND decode

`block_run bench` previously prefilled only to set up the KV cache and timed
decode alone, so `--phase prefill` had nothing to join against. It now times the
prefill phase with the same warmup / median / p95 treatment as decode and emits
`prefill_ms_median` / `prefill_ms_p95` / `prefill_tok_s` — the same keys the
baseline harnesses emit. `--prefill-iters` (default 10) controls it.

Only `prefill_slot` is inside the timer; `begin_slot` and the `act.x` upload are
setup, matching the baseline's compute-only prefill. `prefill_slot` loops
`prefill_chunk` until Done, whose path ends in a **D2H token download**, so it
synchronizes — the wall-clock number is real execution, not a launch. (That
property is what makes the existing decode timing valid too.)

So the full single-block matrix is now symmetric:

| | plow `block_run` | baseline (`block_layer_bench` / `block_baseline`) |
|---|---|---|
| decode, batched, over B x T | yes | yes |
| prefill, over B x T | **yes (new)** | yes |
| emits `sweep.json` | yes | yes |
| joined by `block_compare.py --phase {decode,prefill}` | yes | yes |

### Why single-block is the right unit

A full-model run is expensive and mixes in embed / lm_head / sampling; one block
is the repeating unit, so per-block x num_layers reconstructs the model-level
number (that is exactly the `TPOT_ms / num_layers` anchor above, run in reverse).
It also isolates the thing being optimized. For deeper attribution than a single
latency number, a `-DPLOW_NV_TRACE=1` cubin adds a per-op cycle profile
(`trace_reset` / `trace_summary` in `block_run`), so a block sweep can be broken
down at the interpreter-op level rather than just compared end to end.
