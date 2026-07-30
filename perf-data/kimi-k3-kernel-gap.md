# Kimi-K3 vs plow's gfx950 KERNEL inventory — audit and ranked gap list

> **UPDATE 2026-07-28 — five items in §10 are CLOSED, measured on real weights on one gfx950.**
> See `plans/kimi-k3-block-impl.md` for the residual tables.
>
> | §10 | item | status |
> |---|---|---|
> | 1 | `PLOW_MOE_MAX_TOPK` 8 -> 16 | **CLOSED and EXERCISED.** Raised earlier; this is its first hardware run. 896 experts / top-16, routing SET and ORDER both exact against the reference, gates to 1.0e-3. |
> | 3 | `situ` activation | **CLOSED for decode.** NOT a third `act` code in the `act(g)*u` epilogues (§6's plan) — situ transforms the UP branch, so the expression shape changes. `PLOW_DOP_SITU_GLU = 105` for dense/shared, `PLOW_MOE_ACT_SITU = 2` + a pair-form `moe_glu` for the routed experts, betas in the free `f0`/`f1`. `moe_act` returns **NaN** for code 2 so the two unconverted PREFILL epilogues (`op_moe.h:1285`, `:1584`) poison rather than silently computing gelu. Register cost across all four objects: **zero**. |
> | 5 | Latent-MoE graph | **CLOSED for decode.** The kernels needed nothing (H is a runtime operand; 3584/128 and 3584/32 are exact). One kernel line changed: `d_moe_combine`'s `residual` is now OPTIONAL — it was an unconditional null deref, and a latent-width combine has no hidden-width residual to add. |
> | 7 | Residual-attention block | **CLOSED.** `PLOW_DOP_ATTN_RES = 104`, one packet, `score_weight` folded at prep time as §1b(g) predicted. |
> | 9 (part) | mxfp4 nibble order, §4c | **CLOSED ON HARDWARE.** `PLOW_DOP_GEMV_MXFP4` on the raw bytes of `layers.1...experts.0.w1`: **1.648e-03** for low-nibble-is-even-k against **1.408e+00** for the swap. plow's layout is right; every "COVERED" mxfp4 verdict stands. |
>
> **Two corrections to this document, both found by implementing it:**
> * **§6's site list has drifted.** The GLU ternaries are at `op_gemm.h:579, 1042, 1479, **1920**,
>   **2598**` — 1864 and 2531 are the enclosing template declarations — and the list omits
>   `op_elementwise.h:68` and `:75` (`d_glu`, the unfused `Glu` op). `mla.rs`'s own `K3Gap` text
>   carries the same stale numbers.
> * **§5e's list of expert-width sites is incomplete.** It names six; it omits the four decode
>   `MoeCombine` `d.i[0] = h` sites and the two prefill combine sites. All current line numbers are
>   ~+70 from those printed here.
>
> **And one finding that changes where a K3 gate has to look.** §1b(g) describes AttnRes correctly
> but not its detectability. At a SNAPSHOT layer (`l % 12 == 0`, i.e. 8 of 93) the block output is
> `attn + ffn` and a plain wiring differs by **1.0**. At every OTHER layer `prefix = prefix_in +
> attn`, so the block output is `prefix_in + attn + ffn` — **exactly what a plain residual
> produces**, measured at **3.0e-3** on real layer-1 weights against 8.1e-1 at the AttnRes outputs
> themselves. A block-output-only gate does not see AttnRes at 85 of 93 layers.


Scope: the standing directive is *review the existing kernels first; add only if the inventory is
not enough.* This is that audit for Kimi-K3. **No kernel was written or modified.** The only thing
run on hardware-adjacent tooling was `hipcc -Rpass-analysis=kernel-resource-usage` (compile-only,
no GPU, §11).

Evidence base, in descending order of trust:

1. **The checkpoint on disk** — 41 of 96 shards downloaded; every safetensors header parsed for
   names/dtypes/shapes, and expert scale bytes decoded. This is the authority for layout.
2. **`modeling_kimi_linear.py`** fetched from `moonshotai/Kimi-K3` (1314 lines, `/tmp/modeling_kimi_linear.py`)
   and `configuration_kimi_k3.py` on disk. Authority for semantics.
3. **plow source at `worktree-readme-build-instructions` @ 63e9453**, quoted with file:line.

Denominator throughout is **6200 GB/s measured** (contract §5), never the 8 TB/s datasheet number.

---

## 0. TL;DR

1. **MXFP4 is COVERED, and by more than the brief assumed.** Not just `GemvMxfp4`/`GemvGluMxfp4`/
   the five mxfp4 GEMM rungs — plow already has an **mxfp4 MoE expert path** (`wave_dot_mxfp4`,
   `PLOW_MOE_ENC_MXFP4 = 2`, `op_moe.h:376-600`) and an **A4W4 grouped expert prefill GEMM**
   (`op_moe.h:1241+`). The encoding is already a runtime instruction field (`i[6]` decode / `i[3]`
   prefill) with an emitter selector. K3's on-disk layout is **byte-exact**: `weight_packed`
   `[N, K/2]`, `weight_scale` `[N, K/32]` u8, group 32 along K, E8M0 **bias 127 confirmed
   empirically from the checkpoint** (§4). Nothing to build. This is a large de-risk.
2. **The brief's framing of the bandwidth is inverted.** K3 quantizes routed experts ONLY; the
   `ignore` list exempts every `self_attn.*`. **KDA + MLA attention projections are 53 % of the
   decode weight stream and are bf16; the routed experts are 19 %** (§2). The ops that dominate
   K3 decode are exactly `Gemv`/`GemvGlu`/`GemvQkv`, which the sibling review measured at
   **83–106 % of ceiling**. The single biggest thing plow already has right *is* the thing K3
   needs most.
3. **The one genuinely new kernel is KDA** (69 of 93 layers). plow has **no state-carrying op on
   AMD at all** — `Mamba2Scan` (op 90) exists in `dev_isa.h` and is emitted, but its only
   implementation is `runtime/nvidia/op_mamba.cuh` and it is marked *UNVERIFIED ON GPU*; there is
   no `PLOW_DOP_MAMBA2_SCAN` arm in `runtime/amd/interp.hip`, so on gfx950 it falls to the silent
   `default:`. §7 gives the register/LDS verdict: **the state does not fit the decode megakernel's
   8 VGPRs of headroom**, and the answer is a 4th co-resident code object, not a bigger arm.
4. **Four correctness blockers are cheap and mechanical**: `PLOW_MOE_MAX_TOPK 8u` (K3 wants 16),
   the `situ` activation, the latent-MoE width, and — the one nobody would have looked for —
   **K3's MLA has NO RoPE at all** (`self.rotary_emb = None`, `assert self.use_nope`), while
   plow's MLA emitter emits two `HeadNormRope` packets per layer unconditionally. That is §4's
   recurring bug shape pointed the other way: a correct arm applied where the model does not want
   it, producing silently wrong logits.
5. **Two structures the brief did not know about** and that no existing op expresses:
   a **latent MoE** (routed experts run at width 3584, not hidden 7168, behind a down/norm/up
   sandwich), and a **residual-attention block** (`_apply_attn_res`: a softmax over ≤9 layer
   snapshots, twice per layer, replacing the plain residual add). Both are §5/§10 items.
6. Total ranked list in §10. **Nothing in the top 6 is a GEMM/GEMV kernel.** The kernel inventory
   is, for the memory-bound majority of this model, already enough.

---

## 1. What Kimi-K3 actually is (verified, not from the brief)

`config.json` at `~/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/`.

93 layers, hidden 7168, 96 heads, vocab 163840, `max_position_embeddings` 1048576.
Layer 0 dense (`first_k_dense_replace: 1`, `intermediate_size` **33792**); layers 1-92 MoE.
Layers 3,7,11,…,91 (0-based) are **MLA** (24); the other 69 are **KDA**. Verified against the
shard headers: layer 3 carries `q_a_proj/kv_a_proj_with_mqa/kv_b_proj`, layer 4 carries
`q_conv1d/A_log/dt_bias`.

### 1a. Things the brief got right

| | |
|---|---|
| MLA | `kv_lora 512`, `q_lora 1536`, `qk_rope 64`, `qk_nope 128`, `v_head 128` — `q_b_proj [18432,1536]` = 96×192 ✅, `kv_b_proj [24576,512]` = 96×256 ✅ |
| KDA | `head_dim 128`, `num_heads 96`, `short_conv_kernel_size 4`, `use_full_rank_gate true`, `gate_lower_bound -5.0` |
| MoE | 896 experts, top-16, `moe_intermediate_size 3072`, sigmoid, `noaux_tc`, `moe_renormalize`, `routed_scaling_factor 1.0` |
| quant | `mxfp4-pack-quantized`, group 32, symmetric, `scale_dtype uint8`, routed experts only |

### 1b. Things the brief did NOT have, all load-bearing

**(a) `use_grouped_topk` is a NO-OP at these values.** `num_expert_group: 1`, `topk_group: 1`.
`KimiMoEGate.forward` gates the whole group-mask branch on
`if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:` → **False**. So the
router is plain sigmoid + bias-on-selection + top-16 + renormalize. plow's `MoeRouterTopk` flag
word for GLM is `GLM_ROUTER_FLAGS = 1|2|4` (`mla.rs:388-390`) and K3 wants **the same 1|2|4**.
Group-limited routing is implemented (`moe_group_mask`, `op_moe.h:194-232`) and simply unused.

**(b) LATENT MoE — the routed experts do not run at `hidden`.** From the shard headers:

```
block_sparse_moe.routed_expert_down_proj.weight  BF16 [3584, 7168]
block_sparse_moe.routed_expert_norm.weight       BF16 [3584]
block_sparse_moe.experts.E.w1.weight_packed      U8   [3072, 1792]   # [I_moe, K/2], K=3584
block_sparse_moe.experts.E.w2.weight_packed      U8   [3584, 1536]   # [K, I_moe/2]
block_sparse_moe.routed_expert_up_proj.weight    BF16 [7168, 3584]
```
`KimiSparseMoeBlock.forward`: `x → down_proj(7168→3584) → moe_infer(896/top-16, I=3072) →
routed_expert_norm(3584) → up_proj(3584→7168) → + shared_experts(identity)`.
`routed_expert_hidden_size: 3584` is the expert width. **This is why `w1` is `[3072, 3584]` and
not `[3072, 7168]`.**

**(c) 2 shared experts are ALREADY FUSED in the checkpoint.**
`shared_experts.gate_proj [6144, 7168]` = `moe_intermediate_size * num_shared_experts` = 3072×2.
Confirmed in the model code: `intermediate_size = config.moe_intermediate_size * config.num_shared_experts`.
So "2 shared experts" is one 2×-wide shared MLP on disk — no new op, only an emitter width.

**(d) K3's MLA has NO RoPE.** `KimiMLAAttention.__init__` ends with

```python
        assert self.use_nope
        ...
        self.rotary_emb = None
```
and `forward` does `query_states = torch.cat((q_pass, q_rot), dim=-1)` with **no rotation applied
to `q_rot`/`k_rot`**. `config.json` has **no `rope_theta` and no `rope_scaling`** — consistent.
The 64 "rope" dims are carried as extra un-rotated channels. `scaling = q_head_dim**-0.5 = 192**-0.5`.
Position information lives entirely in the 69 KDA layers.

**(e) MLA has an OUTPUT GATE.** `mla_use_output_gate: true`; layer 3 carries
`self_attn.g_proj.weight BF16 [12288, 7168]`, and

```python
        if self.use_output_gate:
            g = self.g_proj(hidden_states).sigmoid()
            attn_output = attn_output * g
        attn_output = self.o_proj(attn_output)
```
i.e. one extra N=12288 K=7168 GEMV + a sigmoid-multiply per MLA layer, applied on the
`[nh, v_head]` attention output *before* `o_proj` — which in plow's absorbed path is exactly the
output of `MlaMergeFold`.

**(f) `situ` is SiTU-GLU and it transforms BOTH branches** (§6).

**(g) RESIDUAL-ATTENTION BLOCKS.** Every layer carries `self_attention_res_norm[7168]`,
`self_attention_res_proj[1,7168]`, `mlp_res_norm[7168]`, `mlp_res_proj[1,7168]`
(verified present on all 41 downloaded layers). `attn_res_block_size: 12`. The plain
`residual + x` is replaced by `_apply_attn_res`:

```python
def _apply_attn_res(prefix_sum, block_residual, proj, norm):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)   # [T, nb+1, H]
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)
```
A snapshot of the running sum is pushed onto `block_residual` every 12 layers, so `nb ≤ 9`. The
op is: RMS-normalize each of `nb+1` rows of 7168, dot each against one fused per-channel vector
(`norm.weight * proj.weight` — **foldable at load into a single [7168] f32**), softmax over
`nb+1`, then output the probability-weighted mix of the **raw** rows. Twice per layer, 186 times
per token.

**(h) A_log shape disagrees with the modeling code.** The module declares
`torch.empty(self.num_heads)` = `[96]`; the checkpoint has `self_attn.A_log F32 [128]`
(= head_dim). `o_norm.weight` is `[128]` and `dt_bias` is `[12288]`. Flagged for the KDA sibling —
it changes what the decay is indexed by. Do not assume `[96]`.

---

## 2. THE BANDWIDTH REFRAME — read this before ranking anything

The `quantization_config.ignore` list is `re:.*self_attn.*`, `re:.*shared_experts.*`,
`re:.*mlp\.(gate|up|gate_up|down)_proj.*`, `re:.*lm_head.*`, vision. **Only
`block_sparse_moe.experts.*` are mxfp4.** Everything else is bf16, verified in the shard headers.

Decode weight stream per token, whole model, computed from the on-disk shapes
(mxfp4 counted at 4.25 bits/element = weight nibble + 1 E8M0 byte per 32):

| what | GB/token | share | plow's op | measured % of 6200 GB/s ceiling |
|---|--:|--:|---|---|
| **KDA attention projections** (q,k,v,g,o @ [12288,7168] ×69) | **57.05** | **44.7 %** | `Gemv` / `GemvQkv` | **83–106 %** |
| **MLA attention** (incl. `g_proj` 168 MB/layer) | 10.38 | 8.1 % | `Gemv`/`GemvQkv` | 83 % |
| routed experts, top-16, mxfp4 | 24.06 | 18.9 % | `MoeExpertGluFp8Blk` w/ `enc=2` | 33–42 % (GLM proxy) |
| shared experts (I=6144, bf16) | 22.64 | 17.7 % | `GemvGlu` + `Gemv` | 83 % |
| latent down/up (3584↔7168) | 8.80 | 6.9 % | `Gemv` | 83 % |
| router `[896,7168]` | 1.10 | 0.9 % | `Gemv` | see §5c |
| lm_head `[163840,7168]` | 2.19 | 1.7 % | `Gemv` | **106 %** |
| dense L0 (I=33792) | 1.35 | 1.1 % | `GemvGlu`+`Gemv` | 83 % |
| **total** | **127.6** | | | |

| | GB/rank | floor @ 6200 GB/s |
|---|--:|--:|
| TP4 | 31.89 | 5.52 ms/token |
| TP8 | 15.95 | **2.76 ms/token** |
| TP16 | 7.97 | 1.38 ms/token |

Total weights (all 896 experts resident): **≈1.56 TB** → TP8/EP8 on one 8×MI355X node (2.3 TB) fits.

> **CORRECTION 2026-07-30 — the per-rank column above assumes every row is sharded, and two rows
> were not.** `routed_expert_down_proj` and `routed_expert_up_proj` were emitted REPLICATED, so
> the 8.80 GB "latent down/up" row cost `2 × 4.727 = 9.454` GB per RANK rather than 1.10 —
> **52 % of the decode bf16 `Gemv` stream**, measured off the blob rather than derived
> (`plowrt disasm --program 1 --format json`, summing `N*K*2` over every `Gemv`):
>
> | bf16 `Gemv` weight bytes/rank/token | replicated `up` | column-parallel `up` |
> |---|--:|--:|
> | `routed_expert_up_proj` | 4.727 GB | **0.591 GB** |
> | `routed_expert_down_proj` | 4.727 GB | 4.727 GB |
> | `lm_head` (still replicated; `PLOW_K3_SHARD_HEAD=1` exists, off) | 2.349 GB | 2.349 GB |
> | everything else | 6.355 GB | 6.355 GB |
> | **total** | **18.157 GB** | **14.021 GB** |
>
> `up` is now column-parallel and its all-gather is folded into the shared expert's all-reduce, so
> the shard adds no packet and no collective (`emit_k3_latent_moe`). `down` stays replicated: its
> output feeds experts sharded on their INTERMEDIATE, so every rank needs the whole latent and any
> shard of it needs a rendezvous of its own — ~0.74 ms/token of streaming against ≥5.3 µs × 92
> layers of added packet, which is close to a wash. The remaining replicated 7.076 GB
> (`down` + `lm_head`) is where the next 1.1 ms of floor lives.
>
> **The shard is BIT-NEUTRAL and it took a one-line kernel change to make it so.** Column-parallel
> splits the OUTPUT, so every element of `yh` is still one dot product over the whole 3584-wide
> latent — the values never change. What did change was the ROUNDING of the sum it is added to:
> unfolded, the collective stored `f2bf(sum_r shd_r)` and a separate `Residual` re-read it, and
> the folded gather kept that sum in f32. One bf16 ULP per element per layer, 92 layers, and the
> 24-token gate answered ". The capital of X is Y. The capital of …" instead of the reference
> continuation. `d_xreduce` now rounds before the gathered term is added and the two emits are
> token-identical, all 8 ranks, on both the decode-only and the full asset.
>
> **MEASURED, gfx950 TP8, 200 steps, interleaved from ONE binary** (`PLOW_K3_SHARD_UP=0` emits the
> control blob, so the two assets differ only in this one decision):
>
> | ctx | replicated (min of 6) | column-parallel (min of 6) | delta |
> |---|--:|--:|--:|
> | 8 000 | 33.846 | **33.405** | **-0.441 ms (-1.30 %)** |
> | 16 000 | 34.668 | **34.238** | **-0.430 ms (-1.24 %)** |
> | 32 000 | 36.052 | **35.736** | **-0.316 ms (-0.88 %)** |
>
> Weights UNBOUND for these: the schedule, the packet count, the workgroup counts and every buffer
> SIZE are real — size is the whole of what this change touches — and dropping the ~100 s
> checkpoint load per run is what buys 6 reps. Within-arm spread is 0.02–0.90 ms and the shard is
> ahead in all 18 pairs.
>
> **The BOUND runs cannot resolve an effect this size, and it is worth writing down why.** Two
> back-to-back bound runs re-read 195 GiB per rank and the SECOND of each pair is penalised by
> 5–10 ms. Base-first said the shard was 6–9 ms SLOWER; shard-first said the opposite. The sign
> follows the ORDER, not the blob. A bound A/B needs order-balancing (or one load per process)
> before it can see a sub-1-ms change; first-in-pair runs land at 37.6–37.9 ms for both arms.

**Three consequences that should govern the whole campaign:**

* **~77 % of the stream (KDA + MLA + shared + latent + lm_head = 101 GB) runs on ops already at
  83–106 % of ceiling.** There is no kernel win available there and none should be sought.
* MXFP4 on the routed experts saves 19 % of the stream. It is *free* (already implemented) and it
  is **not** where K3's bandwidth is. Do not sell it as the lever.
* The state that grows with context is **27.0 KB/token of MLA latent** (24 layers × 576 × 2 B)
  plus a **fixed** 414 MB of KDA recurrent state per sequence (96×128×128 f32 × 69, TP-shardable).
  Long context is nearly free on this architecture — 1 M tokens of MLA latent is 27 GB across
  24 layers, and the 69 KDA layers cost **nothing** with context. That is the strategic reason to
  care about this model.

---

## 3. Capability table

Verdicts: **COVERED** = exists and is correct for K3's shape · **INSTANTIATE** = arm exists,
needs a parameter/dispatch/emitter change · **NEW** = no arm exists.
Effort: XS ≤ 1 h · S ≤ ½ day · M ≤ 3 days · L ≥ 1 week.

| # | capability | K3's shape | plow op / site | verdict | evidence | effort |
|---|---|---|---|---|---|--:|
| 1 | mxfp4 weight decode | e2m1, 1 E8M0 per 32 K, low nibble = even k | `fp4_to_bf16v8x4`, `e8m0_to_f32` (`amd_common.h:335-403`) | **COVERED** | §4 — bias 127 verified from the checkpoint | — |
| 2 | mxfp4 decode GEMV / GLU | `GemvMxfp4` 91, `GemvGluMxfp4` 92 | `op_gemm.h:1587`, `:2446` | **COVERED** | dev.rs:607-630; layout byte-exact | — |
| 3 | mxfp4 prefill GEMM, 5 tile rungs | `GemmMxfp4` 93 + `GemmMed/Small/Wide/C5Mxfp4` 96-99 | `op_gemm.h:664-776` | **COVERED** | the ≈0.4 % `kv_a_proj` disaster is already fixed on this branch | — |
| 4 | **mxfp4 MoE expert decode** | w1/w3 `[3072,3584]`, w2 `[3584,3072]` | `wave_dot_mxfp4` + `PLOW_MOE_ENC_MXFP4=2`, `op_moe.h:376-600` | **COVERED** | row strides `n*(H>>5)`, `I_moe>>1` match on-disk exactly | — |
| 5 | **mxfp4 MoE expert prefill (A4W4)** | same | `op_moe.h:1241-1560` | **COVERED** | both operands fp4 through `v_mfma_scale_f32_32x32x64_f8f6f4` | — |
| 6 | encoding as a runtime field | `i[6]` decode / `i[3]` prefill | `MoeEnc::code()`, `mla.rs:2704,2716,2744,2758` | **COVERED** | contract §3 satisfied | — |
| 7 | **top_k = 16** | 16 | `#define PLOW_MOE_MAX_TOPK 8u`, `op_moe.h:57` | **INSTANTIATE** | §5a — silently clamps to 8 AND leaves table slots 8-15 uninitialised | S |
| 8 | 896 experts, `[E][3]` table | 21504 B/table/layer | `op_moe.h:18-20`, `orch/moe.rs:84-96`, `exec/amd.rs:821` | **COVERED** | E inferred from declared bytes; no fixed array, no u32 overflow | — |
| 9 | E=896 prefill assert | 896 | `MOE_PF_MAX_EXPERTS: u32 = 512`, `mla.rs:1894` | **INSTANTIATE** | LDS need at 896 ≈ 7.2 KB of a 147464 B arena — the bound is conservative | XS |
| 10 | router: sigmoid + noaux_tc + renorm | flags 1\|2\|4 | `MoeRouterTopk` 56, `op_moe.h:245-327` | **COVERED** | bias on selection only (`:303`), unbiased gate (`:315`), renorm (`:316`) | — |
| 11 | grouped top-k (`n_group`/`topk_group`) | 1/1 → inert | `moe_group_mask`, `op_moe.h:194` | **COVERED** | implemented; K3 does not exercise it | — |
| 12 | router GEMV N=896 | `[896,7168]` | `Gemv` 10 on 256 CUs, `mla.rs:2594` | **COVERED** | wave fill improves 12.5 % (GLM N=256) → **50 %** (§5c) | — |
| 13 | co-resident split at tk=16 | 16 experts | `GLM_MOE_CORESIDENT`, `mla.rs:2522-2556` | **COVERED** | cores=1 → exact 16 CU × 16, 24 channels/wave with no remainder (§5d) | — |
| 14 | **latent MoE width 3584 ≠ hidden** | expert K/N = 3584 | graph pins expert width to `c.hidden`: `mla.rs:2701,2713,2014,2033,662,682` | **NEW (graph) / COVERED (kernels)** | §5e — kernels take H=3584 as a runtime arg unchanged | M |
| 15 | 2 shared experts | fused I=6144 | `imoe_l = imoe / tp`, `mla.rs:2507` — no `n_shared` | **INSTANTIATE** | `hf_config.rs:621` and `rewrite/kimi.rs:221` already do `sh_inter = n_shared * mi` | XS |
| 16 | **`situ` activation** | `β·tanh(g/β)·σ(g) · β_l·tanh(u/β_l)` | `(act==SILU) ? silu : gelu_tanh` at 8 sites | **NEW (arm)** | §6 — 2-value ternary, and the *up* branch is also transformed | S |
| 17 | **KDA linear attention** | 69 layers, state `[96,128,128]` | none on AMD; `Mamba2Scan` 90 is NVIDIA-only and unverified | **NEW** | §7 — no `PLOW_DOP_MAMBA2_SCAN` in `interp.hip` → silent `default:` | L |
| 18 | short depthwise conv (k=4) + SiLU | 3× per KDA layer, `[12288,1,4]` f32 | `mamba_conv_at` (`op_mamba.cuh:45`) — NVIDIA only | **NEW (AMD)** | design precedent exists, port needed | S (inside 17) |
| 19 | **residual-attention block** | softmax over ≤9 × 7168, ×2/layer | none | **NEW** | §1b(g) | M |
| 20 | MLA templates at K3 dims | DK 512, DR 64 | `d_flash_mla_decode<512,64,{2,4}>`, `d_o_uv_fold<512>`, `d_flash_merge<512>` | **COVERED** | §8 — DK/DR identical to GLM; `v_head` and `qk_nope` are runtime / absorbed | — |
| 21 | `d_mla_merge_fold` at V=128 | V=128, VT dispatch | `interp.hip:381-393` picks `VT=256` when `bh*8 > nblk` | **INSTANTIATE** | §8b — `v1-v0 = 128 ≠ VT=256` drops to the 7.7×-slower scalar fallback | XS |
| 22 | **MLA with NO RoPE** | `rotary_emb = None` | 2 unconditional `HeadNormRope` per layer, `mla.rs:1270,1299` | **INSTANTIATE (removal)** | §1b(d) — applying RoPE the model lacks = silently wrong logits | XS |
| 23 | MLA output gate | `σ(g_proj(x)) ⊙ attn_out` | `Gemv` + a new 2-operand elementwise | **INSTANTIATE** | `d_glu` is `act(g)*u`; sigmoid-mul is a 3rd `act` arm or a fold into `MlaMergeFold` | S |
| 24 | GF=8 at 96 heads | 96/8 = 12 exact | emitter defaults GF=8 above ctx 4096; AMD instantiates only {2,4} | **INSTANTIATE** | `interp.hip:426` silently routes GF=8 → GF=4 body | S |
| 25 | `model_type: "kimi_k3"` | — | `hf_config.rs:149` matches `"kimi"` only | **INSTANTIATE** | fails LOUD ("unknown architecture") — correct behaviour | S |
| 26 | mxfp4 detected from config | `quant_method: "compressed-tensors"`, `format: "mxfp4-pack-quantized"` | `parse_weight_dtype` matches `"mxfp4"\|"fp4"`, `hf_config.rs:183-189` | **INSTANTIATE** | falls through to **BF16** — silent, and it is a *precision* fallthrough | S |
| 27 | per-tensor quant from an `ignore` regex list | attention bf16, experts fp4 | one global `weight_dtype` per net | **INSTANTIATE** | GLM already ships mixed precision, but via `glm52_prep.py`, not from config | S |
| 28 | 247k per-expert tensors (896×3×92) | — | `resolve_expert_tables`, `orch/moe.rs:20-52` | **COVERED (verify)** | pointer tables are 43 KB/layer; the index/ingest scale is the open question | — |

---

## 4. MXFP4 — verified, not assumed

### 4a. Layout: byte-exact, no repack

| | K3 on disk (`layers.1.…experts.0`) | plow expects | match |
|---|---|---|---|
| gate/up packed | `w1.weight_packed U8 [3072, 1792]` | `[N, K/2]`, row stride K/2 bytes | ✅ 3584/2 = 1792 |
| gate/up scale | `w1.weight_scale U8 [3072, 112]` | `[N, K/32]` E8M0, row stride K/32 | ✅ 3584/32 = 112 |
| down packed | `w2.weight_packed U8 [3584, 1536]` | `wstr = I_moe >> 1` (`op_moe.h:594`) | ✅ 3072/2 = 1536 |
| down scale | `w2.weight_scale U8 [3584, 96]` | `n * (I_moe >> 5)` | ✅ 3072/32 = 96 |
| group size | 32, along K, symmetric, no zero-point | *"a lane's b128 load is 16 bytes = 32 fp4 = EXACTLY one MX block"* (`amd_common.h:343`) | ✅ |

The alignment property `amd_common.h:343-347` calls out — one 16-byte load consumes exactly one
scale byte, no cross-lane scale reshuffle — holds verbatim for K3.

### 4b. E8M0 bias 127 — CONFIRMED FROM THE CHECKPOINT

Contract §2 says the bias is 127 and that it has bitten twice. Decoded histogram of
`layers.1.block_sparse_moe.experts.{0,1,10}.w1.weight_scale` (344064 bytes each):

```
expert 0 : min 112  max 122  mean 120.82   top: 121 (82.8%), 120 (16.2%), 122 (0.6%)
expert 1 : min 114  max 122  mean 120.81   top: 121 (81.6%), 120 (17.2%), 122 (0.5%)
expert 10: min 112  max 122  mean 120.69   top: 121 (77.0%), 120 (19.5%), 119 (0.7%)
zero bytes: 0        0xFF (MX NaN): 0
```
Byte 121 under bias 127 is `2^-6 = 0.0156`, so `amax ≈ 6 × 0.0156 = 0.094` — exactly the right
magnitude for a bf16 MoE expert weight block. Under any other bias the scale is absurd
(bias 0 → `2^121`). **`e8m0_to_f32(b) = 2^(b-127)` is the right decode for this checkpoint.**

Two corollaries worth recording:
* **No byte is 0**, so the `2^-127`-flushes-to-zero trap (`amd_common.h:427-431`) is not latent in
  the data — but it remains latent in any code that *writes* a neutral scale. `PLOW_E8M0_ONE = 127`
  (`amd_common.h:458`) is the constant to use, and the A4W4 prefill bridge already does.
* The scale distribution (top code 7 present in ~1 % of nibbles) is consistent with plow's own
  quantizer convention `e8m0_for_amax = frexp(amax/6)` (`amd_common.h:487-492`) — the checkpoint
  maps amax onto e2m1's 6.0 top code, same as plow.

### 4c. Nibble order — plow's assumption matches, one cheap check remains

`dev.rs:614` states *"low nibble = even k"*, and plow's own packer at `op_moe.h:1317` is
```c
dst16[i] = (unsigned char)(quant_fp4(v[i*2] * inv) | (quant_fp4(v[i*2+1] * inv) << 4));
```
i.e. element `2i` low, `2i+1` high — which is `compressed_tensors.pack_fp4_to_uint8`'s convention.
The measured low/high nibble histograms are statistically identical
(low `{0:11421, 1:21938, …}` vs high `{0:11486, 1:21859, …}`), so **the data cannot distinguish
the two orders** — a nibble swap permutes elements within a byte and leaves every per-block
multiset unchanged. This is the one MXFP4 fact that is *inferred* rather than verified. It costs
one comparison at bringup: dequantize `layers.1.…experts.0.w1` with the HF reference and diff
against plow's decode. Do it once; if it is wrong, every mxfp4 number is garbage in a way that
looks like "the model is just bad".

### 4d. The gfx950 instruction — what is actually used where

| path | instruction | note |
|---|---|---|
| decode GEMV / GLU / MoE expert (w4a16) | `v_cvt_scalef32_pk_bf16_fp4` ×4 op_sel, then `fdot2` | `amd_common.h:368-381`. The MX scale folds into the cvt **exactly** (E8M0 is a power of two), so there is no epilogue multiply — unlike block-fp8 |
| prefill GEMM (w4a16) | dequant-on-load to bf16, then the bf16 MFMA | `op_gemm.h:409-415` |
| grouped MoE prefill (A4W4) | **`__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4`, cbsz=blgp=4** | `mfma_a4w4`, `amd_common.h:461-465` — this is the instruction the brief asked about, and it is already in use with *runtime* scale operands |

The trap recorded at `amd_common.h:432-441` — compile-time-constant scale args make the backend
silently select the **unscaled** `v_mfma_f32_32x32x64_f8f6f4` — applies to any new A4W4 arm.
`scripts/asm_expect_gfx950.json` is the guard.

**Verdict: MXFP4 needs no kernel work for K3.** The remaining mxfp4 items are host-side
(#26, #27) and one numeric check (§4c).

---

## 5. 896 experts, top-16

### 5a. `PLOW_MOE_MAX_TOPK = 8` — the one hard blocker, and it fails in three ways

`runtime/amd/op_moe.h:37-57` is explicit that this is the file's only hard bound. The clamp
(`op_moe.h:313-314`, and `:133-135` for op 40):
```c
float gate[PLOW_MOE_MAX_TOPK];
if (k > PLOW_MOE_MAX_TOPK) k = PLOW_MOE_MAX_TOPK; /* backstop, see the bound note */
```
At k=16 three separate things go wrong, and only the first is obvious:

1. **Truncation with a wrong denominator.** The rank-selection pass (`op_moe.h:302-311`) is
   k-agnostic and would happily rank 16, but the epilogue writes 8. With `moe_renormalize` the
   sum is then over 8 gates, so **every surviving gate is wrong**, not just the 8 missing ones.
2. **Uninitialised routing slots.** The emitter declares `tab` at `rows * tk * 8` bytes
   (`mla.rs:668`) through a plain `b.tensor` with no zero-init (`mla.rs:616`), but only `j<8` are
   written (`op_moe.h:317-323`). `moe_slot_expert` (`op_moe.h:330`) reads slots 8-15; if the stale
   bytes happen to be `< n_exp`, a **random expert runs with a random gate** and `d_moe_combine`
   (`op_moe.h:754-765`) sums it unconditionally. Non-deterministic, no error.
3. **LDS scratch aliasing.** `unsigned* wl = (unsigned*)(keys + n_exp)` (`op_moe.h:257`) and the
   group-mask scratch sits at `wl + PLOW_MOE_MAX_TOPK` (`op_moe.h:299-301`). At k=16 the rank pass
   writes `wl[0..15]` over `gk[]`. Benign today only because `moe_group_mask` has already
   consumed `gk`, and K3 does not run the group mask at all — but it is a landmine.

**The fix is mechanical**: bump the constant to 16 and move the group-mask base past the wider
`wl`. There is **no bitonic network and no `float top[8]` selection structure** — selection is an
all-pairs rank count (`rank += (keys[f] > myk)`) whose cost is `O(n_exp²/threads)` and completely
independent of k. Effort S, plus a zero-init or a `PLOW_EXPERT_UNUSED` fill of the tail slots so
failure mode 2 can never recur.

### 5b. E=896 is fine everywhere it matters

`[E][3]` u64 = **21504 B/table/layer** (vs GLM's 6144); with the scale table, 43008 B/layer,
≈4 MB over 92 layers. E is *inferred* from the declared table size — `exec/amd.rs:821-828`,
`let n_exp = (blob.tensors[*i_ewt].bytes / 24) as u32;` — so nothing host-side needs teaching.
Device indexing is `wtab[(size_t)eid*3 + j]` with `eid` promoted; max offset 2687. The packed
selection key gives the expert id 20 bits. LDS at n_exp=896: router ≈ 10.8 KB, align ≈ 7.2 KB
against a 147464 B arena (extrapolating the table at `op_moe.h:47-52`). EP divisibility: 896 =
2⁷·7, so EP 2/4/8 all divide (`exec/amd.rs:849-857` hard-errors otherwise).

The only E ceiling that trips is the **prefill** assert `MOE_PF_MAX_EXPERTS: u32 = 512`
(`mla.rs:1894`), and its own message says the bound is about LDS the arena has in abundance.
Raise it (XS). Note NVIDIA is genuinely capped — `#define PLOW_MOE_MAXE 256u` with a
`__shared__ unsigned cnt[PLOW_MOE_MAXE]` (`runtime/nvidia/op_moe.cuh:2097`) — so **K3 is a
gfx950-only model** until that is redone.

### 5c. Router GEMV — N=896 is *better* than GLM's N=256

`Gemv` is `GV_BLOCKED`: `gv_per = ceil(N/nblk)`, then `for (n = gv_n0 + wave; n < gv_n1; n += 8)`
(`op_gemm.h:1319-1332`). One wave owns one output column, so wave fill is `min(gv_per, 8)/8`.

| model | N | K | `gv_per` at 256 wgs | live waves / 2048 | fill |
|---|--:|--:|--:|--:|--:|
| GLM-5.2 | 256 | 6144 | 1 | 256 | 12.5 % |
| **Kimi-K3** | **896** | **7168** | **4** | **1024** | **50 %** |

GLM's measured router pair is 0.860 ms (GEMV, 274 GB/s) + 0.527 ms (1-CU top-k tail) = **1.39 ms/token**
(`perf-data/glm52-decode-attribution.md:222-224`). K3 moves 4.1× the bytes at 4× the fill, so the
GEMV should land at similar wall-clock per layer. **The term that degrades is the 1-CU top-k tail**:
the all-pairs rank is `n_exp²/512` LDS iterations per thread → 1568 at 896 vs 128 at 256, ≈**12×**.
At GLM's 0.527 ms baseline that is a real number and it is on 1 CU. Ranked as a perf item (§10 #11),
not a blocker. Note `GLM_ROUTER_OFF_SHARED` was already measured and rejected
(`mla.rs:2557-2567`: *"+0.12 ms, i.e. nothing or slightly worse … Do not re-propose"*).

### 5d. Co-resident split at tk=16 lands EXACTLY

`mla.rs:2522-2556`. At `cores=1` (the shipping default) `expert_cus = b.split(tk, sl)`:

| tk | cores=1 | cores=2 `shared_w` | cores=2 `routed_w` |
|---|---|---|---|
| 8 (GLM) | 32 CU × 8 | 32 | **28** ← the 1.31× straggler the sibling review priced at 0.63 ms/token |
| **16 (K3)** | **16 CU × 16, exact** | 16 | 15, ragged |

Nothing clamps and nothing breaks (`routed_w = 15 > 0`; 16·15+16 = 256). And the fill arithmetic
is *cleaner* than GLM's: at cores=1, `I_moe = 3072` over 16 CU × 8 waves = 128 waves = **24
channels/wave with no remainder**. GLM's pathology was 512 channels over 224×8 waves = 2.29
channels/wave. **Recommend cores=1 for K3** — which is already the default — and re-measure
cores=2 rather than inheriting GLM's setting.

Caveat inherited from GLM and unchanged: `MoeExpertDownFp8Blk`'s `nchunk` collapse. K3's down op
has K = `I_moe/tp` = 3072/4 = **768** at TP4, so with a 1024-element pass `nchunk = 1` and lanes
past 768 are dead — **the same 15.6 %-of-ceiling shape as GLM**, slightly better (768/1024 of the
wave live vs GLM's 512/1024). The two fixes the sibling identified (adaptive lane chunk width;
the `has2` two-rows-per-wave idiom from `gemv_rows_fp8_blk`) transfer verbatim and are worth more
here because K3 has 92 MoE layers, not 75.

### 5e. `routed_expert_hidden_size` — the kernels are fine, the GRAPH is not

The device kernels are fully shape-parametric: `d_moe_expert_glu_fp8_blk(..., I_moe, H, ...)`
strides `for (n = slice*PLOW_WAVES + wave; n < I_moe; n += wstride)` with no tile constant, and
the scale-row arithmetic needs only 128-divisibility (`KB = (H+127)>>7`) — K3's 3584/128 = 28,
3072/128 = 24, 7168/128 = 56 are all clean, and `(H>>5)` for the mxfp4 rows is clean at 32.
**Hand the kernels H = 3584 and they work.**

What does not work is the emitter graph, which pins expert-in and expert-out to `c.hidden`:
`d.i[2] = h` (`mla.rs:2701`, `:2741`), `d.i[1] = h` (`:2713`, `:2751`), prefill `:2014`/`:2033`;
the activation `xn2` is `rows*h*BF16` (`:662`); `part` is `rows*(tk*h)*F32` (`:682`); and
`d_moe_combine` adds the partials **straight onto the hidden residual** (`op_moe.h:754-765`).

K3 needs, per MoE layer:

```
xn2(7168) --Gemv[3584,7168]--> xe(3584)
        experts(896/top16, K=3584, I=3072, mxfp4) --> part[16][3584]
part --MoeCombine(H=3584, residual=NONE, shared=NONE)--> y(3584)
y --RmsNorm(3584)--> --Gemv[7168,3584]--> yh(7168)
out = residual + shared(identity) + yh
```

Every op in that chain exists. Three small edits are needed:
* a second width `he` in `GlmCfg` (there is no `routed_expert_hidden_size` field today,
  `mla.rs:20-64`), threaded to the expert ops and the `xn2`/`part` allocations;
* `d_moe_combine` dereferences `residual[h]` unconditionally — needs a null guard (1 line) or a
  zero buffer;
* the shared expert now reads the **pre-down-proj** hidden (`identity`), not `xe`, so the two
  branches diverge before the experts. GLM's emitter feeds both from `xn2`.

**Verdict: NEW graph shape, zero new kernels.** Effort M, and it is the single largest emitter
item after KDA.

Sizing note for prefill: `MPF_MAX_ROWS(T,k,n_exp) = T*k + n_exp*(MPF_BM-1)` (`op_moe.h:848`,
mirrored at `mla.rs:688`). At E=896, T=512, k=16 that is 8192 + 56448 = **64640 padded rows**, i.e.
87 % padding, and `moe_fug` becomes 64640 × 3072 × 2 B = **397 MB** of scratch. `MPF_BM = 64` is
the one number `op_moe.h:1271-1275` says to re-tune per model, and 896 experts make its case worse,
not better. Also `mpf_expert_of_tile` (`op_moe.h:986-992`) is a linear scan over `n_exp` per tile —
896 scalar iterations, 3.5× GLM. Both are prefill-only.

### 5f. 2 shared experts

`hf_config.rs:621` (`let sh_inter = synth.n_shared_experts * mi;`) and `rewrite/kimi.rs:221`
already do the right thing; only the shipping MLA/MoE emitter does not — `mla.rs:2507`
`let imoe_l = imoe / tp;`, with no `n_shared` anywhere. Since the checkpoint fuses the two shared
experts into one I=6144 MLP, and SwiGLU/SiTU are per-lane so summing two experts *is* one
2×-wide expert, this is `imoe * n_shared / tp` at three sites plus the tensor declarations.
Effort XS.

---

## 6. `situ` — SiTU-GLU

**What it is** (`modeling_kimi_linear.py:64-85`, registered as `ACT2FN["situ"]`):

```python
class SituAndMul(nn.Module):
    def forward(self, x):
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up   = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)
```
with `beta = 4.0`, `linear_beta = 25.0`. It is a **soft-clipped SiLU**: as β→∞,
`β·tanh(g/β) → g` and `situ_a → g·σ(g) = silu(g)`. β=4 clamps the gate branch at ±4; β_l=25
clamps the up branch at ±25. Used by **every** GLU in the model — dense L0, shared experts, and
routed experts (`hidden_act` is global; both `KimiMLP` and `KimiBlockSparseMLP` construct it).

**Is it expressible in the existing `act` enum? No — for two reasons, one of which is easy to miss.**

* The AMD selector is a two-value **ternary**, not a switch:
  `const float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);`
  at `op_elementwise.h:68`, `:75`; `op_gemm.h:579`, `:1042`, `:1479`, `:1864`, `:2531`; and
  `op_moe.h:79` (`if (act == 1) return silu; else gelu_tanh`). Adding a third value means
  editing 8 sites. Note `runtime/common/kernel.h:22` already has a 7-value enum
  (`PLOW_ACT_NONE/SILU/GELU/GELU_TANH/RELU/SIGMOID/QUICK_GELU`) that the *row* path
  (`row.hip:42-52`) switches on — but the megakernel GLU sites use the 2-value one at
  `op_elementwise.h:52`. Two enums exist; only one is on the fast path.
* **The epilogue shape changes.** Every existing GLU is `act(g) * u`. SiTU is `A(g) * B(u)` —
  the *up* branch is transformed too. A new `act` code is not enough; the epilogue expression has
  to grow a second function whose identity case reproduces today's behaviour byte-for-byte.

**Cost: cheap, and effectively free where it matters.** `tanh(x) = 2σ(2x) − 1`, so
`β·tanh(g/β)·σ(g)` is 2 exponentials + 2 reciprocals + ~4 FMAs, vs SiLU's 1 exp + 1 rcp; the up
branch adds 1 exp + 1 rcp. So ~3× the transcendental work of SwiGLU. Against the measured
context: `MoeExpertGluFp8Blk` runs at **32–42 % of the memory ceiling with one load in flight**
(`glm52-kernel-review.md` §3 row 4, §2a), i.e. the VALU is idle waiting on HBM. Three
transcendentals per output element hide completely under that. Say so plainly: **a fused GLU
activation is free inside a memory-bound kernel** — and every K3 GLU site is memory-bound.

The one place to check rather than assume is the **prefill** GEMM GLU epilogue (`op_gemm.h:579`,
`:1042`), which is MFMA-bound and where the prefill bucket has **zero register headroom** —
measured 256 VGPR / occ 2 / **2 VGPR spill** on this branch (§11). Two extra live f32 constants
and a deeper expression could push it. That is a `check()` run, not a redesign.

**Plumbing**: β and β_l are per-model constants, so either two `#define`s in a K3 build variant
or two `f[]` slots. `Instruction` has `f: [f32; 2]` plus `j: [u32; 2]`, and `dev_isa.h:683`
records the `fj[]` slot map was already compacted once because `f[2]`/`f[3]` were dead — so slot
availability per opcode must be checked, not assumed. Effort S.

---

## 7. KDA — the kernel-feasibility view (the sibling owns semantics)

### 7a. What exists on AMD today: nothing

`DevOp::Mamba2Scan = 90` / `PLOW_DOP_MAMBA2_SCAN = 90` is declared in `dev_isa.h:598`, emitted by
`mla.rs:3981`, and reported by the manifest (`manifest.rs:351`). Its **only** implementation is
`runtime/nvidia/op_mamba.cuh`, whose own header says:

> `============================== UNVERIFIED ON GPU ==============================`
> *"it has ONLY been verified to nvcc-COMPILE … this kernel has never executed on a GPU."*

`grep MAMBA runtime/amd/interp.hip` returns nothing. So on gfx950 op 90 falls to
`interp.hip:1253`'s `default: break;` — the silent no-op that `interp.hip:895` documents as the
reason the lm_head once produced all-zero logits. **plow has no state-carrying op on AMD.**

That is not all bad news: `op_mamba.cuh` is a usable *design* precedent (single-CU, conv1d
computed on demand with no materialized buffer, conv_state/ssm_state as in/out tensor handles
`t6`/`t7`), and the op-90 ABI already establishes how a persistent state buffer is carried in the
packet. Reuse the shape, not the code.

### 7b. The memory profile is fundamentally different from flash-decode, and better

| | MLA flash-decode | KDA decode step |
|---|---|---|
| per-token state | grows: 576 B × ctx × 24 layers | **fixed**: `[96, 128, 128]` f32 = **6.00 MB/layer**, ×69 = **414 MB/sequence** |
| work per token | O(ctx) | O(1) |
| conv state | n/a | `3 × 12288 × 3` f32 = **432 KB/layer** |
| state traffic/token/rank (TP4) | — | 2 × 6.00/4 MB × 69 = **207 MB → 33 µs @ 6200 GB/s** |
| arithmetic/token | O(ctx · d) | 96 heads × ~82 kFLOP × 69 = **0.54 GFLOP** |

**The state traffic and the arithmetic are both negligible.** The 69 KDA layers cost 57 GB/token
of *projection weights* (§2) and ~33 µs of state. A KDA decode kernel that merely reaches the
memory ceiling on its state is not the problem; **fitting it into the interpreter is.**

### 7c. The register/LDS verdict — measure it now, as instructed

Compile-only, this branch, `gfx950` (§11):

```
decode : VGPR 248  AGPR 0  occ 2  VGPR-spill 0  SGPR-spill 80  scratch 136 B/lane  LDS 147464 B
prefill: VGPR 256  AGPR 0  occ 2  VGPR-spill 2  SGPR-spill 85  scratch 1388 B/lane LDS 147464 B
```

**Decode has 8 VGPRs of headroom to the 256 cliff. Prefill has ZERO and is already spilling.**

Now price the state. `S` is `[head_k_dim 128 × head_dim 128]` f32 = **64 KB per head** — exactly
one wave's entire register file (256 VGPR × 64 lanes × 4 B). Three placements:

| placement | per-lane cost | verdict |
|---|---|---|
| **registers**, one head per workgroup, 512 threads = 128 v-cols × 4 threads, each thread holding 32 k-rows | **32 VGPRs live across the whole op** | **Does not fit.** 32 ≫ 8 available. Forces the megakernel to 280+ VGPR → occ 1, or spills. |
| **LDS**, one head per workgroup | 64 KB of the 147464 B arena | Fits *numerically* — the arena is already reserved and `glm52-kernel-review.md` §2b shows 118 KB of it is dead prefill-GEMM space in the decode bucket. But it locks the 2-WG/CU experiment permanently (that needs LDS ≤ 80 KB). |
| **HBM**, streamed per step | ~0 | 207 MB/token/rank = 33 µs. **Cheapest by far**, and it keeps the op stateless from the register allocator's point of view. |

**Recommendation: do not put KDA in the decode megakernel.** Put it in a **4th co-resident code
object**, the way `PLOW_BUCKET_FLASH` already is (4 waves / 512 reg / occ 1). Contract §0-EXT
establishes that a 4th co-loaded object is native to the design and needs zero backend changes —
`run_segmented` issues plain AQL, `module_load` makes a per-module executable. A KDA bucket can
then spend 200 VGPRs on state without touching the 248/occ-2 decode object at all.

**The cost of that choice, stated honestly**: decode's zero-per-op-dispatch property (1 dispatch
per token) is a genuine asset that contract §0-EXT warns against spending. A separate KDA object
means **69 extra segment boundaries per token**, each a full AQL drain. That is exactly the gate
§0-EXT says to A/B before adopting `GemmExt`, and `amd.rs`'s `seg_enq_us`/`seg_drain_us` already
instrument it. **Measure the 69-segment cost before committing** — it may be that LDS placement
inside the decode object wins despite locking the occupancy experiment.

### 7d. Slice map — the failure mode to design against from the start

Every kernel the sibling review found slow failed for the same reason: *achieved % of ceiling ≈
active-wave fraction*. The naive KDA map is one workgroup per head: at TP4, `nh_l = 24` of 256
CUs = **9.4 % occupancy**, which is the `MlaMergeFold` disaster (16 of 256 wgs, 2.9 % of ceiling)
reproduced exactly. Split the v-dimension: `head × v-chunk` with 32 v-columns per chunk gives
`24 × 4 = 96` workgroups; 16 columns gives 192. The state update is an outer product and
partitions cleanly along v with no cross-workgroup communication. **Design the slice map first,
the inner loop second** — that is the one transferable lesson from the GLM audit.

### 7e. Also new, also inside KDA's scope

* **3 depthwise causal convs, k=4, + SiLU** per KDA layer, with a `[d, 3]` conv state. No AMD op;
  `mamba_conv_at` (`op_mamba.cuh:45-60`) is the port target. Note K3 stores `q/k/v_conv1d.weight`
  as **F32** `[12288,1,4]`, not bf16.
* **`FusedRMSNormGated`** — `o_norm(o, g)` with `activation='sigmoid'`, per-head-dim (128) with an
  f32 weight. plow's `d_glu` is `act(g)*u`; this is `rmsnorm(o) * σ(g)`, a different composition.
  Shares a new arm with the MLA output gate (#23).
* **`use_qk_l2norm_in_kernel`, `use_beta_sigmoid_in_kernel`, `safe_gate` with
  `lower_bound = -5.0`** — all fused into the reference kernel, all cheap ALU, all must be
  reproduced or the numerics drift. Sibling's call.

---

## 8. The 24 MLA layers — covered, with two dispatch snags

### 8a. Every fixed template dimension is unchanged from GLM

| template | params | instantiated today | K3 needs |
|---|---|---|---|
| `d_flash_mla_decode` (`op_attention.h:1304`) | `<DK, DR, GF, GATHER>` | `<512,64,2/4,false>`, `<512,64,2/4,true>` | `<512,64,·>` ✅ |
| `d_flash_mla_prefill` (`:1552`) | `<DK, DR, GF>` | `<512,64,2>`, `<512,64,4>` behind `PLOW_MLA_PREFILL` | ✅ |
| `d_mla_merge_fold` (`:2359`) | `<DK, VT, VEC, UNW>` | `<512, 32, 4, 4>`, `<512, 256, 4, 4>` | see §8b |
| `d_o_uv_fold` (`:2250`) | `<DK>` | `<512>` | ✅ |
| `d_flash_merge` (`:1189`) | `<D>` | `<128>`, `<256>`, `<512>` | ✅ |
| `d_headnorm_rope` (`op_norm.h`) | `<HD, INTERLEAVE>` | `<128,true/false>`, `<256>`, `<512>`, `<64,true>` | **not wanted** (§8c) |

**`v_head_dim` is never a template parameter** — `d_mla_merge_fold` and `d_o_uv_fold` take
`unsigned V` at runtime. **`qk_nope` never reaches a kernel at all** — it is folded into
`q_absorb` on the host, and `glm52_prep.py:200-208` reads the split point `QN` from config. The
emitter is fully config-derived: `wqa` is `(nh_l*dk, ql)`, `wuv` is `nh_l*dk*vd`
(`mla.rs:869-897`), `attn_scale = (qk_nope+qk_rope)^-0.5` (`mla.rs:148`) → K3 gets 192^-0.5
automatically. A grep of `mla.rs` for hardcoded 512/64/192/256/2048/6144 on the emit path returns
only doc comments and `#[cfg(test)]` fixtures.

LDS is `FA_MLA_DEC_LDS_FLOATS(DK,DR,GF) = GF·800 + 4112` floats at DK=512/DR=64 → **29248 B at
GF=4**, identical to GLM, ~118 KB under the arena the object already reserves. `v_head`,
`qk_nope`, `heads`, and `hidden` do not appear in the formula.

Head count: `assert!(c.heads % tp == 0)` (`mla.rs:3642`) — 96 divides by 4 and 8. `nh_l % GF` is
**not** checked and a remainder silently corrupts (`n_grp = n_head/GF` truncates, and
`d_mla_merge_fold` then reads uninitialised `Opart` for the tail heads) — but K3 is exact at every
planned TP: TP1 96, TP4 24, TP8 12, all divisible by 2 and 4. **It breaks at TP16** (nh_l=6, GF=4).
Worth an assert.

Also confirmed: contract §4's warning is literally true here — `interp.hip:1253` is
`default: /* PLOW_DOP_NOP */ break;`, and `HEADNORM_ROPE` (`:130-172`) and `FLASH_MERGE`
(`:1050-1060`, `:1241-1251`) are `else if` chains with **no final `else`**, so an unmatched
dimension writes nothing. The host-side ELF-symbol guard (`exec/amd.rs:352`) catches the
`PLOW_MLA_PREFILL`-absent case before launch; it does **not** catch an unmatched `i[2]`.

### 8b. `d_mla_merge_fold` silently drops to the slow body at V=128 — **FIXED 2026-07-28 (rung 3)**

> `exec_mla_merge_fold` (`interp.hip`) now has a third arm: `V in [128, 256)` dispatches
> `d_mla_merge_fold<512, 128>`, so at K3's `v_head_dim = 128` the fast body's `v1 - v0 == VT`
> guard is satisfied (`vtiles = 1`, `v1 - v0 = 128 = VT`) and the op no longer falls through to
> the scalar fallback. **GLM's arm is untouched**: `V >= 256` still takes the literal `<512,256>`,
> same template, same workgroup count, same thread map, so GLM-5.2's fold is bit-identical and
> could not have moved. `V < 128` keeps the old behaviour rather than growing a fourth
> instantiation for a shape no model in this tree has.
>
> **Register cost measured, and it is ZERO** (`scripts/k3_rung3_regcheck.sh`, A/B against the same
> tree with the runtime changes stashed): decode 248 VGPR / occ 2 / 0 vspill, prefill 256 / occ 2 /
> 2 vspill, prefill_mla_moe 256 / occ 2 / 2 vspill, flash 512 / occ 1 / 228 vspill, LDS 147464
> everywhere — every number identical before and after. A new instantiation of an existing template
> was the thing to watch, because the megakernel's allocation is the worst case over every inlined
> arm and prefill already sits AT the cliff. It did not move.
>
> Correctness is covered by `runtime/tests/k3_mla_block_gfx950_test.c`, which runs at exactly the
> broken shape (`bh = 1*96 = 96`, `nblk = 256`, so `96*8 = 768 > 256` takes the `else`) and scores
> `MLA_MERGE_FOLD`'s output against the model's own materialized `k_pass`/`value`.
> **No timing number was taken** — this rung is a correctness gate and the 7.7x figure below is
> inherited from the GLM measurement, not re-measured at V=128.

The original finding, kept for the reasoning:


`interp.hip:381-393`:
```c
const unsigned bh = in->i[0] * in->i[1];               /* n_batch * n_head_local */
if (bh && bh * 8u <= nblk) { d_mla_merge_fold<512, PLOW_MLA_FOLD_VT>(...); }
else                       { d_mla_merge_fold<512, 256>(...); }
```
The fast path inside requires `v1 - v0 == VT` (`op_attention.h:2432`). With `VT=256` and K3's
`V=128`, `vtiles = ceil(128/256) = 1` so `v1-v0 = 128 ≠ 256` → the op takes the **scalar
fallback** (`op_attention.h:2470-2476`), the body the 2026-07-28 rewrite replaced because it was
7.7× slower. GLM's `V=256` never hits this. K3 hits it at TP1 (bh=96, 768 > 256) and at batch ≥ 2
on TP4 (48·8 = 384 > 256). Fix is `VT = min(256, V)` or a `VT=128` arm — a **dispatch tweak, not
a kernel**. Effort XS, and it should be done before anyone benchmarks K3 batched decode and
concludes the merge is slow.

### 8c. RoPE must be REMOVED — **WRONG. It must be NEUTRALIZED. Corrected 2026-07-28 (rung 3)**

> **This section's prescription — "skip both `HeadNormRope` emits", "It is a removal", "effort XS" —
> is the bug, not the fix.** `plans/kimi-k3-frontend.md` §7 first flagged it; rung 3 implemented and
> gated the correct form.
>
> The k-side `HeadNormRope` (`d.t[0] = n.krot[slot]`) is
>  (a) the **only writer of the `kv.{l}.krot` cache row**, which `FlashMlaDecode` keeps reading at
>      `i[5]` whether or not anything wrote it, and
>  (b) the instruction that `plowrt::exec::amd::kv_write_row_field` (`amd.rs:1017`) and
>      `runtime/tests/glm52_decode.c:419` both **SCAN FOR** in order to patch that row's position
>      each step. In `glm52_decode.c` the `ckv_ins`/`krot_ins` arrays are index-paired and `nlk` is
>      incremented **only on the krot match** — so losing the krot writer silently loses the ckv
>      patch too, and every token's KV lands at row 0. There is no count check anywhere.
>
> **Keep the WRITE, remove the ROTATION.** An identity `cos = 1` / `sin = 0` table makes the op a
> bit-exact bf16 copy: `gamma` is already `TENSOR_NONE` and `skip_norm` is already 1 on this emit,
> so with those two constants `HeadNormRope` reduces to `f2bf(bf2f(x))`. No new generator kind was
> needed — `packet::rope::rope_tables` already emits `(1.0, 0.0)` past `rope_angles`, so
> `frac = 0.0` puts every angle past it (`devgen::k3::k3_nope_rope_pair`, with a test asserting the
> table is EXACTLY those two f32 constants at every position, and asserting it differs from GLM's).
>
> So it is **not** a removal and **not** −2 packets/layer: the packet count is unchanged and the
> saving is the rotation's arithmetic, which was never the cost.
>
> Gated by `runtime/tests/k3_mla_block_gfx950_test.c`, which checks the two copies **BITWISE**
> rather than to a tolerance — a table whose angles were merely small would pass a 1.5e-2 residual
> while quietly rotating — and which also asserts the k-side write landed at row `qpos` and left
> rows `0..qpos-1` untouched.
>
> **A note on how easy it is to write a control that proves nothing here.** rung 3's first NoPE
> control rotated q and every cached k at the SAME position and measured the attention output
> moving by **1.2e-7**, i.e. "RoPE is harmless" — because a common rotation is orthogonal and
> preserves every dot product exactly. RoPE is a RELATIVE encoding; the damage only appears when
> key `t` is rotated by `t` and the query by `qpos`. Corrected, the same control reads **2.5e-1**.

The original finding, kept for the reasoning:


`mla.rs:1270` and `:1299` emit `DevOp::HeadNormRope` unconditionally for `q_rope` and `k_rope`
(also `:1724`, `:1751` on the second path). K3 applies **no rotation** (§1b(d)). Running K3 through
`cfg_kimi` (`mla.rs:174-178`, which is `cfg_glm` + `has_dsa = false`) would rotate q/k that the
model never rotated — output that is *plausible-looking and wrong*, degrading with position, and
invisible to any shape or symbol check. This is contract §4's bug shape with the polarity flipped:
not *an arm exists and nothing routes to it* but *an arm is routed to and should not be*.

The fix is a `use_nope` flag in `GlmCfg` that skips both emits. It is a **removal** — 2 fewer
packets per MLA layer, 48 fewer per token. Effort XS. Put an assert on it, because nothing
downstream can detect its absence.

### 8d. GF=8 is emitted and not instantiated

`glm_gf(ctx)` returns 8 above `GLM_GF_CROSSOVER = 4096` (`mla.rs:191-204`), the packet carries
`i[7]=8`, and `interp.hip:426`'s `if (gf == 2) … else …` silently runs the **GF=4** body. This is
a pre-existing GLM condition and it is benign for correctness (GF is an internal fusion factor),
but two comments have drifted from the code (`interp.hip:404` "Both variants are instantiated";
`mla.rs:186` "PLOW_GLM_GF pins GF∈{2,4}"). K3's 96 heads divide by 8 exactly at TP1/TP4/TP8, so
if the GF=8 win is wanted, add `<512,64,8>` and re-run `check decode` against the 256/occ-2 cliff
— `op_attention.h:1262` records GF=8 at 170 VGPR standalone, and NVIDIA already ships it
(`interp_sm120.cu:1245-1261`).

---

## 9. WHAT IS ALREADY GOOD — do not spend time here

| area | the evidence |
|---|---|
| **The whole MXFP4 stack** | Element decode, GEMV, GLU, 5 GEMM tile rungs, MoE expert decode (`enc=2`), and A4W4 grouped expert prefill all exist. The on-disk layout is **byte-exact** — packed `[N,K/2]`, scale `[N,K/32]`, group 32 along K — and **E8M0 bias 127 is confirmed against the actual checkpoint bytes** (§4b). The scale folds into `v_cvt_scalef32_pk_bf16_fp4` exactly, so there is no dequant in any epilogue. **Zero kernel work.** |
| **`Gemv` / `GemvGlu` / `GemvQkv` bf16** | 83–106 % of the 6200 GB/s ceiling, 16–18 loads in flight (`glm52-kernel-review.md` §6). These carry **77 % of K3's decode weight stream** (§2). Nothing to win, and the `GemvQkv` fusion idiom is exactly right for K3's KDA layer, which has four `[12288,7168]` projections that should be fused the same way. |
| **`MoeRouterTopk`'s routing semantics** | sigmoid (`op_moe.h:277`), noaux_tc bias applied to the **selection key only** with the winner's **unbiased** score returned (`:303`, `:315`), renormalize (`:316`), group-limited routing (`:194-232`). K3's flag word is the same `1\|2\|4` GLM already uses. The router is *already* DeepSeek-V3-shaped and K3 is DeepSeek-V3-shaped. |
| **The `[E][3]` expert table at E=896** | 21504 B/layer, `size_t`-promoted indexing, expert id has 20 bits in the packed key, E inferred from the declared table bytes (`exec/amd.rs:821`). No fixed array, no overflow, no host change. The `op_moe.h:37-57` note is explicit that n_exp was never the bound. |
| **Co-resident MoE at tk=16** | An exact 16 CU × 16 partition with 24 channels/wave and no remainder — *better* than GLM's 28-CU 1.31× straggler. The default (`cores=1`) is already the right setting. |
| **MLA templates** | `DK=512`, `DR=64` are identical to GLM; `v_head` and `qk_nope` never reach a template. LDS is bit-identical at 29248 B. The absorb derivation in `mla.rs`/`glm52_prep.py` is fully config-parameterised — grep found no hardcoded MLA dim on the emit path. |
| **Long context** | 69 of 93 layers carry **fixed-size** state. Only 24 layers grow with context, at 27.0 KB/token total. Whatever plow's long-context weaknesses are, this architecture does not exercise them. |

---

## 10. RANKED — what to build, ordered by what unblocks a running model

Ranked by *distance to a correct token*, then by cost. Items 1–8 are correctness; 9+ are perf.

| # | item | verdict | effort | why here |
|--:|---|---|--:|---|
| **1** | **`PLOW_MOE_MAX_TOPK` 8 → 16**, move the group-mask LDS base past the wider `wl`, and fill routing slots with `PLOW_EXPERT_UNUSED` | INSTANTIATE | S | Cheapest blocker in the list. Without it every MoE layer runs 8 of 16 experts with a wrong renormalisation denominator **and** sums 8 slots of uninitialised memory. Silent. (`op_moe.h:57`, `:299-311`, `:313-323`) |
| **2** | ~~**MLA `use_nope`** — skip both `HeadNormRope` emits~~ **KEEP both emits, feed an identity cos=1/sin=0 table** | **CLOSED on hardware (rung 3)** | S | Applying RoPE the model does not have is silently wrong and undetectable downstream — but so is deleting the op, which is the ONLY writer of the `kv.{l}.krot` row AND the instruction the KV-row-writer scan matches on. **NOT a removal and NOT −48 packets/token**: see §8c. `devgen::k3::k3_nope_rope_pair`; gated bitwise by `k3_mla_block_gfx950_test.c`. |
| **3** | **`situ` activation arm** — 3rd `act` code at 8 sites, plus the up-branch transform | NEW arm | S | Every GLU in the model. Free ALU inside memory-bound kernels; re-run `check prefill` because prefill is at 256 VGPR with 2 spills already. (§6) |
| **4** | **`n_shared_experts` → `sh_inter = 2 × moe_inter`** | INSTANTIATE | XS | One-line width. `hf_config.rs:621` is the template. Without it the shared expert is half-width and wrong. |
| **5** | **Latent-MoE graph**: `he = 3584` second width, `MoeCombine` with a null residual, norm + up-proj, shared expert fed from the pre-down hidden | NEW graph, no new kernel | M | Structural; nothing today expresses two widths in one MoE block. The kernels take H=3584 unchanged. (§5e) |
| **6** | **MLA output gate**: `g_proj` GEMV + `σ(g)⊙attn_out` before `o_proj` | **CLOSED on hardware (rung 3)** | S | 24 layers. `PLOW_DOP_MLA_OUT_GATE = 106` (`op_k3.h`), a standalone streaming op — **NOT** folded into `MlaMergeFold` as this row originally proposed. That fold is nearly free in isolation but `MlaMergeFold` is GLM-5.2's op on GLM's critical path, so the fold would cost GLM either a branch and an operand slot or a second template instantiation in a decode object with 8 VGPRs of headroom. Separate also keeps the fold and the gate independently diffable. `g_proj` reads the **MLA sub-layer input** (post-`input_layernorm`), not the attention output, and it is `sigmoid`, not `silu`. Register cost zero. |
| **7** | **Residual-attention block** (`_apply_attn_res`): fused norm + fixed-query score + softmax(≤9) + weighted mix | NEW op | M | 186 invocations/token, replaces the plain residual add. Bandwidth-trivial (~48 MB/token) but it must be **one** packet, not three — at ~5.9 µs of gate per narrow packet, three packets × 186 is 3.3 ms/token of pure protocol. Fold `norm.weight * proj.weight` into one `[7168]` f32 at load. |
| **8** | **KDA decode kernel** + 3 depthwise convs + `FusedRMSNormGated` | NEW | **L** | 69 of 93 layers; nothing runs without it. **Build it as a 4th co-resident object, not an arm** (decode has 8 VGPRs of headroom and the state needs 32/lane). Design the slice map first — one-workgroup-per-head is 9.4 % CU occupancy, the `MlaMergeFold` disaster. **A/B the 69 extra segment drains before committing** (`seg_enq_us`/`seg_drain_us`). (§7) |
| 9 | **Ingest**: `model_type: "kimi_k3"`, `quant_method: "compressed-tensors"` + `format: "mxfp4-pack-quantized"`, and per-tensor quant from the `ignore` regex list | INSTANTIATE | S | #25/#26/#27. The arch mismatch fails **loud**; the quant mismatch falls through to **BF16 silently** — fix that one first. Also verify the nibble order once against the HF dequant (§4c). |
| 10 | `MOE_PF_MAX_EXPERTS` 512 → 896 | INSTANTIATE | XS | Prefill only. `mla.rs:1894`. The bound is conservative by ~20×. |
| 11 | `d_mla_merge_fold` VT dispatch at V=128 (a `<512,128>` arm) | **DONE (rung 3), register cost zero** | XS | Perf. Prevented a 7.7× regression at TP1 or batch ≥ 2 that would have looked like a kernel problem. GLM's `V >= 256` arm is byte-identical. **Not re-timed at V=128** — the 7.7× is GLM's measurement. (§8b) |
| 12 | `MoeRouterTopk` 1-CU all-pairs rank at n_exp=896 | perf | M | `O(n_exp²/512)` → ~12× GLM's 0.527 ms. On 1 CU. The fix is a two-stage partial rank, not a wider clamp. |
| 13 | `MoeExpertDownFp8Blk` lane-width / `has2` fix at K=768 | perf | M | Same shape as GLM's #2, 92 layers instead of 75. The sibling's two fixes transfer verbatim; both are local to the existing kernel. |
| 14 | **KDA prefill (chunked)** | NEW | L | Only after decode works. Until then TTFT is prefill-as-N-decodes, which is GLM's 37.9 s-vs-1.9 s deficit (§7 of the sibling review) reproduced on 69 layers. |
| 15 | `MPF_BM` re-tune + `moe_fug` 397 MB scratch at E=896/T=512/k=16 | perf/sizing | M | Prefill only. 87 % padding waste. `op_moe.h:1271-1275` already names `MPF_BM` as the per-model knob. |
| 16 | `<512,64,8>` GF=8 instantiation, or fix `glm_gf`'s default | INSTANTIATE | S | 96 heads divide by 8 exactly. Today `i[7]=8` silently runs GF=4. Re-run `check decode`. |
| 17 | NVIDIA `PLOW_MOE_MAXE 256` | — | L | K3 is **gfx950-only** until this is redone. Record, do not schedule. |

**Explicitly NOT on this list, with reasons:**
mxfp4 kernels of any kind (§4 — done); `Gemv`/`GemvGlu`/`GemvQkv` (§9 — at the ceiling, and they
carry 77 % of the stream); the MLA flash-decode/merge/fold templates (§8a — DK/DR unchanged);
router semantics (§9 — already DeepSeek-V3-shaped); `RmsNorm`/`Residual`/`HeadNormRope` bodies
(sibling review §6 — 0.3 % of roofline and *correct as they are*; the cost is the gate, not the
kernel).

---

## 11. Method, and what would falsify this

* **Checkpoint evidence** is first-hand: every safetensors header of the 41 downloaded shards was
  parsed for names/dtypes/shapes; `weight_scale` bytes were decoded and histogrammed (§4b);
  per-layer tensor signatures were diffed to establish the MLA/KDA/dense layout. The remaining 55
  shards are more MoE layers of the same shape — nothing in this document depends on them.
* **Semantics** come from `modeling_kimi_linear.py` fetched from the model repo, quoted verbatim
  rather than summarised, because the two facts most likely to be got wrong (`rotary_emb = None`,
  and SiTU transforming *both* branches) are exactly the kind that a paraphrase loses.
* **The register numbers are measured on this branch**, compile-only, outside `nix develop`
  (contract §0a):
  ```
  hipcc --offload-arch=gfx950 -O3 -w -DPLOW_BUCKET_DECODE=1 -DPLOW_GEMV_MM=1 --genco \
        -Rpass-analysis=kernel-resource-usage runtime/amd/interp.hip -o /dev/null \
        -Iruntime/amd -Iruntime/common
  ```
  → decode 248/occ 2/spill 0/LDS 147464; prefill 256/occ 2/**spill 2**. No GPU was used and none
  was leased.
* **Known limitation.** No K3 weights have been dequantized and no K3 packet has been run. Every
  "covered" verdict is a *layout and dispatch* claim — that the bytes on disk match what the
  kernel indexes, and that a template instantiation exists — not a numerical one. The single
  numerical assumption is the fp4 nibble order (§4c), which the data provably cannot settle.
* **What would falsify the ranking**: if the fp4 nibble order is reversed relative to plow, item 9
  jumps to #1 and every mxfp4 "covered" verdict becomes "covered after a repack". If the KDA state
  turns out to be storable in bf16 rather than f32, §7c's register table shifts but its conclusion
  (separate object, not an arm) does not. If `A_log[128]` (§1b(h)) means the decay is indexed
  per-head-dim rather than per-head, the KDA inner loop changes and the sibling's spec is the
  authority, not this document.
* **What would falsify §2** — the claim that attention, not experts, dominates K3's decode
  bandwidth — is a `PLOW_TRACE_RAW` per-op attribution on a running K3 showing the routed-expert
  packets above ~20 % of the token. The arithmetic is from on-disk shapes and is hard to get
  wrong, but it assumes plow keeps the attention path in bf16 as the checkpoint ships it. Quantizing
  the attention projections that K3's `ignore` list exempts is a **separate, unproven lever worth
  53 % of the stream** — larger than every kernel item in this document, and out of scope for it.
