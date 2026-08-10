# Kimi-K3 — KDA (Kimi Delta Attention) layer spec

Target: `moonshotai/Kimi-K3`. **69 of 93 text layers are KDA**, 24 are gated MLA.
This document defines KDA exactly and maps it onto plow. It is a spec, not an implementation.

Every claim below is tagged with its source. `[inferred]` marks anything not directly read from
code, config, or checkpoint bytes.

## 0. Provenance

| tag | source |
|---|---|
| `[cfg]` | `config.json` in the local snapshot (`~/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/config.json`) — present and complete |
| `[cfgpy]` | `configuration_kimi_k3.py` in the same snapshot |
| `[hf]` | `modeling_kimi_linear.py`, fetched from `https://huggingface.co/moonshotai/Kimi-K3/resolve/main/modeling_kimi_linear.py` (1314 lines). **It was missing from the local snapshot but exists upstream** — the snapshot is merely incomplete. Cached at `/tmp/k3ref/modeling_kimi_linear.py` |
| `[fla]` | `flash-linear-attention` @ `main`, files `fla/ops/kda/{naive,gate,chunk,fused_recurrent}.py`, `fla/modules/conv/short_conv.py`. `[hf]` imports these; they hold the actual math |
| `[ckpt]` | safetensors **headers and small tensors** read directly out of the partially-downloaded shards. Headers only + a few KiB of fp32 — no bulk load |
| `[vllm]` | `rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`, `.../layers/fla/ops/kda.py`, `.../layers/mamba/mamba_utils.py` |
| `[card]` | `README.md` model card in the snapshot |
| `[paper]` | *Kimi Linear: An Expressive, Efficient Attention Architecture*, Kimi Team, **arXiv 2510.26692** (v2, 28 pp). Describes KDA for the earlier Kimi-Linear model. **Paper-derived, not code-derived** — flagged at every use |
| `[k3impl]` | The two public K3 KDA implementations: **vLLM PR #50089** (`k3-kernel-model-files` @ `a2e10fbc`, open, adds `vllm/models/kimi_k3/{nvidia,amd,common}/`) and **SGLang branch `kimi-k3`** (`python/sglang/srt/models/kimi_k3.py`). Both handle the K3-specific knobs that `[vllm]` 0.23.0 does not |

**Note on `[vllm]`.** That vLLM build — and vLLM `main` — implement *Kimi-Linear* (the earlier 48B
model), not K3. Both diverge from K3 on exactly the two knobs this task asked about, see §3.4. Use
`[vllm]` for structure (state layout, paging, TP sharding) and `[k3impl]` for K3 semantics.

**No Kimi-K3 tech report exists** as of writing; the model card cites no arXiv ID, and AttnRes does
not appear in `[paper]`. Where `[paper]` and the K3 checkpoint disagree, the checkpoint wins.

---

## 1. Layer inventory, and the off-by-one that will bite the loader

`linear_attn_config.kda_layers` and `full_attn_layers` are **1-based**. `[cfgpy]`:

```python
def is_kda_layer(self, layer_idx: int):
    return (
        self.linear_attn_config is not None
        and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
    )
```
`configuration_kimi_k3.py:152-156`. `[vllm]` agrees (same `+1`).

Converted to 0-based (the indices that appear in tensor names):

| | 0-based indices | count |
|---|---|---|
| KDA | `0,1,2, 4,5,6, 8,9,10, …, 84,85,86, 88,89,90` | **69** |
| MLA | `3,7,11,…,83,87, 91, 92` | **24** |

Run-length pattern over 0..92: `(KKK M) × 22` then `KKK MM`. The tail is **not** the repeating
motif — layers 91 and 92 are *both* MLA. A loader that generates the schedule from a `i % 4 == 3`
rule produces a wrong last block. Drive it from the config list, not from a modulus.

Layer 0 is KDA **and** dense-MLP (`first_k_dense_replace: 1` `[cfg]`); layers 1..92 are MoE.
Verified against real tensor names `[ckpt]`: layer 0 has `mlp.{gate,up,down}_proj`, layer 1 has
`block_sparse_moe.*`.

`[card]` line 75 states "69 KDA + 24 Gated MLA", confirming the split independently.

The 3:1 motif is deliberate. `[paper]` §3.2 (paper-derived): *"a uniform 3:1 ratio, i.e. repeating 3
KDA layers to 1 full MLA layer, provided the best quality–throughput trade-off"*, with NoPE on all
MLA layers — which K3 keeps (`mla_use_nope: true`, §4.3). K3's only departure is the doubled MLA
layer at the very end.

### 1.1 KDA weights are NOT quantized

`quantization_config.ignore` contains `re:.*self_attn.*` `[cfg]`, and KDA modules are named
`self_attn` (§6.1). Confirmed by `[ckpt]`: every KDA tensor is `BF16` or `F32`, none is mxfp4-packed.
So the 57 GiB of KDA projection weights (§6.3) are a **bf16 floor** on any K3 deployment; the
mxfp4-pack-quantized path applies only to routed experts.

---

## 2. The exact KDA recurrence

### 2.1 Dimensions `[cfg]`

| symbol | value | config field |
|---|---|---|
| `Hd` | 7168 | `hidden_size` |
| `H` | 96 | `linear_attn_config.num_heads` |
| `D` | 128 | `linear_attn_config.head_dim` (both key dim `K` and value dim `V`) |
| `HD` | 12288 | `H*D`, the projection width |
| `W` | 4 | `linear_attn_config.short_conv_kernel_size` |
| `LB` | −5.0 | `linear_attn_config.gate_lower_bound` |
| — | true | `linear_attn_config.use_full_rank_gate` |
| `eps` | 1e-5 | `rms_norm_eps` |

K3 is **not** GVA: `[fla]` supports `HV > H` (grouped value heads), but here `num_k_heads ==
num_heads == 96` and `head_k_dim == head_dim == 128` `[hf]:487-488`, so the group factor `G = 1`.
The state is square, `D × D`.

### 2.2 Per-token forward

Input `x_t ∈ R^7168` (already through `input_layernorm`). Per head `h ∈ [0,96)`, the layer carries a
state `S_h ∈ R^{D×D}` indexed `S_h[k_idx, v_idx]`.

> Throughout §2 the state is written **mathematically** as `S[k_idx, v_idx]`, matching `[fla]`'s
> `naive.py`. K3 **stores** it transposed — V-first, `[v_idx][k_idx]` — see §4.1. Keep the two
> apart: the math below is orientation-independent, the memory layout is not.

**Step 1 — projections** `[hf]:580-582`
```
q̃_t = W_q x_t     k̃_t = W_k x_t     ṽ_t = W_v x_t          ∈ R^12288   (no bias)
```

**Step 2 — causal depthwise short convolution, per channel `c ∈ [0,12288)`** `[hf]:504-518`, `[fla]` `short_conv.py:55-72`

```
q_t[c] = silu( Σ_{j=0..3} Wq_conv[c,j] · q̃_{t-3+j}[c] )        (zero-padded on the left)
```
identically for `k` and `v` with their own weights. `groups=hidden_size` → **depthwise**,
`padding=kernel_size-1` → **causal**, `bias=False` `[ckpt]` (no `*_conv1d.bias` tensor exists),
`activation='silu'` applied *after* the convolution `[hf]:507`.

This is three independent depthwise convs of width 4 over 12288 channels. It is what gives KDA
local (4-token) token-mixing that a pure linear-attention recurrence cannot express.

In decode the same thing is a rolling buffer — `[fla]` `short_conv.py:232-235` spells the semantics
out in a comment:
```python
# cache.copy_(cache.roll(shifts=-1, dims=-1))
# cache[:, :, -1] = x
# y = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
```
(then `silu`). That buffer is the conv half of the KDA state (§4.1).

**Step 3 — reshape** `[hf]:605-607`: `q,k,v : R^12288 → [H=96, D=128]`, head-major
(`'... (h d) -> ... h d'`, so channel `c = h*128 + d`).

**Step 4 — L2 normalize q and k, then scale q** `[fla]` `fused_recurrent.py:152-157`
```
q_h ← q_h / sqrt(Σ_d q_h[d]² + 1e-6)
k_h ← k_h / sqrt(Σ_d k_h[d]² + 1e-6)
q_h ← q_h · D^(-1/2)            # scale = K**-0.5 = 128**-0.5 ≈ 0.088388
```
Enabled by `use_qk_l2norm_in_kernel=True` `[hf]:620`. `scale` defaults to `K**-0.5` `[fla]`
`chunk.py:218-220`. **`eps` is inside the sqrt, not added to the norm** — `sqrt(Σq² + 1e-6)`.

`||k||₂ = 1` is load-bearing: it makes the delta-rule update (step 7) an exact rank-1 projection.

**Step 5 — forget gate (per head, per key-channel)** `[hf]:601-602`, `[fla]` `gate.py:66-70` and `fused_recurrent.py:158-170`
```
g̃_t = W_fb ( W_fa x_t )                                  ∈ R^12288, viewed [H,D]
g_t[h,d] = LB · sigmoid( exp(A_log[h]) · ( g̃_t[h,d] + dt_bias[h·D + d] ) )
a_t[h,d] = exp( g_t[h,d] )                                 ∈ ( e^-5 , 1 ) = (0.006738, 1)
```
with `LB = -5.0`. See §3 for why this branch and not the softplus branch.
`A_log` is indexed **per head** (`A_log + i_hv`), `dt_bias` is laid out `[H, D]` row-major
(`dt_bias + i_hv*K + o_k`) `[fla]` `fused_recurrent.py:159,162`.

The gate is a **vector** over the key dimension — GLA-style per-channel decay, not a scalar per
head. Each of the 128 key channels of each head forgets at its own data-dependent rate.

**Step 6 — write strength (per head, scalar)** `[hf]:603`, `[fla]` `fused_recurrent.py:188`
```
β_t[h] = sigmoid( (W_b x_t)[h] )     ∈ (0,1)
```
`W_b : R^7168 → R^96` — **one scalar per head**, not per channel. (`use_beta_sigmoid_in_kernel=True`
`[hf]:622`; `IS_BETA_HEADWISE=False` because `beta.shape[-1] == HV`.)
`allow_neg_eigval` is **not** set `[hf]:610-627`, so β stays in (0,1), not (0,2).

**Step 7 — the recurrence** `[fla]` `naive.py:59-63`, kernel `fused_recurrent.py:175-196`
```
(a)  S_h ← diag(a_t[h,:]) · S_h                     # decay:  scale row k_idx by a[h,k_idx]
(b)  u_t[h] = v_t[h] − S_hᵀ k_t[h]                  ∈ R^D    # delta / prediction error
(c)  S_h ← S_h + β_t[h] · k_t[h] u_t[h]ᵀ            # rank-1 write
(d)  o_t[h] = S_hᵀ q_t[h]                           ∈ R^D    # read
```
Order matters: **decay is applied before the delta correction is computed**, so `u` is the error
against the *already-decayed* state.

Substituting (b) into (c) gives the canonical form and separates the two mechanisms:
```
S_h ← ( I − β_t[h] k_t[h] k_t[h]ᵀ ) · diag(a_t[h,:]) · S_h  +  β_t[h] k_t[h] v_t[h]ᵀ
      └──────── delta rule ────────┘  └── forget gate ──┘     └── write ──┘
```

**These are two distinct, composed memory mechanisms — do not conflate them:**

- **Forget gate** `diag(a)`: *untargeted*, multiplicative, decays the whole state. Per (head,
  key-channel), data-dependent, bounded to `[e^-5, 1)` by `gate_lower_bound`. This is the
  GLA / Mamba-style decay.
- **Delta rule** `(I − β k kᵀ)`: *targeted*. Because `||k||₂ = 1`, this is `I` minus `β` times an
  orthogonal projector onto `k`. It erases only the component of memory stored at key `k` (by a
  factor `1−β`) and leaves everything orthogonal to `k` untouched, then writes `v` there. β=0 = no
  write; β=1 = full overwrite of that key's slot.

KDA = **gated delta rule**. This is why vLLM files it under `layers/mamba/gdn/` ("gated delta net")
`[vllm]`.

`[paper]` §3 Eq. (1) states the identical recurrence, which is an independent check on the whole of
§2.2 (paper-derived):

> `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ ∈ R^{d_k×d_v}` ; `o_t = S_tᵀ q_t`

with `α_t = exp(g_t)` the per-channel decay. `[paper]` §1 names the distinction this document draws
above: *"While GDN, similar to Mamba2, employs a coarse head-wise forget gate, KDA introduces a
channel-wise variant in which each feature dimension maintains an independent forgetting rate, akin
to GLA."* — the **decay vector is per-channel**, while the learned rate `A_log` behind it is
**per-head**, which is exactly what the checkpoint's 96 live entries encode (§3.2).

**Step 8 — output gate + norm** `[hf]:651-659`
```
ĝ_t = W_g x_t                            ∈ R^12288, viewed [H,D]   # full-rank, see §3.3
y_t[h] = RMSNorm_D( o_t[h] ; w_onorm, eps=1e-5 )  ⊙  sigmoid( ĝ_t[h] )
out_t  = W_o · concat_h( y_t[h] )        ∈ R^7168
```
`FusedRMSNormGated(head_dim, eps, activation='sigmoid')` `[hf]:539-540`; the gate is applied
**after** the norm and the sigmoid is on the *un-normalized* gate projection `[vllm]`
`fla/ops/kda.py:484-486`. `w_onorm` is a single `[128]` vector **shared across all 96 heads**
`[ckpt]`. Note the RMSNorm here normalizes over `D=128` (within a head), not over 12288.

Then the block residual: `hidden = residual + out` `[hf]:963`, except that K3 sets
`attn_res_block_size: 12`, which routes the block through `_forward_attn_residual` instead
(`self_attention_res_norm` / `self_attention_res_proj`, §6.2) — **out of scope here, owned by
another agent**.

### 2.3 State does not depend on position

There is **no RoPE, no positional embedding, and no attention mask** in a KDA layer. Order
information is carried entirely by (i) the sequential recurrence and (ii) the width-4 causal conv.
The `attention_mask` argument is used only to unpad variable-length batches `[hf]:567-570`.
This also means a KDA layer has **no context-length limit of its own** — the 1M window is an MLA
property.

---

## 3. Gate semantics — the two K3-specific knobs

### 3.1 `gate_lower_bound: -5.0` — which branch is live

`[fla]` implements two gate activations, selected by `safe_gate`/`lower_bound` (`gate.py:52-70`,
kernel `gate.py:118-124`):

```python
if not USE_LOWER_BOUND:
    b_yg = -exp(b_A) * softplus(b_g)                 # unbounded decay
else:
    b_yg = lower_bound * tl.sigmoid(exp(b_A) * b_g)  # bounded decay
```

K3 passes `safe_gate=self.gate_lower_bound is not None` → `True`, and `lower_bound=-5.0`
`[hf]:623-624`. **The bounded branch is live.** The `chunk_kda` docstring states it explicitly
`[fla]` `chunk.py:246-257`:

> `lower_bound` … When set together with `safe_gate=True`, changes the gate activation from
> `-exp(A_log) * softplus(g + dt_bias)` to `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`,
> which naturally clamps the output to `[lower_bound, 0)`. Recommended value: `-5` (i.e. per-step
> decay `exp(-5) ≈ 0.0067`).

Consequences:
- `g ∈ [-5, 0)` strictly, so the per-step decay `a = exp(g) ∈ (0.006738, 1)`. The state can never be
  fully zeroed by the gate in one step, and it can never grow.
- Over `n` steps the cumulative log-decay is bounded below by `-5n`. In fp32 this means a chunked
  implementation may compute `exp(G_i − G_j)` with `G_i − G_j ∈ [-5·BT, 0]`; for `BT = 64` that is
  `[-320, 0]`, which underflows fp32 to 0 harmlessly but **must not be evaluated as
  `exp(G_i)/exp(G_j)`** — compute the difference first.
- `[fla]` notes the bound is also what "enable[s] M=16 TensorCore acceleration" `[fla]`
  `chunk.py:246-249`. Bounded gates make the intra-chunk decay matrix safe to hold in low precision.
  **[inferred]** this is the reason K3 sets it, not just numerical hygiene.

The bounded gate is **not in `[paper]`** — it postdates it, originating in `fla` PRs #701/#703
(2025-12-29/30, *"[KDA] Add lowerbound gate function"*) and #814 (2026-04-05, `safe_gate`). **K3 is
the first checkpoint to ship `gate_lower_bound`.** So do not expect the paper, or any
Kimi-Linear-era implementation, to describe it.

Measured checkpoint values `[ckpt]`, layer 0: `dt_bias ∈ [-7.894, 0.179]`, mean **−4.575**;
`exp(A_log) ∈ [0.471, 11.776]`. With `g̃ ≈ 0` the gate sits at `LB·sigmoid(negative·large) ≈ 0`, i.e.
`a ≈ 1`. **The learned bias parks the model in "retain" mode**, and the gate has to be actively
driven positive to forget. Consistent with a 1M-context model.

### 3.2 `A_log` — per head, and shipped ZERO-PADDED to 128

`[hf]:520-521` declares `A_log` with `num_heads = 96` entries. The checkpoint ships **`[128]`**.
This looked like a contradiction; it is not. `[ckpt]`, across layers 0, 1, 2, 5, 29:

```
layer  0: len=128  nonzero idx: min=0 max=95 count=96   A[96:] all zero? True
layer  1: len=128  nonzero idx: min=0 max=95 count=96   A[96:] all zero? True
layer  2: len=128  nonzero idx: min=0 max=95 count=96   A[96:] all zero? True
layer  5: len=128  nonzero idx: min=0 max=95 count=96   A[96:] all zero? True
layer 29: len=128  nonzero idx: min=0 max=95 count=96   A[96:] all zero? True
```

Exactly the first 96 entries are non-zero in every layer checked; entries 96..127 are exactly 0.0.
`A_log` is a **per-head `[96]` parameter stored zero-padded to `head_dim = 128`**. The kernel
indexes it as `A_log + i_hv` with `i_hv ∈ [0,96)` `[fla]` `fused_recurrent.py:159`, so the padding
is never read.

> **Loader rule: `A_log = tensor[:96]`.** Do not reshape, do not treat as per-channel. A loader that
> asserts `A_log.numel() == num_heads` will reject a valid checkpoint; one that consumes all 128
> values as if per-head-dim silently computes the wrong decay for every token of every KDA layer.

Both public K3 implementations do exactly this narrow `[k3impl]`, which is independent confirmation:

- SGLang, `python/sglang/srt/models/kimi_k3.py:1403-1429` — *"K3 checkpoint stores A_log as
  `[head_dim]` (128), but the FLA kernel expects exactly `local_num_heads` elements … a custom
  `weight_loader` that handles both the old 4-D format and the K3 1-D `[head_dim]` format by
  narrowing to the first `num_heads` elements then TP-sharding."*
- vLLM PR #50089, `vllm/models/kimi_k3/nvidia/kda.py:412-415` declares
  `nn.Parameter(torch.empty(self.local_num_heads))` with an `a_log_weight_loader` that narrows then
  shards.

Note the consequence for the reference path: **the HF-shipped `modeling_kimi_linear.py` will
size-mismatch on load** (`[96]` parameter vs `[128]` checkpoint tensor, `[hf]:520-521`). The shipped
reference is broken as-is and needs the same narrow. Do not use "it loads in transformers" as a
correctness signal.

`dt_bias` has **no** padding: `[12288]` = `[96,128]` exactly, zero zeros `[ckpt]`. Neither do any of
the projection matrices (zero all-zero rows in `q/k/v/g/f_b/b_proj` `[ckpt]`).

`[inferred]` the padding is a storage/alignment artifact of the training or export pipeline
(padding a per-head vector out to `head_dim`). I did not find code that produces it.

### 3.3 `use_full_rank_gate: true` — output gate is a single dense matrix

`[hf]:531-537`:
```python
if self.use_full_rank_gate:
    self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)   # 7168 -> 12288
else:
    self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)   # 7168 -> 128
    self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)    # 128 -> 12288
```
K3 takes the **first** branch. Confirmed by `[ckpt]`: `self_attn.g_proj.weight [12288, 7168]`
exists in every KDA layer, and **no `g_a_proj`/`g_b_proj` tensor exists anywhere**.

Note the asymmetry, which is easy to get backwards:
- the **output** gate `ĝ` is **full rank** (`g_proj`, 88.1 M params)
- the **forget** gate `g̃` is still **low rank 128** (`f_a_proj` → `f_b_proj`), regardless of this flag

`use_full_rank_gate` is a **K3 change relative to the paper.** `[paper]` §3.2 Eq. (10) gives the
output gate as low-rank — *"the output gate adopts a low-rank parameterization … while maintaining
performance comparable to full-rank gating"* — so `Kimi-Linear` used `g_a_proj`/`g_b_proj` and K3
switched to the flat matrix. That is why every pre-K3 implementation has the low-rank pair and only
`[k3impl]` has `g_proj` (§3.4). `[paper]` §3.2 also confirms the forget gate's low-rank pair has
*"rank equal to the head dimension"* — i.e. 128, matching `f_a_proj [128,7168]` exactly.

### 3.4 Neither vLLM 0.23.0 nor vLLM `main` implements either knob — but `[k3impl]` does

`[vllm]` `kimi_gdn_linear_attn.py` builds `g_a_proj`/`g_b_proj` (low-rank output gate) and its gate
kernel computes only `-exp(A_log) * softplus(g + dt_bias)` (`fla/ops/kda.py:1232-1245`). That build
cannot load K3's `g_proj` and would compute the wrong forget gate. **vLLM `main` is the same**: no
`kimi_k3.py` in `model_executor/models/`, no `KimiK3*` in `registry.py`, and
`kimi_gdn_linear_attn.py:200-217` still has `A_log = nn.Parameter(torch.empty(1,1,H,1))` with the
low-rank output gate and no `gate_lower_bound`.

**Do not use any released vLLM as the K3 oracle, and do not expect it to serve K3.** It remains the
best reference for *state layout, paging and TP sharding* (§4, §7), which are knob-independent.

`[k3impl]` implements both knobs and is the oracle to check against:

- **`use_full_rank_gate`** — vLLM PR #50089 `kimi_k3/nvidia/kda.py:325-327` *asserts* it:
  `assert kda_config.get("use_full_rank_gate", False), "KimiK3DeltaAttention requires a full-rank
  gate"`. The low-rank fallback was **dropped**, and `g_proj` is folded into a merged
  `in_proj_qkvgfab` with output sizes `[proj_size]*4 + [head_dim, num_heads]` (q, k, v, **g**, f_a,
  beta) — i.e. an output-dimension merge of six projections, which is exactly the pattern §7.4
  argues is safe.
- **`gate_lower_bound`** — read from config and asserted into `[-5.0, 0)`
  (`kimi_k3/nvidia/kda.py:417-423`); the vendored kernel takes the same branch this document
  specifies: `if USE_LOWER_BOUND: b_gate = lower_bound * tl.sigmoid(b_a * b_s) else: b_gate =
  -b_a * b_softplus`.
- Its unit test states the intended gate in one line —
  `expected_gate = lower_bound * sigmoid(A_log.exp()[None,:,None] * (raw_g + dt_bias))`
  (`tests/models/kimi_k3/test_kda.py:237-238`) — which matches §3.1 exactly, including `A_log`
  broadcasting over the **head** axis.

---

## 4. What the KV cache becomes

This is the consequence that matters most for plow.

### 4.1 A KDA layer has no KV ring

A KDA layer stores, per sequence, per layer:

| item | shape | elements | note |
|---|---|---|---|
| recurrent state | `[H, D, D]` = `[96,128,128]` | 1 572 864 | fp32 in `[vllm]` (`kda_state_dtype` returns `torch.float32`) |
| conv state (q,k,v) | `[3·H·D, W]` = `[36864, 4]` | 147 456 | `[fla]` convention `[N,D,W]`, `short_conv.py:139-140` |

`[vllm]` uses `W−1 = 3` slots (`mamba_utils.py:237-260`, `conv_state_shape = (3·H·D, W−1)`), `[fla]`
uses `W = 4`. Both are correct — `[fla]` keeps the current token in the buffer, `[vllm]` prepends
it. Pick one and be consistent; the difference is 36 864 elements/layer.

**State layout is V-first.** `[hf]:625` passes `transpose_state_layout=True`. In current `[fla]`
that is a deprecated alias for `state_v_first` (`chunk.py:352-360`: *"`transpose_state_layout` is
deprecated and renamed to `state_v_first`"*), which stores the state as `[V, K]` instead of `[K, V]`
(`chunk.py:267-268`). Since `V == K == 128` the byte count is unchanged, but **the contiguous axis
is the key axis**, i.e. `S[v_idx][k_idx]`. Getting this backwards transposes the state and
silently produces garbage that still has the right norm.

### 4.2 Bytes

Per KDA layer per sequence:

| dtype choice | recurrent | conv (W=4) | total/layer | ×69 layers |
|---|---|---|---|---|
| rec fp32, conv fp32 | 6.000 MiB | 0.5625 MiB | **6.5625 MiB** | **452.8 MiB** |
| rec bf16, conv fp32 | 3.000 MiB | 0.5625 MiB | 3.5625 MiB | 245.8 MiB |

**Constant. Independent of context length.** 69 layers × 6.5625 MiB = **452.8 MiB per sequence**
whether the context is 1 token or 1 M tokens.

Broken down the way a tensor declaration and a kernel tiling need it (fp32):

| granule | bytes |
|---|---|
| one `v`-column, `state[h][v][0:128]` | **512 B** (contiguous — see §7.2) |
| one head, `state[h]` = `[128,128]` | **64 KiB** |
| one layer, one sequence — `state` | 6.000 MiB |
| one layer, one sequence — `conv_state` | 0.5625 MiB |
| one layer, one sequence — total | **6.5625 MiB** |
| 69 KDA layers, one sequence | **452.8 MiB** |
| 69 KDA layers × 64 concurrent sequences | **28.3 GiB** |

Recommendation: **keep the recurrent state fp32.** `[vllm]` hardcodes it (`mamba_utils.py:119-125`
returns `(state_dtype, torch.float32)`) and every `[fla]` kernel accumulates the state in fp32. The
state is a *running accumulator* over up to 10⁶ rank-1 updates; bf16 here is not the same risk class
as bf16 in a KV ring, where each entry is written once. **[inferred]** — I did not measure the
degradation, and it is worth measuring, because bf16 halves the number to 245.8 MiB/seq.

### 4.3 Against the MLA layers and a 1M context

MLA is NoPE — `assert self.use_nope` `[hf]:396`, `mla_use_nope: true` `[cfg]`, and no rotary is ever
applied (`self.rotary_emb = None`, `[hf]:403`; the "rope" split is carried through unrotated,
`[hf]:427-440`). The cacheable latent per token per layer is therefore
`kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` elements = 1152 B in bf16.

At `max_position_embeddings: 1048576` `[cfg]`, one sequence:

| | per layer @1M | × layers | total |
|---|---|---|---|
| 24 MLA layers (latent, bf16) | 1.125 GiB | ×24 | **27.00 GiB** |
| 69 KDA layers (state, fp32) | 6.5625 MiB | ×69 | **0.44 GiB** |
| **K3 as built** | | | **27.44 GiB** |
| counterfactual: all 93 layers MLA | 1.125 GiB | ×93 | 104.63 GiB |

**KDA removes 74% of the 1M-context cache — a 3.81× reduction.** And the 69 KDA layers contribute
1.6% of the remaining total.

Break-even is worth knowing: one KDA layer's 6.5625 MiB state equals one MLA layer's latent ring at
**5 973 tokens**. Below ~6k context KDA costs *more* memory than attention would; above it, KDA is
flat and MLA is linear. K3's advantage is entirely a long-context advantage.

If plow ever materializes MLA instead of caching the latent (96 heads × 192 K-dim + 96 × 128 V-dim =
30 720 elem/token/layer), 24 layers at 1M is **1.41 TiB**. Cache the latent.

### 4.4 What this does to the block/paging abstraction

A KDA layer's cache is **not addressable by token**. There is no `(seq, token) → slot` mapping; there
is one slot per (sequence, layer). Concretely:

- **Allocation** is at sequence admission, not per block. 452.8 MiB/seq is a hard, immediate cost:
  at 64 concurrent sequences that is **28.3 GiB of state before a single token is cached**. This is
  a real scheduling constraint — it caps concurrency independently of context length, and it is the
  opposite shape from a KV ring (which is cheap at admission and grows).
- **Eviction / preemption** cannot drop a suffix. Recomputing a preempted sequence's KDA state means
  replaying the whole prefix. Either keep the state or pay full prefill.
- **Prefix sharing / prefix caching** works only at a *prefix boundary*: two sequences sharing the
  first `n` tokens can share one snapshot of the state at `n`, but they must copy-on-write from
  there (6.5625 MiB × 69 per fork). Block-level sharing at arbitrary offsets is impossible.
- **Beam search / forking** requires an explicit state copy (`[hf]:175-197` `reorder_cache` does
  `index_select` on both conv and recurrent states).
- Batched decode wants a **paged state indirection**: `[fla]`'s decode kernel already takes
  `ssm_state_indices` and does `h0 + ssm_state_indices[i_n]*stride` (`fused_recurrent.py:127-135`)
  with `INPLACE_FINAL_STATE` writing back through the same index. This is the right model for plow:
  a flat pool of `n_slots × 69 × [96,128,128]` fp32, plus an int32 slot table per sequence.

---

## 5. Prefill / decode split

### 5.1 Two genuinely different algorithms

`[hf]:561` picks the mode:
```python
mode = 'fused_recurrent' if use_cache and q_len == 1 else self.mode   # self.mode == 'chunk'
```
So: **`q_len == 1` → sequential recurrent; anything else → chunked.**

**Decode (`fused_recurrent_kda`).** Exactly §2.2 step 7, one token at a time. Per token per layer:
`4·H·D·D = 6.29 M` MACs. It is a pure **GEMV/outer-product** workload — every operation is rank-1
against a `[128,128]` state. No matmul units are usable on the state path. Arithmetic intensity is
terrible: 6 MiB of state read+written per layer per token to do 6.3 M MACs, i.e. **~2 FLOP/byte**.
The state is touched once per token, a 12 MiB/layer/token floor (read + write) that no batching
amortizes *per sequence*. But note the comparison that matters: that 12 MiB sits against **846.7 MiB
of weight traffic in the same layer** (§6.3), so the state is 1.5% of the layer, not a bottleneck —
see §5.3 and §7.2.

**Prefill (`chunk_kda`).** Chunked, with the WY / UT-transform representation. Per chunk of `BT`
tokens (`[fla]` default `BT = 64`, `naive.py:74`), from `naive.py:110-166`:

1. `G = cumsum(g)` within the chunk (per head, per key-channel) — a length-64 causal scan over 128
   channels.
2. Build `A[i,j] = β_i · ⟨ k_i · exp(G_i − G_j), k_j ⟩` for `j < i`, strictly lower triangular.
3. Invert `(I + A)` by forward substitution (`for i in range(1,BT): A[i,:i] += (A[i,:,None]*A[:,:i]).sum(-2)`),
   then scale by `β`. This is the UT transform — it turns 64 sequentially-dependent rank-1 delta
   updates into one dense `[64,64]` operator.
4. `w = A @ (exp(G) · k)`, `u = A @ v` — the chunk's aggregate write.
5. Per chunk, against the carried state `S`:
   ```
   v' = u − w S
   o  = (q · exp(G)) S  +  tril(A_qk, −1) v'          where A_qk[i,j] = ⟨q_i·exp(G_i−G_j), k_j⟩
   S  ← diag(exp(G_last)) S  +  Σ_i exp(G_last − G_i) k_i v'_iᵀ
   ```
   — an *inter-chunk* term (state carried forward) plus an *intra-chunk* term (a masked
   64×64 attention-like product). All four are dense matmuls.

The chunk form is **matmul-bound and tensor-core-friendly**; the recurrent form is not. That is the
entire reason both exist.

### 5.2 Two algorithms, two opcodes

They share the *definition* but not the *shape*:

| | decode | prefill |
|---|---|---|
| parallelism | over (seq, head) only — `T` is serial | over (seq, head, chunk) with `T/64` chunks pipelined |
| inner op | rank-1 outer product, `[128]` vectors | dense `[64,64]×[64,128]` and `[128,128]` matmuls |
| units | VALU / GEMV | MFMA / tensor cores |
| state traffic | read + write once per token | read + write once per **chunk** (÷64) |
| extra work | none | build + invert `[64,64]` `A`, cumsum `G` |

Setting `BT = 1` in the chunk kernel degenerates to the recurrent kernel but keeps all the UT
machinery, and setting `T = 1` in the recurrent kernel is what decode already is. **Write two
device code paths.** `[fla]` and `[vllm]` both do (`chunk_kda` vs `fused_recurrent_kda`;
`kimi_gdn_linear_attn.py:397-441` dispatches prefill vs decode to different entry points).

In plow these become **two opcodes** — `KdaStateStep` and `KdaChunkScan` (§7.6) — not two branches
of one. `Mamba2Scan` puts both behind `i0 = T` in a single op, but that is part of what makes it a
cautionary tale rather than a template (§7.6): the two paths share no inner loop, no tiling, and no
execution units, so a shared opcode buys one dispatch arm and costs a kernel that is two kernels
wearing a trench coat.

They **must** produce bit-comparable states, because a chunked-prefill scheduler will hand a
partially-prefilled sequence to the decode path mid-sequence. Both must agree on:
V-first state layout, fp32 accumulation, `exp` vs `exp2` scaling of `G`, and the `+1e-6`-inside-sqrt
L2 norm. This is the correctness gate for the whole feature (§7.8), and splitting the opcodes makes
it a **cross-op** invariant that has to be tested explicitly rather than assumed.

Two pieces genuinely **are** shared and should be their own packets, used by both paths (§7.4, §7.5):
the **short conv** (§2.2 step 2) and the **gate + β pre-pass** (§2.2 steps 5–6), the latter a pure
elementwise kernel over `[T, 96, 128]`. `[vllm]` factors the gate out for decode (`fused_kda_gate`)
and fuses it into the chunk kernel for prefill (`use_gate_in_kernel`); factoring it out in **both**
costs one `[T,12288]` fp32 round-trip and buys an independently testable op plus two freed tensor
handles (§7.6).

### 5.3 Where the decode time actually goes — the surprise

Per token, per KDA layer, MACs:

| | MACs | share |
|---|---|---|
| big GEMVs: `q,k,v,g` projections + `o_proj` (5 × 7168×12288) | 440 401 920 | **97.86%** |
| low-rank forget gate (`f_a`,`f_b`) + `b_proj` | 3 178 496 | 0.71% |
| **recurrent state update** | 6 291 456 | **1.40%** |
| short conv | 147 456 | 0.03% |
| total | 450 019 328 (900 MFLOP) | |

**The recurrence is 1.4% of KDA decode arithmetic.** 98% is five dense `7168×12288` GEMVs that
plow's existing GEMV path already handles. Likewise the weight traffic — 846.7 MiB/layer (§6.3) —
dwarfs the 6.5625 MiB of state.

The implication for plow: **the new KDA ops are architecturally mandatory but are not where the
decode time is.** The five projections are, and they are already covered by existing opcodes (§7.1).

This does **not** license fusing the KDA layer into one packet to save op count — see §7.4, where
`GLM_GROUP=1`'s measured **+2.88 ms for a 38% op-count reduction** settles that. The distinction
that matters is output-dimension merging (safe, `GemvQkv = 22` already does it) versus loop-dimension
merging (fatal). A merged `qkvg` projection is the former.

---

## 6. Weight names and shapes

### 6.1 Per-KDA-layer tensors

Prefix: `language_model.model.layers.{i}.self_attn.` — note the **`language_model.`** prefix (K3 is
multimodal; `modeling_kimi_k3.py` nests the text tower) and note that KDA modules are called
`self_attn`, the *same* name MLA layers use. Distinguish by layer index, not by name.

All 14 read from real safetensors headers `[ckpt]` (shard 1, layer 0; identical in layers 1,2,5,29):

| tensor | dtype | shape | derivation |
|---|---|---|---|
| `q_proj.weight` | BF16 | `[12288, 7168]` | `[H·D, Hd]` |
| `k_proj.weight` | BF16 | `[12288, 7168]` | `[H·D, Hd]` |
| `v_proj.weight` | BF16 | `[12288, 7168]` | `[H·D, Hd]` |
| `g_proj.weight` | BF16 | `[12288, 7168]` | `[H·D, Hd]` — output gate, full-rank (§3.3) |
| `o_proj.weight` | BF16 | `[7168, 12288]` | `[Hd, H·D]` |
| `f_a_proj.weight` | BF16 | `[128, 7168]` | `[D, Hd]` — forget-gate down-proj |
| `f_b_proj.weight` | BF16 | `[12288, 128]` | `[H·D, D]` — forget-gate up-proj |
| `b_proj.weight` | BF16 | `[96, 7168]` | `[H, Hd]` — β logits, one per head |
| `q_conv1d.weight` | **F32** | `[12288, 1, 4]` | `[H·D, 1, W]` depthwise |
| `k_conv1d.weight` | **F32** | `[12288, 1, 4]` | `[H·D, 1, W]` |
| `v_conv1d.weight` | **F32** | `[12288, 1, 4]` | `[H·D, 1, W]` |
| `A_log` | **F32** | `[128]` | **per-head `[96]`, zero-padded to `D`. Slice `[:96]`** (§3.2) |
| `dt_bias` | **F32** | `[12288]` | `[H·D]`, laid out `[H, D]` row-major, no padding |
| `o_norm.weight` | **F32** | `[128]` | `[D]`, shared across all 96 heads |

No biases exist for any projection or conv. No `g_a_proj`/`g_b_proj`. Weight matrices are stored
`[out, in]` (torch `nn.Linear` convention), which is **already** what plow's `Gemm` wants for its
`B` operand — `C[M,N] = A[M,K]·B[N,K]^T`, `dev.rs:87-89`. No transpose at load (§7.1).

### 6.2 Sibling tensors on the same layer (not KDA, listed so the loader is complete)

`input_layernorm.weight [7168]`, `post_attention_layernorm.weight [7168]`,
`self_attention_res_norm.weight [7168]`, `self_attention_res_proj.weight [1, 7168]`,
`mlp_res_norm.weight [7168]`, `mlp_res_proj.weight [1, 7168]` — the AttnRes block
(`attn_res_block_size: 12`). Layer 0 additionally has `mlp.{gate,up,down}_proj`; layers ≥1 have
`block_sparse_moe.*`. **Owned by other agents.**

For contrast, an MLA layer (0-based 3) has: `self_attn.{q_a_proj, q_a_layernorm, q_b_proj,
kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj, g_proj}` `[ckpt]` — note it *also* has a
`g_proj` (`mla_use_output_gate: true`), so "has `g_proj`" does **not** identify a KDA layer.
`self_attn.A_log` does.

### 6.3 Weight footprint

| | per layer | × layers |
|---|---|---|
| KDA | 846.7 MiB | 69 → **57.05 GiB** |
| MLA | 442.9 MiB | 24 → **10.38 GiB** |
| attention total (unquantized, §1.1) | | **67.43 GiB** |

Of the 846.7 MiB in a KDA layer, 840 MiB is the five `[12288,7168]`/`[7168,12288]` bf16 matrices.
Everything KDA-specific (convs, `A_log`, `dt_bias`, `o_norm`, `f_*`, `b_proj`) is **6.7 MiB**.

### 6.4 Index / shard notes

`model.safetensors.index.json` is **absent from the local snapshot but present upstream**; I fetched
it (59.8 MB, 497 220 tensors, `total_size` 1 560 860 324 864 B ≈ 1.42 TiB). Shard naming is
`model-{i:05d}-of-000096.safetensors` — **96 shards, not 36**; 36 were merely downloaded at the time
of writing. Layer `i`'s tensors live in shard `i+1` for the layers checked.

---

## 7. Mapping onto plow

Inventory read from `crates/packet/src/dev.rs` (1060 lines, `DevOp::COUNT = 102`) and its C mirror
`runtime/common/dev_isa.h`.

**Headline: nothing KDA-shaped exists.** A grep for `KDA | delta.rule | DeltaNet | gated delta |
linear attention` over the tree returns zero hits. There is no L2-norm op, no convolution op, no
standalone sigmoid/exp/softplus/cumsum/multiply op, and no standalone gated RMSNorm. The single
recurrent-state precedent is `Mamba2Scan = 90`.

**The design principle for everything below.** KDA is *state-carrying*, but state-carrying does not
mean *register-resident*. The recurrent state is a **declared tensor in HBM** — the same kind of
object as a KV ring, only fixed-size — and a decode step is a **read–modify–write** over it. Once
that is the frame, three things follow, and they are the substance of §7.2–§7.6:

1. The register-pressure verdict against KDA **inverts**: 32 VGPRs/lane is what one-workgroup-per-head
   costs, not what KDA costs. Tiled by `v`-column it is 4–8 (§7.2).
2. The layer decomposes into **many small packets whose dependencies plow's compiler already
   expresses**, and that is a feature, not a cost: §6g-KNOBS measured that collapsing packets to
   reduce op count **loses** (§7.4).
3. Every proposal gets a **workgroup count checked against 256** before it is allowed (§7.3), because
   head-parallelism alone reproduces the `MlaMergeFold` occupancy defect.

### 7.1 What is already covered

Of the KDA forward, these need **no new opcode** — they are existing dense-linear and elementwise
work:

| step | shape | existing op |
|---|---|---|
| `q/k/v/g` projections | `[12288,7168]` ×4 | `Gemm`/`GemmMed`/… (prefill), `Gemv = 10` (decode) |
| `o_proj` | `[7168,12288]` | same |
| `f_a_proj`, `f_b_proj`, `b_proj` | `[128,7168]`, `[12288,128]`, `[96,7168]` | same (thin-N) |
| `input_layernorm` | RMSNorm over 7168 | `RmsNorm = 1` (`t0=out t1=x t2=gamma · i0=rows i1=feat · f0=eps`) |
| residual add | `[7168]` | `Residual = 4` |

That is 97.9% of the decode arithmetic (§5.3) and ~99.2% of the weight bytes. **The dense path is
already there**, and the operand convention lines up for free: `Gemm` computes
`C[M,N] = A[M,K]·B[N,K]^T` with `B` stored `[out_features, in_features]` (`dev.rs:87-89`) — exactly
how HF stores an `nn.Linear` weight (§6.1). No transpose at load.

Nothing else is covered. Specifically **absent**: L2-norm (no op, no kernel, no helper); depthwise
or causal conv (exists only as `mamba_conv_at`, a `static __device__` helper private to
`runtime/nvidia/op_mamba.cuh:47-64`); standalone sigmoid/exp/softplus/cumsum/elementwise-multiply
(the entire elementwise inventory is `Residual`, `Glu`, `SoftCap`, `Embed`, `Argmax`, `ArgmaxFin`);
standalone gated RMSNorm (exists only inside `Mamba2Scan`'s epilogue).

### 7.2 The state is a DECLARED TENSOR, not registers — and that dissolves the register objection

A sibling kernel audit concluded KDA "does not fit as an arm": the `[128,128]` f32 state costs
**32 VGPRs/lane** against decode's **8 registers of headroom** (248/256, occupancy 2, zero spill),
and recommended a 4th co-resident code object.

**That arithmetic is right and its premise is wrong.** 32 VGPRs/lane is exactly what you get if one
workgroup owns one whole head:

```
128 × 128 f32 per head / 512 lanes per workgroup = 32 f32 per lane
```

The state does not have to be workgroup-resident. It is a **declared tensor in HBM**, exactly like a
KV ring (§7.3), and a decode step is a **read–modify–write over that tensor**. What a lane holds is
a *tile*, and the tile size is a free parameter.

**The tiling that makes it free.** The whole KDA decode state update decomposes over
`(head, v_column)` with **no cross-lane reduction at all** — because of two facts that compose:

1. **V-first storage makes a `v`-column contiguous.** §4.1: K3 stores `S[h][v][k]`. A column
   (fixed `h`, fixed `v`, all 128 `k`) is 512 contiguous bytes.
2. **Both reductions in the step are over `k`, for fixed `v`.** `S'ᵀk` and `S'ᵀq` each produce one
   scalar per `v` by summing over `k`. So each output element is a private, contiguous,
   512-byte dot product. Nothing crosses a column.

And one algebraic step removes the read-after-write hazard between the write and the read. From
§2.2 step 7, with `S' = diag(a)·S`:

```
o = S_newᵀ q = (S' + β k uᵀ)ᵀ q = S'ᵀq + β (k·q) u
```

so **`o` never needs the updated state.** One pass over the column suffices:

Index `j` = the value channel (which column), index `i` = the key channel (position within it):

```
for each (h, j):                             # 96 × 128 = 12288 independent work items per layer
    load Sc[0:128] = state[h][j][0:128]      # 512 B contiguous
    Sc[i] *= exp(g[h][i])                    # decay, elementwise over i   (g from KdaGate)
    p_k = Σ_i Sc[i] · k[h][i]                # private reduction over i
    p_q = Σ_i Sc[i] · q[h][i]                # private reduction over i
    u   = v[h][j] − p_k                      # delta (a scalar, for this column)
    o[h][j] = p_q + β[h] · s[h] · u          # s[h] = q[h]·k[h], one scalar per head
    store state[h][j][i] = Sc[i] + β[h] · u · k[h][i]
```

`s[h] = q[h]·k[h]` is 96 dot products of length 128 per layer — recompute it redundantly per
workgroup (128 MACs) rather than spending a packet and a barrier on it. **[inferred]**, but the
alternative costs a coarse gate on the critical path to save 128 MACs.

**Register budget as a function of the tile.** Let `BV` be the number of `v`-columns a workgroup
holds in flight, 512 lanes:

| what a workgroup owns | f32/lane | VGPRs for state |
|---|---|---|
| a whole head (`BV = 128`) — *the audit's assumption* | 16384/512 | **32** |
| half a head (`BV = 64`) | 8192/512 | 16 |
| `BV = 32` | 4096/512 | **8** |
| `BV = 16` | 2048/512 | **4** |
| `BV = 8` | 1024/512 | 2 |

`BV` is a loop bound, not a constraint: the columns are independent, so a workgroup assigned 48
columns processes them as 3 sub-tiles of 16 with **nothing carried between sub-tiles**. The state
requirement is therefore **whatever you choose it to be**, down to 2 VGPRs/lane.

Broadcast operands per head — `q[128]`, `k[128]`, `g[128]` f32 — go in LDS: **1.5 KiB per head**,
trivial against a 64 KiB budget.

> **Corrected conclusion: the "32 VGPRs, does not fit" result is an artifact of one-workgroup-per-head,
> not a property of KDA.** At `BV = 16` the state costs 4 VGPRs/lane. Whether *that* fits inside the
> decode object's 8 spare registers alongside the kernel's own working set is a **real question I
> did not measure** — the working set is not in this document. But it is now a tuning question with
> a knob, not a structural veto, and the co-resident-object option survives as a fallback for a
> *much smaller* kernel than the one that was priced.

**Traffic, since that is the real cost.** One pass = 6 MiB read + 6 MiB write per layer per token,
plus 1.125 MiB for the conv state: **≈13.1 MiB/layer/token**. At the measured 6200 GB/s HBM ceiling
(contract §5) that is **2.1 µs/layer**, **146 µs/token** across 69 layers — against **9.7 ms/token**
for the 57 GiB of KDA weights (§6.3). **The state is 1.5% of KDA's decode traffic**, matching its
1.4% share of the MACs (§5.3). The ratio is TP-invariant: both the state and the weights shard by
head.

A two-pass variant (no state held at all, read the column twice) costs 18 MiB instead of 12 MiB —
**+50% state traffic, which is +0.75% of layer traffic.** That is the price of dropping to 0 VGPRs
of state, and it is small enough to be a legitimate fallback.

### 7.3 Workgroup counts — check every proposal against 256

The contract's standing defect here is `MlaMergeFold`: **16 of 256 workgroups doing all the work,
8.69 ms of a 34.68 ms token**, now rewritten wave-cooperatively for a −18% win (§6b-STALE). And
`Mamba2Scan` is emitted on **one CU** (`let one = vec![0u32]`, `crates/devgen/src/mla.rs:3981`) —
**1/256 = 0.4%**. Any KDA design must be checked against 256 explicitly.

`blocks` is capped: grid == CU count, persistent kernel, spin-wait counters — **a grid larger than
CU count deadlocks** (`dev.rs:29-31`). So the target is `blocks` as close to 256 as the work allows.

Independent work items in one KDA layer's decode state step = `H × D` **columns**:

| tiling | items | `blocks` @ TP1 | @ TP4 | @ TP8 | VGPR/lane |
|---|---|---|---|---|---|
| one WG per head | 96 | 96 = **37.5%** | 24 = **9.4%** | 12 = **4.7%** | 32 |
| one WG per half-head | 192 | 192 = 75% | 48 = 18.8% | 24 = 9.4% | 16 |
| **column-tiled, `blocks = 256`** | **12288** | **256 = 100%** | **256 = 100%** | **256 = 100%** | 4–8 |

The **9.4%** figure in the audit is one-workgroup-per-head at **TP4** (96/4 = 24 heads/GPU,
24 of 256 CUs) — worse than the `MlaMergeFold` defect that cost 8.69 ms. Head-parallelism alone
*is* the pathology, and it gets worse exactly where K3 has to run, because TP divides the head
count.

Column tiling removes it at every TP degree: even at TP8, 12 heads × 128 columns = **1536 columns**,
so `blocks = 256` with 6 columns each. **Never parallelize KDA over heads alone.**

The same check for the other pieces:

| packet | independent items / layer | `blocks` |
|---|---|---|
| `KdaConv` (decode) | 3·H·D = 36864 channels | 256, 144 channels each |
| `KdaGate` | H·D = 12288 elements | 256, 48 each |
| `KdaStateStep` | H·D = 12288 columns | 256, 48 each |
| `GatedRmsNormHead` | H = 96 rows of 128 | 96 (37.5%) — or fold into `KdaStateStep`'s epilogue |

`GatedRmsNormHead` is the one narrow packet. Its work is tiny (96 × 128 elements), so 37.5%
occupancy on a trivial op is acceptable — but it is also the natural thing to fold into
`KdaStateStep`'s epilogue, which is *producer-side* fusion and therefore the direction `dev.rs:138-141`
endorses. **[inferred]** — not measured.

### 7.4 Decode, decomposed into packets

**Do not collapse these.** §6g-KNOBS is unambiguous: `GLM_GROUP=1` **removed 38% of the ops and cost
+2.88 ms**, because collapsing 8 experts on disjoint 32-CU slices into a loop inside one packet
destroys concurrency. *"Op count is NOT the objective function."*

One KDA layer, one token. `→` is a counter gate (`Builder::emit` returns the counter; consumers pass
it back as a dep, `crates/packet/src/devbuild.rs:212`).

| # | packet | op | reads | writes | gated on |
|---|---|---|---|---|---|
| P0 | pre-norm | `RmsNorm` *(exists)* | `hidden`, `ln_w` | `x[7168]` | prev layer |
| P1 | q proj | `Gemv` *(exists)* | `x`, `W_q` | `q̃[12288]` | P0 |
| P2 | k proj | `Gemv` | `x`, `W_k` | `k̃[12288]` | P0 |
| P3 | v proj | `Gemv` | `x`, `W_v` | `ṽ[12288]` | P0 |
| P4 | output gate | `Gemv` | `x`, `W_g` | `ĝ[12288]` | P0 |
| P5 | forget-gate down | `Gemv` | `x`, `W_fa` | `r[128]` | P0 |
| P6 | beta logits | `Gemv` | `x`, `W_b` | `β̃[96]` | P0 |
| P7 | forget-gate up | `Gemv` | `r`, `W_fb` | `g̃[12288]` | P5 |
| P8 | short conv | **`KdaConv`** | `q̃,k̃,ṽ`, `conv_w`, `conv_state` | `q,k,v`; `conv_state′` | P1,P2,P3 |
| P9 | gate + beta | **`KdaGate`** | `g̃`, `β̃`, `A_log`, `dt_bias` | `g[12288]`f32, `β[96]`f32 | P6,P7 |
| P10 | state step | **`KdaStateStep`** | `q,k,v,g,β`, `state` | `o[12288]`; `state′` | P8,P9 |
| P11 | gated norm | **`GatedRmsNormHead`** | `o`, `o_norm_w`, `ĝ` | `y[12288]` | P4,P10 |
| P12 | out proj | `Gemv` *(exists)* | `y`, `W_o` | `attn[7168]` | P11 |
| P13 | residual | `Residual` *(exists)* | `hidden`, `attn` | `hidden′` | P12 |

**The concurrency this buys.** P1–P6 are **six independent GEMVs gated only on P0**. They read the
same `x` and write disjoint outputs, so all six are ready at once and the scheduler can overlap them
across 256 CUs. A monolithic `KdaScan` would have serialized them behind one packet — precisely the
`GLM_GROUP=1` mistake.

P8 and P9 are also independent of each other (P8 waits on P1–P3, P9 on P6–P7), and P4's output is
not needed until P11, so the output-gate GEMV has the whole conv+gate+state chain to hide under.

**Dependency granularity: coarse, and this is provable, not a guess.** `crates/packet/src/devbuild.rs:278-300`
documents the rule and cites `lean-plow/Plow/CounterGranularity.lean`:

> `collapse` : if every stage's producer map covers the previous stage, and the work is UNIFORM
> across each stage's slices, then the fine schedule's makespan is *identical* to the coarse one —
> for any producer maps whatsoever. The maps do not matter.
>
> […] fine gates pay **only** when a straggling producer feeds a *cheap* consumer […] That needs the
> consumers to do DIFFERENT amounts of work.

Every KDA head is identical, and every column tile is identical. **The work is uniform across
slices, so `Dep::Fine` provably buys nothing here and should be `Dep::Coarse`** — the same
conclusion the file already records for transformer attention (*"every head is identical ⇒ uniform
⇒ downgraded to coarse"*). Do not spend fine gates on P8/P10.

The one place fine gates could pay is the P0 → {P1..P6} fan-out, where the consumers are wildly
non-uniform: P1–P4 are 88.1 M-MAC GEMVs, P5 is 0.9 M and P6 is 0.7 M. A fine edge would let the two
tiny GEMVs start on slices whose `x` rows are already written. **[inferred]** — this is the
`hetero_can_win` case on paper; unmeasured, and worth little since P5/P6 are off the critical path
anyway.

**Packet-count honesty.** 14 packets × 69 KDA layers = **966 packets/token** for the KDA layers
alone, before MoE. Contract §7 measures decode as per-packet-gating-bound at 1134 packets/token for
GLM. So K3's packet count is a genuine concern — but the answer is **not** to collapse P1–P6.

The distinction that resolves it, and it is the lesson of `GLM_GROUP=1` stated precisely:

- **Merging along the OUTPUT dimension is safe.** `GemvQkv = 22` already exists and does exactly
  this — q/k/v projections in one GEMV. The merged op is *wider*, still spreads across all 256 CUs,
  and loses no concurrency. A `GemvQkvg` (`[49152, 7168]`, four output blocks) is the natural
  extension and would remove 3 packets × 69 layers = **207 packets/token**.
- **Merging along a LOOP dimension is fatal.** That is what `GLM_GROUP=1` did — disjoint CU slices
  became a serial loop inside one packet, +2.88 ms.

So: pursue the `GemvQkv`-style output-dim merge, refuse the loop-dim merge. **This supersedes the
weaker claim in §5.3**, which proposed the merge on op-count grounds alone; the justification is
concurrency-preservation, and per §6g-KNOBS it still has to be *shown*, not assumed.

### 7.5 Prefill — the tile dependency structure

The chunked scan (§5.1) splits cleanly into a **fully parallel phase** and a **short serial chain**,
and the parallel phase is the overwhelming majority of the work.

**Phase A — chunk-local. Every chunk of every head, simultaneously. No state involved.**

| # | work | parallel over | serial within |
|---|---|---|---|
| A1 | short conv + SiLU over all `T` | `T × 3·H·D` | — (4-tap causal stencil) |
| A2 | gate + β (`KdaGate`, §2.2 steps 5–6) | `T × H × D` | — |
| A3 | `G = cumsum(g)` inside each chunk | `NC × H × D` channels | 64 (a length-64 scan) |
| A4 | build `A[64,64]`, invert by forward substitution | `NC × H` | 64 rows |
| A5 | `w = A@(e^G·k)`, `u = A@v` | `NC × H` | — (dense GEMMs) |

`NC = ⌈T/BT⌉`. **None of A1–A5 touches the recurrent state**, so for `T = 4096` and `BT = 64` that is
`64 × 96 = 6144` independent (chunk, head) work items — saturating 256 CUs by a factor of 24, and
completely overlappable with the serial phase of *earlier* chunks.

**Phase B — the state chain. Serial over chunks, parallel over (head, v-column) within a chunk.**

For chunk `c = 0 … NC−1`, per head:

| # | work | shape |
|---|---|---|
| B1 | `v′ = u − w·S` | `[64,128] @ [128,128]` |
| B2 | `o = (q·e^G)·S + tril(A_qk,−1)·v′` | `[64,128] @ [128,128]` + `[64,64] @ [64,128]` |
| B3 | `S ← diag(e^{G_last})·S + (e^{G_last−G}·k)ᵀ·v′` | `[128,64] @ [64,128]` |

**The gates, precisely:**

```
A1 → A2 → A3 → A4 → A5          (coarse, once, all chunks together)
A5 ─────────────────┐
                    ├──→ B(c)  for every c          (coarse; A is chunk-local, so it is ready early)
B3(c−1) ────────────┘
B(c) → B(c+1)                    ONE coarse counter edge per chunk — this is the only serial edge
```

- **Parallel:** everything in phase A; and within a `B(c)`, all 96 heads and all column tiles.
- **Serial:** `B(c) → B(c+1)`, depth `NC`. Nothing else.
- **Per-head independence:** the chain is per `(layer, head)` — 96 chains run concurrently, and with
  `cu_seqlens`, per-sequence chains too. This is the same shape as the two-shot all-reduce's
  rendezvous: a narrow ordered edge with wide parallel work hanging off it.
- **Granularity:** uniform work per head ⇒ `Dep::Coarse` again, by the same `collapse` argument.

**Occupancy in phase B**, the thing to watch: 96 heads × 128 columns = 12288 columns, so
`blocks = 256` with column tiling — 100%, and still 100% at TP8 (§7.3). Head-only tiling gives 96 /
24 / 12 WGs and must not be used.

**Serial depth is the prefill risk, and `BT` is the knob.** `NC = T/BT` coarse gates per layer:
`T = 4096, BT = 64` → **64 gates × 69 layers = 4416 serial rendezvous** for the KDA layers of one
prefill. Doubling `BT` to 128 halves the chain but makes A4's forward substitution 4× the work
(`O(BT²)` rows × `O(BT)` each) — and A4 is in the *parallel* phase, where work is cheap and gates
are not. **[inferred]: `BT` should be tuned upward from `[fla]`'s 64 for plow, because plow's cost
model prices a serial gate much higher than `[fla]`'s does.** Unmeasured, and it interacts with
§9 item 1 (`BT` is not pinned by the model).

### 7.6 What genuinely needs a new kernel — and `Mamba2Scan` as precedent

Four new opcodes, sized so that no single one is a mega-kernel. Free values are **88, 89** and
**102+** (`83`–`87` were taken by the MLA MoE-prefill band after the "83-89 gap" comment was
written; `COUNT = 102`).

```rust
    /// KDA short conv: causal depthwise width-`i2` conv + SiLU over concatenated q|k|v,
    /// carrying `t3` across steps. Prefill when `i0 > 1`, single-step when `i0 == 1`.
    ///   t0=out([T,3*H*D] bf16, post-activation)  t1=x([T,3*H*D] bf16, pre-conv)
    ///   t2=w([3*H*D,W] f32)  t3=conv_state(f32[3*H*D,W], in/out)  t4=slot_idx(u32[B])?
    ///   i0=T i1=conv_dim(3*H*D) i2=W i3=act(1=silu) i4=B.
    KdaConv = 88,

    /// KDA gate pre-pass, elementwise. `g = lb*sigmoid(exp(A_log[h])*(g_raw + dt_bias[h,d]))`
    /// when `i3==1`, else `-exp(A_log[h])*softplus(g_raw + dt_bias[h,d])`; `beta = sigmoid(beta_raw)`.
    ///   t0=g([T,H,D] f32)  t1=beta([T,H] f32)  t2=g_raw([T,H*D] bf16)  t3=beta_raw([T,H] bf16)
    ///   t4=A_log(f32[H])  t5=dt_bias(f32[H*D])
    ///   i0=T i1=H i2=D i3=gate_mode   f0=lower_bound.
    KdaGate = 89,

    /// KDA single-step gated delta-rule state update (decode). Read-modify-write on `t6`.
    /// Per (h,v) column: S'=exp(g)*S; u=v-S'^T k; o=S'^T q + beta*(q.k)*u; S=S'+beta*k*u^T.
    /// State is V-FIRST `[h][v][k]`, f32. `i3` is the column tile; it sets VGPR/lane = i3*D/512.
    ///   t0=o([T,H,D] bf16)  t1=q  t2=k  t3=v ([T,H,D] bf16)  t4=g([T,H,D] f32)
    ///   t5=beta([T,H] f32)  t6=state(f32[H,D,D], in/out)  t7=slot_idx(u32[B])
    ///   i0=T i1=H i2=D i3=BV i4=flags i5=B   f0=scale(D^-0.5).
    /// `flags` bit0 = l2norm q/k in kernel. `j[0]` MUST be 0 (shares a wire slot with f0's pair).
    KdaStateStep = 102,

    /// KDA chunked scan (prefill). One chunk of `i3` tokens per invocation, or a chunk loop when
    /// `i6==1`. Phase-A products `w`,`u`,`A_qk`,`G` come in via `t5` (chunk-local, precomputed).
    ///   t0=o([T,H,D] bf16)  t1=q  t2=k ([T,H,D] bf16)  t3=v  t4=beta([T,H] f32)
    ///   t5=chunkaux(f32: G | w | u | A_qk)  t6=state(f32[H,D,D], in/out)  t7=cu_seqlens(u32[N+1])
    ///   i0=T i1=H i2=D i3=BT i4=flags i5=N i6=loop_chunks   f0=scale.
    KdaChunkScan = 103,
```

**Note what the decomposition bought.** The monolithic design needed Mamba's `params`-packing hack —
`A_log | dt_bias | o_norm` crammed into one f32 handle — because 8 tensor slots ran out. Decomposed,
`KdaGate` takes `A_log` and `dt_bias` as **their own handles** and no op is at the slot ceiling
except `KdaStateStep`. **The 8-slot pressure was a symptom of over-fusion, not a constraint on KDA.**
The `A_log[:96]` slice (§3.2) now happens in the ordinary loader against its own declared handle,
not inside an offline packing step.

`GatedRmsNormHead` is deliberately **not** listed: check first whether `RmsNorm = 1` can normalize
over an inner axis of 128 with a `[128]` weight. If it can, the output gate is
`RmsNorm` + an elementwise sigmoid-multiply — and if there is no elementwise multiply (there is not,
§7.1), fold it into `KdaStateStep`'s epilogue rather than adding a fifth opcode (§7.3).

**`Mamba2Scan = 90`: cautionary tale, not a starting point.** It is the only precedent, and every
one of its structural choices is one this design should reject:

| `Mamba2Scan` | KDA should |
|---|---|
| **`blocks == 1`** — one CU, 0.4% occupancy (`crates/devgen/src/mla.rs:3981`) | `blocks = 256` via column tiling (§7.3) |
| Monolithic: conv1d + SiLU + scan + D skip + gated RMSNorm in one op | four ops, six concurrent producer GEMVs (§7.4) |
| All 8 tensor slots + all 8 int slots consumed; scalars packed into `t5` | slots to spare on three of four ops |
| **No arm in `interp.hip`** → AMD dispatch `default:` silently computes nothing | AMD arm + `GFX950_DISPATCHED` from day one (§7.8) |
| *"has ONLY been verified to nvcc-COMPILE … never executed on a GPU. DO NOT treat it as validated"* (`runtime/nvidia/op_mamba.cuh:4-11`) | oracle parity against `[fla]` `naive.py` (§7.8) |
| No weight loader — synthetic handles, never bound, `validate_coverage` never called | real `language_model.…` names + coverage gate (§7.7) |

What *is* worth taking from it: its header states the structural insight in the same words this
design uses — *"the scan itself is embarrassingly parallel across (head, channel) pairs and
sequential over T within each pair, so NO cross-thread reduction is needed."* Whoever wrote it saw
the right decomposition and then emitted it onto one CU anyway. The lesson is that **seeing the
parallelism is not the same as spending it**, which is why §7.3 exists.

Its operand layout is a reasonable *template* for `KdaChunkScan` (the one op that legitimately
carries state and aux tensors together) and a bad one for everything else.

### 7.7 The loader — three traps, all outside `dev.rs`

**(a) The name prefix does not match the binder.** plow has *no* HF-name translation table: the
packet tensor name **is** the safetensors key, and `crates/plowrt/src/exec/gpu.rs:726-737` binds
anything starting with `model.` or `fp8/` by direct index lookup. K3's keys start with
**`language_model.model.layers.…`** because the text tower is nested under a multimodal wrapper
(`modeling_kimi_k3.py:919`, `self.language_model = KimiLinearForCausalLM(config.text_config)`).

Counted over the full index `[ckpt]`, the top-level prefixes of all 497 220 tensors are:

```
language_model : 497052
vision_tower   :    165
mm_projector   :      3
model          :      0
```

**Not one K3 tensor passes the `model.` prefix test.** Either the binder learns
`language_model.`, or the loader strips it before declaring handles. Today it silently binds
nothing — and "silently" is the problem, because `gpu.rs` only asserts byte-size on names it
*did* match.

**(b) The Mamba precedent has no weight loader at all — do not copy it here.**
`emit_nemotron_mamba` declares synthetic handles (`mamba.{l}.conv1d.w`, `mamba.{l}.ssm_params`, …)
that do not start with `model.`, so `gpu.rs:726` never binds them; `nemotron_emit_block` never calls
`validate_coverage`; it has only ever run against a synthetic zero-filled tensor table. Use it as
the **structural** template (`crates/devgen/src/mla.rs:3777-4410`) and the dense/MLA loaders
(`crates/devgen/src/lib.rs:725-796`, `crates/devgen/src/mla.rs:793-960`) as the **binding** template.
`validate_coverage` (`crates/devgen/src/checkpoint.rs:218-310`) is bidirectional — every declared
name must exist and every checkpoint tensor must be covered — so the 14 tensors of §6.1 must all be
accounted for, including the ones that get folded into `t5`.

**(c) TP sharding is a substring match and defaults to replicate.**
`crates/plowrt/src/asset/shard.rs:59-92` classifies COL/ROW/Replicated by substring on the HF name.
A name it does not recognise is **silently replicated** — which for KDA is not a crash, just wrong
math on >1 GPU. Required classification (matches `[vllm]`'s, `kimi_gdn_linear_attn.py:120-226`):

| tensor | shard |
|---|---|
| `q_proj`, `k_proj`, `v_proj`, `g_proj`, `f_b_proj` | **column** (split `H·D` by head) |
| `o_proj` | **row** |
| `b_proj` | **column** (split `H`) |
| `q_conv1d`, `k_conv1d`, `v_conv1d` | **column** (split `H·D` by head) |
| `A_log` | split dim 0 over `H` — *after* the `[:96]` slice (§3.2) |
| `dt_bias` | split dim 0 of `[H,D]` |
| `f_a_proj` | **replicated** (output is the rank-128 bottleneck, not per-head) |
| `o_norm` | **replicated** (`[D]`, shared by all heads) |
| `state`, `conv_state` | split by head |

`f_a_proj` and `o_norm` being replicated while everything around them is column-parallel is the
easy mistake; `[vllm]` uses `ReplicatedLinear` for `f_a_proj` explicitly.

**(d) Two independent config parsers.** `crates/devgen/src/config.rs:65-85` and
`crates/plowc/src/hf_config.rs:139-159` (`HfArch`) both dispatch on `model_type` and both need a K3
arm. K3's `model_type` is **nested**: `"kimi_k3"` at top level, `"kimi_linear"` under `text_config`,
and all the dims this document uses live in `text_config` (§2.1). Reading the top-level object gets
you a `vision_config` and nothing else useful.

### 7.8 Coverage-check discipline — what §4 of the contract requires

§4 of the design notes names the recurring bug shape:

> **an arm exists, is correct, is register-gated, and nothing routes to it.**
>
> The search pattern: *for each arm, what selects it, and is that selector complete over
> precisions?* `check_gfx950_opcode_coverage` enforces this by comparing the emitted **stream**
> against what the target actually dispatches, with the opcode list parsed out of `interp.hip` by
> a drift test. **If you add an arm, you add it there too, or you have not added it.**

And §3, which applies directly because `KdaStateStep` and `KdaGate` carry mode/`flags` immediates:

> Adding a weight encoding means emitting the field **and** routing a dispatch arm **and** adding
> the opcode to the coverage check.

Concretely, for the four new opcodes of §7.6:

1. **Add the AMD arm or the emit hard-fails.** `check_gfx950_opcode_coverage`
   (`crates/devgen/src/lib.rs:4150-4175`, called at `:4721`) compares the emitted stream against
   `GFX950_DISPATCHED` (`lib.rs:4059-4133`), and the drift test
   `dispatched_list_matches_the_amd_interpreter` (`lib.rs:4945-4988`) parses `runtime/amd/interp.hip`
   for both `case PLOW_DOP_X:` and `in->op == PLOW_DOP_X`. The hard failure is the *desired*
   behaviour, because **AMD's dispatch `default:` is a silent NOP** — `lib.rs:4043-4058`:
   *"An opcode with no arm therefore does not trap, it silently leaves the output buffer
   untouched."* `Mamba2Scan` is NVIDIA-only and **not** in `GFX950_DISPATCHED`; KDA on AMD cannot
   inherit that gap.
2. **Both directions.** `every_dispatched_arm_has_an_emit_site` catches the reverse (arm exists,
   nothing emits it) — the exact §4 bug shape.
3. **`flags` must be routed, not just emitted.** Bits 0–3 select real numerical behaviour
   (§2.2 steps 4, 6; §3.1; §4.1). A bit that is emitted but ignored by the kernel is the §3 bug
   verbatim, and it fails *silently* — the output is still finite and plausibly scaled.

K3-specific correctness gates on top of the contract:

4. **Oracle parity.** `[fla]`'s `naive.py`/`gate.py` are pure-PyTorch references
   (`naive_recurrent_kda`, `naive_chunk_kda`, `naive_kda_gate`, `naive_kda_lowerbound_gate`). They
   run on CPU, need no GPU lease, and `naive_chunk_kda` vs `naive_recurrent_kda` cross-checks the two
   modes against each other. Note there is **no CPU reference dispatch over `DevOp`** in plow
   (`crates/plowrt/src/device/cpu.rs:482` matches on packet `Body`, with a `golden numerics TODO`) —
   so the golden goes next to the emitter as a Rust unit test, exactly like `mamba_ref`
   (`crates/devgen/src/mla.rs:6151-6530`, which also ships a prefill-vs-decode equivalence test).
   **Stock `fla` 0.5.2 (current PyPI release) is sufficient** — its `chunk_kda` accepts every kwarg
   K3 passes, including `safe_gate`, `lower_bound`, and `transpose_state_layout` as the deprecated
   alias; no unreleased build is needed. Run it under `nix develop .#quantize` with
   `PYTHONNOUSERSITE=1` (contract §0a). A second oracle is available in `[k3impl]`
   (`tests/models/kimi_k3/test_kda.py`), which tests the K3 knobs the `fla` references do not
   exercise by default.
5. **Prefill/decode agreement** (§5.2): prefill `T` tokens; then prefill `T−1` and decode 1; require
   the states to match. Most likely test to catch a layout bug, and `mamba_ref` already has the
   equivalent.
6. **Single-layer before full network** (§5 of the contract). One KDA layer against the oracle on
   real layer-0 weights, before any 93-layer launch.
7. **Assert on `A_log.len() == 128` then slice `[:96]`** (§3.2) and on **V-first state** (§4.1).
   These are the two silent-wrong-answer traps: both produce finite, plausibly-scaled garbage.

### 7.9 Scope of the change, by precedent

Adding Mamba2 (`9bd7bc0` + `82f6fc6`) was **1301 insertions, entirely additive**, touching:
`crates/packet/src/dev.rs` (enum + `ALL` + `c_name` + `COUNT`) · `runtime/common/dev_isa.h` ·
a new `runtime/*/op_*.cuh` · the interpreter dispatch arm · `crates/devgen/src/mla.rs` (config parse,
per-layer emit, block descriptor, CPU golden + tests) · `crates/devgen/src/lib.rs` (model_type
dispatch) · `crates/devgen/src/manifest.rs` (feature flag) · `crates/plow-asset/src/lib.rs`
(`BlockDims` + `CarriedState`) · `crates/kernelcaps/{probe,spec}.rs` · `crates/plowc/src/tune.rs`
(`ProfileId`) · `crates/plowrt/examples/block_*.rs` · the cubin build script.

KDA adds, beyond that list: the AMD arm + `GFX950_DISPATCHED` (§7.8 item 1), the loader work of
§7.7, and `crates/plowrt/src/asset/shard.rs` for TP.

Deliberately **not** touched by the Mamba precedent, and the same gap applies here: `crates/rewrite/`
has no `mamba.rs`, so there is **no Path-A / cost-model / scheduler support** for a recurrent mixer.
`OpKind` has only four variants (`Gemm`, `Flash`, `Row`, `Layout`, `crates/rewrite/src/tilegraph.rs:94-100`)
and `kimi.rs`'s module doc states the reuse policy is to add none. A KDA layer maps onto
`Gemm` + `Row` for everything except the scan, which fits no variant. Whether that blocks serving is
**unresolved** (§9 item 9).

---

## 8. Numerics checklist

Things that are easy to get wrong and produce plausible-but-wrong output:

- L2 norm epsilon is **inside** the sqrt: `x / sqrt(Σx² + 1e-6)`, not `x / (‖x‖ + eps)`.
- `q` is scaled by `128^-0.5` **after** L2 normalization, and `k` is **not** scaled.
- Decay is applied **before** the delta term is computed (§2.2 step 7).
- The forget gate indexes `A_log` by **head** and `dt_bias` by **`h*128 + d`**.
- `β` is per head (96 values), *not* per channel.
- Output RMSNorm is over `D = 128` inside a head with a `[128]` weight shared by all heads — not
  over 12288.
- The output gate multiplies **after** the RMSNorm, and `sigmoid` is applied to the raw `g_proj`
  output.
- The state accumulates in fp32 in every reference implementation.
- In the chunk kernel, always form `exp(G_i − G_j)`, never `exp(G_i)/exp(G_j)` (§3.1).
- `[fla]` scales the cumulative gate by `1/ln 2` so downstream kernels can use `exp2` `[vllm]`
  (`fla/ops/kda.py:1251-1253`). Cosmetic, but the intermediate `G` is then in log₂ space — do not
  mix conventions between the gate op and the scan op.

---

## 9. What I could NOT determine

> **Update 2026-07-28 — implemented and measured.** Ops 88/89/102/103 (`KdaConv`, `KdaGate`,
> `KdaStateStep`, `KdaGatedNorm`) now exist with gfx950 arms, an emitter
> (`crates/devgen/src/kda.rs`) and a real-weight numeric gate that **passes** on layer 0
> (`runtime/tests/kda_real_oracle.py` + `kda_block_gfx950_test.c`, `scripts/build_kda_real.sh`).
> Items **3** and **4** below are CLOSED; item 2 is unchanged;
> the state dtype is settled as **f32** from `fla/ops/kda/fused_recurrent.py`'s explicit
> `dtype=torch.float32`, against AMD's contradictory 2-byte formula.
>
> Two corrections to this document, both found by implementing it:
>
> - **§7.2's "no cross-lane reduction anywhere" and its "4–8 VGPRs/lane" are inconsistent.**
>   `BV·D/512` f32 per lane means `512/BV` lanes cooperate on a column, so the reduction over `k`
>   *does* cross lanes. The implementation keeps every number in that table and drops the absolute
>   claim: **one WAVE owns one column** (`D = 128 = 64 lanes × 2`), the state costs **2 f32/lane**,
>   both reductions are `wave_sum`, and nothing crosses a *wave*.
> - **§7.3's "100% at every TP degree" assumes `BV` shrinks with the head count.** At TP8 (12
>   heads) a fixed `BV = 16` gives 96 items = 37.5%, not 100%. `BV = 8` restores 192.
>
> §7.6's `KdaChunkScan` is deliberately **not** declared: `KdaStateStep` takes a serial-`T` loop
> which is exact at any `T`, and an opcode declared before its kernel exists is how `Mamba2Scan`
> became dead code. The chunked form remains the right *performance* answer for prefill.

Stated plainly, because someone will implement 69 layers from this.

1. **Whether the chunk size is 64 for K3.** `BT = 64` is `[fla]`'s default (`naive.py:74`) and the
   value the reference uses, but `chunk_kda` autotunes internally and K3 passes no explicit chunk
   size `[hf]:610-627`. Any `BT` is mathematically equivalent; only performance and the exact fp32
   rounding differ. **If bit-exactness against a reference run is ever required, `BT` must be
   pinned.**

2. **Whether the recurrent state may be bf16 without quality loss.** `[vllm]` hardcodes fp32
   (`mamba_utils.py:119-125`); `[fla]` accumulates fp32. Dropping to bf16 halves state memory
   (452.8 → 245.8 MiB/seq) and would matter at high concurrency. **Not measured. Do not assume.**

3. **Why `A_log` is zero-padded from 96 to 128.** The padding is verified and unambiguous (§3.2);
   its *origin* is not. I found no code that writes it.
   **The coverage half is CLOSED (2026-07-28).** `scripts/kda_verify_ckpt.py` now reads the
   headers of all 93 layers plus the 512 B of `A_log` and the 48 KiB of `dt_bias` per KDA layer
   out of the now-complete 96-shard checkpoint, and confirms in **69/69** KDA layers: `A_log` is
   `[128]` f32 with non-zero indices exactly `0..95`; `dt_bias` is `[12288]` f32 with zero zeros;
   all 14 tensors of §6.1 carry the dtypes and shapes stated there; the tensor-implied layer
   classification matches the 1-based config list for all 93 layers; and the tail is `KKK MM`,
   with `i % 4 == 3` disagreeing at 0-based layer **92**. Only the *origin* of the padding is
   still unknown, and it no longer matters.

4. **Whether the column tile `BV` fits the decode object's register headroom. CLOSED — it does,
   at zero cost.** Measured on gfx950 against an identical build of the unmodified tree
   (`scripts/kda_regcheck.sh`):

   | object | baseline | + the four KDA arms |
   |---|---|---|
   | decode | VGPR 248, occ 2, VGPR-spill 0, SGPR-spill 80 | **VGPR 248, occ 2, VGPR-spill 0**, SGPR-spill 82 |
   | prefill | VGPR 256, occ 2, VGPR-spill 2, SGPR-spill 85 | **VGPR 256, occ 2, VGPR-spill 2**, SGPR-spill 84 |

   Zero VGPRs, zero occupancy change, zero new VGPR spill. The sibling audit's "32 VGPRs, does not
   fit, needs a 4th co-resident code object" is confirmed to be an artifact of
   one-workgroup-per-head: at **one wave per column** the state costs 2 f32/lane and the arms fit
   inside the existing decode object. The co-resident object is not needed for KDA.

5. **`BT` for prefill, and the serial-chain cost.** §7.5 argues `BT` should be tuned *upward* from
   `[fla]`'s 64 because plow prices a serial gate higher, and the extra `O(BT³)` forward-substitution
   work lands in the fully-parallel phase. That is reasoning, not measurement: I did not price a
   coarse gate against A4's work at any `BT`. It also interacts with item 1.

6. **Whether `GatedRmsNormHead` needs to exist.** §7.6 defers it pending a check of whether
   `RmsNorm = 1` can normalize over an inner axis of 128 with a `[128]` weight. I did not read the
   kernel to find out. If it cannot, the choice is a fifth opcode (narrow, 96 workgroups = 37.5%) or
   an epilogue fold into `KdaStateStep` — unmeasured either way.

7. **The `situ` activation, AttnRes (`attn_res_block_size: 12`), and LatentMoE. CLOSED for
   DECODE (2026-07-28)**. Two opcodes (`AttnRes = 104`,
   `SituGlu = 105`), `PLOW_MOE_ACT_SITU = 2` inside the routed-expert GLU, and one kernel line
   (`d_moe_combine`'s residual is now optional, which is what a latent-width combine needs).
   **A complete K3 block now runs end to end against a real-weight oracle at both rungs**: layer 0
   (KDA + AttnRes + dense situ FFN) at T=1 and T=4, and layer 1 (both AttnRes applications + KDA +
   Stable LatentMoE, 896 experts / top-16 / mxfp4 / latent 3584) at T=1, every stage inside the B4
   tolerances and the routing table exact. Register cost across all four interpreter objects: zero.

   This note's original text was right that the block structure is not `residual + attn`. What it
   could not have known is **where that becomes visible**: only at the 8 SNAPSHOT layers
   (`l % 12 == 0`) does a plain wiring differ at the block output (rel 1.0). At the other 85 layers
   `prefix = prefix_in + attn`, so the output is `prefix_in + attn + ffn` either way — measured at
   **3.0e-3** on real layer-1 weights, against 8.1e-1 at the AttnRes outputs themselves. **A gate
   must diff the sub-layer inputs, not the block output.**

   Still open, and it is rung 3: Gated MLA (output gate, NoPE-as-identity-rotation, the `V=128`
   `d_mla_merge_fold` VT dispatch) and prefill for both `situ` and LatentMoE.

8. **Multi-token / speculative decode.** `[fla]`'s decode kernel takes `num_accepted_tokens` and
   supports spec-decode state rollback (`fused_recurrent.py:124-135`). Rolling *back* a recurrent
   state is not possible without a snapshot, so spec decode over KDA needs either a saved state per
   speculative step or re-running the accepted prefix. I did not work out which `[fla]` does.
   `num_nextn_predict_layers: 0` `[cfg]`, so K3 ships no MTP head and this is not on the critical
   path today.

9. **Whether the missing Path-A support blocks serving.** `crates/rewrite/` has no module for a
   recurrent mixer and `OpKind` has no variant a scan fits (§7.9). The Mamba work shipped without
   it and the commit says the CPU wire-op was skipped deliberately ("would ripple across
   `NetConfig`/packet/scheduler"). Whether the bucket/cost-model path is *required* for `plowrt
   serve` — which is what §0-BENCH measures — or only for the tile-IR path, I did not establish.
   **This is the most likely thing to turn a working kernel into a non-servable model, and it
   should be checked before any kernel work starts.**
