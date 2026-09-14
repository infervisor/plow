# DeepSeek-V4-Flash on MI300X

Architecture, capacity, attention roofline and kernel bringup for
`DeepSeek-V4-Flash-0731`, which plow can compile pieces of but cannot yet
serve. Merged from four separate notes.


---

## DeepSeek-V4-Flash-0731 — implementation-grade architecture spec

Derived **only** from the shipped reference implementation and the shipped checkpoint. Every
structural claim carries a `file:line`. Where the served `config.json` and the reference code
disagree, the disagreement is stated and the code wins.

**Reference tree** (all `model.py` / `kernel.py` / `convert.py` / `generate.py` citations are
relative to it):

```
/workspace/models/DeepSeek-V4-Flash-0731/
├── config.json                       served HF config
├── model.safetensors.index.json      72 317 tensors, 166 878 536 440 B
├── inference/{model.py,kernel.py,convert.py,generate.py,config.json}
├── encoding/encoding_dsv4.py
└── README.md
```

**plow tree**: `/app/plow` (this worktree). plow citations are repo-relative.

> **Update, 2026-09-08.** §7's gap table is still accurate as an analysis, but five of its seven
> "needs a new kernel body" rows now HAVE one, hardware-tested on gfx942:
> rows **10** (inverse RoPE on O), **18** and **19** (the compressor, both the ratio-128 form and
> the overlapped ratio-4 one), **21** (the indexer score at 64 heads over V4's fake-quantised
> operands) and **29** (hash routing). Rows **32** (w4a8 experts) and **39** (DSpark, optional)
> do not. Several **A** rows are also done — 17 (the learned mHC exit), 26 and 28 (sqrtsoftplus
> and an f32 router logit), 30 (the clamped SwiGLU), and the DR=0 MLA prefill rows 5/8 depend on.
> What each kernel does, what it was verified against, which negative controls prove the test
> bites, and what is still missing on the EMITTER's side:
> **this document**.

No performance claim appears in this document. A separate agent owns the roofline.

---

## 0. Headline

| | |
|---|---|
| Total params (fp4 experts counted as logical elements) | **304.18 B** (main tower 284.33 B, DSpark 19.85 B) |
| Activated params / token (main tower, incl. `lm_head`) | **13.27 B** |
| Checkpoint on disk | **166.88 GB** (88.8 % packed-fp4 expert bytes) |
| Distinct layer kinds in the 43-layer stack | **4** (+3 DSpark stages) |
| KV bytes / token, whole tower, reference bf16 cache | **6 880 B** |
| KV bytes / token, packed to the precision the reference actually stores | **≈ 3 509 B** |
| DeepSeek-V3 for scale (61 × 576 × bf16) | 70 272 B / token |
| Gap-analysis verdicts (40 rows) | **15** reusable as-is · **18** need a new arm · **7** need a new kernel body |

The architecture's whole point is the last three rows: the full-resolution KV cache is a
**fixed 128-entry ring per layer** and never grows. Everything older than 128 tokens is
reachable only through learned-pooled compressed entries.

---

## 1. Per-layer dataflow

### 1.0 Global geometry and notation

From `inference/config.json` and `config.json` (they agree on all of these):

| symbol | value | source |
|---|---|---|
| `H` hidden | 4096 | `config.json:"hidden_size"` |
| `L` layers | 43 | `config.json:"num_hidden_layers"` |
| `NH` heads | 64 | `config.json:"num_attention_heads"` |
| `D` head_dim | 512 | `config.json:"head_dim"` |
| `DR` rope dims | 64 | `config.json:"qk_rope_head_dim"` |
| `DN` = `D-DR` | 448 | `model.py:452` |
| `QL` q_lora_rank | 1024 | `config.json:"q_lora_rank"` |
| `OL` o_lora_rank | 1024 | `config.json:"o_lora_rank"` |
| `G` o_groups | 8 | `config.json:"o_groups"` |
| `W` window | 128 | `config.json:"sliding_window"` = `ModelArgs.window_size`, `model.py:65` |
| `HC` hc_mult | 4 | `config.json:"hc_mult"` |
| `E` routed experts | 256, top-6, 1 shared, `I`=2048 | `config.json` |
| vocab | 129 280 | `config.json:"vocab_size"` |

`num_key_value_heads: 1` in `config.json` is accurate but understated: there is **one 512-wide
latent KV vector per token, shared by all 64 query heads, and it is simultaneously K and V**.
`kernel.py:322` (`T.gemm(q_shared, kv_shared, acc_s, transpose_B=True)`) and `kernel.py:344`
(`T.gemm(acc_s_cast, kv_shared, acc_o)`) both read the same `kv_shared[block, d]` buffer with
`d = D = 512`. There is no `kv_b_proj`, no `v_head_dim`, and no absorb step: **the checkpoint
ships MLA already in absorbed form.** `W_UK` is folded into `wq_b` (which projects
`QL → NH·D` directly) and `W_UV ∘ W_O` is factored into `wo_a`/`wo_b` (§1.5).

### 1.1 The 43-layer stack

`compress_ratios` (`config.json`) has **46** entries for 43 hidden layers. `Attention.__init__`
reads `args.compress_ratios[layer_id]` at `model.py:459`; `DSparkBlock.__init__` passes
`layer_id = args.n_layers + stage_id` (`model.py:894`), i.e. 43/44/45. So entries 43–45 are the
three DSpark stages and are `0`. Verified against the checkpoint: `layers.{2..42}.attn.compressor.*`
exists (41 layers), `layers.{2,4,…,42}.attn.indexer.*` exists (21 layers).

| kind | layers | count | `compress_ratio` | attention | MoE gate |
|---|---|---|---|---|---|
| **A** | 0, 1 | 2 | 0 | sliding window only | **hash** (`tid2eid`) |
| **B** | 2 | 1 | 4 | window + indexed compressed (CSA) | **hash** |
| **C** | 4, 6, …, 42 | 20 | 4 | window + indexed compressed (CSA) | scored |
| **D** | 3, 5, …, 41 | 20 | 128 | window + *all* compressed (HCA) | scored |
| **S** | `mtp.0/1/2` | 3 | 0 | DSpark block-parallel window | scored |

Hash layers are `layer_id < n_hash_layers = 3` (`model.py:561`); the checkpoint carries
`layers.{0,1,2}.ffn.gate.tid2eid` and `layers.{3..42}.ffn.gate.bias` — 3 and 40 tensors, exactly
complementary.

> **Naming.** The reference implementation never uses the terms "CSA" or "HCA". They come from
> plow's own refusal string at `crates/nn-graph/src/models/config/mod.rs:181`. In this document
> **CSA** = the ratio-4 kind (compressed + *selected* + sliding) and **HCA** = the ratio-128 kind
> (compressed + *exhaustive* + sliding). Both are the same `sparse_attn` call over a
> concatenated index list; only how the compressed half of that list is produced differs.

### 1.2 Block skeleton (identical for every kind)

`Block.forward`, `model.py:695-707`. `x` is the **4-stream** residual `[B,S,4,H]`.

```
residual = x                                            # [B,S,4,4096]
x, post, comb = hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)   # -> [B,S,4096]  (§3)
x = attn_norm(x)                                        # RMSNorm(4096), learned gain
x = attn(x, start_pos)                                  # -> [B,S,4096]  (§1.3-1.5)
x = hc_post(x, residual, post, comb)                    # -> [B,S,4,4096]

residual = x
x, post, comb = hc_pre(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
x = ffn_norm(x)                                         # RMSNorm(4096)
x = ffn(x, input_ids)                                   # MoE (§4)
x = hc_post(x, residual, post, comb)                    # -> [B,S,4,4096]
```

Model entry/exit, `Transformer.forward`, `model.py:913-927`:

```
h = embed(input_ids)                     # [B,S,4096]
h = h.unsqueeze(2).repeat(1,1,4,1)       # [B,S,4,4096]   plain broadcast, no gate
for i, layer in enumerate(layers): h = layer(h, start_pos, input_ids)
                                         # taps h.mean(dim=2) at i in {40,41,42} -> DSpark (§5)
h = hc_head(h, hc_head_fn, hc_head_scale, hc_head_base)  # 4 -> 1, LEARNED gate (§3.3)
logits = head(norm(h))                   # RMSNorm(4096) then [129280,4096] in fp32
```

`ParallelHead.forward` (`model.py:731-741`) slices `x[:, -1]` unless `full_logits=True`: base
decoding scores only the last position.

### 1.3 The query path (all kinds, including DSpark)

`model.py:502-506`:

```
qr = q = q_norm(wq_a(x))                 # wq_a: [4096 -> 1024] fp8; q_norm: RMSNorm(1024)
                                         # qr is ALSO the indexer's input (§2) — one projection, two consumers
q = wq_b(q).unflatten(-1, (64, 512))     # wq_b: [1024 -> 32768] fp8   -> [B,S,64,512]
q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)     # model.py:504
                                         # UNWEIGHTED per-(token,head) RMS over all 512 dims, no learned gain,
                                         # applied BEFORE rope, and it normalises the rope dims too
apply_rotary_emb(q[..., -64:], freqs_cis)               # model.py:505, interleaved (GPT-J) pairs
```

`softmax_scale = D**-0.5 = 512**-0.5` exactly (`model.py:470`). **No YaRN `mscale`.**
DeepSeek-V3 multiplies its softmax scale by `0.1*ln(factor)+1`; V4 does not. This is a real
behavioural difference, not an omission in the reference — `sparse_attn` receives
`self.softmax_scale` unmodified at `model.py:533` and `model.py:538`.

### 1.4 The KV path and the two index streams

**Full-resolution ("window") KV**, `model.py:508-512`:

```
kv = wkv(x)                              # [4096 -> 512] fp8; ONE vector per token, K and V both
kv = kv_norm(kv)                         # RMSNorm(512), learned gain
apply_rotary_emb(kv[..., -64:], freqs_cis)
act_quant(kv[..., :-64], 64, "ue8m0", e8m0, inplace=True)   # model.py:511
```

`model.py:511` is a **fake quant**: `act_quant(..., inplace=True)` quantises to e4m3 with a
power-of-2 scale per 64-element block and multiplies straight back
(`kernel.py:84-91`), storing the round-tripped value in bf16. The first 448 dims of every cached
KV vector therefore carry only fp8 e4m3 information; the 64 rope dims stay full bf16
("rope dims stay bf16 for positional precision", `model.py:510`). This is QAT simulation and it
is the single most consequential fact for cache sizing (§1.7).

**Window index list**, `get_window_topk_idxs`, `model.py:261-271`:

* prefill (`start_pos == 0`): `matrix[i] = clamp(i-127, 0) + arange(min(S,128))`, entries `> i`
  set to `-1`. Absolute positions into the `[B,S,512]` prefill `kv` tensor.
* decode: a rotation of `arange(128)` into the **ring buffer**. `sp = start_pos % 128`;
  `matrix = cat(arange(sp+1,128), arange(0,sp+1))` — oldest → newest, 128 entries, the last of
  which is the token just written. When `start_pos < 127` the ring is not yet full and the tail
  is `-1`-padded.

`-1` is the universal mask sentinel: `kernel.py:325-329` zeroes the gathered KV row and sets the
score to `-inf`.

**Compressed index list.** Only for `compress_ratio != 0`. `offset = kv.size(1)` in prefill
(= `S`, evaluated at `model.py:513` *before* the concat at `model.py:531`) and `W = 128` in decode
(the compressed region of the cache starts right after the window ring).

* **HCA (ratio 128)** — `get_compress_topk_idxs`, `model.py:275-282`. No selection at all:
  every completed compressed block is attended. Prefill: block `j` is visible to query `i`
  iff `j < (i+1)//128`. Decode: `arange(0, (start_pos+1)//128) + 128`.
  Column count grows without bound: `128 + floor((p+1)/128)` attended entries at position `p`;
  **8 320 at `p = 2**20 - 1`.**
* **CSA (ratio 4)** — `Indexer.forward`, `model.py:408-440`, returns the top
  `min(512, end_pos//4)` compressed-block ids. Attended entries: `128 + min(512, (p+1)//4)`,
  **capped at 640.**

Coverage differs qualitatively, and this is the design:

* HCA layers see the **entire** context — the last 128 tokens at full resolution and every
  earlier 128-token block at 1/128 resolution, with no gaps and no selection.
* CSA layers see the last 128 tokens at full resolution plus **at most 2 048 tokens' worth** of
  4-token pools, chosen by the indexer. Everything else is dropped.

The two streams are concatenated into one index vector at `model.py:519` and served by **one**
`sparse_attn` call (`model.py:533` prefill / `model.py:538` decode). There is no separate
"local + global then merge" — it is a single online softmax over the union.

### 1.5 The attention kernel and the output projection

`sparse_attn`, `kernel.py:277-353`. Per `(batch, query)`: gather `topk` rows of `kv` by index,
FlashAttention-style online softmax over `block=64`-wide tiles, `scale = softmax_scale`, and

```
sum_exp[h] += exp(attn_sink[h] - scores_max[h])        # kernel.py:346
acc_o[h,d] /= sum_exp[h]                                # kernel.py:348
```

**Attention sink**: one learned f32 logit per head (`attn_sink`, `[64]`, `model.py:462`) added to
the softmax **denominator only**, contributing no value row. Note the reference does *not* fold
the sink into the running max — `scores_max` is computed over the real logits only — so a
sufficiently large sink is a numeric hazard in the reference itself. plow's own sink handling
(`crates/packet/src/dev.rs:180`, `gm' = max(gm, sink_h)`) is the safer form and is *not*
bit-identical to this.

Then, `model.py:539-547`:

```
apply_rotary_emb(o[..., -64:], freqs_cis, inverse=True)   # de-rotate by the QUERY's position
o = o.view(B, S, 8, 4096)                                 # G=8 groups of 8 heads x 512
wo_a = wo_a.weight.view(8, 1024, 4096)                    # (group, rank, in)
o = einsum("bsgd,grd->bsgr", o, wo_a)                     # BLOCK-DIAGONAL: 8 independent 4096->1024
x = wo_b(o.flatten(2))                                    # [B,S,8192] -> [B,S,4096], fp8
```

The inverse rope at `model.py:539` is structural, not cosmetic. `o = Σ_j p_j · kv_j` mixes KV
vectors whose rope dims are rotated by their *own* positions; de-rotating by the query position
`i` leaves `Σ_j p_j R(j-i)·kv_j^rope`, i.e. a position-independent latent that a fixed `wo_a` can
consume. Any implementation that skips it is wrong.

**`o_lora_rank` / `o_groups` versus V3's `o_proj`.** V3 has one dense `[NH·v_head_dim → H]`
matrix. V4 replaces it with:

| | shape | params | structure |
|---|---|---|---|
| V3-equivalent dense | `[4096, 32768]` | 134.2 M | full |
| `wo_a` | `[8192, 4096]` viewed `(8, 1024, 4096)` | 33.55 M | **block-diagonal**: group `g` maps *only* heads `8g..8g+7` |
| `wo_b` | `[4096, 8192]` | 33.55 M | full |
| total | | **67.1 M** | rank-1024-per-group bottleneck |

Two structural consequences for lowering: (a) `wo_a` is not a GEMM against a `[8192,4096]`
matrix — it is 8 independent `[1024,4096]` GEMVs, each consuming a *different* 4096-slice of `o`;
(b) `wo_a` is **declared bf16** (`model.py:468`, `dtype=torch.bfloat16`) but **shipped fp8 e4m3**
with a `[64,32]` e8m0 block scale, so `convert.py:123-127` dequantises it to bf16 at load time.
`model.py:544-545` notes an fp8 einsum is possible and was not done "for simplicity".

### 1.6 The Compressor — what "compressed" actually means

`Compressor`, `model.py:285-384`. One instance per compressing layer, plus a second, narrower one
inside each indexer (§2). `coff = 1 + (ratio == 4)` — the ratio-4 form is *overlapped* and needs
two projections; the ratio-128 form is not.

Weights: `wkv: [H → coff·d]`, `wgate: [H → coff·d]`, `ape: [ratio, coff·d]`, `norm: RMSNorm(d)`,
where `d = 512` for the attention compressor and `d = 128` for the indexer's.

```
x  = x.float()                                     # model.py:328 "compression need fp32"
kv    = wkv(x)                                     # the pooled VALUE
score = wgate(x) + ape[position_within_block]       # the pooled GATE
kv_out = (kv * score.softmax(dim=<slot axis>)).sum(dim=<slot axis>)
kv_out = norm(kv_out.to(bf16))                      # model.py:369
apply_rotary_emb(kv_out[..., -rd:], freqs_cis_of_block_first_token)
<fake-quantise>                                     # model.py:374-378
```

The softmax is **per feature channel**, over the slot axis — 512 (or 128) independent softmaxes
per block, not one shared attention weight. `ape` is an additive, learned, position-*within*-block
bias on the gate logits.

**Rope position of a compressed entry.** Prefill uses `freqs_cis[:cutoff:ratio]`
(`model.py:371`) — block `j` gets the rope of absolute position `j·ratio`, its **first** token.
Decode uses `freqs_cis[start_pos + 1 - ratio]` (`model.py:372`), which at the compress boundary
`start_pos = (j+1)·ratio - 1` is the same `j·ratio`. Consistent.

**Fake-quantisation of the compressed entry** (`model.py:374-378`):

* attention compressor (`rotate=False`): `act_quant(kv[..., :-64], 64, "ue8m0", e8m0, inplace=True)`
  — identical treatment to the window KV.
* indexer compressor (`rotate=True`): `rotate_activation(kv)` (fast Hadamard over all 128 dims,
  scale `128**-0.5`, `model.py:253-257`) then `fp4_act_quant(kv, 32, inplace=True)` — e2m1 with a
  power-of-2 per-32-block scale, dequantised back to bf16 (`kernel.py:160-167`).

**Overlap, the ratio-4 form** (`overlap_transform`, `model.py:313-320`). `wkv`/`wgate` emit
`2·512 = 1024`; the first 512 are the "overlap" projection, the second 512 the "normal" one
(`model.py:302`). Each compressed entry pools over **8 slots**:

```
slots 0..3 : the PREVIOUS block's 4 tokens, through the OVERLAP half   (wkv rows   0..511)
slots 4..7 : the CURRENT  block's 4 tokens, through the NORMAL  half   (wkv rows 512..1023)
```

Slots 0..3 of block 0 are filled with `0` (value) and `-inf` (score) so they contribute nothing.
Each source token therefore contributes twice — once through each projection — to two adjacent
compressed entries. The decode path (`model.py:344-354`) reproduces this with an 8-row
`kv_state`/`score_state` ring: the current block accumulates into rows 4..7, and on the boundary
step rows 4..7 are copied down to rows 0..3.

`should_compress` is `(start_pos + 1) % ratio == 0` in decode (`model.py:350`) and
`seqlen >= ratio` in prefill (`model.py:332`); the entry lands at `kv_cache[start_pos // ratio]`
(`model.py:381`), which for the boundary step is exactly block index `j`.
`score_state` is initialised to `-inf` (`model.py:310`) so a partly-warm ring is self-masking.

### 1.7 KV-cache layout and bytes per token per layer

`Attention.__init__`, `model.py:479-480`:

```
kv_cache_size = window_size + (max_seq_len // compress_ratio if compress_ratio else 0)
kv_cache      = zeros(max_batch, kv_cache_size, 512)      # bf16 (generate.py:77 sets the default dtype)
compressor.kv_cache = self.kv_cache[:, 128:]              # model.py:497 — one buffer, two regions
```

`Indexer` owns a second cache, `zeros(max_batch, max_seq_len // 4, 128)` bf16 (`model.py:405`).

```
| 0 .. 127 |  128 .. 128 + S/ratio - 1                 |
| ring, FIXED, 128 entries, slot = pos % 128            |  compressed, GROWS, slot = pos // ratio
```

The window region is a **ring**: `kv_cache[:, start_pos % 128] = kv` (`model.py:535`). Full-
resolution KV older than 128 tokens is **discarded**. Prefill writes the ring with the same
modular placement (`model.py:526-528`: `cutoff = S % 128`, `kv[:, -128:]` split into slots
`[cutoff,128)` then `[0,cutoff)` — verify: position `S-128+j` has `(S-128+j) % 128 = (cutoff+j) % 128`).

**Marginal bytes per token per layer**, reference bf16 cache:

| kind | window (fixed) | attn compressed | indexer compressed | marginal B/tok/layer |
|---|---|---|---|---|
| A (ratio 0), 2 layers | 131 072 B | — | — | **0** |
| B+C (ratio 4), 21 layers | 131 072 B | 1024 B / 4 tok = 256 | 256 B / 4 tok = 64 | **320** |
| D (ratio 128), 20 layers | 131 072 B | 1024 B / 128 tok = 8 | — | **8** |

**Whole tower: `21·320 + 20·8 + 2·0` = 6 720 + 160 = 6 880 B/token.**
At 2²⁰ tokens: 7.21 GB. Fixed per sequence: `43 · 131 072` = 5.64 MB of window rings, plus
Compressor scratch (`kv_state` + `score_state`, both f32, `model.py:309-310`) —
`20 · 2 · 128·512·4` = 10.49 MB for the ratio-128 layers, `21 · 2 · 8·1024·4` = 1.38 MB for the
ratio-4 attention compressors, `21 · 2 · 8·256·4` = 0.34 MB for the indexer compressors. **≈ 17.8 MB
fixed per sequence**, independent of context length.

**Packed to the precision the reference actually stores.** The 448 non-rope dims of every 512-wide
entry hold only e4m3 information (7 blocks of 64, e8m0 scale); the 128-wide indexer entries hold
only e2m1 information (4 blocks of 32, e8m0 scale). A cache that stores `(q, s)` instead of the
bf16 round-trip:

| | reference bf16 | packed |
|---|---|---|
| 512-wide entry | 1024 B | 448 + 7 + 64·2 = **583 B** |
| 128-wide indexer entry | 256 B | 64 + 4 = **68 B** |
| B+C marginal | 320 B/tok/layer | **162.75** |
| D marginal | 8 | **4.55** |
| **tower marginal** | **6 880 B/token** | **≈ 3 509 B/token** |
| window ring, fixed | 131 072 B/layer | 74 624 B/layer |

This is bit-identical to the reference **only if the kernel reconstructs `round_bf16(q·s)` before
the dot product** — the reference stores `bf16(f32(q)·s)` (`kernel.py:84-91`), so dequantising
straight to f32 gives a *different* (slightly more accurate) answer. See Q3 in §8.

### 1.8 RoPE

`precompute_freqs_cis`, `model.py:206-235`. `apply_rotary_emb`, `model.py:238-250`, uses
`view_as_complex(x.unflatten(-1, (-1,2)))` — pairs **adjacent** dims `(2i, 2i+1)`, i.e. GPT-J
interleaved, **not** NeoX half-split. `inverse=True` conjugates.

Two tables per model, selected per layer by `compress_ratio` (`model.py:481-487`):

| | layers | `base` | `original_seq_len` | YaRN |
|---|---|---|---|---|
| compressing (41) | 2..42 | `compress_rope_theta` = **160 000** | 65 536 | **on**, `factor=16`, `beta_fast=32`, `beta_slow=1` |
| non-compressing (2 + 3 DSpark) | 0, 1, `mtp.*` | `rope_theta` = **10 000** | 0 | **off** — `model.py:484` "disable YaRN and use base rope_theta in pure sliding-window attention" |

Within a compressing layer the *same* table serves the query, the window KV, the compressor
output, and (for ratio-4) the indexer's query and compressed keys (`model.py:497-500`).
`max_position_embeddings = 1048576 = 65536 · 16`, consistent.

### 1.9 Kind-by-kind dataflow, exact shapes

Batch `B`, tokens `S`, decode is `S = 1`.

#### Kind A — layers 0, 1 (`compress_ratio = 0`)

```
in  x[B,S,4,4096]
hc_pre                                            -> x[B,S,4096], post[B,S,4], comb[B,S,4,4]
attn_norm  RMSNorm(4096)                          -> [B,S,4096]
wq_a       [4096->1024] fp8                       -> qr[B,S,1024]
q_norm     RMSNorm(1024)
wq_b       [1024->32768] fp8                      -> [B,S,64,512]
per-head unweighted RMS; rope(last 64), theta 1e4, no YaRN
wkv        [4096->512] fp8; kv_norm RMSNorm(512); rope(last 64); fake-fp8(dims 0..447, blk 64)
                                                  -> kv[B,S,512]  -> ring slot pos%128
topk_idxs  = window(128)                          -> [B,S,128] i32
sparse_attn(q[B,S,64,512], kv, attn_sink[64], idx, 512**-0.5)   -> o[B,S,64,512]
inverse-rope(o[...,-64:])
o.view(B,S,8,4096); einsum with wo_a(8,1024,4096) -> [B,S,8,1024]
wo_b       [8192->4096] fp8                       -> [B,S,4096]
hc_post                                           -> [B,S,4,4096]
hc_pre; ffn_norm; MoE with HASH gate; hc_post
```

KV: 131 072 B/seq fixed, **0 B/token**.

#### Kinds B, C — layers 2 and 4,6,…,42 (`compress_ratio = 4`, CSA)

Identical to Kind A up to the KV write, plus:

```
indexer(x, qr, start_pos, offset)                                    (§2)  -> pool ids [B,S,<=512]
compressor(x, start_pos)   d=512, coff=2, overlap
    wkv/wgate [4096->1024] f32; ape[4,1024]
    prefill: reshape to [B,S/4,4,1024] -> overlap_transform -> [B,S/4,8,512]
             kv_c = (kv * softmax(score,dim=2)).sum(2)                     -> [B,S/4,512]
    decode : 8-slot fp32 ring; fires on (pos+1)%4==0                       -> [B,1,512]
    norm RMSNorm(512); rope at position 4j (theta 1.6e5 + YaRN); fake-fp8(0..447, blk 64)
    -> kv_cache[:, 128 + j]
topk_idxs = cat(window[128], pool_ids + offset)      -> [B,S, 128 + <=512]
sparse_attn over the union
```

Rope theta **160 000 + YaRN(16, orig 65536)** for q, window KV, compressed KV, indexer q and
indexer keys alike. KV: 131 072 B/seq fixed + **320 B/token** (256 attention + 64 indexer).

#### Kind D — layers 3, 5, …, 41 (`compress_ratio = 128`, HCA)

```
compressor(x, start_pos)   d=512, coff=1, NO overlap, NO indexer
    wkv/wgate [4096->512] f32; ape[128,512]
    prefill: reshape to [B,S/128,128,512]; kv_c = (kv*softmax(score,dim=2)).sum(2)
    decode : 128-slot fp32 ring; fires on (pos+1)%128==0
    norm; rope at position 128j; fake-fp8(0..447, blk 64)  -> kv_cache[:, 128 + j]
topk_idxs = cat(window[128], arange((pos+1)//128) + offset)   # ALL of them, no selection
```

KV: 131 072 B/seq fixed + **8 B/token**. Attended entries grow to 8 320 at 1 M context — this is
the layer kind whose attention work is *not* bounded.

#### Kind S — DSpark stages (`mtp.0/1/2`, `compress_ratio = 0`)

See §5.

---

## 2. The indexer

`Indexer`, `model.py:386-440`. Present on exactly the 21 ratio-4 layers
(`model.py:474-477`: `if compress_ratio == 4: Indexer(...) else None`), verified against the
checkpoint (`layers.{2,4,…,42}.attn.indexer.*`).

### 2.1 What it computes

```
q      = wq_b(qr)                      # [1024 -> 64*128] fp8, model.py:399,417
                                       # qr is the MAIN attention's q_norm(wq_a(x)) — shared, model.py:502
q      = q.unflatten(-1, (64,128))
rope(q[..., -64:])                     # the LAYER's freqs_cis: theta 1.6e5 + YaRN, model.py:419
q      = rotate_activation(q)          # Hadamard-128, scale 128**-0.5, model.py:420
fp4_act_quant(q, 32, inplace=True)     # fake e2m1 + e8m0 per-32, model.py:422

compressor(x, start_pos)               # writes kv_cache[j] for the completed 4-token pool, model.py:423
                                       # d=128, coff=2, overlap; rotate=True -> Hadamard + fake-fp4

weights = weights_proj(x) * (128**-0.5 * 64**-0.5)         # [4096 -> 64] bf16, model.py:424
score   = einsum("bshd,btd->bsht", q, kv_cache[:, :end//4]) # model.py:427
score   = (relu_(score) * weights.unsqueeze(-1)).sum(dim=2) # model.py:428  -> [B,S,T]
<causal mask on prefill: pool t visible to query i iff t < (i+1)//4>        # model.py:432-434
topk_idxs = score.topk(min(512, end_pos // 4))[1]                           # model.py:435
<prefill: re-mask violators to -1; else += offset>                          # model.py:436-439
```

There is **no softmax** over index scores and **no attention sink**. The scale is folded into
`weights`, so selection is scale-invariant. `index_topk = 512` is already **pool**-granular —
512 pools × 4 tokens = 2 048 tokens' worth of evidence, the same reach GLM-5.3-Flash gets from
`index_topk = 2048` token-granular ÷ `index_kpool = 4`.

### 2.2 Same / different / absent versus plow's DSA indexer

plow's GLM-5.3-Flash pooled DSA path: `crates/devgen/src/mla.rs:3752-3877` (decode),
`mla.rs:4287-4440` (prefill); ops `DsaPoolStash(132)`, `DsaPoolCompress(130)`, `DsaQQuant(133)`,
`IndexScoreKpool(134)`, `IndexSelect(59)`, `DsaPoolExpand(131)`.

| aspect | plow / GLM-5.3-Flash | DeepSeek-V4-Flash | verdict |
|---|---|---|---|
| pooling granularity | `index_kpool = 4` (`mla.rs:88`) | `compress_ratio = 4` | **same** |
| per-dim softmax over the pool | yes (`dev.rs:1558-1560`) | yes (`model.py:347`) | **same** |
| additive `ape` position bias on the gate | yes, `w.ikpa`, `[pool_size][head_dim]` | yes, `ape[4, 2·128]` (`model.py:300`) | **same shape idea, 2× wider** |
| Hadamard-128, `1/sqrt(128)` | yes (`dev.rs:1561`) | yes (`model.py:253-257`) | **same constant** |
| `index_head_dim` | 128, hard-asserted (`mla.rs:191`) | 128 | **same** |
| `index_n_heads` | **32**, hard-asserted (`mla.rs:184`) | **64** | **different — plow asserts** |
| score form | `Σ_h relu(q·k)·w[h]`, scale folded into `w` (`dev.rs:1690-1694`) | identical (`model.py:428`) | **same** |
| `weights_proj` precision | **fp32**, `GemvF32(135)` (`mla.rs:3787`, rationale `dev.rs:1697-1712`) | bf16 weight, bf16 GEMV (`model.py:400`) | **different — V4 is bf16** |
| what is pooled | the **per-token indexer key** `wk@x`, k_norm'd and rope'd (`mla.rs:3738-3778`) | a **dedicated `wkv` projection of `x`** — V4 has no per-token indexer key at all | **different (structural)** |
| overlap across pool boundaries | none | yes: 8 slots, 2 separate projections (`model.py:313-320`) | **absent in plow** |
| RMSNorm after pooling | none | `norm: RMSNorm(128)`, learned (`model.py:305`) | **absent in plow** |
| rope order | per token, **before** pooling | after pooling+norm, at the pool's **first-token** position | **different** |
| q/k numeric format | fp8 e4m3, one pow-2 scale per vector | **fp4 e2m1, e8m0 per 32**, fake-quantised back to bf16 | **different** |
| pooling arithmetic dtype | bf16 round-trips (`dev.rs:1560-1562`) | fp32 throughout (`model.py:328`) | **different** |
| `topk` granularity | pools, `itk/index_kpool` (`mla.rs:3846`) | pools, `index_topk` directly | **same** |
| pool ids → token ids expansion | required, `DsaPoolExpand(131)` (`mla.rs:3869`) | **not required** — pools are attended *as* KV entries | **absent (and unneeded)** |
| indexer key cache | fp8, `ctx/4 × 128` (`mla.rs:2083-2090`) | bf16 (fake-fp4), `S/4 × 128` (`model.py:405`) | **same shape** |
| shared vs full indexer layers | `indexer_types` (`mla.rs:90`) | every ratio-4 layer owns one; no sharing | **absent (simpler)** |
| decode-only crossover | `ctx > 65536` gate (`mla.rs:154-155`) | always on for ratio-4 layers | **different** |

The one-line summary: **the scoring and selection half is nearly identical; the key-production
half is a different mechanism.** In GLM the pool is a compression *of* the indexer key; in V4 the
pool *is* the key, produced by its own projection, and it is the only thing the indexer ever sees.

---

## 3. mHC residuals

### 3.1 Where it sits

The residual stream is `[B, S, 4, 4096]` for the whole tower. `hc_pre` reduces 4 → 1 immediately
before each sub-layer's norm; `hc_post` expands 1 → 4 immediately after. Two independent
parameter sets per layer (`hc_attn_*`, `hc_ffn_*`), so **two** mHC round trips per block.
`Block.forward`, `model.py:695-707`.

**Per token, not per sequence.** Every gate is a function of that token's own 4-stream state:
`mixes = F.linear(x.flatten(2), hc_fn) * rsqrt(...)` at `model.py:684-685` produces a `[B,S,24]`
tensor and `hc_split_sinkhorn` runs one workgroup per token (`kernel.py:383`).

### 3.2 Exact computation

`mix_hc = (2 + hc_mult)·hc_mult = 24`, `hc_dim = 4·4096 = 16384` (`model.py:676-677`).

```
hc_pre(x[B,S,4,4096]):                                          # model.py:680-688
  xf   = x.flatten(2).float()                                   # [B,S,16384]
  inv  = rsqrt(mean(xf^2, -1, keepdim) + norm_eps)               # norm_eps = 1e-6, NOT hc_eps
  mixes = F.linear(xf, hc_fn[24,16384]) * inv                    # [B,S,24]   (fp32 weight)
  pre, post, comb = hc_split_sinkhorn(mixes, hc_scale[3], hc_base[24], 4, 20, 1e-6)
  y = Σ_c pre[..,c] * xf.view(shape)[..,c,:]                     # [B,S,4096], fp32 accumulate
  return y.to(bf16), post, comb                                  # UNNORMED; attn_norm/ffn_norm follows

hc_split_sinkhorn:                                              # kernel.py:372-427
  pre [j]  = sigmoid(mixes[j]     * hc_scale[0] + hc_base[j])     + eps      j in [0,4)
  post[j]  = 2 * sigmoid(mixes[4+j] * hc_scale[1] + hc_base[4+j])            j in [0,4)
  comb[j,k]=          mixes[8+4j+k]* hc_scale[2] + hc_base[8+4j+k]
  comb = softmax(comb, dim=-1) + eps          # row softmax over k
  comb = comb / (comb.sum(dim=-2) + eps)      # column normalise
  repeat (sinkhorn_iters - 1) = 19 times:
      comb = comb / (comb.sum(-1, keepdim) + eps)
      comb = comb / (comb.sum(-2)          + eps)

hc_post(x[B,S,4096], residual[B,S,4,4096], post, comb):          # model.py:690-694
  out[..,k,:] = post[..,k] * x[..,:] + Σ_j comb[..,j,k] * residual[..,j,:]
```

Index convention, from `model.py:693` (`comb.unsqueeze(-1) * residual.unsqueeze(-2)` summed over
`dim=2`): `comb[j,k]` moves **source** stream `j` into **destination** stream `k`. The final
Sinkhorn step is a column normalise, so each destination receives a convex combination — that is
the norm-preserving property the mechanism exists for.

`hc_eps = 1e-6` serves double duty: additive on `pre`/`comb` and as the Sinkhorn denominator
guard. `hc_mult = 4` and the `2 ×` factor on `post` are both hard-wired.

### 3.3 The final contraction is *not* a mean

`hc_head`, `model.py:709-717`, run once at the end of the tower (`model.py:923`) and once at the
end of the DSpark stack (`model.py:861`):

```
mixes = F.linear(xf, hc_head_fn[4, 16384]) * rsqrt(mean(xf^2) + norm_eps)
pre   = sigmoid(mixes * hc_head_scale[1] + hc_head_base[4]) + hc_eps
y     = Σ_c pre[c] * x[c]
```

A learned sigmoid-gated weighted sum with its own `[4, 16384]` projection — **no Sinkhorn, no
`post`, no `comb`**. Note `hc_head_scale` has shape `[1]` and broadcasts.

The **entry** expansion (`model.py:915`) is by contrast a plain `repeat` with no gate at all, and
the DSpark target taps (`model.py:920`) are a plain `h.mean(dim=2)`.

### 3.4 Against plow

plow already ships this mechanism for GLM-5.3-Flash: `HyperConnPre(128)` / `HyperConnPost(129)`,
kernel at `runtime/amd/op_hyperconn.h`, emit at `crates/devgen/src/mla.rs:466-533`.

Term-by-term against `op_hyperconn.h`:

| term | plow | V4 | |
|---|---|---|---|
| `n3 = 2n + n²` split | 24 (`op_hyperconn.h:59`) | 24 (`model.py:676`) | ✅ |
| RMS scale source | `rsqrt(Σ residual² / (n·hidden) + rms_eps)` (`:75`) | same (`model.py:684`) | ✅ |
| `pre` | `sigmoid(m·inv·s0 + b) + hc_eps` (`:83`) | same (`kernel.py:392`) | ✅ |
| `post` | `sigmoid(...)·PLOW_HC_POST_MULT`, compile-time **2.0** (`:43,:88`) | `2·sigmoid(...)` (`kernel.py:394`) | ✅ |
| `comb` softmax → col-norm → (iters−1)×(row,col) | `:93-127` | `kernel.py:401-423` | ✅ |
| `sinkhorn_repeat` | operand `i3`, emitted as **20** (`mla.rs:497`) | 20 | ✅ |
| `hc_eps` | operand `f1`, emitted as **1e-6** (`mla.rs:499`) | 1e-6 | ✅ |
| `layer_input` UNNORMED, caller chains RmsNorm | `:132-140` | `model.py:697` | ✅ |
| `hc_post` mode 0 | `Σ_i cm[i·n+j]·res[i][d] + pm[j]·x[d]` (`:183-184`) | identical (`model.py:693`) | ✅ |
| entry expand | mode 1 = broadcast copy (`:156-161`) | `repeat` (`model.py:915`) | ✅ |
| **exit contract** | mode 2 = **arithmetic mean** `acc/n` (`:163-171`) | **learned gated sum** (`model.py:709-717`) | ❌ |
| `mixes` projection | `GemvF32(135)`, fp32 weight `[24, 4·hidden]` (`mla.rs:477`) | fp32 weight `[24, 16384]` | ✅ |

**Verdict: `HyperConnPre(128)` and `HyperConnPost(129)` modes 0 and 1 are reusable as-is at
`n=4, hidden=4096, sinkhorn_repeat=20, hc_eps=1e-6`.** The exit contraction needs a mode 3 that
takes `pre` from a `GemvF32` over `hc_head_fn[4,16384]` and a `[1]` scale — a small arm on
op 129, not a new kernel. plow's existing mode 2 (mean) is *exactly* what V4 uses for its DSpark
target taps at `model.py:920`, so it is reusable there.

---

## 4. MoE

### 4.1 Routing

`Gate.forward`, `model.py:569-589`:

```
scores = linear(x.float(), weight.float())      # [n,256] fp32; gate.weight ships BF16, upcast
scores = F.softplus(scores).sqrt()              # "sqrtsoftplus"  = sqrt(log(1+exp(s)))
original_scores = scores
if bias is not None: scores = scores + bias     # SELECTION ONLY  (e_score_correction_bias)
indices = tid2eid[input_ids]  if hash else  scores.topk(6, dim=-1)[1]
weights = original_scores.gather(1, indices)    # UNBIASED score is the gate value
weights /= weights.sum(-1, keepdim=True)        # norm_topk_prob
weights *= 1.5                                  # routed_scaling_factor
```

**`sqrtsoftplus` vs plow.** plow's router supports `sigmoid` (flag bit 0) and `softmax` only
(`crates/packet/src/moe.rs:32`, `runtime/amd/op_moe.h:230`). `sqrt(softplus(s))` is a third
transform: non-negative like sigmoid, but **unbounded above** (≈ `sqrt(s)` for large `s`).
Because `norm_topk_prob` renormalises, the absolute scale is discarded — but the *relative*
weighting is not, and the ordering under `+bias` is not either.

**`noaux_tc` vs plow.** `config.json` says `topk_method: "noaux_tc"`. The reference does a **flat
top-6 over all 256 experts** (`model.py:580`). There is no group-limited stage, and `config.json`
carries **no `n_group` / `topk_group` fields**. So here `noaux_tc` means only "no auxiliary
balancing loss; a learned correction bias shifts the selection ranking" — the group-limited half
of DeepSeek-V3's `noaux_tc` is **absent**. plow's `MoeRouterTopk(56)` takes `i6=n_group`,
`i7=topk_group` and is inert at `n_group ≤ 1` (`dev.rs:596-602`, `mla.rs:6328-6329`) — so V4 wants those
set to 0/1 and gets flat top-k for free.

Everything else about plow's router already matches V4 exactly: bias applied to the selection key
only, unbiased score as the gate (`op_moe.h:221-222`, `:264`, `:272`), lowest-expert-id tie-break,
`norm_topk` then `route_scale` (`op_moe.h:279-280`).

One precision note: plow's split router rounds the score logit to bf16 in the GEMV before
`MoeRouterTopk` reads it (`dev.rs:596-602`); V4 computes it in fp32 end to end
(`model.py:570`). `GemvF32(135)` exists for exactly this class of problem.

**Hash routing.** Layers 0–2 have no scoring stage at all: `indices = tid2eid[input_ids]`, a
`[129280, 6]` lookup by token id (`model.py:562`, `model.py:578`). The gate *weights* still come
from the score path, so the score GEMV still runs. plow has **nothing** resembling this
(subagent grep: zero hits for hash routing in `crates/`).

### 4.2 Expert compute

`Expert.forward`, `model.py:601-611`:

```
gate = w1(x).float()
up   = w3(x).float()
up   = clamp(up,  -10.0, +10.0)     # BOTH sides
gate = clamp(gate, max=10.0)        # UPPER ONLY
x    = silu(gate) * up
x    = weights * x                  # routing weight applied to the INTERMEDIATE, pre-w2
return w2(x.to(bf16))
```

`swiglu_limit = 10.0`. `MoE.forward` (`model.py:634-650`) accumulates in fp32 and adds the shared
expert unconditionally.

**vs plow's `ACT_SWIGLU_OAI = 3`** (`crates/packet/src/dev.rs:1809`, semantics `dev.rs:115-123`):

```
plow act 3 :  A(g) = min(g,L)·σ(α·min(g,L))      B(u) = clamp(u,±L) + 1
V4         :  A(g) = min(g,L)·σ(   min(g,L))     B(u) = clamp(u,±L)
```

With `α = 1.0` the gate half matches, but plow's `+1` on the up branch does not. **V4 needs a new
act code 4** on the GLU family (`Glu(5)`, `GemvGlu(19)`, `GemmGlu(20)`, `MoeGluMx(150)`,
`MoeGluMxPf(152)`, `DenseGluFp8Blk(47)`, …) — an arm, not a kernel. plow's `mla.rs` uses
`GLM_ACT_SILU = 1` unconditionally at ~20 sites (`mla.rs:1411`), so nothing on the MLA path
carries a limit today.

Also: V4 applies the routing weight *before* the bf16 cast and *before* `w2`; plow's
`MoeDownMx(151)` applies it to the f32 down output (`dev.rs:1780`). Algebraically identical
(there is no `w2` bias in V4), numerically not.

### 4.3 Expert weight format — fp4 e2m1 with e8m0, block **1 × 32**

| tensor | ckpt dtype | ckpt shape | logical | scale |
|---|---|---|---|---|
| `ffn.experts.{e}.w1.weight` | `I8` | `[2048, 2048]` | `[2048, 4096]` fp4 | `w1.scale` `F8_E8M0 [2048, 128]` |
| `ffn.experts.{e}.w3.weight` | `I8` | `[2048, 2048]` | `[2048, 4096]` fp4 | `[2048, 128]` |
| `ffn.experts.{e}.w2.weight` | `I8` | `[4096, 1024]` | `[4096, 2048]` fp4 | `[4096, 64]` |

`128 = 4096/32`, `64 = 2048/32`: **one e8m0 exponent per output row per 32 input elements.**
`Linear.__init__`, `model.py:136-141`, is explicit: weight `[out, in//2]` in
`float4_e2m1fn_x2`, scale `[out, in//32]` in `float8_e8m0fnu`.

Nibble order and code table, `convert.py:30-33` and `convert.py:11-14`:

```
low  = byte & 0x0F   -> the EVEN (first) element along K
high = byte >> 4     -> the ODD  (second) element along K
codes 0..7  :  0, 0.5, 1, 1.5, 2, 3, 4, 6
codes 8..15 :  0, -0.5, -1, -1.5, -2, -3, -4, -6      (sign in bit 3)
```

Dequant: `w = code · 2^(e8m0 − 127)` (`kernel.py:29-33`).

Activations for an fp4 GEMM are **fp8 e4m3 with a per-128-K power-of-2 scale**
(`model.py:117-119` → `act_quant(x, 128, "ue8m0", e8m0)`), and `fp4_gemm_kernel`
(`kernel.py:442-516`) applies `scale_a[m, k//4] · scale_b[n, k]` at `block_K = 32`. So it is
**w4a8**, not w4a16 and not A4W4.

**vs the block-fp8 experts plow ships for GLM-5.3:**

| | plow / GLM-5.3 | V4 |
|---|---|---|
| element format | fp8 e4m3, 1 B | fp4 e2m1, 2 per byte |
| scale block | **`[128 out] × [128 K]`** 2-D grid | **`[1 out] × [32 K]`** row-wise |
| scale dtype | **f32** (`dev.rs:480-486`; disk name `.weight_scale_inv`) | **e8m0**, 1 B |
| activation | bf16 (w8a16) | fp8 e4m3, per-128-K pow-2 (w4a8) |
| plow opcode | `MoeGroupGluFp8Blk(48)` / `MoeGroupDownFp8Blk(49)` | — |

But plow's **MXFP4** family is a near-exact match. `MoeGluMx(150)` / `MoeDownMx(151)` (decode) and
`MoeGluMxPf(152)` / `MoeDownMxPf(153)` (prefill) read flat per-expert tensors
`W[E][N][K/2]` fp4 + `S[E][N][K/32]` e8m0, with "LOW nibble = even k, e2m1 LUT
{0,.5,1,1.5,2,3,4,6} with sign in bit 3" (`runtime/common/dev_isa.h:1272-1273`; mirrored at `dev.rs:914` and `dev.rs:1766`) — **byte-identical to V4's
disk layout.** Two deltas: (a) V4 ships `w1` and `w3` as *separate* tensors, so a load-time concat
into the `[E][2I][K/2]` gate|up buffer is needed (`i4 = 1` selects the blocked layout, so no
interleave repack); (b) these ops are w4a16 today, and V4's reference is w4a8.

`config.json`'s `quantization_config` (`fmt: e4m3`, `weight_block_size: [128,128]`) **does not
describe the experts.** It describes the *fp8* tensors — `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`,
`indexer.wq_b`, the shared experts, `mtp.0.main_proj`. Confirmed by scale shapes:
`wq_a.weight [1024,4096]` → `wq_a.scale [8,32]` = `⌈1024/128⌉ × ⌈4096/128⌉`. **This is a real
config-vs-code disagreement and must be recorded in whatever config struct plow grows.**

The shared expert is **fp8 e4m3 block-[128,128]**, not fp4: `model.py:632` constructs it without a
`dtype`, so it takes `default_dtype = float8_e4m3fn` (`model.py:882`). Checkpoint agrees:
`shared_experts.w1.weight F8_E4M3 [2048,4096]`, `.scale F8_E8M0 [16,32]`.

**The scale dtype is e8m0 everywhere in this checkpoint, including the fp8 tensors.** plow's
block-fp8 ops all take an f32 grid. Either the loader converts `2^(v−127) → f32` at load
(cheap, exact, and the scale tensors are only 9.3 GB → 37 GB inflated — probably unacceptable) or
the block-fp8 ops gain an e8m0 scale arm.

---

## 5. DSpark and the nextn/MTP layer

### 5.1 Is it required for correct base output? **No.**

`Transformer.forward` (`model.py:913-927`) is self-contained: it returns `output_ids, logits,
main_hidden` from the 43-layer tower alone. `main_hidden` is consumed only by `forward_spec`
(`model.py:929-937`). **The shipped `generate.py` never calls `forward_spec`** — `generate.py:42`
is `model.forward(tokens[:, prev_pos:cur_pos], prev_pos)[0]` and that is the only model call in
the file. `forward_spec` appears only in `model.py`'s `__main__` demo (`model.py:952-961`).

Construction of the whole stack is gated on one field: `if args.dspark_block_size:`
(`model.py:895`). Setting it to 0 drops all three stages, `main_proj`, both heads, and the
`main_hiddens` taps. **A first bringup can skip DSpark entirely and be bit-exact on base-model
output.** It is 19.85 B logical params / 10.86 GB of the checkpoint (6.5 %).

### 5.2 What it is

Not an EAGLE/Medusa next-token head. It is a **3-layer block-parallel drafter** that emits
`dspark_block_size = 5` tokens at once.

`DSparkBlock` (`model.py:818-876`) subclasses `Block` with `attention_cls = DSparkAttention`
(`model.py:821`), so it has the same mHC + MoE structure. It shares the main model's `embed` and
`head` (`model.py:897-898`); `convert.py:91-92` explicitly skips `mtp.*` embed/head tensors.

```
forward_embed  (mtp.0 only, model.py:851-858)
    main_x = main_norm(main_proj(main_hidden))      # main_proj [4096*3 -> 4096] fp8
                                                    # main_hidden = cat of h.mean(dim=2) at layers 40,41,42
    draft_ids = [input_ids, 128799, 128799, 128799, 128799]   # dspark_noise_token_id
    x = embed(draft_ids).unsqueeze(2).repeat(1,1,4,1)          # [B,5,4,4096]

DSparkAttention.forward  (model.py:752-792)
    prefill (start_pos == 0):  compute main_kv from main_x, fill the 128-ring, RETURN x UNCHANGED
                               -> the whole DSpark stack is a KV-warmup pass during prefill
    decode:  q/kv from the 5 draft positions at start_pos+seqlen .. +5
             topk = [0 .. min(128, start_pos+1))  U  [128 .. 133)          # model.py:745-748
             -> the 5 draft rows are appended to the ring and are ALL visible to ALL 5 queries:
                the draft block is NOT causal within itself

forward_head  (mtp.2 only, model.py:860-875)
    x = hc_head(x, hc_head_fn, hc_head_scale, hc_head_base)   # learned 4->1, own params
    logits = head(norm(x), full_logits=True)                  # [B,5,129280]
    for i in 0..4:                                            # SEQUENTIAL
        bias, emb = markov_head(output_ids[:, i])             # rank-256 bigram logit bias
        logits[:, i] += bias
        output_ids[:, i+1] = sample(logits[:, i], temperature)
    confidence = confidence_head(x, stack(embs))              # Linear(4096+256 -> 1) fp32
```

`DSparkMarkovHead` (`model.py:795-804`) is `markov_w2 @ markov_w1[token]` — a rank-256 factored
bigram table over the full 129 280 vocabulary, `[129280,256]` bf16 each way.

### 5.3 `num_nextn_predict_layers: 1` is wrong

`config.json` says `"num_nextn_predict_layers": 1`. Both the reference config
(`inference/config.json`: `"n_mtp_layers": 3`) and the checkpoint say **3**:
`mtp.0`, `mtp.1`, `mtp.2` all ship a full attention + MoE + mHC parameter set, `mtp.0` alone
carries `main_proj`/`main_norm` (`stage_id == 0`, `model.py:826`) and `mtp.2` alone carries
`norm`/`markov_head`/`confidence_head`/`hc_head_*` (`stage_id == n_mtp_layers - 1`,
`model.py:830`). `mtp.0.main_proj.weight` is `[4096, 12288]` — `4096 × len(dspark_target_layer_ids)`
with 3 target layers, confirming the geometry independently. **Take 3, not 1.**

The README's vLLM line uses `"num_speculative_tokens": 7` against a `dspark_block_size` of 5;
unexplained by the reference. Recorded as Q7 in §8.

### 5.4 What serving it would take

Everything in §7's table, plus: a second program (5-row block) sharing every weight with the main
tower; a `main_hidden` tap that means-reduces the 4 streams at layers 40/41/42 and concatenates;
the non-causal-within-block index list; a sequential 5-step Markov-bias sampling loop; and an
acceptance policy the reference does not define (the confidence head produces a score; nothing in
the shipped code consumes it). plow's speculative machinery is a runtime multi-model pipeline
(`crates/plowrt/src/orch/speculative.rs:1-21`), not an emit feature, and there is no draft-head
emitter anywhere in `devgen`.

---

## 6. Tensor name → role map

All names are as they appear in `model.safetensors.index.json`. The shipped checkpoint is
**already in `inference/model.py`'s naming convention** (`layers.N.attn.*`, `.scale`,
`gate.bias`) — `convert.py`'s renames (`self_attn→attn`, `mlp→ffn`, `weight_scale_inv→scale`,
`e_score_correction_bias→bias`, `convert.py:93-96`) are no-ops on this checkpoint.

`{N}` = layer index, `{e}` = expert index. `I8` means packed fp4 (2 values/byte along K).

### 6.1 Global

| tensor | dtype | shape | role |
|---|---|---|---|
| `embed.weight` | BF16 | `[129280, 4096]` | token embedding |
| `head.weight` | BF16 | `[129280, 4096]` | lm_head; declared f32 (`model.py:729`), upcast at load; not tied |
| `norm.weight` | BF16 | `[4096]` | final RMSNorm |
| `hc_head_fn` | F32 | `[4, 16384]` | tower exit mHC contraction projection (§3.3) |
| `hc_head_base` | F32 | `[4]` | " |
| `hc_head_scale` | F32 | `[1]` | " |

### 6.2 Every layer, `N ∈ [0,43)` — 43 each

| tensor | dtype | shape | role |
|---|---|---|---|
| `layers.{N}.attn_norm.weight` | BF16 | `[4096]` | pre-attention RMSNorm |
| `layers.{N}.ffn_norm.weight` | BF16 | `[4096]` | pre-FFN RMSNorm |
| `layers.{N}.hc_attn_fn` | F32 | `[24, 16384]` | mHC pre projection, attention half |
| `layers.{N}.hc_attn_base` | F32 | `[24]` | mHC additive base |
| `layers.{N}.hc_attn_scale` | F32 | `[3]` | mHC `[pre, post, comb]` scales |
| `layers.{N}.hc_ffn_fn` | F32 | `[24, 16384]` | mHC pre projection, FFN half |
| `layers.{N}.hc_ffn_base` | F32 | `[24]` | |
| `layers.{N}.hc_ffn_scale` | F32 | `[3]` | |
| `layers.{N}.attn.wq_a.weight` | F8_E4M3 | `[1024, 4096]` | q down-projection |
| `layers.{N}.attn.wq_a.scale` | F8_E8M0 | `[8, 32]` | block-[128,128] |
| `layers.{N}.attn.q_norm.weight` | BF16 | `[1024]` | RMSNorm on the q latent |
| `layers.{N}.attn.wq_b.weight` | F8_E4M3 | `[32768, 1024]` | q up-projection, **W_UK pre-absorbed** |
| `layers.{N}.attn.wq_b.scale` | F8_E8M0 | `[256, 8]` | |
| `layers.{N}.attn.wkv.weight` | F8_E4M3 | `[512, 4096]` | the single latent KV projection (K **and** V) |
| `layers.{N}.attn.wkv.scale` | F8_E8M0 | `[4, 32]` | |
| `layers.{N}.attn.kv_norm.weight` | BF16 | `[512]` | RMSNorm on the KV latent |
| `layers.{N}.attn.attn_sink` | F32 | `[64]` | per-head softmax-denominator sink |
| `layers.{N}.attn.wo_a.weight` | F8_E4M3 | `[8192, 4096]` | **block-diagonal** `(8, 1024, 4096)`; declared bf16, dequantised at load (`convert.py:123-127`) |
| `layers.{N}.attn.wo_a.scale` | F8_E8M0 | `[64, 32]` | |
| `layers.{N}.attn.wo_b.weight` | F8_E4M3 | `[4096, 8192]` | output up-projection |
| `layers.{N}.attn.wo_b.scale` | F8_E8M0 | `[32, 64]` | |
| `layers.{N}.ffn.gate.weight` | BF16 | `[256, 4096]` | router logits; used as f32 (`model.py:570`) |
| `layers.{N}.ffn.shared_experts.w1.weight` | F8_E4M3 | `[2048, 4096]` | shared gate |
| `layers.{N}.ffn.shared_experts.w1.scale` | F8_E8M0 | `[16, 32]` | |
| `layers.{N}.ffn.shared_experts.w3.{weight,scale}` | F8_E4M3 / F8_E8M0 | `[2048,4096]` / `[16,32]` | shared up |
| `layers.{N}.ffn.shared_experts.w2.{weight,scale}` | F8_E4M3 / F8_E8M0 | `[4096,2048]` / `[32,16]` | shared down |
| `layers.{N}.ffn.experts.{e}.w1.weight` | **I8** | `[2048, 2048]` | routed gate, logical `[2048,4096]` fp4 |
| `layers.{N}.ffn.experts.{e}.w1.scale` | F8_E8M0 | `[2048, 128]` | 1×32 along K |
| `layers.{N}.ffn.experts.{e}.w3.{weight,scale}` | I8 / F8_E8M0 | `[2048,2048]` / `[2048,128]` | routed up |
| `layers.{N}.ffn.experts.{e}.w2.{weight,scale}` | I8 / F8_E8M0 | `[4096,1024]` / `[4096,64]` | routed down |

`e ∈ [0,256)` — 11 008 tensors per projection across the 43 layers.

### 6.3 Kind-specific

**Hash gate — `N ∈ {0,1,2}`, 3 tensors**

| tensor | dtype | shape | role |
|---|---|---|---|
| `layers.{N}.ffn.gate.tid2eid` | **I64** | `[129280, 6]` | token id → 6 expert ids. Declared `int32` (`model.py:562`) — **checkpoint/code dtype disagreement** |

**Scored gate — `N ∈ [3,43)`, 40 tensors**

| tensor | dtype | shape | role |
|---|---|---|---|
| `layers.{N}.ffn.gate.bias` | F32 | `[256]` | `e_score_correction_bias`; selection ranking only |

**Compressor — `N ∈ [2,43)`, 41 each**

| tensor | dtype | ratio 4 (21 layers) | ratio 128 (20 layers) | role |
|---|---|---|---|---|
| `layers.{N}.attn.compressor.wkv.weight` | BF16 | `[1024, 4096]` | `[512, 4096]` | pooled value proj; declared f32 (`model.py:303`) |
| `layers.{N}.attn.compressor.wgate.weight` | BF16 | `[1024, 4096]` | `[512, 4096]` | pooling gate logits |
| `layers.{N}.attn.compressor.ape` | F32 | `[4, 1024]` | `[128, 512]` | position-in-block gate bias |
| `layers.{N}.attn.compressor.norm.weight` | BF16 | `[512]` | `[512]` | RMSNorm after pooling |

Ratio-4 layers: `{2,4,6,…,42}`. Ratio-128 layers: `{3,5,7,…,41}`. The `coff = 2` width doubling on
the ratio-4 form is the overlap (§1.6).

**Indexer — `N ∈ {2,4,…,42}`, 21 each**

| tensor | dtype | shape | role |
|---|---|---|---|
| `layers.{N}.attn.indexer.wq_b.weight` | F8_E4M3 | `[8192, 1024]` | 64 heads × 128; consumes the **shared** `qr` |
| `layers.{N}.attn.indexer.wq_b.scale` | F8_E8M0 | `[64, 8]` | |
| `layers.{N}.attn.indexer.weights_proj.weight` | BF16 | `[64, 4096]` | per-head score weights |
| `layers.{N}.attn.indexer.compressor.wkv.weight` | BF16 | `[256, 4096]` | indexer key production (`2 × 128`) |
| `layers.{N}.attn.indexer.compressor.wgate.weight` | BF16 | `[256, 4096]` | |
| `layers.{N}.attn.indexer.compressor.ape` | F32 | `[4, 256]` | |
| `layers.{N}.attn.indexer.compressor.norm.weight` | BF16 | `[128]` | |

**DSpark — `mtp.{0,1,2}`**

Each of the 3 stages carries the full attention (`wq_a/q_norm/wq_b/wkv/kv_norm/attn_sink/wo_a/wo_b`
+ scales), `attn_norm`, `ffn_norm`, `hc_{attn,ffn}_{fn,base,scale}`, `ffn.gate.{weight,bias}`,
`ffn.shared_experts.*`, and 256 routed experts — identical shapes to a base layer. No compressor,
no indexer (all three are `compress_ratio = 0`). Stage-unique:

| tensor | dtype | shape | role |
|---|---|---|---|
| `mtp.0.main_proj.weight` | F8_E4M3 | `[4096, 12288]` | `cat(h.mean(2) @ layers 40,41,42) → 4096` |
| `mtp.0.main_proj.scale` | F8_E8M0 | `[32, 96]` | |
| `mtp.0.main_norm.weight` | BF16 | `[4096]` | |
| `mtp.2.norm.weight` | BF16 | `[4096]` | pre-lm_head norm for the draft |
| `mtp.2.hc_head_fn` | F32 | `[4, 16384]` | draft mHC contraction |
| `mtp.2.hc_head_base` | F32 | `[4]` | |
| `mtp.2.hc_head_scale` | F32 | `[1]` | |
| `mtp.2.markov_head.markov_w1.weight` | BF16 | `[129280, 256]` | bigram embed |
| `mtp.2.markov_head.markov_w2.weight` | BF16 | `[129280, 256]` | bigram unembed |
| `mtp.2.confidence_head.proj.weight` | BF16 | `[1, 4352]` | `4096 + 256 → 1`, declared f32 (`model.py:810`) |

`mtp.*` reuses `embed.weight` and `head.weight` (`model.py:897-898`).

### 6.4 Total bytes by dtype class

| dtype | tensors | bytes | share | what |
|---|---|---|---|---|
| `I8` (packed fp4 e2m1) | 35 328 | 148.176 GB | 88.8 % | routed expert w1/w2/w3, all 46 layers |
| `F8_E8M0` | 35 718 | 9.261 GB | 5.5 % | every scale tensor, fp4 and fp8 alike |
| `F8_E4M3` | 390 | 6.304 GB | 3.8 % | attention projections, shared experts, `main_proj` |
| `BF16` | 445 | 2.967 GB | 1.8 % | embed, lm_head, norms, compressors, `weights_proj`, gate, markov |
| `F32` | 433 | 0.151 GB | 0.1 % | all mHC params, `attn_sink`, `ape`, `gate.bias` |
| `I64` | 3 | 0.019 GB | 0.0 % | `tid2eid` |
| **total** | **72 317** | **166.879 GB** | | matches the index's `total_size` exactly |

By region: routed experts 147.17 GB · shared experts 1.08 GB · other per-layer 5.65 GB ·
global 2.12 GB · DSpark 10.86 GB.

---

## 7. Gap analysis against plow

Verdict key: **R** = reusable as-is · **A** = needs a new arm on an existing op (field, act code,
mode, dtype) · **K** = needs a new kernel body.

Counts over 40 rows: **15 R · 18 A · 7 K** (row 39 is two kernel bodies, and is optional — see §5.1).

| # | V4 requirement | nearest existing plow path | verdict |
|---|---|---|---|
| 1 | `q_a` → RMSNorm → `q_b` (`model.py:502-503`); `q_b` already carries the absorb | `emit_glm_mla` q-chain, `crates/devgen/src/mla.rs:3113` (`GemvQkv=22` fused a-side) / `mla.rs:3207` (`GemvQkv` q_absorb + q_rope); q_absorb derived at `mla.rs:1436` | **A** — plow *derives* `q_absorb` at weight-prep from `q_b_proj ⊗ kv_b_proj`; V4 ships it directly. Bind, don't derive. No fused `GemvQkv` for the (q_a, kv) pair is required — V4's `wkv` is one 512-wide output. |
| 2 | Unweighted per-(token,head) RMS over the full 512, no gain, **before** rope (`model.py:504`) | `RowRms(2)` (`dev.rs:91`); `HeadNormRope(3)` normalises with a gamma | **A** — `HeadNormRope`'s `i4=SKIP_NORM` path plus a gainless RMS, or a `RowRms` at `[T·64, 512]` between the GEMV and the rope. |
| 3 | Interleaved (GPT-J) rope on the last 64 of 512 | `HeadNormRope(3)`, `i5=pair_mode`; `0` at `hd==64` ⇒ interleaved (`dev.rs:96-100`); GLM MLA emits it unset at `mla.rs:3291` | **R** |
| 4 | YaRN(`factor=16`, `orig=65536`, `βf=32`, `βs=1`) on `theta=160000` for 41 of 43 layers; plain `theta=10000` for the other 2 | `RopeScale::Yarn` exists at `crates/packet/src/rope.rs:39`; **`mla.rs:1728` hard-codes `RopeScale::None` on the MLA path** | **A** — two rope tables per model, selected per layer. The math is already written and HF-verified (`rope.rs:65-100`); only the MLA emitter's call site is wrong. Note **no `mscale`**: `attention_factor` must be forced to 1.0 (`rope.rs:53` computes one). |
| 5 | Single 512-wide latent that is K and V; `D_qk == D_v == 512`; no `kv_b_proj`, no `O_UV_FOLD` | `FlashMlaDecode(50)` is `q_abs·c_kv + q_rope·k_rope` with a **separate** `krot` cache and PV on `DK` | **A** — V4 is the degenerate `DR=0, DK=512` case with the rope dims living *inside* the 512. plow refuses NoPE MLA today (`mla.rs:111-128`), because `HeadNormRope` is the only `krot` writer. Collapsing `krot` into `ckv` removes a cache, not adds one. |
| 6 | 128-entry **ring** window cache, slot = `pos % 128`, older KV discarded | `kv.{l}.ckv` is a contiguous `dbatch·ctx·kv_lora` cache, `mla.rs:2048-2055`; row addressing `mla.rs:3357-3369` | **A** — the ring is a modulus on the existing row index plus a fixed `ctx=128` allocation. Prefill's split write (`model.py:526-528`) is the only new addressing. |
| 7 | Growing compressed region concatenated after the ring in one cache | same tensor, second region (`model.py:497`) | **R** — a second declared tensor or an offset into the same one; no op change. |
| 8 | Gathered sparse attention over `window ∪ compressed`, `-1` = masked row | `FlashGatherDecode(54)` / `FlashGatherPrefill(55)`, `t7=idx`, `i6=select_width`, emitted `mla.rs:3467` | **A** — the gather semantics match; `D=512` for both QK and PV, and the index list is a plain union with no `IndexUnionPf(119)` needed. |
| 9 | Per-head learned **attention sink** in the softmax denominator | `FlashMerge(13)` `t3=sinks` (`dev.rs:178-186`), emitted only by GPT-OSS (`gptoss.rs:356`). **No sink slot on any MLA/gather op** | **A** — add a sink tensor slot to `FlashGatherDecode/Prefill`, or force a `FlashMerge` at `nsplit=1` (`gptoss.rs:349` already establishes that pattern). Note plow's `gm'=max(gm,sink)` (`dev.rs:180`) differs from the reference (`kernel.py:346`) — see Q4. |
| 10 | **Inverse** rope on the attention output's last 64 dims (`model.py:539`) | nothing — no op applies a conjugate rotation to an attention output | **K** — small elementwise kernel: `[T, 64, 64]` complex-conjugate rotate. |
| 11 | Block-diagonal `wo_a`: 8 × `[1024, 4096]`, group `g` reads `o[:, 8g:8g+8, :]` | `Gemv(10)` / `GemvFp8Blk(44)` at `mla.rs:3554` for a dense `o_proj`; no grouped/block-diag GEMV | **A** — 8 `Gemv`s at `N=1024, K=4096` with strided A, or one op with a group axis. Contrast `MlaMergeFold(57)`, which folds a *per-head* `W_uv`; here the fold is *per-group-of-8-heads* and is a checkpoint tensor, not derived. |
| 12 | `wo_b`: dense `[8192 → 4096]` fp8 block | `GemvFp8Blk(44)` / `GemmFp8Blk(107)` | **R** (subject to #23) |
| 13 | mHC pre: RMS-scaled 24-logit split, sigmoid gates, softmax + 20 Sinkhorn iters | `HyperConnPre(128)`, `runtime/amd/op_hyperconn.h:53-143`; `GemvF32(135)` for `mixes` (`mla.rs:477`) | **R** — term-by-term identical at `n=4, hidden=4096, repeat=20, eps=1e-6`, including the hard-coded `PLOW_HC_POST_MULT = 2.0` (`op_hyperconn.h:43`). Emitter must assert `hc_mult == 4`. |
| 14 | mHC post: `Σ_j comb[j,k]·res[j] + post[k]·x` | `HyperConnPost(129)` mode 0, `op_hyperconn.h:173-187` | **R** — same index convention (`comb[i][j]`, `i` = source). |
| 15 | Entry expand 1 → 4, ungated broadcast (`model.py:915`) | `HyperConnPost(129)` mode 1, `op_hyperconn.h:156-161`, emitted `mla.rs:558` | **R** |
| 16 | DSpark taps: `h.mean(dim=2)` at layers 40/41/42 (`model.py:920`) | `HyperConnPost(129)` mode 2, `op_hyperconn.h:163-171`, emitted `mla.rs:689` | **R** |
| 17 | Tower exit: **learned gated** 4 → 1 with its own `[4,16384]` projection (`model.py:709-717`) | mode 2 is an arithmetic **mean** — wrong function | **A** — a mode 3 on op 129 taking `pre` from a `GemvF32(135)` over `hc_head_fn`, plus a `[1]` scale and `+hc_eps`. |
| 18 | Learned-pooled compressor, `d=512`, per-channel softmax over 128 slots, `ape[128,512]`, RMSNorm, rope at block-first position, fake-fp8 epilogue | `DsaPoolCompress(130)` pools **indexer keys** at `d=128`, `pool=4`, no post-norm, rope-before-pool, fp8-per-vector epilogue | **K** — different input (a dedicated `wkv` projection, not a cached key), different width, different pool size, no Hadamard, a learned RMSNorm, a different quant epilogue, and the output feeds the *attention* KV cache rather than an index cache. |
| 19 | Overlapped ratio-4 compressor: 8 slots from 2 projections, prev-block half + cur-block half | absent — `DsaPoolCompress` pools `pool_size` contiguous slots only | **K** — the `overlap_transform` (`model.py:313-320`) and the 8-row decode ring (`model.py:344-354`) have no analogue. |
| 20 | Compressor decode state ring (`kv_state`/`score_state`, f32, `-inf` init) fired on `(pos+1)%ratio==0` | `DsaPoolStash(132)` is a bf16 `pool_size`-slot ring; `DsaPoolCompress` decode-mode gates on `pos` (`dev.rs:1573-1577`) | **A** — the gating idiom is exactly right; the state is f32, twice as tall for overlap, and `-inf`-initialised. |
| 21 | Indexer: `Σ_h relu(q·k)·w[h]`, 64 heads, `d=128`, over fp4-fake-quantised q/k | `IndexScoreKpool(134)` (`dev.rs:1693`) — same expression, but fp8 q/k with per-vector scales, and **`index_heads==32` / `index_dim==128` are hard-asserted** at `mla.rs:184` and `mla.rs:191` | **K** — the fp4 fake-quant means the dot is over bf16 values with e2m1 magnitudes and per-32 e8m0 scales already folded in, not an fp8 dot with a per-vector rescale. Different inner loop. `HI=32` is also baked into `interp.hip`. |
| 22 | Indexer q: Hadamard-128 then fake-fp4 (`model.py:420-422`) | `DsaQQuant(133)` — Hadamard-128 (same `1/sqrt(128)`) then **fp8** with a pow-2 scale (`dev.rs:1644`) | **A** — the Hadamard half is identical; swap the quant epilogue for e2m1/blk-32 and keep the bf16 round-trip. |
| 23 | Pool-granular top-`k` = 512 over `n_pools` | `IndexSelect(59)` with `i2=pool_size` (`dev.rs:630`, `dev.rs:635`; emitted `mla.rs:3855` at `itk/index_kpool`) | **R** |
| 24 | Selected pool ids used **directly** as KV indices (`+offset`) | `DsaPoolExpand(131)` expands pools → token ids (`mla.rs:3869`) | **R by omission** — V4 does not need op 131 at all. Emit the pool ids plus a constant offset into the gather list. |
| 25 | HCA: index list = `arange((pos+1)//128) + offset`, no scoring | nothing; the gather list always comes from a selector | **A** — a trivial generated i32 ramp; `crates/packet` `GenTensor` machinery can produce it, or a `ZeroF32(147)`-class fill op. |
| 26 | Router: `sqrt(softplus(logit))` | `MoeRouterTopk(56)` / `MoeRouterTopkPf(83)` flags bit0 = sigmoid, else softmax (`op_moe.h:230,242`) | **A** — a third scoring code in the flags word. Everything downstream (bias-on-selection-only `op_moe.h:264`, unbiased gate `:272`, `norm_topk` `:279`, `route_scale` `:280`, lowest-id tie-break) already matches V4 exactly. |
| 27 | Flat top-6, no group limiting | `i6=n_group`, `i7=topk_group`, inert at ≤1 (`mla.rs:6328-6329`) | **R** — set both to 0. |
| 28 | Router logit in **fp32** (`model.py:570`) | split router rounds the logit to bf16 in the GEMV before op 56 | **A** — `GemvF32(135)` already exists for precisely this (`dev.rs:1697-1712`); point op 56 at an f32 logit tensor. |
| 29 | Hash routing: `tid2eid[token_id] → 6 expert ids`, `[129280,6]` I64 | **absent** — every router in the tree is a logit GEMV + top-k | **K** — a small op that writes `routing_table[k] = (eid, gate)` from a LUT gather plus the score-gathered gate. Or fuse into op 56 as a mode with `t=tid2eid` and `t=input_ids`. |
| 30 | Clamped SwiGLU: `silu(min(g,10))·clamp(u,±10)` | `ACT_SWIGLU_OAI=3` gives `A(g)=min(g,L)σ(αmin(g,L))`, `B(u)=clamp(u,±L)`**`+1`** (`dev.rs:115-123`) | **A** — new act code 4 across the GLU family (5/19/20/45/47/48/85/150/152). At `α=1` the gate half is already right; only the `+1` must go. |
| 31 | Routed experts: fp4 e2m1 `[out, in/2]` + e8m0 `[out, in/32]`, low nibble = even k, LUT `{0,.5,1,1.5,2,3,4,6}` | `MoeGluMx(150)` / `MoeDownMx(151)` / `MoeGluMxPf(152)` / `MoeDownMxPf(153)` — **byte-identical disk layout** (`dev.rs:914`, `dev.rs:1766`) | **R** (weights), with a load-time concat of the separate `w1`/`w3` into `[E][2I][K/2]` and `i4=1` (blocked layout). |
| 32 | Expert activation quantised to fp8 e4m3 per-128-K pow-2 (**w4a8**) | ops 150-153 are **w4a16**; A4W4 exists only on `MoeGroupGluPf(85)` `enc=2` (`mla.rs:1050-1053`) | **K** — an fp8-activation arm on the fp4 expert GEMM. The scales already have the right shapes; the MFMA path differs. |
| 33 | Shared expert: fp8 e4m3, block `[128,128]`, **e8m0** scale | `DenseGluFp8Blk(47)` + `GemvFp8Blk(44)`, both taking an **f32** `[N/128][K/128]` grid (`dev.rs:480-486`) | **R** — the fp8-tensor scales total only **385 KB** across the whole checkpoint, so converting e8m0 → f32 at load is essentially free and needs no kernel change. (The 9.26 GB of e8m0 scales are the *fp4* expert scales, which ops 150-153 already consume as e8m0.) |
| 34 | All attention projections fp8 e4m3 block-`[128,128]` with e8m0 scales | same as #33 | **R** (same load-time conversion) |
| 35 | `wo_a` shipped fp8 but consumed bf16 (`convert.py:123-127`) | n/a | **A** — a load-time dequant, or keep it fp8 and use `GemvFp8Blk(44)` per group (`model.py:544-545` says the reference chose bf16 only "for simplicity"). |
| 36 | Fake-fp8 KV epilogue: quantise `dims 0..447` at block 64 with a pow-2 scale, dequantise, store | `HeadNormRopeFp8(37)` writes a *real* fp8 cache with one f32 scale per row (`mla.rs:3324`, `mla.rs:2064`) | **A** — same machinery, different block size (64 not the whole row) and a pow-2-rounded e8m0 scale. See Q3 for whether to store packed or round-tripped. |
| 37 | `MoE` = `Σ_k gate_k·expert_k(x) + shared(x)`, fp32 accumulate | `MoeCombine(43)` / `MoeCombinePf(87)`, fixed slot order | **R** |
| 38 | Routing weight applied to the intermediate, pre-`w2` (`model.py:608`) | `MoeDownMx(151)` applies it to the f32 down output (`dev.rs:1780`) | **R** algebraically (no `w2` bias); numerically different — see Q6. |
| 39 | 3 DSpark stages, block-parallel 5-token draft, non-causal within the block, Markov logit bias, confidence head | nothing in `devgen`; `crates/plowrt/src/orch/speculative.rs` is a runtime pipeline | **K** ×2 (the block program; the sequential Markov-bias sampling loop) — but **not required for correct base output** (§5.1). |
| 40 | Config front end: accept `deepseek_v4` | the refusal at `crates/nn-graph/src/models/config/mod.rs:180-184`; asserted by `crates/plowc/tests/model_metadata_compile.rs:693` | **A** — a `DeepSeekV4Config` variant. `DeepSeekConfig::validate` (`config/deepseek.rs:96-102`) refuses non-sigmoid/non-`[128,128]`, so it cannot be reused. |

**Blocking asserts to relax or route around**, all in `crates/devgen/src/mla.rs`:
`index_heads == 32` (`:184`), `index_dim == 128` (`:191`, satisfied), DSA crossover `ctx > 65536`
(`:154-155`), batched decode + DSA refused (`:3673`), MXFP4 + DSA refused (`:3686`), fp8-KV + DSA
refused (`:3441`), NoPE MLA refused (`:111-128`), `RopeScale::None` on the MLA path (`:1728`).

---

## 8. Open questions the reference does not settle

**Q1 — How many DSpark stages does the served model actually have?**
`config.json` says `num_nextn_predict_layers: 1`; `inference/config.json` says `n_mtp_layers: 3`;
the checkpoint ships `mtp.0/1/2` with stage-0-only and stage-2-only tensors in the right places
(`model.py:826`, `model.py:830`), and `mtp.0.main_proj.weight` is `[4096, 12288] = 4096×3`.
*Experiment:* none needed for correctness — the checkpoint is unambiguous and DSpark is optional.
Record the config field as wrong and read the stage count from the tensor inventory.

**Q2 — Which `index_topk` granularity does a server-side config mean?**
V4's `index_topk: 512` is pool-granular (`model.py:435` bounds it by `end_pos // ratio`);
GLM-5.3-Flash's `index_topk: 2048` is token-granular and plow divides by `index_kpool`
(`mla.rs:3846`). A shared config struct will get this wrong in one direction or the other.
*Experiment:* assert `index_topk * compress_ratio <= max_position_embeddings` and refuse otherwise;
cross-check the first decode step's selected-pool count against a reference trace.

**Q3 — Store the KV cache packed or round-tripped?**
The reference stores `bf16(f32(q)·s)` (`kernel.py:84-91`). Storing `(q, s)` at 583 B/entry instead
of 1024 B (§1.7) is bit-identical **only if** the attention kernel reconstructs
`round_bf16(q·s)` before the dot; dequantising straight to f32 is more accurate and *not*
bit-identical.
*Experiment:* run both against a reference trace of `sparse_attn` output at a 64 K prompt and
measure max-|Δ| on `o`; if the f32 path's drift is below the bf16 ulp of the reference's own
output, take the f32 path and record the deviation.

**Q4 — Does the sink's exclusion from the running max ever matter?**
`kernel.py:346` adds `exp(attn_sink[h] - scores_max[h])` where `scores_max` never saw the sink;
plow's `FlashMerge` uses `gm' = max(gm, sink_h)` (`dev.rs:180`). If any trained
`attn_sink[h]` exceeds the layer's typical max logit, the reference overflows and plow does not,
and the two disagree.
*Experiment:* dump all 43×64 `attn_sink` values and compare against `max(q·k)·512**-0.5` sampled
over a real prompt. If `sink < max` everywhere, the two forms agree to f32 and plow's is safe.

**Q5 — `tid2eid` dtype.** Checkpoint `I64 [129280,6]`; `model.py:562` declares `int32`. Harmless
for values < 2³¹, but a strict loader will reject it and the 19 MB is 2× what it needs to be.
*Experiment:* `max(tid2eid) < 256` (there are 256 experts) — cast to `u8` or `u16` at load and
assert the bound.

**Q6 — Where to apply the routing weight.** V4: fp32 `w·silu(g)·u`, then bf16, then `w2`
(`model.py:606-609`). plow's op 151: `gate · (W_d·fu + b)` in f32 after a bf16 `fu`
(`dev.rs:1780`). At `w ≈ 1/6 · 1.5`, the two round differently.
*Experiment:* per-expert max-|Δ| on the MoE output for one layer at both orderings against a
reference trace; if below 1 bf16 ulp of the combined output, keep plow's ordering.

**Q7 — `num_speculative_tokens: 7` vs `dspark_block_size: 5`.** The README's vLLM invocation asks
for 7 speculative tokens from a module that drafts 5 (`model.py:824`, `model.py:854`).
*Experiment:* irrelevant until DSpark is served; when it is, drive the block program at 5 and
treat 7 as a serving-layer accounting artefact until a vLLM trace says otherwise.

**Q8 — What is the acceptance rule?** `forward_head` returns `(output_ids, logits, confidence)`
(`model.py:875`) and **nothing in the shipped code consumes `confidence`.** The threshold, and
whether acceptance is per-token or per-block, are not in the reference.
*Experiment:* none available from this tree — read it off a vLLM/SGLang `dspark` implementation,
or treat DSpark as out of scope (§5.1).

**Q9 — Prefill chunking and the compressor's carried state.** `Compressor.forward`'s prefill arm
(`model.py:331-348`) assumes `start_pos == 0` and stashes the incomplete tail block in
`kv_state`. A chunked prefill that calls it repeatedly with `start_pos > 0` would take the decode
branch, which handles `seqlen == 1` only (`model.py:351` indexes `ape[start_pos % ratio]` with a
scalar). The reference supports **whole-prompt prefill only**.
*Experiment:* derive the chunked form (carry `kv_state`/`score_state` across chunks and align each
chunk to a `ratio` boundary) and validate against a whole-prompt run on the same prompt; the
ratio-128 layers are the hard case because a chunk shorter than 128 produces no compressed entry.

**Q10 — Is the compressor's `wkv`/`wgate` genuinely fp32-critical?**
`model.py:303-304` declares them `dtype=torch.float32` and `model.py:328` comments "compression
need fp32", but the checkpoint ships them bf16. So the *weights* are bf16-precise and only the
*arithmetic* is fp32.
*Experiment:* run the pooling softmax + weighted sum in bf16 and in fp32 over a real prompt and
compare selected-pool sets on the ratio-4 layers (the indexer's ranking is what a precision loss
would flip, exactly as plow's `GemvF32` note records for GLM at `dev.rs:1697-1712`).

**Q11 — Which layers may share an indexer?** GLM has `indexer_types: ["full","shared",…]`
(`config/glm.rs:63`) and plow honours it (`mla.rs:90`, `:200`). V4 gives every ratio-4 layer its
own indexer with its own compressor cache — 21 independent `S/4 × 128` caches. Whether adjacent
CSA layers select near-identical pools (and could share) is a measurement, not a code fact.
*Experiment:* dump the 21 selected-pool sets at several positions on a long prompt and measure
pairwise Jaccard; sharing is only sound if it is ~1.0.

---

## Appendix — config vs. code disagreements, collected

| # | `config.json` says | the reference code / checkpoint says | resolution |
|---|---|---|---|
| 1 | `num_nextn_predict_layers: 1` | `n_mtp_layers: 3` (`inference/config.json`); `mtp.0/1/2` ship; `main_proj` in-dim `4096×3` | **3** |
| 2 | `quantization_config.weight_block_size: [128,128]` | applies to the **fp8** tensors only; the fp4 experts are **1×32** (`model.py:136-141`, scale shapes `[2048,128]`) | two quant schemes, one config field |
| 3 | `topk_method: "noaux_tc"` | flat `scores.topk(6)` (`model.py:580`); no `n_group`/`topk_group` anywhere | correction-bias only, **no group limiting** |
| 4 | `compress_ratios` has 46 entries | `num_hidden_layers: 43`; entries 43-45 index the DSpark stages (`model.py:894`) | 43 + 3 |
| 5 | `num_key_value_heads: 1`, `head_dim: 512` | one 512-wide latent that is simultaneously K and V; no `v_head_dim`, no `qk_nope_head_dim` | MQA in a 512-wide latent, absorbed at rest |
| 6 | `rope_theta: 10000` (single value) | two tables: `10000` no-YaRN for layers 0/1 + DSpark, `160000` + YaRN for layers 2-42 (`model.py:481-487`) | per-layer |
| 7 | `rope_scaling.factor: 16` (YaRN) | no `mscale` applied to the softmax scale (`model.py:470`) | V3's YaRN attention factor is **absent** |
| 8 | `scoring_func: "sqrtsoftplus"` | `F.softplus(scores).sqrt()` (`model.py:576`) | agree |
| 9 | (`tid2eid` not mentioned) | checkpoint `I64 [129280,6]`; declared `int32` (`model.py:562`) | dtype mismatch, benign |
| 10 | `expert_dtype: "fp4"` | routed experts fp4; **shared expert is fp8** (`model.py:632` takes `default_dtype`) | field describes routed experts only |
| 11 | `moe_intermediate_size: 2048` | `ModelArgs.moe_inter_dim` default is 4096 (`model.py:46`), overridden by both shipped configs | 2048 |
| 12 | (no `scale_fmt`/`scale_dtype`) | `inference/config.json` sets `scale_fmt: "ue8m0"`; `ModelArgs.scale_dtype` defaults to `"fp8"` (`model.py:42`) ⇒ e8m0 scales | matches the checkpoint's `F8_E8M0` |


---

## DeepSeek-V4-Flash-0731 on MI300X — tensor inventory and TP capacity plan

Derived 2026-09-07 from `/workspace/models/DeepSeek-V4-Flash-0731` (48 shards, 72,317 tensors,
`total_size` 166,878,536,440 B = **155.418 GiB**) by reading the safetensors HEADERS only — no
payload, no GPU. Regenerate the machine-readable half with

```bash
python3 scripts/dsv4_tensor_inventory.py --model /workspace/models/DeepSeek-V4-Flash-0731 \
    --out docs/amd/deepseek-v4-flash-tensor-inventory.json
```

Scope: the config type, the tensor inventory, and the capacity/sharding arithmetic. The
per-layer *dataflow* (what CSA actually computes, how the indexer scores, the mHC mixing order,
the DSpark step) is the architecture-spec section above. Where a memory
figure depends on a dataflow question that is still open, this file gives the bound both ways
and says which question decides it.

## 1. What plow does with this checkpoint today

`config.json` now **parses and validates** (`nn_graph::models::config::DeepSeekV4Config`); the
refusal moved to the *emit* path, where `DeepSeekV4Config::unimplemented()` prints the gap list
built from the checkpoint's own numbers. So `plowc --model .../DeepSeek-V4-Flash-0731` still
fails, but it fails with a checklist instead of a dead end at `model_type`.

`architectures: ["DeepseekV4ForCausalLM"]` is claimed BY NAME in `model_type()`'s fallback. It
used to fall through the `DeepseekV3`/`DeepseekV2` prefix arm; a V4 checkpoint reaching the V3
MLA builder would produce a plausible wrong model, because the two share the MoE shape and
nothing else about the attention block.

## 2. Layer taxonomy — `compress_ratios` reconciled

`compress_ratios` has **46** entries for **43** layers. The extra three are the DSpark blocks:
`43 + len(dspark_target_layer_ids)` = `43 + 3` = 46. `num_nextn_predict_layers` is 1 and does
**not** reconcile the length — it counts speculative iterations, not blocks. The three trailing
entries are `0`, and the checkpoint agrees: `mtp.{0,1,2}.attn` carries no `compressor.*`.

| kind | layers | count | ratio | `attn.compressor` | `attn.indexer` | router |
|---|---|---:|---:|---|---|---|
| **A** uncompressed | 0, 1 | 2 | 0 | absent | absent | `tid2eid` hash (layers 0-2) |
| **B** fine + indexed | 2, 4, … 42 | 21 | 4 | `wkv/wgate [1024,4096]`, `ape [4,1024]`, `norm [512]` | present | learned gate + bias (layer 2 is hash) |
| **C** coarse + local | 3, 5, … 41 | 20 | 128 | `wkv/wgate [512,4096]`, `ape [128,512]`, `norm [512]` | absent | learned gate + bias |
| **D** DSpark MTP | `mtp.0/1/2` | 3 | 0 | absent | absent | learned gate + bias |

Two facts here are checkpoint evidence, not config text:

* The indexer lives on the **fine (ratio-4) layers only** — 21 of them, exactly the even layers
  2..42. `config.json` has no per-layer indexer list, so the emit path must read this from the
  tensor index. That is one of the listed gaps.
* `num_hash_layers: 3` means layers 0, 1, 2 carry `ffn.gate.tid2eid [129280, 6] I64` — a
  token-id → expert-id table replacing the router — and carry **no** `ffn.gate.bias`. The bias
  is present on exactly the other 40 layers. Every layer is MoE; there is no
  `first_k_dense_replace` dense prefix.

Shard layout is perfectly regular: shard 1 = `embed.weight`; shard *k*+2 = layer *k*;
shard 45 = `norm.weight` + `head.weight` + `hc_head_*`; shards 46-48 = `mtp.0/1/2`.

## 3. Tensor inventory

Full grouped table (every name template, its instance indices, per-variant shape, dtype, bytes
and shard span) is NOT committed — it is a 64 KB machine dump regenerable in seconds by
the command above, and this document carries every figure derived from it.

### 3.1 Totals by storage dtype

| dtype | tensors | bytes | GiB | share | what it is |
|---|---:|---:|---:|---:|---|
| `I8` | 35,328 | 148,176,371,712 | 138.000 | 88.79 % | **FP4 payload**, nibble-packed 2/byte — the routed experts, nothing else |
| `F8_E8M0` | 35,718 | 9,261,408,000 | 8.625 | 5.55 % | **every scale grid**: MXFP4 microscales *and* the block-FP8 `ue8m0` grids |
| `F8_E4M3` | 390 | 6,304,038,912 | 5.871 | 3.78 % | block-FP8 projections + shared experts |
| `BF16` | 445 | 2,967,134,976 | 2.763 | 1.78 % | embed, head, norms, CSA/indexer compressors, routers, markov/confidence heads |
| `F32` | 433 | 150,966,520 | 0.141 | 0.09 % | mHC parameters, `ape` position tables, `attn_sink`, `gate.bias` |
| `I64` | 3 | 18,616,320 | 0.017 | 0.01 % | the three `tid2eid` hash-routing tables |
| **total** | **72,317** | **166,878,536,440** | **155.418** | | |

**Plainly, which is which:**

* **FP4 (MXFP4)** — the 256 routed experts on all 43 layers *and* on the 3 DSpark blocks, and
  only those. `w1/w3.weight` is `[2048, 2048] I8` = `[N, K/2]` for a logical `[2048, 4096]`;
  `w2.weight` is `[4096, 1024] I8` for a logical `[4096, 2048]`. Their scales are
  `w1/w3.scale [2048, 128] F8_E8M0` and `w2.scale [4096, 64] F8_E8M0` = `[N, K/32]`, one E8M0
  byte per **32** values along K. This is byte-for-byte the layout `DevOp::GemvMxfp4` already
  documents and the Kimi-K3 emitter already binds (`crates/devgen/src/mla/kimi_k3.rs:445`).
  `expert_dtype: "fp4"` in `config.json` is the only field that says so; the safetensors dtype
  is `I8` because safetensors has no fp4.
* **FP8 (block, e4m3, `ue8m0` scales at `[128,128]`)** — every attention projection
  (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`), the indexer's `wq_b`, the shared expert's
  `w1/w2/w3`, and DSpark's `main_proj`. Each has a sibling `.scale` of dtype `F8_E8M0` with
  shape `[ceil(N/128), ceil(K/128)]` — e.g. `wq_b.weight [32768,1024]` → `wq_b.scale [256,8]`.
  Note the scale grid is **E8M0, not F32**: unlike GLM-5.2/5.3's `weight_scale_inv`, these fold
  into the dequant exactly.
* **The ue8m0 scale grids** are therefore *all* the `F8_E8M0` tensors, and they serve two
  different encodings: `[N, K/32]` rows for MXFP4 experts, `[N/128, K/128]` grids for block-FP8.
  Distinguish them by their parent's dtype, never by their own.
* **BF16** — `embed.weight`, `head.weight` (untied, `tie_word_embeddings: false`), every
  `*norm.weight`, both CSA compressors (`attn.compressor.wkv/wgate` and
  `attn.indexer.compressor.wkv/wgate`), `attn.indexer.weights_proj`, `ffn.gate.weight`,
  `mtp.2.markov_head.markov_w1/w2`, `mtp.2.confidence_head.proj`. The compressors and the
  routers are *deliberately* not quantized.
* **F32** — `hc_*_fn/base/scale`, `attn.compressor.ape`, `attn.indexer.compressor.ape`,
  `attn.attn_sink`, `ffn.gate.bias`.

### 3.2 Totals by role

| role | tensors | GiB | share |
|---|---:|---:|---:|
| routed experts (MXFP4 + E8M0) | 70,656 | 146.625 | 94.34 % |
| output LoRA `wo_a`/`wo_b` | 184 | 2.875 | 1.85 % |
| query path `wq_a`/`wq_b`/`q_norm` | 230 | 1.617 | 1.04 % |
| shared expert | 276 | 1.078 | 0.69 % |
| `embed.weight` | 1 | 0.986 | 0.63 % |
| `head.weight` | 1 | 0.986 | 0.63 % |
| CSA compressor | 164 | 0.490 | 0.32 % |
| indexer | 147 | 0.256 | 0.16 % |
| mHC (per layer) | 279 | 0.135 | 0.09 % |
| DSpark markov + confidence heads | 3 | 0.123 | 0.08 % |
| routers (`gate.weight/bias/tid2eid`) | 92 | 0.107 | 0.07 % |
| KV path `wkv`/`kv_norm` | 138 | 0.090 | 0.06 % |
| DSpark `main_proj` | 2 | 0.047 | 0.03 % |
| norms, `hc_head_*`, `attn_sink` | 144 | 0.001 | 0.00 % |

DSpark as a whole (3 MTP blocks + markov + confidence + `main_proj`) is **10.12 GiB**, 6.5 % of
the checkpoint; the 43 main layers plus embed/head are **145.30 GiB**.

### 3.3 The shapes that pin the geometry

| tensor | shape | reads as |
|---|---|---|
| `attn.wq_a.weight` | `[1024, 4096]` | `hidden → q_lora_rank`, then `q_norm [1024]` |
| `attn.wq_b.weight` | `[32768, 1024]` | `q_lora_rank → num_attention_heads(64) × head_dim(512)` |
| `attn.wkv.weight` | `[512, 4096]` | `hidden → num_key_value_heads(1) × head_dim(512)`, then `kv_norm [512]` |
| `attn.wo_a.weight` | `[8192, 4096]` | `o_groups(8) × o_lora_rank(1024)` rows over a **per-group** input of `64/8 heads × 512 = 4096`; block-diagonal, not dense over 32768 |
| `attn.wo_b.weight` | `[4096, 8192]` | `o_groups × o_lora_rank → hidden` |
| `attn.indexer.wq_b.weight` | `[8192, 1024]` | `q_lora_rank → index_n_heads(64) × index_head_dim(128)`; shares `wq_a` with the main query path |
| `attn.attn_sink` | `[64]` F32 | one sink per query head |
| `hc_attn_fn` / `hc_ffn_fn` | `[24, 16384]` F32 | `hc_mult(4) × hidden(4096) = 16384` in; 24 = 16 (`4×4` mixing) + 4 + 4, split by `hc_*_scale [3]` |
| `hc_head_fn` | `[4, 16384]` F32 | the final collapse of the 4 residual streams |
| `mtp.0.main_proj.weight` | `[4096, 12288]` | concat of the 3 `dspark_target_layer_ids` hidden states → hidden |
| `mtp.2.markov_head.markov_w{1,2}` | `[129280, 256]` | `dspark_markov_rank: 256` |
| `mtp.2.confidence_head.proj` | `[1, 4352]` | `4096 + 256` |

`64 heads / 8 o_groups × 512 head_dim = 4096` matching `wo_a`'s stored in-features exactly is
the load-bearing check: it is what establishes that the o-LoRA is block-diagonal per group, and
therefore what caps clean tensor parallelism at 8.

## 4. Tensor-parallel sharding

Constraints, in the order they bind:

1. **`o_groups = 8` caps clean TP at 8.** A group's down projection reads one contiguous
   `heads_per_group × head_dim = 4096` slice of the attention output. A rank owning half a
   group owns rows whose input it only half has, so the group would need its own intra-group
   reduction. TP must divide `o_groups` ⇒ **TP ∈ {1, 2, 4, 8}**. (`64 heads / TP` being a
   multiple of the 8 heads per group is the same condition.)
2. **`num_key_value_heads = 1` forces the whole KV path to be REPLICATED.** `wkv [512,4096]`,
   `kv_norm [512]`, and the KV cache itself cannot be split by head, and every rank's query
   heads need the full 512-wide latent. This is 0.084 GiB of weights — irrelevant — but it is
   what makes the **KV cache per rank independent of TP**, which is the single most important
   fact in this section.
3. `num_attention_heads = 64` and `index_n_heads = 64` shard by head for any TP ≤ 8.
4. `n_routed_experts = 256` divides by 2/4/8 exactly (128 / 64 / 32 experts per rank under
   expert parallelism). Sharding `moe_intermediate_size = 2048` instead gives 1024 / 512 / 256
   columns per rank; the **bytes are identical**, only the collective differs (all-to-all
   dispatch vs. all-reduce), so the capacity table below holds either way.
5. `vocab_size = 129280` divides by 2/4/8 (64640 / 32320 / 16160) so `embed`/`head`/`markov_w*`
   can be vocab-parallel.

### 4.1 What is replicated, what shards

| shards by | GiB (whole model) | tensors |
|---|---:|---|
| expert | 146.625 | routed `w1/w2/w3` + scales, layers and MTP |
| o-group | 2.875 | `wo_a` (rows, and its per-group input), `wo_b` (in-dim) |
| vocab | 2.096 | `embed`, `head`, `markov_w1/w2` |
| q head | 1.438 | `wq_b`, `attn_sink` |
| moe intermediate | 1.125 | shared expert `w1/w2/w3`, `main_proj` |
| index head | 0.174 | indexer `wq_b`, `weights_proj` |
| **replicated** | **1.085** | `wkv`+`kv_norm` (0.090), `wq_a`+`q_norm` (0.180), CSA compressors (0.490), indexer compressors (0.082), routers incl. `tid2eid` (0.107), mHC (0.135), norms |

Replication is only **0.70 %** of the checkpoint, so per-rank weights track `total / TP` almost
exactly.

### 4.2 Weights per rank

| | weights GiB/rank | of 192 GiB | aggregate GiB | replicas on an 8-card host |
|---|---:|---:|---:|---:|
| TP1 | 155.418 | 80.9 % | 155.42 | 8 |
| TP2 | 78.251 | 40.8 % | 156.50 | 4 |
| TP4 | 39.668 | 20.7 % | 158.67 | 2 |
| TP8 | 20.376 | 10.6 % | 163.01 | 1 |

For calibration: the GLM-5.3 campaign (`docs/amd/glm53-mi300x.md` §3) put TP4 at
**178.6 GiB/rank of 192** and measured **181.75 GiB/rank live** — i.e. +3.15 GiB of runtime
over weights at `max_ctx=10240`, and that arrangement is described as tight. DeepSeek-V4-Flash
at TP4 is **39.7 GiB/rank**: 4.5× smaller. Weight capacity is simply not the binding constraint
here — the KV cache is.

### 4.3 KV cache — the part that does not shard

Per token per layer, in cache **elements** (multiply by 2 for bf16, 1 for an fp8 cache):

| layer kind | main latent | CSA compressed | indexer compressed | fixed per sequence |
|---|---:|---:|---:|---:|
| A (×2, ratio 0) | 512 | — | — | — |
| B (×21, ratio 4) | 512 *if retained* | `1024 / 4` = 256 | `256 / 4` = 64 | — |
| C (×20, ratio 128) | — (local only) | `512 / 128` = 4 | — | `128 × 512` window |

The open question is what the ratio-4 layers' indexer selects `index_topk = 512` of. Three
bounds, all bf16, per **sequence** (and per **rank**, since the KV latent is replicated):

| scenario | B/token | 4k | 32k | 128k | 1M |
|---|---:|---:|---:|---:|---:|
| **S2** indexer selects compressed entries (as read) | 15,648 | 0.06 GiB | 0.48 GiB | 1.91 GiB | **15.28 GiB** |
| **S3** indexer selects raw tokens ⇒ B keeps the full latent | 37,152 | 0.14 GiB | 1.14 GiB | 4.54 GiB | **36.28 GiB** |
| **S1** nothing is ever evicted (upper bound) | 57,632 | 0.22 GiB | 1.76 GiB | 7.04 GiB | **56.28 GiB** |

An fp8 cache halves every figure. S2 is the reading that makes `max_position_embeddings:
1048576` plausible on one card; S1 is the number to budget against until the architecture spec
settles it. The `+2.5 MiB` fixed sliding-window term is noise at every context.

With 8 GiB/rank reserved for activations, collectives staging and workspace — up from GLM-5.3's
measured +3.15 GiB, because the mHC residual stream is `hc_mult = 4` copies of hidden and the
CSA/indexer add per-layer scratch — the 1M-context sequence budget is:

| | free GiB/rank | 1M seqs per replica (S2 / S3 / S1) | 1M seqs across the 8-card host |
|---|---:|---|---|
| TP1 | 28.58 | 1 / **0** / **0** | 8 / 0 / 0 |
| TP2 | 105.75 | 6 / 2 / 1 | **24 / 8 / 4** |
| TP4 | 144.33 | 9 / 3 / 2 | 18 / 6 / 4 |
| TP8 | 163.62 | 10 / 4 / 2 | 10 / 4 / 2 |

Because the KV cache is replicated, **raising TP does not raise the host's total token
capacity** — it lowers it, by cutting the number of independent replicas faster than it frees
HBM. TP2 wins on aggregate 1M-context capacity in all three scenarios.

### 4.4 Decode bandwidth, for the other half of the trade

Weight bytes touched per decoded token, as stored (top-6 of 256 routed experts):

| attention | routed experts | shared + gate | CSA + indexer | mHC | lm_head | total |
|---:|---:|---:|---:|---:|---:|---:|
| 4.599 GB | 3.449 GB | 1.172 GB | 0.795 GB | 0.135 GB | 1.059 GB | **11.211 GB** |

Attention is 41 % of decode traffic — unusual, and a direct consequence of `head_dim = 512`
with 64 heads: `wq_b`, `wo_a` and `wo_b` are 33.5 MB each *per layer*. At 5.3 TB/s:

| | GB/rank/token | ms/token | roofline tok/s (single stream) |
|---|---:|---:|---:|
| TP1 | 11.211 | 2.115 | 473 |
| TP2 | 5.605 | 1.058 | 946 |
| TP4 | 2.803 | 0.529 | 1,891 |
| TP8 | 1.401 | 0.264 | 3,782 |

These ignore collectives; at TP8 the per-token all-reduce of a 4096-wide (×`hc_mult`) residual
twice per layer over 43 layers starts to matter, and expert parallelism at 32 experts/rank means
only 0.75 of the 6 selected experts are local on average, so small-batch decode is badly
imbalanced. TP4 is the last degree where both stay comfortable.

## 5. Verdict

* **Minimum viable TP = 1** for contexts up to 128k: 155.42 GiB of weights on a 192 GiB card
  leaves 28.6 GiB after reserve, which holds a 128k sequence under every scenario (7.04 GiB
  worst case) and a 1M sequence only under S2. It has no headroom for concurrency and is the
  slowest arrangement; use it only for single-stream evaluation.
* **Minimum viable TP for a 1M-context service = 2.** TP2 is the smallest degree that holds a
  1M sequence under the conservative S1 bound (78.25 + 56.28 + 8 = 142.5 GiB of 192), and it
  maximizes host-aggregate token capacity (4 replicas × 105.75 GiB of KV).
* **Recommended = TP4.** 39.668 GiB/rank of weights (20.7 % of the card), 144.33 GiB/rank free
  after reserve, 2 replicas on the 8-card host, 3 concurrent 1M sequences per replica even
  under S3 and 9 under S2, 2.803 GB/rank/token of decode traffic (~1.9k tok/s roofline), and
  every divisibility constraint satisfied exactly (`64/4 = 16` heads = 2 whole o-groups,
  `256/4 = 64` experts, `129280/4 = 32320` vocab rows). It also matches the topology the
  GLM-5.3 bringup already exercises on this host, so the collectives path is the measured one.
  Choose TP2 instead only when host-aggregate 1M-context throughput is the objective and
  ~1 ms/token single-stream latency is acceptable; choose TP8 only for latency-critical
  single-stream decode, where the ~2× over TP4 is worth losing half the aggregate KV capacity.
* **TP16 and above are not available.** `o_groups = 8` is the hard cap on clean sharding.

## 6. Do the weights need a prep pass?

**No — for everything the inventory covers.** Unlike GLM-5.2/5.3, which needs
`scripts/glm52_prep_lite.py` to write `derived.q_absorb` / `v_absorb` / `kv_a_latent` because
MLA absorption folds `kv_b_proj` into the query and value paths, DeepSeek-V4-Flash has:

* **no MLA absorption to precompute.** There is no `kv_b_proj`. `wkv` produces the single
  512-wide latent directly and `kv_norm` normalizes it in place; the query side is
  `wq_a → q_norm → wq_b` with nothing to fold across.
* **routed experts already in plow's MXFP4 layout** — `[N, K/2] I8` payload with `[N, K/32]`
  E8M0 scale rows, exactly what `DevOp::GemvMxfp4` reads and what the Kimi-K3 A4W4 path already
  binds. No repack, no dequant.
* **block-FP8 projections in the shape `fp8_scale_shape()` already computes**, with the scales
  in `F8_E8M0` rather than `F32` — a narrower, not a different, encoding.
* **compressors, routers and norms already in BF16.**

So the checkpoint is consumable as-is and this lane writes **no prep script**. Two caveats for
whoever builds the emit path:

1. The loader must accept `F8_E8M0` as a scale dtype for block-FP8 weights. GLM's grids are
   `F32`; if the block-FP8 binder hard-codes that, it needs one dtype added — a loader change,
   not a checkpoint change.
2. If the emit path ends up wanting the CSA `ape` tables or the mHC `hc_*_fn` matrices in BF16
   rather than F32, that is a 0.14 GiB derived-tensor pass and the
   `scripts/glm52_prep_lite.py` symlink-plus-`zz-derived-*` shadowing pattern applies verbatim.
   It is not needed today.

## 7. Gap list blocking emit

Printed by `DeepSeekV4Config::unimplemented()`:

1. The CSA compressor — ratios 4/128, a gated pooling with its own absolute-position table and
   its own `compress_rope_theta: 160000` distinct from `rope_theta: 10000`.
2. The lightning indexer's top-512 selection, 64 heads × 128 dim, on the ratio-4 layers only
   (a per-layer fact the config does not state).
3. `sqrtsoftplus` routing, plus the 3 hash-routed layers' `tid2eid` token→expert tables.
4. mHC residuals: `hc_mult = 4` streams, 20 Sinkhorn iterations, per-layer dynamic mixing.
5. MQA with `num_key_value_heads = 1` at `head_dim = 512`, and the grouped output LoRA
   (`o_groups = 8` × `o_lora_rank = 1024`).
6. MXFP4 routed experts beside block-FP8 `ue8m0` projections **in the same layer** — one
   `projection_weight_dtype` cannot describe the block.
7. The attached DSpark speculative module: 3 MTP blocks, markov rank 256, block size 5.


---

## DeepSeek-V4-Flash attention roofline and kernel plan — MI300X, TP2/TP4/TP8

Analytical, CPU-only. No GPU was leased for this document; every number below is either a
measured ceiling from `docs/amd/glm53-mi300x.md` or arithmetic on the checkpoint's
own geometry. Calculator: `scripts/v4_attn_roofline.py` (extends the GLM-5.3/Gemma-4/Kimi-K3
calculator — same ceilings, same Layer/Geo shape, and it re-emits the GLM/K3 rows so §4 comes out
of one program). Checkpoint tensor shapes: `scripts/v4_checkpoint_shapes.py`.

Comparable to `attention_roofline.md` (GLM-5.3 TP4/TP8, Gemma-4 31B, Kimi-K3, Llama-class) by
construction. §11 lists everything that is not a measurement on this host.

> **Update, 2026-09-08.** §6's Body D (`d_flash_mla_prefill_v2<512, 0, GATHER>`), Body F (the
> compressor) and Body B's 64-head half now exist and are hardware-tested on gfx942. **§6.5's
> BKV=48 proposal is REFUTED** — see the correction in place there: `plow_mfma_bf16_16x16`
> contracts K=32, not 16, and `BKV` is the PV pass's contraction length, so 48 is 1.5 MFMA issues
> and cannot tile it. BKV stays 32 on both arms. Bringup state, and what each body was verified
> against: **this document**.

---

## 0. Ceilings and formulas (stated once, used everywhere)

| ceiling | value | note |
|---|---:|---|
| HBM read, `BW` | **4100 GB/s** | measured 4112 on an 8 GiB reduction (`glm53-mi300x.md:14,26`); spec 5325 is not used |
| bf16 GEMV at decode shapes | 1434–2501 GB/s | `[6144,2048]` TP8 / `[6144,4096]` TP4 o_proj (`:18,19`); weight-stream context only |
| bf16 MFMA peak, `F` | **1307 TFLOP/s** | dense matrix |
| ridge `F/BW` | **318.8 FLOP/B** | AI below → memory-bound, above → compute-bound |
| fp32 VALU | 40.9 T lane-ops/s | 304 CU × 64 lanes × 2.1 GHz; softmax side-roof only |
| Infinity Fabric per GPU | 896 GB/s | the indexer all-reduce rides this; the KV path does not |
| L2 | 8 × 4 MiB per XCD | used for the compressed-stream residency notes in §6.2 and §6.7 |
| LDS per workgroup, gfx942 | 65,536 B | `runtime/amd/amd_arch.h:44` |
| flash-object `fa` LDS arena | 58,368 B | `runtime/amd/op_attention_common.h:3295`, `interp.hip:803` |

All quantities are **per rank** (one GPU's share). `h = 64/P` query heads per rank,
`hi = 64/P` index heads per rank (both `ColumnParallelLinear`).

Decode, one token, all attention layers of one kind on this rank:

```
rows_k(T)   = rows that kind attends to (table in §1)
KV_bytes    = L_k · rows_k(T) · row_B
FLOPs       = L_k · 2 · h · rows_k(T) · (qk + pv)      qk = pv = 512
AI          = FLOPs / KV_bytes = 2·h                   (V4: row_B = qk = pv = 512 lanes)
mem_roof    = KV_bytes / BW ;  cmp_roof = FLOPs / F
```

Indexer, one decode token:

```
idx_rows(T) = T/4                                       (all ratio-4 compressed positions)
idx_bytes   = L_C4 · (T/4) · (row_idx + 8)              +8 = score write + read back
idx_FLOPs   = L_C4 · 2 · hi · 128 · (T/4)
AI_idx      = 2·hi·128 / (row_idx + 8)
```

Prefill, full causal prefill of T tokens chunked at `C` (plow ladder 128/512/2048/8192):

```
FLOPs   = L_k · 2 · h · (qk + pv) · pairs_k(T)          pairs_k(T) = Σ_{s=1..T} rows_k(s)
bytes   = L_k · Σ_chunks [ reach_k(prefix_i, c_i) · row_B + c_i·(Q_row + O_row) ]
          reach_W    = min(prefix, c + 128)
          reach_C128 = min(prefix, c + 128) + prefix/128
          reach_C4   = min(prefix, c + 128) + prefix/4      (union bound, see §2)
          Q_row = O_row = h · 512 · 2
roof = max(mem_roof, cmp_roof)
```

Row bytes:

| cache | row_B bf16 | row_B quantized | source |
|---|---:|---|---|
| attention KV, one 512-wide row serving **both K and V** | **1024** | **583** = 448 e4m3 + 7 ue8m0 (block 64) + 64 bf16 rope | `model.py:466,512` (`act_quant(kv[...,:-rd], 64, ...)`) |
| indexer compressed KV, 128-wide | **256** | **68** = 64 fp4 + 4 ue8m0 (block 32) | `model.py:404` (`Compressor(..., 128, rotate=True)`), `kernel.py:186` `fp4_act_quant` |

The quantized column is the QAT-endorsed form. The shipped reference caches **bf16** and only
simulates the quantization in place ("We performed QAT here, kv could also use fp8 format,
though current implementation uses bf16", `model.py:532`), so bf16 is the day-0 number and the
quantized one is the prize.

---

## 1. Geometry, and what TP does and does not shard

### 1.1 Layer kinds

`compress_ratios` (config.json) has 46 entries for 43 main layers + 3 MTP/DSpark blocks
(`n_layers=43`, `n_mtp_layers=3`; `Transformer.__init__` builds `mtp[i]` at `layer_id =
n_layers + i`, `model.py:900-903`). Layers 0–42 are the served stack:

| kind | ratio | layers | cache (`model.py:479`) | rows attended per query |
|---|---:|---:|---|---|
| **W** | 0 | 2 (ids 0, 1) | 128-row window ring | `min(T, 128)` |
| **C4** | 4 | 21 (ids 2,4,…,42) | 128 window + `T/4` compressed | `min(T,128) + min(512, T/4)` |
| **C128** | 128 | 20 (ids 3,5,…,41) | 128 window + `T/128` compressed | `min(T,128) + T/128` |

The three MTP DSpark blocks are ratio 0 (`DSparkAttention.forward` asserts it, `model.py:752`)
and attend `min(win, start_pos+1) + block_size(5)` rows; they are on the speculative path only
and are excluded from the tables. Add them as three more W layers if MTP is armed.

Only the **C4** layers carry an indexer (`model.py:474-476`: `if self.compress_ratio == 4`), so
21 indexers, one per C4 layer, every layer, every token — there is no `index_topk_freq`
skip-and-reuse as in GLM-5.3.

### 1.2 The attention body is MQA at head_dim 512 with K == V

`wkv = Linear(dim, head_dim)` produces **one** 512-wide row per position (`model.py:466`), and
`sparse_attn(q, kv, ...)` (`model.py:533,538`) feeds that same tensor to both the QK gemm and the
PV gemm (`kernel.py:328,343`: `T.gemm(q_shared, kv_shared, acc_s, transpose_B=True)` then
`T.gemm(acc_s_cast, kv_shared, acc_o)`). RoPE is applied to the last 64 dims of the cached row
**before** caching (`model.py:511`) and to q (`model.py:506`), and an **inverse** RoPE is applied
to the output's last 64 dims after the kernel (`model.py:539`).

Consequences for the roofline: `qk = pv = 512`, per head per position `2·(512+512) = 2048` FLOP,
and `row_B = 1024`. Consequence for the kernel plan (§6): the rope strip lives *inside* the same
512 row rather than in a separate `Krope` buffer, so V4 maps onto plow's **`DR=0`** MLA arm, not
the `DR=64` one.

### 1.3 What TP shards — and what it replicates

| tensor | class in the reference | TP behaviour |
|---|---|---|
| `attn.wkv` → the 512-wide KV row | plain `Linear` (`model.py:466`) | **replicated** |
| `attn.compressor.wkv` / `.wgate` → the compressed KV | plain `Linear` (`model.py:303,304`) | **replicated** |
| `indexer.compressor.*` → the 128-wide index KV | plain `Linear` (`model.py:404` → `:303`) | **replicated** |
| `attn.wq_b`, `indexer.wq_b`, `weights_proj`, `wo_a` | `ColumnParallelLinear` (`:465,:468,:399,:400`) | sharded, `/P` |
| `attn.wo_b` | `RowParallelLinear` (`:469`) | sharded, `/P` |
| routed experts | `n_local_experts = 256/P` (`model.py:623`) | sharded, `/P` |
| shared expert | plain `Expert` (`model.py:632`) | replicated in the reference; plow should shard it (both readings priced below) |

**Every one of V4's three caches is replicated on every rank, exactly like an MLA latent.** TP
shards heads (FLOPs) and the weight stream; it does **not** shrink a single byte of attention
traffic. That is the same trap GLM-5.3 has at TP8 — but V4's absolute cache traffic is 54x
smaller at 128k, so the trap never springs inside the model's window at batch 1 (§7).

### 1.4 Weight stream per rank per token (decode)

From the checkpoint headers (`scripts/v4_checkpoint_shapes.py`), applying §1.3:

| term | bytes/rank/token | shard |
|---|---:|---|
| `wq_a` + `wkv` + `hc_attn_fn` + `hc_ffn_fn` + `ffn.gate`, ×43 | 541 MB | replicated |
| compressors: 21×16.78 MB (ratio 4, `[1024,4096]` ×2 bf16) + 20×8.39 MB (ratio 128) | 520 MB | replicated |
| indexer compressors: 21 × 4.19 MB | 88 MB | replicated |
| **replicated subtotal** | **1.104 GB** | — |
| `wq_b` + `wo_a` + `wo_b`, ×43 | 4.328 GB | `/P` |
| indexer `wq_b` + `weights_proj`, ×21 | 0.187 GB | `/P` |
| 6 of 256 routed experts × 43 (13.37 MB each, fp4 + ue8m0) | 3.449 GB | `/P` |
| shared expert × 43 (25.17 MB fp8) | 1.082 GB | `/P` if plow shards it |

| | TP2 | TP4 | TP8 |
|---|---:|---:|---:|
| weight stream, shared expert sharded | **5.628 GB** | **3.366 GB** | **2.235 GB** |
| weight stream, shared expert replicated (reference reading) | 6.169 GB | 4.178 GB | 3.182 GB |
| decode weight roof at 4100 GB/s | 1.373 ms | 0.821 ms | 0.545 ms |
| active FLOP/token/rank | 13.32 G | 7.28 G | 4.26 G |
| prefill linear-term roof, 8192-chunk | 10.19 µs/tok | 5.57 µs/tok | 3.26 µs/tok |

The 1.104 GB replicated floor is **49% of the TP8 stream** and is the single most TP-hostile fact
about this model: 520 MB of it is compressor `wkv`/`wgate` stored in **bf16** in the checkpoint
while every other projection is fp8. Requantizing the compressors to fp8 would take the
replicated floor to 0.80 GB and the TP8 stream to 1.94 GB (−13%). Flagged for the weights owner,
not the attention owner.

---

## 2. Decode attention — bytes and FLOPs per token per rank

All three tables below have **identical byte columns**: the caches are replicated (§1.3). Only
the FLOP and AI columns move with TP.

### 2.1 Bytes and rows (TP-invariant)

| T | W rows | W MB | C4 rows | C4 MB | C128 rows | C128 MB | IDX rows | IDX MB | **total MB** | mem roof µs | quantized µs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 096 | 128 | 0.26 | 640 | 13.76 | 160 | 3.28 | 1 024 | 5.68 | **22.98** | **5.6** | 2.8 |
| 32 768 | 128 | 0.26 | 640 | 13.76 | 384 | 7.86 | 8 192 | 45.42 | **67.31** | **16.4** | 6.2 |
| 131 072 | 128 | 0.26 | 640 | 13.76 | 1 152 | 23.59 | 32 768 | 181.67 | **219.28** | **53.5** | 18.0 |
| 262 144 | 128 | 0.26 | 640 | 13.76 | 2 176 | 44.56 | 65 536 | 363.33 | **421.92** | **102.9** | 33.6 |
| 1 048 576 | 128 | 0.26 | 640 | 13.76 | 8 320 | 170.39 | 262 144 | 1 453.33 | **1 637.74** | **399.4** | 127.7 |

"quantized" = fp8 attention rows (583 B) + fp4 index rows (68 B).

Large-T closed form: **`decode bytes/token/rank = 1546·T + 16.6 MB`**, of which `1386·T` is the
indexer's scan of the ratio-4 compressed stream and only `160·T` is the C128 attention body. The
W and C4 attention bodies are **constant** in T above 2048 (128 and 640 rows).

Cache **writes** are 50.9 KB/token (43 window rows + `21/4 + 20/128` compressed + `21/4` index
rows) = 0.01 µs. Never a factor; excluded from every table.

### 2.2 Arithmetic intensity and verdict, per TP

| kind | AI (FLOP/B) TP2 | TP4 | TP8 | verdict |
|---|---:|---:|---:|---|
| W, C4, C128 (all identical: `2·h`) | 64.0 | 32.0 | 16.0 | **MEM** at every T — 5.0x / 10.0x / 19.9x under the 318.8 ridge |
| IDX (`2·hi·128/264`) | 31.0 | 15.5 | 7.8 | **MEM** at every T — 10.3x / 20.6x / 41.0x under |

Every decode cell in the grid is memory-bound, and the margin *widens* with TP because the bytes
are fixed while the FLOPs shard. There is no context and no TP degree at which V4 decode
attention is compute-bound.

### 2.3 Against the weight stream

| T | attn µs | TP2 (5.628 GB / 1373 µs) | TP4 (3.366 GB / 821 µs) | TP8 (2.235 GB / 545 µs) |
|---:|---:|---:|---:|---:|
| 4 096 | 5.6 | 0.4% | 0.7% | 1.0% |
| 32 768 | 16.4 | 1.2% | 2.0% | 3.0% |
| 131 072 | 53.5 | 3.9% | 6.5% | 9.8% |
| 262 144 | 102.9 | 7.5% | 12.5% | 18.9% |
| 1 048 576 | 399.4 | 29.1% | 48.7% | 73.3% |

Decode attention bytes equal the weight stream at **T = 3.63 M (TP2) / 2.17 M (TP4) / 1.44 M
(TP8)** — all outside the 1 048 576-token window. With the quantized caches, never. Compare
GLM-5.3, which crosses at 115k at TP8.

At batch `B` the attention bytes multiply by `B` and the weight stream does not, so the crossing
batch is:

| T | B_crit TP2 | TP4 | TP8 | TP8, quantized caches |
|---:|---:|---:|---:|---:|
| 4 096 | 244.9 | 146.5 | 97.3 | 194.6 |
| 32 768 | 83.6 | 50.0 | 33.2 | 87.5 |
| 131 072 | 25.7 | 15.4 | 10.2 | 30.3 |
| 262 144 | 13.3 | 8.0 | 5.3 | 16.2 |
| 1 048 576 | 3.4 | 2.1 | 1.4 | 4.3 |

Per-sequence cache footprint, **replicated on every rank** (this is what bounds `B` in practice
long before the roofline does):

| T | window MB | C4 compressed MB | C128 compressed MB | indexer MB | total GB | quantized GB |
|---:|---:|---:|---:|---:|---:|---:|
| 4 096 | 5.64 | 22.0 | 0.7 | 5.5 | 0.034 | 0.018 |
| 32 768 | 5.64 | 176.2 | 5.2 | 44.0 | 0.231 | 0.118 |
| 131 072 | 5.64 | 704.6 | 21.0 | 176.2 | 0.907 | 0.463 |
| 262 144 | 5.64 | 1 409.3 | 41.9 | 352.3 | 1.809 | 0.923 |
| 1 048 576 | 5.64 | 5 637.1 | 167.8 | 1 409.3 | 7.220 | 3.682 |

(Capacity arithmetic proper is the sibling agent's; quoted here only because it is what makes the
`B_crit` row above unreachable at 1M.)

---

## 3. Prefill attention

Chunked at 8192 (plow's `MAX_CHUNK`). Per-kind rows and the closed forms:

```
pairs_W(T)    = 8256 + 128·(T − 128)                         linear
pairs_C4(T)   = pairs_W(T) + Σ min(512, s/4)  →  640·T − 5.2e5   linear above T = 2048
pairs_C128(T) = pairs_W(T) + Σ (s/128)        →  128·T + T²/256   quadratic, coefficient 1/256
pairs_idx(T)  = Σ (s/4)                        →  T²/8            quadratic, coefficient 1/8
```

**The attention proper is nearly linear in T. The only genuinely quadratic term of any weight is
the indexer**, and it is quadratic at 1/8 density × 128 dims.

Derived, TP-independent because every term carries the same `h`:

| crossing | T | arithmetic |
|---|---:|---|
| indexer T² coefficient vs C128's | 4.2x larger | `21·2·h·128/8 = 672h` vs `20·2·h·1024/256 = 160h` |
| indexer overtakes the C4 attention body | **40 960** | `128T²/8 = 1024·640·T` |
| C128 body overtakes the C4 body | **172 032** | `20T²/256 = 21·640·T` |

### 3.1 TP4, per rank, chunked at 8192

| T | kind | TFLOP | bytes GB | AI | mem ms | cmp ms | roof ms | bound |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4 096 | W | 0.034 | 0.277 | 122 | 0.068 | 0.026 | 0.068 | MEM |
| | C4 | 1.437 | 2.929 | 491 | 0.714 | 1.100 | **1.100** | CMP |
| | C128 | 0.380 | 2.769 | 137 | 0.675 | 0.291 | 0.675 | MEM |
| | IDX | 0.180 | 0.710 | 254 | 0.173 | 0.138 | 0.173 | MEM |
| | **TOTAL** | 2.031 | 6.685 | 304 | 1.630 | 1.554 | **1.630** | MEM |
| 32 768 | W | 0.274 | 2.215 | 124 | 0.540 | 0.210 | 0.540 | MEM |
| | C4 | 14.064 | 23.702 | 593 | 5.781 | 10.761 | **10.761** | CMP |
| | C128 | 5.482 | 22.167 | 247 | 5.407 | 4.194 | 5.407 | MEM |
| | IDX | 11.544 | 17.022 | 678 | 4.152 | 8.833 | 8.833 | CMP |
| | **TOTAL** | 31.365 | 65.106 | 482 | 15.879 | 23.997 | **23.997** | CMP |
| 131 072 | W | 1.099 | 8.862 | 124 | 2.162 | 0.841 | 2.162 | MEM |
| | C4 | 57.358 | 99.044 | 579 | 24.157 | 43.885 | 43.885 | CMP |
| | C128 | 54.928 | 88.801 | 619 | 21.659 | 42.026 | 42.026 | CMP |
| | IDX | 184.715 | 204.435 | 904 | 49.862 | 141.328 | **141.328** | CMP |
| | **TOTAL** | 298.100 | 401.142 | 743 | 97.839 | 228.079 | **228.079** | CMP |
| 262 144 | W | 2.198 | 17.725 | 124 | 4.323 | 1.682 | 4.323 | MEM |
| | C4 | 115.082 | 209.364 | 550 | 51.064 | 88.050 | 88.050 | CMP |
| | C128 | 197.822 | 177.941 | 1 112 | 43.400 | 151.356 | 151.356 | CMP |
| | IDX | 738.866 | 772.465 | 957 | 188.406 | 565.315 | **565.315** | CMP |
| | **TOTAL** | 1 053.969 | 1 177.495 | 895 | 287.194 | 806.403 | **806.403** | CMP |
| 1 048 576 | W | 8.796 | 70.900 | 124 | 17.293 | 6.730 | 17.293 | MEM |
| | C4 | 461.428 | 1 108.048 | 416 | 270.256 | 353.044 | 353.044 | CMP |
| | C128 | 2 902.367 | 719.824 | 4 032 | 175.567 | 2 220.633 | 2 220.633 | CMP |
| | IDX | 11 821.926 | 11 816.160 | 1 000 | 2 881.990 | 9 045.085 | **9 045.085** | CMP |
| | **TOTAL** | 15 194.517 | 13 714.932 | 1 108 | 3 345.105 | 11 625.491 | **11 625.491** | CMP |

### 3.2 TP2 and TP8 totals

| T | TP2 TFLOP | TP2 roof ms | TP8 TFLOP | TP8 roof ms | TP2 attn/linear | TP4 attn/linear | TP8 attn/linear |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 096 | 4.062 | 3.124 (MEM) | 1.016 | 0.884 (MEM) | 0.075x | 0.071x | 0.066x |
| 32 768 | 62.729 | 47.995 | 15.682 | 11.999 | 0.144x | 0.131x | 0.112x |
| 131 072 | 596.199 | 456.159 | 149.050 | 114.040 | 0.341x | 0.312x | 0.267x |
| 262 144 | 2 107.938 | 1 612.806 | 526.984 | 403.202 | 0.603x | 0.552x | 0.472x |
| 1 048 576 | 30 389.035 | 23 250.983 | 7 597.259 | 5 812.746 | 2.175x | 1.990x | 1.701x |

Per-rank prefill attention halves exactly with TP (FLOPs shard with heads; the bytes are
dominated by the Q/O streams which also shard). It is the only V4 attention quantity TP helps.

### 3.3 Memory → compute crossovers, per layer kind

The three regimes really are three regimes, and only one of them ever leaves the memory roof at
short context:

| kind | per-query AI (unchunked, KV read once) | crossover T, TP2 / TP4 / TP8 (8192-chunk) |
|---|---|---:|
| **W** (sliding 128) | `2·rows·1024/(512·4) = 128`, **independent of T, h and TP** | **never** — the window body is pinned at AI ≈ 120–126 by its own Q read and O write, 2.5x under the ridge, at every context and every TP |
| **C4** (window + top-512) | `2·rows·1024/(512·4)`, rows → 640 → AI 640 | **1 664 / 1 792 / 1 792** |
| **C128** (window + T/128) | rows = 128 + T/128 → AI = `rows` | **50 432 / 51 712 / 54 400** |
| whole attention path incl. indexer | — | **4 224 / 5 120 / 9 984** |

Three things to take from this table.

1. The **sliding-window-128 layers never become compute-bound.** With only 128 KV rows per query
   and a full 512-wide Q read and O write per token per head, the ratio is fixed at
   `2·128·1024/(512·2·2) = 128 FLOP/B`. This is completely unlike a 128:1 compressed stream (C128,
   which crosses at ~50k) and unlike GLM's MLA prefill (crossover 672 and AI 15 000 at 32k). A
   kernel for the W layers should be written as a **bandwidth** kernel, not a matrix kernel — fuse
   it with the Q/O projections if the graph allows, because the KV is only 128 rows × 1024 B =
   128 KiB per layer and lives in L2 permanently.
2. The **C4 layers** cross at ~1.7k and then sit at a flat AI of ~500–600 — the same neighbourhood
   as GLM's MLA prefill at 4k, but flat in T instead of growing, because the row count is capped
   at 640. They never reach GLM's 15 000.
3. The **C128 layers** are the one place V4 behaves like a classical quadratic-attention model,
   and they do not become compute-bound until ~50k.

The whole-path crossover (4.2k / 5.1k / 10.0k) is **5–14x later than every other family in
`attention_roofline.md`** (which all land at 0.7–0.9k). That is a direct consequence of the
sliding-window layers and the low-density compressed streams.

### 3.4 Attention vs the linear term

The prefill attention roof overtakes the MoE/GEMM linear-term roof at **T ≈ 472k (TP2) / 516k
(TP4) / 616k (TP8)**. Inside 256k, attention is **≤ 60% of the linear term at TP2 and ≤ 47% at
TP8** — V4-Flash prefill TTFT is owned by the expert stream at every context this host will
realistically serve. For contrast, GLM-5.3 TP4's attention overtakes its linear term at 18.6k
(roofline) / 31.3k (measured).

### 3.5 Chunk ladder

Unlike GLM (where a 1024 ladder costs 8x the KV re-read of an 8192 one), V4's compressed streams
are so thin that the chunk size barely moves prefill bytes:

| T | TP | C=2048 GB | C=8192 GB | ratio |
|---:|---|---:|---:|---:|
| 32 768 | TP4 | 64.4 | 65.1 | 0.99x |
| 131 072 | TP4 | 414.6 | 401.1 | 1.03x |
| 262 144 | TP4 | 1 247.7 | 1 177.5 | 1.06x |
| 262 144 | TP8 | 1 051.7 | 981.5 | 1.07x |

and since every one of those cells is compute-bound, the roof is **identical** at both chunk
sizes. The chunk ladder for V4 is set by the expert stream (which is paid per chunk), not by
attention.

---

## 4. The indexer versus the dense and compressed paths

The indexer (`model.py:386-435`) is a second, independent 128-dim compressed cache with its own
`Compressor` (ratio 4, Hadamard-rotated, `model.py:404`), scored against every ratio-4 compressed
position by `einsum("bshd,btd->bsht")` (`model.py:426`), ReLU'd, weighted per head, summed over
heads, all-reduced across ranks (`model.py:429`), then top-512'd (`model.py:433`). It runs on all
21 C4 layers, every token.

### 4.1 What selection buys, on bytes alone

Against the alternative of reading the whole ratio-4 compressed stream densely (which is what a
C4 layer would do if it behaved like a C128 layer):

| T | dense ratio-4 stream GB | top-512 gather GB | indexer scan GB | sparse total GB | dense/sparse |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 0.0220 | 0.0110 | 0.0057 | 0.0167 | 1.32x |
| 32 768 | 0.1762 | 0.0110 | 0.0454 | 0.0564 | 3.12x |
| 131 072 | 0.7046 | 0.0110 | 0.1817 | 0.1927 | 3.66x |
| 262 144 | 1.4093 | 0.0110 | 0.3633 | 0.3743 | 3.76x |
| 1 048 576 | 5.6371 | 0.0110 | 1.4533 | 1.4643 | 3.85x |

**Sparse selection starts winning on bytes at T = 3 072** (bf16 index cache) / **2 560** (fp4).
It asymptotes at **3.85x**, and never better, because the indexer's own linear `T/4 × 264 B` scan
is 99.2% of the sparse cost at 1M. GLM's DSA reaches 12.9x at 128k for the same reason inverted:
its index row is 264 B against a 1152 B latent (4.4:1), where V4's is 264 B against a 1024 B row
that is already 4:1 compressed (1:1 in effect). **V4's indexer does not save bytes at scale; it
saves FLOPs and cache capacity.** The C4 attention body is capped at 640 rows regardless of T —
that is what the indexer buys. The striking ratio is inside the sparse column: at 1M the gather
fetches 11 MB while the scan that decides *which* 512 rows to fetch costs 1 453 MB —
**132x more to select than to fetch.**

### 4.2 The indexer's share of the decode roof

| T | indexer MB | attention MB | indexer share |
|---:|---:|---:|---:|
| 4 096 | 5.68 | 17.30 | 25% |
| 32 768 | 45.42 | 21.88 | 67% |
| 131 072 | 181.67 | 37.61 | 83% |
| 262 144 | 363.33 | 58.58 | 86% |
| 1 048 576 | 1 453.33 | 184.41 | 89% |

**Above ~16k the indexer *is* V4's decode attention cost.** Everything else is bounded: 0.26 MB
of window rows and 13.76 MB of gathered C4 rows, forever, plus a `160·T` C128 dribble. This is
the single most important structural fact for kernel work on this model, and it is the mirror
image of GLM, where the indexer is 80% of the *DSA* bytes but DSA itself is 12.9x cheaper than
the dense path it replaces.

The lever is the index row width. At fp4 (which the reference already simulates —
`fp4_act_quant(q, fp4_block_size, True)` at `model.py:422`, with the comment "use fp4 simulation
for q and kv in indexer") the row goes 256 → 68 B and the indexer term (key matrix + the 8 B/row
score write and read-back) goes 1 453 → 418 MB at 1M: **−31.6 µs/token/rank at 128k,
−63.1 µs at 256k, −252.4 µs at 1M**, at every TP degree.

### 4.3 Prefill

The indexer is the dominant prefill FLOP term above T = 41k (§3), and at 128k TP4 it is 141 ms of
the 228 ms attention roof (62%). Its scan is `T²/8` pairs at 2·128 FLOP each, and its bytes are
the `T/4 × 256 B` key matrix re-read per chunk plus the `T/4` score row per query.

### 4.4 The all-reduce

`dist.all_reduce(index_score)` (`model.py:429`) sums a `[seqlen, T/4]` fp32 score across ranks,
21 times per token. At decode:

| TP | bytes/token/rank at 1M | ring all-reduce time at 896 GB/s |
|---|---:|---:|
| 2 | 22.0 MB | 24.6 µs |
| 4 | 22.0 MB | 36.9 µs |
| 8 | 22.0 MB | 43.0 µs |

Against a 399 µs decode attention roof that is 6–11%, and it is 21 serialized collectives per
token. The alternative — **replicate the indexer** (all 64 index heads on every rank, no
collective) — costs `21 × (8192·1024 + 64·4096) × (1 − 1/P)` extra weight bytes per token:
90.8 MB (TP2) / 136.2 MB (TP4) / 159.0 MB (TP8) = 22 / 33 / 39 µs. At 1M the two are a wash on
bytes and replication wins on latency (no barrier, no collective, and it restores
`index_n_heads == 32`-sized MFMA tiles — see §5). Below ~256k the all-reduce is smaller
(5.5 MB → 9.2/10.7 µs at 256k) and sharding wins. Both readings are priced; the choice is a
serving-config decision, not a kernel one.

---

## 5. Comparison against GLM-5.3 and Kimi-K3, per rank

| model | ranks | h/rank | decode attn bytes/token/rank @128k | AI | mem roof µs | prefill attn TFLOP @32k | prefill roof ms @32k | weight stream GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V4-Flash** | TP2 | 32 | **0.219 GB** | 36.7 | **53.5** | 62.73 | 48.0 | 5.63 |
| **DeepSeek-V4-Flash** | TP4 | 16 | **0.219 GB** | 18.3 | **53.5** | 31.36 | 24.0 | 3.37 |
| **DeepSeek-V4-Flash** | TP8 | 8 | **0.219 GB** | 9.2 | **53.5** | 15.68 | 12.0 | 2.24 |
| GLM-5.3 (MLA) | TP4 | 16 | 11.778 GB | 30.2 | 2 872.6 | 1 458.00 | 1 115.5 | 17.70 |
| GLM-5.3 (MLA) | TP8 | 8 | 11.778 GB | 15.1 | 2 872.6 | 729.00 | 557.8 | 10.23 |
| Kimi-K3 (MLA layers) | TP8 | 12 | 3.624 GB | 22.7 | 883.9 | 336.46 | 257.4 | 16.78 |

**V4-Flash's decode attention is 54x cheaper than GLM-5.3's and 17x cheaper than Kimi-K3's MLA
layers, at 128k, per rank. Its prefill attention at 32k is 46x cheaper than GLM TP4's.**

What that means for kernel reuse:

| V4 body | closest existing plow body | cost profile |
|---|---|---|
| C4 / C128 / W decode | `d_flash_mla_decode<512, 0, GF, GATHER>` — **already instantiated** | same shape, ~54x fewer rows; latency- and launch-bound, not bandwidth-bound |
| C4 / C128 / W prefill | `d_flash_mla_prefill_v2<512, 64, …>` — needs a `<512, 0, …>` arm | same shape, 46x fewer FLOP at 32k; the flat-AI C4 regime is **new** (GLM's AI grows with T, V4's saturates at 640) |
| indexer decode | `d_index_score_mfma<128, 32>` — exact `DI`, `HIc` needs 32 heads/rank | GLM's indexer is 1/13 of its DSA bytes; **V4's is 83–89% of its total decode attention bytes**. Genuinely new: this is the term to optimize, not the flash body |
| indexer prefill + selection | ops 117 / 118 / 119 + `prefill_v2<…, GATHER=true>` — the whole pipeline exists | GLM: a *gate* armed above 64k. V4: **always on, on 21 of 43 layers**, and the dominant prefill FLOP term above 41k |
| compressor (gated pooling) | **nothing** | new body; see §6.6 |
| sliding-window-128 layers | `window` arm of the same decode/prefill bodies | AI pinned at 128 FLOP/B — the only prefill body in any plow model that is **never** compute-bound |

The one genuinely new cost profile is therefore the **indexer**, in both phases, and the
**compressor**. Both flash bodies are reuse.

---

## 6. Kernel implementation plan

### 6.1 The mapping that makes reuse possible

Plow's MLA kernels split the cache into `Ckv[DK]` + `Krope[DR]` and compute `QK = Qabs·Ckv +
Qrope·Krope` over `DK+DR`, `PV = P·Ckv` over `DK` only. Every dispatch site is hard-coded
`DK=512` with `DR=64` or `DR=0` (`interp.hip:1463,1470,1477,1523,1529,1535,1768,1773` for
`DR=0`; `2241,2314,2346,2357` for the V2 prefill at `DR=64`).

V4 needs `QK` over 512 and `PV` over the **same** 512 (§1.2). That is exactly
**`DK=512, DR=0`**: place the whole roped 512-wide row in `Ckv`, leave `Krope` unused. It is
correct only because V4 caches the RoPE'd row and RoPE's q before the kernel
(`model.py:506,511`) — it is not an approximation.

`d_flash_mla_prefill_v2`'s static asserts pass at `<512, 0>`: `D = DK+DR = 512`, `D % 32 == 0`,
`DK % 16 == 0` (`op_attention_common.h:3328`).

Two pieces of V4's math are **not** in the existing bodies:

* **`attn_sink`** — a learned per-head logit that enters only the softmax denominator
  (`kernel.py:346`: `sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`, no KV row and no
  contribution to `acc_o`). One extra term in the epilogue, or in `d_flash_merge` after the
  split-merge, since it commutes with the online-softmax rescale: `l += exp2(FA_SCALE·(sink_h −
  m_final))`. Cost: one exp per (query, head). Not a tile decision.
* **inverse RoPE on the output's last 64 dims** (`model.py:539`). A 64-wide rotation on the
  512-wide O, per head. Fold into the existing o-fold epilogue slot (`ofold` is already a
  parameter of `d_flash_mla_prefill_v2`, `op_attention_common.h:3321`) or run as a separate elementwise
  pass; 64 of 512 lanes touched, `h·64·2` B/token/layer read-modify-write = 0.09 MB/token/rank at
  TP4 — negligible either way.

### 6.2 Body A — decode, all three layer kinds

**Existing, no new instantiation.** `d_flash_mla_decode<512, 0, GF, /*GATHER=*/true>`
(`op_attention_common.h:2020`), already emitted at `GF ∈ {2,4,8}` (`interp.hip:1463,1470,1477`).

* Head groups: V4 has 32 / 16 / 8 heads per rank at TP2/4/8; `GF=8/4/2` respectively, all
  present.
* Work item is one query token; `idx[b*top_k + t]` is the decode-shaped gather
  (`op_attention_common.h:1964,4135`). `top_k` per layer kind: W = `min(T,128)`, C128 = `min(T,128) + T/128`,
  C4 = `min(T,128) + min(512, T/4)`.
* **The index list must be built for all three kinds, not just C4.** V4's cache is two disjoint
  segments — a 128-row window ring at `[0,128)` plus the compressed rows appended at
  `[128, 128 + T/ratio)` (`model.py:479`, and the concatenation at `model.py:520`). The existing
  `window` parameter models a contiguous ring, not ring-plus-append. Building the index list is
  the simpler route and costs nothing at decode: for W and C128 the list is **deterministic**
  (`get_window_topk_idxs` / `get_compress_topk_idxs`, `model.py:261,275`) and can be produced by
  one trivial kernel per step for all 43 layers, or hoisted to a host-side rotating table.
* Tile/LDS: unchanged from the shipped `DR=0` arm. Register budget unchanged.
* **Prize:** none directly — the decode roof is 53.5 µs at 128k / 399 µs at 1M, and every cell is
  memory-bound by ≥16x. This body is a correctness/enablement item.

### 6.3 Body B — decode indexer score

**Existing kernel, geometry gate.** `d_index_score_mfma<DI, HIc>` (`op_attention_common.h:4295`).
`DI = 128` matches V4's `index_head_dim` exactly. `HIc` carries
`static_assert(HIc == 32, "MFMA subtile assumes index_n_heads == 32")` (`op_attention_common.h:4302`).

| TP | index heads/rank | fit |
|---|---:|---|
| 2 | 32 | **exact** — `d_index_score_mfma<128, 32>` compiles and is fully utilized |
| 4 | 16 | needs `HIc=16` (half the 32-row A-tile wasted → 2x MFMA waste) **or** replicate the indexer and run `<128,32>` twice |
| 8 | 8 | needs `HIc=8` (4x waste) **or** replicate |

Recommendation: **replicate the indexer at TP4/TP8** (§4.4). It keeps the shipped `HIc=32` tile,
removes 21 all-reduces per token from the critical path, and costs 33 µs (TP4) / 39 µs (TP8) of
extra weight stream against a 37 / 43 µs collective at 1M. At short context, shard and accept
the 2x/4x tile waste — the absolute indexer FLOP roof at 128k TP8 is 1.08 µs, so 4x waste is
4.3 µs and the kernel is memory-bound anyway (AI 7.8).

**Byte lever, and this is the biggest decode prize in the model:** the score kernel streams the
`T/4 × 128` key matrix. In bf16 that term is 181.7 MB/token/rank at 128k and 1 453 MB at 1M
(key matrix + score traffic). In fp4 (what the reference simulates, `model.py:422`) it is
52.3 MB and 418 MB.
**−31.6 µs at 128k, −63.1 µs at 256k, −252.4 µs at 1M, per token, per rank, at every TP.** The kernel
change is an fp4 B-operand path in `d_index_score_mfma` plus an ue8m0 per-32 dequant in the
staging loop — the same shape as the existing `FP8` arm of the flash bodies.

### 6.4 Body C — decode top-512 select

**Existing.** Op 118's per-row exact top-k (`op_attention_common.h:4959`) and the op-59 radix run it
wraps. V4 wants top-512 of `T/4` candidates per query; GLM wants top-2048 of `T`. Same kernel
shape, smaller `k`, smaller candidate set (`T/4` at 1M = 262 144, versus GLM's 131 072 at 128k).
No change beyond the `k` and stride operands.

### 6.5 Body D — prefill flash, `d_flash_mla_prefill_v2<512, 0, GATHER, FP8>`

**New instantiation** of an existing template (`op_attention_common.h:3311`). This is the one body that
needs a real tiling decision, so here is the arithmetic.

Shipped constants: `RW = 16` q rows per wave, `BQ = 4·RW = 64`, `WG = 256` (four waves),
`BKV = FA_MLA_PF2_BKV = 32`, `FA_MLA_PF2_PAD = 8` (`op_attention_common.h:3324,3325,3197,3211`).

LDS, from `FA_MLA_PF2_LDS_BYTES(DK,DR) = (BKV·(DK+DR+PAD) + 3·SWZ + 4·16·BKV)·2`
(`op_attention_common.h:3297`), against the flash object's **58,368 B** `fa` arena
(`op_attention_common.h:3295`):

| arm | BKV | LDS bytes | fits 58,368 |
|---|---:|---:|---|
| `<512, 64>` (GLM, shipped) | 32 | 41,472 | yes (16,896 spare) |
| `<512, 64>` | 48 | 62,208 | **no** |
| **`<512, 0>` (V4)** | **32** | **37,376** | yes (**20,992 spare**) |
| **`<512, 0>` (V4)** | **48** | **56,064** | **yes** (2,304 spare) |
| `<512, 0>` | 64 | 74,752 | no |

Dropping the 64-wide rope strip frees 4,096 B of LDS, which is exactly enough LDS to take
**BKV from 32 to 48**.

> **CORRECTION, 2026-09-08 — BKV=48 DOES NOT WORK, and the LDS is not the reason.** The arm was
> built (`d_flash_mla_prefill_v2<512, 0>`, commit "Add the DR=0 MLA prefill arm") and BKV was
> left at 32. The LDS arithmetic below is right; the MFMA arithmetic under it is wrong.
>
> `plow_mfma_bf16_16x16` is not a K=16 issue. On CDNA3 it lowers to **two**
> `v_mfma_f32_16x16x16bf16_1k` over the two halves of a `bf16x8` fragment, so it contracts
> **K=32** per call (`runtime/amd/amd_arch.h:77-87`; CDNA4 takes the single
> `v_mfma_f32_16x16x32_bf16`, same K). **`BKV` is the contraction length of the PV pass** — the
> body reads one A-fragment `Pw[fr*BKV + kg*8]` covering `k = kg*8 + j`, `kg ∈ [0,4)`, `j ∈
> [0,8)`, i.e. exactly 32 KV columns, and issues one MFMA per output tile
> (`op_attention_common.h:4058-4071`). So the rule is **`BKV % 32 == 0`**, not `BKV % 16 == 0`: 48 is
> 1.5 MFMA issues and cannot tile the PV pass at all. The next legal depth is 64, and 64 needs
> 74,752 B against the 58,368 B arena — the row of this table that already says "no".
>
> The `static_assert(FA_MLA_PF2_BKV == 32)` the kernel carries is therefore **binding for the
> DR=0 arm too**, and for a second reason beyond the score tile it names: going deeper needs
> more score n-subtiles, a wider P strip, *and* a second PV MFMA issue. The freed 4,096 B of LDS
> buys nothing here.

The rest of the BKV=48 reasoning, preserved because the LDS half of it is still correct:

* The per-wave P strip `Pw[RW][BKV]` grows from 16×32 to 16×48 halves = +512 B per wave, +2,048 B
  for the workgroup — already counted in the 56,064 above.
* No new live accumulators: `P` is in LDS, not registers, so the 512-register occupancy-1 budget
  the V2 body already runs at (`op_attention_common.h:3219`) is untouched by the BKV change.
* Dropping to `DR=0` *reduces* register pressure independently. **Now measured** on the gfx942
  flash object: 256 VGPR / 256 AGPR / 1,464 B scratch / occ 1 with the NoPE arm compiled in,
  byte-for-byte the same cliff as without it. The arm is free, and the predicted spill relief
  does not show up because the object was already at the worst case over its other arms.
* The claimed gain — one third fewer slab barriers per KV row — is unavailable at any legal
  depth for this DK, so it is withdrawn.

Kernel-level re-stream AI (GLM report convention: one `BQ × BKV` slab's FLOPs over the LDS slab
bytes, slab = `BKV·(DK+DR+PAD)·2`):

| TP | BQ=64, BKV=32 (slab 33,280 B) | BQ=64, BKV=48 (slab 49,920 B) | BQ=32, BKV=32 |
|---|---:|---:|---:|
| 2 | 4 033 FLOP/B (12.7x ridge) | 4 033 (12.7x) | 2 016 (6.3x) |
| 4 | 2 016 (6.3x) | 2 016 (6.3x) | 1 008 (3.2x) |
| 8 | 1 008 (3.2x) | 1 008 (3.2x) | 504 (1.6x) |

(GLM-5.3 TP4 reference: 1 934 FLOP/B.) Every arm is above the ridge, so like MLA prefill this
body has no bandwidth excuse — but note **TP8 at BQ=32 is only 1.6x above the ridge**, so BQ must
stay at 64 at TP8. Keep `BQ = 4·RW = 64`.

Softmax strategy: keep the shipped online max/sum with the unconditional corr-rescale. A lazy
rescale was **tried and rejected twice** on this body — logits-different and slower
(`op_attention_common.h:3229-3236`); do not re-propose it as a guard. The `attn_sink` term (§6.1) rides
the same epilogue.

Split/merge contract: unchanged. `Opart[b][t][n_head][nsplit][DK] + mlpart[..][2]` unnormalized
partials, base-2 `FA_SCALE`'d `(m,l)`, consumed by `d_flash_merge<DK>` (`op_attention_common.h:1806`; operand contract at `:2425`).
V4 uses `DK=512` so `d_flash_merge<512>` is already instantiated. `nsplit` must be 1 for the
prefill arm (`op_attention_common.h:2429`); the causal KV-split `ns` (`op_attention_common.h:3369`) is the
dense-only work-multiplier and stays available for the C128 and W layers.

### 6.6 Body E — prefill sparse selection (C4 layers)

**The entire pipeline already exists** for GLM-5.3's DSA prefill and maps onto V4 without a new
kernel:

| stage | plow op | V4 fit |
|---|---|---|
| per-token indexer score | op 117 `d_index_score_pf` (`op_attention_common.h:4775`), or arm B `d_index_score_pf_row` (`:4864`) | `DI=128` exact; `HIc==32` gate as in §6.3 |
| per-row exact top-k | op 118 (`op_attention_common.h:4959`) | top-512 of `T/4` |
| per-tile union | op 119 `d_index_union_pf` (`op_attention_common.h:5063`) | u64 membership word ⇒ ≤64 queries/tile; V4's `tile_p=8` pack fits |
| gathered flash | `d_flash_mla_prefill_v2<512, 0, /*GATHER=*/true>` | **new instantiation only** |

The head-batched GATHER arm packs `QP = 8` queries × 8 heads into the 64-row M tile and is gated
on `n_head == 8` (`op_attention_common.h:3366-3367`). **V4 at TP8 has exactly 8 heads per rank — the
shipped pack fits with no change.** At TP4 (16 heads) and TP2 (32 heads) either set `QP = 4` /
`QP = 2` so `QP · n_head = 64` still holds, or run 2 / 4 head-batches per pack. Setting `QP` is
an i4 packet field on op 119 already (`op_attention_common.h:5067`), so this is an emitter change, not a
kernel one; the gate at the `prefill_v2` site is the part that must be widened.

The union's value is the whole point at V4's density: the measured GLM figure is 398 KV rows per
query for a union-of-8 versus 1 512 for the dense walk at a 16k prompt
(`op_attention_common.h:3362-3366`). V4's per-query set is 640 rows out of a `T/4` compressed stream, so
at 128k the dense walk would be 32 768 rows and the top-512 set is 1.6% of it — the union is
strictly more valuable here than it is for GLM.

**Note the shipped op-117 arm's operand re-fetch:** it reads both MFMA operands from global for
every (query, 32-key) item — 16 KiB of load per 8 MFMA, **16 FLOP/B**, VMEM-issue bound, not
fixable by L2 (`op_attention_common.h:4830-4832`). Arm B (`d_index_score_pf_row`) is the row-resident
form at 32 FLOP/B. Since V4's indexer is the dominant prefill FLOP term above 41k (§3), arm B is
the one to ship, and the fp4 key format (§6.3) doubles its intensity again.

### 6.7 Body F — the compressor (new)

Not an attention kernel, but it is on the attention path and nothing in plow does it. Gated
pooling over `ratio` consecutive tokens (`model.py:322-380`):

```
kv    = wkv(x)                      [ratio, coff·512] f32,  coff = 2 when ratio == 4 (overlap)
score = wgate(x) + ape              [ratio, coff·512] f32
out   = Σ_r kv[r] · softmax_r(score)[r]        softmax over the `ratio` axis, per dim
```

The two GEMMs are in the weight stream (§1.4). The pooling itself:

* **ratio 4, overlapping** (`model.py:296`, `overlap_transform` at `:313`): state is `2·ratio = 8`
  rows × `2·512` f32 = 32 KiB. Fits LDS trivially at any wave count. Decode form is a ring
  update of one row plus a reduce every 4th token; prefill form is a stride-4 segment reduction.
* **ratio 128**: a naive `[128, 512]` f32 tile is 256 KiB — **4x the gfx942 LDS cap**. Do it as a
  streaming online softmax instead: one pass to find the per-dim max over the 128 rows, one pass
  to accumulate `Σ kv·exp(score − max)` and the denominator. Live state is `2 × 512` f32 per
  workgroup = 4 KiB of LDS (or 8 VGPR/lane at 64 lanes × 512 dims / 64), and the 128 rows stream
  from L2 (128 × 1024 B = 128 KiB per layer per pool). Two passes over 128 KiB per 128 tokens per
  layer = 2 KiB/token/layer = 82 KB/token — negligible. This is the right form; the single-tile
  form does not fit and should not be attempted.
* The `ape` bias is `[ratio, coff·512]` f32 (`[4,1024]` or `[128,512]`, per the checkpoint
  headers) — 512 KiB for the ratio-128 layers, read once per pool, L2-resident.

### 6.8 Ranked by ms recovered per rank

**Decode** (roofs are TP-invariant; every cell memory-bound by ≥16x, so the lever is bytes, not
scheduling):

| rank | item | mechanism | −µs/token/rank @128k | @256k | @1M |
|---:|---|---|---:|---:|---:|
| 1 | **fp4 index cache** (§6.3) | 256 → 68 B/row on the 21-layer `T/4` scan | **−31.6** | **−63.1** | **−252.4** |
| 2 | **fp8 attention rows** (§0) | 1024 → 583 B/row on W+C4+C128 | −4.0 | −6.2 | −19.4 |
| 3 | indexer replication vs all-reduce (§4.4) | removes 21 collectives/token | +33 µs of weights vs −37 µs of IF at TP4 | | |
| 4 | everything else | — | ≤ 5 | ≤ 5 | ≤ 10 |

Items 1 and 2 together take the decode attention roof from 53.5 → 18.0 µs at 128k and
399.4 → 127.7 µs at 1M — **a 3.1x cut, and neither needs a new flash body.**

**Prefill** (roofs per rank, 8192-chunk; the roof is what is on the table, the achieved fraction
is unmeasured for V4 — see §11):

| rank | body | TP4 roof ms @32k | @128k | TP8 roof ms @32k | @128k | why it is worth it |
|---:|---|---:|---:|---:|---:|---|
| 1 | **indexer score (op 117 arm B + fp4 keys)** | 8.83 | **141.33** | 4.42 | **70.66** | dominant above T = 41k; shipped arm is 16 FLOP/B VMEM-bound |
| 2 | **C4 flash `<512,0,GATHER>`** | **10.76** | 43.89 | **5.38** | 21.94 | dominant below T = 41k; flat AI ≈ 550–600 |
| 3 | **C128 flash `<512,0>`** | 5.41 | 42.03 | 2.79 | 21.01 | overtakes C4 at T = 172k |
| 4 | W flash (sliding 128) | 0.54 | 2.16 | 0.28 | 1.11 | never compute-bound; a bandwidth kernel |
| 5 | BKV 32 → 48 on the `<512,0>` arm | — | — | — | — | −1/3 of the slab fences on bodies 2 and 3 |

At 32k the ordering is C4 > IDX > C128 > W; at 128k and above it is IDX > C4 ≳ C128 > W. Build
the flash `<512, 0>` arm first (it serves bodies 2, 3 and 4 at once), then the indexer.

Against the linear term: at TP4/32k the whole attention path is 24.0 ms against a 182.6 ms
MoE/GEMM roof (13%), and at TP4/128k it is 228 ms against 730 ms (31%). **No amount of attention
kernel work moves V4's TTFT below 256k by more than a third**, which is the opposite of GLM,
where attention owns TTFT above 18.6k.

---

## 7. TP recommendation for serving

### 7.1 The arithmetic

| | TP2 | TP4 | TP8 |
|---|---:|---:|---:|
| decode weight stream | 5.628 GB | 3.366 GB | 2.235 GB |
| decode weight roof | 1 373 µs | 821 µs | 545 µs |
| decode attention roof @128k (TP-invariant) | 53.5 µs | 53.5 µs | 53.5 µs |
| attention = 50% of the decode byte roof at | T = 1 809k | T = 1 078k | T = **712k** |
| attention = 100% (attention-bound) at | T = 3 630k | T = 2 166k | T = **1 435k** |
| batch at which attention = weight stream, T=128k | 25.7 | 15.4 | **10.2** |
| prefill attention overtakes the linear term at | T = 472k | T = 516k | T = 616k |
| replicated share of the weight stream | 19.6% | 32.8% | **49.4%** |

### 7.2 The recommendation

**TP8 is a win, not a trap — at batch 1.** V4's caches are replicated exactly as GLM's MLA latent
is (§1.3), but the absolute traffic is 54x smaller, so the point at which replication costs more
than the weight stream it saves lands at **T = 1 435k**, outside the model's 1 048 576-token
window. At every context the ladder will actually serve, TP8's halved weight stream wins
outright. GLM-5.3 crosses at 115k; V4 does not cross at all.

**TP8 becomes the wrong answer for long-context high-batch serving.** At 128k the crossing batch
is 10.2 at TP8 against 15.4 at TP4 and 25.7 at TP2; at 256k it is 5.3 / 8.0 / 13.3. Since the
weight roof also halves with TP, the correct statement is: above `B_crit`, adding ranks stops
buying throughput because the added ranks each re-read the same cache. With the fp4 index cache
(§6.3) `B_crit` at TP8/128k moves 10.2 → 30.3, which restores TP8's headroom to roughly what TP2
has without it. **Ship the fp4 index cache before recommending TP8 for batched long context.**

**The replicated floor caps how far TP is worth pushing.** 1.104 GB of the stream does not shard
(§1.4) — 49.4% of the TP8 stream. Going from TP4 to TP8 cuts the stream by 34%, not 50%.
A hypothetical TP16 would cut it by a further 25% only, and is anyway impossible: `o_groups = 8`
(`model.py:456`, `n_local_groups = n_groups // world_size`) and `n_heads = 64` with a 64-row MFMA
M tile put a hard ceiling at **P = 8**.

**Decode ladder.** Decode is weight-bound at every TP and every context in the window at batch 1.
Pick TP by capacity and by the batch you intend to run, not by context. The per-sequence
replicated cache is 0.91 GB at 128k and 7.22 GB at 1M (§2.3) on **every** rank, so at 1M the
concurrency is capacity-bound long before it is bandwidth-bound, and the fp4/fp8 caches
(3.68 GB at 1M) are worth 2x the concurrency.

**Prefill chunk ladder.** Attention is memory-bound below T ≈ 4.2k (TP2) / 5.1k (TP4) / 10.0k
(TP8) and compute-bound above (§3.3), and the chunk size changes prefill attention bytes by ≤ 7%
and the roof by 0% (§3.5). The chunk ladder is therefore a **pure expert-stream decision**: keep
`MAX_CHUNK 8192`, because the 256-expert stream is paid per chunk and the attention path does not
care. The one attention-side reason to prefer a large chunk is the union tile — a chunk shorter
than a few hundred tokens leaves op 119 with too few queries per pack to amortize the union.

---

## 8. Assumptions where the dataflow is ambiguous

Each of these changes a number; both readings are priced rather than one silently chosen.

1. **Shared expert sharding.** The reference replicates it (plain `Expert`, `model.py:632`).
   Sharded: TP2/4/8 stream 5.628 / 3.366 / 2.235 GB. Replicated: 6.169 / 4.178 / 3.182 GB. All
   attention-share percentages in §2.3 use the sharded reading; divide by the ratio above for the
   replicated one (attention's share at TP8/1M becomes 51.5% instead of 73.3%).
2. **Indexer sharded vs replicated** (§4.4, §6.3). Sharded: `hi = 64/P`, 21 all-reduces per token,
   `HIc ∈ {32,16,8}`. Replicated: `hi = 64`, no collective, `HIc = 32`, +33/+39 µs of weight
   stream at TP4/TP8. Both priced; the FLOP and AI columns in §2.2 use the sharded reading.
3. **Prefill C4 KV reach.** §0's `reach_C4 = prefix/4` assumes the union of a chunk's per-query
   top-512 sets covers the whole compressed prefix. It is an **upper** bound on bytes and is tight
   once `prefix/4 ≫ 512·(queries per union tile)`; at small `prefix` it over-counts. The affected
   cells are compute-bound anyway (§3.1), so the roof is unchanged.
4. **Expert parallelism vs tensor parallelism for the routed experts.** The reference partitions
   whole experts (`n_local_experts = 256/P`, `model.py:621`); plow would shard the intermediate
   dim. At batch 1 the two give the **same** per-rank byte count (`6/P` experts vs `6` experts at
   `1/P` width), so nothing in this document depends on the choice. It does change the tail: EP's
   per-rank active-expert count is a binomial, so the slowest rank streams more than `6/P`.
5. **Whether V4's `fp8` KV is the shipped format.** The reference caches bf16 and only simulates
   (`model.py:531`). Every table gives both columns.
6. **MTP/DSpark blocks.** Excluded (3 ratio-0 blocks, speculative path only). Including them adds
   `3 × 128 × 1024 = 0.39 MB` to every decode row and `3/2 ×` the W-layer prefill terms.

---

## 9. Formula index

| quantity | formula | where |
|---|---|---|
| decode rows | W `min(T,128)`; C4 `min(T,128)+min(512,T/4)`; C128 `min(T,128)+T/128` | §1.1 |
| decode bytes | `Σ_k L_k·rows_k·row_B + 21·(T/4)·(row_idx+8)` = `1546·T + 16.6 MB` | §2.1 |
| decode AI | attention `2·h`; indexer `2·hi·128/(row_idx+8)` | §2.2 |
| prefill pairs | `pairs_W = 8256+128(T−128)`; `pairs_C4 → 640T`; `pairs_C128 → 128T + T²/256`; `pairs_idx = T²/8` | §3 |
| prefill FLOPs | `L_k·2·h·1024·pairs_k(T)` | §0 |
| indexer/C128 T² ratio | `672h / 160h = 4.2` | §3 |
| indexer overtakes C4 | `128T²/8 = 1024·640·T` → `T = 40 960` | §3 |
| C128 overtakes C4 | `20T²/256 = 21·640·T` → `T = 172 032` | §3 |
| W-layer prefill AI | `2·128·1024/(512·2·2) = 128`, constant | §3.3 |
| kernel re-stream AI | `2·h·BQ·BKV·1024 / (BKV·(DK+DR+8)·2)` | §6.5 |
| V2 LDS | `(BKV·(DK+DR+8) + 3·SWZ + 4·16·BKV)·2` | §6.5 |
| weight stream | `REP + (SHARD + SHARED)/P` | §1.4 |
| batch crossing | `B_crit = weight_stream(P) / decode_bytes(T)` | §2.3 |

---

## 10. Reproducing

```
python3 scripts/v4_attn_roofline.py           # every table in §1–§6
python3 scripts/v4_checkpoint_shapes.py       # checkpoint dtype/shape inventory
```

---

## 11. Caveats — everything not measured on this host

* **No number in this document was measured on a V4 checkpoint on any GPU.** No GPU was leased.
  Every V4 quantity is arithmetic on `config.json`, `inference/model.py`, `inference/kernel.py`
  and the checkpoint's safetensors headers, against ceilings measured for *other* models.
* **The ceilings are measured; the model is not.** `BW = 4100 GB/s` (measured 4112),
  `F = 1307 TFLOP/s`, ridge 318.8, VALU 40.9 T lane-ops/s all come from
  `docs/amd/glm53-mi300x.md`, taken on this host with GLM-5.3 blobs.
* **No achieved-rate number for V4 exists.** §6.8 ranks by *roof*, not by measured time. The only
  achieved MLA-prefill rate on this host is GLM-5.3's **227 TFLOP/s = 17.3% of peak**
  (`glm53-mi300x.md`), on a `<512,64>` body at a different AI. Applying 17.3% to V4's
  roofs is an extrapolation across a different template arm, a different tile, a different
  softmax epilogue and a 46x smaller FLOP count; it is not done in any table above and should not
  be done casually.
* ~~**The BKV=48 proposal is not compiled.**~~ **RESOLVED, and REFUTED** — see the correction in
  §6.5. The k-tiling claim was wrong: `plow_mfma_bf16_16x16` contracts K=32, `BKV` is the PV
  contraction length, and 48 is not a multiple of 32. BKV stays 32 on both arms.
* ~~**The `<512, 0>` prefill_v2 arm does not exist.**~~ **RESOLVED** — it exists, is instantiated
  behind `PLOW_MLA_PF_NOPE_ARM` (`interp.hip`, `scripts/build_gfx942.sh`), and is verified on
  gfx942 by `runtime/tests/mla_prefill_nope_gfx942_test.hip`: bit-identical to `<512,64>` fed a
  zero rope strip on (Opart, m, l) over 9 dense/windowed/GATHER shapes, and within 0.0024 max /
  0.0011 rms of an f64 reference on `O/l`.
* **The `HIc = 16 / 8` indexer instantiations do not exist** and the 2x/4x MFMA-tile waste is
  computed from the tile geometry, not measured.
* **The weight stream is a byte count, not a trace.** It assumes every active weight is read
  exactly once per token per rank, no L2 reuse across layers, and that plow's emitter shards
  exactly as the reference's module classes do. The 1.104 GB replicated floor in particular
  depends on plow choosing to replicate the compressors and hyper-connection matrices, which is
  forced for the compressors (they produce a replicated cache) but is a choice for `hc_*_fn`.
* **The prefill "linear-term roof" is a roofline, not a measurement.** It is
  `max(chunk weight bytes / BW, 2·active params · C / F)` per chunk. For GLM the measured linear
  term is 6–10x off its roof (187.5 µs/token measured vs 19.2 µs roofline at 8192-chunks); if V4
  behaves the same way, the attention/linear ratios in §3.2 are **over**-estimates by that factor
  and attention matters even less than stated.
* **The indexer all-reduce cost** uses 896 GB/s Infinity Fabric and a `2(P−1)/P` ring factor. No
  collective of this shape (21 per token, `T/4` fp32 elements) has been measured on this host.
* **Chunked prefill byte counts** are compulsory traffic (each chunk reads its reachable KV once,
  Q once, O once). The per-query-block re-stream is reported separately in §6.5. Neither includes
  `wo_a`/`wo_b`, which are in the linear term.
* **`compress_ratios` has 46 entries for 43 layers.** The last three are read as the MTP/DSpark
  blocks per `Transformer.__init__` (`model.py:900-903`). If a future emitter maps them
  differently, the (2, 21, 20) split in §1.1 changes.
* **Sibling scope.** The exact architecture spec (above) and
  the config type + TP capacity arithmetic are being written in parallel by other agents; where
  this document derives geometry it does so only as far as the roofline needs, and the capacity
  table in §2.3 is quoted for context, not as the authority.


---

## DeepSeek-V4-Flash on MI300X — kernel bringup, 2026-09-08

What now exists on the device, what is verified and how, and what still stops a served run.
Companion to this document (the dataflow and the 40-row gap table) and
this document (the ranked kernel plan). Where this document and the
roofline disagree, this one has been compiled and run; §6 records the one place they do.

Every number below was measured on **gfx942 (MI300X), ROCm 7.14 from the nix dev shell**, under
a GPU lease. Nothing here has been served — see §7.

---

## 0. Headline

| | |
|---|---|
| Refusal-list lines with **no kernel** before this work | **7 of 7** |
| Refusal-list lines with **no kernel** after | **2 of 7** — w4a8 experts, and DSpark (optional) |
| New kernel bodies | **2** (`d_compress_pool`, `d_rope_inverse_o`) |
| New arms on existing bodies | **7** |
| New hardware tests | **4 files**, 39 cases, all passing |
| Negative controls run, all of which bite | **16** |
| Can it emit a graph? | **No.** The device is most of the way there; the emitter has not started. §7 |

The five closed lines are closed at the level the task set: *the kernel exists and has a test*.
None of them is reachable from a blob yet, because there is no `deepseek_v4` graph builder and
no `emit_deepseek_v4` in `devgen`. `unimplemented()` now says exactly that, in two halves that a
test (`emit_gap_list_names_every_unimplemented_piece`) keeps from drifting into each other.

---

## 1. `d_flash_mla_prefill_v2<512, 0, ...>` — the DR=0 MLA prefill

**Closes:** refusal line (5)'s attention half. **Files:** `runtime/amd/op_attention_common.h`,
`runtime/amd/interp.hip`, `scripts/build_gfx942.sh`,
`runtime/tests/mla_prefill_nope_gfx942_test.hip`.

V4 caches one 512-wide latent row per token that is simultaneously K and V, with the 64 rope
dims **inside** that row (`model.py:511` ropes `kv[..., -64:]` before the cache write;
`model.py:506` ropes q). QK and PV therefore both contract over the full 512 and there is no
separate `Krope` cache. That is `DK=512, DR=0`. Decode already had its DR=0 arm; prefill trapped
on the NoPE bit.

The template was DR-generic in its declarations and DR=64-specific in exactly three places, all
of them **constant divisions by zero** rather than dead code at DR=0: the DBUF fragment count
(`bf16v8 rl[9]` = 8 latent + 1 rope, hard-coded), its commit loop, and the non-DBUF rope staging
loop. Those are now `NLDF`/`NRDF` derived from `BKV*D/(WG*8)` and asserted exact, with the rope
halves under `if constexpr (DR > 0)`.

**Verification, two oracles:**

* **Bit-identity** against the shipped `<512,64>` arm fed a **zero rope strip**. At DR=64 the QK
  MFMA contracts the same 512 followed by 64 products of zeros, into the same f32 register in the
  same `kt` order; `x + 0.0f` is exact, so the right statement is exact agreement, not a
  tolerance. **0 of 532,480 words differ** on `Opart`, and 0 on `(m, l)`, across 9 shapes.
* **f64 CPU reference** on the normalized `O/l` (the frame cancels in the ratio, so the reference
  needs no knowledge of `FA_MLA_PF2_DEFER`'s exponent origin): worst **0.0024 max / 0.0011 rms**
  against a 2e-2 / 5e-3 band.

Shapes: dense at `n_tok` 1/64/65/96/130 (straddling the BQ=64 tile edge), `n_batch` 1 and 2,
`window` 0 and 128 (V4's kind-A layers), and the **GATHER** arm over a randomly selected per-query
row set packed into op 119's union layout at three densities.

**Negative control:** a 2% error in the reference's scores fails all 9 cases at 0.018–0.024, so
the passing margin is 20x. Getting there needed the query amplitude raised to 16 — at
`|q| ~ |kv| ~ 0.5` the softmax is near-uniform, the output barely depends on the scores, and a
**5% score error did not cross the tolerance**. `mla_ref.rs` records the same trap for the decode
fixture; it is easy to write a flash-attention test that silently stops testing the softmax.

**Dispatch** is build-gated (`PLOW_MLA_PF_NOPE=1` → `-DPLOW_MLA_PF_NOPE_ARM=1`), the policy
`PLOW_DSA_PF_ARM` already sets. Without the gate the NoPE bit still traps, so a blob that needs
the arm against an object that lacks it is a hard stop and never a silently wrong answer.

**Cost at the cliff: none.** gfx942 flash object, arm off vs on: **256 VGPR / 256 AGPR /
1,464 B scratch / occupancy 1** either way, unchanged from before this work.

---

## 2. `d_compress_pool` — the learned-pooling KV compressor

**Closes:** refusal line (1). **New body.** **Files:** `runtime/amd/op_compress.h`,
`runtime/tests/compress_pool_gfx942_test.hip`.

41 of V4's 43 layers own one, and each of the 21 ratio-4 layers owns a second, narrower one
inside its indexer. A compressed entry is a per-feature-**channel** softmax-weighted pool over
`ratio` consecutive tokens — `d` independent softmaxes over the slot axis, not one shared
attention weight — then a learned RMSNorm, an interleaved RoPE at the pool's **first** token, and
a blocked fake quant.

It is deliberately **not** an arm on `d_dsa_pool_compress` (op 130), the nearest-looking thing in
the tree. Op 130 pools the indexer *key* at d=128 with no post-norm, ropes *before* pooling, and
takes one fp8 scale for the whole vector. V4 pools a dedicated projection at d=512, norms after
pooling with a learned gain, ropes after the norm, and fake-quantizes in blocks of 64 (attention)
or 32 (indexer) with a power-of-two scale, round-tripping to bf16. Different input, width, order
and epilogue.

**Two things the kernel's shape turns on.**

*The overlap.* At `ratio == 4` the projections are `2*d` wide and each pool draws **eight** slots:
the previous block's tokens through channels `[0, d)` and the current block's through `[d, 2d)`
(`overlap_transform`, `model.py:313-320`), so every token contributes to two adjacent entries.
`ape` is added **before** that transform (`model.py:344`), so slot `s` takes
`ape[s % ratio][(s / ratio) * d + c]` — one map that also degenerates correctly to the
non-overlapped ratio-128 form at `coff == 1`. Block 0's overlap half resolves to a negative row
and is dropped, which is the `-inf` `overlap_transform` fills with.

*No `[ratio, d]` tile at any ratio.* The roofline (§6.7) asks for a streaming form because a
materialized `[128, 512]` f32 tile is 256 KiB, 4x the gfx942 LDS cap. It is streaming here for a
better reason: **the softmax is independent per channel**, so the thread owning channel `c` owns
that channel's entire reduction and the pooling needs no tile, no LDS and no cross-lane traffic
at all — at ratio 128 exactly as at 4. LDS enters only for the RMSNorm's cross-channel sum, the
ROTATE arm's Hadamard, and staging the row between epilogue stages.

**Precision is contract, not detail.** The reference rounds to bf16 after pooling
(`kv.to(dtype)`, `model.py:369`), after the norm (`RMSNorm.forward` ends `.to(dtype)`,
`model.py:202`), after the RoPE (`apply_rotary_emb` computes in f32 then `y.copy_(x)`,
`model.py:249`) and after the quant (`act_quant(..., inplace=True)`, `kernel.py:84-91`), and each
stage sees the rounded value. Skipping any round trip is *more accurate* and is not what the
checkpoint was QAT'd against.

**Verification: BIT-EXACT.** Against a host reference written from the Python in f64 that
reproduces every round trip and both quantizer ladders: **0 of 9,472 output channels off, worst
relative error 0.00e+00**, over

| case | ratio | coff | d | rd | qblk | epilogue |
|---|---:|---:|---:|---:|---:|---|
| HCA layers | 128 | 1 | 512 | 64 | 64 | fp8 e4m3, ue8m0 |
| CSA layers | 4 | 2 | 512 | 64 | 64 | fp8 e4m3, ue8m0 |
| indexer | 4 | 2 | 128 | 64 | 32 | Hadamard-128 then fp4 e2m1 |

each at `out_base = 0` and at a non-zero `out_base` (the chunked-write bug op 130 shipped with),
plus the decode ring arm. Three non-boundary decode steps write **zero** words, which is the
`(start_pos + 1) % ratio == 0` gate.

**Negative controls, seven, each breaking one thing:** dropping `ape` (8 cases fail), the overlap
half reading the current block (4 fail — the 4 `coff == 2` cases, and correctly *not* the HCA
ones), the wrong `ape` row (8), no RMSNorm gain (8), half-split instead of interleaved RoPE (8),
one bf16 round trip dropped (7), one quant scale instead of per-64 (2).

**Two earlier controls did not bite, and that is the lesson worth keeping:** patching the
softmax's **max** pass is a no-op, because the softmax is shift-invariant in its max and the
denominator normalizes it away. A control on a softmax has to break the weight pass.

---

## 3. The lightning indexer at 64 heads, and V4's fake quant

**Closes:** refusal line (2). **Files:** `runtime/amd/op_attention_common.h`, `runtime/amd/interp.hip`,
`scripts/build_gfx942.sh`, `runtime/tests/dsv4_indexer_gfx942_test.hip`.

*Decode (`d_index_score_kpool`, a `FAKEQ` arm).* GLM-5.3 stores indexer q and k as **real** fp8
with one power-of-two scale per vector, so the dot is over e4m3 bytes and the scales multiply
back afterwards. V4's `fp4_act_quant(..., inplace=True)` (`model.py:422`, and `rotate=True` in
the indexer's compressor at `model.py:404`) quantizes to e2m1 with a per-32 e8m0 scale and
multiplies **straight back**, storing the round trip in bf16 (`kernel.py:156-167`). By the time
the score kernel sees them there is nothing to dequantize: bf16 values carrying e2m1 magnitudes,
block scales already folded in, no per-vector scale anywhere. **The arm is simpler than the fp8
one** — it deletes both dequants and both scale multiplies. The einsum's bf16 output round trip
is kept, because `model.py:427`'s einsum has bf16 inputs and a bf16 result and the ReLU sees that
rounded value. The head count needed nothing: `index_heads` was already runtime here.

*Prefill (`d_index_score_pf_row`).* `HIc` generalized from `== 32` to a multiple of 32, walked as
`HG = HIc/32` head groups. The score is a plain sum over heads — `Σ_h w[h]·relu(q_h · k)` — so a
second head group is exactly a second 32-row A tile whose `part` adds into the first. The
epilogue reduce, the causal bound and the scale are shared; the arithmetic is untouched.

**Deliberately not generalized below 32.** `HIc` 16 or 8 — V4 at TP4/TP8 with the indexer
*sharded* — would idle half or three quarters of the A tile. The roofline's recommendation there
(§6.3) is to **replicate** the indexer and keep a full tile, and the `static_assert` now enforces
that rather than silently accepting a 4x-wasteful tile.

The op-117 dispatch reads `i[1]`, which the emitter has always written as `index_heads` and this
site has always ignored while baking 32. It now routes 64 under `PLOW_DSA_IDX64=1` and **traps**
on any other non-32 value.

**Verification** against an f64 host reference: decode FAKEQ at HI=64 (3.2e-6) and HI=32
(1.4e-6), with nothing written past the last **complete** pool; prefill row arm at HIc=64
(3.6e-5) and HIc=32 (4.2e-5), with nothing written past the causal bound. **Both 32-head cases
are regression checks** — the shipped fp8 decode arm (2.4e-5) and the shipped 32-head prefill
tile are re-measured in the same run against the same references and are unmoved.

One fixture bug worth recording because it fails *as a kernel regression*: generating random fp8
bytes as `rnd() & 0x7F` produces `0x7F`, which is the OCP e4m3 **NaN** code (e=15, m=7), not 480.
The host reference decoded it as finite and the case failed at rel 2.9e+02 with the kernel
entirely correct.

---

## 4. Router: sqrtsoftplus, f32 logits, hash routing

**Closes:** refusal line (3). **Files:** `runtime/amd/op_moe.h`,
`runtime/tests/dsv4_ops_gfx942_test.hip`. Three flag bits on `d_moe_router_topk` (op 56) and its
prefill wrapper, **all default off**, so every GLM / DeepSeek-V3 / Qwen / Mixtral / Kimi packet is
unchanged.

| bit | arm | note |
|---:|---|---|
| 2 | `sqrt(softplus(logit))` (`model.py:576`) | a third transform beside sigmoid and softmax — non-negative like sigmoid but **unbounded above**. Written `sqrt(log1p(exp(-abs(l))) + max(l,0))`: the exact function, and it cannot overflow the way `exp(l)` does past `l = 88`. |
| 3 | f32 logit | V4 computes the router logit in fp32 end to end (`model.py:570` upcasts the bf16 gate weight); rounding to bf16 first can flip the ranking, the argument `GemvF32` already exists for. The prefill wrapper's per-token row advance takes the **element width** into account — `logit + tok*n_exp` on a bf16 pointer lands halfway into an f32 row. |
| 4 | hash (`tid2eid`) | layers 0–2 have no scoring stage: `indices = tid2eid[token_id]`, `[129280, 6]` (`model.py:562,578`). The key-pack and rank machinery is skipped whole (`hashsel` is workgroup-uniform, so its barriers are not a divergence hazard) and the tail is untouched — still the **unbiased** score at `lds[wl[j]]`, still norm then `route_scale`. The gate comes from the score path, so the logit GEMV still runs. |

**Verification:** 7 cases against an independently written host router (transform,
bias-on-selection-only, lowest-id tie-break, `norm_topk`, `route_scale`): expert ids exact, gates
to 1.1e-7. Two are **regression cases** pinning the shipped sigmoid and softmax arms. One hash
case carries a bias to prove the bias cannot reach the gate.

**Negative controls:** softplus without the sqrt (5 fail), hash leaking bias into the gate (2).

---

## 5. mHC tower exit, inverse RoPE, clamped SwiGLU

**Closes:** refusal line (4), and line (5)'s output half.

*`head_only` on `d_hyperconn_pre` (op 121)* — **not** the mode 3 on op 122 the gap analysis
proposed. V4's `hc_head` (`model.py:709-717`) is
`pre[c] = sigmoid(mixes[c]·inv·scale + base[c]) + eps ; y[d] = Σ_c pre[c]·residual[c][d]`, which
is op 121's `layer_input` half **term for term**. The differences are all input shape:
`hc_head_fn` is `[n, n·hidden]` so `mixes` is `n` wide rather than `2n + n²`, and there is no
Sinkhorn, no `post`, no `comb`. Reading `mrow[n..n3)` would run off the end of a head row, so the
flag skips the lane-0 block entirely. Op 122 mode 2's arithmetic **mean** stays: it is the right
function for V4's *other* n→1 contraction, the plain `h.mean(dim=2)` DSpark target taps at
`model.py:920`.

*`d_rope_inverse_o`* — **a new body.** Nothing in plow applies a conjugate rotation to an
attention *output*. It is structural, not cosmetic: `o = Σ_j p_j·kv_j` mixes rows rotated by
their own positions, so de-rotating by the query's position leaves `Σ_j p_j R(j-i)·kv_j` — the
position-independent latent a fixed `wo_a` can consume (`model.py:539`). An implementation that
skips it builds a plausible wrong model. Interleaved (GPT-J) pairs, so one thread owns both
halves and no shuffle is needed.

*Act code 4, the clamped SwiGLU* (`Expert.forward`, `model.py:601-611`):
`up = clamp(up, ±L)`, `gate = min(gate, L)`, `out = silu(gate)·up`, `L = swiglu_limit = 10`.
A **pair** form, and it is **not** plow's existing act 3. Act 3 is GPT-OSS's `swiglu_oai`:
`A(g) = min(g,L)·σ(α·min(g,L))`, `B(u) = clamp(u,±L) **+ 1**`. At `α = 1` the gate halves agree
exactly; the `+1` does not, and it would be a bias on every one of V4's 256 routed experts plus
the shared one — plausible output, wrong model. So a fourth code, on both `moe_glu` (routed
experts) and a new `act_glu_pair` (`d_glu`, which the shared expert takes); the gate-only entry
points **poison** code 4 with a NaN rather than silently leaving `up` unclamped, the discipline
`situ` already established.

**Verification:** `hc_head` to 3.9e-3 with `post_mix` provably untouched; inverse RoPE to 3.9e-3
with the nope half provably unmoved and the roped half provably moved; act 4 to 1.7e-7 on
`moe_glu` and 3.8e-3 on `d_glu` (bf16 output) with 3,160 of 4,096 elements past the clamp on at
least one branch, and the gate-only poison confirmed NaN.

**Negative controls:** `hc_head` on `scale[1]` (1 fail), forward instead of conjugate rotation
(2 fail).

---

## 6. A correction to the roofline: BKV stays 32

§6 of this document.5 proposed taking `FA_MLA_PF2_BKV` from 32 to **48** on
the `<512,0>` arm, on the grounds that dropping the rope strip frees 4,096 B of LDS and that
`48 = 3 × MFMA_K(16)`. **The LDS half is right and the MFMA half is wrong**, and the arm shipped
at BKV=32.

`plow_mfma_bf16_16x16` is not a K=16 issue. On CDNA3 it lowers to **two**
`v_mfma_f32_16x16x16bf16_1k` over the halves of a `bf16x8` fragment, so it contracts **K=32** per
call (`runtime/amd/amd_arch.h:77-87`; CDNA4 takes the single `v_mfma_f32_16x16x32_bf16`, same K).
**`BKV` is the contraction length of the PV pass** — the body reads one A-fragment
`Pw[fr*BKV + kg*8]` covering `k = kg*8 + j` for `kg ∈ [0,4)`, `j ∈ [0,8)`, i.e. exactly 32 KV
columns, and issues one MFMA per output tile. So the rule is `BKV % 32 == 0`, not `BKV % 16 == 0`:
48 is 1.5 MFMA issues and cannot tile the PV pass at all. The next legal depth is 64, and 64 needs
74,752 B against the 58,368 B arena.

The kernel's own `static_assert(FA_MLA_PF2_BKV == 32)` is therefore binding for the DR=0 arm too,
and for a second reason beyond the score tile it names: going deeper needs more score n-subtiles,
a wider P strip, **and** a second PV MFMA issue. The freed LDS buys nothing here. The roofline has
been amended in place.

Two other roofline caveats are now resolved rather than corrected: the `<512,0>` prefill arm
exists and is tested, and the predicted spill relief from `DR=0` **does not appear** — the flash
object was already at its worst case over its other arms, so the arm is free rather than helpful.

---

## 7. What still blocks a served run

In the order they bite.

1. **There is no `deepseek_v4` graph builder.** `crates/nn-graph/src/models/` has no
   `deepseek_v4.rs`, and `build_graph` still returns `unimplemented()`. Every kernel above is
   reachable only from its own test binary. This is the single biggest remaining item and it is
   emitter work, not kernel work.
2. **There is no `emit_deepseek_v4` in `devgen`.** `mla.rs`'s `GlmCfg` is shaped for
   `DK + qk_nope + qk_rope + v_head`; V4 is one 512-wide latent that is K and V. The pieces the
   emitter needs and does not have: **two rope tables selected per layer** (θ=1e4 no-YaRN for
   layers 0/1, θ=1.6e5 + YaRN(16, orig 65536) with `attention_factor` forced to 1.0 for layers
   2–42 — `mla.rs:1728` hard-codes `RopeScale::None` on the MLA path); the **128-entry window
   ring** (`slot = pos % 128`, with prefill's split write at `model.py:526-528`); the **grouped
   output LoRA** (`wo_a` is block-diagonal, 8 independent `[1024, 4096]` GEMVs each consuming a
   different 4096-slice of `o`); the **attention sink** in the softmax denominator (no MLA or
   gather op has a sink slot today); and the **HCA index ramp** (`arange((pos+1)//128) + offset`,
   a generated i32 list with no selector behind it).
3. **`index_heads == 32` is still asserted in `crates/devgen/src/mla.rs:180-186`.** The kernel
   now takes 64 and the interpreter routes it, so this assert is the only thing left refusing —
   but it is untouched here, so no blob can reach the arm yet. Moving it needs the emitter to
   also decide shard-vs-replicate at TP4/TP8 (§3).
4. **w4a8 for the routed experts.** `MoeGluMx(150)` / `MoeDownMx(151)` read V4's fp4 expert
   layout byte-identically, but they are **w4a16** and the reference is **w4a8** — fp8 e4m3
   activations with a per-128-K power-of-2 scale (`model.py:117-119`, `kernel.py:442-516`). This
   is a real MFMA-path change and the one refusal-list line that is neither done nor optional. It
   also needs a load-time concat of the separate `w1`/`w3` into the `[E][2I][K/2]` gate|up buffer.
5. **Loader work, none of it hard but none of it done.** The e8m0→f32 conversion for the fp8
   tensors' block scales (only ~385 KB across the checkpoint, so free); `tid2eid` narrowed from
   the checkpoint's I64 to i32 with a `< n_routed_experts` bound assert; `wo_a` dequantized from
   its shipped fp8 to bf16 (`convert.py:123-127`), or kept fp8 and run through `GemvFp8Blk`.
6. **Capacity.** this document recommends TP4. Nothing here changes that,
   and no serve has been attempted.

**Not blocking:** DSpark. `Transformer.forward` is self-contained and the shipped `generate.py`
never calls `forward_spec`, so a first bringup is bit-exact on base output with
`dspark_block_size = 0`. It is 6.5% of the checkpoint.

---

## 8. Reproducing

All under `nix develop` (the ROCm 7.14 toolchain is mandatory — `/usr/bin/hipcc` on this host
execs a `clang++` that does not exist), and all under a GPU lease.

```
H=$PLOW_HIPCC; I="-Iruntime/amd -Iruntime/common"
for t in mla_prefill_nope compress_pool dsv4_ops dsv4_indexer; do
  $H --offload-arch=gfx942 -O3 -w -std=c++17 $I -c runtime/tests/${t}_gfx942_test.hip -o /tmp/$t.o
  c++ /tmp/$t.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/$t
  GPU_LEASE_TIMEOUT=1800 perf-data/tools/gpulease -n 1 v4-$t /tmp/$t
done
```

`-Wl,-rpath` rather than `LD_LIBRARY_PATH`: `op_dsa_pool.h`'s header records why (`sg render`
execs through a setuid `newgrp`, glibc's `AT_SECURE` drops the variable, and the binary silently
falls back to the wrong `libamdhip64`).

Objects, to check the arms compile and what they cost:

```
scripts/build_gfx942.sh                      # unchanged; the new arms are all opt-in
PLOW_MLA_PF_NOPE=1 scripts/build_gfx942.sh   # + the DR=0 MLA prefill (flash object)
PLOW_DSA_IDX64=1   scripts/build_gfx942.sh   # + the 64-index-head score (prefill object)
```

---

## 9. Everything above that is NOT measured

* **No serve, no TTFT, no tokens/s, no perplexity.** Not one of these kernels has run inside a
  packet. The tests drive the same `__device__` bodies the interpreter inlines, which is what
  makes them meaningful, but a passing oracle is a correctness statement and nothing else.
* **No end-to-end numerics against the reference implementation.** Every oracle here is a host
  reimplementation of `model.py`/`kernel.py` written from the source. That catches a kernel that
  disagrees with the paper; it cannot catch the paper being misread the same way twice. The
  spec's open questions Q3 (packed vs round-tripped KV), Q4 (the sink's exclusion from the
  running max) and Q6 (where the routing weight is applied) all need a real reference trace and
  none exists on this host.
* **No performance number for any new body.** `d_compress_pool` in particular is written for
  clarity — one thread per block of 64 in the quant epilogue means 8 of 256 threads work there —
  and has never been profiled. The roofline says the compressor's two GEMMs dominate it and the
  pooling is ~2 KiB/token/layer, so this is very likely fine; "very likely" is not a measurement.
* **The `<512,0>` arm's register cost is measured, its speed is not.** Identical cliff numbers to
  the DR=64 arm say it is free to *carry*, not that it is fast to *run*.
* **`PLOW_DSA_IDX64_ARM` costs nothing at the cliff either, and that is measured** (gfx942
  prefill object: 256 VGPR / 0 AGPR / 1,248 B scratch / occupancy 2, arm off, on, and on with
  `PLOW_DSA_IDX_ROW` — identical in all three, object +3,424 bytes). Like the NoPE arm, this
  says it is free to *carry*; the HG=2 body's own throughput at HIc=64 is unmeasured, and the
  roofline's claim that arm B is the one to ship (§6.6) is still an argument, not a number.
* **gfx950 is untouched.** Everything here was built and run for gfx942 only.
