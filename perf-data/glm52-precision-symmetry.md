# GLM-5.2 TP4 — was the published plow-vs-vLLM comparison PRECISION-SYMMETRIC?

**Answer: no.** plow ran **58.3 % of its decode weight stream in bf16 that is fp8 in the checkpoint
file**. vLLM ran that same weight set in fp8. KV was bf16 on **both** sides, so KV is not the
asymmetry — but it is not symmetric-by-design either, and plow could not change it if it wanted to.

Everything below is marked **MEASURED** (a header read, a file:line, an execution of the engine's
own resolver, a logged number) or **INFERRED** (arithmetic on measured inputs). Nothing is read off
a feature flag: the whole point of this audit is that the flags and the reality diverged.

---

## 0. What is under test

`perf-data/glm52-plow-vs-vllm-tp4.md` and `/home/lava/models/glm52_verdict/` carry the headline:
GLM-5.2, TP4, MI355X, `vllm bench serve --backend openai-chat` against both engines, concurrency 1.
Warm-vs-warm medians from `/home/lava/models/glm52_verdict/{plow,vllm}_tp4.log` (MEASURED):

| ctx | plow TPOT med | vLLM TPOT med | ratio |
|---|--:|--:|--:|
| 4,096 | 28.26 | 24.97 | **1.132x slower** |
| 16,384 | 29.15 | 25.51 | **1.143x slower** |

The blob under those numbers is `/home/lava/models/glm52_verdict/tp4/` — `build.json` reports
`precision: {weight_enc: fp8, act_enc: bf16, kv_enc: bf16, expert_enc: fp8blk}`,
`features: {fp8_weights: true, fp8_kv: false, w8a8: false, mxfp4_weights: false}`,
`tuning: {gv_mm_max: 1}`. Its `checkpoint` symlink points at **`/home/lava/models/GLM-5.2-plow`**
(766,871,079,074 B), **not** at `GLM-5.2-plow-q`. That distinction turns out to decide the whole
question — see §2.

---

## 1. The per-tensor-class precision table

Column sources, each cited once here rather than in every cell:

* **HF** — safetensors header read of all 141 shards of
  `~/.cache/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541/`.
  Totals: `F8_E4M3` 751,226,191,872 B · `BF16` 4,207,458,304 B · `F32` (scale grids) 183,490,240 B.
* **plow disk** — header read of all 79 shards of `/home/lava/models/GLM-5.2-plow/`.
  Totals: `F8_E4M3` 725,647,884,288 B · **`BF16` 41,019,879,936 B** · `F32` 177,236,928 B.
  **bf16 grew 9.75x between the two files.** That is the whole finding in one number.
* **plow bind** — `crates/devgen/src/mla.rs`, the emitter, at the line cited per row.
* **vLLM bind** — vLLM `0.23.1.dev1+g9ddef7117` inside
  `rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0`; the layer-class decision
  is `Fp8Config.get_quant_method` (`fp8.py:176-205`), whose only branch is
  `is_layer_skipped(prefix, ignored_layers=modules_to_not_convert)`. That list was read from the real
  `config.json` and classified (MEASURED, §1.1).

| tensor class | HF checkpoint | plow on disk | plow binds | vLLM binds | plow wider than the file? |
|---|---|---|---|---|:--|
| `q_a_proj` `[2048,6144]` | **F8_E4M3** + F32 `[16,48]` | **BF16** | BF16, `t[2]` of `GemvQkv`(22) fusion A — `mla.rs:1264,1335,1900` | **fp8 W8A8-blk** (not skipped) | **YES, 2x** |
| `q_b_proj` `[16384,2048]` | **F8_E4M3** + F32 `[128,16]` | absorbed away | — | **fp8 W8A8-blk** | (see derived rows) |
| ↳ `derived.q_absorb` | (product of two fp8 tensors) | **BF16** | BF16, `t[2]` of `GemvQkv` fusion G — `mla.rs:1337,1936` | n/a — vLLM does not pre-absorb | **YES, 2x** (no fp8 form exists) |
| ↳ `derived.q_rope` | fp8 values verbatim, grid breaks on the 64-row slice | **BF16** | BF16, `t[4]` of `GemvQkv` — `mla.rs:1343,1942` | n/a | **YES, 2x** |
| `kv_a_proj_with_mqa` `[576,6144]` | **F8_E4M3** + F32 `[5,48]` | row-split away | — | **fp8 W8A8-blk** | (see derived rows) |
| ↳ `derived.kv_a_latent` | rows 0..511, fp8 verbatim, grid intact | **BF16** | BF16, `t[4]` of `GemvQkv` fusion A — `mla.rs:1349,1903` | n/a | **YES, 2x** |
| ↳ `derived.k_rope` | rows 512..575, fp8 verbatim | **BF16** | BF16, `t[6]` of `GemvQkv` fusion A — `mla.rs:1351,1905` | n/a | **YES, 2x** |
| `kv_b_proj` `[28672,512]` | **F8_E4M3** + F32 `[224,4]` | split away | — | **fp8 W8A8-blk** | (see derived rows) |
| ↳ `derived.v_absorb` | transpose of the V-slice; fp8 values verbatim, grid breaks | **BF16** | **BF16 unconditionally, every encoding** — `MlaMergeFold`(57) takes `const bf16*` with no encoding operand, `mla.rs:1353-1362,2209` | n/a | **YES, 2x** |
| `o_proj` `[6144,16384]` | **F8_E4M3** + F32 `[48,128]` | **BF16** | BF16, `t[2]` of `Gemv`(10) — `mla.rs:1363 else`, `:2246` | **fp8 W8A8-blk** | **YES, 2x** |
| `mlp.gate` (router) `[256,6144]` | **BF16** (no fp8 source) | BF16 | BF16, `t[2]` of `Gemv`(10) — `mla.rs:1369,3376` | **bf16, skipped** (all 76 in `modules_to_not_convert`); routing math in fp32 (`config.moe_router_dtype = float32`) | no |
| **shared experts** gate/up/down | **F8_E4M3** + F32 grids | **BF16** | BF16, `GemvGlu`(19) + `Gemv`(10) — `mla.rs:1379-1399 else`, `:3458`, `:3487` | **fp8 W8A8-blk** | **YES, 2x** |
| **routed experts** ×256 | **F8_E4M3** + F32 grids | **F8_E4M3 + F32, byte-verbatim** | **fp8blk**, `MoeExpertGluFp8Blk`(45)/`MoeExpertDownFp8Blk`(46), encoding in runtime field `i[6] = MoeEnc::Fp8Blk` — `mla.rs:538,3534,3558` | **fp8 W8A8-blk** (`Fp8MoEMethod`, block_quant) | no — **matched** |
| **dense FFN** (layers 0-2) | **F8_E4M3** + F32 grids | **F8_E4M3 + F32, verbatim** | **fp8blk**, `DenseGluFp8Blk`(47) + `GemvFp8Blk`(44) — `mla.rs:1413-1461,3681,3695` | **fp8 W8A8-blk** | no — **matched** |
| norms (input / post-attn / q_a / kv_a / final) | **BF16** | BF16 | BF16, `t[2]` of `RmsNorm` — `mla.rs:1334,1350,1368,1081` | **bf16, skipped** (all in `modules_to_not_convert`) | no |
| `embed_tokens` `[154880,6144]` | **BF16** | BF16 | BF16, `t[1]` of `Embed` — `mla.rs:1080,3866` | **bf16, skipped** | no |
| `lm_head` `[154880,6144]` | **BF16** (no fp8 source) | BF16 | BF16, `t[2]` of `Gemv`(10) — `mla.rs:1088,4018` | **bf16, skipped** | no |
| DSA indexer `wk` / `wq_b` | **F8_E4M3** + F32 grids | **F8_E4M3, verbatim** | fp8 — declared `mla.rs:~1470-1500`, but **NOT EMITTED**: the verdict `build.json` carries no indexer opcode | **fp8, LIVE** (`index_topk = 2048`) | n/a — see §4 |
| **KV cache** (MLA latent `[512]` + rope `[64]`, ×78 layers) | not a checkpoint property | — | **BF16** — `mla.rs:1250-1251` allocates `ctx*dk*BF16` / `ctx*dr*BF16`; `build.json kv_enc = "bf16"` | **BF16** — MEASURED, §3 | **no — symmetric** |
| **activations** | not a checkpoint property | — | **BF16** (`build.json act_enc = "bf16"`, `w8a8 = false`) | **fp8 e4m3, dynamic, per-token-group-of-128** — `Fp8LinearMethod` block_quant sets `activation_quant_key = create_fp8_quant_key(static=False, GroupShape(1,128))`, `fp8.py:290-297` | **plow is WIDER, but in plow's favour on accuracy and against it on MFMA rate** |

### 1.1 vLLM's skip list, counted (MEASURED)

`quantization_config.modules_to_not_convert` in the real `config.json`, 541 entries, classified:
79× each of `input_layernorm` / `post_attention_layernorm` / `q_a_layernorm` / `kv_a_layernorm`;
**76× `mlp.gate`** and 76× `mlp.gate.e_score_correction_bias`; 22× each of `indexers_proj` /
`indexer.k_norm` / `indexer.k_norm.bias`; and one each of `lm_head`, `model.embed_tokens`,
`model.norm`, `shared_head.norm`, `eh_proj`, `enorm`, `hnorm`.

**Every class vLLM leaves in bf16 is a class that is bf16 in the file.** vLLM never widens anything.
`o_proj`, `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj` and the shared experts are all
absent from the skip list, so all six take `Fp8LinearMethod` with `block_quant = True`.

### 1.2 vLLM consumed the block-fp8 checkpoint NATIVELY — proved by capacity, not by a flag

The question "does vLLM dequantise on load?" does not need a log line, because a dequantising vLLM
**could not have run at all** and the verdict log shows it running (MEASURED — it served 4,096 and
16,384 at TP4). Header-derived totals against `rocm-smi --showmeminfo vram` = 309,220,868,096 B
= 288.0 GiB/card:

| | total | per rank at TP4 | vs the card |
|---|--:|--:|:--|
| as shipped (fp8 + f32 grids + bf16) | 755,617,140,416 B = **703.7 GiB** | **175.9 GiB** | fits (0.61x) |
| if dequantised to bf16 | ≈ 1,506.7 GB = **1,403 GiB** | **350.8 GiB** | **1.22x the card — impossible** |

So vLLM held the checkpoint's e4m3 bytes in HBM and multiplied the `[128,128]` block scales in the
kernel, exactly as `Fp8LinearMethod`/`Fp8MoEMethod` describe. (The same arithmetic is why the
all-bf16 arm of the symmetry target is unreachable — §7.)

---

## 2. Why plow was in this state: `GLM_LINEAR_FP8` was OFF, and the checkpoint could not have satisfied it

`mla.rs:698-700` (MEASURED):

```rust
fn glm_linear_fp8(enc: MoeEnc) -> bool {
    enc == MoeEnc::Fp8Blk && std::env::var("GLM_LINEAR_FP8").ok().as_deref() == Some("1")
}
```

Env-only, default off, no other setter in the tree. Three independent proofs it was off for the
verdict blob, in decreasing strength:

1. **The checkpoint cannot satisfy it.** Under `lin_fp8`, `declare_glm_rows` declares
   `…o_proj.weight_fp8` and `…shared_experts.{gate,up,down}_proj.weight_fp8` (`mla.rs:1282-1290`,
   used at `:1363-1367`, `:1379-1399`). **`/home/lava/models/GLM-5.2-plow` contains no `weight_fp8`
   tensor of any kind** (header read of all 79 shards). Those tensors live only in
   `/home/lava/models/GLM-5.2-plow-q`, which is a symlink farm over `GLM-5.2-plow` **plus** 78 local
   `model-idx-*-of-idx.safetensors` (10.69 GB) carrying exactly those eight classes. The verdict
   blob points at the base directory.
2. **The prefill programs falsify it.** With `lin_fp8` on, both prefill emitters route `o_proj` and
   the shared expert to `DevOp::GemmFp8Blk` (op 107, `mla.rs:735-745`). The verdict `build.json`'s
   four prefill programs contain **no `GemmFp8Blk`** — their GEMMs are bf16
   `GemmC5 / GemmGlu / GemmMed / GemmSmall / GemmWide`.
3. **The declared weight stream.** 18,214.5 MB/rank/token is the LINEAR_FP8-**off** total
   (`perf-data/glm52-weight-stream-split.md` §1); on, it is 15,667.5 MB.

**Corollary.** The `GemvFp8Blk`(44) and `DenseGluFp8Blk`(47) arms that *are* in the verdict decode
program serve **only the 3 dense-FFN layers** (`mla.rs:3629-3635,3681,3695`) — not `o_proj`, not the
shared experts. Those two opcodes in `build.json` therefore do **not** mean what they look like they
mean, and `weight_enc: "fp8"` in `build.json` is true of 30.5 % of the stream by bytes.

### 2.1 The bytes, and what they cost

Per rank per token, TP4, ctx 1k, 78 layers. Byte counts from `perf-data/glm52-weight-stream-split.md`
§1 (themselves header-derived — MEASURED there), re-classified here by *what the HF file holds*:

| | MB/rank/token | share |
|---|--:|--:|
| carried as fp8 (routed experts 5400 + dense FFN 162) — **matches the file** | 5,562.0 | 30.5 % |
| carried as bf16 **because the file is bf16** (`lm_head` 1815 + router 225) | 2,040.0 | 11.2 % |
| **carried as bf16 although the file is fp8 — plow widened it** | **10,612.5** | **58.3 %** |
| **total** | **18,214.5** | |

The widened 10,612.5 MB, itemised: `o_proj` 3744.0 · `q_absorb` 2496.0 · `q_a_proj` 1872.0 ·
shared gate+up 900.0 · shared down 450.0 · `kv_a_latent` 468.0 · `q_rope` 312.0 · `v_absorb` 312.0 ·
`k_rope` 58.5.

**INFERRED (arithmetic on the above):** an all-fp8 plow would read 5,306.25 MB/rank/token fewer.
`perf-data/glm52-weight-stream-split.md` §5 prices that at **−0.897 ms of floor** at the measured
6200 GB/s. A floor is a lower bound on the time.

**The widening buys nothing.** `perf-data/glm52-weight-stream-split.md` §6 checked element-wise
against the real tensors: the bf16 on disk is **bit-for-bit** `round_bf16(fp8 × weight_scale_inv)`,
so the fp8 arm is the *un-rounded* form of the same weight. plow pays 2x the bandwidth for a strictly
*less* precise number. This is a defect, not a quality/speed trade.

---

## 3. KV: measured on both sides, and symmetric

Neither `scripts/bench_vllm_chat.sh` nor `scripts/glm52_tpctx_sweep.sh` passes `--kv-cache-dtype`, so
vLLM ran `auto`. `auto` is a flag, not a value — and in this vLLM it has **two escape hatches that
turn it into fp8**, so it had to be resolved, not assumed:

* `resolve_kv_cache_dtype_string` (`vllm/utils/torch_utils.py:374-393`) → returns `fp8` if
  `hf_config.quantization_config` declares a `kv_cache_quant_algo`;
* `Attention.__init__` (`vllm/model_executor/layers/attention/attention.py:236-243`) →
  *"llm-compressor models declare an FP8 KV-cache scheme in their checkpoint config. Honor it only
  when the user did not explicitly pick a kv_cache_dtype"* — sets `cache_config.cache_dtype = "fp8"`
  when `quant_config.kv_cache_scheme is not None`.

**MEASURED** — vLLM's own resolvers executed inside the benchmark image against the real
`zai-org/GLM-5.2-FP8` config (`scripts/glm52_vllm_precision_probe.sh` carries the reproducer):

```
quant_method                                = fp8
activation_scheme                           = dynamic
weight_block_size                           = [128, 128]
"kv_cache_scheme" in quantization_config    = False
"kv_cache_quant_algo" in quantization_config= False

get_kv_cache_quant_algo_string(quant_cfg)   = None
resolve_kv_cache_dtype_string("auto", cfg)  = auto
kv_cache_dtype_str_to_dtype("auto", cfg)    = torch.bfloat16      <-- THE ANSWER

Fp8Config.kv_cache_scheme attribute         = <ABSENT>            <-- 2nd hatch cannot fire
```

and the sink that consumes it, `vllm/platforms/interface.py:557-560`:

```python
if cache_config.cache_dtype == "auto":
    kv_cache_dtype = model_config.dtype        # = torch.bfloat16, from config.json dtype
```

Corroborated by the one block-fp8 startup log preserved on this box,
`perf-data/vllm-rocm/RedHatAI_gemma-4-31B-it-FP8-block_tp1_serve.log:23` — same image, same
`--dtype bfloat16`, no `--kv-cache-dtype`: `… quantization=compressed-tensors, … kv_cache_dtype=auto`
with `Selected TritonFp8BlockScaledMMKernel` on line 30, i.e. fp8 weights beside a bf16 KV cache.

plow side: `mla.rs:1250-1251` allocates `kv.{l}.ckv = ctx*512*BF16` and `kv.{l}.krot = ctx*64*BF16`;
`build.json` reports `kv_enc: "bf16"`.

> **KV was bf16 on both engines. It is not the asymmetry.** The parent hypothesis — that vLLM's
> `auto` silently resolved to fp8 on an fp8 checkpoint — is **FALSIFIED for this checkpoint**. It
> would have been true had GLM-5.2 shipped a `kv_cache_quant_algo` or an llm-compressor
> `kv_cache_scheme`; it ships neither.

### 3.1 …but plow could not have matched a fp8-KV vLLM, and that matters for the next model

`build.json`'s `fp8_kv: false` is derived from the emitted opcodes
(`manifest.rs:345`: `has("FlashDecodeFp8") || has("HeadNormRopeFp8")`), so it says the emitter never
emits 37/38 for GLM. **The reason is not an unset knob — the arm structurally does not exist for
MLA**, and the codebase already tracks it as debt. `crates/devgen/src/lib.rs:5628-5642`,
the `PRECISION_KNOBS` table (`Knob::Ignored` is defined at `:5592` as *"Not read at all. This is the
bug state."*):

```
("PLOW_FP8_KV", Knob::Wired, Knob::Ignored(
    "KNOWN HOLE ... The MLA family has no e4m3 arm for its compressed latent KV
     (no FlashMlaDecodeFp8), so PLOW_FP8_KV=1 on GLM/Kimi/DeepSeek is silently
     dropped and the asset is bf16-KV."))
```

The blockers, in dependency order (all MEASURED):

1. **The kernel has no fp8 arm.** `runtime/amd/op_attention.h:1342` —
   `template <int DK, int DR, int GF, bool GATHER> … d_flash_mla_decode(…, const bf16* __restrict__ Ckv, const bf16* __restrict__ Krope, …)`.
   No `FP8KV` template parameter, caches hard-typed `const bf16*`, no scale operands. The GQA twin
   at `:788` *does* have `bool FP8KV = false`. Same hole in `d_flash_mla_decode_mfma` (`:1672`).
   There is no overload, specialization or second definition anywhere: all six call sites
   (`interp.hip:451-484`) instantiate `<512,64,GF[,GATHER]>` only. **And the NVIDIA path has the
   identical hole** — `runtime/nvidia/op_mla.cuh:141`, `d_flash_mla_decode_sm120<DK,DR,GF,GATHER>` on
   `const mla_bf16*` — so closing this is a *two-backend* cost, not one. That roughly doubles the
   build side of an item already priced at −0.113 ms, and is the strongest single argument for
   leaving it where it is.
2. **No opcode to dispatch.** `packet/src/dev.rs` has no `FlashMlaDecodeFp8` / `FlashGatherDecodeFp8`;
   `interp.hip:1253-1255` dispatches `FLASH_MLA_DECODE` **outside** every `#if PLOW_FP8_KV` guard, so
   even the `_fp8kv` object contains the MLA arm in bf16 form only.
3. **No quantising writer.** The 512-wide latent row is written by `RmsNorm` (`mla.rs:1982`), which
   has no fp8 twin; `HeadNormRopeFp8` would only cover the 64-wide rope half.
4. **Only then the emitter.** `mla.rs` contains zero occurrences of `PLOW_FP8_KV` / `fp8_kv`, and
   `lib.rs:4213` returns from the GLM branch long before `lib.rs:4779` reads the env var. Setting
   `PLOW_FP8_KV=1` on GLM today is a **silent no-op**.

Note the shape is *not* this codebase's usual "an arm exists and nothing routes to it" — here the arm
is genuinely absent.

**Size of the prize (INFERRED from measured geometry: `kv_lora_rank=512`, `qk_rope_head_dim=64`,
78 layers; the MLA latent is REPLICATED per rank, `mla.rs:2159-2161`, so TP4 does not divide it):**

| | bytes/token/rank | 4,096 ctx | 16,384 ctx |
|---|--:|--:|--:|
| bf16 today | 89,856 | 351.0 MiB | 1,404.0 MiB |
| fp8 + f32 row scales | 45,552 | 177.9 MiB | 711.8 MiB |
| **saved** | | **173.1 MiB** | **692.2 MiB (−49.3 %)** |

At 6200 GB/s that is −0.028 ms at 4k and −0.113 ms of floor at 16k. Real, on the slope, and small
next to §2's −0.9 ms.

---

## 4. The OTHER asymmetry, which runs the same way and is larger on the slope

`config.json`: `model_type = glm_moe_dsa`, `index_topk = 2048`. vLLM serves the checkpoint with the
sparse indexer live (its ROCm path is AITER-only and refuses to build without it —
`perf-data/glm52-ctx-sweep.md`). The verdict blob contains **no indexer opcode at all**
(`build.json`), i.e. plow ran DENSE attention.

**Above ~2k context plow reads every KV row and vLLM reads 2048.** At 16k that is 8x more attention
work for plow. This is not a precision item, but it is the dominant term on the SLOPE, and it runs
*against* plow — so the slope gap cannot be attributed to precision, and fp8-KV cannot close it.

Slopes from the verdict logs (MEASURED): plow 28.26 → 29.15 ms over 4k → 16k = **0.072 ms per 1k
ctx**; vLLM 24.97 → 25.51 = **0.044 ms per 1k**. (`perf-data/glm52-ctx-sweep.md` records the plow
dense slope as 0.115 ms/1k over 32k → 128k and vLLM as flat, `0.98x` over the full range.)

---

## 5. A vLLM-side measurement-integrity finding, recorded because it is in the same log

`/home/lava/models/glm52_verdict/vllm_tp4.log` ends with the server-side counter:

```
Prefix cache hit rate: 46.6% / 52.9% / 57.8% / 63.6% / 67.9%
```

`--dataset-name random` with no `--random-prefix-len` should hit nothing. It hits because the harness
runs the ctx list **twice** (cold pass, warm pass) with the same seed, so the warm pass re-issues
byte-identical prompts into a live prefix cache. That is why vLLM's warm TTFT is 46.72 ms for a
4,096-token prompt (≈88,000 tok/s — not a prefill rate). **plowrt has no prefix cache**, and its TTFT
is unchanged between passes (5122.31 → 5135.98 ms).

**Consequence: the warm-vs-warm TTFT row is not a legal comparison.** TPOT is much less exposed —
decode does not read the prefix cache — but vLLM's TPOT also moved 28.68 → 24.97 ms between the
passes while plow's moved 28.33 → 28.26, so "warm" is doing more work on the vLLM side than on the
plow side and the TPOT ratio should be read with that in mind.

---

## 6. VERDICT

**The published 1.13x / 1.14x TPOT comparison was NOT precision-symmetric, and vLLM was the
advantaged engine.**

* **vLLM ran the checkpoint as shipped**: fp8 e4m3 `[128,128]`-block weights on every linear the
  checkpoint quantised, fp8 dynamic per-128-group activations (W8A8), bf16 only where the file is
  bf16, bf16 KV.
* **plow ran 58.3 % of its decode weight stream at 2x the checkpoint's width**, in bf16, for zero
  accuracy gain (§2.1), plus bf16 activations, plus bf16 KV.

**Bound on the advantage.** plow reads 5,306.25 MB/rank/token more than an all-fp8 plow would;
at the measured 6200 GB/s that is **−0.897 ms of floor** (INFERRED, and a lower bound). Against the
measured gaps of +3.29 ms (4k) and +3.64 ms (16k), the dequantisation accounts for **at least ~27 %
and ~25 %** of the published deficit. Corrected, the ratios move to roughly **1.10x** at both ctx.

The floor is a defensible estimator for at least half of it: the 2,547 MB piece that `GLM_LINEAR_FP8`
already reaches was measured end-to-end at **−0.417 ± 0.175 ms against a −0.431 ms projected floor —
101 % of the floor** (`perf-data/glm52-gemm-fp8-blk.md` §8.1).

**The direction of the headline does not change — plow still loses — but the magnitude is inflated by
about a quarter, and no version of the number should be quoted without this paragraph.**

Two things that are *not* the asymmetry, stated so they stop being suspected: **KV was bf16 on both
sides** (§3), and **`lm_head` + router are bf16 in the file** on both sides (§1) — `lm_head`'s 1.9 GB
is a sharding problem, not a precision one.

### 6.1 The one place the asymmetry runs the OTHER way — and it does not rescue plow

**Activations.** plow computes in bf16 (`act_enc: "bf16"`, `w8a8: false`); vLLM quantises every
activation to e4m3 in dynamic per-128-element groups before every block-fp8 GEMM
(`fp8.py:290-297`). On *output quality* that makes **plow strictly the more precise engine** at this
operating point: its weights are the bf16 rounding of the same fp8 product
(`glm52-weight-stream-split.md` §6, checked element-wise), and its activations are wider. Nobody
should read this audit as "plow was cheating".

But it does not buy plow time, and at concurrency 1 it does not cost vLLM much either:

* **At decode, batch 1, activation width is nearly free on both sides.** Every dense op is a GEMV
  over a single activation row, so the op is weight-bandwidth-bound and the activation's dtype is
  noise in the byte count. If anything the per-token-group quantise kernel is a small *tax on vLLM*
  here, which makes vLLM's 24.97 ms the more impressive number, not the less.
* **At prefill it is a real 2x MFMA-rate advantage for vLLM**, and prefill is exactly where plow is
  10x behind (~750 tok/s vs ~7,700). That gap is TTFT, not TPOT, and is out of scope here — but it
  is the place to look next, and it is not a bandwidth story.

So: correcting the weight asymmetry moves TPOT toward plow (§6); correcting the activation asymmetry
would move TTFT further *against* it. The two do not cancel and they land on different metrics.

---

## 7. REMEDY — what all-fp8-vs-all-fp8 costs, and whether it is reachable here

All-bf16-vs-all-bf16 is **not reachable**: vLLM has no path that dequantises an fp8 checkpoint at
load, and the bf16 form is ~1.4 TB. So the target is both engines all-fp8, which means moving plow.

| plow item | MB/rank/token | what it needs | reachable on this box? |
|---|--:|---|:--|
| `o_proj` + shared gate/up/down | 2,547 removed | **nothing new.** `GLM_LINEAR_FP8=1`, bind the `-q` overlay checkpoint, `GemmFp8Blk`(107) already landed and passes its f64 oracle at 3e-2 | **YES, today.** Costs +41.5 ms TTFT (+3.3 %) from the unfused gate\|up |
| `q_a_proj`+`kv_a_latent`+`k_rope` (fusion A) and `q_absorb`+`q_rope` (fusion G) | 2,447 | a **`GemvQkvFp8Blk`** arm. Un-fusing is refused — fusion A exists to stop N=512/N=64 starving CUs and measures 83 % of ceiling fused | no — new kernel |
| `v_absorb` | 156 | `MlaMergeFold`(57) takes `const bf16*` with no encoding operand (`mla.rs:1353-1362`) | no — new kernel arm |
| `q_absorb` re-blocking | (inside the 2,447) | genuine **requantisation** — the only piece with an accuracy question | no |
| `lm_head`, router | 0 | nothing — bf16 in the file on both engines | n/a |
| KV → fp8 | 692 MiB at 16k | `FlashMlaDecodeFp8` opcode + `FP8KV` template on `d_flash_mla_decode` + an fp8 ckv writer + emitter routing at 4 MLA entry points (§3.1) | no — and it would *break* symmetry, since vLLM's KV is bf16 |

**So the reachable step today closes 2,547 of the 5,306 MB excess — 48 % of the asymmetry — for a
measured −0.417 ms, and needs only a knob and the `-q` checkpoint.** The remaining 52 % is a
`GemvQkvFp8Blk` kernel.

**The overlay is on disk and complete** (MEASURED — header read of the 78 local
`model-idx-*-of-idx.safetensors` in `/home/lava/models/GLM-5.2-plow-q/`, 10,685,500,416 B). It
carries exactly the eight classes `declare_glm_rows` asks for under `lin_fp8`, at shapes and dtypes
identical to the HF file:

| tensor | dtype | shape | n |
|---|---|---|--:|
| `self_attn.o_proj.weight_fp8` | F8_E4M3 | `[6144, 16384]` | 78 |
| `self_attn.o_proj.weight_scale_inv` | F32 | `[48, 128]` | 78 |
| `shared_experts.gate_proj.weight_fp8` / `up_proj.weight_fp8` | F8_E4M3 | `[2048, 6144]` | 75 each |
| `shared_experts.down_proj.weight_fp8` | F8_E4M3 | `[6144, 2048]` | 75 |
| the three matching `weight_scale_inv` | F32 | `[16,48]` / `[16,48]` / `[48,16]` | 75 each |

Nothing has to be prepared, downloaded or requantised. **The verdict blob simply bound the wrong one
of two checkpoint directories that differ by a symlink.** Note the overlay covers *only* `o_proj` and
the shared experts — there is no overlay for `q_a_proj` or any `derived.*` tensor, which is the
on-disk expression of the same `GemvQkvFp8Blk` gap.

**The re-run that would settle it** (not taken here — no GPU capacity; the box was held by a foreign
TP8 job at ~219 GB on all 8 cards for the duration, `rocm-smi --showpids`): re-emit the verdict blob
with `GLM_LINEAR_FP8=1` against `/home/lava/models/GLM-5.2-plow-q`, gate on coherence first
(**mandatory** — on GLM-5.2 wrong numerics run *faster*, because garbage activations collapse the
router's top-k and the experts do less work), then re-run `scripts/glm52_tpctx_sweep.sh` at 4k/16k
against the same vLLM column. Expected: 1.13x → ~1.11x, i.e. half the correction.

## 8. Reproducers added by this work

* `scripts/glm52_vllm_precision_probe.sh` — brings vLLM up exactly as the benchmark does
  (`--dtype auto`, `VLLM_ROCM_USE_AITER=1`, no `--kv-cache-dtype`), keeps the whole startup log,
  runs a coherence prompt before anything is believed, and dumps what the **live engine** resolved.
  **Not yet run against a GPU** — the §3 resolution above was taken CPU-only inside the same image,
  which is the stronger evidence anyway (it executes vLLM's own resolver on the real config rather
  than requiring a log line to be interpreted). The script exists so the log-line form can be taken
  the next time four cards are free.
