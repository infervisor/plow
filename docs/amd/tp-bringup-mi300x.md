# TP bringup on 8x MI300X — GLM-5.3, Kimi-K2.7-Code, DeepSeek-V4-Flash

Bringup status, base numbers and roofline for the three checkpoints in
`/workspace/models`, measured on one 8x MI300X host on 2026-09-08 from
`main` @ `8d98edcb`.

**Headline:** GLM-5.3 serves in production at TP8 and is qualified here, with
end-to-end base numbers. Kimi-K2.7-Code now **emits and disassembles a single
block** at TP8 — the trace is in §5 — but cannot serve. DeepSeek-V4-Flash
cannot emit at all. For the two that do not serve, what is reported is what was
actually measured: the hardware ceiling at their own per-rank shapes, which
bounds them on this part. No end-to-end serving number is invented for a model
that does not serve.

**2026-09-09 correction:** §10 identifies and fixes the ragged-row defect behind
the historical sparse-span failures. The reuse-distance explanations in §7 are
superseded. The H200 serving target has not been matched.

---

## 0. Target parameter block

Filled per [`docs/bringup/target.md`](../bringup/target.md). A row that cannot
be filled is a blocker, not a default.

| parameter | value |
|---|---|
| `$VENDOR` | `amd` |
| `$ISA` | `gfx942` |
| `$GPU` | `MI300X` (192 GiB HBM3, 304 CU) |
| `$NCU` | 304 (`--n-cu 0`, from the `--gpu` spec) |
| `$NGPU` / `$PARALLEL` | 8 (and 4, measured) / `tp` |
| `$MAXCTX` | 18432 |
| `$TOOLCHAIN` | TheRock ROCm 7.14.0 (nix), HIP/clang from the flake |
| `$BUILD` | `scripts/build_gfx942.sh` at `PLOW_DECODE_BATCH=4` |
| `$FEATURES` | `--features hsa` |
| `$BW_BOUND` | **4091.9 GB/s, measured** (see §4) — *not* the 5325 datasheet peak |
| `$COMPUTE_CEIL` | fp8 470–1057 TF/s, shape-dependent (see §4) |
| `$RESULTS` | job scratch; the retained numbers are in this file |

## 1. Harness

No new runner was written. This campaign used the existing harness, per
[`docs/bringup/agents/README.md`](../bringup/agents/README.md) ("extend an
existing harness when its semantic boundary is incomplete; do not create a
campaign-specific runner"):

| stage | harness |
|---|---|
| correctness gate | `perf-data/tools/bringup_gate.sh` |
| measured pass | `perf-data/tools/bringup_bench.sh` |
| GEMM ceiling | `perf-data/tools/bringup_ceiling.py` |
| HBM ceiling | `scripts/glm53_hbm_ceiling.py` |
| serialization | `perf-data/tools/gpulease` |

Two boundary extensions were needed and made:

* `bringup_ceiling.py` took its shapes from a hardcoded Gemma-12B list. It is
  now `--model {gemma12b,glm53,k27,dsv4} --tp N --rows M`, with per-rank sharding
  applied on the correct dimension per projection (column-parallel shards `N`,
  row-parallel shards `K`). `--model gemma12b` reproduces the original list at
  `tp=1`, so numbers already published against it still stand.
* The same tool asked for OCP `e4m3fn` unconditionally. CDNA3 implements the
  `fnuz` variant and `torch._scaled_mm` hard-refuses `e4m3fn` there, so the tool
  could not run on gfx942 at all. The encoding is now probed and printed.

## 2. Bringup state, per model

Verified by running `plowc --emit devblob` against each checkpoint, not by
reading code.

| model | `model_type` | emit path | state |
|---|---|---|---|
| GLM-5.3-FP8 | `glm_moe_dsa` | `glm_main`, full model | **serves** (§3) |
| Kimi-K2.7-Code | `kimi_k25` → text `kimi_k2` | `kimi_emit_block`, single block only | **block emits** (§5); cannot serve |
| DeepSeek-V4-Flash | `deepseek_v4` | none | blocked (§6) |

## 3. GLM-5.3 — production configuration

**TP8 is the production configuration. TP4 is not viable.**

Bundle: `plowc --emit devblob --gpu MI300X --arch gfx942 --max-ctx 18432
--num-gpus 8`, with the frozen RECIPE-gfx942 knob set from
`scripts/glm53_mi300x.sh` (`GLM_FULL=1 PLOW_FP8=1 PLOW_MLA_PF_V2=1
PLOW_GLM_DSA=0 GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1
PLOW_GLM_FUSE_{ROPE,SEAM,B1}=1 PLOW_MOE_PF_DET=1
PLOW_DECODE_BATCH_LADDER=1,2,4`). `lean.verified=true`, `lean.oracle=true`,
pairing hash present. Objects from `scripts/build_gfx942.sh` at
`PLOW_DECODE_BATCH=4`.

Correctness gate (`bringup_gate.sh`, four fixed greedy prompts, temperature 0):
**passed at both TP8 and TP4**, GPU-accelerated backend confirmed.

### Base numbers

`bringup_bench.sh`, vLLM 0.28 `bench serve` client, `/v1/completions`,
`--ignore-eos`, output length 128, 8 prompts, 1 warmup round, seed 42.

| TP | input | conc | TTFT mean / p99 (ms) | TPOT med (ms) | out tok/s |
|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 1 | 171.6 / 176.8 | 28.85 | 33.36 |
| 8 | 1024 | 1 | 266.4 / 273.0 | 29.75 | 31.64 |
| 8 | 4096 | 4 | 1550.5 / 3159.8 | 86.96 | 40.52 |
| 4 | 128 | 1 | 326.1 / 332.8 | 142.00 | 6.97 |
| 4 | 1024 | 1 | 17156.4 / 17164.3 | 142.87 | 3.63 |
| 4 | 4096 | 4 | 58942.2 / 119921.4 | 867.28 | 3.32 |

### Why TP4 is not viable

Not a tuning gap — a capacity wall. Per-rank checkpoint upload:

| TP | GiB/rank | % of the 192 GiB card |
|---:|---:|---:|
| 8 | 93.04 | 48.5% |
| 4 | 181.75 | **94.7%** |

At TP4 roughly 10 GiB is left for KV, activations, the co-resident MoE staging
buffers and every scratch allocation, and TPOT degrades 4.9x while TTFT at 1024
input degrades 64x (267 ms → 17.2 s). The 4096/concurrency-4 cell is where it
stops being a slowdown and becomes a different regime: TTFT mean 58.9 s, p99
119.9 s, TPOT 867 ms — a 10x TPOT gap over the same cell at TP8, and 3.32
aggregate output tok/s against 40.52. Both TP arms compiled the same KV
geometry (`n_kvrow=156`, `max_ctx=18432`), so this is not a KV-budget
difference. The 4.9x TPOT gap is also larger than the 2x expert-bytes-per-rank
increase TP4 implies, so residency pressure — not arithmetic — dominates.
TP4 is left recorded and unpursued; the finding is that GLM-5.3 wants TP8 on a
192 GiB part.

## 4. Roofline

### Denominators, measured on this part

`scripts/glm53_hbm_ceiling.py` on one leased MI300X:

| probe | GB/s |
|---|---:|
| read, 2 GiB (reduction) | 4266.0 |
| read, 8 GiB (reduction) | **4091.9** |
| copy, 2 GiB (read+write) | 3856.1 |
| copy, 8 GiB (read+write) | 3744.3 |

4091.9 GB/s is taken as `$BW_BOUND` — the figure that outlives cache residency.
It is **76.8% of the 5325 GB/s datasheet peak**, and
`crates/hwspec/src/amd/mi300.rs` previously carried `bandwidth_measured: None`,
so every roofline percentage computed for MI300X before this campaign silently
used the datasheet number and understated the kernels by ~30%. The measurement
is now recorded in the registry.

GEMM ceiling, `bringup_ceiling.py --model glm53 --tp 8` (per-rank shapes,
`e4m3fnuz`, M=4096):

| projection | M | N | K | fp8 TF/s | bf16 TF/s |
|---|---:|---:|---:|---:|---:|
| q_a_proj | 4096 | 2048 | 6144 | 881.0 | 549.0 |
| q_b_proj | 4096 | 2048 | 2048 | 755.1 | 552.6 |
| kv_a_proj | 4096 | 576 | 6144 | 721.4 | 419.5 |
| kv_b_proj | 4096 | 3584 | 512 | 470.0 | 388.6 |
| o_proj | 4096 | 6144 | 2048 | 812.1 | 566.4 |
| dense gate/up | 4096 | 3072 | 6144 | **1056.8** | 630.6 |
| dense down | 4096 | 6144 | 1536 | 862.0 | 576.9 |
| expert gate/up | 128 | 4096 | 6144 | 303.6 | 185.4 |
| expert down | 128 | 6144 | 2048 | 160.2 | 131.8 |

The MoE expert GEMMs are the shape that actually carries the model — 75 of 78
layers — and they reach 160–304 TF/s, 15–29% of what the same library reaches
on the dense shapes. That is a property of `M=128` (the routed share of a 4096
chunk at top-8 of 256), not of plow's kernels; it bounds any MoE prefill design
on this part.

### Where decode actually sits

Activated fp8 weight bytes per token per rank at TP8, from the model geometry:

```
attention/layer   q_a 12.58 + q_b 4.19 + kv_a 3.54 + kv_b 1.84 + o 12.58  =  34.73 MB
MoE/layer         8 routed x 4.72 + shared 4.72 + router 1.57            =  44.04 MB
75 MoE layers x 78.77 MB                                                 =   5.91 GB
3 dense layers x 63.04 MB                                                =   0.19 GB
lm_head 154880 x 6144 / 8                                                =   0.12 GB
                                                                    total ~  6.22 GB
```

At the measured 4091.9 GB/s that floor is **1.52 ms/token**. Observed TPOT at
concurrency 1 is **28.85 ms — 19.0x the bandwidth-bound floor, i.e. ~5.3% of
roofline**. GLM-5.3 decode on this part is **not** bandwidth-bound; it is bound
by dispatch count and per-op efficiency. Three corroborating measurements:

* single GEMV probes reach 1439.8 GB/s (o_proj TP8) and 2282.5–2499.1 GB/s
  (q_absorb / o_proj TP4) — 35–61% of ceiling *per op*, before op count;
* the emitter's own dispatch audit reports `GemmSmall` dispatches filling
  **10.5–42.1%** of the 304 CUs (e.g. `128x2048x6144` at 10.5%, 32 tiles over
  304 CUs);
* only **228 of 2472** dense-GEMM tiles were chosen by measurement; the other
  2244 fell back to the analytical model (`plowc tune status` names them).

The ordered leads for the next performance campaign are therefore: (1) tile
coverage for the 2244 analytically-chosen shapes, (2) dispatch fill on the
`GemmSmall` rows, (3) decode-tier objects — `PLOW_DECODE_TIERS` low-rung
objects were **not** built for this campaign (`PLOW_HSACO_LOWRUNG` was unset),
and on Gemma-4 the batch-width-matched decode object was worth ~29% TPOT at
concurrency 1.

## 5. Kimi-K2.7-Code — blocked, one gate advanced

`/workspace/models/kimi_k27_code` is `KimiK25ForConditionalGeneration`,
`model_type: kimi_k25`, wrapping a `kimi_k2` MLA text tower: 61 layers, hidden
7168, 384 experts top-8, `q_lora_rank` 1536, `kv_lora_rank` 512, and 208,550
tensors under a `language_model.model.` prefix.

**Before:** `plowc` died at `crates/devgen/src/config.rs:241` on
`Option::unwrap` — `kimi_k25` was claimed by no arm, so the `text_config` probe
routed it into the **Gemma-4** parser, which then unwrapped Gemma's
`layer_types`. The error named the wrong field of the wrong architecture.

**Now:** `kimi_k25` is claimed and routed to the MLA path, and `cfg_glm` reads
both the nested `text_config` geometry and the `language_model.model.` weight
prefix off one probe. The failure is now an accurate, architectural one:

```
kimi_k2: rope_type = "default", rope_scaling present = true. The MLA tables are
built with RopeScale::None, so a scaled scheme (yarn / linear / llama3) would be
emitted as an UNSCALED RoPE at theta 50000 — correct-looking tables, wrong
long-context behaviour.
```

**YaRN is now wired** (`RopeScale::YarnDeepSeek`, `ROPE_SCALE_YARN_DS`). It was not
a matter of passing the existing `RopeScale::Yarn`: HF's generic YaRN and the
DeepSeek family disagree about the attention factor, and taking the generic one
would have been a quieter version of the bug the gate was catching.

| | table factor (cos/sin) | softmax fold |
|---|---|---|
| HF generic YaRN | `0.1*ln(f)+1` = **1.4159** at f=64 | none |
| DeepSeek family | `mscale(f,mscale) / mscale(f,mscale_all_dim)` = **1.0** | `mscale(f,mscale_all_dim)^2` = 2.0047 |

K2.7 ships `mscale = mscale_all_dim = 1.0`, so the table ratio is exactly 1.0 —
a 41.6% error had the generic formula been reused — while the `0.1*ln(f)+1`
term reappears squared in `softmax_scale`. The frequency interpolation is
identical between the two and is shared rather than restated, so they cannot
drift. Only the `mscale == mscale_all_dim` case is representable and the
emitter refuses the unequal one: an unequal pair needs a per-table attention
factor the ABI-locked 72-byte `GenTensor` has no room for.

### Single-block trace, TP8

`plowc --hf-dir /workspace/models/kimi_k27_code --emit devblob --gpu MI300X
--arch gfx942 --num-gpus 8 --block 3` now emits:
`kimi_mla_moe --block 3..4: bf16 block, 1 layer(s), 36 decode ops, ctx=4096 tp=8`.

`plowrt disasm` on it confirms the geometry is the model's and not a default:
`n_head=8` (64/8), `n_exp=384`, `k=8`, `I_moe=256` (2048/8), `H=7168`, the MLA
absorb path bound to `derived.{q_absorb,kv_a_latent,k_rope,q_rope,v_absorb}`,
and one `MoeExpertGlu`/`MoeExpertDown` pair per top-k slot — 8, not 384.

The trace also independently validates the RoPE work above. `FlashMlaDecode`
carries `scale=0.14467962`, and

```
qk_head_dim^-0.5 * mscale(64, 1.0)^2 = 192^-0.5 * 1.4158883^2
                                     = 0.07216878 * 2.00473  = 0.14467962
```

which is the emitted constant exactly. Had the table/softmax split been taken
the other way round, this number would have been 0.07216878.

Op inventory for the 36-op decode block (dispatch width `b` in CUs):

| ops | count | width |
|---|---:|---:|
| `RmsNorm` | 3 | 1 |
| `GemvQkv` | 2 | 302 / 288 |
| `HeadNormRope` | 2 | 1 |
| `FlashMlaDecode` + `MlaMergeFold` | 2 | 64 / 304 |
| `Gemv` (o_proj, router, shared down) | 3 | 304 |
| `MoeExpertGlu` + `MoeExpertDown` | 16 | 38 |
| `XReduce` / `MoeCombine` | 3 | 14 |
| `Residual` / `GemvGlu` / `MoeRouterTopk` | 5 | 1–304 |

The 16 expert dispatches at **b=38** are the structural finding: 8 top-k slots
issued as 16 separate dispatches, each occupying 38 of 304 CUs (12.5%). That is
the decode-side twin of §4's `GemmSmall` occupancy result, and it is the reason
K2.7's expert shapes measure as badly as they do below.

### The block executes on GPU — and one layer kind faults

`plowrt amd-block` runs a block through the AMD engine. With no `--checkpoint`
it says so itself — *"NO CHECKPOINT — weights are uninitialised; timings are
real, tokens are not"* — which makes it a legitimate execution check but not a
numerics one. (It is also not a timer: it is the A/B vehicle for numerics and
reports no latency. `plowrt amd-probe` would, but that path documents itself as
"never a model-quality or performance result", so no block latency is claimed
here.)

Emitted at `--num-gpus 1` (the engine refuses a tp=8 packet on one rank, by
design: "every projection in it is 1/8 wide, so binding it here would fail at
the first weight") with `PLOW_MLA_PREFILL=full:128`:

| block | layer kind | emit | GPU run |
|---|---|---|---|
| `--block 3` | MoE (top-8 of 384) | 33 decode ops + 23-op T=128 prefill | **runs clean** |
| `--block 0` | dense (`first_k_dense_replace=1`) | 15 decode ops + 19-op T=128 prefill | **memory access fault** |

The dense block faults on gfx942:

```
INFO  dense-FFN prefill pointer tables bound (grouped-arm 1-expert path)
Memory access fault by GPU node-8 ... on address 0x7ff1a13a4000. Reason: Unknown.
```

The MoE block, emitted and run the same way in the same lease, does not. That
localises the defect to the **dense-FFN "grouped-arm 1-expert path"** in the
Kimi block prefill — the arm that expresses a dense MLP as a one-expert grouped
MoE — rather than to the MLA attention half or to block emit generally. It is
the next concrete bug on this path, and it reproduces in two commands.

### Measured ceiling at K2.7's own shapes

`bringup_ceiling.py --model k27 --tp 8`, per-rank, M=4096 (`e4m3fnuz`):

| projection | M | N | K | fp8 TF/s | bf16 TF/s |
|---|---:|---:|---:|---:|---:|
| q_a_proj | 4096 | 1536 | 7168 | 842.4 | 508.1 |
| q_b_proj | 4096 | 1536 | 1536 | 650.1 | 435.7 |
| kv_a_proj | 4096 | 576 | 7168 | 744.1 | 468.5 |
| kv_b_proj | 4096 | 2048 | 512 | 401.2 | 334.8 |
| o_proj | 4096 | 7168 | 1024 | 679.4 | 544.1 |
| expert gate/up | 85 | 4096 | 7168 | 234.8 | 146.7 |
| expert down | 85 | 7168 | 2048 | 115.4 | 110.6 |

K2.7 routes top-8 of **384** experts, so a 4096-row chunk gives each expert 85
rows against GLM-5.3's 128 at top-8 of 256. The expert GEMMs land at 115–235
TF/s versus GLM's 160–304 on the same part — the wider expert table makes the
per-expert GEMM smaller, and smaller is worse here. Any K2.7 prefill design on
gfx942 is bounded by this before a single plow kernel is written.

**Remaining gates, in order:**

1. ~~YaRN in the MLA RoPE tables.~~ **Done** — see above.
2. **Full-model Kimi device emit.** `crates/devgen/src/lib.rs` supports only
   `--block <l>[..<r>]` for this family; the `glm_main` analogue is an
   unimplemented milestone. Single-block traces are reachable today (above); a
   served model is not.
3. **Expert precision.** The checkpoint is compressed-tensors w4a16 at
   `group_size 32`, and the block above emits the **bf16** arm, so `amd-block`
   cannot bind these weights for a measured block latency. A w4-group-32 expert
   arm is the gate between "the block emits and disassembles" and "the block
   runs on hardware with its own weights". `plowrt amd-probe` would execute it
   with zero-filled weights, but that path documents itself as "never a
   model-quality or performance result" and is not reported as one here.
4. **Tokenizer.** The checkpoint ships `tiktoken.model` +
   `tokenization_kimi.py`, not `tokenizer.json`, so the bundle's tokenizer link
   has no source file.

## 6. DeepSeek-V4-Flash — blocked, no serving path

`plowc` refuses with a complete statement of what is missing; the refusal is the
authority and is reproduced by
`plowc --hf-dir /workspace/models/DeepSeek-V4-Flash-0731 --emit devblob`.
Stage 1 exists (`parse_deepseek_v4` in `crates/nn-graph`), and per
[`deepseek-v4-mi300x.md`](deepseek-v4-mi300x.md) five of the
seven "needs a new kernel body" rows are implemented and hardware-tested on
gfx942 — the compressor, the indexer, hash routing, inverse RoPE on O, the
clamped SwiGLU. What blocks serving is the **emitter**, not the kernels:

* no `deepseek_v4` graph builder, and no `deepseek_v4` arm in the devgen
  dispatch at all (it falls through to `config.rs:228`,
  `unsupported model_type "deepseek_v4"`);
* per-layer RoPE tables, the 128-entry full-resolution window ring, and the
  grouped output LoRA (`o_groups=8 x o_lora_rank=1024`) are unimplemented;
* w4a8 routed experts: `MoeGluMx`/`MoeDownMx` already read the fp4 expert
  weights byte-identically, but those arms are w4a16 and the reference is w4a8
  (fp8 e4m3 activations, per-128-K power-of-2 scale).

### Measured ceiling at DSV4's own shapes

plow cannot serve this model, so this is a bound on what it could reach here,
not a measurement of it. `bringup_ceiling.py --model dsv4 --tp 8`, per-rank,
M=4096 (`e4m3fnuz`), geometry from the shipped `config.json` (hidden 4096, 64
heads at head_dim 512 with `num_key_value_heads=1`, `q_lora_rank` 1024, 256
experts top-6, and the grouped output LoRA `o_groups=8 x o_lora_rank=1024`
priced as its two factors):

| projection | M | N | K | fp8 TF/s | bf16 TF/s |
|---|---:|---:|---:|---:|---:|
| q_a_proj | 4096 | 1024 | 4096 | 671.1 | 436.8 |
| q_b_proj | 4096 | 4096 | 1024 | 733.2 | 521.0 |
| kv_a_proj | 4096 | 576 | 4096 | 640.1 | 424.1 |
| o_lora_down | 4096 | 1024 | 4096 | 736.6 | 496.0 |
| o_lora_up | 4096 | 4096 | 1024 | 713.1 | 522.8 |
| expert gate/up | 96 | 4096 | 4096 | 157.4 | 131.6 |
| expert down | 96 | 4096 | 2048 | **59.2** | **101.1** |

`expert down` is the one row in this whole campaign where **fp8 is slower than
bf16** — 59.2 against 101.1 TF/s. At M=96 the per-tile scale handling costs more
than the narrower operand saves, so the w4a8 expert arm DSV4 needs (§6) should
not be assumed to pay for itself at decode-shaped M on this part; it has to be
measured against a bf16 arm rather than adopted because the weights are narrow.
The grouped output LoRA, by contrast, prices well (713–737 TF/s) — it is two
ordinary GEMMs and is not where this model will be slow.

DSpark (3 MTP blocks) is optional — the shipped `generate.py` never calls
`forward_spec`, so a first bringup is bit-exact without it.

## 7. Reproduction

```bash
# objects (no GPU)
PLOW_DECODE_BATCH=4 nix develop --command scripts/build_gfx942.sh <objdir>

# packet (no GPU); PLOW_VERIFY_BIN must be set or the gate refuses the bundle
PLOW_VERIFY_BIN=lean-plow/.lake/build/bin/plow_verify \
GLM_FULL=1 PLOW_FP8=1 PLOW_MLA_PF_V2=1 PLOW_GLM_DSA=0 \
GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 \
PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1 PLOW_GLM_FUSE_B1=1 \
PLOW_MOE_PF_DET=1 PLOW_DECODE_BATCH_LADDER=1,2,4 \
PLOW_MLA_PREFILL=full:128,512,2048,8192 \
  plowc --hf-dir /workspace/models/GLM-5.3-plow-lite --emit devblob \
        --gpu MI300X --arch gfx942 --max-ctx 18432 --num-gpus 8 --out <b>/model.pkt

# gate + bench (8 GPUs, one lease, INSIDE nix — outside it plowrt finds no
# libhsa and refuses rather than serving the CPU interpreter)
perf-data/tools/gpulease -n 8 glm53 nix develop --command <driver>

# ceilings (1 GPU)
perf-data/tools/gpulease -n 1 ceil python3 perf-data/tools/bringup_ceiling.py --model glm53 --tp 8
perf-data/tools/gpulease -n 1 hbm  python3 scripts/glm53_hbm_ceiling.py
```

The bundle must carry `tokenizer_config.json`, `chat_template.jinja` and
`generation_config.json`, not just `tokenizer.json`: `serve/chat.rs` prefers the
checkpoint's own template over the built-in builders, and the TP4/TP8 bundles
found on this host were missing all three, so they had been served through an
approximation of a file the weights already carry.

## 7b. Optimization — measured against the §3 baseline

Every arm below ran in ONE lease, one lever at a time, against the same bundle,
with the base arm re-measured in that lease rather than carried from §3. It
reproduced the frozen numbers to within noise (33.34 vs 33.36 out tok/s, TPOT
28.88 vs 28.85), which is what makes the deltas trustworthy.

| arm | in=128 c=1 tok/s | TPOT ms | in=1024 c=1 tok/s | TPOT ms |
|---|---:|---:|---:|---:|
| base (§3 recipe) | 33.34 | 28.88 | 31.56 | 29.79 |
| + low-rung decode tiers | 41.64 | 22.90 | 38.95 | 23.80 |
| + TP audit off | **43.88** | **21.63** | **40.93** | **22.55** |
| | **+31.6%** | **−25.1%** | **+29.7%** | **−24.3%** |

**Low-rung decode tiers (+24.9%) were free and were simply not built.** The §3
objects are compiled at `PLOW_DECODE_BATCH=4` while the packet's decode ladder
is `1,2,4`, so at concurrency 1 a rung-1 packet ran the width-4 object's body:
`PLOW_GEMV_MM` is a compiled CEILING, so every decode GEMV computed four rows
and discarded three. `scripts/build_gfx942.sh PLOW_DECODE_TIERS=1,2` builds the
matched objects into `lowrung{1,2}/`, and plowrt co-loads them when
`PLOW_HSACO_LOWRUNG` names them. The baseline served with
`hsaco_lowrung: None`.

**TP audit off is a further +5.4%, and is a real tradeoff, not a free win.**
`audit` is one 12 KiB readback per rank per decode step; the code sanctions
`PLOW_TP_NO_AUDIT=1` "for a timing run" and what it catches is a silently wrong
token. It is reported here because it is measurable, not recommended for
production without deciding that risk. `PLOW_TP_AGREE_EVERY` reduces the cadence
instead of removing the check and is the safer knob.

### Where the remaining time is — decode-step attribution

`PLOW_DSTEP_LOG=1` on the best config, single steady stream, n=64 tokens:

| phase | µs/token | % |
|---|---:|---:|
| **GPU drain (all ranks)** | **19832.9** | **91.6%** |
| pre rearm_prog (local counters), 8 calls/tok | 1382.0 | 6.4% |
| pre decode_prepare | 140.2 | 0.6% |
| pre zero_xctr | 88.9 | 0.4% |
| post read_sampled | 74.3 | 0.3% |
| idle between mux ticks | 53.0 | 0.2% |
| post TP safety audit | 0.0 | 0.0% |
| HOST TOTAL | 1699.1 | 7.8% |

This bounds the rest of the campaign: **host work is 7.8% of a token**, so every
remaining host-side optimization together cannot buy more than that, and the
next real gains have to come from the GPU side. `rearm_prog` at 1382 µs across
8 calls is the only host term worth attacking, and it is worth at most 6.4%.

Against §4's roofline this also re-frames the 5.3% figure: the GPU is busy for
91.6% of the token while moving 6.22 GB, so the gap is not host stalls — it is
what the GPU is doing during those 19.8 ms.

### Token batch — the packet is eligible

`plowrt op-audit` classifies the whole TP8 decode program **`[PACKABLE]`**:
1860 instructions, 18 distinct opcodes, 1392 class-A (ready) and 468 class-B
(descriptor-fills), none class-C/D. So the unified token-batch route is legal
for this model; it was simply never requested — the route reports
`armed=true fires=false`, "opt-in per (backend, family) pair until that pair is
measured". Measuring that pair is §7c.

## 7c. Throughput, long context, and the levers that did NOT work

Continuing §7b. All arms: 8x MI300X TP8, one lease per group, base re-measured
where the group allowed it. Run-to-run noise was measured directly — two
independent servers on the same config gave 52.12 and 52.06 out tok/s (0.1%) —
so anything under ~1% here is noise, not a result.

### Decode batch ladder 4 -> 8 — the throughput lever

The frozen recipe emits `PLOW_DECODE_BATCH_LADDER=1,2,4` against objects built
at `PLOW_DECODE_BATCH=4`. At concurrency 8 the admission log shows the widest
rung reached is **4**, so eight concurrent requests decode in two batches per
step. Rebuilding at `PLOW_DECODE_BATCH=8` with `PLOW_DECODE_TIERS=1,2,4` and
emitting the ladder `1,2,4,8`:

| in=4096 | conc | out tok/s | TTFT mean | TPOT med |
|---|---:|---:|---:|---:|
| §3 baseline (ladder 4, no tiers) | 4 | 40.52 | 1550 ms | 86.96 |
| tiers + audit off, ladder 4 | 8 | 43.08 | 10270 ms | 85.81 |
| **tiers + ladder 8** | **8** | **52.12** | **2511 ms** | 139.40 |
| tiers + ladder 8 | 16 | 52.54 | 16666 ms | 144.45 |

**Throughput saturates at ~52 tok/s.** Concurrency 16 buys 0.8% over
concurrency 8 while doubling TPOT, so the knee is at 8 and the ladder is now
matched to it. Against the §3 row this is **+28.6%**; against the same
concurrency with the old ladder it is +21.0%.

### The unified token batch — eligible, still unmeasured

`plowrt op-audit` classifies the TP8 decode program `[PACKABLE]`, so the route
is legal. It still never fires, and the refusal is precise:

```
capability `token_batch_unsplit_attention`: no prefill bucket in this blob has
nsplit=1 with a fused flash epilogue
```

Two conditions, and the recipe defeats both. `PLOW_GLM_PF_NS=2` forces
nsplit=2, and the fused epilogue is `PLOW_GLM_OFOLD`, which the recipe never
sets. Setting `PLOW_GLM_PF_NS=1` alone is **not** enough — the route still
refuses. Setting `PLOW_GLM_OFOLD=1` emits, then refuses at LOAD:

```
MISSING WEIGHT: model.layers.0.self_attn.derived.o_fold.weight
```

so the fused epilogue needs a derived tensor `scripts/glm53_prep.py` does not
write. **The token-batch route is therefore blocked on a checkpoint prep step,
not on the runtime.** Any earlier number attributed to it is not one: the
`PLOW_TOKEN_BATCH=1` arms measured 43.38 and 43.11 out tok/s against 43.08 with
the route inert, which is the route not running.

### Levers that did not work — recorded because they were measured

| lever | result | why |
|---|---|---|
| `PLOW_GLM_PF_NS=1` vs `=2` | 43.60 vs 43.08 @4096c8; 33.41 vs 33.67 @8192c8 | noise. The knob earns nothing at these shapes and costs the token-batch precondition — **drop it from the recipe** |
| Full GEMM tile retune | 43.88 / 52.13 vs 43.88 / 52.12 | **no change.** `plowc tune status` reported all 4043 MI300X records STALE and selection falling back to the analytical model; a fresh campaign published 288 records and took the emit from "228 of 2472 tiles by measurement" to "**all 2472 by measurement**" — and the served numbers did not move. For these shapes the analytical model was already choosing equivalent tiles |
| `PLOW_GLM_DSA=1` | 52.06 vs 52.12 | **the lever never armed.** `dsa()` requires `ctx > 65536` and this blob is `--max-ctx 18432`, so no `FlashGather`/`Index` op is emitted — verified by disassembling both packets. DSA is a >64k-context feature, not a knob for an 18k deployment. Its crossover constant is also documented as measured at TP4 with an explicit note to recalibrate for TP8 |
| `GLM_SHARED_CUS` 32 / 48 / 76 | 43.77 / 43.88 / 43.24 | the recipe's 48 is already the best of the three; the MoE CU partition is not where the remaining time is |

### What bounds the rest

§7b's attribution is the answer: GPU drain is 91.6% of a decode token and all
host work together is 7.8%. Every lever above that worked did so by removing
GPU work (a narrower decode object, a wider batch per step); every lever that
failed either never engaged or moved a term that was not the cost. The
remaining gap to a 50% target is **kernel work on the MoE expert path** — §4's
ceiling puts those GEMMs at 160-304 TF/s against 1057 on the dense shapes — not
another knob.

## 7d. Reproducing a SERVE, not just a compile

`build.json` reproduces the compile: 149 emit knobs with clap-sourced provenance, a `replay` map
in env spelling, `unrecorded_env` for the vars that escape `EmitConfig`, a `pairing.hash`
(fnv1a64 over `union`, `objects`, `tuning`) that a cubin stamps and the loader refuses on
mismatch, and arm-level `requires` checked against the object's symbol table.

Nothing reproduced the serve, and the two are different configurations. `PLOW_HSACO_LOWRUNG`,
`PLOW_TP_NO_AUDIT` and `PLOW_L2_PLACE_DISPATCH` change throughput and — for the audit one — the
odds of a silently wrong token, and none of them appears in the packet's manifest. §7b is what
that costs: the frozen baseline served with `hsaco_lowrung: None` and gave up 24.9% output tok/s,
and nothing in the bundle recorded it.

Two records close it:

* **`serve replay`** — the runtime twin of `emit_config.replay`: only the knobs this serve
  resolved away from their defaults, in the spelling that sets them again, provenance from clap's
  `ValueSource` rather than probed from the environment (inferring is wrong the moment a flag
  overrides an env var). For the tuned TP8 arm it is exactly four entries:
  `PLOW_HSACO`, `PLOW_L2_PLACE_DISPATCH`, `PLOW_MLA_PF_V2`, `PLOW_TP_NO_AUDIT`. The pre-existing
  `resolved serve configuration` line stays: it is every knob plus every ambient `PLOW_*` var
  (`PLOW_HIPCC`, `PLOW_NVCC`, toolchain paths), which records the machine. This one records the
  decision.
* **the discovered tier spec** — a DERIVED decision, which is the one thing an env dump cannot
  show. Auto-discovery (§7b) fills in `PLOW_HSACO_LOWRUNG` from the directory layout, so the
  operator sets nothing and the env record stays silent about a 24.9% difference. It now logs
  what it found, and logs the absence too, naming the build flag that produces them.

**Cost: none measurable.** 43.92 out tok/s with both records against 43.88 without, inside the
0.1% run-to-run noise established in §7c. Both run once at startup — one clap parse and one map —
and read no per-token state, so the decode step is unchanged.

The tension worth naming is not reproducibility against performance. It is the reverse: the
missing record WAS the performance loss, because a silent default cannot be A/B'd. The one real
tradeoff on this list is `PLOW_TP_NO_AUDIT` (+5.4%, at the cost of the check that catches a
silently wrong token); recording it does not resolve that, it just makes it auditable before
someone ships it by accident.

## 5b. Kimi-K2.7-Code — the critical path to serving, costed

§5's gate list is now ordered by what actually blocks: the **checkpoint encoding**, not the
emitter. `mla_ckpt_enc` refuses with `ckpt_quant_compressed-tensors` before any emit decision is
reached, so every other gate is downstream of this one.

### What the checkpoint is, measured from a shard header

Routed experts only — the `ignore` list leaves attention, shared experts, dense MLP and `lm_head`
in bf16:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `…experts.N.gate_proj.weight_packed` | `I32` | `[2048, 896]` | 8 x int4 per int32; 896*8 = 7168 = K |
| `…experts.N.gate_proj.weight_scale` | `BF16` | `[2048, 224]` | 224 = 7168/32, one scale per group |
| `…experts.N.gate_proj.weight_shape` | `I32` | `[2]` | |

`kimi_k3_prep.py` records that K3's mxfp4 experts are *byte-for-byte* what `DevOp::GemvMxfp4`
wants — `weight_packed` `[N, K/2]` u8 with the low nibble at even k, `weight_scale` one **E8M0
byte** per 32 of K. K2.7-Code shares the tensor NAMES and the group size and matches on neither
of the three things that matter:

1. **container** — `I32 [N, K/8]` against `U8 [N, K/2]`;
2. **scale dtype** — `BF16` against `E8M0` (a power-of-two exponent byte);
3. **value encoding** — int4 symmetric (uniform) against e2m1 (non-uniform float4).

So this is a requantization, not a relabelling.

### The three routes, with the capacity each needs

1.015 T routed-expert params (384 experts x 3 mats x 2048 x 7168 x 60 MoE layers):

| route | total | GB/rank TP8 | % of a 206.1 GB card | arm |
|---|---:|---:|---:|---|
| int4 g32 + bf16 scale (as shipped) | 571 GB | 71.3 | 34.6% | **none — no MoeEnc variant** |
| requantize to mxfp4 g32 + e8m0 | 539 GB | 67.4 | 32.7% | shipped (`MoeEnc::Mxfp4`) |
| dequantize to fp8 e4m3 + [128,128] block | 1015 GB | **126.9** | **61.6%** | shipped (`MoeEnc::Fp8Blk`) |

**Measurement overturned the first recommendation. The native int4 arm is the route.**

This section originally argued for the fp8 route because it reuses a qualified arm. Two facts
measured afterwards removed both conversion routes:

**1. fp8 does not fit on this host.** 1015 GB of experts plus the non-expert weights is ~1.03 TB.
`/workspace` has **881 GB free** (8.7 TB, 90% used, 2.1 TB of it models). The route that reuses
the shipped arm cannot be materialised here without deleting other checkpoints.

**2. mxfp4 costs 4.4x the error of fp8.** Same verifier, same expert, same round-trip:

| route | mean rel. err | max rel. err | total | GB/rank TP8 | fits? |
|---|---:|---:|---:|---:|---|
| int4 g32, native arm | **0** | **0** | 571 GB | 71.3 | yes |
| block-fp8 `[128,128]` | 2.26% | 3.5% | 1015 GB | 126.9 | **no — 881 GB free** |
| mxfp4 g32 | **9.96%** | 15.2% | 539 GB | 67.4 | yes |

e2m1 has eight magnitudes — `{0, .5, 1, 1.5, 2, 3, 4, 6}` — against int4's sixteen uniform
levels, so the remap discards information the checkpoint carries however the scale is chosen.
~10% RMS on the routed-expert weights of a model QAT-trained for int4 is not a first-bringup
risk worth taking, and it is the same order as int4's own 12.5% step rather than a fraction of
it.

So the ordering is: **build the int4-group-32 arm.** It is the only route that is exact, it is the
smallest of the three, it fits with the most headroom, and it needs no 555 GB conversion pass at
all — the checkpoint serves as shipped. The kernel work it requires is real, but it is now the
*cheapest* path to a correct Kimi serve rather than the most expensive, because the two routes
that avoided kernel work are respectively unmaterialisable and inaccurate on this host.

`scripts/kimi_k27_prep.py` keeps both conversions and the verifier: they are how the above was
established, they price any future host with more disk, and the exact dequantizer in it is the
reference an int4 arm has to match.

### Gates after the encoding, unchanged in substance

Full-model emit (`kimi_emit_block` is `--block`-only; the `glm_main` analogue is unwritten), the
derived MLA tensors a prep must write (`q_absorb`, `kv_a_latent`, `k_rope`, `q_rope`, `v_absorb` —
`kimi_k3_prep.py` is the working template, and it already handles this checkpoint's
`language_model.model.` wrapper prefix), the tokenizer (`tiktoken.model`, no `tokenizer.json`),
and the dense-arm prefill fault localised in §5.

## 7e. The 70k-context target — where plow actually stands

Cross-hardware reference, from a vLLM run on H200 with GLM-5.3-FP8 at ~70k input /
~700 output, concurrency 20, 100 prompts (workload details and speculation caveat in §7f):
**273.67 output tok/s, 27,269 total tok/s, TTFT median 905 ms, TPOT median 77.82 ms.**

plow could not accept that workload at all before this: the frozen blob is `--max-ctx 18432` and
a request above the compiled context is refused, so the first step was re-emitting at 81920.

### Result

20 prompts, 70k input, concurrency 20, TP8:

| arm | out tok/s | total tok/s | TTFT med | TPOT med |
|---|---:|---:|---:|---:|
| chunk 8192, no packed route | 16.09 | 1,625 | 408 s | 439 ms |
| chunk 2048, no route | 15.91 | 1,606 | 554 s | 177 ms |
| **packed prefill firing** | **16.76** | 1,692 | 533 s | 173 ms |
| *vLLM target* | *273.67* | *27,269* | *0.9 s* | *77.8* |

**Decode already beats the target and prefill misses it by ~10x.** A single-stream 70k request
measures TPOT **35.2 ms** against the target's 77.8. Total throughput is not decode-limited.

### A wrong diagnosis, corrected

Median TTFT at concurrency 20 is almost exactly 20x the single-request TTFT, which reads like
serialized prefill. It is not. ONE 70k prefill takes 27.5 s — 2,550 tok/s — so twenty of them is
~549 s of work at that rate against 835 s measured. The GPU is saturated by a single request;
there was no idle capacity for overlap to reclaim, which is why co-packing bought 4% and not 10x.
Check the single-request rate before attributing a concurrency gap to scheduling.

### Packed prefill needs FIVE preconditions, and each fails differently

Worth recording because four of the five fail silently or name only themselves:

| # | precondition | symptom when missing |
|---|---|---|
| 1 | family objects built (`PLOW_PACKED_PREFILL_CONSUMERS=1`) | none — `load_packed_family` returns `Ok(None)` on a missing file. gfx942 had never built them |
| 2 | `PLOW_PACKED_PREFILL_ROUTE=1` | "every prefill rung refuses route=false" |
| 3 | `--pf-batch` | — |
| 4 | chunk **strictly narrower** than the widest rung | `admit` packs whole spans into one rung; chunk 8192 on an 8192 ladder admits exactly one span and `packed.len() >= 2` never fires |
| 5 | `PLOW_EMIT_PACKED_PREFILL=1` at EMIT | "packed-prefill MLA consumer is in a mixed segment" |

### Decode width and context multiply — the OOM is KV, not activations

Activations are single-block and address-reused; they are megabytes. `kv.{l}.ckv` is
`dbatch x ctx x dk` **per layer**, x78, and none of it is aliasable: layer `l`'s history is re-read
at every later step and each slot owns its own. At ctx 81920 that is 58.9 GB at `dbatch=8` and
117.8 GB at 16 — against 99.9 GB of weights on a 206.1 GB card, so M=16 OOMs
(`hsa_amd_memory_pool_allocate(1207959552)` / `HSA_STATUS_ERROR_OUT_OF_RESOURCES`) and M=8 fits.
Long context therefore CAPS decode width: this model affords M=8 at 80k, so concurrency-20
traffic decodes in 8-wide batches whatever the ladder says. A 32768 prefill rung OOMs for the
same reason (prefill scratch scales with rung width).

### Where the 10x actually is

Prefill FLOPs per rank for one 70k request, from the geometry:

| term | FLOPs/layer | share |
|---|---:|---:|
| attention (causal, n^2) | 2.01e13 | **64.5%** |
| projections + MoE (linear in n) | 1.10e13 | 35.5% |
| x78 layers | **2.43e15** | |

At 27.45 s that is **88 TF/s per rank**, against §4's measured ceiling of 160–304 TF/s on the
expert shapes and 1057 TF/s dense. Two independent deficits:

* **~2.7x — dense attention.** vLLM serves this checkpoint with DSA armed (`index_topk 2048`);
  plow always emits the dense `FlashMlaPrefill`. Sparse selection would cut the n^2 term ~34x,
  which is 2.7x off the TOTAL. `FlashGatherPrefill` exists and is correct, but nothing produces
  its per-query `idx` array (§4's note), so the win is unclaimed.
* **~3.7x — prefill kernel efficiency**, the residue after that.

Neither is a knob. The target is reachable only through a T-row indexer feeding
`FlashGatherPrefill` plus prefill kernel work; no combination of ladder, chunk, tier, route or
audit setting closes a 10x deficit in work done.

### Frozen

`scripts/freeze_serving_set.sh` wrote the packet, the 53 objects + 4 tier dirs, the HSA-linked
`plowrt` and the serve replay to one directory (455 MB, pairing `0x9fd0e880fb6fbf09`), so the
numbers above have an artifact behind them rather than a scratch path.

## 7f. Where the prefill gap is, measured against AITER's own kernel

§7e put ~10x of the 70k deficit in prefill. `PLOW_PREFILL_SEG_TIMING` then splits prefill itself,
and the split is not subtle. Per layer, per 8192-row chunk (156 segments = 78 layers x 2):

| segment family | critical_us |
|---|---:|
| `flash_interpreter` (attention) | **~31,500** |
| `interpreter` (MoE + projections) | ~10,700 |

**Attention is ~75% of prefill wall time.** At the last chunk's shape (6465 queries x 72001 keys,
8 heads/rank, QK+PV) that segment does ~3.8e12 FLOP in 31.5 ms = **~121 TF/s**, against the
549-630 TF/s bf16 GEMM rate §4 measures on this part — **~20% of the achievable matrix rate.**

Two things this rules out, both measured rather than argued:

* **Not tile selection.** A GEMM tile campaign re-run against this exact recipe took the emit from
  "all analytical" to "**all 4944 tiles chosen BY MEASUREMENT**". Prefill went 2,550 -> 2,529 tok/s.
  No change, the second time tile coverage has come up empty here (§7c).
* **Not the absorbed/materialized choice.** plow's prefill already uses the absorbed form
  (`MlaMergeFold` folding `derived.v_absorb`, `nsplit=1`), so it is not making the naive
  materialize-K/V mistake.

### Against AITER's shipped gfx942 kernel

AITER's MLA prefill for DeepSeek's 192x128 shape is hand-written GCN assembly shipped as `.co`
(`hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd192x128_bf16_causal_*.co`; no source is published, so the
figures below are from its manifest CSV, its host launcher, ELF metadata and disassembly).
Disassembling plow's `interp_flash_gq.elf` the same way:

| | AITER `hd192x128` | plow `interp_flash_gq` |
|---|---|---|
| workgroup | 256 thr / 4 waves | 256 thr / 4 waves |
| VGPR | 512 (1 wave/SIMD) | 512 (1 wave/SIMD) |
| LDS | 65536 B | 58376 B |
| Q tile (BM) | 128 | 128 (`FA_BM`) |
| KV tile (BN) | 32 | 32 (`FA_BKV`) |
| MFMA | 216 x `32x32x8_bf16`, only | 272 x `16x16x16_bf16` + 176 x `32x32x8_bf16` |
| **global->LDS DMA** | **132 x `buffer_load_dword … lds`** | **0** |

**The tiling already matches. The staging does not.** AITER moves K/V global->LDS without the data
ever entering a VGPR; plow stages it through registers, in a kernel whose register file is already
fully committed (512 VGPR at one wave per SIMD). That is the shape of a kernel pinned at 20% of
matrix rate.

plow ALREADY HAS the primitive: `cp_async16` in `runtime/amd/amd_common.h`, over
`__builtin_amdgcn_global_load_lds`, whose own comment says it "writes straight into LDS with NO
VGPRs, which is the only reason this is [worth it]". Its call sites are all in `op_gemm_common.h`. The
attention path never uses it. So the GEMMs stream and the flash kernel — 75% of prefill — does not.

### Ranked, from the AITER comparison

1. **`cp_async16` in the flash K/V stager.** The one verified structural difference, in the kernel
   that owns 75% of prefill. plow has the primitive and uses it elsewhere.
2. **Causal head/tail Q-tile pairing.** AITER pairs an early and a late Q tile per workgroup so the
   triangle balances (`get_grid_dim`, `tg_div = mask ? 2 : 1`) — and **explicitly disables it for
   192x128 on gfx942**. It is unclaimed headroom in their kernel too, and the imbalance is large at
   an 8192-row chunk against 70k of context.
3. **MFMA shape.** AITER's materialized kernel is `32x32x8` exclusively; it uses `16x16x16` only for
   the *absorbed* skinny-M kernels. plow's flash is 60% `16x16x16`.

### Adapted, and measured: +21.7% prefill

`cp_async4` (`amd_common.h` [LDS-DMA-4B]) + `FA_LDS_DMA` (`op_attention_common.h`, opt-in via
`PLOW_FA_LDS_DMA=1`). Single request, 70k input, TP8, same packet and serve env; the only
difference is a flash object built with the axis on:

| | baseline | + direct-to-LDS | Δ |
|---|---:|---:|---|
| prefill | 2,529 tok/s | **3,078 tok/s** | **+21.7%** |
| TTFT (70k) | 27,678 ms | **22,742 ms** | **−17.8%** |
| TPOT | 35.24 ms | 35.14 ms | unchanged |

TPOT not moving is the control: this is a prefill-staging change and decode does not use the path.

**`cp_async16` could not supply it.** It asks for 16 B/lane, which is CDNA4-only, so its CDNA3 arm
is a VGPR-staged copy — it spends exactly the registers the technique exists to save. gfx942
implements 4 B/lane, and a HIP probe confirms clang emits `global_load_lds_dword` for it.

**Issue count is not the lever.** Unrolling the staging loops took the object from 6 to 36
direct-to-LDS sites and prefill did not move (22,745 vs 22,742 ms). The win is removing the VGPR
staging, not the number of issues — so AITER's 132 sites are a consequence of their unrolling, not
the reason their kernel is fast. `unroll 1` is kept: same throughput, less code.

Three gates caught real defects on the way, none of which would have shown up as a wrong number:
a fallback that loaded FP8KV as bf16 (the ASM contract refused it — that arm dequantizes WHILE
staging); a blanket `static_assert` that failed the build for a 64-wide rope tile which cannot use
a 128-bf16 issue (now `if constexpr`); and a dropped preprocessor `#else` that removed the loop
header at `FA_LDS_DMA=0`.

Remaining gap to §7e's target is still large — this is ~22% of a ~10x deficit — but it is the
first measured movement on the prefill bottleneck, and it came from the one structural difference
the AITER disassembly identified.

### The single-stream win does NOT transfer to the target concurrency

The same object, at the actual target workload (70k in / 700 out, concurrency 20, TP8):

| arm | out tok/s | total tok/s | TTFT med | TPOT med |
|---|---:|---:|---:|---:|
| baseline (chunk 8192, no route) | 16.09 | 1,625 | 408 s | 439 ms |
| packed prefill firing | 16.76 | 1,692 | 533 s | 173 ms |
| **+ direct-to-LDS staging** | **16.53** | **1,670** | 496 s | 191 ms |
| *vLLM target* | *273.67* | *27,269* | *0.9 s* | *77.8* |

**+21.7% single-stream, 0% at concurrency 20.** That is the result, and it is worth more than the
21.7% was: a prefill kernel improvement measured in isolation did not move the saturated
multi-request case at all. Whatever governs throughput at concurrency 20 is not the flash
kernel's staging, so quoting the single-request figure as progress toward this target would have
been wrong.

It also re-opens a question §7e closed too early. §7e concluded the GPU is saturated by one
request because 20 requests took ~20x one request's time. That is consistent with saturation, but
it is equally consistent with a per-request serialization that a faster kernel cannot help —
and the fact that a 21.7% faster prefill kernel bought nothing at concurrency 20 is evidence for
the second reading, not the first. The next diagnostic is not another kernel: it is finding what
20 concurrent prefills contend on that one does not.

### The concurrency ceiling is kernel-independent — 1,960 tok/s

Backing the aggregate prefill rate out of the two concurrency-20 runs (total input over the
benchmark duration minus the decode time its own TPOT implies):

| arm | chunk | single-stream prefill | **aggregate prefill @ conc 20** |
|---|---:|---:|---:|
| packed prefill, no DMA | 2048 | 2,529 tok/s | **1,960 tok/s** |
| **+ direct-to-LDS staging** | **2048** | **3,078 tok/s** | **1,951 tok/s** |
| + direct-to-LDS staging | 8192 | 3,078 tok/s | 1,962 tok/s |

The middle row is the single-variable A/B. The first pair compared here differed in CHUNK as well
as in the object (8192 against 2048), so it isolated nothing; re-run with the chunk held at 2048,
a kernel 21.7% faster single-stream produces an aggregate of 1,951 against 1,960 — no difference,
and if anything marginally lower.

**Two kernels 21.7% apart land on the same number.** And that number is BELOW the slower kernel's
single-stream rate: concurrency does not merely fail to help prefill here, it costs 22%. A ceiling
that does not move when the kernel underneath it gets faster is not a kernel ceiling.

The mechanism is in `serve/mux.rs`, and it is stated in the source: **"ONE prefill chunk per tick,
oldest pending request first"**, and "PREFILL AND DECODE NOW SHARE THE TICK". Sharing the tick was
itself a fix — before it, a tick was *either* a prefill *or* a decode and every decode stream
stalled for the whole of someone else's prefill (measured 49.3 tok/s at concurrency 16 against
91.3 under `amd-bench`). But the pairing is one-to-one: each tick advances exactly one request's
prefill by one chunk, and also runs a decode step, which at concurrency 20 costs ~190 ms. Prefill
progress is therefore bounded by the tick rate rather than by how fast a chunk computes, which is
exactly the invariance the table shows.

There is no knob for it. `PLOW_DSTEP_EVERY` is a per-step *timing* interval, not a scheduling
cadence. Changing it means changing the mux: admit more than one prefill chunk per tick, or
decouple the decode cadence from the prefill cadence, both of which have to keep the property the
shared tick was introduced to get.

**This supersedes §7e's reading.** That section concluded the GPU was saturated by a single
request, because twenty took ~20x one request's time. The evidence now says otherwise: if the GPU
were saturated, a 21.7% faster prefill kernel would have produced ~21.7% more aggregate
throughput. It produced 0.1%. The 20x is a scheduling artifact, not a saturation one, and the next
work on this target belongs in the scheduler.

### Lifting the cap: +13% throughput, and the decode latency it protected never appeared

`amd_prefill_tick_cap` returns `interleave` — **default 2048** — as soon as ANY request is
decoding. Prefill therefore advances at most 2048 rows per tick however large the chunk is, which
is why an 8192-row chunk behaved like a 2048-row one and why a 21.7%-faster flash kernel produced
no aggregate difference. `PLOW_PF_INTERLEAVE=0` maps to `u32::MAX`.

| in=70k, conc=20, TP8 | out tok/s | total tok/s | agg prefill | TTFT med | TPOT med |
|---|---:|---:|---:|---:|---:|
| capped at 2048 (default) | 16.53 | 1,670 | 1,951 | 496 s | 177.0 ms |
| **uncapped** | **18.67** | **1,886** | **2,239** | **466 s** | **177.8 ms** |
| | **+13.0%** | +13.0% | **+14.8%** | −6% | unchanged |

**The cap was paying for nothing on this workload.** It exists to keep a long prefill from
stalling live decode streams, and that is a real hazard — the shared-tick comment records 49.3
tok/s at concurrency 16 before prefill and decode were interleaved. But here TPOT does not move
when the cap is lifted, so the latency it was protecting did not materialise, while the throughput
it cost was 14.8% of prefill. There is also headroom by construction: TPOT is 2.2x better than the
target being compared against, so decode latency is the one budget this workload can spend.

It is not the whole gap. Uncapped aggregate prefill is 2,239 tok/s against a single-stream 3,078,
so ~27% is still lost to concurrency after the cap is gone.

### The full 2x2: the cap only bites when the chunk exceeds it

Two levers, crossed, at 70k / conc 20 / TP8 (all on the direct-to-LDS object):

| prefill chunk | per-tick row cap | co-packing | out tok/s |
|---|---|---|---:|
| 2048 | 2048 (default) | fires, `spans=2` | 16.76 |
| 2048 | uncapped | fires, `spans=2` | 16.67 |
| 8192 | 2048 (default) | never fires | 16.53 |
| **8192** | **uncapped** | never fires | **18.67** |

**The cap is only binding when the chunk is larger than it.** At chunk 2048 lifting it changes
nothing (16.76 -> 16.67); the whole +13% comes from letting an 8192-row chunk actually deliver
8192 rows in a tick instead of being truncated to 2048. Stated the other way round: with the
shipped default, `PLOW_PF_CHUNK` above 2048 buys nothing at all once anything is decoding, which
is a silent interaction between two knobs that are documented independently.

**Co-packing is not the lever here.** It DOES fire — `AMD packed prefill fired spans=2 program=3`
— and always at exactly two spans, which is the bootstrap defect
`docs/amd/gemma4-31b-mi300x.md` records: a pack can only consider a slot that already holds a
prefill cursor, the isolated path is what grants one, and the mux skips that path on any tick where
a pack ran, so N simultaneous arrivals bootstrap to a two-member pack and stop. Two spans is worth
less than the larger chunk it costs: the packing arms (16.67-16.76) lose to the non-packing
uncapped arm (18.67). Fixing the bootstrap to reach 4 members is the open question; two members is
not worth the chunk it requires.

(An earlier note here claimed co-packing never fires. That was a grep for `advanced`, which is a
DEBUG line; the INFO marker is `fired`. It fires.)

### Two nulls that bound where the remaining 27% is not

**`PLOW_PF_DEFER_DECODE=1` adds nothing** once the interleave is uncapped: 18.61 out tok/s against
18.67, TPOT 173.3 against 177.8. That is not a surprising measurement, it is a redundant one —
`amd_prefill_tick_cap` returns `u32::MAX` when `interleave == 0` OR `defer_decode`, so with the
cap already lifted the second flag cannot change the cap it shares. Recorded because "defer decode
during prefill" is an obvious thing to reach for next, and it is already covered.

**Co-packing is armed but never fires.** The serve log carries `packed prefill armed route` on all
8 ranks and no `AMD packed prefill advanced` on any of them. All five emit/build/runtime
preconditions are satisfied (§7e) and the route still does not pack a launch, which points at the
second-order defect `docs/amd/gemma4-31b-mi300x.md` already records: co-packing can only consider a
slot that already holds a prefill cursor, the isolated prefill path is what creates one, and the
mux skips that path on any tick where a pack ran — so a burst of N fresh requests bootstraps to a
two-member pack and stops. With 20 simultaneous arrivals that is the shape here.

So the residual ~27% (2,239 aggregate against 3,078 single-stream) is **not** the tick cap, and
**not** decode stealing prefill ticks. The unexamined candidates are the one-slot-per-tick pick
itself (`amd_prefill_pick` advances a single request's chunk per tick even uncapped) and the
co-packing bootstrap above — both scheduler-side, neither a kernel.

### fp8 KV: the memory is there, the arm is not

At ctx 81920 the bf16 latent cache is 58.9 GB/rank and fp8 would be 29.4, which is exactly what a
decode ladder of 16 needs to fit (58.9 + 99.9 = 158.8 GB against 217.7 for bf16 at 16, on a
206.1 GB card). It would also halve the KV bytes the flash kernel reads during prefill. Two
constraints at once, so it was worth trying.

It cannot be emitted for this workload:

```
PLOW_GLM_FP8_KV=1 with a batched decode program (rows=2): the fp8 latent writer's
batch-ring form is unvalidated on GLM. Emit the ladder blob without fp8-KV.
```

fp8 KV forces `PLOW_DECODE_BATCH_LADDER=1`. At concurrency 20 that trades a decode batch of 8 for
a batch of 1, which costs far more than the KV it saves. The capability gap is named and specific
— the batch-ring form of the fp8 latent writer — not a knob.

### The operand budget is what stops features combining

Three separate attempts in this campaign died the same way, and it is worth stating as one fact
rather than three incidents. `t[7]` and `i[6]` are the last free operand slots on the MLA decode
op, and **DSA, fp8-KV and the fused q-rope all need them**:

* `PLOW_GLM_DSA=1` + `PLOW_GLM_FUSE_ROPE=1` — "the q-rope fold needs t[7] for the cos table and
  i[6] for the sin handle, which that arm already spends (GATHER: idx/top_k)"
* `PLOW_GLM_FP8_KV=1` + `PLOW_GLM_FUSE_ROPE=1` — the same message, with `fp8-KV: kv_scale`
* and DSA additionally requires single-row decode, as fp8-KV does

So the shipped recipe's `PLOW_GLM_FUSE_ROPE=1` is not free: it forecloses both sparse attention
and fp8 KV. Any future attempt at either has to drop it first, and a packet ABI with more operand
room would remove the exclusivity entirely. That is a structural note for whoever picks this up —
it is not visible from any one flag's documentation.

### Sparse attention cannot close this gap — the LINEAR term is the constraint

Fit `T(n) = a·n + b·n²` to the concurrency-1 scaling sweep (639.54 ms at 4096, 2902.27 at 16384,
20866.24 at 65536): the linear term is the MoE and projection GEMMs, the quadratic is attention.

```
a = 0.13142 ms/token      b = 2.853e-6 ms/token^2
at n = 70000:  linear 9,199 ms (40%)   attention 13,979 ms (60%)   total 23,178 ms
```

**Set the attention term to zero and plow still tops out at 1000/a = 7,609 tok/s.** vLLM reaches
**27,000 tok/s with attention included**. So plow is **3.5x short of the target even with a
perfect, free attention kernel** — and that bound holds however good any sparse-prefill
implementation turns out to be.

This supersedes two earlier claims in this document and one in the campaign log:

* that sparse attention was worth ~2.7x of the total — it is worth at most the 60% attention share,
  and the DSA research puts the realistic ceiling near 6x on that share rather than 34x;
* that DSA sparse prefill is "the single highest-value item" — it is not. Even done perfectly it
  leaves a 3.5x deficit.

**The binding constraint is the MoE and projection GEMM path**, and §4 already measured why: the
routed-expert shapes reach 160-304 TF/s where the same library reaches 1,057 TF/s on the dense
shapes on this part, i.e. 15-29%. At top-8 of 256 over a 4096-row chunk each expert sees M=128, and
that is where the model's weight lives — 75 of 78 layers.

So the ordered work for this target is:

1. **The grouped/fused MoE GEMM at small per-expert M.** The reference exists and targets this
   part: CK's `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8_blockscale.cpp` with
   `Scale_Block_{M,N,K} = 1,128,128` — plow's exact scale layout — device-side routing through
   `p_sorted_token_ids`/`p_sorted_expert_ids`, and `gufusion` pipeline variants that fuse the
   gate/up GLU. Source, not disassembly.
2. Sparse prefill attention, for the 60% above it.
3. The scheduler residue (one-slot-per-tick, the two-span co-packing bootstrap).

Doing (2) before (1) cannot reach the target, which is worth knowing before anyone spends a month
on an indexer.

### The MoE kernel is not shape-limited — a falsified prediction

The expert GEMM's efficiency depends on M per expert = `chunk x top_k / n_exp`, and the library
ceiling at those shapes climbs steeply with it (`bringup_ceiling.py --model glm53 --tp 8 --rows N`):

| chunk | M/expert | gate/up | down |
|---:|---:|---:|---:|
| 4,096 | 128 | 303.6 | 160.2 TF/s |
| 8,192 (shipped) | 256 | 478.9 | 310.4 |
| 16,384 | 512 | 681.1 | 583.0 |
| 32,768 | 1024 | 805.1 | 730.6 |

**Prediction: chunk 8192 -> 16384 should be worth 1.4-1.9x on the MoE share.** A 16384 rung was
emitted (32768 still OOMs on prefill scratch at ctx 81920) and served at chunk 16384.

**Measured: +3.3%.** 19.28 out tok/s against 18.67; aggregate prefill 2,239 -> 2,347.

The prediction is falsified and that is the answer: **plow's MoE prefill is not shape-limited.**
The library gains 1.4-1.9x from exactly this M change and plow gains 3%, which can only mean plow
is nowhere near the shape ceiling to begin with. The arithmetic agrees — §7f's fit gives the whole
linear path 0.13142 ms/token, i.e. **~95 TF/s per rank** against a library reaching 310-681 TF/s at
the same shapes. **3-7x off, and it is kernel quality, not the shape it is handed and not the
scheduler.**

That closes the diagnosis. Ranked, with the evidence for each:

1. **The MoE / dense-GEMM prefill kernel, ~3-7x off its own shape ceiling.** Bounds everything:
   even a free attention kernel leaves 3.5x. CK ships a source reference at plow's exact scale
   layout — `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8_blockscale.cpp`,
   `Scale_Block_{M,N,K} = 1,128,128`, device-side routing via `p_sorted_token_ids` /
   `p_sorted_expert_ids`, and `blockwise_gemm_pipeline_xdlops_moe_blockscale_b_preshuffle_gufusion_*`
   fusing the gate/up GLU.
2. **Sparse prefill attention**, for the 60% share above it — and blocked today by the operand
   budget, not by the kernel.
3. **The scheduler residue** — one-slot-per-tick, the two-span co-packing bootstrap.

Chunk 16384 is nonetheless the best measured configuration and is kept: 19.28 out tok/s, +19.8%
over the §7e baseline of 16.09.

### Reference hardware correction (2026-09-09)

The user confirmed that the `10.7.21.13:8080` vLLM run was on **H200**, not MI300X.
The earlier inference from neighbouring IP addresses was wrong. Its 273.67 output tok/s
is a cross-hardware reference, not evidence of a 14.5x software gap on identical silicon.
The reference GPU count, TP configuration and server flags were not supplied.

The supplied client used 100 prompts, seed 0, concurrency 20, random input length 70000,
random output length 700 and `--random-range-ratio 0.14`: inputs span 60200–79801 tokens
and outputs 602–799. It did not set `--ignore-eos` or `--temperature=0`. Earlier plow
cells used 20 fixed-length prompts, so those cells do not reproduce this workload.

The reference also enabled speculative decoding (acceptance length 1.25). Future plow
work excludes speculation as requested; the pasted throughput cannot be converted into
a non-speculative baseline by subtracting accepted tokens or dividing by acceptance length.

Prioritise measured plow A/B improvements with the same workload and correctness gates.
A same-hardware engine comparison would require a separate reference run.

### AITER's ASM kernel set does not cover this geometry

Worth stating before adapting anything: **GLM-5.3's head geometry has no AITER ASM kernel.** The
v3 dispatcher admits exactly `(128,128)`, `(192,128)` and `(256,256)`, and the only shipped 256/256
row is fp8 on **gfx950**. GLM-5.3 is qk 256 / v 256 bf16 on gfx942, so on ROCm it would fall to CK
— which penalizes that shape specifically: in `fmha_fwd.py`'s `get_pipelines`,
`if hdim == 256 and hdim_v == 256:` selects only `qr` pipelines while every other head dim gets
`qr_async`, losing the async direct-to-LDS pipeline. There is no vendor kernel for this geometry to
benchmark against, and the target numbers in §7e came from H200. They do not establish the size of
a software-only gap on MI300X.

## 7g. The gap is attention after all, and the sparse arm was never armed

Everything in 7f rests on a fit — `T(n) = a·n + b·n²` against three points of a scaling sweep —
and a fit cannot tell a kernel from a collective. `PLOW_PREFILL_SEG_TIMING=1` measures it
directly: the AMD TP executor already splits each prefill chunk into per-family segments and
times each one's critical path. One 70k request, chunk 8192, on the frozen best recipe:

```
c0        flash_interpreter        interpreter
          (MLA attention)          (MoE + projections + collectives)
0            293 ms                   932 ms
8192         675                      923
16384      1,054                      924
24576      1,433                      925
32768      1,807                      924
40960      2,184                      946
49152      2,563                      923
57344      2,929                      944
65536      2,454 (tail)               813
        --------                 --------
          17,391 ms                8,254 ms      total 25.6 s
```

Two facts fall straight out.

* **The linear term is exactly what the fit said.** 923 ms per 8192-token chunk is 0.1127
  ms/token against the fit's a = 0.13142. The MoE/projection path is real and it is 8.25 s.
* **Attention is 68% of the wall, not 60%, and it is the term that grows.** Per-layer flash time
  goes 3.76 ms at c0=0 to 37.6 ms at c0=57344 — dead linear in context, i.e. quadratic in the
  request.

### The target is not reachable with dense attention — an arithmetic proof

GLM-5.3 dense MLA prefill for one 70k request is

```
78 layers x 2 x 64 heads x (576 qk + 512 v) x 70000^2/2  =  2.661e16 FLOP
```

Eight MI300X at their bf16 peak (8 x 1307 TF/s) is 10.5 PF/s, so **2.55 s of attention at 100%
of peak, for one request**. The reference run does 100 x 70k prompts at 27,269 total tok/s, i.e.
**2.57 s of wall per request** — inclusive of MoE, collectives, decode and scheduling.

No kernel is 100% of peak. The reference is therefore not computing dense attention, and the
checkpoint says what it is computing instead: `index_topk: 2048`. Top-2048 selection replaces
`70000²/2 = 2.45e9` query-key pairs with `70000 x 2048 = 1.43e8` — **17.1x less attention work**,
0.149 s at peak, which leaves room for everything else in 2.57 s.

This retires the framing of 7f's "sparse attention cannot close this gap". The claim there was
that a free attention kernel still leaves 3.5x; that remains arithmetically true and the 8.25 s
linear term still has to be attacked. But it was used to rank sparse prefill *below* the MoE
GEMM, and that ranking was wrong: sparse attention is the difference between a term that is 68%
of the wall and a term that is 4%, and no amount of MoE work reaches the target while the other
two thirds stay quadratic.

### `PLOW_GLM_DSA=1` was the wrong knob — there are two gates, not one

7d recorded DSA as tried and null ("the lever never armed... `dsa()` requires `ctx > 65536`").
That is correct and it is about **decode**. `mla.rs` has a second, independent gate:

```rust
fn dsa(&self, ctx: u32) -> bool     // DECODE indexer. CROSSOVER = 65536 on --max-ctx.
fn dsa_pf(&self) -> bool            // PREFILL. `PLOW_GLM_DSA_PF`. No ctx crossover at all.
```

and the comment on `dsa_pf` states the economics plainly: prefill selection is per query token,
so it pays "as soon as the causal mean row length exceeds `index_topk`", from T ≈ 2·2048 up. The
bucket gate is `glm_dsa_pf_bucket`: `t >= 2048 && t > index_topk`. At the shipping chunk of 8192
that is **true**, and it has been true for every configuration this campaign measured. The knob
was simply never set — the campaign tested the decode gate, read "never armed", and moved on.

Emitting with `PLOW_GLM_DSA_PF=1` and nothing else changed adds four opcodes, all of them
already implemented in `interp.hip`:

```
IndexScorePf (117)  IndexSelectPf (118)  IndexUnionPf (119)  LayerNorm
```

and grows the T=8192 bucket from 2246 to 2435 ops while T=128/512/2048 are untouched — the
indexer chain appearing in exactly the rung the gate admits. The 22 indexer layers the
checkpoint carries (`index_topk_freq: 4`, layers 0,1,2 then every fourth) are all present in the
prepped lite checkpoint: `indexer.wq_b`, `indexer.wk`, `indexer.k_norm`, `indexer.weights_proj`.

The object side needs `PLOW_DSA_PF=1`, which arms the gathered body of the V2 MLA prefill. It is
opt-in for a real reason recorded in `build_gfx942.sh` — the megakernel inlines every arm and its
register allocation is the worst case over all of them, so the gathered body takes the flash
object's spill from 98 to 287 whether or not a blob ever emits a union table. A sparse blob run
against an object built without it reads no `t7` and silently runs dense, which is why the
control arm below runs the *dense* packet against the *sparse* object.

### Measured: the gather works, and it reached 21 of 78 layers

`PLOW_GLM_DSA_PF=1` packet, `PLOW_DSA_PF=1` objects, everything else the frozen recipe. The
object build is exactly one define wide — `-DPLOW_DSA_PF_ARM=1` on the four flash rows and
nothing else — and the ASM contract passed over all 47 objects unchanged, so no bless was
needed. The register cost the build script warns about (spill 98 -> 287) did not materialise
here: `interp_flash` went 454/198 to 458/202 VGPR/spill, and the object grew 180 KB to 193 KB.

Segment timing, same 70k request and chunk as the dense table above:

```
              dense        sparse        delta
 flash      17,391 ms    12,644 ms      -27.3%
 interp      8,254 ms    10,926 ms      +32.4%
            ---------    ---------
 total      25,645 ms    23,570 ms       -8.1%
```

End to end at the target cell (70000 in / 700 out, concurrency 20, TP8) that is a wash:
**18.22 out tok/s sparse against 18.22 dense**, TTFT median 465.2 s against 465.7 s, TPOT 188.4
against 177.4. Both arms ran in the same lease; the control reproduces the previous lease's
18.67 to within 2.4%.

The two halves of that table are both worth reading exactly.

**The flash number is a hit, not a miss.** 27.3% of the attention wall disappeared, and 21 of
GLM-5.3's 78 layers are `indexer_types: "full"` — 26.9%. The gathered layers went essentially
free. Nothing is wrong with the gather arm; there were simply 57 layers it never applied to.

**The interpreter number is the indexer's own cost**, +2,672 ms, and it grows with `c0` (1,021 ms
at the first chunk to 1,388 ms at the last) because the lightning indexer's score is itself
O(T·S) — a small GEMM per key, but still quadratic in the request. That is the term that ate the
attention saving.

### Why only 21 layers — a predicate that disagrees with its own comment

```rust
let sparse = glm_dsa_pf_bucket(c, t) && w.iwqb != TENSOR_NONE && nh_l == 8;
```

`iwqb` is bound only where `indexer_is_full(l)`, so `sparse` is false on every 'shared' layer.
The comment directly above it says the opposite — that "shared layers reuse the previous full
layer's union table" and that "`c_uni` threads that reuse". **`c_uni` does not exist**; it appears
nowhere in the tree but in that sentence. The reuse was documented and never implemented.

The DECODE chain does implement it, and gates the way prefill should:

```rust
if dsa {
    d.t[7] = n.iidx; // idx table (this or the last full layer's selection)
```

keyed on the model-level `dsa`, not on `full`, with `c_sel` pushed into the flash's dep list only
on full layers — the ordering carried transitively through the residual stream rather than by an
explicit edge. `n.iuni` is likewise a single buffer, not one per layer, so a shared layer reading
it costs nothing to produce.

Dropping the `iwqb` requirement (keeping it only to decide which layers RUN the indexer) is
therefore the free half of the feature: the scoring, top-k and union work is already paid on the
21 full layers, and the other 57 would gather against a table that is already sitting there.

### Extending the gather to all 78 layers: 1.88x, and wrong

The `iwqb` term looked like an oversight, because the comment sitting on top of it describes a
mechanism (`c_uni`) that exists in no other line of the tree, and because the DECODE chain gates
the same decision on the model-level `dsa` rather than on `full`. It was tried.

Dropping it moves every flash op onto the gather arm. Verified statically before spending a
lease, by reading the raw operand rather than the instruction mix — both arms are opcode 51, and
it is `t7`'s presence that selects the gather:

```
before   t7=NONE  i6=0      x57      t7=iuni i6=16384  x21
after                                t7=iuni i6=16384  x78
```

and the speedup is large and exactly where the model says it should be:

```
                dense      21 layers    78 layers
 flash        17,391 ms    12,644 ms     2,747 ms     6.3x, and FLAT in context
 interp        8,254 ms    10,926 ms    10,893 ms     unchanged — the indexer still runs on 21
              ---------    ---------    ---------
 total        25,645 ms    23,570 ms    13,640 ms     1.88x
```

Per-layer flash goes 2,960 us at `c0=0` to 4,562 us at `c0=57344`, against dense's 3,758 to
37,550: the quadratic term is gone, which is the O(T·top_k) signature and the whole point of the
feature. Attention drops from 68% of prefill to 20%, and the MoE/linear term becomes 80% of what
is left — the crossover this document predicted.

**And the output is wrong.** Three configurations, same probes, one lease:

```
                      short == dense   repeat stable   40k needle
 dense                     —                yes          FOUND
 21-layer gather          yes               yes          FOUND
 78-layer gather          yes               yes          MISSED
```

The short prompt agrees across all three, as it must — it rides the t=128 bucket, which the gate
keeps dense, so it only proves the object and emit are sound. The long probes are the result: the
78-layer build misses a fact planted at the FRONT of a 40k-token haystack that both other builds
recover, and degrades a repeated-phrase continuation to `' the fox over dog over over the the'` —
the right vocabulary in the wrong order.

It is DETERMINISTIC (`rep1 == rep2`), which rules out the obvious suspicion: the shared `n.iuni`
buffer is not being read out from under its writer. The transitive ordering argument holds. What
fails is the semantics — a 'shared' layer is not entitled to attend the previous full layer's
8-query union, whatever else it is entitled to reuse.

So the guard stays, with this recorded on it. The prize is quantified and it is the largest single
item left in this campaign — **1.88x on prefill, and the removal of the quadratic term entirely** —
but claiming it needs the per-layer selection semantics to be established against the reference,
not the plumbing, which already works.

### The next term, sized: the down GEMM's K is 256 because TP8 shards it

The 78-layer gather run makes this concrete rather than hypothetical: with attention at
2,747 ms the linear term is 10,893 ms, i.e. **80% of prefill**. Segment timing prices it exactly
even in the dense run -- 923 ms per 8192-token chunk over 78 layers is 11.8 ms per layer per
rank. Against the
active-parameter count that is

```
per layer, per rank: 8192 tokens x (9 experts x 3 x 6144 x 2048 + attn proj) x 2 / 8 ranks
                   = 1.03e12 FLOP in 11.8 ms = 87.7 TF/s
```

which is **3.4% of the part's fp8 peak** and 2.5% of its measured HBM bandwidth — so the path is
neither compute- nor bandwidth-bound, and even against plow's OWN measured 160-304 TF/s on these
shapes it is 2-3.5x down. The overhead is real and it is per-tile.

TP8 is what makes it per-tile. The two grouped GEMMs see opposite shapes on a rank:

```
                 K      N       NT = K/MPF_BK     tn = N/NB      tiles per m-tile
 gemm1 (glu)   6144    512           96               4                 4
 down           256   6144            4              24                24
```

`down`'s K is `moe_intermediate / TP = 2048 / 8`, so its k-loop runs **four** iterations — and
every fixed cost the grouped form carries (the LDS stage, two `__syncthreads()` per k-tile, the
prologue address math, the epilogue's scatter) amortizes over four, against gemm1's ninety-six.
It also emits six times as many output tiles as gemm1 while doing half the FLOPs.

The lever that follows from this is `MPF_BM`: it is `#ifndef`-guarded, already threaded into
`AX_DECODE` by `build_gfx942.sh`, `PLOW_GEOM_MARK`ed, and at 128 the arena still fits
(`(128+256)*64*2 = 49,152 B` against 64,512, DBUF=1 exactly as now). Doubling it halves the tile
count for both GEMMs and so halves the per-tile overhead the `down` arm is dominated by.

Two things must be handled together with it, which is why this is a change rather than a sweep:

* `PLOW_MOE_PF_EPI` `#error`s unless `MPF_BM == PLOW_WAVE`, because its hoist puts one m-tile row
  per lane of a single wave and reads it back with `ds_bpermute_b32`. At BM=128 the rows span two
  waves, so the hoist needs two loads and a lane-half select. Testing BM=128 with EPI=0 instead
  would confound the measurement with the loss of a hoist already measured to pay.
* `build_gfx942.sh` threads `MPF_BM` into the decode row only; the prefill/MLA-MoE objects need
  the same passthrough.

Note this is NOT the "raise M per expert" hypothesis that chunk 16384 already falsified (+3.3%).
That test doubled the ROWS per expert, which amortizes the weight STREAM; this changes the tile
the rows are cut into, which amortizes the per-tile FIXED cost. The falsified result is in fact
evidence for this one: if weight bandwidth were the constraint, halving it would not have
returned 3.3%.

#### Measured: MPF_BM=128 is -5.1%, which bounds the tile hypothesis rather than confirming it

Two objects differing in `MPF_BM` and nothing else, both with `PLOW_MOE_PF_EPI=0` because the
hoist `#error`s unless `MPF_BM == PLOW_WAVE` — comparing a BM=128/EPI=0 build against the
shipped BM=64/EPI=1 one would have measured the hoist. Same dense packet, same lease, one 70k
request each:

```
                    flash        interp        total
 BM=64  EPI=0     15,016 ms     9,381 ms     24,397 ms
 BM=128 EPI=0     15,038 ms     8,903 ms     23,941 ms
                   +0.1%         -5.1%
```

Flash is unchanged to 0.1%, which is the control working: the lever must not touch attention.
Both arms return the correct continuation, so the wider tile is numerically sound — worth
recording, because it means the 64-row assumptions in the epilogue are the only thing standing
between this and a shipping config.

**-5.1% is the useful part, and it is a bound, not a win.** Halving the tile count halves every
per-tile fixed cost the short k-loop cannot amortize, and it bought 5%. So the per-tile overhead
is NOT where 87.7 TF/s against a 2,615 TF/s fp8 peak goes. Together with the chunk-16384 test
(+3.3%, which doubled rows per expert and so halved the weight stream), the two amortization
hypotheses are now both bounded at about 5%:

* not the weight stream — halving it returned 3.3%;
* not the tile count — halving it returned 5.1%.

What is left is the inside of the k-loop, which is where `PLOW_MOE_PF_PIPE`'s note already
locates it: the shipped loop's waits are dataflow-forced, draining to `vmcnt(0)` three or four
times per k-tile where aiter's `fmoe` sustains 13-39 outstanding loads on partial waits. That is
a kernel rewrite against a reference, not a tile parameter, and it is the honest description of
the remaining 3-7x.

Note also that BM=128/EPI=0 (8,903 ms) is still slower than the SHIPPED BM=64/EPI=1 (8,254 ms):
banking the 5% requires generalising the epilogue hoist to two waves — one load and a lane-half
select per row block instead of one `ds_bpermute_b32` — which is worth doing only alongside the
k-loop work, not for its own sake.

### Correction: the reference DOES gather on every layer

The commit above concluded that "a 'shared' layer is not entitled to attend the previous full
layer's 8-query union". **That is wrong**, and the reference implementation on this host says so
directly (`vllm/models/deepseek_v32/attention.py`, which is what serves `glm_moe_dsa`):

```python
skip_topk = max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
...
indexer = None
if not skip_topk or is_mtp_layer:
    indexer = type(self).indexer_cls(..., topk_indices_buffer, ...)
...
self.skip_topk = False          # ALWAYS, unconditionally
```

`skip_topk` decides only whether the layer **owns an indexer**. The runtime flag `self.skip_topk`
is hardcoded `False`, and `self.topk_indices_buffer` — one buffer shared by every layer — is
passed into the attention call unconditionally, whether or not `self.indexer is None`. A layer
without an indexer contributes nothing to the buffer and attends against whatever the last
indexer layer wrote. That is exactly the reuse plow's comment described and plow's code did not
do, and `index_skip_topk_offset: 3` / `index_topk_freq: 4` reproduce GLM-5.3's 21-full/57-shared
split from that formula.

So the 78-layer configuration is what the reference computes, the 1.88x is the right target, and
the needle failure is a DEFECT IN PLOW that the 21-layer configuration masks — not a semantic
boundary. Three candidate causes have been eliminated by measurement rather than argument:

* **not a race** — the degraded output is identical across repeated identical requests;
* **not the indexer weights** — `scripts/glm53_dsa_verify.py` against the loader's fp8->bf16
  upcast is **4/4 bit-identical**, `max|d| = 0.000e+00` over 18.4 M elements, which is the check
  whose own docstring warns that a scale-grid error yields "an indexer of exactly the right SIZE
  and the right general magnitude that selects the wrong KV rows";
* **not a truncated scoring range** — `IndexScorePf` reads `n.kidx[slot]`, the persistent
  per-full-layer indexer K cache sized `dbatch * ctx * di`, not the chunk-sized `kidx_pf`
  staging buffer, so the selection can see position 0 from any chunk.

What remains is either the ordering of the shared `n.iuni` buffer against its readers — the
all-layer emit's counter-graph reduction removes exactly 78 edges as "implied by a path", one per
layer — or a selection-quality effect that only shows once more than a quarter of the layers
depend on it. `PLOW_GLM_DSA_PF_SPAN` bisects those: it admits reuse only within `n` layers of an
indexer, so span 1 tests distance-1 reuse alone and span >= 3 is every layer.

### The bisect: reuse is sound, and 40 of 78 layers gather correctly

`PLOW_GLM_DSA_PF_SPAN=n` admits the gather on any layer within `n` of an indexer, so the union's
reuse DISTANCE and the SHARE of sparse layers can be moved together and read off separately:

```
 span   flash ops on the GATHER arm    40k needle    repeat continuation
   0            21 / 78                 FOUND        clean   (the shipped guard)
   1            40 / 78                 FOUND        clean
   2            59 / 78                 -            -
   3+           78 / 78                 MISSED       ' the fox over dog over over the the'
```

**span=1 is correct.** Every reuse in it is at distance 1 from the indexer that wrote the union,
40 of 78 flash ops take the gather arm — verified on the raw operand, `t7=iuni i6=16384 x40` — and
it recovers `71-ALPHA-93` from the front of a 40k haystack while returning the clean repeated
phrase. So sharing one `n.iuni` across layers is SOUND: the transitive ordering through the
residual stream holds, and a layer with no indexer of its own attends correctly against the
previous indexer layer's selection.

That retires the conclusion recorded on the guard two commits ago — "a 'shared' layer is not
entitled to attend the previous full layer's union" is false, and the reference agrees (see the
correction above). What is actually true is narrower: reuse works, and something degrades as the
sparse share grows past roughly half the model.

Two candidates remain for the span-3 failure, and the span-2 row separates them:

* **distance** — a union reused three layers later is too stale, in which case span=2 also fails;
* **share** — plow's selection is good enough for half the layers but not all of them, i.e. a
  selection-QUALITY gap against the reference (which does run all 78), in which case span=2 is
  correct or marginal and the ceiling is a property of the indexer, not the plumbing.

Either way there is now a correct configuration strictly faster than dense, which there was not
before: span=1 gathers on 51% of layers where the shipped guard gathers on 27%.

### span=1 measured at the target cell: 19.47 out tok/s, and correct

Same lease, same client, one variable — the packet:

```
                       out tok/s   TTFT median   TPOT     40k needle
 ilv0 (dense control)    18.22       465.7 s     177.4      FOUND
 span1 (40/78 gather)    19.47       420.8 s     187.6      FOUND
                         +6.9%        -9.6%
```

This is the **first correct configuration in this campaign that beats the dense baseline**, and
the first improvement in it that comes from doing less work rather than from scheduling. It also
tops the previous best cell on record (19.28, the `c16k` arm), which was dense.

The size of it is instructive and worth stating plainly rather than rounding up. Attention is 68%
of prefill, gathering removes essentially all of a gathered layer's cost, and 40 of 78 layers
gather — yet the end-to-end gain is 6.9%. Three things absorb the rest:

* the indexer's own O(T·S) score, which costs +2.67 s on the 21 layers that run it and does not
  shrink when more layers consume its output;
* the 8,254 ms MoE/projection term, untouched by any of this and now the majority of the wall;
* concurrency-20 scheduling, where TTFT improves more (-9.6%) than throughput (+6.9%).

So the lever is real, it is directionally the right one, and it is not close to sufficient on its
own — the reference's 273.67 out tok/s needs the linear term as well, which 7h sizes.

### span=2 fails: the boundary is the reuse DISTANCE, and it is sharp

```
 span   gather ops   40k needle   repeat continuation
   0      21 / 78      FOUND      clean
   1      40 / 78      FOUND      clean
   2      59 / 78      MISSED     ' quick brown fox the the..</think></think></think>}}'
   3+     78 / 78      MISSED     ' the fox over dog over over the the'
```

A layer may reuse the union written one layer earlier and not two. That is a sharper statement
than "too many layers are sparse", and it is diagnostic: the SHARE hypothesis would degrade
gradually with the layer count, and this does not — 40 layers is clean and 59 is broken, with the
only structural difference being that span 2 admits readers two layers downstream of the writer.

It is also a plow-specific failure, because the reference reuses at distances 1, 2 AND 3 by
construction (`indexer_types` runs are three long, and every layer in a run reads the same
`topk_indices_buffer`). So there is a defect with a narrow signature: something invalidates
`n.iuni` for readers more than one layer after its writer, while leaving the immediate next layer
correct. Ordering against the WRITER is not it — that is the same edge span 1 relies on, and span
1 is correct.

`PLOW_GLM_DSA_PF_SPAN` therefore defaults to **1**, not 0: span 1 is measured both faster and
correct against span 0, and a knob whose best value is known should not default to a worse one.
The whole path stays behind `PLOW_GLM_DSA_PF`, so a build that does not ask for sparse prefill is
unaffected. `=0` restores the indexer-layers-only behaviour; `=2` or more reproduces the failure
for whoever picks up the defect.

**Where that leaves the target.** Best correct cell is now 19.47 out tok/s against the reference's
273.67. Attention is no longer the first-order term in a span-1 build; the 8,254 ms MoE and
projection path is, and 7h sizes it at 87.7 TF/s per rank with both amortization hypotheses
bounded at ~5%. The two pieces of work that remain are both kernel-level: fix the distance-2
union defect (worth the rest of the 6.3x attention reduction, i.e. prefill 1.88x) and rewrite the
grouped MoE k-loop against CK/aiter's pipelining (worth 3-7x on what would then be 80% of the
wall). Neither is a configuration change, and neither was reachable by the flag search this
campaign began with.

## 7i. The MoE GEMM is not the constraint — it is 21.6% of the term blamed on it

Everything this document has said about the linear path, including 7h above, derives its GEMM
rate by dividing the whole `interpreter` segment by the MoE's FLOP count. That segment holds the
MoE and dense GEMMs, the router, the align op, the gather and scatter, the norms AND the TP
collectives, so 87.7 TF/s was never a measurement of the GEMM — it was a measurement of all of
them at once, attributed to one.

`PLOW_MOE_PF_ABL=1` caps the grouped prefill GEMM's k-loop at a single tile. Wrong output by
construction — it computes a 1/NT slice of each dot product — and it must never touch a serve
asset, exactly like `PLOW_MLA_PF_ABL` and `PLOW_XR_NOWAIT`. Everything else is still paid:
staging, routing, epilogue, scatter, collectives. Two objects differing in that one define, same
packet, same lease, one 70k request each:

```
                     flash        interp        total
 full              15,003 ms     8,251 ms     23,254 ms
 k-loop ablated    14,749 ms     6,471 ms     21,221 ms
                    -1.7%        -21.6%
```

The full arm reproduces the 8,254 ms baseline to 3 ms, and flash moves 1.7% — the controls hold.

**The entire MoE k-loop is 1,780 ms of the 8,251 ms linear term.** `down` runs NT=4 so a quarter
of its inner work survives the cap; gemm1 runs NT=96 so essentially none of its does. Against the
work actually done,

```
75 MoE layers x 9 active experts x 3 matrices x 6144 x 2048 x 2 x 70,000 / 8 ranks
  = 4.459e14 FLOP per rank,  in 1.780 s  =  251 TF/s per rank
```

which sits inside the 160-304 TF/s band 4 already measured for these exact shapes. **The grouped
MoE GEMM is performing as expected.** It is not 3-7x off, there is no 4x hiding in its k-loop,
and the CK port that 7f, 7g and 7h all rank first would, at its theoretical best, remove 1.78 s
from a 25.6 s prefill — 7%.

This supersedes, in this document alone: 7f's "the binding constraint is the MoE and projection
GEMM path"; 7g's "the MoE/linear term is 80% of prefill and 87.7 TF/s"; and 7h's entire framing,
whose MPF_BM result (-5.1%) and chunk-16384 result (+3.3%) are now explained rather than merely
bounded — both moved a small term, because the term they moved was small.

### Where the 6,471 ms actually is

Not measured individually here, but the candidates are enumerable and two are already on record:

* **the dense projections.** `build.json`'s own dispatch audit flags 21 occupancy findings on
  this blob, `GemmSmall` shapes filling **10.5% to 42.1%** of their dispatch — e.g.
  `8192x64x6144 @t8192` at 42.1% (128 tiles over 304 CUs) and `2048x64x6144` at 10.5%. A GEMM
  that occupies a tenth of the machine is not a kernel-quality problem, it is a shape problem,
  and the emitter already knows it;
* **the TP collectives**, two per layer over `[8192, 6144]` bf16 — 100 MB each, ~175 MB per rank
  per layer of xGMI traffic after the two-shot;
* **the MoE fixed costs the ablation deliberately keeps**: the align op, the row_token gather,
  the padded-row scatter and the epilogue, plus `fu_g` — `[65536, 6144]` bf16 of gathered
  activation written and read back per layer.

The ranked work for this target therefore changes. It is no longer "port CK's grouped GEMM". It
is: find which of those three owns the 6,471 ms, with the same ablation discipline used here —
one define, one variable, controls that hold — and attack that. The instrument for the first is
already in the tree (`PLOW_XR_NOWAIT` prices the collective's synchronization); the occupancy
findings for the second are already computed on every emit and have simply never been acted on.

### And it is not the collective's synchronisation either — 1.3%

`PLOW_XR_NOWAIT=1` deletes both two-shot rendezvous waits from the prefill collective
(op_collective.h). Wrong output by construction, same ceiling-instrument contract. Same packet,
same lease:

```
                     flash        interp        total
 full              15,003 ms     8,253 ms     23,256 ms
 xr nowait         14,912 ms     8,144 ms     23,056 ms
                    -0.6%         -1.3%
```

The full arm now reproduces across three independent leases at 8,251 / 8,253 / 8,254 ms — a
±3 ms control on an 8.25 s measurement, which is what makes 1.3% readable as a real null rather
than noise.

So the decomposition of the linear term stands at:

```
 MoE GEMM k-loop         1,780 ms   21.6%   (PLOW_MOE_PF_ABL)
 TP collective sync        109 ms    1.3%   (PLOW_XR_NOWAIT)
 everything else         6,364 ms   77.1%   (by difference)
```

Two of the three candidates named in 7i are now measured and both are small. What remains in the
6,364 ms is the dense projections, the MoE's own fixed costs (align, `row_token` gather, padded
scatter, epilogue, the `fu_g` round trip), and the collective's DATA movement — `XR_NOWAIT`
removes the waits, not the 175 MB per rank per layer the two-shot actually moves.

Of those, one is already quantified and has never been acted on: this blob's dispatch audit
reports 21 occupancy findings, `GemmSmall` shapes filling 10.5% to 42.1% of their dispatch. A
projection that occupies a tenth of the machine cannot be fixed by a better inner loop, and the
emitter computes that number on every build.

**The methodological point is the durable one.** Three of this campaign's ranked-first items —
the CK GEMM port, the MPF tile, sparse attention as "worth 2.7x" — were ranked from arithmetic
over a segment total rather than from an ablation, and each was wrong about its own size by
between 3x and an order of magnitude. Every number in this section and 7i cost one define, one
rebuild and one 70k request, and the controls hold to 0.04%.

### The map, finished: 12.8% of prefill is GEMM math

`PLOW_GEMM_ABL=1` is the dense twin of the MoE instrument — it caps `d_gemm_t`'s and the
fused-8 GEMM's k-loop at one tile. Same contract, same discipline:

```
                     flash        interp        total
 full              15,021 ms     8,266 ms     23,287 ms
 dense k-loop cap  14,786 ms     6,767 ms     21,553 ms
                    -1.6%        -18.1%
```

The full arm has now reproduced across four independent leases at 8,251 / 8,253 / 8,254 /
8,266 ms — ±0.09% on an 8.25 s measurement.

With four ablations the 70k prefill decomposes completely:

```
 total                          25,657 ms
   attention                    17,391 ms   67.8%
   linear                        8,266 ms   32.2%
     MoE GEMM k-loop             1,780 ms   21.5% of linear   PLOW_MOE_PF_ABL
     dense projection k-loop     1,499 ms   18.1%             PLOW_GEMM_ABL
     TP collective sync            109 ms    1.3%             PLOW_XR_NOWAIT
     orchestration / movement    4,878 ms   59.0%             by difference

 ALL GEMM inner-loop math        3,279 ms = 12.8% of prefill
 everything else                22,378 ms = 87.2%
```

**Twelve point eight percent of this prefill is the multiply-accumulate loops.** Every kernel
this campaign proposed to rewrite — the CK grouped GEMM, the MPF tile, the aiter-style k-loop
pipeline — lives inside that 12.8%, and their combined theoretical ceiling is a 1.15x on the
request.

The dense projections run at 150 TF/s per rank over their 2.25e14 FLOP, which is consistent with
the occupancy the emitter already reports for them (10.5-42.1% of dispatch). But like the MoE
GEMM, they are too small a share to carry the target on their own.

The 4,878 ms of orchestration is now the largest single item in the linear path and the least
examined thing in this document: the MoE align op, the `row_token` gather, the padded-row
scatter and combine, the `fu_g` round trip, the norms and ropes, the per-op dispatch, and the
collective's data movement (which `XR_NOWAIT` does not touch — it removes waits, not the ~175 MB
per rank per layer the two-shot actually moves). Attention at 17,391 ms remains the larger term
overall and is 6.3x recoverable once the distance-2 union defect is fixed.

**The lesson that generalises past this model.** Four ablations, one define each, cost about
three hours and moved every ranked item in this document. The rankings they replaced were built
by dividing segment totals by FLOP counts — a method that cannot distinguish a kernel from the
sixty percent of the segment that is not a kernel, and which was wrong here by between 3x and an
order of magnitude every time it was used. The instruments are now in the tree
(`PLOW_MOE_PF_ABL`, `PLOW_GEMM_ABL`, alongside the existing `PLOW_MLA_PF_ABL` and
`PLOW_XR_NOWAIT`); the next person to rank work on this target should ablate first.

### Matched-count control: it is the DISTANCE, not the share

Span moves two things at once — the reuse distance and the number of sparse layers — so span 1
correct / span 2 wrong could not distinguish "a union goes stale two layers on" from "plow's
selection is only good enough for half the model". `PLOW_GLM_DSA_PF_DEXACT=d` admits ONLY layers
exactly `d` past an indexer, which holds the count fixed while moving the distance:

```
 config     gathering layers   reuse distance   40k needle
 span1            40                  1           FOUND
 dexact2          41                  2           MISSED
```

Forty against forty-one, and the outcomes differ. **The reuse distance is the cause.** The share
of sparse layers is not: 41 sparse layers are fine at distance 1 and broken at distance 2.

The failure is also graded rather than catastrophic — dexact2 returns `' -ALPHA-93.00...'`,
recovering most of the planted string and dropping its leading `71`, and its repeated-phrase
continuation is still clean (`' quick brown fox jumps over the lazy dog...'`) where the 78-layer
build's was not. A union that is stale or subtly wrong degrades the selection; it does not
corrupt the arithmetic.

So the defect has a narrow, actionable signature: **`n.iuni` is correct for the layer immediately
after its writer and wrong for the next one**, while nothing writes it in between and the
reference reuses it across all three layers of a run. Candidates worth checking in that order:

* the counter-graph reduction, which removes exactly 78 edges as "implied by a path" on the
  all-layer emit — one per layer — and whose implication argument runs through the residual
  stream that also carries the ordering;
* whether the flash's GATHER arm writes anything back into the union's mask words (`mlo`/`mhi`
  live inside the same `n.iuni` allocation as the positions it walks);
* the `n.iumask` scratch, which op 119 slice-indexes per in-flight workgroup and which is sized
  by `n_cu`, not by the pack count.

Closing it is worth the rest of the attention term: the 78-layer configuration already measured
2,747 ms of flash against dense's 17,391 ms, and 13,640 ms of prefill against 25,645 ms.

#### The likeliest mechanism is write-after-read, not read-after-write

Every hypothesis above assumed the risk was a reader running before its writer. The matched-count
result points the other way. In the run `0F 1F 2F 3S 4S 5S 6F ...`, layer 2 writes the union and
the NEXT writer is layer 6, so what varies with distance is not the gap behind the reader but the
slack in front of it:

```
 reader   reads union of   next writer   slack   observed
 layer 3      layer 2        layer 6       3     correct   (span 1)
 layer 4      layer 2        layer 6       2     wrong     (dexact 2)
 layer 5      layer 2        layer 6       1     wrong     (span 3 / all-layer)
```

The failure orders exactly by slack, and it degrades progressively rather than switching — span 3
loses the needle entirely and mangles the repeated phrase, dexact 2 loses one leading token of the
needle and keeps the phrase clean. That is the signature of layer 6's `IndexUnionPf` overwriting
`n.iuni` while a nearer reader's flash is still walking it: a WRITE-AFTER-READ hazard on a buffer
that is shared by every layer and rebuilt every four.

It also explains why the read-after-write argument checked out. `n.iuni`'s writer IS ordered
before its readers, transitively through the residual stream, and span 1 proves that ordering
holds. Nothing in the emit orders a writer AFTER the readers of the previous generation.

Two fixes follow, and they are cheap relative to what they unlock (prefill 25,645 -> 13,640 ms):

* **Double-buffer `n.iuni`**, alternating per full layer, so a writer never lands on the
  generation a live reader is walking. Costs one more union allocation and no synchronisation.
* **Add the WAR edge** — make each `IndexUnionPf` depend on the flash ops of the layers that read
  the previous union. Free in memory, but it serialises the indexer behind the previous group's
  attention, which is exactly the overlap the chain is built to get.

The first is preferable. Note also that `PLOW_GLM_DSA_PF_SPAN=1` is safe under this reading for a
structural reason rather than a lucky one: a distance-1 reader always has the full run of shared
layers between it and the next writer.

#### Falsified: double-buffering the union changes nothing

The write-after-read reading above is wrong, and the test that would have confirmed it refutes it.
`n.iuni` was made two buffers indexed by full-layer parity (`glm_uni_gen`), so a writer always
lands on the generation opposite the one its predecessors' readers are walking — the exact fix the
hazard argument prescribes. Verified in the packet: the flash ops alternate two distinct handles,
37 on one and 41 on the other, all 78 gathering.

The all-layer build still fails, identically:

```
 needle: MISSED   ' was without without without without without ...'
 repeat: ' the fox over dog over over the the the the the``'
```

So the mechanism is now constrained from four sides and none of them is it:

* **not read-after-write** — span 1 reuses across a layer boundary and is correct;
* **not write-after-read** — double-buffering removes that hazard entirely and changes nothing;
* **not the indexer weights** — 4/4 bit-identical against the reference dequantisation;
* **not a race** — the degraded output is identical across repeated identical requests, and it
  stayed identical after the buffering change.

What survives is that a 'shared' layer two or more past its indexer computes something wrong from
a union that is byte-correct and correctly ordered. That points at the READ side — how a layer far
from the writer indexes or interprets the table — rather than at the table's lifetime. Two things
worth checking that this campaign did not: whether `IndexUnionPf`'s per-pack header count
(`cnt = (unsigned*)uni`, `n_qt` entries before `hdr`) is sized or indexed against the WRITING
layer's geometry rather than the reader's, and whether the gathered flash's `kv_len` operand
(`d.t[6] = n.kvlen`) is per-layer state that a distant reader pairs with a stale union.

The slack correlation in the table above is therefore a coincidence of the layer pattern, not a
mechanism — with runs of three, distance and slack move together and cannot be separated by
choosing layers. Separating them needs a synthetic `indexer_types`, which the checkpoint fixes.

`PLOW_GLM_DSA_PF_SPAN=1` remains the correct, measured configuration; the defect behind spans 2
and 3 is open, and this section records the four things it is not so the next attempt starts
narrower than this one did.

## 7o. The reference could not be reproduced on this host

The historical attempt below tried to establish a local MI300X reference. The user has
since confirmed that the pasted 273.67 output tok/s / 905 ms median TTFT run used H200.
Failure of the local ROCm stack does not explain or invalidate that H200 result.

vLLM 0.28.0 from `/app/plow/.venv-vllm028`, serving `/workspace/models/GLM-5.3-FP8` at
`--tensor-parallel-size 8 --max-model-len 80000` on these eight MI300X:

```
 without AITER   loads all 141 shards, initialises the MoE, then raises
                 RuntimeError("Sparse attention indexer ROCm path is only supported on
                 AITER. Please enable aiter with VLLM_ROCM_USE_AITER=1")
                 -- vllm/model_executor/layers/sparse_attn_indexer.py:878

 with AITER      *** stack smashing detected ***: terminated
                 aborts before emitting a single log line. Reproduced twice, with
                 PYTHONUNBUFFERED=1 and PYTHONFAULTHANDLER=1 on the second.
```

So on this host, in this environment, vLLM cannot serve GLM-5.3 with sparse attention at all —
it refuses without AITER and crashes with it.

The local failure leaves no same-hardware vLLM baseline for this campaign. Keep the
recorded plow A/B deltas separate from the H200 comparison; those deltas do not require
a working local vLLM server. The host and client-workload correction in §7f supersedes
the earlier claims of identical hardware.

## 7p. Every GEMM tile in this model was chosen by the analytical model, not measured

`plowc tune status --gpu MI300X` says it plainly, and it has been saying it on every emit this
campaign ever ran:

```
records     : 4619 (4619 qualified)
digest census — which build each record is keyed to:
  0f4260a582715fa8      96 records   STALE — changed: implementation, interpreter, toolchain
  ... twelve digests ...
  9994a4419860debf     288 records   STALE — changed: implementation, interpreter

*** EVERY RECORD IN THIS CELL IS STALE. ***
```

The MI300X cell held 4,619 measurements spread over twelve historical build digests and the
compiler could use **none** of them, so `pick_tile` fell back to the analytical model for all
5,070 dense-GEMM tile lookups and reported tier `portable`. Every GEMM number in 7f through 7o —
the dense projections at 150 TF/s per rank, the 10.5-42.1% dispatch occupancies, the whole
`interpreter` segment — was measured on a MODELLED tile. The emit prints this on every build; no
campaign in this document acted on it.

Re-running it is one command and about ninety seconds of GPU:

```
GLM_FULL=1 PLOW_FP8=1 ... plowc --hf-dir <ckpt> --gpu MI300X --arch gfx942 \
  --max-ctx 81920 --num-gpus 8 tune gemm --gpu MI300X \
    --obj build-amd/hsaco/db16-dsapf --samples <out.jsonl> --campaign glm53-tp8-2026-09
```

`--shapes auto` derives the shape list from the compiler's own demand rather than a hand-kept
list, which is why it needs the full emit env: the demand only exists for the whole model. It
derived **34 shapes, every one a MISS**, and they are exactly the shapes the dispatch audit
complains about — `8192x64x6144`, `8192x32x6144`, `2048x64x6144`, `128x2048x6144`. It published
**204 qualified records**, and the cell now reads:

```
  434de661e3cf28bc     204 records    204 qualified  CURRENT — the compiler can use these
selectable  : 34 op cases against the probed build
```

The next emit changed its verdict from

```
>>> tunedb: ALL 5070 dense-GEMM tile(s) chosen by the ANALYTICAL MODEL. This build is UNMEASURED
```

to

```
tunedb: all 5070 dense-GEMM tile(s) chosen BY MEASUREMENT
```

with `build.json` carrying `"tile_source": "measured", "tile_measured": 5070`. The packets differ
(`cmp` at byte 113,520,661), so the measured tiles are genuinely not the modelled ones.

**This is the technique the rest of this document was missing.** Not a new kernel — plow already
has the mechanism, and it is the designed answer to "which tile for this shape on this part". It
went stale because the digest keys on implementation, interpreter and toolchain, all of which this
campaign changed repeatedly, and nothing in the workflow re-ran it. Any future object-recipe change
(a new `-D`, a toolchain bump) stales the cell again and silently reverts every shape to the model.
Re-running the campaign belongs in the build/freeze procedure, not in someone's memory.

### Measured: the campaign is worth about 1%, and that is the point

Same lease, one variable (the packet's tiles), target cell:

```
                       out tok/s   TTFT median
 span1  (modelled)       19.60       415.4 s
 tuned  (measured)       19.80       415.5 s
                         +1.0%        +0.0%
```

span1 alone drifts 19.47 to 19.60 between leases (0.7%), so +1.0% is at the edge of what this
cell resolves. **The tile campaign is roughly neutral on throughput.**

That is worth stating as plainly as the discovery was. The analytical model was already choosing
near-optimal tiles for these thirty-four shapes; being UNMEASURED was a provenance defect, not a
performance one, and the `10.5-42.1%` dispatch occupancies the audit reports are a property of the
SHAPES (N=32, N=64 against a 256-wide tile) that no tile choice in the inventory can fix.

It is the fifth time in this document that a lever which looked large by inspection measured small
once ablated or A/B'd -- after the CK GEMM port, the MPF tile, sparse attention's "2.7x", and the
collective. The pattern is now the most reliable finding here: **on this stack, size the lever
before building it.**

What the campaign does buy is real but not throughput: the build is reproducible, `tile_source` is
`measured`, and a future object-recipe change that genuinely does move the right tile will now be
visible as a diff against measurements rather than silently absorbed by the model. Re-running it
belongs in the freeze procedure for that reason.

## 8. Unrelated issue observed

`cargo test -p devgen mla` fails `k3::tests::the_mla_prefill_arm_forces_one_split`
(`crates/devgen/src/k3.rs:6126`, `Option::unwrap` on `None`) on **clean `main`**.
It passes when run in isolation, so it is order- or environment-dependent. Not
touched by this work; recorded here because it will fail anyone else's gate run.

## 9. Consolidated review (2026-09-09)

Measured GLM-5.3 levers on TP8, with their original comparison conditions:

| Change | Recorded gain | Scope |
|---|---:|---|
| Matched low-rung decode objects | +24.9% output tok/s | concurrency 1; §7b |
| Decode ladder 4 → 8 | +21.0% output tok/s | 4096 input, concurrency 8; TPOT increases; §7c |
| `PLOW_FA_LDS_DMA=1` | +21.7% prefill tok/s | single 70k request; no measured gain at concurrency 20; §7f |
| `PLOW_PF_CHUNK=8192 PLOW_PF_INTERLEAVE=0` | +13.0% output tok/s | 70k input, concurrency 20; §7f |
| Sparse prefill, span 1 | +6.9% output tok/s | 70k input, concurrency 20; limited quality probes only; §7g |
| GEMM retune | approximately neutral | +1.0%, near measured run-to-run noise; §7p |

These deltas have different baselines and must not be added. Sparse spans 2 and 3
failed the original quality probes; §10 corrects the cause and records the fix.
They remain experimental pending broader qualification. Audit-off results also
remove a correctness check; matched decode tiers provide a gain independently.

The review fixes two runtime configuration defects: serving replay now uses the actual
parsed CLI, including overrides of environment settings; an explicit empty
`PLOW_HSACO_LOWRUNG` now disables discovery as documented. Neither changes kernel math
or establishes a new throughput result. No speculative decoding was added.

Review validation: 28 GLM emitter tests passed with one test thread;
`cargo test -p plowrt --features hsa --test config_env` passed, including a CLI
override after the `serve` subcommand and an explicit empty tier setting.
Formatting and `git diff --check` passed. No new GPU timing was taken in this review.

## 10. Ragged sparse prefill: correctness fix and union compaction (2026-09-09)

The all-layer failure was a runtime row-count defect. `rebase_chunk_rows` shrank
flash and the projections to the final chunk's live length, but left `LayerNorm`,
`IndexScorePf`, `IndexSelectPf`, and `IndexUnionPf` at the compiled bucket length.
Selection consequently computed the wrong causal query positions. Union and flash
also disagreed on the size of the aligned count header, so flash read the wrong
locations in the union table. This invalidates the earlier reuse-distance and
selection-quality explanations; those experiments ran with inconsistent layouts.

Add those four operations to `PREFILL_ROW_FIELDS`. A regression test fails before
the fix and checks the complete selection chain against flash for live row counts
1, 2049, 4097, and 8192. Eight ragged-prefill tests pass afterward. The serving
release binary builds with `--features hsa`.

Same-host TP8 quality checks, BF16 KV, sparse prefill on, sparse decode off,
8192-token chunks, no prefill interleaving, V2 flash, audit retained:

| Runtime / sparse layers | Actual prompt tokens | Fact retrieval |
|---|---:|---:|
| Original, span 1 (40/78) | 27,505–27,510 | 3/3 |
| Original, all 78 | 27,505–27,510 | 0/3 |
| Original, all 78, ragged chunks disabled | 41,269–41,274 | 3/3 |
| Fixed, all 78, ragged chunks enabled | 27,505–27,510; 41,269–41,274; 68,797–68,802 | 9/9 |
| Fixed + optimized union, all 78, depths 0.5 and 0.9 | 68,797–68,802 | 6/6 |

These use `scripts/glm53_needle_probe.py`, three facts at depth 0.1 unless stated, temperature 0,
24 output tokens. The script's requested lengths are estimates; the table reports
tokenized lengths. Retrieval checks are limited quality evidence, not a general
model-quality evaluation. The emitter default remains span 1. Explicit
`PLOW_GLM_DSA_PF_SPAN` and `PLOW_GLM_DSA_PF_DEXACT` settings now appear in the build
manifest's `unrecorded_env` section so different sparse packets are distinguishable;
these raw settings still need to be supplied explicitly when reproducing an emit.

With the fixed runtime and the same optimized union objects in both arms, a
same-lease `amd-bench --prefill-sweep 70000 --prefill-reps 3 --steps 1` comparison
measured span 1 at **18,899.957 ms** (0.89% spread) and all 78 layers at
**13,183.719 ms** (0.47% spread): 30.2% less prefill time, or 1.43x throughput.
Each arm discarded one warmup. This is a single-prompt prefill measurement, not
concurrency-20 serving. Both assets use max context 81920, decode batch 8, and
prefill buckets 128/512/2048/8192. The checkpoint is `GLM-5.3-plow-lite`.

Separately, `d_index_union_pf` replaces the workgroup-wide LDS prefix scan with
wave ballots and a scan of wave counts. It preserves ascending positions and all
64 membership bits. The independent HIP test compares counts, positions, and
masks exactly against a host union, including short/ragged packs and a representative
8192-row, 81920-context, top-k 2048 case.

An A/B/A/B on one leased MI300X measured the representative union at
1.7160/1.6959 ms before and 1.1434/1.1725 ms afterward: approximately 32% less time.
All three cases passed exact comparison on both runs. A separate full-model
70k prefill comparison, span 1 with the original runtime, measured medians
19,133.384 ms before and 19,044.211 ms afterward (three samples after warmup).
The 0.47% difference is within the observed 0.50–0.75% spread; it does **not**
establish a serving speedup. The production interpreter build passes its object
contract check with unchanged build defines and resource usage.

Reproduce the union test inside `nix develop` (use a free GPU lease):

```sh
"$PLOW_HIPCC" --offload-arch=gfx942 -O3 -w -std=c++17 \
  -Iruntime/amd -Iruntime/common -c runtime/tests/dsa_union_gfx942_test.hip \
  -o /tmp/dsa-union.o
c++ /tmp/dsa-union.o -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" \
  -lamdhip64 -o /tmp/dsa-union
perf-data/tools/gpulease -n 1 dsa-union /tmp/dsa-union
```

The target remains 100 requests at concurrency 20, 70k/700 random lengths with
ratio 0.14 and seed 0: 273.67 output tok/s, 1.431 s mean TTFT, and 65.92 ms mean
TPOT on the supplied H200 run. No speculative decoding is enabled by these changes.
The reference's aggregate serving measurements include speculation and cannot be
converted into a measured non-speculative baseline by ignoring its acceptance table.

## 11. AITER, hipBLASLt, ROCm and vLLM kernel review (2026-09-09)

The strongest isolated candidate is AITER's **gfx942 BF16 latent MLA assembly**
for 8 heads and 576 QK / 512 value dimensions. Earlier inspection of ordinary
QK256/V256 MHA kernels did not answer whether this absorbed MLA shape exists.
It does, and the installed object executes successfully with two KV splits.
The [reproducible comparison](../../runtime/bench/amd/mla_sparse_aiter/README.md)
includes plow's actual union/flash bodies, a vectorized native packing adapter,
independent FP32 oracle checks and recorded GPU graph timings. The opt-in
`PLOW_MLA_PF_AITER=1` runtime now dispatches the qualified object through HSA.

The isolated gain depends on selection overlap and index order. Packing costs
do not erase the distinct-selection advantage. Shared-indexer layers amortize
plow's union construction, so its separate cost cannot be charged on every layer.
An actual layer-77 capture and a native FP32 split reducer qualify the
merge/fold output contract; runtime and limited retrieval qualification are
recorded below. The installed AITER automatic
single-split path produced NaNs and a GPU memory fault; the benchmark fixes two
splits. This failure is not attributed to a proven root cause.

### What the source and kernel review establishes

| Area | Finding | Adaptation constraint |
|---|---|---|
| Sparse MLA | vLLM maps selected queries to a ragged decode-style batch; AITER supplies a matching per-query assembly kernel | Plow's split latent/rope layout and unnormalized partial/fold contract differ |
| MoE | AITER's block-scale CK path uses pre-shuffled weights and explicit pipeline scheduling; assembly is not always its selected implementation | Match TP-sharded width, expert routing, activation quantization, scale granularity and epilogue before comparing |
| GEMM | hipBLASLt supplies tuned kernels and grouped GEMM with device-side arguments; TensileLite can generate new schedules | A host library call cannot be placed inside plow's persistent device interpreter; use a dispatch boundary or adapt the device loop |
| ISA | Plow already issues MFMA through intrinsics and uses direct LDS loading; the tested AITER object uses 16x16 BF16 MFMA, vector loads and explicit waits | Assembly alone is not the measured independent variable; decomposition and memory access also change |

Sources inspected: [AITER MLA](https://github.com/ROCm/aiter/blob/10f8874dc2cd69c07ed84b5f125c27d12baccb10/aiter/mla.py),
[AITER MoE dispatch](https://github.com/ROCm/aiter/blob/10f8874dc2cd69c07ed84b5f125c27d12baccb10/aiter/fused_moe.py),
[CK block-scale implementation](https://github.com/ROCm/aiter/blob/10f8874dc2cd69c07ed84b5f125c27d12baccb10/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh),
[vLLM sparse MLA](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py),
[hipBLASLt grouped API](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/ext-reference.html),
[TensileLite tuning](https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html),
and [AMD workload guidance](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html).
AITER source review is pinned to `10f8874dc2cd69c07ed84b5f125c27d12baccb10`;
measurements use the installed `amd-aiter 0.1.19` object identified in the comparison.

### Actual MLA capture and matched grouped MoE

At GLM layer 77, the last 4464 queries of a 70,000-token deterministic prefill
have a mean 8-query union of **3931.8**. The native AITER adapter, including
packing Q and the entire KV cache plus FP32 split reduction, measured
**1.377 ms vs plow's 1.984 ms**: 30.6% lower kernel latency. Sampled relative
L2 vs an independent FP32 oracle was 0.000368 vs plow's 0.000387. The native
launcher pins the assembly object by hash and writes `(m,l)=(0,1)` with its
normalized partial. All nine native synthetic cells also passed. See the
[MLA capture and ABI record](../../runtime/bench/amd/mla_sparse_aiter/README.md#native-abi-and-actual-model-capture).
This supersedes the earlier T2048 capture, whose dense bucket retained indices
from the preceding chunk. The corrected capture explicitly selects T8192 at
chunk base 65536 and uses the current layer-74 indices.

The [matched grouped MoE benchmark](../../runtime/bench/amd/moe_aiter/README.md)
now includes activation quantization, sorting and combine. At 8192 tokens and
the actual TP8 expert width 256, plow measured **4.732 ms**, AITER assembly
**2.640 ms**, and AITER two-stage CK **2.439 ms**. Assembly wins at 128/2048
tokens; CK wins at 8192. Inputs have numerically identical OCP/FNUZ weights and
independently varying 128x128 block scales. Preshuffling is outside timing.
All nine cells passed their explicit screening thresholds, but AITER's sampled
relative L2 was **4.06–4.37%**, vs plow's **0.23–0.24%**. Its additional FP8
activation quantization needs actual-model quality checks before adoption.

Neither isolated result is a serving speedup. MoE changes activation precision
and remains a benchmark candidate. Native MLA integration is qualified below.

### Native MLA runtime integration

Re-emitting with `PLOW_MLA_PF_AITER=1 PLOW_MLA_PF_V2=1` removes cross-segment
counter edges at ordered launch boundaries. The loader checks geometry, pure
segment ownership, counter obligations, adapter ABI and the exact AITER object
hash. Eligible segments enqueue packing, assembly attention and FP32 reduction
through HSA. One 438,961,152-byte workspace per rank is reused without hot-path
allocation. Early causal rows, dense buckets and packed prefill retain their
interpreter route. Ragged rows and rebased KV slots are supported.

A paired full-model TP8 B8 prefill sweep, all 78 sparse layers, BF16 KV,
81920 context and audit retained, measured 70k mean time at **13,254.466 ms
interpreter vs 12,379.396 ms native**: **6.60% lower** over three repeats after
one warmup per arm. The same packet and runtime build were used for both timed
processes. Short 8192/8321-token fallback controls were effectively unchanged.
A separate instrumented run confirms all 78 native attention segments execute
in a ragged second chunk.

Each arm passed 18/18 concurrent retrieval cases across two lengths, three
depths and three facts; 13/18 paired continuations were text-identical. This
screen does not establish general model-quality equivalence. Native HSA tests
also pass rows 1/129 across two KV slots and verify the merge normalizer.
See [runtime results and reproduction](../../runtime/bench/amd/mla_sparse_aiter/README.md#full-model-runtime-qualification).
Concurrency-20 serving throughput has not been remeasured for this route.

### Corrected GEMM ceiling and FP8 assumptions

`bringup_ceiling.py --model glm53 --tp 8 --rows 8192` now follows the emitted
absorbed projections. The index-query projection remains 4096-wide independently
of TP. Expert intermediate width is **256 per rank**, not the unsharded 2048:
at uniform routing, gate/up is `(M,N,K)=(256,512,6144)` and down is
`(256,6144,256)`. These shapes were checked against packet disassembly.

The local Torch library calls measured q-a FP8/BF16 at 0.195/0.323 ms and o-proj
at 0.219/0.362 ms. The small per-expert calls were approximately 20–22 us in
either dtype. These are standalone API measurements with unit row/channel FP8
scales, not grouped, block-scaled checkpoint MoE; activation quantization is
excluded. They neither measure the complete MoE path nor exhaust library tuning.

Two comments incorrectly claimed CDNA3 FP8 has no compute advantage over BF16.
They are corrected: native FP8 has twice the theoretical throughput. At 32x32,
K=64 takes four FP8 K16 issues vs eight BF16 K8 issues. Plow's BF16-dequantizing
path preserves compact weights but forgoes that compute ceiling. Any native FP8
arm must retain OCP/FNUZ handling and measured staging/epilogue costs.
[AMD instruction/format reference](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html).

### Whole-model boundary

The fixed all-layer sparse configuration completed a 20-request screening with
the reference's variable-length distribution and concurrency 20: 20 successes,
0 failures, 583.69 s, 1,414,538 input and 13,795 output tokens, **23.63 output
tok/s**, mean TTFT **223.008 s**, mean TPOT **264.88 ms**. Active decode capacity
is 8. This is not the required 100-request comparison. AMD currently ignores
nonzero sampling temperature and uses device argmax, another fidelity difference
from the reference's unspecified server default.

A traced single 70k prefill took 13.366 s. Its segment timings were 10.705 s
interpreter and 2.584 s flash. The final 4464-row chunk showed substantial MoE,
index scoring/selection, attention and collective work. Per-packet body/stall
times overlap and include protocol work; they are not additive wall attribution.
The H200 serving target remains unmet, and the kernel ratios above are not a
claim of a new serving speedup. No speculative decoding was added.

## 12. Batched sparse decode and the CDNA3 index-score LDS fix (2026-09-09)

The decode indexer now supports the 1/2/4/8 ladder. Scratch is sized by decode
capacity; projections, normalization and scoring use the live rung width.
Indexer key writes use each slot's position, including the one-row rung of a
B8 blob. Selection serializes one row per packet over the shared radix scratch,
with `IndexSelect.i[3]` selecting the score/index/KV-length row. Gathered MLA
already has the corresponding batch axis. The loader requires the new object
marker and retains the CUDA refusal for batched DSA.

Full-model testing exposed a separate existing defect: the MI300X decode
index-score kernel carved **78,464 bytes** from a **64 KiB LDS** arena. It used
the CDNA4 256-position tile. CDNA3 now uses a 128-position tile occupying
**43,648 bytes**, with a compile-time assertion against the actual interpreter
arena. CDNA4 retains its original tile. The loader also rejects old CDNA3
score objects for single-row DSA.

The first concurrent retrieval screen passed only **2/18** cases. Rebuilding
the score kernel with the smaller tile, using the same packet and runtime settings,
raised that to **18/18**: concurrency 8, actual prompt lengths 5433–5438 and
68797–68802, three depths and three facts. This is limited retrieval evidence.
The independent GPU oracle checks all live scores for eight distinct rows,
exact top-k sets with ties and poisoned tails, and repeated shared-scratch use.
Emitter tests cover row strides, capacities and selection dependencies;
loader/manifest tests cover object compatibility. The final emitter reproduces
the tested B8 packet byte-for-byte.

A 20-request serving screen with the reference's variable 70k/700 lengths,
ratio 0.14 and concurrency 20 completed **20/20, zero failures** in **489.35 s**.
Output throughput increased **23.63 → 28.19 tok/s (+19.3%)**, mean TPOT fell
**264.88 → 218.68 ms**, and mean TTFT fell **223.01 → 195.14 s**. Both screens
have identical per-request input/output counts, totaling 1,414,538 input and
13,795 output tokens. This before/after comparison combines native AITER
prefill and sparse decode; it does not isolate either change. The candidate
retains TP audit and B8 capacity. It is not the 100-request H200 comparison.

See the [DSA decode qualification and reproduction](../../runtime/bench/amd/dsa_decode/README.md).
Pooled batched DSA and MXFP4 indexers remain unsupported. Batched FP8 KV
was subsequently qualified as an opt-in path in section 14.
Sparse decode requires disabling the incompatible q-RoPE fusion. No speculative
decoding is added; H200 serving parity remains unproven.

## 13. FP8 KV capacity investigation (2026-09-09)

The existing gathered decode template passes FP32 oracles through B20 with
per-row OCP FP8 latent scales and BF16 rope. Twelve cache-writer cases cover
slot positions, ring wrapping, zeros and untouched tails; 24 attention cases
cover short/empty/70k contexts and split merging. The maximum FP8 decode
kernel relative L2 error is 8.10e-7 against the quantized-cache oracle.

The unexposed gathered FP8 V2 prefill template had a scale-address defect:
KV loads followed the union's selected cache positions, while scales used
union entry numbers. Correct both softmax variants and PV scale addressing.
Matched synthetic T129 relative L2 falls from 1.543 to 0.001557. Tests also
cover T1/4464 and inactive union/cache tails. At this stage, production sparse FP8 and batched
FP8 emission remained refused. Section 14 records the subsequent integration.

FP8 is a capacity opportunity with a prefill cost. B20/16-split standalone
decode measures 0.1343 ms vs BF16 0.1459 ms, but the actual layer-77 sparse
capture measures **2.414 ms FP8 vs 1.901 ms BF16**. Its five sampled queries
show 0.0718% kernel error against quantized-cache FP32 and 1.321% difference
against original BF16-cache FP32. This is not model-level qualification.

At ctx81920, B20's calculated cache storage falls from 145.31 to 84.85 GiB/rank,
excluding weights and scratch. Preserve native AITER prefill through an FP8
packing/dequantization adapter before pursuing B16/20 serving; the interpreter
FP8 prefill regression makes a flag-only enablement unattractive. No new
serving gain is claimed. See [tests, results and reproduction](../../runtime/bench/amd/mla_fp8_kv/README.md).


## 14. Sparse FP8 KV runtime and batch-16 serving (2026-09-09)

Sparse FP8 now uses the existing 109/110 opcodes with a demoted selected-index
or union handle. The AMD loader requires new object markers and validates
capacities, gfx942 QH8 geometry and pure V2 prefill routing. Decode writers
use per-slot positions, including rung 1. Native AITER packing dequantizes the
live FP8 latent cache once into its existing BF16 workspace; the pinned
assembly and FP32 reduction remain unchanged. This is opt-in through
`--glm-fp8-kv=true`, with the incompatible q-RoPE fusion disabled.

TP8 batch 16 passes **18/18** retrieval cases at concurrency 16 through actual
68.8k-token prompts. The HSA adapter test checks varying row scales, two cache
slots, exact packed values and attention output for rows 1/129. Emitter,
manifest, loader and packet-ABI tests pass, as does the CUDA/HSA compile check.
This is limited retrieval qualification, not a broad model accuracy claim.

The 20-request C20 serving screen has identical per-request input/output
lengths to section 12 and completes 20/20 with zero failures. Output throughput
is **28.19 → 29.51 tok/s (+4.70%)** and mean TTFT **195.14 → 158.19 s**.
However, mean TPOT regresses **218.68 → 327.23 ms (+49.64%)**. KV format and
batch capacity change together; this single comparison cannot isolate either.
Keep this as a capacity option, not a default latency improvement. B20 is still
only a cache-allocation estimate and template test, not a serving qualification.

See the [runtime record and reproduction](../../runtime/bench/amd/mla_fp8_kv/README.md#runtime-qualification).
The 100-request H200 target remains unmet; no speculative decoding was added.
Prefill index scoring/selection and native FP8 MoE remain the next measured
kernel opportunities. vLLM also supports data-parallel attention with expert
parallelism, which warrants a separate architecture study for MLA/MoE; the
reference server's topology is unknown.
[vLLM deployment documentation](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/).


## 15. Captured GLM MoE and bounded packing cost (2026-09-09)

The AITER MoE comparison now uses layer-40 activations, actual expert routing
and the checkpoint's TP8 weight slices at two positions in a 70k prefill.
The standalone plow GLU matches every captured value bit-for-bit: 16.78 million
values in the full chunk and 9.14 million in the tail. The harness must match
the runtime's negative-zero upload scrub; 4523 checkpoint bytes require it.
OCP-to-FNUZ conversion reuses canonicalized bytes and doubles block scales,
with exact numerical equality checked across all weights.

At 8192 live rows, plow takes **4.833 ms**, resident AITER assembly **1.971 ms**,
and CK **2.371 ms**. A native packing helper accepts the existing weight/scale
pointer tables and uses a reusable **1.125 GiB** output workspace, avoiding
about **84.4 GiB/rank** of persistent weight duplication across 75 routed layers.
Hoisting expert pointers out of the packing loop cuts packing time from
**1.285 to 0.674 ms**. Every packed byte matches AITER, including reversed
expert tables and restored reuse.

Including packing on every dispatch, assembly takes **2.647 ms at 8192 rows**
and **1.918 ms at 4464 rows**, versus plow's **4.833 / 2.825 ms**. The full-chunk
kernel boundary improves 45.2%. This includes activation quantization, sorting,
expert computation and reduction, but excludes shared expert, residual, TP
communication and a production HSA launch boundary.

Sampled FP32-oracle error is **0.237% plow vs 3.55–3.85% AITER**. All-output
relative L2 against plow is 3.54–3.60%; resident and repacked assembly outputs
are not bit-identical. Native integration must address the existing deterministic
FP64 combine contract as well as full-model quality. No AITER MoE production
route or new serving gain is claimed here; H200 parity remains unmet.

See [capture reproduction and measured records](../../runtime/bench/amd/moe_aiter/README.md#actual-glm-capture-and-reusable-weight-packing).

## 16. Native A8 MoE prefill (2026-09-09)

An opt-in `MoeAiterFp8Pf` instruction now launches the qualified gfx942 AITER
assembly through HSA. Its contract explicitly includes per-128-element A8
quantization and BF16 routed accumulation. It stores FP32 for the existing
shared-expert/TP combine. The default FP64 path and decode remain unchanged.
Enable with `--glm-moe-aiter=true --emit-packed-prefill=false` on the qualified
TP8 GLM configuration.

Four ordered launches pack weights, prepare activations/routing, run assembly
and store output. The workspace is **1.268 GiB/rank**, reused across all 75
routed layers. The loader validates the pinned object hash, helper ABI,
alignment geometry, tensor capacities and a pure segment without interpreter
counter obligations. The serving process needs neither HIP nor Python.

On captured layer-40 inputs, the direct HIP harness measures **4.833 → 2.975 ms**
at 8192 rows and **2.825 → 2.186 ms** at 4464 rows, including weight packing and
the output conversion. Input bytes, scales and routing match AITER exactly.
Sampled FP32-oracle error remains 3.55–3.85%. A per-slot FP64 reduction prototype
passed its repeat checks but took 4.194 ms at 8192 rows; the faster fused path
therefore has its own opt-in numerical contract.

TP8 B8/BF16-KV retrieval passes **18/18** through 68.8k prompt tokens, with
**17/18** continuations text-identical to the B8 baseline. Packet/operand ABI,
native route, HSA ragged-row and CUDA+HSA checks pass. Lean verifies every
emitted program. The option-off compiler reproduces the old B8 packet exactly.
These checks qualify the limited retrieval workload, not broad model accuracy.

An adjacent 20-request C20 screen with the same final runtime and GPU objects
confirms **28.216 → 29.960 output tokens/s (+6.2%)**. Mean TTFT falls
**196.985 → 178.419 s (−9.4%)**, mean TPOT **216.916 → 205.565 ms (−5.2%)**,
and P99 ITL **1686.281 → 1471.807 ms (−12.7%)**. Median ITL regresses
**116.833 → 119.005 ms (+1.9%)**. Both runs complete 20/20 with zero failures
and identical per-request token lengths, totaling 1,414,538 input and 13,795
output tokens. The earlier B8 result, 28.191 tokens/s, agrees with this new
baseline within 0.1%.

This is a measured serving improvement, smaller than the isolated MoE gain.
Keep the changed numerical path opt-in. No speculative decoding was added;
the H200 100-request target of 273.67 tokens/s remains unmet.

See [native adapter reproduction](../../runtime/bench/amd/moe_aiter/README.md#native-gfx942-serving-adapter)
and the [measured record](../../runtime/bench/amd/moe_aiter/mi300x-native-results.json).

## 17. Query-partitioned prefill indexer prototype (2026-09-09)

The packet repeats index score and exact radix top-k work on every TP rank.
A captured-input TP8 prototype partitions query rows while retaining the
global causal base and all 32 index heads. It gathers selected int32 indices:
64 MiB total at 8192 rows, rather than the multi-GiB score matrix. Equal,
padded row bands handle ragged chunks without splitting an index row.

The full layer-38 chunk measures **19.000 → 4.051 ms** at 8192 query rows and
65536 context tokens. Its actual 4464-row/70000-token tail measures
**12.094 → 2.359 ms**. Both include all eight GPUs, host submission/completion
and the direct GPU gather. Tiny early chunks regress, so the eventual route
must retain replicated execution for those cases.

Standalone all-row and partitioned scores match bit-for-bit; top-k sets match
the model captures; gathered buffers match across all ranks. Checks also
cover empty ranks, 129-row chunks, causal masking and unowned/padded writes.
Assembly inspection caught LLVM splitting some contracted head-reduction FMAs
after inlining. Explicit GLM FMAs preserve the interpreter's fused rounding,
and the benchmark uses the exact emitted scale bits `0x3c7fffff`.

These prototype measurements preceded native integration. Normal segment-major
dispatch queues later work without a host barrier per segment, so the serving
route needs device rendezvous and audited scratch reuse. Section 18 records
that integration and its separate quality and serving checks.

The retained reproduction sources are
[`captured.py`](../../runtime/bench/amd/dsa_pf_tp/captured.py) and
[`kernels.hip`](../../runtime/bench/amd/dsa_pf_tp/kernels.hip).

## 18. Native TP8 prefill indexer (2026-09-10)

Opt-in `--glm-index-tp=true` replaces replicated score/top-k work with four
ordered HSA launches for prefill buckets of 2048–8192 rows. It keeps global
causal positions, partitions query rows across eight ranks and gathers live
int32 indices. Smaller buckets retain the existing route. The qualified
configuration uses TP8, B8, BF16 KV, native AITER MLA/MoE and unpacked prefill.

Three system-scope gates protect entry to reused TP scratch, readiness of
selected bands and completion of all peer reads. A separate completion kernel
ensures every gather workgroup has finished. The route allocates no additional
activation workspace, validates gate collisions and segment isolation, and
checks timeout status after each chunk even when general auditing is disabled.

The native device protocol measures **4.021 ms** at 8192 rows/65536 context
and **2.433 ms** for the actual 4464-row/70000-context tail. Captured top-k sets
and partitioned score bits match the all-row reference. Sixteen queued
scratch-poison reuse iterations per case preserve every gathered output;
all used gates receive eight arrivals and all timeout statuses stay zero.

Retrieval passes **18/18**, with **15/18** continuations text-identical to the
MoE-only baseline. Atomic top-k ordering and downstream floating-point
accumulation do not guarantee identical continuations. Keep this limited
quality qualification and the route's opt-in status explicit. Packet tests
(120), manifest tests (47), route/gate checks and CUDA+HSA compilation pass.
Lean verifies all eight programs; option-off emission reproduces the prior
MoE packet byte-for-byte.

An adjacent C20 screen confirms **30.010 → 32.478 output tokens/s (+8.2%)**.
Mean TTFT falls **175.661 → 160.254 s (−8.8%)**, mean TPOT
**203.923 → 188.560 ms (−7.5%)**, and P99 ITL
**1475.313 → 1163.319 ms (−21.1%)**. Median ITL is effectively unchanged
at 117.248 → 117.134 ms. Both arms use the same frozen runtime and GPU object
directory; only the indexer option differs. Both complete 20/20 without failure,
with identical per-request lengths totaling 1,414,538 input and 13,795 output
tokens. Native ran first; this is one screen per arm, not the H200 100-request
benchmark. No speculative decoding was added; H200 parity remains unmet.

The benchmark sources remain under `runtime/bench/amd/dsa_pf_tp`; raw protocol
and serving records were removed from the repository.

## 19. hipBLASLt projection assembly qualification (2026-09-10)

With native indexer and MoE enabled, a 70k-token diagnostic profile attributes
**4.689 / 9.061 s** to ordinary interpreter segments, **2.076 s** to MoE,
**1.585 s** to sparse MLA and **0.488 s** to the indexer. Interpreter segments
combine dense projections, normalization, shared experts and TP collectives.
The profile drains every segment; it does not measure normal serving latency.

An isolated comparison covers ten emitted BF16 projection shapes at 8192 and
4464 rows. Q-A, KV-latent and absorbed-Q use actual layer-38 inputs/weights;
other shapes use seeded random inputs. Plow's Q-A and KV outputs reproduce
the model captures bit-for-bit. All 20 cases pass sampled FP32-oracle checks.
The hipBLASLt preference measures **1.3–2.0× faster at 8192 rows**. Q-A takes
**495.00 → 318.76 µs**, absorbed-Q **363.89 → 227.90 µs**, KV-latent
**152.64 → 104.63 µs**. These exclude interpreter and TP overhead.

Library tracing identifies the selected gfx942 Tensile kernels. Four unique
assembly kernels cover the large-row choices for those three captured
projections. Direct HIP module launches pass 24 cases per workgroup mapping,
including 1/129-row inputs and both large-row choices. The selected XCC mapping
measures Q-A at **315.29 µs**. Sixteen of 24 outputs are bit-identical to the
library; the largest all-output relative L2 difference is below 0.000047.

Direct plow C HSA launches pass **12/12** cases, with complete outputs exact
against HIP exports over three repeats and 512-byte guards intact. This exposed
a 128-byte symbol-name limit in that C loader; it now allocates the full name
at lookup. The kernels' notes declare a 160-byte inline ABI and no private
scratch; their descriptors leave argument size zero, requiring an explicit
size in the native host binding.

These qualify a native assembly candidate, not a serving route. The next gates
are opt-in isolated HSA dispatch, full-model quality and paired serving tests.
The qualified serving rate remains **32.478 output tokens/s**; H200 parity is
unmet. See [reproduction, source references and raw records](../../runtime/bench/amd/glm_projection/README.md).


## 20. Native hipBLASLt projection route (2026-09-10)

`--glm-gemm-lt=true` adds direct HSA dispatch for three qualified BF16
projection shapes in unpacked TP8 gfx942 prefill buckets of 2048–8192 rows.
Q-A, KV-latent, absorbed-Q and selected indexer-Q sites account for 255 native
projections per large program. Four pinned assembly kernels cover full and
ragged chunks; the actual row count selects the kernel. Smaller buckets and
unmatched shapes retain the previous implementation.

Each projection uses one ordered launch, stack arguments and no additional
activation workspace. Loading validates the full object hash before filling
four pinned descriptors' missing argument-size fields. Resource metadata,
operand capacities and segment isolation are checked before dispatch. HIP and
hipBLASLt library calls are absent from this serving route.

Retrieval passes **18/18**, with **12/18** continuations text-identical to the
preceding native-indexer packet. This limited quality check does not establish
general equivalence; the option stays opt-in. Packet tests (120), manifest
tests (47), native route/ABI tests (2), CUDA+HSA compilation and release build
pass. Lean verifies all eight programs, and option-off emission reproduces
the baseline packet byte-for-byte. The config suite passes 23/24; the existing
`no_raw_env_reads` failure names DSA span/exact knobs present before this change.

An adjacent C20 screen measures **32.726 → 33.400 output tokens/s (+2.1%)**.
Mean TTFT falls **160.345 → 155.044 s (−3.3%)**, mean TPOT
**188.317 → 186.924 ms (−0.7%)**, and P99 ITL
**1161.941 → 1132.449 ms (−2.5%)**. Median TPOT regresses
**195.904 → 198.048 ms (+1.1%)**. Both arms complete 20/20 without failure,
with identical per-request lengths totaling 1,414,538 input and 13,795 output
tokens. Both use the same frozen runtime/objects with native MLA/MoE/indexer;
only the projection option differs. Native runs first.

This is a modest gain in one screen per arm, not a repeated estimate or the
H200 100-request benchmark. H200 parity remains unmet. After timing, loader
ownership validation was reduced to one stream scan per program; dispatch
and arithmetic are unchanged. The final loader also passes 18/18 retrieval
cases; validation and the two runtime hashes are recorded separately. See [reproduction and all metrics](../../runtime/bench/amd/glm_projection/README.md#paired-serving-screen)
and [serving evidence](../../runtime/bench/amd/glm_projection/mi300x-serving.json).

## 21. Continuous batching by default: step budget, packed siblings beside native MoE, one planner (2026-09-11)

Four things kept cross-request prefill packing off on the production GLM-5.3 TP8 recipe, and
the fourth — not packing — is what costs throughput on the 70k/C20 workload:

| gate | before | now |
|---|---|---|
| emit | `PLOW_EMIT_PACKED_PREFILL` asserted off beside `PLOW_GLM_MOE_AITER/RESIDENT/INDEX_TP/GEMM_LT` | siblings ride beside the native segments (class A, row-agnostic over `T`); index-TP keys on the builder topology; sparse (DSA) buckets get no sibling; GLM gfx942 emits siblings by default with the ordinary programs byte-identical (`glm_tests`) |
| chunk policy | `admit` packs whole spans, and the widest capable rung ≤ the tick cap is also the chunk size, so one chunk fills it | one backend-neutral step planner (`crates/plowrt/src/sched/step.rs`): decodes, then packs, then oldest-first whole chunks under one budget |
| route | `PLOW_PACKED_PREFILL_ROUTE`/`PLOW_PF_BATCH` off; a missing family object was a silent `Ok(None)`; no FP8-KV packed objects existed | the route follows the packet; a missing `interp_packed_mla_{norm,flash}[_fp8kv]` is a load error by name; `_fp8kv` rows added to `build_gfx942.sh`; the native AITER MoE / hipBLASLt / MLA-fold routes accept packed siblings (they refused `packed_prefill_only` on their own) |
| **tick cap** | `PLOW_PF_INTERLEAVE=2048` as soon as any slot decodes → every steady-state chunk a 2048-row **dense** launch (2048 is not a DSA bucket) | AMD default = the widest compiled rung: one 8192-row **sparse** chunk per tick |

**Why packing is not the lever on this packet.** The 8192 rung is a DSA bucket
(`IndexTpPf`/`IndexUnionPf`/`FlashGatherPrefill` derive every row's position from one scalar —
class C) and has no packed sibling; the widest packable rung is the dense 2048. Under the default
ragged plan every non-final chunk of a 70k prompt is 8192 rows and the final chunk must run
isolated (one sampled token per launch), so at default planning nothing is packable — for 70k and
for ≤ 8192-token prompts alike. Packing fires only under a `PLOW_PF_CHUNK ≤ 1024` override, and
there it can only fold N dense 1024-row launches into one dense 2048, never a sparse 8192. It is
armed by default because it is free when it applies (dense packets, mixed short/long traffic).

**The planner** is vLLM's `schedule()` shape with no device type in it:
`plan(backend, tick, decodes, candidates) → Plan { decodes, launches }`. A backend declares
`Backend { step_budget, packing, split_spans, decode_rows_join_prefill }`
(`ServeEngine::step_backend`); the AMD/CPU tick lowers each launch to `advance_packed_prefill` /
`prefill_chunked_at_most` and the CUDA pass lowers its single fair-split launch. A chunk wider than
the remainder waits rather than being re-planned narrower; a plan element a backend cannot run is
refused by name (`serve/step_lowering_tests.rs`, the CPU declaration as the oracle).

**Decode rung fill** (verified, unchanged): rung = highest live-or-mid-prefill slot + 1, admission
takes the lowest free slot, no compaction on release (a KV block copy costs more than the rows it
saves); at C20 with a full slot table the rung tracks 20.

**Measured** — 20 × 70k/700/.14 at C20, the frozen packet, the same runtime and objects, identical
request set (1,414,538 in / 13,795 out), one exclusive-lease run per arm:

| arm | knobs | out tok/s | duration | TTFT med / mean / p99 | TPOT med / mean | ITL med / p99 |
|---|---|---:|---:|---|---|---|
| control (old defaults) | `pf_interleave=2048 pf_batch=0 route=0` | **16.81** | 820.8 s | 352.9 / 361.3 / 737.6 s | 646.8 / 614.1 ms | 109 / 1832 ms |
| new defaults | `pf_interleave=widest pf_batch=1 route=auto` | **49.68 (+196 %)** | 277.7 s | **100.3 / 102.6 / 198.7 s** | **234.9 / 232.7 ms** | 101 / 1348 ms |
| new defaults, re-emitted packet WITH siblings | route armed on 128/512/2048 | 49.83 | 276.8 s | 101.2 / 102.7 / 197.3 s | 233.6 / 233.6 ms | 103 / 1357 ms |
| **new defaults, the user's 100-prompt command** (frozen packet; 7,018,227 in / 71,149 out) | unset | **47.09** | 1511.0 s | 15.97 / 35.6 / 191.2 s | 395.8 / 366.6 ms | 108 / 3333 ms |
| reference recipe before this merge, same 100-prompt command (`PLOW_PF_INTERLEAVE=0`, i.e. the 2048 cap switched off by hand) | env | 47.80 | 1489 s | 16.0 / 34.6 s | 385 / 362 ms | 102 / 3361 ms |

Read together: the +196 % is against the shipped *default* (2048 cap once anything decodes), which
nobody used for the production numbers — the serving recipe already switched the cap off. On the
user's 100-prompt workload the new defaults reproduce that recipe with no env (47.09 vs 47.80,
within run noise) and do not add to it: ≈ 857 full 8192 chunks + 100 ragged tails ≈ 957 prefill
launches at ~1.3 s ≈ 1240 s, plus ≈ 2600 decode-only ticks at 108 ms (the median ITL) ≈ 280 s,
sums to the measured 1511 s. The remaining 3.2× to 150 tok/s is per-tick kernel time on the 8192
sparse rung (and the rung-20 decode step), not scheduling.

The gain is the tick cap: the steady-state prefill launch went from a 2048-row dense chunk to an
8192-row sparse chunk — 4× fewer launches and ~30× fewer attention FLOPs per row at 60–70k keys —
and the decode step between prefill launches got closer, not further apart, because one sparse 8192
launch is shorter than four dense 2048 ones. Co-packing fired in neither arm, as argued above. The
re-emitted packet with siblings loads with every rung armed (its first load refused at the native
AITER MoE route, whose own `packed_prefill_only` refusal is lifted), passes the 18-case retrieval
screen 18/18 at C20 and matches the frozen packet's throughput; the C1-vs-C8 identity screen diverges
on greedy near-ties exactly as the token-batch baseline does (decode rung 1 vs 8). Its sibling programs
have not EXECUTED here — no pack forms at default planning — so `--emit-packed-prefill=false` remains
the rollback for that half.

Rollbacks: `PLOW_PF_INTERLEAVE=2048`, `PLOW_PF_BATCH=0`, `PLOW_PACKED_PREFILL_ROUTE=0`,
`--emit-packed-prefill=false`. Flag rows in `docs/flags-reference.md`.

## 22. The sparse 8192 rung goes span-aware: packed siblings and token-batch bodies for DSA prefill (2026-09-11)

§21 left the one rung that matters at 64K context — the sparse (DSA) 8192 bucket — without a
packed sibling or a body, because its selection chain was class C: `IndexTpPf` scored and selected
every row against `kv_len[0] - T`, `IndexUnionPf` unioned 8-query tiles under the same scalar, and
the AITER sparse flash converted ONE slot's cache over `[0, kv_len)`. Each class-C site now reads its
request span instead; the ordinary programs are byte-identical (`sparse_rung_joins_the_packed_passes_only_under_packed_sparse_pf`).

| site | was (one scalar per launch) | now (per span) |
|---|---|---|
| `IndexTpPf` score/select (`runtime/amd/dsa_tp_adapter.hip`, ABI 2 `plow_dsa_tp_abi_2`) | `q_pos0 = kv_len[0] - rows`, key base = the bound cache tensor | the kernargs carry a `PlowKvSpan` table (`dev_isa.h`: `row0, n_rows, kv_row0, kv_len, kv_base`); per span the unchanged `d_index_score_pf_row` / `d_index_select_pf` bodies run on the span's row window with `len = &span.kv_len`, `n_tok = row0 + n_rows` (so `q_pos0 = kv_row0 - row0`, wrapping) and `k += kv_base * 128`. `spans == NULL` is the ABI-1 isolated form. Rows no span covers (the band, parked padding) are skipped. `kv_base` is an explicit host-written field, never slot arithmetic on the device (DCP places it). |
| `IndexUnionPf` | per-8-query union, `q_pos0 = kv_len[0] - n_tok` | not emitted in packed programs: a tile could straddle two requests, and the AITER route gathers the per-row selection anyway. The sparse flash names `iidx_pf` directly (`fj[1] = iidx_pf + 1`); `amd_sparse_mla::routes` accepts an `IndexTpPf` producer in packed programs only. |
| sparse `FlashMlaPrefillFp8` (AITER route, `exec/amd_sparse_mla.rs`) | one pack → attention → reduce over `rows`, cache `[0, kv_len)` of the bound slot | `enqueue_spans`: one chain per span with every per-row operand offset by `row0` and the cache/scale operands by `slot * ctx`, converted over `[0, span.kv_len)`. A span must start ≥ 2047 keys deep (fixed-width 2048-key CSR): `packed_span_admissible` gates the scheduler (packs and body members), staging refuses by name. |
| DSA decode select over the band (`IndexSelect`, `check_dsa_select_local`) | — | unchanged: the body's band chain is the decode form (row t = slot t, class B). |
| KV / index-key writers (`op_norm.h`, `i7 = ctx`) | — | already span/band-aware (§ token batch); the packed `_tb` norm object writes `kidx` at `slot * ctx + position`. |

Emit: `PLOW_PACKED_SPARSE_PF=1` (`--packed-sparse-pf`) lets sparse buckets join both packed passes
when `PLOW_GLM_INDEX_TP=1`, the indexer is unpooled and KV is FP8; the `PLOW_GLM_INDEX_TP` assert
takes the builder's topology (packed allowed), the interpreter selectors assert `!packed`.
Runtime: `packed_mla_compatible` / `token_batch_body_compatible` admit `IndexTpPf` + the sparse
FP8 flash when the ABI-2 adapter is loaded and every such instruction got its native route
(`packed_sparse_refusal` names the missing piece otherwise; the interpreter selectors stay refused).
Native routes now dispatch under an ACTIVE packed binding too — `enqueue_segment` used to skip
every native route while a binding was staged and hand the segment to the primary interpreter,
which has no `MoeAiterFp8Pf`/`GemmLtPf`/`IndexTpPf` arm; that gate is what every dense body launch
ran through before this change.

Measured (this section is completed by the qualification below): identity C2/C3/C8, 18-case
retrieval, the 20-prompt 70k/700/.14/C20 A/B against §21's 49.68 tok/s, packs and body fires.
