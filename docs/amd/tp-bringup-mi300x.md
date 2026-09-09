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

**The fp8 route is the one to take first**, despite being the largest:

* it fits with room — 126.9 GB/rank leaves ~79 GB for KV, activations and the non-expert weights,
  where §3's TP4 failure was at 94.7% occupancy;
* it reuses the arm GLM-5.3 is **already qualified on** in this document (ops 45/46/48/49,
  block-fp8 e4m3 with a [128,128] `weight_scale_inv` grid) rather than a second arm;
* it is numerically a WIDENING. e4m3 carries more precision than int4, so dequantizing the
  shipped values into fp8 reproduces them closely. Requantizing to mxfp4 instead stacks a
  *second* lossy quantization — into a different, non-uniform grid — on top of a checkpoint
  Moonshot trained with QAT **for int4**. That risk is real and unquantified, and it is the wrong
  thing to accept on a first bringup when a safer arm fits.

The native int4-g32 arm (route 1) stays the right end state: it is exact, and it is the smallest
of the three. It is new kernel work, and it should be measured against the fp8 route rather than
assumed faster — §4 already shows fp8 LOSING to bf16 on DSV4's `expert down` at M=96, so narrow
weights do not automatically win at decode-shaped M.

### Measured: what the fp8 route costs numerically

`scripts/kimi_k27_prep.py --verify` dequantizes a real expert exactly (unpack order mirrored from
`compressed_tensors.unpack_from_int32`, not guessed: nibble `i` of packed word `c` is column
`8c+i`, stored unsigned with an offset of 8), re-encodes it to block-fp8, and round-trips it
through the same arithmetic the kernel does — `fp8 * scale_inv`. Layer 3, `gate_proj`:

| scale grid | mean rel. err | max rel. err | in-block scale spread |
|---|---:|---:|---:|
| `[128,128]` (the shipped arm) | **2.264%** | 3.5% | 6.4x |
| `[32,32]` (ablation) | **2.169%** | 3.0% | 6.4x |

**The grid is not the problem.** Refining it 4x along K — to the source's own group size — buys
0.1 percentage points. The 2.2% is e4m3's own precision, and it has a specific cause: the int4
values are integers in ±1..8, which e4m3 represents EXACTLY, but dividing them by a block scale
incommensurate with their per-32 group scale turns them into arbitrary reals that need mantissa
bits e4m3 does not have. (This is the same property mxfp4 sidesteps with power-of-two E8M0
scales — and the reason a finer f32 grid cannot recover it.)

So the fp8 route costs ~2.2% RMS on the routed-expert weights, on top of a checkpoint already
quantized to int4. Whether that is acceptable is a judgement, and the honest framing is the ratio:
int4's own step is 1/8 = 12.5% of a group's maximum, which this model was QAT-trained to tolerate,
and 2.2% RMS is a fraction of that. It is also the same order as the fp8 error GLM-5.3 already
serves under in §3. It is NOT free, and it is not something to discover after converting 555 GB —
which is why the verifier exists and why it runs on four experts in seconds.

The native int4-g32 arm remains the only route with **zero** added error, and it is also the
smallest (71.3 GB/rank). The fp8 route's case is that it reuses a qualified arm and can be
measured this week; the int4 arm's case is that it is exact. Both should be built; fp8 first.

### Gates after the encoding, unchanged in substance

Full-model emit (`kimi_emit_block` is `--block`-only; the `glm_main` analogue is unwritten), the
derived MLA tensors a prep must write (`q_absorb`, `kv_a_latent`, `k_rope`, `q_rope`, `v_absorb` —
`kimi_k3_prep.py` is the working template, and it already handles this checkpoint's
`language_model.model.` wrapper prefix), the tokenizer (`tiktoken.model`, no `tokenizer.json`),
and the dense-arm prefill fault localised in §5.

## 8. Unrelated issue observed

`cargo test -p devgen mla` fails `k3::tests::the_mla_prefill_arm_forces_one_split`
(`crates/devgen/src/k3.rs:6126`, `Option::unwrap` on `None`) on **clean `main`**.
It passes when run in isolation, so it is order- or environment-dependent. Not
touched by this work; recorded here because it will fail anyone else's gate run.
