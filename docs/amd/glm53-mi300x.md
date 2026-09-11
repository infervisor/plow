# GLM-5.3-FP8 on MI300X — bringup and TP4 baseline

Measured 2026-09-07 on an 8x MI300X (gfx942, CDNA3, 304 CU, 192 GiB/card) host, ROCm 7.14.0
from the flake. **Read the caveats in §5 before quoting any number here.**

## 1. GLM-5.3 needs no new architecture support

`zai-org/GLM-5.3` is `model_type: glm_moe_dsa` — the same architecture as GLM-5.2-FP8, which
this tree already serves through `glm_main`. The evidence:

- `config.json` differs from GLM-5.2-FP8 in `transformers_version` alone (5.15.0 vs 5.12.0).
- The two checkpoints' tensor name sets are **equal**: 118,629 names each, `set(a) == set(b)`.
- Same 141 shards, same `total_size` (755,617,140,416 B). Only the VALUES differ, which matches
  the model card: same base model, gains from post-training.

So GLM-5.3 required **zero devgen changes**. Do not confuse it with **GLM-5.3-Flash**, which is
`model_type: glm5_next` (KDA + hyper-connections) and is a genuinely different emit path
(`mla::glm53_emit_full`).

## 2. Precision: the checkpoint is block-FP8, and the served blob keeps it

The weights are native **e4m3 with a `[128,128]` `weight_scale_inv` grid**
(`quantization_config.weight_block_size = [128,128]`, `activation_scheme: dynamic`).
`"dtype": "bfloat16"` in config.json is the COMPUTE dtype, not storage.

| stored as | tensors |
|---|---|
| F8_E4M3 + F32 scale grid | q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj, dense MLP (layers 0-2), all 256 routed experts x 78 layers, shared expert, DSA indexer wq_b/wk |
| BF16 | embed_tokens, lm_head, every layernorm, router `mlp.gate.weight`, indexer k_norm(+bias), indexer weights_proj |
| F32 | the scale grids, `mlp.gate.e_score_correction_bias` |

The emitted blob's manifest confirms FP8 survives compilation:

```json
"precision": {"weight_enc":"fp8","act_enc":"bf16","kv_enc":"bf16","expert_enc":"fp8blk"}
```

`act_enc: bf16` is not an omission — this is W8A16. MLA+MoE has no W8A8 profile in this tree
(`docs/flags-reference.md`), so activations stay bf16 by construction.

The host prep (`scripts/glm52_prep_lite.py`, reused unmodified because the names are identical)
dequantizes only what the MLA absorption needs — `derived.q_absorb`/`v_absorb`/`kv_a_latent`,
plus `q_a_proj`, `o_proj` and the shared expert — and symlinks the 141 raw shards so the routed
experts and dense MLP resolve **verbatim as block-fp8**. 40 GB of new data instead of a 700 GB
copy. `GLM_LINEAR_FP8=1` would additionally keep `o_proj` + shared expert in fp8; it needs
`glm52_prep_fp8_linear.py`'s `.weight_fp8` shards and is NOT in the numbers below.

## 3. Capacity on a 192 GiB card

Prepped checkpoint ~714 GiB.

| | GiB/rank, weights | verdict |
|---|---:|---|
| TP2 | 357.1 | impossible — 165 GiB over the card, weights alone |
| TP4 | 178.6 | fits; measured 181.75 GiB/rank live |
| TP8 | 89.3 | comfortable |

## 4. Reproducing

```bash
# prep (no GPU; nix's torch is broken here, so use the qualified vLLM interpreter)
VLLM_ROCM_LIB=/opt/rocm/core-7.14/lib build-gemma31/vllm-python \
  scripts/glm52_prep_lite.py --model /workspace/models/GLM-5.3-FP8 \
                             --out /workspace/models/GLM-5.3-plow-lite

# objects (no GPU)
nix develop --command bash -c 'PLOW_DECODE_BATCH=4 JOBS=24 \
  bash scripts/build_gfx942.sh /app/plow/build-glm53/hsaco'
# ...or, once the packet exists, FROM the packet: `PLOW_HSACO_CONFIG=<assets dir>` reads the
# `plow_config.h` plowc writes beside model.pkt, stamps every object with the packet's pairing
# hash (plowrt refuses a stamped object against any other packet), and derives the decode
# batch/walk, the low-rung tiers from the decode ladder, the packed-family rows and their `_tb`
# twins (for a packet with token-batch body programs), and the opt-in arms
# `backends.gfx942.requires` names (PLOW_DSA_PF, PLOW_MOE_PF_*, ...) instead of taking them
# from the environment. An env var that would build an object the loader refuses by name (a
# GM_BM that disagrees with the packet, a narrower decode batch) fails the build instead.
nix develop --command bash -c 'PLOW_HSACO_CONFIG=/app/plow/build-glm53/tp4 JOBS=24 \
  bash scripts/build_gfx942.sh /app/plow/build-glm53/hsaco'

# emit, serve, smoke, bench — every GPU process takes a gpulease
MAXCTX=10240 scripts/glm53_mi300x.sh emit 4
MAXCTX=10240 scripts/glm53_mi300x.sh serve 4 18930
scripts/glm53_mi300x.sh smoke 18930
IN_LENS="1024 4096" CONCS="1 4" scripts/glm53_mi300x.sh bench 4 18930 plow
```

Load: 4 ranks bound in 30.5 s, 181.75 GiB/rank at ~6.0 GiB/s. Engine banner
`gfx942 n_cu=304 batch=4 decode_rungs=[1,2,4] max_ctx=10240`.
Coherence: "The capital of France is Paris."; a 214-token free-form answer was accurate and
drift-free.

### Two defects this bringup found

1. **plowrt outside `nix develop` silently serves from the CPU.** The flake's ROCm 7.14 tree is
   what carries `libhsa-runtime64.so.1`; outside it the dlopen fails and plowrt logs
   `CPU reference backend active` and then answers correctly at fictional speed. The serve now
   runs inside nix (`scripts/glm53_serve_inner.sh`).
2. **`build_gfx942.sh` armed `PLOW_MOE_PF_DET` on prefill objects only.** Above
   `PLOW_DECODE_BATCH=1` the GLM decode program emits its MoE seam with the grouped PREFILL
   family, so op 86/87's deterministic arm is reached from the DECODE object. The load was
   refused with a packet/object mismatch. Fixed in the `PLOW_DECODE_BATCH>1` branch.

## 5. TP4 baseline — plow only, and NOT yet a comparison

128 output tokens, 8 prompts per cell, random dataset, prefix caching off, greedy
(the AMD engine samples argmax on device and ignores temperature).

| input | conc | output tok/s | median TTFT ms | median TPOT ms | median E2EL ms |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 23.78 | 383 | 38.70 | 5308 |
| 1024 | 4 | 38.93 | 840 | 95.06 | 13086 |
| 4096 | 1 | 18.31 | 1024 | 47.02 | 6992 |
| 4096 | 4 | 29.48 | 1698 | 122.23 | 17171 |

**Every one of these numbers carries three caveats, and none of them is cosmetic.**

1. **No vLLM arm yet.** All eight cards were leased throughout by a concurrent agent session
   running an unrelated Gemma-4-31B campaign, so the matched vLLM 0.28 reference — which needs
   four of them — has not been taken. Nothing here says anything about beating vLLM.
2. **The tuning store was fully stale for this table** — 868 records, zero usable — so both
   blobs reported `tile_source: analytical`, `tile_measured: 0` and the TTFT column above is
   unmeasured-tile. **FIXED since**, see §5b: the store was fine; the PROBE was wrong.
3. **Measured under contention.** Four to six foreign GPU processes were live on the other
   cards. TP4 collectives cross the fabric those processes also touch. gpulease's own rule is
   that a contended run invalidates timings. Re-measure on a quiet box.

**Known asymmetry for whenever the vLLM arm is taken:** vLLM serves this checkpoint with DSA
armed (`index_topk` 2048) while this blob is emitted `PLOW_GLM_DSA=0` and reads EVERY KV row.
Above ~2k context plow is doing strictly more attention work per token. A plow win under that
asymmetry is a real win; a plow loss is not by itself evidence about plow's kernels.

## 5a. DEC_SQUEEZE decode tier — a rung-1 lever, and not yet adoptable

`PLOW_DEC_SQUEEZE=1` (WPE=3 register recut, `MPF_BK=32`, `NO_MLA_DEC`) built into
`build-glm53/hsaco-squeeze` and served as `PLOW_HSACO_LOWRUNG=<dir>:4`. Its K3 rows fail to
compile, which does not matter — GLM loads `interp_decode_gq.elf`. Both arms ran on the same
four cards minutes apart.

| input | conc | TPOT base | TPOT squeeze | delta | TTFT base | TTFT squeeze |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 38.70 | 34.98 | **−9.6%** | 383.1 | 380.4 |
| 1024 | 4 | 95.06 | 95.23 | +0.2% | 839.9 | 841.4 |
| 4096 | 1 | 47.02 | 45.53 | −3.2% | 1023.8 | 1033.9 |
| 4096 | 4 | 122.23 | 122.07 | −0.1% | 1697.8 | 1698.4 |

TTFT unmoved is the correct negative control: the tier swaps only the decode object. The win is
confined to rung 1, matching GLM-5.2's record that this is a rung-1 lever.

**It is a measurement, not an adoption.** The same greedy prompt diverges between the two arms
(they agree for ~250 characters, then the baseline stops at 214 tokens and the squeeze arm runs
to the 220 cap), so on GLM-5.3 this is a new numerics class and needs the paired accuracy gate.
DEC_SQUEEZE was recorded byte-identical on GLM-5.2; that record does not transfer.

The conc-4 aggregate-throughput cell (38.93 -> 29.07 tok/s) is NOT a regression: medians agree
to within 0.2% and the whole difference is one request that waited 11 s to first token on a
contended box.

## 5b. The tuning store was never stale — the probe was

`plowc` folds the TOOLCHAIN into the tuning digest, and a plowc run outside `nix develop` sees a
different toolchain from the one that built the objects. Every emit above ran outside nix:

    outside nix: build digest gfx942-f390fbaa86c582ff -> 868 records, all "stale"
    inside  nix: build digest gfx942-d042b33e5e07ea56 -> matches the campaign

The fix is procedural. Run plowc inside `nix develop`, the same shell that built the objects.
Two campaigns (`scripts/glm53_tune_gemm_inner.sh`, ~50 s each on one leased GPU) cover TP4 and
TP8 — the per-rank N halves at TP8, so it is a different shape set and the TP4 campaign alone
leaves TP8 at `mixed`, 1236/2472.

    build-glm53/tp4m   tile_source measured  2472/2472
    build-glm53/tp8m   tile_source measured  2472/2472
    build-glm53/tp4f   + PLOW_GLM_XR_RES=1 GLM_FUSE_XRN=1  (see the collectives analysis)
    build-glm53/tp4q   + GLM_LINEAR_FP8=1 against GLM-5.3-plow-q

## 5c. vLLM 0.28 above TP1 on this host — three breakages, one root cause

The vLLM interpreter needs the nix glibc (the wheel wants >= 2.39; the host has 2.35), and its
workers inherit that `LD_LIBRARY_PATH`. Every SYSTEM binary they fork then dies.

1. `ROCm version file not found` — workers read `<ROCM_PATH>/.info/version` at `init_device`;
   `/opt/rocm` has no `.info`. TP1 never reached this path.
2. `Get GPU arch from rocminfo failed ... exit status 127` — AITER forks `rocminfo`. Both
   `/opt/rocm/core-7.14/bin/rocminfo` and `/opt/rocm-7.2.4/bin/rocminfo` return 127 under that
   `LD_LIBRARY_PATH` and 0 without it.
3. Triton forks `/usr/bin/gcc` for `hip_utils.c`, same death.

The shim has to be on PATH (`aiter/jit/utils/cpp_extension.py:78` tries `shutil.which` first),
its shebang has to be a nix binary (a system `/bin/sh` cannot load), and it has to clear the
variable with a SHELL BUILTIN — coreutils `env -u` is itself a system binary and dies with
`undefined symbol: __tunable_is_initialized`. `build-glm53/rocm-shim/bin/{rocminfo,vllm-cc}`.
With those, vLLM 0.28 loads GLM-5.3 at TP4: 176.22 GiB/rank in 161.8 s.

## 5d. All plow TP4 arms, same four cards, minutes apart

| cell | metric | plow (analytical) | tp4m (measured tiles) | **tp4f (+ collective folds)** |
|---|---|---:|---:|---:|
| 1024 / c1 | tok/s | 23.78 | 23.97 | **24.20** |
| 1024 / c1 | TTFT ms | 383 | 388 | 385 |
| 1024 / c1 | TPOT ms | 38.70 | 38.59 | **38.21** |
| 1024 / c4 | tok/s | 38.93 | 39.23 | **39.94** |
| 1024 / c4 | TTFT ms | 840 | 844 | **835** |
| 1024 / c4 | TPOT ms | 95.06 | 94.75 | **93.62** |
| 4096 / c1 | tok/s | 18.31 | 17.66 | **19.22** |
| 4096 / c1 | TTFT ms | 1024 | 1042 | 1030 |
| 4096 / c1 | TPOT ms | 47.02 | 49.84 | **43.69** |
| 4096 / c4 | tok/s | 29.48 | 29.05 | **29.59** |
| 4096 / c4 | TTFT ms | 1698 | 1782 | **1692** |
| 4096 / c4 | TPOT ms | 122.23 | 123.65 | 122.24 |

**Measured tiles are a NULL on this model.** Every cell is inside this box's noise and the sign
is not even consistent. The recipe ranks the tile campaign as lever #1, but that ranking was
written when the store was stale and the prize was UNKNOWN. It is consistent with the GLM-5.2
sweep's own note that "every GLM-5.2 narrow shape agrees" with the analytical model — the
campaign CONFIRMS the model on GLM shapes rather than correcting it. Its value is that TTFT can
now be REPORTED as measured-tile, not that it is faster. Do not re-run it expecting a win.

**The collective folds are the only lever that paid** — see
this document. Best arm in every cell, and clearly better at 4096/c1.

## 5e. GLM_LINEAR_FP8 HANGS on GLM-5.3 / gfx942

The fourth arm (`tp4q`: `GLM_LINEAR_FP8=1` against the `GLM-5.3-plow-q` weight dir, which keeps
`o_proj` and the shared expert in their checkpoint block-fp8 form) loads and passes the
coherence smoke, then hangs on the first benchmark cell: the server log stops at
`decode ladder rung rung=1 occupied=1` and never advances, the four leased GPUs sit at 100%
utilization producing no tokens, and the process burns 2685 s of CPU over 39 minutes. That is
the persistent-megakernel spin. Torn down with TERM, never `-9` — the recipe records that a
`-9` leaves the megakernel resident and wedges the box.

The prep is exonerated: it republishes the checkpoint's own fp8 bytes and `[128,128]` scale
grids VERBATIM (10.69 GB in 9 s, no requant), and the blob binds the right names —
`plowrt disasm build-glm53/tp4q --program 1` shows `o_proj` served by `GemvFp8Blk` at `b=304`
with its scale grid, 303 fp8 bindings in the decode program. The suspect is the decode-side
`GemvFp8Blk` arm at these shapes. Diagnosing it needs a single-layer block harness, not a
full-model serve.

This parks the largest remaining decode lever: it would take ~2.7 GB/rank/token off a 16.0 GB
stream, about −17% if decode were purely bandwidth-bound. The blob and the weight dir are both
built and on disk, so the retry after a kernel fix is cheap.

## 5f. No reference engine can serve GLM-5.3 on this host

Both candidate references were taken as far as they go. Neither runs, and in both cases the
blocker is GLM's DSA sparse-attention indexer meeting this host's toolchain.

**vLLM 0.28.** With AITER the server loads (176.22 GiB/rank in 161.8 s), allocates 66,912
tokens of KV cache, reports startup complete, and then the FIRST forward dies inside the
indexer's own kernel:

```
aiter/ops/flydsl/kernels/fp8_mqa_logits.py
  -> flydsl/compiler/jit_function.py:784 _run_pipeline
  -> MLIRError: Failure while executing pass pipeline:
     error: "-":2:3: lld invocation failed
```

reached from `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`. The `lld` call is INSIDE
flydsl's bundled MLIR, not a subprocess, so the PATH shims that fixed `rocminfo` and `gcc`
cannot reach it; pointing `ROCM_PATH` at the complete `/opt/rocm-7.2.4` tree changed nothing.
With AITER off vLLM refuses at startup — `Sparse attention indexer ROCm path is only supported
on AITER` — because the indexer has no non-AITER ROCm kernel and no flag disables DSA.

**SGLang 0.5.19.** The model is first-class (`GlmMoeDsaForCausalLM` in `models/glm4_moe.py`,
registered with NextN/MTP handling) and its ROCm DSA branch is real: `_use_aiter` gated on
`is_hip()`, routing through `aiter_paged_mqa_logits`, which lives in `aiter/ops/**triton**/`
rather than `aiter/ops/flydsl/` — so SGLang would sidestep the exact JIT that kills vLLM. It
still cannot run: `sglang` imports `sgl_kernel`, PyPI ships only a CUDA build of it
(`cp310-abi3-linux_x86_64`, fails with `libnvrtc.so.12`), there is no `sgl-kernel-rocm`, and
SGLang's ROCm kernels ship inside container images while this host has no container runtime
(docker, podman, nerdctl, singularity, apptainer all absent).

**Consequence.** The plow-vs-reference comparison is unavailable for GLM-5.3 on this host. That
is a statement about ROCm DSA support in both engines, NOT a plow result, and no number in this
document should be presented as beating either of them.

## 6. Open levers, in the order worth taking

1. **Measured GEMM tiles** (§5 caveat 2). Cheapest, and it gates the ranking of everything else.
2. **`PLOW_DEC_SQUEEZE=1` decode tier.** The validated WPE=3 register recut; on GLM-5.2 it was
   rung-1 TPOT −12.3%. Built here into `build-glm53/hsaco-squeeze` (its K3 rows fail to
   compile, which is fine — GLM loads `interp_decode_gq.elf`) and served via
   `PLOW_HSACO_LOWRUNG=<dir>:4`. A/B not yet taken.
3. **`GLM_LINEAR_FP8=1`.** Decode is HBM-bound — the emit's own lean oracle puts the TP4 decode
   step's floor at 3.0 ms against 16.0 GB touched per step — and this keeps ~5 GB/rank/token of
   `o_proj` + shared expert in fp8 instead of bf16.
4. **TP8**, once eight cards are free.

**Closed:** cross-request co-packed prefill (`PLOW_PACKED_PREFILL_ROUTE`). Qualified 2026-09-08
and it is a null — see the corresponding section of this document. Getting it to fire at all needed four
gates opened (gfx942 family objects, a GLM packed-prefill emit topology, the runtime route, and
`PLOW_PF_CHUNK`); with all four open it is +0.7-1.1% throughput and -2.5-2.9% TTFT at the only
two shapes it reaches, inside the noise. The by-product worth keeping is that the runtime now
says at load which gate is shut instead of silently packing nothing.


---

## GLM-5.3 on MI300X: roofline, trace attribution, and what to fix

Measured 2026-09-07, 8x MI300X (gfx942, 304 CU, 192 GiB), ROCm 7.14 from the flake.
Blobs: `build-glm53/tp4f` and `tp8f` — block-fp8 weights, measured GEMM tiles, collective folds,
decode rungs 1/2/4, max-ctx 10240.

## 1. The ceiling is not the spec sheet

MI300X advertises 5325 GB/s. Nothing reaches it, and a roofline drawn against it flatters the
kernel. Measured on one card (`scripts/glm53_hbm_ceiling.py`):

| probe | GB/s | % of 5325 |
|---|---:|---:|
| read, 8 GiB reduction | 4112 | 77% |
| read, 2 GiB reduction | 4290 | 81% |
| copy, 8 GiB (read+write) | 3729 | 70% |
| **GEMV [8192,2048] bf16** (q_absorb, TP4) | **2204** | **41%** |
| **GEMV [6144,4096] bf16** (o_proj, TP4) | **2501** | **47%** |
| **GEMV [6144,2048] bf16** (o_proj, TP8) | **1434** | **27%** |

The last row is the important one. **At TP8 the per-rank matrices are small enough that a single
GEMV is launch-bound, not bandwidth-bound**: 25.2 MB should take 6.1 µs at the read ceiling and
takes 17.5 µs. Halving the bytes per rank does not halve the time, and any TP8 projection that
assumes it will is wrong.

All rooflines below use **4100 GB/s**, the measured read ceiling.

## 2. Where the bytes are — computed from the emitted packet stream

`scripts/glm53_roofline.py` reads `plowrt disasm` output, so it prices the shapes that actually
execute, not the ones the model card implies. Per rank, one decode token:

| | TP4 | TP8 |
|---|---:|---:|
| GemvQkv (q_a, kv_a_latent, k_rope, q_absorb, q_rope) | 5.46 GB (29%) | 3.99 GB (36%) |
| Gemv (o_proj, lm_head, …) | 5.11 GB (27%) | 2.67 GB (24%) |
| MoE expert walk (glu + down, block-fp8) | 5.66 GB (30%) | 2.83 GB (25%) |
| FlashMlaDecode (MLA latent cache) | 0.92 GB (5%) | 0.92 GB (8%) |
| everything else | 1.46 GB | 0.74 GB |
| **total** | **18.62 GB** | **11.15 GB** |
| arithmetic intensity | 2.05 FLOP/B | 1.84 FLOP/B |
| **memory roof @ 4100 GB/s** | **4.54 ms** | **2.72 ms** |

Ridge point is 319 FLOP/byte, so decode is memory bound by two orders of magnitude — the
compute roof is 0.03 ms. **The attention projections are 57% of the decode byte stream**, more
than the MoE experts, because only 8 of 256 experts are read per token while every attention
matrix is read in full. TP8 does NOT halve the total: `q_a_proj`, `kv_a_latent` and `k_rope` are
REPLICATED per rank (`Nq=2048 K=6144` at both degrees), so only the sharded half shrinks.

Prefill at T=2048, TP4: 276.5 GB and 51.4 TFLOP, AI 186 FLOP/byte against a ridge of 319 — it
sits just on the memory side, roof 51.9 ms memory / 39.3 ms compute.

## 3. Where the time goes — traced

`plowrt bench --trace-raw` writes one `PlowTraceRec` per (workgroup, packet) carrying
`t_arrive`, `t_ready`, `t_end` on a constant 100 MHz clock. `scripts/glm53_trace_attrib.py`
folds them per packet into **stall** (`t_ready - t_arrive`, waiting on wait-counters) and
**body** (`t_end - t_ready`, the op itself). TP4, in=1024, one decode token:

| op | packets | wg/pkt | stall ms | body ms | span ms | % |
|---|---:|---:|---:|---:|---:|---:|
| MOE_EXPERT_GLU_FP8_BLK | 600 | 32 | 16.86 | 32.00 | 48.86 | 26.7 |
| MOE_EXPERT_DOWN_FP8_BLK | 600 | 32 | 3.73 | 33.73 | 37.46 | 20.5 |
| GEMV | 229 | 220 | 13.10 | 12.81 | 25.91 | 14.2 |
| GEMV_QKV | 156 | 295 | 11.50 | 7.11 | 18.61 | 10.2 |
| HEADNORM_ROPE | 156 | **1** | 8.77 | 1.36 | 10.13 | 5.5 |
| FLASH_MLA_DECODE | 78 | 160 | 4.88 | 3.65 | 8.53 | 4.7 |
| MLA_MERGE_FOLD | 78 | 128 | 4.21 | 2.02 | 6.24 | 3.4 |
| ADD_NORM | 77 | **1** | 4.28 | 0.76 | 5.04 | 2.8 |
| XREDUCE | 78 | 12 | 3.99 | 0.62 | 4.61 | 2.5 |
| RMSNORM | 80 | **1** | 3.53 | 0.69 | 4.21 | 2.3 |
| **total (serialized)** | **2445** | | **81.21** | **101.58** | **182.78** | |

Wall span of the trace is 36.4 ms against a 182.8 ms serialized sum, so the global queue is
overlapping about 5x — that part is working. Two things stand out:

**Stall is 44.4% of the chain.** Nearly half of all packet time is spent waiting on producers,
not computing. The single-workgroup packets are the worst offenders by ratio: `HEADNORM_ROPE`
is 8.77 ms of stall against 1.36 ms of body, and `ADD_NORM` and `RMSNORM` are similar. These are
tiny ops sitting on the critical path, and 313 of them run on ONE workgroup of 304.

**The MoE expert walk is 47% of the chain and is far off its own roof.** Its 32-workgroup width
is correct by design — 8 routed experts x 32 CU + 48 for the shared expert fills all 304
concurrently — so this is a kernel-efficiency problem, not a dispatch-width one. Per packet it
moves 6.3 MB in 53 µs = 119 GB/s, against the ~432 GB/s a 32-of-304 CU slice should reach:
**3.7x off its fair share of bandwidth**.

## 4. The gap, stated plainly

| | TP4 decode |
|---|---:|
| memory roof @ measured 4100 GB/s | 4.54 ms |
| measured TPOT | 36.74 ms |
| **off roofline** | **8.1x** (12% of roof) |

Host time is not the problem: the DSTEP breakdown puts **94.0% of the token in the GPU drain**
and 5.7% in host work. Of that host share the largest single item was the cross-rank safety
audit at 1.21 ms/token.

## 5. What was fixed, and what is measured

**TP safety audit off** (`PLOW_TP_NO_AUDIT=1`) — the audit is a redundant-rank counter check
that runs every token. A/B at in=1024/out=32/conc=1, same cards, minutes apart:

| | TPOT p50 | tok/s |
|---|---:|---:|
| audit on (default) | 37.99 ms | 21.00 |
| audit off | **36.74 ms** | **21.55** |

−3.3% TPOT for one environment variable. It removes a correctness net, so the better shipping
answer is `--amd-tp-agree-every N` to keep the check at a cadence rather than per token.

**Collective folds** (`PLOW_GLM_XR_RES=1 GLM_FUSE_XRN=1`) — already measured in
this document; best arm in every cell.

### Real-world workloads, TP4, folds + audit off

| workload | in/out | conc | tok/s | TTFT p50 | TPOT p50 |
|---|---|---:|---:|---:|---:|
| chat | 512 / 256 | 1 | 27.00 | 276 ms | 36.07 ms |
| chat | 512 / 256 | 8 | 45.03 | 1266 ms | 87.18 ms |
| code | 2048 / 512 | 1 | 25.14 | 540 ms | 38.82 ms |
| code | 2048 / 512 | 8 | 41.70 | 2320 ms | 94.71 ms |

**Concurrency 8 buys only 1.67x aggregate throughput** and more than doubles per-request TPOT.
The compiled ladder tops out at rung 4, so half the requests queue — but the deeper reason is
that a fine-grained MoE does not amortize across a batch: 4 tokens with top-8-of-256 routing
touch up to 32 distinct experts, so the expert stream grows nearly linearly with batch size
while the attention stream amortizes. Widening the ladder helps less than the roofline suggests.

## 7. What landed, stacked and measured

Two fixes, both bit-identical, both measured on the same four cards minutes apart.

**a. The MoE decode DOWN lane-group map was silently inactive at TP4.** The narrow-K arm covers
`K <= (64/RG)*16` and `RG` was a build-time constant pinned at 4, i.e. `K <= 256`. `I_moe` is a
PER-RANK quantity, so GLM-5.3 routes DOWN at 256 on TP8 and **512 on TP4** — the arm fired at
TP8 and fell through to the per-row walk at TP4, with no diagnostic. It is now chosen from the
actual contraction (`runtime/amd/op_moe.h`); `PLOW_MOE_DEC_LG_RG=2|4` still pins one map for A/B.

| in=1024 out=32 conc=1 | TPOT p50 | tok/s | TTFT p50 |
|---|---:|---:|---:|
| shipped, RG=4 pinned (arm inactive) | 36.73 ms | 21.59 | 343.1 ms |
| **adaptive (picks RG=2)** | **35.46 ms** | **22.17** | 343.6 ms |
| pinned RG=2 (control) | 35.45 ms | 22.17 | 344.4 ms |

−3.5% TPOT. TTFT unmoved, the correct negative control for a decode-only axis. The adaptive arm
reproduces the pinned control exactly, a 220-token greedy generation is CHARACTER-IDENTICAL to
the shipped object, and the register cliff does not move (VGPR 256, LDS 64560, occupancy 2).

**b. The per-token cross-rank audit** costs 1.21 ms/token: 37.99 → 36.73 ms, −3.3%.

Stacked, at in=1024/out=32/conc=1: **37.99 → 35.46 ms TPOT, −6.7%.**

### TP4 real-world workloads, everything on

| workload | in/out | conc | tok/s | TTFT p50 | TPOT p50 |
|---|---|---:|---:|---:|---:|
| chat | 512 / 256 | 1 | 28.03 | 275 ms | 34.72 ms |
| code | 2048 / 512 | 1 | 26.23 | 535 ms | 37.14 ms |
| long context | 8192 / 256 | 1 | 21.22 | 1891 ms | 38.66 ms |
| chat | 512 / 256 | 8 | 45.03 | 1266 ms | 87.18 ms |
| code | 2048 / 512 | 8 | 41.70 | 2320 ms | 94.71 ms |

## 8. Two things that shape real-world use

**Concurrency does not amortize on a fine-grained MoE.** Eight concurrent requests buy 1.67x
aggregate throughput and more than double per-request TPOT. Part of that is the compiled ladder
topping out at rung 4 so half the requests queue — but the deeper reason is structural: 4 tokens
with top-8-of-256 routing touch up to 32 distinct experts, so the expert stream grows nearly
linearly with batch while only the attention stream amortizes. Widening the ladder will help
less than the decode roofline suggests, because the roofline per token barely falls with batch.

**GLM-5.3 always thinks.** Its template has no disable branch and leaves `<think>` open, so a
short question spends its whole budget on the trace: a 220-token greedy answer to "explain MoE
routing in one paragraph" is entirely reasoning. plowrt splits the trace into
`reasoning_content` once `</think>` arrives, so a client sees the answer in `content` — but
`max_tokens` has to be budgeted for trace plus answer, not answer alone. Any workload sizing
that assumes otherwise will look like the model is producing nothing.

## 6. Ranked by measured prize, not by intuition

1. **The MoE expert-walk kernel body** — 47% of the chain, 3.7x off its fair-share bandwidth,
   and it is BODY time (65.7 ms of the 101.6 ms total body), not stall. Nothing else on this
   list is worth as much.
2. **The single-workgroup spine** — `HEADNORM_ROPE`, `ADD_NORM`, `RMSNORM` are 313 packets on
   one workgroup each, 16.6 ms of stall for 2.8 ms of body. They are latency, not throughput:
   fusing them into their neighbours removes packet boundaries from the critical path.
   `PLOW_GLM_FUSE_QNORM` does exactly this for the q-norm but is refused alongside a decode
   ladder, so it is reachable only for a latency-tuned rung-1 blob.
3. **`GLM_LINEAR_FP8`** would cut the bf16 attention projections — 57% of the byte stream — but
   it HANGS the megakernel on GLM-5.3 (see §5e of this document). Fixing that kernel is worth
   about 2.7 GB/token off an 18.6 GB stream.
4. **Measured GEMM tiles: a null.** Already run; do not re-run expecting a win.

## 9. TP8 is emitted and gated but NOT measured

`build-glm53/tp8f` is built with the same recipe as the TP4 arm (measured tiles, collective
folds, rungs 1/2/4) and its roofline is in §2 — 11.15 GB/rank/token for a 2.72 ms memory roof
against TP4's 4.54 ms. It has never been served.

The reason is contention, not the model: a second agent campaign on this host cycles
single-GPU leases continuously, and an 8-of-8 `gpulease` request loses every race against a
stream of 1-of-8 ones. One attempt waited the full 5400 s and timed out with the message
`glm53-TP8 ... TIMEOUT after 5400s (wanted 8 of 8)`.

**Do not project the TP8 number from the roofline.** §1 measures a [6144,2048] bf16 GEMV — the
TP8 per-rank `o_proj` shape — at 1434 GB/s against 2501 GB/s for the TP4 [6144,4096] shape.
The smaller matrix is launch-bound: 25.2 MB takes 17.5 µs where the read ceiling implies 6.1 µs.
So TP8 halves the bytes and does NOT halve the time, and the gap between the 1.67x the roofline
suggests and whatever is actually measured is the interesting quantity. That is precisely why
the measurement is worth taking rather than inferring.

One thing the TP8 emit already benefits from: the MoE DOWN lane-group arm in §7a fires natively
at TP8 (`I_moe = 256`, inside the RG=4 map) and always did. The adaptive map fixed TP4, which
was the degree that had silently lost it.

## 10. Decode KV format and split policy: both byte levers cost more than they save

Measured 2026-09-07, written up in full in
this document. The short
form, because it revises a recommendation this document makes:

§8 item 2 above and the attention roofline both say decode wants fewer KV bytes rather than a
faster kernel. **Both byte levers were served end to end and neither returned time.**

| lever | result |
|---|---|
| fp8 latent KV (`PLOW_GLM_FP8_KV`, ops 109/110) | TPOT **+1.5 to +3.6%**, TTFT **+4 to +38%** across 1k–65.5k, both growing with context. Buys 1.79x max context: 131 072 loads where bf16 OOMs. |
| DSA sparse decode (`PLOW_GLM_DSA`) | **Cannot load on GLM-5.3 at all** — the checkpoint's `indexer.wq_b`/`wk` are block-fp8 and `devgen` declares them bf16. |
| live-`kv_len` split count (`PLOW_MLA_NS_LIVE`, new) | **+4.3% at live 1024**: the gfx950 GLM-5.2 split ladder does not transfer to gfx942. |

What DOES transfer from the roofline is the shape: dense decode TPOT is linear in context at
**0.134–0.143 ms per 1 000 tokens** on every arm measured here, reproducing the 0.136 ms/1k the
`GlmCfg::dsa` note measured on MI350X. Decode attention is small, memory-bound and linear — and
that is exactly why halving its bytes cannot pay: a roof 10x above the measured kernel is not a
budget, and a per-KV-row cost paid to halve it is charged in full.

So the ranking in §8 stands with one correction: **do not spend effort on decode KV bytes for
GLM-5.3 on this hardware.** The remaining prizes are unchanged — the MoE expert walk and the
MLA prefill kernel.


---

## GLM-5.3 TP4 on MI300X: long context and throughput mode

Measured 2026-09-07 on 8x MI300X (gfx942), ROCm 7.14 from the flake, through `plowrt bench`
(the production mux, no HTTP). Blobs carry measured GEMM tiles, the collective folds, and the
adaptive MoE lane-group map; every run has `PLOW_TP_NO_AUDIT=1`.

- `build-glm53/tp4-long` — max-ctx 32768, decode rungs 1,2 (KV 5.5 GiB/rank of ~13.4 GiB free)
- `build-glm53/tp4-thru` — max-ctx 8192, decode rungs 1,2,4,8, served on `hsaco-b8` (GEMV_MM=8)

## 1. Long context: TTFT is the problem, TPOT is not

| context | TTFT p50 | TPOT p50 | tok/s |
|---:|---:|---:|---:|
| 4 096 | 914 ms | 38.21 ms | 22.19 |
| 8 192 | 1 938 ms | 39.83 ms | 18.33 |
| 16 384 | 4 618 ms | 39.29 ms | 13.31 |
| 32 000 | 12 136 ms | 40.49 ms | 7.39 |

**TPOT moves 6% across an 8x context increase.** The MLA latent cache is one 576-element row per
position shared across heads, so decode attention grows slowly against a weight stream that does
not grow at all. Whatever is wrong with decode, long context is not it — and the decode-side
attention kernel does not need work.

**TTFT grows superlinearly**, 2.12x / 2.38x / 2.63x for successive doublings against 2.0x for
linear. Fitting `TTFT = a*T + b*T^2` on the 8192 and 32000 points:

    TTFT = 0.1875*T + 5.993e-6*T^2   ms

| context | linear term | quadratic term | attention share |
|---:|---:|---:|---:|
| 4 096 | 768 ms | 101 ms | 11.6% |
| 8 192 | 1 536 ms | 402 ms | 20.8% |
| 16 384 | 3 072 ms | 1 609 ms | 34.4% |
| 32 000 | 5 999 ms | 6 137 ms | **50.6%** |

The quadratic term is prefill attention. At T=32000, TP4 (16 heads/rank), causal, with MLA's
576-wide QK and 512-wide PV over 78 layers, that is **1390 TFLOP per rank**. Against the 6137 ms
the fit attributes to it, the MLA prefill flash kernel is running at **227 TFLOP/s, 17% of the
1307 TFLOP/s bf16 matrix peak**.

So long-context work splits cleanly: below ~8k it is the linear term (MoE weight streaming and
the dense GEMMs), and above ~16k it is the attention kernel.

## 2. Both long-attention knobs are nulls

| arm | TTFT at 32k | vs shipped |
|---|---:|---:|
| `PLOW_GLM_PF_NS=2` (shipped) | 12 136 ms | — |
| `PLOW_GLM_PF_NS=4` | 12 126 ms | −0.08% |
| `PLOW_MLA_PF_SMX=1` (shipped softmax dedup) | 12 194 ms | — |
| `PLOW_MLA_PF_SMX=0 PLOW_MLA_PF_QK1=1` | 12 214 ms | +0.16% |

`PF_NS` is the causal KV-split factor and `QK1` is the alternative softmax dedup — the two are
mutually exclusive (`op_attention_common.h` refuses the pair) and `SMX` is the CDNA3 default
(`op_attention_gfx942.h`). Neither
moves the number. **The 17%-of-peak attention kernel is not limited by softmax redundancy or by
KV-split parallelism**, which is worth knowing because both are the obvious first guesses. The
QK1 object builds at an identical register cliff (VGPR 256, LDS 64560, spill 126), so this is a
real comparison and not a resource artefact.

What that leaves, untested here: the MFMA tiling and the 64 KiB LDS arena. MLA's absorbed form
gives a 576-wide QK and 512-wide PV per head, which is unusually fat, and the flash object
already sits at VGPR 512 / AGPR 256 / occupancy 1.

## 3. Throughput mode: do NOT widen the decode ladder past 4

in=1024, out=128, TP4.

| blob | objects | conc | tok/s | TTFT p50 | TPOT p50 |
|---|---|---:|---:|---:|---:|
| rungs 1,2,4 | MM=4 | 1 | 26.37 | 355 ms | 35.41 ms |
| rungs 1,2,4 | MM=4 | 8 | **41.21** | 12 688 ms | 94.40 ms |
| rungs 1,2,4,8 | MM=8 | 8 | 26.59 | 1 437 ms | **286.19 ms** |
| rungs 1,2,4,8 | MM=8 | 16 | 26.94 | 38 099 ms | 289.62 ms |

Adding rung 8 **costs 35% of aggregate throughput and triples TPOT**. The wider ladder does
admit all 8 requests at once — TTFT drops from 12.7 s to 1.4 s because nothing queues — but each
token then costs 3x as much, and the trade is a net loss.

Isolated, so the cause is not ambiguous. The same rungs-1,2,4 blob on both object sets:

| blob | objects | conc | tok/s | TPOT p50 |
|---|---|---:|---:|---:|
| rungs 1,2,4 | MM=4 | 8 | 41.19 | 94.54 ms |
| rungs 1,2,4 | MM=8 | 8 | 40.82 | 94.83 ms |

The wider GEMV bucket costs 0.3%. **The loss is rung 8 itself.** That is the fine-grained-MoE
non-amortization made quantitative: 8 tokens with top-8-of-256 routing touch up to 64 distinct
experts per layer against up to 32 at rung 4, so the expert weight stream roughly doubles while
only the attention stream amortizes. Batching a 256-expert MoE does not buy what batching a
dense model buys, and the decode roofline per token barely falls with batch size.

## 4. All-reduce is already small; the one lever is structurally blocked

At T=8192 an `XReduceTwoShot` seam moves `T*hidden*2` = 100.7 MB, and there are 156 of them, so
15.70 GB per chunk. Over the 896 GB/s per-GPU fabric that is 17.5 ms per chunk, and a 32000-token
prefill is about four chunks: **~70 ms of a 12 194 ms TTFT, 0.6%.**

On the decode side the trace puts `XREDUCE` at 1.8-2.5% of the serialized chain, and 73-87% of
that span is STALL rather than transfer — the collective is waiting for its producers, not moving
bytes slowly.

The available lever, `PLOW_GLM_XR_BAND`, cannot be measured on the shipping path at all:

    PLOW_RAGGED_CHUNK cannot serve this packet: prefill bucket T=2048 instruction #15 (op 15)
    carries a non-zero row-band offset, which is the PLOW_GLM_XR_BAND layout. The ragged row
    shrink would rescale the banded collective without shrinking the band.

Banding is incompatible with ragged chunking, which is default-on and is the fewest-launch cover.
Testing it would mean giving up ragged chunking to chase 0.6%. **Not worth doing, and the reason
is arithmetic rather than taste.**

## 5. Where the remaining prize is

1. **The MLA prefill flash kernel at 17% of bf16 peak.** It is 50.6% of TTFT at 32k and rises with
   context. The two obvious knobs are measured nulls, so this is tiling and LDS work, not a flag.
   *(§7: the deferred-frame softmax took it to 18.9% of peak; what is left is the PV transpose's
   256 `ds_read_u16` per KV tile.)*
2. **The MoE decode expert walk**, still 47% of the decode chain at 3.7x off its fair-share
   bandwidth (§3 of this document).
3. **NOT the all-reduce** (0.6%), **NOT the decode attention** (TPOT flat across 8x context),
   **NOT a wider decode ladder** (a measured 35% throughput loss).

## 6. Inside the MLA prefill flash kernel: what the ablations say

The kernel carries four ablation probes (`PLOW_MLA_PF2_ABL=1..4`, wrong output by design, one
inner-loop term deleted each). Measured at T=32000, TP4, against a 12 094 ms baseline TTFT:

| probe | TTFT | term cost | share of TTFT |
|---|---:|---:|---:|
| 3 — no softmax math | 8 432 ms | **3 662 ms** | **30.3%** |
| 2 — no QK MFMA | 9 840 ms | 2 255 ms | 18.6% |
| 4 — no PV | 9 930 ms | 2 164 ms | 17.9% |
| 1 — no K-slab global loads | 10 613 ms | 1 482 ms | 12.2% |

Terms are not additive — removing one also unblocks latency behind it — but the ranking is
unambiguous: **the softmax is the largest single term in the kernel, larger than either MFMA
pass.** That is VALU work (running max, running sum, exp, and the 512-wide accumulator rescale)
on a kernel that runs at occupancy 1, one wave per SIMD, where there is no co-resident wave to
hide it behind.

### The bank-conflict arm was already on

`PLOW_MLA_PF_SV` — the kv-block LDS swizzle that turns the PV transpose's 4-way bank conflict
into conflict-free reads, plus two fragment double-buffers — is `${PLOW_MLA_PF_SV:-1}` in
`build_gfx942.sh`, i.e. default ON and already in these objects. Not a missed win.

### The KV slab depth is a closed axis, and a trap

`FA_MLA_PF2_BKV` is the kernel's one tiling constant and is now a build axis. Deepening it
should pay: the softmax bookkeeping is charged per (query, KV tile), so a deeper slab pays it
proportionally fewer times. It does not work here, for two independent reasons.

**BKV must be a multiple of 16** — it is the MFMA contraction length of the PV pass. A
non-multiple compiles, runs, and is silently wrong. BKV=40 measured:

| depth | TTFT at 32k | vs shipped |
|---|---:|---:|
| 32 (shipped) | 12 153 ms | — |
| 36 | 11 481 ms | −5.5% |
| 40 | 10 903 ms | **−10.3%** |

That −10.3% is **not real**. The character-identity gate on an 8k-token prompt came back
DIFFERENT: the answer described the input as garbled and the thinking block vanished entirely.
It was fast because it was dropping KV rows in the ragged tail. The monotonic "improvement" with
depth was monotonically more dropped work. A `static_assert` now fails the build on a
non-multiple rather than letting the next person rediscover this the same way.

**The next legal depth does not fit.** BKV=48 needs `48*584 + 48 + 4*16*48` halves = 62,304 B
against a flash arena of about 58 KB, and the existing `static_assert` in `interp.hip` catches
it. So 32 is the only usable depth, which is why it was hardcoded. Going deeper means shrinking
`KSTR` — splitting the 576-wide D of MLA's absorbed form — which is a restructure, not a
constant.

### Where a real win has to come from

The softmax, and not as a guard. The file already records that a lazy corr-rescale was tried
twice and rejected: it measured logits-different AND slower (spill 248 → 293, because a
divergent per-row branch costs more scheduling room at occupancy 1 than the skipped multiplies
are worth). Its own conclusion is the right one — *"a rescale saving here must come from a
deferred-rescale restructure, not a guard"* — i.e. accumulate against a per-tile scale and apply
the correction once at the end, rather than rescaling a 512-wide accumulator every time the
running max moves. That is a genuine kernel rewrite with real numerics risk, and it is the
remaining prize on this kernel: 30% of TTFT at 32k, rising with context.

**§7 is that rewrite, measured.** It is worth −7.9% of TTFT at 32k and −14.1% of the fitted
attention term, and the numerics risk was real — it cost one 10x error regression on the way,
found by a unit test that had to be written first.

## 7. The deferred-rescale restructure: what it bought

§6 ends by naming the one thing left on this kernel — *"a rescale saving here must come from a
deferred-rescale restructure, not a guard."* This is that restructure. It ships as
`FA_MLA_PF2_DEFER` (default on for CDNA3), plus `FA_MLA_PF2_FASTBF`, a conversion fix that only
becomes visible once the rescale is gone. Both are `FA_MLA_PF2_*` build axes in
`scripts/build_gfx942.sh`; `FA_MLA_PF2_DEFER=0` restores the shipped form.

Measured 2026-09-07 on the same host and the same `build-glm53/tp4-long` blob as §1, through
`plowrt bench --prefill-sweep` (3 timed reps, 1 warm-up, cold prompts, `PLOW_TP_NO_AUDIT=1`).
**Only the four `interp_flash*.elf` differ between arms** — every other object, and the blob, is
byte-identical across all three — so nothing but the MLA prefill kernel body is in the
comparison. Two TP4 leases, arms interleaved inside each.

### 7.1 TTFT

| T | shipped | frame | frame + fast bf16 | frame vs shipped | both vs shipped |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 904 ms | 886 ms | **882 ms** | −2.0% | **−2.5%** |
| 8 192 | 1 881 ms | 1 820 ms | **1 810 ms** | −3.3% | **−3.8%** |
| 16 384 | 4 611 ms | 4 386 ms | **4 353 ms** | −4.9% | **−5.6%** |
| 32 000 | 12 235 ms | 11 346 ms | **11 263 ms** | −7.3% | **−7.9%** |

Four rounds each for the first two arms, two for the third; round-to-round spread inside an arm
is 0.16–0.86%, and every frame round is below every shipped round at every length. **−972 ms of
a 12.2 s TTFT at 32k**, and the win grows with context, which is the signature of a change inside
the attention kernel rather than around it. Fitting `TTFT = a·T + b·T²` on the 8192 and 32000
points of each arm:

| arm | a (linear, ms/token) | b (attention, ms/token²) | attention @32k | implied rate | % of bf16 peak |
|---|---:|---:|---:|---:|---:|
| shipped | 0.17713 | 6.413e-6 | 6 567 ms | 212 TFLOP/s | 16.2% |
| frame | 0.17660 | 5.561e-6 | 5 694 ms | 244 TFLOP/s | 18.7% |
| frame + fast bf16 | 0.17579 | 5.506e-6 | 5 638 ms | **247 TFLOP/s** | **18.9%** |

**The quadratic term falls 14.1% and the linear term moves 0.8%.** That split is the result: the
change is confined to the term §1 attributes to prefill attention, and it does not touch the
MoE/GEMM stream. Against §5's roofline the kernel goes from 5.8x off its compute roof to 5.1x.

### 7.2 What changes in the kernel

The shipped online softmax charges, per (query row, KV tile): a quarter-wave max reduce (4
shuffles), a quarter-wave sum reduce (4 shuffles), a correction `exp`, and `oacc[t][i] *= corr`
over all `NT = 32` output tiles — **128 `v_mul` per lane per tile**, paid whether the running max
moved or not. At occupancy 1 there is no co-resident wave to hide any of it.

The flash identity is `O_n = exp2(−m_n) · Σ_j exp2(s_j)·V_j`, so the running max is not
privileged: **any** per-row constant serves as the exponent origin, provided `(m, l, O)` are all
reported in it. They are — the epilogue writes `m := f` and `l := Σ exp2(s_j − f)`, exactly the
pair `d_flash_merge` / `d_mla_merge_fold` already consume, so nothing downstream moves. So the
accumulator stops tracking the max and lives in a **frame** `f`, an integer upper bound on the
row's scores, accumulating `Σ exp2(s_j − f)·V_j` directly. The frame is re-taken only when a
lane's score would push `p` past 1, and the new frame is `ceil(rmax) + 8`:

- `p ≤ 2⁻⁸` always, so `l` and the accumulator can never overflow — a *stronger* bound than the
  running-max form, where `p == 1` is reachable.
- A re-take needs a score 8 log2-units (a factor of 256) above the current row max, so it is a
  record-break by a large margin rather than by any margin: a handful over the ~1000 KV tiles of
  a 32k row, against 1000 unconditional rescales before.
- Both frames are integers, so `corr = exp2(f_old − f_new)` is an exact power of two and the
  re-take is **lossless** — unlike the shipped form, where every rescale multiplies the whole
  512-wide accumulator by a rounded `corr`.
- The uniform `2⁻⁸` on P costs no precision: P is stored bf16, and scaling a bf16 by a power of
  two only moves its exponent.

The `l` reduce leaves the loop with it: each lane sums its own KV columns in the frame and the
quarter-wave reduce happens **once**, in the epilogue — valid because inside a frame there is no
per-tile rescale to interleave, and a re-take scales the partial by the same
quarter-wave-uniform `corr`.

Why this is not the guard the file rejected twice: the guard skipped an operation whose result
was already the identity, so it could only save the tiles where the max did not move, and it paid
a divergent per-row branch on every tile to find out. The frame changes *what is accumulated*, so
the rescale does not exist except at a re-take. `interp_flash` goes VALU 5735 → 5465 static with
**spill 0 → 0** and VGPR 512 / AGPR 256 / LDS 58,368 unchanged; the 128 rescale multiplies move
out of the straight-line path into a conditional block.

### 7.3 The trap: `l` and `O` must see the same P

The first working version measured **10x worse against the decode oracle** than the shipped body
while being perfectly "correct". The cause is specific enough to be worth writing down.

`O` is accumulated by the PV MFMA from a **bf16** P; `l` sums the **f32** `p`. With a running max
the dominant `p` is exactly `1.0`, which is exactly representable in bf16, so that mismatch
cancels out of `O/l` and only the small terms carry it. Against a frame the dominant `p` is a
general value, its bf16 rounding is a full `2⁻⁹`, and it lands straight on the output. The fix is
to convert once and let `l` sum the **same rounded value** the MFMA consumes, which also moves
the P-strip write into the softmax loop (no second pass, and the fp8 arm reuses the dequant scale
its score already loaded).

| `mla_test` V2 vs decode oracle, max rel err across the case grid | shipped | frame, f32 `l` | frame, matched `l` |
|---|---:|---:|---:|
| bf16 dense, 12 shapes | 5.9e-6 – 3.3e-4 | 2.0e-3 – 3.5e-3 | 3.1e-5 – 1.4e-3 |
| fp8 latent, 4 shapes | 2.7e-3 – 3.3e-3 | — | 2.6e-3 – 3.3e-3 |

Tolerance is 2e-2; all 16 shapes PASS. The fp8 arm already carried this error before the change,
for the same reason — its P carries a per-KV dequant scale, so its dominant `p·c` was never 1.0
either — and it is unmoved by the frame.

**`mla_test` had no coverage of the bf16 V2 body at all**: the "dense" and "tiled MFMA" rows
exercise `d_flash_mla_prefill` and `d_flash_mla_prefill_mfma`, and the only V2 row was the fp8
arm. `mla_flash_prefill_v2_512` and its oracle comparison are new with this work, and they are
what caught this. A content gate would only have said "different".

### 7.4 The second increment: a NaN-free P conversion

With the rescale gone, the largest remaining per-tile VALU item is the P conversion itself. The
shared `f2bf` carries a NaN/Inf guard that gfx942 lowers to an exec-mask save/branch/restore, and
this kernel runs it 8 times per KV tile per lane. Under the frame that guard is **provably dead**
— `p = exp2(sv − f)` with `f ≥ sv` lies in `[0, 1]`, and in `[0, cs]` after the fp8 arm's finite
positive scale — so `FA_MLA_PF2_FASTBF` uses `f2bf`'s RNE half alone. Worth another −1.0% at 32k
on top of the frame, and value-identical over that domain: the `mla_test` error columns are
bit-for-bit the same with it on and off, and all 12 served completions below are
character-identical between the two arms.

This is **not** the branchless-`f2bf`-everywhere experiment `amd_common.h` refutes: that one kept
the guard as a select and paid its arithmetic on every conversion in the program.

### 7.5 Qualification — and why character identity is the wrong gate here

`PLOW_MLA_PF_SV` and `PLOW_MLA_FOLD_TB` were adopted on character-identical answers because they
are **bit-identical by construction**. This one is not: the exp arguments move by the frame
offset, so a greedy continuation eventually diverges on a near-tie and **a character-identity
gate cannot pass, by construction**. Reporting it as passed would be false; the honest thing is
to state the result and put a gate underneath it that measures what actually matters.

Six free-form long prompts (3 corpora × {7.8k, 30.2k} tokens, greedy, 96 tokens each) and six
needle-in-a-haystack retrievals (the same corpora with a distinctive code planted at 15%, 50% and
90% depth, "reply with just the code"), all through the served OpenAI endpoint at temperature 0:

| comparison | free-form (6) | needle (6) | needle retrieval |
|---|---|---|---|
| **shipped, lease A vs lease B** | 6/6 **identical** | — | — |
| **frame vs frame + fast bf16** | 6/6 **identical** | 6/6 **identical** | 6/6 vs 6/6 |
| shipped vs frame | 0/6 identical | 0/6 identical | 6/6 vs 6/6 |
| shipped vs frame + fast bf16 | 0/6 identical | 0/6 identical | 6/6 vs 6/6 |

Read the first two rows first. **The instrument works**: the shipped object reproduces
character-for-character across two independent leases and process restarts, so a difference in
row 3 is the kernel and not the harness — and the fast-bf16 arm, which is value-identical by
argument, is character-identical in fact on all twelve prompts, which is that argument confirmed
end to end.

Rows 3 and 4 are the honest cost, and the divergences are all of one kind — "…in this
document" vs "…in a document", "GPU/H100" vs "GPU (H100)", one paraphrase of the same
sentence against another, always mid-generation, both arms discussing the same content of the
same document. That is the signature of a near-tie argmax flip under a perturbation of order
1e-4, not of a damaged attention walk. For contrast, §6's BKV=40 — which was dropping KV rows —
answered that the prompt was garbled and lost its thinking block outright, and it would fail the
needle column here.

**All three arms retrieve the planted code exactly, at 8k and at 32k, at all three depths**
(`74-BRAVO-1908`, 6/6 each). Needle retrieval at 32k is the direct test for the failure mode this
kernel can actually have, and it also exercises the shipped `PLOW_GLM_PF_NS=2` causal KV-split
path end to end, which `mla_test` does not reach.

### 7.6 What is still on this kernel, and what BKV really costs

The frame removes the softmax's per-tile bookkeeping, and with it most of the reason to want a
deeper KV slab: BKV's argument in §6 was that a deeper slab amortizes the per-(query, tile)
softmax over more KV, and that term is now small. What is left in the inner loop, per lane per
KV tile, is **256 `ds_read_u16` for the PV transpose** against 68 MFMA and roughly 130 VALU. That
is now the largest single instruction group in the kernel by a factor of two, every byte of it
non-redundant (64 lanes × 256 halves is exactly the 32 KiB slab), and it is 2-byte-granular only
because the MFMA B-fragment contracts over the *minor* axis of a `[kv][d]` slab. `ds_read_b64_tr_b16`
solves it in one instruction and is gfx950-only (`PLOW_MLA_PF_TR16`); on gfx942 the alternatives
are a second, transposed LDS copy of V (32 KiB more, which does not fit the 58 KiB arena) or a
cross-lane register transpose (~32 permutes to replace 8 LDS reads — not a trade).

And BKV is now asserted at exactly 32, not merely at a multiple of 16. The score tile is fixed at
two 16-wide MFMA n-subtiles — `sacc[2]`, the `kv0 + nt*16 + fr` column map, the `Pw[fr*BKV + kg*8]`
A-fragment and the `kg*8 + j` V rows all walk exactly 32 KV columns per pass — so **a slab deeper
than 32 is staged and then never scored past row 31**. That is the complete explanation of the
BKV=40 result in §6: not a ragged-tail bug, 8 of every 40 KV rows silently deleted. Deepening the
slab is a third change stacked on the LDS one (more score subtiles, a wider P strip), not a
constant.


---

## GLM-5.3 on MI300X: are plow's designed collectives actually paying?

Static packet analysis of the emitted device blobs — `plowrt disasm`, no GPU, no driver.
Blobs: `build-glm53/tp4m` and `build-glm53/tp8m` (GLM-5.3-FP8, gfx942, 304 CU, max-ctx 10240,
decode rungs 1/2/4, measured GEMM tiles). Dispatch width is the `b=` field, i.e. how many of
the 304 CUs that packet occupies.

## Answer in one line

**Prefill: yes.** **Decode: the collective is cheap and is not the bottleneck.**
**But none of the designed collective FOLDS were armed, and arming them removes 156 full-width
prefill packets (−7.8% of CU-weighted work) and 78 decode packet boundaries per token.**

## 1. What the emitted stream actually contains

| program | packets | mean dispatch | collective packets | collective share of count | of CU-weighted work |
|---|---:|---:|---:|---:|---:|
| TP4 decode T=1 | 2523 | 66 / 304 CU | 157 | 6.2% | 1.13% |
| TP8 decode T=1 | 2523 | 61 / 304 CU | 157 | 6.2% | 1.21% |
| TP4 prefill T=2048 | 2246 | 269 / 304 CU | 157 | 7.0% | 7.84% |
| TP8 prefill T=2048 | 2246 | 269 / 304 CU | 157 | 7.0% | 7.84% |

Two collectives per layer (attention seam + MoE seam), 78 layers, so 156 per program, plus one
`XArgmaxFin` for the vocab-parallel lm_head.

**Prefill uses `XReduceTwoShot` at full chip width.** `n=12582912` elements — 2048 tokens x 6144
hidden — dispatched `b=304`, `gate_ag=1`. That is plow's two-shot all-gather doing exactly what
it was designed for, and 7.8% of CU-weighted work for 156 seams of 25 MB each is proportionate.

**Decode uses the narrow `XReduce`.** `H=6144`, so the payload is one token's hidden state, ~24 KB,
dispatched on `b=12` — 3.9% of the chip. It is 6.2% of the packet count but 1.1% of CU-weighted
work. The decode collective is a LATENCY cost (156 serialization points per token), not a
bandwidth cost. Its width does not grow with rank count: `b=12` at both n_gpu=4 and n_gpu=8, and
its weighted share only moves 1.13% -> 1.21%.

## 2. The designed folds were NOT armed — this is the finding

`XReduceAddNorm` appeared **zero** times. In both programs the seam's residual/norm was a
SEPARATE packet immediately consuming the collective's output:

```
#10  XReduce         b=12   out<-act.attn      | H=6144 n_gpu=4
#11  AddNorm         b=1    b<-act.attn        | rows=1 feat=6144      <- decode, ONE workgroup
#14  XReduceTwoShot  b=304  out<-act.attn      | n=12582912 n_gpu=4
#15  Residual        b=304  b<-act.attn        | n=12582912            <- prefill, FULL width
```

In prefill that un-folded `Residual` costs 7.8% of CU-weighted work — the same as the entire
collective it follows. `PLOW_GLM_XR_RES` folds it into the two-shot; `GLM_FUSE_XRN` folds the
decode pair into `XReduceAddNorm`. Neither was set in the emit; both are in this tree.

Re-emitting with `PLOW_GLM_XR_RES=1 GLM_FUSE_XRN=1` (`build-glm53/tp4f`), measured by the same
static census:

| program | packets | change | CU-weighted work |
|---|---:|---:|---:|
| decode T=1 | 2523 -> 2445 | −78 (78 XReduce+AddNorm pairs -> 78 XReduceAddNorm) | −0.6% |
| prefill T=2048 | 2246 -> 2090 | −156 (every Residual folded away) | **−7.8%** |

The decode win is not the 0.6% of work; it is 78 fewer packet boundaries per token, on a chain
where the GLM-5.2 campaign priced the boundary at ~2.13 us each.

### And it measures — the folds are the best plow arm on this model

Served end to end at TP4 on the same four cards, all arms minutes apart, 8 prompts per cell,
128 output tokens. `tp4f` is the folded blob; `tp4m` is the same blob without the folds; `plow`
is the original analytical-tile baseline.

| cell | metric | plow | tp4m | **tp4f** | tp4f vs plow |
|---|---|---:|---:|---:|---:|
| 1024 / c1 | tok/s | 23.78 | 23.97 | **24.20** | +1.8% |
| 1024 / c1 | TPOT ms | 38.70 | 38.59 | **38.21** | −1.3% |
| 1024 / c4 | tok/s | 38.93 | 39.23 | **39.94** | +2.6% |
| 1024 / c4 | TPOT ms | 95.06 | 94.75 | **93.62** | −1.5% |
| 4096 / c1 | tok/s | 18.31 | 17.66 | **19.22** | **+5.0%** |
| 4096 / c1 | TPOT ms | 47.02 | 49.84 | **43.69** | **−7.1%** |
| 4096 / c4 | tok/s | 29.48 | 29.05 | **29.59** | +0.4% |
| 4096 / c4 | TPOT ms | 122.23 | 123.65 | 122.24 | 0.0% |

The folded arm is at least as good as the baseline in every cell and clearly better at 4096/c1.
The win is on the DECODE axis, which is what the packet census predicted: the fold removes
packet boundaries, and 78 of them per token is worth about 0.17 ms against a 38-47 ms TPOT
before any second-order effect.

**TTFT did not move** (383 -> 385, 1024 -> 1030, 1698 -> 1692). That is a real negative result
about the prefill half: folding `Residual` into `XReduceTwoShot` removes 7.8% of CU-weighted
prefill WORK and buys no TTFT, so prefill at these lengths is not CU-throughput bound. Do not
spend more effort on prefill packet-count reduction on this evidence.

**Not yet adopted.** `PLOW_GLM_XR_RES` is recorded byte-identical and `GLM_FUSE_XRN` requires
`fuse_b1` + tp>1 (both hold here), but that record is GLM-5.2's. On GLM-5.3 these need the
paired accuracy gate before the deltas above can be called wins.

## 3. Collectives are not where decode is losing

The same census says where decode actually spends the chip:

| | TP4 | TP8 |
|---|---:|---:|
| Gemv | 30.3% | 32.7% |
| GemvQkv | 27.7% | 29.3% |
| MoE expert walks (Glu+Down, block-fp8) | 23.0% | 24.8% |
| FlashMlaDecode | 7.5% | 4.0% |
| MlaMergeFold | 6.0% | 3.2% |
| **collectives** | **1.13%** | **1.21%** |

**75.3% of decode packets dispatch on 32 CUs or fewer**, mean width 66 of 304. 155 of them are
`AddNorm` on a SINGLE workgroup. Projections plus expert walks are 58% (TP4) / 62% (TP8) of
CU-weighted work, and the emit's own lean oracle puts the TP4 decode step at 16.0 GB touched
with a 3.0 ms HBM-bandwidth floor against 38.7 ms measured.

So decode is weight-streaming and packet-boundary bound, not collective bound. Going after the
collectives to fix decode TPOT would be attacking 1% of the chip. The levers that match the
evidence are the ones that cut the weight stream (`GLM_LINEAR_FP8`, already prepped here as
`GLM-5.3-plow-q`) or the packet count (the folds above, and the narrow-op problem).

## Reproducing

```bash
plowrt disasm build-glm53/tp4m --program 1      # decode
plowrt disasm build-glm53/tp4m --program 2048   # one prefill bucket
```


---

## GLM-5.3 TP4 on MI300X: decode KV format and split policy

Measured 2026-09-07 on 8x MI300X (gfx942), ROCm 7.14 from the flake, through the OpenAI
HTTP surface (`plowrt serve` + `scripts/bench_packed_serve.py`), concurrency 1, output 128,
`PLOW_TP_NO_AUDIT=1`, one warmup per cell and five timed repeats. Every arm holds a 4-GPU
`gpulease`.

The premise, from `attention_roofline.md` §2/§3/§10: every decode-attention cell in the GLM
grid is memory-bound by 10x or more, so decode does not need a faster attention kernel, it
needs fewer bytes. This note tests the three byte/policy levers that already have opcodes or
fields, on the machine, end to end.

## 0. What every number here is relative to — the tile store is mid-campaign

`plowc` keys its dense-GEMM tile measurements to the PREPROCESSED interpreter digest. At this
branch that digest is `gfx942-cae47801184a4196`, and the committed store
(`tuning/amd/gfx942/mi300x/kernel_measurement.jsonl`, 1588 records) holds **zero** records for
it — the six digests it does hold are all stale. The main checkout's working copy is being
filled right now by the `gemma31-gfx942-mi300x-tile` campaign (120 records at the start of
this session, 360 an hour later), and those are Gemma shapes.

**Consequence, stated once so no table below has to repeat it:** every GLM blob emitted from
this branch today reports `tile_source: analytical`, `tile_measured: 0` of 2472 lookups. The
shipped `build-glm53/tp4-long` blob was emitted BEFORE the digest moved and carries 2472
measured tiles, so it is NOT comparable cell-for-cell with the blobs emitted here. Each
experiment below is therefore run against its OWN control, emitted in the same minute with
the same knobs, and the two arms of a pair always share the tile source. Cross-experiment
absolute TPOT should not be compared.

This is not a general "the store is empty" state and it should not be reported as one: a
Gemma-4 31B bundle emitted an hour into this session (`build-gemma31/assets-glm53`) reports
`tile_source: measured`. The running campaign is filling GEMMA shapes at the current digest,
so Gemma blobs are measured and GLM blobs are not. **The GLM tile campaign needs a re-run
against `gfx942-cae47801184a4196` before any absolute GLM TPOT number from this branch is
quotable.**

## 1. Live-`kv_len` MLA split policy — implemented, verified live, and a REGRESSION

`devgen::mla::glm_nsplit` sizes the flash-decode KV-split count from the emit-time `max_ctx`,
so one blob runs one split count at every live context. The `tp4-long` blob (max_ctx 32768,
TP4, nh_l=16) bakes `ns=64` and runs it at live 1024, where that function's own header calls
the gap "the largest remaining item in this knob" on the strength of a GLM-5.2 / gfx950
ladder that puts the per-layer chain optimum at 16 there (61.1 µs against 81.1, +33%).

`PLOW_MLA_NS_LIVE=1` (this branch) re-points the flash's and the merge's `i[4]` at the live
`kv_len` whenever the policy value moves. Same blob, same objects, same binary, one env var.

### 1.1 The patch fires — and that had to be proven

A first pass of this change wrote `i[4]` only into `progs[self.decode]`, which is always the
WIDEST decode rung. `tp4-long` has two rungs (2445 and 1554 instructions), the mux drops to
rung 1 at concurrency 1, and `decode_step_batched_at(.., dp)` dispatches that named rung — so
every measured request ran the unpatched program. It produced a textbook clean null: ±0.7% on
TPOT and 20/20 character-identical output. **A silent no-op is indistinguishable from a null
lever**, and the only thing that separated them was making the change observable. The engine
now logs at load and at every move:

    PLOW_MLA_NS_LIVE: MLA decode split count tracks the live kv_len rungs=2 baked=64 sites=312
    MLA split count re-pointed at the live kv_len baked=64 from=64 to=16 kvlen=1025
    MLA split count re-pointed at the live kv_len baked=64 from=16 to=32 kvlen=8193
    MLA split count re-pointed at the live kv_len baked=64 from=32 to=64 kvlen=16385

312 sites = 2 rungs × (78 flash + 78 merge), and 84 re-points over the whole arm — a step
function of `kv_len`, not a per-token cost.

### 1.2 Measured: the ladder is wrong for this chip

`tp4-long`, TP4, concurrency 1, output 128, five timed repeats after a warmup, both arms back
to back inside one lease.

| live ctx | baked ns | live ns | TPOT p50 OFF | TPOT p50 ON | Δ TPOT | Δ TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 024 | 64 | **16** | 35.756 ms | 37.371 ms | **+4.52%** | +0.06% |
| 4 096 | 64 | **16** | 37.976 ms | 38.101 ms | +0.33% | +0.29% |
| 8 192 | 64 | **32** | 38.289 ms | 38.349 ms | +0.16% | +0.05% |
| 16 384 | 64 | 64 | 38.967 ms | 39.050 ms | +0.21% | +0.13% |
| 30 000 | 64 | 64 | 39.927 ms | 40.016 ms | +0.22% | +0.05% |

The last two rows are the negative control: at those lengths the live rule returns the baked
64 and the packet is unchanged, so their +0.21% / +0.22% is the floor — a small systematic
bias from arm order, not a signal. Against it:

* **1024 is a +4.3% REGRESSION.** Dropping the split count from 64 to 16 at short context
  makes decode measurably WORSE on gfx942, which is the opposite of what the gfx950 ladder
  predicts.
* 4096 and 8192 are inside the floor.

**The gfx950 GLM-5.2 ladder does not transfer to gfx942 GLM-5.3 TP4.** On this chip more
splits win at short context: the flash's latent-stream parallelism is worth more than the
`O(nsplit)` merge costs, and the emit-time convention — size for `max_ctx`, i.e. take the
LARGEST split the blob will ever want — happens to be the right policy here rather than a
compromise. The `glm_nsplit` header's "largest remaining item in this knob" is, on gfx942, a
4.3% regression waiting to be shipped.

### 1.3 Numerics: NOT output-preserving

| ctx | live ns vs baked | cells | character-identical | median first-divergence char |
|---:|---|---:|---:|---:|
| 1 024 | 16 vs 64 | 4 | **0/4** | 108 |
| 4 096 | 16 vs 64 | 4 | **0/4** | 78 |
| 8 192 | 32 vs 64 | 4 | **0/4** | 54 |
| 16 384 | 63 vs 64 | 4 | **0/4** | 131 |
| 30 000 | 64 vs 64 | 4 | **4/4** | — |

**The split policy is not character-identical, and it never could have been.** Each split runs
its own online softmax over its own slice and the partials are merged in split order, so a
different partition reassociates the sum. Every row where the count differs diverges; the one
row where it does not differ is identical, 4/4 — which is the control that says the divergence
is caused by the split and nothing else.

The 16384 row is worth reading carefully because it looks like it should be a control and is
not: the greedy prompts are prose-padded and land at ~16 345 tokens, and `16345 / 256 = 63`,
so that cell runs ns=63 against 64. The bench at 16384 uses an exact-token prompt
(`kv_len = 16385`, `16385 / 256 = 64`) and IS a true control. Same rule, two different answers,
because one input is 39 tokens shorter.

The earlier 20/20-identical result for this change is **withdrawn**: it was produced by the
unpatched-rung bug in §1.1, and identity was the symptom of the no-op.

### 1.4 Recommendation

**Keep `PLOW_MLA_NS_LIVE` off by default, and do not lower `NS_PER`/`NS_FLOOR` for gfx942
either.** The mechanism is correct, cheap (a step function of `kv_len`, ~84 uploads over a
whole arm, no dispatch or counter change) and now observable — but the policy it implements is
a measured 4.3% regression at 1k on this hardware, and it changes the token stream. Keep the
flag as the one-line experiment that makes the emit-time convention testable on a NEW
geometry; do not turn it on for this one.

If someone wants the split policy revisited on gfx942, the missing input is a gfx942 ladder:
sweep `PLOW_MLA_NS` at emit across 16/32/64/128 at live 1k/4k/8k/16k/32k on the full model,
the way the gfx950 table in `glm_nsplit`'s header was produced. This branch's result says the
answer will not look like the gfx950 one.

## 2. fp8 latent KV, end to end

### 2.1 The bytes, from the emitted blobs

`PLOW_GLM_FP8_KV=1` swaps the latent cache to e4m3 with a per-row f32 scale and routes the
dense flash to ops 109/110. `plowrt disasm --program 1` confirms the swap is complete, not
partial — the decode program's 78 `FlashMlaDecode` become 78 `FlashMlaDecodeFp8` and the 78
`RmsNorm` latent writers become 78 `HeadNormRopeFp8` (RMSNorm + quantize + scale record in one
pass), with the 156 q-rope/k-rope `HeadNormRope` untouched:

| arm | latent flash | latent writer | kv_stride |
|---|---|---|---:|
| `tp4-ctl` | `FlashMlaDecode` ×78 | `RmsNorm` ×78 | 32 768 |
| `tp4-fp8` | `FlashMlaDecodeFp8` ×78 | `HeadNormRopeFp8` ×78 | 32 768 |
| `tp4-ctl131k` | `FlashMlaDecode` ×78 | `RmsNorm` ×78 | 131 072 |
| `tp4-fp8131k` | `FlashMlaDecodeFp8` ×78 | `HeadNormRopeFp8` ×78 | 131 072 |

Per position per layer per rank — and the latent is REPLICATED across TP ranks, so this is the
same at TP4 and TP8:

| | ckv (kv_lora 512) | scale | krot (rope 64) | total | per token, 78 layers |
|---|---:|---:|---:|---:|---:|
| bf16 | 1 024 B | — | 128 B | **1 152 B** | **89 856 B** |
| e4m3 + f32 row scale | 512 B | 4 B | 128 B | **644 B** | **50 232 B** |
| | −50% | | 0% | **−44.1%** | **−44.1%** |

The rope half stays bf16 (`i[6]=0`, the shipped K3 form), which is why the saving is 44% and
not 50%. At TP4 with a 181.75 GiB weight slab on a 191.98 GiB card, ~10.2 GiB of KV budget
per rank buys:

| max_ctx | bf16 KV/rank | fp8 KV/rank | fits in ~10.2 GiB? |
|---:|---:|---:|---|
| 32 768 | 2.74 GiB | 1.53 GiB | both |
| 65 536 | 5.48 GiB | 3.06 GiB | both |
| 131 072 | **10.97 GiB** | 6.13 GiB | **fp8 only** |
| 218 000 (fp8 ceiling) | 18.2 GiB | 10.2 GiB | fp8 only |

### 2.2 What the control has to give up

The fp8-latent emitter refuses three knobs the shipped GLM-5.3 recipe carries, so the control
arm drops them too — an arm that merely omits the flag would be comparing fp8-KV against a
different program, not against bf16 KV:

* `PLOW_GLM_FUSE_ROPE` — the q-rope fold needs `t[7]` for the cos table and `i[6]` for the sin
  handle, and the fp8 scale strip already owns `t[7]`. Slot-exclusive, refused at emit.
* `PLOW_GLM_PF_NS > 1` — forced to 1 under fp8 in the prefill emitter.
* the decode BATCH ladder — the fp8 latent writer has no validated batch-ring form, so both
  arms are emitted rung-1 only.

`PLOW_GLM_DSA` and `PLOW_GLM_OFOLD` are also refused alongside fp8 KV; neither is on in the
shipped recipe.

### 2.3 Served numbers — fp8 latent is SLOWER at every measured context

`tp4-ctl` and `tp4-fp8`, TP4, concurrency 1, output 128, five timed repeats after a warmup.
Both blobs were emitted minutes apart with the identical knob set; the only difference is
`PLOW_GLM_FP8_KV`.

| live ctx | TPOT bf16 | TPOT fp8 | Δ TPOT | TTFT bf16 | TTFT fp8 | Δ TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 024 | 37.072 ms | 37.892 ms | **+2.21%** | 250.8 ms | 261.5 ms | **+4.25%** |
| 4 096 | 39.282 ms | 39.884 ms | **+1.53%** | 801.2 ms | 887.6 ms | **+10.79%** |
| 16 384 | 40.250 ms | 40.945 ms | **+1.73%** | 4 250.4 ms | 5 114.4 ms | **+20.33%** |
| 30 000 | 41.217 ms | 42.615 ms | **+3.39%** | 10 287.8 ms | 13 147.3 ms | **+27.80%** |
| 65 536 † | 48.034 ms | 49.761 ms | **+3.60%** | 35 685.9 ms | 49 277.3 ms | **+38.06%** |

† the 65 536 row is `tp4-dense66k` (bf16, max_ctx 66560) against `tp4-fp8131k` (fp8, max_ctx
131072) at the same live length — the 32 768 control cannot reach it and the bf16 131072 blob
does not load (§2.4). max_ctx is shown not to move TPOT in §2.6.

Against a ±0.2% noise floor (§1), every cell is a real regression, and **both regressions grow
monotonically with context**: TPOT +2.2 → +3.6%, TTFT +4.3 → +38.1%. That is the signature of
a cost inside the flash inner loop scaling with the KV rows read, not a fixed overhead.

**This is not a routing accident.** The obvious alternative explanation — that the fp8 arm
falls off the V2 prefill body onto the 8-wave kernel, the way `PLOW_FP8_KV` costs the
cp.async pipeline on NVIDIA without `PLOW_FP8_KV_FASTPF` — does not hold here:
`interp_flash_fp8kv_gq.elf` carries **both** `plow_mla_pf_v2_arm_1` and
`plow_mla_pf_v2_fp8_arm_1`, so op 110 runs the V2 body. The engine log confirms the whole
object set swapped (`variant=Fp8Kv`, `interp_decode_fp8kv_gq.elf`,
`interp_prefill_fp8kv_mla_moe_gq.elf`).

**Why the bytes do not pay, in one line of arithmetic.** Decode attention at 30 000 tokens is
a 0.72 ms/token memory roof out of a 41 ms TPOT — 1.7% of the step. Halving it can return at
most 0.35 ms (0.85%), and only if the kernel were AT its roof; it is 10x off it
(`attention_roofline.md` §2.1), so the achievable return is a fraction of that. The e4m3
dequant is per element on both phases and is not free. The measured +1.4 ms/token at 30 000
is that trade landing on the wrong side. The same conclusion in the roofline's own terms: fp8
KV halves a roof that is not the binding constraint at any context this blob can serve.

`PLOW_GLM_FP8_KV` is therefore **not a TPOT lever on GLM-5.3 at TP4/gfx942**. It is a
capacity lever, and §2.4 is where it pays.

**What the measurement does and does not attribute.** `PLOW_FP8_KV` is an OBJECT SWAP, not an
extra instruction: the whole decode and prefill megakernel is rebuilt
(`interp_decode_fp8kv_gq.elf` for `interp_decode_fp8_gq.elf`), so the regression is
attributable to "the fp8-KV object", and the dequant is the leading but not the only candidate
inside it — a different register or occupancy profile in a 1.4 MB interpreter would land on
every op, not just the flash. Two observations narrow it without settling it: the regression
scales with context (§2.6), which a flat occupancy penalty would not; and it appears on
prefill more strongly than on decode, which is where the per-element dequant work is largest.
Separating the two would need a per-op trace, which this session did not run.

### 2.4 Capacity: fp8 serves a context bf16 cannot load

Both blobs emitted at `max_ctx 131072`, rung 1, same knobs, same 4 cards:

| arm | latent | KV/rank at 131 072 | load result |
|---|---|---:|---|
| `tp4-ctl131k` | bf16 | 10.97 GiB | **REFUSED** — `hsa_amd_memory_pool_allocate(2415919104)` → `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (4104) |
| `tp4-fp8131k` | e4m3 + f32 row scale | 6.13 GiB | **serves** — `AMD engine ready … variant=Fp8Kv max_ctx=131072` |

The weight slab is 181.75 GiB of a 191.98 GiB card, so the KV budget is ~10.2 GiB per rank
and 10.97 GiB does not fit. At 644 B per position per layer the same budget reaches
**~218 000 tokens**, 1.79x the bf16 ceiling — which is the whole return on this flag.

### 2.5 Numerics: 0/16 identical, divergence at 6–18% of the answer

Greedy streams, 4 prompts × 4 contexts × 128 tokens, temperature 0, `ignore_eos`:

| ctx | cells | character-identical | median first-divergence char | as a fraction of the answer |
|---:|---:|---:|---:|---:|
| 1 024 | 4 | **0/4** | 36 | 0.06 |
| 4 096 | 4 | **0/4** | 51 | 0.08 |
| 16 384 | 4 | **0/4** | 81 | 0.12 |
| 30 000 | 4 | **0/4** | 107 | 0.18 |

**Every cell diverges, and it diverges early** — around generated token 10 at 1k and token 25
at 30k, consistent with the "greedy diverges after ~21 tokens" already recorded for the dense
fp8-KV path in `flags-reference.md`. That is expected (e4m3 has three mantissa bits) and it is
the number to quote, not a pass/fail: fp8 latent KV produces a DIFFERENT completion for every
prompt at every context, not an occasionally-different one.

Divergence position rises with context (0.06 → 0.18 of the answer), which is the opposite of
what a context-dependent DEGRADATION would look like and is explained by the answers getting
longer, not by the arm getting more accurate.

### 2.6 The context SLOPE is worse, so the gap widens — the roofline's prediction inverted

`tp4-fp8131k` reaches contexts nothing else here can, and its TPOT is linear in context:

| live ctx | TPOT (fp8, max_ctx 131 072) | TTFT |
|---:|---:|---:|
| 30 000 | 42.517 ms | 13.1 s |
| 65 536 | 49.480 ms | 49.0 s |
| 120 000 | 60.345 ms | 150.2 s |

Fitted slope **0.197 ms of TPOT per 1 000 tokens of context**, flat across both intervals
(0.196 and 0.199). The bf16 control's slope over 1 024 → 30 000 is **0.143 ms/1k**
(37.072 → 41.217).

So fp8 latent KV makes the per-context TPOT slope **35–38% worse** (0.197 against 0.143 for
`tp4-ctl` over 1k → 30k and 0.146 for `tp4-dense66k` over 4k → 65.5k), and the two arms therefore
diverge FURTHER as context grows — exactly backwards from the byte argument, and it is the
same fact as §2.3 seen from a different angle: the e4m3 dequant is per element per KV row, so
its cost scales with the rows read, which is the same axis the byte saving scales on. On a
kernel sitting an order of magnitude off its bandwidth roof, the dequant term is larger than
the bandwidth term, and the difference grows with the row count. Extrapolating the bf16 slope
to 120 000 gives ~54.1 ms against fp8's measured 60.345 — **fp8 would be ~11% slower at 120k
if bf16 could run there at all**, which §2.4 shows it cannot.

The cross-check that the blob's `max_ctx` is not itself a variable: fp8 at live 30 000
measures 42.615 ms on the 32 768 blob and 42.517 ms on the 131 072 blob, 0.2% apart — inside
the noise floor. Nothing in these tables is a max_ctx artefact.

### 2.7 Quality: no retrieval regression up to 20.6k

Divergence is not degradation, and fp8 KV's documented failure mode is retrieval degrading
with context (`flags-reference.md`: at 7.8k every arm finds the needle, at 66.9k only bf16
does). `scripts/glm53_needle_probe.py` plants three unique facts at three depths (0.1 / 0.5 /
0.9 of the filler) across three context sizes and asks for each through `/v1/completions`:

| arm | needles retrieved | contexts (achieved prompt tokens) |
|---|---:|---|
| `tp4-ctl` (bf16 latent) | **27/27** | 2 771 / 11 203 / 20 565 |
| `tp4-fp8` (e4m3 latent) | **27/27** | 2 771 / 11 203 / 20 565 |

**Paired, no regression.** State the limit with it: the probe's prose filler is denser than
estimated, so the top cell landed at 20.6k rather than 30k, and **the context at which the
NVIDIA dense fp8-KV path was recorded losing the needle (66.9k) was not reached here.** A
deployment turning this flag on for the capacity in §2.4 — i.e. to serve past 122k — is
turning it on for exactly the regime this probe did not cover, and should re-run it there.

## 3. DSA arming threshold

### 3.1 The gate cannot express the threshold it names

`GlmCfg::dsa(ctx)` reads:

    const CROSSOVER: u32 = 65536;
    self.has_dsa && ctx > CROSSOVER && PLOW_GLM_DSA != "0"

and `ctx` there is the **emit-time `max_ctx`**, not the live length — the same
emit-time/live confusion as `glm_nsplit` one function below it, and with a much larger
blast radius. The gate produces ONE static decode program, so:

* a blob emitted at `max_ctx <= 65536` runs dense at **every** context, including 128k
  if it could reach it;
* a blob emitted at `max_ctx > 65536` runs the gather at **every** context, including
  1024, where the roofline puts the gather at 0.94x of dense on bytes alone
  (`attention_roofline.md` §4) and the measured selector/gather fixed cost makes it far
  worse.

So "the crossover is 69k" is not what the shipped emitter implements, and lowering the
constant would not implement a lower crossover either — it would move the whole blob to
the gather at every length. **The DSA crossover is a runtime routing decision that the
emitter has no way to express.** That is the finding, independent of where the crossover
actually sits.

Making it a runtime decision is NOT the same one-field patch `PLOW_MLA_NS_LIVE` is:

* the arms are different OPCODES (`FlashGatherDecode` 54 vs `FlashMlaDecode` 50), so the
  switch is an `op` rewrite, not an immediate;
* the operand MEANINGS differ in the same slots — op 54 puts the idx table in `t[7]` and
  the selection width in `i[6]`, and on op 50 a non-`NONE` `t[7]` is the q-rope-fold
  discriminator (`t[7]` = cos table, `i[6]` = sin handle). Flipping the opcode without
  clearing both slots makes the dense kernel read the index table as a trig table;
* the sparse arm carries 42 extra instructions (21 `IndexScore` + 21 `IndexSelect`) that
  a dense step must not run, so they would have to be `Nop`ed and their counters still
  satisfied.

Feasible, but it is a routing feature with a counter contract, not a field patch.

### 3.2 The DSA arm does not load on GLM-5.3 at all — the indexer weights are block-fp8

Both arms were emitted at `max_ctx 66560`, one block above the gate's 65536, so `tp4-dsa`
gathers at every live length and `tp4-dense66k` is the identical blob with `PLOW_GLM_DSA=0`.
`plowc` emitted the sparse blob without complaint — 78 `FlashGatherDecode`, 21 `IndexScore`,
21 `IndexSelect`, `nsplit=16` — and `plowrt` refused it at load:

    Error: Device("shard model.layers.0.self_attn.indexer.wq_b.weight: replicated but the
    checkpoint has 8388608 B and the blob declares 16777216 B")

That is not a sizing bug. Reading the safetensors headers of BOTH the prepped
(`GLM-5.3-plow-lite`) and the raw (`GLM-5.3-FP8`) checkpoints — they agree exactly:

| indexer tensor | checkpoint dtype | shape | bytes | scale grid |
|---|---|---|---:|---|
| `indexer.wq_b.weight` | **F8_E4M3** | [4096, 2048] | 8 388 608 | `weight_scale_inv` [32, 16] |
| `indexer.wk.weight` | **F8_E4M3** | [128, 6144] | 786 432 | `weight_scale_inv` [1, 48] |
| `indexer.weights_proj.weight` | BF16 | [32, 6144] | 393 216 | — |
| `indexer.k_norm.{weight,bias}` | BF16 | [128] | 256 | — |

`devgen` declares `wq_b` at 2 B/element (16 777 216 = 2 × 8 388 608), i.e. bf16. The field
documentation on `GlmLW::iwqb` states that "no shipped checkpoint quantizes any of the three"
— **for GLM-5.3 that is false for two of the three**, and the block-fp8 grids are right there
next to the weights.

So the DSA decode path is **unservable on GLM-5.3 as shipped**, and the missing capability is
a block-fp8 indexer projection (a `GemvFp8Blk`-class arm for `wq_b`/`wk` plus their
`[128,128]`-convention scale grids), not a crossover constant. This is also why every GLM-5.3
blob in the tree carries `PLOW_GLM_DSA=0`: the arm has never been served on this checkpoint.

Two things follow that are worth fixing regardless of whether anyone wants sparse decode:

* **`plowc` emits an unservable blob silently.** The indexer dtype is knowable at emit — the
  checkpoint is right there — so the refusal belongs next to the `has_dsa` gate, with this
  message, rather than 200 GiB of weight upload later in the loader.
* **The `GlmLW::iwqb` doc comment is wrong** and is the reason the emitter believes bf16.

### 3.3 What the dense arm alone still says about the crossover

`tp4-dense66k` (bf16, max_ctx 66560, `PLOW_GLM_DSA=0`) served the full sweep:

| live ctx | TPOT p50 | TTFT p50 |
|---:|---:|---:|
| 4 096 | 39.112 ms | 796.9 ms |
| 8 192 | 39.449 ms | 1 726.5 ms |
| 16 384 | 40.140 ms | 4 263.4 ms |
| 32 768 | 42.038 ms | 11 658.5 ms |
| 60 000 | 46.590 ms | 31 253.8 ms |

Dense decode TPOT is linear in context at **0.134 ms per 1 000 tokens** (4 096 → 60 000),
which reproduces the 0.136 ms/1k the existing `GlmCfg::dsa` crossover note measured on MI350X
almost exactly. That note's other number — gather TPOT flat at ~48.6 ms — is the one this
session could not re-measure, and the two together are what put the crossover at ~69k. Since
the dense slope transfers, there is no evidence here that the 65536 constant is wrong as a
NUMBER; the finding in §3.1 is that it is the wrong KIND of gate, and the finding in §3.2 is
that the arm behind it cannot run on this checkpoint at all.

**Recommendation: leave `PLOW_GLM_DSA` at 0 and do not touch `CROSSOVER`.** The two changes
that would make the question answerable, in order: (1) a block-fp8 indexer projection so the
sparse blob loads, (2) an emit-time refusal so the current combination fails in `plowc` with a
message that names the dtype. Only after (1) is the threshold measurable on this checkpoint,
and only after §3.1's routing work is it expressible.

## 4. Recommended defaults, and what the roofline got right and wrong

| lever | default now | recommended | why |
|---|---|---|---|
| `PLOW_GLM_FP8_KV` | off | **stay off** — turn on ONLY to buy context | +1.5–3.6% TPOT, +4–38% TTFT, and a 35–38% worse per-context TPOT slope. Buys 1.79x max context at TP4 (131 072 loads where bf16 does not). |
| `PLOW_MLA_NS_LIVE` | off (new flag) | **stay off** | +4.3% TPOT at live 1024 on gfx942; the gfx950 ladder does not transfer. Not output-preserving. |
| `PLOW_GLM_DSA` | 0 | **stay 0**, and leave `CROSSOVER` alone | The arm cannot load on GLM-5.3 — block-fp8 indexer weights against a bf16 declaration. |
| `glm_nsplit`'s `NS_PER` / `NS_FLOOR` | 256 / 16 | **unchanged** | Same measurement as `PLOW_MLA_NS_LIVE`: fewer splits is worse here. |

### What the roofline predicted, checked against the machine

The roofline's structural claim is confirmed and its actionable claim is not.

**Confirmed:** decode attention is a small, linear-in-context share of the step. Every dense
arm measured here has a TPOT slope of 0.134–0.143 ms per 1 000 tokens of context
(`tp4-ctl` 0.143, `tp4-dense66k` 0.134), reproducing the 0.136 ms/1k measured on MI350X for
the same model. Decode attention is memory-bound and it is small.

**Not confirmed:** "decode does not need a faster kernel, it needs fewer bytes." Both byte
levers were tested and neither returned time.

* fp8 latent halves the KV row and makes decode **slower**, because the kernel is an order of
  magnitude off the roof the halving applies to, while the dequant it adds scales on the same
  per-row axis. §2.6.
* the DSA gather removes the linear term entirely and **could not be run at all** on this
  checkpoint. §3.2.

The lesson is not that the roofline is wrong — the byte counts in it are right, and §2.1
reproduces them from the emitted blobs. It is that a roof 10x above the measured kernel is not
a budget you can spend: halving it returns a fraction of a fraction, and any per-element cost
paid to halve it is charged in full. **Decode attention on this geometry is not bandwidth
work, it is a small latency term hiding behind the MoE expert walk**, and the two levers that
shrink its bytes both cost more than they save.

The one number that would change this conclusion is the one lever this session could not
reach: whether a gathered decode's fixed cost on MI300X is small enough to beat the linear
term before the context runs out. That is §3.2's blocked work.

## 5. Not attempted: dense-GQA fp8 KV on Gemma-4 31B

The fourth lever in this family — ops 38/39, `PLOW_FP8_KV` on a dense GQA cache — was not
measured, and the reason is scheduling rather than difficulty. The emit path is
arch-generic (`devgen/src/lib.rs:4478`/`:4510` pick `FlashDecodeFp8`/`FlashPrefillFp8` off the
same `fp8_kv` flag for every geometry) and the objects already exist
(`build-gemma31/hsaco-glm53/interp_decode_fp8kv*.elf`), so it is a two-emit, two-serve
experiment. What is missing is the Gemma-4 31B emit recipe: `build-gemma31/assets-glm53`
records no emit environment in its `build.json`, and every Gemma bundle in that tree was
produced by another campaign that held gfx942 leases continuously through this session.
Reconstructing the recipe by inspection risks producing a bundle that is not comparable with
that campaign's own controls, which is worse than not running it.

It is worth running, and the prize is bigger there than here: Gemma-4 31B's decode attention
is **1.5–18.9% of its weight stream** across 1k–128k (`attention_roofline.md` §5.1) against
GLM's 0.5–1.7% at TP4, so the same halving applies to a much larger share. The prediction from
this note is nonetheless a warning: the loss here was not the bytes, it was what the fp8 object
costs per KV row, and that term scales with the same row count on any geometry. The experiment
worth running is the one that measures the two terms separately — bytes saved against
per-row cost — rather than assuming the byte count decides.

## 6. Reproduction

Every arm here ran inside ONE 4-GPU lease per campaign — a lease per arm re-queues behind
every other agent's single-GPU job between arms, which on a busy box is most of the wall
clock.

    # emit, no GPU. Every arm shares the confound set the fp8 emitter forces
    # (no fuse-rope, PF_NS=1, rung 1) so the pair differs in one bit.
    nix develop /app/plow --command ./scripts/glm53_decode_kv.sh emit ctl        # bf16, 32768
    nix develop /app/plow --command ./scripts/glm53_decode_kv.sh emit fp8        # e4m3, 32768
    nix develop /app/plow --command ./scripts/glm53_decode_kv.sh emit ctl131k    # bf16, 131072
    nix develop /app/plow --command ./scripts/glm53_decode_kv.sh emit fp8131k    # e4m3, 131072
    nix develop /app/plow --command ./scripts/glm53_decode_kv.sh emit dense66k   # bf16, 66560
    nix develop /app/plow --command ./scripts/glm53_decode_kv.sh emit dsa        # DSA armed, 66560

    # the whole sweep under one lease: serve, bench, greedy stream, stop, next arm
    ./scripts/glm53_kv_campaign.sh                     # ctl fp8 ctl131k fp8131k nslive0 nslive1 dense66k dsa
    ARMS="ctl fp8" ./scripts/glm53_kv_campaign.sh      # a subset

    # the follow-up pair: needle retrieval, and the matched bf16/fp8 point at 65536
    ./scripts/glm53_needle_campaign.sh

    # one arm by hand
    PLOW_MLA_NS_LIVE=1 ./scripts/glm53_decode_kv.sh serve build-glm53/tp4-ctl 19400
    python3 scripts/bench_packed_serve.py --url http://127.0.0.1:19400 \
        --out ctl.jsonl --label ctl --inputs 1024 4096 16384 30000 \
        --outputs 128 --concurrency 1 --repeats 5 --warmups 1
    python3 scripts/glm53_greedy_probe.py --url http://127.0.0.1:19400 \
        --arm ctl --out greedy-ctl.json --lens 1024,4096,16384,30000
    python3 scripts/glm53_needle_probe.py --url http://127.0.0.1:19400 \
        --arm ctl --out needle-ctl.json --lens 4096,16384,30000
    python3 scripts/glm53_greedy_agree.py --baseline greedy-ctl.json --candidate greedy-fp8.json

Raw results (bench JSONL, greedy and needle JSON, and every serve log including the two load
refusals) are under `build-glm53/kvbench/`, which is gitignored — the tables above are the
record.

### The two instruments, and why there are two

* `glm53_greedy_probe.py` + `glm53_greedy_agree.py` answer **"did the arithmetic change"**:
  raw `/v1/completions` continuations at temperature 0 with `ignore_eos`, compared character
  by character. It uses the completions endpoint deliberately — GLM-5.3 is a reasoning model,
  so a 288-token *chat* cell records an empty `message.content` and every arm agrees trivially
  (this is why `perf-data/probes/facts_gate.py` could not be reused as the identity
  instrument here; it remains the right *quality* instrument for a model whose content channel
  fits the budget).
* `glm53_needle_probe.py` answers **"did the answer get worse"**: planted facts at three
  depths, graded by substring containment, paired between arms.

A divergence number without a retrieval number says nothing about quality, and a retrieval
number without a divergence number hides that the token stream changed at all. §2.5 and §2.7
are the same arm measured both ways.


---

## GLM-5.3 TP4 on MI300X: packed prefill — four gates, and the three that were shut

`PLOW_PACKED_PREFILL_ROUTE` (`crates/plowrt/src/config.rs`, default **false**) loads three lean
operator-family objects — `interp_packed_mla_norm`, `interp_packed_mla_flash`, `interp_packed_kda`
— and lets the mux route a co-packed prefill's segments to them instead of the production
interpreter. This is the record of qualifying it on GLM-5.3 at TP4 on 8x MI300X (gfx942), ROCm
7.14 from the flake.

## 0. Summary

* **It loads.** `scripts/build_gfx942.sh` never built the three family objects; it does now,
  behind `PLOW_PACKED_PREFILL_CONSUMERS=1`, and all six (static + `_gq`) pass every marker,
  ABI and resource gate on a live MI300X.
* **It did not fire, and could not have.** GLM's prefill program has its norm ops in mixed
  segments, and the emit flag that fixes that (`PLOW_EMIT_PACKED_PREFILL`) was wired only into
  the Kimi-K3 emitter — and even there was dead behind a Hopper-only assertion. Both are fixed;
  with a re-emitted blob the route arms on all four rungs.
* **Even armed, it needs a fourth thing nobody documented.** Under the shipped chunk policy
  every prompt up to the widest bucket is a single chunk, and the final chunk is never packable,
  so co-packing is unreachable at *every* prompt length. `PLOW_PF_CHUNK` is a precondition, not
  a companion knob. With `PLOW_PF_CHUNK=2048` it fires, in pairs, on prompts >= 8192.
* **It buys nothing attributable.** Where it fires — 8192, 16384 and 32768 input at concurrency
  4 — throughput moves +0.7%, +1.1%, +4.1% and TTFT −2.5%, −2.9%, −4.6%. The signs are
  consistent, which is worth noting. The magnitudes are not: the first two sit inside a 4–6.5%
  within-arm spread, and the third is exactly matched by a **−4.1% arm-level drift in the same
  run's concurrency-1 cell, where packing provably cannot fire.**
* **Do not make it a default.** Not because it is harmful — it is inert when it cannot fire, and
  the A/A proves that costs nothing — but because turning the runtime half on alone changes
  nothing (the objects and the packet topology are separately opt-in) while arming a warning on
  every ordinary AMD serve. §7.


## 1. Packed prefill on AMD is four gates, not one

A GLM-5.3 (MLA) prompt can only be co-packed with another request's prompt when **all four** of
these hold. Every one of them was shut on this tree, and three of them said nothing.

| gate | where | state before this work |
|---|---|---|
| the runtime route | `PLOW_PACKED_PREFILL_ROUTE=1` + `PLOW_PF_BATCH=1` | both default off |
| the family objects | `scripts/build_gfx942.sh` | **no rows at all** — gfx942 never built them |
| the packet topology | `PLOW_EMIT_PACKED_PREFILL=1` in devgen | **not implemented for GLM** |
| a multi-chunk plan | `PLOW_PF_CHUNK=C` | default off ⇒ no prompt is ever packable (§4) |

Two of the three failed **silently**:

* **Objects.** `load_packed_family` (`crates/plowrt/src/exec/amd.rs`) returns `Ok(None)` for an
  object that is simply absent from the HSACO directory. `build_gfx942.sh` had no rows for the
  three family stems — the gfx950 side gets them from `runtime/CMakeLists.txt` behind
  `PLOW_HSACO_PACKED_PREFILL_CONSUMERS` (also default OFF), and the gfx942 script has no cmake
  path at all. So `PLOW_PACKED_PREFILL_ROUTE=1` on MI300X loaded nothing.
* **Capability.** With no family object loaded, `AmdEngine::packed_prefill_prog_capable` is false
  for every rung, `Engine::packable_prefill_span` returns `None`, `amd_prefill_pack` gets no
  candidates, and the mux falls through to isolated prefill. **No error, no log line, an
  unchanged number.** That is exactly the failure mode this branch has hit twice before
  (`PLOW_MLA_NS_LIVE`, `GM_AX`): a flag that reaches the process, changes nothing, and measures
  as a clean null.

Both are now closed — see §6.

## 2. The family objects: built, and they load

`scripts/build_gfx942.sh` gains three rows behind `PLOW_PACKED_PREFILL_CONSUMERS=1` (default off,
matching the cmake option it mirrors). The axes are the CMakeLists ones with the CDNA3 tile and
arch suffix this script already applies, minus the Gemma-MoE and expert arms a family object can
never dispatch:

```
interp_packed_mla_norm   -DPLOW_BUCKET_PACKED_MLA_NORM=1  -DPLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS=1
interp_packed_mla_flash  -DPLOW_BUCKET_FLASH -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS=1
interp_packed_kda        -DPLOW_BUCKET_PACKED_KDA=1       -DPLOW_PACKED_PREFILL_KDA_CONSUMERS=1
```

Built with ROCm 7.14 from the flake; every row inside its cliff:

| object | VGPR | AGPR | LDS | spill |
|---|---:|---:|---:|---:|
| `interp_packed_mla_norm` | 100 | 0 | 64 560 | 0 |
| `interp_packed_mla_norm_gq` | 98 | 0 | 64 560 | 0 |
| `interp_packed_mla_flash` | 512 | 256 | 58 368 | 101 |
| `interp_packed_mla_flash_gq` | 512 | 256 | 58 376 | 100 |
| `interp_packed_kda` | 133 | 0 | 64 560 | 0 |
| `interp_packed_kda_gq` | 133 | 0 | 64 560 | 0 |

The two norm/KDA objects are what the "lean" adjective claims: 100 and 133 VGPR against the
production prefill interpreter's 256/spill-6. The flash object spills 101 — it is the 4-wave,
512-register bucket and carries the V2 MLA arm, which is the same body the shipped
`interp_flash` runs.

The script's blanket "default objects must remain resource-clean" check (which refuses
`plow_packed_prefill_(mla|kda)_consumers_1` in any object) is narrowed to non-family rows and
replaced, on the family rows, by a **positive** assertion of the exact markers
`exec/amd.rs` demands. A family object that silently lost its marker is loaded as `Ok(None)`,
which is the silent-null door again.

**They load.** Served against the GLM-5.3 TP4 blob with `PLOW_PACKED_PREFILL_ROUTE=1`, the new
load-time line reports `objects_missing=[]` — all three passed the ELF marker set, the KV-encoding
check, the packet-pairing stamp, the kernarg-size ABI check and `module_load` on the live agent.

## 3. The packet topology: GLM never had it

With the objects present, the route still refused every rung:

```
WARN packed prefill cannot fire on this blob — every prefill rung refuses
     route=true pf_batch=true objects_missing=[]
     reason="packed-prefill MLA consumer is in a mixed segment; re-emit with PLOW_EMIT_PACKED_PREFILL=1"
```

This is not a tuning problem, it is a missing emitter feature. `exec/amd.rs`
`packed_family_segments_cover` requires every family-classed op to sit in a segment containing
**only** its own family, because the production interpreter's norm and MLA arms were not compiled
to consume span descriptors — routing a mixed segment there would run the packed rows as one
sequence and produce plausible wrong output. GLM-5.3's emitted prefill program at bucket 2048
looks like this:

| segment | ops |
|---|---|
| 0 | Embed, GemmMed, GemmSmall, GemmWide, HeadNormRope/hd64, RmsNorm |
| 1 | FlashMlaPrefill |
| 2 | Gemm, GemmMed, GemmSmall, GemmWide, HeadNormRope/hd64, MlaMergeFold, MoeAlignPf, MoeCombinePf, MoeGroupDownPf, MoeGroupGluPf, Residual, RmsNorm, XReduceTwoShot |
| 3 | FlashMlaPrefill |
| … | … |

The **flash** half (class 6) is already pure — that is `mla_v2_segment`, which isolates
`FlashMlaPrefill` at T>=2048 for the 4-wave object. The **norm** half (class 5) is not: 79 of the
157 segments carry a norm op alongside the Gemms and the whole MoE chain.

### The emitter change

The flag the error names is real, and pointed at a door that did not exist.
`Builder::set_packed_prefill_segments` had exactly one caller in the tree,
`crates/devgen/src/mla/kimi_k3.rs` — **the GLM emitter never called it**, so on a GLM blob
`PLOW_EMIT_PACKED_PREFILL=1` was a no-op and the error sent the reader in a circle.
`docs/flags-reference.md` compounded it by calling the flag "NVIDIA-only".

Even the K3 caller was dead: the config validation in `crates/devgen/src/lib.rs` asserts

    packed request emission requires Hopper single-GPU BF16 KV

for **any** emit with the flag. One flag, two features. On NVIDIA it emits the packed-**request**
ABI (a metadata section plus `pf.request.*` tensors), which genuinely is a Hopper / TP1 / BF16-KV
contract. On AMD it emits the family-segmented sibling **topology**, which has no such
requirement. The assertion applied the first contract to both, so the AMD half had never run
anywhere. Scoped to non-AMD emits.

`glm_emit_full` now builds each prefill bucket a second time with the family split on, tagged
`packed_prefill_program_t`. The runtime's `packed_prefill_prog_for` resolves an ordinary rung to
that sibling only while a packed binding is being staged, so **ordinary prefill keeps the shipped
program byte-for-byte**. With `PLOW_EMIT_PACKED_PREFILL` unset the whole blob is byte-identical —
`md5(model.pkt)` matches an emit from the unmodified compiler, which is the property that bounds
the risk of touching a shipping emitter.

What the flag produces, at bucket 2048:

| | ordinary | packed sibling |
|---|---:|---:|
| segments | 157 | **705** |
| `model.pkt` (TP4, max-ctx 16640) | 240 MB | **347 MB** |

and the segments are pure:

| segment | ops |
|---|---|
| 0 | Embed |
| 1 | RmsNorm |
| 2 | GemmMed, GemmSmall |
| 3 | RmsNorm |
| 4 | GemmSmall, GemmWide |
| 5 | HeadNormRope/hd64, RmsNorm |
| 6 | FlashMlaPrefill |
| 7 | Gemm, MlaMergeFold, Residual, XReduceTwoShot |
| 8 | RmsNorm |
| 9 | MoeAlignPf, MoeCombinePf, MoeGroupDownPf, MoeGroupGluPf, Residual, XReduceTwoShot |

Segment 5 is two different opcodes but one family (both class 5), which is what the purity rule
actually asks for. **4.5x the segment count is the price**, and it is paid on every packed
dispatch: a segment is a separate device launch under the counter protocol.


## 4. When co-packing can engage at all

Armed is not the same as firing, and the gap between them is a scheduling rule that decides which
shapes this feature can ever touch.

`Engine::packable_prefill_step` offers a slot's next chunk to the packer only when

    cur.next + 1 < cur.steps.len()

— the **final** chunk of a prompt is always isolated, because the model prefill head exposes one
sampled token, not one per span. So a prompt whose plan is a single chunk is never packable, no
matter how many of them arrive together.

That makes the chunk plan the gate:

| situation | per-request row cap at admission | 8192-token prompt plans as | packable chunks |
|---|---|---|---|
| shipped default | `u32::MAX` | one 8 191-row chunk | none |
| `PLOW_PF_CHUNK=2048` | 2048 | 2048 / 2048 / 2048 / 2047 | 3 |

The cap that matters is the one in force when the **cursor is created**
(`prepare_prefill_cursor`), not the tick cap later: `PLOW_PF_INTERLEAVE` only narrows a tick
once a decode row is already live, and a batch of requests that arrive together are all
admitted in the same cold tick. So under the shipped chunk policy every prompt up to the widest
bucket is a single chunk.

Nor does a longer prompt rescue it. A 32 768-token prompt plans as 8192 / 8192 / 8192 / 8191 —
three packable chunks — but `sched::prefill::admit` under `SpanPolicy::Whole` only takes spans
that fit the chosen rung's remaining rows, and **two 8192-row chunks cannot share an 8192-row
rung**. The pack ends with one member, and `packed.len() >= 2` fails.

**`PLOW_PF_CHUNK` is therefore not an optional companion knob — it is the third runtime
precondition.** Without it, AMD co-packed prefill on this blob is unreachable at every prompt
length.

Measured, not argued: an `armed` route with `--pf-batch` and the shipped chunk policy produced
**`packed dispatches: 0`** at every probe (128 / 512 / 2048, four simultaneous requests) and no
`packed prefill fired` line anywhere in the whole 128–8192 grid. So the entire first campaign is
an **A/A by construction** — and would have been reported as a clean null by anyone who trusted
the `armed` line alone. This is exactly the trap the branch has already walked into twice.

### That A/A bought a control worth more than the probe

Those two runs — route on and route off, both with **zero** packed dispatches, same blob, same
objects, same host, minutes apart — are an A/A on greedy output. They do not agree:

| request mode | A/A #1 (8-token) | A/A #2 (64-token, 128–8192) |
|---|---|---|
| one request at a time (`isolated`) | **12 / 12** | **12 / 12** |
| several simultaneous requests (`concurrent`) | **8 / 12** | **8 / 12** |

Two independent A/A pairs, different prompt lengths, different completion lengths, and a
different set of prompts diverging each time.

So on this model **concurrent greedy decode is already not reproducible run-to-run**, before
packed prefill enters the picture. Four simultaneous requests do not occupy the same decode rungs
in the same ticks on two different runs, the rung selects a different compiled `PLOW_GEMV_MM`
instantiation, and the accumulation order moves with it. Any "packed prefill changed the output"
claim has to clear this bar first, and a raw on-vs-off diff at concurrency cannot.

## 5. Measurement

### 5a. The A/A: route on vs route off with nothing packing

Same blob (`packbench-pp/tp4`, TP4, max-ctx 16640, decode rungs 1/2/4, `tile_source: measured`),
same HSACO directory, same host, minutes apart, `--pf-batch` on in both, shipped chunk policy.
`packed dispatches: 0` and no `packed prefill fired` line in either arm — so this is the
instrument's noise floor, not a result about packing.

3 repeats after 1 warmup, 64 output tokens, medians.

| input | conc | tok/s off | tok/s on | Δ | TTFT off | TTFT on | Δ | TPOT off | TPOT on | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 25.32 | 25.44 | +0.5% | 130 | 128 | −1.5% | 38.06 | 37.90 | −0.4% |
| 128 | 4 | 42.49 | 42.39 | −0.2% | 429 | 428 | −0.1% | 87.58 | 87.80 | +0.3% |
| 512 | 1 | 25.25 | 25.25 | +0.0% | 177 | 177 | +0.1% | 37.42 | 37.40 | −0.0% |
| 512 | 4 | 40.62 | 40.61 | −0.0% | 571 | 569 | −0.3% | 89.71 | 89.78 | +0.1% |
| 2048 | 1 | 22.36 | 22.34 | −0.1% | 430 | 429 | −0.1% | 38.61 | 38.66 | +0.1% |
| 2048 | 4 | 33.97 | 33.96 | −0.0% | 1173 | 1144 | −2.5% | 99.29 | 99.52 | +0.2% |
| 8192 | 1 | 14.89 | 14.89 | +0.0% | 1760 | 1760 | +0.0% | 40.31 | 40.31 | +0.0% |
| 8192 | 4 | 17.32 | 17.30 | −0.1% | 7941 | 7938 | −0.0% | 107.20 | 107.44 | +0.2% |

Within-arm spread over the three repeats: worst **3.51%** (off) and **6.47%** (on), both at
8192 / concurrency 4. **Read every delta below against that 6.5% floor at c4 and ~1% at c1.**

Two things this table already settles. Loading three extra HSACO objects and carrying a second,
4.5x-segmented copy of every prefill program in the packet costs **nothing at run time** — the
packed programs are never selected while no binding is staged. And the harness is tight enough
at concurrency 1 to see a 1% effect.

### 5b. The A/B: `PLOW_PF_CHUNK=2048` in both arms

Same blob, same objects, `--pf-batch` and `PLOW_PF_CHUNK=2048` **held equal in both arms**, so
the only difference is `PLOW_PACKED_PREFILL_ROUTE`. The chunk cap is in the control too because
it changes prefill cost on its own; leaving it out of the control would have measured chunking,
not packing.

**It fires.** With four simultaneous 8192-token prompts:

```
INFO  AMD packed prefill fired spans=2 program=3
DEBUG AMD packed prefill advanced spans=2   (x4)
```

It also does **not** fire below 8192 input. A prompt needs at least *three* chunks before a
middle chunk survives long enough to meet a second cursor: at 4096 with a 2048 cap the plan is
2048 / 2047, the single packable chunk is consumed on that request's own first tick, and by the
time another slot has a cursor it is gone. Measured: `packed dispatches: 0` at 4096, 4 at 8192.

Always `spans=2`, never 3 or 4. The mux only offers a slot whose cursor already exists, and the
isolated path creates one cursor per tick, so by the time slot 2 has a cursor the earlier pair
has already advanced. Of the twelve packable middle chunks in that probe (4 requests x 3
non-final chunks), eight were co-packed in pairs and four ran isolated.

#### What the chunk policy costs on its own

Before the A/B, the price of admission. Same arm (route on), campaign 1's shipped chunk policy
against campaign 2's `PLOW_PF_CHUNK=2048` — in campaign 1 nothing packed, in campaign 2 only the
8192 cells did:

| input | conc | tok/s shipped | tok/s chunked | Δ | TTFT shipped | TTFT chunked | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 25.44 | 25.71 | +1.1% | 128 | 128 | −0.0% |
| 128 | 4 | 42.39 | 42.53 | +0.3% | 428 | 428 | −0.1% |
| 512 | 1 | 25.25 | 25.26 | +0.0% | 177 | 178 | +0.3% |
| 512 | 4 | 40.61 | 40.61 | +0.0% | 569 | 570 | +0.2% |
| 2048 | 1 | 22.34 | 22.35 | +0.0% | 429 | 429 | −0.1% |
| 2048 | 4 | 33.96 | 33.99 | +0.1% | 1144 | 1196 | +4.6% |
| 8192 | 1 | 14.89 | **13.96** | **−6.2%** | 1760 | **2038** | **+15.8%** |
| 8192 | 4 | 17.30 | **18.00** | **+4.1%** | 7938 | **7247** | **−8.7%** |

Below 8192 the cap does nothing (those prompts already fit one chunk). At 8192 it is a real
trade: **−6.2% throughput and +15.8% TTFT at concurrency 1**, where the extra launches buy
nothing because a single stream can never co-pack, against **+4.1% / −8.7% at concurrency 4**.
The c4 gain is the thing the A/B has to attribute — chunking, or packing?

#### The A/B

3 repeats after 1 warmup, 64 output tokens, medians. `packed prefill fired` present in the on
arm, `packed prefill cannot fire ... route=false` in the off arm.

| input | conc | tok/s off | tok/s on | Δ | TTFT off | TTFT on | Δ | TPOT off | TPOT on | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 23.76 | 25.71 | +8.2%◊ | 135 | 128 | −5.2%◊ | 40.61 | 37.46 | −7.7%◊ |
| 128 | 4 | 42.48 | 42.53 | +0.1% | 429 | 428 | −0.3% | 87.59 | 87.50 | −0.1% |
| 512 | 1 | 25.13 | 25.26 | +0.5% | 178 | 178 | −0.1% | 37.57 | 37.38 | −0.5% |
| 512 | 4 | 40.44 | 40.61 | +0.4% | 569 | 570 | +0.1% | 90.12 | 90.05 | −0.1% |
| 2048 | 1 | 22.25 | 22.35 | +0.4% | 430 | 429 | −0.4% | 38.82 | 38.66 | −0.4% |
| 2048 | 4 | 33.87 | 33.99 | +0.3% | 1210 | 1196 | −1.1% | 99.51 | 99.23 | −0.3% |
| 8192 | 1 | 13.98 | 13.96 | −0.1% | 2039 | 2038 | −0.1% | 40.29 | 40.39 | +0.3% |
| **8192** | **4** | 17.88 | 18.00 | **+0.7%** | 7429 | 7247 | **−2.5%** | 107.93 | 109.35 | +1.3% |
| 16384 | 1 | 8.44 | 8.47 | +0.3% | 4976 | 4976 | +0.0% | 41.31 | 41.00 | −0.7% |
| **16384** | **4** | 9.72 | 9.83 | **+1.1%** | 18884 | 18332 | **−2.9%** | 115.85 | 119.73 | +3.3% |

◊ The 128/c1 row is an artefact, not a result: packing cannot fire at 128 (single chunk) or at
concurrency 1 (single stream), and that cell's off arm has a **21.1%** within-arm spread — one
slow repeat in the first cell of the run. Every other cell's spread is under 4%.

**The two cells in this table where packing actually fires are 8192/c4 and 16384/c4, and both are nulls.**
+0.7% and +1.1% throughput, −2.5% and −2.9% TTFT, against a c4 within-arm spread of up to 6.5%
(§5a) and 4% here. TPOT moves the *wrong* way at 16384/c4 (+3.3%). The signs are consistent —
TTFT down, throughput up, TPOT up — which is what co-packing should do if it does anything, but
none of it clears the noise.

Nothing packed in the other eight cells, and those eight rows read as the A/A they are: eight of
sixteen deltas at or under 0.4%.

#### 32768, on the two-slot long blob

`packbench-pplong/tp4` (max-ctx 33280, decode rungs 1,2 — the largest that fits the 13.4 GiB
this checkpoint leaves free per rank). At concurrency 4 the four clients queue through two
slots. Same `PLOW_PF_CHUNK=2048`; `packed prefill fired` in the on arm.

| input | conc | tok/s off | tok/s on | Δ | TTFT off | TTFT on | Δ | TPOT off | TPOT on | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 1 | 3.97 | 3.80 | **−4.1%** | 13521 | 13644 | +0.9% | 41.54 | 50.61 | **+21.8%** |
| 32768 | 4 | 3.90 | 4.06 | **+4.1%** | 43041 | 41074 | **−4.6%** | 101.58 | 100.37 | −1.2% |

Per-repeat throughput:

```
c1  off 3.796 3.966 3.971   on 3.799 3.802 3.803
c4  off 3.878 3.905 3.898   on 4.054 4.058 4.074
```

**Read the c1 row first, because it is the negative control.** Packing cannot fire at
concurrency 1 — a single stream never co-packs — so the two arms should be identical there, and
they are not: −4.1% throughput and +21.8% TPOT. The off arm's first repeat (TPOT 50.7 ms) agrees
with the on arm exactly and its next two do not (41.5 ms), i.e. that server drifted into a
cheaper decode path partway through its run and stayed there. Whatever the mechanism, it is
**arm-level drift of the same 4% magnitude as the c4 result, in the opposite direction.**

So the c4 row — +4.1% throughput, −4.6% TTFT, repeats tight to 0.7% within each arm — is the
largest effect in the whole campaign and is *still* not attributable. It is exactly the size of
the drift its own control shows.

### 5c. Output identity

Greedy, `temperature: 0`, 64 tokens, three prompts at each context, captured twice: once one
request at a time and once as one simultaneous batch. Character-identical means byte-for-byte
equal completion text.

| pair | packing? | isolated | concurrent |
|---|---|---:|---:|
| A/A #1 (8-token completions) | never | **12 / 12** | 8 / 12 |
| A/A #2 (64-token, 128–8192) | never | **12 / 12** | 8 / 12 |
| **A/B** (64-token, 2048–16384, `PF_CHUNK=2048`) | at 8192 and 16384 | **12 / 12** | 9 / 12 |
| **A/B** (64-token, 32768, `PF_CHUNK=2048`) | yes | **3 / 3** | 1 / 3 |

**Isolated single-stream output is identical in every pair, including the arm where packing
fires** — because a single stream never co-packs, so the route provably changes nothing it does
not touch.

**Concurrent output is not identical — and it is not identical in the controls either.** The
divergences in the A/B arm are:

| context | seed | characters agreed before diverging |
|---:|---:|---:|
| 8192 | 0 | 0 of 93 |
| 16384 | 0 | 0 of 127 |
| 16384 | 2 | 98 of 111 |
| 32768 | 0 | 0 of 113 |
| 32768 | 2 | 1 of 128 |

against A/A #2's 34/126, 2/140, 12/133 and 25/93. Three divergences out of twelve with packing
on, four out of twelve with packing off, twice. **The rate is indistinguishable, and so is the
severity.** The 32768 pair (three prompts, on a two-slot blob) diverges on two of three, which
is a higher rate on a much smaller sample and at the shape where the most packing happens; it is
not enough to call.

So the honest statement is *not* "identity holds". It is:

* identity holds for single-stream greedy decode, which is where the route cannot fire;
* at concurrency, this model's greedy output is already unstable run-to-run at ~1/3 of prompts
  with packing entirely absent, and packing does not measurably raise that;
* co-packed prefill **is** expected to move arithmetic — two 2048-row chunks become one
  4096-row dispatch, which re-tiles every dense GEMM in the packed program — so the absence of a
  detectable change here is a statement about the instrument, not a guarantee. Anyone who needs
  bit-reproducible concurrent output on GLM-5.3 does not have it today, before this feature is
  considered.

## 6. Production readiness

### It no longer refuses silently

`AmdEngine::report_packed_prefill_route` runs once per rank at load whenever the route **or**
`--pf-batch` is set, and names whichever gate is shut:

```
INFO  packed prefill armed route=true rungs=[128, 512, 2048, 8192] objects_missing=[]

WARN  packed prefill cannot fire on this blob — every prefill rung refuses
      route=true pf_batch=true objects_missing=[]
      reason="packed-prefill MLA consumer is in a mixed segment; re-emit with PLOW_EMIT_PACKED_PREFILL=1"

WARN  packed prefill cannot fire on this blob — every prefill rung refuses
      route=false pf_batch=true
      objects_missing=["interp_packed_mla_norm_gq.elf", "interp_packed_mla_flash_gq.elf"]
      reason="packed-prefill MLA norm/cache segment requires interp_packed_mla_norm"

WARN  packed prefill is capable but --pf-batch is off — co-packing will never be attempted
```

The first three are verbatim from runs in this campaign; the fourth is the `--pf-batch`-off
branch of the same function, which this campaign never exercised because `--pf-batch` was on
in every arm. Before this, every one of those states was an
`Ok(None)` and an unchanged number. The mux additionally logs `packed prefill fired` once, the
first time a co-packed dispatch actually lands — *armed* and *firing* are different claims and
only the second one licenses a measurement.

The one hole this does **not** close: the load line cannot tell you that the chunk plan will
never offer a packable chunk (§4). That depends on `PLOW_PF_CHUNK` and on prompt length, both of
which are per-request, not per-load. The `packed prefill fired` line is the backstop — its
absence over a whole benchmark is the evidence, and it is what turned this campaign's first grid
from "a null" into "an A/A by construction".

### Build recipe

`scripts/build_gfx942.sh` builds the three family objects only under
`PLOW_PACKED_PREFILL_CONSUMERS=1`, mirroring cmake's `PLOW_HSACO_PACKED_PREFILL_CONSUMERS` (also
OFF). That is deliberate and should stay: the objects are useless without a blob emitted
`PLOW_EMIT_PACKED_PREFILL=1`, and shipping them by default would add six objects and ~600 KB to
every gfx942 build for a feature no shipped blob can use. The rule the tree already states — *a
default that silently falls back because its objects are missing is worse than an opt-in* — is
satisfied by keeping all four gates opt-in **and** by the load-time line that names whichever one
is shut.

## 7. Recommendation

**Leave `PLOW_PACKED_PREFILL_ROUTE` at `false`.** `crates/plowrt/src/config.rs`'s new test
records that decision and the reason, so a future flip has to argue with it rather than with a
comment.

The reasoning, in the order that matters:

1. **The route alone is not the feature.** Three of the four gates are build-time or emit-time
   and all are opt-in. Flipping only the runtime half would load three objects that no shipped
   gfx942 build produces and no shipped blob can use, and — because the load-time report now
   runs whenever the route is on — would print a `cannot fire` warning on every ordinary AMD
   serve. That is strictly worse than the honest opt-in.
2. **The measured win does not exist yet.** +0.7% / +1.1% / +4.1% at the three cells that fire.
   The first two are inside a 4–6.5% within-arm spread. The third has tight repeats but its own
   negative control — the same run's concurrency-1 cell, where packing cannot fire — moved 4.1%
   the *other* way, so the arms differ by ~4% before the feature is considered. A default needs
   a result that survives its control, and this one does not.
3. **The chunk policy it requires is itself a trade.** `PLOW_PF_CHUNK=2048` costs −6.2%
   throughput and +15.8% TTFT at 8192 / concurrency 1 — which is a real regression for anyone
   whose traffic is not concurrent long prompts — and pays back only +4.1% at 8192 / concurrency
   4, of which packing contributes +0.7%. The other +3.4% is the chunking. **If someone wants
   the concurrency-4 long-prompt win, `PLOW_PF_CHUNK` is the lever, not this route.**
4. **Output reproducibility is unresolved on this model, independently of this feature.**
   Concurrent greedy decode diverges on ~1/3 of prompts run-to-run with packing entirely absent
   (§5c). That is worth its own investigation and is a prerequisite for qualifying *any*
   scheduling change on GLM-5.3 at concurrency.

### Where it would be worth revisiting

* **First, fix the instrument.** A 4% between-arm drift at concurrency 1 on a two-slot blob
  (§5b) and a ~1/3 run-to-run divergence rate on concurrent greedy output (§5c) are both larger
  than anything this feature does. Interleaving the two arms round-by-round instead of running
  one server then the other would kill the drift; the output instability needs its own root
  cause. Until both are done, no scheduling change on GLM-5.3 at concurrency can be qualified
  below ~5%.
* When a pack can carry more than two spans. `spans=2` every time is a scheduling limit, not a
  kernel one: the mux only offers slots that already have cursors and the isolated path creates
  one per tick. A cold-admission path that seeds several cursors before the first dispatch would
  let 4 x 2048 fill an 8192 rung instead of 2 x 2048 filling half of it, which is where the
  weight-streaming saving actually lives.
* When the packed sibling's segment count comes down. 705 segments against the ordinary
  program's 157 is 4.5x the launches for the same rows, and every one of them is a global sync
  under the counter protocol. That is the most likely reason the co-packed dispatch gives back
  what it saves. The split is currently one segment per family transition; a coarser split that
  still keeps family purity would be the first thing to try.
* On a model whose prefill program is *already* family-segmented, where the topology costs no
  extra segments. Kimi-K3 is the case the machinery was written for.

## 8. Reproducing

```bash
# objects (no GPU, ~1 min). PLOW_ROWS_ONLY leaves a PARTIAL directory -- either drop the
# three rows into an existing full object set, or omit it and build all 31 rows.
PLOW_PACKED_PREFILL_CONSUMERS=1 PLOW_ROWS_ONLY=interp_packed \
  nix develop . --command ./scripts/build_gfx942.sh <hsaco-dir>

# blob (no GPU, ~4 min) -- PLOW_EMIT_PACKED_PREFILL=1 is what adds the packed sibling topology
PLOW_EMIT_PACKED_PREFILL=1 GLM53_DIR=<out> PLOW_HSACO=<hsaco-dir> MAXCTX=16640 \
  BATCH_LADDER=1,2,4 nix develop . --command ./scripts/glm53_mi300x.sh emit 4

# serve (4-GPU lease) -- ALL THREE runtime flags, or it silently does not pack
PLOW_PACKED_PREFILL_ROUTE=1 PLOW_PF_BATCH=1 PLOW_PF_CHUNK=2048 \
  ./scripts/glm53_mi300x.sh serve 4 20200
```

Check the server log for `packed prefill armed`, then drive >= 8192-token prompts at concurrency
>= 2 and check for `AMD packed prefill fired`. Both lines are new; before them, every failure
mode above was an unchanged number.

## 9. Caveats

* **Contended.** Two sibling agents held GPUs on this host throughout (157–158 GiB resident on
  two cards outside the lease). TP4 collectives cross the fabric those processes also touch.
  Every delta here is small enough that contention matters; the A/A control is the honest bound
  on how much.
* The bench's prompt is `(" hello") x N` repeated, so the completions are degenerate in both
  arms at every length. That is the harness's standard prompt and it is fine for timing and for
  a byte-for-byte comparison, but the divergence *examples* in §5c should not be read as a
  quality signal.
* `pick_tile_tests::amd_tile_selection_follows_the_target_hwspec` fails in `cargo test -p devgen`
  on this tree, at this branch's base as well as with these changes (verified by checking
  `crates/devgen` out at `092e801d` and re-running). It is the tunedb build-digest staleness
  documented in §5b of this document, not a regression from this work.
* `AmdEngine::plan_for_at_most` builds its bucket list from `progs[..dec_lo]` without filtering
  `packed_prefill_only`, so a packed-topology blob reports each width twice
  (`buckets=[128, 512, 2048, 8192, 128, 512, 2048, 8192]` in the chunk-policy line).
  `plan_chunks_cfg` sorts and dedups before planning, so the cover is provably unchanged and
  this is a log-line defect, not a behaviour one — but it is inconsistent with
  `prefill_rungs()`, which does filter, and it will confuse the next reader. Left alone.


---

## GLM-5.3 TP4 on MI300X: DSA sparse attention now loads — and must not be shipped

Measured 2026-09-08 on 8x MI300X (gfx942), ROCm 7.14 from the flake, `plowc`/`plowrt` built in
the worktree so the tuning digest matches the objects (`tile_source: measured`, 2472/2472 on
both arms). Both blobs `--max-ctx 69632`, `PLOW_DECODE_BATCH_LADDER=1`, ladder
`full:128,512,2048,8192`, and **`PLOW_GLM_FUSE_ROPE` off in BOTH** — the armed DSA gate refuses
it (the q-rope fold and the gather arm both want `t[7]`), so the dense control gives it up too
and the A/B measures sparse attention rather than a lost fold. Driver:
`scripts/glm53_dsa_run.sh`.

## Answer in one line

The missing fp8 upcast was real and is fixed — a DSA blob now loads, serves and answers
coherently — but it was **not the only thing missing**: sparse decode is **+58 to +63 ms/token**
against dense and it **loses the needle the moment the selection stops being the identity**
(5/27 vs 27/27). The sparse attention *kernel* is 3.7x cheaper than dense exactly as predicted;
everything else about the arm is broken.

## 1. What the loader now does

`plowrt::asset::dsa_indexer` dequantises the DSA lightning indexer's two projections from
block-fp8 into the bf16 the blob declares, at bind:

    out[n][k] = bf16( f32(fp8[n][k]) * scale[n/128][k/128] )

for `self_attn.indexer.wq_b.weight` (F8_E4M3 `[4096,2048]`, scale F32 `[32,16]`) and
`self_attn.indexer.wk.weight` (F8_E4M3 `[128,6144]`, scale F32 `[1,48]`). This is the reference's
own policy — vLLM builds the fused `wk_weights_proj` with `quant_config=None` unconditionally and
dequantises a checkpoint's fp8 `wk` into it at load, "to maintain fusion" — and it is a shim for
those two names only. Every other block-fp8 tensor still reaches the kernel as fp8 and is
dequantised per 128-K block in the accumulator.

Two design points worth keeping:

* **The scale grid is REQUIRED to be `[ceil(N/128), ceil(K/128)]`, not derived from its own
  shape.** Deriving would accept a `[64,64]`-quantised checkpoint and index it as `[128,128]`:
  same byte count, plausible magnitudes, and an indexer that picks the wrong KV rows. Measured,
  that mistake differs on 98.8% of elements at rms 0.024 against a tensor whose own rms is
  0.050 — and no serving smoke test can catch it, because sparse attention is *supposed* to
  change the tokens.
* **`scrub = false` on the push.** `is_fp8_e4m3` is true for the source name, so the obvious
  code reuses the upload ring's 0x80 scrub and silently rewrites every bf16 whose high byte is
  `-0.0`'s.

Cost at load: `named tensors` gather goes 3.3–3.9 s → 4.2–5.1 s per rank (≈+1 s), and the
per-rank named-tensor slab grows 12.96 → 13.33 GiB (43 indexer layers x 8 MiB of `wq_b` that
stays fp8 in the reference). Total engine load 57.5 s (dense) → 59.7 s (DSA).

`preflight_weights` then moves *every* checkpoint-weight resolution ahead of the first DMA. The
old failure was a `slice_for` byte-count error **after ~200 GiB had been uploaded**; the
safetensors index answers the same question at t=0 for no page faults. It mirrors the upload
loop's rules rather than restating them (`is_checkpoint_weight`, `fp8/` twin routing,
two-spelling lookup, the derived `_res_score.weight` exemption); only the row-parallel gather
keeps its late failure, because measuring it means doing it.

## 2. The dequantisation is bit-identical to the reference

`scripts/glm53_dsa_verify.py` compares the loader's own output against torch's
`(fp8.float() * scale).bfloat16()` — the Rust half is an `#[ignore]`d test in the module that
writes raw bf16, the Python half reads the safetensors and grades it.

| tensor | shape | rms(ref) | max abs err | rms err | differing |
|---|---|---:|---:|---:|---:|
| `layers.0…indexer.wk.weight` | 128 x 6144 | 0.06637 | 0 | 0 | 0 / 786 432 |
| `layers.0…indexer.wq_b.weight` | 4096 x 2048 | 0.05041 | 0 | 0 | 0 / 8 388 608 |
| `layers.1…indexer.wk.weight` | 128 x 6144 | 0.03008 | 0 | 0 | 0 / 786 432 |
| `layers.1…indexer.wq_b.weight` | 4096 x 2048 | 0.03964 | 0 | 0 | 0 / 8 388 608 |

**18.3M elements, bit-identical, four tensors.** An exact match is the right bar and not a lucky
one: e4m3 carries three mantissa bits and bf16 has seven, so the decode is exact and the only
rounding in the whole path is the scale multiply and one round-to-nearest-even narrowing — the
same two steps torch takes, in the same order. Negative controls on the same tensor: reading the
grid with its axes swapped differs on 8 287 300 / 8 388 608 elements (rms 0.024), and reading it
as a 64-wide block differs on 8 300 048 (rms 0.018). The check has teeth.

*(Aside, harmless: `GLM-5.3-plow-lite` stores each indexer tensor TWICE — once in the raw shard
and once in `zz-derived-*` — byte-identical in all four cases, verified by sha256.)*

## 3. It serves

    AMD engine ready arch=gfx942 n_cu=304 max_ctx=69632 n_kvrow=177
    AMD serve engine ready n_gpu=4 batch=1 decode_rungs=[1]
    "The capital of France is Paris."   (coherent, drift-free)

21 `indexer.wq_b.weight` tensors bound (the `full` indexer layers). `max_ctx` must exceed the
emit-time crossover — `GlmCfg::dsa` arms on `ctx > 65536`, and **that is the emit-time
`max_ctx`, not the live context**, so the sparse path runs at *every* live length including
those below `index_topk` where the selection is the identity. 131072 does not fit at TP4
(`HSA_STATUS_ERROR_OUT_OF_RESOURCES` on a 2.25 GiB pool allocate); 69632 does, which leaves a
4 096-token window between the crossover and the capacity ceiling.

## 4. TPOT: sparse decode is 2.5x SLOWER, at every context

vLLM `bench serve` through the OpenAI surface, 4 prompts, 64 output tokens, concurrency 1.

| context | dense TPOT | DSA TPOT | delta | dense TTFT | DSA TTFT |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 40.82 ms | 103.52 ms | **+62.70 (+154%)** | 1 392 ms | 9 694 ms (+596%) |
| 16 384 | 41.79 ms | 103.56 ms | **+61.77 (+148%)** | 4 977 ms | 20 892 ms (+320%) |
| 32 768 | 44.10 ms | 102.95 ms | **+58.85 (+133%)** | 13 233 ms | 41 458 ms (+213%) |
| 65 536 | 50.39 ms | 102.87 ms | **+52.48 (+104%)** | 39 433 ms | 90 486 ms (+129%) |

Confirmed decode-only through `plowrt amd-bench --ctx 32768 --steps 32 --tp 4`, no HTTP and no
prefill: **110.44 ms/token sparse against 42.65 ms/token dense**, all four ranks token-identical
on every step in both arms.

**The dense control is not a straw man**: 40.8–50.4 ms reproduces the shipped baseline
(§"Long context", 38.2–40.5 ms at 4k–32k) despite the lost q-rope fold and the doubled
`max_ctx`, so the regression belongs to the DSA gate and to nothing else in the emit.

**DSA TPOT is FLAT** — 103.5 / 103.6 / 103.0 / 102.9 across a 16x context range — while dense
grows 40.8 → 50.4. The gap therefore closes at 1.92e-4 ms per token of context, which puts the
break-even near **338k context**. The KV ceiling at TP4 is ~70k. On this hardware, in this
configuration, the crossover does not exist.

**TTFT regresses too, and that one is structural rather than mysterious.** `PLOW_GLM_DSA_PF` is
off (no `plow_dsa_pf_arm` in `build-glm53/hsaco`, so the marker check would refuse it), so
prefill attention stays dense — but the `full` layers still run the whole indexer chain over
every prefill token to populate the `kidx` cache. That is the indexer's O(T·S) cost with none of
the gathered arm's O(T·top_k) saving: pure overhead, 3x to 7x on TTFT.

## 5. Where the 68 ms goes — and the good news inside it

`plowrt amd-bench --trace-raw`, one decode step at ctx 32768, TP4, folded per opcode into stall
(`t_ready - t_arrive`) and body (`t_end - t_ready`) — aggregated over workgroups, so these are
serialized sums against a 40.9 ms (dense) / 86.9 ms (DSA) wall.

| op | wg/pkt | dense span | DSA span | note |
|---|---:|---:|---:|---|
| `FLASH_MLA_DECODE` | 256 | 2 325.9 ms | — | dense attention |
| `FLASH_GATHER_DECODE` | 64 | — | 634.2 ms | **sparse attention, 3.7x cheaper** |
| `INDEX_SCORE` | **304** | — | 308.8 ms | the indexer's own score |
| `INDEX_SELECT` | 32 | — | 116.8 ms | top-k |
| `DENSE_GLU_FP8_BLK` | 304 | 68.3 ms | **12 411.3 ms** | body 42.9 vs 43.8 ms — **all stall** |
| `GEMV` | ~230 | 3 556.5 ms | 5 084.5 ms | |
| `MLA_MERGE_FOLD` | 128 | 736.6 ms | 1 529.4 ms | |
| **total serialized** | | **12 277.4 ms** | **26 201.8 ms** | stall 3 920 → 18 266 |

Read the first three rows together: **the arithmetic trade is exactly what was promised.**
Sparse attention costs 634 ms where dense costs 2 326 ms, and the indexer that buys that saving
costs 426 ms. Net −1 266 ms of serialized span, in the right direction, at the right magnitude
for the −2.65 ms/token estimate.

Then read the fourth row. `DENSE_GLU_FP8_BLK` does the *same 43 ms of work* in both arms and
spends **12 367 ms waiting** in the sparse one against 25 ms in the dense one — 13.5 ms per
workgroup, on 3 packets x 304 workgroups. Stall across the whole chain goes 3 920 → 18 266 ms;
body goes 8 357 → 7 936 ms, i.e. **the DSA arm does LESS work and takes twice as long.**

The shape of it points at the dispatch, not the kernels. `INDEX_SCORE` runs on all **304**
workgroups, and the DSA arm inserts 21 of them (one per `full` layer) into a stream whose other
full-grid op is `DENSE_GLU_FP8_BLK`. In a persistent-workgroup interpreter a full-grid packet
starts when its slowest workgroup arrives, so 21 extra full-grid rendezvous per token land as
stall on whatever full-grid op comes next. Overlap itself is unchanged (serialized/wall is 300x
in both arms) — there is simply 14 s more aggregate waiting to overlap.

## 6. Quality: it loses the needle, and the boundary is exactly `index_topk`

Character identity is the wrong gate here — sparse attention changes the arithmetic by design —
so: `scripts/glm53_needle_probe.py` (paired retrieval) and `scripts/glm53_greedy_agree.py`.

**Needle retrieval, 3 needles x 3 depths x 3 lengths, paired:**

| context | dense | DSA |
|---:|---:|---:|
| 4 096 | 9/9 | 4/9 |
| 16 384 | 9/9 | 1/9 |
| 32 768 | 9/9 | **0/9** |
| **total** | **27/27** | **5/27** |

Paired: both correct 5, **dense-only 22, DSA-only 0**, neither 0. Twenty-two discordant pairs all
in one direction is not a model limit at depth — the dense arm at identical settings retrieves
every cell — and McNemar on 22/0 needs no table.

**And the boundary is not gradual.** Sweeping the prompt length across `index_topk = 2048`, at
depth 0.5:

| achieved prompt tokens | DSA result |
|---:|---|
| 601 / 1 035 / 1 717 | 3/3, clean: `" 7429-BLUE. The code is 7429-BLUE."` |
| 1 903 | 3/3, clean |
| **2 089** | degraded: `" 742 The engine keeps one. 742 The engine keeps"` |
| 2 275 / 2 523 / 2 771 | degraded, same signature |

Below `top_k` the selection is the identity and **everything works**: the upcast weights, the
indexer GEMMs, `k_norm`, `INDEX_SCORE`, the `kidx` cache and `FLASH_GATHER_DECODE` are all
exercised on that path and produce clean, correct text. The failure begins at the first length
where the selection actually has to *choose*, when it is dropping at most 41 rows out of 2 089.
Dropping 2% of the context cannot cost the needle. **The selection is picking wrong rows** (or
the index it produces is not the index the gather consumes) — that is where the next person
should look, and it is upstream of anything in this change.

Greedy agreement (dense baseline vs DSA candidate, `/v1/completions`, temperature 0,
`ignore_eos`, 128 tokens, 4 seeds):

| context | identical | first divergence (char) |
|---:|---:|---:|
| 1 024 | 0/4 | 90 |
| 4 096 | 0/4 | 6 |
| 16 384 | 0/4 | 5 |

Consistent with the same boundary: below `top_k` the two arms track for ~90 characters (the
gathered kernel accumulates in a different order, so bit-identity was never expected); above it
they separate at the fifth character.

## 7. What still blocks making DSA a default

1. **Selection correctness — the blocker.** 22/0 discordant needle pairs, with the onset pinned
   to the exact length where selection stops being the identity. Nothing else matters until this
   is understood. Suspects, in order: the `iidx` → `FLASH_GATHER_DECODE` `t[7]` index mapping,
   `INDEX_SELECT`'s histogram top-k at `select_width` vs `index_topk`, and the `kidx` cache's
   position addressing. Note plow's DSA decode arm appears never to have been qualified against
   real weights — it could not be, since no blob could load one.
2. **21 full-grid rendezvous per token.** +14 s of aggregate stall for −1.3 s of aggregate body.
   The kernels are fine; the dispatch shape is not. `INDEX_SCORE` at 304 workgroups is the thing
   to look at.
3. **The gate arms on emit-time `max_ctx`, not live context.** A blob emitted for 69 632 runs the
   indexer at 601 tokens, where the selection is provably a no-op. A live gate would make the
   arm free below `top_k` and would also make (1) far easier to bisect.
4. **No sparse prefill on the shipped objects.** `build-glm53/hsaco` carries no
   `plow_dsa_pf_arm`, so `PLOW_GLM_DSA_PF` cannot be armed against it — and without it the
   indexer runs over every prefill token for no benefit at all, which is 3x–7x of the TTFT
   regression. Rebuild the flash object with `PLOW_DSA_PF=1` before measuring prefill again.
5. **Capacity.** 128k does not fit at TP4, so the only servable window above the 64k crossover is
   64k–70k. TP8 or a KV-format lever would be needed to reach a context where the dense/sparse
   trade could even be argued — and per §4 the extrapolated crossover is ~338k regardless.
6. **`PLOW_GLM_FUSE_ROPE` is mutually exclusive with the gate.** Not a blocker on its own, but a
   real DSA default would give up a fold the dense arm keeps, so the honest comparison for
   shipping is DSA against dense *with* the fold — a bigger gap than the table in §4.

## Reproducing

```bash
# loader verification — no GPU
PLOW_DSA_VERIFY_CKPT=/workspace/models/GLM-5.3-plow-lite PLOW_DSA_VERIFY_OUT=/tmp/dsa \
  nix develop /app/plow --command cargo test --release -p plowrt --features hsa \
    --lib -- --ignored --nocapture dsa_indexer
build-gemma31/vllm-python scripts/glm53_dsa_verify.py \
  --ckpt /workspace/models/GLM-5.3-plow-lite --dir /tmp/dsa

# blobs — no GPU. `dense` is the control: same knobs, PLOW_GLM_DSA=0.
scripts/glm53_dsa_run.sh emit dsa   69632
scripts/glm53_dsa_run.sh emit dense 69632

# decode-only timing + packet traces for both arms, one 4-GPU lease
scripts/glm53_dsa_run.sh attrib 32768 32

# serve + quality (one arm at a time; each takes its own lease)
scripts/glm53_dsa_run.sh serve dsa 20700
python3 scripts/glm53_needle_probe.py --url http://127.0.0.1:20700 --arm dsa \
  --out build-glm53/qual/needle-dsa.json --lens 4096,16384,32768 --depths 0.1,0.5,0.9
python3 scripts/glm53_greedy_probe.py --url http://127.0.0.1:20700 --arm dsa \
  --out build-glm53/qual/greedy-dsa.json --lens 1024,4096,16384
python3 scripts/glm53_greedy_agree.py --baseline build-glm53/qual/greedy-dense.json \
                                      --candidate build-glm53/qual/greedy-dsa.json
```

## Caveats

* Both arms give up `PLOW_GLM_FUSE_ROPE`, which the shipped dense config keeps. This makes the
  A/B honest about *sparse attention* and optimistic about *shipping DSA* — see §7.6.
* The needle grader is substring containment, and it is crude on purpose. One DSA cell was scored
  OK on the string `" 742 742942 742"`, which contains `7429` by accident; the neighbouring cells
  in the same sweep show the same degraded signature and were scored MISS. Read the totals, not
  a single cell.
* Sibling agents held cards on this host for parts of the campaign. Every delta reported here is
  1.3x or larger, so contention is not a plausible explanation for any of them.
* The `attrib` traces are one decode step each. They locate where time is spent; they are not a
  distribution.
