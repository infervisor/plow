# Gemma-4 on H100 NVL (sm_90a) — vLLM baseline + plowrt bring-up (2026-07-23)

First Hopper numbers in this repo. **Every prior campaign in `perf-data/` was sm_120**
(RTX PRO 6000 Blackwell, 188 SM); the sm_90a interpreter is one commit old (`dac599e`).
Nothing here is comparable to the committed sm_120 rows.

## Environment

| | |
|---|---|
| GPU | 1x NVIDIA H100 NVL, sm_90a (cc 9.0), **132 SM**, 95830 MiB, smem_optin 232448 |
| Kernel driver | 570.133.20 (**CUDA 12.8**) |
| Toolkit | CUDA **13.0** (V13.0.88) — see "cubin ABI" below |
| vLLM | 0.25.1, torch 2.11.0+cu130, FLASH_ATTN backend, full cudagraphs |
| Benchmarker | HuggingFace `inference-benchmarker` v1.1.0 (`bad4f947`) |
| GPU serialization | `perf-data/harness/gpulease`; card shared with other agents |

Models (both `model_type: gemma4`, `Gemma4ForConditionalGeneration`, bf16, vocab 262144,
tied embeddings, softcap 30.0, per-layer-type RoPE):

| | gemma-4-26B-A4B-it | gemma-4-E4B-it |
|---|---|---|
| kind | MoE, 128 experts, top-8, `moe_inter` 704 | dense |
| layers / hidden / inter | 30 / 2816 / 2112 | 42 / 2560 / 10240 |
| heads, hd (slide/full) | 16, 256/512 | 8, 256/512 |
| kv heads (slide/full) | 8 / 2 | 2 / 2 (`num_global_key_value_heads: null`) |
| sliding window | 1024 | 512 |
| `attention_k_eq_v` | true | **false** |
| per-layer input embeds | 0 | **256** (`vocab_size_per_layer_input` 262144) |
| cross-layer KV sharing | 0 | **18 of 42** |
| towers | vision | vision + audio |
| on disk | 49 GiB (2 shards, 1013 tensors) | 15 GiB (1 file, 2130 tensors) |

Benchmark protocol (identical for both engines): OpenAI `/v1/chat/completions`, streaming,
prompt pinned **1024** tokens and output pinned **256** (`variance=0`), 120 s duration,
30 s warmup. Single-user = `--benchmark-kind throughput --max-vus 1`.
Correctness gate before every measurement: `"What is the capital of France?"`, temp 0.

## 1. vLLM baseline — all 4 configs CLEAN, 0 failed requests

**These are "pass 2" numbers, measured with `--no-enable-prefix-caching`.** An earlier pass
was discarded: it was degraded by **un-leased** GPU contention (`gpulease` is advisory — it
proves non-overlap only among processes that call it). Pass-1 26B bf16 ITL read 11.13 ms vs
4.833 ms here, and the deviation was inconsistent in magnitude *and direction* across
configs — the signature of interference, not a mechanism. A controlled A/B with both arms
back-to-back in ONE lease hold showed prefix caching is **not** the cause (ITL 4.851 ON vs
4.859 OFF, within 0.2%); three independent measurements now agree within 0.5%.

Single-user (`--max-vus 1`), `--gpu-memory-utilization 0.90`, tp=1, prefix caching OFF:

| model | quant | ok/fail | TTFT avg/p90 | ITL avg/p90 | e2e avg | tok/s |
|---|---|---|---|---|---|---|
| 26B-A4B | bf16 | 95/0 | 55.02 / 57.85 | 4.833 / 4.861 | 1266.2 | 197.7 |
| 26B-A4B | **fp8** | 105/0 | **45.48** / 48.41 | **4.417** / 4.423 | 1145.1 | **218.1** |
| E4B | bf16 | 84/0 | 47.58 / 49.57 | 5.539 / 5.548 | 1430.5 | 175.2 |
| E4B | **fp8** | 96/0 | **42.81** / 44.74 | **4.795** / 4.804 | 1251.8 | **202.1** |

All gates PASSED (`'The capital of France is **Paris**.'`). p90 within 1% of avg throughout.

**fp8 wins on every axis for both models** (26B −8.6% ITL, E4B −13.4% ITL). An earlier
version of this document claimed fp8 *hurt* E4B; that was an artifact of the discarded pass
and is **retracted**.

Note the 26B-A4B MoE decodes **faster** than dense E4B (4.833 vs 5.539 ms) — only ~4B of
26B params are active.

Saturation (sweep, 8 rates). **p90 TTFT never crosses 1 s in any config** (worst 816.9 ms):
with chunked prefill at 8192 tokens vLLM keeps queueing delay off TTFT and pushes back-
pressure into ITL, so *TTFT is the wrong saturation signal here* — use ITL or achieved-vs-
target rate.

| model | quant | peak out tok/s | peak req/s | knee req/s | ITL at knee |
|---|---|---|---|---|---|
| 26B-A4B | bf16 | 1549.0 | 6.515 | ~4.6 | 25.2 ms |
| 26B-A4B | fp8 | 2285.3 | 9.391 | ~8.2 | 21.6 ms |
| E4B | bf16 | 3967.0 | 16.757 | ~14.2 | 12.0 ms |
| E4B | fp8 | 4780.1 | 19.960 | ~17.6 | 11.0 ms |

Prefix caching (vLLM's default ON) inflated the E4B sweep **1.60x** (26.795 -> 16.757 req/s):
E4B's 1.25M-token KV cache evicts nothing so 99.1% of prefill was skipped, while the 26B's
143K cache evicts constantly (5.1-5.8% hits). Any engine comparison must match this setting.

Server-side (startup logs):

| model | quant | weights in GPU | avail KV | KV tokens | max conc. @8192 | startup |
|---|---|---|---|---|---|---|
| 26B-A4B | bf16 | 48.54 GiB | 29.99 GiB | 142,705 | 17.42x | 155 s |
| 26B-A4B | fp8 | **25.81 GiB** | 52.73 GiB | 250,862 | 30.62x | 155 s |
| E4B | bf16 | 15.31 GiB | 66.68 GiB | 1,245,006 | 151.98x | 125 s |
| E4B | fp8 | 11.24 GiB | 70.74 GiB | 1,320,895 | 161.24x | 130 s |

fp8 helps the 26B most because weights drop 48.54 -> 25.81 GiB, handing 22.7 GiB back to KV.
fp8 path: `CutlassFP8ScaledMMLinearKernel`; MoE via the **TRITON Fp8 MoE backend**.

### ⚠️ The 26B MoE runs on an UNTUNED Triton fused-MoE kernel

Both 26B configs warn: `Using default MoE config. Performance might be sub-optimal! Config
file not found at .../E=128,N=704,device_name=NVIDIA_H100_NVL.json`. vLLM ships hand-tuned
fused-MoE tile configs per (experts, intermediate, GPU) and **has no entry for this shape on
H100 NVL**. Carry this when quoting 26B numbers in either direction: vLLM's 26B figures are
not its tuned best, so beating them would be a weaker claim than beating a tuned kernel —
and losing to them is correspondingly worse. E4B (dense) has no such warning.

## 2. plowrt — 26B-A4B

Assets: `scripts/build_gemma4_h100_assets.sh`, n_cu=**132**, max_ctx 8192, default
(non-w8a8) sm_90a cubins. fp8 twins from `perf-data/harness/quantize_fp8.py`
(22.86 GiB, 265 weight/scale pairs, relL2 **2.64e-2**, scale bit-exact vs amax/448).

| | bf16 | fp8 |
|---|---|---|
| weights | 47.00 GiB | **24.26 GiB** |
| KV | 1.72 GiB | 1.72 GiB (bf16 in **both**) |
| programs | 7 (6 prefill buckets + decode) | **1 (decode t=1 only)** |
| gate | PASSED | PASSED |

### 2a. HEADLINE — decode-only kernel TPOT (`step_bench`, slots=1, 128 steps)

No HTTP/mux/SSE. bf16 and fp8 interleaved to control thermal drift, each under its own lease.

| ctx | bf16 mean ms | sd | fp8 mean ms | sd | fp8 gain |
|---|---|---|---|---|---|
| 128 | 9.198 | 0.017 | 7.278 | 0.018 | **+20.9%** |
| 1024 | 9.280 | 0.021 | 7.368 | 0.015 | **+20.6%** |
| 4096 | 9.341 | 0.021 | 7.412 | 0.012 | **+20.7%** |

bf16 grows 9.198 → 9.341 ms (+1.6%) over a 32x context range and fp8 7.278 → 7.412 ms
(+1.8%): attention is a rounding error next to the weight stream at B=1, which is exactly
why the ratio does not move.

**The fp8 win is flat in context** (+20.9% → +20.6% → +20.7%), i.e. it comes from the weight stream
(47.00 → 24.26 GiB), not from attention. Decode at B=1 is weight-bandwidth bound; the KV
cache is bf16 in both packets (`kv_gib` identical), so ctx does not move the ratio.
sd ≤ 0.021 ms over 128 steps — clean, uncontended.

vLLM sees a smaller fp8 gain on the same model (−8.6% ITL), so plow extracts more from
fp8 in relative terms — but from a ~1.9x slower bf16 starting point (see 2b).

### 2b. Cross-engine: plowrt decode is ~1.9x SLOWER than vLLM

Comparing clean-window measurements only (see the timing caveat below):

| | plowrt kernel TPOT | vLLM served ITL | plowrt / vLLM |
|---|---|---|---|
| bf16 | 9.280 ms | 4.833 ms | **1.92x slower** |
| fp8 | 7.368 ms | 4.417 ms | **1.67x slower** |

plowrt's served bf16 ITL (9.44 ms) sits 0.16 ms above its own kernel TPOT, so the gap is
kernel, not serving overhead. **An earlier version of this document claimed plowrt bf16
decode beat vLLM by 15.2%; that compared against the discarded pass-1 vLLM number and is
retracted — the true relationship is the reverse.** Note the vLLM 26B figures come from an
untuned Triton fused-MoE (above), which makes the deficit more notable, not less.

Served TTFT is **not** comparable in either direction:

| engine | config | ok/fail | TTFT avg | ITL avg | tok/s |
|---|---|---|---|---|---|
| vLLM | bf16 | 95/0 | 55.0 | 4.833 | 197.7 |
| plowrt | bf16, **prefill OFF** | 10/0 | 9629.0 | 9.44 | 21.3 |
| plowrt | fp8 (no prefill program) † | 7/0 | 15593.9 | 14.78 | 13.2 |

† **Measured in the contaminated window (21:54) — do not use.** See 2c.

**Neither plowrt row measures a prefill kernel.** The fp8 packet contains no prefill
program, so the server falls back to decode-only prompt consumption; the bf16 row was
deliberately run with prefill disabled so it is comparable. **plowrt's prefill throughput is
unmeasured — the largest gap in this campaign.**

Also: vLLM applied each model's `chat_template.jinja`, so the 1024 *counted* prompt tokens
are pre-template. An apples-to-apples plowrt prefill run must apply the same template or the
real prefill lengths differ.

Fairness: the packet was emitted with `PLOW_DECODE_BATCH` unset -> `batch=1`, provisioning
exactly one 8192-token sequence (1.72 GiB KV, **1.00x** concurrency) against vLLM's 17.42x.
Per-token KV cost is essentially identical (220.0 vs 220.4 KiB/token), so that gap is
*provisioning*, not layout. A concurrency comparison needs a B=4/B=8 re-emit, not attempted.

### 2b-i. The two fp8 columns are NOT the same scheme

vLLM fp8 is **runtime-quantized** (`--quantization fp8`, on-the-fly weight-only e4m3 from the
bf16 checkpoint). plowrt fp8 uses **pre-quantized twins** baked at compile time
(`quantize_fp8.py`, per-output-channel scale, relL2 2.64e-2). So "vLLM fp8 vs plowrt fp8" is
two different fp8 schemes on two engines, not one scheme on two engines. The
**within-engine** fp8-vs-bf16 deltas (−8.6% ITL for vLLM, +20.6% TPOT for plowrt) are the
sound comparisons; the cross-engine fp8 row is not.

### 2c. Measurement-window caveat (retracts the "fp8 serving anomaly")

Cross-referencing the lease log against the clean window (vLLM pass-2 starts ~22:26):

| plowrt measurement | time | window |
|---|---|---|
| `step_bench`, all 6 TPOT points | 22:25-23:19 | **clean** |
| served bf16 (prefill OFF) | 22:25:45 | **clean** |
| served fp8 | 21:54:01 | **contaminated** |

An earlier version reported an "fp8 serving-path anomaly" — served fp8 ITL exceeding kernel
TPOT by ~100% (14.78 vs 7.368 ms) against bf16's ~2%. That row was measured in the
contaminated window, so the anomaly is most plausibly the same un-leased contention that
degraded vLLM pass 1, **not** a serving-path property of fp8. Retracted as a finding;
re-measure before treating it as real.

### 2d. Open: one unexplained bf16 empty generation

One Phase-1 bf16 gate returned `''`. Initially attributed to the sm_90a prefill kernel;
**that attribution is retracted** — the plowrt binary was rebuilt between the failure and
every later test, confounding "prefill on/off" with "old/new binary". On the current binary:
6/6 prefill buckets correct, byte-exact replay 3/3 correct as a first-request-after-load,
**24 chat requests with prefill enabled, 0 empty**. Not reproducible; cause unidentified.
Re-test deliberately before trusting bf16 serving.

## 3. plowrt — E4B: BLOCKED, and correctly so

`plowc` cannot compile E4B. This is not a tuning gap; three architecture features are
unimplemented:

1. **Per-layer input embeddings** (`hidden_size_per_layer_input: 256`) — 129 tensors:
   `embed_tokens_per_layer` [262144,10752], `per_layer_model_projection` [10752,2560],
   `per_layer_projection_norm`, and per-layer `per_layer_input_gate` / `per_layer_projection`
   / `post_per_layer_input_norm`. **5.4 GiB of trained weights.**
2. **`attention_k_eq_v: false`** — `bin/gemma4.rs` hardcoded `k_eq_v: true`, feeding K into
   the V slot on the 7 full-attention layers. Shapes match, so nothing tripped.
3. **Cross-layer KV sharing** (`num_kv_shared_layers: 18` of 42) — unread by both entry
   points. Name-based coverage *cannot* catch this: the checkpoint ships real k/v tensors on
   all 42 layers, but the reference model discards KV on layers ≥ 24.

E4B has **no** altup/laurel tensors — those are Gemma-3n, not Gemma-4. Attention geometry,
RoPE, sandwich norms, GeGLU and the tied head are all identical to 26B. Estimated fix:
**~2–3 engineer-weeks, entirely in Rust, zero `.cu` changes** — every PLE op maps onto
existing kernels (`d_embed` already takes runtime hidden/scale; the gate is
`d_glu(gate, up, gelu_tanh)`). Main costs are emitter plumbing, PLE activation memory
([T,42,256] = 176 MiB at T=8192), and KV-allocator work for shared layers.

**No E4B plowrt number is reported because any such number would measure a different model.**
With the coverage check bypassed, both `plowc` entry points emit a loadable packet reporting
`weights 8.60 GiB` — exactly `14.00 − 5.404` — and it would generate fluent, wrong text.

## 4. Bring-up bugs found (all fixed on this branch)

| # | Bug | Fix |
|---|---|---|
| 1 | `plowrt --features cuda` **did not compile** at HEAD (`E0603: function 'on' is private`, `mux.rs:850,971`). Hidden from CI: without `cuda` the call sites are cfg'd out. | `1e324b5` |
| 2 | **`gemma4` had no checkpoint coverage gate** — the shipping asset builder. `--hf-dir` has had one since day one. E4B emitted a clean, warning-free, structurally-wrong packet. | `97fe891` |
| 3 | **CUDA_ERROR_INVALID_IMAGE on every cubin load.** nvcc 13.0 emits ELF ABI 8; the distro driver is 570.133.20 (12.8). A forward-compat 580.167.08 driver is installed, and `ldconfig` resolves to it — but a nix-glibc binary's `dlopen` ignores the host ld cache, so the bare SONAMEs miss and we fell through to the hardcoded distro path. Verified: 12080 → INVALID_IMAGE, 13000 → SUCCESS (one libcuda per process; two in one address space makes the second `cuInit` fail spuriously). | `cd717de` |
| 4 | Load line hardcoded `"sm_120 decode program"` — printed sm_120 on Hopper while correctly running the sm90a object. | `cd717de` |
| 5 | `gpulease` reported **false contention**: it compares nvidia-smi **host** PIDs against `$$`, so in a PID namespace every process — including the lease holder's own child — looks foreign, and rc was rewritten to 76 on **every** successful GPU benchmark. | `fc11112` |
| 6 | README's asset manifest `{"buckets": []}` does not deserialize (`plow_asset::Manifest` needs network/gpu/num_gpus/parallel/weight_shared, no serde defaults). `network` also names the model in plowrt's registry. | `fc11112` |

**fp8 w8a8 is emulated on sm_90a — do not use the `_w8a8` cubins.** The arm uses
`mma.sync m16n8k32 ... e4m3`; ptxas lowers it to `F2FP` conversions + 2x fp16 `HMMA`.
Object diff: bf16 prefill has 384 `HMMA...BF16`; w8a8 has 288 `.BF16` + 384 plain `HMMA`
plus conversions and +768 B stack spill — same MMA issue count as bf16 plus overhead.
`runtime/CMakeLists.txt:242-256` already omits w8a8 from Hopper targets. Native Hopper fp8
needs `wgmma.mma_async`, unimplemented here. sm_90a prefill also sits at the **255-register
ceiling** (vs 238 on sm_120) with progressive spill in the fp8 variants — no headroom.

## 5. What did NOT run

- plowrt **prefill / TTFT** (the biggest gap) — bf16 can prefill and is verified correct at
  all 6 buckets, but the served run was configured prefill-OFF for comparability.
- (ctx=4096 landed after the first writeup; the TPOT table is now complete at 128/1024/4096.)
- plowrt **concurrency sweep** — deliberately dropped; B=1 makes it meaningless.
- **E4B on plowrt** — architecturally blocked (§3).
- `PLOW_FP8_KV` / `_fp8kv` cubins — built and symbol-verified, not benchmarked.

## Reproduce

```bash
# assets (CPU only)
python perf-data/harness/quantize_fp8.py /workspace/models/gemma-4-26B-A4B-it \
    /workspace/models/gemma-4-26B-A4B-it/fp8-full-plow
PLOW_ROOT=$PWD scripts/build_sm90a_cubin.sh /workspace/assets/cubin-sm90a/interp_sm90a.cubin
scripts/build_gemma4_h100_assets.sh /workspace/models/gemma-4-26B-A4B-it \
    /workspace/assets/plowrt-26b 8192

# decode TPOT (the headline number)
gpulease sw-bf16 env PLOW_STEP_TIME=1 \
    target/release/examples/step_bench /workspace/assets/plowrt-26b/bf16 1 1024 128
```
Raw artifacts: `/workspace/bench/vllm/`, `/workspace/bench/plowrt/REPORT.md`.
