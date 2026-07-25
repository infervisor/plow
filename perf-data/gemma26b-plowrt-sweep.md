# Gemma-4-26B-A4B — plowrt single-stream decode TPOT sweep, bf16 vs fp8 (2026-07-23)

Kernel-only decode-step sweep on the `worktree-gpu-exec-stage1` engine
(HEAD `ec4b555` "plowrt: device stochastic sampler kernel"), measured with
`crates/plowrt/examples/step_bench` — drives `GpuEngine` directly (no HTTP / mux
/ SSE), so the numbers are pure single-stream TPOT (time-per-output-token) at
batch B and context `ctx`.

- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SMs), shared
  and contended — every GPU command ran under `gpulease` (single exclusive lease),
  bf16 and fp8 **interleaved** to control for thermal/clock drift.
- Model: `/workspace/models/gemma-4-26B-A4B-it` (26B-A4B MoE, ~4B active, bf16).
- fp8 twin: `/workspace/models/gemma-4-26B-A4B-it/fp8-full-plow` (weight-only fp8
  experts+projections; **embeddings / lm_head / norms / KV stay bf16** — no
  `PLOW_FP8_HEAD`, matching vLLM's fp8 recipe).
- 30 layers (5 full-attn), hidden 2816, inter 2112, heads 16, hd 256/512,
  kvh 8/2, vocab 262144. Packet `max_ctx=8192`, `n_cu=188`.

## Headline

fp8 (weight-only, bf16 head) cuts single-stream decode TPOT by **~29-30 %**
(**1.29-1.30×**) at B=1 across 128/1k/4k ctx, and **~26 %** (**1.26×**) at B=4
ctx=2048. Decode is weight-bandwidth-bound and the fp8 packet moves **24.3 GiB**
of weights vs bf16's **47.0 GiB** (1.94× fewer bytes); the win is below that
ideal because the bf16-retained tensors (large 262144-vocab lm_head GEMV, KV
reads, norms, embeddings) are not halved. Measurements are extremely stable
(sd ≤ 0.022 ms across all rows).

## Results (step_bench, steps=128, warmup 16 discarded)

| config | precision | B | ctx | mean ms/tok | median ms | sd ms | tok/s (per-user) | tok/s (aggregate) | dev_interp_ms |
|--------|-----------|---|-----|-------------|-----------|-------|------------------|-------------------|---------------|
| bf16-b1 | bf16 | 1 | 128  | 7.930 | 7.926 | 0.022 | 126.1 | 126.1 | 7.897 |
| fp8-b1  | fp8  | 1 | 128  | 6.104 | 6.101 | 0.018 | 163.8 | 163.8 | ≈6.05 |
| bf16-b1 | bf16 | 1 | 1024 | 8.049 | 8.051 | 0.012 | 124.2 | 124.2 | 8.018 |
| fp8-b1  | fp8  | 1 | 1024 | 6.219 | 6.218 | 0.007 | 160.8 | 160.8 | ≈6.17 |
| bf16-b1 | bf16 | 1 | 4096 | 8.110 | 8.109 | 0.012 | 123.3 | 123.3 | 8.071 |
| fp8-b1  | fp8  | 1 | 4096 | 6.265 | 6.264 | 0.010 | 159.6 | 159.6 | ≈6.21 |
| bf16-b4 | bf16 | 4 | 2048 | 11.284 | 11.281 | 0.016 | 88.6 | 354.5 | 11.243 |
| fp8-b4  | fp8  | 4 | 2048 | 8.990 | 8.990 | 0.012 | 111.2 | 445.0 | ≈8.94 |

Device-event breakdown (from `PLOW_STEP_TIME=1`, the engine's `step_slots means`
CUDA-event line) for the clean bf16 rows:

| config | dev_interp_ms | dev_upload_us | dev_download_us |
|--------|---------------|---------------|-----------------|
| bf16-b1 ctx128  | 7.897 | 14.2 | 4.1 |
| bf16-b1 ctx1024 | 8.018 | 13.6 | 4.3 |
| bf16-b1 ctx4096 | 8.071 | 18.0 | 4.5 |
| bf16-b4 ctx2048 | 11.243 | 16.7 | 4.5 |

The step is entirely device-bound: `dev_interp_ms ≈ mean_ms`, host
gap+submit ≈ 25 µs, upload ≈ 14-18 µs, download ≈ 4 µs. For the **fp8** rows the
engine's cumulative breakdown line is averaged over the decode-only
prompt-consumption ramp (see note below), so its `dev_interp_ms` slightly
understates the timed-decode kernel; the `mean_ms` column is the TPOT of record
(host overhead is the same ~50 µs, so timed fp8 interp ≈ mean − 0.05 ms, shown
as ≈ above).

## bf16 vs fp8 speedup (mean ms/tok)

| B | ctx | bf16 ms | fp8 ms | speedup (bf16/fp8) | tok/s gain |
|---|-----|---------|--------|--------------------|------------|
| 1 | 128  | 7.930 | 6.104 | **1.299×** | 126.1 → 163.8 |
| 1 | 1024 | 8.049 | 6.219 | **1.294×** | 124.2 → 160.8 |
| 1 | 4096 | 8.110 | 6.265 | **1.294×** | 123.3 → 159.6 |
| 4 | 2048 | 11.284 | 8.990 | **1.255×** | 354.5 → 445.0 agg |

B=4 raises aggregate throughput (bf16 354.5 → fp8 445.0 tok/s) at the cost of
per-user TPOT (4 slots share the decode kernel); fp8's per-slot cost stays lower.

## Build commands (exact)

Sync + confirm engine:
```
git fetch origin && git reset --hard origin/worktree-gpu-exec-stage1
git log --oneline -1   # ec4b555 plowrt: device stochastic sampler kernel
```

Compiler (inside nix):
```
nix develop -c cargo build --release -p plowc --bin gemma4
```

Cubins — GF_FULL=4 recipe, built OUTSIDE nix (`env -i` inside the script keeps
nix glibc out of nvcc's RUNPATH):
```
sed 's/-DPLOW_NV_FA_GF=2/-DPLOW_NV_FA_GF=2 -DPLOW_NV_FA_GF_FULL=4/g' \
    scripts/build_sm120_cubin.sh > /tmp/bc.sh
PLOW_ROOT=$PWD bash /tmp/bc.sh <out>/interp_sm120.cubin
# -> interp_sm120.cubin (decode) + interp_sm120_pf.cubin (prefill)
```

Packets (`gemma4 <model-dir> <max_ctx> <out.pkt> <n_cu>`, n_cu=188, max_ctx=8192):
```
# bf16  (B in {1,4})
PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_DECODE_BATCH=B \
  gemma4 /workspace/models/gemma-4-26B-A4B-it 8192 bf16-bB.pkt 188
# fp8   (B in {1,4}) — PLOW_FP8=1, NO PLOW_FP8_HEAD (twins keep lm_head bf16)
PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_DECODE_BATCH=B PLOW_FP8=1 \
  gemma4 /workspace/models/gemma-4-26B-A4B-it 8192 fp8-bB.pkt 188
```

step_bench:
```
nix develop -c cargo build --release -p plowrt --features cuda --example step_bench
PLOW_STEP_TIME=1 step_bench <assets_dir> <slots> <ctx> 128
```

Sweep driver (interleaves bf16/fp8, one gpulease per config, retries rc=75):
```
scripts/bench_gemma26b_stepsweep.sh <assets_root> <out_log>
```

## Asset layout

One serve-asset dir per (precision, batch): `bf16-b1 fp8-b1 bf16-b4 fp8-b4`.
Each dir contains:
```
<dir>/model.pkt                 # the compiled packet for that (prec,B)
<dir>/interp_sm120.cubin        # decode object (GF_FULL=4)
<dir>/interp_sm120_pf.cubin     # prefill object (bf16 only uses it)
<dir>/tokenizer.json
<dir>/checkpoint/               # what GpuEngine::load mmaps
```
`checkpoint/` (the engine scans every `*.safetensors` in it, mapping tensors by
name — no index.json needed):
- **bf16 dirs**: symlinks to `model-0000{1,2}-of-00002.safetensors`
  + `config.json` + `generation_config.json` + `tokenizer.json`.
- **fp8 dirs**: BOTH sets, renamed so filenames don't collide (tensor names
  don't — bf16 is `model.*`, fp8 is `fp8/model.*`):
  - `bf16-0000{1,2}-of-00002.safetensors` → the bf16 shards (embeddings, norms,
    lm_head, KV geometry come from here)
  - `fp8-0000{1,2}-of-00002.safetensors`  → the `fp8-full-plow/` shards
  - + `config.json` + `generation_config.json` + `tokenizer.json`.

## Sanity / correctness gate (PASS)

- **Load**: all 4 packets load cleanly. Engine log confirms
  `weights_gib=47.00` (bf16) / `24.26` (fp8), `kv_gib=1.72` (B=1) / `6.88` (B=4),
  `batch=1`/`4`, `vocab=262144`, `max_ctx=8192`, `stop_ids=[1,106,50]`.
- **Coherence**: `gpu_lifecycle` acceptance test (load → serve the chat-template
  "capital of France" prompt → greedy decode → unload → reload → serve again).
  Both **bf16-b1** and **fp8-b1** reply `"Paris"` on both load cycles and PASS.
  VRAM curve is clean: baseline 33509 MiB → +52018 (bf16) / +27762 (fp8) on load
  → back to 33509 on unload → identical on reload (no leak / double-free).
- **Plausibility**: bf16 26B-A4B decode (~4B active) at 1k = 8.05 ms/tok, on par
  with dense 12B (~8.24 ms/tok in `gemma4-26b-plow-sm120.md`) as expected for
  similar active-param bandwidth; sd ≤ 0.022 ms on every timed run.

## Notes / caveats

- **fp8 packets carry no prefill program.** `gemma4` with `PLOW_FP8=1` emits only
  the T=1 decode program (fp8 prefill is not implemented — consistent with
  `gemma4-26b-plow-sm120.md` "fp8 prefill not implemented"). step_bench therefore
  logs `prefill disabled — decode-only prompt consumption` for fp8 and builds the
  KV context by stepping the decode kernel token-by-token. This does **not**
  affect the decode-TPOT measurement (the timed region is pure decode at the
  target ctx); it only makes fp8 context setup slower in the harness. bf16
  packets carry all 6 prefill buckets and use the prefill kernel.
- Because the fp8 decode-only ramp runs through the same `step_slots` counter,
  the engine's periodic `PLOW_STEP_TIME` breakdown line for fp8 is a
  run-cumulative mean over the growing ctx, so its `dev_interp_ms` is marked ≈.
  The `RAW_STEP mean_ms` is the clean timed-decode TPOT for every row.
- **No config failed to load or run.** All 8 (config × precision) points produced
  valid RAW_STEP output.
- This is a `PLOW_FP8` (bf16 lm_head) run. The committed
  `gemma4-26b-plow-sm120.md` reports a faster `fp8+head` variant (fp8 lm_head
  twin, ~0.4 ms/tok cheaper) — not measured here per the twin's constraints
  (`PLOW_FP8_HEAD` would demand a nonexistent `fp8/…embed_tokens` here).

## Reproduce
```
scripts/bench_gemma26b_stepsweep.sh /root/.claude/jobs/gemma26b-sweep/assets results.log
```
Env: `GPU_LEASE_TIMEOUT=7200 GPU_LEASE_IDLE_MIB=40000` (a ~33 GiB resident server
holds VRAM so the default idle mark never clears).
