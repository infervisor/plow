# Gemma-4-31B prefill: plow vs vLLM baseline, H100 (2026-09-04)

First plow-vs-vLLM prefill number for Gemma-4-31B on this box (NVIDIA H100
80GB HBM3, sm_90a, 132 SM) — a fresh machine for plow (no prior build, no
nix, no CUDA toolkit). **Baseline only, per explicit user scope: no tuning
attempted.** The 12B campaigns (RTX 5090 sm_120a, GH200 sm_90a) already have
a validated lever catalog (single-chunk bucketing, `PGM_BN` tile changes,
native w8a8) — closing this gap is deferred to a follow-up tuning pass that
can draw on that catalog.

**Setting**: `vllm bench serve --backend openai-chat`, `--dataset-name
random --random-output-len 8 --ignore-eos` (TTFT-only), concurrency 1,
`--num-prompts 5 --seed 0` — the protocol used by every number in this
repo's prior prefill-vs-vLLM reports (`prefill-beats-vllm-w8a8-2026-08-25.md`,
`sm120-prefill-w8a8-multictx-2026-08-26.md`). Both engines bf16, run
sequentially (not concurrently — the box has one GPU and both engines need
the full checkpoint resident).

## Headline

| input tokens | vLLM 0.28.0 bf16, mean TTFT (ms) | plow bf16, mean TTFT (ms) | plow/vLLM |
|---|---|---|---|
| 2,048  | 211.73  | 942.54  | 0.22x (plow **4.45x slower**) |
| 8,192  | 814.25  | 4005.20 | 0.20x (plow **4.92x slower**) |
| 16,000 | 1855.62 | 8882.37 | 0.21x (plow **4.79x slower**) |

Zero failed requests at every point, both engines.

**The gap is a prefill problem, not a decode problem.** `vllm bench serve`
also reports TPOT (decode, time per output token after the first) from the
same runs:

| input tokens | vLLM TPOT (ms) | plow TPOT (ms) | plow/vLLM |
|---|---|---|---|
| 2,048  | 23.77 | 32.85 | 1.38x slower |
| 8,192  | 23.18 | 33.50 | 1.45x slower |
| 16,000 | 22.28 | 34.21 | 1.54x slower |

Decode trails by a real but modest ~1.4-1.5x, roughly flat with context
length — consistent with the 12B campaigns' decode-side numbers. TTFT
(prefill) trails by 4.5-4.9x and *also* degrades faster with input length
than vLLM's does (plow's TTFT grows ~9.4x from 2048->16000 tokens vs vLLM's
~8.8x). This points at prefill specifically — consistent with the
small-bucket re-chunking diagnosis below, which is a prefill-only pathology
(decode is never chunked this way).

## What was built (fresh box, no prior plow install)

- Nix installed (Determinate installer), `plowc`/`plowrt --features cuda`
  built via `nix develop -c cargo build --release`.
- CUDA 13.2 toolkit installed via apt (`cuda-keyring` repo, already
  configured) — matches driver 595.91.07.
- sm_90a decode + prefill cubins built via `scripts/build_sm90a_cubin.sh`,
  run in a clean (non-nix) shell as its header requires.
- Checkpoint access: `google/gemma-4-31B-it` was already cached at
  `/opt/dlami/nvme/hf-cache` (ACL-restricted to the `vllm`/`ubuntu` users) —
  granted read via `setfacl` rather than copying the 57 GiB checkpoint.
- Asset emitted via `plowc --hf-dir <ckpt> --emit devblob --gpu h100 --arch
  sm_90a --max-ctx 20000` (`PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33`) — the exact
  command form already validated for this model+arch in
  `perf-data/coldstart-plow-vs-vllm-gh200.md` §1b, run at `--max-ctx 20000`
  instead of that report's 8192 so the 16,000-token test point fits (the
  first emit used 8192 per this campaign's plan and had to be redone).
- `scripts/build_gemma4_h100_assets.sh` is **stale** — it references a
  `gemma4` binary removed from the tree (`plowc --hf-dir --emit devblob`
  replaces it directly, per `crates/plowc/src/main.rs:96-99`). Not used.

## vLLM config

`gemma-31b.service`'s `ExecStart`, temporarily set to (and restored after):
`vllm serve google/gemma-4-31B-it --dtype bfloat16 --gpu-memory-utilization
0.97 --max-model-len 20000` (no `--kv-cache-dtype`, no prefix-caching, no
chunked-prefill — the plain bf16 config, matching the box's own saved
`.bak-pre-fp8kv` unit modulo the mem-util/max-len bump forced by the 16k
test point: `--gpu-memory-utilization 0.95` OOM'd on KV cache at
`--max-model-len 20000`, `0.97` fixed it).

**Scope note**: an fp8 vLLM leg (`--quantization fp8`, weight quantization
— explicitly NOT KV-cache fp8) was planned but dropped mid-session per
explicit user direction. This report is bf16-vs-bf16 only.

## Correctness discipline

Minimum bar for a bf16, non-precision-changing first pass (this repo's
documented floor): `libcuda` linkage + live-compute-process check (`nvidia-
smi --query-compute-apps` shows the `plowrt` PID), and a greedy "What is the
capital of France?" exact-match smoke test, re-run after every asset
re-emit. No prior-baseline output exists for this model+arch to exact-match
a longer completion against (this is the first-ever plow build for
Gemma-4-31B on sm_90a), so a bicycle-balance-style paragraph was checked for
coherence instead of exact match — it was coherent and on-topic.

## What this report does and does not claim

- **Does claim**: on this H100, at concurrency 1, TTFT, bf16-vs-bf16, this
  session — plow trails vLLM 0.28.0 by 4.5-4.9x across 2048/8192/16000
  input tokens, zero failed requests either side. This is an out-of-the-box
  baseline: the plow asset was emitted with default settings, no tuning
  knobs applied.
- **Does not claim**: this is close to plow's achievable ceiling on this
  model/arch. The prefill buckets plowc chose (`[128, 512, 1024]`) are far
  smaller than the tested input lengths, meaning every request is served as
  many small chunks with per-chunk weight re-streaming — structurally the
  same "unoptimized chunked baseline" the 12B/RTX-5090 campaign measured as
  its *worst* configuration, before the single-chunk-bucket fix
  (`PLOW_MAX_CHUNK`) and further leverage (native w8a8) closed most of a
  similarly-shaped gap and then reversed it to a plow win at one fixed
  setting (`prefill-beats-vllm-w8a8-2026-08-25.md`). None of that tuning was
  attempted here, per explicit scope. Does not claim an fp8 comparison on
  either side (both dropped/deferred — see Scope note above).

## Open items (Phase 2 — tuning, not started)

1. Single-chunk bucketing (`PLOW_MAX_CHUNK`) at 31B/H100 — the highest-prior
   lever from the 12B campaigns, untested at this model size/arch.
2. Re-measure whether `PLOW_NS_FULL_ABS=33` (the documented H100-specific
   value, already applied here) and the default prefill-bucket choice
   interact badly with a 60-layer, hidden=5376 model — the 12B campaigns'
   bucket-size intuition was built on a smaller model.
3. Native w8a8 (fp8 weights + fp8 activations) — proven to *beat* vLLM bf16
   at one fixed setting on 12B/RTX-5090; not yet built or tested for 31B on
   sm_90a (needs `PLOW_BUILD_W8A8=1` cubins + fp8 twin weights via
   `perf-data/tools/quantize_fp8.py`).
4. GSM8K accuracy gate (`scripts/bench_gsm8k.sh`, N=200 floor per this
   repo's own convention) once any precision-changing lever is tried.
5. An apples-to-apples fp8 vLLM comparison (`--quantization fp8`) was
   dropped from this pass's scope; revisit if/when plow has its own fp8 leg
   to compare against.
