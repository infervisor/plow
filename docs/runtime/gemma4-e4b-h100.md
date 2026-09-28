# Gemma 4 E4B on H100 (voice-agent LLM)

`google/gemma-4-E4B-it` compiled by plowc and served by `plowrt serve` on one H100 SXM (sm_90a) as
`gemma-4-e4b` on `/v1/chat/completions` and `/v1/completions`, with OpenAI `logprobs`. It is the
LLM of the voice-agent server, next to Qwen3-ASR and Chatterbox ([tts.md](tts.md), [asr.md](asr.md)).

## Build

```sh
# no nix: PLOW_CAMPAIGN_NO_NIX=1, CARGO_TARGET_DIR holding a release plowc, PLOW_NVCC (+ NVCC_PREPEND_FLAGS)
python3 scripts/campaign/campaign.py build recipes/infervisor/gemma-4-e4b/sm90a-h100-tp1.toml --out $OUT
PLOW_HSACO=$OUT/assets plowrt serve --assets $OUT/assets --multistep-adaptive
```

The recipe runs base emit -> `scripts/build_sm90a_gemma4_segments.sh` -> role emit and keeps only
the two flash objects (`interp_sm90a_pffa.cubin`, `interp_sm90a_pfpackedfa.cubin`). Knobs:

| knob | why |
|---|---|
| `PLOW_DECODE_BATCH_LADDER=1,2,4,8,16,32,48,64,96,128` | decode rungs up to 128 concurrent turns |
| `PLOW_MAX_CHUNK=2048`, `PLOW_PF_LADDER_APPEND=64,256` | prefill rungs 64..2048 (prompts 200-2000) |
| `PLOW_EMIT_PREFILL_CUBLASLT=1` | dense prefill projections on cuBLASLt (E4B shapes in `segment_roles.rs`) |
| `PLOW_SEG_FA512=all` | every prefill flash segment (hd256 sliding and hd512 full) isolated in `pffa` |
| `[serve] PLOW_MULTISTEP_ADAPTIVE=true` | single-step while anything prefills, 8-step quanta otherwise |

Two builds from the same recipe (shared tree and a clean worktree) produce a byte-identical
`model.pkt`.

## Model mapping

* **Per-layer inputs (PLE).** Op 155 (`PerLayerInput`) has CPU and Metal arms only. On NVIDIA the
  block is emitted as generic ops: `Gemv` (x·W_gate, `[rows][256]`), `GluStrided` (new op 205:
  `gelu_tanh(gate) * table[r][layer*256 ..]`), `Gemv` (·W_proj, `[rows][2560]`), then the
  post-norm, the residual and the next layer's input norm fused in one `NormResidualNorm`. The
  `[T][42*256]` per-layer table (5.25 GiB bf16) **stays on device**. Decode gathers 21.5 KB per
  token from it, so host streaming would save 5.25 GiB of HBM but put a PCIe round trip in every
  step. Prefill runs the same ops at T rows.
* **KV sharing.** The last 18 layers read an earlier layer's cache. The live-KV manifest dedups
  the cache entries, a repeated reader with identical geometry is accepted (`live_kv.rs`,
  `decode_rung.rs`), and the VMM geometry skips the shared layers (`vmm.rs`).
* **Attention.** Sliding window 512 on 35 hd256 layers and full attention on 7 hd512 layers,
  both with 8 q / 2 kv heads. Sliding decode uses GQA fusion 4 (`PLOW_NV_FA_GF_HD256=4`: every
  KV row is read once for all 4 query heads). Final-logit softcap 30; tied embeddings.
* **Prefix cache.** VMM prefix caching is on by default for window 512 as well as 1024
  (`prefix.rs`). Cached and cold prompts give the same tokens (see below). Sessions
  (`X-Session-Id`) are served from the prefix cache (`X-Session-Cache: prefix-cache`).

## Logprobs API

Served on the CUDA engine. Other backends return 400.

| request | response |
|---|---|
| chat `logprobs: true`, `top_logprobs: 0..=20` | `choices[].logprobs.content[] = {token, logprob, bytes, top_logprobs[]}`; streamed chunks carry their tokens' entries |
| completions `logprobs: 0..=20` | `choices[].logprobs = {tokens, token_logprobs, top_logprobs[], text_offset}` |
| `logprobs_mode: "raw_logprobs"` (default) \| `"raw_logits"` | per request (vLLM names it as a server flag); `raw_logits` puts the raw last-position logits in the same fields |
| `return_tokens_as_token_ids: true` | `token` = `"token_id:<id>"` |

* **Semantics.** Values come from the raw model distribution: the logits row as the packet
  writes it (bf16, after the checkpoint's softcap), before temperature, top-k/p, penalties and
  logit bias. This is vLLM's `raw_logprobs`. The sampled token's entry is always present even
  when it is outside the top list.
* **Raw logits.** `raw_logits` returns the chosen token's logit plus up to 20 alternatives per
  position. The full 262144-entry row is not exposed over HTTP.
* **Not supported.** Prompt logprobs (`echo` / `prompt_logprobs`) are still refused.

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -H 'X-Session-Id: call-17' -d '{
  "model": "gemma-4-e4b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8,
  "temperature": 0, "logprobs": true, "top_logprobs": 3}' | jq '.choices[0].logprobs.content[0]'
# {"token":"Hello","logprob":-0.0312,"bytes":[72,101,108,108,111],"top_logprobs":[{"token":"Hello",...},...]}

curl -s localhost:8000/v1/completions -d '{"model":"gemma-4-e4b","prompt":"The capital of France is",
  "max_tokens":4,"temperature":0,"logprobs":5,"logprobs_mode":"raw_logits","return_tokens_as_token_ids":true}'
```

**Cost.** A row that asks for logprobs downloads its logits row (512 KB bf16) every step. It
also takes the batch off device multi-step for that tick. The host computes the log-sum-exp
(vectorized, AVX2 when present, ~150 µs per row) and the top-k (~60 µs). Greedy rows keep the
device argmax. 

The engine applies three measures in order:

* A greedy, unadjusted row (the voice default) gets its logsumexp, sampled logit and top-k from
  `plow_logprob_stats` (`sample_sm120.cu`) and downloads 43 floats instead of the row.
* Other rows (temperature > 0, penalties, bias) download the row. The host computes the
  log-sum-exp (vectorized, AVX2 when present, ~150 µs per row) and the top-k (~60 µs).
* A logprobs row still leaves device multi-step (one host round trip per token).

Output tok/s, same prompts, `top_logprobs=20` on every request vs none:

| conc | first version (libm exp, host row) | vectorized host | device stats (now) |
|---|---|---|---|
| 1 | -17.8% | -6.3% | -2.7% |
| 32 | -81.7% | -54.0% | -32.0% |
| 64 | -87.1% | -63.6% | -43.4% |

## Correctness vs HF transformers (bf16)

`scripts/llm/gemma_logit_parity.py`:

* It serves 6 chat and 3 completion cases through plow with `top_logprobs=20`.
* HF `Gemma4ForConditionalGeneration` (bf16, eager, same GPU) teacher-forces plow's tokens.
* It compares the per-position distributions.
* It checks prompt ids against `apply_chat_template`: the rendered chat prompt must equal HF's
  ids exactly, including the 1929-token case.

Results on the final build (clean worktree at d9ad0353 + this change; device logprob stats):

| case | prompt ids (HF/plow) | gen | top1 agree | top5 overlap | mean/max abs dlp | KL(HF||plow) mean/max | rel-L2 top20 | greedy identical |
|---|---|---|---|---|---|---|---|---|
| chat0 | 21/21 | 7 | 7/7 | 0.943 | 0.0000/0.0000 | 1.83e-07/1.88e-06 | 0.0071 | 7/7 |
| chat1 | 46/46 | 18 | 16/18 | 1.000 | 0.0043/0.0305 | 3.87e-04/4.06e-03 | 0.0091 | 0/18 |
| chat2 | 76/76 | 9 | 9/9 | 0.978 | 0.0007/0.0032 | 8.15e-05/4.14e-04 | 0.0098 | 9/9 |
| chat3 | 20/20 | 49 | 49/49 | 0.996 | 0.0130/0.1071 | 7.80e-04/7.70e-03 | 0.0098 | 49/49 |
| chat4 | 1929/1929 | 39 | 37/39 | 0.964 | 0.0173/0.1275 | 1.18e-03/6.32e-03 | 0.0130 | 39/39 |
| chat5 | 23/23 | 64 | 64/64 | 0.975 | 0.0096/0.0690 | 4.68e-04/5.31e-03 | 0.0082 | 64/64 |
| cmpl0 | 5/5 | 18 | 17/18 | 0.989 | 0.0086/0.0471 | 5.55e-04/4.61e-03 | 0.0120 | 18/18 |
| cmpl1 | 8/8 | 64 | 63/64 | 0.978 | 0.0029/0.0528 | 1.53e-04/1.91e-03 | 0.0085 | 64/64 |
| cmpl2 | 6/6 | 64 | 62/64 | 0.981 | 0.0145/0.1111 | 1.14e-03/7.59e-03 | 0.0144 | 64/64 |

* Top-1 agreement 324/332 = 0.976. Top-5 overlap 0.980. KL mean 6.5e-4 (max 7.7e-3).
  rel-L2 of the top-20 logprobs ≈ 0.01. (Host-path build: 371/376, KL max 1.6e-2.)
* Every greedy continuation equals HF's except `chat1`.
* `chat1`'s first token is a near-tie: plow 9259 -1.146 vs HF 236777 -1.132, the top two swap
  with a 0.125 gap. Across builds `chat1` or `cmpl0` flips on such ties. This is bf16
  accumulation-order noise, the same size as the |Δlogprob| column.
* `raw_logits`: plow top logit 26.25 vs HF 26.375 (bf16 ulp at 26 is 0.125).
* **Logprobs with `X-Session-Id`.** Turn 2 with 384/395 prompt rows cached gives the same text,
  tokens and logprobs (Δ = 0) as the same turn sent without a session. The logprobs entry count
  equals `completion_tokens`, and the top-5 are sorted with mass ≤ 1.
* **Cached vs uncached prefill.** An uncached vs cached serve of one prompt gives the same
  tokens with |Δlogprob| ≤ 0.11. The prefix kernels differ, so this is the same bf16 noise as
  vs HF.

## Performance (H100 SXM, 132 SMs)

### Decode step (step_bench, ctx 1024)

Roofline = (9.22 GB weights incl. the 1.34 GB tied lm_head + B × 66 MB KV at ctx 1024:
35 sliding × 512 rows × 2 KiB + 7 full × 1024 rows × 4 KiB) / 3.35 TB/s.

| slots | 1 | 8 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|
| step ms | 6.87 | 7.40 | 10.21 | 12.25 | 12.62 | 16.74 | 17.59 |
| roofline ms | 2.83 | 2.96 | 3.44 | 3.75 | 4.07 | 4.70 | 5.33 |
| % of roofline | 41 | 40 | 34 | 31 | 32 | 28 | 30 |
| tok/s (slots / step) | 146 | 1081 | 3134 | 3918 | 5071 | 5735 | 7277 |

### Served: `vllm bench serve` random ISL 1000 / OSL 128, completions, ignore_eos

`scripts/llm/gemma_voice_bench.sh plow|vllm` runs the same client against both servers: TTFT
p50 ms / TPOT p50 ms / output tok/s. vLLM 0.28 runs from `$PYREF` (`--max-num-seqs 256`,
`--max-model-len 8192`). Plow runs the recipe (`--multistep-adaptive`, unified token batch).

The client sends no `temperature`, so every request samples with the checkpoint's generation
config (T=1, top_k=64, top_p=0.95). "greedy" pins `temperature: 0` (`BENCH_ARGS`).

| conc | vLLM sampled | vLLM greedy | plow before (host-sampled rows) | plow sampled | plow greedy (fixed ride) |
|---|---|---|---|---|---|
| 1 | 39.6 / 5.88 / 162 | 36.3 / 5.70 / 168 | 29.4 / 8.07 / 120 | **26.0** / 7.02 / 138 | **25.1** / 6.86 / 141 |
| 8 | 66.9 / 6.13 / 1176 | 62.8 / 5.91 / 1228 | 146 / 9.76 / 741 | 90.5 / 8.27 / 894 | 88.9 / 8.05 / 916 |
| 32 | 232 / 7.61 / 3160 | 215 / 7.22 / 3303 | 414 / 20.9 / 1219 | **167** / 13.0 / 2119 | **168** / 12.7 / 2158 |
| 64 | 270 / 9.91 / 4994 | 254 / 9.09 / 5302 | 1038 / 49.1 / 943 | 335 / 19.5 / 2701 | 357 / 19.0 / 2763 |
| 128 | 483 / 16.1 / 6311 | 381 / 16.5 / 6786 | 2174 / 150 / 692 | 446 / 28.8 / 3398 | 445 / 30.4 / 3251 |
| 200 | 1349 / 33.2 / 3801 | 887 / 16.3 / 7525 | 15357 / 130 / 689 | 3060 / 29.3 / 3555 | 3022 / 28.9 / 3373 |

Measured on the packet before the E-series decode kernels in the table further down (`gemv_k8`
gate, row-group flash decode: greedy c1 141 -> 158, c32 2150 -> 2321, c128 3178 -> 3322).

* **What was wrong before.** Each sampled row that rode a prefill launch downloaded its 512 KB
  logits row and ran the host sampler (a full sort of 262144 entries, ~4 ms per row). At c64 that
  put 264 ms into every mixed tick. Every sampled row now draws on the device (`plow_sample`,
  `launch_sampler_rows` after the packed terminal), including each prompt's first token and the
  serial and compact paths. c64 943 -> 2719 tok/s, c128 692 -> 3398 (with `sched::ride`).
* **`plow_sample`** keeps the host sampler's kept set: exact bf16 radix-select for top-k, the
  top_p cut keeps the token that crosses p, min_p, no f32 scratch (`sample_sm120.cu`). Sampled
  TV distance device vs host on real prefill logits: 0.0024 (T=1, k=64, p=0.95), 0.0000
  (T=0.7, p=0.9), 0.0005 (T=1.3, k=40), 0.0276 (p=0.99, 390 tokens kept; bound 0.105)
  (`tests/gpu_sample_serve.rs`, `tests/gpu_sample.rs`).
* **Where plow stands.** TTFT is ahead at c1 and c32 (c128 within noise of vLLM sampled). From
  c8 up vLLM wins throughput: 1.3x at c8, 1.5x at c32, 1.9-2.3x at c64-c200. c8 TTFT (90 vs
  63 ms) is lost to the decode step: a new prompt waits for the running 8-step quantum.

### Where the tick time goes (`PLOW_PACKLOG`, c64 sampled)

| | mixed ticks (share of wall, mean) | decode ticks | host gap | idle |
|---|---|---|---|---|
| before | 82%, 264 ms | 17% | <1% | <0.5% |
| after | 51%, 53 ms | 48% | <1% | <0.5% |

The device is busy >99% of wall in both. So host overlap (pipelining the next tick's staging)
buys at most 1%, and it was not built. What is left is device time per token:

* **Prefill rate (OSL=1, packed 1000-row prompts).** Plow 21.4 µs/row vs vLLM 13.2 µs/row.
  The prefill-only served rate at c32 is 50 req/s vs 76. Per-op split of a 2000-row launch
  (44 ms, segment-site timing):

  | op | share |
  |---|---|
  | Gemm (cuBLASLt, ~700 TFLOP/s) | 51% |
  | FlashPrefill | 18% |
  | norm / head-norm+RoPE | 16% |
  | Glu | 9% |
  | final norm + lm_head + argmax | 3% |
  | PLE (op 205) | 2% |

  The GEMMs are near vLLM's. The ~10 µs/row of non-GEMM work is the gap.
* **Ceiling.** At ISL 1000 / OSL 128 every output token carries 1000/128 = 7.8 prompt rows,
  195 µs of prefill at 21.4 µs/row. That caps output at ~5100 tok/s with zero decode cost. Add
  the decode step (13.4 ms at B=63, ~18 ms at B=128) and the measured 2700-3300 tok/s is what
  this prefill and step allow. The first wave's TTFT at c64+ is also bound by the prefill rate.
  The throughput gap at c≥32 is a kernel gap (prefill non-GEMM ops, decode step at 30-40% of
  roofline), not a scheduling one.
* **Ride or step.** A decode row riding a prefill launch costs ~0.16 ms. A standalone step costs
  8.5 + 0.077·B ms. Below ~50 rows riding is cheaper, above ~100 the step is. At c128 the
  non-riding arm (`PLOW_PF_NO_INTERLEAVE=1`) served 3405 vs 3097 tok/s. `sched::ride` measures
  both costs per engine and picks per launch: rows ride iff `per-row ride ms × rows ≤ step ms`
  (EWMAs of the observed launches and steps, one launch in 32 explores the other arm).
  `PLOW_RIDE_FIXED=1` restores always-ride. Sampled, adaptive vs fixed (two adaptive runs):

  | conc | fixed | adaptive |
  |---|---|---|
  | 1-64 | 138 / 895 / 2127 / 2719 | 138 / 892-893 / 2118-2122 / 2717-2720 |
  | 128 | 435 / 30.9 / 3193 | 446-459 / 28.8-28.9 / 3391-3398 (+6%) |
  | 200 | 3117 / 30.2 / 3294 | 3060-3077 / 29.3-29.6 / 3555-3558 (+8%) |

  TPOT p99 at c128/c200 drops from 44-45 ms to 39-40 ms.
* **Rejected.** Smaller prefill launches (`PLOW_PF_INTERLEAVE=1024` / `512`) lose 18-35%: they
  pay the launch skeleton more often for the same rows. Host overlap is worth <1% (above).

### Voice sessions: `scripts/llm/session_bench.py`

N calls × 6 streamed turns. Each call shares a ~350-token system prompt, and each turn adds the
previous answers and a new 40-60-token utterance (later prompts ≈ 590 tokens, 87% cached). Every
turn is `max_tokens` 48 with 0.25-0.75 s of think time and `X-Session-Id` per call. vLLM uses
its default automatic prefix caching.

TTFT first turn p50 / later turns p50 (p90) ms, TPOT p50 ms, out tok/s:

| calls | vLLM 0.28 (APC) | plow A: default (token batch) | plow B: `--token-batch=false` | plow A, device-sampled rows |
|---|---|---|---|---|
| 64 | 33 / 47 (77), 8.5, 2322 | 51 / **42 (65)**, 18.6, 1754 | 74 / 143 (293), 28.3, 1334 | 39 / **39 (57)**, 18.7, 1771 |
| 200 | 98 / 202 (488), 17.3, 4943 | 87 / 622 (924), 39.4, 3016 | 1793 / 4522 (4646), 33.2, 1428 | 82 / 657 (841), 40.5, 2977 |

* **Cache hits.** Plow reports 86-87% of later-turn prompt rows cached (`X-Session-Cache:
  prefix-cache`). vLLM caches too but does not report it in usage.
* **Which config.** Voice turns are short suffixes on a cached prefix, and there A (the recipe
  default) wins: at 64 calls the later-turn TTFT beats vLLM. At 200 calls TPOT doubles and
  queueing sets TTFT.
* **Publish rule.** A prompt tail shorter than one VMM block is published only from its second
  sighting (`vmm.rs` `enable_shared_publish`), except for a session (`X-Session-Id`): its
  boundaries publish on the first sighting (`VmmKv::note_session`), so a new call's turn 2 reuses
  its own turn 1 tail. A unique ~380-token first turn sent back-to-back: turn 2 cached 0/397 ->
  352/397 rows.

## Kernel work and where the time goes

B=1 op costs (`step_bench --sweep`, instruction-cap deltas, earlier build of the same program):

| op | per layer | total | vs bandwidth |
|---|---|---|---|
| launch/skeleton (cap 0) | | 1.20 ms | not bandwidth; persistent-interpreter fixed cost |
| GemvGlu gate+up (105 MB/layer) | 36 µs | 1.51 ms | 2.9 TB/s, at roofline |
| down (52 MB/layer) | 33 µs | 1.40 ms | 1.6 TB/s, ~50% |
| lm_head (1.34 GB tied) | | 0.43 ms | 3.1 TB/s |
| NormResidualNorm ×84 | 4.7 µs | 0.40 ms | latency-bound |
| FlashDecode + merge | 12 µs | 0.51 ms | latency-bound at B=1 |
| PLE (gate Gemv, GluStrided, proj Gemv) | 9 µs | 0.38 ms | latency-bound; table gather 21.5 KB/token |

At B=64 FlashDecode is the largest item: 3.2 ms, 76 µs/layer. Sliding hd256 runs ~3x its KV
floor and full hd512 ~1.7x.

| change | effect | status |
|---|---|---|
| NVIDIA PLE decomposition + `GluStrided` (op 205, grid-stride over all CUs) | op 155 had no CUDA arm; GluStrided 12.7 µs/layer at B=64 | in |
| `PLOW_NV_FA_GF_HD256=4` (sliding decode reads each KV row once for 4 q heads) | B=64 13.32 -> 12.63 ms, B=128 18.63 -> 17.59; B=1 +0.17 ms | in |
| device `plow_logprob_stats` + vectorized host log-sum-exp | logprobs cost at c64 -87% -> -43% | in |
| `PLOW_SEG_FA512=all` + cuBLASLt prefill projections | prefill 2000 rows 70 -> 48 ms (single request); packed/cached serve 16 ms | in |
| `PLOW_FUSE_KV_HNR=1` | 604 -> 580 decode insts; step within noise | not adopted |
| `gemv_k8` for hd <= 512 (K-split tensor-core GEMV, 1..32 rows; manifest gate: E-series packets, i.e. `GluStrided` present) | B=1/2/8/16/32 6.93/6.82/7.40/8.50/10.22 -> 6.22/6.25/6.83/7.96/9.65 ms (B=1 via `PLOW_NV_GEMV_K8_MIN=1`) | in |
| `PLOW_NV_FA_RG_WIDE` (hd256/512 flash decode on the row-group body: a warp per row, K and V of 4 rows in flight; E-series gate) | B=1/8/32/64/128 6.22/6.83/9.61/12.63/17.75 -> 6.15/6.63/9.05/11.98/16.42 ms; B=64 sliding 52 -> 40 us/layer, full 153 -> 110 | in |
| `PLOW_NV_GEMV_K8_UNB1=2` (two k32 steps in flight on the gemv_k8 one-tile arm, M <= 8) | B=1/4/8 6.14/6.30/6.62 -> 5.82/6.02/6.36 ms; UNB1 3 and 6 lose 0.2-0.3 ms | in |
| `PLOW_NV_GEMV_K8_MAX_GW=64` (gemv_k8 8-tile arm for 33..64 rows in the `_gw` object) | B=48/64 11.66/11.98 -> 10.99/11.79 ms; B=32/128 unchanged | in |
| RG flash with the next rows' K/V prefetched in registers | stack 272 -> 3744 B (the entry is at the 255-register cap), 2x slower | rejected |
| `gemv_k8` + 64 KiB L2 prefetch | B=1 +0.25 ms, B>=64 +0.2 ms | rejected |
| wide GEMV `_gw` (B >= 48) | already selected by the manifest (`gemv_wide`); B=48 -> 64 costs +0.4 ms only | unchanged |

**PLE table: on device.** It costs 5.25 GiB of HBM. Host streaming would need a pinned host
gather plus an H2D copy of 21.5 KB per token on the critical path of every step and prefill.
The weights (9.4 GB) plus the table fit with 36 GiB of KV at 128 × 8192. For co-serving with
ASR and TTS, lower `max_ctx` (4096 halves the KV) before moving the table.

## Gaps

* **Decode step at 30-41% of roofline.**
  * B=1: the fixed interpreter skeleton (1.2 ms) and down_proj at 50% bandwidth.
  * B>=64: sliding flash decode at ~3x its floor.
  * vLLM's TPOT is lower at c>=8 as a result.
* **Cold-prompt throughput at c>=32 (KERNEL).** Sampled rows now draw on the device, so mixed
  ticks no longer collapse (c64 943 -> 2701 tok/s). vLLM still serves 1.5-2.3x. The limits are
  the prefill rate (21.4 vs 13.2 µs/row, non-GEMM ops) and the decode step (above). No tick
  policy closes this; see "Where the tick time goes".
* **c8 TTFT (SCHED).** 90 ms vs vLLM 63: a new prompt waits for the running multistep quantum.
* **Logprobs and multi-step.** A logprobs row leaves device multi-step; c64 still costs -43%.
  Next steps:
  * batch the stats kernel over all rows of a tick;
  * let it run inside a multi-step quantum;
  * temperature > 0 logprobs rows still download the row.
* **Prompt logprobs** (`echo`, `prompt_logprobs`) are refused. The full raw logits row is not
  exposed over HTTP (the top 20 logits only).
* **Unused segment objects.** The segment script's pfseg/pfgemm objects fault on this packet
  (`CUDA_ERROR_LAUNCH_FAILED`). Only the flash objects are used.
