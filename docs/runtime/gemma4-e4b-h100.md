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

| conc | first version (libm exp, host row) | vectorized host | device stats per row | batched device stats (now) |
|---|---|---|---|---|
| 1 | -17.8% | -6.3% | -2.7% | – |
| 32 | -81.7% | -54.0% | -32.0% (-25.8% re-measured) | -1.2% |
| 64 | -87.1% | -63.6% | -43.4% (-31.6% re-measured) | -1.3% |

The mux gathers every greedy logprobs row of a step (decode, mixed step, token batch) into one
`plow_logprob_stats_rows` launch and one readback (`gpu_batch_logprobs`); the per-row path
remains for inexact top-k and older sampler objects. Re-measured greedy, `--logprobs 20`, same
build: c32 2004 -> 1980 tok/s, c64 2636 -> 2599 (per row: 1487, 1804). Parity with batched
stats: top1 371/376, KL mean 7.1e-4.

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
| step ms | 5.91 | 6.42 | 9.11 | 9.89 | 10.19 | 12.35 | 12.84 |
| roofline ms | 2.83 | 2.96 | 3.44 | 3.75 | 4.07 | 4.70 | 5.33 |
| % of roofline | 48 | 46 | 38 | 38 | 40 | 38 | 42 |
| tok/s (slots / step) | 169 | 1246 | 3512 | 4853 | 6281 | 7773 | 9969 |

Rungs of 48+ rows run their projections on cuBLASLt (see "Decode rungs vs roofline").

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

The same grid on later packets. Cells are TTFT p50 ms / TPOT p50 ms / output tok/s:

* 100166a6: E-series decode kernels, FA3 prefill, `gemv_k8` tuning.
* 54f789dd: adds the KV-shared-tail prefill skip (f8ed564e), where layers 24-41 run only for sampled rows.

| conc | 100166a6 sampled | 100166a6 greedy | 54f789dd sampled | 54f789dd greedy | vLLM greedy |
|---|---|---|---|---|---|
| 1 | 23.8 / 6.03 / 159 | 23.4 / 5.86 / 165 | **19.2** / 6.02 / 161 | **19.0** / 5.86 / 166 | 36.3 / 5.70 / 168 |
| 8 | 78.3 / 7.21 / 1025 | 76.7 / 6.95 / 1060 | **56.0** / 6.97 / 1074 | **55.7** / 6.76 / 1105 | 62.8 / 5.91 / 1228 |
| 32 | 162 / 11.6 / 2377 | 152 / 11.2 / 2454 | **128** / 10.7 / 2626 | **129** / 10.5 / 2694 | 215 / 7.22 / 3303 |
| 64 | 301 / 17.6 / 3041 | 298 / 16.9 / 3117 | 270 / 15.8 / 3447 | 264 / 15.4 / 3525 | 254 / 9.09 / 5302 |
| 128 | 890 / 24.2 / 3570 | 399 / 25.6 / 3838 | 402 / 23.4 / 4274 | 387 / 23.0 / 4381 | 381 / 16.5 / 6786 |
| 200 | 4013 / 26.9 / 3493 | 2555 / 25.4 / 4010 | 2105 / 23.9 / 4385 | 2048 / 22.3 / 4538 | 887 / 16.3 / 7525 |

On 54f789dd:

* Prefill-only (OSL=1): c1 19.0 ms, c32 77.1 req/s, c64 81.8 req/s, i.e. 12.2 µs/row served,
  against vLLM's 75.7 req/s at c32. A tick is 25.9 ms per 1979 packed rows, with a 1.0% host gap.
* Sessions (`session_bench.py`, sampled): 64 calls, later-turn TTFT p50 33.7 ms (p90 64), TPOT
  15.2 ms, 1943 tok/s. 200 calls: 406 ms (p90 578), TPOT 32.3 ms, 3572 tok/s.
* Plow's TTFT now leads vLLM from c1 through c32 (c8 56 vs 63 ms). vLLM's throughput leads by
  1.23x at c32, 1.5x at c64, 1.55x at c128 and 1.65x at c200.
* **The vLLM columns at c64 and up are inflated by prefix-cache hits.** Every cell reused the
  previous cell's prompts (one `--seed`). With unique prompts per cell (`pb_bench` now seeds per
  cell), greedy c64 is 3438 vs 4187 tok/s (1.22x) and c128 3973 vs 5103 (1.28x).
  [throughput-audit.md](throughput-audit.md) has the accounting.
* c64 `PLOW_PACKLOG`: decode ticks are 54% of the device time (49 rows, 52.7 ms per multistep
  tick), mixed ticks 41% (1922 prefill rows + 51 riders, 31.4 ms), host 0.3%.
* Riding costs ~0.09 ms/row against ~0.2 ms/row for a separate step, so `sched::ride` keeps
  riding. Smaller launches that would let more ticks carry riders (`PLOW_PF_INTERLEAVE=1024`)
  still lose 15-22%: c32 / c64 / c128 2093 / 2659 / 3297 vs 2453 / 3313 / 4231 tok/s, two runs
  each.
* The gap is the decode step (KERNEL).

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
* **Where plow stood at 100166a6.** TTFT was ahead at c1 and c32. From c8 up vLLM won
  throughput: 1.2x at c8, 1.3x at c32, 1.7x at c64-c128 (greedy).
* **c8 TTFT (77 vs 63 ms at 100166a6; 56 ms after the tail skip) was the prefill rate, not the
  quantum.**
  * Cutting a running multistep quantum on arrival measured nothing: c8 TTFT 76.0 vs 77.0 ms,
    c1 TPOT unchanged. Adaptive multistep already runs single steps while an arrival is likely.
  * Per request, the closed-loop client keeps prompts arriving in groups of 2-8. vLLM prefills a
    group of 8 x 1000 rows in one ~60 ms step (every request at 62-64 ms).
  * Plow runs 2048-row launches (~36 ms each), so a group finishes pair by pair: 31, 62, 77 ms.
  * Only a faster prefill closes it: 17.4 µs/row in-kernel against the ~8 µs/row vLLM's
    one-step group implies.
* **Prefill-only launches carry no scheduler overhead.**
  * Served OSL=1 at c64 runs 58 req/s = 17.2 µs/row. Launches are packed full, and partial
    chunks fill the tail (`chunks=[72,1000,976]` in 2048 rows).
  * A tick is 35.5-36.6 ms per 2048 rows against the 34.8 ms kernel launch. The host gap is 0.8%.
  * The older 21.3 µs/row served figure predates the FA3 prefill.
* **TPOT at c64-c128 is throughput, not tick policy.**
  * In a closed loop at fixed concurrency, E2E = c x 128 / throughput. At c64 that is 2.69 s,
    and TTFT + 127 x TPOT must fit in it.
  * vLLM's 9.1 ms TPOT comes with 5302 tok/s. A tick policy can only move time between TTFT
    and TPOT.
  * At c64 the device is busy 99.7% of wall: 48% mixed ticks (1922 prefill rows + 51 riders,
    42.5 ms), 47% decode, 0.3% host.
  * Riding costs 0.13 ms/row against a 0.2 ms/row step at B=51, so the ride model picks right.
  * Per request, 1000 prefill rows are 17.4 ms of device and 128 decode tokens at B~50 are
    ~28 ms. That bounds c64 near 2900-3100 tok/s, and plow measures 3041-3117.

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

## Voice co-serving: ASR + E4B + Chatterbox-MTL on one H100

One `plowrt serve --assets <qwen3-asr> --assets <chatterbox-mtl> --assets <gemma-4-e4b>` (LLM
last: its KV admission budget is sampled from what the others leave). Load
`scripts/voice/call_sim.py` (streaming ASR appends, chat turn, streaming TTS, `X-Session-Id`, 3
turns, `--language en`).

**Memory.** Every packet reserves `[slots][ctx]` KV at its emitted ceiling: ASR 28 GiB, MTL
30 GiB, and E4B ~36 GiB resident (sliding rings of 4096 rows × 128 slots are 20 GiB of it). That
is 94 GiB plus ~7-8 GiB each for the ASR encoder and the S3Gen vocoder, so the planner swapped
models on every turn (S1 thrash, `cuMemAlloc` OOM in the vocoder). Per model live context bounds
fit the three in 73 GiB:

```sh
PLOW_LIVE_CTX_MODELS=qwen3-asr=768,chatterbox-mtl=512   # --live-ctx-models
```

* `ctx_bound::narrow` takes non-indexer packets below 2048 (the 2048 floor is the DSA indexer's
  selection width). The planner narrows the same way, so it plans what the engine loads.
* A speech job's `max_tokens` is its packet's cap (ASR 1024 transcript tokens, MTL 1000 speech
  tokens), not the client's, so under a narrowed bound the mux fits it to the context instead of
  refusing the request (`fit_speech_budget`). At 768 an ASR prompt of 29 s of audio leaves ~330
  transcript tokens; at 512 an MTL utterance keeps ~400 speech tokens (16 s).

**Scheduling.** The persistent cooperative grid takes the whole device, so the three models
time-share it. `--co-sched deadline` gives the turn by urgency:

1. A deadline: a prompt owed its first token (LLM TTFT, an ASR final, a speech stream's first
   audio), or a critical job in its first second.
2. Decode throughput.
3. Partial transcripts (bulk).

A waiter moves up one class per 100 ms waited. A holder keeps the device for 20 ms while a peer
of its class waits, and runs single-step ticks while outranked.

call_sim p50/p95 ms (ASR final, LLM TTFT, TTS time to first audio), and turns with >100 ms
playback underrun:

| calls | free | rr | deadline |
|---|---|---|---|
| 10 | 379/1422, 46/201, 404/1421, 2/30 | 308/1138, 27/318, 493/1442, 7/30 | 107/957, 16/52, 271/687, 5/30 |
| 20 | 900/2605, 173/852, 711/2495, 29/60 | 609/2205, 32/580, 972/2329, 46/60 | 203/894, 41/351, 421/1165, 32/60 |
| 30 | 2292/5280, 252/776, 703/2620, 43/90 (4 errors) | 1268/2858, 41/730, 1489/5978, 80/90 | 546/2424, 135/875, 1443/5442, 61/90 |
| 50 | 1930/8557, 355/1223, 1987/9296, 117/150 (10 errors) | 2611/5361, 245/1437, 2732/3913, 138/150 | 699/6705, 110/1100, 819/3433, 128/150 |

Deadline turns cut the p50s 2-4x at 10-20 calls and meet the TTFT and TTFA SLOs at 10 calls.
The ASR final p95 and underrun SLOs fail from 10 calls, and everything fails from 20.

With cheap ASR partials (forced prefix, duty cycle; [sessions.md](sessions.md)) and a deadline
co-tenant cutting a running LLM multistep quantum, measured on the current E4B packet and the
windowed-render MTL assets. Each cell is p50/p95 ms for ASR final, ASR partial, LLM TTFT and TTS
TTFA, then underrun turns and the run's wall time:

| calls | HEAD 100166a6 | + sched_6 |
|---|---|---|
| 10 | 114/621, 228/1005, 17/196, 187/504, 2/30, 82 s | 108/583, 34/751, 14/413, 192/409, 0/30, 83 s |
| 20 | 181/809, 573/1907, 39/746, 254/1135, 13/60, 106 s | 405/1066, 29/1587, 65/666, 345/1309, 24/60, 90 s |
| 30 | 316/1574, 538/1726, 100/775, 428/1674, 45/90 (5 errors), 120 s | 319/1783, 116/602, 202/1154, 495/1846, 72/90, 93 s |

* With HEAD, slow partials held each call's append loop back, and 20-30 calls ran 16-29% longer
  (wall).
* With partials back at the 1 s audio cadence, the same calls offer the load a real user would.
  The TTS-bound SLOs then look worse per turn at 20-30 calls, while ASR device time per turn
  falls 27-30%.
* The binding limit at 54f789dd was still the TTS render.

On 6492ac30 (windowed streaming TTS, MTL assets with the re-emitted `s3gen.pkt`, E4B packet
re-emitted with the KV-shared-tail skip), `--co-sched deadline`:

| calls | ASR final | ASR partial | LLM TTFT | TTS TTFA | underrun | errors | wall | SLOs |
|---|---|---|---|---|---|---|---|---|
| 10 | 95/388 | 36/491 | 14/166 | 171/774 | 0/30 | 0 | 78 s | **pass** |
| 20 | 180/561 | 32/454 | 37/656 | 249/899 | 0/60 | 0 | 85 s | TTFA p95, ASR p95 |
| 30 | 353/789 | 71/349 | 85/531 | 806/2406 | 4/90 | 0 | 85 s | TTFA, ASR p95 |
| 50 | 464/1353 | 149/734 | 88/598 | 708/2194 | 0/150 | 0 | 89 s | TTFA, ASR p95 |
| 100 | 1216/3282 | 218/678 | 128/1728 | 2720/4455 | 245/300 | 0 | 112 s | all but partials |

* Windowed render moves the playback limit out to 50-100 calls: 0 underruns through 50 calls,
  where every earlier build underran on most turns.
* The device is 65% busy in mux ticks at 50 calls and 82% at 100.
* Through 50 calls the SLOs lost are the p95 tails of ASR final (≤500 ms) and TTS first audio
  (≤800 ms), while the medians pass.

**Turn order for the finals (sched_8).** A traced ASR final at 30-50 calls (debug log `ASR
completed (serve mux)`: encoded / submitted / first_token / total) spent its time decoding. The
median was 330 ms and p95 1.9 s from first token to done, for ~22 tokens. Each token needed a
turn, and the final's urgency tied with every LLM prompt and TTS stream start.

* `JobClass::Final` / `Urgency::Final`: an ASR final outranks every other deadline. The
  first-token-to-done time drops to 64 / 263 ms (p50 / p95).
* Finals go ahead of partials in the encoder queue.
* A critical speech stream keeps deadline urgency for its first 48 tokens as well as its first
  second.
* The S3Gen render takes the device turn like a mux tick (`DownstreamCredit::device_turn`). A
  launch with a first chunk, or a stream at or under its playback slack, takes it at deadline
  urgency, otherwise normal. This only happens under `--co-sched deadline`, which replaces the
  LM pause (`set_urgent`) there.
* Rejected: running the ASR encoder under the turn. The final's encode p50 went 52 -> 212 ms,
  because concurrent with other models' ticks it costs less than waiting them out.

call_sim at 5eea79c4 vs + sched_8, `--co-sched deadline`. Cells are p50/p95 ms, with SLO passes
in bold:

| calls | build | ASR final | LLM TTFT | TTS TTFA | underrun |
|---|---|---|---|---|---|
| 30 | HEAD | 383/945 | 153/**510** | 423/1338 | 0/90 |
| 30 | + sched_8 | 160/**428** | 44/**351** | 412/1024 | 0/90 |
| 50 | HEAD | 501/1680 | 178/1016 | 658/1621 | 3/150 |
| 50 | + sched_8 | 193/546 | 84/**375** | 541/1043 | 2/150 |
| 100 | HEAD | 1255/3492 | 368/2881 | 2219/4043 | 227/300 |
| 100 | + sched_8 | 205/1356 | 150/**490** | 1748/3440 | 215/300 |

* LLM total time per turn rises (1.3 -> 1.5 s at 30 calls, 1.4 -> 2.4 s at 50): the decode yields
  to the finals and the stream starts.
* Chatterbox-MTL alone, streamed, two interleaved runs each, audio s/s:

  | | c64 | c128 |
  |---|---|---|
  | HEAD, free | 33.8 / 31.1 | 39.3 / 35.9 |
  | + sched_8, free | 34.2 / 33.9 | 39.2 / 36.1 |
  | + sched_8, `--co-sched deadline` | 33.1 / 31.3 | 35.8 / 35.3 |

  Free (the default) is unchanged. Deadline is within the run-to-run spread at c64 and ~5%
  under the free mean at c128. 0 failures.
* What remains through 50 calls is TTS first audio (p95 ~1.0 s against 0.8 s) and the final's
  encode tail (frontend + encoder queue, server-side p95 553 ms at 50 calls).

**Render launches bound the final's tail (sched_9).** Traced finals (debug logs `asr: front +
encode`, `asr: single|packed encoder launch` with gpu_ms/wall_ms, `vocoder render`):

* The frontend costs 1.2 / 2.8 ms (p50 / p95), and host copies ~1 ms per launch. Neither is the
  tail.
* An encoder launch is ~10 ms of work, yet its event-timed GPU span reaches 240-380 ms p95. The
  encoder is a cooperative grid, so it waits until the device is free.
* The render is what it waits behind. A vocoder launch is one cooperative grid: 1 stream takes
  140 ms, 16 take 440 ms, and 64 take 1.35-1.43 s. While it runs, the render also holds the
  device turn, so the final's prefill and decode ticks wait too (submit -> first token p95
  500 ms at 100 calls).
* `--tts-turn-batch` / `PLOW_TTS_TURN_BATCH` (default 16, only under `--co-sched deadline`, 0 =
  the packet's largest capacity) caps the streams in one render launch. A launch then holds the
  device for at most ~460 ms.
* Rejected: the encoder on the device turn at `Urgency::Final`. Final p95 went 444 -> 616 ms at
  50 calls and 1134 -> 1575 ms at 100, because the encoder waits out whole ticks and renders.
* Rejected: the encoder on a greatest-priority CUDA stream. A pending grid cannot preempt a
  running one, so there was no gain (1088 ms at 100 calls). At 30-50 calls p50 went up 20-40 ms.
* Rejected: a cap of 8 (at 5eea79c4+). ASR final p95 at 100 calls improves to 690 ms, but TTS
  first audio p95 gets worse than with a cap of 16 (3.1 vs 2.5 s).

call_sim at 827ab0b3, `--co-sched deadline`, seed = call count, two interleaved runs each. Cells
are p50/p95 ms (ASR final, LLM TTFT, TTS TTFA) and underrun turns:

| calls | cap | ASR final | LLM TTFT | TTS TTFA | underrun |
|---|---|---|---|---|---|
| 30 | 0 (HEAD) | 162/348, 168/428 | 100/286, 51/248 | 358/1036, 361/930 | 0, 0 |
| 30 | 16 | 202/446, 162/455 | 70/283, 91/228 | 419/1000, 431/976 | 1, 0 |
| 50 | 0 (HEAD) | 224/586, 209/490 | 93/337, 69/274 | 664/3207, 488/944 | 5, 0 |
| 50 | 16 | 193/506, 234/484 | 62/380, 92/338 | 549/1031, 452/933 | 11, 0 |
| 100 | 0 (HEAD) | 277/911, 221/940 | 199/4157, 129/707 | 2501/4900, 1680/2852 | 193, 181 |
| 100 | 16 | 245/**644**, 223/**635** | 128/824, 121/685 | 1244/2222, 1211/3017 | 225, 197 |

* At 100 calls the render launch p95 drops from 444 ms to 274 ms (max 1432 -> 463 ms), and the
  final's encode p95 from 388 to 182 ms. ASR final p95 meets the 800 ms target. TTS first audio
  p50 drops by 0.5-1.3 s.
* At 100 calls underruns rise by 12-32 turns. A launch of 16 costs more per stream than one of
  64, and TTS is already over capacity there.
* At 30-50 calls the cap rarely binds, and the cells sit inside the run-to-run spread. Four
  runs of the same binary at 5eea79c4+ put the 50-call ASR final p95 at 444-711 ms.
* The ≤500 ms p95 at 50 calls is borderline: 506 and 484 ms. What remains there is the
  prefill and decode waits behind renders and E4B ticks (submit -> first token p95 ~280 ms).
  It needs shorter render grids, a render that yields between flow steps, or both.

**Capacity, not order, is the limit.**

* A call_sim turn is ~14 s: ~7 s of user speech, ~1 s for the ASR final and the LLM, 5.1 s of
  reply audio, 1 s of think time.
* 200 calls therefore need ~72 s of Chatterbox-MTL audio per second in real time.
* MTL streaming tops out at ~12.5 audio-s/s with the whole device ([tts.md](tts.md)): each chunk
  re-renders the utterance's whole token prefix through S3Gen.
* Mux ticks alone keep the device 50-58% busy at 10-20 calls, and the vocoder renders run on top
  of them outside the turn.
* 200 calls need the TTS render ~6x cheaper. Streaming partials make ASR the next cost (~0.3 s of
  device per turn at 50 calls).
* E4B is the smallest share: ~64 tokens per turn at low batch.

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
| `plow_logprob_stats_rows`, one launch per step from the mux | logprobs cost at c64 -32% -> -1.3% | in |
| `PLOW_SEG_FA512=all` + cuBLASLt prefill projections | prefill 2000 rows 70 -> 48 ms (single request); packed/cached serve 16 ms | in |
| `PLOW_FUSE_KV_HNR=1` | 604 -> 580 decode insts; step within noise | not adopted |
| `gemv_k8` for hd <= 512 (K-split tensor-core GEMV, 1..32 rows; manifest gate: E-series packets, i.e. `GluStrided` present) | B=1/2/8/16/32 6.93/6.82/7.40/8.50/10.22 -> 6.22/6.25/6.83/7.96/9.65 ms (B=1 via `PLOW_NV_GEMV_K8_MIN=1`) | in |
| `PLOW_NV_FA_RG_WIDE` (hd256/512 flash decode on the row-group body: a warp per row, K and V of 4 rows in flight; E-series gate) | B=1/8/32/64/128 6.22/6.83/9.61/12.63/17.75 -> 6.15/6.63/9.05/11.98/16.42 ms; B=64 sliding 52 -> 40 us/layer, full 153 -> 110 | in |
| `PLOW_NV_GEMV_K8_UNB1=2` (two k32 steps in flight on the gemv_k8 one-tile arm, M <= 8) | B=1/4/8 6.14/6.30/6.62 -> 5.82/6.02/6.36 ms; UNB1 3 and 6 lose 0.2-0.3 ms | in |
| `PLOW_NV_GEMV_K8_MAX_GW=64` (gemv_k8 8-tile arm for 33..64 rows in the `_gw` object) | B=48/64 11.66/11.98 -> 10.99/11.79 ms; B=32/128 unchanged | in |
| cuBLASLt projections on the 48+ row rungs (`PLOW_EMIT_DECODE_CUBLASLT=1`, `PLOW_EMIT_DECODE_CUBLASLT_MIN_ROWS=48`) | B=48/64/96/128 11.0/11.9/15.6/16.4 -> 9.9/10.2/12.3/12.8 ms; served greedy (unique seed per cell) c64 3516 -> 3708 tok/s (TPOT 16.5 -> 15.7 ms), c128 3974 -> 4301 (29.0 -> 27.1); c1-c32 unchanged; parity with 60 background streams top1 0.984, KL 7.1e-4 | in |
| the same route from 32 rows, or with k/v/PLE projections left in the interpreter (4 MiB floor) | B=32 9.11 -> 9.68; B=64/128 10.19/12.84 -> 10.31/14.54 (the PLE gate is 1.2 ms native at B=128) | rejected |
| stream-K fixup by the last contributor (`_gw` wide GEMV) | standalone GLU M=64 57 -> 51 us (down unchanged); in the entry B=32..128 +0.2..+1.7 ms | rejected |
| flash KV pulled into L2 before the gate (128 KiB..1 MiB per slice) / KV loads `evict_last` or default-cached | B>=32 +0.04..+0.3 ms / +0.3..+1.4 ms (weight streaming loses its L2) | rejected |
| RG flash with the next rows' K/V prefetched in registers | stack 272 -> 3744 B (the entry is at the 255-register cap), 2x slower | rejected |
| `gemv_k8` + 64 KiB L2 prefetch | B=1 +0.25 ms, B>=64 +0.2 ms | rejected |
| wide GEMV `_gw` (B >= 48) | already selected by the manifest (`gemv_wide`); B=48 -> 64 costs +0.4 ms only | unchanged |

**PLE table: on device.** It costs 5.25 GiB of HBM. Host streaming would need a pinned host
gather plus an H2D copy of 21.5 KB per token on the critical path of every step and prefill.
The weights (9.4 GB) plus the table fit with 36 GiB of KV at 128 × 8192. For co-serving with
ASR and TTS, lower `max_ctx` (4096 halves the KV) before moving the table.

## Prefill per op (packed, one cold request)

Measured is `PLOW_PF_SEG_TIME` segment wall time, which includes the 4-6 µs per-segment
interpreter/launch floor (a body-less skeleton build runs the 255 light segments in 1.17 ms).
Roofline is max(FLOP / 989 TF/s, bytes / 3.35 TB/s). The rows are one non-shared sliding layer,
plus the hd512 flash of a full layer. Build: `PLOW_TMA_GEMM=1`, `PLOW_SEG_CLASS_SLICE=1`, v3
flash, FATLITE light object, cuBLASLt projections.

| op (per layer) | rows | GFLOP | MB | roofline µs | measured µs | % of roofline | implementation |
|---|---|---|---|---|---|---|---|
| q_proj GEMM | 1000 | 10.5 | 19.7 | 10.6 | 23 | 46% | cuBLASLt |
| k_proj GEMM | 1000 | 2.6 | 8.8 | 2.7 | 12 | 22% | cuBLASLt |
| v_proj GEMM | 1000 | 2.6 | 8.8 | 2.7 | 12 | 22% | cuBLASLt |
| HeadNormRope q/k/v | 1000 | 0.0 | 13.3 | 4.0 | 19 | 21% | interp (FATLITE) |
| FlashPrefill hd256 sliding | 1000 | 4.2 | 12.3 | 4.2 | 50 | 8% | interp v3 (pffa) |
| FlashPrefill hd512 full | 1000 | 8.2 | 20.5 | 8.3 | 106 | 8% | interp v3 (pffa) |
| o_proj GEMM | 1000 | 10.5 | 19.7 | 10.6 | 25 | 42% | cuBLASLt |
| NormResidual+RmsNorm | 1000 | 0.0 | 20.5 | 6.1 | 28 | 22% | interp |
| gate_proj GEMM | 1000 | 52.4 | 78.0 | 53.0 | 76 | 70% | cuBLASLt |
| up_proj GEMM | 1000 | 52.4 | 78.0 | 53.0 | 74 | 72% | cuBLASLt |
| GeGLU | 1000 | 0.0 | 61.4 | 18.3 | 39 | 47% | interp |
| down_proj GEMM | 1000 | 52.4 | 78.0 | 53.0 | 74 | 72% | cuBLASLt |
| NormResidual | 1000 | 0.0 | 15.4 | 4.6 | 19 | 24% | interp |
| PLE gate GEMM | 1000 | 1.3 | 6.9 | 2.1 | 11 | 19% | cuBLASLt |
| PLE GluStrided (205) | 1000 | 0.0 | 1.5 | 0.5 | 12 | 4% | interp |
| PLE proj GEMM | 1000 | 1.3 | 6.9 | 2.1 | 9 | 23% | cuBLASLt |
| NormResidual+RmsNorm (layer end) | 1000 | 0.0 | 20.5 | 6.1 | 22 | 28% | interp |
| q_proj GEMM | 2000 | 21.0 | 28.9 | 21.2 | 40 | 53% | cuBLASLt |
| k_proj GEMM | 2000 | 5.2 | 14.9 | 5.3 | 17 | 31% | cuBLASLt |
| v_proj GEMM | 2000 | 5.2 | 14.9 | 5.3 | 16 | 33% | cuBLASLt |
| HeadNormRope q/k/v | 2000 | 0.0 | 26.6 | 7.9 | 30 | 26% | interp (FATLITE) |
| FlashPrefill hd256 sliding | 2000 | 8.4 | 24.6 | 8.5 | 54 | 16% | interp v3 (pffa) |
| FlashPrefill hd512 full | 2000 | 32.8 | 41.0 | 33.1 | 199 | 17% | interp v3 (pffa) |
| o_proj GEMM | 2000 | 21.0 | 28.9 | 21.2 | 38 | 56% | cuBLASLt |
| NormResidual+RmsNorm | 2000 | 0.0 | 41.0 | 12.2 | 38 | 32% | interp |
| gate_proj GEMM | 2000 | 104.9 | 103.6 | 106.0 | 142 | 75% | cuBLASLt |
| up_proj GEMM | 2000 | 104.9 | 103.6 | 106.0 | 138 | 77% | cuBLASLt |
| GeGLU | 2000 | 0.0 | 122.9 | 36.7 | 66 | 56% | interp |
| down_proj GEMM | 2000 | 104.9 | 103.6 | 106.0 | 136 | 78% | cuBLASLt |
| NormResidual | 2000 | 0.0 | 30.7 | 9.2 | 25 | 37% | interp |
| PLE gate GEMM | 2000 | 2.6 | 12.6 | 3.8 | 11 | 34% | cuBLASLt |
| PLE GluStrided (205) | 2000 | 0.0 | 3.1 | 0.9 | 15 | 6% | interp |
| PLE proj GEMM | 2000 | 2.6 | 12.6 | 3.8 | 11 | 34% | cuBLASLt |
| NormResidual+RmsNorm (layer end) | 2000 | 0.0 | 41.0 | 12.2 | 34 | 36% | interp |

pf_4 (current build; nsys kernel time, same layer; `*` = changed by pf_4):

| op (per layer) | rows | roofline µs | measured µs | % | implementation |
|---|---|---|---|---|---|
| q_proj GEMM | 1000 | 10.6 | 17 | 62% | cuBLASLt |
| k+v proj GEMM * | 1000 | 5.3 | 12 | 44% | cuBLASLt strided batch of 2 (was 2 × 12) |
| HeadNormRope q/k/v | 1000 | 4.0 | 20 | 20% | interp, 264 slices |
| NormResidualNorm (post-attn) * | 1000 | 6.1 | 17 | 36% | interp row teams (was NR+RmsNorm 28) |
| gate+up GEMM * | 1000 | 106 | 139 | 76% | cuBLASLt strided batch of 2 (was 76 + 74) |
| GeGLU * | 1000 | 18.3 | 28 | 65% | interp, 264 slices (was 39) |
| down_proj GEMM | 1000 | 53.0 | 69 | 77% | cuBLASLt |
| NormResidual * | 1000 | 4.6 | 13 | 35% | interp row teams (was 19) |
| PLE GluStrided (205) * | 1000 | 0.5 | 9 | 6% | interp (was 12) |
| NormResidualNorm (layer end) * | 1000 | 6.1 | 15 | 42% | interp row teams (was 22) |
| q_proj GEMM | 2000 | 21.2 | 29 | 73% | cuBLASLt |
| k+v proj GEMM * | 2000 | 10.6 | 16 | 66% | cuBLASLt strided batch of 2 (was 17 + 16) |
| HeadNormRope q/k/v | 2000 | 7.9 | 30 | 26% | interp |
| NormResidualNorm (post-attn) * | 2000 | 12.2 | 26 | 47% | interp row teams (was 38) |
| gate+up GEMM * | 2000 | 212 | 267 | 79% | cuBLASLt strided batch of 2 (was 142 + 138) |
| GeGLU * | 2000 | 36.7 | 51 | 72% | interp (was 66) |
| NormResidual * | 2000 | 9.2 | 19 | 48% | interp row teams (was 25) |
| NormResidualNorm (layer end) * | 2000 | 12.2 | 22 | 55% | interp row teams (was 34) |

What pf_4 changed:
* **Paired projections** (`PLOW_LT_PAIR`, on by default). This is generic across models. Two
  adjacent cuBLASLt projection segments that have the same shape and the same input, and
  write disjoint outputs, run as one strided-batch matmul. The input's batch stride is 0.
  The algorithm is timed at load on the live operands. The second segment becomes a no-op.
  * On E4B this pairs k/v and gate/up.
  * On Veena (Llama, 28 layers) it pairs 56 segments.
  * Veena in-kernel prefill: 500 rows 9.62 → 9.37 ms, 1000 rows 21.2 → 20.7 ms.
* **Workspace memset hoist.** cuBLASLt's captured workspace memset now runs beside the
  preceding interpreter segment instead of after it. Removes the ~7 µs bubble before each
  paired GEMM.
* **Light slices** (`PLOW_SEG_SLICE_ALL=1` in the recipe). Light ops were emitted with n_cu
  (132) slices for the occ-2 FATLITE grid of 264, so half of the grid idled. GLU went
  39 → 28 µs.
* **Row teams + prefill seam fusion** (`PLOW_PF_GFUSE=1` in the recipe, row-team body behind
  `PLOW_NV_PF_ROW_TEAM`).
  * Each row is owned by a team of 128 threads. Every operand of the row is loaded in one
    round trip.
  * Replaces T17 warp-per-row, which ran 2–3 dependent load batches per pass and parked
    half the warps.
  * The final seam stays unfused, because the packed terminal starts at the final RmsNorm.
  * Reduction order differs from T17: parity 0.992 / KL 7.2e-4 (gate: top-1 >= 0.97, KL < 1e-3).
    Greedy text changes on one of the three probe prompts (hash 6e26e0ab -> 68b329da);
    paired projections alone keep all three hashes.

**Where a light segment's time goes** (pf_5; `-DPLOW_NV_ENTRY_TRACE=1` object with
`PLOW_PF_ENTRY_TRACE=1`: per-block `%globaltimer` stamps, medians over 264 blocks, 1000 rows):

| segment | nsys µs | block start spread | claim | entry + inst | gate | body (first item) | exit |
|---|---|---|---|---|---|---|---|
| NormResidual | 10.6 | 0.6 | 0.4 | 0.6 | 0.1 | 7.8 | 0.4 |
| PLE GluStrided (205) | 7.5 | 0.3 | 0.4 | 0.6 | 0.1 | 2.3 | 0.4 |
| NormResidualNorm | 12.7 | 0.3 | 0.4 | 0.6 | 0.1 | 9.6 | 0.5 |
| HeadNormRope q,k,v (3 items/block) | 15.3 | 0.5 | 0.4 | 0.6 | 0.1 | 7.0 | 7.2 (items 2–3) |
| NormResidualNorm | 15.3 | 0.8 | 0.4 | 0.6 | 0.1 | 11.6 | 0.5 |
| GeGLU | 27.4 | 0.7 | 0.5 | 0.6 | 0.1 | 25.0 | 0.8 |

* The fixed interpreter cost is ~2.5 µs per segment:
  * ~1 µs launch (nsys duration minus the in-kernel span);
  * ~0.5 µs block start spread;
  * ~1 µs of dependent loads: segment window, claim atomic, stream entry, instruction.
* The rest is op body. Most bodies are latency-bound: 2–3 dependent HBM round trips per row
  or head, not bandwidth.
* PDL cannot hide the launch here. The gap between an nvjet node and the next interpreter node
  is already 0.37 µs, with or without PDL, because nvjet does not trigger its dependents early.

What pf_5 changed:
* **Segment gates** (`PLOW_PF_SEGMENT_GATES`, on; all models; `exec/gpu/segment_gates.rs`).
  In a multi-segment prefill bucket, every segment is its own ordered launch. At load, the
  runtime drops:
  * each wait on a producer in an earlier segment;
  * each counter bump that no same-segment consumer waits on.

  Per work item that removes the counter poll and its acquire fence, plus the release fence
  and the atomic bump. Results:
  * E4B HeadNormRope 20 → 17 µs;
  * E4B 1000 rows −0.28 ms;
  * Veena 500 rows 9.38 → 9.18 ms, 1000 rows 20.72 → 20.45 ms.
* **PDL helper, not wired into prefill.** The helper is `CudaBackend::launch_cooperative_pdl`
  plus `function_waits_on_pdl`, and on the device `plow_pdl_wait()` in `sm120_common.cuh`.
  Every interpreter entry now waits first and exports `plow_pdl_wait_1`.
  * Launching each prefill interpreter segment that follows a kernel node programmatically saved
    only ~0.04 ms.
  * It also shifted E4B numerics deterministically: parity 0.985 → 0.974 over 3 runs, versus
    0.985 with it off. The cause is not identified, so prefill does not use it.
  * Any PDL user should gate on parity.
* **HeadNormRope, hd256:** the rope-table fetch (`pos[t]` → cos/sin) is issued with the x
  fetch instead of after the norm: 16.8 → 15.3 µs at 1000 rows, 26 → 22.6 µs at 2000.
* **NormResidualNorm row teams** load `gn` with the other operands
  (`PLOW_NV_PF_TEAM_GN_EARLY`): one load round trip per pass instead of two.
* **Tried and dropped:**
  * static first claim (block b takes item b; no exit atomic): neutral;
  * two (token, head) tasks per HeadNormRope warp iteration: slower, because the 128-register
    FATLITE object spills;
  * q beside the k/v pair on a side stream inside the capture: slower (13.36 → 13.48 ms), the
    two GEMMs slow each other more than they overlap.

Whole launch (in-kernel):

| rows | all 42 layers on every row | tail as prefill segments (pf_2) | tail as a decode step | pf_4 | pf_5 |
|---|---|---|---|---|---|
| 1000 | 21.3 ms (21.3 µs/row) | 17.3 ms | 14.7 ms (14.7 µs/row) | 13.7 ms (13.7 µs/row) | 13.3 ms (13.3 µs/row) |
| 2000 | 34.7 ms (17.4 µs/row) | 25.2 ms | 22.4 ms (11.2 µs/row) | 20.9 ms (10.5 µs/row) | 20.6 ms (10.3 µs/row) |

pf_4 → pf_5 measured back to back on one GPU: 1000 rows 13.59 → 13.31 ms (13.3 µs/row), 2000 rows 20.74 →
20.62 ms. At 1000 rows the remaining time is:

| part | ms |
|---|---|
| KV-shared tail decode step | 2.7 |
| flash | 1.3 |
| cuBLASLt GEMMs | 6.4 |
| light ops, ~2.5 µs fixed per segment × 146 | 2.2 |

Reaching 12 µs/row needs the tail and flash work (decode and attention owners), not light ops.

**KV-shared tail** (`plow_asset::kv_shared_tail`, `PLOW_PF_SHARED_TAIL`, on by default). The last
18 layers of E4B write no KV cache, so on the packed/token-batch route only the sampled rows
(prompt-final rows and riding decode rows) run them:
* the body stops at the tail boundary;
* RowGather moves the sampled rows' `act.x`, `act.hn` and `act.ple` to rows 0..n;
* the smallest decode rung with at least n rows then runs from its own tail boundary to its
  end: its GEMV/gemv_k8 projections, flash decode reading each row's slot through the packed
  slot map (`t6`), and its lm_head/argmax, which leave the ids and logits in rows 0..n for the
  terminal.

A counter image marks the skipped prefix as done. For 1-2 rows this costs 2.76 ms, including the
lm_head. The earlier path ran the smallest prefill bucket's tail segments and the separate
terminal, which took 5.0 + 0.9 ms. That path remains the fallback when no decode rung holds n
rows. A packet without the section, or a launch with more sampled rows than slots, runs the
whole program.

Served prefill-only (ISL 1000, OSL 1, temperature 0, `--multistep-adaptive`):

| | c1 req/s | c32 req/s | c32 TTFT p50 |
|---|---|---|---|
| all rows, all layers | 30.3 | 55.4 | 72.6 ms |
| KV-shared tail, prefill segments | 34.9 | 77.3 | 50.9 ms |
| KV-shared tail, decode step | 35.1 | 83.4 | 47.6 ms |
| vLLM | 26.5 | 75.7 | 384 ms |

Voice sessions (`session_bench.py`, 64 calls × 6 turns, 87% of later-turn prompt tokens cached):
later-turn TTFT p50 is 35.5 ms without the tail and 31.8 ms with it.

### Decode rungs vs roofline

Per-op instruction-cap sweeps (`step_bench --sweep`, ctx 1024) of the native program on every
rung. Roofline = bytes / 3.35 TB/s; cell = measured ms (% of roofline).

| B | step | skeleton | roofline | gate+up | down | o_proj | qkv | flash hd256 | flash hd512 | lm_head | PLE gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.81 | 1.22 | 2.79 (48%) | 1.52 (87%) | 0.81 (81%) | 0.25 (61%) | 0.15 (86%) | 0.24 (5%) | 0.10 (9%) | 0.43 (93%) | 0.07 |
| 8 | 6.36 | 1.36 | 2.95 (46%) | 1.58 (84%) | 0.93 (71%) | 0.19 (82%) | 0.14 (95%) | 0.57 (15%) | 0.22 (31%) | 0.48 (84%) | 0.12 |
| 16 | 7.45 | 1.42 | 3.12 (42%) | 1.66 (80%) | 0.93 (71%) | 0.27 (57%) | 0.20 (65%) | 1.04 (17%) | 0.40 (35%) | 0.49 (82%) | 0.17 |
| 32 | 9.08 | 1.48 | 3.46 (38%) | 2.06 (64%) | 1.27 (53%) | 0.37 (42%) | 0.27 (49%) | 0.96 (36%) | 0.61 (46%) | 0.60 (67%) | 0.35 |
| 48 | 10.95 | 1.53 | 3.81 (35%) | 3.11 (43%) | 1.77 (38%) | 0.54 (30%) | 0.38 (35%) | 1.06 (50%) | 0.69 (61%) | 0.51 (81%) | 0.29 |
| 64 | 11.76 | 1.55 | 4.16 (35%) | 3.24 (41%) | 1.95 (35%) | 0.62 (26%) | 0.48 (28%) | 1.08 (65%) | 0.74 (76%) | 0.50 (82%) | 0.34 |
| 96 | 15.51 | 1.67 | 4.85 (31%) | 3.19 (42%) | 1.60 (43%) | 0.66 (25%) | 0.70 (19%) | 2.34 (45%) | 1.41 (60%) | 0.72 (58%) | 0.95 |
| 128 | 16.39 | 1.77 | 5.54 (34%) | 3.33 (41%) | 1.45 (48%) | 0.57 (30%) | 0.71 (19%) | 2.55 (55%) | 1.46 (77%) | 0.70 (60%) | 1.19 |

Implementation per rung: gate+up/down/o/qkv run `gemv_k8` (1-tile arm B<=8, 2/4/8 tiles to 64
rows) and the `_gw` wgmma wide GEMV above 64; lm_head the row-block walk; flash the row-group
body. The skeleton (cap 0: every gate and dispatch, no bodies) is 1.2-1.8 ms.

The projections are the gap from 32 rows up. cuBLAS on the same shapes (42 layer copies so the
weights stream from HBM, CUDA graph, % of 3.35 TB/s), per layer in us:

| op (N x K) | M=1 | M=32 | M=64 | M=128 |
|---|---|---|---|---|
| gate+up (2 x 10240 x 2560) | 37.3 (84%) | 38.3 (83%) | 38.3 (84%) | 41.1 (80%) |
| down (2560 x 10240) | 21.8 (72%) | 23.1 (69%) | 23.9 (68%) | 26.7 (62%) |
| o_proj (2560 x 2048) | 6.3 (49%) | 6.7 (48%) | 6.8 (49%) | 7.2 (49%) |
| q+k+v (3072 x 2560) | 8.0 (58%) | 8.4 (57%) | 8.4 (58%) | 9.1 (57%) |
| lm_head (262144 x 2560) | 443 (91%) | 456 (89%) | 474 (87%) | 485 (87%) |

Native at B=64: gate+up 77 us, down 46, o 15, qkv 20 per layer. The standalone `_gw` wide GEMV
streams at cuBLAS speed without its stream-K fixup (down M=64 23.7 us, gate+up 44), and the
fixup costs 10-14 us per op. `PLOW_EMIT_DECODE_CUBLASLT_MIN_ROWS=48` therefore routes every
layer projection of the 48+ row rungs to cuBLASLt:

* The rungs are unfused (gate, up, q, k, v separate) and segmented, one captured graph per rung.
* The rungs below 48 keep the fused interpreter program, with bit-identical tokens.
* The routed rungs' interpreter windows launch on the `_gw` object.
* Multistep stays on.

| B | 48 | 64 | 96 | 128 |
|---|---|---|---|---|
| native ms | 11.01 | 11.86 | 15.57 | 16.42 |
| routed ms | 9.89 | 10.19 | 12.35 | 12.84 |

A routed step is ~14 launches per layer (8 matmuls, 6 interpreter windows): about 3 ms of the
B=128 step is launch gaps. Leaving k/v and the PLE projections in the interpreter cuts launches
but loses (the PLE input gate runs at 2% of roofline natively at B>=96).

## Gaps

* **Decode step at 30-41% of roofline.**
  * B=1: the fixed interpreter skeleton (1.2 ms) and down_proj at 50% bandwidth.
  * B>=64: sliding flash decode at ~3x its floor.
  * vLLM's TPOT is lower at c>=8 as a result.
* **Cold-prompt throughput at c>=32 (KERNEL).** Sampled rows now draw on the device, so mixed
  ticks no longer collapse (c64 943 -> 2701 tok/s). vLLM still serves 1.5-2.3x. The limits are
  the decode step (above); the prefill rate now matches vLLM on long prompts (see "Prefill per op"). No tick
  policy closes this; see "Where the tick time goes".
* **c8 TTFT (SCHED).** 90 ms vs vLLM 63: a new prompt waits for the running multistep quantum.
* **Logprobs.** Greedy logprobs rows cost ~1% at c32/c64 (batched device stats). Left:
  * run the stats inside a multi-step quantum (the kernel's `ids` argument reads each row's
    token from the device, so a quantum can run it after each step's sampler);
  * temperature > 0 logprobs rows still download the row.
* **Prompt logprobs** (`echo`, `prompt_logprobs`) are refused. The full raw logits row is not
  exposed over HTTP (the top 20 logits only).
* **Unused segment objects.** The segment script's pfseg/pfgemm objects fault on this packet
  (`CUDA_ERROR_LAUNCH_FAILED`). Only the flash objects are used.
