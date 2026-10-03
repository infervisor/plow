# Gemma-4 12B FP8 comparison

No production-qualified win over vLLM is established. Cross-stack quality is gated against an independent FP32 reference ([below](#fp32-reference-quality-gate)); the current best packet passes it. Internal gains below do not qualify a production recipe.

[All serving measurements](comparison.csv) are consolidated into one CSV. `campaign_source` identifies the original experiment. Existing metrics and qualifications are preserved. Fixed-16K experiments do not replace the best short-context configuration.
The required production profile has prefix caching enabled and a compiled 128-row maximum decode rung; the cache-off 64-row comparisons in this CSV do not qualify that profile.
The 16K/B128 BF16-KV packet compiled, but a one-H100 rung test failed at load: its 128 preallocated 2,048-row sliding rings exhaust VRAM. Live ring allocation loaded the packet but exhausted VRAM while admitting slot 98. The FP8-KV alternative is experimental and requires matched vLLM precision, correctness and cache validation.

At concurrency 128, the recorded serving results are:

| Input/output tokens | Configuration and traffic | Infervisor output tok/s | vLLM output tok/s | Ratio |
|---|---|---:|---:|---:|
| 128/128 | Current experimental 16K-capacity, 128-slot FP8-KV; sampled | 4,228 | 6,552 | 64.5% |
| 128/128 | Earlier 1K-capacity BF16-KV; sampled | 5,814 | 7,526 | 77.2% |
| 4096/128 | Earlier 16K-capacity, 64-slot BF16-KV; greedy | 665 | 893 | 74.4% |
| 15000/128 | Earlier 16K-capacity, 64-slot BF16-KV; greedy | 196 | 244 | 80.0% |

Each row has a matched vLLM arm and two repeats in [comparison.csv](comparison.csv). The earlier rows use different KV precision, decode-slot capacity, or context capacity from the current FP8-KV packet, and cross-stack output equivalence is unresolved. There is no medium- or long-context serving result for the current 16K/128-slot FP8-KV packet.

## Validated internal improvements

| Change | Measured improvement | Validation scope |
|---|---|---|
| FATLITE packed prefill | 8.1% lower instrumented prefill time; 5.7% higher long-serving throughput | Tested Plow logits preserved; still behind vLLM |
| Dedicated cached GLU+quant | 2.5–3.4% lower block time across six rungs | Four-arm block measurements; 13 full-model packed-prefill plus decode cases bit-exact against Plow control |
| Lightweight FP8 attention route | 8.6% lower decode time | Four-arm decode measurement; tested Plow logits preserved |
| B128 FP8-KV light attention on cached norm/quant | 4.80% lower decode-step latency at 128-token context; 4.82% at 4K | Four-arm, 40-step packet measurements; matching token digests and one-step full logits vs Plow control; 48 six-launch routes confirmed by Nsight. Attention remains about 13.7% of its conditional roofline; no vLLM quality or serving qualification. |
| B128 cuBLASLt output head | 13.38% lower decode-step latency at 128-token context; 5.40% at 4K | Four-arm, 40-step packet measurements; matching token digests. One-step 128/128 top-1 logits match Plow control (max KL 2.17×10⁻⁵); Nsight measures the Lt head at 0.699 ms plus 0.065 ms softcap/argmax. Matched-vLLM quality and serving are not qualified. |
| B128 head-Lt plus FP8 light attention | Further 4.96% lower decode-step latency at 128-token context; 3.21% at 4K vs the head-only packet | Four-arm, 40-step packet measurements with matching digests; one-step full logits byte-identical to head-only. Nsight confirms both routes; attention remains far below its estimated roofline. No matched-vLLM qualification. |
| B128 FP8 attention with 16-byte loads | Further 2.76% lower decode-step latency at 128-token context; 6.71% at 4K vs head-Lt plus FP8 light attention | Four-arm, 40-step packet measurements with matching token digests; one-step B128 full logits and pre-head tensors byte-identical. Nsight attributes 0.503 ms of a 0.465 ms shorter short-context graph to attention. Experimental: no matched-vLLM quality or serving qualification. |
| B128 direct FP8 value conversion with 16-byte K loads | Further 1.65% lower decode-step latency at 128-token context; 4.49% at 4K; 3.24% at 15K vs 16-byte-load packet | Four-arm, 40-step packet measurements at all three contexts with matching token digests; one-step full logits and pre-head tensors byte-identical. Nsight attributes 0.273 ms of a 0.274 ms shorter short-context graph to sliding attention. Exact opt-in cubin flags and hash are in the scratch build record. Experimental: matched-vLLM quality and serving remain open. |
| B128 segmented FP8 HD256 FlashDecode with a 1,024-block grid | 10.7% lower decode-step latency at 128-token context and 9.6% at 4K vs the preceding packet; 7.0% higher sampled short-serving output throughput vs its Plow control | Four-arm, 40-step A/B/A/B with matching token digests; one-step full logits and pre-head tensors byte-identical at 128 and 4K. Nsight confirms 40 dedicated launches, 95 registers, 11,328 B dynamic shared memory, zero spills. Serving at 128 concurrent requests used the same runtime and 16K/128-slot packet in both arms, two repeats, zero Plow prefix hits. The sampled serving repeat spread was under 0.5%; greedy repeats had 12–13% spread. Still below matched vLLM. Failed the retired bit-exact vLLM gate; passes the FP32-reference quality gate below. |

The resumed H100 single-kernel packed FP8 HD256 prefill trial selected `PLOW_NV_FA_PIPE=1` with `PLOW_NV_FP8_PACKED_VARLEN=1`. At 16 packed requests and 2,048 query rows, mean CUDA-event time across two repeats fell from 6.5731 to 0.2683 ms at KV position 256, 10.4859 to 0.3621 ms at 4K, and 10.7184 to 0.3724 ms at 15K versus the existing `PIPE=0`, serial-request wrapper. A ragged four-request case fell from 0.3445 to 0.0504 ms. All four precision/scheduling variants passed the independent FP32 gate and byte-exact packed-versus-direct BF16 output checks. These are wrapper timings, not packet or serving gains. The conditional compulsory-byte/compute floor is about 10% of measured time at the three 16-request shapes. The packet-matched dedicated candidate cubin (`9c66eeab…`) builds with 255 registers and 160/188 bytes of reported spill stores/loads.

A synthetic one-op packet launched through that dedicated interpreter cubin preserved byte-exact BF16 output against direct per-request calls. With matched dedicated objects differing only in flat scheduling, two-repeat mean cubin time fell from 1.2890 to 0.3666 ms at KV position 256, 1.8563 to 0.4809 ms at 4K, and 1.9011 to 0.4924 ms at 15K: 3.52×, 3.86× and 3.86× faster. The ragged four-request case improved 0.1388 to 0.0670 ms. The conditional floor is still only 7.3–7.4% of the three 16-request cubin times; hardware-counter roofline, block qualification and matched serving remain open. Exact compiler commands, hashes, per-repeat results and the floor model are in `packed-bench-build/production-flags/actual-cubin-ab.json` and `head-lt-build/grid-trial/fp8-fa-varlen0-matched/build-record.json` under the campaign scratch directory. No production recipe or matched-vLLM win follows from this isolated result.

The next layer-0 block A/B used one Lean-verified 16K/128-slot FP8-KV packet and two complete, packet-matched 18-cubin object sets differing only in the dedicated packed-attention cubin. The ragged two-request check passed with byte-identical activation, FP8 KV and scales across both arms and against each arm's serial reference. For 16 requests and 2,048 query rows, one exploratory block timing per arm was 3.3782→2.4099 ms at KV position 256, 3.9385→2.5003 ms at 4K, and 3.9963→2.5126 ms at 15K. Packed activation and FP8 KV/scale bytes were identical between the two arms at every shape. The block's serial-vs-packed FP8 KV and scales were also exact, but final BF16 activations differed by about 0.45–0.47% relative L2 with a 0.25 maximum, exceeding the existing absolute parity bound. At fixed 16-request concurrency, 1,152 query rows were byte-exact and 1,168 rows failed, exactly where the packet switches from its 1,152-row to 2,048-row prefill bucket; 8 requests × 256 rows also failed, while 16 × 64 passed. Intermediate dumps localize the first difference to the cuBLASLt FP8 `down_proj` output (`act.dg`): its quantized input and scales, both feed-forward projection outputs, the normalized input, and the attention output projection are byte-identical. `act.dg` differs by 0.413% relative L2 in the first request. This block gate remains **failed** pending a quality decision against an independent reference. The timing is a single, host-inclusive observation, not a repeated kernel roofline or serving result. Nsight Compute initially could not access hardware counters as the ordinary user (`ERR_NVGPUCTRPERM`), so the earlier conditional floor is not counter-verified. Raw logs and dumps are under `head-lt-build/grid-trial/block-fa-probe` in campaign scratch; no candidate is promoted.

Privileged Nsight Compute profiling of the actual flat dedicated cubin at the 16-request/2,048-row shapes measured 12.5% occupancy (one 256-thread block per SM), 255 registers/thread and 132,160 bytes dynamic shared memory. DRAM throughput was 6.14%, 5.88%, and 5.79% of peak at KV positions 256, 4K, and 15K; SM throughput was 27.18%, 30.21%, and 30.06%. L2 hit rates were 83.41%, 82.42%, and 82.25% on the repeated synthetic input. About 68% of scheduler cycles had no eligible warp; fixed-latency waits and short scoreboards dominated the stall sample. The serial-request cubin had the same 12.5% occupancy and similar per-SM active work, but its short-rung profiled duration was 1.48 ms versus 0.395 ms for flat scheduling, consistent with work imbalance across SMs. The candidate is **not at a measured roofline**; register/shared-memory occupancy and dependent instruction latency are the next kernel targets. An HD256-only object trial retained the same register, spill and shared-memory counts, so pruning the HD512 arm alone is insufficient. Profiler reports and the rejected compile trial are in `packed-bench-build/production-flags` and `head-lt-build/grid-trial/block-fa-probe/hd256-only-trial` in campaign scratch. These counter readings are from an isolated synthetic one-op packet, not full-model serving.

The latest matched short-serving comparison uses Gemma-4-12B-IT-FP8 on one H100 SXM, FP8 weights and per-token-head FP8 KV on both stacks, 128 input / 128 output tokens, concurrency 128, the same vLLM 0.28.0 client, and prefix caching enabled. The table reports the mean of two repeats; both arms completed 384 requests per repeat without failures. Plow's tested outputs match its control byte-for-byte, but only 37/64 exact-history decode top-1 tokens matched the vLLM oracle in the exact-match quality test. That test is retired as the promotion gate: its flips are near-ties and vLLM's own repeat floor is zero. See the FP32-reference gate below.

| Traffic | Stack | Output tok/s/GPU | TTFT P99 ms | TPOT P99 ms | Peak GPU memory GiB |
|---|---|---:|---:|---:|---:|
| Greedy | Infervisor, dedicated grid | 4,002 | 3,052 | 36.98 | 63.23 |
| Greedy | vLLM 0.28.0 | 6,116 | 1,558 | 17.51 | 72.85 |
| Sampled | Infervisor, dedicated grid | 4,228 | 2,222 | 31.67 | 63.41 |
| Sampled | vLLM 0.28.0 | 6,552 | 647 | 18.20 | 72.85 |

The greedy throughput repeat spreads were 13.4% for Infervisor and 24.5% for vLLM, so those means are diagnostic. Sampled spreads were 0.47% and 1.28%. Plow's measured prefix hit rate was zero; vLLM's token hit rate was 0.2–0.5%. The user-reported production 20% versus 80% cache-hit gap is not reproduced by this unique-prompt grid. The candidate remains experimental. The main FP8-KV cubin uses `PLOW_NV_LIGHT_FP8_ATTN=1`, `PLOW_FP8_LD16=1` and `PLOW_FP8_FAST=1` on the existing H100 Gemma FP8 flag set; the isolated sweep used `PX11_GRID_MULT=8`. Its exact cubin command, source hashes and binary hashes are in `head-lt-build/grid-trial/build-record.json` under the campaign scratch directory; per-repeat serving data are in the single `serving-summary.csv` there. The 15K tensor capture was stopped at the user's request, so long-context validation of this dedicated route remains open.

## FP32-reference quality gate

Bit-exact agreement with vLLM is replaced as the promotion gate by `campaign.py gate` kind `llm_fp32_ref` ([agent-tools.md §5](../../agent-tools.md#5-numerics-and-retrieval-gates)). Reference: the same `gemma-4-12b-it-fp8` checkpoint dequantized to FP32 weights, FP32 activations, TF32 off, exact attention (H100, torch 2.13, transformers 5.16). Prompt set: 46 cases and 1,172 teacher-forced positions. That is 20 natural-text cases (128 to 15,872 tokens), 8 agentic chat cases (54 to 15,255 tokens) and 18 chat needles (4K and 15,872 tokens). Both stacks are teacher-forced on the FP32 greedy continuation through `/v1/completions` at concurrency 16. Plow is the current best packet: grid-trial dedicated-grid 16K/128-slot FP8-KV, with the `serve-short-grid` serve env and prefix cache on. vLLM is 0.28.0 with the `serve-short-vllm-fp8pth` flags. Each stack was captured twice against the same server.

| stack | KL mean | KL p99 | top-1, FP32 margin > 1 nat (1,016 pos.) | greedy agreement | needle | KL mean, natural / chat / needle |
|---|---:|---:|---:|---:|---:|---|
| Plow r1 | 0.1038 | 2.279 | 0.9852 | 0.608 | 18/18 | 0.184 / 0.0086 / 6.2e-7 |
| Plow r2 | 0.0971 | 2.431 | 0.9852 | 0.642 | 18/18 | 0.172 / 0.0082 / 5.0e-7 |
| vLLM r1 | 0.1277 | 3.188 | 0.9823 | 0.574 | 18/18 | 0.226 / 0.0114 / 1.7e-6 |
| vLLM r2 | 0.1084 | 3.011 | 0.9882 | 0.595 | 18/18 | 0.190 / 0.0122 / 1.6e-6 |

**Verdict: PASS.** Plow passes against both vLLM repeats, and so does Plow r2. Plow is at least as close to FP32 as vLLM on every metric. Across all positions, top-1 differs from FP32 at 65–72 of 1,172 positions on both stacks. The decisive flips and the largest KL values come from repetition loops in long natural-text continuations, and they occur on both stacks. Chat and needle positions are close to exact on both (needle KL ≤ 4.4e-5).

Thresholds are kept relative to vLLM and checked against the measured repeat spread:
- KL: Plow ≤ vLLM × 1.25 + 0.002. vLLM's own repeats differ by 1.18× in mean and 1.06× in p99.
- Top-1: −0.01. vLLM repeats differ by 0.006.
- Greedy agreement: −0.05. Repeat spreads were 0.021 for vLLM and 0.034 for Plow.
- Needle: no drop allowed, and at least 0.9.

vLLM run against its own repeat passes in both directions, so the tolerances admit the reference stack's own noise. None of the provisional values needed to change.

During the run, plowrt was found to omit a stop token from completions logprobs (`finish_reason=stop`, `tokens=[]`), unlike vLLM. The capture now sends `ignore_eos`. Evidence is in `/opt/dlami/nvme/lava-tts/fp32gate/`:
- `ref.json`: sha256 `b2988c51…`.
- `prompts.json`: `10302403…`.
- Captures in `best/gate/llm_fp32_ref/`: plow `3d79bd08…`, vllm `74a7a572…`.
- Gate recipe: `best/recipe.toml`.

## Selection and reproduction

Optimize native Plow kernels against the roofline for each rung, then select only the fastest correct implementation. Where cuBLASLt is faster, use the corresponding segmented route. A standalone kernel improvement must survive block and serving validation before promotion.

Maintain one canonical production TOML under `recipes/infervisor`, updated in place with validated per-rung compilation flags, implementation choices and runtime settings. Earlier Gemma 12B trials reside in `scripts/campaign/recipes`; none is production-qualified. A 2.15 GB H100 streaming probe measured up to 3.21 TB/s; the latest B128 short-context attention group still takes 6.70 ms against an approximately 1.07 ms compulsory-byte floor at that bandwidth. This is a proxy ceiling rather than a measured per-kernel hardware-counter roofline.

Detailed manifests, JSON, HTML, logs and kernel audits are archived outside the repository at:

`/opt/dlami/nvme/tmp/gemma12b-main-20260930/repo-evidence-archive-327413c3`

The archive manifest records SHA256s for every moved file. Live artifacts stay in the campaign scratch directory. Append future serving measurements to this CSV and record qualified wins here.

## 128 slots at 16K context (live rings, byte admission)

Not a win. The 16K/B128 BF16-KV packet that ran out of VRAM at load now loads and serves: each 640 MiB sliding ring is committed only while a request owns its slot (`PLOW_VMM_LIVE_RINGS=1` under `PLOW_VMM_PREFIX=1`; idle decode rows write one shared scratch unit), and the mux admits requests by bytes under `PLOW_KV_MEM_UTIL=0.9`. That admits 71-78 requests at 4K and 64 at 15K, against 64 slots on the current packet. Serving is prefill-bound at both lengths, so the extra live requests do not raise output tok/s.

Setup: one H100, greedy, OSL 128, 3×c prompts per cell, same vLLM 0.28 client, Plow runtime `6f3fe2d5` (the margin row uses `da8b526d`). 64-slot = `glu-quant-wpr/full-model` packet with prefix cache off. 128-slot = `production-16k128-lt` packet with prefix cache on and `PLOW_PF_ATTN_GEMM=1`. Values are the mean of two repeats unless marked r1.

| ISL | c | packet | out tok/s | TTFT p50 / p99 s | TPOT p50 / p99 ms | peak GiB | max live |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4096 | 64 | 64-slot | 617 | 1.2 / 9.9 | 92 / 99 | 63.3 | 64 |
| 4096 | 64 | 128-slot | 555 | 1.4 / 10.9 | 98 / 111 | 71.8 | 64 |
| 4096 | 128 | 64-slot | 616 | 14.3 / 22.4 | 96 / 102 | 63.3 | 64 |
| 4096 | 128 | 128-slot | 557 | 13.1 / 25.0 | 129 / 136 | 78.2 | 78 |
| 4096 | 128 | 128-slot, margin charged | 547 | 14.7 / 25.0 | 118 / 126 | 73.0 | 71 |
| 15000 | 64 | 64-slot | 189 | 22.4 / 40.3 | 158 / 164 | 64.3 | 64 |
| 15000 | 64 | 128-slot (r1) | 186 | 20.4 / 42.1 | 176 / 179 | 73.7 | 64 |
| 15000 | 128 | 64-slot | 195 | 61.4 / 80.0 | 172 / 174 | 65.5 | 64 |
| 15000 | 128 | 128-slot (r1) | 189 | 63.0 / 82.6 | 176 / 179 | 73.7 | 64 |
| agentic16k | 64 | 128-slot | 301 | 5.9 / 21.7 | 178 / 186 | | |
| agentic16k | 128 | 128-slot | 290 | 27.8 / 67.0 | 180 / 192 | | |
| agentic16k | 128 | 64-slot, prefix cache on | 307 | 28.8 / 64.6 | 171 / 179 | | |

The agentic rows use `llm_grid.sh --agentic`, one repeat, with 0-5% prefix hits. A retained 16K session costs about 564 MiB on BF16 KV: a 320 MiB sliding snapshot plus 244 MiB of full-attention blocks.

Memory per request: 640 MiB of sliding ring, plus 16 KiB per full-attention row in 2048-row blocks, plus the admission margin (one prefill bucket, 96 MiB). A 4K request costs about 800 MiB and a 16K request about 960 MiB. The KV budget after load is 56 GiB. The row without the margin charge peaked at 78-80 GiB because admission maps one widest prefill bucket past each request; charging that margin brings the peak to 73 GiB. Prefix-cache snapshots still sit outside the budget.

FP8 KV (`sm90a-h100-tp1-fp8kv-16k-c128-rq1k-live`) would fit 128 × 16K: 320 MiB ring + 8 KiB per row, or about 512 MiB per 16K slot. It cannot serve at 4K yet. Its packed prefill attention costs about 55 ms per riding decode row: a 2112-row pack with 62 riders takes 3.8 s, against 0.17 s for 4096 rows with 63 riders on BF16 KV. `PLOW_TOKEN_BATCH=0` is refused for FP8-KV segmented prefill.

Gates on the 128-slot packets: cached and cold prompts of 0.9K, 3.4K and 8.5K tokens give the same 48 greedy tokens, including with 32 concurrent sharers. Max |Δlogprob| is 0.33 on BF16 KV and 0.12 on FP8 KV; the logprobs are not byte-identical. A raw-prompt 16K needle (3 items × 3 depths × 10) scores 47/90 on BF16 and 41/90 on FP8. Its failures are deterministic by item and depth, so the raw-prompt probe needs the 64-slot baseline before it can count as a gate.

Raw cells and logs: `/opt/dlami/nvme/lava-tts/cap128/serve`.
