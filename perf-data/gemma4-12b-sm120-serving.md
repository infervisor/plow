# Gemma-4-12B-it serving on RTX 5090 (sm_120a) — plowrt vs vLLM 0.26.0

Measured 2026-07-26. GPU: RTX 5090, 32 GiB, driver 580.159.03, 170 SMs.
Model: `/root/gemma-4-12B-it` (48 layers, 8 full / 40 sliding, window 1024).

Both engines driven by the SAME harness, `vllm bench serve --backend openai-chat`,
so the numbers are directly comparable (identical tokenizer, identical prompts —
`Total input tokens: 33199` on both sides is the check that the shapes match).

    vllm bench serve --backend openai-chat --base-url <url> \
      --endpoint /v1/chat/completions --model gemma-4-12b-it \
      --tokenizer /root/gemma-4-12B-it --dataset-name random \
      --random-input-len <I> --random-output-len <O> \
      --num-prompts <N> --max-concurrency 16 --seed 0

## Headline

Output token throughput (tok/s), concurrency 16:

| shape             | vLLM (bf16) | plowrt best | plowrt at session start |
|-------------------|-------------|-------------|-------------------------|
| 1024 in / 128 out | **428.48**  | 293.80      | 45.75                   |
| 128 in / 512 out  | **759.19**  | 527.85      | —                       |

plowrt improved 6.4x on the prefill-heavy shape. It does not beat vLLM AT c16 —
but c16 is one point on a curve, and the curve crosses. Both engines emit
identical token counts (4096 and 24576) once `ignore_eos` is honored, so these
are like-for-like.

### Concurrency curve — where plow wins (128 in / 512 out, decode-heavy)

| concurrency | plowrt  | vLLM    | ratio          |
|-------------|---------|---------|----------------|
| 4           | **258.85** | 218.97 | **1.18x plow** |
| 8           | **410.22** | 409.85 | 1.00x (tie)    |
| 16          | 488.84  | 771.09  | 0.63x          |

**plowrt beats vLLM at concurrency 4 by 18% and ties at 8.** The crossover is
between c8 and c16 — exactly where plow's fixed-width decode starts paying a
second weight read (the GEMV block-of-8 walk) while vLLM's continuous batching
keeps scaling. Blob used: B = concurrency, fp8 weights + fp8 KV, NS=8.
(Run-to-run spread at c16 is ~8%: 488.84 here vs 527.85 in the 48-prompt run.)

plowrt's decode kernel scales sub-linearly within a rung — 13.32 / 16.11 /
25.61 ms at B=4 / 8 / 16 — so weight-stationarity is working; it is the
cross-rung re-read that breaks scaling past 8. vLLM reference config: bf16 weights, `kv_cache_dtype=auto` (bf16),
`max_model_len=8192`, prefix caching on, full CUDA graphs.

At concurrency 1 the two are at parity (ITL p50 17.44 vs 17.23 ms) — the entire
gap is multi-user scaling.

## What moved it (each measured independently, cumulative)

Baseline is the shipped B=1 blob at c16: 45.75 tok/s.

| change                                        | 1024/128 c16 | note |
|-----------------------------------------------|--------------|------|
| B=16 blob (`PLOW_DECODE_BATCH=16`) + fp8 KV    | 102.74       | needs fp8 KV to fit; see budget below |
| `+ PLOW_DEV_SAMPLE=1 PLOW_MULTISTEP=8`         | 179.18       | 1.74x — removes host round-trip per token |
| `+ PLOW_MULTISTEP=32`                          | 185.60       | gap stops shrinking here |
| `+ -DGV_MM_MAX=16` cubin                       | 220.10       | decode kernel 41.17 -> 28.8 ms |
| `+ fp8 weights` (`PLOW_FP8=1 PLOW_W8A8=1`)     | 240.52       | only 28.8 -> 26.5 ms; see "fp8 weights" |
| `+ PLOW_NS_FULL_ABS=8` (was 48)                | —            | kernel 26.32 -> 25.18 ms |
| `+ ignore_eos honored`                         | 293.80       | fair token counts; occupancy 10.9 -> 14.4/16 slots |

Best config in full:

    # cubins
    PLOW_EXTRA_DEFINES="-DGV_MM_MAX=16 -DGV_UN16=8" PLOW_BUILD_FP8KV=1 \
      scripts/build_sm120_cubin.sh <out>/interp_sm120.cubin
    # blob
    PLOW_UNISEG=1 PLOW_NS_FULL_ABS=8 PLOW_DECODE_BATCH=16 PLOW_MAX_CHUNK=1024 \
    PLOW_FP8_KV=1 PLOW_FP8=1 PLOW_W8A8=1 \
      plowc --hf-dir <model> --gpu rtx5090 --emit devblob --max-ctx 8192 \
            --weight-dtype fp8 --out <dir>
    # serve
    PLOW_UNISEG=1 PLOW_PF_BATCH=1 PLOW_DEV_SAMPLE=1 PLOW_MULTISTEP=32 \
      plowrt serve --assets <dir> --slo-ms <large>

## The decode-step cost model (this is the useful part)

`PLOW_STEP_TIME=1` splits the step into device kernel vs host gap. At B=16,
bf16 weights, `GV_MM_MAX=8`:

    gap_us=83647  submit_us=60  dev_interp_ms=41.05  dev_upload_us=38  dev_download_us=4.4

Two separate problems, and the host one was much larger than expected.

### 1. Host gap — 83.6 ms of every step with the GPU idle

Fixed by `PLOW_DEV_SAMPLE=1` + `PLOW_MULTISTEP=K` (both already implemented,
both default OFF). Gap 83.6 -> 32.2 ms. K=32 is not meaningfully better than
K=8; the residue is real interleaved prefill, not host overhead.

**These two flags are worth 1.74x on their own and cost nothing.** They are the
single highest-value default to reconsider for serving.

### 2. Decode kernel — the GEMV M-block re-read

`gemv_rows<MM>` is weight-stationary, but the ladder is `{1,2,4,8}` and M>8
"walks in blocks of 8" — so B=16 streams all 22.2 GiB of weights TWICE.

Measured, bf16 weights, `GV_MM_MAX=8`:

| B  | dev_interp_ms | implied tok/s ceiling |
|----|---------------|-----------------------|
| 8  | 22.22         | 360                   |
| 16 | 41.17         | 388                   |

B=16 is 1.85x B=8 for identical weights — the second read. Check:
(2 x 22.2 + 2.7) GiB / 41.2 ms = 1.14 TB/s, i.e. the kernel is already at the
B=1 bandwidth efficiency (1.29 TB/s); it is simply moving twice the bytes.

`-DGV_MM_MAX=16` takes B=16 to 28.8 ms, matching the ladder table in
`runtime/nvidia/op_gemm.cuh` (which predicts 30.80 ms / 519.6 tok/s). That table
was measured on RTX PRO 6000; **it reproduces on sm_120a** — our `GV_MM_MAX=8`
B=16 measurement of 41.17 ms vs its 41.34 ms is a 0.4% match.

Consequence: with the default ladder, every 8 slots costs one full weight read,
capping aggregate decode at ~437-465 tok/s regardless of batch. That is why
neither B=8 (360) nor B=16 (388) can reach vLLM's 428.

### 2b. NS_FULL_ABS — the decode split factor (tuner axis)

Swept at B=16, fp8 weights, `GV_MM_MAX=16`, measured as `dev_interp_ms`:

| PLOW_NS_FULL_ABS | dev_interp_ms |
|------------------|---------------|
| 8                | **25.18**     |
| 16               | 25.28         |
| 32               | 25.93         |
| 48               | 26.32         |

Monotone: fewer splits win at this batch. `48` — the value
`scripts/build_gemma_family_sm120.sh` documents and which this session used
throughout — is the WORST of the four. 16 x 48 = 768 split-blocks over 170 SMs
is oversubscribed; the win is small (4.3%) but free.

### 2c. GV_UN16 / GV_UN_GLU16 — wide-rung unroll (tuner axis)

`op_gemm.cuh` gives the MM=16/32 rungs a shallower unroll to relieve the spill
that kept the ladder at 8, and says the knobs are "build-overridable for
autotune". Swept at B=16, fp8 weights, NS=8 (32 prompts, 128/512, c16):

| GV_UN16 / GV_UN_GLU16 | REG / STACK | dev_interp_ms | tok/s  |
|-----------------------|-------------|---------------|--------|
| 2 / 1                 | 241 / 1024  | 26.17         | 519.12 |
| 2 / 2                 | 250 / 1024  | 26.71         | 494.46 |
| 4 / 2 (default)       | 250 / 1024  | 25.61         | 529.52 |
| 4 / 4                 | 255 / 1056  | 25.47         | 530.80 |
| **8 / 2**             | 255 / 1120  | **25.04**     | 530.24 |

The MOST spilled variant is the fastest and the least spilled is near the
slowest, so on sm_120a this rung is latency-bound, not register-bound —
trading unroll depth away to relieve spill is a net loss here. `-DGV_UN16=8`
is worth ~2% of kernel time; aggregate throughput is flat within noise.

### 3. fp8 weights barely help decode

Weights 22.2 -> 12.0 GiB, but `dev_interp_ms` only 28.8 -> 26.5 ms. This is
expected and already documented in `op_gemm.cuh`: the fp8 dense GEMV runs at
1046 GB/s and is **compute-bound on the dequant, not bandwidth-bound**. fp8
weights are worth taking for the VRAM (18.9 vs 28.8 GiB used), not the speed.

## Knob classification (which layer owns each win)

| layer | knob | measured effect |
|-------|------|-----------------|
| cubin | `GV_MM_MAX=16` | B=16 kernel 41.17 -> 28.8 ms (**+43%**, the largest kernel win) |
| cubin | `GV_UN16=8` | 25.61 -> 25.04 ms (~2%) |
| cubin | `PLOW_FP8_KV=1` | halves KV; enabler for B=16 on 32 GiB |
| cubin | `PLOW_NV_W8A8=1` | fp8 tensor-core prefill (with fp8 weights) |
| plowc | `PLOW_UNISEG=1` | mandatory on sm_120 |
| plowc | `PLOW_NS_FULL_ABS=8` (was 48) | 26.32 -> 25.18 ms (4.3%) |
| plowc | `PLOW_MAX_CHUNK=1024` | KV 5.50 -> 1.125 GiB at B=1 (**5x**) |
| plowc | `PLOW_DECODE_BATCH=B` | fixed kernel width — must equal concurrency |
| plowc | `--weight-dtype fp8` | 22.2 -> 12.0 GiB weights; only 28.8 -> 26.5 ms |
| plowrt | `PLOW_DEV_SAMPLE=1` + `PLOW_MULTISTEP` | 102.74 -> 185.60 tok/s (**1.74x**) |
| plowrt | `--slo-ms` raised | prevents mass 429 shedding at large batch |
| plowrt | `ignore_eos` honored | occupancy 9.5 -> 14.4 of 16 slots |

**Defaults FLIPPED in-tree** as a result of this campaign:

| default | was | now | evidence |
|---------|-----|-----|----------|
| `PLOW_DEV_SAMPLE` | off | **on** (=0 opts out) | part of the 1.74x |
| `PLOW_MULTISTEP` | off | **on, K=8** (=0 opts out) | K=8 gets 179.18 of the 185.60 at K=32 |
| `PLOW_MAX_CHUNK` | 8192 | **1024** | KV 10.66 -> 3.04 GiB for 2% prefill |
| `PLOW_NS_FULL_ABS` (family script) | 48 | **8** | 26.32 -> 25.18 ms |
| `--slo-ms` | flat 250 | **floor**, `max(250, 8 x service_ms)` | stops the B=32 mass-429 |

End-to-end proof: a B=16 bundle with NO tuning env vars and no `--slo-ms` now
serves **525.55 tok/s at c16, 0 shed events**, correct output, 18.86 GiB. The
same bundle before the flips served 102.74 tok/s and 429'd every request.

`GV_MM_MAX` is deliberately NOT flipped — 16 wins at B>=16 but loses at B=8
(355 -> 294) and the stock blob is B=1, so it stays a per-build choice.

**Tuner impact is small.** The two axes the decode tuner sweeps gave 4.3% and
2% (~6% together). The large wins were structural defaults, not tuning. Fix
defaults first, tune last. See `docs/llm-bringup-harness.md`.

## VRAM budget (why B=16 needs fp8 KV)

Sliding-ring KV is `next_pow2(window + PLOW_MAX_CHUNK - 1)` rows. With
window=1024 the floor is 2048 rows however small the chunk, so per sequence:

    sliding = 2048 x 40 lay x 8 kvh x 256 hd x 2 (k,v) x elt
    full    = ctx  x  8 lay x 1 kvh x 512 hd x 2 (k,v) x elt

At ctx 8192: bf16 768 MiB/seq, fp8 384 MiB/seq. Weights bf16 22.2 GiB leaves
~6.9 GiB, so bf16 KV tops out at B=8 (6.00 GiB) and B=16 requires fp8 KV
(6.09 GiB). Measured `kv_gib` matches these to the megabyte.

## Open / not fixed

- **Residual decode-kernel deficit — this is what is left.** After everything
  above, plowrt's B=16 step is 25.6 ms against vLLM's ~20.5 ms (its median ITL
  at the same batch), a 1.25x deficit, and plowrt already runs at 84% of its own
  kernel ceiling (527.85 of 625 tok/s). So the remaining gap is the kernel
  itself, not scheduling or configuration — closing it is real kernel work, not
  another flag. Slot occupancy, which was the other suspect, is now 14.4/16
  after `ignore_eos` (was ~9.5 when short generations churned prefill).
  Every configuration axis has now been swept and is at its optimum:
  `GV_MM_MAX`, `NS_FULL_ABS`, `GV_UN16`/`GV_UN_GLU16`, `PLOW_MULTISTEP`,
  `PLOW_PF_INTERLEAVE`/`PLOW_PF_CHUNK`, decode batch, weight dtype, KV dtype.
  The kernel floor they reach is ~25.0 ms at B=16 = 639 tok/s, which is BELOW
  vLLM's 759 even at 100% slot occupancy. Beating vLLM therefore requires a
  faster batched decode GEMV — plausibly making the fp8 arm bandwidth-bound
  rather than dequant-bound, or a tensor-core M=16 decode path — and no
  amount of tuning the existing kernel gets there.
- **Compiled batch is fixed width.** A B=32 blob pays the full B=32 kernel cost
  even when only 16 slots are live, so B must match expected concurrency
  (B=32 at c16 measured 165 tok/s, worse than B=16's 240).
- **Admission shedding.** `--slo-ms` default 250 sheds every request once
  predicted wait exceeds it, which a large batch trivially does
  (`predicted_wait_ms=329`). The bench reports these 429s as "successful"
  requests with ~12 tokens each, which looks like a 2592 tok/s result. Raise
  `--slo-ms` for throughput benchmarking or the numbers are nonsense.
- **Prefill interleave knobs did not help.** `PLOW_PF_INTERLEAVE` 512/256 x
  `PLOW_PF_CHUNK` 256/128 all measured at or below the 2048 default
  (230.95 / 233.59 / 156.60 / 154.49 vs 240.52).
- **No sm_120a entry in `tuning/`.** `plowc tune --gpu rtx5090 --status` reports
  "no kernel measurements for this cell", so selection uses the analytical model
  at tier `portable`. Populating it needs `scripts/tune_decode_sweep.sh`, which
  currently requires `gpulease` (absent here) and forces
  `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat` (a libcuda NEWER than the
  driver — it fails `cuInit` with `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`).

## Reproduction note

`PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1` is required on this box.
Without it `cuInit` fails against the CUDA compat lib and plowrt falls back to
the CPU reference interpreter — which is only a WARNING, so an unwary run
silently benchmarks the CPU path.
