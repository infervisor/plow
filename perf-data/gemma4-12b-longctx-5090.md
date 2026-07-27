# Gemma-4-12B long context on RTX 5090 (sm_120a) — plowrt vs vLLM 0.26.0

Measured 2026-07-26. GPU: RTX 5090, 32 GiB (31.36 usable), 170 SMs, driver 580.159.03.
Model `/root/gemma-4-12B-it`: 48 layers = 40 sliding (window 1024, kvh 8, hd 256)
+ 8 full (kvh 1, hd 512), vocab 262144, `attention_k_eq_v`.

Both engines driven by the SAME client (`vllm bench serve --backend openai-chat`,
`--ignore-eos`, same tokenizer), so shapes are identical — `Total input tokens`
matches on both sides.

    plow : fp8 weights (W8A8) + bf16 KV, PLOW_NS_FULL_ABS=32, chunk 8192, UNISEG
    vLLM : --quantization fp8 --kv-cache-dtype fp8 --max-model-len 131200
           --gpu-memory-utilization 0.90 --max-num-batched-tokens 8192

## 1. Capacity: vLLM in bf16 cannot serve long context on this GPU at all

    ValueError: To serve at least one request with the model's max seq len (131200),
    7.32 GiB KV cache is needed, which is larger than the available KV cache memory
    (4.9 GiB). ... the estimated maximum model length is 15280.

**vLLM bf16 caps at 15,280 tokens.** plow serves 132,096 tokens × 4 sequences in
22.9 GiB total (weights 12.0 + KV 13.1 + activations 0.5) — bf16 KV throughout.

Why: plow RINGS the sliding layers at `next_pow2(window + chunk - 1)` rows
(`devgen::kv_ring`) instead of allocating `ctx` rows. Per sequence at 132k:

| layer class | plow bf16 KV / seq          | note                          |
|-------------|-----------------------------|-------------------------------|
| 40 sliding  | 0.625 GiB (ring 2048 rows)  | INDEPENDENT of context length |
| 8 full      | 2.02 GiB (linear, 132k rows)| the only part that grows      |
| total       | **2.65 GiB**                | vLLM needs 7.32 GiB — 2.76×   |

vLLM only reaches 131k by dropping to fp8 weights AND fp8 KV (3.99× concurrency).
That is the config compared below; plow is at bf16 KV and still wins on footprint.

## 2. Measured, concurrency 1, output 128

True TTFT for plow is `127 x (TPOT - ITL)` — `vllm bench` mis-attributes plow's
prefill because plow emits an SSE role-delta chunk before the first content token,
so the client stops its TTFT clock early. The reconstruction is validated against
direct wall-clock measurement (`pf_time.py`), agreeing to <1%.

| input tokens | plow TTFT (s) | vLLM TTFT (s) | plow ITL (ms) | vLLM ITL (ms) |
|--------------|---------------|---------------|---------------|---------------|
| 4096         | **0.50**      | 0.87          | 13.46         | 11.70         |
| 16384        | 2.14          | 1.11          | 14.29         | 12.56         |
| 32768        | 4.89          | 1.91          | 15.42         | 13.63         |
| 65536        | 12.31         | 5.04          | **14.81**     | 14.47         |
| 126976       | 33.09         | 14.20         | **16.21**     | 16.07         |

**Decode is a tie** — 0.9% behind at 127k, 2.4% at 64k, after the NS fix below.
**Prefill is 2.2-2.6× behind** beyond 16k. That is the whole remaining gap.

## 2b. Matched-workload throughput at 127k — plow LOSES 1.74x

Identical offered load: 8 concurrent 126,976-token requests, 1024 output tokens
each. Both engines report the same token counts (1,015,914 in / 8,192 out), so
this is like-for-like. Each engine runs at its own best concurrency: plow B=8
(mixed fp8 KV, 1.68 GiB/seq), vLLM ~4 resident + scheduler preemption.

| engine | aggregate out tok/s | wall (s) | median ITL (ms) |
|--------|---------------------|----------|-----------------|
| vLLM   | **42.63**           | 192      | 18.85           |
| plow   | 24.53               | ~334     | 53.22           |

**The entire deficit is prefill.** plow spends ~250 s of the 334 s prefilling
8 x 127k serially (2.2x slower per token AND not overlapped with decode);
vLLM spends ~114 s. Neither ITL figure is a steady-state decode number — both
are inflated by decode steps interleaved with other requests' prefill, plow's
much more so because its prefill phase is longer.

Holding more sequences resident does NOT rescue this: plow's 2x slot count buys
nothing while prefill is the serial bottleneck. Back-of-envelope, plow only
crosses ahead once output length exceeds ~6000 tokens per request, where the
decode phase finally dominates the prefill phase. That was not measured.

## 3. What moved it (each measured independently)

### 3a. The fp8-KV prefill object costs 4x at long context — not "20-30%"

`build_sm120_cubin.sh` must build the fp8-KV prefill object with
`-DPLOW_NV_FA_PIPE=0` (cp.async cannot dequant e4m3 inline). PIPE=0 also disables
`FA_PX4_ELIGIBLE(HD) = PLOW_NV_FA_PIPE && PLOW_NV_FA_PX4 && HD == 512` — i.e. the
px4 fast path for exactly the hd512 FULL layers that dominate long-context prefill.

    65549-token prefill:  fp8-KV pf object 49.60 s  ->  plain pf object 12.37 s   (4.0x)

`perf-data/rtx19-e3-fp8kv.md` claims fp8-KV prefill is "~20-30% slower". That was
measured at short context, where px4 barely matters. At 64k it is **4x**.

Consequence: **fp8 KV is the wrong choice for long context**, on speed alone,
before the correctness bug below.

### 3b. PLOW_NS_FULL_ABS was tuned at ctx 1024 and is wrong at long ctx

The full-layer flash-decode split factor. The serving campaign measured 8 best at
ctx 1024, B=16. At long context with B=4 it is badly wrong:

| NS_FULL_ABS | ITL @ 65536 | ITL @ 126976 |
|-------------|-------------|--------------|
| 8 (shipped) | 17.69 ms    | 21.86 ms     |
| **32**      | **14.81**   | **16.21**    |
| 64          | 15.37       | 17.29        |
| 128         | 16.58       | —            |

**32 is the long-context winner: -16% at 64k, -26% at 127k.** This single knob is
what closes the decode gap with vLLM. It is an emit-time (plowc) knob — no cubin
rebuild — so it can be selected per deployment context length.

### 3c. Prefill chunk: bigger is better once px4 is on

65549-token prefill, B=2: chunk 2048 -> 12.39 s, 4096 -> 11.57 s, 8192 -> 11.31 s.
Only 9%, and it costs KV: the sliding ring is `next_pow2(1024 + chunk - 1)`, so
chunk 8192 doubles sliding KV per sequence to 5.0 GiB. Worth it at B<=2, not at B=8.

## 4. CORRECTNESS: fp8 KV corrupted every multi-launch prefill — FOUND AND FIXED

**Root cause: a host-side patch-site miss, not a kernel bug.** `exec/gpu.rs`
collects the per-chunk patch sites by opcode:

```rust
if inst.op == DevOp::HeadNormRope as u16 && inst.fj[1] != 0 { rope.push(ix) }
else if inst.op == DevOp::FlashPrefill as u16 { flash.push(ix) }
```

An fp8-KV packet emits `HeadNormRopeFp8` / `FlashPrefillFp8` (dev.rs 37/39),
which carry the SAME operands at the SAME indices — rope `i[3]`=out_row0,
`fj[1]`=out_stride; flash `i[1]`=seq_kv, `i[4]`=q_pos0 (see the
`HEADNORM_ROPE_FP8` / `FLASH_PREFILL_FP8` arms of interp_sm120.cu). Neither
matched, so **no fp8 site was ever patched**: every prefill chunk after the
first wrote its KV at `out_row0 = 0` (clobbering chunk 1) and read back with
`q_pos0 = 0`, `seq_kv = chunk`.

`crates/packet/src/devbuild.rs:434` already pairs the two FlashPrefill opcodes;
only the runtime's patch loop was missed. Fix: match the Fp8 twins too, and mark
the bucket `fp8_kv` so PX-1 batched prefill stays off for it (the fp8 arm spends
t6/t7 on the k/v dequant scales, not on the request table).

### How it was localized

The boundary looked like the 1024 sliding window, which sent me at the layer
class. It is not: `pick_prefill_bucket` covers a 1049-token prompt as
`[1024, 128]` — **two launches**. The decisive experiments:

| test                                              | result | conclusion             |
|---------------------------------------------------|--------|------------------------|
| `PLOW_PF_COVER=1` (single covering launch) @ 1049 | PASS   | trigger is multi-launch|
| default multi-launch @ 1049                       | FAIL   | ″                      |
| chunk 8192, COVER, 8111 tok (one launch)          | PASS   | not nsplit             |
| chunk 8192, COVER, 9441 tok (8192+2048, both ns=1)| FAIL   | trigger is `q_pos0 > 0`|

Both fp8 prefill arms (px4 and the PIPE=0 NO-GO arm) failed identically, which
is what pointed at shared host state rather than kernel code. bf16 was immune
because it runs `d_flash_prefill_mux`, which derives position from the request
table instead of `i[1]`/`i[4]`.

The earlier "sliding layers are corrupt, full layers merely lossy" reading was
wrong. Nothing is layer-class-specific — the damage just scales with how many
layers use fp8 KV. `PLOW_FP8_KV_FULL=1` looked "lossy" only because 40 of 48
layers were still bf16 and correct.

### After the fix

Needle-in-haystack, greedy, mixed fp8 KV, chunk 8192: PASS at 4691 / 8111 /
9441 / 32941 / **68941** tokens (9441 was the first failure before).

### Before the fix (kept: this is what the symptom looked like)

| config                             | 995 tok | 1049 tok | 4691 tok | 69k tok |
|------------------------------------|---------|----------|----------|---------|
| bf16 KV                            | PASS    | PASS     | PASS     | PASS    |
| fp8 KV, all layers (`PLOW_FP8_KV`) | PASS    | **FAIL** | FAIL     | FAIL    |
| fp8 KV, full layers only (`_FULL`) | PASS    | degraded | degraded | —       |
| fp8 WEIGHTS (`PLOW_FP8`+`W8A8`)    | PASS    | PASS     | PASS     | PASS    |

The boundary is **exactly the 1024 sliding window** (995 PASS / 1049 FAIL).
Failure output is non-linguistic: `'7777777>();>();>();>()...'`.

Ruled out:
- **Not chunk-crossing.** Reproduces with `PLOW_MAX_CHUNK=8192`, where a 1049-token
  prompt is a single chunk.
- **Not the fp8-mma arm.** `-DPLOW_NV_FA_FP8MMA=0` (the dequant-to-bf16 NO-GO arm)
  produces byte-identical garbage.
- **Not the shared window masking.** bf16 KV runs the same `eff_lo`/mask code and is
  correct to 69k.
- **Not the scale tensors.** `kv.{l}.k_scale` is sized to the ring and indexed
  identically (`rowidx`) by writer (`d_headnorm_rope_fp8`) and reader.

Localized to: **fp8 KV on SLIDING (hd256, windowed) layers**. `PLOW_FP8_KV_FULL=1`
keeps sliding layers bf16 and yields coherent-but-lossy output
(`"The secret password is PELIC-77"` vs the true `PELICAN-7734`), so full-layer
fp8 KV is *lossy* while sliding-layer fp8 KV is *corrupt*.

Why it was never caught: `perf-data/rtx19-e3-fp8kv.md` validated fp8 KV against
bf16 on **21 tokens**. The windowed regime was never exercised.

Note this implicates the earlier serving campaign: `gemma4-12b-sm120-serving.md`
headline numbers ran fp8 KV at 1024-token inputs, which after the chat template
land just inside the broken regime. Those are valid timings of an incorrect
computation.

## 5. A second trap: PLOW_W8A8=1 needs -DPLOW_NV_W8A8=1

Emitting with `PLOW_W8A8=1` produces `QUANT_FP8` / w8a8 GEMM opcodes. The cubin
defaults to `PLOW_NV_W8A8 0`, whose `default: __trap()` surfaces as
`CUDA_ERROR_LAUNCH_FAILED` from `cuStreamSynchronize` at the first prefill —
with no hint that a define is missing. `build_sm120_cubin.sh` does not pass it;
it must come through `PLOW_EXTRA_DEFINES`. Worth a load-time gate.

## 6. Where the remaining gap is, quantitatively

Fitting `T = a·N + b·N²` to the prefill curves (both fits predict the 127k point
to <1%, so the split is trustworthy):

| term                    | plow        | vLLM        | plow deficit |
|-------------------------|-------------|-------------|--------------|
| linear a (GEMM)         | 8,693 tok/s | ~25,200 tok/s | **2.9x**   |
| quadratic b (full attn) | 1.13e-9     | 5.68e-10    | 2.0x         |

Cross-checks that make this a GEMM statement and not a guess:

* **w8a8 IS engaged.** bf16 weights give a = 5,794 tok/s; fp8 w8a8 gives 8,693.
  A real 1.50x. So the fp8 tensor-core path works and 8,693 is its ceiling here.
* **b is IDENTICAL (1.117e-9 vs 1.131e-9) between bf16 and fp8 weights**, which
  is what you expect when the quadratic term is bf16 attention in both. vLLM's
  2x-better b is its fp8 KV halving prefill attention bytes — the thing this
  session's fix now makes available to plow.
* **Not staging-latency bound.** The w8a8 GEMM arena is 49,152 B against the
  85,248 B the megakernel already reserves for flash prefill, so a deeper
  cp.async pipeline is free smem. Swept `PGM_STAGES` 3/4/5 x `PGM_GLU_STAGES`
  2/3: 12.43 / 12.43 / 12.49 s at 65k. **Zero.** Dead end, recorded as such.

### The GEMM number, against this GPU's real fp8 peak

`op_gemm.cuh` records the in-tree measurement (rtx-05): **fp8 peak 503.8
TFLOP/s** vs bf16 209.5, using `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
— f32 accumulate. Against that:

| engine | prefill GEMM | % of 503.8 peak |
|--------|--------------|-----------------|
| vLLM   | ~530 TFLOP/s | ~100% (at peak) |
| plow   | ~190 TFLOP/s | **38%**         |

Both use the same instruction class, so this is not an accumulate-precision
gap — it is 2.6x of headroom left on the table inside plow's own kernel.

**The structural reason: occupancy.** The megakernel's smem is the UNION over
every op it can run, and flash-prefill claims 85,248 B — so the prefill object
launches at `occ_per_sm = 1` (2 blocks/SM needs <=50,176 B). The GEMM therefore
runs with 8 warps/SM no matter how its own tiles are shaped. plow already HAS
the intended answer: the "lean occ-2 GEMM segment object" (`PLOW_NV_SEG_GEMM`,
with `PGM_BN` overridable to 64 exactly so its arena fits 2 blocks/SM under the
100 KiB cap — see the PX-3 note at op_gemm.cuh:712). It is unreachable from
`plowrt serve`: that path requires `check_coarse_single_segment()`, and a
segmented bucket disables prefill outright, which is why `PLOW_UNISEG=1` is
mandatory on sm_120.

So closing the prefill gap is **runtime work (make segmented dispatch reachable
from serve), not tuning**. Every knob in reach has now been swept and none of
them move it.

The second, independent problem is that plow prefills **serially** and does not
interleave prefill with decode. At concurrency 8 that alone is worth more than
the per-token GEMM gap.

## 6b. Final scoreboard (everything configurable, tuned)

| axis                            | plow          | vLLM      | verdict          |
|---------------------------------|---------------|-----------|------------------|
| max context, bf16 KV            | **132,096x4** | 15,280    | **plow, 8.6x**   |
| KV per sequence @132k           | **1.68 GiB**  | 3.66 GiB  | **plow, 2.2x**   |
| decode ITL @127k, conc 1        | 16.19 ms      | 16.07 ms  | tie (plow -0.7%) |
| prefill @127k                   | 31.0 s        | 14.2 s    | vLLM, 2.18x      |
| aggregate tok/s, 8x127k/1024out | 24.53         | 42.63     | vLLM, 1.74x      |

Decode was swept to convergence on the fp8-KV config at 127k, on BOTH its knobs:
NS_FULL_ABS 16 / 24 / **32** / 48   -> 18.62 / 17.01 / **16.19** / 17.13 ms
GF_FULL     1 / 2 / **4** / 8       -> 20.75 / 17.73 / **16.22** / 18.14 ms
GF_FULL=4 is the shipped value and it is confirmed optimal at 127k, not just at the
ctx=1024 where it was originally chosen. Both knobs are at their optimum; plow lands
0.9% behind vLLM on decode and there is nothing left to turn.
Prefill was swept over chunk (2048/4096/8192), KV dtype (bf16/mixed fp8),
prefill object (PIPE=0/PIPE=1 px4), and GEMM pipeline depth
(PGM_STAGES 3/4/5 x PGM_GLU_STAGES 2/3). 31.0 s is the floor.

## 7. Reproduce

    # cubins — PLAIN objects (PIPE=1, px4) for a bf16-KV long-context deployment
    PLOW_ROOT=$PWD PLOW_EXTRA_DEFINES="-DGV_MM_MAX=16 -DPLOW_NV_W8A8=1" \
      scripts/build_sm120_cubin.sh <dir>/interp_sm120.cubin
    # blob
    PLOW_UNISEG=1 PLOW_NS_FULL_ABS=32 PLOW_DECODE_BATCH=4 PLOW_FP8=1 PLOW_W8A8=1 \
    PLOW_MAX_CHUNK=2048 \
      plowc --hf-dir <fp8-ckpt> --gpu rtx5090 --emit devblob \
            --max-ctx 132096 --weight-dtype fp8 --out <dir>
    # serve
    PLOW_UNISEG=1 PLOW_DEV_SAMPLE=1 PLOW_MULTISTEP=32 \
      plowrt serve --assets <dir> --slo-ms <large>
    # bench
    ENGINE=plow PORT=8200 MODEL=<slug> bash perf-data/bench_longctx_5090.sh

## 8. Batched long-context decode — a third, separate deficit (measured, unfixed)

At 127k / conc 8 / 2048 out, plow aggregates **42.52 tok/s** (16,384 tokens). Subtracting the
~248 s prefill phase leaves ~137 s of decode, i.e. **~66 ms per B=8 step**. A pure-bandwidth
model says 25.4 GiB/step (12 GiB weights + 8 x 1.68 GiB KV) / 1.3 TB/s = **19.5 ms**.
plow's batched long-context decode is therefore **~3.4x off bandwidth** — a deficit that does
NOT show up at concurrency 1, where plow ties vLLM (16.19 vs 16.07 ms).

`NS_FULL_ABS` is not the lever. Swept at B=8 (ctx 32768, conc 8), including values that make
`B * ns` land on the 170-SM grid (`ns=21` -> 168 items):

| NS_FULL_ABS | 8     | 16    | 21    | 32    |
|-------------|-------|-------|-------|-------|
| agg tok/s   | 36.56 | 37.03 | 36.83 | 36.85 |

Flat to 1.3%. So the batched decode loses its 3.4x somewhere other than the full-layer split —
candidates not yet separated: the 40 sliding layers' per-slot flash at B=8, the GEMV ladder at
M=8 across 8 slots, or per-step host/gate overhead scaling with batch.

**Consequence for the campaign.** This is a THIRD independent gap, alongside prefill GEMM and
the hd512 flash prefill. It matters because it is the one that decides the multi-request
throughput number: fixing prefill alone would leave plow decoding 8 slots at 3.4x off
bandwidth. It should be attributed (PLOW_STEP_TIME at B=8 with a long KV) before any more
prefill kernel work, because it is likely cheaper to fix and it gates the same headline metric.

## 9. CORRECTNESS: fp8 KV loses the 67k needle — and the campaign measured the wrong prefill arm

Two findings from PX-8's gate runs (`perf-data/px8-flash-fp8-pv.md`), both of which change how
this document's earlier numbers should be read.

### 9a. The needle regression is the fp8 KV CACHE, not any flash arm

Four binaries differing only in the prefill arm, same 66.9k haystack:

| arm | KV | prefill arm | 66.9k needle | prefill ms |
|-----|----|-------------|--------------|-----------|
| `ref` | bf16 | PIPE=1 | **RETRIEVED** | 15643.5 |
| `px0` | fp8 | PIPE=0, no fp8 mma | MISS | 13843.3 |
| `px4` | fp8 | PIPE=1 fp8-mma | MISS (byte-identical stream to px0) | 10947.3 |
| `px8` | fp8 | + e4m3 P.V | MISS (byte-identical stream to px0) | 10781.6 |

All three fp8-KV arms emit the SAME 96 tokens, so the retrieval loss is upstream of the prefill
flash arm entirely — it is the e4m3 KV cache. bf16 KV retrieves; every fp8 KV arm misses.

**The 7.8k control (PX-8 Gate 2b) makes this attributable.** At 7.8k ALL FOUR arms retrieve —
bf16 and all three fp8-KV variants. So the 67k miss is NOT a broken fp8-KV build, which is the
other explanation the 67k table alone permits: it is a **context-scaling property of the e4m3
cache**, with the failure sitting somewhere between 7.8k and 66.9k. Locating that knee is the
first task in the revised order below.

Also measured there, and load-bearing for §9b: at 7.8k `px0` (the DEFAULT fp8-KV build) takes
**1670.3 ms against bf16's 1315.6 ms** — the PIPE=0 arm's inline dequant costs more than halved
KV bytes save. So the default fp8-KV configuration is a prefill regression at BOTH ends of the
context range, not just at long context.

**This is a direct hit on the campaign's headline.** The long-context capacity win (§1) is
predicated on fp8 KV: it is what makes B=8 at 127k fit in 31.36 GiB. If fp8 KV cannot retrieve
a needle at 67k, the capacity win is partly bought with retrieval quality, and "plow serves
127k where vLLM cannot" needs the qualifier that vLLM's fp8-KV path should be held to the same
test before any comparison is claimed. NOT yet attributed: which of the 40 sliding / 8 full
layers loses it, and whether it is the e4m3 rounding itself or the per-(token,kv_head) scale
granularity.

A first needle attempt was INVALID and is recorded as such: a source+markdown haystack with a
raw-token question made the **bf16 reference itself** degenerate into repeating the question.
A gate whose reference fails is not a gate. Rebuilt with benign filler + the chat template.

### 9b. PX-7's flash-prefill attribution measured the SLOW arm

`PLOW_FP8_KV=ON` alone forces the prefill objects to `PLOW_NV_FA_PIPE=0` (the fp8 prefill arm
dequants at the smem stage, so it only exists on synchronous staging and traps under PIPE=1).
`PLOW_FP8_KV_FASTPF=ON` is what keeps prefill on PIPE=1 and routes to the px4 fp8-mma arm.

> **RETRACTED by PX-12 (`px12-consolidated-baseline.md`): "It was OFF for this campaign's runs"
> was WRONG.** PX-12 measured the script-built PIPE=0 object at **175.90 s** against **32.39 s**
> for PIPE=1 on a 127k prefill — 5.4x. Since §2 recorded **33.09 s**, this campaign's serving
> runs were ALREADY on the fast arm; only objects built through `scripts/build_sm120_cubin.sh`
> (which hardcodes `-DPLOW_NV_FA_PIPE=0` for the fp8-KV prefill object) get the slow one.
> So FASTPF is **not** an unclaimed 21% on the headline cell — the build script is a landmine
> for anyone who rebuilds, but the numbers below were not taken through it. PX-8's 21% at 67k
> stands as a statement about the two ARMS, not about what this campaign ran.

Measured A/B, same packet, same fp8 cache: **13843.3 → 10947.3 ms = −21% total prefill at 67k**,
from one flag on an arm already in the tree.

So §6's "flash prefill is 16.48 s, 53% of prefill" was measured on the px0-class arm. Scaling
PX-8's isolated op ratio quadratically to 127k gives ~12.5 s — the same order, which is
consistent with PX-7 having measured px0 all along. The 53% share and the "1.80x behind vLLM"
figure derived from it are therefore attributed to the wrong arm and need re-measuring with
FASTPF on before any further flash-prefill kernel work is scheduled.

**Revised order of work:** (1) attribute the fp8-KV needle loss, (2) flip `PLOW_FP8_KV_FASTPF`
(−21%, one line), (3) re-measure the prefill attribution, (4) only then consider px8's e4m3 P.V
(+1.5% on total prefill, because FASTPF has already shrunk that op to ~5%).

## 10. §8 CORRECTED — the batched-decode gap is a BATCH gap, and two-thirds of it was a build flag

PX-10 (`perf-data/px10-batched-decode.md`) attributed §8 on the single-block harness. Three
corrections to what §8 claims.

**10a. "3.4× off bandwidth" was wrong, and so was "long-context".** §8 derived its ratio by
subtracting an assumed prefill cost from an end-to-end aggregate; that subtraction was too
small. Measured device time at B=8/131k is **42.5 ms**, not 66 ms — **2.50× off** the 17.06 ms
floor, not 3.4×. And it is **not a long-context effect at all**: at ctx **1024**, with
essentially no KV, B=8 is already **2.33× off**. Going to 127k moves it only to 2.50×. The whole
deficit is one discontinuity at **B=4 → B=8**; B=1/2/4 track the floor at 1.13–1.83×.

**10b. The bandwidth model is right.** §8 hedged that the model might be wrong. It is not:
plow's own M=1 GEMV streams fp8 weights at **1495 GB/s**, 83% of the 1792 GB/s pin.

**10c. Attribution, and a free 19–34%.** Device step at B=8/131k:

| component | B=1 | B=8 | GB/s @B=8 | floor |
|---|---|---|---|---|
| GEMV ladder | 9.47 | **21.00** | 519 | 8.38 |
| full-layer flash (fp8 KV) | 2.23 | **16.31** | 527 | 6.61 |
| full-layer FlashMerge (gate) | ~0 | 2.20 | — | — |
| sliding-layer flash | 0.53 | 2.31 | **1161** | 2.06 |
| norms/rope | 0.05 | 0.66 | — | — |
| **total** | **12.29** | **42.49** | 522 | **17.06** |

The sliding layers are AT bandwidth and are 5% of the step — §8's candidate (a) is dead.
`NS_FULL_ABS` is confirmed inert from the device side.

Two-thirds of the GEMV penalty is **`-DGV_MM_MAX=16` in the asset this campaign built** (§7).
It does not change which rung runs (still `gv_mm<8>`) — it changes how `rows` ARRIVES:
`gemv_walk`'s compile-time-full loop passes a literal `8`, its remainder arm passes a runtime
value, so `if (m >= M) continue;` inside the innermost `#pragma unroll` stops folding and
**every FMA in the hottest loop in the model becomes predicated**.

| ctx | `=16` (what we built) | `=8` (the source default) | Δ |
|---|---|---|---|
| 1024 | 6106.5 µs | 4040.9 µs | **−33.8%** |
| 131072 | 10649.2 µs | 8588.1 µs | **−19.4%** |

`op_gemm.cuh`'s own comment already says `=16` is only for deployments pinned at B≥16. §7 built
the B=8 asset with it anyway, so **every decode number in this document is on a mis-built
cubin**. Rebuilding takes B=8/131k from 42.60 → **34.35 ms**, 2.50× → **2.01×** off floor.

The remaining half is the fp8-KV full-layer flash at **527 GB/s, 2.5× off at every batch size**
— not a batching regression, batching just multiplies it by 8. bf16 KV is **worse** (+25% at
B=8), so that one needs a kernel, not a flag. PX-11 is measuring it.

**10d. Open bug, found by PX-10, not diagnosed.** With `PLOW_FP8_KV=1` the hd512 full-layer
**prefill crashes at bucket ≥ 4096** (`CUDA_ERROR_LAUNCH_FAILED`). Worked around with
`--pf-chunk 2048`. Not hit by the window-derived 1024-row default (§ README `PLOW_MAX_CHUNK`),
which is why this campaign never saw it — but any deployment raising the chunk with fp8 KV will.


## 11. PX-11 — the fp8 full-layer flash-decode is an ACCESS-MAP problem, and three flags fix most of it

`perf-data/px11-flash-decode.md`. §10 left one kernel-level number unexplained: the hd512
fp8-KV flash-decode at 527 GB/s while plow's own GEMV hits 1495 GB/s. PX-11 isolated it.

**The body is not the problem.** At `GF_FULL=8` the kernel runs at **87-93% of the ceiling its
own access map allows**. The MAP is the wall: a row-per-thread score phase caps at 1263 GB/s on
a 512 B fp8 row (vs 1700 coalesced), and the shipped `GF_FULL=4` demands every KV byte **4x** on
a model with ONE kv head.

| config | ms @ B=8/131k | vs deployed |
|---|---|---|
| deployed `GF_FULL=4`, ns=32 | 2.6918 | 1.00x |
| `-DPLOW_NV_FA_GF_FULL=8`, ns=21 | 1.7656 | **1.52x** |
| `+ -DPLOW_FP8_LD16 -DPLOW_FP8_FAST` | **1.6710** | **1.61x** |

All bit-exact (GF at fixed nsplit: `maxdiff 0.000e+00`, 100+ cells; LD16+FAST: 65536/65536
identical), registers 241 -> 241 with 0 spills. **Nothing in the tree sets LD16/FAST today.**

**Two in-tree corrections.** Occupancy is NOT a limiter here (occ1 == occ4). And
`rtx19-e4-tc-fp8-decode.md`'s "achievable fp8-byte bandwidth is ~55-62% of the 1535 GB/s bf16
wall" is an **MLP artifact of a non-unrolled probe, not hardware**: at 1 load in flight 8-byte
vs 16-byte reads give 1033 vs 1559 GB/s, but at 8 loads in flight BOTH reach 1697. Any sizing
that used the 55-62% figure is wrong.

**Combined projection, DERIVED not measured — this is the open gate.** Folding PX-11's op win
through PX-10's reconstruction: 42.6 -> ~36.4 ms, or **~28.2 ms with PX-10's `GV_MM_MAX` fix**,
i.e. 2.50x -> **~1.65x** off floor. Nobody has yet run a real serve at B=8/131k with both sets
of flags. That end-to-end A/B is the next action, and it must include greedy parity, because
"bit-exact per op" does not by itself prove an unchanged token stream through 48 layers.


## 12. CONSOLIDATED BASELINE (PX-12) — plow still loses, 1.64x, and the bound is arithmetic

Same cell as §2b, both engines re-run in one session: 8 x 126,976 in / 1024 out, conc 8,
`--ignore-eos --seed 0`, identical 1,015,914 in / 8,192 out on every row.

| engine / arm | aggregate out tok/s | wall (s) | median ITL (ms) |
|---|---|---|---|
| **vLLM 0.26.0** (same session) | **42.49** | **192.8** | 18.87 |
| plow A — control (deployed flags) | 23.76 | 344.8 | 63.97 |
| **plow B — tuned** (`GV_MM_MAX`=8 + `GF_FULL=8` + `LD16` + `FAST`) | **25.96** | **315.6** | **43.16** |
| *§2b as recorded — plow* | *24.53* | *~334* | *53.22* |
| *§2b as recorded — vLLM* | *42.63* | *192* | *18.85* |

Both columns reproduce in-session (plow control within 3%, vLLM within 0.3%), so §2b was sound.

**Verdict: plow LOSES 1.64x** (was 1.74x). Tuned - control = **+9.3% aggregate, 1.48x median
ITL**, and greedy parity control-vs-tuned is **identical token ids** — the end-to-end gate §11
left open is now CLOSED, and the 1.48x matches PX-11's isolated `GF=2->8` ratio at the packet's
`ns=32`.

**Why no decode flag can win this cell.** 82% of plow's wall is serial prefill (8 x 32.4 s, not
overlapped with decode). **A decode step of zero still leaves plow 1.34x behind.** Every
remaining decode lever - `NS_FULL_ABS=21`, px8's e4m3 P.V, the flash-decode access map - is
bounded by that 1.34x. The headline number is a PREFILL problem and always was; §2b said so and
the intervening kernel work did not change it.

### 12b. Three corrections to earlier sections

1. **`/root/plow-out/lc-b8` is NOT the §2b asset.** It is an **all-layer** fp8-KV packet
   (1.333 GiB/seq); §2b describes a **mixed** one (1.68). Proven twice: KV arithmetic, and a
   PIPE=1 fp8 prefill object trapping at an 18-token prompt on the hd256 arm. PX-12 re-emitted
   the mixed packet (13.13 GiB = 1.641 GiB/seq). Anything measured on `lc-b8` and compared to
   §2b was comparing two different packets.
2. **§9b's "FASTPF was OFF for this campaign" is retracted** — see the call-out in §9b.
3. **The deployed `GF_FULL` is 2, not 4.** `build_sm120_cubin.sh` passes
   `-DPLOW_NV_FA_GF_FULL=4` only to the PLAIN decode object; the `_fp8kv` object that actually
   loads omits it and falls back to 2. Every PX-11 ratio quoted against "deployed GF=4"
   therefore UNDERSTATES the win.

Provenance gate: PX-12's control build is **md5-identical** to the shipped `interp_sm120.cubin`,
and each of the four `-D` flags provably changes the cubin.

**Not run:** the GEMV-vs-flash split of the +9.3%; the `NS_FULL_ABS=21` re-emit (a further 1.29x
on the op, NOT bit-exact, needs its own greedy gate).


## 13. PX-12 arm E — the prefill object had never been tuned, and FP8PV is the biggest lever

Follow-up to §12, same cell. §12 tuned the DECODE object only; the prefill object had never been
given flags of its own (`GV_MM_MAX`/`GF_FULL`/`LD16`/`FAST` are provably inert in it — identical
md5 with and without).

| engine / arm | out tok/s | wall (s) | median ITL (ms) | greedy parity |
|---|---|---|---|---|
| **vLLM 0.26.0** (same session) | **42.49** | 192.8 | 18.87 | — |
| plow A — control | 23.76 | 344.8 | 63.97 | reference |
| plow B — tuned decode | 25.96 | 315.6 | 43.16 | **PASS** |
| **plow E — + tuned prefill** | **29.91** | **273.9** | 40.39 | **FAIL @ tok 28** |

**Verdict: plow still loses — 1.64x if you require an unchanged token stream (arm B), 1.42x if
you accept `FP8PV`'s numerics change (arm E).** Was 1.74x.

127k conc-1 prefill: B **32.39 s** -> +PX-6 A **32.51 (inert)** -> +`FP8PV` **27.59 s, 1.18x**.

### 13a. Two of my estimates were wrong, both the same way

§9b ranked `PLOW_NV_FA_FP8PV` **last**, at "+1.5% on total prefill". It is the **largest single
lever in the campaign** (1.18x on a 127k prefill, and the whole A->E gain above). §9b also
ranked PX-6 A (`PLOW_NV_PF_GEMV_HEAD`) ahead of it; PX-6 A is **inert** here — its -39% is
per-launch on `lm_head`, ~0.3% of a 127k prefill at chunk 1024.

Both errors came from the same move: **scaling an isolated-op ratio through an ASSUMED prefill
budget** instead of measuring end to end. The op ratios were right; the budgets they were scaled
through were not. Isolated-op ratios rank levers only when the denominators are measured.

Related: `FP8PV` was **unreachable by construction** until FASTPF was enabled —
`op_attention.cuh:1059` `#error`s without `PIPE=1`, and `scripts/build_sm120_cubin.sh` hardcodes
`PIPE=0` for the fp8-KV prefill object. The lever existed in the tree and no build could select it.

### 13b. Arm E is NOT shippable yet

`FP8PV` diverges from the parity reference at completion token 28 ("persistent" -> "steady").
Both continuations are fluent and the first 28 tokens are bit-identical, but **one 34-token
sample is not a quality gate** — and §9a already shows this model's fp8 KV losing a 67k needle.
Arm E needs a retrieval test before it ships. Arm B is the parity-preserving configuration.

### 13c. The bound, restated

81% of arm E's wall is still serial prefill. **A decode step of zero leaves arm E 1.14x behind.**
No remaining flag closes this. What is left is runtime work: the prefill GEMM (38% of fp8 peak at
occupancy 1) and the missing prefill/decode overlap (`plans/mixed-batching.md`).


## 14. PX-14 — batched prefill is a NO-OP on this cell, and a campaign invariant is FALSE

`perf-data/px14-batched-prefill-fp8.md`. Two results, one negative and one that invalidates a
gate we have been relying on.

### 14a. `PLOW_PF_BATCH` cannot pay here — measured, not argued

The premise (§2b: prefill is serial across 8 requests, so pack them) does not survive contact
with the mux's budget. `pf_max_rows()` = the largest prefill bucket = `max_chunk` = **1024**, and
the serialized path already fills it exactly: 126,976 = 124 x 1024, with no tail to co-pack.
Packing 8 requests splits the SAME 1024 rows 8 ways — same 992 launches, same M, same flash work.
**Enabling `PLOW_PF_BATCH` for fp8-KV is a measured no-op on this cell.** The fp8 blocker (t6/t7
exhaustion) is real and cheap to fix; it is simply not what is costing us anything here, so the
kernarg move was correctly NOT implemented.

Ceiling of the whole idea, as an upper bound rather than a projection: **<= 33.2 tok/s** (from
29.91) — vLLM still **1.28x** ahead.

The one genuine argument for packing is different from the brief's: a pack of 8 x 1024 rows
reaches **M = 8192** while each request writes <= 1024 rows into its own 2048-row ring, so KV
stays 13.13 GiB — a big-M launch that VRAM forbids serially. Getting it requires decoupling
`max_chunk` (which today sets BOTH the ladder top and the sliding ring) — unscoped devgen work.

Launch count is now priced and closed. Chunk sweep, four packets differing only in
`PLOW_MAX_CHUNK`, one 126,976-token prompt, cubins fixed at px12 arm E:

| chunk | launches | wall | vs 1024 | KV @ B=8 |
|---|---|---|---|---|
| 1024 (deployed) | 124 | **27.51 s** | 1.00x | 13.13 GiB OK |
| 2048 | 62 | 26.32 s | 1.045x | 18.13 — **planner refuses** (needs 31.82 of 30.86 free) |
| 4096 | 31 | 24.63 s | 1.117x | 28.2 X |
| 8192 | 16 | **24.08 s** | 1.142x | 48.2 X |

Per-launch fixed cost is **~32 ms**, not the 60.1 ms the 8k-era regression predicts. The whole
lever is 12.5% of prefill, and at B=8 everything above chunk 1024 is VRAM-blocked.

### 14b. FALSE INVARIANT: chunk boundaries DO change the token stream

`mux.rs` claims a request's tokens cannot depend on how prefill is chunked. **They can.** Two
packets differing only in `PLOW_MAX_CHUNK` diverge at completion token **6** (arm E) and token
**11** (arm C, which has NO `FP8PV`) — so this is not px12's FP8PV numerics. Both continuations
are fluent; the cause is flash merge order, not corruption.

**Consequence for every parity gate in this document.** Greedy-token parity is only meaningful
between runs with the SAME chunk. Any A/B that varied chunking and reported a divergence index
was measuring this, not the change under test — and §13b's "FP8PV diverges at token 28" needs
re-checking at fixed chunk before it is attributed to FP8PV.

### 14c. PX-10's fp8-KV prefill crash did NOT reproduce

Buckets 4096 and 8192 both served a 127k prompt cleanly on the mixed packet with the FASTPF
object. §10d's report stands as observed but is not reproducible under these conditions; treat it
as unexplained rather than as a known bug.


## 15. PX-20 — matched precision. §9a is WRONG, §2's prefill claim is WRONG, and the real gap moves.

`perf-data/px20-matched-precision.md`. Both engines re-run at MATCHED KV dtype, no mixed packet.

### 15a. RETRACTED: fp8 KV does NOT lose the 67k needle

§9a claimed bf16 KV retrieves a 66.9k needle and every fp8-KV arm misses. **All four cells
RETRIEVE** (identical `prompt_tokens=66901` both engines): vLLM e4m3, vLLM bf16, plow all-layer
e4m3, plow bf16.

**Cause: a SECOND live copy of the PX-17 patch-site bug**, in `runtime/tests/gemma4_sm120_chat.cu`
— the harness PX-8 used for its gates. Its per-chunk loop matches `PLOW_DOP_HEADNORM_ROPE` /
`FLASH_PREFILL` but NOT the `_FP8` twins (opcodes 37/39), so every fp8-KV chunk after the first
wrote KV at row 0. PX-8's own raw timings carry the fingerprint: the same fp8 arm is 1.27x
**slower** than bf16 at 7.8k and 1.43x **faster** at 66.9k. Re-running PX-8's exact mixed packet
through fixed plowrt: **RETRIEVED**.

**Nobody's capacity is bought with retrieval quality.** The bug is still live in that harness —
NOT fixed, because fixing it invalidates PX-8's binaries.

### 15b. The gap at matched precision — it is not one number

| 127k cell (8 × 126,976 / 1024, conc 8) | vLLM | plow | gap |
|---|---|---|---|
| matched **all-layer e4m3** | 42.55 | **5.59** | vLLM **7.61x** |
| matched **bf16** (at 16,384 — see VRAM) | 25.01 | 22.66 | vLLM **1.10x** |
| *unmatched headline (plow mixed)* | *42.49* | *29.91* | *1.42x* |

**All-fp8 is catastrophic and structural.** An all-layer e4m3 packet emits hd256
`FLASH_PREFILL_FP8`; the PIPE=1 px4 arm `__trap()`s on hd256 (`interp_sm120.cu:846`), so plow
falls back to PIPE=0 at **176 s of prefill per request** (reproduces PX-12's 177.50 s to 0.6%).
**plow has NO fast prefill arm for the configuration vLLM ships by default.** That is the single
largest gap in the campaign and it was invisible while we benchmarked a mixed packet.

**VRAM:** 8 × 127k at bf16 fits on NEITHER engine — vLLM admits 4 of 8, plow 6. Row B ran at
16,384 where both held 8 with 0 preemptions.

### 15c. RETRACTED: "prefill is 2.2–2.6x behind"

§2 compared plow's **bf16** prefill against vLLM's **fp8** prefill. vLLM's prefill slows 1.69x at
bf16. **At matched dtype, 127k prefill is 34.9 s (plow) vs 35.8 s (vLLM) — a TIE.**

### 15d. What actually survives: batched decode

At conc 1 plow **WINS** all three bf16 contexts (1.07x / 1.09x / 1.13x) and has the lower median
ITL in **all six** conc-1 cells. vLLM's conc-1 127k ITL is 16.07 ms — matching the campaign anchor.

conc1 → conc8 at 127k bf16 moves **plow's median ITL 3.52x** and **vLLM's 1.03x**. That is the
deficit, and it is scheduling, not kernels: plow's per-request work is competitive or better;
what it cannot do is hold quality of service as the batch grows.

### 15e. Gate closed

7,826-token needle, all-layer e4m3, chunk 1024 = **8 prefill launches → RETRIEVED**; also at
66,901 (66 launches) and on the mixed packet. **PX-17's fix is now verified by retrieval**, not
only by code path.
