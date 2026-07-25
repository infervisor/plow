# Gemma-4-12B fp8 DECODE — plow beats vLLM by a clear margin on sm_120 (beat12b-fp8-margin)

**Campaign beat12b-fp8-margin** · branch `beat12b-fp8-margin` · **date 2026-07-23**
GPU: 1× RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SMs, 96 GB). CUDA 13.0.
Goal: beat vLLM by **>=10 % decode TPOT at every ctx 1k..128k** (stretch 15 %) on
Gemma-4-12B, **fp8 weight-only e4m3** (no 4-bit), using plow's compiler (packet
structure) and kernel/interpreter levers. **Result: 12.4 %–23.4 % at every rung —
target met everywhere, stretch met at 4k..128k.**

## HEADLINE — decode TPOT ms/token (mean, n=112), margin vs vLLM's best fp8 config

plow config: fp8 weights + fp8 lm_head (E5) + fp8-KV (`PLOW_FP8_KV=1`, `PLOW_FP8_FAST`
cubin) + this campaign's two grid-alignment nsplit levers (now the emitter DEFAULT for
fp8-KV packets). vLLM: `gemma4-12b-vllm-sm120.md` (trusted baseline, 0.25.1, TRITON_ATTN,
cudagraphs ON); best = min(fp8, fp8kv) per ctx.

| ctx | plow (sd) | vLLM fp8 | vLLM fp8kv | vs fp8 | vs fp8kv | **vs BEST** |
|------:|--------------:|---------:|-----------:|-------:|---------:|--------:|
| 1024 | **10.918** (0.013) | 12.46 | 12.82 | −12.4 % | −14.8 % | **−12.4 %** |
| 4096 | **10.972** (0.013) | 12.97 | 13.09 | −15.4 % | −16.2 % | **−15.4 %** |
| 16384 | **11.209** (0.016) | 14.27 | 13.95 | −21.5 % | −19.7 % | **−19.7 %** |
| 32768 | **11.535** (0.013) | 15.98 | 15.05 | −27.8 % | −23.4 % | **−23.4 %** |
| 65536 | **12.198** (0.013) | 17.47 | 15.78 | −30.2 % | −22.7 % | **−22.7 %** |
| 98304 | **12.877** (0.013) | 18.91 | 16.57 | −31.9 % | −22.3 % | **−22.3 %** |
| 131072 | **13.691** (0.014) | 20.71 | 17.30 | −33.9 % | −20.9 % | **−20.9 %** |

Same-config baseline BEFORE this campaign's levers (identical build/pkt flags, nsplit at
the old CU-fill default ns=24): 11.219 / 11.295 / 11.761 / 12.401 / 13.625 / 14.897 /
16.160 ms — i.e. the old config only cleared −6.6 % vs vLLM fp8kv at 128k and −10.0 %
at 1k (borderline). **The two nsplit levers moved 128k by −2.47 ms (−15.3 %) and 1k by
−0.30 ms.**

## Method (repo convention, per gemma4-12b-plow-sm120-decode.md)

- Harness `runtime/tests/gemma4_sm120_chat.cu`, prompt = exact vLLM RandomDataset
  seed-0 ids (`perf-data/harness/make_vllm_random_ids.py`, p0), prompt consumed via
  PREFILL buckets, then 128 generated tokens, first 16 discarded → **n=112 timed**,
  mean/median/sd reported; `argmax check device==host AGREE` at every point.
- Build: `cmake -DPLOW_CUDA=ON -DPLOW_FP8_KV=ON -DCMAKE_CUDA_FLAGS=-DPLOW_FP8_FAST`
  (decode megakernel 209 regs, GF=2). Packet: `gemma4 <model> 132096 out.pkt 188` with
  `PLOW_UNISEG=1 PLOW_FP8=1 PLOW_FP8_HEAD=1 PLOW_FP8_KV=1` (one pkt serves all ctx).
- fp8 twins `/workspace/models/gemma-4-12B-it/fp8` + the E5 head twin regenerated with
  `perf-data/harness/quantize_fp8_head.py` (per-row amax/448 e4m3), presented to the
  harness as a 2-shard symlink dir (st_open only probes model[-NNNNN-of-NNNNN].safetensors).
- Every GPU command under `gpulease beat12b-fp8`.

## Trace ranking (PLOW_NV_TRACE decode block-0, fp8kv ns47 pkt)

| op | ctx 1k share | ctx 128k share |
|----|-------------:|---------------:|
| GemvFp8 (30) — qkv/o/lm_head | 54.1 % (body 11.41 M cyc) | 44.4 % |
| GemvGluFp8 (31) — gate/up/down | 38.1 % (9.43 M) | 30.5 % |
| FlashDecodeFp8 (38) | 7.5 % (1.39 M) | 24.8 % (7.72 M) |

GEMV aggregate ≈ 8.6 ms vs the ~7.8 GB fp8 weight-read floor (≈7.8 ms at the measured
fp8 cold-read ceilings) → the weight arms run at ~90 % of their aggregate ceiling; the
big exploitable slack was flash-decode occupancy at long ctx and the fixed sliding-layer
flash tax — both fixed below.

## Levers

### GO 1 — full-layer flash-decode nsplit grid-alignment (emitter DEFAULT, commit c8e84ee)

12B full layers have **ONE global kv head** (kvh_full=1, 16 q heads, hd 512). The CU-fill
formula emitted ns=24 → n_work = n_grp(8)×24 = **192 items on 188 SMs**: 4 blocks run 2
items, 184 run 1, FLASH_MERGE waits for the 2× tail. Rounding to n_cu/gcd(n_grp,n_cu)=47
(376 = exactly 2/block) fixes it.

- Microbench `runtime/tests/flashdec_fp8_bw_12b.cu` (real 12B geometry, flash+merge ms
  per full layer, fp8 KV, min/40): @128k ns24 0.553 → ns23 0.417 → **ns47 0.304 (−45 %)**;
  @64k 0.283→0.163; @1k free (+0.001). Ragged 192-item configs ~2× their aligned
  neighbours at every GF — alignment, not split count, is the lever. Merge ≤0.02 ms at
  every nsplit (16 merge items).
- Network: @128k 16.163 → **13.988 (−13.5 %)**; @1k 11.219 → 11.213 (free).
- Gated on the emit-time `PLOW_FP8_KV` flag + kvh_full==1 signature: bf16-KV optimum
  differs (microbench @128k bf16 prefers ns23: 0.436 vs ns47 0.497), and flag-unset
  packets stay **byte-identical** (cmp-verified vs pre-change emitter, bf16 + fp8-no-KV).

### GO 2 — windowed-layer nsplit cap (emitter DEFAULT, commit 8dffecf)

The 40 sliding layers' flash span is window-capped (1024 rows) — their flash cost is
**ctx-independent, part of the fixed intercept** — yet they were split ns=24 into 43-row
items on the same ragged 192/188 grid. Cap ns at win/64 (=16; n_work 128, no 2× tail,
half the merge partials).

- Sweep (full ns=47 held, ms/tok @1k): ns8 10.937 | ns12 10.956 | **ns16 10.921** |
  ns23 10.990 | ns24(old) 11.221 | ns47 11.212.
- **Fixed −0.30 ms at every ctx** (@128k 13.978 → 13.684). Same fp8_kv gate, flag-unset
  byte-identical; `PLOW_NS_ABS`/`PLOW_NS_FULL_ABS` still override both levers.

### Measured, NOT shipped as default — GF_FULL=4/8 kernel rebuild (long-ctx option)

Raising the flash GQA fusion (each KV byte read 8×/GF at kvh=1) needs a cubin rebuild
(`-DPLOW_NV_FA_GF_FULL=N`) and taxes the whole megakernel's registers:

| config | regs | ms/tok @1k | @128k |
|---|---:|---:|---:|
| GF2 / ns47 (shipped) | 209 | 11.213 | 13.988 |
| GF4 / ns94 | 219 | 11.468 (+0.26) | 13.502 (−0.49) |
| GF8 / ns94 | 234 | 11.593 (+0.37) | 13.255 (−0.73) |

(sweep taken before GO 2; deltas transfer.) GF8 would trade the 1k margin down to ~7 %
for −5 % more at 128k — wrong trade for the every-rung target. Documented for
long-ctx-dominant serving; microbench rows in `flashdec_fp8_bw_12b.cu` output. NOTE the
family-assets combo GF4+`PLOW_NS_FULL_ABS=48` (scripts/build_gemma_family_sm120.sh) is
RAGGED on 12B (192 items): with a GF4 cubin use ns 47 or 94, never 48.

### NO-GO — smem x-staging for the long-K GEMV arms (reverted)

The two slowest fp8 GEMV arms (E4: down 640 GB/s, o_proj 466 vs gate/up 823) are exactly
the K>6176 shapes that fall back to the global-x body. Growing the arena
(`PLOW_NV_GEMV_ARENA_K=15360`, 30.7 KB smem, still 1 block/SM, 209 regs, staged body
byte-identical outputs) **regressed**: @1k 11.55 (+0.34), @128k 14.30 (+0.32) — the
staging syncs + L1-carveout loss beat the saved global-x traffic. Reverted; op test was
ok; default SASS byte-identity was verified before revert.

### N/A — f32-direct dequant in the fp8 weight GEMV (premise already satisfied)

`dot8_fp8` (op_gemm.cuh:1648) already dequants e4m3→half2→f32 with the native cvt — no
bf16 round-trip exists in the weight-GEMV arms. The round-trip the campaign brief
remembered was in the fp8-KV **flash** arm and was already fixed by `PLOW_FP8_FAST`
(beat26b), which this campaign's cubin builds with.

### Reasoned skip — emit-time pre-swizzled fp8 weight layout

GEMV weight reads are already perfectly coalesced (contiguous 256 B/warp-pass); the
in-network GEMV aggregate runs at ~90 % of the fp8-byte ceiling, and E4 shows the fp8
cold-read ceiling itself (853–958 GB/s on these tensors), not dequant placement, binds.
Upside bounded <0.9 ms with a layout-migration risk across every fp8 consumer; the
targeted x-stage attempt for the same arms measured a regression. Not pursued.

### Inherited GO levers (in the shipped config)

- **E5 fp8 lm_head** (`PLOW_FP8_HEAD=1`, rtx19-e5-lmhead.md): −0.64 ms fixed. Head twin
  generator now committed (`perf-data/harness/quantize_fp8_head.py`).
- **fp8-KV + FP8_FAST** (beat26b): halves KV bytes; f32-direct flash dequant.

## Correctness gates

- `sm120_interp_op_test: ok` on the ladder build (GF2 + PLOW_FP8_KV + PLOW_FP8_FAST);
  kernels untouched by the shipped levers (emitter-only).
- **Default-build identity**: with `PLOW_FP8_KV` unset the emitted packet is
  byte-identical to the pre-campaign emitter (cmp, bf16 and fp8-no-KV); the one kernel
  experiment (x-stage) was reverted and its flag-off SASS was md5-identical.
- **argmax device==host AGREE** at all 7 ladder points, both configs.
- **Near-tie class (nsplit changes flash-merge fp summation order):** step-0 logits
  **byte-identical** ns24 vs ns47 vs GF8 at ctx 1k & 4k (relL2 = 0, maxabs = 0); on the
  128k random-token prompt (high-entropy logits) greedy flips within a few steps; on the
  natural-text prompt (`p_real.ids`) streams agree 18 tokens then flip — same class and
  magnitude as the documented fp8-KV drift (rtx19-e3: ~21 tokens). Microbench merged
  outputs maxdiff 0.0000 vs shipped config at every swept (GF, nsplit, ctx).
- Post-merge (origin/worktree-gpu-exec-stage1, host-side plowrt): plowc blob
  byte-identical; plowrt `--features cuda,hf-tokenizer` compiles; chat-harness numbers
  unaffected (it does not link plowrt).

## Reproduce

```
# build (outside nix develop)
cmake -S runtime -B build-fp8kv -DPLOW_CUDA=ON -DPLOW_FP8_KV=ON \
      -DCMAKE_CUDA_FLAGS="-DPLOW_FP8_FAST" -DCMAKE_BUILD_TYPE=Release
cmake --build build-fp8kv --target gemma4_sm120_chat sm120_interp_op_test -j8
# packet (nsplit levers are the DEFAULT under PLOW_FP8_KV=1)
PLOW_UNISEG=1 PLOW_FP8=1 PLOW_FP8_HEAD=1 PLOW_FP8_KV=1 \
  target/release/gemma4 /workspace/models/gemma-4-12B-it 132096 g12b.pkt 188
# head twin (once) + 2-shard dir the harness can probe
python perf-data/harness/quantize_fp8_head.py /workspace/models/gemma-4-12B-it head.safetensors
#   fp8dir/model-00001-of-00002.safetensors -> <model>/fp8/model.safetensors
#   fp8dir/model-00002-of-00002.safetensors -> head.safetensors
# one ladder point
gpulease <tag> env PLOW_WARMUP=16 PLOW_FP8_DIR=<fp8dir> \
  build-fp8kv/gemma4_sm120_chat g12b.pkt /workspace/models/gemma-4-12B-it ids_131072_p0.bin 128
```

## Left on the table

- **GF8 cubin at long ctx** (−0.73 ms @128k) — blocked on its +0.37 ms register tax at
  short ctx; a per-arm register cap or a flash-only GF split could unlock both ends.
- **bf16-KV alignment** (ns23 full layers): microbench −38 % @128k flash vs today's
  bf16 default; not wired (campaign is fp8; bf16 default must stay byte-identical).
- **fp8 GEMV last ~10 %**: o_proj (59 % of its cold-read ceiling) and down (67 %) still
  lag gate/up (86 %); x-staging is disproven, TC-fp8 loses at B=1 (E4) — the remaining
  ideas are load-width (uint4) + N-partition tail shaping, unproven.
- **Served-path TPOT** with the newly merged bounded device multi-step decode
  (gpu-exec-stage1): cuts host round-trips on `plowrt serve`; not measured here (chat
  harness ≈ kernel-only, host_ms 0.04). A serve A/B is the natural next spot-check.
- vLLM re-baseline on a newer release; 3-prompt averaging (plow sd ≤0.016 ms makes
  prompt variance negligible for decode).
