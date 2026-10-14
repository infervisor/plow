# GLM-5.3-Flash on 8x MI350X (gfx950) — bring-up and serving campaign (2026-10-12)

Recipe: `recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-16k-fp8.toml`. Campaign plan and
research notes: `plans/glm53-flash-dsv41-mi350x.md` (local). Baseline to beat: vLLM on the
`vllm/vllm-openai-rocm` nightly image (AITER, TP4/TP8), measured on this box with the same client.

## 1. What GLM-5.3-Flash is, for the emitter

`zai-org/GLM-5.3-Flash`, `model_type: glm5_next` (text tower `glm5_next_text` under `text_config`,
tensors under `model.language_model.`), 320B total / 18B active, 45 layers: 34 KDA linear-attention
layers (Kimi-Linear tensor names: `q/k/v_proj`, `*_conv1d`, `f_a/f_b_proj`, `g_a/g_b_proj`, `b_proj`,
`A_log`, `dt_bias`, `o_norm`) and 11 NoPE sparse-MLA layers at 3, 7, ..., 43 (`q_a/q_b`, `kv_a_proj_with_mqa`,
`kv_b` bf16, `o_proj`, kpool-4 indexer with `index_kpool_compress_{ape,gate}`, top-2048 tokens);
mHC residuals (4 streams, 20 Sinkhorn iterations, eps 1e-6, mean exit); 3 dense FFN layers then
288-expert / top-8 / 1-shared MoE with sigmoid noaux_tc routing, scale 2.5, **clamped SwiGLU
(`swiglu_limit` 10)**; vocab 154880; block-FP8 e4m3 [128,128] projections and experts, bf16 KDA
projections, with `quantization_config` at the ROOT of the wrapper config. A vision tower and an
MTP layer (`layers.45.*`) ship in the checkpoint and are ignored. It is not GLM-5.3 (`glm_moe_dsa`,
78 layers, no mHC): `docs/amd/glm53-mi300x.md` §1.

## 2. Emitter state and what this campaign changed

`mla::glm53_emit_full` (commit ba857841, 2026-09-02) emits the full model; the last bring-up
(98a369ed, 2026-09-04) reached a correct first token at TP4/TP8 on gfx950 with decode degenerating
afterwards. Changes made here before the first serve on this box:

| change | where | why |
|---|---|---|
| clamped SwiGLU on every GLU (`ACT_SWIGLU_CLAMP`, `f1` = limit) when the config carries `swiglu_limit` | `crates/devgen/src/mla.rs` (`glm_glu_act`, 13 pair-form sites; the 7 gate-only bf16/MXFP4 GEMM/GEMV GLU sites refuse), `crates/packet/src/dev.rs` | every FFN of every layer differed from the reference in the tail of `up`; the kernel arm existed for DeepSeek-V4 but no emitter set it |
| `quantization_config` read from the root when `text_config` has none | `mla_ckpt_enc` | the Flash config keeps it at the root, so the emit chose **bf16** expert ops (`MoeExpertGlu`, `GemvGlu`) for a block-FP8 checkpoint |
| `index_kpool` capability probe also reads `text_config` | `crates/devgen/src/lib.rs` | the `indexer_pooled` knob cap never registered for the nested config |
| mHC constants and pooled-indexer flags asserted against the config | `cfg_glm` | the emit hard-codes n=4 / 20 iterations / 1e-6 and always compresses + appends the tail |
| KDA carried state sized to the decode ladder (`dbatch` slots), as Kimi-K3 does | `declare_glm53` | one slot made every batched decode row share one recurrence |
| bf16 shared expert GLU split into two GEMM/GEMV halves + the pair-form clamped `Glu` when a limit is set (prefill, batched decode, ladder-1 decode) | shared-expert sites in `mla.rs` | the prep dequantises the shared expert to bf16, whose fused `GemmGlu`/`GemvGlu` epilogue is gate-only and has no clamp arm; `GLM_LINEAR_FP8` (fp8 shared expert) hung on gfx942 in the GLM-5.3 campaign, so it stays a later lever |

Test: `cargo test -p devgen --lib glm5_next_cfg_reads_root_quantization_and_clamped_swiglu`.

## 3. Build chain on this box

```bash
# CPU, inside `nix develop` (worktree build: cargo build --release -p plowc; -p plowrt --features hsa)
nix develop .#quantize --command python3 scripts/glm53_prep.py --model /home/lava/models/GLM-5.3-Flash \
    --out /home/lava/models/plow-glm53-flash-prepped                       # ~40 GB sidecar + symlinks
env PLOW_MLA_PREFILL=full:128,512,2048,8192 PLOW_DECODE_BATCH_LADDER=1 PLOW_GLM_DSA=0 \
    PLOW_VERIFY_BIN=lean-plow/.lake/build/bin/plow_verify \
    target/release/plowc --hf-dir /home/lava/models/plow-glm53-flash-prepped --emit devblob \
    --gpu MI350X --arch gfx950 --num-gpus 8 --n-cu 256 --max-ctx 16384 --out <pkt>
env PLOW_HSACO_CONFIG=<pkt>/plow_config.h PLOW_DECODE_BATCH=1 JOBS=32 scripts/build_gfx950.sh <obj>
# GPU, through the queue
python3 scripts/bench/gpuq.py submit glm53f-smoke 8 <serve + smoke script>
```

The packet from config + index alone (no weights) is 70 MB: 4 prefill buckets + 1 decode program,
7 s to emit; gfx950 objects build in about 1 minute (interp prefill/decode/flash + `_gq` twins).

## 4. Status log

- 10:41 UTC: first packet emitted from the config/index stub; objects built (bf16 expert ops —
  the root-quantization bug above, found by reading `plow_config.h`).
- 11:09 UTC: emitter changes committed (00244f7e) and pushed. Emit from the stub with the fixed
  plowc hit the shared-expert gate-only GLU refusal (fixed by the split above); emit needs
  `PLOW_VERIFY_BIN` (the worktree has no lean-plow build; the main checkout's is used).
- 11:29 UTC: download complete (328 GB, 62 shards); `glm53_prep.py` wrote the 5.23 GB derived
  sidecar in under a minute; packet v2 from the prepped dir (8 s, `plow_config.h` identical to the
  stub packet's), objects 93 s, assets assembled; doctor: 1 FAIL (its vLLM client check wants
  `/app/.../vllm-python`; this box's client is the Docker wrapper) + 2 WARN (shared plowrt binary,
  no lean-qualification.json). Smoke queued (gpuq) behind the vLLM ISL-1024 grid.
- 11:35 UTC: vLLM TP8 server up after ~25 min of AITER JIT; first cell g1 (ISL 1024, c1) running
  at ~11 s per 128-token request.
- CPU roofline of packet v2 (`scripts/campaign/packet_roofline.py` / `op_roofline.py`, MI350X
  ceilings HBM 6200 GB/s, bf16 2300 / fp8 4600 TFLOPS, xGMI 1075 GB/s): see §6.
- 11:37 UTC: vLLM cell g1 (ISL 1024 / OSL 128, c1, TP8, greedy): TPOT 83.9 ms, TTFT 230 ms,
  11.8 tok/s. The server log shows the vendor path is in use (AITER 2-stage fused MoE per-1x128
  fp8 with the shared expert fused as expert 289, AITER mHC tilelang kernels, custom all-reduce,
  full CUDA graphs); `torch.compile` is refused for this model; no MTP (the comparison is
  single-token on both sides). This is the bar for c1, not the vendor's published 14 ms.
- Decode op-count experiments (CPU emits, decode program ops): base 1846; `PLOW_KDA_DECODE_FUSED[_ARM]`
  1778 (-68 = the 34 KDA chains fused); `PLOW_GLM_FUSE_B1+SEAM` 1835; `+GLM_FUSE_XRN` 1835 with 11
  XReduce folded; `PLOW_KDA_FB_FOLD` 1812; flat decode / gemm group / shard head: no change;
  `PLOW_GLM_ROUTED_W8A8` refused (needs `GLM_LINEAR_FP8`, and the MoeGluFp8Block128 kernel had no
  clamp) -> clamp added to the block128 GLU epilogue (`glu_limit` = f[1]) and the routed-W8A8
  packet (3 MoE ops/layer instead of 16) is being built.
- 11:40 UTC: vLLM ISL-1024 / OSL-128 grid complete (TP8, greedy, 2 repeats). The first c1 repeat
  (TPOT 83.9 ms) was AITER JIT warm-up for the M=1 shapes; the second is the number:

  | conc | TTFT mean ms | TPOT mean ms | out tok/s |
  |---:|---:|---:|---:|
  | 1 (r2) | 90 | 6.87 | 133 |
  | 8 | 209 | 9.2 | 741 |
  | 32 | 481 | 12.8 | 1974 |
  | 64 | 638 | 16.6 | 2978 |
  | 128 | 901 | 25.9 | 3900 |

  The grid is re-run with the JIT cache warm before it is used in a report (both repeats must
  agree for the strict renderer).
- 11:41 UTC: first plow smoke refused at load: the loader resolves this packet's prefill object
  as `interp_prefill_k3_moe_gq.elf` and its decode object as `interp_decode_k3(_gq).elf`
  (`exec/amd/object.rs` PrefillArm), names only `scripts/build_gfx942.sh` produced;
  `build_gfx950.sh` now builds them when the packet's `PLOW_PACKET_OBJECT_REQUIRES` carries
  `PLOW_K3=1` (same flags as its MLA+MoE prefill and decode rows, plus `-DPLOW_K3=1`).
- 11:49 UTC: second smoke refused at load: the flash object lacked the zero-rope V2 MLA prefill
  arm (`PLOW_MLA_PF2_NOPE_ARM=1` in the packet's requires; gfx942 spells it `PLOW_MLA_PF_NOPE=1`,
  gfx950 never set it). `build_gfx950.sh` now adds `-DPLOW_MLA_PF2_NOPE_ARM=1` to the flash
  objects when the packet requires it. Lesson: the gfx950 script keys only a few requirement
  flags off `PLOW_PACKET_OBJECT_REQUIRES`; each unmapped one costs a GPU round trip.
- 11:53 UTC: third smoke SERVES (TP8, load fast, requests answered). Reproduces the known bug
  exactly: first token right (" Paris"; logprob -0.65 vs -2.71 for the runner-up) and every
  later token garbage on all three prompts. HSA host logprobs validated: all 8 served top-1
  entries equal the device argmax tokens. Decode bisect under way (`PLOW_LAYERS` truncations,
  prefill-vs-decode consistency, opcode ablation).
- 12:12 UTC: first plow rung numbers (amd-bench, TP8, packet v2, dense attention, ladder 1):

  | rung | plow | vLLM (same box) | roofline floor |
  |---|---:|---:|---:|
  | prefill T=128 | 857 ms | - | ~1 ms |
  | prefill T=512 | 3.1 s | - | - |
  | prefill T=1024 / 2048 (bucket 2048) | 12.5 s / 12.6 s | TTFT 90 ms @1k | 10.8 ms @2048 |
  | prefill T=4096 / 8192 (bucket 8192) | 70.6 s / 71.0 s | - | 34.7 ms @8192 |
  | decode M=1, ctx 1k / 8k | 47.0 / 46.8 ms per token | TPOT 6.9 ms @c1 | 0.73 ms bytes; ~37 ms at 20 us/op |

  Decode is context-independent at ~47 ms: ~25 us per op over 1846 ops, the dispatch floor
  predicted in §6. Prefill is three orders of magnitude off its byte/compute floor and runs the
  whole bucket (1024 and 2048 cost the same): a per-op pathology, not bandwidth. Both need the
  per-op device trace before any kernel work. All 8 ranks agree on every token.
- 12:20 UTC: decode-vs-prefill consistency (amd-bench `--dump-logits`, prompt P = "The capital
  of France is"): prefill of P gives " Paris" (16.75), prefill of P+" Paris" gives "." (18.00),
  both sane; the decode row for " Paris" after P (the same computation on the T=1 program) has
  cos 0.31 against that prefill row and argmax 'es'. The mismatch is already present in a
  1-layer packet (`PLOW_LAYERS=1`: one KDA layer + dense MLP, cos 0.32), so it is in the KDA /
  dense / head T=1 path, not in MLA or MoE. `PLOW_STATE_CLEAR_DEVICE` makes no difference.
  The runtime does patch the KDA ops' row count to the real prompt length (`KDA_ROW_COUNT_OPS`),
  so pad rows are not the cause. Next: per-op activation dumps (`PLOW_DUMP_ACT`) on the 1-layer
  packet, row 5 of the 6-token prefill against row 0 of the decode step.
- 12:30 UTC: prefill attributed from the 8192-token device trace (`PLOW_TRACE_RAW`, read with
  `scripts/glm53_trace_attrib.py`): of 71.7 s, **65.7 s is `GemvF32`**, the mHC mix projection
  (`[T,16384] x [24,16384]^T`, 90 per pass), ~800 ms per packet. `d_gemv_f32` was written for the
  DSA indexer's 32-wide output at M=1: it walked only the N columns across waves and ran M
  serially, so 3 of 256 CUs did all the work at T=8192. Fixed to one wave per (row, column)
  output (M=1 mapping unchanged, so decode/indexer results are bit-identical). Second item is
  `XReduceTwoShot`: 4.5 s over 87 packets (52 ms per 64 MB), to be re-measured after the fix.
- 12:30 UTC: prefill after the `d_gemv_f32` fix (same sweep): T=128 113 ms, 512 183 ms,
  1024 431 ms, 2048 494 ms, 4096 1.64 s, 8192 2.05 s (was 71.0 s). Remaining 8192 trace: still
  `GemvF32` 747 ms (90 x 8.3 ms, latency-bound scalar K walk), `HyperConnPre` 327 ms (90 x 3.6 ms:
  the one-lane Sinkhorn through LDS costs ~110 us per ROW), `FlashMlaPrefill` 240 ms (11 x 22 ms),
  `HyperConnPost` 162 ms, `MoeGroupDownPf` 154 ms, `XReduceTwoShot` 58 ms (the earlier 4.5 s was
  peer skew behind the GEMV, not the fabric).
- 12:45 UTC: **decode bug found by per-op dumps.** `PLOW_DUMP_ACT` on the full packet, row 5 of a
  6-token prefill against row 0 of the decode step that computes the same token: every KDA layer
  tensor (pre-norm input, q/k/v raw and post-conv, gate, recurrent and conv states, gated-norm
  output) is bit-identical; every KDA layer's own pre-norm input matches through layer 2 and
  diverges at layer 4, i.e. after layer 3 — the first MLA **and** first sparse-MoE layer. On the
  4-layer packet (its shared MLA/MoE buffers hold layer 3 at the end) the MLA chain matches to
  1e-5 (q latent, absorbed q, the KV row prefill wrote and the row decode wrote, attention output,
  o_proj partial), the router logits and the shared expert match, and the MoE partial `dg_tp` is
  3.3x too large. Cause (`crates/devgen/src/mla.rs`, decode shared-expert down): under
  `raw_output` (the mHC packet) the combine reads the shared partial from `n.attn` and the fp8 and
  W8A8 shared arms write it there, but the **bf16 arm wrote `n.shared`** — GLM-5.3-Flash's shared
  expert is bf16, so the combine added the attention output and dropped the shared expert. One-line
  fix; packet v2d re-emitted (plow_config.h unchanged, objects reused).
- Decode trace (ctx 8k, 1846 packets, 45 ms): the serial mHC chain dominates —
  `GemvF32` ~130 us x 90 (one dependent scalar load pair per step, 256 steps), `HyperConnPre`
  ~113 us x 90 (one-lane LDS Sinkhorn), router `GemmF32` ~78 us x 42 (same scalar walk); the
  routed experts are ~70 us per layer in parallel. Kernel fixes queued as objects v2e: 8-wide
  vector loads with 4 steps in flight in `d_gemv_f32` and `d_gemv_bf16_f32`, and a
  register-resident n=4 Sinkhorn in `d_hyperconn_pre` (bit-identical arithmetic). Expected: the
  three chains fall from ~25 ms to ~2 ms of the 45 ms step; prefill `GemvF32`/`HyperConnPre`
  (1.07 s of the 2.05 s at 8192) by a similar factor.
- Parked: `GLM_LINEAR_FP8=1` (routed-W8A8 packet v3rw) refuses to load — it expects
  `self_attn.o_proj.weight_fp8` tensors the prep did not produce.
- 13:05 UTC: **decode correct** on packet v2d (commit 2b890c19). "The capital of France is" ->
  " Paris. ... , Paris is"; the 6-token prefill reproduces the decode continuation token for
  token; served smoke coherent, logprob top-1 = device argmax on 8/8 tokens.
- 13:15 UTC: rungs on objects v2e (vector-load GEMVs + register Sinkhorn), amd-bench TP8:

  | rung | v2 (12:12) | v2c | v2e | vLLM |
  |---|---:|---:|---:|---:|
  | prefill T=128 | 857 ms | 113 ms | 95 ms | - |
  | prefill T=1024 | 12.5 s | 431 ms | 252 ms | TTFT 90 ms |
  | prefill T=2048 | 12.6 s | 494 ms | 316 ms | - |
  | prefill T=4096 | 70.6 s | 1.64 s | 762 ms | TTFT 105 ms |
  | prefill T=8192 | 71.0 s | 2.05 s | 1.19 s | - |
  | decode M=1 | 47.0 ms | 46.6 ms | **31.0 ms** | TPOT 6.9 ms |

  Decode trace at 31 ms: 1846 packets at ~17 us each on the critical path, no single op above
  10% — the interpreter's per-packet floor (12 us measured on K3) times the packet count. The
  c1 TPOT gap to vLLM (4.5x) is a packet-count problem (W8A8 route 1384, KDA fused 1778, ...)
  and is not closable in this campaign; prefill/TTFT (2.8x at 1k, 7x at 4k) is where the
  remaining kernel work goes. 8192 prefill now: FlashMlaPrefill 238 ms (11 x 22 ms),
  HyperConnPost 162 ms, GemvF32 121 ms (now L2-bound: x re-read per column), HyperConnPre
  124 ms, MoeGroupDownPf 101 ms, KdaChunkCarry 76 ms, XReduceTwoShot 57 ms.
- Numerics note: v2e's reassociated GEMVs move the logits by cos 0.996-0.999 / max 1.1 on the
  same tokens (routing near-tie flips across 42 MoE layers); the 300-token README prompt's
  greedy continuation changed at a 0.25-logit tie. The FP32 gate (queued) adjudicates.
- 14:10 UTC: objects v2f (commit c6135217: HyperConnPre/Post/GemvF32 in the ragged-row table,
  16 B walks in both HyperConn kernels): prefill T=128 91 ms, 512 130, 1024 **173**, 2048 273,
  4096 **471**, 8192 994; decode M=1 **27.3 ms**. `PLOW_XR_SCHED=aiter` (v2g) is within noise.
  1024-token trace (175 ms): GemmSmall 23 ms (367 x 63 us), MoeGroupDownPf 26 + GluPf 19 +
  CombinePf 7 (the routed experts are ~2-3x their weight-read floor), XReduceTwoShot 14,
  GemvF32 16 (L2-bound x re-reads), HyperConnPre 12 (one-lane Sinkhorn per row), KDA chunk
  ops ~37 (carry runs on 64 CUs), router GemmF32 7.5. No single op is above 15%: reaching
  vLLM's 90 ms TTFT at 1k needs every one of these halved, which is not a 24 h job.
- Assessment at 14:20 UTC. plow c1: TPOT 27 ms vs 6.9 (dispatch-bound packet count), TTFT 173
  vs 90 ms at 1k, 471 vs 105 ms at 4k. Neither c1 cell is winnable this campaign. vLLM's own
  numbers fall steeply with context and concurrency (c128: 3900 tok/s at 1k, 1463 tok/s at
  4k with TPOT 69 ms) while plow's step cost is nearly context-independent, so the candidate
  cells are high concurrency at 4k+; the decode ladder (1,8,32,128; `PLOW_GEMV_WALK=1`
  objects) grid at 1024/4096 is running.
- 13:10 UTC: the ladder grid (1,8,32,128; walk objects) refused at load: "PLOW_DECODE_BATCH = 128
  with a recurrent-state tensor `kv.0.state` whose decode rung T=1 does NOT carry
  PLOW_KDA_F_SEQ_ROWS". `emit_glm53_program` passed `seq_rows = !prefill && rows > 1` to the KDA
  mixer, so the one-row rung addressed a one-slot state while `declare_glm53` sized it 128 slots
  wide (the K3 path already uses `ladder_on || t > 1`). Fixed in `mla.rs`; the T=1 rung now carries
  flags=3 and the parked-slot tensor. Under `seq_rows` the one-row rung loses the f_b fold and the
  fused-arm KDA step, so the ladder packet's c1 decode is a little slower than the single-rung
  packet; c1 is not a candidate cell. Ladder packet v2y re-emitted (only `PLOW_PACKET_HASH`
  changed in `plow_config.h`), objects rebuilt for the hash.
- 13:40 UTC, ladder grid at isl1024 (objects `PLOW_DECODE_BATCH=128 PLOW_GEMV_WALK=1`, GEMV row
  body at the 16-row cap): c1 TPOT 52 ms / TTFT 174 ms, c8 60 ms / 124 tok/s, c32 117 ms / 256,
  c64 250 ms / 235, c128 342 ms / 323 tok/s. vLLM at the same cells: 6.9 / 9.2 / 12.8 / 16.6 /
  25.9 ms, 3900 tok/s at c128. A 4-row GEMV body (`PLOW_GEMV_MM=4`, walk) changes nothing
  (c1 49, c8 56, c32 113 ms). c1 runs on the rung-8 program (`PLOW_DECODE_MIN_RUNG` = 8 on TP
  engines), hence 52 ms against the single-rung packet's 27.
- Batched amd-bench steps (1024-token prompts, every slot token-identical, ranks agree):
  B=8 50.8 ms, B=32 71 ms, B=128 162 ms. Per-op at B=128: GEMV 126 ms of the 162 (477 packets,
  ~265 us each; the row-bucketed GEMV body pays per row on the VALU and the walk re-reads the
  weights every 16 rows), everything else under 12 ms. The served c128 step (342 ms) is twice
  the bench step: new requests' prefills (173 ms each, 384 per cell) stall the decode batch.
  The 128-row PREFILL bucket runs the same projections as `GemmSmall` tiles in 91 ms for the
  whole forward, so the batched rungs were simply on the wrong GEMM family: the native
  batched-decode GEMM route (`emit_glm_decode_gemm_lt`) is hard-wired to GLM-5.2's shapes on
  gfx942, and `emit_k3_linear_rows` takes the GEMV arm for every sequence-row rung.
- Fix: `PLOW_GLM_DECODE_GEMM_ROWS=N` (devgen knob, unset = byte-identical packets). Decode
  rungs with at least N rows route their bf16 projections (KDA q/k/v/o/f_a/f_b/beta, MLA
  q/kv/o, shared expert, router, lm_head) through `pick_tile` (`GemmSmall` at these shapes;
  the decode object compiles it because the packet's decode inventory now lists it, and its
  LDS arena already covers the 64x128 tile). The fused `GemvQkv` arms yield to the split tiles.
  Packet v3 = ladder (1,8,32,128) with N=8: rungs 8/32/128 carry 477 tiles and no `Gemv`.
- 14:05 UTC, v3 batched amd-bench: B=8 44.8 ms (was 50.8), B=32 55.6 (71.3), B=128 90.6 (162);
  `GemmSmall` 28.5 ms where `Gemv` was 126. But every decode step sampled token 0: the
  interpreter's `case PLOW_DOP_GEMM_SMALL` lives in the `#if PLOW_BUCKET_PREFILL ||
  PLOW_MIXED_STEP` region, so a decode object dispatches nothing for it (the `default:` the
  K3 emitter warns about). Fixed in `interp.hip`: the decode region gets the case when the
  packet's decode inventory lists the op (`PLOW_DECODE_INVENTORY_PRUNE && PLOW_HAS_GEMM_SMALL`),
  so objects for packets without such a rung are byte-identical. Objects v3b.
- The FP32-gate captures teacher-force on `ref.json`, so both wait for the reference run; the
  vLLM capture queued ahead of it failed on the missing file. Order from here: v3b trace, v3b
  grids 1024/4096, FP32 reference (1 GPU, 2-3 h), vLLM capture, plow capture, score.
- 14:25 UTC, c1 decode (single-rung packet v2d/v2f, 26.4 ms wall, 1846 packets, span sum
  85 ms = 3.2x overlap): per-op tails — GEMV 342 x 23 us, per-expert `MoeExpertDownFp8Blk`
  336 x 38 us and `GluFp8Blk` 336 x 31 us (8 experts x 42 layers, 32 CUs each, concurrent
  within a layer), GemvF32 90 x 40, HyperConnPre 90 x 32 (ONE workgroup), router GemmF32
  42 x 42, GemvQkv 56 x 30, XReduce 90 x 17, RmsNorm 113 x 8. The layer's serial chain is ~16
  packets (hc_pre, norm, qkv, conv, step, norm, o, xreduce, hc_post, hc_pre, norm, router,
  topk, expert glu, expert down, combine, xreduce, hc_post) at 25-40 us each = ~590 us/layer
  x 45; vLLM's 6.9 ms is ~150 us/layer. CPU op-count sweep of every decode fusion knob
  (`PLOW_KDA_DECODE_FUSED[_ARM]`, `PLOW_KDA_FB_FOLD`, `PLOW_GLM_FUSE_B1/SEAM`, `GLM_FUSE_XRN`,
  `PLOW_GLM_MOE_FLAT_DECODE`, `GLM_SHARD_HEAD`): 1846 -> 1733 packets all together (-6%);
  B1/XRN/FLAT/SHARD_HEAD change nothing on this model. The c1 gap is the per-packet latency
  of a serial chain, not fusable by existing knobs; it needs a per-layer megakernel.
- 14:35 UTC, v3b (packet v3 + objects with the decode `GemmSmall` case) batched amd-bench:
  every slot streams the GEMV packet's tokens, ranks agree; B=8 48.9 ms, B=32 63.4, B=128
  **88.9 ms** (1440 tok/s aggregate; was 50.8 / 71.3 / 162). B=128 top spans: GemmSmall
  28.5, MoE down 9.6, RmsNorm 11.8 (one block per row, 54 us tails), KDA step 11.3, GemvF32
  9.2, XReduce 8.6, KDA conv 8.5, flash 8.1, merge 7.5. Serving grids on v3b queued.
- 15:10 UTC, served v3b grid at isl1024 (two passes agree within 1%): c1 TPOT 50.9 ms / TTFT
  176 ms / 19 tok/s, c8 58.9 / 128, c32 102 / 292, c64 178 / 333, c128 269 ms / **428 tok/s**
  (p50; means: c128 241 ms, 427 tok/s). Against the 16-row GEMV objects: c32 +14%, c64 +42%,
  c128 +32%. The served c128 step (241 ms) is 2.7x the bench step (89 ms) because each
  admitted request's whole 173 ms prefill stalls the batch; 384 prefills per cell. vLLM at the
  same cell: 25.9 ms, 3900 tok/s.
- 14:50 UTC, served v3b grid at isl4096 (two passes within 2%): c1 TPOT 50.9 ms / TTFT 456 ms,
  c8 71.7 / 101 tok/s, c32 168 / 179, c64 317 / 190, c128 543 ms / 222 tok/s (p50). vLLM:
  7.0 / 11.0 / 20.9 / 36.2 / 69.3 ms and 130 / 556 / 1089 / 1322 / 1463 tok/s. At c128 the
  cell is 384 x 471 ms of single-request prefill = 181 s of the ~220 s run; the decode batch
  is idle while a prefill runs. FP32 reference started 14:50 (1 GPU).
- Runtime parity on gfx950 (task "NVIDIA-side advancements"): working here — HSA global-queue
  interpreter (`PLOW_GLOBAL_QUEUE`, on), decode ladder rungs with per-slot state, segmented
  prefill with the ragged-row table, TP8 two-shot XReduce, greedy top-k logprobs (added this
  campaign), `PLOW_TRACE_RAW` / `PLOW_DUMP_ACT` diagnostics. Fail closed or absent on this
  packet — unified token batch / mixed step (prefill rows packed into the decode step; the one
  mechanism that would remove the prefill stalls above), shared prefix cache, the hipBLASLt /
  AITER native GEMM routes (gfx942 GLM-5.2 shapes only), decode lookahead pipelining
  (`PLOW_DECODE_PIPELINE`, CUDA only), sampling (argmax only; every cell runs greedy).
- 15:26 UTC, FP32 reference done in 37 min on one MI350X (offload): 46 cases (20 natural, 8
  chat, 18 needle up to 15,872 tokens), 1030 scored positions, 18/18 needles correct,
  teacher-forcing mismatch 0; `ref.json` sha256 45c79250. vLLM and plow (v3b) captures queued.
- 15:45 UTC, FP32 gate on the served v3b packet (capture at concurrency 16): **FAIL** — KL
  mean 4.75 vs vLLM 0.037, top-1 agreement at decisive positions 14% vs 99%, continuation
  agreement 12% vs 73%, needles 0/18 vs 18/18. The free runs show the first token right and
  garbage after it ("breakfast set Interr setam...", "77-, ,,"), i.e. the batched decode rungs
  are wrong on real prompts. The batched amd-bench check that "validated" v2y/v3/v3b used a
  synthetic prompt (ids 100..1123) whose continuation degenerates to one repeated token, so
  it could not see this. Diagnostic queued: the same 1024-token Pride-and-Prejudice prompt
  through the single-rung packet (B=1), the GEMV ladder (B=8/32) and the GEMM ladder (B=8/32),
  16 steps each, against the FP32 continuation.
- 16:00-16:20 UTC, batched-rung bug found and fixed. Dumps on the real prompt: the ladder's
  rung-1 program is exact, every rung wider than one row is wrong from the first decode step,
  and the per-layer dump put the first divergence at layer 3's block input, i.e. layer 3 — the
  first *routed-MoE* layer (layers 0-2 are dense FFN). A 4-layer ladder packet showed the MLA
  attention chain matching at rung 8 (q path, latent cache, flash partials, merge, o_proj,
  post-attention norm), the shared-expert output not. Cause: `emit_glm_moe_ffn_rows` (the
  rows > 1 decode FFN, which takes the grouped prefill MoE family) wrote the bf16 shared
  expert's down-projection to `act.shared` while its combine reads the shared partial from
  `act.attn` under the hyper-connection — the same destination mismatch fixed for the one-row
  block in 2b890c19, in the other function. Fixed in mla.rs (both the GEMM-rows and the
  hipBLASLt-segment arms). Packet v4 (objects unchanged, v3b): real-prompt amd-bench streams
  equal the FP32 continuation for 16 steps in every slot at B=8, 32 and 128; steps 51.0 /
  59.8 / 91.2 ms (1403 tok/s aggregate at B=128). Lesson recorded: a correctness gate must
  use a real prompt and must cover a routed-MoE layer; the earlier synthetic-prompt check and
  the 4-layer/first-3-layers dumps could not see this layer class at all.
- 16:20 UTC, served v4 grids (1024/4096), the plow FP32 capture and the gate are queued;
  decode tier objects (`lowrung1/`, `lowrung8/`, narrower GEMV bodies for rungs 1 and 8 — the
  loader logged "no decode tiers", so every rung ran the 128-wide object: c1 51 ms vs 28 ms on
  the single-rung packet) are building for a follow-up grid.
- 16:50 UTC, E8 mHC sequence-parallel prefill seam (mla.rs, pending validation). On this
  model `PLOW_GLM_SEQ_PAR` was inert: `glm_sp` was consulted only under `!raw_output`, and the
  hyper-connection program makes every seam `raw_output` (the v4sp packet differed from v4 by
  one `XAllGather` per 2048/8192 bucket). Now, for prefill buckets with t >= 2048 and the knob
  on, `Glm53Band` declares `@band<t>` views of the four residual streams, the mixes, the layer
  input, `x`, `og_tp` and `dg_tp`; every hyper-connection op (expand/contract HyperConnPost,
  mix GemvF32, HyperConnPre, both HyperConnPost per layer) runs on this rank's t/tp rows; the
  block inputs are all-gathered (MLA layers through the prefill's existing band input-norm arm,
  KDA layers from a HyperConnPre that writes peer slot 3, the FFN through `glm_sp_norm_gather`);
  the o_proj / KDA / MoE / dense partials are reduce-scattered in place (new `raw_output` arms
  at the three seams) and the HyperConnPost reads the band. Same collective bytes as the
  all-reduce; the per-op trace at 4k put ~130 of 471 ms in these row-local ops, so the expected
  TTFT gain is ~24% at 4k and 8k (the 128/512 buckets are unchanged). `cargo test -p devgen
  --lib mla` 208 pass. Packet v4sp2 (`PLOW_GLM_SEQ_PAR=1 PLOW_GLM_SEQ_PAR_PROJ=0`, GEMMs
  byte-identical), objects = v4sp (`PLOW_HAS_XALLGATHER=1`). Validation queued behind the FP32
  capture: real-prompt streams at the 2048 bucket (ragged), v4 vs v4sp2 at a 4096-token prompt
  (8192 bucket), prefill sweep 1k/2k/4k/8k.
- 17:00 UTC, v4 FP32 gate scored (capture at concurrency 16, 46 cases / 88 requests): FAIL,
  but now kl_mean 0.43 (was 4.75), needles 18/18 (was 0/18), cont_frac 0.754 (vLLM 0.730).
  Per-position KL: 5.3 at position 0 (the first token after prefill), 0.0-0.26 after; the 35
  "decode" positions with KL > 1 all sit where teacher-forcing re-issued a request, i.e. they
  are prefill first tokens too. At those positions the sampled token is right and the reported
  top-k is another distribution (token ids 0, 3, 9, 13 at the top). Cause: `AmdServe::logits_row`
  read row `slot` of `act.logits`, but a per-slot prefill tail writes ONE logit row at row 0
  (`i[0] = 1`, `a_row0 = clen - 1`); only decode rows are slot-indexed. Fix (serve/engine.rs):
  a per-slot `lp_row` as the CPU engine keeps — 0 after a prefill, the slot after a decode
  dispatch, unknown (logprobs refused) for token-batch body rows. Re-capture queued
  (`fp32cap-plow-v4b`, same v4 packet, tiered objects). The model math was never the problem:
  the decode positions are already at KL ~0.2 vs vLLM's 0.03, which is the next thing to look at
  if the gate still fails on kl_mean after the fix.
- 17:00 UTC, tiered objects validated (`validate-v4t`): streams exact at B=8/32/128; steps
  B=8 49.3 ms (128-wide object 51.0), B=32 58.5 (59.8), B=128 97.5 (91.2, noise or the tier
  pick). Rung 8 stays ~50 ms because the ladder emits its GEMVs as `GemmSmall` from
  `PLOW_GLM_DECODE_GEMM_ROWS=8`; c1 at 27 ms needs rung 1 (`PLOW_DECODE_MIN_RUNG=1`, in the
  running v4t grid). The first E8 validation refused to load: the sp2 object dir carried the v4
  tier links, which stamp the v4 packet hash; tiers are being rebuilt from the sp2 config and
  the validation re-queued without them.
- 17:05 UTC, served grid v4t (tiers + `PLOW_DECODE_MIN_RUNG=1` + `PLOW_PF_INTERLEAVE=512` +
  `PLOW_PF_CHUNK=512`), isl 1024 rep 1: c1 TPOT 27.2 ms (v4: 51.0; the rung-1 program on the
  rung-1 tier) / TTFT 267 (176); c8 55.2 (59.5) / 1704; c32 82.0 (105) / 7057; c64 147 (180) /
  14900, out tok/s 242 (326). Rung-1 admission is the c1 win; the 512-row chunk cap is a loss:
  every prompt runs as 512-bucket chunks with a decode step between them, so TTFT and prefill
  throughput collapse at c32+. Next grid v4t2: tiers + MIN_RUNG=1 + `PLOW_PF_INTERLEAVE=4096`,
  no chunk cap (bucket-shaped prefill, a decode step after at most 4096 prefill rows).
- 17:05 UTC, decode numerics are not the gate's problem: at the 868 pure decode positions of
  the v4 capture (every scored position that is not a request's first token) plow's KL against
  the FP32 reference is mean 0.013 / p50 0.0003 / p99 0.17 / max 0.70, against vLLM's 0.036 /
  0.0008 / 0.56 / 1.84 on the same positions; by prompt length plow is tighter at every length.
  The 162 request-first positions carry mean 2.65 (vLLM 0.04) — the logits-row bug above.
- 17:15 UTC, E8 validated (`validate-sp2b`, packet v4sp2 + its own tiers): real prompt
  nat-pride-1024 (2048 bucket, ragged) scalar and B=8 streams equal the FP32 continuation; a
  4096-token prompt (8192 bucket, ragged) gives the same 16-token stream as v4. amd-bench
  prefill sweep, median of 3 (v4 → v4sp2): T=1024 173 → 151 ms, 2048 256 → 212, 4096 471 → 381,
  8192 995 → 806 (−13/−17/−19/−19%). Decode unchanged (same programs below T=2048).
- 17:20 UTC, E1b (`validate-g32`, packet v4g32 = v4 with `PLOW_GLM_DECODE_GEMM_ROWS=32`, rung-8
  tier at `PLOW_GEMV_MM=8`): the rung-8 program carries 444 Gemv + 22 GemvQkv instead of 477
  GemmSmall; streams exact at rungs 1/8/32/128; steps rung 1 26.6 ms, B=8 46.2 (49.3 with the
  GEMM route), B=32 58.1, B=128 90.9. Taken into the final packet. The build_gfx950.sh
  `PLOW_DECODE_TIERS` loop reproduces the hand-built rung-1 tier byte for byte; the hand-built
  rung-8 tier (MM=4 + walk) was byte-identical to the wide object, i.e. the decode object depends
  on MM/WALK and not on PLOW_DECODE_BATCH — which is why that tier gained nothing at rung 8.
- 17:20 UTC, decode op-count knob scan (emit only): FUSE_ROPE, FUSE_B1, GLM_FUSE_XRN,
  DECODE_GEMM_GROUP, KDA_FB_FOLD, KDA_DECODE_FUSED_ARM, XR_COMBINE_FOLD, DECODE_GLUE_CUS,
  K3_KDA_CONV_STEP_DB, FUSE_SEAM_RN, XR_RES, MOE_CORESIDENT=2, MOE_SHARED_SEED leave the T=1
  program at 1846 ops; FUSE_SEAM removes 11; FUSE_QNORM aborts the emit ("no structurally
  eligible KDA graph"). The c1 dispatch floor on this model is structural (per-expert MoE ops,
  34 KDA layers' chains), not a knob.
- 17:20 UTC, final candidate packet v5 = v4 + mHC seam + GEMM rows 32, objects with the tier
  loop; validation, FP32 capture and the served grids (tiers, `PLOW_DECODE_MIN_RUNG=1`,
  `PLOW_PF_INTERLEAVE=4096`, no chunk cap) queued.
- 17:25 UTC, FP32-reference gate PASS on the v4 packet (sha c30dc0b7…, capture with the fixed
  logits row, concurrency 16, 46 cases / 88 requests): kl_mean 0.0134 (vLLM 0.0368), kl_p99
  0.184 (0.549), kl_max 0.70 (1.84), top1_decisive 0.9967 (0.9868), cont_frac 0.754 (0.730),
  needles 18/18 (18/18). Gate dir /home/lava/bench/glm53flash/fp32ref/gate-v4b.
- 17:35 UTC, E6 prefix cache, first reading (WRONG, corrected at 17:50): the load-time warning
  "prefix cache requested but not built; serving with slot-local KV and a permanent 0% hit rate.
  Layout::from_blob refused this packet" was read as the cache being unavailable.
- 17:50 UTC, E6 prefix cache, corrected: the warning describes only the gfx942 shared pool
  (`exec/amd.rs`: `shared_layout` is attempted only when `arch == "gfx942"`); the same log has
  "AMD prefix cache selection requested=true selected=true" and the probe hit: same 1024-token
  prompt TTFT 453 -> 273 ms, prompt + 0 tokens 290 ms, one token changed at position 512 465 ms
  (the snapshot was armed at row 992, above the 512-row common prefix, so a miss that re-arms
  at 512); /metrics prefix_cache_queries_total 4096, hits 1984 (2 x 992). The route is the
  per-slot snapshot (`exec/amd/prefix.rs::snapshot_regions`): this packet's `kv.N.state` and
  `kv.N.conv_state.{q,k,v}` (136 tensors, 18 MiB/slot/rank) are the carried regions; the MLA
  `kv.N.ckv` rows are full-context (`HeadNormRope` kv_mask u32::MAX, stride = max_ctx) so they
  need no copy. plowrt now says so at load ("shared prefix pool is gfx942-only ...; the prefix
  cache serves per-slot snapshots" + "prefix snapshot route: regions=136 per_slot_mib=18") and
  the "0% hit rate" warning is reserved for gfx942 refusals. The recipe serves with the cache
  ON (the runtime default); the grid/capture/validate scripts default it on; the prefix-on
  gate capture of v5 (`fp32cap-plow-v5-pfx`) is queued — teacher-forced re-requests share
  prefixes, so it exercises the restore path on every case.
- 17:50 UTC, FP32-reference gate PASS on the v5 packet (sha e5007402…, prefix cache off, c16):
  kl_mean 0.0139 (vLLM 0.0368), kl_p99 0.205 (0.549), kl_max 0.59 (1.84), top1_decisive 0.9978
  (0.9868), cont_frac 0.770 (0.730), needles 18/18. Gate dir fp32ref/gate-v5. The v5 knob set
  (ladder 1/8/32/128, GEMM rows 32, mHC seam, tiers) is the final configuration.
- 17:55 UTC, final packet v6 = v5 knobs at `--max-ctx 65536` (user: one production recipe,
  `recipes/infervisor/glm-5.3-flash/gfx950-mi350x-tp8-fp8.toml`, context-specific choices
  inside it; probe long context because the KDA recurrence only shows at length). Emit 11 s
  (sha 7649e082…), objects + tiers building. Queued after the vLLM 64k baseline and the
  DeepSeek grids: validate-v6 (streams at every rung, 4k, 16k, 32k and 60k synthetic prompts
  through the 8192-bucket chunk loop, prefill sweep to 32k), needle-plow-v6 and needle-vllm-v6
  (`scripts/glm53_needle_probe.py`, exact lengths 4k/16k/32k/60k at depths 0.1/0.5/0.9, paired
  verdict), fp32cap-plow-v6 (prefix on), vLLM isl 61440 c1/c8, v6 grids 1024/4096 (c1..c128)
  and 16384/32768/61440 (c1/c8). The long cell is 61440 not 65536: the grid's prompt + 128
  output tokens must fit the packet's 65536 context (vLLM's max-model-len is isl + 2176).
- 18:05 UTC, FP32-reference gate on v5 with the prefix cache ON (`fp32cap-plow-v5-pfx`, c16):
  PASS, kl_mean 0.01415 (off: 0.01388), kl_p99 0.235 (0.205), top1_decisive 0.9956 (0.9978),
  cont_frac 0.748 (0.770), needles 18/18 — within run-to-run noise of the prefix-off capture.
  Caveat: the capture took 80 s against 75 s off, so its teacher-forced request pairs did not
  hit (at c16 the pair's second request lands on another slot; the snapshot is per slot). The
  hit path's correctness is checked directly by `prefix-check-v6` (queued before the grids):
  every natural/chat gate prompt <= 8k cold, then again (hit), then prompt + 8 output tokens
  (hit past the snapshot), streams compared token-for-token; then c8 re-sends. vLLM long
  baselines (TP8, c1/c8): isl 32768 TTFT 706/3392 ms, TPOT 6.95/28.7 ms, 78/142 tok/s; isl
  65536 TTFT 1439/5619 ms, TPOT 6.81/50.4 ms, 53/82 tok/s.
- 18:17 UTC, validate-v6 (packet sha 7649e082…, max-ctx 65536, prefix cache on): streams exact
  (= FP32 continuation) at rung 1 (27.8 ms), B=8 (47.2), B=32 (59.2), B=128 (98.8 ms, 1295
  tok/s; v5 at 16k: 90.7 — the 64k packet costs +9% on the widest rung). Long synthetic prompts
  (`seq` ids, no reference) run through the chunk loop without error: 16k/32k/60k prefill then
  one-row decode at 28.1/28.4/29.2 ms — decode is flat in context. Prefill sweep 1k/2k/4k/8k/
  16k/32k = 150/212/382/802/2081/6309 ms: quadratic above 8k (FlashMlaPrefill at ~1% of MFMA
  peak, §6a), so plow's 32k TTFT is ~9x vLLM's 706 ms. Long-context correctness is judged by
  the needle pair (queued), not by these garbage-in streams.
- 18:35 UTC, DeepSeek-V4.1-Flash vLLM TP4 baseline isl 1024 (repeat 1; out tok/s, TTFT p50,
  TPOT p50): c1 162 / 84 ms / 4.27 ms, c8 132 / 1600 ms / 5.37 ms (TTFT outlier: first cell
  after JIT warmup), c32 1160 / 875 / 11.3, c64 2237 / 1174 / 19.7, c128 2990 / 1202 / 33.2.
  Load took 24 min (510 GB). Results /home/lava/bench/dsv41flash/vllm-tp4/isl1024.
- 18:35 UTC, needle-plow-v6 first run failed before any request: the probe needs
  `transformers` for the chat frame and the default shell has none; the needle scripts now run
  it under `nix develop .#quantize` (transformers 5.5.4). Re-queued after prefix-check-v6.
- 18:30 UTC, FP32-reference gate on v6 (65536 packet, prefix cache ON, c16): PASS, kl_mean
  0.01401 (vLLM 0.03677), kl_p99 0.225 (0.549), kl_max 0.52 (1.84), top1_decisive 0.9956
  (0.9868), cont_frac 0.743 (0.730), needles 18/18. Gate dir fp32ref/gate-v6.
- 18:30 UTC, needle-vllm-v6 first run: 0/36 — every answer is reasoning text ("The user has
  pasted a huge amount of repeated text..."): GLM-5.3-Flash's template opens a reasoning block
  even with thinking off, and 16 tokens never reach the answer. The FP32 gate's needles append
  `</think>` (`--gen-suffix`); `scripts/glm53_needle_probe.py` now takes the same `--gen-suffix`
  and both arms re-run with it.
- 18:55 UTC, long-context needles with `</think>` (exact lengths 4096/16384/32768/61440, depths
  0.1/0.5/0.9, three needles, served v6 with prefix cache on): plow 36/36, vLLM 36/36, paired
  verdict PASS (no regressions). The KDA recurrence carries correctly through the chunk loop to
  60k tokens.
- 18:55 UTC, prefix-check-v6 (v6, prefix cache on, gate natural/chat prompts): every same-prompt
  hit stream equals its cold stream (22/22); cached tokens 992/4064/8160 of 1024/4096/8192;
  hit TTFT ~0.71 s flat vs cold 0.87/1.10/1.51 s. The "prompt + 8 output tokens" re-requests
  differ from the cold continuation in 9/22, including `nat-code-128` with 0 cached tokens: a
  prefill of 8 tokens is not bit-equal to 8 one-row decode steps (KDA chunk vs step), not a
  cache effect. The concurrent phase (c8, each prompt twice) had 0 cached tokens: per-slot
  snapshots only serve a prompt seated on the same slot, admission seats on the lowest free
  slot, so a shared prefix at concurrency never meets its snapshot; its stream differences
  vs the sequential run are rung-8 vs rung-1 numerics (also at 0 cached).
- 19:00 UTC, cross-slot prefix reuse (plowrt, uncommitted until checked): `prefix::clone_spans`
  derives each full-context KV cache's per-slot `[heads][ctx][row]` layout from its
  `HeadNormRope` writer (refuses rings, pooled indexers, unknown kv tensors);
  `AmdEngine::clone_prefix(src, dst)` copies the donor's carried-state snapshot into the
  destination's snapshot buffer and the first `snap` rows of every full-context cache, never
  touching the destination's live state; `AmdServe::adopt_prefix` picks the slot with the
  longest armed prefix of the new prompt before planning, so the ordinary resume restores it.
  Seats do not move (decode rung unchanged). Check `prefix-check2-v6`: 8 prompts at c8, then
  reversed (each on another slot), then + 8 tokens, cache off vs on, streams compared.
- 19:00 UTC, kernel A/B (subagent, wide-rung RmsNorm wave-per-row + KdaStateStepG per (row,
  head)), B=128 on nat-pride-1024, streams exact (128/128) in all four runs: new 103.5 / traced
  86.1 ms vs base 91.9 / 92.4 ms. Inconsistent between the two new runs; analysis pending.
- 19:15 UTC, prefix-check2-v6 (cross-slot reuse, c8, cache off vs on on one lease): the
  cross-slot hits are exact — with the cache on, every phase-B stream (the same 8 prompts
  re-sent in reversed order, each on another slot) equals its phase-A cold stream, 8/8, and the
  four 4096-token prompts each took 4064 cached tokens from another slot (B wall 6.0-6.2 s vs
  7.1-7.4 s off). Phase C (prompt + 8 tokens) hit 8160 rows on three 8192 prompts (3.3-3.9 s
  vs 5.8-8.9 s off). The 8192 prompts missed in B: one snapshot per slot, and the 4096 prompts'
  hits replaced the 8192 donors' snapshots on the slots they were seated on — a capacity limit,
  not a correctness one. Three off-vs-on stream differences are near-tie numerics: one is in
  phase A with 0 cached tokens (`nat-docs-4096` token 9, the same 95496/564 tie seen at c8 in
  prefix-check-v6), the others diverge at tokens 9 and 14 and their on-arm hit streams equal
  the on-arm cold streams. The script's strict off-vs-on rule cancelled the grids; re-queued.
- 19:40 UTC, wide-rung decode kernels (subagent; commits 0b6d807c + 81490ae3, `op_norm.h`
  wave-per-row RmsNorm when rows > blocks, `op_kda.h` one KdaStateStepG item per (row, head)
  with the column loop inside; decode objects only, bit-identical arithmetic). Trace, B=128
  rank 0: KdaStateStepG span 11.2 -> 7.3 ms (body 200 -> 86 us/packet; floor ~21 us), RmsNorm
  span 11.7 -> 10.3 ms (the 22 one-block packets 178 -> 118 us; more blocks needs the emitter).
  Objects v6k (v6 packet, tiers 1/8) validate-v6k: every stream line identical to v6's (r1, B=8,
  B=32, B=128, 4k/16k/32k/60k prompts), step r1 27.9, B=8 46.0 (v6 47.2), B=32 56.6 (59.2),
  B=128 84.3 ms (98.8; 1519 tok/s). Same numerics, so the v6 FP32 gate and needles carry
  over. The v6 grids now run on v6k objects; the recipe's objects are this source.
- 20:50 UTC, final packet v7 = v6 + `PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=2` (subagent finding: the
  v6 emit had no PLOW_MLA_PF_V2, so programs were one uniform segment and every NoPE
  FlashMlaPrefill ran the scalar decode body on the 8-wave interpreter; V2 isolates it on the
  4-wave V2 MFMA flash arm). Packet sha 451b1b07…, objects from the 81490ae3 sources with tiers.
  FlashMlaPrefill at 4k: 11 x 5.55 ms -> 12.4 ms total. validate-v7: streams exact at rungs
  1/8/32/128; prefill sweep 1k/2k/4k/8k/16k/32k = 164/217/335/599/1273/2752 ms (v6 32k 6309).
  The 128-row decode program is byte-identical to v6k's (disasm --stream diff empty); its step
  measured 113.6 ms in this run against 84.3 in validate-v6k — the same bimodal B=128 step
  seen in the kernel A/Bs (85 vs 98-103 ms runs in both arms), not a v7 change; unexplained.
  FP32-reference gate v7 (prefix on): PASS, kl_mean 0.01396 (vLLM 0.03677), kl_p99 0.233
  (0.549), top1_decisive 0.9967 (0.9868), cont_frac 0.756 (0.730), needles 18/18. Long-context
  needles v7: 36/36 at 4k-61k. The v7 grids replace the v6 grids (stopped).
- 23:00 UTC, v7 grids and strict reports: every cell MATCHED/EQUIVALENT; plow total throughput
  0.07-0.26x vLLM in all 19 cells (results summary docs/bringup/results/glm53flash-mi350x-
  2026-10-12). Recipe promoted to `validated` on v7 (8ed1b8f5).
- 00:00 UTC (10-13), prefill kernel subagents: (a) prefill mHC mix GEMV + HyperConnPre lane-
  parallel arms (62e06055; GemvF32 9.6 -> 4.9 ms, HyperConnPre 7.7 -> 3.3 ms at 4k; the §6a
  72/37 ms figures were pre-seq-par and stale), and the HyperConnPre arm in decode objects too
  (bc44e45c; r1 27.8 -> 26.2, B=8 47.3 -> 44.4, B=32 57.8 -> 55.0 ms), activation dumps
  bit-identical; (b) GLM prefill routed to the existing lean KDA chunk objects + MoE prefill
  axes (tile binsearch, down-epilogue hoist, raw-fp8 B prefetch, vector combine) (819034ad):
  prefill 1k/4k/8k/16k 146/332/595/1254 -> 114/227/376/793 ms, amd-bench streams identical.
- 00:10 UTC, v8 (= v7 + both) FAILED serving: FP32 gate capture had 15/18 needle cases return
  null logprobs and the needle probe found 4/36. Diagnostic: v8 with PLOW_PREFIX_CACHE=0 18/18,
  with the cache on 4/18 (6624 cached tokens); v7 with the cache on 36/36. A prefill resumed
  from a prefix snapshot (chunk starting at row > 0 with restored KDA state) breaks on the lean
  KDA path; amd-bench never resumes, so its stream checks could not see it. v8 grids cancelled;
  the recipe stays on v7. Fix delegated (route or repair resumed chunks), then re-gate.
- 00:30 UTC, cause and fix (84c775e9): the two KDA intra routes kept the bucket's row count, so
  a chunk whose length is not a multiple of 64 (a resumed prefix or a split prefill) read stale
  g_prefix rows past its end; intra routes now take the chunk's real row count, interpreter below
  512 rows. Needles cache-on 18/18, PF_CHUNK=512 split 9/9, streams equal v7.
- 02:40 UTC, v8b (v8 packet 980625b3 + bc44e45c objects + 84c775e9 plowrt) passes every gate:
  validate rungs r1/B8/B32/B128 26.1/44.5/56.0/84.1 ms, all ranks token-identical; FP32 gate
  PASS kl_mean 0.01424 (vLLM 0.03677), kl_p99 0.233, top1_decisive 0.9956, needles 18/18;
  long needles 36/36; cross-slot prefix reuse at c8: cached streams equal their cold runs
  4/4 (off-vs-on differences also occur at 0 cached tokens: batch numerics). Strict reports,
  all 19 cells MATCHED/EQUIVALENT: plow total throughput 0.08-0.31x vLLM (v7 0.07-0.26x).
  Recipe moved to v8 (plow_git 84c775e9).
- 03:00-04:15 UTC (10-13), serving scheduler (branch serve-packed-prefill-amd, a58ab5a4 +
  1cdc536f). Tick log at isl1024 c128 on v8b: every cold prompt ran as a 992-row head (113 ms +
  13 ms snapshot) plus a 32-row tail on the 128 bucket (62 ms) — the prefix cache's arm split —
  and the planner advances a slot once per tick and admits a fresh prompt only into an untouched
  budget, so a prompt took two ticks and the c128 decode batch settled near 64 of 128 slots.
  True packing (several prompts in one bucket) is not reachable on this packet: `glm53_emit_full`
  emits no packed-prefill siblings, the packed MLA flash refuses NoPE, and the lean KDA chunk
  routes are single-sequence. Three runtime levers instead: `PLOW_PF_BACK_TO_BACK=1` (a tick keeps
  launching chunks until its `PLOW_PF_INTERLEAVE` budget is spent), `PLOW_PREFIX_ARM_END=1` (a
  cold prompt snapshots after its last chunk; continuations resume at the full prompt, identical
  re-sends run cold), TP snapshot/restore on all ranks at once (13 -> ~2 ms). Served, 2 reps,
  total tok/s / TTFT p99 / TPOT p99, v8 -> new: isl1024 c32 2427 / 9.3 s / 108 ms -> 3106 /
  4.1 s / 91 ms, c128 2861 / 45.4 s / 191 ms -> 5199 / 17.1 s / 213 ms; isl4096 c32 6303 /
  16.5 s / 152 ms -> 7661 / 10.5 s / 132 ms, c128 6330 / 78.7 s / 217 ms -> 10218 / 46.7 s /
  384 ms; c1 TTFT p50 1k 205 -> 122 ms mean. Gates on the final binary: c1 greedy streams equal
  v8 with the cache off 22/22 (cold prompts are now the unsplit plan; back-to-back alone equalled
  v8 cache-on 22/22), needles 36/36 cache on and 36/36 with `PLOW_PF_CHUNK=512`, prefix-check2
  phase C hits (8192 cached) equal the cache-off streams, FP32 gate PASS kl_mean 0.0143, kl_p99
  0.240, top1_decisive 0.9967, needles 18/18. Phase B (identical re-sends on other slots) no
  longer hits: the cost of the end arm. Recipe [serve] takes both knobs.
- 04:45 UTC (10-13), end-armed re-sends hit again (branch serve-prefix-end-resend). An end-arm
  snapshot now also keeps the sampled first token and rank 0's `act.logits` row (host copy, only
  when the packet keeps one); an identical prompt (on its slot or adopted cross-slot) restores the
  snapshot at n, runs no chunk and goes straight to decode with that token; logprobs read the kept
  row. Resumed rows no longer count against the back-to-back tick budget. No new knob. v8b, recipe
  env: c1 re-sends cached = prompt for all 8 prompts >= 128 rows (339..8192), streams and
  logprobs equal to the cold run 11/11 and to the cache-off run 22/22; re-send wall 1.0-1.15 s ->
  0.78-0.80 s for 32 tokens; prefix-check2 phase B 4/8 hit with 4096 cached (the other four lose
  their donor slots to those adoptions: the per-slot design), B equal A 8/8; the only off-vs-on
  differences (one per phase) sit at exact top-2 ties (gap 0.0) that the cache-off control also
  flips between A and B; long needles 36/36.
- 06:20 UTC, v9 = v8 + back-to-back prefill / end prefix arm / identical-resend hits (a58ab5a4,
  1cdc536f, 21d004f6) + decode mHC folds and inlined plow_exec (packet 1533860b). Gates: validate
  r1/B8/B32/B128 12.5/31.7/52.4/84.7 ms, ranks token-identical; FP32 PASS kl_mean 0.01384,
  top1_decisive 0.9989; needles 36/36, split (PF_CHUNK=512) 18/18; prefix_check2 re-sends cached
  4096 = cold 4/4. Strict reports 19 cells MATCHED/EQUIVALENT: 0.14-0.56x vLLM (v8 0.08-0.31x).
  PLOW_PF_INTERLEAVE=2048 screen: -4..-18% throughput, kept 4096. Recipe moved to v9.
- 10:40 UTC, v10 = v9 + decode bodies, collectives/attention defaults, prefill kernels, packed
  multi-prompt prefill with decode rows riding (mixed prefill+decode, default on), DP2xTP4 serve,
  Codex review fixes (packet e4786a2c, plowrt 927f76eb). Gates: validate r1/B8/B32/B128
  12.4/22.9/30.2/40.3 ms ranks agree; FP32 PASS kl_mean 0.01342 (DP2xTP4 0.01295); needles 36/36,
  split 18/18; prefix re-sends cached 4096 == cold 4/4. Strict TP8 reports 19 cells
  MATCHED/EQUIVALENT: 0.25-0.62x vLLM (v9 0.14-0.56x). Recipe moved to v10.
- 13:45 UTC, v11 (one-shot integration of the v11 branches; plowrt ac229648, packet 9328c621):
  scheduler (tick trace, PLOW_PF_INTERLEAVE=8192 removes half-empty packed launches: +26% isl1024
  c128), rung-1 hop fusions + sharded lm_head, bit-identical V2 flash softmax and KDA carry
  schedule, Codex review fixes, AMD co-resident multi-model serving. Dropped: PLOW_GLM_XR_HCPOST +
  PLOW_XR_DEC_CUS (serve fails "cross-GPU gate 0 reads 16, expected 8", with or without
  PLOW_XR_TAG_SLOT; amd-bench passes) and W8A8 MoE prefill (1.35x on the expert GEMMs but ~25x the
  FP32 error, relL2 4% vs 0.16%). Gates: validate r1/B8/B32/B128 12.7/22.7/30.4/40.1 ms, FP32 PASS
  kl_mean 0.01475, needles 36/36, split 18/18, prefix re-sends cached == cold. Strict TP8 reports 19
  cells MATCHED/EQUIVALENT: 0.28-0.61x vLLM (v10 0.25-0.62x; isl16384 c32 0.31 -> 0.49). Recipe -> v11.
- 14:20 UTC, v11 DP2xTP4 (TP4 packet 92ae9bd5, collective knobs off, PF_INTERLEAVE 8192): FP32 PASS
  kl_mean 0.01251, no collective errors; strict reports vs vLLM DP2xTP4 15 cells MATCHED/EQUIVALENT,
  0.44-0.76x (isl16384 c64/c128 0.73/0.72). No qualified win.
- 21:00 UTC, v12 (bedd7980; packet 3707ed52): sequence-parallel band router + KDA norm, 16-byte prefill
  collective schedule, flash prefill heaviest-first work order (PLOW_MLA_PF_SNAKE), blocked HSA drain wait
  (removes a random ~18 ms stall), decode router tile + split-K 2P + XR_DEC_CUS, deferred-frame flash softmax.
  Gates PASS kl 0.01305; prefill 8k 272 -> 245 ms; serving +3-9% at c32/c128. Per-segment geometry sweep: no
  gain (the interpreter object's 148 KB LDS union + 256 VGPR cap every op at 1 WG x 8 waves/CU).
- 07:10 UTC (10-14), v15b = v15a + serve `PLOW_AMD_PACKED_SEG_MAJOR=1` (packed prefill chunks enqueue every segment
  and drain once instead of a host drain per segment). Screen vs v15a: c128 +4.2% (1k) / +6.5% (4k), c32 +0.8 / +4.6%.
  Gates PASS kl 0.01995, needles 36/36, split 18/18, prefix 49152 hit tokens (6/8 cached == cold, cache-off 5/8).
  Strict TP8 reports 19 cells MATCHED/EQUIVALENT: 0.37-0.66x vLLM (isl4096 c128 0.53 -> 0.57). Recipe -> v15b.
- 06:20 UTC (10-14), v15a (42c4efc2; packet 0abfd016; plowrt afd1509d): v14 (decode W8A8 split-N GLU tiles,
  standalone r1 mHC mix) + dedicated decode W8A8 MoE kernels (`PLOW_MOE_W8A8_DEC`, bit-identical; distinct-prompt
  B8/B32/B128 27.6/30.1/40.4 -> 25.5/28.0/39.4 ms) + KDA prefill intra/prepare/gated-norm/carry rework (object
  defaults, bit-identical, 8k -4%) + the dedicated MoE prefill kernels actually running (`PLOW_MOE_FP8_PF_XQ`,
  default on: v13/v14 built them but the FQ GLU arm missed the dedicated predicate, so `PLOW_MOE_FP8_PF_LEAN` ran
  the interpreter; bit-identical, 8k -12 ms) + fused weighted combine with the shared expert as expert 288 in
  prefill and packed serving (`PLOW_MOE_FP8_PF_ACC`, `PLOW_GLM_MOE_SHARED_FOLD`, `PLOW_MOE_FP8_PF_PACKED`;
  bf16 atomics, nondeterministic, FP32-gated). Gates PASS kl 0.02133 (vLLM 0.03677), needles 36/36, split 18/18,
  prefix 53248 hit tokens (7/8 cached == cold, as v14). Prefill 1k/4k/8k/16k/32k 73/121/196/408/883 ms; amd-bench
  r1/B8/B32/B128 11.89/24.43/24.81/33.69 ms. Strict TP8 reports 19 cells MATCHED/EQUIVALENT: 0.36-0.66x vLLM
  (isl16384 c32 0.58 -> 0.66, isl4096 c32 0.45 -> 0.50). No qualified win. Not taken: decode GEMM fold (no gain,
  spills), in-interpreter mHC tag (≈ the r1 mix kernel), adaptive MLA splits (-0.25 ms r1), DSA prefix fix
  (unvalidated). c128 serving: prefill and decode never overlap (prefill 47%/68% of wall at 1k/4k), ~6.5 ms per
  extra packed span, served B128 42-46 ms vs 33.7 bench. Vendor kernels (AITER asm fmoe, Tensile) not imported:
  the existing routes are gfx942/GLM-5.2-only. Recipe -> v15a.
- 01:16 UTC (10-14), v13 (16ca2dec; packet ab03f74f; plowrt 7b73692d): dedicated kernels outside the
  interpreter — NoPE MLA flash prefill (mla_pf_dense, flash 18.1 -> 9.9 ms per 8k chunk), block-FP8 grouped MoE
  prefill (moe_fp8_pf, pair 59 -> 47 ms), batched KDA decode chain (kda_decode_wide), grouped KDA prefill GEMM;
  fused decode mHC + grouped split-K projections (B128 packets 1714 -> 1056); lean KDA in packed serve prefill
  (packed launch 355 -> 312 ms); W8A8 MoE prefill+decode with vLLM's 1x128 activation groups (bug fixed: the
  quant op wrote scales at the chunk's runtime-row pitch, the GLU read them at the bucket's; fixed pitch on both
  sides). Gates PASS kl 0.01636, needles 36/36, split 18/18, prefix hits == cold. Prefill 1k/4k/8k/16k/32k
  87/143/229/475/1019 ms. Strict TP8 reports 19 cells MATCHED/EQUIVALENT: 0.34-0.61x vLLM (isl4096 c128 0.35 ->
  0.48, isl16384 c32 0.49 -> 0.58). No qualified win. vLLM per-kernel profile (same image/args): 8k prefill 170 ms,
  B=1 6.87 ms, B=128 ~14-17 ms; vLLM runs sparse MLA (plow DSA serves, gate PASS kl 0.0129, but no prefix hits or
  packed prefill yet) and W8A8 only for MoE + dense layers. Collectives at the XGMI floor. amd-bench --batched
  feeds one prompt to every slot, so it understates decode MoE (distinct prompts B128 41.4 vs 34.0 ms). Recipe -> v13.
- 17:35 UTC, E10 TP4 bench (packet pkt-glm53flash-tp4-ladder-v4, 4 GPUs, real prompt, streams
  exact at every rung): scalar (rung 8) 55.5 ms, B=8 51.4, B=32 63.3, B=64 74.6 ms (858 tok/s
  per group); prefill 1k 204 ms (TP8 173). Two such groups (DP2xTP4) would serve 128 streams at
  ~1715 tok/s / 75 ms vs TP8's 1408 tok/s / 91 ms at B=128 — a +22% throughput lever at c128,
  but plowrt has no data-parallel serving on AMD (two engines behind one router); hand-off.
- 05:00 UTC (10-13), decode floor at rungs 1/8 (branch kern-c1-decode-floor). Rung-1 trace on
  v8b: 1846 packets, ~953-hop chain, inter-hop gap ~0.5 us, op body ~22 us/hop; workgroups with
  NO work in a wide packet still spent ~18-20 us from ready to end. Not L2 invalidation
  (PLOW_GATE_NOINV diagnostic: -3%) and not I-cache: `plow_exec` was a real call whose prologue
  saved ~110 VGPR dwords/lane to scratch before selecting the op, every packet, every WG (the
  gfx942 direct-GEMV path that skips it is off for FP8/K3). Landed, all streams byte-equal to
  v8b at every rung and on 4k/16k prompts (32-step validate):
  (a) f2160842 + 7545d2fa: workgroup-per-output decode GemvF32 and two emit folds
  (PLOW_GLM_HC_NORM_FOLD, PLOW_GLM_HC_POST_FOLD; rung 1 1846 -> 1678 packets, critical path
  909 -> 741): r1/B8/B32/B128 26.1/44.5/55.4/84.9 -> 24.1/42.1/54.4/82.9 ms;
  (b) 76d03d94 + d70122b6: gfx950 K3 decode objects inline `plow_exec` (spills tiers 0 -> 83,
  wide 6 -> 198): r1 24.1 -> 12.3-12.9, B8 42.1 -> 31.2, B32 54.4 -> 53.0 ms, B128 82.0-86.1 vs
  82.3-91.3 (bimodal, within noise), 4k/16k r1 23.5/23.7 -> 13.0/13.1 ms. A GEMV-only direct
  arm instead gave r1 19.4 ms. Serving cells not re-run.
- 07:40 UTC (10-13), prefill toward roofline (branch kern-prefill-roofline). Per-segment timing
  (`PLOW_PREFILL_SEG_TIMING`) on pkt-c: a 4k chunk is ~85% interpreter segments, then flash and
  the lean KDA intra/carry; the first prefill after load pays ~140 ms of cold cost, so per-op
  numbers below are warm chunks (rank-0 trace, ms per chunk, before -> after; floor at 8192):
  FlashMlaPrefill 105.7 -> 78.5 (8192 chunk at 16k ctx; floor ~8), MoeGroupDownPf 56.7 -> 32.9
  (~7.5), MoeGroupGluPf 44.4 -> 33.4 (~6.4), MlaMergeFold 19.4 -> 7.0, router top-k 18.7 -> 9.2,
  GemmSmall 25.0 -> 19.1, GemmF32 8.7 -> 5.9, KdaConv3 8.4 -> 3.8, lean KDA intra 18.4 -> 11.9;
  collectives unchanged (RS ~30, AG ~20 vs ~17.6 each at 300 GB/s). All bit-identical and code
  defaults for GLM-5.3-Flash objects: 288-expert wave router + all-wave align, KDA intra solve /
  conv3 rows / prepare ILP, MoE DEEP + HDR, GEMM rung re-pick + BK=128 small rung, TR16 flash,
  token-blocked fold. Validate on the recipe-env objects: every stream byte-equal at rungs
  1/8/32/128 and 4k/16k/32k/60k prompts; prefill sweep 1k/2k/4k/8k/16k/32k (min of 2)
  112/139/198/359/759/1769 -> 94/116/167/283/625/1421 ms. Knobs left off: deferred-frame flash
  softmax (`FA_MLA_PF2_DEFER=1`, 16k flash 79 -> 59 ms, streams differ at 16k), MFMA fold
  (`PLOW_K3_PF_FOLD_MFMA=1`, streams differ), `PLOW_K3_PF_XR_SCHED=aiter` (~2%, quick streams
  equal, no full validate yet).
- 08:35 UTC (10-13), DP2xTP4 serve (branch serve-dp2tp4-amd, 640f2794): `PLOW_DP=2` on an AMD
  serve loads the TP4 packet twice (GPUs 0-3, 4-7) as `slug#0/#1`, each with its own dispatcher,
  per-slot prefix cache and socket-pinned engine thread, behind the existing DP router
  (least-loaded; continuation affinity; the cross-request prefix probe is CUDA-VMM only).
  TP4 packet = v9 [emit].env, `--num-gpus 4`, ladder 1,8,32,64 (64 slots/group), max-ctx 65536
  (5fd6b2ca; 141 GiB free per GPU after load). amd-bench, both groups at once, nat-pride-1024:
  r1/B8/B32/B64 13.1/33.9/61.2/81.0 ms, every slot equal to the FP32 continuation; 4k/16k
  synthetic prompts keep TP8's first token and then differ (repetition near-ties). Serve: load
  2 x 12 s; c1 greedy pairs (one copy per group) equal 8/8; vs TP8 serve 7/8, the eighth at a
  TP8 exact tie (gap 0.0); needles 4k/16k 18/18 (cache on); FP32 gate PASS kl_mean 0.01258,
  kl_p99 0.172, top1_decisive 0.9967, needles 18/18 (v9 TP8: 0.01384 / 0.237 / 0.9989).
- 11:15 UTC (10-13), co-resident models on AMD (branch serve-multimodel-amd2): an AMD serve places
  every bundle (and `--dp` rank) with `serve::placement` on its own TP group (`--pin S@first`,
  else spread) and loads it there; `--assets DIR,slug=NAME` serves one packet under two names.
  Each model keeps its own engine, dispatcher, per-slot prefix cache and socket-pinned engine
  thread; turns stay free for a group of one. Launch: `plowrt serve --assets <tp4>,slug=glm-a
  --pin glm-a@0 --assets <tp4>,slug=glm-b --pin glm-b@4`. v10 TP4 packet twice (GPUs 0-3, 4-7):
  load 2 x 13 s; `/v1/models` lists both; c1 greedy streams of each model equal a single-model
  TP4 serve 8/8, one model at a time and both at once; needles 4k/16k 18/18 per model (cache on,
  both models at once); c64 per model, isl 1024 / osl 128: 6151 + 6163 = 12303 total tok/s,
  128/128 each; metrics carry `model_name` per model.
- 20:00 UTC (10-13), sparse attention like vLLM (branch v12-dsa). vLLM's TP8 log runs
  `ROCM_AITER_MLA_SPARSE` with the `DEEPSEEK_V32_INDEXER/KPOOL_TAIL` indexer
  (`vllm/models/glm5next/amd/sparse_indexer.py`): kpool-4 softmax(gate+ape)-pooled, Hadamard,
  power-of-2-scaled FP8 index keys; FP8 Hadamard queries; fp32 weights; top `2048/4 = 512` pools
  per query row, expanded to tokens plus the incomplete-pool tail (<= 2051 indices); at prefill
  AND decode; rows whose length is <= 2048 take every causal token (the identity). plow's pooled
  ops were ported from the same kernels, so the selected SET is the same (tie order aside).
  `PLOW_GLM_DSA=topk` arms it at every decode rung (batched arm added: per-slot ring/pool cache,
  per-row radix, compact expand) and makes every prefill bucket write the pool cache (dense
  buckets used to skip it; packed programs write it per span). Differences left: prefill
  attention stays dense (`PLOW_GLM_DSA_PF=1` faults on a ragged 4096-row chunk in the 8192
  bucket; a dense row is a superset of its selection), and a pool straddling an unaligned packed
  span start is not compressed. Production (`PLOW_GLM_DSA=0`) packets are byte-identical.
  Validate (packet caf12efe, `PLOW_GLM_DSA=topk`): nat-pride-1024 equals the FP32 continuation
  in 1/8/32/128 of 1/8/32/128 slots; real 4k/16k prompts give the dense packet's stream at r1 and
  in all 8 b8 slots; 32k/61k run. Not qualified: decode cost. ms/step sparse vs dense:
  r1 26.2 vs 12.4, B8 39.8 vs 22.9, B32 85.6 vs 30.2, B128 263.7 vs 39.9 (1k ctx);
  r1 at 4k/16k/61k 27.4/29.2/37.1 vs 11.5/11.6/12.6. Prefill (dense attention + pool writes)
  4k/16k/32k/61k 164/601/1345/3053 ms. Needles and the FP32 gate not run.
  Later (v12 env + `PLOW_GLM_DSA=topk PLOW_GLM_DSA_PF=1 PLOW_GLM_DSA_FUSE=1
  PLOW_GLM_DECODE_GLUE_CUS=1`, objects default `PLOW_DSA_KPOOL_MFMA=1 PLOW_DSA_QQUANT_WAVE=1`):
  IndexScoreKpool on bf16 MFMA (fp8 operands are exact in bf16), items slab-major (row-major
  pinned each workgroup to one slab: 4.8 ms -> 0.5 ms per 8k chunk layer), 8 prefill rows per
  key slab; ragged prefill chunks shrink the pooled chain (the 4096-in-8192 fault); stash/expand
  fused into compress/select; q quant one wave per row. Sparse vs dense (v12 int), ms/step:
  r1 11.9 vs 11.5, B8 23.7 vs 22.8, B32 27.7 vs 26.9-29.3, B128 41.0 vs 36.5-38.7, r1 at
  4k/16k/61k 12.4/12.5/12.7 vs 11.7/11.9/12.8; prefill 4k/16k/32k/61k 173-191/598/1254/2560 vs
  159/561/1271/- ms. Streams = FP32 continuation at all rungs; 4k/16k = dense stream; needles
  36/36 (serve, cache on). Serve with the packed route crashed (per-span pool writes trap in the
  interpreter object): the packed route is refused for DSA packets until those writes move into
  a packed family. On the v13 integration env + DSA (packet 21ffb3d1, serve PF_IL 8192, XR tag
  90112, dense flash object): FP32 gate PASS kl_mean 0.01289 (vLLM 0.03677), kl_p99 0.199,
  top1_decisive 0.9978, c16 serve, no exception; PF_CHUNK=512 split needles 18/18. Prefix check:
  the cache never hits on the DSA packet (cached=0 in every case) and 7 of 24 streams differ
  off-vs-on (run-to-run concurrency; the dense v12 packet also FAILs this check with 2) — open.
- 23:00 UTC (10-13), dedicated decode kernels for KDA and the router (branch v13-kda-router).
  `PLOW_KDA_DECODE_WIDE` (default on): at batched rungs each KDA layer's Conv3 -> StateStepG ->
  GatedNorm runs as one standalone op-125 v3 launch (`kda_decode_wide.hip`, one 512-thread
  workgroup per (row, head), 69 VGPR / occ 7, the interpreter's own bodies). Streams equal base at
  r1/B8/B32/B128, with a distinct prompt per slot at B32/B128 (30.92/44.44 -> 29.36/41.42 ms) and
  for 4k/16k prompts; B=8/32/128 23.98/27.02/36.87 -> 22.00/25.92/34.03 ms/step; rung 1 keeps the
  fused arm (a segment boundary costs ~7 us, more than a raw kernel saves on 8 workgroups). An
  agent-scope `__threadfence` in that kernel made B=128 47.5 ms (L2 writeback per workgroup);
  workgroup-scope ordering is enough because the intermediates never leave the workgroup.
  Dropped (measured together, streams equal): a standalone router at batched rungs (GemmF32 tile +
  top-k + align, agent-scope last-arriver counters) plus wave-per-token op-83 top-k in the decode
  objects and at rung 1: r1/B8/B32/B128 +0.66/+1.1/+2.4/+4.6 ms.
- 21:50 UTC (10-13), collectives vs vLLM/AITER (branch v13-allreduce, v12 int packet 3707ed52).
  All-rank traces (`PLOW_TRACE_ALLRANKS=1`); "min-rank" = per instance the smallest body over the
  8 ranks (the last arriver: transfer + protocol, no peer wait).
  - Microbench, 8x MI350X (`tp_allreduce_prefill_bench` TP_PHASES, 8192x4096, 16-byte schedule):
    reduce-scatter + all-gather 322 us at 24-32 workgroups (RS 164 + AG 158), 328 us at the shipped
    caps (RS 16 = 169, AG 48 = 159). vLLM's `cross_device_reduce_2stage` is 307-345 us for the same
    64 MB, so the data movement is already at parity; caps 24/32 would save ~0.5 ms per 8k chunk.
  - Warm 8k prefill (244 ms): RS 90 x 213 us rank 0 / 175 us min-rank; the 91 hidden-wide AGs
    170-260 us min-rank (median ~195), plus 42 router-table AGs of ~24 us. Rank-0 total 43.6 ms,
    min-rank 34.8 ms; the ~9 ms difference is arrival skew from the band ops, not the collective.
    The brief's "RS 31 + AG 20 = 51 ms" was cold/rank-0 envelopes.
  - Decode B=1: XReduce 90 x 9.6 us rank 0 (0.87 ms/step), 7.0 us min-rank; the earlier 16.8 us
    figure included the gate wait. Tagged one-shot microbench: 4.7 us with a producer (8 WGs).
    The chain's large hop is the mHC GemvF32 packet after it (37.7 us/hop, 3.39 ms/step).
  - Tried and dropped (bit-identical, streams equal at every rung, not shipped):
    aiter `ar_ll`-style PUSH tagged one-shot (sender stores into every receiver's sub-slot, receiver
    polls local memory): microbench 6.0 vs 4.7 us (pull) at B=1, in situ r1 11.83 vs 11.69 ms.
    Decode two-shot on the 16-byte schedule for B >= 32 (`PLOW_XR_DEC_TWOSHOT_MIN=131072` + the
    decode objects built with `PLOW_XR_SCHED_AITER`, caps 32/16): microbench [128,4096] 24.5 vs
    31.9 us one-shot, in situ b128 36.75 vs 36.85 ms, b32 28.29 (x3) vs base 27.07-28.28
    (bimodal) — not better. B=1 tagged one-shot at 1/2/4/8 workgroups: 13.8/9.3/5.5/4.7 us, so
    the emitted 8 stays.
  - Kept: `PLOW_XR_TAG_SLOT=90112` (decode objects AND serve env; existing knob, bit-identical)
    lets B=8's [8,4096] message take the tagged one-shot: b8 22.80 -> 22.46 ms/step (three
    alternating passes each, streams equal at r1/b8/b32/b128 and 4k/16k prompts); other rungs
    unchanged. Serve FP32 gate with the slot (c16, PF_INTERLEAVE 8192): PASS kl_mean 0.01297,
    top1_decisive 0.9978, 0 errors, no cross-GPU gate faults.
  - Fusion: RS + HyperConnPost on the band does not pay (the RS runs on 16 fabric-bound
    workgroups; HC post moves 8x its bytes locally). The remaining decode lever is folding the
    tagged poll+reduce into the consumer mHC packet (removes the XReduce hop, ceiling ~0.6 ms/step
    at B=1); it overlaps the v13 mHC post+mix+pre packet and is left for after that lands.

## 5. Results

Strict `campaign.py report` output per input length and the combined comparison.csv:
[docs/bringup/results/glm53flash-mi350x-2026-10-12](../bringup/results/glm53flash-mi350x-2026-10-12/summary.md).
No qualified win: plow total throughput 0.37-0.66x vLLM (v15b; v15a 0.36-0.66x, v13 0.34-0.61x, v11 0.28-0.61x).

## 6. Roofline (CPU, before any plow GPU time)

Decode program (T=1, per rank, TP8), weight-stream floor from `packet_roofline.analyze`:

| ctx | logical operand bytes / rank-step | floor @ 6.2 TB/s | ops / step |
|---|---:|---:|---:|
| 1024 | 4.52 GB | 0.73 ms | 1854 |
| 8192 | 4.52 GB (+KV, excluded) | 0.73 ms | 1854 |

Where the bytes go: `lm_head` bf16 **replicated on every rank** 1.27 GB (28%); KDA bf16
projections (`GemvQkv` 34x + gates) 1.18 GB (26%); routed experts fp8 (8 of 288 per layer, 42
layers) 1.06 GB (23%); MLA fp8 projections ~0.1 GB; everything else < 5%. The excluded ops
(90 `XReduce`, 90 `HyperConnPre`/`Post`, 90 `GemvF32`, 34x3 KDA state ops, norms) are
latency-bound, not byte-bound: the gfx950 GLM-5.3 campaign measured ~20 us per op in the
persistent interpreter, so 1854 ops/step puts the **dispatch floor near 37 ms/token**, far above
the 0.73 ms byte floor and above vLLM's TPOT. Op count, not bandwidth, is the decode roofline.

Prefill (per rank, bucket T): `op_roofline.py` floor 2048 -> 10.8 ms, 8192 -> 34.7 ms; the 87
`XReduceTwoShot` collectives are 22-27% of the floor (fabric-bound, 67 MB each at T=8192), then
the grouped MoE GLU/down (priced at the MXFP4 contract; the packet streams fp8, so x2) and
`MoeCombinePf`. Levers, in roofline order: decode op-count fusions (`PLOW_KDA_DECODE_FUSED[_ARM]`,
`PLOW_GLM_FUSE_B1/SEAM`, `GLM_FUSE_XRN`, `PLOW_GLM_ROUTED_W8A8` = 3 ops/layer instead of 16,
`PLOW_GLM_MOE_FLAT_DECODE`, `GLM_SHARD_HEAD`), then prefill seq-par collectives
(`PLOW_GLM_SEQ_PAR`). Each is an op-count experiment first (CPU emit + disasm), then a smoke, then
a cell. Measured per-op times replace the 20 us assumption once the smoke serves.

### 6a. Measured worklist (after the traces, 13:50 UTC)

Per-op spans from `PLOW_TRACE_RAW` on the final objects (v2f single-rung, v2y ladder), per
rank, TP8. "Floor" is the weight/activation byte floor at 6.2 TB/s or the fabric floor for the
collectives; the gap column is what a kernel change can win at most.

| phase | op | measured | floor | gap | lever |
|---|---|---:|---:|---:|---|
| decode B=1 | whole step (1846 packets) | 27.3 ms | 0.73 ms bytes, ~22 ms dispatch (12 us/packet) | packet count | fewer packets per layer: fused KDA step arm (`PLOW_KDA_DECODE_FUSED_ARM`, off under the ladder), fold HC_PRE/Post into neighbours (3 of 41 packets/layer), fold XReduce into the consumer. No kernel is above 10% of the step. |
| decode B=128 | GEMV (477) | 126 ms | ~6 ms (weights once + 128 rows of x) | 20x | DONE in packet v3: route to `GemmSmall` (`PLOW_GLM_DECODE_GEMM_ROWS=8`). |
| decode B=128 | KdaStateStepG (34) | 11.7 ms | 34 x 64 MB state r+w = 0.7 ms | 16x | state in registers across the B axis is the prefill carry design; one block per (row, head) with the tile loop inside. |
| decode B=128 | RmsNorm (102) | 11.0 ms | 0.3 ms | 30x | `pf_wide_cus` gives 101 blocks for 128 rows; fuse into the consumer GEMM prologue, or 1 wave/row. |
| decode B=128 | XReduce (90) | 8.8 ms | 90 x 1 MB over xGMI ~ 3 ms | 3x | two-shot form (prefill already uses it); or `GLM_FUSE_XRN`. |
| decode B=128 | FlashMlaDecode (11) + MlaMergeFold (11) | 15.2 ms | 11 x 128 rows x 1 K ctx x 576 B = 0.8 GB -> 0.13 ms | 100x | the decode flash is one block per (row, head-group) at 1 K context; at 128 rows it needs the prefill-style row tiling. Largest post-v3 item at long contexts. |
| prefill 4 K | GemvF32 hyper-conn mix (90) | 72 ms | 90 x 4096 x 16384 x 4 B x re-reads; LDS-tiled ~6 ms | 12x | LDS-tile the 24 mix columns per row block; the op is 15% of the 4 K forward. |
| prefill 4 K | MoeGroupDownPf / GluPf (45+45) | 58 + 36 ms | fp8 weight stream 2 x 1.06 GB -> 0.35 ms; tokens x top-8 x 2 x 4096 x 2048 MACs ~ 20 ms MFMA | 2-3x | the grouped kernels are at ~35% of MFMA peak at this shape; the A4W4/fp8 tile binsearch (`PLOW_MOE_TILE_BINSEARCH`) is the knob to sweep. |
| prefill 4 K | FlashMlaPrefill (11) | 59 ms | 11 x 4096^2/2 x 8 heads x 576 B ~ 0.4 TFLOP -> 0.4 ms | 100x | the NoPE flash prefill runs at ~1% of MFMA peak: it is the full:128..8192 bucket path with `PLOW_GLM_DSA=0`; the V2 raw-MLA object (`PLOW_MLA_PF_V2`) is the qualified gfx942 route and was not brought up here. DONE (recipe): the packet was emitted without `PLOW_MLA_PF_V2=1`, so the program was one segment and op 51 ran the scalar `d_flash_mla_prefill<512,0,4>` decode body on the 8-wave object. Emitting with `PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=2` routes the 2048/8192 buckets to the NoPE V2 MFMA arm: 11 x 5.55 -> 11 x 1.13 ms at 4 K, 32 K prefill 6431 -> 2767 ms. |
| prefill 4 K | HyperConnPre (90) | 37 ms | 90 x 4096 rows x 16 KB -> 1 ms | 30x | one-lane Sinkhorn per row; row-parallel 4x4 Sinkhorn on a wave (designed, not implemented). |
| prefill 4 K | XReduceTwoShot (87) | 36 ms | 87 x 32 MB / 8 ranks over xGMI ~ 25 ms | 1.4x | at the fabric floor; only `PLOW_GLM_SEQ_PAR` (sequence-parallel seams) removes bytes. |
| prefill 4 K | KDA chunk carry (34) | 36 ms | runs on 64 of 256 CUs | 4x | spread the carry over the head x chunk grid. |
| serving c128 | prefill stalls | step 162 -> TPOT 342 ms | - | 2x | chunked prefill co-scheduled with the decode batch (vLLM does this); plow admits whole prefills between decode quanta. |

What this says about the campaign goal: vLLM's c1 TPOT (6.9 ms) is below plow's dispatch
floor (1846 packets x 12 us); vLLM's 4 K TTFT (105 ms) needs every prefill op above halved
at once; and at c128 the served throughput is bounded by prefill rate (4096 tokens / 471 ms =
8.7 K tok/s per request stream vs vLLM's ~39 K), so no (context, concurrency) cell is winnable
on this packet. The v3 batched rungs are still worth measuring: GEMV was 78% of the B=128
step.
