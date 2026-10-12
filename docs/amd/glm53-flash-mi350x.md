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

## 5. Results

(pending: the strict 12-row `campaign.py report` tables go here with the comparison.csv link.)

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
| prefill 4 K | FlashMlaPrefill (11) | 59 ms | 11 x 4096^2/2 x 8 heads x 576 B ~ 0.4 TFLOP -> 0.4 ms | 100x | the NoPE flash prefill runs at ~1% of MFMA peak: it is the full:128..8192 bucket path with `PLOW_GLM_DSA=0`; the V2 raw-MLA object (`PLOW_MLA_PF_V2`) is the qualified gfx942 route and was not brought up here. |
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
