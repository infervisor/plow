# Kimi-K3 on gfx950 — how to build, run, measure, and every knob

Everything needed to reproduce the K3 numbers on this branch. Written because the
campaign repeatedly lost time to environment rules and measurement traps that are
invisible from outside, and to a set of knobs that existed only in commit messages.

**Model.** 93 layers (69 KDA linear-attention + 24 MLA), hidden 7168, latent 3584,
896 routed experts top-16 mxfp4, 2 shared experts, vocab 163840. TP8 is not optional:
the checkpoint is ~1.5 TB and a rank uploads 195 GiB.

---

## 0. THE FOUR ENVIRONMENT RULES. All load-bearing.

1. **`sg render -c "…"` MUST be OUTSIDE `nix develop`.** nix runs in a user namespace
   where root maps to `nobody`, so `/usr/bin/newgrp`'s setuid bit is inert inside it and
   `sg` dies with `setgroups: Operation not permitted`. Without the render gid `hsa_init`
   fails 4104.
2. **Every GPU command takes `flock /tmp/plow_gpu.lock`**, INSIDE the `sg` quote so the
   lock is held for the whole run. The box is shared. A benchmark on a contended GPU is
   worthless — one measured rep came in at 54.6 ms against a 40.6 baseline.
   *Do not infer from `ps` that someone is not locking:* `flock CMD` never appears in the
   CHILD's argv, only the parent's. Grepping for `amd-bench` returns only the leaf.
3. **cmake from nix, hipcc from SYSTEM ROCm** (`/opt/rocm`). If hipcc needs it:
   `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu`.
4. **`plowrt` MUST be built `--features hsa`.** A default `cargo build -p plowrt` produces a
   binary that does NOT fail — it serves from the CPU reference interpreter through the
   byte-fallback tokenizer, i.e. fluent garbage, "ready after 2s" instead of a 105 s load.
   `target/release/plowrt` is SHARED; copy it aside before a long benchmark
   (`PLOWRT_BIN`) so a concurrent build cannot swap it mid-run.

---

## 1. Build

```bash
# kernels (49 code objects)
nix develop --command bash -c "LD_LIBRARY_PATH=/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu \
  cmake --build build-amd --target gfx950_hsaco -j 32"
# host
nix develop --command cargo build --release -p plowc
nix develop --command cargo build --release -p plowrt --features hsa
```

A fresh cmake configure needs `-DPLOW_GFX950_HSACO=ON` plus
`-DPLOW_HSACO_{K3,MLA,GQ,FP8,FP8KV,MXFP4}=ON`; without `PLOW_GFX950_HSACO` the
`gfx950_hsaco` target does not exist and `cmake --build` says "No rule to make target".

**The canary.** `interp_decode_fp8kv_k3_gq` must build at **248 VGPR / 0 AGPR / occ 2 /
0 spill / LDS 147472 B**. The build prints it. 248 is `512/waves_per_eu` minus a granule
and is INDEPENDENT of which opcode arms compile — the bare decode bucket is also 248.

---

## 2. Prepare the model

```bash
python3 scripts/kimi_k3_tokenizer.py   …   # tokenizer.json from tiktoken.model
python3 scripts/kimi_k3_prep.py --derived --farm /home/lava/models/k3_farm
```

The checkpoint is consumed as a **symlink farm**: `Checkpoint::open` globs every
`*.safetensors` in ONE directory, never reads an index, sorts the paths and lets the last
writer win — so the sidecar of derived tensors must sort LAST. `build_farm` asserts that.

---

## 3. Emit a blob

```bash
nix develop --command bash -c "K3_FULL=1 PLOW_FP8_KV=1 PLOW_MXFP4=1 ./target/release/plowc \
  --hf-dir /home/lava/models/k3_farm --emit devblob --arch gfx950 --gpu mi350 \
  --num-gpus 8 --parallel tp --max-ctx 32768 --n-cu 256 --out /home/lava/models/k3_tr"
```

Expect `emitted 93 layers (69 KDA, 24 MLA), tp=8, 5503 tensors, 2459 decode instructions`
and `counter-graph reduction: 69 … 138 duplicate waits; 207 wait entries removed`.

To serve from it, the assets dir also needs `checkpoint`, `tokenizer.json` and `hsaco`
(symlinks are fine). Serve reads `<assets>/hsaco`; there is no `--hsaco` flag, and it will
REFUSE to start if a prefill object is missing rather than load one without the arms.

---

## 4. Run

**Decode bench** (this is the number quoted as ms/token):

```bash
sg render -c "flock /tmp/plow_gpu.lock nix develop --command ./target/release/plowrt amd-bench \
  --blob /home/lava/models/k3_tr/model.pkt --hsaco build-amd/hsaco \
  --checkpoint /home/lava/models/k3_farm --steps 200 --ctx 32000 --tp 8"
```

**Correctness gate.** `--prompt` takes COMMA-SEPARATED TOKEN IDS, not text.
`"The capital of France is"` = `1008,10484,318,15383,387`, and the known-good continuation is

> ` Paris. The population is approximately 67 million people. The official language is French. The currency is the Euro. France is`

All 8 ranks must be token-identical. Anything else is a fail, however plausible it reads.

**Serving.** `scripts/bench_plowrt_serve.sh <assets> <port> <model> <tokenizer> [ready]`
with `TOKZ_MOUNT='-v /home/lava/models/k3_tokz:/tokz:ro' TOKZ=/tokz` — K3's HF snapshot has
no loadable fast tokenizer (it declares `tokenizer_class=TikTokenTokenizer` with an
`auto_map`, so the client would need trust_remote_code AND tiktoken). The served model id
comes from the blob's network name, not a flag: query `/v1/models`, don't guess.

---

## 5. MEASUREMENT DISCIPLINE — five traps that produced wrong numbers

1. **`--steps 24` is not a measurement.** 24 steps is ~1 s of decode behind a ~105 s weight
   load; measured spread 18% vs 1.2% at `--steps 200`. Use >= 200 and report minima.
2. **Back-to-back `--checkpoint` arms are ORDER-BIASED by 5–10 ms.** Two bound runs re-read
   195 GiB/rank and the SECOND of each pair is penalised. Base-first said a change was 6–9 ms
   slower; shard-first said the opposite — the sign followed the ORDER, not the blob. For
   sub-1-ms work use unbound-weight runs (identical schedule and buffers, no load) and always
   run the order-reversed control. See `perf-data/amd-bench-ab-order-bias.md`.
3. **Without `--checkpoint`, routed experts do not execute.** They bail at `wg_base == 0`, so
   the MoE is silently absent and every ms/token is optimistic. Ratios stay usable; absolute
   numbers do not.
4. **`vllm bench serve` counts a REJECTED request as successful.** A 131072-input point
   against a `max_ctx=131072` blob (the chat template adds ~13 tokens, so every prefill was
   refused) still reported 4 successful requests, 99.1 tok/s and ITL 0.00. Gate on
   `gen_toks == num_prompts * OUTLEN`, which the harness now emits.
5. **`timeout N sg render -c "flock …"` kills the LOCK WAIT.** A queued run dies before it
   starts and reports as a blank row indistinguishable from a failure. Don't wrap the lock.
6. **`--ctx N` decodes over KV NOBODY PREFILLED** (`plowrt/src/main.rs:391`), so at `--ctx 32000`
   with a 5-token prompt every decode step attends over ~32k rows of uninitialised memory. Timing
   is fine; **the step logits are not a function of the program alone**, so a CORRECTNESS A/B must
   run at `--ctx <prompt length>` (`--ctx 5` for the gate prompt) where decode continues from the
   rows the prefill actually wrote. Before reading any A/B, run the SAME blob twice and confirm it
   reproduces itself — it does, 33/33 `--dump-logits` files — otherwise a divergence is unreadable.

Warmup: arms emitted with different `--n-cu` were observed to sample warmup token 0 where
`--n-cu 256` samples 220 — not a like-for-like computation. `scripts/geom_check.sh` gates it.

---

## 6. Knobs

### Emit-time (env, read by `plowc`)

| knob | default | effect |
|---|---|---|
| `K3_FULL=1` | off | REQUIRED for the real 93-layer emit; otherwise an analysis-and-refusal path |
| `K3_NLAYERS=n` | all | truncate. NOTE: a truncated K3 is not a language model — degenerate output at small n carries NO signal |
| `PLOW_FP8_KV=1` | off | fp8 KV cache. Selects `FlashMlaDecodeFp8`/`HeadNormRopeFp8` |
| `PLOW_MXFP4=1` | off | mxfp4 routed experts |
| `K3_MOE_GROUP` | **on** (`!= "0"`) | grouped MoE path |
| `PLOW_K3_SHARD_UP` | **on** (`!= "0"`, needs tp>1) | column-parallel `routed_expert_up_proj`, gather folded into the shared reduce. −22.8% GEMV bytes |
| `PLOW_K3_SHARD_HEAD=1` | off | column-parallel `lm_head` (`XArgmaxFin`). Gate passes; `k3_tp_equivalence.sh` CANNOT gate it — `--dump-logits` dumps `act.logits`, `vocab/tp` wide at tp=8, so its shape check fails by construction |
| `PLOW_K3_FUSE_ARNORM` | **on** (`!= "0"`) | fuse the AttnRes norm |
| `PLOW_K3_FUSE_NGEMV=1` | **off** | fold the `b=1` RMSNORM gates into the `b=256` GEMVs that read them (`routed_expert_norm` x92 fan=1, `q_a_layernorm` x24 fan=2). Removes all 116 RMSNORM packets; critical path 1831 -> 1715. **NOT BIT-EXACT END-TO-END YET** — bit-exact in the isolated hardware gate, diverges ~1 ULP through the full model. Do not enable for a real serve. `perf-data/k3-narrow-gate-fusion.md` |
| `PLOW_SEG_PER_OP=1` | off | one AQL segment per op (host-side chaining). **BROKEN at TP8** — ranks desync, a collective hits its deadline and returns WITHOUT reducing |
| `PLOW_FINE_FORCE=1` | off | keep genuinely-sparse `Dep::Fine` edges. **No-op on K3** — the K3 emitter creates zero fine deps |
| `PLOW_UNISEG=1` | off | force one wave-class segment |
| `PLOW_L2_PLACE` | off | L2-domain packet placement. Measured ZERO (39.579 vs 39.628) |
| `PLOW_TR_QUIET=1` | off | silence the counter-graph reduction line |
| `PLOW_SKIP_COVERAGE=1` | off | downgrade the checkpoint-coverage gate to a warning |
| `PLOW_K3_UP_NOGATHER`, `PLOW_K3_UP_GATHER_ONLY` | off | bisection instruments, WRONG BY CONSTRUCTION, never serving modes. Exporting either makes `cargo test -p devgen` fail |

### Kernel build-time (`-DPLOW_HSACO_EXTRA_DEFINES='-DX=1'`)

| knob | default | effect |
|---|---|---|
| `PLOW_GATE_SC1` | 0 | device/system-scope activation stores + relaxed signal, deleting the release `buffer_wbl2`. **BROKEN under TP in both variants tried** (`sc1` and `sc0 sc1`): the ragged scalar activation tails still store plainly, and the release was publishing those too. Scope AND coverage must both be right |
| `PLOW_GATE_NOINV`, `PLOW_GATE_RELAXSIG` | 0 | ceiling instruments. Drop `buffer_inv` / `buffer_wbl2`. **REAL DATA RACES, never ship.** Together −5.04 ms/token (−13.7%) — that is the CEILING on any protocol rewrite |
| `PLOW_ACT_NT` | 0 | non-temporal activation stores. MEASURED 4% SLOWER — the writeback is not volume-proportional and activations ARE re-read. Do not re-try |
| `PLOW_ICACHE_INV_PROBE` | undefined | `s_icache_inv` per packet to bound the I-cache loss from ABOVE. gfx950 has NO instruction-prefetch instruction (`s_inst_prefetch` is GFX10/11, `s_prefetch_inst` GFX12), so bounding from above is the only lever. L1I hit is 99.88%; forced 100% miss costs 2.2% |
| `PLOW_WG_WAVES` | 8 | 4 loses end-to-end (44.8–47.4 vs 40.6 — half the resident waves per CU); 16 does not compile (`op_gemm.h:140`, wave grid is 4 or 8 only) |
| `PLOW_ALLOW_UNDERFILL`, `PLOW_DECODE_WAVES`, `PLOW_WG_THREADS` | inert | geometry instruments (branch `geom-sweep`) |
| `GV_UNROLL` | 10 (14 for the K3 decode axis) | GEMV prefetch depth; tuned against the ragged tail, not latency |

### Runtime (env, read by `plowrt`)

| knob | default | effect |
|---|---|---|
| `PLOW_GLOBAL_QUEUE` | on for decode | shared-cursor work stealing. Beats static 39.571 vs 41.657 — do not disable casually |
| `PLOW_SHARE_CKPT` | **on** (`!= "0"`) | rank 0 populates the PTEs; per-rank mmap costs 44 s/rank in minor faults |
| `PLOW_TP_AGREE_EVERY` | sampled | cross-rank token agreement cadence |
| `PLOW_TP_NO_AUDIT=1` | off | disables the per-token `xctr` arrival check. Then a timed-out collective is SILENT |
| `PLOW_TRACE_RAW=<path>` | off | per-(workgroup, packet) timeline; read with `scripts/k3_trace_report.py` |
| `PLOWRT_BIN`, `TOKZ_MOUNT` | — | benchmark harness isolation (see §0.4 and §4) |

---

## 7. Where the time goes (measured, 93 layers, TP8, ctx 32000)

```
token ~36-40 ms       target 10.0 ms (100 tok/s)
  ~20 ms  PROTOCOL    2459 packets; an EMPTY b=256 packet still costs 5.72 us
                      of which the GATE (waiting) is only ~5.8 ms — the rest is
                      the signal side: 256 workgroups each issue a CACHE-WIDE
                      buffer_wbl2/buffer_inv, which are PER-L2, so each XCD's L2
                      repeats the work 32x and serialises.
                      Grid sweep WG/XCD 32/16/8/4/1 -> 13.20/4.35/2.73/1.85/1.05 us
  ~19 ms  BODIES      92% the weight-bandwidth-bound GEMV family
                      14.021 GB/rank/token after up-sharding; 2.837 ms pure-BW floor
```

The dependency graph is nearly a chain: **critical path 1739 of 2459 packets, mean width
1.41, at most 5 counters live at once**. So chain depth x per-packet cost is a hard floor —
1739 x 5.72 us = 9.95 ms, the whole budget, before a single weight byte is read.

**Even a free protocol leaves bodies at ~19 ms.** 100 tok/s needs both halves.

## 8. Ruled out, with the number (do not re-buy)

| lever | result |
|---|---|
| per-XCD counter fan-out (FAN=8, XCCFAN) | 13.55 / 13.51 vs 13.20 — nothing |
| counter stride / placement | 128 B is sequential, non-aliasing |
| reordering the wait sweep | one thread per counter, mean wait list 1.20 — no variable exists |
| L2-domain placement | 39.579 vs 39.628 |
| non-temporal activation stores | 4% slower |
| i-cache work | 99.88% hit; forced 100% miss costs 2.2% |
| interpreter specialised to K3 opcodes | .text −51%, VGPR unchanged (248 is emergent) |
| segmented decode | 48–1026 extra launches/token; benefit zero |
| occupancy 4 (128 VGPR) | **+26.4%**, and unreachable anyway (grid pinned to CU count) |
| device-side AQL enqueue | 12.1–13.2 us/packet round trip |
| host op-by-op AQL at TP8 | ranks desync; collective returns without reducing |

## 9. Open, ranked

1. **`sc1` done properly** — mechanism proven (13.20 -> 3.84 us, ZERO stale words of
   104,595,456 against a control at 99.75% stale). Needs BOTH the local/peer scope split
   AND conversion of every ragged scalar activation tail. Load side is reachable via
   `__builtin_amdgcn_raw_buffer_load_b128` with **`aux |= 16`** (bit 4 = `sc1`).
2. **`Residual -> AttnRes` fusion** — 178 packets, elementwise producer into a `b=1`
   consumer, so zero duplicated work. The ONLY fusion the measured rule permits:
   *fusion that duplicates a reduction across N consumers costs (N-1) extra reductions.*
   `AttnRes` itself is NOT fusable (fan-out 3-4); folding a norm into its consumers was
   tried and measured 22.4 -> 24.4 ms/token.
3. **`lm_head` sharding** — machinery exists, gate passes, −14.7% GEMV bytes; needs an
   equivalence check that tolerates a `vocab/tp`-wide logits dump.
4. **HIER2** — 3.46 us vs 13.20, blocked on the global queue having no per-packet leader.
5. ~~**Prefill**: `wave_class` names only ops 11/39...~~ **RETRACTED — do not re-buy.**
   Every fact in the old entry is true and the conclusion was wrong. `wave_class`
   (`packet/src/devbuild.rs`) and `derive_segments` (`plowrt/src/exec/amd.rs`) do name only
   ops 11/39, K3 does emit op 110, and the flash object IS rejected on every K3 run
   (`object_name` forces `PrefillArm::None` for `Phase::Flash`, so `check_k3_arms` refuses
   it and `k_flash` becomes `None`). None of it costs anything: `enqueue_segment` changes
   only the kernel object and the block size — the grid is `n_cu` either way — and with no
   flash object a class-4 segment already falls back to the 8-wave path. Adding op 110 to
   `wave_class` alone is a no-op; adding it *and* building a K3 flash object is WORSE than
   a no-op, because the `PLOW_BUCKET_FLASH` body is one `if (op == FLASH_PREFILL)` with no
   switch and no default, so op 110 would silently do nothing while the interpreter still
   signalled its successors — fluent, wrong output. At TP8 more segments also cost an
   all-rank barrier each (`amd_tp.rs`). The 4-wave object exists for `d_flash_prefill`,
   the dense MFMA kernel; K3's MLA prefill never calls it.

   The real cost was the kernel. `d_flash_mla_prefill` was a wrapper over the DECODE body
   with `n_tok > 1`, so its work item was ONE query token: the causal latent prefix was
   re-streamed per token and every dot ran on the vector ALU. Prefill is not protocol-bound
   the way decode is — the T=8192 program's critical path is 2109 packets, ~12 ms/chunk at
   5.72 us, 0.2% of a 24 s TTFT. It is bodies.

   **Fixed**: `d_flash_mla_prefill_mfma` (op_attention.h) tiles the QUERY axis and runs the
   32x32x16 MFMA, staging the latent once per (q-tile, kv-tile) for all 64 query rows.
   Measured on gfx950, one MLA layer, fp8 latent (op 110), n_head=12 (K3 at TP8):

   | ctx | n_tok | scalar | tiled | |
   |---|---|---|---|---|
   | 8192 | 8192 | 20.5 ms | 9.1 ms | 2.25x |
   | 32768 | 8192 | 141.6 ms | 50.7 ms | 2.79x |

   The four 8192-row chunks of a 32k prompt go 324.7 ms -> 120.1 ms per layer, x24 MLA
   layers = **7.8 s -> 2.9 s per rank**. bf16 (op 51) gains more, 7.2-8.4x, because the
   scalar body it replaces is bandwidth-bound at twice the bytes.

   NOT fixed, and the next thing to buy: below ~0.375 machine fill the tiling has fewer
   work items than the scalar decomposition and LOSES (0.35x at n_tok=128, n_head=12).
   `mla_pf_tiled_fills` picks the better body so this is never a regression, but a short
   chunk still runs at the old speed. The fix is `nsplit` over the KV range, which is a
   PACKET change: Opart must be sized for nsplit>1 and `MlaMergeFold` told. Only bites at
   high TP — at TP1's n_head=96 the tiled kernel wins at every size measured (2.1-2.8x).

## 10. Republish the tuning DB before trusting any GEMM tile

ANY edit under `runtime/amd/*` moves the build digest and stales EVERY record at once;
`pick_tile` then silently reverts to the analytical model and reports tier `portable` —
which is exactly what it reports when nothing was ever measured.
`cargo test -p devgen --test tuned_tile_selection` is the signal (fails 2/4).
`scripts/rebench_tune_gemm_all.sh` is the fix, and it needs a quiet GPU.
