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
| `PLOW_K3_FUSE_NGEMV` | **on** | fold the `b=1` RMSNORM gates into their GEMV consumers (`routed_expert_norm` x92, `q_a_layernorm` x24). Removes 116 packets and moves critical path 1831 -> 1715. Current TP8 full-logit A/B is byte-exact; two served pairs improve TPOT 1.33--1.45 ms. `=0` restores the control. `perf-data/k3-narrow-gate-fusion.md` |
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

---

# 11. SERVING BRING-UP — batched decode, chunked prefill, interleave, prefix cache

Everything in §0-§10 describes K3 as a **single-sequence decode engine** driven by `amd-bench`.
This section is the serving path: what `plowrt serve` now does, how to turn each piece on, and what
each one measured. Branch `k3-batched-decode`.

## 11.0 The shape of it, in one table

| capability | how it is turned on | default | what it bought (MEASURED) |
|---|---|---|---|
| **batched decode** | `PLOW_DECODE_BATCH=B` at EMIT (hsaco must match) | B=1 | 91.3 tok/s aggregate at B=16 vs 34.5 at B=1 |
| **prefill + decode interleave** | on; `PLOW_PF_NO_INTERLEAVE=1` disables | on | TTFT median **3.0x** |
| **chunked prefill** | on; `PLOW_PF_NO_CHUNK=1` disables | on | ITL p99 **-26%**, TTFT median 1.34x, throughput -4.9% |
| **prefix cache** | `PLOW_PREFIX_CACHE=1` | off | 20% less wall at full hit; TTFT median 1.92x at 75% hit |
| **per-row parked mask** | automatic on a batched blob | on | correctness: idle rows stop advancing their recurrence |

Composed (chunked prefill + prefix cache), GSM8K B=4/CONC=4: **758 s -> 561 s, 26% less wall**, at
195/200 = 0.9750 with zero request errors.

## 11.1 Emitting a batched blob

`PLOW_DECODE_BATCH=B` makes the DECODE program carry **B independent sequences**, which is a
different thing from a prefill bucket's `t` rows — see `k3-batched-decode-design.md` §1 for why the
distinction is the whole problem. B is capped at 16 (`PLOW_GEMV_MAXM`).

```bash
nix develop --command bash -c "K3_FULL=1 PLOW_FP8_KV=1 PLOW_MXFP4=1 PLOW_DECODE_BATCH=4 \
  ./target/release/plowc --hf-dir /home/lava/models/k3_farm --emit devblob --arch gfx950 \
  --gpu mi350 --num-gpus 8 --parallel tp --max-ctx 32768 --n-cu 256 --out /home/lava/models/k3_b4"
```

Expect `2942 decode instructions` and `2916 tensors` at B>1 (against 2459 / 5411 at B=1 — the
decode program is a different program, not the same one with a bigger `t`).

**THE HSACO MUST MATCH THE BLOB.** `PLOW_DECODE_BATCH` sizes `PLOW_GEMV_MM` in the kernels as well
as `t` in the packet, so a B=4 blob on a B=1 object is a wrong-answer configuration, not a slow one:

```bash
cmake -S runtime -B ba_b4 -DPLOW_DECODE_BATCH=4 ...   # then
nix develop --command env -u LD_LIBRARY_PATH cmake --build ba_b4 -j 16
```

`B=1 is byte-identical to the pre-batch blob` — md5 `7db2fbb34230050f0508a4e706523a98`. That
invariant is checked at every step of this work and is the fastest way to tell whether an emitter
change leaked into the shipped configuration.

## 11.2 Serving

```bash
perf-data/tools/gpulease -n 8 serve sg render -c \
  "PLOW_L2_PLACE_DISPATCH=1 nix develop --command ./target/release/plowrt serve \
   --assets /home/lava/models/k3_b4 --port 8000"
```

`sg render -c` is load-bearing, not decoration: `/dev/kfd` is `root:render 0660`, and a shell
without the `render` group gets `hsa_init failed: 4104` and **silently falls back to the CPU
backend**. The harnesses' coherence gate is what catches it.

## 11.3 What a serving tick does now

The AMD tick used to be **prefill XOR decode** — it ran one whole prompt and `return`ed, so every
live decode stream stalled for the duration of someone else's prefill. It now does both, in this
order:

1. **One prefill CHUNK** for the lowest pending slot (not the whole prompt).
2. **One decode dispatch** advancing every live slot.

Prefill runs first so the new slot's first token is already in `out_ids` when the decode's feed list
is built, and it decodes in the same tick rather than waiting for the next one.

Three invariants hold it together, and they are the interesting part:

* **`prefill_slot` restores KV base 0 before returning**, and `decode_step_batched` refuses a
  non-zero base. The decode cannot run rebased onto a slot.
* **A mid-prefill slot is PARKED.** A decode dispatch advances all B rows because `t` is compiled,
  which is harmless for an append-only KV cache and fatal for the KDA recurrence. `in.parked`
  (non-zero = skip) is published as `!live[s]`, so a half-prefilled slot's recurrence does not move.
  The sense is "parked" rather than "active" so that an all-zero or never-written mask means *every
  row participates* — `amd-bench` never publishes one.
* **A mid-prefill slot is fed `pos = frontier`**, the row its next chunk overwrites anyway, so the
  dispatch's KV write cannot clobber the prefix already built.

## 11.4 Chunked prefill

A prompt is covered by the compiled bucket ladder `[128, 512, 1024, 2048, 4096, 8192]` via a
cost-minimising DP that trades padding against launch count (`plan_chunks`). A 1038-token prompt
plans as **2 chunks**, not 1 — check the server log, which prints
`TP prefill plan tokens=1038 chunks=2`.

The mux advances **one chunk per tick**. That is a TAIL-LATENCY feature and should be quoted as
one: it buys a decode stream the right to wait for one chunk instead of a whole prompt, and pays
for it in extra dispatches and less efficient tiles.

| B=16, `IN_LENS=1024 CONCS=16 NPROMPT=64` | TTFT med | ITL p99 | out tok/s |
|---|--:|--:|--:|
| whole-prompt prefill | 2435.0 ms | 1530.3 | 41.1 |
| chunked prefill | **1818.4 ms** | **1132.2** | 39.1 |

Falls back to whole-prompt where there is no ladder to walk: single-GPU, `decode_only`, or a
1-token prompt.

## 11.5 Prefix cache

`PLOW_PREFIX_CACHE=1`. Per slot, keep the last prompt plus a checkpoint of that slot's **carried
recurrent state**; a later prompt agreeing over that span restores the recurrence and prefills only
the suffix. The KV rows are already the slot's own.

The design and the general problem — why prefix caching, paged attention and speculative decoding
all assume positional state, and what changes when 69 of 93 layers keep a *folded* state instead —
are written up separately in **`perf-data/k3-prefix-cache-design.md`**. Read that before changing
anything here.

Operationally: 56 MiB per slot per rank, allocated lazily on first arm (a workload with no shared
prefixes never allocates), ~3.7 ms to snapshot or restore, about **1% of wall**
(`PLOW_PFX_LOG=1`). Composes with chunking.

## 11.6 Correctness gates for this path

Run these before believing any serving number:

```bash
# 1. Semantic gate: does batching contaminate slots?
./scripts/k3_batch_gate.sh /home/lava/models/k3_b4 <hsaco> /home/lava/models/k3_farm 4 \
                           /home/lava/models/k3_b16 <hsaco16>
#    check A: B copies of one prompt -> B identical streams
#    check B: B ragged prompts -> same per-slot streams at a SECOND batch width
#    NOTE it compares two BATCHED widths, never batched-vs-solo: a batched decode routes MoE
#    through the GROUPED expert kernel and B=1 through the per-slot one, so they accumulate in
#    different orders and greedy decoding turns any tie into a different token a few steps later.

# 2. Accuracy gate: 8-shot GSM8K end to end
N=200 CONC=4 ./scripts/bench_gsm8k.sh /home/lava/models/k3_b4 8000 auto 1800
```

Expect **~97-98%**. Anything materially below that is a per-sequence state bug, not sampling noise —
which is exactly how the missing `begin_slot` on the TP path was found (81.0% -> 98.0% when fixed).

## 11.7 Two failure modes that look like something else

* **`cargo test --workspace` silently strips `--features hsa`.** The next GPU run then reports
  `hsa=false` and the gate fails in a way indistinguishable from a correctness break. Always
  `cargo build --release -p plowrt --features hsa` after a workspace test. Hit twice.
* **A 500 from `serve` at B>1 kills every in-flight request at once**, because they share the decode
  step. A burst of exactly B failures is ONE event. The known open one is a rare cross-rank
  divergence in `d_xargmax_fin_mega` (~1 request in 200) — `k3-batched-decode-design.md` §9.

---

# 12. LONG CONTEXT — 250K tokens, measured end to end

`--max-ctx 262144` at emit. **The whole model runs**: prefill, all 93 layers, decode, and a
coherent answer that demonstrably read the prompt.

## 12.1 What was measured

| packet | prompt tokens | wall | prefill tok/s | output |
|---|--:|--:|--:|---|
| B=1, max_ctx 256K | **200,019** | 227.5 s | 879 | coherent, context-aware |
| B=1, max_ctx 256K | **250,014** | 329.1 s | 760 | coherent, context-aware |
| **B=4**, max_ctx 256K | **250,014** | 325.2 s | 769 | coherent, context-aware |

The completion is the evidence that this is a real long-context run and not an allocation test.
Fed ~1.2 MB of `"the quick brown fox jumps over the lazy dog"`, K3 answers:

```
I see you've shared the classic pangram "the quick brown fox jumps over the
```

It identified the content of a 250,000-token prompt. Driven through `plowrt serve`, so this is the
serving path with chunked prefill, not a bench harness.

## 12.2 Emitting and running it

```bash
nix develop --command bash -c "K3_FULL=1 PLOW_FP8_KV=1 PLOW_MXFP4=1 ./target/release/plowc \
  --hf-dir /home/lava/models/k3_farm --emit devblob --arch gfx950 --gpu mi350 \
  --num-gpus 8 --parallel tp --max-ctx 262144 --n-cu 256 --out /home/lava/models/k3_256k"
```

Nothing else changes. The bucket ladder is unaffected (`[128, 512, 1024, 2048, 4096, 8192]` at both
32K and 256K), so a long prompt is simply more chunks of the same widths — a 250K prompt is ~31
chunks of 8192.

## 12.3 The memory arithmetic, and why K3 is a good long-context citizen

DERIVED from the MEASURED 442 MB of MLA KV per sequence per rank at ctx 32000:

| | MLA KV (24 layers) | KDA state (69 layers) | total / rank |
|---|--:|--:|--:|
| ctx 32K, B=1 | 0.45 GB | 0.44 GB | 0.89 GB |
| ctx 256K, B=1 | 3.62 GB | **0.44 GB** | 4.06 GB |
| ctx 256K, B=4 | 14.48 GB | **1.76 GB** | 16.24 GB |

Against 191.2 GiB of weights on a 288 GB card, so **256K at B=4 uses ~207 of 288 GB and fits with
room**.

The interesting column is the middle one. **The KDA state does not grow with context at all** —
69 of 93 layers cost a flat 0.44 GB per sequence whether the context is 5 tokens or 250,000. Only
the 24 MLA layers scale. That is the architectural payoff of linear attention showing up exactly
where it was supposed to: a dense 93-layer model at this context would pay the 3.62 GB figure
almost four times over.

It is also why the batch axis and the context axis trade so differently here: batch multiplies
BOTH terms, context multiplies only one.

## 12.4 Correctness

The 256K blob is **token-identical to the 32K blob** on the same prompt at the same effective
context — `1008,10484,318,15383,387` continues to
`[13, 646, 12259, 387, 14868, 220, 5807, 6017, 1873, 13, 646, 7695, 5793, 387, 12516, 13]` under
both, all 8 ranks agreeing. `max_ctx` changes tensor extents and `out_stride`, not arithmetic.

## 12.5 THE COST, which is real and is NOT the context you use

A bigger `max_ctx` costs decode throughput even when the context is empty:

| max_ctx | ms/token at an effective ctx of 5 | tok/s |
|---|--:|--:|
| 32,768 | 33.522 | 29.8 |
| **262,144** | **42.575** | **23.5** |

**+27% per token for a context neither run used.** Same hsaco, same prompt, same everything except
the compiled extent. The KV tensors are 8x larger and `out_stride` is `ctx`, so every KV write is
spread across a much larger address range — a TLB and cache-locality cost, not an arithmetic one.

**So do not emit at 256K by default.** Emit at the context you serve. If you need both, emit both
blobs; they are 211 MB each and the emit takes 3.4 s.

This is the largest un-chased item this section leaves behind: nothing here has tried to make the
long-context blob's decode as fast as the short one's, and the 27% is a measurement rather than a
diagnosis.
