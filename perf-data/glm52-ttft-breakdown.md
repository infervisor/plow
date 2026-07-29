# GLM-5.2 TP4 TTFT, accounted end to end — the 1.2 s was never missing

**The question.** `perf-data/glm52-plow-vs-vllm-tp4.md` measures plow TTFT = 1367 ms against vLLM's
133 ms on 1024-in/128-out, concurrency 1. A sibling measurement of the AMD prefill engine reported
5,737 tok/s at 1024 tokens — i.e. ~178 ms of device prefill — which would leave **~1.2 s
unaccounted** between the HTTP request arriving and the first token leaving. This page instruments
the serve path end to end and looks for that interval.

**The answer: there is no unaccounted interval.** Every phase from handler entry to the first SSE
frame sums to the measured TTFT, and **99.8% of it is device wall inside two prefill launches**. The
entire host path — chat template, tokenizer, mux queue, chunk planning, instruction patching,
counter re-arm, AQL submission, detokenisation, SSE serialisation — is **4.2 ms**.

What the decomposition *does* turn up is a lever the record explicitly wrote off: **the padded
128-row tail chunk costs 260 ms to place 6 real tokens, 19% of TTFT.**

## The run

`vllm bench serve --backend openai-chat --dataset-name random --random-input-len 1024
--random-output-len 128 --max-concurrency 1 --num-prompts 32 --num-warmups 4` against a `plowrt`
endpoint (§0-BENCH). Measured 2026-07-28, 36 requests.

| GLM-5.2 TP4 block-fp8, 1024-in/128-out, conc 1 | this run | the record it reproduces |
|---|--:|--:|
| TTFT mean (ms) | **1373.21** | 1367.27 |
| TTFT median (ms) | **1370.03** | 1354.70 |
| TPOT mean (ms) | 30.06 | 28.59 |
| Output throughput (tok/s) | 24.7 | 25.6 |

Within 0.4% on TTFT against an independent measurement on different objects, which is what makes
the decomposition below a decomposition of the recorded number and not of a different one. TPOT is
5% higher; that is not investigated here and is not claimed as a new decode number.

## The breakdown

`PLOW_TTFT_LOG=1`, one table per request. Concurrency 1, so exactly one request is in flight from
arrival to first token; the phases partition `[handler entry, first SSE frame]`, which is the
interval `vllm bench serve`'s chat backend stamps as TTFT.

Means over the 36 requests, and one representative table beside them:

| phase | mean ms | one request | n | % of TTFT |
|---|--:|--:|--:|--:|
| chat template render | 0.025 | 0.026 | 1 | 0.0% |
| tokenize (HF BPE) | 2.95 | 1.322 | 1 | 0.2% |
| queue: submit -> prefill call | 0.044 | 0.049 | 1 | 0.0% |
| **prefill TOTAL (engine thread)** | **1368.0** | **1372.282** | 1 | **99.8%** |
| &nbsp;&nbsp;plan_chunks + chunk_steps | 0.006 | 0.011 | 1 | 0.0% |
| &nbsp;&nbsp;prefill_prepare (ids/pos/patch upload) | 0.848 | 0.832 | 8 | 0.1% |
| &nbsp;&nbsp;rearm_prog (counter zeroing) | 0.205 | 0.206 | 8 | 0.0% |
| &nbsp;&nbsp;zero_xctr (cross-GPU gates) | 0.074 | 0.074 | 2 | 0.0% |
| &nbsp;&nbsp;enqueue_segment (AQL launch) | 0.006 | 0.006 | 2 | 0.0% |
| &nbsp;&nbsp;**drain (device wall)** | **1366.6** | **1370.875** | 2 | **99.8%** |
| &nbsp;&nbsp;read_sampled (D2H of `in.ids`) | 0.032 | 0.033 | 1 | 0.0% |
| first token detok + channel send | 0.013 | 0.018 | 1 | 0.0% |
| UNACCOUNTED (HTTP, axum, SSE serialise) | 0.044 | 0.040 | | 0.0% |
| **TTFT** | **1371.1** | **1373.74** | | |

`prompt=1030 tok, cover=[1024, 128] = 1152 padded rows.` TTFT spread over the 36: min 1349.9,
median 1368.4, max 1421.2 — ±2.6%, no cold outlier.

The indented rows are children of `prefill TOTAL`; the top-level rows plus UNACCOUNTED sum to TTFT
by construction, and the residual is 0.044 ms.

## Every candidate, priced

| candidate | measured | verdict |
|---|--:|---|
| Tokenisation / chat-template rendering | 2.98 ms | 0.2%. HF `tokenizers` on a 1030-token prompt. |
| Bucket cover and padding | **260 ms** | **19% — written off as a rounding error; it is not. See below.** |
| Per-chunk fixed cost, HOST side | 1.13 ms over 8 rank-chunks | 0.14 ms per rank-chunk, not the CUDA path's 60.1 ms/launch |
| Per-chunk fixed cost, DEVICE side | **139 ms per launch** | **the biggest single lever. See below.** |
| Scheduler / mux tick | 0.044 ms | submit -> `prefill()`, including the formation hold |
| Weight/KV first-touch | 0 | `--num-warmups 4`; the 36 tables are flat to ±2.6% with no first-request outlier |
| The first decode step | **0** | **TTFT contains no decode step at all — see below** |
| **Device prefill** | **1366.6 ms** | **99.8%** |

### TTFT contains ZERO decode steps

Worth stating because the brief assumed otherwise. `AmdServe::prefill` returns the token the
**prefill program itself** sampled — the device argmax left in `in.ids`, read back by
`read_sampled` — and the first decode dispatch happens *after* the first token is already on the
wire. So GLM's ~29 ms/token decode contributes nothing to TTFT, not ~2%.

### The launch has a 139 ms floor, and the padded tail pays it in full

Per-chunk device wall, over 11 warm requests (`PF CHUNK` lines):

| chunk | bucket T | real rows | device wall (median) | tok/s over real rows |
|---|--:|--:|--:|--:|
| 1 | 1024 | 1024 | **1104.6 ms** | 927 |
| 2 (tail) | 128 | **6** | **259.7 ms** | **23** |

n=11 each, both tight: T=1024 spans 1088.1-1136.5, T=128 spans 256.0-266.3.

Two numbers fall straight out of two points on the ladder, `cost(T) = a + b·T`:

* **b = 0.943 ms per bucket row** — the marginal rate, i.e. an asymptotic 1,060 tok/s.
* **a = 139.0 ms fixed per launch** — paid whole whether the bucket carries 1024 tokens or 6.

So the 128-row tail chunk spends **259.7 ms to place 6 real tokens**: 139 ms of launch floor plus
121 ms computing 122 padded rows that write KV nothing reads. That is **19% of TTFT**, and it
contradicts `glm52-plow-vs-vllm-tp4.md`'s "a tighter bucket ladder ... is worth at most the padded
128 of the 2-chunk split ... a rounding error, not the fix." Correcting that claim, under the fitted
model:

| change | modelled TTFT | saving |
|---|--:|--:|
| today: `[1024, 128]` | 1364 ms | — |
| bound the tail kernel at `clen` rather than the bucket width | 1250 ms | **8%** |
| one bucket covering 1030 in a single launch (e.g. T=1152) | 1110 ms | **19%** |
| remove the 139 ms launch floor entirely, keep 2 chunks | 1086 ms | 20% |

The ladder itself is not the bug — `plan_chunks` already picks the cheapest cover under this cost
model (`[2048]` models at 2082 ms, `[1024,512]` at 1735 ms, both worse than 1364). What is wrong is
that **`LAUNCH_ROWS = 416` overprices a launch by 2.8x**: the measured floor is 139.0/0.943 =
**147 rows-equivalent**. That misprice does not change the choice at 1030 tokens, but it is the
DP's only knowledge of this cost and should be corrected before the ladder is tuned. (`a` and `b`
are a 2-point fit on one prompt length; confirm on T=512 and T=2048 before acting on the constant.)

None of this closes the gap to vLLM's 133 ms on its own — 19% off 1367 is still 8x slower. It is
worth having because it is cheap and because it was being written off.

### The TP per-segment host barrier fires TWICE, not ~200 times

`AmdTpGroup::prefill_chunk` runs per-segment across all ranks with a `drain()` after each segment,
and that shape looks expensive. It is not, because **every GLM prefill bucket compiles to exactly
ONE segment**: `derive_segments` marks a segment class-4 only for `FlashPrefill`/`FlashPrefillFp8`,
and GLM emits `FlashMlaPrefill` (op 51), so the whole program is one class-8 segment. Verified
host-side, no GPU, from the packet itself (`crates/plowrt/tests/glm_pf_shape.rs`):

```
packet /home/lava/models/glm52_ttft/model.pkt: 5 programs, tp=4
  prog 0: T=128    insts=2021  stream=377444  segments=1  (class-4: 0)
  prog 1: T=512    insts=2021  stream=377444  segments=1  (class-4: 0)
  prog 2: T=1024   insts=2021  stream=377444  segments=1  (class-4: 0)
  prog 3: T=2048   insts=2021  stream=377444  segments=1  (class-4: 0)
  prog 4: T=1      insts=2756  stream=259169  segments=1  (class-4: 0)
```

A 1030-token prompt therefore costs **2 launches per rank and 2 host barriers**, and
`enqueue_segment` measures 0.006 ms of submission for all 8 rank-launches. The barrier is not a cost
centre; the kernel behind it is. Run this test before proposing any segment-granularity fix — it
costs no GPU and it killed this hypothesis in one command.

## Reconciling with "5,737 tok/s at 1024 tokens"

The served rate is **1030 real tokens (1152 padded rows) in 1366.6 ms = 754 tok/s over the real
tokens, 843 tok/s over the padded cover**. That is the same ~750 tok/s
`glm52-plow-vs-vllm-tp4.md` already recorded, arrived at independently and now attributed to the
launch rather than inferred from the wall.

A 5,737 tok/s figure and this one cannot both describe the same 1024 tokens through this packet on
these four cards — they differ by 6.2x, against a per-chunk measurement whose spread is ±2%.
Whatever the sibling harness measured, it was not the arm `AmdTpGroup::prefill` dispatches on a TP4
GLM-5.2 serving packet. **The 178 ms number should not be used to size any TTFT headroom until it is
re-derived through `AmdTpGroup::prefill_chunk` with bound weights on four cards**, which is the only
arm the served token goes through. The most likely candidates for the discrepancy, in order: a
single rank rather than four; a block asset rather than the model; unbound weights.

## What this closes and what it opens

**Closed.** There is nothing to find on the host side of GLM's TTFT. Tokenisation, template
rendering, admission, the mux tick, chunk planning, instruction patching, counter re-arm, AQL
submission, detokenisation and SSE framing together are 4.2 ms of a 1371 ms TTFT — 0.3%. Any TTFT
work that does not change what the two prefill launches do is worth at most 0.3%.

**Open, in order of size:**

1. **The prefill kernel's marginal rate.** 0.943 ms/row = 1,060 tok/s asymptotic, against vLLM's
   ~7,700. This is the whole gap and it is entirely inside `d_flash_mla_prefill` plus the grouped
   block-fp8 MoE arms. §6g-WAVE4 already showed the register budget is not the constraint (MLA
   prefill uses 148 of 256 VGPRs, zero spill), so this is a throughput question about the bodies,
   not a dispatch or occupancy one.
2. **The 139 ms per-launch floor** — 10% of TTFT at one chunk, 20% at two. Unattributed: it is not
   host submission (0.006 ms), not counter re-arm (0.205 ms), not the barrier. It is inside the
   launch, and 2021 instructions over 78 layers put it at ~69 µs per op of fixed cost, which is the
   shape §7b describes for decode.
3. **The padded tail**, 121 ms of the 260 — bound the kernel at `clen`, or add a bucket that covers
   prompt+template in one launch.

## The instrument

`PLOW_TTFT_LOG=1`, off by default; every call site is behind a cached `OnceLock<bool>`, so a serving
build pays nothing. It is a **concurrency-1 instrument** — the accumulators are global and are reset
by the arriving request, which is unambiguous only while one request is in flight. Above
concurrency 1 the timeline needs threading through `Job` -> `Slot` -> `ServeEngine` ->
`AmdTpGroup` -> `AmdEngine`; until something needs it, six signatures on the decode critical path
are not worth a phase that happens once per request.

`crates/plowrt/src/obs/ttft.rs` holds the phases; the call sites are `serve/chat.rs` (template,
encode, and the dump on the first token frame), `serve/mux.rs` (queue, prefill total, first-token
handling) and `exec/amd_tp.rs` + `exec/amd.rs` (the prefill internals, plus the per-chunk
`PF CHUNK` line).

**Note on `PLOW_PF_PACKLOG`.** The `packlog` module in `serve/mux.rs` is CUDA-only and dead in an
`hsa` build — `record()` is called only from the `gpu_prefill_*` path, so `PREFILL_NS`, `DECODE_NS`
and `PREFILL_TICKS` compile with `never used` warnings and would have measured nothing here. It is
still dead; this instrument does not replace it, it answers a different question.

**The SSE role-frame artefact was NOT reintroduced.** The dump fires on the first chunk that carries
a `role`, which is the first chunk carrying a real token — the same frame `vllm bench serve` stamps.
The reported TTFT is therefore the client's definition, and it agrees with the client's own mean to
0.2%.

## Reproduce

```sh
# objects — OUTSIDE nix (nix breaks system ROCm: GLIBC_2.38). MLA + MoE prefill arms.
PLOW_MLA_PREFILL=1 PLOW_MOE_PREFILL=1 \
  /usr/bin/env -i PATH=/opt/rocm/bin:/usr/bin:/bin HOME="$HOME" \
  PLOW_MLA_PREFILL=1 PLOW_MOE_PREFILL=1 bash scripts/build_gfx950.sh <objdir>

# packet — INSIDE nix.
nix develop -c bash scripts/rebench_emit_glm.sh <bundle>/model.pkt

# bundle: model.pkt + hsaco -> <objdir>, checkpoint -> GLM-5.2-plow, tokenizer.json, weights.json

# the run. sg render (this login predates the render gid); ttft_run.sh unsets
# HIP_/CUDA_VISIBLE_DEVICES, which COMPOSE with gpulease's ROCR ids.
GPU_LEASE_TIMEOUT=7200 perf-data/harness/gpulease -n 4 ttft \
  sg render -c "bash $PWD/scripts/ttft_run.sh"
```

`scripts/ttft_run.sh` is `scripts/bench_plowrt_serve.sh` with `PLOW_TTFT_LOG=1`, `NPROMPT=32` and
`--num-warmups 4` — the §0-BENCH-legal configuration, unchanged apart from the env flag.
`crates/plowrt/tests/glm_pf_shape.rs` reports the packet's buckets, chunk cover and segment counts
with no GPU: `PLOW_PF_PKT=<pkt> cargo test --features hsa --test glm_pf_shape -- --nocapture`.

## `gpulease` returned rc=76 on this run, and it is a FALSE POSITIVE

Both runs of this measurement returned `rc=76` (contended), and neither was. `gpulease`'s own
comment says why: *"on this ROCm build `amd-smi process --json` reports the SAME process list under
EVERY gpu entry, so its per-GPU attribution is fiction."* Its mitigation is to intersect that
fiction with per-card VRAM — but **the intersection fails exactly when the leaseholder's own model
is resident**, because then every leased card is "busy" and every foreign pid on the box is
attributed to it.

The disproof is arithmetic. The audit claims pid 3308415 held 203.4 GB on **each** of cards 4-7.
Live `rocm-smi --showmeminfo vram --csv` during the run:

```
card2,309220868096,143171485696     <- the other agent, loading
card3,309220868096,203588468736     <- the other agent
card4..7,309220868096,~199000000000 <- 199.0 GB each: my four TP ranks, and nothing else
```

199 + 203 = 402 GB on a 309 GB card. The foreign process was on cards 0-3, holding a lease for
0-3, exactly where it belonged. Corroborating: `rocm-smi --showpids` gave my `plowrt` CU occupancy
11 and the "contender" 0 (it was loading, not computing); the two runs' TTFT means agree to 0.1%
(1374.79 / 1373.21) and both agree with the independent record to 0.4%; and the per-chunk
distributions span ±2%.

**This makes `rc=76` unusable for any serving benchmark**, since a serving benchmark by definition
leaves a large model resident on its own cards. §5's "re-run, do not report it" cannot be followed
if the signal fires unconditionally. Either `busy_gpus` must exclude cards whose VRAM is accounted
for by the leaseholder's own processes, or the AMD audit should fall back to the one signal that is
sound here: **a foreign process whose claimed VRAM plus the card's own total exceeds the card's
capacity is not on that card.** Filed here rather than fixed because it is the harness, not this
measurement, and other agents' rc values are affected by it too.

## Provenance and caveats

* Objects `/home/lava/plow/build-amd/ttft-objs`, built from this tree at ABI 144
  (`sizeof(PlowProgram)`, grown by 4e5937b at 12:18). `build-amd/rebench3-objs` — the objects the
  1367 ms record was taken with — are ABI 136 (built 11:49) and are refused by name by
  `AmdEngine::load` against this tree. The packet was re-emitted to match; bundle
  `/home/lava/models/glm52_ttft`.
* **The tile campaign was NOT re-run.** `plowc` reports `900 record(s) skipped as STALE against the
  probed build gfx950-14811518192412b8`, so tile selection fell back to the analytical model.
  `glm52-plow-vs-vllm-tp4.md` states that for GLM's own narrow shapes the campaign confirms the
  analytical model rather than correcting it, and the TTFT here agrees with that record to 0.4%,
  which is the evidence that the fallback did not move the number. Do not reuse that reasoning for
  a shape set where the campaign is known to correct the model.
* The correctness gate passed on this bundle before any timing: `What is the capital of France?`
  -> `The capital of France is Paris.`
* `--max-ctx 4096`, greedy only (the device samples and the host never sees the logit row), TP4,
  concurrency 1. Everything §0-BENCH-legal about the source record applies unchanged.
