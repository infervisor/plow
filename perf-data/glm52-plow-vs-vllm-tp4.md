# GLM-5.2 block-fp8, TP4, MI355X (gfx950) — plow vs vLLM, §0-BENCH-legal

> ## ⚠️ TWO CLAIMS ON THIS PAGE ARE SUPERSEDED — read before quoting anything here
>
> Both retractions were measured on this same branch, after this page was written, and neither
> was linked from here until now.
>
> **1. The headline margin is NOT like-for-like.** This page publishes 1.13x / 1.18x slower with
> a five-item "Disclosures — read these before quoting the margin" list that omits the largest
> disclosure. `perf-data/glm52-precision-symmetry.md` opens with: *"**Answer: no.** plow ran
> **58.3 % of its decode weight stream in bf16 that is fp8 in the checkpoint file**"* — i.e. the
> two engines were not running the same precision, and vLLM was advantaged. That document names
> this file by path in its §0. **Do not quote the margin without reading it.**
>
> **2. `GLM_LINEAR_FP8` is not a regression.** The "+0.39 ms" figure below (§ "NOT set,
> deliberately") is superseded by `perf-data/glm52-linear-fp8-reeval.md`, which carries its own
> "§4 IS SUPERSEDED (2026-07-29)" banner and re-measures the knob on a stacked blob at
> **−0.417 ± 0.175 ms, n=6 — 97 % of the −0.431 ms floor.** Commit `b3f77fd` records the same.

**Instrument, both engines:** `vllm bench serve --backend openai-chat`, the SAME client binary
against two different base-urls. `--dataset-name random --random-input-len 1024
--random-output-len 128 --max-concurrency 1 --num-prompts 32 --num-warmups 4`. Under
`perf-data/harness/gpulease -n 4`, `rc=0` on every run quoted here (no contention).
Measured 2026-07-28.

Nothing on this page comes from `plowrt amd-bench`, `glm52_decode`, or any other bespoke timing
binary. Those instruments are legitimate for bring-up and correctness and their numbers must never
appear beside a vLLM number (per the design notes §0-BENCH).

**Provenance — the tree these numbers describe.** plow side: the merge commit that carries
`73a3ec2` plus this branch's `tp:`/`bench:`/`tuning:` commits. Objects
`/home/lava/plow/build-amd/rebench3-objs`, bundle `/home/lava/models/glm52_stack3`, tile store
keyed to `gfx950-f9c85e0acd64d50d` and verified live by `cargo test -p devgen --test
tuned_tile_selection` (4/4). `worktree-readme-build-instructions` moved again during the run
(L2 placement, per-(CU, segment) stream windows — both touch `interp.hip`/`dev_isa.h`), so these
numbers are a snapshot, not a claim about the current tip. **Re-derive the build digest before
reusing any of these objects; see §6g-STALE-2.**

## Result

| GLM-5.2 TP4 block-fp8, 1024-in / 128-out, conc 1, n=32 + 4 warm-ups | plow | vLLM | |
|---|--:|--:|:--|
| **TPOT** mean (ms) | 28.59 | **25.35** | plow **1.13x slower** |
| **TPOT** median (ms) | 28.28 | **23.89** | plow **1.18x slower** |
| TPOT P99 (ms) | **29.53** | 35.27 | plow's tail is tighter |
| **TTFT** mean (ms) | 1367.27 | **133.28** | plow **10.3x slower** |
| **TTFT** median (ms) | 1354.70 | **133.83** | plow **10.1x slower** |
| Output throughput (tok/s) | 25.6 | **38.2** | plow 0.67x |

**plow loses on both metrics.** The margin narrowed a lot and the direction did not change.

### Against the record this replaces

| | plow before | vLLM before | plow now | vLLM now | gap before -> now |
|---|--:|--:|--:|--:|---|
| TPOT (ms) | 36.97 | 22.48 | **28.59** | 25.35 | 1.64x -> **1.13x** |
| TTFT (ms) | 37,900 | 1,880 | **1,367** | 133 | 20x -> **10.3x** |

Read the vLLM columns before celebrating the plow ones. **plow's TTFT improved 27x and the GAP only
halved**, because the old vLLM TTFT of 1880 ms was itself inflated ~14x by an unwarmed cold
request. The same correction runs the other way on TPOT: vLLM's honest warm TPOT is 25.35 ms, not
22.48, so part of that gap closing is plow and part is the instrument.

**The single largest correction on this page is to the vLLM column, not the plow one.** A prior
claim of "plow 1.64x slower on TPOT, 20x on TTFT" and the claim here of "1.13x and 10.3x" differ
by more than the engineering that landed in between.

### The remaining TTFT gap is prefill THROUGHPUT, not dispatch count

The old blob's 37.9 s was 1024 decode dispatches for a 1024-token prompt, and emitting prefill
programs removed exactly that. What is left is a straight rate comparison: plow prefills 1030
tokens in ~1.37 s (**~750 tok/s**), vLLM does 1024 in ~0.133 s (**~7,700 tok/s**). That is a 10x
kernel/schedule gap in the prefill path itself, and it is now the single biggest item on the GLM
board. It will not be closed by a tighter bucket ladder — that is worth at most the padded 128 of
the 2-chunk split.

### The `gemv_rows_fp8_blk` fix (73a3ec2) did NOT move the served token

Measured because the branch tip landed it mid-run. Two uncontended runs, same client, same flags,
same knobs, same bundle recipe, differing only in whether `73a3ec2` is in the objects:

| plow GLM-5.2 TP4 | TPOT mean | TPOT med | TTFT mean |
|---|--:|--:|--:|
| without 73a3ec2 (`rebench2-objs`) | 28.530 | 28.310 | 1373.06 |
| with 73a3ec2 (`rebench3-objs`) | 28.590 | 28.280 | 1367.27 |

**0.2% — noise.** The isolated kernel genuinely went 1.8-2.0x (o_proj 4245 -> 6899 GB/s, 68.5% ->
111.3% of the 6200 GB/s ceiling) and the served token did not notice. This is §7b's pathology
stated from the other side: GLM decode is not limited by the throughput of the block-fp8 GEMV, so
making that GEMV faster buys nothing at the token. An isolated-kernel win is a hypothesis about the
token, and this one was falsified. Both objects were built from their own source tree with their
own freshly re-run tile campaign, so this is not a stale-store artefact.

## What changed on the plow side since the 36.97 ms / 37.9 s record

All device-side, all landed before this run:

| change | effect | gate |
|---|---|---|
| MLA merge-fold rewritten wave-cooperative | 34.4 -> 28.2 ms/token | B4 oracle gate PASSED, `attn_out` 0.00762 identical to baseline |
| `GLM_SHARD_HEAD=1` (vocab-parallel lm_head) | −0.26 ms | bit-identical tokens; `XArgmaxFin` went stub -> implemented |
| `GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48` | −0.81 ms | bit-identical tokens |
| **GLM prefill programs emitted** (`PLOW_MLA_PREFILL=full:128,512,1024,2048`) | **the whole TTFT gap** | oracle gate PASSED, per-stage residuals BETTER than decode |

The TTFT lever is the structural one and nothing else on that axis changed. The old blob was
decode-only (`n_prog=1`), so a 1024-token prompt cost **1024 decode dispatches**; it now costs
**2 prefill chunks** (1030 tokens = the 1024 bucket + a padded 128).

NOT set, deliberately (**the `GLM_LINEAR_FP8` half of this sentence is SUPERSEDED — see the
banner at the top and `perf-data/glm52-linear-fp8-reeval.md`; re-measured at −0.417 ms, a WIN.
The `GLM_GROUP` half still stands**): `GLM_LINEAR_FP8` (measured **+0.39 ms**, a regression — `GemvFp8Blk` runs
at 966 GB/s where bf16 `Gemv` on the same shapes runs at 1728, so halving the bytes through a
kernel 1.8x slower per byte is break-even by construction) and `GLM_GROUP` (§6g-KNOBS, **+2.88 ms**
— it removes 38% of the ops and is slower, because collapsing the per-expert packets destroys the
concurrency `GLM_MOE_CORESIDENT` buys).

## Reproduce

```sh
# 1. tile campaign — MUST be re-run whenever interp.hip / op_gemm.h / op_moe.h changes,
#    or pick_tile silently falls back to the analytical model. Outside nix, under a lease.
scripts/rebench_tune_gemm.sh <objdir> /tmp/sweep.jsonl
tunedb-gemm ingest --db tuning --samples /tmp/sweep.jsonl   # x3 passes
cargo test -p devgen --test tuned_tile_selection            # the gate — 4 tests, all must pass

# 2. objects, BOTH prefill arms. Outside nix (nix breaks system ROCm: GLIBC_2.38).
PLOW_MLA_PREFILL=1 PLOW_MOE_PREFILL=1 scripts/build_gfx950.sh <objdir>

# 3. the stacked blob. Inside nix.
scripts/rebench_emit_glm.sh <bundle>/model.pkt

# 4. CORRECTNESS BEFORE TIMING. Reads the text out of three buckets.
scripts/rebench_glm_coherence.sh <bundle> <port> glm-5.2

# 5. the numbers.
NPROMPT=32 BENCH_EXTRA_ARGS="--num-warmups 4" \
  scripts/bench_plowrt_serve.sh <bundle> <port> glm-5.2 zai-org/GLM-5.2-FP8
NPROMPT=32 BENCH_EXTRA_ARGS="--num-warmups 4" EXTRA_ENV="VLLM_ROCM_USE_AITER=1" \
  scripts/bench_vllm_chat.sh zai-org/GLM-5.2-FP8 4
```

Build binaries must be rebuilt with `--features hsa`; a default `cargo build` gives a plowrt that
selects the **CPU reference backend** and serves fluent-looking garbage through a byte-fallback
tokenizer. It says so in the log (`hsa=false hf_tokenizer=false` + a boxed WARN) and it is easy to
miss.

## Correctness gate (run before the timings, same bundle, same objects)

The stacked blob had never been run, and a sharded lm_head on a PREFILL bucket had never been run
at TP4 at all — that failure shows as WRONG TOKENS, not a crash.

| prompt | bucket | output |
|---|---|---|
| "What is the capital of France?" | T=128 | `The capital of France is Paris.` |
| "Compute 17 * 23 and state the result in words." | T=128 | `17 * 23 = 391\n\nIn words, the result is three hundred ninety-one.` |
| ~2116 tokens of filler + "ignoring the above, capital of Japan?" | T=1024 + T=2048, 2 chunks | `The capital of Japan is Tokyo.` |

The streaming check also passes: the first SSE chunk carries `{"role":"assistant","content":"The"}`
— a real token, not a bare role frame. That artefact (removed in 63f9957) stamped `vllm bench
serve`'s TTFT at request arrival and flattered every plow TTFT taken before it.

## Disclosures — read these before quoting the margin

1. **vLLM's GLM run has no tuned AITER config for gfx950**, and says so per shape:
   `[aiter] shape is M:8192, N:2624, K:6144, not found tuned config in
   /tmp/aiter_configs/a8w8_blockscale_tuned_gemm.csv, will use default config!`
   This is an untuned floor for vLLM on this shape set, not vLLM's ceiling. plow's own tile
   inventory, by contrast, WAS re-tuned for this run (`tuning/`, 270 fresh records) — though for
   GLM's own narrow shapes the campaign confirms the analytical model rather than correcting it,
   so the asymmetry is smaller than it looks.
2. **vLLM runs SPARSE attention where plow runs DENSE.** The container selects
   `ROCM_AITER_MLA_SPARSE` (`rocm.py:592`); plow has no sparse path and computes the full
   attention. plow is doing strictly more work per token.
3. **vLLM serves with `enable_prefix_caching=True`; plow has no prefix cache at all.** On a random
   dataset this matters little, but it is not symmetric.
4. **vLLM pays a large one-time AITER JIT.** It builds into the container's `site-packages`, not
   into the mounted `/root/.cache`, so it is paid on every container start. `--num-warmups 4`
   keeps it out of the measured window on both sides; without warm-ups it lands in whichever
   metric the first request happens to stress and moves the mean by up to 3.3x (see below).
5. **Greedy only, no token-identity claim.** plow's device samples the argmax and the host never
   sees the logit row, so temperature/top_p/top_k are ignored; `vllm bench` 0.23 does not send
   `temperature=0`. These runs support a LATENCY comparison, not an output-identity one.

## Why `--num-warmups 4` and `--num-prompts 32`

Two runs of the IDENTICAL vLLM config, same image, same client, hours apart, both at `NPROMPT=8`
with no warm-up:

| vLLM run | Mean TTFT | Median TTFT | P99 TTFT | Mean TPOT | Median TPOT | out tok/s |
|---|--:|--:|--:|--:|--:|--:|
| 02:38 | 1880.55 | — | — | 22.48 | — | 27.03 |
| 10:06 | 573.00 | 137.20 | 3156.63 | 31.67 | 27.74 | 27.85 |

3.3x apart on mean TTFT and 1.4x on mean TPOT while whole-run throughput agrees to 3% — one cold
request owning the mean of eight. **The 22.48 ms / 1880 ms pair that the previous record compared
against was one draw from that distribution, not vLLM's performance.** The fix is on the client, so
it applies to both engines identically.

plow does not have this problem — no JIT, no cudagraph capture, no prefix cache, so no cold
request — and its mean and median agree to within 1% on both metrics either way. The warm-up is
there for vLLM's sake, and it costs plow nothing to accept it.

## What this run does NOT establish

* **Concurrency > 1.** `AmdTpGroup::submit_decode` is scalar and `AmdServe::load` refuses a
  `PLOW_DECODE_BATCH > 1` TP packet outright, so GLM is structurally batch-1 at TP4. A conc-4 GLM
  curve would measure QUEUEING and must not be tabled beside a vLLM batching number.
* **Long context.** `--max-ctx 4096`. §0-MARGIN's argument is that plow's advantages compound at
  long context x high concurrency, and this run is the *hardest* case for plow on both axes.
* **Token identity** — see disclosure 5.
* **A tighter bucket ladder.** 1030 tokens still costs 2 chunks on `128/512/1024/2048`. A bucket
  covering prompt+template in one chunk is unmeasured TTFT headroom — but see above: at a 10x
  prefill-rate gap it is a rounding error, not the fix.
* **vLLM's ceiling.** Every vLLM number here is from an AITER build with no tuned gfx950 config.

## Verdict

plow does not beat vLLM on GLM-5.2 at 1024-in / 128-out / concurrency 1. It is **1.13x slower per
output token** and **10.3x slower to first token**, and it produces 0.67x the output throughput.

What is genuinely won: TPOT closed from 1.64x to 1.13x, plow's TPOT P99 (29.53 ms) is now *better*
than vLLM's (35.27 ms) — a flatter distribution, which is the shape §0-MARGIN predicts from
one dispatch per token — and plow does this while running DENSE attention against vLLM's sparse
one, with no prefix cache.

What is not: the headline. §0-MARGIN asks for a decisive margin and this is a loss on both
headline metrics. The next lever is the prefill rate, not decode; and the configuration where
plow's advantages are supposed to compound (long context x high concurrency) is still structurally
unavailable at TP4, because `AmdTpGroup::submit_decode` is scalar.
