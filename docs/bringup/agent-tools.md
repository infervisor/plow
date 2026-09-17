# Agent tools: the scripts to use, and the environment they run in

`scripts/` holds 251 scripts. Most campaigns have nonetheless written their own throwaway probe,
rediscovered the same environment failures, and burned leased GPU time doing it. This file is the
tool surface an agent should reach for FIRST, so that does not keep happening.

**The rule: look here before writing a probe. If a tool below does the job, use it. If it almost
does the job, extend it and keep the extension.** A new one-off script in a scratch directory is
the last resort, not the first move.

---

## 0. Environment contract

**Everything runs inside `nix develop`.** The flake is self-contained and pins ROCm 7.14 (TheRock)
and CUDA 12.9. Build, emit, serve and bench all assume it:

```bash
nix develop --command <script> <args>
```

`ROCM_PATH` being unset is the signal you are outside it; every tool below refuses in that case.

**What does NOT come from nix**, and must be pointed at explicitly:

| dependency | where | how it is passed |
|---|---|---|
| vLLM client (`vllm bench serve`, and the vLLM reference server) | `/app/plow/build-gemma31/vllm-python` — a prebuilt venv/launcher, from source, not nix | `PB_VLLM`, or the path baked into the bench scripts |
| ROCm runtime the vLLM client links | `/opt/rocm/core-7.14/lib` — the LAB ROCm, not nix's | `VLLM_ROCM_LIB=...` in the client's env, always |
| `gpulease` | `/app/plow/perf-data/tools/gpulease` — **not on PATH** | absolute path |
| model checkpoints | `/workspace/models/...` | `PLOW_CKPT`, `GLM_RAW` |

The vLLM client is the same binary used against both servers on purpose: same client, same metric
definitions, different `--base-url`. That symmetry is what makes a plow-vs-vLLM number comparable,
so do not swap in a different client for one side.

**Every GPU process goes through the queue**, never a raw `gpulease` and never a bare run.

---

## 1. Check the environment before leasing anything

```bash
nix develop --command scripts/bench/plowbench-doctor.sh <assets-dir> <object-dir> [plowrt]
```

CPU only; leases nothing. Exit 0 = safe to lease, 1 = something will fail after the weights load,
2 = warnings only. It checks, in order: the nix shell; hazardous `PLOW_*` overrides left in the
environment; the binaries; the packet hash and the object set (including the pinned vendor `.co`
kernels that `build_gfx942.sh` does **not** emit); `gpulease` and the queue runner; and scratch
space. Run it first. Each check exists because its absence cost a leased run.

---

## 2. The GLM-5.3 driver — the main tool

`scripts/glm53_mi300x.sh` already exposes the whole loop, and is the thing most campaign probes
should have called:

```bash
./scripts/glm53_mi300x.sh emit  8            # compile a packet. No GPU, no lease.
./scripts/glm53_mi300x.sh serve 8 8100       # plowrt server; takes its own N-GPU lease
./scripts/glm53_mi300x.sh bench 8 8100       # client only, no lease (the server holds it)
./scripts/glm53_mi300x.sh vllm  8 8200       # vLLM reference server, own lease
./scripts/glm53_mi300x.sh smoke 8
```

Knobs it reads: `PLOW_CKPT` (the **prepped** checkpoint — a raw-HF dir will refuse or fault),
`GLM_RAW` (raw HF, for the vLLM side and the tokenizer), `PLOW_HSACO`, `GLM53_DIR`,
`PLOW_BIN_DIR`, `MAXCTX`, `LADDER`, `BATCH_LADDER`.

`serve` and `vllm` are the two halves of a comparison: same prompts, same client, two base URLs.

---

## 3. Serving and benching, lower level

| tool | use it for |
|---|---|
| `scripts/bench_plowrt_serve.sh <assets> <port> <model> <tokenizer> <ready-timeout>` | `vllm bench serve` against a plowrt endpoint, sweeping `IN_LENS` x `CONCS`. Handles tokenizer-by-repo-id resolution and process-group teardown. |
| `scripts/bench_vllm_chat.sh` | the symmetric vLLM point — same client binary, same `--backend openai-chat`. |
| `scripts/bench_vllm_rocm.sh`, `scripts/bench_plow_rocm.sh` | the ROCm-side pair. |
| `scripts/plow_vs_vllm_rocm.py` | the comparison itself. |
| `scripts/glm53_bench_table.py` | render a result table. |
| `scripts/bench/plowbench.sh` | **source** this in any new probe. Gives `pb_free_port`, `pb_serve_start/wait/stop`, `pb_bench`, `pb_result`, and the artifact checks. Do not re-implement the readiness poll or the result parsing again. |

### The result-path trap

`--result-dir X --result-filename bench.json` does **not** reliably write `X/bench.json`; it can
write `X/main/bench.json`. A summary that looks in one place prints "(missing)" after a 20-minute
run. `pb_result <resdir> <tag>` resolves both, then falls back to any JSON under the tag dir. Use it.

---

## 4. Objects, packets, serving sets

| tool | use it for |
|---|---|
| `scripts/build_gfx942.sh` | build the gfx942 persistent-interpreter code objects. |
| `scripts/build_glm53_gfx942_serving_objects.sh` | regenerate the full GLM-5.3 TP8 serving object set for one packet. **Prefer this over `build_gfx942.sh`** when you need a set that actually serves. |
| `scripts/freeze_serving_set.sh <assets> <objdir> <plowrt> <out>` | freeze packet + objects + binary + replay into one directory. This is what makes a number reproducible; its header lists the four live failures that motivated it. |

**`build_gfx942.sh` does not emit the pinned vendor `.co` kernels** (AITER fmoe x3, MLA x2). A set
built from scratch loads fine until the first MLA decode segment, then dies with
`mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co: No such file or directory`. Copy them from a known-good
set; they are byte-identical across sets from the same vendor drop. `plowbench-doctor.sh` checks
for exactly this.

**Emit needs `PLOW_VERIFY_BIN`.** Without it the emit aborts rc=134 with "checkpoint K rejected the
knob configuration: spawn failed". Do not reach for `--no-knob-verify` to get past it — that marks
the packet a bring-up artifact and `freeze_serving_set.sh` will refuse it.

---

## 5. Numerics and retrieval gates

| tool | use it for |
|---|---|
| `scripts/glm53_needle_campaign.sh` | needle retrieval under one lease. |
| `scripts/glm53_dsa_verify.py`, `scripts/glm53_greedy_agree.py` | greedy/DSA agreement. |
| `scripts/logit_quality_compare.py`, `scripts/block_compare.py`, `scripts/tensor_boundary_compare.py` | numerics comparisons. |
| `scripts/asm_audit.py` + `scripts/asm_expect_gfx942*.json` | assembly expectations for shipped objects. |

A perf change that is not bit-identical needs a numerics gate, not just a faster number.

---

## 6. Measurement hazards

These change what is measured and are easy to leave set from a previous probe. `plowbench-doctor.sh`
warns on every one; `pb_hazard_env` refuses silently-inherited ones in a probe.

| env var | what it silently does |
|---|---|
| `PLOW_PREFILL_SEG_TIMING=1` | **disables segment-major** and forces an all-rank drain per segment — about 3.7x inflation. Its totals are attribution shares, never latency. One rung-width experiment was scored and written up before anyone noticed only one arm had it. |
| `PLOW_TUNEDB` | selects measured GEMM tiles. Measured tiles are **slower** at every rung (5-8 ms); the store must not reach a shipping packet. |
| `PLOW_AMD_DECODE_MIN_RUNG` | picks the decode program. Default 8 is the measured optimum (41.19 ms vs 50.55 at rung 1, re-measured 2026-09-15). |
| `PLOW_HSACO_LOWRUNG` | swaps a different object tier for narrow rungs. |
| `PLOW_GLM_ROWBAND`, `PLOW_TICK_LOG` | fine, but both arms of an A/B must agree on them. |

Two more that are not env vars:

* **`target/release/plowrt` is shared.** A concurrent `cargo build -p plowrt` with no features
  replaces it mid-run, and it does not fail — it serves from the CPU reference interpreter, i.e.
  fluent garbage, ready in 2 s instead of a 12 s weight upload. Copy the binary before any run you
  will publish a number from.
* **Arm order matters.** Benching several rungs against one warmed server and comparing those
  numbers to a single-rung run is invalid: the same 8192 rung read 952 vs 586 depending on
  position. `pb_bench` stamps arm order into the result dir so a reader can see it.

---

## 7. A/B scoring

A four-arm T4 (`ctl / treat / ctl2 / treat2`) is scored against the larger of two spreads: the
control drift `|ctl - ctl2|` **and** the treatment spread `|treat - treat2|`. Flooring on control
drift alone produces false positives — one probe printed `VERDICT: REAL -3.11 ms` while its two
identical treatment arms disagreed by 5.59 ms, 47x the control drift. If
`tspread > 3 * drift`, the run is not convictable regardless of the delta; re-run rather than
report it.

---

## 8. The GPU queue

Submit; do not lease directly.

```bash
<gpuq>/submit.sh <label> <ngpu> <cmd...>       # "0-" label prefix = quick lane
```

The runner **idle-exits after 1800 s on an empty spool and releases the lease**. A job submitted
after that sits in the spool with nothing to pick it up — this has happened. Check the runner is
alive (the doctor does, given `PB_GPUQ`) and restart it before submitting into a cold queue.

Never kill another lease holder, never kill by session or process group, and never read another
process's `/proc` environ.
