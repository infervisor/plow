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
| vLLM client (`vllm bench serve`, and the vLLM reference server) | `/app/plow/build-gemma31/vllm-python` — a prebuilt venv/launcher, from source, not nix | `PB_VLLM` (that path is `plowbench.sh`'s default), or the path baked into the bench scripts |
| ROCm runtime the vLLM client links | `/opt/rocm/core-7.14/lib` — the LAB ROCm, not nix's | `PB_VLLM_ROCM_LIB` (its default), which `pb_bench` **exports** to the client as `VLLM_ROCM_LIB`. `VLLM_ROCM_LIB` is never read as an input |
| `gpulease` | `<repo>/perf-data/tools/gpulease`, else `/app/plow/perf-data/tools/gpulease` — **not on PATH** | absolute path; `plowbench-doctor.sh` probes repo-relative first, then `/app/...`, then `PATH` |
| model checkpoints | `/workspace/models/...` | `PLOW_CKPT`, `GLM_RAW` |

**The absolute `/app/...` and `/workspace/...` paths are one lab box's layout.** On a checkout that
is not that box they do not exist; the repo-relative form is what the scripts resolve first. Check
before pasting a path out of this table.

The vLLM client should be the same binary against both servers: same client, same metric
definitions, different `--base-url`. That symmetry is what makes a plow-vs-vLLM number comparable,
so do not swap in a different client for one side. **This is a rule, not a property of the
scripts** — they do not all agree today. `plowbench.sh` uses `$PB_VLLM`;
`glm53_mi300x.sh vllm` uses `$WT/build-gemma31/vllm-python` while `glm53_mi300x.sh bench` uses
`$WT/.venv-vllm028/bin/python`; `bench_plowrt_serve.sh` and `bench_vllm_chat.sh` run the client
from a `rocm/vllm` **Docker image** unless `VLLM_VENV` is set. Pin one and say which.

**Every GPU process goes through the queue**, never a raw `gpulease` and never a bare run.

---

## 1. Check the environment before leasing anything

```bash
nix develop --command scripts/bench/plowbench-doctor.sh [assets-dir] [object-dir] [plowrt] [arch]
```

Every argument is optional and has an env fallback: `PB_ASSETS`, `PLOW_HSACO`, `PLOWRT_BIN`
(default `<repo>/target/release/plowrt`), `PB_ARCH` / `TARGET_ARCH`. Omitting assets or objdir is a
*warning*, not a failure — you get exit 2 and the checks that need them are skipped.

CPU only; leases nothing. Exit 0 = safe to lease, 1 = something will fail after the weights load,
2 = warnings only. It checks, in order: the nix shell (plus `python3`, `curl`); hazardous `PLOW_*`
overrides left in the environment; the binaries (plowrt, the vLLM client, `plowc`); the packet hash
and the object set; `gpulease` and the queue runner; and scratch space (warn at 85% full, fail at
95%). Run it first. Each check exists because its absence cost a leased run.

**It is arch-aware, not AMD-only.** `pb_detect_arch` resolves gfx942/gfx950/sm_90a/sm_120/sm_89
from the hint, then `build.json`'s `arch`, then an objdir glob, then `nvidia-smi`/`rocminfo`. On
AMD it checks the pinned vendor `.co` kernels that `build_gfx942.sh` does **not** emit (3 `fmoe`
plus 2 MLA) and the four required `.elf`s; on NVIDIA it checks `.cubin` objects and CUDA symbols in
plowrt instead. The queue-runner check runs only when `PB_GPUQ` is set *and* `$PB_GPUQ/runner.log`
exists.

---

## 2. The GLM-5.3 driver — the main tool

`scripts/glm53_mi300x.sh` already exposes the whole loop, and is the thing most campaign probes
should have called:

```bash
./scripts/glm53_mi300x.sh emit  8                 # compile a packet. No GPU, no lease.
./scripts/glm53_mi300x.sh serve 8 8100            # plowrt server; takes its own N-GPU lease
./scripts/glm53_mi300x.sh bench 8 8100 [label]    # client only, no lease (label default "plow")
./scripts/glm53_mi300x.sh vllm  8 8200            # vLLM reference server, own lease
./scripts/glm53_mi300x.sh smoke 8100              # readiness + coherence — takes a PORT, not a TP
./scripts/glm53_mi300x.sh stop  [assets-pattern]  # kills the plowrt, not the gpulease wrapper
```

**`smoke` takes the port.** `smoke 8` polls `http://127.0.0.1:8` and hangs; it is the one
subcommand whose first argument is not the TP degree.

Knobs it reads: `PLOW_CKPT` (the **prepped** checkpoint — a raw-HF dir will refuse or fault),
`GLM_RAW` (raw HF, for the vLLM side and the tokenizer), `PLOW_HSACO`, `GLM53_DIR`,
`PLOW_BIN_DIR`, `MAXCTX`, `LADDER`, `BATCH_LADDER`, plus `NCU`, `OUTLEN`, `NPROMPT`, `IN_LENS`,
`CONCS`, `SMOKE_TIMEOUT`, `GPU_LEASE_TIMEOUT`, `GPU_MEM_UTIL`, `VLLM_SEQS`, `VLLM_ROCM_ROOT`,
`VLLM_ROCM_USE_AITER`. It does **not** read `PB_VLLM`.

`serve` and `vllm` are the two halves of a comparison: same prompts, same client, two base URLs.

---

## 3. Serving and benching, lower level

| tool | use it for |
|---|---|
| `scripts/campaign/campaign.py <build\|serve\|bench\|probe\|cert\|compare\|roofline\|loop\|sweep\|ledger> <recipe>` | the unified campaign driver — a recipe TOML instead of a bespoke probe. See [07 — perf campaign](07-perf-campaign.md). |
| `scripts/bench_plowrt_serve.sh <assets> <port> <model> <tokenizer> [ready-timeout]` | `vllm bench serve` against a plowrt endpoint, sweeping `IN_LENS` x `CONCS`. Handles tokenizer-by-repo-id resolution and process-group teardown. **It runs the client from a `rocm/vllm` Docker image unless `VLLM_VENV` is set** — so by default it is *not* the same client binary as `plowbench.sh`'s. |
| `scripts/bench_vllm_chat.sh <hf-repo-id> <tp>` | the symmetric vLLM point, same `--backend openai-chat`. Same Docker-image client as `bench_plowrt_serve.sh` unless `VLLM_VENV` is set — pair it with that script, not with `plowbench.sh`. |
| `scripts/bench_vllm_rocm.sh`, `scripts/bench_plow_rocm.sh` | the ROCm-side pair. |
| `scripts/plow_vs_vllm_rocm.py` | the comparison itself. |
| `scripts/glm53_bench_table.py` | render a result table. |
| `scripts/bench/plowbench.sh` | **source** this in any new probe. Gives `pb_free_port`, `pb_serve_start <plowrt> <assets> <objdir> <port> <log> [timeout]`, `pb_serve_wait [secs]`, `pb_serve_stop`, `pb_bench <resdir> <tag> <model> <conc> <nprompts> <isl> <osl> [extra…]`, `pb_result <resdir> <tag>`, `pb_model_id`, `pb_detect_arch`, and the artifact checks (`pb_require_nix`, `pb_hazard_env`, `pb_check_plowrt/assets/objects/vllm`). Do not re-implement the readiness poll or the result parsing again. |

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
| `scripts/freeze_serving_set.sh <assets> <objdir> <plowrt> <out> [serve.log]` | freeze packet + objects + binary + replay into one directory. This is what makes a number reproducible; its header lists the four live failures that motivated it. It refuses a packet whose `build.json` `knobs.K` is not `verified`. |

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

These change what is measured and are easy to leave set from a previous probe. `pb_hazard_env`
**warns** on each one it finds inherited (it does not refuse) and bumps the warning count, so
`plowbench-doctor.sh` exits 2. Declare an intentional one to silence it:
`PB_ALLOW_HAZARD="PLOW_TICK_LOG PLOW_GLM_ROWBAND"`.

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

The queue lives outside the repo — `submit.sh` and the runner are not in `scripts/`, so nothing
here can be checked against source; the recorded location is in
[`tp-bringup-upstream-review-log.md`](tp-bringup-upstream-review-log.md) (row 38), which is also
where the two rules below come from. One runner holds all 8 GPUs under a single `gpulease -n 8`.

* **A job runs under `nix develop --command`, from the cwd you submitted from.**
* **No environment is captured.** Pass anything the job needs explicitly: `env VAR=value <cmd>`.

The runner **idle-exits after 1800 s on an empty spool and releases the lease**. A job submitted
after that sits in the spool with nothing to pick it up — this has happened. Check the runner is
alive (the doctor does, given `PB_GPUQ` and an existing `$PB_GPUQ/runner.log`) and restart it
before submitting into a cold queue.

Never kill another lease holder, never kill by session or process group, and never read another
process's `/proc` environ.
