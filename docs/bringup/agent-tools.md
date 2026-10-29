# Agent tools: the scripts to use, and the environment they run in

`scripts/` holds 400+ scripts. Most campaigns have nonetheless written their own throwaway probe,
rediscovered the same environment failures, and burned leased GPU time doing it. This file is the
tool surface an agent should reach for FIRST, so that does not keep happening.

**The rule: look here before writing a probe. If a tool below does the job, use it. If it almost
does the job, extend it and keep the extension.** Experiments belong in existing `scripts/bench`,
`scripts/campaign` or `perf-data` workflows; put their output in scratch outside the repo. Add a
new script only for a reusable capability the existing tools cannot express.

---

## HARD RULE: perf work iterates on a single block, against that block's roofline

Every kernel, fusion, route or knob candidate is screened on **one block**: one layer (or the
smallest layer span containing each layer kind the candidate touches) at the served shape,
precision, emit/compile flags, rung width and context, compared with a **calibrated**
bandwidth/compute roofline for that block. Whole-network `step_bench`, FP32 gates and prod-mix
grids run only for a candidate that already wins at block level; `step_bench` at the affected
rungs then confirms, it does not screen. Full-network iterations cost hours of queue per point, and
most candidates refuted in the 2026-10 campaigns (DeepGEMM, PDL, hot-expert split, glue fusion,
prefill router) could have been killed at block level.

A campaign report (agent result, PR text, `comparison.md`) **starts with the block-vs-roofline
table**, one row per affected block × rung (decode B or prefill rows) × context:

| block (layer kind, rung, ctx) | block time | roofline floor | % of floor | delta vs current |
|---|---|---|---|---|

% of floor = floor / measured time. Delta vs current = candidate minus the production packet's
same block at the same point, measured in the same lease. Name the ceilings behind the floor and
whether they were measured on this GPU; a datasheet floor is a hypothesis ("Per-rung gate" below).

Harness (read [05-single-block-sweep.md](05-single-block-sweep.md) and the scripts it names):

| step | NVIDIA | AMD |
|---|---|---|
| emit one block | `campaign.py build <production recipe> --block L[..R] --out <fresh>`: the recipe's emit env, objects, library-attention and cuBLASLt routes, `block.json` in the assets (Gemma-4 dense/MoE/KV-shared, Qwen3.8 GDN+full at every decode rung) | `plowc --emit devblob --block`; MLA families emit `act.x`-in/out only |
| time | `block_run <asset> bench --batch .. --ctx ..` (decode rung B at live context); `block_run <asset> prefill-bench --query-rows .. --prior 0,..,16384 [--requests N --riders D --rider-ctx C]` (one packed launch after a prefix, riders on the same launch); `sweep.json` records packet/binary sha256, device, declared routes and library-attention launches served vs native fallback | `campaign.py block-bench`; truncated-model `PLOW_LAYERS=N` sweeps |
| ceilings | `scripts/bench/calibrate_ceilings.py --out ceilings.json` (one short lease per box: HBM read, BF16/FP8 GEMM peaks) | recipe `[roofline]` ceilings |
| floor + report table | `scripts/campaign/block_roofline.py <block assets> --ceilings ceilings.json --plow sweep.json --current <production sweep.json> [--router-table t.bin]`: per-instruction floor from the block's own packet (`op_roof.py`), prints the table above | `campaign.py block-roofline` (`--router-table`) |
| A/B | `campaign.py nv-block-ab --control <prod block> --treatment <candidate block> --verb bench\|prefill-bench --args ".." --ceilings ..`: four frozen arms in one lease, `comparison.md` = the table + per-point verdict (§7 spread rule) | `campaign.py block-ab` |
| real MoE routing | serve the full packet with `PLOW_NV_DUMP_KV=<dir> PLOW_NV_DUMP_ACT=<len>:<pc>`, then `scripts/bench/router_table.py <dump>.json --out t.bin` | AMD capture `rank0.act.tab.bin` |
| reference block | `scripts/block_layer_bench.py` (vLLM's own decoder layer; joined with `--vllm`) + `scripts/block_compare.py` | `runtime/tests/*_block_*_test.c` oracles |
| block parity | `block_run packed-check` / `mixed-check` (packed / mixed vs serial on the block) | `amd-block --dump`, C oracle harnesses |

Remaining gaps. List the ones a candidate hits in its report; do not write new harness code inside
a perf campaign (extend the harness as its own change):

* MLA / DSA blocks on NVIDIA (no NVIDIA port yet). `block_layer_bench.py` runs only the Gemma family.
* A KV-shared block must start at or before its source layers (the emitter refuses otherwise;
  E4B: `--block 22..42`), and the per-layer-input prologue is not in a block (`act.ple` is an
  input, timing only). The KV-shared tail route is in the block when the range contains the first
  shared layer and ends at the last layer; its prefill floor still prices the tail layers at every
  row of the program, not the tail's few rows.
* A prefill point's floor prices the bucket program the launch ran, padding rows included: when
  the request rows are well below the bucket (2048 rows in a 4096 bucket), % of floor overstates
  efficiency; read it beside the next-smaller bucket's floor.
* The prefill floor of a packed point charges attention per request at a shared prefix; requests
  with different prefixes in one launch are not modelled.
* Block times are host-observed synchronous step/launch times: a few microseconds of launch
  overhead sit above the device floor at small rungs; per-op device time still needs
  `-DPLOW_NV_TRACE=1` or `op_roof.py --nsys`.
* Prefill bucket policy, prefill dispatch/classing and host-gap knobs are invisible at block level
  ([05](05-single-block-sweep.md) "What tier 2 cannot score"): they go to `step_bench` per rung,
  still before any grid.

## Working rules (every agent)

Product invariants:

* Prefix cache always on, VMM prefix included (§9 "Runtime environment").
* Packets always carry the 8192-row prefill rung; launch width is plowrt `auto`'s job, never a
  dropped rung or a per-recipe width knob.
* One plowrt binary and one runtime library set (one cuBLASLt) serve every bundle. An asset's
  requirement is declared in the packet and met by the standard runtime; never a per-model plowrt
  build or runtime dir. Validate every bundle on the candidate binary before shipping.
* Packet contract, not plowrt: model knowledge (front-ends, loop kinds, prompt/parse/tool rules,
  state layouts, capability checks) is emitted by devgen/plowc into packet metadata, `serve.json`
  and object symbols, documented in `docs/runtime/*-packet-contract.md`. plowrt executes generic
  ops, validates the contract version and fails closed. Reject model-name branches in plowrt.
* Static runtime: plowrt links libc/libm/libgcc_s and dlopens only vendor libraries (libcuda,
  cuBLASLt, libhsa-runtime64). Third-party kernels (FA4/FA3/FlashInfer, ...) ship as packet cubins
  launched by plowrt's own host stub (`cuTensorMapEncodeTiled`, `cuLaunchKernelEx`), their param
  layout versioned in the packet ABI and refused on mismatch. Never add a dlopen of a non-vendor
  library; today's (FA4 `.so` in `exec/gpu/lib_attention.rs`, `device/cuda/cutlass_fp8.rs`,
  `device/cuda/qwen_gdn.rs`, `asr/nemotron/external.rs`) are debt being removed, not precedent.
* No default-off knob ships; every `PLOW_*` is registered with passing knob tests (§9 "Builds").
* Customer hosts get only assets + plowrt + launcher (deploy script / systemd unit). Evals, perf
  clients and venvs run from our box through `ssh -L`.

Judging:

* Judge every candidate against the same-concurrency vLLM peer, never against plowrt's own
  production packet as the baseline. No accepted losses vs vLLM: a remaining loss gets a new perf
  lever, not a waiver or a "known loss" label. Threshold misses only under §5 "Gate relaxation".
* Attention kernels are chosen per rung by measurement (library or native); the packet declares
  the choice, plowrt stays generic.

GPU budget (write it into every GPU agent's prompt):

* Screen first: single block (HARD RULE), then `step_bench` at the affected rungs plus one c16
  FP32 gate (c8 for Qwen3.8) in one lease. c1/c8/c32 gates and the full prod mix (2 rates × 2-3
  reps) only for a promotion candidate (§5 "Screen, then promote", §9 "Measurement").
* One lease script per decision (one server start per packet); at most one queued lease per agent;
  never recapture a vLLM arm, reuse shared plowrt base arms; merge blockers' jobs go first.
* Traces and profiles run on the lab box (gpu03) under `gpulease` with `--max-hold`. A production
  (gpu04) swap needs an explicit user go each time; gpu04's vLLM is a Slurm deployment owned by the
  cluster orchestrator, and a hand-started server there gets no traffic.

Branches, review, verification:

* All changes consolidate onto `prod-next` through the coordinator. Agents cut short branches from
  the latest `origin/prod-next`, rebase before promotion leases and report their tip; the
  coordinator merges (checks: `check_recipe`, `render --check`, fmt, `cargo check`, the docs/arch
  tool-name grep). Never push `main` or `prod-next`, never merge into other agents' branches, never
  force-push; `prod-next` → `main` only on the user's go.
* Every kernel implemented or adapted gets a design consult before coding (shapes and precision
  per rung, tiling, smem/register budget, sync, source library, gating, roofline sanity) and a diff
  review once it passes local rung gates (bounds, tail rows, KV past `kv_len`, races, alignment,
  FP8 scaling/accumulation, resources, ABI/route gating). BLOCKING findings are fixed before the
  rung gate or the recipe; findings and dispositions are listed per kernel in the report (saved as
  `<scratch>/codex/<kernel>-design.md`, `-review.md`; pipe diff and sources on stdin, the sandbox
  shell can fail). Significant landed change sets also get a review of the merge-base diff from a
  clean detached worktree.
* Lean: contract changes keep their Lean verification (write and run it, never bypass it).
  Approving a new verifier hash in `lean-plow/approved-verifiers.json` is the user's call, per hash.

Docs:

* `docs/arch` never names a tool or assistant (grep `codex|claude` before committing); model names
  are fine as examples. Chapters are organised by technique (sliding window, MLA, recurrent state,
  MoE routing, KV-shared tails, ...), not by model.
* Backend parity: a technique landing on one backend gets the other backend's gap recorded in
  `docs/arch/29` §8, and new work keeps the packet contract reusable by the other backend.

Agent-session hazards:

* `EnterWorktree` in a coordinating session isolates the whole session: running subagents get
  every Bash call refused. Edit other worktrees by absolute path and run git from script files.
  `Agent` `isolation: worktree` fails here (`.claude` is a symlink): `git worktree add` instead.
* A hook refuses git inside loops, `$(...)` with variables or `cd X && ...` chains in worktree
  sessions: put multi-step git in a script file.
* `pkill -f` / `pgrep -f` self-match: §"Lease hygiene". `CARGO_TARGET_DIR` after a box env file:
  §"Shared worktree".
* `compute-sanitizer` under `gpulease`: the lease log shows the sanitized process as
  `foreign-during` and the run ends rc 76. That is expected for a sanitizer run (take no timing
  from it). Run it with `--report-api-errors no`.

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
| vLLM client (`vllm bench serve`, and the vLLM reference server) | `/app/plow/build-gemma31/vllm-python` — a prebuilt venv/launcher, from source, not nix | `PB_VLLM`; else `plowbench.sh` takes the first of that path, `/opt/pytorch/bin/vllm`, `vllm` on `PATH`, `$VLLM_VENV/bin/vllm`. `llm_grid.sh` sets it to `PYREF` |
| ROCm runtime the vLLM client links | `/opt/rocm/core-7.14/lib` — the LAB ROCm, not nix's | `PB_VLLM_ROCM_LIB` (its default), which `pb_bench` **exports** to the client as `VLLM_ROCM_LIB`. `VLLM_ROCM_LIB` is never read as an input |
| `gpulease` | `<repo>/perf-data/tools/gpulease`, else `/app/plow/perf-data/tools/gpulease` — **not on PATH** | absolute path; `plowbench-doctor.sh` probes repo-relative first, then `/app/...`, then `PATH` |
| model checkpoints | `/workspace/models/...` | `PLOW_CKPT`, `GLM_RAW` |

**The absolute `/app/...` and `/workspace/...` paths are one lab box's layout.** On a checkout that
is not that box they do not exist; the repo-relative form is what the scripts resolve first. Check
before pasting a path out of this table.

The vLLM client should be the same binary against both servers: same client, same metric
definitions, different `--base-url`. That symmetry is what makes a plow-vs-vLLM number comparable,
so do not swap in a different client for one side. **This is a rule, not a property of the
scripts** — they do not all agree today. `plowbench.sh` uses `$PB_VLLM` (above);
`glm53_mi300x.sh vllm` and `bench` both use `$WT/build-gemma31/vllm-python`;
`bench_vllm_chat.sh` always runs server and client from a `rocm/vllm` **Docker image**, and
`bench_plowrt_serve.sh` uses that image unless `VLLM_VENV` is set. Pin one and say which.

**Every GPU process goes through the queue**, never a raw `gpulease` and never a bare run. On the
Gemma/H100 campaign use `scripts/bench/gpuq.py submit <label> <ngpu> <cmd...>` (FIFO, root
`/tmp/plow-gpuq`, `status` lists jobs; submit restarts an idle-exited runner); the external
`submit.sh` queue in §8 is a separate lab workflow. A script that leases internally is not safe to
run inside either queue unchanged.
Put every required `PLOW_*` setting in the queued command (for example, `env PLOW_FOO=1 ...`);
the persistent queue runner removes inherited `PLOW_*` values before starting each job.

---

## Campaign playbook (any model, any GPU)

Order of operations. Each step has a tool; do not write a probe for it. Complete the per-rung
gate before the full-model accuracy and serving stages, then repeat that gate for affected rungs
after a kernel change. Existing serving evidence can prioritize work but cannot replace the gate.
In the optimisation loop (steps 7-9) a candidate is screened on a single block first (HARD RULE at
the top); steps 3-6 run only for a block-level winner.

| # | step | tool |
|---|---|---|
| 0 | start from everything on | copy `scripts/campaign/recipes/templates/llm-sm90a-default-on.toml`: the first build carries every route plowrt has (packed prefill + compact terminal, decode ladder, Lt prefill/decode, generated attention roles, live KV, prefix cache, token batch). The campaign measures *removing* routes; a route the packet lacks needs a `[capabilities]` waiver with the measured reason |
| 1 | preflight (CPU) | `plowbench-doctor.sh <assets> <objdir> <plowrt> <arch>`: env, hazards, binaries, packet, lean qualification, objects, lease, disk, **stale harness/recipe copies vs HEAD**, and the **capability inventory** (`plowrt inventory <assets>`: fails on a route that is off and not waived) |
| 2 | build from the recipe | `campaign.py build recipes/<ns>/<model>/<cell>.toml --out <fresh dir>`; it ends with `plowrt qualify` into `<out>/lean-qualification.json` (receipts, approved verifier, replay, media geometry). `campaign.py qualify --assets <out>/assets [--strict]` reruns it |
| 3 | per-rung accuracy, resources + measured roofline | `scripts/bench/step_grid.sh <assets> <out>` (step_bench B × ctx; instruction-cap sweeps for native decode only), then `scripts/bench/step_boundary.sh <assets> <out>` (ctx 4094/4095/4096/8191/8192 under `CUDA_LAUNCH_BLOCKING`: live KV commits 4096-row blocks and a kernel that stages rows past kv_len faults only there), then `scripts/bench/op_roof.py <out>/disasm.txt --ctx N --sweep B=<jsonl>` (bytes, FLOPs, floor, measured, % roof per op). Library-routed decode needs `op_roof.py --nsys` with JSON disassembly and a measured CUDA trace; prefill CUDA graphs use `--nsys-correlation <id>` for exact packet-segment mapping, while `--segtime` covers per-chunk prefill diagnostics. No measurement = floor only. Use block/route harnesses below for other kernels; GLM/MLA/MoE: `scripts/campaign/op_roofline.py` |
| 4 | full-model accuracy gates | `campaign.py gate <recipe> --assets <out>/assets --out <dir>` runs the recipe's `[gates]` in one lease (logit parity top1/KL, ASR WER, TTS CER per language, S3Gen rel-L2, media parity `mm_parity`); `--score-only` re-scores, `--dry-run` prints `run.sh` (under `gpuq.py`, queue `bash <out>/run.sh` from the dry run, then `--score-only`: `gate` itself leases) |
| 5 | serving grid vs the reference | `scripts/bench/llm_grid.sh plow\|vllm <res>` (same client, unique prompts per cell and repeat, greedy + sampled, 2 repeats, prefill-only and decode-only cells, vLLM `/metrics`, plow PACKLOG); agentic multi-turn with prefix caching on both sides: `llm_grid.sh plow\|vllm <res> --agentic` (`agentic_turns.py`, c32/64/128 x 10 turns to ~16K); open-loop production mix (Poisson sessions, think time, sampled turns/lengths, goodput): `PROD_RATES='0.5 1' llm_grid.sh plow\|vllm <res> --prod` (`agentic_turns.py --open-loop`, cells `q<1000*rate>`); speech: `scripts/tts/tts_bench.py`, `scripts/asr/nvidia/served_bench.py`; voice: `scripts/voice/serve_voice_agent.sh calls` |
| 6 | audit waterfall | `scripts/bench/waterfall.py <res>/plow <res>/vllm`: grid with spread, reference prefix-cache hits (fails > 5%), wall ms/request split (mixed, prefill-only, decode, host gap, idle, padding, riders), decode ms/step side by side. `vllm_metrics.py cells`, `packlog_audit.py`, `nsys_busy.py` are its parts |
| 7 | fix, one variable | kernel/runtime change in your own detached worktree |
| 8 | verify the patch | `scripts/campaign/verify_patch.sh <patch>`: applies to HEAD in a private index, archives `git write-tree`, builds bins/examples/tests, plowrt lib (cuda+hsa), knob tests, plow-asset + packet tests, `lean` (in-tree verifier build, proof audit, approved verifier, CPU-verifier suites via `scripts/lean_correctness_ci.sh`) for Lean/receipt changes, `perf` (checkpoint P, `perf_gate_ci.sh`) for knob registry changes, py_compile / `bash -n` / TOML of touched scripts |
| 9 | re-measure | repeat steps 3–6 for affected rungs and the selected full model; A/B scoring as in §7 |

### Per-rung gate and recipe selection

For every affected prefill bucket and decode batch/context rung, use the existing kernel harness
(`step_grid.sh` + `op_roof.py` for dense decode; `campaign.py block-roofline`, `block-bench` and
`block-ab` where applicable; `gemma4_route_matrix.sh` for native-vs-cuBLASLt projection shapes)
before a full-model performance run. The route-matrix script's build stage uses `quiets.sh` and
its run stage submits through `gpuq.py`; use those stages rather than launching the executable directly. Match the production
cubin build command and flags, including rung-specific defines. Measure on the target GPU with
repeated, uncontended timings;
use a measured bandwidth/compute ceiling when available. A datasheet floor or unmeasured
`op_roof.py` row is a hypothesis, not roofline attainment. Record achieved fraction and investigate
material unexplained gaps before promotion; do not invent a universal percent threshold.

Check kernel outputs against the appropriate reference on representative and boundary inputs,
including quantization scales where used. Inspect registers, spills/local memory, stack and shared
memory in the built object; confirm the object and launch geometry are the ones the packet uses.
Compare native Plow and segmented cuBLASLt under the same shape, precision, input and timing
protocol; select only the fastest correct route for each rung. Validate the chosen route in an
isolated block, including ragged rungs, before full-model accuracy and matched serving. An
unmeasured rung or a correctness/resource failure keeps the candidate experimental.

**Per-rung knob values.** An emit knob that only picks a rung program's op or library route is
declared rung-eligible in `crates/devgen/src/knob_spec.rs` (`.per_rung(&["prefill"])`; today only
`PLOW_NO_GLU_FUSE`). Choose it per rung, not packet-wide:
`campaign.py rung-sweep <recipe> --knob K --values 0,1 --stage build|run|score|verify`
(`scripts/campaign/README.md`, "Per-rung knob choices") times every rung in every value arm in one
queue job, compares the same launches' logits, lists object resources and writes the
`[emit.rung_env.prefill.<rows>]` recipe snippet. devgen refuses an override of a packet-wide knob
(tensors, objects, program set, checked constraints); making a knob eligible needs that audit plus a
mixed-rung emit test (`crates/devgen/tests/rung_env.rs`). The score labels a choice `experimental`
unless accuracy is accepted (bitwise, or `--kl-max` + `--top1-min`), the arms' objects show no
spills, and `--roofline` / `--library-resources` records are given; only `qualified` choices go
into a production recipe, and the mixed packet then needs its own full gates (FP32 included).

Store tuning evidence with source and object SHA256s, exact compiler/cubin flags, launch geometry,
device/compiler versions, shape/precision and correctness results. A Git commit is useful
traceability, but evidence remains valid across commits if these effective inputs match; changed
inputs require revalidation. Do not reject a valid kernel measurement solely because HEAD moved.

Maintain **one canonical production TOML per model/GPU/precision campaign** in
`recipes/infervisor`, updating it in place as each rung qualifies. It must express the actual
per-rung object/compile choice and runtime settings; a comment or ignored metadata field is not a
reproducible recipe. Keep trial TOMLs in `scripts/campaign/recipes` or outside-repo scratch until
qualified. Do not create a second production variant to record an experiment.

Scheduling is not a recipe choice: recipes (production and trial) leave `PLOW_OBJECTIVE` at the
default `auto`, which runs the latency rules at narrow width and switches to throughput at 8 rows,
any queue or a 90%-reserved KV budget (`docs/flags-reference.md`). Pin it only as a deliberate, commented override in a
profile's `serve_env`; expert overrides such as `PLOW_MULTISTEP` or `PLOW_DECODE_MAX_RUNG` stay
where a cell measured them.

### Campaign result placement

Use one designated `comparison.csv` per campaign for matched serving results and a neighboring
`comparison.md` for qualified wins and concise provenance; per-family summaries are indexed in
[results/README.md](results/README.md). Put raw client/server logs, JSON,
captures, profiler traces, generated HTML, temporary CSVs and full manifests in named scratch
outside the repo; link or hash the evidence from the summary. Never put raw benchmark data in
arbitrary repo directories, including `docs/`, `scripts/` and `perf-data/`. Curated, validated
tuning summaries can live in an established campaign area, with raw evidence kept in scratch.
Use `campaign.py bench/compare`, `llm_grid.sh` and `waterfall.py` while iterating; a final
comparison is rendered only by `campaign.py report` (next section). Keep the same client, model/weights,
quantization, traffic, concurrency, GPU count and cache policy on both serving arms; record
throughput, throughput/GPU, TTFT P99, TPOT P99, peak memory and output correctness. Render in
scratch and append the vetted rows to the sole campaign CSV rather than committing each tool's
generated result tree. Compare prefix-cache token-hit rates with token-hit rates and request-hit
rates with request-hit rates; the two denominators can differ sharply on shared short prefixes.

### Final performance report (strict)

**This is the only accepted format for a final performance comparison** (Infervisor/plow vs vLLM
or any other baseline): in a summary, a `comparison.md`, a PR or a chat report. Do not hand-write
the table and do not quote another renderer's numbers as final. The harness renders it and
refuses anything incomplete:

```bash
# both arms with llm_grid.sh (single-turn, --agentic and/or --prod open loop with PROD_RATES);
# each writes <res>/provenance.json
env ... KV_DTYPE=<plow KV dtype, named as vLLM names it> scripts/bench/llm_grid.sh plow <res>/plow [--agentic|--prod]
env ... scripts/bench/llm_grid.sh vllm <res>/vllm [--agentic|--prod]
campaign.py gate <recipe> --assets <served assets> --out <gate> --only llm_fp32_ref   # same packet
campaign.py report --baseline <res>/vllm --infervisor <res>/plow --gate <gate>/gates.json --out <dir> \
    [--cells g128,s128,a64.g]
# = scripts/bench/serving_comparison.py render ...; writes comparison.md / .json / .csv
```

One table per compared cell (workload x input/output x concurrency x traffic):

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | | Same as baseline |
| Precision / quantization | | Same as baseline |
| Input / output length | | Same as baseline |
| Traffic / concurrency | | Same as baseline |
| GPU type & count | | Same as baseline |
| Serving stack | | Infervisor |
| Output quality / correctness | | Equivalent |
| Peak GPU memory | | |
| Total throughput | | |
| Throughput / GPU | | |
| TTFT P99 | | |
| TPOT P99 | | |

Rules (enforced in `serving_comparison.py`, tested in `scripts/bench/test_serving_comparison.py`):

* Exactly these 12 rows, in this order. Every cell is filled; missing data is an error, not a blank.
* "Same as baseline" only when the recorded value is literally identical; otherwise the Infervisor
  value is shown and the cell is **NOT MATCHED**. Matched = the first five rows identical.
  Input/output also compares the per-request length lists.
* Serving stack: Baseline = stack, version and serve flags; Infervisor =
  `Infervisor (plowrt <commit>, packet <model.pkt sha256 prefix>)`.
* Output quality: from the FP32-reference gate (`gates.json`, `llm_fp32_ref`, §5). "Equivalent" only
  if it passed for the exact served packet (its `packet_sha256` = the Infervisor arm's); otherwise
  "Not equivalent (<failing metric>)". No gate, or a gate for another packet, is an error.
* Total throughput = (input + output) tok/s; Throughput / GPU = total / GPU count; TTFT/TPOT P99 in
  ms; peak GPU memory in GiB from `gpu_peak_mem.py` (sampled per cell by `llm_grid.sh`). The
  Infervisor column shows the Infervisor / Baseline ratio, e.g. `1,234 tok/s (1.12x)`.
* Values are means over repeats (at least 2, the recorded count, identical on both arms). The
  spread, (max - min) / mean, is a note under each table; a spread > 10% marks the value `*` and
  is listed as FLAGGED.
* Open-loop (`--prod`, cells `q<1000*rate>.g`): the same 12 rows. Input/output pairs on a hash of
  the whole plan config plus the SLO, Traffic lists each repeat's seed (both arms must replay the
  same seeds), and any failed request refuses the cell. Goodput, SLO attainment, achieved
  concurrency, P50s and cached fraction go in a supplementary note under the table, never as rows.
* Exit 2 = refused, nothing written (missing data, provenance, gate, repeats, unpaired cells).
  Exit 1 = written but NOT MATCHED or NOT EQUIVALENT: not a valid final comparison. Exit 0 = valid.

Provenance comes from the run, never from a person: `llm_grid.sh` calls
`serving_comparison.py record` (model id + config/weights hash, precision from `config.json` plus
the KV dtype, GPU name/count from `nvidia-smi` honoring `CUDA_VISIBLE_DEVICES`, vLLM version and
flags, plowrt commit, packet sha256, repeats, sampling flags). The plow arm needs `KV_DTYPE` (or
`PRECISION`), and `PLOWRT_GIT_SHA` when `PLOWRT` is outside a checkout; `MODEL_VERSION` overrides
the model id. A result dir from before the recorder needs a provenance file written from its logs
(`--baseline-provenance` / `--infervisor-provenance`). `campaign.py gate` records `packet_sha256`;
re-score an older gate with `--score-only` against the served assets.

Pitfalls this playbook exists for:

* **Prompt reuse.** A fixed client seed replays earlier cells' prompts; vLLM's prefix cache then
  skips prefill plow computes (41-58% hits made a 1.22x gap look like 1.5x,
  [throughput-audit.md](../runtime/throughput-audit.md)). `pb_bench` seeds per cell, `llm_grid.sh`
  per cell and repeat, and `waterfall.py` / `vllm_metrics.py cells` fail on hit rates > 5%.
* **Temperature.** Always pass it explicitly (`--temperature 0` greedy); report greedy and sampled.
  Unset, each server applies its own generation-config defaults.
* **One run per cell.** Two repeats and the spread; a delta inside the spread is not a result.
* **Unmatched decode.** Compare decode steps at the same B and context (`step_bench` vs the
  reference's decode-only phase), not TPOT across different batch compositions.
* **Shared worktree.** Never checkout/stash/reset it; its copies can be stale vs HEAD (the doctor
  warns). Work in `git worktree add --detach <dir> HEAD`, own `CARGO_TARGET_DIR`, deliver patches.
  Set `CARGO_TARGET_DIR` **after** sourcing a box env file: a hand-built env (e.g. the lab's
  `g26main/env.sh`) exports a shared target dir, and a build into it silently replaces other
  campaigns' `plowc`/`plowrt`/`step_bench` with a newer mtime that defeats every staleness check.
* **Lease hygiene.** One lease per decision (one server start per packet), at most one queued
  waiter per agent: never stack duplicate waiters for the same job. A lease command that is a
  script must be executable (`chmod +x`; otherwise rc 126 after the wait). `GPU_LEASE_TIMEOUT`
  (default 1800 s) bounds the wait for the cards. Never `pkill -f`/`pgrep -f` a pattern that occurs
  in your own command line or in the kill script's file name (it kills your own shell, exit 144):
  write the kill logic to a neutrally named script file and kill by pid. `gpulease` serves waiters FIFO (`gpulease --status` lists the queue) and
  runs the command in its own process group: the whole job tree stops when the command exits or
  the lease is killed, so kill the `gpulease` process, not its children. A run that holds cards
  past `GPU_LEASE_MAX_HOLD` (default 7200 s) while someone waits is stopped with rc 124
  (`PREEMPTED` in the log); a longer job passes `--max-hold <s>` explicitly, ideally with
  done-marker resume. Still wrap GPU commands in `timeout` and keep leases short.
  `gpulease --help` is not help: a label without a command is refused (exit 2).
* **No nix.** On a hand-built box set `PLOW_CAMPAIGN_NO_NIX=1` (campaign.py, the doctor,
  verify_patch.sh, `gpuq.py submit`) instead of faking `ROCM_PATH`.

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
overrides left in the environment; the binaries (plowrt, the cuBLASLt it will bind — beside the
binary in `<bin>/../lib` or `<bin>`, else the loader's `libcublasLt.so.13` — against the packet's
`build.json` `runtime_requires.cublaslt`, the vLLM client, `plowc`); the packet hash
and the object set; `gpulease` and the queue runner; scratch space (warn at 85% full, fail at
95%); harness/recipe files vs HEAD; and the capability inventory. Run it first. Each check exists
because its absence cost a leased run.

**Capability inventory ([8]).** `plowrt inventory <assets> [--format json]` is a static (CPU) table
of the routes the packet declares, each with the reason it is on, off or not applicable: prefill
buckets, decode ladder, recurrent state, prefix cache, live KV, packed prefill, masked padding,
compact terminal (the first token from the prefill launch; needs the generic or post-norm terminal
tail), segment roles, token batch, generated kernels, device sampler, Lt prefill. The doctor
(`scripts/bench/capability_check.py`) fails on an `OFF` unless the recipe beside the assets waives
it in `[capabilities]` (`name = "measured reason"`) or `PB_CAPABILITY_WAIVERS=name,...` does. A
recurrent network's prefix cache is `ON` when the packet's `recurrent_state` section declares its
state buffers (the block store captures them at a boundary; CUDA does not auto-select it, so the
recipe sets `PLOW_VMM_PREFIX=1`), and its token batch is `ON` only when the packet
carries the packed scan's init-state scratch (`act.gdn.initstate`), otherwise `OFF`. The Qwen3.8 campaign ran
some forty builds before packed prefill was turned on; this check exists so the next model starts there.

**It is arch-aware, not AMD-only.** `pb_detect_arch` resolves gfx942/gfx950/sm_90a/sm_120/sm_89
from the hint, then `build.json`'s `arch`, then an objdir glob, then `nvidia-smi`/`rocminfo`. On
AMD it checks the pinned vendor `.co` kernels that `build_gfx942.sh` does **not** emit (3 `fmoe`
plus 2 MLA) and the four required `.elf`s; on NVIDIA it checks `.cubin` objects and CUDA symbols in
plowrt instead. The queue-runner check runs only when `PB_GPUQ` (the queue root) is set *and*
`$PB_GPUQ/runner.log` exists; it recognizes both the `gpuq.py` runner and the lab `runner.py`.
A fifth argument `block` skips the vLLM client check.

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
./scripts/glm53_mi300x.sh stop  [assets-pattern]  # kills the plowrt (its gpulease wrapper then releases)
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
| `scripts/campaign/campaign.py <build\|serve\|bench\|serve-bench\|probe\|gate\|roofline\|loop\|sweep\|block-roofline\|block-bench\|block-ab> <recipe>`, plus `report`, `cert`, `compare`, `ledger` (no recipe; see `--help`) | the unified campaign driver — a recipe TOML instead of a bespoke probe. See [07 — perf campaign](07-perf-campaign.md). |
| `scripts/bench_plowrt_serve.sh <assets> <port> <model> <tokenizer> [ready-timeout]` | `vllm bench serve` against a plowrt endpoint, sweeping `IN_LENS` x `CONCS`. Handles tokenizer-by-repo-id resolution and process-group teardown. **It runs the client from a `rocm/vllm` Docker image unless `VLLM_VENV` is set** — so by default it is *not* the same client binary as `plowbench.sh`'s. |
| `scripts/bench_vllm_chat.sh <hf-repo-id> <tp>` | the symmetric vLLM point, same `--backend openai-chat`. Server and client always run from the same Docker image as `bench_plowrt_serve.sh`'s default (it has no `VLLM_VENV`); pair it with that script without `VLLM_VENV`, not with `plowbench.sh`. It takes no lease itself. |
| `scripts/bench_vllm_rocm.sh`, `scripts/bench_plow_rocm.sh` | the ROCm-side pair. |
| `scripts/plow_vs_vllm_rocm.py` | the comparison itself. |
| `scripts/glm53_bench_table.py` | render a result table. |
| `scripts/serve_test/smoke.sh <plowrt> <out> <serve args...>` | one `plowrt serve` (any `--assets` set, clean env) smoked on every endpoint `/v1/models` advertises: chat, transcription (HTTP/SSE/WebSocket/Realtime), speech, VAD (`smoke_client.py`). `EVAL=all` adds `eval.py` (ASR WER, TTS Whisper CER, LLM checks; also usable alone against `PLOW_URL`). Run through `gpuq.py`. See [serving-deploy-runtime.md](../serving-deploy-runtime.md#running). |
| `scripts/bench/thinking_gates.sh <assets> <objdir> <out>` | thinking-control soundness gates (F1 byte-identical output across pipelined / speculative / multistep / plain decode under a forced trace end, F2 penalties recorded, F4 seeded reproducibility, F3 no-budget TPOT vs `PLOWRT_BASE_BIN`); one server per arm, run as one `gpuq.py` job with the recipe's serve env on the command. |
| `scripts/bench/thinking_stress.py --endpoint name=url … --model M [--arm name=JSON] [--prompts convs.jsonl] --out <scratch>` | thinking-control stress report per endpoint and request variant (the S0–S4 ladder, bridge A/B): strict and judged leaks, `finish=length` %, longest reply, p50/p90 first answer, forced %, answer-reasoning %; every leak and cut-off reply printed. Point plow and the vLLM reference at the same conversations. |
| `scripts/bench/thinking_leaks.py --url URL --name N --model M --raw format\|URL\|FILE --out <scratch> [--temperature T --seeds K]` | thinking/answer separation audit: markers in the answer or trace, the split against the raw generation, deliberation in the answer, streamed vs buffered, empty answers, trace/reply/total budgets; `--record-raw` captures a parser-less server's raw text when one GPU fits one server. |
| `scripts/bench/plowbench.sh` | **source** this in any new probe. Gives `pb_free_port`, `pb_serve_start <plowrt> <assets> <objdir> <port> <log> [timeout]`, `pb_serve_wait [secs]`, `pb_serve_stop`, `pb_bench <resdir> <tag> <model> <conc> <nprompts> <isl> <osl> [extra…]`, `pb_cell` (pb_bench + cells.log markers), `pb_metrics_start/stop` (reference `/metrics`), `pb_result <resdir> <tag>`, `pb_model_id`, `pb_detect_arch`, and the artifact checks (`pb_require_nix`, `pb_hazard_env`, `pb_check_plowrt/assets/objects/vllm`). Do not re-implement the readiness poll or the result parsing again. |

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

**Lean checks a change must keep green.** `scripts/lean_correctness_ci.sh` (lake build, `lake exe
proof_audit`, approved-verifier identity, the explicit ignored CPU-verifier suites) and `plowrt
qualify --assets <dir>` on the affected bundles (receipts plus the bundle obligations
`media_geometry.v1`, `kv_ring.v1`, `speech_fusion.v1`). A new Lean endpoint lands with its manifest entry in
`lean-plow/proof-manifest.json` (the audit fails otherwise), an `approved-verifiers.json` entry for the
rebuilt verifier (a `lean_verify` test fails otherwise), and its ignored test listed in the CI script.
New perf gates go through checkpoint P (`perf_cert.py`, `perf_gate_ci.sh`) and S (`knob_scope_ci.sh`);
`verify_patch.sh` runs `lean` and `perf` for the files they cover. Coverage and gaps:
`docs/bringup/lean-correctness-inventory.md`.

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

**Cross-stack quality is gated against FP32, not against vLLM's bits** (`scripts/llm/fp32_ref_gate.py`,
`campaign.py gate` kind `llm_fp32_ref`). Two FP8 stacks never agree bit for bit, and near-tie
flips fail any exact-match gate (vLLM's repeat floor is 0). Instead an independent FP32 reference
(the served FP8 checkpoint dequantized to FP32 weights, FP32 activations, TF32 off, exact chunked
attention) scores both stacks on one fixed token-id prompt set, and Plow must stay within vLLM's
distance to FP32:

| metric (vs FP32) | definition | pass if Plow ... (default) |
|---|---|---|
| `kl_mean`, `kl_trim_mean`, `kl_p95` | KL(FP32 ‖ stack) per teacher-forced position, FP32 top-20 + remainder bucket (stack's 20th logprob for a missing token); the trimmed mean drops the worst `kl_trim_frac` (1%) | each ≤ vLLM × `kl_ratio_max` (1.25) + `kl_slack_max` (0.002) |
| `kl_over_frac` | fraction of positions with KL > `kl_over_at` (0.1 nat); `kl_over` is the count | ≤ vLLM + `kl_over_slack` (0.005, 7 of 1414) |
| `kl_p99`, `kl_max` | reported, not gated: on 1414 positions p99 is the 14th-worst position and one co-batching flip moves it 60% while the trimmed mean and p95 hold | — |
| `top1_decisive` | top-1 agreement where the FP32 top-1/top-2 margin > `tie_margin` (1.0 nat) | ≥ vLLM − `top1_drop_max` (0.01) |
| `cont_frac` | stack's own greedy continuation: tokens matching FP32 before the first divergence / reference length | ≥ vLLM − `cont_drop_max` (0.05) |
| `cont_decisive_frac` | cases whose continuation never diverges where FP32 was decisive (margin > `tie_margin`); a near-tie divergence is rounding luck, after which greedy histories are incomparable | ≥ vLLM − `cont_decisive_drop_max` (0; a recipe sets a measured margin, with the swing case named, where a documented position flips on every qualified packet) |
| `needle_acc` | needle answer contained in the greedy continuation | ≥ vLLM − `needle_drop_max` (0) and ≥ `needle_min` (0.9) |

Any missing scored position fails; the position set is the reference's, identical for every arm
and every run. **Qualify at the serving concurrencies, not only c1**: `concurrency = [1, 8, 32]` in
`[gates.llm_fp32_ref]` captures plow at each (`plow-c<N>.json`) and scores each against the vLLM
capture taken at the same concurrency (`vllm_capture = { c1 = ..., c8 = ..., c32 = ... }`; with
`vllm_hf` the lease captures them). Co-batching changes which decode rung and which kernel a row
runs on, so a packet's c8 result is not its c1 result: one Qwen3.8 packet scored kl_p99 0.0158,
0.0181 and 0.0255 on three c8 gates with an identical c1. The headline metrics in `gates.json` are c1's; `by_concurrency` holds the rest and
every concurrency must pass. Capture is teacher-forced through `/v1/completions` (token-id
prompt, greedy, `logprobs: 20`): each request starts at prompt + FP32 continuation[:k]; positions up
to and including the first disagreement have the exact FP32 history, the next request restarts one
past it (1 + flips requests per case). Prompt set (`prompts`): natural text at 128/1K/4K/8K/15.9K
tokens from two Gutenberg books, repo docs and repo Rust; eight chat-template agentic samples
(tools, tool calls/results, multi-turn, 54 to 15.3K tokens); 18 chat needles (4K and 15.9K, depths
0.1/0.5/0.9). Build once, cache with sha256s, gate many times:

```bash
P=scripts/llm/fp32_ref_gate.py; HF=<hub gemma-4-12b-it-fp8 dir>
python3 $P prompts --hf $HF --corpus pride=pg1342.txt beagle=pg944.txt docs=repo-docs.md code=repo-code.rs --out prompts.json
scripts/bench/gpuq.py submit fp32-ref 1 timeout 5400 python3 $P reference --hf $HF --prompts prompts.json --out ref.json  # ~50 GB VRAM
# recipe: [gates.llm_fp32_ref] reference = ".../ref.json", vllm_hf = $HF, vllm_args = "<matched vLLM serve flags>"
#         (or vllm_capture = a cached vllm.json against the same ref.json); thresholds as above
campaign.py gate <recipe> --assets <assets> --out <dir> [--only llm_fp32_ref] [--score-only]
python3 $P gate --ref ref.json --cand <dir>/llm_fp32_ref/plow.json --peer <dir>/llm_fp32_ref/vllm.json
```

The tolerances were calibrated on Gemma-4 12B FP8 (H100), where vLLM's own repeats differ by 1.18× in
KL mean, 0.006 in top-1 and 0.021 in agreement. vLLM gated against its own repeat passes in both
directions. Results are in the [gemma12b comparison](results/gemma12b-fp8-20260930/comparison.md#fp32-reference-quality-gate).
The in-lease vLLM server needs `nvcc` (CUDA toolkit on `PATH`, or `CUDA_HOME`), so run it inside
`nix develop` or set it in `[gates.llm_fp32_ref.env]`. Captures send `ignore_eos`, because plowrt
leaves a stop token out of completions logprobs. A case is scored through the reference's first
stop token and no further: a server never produces the tokens after it, and the post-stop
distribution is ill-conditioned (Qwen3.8's FP32 continuation kept emitting
`<|im_start|><|im_end|>` for 36 tokens past `<|im_end|>`, with 1.6-nat decisive margins that bf16
stacks missed by 0.6-1.2 nats of KL; a c1 gate failed `cont_decisive_frac` there). `reference`
records the checkpoint's `eos_token_id` in the reference's meta; a reference made before that
(Gemma's ids were hard-coded) takes `eos = [248046, 248044]` in `[gates.llm_fp32_ref]`.

Unit tests: `cd scripts/llm && python3 -m unittest test_fp32_ref_gate`.

**Screen, then promote.** Judge every arm against the vLLM capture taken at the **same
concurrency** (the gate refuses without one), never against plowrt's own production packet. A
candidate reaches any gate only after it wins at block level (HARD RULE at the top). Screening a
candidate = one c16 gate (c8 for a model
whose recipe gates at 1/8/32, e.g. Qwen3.8) plus step_bench at the affected rungs, in one lease;
c1/c8/c32 and the full production mix run only for a promotion candidate. An **arm** is the same
packet + plowrt + serve env: a run with another knob set, binary or packet is a different arm.
`kl_p99` is knife-edge (one co-batching flip moves it; reported, not gated): compare
`kl_trim_mean`/`kl_p95`, and re-run a miss by < 0.01 on `cont_frac` at c32 before touching kernels.
A c1 failure on a short chat case: first check what is scored past the reference's stop token.

**Gate relaxation for significant perf gains (2026-10-28, user decision).** A change may miss a
recipe's `llm_fp32_ref` thresholds only if both hold:

1. The gain is significant: ≥ 5% on an affected serving rung's step time (step_bench at the rung's
   actual shape), or ≥ 3% on a production-mix metric (goodput, TTFT P99 or TPOT P99, `campaign.py
   report`).
2. Accuracy stays at the vLLM peer's level on the same gate and concurrency: `kl_trim_mean` and
   `kl_p95` ≤ 1.10 × vLLM's; `top1_decisive` ≥ vLLM's − 0.005; `cont_decisive` ≥ vLLM's − 2 cases
   (of the gate's cases); `needle_acc` ≥ vLLM's; in ≥ 2 of 3 runs of the same arm (packet, plowrt,
   serve env). A run of a different arm (another knob set) does not count toward the three.

The recipe then carries only the threshold the rule allows (e.g. `cont_decisive_drop_max` = 2 cases
/ cases), and its `[gates]` block and header record the measured gain, the per-run numbers vs vLLM
and the date. A change that misses the rule keeps the recipe's thresholds. Re-score existing
captures with `--score-only` before running new ones.

**Media (images, audio) are gated by `mm_parity`** (`campaign.py gate`, experimental recipes
`scripts/campaign/recipes/gemma4-e4b.h100.multimodal.toml` and `gemma4-12b.h100.fp8-multimodal.toml`).
One lease runs:

- before the server: the HF reference (`scripts/mm/hf_ref.py all`: processor outputs, projected
  rows, greedy chat cases from `scripts/mm/cases/*.json`), only when `reference/ref.json` is
  missing;
- served: `scripts/mm/gate.py` (prompt-token equality, greedy match vs HF, repeat stability across
  cold and cached prefill, media-collision check) and `scripts/mm/refusals.py` (400s for limits,
  bad media and unsupported formats; mp3/flac/ogg accepted; streaming with logprobs);
- after the server stops: `mm_check --encode` (preprocessing and encoder rows vs HF; build it with
  `cargo build --release -p plowrt --features cuda --example mm_check`).

A cold and a cached prefill split the prompt differently, so their bf16 rounding differs
(|Δlogprob| ≈ 0.1, as on text). The gate accepts a divergence only at a near tie (top-2 margin
≤ `tie_margin`, default 0.25). The served load test (`scripts/mm/load.py`: concurrency, cancel,
slab exhaustion, slab leaks) is a separate run.

### Serve goldens (prompt bytes and token ids of the production packets)

`crates/plowrt/tests/fixtures/serve_golden/<packet>.json` freezes, per production asset
directory, what the chat handler decides for a fixed set of conversations: rendered prompt bytes,
token ids, tool syntax and forced-call opener, reasoning split state, the template's thinking
switch, the stop set, and the digests of the inputs those decisions come from (tokenizer,
tokenizer config, `serve.json`, template). A re-emit must reproduce them byte for byte or it needs
GPU re-qualification. CPU only; needs the asset directories:

```
PLOW_SERVE_GOLDEN="qwen38-27b-bf16=<assets>,qwen38-27b-fp8=<assets>,gemma4-12b=<assets>,\
gemma4-26b-a4b-fp8=<assets>,gemma4-26b-a4b-bf16=<assets>,gemma4-e4b=<assets>" \
  cargo test -p plowrt --features cuda,hsa --lib golden_tests
```

A check run must name every checked-in golden. `PLOW_SERVE_GOLDEN_RECORD=1` re-records (review
the fixture diff: a changed case is a changed prompt). To compare a re-emitted packet with its
predecessor, record it under another name and diff the two files case by case; the device bytes
and the other sections are compared with `scripts/packet_sections.py` (section-aware; the
generated-kernel digests in `segment_roles.json` are not reproducible across nvcc runs).

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
| `PLOW_DECODE_MIN_RUNG` | picks the narrowest decode program. TP engines default to 8 (GLM-5.3 TP8 41.4 vs 51.9 ms at rung 1); single-GPU engines leave it unset. |
| `PLOW_OBJECTIVE` | pins `latency` or `throughput` scheduling (decode quantum, prefill packing, queue TTL, rung fast probe). Recipes and production leave it at the default `auto`, which switches per tick. |
| `PLOW_HSACO_LOWRUNG` | swaps a different object tier for narrow rungs. |
| `PLOW_GLM_ROWBAND`, `PLOW_TICK_LOG` | fine, but both arms of an A/B must agree on them. |

Two more that are not env vars:

* **`target/release/plowrt` (or any shared `CARGO_TARGET_DIR`) is shared.** A concurrent `cargo build -p plowrt` with no features
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

## 8. The external GPU queue (lab workflow)

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

---

## 9. Campaign rules (lessons from the 2026-10 campaigns)

Each rule cost at least one campaign a day of misattributed results.

**Builds and binaries**

* `campaign.py build` runs `$CARGO_TARGET_DIR/release/plowc` (else `target/`) as found: it never
  runs cargo, and refuses only a plowc older than a source in its cargo dep-info. After any
  devgen/plowc change run `cargo build --release -p plowc` (same `CARGO_TARGET_DIR`) first. Never
  rebuild plowc while an emit runs from it. The runtime `.cu` sources are compiled at emit time from
  the worktree, so kernel edits need no plowc rebuild.
* A killed `campaign.py build` leaves cmake's `make` and its nvcc children writing cubins stamped
  with the old packet hash into `<out>/assets` (load then fails "packet/interpreter MISMATCH").
  After a kill: check `pgrep -af "make|cmake|cicc|ptxas"`, then build into a **fresh** out dir.
* `PLOW_EXTRA_DEFINES` in the environment of an emit does nothing: object defines come from the
  manifest (`requires`/`recommends`/tuning). An object variant must be a manifest tuning key.
* `cargo build -p plowrt --example step_bench` builds only the example; the serving binary stays
  stale. Build `--bin plowrt --example step_bench`. Before a gate or cell, confirm the binary carries
  the change (sha256 / mtime, or `grep -c '<new log string>' plowrt`). Gate digits identical to 4
  places across "two builds" mean the same binary or packet ran, not a neutral change.
* `interp_sm90a.cubin` is not byte/SASS-reproducible across nvcc runs for some configs: prove a
  "no-op for other packets" claim with `cuobjdump --dump-resource-usage`, a re-gate and step_bench
  token digests, not a cubin diff.
* A standalone CUDA-runtime `.cu` probe compiled inside `nix develop` aborts at runtime init
  ("stack smashing detected"): compile it in the plain shell, queue only the run. nvcc compiles
  that include `cuda_fp16.hpp` (TileLang, `build_catalog.py`) fail inside the nix shell because its
  ROCm `nv/target` shadows CUDA's: run them under `env -i` with `NVCC_PREPEND_FLAGS` (as
  `build_sm90a_cubin.sh` does) or in `nix develop .#tilelang`.
* No default-off knob ships: a knob is default on (qualified) or removed. Every new `PLOW_*` env
  read, emit knob or `#if PLOW_*` define is registered in `crates/plowrt/src/knob_spec.rs` or
  `crates/devgen/src/knob_spec.rs`, and `cargo test -p plowrt --features cuda,hsa --lib knob` +
  `cargo test -p devgen --lib knob` pass (they also refuse a removed knob set in `recipes/` or
  `scripts/`, and an unregistered name in `docs/flags-reference.md`).

**Runtime environment**

* plowrt binds the first loadable cuBLASLt: `<bin>/../lib`, `<bin>`, then the loader's
  `libcublasLt.so.13`. Packets that declare `runtime_requires.cublaslt = "13.4"` (Gemma-4 26B,
  grouped-Lt MoE routes) refuse older at load, a few seconds into the lease. On the lab box the
  loader's `.so.13` is `/opt/pytorch/cuda/lib` (13.1), so a bare `plowrt` under the CUDA 12.9 env
  fails: ship `lib/libcublasLt.so.13` (13.4) beside the binary as the deploy kits do, or put a 13.4
  lib dir first on `LD_LIBRARY_PATH`. The doctor checks this ([3]).
* cuBLASLt's load-time algorithm choice is timing-based and re-rolls per server start, prefill
  shapes included, so two starts of one build can differ in c1 numerics. For a digest or numerics
  A/B pin the algorithms: `campaign.py probe` (or `build` without `--no-probe`) writes
  `<assets>/cublaslt_algos.jsonl`, and plowrt serves with it as `PLOW_LT_ALGOS` whenever it sits in
  the assets dir (`asset::serve::asset_env_defaults`). A packet without the table (`--no-probe`, or a
  probe that died, e.g. on the cuBLASLt version above) re-rolls on every start. Diff the serve logs' `algorithm selected`
  lines before blaming a build.
* A serving plowrt with the host tier on auto pins host RAM down to the reserve (default tier
  = `min(MemAvailable − max(total/8, 16 GiB), total/4)`, `memory::vmm::host_tier_default`; the 12B
  recipe sets `PLOW_VMM_HOST_CACHE_MIB` 64 GiB). An `interp_sm90a` compile takes 22–38 GB per nvcc:
  never start an emit or a large cargo build beside a serving grid; chain it after.
* The prefix cache is always on (`PLOW_PREFIX_CACHE`, `PLOW_VMM_PREFIX`); `=0` is an attribution
  A/B, never a recipe or release setting. A prefix-cache cost is a plowrt bug to fix.
* Packets declare the attention route per rung (native role vs library objects in the packet, e.g.
  `PLOW_BUILD_FA4_HD512` puts `plow_fa4_hd512.aot`, the FA4 cubins plowrt launches itself, in
  it); plowrt runs what the packet declares. A route change is a re-emit and a re-gate, never a
  serve knob. An AOT library kernel ships as a packet cubin plus a host stub in plowrt
  (`device/cuda/aot.rs`), never as a `.so` (static runtime rule, "Working rules").

**KV reads and device faults**

* Under live KV (`PLOW_VMM_LIVE`) only rows `[0, kv_len)` of a slot are guaranteed mapped (4096-row
  commit blocks): a kernel must never load a KV row at or past `kv_len`, masked or not. Run
  `scripts/bench/step_boundary.sh` (ctx 4094/4095/4096/8191/8192, `CUDA_LAUNCH_BLOCKING=1`) on every
  new KV-reading arm before any serving run.
* A library attention route's page table must cover every entry the library's loader reads, not
  only the request's `n` pages: FA4's SM90 producer reads up to `n + splits - 2` entries of a row
  (and entry -1 when `n_block_max == 0`). Each row carries `MAX_SPLITS` padding columns of the
  request's own last page and the word before row 0 its first page. Stale entries point at pages of
  retired slots that are unmapped asynchronously, so an over-index faults rarely (CUDA 700 in the
  prod mix). Verify a new route with a poison table (entries past the row encoded as unmapped
  pages), which faults deterministically at 1-3-row tail chunks. A launch outside the library's
  envelope runs the native role.
* Diagnose a CUDA 700 with the kernel log (`sudo -n journalctl -k`; it works from the coordinating
  session, agent sessions cannot read it, so ask the coordinator): the NVRM Xid 31 line gives the engine (GRAPHICS = an SM kernel, CE = a copy), the access type
  (READ/WRITE) and the faulting VA; decode the VA against the serve log's KV pool window and track
  bases to name slot, head and row. Diagnostic error wrappers must pass `DeviceFault` through.

**Measurement**

* Production-mix screen = 1 rate x 2 reps (`PROD_RATES=<rate> llm_grid.sh ... --prod`, default
  `REPS=2`), reported with `campaign.py report --cells q<1000*rate>.g`. `REPS=1` cannot be reported
  (the report requires >= 2 repeats per cell); the full 2 rates x 2-3 reps is for promotion only.
* Reuse captured vLLM peers and shared plowrt base arms; never recapture a vLLM arm for a screen.

