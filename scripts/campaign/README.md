# Perf campaign harness

One recipe per measured cell. One command per stage. Every number carries its
recipe, commit, object hashes, GPU identity, protocol, and whether the lease
audit saw a co-tenant. `campaign.py --help` lists the subcommands.

```
python3 scripts/campaign/campaign.py build   <recipe.toml> --out <scratch>/<id>
python3 scripts/campaign/campaign.py bench   <recipe.toml> --assets <scratch>/<id>/assets --out <scratch>/<id>/c1
python3 scripts/campaign/campaign.py compare <scratch>/<id>/c1/results.csv perf-data/campaign/<ref>.csv
python3 scripts/campaign/campaign.py ledger  <scratch>/<id>/c1/results.csv --cell <cell> --note "<one variable changed>"
```

Policy (HARD RULE, vLLM-peer judging, GPU budget, gates, final-report format, A/B rule):
[docs/bringup/agent-tools.md](../../docs/bringup/agent-tools.md). Production recipes live in
`recipes/infervisor/<model>/<label>.toml`; experimental ones in `scripts/campaign/recipes/`.

## The method

1. **Fix the cell first.** A cell is `(model revision, weight/KV dtype, GPU SKU,
   TP, prompt/output shape, concurrency, cache policy, client protocol)`. The
   reference engine runs the same client binary with the same flags on the same
   box. A comparison across cells is not a comparison.
2. **The recipe is the only source of the build.** Emit knobs, flags, object
   build gates, role objects, the composed checkpoint, and serve-side mirrors
   live in the TOML — not in a shell history.
3. **Fail closed before timing.** `bench` refuses without `model.pkt`, copies the
   runtime binary so a concurrent `cargo build` cannot swap it mid-run, runs the
   coherence gate before the first timed request, and exits non-zero if the gate
   fails. A model that answers the gate wrong has no numbers.
4. **Every GPU stage is a gpuq job.** `probe`, `bench`, `gate`, `dist-gate`, `block-bench`,
   `block-ab`, `nv-block-ab` and `serve-bench` submit to `scripts/bench/gpuq.py` (`--queue`,
   default `/tmp/plow-gpuq`) and wait; the job's lease wraps server and client together, and its
   wrapper exports every `PLOW_*` knob because the runner strips them. If the audit sees a foreign
   process the run is recorded `contended`; `ledger` refuses it unless `--provisional`.
5. **One variable per experiment.** `ledger --note` is mandatory and names the
   single change. Rejected candidates stay in the ledger so they are not re-run.
6. **Price against ceilings before building.** For decode the weight-stream floor
   (`bytes / HBM BW`); for prefill the library ceiling (cuBLASLt / FA). Kernel
   work goes to the top-two stages of the attribution, never to a hunch.
7. **Memory is a column.** `bench` drives `scripts/bench_plowrt_serve.sh`, which samples the
   server's process tree with `nvidia-smi` (`MEM_SAMPLE_MS`, default 1000, `0` = off); `bench`
   writes the per-cell peak as `peak_mem_mib` in `results.csv` and the ledger. Both engines
   preallocate weights + KV pools, so the number is the configured footprint plus transient
   workspace: compare it at equal `max_ctx` / slot count, and quote the KV budget beside it. On the
   plow arm host resources are sampled too (`HOST_SAMPLE_MS`, default 2000,
   `scripts/bench/host_res.py`) and land in `run-record.json` under `host_resources`.
8. **Quiet box for report-grade cells.** `bench --quiet-lock FILE` holds `FILE` exclusively for
   the whole session through `scripts/bench/quietx.sh`, taken INSIDE the GPU lease; builds and
   other CPU-heavy work hold it shared through `scripts/bench/quiets.sh FILE nice -n 19 <cmd>`.
   `quietx.sh` holds `FILE.gate` while it waits, so builds queue behind a waiting session instead
   of starving it; `quiets.sh` closes the descriptor before exec (`flock -o`), so a daemon the build
   spawns (the sccache server) cannot keep the lock after the build ends. Lease first, then lock,
   everywhere: a run holding the lock while queued for the GPU deadlocks against a run that holds
   the GPU and wants the lock.

## Subcommands

| subcommand | inputs | what it freezes / checks | outputs |
|---|---|---|---|
| `build <recipe> --out DIR` | recipe; `--env K=V` (emit), `--object-env K=V` (object build only), `--rung-env PHASE:ROWS:K=V`, `--block L[..R]`, `--hf-dir`, `--no-probe` | runs the prebuilt `plowc` (never cargo) base emit → objects → role emit, in that order because role objects are validated from the emit `--out` directory; `--lean-qualify strict` for a `recipes/infervisor` recipe; then the queued cuBLASLt probe unless `--no-probe` | `DIR/assets`, `build-record.json` (every hash, every override), `lean-qualification.json`, `<assets>/cublaslt_algos.jsonl` |
| `qualify --assets DIR [--strict]` | bundle | `plowc qualify`: receipts, approved verifier, replay, media geometry | `lean-qualification.json`; `--strict` exits non-zero unless every packet qualifies |
| `probe <recipe> --assets DIR` | built assets | serves once with `--lt-algos-write`, waits for load, answers the coherence gate | `<assets>/cublaslt_algos.jsonl` |
| `serve <recipe> --assets DIR --profile P` | profile from `[bench.profiles.*]` | production `plowrt serve` in the foreground; no lease, no client; `--dry-run` prints env and command | — |
| `bench <recipe> --assets DIR --out DIR` | `--profile`, `--concs`, `--in-lens`, `--nprompt`, `--env`, `--quiet-lock`, `--reference` | copies the binary, coherence gate, lease audit | `results.csv` (+ `peak_mem_mib`), `run-record.json`; compares against `[reference].csv` |
| `compare <results.csv> <reference.csv>` | two CSVs (`perf-data/campaign/` holds reference CSVs) | diagnostic only, matches on `(input_len, concurrency)` | table on stdout |
| `ledger <results.csv> --cell C --note N` | a bench run | refuses a contended run unless `--provisional` | ledger row |
| `roofline`, `loop`, `sweep` | recipe | roofline report; closed loop doctor → build → probe → bench → roofline → compare; one-parameter sweep | stdout / run dirs |
| `rung-sweep` | recipe, `--knob`, `--values`, `--stage` | "Per-rung knob choices" below | `choice.md`, `recipe-snippet.toml`, ... |
| `gate <recipe> --assets DIR --out DIR` | recipe `[gates]`; `--only`, `--model`, `--dry-run`, `--score-only` | runs every gate in one queued job | `gates.json` (records `packet_sha256`) |
| `report --baseline R --infervisor R --gate gates.json --out DIR` | two `llm_grid.sh` result dirs + gate | strict 12-row render (`scripts/bench/serving_comparison.py`) | `comparison.md/.json/.csv` |
| `cert --knob K --job J --cell C --ctrl .. --ctrl2 .. --treat .. --treat2 .. --out DIR` | four bench runs | checkpoint P certificate via `scripts/perf_cert.py make` | `ledger.jsonl`, `request.json` |
| `dist-gate <recipe> --assets DIR --out DIR [--gate]` | built assets | push → clean `PLOW_HOME` load → bundle == push input → queued `serve --model` (smoke or `[gates]`) | `dist-gate.json` |
| `nv-block-ab --control A --treatment B --verb bench\|prefill-bench --args ".." --out DIR --note N` | two `build --block` asset dirs; `--ceilings`, `--router-table` | four frozen arms in one queued lease | `comparison.md` (block-vs-roofline table + verdict) |
| `block-roofline`, `block-bench`, `block-ab`, `serve-bench` | AMD block packets / ROCm serving sets | see "AMD" below | — |

Per-subcommand flags: `campaign.py <subcommand> --help`.

## Recipe schema

The schema is what `scripts/check_recipe.py` (with `scripts/recipe_common.py`) accepts; start a
new recipe from `scripts/campaign/recipes/templates/llm-sm90a-default-on.toml`.

## Profiles

A cell carries named workloads under `[bench.profiles.<name>]`; `bench --profile <name>` (and
`serve --profile`) applies its keys over `[bench]` and its `serve_env` over `[serve].env`. Typical
profiles: `realtime` (owner metric TTFT + TPOT, per-token ITL, low concurrency) and a throughput
profile (`throughput` / `high_concurrency`: output tok/s and TPOT under load).

A throughput run needs a packet whose decode ladder covers the concurrency
(`PLOW_DECODE_BATCH_LADDER`) and the KV to back it, i.e. an unshared GPU. The reference CSV holds
every profile's rows; `compare` matches on `(input_len, concurrency)`. Never table a row against a
different precision.

## Per-rung knob choices

Rung-eligible emit knobs are the `.per_rung(...)` entries in `crates/devgen/src/knob_spec.rs`
(policy: agent-tools.md "Per-rung gate and recipe selection"). One can take a different value per
prefill bucket (explicit GLU roles still win at their rung via `select_fused`):

```toml
[emit.env]
PLOW_NO_GLU_FUSE = "1"                  # packet-wide
[emit.rung_env.prefill.128]
PLOW_NO_GLU_FUSE = "0"                  # rung 128 only
```

`build` renders the table into the one `PLOW_EMIT_RUNG_ENV` both emits get
(`prefill:128:PLOW_NO_GLU_FUSE=0;...`, plus `--rung-env phase:rows:K=V` A/B overrides) and refuses
it from any other source. devgen lowers each rung under its resolved config, refuses packet-wide
knobs and overrides that match no emitted rung, and writes the per-program resolution to build.json
`rung_knobs`; `check_recipe.py` holds the recipe to it.

`rung-sweep` chooses the value per rung. Four CPU/GPU stages, one queue job:

```
campaign.py rung-sweep <recipe> --knob PLOW_NO_GLU_FUSE --values 0,1 --rungs 128,1024,8192 --out DIR --stage build \
    [--reference-build <build of the unchanged recipe>]
campaign.py rung-sweep ... --stage run --run-env LD_LIBRARY_PATH=... --run-env PLOW_LIVE_CTX=16384
scripts/bench/gpuq.py submit rung-sweep 1 timeout <s> bash DIR/run.sh
campaign.py rung-sweep ... --stage score       # choice.md / choice.json / resources.md / recipe-snippet.toml
campaign.py rung-sweep ... --stage verify      # mixed packet; per-rung program == chosen arm; verify-run.sh
```

* build: one packet per value with the value on every prefill rung (`--rung-env`), so arm v's
  rung program is the program a mixed recipe gets at that rung (verify proves it on the mixed
  packet by per-rung program digest: kernel_cases + dispatch chain).
* run: step_bench packed prefill at exactly the rung's rows (`slots x ctx` within the request
  chunk; the score refuses a launch of any other width), graph on, `--reps` per run with rep 0
  (capture) dropped, ABBA blocks, the packet's serve defaults exported (`--no-serve-env` drops
  them). The same launches dump the compact-terminal logits.
* score: per rung, the reference value is the reference build's resolution, and the reference
  arm's program must equal the reference build's. Another arm wins only if every run has exactly
  `--reps` reps at the rung width, every block's logits (reference too) are complete, equally sized
  and finite, its program differs, and it is faster under the A/B rule over all block medians (no
  floor). resources.md lists REG/STACK/SHARED/LOCAL for every object that differs between arms.
* Label: `experimental` unless accuracy is accepted (bitwise, or `--kl-max` and `--top1-min` both
  given and passed), no switched arm's object spills, and `--roofline <file>` (calibrated
  bandwidth/compute bound per rung) and `--library-resources <file>` (library kernels, e.g.
  cuBLASLt, which cuobjdump of our objects does not cover) are given. The snippet header carries the
  label.
* Promotion: nothing here edits a recipe. A qualified switch needs verify, the re-timed switched
  rungs and the mixed packet's own full gates (`campaign.py gate`, llm_fp32_ref included); then the
  snippet goes into the canonical recipe.

## Production bucket demand

`bucket_demand.py <metrics URL or file> [<later file>]` reads a serving plowrt's `plowrt_bucket_*`
series (`docs/runtime/host-path.md` "Bucket demand") and prints, per model and launch family, the
declared capacities, the row counts live traffic launched with, the padded fraction, and the 1 to
3 capacities whose addition removes the most padded rows. CPU only; it changes no recipe.

## AMD

AMD (GLM) block harness and matched ROCm serving: [docs/bringup/05-single-block-sweep.md](../../docs/bringup/05-single-block-sweep.md)
"AMD (GLM) block harness". One command each:

```
python3 scripts/campaign/campaign.py block-roofline <recipe.toml> --packet <block.pkt> --program 1 --ctx 8192 --out <scratch>/roofline
python3 scripts/campaign/campaign.py block-bench <recipe.toml> --packet <block.pkt> --objects <objects> \
  --checkpoint <checkpoint> --inputs <operands> --ctx 512 --out <scratch>/block-control
python3 scripts/campaign/campaign.py block-ab <recipe.toml> --control-build <control> --treatment-build <candidate> \
  --inputs <operands> --checkpoint <checkpoint> --ctx 512 --out <scratch>/ab --note "one changed lever"
python3 scripts/campaign/campaign.py serve-bench scripts/campaign/recipes/glm53.mi350x.fp8-full.toml \
  --server plow|vllm --assets <assets> --objects <objects> --out <scratch>/run --in-lens 8192 --concs 8 --nprompt 16
```

Accept a block `PASS` only after the enclosing queue job exits 0: the lease's final contention
audit can still invalidate an otherwise passing arm.

## Known protocol traps

- plowrt names a model by checkpoint slug, not the HF repo id (`model_id`).
- plowrt `/v1/completions` adds no BOS. Use a chat-formatted `gate_prompt` with explicit special
  tokens.
- The HF cache root may be unreadable; pass the tokenizer by snapshot path.
- Packed prefill support differs per target and precision: check `plowrt inventory <assets>`
  before assuming it is on.
