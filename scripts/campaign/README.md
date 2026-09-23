# Perf campaign harness

One recipe per measured cell. One command per stage. Every number carries its
recipe, commit, object hashes, GPU identity, protocol, and whether the lease
audit saw a co-tenant. This is the contract for every performance claim.

```
python3 scripts/campaign/campaign.py build   recipes/<cell>.toml --out /nvme/run/<id>
python3 scripts/campaign/campaign.py bench   recipes/<cell>.toml --assets /nvme/run/<id>/assets --out /nvme/run/<id>/c1
python3 scripts/campaign/campaign.py compare /nvme/run/<id>/c1/results.csv perf-data/campaign/<ref>.csv
python3 scripts/campaign/campaign.py ledger  /nvme/run/<id>/c1/results.csv --cell <cell> --note "<one variable changed>"
```

## The method

1. **Fix the cell first.** A cell is `(model revision, weight/KV dtype, GPU SKU,
   TP, prompt/output shape, concurrency, cache policy, client protocol)`. The
   reference engine runs the same client binary with the same flags on the same
   box (`bench_vllm26_native.sh` and `bench_plowrt_serve.sh` share the client).
   A comparison across cells is not a comparison.
2. **The recipe is the only source of the build.** Emit knobs, flags, object
   build gates, role objects, the composed checkpoint, and serve-side mirrors
   live in the TOML — not in a shell history. `build` runs base emit → objects →
   role emit in that order because role objects are validated from the emit
   `--out` directory; it writes `build-record.json` with every hash.
3. **Fail closed before timing.** `bench` refuses without `model.pkt`, copies the
   runtime binary so a concurrent `cargo build` cannot swap it mid-run, runs the
   coherence gate before the first timed request, and exits non-zero if the gate
   fails. A model that answers the gate wrong has no numbers.
4. **Every GPU stage is leased.** `perf-data/tools/gpulease -n <k>` wraps the
   server and client together. If the audit sees a foreign process the run is
   recorded `contended`; `ledger` refuses it unless `--provisional`.
5. **One variable per experiment.** `ledger --note` is mandatory and names the
   single change. Rejected candidates stay in the ledger so they are not re-run.
6. **Price against ceilings before building.** For decode the weight-stream floor
   (`bytes / HBM BW`); for prefill the library ceiling (cuBLASLt / FA). Kernel
   work goes to the top-two stages of the attribution, never to a hunch.
7. **Key everything on geometry.** Roles, tiles, and objects are selected by
   `(arch, dtype, M, N, K, head geometry, KV bucket)` through tunedb, never by a
   model name or a literal `hidden == N`. A new variant re-tunes by running the
   campaign, not by editing the emitter.
8. **Memory is a column.** Both bench scripts sample the server's process tree with
   `nvidia-smi` (`MEM_SAMPLE_MS`, default 1000, `0` = off) and `bench` writes the per-cell peak
   as `peak_mem_mib` in `results.csv` and the ledger (a ledger from before the column is widened
   in place). Both engines preallocate weights + KV pools, so the number is the configured
   footprint plus transient workspace: compare it at equal `max_ctx` / slot count, and quote the
   KV budget beside it.
9. **Quiet box for report-grade cells.** `bench --quiet-lock FILE` holds `FILE` exclusively for
   the whole session through `scripts/bench/quietx.sh`, taken INSIDE the GPU lease; builds and
   other CPU-heavy work hold it shared through `scripts/bench/quiets.sh FILE nice -n 19 <cmd>`.
   `quietx.sh` holds `FILE.gate` while it waits, so builds queue behind a waiting session instead
   of starving it; `quiets.sh` closes the descriptor before exec (`flock -o`), so a daemon the build
   spawns (the sccache server) cannot keep the lock after the build ends. Concurrent compiles
   inflated a CPU-bound 128-token vLLM TTFT 31 -> 45 ms while leaving GPU-bound TPOT alone. Lease
   first, then lock, everywhere: a run holding the lock while queued for the GPU deadlocks against
   a run that holds the GPU and wants the lock.

## Profiles: realtime and throughput are both first-class

### AMD modular blocks

`serve-bench --quality-lens 8192,71680` runs the existing needle probe before
timing, using exact token-ID prompt lengths and recording prompt hashes. Compare
the achieved lengths and hashes across servers before interpreting paired retrieval
results. Retrieval results do not replace the full-logit numerical gate.

`block-roofline` reads one disassembled decode program, preserving its TP shapes,
FP8 scale grids and sparse selection width. The result is an optimistic HBM/compute
floor, with excluded operations listed; configured ceilings are not same-session
bandwidth measurements.
Add `--router-table /path/outputs/rank0.act.tab.bin` to charge the captured
expert union instead of assuming every row shares the same top-k experts. The
table's shape, IDs, gates and per-row uniqueness are checked and its hash recorded.
This remains a one-stream traffic model, not measured HBM utilization.

```
python3 scripts/campaign/campaign.py block-roofline recipes/<block>.toml \
  --packet /path/block.pkt --program 1 --ctx 8192 --out /path/roofline
python3 scripts/campaign/campaign.py block-bench recipes/<block>.toml \
  --packet /path/block.pkt --objects /path/objects --checkpoint /path/checkpoint \
  --inputs /path/operands --ctx 512 --out /path/block-control
```

Run inside `nix develop`, with render-device access. `block-bench` freezes the
packet, objects, input fixtures and runtime, runs the doctor, then submits to
`scripts/bench/gpuq.py`. The FIFO worker waits for foreign GPU processes to exit
before acquiring a lease. Inspect `gpuq.py status` and the job's log in
`/tmp/plow-gpuq`; a queued job is not a measurement.

Add `--dstep-log` for host preparation/submission/wait/audit timing windows.
These are host-observed intervals: rearming can overlap execution, and drain is
the remaining wait, not total GPU duration. `block-ab` rejects these diagnostic timings.

Add `--trace` for a separate instrumented run with per-rank device traces; do not
score its host timings against uninstrumented controls. Failed numerical gates
leave intermediate dumps in `outputs/` but no timing report. Diagnose those with
`glm52_real_oracle.py --candidate-dir DIR` (CPU, no fixture written).

`build` also supports gfx recipes with `[objects].script = "scripts/build_gfx950.sh"`.
It passes the freshly emitted `plow_config.h` to the object build and records
packet, manifest, header and object hashes. `block-bench` preserves the matched
manifest when its packet comes from such a campaign build, rejecting changed assets.
Use `build --object-env K=V` for an object-only A/B; this is recorded separately
from emit-side `--env` and does not change the packet configuration.

Use `block-ab` to keep a rung's four arms in one queued lease (not four separate
submissions). It freezes each arm, runs the doctor, takes the CPU-quiet lock inside
the lease, and runs control/candidate/control/candidate. Builds must cooperate via
`scripts/bench/quiets.sh /tmp/plow-campaign-quiet.lock` for CPU isolation.

```
python3 scripts/campaign/campaign.py block-ab recipes/<block>.toml \
  --control-build /path/control --treatment-build /path/candidate \
  --inputs /path/operands --checkpoint /path/checkpoint --ctx 512 \
  --out /path/ab --note "one changed lever"
```

The frozen scorer refuses mismatched cells/inputs/runtime, instrumented timings,
failed numerical gates and nonrepeatable same-variant outputs. A block `PASS`
also requires identical control/candidate output files when `--require-bitwise`
is set (use for scheduling-only levers). Otherwise the captured oracle gates apply.
A performance `PASS`
requires a median saving above `max(control drift, treatment spread) + 2*max(MAD)`
and treatment spread at most three times control drift. Control/control therefore
cannot pass as a performance improvement. Failure exits nonzero; results remain
in `comparison.json`. This is a block-candidate gate, not serving/default qualification.
For the experimental native routed W8A8 path, use `--routed-reference audit.json`
instead of `--require-bitwise`. The audit must be a passed pinned connected
`--block-routed --routed-w8a8 --routed-down-isolate` capture. The campaign freezes
it and runs the CPU checker in the pinned Docker image after all four arms.
It checks route coverage and gate bits, five bitwise stable boundaries, BF16
atomic addition-order bounds, and exact shared/TP/residual rounding. Only those
validated reordered/atomic-dependent outputs may differ; every other captured
output must match across all arms. The certificate is bound to all output hashes.
This does not establish full-model precision parity or relax the block oracle.
Accept `PASS` only after the enclosing queue job finishes with return code zero;
the lease's final contention audit can still invalidate an otherwise passing arm.

The capture runner supports fixed-width decode blocks, with capture batch equal
to the packet's allocated decode batch. Fixtures contain
raw `act.x.bin`, the block's carried `kv.*.bin` / `act.iidx.bin`, and
`reference.bf16` with `reference.json` specifying `batch` (default 1), `ctx`,
`tolerance_rel_l2` and stage names. KV files pack `[batch, ctx, dimension]`;
replay uploads each sequence at the packet's full-context stride.
`runtime/tests/glm52_real_oracle.py --block-inputs DIR` exports the layer-3
dense-context gate (context at most the DSA selection width). Add
`--batch 8 --inputs-only` for distinct seeded sequences without duplicating the
weight-bearing fixture. Every row, stage and rank must pass before timing;
TP counter audits remain enabled. Measurements
are explicitly host-clock dispatch/drain timings, not device kernel timings.
For a shared-index block, `--block-context 71680` with `GLM_L=2048` places supplied
keys across the logical context and uses the same positions in the HF reference.
This checks sparse gathers and RoPE, not the learned indexer's choices. Prefill
still requires its own captures and runner support. Block results are never served tok/s.

### Matched ROCm serving

`serve-bench` freezes a full-model serving set and queues either plow or the pinned
vLLM 0.29 Docker image. Both use the same digest-pinned Docker client, tokenizer,
random seed, raw-completion protocol, output length and concurrency grid.

```
python3 scripts/campaign/campaign.py serve-bench recipes/glm53.mi350x.fp8-full.toml \
  --server plow --assets /path/assets --objects /path/objects --out /path/plow-run \
  --in-lens 8192 --concs 8 --nprompt 16
```

Repeat with `--server vllm` and a fresh output directory. Run inside `nix develop`
with render access and noninteractive Docker access. The host paths currently
assume `/opt/models`. `--dry-run` prepares and checks the shell without queueing.
Missing raw checkpoint shards are refused; verify download checksums and prepare
every layer before running. Runtime and objects are private copies. Teardown
signals only the owned timeout/server or stops the exact created container ID.

Client results must contain every requested completion/output token and finite
mean/median/P99 latency metrics. `run-record.json` deliberately leaves
`numerics_qualified=false`: a coherence smoke is not full-model numerical or
retrieval qualification. The initial full recipe uses B16 capacity; C32/C64 can
queue and are not evidence that all requests reside in KV simultaneously.

A cell carries named workloads under `[bench.profiles.<name>]`; `bench --profile
<name>` applies its keys over `[bench]` and its `serve_env` over `[serve].env`.

| profile | owner metric | cells | serve |
|---|---|---|---|
| `realtime` | TTFT + TPOT, per-token ITL | C1, C4 at in 128/1024/4096 | `PLOW_MULTISTEP=0` |
| `throughput` | output tok/s, TPOT under load | C4, C16 at in 1024/4096 (extend to 8192+ with a matched reference) | `PLOW_MULTISTEP=8` |

A throughput run needs a packet whose decode ladder covers the concurrency
(`PLOW_DECODE_BATCH_LADDER=1,2,4,8,16`) and the KV to back it (≈41 GiB at
ctx 8192 for Gemma-4-12B), i.e. an unshared GPU. The reference CSV holds both
profiles' rows; `compare` matches on `(input_len, concurrency)`. Never table a
row against a different precision.

## Recipe schema

```toml
[cell]        name, model, revision, hf_dir, checkpoint_dir?, gpu, arch, n_cu, n_gpu?, max_ctx, precision
[emit]        emit?, args = [...], env = { PLOW_... }
[objects]     script, env = { PLOW_BUILD_... }, role_files = [...]      # optional
[emit_roles]  env = { PLOW_..._ROLE = 1 }                               # optional, merged over [emit].env
[serve]       env = { PLOW_PF_SEG_DIR = "...", ... }, extra_args?, plowrt?
[bench]       backend, in_lens, concs, nprompt, outlen, warmups, seed, gate_prompt, tokenizer, vllm_venv?, port?, model_id?
[reference]   csv                                                        # same CSV schema as results.csv
```

## Known protocol traps (each cost a run once)

- plowrt names a model by checkpoint slug, not the HF repo id (`model_id`).
- plowrt `/v1/completions` adds no BOS; Gemma degenerates without it. Use a
  chat-formatted `gate_prompt` with explicit special tokens.
- The HF cache root may be unreadable; pass the tokenizer by snapshot path.
- plowrt validates a packet's cuBLASLt segments with the same `plow-asset`
  admission function the emitter used, so any change to that policy needs
  `plowrt` rebuilt before the probe or bench — a stale serve binary refuses the
  packet with "invalid packet-declared projection segments".
- Long benches must run detached (setsid/nohup, watch the results file): the
  CLI's low-memory guard kills tracked background commands while a server pages
  24 GB of weights into the page cache, even with 240 GB available.
- Queue scripts that wait on process names must spell the pattern so their own
  command line does not match (`serv[e]`), and a helper must not carry the
  pattern it kills in its own name or arguments.
- `PLOW_UNISEG=1` is not needed for BF16 sm90a Gemma; the emit audit's "impure
  flash segment" warning is an AMD relaunch concern.
- Packed prefill is default-on only for BF16 Hopper packets; W8A8 needs
  `PLOW_EMIT_PACKED_PREFILL=1` explicitly or the roles have no metadata.
- **Prefix cache.** plowrt serves with the prefix cache ON by default; vLLM-bench's random
  prompts share long prefixes, so cells silently become cache-hit suffixes (a 128-token
  request ran as a 39-row token batch after a 96-row hit). The vLLM references ran with
  prefix caching disabled: every vLLM-matched recipe sets `PLOW_PREFIX_CACHE=0`.
- A W8A8 packet loads `fp8/…` twins from any safetensors in `checkpoint/`;
  compose a directory (BF16 shards + `fp8-model.safetensors`) and point
  `checkpoint_dir` at it.
