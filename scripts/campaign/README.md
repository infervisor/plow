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

## Profiles: realtime and throughput are both first-class

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
