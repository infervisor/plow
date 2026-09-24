# Packet recreate records

One `<label>.json` per packet the Gemma-4 H100 campaign used, plus that packet's
`<label>.cublaslt_algos.jsonl`. Together with the recipe and `tuning/` — both already in the repo
— these are what recreate the packet and its cubins.

The packets themselves live under `/opt/dlami/nvme/tmp`, which is scratch. The cubins, `model.pkt`
and `build.json` do not survive a cleanup, and `build.json` is the only record of what produced
them. These files are the durable half.

## Why the recipe alone is not enough

A packet is a recipe **plus** `--env` overrides, and recipes change. Of the eleven packets here,
ten were built from `bf16-l8192-16k.toml` (or its 26B twin) with between one and five overrides —
including `p12rq`, which was compared all session against packets built from
`bf16-c32-req1k-16k.toml`. That comparison was ten build-keys wide, not one.

So each record carries the ground truth from `build.json`:

| field | what it is |
|---|---|
| `emit_env_replay` | `emit_config.replay` — env that reproduces the configuration, defaults omitted so a replay follows the tree's defaults |
| `unrecorded_env` | `emit_config.unrecorded_env` — env read *outside* `EmitConfig` (the `PLOW_SEG_*` family), which `replay` cannot describe |
| `tuning` | shape constants the tune store resolved (`gv_mm_max`, `xreg_k`, `fa_spart`, …) |
| `registry_digest` | verifies a rebuild resolved the same knob registry |
| `artifacts` | every emitted cubin / `model.pkt` / header with size and sha256 |
| `recipe`, `recipe_overrides` | **derived** by matching `emit_env_replay` against the recipes present today; `recipe_match` says how much to trust it |
| `build_mtime` | when the packet was emitted. `preserved_at_commit` is HEAD when the record was written and is explicitly **not** the build commit — `build.json` carries no commit |

## Recreating a packet

```sh
nix develop --command python3 scripts/campaign/campaign.py build <recipe> \
    --out <dir> --no-probe --env K=V ...        # exactly rebuild_command in the record
```

Then verify `registry_digest`, `tuning` and `shapes` against the record. A mismatch means the tree
moved under you; check out a commit near `build_mtime` and retry. `--no-probe` reuses the tune
store in `tuning/`; the packet's own `cublaslt_algos.jsonl` is preserved beside its record because
a fresh probe can pick different algorithms.

## Before quoting any A/B

```sh
nix develop --command python3 scripts/campaign/preserve_packet.py diff <pktA> <pktB> \
    --expect PLOW_MAX_REQUEST_CHUNK     # exit 2 unless that is the ONLY difference
```

Three results in this campaign were written up as single-variable A/Bs and were not — `GV_MM_MAX`
(a "0.23–0.37 ms tax" a real A/B put at 0.7%), the ring tax quoted in two recipe headers, and the
`req_chunk` result. Each was a cross-packet delta. `diff` is one second and would have caught all
three; run it before the lease, not after the write-up.

## The set

| label | cell | slots/chunk | req_chunk | digest | built | ovr | recipe |
|---|---|---|---|---|---|---|---|
| `ab_r2048` | 12b.bf16-l8192-16k | 32/1024 | — | `29252ee3` | 2026-09-21 | 3 | `gemma4-12b.h100.bf16-l8192-16k.toml` |
| `ab_r4096` | 12b.bf16-l8192-16k | 32/2048 | — | `29252ee3` | 2026-09-21 | 3 | `gemma4-12b.h100.bf16-l8192-16k.toml` |
| `p12-ladder16k` | 12b.bf16-l8192-16k | 16/16384 | — | `4faaa1e0` | 2026-09-20 | 4 | `gemma4-12b.h100.bf16-l8192-16k.toml` |
| `p12fp8c` | 12b.fp8-ladder16k | 16/4224 | — | `ce9448d8` | 2026-09-23 | 3 | `gemma4-12b.h100.fp8-ladder16k.toml` |
| `p12l8` | 12b.bf16-l8192-16k | 16/8192 | 4224 | `e1422a5f` | 2026-09-21 | 1 | `gemma4-12b.h100.bf16-l8192-16k.toml` |
| `p12rq` | 12b.bf16-l8192-16k | 32/4096 | 1024 | `29252ee3` | 2026-09-21 | 5 | `gemma4-12b.h100.bf16-l8192-16k.toml` |
| `p12rw` | 12b.bf16-l8192-16k | 16/4224 | — | `60f9b31d` | 2026-09-22 | 3 | `gemma4-12b.h100.bf16-l8192-16k.toml` |
| `p26-c32` | 26b-a4b.bf16-l8192-16k | 32/4096 | 1024 | `29252ee3` | 2026-09-21 | 5 | `gemma4-26b-a4b.h100.bf16-l8192-16k.toml` |
| `p26l8` | 26b-a4b.bf16-l8192-16k | 16/8192 | 4224 | `e1422a5f` | 2026-09-21 | 2 | `gemma4-26b-a4b.h100.bf16-l8192-16k.toml` |
| `p26l8r` | 26b-a4b.bf16-l8192-r3072-16k | 16/8192 | 3072 | `e1422a5f` | 2026-09-21 | 2 | `gemma4-26b-a4b.h100.bf16-l8192-r3072-16k.toml` |
| `rc2048` | 12b.bf16-c32-req1k-16k | 32/4096 | 2048 | `ce9448d8` | 2026-09-23 | 2 | `gemma4-12b.h100.bf16-c32-req1k-16k.toml` |

`ab_r2048`/`ab_r4096` are the ring A/B pair and are the one verified single-variable pair in the
set (`diff --expect PLOW_MAX_CHUNK` exits 0). Note the five distinct `digest` values: these
packets were built against **five different knob-registry states**, so any comparison spanning two
digests is comparing two compilers as well as two configurations.

## Adding a packet

```sh
nix develop --command python3 scripts/campaign/preserve_packet.py preserve <packet-dir> --label NAME
```

Do this before the scratch directory is cleaned, i.e. as part of the run that produced it. Records
are small (~7 KB each, plus a ~20 KB algos file); the 44 MB `model.pkt` is recorded by sha256, not
copied.
