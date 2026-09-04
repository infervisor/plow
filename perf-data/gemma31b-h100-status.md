# Gemma-4-31B / H100 prefill baseline — status (resumable)

Tracks progress on the plan at `~/.claude/plans/abundant-roaming-squirrel.md`
(Gemma-4-31B prefill: vLLM baseline vs plow, Phase 1 — benchmark only, no
tuning). Update this file at every checkpoint; commit after each phase.

Box: H100 80GB HBM3 (single GPU, `nvidia-smi -L`), driver 595.91.07.
Fresh machine for plow: no nix, no cargo/rustc, no CUDA toolkit for
`shaswot`, no `/workspace` at session start (2026-09-04).

## Phase 0 — environment bring-up: IN PROGRESS

- [x] Install Nix (Determinate installer, multi-user daemon) — `nix (Determinate Nix 3.22.3) 2.35.2`.
      Must `source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh` each new shell.
- [x] `nix develop -c cargo build --release -p plowc` — built, `target/release/plowc` runs.
- [ ] `nix develop -c cargo build --release -p plowrt --features cuda` — running in background
- [x] ACL grant on `/opt/dlami/nvme/hf-cache/hub/models--google--gemma-4-31B-it`
      (`sudo setfacl -R -m u:shaswot:rX ...` on the model dir, plus
      `sudo setfacl -m u:shaswot:x /opt/dlami/nvme/hf-cache` for traversal —
      `hub/` itself is already `o+rx`). Verified readable (config.json + a
      safetensors shard). Checkpoint dir:
      `/opt/dlami/nvme/hf-cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475`
      (2 safetensors shards, standard HF layout).
- [x] Install CUDA toolkit ≤13.2 (cuda-keyring apt repo already configured) —
      `cuda-toolkit-13-2` (13.2.86) installed at `/usr/local/cuda-13.2`,
      `nvcc --version` confirms release 13.2, matches driver 595.91.07.
- [x] Build sm_90a cubins: `scripts/build_sm90a_cubin.sh` run OUTSIDE
      `nix develop` with `env -i PATH=/usr/local/cuda-13.2/bin:/usr/bin:/bin
      PLOW_NVCC=/usr/local/cuda-13.2/bin/nvcc`. Both objects built + kernel
      symbols verified present:
      `assets-run/gemma4-31b-bf16/interp_sm90a.cubin` (1.30 MB, decode) and
      `interp_sm90a_pf.cubin` (678 KB, prefill).
- [x] Emit plow bf16 asset:
      `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33 ./target/release/plowc --hf-dir <ckpt>
      --emit devblob --gpu h100 --arch sm_90a --max-ctx 8192
      --out assets-run/gemma4-31b-bf16` — succeeded: 60 layers, weights
      57.2 GiB, KV cache 2.19 GiB (matches `coldstart-plow-vs-vllm-gh200.md`
      §1b exactly). **IMPORTANT for benchmarking**: `weights.json`'s
      `"network"` field (the model-routing key clients must pass as
      `"model"`) is `842da3794eaa0b77d5f08bae87a17459d91ff475` (the HF
      snapshot dir's commit-hash basename, not a friendly name) — use that
      exact string as `--model` in `vllm bench serve` calls against plow.
- [x] `plowrt serve --assets assets-run/gemma4-31b-bf16 --port 8090` — running
      as PID (see `ps`/`assets-run/plowrt-serve.log`), healthy: weights
      uploaded in 5.8s (9.92 GiB/s), `GET /v1/models` returns 200, model slug
      `842da3794eaa0b77d5f08bae87a17459d91ff475` (== the checkpoint's HF
      snapshot commit hash — pass this as `"model"` in every plow request).
- [x] Correctness gate: `libcuda.so.595.91.07` confirmed mapped into the
      plowrt process AND registered as a live compute app via `nvidia-smi
      --query-compute-apps`; greedy "Paris" — exact; bicycle-balance
      paragraph — coherent, on-topic, correctly terminated at `max_tokens=80`
      (no prior-baseline text to exact-match against — this is the first-ever
      build for this model, so coherence is the applicable bar, not
      bit-exact refactor parity).

**Phase 0 COMPLETE.**

## Phase 1 — vLLM baselines (bf16, fp8-weights): NOT STARTED

Base bf16 flags (from `/etc/systemd/system/gemma-31b.service.bak-pre-fp8kv`):
`--dtype bfloat16 --gpu-memory-utilization 0.95 --max-model-len 8192`, no
kv-cache-dtype override, no prefix-caching, no chunked-prefill.
fp8 leg = same flags + `--quantization fp8` (weight quantization, NOT KV
cache — explicitly ruled out by user).

- [ ] bf16 leg: edit `gemma-31b.service` ExecStart, restart, wait healthy
- [ ] bf16 leg: `vllm bench serve` sweep at input-len 2048/8192/16000,
      concurrency 1, `--random-output-len 8 --ignore-eos --num-prompts 5
      --seed 0`
- [ ] fp8 leg: edit ExecStart (+`--quantization fp8`), restart, wait healthy
- [ ] fp8 leg: same sweep
- [ ] Restore `gemma-31b.service` to its original (pre-session) content,
      restart, confirm healthy — MANDATORY before ending session

## Phase 2 — plow bf16 baseline: NOT STARTED

- [ ] Same `vllm bench serve` sweep against plow on :8090
- [ ] Re-check correctness gate before trusting numbers

## Write-up: NOT STARTED

- [ ] `perf-data/gemma4-31b-h100-prefill-baseline-<date>.md` — headline table,
      correctness discipline, claims/non-claims, open items (Phase 2 tuning)

## Blockers / notes log

(none yet)
