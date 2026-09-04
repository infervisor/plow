# Kimi-K3 on 8x MI355X: 2026-09-04 planning snapshot

Planning documents and the gate/bundle scripts used for the 2026-09-04 campaign,
preserved here because `plans/` is gitignored and the working machine was retired.

- `k3-beat-vllm-0.28-v3.md`: the plan, with the live execution log (§7.0) and the
  TTFT lever research summary.
- `decode-gap-plan-20260904.md`: decode attribution vs vLLM 0.28 and ranked levers.
- `scaling-audit-20260904.md`: which promoted changes generalize beyond the C1
  8192→1024 cell (long context, batching, throughput) and the gates required.
- `seq-parallel-seams-feasibility-20260904.md`: the sequence-parallel seams lever
  (emit prototype on branch `codex/seq-parallel-seams`, runtime arms pending).
- `scripts/`: bundle build (`showdown_bundle.sh`), served showdown launcher
  (`run_showdown.sh`, needs `sg docker` when the login lacks the docker group),
  stack gates, the regstate exactness probe, and the C1 publication generator.
  Paths under `/tmp` and the worktree path are machine-specific.

Served state at the snapshot (`perf-data/kimi-k3-plowrt-mi355x-baseline.md`):
Plow 1113 ms TTFT / 25.25 ms TPOT / 38.0 tok/s vs vLLM 566 / 20.88 / 46.7, with a
further −22.9 ms TTFT engine-gated after publication. Remaining gap and the order
of attack are in the three plan documents.

## Reproducing the exact compilation

Everything the served packet depends on is either in this tree or pinned by the
`nix develop` shell (ROCm 7.14.0, cargo, Lean 4.15.0); the toolchain label is
`rocm-7.14.0-nix`. There are no hand-set flags: every promoted route is a
default, so a flag-free emit at this commit reproduces the packet.

1. Checkpoint: `scripts/kimi_k3_prep.py` builds the symlink farm (HF Kimi-K3
   shards plus derived sidecars) that `--rt-checkpoint` points at; the HF
   `config.json` sha256 starts `9710e121`, `tokenizer.json` `9ca6299a`.
2. Emit (from `scripts/showdown_bundle.sh`):
   `K3_FULL=1 PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix PLOW_VERIFY_BIN=<lean-plow>/.lake/build/bin/plow_verify
   plowc --hf-dir <checkpoint> --emit devblob --arch gfx950 --gpu mi350 --num-gpus 8
   --parallel tp --max-ctx 16384 --n-cu 256 --out <bundle>/assets`.
   `build.json` must show `lean.verified=true`, `lean.oracle=true`, and
   `tuning.tile_measured == tuning.tile_lookups` (7650/7650 here). The emit log
   prints `want gfx950-<label>` if the TuneDB route records are stale for the
   build; `tuning/amd/gfx950/mi350x/*.jsonl` carry the records and their label.
3. Objects: `cmake -S runtime -B <bundle>/cmake -DPLOW_GFX950_HSACO=ON
   -DPLOW_HSACO_ARCH=gfx950 -DPLOW_HSACO_CONFIG=<bundle>/assets/plow_config.h
   -DPLOW_HSACO_DIR=<bundle>/hsaco -DPLOW_HSACO_DECODE_INVENTORY_PRUNE=ON
   -DPLOW_HSACO_DECODE_MLA_SEGMENTS=ON -DPLOW_HSACO_MOE_DECODE_GROUPED=ON
   -DPLOW_HSACO_KDA_KEY_FACTOR=OFF` then `cmake --build <bundle>/cmake --target
   gfx950_hsaco`. The packet's `plow_config.h` selects every paired object
   (regstate carry, f32-mix AttnRes, grouped decode, seams arm when enabled);
   `plowrt` refuses an object whose stamped pairing hash disagrees with the packet.
4. Runtime: `cargo build --release -p plowrt --features hsa` (without `hsa` the
   binary has no AMD backend and `bench` refuses to run).
5. Identity to compare against: `perf-data/kimi-k3-plowrt-mi355x-c1.json`
   (`artifact_identity`: source head, packet and manifest sha256, pairing hash,
   plowrt sha256, object count, tuned tiles). Re-emits at the same commit were
   byte-identical during the campaign (e.g. packet sha `204d94bb…` twice).
   Object ELFs are disassembly-identical across rebuilds; raw bytes differ only
   by embedded build paths.
6. Served numbers: `scripts/run_showdown.sh <bundle> <run-id>` (pinned vLLM 0.28
   image digest inside; needs the docker group, `sg docker` if the login lacks it).
