# Kimi-K3 MI325X Stage 5: real-weight block gate

Date: 2026-08-10. Hardware: 8 leased MI325X GPUs (gfx942, 304 CUs each).
Toolchain: flake-pinned ROCm 7.14.0, HIP 7.14.60850, clang 23. Every emit,
build, and device run was entered through `nix develop`; timings held `gpulease`.

## Artifact and correctness

The target is Kimi-K3 TP8 with native MXFP4 expert weights and FP8 KV. The full
checkpoint is `/home/lava/models/k3_farm`; the production gfx942 objects are in
`/home/lava/models/k3_mi325x/hsaco`.

Stage-5 real-weight substep oracles passed for the dense KDA first layer, latent
MoE KDA layer, and MLA+latent-MoE layer. The production grouped-A4W4 interpreter
fixture also passes its FP64 bridge and DOWN oracles.

The TP equivalence gate emitted the first one and two layers independently at
TP1 and TP8, then compared the full 163,840-element BF16 logit vectors after
prefill and three decode steps. The two-layer depth is mandatory because it is
the first routed/shared-expert layer.

| depth | point | minimum cosine | argmax |
|---:|---|---:|---|
| 1 | prefill + 3 decode steps | 0.99999005 | equal at every point |
| 2 | prefill + 3 decode steps | 0.99998174 | equal at every point |

Both exceed the 0.9999 gate. Every TP8 rank also emitted identical token ids.

## Five-layer latency sweep

`scripts/k3_block_sweep.sh` emits the smallest span containing both mixer types:
five layers = four KDA and one MLA, with four routed-MoE layers. It binds real
weights and runs 64 decode steps at TP8.

| context | latency | throughput |
|---:|---:|---:|
| 8,000 | 4.504 ms/token | 222.0 token/s |
| 16,000 | 4.499 ms/token | 222.3 token/s |
| 32,000 | 4.503 ms/token | 222.1 token/s |

The sweep is context-flat. It is a ranking instrument, not a 93-layer latency
estimate: embedding and the 163,840-vocabulary head occur once in both the
truncated and full model.

## Integration fixes found by the gate

- `K3_NLAYERS` had been retired, so the old script silently emitted all 93
  layers. The script now uses `PLOW_K3_LAYERS=5` and checks the emit census.
- The gfx942 inventory lacked `interp_prefill_fp8kv_k3{,_gq}.elf`, required by
  decode-only K3 truncations. `scripts/build_gfx942.sh` now builds that row.
- Both Stage-5 scripts now refuse non-Nix ROCm, target gfx942/MI325X/304 CUs,
  bind the real checkpoint, and use `gpulease` instead of `sg`/private locks.
- TP comparison no longer needs an unpinned NumPy environment; its BF16 vector
  comparison uses the Python standard library inside the flake shell.

## Gate decision

The numerical and isolated-latency portions of Stage 5 pass. The remaining
anchor is a same-box reference-framework block/decode floor; no MI325X Kimi-K3
vLLM/SGLang environment is installed yet. Stage 6 full-model serving may proceed
for Plow profiling, but no "beats vLLM" claim is valid until the Stage-7
same-session comparator exists.
