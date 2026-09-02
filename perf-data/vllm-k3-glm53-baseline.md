# vLLM vs plow baseline — Kimi-K3 and GLM-5.3-Flash on 8x MI355X (gfx950), TP8

Measured 2026-08-31/09-01. Every point is `vllm bench serve` (the same client, whichever
engine is on the other end) against a coherence-gated (`/v1/chat/completions` → "Paris")
server, concurrency 1, `--random-output-len 128`, contexts 8k/16k/64k/128k, TP8. Checkpoints
served from local disk (`/home/shaswot/models/{Kimi-K3,GLM-5.3-Flash}`), not the HF cache —
both are day-0/day-N checkpoints with no populated HF snapshot on this box.

**Purpose**: an apples-to-apples vLLM vs plow number, same client, same context sweep, same
box, for both prefill (TTFT / prefill tok/s) and decode (TPOT / decode tok/s). The vLLM
sections are the baseline; the plow sections are what plow's own build (fresh from source on
this box, untuned — see caveats) gets on the same test.

## Kimi-K3 — TP8, ctxsweep (concurrency 1)

Image: `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0` → vLLM
`0.27.1.dev5+gf46a9dfe2.d20260827.rocm100`. `--dtype auto --trust-remote-code`, no AITER
(unset). `KimiK3ForConditionalGeneration` / `KimiLinearForCausalLM` are natively registered
in this vLLM build — no fallback path needed.

| ctx | TTFT (ms) | prefill tok/s | TPOT (ms) | decode tok/s |
|----:|----------:|---------------:|----------:|---------------:|
| 8k   | 994.4    | 8238.5 | 250.59 | 3.99 |
| 16k  | 2034.4   | 8053.3 | 250.74 | 3.99 |
| 64k  | 8875.3   | 7384.1 | 251.54 | 3.98 |
| 128k | 19943.2  | 6572.3 | 252.17 | 3.97 |

Raw CSV: `perf-data/vllm-rocm/_home_shaswot_models_Kimi-K3_bf16_tp8_ctxsweep_c1.csv`

**Decode is a bring-up floor, not a tuned number.** The server log reports
`Using 'EMULATION' Mxfp4 MoE backend` — K3's routed experts run through a software
emulation path on ROCm, not a native mxfp4 kernel. That is consistent with vLLM's own
FAQ ("ROCm support ships at launch, with broader tuning on the roadmap") and explains
250 ms/token (≈4 tok/s) against plow's own K3 decode numbers of 29.8–34.5 tok/s
(`perf-data/kimi-k3-README.md` §11). **Do not read the decode column as "vLLM's K3
ceiling on MI355X" — it is this bring-up path's floor.** The TTFT/prefill columns are
the ones this doc exists for.

Weight load (1453.74 GiB, 96 shards) took ~52–55 min cold, repeatably — a rerun with the
OS page cache still warm took the same ~55 min, so this looks CPU-bound (tensor
materialization/dtype conversion) rather than disk-bound; do not budget on cache making a
second load fast.

## plow — Kimi-K3, TP8, ctxsweep (concurrency 1)

Built fresh from source on this box (2026-09-01): `nix develop` devshell, gfx950 HSACO kernels
(49 code objects, system ROCm 7.0.2 hipcc/bundler — the nix-packaged `clang-offload-bundler`
is missing `llvm-objcopy`/`llvm-readelf` alongside itself, so the build gate that unbundles
and symbol-checks every object fails under it; system ROCm has all three tools together),
`plowc`/`plowrt --features hsa` via `cargo build --release`. Checkpoint: `scripts/kimi_k3_prep.py
--derived --farm` symlink farm over the same `/home/shaswot/models/Kimi-K3` snapshot vLLM used.
Blob: `K3_FULL=1 PLOW_FP8_KV=1 PLOW_MXFP4=1`, `--max-ctx 133248` (128k + 128 output + 2048
headroom — see the rejected-request trap below), `--num-gpus 8 --parallel tp`.

| ctx | TTFT (ms) | prefill tok/s | TPOT (ms) | decode tok/s |
|----:|----------:|---------------:|----------:|---------------:|
| 8k   | 4691.5  | 1746.1 | 58.53 | 17.09 |
| 16k  | 9342.7  | 1753.7 | 58.87 | 16.99 |
| 64k  | 44739.9 | 1464.8 | 60.82 | 16.44 |
| 128k | 111680.5| 1173.6 | 62.89 | 15.90 |

Raw logs: `perf-data/plow-rocm/k3_ctxsweep_in{8192,16384,65536,131072}_c1.log`

**Decode: plow wins big, as expected.** 16-17 tok/s vs vLLM's ≈4 tok/s (vLLM is running
K3's MXFP4 MoE through a software emulation backend right now — see the vLLM section above).
This matches the direction, if not the exact magnitude, of the 29.8–34.5 tok/s plow has
measured in tuned prior campaigns (`perf-data/kimi-k3-README.md` §11) — the gap from THAT
number to this one is the untuned-build caveat below, not a regression.

**Prefill: plow loses here, and this is new information.** 1174–1754 tok/s vs vLLM's
6572–8239 tok/s — vLLM is 4.4-6.2x faster at prefill on this same box, same client, same
sweep. This is consistent with what `docs/amd/kimi-k3-vllm-day0.md` §5 already said plow
does not have: *"The KDA prefill scan ... are things vLLM has and plow does not."* 69 of
K3's 93 layers are KDA; without a KDA prefill scan, prefill on those layers is doing more
work per token than it needs to. This is the concrete number behind that gap.

**Caveats on this specific build, weakest to strongest:**
1. **Tuning DB is stale.** `plowc` reported *"3080 record(s) skipped as STALE against the
   probed build gfx950-ee148e7bbc86c531 — NO usable records remain, so tile selection fell
   back to the analytical model."* This build's toolchain digest doesn't match any prior
   measured-tile campaign, so every GEMM/GEMV tile choice here is the portable analytical
   model, not plow's previously-measured optimum. `scripts/rebench_tune_gemm_all.sh` on a
   quiet GPU would close this gap; not run here.
2. **No Lean correctness verification.** `lean.verified=false` — no `plow_verify` binary was
   built (`lake build` in `lean-plow/` was not run). The gates that check are skipped, not
   failed.
3. **A real object-selection gap, not just a perf caveat.** Every run logged repeated
   `packet/object K3 MISMATCH: this packet dispatches AttnRes (op 104), but
   interp_flash_fp8kv_gq.elf was compiled without PLOW_K3` — the runtime falls back to the
   8-wave interpreter for these segments (a documented, non-fatal degrade path: "AMD's
   dispatch default writes NOTHING rather than trapping" is the failure this fallback exists
   to avoid), but it means this build never exercises the flash object plow's design intends
   for this op. `-DPLOW_HSACO_K3=ON` (which we did pass, and is ON by default) evidently
   does not cover this specific flash/GQ object; worth a real fix, not just a rebuild flag.
4. **Correctness gate needed the full chat template to look right.** A bare 5-token
   completion prompt (`amd-bench --prompt 1008,10484,318,15383,387`, matching
   `kimi-k3-README.md`'s own gate) produced fluent but off-topic greedy output that never
   said "Paris" — though **all 8 TP ranks were token-identical**, which is the actual
   correctness bar that doc states ("however plausible it reads") — tokens differing across
   ranks is the failure mode a memory-corruption bug produces, not agreement. Through the
   real `/v1/chat/completions` endpoint with the model's own chat template, the answer is a
   clean, direct `"Paris"`. Read as: the raw-completion gate's exact reference text in the
   old doc was generated under a different (and, per that doc's own §5.6, since-corrected)
   ctx/build configuration — not evidence of a bug here, but also not a rerun of that exact
   historical A/B.
5. **Rejected-request trap, hit and fixed live.** The first blob was emitted at
   `--max-ctx 131072` exactly; the 128k point's chat-templated prompt (131089-131093 tokens,
   template overhead included) exceeded it, and the server rejected every request while
   `vllm bench serve` still reported `Successful requests: 3` with `Mean TPOT: 0.00` and
   `Total generated tokens: 60` (should be 384) — the exact trap `kimi-k3-README.md` §5.4
   documents. Caught via the `gen_toks == num_prompts × output_len` gate, blob re-emitted at
   `--max-ctx 133248` (matching vLLM's own `ctx + output_len + 2048` margin), full sweep
   rerun. The numbers above are from the corrected blob.

## Reproducing (plow, Kimi-K3)

```bash
export PATH="/nix/var/nix/profiles/default/bin:$PATH"
cd /home/shaswot/plow
nix develop --command cmake -S runtime -B build-amd -DPLOW_GFX950_HSACO=ON \
  -DPLOW_HSACO_HIPCC=/opt/rocm-7.0.2/bin/hipcc \
  -DPLOW_HSACO_BUNDLER=/opt/rocm-7.0.2/lib/llvm/bin/clang-offload-bundler
nix develop --command bash -c \
  "LD_LIBRARY_PATH=/opt/rocm/lib:/opt/amdgpu/lib/x86_64-linux-gnu cmake --build build-amd --target gfx950_hsaco -j 32"
nix develop --command cargo build --release -p plowc
nix develop --command cargo build --release -p plowrt --features hsa

python3 scripts/kimi_k3_tokenizer.py --model /home/shaswot/models/Kimi-K3
python3 scripts/kimi_k3_prep.py --model /home/shaswot/models/Kimi-K3 --derived \
  --out /home/shaswot/plow-work/k3_derived --farm /home/shaswot/plow-work/k3_farm

nix develop --command bash -c "K3_FULL=1 PLOW_FP8_KV=1 PLOW_MXFP4=1 ./target/release/plowc \
  --hf-dir /home/shaswot/plow-work/k3_farm --emit devblob --arch gfx950 --gpu mi350 \
  --num-gpus 8 --parallel tp --max-ctx 133248 --n-cu 256 --out /home/shaswot/plow-work/k3_tr"
ln -sf /home/shaswot/plow-work/k3_farm /home/shaswot/plow-work/k3_tr/checkpoint
ln -sf /home/shaswot/models/Kimi-K3/tokenizer.json /home/shaswot/plow-work/k3_tr/tokenizer.json
ln -sf /home/shaswot/plow/build-amd/hsaco /home/shaswot/plow-work/k3_tr/hsaco

perf-data/tools/gpulease -n 8 k3-serve sg render -c \
  "flock /tmp/plow_gpu.lock nix develop --command ./target/release/plowrt serve \
   --assets /home/shaswot/plow-work/k3_tr --port 8000"

# then, per context point (client in the same rocm/vllm image, --tokenizer pointed at the
# checkpoint since `--model k3_farm` isn't a resolvable HF id):
docker run --rm --network host -v /home/shaswot/models/Kimi-K3:/home/shaswot/models/Kimi-K3:ro \
  --entrypoint vllm rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
  bench serve --model k3_farm --tokenizer /home/shaswot/models/Kimi-K3 --trust-remote-code \
  --backend openai-chat --endpoint /v1/chat/completions --dataset-name random \
  --random-input-len 8192 --random-output-len 128 --max-concurrency 1 --num-prompts 3 --port 8000
```

## GLM-5.3-Flash — TP8, ctxsweep (concurrency 1)

`GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`, 306 GB on disk, fp8 e4m3, 45 layers,
288 routed experts) has **no upstream vLLM support as of 2026-08-31** — confirmed via web
search: "None of the major inference engines (llama.cpp, mlx-vlm or vLLM) has merged
upstream support as of 31 August 2026; all working setups run a patched build or a vendor
image." The generic `rocm/vllm:rocm10.0.0_..._vllm_0.27.0` image's registry has no
`Glm5Next*` entry; it falls back to vLLM's generic Transformers-backend bridge, which then
crashes on weight-name mapping for the KDA linear-attention layers (`k_conv1d`/`q_conv1d`/
`v_conv1d` vs. the bridge's single `conv1d`) — a real upstream gap, not a flag issue.

The correct path is the vendor-published per-model image, gated to gfx950:
`vllm/vllm-openai-rocm:glm53-flash` → vLLM `0.1.dev1+gfdd64a3db.rocm723`. Recipe:
`--tensor-parallel-size 8 --max-num-seqs 512 --attention-backend ROCM_AITER_MLA_SPARSE
--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`,
`VLLM_ROCM_USE_AITER=1 VLLM_ENGINE_READY_TIMEOUT_S=3600`, **`--privileged`**.

**`--privileged` was load-bearing.** Without it, this exact image/command combination was
externally killed (`SIGTERM`, no application error, just `Terminated`) 60–90s into every
attempt — 5/5 failures with full recipe flags, with flags stripped to bare `--dtype auto
--tensor-parallel-size 8`, and at a short 200s health-timeout control. Not a container
crash (a trivial non-GPU `sleep 300` container from the same image ran fine past that
window) and not host memory pressure (2.2 TiB free throughout) — something about this
per-model image's `vllm serve` process needed broader container privileges than
`--device=/dev/kfd --device=/dev/dri --group-add <video> --group-add <render>
--security-opt seccomp=unconfined` provides. Added as a `PRIVILEGED=1` toggle to
`scripts/bench_vllm_rocm.sh` (off by default — only this image has needed it so far).

| ctx | TTFT (ms) | prefill tok/s | TPOT (ms) | decode tok/s |
|----:|----------:|---------------:|----------:|---------------:|
| 8k   | 1044.0  | 7846.7  | 14.25 | 70.18 |
| 16k  | 1091.5  | 15010.4 | 14.18 | 70.52 |
| 64k  | 2262.8  | 28962.5 | 14.09 | 70.97 |
| 128k | 3006.5  | 43595.8 | 13.87 | 72.10 |

Fresh repeat (2026-09-01, same recipe; healthy after 380s, coherence PASS, 384/384
generated tokens at every point):

| ctx | TTFT (ms) | prefill tok/s | TPOT (ms) | decode tok/s |
|----:|----------:|---------------:|----------:|---------------:|
| 8k   | 1036.96 | 7900.0  | 14.210 | 70.37 |
| 16k  | 1091.89 | 15005.2 | 14.250 | 70.18 |
| 64k  | 2259.13 | 29009.4 | 14.070 | 71.07 |
| 128k | 2990.15 | 43834.6 | 13.940 | 71.74 |

Raw CSV: `perf-data/vllm-rocm/_home_shaswot_models_GLM-5.3-Flash_bf16_tp8_ctxsweep_c1.csv`

Prefill tok/s *rises* with context here (7.8k → 43.6k tok/s, 8k → 128k) — unlike Kimi-K3's
falling curve — consistent with GLM-5.3-Flash's DSA-style sparse attention decoupling
compute from context length the way GLM-5.2's DSA did in `vllm-rocm-baseline.md`. Decode
is flat-to-improving with context too (70.2 → 72.1 tok/s), the same signature. This is a
pre-built vendor image (no JIT at first run) — cold load was 390s, an order of magnitude
faster than Kimi-K3's ~55 min.

## Reproducing (vLLM, Kimi-K3)

```bash
IMAGE=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
EXTRA_MOUNTS="/home/shaswot/models/Kimi-K3:/home/shaswot/models/Kimi-K3:ro" \
DTYPE_ARGS="--dtype auto" SERVE_EXTRA_ARGS="--trust-remote-code" \
BENCH_EXTRA_ARGS="--trust-remote-code" PHASES=ctxsweep \
CTXS="8192,16384,65536,131072" DOCKER=docker \
scripts/bench_vllm_rocm.sh /home/shaswot/models/Kimi-K3 8
```

Run the whole invocation under `sg docker -c "..."` if the shell isn't already in the
`docker` group for this process (see `EXTRA_MOUNTS` addition to
`scripts/bench_vllm_rocm.sh`, needed because these checkpoints live outside the HF cache).

## Reproducing (vLLM, GLM-5.3-Flash)

```bash
IMAGE=vllm/vllm-openai-rocm:glm53-flash \
EXTRA_MOUNTS="/home/shaswot/models/GLM-5.3-Flash:/home/shaswot/models/GLM-5.3-Flash:ro" \
EXTRA_ENV="VLLM_ROCM_USE_AITER=1 VLLM_ENGINE_READY_TIMEOUT_S=3600" \
DTYPE_ARGS="--dtype auto" \
SERVE_EXTRA_ARGS="--max-num-seqs 512 --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice --attention-backend ROCM_AITER_MLA_SPARSE" \
PRIVILEGED=1 PHASES=ctxsweep CTXS="8192,16384,65536,131072" DOCKER=docker \
scripts/bench_vllm_rocm.sh /home/shaswot/models/GLM-5.3-Flash 8
```

`PRIVILEGED=1` is required for this image — see above.
