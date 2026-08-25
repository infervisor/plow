# Handoff: increase occupancy, fuse more ops, win prefill vs vLLM — Gemma-4-12B/RTX 5090

2026-08-25. Written because the sandbox's GPU access was lost mid-campaign (see "Blocker" below)
and the pod needs a restart to recover it — **everything in `/workspace` outside this git repo
(model weights, venvs, built cubins, the live vLLM server) will be wiped by that restart.**
This file plus the two other perf-data files it references are the durable record. Read this
file top to bottom before doing anything else; it tells you exactly what to re-run and why.

## Where things stand — the live comparison

`perf-data/gemma4-12b-sandbox-5090-2026-08-25.md` has the full writeup. Headline, live/same-box,
vLLM 0.27.0 (patched — see below) vs plow, `google/gemma-4-12B-it`, RTX 5090:

- **Decode**: plow loses by 33-37% at every concurrency 1-16, even after a validated
  `GV_MM_MAX=16` win (+15-29% decode throughput, register-neutral).
- **Prefill**: plow loses by 28-39% at 2k/8k/16k context, concurrency 1.

Three research passes (findings also in that file, §-cross-referenced from
`/root/.claude/plans/cuddly-sleeping-kazoo.md` — read that plan file too, it has the full
evidence chain with file:line citations for every claim below) established *why*, dissecting the
actual kernel bodies rather than emit-time config:

1. **Persistent-kernel dispatch overhead is not the cause.** Measured on this exact cell: ~3.7%
   ceiling on wall-clock even if dispatch were free. The gap is in the compute kernel bodies.
2. **Decode GEMV runs at ~21-22% of peak HBM bandwidth**, capped at occupancy 1 block/SM by
   **register pressure across the whole monolithic megakernel** (`REG:255`, the worst-case union
   over every inlined opcode — GEMV, flash-attention, GEMM, MoE, norm all share one `ptxas`
   allocation in `runtime/nvidia/interp_sm120.cu`'s one `__global__` function). Not fixable by
   tuning the GEMV body alone.
3. **Prefill uses the correct tensor-core ISA already** (`mma.sync.m16n8k16` — the right
   primitive for consumer Blackwell/sm_120a, confirmed against the repo's own
   `runtime/nvidia/gemma_sm120.cu:1-10`). Its occupancy-1 ceiling is the **same register-pressure
   mechanism** as decode's, not the flash-attention shared-memory arena (an ~85 KiB figure
   documented elsewhere in this repo turned out to be simply wrong — corrected value 81,664 B —
   and even a 0-byte arena wouldn't raise occupancy past 1).
4. An unshipped, already-measured tensor-core batched-GEMV experiment exists
   (`runtime/tests/e4_tc_fp8_decode_sm120.cu`, `perf-data/rtx19-e4-tc-fp8-decode.md`) — 1.05x/
   1.38x/3.67x vs the shipped kernel at B=1/8/32 — never integrated (needs a global f32
   split-K partials buffer, foreign to the single-launch persistent design).

## What's already done (safe, committed or ready to commit)

1. **A real usability/correctness bug found and fixed**: `plowc --hf-dir --arch sm_120a` used to
   silently compile a prefill asset that fails at *serve* time (a different binary) with a
   message that doesn't name the missing flag (`PLOW_UNISEG=1` is required for the sm_120
   interpreter to accept any prefill object at all). **Fixed in `crates/plowc/src/main.rs`
   (`fn main`)** — defaults `PLOW_UNISEG=1` whenever `--arch` starts with `sm_120` and the env
   var isn't already set by the caller. Verified: `cargo check --release -p plowc` compiles
   clean. **This diff is sitting uncommitted in the working tree — commit it, don't lose it.**
2. **A real vLLM 0.27.0 bug found, root-caused, and fixed** (needed to get *any* live comparison
   at all): every full-attention layer's `q_norm`/`k_norm` was allocated at the wrong head_dim
   (256 instead of 512) — a version-skew bug between vLLM's `gemma4.py` and the paired
   `transformers` 5.15.1 release's new heterogeneous-config schema. Fixed via an in-process
   monkeypatch, no edits to installed packages: `perf-data/tools/vllm_gemma4_launch.py` (copied
   into this repo from the working sandbox so it isn't lost — full detail in
   `gemma4-12b-sandbox-5090-2026-08-25.md` §3). Worth reporting upstream to vLLM/transformers.
3. **`GV_MM_MAX=16`** (decode GEMV weight-tile-residency win) — validated, register-neutral,
   +15-29% decode throughput across c1-c16. Not yet the shipped default; a `-D` flag on
   `scripts/build_sm120_cubin.sh` (`PLOW_EXTRA_DEFINES="-DGV_MM_MAX=16"`), not a source change.
4. **In progress, blocked by the restart (see below)**: `PLOW_NV_FORCE_MINBLK=2` stacked on
   `GV_MM_MAX=16` for the decode object — the single highest-priority *untried* lever per the
   plan (occupancy 1→2 blocks/SM, trading register spill for more concurrent warps; two prior
   campaigns found GEMV-family kernels occupancy-*positive* at 2 blocks/SM on other GPUs/models,
   but nobody has ever run the *full* decode object this way, end-to-end, on this exact
   model/GPU). Was mid-benchmark when GPU access was lost — see exact resume steps below.

## Blocker that triggered this handoff

`cudaGetDeviceCount()` returns `count=0, err=100` (`CUDA_ERROR_NO_DEVICE`) — confirmed at the
lowest level (a trivial `cudaGetDeviceCount` C program, not just `nvidia-smi`/plowrt). Root
cause: this is a RunPod pod (`RUNPOD_GPU_NAME=NVIDIA+GeForce+RTX+5090` in env); the container's
cgroup v2 device filter (eBPF-based, invisible to `ls`/file-mode bits) is denying `open()` on
`/dev/nvidia3`/`/dev/nvidiactl` with `EPERM`, even though the driver still shows the GPU
registered and not excluded (`/proc/driver/nvidia/gpus/0000:01:00.0/information`). The container
has no `CAP_SYS_ADMIN`/`CAP_BPF` to fix this from inside. **This needs a full pod stop→start
(not a soft/process restart) to force GPU reattachment.** Not an nvcc/CUDA-toolkit bug — nvcc
itself compiled cubins successfully minutes before this was discovered.

## Exact steps to resume after the restart

Assume a **fresh container**: no `/nix`, no model weights, no venvs, no built cubins. The git
repo (`/workspace/plow`, this branch) is the only thing that survives.

### 0. Re-clone / pull this branch, re-bootstrap the toolchain

```bash
git checkout shaswot/prefill   # or wherever this landed — check git log for the handoff commit
```

Nix bring-up (see `gemma4-12b-sandbox-5090-2026-08-25.md` §0 for the exact gotchas — `mount` is
unavailable in this sandbox class, `/nix` must go on `/` not a bind-mounted bigger disk; nixbld
group needs a non-root member; `pkill -f` self-matches, use `pkill -x`):

```bash
sh <(curl -L https://nixos.org/nix/install) --no-daemon --yes   # after: groupadd/useradd nixbld dance, see §0
mkdir -p ~/.config/nix && echo -e "experimental-features = nix-command flakes\nsandbox = false" > ~/.config/nix/nix.conf
for f in /root/.nix-profile/bin/*; do ln -sf "$f" /usr/local/bin/; done
```

Build host binaries (avoids the ROCm pull `nix develop` would force — see §0 for why):
```bash
export NIX_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
cd /workspace/plow
nix build .#plowc .#plowrt --no-write-lock-file -L --out-link /workspace/plow-work/result-host
nix build .#plow-interp-sm120a --no-write-lock-file -L --out-link /workspace/plow-work/result-cubin
mkdir -p /workspace/plow-work/bin /workspace/plow-work/cubin
cp result-host/bin/plowc result-host-1/bin/plowrt /workspace/plow-work/bin/ && chmod +w /workspace/plow-work/bin/*
cp result-cubin/cubin/*.cubin /workspace/plow-work/cubin/ && chmod +w /workspace/plow-work/cubin/*
rm -f /workspace/plow/result-host /workspace/plow/result-host-1 /workspace/plow/result-cubin  # don't leave these in the repo dir
```
The `PLOW_UNISEG` default fix (item 1 above) means `plowc --hf-dir --arch sm_120a` no longer
needs `PLOW_UNISEG=1` passed explicitly — but pass it anyway until you've confirmed the rebuilt
`plowc` picked up the committed change.

### 1. Re-fetch weights and re-quantize fp8 (the 23.9 GiB checkpoint is NOT in git, can't be)

```bash
huggingface-cli login --token "$HF_TOKEN"   # ask the user for this again — it was pasted in a prior session, don't hardcode it anywhere
hf download google/gemma-4-12B-it --local-dir /workspace/models/gemma-4-12B-it
```
fp8 quantize (38s on a decent CPU last time, uses `perf-data/tools/quantize_fp8.py` against any
available torch — the vLLM venv's torch worked fine, no need for the `nix develop .#quantize` shell):
```bash
python3 -m venv /workspace/venvs/vllm && source /workspace/venvs/vllm/bin/activate
pip install vllm==0.27.0   # also gives us the vllm bench serve client + a torch for quantizing
python3 perf-data/tools/quantize_fp8.py /workspace/models/gemma-4-12B-it /workspace/models/gemma-4-12B-it-fp8
```
Rebuild the merged bf16+fp8 checkpoint dir (`--fp8-dir` is AMD-only, doesn't work on this CUDA
path — see sandbox report §0):
```bash
mkdir -p /workspace/models/gemma-4-12B-it-merged
for f in model.safetensors:../gemma-4-12B-it/model.safetensors \
         model_fp8.safetensors:../gemma-4-12B-it-fp8/model.safetensors \
         generation_config.json config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
  dst="${f%%:*}"; src="${f#*:}"
  [ "$dst" = "$src" ] && src="/workspace/models/gemma-4-12B-it/$dst"
  ln -sf "$src" "/workspace/models/gemma-4-12B-it-merged/$dst" 2>/dev/null || \
    ln -sf "/workspace/models/gemma-4-12B-it/$dst" "/workspace/models/gemma-4-12B-it-merged/$dst"
done
# (the loop above is a compressed reconstruction — simplest is just 7 explicit `ln -sf` lines,
#  see gemma4-12b-sandbox-5090-2026-08-25.md §0 if this doesn't paste cleanly)
```

### 2. Rebuild the GV_MM_MAX=16 + FORCE_MINBLK=2 decode cubin — this is where we got cut off

```bash
export PLOW_ROOT=/workspace/plow
PLOW_EXTRA_DEFINES="-DGV_MM_MAX=16 -DPLOW_NV_FORCE_MINBLK=2" \
  scripts/build_sm120_cubin.sh /workspace/plow-work/cubin-mm16-occ2/interp_sm120.cubin
cuobjdump -res-usage /workspace/plow-work/cubin-mm16-occ2/interp_sm120.cubin | head -6
# expect: REG:128 STACK:1440ish (up from 1024 baseline — real spill, confirmed via
# `cuobjdump -sass | grep -c "STL\|LDL"`: 439 vs 66 at GV_MM_MAX=16/occ-1. This is the
# thing that needs measuring — spill traffic could eat the occupancy gain, that's the
# whole point of this experiment.)
```
occ-2 needs the packet re-sliced to `n_cu = 2 * sm_count = 340` (170 SMs × 2) — the interpreter
refuses to load a grid/n_cu mismatch (`"interpreter grid 340 (2/SM × 170 SMs) != packet n_cu 170
— recompile the packet with n_cu=340"`):
```bash
PLOW_UNISEG=1 PLOW_DEV_SAMPLE=1 PLOW_MULTISTEP=8 /workspace/plow-work/bin/plowc \
  --hf-dir /workspace/models/gemma-4-12B-it --arch sm_120a --gpu rtx5090 \
  --max-ctx 2048 --n-cu 340 --batch 1 --seq 128 --phase both \
  --emit-decode-batch-ladder 1,4,8,16 --fp8 --weight-dtype fp8 \
  --out /workspace/plow-work/assets/gemma4-12b-decode-occ2
```
Serve (correctness-gate with the two prompts below BEFORE trusting any speed number — this repo
has a documented recurring "fluent but wrong" numerics-defect class):
```bash
/workspace/plow-work/bin/plowrt serve --assets /workspace/plow-work/assets/gemma4-12b-decode-occ2 \
  --port 8080 \
  --nv-cubin /workspace/plow-work/cubin-mm16-occ2/interp_sm120.cubin \
  --nv-cubin-pf /workspace/plow-work/cubin-mm16-occ2/interp_sm120_pf.cubin \
  --nv-cubin-sample /workspace/plow-work/cubin-mm16-occ2/sample_sm120.cubin \
  --rt-checkpoint /workspace/models/gemma-4-12B-it-merged &
```
Correctness gate (both must match: "Paris"; a coherent bicycle-balance explanation):
```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"gemma-4-12b-it","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":10,"temperature":0}'
```
Then the actual measurement — same protocol as every number in `gemma4-12b-sandbox-5090-2026-08-25.md`
§2/§5 (`vllm bench serve --backend openai-chat --random-input-len 128 --random-output-len 1024
--ignore-eos`, concurrency 1/4/8/16, `--num-prompts` = 4× concurrency), compare against that
file's `GV_MM_MAX=16`-only numbers (c1=38.41, c4=136.51, c8=250.50, c16=489.84 tok/s) to see what
occupancy alone bought.

### 3. Get vLLM's live baseline back for comparison

```bash
python3 perf-data/tools/vllm_gemma4_launch.py serve /workspace/models/gemma-4-12B-it \
  --served-model-name gemma-4-12b-it --host 127.0.0.1 --port 8081 \
  --max-model-len 2048 --gpu-memory-utilization 0.93 --no-enable-prefix-caching
```
(`0.93` gave a 17,100-token KV pool last time — enough for ~14-15 of the 1,152-token decode
requests concurrently, slightly short of a clean c16; `0.95` OOM'd during CUDA-graph capture,
don't go there again without backing off `--kv-cache-memory` explicitly first.)

## What's left — in priority order (full detail + evidence citations in the plan file)

Read `/root/.claude/plans/cuddly-sleeping-kazoo.md` in full for the complete evidence chain —
this is a compressed pointer list, not a replacement for it.

1. **Finish the `FORCE_MINBLK=2` measurement** (interrupted by the restart — resume steps above).
2. Check whether `GV_MM_MAX=16` and `FORCE_MINBLK=2` compose without worse spill than either alone.
3. Re-sweep `GV_UNROLL`/`PLOW_NS_ABS` alongside `GV_MM_MAX` (reconcile the B=8 discrepancy with
   `perf-data/px15-tunedb-sm120.md` noted in the sandbox report §2).
4. FASTPF on/off A/B and `PLOW_NV_FA_FP8PV` on prefill (not attempted yet, now directly
   comparable to a live vLLM baseline).
5. Populate `tunedb` properly (`tunedb-decode ingest`) instead of hand A/Bs.
6. **The big lever — tensor-core batched fp8 GEMV integration** (Phase 2 of the plan). Real
   kernel-numerics work: new w8a8 precision path, split-K global-partials-buffer + finalize pass,
   shape/batch-gated dispatch (the experimental kernel *regresses* at B≤4 on wide-N shapes — do
   not ship a blanket replacement). **Needs explicit user sign-off before merging** — this repo
   has a documented near-identical prior bug in the adjacent code (unwritten batch rows). Gate
   against exact-token-match + GSM8K on the *integrated* serving path, not just the isolated
   kernel's synthetic gate.
7. **"Fuse more ops"** (the user's own framing, 2026-08-25): no specific fusion opportunity is
   evidenced yet beyond what's already fused (`GemvQkv`, `GemvGlu`, `NormResidualNorm` are already
   shipped fusions per `crates/packet/src/dev.rs`'s opcode list). The plan's dissection didn't
   surface an un-fused pair of ops on the *prefill* hot path specifically — if pursuing this
   angle, start by getting the `ncu` profiling read (item 8 below) to see where prefill wall time
   actually concentrates before guessing at a fusion target; don't fuse speculatively.
8. **`ncu` profiling smoke test** — untested whether this sandbox's `ncu` actually works (every
   prior sandbox in this repo's history hit `ERR_NVGPUCTRPERM`; unknown for this one). A 30-second
   check that gates whether the prefill GEMM `cp.async`/`LDGSTS` hypothesis
   (`perf-data/px9-gemm-body.md`, "elimination argument, not a counter reading") can be confirmed
   before any prefill-GEMM code change is attempted.
9. Prefill occ-2 GEMM "Stage-3" re-slicing: **deprioritized** — this repo's own cheap check on the
   exact object (`perf-data/px3-bn64-occ2.md`) already found it not worth the emitter investment.
   Re-run that cheap check on this box's actual shapes before reconsidering, don't build the
   emitter work on the object merely existing.

## Correctness discipline (non-negotiable, applies to every step above)

- Every timed number: `grep -aq libcuda.so.1` build-integrity check on the `plowrt` binary,
  no sibling GPU process, greedy "Paris" + coherent-paragraph gate before trusting any number.
- Every numerics-touching change: exact greedy-token match on a fixed prompt set + GSM8K accuracy
  against the actual serving path, not a synthetic isolated-kernel gate. This repo has 14+
  documented instances of kernel changes that were "fluent but wrong" rather than crashing.
- Record results in a new dated `perf-data/*.md` file per this repo's convention — state negative
  results, don't delete them. Update this handoff file's checklist as items land.
