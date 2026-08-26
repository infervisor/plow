# Prefill-vs-vLLM iteration — Gemma-4-12B / RTX 5090 (sm_120a), same sandbox as the 08-25 handoff

Follow-on to `perf-data/gemma4-12b-sandbox-5090-2026-08-25.md` and
`perf-data/prefill-occupancy-handoff-2026-08-25.md`, executing that handoff's Phase 0-2 plan
(`/root/.claude/plans/zazzy-skipping-hellman.md`). **Net result: no win. Every lever tried either
didn't apply to the object actually being served, or measured negative. The 28-39% prefill gap
vs vLLM stands, unchanged from the prior report.**

## 0. Environment note: this was NOT a fresh container

Contrary to the prior handoff's assumption, this pod's `/workspace` (models, venvs, built
cubins/binaries) survived a restart intact. Two things did NOT survive and had to be redone:

- **`/nix`'s store contents changed** (different glibc derivation hash) — the prebuilt
  `plowc`/`plowrt` binaries in `/workspace/plow-work/bin` were dynamically linked against a
  glibc path that no longer existed in the store, so they failed to exec at all
  (`No such file or directory`, actually a missing-interpreter error, not a missing-file error).
  Rebuilt via `nix build .#plowc .#plowrt` (~5 min). `/nix` itself had landed on
  `/workspace/nix` this boot rather than `/` — nix refuses a symlinked store root, so it was
  physically moved (`mv /workspace/nix /nix`) before building; `/` had the 40 GB headroom this
  needed.
- **plow and vLLM cannot share the card.** Both want ~23-24 GiB resident (weights + KV pool) on
  a 32 GB RTX 5090; the prior report's "live, same-box" comparison ran them **sequentially**, not
  concurrently, and this session had to rediscover that after an initial concurrent-launch
  attempt failed the plow-side VRAM planner (`no model fits the VRAM budget at startup`).

Baseline reproduction, once both were up (bf16, `--random-output-len 8`, concurrency 1,
`vllm bench serve --backend openai-chat`):

| input | plow TTFT (ms) | vLLM TTFT (ms) | plow/vLLM |
|---|---|---|---|
| 2,048 | 465.0 | 290.3 | 0.62x |
| 8,192 | 1744.3 | 1204.0 | 0.69x |
| 16,000 | 3517.7 | 2533.0 | 0.72x |

Matches the prior report's numbers closely (464.0/1729.4/3494.3 ms plow; 7,027/6,819/6,376 tok/s
vLLM — this table's vLLM TTFT converts to 7056/6803/6317 tok/s). Environment confirmed stable
before spending any time on knobs.

## 1. Handoff item 4 (`FASTPF`/`PLOW_NV_FA_FP8PV` A/B) — does not apply to this baseline, skipped

`runtime/CMakeLists.txt:35,241`: both `PLOW_FP8_KV_FASTPF` and `PLOW_NV_FA_FP8PV` gate the
**fp8-KV prefill** arms specifically. The baseline asset (`gemma4-12b-prefill`, matching §4 of the
prior report) has bf16 KV (`build.json`: `"kv_dtype":{"hd256":"bf16","hd512":"bf16"}`), same as
vLLM's own comparison arm (no quantization flags). Neither knob has anything to flip on a
bf16-KV object — testing them would require emitting a whole separate fp8-KV asset, a bigger and
separately-risky change, not the cheap rebuild-only A/B the handoff described it as. **Not
attempted**; flagged for a future campaign that's willing to compare plow-fp8-KV against a
vLLM-fp8 baseline (apples-to-apples), not mixed into this bf16-vs-bf16 comparison.

## 2. Handoff item 6 / PX-9's ranked lever #2 (per-op `BN` selector) — implemented, oracle-clean, **regresses e2e**

**Scoping correction to the handoff**: `perf-data/px9-gemm-body.md` and `perf-data/px3-bn64-occ2.md`
measured their BN=64-for-GLU findings on the **w8a8** GEMM body (`d_gemm_w8a8`/
`d_gemm_glu_w8a8`, fp8 weights AND fp8 activations, native `mma.m16n8k32.e4m3`). The baseline
actually being served here — and being compared against vLLM's bf16 baseline — runs the **plain
bf16** body (`d_gemm`/`d_gemm_glu`, `mma.sync.m16n8k16.bf16`). These are different functions in
`runtime/nvidia/op_gemm.cuh`; PX-9's specific numbers (register-neutral, +6.4% weighted) do not
transfer by citation — they needed re-measuring on the object actually in play. Also: the
"already shipped, default ON" PX-9 body fixes (`PGM_W8A8_LDS64`, `PGM_SW8_V2`) are w8a8-only by
name and don't touch the bf16 body either — they were never in this baseline's critical path to
begin with, contrary to what the handoff implied.

**Change** (`runtime/nvidia/op_gemm.cuh`): split `PGM_BN` into a GLU-specific `PGM_BN_GLU`
(defaults to `PGM_BN`, i.e., no-op unless overridden) with its own derived `PGM_WN_GLU` /
`PGM_NFRAG_GLU` / `PGM_BBUF_GLU`. Templated `pgm_stage_b`/`pgm_load_bfrags` on `(BN)` /
`(WN, NFRAG)` so `d_gemm_glu` can use the GLU-specific tile width through the same body while
`d_gemm` (plain GEMM/GEMM_MED/GEMM_SMALL) stays on the untouched default path. `PGM_ARENA_GLU`
recomputed from `PGM_BBUF_GLU`.

**Gates — all pass:**
- Numeric oracle (bit-exact CPU-f64-reference relL2, extracted from `runtime/tests/
  sm120_interp_op_test.cu`'s `test_gemm`/`test_gemm_glu` into a standalone harness since that
  file has pre-existing, unrelated MoE/quant_fp8 signature-drift build breaks on this branch,
  confirmed present before this change too via `git stash`): BN_GLU=64 vs BN_GLU=128 (control)
  give **identical relL2 to displayed precision** on q_proj, down_proj, and GLU at M=1024/256/37
  (non-tile-multiple), gelu and silu — control matches the file's own baseline numbers exactly.
- Register/spill budget on the **actual production prefill object**
  (`interp_sm120_pf11PlowProgram`, built via `scripts/build_sm120_cubin.sh` — the real serving
  path, not the isolated PX-9/PX-3 lean-segment object): `REG:240 STACK:1024 SHARED:3344
  LOCAL:0` identical between baseline and BN_GLU=64; spill instruction count (`grep -c
  "STL\|LDL"` over `cuobjdump -sass`) identical at 66/66.
- Serving correctness: `grep -aq libcuda.so.1` on `plowrt` — pass. Greedy "Paris" — pass. The
  bicycle-balance paragraph — **exact text match**, word-for-word identical to the baseline's
  output, at `max_tokens=80, temperature=0`.

**Measured (same protocol as §0, same asset, only the cubin swapped):**

| input | baseline TTFT (ms) | BN_GLU=64 TTFT (ms) | delta |
|---|---|---|---|
| 2,048 | 465.0 | 515.2 | **+10.8% (worse)** |
| 8,192 | 1744.3 | 1933.0 | **+10.8% (worse)** |
| 16,000 | 3517.7 | 3885.6 | **+10.5% (worse)** |

Reproduced at 2,048 a second time (514.1 ms) — consistent, not noise.

**Reading**: T2's own comment in `op_gemm.cuh` (`BN 64->128 halves the activation re-read
traffic — A is streamed once per N-tile, so a wider N-tile is a direct global-bandwidth win on
the memory-bound prefill GEMM`) applies twice as hard to bf16 (2 B/element) as to w8a8's fp8
operands (1 B/element) — this bf16 GLU body is more bandwidth-bound than register-bound, the
opposite regime from what made BN=64 a register-pressure win on the w8a8 arm. **The lever is real
but its sign flips with the weight/activation precision it's applied to.**

**Verdict: negative, reverted.** The code (default-preserving, oracle-validated, register-neutral
at default) is left in the tree as an overridable knob — consistent with this repo's existing
convention for `PGM_BN`/`PGM_STAGES` themselves — but `PGM_BN_GLU` is NOT overridden anywhere,
so production behavior is unchanged. Do not set `-DPGM_BN_GLU=64` on a bf16 object; it may still
be worth trying on a genuine w8a8/fp8-weight prefill object if one is ever built and compared
against a like-for-like (fp8) vLLM baseline.

## 3. Net result

Phases 0-2 of the handoff's plan are exhausted for the bf16-vs-bf16 comparison actually in play:
- Phase 1's cheap knobs don't apply to this object (§1).
- Phase 2's best-evidenced lever, once correctly re-scoped and measured on the real object, is a
  ~10.5-10.8% regression, not a win (§2).

**The 28-39% prefill gap vs vLLM is unchanged.** Per the approved plan, this is the honest
stopping point for the cheap/medium-effort lever set — the plan's own next lever (a TMA-based
GEMM mainloop port for sm_120a, PX-9 §Result 7 item 1) is new kernel-body work with no existing
sm_120a precedent in this tree and needs the user's explicit go-ahead before starting, which was
deliberately not given for this session (see the plan file's Phase 4 gate).

## Correctness discipline applied throughout

`libcuda.so.1` link check, no sibling GPU process, greedy "Paris" + exact-paragraph-match gate
before every timed number, register/spill diff on the real production object before trusting any
"register-neutral" claim, negative results recorded rather than discarded.
