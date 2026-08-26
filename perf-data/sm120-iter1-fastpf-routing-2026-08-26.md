# Iteration 1 — attention fast-path routing: FASTPF is safe for all-layer fp8-KV, docs were stale

Iteration:      1 (`/root/.claude/plans/glimmering-soaring-stream.md`)
Commit before change: `08ed1a2` (Iteration 0 baseline)
Hypothesis:     `PLOW_FP8_KV_FASTPF=ON` traps on all-layer (non-`PLOW_FP8_KV_FULL`) fp8-KV packets,
                per `docs/flags-reference.md` and `runtime/CMakeLists.txt`'s comments — mission
                Iteration 1 asks to prove/wire this routing safely or add load-time capability
                validation if it doesn't fail cleanly.
Expected mechanism: pre-PX-23, the hd256 sliding-layer flash-prefill arm had no PIPE=1 fp8-mma
                body, so an all-layer packet (which needs fp8-KV on hd256 layers too) hit a
                device-side `default: __trap()` in the interpreter's opcode dispatch.
Expected maximum end-to-end benefit: N/A — this is a routing-safety/documentation iteration, not
                a kernel change. If FASTPF turns out unsafe, the deliverable is a load-time
                capability gate (correctness fix, exempt from the 1%-minimum-value rule).

## What was actually found (research, before any GPU time)

A fresh research pass (not reusing the docs' own claims) found `runtime/nvidia/op_attention.cuh:2443`
(the PX-23 arm) already gives **both** head dims a PIPE=1 fp8-mma flash-prefill body (hd256 px23,
hd512 px4/px8). `runtime/nvidia/interp_sm120.cu:1084-1135`'s dispatch has no branch on "mixed vs
all-layer" packet — any fp8-KV packet at hd256 routes to px23, at hd512 to px4/px8. The historical
trap this iteration was scoped to investigate **had already been fixed** in a prior session's
PX-23 work; `runtime/CMakeLists.txt:450-461`'s own comment already says this correctly. But
`docs/flags-reference.md:392,433` and `runtime/CMakeLists.txt:255-264`'s comment/`WARNING` text
still described the pre-PX-23 trap as current fact — stale documentation, not a code defect. This
is exactly the kind of thing "prove it, don't trust the flag/doc" is for.

## Live verification (not just code reading)

Rather than trust the corrected reading of the source, ran it for real:

1. Emitted an all-layer (non-`FULL`) fp8-KV packet: `plowc --hf-dir .../gemma-4-12B-it --gpu
   rtx5090 --arch sm_120a --fp8-kv --emit-max-chunk 8192 --out
   assets/gemma4-12b-prefill-fp8kv-full-mc8192` (bf16 weights/activations, fp8 KV on all 48
   layers — the exact configuration the stale docs said would trap).
2. Built `interp_sm120_pf_fp8kv.cubin` with `PLOW_BUILD_FP8KV=1 build_sm120_cubin.sh ...
   -DPLOW_FP8_KV=ON -DPLOW_FP8_KV_FASTPF=ON`. Confirmed defines actually compiled:
   `-DPLOW_FP8_KV=1 -DPLOW_NV_FA_PIPE=1` (i.e. genuinely on the PIPE=1 fast arm, not silently
   falling back to PIPE=0).
3. Served it (`plowrt serve --nv-cubin-pf .../interp_sm120_pf_fp8kv.cubin ...`), no load-time
   refusal: `smem_pf=89104` (device opt-in cap 101376).
4. **Short prompt** (25 tokens): "What is the capital of France?" → `"Paris"`. Correct, no crash.
5. **Long prompt** (2688 tokens, well past the 1024-token sliding window, forcing ring-wrap on
   every sliding layer): asked for a 5-word summary → `"Fox jumps over lazy dog."` — fluent,
   correct, 0.67s wall time, server process still alive, zero errors in the log. This is the
   configuration the docs claimed would trap; it did not.

## Correctness result

2 smoke prompts (short + long-context/ring-wrap), both fluent and correct, zero crashes/launch
failures/trap signatures in the server log. **Not** a full GSM8K/needle-retrieval pass — this
iteration is about routing safety (does it launch and produce sane output), not the KV
quantization's own lossy-precision characteristics, which are separately documented elsewhere
(`docs/flags-reference.md:432`: fp8 KV alone is lossy, ~3-6% logit relL2, greedy diverges after
~21 tokens — orthogonal to the FASTPF routing question this iteration answers, and unaffected by
which arm serves it).

## Isolated / complete-object / end-to-end result

Not run — this iteration found no kernel-body change to make (PX-23 already shipped the fix in a
prior session; nothing here is new production code). No performance claim is made.

## Register count / Stack / Spills / Dynamic shared memory

Unchanged from Iteration 0's baseline numbers — no kernel code touched this iteration.
`interp_sm120_pf_fp8kv.cubin`'s load-time smem (89104 B) is below the device's queried opt-in cap
(101376 B), confirmed live via the server's own load-gate log line, not computed by hand.

## Decision: ACCEPT (documentation fix) / no kernel change

## Reason

The routing this iteration was scoped to validate is already correct and already shipped
(`op_attention.cuh:2443`, PX-23, prior session). The actual defect found was **two stale
doc/comment sites** actively contradicting the correct code and a third, already-correct comment
site (`runtime/CMakeLists.txt:450-461`) — a real hazard, since a future session (or a different
engineer) trusting the stale text could leave `FASTPF` off "for safety" and silently eat the -21%
prefill regression at long context for no reason, or spend an iteration re-investigating a trap
that no longer exists. Fixed both stale sites to match the code and cite this report's live
verification. No CMake *default* was changed (`PLOW_FP8_KV_FASTPF` stays OFF at the raw-CMake
level; the served nix package already defaults it ON, `flake.nix:292`) — flipping a project-wide
build default is a broader decision than this iteration's scope and wasn't asked for.

`PLOW_NV_FA_FP8PV` was not re-investigated this iteration: already OFF everywhere including the
served nix package, already documented as non-parity-preserving ("greedy diverges at completion
token 28", `docs/flags-reference.md:448`), and nothing found this iteration changes that
assessment. Correctly off; no action needed.

## Files changed

- `docs/flags-reference.md` — corrected `PLOW_FP8_KV_FASTPF` rows (lines ~392, ~433).
- `runtime/CMakeLists.txt` — corrected the stale comment/`message(WARNING ...)` at the
  `PLOW_FP8_KV`/`PLOW_FP8_KV_FASTPF` option-processing block (~line 255-264). `cmake -S runtime -B
  <dir> -DPLOW_CUDA=ON -DPLOW_SM120_CUBIN=ON -DPLOW_FP8_KV=ON -DPLOW_FP8_KV_FASTPF=OFF` re-run to
  confirm the corrected warning text renders and configure still succeeds.
- `perf-data/sm120-iter1-fastpf-routing-2026-08-26.md` — this report.

## Exact build/serve commands (for reproduction)

```
PLOW_UNISEG=1 plowc --hf-dir /workspace/models/gemma-4-12B-it --gpu rtx5090 --arch sm_120a \
  --fp8-kv --emit-max-chunk 8192 --out assets/gemma4-12b-prefill-fp8kv-full-mc8192

PLOW_BUILD_FP8KV=1 scripts/build_sm120_cubin.sh cubin-fp8kv-fastpf/interp_sm120.cubin \
  -DPLOW_FP8_KV=ON -DPLOW_FP8_KV_FASTPF=ON

plowrt serve --assets assets/gemma4-12b-prefill-fp8kv-full-mc8192 \
  --rt-checkpoint /workspace/models/gemma-4-12B-it-merged \
  --nv-cubin-pf cubin-fp8kv-fastpf/interp_sm120_pf_fp8kv.cubin \
  --nv-cubin cubin-fp8kv-fastpf/interp_sm120_fp8kv.cubin \
  --nv-cubin-sample cubin-fp8kv-fastpf/sample_sm120.cubin --port 8091
```

## Commit

(this iteration's commit follows this report)

## Next experiment

Iteration 2: integrate `px22_ws_stage_bench.cu`'s proven producer/consumer warp specialization
into the production plain w8a8 GEMM body (`op_gemm.cuh`'s `d_gemm_w8a8`), the concrete
already-scoped candidate identified in Iteration 0's baseline.
