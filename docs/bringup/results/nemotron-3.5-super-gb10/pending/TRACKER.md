# GB10 pending experiments: tracker (2026-11-26)

This directory sits in a top commit marked WIP; drop that commit before merge. It holds every
in-flight experiment as a patch, because the jg02 box is being handed back. None of these patches is
gated, and none of them is in the consolidated commit below this one. The consolidated commit holds
the gated work: the GB10 port, Nemotron and Qwen bring-up, the decode loops, the prefix cache,
hd256 prefill, the ubench, the NVFP4 study and the serving scoreboard.

Base for every patch: the consolidated commit (tree = 92de9f1d8 plus docs). Apply with
`git apply --3way <patch>` from the repo root. `scripts/` holds the gate and driver scripts the
agents used; they are scratch-quality and reference host-specific paths. `nx.sh` sourced a captured
`nix print-dev-env` because AppArmor blocked user namespaces after a reboot. On a box with working
`nix develop`, use `nix develop --command` instead and drop `PLOW_CAMPAIGN_NO_NIX=1`.

| # | Experiment | Patch | State | Next step |
|---|---|---|---|---|
| 1 | Mamba-2 CHUNK_OUT grouped body (v1) | `mamba-chunk-out-v1.patch` | Single SM: bitwise, 3.5× | Run gate `scripts/mamba/g1.sh` |
| 2 | Mamba-2 CHUNK_OUT dev variants | `mamba-chunk-out-dev.patch` | Dev macros, never landed | Pick v2 from the j2 split |
| 3 | FP4 W4A4 TMA as a separate kernel | `fp4-tma-split-kernel.patch` | g4 failed at 1024 rows (cause found); fix not yet measured | Run gate `scripts/fp4loop/g5.sh` |
| 4 | Merge hardening (GB10 knobs) | `merge-hardening.patch` | Does not compile | Fix the 2 errors, then CPU suite and SASS diff |
| 5 | Qwen FP8 TC row-invariant GEMV | `qwen-fp8tc-row-invariant.patch` | Built, not verified | Spec identity at B=4 |
| 6 | Attention: native vs libraries | `libattn-bench.patch` | Bench written, never run | Run it, then write the comparison table |
| 7 | Qwen hd256 decode (per-case GF, staging) | `hd256-decode-rgm-dev.patch` | Dev, not gated | Pick GF per case, then gate |

## 1. Mamba-2 CHUNK_OUT grouped body (v1)

The patch adds `d_mamba2_chunk_out_grp<PK>` to `runtime/nvidia/op_mamba_pf.cuh`, routed only when
the entering states are bf16, which is the production fused-scan path. The new knob
`PLOW_NV_MAMBA_OUT_GRP` (default 1) is registered in `crates/devgen/src/knob_spec.rs`.
`runtime/tests/mamba2_oracle.py` now launches with 73728 B of shared memory.

How the body works:
- One block per (chunk, group, run of heads). C·Bᵀ is computed once per group and kept in registers.
- E, x, cum and dt for the next head load with `cp.async`, double-buffered.
- Rows map to warps so causal work is balanced across warps.
- The accumulation order is the same as HEAD, so the output is bitwise identical.
- The staging arena grows from 55296 to 73728 B, still under the W4A4 arena (about 92 KB).

Single-SM ubench (production defines, 8192 rows, `compilers_during_run=0`):

| case | slice max µs | whole op on 1 SM µs | % of 2.64 TF/SM | digest |
|---|---:|---:|---:|---|
| chunk_out, HEAD | 3128.3 | 142370 | 9.2% | fd2add985cae8c68 |
| chunk_out, v1 | 898.4 | 41557 | 32% | fd2add985cae8c68 (bitwise) |
| chunk_scan_state, HEAD / v1 | 1618 / 1624 | | | efc906746bf2507d (same) |

Registers and stack are unchanged (255 regs, 1504 B stack).

Full-GPU estimate: with bf16 entering states this op is DRAM-bound. The floor is about 1.7 ms against
2.47 ms today, a saving of about 0.7 ms per Mamba layer, or about 28 ms per 8192-row chunk over 40
layers. `op_roof.py` prices the entering states as f32 and overstates that floor; fix the pricing
too.

Gate `scripts/mamba/g1.sh` (`ROOT=` the candidate tree) runs:
- the build and knob tests;
- a four-arm `nv-block-ab` on layer 6 at 1024, 4096, 8192 and 16384 rows, then `certify_kernel.py`;
- the block oracle on layer 6, and on layers 6–8 with `--kv fp8 --route-check`;
- `packed-check`, 5 cases;
- TTFT ABBA at 8K and 32K, starting each arm at ≤58 °C.

It has never run. If it passes, apply the patch and add "Single-SM inner loops: Mamba-2 prefill" to
comparison.md, block-vs-roofline table first.

Next levers:
- **Gated-norm fusion into chunk_out.** One v1 block holds exactly one norm group (16 heads × 64).
  Fusing the RMSNorm and FP8 quant into the epilogue removes the y write and re-read: about 268 MB
  per layer at 8192 rows, about 1.1 ms per layer, about 45 ms per chunk. This is the bigger lever.
- **v2 inner loop.** v1 spends about 5 µs per head against about 1.3 µs of MMA. The variants in
  `scripts/mamba/j2.sh` (v0–v6) split that time between C·E and P·x, non-volatile asm, and P
  precomputed.
- **chunk_scan_state.** At 8% per SM because global loads are synchronous; prefetch the next chunk.

## 2. Mamba-2 CHUNK_OUT dev tree

`mamba-chunk-out-dev.patch` is the same body plus the `MPF_OG_{RBMAP,NV,SPLITP,DIAG}` dev switches
that j2 used. Use it only to explore v2; it must never land.

## 3. FP4 W4A4 TMA as a separate kernel

g4 tried an emitter threshold alone: `W4A4_TMA_MIN_ROWS = 2048` in
`crates/devgen/src/components/nvfp4_moe.rs`.

| Measurement | Result |
|---|---|
| L1 A/B, 4096 rows | −1.460 ms, PASS |
| L1 A/B, 8192 rows | −0.598 ms, PASS |
| L1 A/B, 16384 rows | PASS |
| L1 A/B, 1024 rows | +0.165 ms (+1.8%), REGRESSION |
| Certify | QUALIFIED |
| Oracle | L1 20/20; L6–8 22/25 in both arms (pre-existing decode B=4 row) |
| packed-check | 5/5 |
| TTFT 8K, first pair | C 1.915 s, T 1.885 s |

The 1024-row cause was codegen, not the TMA feed. With TMA instantiations in the same call graph,
the cp.async walks spill:
- `plow_moe_w4a4_pf_pfseg` stack grows from 16 to 64 B;
- STL goes from 4 to 170, LDL from 10 to 81, CALL from 44 to 198.

The same cause explains g3's losses at 128 and 1024 rows.

The fix in this patch (cand3):
- The cp.async entries `d_moe_{up,down}_nvfp4_pf_w4a4` and `plow_moe_w4a4_pf` go back to HEAD
  byte-for-byte.
- TMA gets its own entries: `d_moe_*_w4a4_tma`, plus a `plow_moe_w4a4_pf_tma` kernel and its
  `_tma_abi` symbol in `interp_sm120.cu`. They compile only under `PLOW_NV_W4A4_V2_TMA`.
- `crates/plowrt/src/exec/gpu.rs` loads `SegPf.moe_w4a4_pf_tma` when the ABI symbol exists, and
  launches it when the instruction has `i[7] != 0`.
- The TMA map uses `GEN_TMAP_U8_SW64 = 11` (SWIZZLE_64B).

The tests that ran passed (packet 167, plow-asset 367, devgen knob 18, nemotron 16, plowrt knob 28).
Gate `scripts/fp4loop/g5.sh` never finished: the first try produced empty emits (nested nix). The
fixed script was queued and lost with the box.

That gate:
1. Fails unless the SASS of `plow_moe_w4a4_pf_pfseg` is identical between T and C.
2. Runs the L1 A/B at 128, 1024, 2048, 2176, 4096, 8192 and 16384 rows.
3. Runs the oracle, packed-check and TTFT ABBA at 8K and 32K.

If it passes:
- Rewrite the `W4A4_TMA_MIN_ROWS` comment; the "box overhead" rationale in it is wrong.
- Decide whether the 2048 threshold is still needed.
- Re-add the README line and the comparison.md subsection.

## 4. Merge hardening (GB10 knobs)

The goal: non-GB10 targets keep pre-branch behaviour by default, and the GB10 recipes opt in
explicitly. The patch:

- Adds five RuntimeConfig knobs, default off, registered in `crates/plowrt/src/knob_spec.rs` and
  `docs/flags-reference.md`; both GB10 recipes set all five to 1:
  - `PLOW_PF_COLD_BURST`
  - `PLOW_PF_TICK_BUDGET`
  - `PLOW_PF_CHUNK_COST_FIT`
  - `PLOW_SPEC_AFTER_PREFILL`
  - `PLOW_VMM_PROMPT_END_EXPECTED`
- Fixes the burst-wait race:
  - `Ingress` gets a `rendering` count, a `rendered` Notify and a `cold_burst` flag.
  - A new `cold_burst_wait()` in `serve/mux.rs` wakes on rx, on a renderer finishing, or on the SLO
    deadline.
  - Three tests are added in `serve::mux::tests`.
- Changes the `PLOW_NV_FP8TC_PF` default to `PLOW_SM12X`. That also turns it on for RTX 5090, which
  is unmeasured there.

CPU suite on this patch (merge-cpu): devgen 809, packet, plow-asset, hwspec, costmodel, kernelcaps,
tunedb, nn-graph, rewrite, the Python self-tests and check_recipe all pass. Two failures remain:
- The plowrt lib test does not compile: `error[E0507]: cannot move out of 'm' in pattern guard`
  (new code in `serve/mux.rs`, `cold_burst_wait`).
- An earlier compiled run had 1 failure out of 2229: `knob_spec::tests::every_runtime_arg_has_exactly_one_spec`
  (`crates/plow-asset/src/knob.rs:797`). One of the new knobs is missing its spec or has two.

Still to do:
- Fix both failures and re-run `scripts/merge/harden_c.sh`.
- Run `harden_d.sh` and `res_diff.py`: ptxas register and spill diffs for sm_89, sm_90a and
  sm_120a, base vs hardened.
- Insert `scripts/merge/e_note.md` before `## Protocol` in comparison.md.

Still needs a run on a non-GB10 GPU:
- FlashMerge at 2 items per block on sm_89 and sm_90a. It is ungated on every NVIDIA arch.
- FP8TC_PF on RTX 5090, or restrict it to sm_121.

Merging also needs:
- a rebase onto origin/main, with three text-only doc conflicts: `agent-tools.md`, `isa/sm120.md`
  and `target.md`;
- a check that `interp_sm120.cu` changes are safe in sm_90a and sm_89, since `interp_sm90a.cu` and
  `interp_sm89.cu` include it.

## 5. Qwen FP8 TC row-invariant GEMV

Spec identity breaks at B=4 because `d_gemv_fp8` routes on M (`op_gemm.cuh` around line 3582). The
target and MTP verify rows then take different accumulation orders.

The patch makes the TC GEMV reduction order independent of the number of rows. It touches:
- `runtime/nvidia/op_gemm.cuh`
- `runtime/nvidia/op_gemv_fp8_tc.cuh`
- `crates/devgen/src/{knob_spec.rs, manifest.rs, emit_config.rs}`
- `runtime/tests/sm120_interp_op_test.cu`

j2 built the C, R and T cubins and the Nemotron base and dev cubins. The Nemotron cubin sizes are
identical (3661688 B), which is expected because the change is a no-op there. No verdict was
recorded.

Still to do:
- Row-invariance test: `rows_test.cu` and `mkrows.py`.
- Qwen spec identity at B=1, 2, 4 and 8 with the round-to-nearest packet. The SR packet never passes
  identity.
- Plain-decode A/B.

`PLOW_FP8_DECODE_TC="1"` is already in the experimental Qwen recipe.

## 6. Attention: native vs libraries

`scripts/gen_kernels/bench_lib_attn.py` has about 550 new lines. It benchmarks plow native attention
against FlashInfer, cuDNN SDPA, Triton (the vLLM TRITON_ATTN path) and the torch flash backend.
Shapes:
- Nemotron: hd128 GQA, decode and prefill.
- Qwen: hd256.

The job sat in the queue and never ran. Facts already established:

| Backend | Facts |
|---|---|
| Repo FA4 AOT | FA4's SM90 forward (sm_90a cubins). Cannot run on sm_121 (no wgmma). FA4's Blackwell path needs sm_100 tcgen05/TMEM, which GB10 lacks. Not a candidate. |
| vLLM on GB10 | TRITON_ATTN for Nemotron's LLM layers, FLASH_ATTN for the ViT |
| Plow native | FA2 hd128 E4M3 prefill at 66% of peak; FlashDecode with 0 gap; hd256 pf256 at 52% |

Still to do: run the bench, add "Attention: native vs libraries on GB10" to comparison.md, and
follow the bench's own self-test convention.

## 7. Qwen hd256 decode (dev)

`hd256-decode-rgm-dev.patch`: per-case GF choice and staging for hd256 FlashDecode. Not gated.

## Serving follow-ups not started (from the scoreboard)

- **Agentic multi-turn c8 TTFT** loses (p50 2.41 s vs 2.17 s for vLLM). Serial prefill ticks take
  53% of wall at about 680 ms each, and decode rides only 12 of 83 prefill ticks. Fix prefill speed
  first (items 1 and 3 above), then let decode ride prefill ticks.
- **Shared-prompt c1 TTFT** loses (1.84 s vs 1.50 s for vLLM). Needs recurrent-state capture inside
  the launch.
- **Vision tower** is not emitted. All results so far are text-only.
- **jg01 grid.** The matched long-context plow grid on jg01 (ISL 1000, 8192, 32768, 131072 and
  262000) was in progress at hand-back. Results:
  - plow, per ISL: `jg01:/home/ifv/plow-scratch/res/grid-plow-mtp/i<ISL>`;
  - vLLM grid (complete): `jg01:/home/ifv/plow-scratch/res/`.

## Environment notes

- After a reboot, jg02's AppArmor sets `kernel.apparmor_restrict_unprivileged_userns=1`. That breaks
  `nix develop` in both bwrap and proot modes inside gpuq jobs. The fix is
  `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, persisted under
  `/etc/sysctl.d/`.
- Inside a job that already has the flake env, `campaign.py build` runs a nested `nix develop`. That
  fails quietly and leaves empty emits. Set `PLOW_CAMPAIGN_NO_NIX=1` there, and fail loudly when
  `model.pkt` or a cubin is missing.
