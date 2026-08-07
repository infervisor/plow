# Repo-wide consistency & code-reuse review (2026-08-07)

Four parallel reviewers over `crates/plowrt`, `crates/devgen`+`packet`+`plowc`,
`runtime/nvidia`+`runtime/amd`, and the cross-cutting build/test/script layer.
Every claim below was re-verified at both sites. Nothing here was fixed except
the two items marked FIXED.

The unifying theme: **plow's real bugs are not logic errors, they are two
copies of one fact that drifted.** Six of the top ten findings are a constant,
an enum, or a guard restated in a second place — and in three cases the file
that got it wrong carries a comment explaining why getting it wrong is fatal.

---

## Tier 1 — live or latent correctness

### 1. NVIDIA MoE computes the WRONG ACTIVATION (silent)
- ISA `runtime/common/dev_isa.h:374`: `i5=act: 0 gelu_tanh, 1 silu, 2 situ`.
- AMD `runtime/amd/op_moe.h:141`: `GELU_TANH 0` / `SILU 1` — matches.
- NVIDIA `runtime/nvidia/op_moe.cuh:197`: *"act 0 = SiLU, 1 = GELU-tanh"* —
  **inverted**, and the body follows its own wrong comment.
- Emitter `crates/devgen/src/mla.rs:866` `GLM_ACT_SILU = 1`, so on sm_120 every
  GLM/DeepSeek block-fp8 FFN and routed expert (`op_moe.cuh:230,292,360`,
  arm on by default) computes `gelu_tanh(g)·u` instead of `silu(g)·u`.
  K3's `act=2` also lands in the gelu branch; AMD poisons it with NaN.
- Latent only because the GLM/K3 campaigns run on gfx950 — the first NVIDIA
  GLM serve is silently wrong, with fluent output.
- `plow_moe_act` is a **third** private copy of these activations; 19 other
  NVIDIA sites use the shared `sm120_common.cuh:82` enum correctly.
- Fix: delete `plow_moe_act`/`plow_moe_gelu_tanh`, call the shared helpers.
  Add an NVIDIA MoE relL2 case — every MoE numeric test today is gfx950.

### 2. AMD TP decode has no `max_ctx` guard
`exec/amd.rs:5072 decode_prepare_batched` lacks the bounds check that
`decode_step_batched` (`:5122`), `decode_prepare` (`:4448`) and both CUDA paths
(`exec/gpu.rs:3313,3574`) all have — and `exec/amd_tp.rs:581` calls it
**directly**, so on TP an over-long position walks past the KV geometry with no
refusal. `amd_tp.rs` checks `max_ctx` for prefill only (`:696`).
Fix: move the guard into `decode_prepare_batched` — covers B=1, batched and TP
in one edit, and lets `decode_prepare` become a delegate (−18 dup lines).

### 3. `llama3.rs` silently drops Llama-3.1 RoPE scaling
`crates/plowc/src/bin/llama3.rs:709` recomputes RoPE inverse frequencies with
no `rope_scaling`; `cfg_from:72` never reads the field. `packet/src/rope.rs:71`
says: *"ONE function owns the table and nothing else recomputes it."* The file
targets exactly the checkpoints that ship `rope_type: llama3, factor: 8.0`, and
it is in the default build (no `required-features`, unlike `tinygemma`).
`devgen::config::Arch::Llama` already covers this model through the shared
`DenseGqaEmitter`.

### 4. ISA contract violations in NVIDIA flash/MLA
- `op_mla.cuh:181` walks the full `top_k` on a short context, reading index
  slots never written. AMD fixed exactly this (`op_attention.h:1951-1959`
  documents the bug); NVIDIA is still the pre-fix form, and GATHER=true IS
  instantiated (`interp_sm120.cu:1485`).
- `op_attention.cuh:1390` (+5 twins) takes the fused-O path at `nsplit==1`
  with no `O != nullptr` guard, skipping the partials `dev_isa.h:157` says must
  always be produced — null-deref or a stale merge.

### 5. Fused NORM_RESIDUAL_NORM is not bit-exact to its own unfused pair
`dev_isa.h:139` requires it. NVIDIA ops 1/16/21 use `__fdividef` (approx, ≤2ulp)
for `1/feat`; op 23 uses exact `/` (`op_norm.cuh:471`). Every shipped hidden is
non-power-of-two, so the forms genuinely differ, and both are live in one model.
AMD uses exact `/` throughout — so this is also an NV↔AMD mean divergence.
Fix: one `plow_rms_inv(ss, feat, eps)` for all 12 NV + 8 AMD sites.

### 6. AMD's dispatch `default:` is a silent NOP
`runtime/amd/interp.hip:2252` falls through where NVIDIA traps
(`interp_sm120.cu:1711`). The file documents this biting **four** times
(`:1543` — *"the lm_head fell to `default:`, wrote NOTHING, and the logits
stayed zero"*). Live instance: op 80 `GEMV_ARGMAX` has an NVIDIA arm and no AMD
arm, and is not on the reserved list. Fix: `default: __builtin_trap();` plus an
`else` on the flash-bucket `if` at `:1249`.

---

## Tier 2 — FIXED in this pass

### 7. FIXED (`ce58ff4`) — no `plowrt serve --<flag>` invocation ever parsed
Two causes at once: the flattened args were not `global` (knobs had to precede
the subcommand), and the boolish `num_args = 0..=1` shape made a bare `--flag`
swallow the following subcommand name. Every documented example was wrong,
including the canonical campaign command added two commits earlier. Unnoticed
because every script sets the env var instead. Fixed with `global = true`
(72 args) + `require_equals = true` (37 bool flags).

### 8. FIXED (`ce58ff4`) — docs contradicted the code
`flags-reference.md` showed `RuntimeConfig::global()` as the hot-path idiom —
the exact call that panics off the CLI path, fixed in code at `25ea646` — and
asserted "CLI takes precedence over env" without the eight-knob exception where
env deliberately wins so tests can flip values mid-process.

---

## Tier 3 — drift-prone constants (unpinned)

| what | sites | consequence if edited alone |
|---|---|---|
| `PGM_BM` vs host MoE-PF pad | `op_moe.cuh:2104` (sweepable) vs `devgen/src/lib.rs:1303` (`n_exp*128`) | `-DPGM_BM=256` → device writes past `moe_row*` |
| `PGM90_FP8_PROMOTE` | honored by 2 arms, **silently ignored** by 5 in the same build (`op_gemm_sm90.cuh:859,1083,1305,1451,1563`); header default 0, script default 1 | relL2 1.04e-4 vs 1.14e-3 depending on which object claims the packet |
| launch block size | device `plow_block` (384 under WS384) vs C harness `dim3(256)` (`interp_sm120.cu:2160,2201`) | consumer warpgroup never exists → rows 64..127 of every tile unwritten |
| `PLOW_THREADS`/`WAVES` | 4 declarations; `k3.rs` uses two of them in one file | a `-DPLOW_WG_WAVES` change updates one, mis-sizes every workgroup narrowing |
| `GM_LDS_HALVES` | kernel `op_gemm.h:268` vs emitter `lib.rs:1834` hardcoded | 2.29× over-statement on gfx942 → LDS overrun |
| MoE enc slot | `mla.rs:562 DECODE_SLOT` (documented: moving it yields **silent all-zero MoE**) vs `k3.rs:1116,1130,1153,1168` bare `i[6]` literals | K3 emits dead MoE layers |
| `SE_FINE`/`SE_XCTR`/`WG_WAVES` etc. | C `dev_isa.h` ↔ Rust `dev.rs` | `dev_abi.rs` pins struct sizes and opcodes but not these scalars |

The pattern to copy is already in-tree: `plow_arena_bytes` / `PLOW_GEMV_MAXM`
export a capability symbol the loader checks and refuses on mismatch.

---

## Tier 4 — duplication worth collapsing (measured)

- `mux.rs`: eight device-fault stanzas, ~130 lines, two shapes; two have already
  drifted (missing the `model` field the other six carry).
- `op_gemm_sm90.cuh`: four TMA GEMM bodies ~500 lines that the file itself
  proves are one `template <bool E4M3>` (it already uses that template twice);
  `d_quant_fp8` copied verbatim (~70 lines) between the fat and ws384 objects,
  so one edit silently diverges the fp8 activation scale.
- `sm90_tile_remap` applied in 8 of 13 tile loops — and the GLU fork, the
  largest prefill GEMM, is one that misses it. Computed at two sites per body:
  fixing one and not the other stages tile X while computing tile Y.
- `op_attention.cuh`: the causal/window preamble copy-pasted 7×.
- `mla.rs`: ~170 provably output-identical lines across `glm_emit_block` /
  `kimi_emit_block` and four `emit`-block variants; plus two byte-identical
  24-line match arms.
- GPU tests: ~250 lines of setup duplicated across 14 files (`used()` 4×
  byte-identical, the mux-drain loop 5× at ~143 lines) while
  `tests/common/mod.rs` exists and no GPU test uses it. Three gate shapes,
  12 skip strings, three incompatible `PLOW_GPU_ASSETS` policies.
- Scripts: **zero** `source` statements repo-wide; repo root derived 89 times
  under 6 names, serve+poll+cleanup copy-pasted 11× (~220 lines).

---

## Tier 5 — dead code (reported, not deleted, per CLAUDE.md §3)

`env_flag!`/`env_usize!` (`plowrt/src/lib.rs:49,90`, ~80 lines) — zero call
sites since the config migration; `flags-reference.md` still describes them as
serving unmigrated code. Six config fields parsed but never read (`nv.cubin`,
`nv.cubin_pf`, `nv.kernel`, `nv.kernel_pf`, `nv.l2_place_dispatch`,
`amd.trace_raw`) — those flags lie to the operator. Eleven packet-side emit
knobs unreachable from the CLI. `EmitConfig::validate()` never invoked (three
cross-field asserts and both deprecation warnings are dead). Plus ~30 frozen
ablation knobs in the kernels and one `#if defined(...) && 0`.

---

## Test-isolation hazards (cause false-green / flaky CI)

- `devgen/src/mla.rs:8532-8540` sets `PLOW_GLM_WGFIT=0` while
  `blocked_gemv_drops_only_the_empty_ceiling_tail` (`:8560`) reads it live via
  `wgfit()` (`:1784`) — the known ~1-in-7 flake, root cause confirmed.
- `gpu_vmm_prefix.rs`, `hsa_vmm.rs`, `gpu_multimodel.rs` have the same shape and
  "run with `--test-threads=1`" in a comment — **no script passes that flag**.
- `devgen/tests/golden_blob.rs:19,617` already contains the correct primitive
  (`EMIT_LOCK` + `EnvScope` RAII restore) and nothing else uses it.

---

## Highest-value fixes, in order

1. **NVIDIA MoE activation** (#1) — ~3 lines, closes a silent wrong-model bug,
   and deletes the 3rd/4th copy of the activation math. Add an NVIDIA MoE test.
2. **AMD `default: __builtin_trap()`** (#6) — converts every present and future
   AMD coverage gap from silent-wrong to loud. The file documents four incidents.
3. **`max_ctx` guard into `decode_prepare_batched`** (#2) — closes the TP hole
   and removes duplication in the same edit.
4. **Shared `plow_rms_inv`** (#5) — restores the ISA's bit-exactness contract.
5. **`ENV_LOCK` + `EnvScope` in the four racing test files** — lift the
   primitive that already works in `golden_blob.rs`; kills the known flake and
   makes three unenforceable doc comments real.

Runners-up: wire the six ignored CLI flags; one `env_bool_or` helper (six
override sites currently have four incompatible parse semantics, so
`PLOW_VMM_PREFIX=true` *disables* VMM prefix); extend `tests/common/mod.rs`
with a cuda-gated GPU section (−200 lines); `scripts/common.sh` (−700 lines).
