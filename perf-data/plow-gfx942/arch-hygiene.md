# Arch hygiene: one geometry table, a gfx942 instruction-selection contract, and the fork verdict

> **Scope:** no GPU -- hipcc + llvm-objdump only · **CROSS-ARCH BY CONSTRUCTION** — this file is ABOUT gfx942/gfx950 divergence; it is the one place in this directory where both arches are first-class.

Branch `arch-hygiene`, off `worktree-glm52-bringup` @ `529654b`. **No GPU was used**:
everything here is compile-time and test-time. `hipcc` runs on the CPU
(`ROCM_PATH=/opt/rocm-7.2.4`), `llvm-objdump` reads ELF files, and no benchmark,
`amd-bench` or `plowrt serve` was run.

Commissioned after a design review concluded the gfx942/gfx950 kernels should NOT
be forked into separate source trees. The verdict is written up as
`docs/arch/14-amd-arch-divergence.md`; this file is the audit that supports it and
the record of the two guards that came out of it.

---

## 1. The class of bug, and why `GM_LDS_HALVES` was only the instance

`devgen`'s `GM_LDS_HALVES` mirror was the CDNA4 value (73,728 halves) on **every**
part while gfx942's shipped occ4 decode object holds **15,360**. Fused decode
GEMVs stage `x` only through LDS, so the emitter choosing a fused opcode is a
promise that `M*K` fits; at Gemma-4-12B's `hidden = 3840` a host believing 73,728
fused every batch up to M=19 onto an arena that holds four rows. That was fixed
before this branch, with a local `match` in `devgen`.

The `match` removes the instance. It does not remove the class: **device geometry
duplicated into host code, keyed by arch ad hoc.** A sweep for the same shape
found five more copies of the same three numbers.

| copy | file | state after this branch |
|---|---|---|
| decode arena | `devgen::gm_lds_halves` | derived from `hwspec` |
| stage buffers (`GM_DBUF`) | `devgen::stage_buffers` | derived from `hwspec` |
| CDNA3 `requires` list | `devgen::manifest::backend_amd` | derived from `hwspec` |
| CDNA3 probe recipe | `kernelcaps::AMD_PREFILL_DEFINES_CDNA3` | still a `&'static [&str]` (it keys a static table), now **checked** against `hwspec` |
| decode arena, again | `plowc/src/bin/llama3.rs` | derived from `hwspec` |
| device-side LDS ceiling | `PLOW_LDS_MAX_BYTES` in `amd_arch.h` | **checked** against `hwspec` |
| device-side tile/stage defaults | `#ifndef GM_BM ... #if PLOW_CDNA4` in `op_gemm.h` | **checked** against `hwspec` |
| shipped decode profile | `PLOW_OCC4=1` branch in `build_gfx942.sh` | **checked** against `hwspec` |

The single source is `hwspec::IsaLevel::geometry() -> Option<ArchGeometry>`:

| | gfx942 (CDNA3) | gfx950 (CDNA4) |
|---|--:|--:|
| `lds_bytes` | 65,536 | 163,840 |
| `gemm_stage_buffers` (`GM_DBUF`) | 1 | 2 |
| `gemm_tile` (prefill) | 192x256x64 | 256x256x64 |
| `decode_gemm_tile` (shipped) | 128x256x32 (`PLOW_OCC4=1`) | 256x256x64 |
| `decode_arena_halves()` | **15,360** | **73,728** |
| prefill stage bytes | 64,512 of 65,536 | 147,456 of 163,840 |

Corroborated independently by the disassembly: the gfx942 fp8 prefill object
outlines `d_gemm_fp8_t<192,256,64,2,4,...>`, i.e. the header really does
instantiate 192x256x64 on that part.

### The guard

`crates/hwspec/tests/device_header_agreement.rs` — 6 tests. It reads
`runtime/amd/op_gemm.h`, `runtime/amd/amd_arch.h`, `scripts/build_gfx942.sh` and
`scripts/build_gfx950.sh` **as text** and fails when the host table disagrees.
Text and not the preprocessor deliberately: `kernelcaps` already probes these
macros through hipcc, and a check that needs a toolchain is a check that gets
skipped on the machine where the edit is made.

It also asserts the direction nobody writes down — that gfx950's decode tile IS
the header default, by requiring `build_gfx950.sh` to pass no `-DGM_B*` at all. If
that ever changes, the table has to say so.

### Both directions demonstrated failing

Perturbing the HOST table (gfx942 `decode_gemm_tile.bk` 32 → 64):

```
test decode_tile_matches_the_shipped_build_profile ... FAILED
  assertion `left == right` failed: gfx942 decode tile disagrees with the PLOW_OCC4 profile
    left: GemmTile { bm: 128, bn: 256, bk: 64 }
   right: GemmTile { bm: 128, bn: 256, bk: 32 }
test the_decode_arenas_are_the_measured_ones ... FAILED
  assertion `left == right` failed
    left: 27648
   right: 15360
test result: FAILED. 3 passed; 2 failed
```

Perturbing the DEVICE header (`op_gemm.h`'s `!PLOW_CDNA4` `GM_BM` 192 → 128):

```
test prefill_tile_and_dbuf_match_op_gemm_h ... FAILED
  assertion `left == right` failed: gfx942 default tile disagrees with op_gemm.h's !PLOW_CDNA4 arm
    left: GemmTile { bm: 192, bn: 256, bk: 64 }
   right: GemmTile { bm: 128, bn: 256, bk: 64 }
test result: FAILED. 4 passed; 1 failed
```

Both sources restored; the suite is green at HEAD of this branch.

---

## 2. gfx942 had no instruction-selection audit, and the gfx950 one was inverted

`scripts/asm_audit.py` hardcoded `--mcpu=gfx950`, and `asm_expect_gfx950.json`
**forbids** `v_mfma_f32_32x32x16_fp8_fp8` — correct on CDNA4, where selecting it
is half rate, and exactly backwards on CDNA3, where it is the only fp8 matrix
instruction the silicon has.

Measured on freshly built objects (`PLOW_ROWS_ONLY=interp_prefill bash
scripts/build_gfx942.sh`, ROCm 7.2.4 / clang, 18 objects):

| object | `plow_exec` MFMA mix |
|---|---|
| gfx942 `interp_prefill` | 120x `v_mfma_f32_32x32x8_bf16` |
| gfx942 `interp_prefill_fp8` | 120x `v_mfma_f32_32x32x8_bf16` + 56x `v_mfma_f32_32x32x16_fp8_fp8` |
| gfx942 `interp_prefill_mla_moe` | 138x `v_mfma_f32_32x32x8_bf16` |
| gfx950 `interp_prefill` (control) | 116x `v_mfma_f32_32x32x16_bf16` |

So the contracts are inverses on **both** matrix axes, not just fp8. One
expectation file cannot state both.

**Spelling note:** ROCm 7.2.4 / clang-23 prints the CDNA3 bf16 MFMA as
`v_mfma_f32_32x32x8_bf16`. The `..._bf16_1k` spelling in older AMDGPU asm printers
names the same instruction; the new `require_min` entries match the common prefix
so a toolchain bump does not read as an instruction-selection regression, while
the `forbid` entries stay exact.

### Changes

- `asm_audit.py` reads each object's arch out of its **own ELF header**
  (`Flags: 0x54c, gfx942, ...`) and uses it for `--mcpu`. A caller-supplied
  `--mcpu` is a second opinion about a fact the file already states — and in this
  case llvm-objdump was overriding the hardcoded flag from the ELF anyway, which
  is why the wrong-arch audit produced a *correct disassembly checked against an
  inverted contract* rather than garbage.
- Expectation files declare a top-level `_arch`, asserted against every audited
  object before any rule runs.
- `scripts/asm_expect_gfx942.json` — the CDNA3 contract, covering the three
  prefill families. Decode is deliberately absent: gfx942 decode ships
  `PLOW_OCC4=1`, a different tile and register budget, and an expectation written
  against a non-occ4 decode object would describe an object nobody serves.
- `scripts/build_gfx942.sh` runs the audit as a build gate, the twin of the one in
  `build_gfx950.sh`. Its exit code is captured rather than piped: `cmd | tail`
  reports `tail`'s status, so piping would print FAIL lines and then declare the
  build ready.

### PASS

```
$ python3 scripts/asm_audit.py --expect scripts/asm_expect_gfx942.json build-amd/hsaco/gfx942/*.elf
PASS  all gfx942 assertions held over 18 audited object(s)
exit=0
```

Control, unchanged behaviour on the arch that already had a contract:

```
$ python3 scripts/asm_audit.py --expect scripts/asm_expect_gfx950.json /workspace/assets/gfx950-cprv/interp_prefill.elf
       116  v_mfma_f32_32x32x16_bf16
PASS  all gfx950 assertions held over 1 audited object(s)
exit=0
```

### FAIL — wrong-arch pairing

```
$ python3 scripts/asm_audit.py --expect scripts/asm_expect_gfx950.json build-amd/hsaco/gfx942/interp_prefill.elf build-amd/hsaco/gfx942/interp_prefill_fp8.elf
FAIL  interp_prefill.elf: built for gfx942, but these expectations are for gfx950 — the fp8 and
      bf16 MFMA contracts are INVERTED between the two CDNA levels, so this audit would assert
      the opposite of the truth
FAIL  interp_prefill_fp8.elf: built for gfx942, but these expectations are for gfx950 — ...
exit=1
```

### FAIL — perturbed expectations

`require_min` on the fp8 MFMA raised to 1000 (stands in for "the fp8 arm vanished")
and `v_mfma_f32_32x32x8` added to `interp_prefill`'s `forbid` (the gfx950 file's
polarity, applied here):

```
FAIL  interp_prefill.elf: plow_exec: 120 forbidden 'v_mfma_f32_32x32x8'
FAIL  interp_prefill_fp8.elf: plow_exec: 56x 'v_mfma_f32_32x32x16_fp8_fp8' < required 1000
      (instruction selection changed)
exit=1
```

---

## 3. Decode-GEMV tuning on gfx942 — ALREADY DONE, no change needed

The commissioned work ("`GFX950_CELL` is hardcoded at ~7 sites in
`tunedb-gemv.rs`") had already landed on the base branch as commit `7a85d31`,
*"tunedb: the decode-GEMV cell follows --gpu, so a gfx942 campaign is possible at
all"*. Verified rather than assumed:

- `tunedb-gemv` resolves a `Target { cell, isa, sku }` once from `--gpu` via
  `tunedb::amd_tuning_cell`, threaded to every site. `--gpu` is required.
- `IsaLevel::Gfx950` appears nowhere in the binary outside comments.
- Its rows are filed under `profile: "decode_gemv"`, not `prefill_dense` — the
  writer already separates the two rooflines so `best_for` cannot rank one against
  the other at a coincidentally equal shape.
- The reader half (`plowc tune`, `devgen::amd_tuning_cell`) keys off `--gpu` too,
  and `crates/tunedb/src/gemm.rs` carries the named regression test for the writer
  defect.

`GFX950_CELL` still exists, correctly: it is the historical cell label (MI355X
silicon measured into `amd/gfx950/mi350x`), used only to preserve that provenance
and as the non-gfx942 default. Nothing to do. **No tuning run was invented.**

---

## 4. Fork verdict — the measurement behind `docs/arch/14`

Counting rule: non-comment lines in `runtime/amd` mentioning `PLOW_CDNA4`,
`PLOW_HAS_MX_CVT`, `PLOW_HAS_MX_MMA`, `PLOW_LDS_MAX_BYTES`, `__gfx950__` or
`__gfx942__`.

| file | lines | arch sites | % |
|---|--:|--:|--:|
| `amd_arch.h` | 408 | 15 | 3.68% |
| `amd_common.h` | 1165 | 12 | 1.03% |
| `op_gemm.h` | 4988 | 8 | 0.16% |
| `op_attention.h` | 5121 | 7 | 0.14% |
| `op_moe.h` | 4644 | 6 | 0.13% |
| `op_norm.h` | 833 | 2 | 0.24% |
| `test_kernels.hip` | 1039 | 4 | 0.38% |
| `op_collective.h`, `op_k3.h`, `op_kda.h`, `op_elementwise.h`, `interp.hip` | 5648 | 0 | 0.00% |
| **total** | **24,373** | **54** | **0.22%** |

15 of the 54 are inside `amd_arch.h`, the file whose job is to hold them; the op
bodies carry 39 over 23,965 lines (0.16%). 31 of the 54 are true `#if`/`#elif`
directives.

The review reported ~100 sites over 14.5k lines (0.7%) on a wider rule that
included host-side and build-script conditionals. Both numbers are two orders of
magnitude below the ~15% split threshold, which is all the number has to decide.
Recorded here so the next person re-measures rather than re-argues.

---

## 5. Test state

Baseline on this box, `cargo test --workspace --no-fail-fast`:
**109 suites `ok`, 1 FAILED** — `devgen --test tuned_tile_selection`, 2 of 4 tests.

Both failures are the *same* pre-existing gfx950 tuning-cell staleness:

```
tunedb amd/gfx950/mi350x: 3080 record(s) skipped as STALE against the probed build
gfx950-1ad5483e08645ac0 -- NO usable records remain, so tile selection fell back to
the analytical model.
```

**It is left failing on purpose.** There is no gfx950 hardware on this box
([[plow-devbox-is-gfx942]]), so "fixing" it means re-stamping measurements that
were never taken — strictly worse than a failing test, and precisely the
silent-emptiness failure that test exists to make loud.

After this branch: **1 suite still failing, the same one, with 1 of 4 tests failing
instead of 2.** The extra pass is not attributable to anything here — both tests
fail on the same stale-cell condition and only one of them observed it on that run;
nothing in this branch touches the tuning store, the build digest, or the probe.
No new failures.

---

## 6. Files

| file | change |
|---|---|
| `crates/hwspec/src/isa.rs` | `GemmTile`, `ArchGeometry`, `IsaLevel::geometry()` |
| `crates/hwspec/src/lib.rs` | re-export |
| `crates/hwspec/tests/device_header_agreement.rs` | new — the drift guard |
| `crates/devgen/src/lib.rs` | `gm_lds_halves` / `stage_buffers` read the table; `GM_LDS_HALVES_CDNA4` deleted |
| `crates/devgen/src/manifest.rs` | CDNA3 `requires` derived from the table |
| `crates/kernelcaps/src/targets.rs` | `cdna3_recipe_matches_the_arch_geometry` |
| `crates/plowc/src/bin/llama3.rs` | its own CDNA4 `GM_LDS_HALVES` copy removed |
| `scripts/asm_audit.py` | arch read from the ELF; `_arch` contract |
| `scripts/asm_expect_gfx942.json` | new — the CDNA3 contract |
| `scripts/asm_expect_gfx950.json` | `_arch: gfx950` |
| `scripts/build_gfx942.sh` | instruction-selection gate |
| `docs/arch/14-amd-arch-divergence.md` | new — the verdict |
| `docs/arch/00-overview.md` | index |
