# The object contract

The object-side twin of [the dispatch audit](15-dispatch-audit-and-knob-manifest.md). That one
asserts what the **emitter** decided, from the packet it wrote. This one asserts what the
**compiler** did, from the code object it produced — and it runs on every
`scripts/build_gfx942.sh`.

## The bug class

The dispatch audit exists for the *compiled ceiling*: a compile-time constant that does not
match the shape the runtime presents. This one exists for its neighbour — **a fact about the
object that nobody could see**. Six of them cost real performance on this branch, and every one
was found by hand, by disassembling:

| | what was wrong | what it cost |
|---|---|---|
| a `-D` that did not take | `op_gemm.h` declared five per-rung GEMM knobs (`GM_SM_BK`, `GM_MD_*`) as **bare `#define`s** while `build_gfx942.sh` documented them as reachable through its `GM_AX` raw-`-D` hatch. A bare `#define` after a command-line `-D` is a redefinition the **header wins**, and the recipe compiles `-w`. Every A/B ever run through that hatch measured an *unchanged object*. | a published "`GM_SM_BK=128` is a flat null" row that is **−15.2% TTFT at 128 tokens** once the guards are in |
| silent spill | candidates have reported `.vgpr_spill_count: 0` and spilled anyway (119 `scratch_load_dword` in one case) | the whole point of a register budget |
| LDS-crossbar reductions | `__shfl_xor` lowers to `ds_bpermute` on gfx9. One dense flash-prefill KV tile was 1651 instructions, **160 `ds_bpermute` and 163 `s_waitcnt` against 64 MFMA** | DPP + one `ds_swizzle` → 64 permutes, **−3.5% TTFT** |
| 2-byte LDS reads | the MLA prefill PV transpose contracts over the minor axis of the `[kv][d]` slab and pays it as 8 strided `ds_read_u16` per output tile, **256 per KV tile** | the largest item in that loop — a *deliberate* trade |
| a dispatch arm that does not exist | `hd=64` had **no arm** in prefill, decode or either merge, and no `else`. On AMD a missing arm does not trap. | GPT-OSS attention silently wrote **nothing** |
| a stale resource table | `build_gfx942.sh` carried the table as a header comment claiming `interp_prefill` spill **6** | measured 126 by the note, **1799** scratch ops by the ISA |

## What runs

`scripts/asm_audit.py`, from the bottom of `scripts/build_gfx942.sh`, in the **same invocation**
as the instruction-selection gate so both read one disassembly pass. 45 objects, ~25 s, threaded
over `PLOW_AUDIT_JOBS` (8).

```
asm_audit.py --quiet --expect asm_expect_gfx942.json \
             --contract obj_baseline_gfx942.json --defines build_defines.json ./*.elf
```

Counting lives in **`scripts/plow_isa.py`**, shared with `scripts/kx_isa.py` (`kx --isa`). Two
copies of the regex table meant two definitions of `spill`; one module means a number measured
in the `kx` harness is the number the build asserts.

| # | check | verdict | what it asserts |
|---|---|---|---|
| 1 | macro | **FAIL** | every geometry `-D` the build passed appears in the object as a `plow_geom_<MACRO>` marker holding the value that *compiled*; and every `#ifndef`-guarded knob in the marked headers has a marker |
| 2 | spill | **FAIL** | scratch traffic counted from the ISA, against a per-object budget; a kernel whose note claims 0 while its ISA disagrees |
| 3 | permute | report | bodies whose `ds_bpermute` count is disproportionate to their MFMA count, with `s_waitcnt` density |
| 4 | lds16 | report | bodies issuing a KV tile's worth of 2-byte `ds_read` per MFMA |
| 5 | arms | **FAIL** | every head dim the object must serve is present as a template instantiation |
| 6 | budget | **FAIL** | VGPR / AGPR / LDS / occupancy against a committed per-object budget |

Four refuse and two report, by the rule the emit side already states: a check refuses when the
object is *wrong*, and reports when the object is a deliberate trade whose number is worth
watching. `PLOW_AUDIT_STRICT=1` promotes the reports to refusals — the same variable, the same
meaning, as on the emit side.

## The marker symbols

Class 1 needs the object to *state* what compiled. `runtime/amd/geom_contract.h` emits one
`__device__` word per knob:

```c
#define PLOW_GEOM_MARK(m) extern "C" __device__ unsigned plow_geom_##m = (unsigned)(m);
```

One level of macro, deliberately: expansion of an argument is decided *per occurrence*, so `m`
next to `##` stays the macro's **name** while the `(m)` in the initialiser expands to its
**value**. A second level pastes `plow_geom_` onto the value and fails to compile for every
expression-valued knob.

This is the tree's existing idiom (`plow_gemv_mm_cap_4`, `plow_mixed_block`,
`plow_mla_pf_v2_fp8_arm_1`) with one difference: those encode the value in the **name**, because
plowrt reads `.symtab` before the object is on a device and a value would cost a device round
trip on the load path. Nothing on the load path needs these, so the value is read out of `.data`
instead (`.bss`, and therefore 0, when the knob is 0) — which is what lets a marker carry
`FA_FAST_RCP`, whose value is `(!PLOW_CDNA4)` and cannot be pasted into a token.

86 markers, ~350 bytes of `.data`, no code.

**The roster is checked, not trusted.** The audit re-derives the knob list by scanning
`amd_arch.h`, `op_attention.h`, `op_moe.h` and `op_gemm.h` for `#ifndef (GM_|FA_|GV_|MPF_)*`,
and fails when one has no `PLOW_GEOM_MARK` line. It fails in the other direction too: a knob
that is marked but is no longer `#ifndef`-guarded is a knob whose bare `#define` now silently
beats the command line — which is the `GM_SM_BK` defect itself, catchable without anyone
passing a `-D`.

`build_gfx942.sh` writes `build_defines.json` next to the objects, composed from the same
`$ROWS` and the same `$AX_GQ` branch the compile loop uses one line up, so the recorded `-D` set
cannot drift from the compiled one.

## The baseline

`scripts/obj_baseline_gfx942.json`, committed, re-blessed with `--bless`. A regression is a line
in a `git diff` that has to be explained in a commit message — the property the emit-side audit
was built for.

```
objects.<stem>   defines         the exact -D set the row was blessed from
                 vgpr agpr lds occ    the resource table, from the object's own AMDHSA note
                 meta_spill      .vgpr_spill_count — recorded to be DISBELIEVED
                 isa_spill       scratch_load_*/scratch_store_* over every symbol: the budget
                 unmetered_spill the part of it in symbols with no note at all
                 meta_lies       kernels whose note says 0 while their ISA says otherwise
                 bpermute lds16  the two advisory counts
                 arms            d_flash_* instantiations, by head dim
                 geom            which shared geometry profile the object compiled
geom_profiles    the 86 knob values, shared across the rows that compile identically
```

A row is **only compared when the axes match**. A `PLOW_OCC4=1` or `PLOW_DECODE_BATCH=4` build
compiles a different object; it is still checked by the rules that need no baseline (the
geometry markers, the note-vs-ISA disagreement, head-dim coverage) and is not held to numbers
describing a different compile. Six geometry profiles cover the 45 default objects, so one knob
moving is one changed line rather than forty.

`isa_spill` is a ceiling in one direction only: exceeding it fails, undercutting it asks to be
re-blessed so the improvement is held.

## Thresholds, and where each number comes from

Every threshold is a measured number from this branch, not a taste.

| threshold | value | measured from |
|---|---|---|
| permute per MFMA | 2.0 | the pre-fix `d_flash_prefill<256>` KV tile: 160 `ds_bpermute` against 64 MFMA = **2.5**; the DPP form of the same body is **0** permutes and 64 `ds_swizzle`. The threshold sits between two forms of one body. A softmax spending one wave reduce per MFMA (1.0) is doing ordinary work; at 2.0 the crossbar is the loop. |
| permute floor | 64 | one permute per lane-quarter per wave — keeps the small norm/argmax helpers, whose reduce *is* the kernel, out of the report |
| 2-byte reads per MFMA | 1.5 | the compiled `d_flash_mla_prefill_v2<512,64,false,true>` shows 256 2-byte reads against 136 MFMA = **1.88**; a loop staging its operands `b128`-wide is ~1.0 |
| 2-byte read floor | 256 | exactly one KV tile's worth, the figure `op_attention.h` states. The dense flash arms at 128 reads are deliberately under it — their transpose is half the width and is not the item this reports. |
| occupancy | derived | 512 VGPRs per SIMD at an 8-register granule, and LDS workgroups × waves ÷ 4 SIMDs, min of the two. Checked against the numbers `build_gfx942.sh` states from measurement: VGPR 253 → 2, VGPR 104 with a 30,736 B arena → 4, VGPR 512 → 1. |

`.vgpr_count` is **already** the unified arch+acc total on gfx90a and later; `.agpr_count` is its
accumulator *subset*, not an addition. Summing them says 768 for a register file that holds 512.

## What it found on the current objects

Nobody had looked at these.

**`.vgpr_spill_count` is blind to the bodies that spill.** It is a *per-kernel* note, and every
outlined `__device__` body — `plow_exec`, `d_gemm_glu`, `d_flash_prefill<D>` — has no note at
all. The cliff table prints `interp_flash … spill 0`; its four flash arms issue **154 / 226 /
460 / 708** scratch instructions. Across the default set, 1548 of `interp_flash`'s 1548 scratch
ops and 1588 of `interp_prefill`'s 1799 are in symbols the metadata cannot see. The `unmetered`
column is that number, per object.

**Two objects under-report outright.** `interp_mixed`'s own kernel issues 43 scratch ops with
`.vgpr_spill_count: 0`, and `test_kernels`' `mla_flash_decode_mfma_512` /
`mla_gather_decode_mfma_512` issue 168 each. These are the class the check was written for,
alive in the current build.

**The DPP softmax fix is flash-object-only, and the prefill interpreter still pays the
crossbar.** `PLOW_FA_RED_DPP` is added to `AX_FLASH` and to nothing else, for a documented
register-cliff reason. So in every 8-wave prefill object `d_flash_prefill<256>` and `<512>`
still issue **160 `ds_bpermute` against 24 MFMA** (6.7×, `s_waitcnt` 7.9/100) and `<64>` 160
against 12 (13.3×), while the same bodies in `interp_flash` issue **zero** permutes and 64
`ds_swizzle`. The advisory names them on every build.

**A grouped-MoE body nobody has looked at.** `d_moe_group_down_pf` issues **384 LDS permutes and
608 `s_waitcnt` against 32 MFMA** — 12.0× and a 12.1/100 wait density, the highest in the tree,
higher than the flash body whose fix was worth −3.5% TTFT. `d_moe_group_down_gemma_pf` is 64
against 16 (4.0×). Neither is a failure and neither is claimed to be a win; they are the two
bodies the measured threshold points at next.

**The header table was wrong by two orders of magnitude.** `interp_prefill` is VGPR 256 / AGPR 0
/ LDS 64,560 / occ 2, note spill **126**, ISA spill **1799**. The comment said LDS 64,520 and
spill 6. It is now derived and asserted rather than typed.

## Reproducing the defects

Each class was re-broken and the audit re-run.

**Class 1** — the `#ifndef` on `GM_SM_BK` reverted to a bare `#define`, built with
`GM_AX="-DGM_SM_BK=64"`:

```
FAIL  interp_prefill: -DGM_SM_BK=64 did NOT take — the object compiled GM_SM_BK=128
      (a bare #define in the header wins over the command line, and the recipe compiles -w)
FAIL  geom_contract.h: marks GM_SM_BK, but no header #ifndef-guards it any more
      — a -DGM_SM_BK would be silently overridden by the header
```

With the guard restored, the same build passes and the object carries `GM_SM_BK = 64`.

**Class 5** — the `hd=64` arm deleted from both `d_flash_merge` chains in `interp.hip`:

```
FAIL  interp_prefill: d_flash_merge has no arm for head_dim [64] — on AMD a missing arm
      does not trap, the dispatch falls through and WRITES NOTHING
FAIL  interp_prefill: flash arm set {'d_flash_merge': [128, 256, 512], ...}
      != baseline {'d_flash_merge': [64, 128, 256, 512], ...}
```

The pre-fix `FA_DBUF` V-slab overrun (`VPT = BKV * VL / …` with `VL = FA_DC`, which read past
each V row and wrote past `Vsm` into `Psm`) is caught too, by class 2 — the wider prefetch holds
more registers:

```
FAIL  interp_flash: 1572 scratch ops > budget 1548
```

**Class 3** — the two softmax arms of `runtime/bench/amd/kx/exp_fa_softmax.hip`, through the
shared counter:

```
ISA  variant 'shfl'          total  scratch bpermute  swizzle     mfma  waitcnt
k_soft                         804        0      160        0        0      122
ISA  variant 'dpp'
k_soft                         748        0        0       64        0       47
```

which is the documented 160 permutes → 64 swizzles, at 73 → 65 VGPR.

**Class 6** — the stale figure is gone from the script header and the real one is in the
baseline, asserted.

`d_flash_merge` is the family the head-dim rule keys on, and the choice is measured: it survives
inlining in **44 of 44** objects that have it, while `d_flash_prefill<128>` is inlined into
`plow_exec` in every 8-wave prefill object and `<128,true>`/`<256,true>` in
`interp_flash_fp8kv`. It is also the family the *emit-side* coverage check goes blind on —
`manifest.rs` reads the head dim from `i[6]` for every flash op, but `FlashMerge` carries it in
`i[3]`, "the exact template family a dispatch bug has already been found in once". The flash
bucket carries no merge at all; its arms are held by the baseline arm set instead, and the audit
says so rather than passing silently.

## Where this does not reach

- **gfx950.** The contract is wired into `build_gfx942.sh` only. `plow_isa.occupancy` knows
  CDNA4's 160 KiB LDS and the geometry markers land in any object, but `build_gfx950.sh` writes
  no `build_defines.json` and there is no blessed gfx950 baseline. The audit refuses a
  cross-arch pairing rather than asserting numbers that describe the other silicon — the same
  refusal the instruction-selection half already makes, and for the same reason.
- **Inlined arms.** A `__device__` body the compiler inlines leaves no symbol, so "absent" and
  "inlined" are indistinguishable from the symbol table. That is why class 5 is a rule on one
  measured-stable family plus a baseline diff on the rest, and not a coverage requirement over
  every family.
- **Head dim never reaches `requires` on AMD.** `manifest.rs` already emits
  `PLOW_HAS_FLASH_HD{64,128,256,512}` into the packet's `plow_config.h`; only
  `runtime/nvidia/interp_sm120.cu` consumes it. `interp.hip` does not, so the object-side check
  derives the required set from the object's own axes (`hd=64` is `#if !PLOW_FP8_KV`) rather
  than from the blob. Making the AMD interpreter consume the packet's head-dim capability the
  way the NVIDIA one does would let the two halves agree; that is the follow-up.
