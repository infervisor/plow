#!/usr/bin/env python3
"""asm_audit.py — the static performance contract for a built code object.

The register-cliff check in build_gfx950.sh / build_gfx942.sh catches a kernel that will not
launch. It does not catch the failure that costs the most: a kernel that compiles, launches,
produces correct numbers, and is slow — because the backend picked the narrow MFMA, because a
`-D` never reached the compiler, because the accumulator went to scratch, or because a
dispatch arm was never instantiated at all. On a box with no GPU every one of those is
invisible. So: disassemble the code object and assert on what is actually in it.

Two contracts, one disassembly pass:

    --expect <expectations.json>   INSTRUCTION SELECTION, per kernel. Which MFMA a body must
                                   use, which it must not, which convert proves an operand
                                   was not silently widened. Documented below.
    --contract <baseline.json>     THE PERFORMANCE CONTRACT, per object. Six classes, each
                                   from a defect that cost real performance on this branch
                                   and was found by hand. Documented under CONTRACT below.

Usage
    asm_audit.py <object.elf|object.co> [...]                    # report
    asm_audit.py --expect expectations.json <object...>          # + instruction selection
    asm_audit.py --contract baseline.json --defines d.json <object...>
    asm_audit.py --contract baseline.json --defines d.json --bless <object...>

`--expect` takes a two-level map, object-substring -> kernel-substring -> checks,
plus a mandatory top-level `_arch`:

    {"_arch": "gfx950",
     "interp_prefill_fp8": {"plow_exec": {"cbsz": 0, "blgp": 0}}}

THE ARCH IS PART OF THE CONTRACT, not a formality. `v_mfma_f32_32x32x16_fp8_fp8`
is the only fp8 MFMA gfx942 has and it is the WRONG one on gfx950 (half rate
against the K=64 f8f6f4 form), so the gfx950 file `forbid`s exactly the
instruction the gfx942 file `require_min`s. One expectation file cannot state
both, and quietly auditing an object against the other arch's contract inverts
every fp8 assertion. Each object's arch is read from its own ELF header and
checked against `_arch` before anything is asserted.

Scoping by object matters: a kernel absent from an object is only a failure if
that object was supposed to contain it, and every object is audited against a
different set of arms. A check is one of:

    mfma            exact instruction name that must dominate the MFMA mix
    mfma_min        minimum MFMA count
    cbsz / blgp     required operand format on every MFMA (0=e4m3 1=e5m2
                    2=fp6 3=bf6 4=fp4); absent in the text means 0
    burst_min       minimum longest back-to-back MFMA run (pipeline depth)
    stalled_max     maximum MFMAs issued straight after an s_waitcnt
    require_min     {mnemonic-substring: minimum count} — for kernels whose
                    point is not MFMA (an mxfp4 GEMV must show the packed fp4
                    convert, or the compiler silently widened the weights)
    no_scratch      true => zero spill traffic
    scratch_max     maximum spill instruction count
    forbid          list of instruction-name substrings that must NOT appear

CONTRACT
--------
`--contract` runs six checks. Four REFUSE, two REPORT — the split follows the emit-side
twin's rule (`crates/devgen/src/dispatch_audit.rs`): a check refuses when the object is
WRONG, and reports when the object is a deliberate trade whose number is worth watching.
`PLOW_AUDIT_STRICT=1` promotes the reports to refusals, exactly as it does there.

  1 macro    FAIL. Every geometry `-D` the build passed must appear in the object as a
             `plow_geom_<MACRO>` marker carrying the value that COMPILED. `op_gemm.h`
             declared five per-rung GEMM knobs as bare `#define`s while build_gfx942.sh
             documented them as reachable through its `GM_AX` raw-`-D` hatch; a bare
             `#define` after a command-line `-D` is a redefinition the header wins, the
             recipe compiles `-w`, and every A/B ever run through that hatch measured an
             UNCHANGED OBJECT — including a published "GM_SM_BK=128 is a flat null" row that
             is worth -15.2% TTFT at 128 tokens now that the guards are there. Also checks
             the roster: a knob `#ifndef`-guarded in the marked headers with no
             PLOW_GEOM_MARK line in geom_contract.h is a failure, so the guard cannot rot.
  2 spill    FAIL. Spill counted from `scratch_load_*`/`scratch_store_*`, never from
             `.vgpr_spill_count` — candidates in this tree have reported 0 there and issued
             119 scratch_load_dword anyway. A kernel whose note says 0 while its ISA
             disagrees is a failure on its own; the per-object budget is the baseline.
  3 permute  REPORT. `__shfl_xor` lowers to `ds_bpermute` on gfx9, an LDS-crossbar op. One
             dense flash-prefill KV tile was 1651 instructions, 160 ds_bpermute and 163
             s_waitcnt against 64 MFMA; DPP + one ds_swizzle took the permutes to 64 and was
             worth -3.5% TTFT. Bodies whose permute count is disproportionate to their MFMA
             count are named, with their s_waitcnt density, which was the co-signal.
  4 lds16    REPORT. 2-byte LDS reads. The MLA prefill PV transpose issues 256 ds_read_u16
             per lane per KV tile against 68 MFMA — the largest item in that loop, and a
             deliberate trade documented in op_attention.h. Reported, never failed.
  5 arms     FAIL. Every head-dim arm the object must serve must be present as a template
             instantiation. `hd=64` had NO arm in prefill, decode or either merge and no
             `else`, so GPT-OSS attention silently wrote nothing — on AMD a missing arm does
             not trap.
  6 budget   FAIL. VGPR/AGPR/LDS/occupancy against a committed per-object budget. This
             replaces the hand-maintained resource table in build_gfx942.sh's header
             comment, which was already stale: it claimed interp_prefill spill 6 against a
             measured 126.

The baseline (`scripts/obj_baseline_gfx942.json`) is COMMITTED, so a regression is a line in
a `git diff` and not a number in a log nobody reads. `--bless` rewrites it. A baseline row is
only compared when the object's `-D` axes match the ones the baseline recorded: a
`PLOW_OCC4=1` or `PLOW_DECODE_BATCH=4` build is a different object and is checked by the
rules that need no baseline, not against numbers describing a different compile.

Exit status is nonzero if any assertion fails, so it drops straight into a build script.
"""

import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plow_isa as isa  # noqa: E402

FMT_NAME = {0: "e4m3", 1: "e5m2", 2: "fp6", 3: "bf6", 4: "fp4"}


def report(path, kernels):
    print(f"\n=== {path} ===")
    for k in kernels:
        print(f"\n  {k.name}")
        print(f"    instructions {k.total}   MFMA {k.fam['mfma']} "
              f"({k.density:.1f}/100)   spill {k.spill}")
        if k.fam["mfma"]:
            print(f"    pipeline: longest MFMA burst {k.burst}   "
                  f"wait-stalled MFMAs {k.stalled}/{k.fam['mfma']}")
        if k.mfma:
            for name, n in k.mfma.most_common():
                print(f"      {n:6d}  {name}")
            for (cbsz, blgp), n in sorted(k.fmt.items()):
                # Only interesting for the f8f6f4 family; a bf16 MFMA has no
                # format field and lands in (0,0).
                if (cbsz, blgp) != (0, 0) or any("f8f6f4" in m for m in k.mfma):
                    print(f"      {n:6d}  A={FMT_NAME.get(cbsz, cbsz)} "
                          f"B={FMT_NAME.get(blgp, blgp)}")
        mix = "  ".join(f"{f}={k.fam[f]}" for f, _ in isa.FAMILIES
                        if k.fam[f] and f != "mfma")
        if mix:
            print(f"    {mix}")


def check(kernels, expect):
    """Assert `expect` against the parsed kernels. Returns a list of failures."""
    fails = []
    for pat, rules in expect.items():
        hits = [k for k in kernels if pat in k.name]
        if not hits:
            fails.append(f"{pat}: no kernel matched")
            continue
        for k in hits:
            for rule, want in rules.items():
                if rule == "mfma":
                    got = k.mfma.most_common(1)
                    if not got or got[0][0] != want:
                        fails.append(
                            f"{k.name}: dominant MFMA is "
                            f"{got[0][0] if got else 'none'}, expected {want}")
                elif rule == "mfma_min":
                    if k.fam["mfma"] < want:
                        fails.append(
                            f"{k.name}: {k.fam['mfma']} MFMA < required {want}")
                elif rule in ("cbsz", "blgp"):
                    idx = 0 if rule == "cbsz" else 1
                    bad = {f[idx] for f in k.fmt if f[idx] != want}
                    if bad:
                        fails.append(
                            f"{k.name}: {rule} {sorted(bad)} present, "
                            f"expected all {want} ({FMT_NAME.get(want, want)})")
                elif rule == "fmt_on":
                    # PER-MNEMONIC operand format. The blanket cbsz/blgp rules above assert
                    # over EVERY MFMA in the kernel, which is unusable in an object that
                    # legitimately mixes families -- the A4W4 grouped MoE GEMM shares plow_exec
                    # with the bf16 MLA attention arms, and a bf16 MFMA carries no cbsz at all
                    # (so it reads as 0 and trips a `cbsz: 4` check). This scopes the assertion
                    # to the instruction that is supposed to carry the format.
                    for mnem, spec in want.items():
                        seen = Counter()
                        for m, c in k.fmt_on.items():
                            if mnem in m:
                                seen.update(c)
                        if not seen:
                            fails.append(f"{k.name}: no '{mnem}' to check format on")
                            continue
                        for key, idx in (("cbsz", 0), ("blgp", 1)):
                            if key not in spec:
                                continue
                            bad = {f[idx] for f in seen if f[idx] != spec[key]}
                            if bad:
                                fails.append(
                                    f"{k.name}: {mnem} {key} {sorted(bad)} present, expected "
                                    f"all {spec[key]} ({FMT_NAME.get(spec[key], spec[key])})")
                elif rule == "burst_min":
                    if k.burst < want:
                        fails.append(
                            f"{k.name}: longest MFMA burst {k.burst} < {want} "
                            f"(operands are not staged far enough ahead)")
                elif rule == "stalled_max":
                    if k.stalled > want:
                        fails.append(
                            f"{k.name}: {k.stalled} wait-stalled MFMAs > {want}")
                elif rule == "no_scratch":
                    if want and k.spill:
                        fails.append(f"{k.name}: {k.spill} spill instructions, "
                                     f"expected none")
                elif rule == "scratch_max":
                    if k.spill > want:
                        fails.append(
                            f"{k.name}: {k.spill} spill instructions > {want}")
                elif rule == "require_min":
                    for mnem, least in want.items():
                        n = sum(v for m, v in k.insn.items() if mnem in m)
                        if n < least:
                            fails.append(
                                f"{k.name}: {n}x '{mnem}' < required {least} "
                                f"(instruction selection changed)")
                elif rule == "forbid":
                    for bad in want:
                        n = sum(v for m, v in k.mfma.items() if bad in m)
                        if n:
                            fails.append(f"{k.name}: {n} forbidden '{bad}'")
                else:
                    fails.append(f"{pat}: unknown check '{rule}'")
    return fails


# ------------------------------------------------------------------ the contract

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = int(os.environ.get("PLOW_AUDIT_JOBS", "8"))

# Class 1. The headers whose `#ifndef`-guarded knobs geom_contract.h must mark, and the
# namespaces that count as geometry. Both are asserted against the source, so a knob added
# to one of these headers without a marker fails the build rather than silently opting out.
GEOM_HEADERS = ["amd_arch.h", "op_moe.h",
                "op_attention.h", "op_attention_common.h", "op_attention_gfx942.h",
                "op_attention_gfx950.h",
                "op_gemm.h", "op_gemm_common.h", "op_gemm_gfx942.h", "op_gemm_gfx950.h"]
GEOM_NS = re.compile(r"^(GM|FA|GV|MPF)_[A-Z0-9_]*$")
# The numeric PLOW_ knobs build_gfx942.sh passes as raw `-D`. `PLOW_*` at large is a
# capability axis (PLOW_FP8, PLOW_K3) whose presence is already checked by the marker
# symbols in interp.hip and by plowrt's arm checks; these three are GEOMETRY wearing a
# PLOW_ name, and a `-D` that failed to reach them is as invisible as one for GM_BM.
GEOM_EXTRA = {"PLOW_WG_WAVES", "PLOW_WPE", "PLOW_GEMV_MM"}
GEOM_PREFIX = "plow_geom_"

# Class 5. Head dims interp.hip dispatches, and the ONE flash family whose instantiations
# survive inlining in every object built here (44/44 measured), so the check is a rule and
# not a baseline diff: `d_flash_prefill<128>` is inlined into plow_exec in every 8-wave
# prefill object, and `d_flash_prefill<128,true>`/`<256,true>` in interp_flash_fp8kv.
#
# d_flash_merge is ALSO the family that the emit-side coverage check goes blind on:
# manifest.rs reads the head dim from i[6] for every flash op but FlashMerge carries it in
# i[3], "the exact template family a dispatch bug has already been found in once".
HD_REQUIRED = [128, 256, 512]
# hd=64 is GPT-OSS. The arms are `#if !PLOW_FP8_KV` (no fp8-KV model has a 64-wide head, and
# the extra instantiation outlines K3's hot MLA prefill body out of the interpreter), so an
# fp8kv object legitimately has none.
HD_GPTOSS = 64
MERGE_HD_PREFIX = "plow_flash_merge_hd_"
ARM_SYM = re.compile(r"^_Z\d+(d_flash_[a-z_0-9]*)I(.+?)Ev")
ARM_HD = re.compile(r"^Li(\d+)E")

# Class 3. A body is named when it issues at least PERMUTE_PER_MFMA LDS permutes per MFMA and
# at least PERMUTE_FLOOR of them.
#
# 2.0 and 64 are both read off the case that paid: the pre-fix d_flash_prefill<256> KV tile
# issued 160 ds_bpermute against 64 MFMA (2.5) and the DPP + single-ds_swizzle form issues 64
# ds_swizzle and ZERO permutes, so the threshold sits between the two forms of the same body
# and is not a taste. A softmax that spends one wave reduce per MFMA (1.0) is doing ordinary
# work; at 2.0 the crossbar is the loop. The floor of 64 is one permute per lane-quarter per
# wave and keeps the small norm/argmax helpers, whose reduce IS the kernel, out of the report.
PERMUTE_PER_MFMA = 2.0
PERMUTE_FLOOR = 64

# Class 4. From the MLA prefill PV transpose, which is the body this check exists to keep
# visible: the MFMA B-fragment for PV contracts over kv, the minor axis of the [kv][d] slab,
# so the transpose is paid as "8 strided ds_read_u16 per output tile, 256 per KV tile"
# (op_attention.h, PLOW_MLA_PF_SV). The compiled d_flash_mla_prefill_v2<512,64,false,true>
# body shows exactly that: 256 2-byte reads against 136 MFMA, 1.88x.
#
# So the floor is ONE KV TILE'S WORTH, 256, and the ratio sits below the 1.88 that body shows
# and above the ~1.0 of a loop that stages its operands b128-wide. The dense flash arms, at
# 128 reads against 64 MFMA, are deliberately under the floor -- their transpose is half the
# width and is not the item this reports. Reported, never failed: op_attention.h documents the
# trade and the alternative (a transposed V store) costs LDS the 64 KiB budget does not have.
LDS16_PER_MFMA = 1.5
LDS16_FLOOR = 256


def defines_of(axes):
    """{MACRO: value} for the `-D` flags in one row's axis string. `-DX` alone is 1."""
    got = {}
    for tok in axes.split():
        if not tok.startswith("-D"):
            continue
        body = tok[2:]
        name, _, val = body.partition("=")
        got[name] = val if val else "1"
    return got


def geom_roster():
    """The knobs the headers declare, and the ones geom_contract.h marks."""
    declared = set()
    for h in GEOM_HEADERS:
        with open(os.path.join(REPO, "runtime/amd", h)) as f:
            for line in f:
                m = re.match(r"^#ifndef ([A-Z0-9_]+)\s*$", line)
                if m and GEOM_NS.match(m.group(1)):
                    declared.add(m.group(1))
    marked = set()
    with open(os.path.join(REPO, "runtime/amd/geom_contract.h")) as f:
        for line in f:
            m = re.match(r"^PLOW_GEOM_MARK\(([A-Z0-9_]+)\)", line)
            if m:
                marked.add(m.group(1))
    return declared, marked


def arms_of(syms):
    """{template family: sorted head dims} over the d_flash_* instantiations in an object."""
    got = {}
    for _, name in syms:
        m = ARM_SYM.match(name)
        if not m:
            continue
        hd = ARM_HD.match(m.group(2))
        if hd:
            got.setdefault(m.group(1), set()).add(int(hd.group(1)))
    return {k: sorted(v) for k, v in sorted(got.items())}


class ObjectFacts:
    """Everything the contract reads out of one object, gathered in one pass."""

    def __init__(self, path, kernels):
        self.path = path
        self.arch = isa.elf_arch(path)
        self.stem = os.path.basename(path).rsplit(".", 1)[0]
        self.syms = kernels
        self.notes = isa.notes(path)
        self.table = isa.symbols(path)
        self.globals = isa.globals_u32(path)
        self.geom = {k[len(GEOM_PREFIX):]: v for k, v in self.globals.items()
                     if k.startswith(GEOM_PREFIX)}
        self.arms = arms_of(self.table)
        # `exec_flash_merge` is __forceinline__ and `d_flash_merge<D>` inlines into it, so a
        # decode object emits no symbol for the family and `arms_of` cannot see it — which
        # silently turned the head-dim check into its "carries no instantiation" branch and
        # lost exactly the coverage it exists to assert. `plow_flash_merge_hd_<D>`
        # (interp.hip) states what the chain compiled and survives any inlining decision.
        # Applied HERE rather than in the check so `--bless` persists the same facts the
        # check reads; a check-only fallback blesses a baseline that then disagrees with it.
        if "d_flash_merge" not in self.arms:
            marked = sorted(
                int(k[len(MERGE_HD_PREFIX):])
                for k, v in self.globals.items()
                if k.startswith(MERGE_HD_PREFIX) and v == 1
            )
            if marked:
                self.arms["d_flash_merge"] = marked
        # The kernel the object exists for. Its note carries the resource numbers the cliff
        # check prints; the OUTLINED bodies (plow_exec, d_gemm_glu, d_flash_prefill<D>) carry
        # NO note at all, which is why their spill is invisible to `.vgpr_spill_count`.
        main = max(self.notes.values(), key=lambda r: r.get("vgpr_count", 0), default={})
        self.vgpr = main.get("vgpr_count", 0)
        self.agpr = main.get("agpr_count", 0)
        self.lds = main.get("group_segment_fixed_size", 0)
        self.meta_spill = main.get("vgpr_spill_count", 0)
        self.waves = self.geom.get("PLOW_WG_WAVES", 0) or (
            main.get("max_flat_workgroup_size", 256) // 64)
        self.occ = isa.occupancy(self.vgpr, self.lds, self.waves, self.arch)
        self.isa_spill = sum(s.spill for s in kernels)
        # Scratch traffic in symbols that have no kernel note. `.vgpr_spill_count` cannot see
        # any of it, by construction.
        self.unmetered = sum(s.spill for s in kernels if s.name not in self.notes)
        # Kernels whose note claims no spill while their ISA shows scratch traffic. The
        # baseline carries the known ones so a NEW one refuses; see class 2.
        self.meta_lies = {s.name: s.spill for s in kernels
                          if s.spill and s.name in self.notes
                          and not self.notes[s.name].get("vgpr_spill_count")}
        self.bpermute = sum(s.fam["bpermute"] for s in kernels)
        self.narrow_ds = sum(s.narrow_ds for s in kernels)

    def row(self, defines):
        """The committed baseline row for this object."""
        return {
            "defines": defines,
            "vgpr": self.vgpr, "agpr": self.agpr, "lds": self.lds, "occ": self.occ,
            "meta_spill": self.meta_spill,
            "isa_spill": self.isa_spill, "unmetered_spill": self.unmetered,
            "meta_lies": self.meta_lies,
            "bpermute": self.bpermute, "lds16": self.narrow_ds,
            "arms": self.arms,
        }


def contract(facts, baseline, defines, strict):
    """Run the six checks over one object. -> (failures, advisories)."""
    fails, notes = [], []
    o = facts.stem
    want = baseline.get("objects", {}).get(o)
    profiles = baseline.get("geom_profiles", {})

    # --- 1. every geometry -D must appear in the object, carrying the value that compiled.
    for name, val in sorted(defines.items()):
        if not (GEOM_NS.match(name) or name in GEOM_EXTRA):
            continue
        if name not in facts.geom:
            fails.append(
                f"{o}: -D{name}={val} was passed but the object carries no "
                f"{GEOM_PREFIX}{name} marker — add PLOW_GEOM_MARK({name}) to "
                f"runtime/amd/geom_contract.h, or the -D cannot be shown to have taken")
            continue
        try:
            iv = int(val, 0)
        except ValueError:
            continue          # a non-numeric -D is not a geometry value
        if facts.geom[name] != iv:
            fails.append(
                f"{o}: -D{name}={iv} did NOT take — the object compiled {name}="
                f"{facts.geom[name]} (a bare #define in the header wins over the command "
                f"line, and the recipe compiles -w)")

    # --- 6. resource budget, and the geometry the object actually compiled.
    if want:
        if want.get("defines") != defines:
            notes.append(f"{o}: built with different axes than the baseline records; "
                         f"resource and arm budgets not compared")
            want = None
    if want:
        for field, got in (("vgpr", facts.vgpr), ("agpr", facts.agpr),
                           ("lds", facts.lds), ("occ", facts.occ)):
            if want.get(field) != got:
                fails.append(f"{o}: {field} {got} != budget {want.get(field)} "
                             f"(re-bless with --bless if this is intended)")
        prof = profiles.get(want.get("geom"), {})
        for knob, val in sorted(prof.items()):
            if facts.geom.get(knob) != val:
                fails.append(f"{o}: compiled {knob}={facts.geom.get(knob)}, "
                             f"baseline {val}")
        for knob in sorted(set(facts.geom) - set(prof)):
            fails.append(f"{o}: compiled {knob}={facts.geom[knob]}, absent from the baseline "
                         f"geometry profile")

    # --- 2. spill, from the ISA, against the note and against the budget.
    known = (want or {}).get("meta_lies", {})
    for name, n in sorted(facts.meta_lies.items()):
        msg = (f"{name} issues {n} scratch op(s) while its note reports vgpr_spill_count 0 "
               f"— the metadata is wrong, trust the ISA")
        if not want:
            notes.append(f"{o}: {msg}")
        elif n > known.get(name, -1):
            fails.append(f"{o}: {msg} (baseline records {known.get(name, 'none')})")
    for name in sorted(set(known) - set(facts.meta_lies)):
        notes.append(f"{o}: {name} no longer under-reports its spill — re-bless")
    if want:
        if facts.isa_spill > want.get("isa_spill", 0):
            fails.append(f"{o}: {facts.isa_spill} scratch ops > budget "
                         f"{want.get('isa_spill')}")
        elif facts.isa_spill < want.get("isa_spill", 0):
            notes.append(f"{o}: {facts.isa_spill} scratch ops, below the budget of "
                         f"{want.get('isa_spill')} — re-bless to hold the improvement")

    # --- 5. every head-dim arm the object must serve is instantiated.
    need = list(HD_REQUIRED)
    if defines.get("PLOW_FP8_KV") != "1":
        need.append(HD_GPTOSS)
    if "d_flash_merge" in facts.arms:
        missing = [h for h in need if h not in facts.arms["d_flash_merge"]]
        if missing:
            fails.append(
                f"{o}: d_flash_merge has no arm for head_dim {missing} — on AMD a missing "
                f"arm does not trap, the dispatch falls through and WRITES NOTHING")
        extra = [h for h in facts.arms["d_flash_merge"] if h not in need]
        if extra:
            fails.append(f"{o}: d_flash_merge instantiates head_dim {extra}, which this "
                         f"object's axes say it must not serve")
    else:
        notes.append(f"{o}: carries no d_flash_merge instantiation (the flash bucket runs "
                     f"class-4 segments and merges elsewhere); its head-dim arms are held by "
                     f"the baseline arm set, not by the coverage rule")
    if want and facts.arms != want.get("arms"):
        fails.append(f"{o}: flash arm set {facts.arms} != baseline {want.get('arms')}")

    # --- 3. LDS-crossbar-bound reductions (report).
    for s in facts.syms:
        m, b = s.fam["mfma"], s.fam["bpermute"]
        if m and b >= PERMUTE_FLOOR and b >= PERMUTE_PER_MFMA * m:
            notes.append(
                f"{o}: {s.name[:52]} issues {b} LDS permute(s) against {m} MFMA "
                f"({b / m:.1f}x), s_waitcnt {s.fam['waitcnt']} ({s.wait_density:.1f}/100), "
                f"{s.fam['swizzle']} ds_swizzle")

    # --- 4. 2-byte LDS read density (report).
    for s in facts.syms:
        m, n = s.fam["mfma"], s.narrow_ds
        if m and n >= LDS16_FLOOR and n >= LDS16_PER_MFMA * m:
            notes.append(
                f"{o}: {s.name[:52]} issues {n} 2-byte ds_read against {m} MFMA "
                f"({n / m:.1f}x)")

    if strict:
        fails += [f"(strict) {n}" for n in notes]
        notes = []
    return fails, notes


def run_contract(objects, baseline_path, defines_path, bless, strict):
    with open(baseline_path) as f:
        baseline = json.load(f)
    defines_map = {}
    if defines_path:
        with open(defines_path) as f:
            defines_map = {k: defines_of(v) for k, v in json.load(f).items()}

    fails, notes = [], []

    # The roster check is source-side and runs once: it is what stops the marker set from
    # rotting as knobs are added.
    declared, marked = geom_roster()
    for knob in sorted(declared - marked):
        fails.append(f"geom_contract.h: {knob} is a tunable in runtime/amd but has no "
                     f"PLOW_GEOM_MARK line, so a -D{knob} could not be shown to have taken")
    for knob in sorted(marked - declared - GEOM_EXTRA):
        # The reverse direction, and it is a REFUSAL rather than a note: a marked knob that no
        # header `#ifndef`-guards any more is a knob whose bare `#define` now silently beats the
        # command line. That is the GM_SM_BK defect itself, catchable without anyone passing a
        # -D at all.
        fails.append(f"geom_contract.h: marks {knob}, but no header #ifndef-guards it any "
                     f"more — a -D{knob} would be silently overridden by the header")

    # The arch guard the `--expect` half already has: gfx942 and gfx950 are different LDS
    # budgets, different occupancy arithmetic and a different geometry profile, so a baseline
    # blessed on one describes nothing about the other.
    want_arch = baseline.get("_arch")
    for path, _ in objects:
        got = isa.elf_arch(path)
        if want_arch and got != want_arch:
            print(f"FAIL  {os.path.basename(path)}: built for {got}, but "
                  f"{os.path.basename(baseline_path)} is the {want_arch} contract")
            return 1

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        all_facts = list(pool.map(lambda pk: ObjectFacts(*pk), objects))

    rows, profiles = {}, {}
    print("\n   object                              vgpr agpr    lds occ  meta   ISA  unmet "
          "bperm  lds16")
    for facts in all_facts:
        defines = defines_map.get(facts.stem, {})
        print(f"   {facts.stem:<34}{facts.vgpr:>6}{facts.agpr:>5}{facts.lds:>7}"
              f"{facts.occ:>4}{facts.meta_spill:>6}{facts.isa_spill:>6}"
              f"{facts.unmetered:>7}{facts.bpermute:>6}{facts.narrow_ds:>7}")
        if bless:
            rows[facts.stem] = facts.row(defines)
            profiles[facts.stem] = facts.geom
            continue
        if not defines_map:
            notes.append(f"{facts.stem}: no --defines entry; the geometry -D contract "
                         f"(class 1) cannot run on this object")
        f, n = contract(facts, baseline, defines, strict)
        fails += f
        notes += n

    if bless:
        # Geometry is identical across most rows, so the profiles are shared and named in
        # first-appearance order over the sorted object list: one knob moving is then one
        # changed line in the diff, not forty.
        seen, named = {}, {}
        for stem in sorted(rows):
            key = json.dumps(profiles[stem], sort_keys=True)
            if key not in seen:
                seen[key] = f"p{len(seen)}"
                named[seen[key]] = profiles[stem]
            rows[stem]["geom"] = seen[key]
        with open(baseline_path) as f:
            old = json.load(f)
        old["geom_profiles"] = named
        old["objects"] = {k: rows[k] for k in sorted(rows)}
        with open(baseline_path, "w") as f:
            json.dump(old, f, indent=1, sort_keys=False)
            f.write("\n")
        print(f"\nBLESSED  {len(rows)} object(s), {len(named)} geometry profile(s) "
              f"-> {os.path.relpath(baseline_path, REPO)}")
        return 0

    # Identical advisories across objects collapse to one line: the same three flash-prefill
    # bodies are outlined into thirty objects, and thirty copies of one finding is a wall of
    # text nobody reads rather than a report.
    grouped = {}
    for n in notes:
        obj, _, msg = n.partition(": ")
        grouped.setdefault(msg, []).append(obj)
    for msg, objs in grouped.items():
        where = objs[0] if len(objs) == 1 else f"{objs[0]} +{len(objs) - 1} more"
        print(f"NOTE  [{where}] {msg}")
    if fails:
        for f in fails:
            print(f"FAIL  {f}")
        return 1
    print(f"PASS  contract held over {len(objects)} object(s)"
          f"{' (strict)' if strict else ''}")
    return 0


def main(argv):
    expect, contract_path, defines_path, paths = None, None, None, []
    bless = False
    strict = os.environ.get("PLOW_AUDIT_STRICT", "0") not in ("0", "", "false")
    quiet = False
    it = iter(argv)
    for a in it:
        if a == "--expect":
            expect = json.load(open(next(it)))
        elif a == "--contract":
            contract_path = next(it)
        elif a == "--defines":
            defines_path = next(it)
        elif a == "--bless":
            bless = True
        elif a == "--strict":
            strict = True
        elif a == "--quiet":
            quiet = True        # the contract table only; skip the per-kernel dump
        else:
            paths.append(a)
    if not paths:
        sys.exit(__doc__)

    want_arch = expect.pop("_arch", None) if expect else None
    if expect is not None and not want_arch:
        sys.exit("asm_audit: the expectation file must declare a top-level \"_arch\"")

    # One disassembly for both contracts, and one thread per object: the whole cost here is
    # llvm-objdump and llvm-readelf subprocesses, which release the GIL. 45 objects serially is
    # 31 s, which is where a build-time check starts getting skipped.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        arches = list(pool.map(isa.elf_arch, paths))
        parsed = list(zip(paths, pool.map(isa.counts, paths, arches)))

    fails, checked = [], 0
    for (p, kernels), arch in zip(parsed, arches):
        if not quiet:
            report(p, kernels)
        if not expect:
            continue
        if arch != want_arch:
            fails.append(
                f"{os.path.basename(p)}: built for {arch}, but these expectations are "
                f"for {want_arch} — the fp8 and bf16 MFMA contracts are INVERTED between "
                f"the two CDNA levels, so this audit would assert the opposite of the truth"
            )
            continue
        base = os.path.basename(p)
        # Longest matching object key wins, so "interp_decode_fp8" does not also
        # pick up the rules written for "interp_decode".
        keys = sorted((k for k in expect if k in base), key=len, reverse=True)
        if keys:
            checked += 1
            fails += [f"{base}: {f}" for f in check(kernels, expect[keys[0]])]

    rc = 0
    if expect:
        print()
        if fails:
            for f in fails:
                print(f"FAIL  {f}")
            rc = 1
        else:
            print(f"PASS  all {want_arch} assertions held over {checked} audited object(s)")
    if contract_path:
        rc |= run_contract(parsed, contract_path, defines_path, bless, strict)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
