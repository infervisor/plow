#!/usr/bin/env python3
"""asm_audit.py — assert what the compiler ACTUALLY emitted for gfx950.

The register-cliff check in build_gfx950.sh catches a kernel that will not
launch. It does not catch the failure that costs the most: a kernel that
compiles, launches, produces correct numbers, and is 4x slow because the
backend picked the narrow MFMA, widened an fp4 operand to bf16, or spilled the
accumulator to scratch. On a box with no GPU that failure is invisible.

So: disassemble the code object and assert on the instruction stream.

Usage
    asm_audit.py <object.elf|object.co> [...]            # report
    asm_audit.py --expect expectations.json <object...>  # report + assert

`--expect` takes a two-level map, object-substring -> kernel-substring -> checks:

    {"interp_prefill_fp8": {"plow_exec": {"cbsz": 0, "blgp": 0}}}

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

Exit status is nonzero if any assertion fails, so it drops straight into a
build script or CI.
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter

# Spill traffic on CDNA is scratch_load/scratch_store. Deliberately NOT matching
# buffer_* here: the GEMV weight streams are raw buffer loads by design
# (buf_ld_fp8 -> buffer_load_dwordx4 ... offen), and counting those as spills
# reported 36 spills for a kernel the compiler says spills nothing.
SPILL = re.compile(r"^\s*scratch_(load|store)")
SYMBOL = re.compile(r"^([0-9a-f]{16})\s+<(.+)>:\s*$")
# llvm-objdump prints "\tv_mfma_f32_32x32x64_f8f6f4 v[0:15], ... cbsz:4 blgp:4 // encoding"
INSN = re.compile(r"^\s*([a-z][a-z0-9_]*)\s")
MODIFIER = re.compile(r"\b(cbsz|blgp|abid|op_sel|op_sel_hi|neg|neg_hi):")

# Instruction families we care about, in report order. Anything unmatched is
# lumped into "other" — the point is the shape of the kernel, not a full ISA
# taxonomy.
FAMILIES = [
    ("mfma", lambda m: m.startswith("v_mfma") or m.startswith("v_smfmac")),
    ("cvt_scale", lambda m: m.startswith("v_cvt_scalef32")),
    ("cvt", lambda m: m.startswith("v_cvt") ),
    ("ds_read", lambda m: m.startswith("ds_read")),
    ("ds_write", lambda m: m.startswith("ds_write")),
    ("global", lambda m: m.startswith("global_") or m.startswith("flat_")),
    ("buffer", lambda m: m.startswith("buffer_")),
    ("scratch", lambda m: m.startswith("scratch_")),
    ("barrier", lambda m: m == "s_barrier" or m.startswith("s_barrier")),
    ("waitcnt", lambda m: m.startswith("s_waitcnt")),
    ("valu", lambda m: m.startswith("v_")),
    ("salu", lambda m: m.startswith("s_")),
]


def tool(name):
    """Locate a ROCm LLVM tool without pinning a ROCm version."""
    root = os.environ.get("ROCM_PATH", "/opt/rocm")
    for cand in (f"{root}/llvm/bin/{name}", f"{root}/lib/llvm/bin/{name}", name):
        if os.path.isfile(cand) or subprocess.run(
            ["which", cand], capture_output=True
        ).returncode == 0:
            return cand
    sys.exit(f"asm_audit: cannot find {name} (set ROCM_PATH)")


def disassemble(path):
    out = subprocess.run(
        [tool("llvm-objdump"), "-d", "--mcpu=gfx950", path],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(f"asm_audit: objdump failed on {path}:\n{out.stderr}")
    return out.stdout


class Kernel:
    def __init__(self, name):
        self.name = name
        self.fam = Counter()
        self.insn = Counter()     # every mnemonic, for `require_min`
        self.mfma = Counter()
        self.fmt = Counter()      # (cbsz, blgp) pairs seen on MFMAs
        self.fmt_on = {}          # mnemonic -> Counter of its own (cbsz, blgp) pairs
        self.spill = 0
        self.total = 0
        # --- pipeline quality (see burst/stalled below) ---
        self.burst = 0            # longest back-to-back MFMA run
        self._run = 0
        self.stalled = 0          # MFMAs immediately preceded by a wait
        self._pending_wait = False

    def feed(self, mnemonic):
        """Track MFMA clustering as instructions stream past.

        These two numbers are the closest thing to a pipeline-quality reading
        that a disassembly can give, and they are what separates a CK
        Intrawave-style deep-prefetch loop from a naive one:

        `burst`   — longest run of consecutive MFMAs. A loop that stages its
                    operands far enough ahead issues MFMAs back to back; one
                    that reads LDS just in time cannot, because every MFMA is
                    separated by the ds_read feeding the next. op_gemm.h
                    identifies that LDS-read -> MFMA dependency chain as the
                    measured wall, so `burst` is the metric that moves when the
                    chain is actually broken.
        `stalled` — MFMAs issued immediately after an s_waitcnt, i.e. the
                    operand was not ready and the wave blocked. Ideally 0: the
                    prefetch should have landed several iterations earlier.

        Neither is a cycle count and neither replaces a GPU. They are ordinal:
        strictly better on both, at equal MFMA count, is a strictly better
        pipeline, and that is enough to iterate on a box with no hardware.
        """
        is_mfma = mnemonic.startswith("v_mfma") or mnemonic.startswith("v_smfmac")
        if is_mfma:
            self._run += 1
            self.burst = max(self.burst, self._run)
            if self._pending_wait:
                self.stalled += 1
        elif mnemonic.startswith("s_nop") or mnemonic.startswith("s_setprio"):
            pass          # scheduling padding, does not break a burst
        else:
            self._run = 0
        # A wait "arms" only the next instruction.
        self._pending_wait = mnemonic.startswith("s_waitcnt") if not is_mfma else False

    @property
    def density(self):
        """MFMA per 100 instructions — the crude 'is this MFMA-bound' signal."""
        return 100.0 * self.fam["mfma"] / self.total if self.total else 0.0


def parse(text):
    kernels, cur = [], None
    for line in text.splitlines():
        m = SYMBOL.match(line)
        if m:
            cur = Kernel(m.group(2))
            kernels.append(cur)
            continue
        if cur is None:
            continue
        # Strip the trailing "// encoding" comment so it cannot be mistaken for
        # an operand modifier.
        code = line.split("//")[0]
        i = INSN.match(code)
        if not i:
            continue
        mn = i.group(1)
        cur.total += 1
        cur.insn[mn] += 1
        cur.feed(mn)
        if SPILL.match(code):
            cur.spill += 1
        for fam, pred in FAMILIES:
            if pred(mn):
                cur.fam[fam] += 1
                break
        else:
            cur.fam["other"] += 1
        if mn.startswith("v_mfma") or mn.startswith("v_smfmac"):
            cur.mfma[mn] += 1
            # Unspecified modifiers default to 0 in the AMDGPU asm printer.
            cbsz = re.search(r"\bcbsz:(\d+)", code)
            blgp = re.search(r"\bblgp:(\d+)", code)
            pair = (int(cbsz.group(1)) if cbsz else 0,
                    int(blgp.group(1)) if blgp else 0)
            cur.fmt[pair] += 1
            cur.fmt_on.setdefault(mn, Counter())[pair] += 1
    return [k for k in kernels if k.total]


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
        mix = "  ".join(f"{f}={k.fam[f]}" for f, _ in FAMILIES
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


def main(argv):
    expect, paths = None, []
    it = iter(argv)
    for a in it:
        if a == "--expect":
            expect = json.load(open(next(it)))
        else:
            paths.append(a)
    if not paths:
        sys.exit(__doc__)

    fails, checked = [], 0
    for p in paths:
        kernels = parse(disassemble(p))
        report(p, kernels)
        if not expect:
            continue
        base = os.path.basename(p)
        # Longest matching object key wins, so "interp_decode_fp8" does not also
        # pick up the rules written for "interp_decode".
        keys = sorted((k for k in expect if k in base), key=len, reverse=True)
        if keys:
            checked += 1
            fails += [f"{base}: {f}" for f in check(kernels, expect[keys[0]])]

    if expect:
        print()
        if fails:
            for f in fails:
                print(f"FAIL  {f}")
            return 1
        print(f"PASS  all assertions held over {checked} audited object(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
