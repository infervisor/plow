#!/usr/bin/env python3
"""plow_isa.py — read a compiled AMD code object: disassembly, ELF notes, symbols.

ONE counter, two consumers. `scripts/asm_audit.py` (the build-time contract) and
`scripts/kx_isa.py` (the `kx --isa` single-block harness) both need "how many of each
instruction does this symbol issue, and does it really spill", and both had their own
regex table. Two tables meant two definitions of `spill`: this module is the single one,
so a threshold measured in `kx` is the same number the build asserts.

Nothing here decides anything. It reports what the object says; the policy — what is a
violation, what is an advisory, what the budget is — lives in asm_audit.py.

Four readers, all from the object itself and never from the build's intent:

    counts(path)      llvm-objdump -d, counted per kernel symbol
    notes(path)       the AMDHSA msgpack note: vgpr/agpr/sgpr/lds/spill, per kernel
    symbols(path)     .symtab/.dynsym names — the marker and template-arm symbols
    globals_u32(path) the VALUE of each marker word, out of .data (or .bss, hence 0)

SPILL IS COUNTED FROM THE ISA, not from `.vgpr_spill_count`. Candidates in this tree have
reported `.vgpr_spill_count: 0` in the note and issued 119 `scratch_load_dword` anyway, so
the note is reported alongside and the disagreement is itself a finding (asm_audit's class 2).
Deliberately NOT matching `buffer_*`: the GEMV weight streams are raw buffer loads by design
(`buf_ld_fp8` -> `buffer_load_dwordx4 ... offen`), and counting those as spill reported 36
spills for a kernel that spills nothing.
"""

import os
import re
import shutil
import subprocess
import sys
from collections import Counter

SYMBOL = re.compile(r"^([0-9a-f]{16})\s+<(.+)>:\s*$")
# llvm-objdump prints "\tv_mfma_f32_32x32x64_f8f6f4 v[0:15], ... cbsz:4 blgp:4 // encoding"
INSN = re.compile(r"^\s*([a-z][a-z0-9_]*)\s")
SPILL = re.compile(r"^\s*scratch_(load|store)")
ELF_ARCH = re.compile(r"\b(gfx\d+[a-z]*)\b")

MFMA = ("v_mfma", "v_smfmac")

# Instruction families, in report order; the first predicate that matches wins, so the
# specific LDS and scratch forms are classified before the generic ds_/v_/s_ buckets.
# Anything unmatched is "other" — this is the shape of a kernel, not an ISA taxonomy.
FAMILIES = [
    ("mfma", lambda m: m.startswith(MFMA)),
    ("cvt_scale", lambda m: m.startswith("v_cvt_scalef32")),
    ("cvt", lambda m: m.startswith("v_cvt")),
    ("bpermute", lambda m: m.startswith("ds_bpermute") or m.startswith("ds_permute")),
    ("swizzle", lambda m: m.startswith("ds_swizzle")),
    ("ds_read", lambda m: m.startswith("ds_read")),
    ("ds_write", lambda m: m.startswith("ds_write")),
    ("global", lambda m: m.startswith("global_") or m.startswith("flat_")),
    ("buffer", lambda m: m.startswith("buffer_")),
    ("scratch", lambda m: m.startswith("scratch_")),
    ("barrier", lambda m: m.startswith("s_barrier")),
    ("waitcnt", lambda m: m.startswith("s_waitcnt")),
    ("valu", lambda m: m.startswith("v_")),
    ("salu", lambda m: m.startswith("s_")),
]

# 2-byte LDS reads. Their own bucket because they are asm_audit's class 4: the MLA prefill
# PV transpose issues 256 of them per lane per KV tile against 68 MFMA, which is the largest
# item in that loop and a deliberate trade (op_attention.h), so it is reported and not failed.
NARROW_DS = re.compile(r"^ds_read_(u16|i16|b16)")


def tool(name):
    """Locate an llvm tool, preferring the toolchain that BUILT the object.

    `PLOW_READELF` is set by the flake's dev shell and points into /nix/store; the
    disassembler next to it is the one matching the compiler that produced the object.
    A system llvm-objdump can decode gfx942 differently, and asm_audit.py previously took
    /opt/rocm unconditionally.
    """
    ref = os.environ.get("PLOW_READELF", "")
    if ref:
        cand = os.path.join(os.path.dirname(ref), name)
        if os.path.exists(cand):
            return cand
    root = os.environ.get("ROCM_PATH", "/opt/rocm")
    for cand in (f"{root}/llvm/bin/{name}", f"{root}/lib/llvm/bin/{name}"):
        if os.path.isfile(cand):
            return cand
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f"plow_isa: cannot find {name} (set ROCM_PATH, or run inside nix develop)")


def _run(argv, what):
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"plow_isa: {what} failed:\n{out.stderr}")
    return out.stdout


def elf_arch(path):
    """The arch the object was BUILT for, out of its own ELF header.

    Read rather than passed in: the objects carry it (`Flags: 0x54c, gfx942, xnack,
    sramecc`), and a `--mcpu` the caller supplies is a second opinion about a fact the file
    already states. That mattered — asm_audit.py once had `--mcpu=gfx950` hardcoded, which
    the disassembler overrode from the ELF, so a wrong-arch audit produced a correct
    disassembly checked against an inverted contract.
    """
    for line in _run([tool("llvm-readelf"), "-h", path], f"readelf -h {path}").splitlines():
        if line.strip().startswith("Flags:"):
            m = ELF_ARCH.search(line)
            if m:
                return m.group(1)
    sys.exit(f"plow_isa: {path} declares no gfx target in its ELF flags")


def unbundle(co, arch, tmpdir):
    """`hipcc --genco` emits a clang offload BUNDLE, not an ELF. Peel it; pass an ELF through."""
    elf = os.path.join(tmpdir, "k.elf")
    bun = tool("clang-offload-bundler")
    for tgt in (f"hipv4-amdgcn-amd-amdhsa--{arch}", f"hip-amdgcn-amd-amdhsa--{arch}"):
        r = subprocess.run(
            [bun, "--type=o", "--unbundle", f"--input={co}", f"--targets={tgt}",
             f"--output={elf}"], capture_output=True, text=True)
        if r.returncode == 0 and os.path.getsize(elf) > 0:
            return elf
    return co


class Sym:
    """One kernel or outlined device function, as the disassembly shows it."""

    def __init__(self, name):
        self.name = name
        self.insn = Counter()     # every mnemonic
        self.fam = Counter()      # FAMILIES buckets
        self.mfma = Counter()     # MFMA mnemonic -> count
        self.fmt = Counter()      # (cbsz, blgp) pairs over all MFMAs
        self.fmt_on = {}          # MFMA mnemonic -> Counter of its own (cbsz, blgp)
        self.spill = 0            # scratch_load/store instructions
        self.narrow_ds = 0        # 2-byte ds_read
        self.total = 0
        # --- pipeline quality ---
        self.burst = 0            # longest back-to-back MFMA run
        self.stalled = 0          # MFMAs issued immediately after an s_waitcnt
        self._run = 0
        self._pending_wait = False

    def _feed(self, mnemonic):
        """Track MFMA clustering as instructions stream past.

        `burst`   — longest run of consecutive MFMAs. A loop that stages operands far enough
                    ahead issues MFMAs back to back; one that reads LDS just in time cannot,
                    because every MFMA is separated by the ds_read feeding the next. op_gemm.h
                    identifies that LDS-read -> MFMA chain as the measured wall.
        `stalled` — MFMAs issued immediately after an s_waitcnt, i.e. the operand was not ready
                    and the wave blocked. Ideally 0.

        Neither is a cycle count. They are ordinal: strictly better on both, at equal MFMA
        count, is a strictly better pipeline, which is enough to iterate on a box with no GPU.
        """
        is_mfma = mnemonic.startswith(MFMA)
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

    @property
    def wait_density(self):
        """s_waitcnt per 100 instructions. The co-signal on the LDS-crossbar case: the
        pre-fix d_flash_prefill<256> KV tile read 163/1651 = 9.9%, the DPP form 91/1584 = 5.7%."""
        return 100.0 * self.fam["waitcnt"] / self.total if self.total else 0.0


def parse(text):
    """Disassembly text -> [Sym], in the order the symbols appear."""
    syms, cur = [], None
    for line in text.splitlines():
        m = SYMBOL.match(line)
        if m:
            cur = Sym(m.group(2))
            syms.append(cur)
            continue
        if cur is None:
            continue
        # Strip the trailing "// encoding" comment so it cannot be mistaken for an operand.
        code = line.split("//")[0]
        i = INSN.match(code)
        if not i:
            continue
        mn = i.group(1)
        cur.total += 1
        cur.insn[mn] += 1
        cur._feed(mn)
        if SPILL.match(code):
            cur.spill += 1
        if NARROW_DS.match(mn):
            cur.narrow_ds += 1
        for fam, pred in FAMILIES:
            if pred(mn):
                cur.fam[fam] += 1
                break
        else:
            cur.fam["other"] += 1
        if mn.startswith(MFMA):
            cur.mfma[mn] += 1
            # Unspecified modifiers default to 0 in the AMDGPU asm printer.
            cbsz = re.search(r"\bcbsz:(\d+)", code)
            blgp = re.search(r"\bblgp:(\d+)", code)
            pair = (int(cbsz.group(1)) if cbsz else 0,
                    int(blgp.group(1)) if blgp else 0)
            cur.fmt[pair] += 1
            cur.fmt_on.setdefault(mn, Counter())[pair] += 1
    return [s for s in syms if s.total]


def counts(path, arch=None):
    """[Sym] for every symbol in `path`, disassembled for the arch the object declares."""
    arch = arch or elf_arch(path)
    return parse(_run([tool("llvm-objdump"), "-d", f"--mcpu={arch}", path],
                      f"objdump {path}"))


# The AMDHSA note is msgpack printed by llvm-readelf as a YAML-ish block: a list under
# `amdhsa.kernels`, one record per kernel, whose fields are `.name`, `.vgpr_count` and so on.
# A record starts at the first field of a new list item; `.name` may appear anywhere in it,
# so the record is buffered and keyed once it closes.
NOTE_FIELD = re.compile(r"^\s*-?\s*\.([a-z0-9_]+):\s*(.*?)\s*$")
NOTE_ITEM = re.compile(r"^(\s*)-\s+\.")
NOTE_KEEP = {
    "name", "symbol", "vgpr_count", "agpr_count", "sgpr_count", "vgpr_spill_count",
    "sgpr_spill_count", "group_segment_fixed_size", "private_segment_fixed_size",
    "max_flat_workgroup_size", "kernarg_segment_size",
}


def notes(path):
    """{kernel name: {field: int|str}} from the object's AMDHSA metadata note.

    A new kernel record starts at the SHALLOWEST `- .` in the block. Splitting on every
    `- .` instead splits inside `.args`, whose entries are `- .offset:` at a deeper indent,
    and every field before `.args` (`.agpr_count` among them) is then thrown away — which
    read AGPR 0 for interp_flash, whose four flash arms hold 256.
    """
    text = _run([tool("llvm-readelf"), "--notes", path], f"readelf --notes {path}")
    lines = text.splitlines()
    indents = [len(m.group(1)) for m in
               (NOTE_ITEM.match(ln) for ln in lines) if m]
    if not indents:
        return {}
    top = min(indents)

    out, rec = {}, None

    def close(rec):
        if rec and "name" in rec:
            out[rec["name"]] = rec

    for line in lines:
        m = NOTE_ITEM.match(line)
        if m and len(m.group(1)) == top:
            close(rec)
            rec = {}
        m = NOTE_FIELD.match(line)
        if not m or rec is None:
            continue
        key, val = m.group(1), m.group(2)
        if key not in NOTE_KEEP:
            continue
        rec[key] = int(val) if val.isdigit() else val
    close(rec)
    return out


# "   13: 00000000000a9c34     4 OBJECT  GLOBAL PROTECTED  10 plow_gemv_mm_cap_1"
SYMTAB = re.compile(
    r"^\s*\d+:\s+([0-9a-f]+)\s+(\d+)\s+(\S+)\s+\S+\s+\S+\s+(\S+)\s+(\S+)\s*$")
SECTION = re.compile(r"^\s*\[\s*(\d+)\]\s+(\S+)\s")
HEXDUMP = re.compile(r"^0x([0-9a-f]+)\s+((?:[0-9a-f]{2,8}\s+){1,4})")


def symbols(path):
    """[(type, name)] from .symtab/.dynsym — the marker variables and the template arms.

    Deduplicated: every global appears twice, once in each table.
    """
    return sorted({(s[2], s[4]) for s in _symtab(path)})


def _symtab(path):
    out = []
    for line in _run([tool("llvm-readelf"), "-sW", path], f"readelf -s {path}").splitlines():
        m = SYMTAB.match(line)
        if m:
            out.append((int(m.group(1), 16), int(m.group(2)), m.group(3), m.group(4),
                        m.group(5)))
    return out


def _sections(path):
    got = {}
    for line in _run([tool("llvm-readelf"), "-SW", path], f"readelf -S {path}").splitlines():
        m = SECTION.match(line)
        if m:
            got[m.group(1)] = m.group(2)
    return got


def globals_u32(path):
    """{name: value} for every 4-byte OBJECT global — the marker variables.

    The value is read out of `.data`, and a marker initialised to 0 is in `.bss` (NOBITS,
    no bytes to read) and is 0 by definition. This is what lets a marker carry an
    EXPRESSION-valued knob, which the `plow_gemv_mm_cap_<n>` name-pasting idiom cannot.
    """
    secs = _sections(path)
    words, bss = {}, set()
    for idx, name in secs.items():
        if name not in (".data", ".rodata"):
            continue
        for line in _run([tool("llvm-readelf"), f"-x{name}", path],
                         f"readelf -x {name} {path}").splitlines():
            m = HEXDUMP.match(line)
            if not m:
                continue
            addr = int(m.group(1), 16)
            blob = "".join(m.group(2).split())
            for i in range(0, len(blob) - 7, 8):
                # llvm-readelf prints target-endian bytes in order; AMDGCN is little-endian.
                b = blob[i:i + 8]
                words[addr + i // 2] = int.from_bytes(bytes.fromhex(b), "little")
    for idx, name in secs.items():
        if name == ".bss":
            bss.add(idx)
    got = {}
    for value, size, typ, ndx, name in _symtab(path):
        if typ != "OBJECT" or size != 4:
            continue
        if ndx in bss:
            got[name] = 0
        elif value in words:
            got[name] = words[value]
    return got


# gfx942 (CDNA3) occupancy, computed rather than read: the objects carry no `.occupancy`
# field and the build does not run -Rpass-analysis. Both limits, min taken.
#
#   register  512 VGPRs per SIMD lane, unified arch+acc (`.vgpr_count` is ALREADY the total;
#             `.agpr_count` is the accumulator SUBSET of it, not an addition — verified
#             against -Rpass-analysis on interp_flash, which reports VGPRs 256 + AGPRs 256 at
#             occupancy 1 while the note reads vgpr_count 512). Allocation granule is 8.
#   LDS       64 KiB per workgroup; workgroups/CU = floor(65536 / lds), spread over 4 SIMDs.
#
# Checked against the numbers build_gfx942.sh states from measurement: VGPR 253 -> 2, VGPR 104
# with a 30,736 B arena -> 4, VGPR 512 -> 1.
VGPR_PER_SIMD = 512
VGPR_GRANULE = 8
SIMDS_PER_CU = 4
# LDS per workgroup. CDNA4 has 160 KiB where CDNA3 has 64 -- the divergence build_gfx942.sh
# exists for, and the reason its GEMM stage arena is a different tile.
LDS_PER_CU = {"gfx942": 65536, "gfx950": 163840}


def occupancy(vgpr, lds, waves_per_wg, arch="gfx942"):
    """Waves per SIMD this kernel can hold, by register and by LDS."""
    if not vgpr:
        return 0
    granulated = max(VGPR_GRANULE, -(-vgpr // VGPR_GRANULE) * VGPR_GRANULE)
    by_reg = VGPR_PER_SIMD // granulated
    if not lds:
        return by_reg
    wgs = LDS_PER_CU.get(arch, 65536) // lds
    by_lds = (wgs * waves_per_wg) // SIMDS_PER_CU
    return min(by_reg, by_lds)
