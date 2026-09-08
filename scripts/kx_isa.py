#!/usr/bin/env python3
"""kx --isa: what does this body ACTUALLY issue, and does it actually spill?

Two independent sources, deliberately printed side by side:

  ISA      llvm-objdump of the code object, counted per kernel symbol. This is where SPILL comes
           from -- `scratch_load_*` / `scratch_store_*` -- because candidates in this tree have
           reported `.vgpr_spill_count: 0` in the metadata and spilled anyway. It is also where
           the instruction MIX comes from, which has been load-bearing: the flash-prefill softmax
           win came from noticing one KV tile issues 160 ds_bpermute and 163 s_waitcnt against
           64 MFMAs.
  META     the compiler's own -Rpass-analysis=kernel-resource-usage remarks (VGPR/AGPR/SGPR/LDS/
           occupancy). Useful, but not trusted for spill: the row is flagged when it disagrees
           with the ISA count.

THE COUNTING LIVES IN scripts/plow_isa.py, shared with `asm_audit.py --contract`, which asserts
budgets on the shipped objects. Two copies of the regex table meant two definitions of `spill`;
one module means a number measured here is the number the build asserts. Only the PRESENTATION
is local -- kx wants loads and stores split, the build audit wants global and buffer split.

Run through scripts/kx.sh --isa; it supplies the nix toolchain this needs.
"""
import argparse
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plow_isa as isa  # noqa: E402

# kx's view of the shared mnemonic counter: ordered, first match wins.
CLASSES = [
    ("scratch", re.compile(r"^scratch_(load|store)")),
    ("bpermute", re.compile(r"^ds_(b?permute)")),
    ("swizzle", re.compile(r"^ds_swizzle")),
    ("ds_read", re.compile(r"^ds_read")),
    ("ds_write", re.compile(r"^ds_write")),
    ("mfma", re.compile(r"^v_(mfma|smfmac)")),
    ("waitcnt", re.compile(r"^s_waitcnt")),
    ("glb_ld", re.compile(r"^(global_load|buffer_load|flat_load)")),
    ("glb_st", re.compile(r"^(global_store|buffer_store|flat_store)")),
    ("valu", re.compile(r"^v_")),
    ("salu", re.compile(r"^s_")),
]
COLS = ["total", "scratch", "bpermute", "swizzle", "ds_read", "ds_write", "mfma", "waitcnt",
        "glb_ld", "glb_st", "valu", "salu"]


def bucket(sym):
    c = dict.fromkeys(COLS, 0)
    c["total"] = sym.total
    for op, n in sym.insn.items():
        for name, pat in CLASSES:
            if pat.match(op):
                c[name] += n
                break
    return c


def meta_from_remarks(path):
    """-Rpass-analysis=kernel-resource-usage remarks, as hipcc printed them at build time."""
    if not path or not os.path.exists(path):
        return {}
    txt = open(path, errors="replace").read()
    per, cur = {}, None
    for line in txt.splitlines():
        m = re.search(r"Function Name:\s*(\S+)", line)
        if m:
            cur = m.group(1)
            per[cur] = {}
            continue
        if cur is None:
            continue
        m = re.search(r"(SGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Occupancy \[waves/SIMD\]|"
                      r"SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]):\s*([0-9.]+)", line)
        if m:
            per[cur][m.group(1)] = m.group(2)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co", required=True)
    ap.add_argument("--arch", default="gfx942")
    ap.add_argument("--label", default="")
    ap.add_argument("--res", default="")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        elf = isa.unbundle(a.co, a.arch, tmp)
        counts = {s.name: bucket(s) for s in isa.counts(elf, a.arch)}
    meta = meta_from_remarks(a.res)

    arms = sorted(k for k in counts if not k.startswith("kx_") and counts[k]["total"] > 1)
    print(f"\nISA  variant '{a.label}'  {os.path.basename(a.co)}  [{a.arch}]")
    print(f"{'arm':<16}" + "".join(f"{c:>9}" for c in COLS))
    for k in arms:
        c = counts[k]
        print(f"{k:<16}" + "".join(f"{c[x]:>9}" for x in COLS))

    print(f"\nMETA (compiler remarks)  vs  ISA spill")
    print(f"{'arm':<16}{'VGPR':>6}{'AGPR':>6}{'SGPR':>6}{'LDS':>8}{'occ':>5}"
          f"{'meta_scratch_B':>15}{'meta_spill':>11}{'ISA scratch ops':>17}  note")
    for k in arms:
        m = meta.get(k, {})
        isa_scratch = counts[k]["scratch"]
        mspill = int(float(m.get("VGPRs Spill", 0) or 0)) + int(float(m.get("SGPRs Spill", 0) or 0))
        mscr = float(m.get("ScratchSize [bytes/lane]", 0) or 0)
        note = ""
        if isa_scratch and mspill == 0:
            note = "<- METADATA SAYS NO SPILL, THE ISA DISAGREES. Trust the ISA."
        elif isa_scratch:
            note = "spills"
        print(f"{k:<16}{m.get('VGPRs','?'):>6}{m.get('AGPRs','?'):>6}{m.get('SGPRs','?'):>6}"
              f"{m.get('LDS Size [bytes/block]','?'):>8}{m.get('Occupancy [waves/SIMD]','?'):>5}"
              f"{mscr:>15.0f}{mspill:>11}{isa_scratch:>17}  {note}")


if __name__ == "__main__":
    main()
