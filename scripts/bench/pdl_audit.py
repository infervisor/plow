#!/usr/bin/env python3
"""pdl_audit.py <cubin|dir>...: every kernel a cubin marks `plow_pdl_entry__<kernel>`
(runtime/nvidia/pdl.cuh) must execute griddepcontrol.wait (SASS ACQBULK) before any memory
access or control flow, and griddepcontrol.launch_dependents (PREEXIT) after it, because plowrt
launches it as a programmatic dependent. A marker must name exactly one entry. Prints one line
per marked kernel and exits 1 on any violation or a marker without its kernel.
Env CUOBJDUMP (default: cuobjdump on PATH, else /usr/local/cuda/bin/cuobjdump)."""
import os, re, shutil, subprocess, sys

CUOBJDUMP = os.environ.get("CUOBJDUMP") or shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
MARK = "plow_pdl_entry__"
# Before the wait only register, predicate, special-register and constant-bank instructions may
# run: no memory access, no control flow, nothing unknown.
ALLOW = re.compile(r"^(LDC|ULDC|S2R|S2UR|CS2R|MOV|MOV32I|UMOV|IMAD|UIMAD|IADD3|UIADD3|IADD|VIADD|UVIADD|VIADDMNMX|SHF|USHF|LEA|ULEA|LOP3|ULOP3|R2UR|NOP|SEL|USEL|ISETP|UISETP|PRMT|UPRMT|PLOP3|UPLOP3|P2R|R2P|IABS|IMNMX|FLO|POPC|BMSK|UBMSK|VOTEU|VOTE|I2F|I2FP|F2I|F2F|F2FP|FRND|MUFU|FMUL|FADD|FFMA|FSEL|FSETP|FMNMX|HADD2|HMUL2|HFMA2|IMUL|SGXT|BREV|UFLO|UPOPC)(\.|$)")
INST = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")


def run(*args):
    return subprocess.run([CUOBJDUMP, *args], capture_output=True, text=True, check=True).stdout


def audit(path):
    elf = run("-elf", path)
    marks = sorted({m for m in re.findall(MARK + r"(\w+)", elf)})
    bad = 0
    for mark in marks:
        # A C++ entry is marked by its identifier; plowrt strips the mangling the same way.
        # plowrt strips the mangling, so the marker must name exactly one entry.
        entries = sorted(set(re.findall(r"\.text\.(%s|_Z%d%s\w*)\b" % (mark, len(mark), mark), elf)))
        if len(entries) > 1:
            print(f"FAIL {path}: {mark}: marker names {len(entries)} entries {entries}")
            bad += 1
            continue
        kernel = entries[0] if entries else mark
        try:
            sass = run("-sass", "-fun", kernel, path)
        except subprocess.CalledProcessError:
            sass = ""
        insts = INST.findall(sass)
        ops = [op for _, op in insts]
        if not ops:
            print(f"FAIL {path}: {kernel}: marked but no such kernel")
            bad += 1
            continue
        if "ACQBULK" not in ops:
            print(f"FAIL {path}: {kernel}: no griddepcontrol.wait")
            bad += 1
            continue
        at = ops.index("ACQBULK")
        early = [op for op in ops[:at] if not ALLOW.match(op)]
        if insts[at][0]:
            print(f"FAIL {path}: {kernel}: predicated wait")
            bad += 1
        elif early:
            print(f"FAIL {path}: {kernel}: {early[0]} before the wait")
            bad += 1
        elif "PREEXIT" not in ops[at:]:
            print(f"FAIL {path}: {kernel}: no griddepcontrol.launch_dependents after the wait")
            bad += 1
        else:
            print(f"ok   {path}: {kernel} (wait at instruction {at})")
    return len(marks), bad


def main():
    files = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for root, _, names in os.walk(arg):
                files += [os.path.join(root, n) for n in sorted(names) if n.endswith(".cubin")]
        else:
            files.append(arg)
    if not files:
        raise SystemExit(__doc__)
    total = bad = 0
    for f in files:
        n, b = audit(f)
        total, bad = total + n, bad + b
    print(f"pdl_audit: {total} marked kernels in {len(files)} cubins, {bad} violations")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
