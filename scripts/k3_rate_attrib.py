#!/usr/bin/env python3
"""k3_rate_attrib.py — per-opcode ACHIEVED BANDWIDTH for a K3 decode token.

`k3_trace_report.py` answers "where does the time go" (gate vs body per opcode). It carries no
byte accounting at all, so it cannot answer the question that actually ranks the body half:

    WHICH TENSORS ARE MOVING SLOWLY?

That distinction is not academic. `PLOW_K3_SHARD_HEAD` removes 14.7% of the GEMV bytes and
measured **-1.14% of the token**, because `lm_head` was already running at ~94% of peak — its
bytes were the cheapest bytes in the model. A byte-reduction lever is worth

    bytes_removed / achieved_bandwidth OF THOSE BYTES        not   bytes_removed / peak

so the money is in the worst-RATE tensors, not the biggest ones. This is the instrument that
finds them. It is the K3 twin of `glm52_token_attrib.py`'s `%6200` column, which K3 never had.

    k3_rate_attrib.py <trace.bin> <disasm.txt> [--peak 6200] [--tpus 100]

`<disasm.txt>` is `plowrt disasm --program 1 <model.pkt>` (operand names + M/N/K), and
`<trace.bin>` is what `PLOW_TRACE_RAW=<path>` leaves behind for the same blob.

BYTES ARE THE WEIGHT OPERAND, not the activation: decode is M=1, so a GEMV reads N*K weight
elements and ~N+K activation elements, and at N,K in the thousands the activation is noise. dtype
is inferred from the opcode family (see `DT`), which is the same convention
`glm52-decode-attribution.md` §1.3 uses, including its correction that a declared tensor size is
not what is read.

CLOCK: s_memrealtime is ~100 MHz on gfx950 — MEASURED at 94.6 MHz against wall clock, which
settles a disagreement between `interp.hip` (which said 1 GHz) and `dev_isa.h`. HSA's
TIMESTAMP_FREQUENCY reports 1 GHz but that is the SYSTEM timestamp, a different counter.
"""
import re
import struct
import sys
from collections import defaultdict

REC = struct.Struct("<IIIHHQQQ")


def op_names(path="runtime/common/dev_isa.h"):
    try:
        return {int(m.group(2)): m.group(1)
                for m in re.finditer(r"PLOW_DOP_(\w+)\s*=\s*(\d+)", open(path).read())}
    except OSError:
        return {}


# Bytes per weight element, by opcode-name substring. Order matters: first match wins.
DT = [
    ("MXFP4", 0.53125),   # 4 bits + E8M0 scale per 32
    ("FP8_BLK", 1.0625),  # 1 B + f32 scale per [128,128] block
    ("FP8", 1.0),
    ("", 2.0),            # bf16
]


def elem_bytes(name):
    for k, v in DT:
        if k in name:
            return v
    return 2.0


def main():
    argv = sys.argv[1:]
    if len(argv) < 2:
        sys.exit(__doc__)
    trace, disasm = argv[0], argv[1]
    peak = float(argv[argv.index("--peak") + 1]) if "--peak" in argv else 6200.0
    tpus = float(argv[argv.index("--tpus") + 1]) if "--tpus" in argv else 100.0

    names = op_names()

    # ---- per-op WEIGHT BYTES, off the disassembly -------------------------------------------
    # `#<inst>  <OPCODE>  b=<blocks> ... | M=.. N=.. K=..`
    bytes_of, shape_of = {}, {}
    for line in open(disasm, errors="ignore"):
        m = re.match(r"#(\d+)\s+(\S+)\s+b=(\d+)", line)
        if not m:
            continue
        inst, opn = int(m.group(1)), m.group(2)
        nk = re.search(r"\bN=(\d+)\s+K=(\d+)", line)
        if nk:
            N, K = int(nk.group(1)), int(nk.group(2))
            bytes_of[inst] = N * K * elem_bytes(opn)
            shape_of[inst] = f"N={N} K={K}"

    # ---- per-op WALL TIME, off the trace ----------------------------------------------------
    blob = open(trace, "rb").read()
    pk = {}
    for i in range(len(blob) // REC.size):
        cu, pc, inst, op, sl, ta, tr, te = REC.unpack_from(blob, i * REC.size)
        if not te:
            continue
        e = pk.get(inst)
        if e is None:
            pk[inst] = [op, 1, ta, tr, te]
        else:
            e[1] += 1
            e[2] = min(e[2], ta)
            e[3] = max(e[3], tr)
            e[4] = max(e[4], te)

    # ---- aggregate by opcode ---------------------------------------------------------------
    agg = defaultdict(lambda: [0, 0.0, 0.0])  # op -> n, body_us, bytes
    for inst, (op, b, arr, rdy, end) in pk.items():
        a = agg[op]
        a[0] += 1
        a[1] += (end - rdy) / tpus
        a[2] += bytes_of.get(inst, 0.0)

    tot_us = sum(v[1] for v in agg.values())
    tot_gb = sum(v[2] for v in agg.values()) / 1e9
    print(f"{trace}: {len(pk)} packets, body {tot_us/1000:.3f} ms, "
          f"weight bytes {tot_gb:.3f} GB/rank/token\n")
    print(f"{'opcode':<26}{'pkts':>6}{'ms':>9}{'%body':>7}{'GB':>8}{'GB/s':>9}{'%peak':>7}   "
          f"{'ms if at 94% peak':>17}")
    rows = []
    for op, (n, us, by) in agg.items():
        ms, gb = us / 1000.0, by / 1e9
        rate = gb / (us / 1e6) if us else 0.0
        ideal = (gb / (peak * 0.94)) * 1000.0 if gb else 0.0
        rows.append((ms, names.get(op, f"op{op}"), n, gb, rate, 100 * rate / peak, ideal))
    for ms, nm, n, gb, rate, pct, ideal in sorted(rows, reverse=True):
        r = f"{rate:>9.0f}{pct:>6.1f}%" if gb else f"{'—':>9}{'—':>7}"
        i = f"{ideal:>17.3f}" if gb else f"{'—':>17}"
        print(f"{nm:<26}{n:>6}{ms:>9.3f}{100*ms/tot_us*1000:>6.1f}%{gb:>8.3f}{r}{i}")

    # ---- the ranking this exists for -------------------------------------------------------
    print("\nRECOVERABLE BY RATE ALONE — ms that would vanish if this opcode ran at 94% of peak")
    print("(94% is what lm_head actually achieves with the SAME d_gemv, so it is a demonstrated")
    print(" target for these shapes, not a theoretical one)\n")
    cand = [(ms - ideal, nm, ms, ideal, pct)
            for ms, nm, n, gb, rate, pct, ideal in rows if gb and ms > ideal]
    for gain, nm, ms, ideal, pct in sorted(cand, reverse=True)[:12]:
        print(f"  {nm:<26} {ms:>7.3f} -> {ideal:>6.3f} ms   saves {gain:>6.3f} ms  (at {pct:.1f}% of peak)")
    print(f"\n  TOTAL recoverable by rate: {sum(c[0] for c in cand):.3f} ms")


if __name__ == "__main__":
    main()
