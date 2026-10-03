"""The FP8 kernel set for Gemma-4 26B-A4B and 12B: every op, its DIMS, its RUNGS, and a roofline
verdict per kernel -- plus the bf16 packet's same shapes so the native-vs-cuBLASLt route question is
answered per kernel rather than per phase.

Source is the emitted packet's dispatch_audit, which carries per-op m/n/k, the rung `t`, `insts`
(layer multiplicity), `bytes` moved, `tile`, `occupancy` and `waste_bytes`. Roofline:
  FLOPs = 2*m*n*k*insts ; AI = FLOPs / bytes
  ridge(H100 SXM5) = peak_flops / 3352 GB/s  ->  bf16 295, fp8 590 FLOP/B
  AI < ridge => memory bound, floor = bytes / BW ; else compute bound, floor = FLOPs / peak

An arm whose name ends Fp8/W8a8/W8a16 is a quantised body. cuBLASLt can only ever appear on
Gemm/GemmMed/GemmSmall (dense_cublaslt::prefill_eligible), so the ROUTE column is derivable:
fp8 ops are native by construction -- there is no fp8 Lt path in plowrt at all.
"""
import collections
import json
import sys

BW = 3352e9
PEAK = {"bf16": 989e12, "fp8": 1979e12}
RIDGE = {k: v / BW for k, v in PEAK.items()}

PKTS = [
    ("26B-A4B fp8", "/opt/dlami/nvme/tmp/agent-geom/p26fp8/assets/build.json", "fp8"),
    ("26B-A4B bf16", "/opt/dlami/nvme/tmp/agent-geom/p26lad2/assets/build.json", "bf16"),
    ("12B fp8", "/opt/dlami/nvme/tmp/agent-geom/p12fp8c/assets/build.json", "fp8"),
]
Q = ("Fp8", "W8a8", "W8a16")
LT_ELIGIBLE = {"Gemm", "GemmMed", "GemmSmall"}


def prec_of(op, packet_prec):
    return "fp8" if any(q in op for q in Q) else packet_prec if packet_prec == "bf16" else "bf16"


def main():
    for tag, path, pprec in PKTS:
        try:
            d = json.load(open(path))
        except OSError:
            print(f"== {tag}: MISSING {path}\n")
            continue
        ops = (d.get("dispatch_audit") or {}).get("ops") or []
        print("=" * 108)
        print(f"== {tag}   precision={json.dumps(d.get('precision'))}")
        print(f"   {len(ops)} audited op rows;  ridge bf16 {RIDGE['bf16']:.0f} / fp8 "
              f"{RIDGE['fp8']:.0f} FLOP/B")

        for kind in ("prefill", "decode"):
            rows = [o for o in ops if o.get("kind") == kind]
            if not rows:
                continue
            # group by the SHAPE identity: op + n + k. m/t vary with the rung.
            g = collections.defaultdict(list)
            for o in rows:
                g[(o["op"], o.get("n"), o.get("k"))].append(o)
            print(f"\n   --- {kind.upper()}: {len(rows)} rows, {len(g)} distinct (op,N,K) ---")
            print(f"   {'op':<26} {'N':>7} {'K':>6} {'ins':>4} {'rungs t':>22} "
                  f"{'tile':>9} {'occ':>5} {'AI@max':>8} {'bound':>7} {'1x ms':>7} {'xins ms':>8} {'route':>7}")
            for key in sorted(g, key=lambda k: (-max(x.get("bytes", 0) for x in g[k]),)):
                op, n, k = key
                rs = g[key]
                ts = sorted({x.get("t") for x in rs if x.get("t") is not None})
                big = max(rs, key=lambda x: x.get("bytes", 0))
                insts = big.get("insts", 1)
                m = big.get("m") or 0
                by = big.get("bytes") or 0
                # bytes in the audit is PER INSTRUCTION, so FLOPs must be too; the per-layer
                # total is floor*insts, reported separately.
                fl = 2 * m * (n or 0) * (k or 0)
                ai = fl / by if by else 0.0
                pr = prec_of(op, pprec)
                mem = by / BW * 1e3
                cmp_ = fl / PEAK[pr] * 1e3
                bound = "mem" if ai < RIDGE[pr] else "compute"
                floor = max(mem, cmp_)
                total = floor * insts
                route = "Lt?" if op in LT_ELIGIBLE and pr == "bf16" else "native"
                tshow = ",".join(str(x) for x in ts)
                if len(tshow) > 22:
                    tshow = f"{ts[0]}..{ts[-1]} ({len(ts)})"
                print(f"   {op:<26} {n or 0:>7} {k or 0:>6} {insts:>4} {tshow:>22} "
                      f"{str(big.get('tile')):>9} {(big.get('occupancy') or 0.0):>5.2f} "
                      f"{ai:>8.1f} {bound:>7} {floor:>7.3f} {total:>8.3f} {route:>7}")
        print()


main()
