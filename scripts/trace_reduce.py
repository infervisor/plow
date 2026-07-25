#!/usr/bin/env python3
# trace_reduce.py — reduce PLOW_NV_TRACE=1 block-0 per-packet dump into a per-opcode
# gate/body/sig cycle table. Reads harness stdout on stdin (or a file arg).
#
# Input lines (from interp_sm120.cu launch helper):
#   PLOW_TRACE_N <n>
#   PLOW_TRACE <i> op=<op> wait=<wait_len> gate=<cyc> body=<cyc> sig=<cyc>
#
# Reads THE SHAPE, not the absolute total — clock64() serializes on the recording
# thread, so the sum over-reports vs the untraced step. Ratios/per-op splits are the
# finding (see the trace facility comment in interp_sm120.cu).
import sys, re

OPNAME = {
 0:"NOP",1:"RMSNORM",2:"ROWRMS",3:"HEADNORM_ROPE",4:"RESIDUAL",5:"GLU",6:"EMBED",
 7:"SOFTCAP",8:"GEMM",9:"GEMM_NORM",10:"GEMV",11:"FLASH_PREFILL",12:"FLASH_DECODE",
 13:"FLASH_MERGE",14:"GEMM_SMALL",15:"GEMM_MED",16:"NORM_RESIDUAL",17:"ARGMAX",
 18:"ARGMAX_FIN",19:"GEMV_GLU",20:"GEMM_GLU",21:"ADD_NORM",22:"GEMV_QKV",
 23:"NORM_RESIDUAL_NORM",30:"GEMV_FP8",31:"GEMV_GLU_FP8",
}

def main():
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    rows = []
    for ln in src:
        m = re.search(r"PLOW_TRACE (\d+) op=(\d+) wait=(\d+) gate=(\d+) body=(\d+) sig=(\d+)", ln)
        if m:
            i,op,wl,ga,bo,si = map(int, m.groups())
            rows.append((op,wl,ga,bo,si))
    if not rows:
        print("no PLOW_TRACE lines found"); return
    # per-op aggregate
    agg = {}   # op -> [count, gate, body, sig, wait_sum]
    for op,wl,ga,bo,si in rows:
        a = agg.setdefault(op, [0,0,0,0,0])
        a[0]+=1; a[1]+=ga; a[2]+=bo; a[3]+=si; a[4]+=wl
    tot_g = sum(a[1] for a in agg.values())
    tot_b = sum(a[2] for a in agg.values())
    tot_s = sum(a[3] for a in agg.values())
    tot   = tot_g+tot_b+tot_s
    n = len(rows)
    print(f"packets(block0)={n}  total_cyc={tot}  gate={tot_g} ({100*tot_g/tot:.1f}%)  "
          f"body={tot_b} ({100*tot_b/tot:.1f}%)  sig={tot_s} ({100*tot_s/tot:.1f}%)")
    print()
    hdr = f"{'op':<20}{'cnt':>5}{'gate_cyc':>12}{'body_cyc':>12}{'sig_cyc':>10}{'tot_cyc':>12}{'%tot':>7}{'g/op':>9}{'b/op':>9}"
    print(hdr); print("-"*len(hdr))
    for op in sorted(agg, key=lambda o:-(agg[o][1]+agg[o][2]+agg[o][3])):
        c,g,b,s,w = agg[op]
        t = g+b+s
        name = OPNAME.get(op, f"op{op}")
        print(f"{name:<20}{c:>5}{g:>12}{b:>12}{s:>10}{t:>12}{100*t/tot:>6.1f}%{g//c:>9}{b//c:>9}")
    print("-"*len(hdr))
    print(f"{'TOTAL':<20}{n:>5}{tot_g:>12}{tot_b:>12}{tot_s:>10}{tot:>12}{100:>6.0f}%")

if __name__ == "__main__":
    main()
