import re, collections, sys
names = {0:"NOP",1:"RMSNORM",2:"ROWRMS",3:"HEADNORM_ROPE",4:"RESIDUAL",5:"GLU",6:"EMBED",7:"SOFTCAP",
8:"GEMM",9:"GEMM_NORM",10:"GEMV",11:"FLASH_PREFILL",12:"FLASH_DECODE",13:"FLASH_MERGE",14:"GEMM_SMALL",
15:"GEMM_MED",16:"NORM_RESIDUAL",17:"ARGMAX",18:"ARGMAX_FIN",19:"GEMV_GLU",20:"GEMM_GLU",21:"ADD_NORM",
22:"GEMV_QKV",23:"NORM_RESIDUAL_NORM",30:"GEMV_FP8",31:"GEMV_GLU_FP8",
61:"MOE_ROUTER_GEMMA",62:"MOE_EXPERT_GLU_GEMMA",63:"MOE_EXPERT_DOWN_GEMMA",64:"MOE_COMBINE_GEMMA",
65:"MOE_EXPERT_GLU_GEMMA_FP8",66:"MOE_EXPERT_DOWN_GEMMA_FP8",67:"MOE_ROUTER_GEMMA_SCORE",
68:"MOE_ROUTER_GEMMA_TOPK",69:"MOE_ROUTER_GEMMA_SCORE_FAST",70:"MOE_COMBINE_NORM_GEMMA",
71:"MOE_EXPERT_GLU_NORM_GEMMA",72:"MOE_COMBINE_RESID_NORM_GEMMA"}
rows=[]
for ln in open(sys.argv[1]):
    m=re.match(r"PLOW_TRACE \d+ op=(\d+) wait=(\d+) gate=(\d+) body=(\d+) sig=(\d+)",ln)
    if m: rows.append(tuple(int(x) for x in m.groups()))
agg=collections.defaultdict(lambda:[0,0,0,0])  # count, gate, body, sig
for op,wait,g,b,s in rows:
    a=agg[op]; a[0]+=1; a[1]+=g; a[2]+=b; a[3]+=s
tot_g=sum(a[1] for a in agg.values()); tot_b=sum(a[2] for a in agg.values()); tot_s=sum(a[3] for a in agg.values())
grand=tot_g+tot_b+tot_s
print(f"packets(block0)={len(rows)}  totals: gate={tot_g} body={tot_b} sig={tot_s} sum={grand}")
print(f"  gate%={100*tot_g/grand:.1f} body%={100*tot_b/grand:.1f} sig%={100*tot_s/grand:.1f}\n")
# rank by total cycles (gate+body+sig)
order=sorted(agg.items(), key=lambda kv:-(kv[1][1]+kv[1][2]+kv[1][3]))
print(f"{'opcode':<30}{'cnt':>4}{'tot_cyc':>10}{'%':>6}{'mean':>9}{'body':>10}{'gate':>10}{'sig':>8}{'body%':>7}")
for op,a in order:
    cnt,g,b,s=a; t=g+b+s
    nm=names.get(op,f"op{op}")
    print(f"{nm:<30}{cnt:>4}{t:>10}{100*t/grand:>6.1f}{t/cnt:>9.0f}{b:>10}{g:>10}{s:>8}{100*b/t if t else 0:>7.1f}")
