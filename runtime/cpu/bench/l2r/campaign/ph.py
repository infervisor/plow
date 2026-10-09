import json,glob,statistics as st,sys
d='/tmp/g4c/l2r/results/p3v2'
for ref in ['e2b.L0.c2048','e2b.L4.c16384','e4b.L0.c2048']:
  for m in ['nobcast.diss','repnt.diss','fid.diss']:
    v=[json.loads(open(f).read().strip().splitlines()[-1]) for f in sorted(glob.glob(f'{d}/{ref}.{m}.r*.json'))]
    c=[st.mean(x['phase_us'][i]['compute_max'] for x in v) for i in range(8)]
    b=[st.mean(x['phase_us'][i]['barrier_mean'] for x in v) for i in range(8)]
    p=[st.mean(x['phase_us'][i]['prologue_mean'] for x in v) for i in range(8)]
    print(ref,m,'C',' '.join(f'{a:.1f}' for a in c),'| P',' '.join(f'{a:.1f}' for a in p),'| B',' '.join(f'{a:.1f}' for a in b))
