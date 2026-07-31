import json, statistics
# Prompt-lookup (n-gram) speculation, simulated at WORD granularity on GSM8K 8-shot.
# For each generated position, find the most recent earlier occurrence of the last N words in the
# context (prompt + text so far) and propose the K words that followed it. Count the matched prefix.
data=[json.loads(l) for l in open('/home/lava/models/gsm8k/test.jsonl')]
shots=data[:8]; qs=data[8:58]
def norm(a): return a.replace('####','The answer is')
pre=[]
for s in shots:
    pre += s['question'].split() + norm(s['answer']).split()
def sim(ctx, gen, N, K):
    ctx=list(ctx); acc=[]
    i=0
    while i < len(gen):
        # propose
        prop=[]
        if len(ctx)>=N:
            key=tuple(ctx[-N:])
            for j in range(len(ctx)-N-1, -1, -1):
                if tuple(ctx[j:j+N])==key:
                    prop=ctx[j+N:j+N+K]; break
        m=0
        while m<len(prop) and i+m<len(gen) and prop[m]==gen[i+m]: m+=1
        acc.append(m)
        take=m+1                      # accepted prefix + the target's own bonus token
        ctx += gen[i:i+take]; i+=take
    return acc
for N,K in ((2,4),(3,4),(3,8),(4,8)):
    allacc=[]
    for q in qs:
        ctx = pre + q['question'].split()
        gen = norm(q['answer']).split()
        allacc += sim(ctx, gen, N, K)
    mean=statistics.mean(allacc)
    print(f"N={N} K={K}: mean accepted={mean:.2f} -> tokens per verify step={mean+1:.2f} "
          f"(steps saved {100*(1-1/(mean+1)):.0f}%)  full-accept rate={100*sum(1 for a in allacc if a==K)/len(allacc):.0f}%")
