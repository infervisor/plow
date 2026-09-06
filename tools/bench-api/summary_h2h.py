import re, sys
W = '/home/lava/plow/.claude/worktrees/cpu-backend/perf-data/'
SRC = [
    ('plow bf16', W + 'cpu-gemma/h2h/plow-bf16.md', W + 'cpu-gemma/h2h/plow-bf16-rag.md'),
    ('plow fp8', W + 'cpu-gemma/h2h/plow-fp8.md', W + 'cpu-gemma/h2h/plow-fp8-rag.md'),
    ('vLLM bf16', W + 'cpu-gemma/h2h/vllm-bf16.md', W + 'cpu-gemma/h2h/vllm-bf16-rag.md'),
    ('llama.cpp bf16', W + 'cpu-gemma/llamacpp/results-bf16.md', None),
    ('llama.cpp Q8_0', W + 'cpu-gemma/llamacpp/results-q8_0.md', None),
    ('llama.cpp Q4_K_M', W + 'cpu-gemma/llamacpp/results-q4_k_m.md', None),
]
ROW = re.compile(r'^\| (\w+) \| (\d+) \| \d+ \| (\d+) \| (\S+) \| \S+ \| (\S+)/(\S+)/\S+ \| (\S+)/(\S+)/\S+ \| \S+ \| (\S+) \| \S+ \|')


def load(paths):
    d = {}
    for p in paths:
        if not p:
            continue
        try:
            for line in open(p):
                m = ROW.match(line)
                if m:
                    wl, c, err, intok, ttft_mean, ttft_p50, tpot_mean, tpot_p50, tps = m.groups()
                    d[(wl, int(c))] = (int(err), intok, ttft_p50, tpot_p50, tps)
        except FileNotFoundError:
            pass
    return d


data = {name: load([a, b]) for name, a, b in SRC}
out = ['# Gemma-4-12B-it on CPU (16 threads): plow vs vLLM 0.28 vs llama.cpp 6a1a922',
       '',
       'Same seeded bench-api workloads (64 output tokens, 8 requests per cell, rag_4k 4), each server alone on the box. '
       'p50 TTFT ms / p50 TPOT ms / aggregate out tok/s. plow numbers are the 2026-09-06 00:20 build (before the fp8 dequant, '
       'chunked-prefill and AMX staging changes).', '',
       '**Caveat (c >= 2 cells):** the same 8 prompts were reused across concurrency levels, and llama-server (slot prompt '
       'cache) and vLLM (prefix caching, on by default) serve repeated prompts from cache — llama.cpp bf16 chat_long TTFT '
       'drops from 13 s at c=1 to 0.66 s at c=2, impossible at its ~35 tok/s prefill. plow has no prefix cache. Only the '
       'c=1 rows (first use of each prompt) compare prefill fairly; c >= 2 TTFT/throughput flatter the baselines. '
       'Re-run with `bench.py --fresh-prompts` pending.', '']
for wl, label in [('chat_short', 'chat_short (34 in)'), ('chat_long', 'chat_long (398 in)'), ('code', 'code (365 in)'),
                  ('summarize', 'summarize (1105 in)'), ('rag_4k', 'rag_4k (~3000 in)')]:
    out.append(f'## {label}')
    out.append('')
    out.append('| conc | ' + ' | '.join(n for n, _, _ in SRC) + ' |')
    out.append('|---|' + '---|' * len(SRC))
    for c in (1, 2, 4, 8):
        cells = []
        for name, _, _ in SRC:
            v = data[name].get((wl, c))
            if not v:
                cells.append('-')
            elif v[0]:
                cells.append(f'err {v[0]}')
            else:
                cells.append(f'{v[2]} / {v[3]} / {v[4]}')
        if any(x != '-' for x in cells):
            out.append(f'| {c} | ' + ' | '.join(cells) + ' |')
    out.append('')
open(W + 'cpu-gemma/h2h/SUMMARY.md', 'w').write('\n'.join(out))
print('\n'.join(out))
