import sys, json
for l in sys.stdin:
    if ' {' not in l:
        continue
    a, j = l.split(' ', 1)
    d = json.loads(j)
    t = ' ttfa %.0f/%.0f ms' % (d['med_ttfa_ms'], d['p90_ttfa_ms']) if 'med_ttfa_ms' in d else ''
    print(f"{a:8s} {d['tag']:12s} fail {d['failed']} audio_s/s {d['audio_s_per_s']:7.2f} rtf {d['med_rtf']:.3f}{t}")
