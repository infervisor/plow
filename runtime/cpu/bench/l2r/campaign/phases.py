"""phases.py <stage.json>...: per-phase compute max / mean, prologue mean and barrier mean (us) per run."""
import json, sys
N = ["qkv", "attn", "comb", "o", "gu", "down", "pleg", "plep"]
for f in sys.argv[1:]:
    j = json.loads(open(f).read().strip().splitlines()[-1])
    ph = j["phase_us"]
    print(f"{f.split('/')[-1]:28s} p50 {j['step_us']['p50']:7.1f} | " +
          " ".join(f"{n}:{p['compute_max']:.1f}/{p['prologue_mean']:.1f}/{p['barrier_mean']:.1f}" for n, p in zip(N, ph)))
