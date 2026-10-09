"""rows_e4b.py: final-table numbers for the E4B L5 stage runs (p5_stab): means of 2 reps."""
import glob, json, os, statistics as st, sys
sys.path.insert(0, "/home/ec2-user/plow/.claude/worktrees/gemma4-xeon-bf16/runtime/cpu/bench/l2r")
import p2_gate
GEMV = (0, 3, 4, 5, 6, 7)
O = "/tmp/g4c/l2r/results/p5_stab"
for c in (2048, 16384, 131072):
    js, cs, ok_all, fails = [], [], True, []
    for f in sorted(glob.glob(f"{O}/e4b.L5.c{c}.avx.lock12.r*.json")):
        if f.endswith(".ctr.json"):
            continue
        j = json.loads(open(f).read().strip().splitlines()[-1]); js.append(j)
        ok, rows = p2_gate.gate(j["err"], j["ref"]); ok_all &= ok; fails += [x for x in rows if x.endswith("!")]
        cs.append(json.load(open(f[:-5] + ".ctr.json")))
    m = lambda f: st.mean(f(j) for j in js)
    print(f"c{c}: step p50 {m(lambda j: j['step_us']['p50']):.1f} p99 {m(lambda j: j['step_us']['p99']):.1f} | "
          f"gemv {m(lambda j: sum(j['phase_us'][i]['compute_max'] for i in GEMV)):.1f} "
          f"attn {m(lambda j: j['phase_us'][1]['compute_max'] + j['phase_us'][2]['compute_max']):.1f} "
          f"sync {m(lambda j: sum(p['barrier_mean'] for p in j['phase_us'])):.1f} | "
          f"bytes/worker {js[0]['weight_bytes_per_worker']:.0f} resident_kib {js[0]['resident_kib']} | "
          f"held {min(j['lock']['held_l2_after_min'] for j in js):.4f} | "
          f"DRAM {st.mean(x['dram_mib_step'] for x in cs):.1f} MiB/step (KV {cs[0]['kv_mib_step']:.1f}) L2 in {st.mean(x['l2_lines_in'] for x in cs):,.0f}/worker | "
          f"gate {'PASS' if ok_all else 'FAIL ' + ' '.join(sorted(set(fails)))} out cos {min(j['err']['out'][1] for j in js) if js[0]['err']['out'] else 'n/a'}")
