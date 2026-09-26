import json, glob, sys, os
# summ.py <resdir> : every <arm>/<tag>.json under it, one line each
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*.json"))):
    try:
        s = json.load(open(f))["summary"]
    except Exception:
        continue
    t = " ttfa %.0f/%.0f ms" % (s["med_ttfa_ms"], s["p90_ttfa_ms"]) if "med_ttfa_ms" in s else ""
    print(f"{f.split('/')[-2]:8s} {s['tag']:11s} fail {s['failed']} audio_s/s {s['audio_s_per_s']:6.2f} rtf {s['med_rtf']:.3f}{t}")
