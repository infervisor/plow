import json, glob, sys
root = sys.argv[1]
for d in sorted(glob.glob(root + "/*.[a-z]")):
    for f in sorted(glob.glob(d + "/g*.r*/bench.json")):
        j = json.load(open(f))
        print(f[len(root) + 1:], "tpot p50 %.2f p99 %.2f ttft p50 %.1f p99 %.1f out %.1f" % (
            j["median_tpot_ms"], j["p99_tpot_ms"], j["median_ttft_ms"], j["p99_ttft_ms"], j["output_throughput"]))
