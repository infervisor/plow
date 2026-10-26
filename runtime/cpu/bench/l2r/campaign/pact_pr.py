"""pact_pr.py <refbase>: per row, effective element count of ref.pact (participation ratio (sum x^2)^2 / sum x^4) and the
BF16 bar of pact vs pg; a low count means the pact rel-RMS is set by a few elements."""
import json, sys, array
base = sys.argv[1]
for r in range(16):
    d = base if r == 0 else f"{base}.o{r}"
    x = array.array("f"); x.frombytes(open(d + "/ref.pact.f32", "rb").read())
    pr = sum(v * v for v in x) ** 2 / sum(v ** 4 for v in x)
    bar = json.load(open(d + "/meta.json"))["bf16_ref_err"]
    print(f"{d.split('/')[-1]:22s} eff_n {pr:6.1f} / {len(x)}  pact bar {bar['pact']['rel_rms']:.2e}  pg bar {bar['pg']['rel_rms']:.2e}")
