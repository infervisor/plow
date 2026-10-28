"""slices.py: per-worker weight slice bytes of an l2r_layer stage (same split() as the C), AMX (16-row units) and AVX
(4-row units): mean and max over the 90 workers, and the matrix that sets the max."""
import json, sys
NW = 90


def split(n, unit, w):
    u = n // unit
    return u * w // NW * unit, u * (w + 1) // NW * unit


for ref in sys.argv[1:]:
    m = json.load(open(ref + "/meta.json"))
    man = {l.split()[0]: [int(x) for x in l.split()[2:]] for l in open(ref + "/manifest.txt")}
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
             "mlp.down_proj", "per_layer_input_gate", "per_layer_projection"]
    shp = [man[f"w.{n}.weight"] for n in names if f"w.{n}.weight" in man]
    for unit, tag in ((16, "amx"), (4, "avx")):
        per = []
        for w in range(NW):
            b = 0
            for n, k in shp:
                a, z = split(n, unit, w)
                b += (z - a) * k * 2
            per.append(b)
        print(f"{ref.split('/')[-1]:22s} {tag}: mean {sum(per) / NW:,.0f} B  max {max(per):,.0f} B  min {min(per):,.0f} B  total {sum(per) / 2**20:.2f} MiB")
