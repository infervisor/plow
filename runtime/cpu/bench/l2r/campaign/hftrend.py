import json, os
for m in ("e2b.L4", "e4b.L5", "e2b.L0", "e4b.L0"):
    for c in (2048, 8192, 16384, 32768, 65536, 131072):
        f = f"/tmp/g4c/l2r/ref/{m}.c{c}/meta.json"
        if os.path.exists(f):
            j = json.load(open(f))
            print(m, c, "hf attn rel %.2e out %.2e | bf16 attn %.2e" % (j["hf_check"]["attn"]["rel_rms"], j["hf_check"]["out"]["rel_rms"], j["bf16_ref_err"]["attn"]["rel_rms"]))
