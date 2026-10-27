import json, struct, glob, collections
for m in ("E2B", "E4B"):
    d = f"/tmp/models/google/gemma-4-{m}-it"
    tot = collections.Counter()
    for f in glob.glob(d + "/*.safetensors"):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k == "__metadata__":
                continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            if "language_model" not in k:
                tot["non-text"] += b
            elif ".layers." in k:
                tot["layers"] += b
            elif "embed_tokens_per_layer" in k:
                tot["ple_table"] += b
            elif "embed_tokens" in k:
                tot["embed(tied lm_head)"] += b
            else:
                tot["other_text:" + k.split("language_model.")[-1][:40]] += b
    print(m, {k: f"{v / 2**20:.0f} MiB" for k, v in tot.items() if v > 2**20})
