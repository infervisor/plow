import json, sys
b = json.load(open(sys.argv[1]))
print("top keys:", list(b.keys()))
print("knobs verdict:", json.dumps(b.get("knobs", {}).get("K", b.get("knobs", {}).get("verdict")))[:300])
print("backends.nvcc:", json.dumps(b.get("backends", {}).get("nvcc")))
print("tuning:", json.dumps(b.get("tuning"))[:400])
w = json.dumps(b)
for k in ["lm_head", "embed_tokens"]:
    print(k, "mentions:", w.count(k))
