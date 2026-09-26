import json, sys
b = json.load(open(sys.argv[1]))
print("l2_placement:", json.dumps(b.get("l2_placement"))[:600])
for p in b["programs"]:
    keep = {k: v for k, v in p.items() if not isinstance(v, (list, dict)) or k in ("segments",)}
    print(json.dumps(keep)[:300])
