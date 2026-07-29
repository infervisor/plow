#!/usr/bin/env bash
# Inspect build-amd assets: decode_batch, gv_mm_max, model.
for d in "$@"; do
  echo "=== $d ==="
  ls "$d" 2>/dev/null | tr '\n' ' '; echo
  python3 - "$d/build.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("  no build.json:",e); raise SystemExit
print("  decode_batch:",d.get('shapes',{}).get('decode_batch'),
      "gv_mm_max:",d.get('tuning',{}).get('gv_mm_max'),
      "hd:",d.get('shapes',{}).get('hd'),
      "feat:",{k:v for k,v in d.get('features',{}).items() if v})
PY
done
