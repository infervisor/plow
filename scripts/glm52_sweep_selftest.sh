#!/usr/bin/env bash
# glm52_sweep_selftest.sh — GPU-free scaffold test for the GLM-5.2 decode sweep pipeline.
# Feeds a CANNED SWEEP block (the tp_decode-format harness output, incl. a header + an "exceeds
# max_ctx" line that MUST be skipped) through the same awk parser glm52_sweep.sh uses, then through
# glm52_sweep_json.py, and asserts the emitted JSON matches the glm52-vllm-decode.json schema.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mktemp --suffix=.json)"; trap 'rm -f "$OUT"' EXIT

# canned harness output for TP=4 and TP=8 (median ms/tok), plus lines the parser must ignore.
gen() {
cat <<'EOF'
SWEEP (decode-only, 1 tok, median of 21): TP=%TP%
  ctx          ms/tok      tok/s
  1024          5.900      169.5
  4096          6.050      165.3
  8192          6.300      158.7
  131072      (exceeds pkt max_ctx 65536 — recompile pkt)
EOF
}
{ gen | sed 's/%TP%/4/'; gen | sed 's/%TP%/8/'; } \
  | awk '/^  *[0-9]+ +[0-9.]+ +[0-9.]+/ { tp=(NR<=8?4:8); print tp, $1, $2 }' \
  | sed 's/^0 //' > /dev/null 2>&1 || true

# Reproduce the driver's per-TP parse exactly (tp injected per block).
ROWS="$(mktemp)"; trap 'rm -f "$ROWS" "$OUT"' EXIT
for tp in 4 8; do
  gen | sed "s/%TP%/$tp/" \
    | awk -v tp="$tp" '/^  *[0-9]+ +[0-9.]+ +[0-9.]+/ { print tp, $1, $2 }' >> "$ROWS"
done

echo "parsed rows:"; cat "$ROWS"
python3 "$REPO/scripts/glm52_sweep_json.py" --out "$OUT" --version "selftest" < "$ROWS"

python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["model"] == "GLM-5.2-FP8" and d["engine"] == "plow" and d["precision"] == "fp8", d
r = d["results"]
assert len(r) == 6, f"want 6 rows (2 TP x 3 ctx, exceeds-line skipped), got {len(r)}"
assert {x["tp"] for x in r} == {4, 8} and {x["ctx"] for x in r} == {1024, 4096, 8192}, r
for x in r:
    assert set(x) >= {"tp", "ctx", "tpot_ms", "notes"}, x
    assert isinstance(x["tpot_ms"], float)
print("SELFTEST OK — schema matches glm52-vllm-decode.json; exceeds-ctx line correctly skipped")
PY
