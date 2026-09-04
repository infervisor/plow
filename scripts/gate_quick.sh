#!/usr/bin/env bash
# One B=4 production-mux gate with bounded token and dispatch diagnostics.
set -euo pipefail
# Repo root, derived rather than hardcoded: this was an absolute path into a
# `.claude/worktrees/` directory, which is gitignored and belongs to an agent worktree
# that no longer exists — the script was dead on arrival for every other reader.
# Override with PLOW_REPO to point at a different checkout.
WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT="${CKPT:-/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475}"
B="${B:-4}"
ASSETS="${ASSETS:-$WT/build-amd/g31b-db$B}"
HSACO="${HSACO:-$WT/build-amd/hsaco-b$B}"
PROMPT="${P:-2,106,1645}"
OUT="${GATE_QUICK_OUT:-/tmp/gate-quick-b$B.json}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
NIX="${PLOW_NIX_BIN:-nix}"

[[ "$B" =~ ^[1-9][0-9]*$ ]] || { echo "B must be positive" >&2; exit 2; }
[[ "$PROMPT" != *';'* ]] || { echo "P is one exact prompt; requests are replicated by bench" >&2; exit 2; }

printf -v PLOW_CMD \
  'exec %q --rt-checkpoint %q --rt-hsaco %q bench --assets %q --prompt-ids %q --concurrency %q --requests %q --warmup-requests 0 --output-len 4 --token-audit --engine-diagnostics >%q' \
  "$BIN" "$CKPT" "$HSACO" "$ASSETS" "$PROMPT" "$B" "$B" "$OUT"
"$NIX" develop "$WT" --command bash -c "$PLOW_CMD"

python3 - "$OUT" "$B" "$PROMPT" <<'PY'
import json, sys

path, expected = sys.argv[1], int(sys.argv[2])
prompt = [int(token) for token in sys.argv[3].split(",")]
with open(path) as source:
    report = json.load(source)
if report.get("schema") != "plowrt.bench.v1":
    raise SystemExit("FAIL: unsupported bench report")
if (report.get("requests"), report.get("completed"), report.get("failed")) != (expected, expected, 0):
    raise SystemExit("FAIL: incomplete production-mux requests")
audit = report.get("token_audit") or {}
prompts = audit.get("prompt_token_ids")
rows = audit.get("output_token_ids")
if prompts != [prompt] * expected:
    raise SystemExit("FAIL: token audit did not preserve the exact replicated prompt")
if not isinstance(rows, list) or len(rows) != expected or any(len(row) != 4 for row in rows):
    raise SystemExit("FAIL: missing bounded token-audit rows")
if not any(token != 0 for row in rows for token in row):
    raise SystemExit("FAIL: all generated tokens are zero")
if any(row != rows[0] for row in rows[1:]):
    raise SystemExit("FAIL: identical prompts produced different token streams")
engine = report.get("engine") or {}
if engine.get("batch_capacity") != expected:
    raise SystemExit("FAIL: loaded engine batch capacity does not match B")
diagnostics = report.get("diagnostics") or {}
if diagnostics.get("supported") is not True or diagnostics.get("complete") is not True or diagnostics.get("overflowed") is not False:
    raise SystemExit("FAIL: missing or partial engine diagnostics")
if not any(row.get("occupied_rows") == expected and row.get("bucket", 0) >= expected for row in diagnostics.get("decode_selections", [])):
    raise SystemExit("FAIL: diagnostics contain no B-wide decode dispatch")
print(f"PASS: {expected} production-mux slots produced one identical nonzero stream")
print(rows[0])
PY
