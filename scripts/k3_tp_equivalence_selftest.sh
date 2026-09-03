#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d /tmp/plow-k3-tp-equivalence-selftest.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"/{snap1,snap8,objects,checkpoint,asset1,asset8}
touch "$TMP"/objects/interp_decode.elf "$TMP"/asset{1,8}/{model.pkt,build.json}

python3 - "$TMP" <<'PY'
import json, os, struct, sys
root = sys.argv[1]
prompt, steps = [2, 3], 1

def bf16(values):
    return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16)
                    for value in values)

vectors = [[0.0, 3.0, 1.0], [0.0, 5.0, 2.0], [1.0, 2.0, 7.0]]
for tp in (1, 8):
    snap = os.path.join(root, f"snap{tp}")
    for tick, vector in enumerate(vectors):
        with open(os.path.join(snap, f"t{tick:05}_r1_act_logits.bin"), "wb") as output:
            output.write(bf16(vector))
    agreement = None if tp == 1 else {
        "ranks": 8, "sampled_token_every": 1,
        "counter_audit_every_dispatch": True, "prefill_completion_all_ranks": True,
    }
    report = {
        "schema": "plowrt.bench.v1", "vendor": "Some(Amd)",
        "num_gpus": tp, "parallel": "tp", "requests": 1, "completed": 1, "failed": 0,
        "asset_dir": os.path.join(root, f"asset{tp}"),
        "warmup_requests": 0, "prompt_tokens": len(prompt), "output_tokens": steps + 1,
        "scheduler": {"rejected": 0, "admit_shed": 0},
        "engine": {"batch_capacity": 1},
        "artifacts": {
            "packet": {"path": os.path.join(root, f"asset{tp}", "model.pkt"), "checksum": "pkt"},
            "build_manifest": {"path": os.path.join(root, f"asset{tp}", "build.json"), "checksum": "build"},
            "checkpoint": {"path": os.path.join(root, "checkpoint"), "layout_checksum": "ckpt"},
            "object_inventory": [{"path": os.path.join(root, "objects", "interp_decode.elf"), "checksum": "obj"}],
        },
        "diagnostics": {"supported": True, "complete": True, "overflowed": False,
                        "rank_agreement": agreement, "decode_selections": []},
        "token_audit": {"prompt_token_ids": [prompt], "output_token_ids": [[1, 2]]},
    }
    with open(os.path.join(root, f"report{tp}.json"), "w") as output:
        json.dump(report, output)
PY

compare=(python3 "$ROOT/scripts/k3_tp_equivalence_compare.py"
  --report1 "$TMP/report1.json" --snap1 "$TMP/snap1" --asset1 "$TMP/asset1" --packet1 "$TMP/asset1/model.pkt"
  --report8 "$TMP/report8.json" --snap8 "$TMP/snap8" --asset8 "$TMP/asset8" --packet8 "$TMP/asset8/model.pkt"
  --objects "$TMP/objects" --checkpoint "$TMP/checkpoint" --prompt 2,3
  --steps 1 --cos 0.9999 --layers 2 --vocab 3)
"${compare[@]}" >"$TMP/pass.log"
grep -q 'prefill.*ok' "$TMP/pass.log"
grep -q '000.*ok' "$TMP/pass.log"

mv "$TMP/snap8/t00002_r1_act_logits.bin" "$TMP/missing.bin"
if "${compare[@]}" >"$TMP/fail.log" 2>&1; then
  echo "FAIL: missing production snapshot passed" >&2
  exit 1
fi
grep -q 'snapshot tick 2 has 0 act.logits files' "$TMP/fail.log"

mv "$TMP/missing.bin" "$TMP/snap8/t00002_r1_act_logits.bin"
python3 - "$TMP/report8.json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as source:
    report = json.load(source)
report["artifacts"]["packet"]["checksum"] = ""
with open(path, "w") as output:
    json.dump(report, output)
PY
if "${compare[@]}" >"$TMP/checksum-fail.log" 2>&1; then
  echo "FAIL: missing artifact checksum passed" >&2
  exit 1
fi
grep -q 'missing packet checksum' "$TMP/checksum-fail.log"
echo "PASS: K3 TP comparator rejects missing snapshots and validates exact production evidence"
