#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CHECK="$ROOT/perf-data/tools/check_build_matrix.py"
TMP=$(mktemp -d /tmp/plow-build-matrix.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

python3 - "$TMP" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
manifest = {
  "schema": 1, "arch": "gfx950", "shapes": {"decode_batch": 128},
  "lean": {"verified": True, "oracle": True},
  "tuning": {"tile_measured": 4, "tile_source": "measured"},
  "programs": [{"kind": "decode", "batch": b} for b in [1, 32, 64, 128]],
  "attention_policy": {"entries": [{
    "cell": {"hardware": "gfx950", "n_cu": 256, "decode_rung": 128,
             "kv_bucket": 8192, "shape": "mla-hd128"},
    "qualified": True, "selected": {"nsplit": 64, "source": "qualified"}}]},
  "backends": {"gfx950": {"requires": ["PLOW_K3=1", "PLOW_KDA_CHUNK=1"]}},
  "pairing": {"hash": "abc123"}}
(p / "build.json").write_text(json.dumps(manifest))
cell = {
  "name": "example-throughput", "artifact_kind": "devblob", "mode": "throughput",
  "manifest": "build.json",
  "requested": {
    "compiler": {"PLOW_DECODE_BATCH_LADDER": "1,32,64,128", "PLOW_TUNEDB": "/tuning"},
    "runtime": {"PLOW_DECODE_DEFER": "1"},
    "lean": {"verified": True, "oracle": True}, "tuning": {"measured": True},
    "decode_ladder": [1, 32, 64, 128],
    "attention": [{"hardware": "gfx950", "n_cu": 256, "decode_rung": 128,
                   "kv_bucket": 8192, "shape": "mla-hd128", "nsplit": 64,
                   "qualified": True}],
    "objects": {"launch_rows": 128, "markers": ["plow_kda_chunk_arm"]}},
  "observed": {
    "compiler": {"PLOW_DECODE_BATCH_LADDER": "1,32,64,128", "PLOW_TUNEDB": "/tuning"},
    "runtime": {"PLOW_DECODE_DEFER": "1"},
    "object_defines": ["PLOW_K3=1", "PLOW_KDA_CHUNK=1"],
    "object_markers": ["plow_kda_chunk_arm", "plow_gemv_mm_cap_8",
                       "plow_gemv_walk_1", "plow_xargmax_max_batch_128"],
    "object_pairing_hash": "abc123"}}
(p / "good.json").write_text(json.dumps({"schema": 1, "cells": [cell]}))

scheduled = {
  "name": "scheduled-latency", "artifact_kind": "scheduled", "mode": "latency",
  "requested": {"compiler": {"--counter-elim": True}, "runtime": {}, "counter_elim": True},
  "observed": {"compiler": {"--counter-elim": True}, "runtime": {}}}
(p / "scheduled.json").write_text(json.dumps({"schema": 1, "cells": [scheduled]}))

def bad(name, mutate):
    import copy
    c = copy.deepcopy(cell); mutate(c)
    (p / name).write_text(json.dumps({"schema": 1, "cells": [c]}))
bad("missing-knob.json", lambda c: c["observed"]["compiler"].pop("PLOW_TUNEDB"))
bad("wrong-rungs.json", lambda c: c["requested"].__setitem__("decode_ladder", [1, 32, 128]))
bad("wrong-nsplit.json", lambda c: c["requested"]["attention"][0].__setitem__("nsplit", 32))
bad("wrong-pair.json", lambda c: c["observed"].__setitem__("object_pairing_hash", "stale"))
bad("narrow-object.json", lambda c: c["observed"].__setitem__(
    "object_markers", ["plow_kda_chunk_arm", "plow_gemv_mm_cap_8"]))
bad("inapplicable.json", lambda c: c["requested"].__setitem__("counter_elim", True))
analytical = json.loads(json.dumps(manifest))
analytical["tuning"] = {"tile_measured": True, "tile_source": "Mixed-Analytical"}
(p / "analytical-build.json").write_text(json.dumps(analytical))
analytical_cell = json.loads(json.dumps(cell))
analytical_cell["manifest"] = "analytical-build.json"
(p / "analytical.json").write_text(json.dumps({"schema": 1, "cells": [analytical_cell]}))
PY

python3 "$CHECK" "$TMP/good.json" >/dev/null
python3 "$CHECK" "$TMP/scheduled.json" >/dev/null
for bad in missing-knob wrong-rungs wrong-nsplit wrong-pair narrow-object inapplicable analytical; do
  if python3 "$CHECK" "$TMP/$bad.json" >/dev/null 2>&1; then
    echo "FAIL: accepted $bad" >&2
    exit 1
  fi
done
echo "check_build_matrix selftest: PASS"
