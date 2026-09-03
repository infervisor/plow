#!/usr/bin/env python3
"""Fail-closed static validation for plow compiler/runtime build cells."""

import argparse
import json
import sys
from pathlib import Path


DEVBLOB_ONLY = ("lean", "tuning", "decode_ladder", "attention", "objects")
SCHEDULED_ONLY = ("counter_elim",)
COMMON = ("compiler", "runtime")


class Invalid(Exception):
    pass


def need(condition, message):
    if not condition:
        raise Invalid(message)


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Invalid(f"{path}: cannot read JSON: {exc}") from exc


def exact_map(cell, surface):
    requested = cell.get("requested", {}).get(surface, {})
    observed = cell.get("observed", {}).get(surface, {})
    need(isinstance(requested, dict), f"requested.{surface} must be an object")
    need(isinstance(observed, dict), f"observed.{surface} must be an object")
    for knob, value in requested.items():
        need(knob in observed, f"missing {surface} knob {knob}")
        need(observed[knob] == value,
             f"contradictory {surface} knob {knob}: requested {value!r}, observed {observed[knob]!r}")


def decode_rungs(manifest):
    programs = manifest.get("programs")
    need(isinstance(programs, list), "build manifest has no programs array")
    rows = []
    for program in programs:
        if program.get("kind") == "decode":
            need(isinstance(program.get("batch"), int), "decode program has no integer batch")
            rows.append(program["batch"])
    need(rows, "build manifest has no decode programs")
    return rows


def attention_key(entry):
    cell = entry.get("cell", {})
    return tuple(cell.get(k) for k in
                 ("hardware", "n_cu", "decode_rung", "kv_bucket", "shape"))


def check_devblob(cell, manifest):
    requested = cell.get("requested", {})
    need(manifest.get("schema") == 1, "unsupported or missing devblob manifest schema")

    if "lean" in requested:
        spec = requested["lean"]
        need(isinstance(spec, dict) and spec and not (set(spec) - {"verified", "oracle"}),
             "requested.lean must contain only verified/oracle booleans")
        got = manifest.get("lean", {})
        for field in ("verified", "oracle"):
            if field in spec:
                need(isinstance(spec[field], bool), f"requested.lean.{field} must be boolean")
                need(got.get(field) is spec[field],
                     f"Lean {field} requested {spec[field]!r}, manifest has {got.get(field)!r}")

    if "tuning" in requested:
        spec = requested["tuning"]
        need(isinstance(spec, dict) and not (set(spec) - {"measured", "source"}),
             "requested.tuning may contain only measured/source")
        if "measured" in spec:
            need(isinstance(spec["measured"], bool), "requested.tuning.measured must be boolean")
        if "source" in spec:
            need(isinstance(spec["source"], str) and spec["source"],
                 "requested.tuning.source must be a nonempty string")
        got = manifest.get("tuning", {})
        if spec.get("measured"):
            measured = got.get("tile_measured")
            source = got.get("tile_source")
            need(isinstance(measured, int) and not isinstance(measured, bool) and measured > 0,
                 "measured tuning requested but tile_measured is not positive")
            need(isinstance(source, str) and source and "analytical" not in source.casefold(),
                 "measured tuning requested but tile_source is analytical or empty")
        if "source" in spec:
            need(got.get("tile_source") == spec["source"],
                 f"tuning source requested {spec['source']!r}, manifest has {got.get('tile_source')!r}")

    if "decode_ladder" in requested:
        want = requested["decode_ladder"]
        need(isinstance(want, list) and want and
             all(isinstance(row, int) and not isinstance(row, bool) and row > 0 for row in want) and
             want == sorted(set(want)),
             "requested.decode_ladder must be a nonempty strictly ascending integer list")
        got = decode_rungs(manifest)
        need(got == want, f"decode ladder requested {want}, manifest programs have {got}")
        need(manifest.get("shapes", {}).get("decode_batch") == want[-1],
             "shapes.decode_batch does not equal the widest requested decode rung")

    if "attention" in requested:
        need(isinstance(requested["attention"], list),
             "requested.attention must be an array")
        entries = manifest.get("attention_policy", {}).get("entries")
        need(isinstance(entries, list), "build manifest has no attention_policy.entries")
        by_key = {attention_key(entry): entry for entry in entries}
        for spec in requested["attention"]:
            need(isinstance(spec, dict), "each requested attention cell must be an object")
            fields = {"hardware", "n_cu", "decode_rung", "kv_bucket", "shape", "nsplit", "qualified"}
            need(not (set(spec) - fields),
                 f"unknown requested attention fields: {sorted(set(spec) - fields)}")
            need(isinstance(spec.get("nsplit"), int) and not isinstance(spec.get("nsplit"), bool)
                 and spec["nsplit"] > 0, "attention nsplit must be a positive integer")
            if "qualified" in spec:
                need(isinstance(spec["qualified"], bool), "attention qualified must be boolean")
            key = tuple(spec.get(k) for k in
                        ("hardware", "n_cu", "decode_rung", "kv_bucket", "shape"))
            need(None not in key, f"attention request has an incomplete exact cell: {spec!r}")
            need(key in by_key, f"attention exact cell is absent: {key!r}")
            selected = by_key[key].get("selected", {})
            need(selected.get("nsplit") == spec.get("nsplit"),
                 f"attention cell {key!r} requested nsplit={spec.get('nsplit')}, has {selected.get('nsplit')}")
            if spec.get("qualified"):
                need(by_key[key].get("qualified") is True,
                     f"attention cell {key!r} is not qualified")

    if "objects" in requested:
        spec = requested["objects"]
        need(isinstance(spec, dict), "requested.objects must be an object")
        object_fields = {"markers", "launch_rows"}
        need(not (set(spec) - object_fields),
             f"unknown requested.objects fields: {sorted(set(spec) - object_fields)}")
        observed = cell.get("observed", {})
        arch = manifest.get("arch")
        backend = manifest.get("backends", {}).get(arch)
        need(isinstance(backend, dict), f"manifest has no backend entry for arch {arch!r}")
        defines = observed.get("object_defines")
        markers = observed.get("object_markers")
        need(isinstance(defines, list), "observed.object_defines must be an array")
        need(isinstance(markers, list), "observed.object_markers must be an array")
        need(isinstance(spec.get("markers", []), list), "requested.objects.markers must be an array")
        required = backend.get("requires", [])
        need(isinstance(required, list), "manifest backend requires must be an array")
        missing = sorted(set(required) - set(defines))
        need(not missing, f"object is missing manifest-required defines: {missing}")
        missing = sorted(set(spec.get("markers", [])) - set(markers))
        need(not missing, f"object is missing requested markers: {missing}")
        if "launch_rows" in spec:
            rows = spec["launch_rows"]
            need(isinstance(rows, int) and not isinstance(rows, bool) and rows > 0,
                 "requested.objects.launch_rows must be a positive integer")
            if "decode_ladder" in requested:
                need(rows in requested["decode_ladder"],
                     f"object launch_rows={rows} is absent from the requested decode ladder")
            caps = []
            for marker in markers:
                if marker.startswith("plow_gemv_mm_cap_"):
                    try:
                        caps.append(int(marker.removeprefix("plow_gemv_mm_cap_")))
                    except ValueError:
                        raise Invalid(f"malformed GEMV capacity marker {marker!r}")
            if rows > 1:
                need("plow_gemv_walk_1" in markers or (caps and max(caps) >= rows),
                     f"object cannot cover {rows} launch rows: no walk marker or sufficient GEMV capacity")
            if rows > 32:
                need("plow_xargmax_max_batch_128" in markers,
                     f"object cannot cover {rows} launch rows: B128 argmax marker is absent")
        pairing = manifest.get("pairing", {}).get("hash")
        need(pairing not in (None, ""), "manifest pairing.hash is missing")
        need(observed.get("object_pairing_hash") == pairing,
             f"object pairing hash {observed.get('object_pairing_hash')!r} does not match manifest {pairing!r}")


def check_scheduled(cell, manifest):
    requested = cell.get("requested", {})
    if "counter_elim" in requested:
        want = requested["counter_elim"]
        need(isinstance(want, bool), "requested.counter_elim must be boolean")
        observed = cell.get("observed", {}).get("compiler", {})
        need(observed.get("--counter-elim") is want,
             f"counter elimination requested {want}, observed --counter-elim={observed.get('--counter-elim')!r}")
    if "lean" in requested:
        need(manifest is not None,
             "scheduled Lean validation requires the emitted weights.json manifest")
        need(manifest.get("lean_verified") is requested["lean"].get("verified"),
             "scheduled Lean verification does not match weights.json")


def check_cell(cell, base):
    name = cell.get("name")
    kind = cell.get("artifact_kind")
    need(isinstance(name, str) and name, "cell name is required")
    need(kind in ("devblob", "scheduled"), f"{name}: artifact_kind must be devblob or scheduled")
    need(cell.get("mode") in ("latency", "throughput"),
         f"{name}: mode must be latency or throughput")
    requested = cell.get("requested", {})
    need(isinstance(requested, dict), f"{name}: requested must be an object")
    allowed = set(COMMON + DEVBLOB_ONLY + SCHEDULED_ONLY)
    unknown = sorted(set(requested) - allowed)
    need(not unknown, f"{name}: unknown requested knobs: {unknown}")
    invalid = SCHEDULED_ONLY if kind == "devblob" else tuple(k for k in DEVBLOB_ONLY if k != "lean")
    bad = sorted(set(requested).intersection(invalid))
    need(not bad, f"{name}: requested knobs are inapplicable to {kind}: {bad}")
    exact_map(cell, "compiler")
    exact_map(cell, "runtime")
    manifest = None
    if "manifest" in cell:
        path = Path(cell["manifest"])
        manifest = load(path if path.is_absolute() else base / path)
    if kind == "devblob":
        need(manifest is not None, f"{name}: devblob validation requires manifest")
        check_devblob(cell, manifest)
    else:
        check_scheduled(cell, manifest)
    applicable = DEVBLOB_ONLY if kind == "devblob" else SCHEDULED_ONLY + ("lean",)
    inapplicable = SCHEDULED_ONLY if kind == "devblob" else tuple(k for k in DEVBLOB_ONLY if k != "lean")
    return name, kind, applicable, inapplicable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", help="build-matrix request/evidence JSON")
    args = parser.parse_args()
    try:
        matrix_path = Path(args.matrix)
        matrix = load(matrix_path)
        need(matrix.get("schema") == 1, "unsupported or missing matrix schema")
        cells = matrix.get("cells")
        need(isinstance(cells, list) and cells, "matrix cells must be a nonempty array")
        names = set()
        reports = []
        for cell in cells:
            report = check_cell(cell, matrix_path.resolve().parent)
            need(report[0] not in names, f"duplicate cell name {report[0]!r}")
            names.add(report[0])
            reports.append(report)
        for name, kind, applicable, inapplicable in reports:
            print(f"PASS {name} ({kind})")
            print(f"  applicable: {', '.join(applicable)}")
            print(f"  inapplicable: {', '.join(inapplicable)}")
        return 0
    except (Invalid, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
