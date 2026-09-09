#!/usr/bin/env python3
"""Assert a recipe describes the artifacts that were actually built.

The tree already records almost everything a recipe needs to state; what it has
never had is a check that the two agree. `build.json` carries
`emit_config.replay` — the knobs a rebuild must be told — and `build_defines.json`
carries the exact `-D` set each object compiled from, written by
`build_gfx942.sh` from the same `$ROWS` the compile loop uses "so the recorded -D
set cannot drift from the compiled one". This joins them to the recipe.

Requested-vs-observed, in the shape `perf-data/tools/check_build_matrix.py`
already uses: every knob the recipe REQUESTS must appear in what the build
OBSERVED, with the same value. A recipe whose `[emit].env` is edited by one knob
must fail.

  check_recipe.py recipes/infervisor/kimi-k3/<label>.toml \\
      --bundle build-amd/k3-assets --objects build-amd/k3-objects
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import plow_dist as pd

REPO = Path(__file__).resolve().parent.parent


class Invalid(pd.Fail):
    pass


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise Invalid(msg)


def exact_map(requested: dict, observed: dict, surface: str) -> None:
    """Every requested knob must be observed with the same value.

    One-directional on purpose: a build may record knobs the recipe does not
    mention (defaults it resolved), but a knob the recipe NAMES and the build
    did not honour is a recipe that describes something else.
    """
    for knob, value in requested.items():
        need(knob in observed, f"{surface}: recipe requests {knob} but the build recorded no such knob")
        need(
            str(observed[knob]) == str(value),
            f"{surface}: contradictory {knob} — recipe says {value!r}, build recorded "
            f"{observed[knob]!r}",
        )


def check_emit(recipe: dict, build_json: dict) -> list[str]:
    """`[emit].env` against `build.json`'s own knob record."""
    ec = build_json.get("emit_config")
    if not ec:
        return [
            "emit: build.json has no `emit_config` (written by a plowc that predates the "
            "knob record) — cannot verify the emit knobs"
        ]
    observed = dict(ec.get("replay") or {})
    # `unrecorded_env` names knobs read outside EmitConfig. They shaped the
    # packet just as much, so a recipe may state them and they must agree.
    observed.update(ec.get("unrecorded_env") or {})
    requested = (recipe.get("emit") or {}).get("env") or {}
    exact_map(requested, observed, "emit")

    # The other direction, as a report rather than a refusal: a knob the build
    # replayed but the recipe does not mention is a reproduction gap.
    missing = sorted(set(observed) - set(requested))
    return [f"emit: build replayed {k}={observed[k]!r}, recipe does not state it" for k in missing]


# Knobs whose effect lands under another define. `build_gfx942.sh` turns
# `PLOW_DECODE_BATCH` into the GEMV width cap `GVMM`, emitted as
# `-DPLOW_GEMV_MM=<n>`, so checking for the knob's own name would always miss.
INDIRECT = {"PLOW_DECODE_BATCH": "PLOW_GEMV_MM", "PLOW_DECODE_TIER": "PLOW_GEMV_MM"}


def check_objects(recipe: dict, defines: dict) -> list[str]:
    """`[objects].env` against the `-D` set the objects compiled from."""
    obj = recipe.get("objects") or {}
    requested = obj.get("env") or {}
    if not defines:
        return ["objects: no build_defines.json — cannot verify the -D set"]

    # `build_defines.json` is stem -> the exact `-D` string. A recipe states the
    # ENV the script was driven with, and the script expands env to defines, so
    # the check is that every requested env knob left a mark somewhere.
    blob = " ".join(defines.values())
    notes = []
    for knob, value in requested.items():
        # Build-driver settings, not compile axes: they shape how the build RUNS,
        # never what it compiles, so their absence from the -D set is correct.
        if knob in ("JOBS", "PLOW_BUILD_DIR", "PLOW_AUDIT_JOBS", "PLOW_ROWS_ONLY", "PLOW_DECODE_TIERS"):
            continue
        # Some knobs reach the object under a DIFFERENT define: the decode batch
        # width drives `GVMM`, which is emitted as `-DPLOW_GEMV_MM=<n>`, so
        # looking for its own name would always miss.
        name = INDIRECT.get(knob, knob)
        present = f"-D{name}=" in blob or f"-D{name} " in blob or blob.endswith(f"-D{name}")
        falsy = str(value) in ("0", "false", "off", "")

        if falsy:
            # A knob turned OFF removes its define rather than adding one — so
            # the check is that it is ABSENT, or present as 0. Skipping falsy
            # values instead would let a recipe flip a knob to 0 against a build
            # that had it on, which is precisely the drift this exists to catch.
            if present and f"-D{name}=0" not in blob:
                notes.append(
                    f"objects: recipe sets {knob}={value}, but the objects were compiled WITH "
                    f"-D{name}; the recipe and the build disagree"
                )
        elif not present and name not in blob:
            notes.append(
                f"objects: recipe sets {knob}={value} but no object's -D set mentions it; "
                f"either the knob does not reach a define or the objects are from another build"
            )
    return notes


def check_target(recipe: dict, build_json: dict, weights: dict) -> None:
    t = recipe.get("target") or {}
    if not t:
        raise Invalid("recipe has no [target]")
    if build_json.get("arch") and t.get("isa"):
        need(
            build_json["arch"] == t["isa"],
            f"target: recipe says isa={t['isa']}, build.json says arch={build_json['arch']}",
        )
    if build_json.get("n_cu") and t.get("units"):
        need(
            int(build_json["n_cu"]) == int(t["units"]),
            f"target: recipe says units={t['units']}, build.json says n_cu={build_json['n_cu']}",
        )
    if weights:
        if weights.get("num_gpus") and t.get("num_gpus"):
            need(
                int(weights["num_gpus"]) == int(t["num_gpus"]),
                f"target: recipe says num_gpus={t['num_gpus']}, weights.json says "
                f"{weights['num_gpus']}",
            )
        if weights.get("parallel") and t.get("parallel"):
            need(
                weights["parallel"] == t["parallel"],
                f"target: recipe says parallel={t['parallel']}, weights.json says "
                f"{weights['parallel']}",
            )


def check_artifacts(recipe: dict, roots: list[Path]) -> list[str]:
    """Every recorded digest must match a file actually present."""
    notes = []
    for name, want in (recipe.get("artifacts") or {}).items():
        need(
            isinstance(want, str),
            f"artifacts: {name} must map to a digest string. A filename contains dots, so its "
            f"key must be QUOTED in TOML — a bare `model.pkt = ...` parses as a nested table.",
        )
        found = None
        for root in roots:
            p = root / name
            if p.is_file():
                found = p
                break
        if found is None:
            notes.append(f"artifacts: {name} is recorded but not present in any given directory")
            continue
        got = pd.sha256_file(found)
        need(
            got == want,
            f"artifacts: {name} hashes to {got}, recipe records {want}",
        )
    return notes


def check_provenance(recipe: dict, strict: bool) -> list[str]:
    git = recipe.get("plow_git")
    if not git:
        raise Invalid("recipe has no `plow_git`")
    need(len(git) == 40, f"plow_git must be a full 40-hex commit, got {git!r}")
    r = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", git, "HEAD"],
        capture_output=True,
        check=False,
    )
    if r.returncode != 0:
        msg = (
            f"plow_git {git[:12]} is not an ancestor of HEAD — this recipe describes a build "
            f"from a commit this checkout does not contain"
        )
        if strict:
            raise Invalid(msg)
        return [msg]
    return []


def check(recipe_path: Path, bundle: Path | None, objects: Path | None, strict: bool) -> list[str]:
    recipe = pd.load_toml(recipe_path)
    notes: list[str] = []

    # The status/measured/refusal rules live in plow_dist so the publisher and
    # this checker cannot disagree about what may be called `validated`.
    pd.recipe_invariants(recipe)
    if recipe.get("status", "emits") == "refused":
        return check_provenance(recipe, strict)

    notes += check_provenance(recipe, strict)

    if bundle:
        bj = pd.load_json(bundle / "build.json") if (bundle / "build.json").exists() else {}
        wj = pd.load_json(bundle / "weights.json") if (bundle / "weights.json").exists() else {}
        check_target(recipe, bj, wj)
        notes += check_emit(recipe, bj)
    if objects:
        dp = objects / "build_defines.json"
        notes += check_objects(recipe, pd.load_json(dp) if dp.exists() else {})

    roots = [p for p in (bundle, objects) if p]
    if roots:
        notes += check_artifacts(recipe, roots)
    return notes


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recipe", nargs="*", help="recipe TOML files")
    ap.add_argument("--bundle", help="the compiled asset directory")
    ap.add_argument("--objects", help="the built object directory")
    ap.add_argument("--strict", action="store_true", help="promote reports to refusals")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        print("check_recipe selftest: PASS")
        return
    if not args.recipe:
        pd.die("give at least one recipe")

    failed = 0
    for rp in args.recipe:
        p = Path(rp)
        try:
            notes = check(
                p,
                Path(args.bundle) if args.bundle else None,
                Path(args.objects) if args.objects else None,
                args.strict,
            )
            print(f"{p}: OK")
            for n in notes:
                print(f"  note: {n}")
                if args.strict:
                    failed += 1
        except pd.Fail as e:
            print(f"{p}: FAIL\n  {e}")
            failed += 1
    if failed:
        pd.die(f"{failed} problem(s)")


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        bundle = d / "assets"
        objects = d / "obj"
        bundle.mkdir()
        objects.mkdir()

        (bundle / "build.json").write_text(
            json.dumps(
                {
                    "arch": "gfx942",
                    "n_cu": 304,
                    "emit_config": {
                        "replay": {"PLOW_FP8_KV": "1", "PLOW_MXFP4": "1"},
                        "unrecorded_env": {"PLOW_MLA_PF_V2": "1"},
                    },
                }
            )
        )
        (bundle / "weights.json").write_text(
            json.dumps({"network": "kimi-k3", "num_gpus": 8, "parallel": "tp"})
        )
        (objects / "build_defines.json").write_text(
            json.dumps({"interp_decode": "-DPLOW_ARCH_SUFFIX=gfx942 -DPLOW_GEMV_MM=16"})
        )
        (objects / "i.elf").write_bytes(b"an object")

        head = pd.git_commit(REPO)
        good = {
            "schema": "plow.recipe.v1",
            "namespace": "infervisor",
            "name": "kimi-k3",
            "label": "gfx942-mi325x-tp8",
            "status": "validated",
            "plow_git": head,
            "target": {"isa": "gfx942", "units": 304, "num_gpus": 8, "parallel": "tp"},
            "emit": {"env": {"PLOW_FP8_KV": "1", "PLOW_MLA_PF_V2": "1"}},
            "objects": {"env": {"PLOW_GEMV_MM": "16"}},
            "measured": {"tok_s": 131.162},
            "artifacts": {"i.elf": pd.sha256_file(objects / "i.elf")},
        }

        def write(doc) -> Path:
            import tomllib  # noqa: F401  (validate we can re-read what we wrote)

            p = d / "r.toml"
            p.write_text(to_toml(doc))
            return p

        p = write(good)
        notes = check(p, bundle, objects, strict=False)
        assert not any("contradictory" in n for n in notes), notes
        # PLOW_MXFP4 was replayed but not stated: reported, not fatal.
        assert any("PLOW_MXFP4" in n for n in notes), notes

        # THE test this file exists for: one edited knob must fail.
        bad = json.loads(json.dumps(good))
        bad["emit"]["env"]["PLOW_FP8_KV"] = "0"
        try:
            check(write(bad), bundle, objects, strict=False)
            raise AssertionError("accepted a contradicted emit knob")
        except pd.Fail as e:
            assert "contradictory PLOW_FP8_KV" in str(e), e

        # A knob the build never saw.
        bad = json.loads(json.dumps(good))
        bad["emit"]["env"]["PLOW_INVENTED"] = "1"
        try:
            check(write(bad), bundle, objects, strict=False)
            raise AssertionError("accepted an unknown emit knob")
        except pd.Fail as e:
            assert "no such knob" in str(e)

        # Target disagreements.
        for field, value in [("isa", "gfx950"), ("units", 256), ("num_gpus", 4)]:
            bad = json.loads(json.dumps(good))
            bad["target"][field] = value
            try:
                check(write(bad), bundle, objects, strict=False)
                raise AssertionError(f"accepted target {field}={value}")
            except pd.Fail:
                pass

        # A recorded digest that does not match the file.
        bad = json.loads(json.dumps(good))
        bad["artifacts"]["i.elf"] = "0" * 64
        try:
            check(write(bad), bundle, objects, strict=False)
            raise AssertionError("accepted a wrong artifact digest")
        except pd.Fail as e:
            assert "hashes to" in str(e)

        # validated without a measurement.
        bad = json.loads(json.dumps(good))
        del bad["measured"]
        try:
            check(write(bad), bundle, objects, strict=False)
            raise AssertionError("accepted validated with no measurement")
        except pd.Fail as e:
            assert "must carry `[measured] tok_s`" in str(e)

        # A refused recipe states what blocks it and carries no build.
        refused = {
            "schema": "plow.recipe.v1",
            "namespace": "infervisor",
            "name": "deepseek-v4",
            "label": "gfx942-mi300x-tp8",
            "status": "refused",
            "plow_git": head,
            "refusal": "crates/nn-graph/src/models/config/deepseek_v4.rs:261",
            "target": {"isa": "gfx942", "units": 304},
        }
        assert check(write(refused), None, None, strict=False) == []
        bad = dict(refused)
        del bad["refusal"]
        try:
            check(write(bad), None, None, strict=False)
            raise AssertionError("accepted a refused recipe with no reason")
        except pd.Fail:
            pass


def to_toml(doc: dict) -> str:
    """Minimal TOML writer for the self-test (python has no stdlib writer)."""

    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        return json.dumps(str(v))

    # Keys are always quoted: an artifact name like `model.pkt` is ONE key, and
    # a bare dotted key would parse as a nested table instead.
    def key(k):
        return json.dumps(str(k))

    out = []
    for k, v in doc.items():
        if not isinstance(v, dict):
            out.append(f"{key(k)} = {val(v)}")
    for k, v in doc.items():
        if isinstance(v, dict):
            out.append(f"\n[{key(k)}]")
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    inner = ", ".join(f"{key(a)} = {val(b)}" for a, b in v2.items())
                    out.append(f"{key(k2)} = {{ {inner} }}")
                else:
                    out.append(f"{key(k2)} = {val(v2)}")
    return "\n".join(out) + "\n"


main = pd.main_guard(run)

if __name__ == "__main__":
    raise SystemExit(main())
