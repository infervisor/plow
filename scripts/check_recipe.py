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
import re
import subprocess
import sys
from pathlib import Path

import recipe_common as rc

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "campaign"))
import rung_env  # noqa: E402
import spec_env  # noqa: E402


class Invalid(rc.Fail):
    pass


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise Invalid(msg)


def same_value(requested: object, observed: object) -> bool:
    """A `{out}`, `{repo}` or `{hf:…}` placeholder names a path on the build
    host, which the knob record holds expanded: match on the part the recipe
    spells after it."""
    want, got = str(requested), str(observed)
    if want == got:
        return True
    m = re.match(r"\{(out|repo|hf:[^}]+)\}(.*)", want)
    return bool(m) and got.endswith(m.group(2))


def exact_map(requested: dict, observed: dict, surface: str) -> None:
    """Every requested knob must be observed with the same value.

    One-directional on purpose: a build may record knobs the recipe does not
    mention (defaults it resolved), but a knob the recipe NAMES and the build
    did not honour is a recipe that describes something else.
    """
    for knob, value in requested.items():
        need(knob in observed, f"{surface}: recipe requests {knob} but the build recorded no such knob")
        need(
            same_value(value, observed[knob]),
            f"{surface}: contradictory {knob} — recipe says {value!r}, build recorded "
            f"{observed[knob]!r}",
        )


def recipe_rung_env(recipe: dict) -> dict:
    """`[emit.rung_env]`, refused unless every override names a rung-eligible knob."""
    src = rung_env.recipe_source_errors(recipe)
    need(not src, "emit: " + "; ".join(src))
    try:
        over = rung_env.from_recipe(recipe)
    except rung_env.RungEnvError as e:
        raise Invalid(f"emit: {e}")
    errs = rung_env.check(over)
    need(not errs, "emit: " + "; ".join(errs))
    return over


def check_rung_env(over: dict, build_json: dict, observed: dict) -> None:
    """`[emit.rung_env]` against the build's replayed string and its per-program resolution."""
    try:
        built = rung_env.parse(observed.get(rung_env.ENV))
    except rung_env.RungEnvError as e:
        raise Invalid(f"emit: build replayed a malformed {rung_env.ENV}: {e}")
    need(
        built == over,
        f"emit: contradictory {rung_env.ENV} — recipe says {rung_env.render(over)!r}, build "
        f"replayed {rung_env.render(built)!r}",
    )
    if not over:
        return
    programs = (build_json.get("rung_knobs") or {}).get("programs")
    need(programs is not None, "emit: build.json has no `rung_knobs` (a plowc that predates per-rung overrides)")
    ids, _ = rung_env.registry()
    for (phase, rows), knobs in sorted(over.items()):
        hit = [p for p in programs if p.get("phase") == phase and p.get("rows") == rows]
        need(hit, f"emit: [emit.rung_env.{phase}.{rows}] matches no program the build emitted")
        for env, value in knobs.items():
            want = rung_env.boolish(value)
            for p in hit:
                got = (p.get("knobs") or {}).get(ids[env]) or {}
                need(
                    got.get("source") == "rung" and (want is None or got.get("value") == want),
                    f"emit: {phase}:{rows} {env}={value!r} but program {p.get('program')} resolved "
                    f"{got.get('value')!r} from {got.get('source')!r}",
                )


INVENTORY_RS = REPO / "crates/plowrt/src/inventory.rs"
UNMEASURED = re.compile(r"not (yet )?measured|unmeasured|\btodo\b|\btbd\b", re.I)


def check_capabilities(recipe: dict) -> None:
    """`[capabilities]` waives `plowrt inventory` items by name, each with a reason (doctor [8])."""
    caps = recipe.get("capabilities") or {}
    if not caps:
        return
    items = set(re.findall(r'item\(\s*"([a-z_]+)"', INVENTORY_RS.read_text()))
    for name, reason in sorted(caps.items()):
        need(name in items, f"capabilities: {name!r} is not a plowrt inventory item ({', '.join(sorted(items))})")
        need(isinstance(reason, str) and reason.strip(), f"capabilities: {name} needs the measured reason")
        need(not UNMEASURED.search(reason), f"capabilities: {name}: an unmeasured route is not a waiver ({reason!r})")


def check_rung_choices(recipe: dict) -> None:
    """A `validated` recipe states a rung-eligible knob (packet-wide or in `[emit.rung_env]`) only
    with `[rung_choices.<KNOB>]`: `label = "qualified"` and the per-rung measurement as `evidence`
    (docs/bringup/agent-tools.md "Per-rung gate and recipe selection")."""
    if recipe.get("status") != "validated":
        return
    _, rungs = rung_env.registry()
    stated = set()
    for table in ((recipe.get("emit") or {}).get("env"), (recipe.get("emit_roles") or {}).get("env")):
        stated |= set(table or {}) & set(rungs)
    for knobs in rung_env.from_recipe(recipe).values():
        stated |= set(knobs)
    choices = recipe.get("rung_choices") or {}
    for env in sorted(stated):
        c = choices.get(env)
        need(isinstance(c, dict), f"rung_choices: {env} is a per-rung choice; a validated recipe needs [rung_choices.{env}]")
        need(c.get("label") == "qualified", f"rung_choices: {env} label {c.get('label')!r}, a validated recipe needs 'qualified'")
        ev = c.get("evidence")
        need(
            isinstance(ev, list) and ev and all(isinstance(e, str) and e.strip() for e in ev),
            f"rung_choices: {env} needs `evidence`, a non-empty list of the per-rung measurements",
        )


# Emit env keys that locate a tool on the build host rather than shape the
# packet; a recipe states them for the build command, the checker ignores them.
MACHINE_PATH_KNOBS = ("PLOW_VERIFY_BIN",)
# nvcc's own environment: `[emit.env]` may state it (the -D set the emitted
# kernels are compiled with). plowc's knob record does not carry it; the
# campaign records the environment it ran plowc with in `build-record.json`
# under `compilation.emit_env`, which is what it is checked against.
TOOLCHAIN_ENV_KNOBS = ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS")


def check_toolchain_env(requested: dict, record: dict | None) -> list[str]:
    """The nvcc environment `[emit.env]` states, against the campaign's build
    record: verified when the record is there, reported when it is not."""
    notes = []
    recorded = ((record or {}).get("compilation") or {}).get("emit_env")
    for knob in TOOLCHAIN_ENV_KNOBS:
        if knob not in requested:
            continue
        if recorded is None:
            notes.append(f"emit: {knob} is not in the knob record and no build-record.json is beside the bundle — unverified")
            continue
        need(knob in recorded, f"emit: recipe requests {knob} but the build ran plowc with no such variable")
        need(
            str(recorded[knob]) == str(requested[knob]),
            f"emit: contradictory {knob} — recipe says {requested[knob]!r}, build recorded {recorded[knob]!r}",
        )
    return notes


def check_emit(recipe: dict, build_json: dict, record: dict | None = None) -> list[str]:
    """`[emit].env` against `build.json`'s own knob record (and the nvcc
    environment against the campaign's build record)."""
    rungs = recipe_rung_env(recipe)
    ec = build_json.get("emit_config")
    need(ec or not rungs, "emit: the recipe states [emit.rung_env] but build.json has no `emit_config` to verify it")
    if not ec:
        return [
            "emit: build.json has no `emit_config` (written by a plowc that predates the "
            "knob record) — cannot verify the emit knobs"
        ]
    observed = dict(ec.get("replay") or {})
    # `unrecorded_env` names knobs read outside EmitConfig. They shaped the
    # packet just as much, so a recipe may state them and they must agree.
    observed.update(ec.get("unrecorded_env") or {})
    # `[spec]` derives its knobs; a drafter dir stated through `{env:...}` is a host path the
    # shard pin already identifies, so only its presence is checked.
    derived = spec_env.emit_env(recipe)
    if "{env:" in derived.get("PLOW_SPEC_DRAFT_DIR", ""):
        need("PLOW_SPEC_DRAFT_DIR" in observed, "emit: [spec] builds a drafter but the build recorded no PLOW_SPEC_DRAFT_DIR")
        observed = {k: v for k, v in observed.items() if k != "PLOW_SPEC_DRAFT_DIR"}
        derived.pop("PLOW_SPEC_DRAFT_DIR")
    requested = {**derived, **((recipe.get("emit") or {}).get("env") or {})}
    # Machine paths are not decisions: `PLOW_VERIFY_BIN` names where the Lean
    # verifier lives on the build host and the knob record never carries it
    # (the receipts beside the packet are what prove the verification ran).
    for machine in MACHINE_PATH_KNOBS:
        requested.pop(machine, None)
        observed.pop(machine, None)
    notes_tool = check_toolchain_env(requested, record)
    for knob in TOOLCHAIN_ENV_KNOBS:
        requested.pop(knob, None)
        observed.pop(knob, None)
    exact_map(requested, observed, "emit")
    check_rung_env(rungs, build_json, observed)

    # The other direction, as a report rather than a refusal: a knob the build
    # replayed but the recipe does not mention is a reproduction gap.
    missing = sorted(set(observed) - set(requested) - {rung_env.ENV})
    return notes_tool + [f"emit: build replayed {k}={observed[k]!r}, recipe does not state it" for k in missing]


# Knobs whose effect lands under another define. `build_gfx942.sh` turns
# `PLOW_DECODE_BATCH` into the GEMV width cap `GVMM`, emitted as
# `-DPLOW_GEMV_MM=<n>`, so checking for the knob's own name would always miss.
# `PLOW_K3_DECODE_GROUPED=1` puts the grouped A4W4 body on the K3 decode rows
# (`$AX_K3_DECODE_A4W4`), whose first define is `-DPLOW_MOE_PF_A4W4=1`.
INDIRECT = {
    "PLOW_DECODE_BATCH": "PLOW_GEMV_MM",
    "PLOW_DECODE_TIER": "PLOW_GEMV_MM",
    "PLOW_K3_DECODE_GROUPED": "PLOW_MOE_PF_A4W4",
}


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


def check_lowrung_tiers(recipe: dict) -> list[str]:
    """Explicit `[[objects.lowrung]]` rungs and an UNSET `PLOW_DECODE_TIERS` collide.

    build_gfx942.sh treats an unset PLOW_DECODE_TIERS as "auto": it builds lowrung{1,2,4,8}
    from the wide object's -D set, and the recipe's explicit rungs then rebuild the same
    directories with their own defines — last writer wins, and build_defines.json records
    whichever ran last. A recipe that drives its own rungs must say so with an empty value.
    """
    obj = recipe.get("objects") or {}
    if not obj.get("lowrung"):
        return []
    if "PLOW_DECODE_TIERS" in (obj.get("env") or {}):
        return []
    return [
        "objects: [[objects.lowrung]] entries with PLOW_DECODE_TIERS unset in [objects].env; "
        "build_gfx942.sh auto-builds lowrung{1,2,4,8} first and the explicit rungs overwrite "
        'them. Set PLOW_DECODE_TIERS = "" (drive the rungs yourself) or list the tiers.'
    ]


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
        got = rc.sha256_file(found)
        need(
            got == want,
            f"artifacts: {name} hashes to {got}, recipe records {want}",
        )
    return notes


def check_provenance(recipe: dict, strict: bool, verified: bool = False) -> list[str]:
    """`plow_git` is a record. A commit this checkout lacks refuses under `strict` only when no
    content evidence stands in for it: with `verified` (every recorded artifact present and
    matching its sha256) the bytes, not the commit, qualify the recipe."""
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
        if verified:
            return [f"info: {msg}; its artifacts match by sha256, so the commit stays a record"]
        if strict:
            raise Invalid(msg)
        return [msg]
    return []


def check(recipe_path: Path, bundle: Path | None, objects: Path | None, strict: bool) -> list[str]:
    recipe = rc.load_toml(recipe_path)
    notes: list[str] = []

    # The status/measured/refusal rules live in recipe_common so the renderer and
    # this checker cannot disagree about what may be called `validated`.
    rc.recipe_invariants(recipe)
    if recipe.get("status", "emits") == "refused":
        return check_provenance(recipe, strict)

    notes += check_lowrung_tiers(recipe)
    recipe_rung_env(recipe)
    check_rung_choices(recipe)
    check_capabilities(recipe)
    spec_errors = spec_env.errors(recipe)
    need(not spec_errors, "; ".join(spec_errors))

    if bundle:
        bj = rc.load_json(bundle / "build.json") if (bundle / "build.json").exists() else {}
        wj = rc.load_json(bundle / "weights.json") if (bundle / "weights.json").exists() else {}
        rec_path = bundle.parent / "build-record.json"
        rec = rc.load_json(rec_path) if rec_path.exists() else None
        check_target(recipe, bj, wj)
        notes += check_emit(recipe, bj, rec)
    if objects:
        dp = objects / "build_defines.json"
        notes += check_objects(recipe, rc.load_json(dp) if dp.exists() else {})

    roots = [p for p in (bundle, objects) if p]
    artifact_notes = check_artifacts(recipe, roots) if roots else []
    verified = bool(roots and recipe.get("artifacts") and not artifact_notes)
    return check_provenance(recipe, strict, verified) + notes + artifact_notes


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
        rc.die("give at least one recipe")

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
                if args.strict and not n.startswith("info:"):
                    failed += 1
        except rc.Fail as e:
            print(f"{p}: FAIL\n  {e}")
            failed += 1
    if failed:
        rc.die(f"{failed} problem(s)")


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
            json.dumps(
                {
                    "interp_decode": "-DPLOW_ARCH_SUFFIX=gfx942 -DPLOW_GEMV_MM=16",
                    "interp_decode_k3": "-DPLOW_ARCH_SUFFIX=gfx942 -DPLOW_GEMV_MM=16 -DPLOW_MOE_PF_A4W4=1",
                }
            )
        )
        (objects / "i.elf").write_bytes(b"an object")

        head = rc.git_commit(REPO)
        good = {
            "schema": "plow.recipe.v1",
            "namespace": "infervisor",
            "name": "kimi-k3",
            "label": "gfx942-mi325x-tp8",
            "status": "validated",
            "plow_git": head,
            "target": {"isa": "gfx942", "units": 304, "num_gpus": 8, "parallel": "tp"},
            "emit": {"env": {"PLOW_FP8_KV": "1", "PLOW_MLA_PF_V2": "1"}},
            # `PLOW_K3_DECODE_GROUPED` reaches the object as -DPLOW_MOE_PF_A4W4 (INDIRECT).
            "objects": {"env": {"PLOW_GEMV_MM": "16", "PLOW_K3_DECODE_GROUPED": "1"}},
            "measured": {"tok_s": 131.162},
            "artifacts": {"i.elf": rc.sha256_file(objects / "i.elf")},
        }

        def write(doc) -> Path:
            import tomllib  # noqa: F401  (validate we can re-read what we wrote)

            p = d / "r.toml"
            p.write_text(to_toml(doc))
            return p

        p = write(good)
        notes = check(p, bundle, objects, strict=False)
        assert not any("contradictory" in n for n in notes), notes
        # A build-host path placeholder matches the path the record expanded it to.
        assert same_value("{out}/prep/s3gen", "/nvme/build/chatterbox-h100/prep/s3gen")
        assert same_value("{hf:ResembleAI/chatterbox}/t3", "/hf/hub/models--x/snapshots/abc/t3")
        assert not same_value("{out}/prep/s3gen", "/nvme/build/other/prep/codec")
        assert not same_value("1", "0")
        # PLOW_MXFP4 was replayed but not stated: reported, not fatal.
        assert any("PLOW_MXFP4" in n for n in notes), notes

        # THE test this file exists for: one edited knob must fail.
        bad = json.loads(json.dumps(good))
        bad["emit"]["env"]["PLOW_FP8_KV"] = "0"
        try:
            check(write(bad), bundle, objects, strict=False)
            raise AssertionError("accepted a contradicted emit knob")
        except rc.Fail as e:
            assert "contradictory PLOW_FP8_KV" in str(e), e

        # Explicit rungs with PLOW_DECODE_TIERS unset would be overwritten by the
        # script's auto-built tiers: reported. Stating the knob (even empty) is fine.
        rung = json.loads(json.dumps(good))
        rung["objects"]["lowrung"] = [{"max": 1, "env": {"PLOW_DECODE_BATCH": "1"}}]
        notes = check(write(rung), bundle, objects, strict=False)
        assert any("PLOW_DECODE_TIERS unset" in n for n in notes), notes
        rung["objects"]["env"]["PLOW_DECODE_TIERS"] = ""
        notes = check(write(rung), bundle, objects, strict=False)
        assert not any("PLOW_DECODE_TIERS unset" in n for n in notes), notes

        # An objects knob that reaches no define is reported (it is what
        # `scripts/tests/test_recipe_objects_env.py` turns into a failure).
        bad = json.loads(json.dumps(good))
        bad["objects"]["env"]["PLOW_INVENTED"] = "1"
        notes = check(write(bad), bundle, objects, strict=False)
        assert any("PLOW_INVENTED" in n and "no object's -D set" in n for n in notes), notes

        # A knob the build never saw.
        bad = json.loads(json.dumps(good))
        bad["emit"]["env"]["PLOW_INVENTED"] = "1"
        try:
            check(write(bad), bundle, objects, strict=False)
            raise AssertionError("accepted an unknown emit knob")
        except rc.Fail as e:
            assert "no such knob" in str(e)

        # [spec]: derived knobs must match the build record; a malformed table fails.
        spec = json.loads(json.dumps(good))
        spec["spec"] = {"kind": "gemma4_assistant", "draft_dir": "{env:DRAFT}", "draft_revision": "r1",
                        "shard_sha256": "b" * 64, "k_max": 7, "rows": [1, 2, 4, 8, 16],
                        "emit": False, "serve": False}
        assert not any("contradictory" in n for n in check(write(spec), bundle, objects, strict=False))
        spec["spec"]["emit"] = True
        try:
            check(write(spec), bundle, objects, strict=False)
            raise AssertionError("accepted [spec] knobs the build did not record")
        except rc.Fail as e:
            assert "no such knob" in str(e) or "no PLOW_SPEC_DRAFT_DIR" in str(e), e
        spec["spec"]["k_max"] = 99
        try:
            check(write(spec), bundle, objects, strict=False)
            raise AssertionError("accepted k_max 99")
        except rc.Fail as e:
            assert "k_max" in str(e), e

        # Target disagreements.
        for field, value in [("isa", "gfx950"), ("units", 256), ("num_gpus", 4)]:
            bad = json.loads(json.dumps(good))
            bad["target"][field] = value
            try:
                check(write(bad), bundle, objects, strict=False)
                raise AssertionError(f"accepted target {field}={value}")
            except rc.Fail:
                pass

        # A recorded digest that does not match the file.
        bad = json.loads(json.dumps(good))
        bad["artifacts"]["i.elf"] = "0" * 64
        try:
            check(write(bad), bundle, objects, strict=False)
            raise AssertionError("accepted a wrong artifact digest")
        except rc.Fail as e:
            assert "hashes to" in str(e)

        # validated without a measurement.
        bad = json.loads(json.dumps(good))
        del bad["measured"]
        try:
            check(write(bad), bundle, objects, strict=False)
            raise AssertionError("accepted validated with no measurement")
        except rc.Fail as e:
            assert "must carry `[measured] tok_s`" in str(e)

        # Per-rung overrides: stated in [emit.rung_env], replayed as PLOW_EMIT_RUNG_ENV, resolved
        # per program in build.json `rung_knobs`.
        rb = d / "rung-assets"
        rb.mkdir()
        rung_build = {
            "arch": "gfx942",
            "n_cu": 304,
            "emit_config": {
                "replay": {"PLOW_FP8_KV": "1", "PLOW_EMIT_RUNG_ENV": "prefill:128:PLOW_NO_GLU_FUSE=0"},
                "unrecorded_env": {"PLOW_MLA_PF_V2": "1"},
            },
            "rung_knobs": {"programs": [
                {"program": 0, "phase": "prefill", "rows": 128,
                 "knobs": {"emit.no_glu_fuse": {"value": "false", "source": "rung"}}},
                {"program": 1, "phase": "prefill", "rows": 512,
                 "knobs": {"emit.no_glu_fuse": {"value": "true", "source": "packet"}}},
            ]},
        }
        (rb / "build.json").write_text(json.dumps(rung_build))
        (rb / "weights.json").write_text((bundle / "weights.json").read_text())
        rung = json.loads(json.dumps(good))
        del rung["artifacts"]
        rung["emit"]["rung_env"] = {"prefill": {"128": {"PLOW_NO_GLU_FUSE": "0"}}}
        rung["rung_choices"] = {"PLOW_NO_GLU_FUSE": {"label": "qualified", "evidence": ["rung-sweep/choice.json"]}}
        notes = check(write(rung), rb, None, strict=False)
        assert not any("PLOW_EMIT_RUNG_ENV" in n for n in notes), notes
        # A validated recipe's per-rung choice carries qualified evidence.
        for edit, why in [
            (lambda x: x.pop("rung_choices"), "needs [rung_choices.PLOW_NO_GLU_FUSE]"),
            (lambda x: x["rung_choices"]["PLOW_NO_GLU_FUSE"].update(label="experimental"), "needs 'qualified'"),
            (lambda x: x["rung_choices"]["PLOW_NO_GLU_FUSE"].update(evidence=[]), "needs `evidence`"),
        ]:
            bad = json.loads(json.dumps(rung))
            edit(bad)
            try:
                check(write(bad), rb, None, strict=False)
                raise AssertionError(f"accepted a per-rung choice without evidence ({why})")
            except rc.Fail as e:
                assert why in str(e), e
        for caps, why in [
            ({"no_such_route": "x"}, "not a plowrt inventory item"),
            ({"live_kv": ""}, "needs the measured reason"),
            ({"live_kv": "not measured yet"}, "unmeasured route is not a waiver"),
        ]:
            bad = json.loads(json.dumps(rung))
            bad["capabilities"] = caps
            try:
                check(write(bad), rb, None, strict=False)
                raise AssertionError(f"accepted a bad waiver ({why})")
            except rc.Fail as e:
                assert why in str(e), e
        unvalidated = json.loads(json.dumps(rung))
        unvalidated.pop("rung_choices")
        unvalidated["status"] = "emits"
        unvalidated.pop("measured", None)
        check(write(unvalidated), rb, None, strict=False)
        for edit, why in [
            (lambda x: x["emit"]["rung_env"]["prefill"]["128"].update(PLOW_NO_GLU_FUSE="1"), "contradictory PLOW_EMIT_RUNG_ENV"),
            (lambda x: x["emit"]["rung_env"]["prefill"]["128"].update(PLOW_FP8_KV="1"), "packet-wide"),
            (lambda x: x["emit"]["rung_env"].update(prefill={"0": {"PLOW_NO_GLU_FUSE": "0"}}), "positive"),
            (lambda x: x["emit"]["env"].update(PLOW_EMIT_RUNG_ENV="prefill:128:PLOW_NO_GLU_FUSE=0"), "[emit.rung_env]"),
        ]:
            bad = json.loads(json.dumps(rung))
            edit(bad)
            try:
                check(write(bad), rb, None, strict=False)
                raise AssertionError(f"accepted a bad rung override ({why})")
            except rc.Fail as e:
                assert why in str(e), e
        # The build resolved the value from the packet, not the rung.
        stale = json.loads(json.dumps(rung_build))
        stale["rung_knobs"]["programs"][0]["knobs"]["emit.no_glu_fuse"]["source"] = "packet"
        (rb / "build.json").write_text(json.dumps(stale))
        try:
            check(write(rung), rb, None, strict=False)
            raise AssertionError("accepted a rung override the build did not apply")
        except rc.Fail as e:
            assert "resolved" in str(e), e

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
        except rc.Fail:
            pass


def to_toml(doc: dict) -> str:
    """Minimal TOML writer for the self-test (python has no stdlib writer)."""

    def val(v):
        if isinstance(v, dict):
            return "{ " + ", ".join(f"{key(a)} = {val(b)}" for a, b in v.items()) + " }"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
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


main = rc.main_guard(run)

if __name__ == "__main__":
    raise SystemExit(main())
