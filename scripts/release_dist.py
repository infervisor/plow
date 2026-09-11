#!/usr/bin/env python3
"""Release one model x hardware optimization.

An asset release is a set of compiled artifacts tied to an exact `plowc` source
commit. It advances independently of the runtime: publishing a new asset
generation needs no new `plowrt` binary, and a new binary does not invalidate
published assets.

    release_dist.py --recipe recipes/infervisor/kimi-k3/<label>.toml --dry-run
    release_dist.py --recipe ... --commit <full-sha> --publish

Five stages, in order, each refusing rather than degrading:

 1. RESOLVE   the release source to a full commit, once. `main` by default;
              `--commit` selects another. A branch name is not provenance, so
              the resolved SHA is what gets recorded and what everything else
              uses. Retries do not re-resolve.
 2. BUILD     the artifacts from a clean checkout of that commit. A compiler
              whose source provenance disagrees with the recipe is refused even
              when its version string matches.
 3. VALIDATE  recipe, arm and pairing checks, then the recipe's gates on the
              recorded hardware. `validated` requires a measurement; an
              unmeasured build is publishable only as `emits`.
 4. PREPARE   a release report: generation, supersedes, changed blobs, transfer
              bytes, runtime compatibility. `--dry-run` stops here.
 5. PUBLISH   blobs, manifests, then the index — serialized per model, refusing
              to reuse a generation that already exists.

This script runs on the build machine. Serving hosts use neither it nor nix.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import plow_dist as pd

REPO = Path(__file__).resolve().parent.parent


# --- 1. resolve --------------------------------------------------------------


def resolve_source(commit: str | None) -> str:
    """Resolve the release source to a full SHA, exactly once.

    A moving branch name cannot be published as provenance: two runs would mean
    two different sources under one label.
    """
    rev = commit or "main"
    sha = pd.git_commit(REPO, rev)
    if commit and not pd.git_commit(REPO, f"{sha}^{{commit}}") == sha:
        pd.die(f"{commit} does not name a commit")
    return sha


def recipe_requires(recipe: dict) -> str:
    git = recipe.get("plow_git")
    if not git:
        pd.die("recipe has no `plow_git`: an asset is tied to the compiler source that built it")
    if len(git) != 40:
        pd.die(f"recipe `plow_git` must be a full 40-hex commit, got {git!r}")
    return git


# --- 2. build ----------------------------------------------------------------


def check_compiler_provenance(recipe_git: str, release_git: str) -> None:
    """The compiler's SOURCE must match the recipe, not merely its version.

    Two builds of `plowc` can report the same version string and emit different
    packets; the commit is what identifies the emitter.
    """
    if recipe_git != release_git:
        pd.die(
            f"recipe pins plow_git {recipe_git[:12]} but this release resolves to "
            f"{release_git[:12]}. Update the recipe in the same commit that changes the "
            f"compiler, so the recorded provenance is the one that built the bytes."
        )


def run_step(cmd: list[str], what: str, dry: bool) -> None:
    print(f"  $ {' '.join(cmd)}")
    if dry:
        return
    r = subprocess.run(cmd, cwd=REPO, check=False)
    if r.returncode != 0:
        pd.die(f"{what} failed (exit {r.returncode})")


# --- 3. validate -------------------------------------------------------------


def validate(recipe: dict, bundle: dict, objset: dict, status: str) -> list[str]:
    """Checks that need no hardware. The recipe's own gates need the GPU."""
    notes = []
    # Shared with check_recipe.py: one implementation of what may be called
    # `validated`, so the checker and the publisher cannot disagree.
    pd.recipe_target_matches(recipe, objset["target"], "objset")

    # The pairing rule, enforced here as well as in pack_bundle: a release must
    # not be the first place it is noticed.
    pairing = bundle.get("pairing_hash")
    for o in objset.get("objects", []):
        stamp = o.get("packet_hash")
        if stamp and stamp != pairing:
            pd.die(f"{o['name']} is stamped for packet {stamp}, the bundle's packet is {pairing}")

    if status == "validated":
        pd.recipe_invariants({**recipe, "status": "validated"})
        notes.append(f"measured {(recipe.get('measured') or {})['tok_s']} tok/s")
    else:
        notes.append("unmeasured — publishable only as `emits`")
    return notes


def run_gates(recipe: dict, dry: bool) -> None:
    gates = recipe.get("gates") or []
    if not gates:
        print("  (recipe declares no gates)")
        return
    for g in gates:
        name = g.get("name", "<unnamed>")
        cmd = g.get("cmd")
        if not cmd:
            print(f"  gate {name}: no command; expectation `{g.get('expect')}` is recorded only")
            continue
        env = " ".join(f"{k}={v}" for k, v in (g.get("env") or {}).items())
        print(f"  gate {name}: {env} {cmd}".rstrip())
        if dry:
            continue
        r = subprocess.run(["bash", "-c", cmd], cwd=REPO, env={**os.environ, **(g.get("env") or {})}, check=False)
        if r.returncode != 0:
            pd.die(f"gate `{name}` failed; this release cannot be promoted")


# --- 5. publish --------------------------------------------------------------


class ModelLock:
    """Serialize publication per model.

    Two concurrent releases of one model would race on the index and could reuse
    a generation. The lock is per (namespace, name), so unrelated models publish
    in parallel.
    """

    def __init__(self, store: Path, ns: str, name: str, timeout: float = 300.0):
        self.path = store / ".locks" / f"{ns}--{name}.lock"
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"{os.getpid()}\n".encode())
                return self
            except FileExistsError:
                if time.time() > deadline:
                    pd.die(
                        f"another release of this model holds {self.path}. Publication is "
                        f"serialized per model so two runs cannot reuse a generation."
                    )
                time.sleep(0.2)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def next_generation(store: Path, ns: str, name: str, label: str) -> tuple[int, dict | None]:
    p = pd.model_dir(store, ns, name) / "index.json"
    if not p.exists():
        return 1, None
    idx = pd.load_json(p)
    same = [v for v in idx["variants"] if v["label"] == label]
    if not same:
        return 1, None
    newest = max(same, key=lambda v: v["generation"])
    return newest["generation"] + 1, newest


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", action="append", default=[], help="recipe TOML; repeatable")
    ap.add_argument("--store", help="destination store root")
    ap.add_argument("--commit", help="release from this commit instead of main")
    ap.add_argument("--assets", help="prebuilt asset directory (skips the build step)")
    ap.add_argument("--objects", help="prebuilt object directory")
    ap.add_argument("--status", choices=["validated", "emits"], default="emits")
    ap.add_argument("--publish", action="store_true", help="write to the store")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        print("release_dist selftest: PASS")
        return
    if not args.recipe:
        pd.die("give at least one --recipe")
    if args.publish and args.dry_run:
        pd.die("--publish and --dry-run are mutually exclusive")
    if args.publish and not args.store:
        pd.die("--publish needs a --store")

    dry = not args.publish

    # 1. Resolve once. Every later stage uses this SHA.
    release_git = resolve_source(args.commit)
    clean = pd.git_is_clean(REPO)
    print(f"release source  {release_git}  ({'clean' if clean else 'MODIFIED'})")
    if not clean and not args.allow_dirty:
        pd.die(
            "the working tree is modified, so the recorded commit would not describe the bytes "
            "built. Commit first, or pass --allow-dirty for a throwaway run."
        )

    for rp in args.recipe:
        recipe = pd.load_toml(Path(rp))
        label = recipe.get("label") or Path(rp).stem
        ns = recipe.get("namespace", "infervisor")
        name = recipe.get("name") or recipe.get("network")
        if not name:
            pd.die(f"{rp}: no `name`")
        print(f"\n=== {ns}/{name}  {label} ===")

        # 2. Provenance: the recipe pins the compiler source.
        check_compiler_provenance(recipe_requires(recipe), release_git)
        print("  compiler provenance matches the recipe")

        if not args.assets or not args.objects:
            print("  build: pass --assets/--objects, or run the recipe's build steps first")
            if not dry:
                pd.die("--publish needs --assets and --objects")
            continue

        # 3. Validate.
        objset = pd.load_json(Path(args.objects) / "objset.json")
        bundle = pd.load_json(Path(args.assets) / "bundle.json")
        for n in validate(recipe, bundle, objset, args.status):
            print(f"  {n}")
        run_gates(recipe, dry)

        # 4. Report.
        store = Path(args.store) if args.store else None
        if store is None:
            print("  (no --store: nothing to compare against)")
            continue
        gen, prior = next_generation(store, ns, name, label)
        print(f"  generation  g{gen}" + (f" (supersedes g{prior['generation']})" if prior else ""))

        if dry:
            print("  (dry run: nothing written remotely)")
            continue

        # 5. Publish, serialized per model.
        with ModelLock(store, ns, name):
            gen2, _ = next_generation(store, ns, name, label)
            if gen2 != gen:
                pd.die(
                    f"another release published g{gen2 - 1} while this one was preparing. "
                    f"Re-run; generations are never reused."
                )
            cmd = [
                sys.executable,
                str(REPO / "scripts" / "publish_dist.py"),
                "--bundle", str(Path(args.assets) / "bundle.json"),
                "--objset", str(Path(args.objects) / "objset.json"),
                "--assets", str(args.assets),
                "--objects", str(args.objects),
                "--store", str(store),
                "--status", args.status,
            ]
            m = recipe.get("measured") or {}
            if m.get("tok_s"):
                cmd += ["--tok-s", str(m["tok_s"])]
            if m.get("concurrency"):
                cmd += ["--concurrency", str(m["concurrency"])]
            run_step(cmd, "publish", dry=False)


def self_test() -> None:
    import tempfile

    # Provenance: a recipe pinning a different compiler commit is refused.
    try:
        check_compiler_provenance("a" * 40, "b" * 40)
        raise AssertionError("accepted a provenance mismatch")
    except pd.Fail as e:
        assert "Update the recipe" in str(e)
    check_compiler_provenance("a" * 40, "a" * 40)

    # A validated claim needs a measurement.
    objset = {
        "target": {"isa": "gfx942", "sku": "MI325X"},
        "objects": [{"name": "i.elf", "packet_hash": "0x1"}],
    }
    bundle = {"pairing_hash": "0x1"}
    recipe = {"target": {"isa": "gfx942", "sku": "MI325X"}, "plow_git": "a" * 40}
    try:
        validate(recipe, bundle, objset, "validated")
        raise AssertionError("accepted validated with no measurement")
    except pd.Fail as e:
        assert "must carry `[measured] tok_s`" in str(e)
    assert validate(recipe, bundle, objset, "emits")

    recipe_ok = dict(recipe, measured={"tok_s": 131.162})
    assert "131.162" in " ".join(validate(recipe_ok, bundle, objset, "validated"))

    # A target the recipe does not describe is a mispaired release.
    try:
        validate({"target": {"isa": "gfx950", "sku": "MI355X"}}, bundle, objset, "emits")
        raise AssertionError("accepted a target mismatch")
    except pd.Fail:
        pass

    # A stamped object must match the bundle's packet.
    try:
        validate(recipe, {"pairing_hash": "0xdead"}, objset, "emits")
        raise AssertionError("accepted a pairing mismatch")
    except pd.Fail:
        pass

    # Generations advance and never repeat.
    with tempfile.TemporaryDirectory() as td:
        store = Path(td)
        assert next_generation(store, "ns", "m", "L") == (1, None)
        pd.write_atomic(
            pd.model_dir(store, "ns", "m") / "index.json",
            pd.canonical_json(
                {
                    "schema": pd.INDEX_SCHEMA,
                    "namespace": "ns",
                    "name": "m",
                    "hf": "o/r",
                    "revision": "x",
                    "network": "m",
                    "checkpoint": {"shards": 1, "layout": "l", "bytes": 1},
                    "variants": [{"label": "L", "generation": 2, "variant_id": "v2"}],
                }
            ),
        )
        gen, prior = next_generation(store, "ns", "m", "L")
        assert (gen, prior["generation"]) == (3, 2)

        # The per-model lock is exclusive.
        with ModelLock(store, "ns", "m"):
            try:
                with ModelLock(store, "ns", "m", timeout=0.3):
                    raise AssertionError("two releases held the lock at once")
            except pd.Fail as e:
                assert "serialized per model" in str(e)
        # And released afterwards.
        with ModelLock(store, "ns", "m", timeout=1.0):
            pass


main = pd.main_guard(run)

if __name__ == "__main__":
    raise SystemExit(main())
