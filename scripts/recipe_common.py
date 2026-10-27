#!/usr/bin/env python3
"""Shared helpers for `check_recipe.py` and `render_recipe.py`.

Recipes are read here, with the one implementation of "may this be called
validated?" both tools agree on. Model bundles are packed and published by
`plowrt pack`/`plowrt publish` alone; nothing here forms a manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


class Fail(Exception):
    """A refusal the operator can act on."""


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    raise Fail(msg)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- git provenance ----------------------------------------------------------


def git_commit(repo: Path, rev: str = "HEAD") -> str:
    """Full 40-character SHA. Abbreviations are not provenance."""
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", rev],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        die(f"cannot resolve {rev} in {repo}: {out.stderr.strip()}")
    sha = out.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        die(f"{rev} resolved to {sha!r}, which is not a full commit SHA")
    return sha


def write_atomic(path: Path, data: bytes) -> None:
    """Stage and rename.

    Writing in place would let a concurrent reader see a partial manifest under
    a name that promises the whole thing — the same hazard
    `scripts/install_hsaco.sh` exists for on the object side.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".part{os.getpid()}")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        die(f"{path} does not exist")
    except json.JSONDecodeError as e:
        die(f"{path}: not valid JSON: {e}")


def load_toml(path: Path):
    import tomllib

    try:
        return tomllib.loads(path.read_text())
    except FileNotFoundError:
        die(f"{path} does not exist")
    except tomllib.TOMLDecodeError as e:
        die(f"{path}: not valid TOML: {e}")


def recipe_invariants(recipe: dict) -> None:
    """The rules a recipe must satisfy, wherever it is read.

    One implementation of "may this be called validated?", so the checker and
    the renderer cannot disagree — the same hazard the packet-pairing check
    refuses on principle.
    """
    status = recipe.get("status", "emits")
    if status not in ("validated", "emits", "refused"):
        die(f"status must be validated|emits|refused, got {status!r}")

    if status == "refused":
        if not recipe.get("refusal"):
            die("a `refused` recipe must record `refusal` — the file:line that refuses it")
        if recipe.get("emit") or recipe.get("measured"):
            die("a `refused` recipe cannot carry [emit] or [measured]")
        return

    if status == "validated" and not (recipe.get("measured") or {}).get("tok_s"):
        die(
            "a `validated` recipe must carry `[measured] tok_s`: a validated variant outranks "
            "an unmeasured one at selection, so publishing one without a measurement would "
            "misreport it. Use `emits` until a gate run fills it in."
        )

    git = recipe.get("plow_git")
    if not git:
        die("recipe has no `plow_git`: an asset is tied to the compiler source that built it")
    if len(git) != 40:
        die(f"`plow_git` must be a full 40-hex commit, got {git!r}")


def main_guard(fn):
    """Run `fn`, turning a refusal into a clean message and exit 1."""

    def wrapper() -> int:
        try:
            fn()
        except Fail as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    return wrapper
