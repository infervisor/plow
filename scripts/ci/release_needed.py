#!/usr/bin/env python3
"""Say whether main holds anything a new plowrt/plowc release would carry.

    release_needed.py            # prints "true" or "false"
    release_needed.py --paths    # prints the watched paths, one per line

A release is needed when a watched path differs between the last release and
HEAD. Watched: every workspace crate `plowrt` or `plowc` is built from (from
`cargo metadata`), the C sources `plowrt`'s build script compiles, the lock
files, the flake, and `install.sh`. A change to docs, recipes, perf data, CI
or a crate neither binary links releases nothing.

The last release is the newest `vX.Y.Z` tag. CI tags its own bump commit, so
the source that release was built from is that commit's parent. The tag is
pushed before the upload, so a tag whose version the registry
(`PLOW_REGISTRY`) does not list is an upload that failed: a release is needed
then too, and takes the next version.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request

# Not crates, but part of what a release is built from or publishes.
ALWAYS = ["Cargo.toml", "Cargo.lock", "flake.nix", "flake.lock", ".cargo",
          "runtime", "include", "scripts/install.sh"]
BINARIES = ("plowrt", "plowc")
SEMVER = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def git(*args):
    return subprocess.run(("git",) + args, check=True, text=True, capture_output=True).stdout.strip()


def crate_paths():
    """Workspace directories of the path crates `plowrt` and `plowc` depend on."""
    meta = json.loads(subprocess.check_output(
        ("cargo", "metadata", "--format-version", "1", "--locked", "--all-features"), text=True))
    root = meta["workspace_root"].rstrip("/") + "/"
    packages = {p["id"]: p for p in meta["packages"]}
    edges = {n["id"]: n["dependencies"] for n in meta["resolve"]["nodes"]}
    todo = [i for i, p in packages.items() if p["name"] in BINARIES and p["source"] is None]
    if len(todo) != len(BINARIES):
        raise SystemExit(f"cargo metadata did not find the workspace crates {BINARIES}")
    seen = set()
    while todo:
        package = todo.pop()
        if package in seen:
            continue
        seen.add(package)
        todo.extend(edges[package])
    paths = set()
    for package in seen:
        manifest = packages[package]["manifest_path"]
        if packages[package]["source"] is None and manifest.startswith(root):
            paths.add(manifest[len(root):].rsplit("/", 1)[0])
    return sorted(paths)


def last_release():
    """The newest `vX.Y.Z` tag and the commit it was built from, or None."""
    tags = [t for t in git("tag", "--list", "v*").split() if SEMVER.match(t)]
    if not tags:
        return None
    tag = max(tags, key=lambda t: tuple(int(n) for n in SEMVER.match(t).groups()))
    commit = git("rev-parse", f"{tag}^{{commit}}")
    if git("show", "-s", "--format=%s", commit) == f"plow: {tag} [skip ci]":
        commit = git("rev-parse", f"{commit}^")
    print(f"last release {tag}, built from {commit[:12]}", file=sys.stderr)
    return tag, commit


def published(tag):
    """Whether the registry's index lists a release of this tag's version."""
    registry = os.environ.get("PLOW_REGISTRY", "https://pdist.infervisor.ai").rstrip("/")
    with urllib.request.urlopen(f"{registry}/v1/plow-runtime/index.json", timeout=30) as response:
        index = json.load(response)
    return any(r.get("semver") == tag[1:] for r in index.get("releases", []))


def needed(paths):
    release = last_release()
    if release is None:
        print("no release yet", file=sys.stderr)
        return True
    tag, base = release
    try:
        if not published(tag):
            print(f"{tag} is tagged but not in the registry", file=sys.stderr)
            return True
    except OSError as error:
        # An unreadable registry (or one with no release yet) is not a reason
        # to skip: publishing is create-only, so trying again is safe.
        print(f"registry index not read ({error}); releasing", file=sys.stderr)
        return True
    changed = git("diff", "--name-only", base, "HEAD", "--", *paths).splitlines()
    for name in changed[:20]:
        print(f"changed: {name}", file=sys.stderr)
    if len(changed) > 20:
        print(f"... and {len(changed) - 20} more", file=sys.stderr)
    if not changed:
        print("nothing a release is built from changed", file=sys.stderr)
    return bool(changed)


def main():
    # `--watch a b …` replaces the cargo lookup, for the self-test.
    if sys.argv[1:2] == ["--watch"]:
        paths = sys.argv[2:]
    else:
        paths = ALWAYS + crate_paths()
    if sys.argv[1:] == ["--paths"]:
        print("\n".join(paths))
        return
    print("true" if needed(paths) else "false")


if __name__ == "__main__":
    main()
