#!/usr/bin/env python3
"""Publish built `plowrt` binaries as a runtime release.

Consumes the output of `nix build .#plowrt-release` (a tarball, its sha256, and
a `release.json` describing the build) for one or more platform triples, and
writes the tree a client reads:

    v1/runtime/index.json
    v1/runtime/<version>/<triple>/plowrt.tar.gz

The runtime track is independent of the asset track. A new asset generation does
not need a new binary, and a new binary does not invalidate published assets —
they are separately versioned on purpose, which is why this is its own script
rather than a mode of `publish_dist.py`.

  nix build .#plowrt-release --no-link --print-out-paths
  release_runtime.py --release <that path> --store ./dist

The index is written LAST, so an interrupted publish leaves the previous
release downloadable.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import plow_dist as pd

RUNTIME_SCHEMA = "plow.dist.runtime.v1"


def read_release(d: Path) -> dict:
    meta = pd.load_json(d / "release.json")
    for field in ("version", "semver", "commit", "triple", "features", "file"):
        if field not in meta:
            pd.die(f"{d}/release.json has no `{field}`")
    tar = d / meta["file"]
    if not tar.is_file():
        pd.die(f"{tar} does not exist")
    meta["_path"] = tar
    meta["sha256"] = pd.sha256_file(tar)
    meta["bytes"] = tar.stat().st_size
    return meta


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", action="append", default=[], help="a nix release output dir")
    ap.add_argument("--store", help="destination store root")
    ap.add_argument("--allow-dirty", action="store_true", help="publish a non-reproducible build")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        print("release_runtime selftest: PASS")
        return
    if not args.release or not args.store:
        pd.die("give at least one --release and a --store")

    store = Path(args.store)
    rels = [read_release(Path(r)) for r in args.release]

    versions = {r["version"] for r in rels}
    if len(versions) > 1:
        pd.die(
            f"a release is one version across every triple, got {sorted(versions)}. "
            f"Build them all from the same commit."
        )
    version = versions.pop()

    # A "-dirty" build cannot be reproduced from the commit it names, so it is
    # not publishable by default: the whole point of `<semver>+<git12>` is that
    # the suffix identifies the source.
    if "dirty" in version and not args.allow_dirty:
        pd.die(
            f"{version} was built from a modified checkout, so its commit does not identify its "
            f"source. Commit first, or pass --allow-dirty for a throwaway build."
        )

    triples = [r["triple"] for r in rels]
    if len(set(triples)) != len(triples):
        pd.die(f"two releases claim the same triple: {sorted(triples)}")

    print(f"runtime {version}")
    for r in rels:
        print(
            f"  {r['triple']:<32} {pd.human(r['bytes']):>10}  "
            f"features={','.join(r['features'])}"
        )
    if args.dry_run:
        print("  (dry run: nothing written)")
        return

    # 1. Payloads first: an index must never name a tarball that is not there.
    for r in rels:
        dst = store / "v1" / "runtime" / version / r["triple"] / "plowrt.tar.gz"
        pd.write_atomic(dst, r["_path"].read_bytes())
        pd.write_atomic(
            dst.with_suffix(".gz.sha256"), f"{r['sha256']}  plowrt.tar.gz\n".encode()
        )

    # 2. Then the index, merging with what is already published so an older
    #    release stays installable.
    index_path = store / "v1" / "runtime" / "index.json"
    existing = pd.load_json(index_path) if index_path.exists() else {"runtime": []}
    rows = [
        row
        for row in existing.get("runtime", [])
        if not (row["version"] == version and row["triple"] in triples)
    ]
    for r in rels:
        rows.append(
            {
                "version": version,
                "semver": r["semver"],
                "commit": r["commit"],
                "triple": r["triple"],
                "features": r["features"],
                "path": f"v1/runtime/{version}/{r['triple']}/plowrt.tar.gz",
                "sha256": r["sha256"],
                "bytes": r["bytes"],
                "released": _dt.date.today().isoformat(),
            }
        )
    rows.sort(key=lambda row: (row["version"], row["triple"]))
    doc = {"schema": RUNTIME_SCHEMA, "latest": version, "runtime": rows}
    pd.no_nix_store_paths(doc, "v1/runtime/index.json")
    pd.write_atomic(index_path, pd.canonical_json(doc))
    print(f"  published   {index_path}")


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        store = d / "store"

        def make(triple: str, version: str, body: bytes) -> Path:
            rd = d / f"rel-{triple}-{version}"
            rd.mkdir(parents=True)
            name = f"plowrt-{version}-{triple}.tar.gz"
            (rd / name).write_bytes(body)
            (rd / "release.json").write_text(
                json.dumps(
                    {
                        "version": version,
                        "semver": version.split("+")[0],
                        "commit": version.split("+")[1],
                        "triple": triple,
                        "features": ["cpu", "dist"],
                        "file": name,
                    }
                )
            )
            return rd

        def publish(dirs, extra=()):
            import sys

            argv = sys.argv
            sys.argv = [
                "release_runtime.py",
                *sum([["--release", str(x)] for x in dirs], []),
                "--store",
                str(store),
                *extra,
            ]
            try:
                run()
            finally:
                sys.argv = argv

        a = make("x86_64-unknown-linux-gnu", "0.2.0+abc123def456", b"tarball-a")
        b = make("aarch64-apple-darwin", "0.2.0+abc123def456", b"tarball-b")
        publish([a, b])

        idx = pd.load_json(store / "v1/runtime/index.json")
        assert idx["latest"] == "0.2.0+abc123def456"
        assert len(idx["runtime"]) == 2, idx
        assert {r["triple"] for r in idx["runtime"]} == {
            "x86_64-unknown-linux-gnu",
            "aarch64-apple-darwin",
        }
        # The digest a client verifies must be of the published bytes.
        row = [r for r in idx["runtime"] if r["triple"] == "aarch64-apple-darwin"][0]
        assert row["sha256"] == pd.sha256_bytes(b"tarball-b")
        assert (store / row["path"]).read_bytes() == b"tarball-b"

        # A second release must not remove the first.
        c = make("x86_64-unknown-linux-gnu", "0.3.0+beefbeefbeef", b"tarball-c")
        publish([c])
        idx = pd.load_json(store / "v1/runtime/index.json")
        assert len(idx["runtime"]) == 3, idx
        assert idx["latest"] == "0.3.0+beefbeefbeef"

        # Mixed versions in one release are a mistake, not a feature.
        try:
            publish([a, c])
            raise AssertionError("accepted mixed versions")
        except pd.Fail as e:
            assert "one version across every triple" in str(e)

        # A dirty build names a commit it does not correspond to.
        dirty = make("x86_64-unknown-linux-gnu", "0.4.0+cafe-dirty", b"x")
        try:
            publish([dirty])
            raise AssertionError("accepted a dirty build")
        except pd.Fail as e:
            assert "modified checkout" in str(e)
        publish([dirty], ["--allow-dirty"])

        # A dry run writes nothing.
        e2 = make("aarch64-unknown-linux-gnu", "0.5.0+aaaaaaaaaaaa", b"y")
        before = pd.load_json(store / "v1/runtime/index.json")
        publish([e2], ["--dry-run"])
        assert pd.load_json(store / "v1/runtime/index.json") == before


main = pd.main_guard(run)

if __name__ == "__main__":
    raise SystemExit(main())
