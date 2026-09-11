#!/usr/bin/env python3
"""Publish a packed bundle + objset into the HTTPS store.

Writes the tree a `plowrt` client reads:

    v1/blobs/sha256/<hex>[.zst]          content-addressed, shared by everything
    v1/objsets/<objset-id>.json
    v1/<namespace>/<name>/manifests/<label>@g<n>
    v1/<namespace>/<name>/index.json     written LAST
    catalog.json                         model names only

Ordering is the safety property. Blobs go first, then the immutable manifests,
and only then the per-model index — so an interrupted publish leaves the
previous generation completely servable, and a client never sees an index row
pointing at bytes that are not there yet.

  publish_dist.py --bundle bundle.json --objset objset.json --store ./dist
  publish_dist.py --bundle bundle.json --objset objset.json --store ./dist --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import plow_dist as pd


def variant_id(bundle: dict, generation: int) -> str:
    """Identity over the full key INCLUDING the build.

    Two generations of one label are different artifacts with the same name, so
    the build must be part of the id or they would collide.
    """
    t = bundle["target"]
    return pd.short_id(
        bundle["namespace"],
        bundle["name"],
        bundle["label"],
        str(generation),
        t["isa"],
        t["sku"],
        str(t["units"]),
        f"{bundle['parallel']['mode']}{bundle['parallel']['n']}",
        str(bundle["max_ctx"]),
        bundle["plow_git"],
        bundle["objset"]["objset_id"],
    )


def existing_index(store: Path, ns: str, name: str) -> dict | None:
    p = pd.model_dir(store, ns, name) / "index.json"
    return pd.load_json(p) if p.exists() else None


def blob_inventory(bundle: dict, objset: dict) -> dict[str, int]:
    """Every blob a variant needs → its size."""
    inv = {f["sha256"]: f["bytes"] for f in bundle["files"]}
    for o in objset["objects"]:
        inv[o["sha256"]] = o["bytes"]
    return inv


def choose_generation(idx: dict | None, label: str, reuse: bool) -> int:
    if idx is None:
        return 1
    gens = [v["generation"] for v in idx["variants"] if v["label"] == label]
    if not gens:
        return 1
    if reuse:
        return max(gens)
    return max(gens) + 1


def find_predecessor(idx: dict | None, label: str, generation: int) -> dict | None:
    if idx is None:
        return None
    prior = [
        v for v in idx["variants"] if v["label"] == label and v["generation"] < generation
    ]
    return max(prior, key=lambda v: v["generation"]) if prior else None


def source_path_for(digest: str, bundle: dict, objset: dict, assets: Path, objdir: Path) -> Path:
    for f in bundle["files"]:
        if f["sha256"] == digest:
            return assets / f["name"]
    for o in objset["objects"]:
        if o["sha256"] == digest:
            return objdir / o["name"]
    pd.die(f"no source file for blob {digest}")


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle")
    ap.add_argument("--objset")
    ap.add_argument("--assets", help="directory the bundle files live in")
    ap.add_argument("--objects", help="directory the .elf objects live in")
    ap.add_argument("--store", help="destination store root")
    ap.add_argument("--status", choices=["validated", "emits", "refused"], default="emits")
    ap.add_argument("--tok-s", type=float, help="measured aggregate throughput")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--p50-tpot-ms", type=float)
    ap.add_argument("--hf", help="override <org>/<repo>; defaults to the bundle's")
    ap.add_argument("--alias", action="append", default=[])
    ap.add_argument(
        "--reuse-generation",
        action="store_true",
        help="republish the current generation instead of advancing it",
    )
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        print("publish_dist selftest: PASS")
        return
    for need in ("bundle", "objset", "store"):
        if not getattr(args, need):
            pd.die(f"--{need} is required")

    bundle = pd.load_json(Path(args.bundle))
    objset = pd.load_json(Path(args.objset))
    store = Path(args.store)
    assets = Path(args.assets) if args.assets else Path(args.bundle).parent
    objdir = Path(args.objects) if args.objects else Path(args.objset).parent

    if args.status == "validated" and args.tok_s is None:
        pd.die(
            "--status validated requires --tok-s: a validated variant outranks an unmeasured "
            "one in selection, so publishing one without a measurement would be a lie"
        )

    ns, name, label = bundle["namespace"], bundle["name"], bundle["label"]
    idx = existing_index(store, ns, name)
    generation = choose_generation(idx, label, args.reuse_generation)
    predecessor = find_predecessor(idx, label, generation)

    bundle = dict(bundle, generation=generation)
    bundle["variant_id"] = variant_id(bundle, generation)

    inv = blob_inventory(bundle, objset)
    have = {d for d in inv if pd.blob_path(store, d).exists() or (
        pd.blob_path(store, d).with_suffix(pd.blob_path(store, d).suffix + ".zst").exists()
    )}
    missing = {d: n for d, n in inv.items() if d not in have}

    # The upgrade cost a client will actually pay, computed from digests rather
    # than from what kind of change this was.
    upgrade_bytes = 0
    delta = None
    if predecessor is not None:
        # `manifest` is already relative to the model directory.
        prior_manifest = pd.model_dir(store, ns, name) / predecessor["manifest"]
        prior = pd.load_json(prior_manifest) if prior_manifest.exists() else None
        if prior is not None:
            prior_objset = pd.load_json(pd.objset_path(store, prior["objset"]["objset_id"]))
            prior_inv = blob_inventory(prior, prior_objset)
            upgrade_bytes = sum(n for d, n in inv.items() if d not in prior_inv)
            packet_changed = any(
                f["role"] == "packet" and f["sha256"] not in prior_inv for f in bundle["files"]
            )
            delta = "packet" if packet_changed else "objset"

    print(f"{ns}/{name}  {label}@g{generation}")
    print(f"  objset      {objset['objset_id']}")
    print(f"  blobs       {len(inv)} total, {len(missing)} to upload ({pd.human(sum(missing.values()))})")
    if predecessor:
        print(
            f"  supersedes  {predecessor['label']}@g{predecessor['generation']}"
            f"  delta={delta}  upgrade={pd.human(upgrade_bytes)}"
        )
    if args.dry_run:
        print("  (dry run: nothing written)")
        return

    # 1. Blobs first: a manifest must never name bytes that are not there.
    for digest, _ in sorted(missing.items()):
        src = source_path_for(digest, bundle, objset, assets, objdir)
        data = src.read_bytes()
        got = pd.sha256_bytes(data)
        if got != digest:
            pd.die(f"{src}: hashes to {got}, manifest says {digest}")
        dst = pd.blob_path(store, digest)
        packed = None if args.no_compress else pd.compress(data)
        if packed is not None and len(packed) < len(data):
            pd.write_atomic(dst.with_name(dst.name + ".zst"), packed)
        else:
            pd.write_atomic(dst, data)

    # 2. Immutable manifests.
    objset_bytes = pd.canonical_json(objset)
    pd.write_atomic(pd.objset_path(store, objset["objset_id"]), objset_bytes)
    bundle["objset"]["sha256"] = pd.sha256_bytes(objset_bytes)

    bundle_bytes = pd.canonical_json(bundle)
    manifest_rel = f"manifests/{label}@g{generation}"
    pd.write_atomic(pd.model_dir(store, ns, name) / manifest_rel, bundle_bytes)

    # 3. The index LAST, so an interrupted publish leaves the old one usable.
    row = {
        "variant_id": bundle["variant_id"],
        "label": label,
        "generation": generation,
        "status": args.status,
        "target": bundle["target"],
        "parallel": bundle["parallel"],
        "max_ctx": bundle["max_ctx"],
        "features": {},
        "build": {
            "plow_git": bundle["plow_git"],
            "plowc": "0.1.0",
            "objset_id": objset["objset_id"],
            **({"pairing_hash": bundle["pairing_hash"]} if bundle.get("pairing_hash") else {}),
        },
        "manifest": manifest_rel,
        "sha256": pd.sha256_bytes(bundle_bytes),
        "released": _dt.date.today().isoformat(),
    }
    if predecessor:
        row["supersedes"] = predecessor["variant_id"]
        row["delta"] = delta
    if args.tok_s is not None:
        row["measured"] = {"tok_s": args.tok_s}
        if args.p50_tpot_ms is not None:
            row["measured"]["p50_tpot_ms"] = args.p50_tpot_ms
        if args.concurrency is not None:
            row["measured"]["concurrency"] = args.concurrency

    variants = [] if idx is None else list(idx["variants"])
    variants = [v for v in variants if not (v["label"] == label and v["generation"] == generation)]
    variants.append(row)
    variants.sort(key=lambda v: (v["label"], v["generation"]))

    new_idx = {
        "schema": pd.INDEX_SCHEMA,
        "namespace": ns,
        "name": name,
        "hf": args.hf or bundle["checkpoint"]["source"].removeprefix("hf:"),
        "revision": bundle["checkpoint"]["revision"],
        "network": bundle["network"],
        "aliases": sorted(set((idx or {}).get("aliases", []) + args.alias)),
        "checkpoint": {
            "shards": bundle["checkpoint"]["shards"],
            "layout": bundle["checkpoint"]["layout"],
            "bytes": bundle["checkpoint"]["bytes"],
        },
        "variants": variants,
    }
    pd.no_nix_store_paths(new_idx, "index.json")
    pd.write_atomic(pd.model_dir(store, ns, name) / "index.json", pd.canonical_json(new_idx))

    # 4. The discovery catalog: names only, so staleness is harmless.
    catalog_path = store / "catalog.json"
    catalog = pd.load_json(catalog_path) if catalog_path.exists() else {"models": []}
    entry = {"namespace": ns, "name": name, "hf": new_idx["hf"]}
    catalog["models"] = [
        m for m in catalog["models"] if (m["namespace"], m["name"]) != (ns, name)
    ] + [entry]
    catalog["models"].sort(key=lambda m: (m["namespace"], m["name"]))
    pd.write_atomic(catalog_path, pd.canonical_json(catalog))

    print(f"  published   {store}/v1/{ns}/{name}/{manifest_rel}")


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        store = d / "store"
        assets = d / "assets"
        objdir = d / "obj"
        assets.mkdir()
        objdir.mkdir()

        pkt = b"PLOWDEV" + b"\x0b" * 4096
        (assets / "model.pkt").write_bytes(pkt)
        target = {
            "vendor": "amd",
            "isa": "gfx942",
            "sku": "MI325X",
            "units": 304,
            "mem_bytes": 274877906944,
        }

        def make(obj_bytes: bytes, objset_id: str):
            (objdir / "i.elf").write_bytes(obj_bytes)
            objset = {
                "schema": pd.OBJSET_SCHEMA,
                "objset_id": objset_id,
                "target": target,
                "toolchain": "rocm-7.14.0-nix",
                "plow_git": "a" * 40,
                "script": "scripts/build_gfx942.sh",
                "env": {},
                "defines": {},
                "objects": [
                    {
                        "name": "i.elf",
                        "sha256": pd.sha256_bytes(obj_bytes),
                        "bytes": len(obj_bytes),
                        "arms": [],
                    }
                ],
            }
            bundle = {
                "schema": pd.BUNDLE_SCHEMA,
                "namespace": "infervisor",
                "name": "kimi-k3",
                "label": "gfx942-mi325x-tp8-32k",
                "generation": 0,
                "variant_id": "",
                "plow_git": "a" * 40,
                "network": "kimi-k3",
                "target": target,
                "parallel": {"mode": "tp", "n": 8},
                "max_ctx": 32768,
                "files": [
                    {
                        "role": "packet",
                        "name": "model.pkt",
                        "sha256": pd.sha256_bytes(pkt),
                        "bytes": len(pkt),
                    }
                ],
                "tokenizer": {"source": "checkpoint", "file": "tokenizer.json", "verified": False},
                "checkpoint": {
                    "source": "hf:moonshotai/Kimi-K3",
                    "revision": "rev1",
                    "shards": 96,
                    "layout": "native-mxfp4",
                    "bytes": 1590000000000,
                },
                "objset": {
                    "objset_id": objset_id,
                    "manifest": f"v1/objsets/{objset_id}.json",
                    "sha256": "",
                    "lowrung": [],
                },
                "runtime_env": {},
            }
            bp, op = d / "bundle.json", d / "objset.json"
            bp.write_text(json.dumps(bundle))
            op.write_text(json.dumps(objset))
            return bp, op

        def publish(bp, op, extra=()):
            import sys

            argv = sys.argv
            sys.argv = [
                "publish_dist.py",
                "--bundle", str(bp),
                "--objset", str(op),
                "--assets", str(assets),
                "--objects", str(objdir),
                "--store", str(store),
                *extra,
            ]
            try:
                run()
            finally:
                sys.argv = argv

        # g1.
        bp, op = make(b"objset one" * 100, "obj1")
        publish(bp, op)
        idx = pd.load_json(store / "v1/infervisor/kimi-k3/index.json")
        assert len(idx["variants"]) == 1 and idx["variants"][0]["generation"] == 1
        assert (store / "catalog.json").exists()

        # g2 changes only the objset: the packet blob must be reused.
        bp, op = make(b"objset two" * 100, "obj2")
        publish(bp, op)
        idx = pd.load_json(store / "v1/infervisor/kimi-k3/index.json")
        assert len(idx["variants"]) == 2, idx
        g2 = [v for v in idx["variants"] if v["generation"] == 2][0]
        assert g2["delta"] == "objset", g2
        assert g2["supersedes"] == idx["variants"][0]["variant_id"]
        # Two objsets + one shared packet = three blobs, not four.
        blobs = list((store / "v1/blobs/sha256").iterdir())
        assert len(blobs) == 3, [b.name for b in blobs]

        # Generations of one label are distinct artifacts.
        assert g2["variant_id"] != idx["variants"][0]["variant_id"]

        # A validated claim needs a measurement.
        bp, op = make(b"objset three" * 100, "obj3")
        try:
            publish(bp, op, ["--status", "validated"])
            raise AssertionError("accepted validated with no measurement")
        except pd.Fail as e:
            assert "requires --tok-s" in str(e)

        # A dry run writes nothing.
        before = sorted(p.name for p in (store / "v1/blobs/sha256").iterdir())
        publish(bp, op, ["--dry-run"])
        after = sorted(p.name for p in (store / "v1/blobs/sha256").iterdir())
        assert before == after, "dry run wrote blobs"
        idx2 = pd.load_json(store / "v1/infervisor/kimi-k3/index.json")
        assert len(idx2["variants"]) == 2, "dry run touched the index"


main = pd.main_guard(run)

if __name__ == "__main__":
    raise SystemExit(main())
