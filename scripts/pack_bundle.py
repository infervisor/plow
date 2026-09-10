#!/usr/bin/env python3
"""Turn a compiled asset directory into a `bundle.json`.

Input is what `plowc --emit devblob --out <dir>` leaves: `model.pkt`,
`build.json`, `weights.json`, `plow_config.h`, and any derived sidecar. This
script records identity and provenance, and enforces the two rules that would
otherwise fail much later and much less clearly:

* the objset must PAIR with the packet — a specialised object stamps the packet
  it belongs to, and `plowrt` refuses a mismatch deep in the loader;
* the tokenizer's provenance must be declared, and a `checkpoint`-sourced one
  must NOT be shipped, because the distribution carries only what plow produced.

  pack_bundle.py build-amd/k3-assets --objset objset.json \\
      --namespace infervisor --name kimi-k3 \\
      --hf moonshotai/Kimi-K3 --revision <sha> --out bundle.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plow_dist as pd

# Files plow produced and therefore distributes. Anything the HF snapshot
# already carries is linked at prepare time instead; see `--tokenizer`.
ROLES = {
    "model.pkt": "packet",
    "build.json": "build_manifest",
    "weights.json": "asset_manifest",
    "plow_config.h": "config_header",
}


def classify(p: Path) -> str | None:
    if p.name in ROLES:
        return ROLES[p.name]
    if p.name.endswith(".safetensors"):
        return "derived_shard"
    return None


def build(args) -> dict:
    assets = Path(args.assets)
    if not assets.is_dir():
        pd.die(f"{assets} is not a directory")
    objset = pd.load_json(Path(args.objset))

    build_json = pd.load_json(assets / "build.json") if (assets / "build.json").exists() else {}
    pairing = (build_json.get("pairing") or {}).get("hash")

    # A specialised object pairs only with the packet that produced it; a
    # general one carries no stamp and pairs with any. Catching a mismatch here
    # turns a refusal twenty frames into the loader into an obvious one.
    for o in objset.get("objects", []):
        stamp = o.get("packet_hash")
        if stamp is None:
            continue
        if pairing is None:
            pd.die(
                f"{o['name']} is stamped for packet {stamp}, but {assets/'build.json'} records no "
                f"pairing hash. Publishing this pair would be refused at load."
            )
        if stamp != pairing:
            pd.die(
                f"{o['name']} is stamped for packet {stamp}, but this packet is {pairing}. "
                f"The objset was built against a different emit; rebuild one of them."
            )

    files = []
    for p in sorted(assets.iterdir()):
        if not p.is_file():
            continue
        role = classify(p)
        if role is None:
            continue
        files.append(
            {
                "role": role,
                "name": p.name,
                "sha256": pd.sha256_file(p),
                "bytes": p.stat().st_size,
            }
        )
    if not any(f["role"] == "packet" for f in files):
        pd.die(f"{assets}: no model.pkt — that is not a compiled asset directory")

    # The tokenizer. Distributed ONLY when plow generated it: `plowc` symlinks
    # the checkpoint's own file, so shipping that would redistribute HF content
    # and risk it diverging from the checkpoint it must match. K3 is the real
    # exception — it ships tiktoken.model and no tokenizer.json.
    tok_path = assets / "tokenizer.json"
    if args.tokenizer == "derived":
        if not tok_path.is_file() or tok_path.is_symlink():
            pd.die(
                "--tokenizer derived requires a REAL tokenizer.json in the asset directory "
                "(a symlink means it came from the checkpoint)"
            )
        tokenizer = {
            "source": "derived",
            "file": "tokenizer.json",
            "sha256": pd.sha256_file(tok_path),
            "generator": args.tokenizer_generator,
            "verified": bool(args.tokenizer_verified),
        }
        files.append(
            {
                "role": "tokenizer",
                "name": "tokenizer.json",
                "sha256": tokenizer["sha256"],
                "bytes": tok_path.stat().st_size,
            }
        )
        if not args.tokenizer_generator:
            pd.die("--tokenizer derived requires --tokenizer-generator <script>")
    else:
        tokenizer = {"source": "checkpoint", "file": "tokenizer.json", "verified": False}

    lowrung = build_lowrung(args.lowrung)

    weights = pd.load_json(assets / "weights.json") if (assets / "weights.json").exists() else {}
    network = args.network or weights.get("network")
    if not network:
        pd.die(
            "no network slug: pass --network, or emit a weights.json carrying one "
            "(it is the registry key `plowrt serve` registers under)"
        )

    target = dict(objset["target"])
    label = args.label or default_label(target, args.parallel, args.max_ctx, build_json)

    doc = {
        "schema": pd.BUNDLE_SCHEMA,
        "namespace": args.namespace,
        "name": args.name,
        "label": label,
        "generation": 0,  # assigned by publish_dist.py against the live index
        "variant_id": "",  # ditto
        "plow_git": args.plow_git,
        "network": network,
        "target": target,
        "parallel": {"mode": args.parallel_mode, "n": args.parallel},
        "max_ctx": args.max_ctx,
        "files": files,
        "tokenizer": tokenizer,
        "checkpoint": {
            "source": f"hf:{args.hf}",
            "revision": args.revision,
            "shards": args.shards,
            "layout": args.layout,
            "bytes": args.checkpoint_bytes,
        },
        "objset": {
            "objset_id": objset["objset_id"],
            "manifest": f"v1/objsets/{objset['objset_id']}.json",
            "sha256": pd.sha256_bytes(pd.canonical_json(objset)),
            "lowrung": lowrung,
        },
        "runtime_env": parse_env(args.serve_env),
    }
    if pairing:
        doc["pairing_hash"] = pairing
    if args.recipe:
        doc["recipe"] = args.recipe
    return doc


def build_lowrung(specs) -> list:
    """Narrow-rung decode overrides, each pinned by its manifest DIGEST.

    Not just an id: the objects inside a rung manifest are verified against
    digests that manifest itself names, so an unverified one could serve
    arbitrary decode objects — and the rungs are the decode hot path.
    """
    out, seen = [], set()
    for spec in specs or []:
        if ":" not in spec:
            pd.die(f"--lowrung expects <max>:<objset.json>, got {spec!r}")
        max_s, path = spec.split(":", 1)
        try:
            max_n = int(max_s)
        except ValueError:
            pd.die(f"--lowrung: {max_s!r} is not a rung width")
        if max_n <= 0 or max_n in seen:
            pd.die(f"--lowrung: bad or duplicate rung width {max_s!r}")
        seen.add(max_n)
        rung = pd.load_json(Path(path))
        out.append(
            {
                "max": max_n,
                "objset_id": rung["objset_id"],
                "sha256": pd.sha256_bytes(pd.canonical_json(rung)),
            }
        )
    out.sort(key=lambda r: r["max"])
    return out


def parse_env(items) -> dict:
    env = {}
    for kv in items or []:
        if "=" not in kv:
            pd.die(f"--serve-env expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        env[k] = v
    return env


def default_label(target: dict, n: int, max_ctx: int, build_json: dict) -> str:
    """`<isa>-<sku>-<parallel>-<ctx>[-<features>]`, lowercased.

    Derived rather than typed so two builds of the same point cannot be given
    different names by accident.
    """
    feats = build_json.get("features") or {}
    on = [k for k in ("fp8_kv", "mxfp4_weights", "w8a8", "fp8_weights") if feats.get(k)]
    ctx = f"{max_ctx // 1024}k" if max_ctx % 1024 == 0 else str(max_ctx)
    parts = [target["isa"], target["sku"].lower(), f"tp{n}", ctx, *on]
    return "-".join(parts)


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("assets", nargs="?", help="a `plowc --emit devblob` output directory")
    ap.add_argument("--objset", help="objset.json from pack_objset.py")
    ap.add_argument("--namespace", default="infervisor")
    ap.add_argument("--name", help="model name within the namespace")
    ap.add_argument("--network", help="serving slug; defaults to weights.json's")
    ap.add_argument("--label", help="variant label; derived from the target when omitted")
    ap.add_argument("--hf", help="<org>/<repo> on HuggingFace")
    ap.add_argument("--revision", default="")
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--layout", default="hf-safetensors")
    ap.add_argument("--checkpoint-bytes", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=1, help="GPU count")
    ap.add_argument("--parallel-mode", default="tp")
    ap.add_argument("--max-ctx", type=int, default=0)
    ap.add_argument("--tokenizer", choices=["checkpoint", "derived"], default="checkpoint")
    ap.add_argument("--tokenizer-generator")
    ap.add_argument("--tokenizer-verified", action="store_true")
    ap.add_argument("--serve-env", action="append", help="KEY=VALUE validated with this bundle")
    ap.add_argument(
        "--lowrung",
        action="append",
        help="<max>:<objset.json> — a narrow-rung decode override, repeatable",
    )
    ap.add_argument("--recipe", help="path of the recipe that produced this")
    ap.add_argument("--plow-git", dest="plow_git", default=None)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        print("pack_bundle selftest: PASS")
        return

    for need in ("assets", "objset", "name", "hf"):
        if not getattr(args, need):
            pd.die(f"--{need.replace('_', '-')} is required")
    if args.plow_git is None:
        args.plow_git = pd.git_commit(Path(__file__).resolve().parent.parent)

    doc = build(args)
    pd.no_nix_store_paths(doc, "bundle.json")
    data = pd.canonical_json(doc)
    if args.out:
        pd.write_atomic(Path(args.out), data)
        total = sum(f["bytes"] for f in doc["files"])
        print(f"{args.out}: {doc['label']}, {len(doc['files'])} files, {pd.human(total)}")
    else:
        print(json.dumps(doc, indent=2))


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        assets = d / "assets"
        assets.mkdir()
        (assets / "model.pkt").write_bytes(b"PLOWDEV\x0b")
        (assets / "weights.json").write_text(json.dumps({"network": "kimi-k3"}))
        (assets / "build.json").write_text(
            json.dumps(
                {
                    "pairing": {"hash": "0x9fd0e880fb6fbf09"},
                    "features": {"fp8_kv": True, "mxfp4_weights": True, "w8a8": False},
                }
            )
        )
        objset = {
            "schema": pd.OBJSET_SCHEMA,
            "objset_id": "abc123",
            "target": {
                "vendor": "amd",
                "isa": "gfx942",
                "sku": "MI325X",
                "units": 304,
                "mem_bytes": 274877906944,
            },
            "objects": [{"name": "i.elf", "sha256": "a" * 64, "bytes": 1, "arms": []}],
        }
        op = d / "objset.json"
        op.write_text(json.dumps(objset))

        base = dict(
            assets=str(assets),
            objset=str(op),
            namespace="infervisor",
            name="kimi-k3",
            network=None,
            label=None,
            hf="moonshotai/Kimi-K3",
            revision="rev1",
            shards=96,
            layout="native-mxfp4",
            checkpoint_bytes=1590000000000,
            parallel=8,
            parallel_mode="tp",
            max_ctx=32768,
            tokenizer="checkpoint",
            tokenizer_generator=None,
            tokenizer_verified=False,
            serve_env=["PLOW_CTR_DBUF=1"],
            lowrung=None,
            recipe=None,
            plow_git="a" * 40,
        )
        doc = build(argparse.Namespace(**base))
        assert doc["network"] == "kimi-k3", "the slug comes from weights.json"
        assert doc["label"] == "gfx942-mi325x-tp8-32k-fp8_kv-mxfp4_weights", doc["label"]
        assert doc["runtime_env"] == {"PLOW_CTR_DBUF": "1"}
        assert doc["pairing_hash"] == "0x9fd0e880fb6fbf09"
        assert not any(f["name"] == "tokenizer.json" for f in doc["files"]), (
            "a checkpoint tokenizer must not be shipped"
        )

        # A stamped object that names a different packet is refused HERE, not at load.
        bad = json.loads(json.dumps(objset))
        bad["objects"][0]["packet_hash"] = "0xdeadbeefdeadbeef"
        op.write_text(json.dumps(bad))
        try:
            build(argparse.Namespace(**base))
            raise AssertionError("accepted a mismatched pairing")
        except pd.Fail as e:
            assert "stamped for packet" in str(e), e

        # A stamped object with no pairing hash recorded is equally unpublishable.
        (assets / "build.json").write_text(json.dumps({"features": {}}))
        try:
            build(argparse.Namespace(**base))
            raise AssertionError("accepted a stamp with no packet hash")
        except pd.Fail:
            pass

        # A general (unstamped) objset pairs with anything.
        op.write_text(json.dumps(objset))
        doc = build(argparse.Namespace(**base))
        assert "pairing_hash" not in doc

        # A derived tokenizer must be a real file and must be shipped.
        (assets / "build.json").write_text(json.dumps({"pairing": {"hash": "0x1"}, "features": {}}))
        derived = dict(base, tokenizer="derived", tokenizer_generator="scripts/x.py")
        try:
            build(argparse.Namespace(**derived))
            raise AssertionError("accepted a derived tokenizer with no file")
        except pd.Fail:
            pass
        (assets / "tokenizer.json").write_text("{}")
        doc = build(argparse.Namespace(**derived))
        assert doc["tokenizer"]["source"] == "derived"
        assert any(f["name"] == "tokenizer.json" for f in doc["files"])


main = pd.main_guard(run)

if __name__ == "__main__":
    raise SystemExit(main())
