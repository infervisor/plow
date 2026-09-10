#!/usr/bin/env python3
"""Turn a built object directory into an `objset.json`.

Input is what `scripts/build_gfx942.sh` already leaves behind: the `.elf` files
and the `build_defines.json` it writes from the same `$ROWS` the compile loop
uses, "so the recorded -D set cannot drift from the compiled one". This script
adds identity — a digest per object, the marker symbols read out of `.symtab`,
and the packet-pairing stamp when the object carries one.

  pack_objset.py build-amd/k3-mi325x --target gfx942:MI325X:304:274877906944 \\
      --toolchain rocm-7.14.0-nix --out objset.json

The objset id is a hash over (isa, sku, toolchain, defines digest, plow_git), so
two builds that differ in any `-D` cannot collide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plow_dist as pd


def parse_target(s: str) -> dict:
    parts = s.split(":")
    if len(parts) != 4:
        pd.die(
            f"--target must be `<isa>:<sku>:<units>:<mem_bytes>`, got {s!r}"
        )
    isa, sku, units, mem = parts
    vendor = "amd" if isa.startswith("gfx") else "nvidia" if isa.startswith("sm_") else "cpu"
    try:
        return {
            "vendor": vendor,
            "isa": isa,
            "sku": sku,
            "units": int(units),
            "mem_bytes": int(mem),
        }
    except ValueError:
        pd.die(f"--target: units and mem_bytes must be integers, got {units!r} {mem!r}")


def build(args) -> dict:
    objdir = Path(args.objects)
    if not objdir.is_dir():
        pd.die(f"{objdir} is not a directory")

    defines_path = objdir / "build_defines.json"
    defines = pd.load_json(defines_path) if defines_path.exists() else {}
    if not defines:
        print(
            f"warning: {defines_path} is absent, so this objset records no -D set and the "
            f"recipe checker cannot verify it",
        )

    elfs = sorted(p for p in objdir.glob("*.elf") if p.is_file())
    if not elfs:
        pd.die(f"{objdir}: no *.elf found")

    objects = []
    stamps = set()
    for p in elfs:
        syms = pd.elf_symbols(p)
        stamp = pd.packet_hash_from_symbols(syms)
        stamps.add(stamp)
        objects.append(
            {
                "name": p.name,
                "sha256": pd.sha256_file(p),
                "bytes": p.stat().st_size,
                # Only the plow_* markers matter to the loader; recording every
                # symbol would bloat the manifest without adding a check.
                "arms": [s for s in syms if not s.startswith("plow_packet_hash_")],
                **({"packet_hash": stamp} if stamp else {}),
            }
        )

    if len(stamps - {None}) > 1:
        pd.die(
            "objects in this directory are stamped for different packets: "
            f"{sorted(s for s in stamps if s)}. That is a mixed build, not an objset."
        )

    target = parse_target(args.target)
    defines_digest = pd.sha256_bytes(pd.canonical_json(defines))
    # The CONTENT is part of the identity, not just the build inputs. Keying on
    # inputs alone collides whenever they do not fully determine the output —
    # a `PLOW_ROWS_ONLY` partial directory and a full build share their defines,
    # and a directory with no `build_defines.json` shares them with every other
    # such directory. Two objsets with one id would overwrite each other at
    # `v1/objsets/<id>.json`.
    content_digest = pd.sha256_bytes(
        pd.canonical_json([[o["name"], o["sha256"]] for o in objects])
    )
    objset_id = pd.short_id(
        target["isa"],
        target["sku"],
        args.toolchain,
        defines_digest,
        content_digest,
        args.plow_git,
    )

    env = {}
    for kv in args.env or []:
        if "=" not in kv:
            pd.die(f"--env expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        env[k] = v

    return {
        "schema": pd.OBJSET_SCHEMA,
        "objset_id": objset_id,
        "target": target,
        "toolchain": args.toolchain,
        "plow_git": args.plow_git,
        "script": args.script,
        "env": env,
        "defines": defines,
        "objects": objects,
    }


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("objects", nargs="?", help="directory of built .elf objects")
    ap.add_argument("--target", help="<isa>:<sku>:<units>:<mem_bytes>")
    ap.add_argument("--toolchain", default="rocm-7.14.0-nix")
    ap.add_argument("--plow-git", dest="plow_git", default=None, help="full 40-hex commit")
    ap.add_argument("--script", default="scripts/build_gfx942.sh")
    ap.add_argument("--env", action="append", help="KEY=VALUE the build was driven with")
    ap.add_argument(
        "--readelf",
        default=pd.default_readelf(),
        help="readelf used to read each object's symbol table "
        "(default: $PLOW_READELF, then llvm-readelf or readelf on PATH)",
    )
    ap.add_argument("--out", help="write objset.json here (default: stdout)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    pd.set_readelf(args.readelf)

    if args.self_test:
        self_test()
        print("pack_objset selftest: PASS")
        return

    if not args.objects or not args.target:
        pd.die("give an objects directory and --target")
    if args.plow_git is None:
        args.plow_git = pd.git_commit(Path(__file__).resolve().parent.parent)

    doc = build(args)
    pd.no_nix_store_paths(doc, "objset.json")
    data = pd.canonical_json(doc)
    if args.out:
        pd.write_atomic(Path(args.out), data)
        print(f"{args.out}: objset {doc['objset_id']}, {len(doc['objects'])} objects")
    else:
        print(json.dumps(doc, indent=2))


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "a.elf").write_bytes(b"\x7fELF-a")
        (d / "build_defines.json").write_text(json.dumps({"a": "-DX=1"}))

        # Stub readelf out: the symbol table is the runtime's business, and the
        # identity logic is what this test is about.
        real = pd.elf_symbols
        pd.elf_symbols = lambda p: ["plow_k3_arms_1", "plow_gemv_walk_1"]
        try:
            args = argparse.Namespace(
                objects=str(d),
                target="gfx942:MI325X:304:274877906944",
                toolchain="rocm-7.14.0-nix",
                plow_git="a" * 40,
                script="scripts/build_gfx942.sh",
                env=["PLOW_DECODE_BATCH=32"],
            )
            one = build(args)
            assert one["schema"] == pd.OBJSET_SCHEMA
            assert one["objects"][0]["arms"] == ["plow_k3_arms_1", "plow_gemv_walk_1"]
            assert "packet_hash" not in one["objects"][0], "no stamp means general"
            assert one["env"] == {"PLOW_DECODE_BATCH": "32"}

            # The id must move when any -D moves, and not otherwise.
            two = build(args)
            assert two["objset_id"] == one["objset_id"], "identity must be stable"
            (d / "build_defines.json").write_text(json.dumps({"a": "-DX=2"}))
            three = build(args)
            assert three["objset_id"] != one["objset_id"], "a changed -D must change the id"

            # Two directories with the SAME inputs but DIFFERENT objects must not
            # share an id: they would overwrite each other in the store. This is
            # the `PLOW_ROWS_ONLY` shape — a rung override compiled from the same
            # defines as the full set.
            (d / "build_defines.json").write_text(json.dumps({"a": "-DX=1"}))
            (d / "a.elf").write_bytes(b"\x7fELF-different")
            four = build(args)
            assert four["objset_id"] != one["objset_id"], (
                "different objects must not collide on one id"
            )
            (d / "a.elf").write_bytes(b"\x7fELF-a")
            assert build(args)["objset_id"] == one["objset_id"], "identity must be stable"

            # A half-stamped object is a broken build, not a general one.
            pd.elf_symbols = lambda p: ["plow_packet_hash_lo_0000dead"]
            try:
                build(args)
                raise AssertionError("accepted a half stamp")
            except pd.Fail:
                pass

            # Both halves compose into the pairing hash the bundle must match.
            pd.elf_symbols = lambda p: [
                "plow_packet_hash_lo_fb6fbf09",
                "plow_packet_hash_hi_9fd0e880",
            ]
            stamped = build(args)
            assert stamped["objects"][0]["packet_hash"] == "0x9fd0e880fb6fbf09", stamped
        finally:
            pd.elf_symbols = real


main = pd.main_guard(run)

if __name__ == "__main__":
    raise SystemExit(main())
