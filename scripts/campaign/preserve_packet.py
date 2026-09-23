#!/usr/bin/env python3
"""Preserve what recreates a packet, and diff two packets' build environments.

WHY. A packet lives under /opt/dlami/nvme/tmp, which is scratch: the cubins, `model.pkt` and
`build.json` do not survive a cleanup, and with them go the only record of what produced them.
The recipe in the repo is NOT sufficient on its own -- a packet can be built from a recipe plus
`--env` overrides, and older packets were built from recipes that have since changed. `build.json`
carries the missing half:

  * `emit_config.replay`        -- the env assignments that reproduce the configuration, with
                                   defaults omitted so a replay follows the tree's defaults
  * `emit_config.unrecorded_env`-- env read OUTSIDE EmitConfig (e.g. PLOW_SEG_*), which the
                                   replay block cannot describe
  * `tuning`                    -- shape constants the tune store resolved (gv_mm_max, xreg_k, ...)
  * `knobs.registry_digest`     -- verifies a rebuild resolved the same knob registry

`preserve` writes those into perf-data/packets/<label>.json together with the matched recipe, the
tree commit and sha256 of every emitted artifact, and copies the packet's `cublaslt_algos.jsonl`
(a per-packet probe result that `--no-probe` reuses, and which exists nowhere else).

`diff` is the other half and the reason this file exists at all. Three times in one campaign a
cross-packet delta was written up as a single-variable A/B -- GV_MM_MAX, the ring tax in two
recipe headers, and the req_chunk result -- because nobody compared the two packets' build
environments. `diff A B` prints exactly that, and `--expect KEY` exits non-zero unless KEY is the
ONLY difference, so a harness can assert it before spending a lease.

    preserve_packet.py preserve <packet-dir> [--label NAME]
    preserve_packet.py diff <packet-dir> <packet-dir> [--expect PLOW_MAX_REQUEST_CHUNK]
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    tomllib = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "perf-data" / "packets"
RECIPES = ROOT / "scripts" / "campaign" / "recipes"


def build_env(assets: pathlib.Path):
    """(replay, unrecorded, whole build.json). The two env maps are kept separate because only
    `replay` is replayable as-is; `unrecorded_env` reports values for env read outside EmitConfig."""
    d = json.loads((assets / "build.json").read_text())
    ec = d.get("emit_config", {}) or {}
    replay = dict(ec.get("replay") or {})
    unrec = dict(ec.get("unrecorded_env") or {})
    return replay, unrec, d


def merged_env(assets: pathlib.Path):
    replay, unrec, d = build_env(assets)
    env = dict(replay)
    env.update({f"(unrecorded){k}": v for k, v in unrec.items()})
    return env, d


def match_recipe(replay: dict):
    """Best-matching recipe by its [emit.env] block. Returns (path, score, missing, extra)."""
    if tomllib is None:
        return None, 0, [], []
    best = (None, -1, [], [])
    for rp in sorted(RECIPES.glob("*.toml")):
        try:
            r = tomllib.loads(rp.read_text())
        except Exception:
            continue
        env = (r.get("emit") or {}).get("env") or {}
        if not env:
            continue
        agree = sum(1 for k, v in env.items() if str(replay.get(k)) == str(v))
        missing = [k for k in env if k not in replay]
        extra = [k for k in replay if k not in env]
        if agree > best[1]:
            best = (rp, agree, missing, extra)
    return best


def sha256(p: pathlib.Path, cap=None):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def cmd_preserve(a):
    pkt = pathlib.Path(a.packet).resolve()
    assets = pkt / "assets" if (pkt / "assets").is_dir() else pkt
    if not (assets / "build.json").is_file():
        sys.exit(f"no build.json under {assets}")
    label = a.label or pkt.name
    replay, unrec, d = build_env(assets)
    recipe, score, missing, extra = match_recipe(replay)

    cell = {}
    if recipe and tomllib:
        cell = (tomllib.loads(recipe.read_text()).get("cell") or {})

    # Overrides = replay keys whose value differs from the matched recipe's [emit.env].
    overrides = {}
    if recipe and tomllib:
        renv = (tomllib.loads(recipe.read_text()).get("emit") or {}).get("env") or {}
        overrides = {k: v for k, v in replay.items() if str(renv.get(k)) != str(v)}

    artifacts = []
    for f in sorted(assets.iterdir()):
        if f.is_symlink() or not f.is_file():
            continue
        if f.suffix in (".cubin", ".pkt", ".h") or f.name in ("build.json", "weights.json",
                                                              "cublaslt_algos.jsonl"):
            artifacts.append({"name": f.name, "bytes": f.stat().st_size,
                              "sha256": sha256(f) if a.hash else None})

    rebuild = ["nix develop --command python3 scripts/campaign/campaign.py build",
               str(recipe.relative_to(ROOT)) if recipe else "<recipe>",
               f"--out <dir> --no-probe"]
    for k, v in sorted(overrides.items()):
        rebuild.append(f"--env {k}={v}")

    rec = {
        "label": label,
        "preserved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "preserved_from": str(pkt),
        # HEAD when this record was written -- NOT necessarily the commit that built the packet.
        # build.json carries no commit, so the build commit can only be bounded by build_mtime.
        "preserved_at_commit": git("rev-parse", "HEAD"),
        "preserved_at_commit_is_build_commit": False,
        "tree_dirty_at_preservation": bool(git("status", "--porcelain")),
        "build_mtime": datetime.datetime.fromtimestamp(
            (assets / "build.json").stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        "recipe": str(recipe.relative_to(ROOT)) if recipe else None,
        "recipe_match": {"agreeing_keys": score, "recipe_keys_absent_from_packet": missing,
                         "packet_keys_absent_from_recipe": extra},
        "recipe_overrides": overrides,
        "cell": {k: cell.get(k) for k in ("name", "model", "revision", "gpu", "arch", "n_cu",
                                          "max_ctx", "precision") if k in cell},
        "arch": d.get("arch"),
        "n_cu": d.get("n_cu"),
        "shapes": d.get("shapes"),
        "precision": d.get("precision"),
        "features": {k: v for k, v in (d.get("features") or {}).items() if v},
        "emit_env_replay": replay,
        "unrecorded_env": unrec,
        "tuning": d.get("tuning"),
        "registry_digest": (d.get("knobs") or {}).get("registry_digest"),
        "knobs_K": (d.get("knobs") or {}).get("K"),
        "lean": d.get("lean"),
        "artifacts": artifacts,
        "rebuild_command": " ".join(rebuild),
        "note": ("Recreate: run rebuild_command, then verify registry_digest, tuning and shapes "
                 "match this record -- a mismatch means the tree moved under you, which is the "
                 "signal to check out a commit near build_mtime and retry. tuning/ in the repo "
                 "supplies the shape constants; cublaslt_algos.jsonl beside this file is the "
                 "packet's own probe result, which --no-probe reuses. recipe/recipe_overrides "
                 "are DERIVED by matching emit_env_replay against the recipes present today, so "
                 "recipe_match.agreeing_keys and the *_absent_from_* lists say how much to trust "
                 "them; emit_env_replay + unrecorded_env are the ground truth."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{label}.json"
    dst.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n")
    print(f"wrote {dst.relative_to(ROOT)}")

    algos = assets / "cublaslt_algos.jsonl"
    if algos.is_file():
        shutil.copy2(algos, OUT / f"{label}.cublaslt_algos.jsonl")
        print(f"wrote {(OUT / f'{label}.cublaslt_algos.jsonl').relative_to(ROOT)} "
              f"({algos.stat().st_size} B)")
    cfg = assets / "plow_config.h"
    if cfg.is_file() and a.config:
        shutil.copy2(cfg, OUT / f"{label}.plow_config.h")
        print(f"wrote {(OUT / f'{label}.plow_config.h').relative_to(ROOT)}")

    print(f"  recipe   {rec['recipe']}  (agreeing keys {score})")
    print(f"  overrides {overrides or '(none)'}")
    print(f"  digest   {rec['registry_digest']}")
    print(f"  rebuild  {rec['rebuild_command']}")


def cmd_diff(a):
    pa, pb = (pathlib.Path(x).resolve() for x in (a.a, a.b))
    aa = pa / "assets" if (pa / "assets").is_dir() else pa
    ab = pb / "assets" if (pb / "assets").is_dir() else pb
    ea, da = merged_env(aa)
    eb, db = merged_env(ab)
    keys = sorted(set(ea) | set(eb))
    diffs = [k for k in keys if ea.get(k) != eb.get(k)]
    print(f"{pa.name} vs {pb.name}")
    for k in diffs:
        print(f"   {k}: {ea.get(k, '—')} -> {eb.get(k, '—')}")
    print(f"   {len(diffs)} differing key(s) of {len(keys)}")
    ta, tb = json.dumps(da.get("tuning")), json.dumps(db.get("tuning"))
    if ta != tb:
        print(f"   TUNING DIFFERS\n     {ta}\n     {tb}")
    ga = (da.get("knobs") or {}).get("registry_digest")
    gb = (db.get("knobs") or {}).get("registry_digest")
    if ga != gb:
        print(f"   REGISTRY DIGEST DIFFERS: {ga} vs {gb}")
    if a.expect:
        ok = diffs == [a.expect]
        print(f"   {'SINGLE VARIABLE (' + a.expect + ')' if ok else '*** NOT single-variable ***'}")
        return 0 if ok else 2
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("preserve", help="write a recreate record for a packet")
    p.add_argument("packet")
    p.add_argument("--label")
    p.add_argument("--no-hash", dest="hash", action="store_false",
                   help="skip sha256 of artifacts (faster on a 44 MB model.pkt)")
    p.add_argument("--config", action="store_true", help="also copy plow_config.h")
    p.set_defaults(f=cmd_preserve)
    q = sp.add_parser("diff", help="diff two packets' build environments")
    q.add_argument("a"); q.add_argument("b")
    q.add_argument("--expect", metavar="KEY",
                   help="exit 2 unless KEY is the only differing key")
    q.set_defaults(f=cmd_diff)
    a = ap.parse_args()
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()
