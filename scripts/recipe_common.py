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

    # A bundle names the bytes it was validated against: a Hugging Face
    # checkpoint is `hf:<org>/<repo>` at a 40-hex commit, never a branch.
    # The same rule `plowrt pack` applies; a source that is neither `hf:`
    # nor `dist` (`local:`, `pip:`) was never packable and is left to it.
    ckpt = recipe.get("checkpoint") or {}
    source = ckpt.get("source")
    if source and source != "dist":
        bare = ":" not in source and source.count("/") == 1
        if bare:
            die(f"[checkpoint] source {source!r}: write it as `hf:{source}` and pin `revision`")
        if source.startswith("hf:"):
            rev = ckpt.get("revision") or ""
            if not re.fullmatch(r"[0-9a-f]{40}", rev):
                die(
                    f"[checkpoint] revision {rev!r} is not the 40-hex commit of {source}: a branch "
                    "can move after the build and hand a client other weights"
                )



CANONICAL_TAGS = tuple("laugh chuckle giggle sigh gasp cough clear_throat inhale exhale sniff yawn groan whisper excited happy sad angry calm serious surprised friendly casual rate pitch emphasis pause".split())


def is_speech_recipe(recipe: dict) -> bool:
    gates = recipe.get("gates") or {}
    return (recipe.get("bench", {}).get("kind") == "speech"
            or any(isinstance(k, str) and k.startswith(("tts_cer", "tts_tags")) for k in gates)
            or "speech_tags" in recipe)


def default_speech_tag_map() -> dict:
    return dict(syntax=dict(bracket=True, ssml=True, asterisk="known", markers="strip"),
                unknown="strip", literal_rewrite=[["<|", "< |"]],
                tags={k: dict(silence_ms=dict(default=400, max=3000)) if k == "pause" else dict(drop=True)
                      for k in CANONICAL_TAGS})


def speech_tag_map(value: dict) -> dict:
    def require(ok, message):
        if not ok:
            die("speech_tags.map: " + message)
    require(isinstance(value, dict), "expected a table")
    require(set(value) <= {"syntax", "unknown", "literal_rewrite", "tags"}, "unknown field")
    syntax = value.get("syntax")
    require(isinstance(syntax, dict) and set(syntax) == {"bracket", "ssml", "asterisk", "markers"}, "declare every syntax")
    require(type(syntax["bracket"]) is bool and type(syntax["ssml"]) is bool and syntax == default_speech_tag_map()["syntax"], "accepted syntax must be enabled and markers stripped")
    require(value.get("unknown") in ("strip", "literal", "passthrough", "reject"), "invalid unknown policy")
    rewrites = value.get("literal_rewrite", [])
    require(isinstance(rewrites, list) and len(rewrites) <= 16, "invalid literal_rewrite")
    for pair in rewrites:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(isinstance(x, str) and len(x.encode()) <= 64 for x in pair)
                and bool(pair[0]), "invalid literal rewrite pair")
    for _, to in rewrites:
        require("<|" not in to and not any(source in to for source, _ in rewrites), "recursive literal rewrite")
    tags = value.get("tags")
    require(isinstance(tags, dict) and set(tags) == set(CANONICAL_TAGS), "declare exactly every canonical tag")
    result = dict(syntax=dict(syntax), unknown=value["unknown"], literal_rewrite=rewrites, tags={})
    actions = {"text", "instruction", "silence_ms", "speed", "alias", "drop", "reject"}
    defaults = dict(carry=False, single_token=False, budget_chars=None, langs=[])
    for name, entry in tags.items():
        require(isinstance(entry, dict) and set(entry) <= actions | set(defaults), f"{name}: unknown field")
        active = set(entry) & actions
        require(len(active) == 1, f"{name}: exactly one action required")
        action = next(iter(active))
        data = entry[action]
        if action in ("drop", "reject", "speed"):
            require(data is True and (action != "speed" or name == "rate"), f"{name}: invalid {action}")
        elif action in ("text", "instruction", "alias"):
            require(isinstance(data, str) and bool(data) and "<|" not in data, f"{name}: invalid {action}")
            if action == "instruction":
                require(len(data.encode()) <= 1024 and ("{value}" not in data or name == "rate"), f"{name}: invalid instruction")
        elif action == "silence_ms":
            require(name == "pause" and isinstance(data, dict) and set(data) == {"default", "max"} and all(type(v) is int for v in data.values()) and data == dict(default=400, max=3000), "pause requires 400 ms default / 3000 ms max")
        require(name != "pause" or action == "silence_ms", "pause must use runtime silence")
        e = {**defaults, **entry}
        require(type(e["carry"]) is bool and type(e["single_token"]) is bool, f"{name}: invalid modifier")
        require(not (e["carry"] or e["single_token"] or e["budget_chars"] is not None) or action == "text", f"{name}: native modifiers need text")
        budget = e["budget_chars"]
        require(budget is None or type(budget) is int and budget >= 0, f"{name}: invalid budget")
        langs = e["langs"]
        require(isinstance(langs, (dict, list)) and len(langs) <= 32 and all(isinstance(l, str) and re.fullmatch(r"[a-z-]{1,16}", l) for l in langs), f"{name}: invalid langs")
        require(not isinstance(langs, dict) or action == "text" and bool(langs), f"{name}: language texts require native text")
        require(action != "silence_ms" or not langs, "pause cannot be language-limited")
        if action == "text":
            require(all(isinstance(t, str) and 0 < len(t) <= min(budget if budget is not None else 64, 64) and "<|" not in t
                        for t in [data] + (list(langs.values()) if isinstance(langs, dict) else [])), f"{name}: invalid native text")
        result["tags"][name] = e
    for name in tags:
        seen, target = set(), name
        while "alias" in tags.get(target, {}):
            require(target not in seen, f"{name}: alias cycle")
            seen.add(target)
            target = tags[target]["alias"]
            require(target in tags, f"{name}: unknown alias {target}")
    return result


def legacy_speech_tag_notice(recipe: dict, path: Path | None) -> str | None:
    if "speech_tags" in recipe or not is_speech_recipe(recipe) or path is None:
        return None
    path = path.resolve()
    if path.name == "recipe.toml" and (path.parent / "build-record.json").is_file():
        record = load_json(path.parent / "build-record.json")
        if record.get("recipe_snapshot") == path.name:
            path = Path(record.get("recipe", "")).resolve()
    inventory = load_json(Path(__file__).resolve().parents[1] / "crates/plow-asset/src/speech_tag_legacy.json")
    for identity, date in inventory.items():
        parts = Path(identity).parts
        if path.parts[-len(parts):] == parts:
            return f"speech_tags: legacy recipe without tag policy (grandfathered {date})"
    return None


def speech_tag_policy(recipe: dict, path: Path | None = None) -> dict | None:
    policy = recipe.get("speech_tags")
    if policy is None:
        if is_speech_recipe(recipe) and not legacy_speech_tag_notice(recipe, path):
            die("speech_tags: missing policy (declare the emitted map or a measured waiver)")
        return None
    if not isinstance(policy, dict) or set(policy) - {"version", "profile", "map", "waiver", "evidence", "native_sweep"}:
        die("speech_tags: invalid policy fields")
    if type(policy.get("version")) is not int or policy["version"] != 1:
        die("speech_tags: version must be 1")
    if ("map" in policy) == ("waiver" in policy):
        die("speech_tags: declare exactly one map or waiver")
    evidence = policy.get("evidence", [])
    if not isinstance(evidence, list) or any(not isinstance(e, str) or not e.strip() for e in evidence):
        die("speech_tags: evidence must list measurement references")
    if "waiver" in policy:
        reason = policy["waiver"]
        if (not isinstance(reason, str) or not reason.strip() or not evidence
                or re.search(r"not (yet )?measured|unmeasured|\btodo\b|\btbd\b", reason, re.I)
                or "profile" in policy or "native_sweep" in policy):
            die("speech_tags: waiver needs a measured reason and evidence, without a map/profile/sweep")
    else:
        if not isinstance(policy.get("profile"), str) or not policy["profile"].strip():
            die("speech_tags: map needs an explicit profile name")
        speech_tag_map(policy["map"])
        if recipe.get("status") == "validated" and any("text" in e for e in policy["map"]["tags"].values()) and not policy.get("native_sweep"):
            die("speech_tags: a validated native map requires its measured native_sweep report")
        sweep = policy.get("native_sweep")
        if sweep is not None and (not isinstance(sweep, str) or not sweep.strip()):
            die("speech_tags: native_sweep must name its measured report")
    return policy


def speech_tag_policy_lines(recipe: dict) -> list[str]:
    if "speech_tags" not in recipe:
        return []
    policy = speech_tag_policy(recipe)
    if policy is None:
        return []
    return ["# Speech tag policy", *["# " + line for line in json.dumps(policy, ensure_ascii=False, sort_keys=True, indent=2).splitlines()]]


def speech_pipelines(packet: Path) -> list[dict]:
    from packet_sections import read_section

    raw = read_section(packet, "packet_pipeline.json")
    if raw is None:
        return []
    metadata = json.loads(raw)
    if type(metadata.get("version")) is not int or metadata["version"] != 1 or not isinstance(metadata.get("pipelines"), list):
        die("speech_tags: unsupported packet pipeline metadata")
    return [p for p in metadata["pipelines"] if p.get("driver", "").startswith("tts.")]


def check_packet_speech_tags(packet: Path, recipe: dict | None, path: Path | None = None) -> list[tuple[str, str]]:
    pipelines = speech_pipelines(packet)
    if not pipelines:
        if recipe and is_speech_recipe(recipe):
            die("speech_tags: recipe is speech but packet has no speech pipeline")
        return [("ok", "speech_tags: n/a (no speech pipeline)")]
    if recipe is None:
        die("speech_tags: speech packet needs its recipe (../recipe.toml)")
    notice = legacy_speech_tag_notice(recipe, path)
    if notice:
        return [("NOTICE", notice)]
    policy = speech_tag_policy(recipe, path)
    if policy is None:
        die("speech_tags: speech packet needs a declared policy")
    if "waiver" in policy:
        return [("WARN", "speech_tags: waived: " + policy["waiver"] + "; evidence: " + "; ".join(policy["evidence"]))]
    expected = speech_tag_map(policy["map"])
    for pipeline in pipelines:
        strings, params = pipeline.get("strings", {}), pipeline.get("parameters", {})
        if "text.tags" not in strings or type(params.get("text.tags_version")) is not int or params["text.tags_version"] != policy["version"]:
            die(f"speech_tags: {pipeline['name']}: emitted text.tags/version absent or unsupported; re-emit")
        observed = speech_tag_map(json.loads(strings["text.tags"]))
        if observed != expected:
            differences = [k for k in expected if observed[k] != expected[k]]
            die(f"speech_tags: {pipeline['name']}: emitted map differs from recipe ({', '.join(differences)})")
    return [("ok", f"speech_tags: emitted map matches recipe profile {policy['profile']} ({len(pipelines)} pipeline(s))")]


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
