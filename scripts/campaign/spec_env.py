"""A recipe's `[spec]` table: the speculative-decode drafter its packet carries.

    [spec]
    kind = "gemma4_assistant"            # drafter family (the packet's spec.json `family`)
    draft_dir = "{env:G26_MTP_ASSISTANT}" # HF drafter dir (config, tokenizer, weights)
    draft_revision = "<hub revision>"
    shard_sha256 = "<64 hex>"            # the re-keyed model-mtp.safetensors beside the target shards
    k_max = 7
    rows = [1, 2, 4, 8, 16]
    draft_fp8 = "all"                    # optional: FP8 draft weights (head|all, model-mtp-fp8.safetensors)
    emit = false                         # build the draft programs and span verify into the packet
    serve = false                        # PLOW_SPEC=1 in the packet's serve defaults (needs emit)

The table is the one place these are stated: `campaign.py build` derives the emit knobs and the
serve default from it, `render_recipe.py` renders them and the re-key command, and
`check_recipe.py` checks the table and the knobs the build recorded against it.
"""

from __future__ import annotations

import re

KINDS = {"gemma4_assistant": "scripts/gemma4_mtp_rekey.py"}
SHARD = "model-mtp.safetensors"
EMIT_KNOBS = (
    "PLOW_EMIT_SPAN_VERIFY",
    "PLOW_EMIT_SPEC",
    "PLOW_SPEC_DRAFT_DIR",
    "PLOW_SPEC_KMAX",
    "PLOW_SPEC_ROWS",
    "PLOW_SPEC_SHARD_SHA256",
    "PLOW_SPEC_DRAFT_FP8",
)
FIELDS = {"kind", "draft_dir", "draft_revision", "shard_sha256", "k_max", "rows", "emit", "serve"}
OPTIONAL = {"draft_fp8"}


def table(recipe: dict) -> dict | None:
    return recipe.get("spec")


def errors(recipe: dict) -> list[str]:
    s = table(recipe)
    if s is None:
        return []
    out = []
    if not isinstance(s, dict):
        return ["[spec] must be a table"]
    if extra := sorted(set(s) - FIELDS - OPTIONAL):
        out.append(f"[spec] unknown field(s) {extra}")
    if missing := sorted(FIELDS - set(s)):
        out.append(f"[spec] missing field(s) {missing}")
    if s.get("kind") not in KINDS:
        out.append(f"[spec] kind {s.get('kind')!r}: one of {sorted(KINDS)}")
    if not isinstance(s.get("draft_dir"), str) or not s.get("draft_dir"):
        out.append("[spec] draft_dir must name the drafter directory")
    if not isinstance(s.get("draft_revision"), str) or not s.get("draft_revision"):
        out.append("[spec] draft_revision must pin the drafter revision")
    if not re.fullmatch(r"[0-9a-f]{64}", str(s.get("shard_sha256", ""))):
        out.append("[spec] shard_sha256 must be 64 lowercase hex")
    k = s.get("k_max")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 15:
        out.append(f"[spec] k_max {k!r}: an integer in 1..=15")
    rows = s.get("rows")
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(r, int) and not isinstance(r, bool) and r > 0 for r in rows)
        or rows != sorted(set(rows))
    ):
        out.append(f"[spec] rows {rows!r}: ascending positive row counts")
    if s.get("draft_fp8", "head") not in ("head", "all"):
        out.append(f"[spec] draft_fp8 {s.get('draft_fp8')!r}: head or all")
    for f in ("emit", "serve"):
        if not isinstance(s.get(f), bool):
            out.append(f"[spec] {f} must be true or false")
    if s.get("serve") is True and s.get("emit") is not True:
        out.append("[spec] serve = true needs emit = true")
    stated = set((recipe.get("emit") or {}).get("env") or {}) & set(EMIT_KNOBS)
    if stated:
        out.append(f"[spec] owns {sorted(stated)}: remove them from [emit.env]")
    if "PLOW_SPEC" in ((recipe.get("serve") or {}).get("env") or {}):
        out.append("[spec] owns PLOW_SPEC: state it as [spec] serve")
    return out


def emit_env(recipe: dict) -> dict[str, str]:
    """The emit knobs `[spec]` derives (none unless `emit = true`)."""
    s = table(recipe)
    if not s or s.get("emit") is not True:
        return {}
    return {
        "PLOW_EMIT_SPAN_VERIFY": "1",
        "PLOW_EMIT_SPEC": "1",
        "PLOW_SPEC_DRAFT_DIR": str(s["draft_dir"]),
        "PLOW_SPEC_KMAX": str(s["k_max"]),
        "PLOW_SPEC_ROWS": ",".join(str(r) for r in s["rows"]),
        "PLOW_SPEC_SHARD_SHA256": str(s["shard_sha256"]),
        **({"PLOW_SPEC_DRAFT_FP8": str(s["draft_fp8"])} if "draft_fp8" in s else {}),
    }


def serve_env(recipe: dict) -> dict[str, str]:
    s = table(recipe)
    return {"PLOW_SPEC": "1"} if s and s.get("serve") is True else {}


def prepare_commands(recipe: dict) -> list[str]:
    """The drafter's checkpoint preparation: the re-key, then the FP8 twins when `draft_fp8` is set."""
    cmd = rekey_command(recipe)
    if cmd is None:
        return []
    if "draft_fp8" in recipe["spec"]:
        return [cmd, 'python3 scripts/gemma4_mtp_fp8.py "$CKPT"']
    return [cmd]


def rekey_command(recipe: dict) -> str | None:
    s = table(recipe)
    if not s or s.get("kind") not in KINDS:
        return None
    draft = re.sub(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", r'"$\1"', str(s["draft_dir"]))
    return f"python3 {KINDS[s['kind']]} {draft} \"$CKPT\""


def self_test() -> None:
    good = {
        "spec": {
            "kind": "gemma4_assistant",
            "draft_dir": "{env:D}",
            "draft_revision": "abc",
            "shard_sha256": "0" * 64,
            "k_max": 7,
            "rows": [1, 2, 4, 8, 16],
            "emit": True,
            "serve": False,
        }
    }
    assert errors(good) == [], errors(good)
    assert emit_env(good)["PLOW_SPEC_ROWS"] == "1,2,4,8,16"
    assert serve_env(good) == {}
    off = {"spec": dict(good["spec"], emit=False)}
    assert emit_env(off) == {} and errors(off) == []
    assert errors({"spec": dict(good["spec"], emit=False, serve=True)})
    assert errors({"spec": dict(good["spec"], rows=[4, 2])})
    assert errors({"spec": dict(good["spec"], k_max=16)})
    assert errors({**good, "emit": {"env": {"PLOW_EMIT_SPEC": "1"}}})
    assert errors({"spec": {k: v for k, v in good["spec"].items() if k != "shard_sha256"}})
    assert errors({}) == [] and emit_env({}) == {}
    fp8 = {"spec": dict(good["spec"], draft_fp8="all")}
    assert errors(fp8) == [] and emit_env(fp8)["PLOW_SPEC_DRAFT_FP8"] == "all"
    assert len(prepare_commands(fp8)) == 2 and len(prepare_commands(good)) == 1
    assert errors({"spec": dict(good["spec"], draft_fp8="int4")})
    assert rekey_command(good) == 'python3 scripts/gemma4_mtp_rekey.py "$D" "$CKPT"', rekey_command(good)


if __name__ == "__main__":
    self_test()
    print("spec_env selftest: PASS")
