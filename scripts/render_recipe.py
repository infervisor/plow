#!/usr/bin/env python3
"""Render a recipe's command blocks into its prose document.

A recipe has two halves. The TOML is what builds and publishes consume; the
Markdown is what a person reads. Keeping the commands in both by hand is how
`docs/amd/gemma4-31b-mi300x.md` ended up superseding its own `## Build and run`.

So the commands are GENERATED into the `.md` between markers, and everything
outside the markers is hand-written and untouched:

    <!-- plow:recipe:objects -->
    ```bash
    ...generated...
    ```
    <!-- /plow:recipe:objects -->

CI renders and diffs, so a recipe change that is not reflected in its prose is a
failing check rather than a stale document.

  render_recipe.py recipes/infervisor/kimi-k3/<label>.toml --check
  render_recipe.py recipes/infervisor/kimi-k3/<label>.toml --write
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import plow_dist as pd

SECTIONS = ("prepare", "objects", "emit", "serve", "gates")


def env_prefix(env: dict) -> list[str]:
    return [f"  {k}={v} \\" for k, v in sorted(env.items())]


def block_prepare(r: dict) -> str:
    steps = r.get("prepare") or []
    if not steps:
        return "# This model needs no preparation step."
    out = []
    for s in steps:
        shell = s.get("shell", "nix develop")
        args = " ".join(s.get("args") or [])
        out.append(f"{shell} --command python3 {s['tool']} \\\n  {args}")
        if s.get("produces"):
            out.append(f"# produces: {', '.join(s['produces'])}")
        out.append("")
    return "\n".join(out).rstrip()


def block_objects(r: dict) -> str:
    o = r.get("objects") or {}
    if not o:
        return "# This target builds no interpreter objects (CPU has no emit target)."
    out = ["nix develop --command env \\"]
    out += env_prefix(o.get("env") or {})
    out.append(f"  {o['script']} \\")
    out.append("  $OBJDIR")
    for rung in o.get("lowrung") or []:
        out.append("")
        out.append(f"# rung <= {rung['max']}: a PARTIAL directory, valid only as an override")
        out.append("nix develop --command env \\")
        out += env_prefix(rung.get("env") or {})
        out.append(f"  {o['script']} \\")
        out.append(f"  $OBJDIR/lowrung{rung['max']}")
    return "\n".join(out)


def block_emit(r: dict) -> str:
    e = r.get("emit") or {}
    if not e:
        return "# This recipe does not emit: see `refusal`."
    out = ["nix develop --command env \\"]
    out += env_prefix(e.get("env") or {})
    out.append("  ./target/release/plowc \\")
    out.append("  --hf-dir $CKPT \\")
    args = e.get("args") or []
    for i in range(0, len(args), 2):
        pair = " ".join(args[i : i + 2])
        out.append(f"  {pair} \\")
    out.append("  --out $ASSETS")
    return "\n".join(out)


def block_serve(r: dict) -> str:
    s = r.get("serve") or {}
    ref = f"{r.get('namespace', 'infervisor')}/{r['name']}:{r['label']}"
    out = [
        "# From the distribution — resolves the variant for this machine:",
        f"plowrt load {ref}",
        f"plowrt serve --model {ref} --port 8080",
        "",
        "# Equivalently, from a locally built asset directory:",
    ]
    if s.get("env"):
        out.append("nix develop --command env \\")
        out += env_prefix(s["env"])
        out.append("  ./target/release/plowrt serve --assets $ASSETS --port 8080")
    else:
        out.append("./target/release/plowrt serve --assets $ASSETS --port 8080")
    if s.get("lowrung_from_objects"):
        out.append("")
        out.append("# The rung overrides travel in the bundle; `serve --model` wires them")
        out.append("# automatically, so PLOW_HSACO_LOWRUNG needs no absolute build-host path.")
    return "\n".join(out)


def block_gates(r: dict) -> str:
    gates = r.get("gates") or []
    if not gates:
        return "# This recipe declares no gates."
    out = []
    for g in gates:
        out.append(f"# {g['name']}")
        if g.get("cmd"):
            env = " ".join(f"{k}={v}" for k, v in sorted((g.get("env") or {}).items()))
            out.append(f"{('env ' + env + ' ') if env else ''}{g['cmd']}")
        if g.get("expect"):
            out.append(f"# expect: {g['expect']}")
        out.append("")
    return "\n".join(out).rstrip()


BLOCKS = {
    "prepare": block_prepare,
    "objects": block_objects,
    "emit": block_emit,
    "serve": block_serve,
    "gates": block_gates,
}


def render(md: str, recipe: dict) -> str:
    """Replace each marked region; leave everything else alone."""
    for name in SECTIONS:
        open_m = f"<!-- plow:recipe:{name} -->"
        close_m = f"<!-- /plow:recipe:{name} -->"
        if open_m not in md:
            continue
        body = BLOCKS[name](recipe)
        replacement = f"{open_m}\n```bash\n{body}\n```\n{close_m}"
        md = re.sub(
            re.escape(open_m) + r".*?" + re.escape(close_m),
            lambda _m: replacement,
            md,
            flags=re.S,
        )
    return md


def default_doc(recipe_path: Path) -> Path:
    return recipe_path.with_suffix(".md")


def scaffold(recipe: dict) -> str:
    """A new prose file: the K3 doc's shape, with every block marked."""
    ns = recipe.get("namespace", "infervisor")
    lines = [
        f"# {recipe['name']} — {recipe['label']}",
        "",
        f"Status: **{recipe.get('status', 'emits')}**. "
        f"Reference: `{ns}/{recipe['name']}:{recipe['label']}`.",
        "",
        "The command blocks below are generated from the recipe TOML by",
        "`scripts/render_recipe.py`; edit the TOML, not the blocks. Prose outside",
        "the markers is hand-written.",
        "",
        "## 1. Prepare the checkpoint",
        "",
        "<!-- plow:recipe:prepare -->",
        "<!-- /plow:recipe:prepare -->",
        "",
        "## 2. Build the interpreter objects",
        "",
        "<!-- plow:recipe:objects -->",
        "<!-- /plow:recipe:objects -->",
        "",
        "## 3. Emit the packet",
        "",
        "<!-- plow:recipe:emit -->",
        "<!-- /plow:recipe:emit -->",
        "",
        "## 4. Gates",
        "",
        "<!-- plow:recipe:gates -->",
        "<!-- /plow:recipe:gates -->",
        "",
        "## 5. Serve",
        "",
        "<!-- plow:recipe:serve -->",
        "<!-- /plow:recipe:serve -->",
        "",
    ]
    return "\n".join(lines)


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recipe", nargs="*")
    ap.add_argument("--doc", help="the markdown file (default: the recipe's .md sibling)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        print("render_recipe selftest: PASS")
        return
    if not args.recipe:
        pd.die("give at least one recipe")

    stale = 0
    for rp in args.recipe:
        p = Path(rp)
        recipe = pd.load_toml(p)
        doc = Path(args.doc) if args.doc else default_doc(p)
        before = doc.read_text() if doc.exists() else scaffold(recipe)
        after = render(before, recipe)
        if after == before and doc.exists():
            print(f"{doc}: up to date")
            continue
        if args.check:
            print(f"{doc}: STALE — run `render_recipe.py {rp} --write`")
            stale += 1
            continue
        if args.write:
            pd.write_atomic(doc, after.encode())
            print(f"{doc}: written")
        else:
            print(after)
    if stale:
        pd.die(f"{stale} document(s) out of date")


def self_test() -> None:
    recipe = {
        "namespace": "infervisor",
        "name": "kimi-k3",
        "label": "gfx942-mi325x-tp8",
        "status": "validated",
        "prepare": [
            {
                "tool": "scripts/kimi_k3_tokenizer.py",
                "shell": "nix develop .#quantize",
                "args": ["--model", "$CKPT", "--verify"],
                "produces": ["tokenizer.json"],
            }
        ],
        "objects": {
            "script": "scripts/build_gfx942.sh",
            "env": {"PLOW_DECODE_BATCH": "32"},
            "lowrung": [{"max": 1, "env": {"PLOW_DECODE_BATCH": "1"}}],
        },
        "emit": {
            "env": {"PLOW_FP8_KV": "1"},
            "args": ["--arch", "gfx942", "--num-gpus", "8"],
        },
        "serve": {"env": {"PLOW_CTR_DBUF": "1"}, "lowrung_from_objects": True},
        "gates": [{"name": "gsm8k", "expect": "197/200"}],
    }

    md = scaffold(recipe)
    out = render(md, recipe)

    # Every block is filled from the TOML.
    assert "PLOW_DECODE_BATCH=32" in out
    assert "lowrung1" in out, "the rung override must appear"
    assert "PLOW_FP8_KV=1" in out
    assert "--arch gfx942" in out
    assert "plowrt load infervisor/kimi-k3:gfx942-mi325x-tp8" in out
    assert "expect: 197/200" in out

    # Rendering is idempotent, which is what makes `--check` meaningful.
    assert render(out, recipe) == out

    # Prose outside the markers survives.
    edited = out.replace(
        "## 5. Serve", "## 5. Serve\n\nA hand-written paragraph that must survive.\n"
    )
    again = render(edited, recipe)
    assert "hand-written paragraph that must survive" in again

    # A changed recipe changes the rendering — that is the staleness signal.
    changed = dict(recipe, emit={"env": {"PLOW_FP8_KV": "0"}, "args": []})
    assert render(out, changed) != out

    # A refused recipe renders without inventing an emit command.
    refused = {
        "namespace": "infervisor",
        "name": "deepseek-v4",
        "label": "gfx942-mi300x-tp8",
        "status": "refused",
    }
    out = render(scaffold(refused), refused)
    assert "does not emit" in out
    assert "no preparation step" in out


main = pd.main_guard(run)

if __name__ == "__main__":
    raise SystemExit(main())
