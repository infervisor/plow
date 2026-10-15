"""Per-rung emit knob overrides: recipe `[emit.rung_env]` <-> `PLOW_EMIT_RUNG_ENV`.

    [emit.rung_env.prefill.128]
    PLOW_NO_GLU_FUSE = "0"

renders to `prefill:128:PLOW_NO_GLU_FUSE=0` entries joined by `;` (sorted: phase, rows, knob),
which plowc resolves per rung (crates/devgen/src/emit_config/rung.rs). Only knobs the devgen
registry marks `.per_rung(...)` for that phase are accepted; devgen refuses the rest too.
Stdlib only: campaign.py, check_recipe.py and render_recipe.py all import it.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KNOB_SPEC = REPO / "crates/devgen/src/knob_spec.rs"
ENV = "PLOW_EMIT_RUNG_ENV"
PHASES = ("prefill", "decode")
TRUE = ("1", "true", "yes", "on")
FALSE = ("0", "false", "no", "off")

Overrides = dict  # {(phase, rows): {env: value}}


class RungEnvError(ValueError):
    pass


def registry(src: str | None = None) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """(every emit env -> knob id, per-rung env -> phases) from the devgen knob registry source."""
    src = KNOB_SPEC.read_text() if src is None else src
    consts = {m[1]: tuple(re.findall(r'"([a-z]+)"', m[2]))
              for m in re.finditer(r"const ([A-Z_]+): &\[&str\] = &\[([^\]]*)\];", src)}
    emit, rungs = {}, {}
    for line in src.splitlines():
        m = re.search(r'KnobSpec::new\("(emit\.[a-z0-9_]+)", Some\("(PLOW_[A-Z0-9_]+)"\)', line)
        if not m:
            continue
        emit[m[2]] = m[1]
        r = re.search(r"\.per_rung\((?:&\[([^\]]*)\]|([A-Z_]+))\)", line)
        if r:
            rungs[m[2]] = tuple(re.findall(r'"([a-z]+)"', r[1])) if r[1] is not None else consts.get(r[2], ())
    return emit, rungs


def check(over: Overrides, src: str | None = None) -> list[str]:
    """Problems with `over`: unregistered or packet-wide knobs, unknown phases, bad rows/values."""
    emit, rungs = registry(src)
    errs = []
    for (phase, rows), knobs in sorted(over.items()):
        if phase not in PHASES:
            errs.append(f"{phase}:{rows}: phase must be one of {', '.join(PHASES)}")
        if not isinstance(rows, int) or rows <= 0:
            errs.append(f"{phase}:{rows}: rows must be a positive integer")
        for env, value in sorted(knobs.items()):
            where = f"{phase}:{rows}:{env}"
            if env not in emit:
                errs.append(f"{where}: not a registered emit knob")
            elif phase not in rungs.get(env, ()):
                errs.append(f"{where}: {emit[env]} is packet-wide (no .per_rung({phase!r}) in knob_spec)")
            if not isinstance(value, str) or not value or any(c in value for c in ";:="):
                errs.append(f"{where}: value must be a non-empty string without ; : =")
    return errs


def recipe_source_errors(recipe: dict) -> list[str]:
    """Ways a recipe could pass PLOW_EMIT_RUNG_ENV other than [emit.rung_env]; each would win
    silently over the rendered table."""
    errs = []
    for where, env in (("[emit.env]", (recipe.get("emit") or {}).get("env") or {}),
                       ("[emit_roles.env]", (recipe.get("emit_roles") or {}).get("env") or {})):
        if ENV in env:
            errs.append(f"{ENV} set in {where}: state rung overrides as [emit.rung_env]")
    if any(str(x).startswith("--emit-rung-env") for x in (recipe.get("emit") or {}).get("args") or []):
        errs.append("--emit-rung-env in [emit].args: state rung overrides as [emit.rung_env]")
    return errs


def from_recipe(recipe: dict) -> Overrides:
    """`[emit.rung_env]` as {(phase, rows): {env: value}}; shape errors raise RungEnvError."""
    table = (recipe.get("emit") or {}).get("rung_env") or {}
    if not isinstance(table, dict):
        raise RungEnvError("[emit.rung_env] must be a table of phases")
    out: Overrides = {}
    for phase, by_rows in table.items():
        if not isinstance(by_rows, dict):
            raise RungEnvError(f"[emit.rung_env.{phase}] must be a table of rung rows")
        for rows, knobs in by_rows.items():
            if not str(rows).isdigit() or int(rows) <= 0:
                raise RungEnvError(f"[emit.rung_env.{phase}.{rows}]: rows must be a positive integer")
            if not isinstance(knobs, dict) or not all(isinstance(v, str) for v in knobs.values()):
                raise RungEnvError(f"[emit.rung_env.{phase}.{rows}] must map knob env names to strings")
            key = (phase, int(rows))
            if key in out:
                raise RungEnvError(f"[emit.rung_env.{phase}.{rows}] given twice (rows {rows!r})")
            out[key] = dict(knobs)
    return out


def parse(spec: str | None) -> Overrides:
    """`PLOW_EMIT_RUNG_ENV` (or `--rung-env` entries joined by `;`) as {(phase, rows): {env: value}}."""
    out: Overrides = {}
    for entry in (e.strip() for e in (spec or "").split(";")):
        if not entry:
            continue
        m = re.fullmatch(r"([a-z]+):(\d+):(PLOW_[A-Z0-9_]+)=(.+)", entry)
        if not m:
            raise RungEnvError(f"rung override {entry!r}: expected phase:rows:PLOW_KNOB=value")
        knobs = out.setdefault((m[1], int(m[2])), {})
        if m[3] in knobs:
            raise RungEnvError(f"rung override {entry!r}: {m[3]} given twice for {m[1]}:{m[2]}")
        knobs[m[3]] = m[4].strip()
    return out


def merge(base: Overrides, extra: Overrides) -> Overrides:
    """`extra` (a CLI A/B) replaces `base`'s value for the same (phase, rows, knob)."""
    out = {k: dict(v) for k, v in base.items()}
    for key, knobs in extra.items():
        out.setdefault(key, {}).update(knobs)
    return out


def render(over: Overrides) -> str:
    order = {p: i for i, p in enumerate(PHASES)}
    return ";".join(f"{phase}:{rows}:{env}={value}"
                    for (phase, rows), knobs in sorted(over.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1]))
                    for env, value in sorted(knobs.items()))


def boolish(value: str) -> str | None:
    """clap BoolishValueParser's reading of `value` as devgen records it ("true"/"false")."""
    v = value.strip().lower()
    return "true" if v in TRUE else "false" if v in FALSE else None


def toml_snippet(over: Overrides) -> str:
    order = {p: i for i, p in enumerate(PHASES)}
    out = []
    for (phase, rows), knobs in sorted(over.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])):
        out.append(f"[emit.rung_env.{phase}.{rows}]")
        out += [f'{env} = "{value}"' for env, value in sorted(knobs.items())]
        out.append("")
    return "\n".join(out)
