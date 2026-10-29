#!/usr/bin/env python3
"""Doctor check: every route the packet could carry is on, or the recipe says why it is off.

  capability_check.py <plowrt> <assets> [--waive NAME,...]

Runs `plowrt inventory <assets> --format json` and prints one doctor line per item
(`ok`, `WARN`, `FAIL` + tab + text; the doctor routes them through pb_ok/pb_warn/pb_bad). An `off`
item fails unless it is waived: `--waive` / PB_CAPABILITY_WAIVERS (comma-separated names) or the
`[capabilities]` table of the recipe beside the assets (<assets>/../recipe.toml, as campaign.py
build copies it), `name = "reason"`. `n/a` items are informational. Exit 1 on any FAIL.

A recipe that requests a route at emit (PLOW_EMIT_ATTN_LIBRARY) fails on a packet that does not
declare it, whatever the waivers.

A campaign starts from the full plowrt feature set and measures removing routes; an `off` that
nobody waived is a route the campaign would rediscover later as a "new technique".
"""
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


def waivers(assets: Path, extra: str) -> dict[str, str]:
    out = {n: "PB_CAPABILITY_WAIVERS" for n in filter(None, extra.replace(",", " ").split())}
    recipe = assets.parent / "recipe.toml"
    if recipe.is_file():
        try:
            with recipe.open("rb") as f:
                caps = tomllib.load(f).get("capabilities", {})
        except (tomllib.TOMLDecodeError, OSError) as e:
            print(f"WARN\t{recipe}: unreadable ({e}); recipe waivers ignored")
            caps = {}
        out.update({k: f"recipe: {v}" for k, v in caps.items() if isinstance(v, str)})
    return out


def requested_routes(assets: Path) -> dict[str, str]:
    """Inventory items the recipe beside the assets requests at emit: {item: knob}."""
    recipe = assets.parent / "recipe.toml"
    found: dict[str, str] = {}

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if k == "PLOW_EMIT_ATTN_LIBRARY" and str(x).lower() not in ("", "0", "false", "off"):
                    found["library_prefill_attention"] = k
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    if recipe.is_file():
        try:
            with recipe.open("rb") as f:
                walk(tomllib.load(f))
        except (tomllib.TOMLDecodeError, OSError):
            pass
    return found


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    plowrt, assets = Path(argv[1]), Path(argv[2])
    extra = argv[argv.index("--waive") + 1] if "--waive" in argv else os.environ.get("PB_CAPABILITY_WAIVERS", "")
    r = subprocess.run([str(plowrt), "inventory", str(assets), "--format", "json"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        err = (r.stderr.strip().splitlines() or ["no output"])[-1][:160]
        print(f"WARN\tplowrt inventory failed ({err}) — plowrt predates the subcommand, or the packet is not a devblob")
        return 0
    try:
        items = json.loads(r.stdout)
    except ValueError as e:
        print(f"WARN\tplowrt inventory: malformed JSON ({e})")
        return 0
    w = waivers(assets, extra)
    want = requested_routes(assets)
    fails = 0
    for it in items:
        name, state, detail = it["name"], it["state"], it["detail"]
        if name in want and state != "on":
            fails += 1
            print(f"FAIL\t{name}: the recipe requests it ({want[name]}) but the packet does not declare it: {detail}")
        elif state == "on":
            print(f"ok\t{name}: {detail}")
        elif state == "not_applicable":
            print(f"ok\t{name}: n/a — {detail}")
        elif name in w:
            print(f"WARN\t{name} OFF, waived ({w[name]}): {detail}")
        else:
            fails += 1
            print(f"FAIL\t{name} OFF and not waived: {detail}")
    if fails:
        print(f"FAIL\t{fails} route(s) off without a waiver — turn them on in the recipe, or add [capabilities] "
              f"name = \"reason\" (or PB_CAPABILITY_WAIVERS=name,...) with the measured reason")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
