"""Every checked-in recipe's `[objects].env` must produce the `-D` set it claims.

`scripts/check_recipe.py` asserts `[objects].env` against a build's `build_defines.json`
post hoc — and CI only ever ran it without `--objects`, so the objects surface was never
checked against anything. This resolves each recipe's env through
`scripts/build_gfx942.sh` in `PLOW_DEFINES_ONLY=1` mode (seconds, no hipcc) and runs the
same `check_objects` the post-hoc check runs, for the base object set and for every
`[[objects.lowrung]]` tier.

Needs the flake's toolchain (the build script insists on it): run under `nix develop`.
Outside it the test is skipped, not passed.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
# `check_recipe.py` imports its sibling `plow_dist` bare, as the CLI does.
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _defines_for(env: dict, out: pathlib.Path) -> dict:
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("PLOW_") and k not in ("JOBS", "GM_BM", "GM_BN", "GM_BK", "GM_DBUF", "GM_AX")}
    # The nix shell's own PLOW_* toolchain pins must survive the scrub.
    for k in ("PLOW_HIPCC", "PLOW_BUNDLER", "PLOW_READELF", "PLOW_TOOLCHAIN_LABEL"):
        if k in os.environ:
            run_env[k] = os.environ[k]
    run_env.update({k: str(v) for k, v in env.items()})
    run_env["PLOW_DEFINES_ONLY"] = "1"
    r = subprocess.run(
        [str(ROOT / "scripts" / "build_gfx942.sh"), str(out)],
        env=run_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise AssertionError(f"build_gfx942.sh refused env {env}:\n{r.stdout}\n{r.stderr}")
    pd = _load("plow_dist")
    return pd.load_json(out / "build_defines.json")


@unittest.skipUnless(os.environ.get("IN_NIX_SHELL"), "build_gfx942.sh needs the nix toolchain")
class RecipeObjectsEnvTests(unittest.TestCase):
    def test_every_gfx942_recipe_env_produces_the_defines_it_claims(self):
        pd = _load("plow_dist")
        check_recipe = _load("check_recipe")
        recipes = sorted(ROOT.glob("recipes/*/*/*.toml"))
        self.assertTrue(recipes, "no recipes found")
        checked = 0
        for path in recipes:
            recipe = pd.load_toml(path)
            obj = recipe.get("objects") or {}
            if obj.get("script") != "scripts/build_gfx942.sh":
                continue
            base_env = dict(obj.get("env") or {})
            with tempfile.TemporaryDirectory() as td:
                with self.subTest(recipe=path.name, tier="base"):
                    defines = _defines_for(base_env, pathlib.Path(td) / "base")
                    self.assertTrue(defines, f"{path}: empty build_defines.json")
                    notes = check_recipe.check_objects(recipe, defines)
                    self.assertEqual(notes, [], f"{path}: {notes}")
                    checked += 1
                # A rung is rendered as its own invocation (render_recipe.py
                # `block_objects`), so its env must stand alone.
                for k, tier in enumerate(obj.get("lowrung") or []):
                    tier_env = dict(tier.get("env") or {})
                    with self.subTest(recipe=path.name, tier=k):
                        defines = _defines_for(tier_env, pathlib.Path(td) / f"tier{k}")
                        tier_recipe = dict(recipe)
                        tier_recipe["objects"] = {"env": tier_env}
                        notes = check_recipe.check_objects(tier_recipe, defines)
                        self.assertEqual(notes, [], f"{path} lowrung[{k}]: {notes}")
                        # A tier is exactly the decode rows at its own width.
                        self.assertTrue(
                            all(stem.startswith("interp_decode") or stem == "test_kernels" for stem in defines),
                            f"{path} lowrung[{k}] built non-decode rows: {sorted(defines)}",
                        )
        self.assertGreater(checked, 0, "no recipe uses scripts/build_gfx942.sh")


if __name__ == "__main__":
    unittest.main()
