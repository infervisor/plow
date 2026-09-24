"""Print a recipe's [objects.env] as shell-sourceable KEY=VALUE lines.

Used when the segment objects have to be rebuilt outside `campaign.py build` -- e.g. to
re-specialise them against the final assets config when a role emit changes the packet hash.
Reading them from the recipe keeps that rebuild from drifting out of sync with the build.
"""

import shlex
import sys
import tomllib


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <recipe.toml>", file=sys.stderr)
        return 2
    with open(sys.argv[1], "rb") as fh:
        recipe = tomllib.load(fh)
    env = recipe.get("objects", {}).get("env", {})
    for key, value in env.items():
        print(f"{key}={shlex.quote(str(value))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
