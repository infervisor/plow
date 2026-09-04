#!/usr/bin/env python3
"""Report and optionally require measured tile provenance from build.json."""

import json
import pathlib
import sys


def fail(message: str) -> None:
    raise SystemExit(message)


if len(sys.argv) != 3 or sys.argv[2] not in {"0", "1"}:
    fail(f"usage: {sys.argv[0]} BUILD_JSON REQUIRE_TUNED(0|1)")

path = pathlib.Path(sys.argv[1])
required = sys.argv[2] == "1"
if not path.is_file():
    if required:
        fail(f"tuned profile required but build manifest is missing: {path}")
    print("0\t<missing>\tanalytical-fallback")
    raise SystemExit(0)

try:
    manifest = json.loads(path.read_text())
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    fail(f"cannot read tuning provenance from {path}: {error}")

tuning = manifest.get("tuning")
if not isinstance(tuning, dict):
    tuning = {}
measured = tuning.get("tile_measured", 0)
source = tuning.get("tile_source", "")
valid_count = isinstance(measured, int) and not isinstance(measured, bool) and measured > 0
valid_source = (
    isinstance(source, str)
    and bool(source.strip())
    and "analytical" not in source.casefold()
    and not any(c in source for c in "\t\r\n")
)
profile = "measured" if valid_count and valid_source else "analytical-fallback"
display_source = " ".join(source.split()) if isinstance(source, str) and source.strip() else "<missing>"

if required and profile != "measured":
    fail(
        f"tuned profile required but {path} reports "
        f"tuning.tile_measured={measured!r}, tuning.tile_source={display_source!r}; "
        "require a positive measured count and a non-analytical source"
    )

print(f"{measured if valid_count else 0}\t{display_source}\t{profile}")
