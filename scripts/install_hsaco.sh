#!/usr/bin/env bash
# Install rebuilt interpreter objects into a shared hsaco tree, atomically.
#
#   scripts/install_hsaco.sh <src-dir> <dst-dir> <name.elf> [name.elf ...]
#
# `cp` onto a live path TRUNCATES it, so an agent mid-run whose HSA loader is
# reading that file gets a short read and a bogus INVALID_CODE_OBJECT. Copy to a
# sibling temp name and rename: rename is atomic within a filesystem, so a
# concurrent reader sees either the whole old file or the whole new one.
set -euo pipefail
SRC="${1:?src dir}"; shift
DST="${1:?dst dir}"; shift
[ "$#" -gt 0 ] || { echo "no objects named" >&2; exit 1; }
for f in "$@"; do
  [ -s "$SRC/$f" ] || { echo "missing or empty: $SRC/$f" >&2; exit 1; }
  cp "$SRC/$f" "$DST/.install.$f"
  mv -f "$DST/.install.$f" "$DST/$f"
  echo "   installed $DST/$f ($(stat -c%s "$DST/$f") B)"
done
