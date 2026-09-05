#!/usr/bin/env bash
# Print the SHA-256 of the executable .text section of each code object given.
set -euo pipefail
HIPCC=$(command -v hipcc)
TOOLROOT=$(cd "$(dirname "$HIPCC")/.." && pwd)
OBJCOPY="$TOOLROOT/lib/llvm/bin/llvm-objcopy"
for f in "$@"; do
    tmp=$(mktemp)
    "$OBJCOPY" --dump-section=.text="$tmp" "$f" /dev/null
    printf '%s  %s\n' "$(sha256sum "$tmp" | cut -d' ' -f1)" "$f"
    rm -f "$tmp"
done
